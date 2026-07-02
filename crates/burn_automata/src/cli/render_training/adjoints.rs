#![allow(clippy::too_many_arguments)]

mod temporal;

use super::*;

pub(crate) use temporal::*;

pub(crate) fn terminal_render_state_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    opacity_gain: f32,
    scale_gain: f32,
    scale_budget_weight: f32,
    liveness_gain: f32,
    liveness_front_radius: f32,
    liveness_step_fraction: f32,
    max_opacity_update: f32,
    render_cfg: RenderLossConfig,
    rows: usize,
) -> Vec<f32> {
    let mut state_adjoint = vec![0.0; trace.states.len()];
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row * trace.state_dims + config.state_dims > trace.states.len() {
            continue;
        }
        let state_base = particle_row * trace.state_dims;
        if opacity_gain > 0.0
            && let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims)
        {
            let final_logit =
                trace.states[state_base + opacity_channel] + render_cfg.opacity_logit_bias;
            state_adjoint[state_base + opacity_channel] += opacity_gain
                * gradient.opacity_gradients[gradient_row]
                * sigmoid_unit_derivative(final_logit);
        }
        if scale_gain > 0.0
            && render_cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale
            && config.state_dims >= 5
        {
            let scale_channel = config.state_dims - 5;
            state_adjoint[state_base + scale_channel] +=
                scale_gain * gradient.scale_gradients[gradient_row];
        }
        if config.state_dims >= 3 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                let state_value = trace.states[state_base + tail + channel];
                if state_value > -1.0 && state_value < 1.0 {
                    state_adjoint[state_base + tail + channel] +=
                        gradient.color_gradients[gradient_row][channel];
                }
            }
        }
    }
    add_gaussian_scale_budget_state_adjoint(
        config,
        trace,
        render_cfg,
        scale_budget_weight,
        &mut state_adjoint,
    );
    add_liveness_front_state_adjoint(
        config,
        &trace.positions,
        &trace.states,
        liveness_gain,
        liveness_front_radius,
        liveness_step_fraction,
        max_opacity_update,
        &mut state_adjoint,
    );
    add_temporal_activation_schedule_state_adjoint(
        config,
        &trace.positions,
        &trace.states,
        liveness_gain,
        liveness_front_radius,
        liveness_step_fraction,
        max_opacity_update,
        &mut state_adjoint,
    );
    state_adjoint
}

pub(crate) fn add_liveness_front_state_adjoint(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    front_radius: f32,
    step_fraction: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }
    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return;
    }
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for (row, front_weight) in front_weights.iter().copied().enumerate() {
        if front_weight <= 0.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let liveness_index = state_base + GROWTH_3D_LIVENESS_CHANNEL;
        let current_liveness = states[liveness_index];
        let scheduled_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT
            + schedule
                * front_weight
                * (UV_TORUS_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
        let target_liveness = if current_liveness > -1.0 {
            current_liveness.max(scheduled_target)
        } else {
            scheduled_target
        };
        let adjoint =
            (liveness_gain * (current_liveness - target_liveness)).clamp(-max_adjoint, max_adjoint);
        state_adjoint[liveness_index] += adjoint;
    }
}

pub(crate) fn liveness_front_temporal_target_updates(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    front_radius: f32,
    step_fraction: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return updates;
    }
    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let schedule = step_fraction.clamp(0.0, 1.0);
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for (row, front_weight) in front_weights.iter().copied().enumerate() {
        if front_weight <= 0.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let current_liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let scheduled_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT
            + schedule
                * front_weight
                * (UV_TORUS_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
        let target_liveness = if current_liveness > -1.0 {
            current_liveness.max(scheduled_target)
        } else {
            scheduled_target
        };
        updates[row] += liveness_gain * (target_liveness - current_liveness);
    }

    let allowed_fraction = temporal_activation_allowed_fraction(schedule);
    let allowed_active = ((positions.len() as f32) * allowed_fraction).ceil() as usize;
    let mut active_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some((row, liveness))
        })
        .collect::<Vec<_>>();
    if active_rows.len() > allowed_active {
        active_rows.sort_by(|(_, lhs), (_, rhs)| {
            lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
        });
        let suppress_count = active_rows.len().saturating_sub(allowed_active);
        for (row, current_liveness) in active_rows.into_iter().take(suppress_count) {
            updates[row] += liveness_gain * (-1.0 - current_liveness);
        }
    }

    for update in &mut updates {
        *update = update.clamp(-max_update, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_escape_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_escape_gain: f32,
    opacity_gain: f32,
    liveness_gain: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || surface_escape_gain <= 0.0
        || !surface_escape_gain.is_finite()
        || ((opacity_gain <= 0.0 || !opacity_gain.is_finite())
            && (liveness_gain <= 0.0 || !liveness_gain.is_finite()))
    {
        return;
    }
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let threshold = GROWTH_3D_SURFACE_MAX_DISTANCE.max(1.0e-6);
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness_index = state_base + GROWTH_3D_LIVENESS_CHANNEL;
        let liveness = states[liveness_index];
        if liveness <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance <= threshold {
            continue;
        }
        let escape = (projection.distance / threshold - 1.0).max(0.0);
        let strength = (surface_escape_gain * escape).min(8.0);
        if liveness_gain > 0.0 && liveness_gain.is_finite() {
            let adjoint = (liveness_gain * strength).clamp(0.0, max_adjoint);
            state_adjoint[liveness_index] = state_adjoint[liveness_index].max(adjoint);
        }
        if opacity_gain > 0.0
            && opacity_gain.is_finite()
            && let Some(opacity_channel) = opacity_channel
        {
            let opacity_index = state_base + opacity_channel;
            if opacity_index != liveness_index {
                let adjoint = (opacity_gain * strength).clamp(0.0, max_adjoint);
                state_adjoint[opacity_index] = state_adjoint[opacity_index].max(adjoint);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_material_opacity_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
        || seed_scale <= 0.0
        || !seed_scale.is_finite()
        || !target_opacity_logit.is_finite()
    {
        return;
    }
    let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        let opacity_index = state_base + opacity_channel;
        let current_opacity = states[opacity_index];
        let surface_weight = frontier_material_assignment_weight(
            projection.distance,
            strict_threshold,
            soft_threshold,
            frontier_threshold,
        );
        if surface_weight <= 0.0 {
            continue;
        }
        let adjoint =
            (material_opacity_gain * surface_weight * (current_opacity - target_opacity_logit))
                .clamp(-max_adjoint, max_adjoint);
        state_adjoint[opacity_index] += adjoint;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_target_coverage_opacity_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    material_target_coverage_opacity_updates_weighted(
        config,
        target,
        positions,
        states,
        None,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_update,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_target_coverage_opacity_updates_weighted(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    row_weights: Option<&[f32]>,
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || row_weights.is_some_and(|weights| weights.len() < positions.len())
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
        || !target_opacity_logit.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let candidate_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            if let Some(weights) = row_weights {
                return (weights[row] > 1.0e-3 && weights[row].is_finite()).then_some(row);
            }
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some(row)
        })
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return updates;
    }

    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    let frontier_threshold2 = frontier_threshold * frontier_threshold;
    let sample_count = coverage_samples.max(candidate_rows.len().max(512));
    let mut assigned_weights = vec![0.0_f32; positions.len()];
    let mut assigned_counts = vec![0usize; positions.len()];
    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = candidate_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in &candidate_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() || best_distance2 > frontier_threshold2 {
            continue;
        }
        let distance = best_distance2.sqrt();
        assigned_weights[best_row] += frontier_material_assignment_weight(
            distance,
            threshold,
            soft_threshold,
            frontier_threshold,
        );
        assigned_counts[best_row] += 1;
    }

    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    for &row in &candidate_rows {
        let count = assigned_counts[row];
        if count == 0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let material_index = state_base + material_channel;
        let assignment_weight = (assigned_weights[row] / count as f32).clamp(0.0, 1.0);
        if assignment_weight <= 0.0 {
            continue;
        }
        let row_weight = row_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 {
            continue;
        }
        let material_opacity = states[material_index];
        updates[row] = (material_opacity_gain
            * row_weight
            * assignment_weight
            * (target_opacity_logit - material_opacity))
            .clamp(-max_update, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_target_coverage_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if state_adjoint.len() < states.len() {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let updates = material_target_coverage_opacity_updates(
        config,
        target,
        positions,
        states,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_adjoint,
    );
    for (row, update) in updates.iter().copied().enumerate() {
        if update == 0.0 {
            continue;
        }
        state_adjoint[row * config.state_dims + material_channel] -= update;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_surface_strata_opacity_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    material_surface_strata_opacity_updates_weighted(
        config,
        target,
        positions,
        states,
        None,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_update,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_surface_strata_opacity_updates_weighted(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    row_weights: Option<&[f32]>,
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || row_weights.is_some_and(|weights| weights.len() < positions.len())
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
        || !target_opacity_logit.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let candidate_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            if let Some(weights) = row_weights {
                return (weights[row] > 1.0e-3 && weights[row].is_finite()).then_some(row);
            }
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some(row)
        })
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(candidate_rows.len().max(512));
    let bin_count = sample_count
        .min((candidate_rows.len().saturating_mul(2)).clamp(32, 128))
        .max(1);
    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let threshold2 = threshold * threshold;
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    let frontier_threshold2 = frontier_threshold * frontier_threshold;
    let mut bin_samples = vec![0usize; bin_count];
    let mut bin_material_covered = vec![0usize; bin_count];
    let mut bin_candidate = vec![None::<(usize, f32)>; bin_count];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        bin_samples[bin] += 1;

        let mut best_active = None::<(usize, f32)>;
        let mut best_material_distance2 = f32::MAX;
        for &row in &candidate_rows {
            if row >= positions.len() {
                continue;
            }
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if !distance2.is_finite() {
                continue;
            }
            if best_active.is_none_or(|(_, best_distance2)| distance2 < best_distance2) {
                best_active = Some((row, distance2));
            }
            let material_opacity = states[row * config.state_dims + material_channel];
            if material_opacity > -1.0 {
                best_material_distance2 = best_material_distance2.min(distance2);
            }
        }

        if best_material_distance2 <= threshold2 {
            bin_material_covered[bin] += 1;
        }
        if let Some((row, distance2)) = best_active
            && distance2 <= frontier_threshold2
            && bin_candidate[bin].is_none_or(|(_, current_distance2)| distance2 < current_distance2)
        {
            bin_candidate[bin] = Some((row, distance2));
        }
    }

    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let target_bin_fraction = GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        .max(GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE)
        .clamp(0.0, 1.0);
    for bin in 0..bin_count {
        let samples = bin_samples[bin];
        if samples == 0 {
            continue;
        }
        let covered_fraction = bin_material_covered[bin] as f32 / samples as f32;
        let deficit = (target_bin_fraction - covered_fraction).max(0.0);
        if deficit <= 0.0 {
            continue;
        }
        let Some((row, distance2)) = bin_candidate[bin] else {
            continue;
        };
        let state_base = row * config.state_dims;
        let current_opacity = states[state_base + material_channel];
        let distance = distance2.sqrt();
        let surface_weight = frontier_material_assignment_weight(
            distance,
            threshold,
            soft_threshold,
            frontier_threshold,
        );
        let opacity_gap = (target_opacity_logit - current_opacity).max(0.0);
        let row_weight = row_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 || surface_weight <= 0.0 || opacity_gap <= 0.0 {
            continue;
        }
        updates[row] += material_opacity_gain * row_weight * deficit * surface_weight * opacity_gap;
        updates[row] = updates[row].clamp(-max_update, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_surface_strata_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if state_adjoint.len() < states.len() {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let updates = material_surface_strata_opacity_updates(
        config,
        target,
        positions,
        states,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_adjoint,
    );
    for (row, update) in updates.iter().copied().enumerate() {
        if update == 0.0 {
            continue;
        }
        state_adjoint[row * config.state_dims + material_channel] -= update;
    }
}

pub(crate) fn add_material_liveness_state_adjoint(
    config: &NpaConfig,
    states: &[f32],
    material_opacity_gain: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.is_empty()
        || state_adjoint.len() < states.len()
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
    {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    for (row, state) in states.chunks_exact(config.state_dims).enumerate() {
        let liveness = state[GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = state[material_channel];
        if liveness > -1.0 || material_opacity <= GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT {
            continue;
        }
        let adjoint = (material_opacity_gain
            * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT))
            .clamp(0.0, max_adjoint);
        state_adjoint[row * config.state_dims + material_channel] += adjoint;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_liveness_target_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    surface_threshold: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || surface_threshold <= 0.0
        || !surface_threshold.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        if liveness > -1.0 || material_opacity <= material_visible_threshold {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > surface_threshold {
            continue;
        }
        let surface_weight = (1.0 - projection.distance / surface_threshold).clamp(0.0, 1.0);
        let material_weight =
            ((material_opacity - material_visible_threshold) / 4.0).clamp(0.25, 1.0);
        let target_liveness = 0.0_f32;
        updates[row] =
            (liveness_gain * surface_weight * material_weight * (target_liveness - liveness))
                .clamp(0.0, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_liveness_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    surface_threshold: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if state_adjoint.len() < states.len() {
        return;
    }
    let updates = material_visible_liveness_target_updates(
        config,
        target,
        positions,
        states,
        liveness_gain,
        surface_threshold,
        max_adjoint,
    );
    for (row, update) in updates.iter().copied().enumerate() {
        if update == 0.0 {
            continue;
        }
        state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] -= update;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_tail_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_tail_gain: f32,
    surface_threshold: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || material_tail_gain <= 0.0
        || !material_tail_gain.is_finite()
        || surface_threshold <= 0.0
        || !surface_threshold.is_finite()
    {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let threshold = surface_threshold.max(1.0e-6);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let material_index = state_base + material_channel;
        let material_opacity = states[material_index];
        if material_opacity <= GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance <= threshold {
            continue;
        }
        let escape = (projection.distance / threshold - 1.0).max(0.0);
        let adjoint = (material_tail_gain
            * escape.min(8.0)
            * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT))
            .clamp(0.0, max_adjoint);
        state_adjoint[material_index] += adjoint;
    }
}
