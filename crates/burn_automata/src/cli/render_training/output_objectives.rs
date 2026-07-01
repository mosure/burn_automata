#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_temporal_materialization_output_objective_with_candidate_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    material_gain: f32,
    front_radius: f32,
    candidate_weights: &[f32],
    max_material_update: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let material_output = config.spatial_dims + material_channel;
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if material_output >= output_dims
        || liveness_output >= output_dims
        || positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || candidate_weights.len() < positions.len()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || material_gain <= 0.0
        || !material_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }

    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return;
    }
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let target_visible =
        ((positions.len() as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
    let predicted_visible = (0..positions.len())
        .filter(|row| {
            let state_base = row * config.state_dims;
            let output_base = row * output_dims;
            let predicted_liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL]
                + raw_updates[output_base + liveness_output];
            let predicted_material =
                states[state_base + material_channel] + raw_updates[output_base + material_output];
            predicted_liveness > -1.0 && predicted_material > material_visible_threshold
        })
        .count();
    let deficit = target_visible.saturating_sub(predicted_visible);
    if deficit == 0 {
        return;
    }

    let front_weights = local_front_weights_with_min_candidates(
        config,
        positions,
        states,
        front_radius,
        temporal_front_candidate_count(positions.len(), deficit),
    );
    let target_material = temporal_materialization_target_logit(schedule);
    let max_material_update = if max_material_update.is_finite() && max_material_update > 0.0 {
        max_material_update
    } else {
        f32::INFINITY
    };
    let mut candidates = (0..positions.len())
        .filter_map(|row| {
            let state_base = row * config.state_dims;
            let output_base = row * output_dims;
            let material = states[state_base + material_channel];
            let predicted_material = material + raw_updates[output_base + material_output];
            if predicted_material >= target_material {
                return None;
            }
            let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
            let candidate_weight = candidate_weights[row].clamp(0.0, 1.0);
            if front_weight <= 0.0 || candidate_weight <= 0.0 {
                return None;
            }
            let score = (front_weight * candidate_weight).clamp(0.0, 1.0);
            (score > 0.0 && score.is_finite()).then_some((row, score, material))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(_, lhs_score, lhs_material), (_, rhs_score, rhs_material)| {
            rhs_score
                .partial_cmp(lhs_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    lhs_material
                        .partial_cmp(rhs_material)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        },
    );
    for (row, score, material) in candidates.into_iter().take(deficit) {
        let output_index = row * output_dims + material_output;
        let raw = raw_updates[output_index];
        let target_update = (target_material - material)
            .max(0.0)
            .clamp(0.0, max_material_update);
        if target_update <= 0.0 {
            continue;
        }
        output_gradients[output_index] += material_gain * score * (raw - target_update);
    }
}

pub(crate) fn add_material_coverage_materialization_output_objective(
    config: &NpaConfig,
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    material_gain: f32,
    candidate_weights: &[f32],
    max_material_update: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_output = config.spatial_dims + material_channel;
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || liveness_output >= output_dims
        || material_output >= output_dims
        || config.state_dims == 0
        || output_dims == 0
        || material_gain <= 0.0
        || !material_gain.is_finite()
        || states.len() < candidate_weights.len().saturating_mul(config.state_dims)
        || raw_updates.len() < candidate_weights.len().saturating_mul(output_dims)
        || output_gradients.len() < candidate_weights.len().saturating_mul(output_dims)
    {
        return;
    }

    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return;
    }
    let max_material_update = if max_material_update.is_finite() && max_material_update > 0.0 {
        max_material_update
    } else {
        f32::INFINITY
    };
    let target_material = temporal_materialization_target_logit(schedule);

    for row in 0..candidate_weights.len() {
        let candidate_weight = candidate_weights[row].clamp(0.0, 1.0);
        if candidate_weight <= 1.0e-3 || !candidate_weight.is_finite() {
            continue;
        }
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let material = states[state_base + material_channel];
        let predicted_material = material + raw_updates[output_base + material_output];
        if predicted_material >= target_material {
            continue;
        }
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let predicted_liveness = liveness + raw_updates[output_base + liveness_output];
        let activity_weight = if liveness > -1.0 || predicted_liveness > -1.0 {
            1.0
        } else {
            0.75
        };
        let target_update = (target_material - material)
            .max(0.0)
            .clamp(0.0, max_material_update);
        if target_update <= 0.0 || !target_update.is_finite() {
            continue;
        }
        output_gradients[output_base + material_output] += material_gain
            * candidate_weight
            * activity_weight
            * (raw_updates[output_base + material_output] - target_update);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_active_surface_materialization_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    material_gain: f32,
    seed_scale: f32,
    max_material_update: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let material_output = config.spatial_dims + material_channel;
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if material_output >= output_dims
        || liveness_output >= output_dims
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
        || material_gain <= 0.0
        || !material_gain.is_finite()
    {
        return;
    }

    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let max_material_update = if max_material_update.is_finite() && max_material_update > 0.0 {
        max_material_update
    } else {
        f32::INFINITY
    };

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let predicted_liveness = liveness + raw_updates[output_base + liveness_output];
        let front_weight = front_weights
            .as_ref()
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(0.0);
        let activation_weight = activation_candidate_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let activity_weight = if liveness > -1.0 {
            1.0
        } else if predicted_liveness > -1.0 {
            (0.5 + 0.5 * activation_weight).clamp(0.0, 1.0)
        } else {
            front_weight * activation_weight
        };
        if activity_weight <= 1.0e-3 || !activity_weight.is_finite() {
            continue;
        }

        let projection = target.project(position3(*position));
        let surface_weight =
            soft_material_assignment_weight(projection.distance, strict_threshold, soft_threshold);
        if surface_weight <= 0.0 {
            continue;
        }

        let material = states[state_base + material_channel];
        let predicted_material = material + raw_updates[output_base + material_output];
        if predicted_material >= GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET {
            continue;
        }
        let target_update = (surface_weight
            * activity_weight
            * (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material))
            .max(0.0)
            .clamp(0.0, max_material_update);
        if target_update <= 0.0 {
            continue;
        }
        let raw = raw_updates[output_base + material_output];
        output_gradients[output_base + material_output] += material_gain * (raw - target_update);
    }
}

#[allow(dead_code)]
pub(crate) fn add_temporal_liveness_output_objective(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    liveness_gain: f32,
    front_radius: f32,
    output_gradients: &mut [f32],
) {
    add_temporal_liveness_output_objective_with_candidate_weights(
        config,
        positions,
        states,
        raw_updates,
        step_fraction,
        liveness_gain,
        front_radius,
        None,
        output_gradients,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_temporal_liveness_output_objective_with_candidate_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    liveness_gain: f32,
    front_radius: f32,
    candidate_weights: Option<&[f32]>,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || candidate_weights
            .map(|weights| weights.len() < positions.len())
            .unwrap_or(false)
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims {
        return;
    }
    let rows = positions.len();
    let schedule = step_fraction.clamp(0.0, 1.0);
    let predicted_liveness = (0..rows)
        .map(|row| {
            states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL]
                + raw_updates[row * output_dims + liveness_output]
        })
        .collect::<Vec<_>>();
    let allowed_active =
        ((rows as f32) * temporal_activation_allowed_fraction(schedule)).ceil() as usize;
    let mut predicted_active = predicted_liveness
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(row, liveness)| {
            let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some((row, liveness, state_liveness))
        })
        .collect::<Vec<_>>();
    if predicted_active.len() > allowed_active {
        predicted_active.sort_by(
            |(_, lhs_predicted, lhs_state), (_, rhs_predicted, rhs_state)| {
                (lhs_state > &-1.0).cmp(&(rhs_state > &-1.0)).then_with(|| {
                    lhs_predicted
                        .partial_cmp(rhs_predicted)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            },
        );
        let suppress_count = predicted_active.len().saturating_sub(allowed_active);
        for (row, _predicted, _state_liveness) in predicted_active.into_iter().take(suppress_count)
        {
            let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            let raw = raw_updates[row * output_dims + liveness_output];
            let target_update = -1.0 - state_liveness;
            output_gradients[row * output_dims + liveness_output] +=
                liveness_gain * (raw - target_update);
        }
    }

    let target_active =
        ((rows as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
    let active_count = predicted_liveness
        .iter()
        .filter(|liveness| **liveness > -1.0)
        .count();
    let deficit = target_active.saturating_sub(active_count);
    let temporal_front_candidates = temporal_front_candidate_count(rows, deficit);
    let front_weights = local_front_weights_with_min_candidates(
        config,
        positions,
        states,
        front_radius,
        temporal_front_candidates,
    );
    for row in 0..rows {
        let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if state_liveness > -1.0 {
            continue;
        }
        let output_base = row * output_dims;
        let raw = raw_updates[output_base + liveness_output];
        if raw <= 0.0 {
            continue;
        }
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        let candidate_weight = candidate_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if front_weight > 0.0 && candidate_weight > 0.0 {
            continue;
        }
        output_gradients[output_base + liveness_output] +=
            liveness_gain * TEMPORAL_NONLOCAL_LIVENESS_SUPPRESSION_GAIN_FRACTION * raw;
    }
    if active_count >= target_active || schedule <= 0.0 {
        return;
    }
    let mut candidates = (0..rows)
        .filter_map(|row| {
            let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            let predicted = predicted_liveness[row];
            let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
            let candidate_weight = candidate_weights
                .and_then(|weights| weights.get(row))
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            (state_liveness <= -1.0
                && predicted <= -1.0
                && front_weight > 0.0
                && candidate_weight > 0.0)
                .then_some((row, front_weight * candidate_weight, state_liveness))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(_, lhs_front, lhs_liveness), (_, rhs_front, rhs_liveness)| {
            rhs_front
                .partial_cmp(lhs_front)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    rhs_liveness
                        .partial_cmp(lhs_liveness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        },
    );
    let target_liveness = temporal_activation_candidate_liveness_target(schedule);
    for (row, front_weight, state_liveness) in candidates.into_iter().take(deficit) {
        let raw = raw_updates[row * output_dims + liveness_output];
        let target_update = target_liveness - state_liveness;
        output_gradients[row * output_dims + liveness_output] +=
            liveness_gain * front_weight * (raw - target_update);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_escape_liveness_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    surface_escape_gain: f32,
    liveness_gain: f32,
    max_liveness_update: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || surface_escape_gain <= 0.0
        || !surface_escape_gain.is_finite()
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || weight <= 0.0
        || !weight.is_finite()
    {
        return;
    }
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims {
        return;
    }
    let max_liveness_update = if max_liveness_update.is_finite() && max_liveness_update > 0.0 {
        max_liveness_update
    } else {
        f32::INFINITY
    };
    let threshold = GROWTH_3D_SURFACE_MAX_DISTANCE.max(1.0e-6);
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance <= threshold {
            continue;
        }
        let escape_ratio = (projection.distance / threshold - 1.0).max(0.0);
        let strength = (surface_escape_gain * escape_ratio).min(8.0);
        let target_update =
            (liveness_gain * strength * (-1.0 - liveness)).clamp(-max_liveness_update, 0.0);
        if target_update >= 0.0 {
            continue;
        }
        let output_index = row * output_dims + liveness_output;
        let raw = raw_updates[output_index];
        output_gradients[output_index] += weight * (raw - target_update);
    }
}

pub(crate) fn add_candidate_liveness_output_objective(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    candidate_weights: &[f32],
    liveness_gain: f32,
    max_liveness_update: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || candidate_weights.len() < positions.len()
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
    {
        return;
    }
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims {
        return;
    }
    let max_liveness_update = if max_liveness_update.is_finite() && max_liveness_update > 0.0 {
        max_liveness_update
    } else {
        f32::INFINITY
    };
    for row in 0..positions.len() {
        let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if state_liveness > -1.0 {
            continue;
        }
        let candidate_weight = candidate_weights[row].clamp(0.0, 1.0);
        if candidate_weight <= 0.0 || !candidate_weight.is_finite() {
            continue;
        }
        let output_base = row * output_dims;
        let predicted_liveness = state_liveness + raw_updates[output_base + liveness_output];
        if predicted_liveness > -1.0 {
            continue;
        }
        let target_update =
            (candidate_weight * (0.0 - state_liveness)).clamp(0.0, max_liveness_update);
        if target_update <= 0.0 {
            continue;
        }
        let raw = raw_updates[output_base + liveness_output];
        output_gradients[output_base + liveness_output] +=
            liveness_gain * candidate_weight * (raw - target_update);
    }
}

pub(crate) fn add_mesh_motion_liveness_output_objective(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    motion_candidate_weights: &[f32],
    liveness_gain: f32,
    max_liveness_update: f32,
    output_gradients: &mut [f32],
) {
    add_candidate_liveness_output_objective(
        config,
        positions,
        states,
        raw_updates,
        motion_candidate_weights,
        liveness_gain,
        max_liveness_update,
        output_gradients,
    );
}

pub(crate) fn max_candidate_weights(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let len = lhs.len().max(rhs.len());
    let mut weights = vec![0.0_f32; len];
    for (row, weight) in weights.iter_mut().enumerate() {
        let lhs_weight = lhs.get(row).copied().unwrap_or(0.0);
        let rhs_weight = rhs.get(row).copied().unwrap_or(0.0);
        *weight = lhs_weight.max(rhs_weight).clamp(0.0, 1.0);
    }
    weights
}

pub(crate) fn extent_front_liveness_candidate_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> Vec<f32> {
    let mut weights = vec![0.0_f32; positions.len()];
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return weights;
    }

    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..config.spatial_dims {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_count == 0 {
        return weights;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let (target_min, target_max) = target.bounds();
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        if liveness > -1.0 || front_weight <= 0.0 || !front_weight.is_finite() {
            continue;
        }
        let mut extent_weight = 0.0_f32;
        for axis in 0..config.spatial_dims {
            let target_extent = (target_max[axis] - target_min[axis]).abs().max(1.0e-6);
            let lower_room =
                ((active_min[axis] - target_min[axis]) / target_extent).clamp(0.0, 1.0);
            if position[axis] < active_min[axis] && lower_room > 0.0 {
                extent_weight = extent_weight.max(lower_room.sqrt());
            }
            let upper_room =
                ((target_max[axis] - active_max[axis]) / target_extent).clamp(0.0, 1.0);
            if position[axis] > active_max[axis] && upper_room > 0.0 {
                extent_weight = extent_weight.max(upper_room.sqrt());
            }
        }
        weights[row] = (front_weight * extent_weight).clamp(0.0, 1.0);
    }
    weights
}

pub(crate) fn render_proxy_extent_front_motion_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
    extent_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0_f32; 3]; positions.len()];
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || extent_gain <= 0.0
        || !extent_gain.is_finite()
    {
        return updates;
    }

    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..config.spatial_dims {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_count == 0 {
        return updates;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let (target_min, target_max) = target.bounds();
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        if liveness > -1.0 || front_weight <= 0.0 || !front_weight.is_finite() {
            continue;
        }
        for axis in 0..config.spatial_dims {
            let target_extent = (target_max[axis] - target_min[axis]).abs().max(1.0e-6);
            let lower_room =
                ((active_min[axis] - target_min[axis]) / target_extent).clamp(0.0, 1.0);
            let upper_room =
                ((target_max[axis] - active_max[axis]) / target_extent).clamp(0.0, 1.0);
            if position[axis] < active_min[axis] && lower_room > 0.0 {
                updates[row][axis] += extent_gain
                    * front_weight
                    * lower_room.sqrt()
                    * (target_min[axis] - position[axis]);
            }
            if position[axis] > active_max[axis] && upper_room > 0.0 {
                updates[row][axis] += extent_gain
                    * front_weight
                    * upper_room.sqrt()
                    * (target_max[axis] - position[axis]);
            }
        }
        clamp_vector3(&mut updates[row], max_update_norm);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_proxy_temporal_extent_motion_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    step_fraction: f32,
    front_radius: f32,
    extent_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0_f32; 3]; positions.len()];
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || extent_gain <= 0.0
        || !extent_gain.is_finite()
    {
        return updates;
    }

    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return updates;
    }
    let (target_min, target_max) = target.bounds();
    let mut target_center = [0.0_f32; 3];
    let mut target_half_extent = [0.0_f32; 3];
    for axis in 0..config.spatial_dims {
        target_center[axis] = 0.5 * (target_min[axis] + target_max[axis]);
        target_half_extent[axis] = 0.5 * (target_max[axis] - target_min[axis]).abs();
    }

    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..config.spatial_dims {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_count == 0 {
        return updates;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let linear_schedule = temporal_activation_target_fraction(schedule).powf(1.0 / 3.0);
    for (row, position) in positions.iter().enumerate() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        let row_weight = if liveness > -1.0 {
            1.0
        } else {
            front_weights.get(row).copied().unwrap_or(0.0)
        };
        if row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }

        for axis in 0..config.spatial_dims {
            let half_extent = target_half_extent[axis];
            if half_extent <= 1.0e-6 || !half_extent.is_finite() {
                continue;
            }
            let active_half_extent = 0.5 * (active_max[axis] - active_min[axis]).abs().max(1.0e-6);
            let desired_half_extent = (half_extent * linear_schedule).clamp(0.0, half_extent);
            if active_half_extent >= desired_half_extent {
                continue;
            }
            let delta = position[axis] - target_center[axis];
            let side = if delta >= 0.0 { 1.0 } else { -1.0 };
            let boundary_floor = if liveness > -1.0 {
                if delta.abs() > 1.0e-6 { 0.25 } else { 0.0 }
            } else {
                0.5
            };
            let boundary_weight = (delta.abs() / active_half_extent.max(front_radius * 0.5))
                .clamp(0.0, 1.0)
                .powi(2)
                .max(boundary_floor);
            if boundary_weight <= 0.0 {
                continue;
            }
            let extent_deficit =
                ((desired_half_extent - active_half_extent) / half_extent).clamp(0.0, 1.0);
            let target_coord = target_center[axis] + side * desired_half_extent;
            updates[row][axis] += extent_gain
                * schedule
                * row_weight
                * boundary_weight
                * extent_deficit.sqrt()
                * (target_coord - position[axis]);
        }
        clamp_vector3(&mut updates[row], max_update_norm);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_temporal_extent_motion_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    front_radius: f32,
    extent_gain: f32,
    max_update_norm: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.spatial_dims == 0
        || config.spatial_dims > 3
        || output_dims < config.spatial_dims
        || positions.is_empty()
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || weight <= 0.0
        || !weight.is_finite()
    {
        return;
    }
    let motion_updates = render_proxy_temporal_extent_motion_updates(
        config,
        target,
        positions,
        states,
        step_fraction,
        front_radius,
        extent_gain,
        max_update_norm,
    );
    for row in 0..positions.len() {
        let output_base = row * output_dims;
        if motion_updates[row]
            .iter()
            .take(config.spatial_dims)
            .all(|value| *value == 0.0)
        {
            continue;
        }
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - motion_updates[row][axis]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_extent_front_motion_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    front_radius: f32,
    extent_gain: f32,
    max_update_norm: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.spatial_dims == 0
        || config.spatial_dims > 3
        || output_dims < config.spatial_dims
        || positions.is_empty()
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || weight <= 0.0
        || !weight.is_finite()
    {
        return;
    }
    let motion_updates = render_proxy_extent_front_motion_updates(
        config,
        target,
        positions,
        states,
        front_radius,
        extent_gain,
        max_update_norm,
    );
    for row in 0..positions.len() {
        let output_base = row * output_dims;
        if motion_updates[row]
            .iter()
            .take(config.spatial_dims)
            .all(|value| *value == 0.0)
        {
            continue;
        }
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - motion_updates[row][axis]);
        }
    }
}

pub(crate) fn mesh_motion_candidate_weights(
    config: &NpaConfig,
    output_dims: usize,
    output_gradients: &[f32],
) -> Vec<f32> {
    if config.spatial_dims == 0 || output_dims == 0 || output_gradients.len() % output_dims != 0 {
        return Vec::new();
    }
    let rows = output_gradients.len() / output_dims;
    let mut motion_norms = vec![0.0; rows];
    let mut max_norm = 0.0_f32;
    for row in 0..rows {
        let base = row * output_dims;
        let motion_norm = output_gradients[base..base + config.spatial_dims.min(output_dims)]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if motion_norm.is_finite() && motion_norm > 1.0e-12 {
            motion_norms[row] = motion_norm;
            max_norm = max_norm.max(motion_norm);
        }
    }
    if max_norm <= 1.0e-12 || !max_norm.is_finite() {
        return motion_norms;
    }
    for norm in &mut motion_norms {
        if *norm > 0.0 {
            *norm = (*norm / max_norm).sqrt().clamp(0.0, 1.0);
        }
    }
    motion_norms
}

pub(crate) fn mesh_motion_candidate_weights_with_local_front_floor(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
    floor: f32,
    motion_weights: &[f32],
) -> Vec<f32> {
    if positions.is_empty()
        || motion_weights.len() < positions.len()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || floor <= 0.0
        || !floor.is_finite()
    {
        return motion_weights.to_vec();
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let mut weights = motion_weights.to_vec();
    for row in 0..positions.len() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 || front_weights.get(row).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        weights[row] = weights[row].max(floor.clamp(0.0, 1.0));
    }
    weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn target_coverage_liveness_candidate_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<f32> {
    let mut weights = vec![0.0_f32; positions.len()];
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || coverage_gain <= 0.0
        || !coverage_gain.is_finite()
    {
        return weights;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let mut row_weights = vec![0.0_f32; positions.len()];
    let mut has_dormant_front = false;
    for row in 0..positions.len() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 {
            row_weights[row] = 1.0;
            continue;
        }
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        if front_weight <= 1.0e-3 || !front_weight.is_finite() {
            continue;
        }
        row_weights[row] = front_weight.clamp(0.0, 1.0);
        has_dormant_front = true;
    }
    if !has_dormant_front {
        return weights;
    }

    let coverage_updates = render_proxy_weighted_target_coverage_updates(
        target,
        positions,
        &row_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    );
    let normalizer = if max_update_norm > 0.0 && max_update_norm.is_finite() {
        max_update_norm
    } else {
        coverage_updates
            .iter()
            .map(|update| {
                update
                    .iter()
                    .take(config.spatial_dims)
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
            })
            .filter(|norm| norm.is_finite())
            .fold(0.0_f32, f32::max)
    }
    .max(1.0e-6);

    for row in 0..positions.len() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 || row_weights[row] <= 1.0e-3 {
            continue;
        }
        let norm = coverage_updates[row]
            .iter()
            .take(config.spatial_dims)
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm <= 1.0e-8 || !norm.is_finite() {
            continue;
        }
        weights[row] = (norm / normalizer).sqrt().clamp(0.0, 1.0);
    }
    weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_coverage_liveness_candidate_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    front_radius: f32,
    activation_candidate_weights: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<f32> {
    let mut weights = vec![0.0_f32; positions.len()];
    let output_dims = config.update_dims();
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || activation_candidate_weights.len() < positions.len()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || coverage_gain <= 0.0
        || !coverage_gain.is_finite()
    {
        return weights;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return weights;
    };
    let material_output = config.spatial_dims + material_channel;
    if material_output >= output_dims {
        return weights;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let mut row_weights = vec![0.0_f32; positions.len()];
    let mut has_dormant_front_candidate = false;
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        let predicted_material =
            material_opacity + raw_updates[row * output_dims + material_output];
        let visible_logit = material_opacity.max(predicted_material);
        let material_weight = if visible_logit > material_visible_threshold {
            ((visible_logit - material_visible_threshold) / 4.0).clamp(0.25, 1.0)
        } else {
            0.0
        };
        if liveness > -1.0 {
            row_weights[row] = material_weight;
            continue;
        }
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        let activation_weight = activation_candidate_weights[row].clamp(0.0, 1.0);
        if front_weight <= 1.0e-3
            || activation_weight <= 0.0
            || !front_weight.is_finite()
            || !activation_weight.is_finite()
        {
            continue;
        }
        row_weights[row] = front_weight * activation_weight * material_weight.max(0.25);
        has_dormant_front_candidate = true;
    }
    if !has_dormant_front_candidate {
        return weights;
    }

    let coverage_updates = render_proxy_weighted_target_coverage_updates(
        target,
        positions,
        &row_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    );
    let normalizer = if max_update_norm > 0.0 && max_update_norm.is_finite() {
        max_update_norm
    } else {
        coverage_updates
            .iter()
            .map(|update| {
                update
                    .iter()
                    .take(config.spatial_dims)
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
            })
            .filter(|norm| norm.is_finite())
            .fold(0.0_f32, f32::max)
    }
    .max(1.0e-6);

    for row in 0..positions.len() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 || row_weights[row] <= 1.0e-3 {
            continue;
        }
        let norm = coverage_updates[row]
            .iter()
            .take(config.spatial_dims)
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm <= 1.0e-8 || !norm.is_finite() {
            continue;
        }
        weights[row] = (norm / normalizer).sqrt().clamp(0.0, 1.0);
    }
    weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_coverage_front_motion_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    front_radius: f32,
    candidate_weights: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let updates = vec![[0.0_f32; 3]; positions.len()];
    let output_dims = config.update_dims();
    if positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || candidate_weights.len() < positions.len()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || coverage_gain <= 0.0
        || !coverage_gain.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let material_output = config.spatial_dims + material_channel;
    if material_output >= output_dims {
        return updates;
    }

    let front_weights = local_front_weights(config, positions, states, front_radius);
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let mut row_weights = vec![0.0_f32; positions.len()];
    let mut has_candidate = false;
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        let predicted_material =
            material_opacity + raw_updates[row * output_dims + material_output];
        let visible_logit = material_opacity.max(predicted_material);
        let material_weight = if visible_logit > material_visible_threshold {
            ((visible_logit - material_visible_threshold) / 4.0).clamp(0.25, 1.0)
        } else {
            0.0
        };
        if liveness > -1.0 {
            row_weights[row] = material_weight;
        } else {
            let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
            let candidate_weight = candidate_weights[row].clamp(0.0, 1.0);
            if front_weight <= 1.0e-3
                || candidate_weight <= 0.0
                || !front_weight.is_finite()
                || !candidate_weight.is_finite()
            {
                continue;
            }
            row_weights[row] = front_weight * candidate_weight * material_weight.max(0.25);
        }
        has_candidate |= row_weights[row] > 1.0e-3 && row_weights[row].is_finite();
    }
    if !has_candidate {
        return updates;
    }

    render_proxy_weighted_target_coverage_updates(
        target,
        positions,
        &row_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_coverage_front_motion_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    front_radius: f32,
    candidate_weights: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if weight <= 0.0
        || !weight.is_finite()
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
    {
        return;
    }
    let updates = material_coverage_front_motion_updates(
        config,
        target,
        positions,
        states,
        raw_updates,
        front_radius,
        candidate_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    );
    for (row, update) in updates.iter().enumerate() {
        if update
            .iter()
            .take(config.spatial_dims)
            .all(|value| value.abs() <= 1.0e-8)
        {
            continue;
        }
        let output_base = row * output_dims;
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - update[axis]);
        }
    }
}

pub(crate) fn temporal_liveness_candidate_weights_with_local_front_floor(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    front_radius: f32,
    floor: f32,
    candidate_weights: &[f32],
) -> Vec<f32> {
    let output_dims = config.update_dims();
    if positions.is_empty()
        || candidate_weights.len() < positions.len()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || floor <= 0.0
        || !floor.is_finite()
    {
        return candidate_weights.to_vec();
    }
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims {
        return candidate_weights.to_vec();
    }

    let rows = positions.len();
    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return candidate_weights.to_vec();
    }
    let active_count = (0..rows)
        .filter(|row| {
            let state_base = row * config.state_dims;
            let output_base = row * output_dims;
            states[state_base + GROWTH_3D_LIVENESS_CHANNEL]
                + raw_updates[output_base + liveness_output]
                > -1.0
        })
        .count();
    let target_active =
        ((rows as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
    let deficit = target_active.saturating_sub(active_count);
    if deficit == 0 {
        return candidate_weights.to_vec();
    }

    let front_weights = local_front_weights_with_min_candidates(
        config,
        positions,
        states,
        front_radius,
        temporal_front_candidate_count(rows, deficit),
    );
    let mut weights = candidate_weights.to_vec();
    let floor = floor.clamp(0.0, 1.0);
    for row in 0..rows {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        if liveness > -1.0 || front_weight <= 0.0 || !front_weight.is_finite() {
            continue;
        }
        weights[row] = weights[row].max(floor);
    }
    weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_mesh_residual_velocity_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    coverage_gain: f32,
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
    front_radius: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(velocity_channels) = growth_3d_velocity_channels(config.state_dims) else {
        return;
    };
    let driver_gain = coverage_gain.max(surface_gain);
    if config.spatial_dims != 3
        || positions.is_empty()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || driver_gain <= 0.0
        || !driver_gain.is_finite()
        || weight <= 0.0
        || !weight.is_finite()
    {
        return;
    }
    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let velocity_outputs = velocity_channels
        .map(|channel| config.spatial_dims + channel)
        .collect::<Vec<_>>();
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let active = states[state_base + GROWTH_3D_LIVENESS_CHANNEL] > -1.0;
        let row_weight = if active {
            1.0
        } else {
            front_weights
                .as_ref()
                .and_then(|weights| weights.get(row))
                .copied()
                .unwrap_or(0.0)
        };
        if row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() {
            continue;
        }
        let surface_weight = surface_escape_weight(
            projection.distance,
            GROWTH_3D_SURFACE_MAX_DISTANCE,
            surface_escape_gain,
        );
        let mut target_velocity = [0.0_f32; 3];
        for axis in 0..config.spatial_dims {
            target_velocity[axis] =
                driver_gain * row_weight * surface_weight * projection.residual[axis];
        }
        clamp_vector3(&mut target_velocity, max_update_norm);
        if target_velocity
            .iter()
            .take(config.spatial_dims)
            .all(|value| value.abs() <= 1.0e-8)
        {
            continue;
        }
        let output_base = row * output_dims;
        for (axis, velocity_output) in velocity_outputs.iter().copied().enumerate() {
            if velocity_output >= output_dims || axis >= config.spatial_dims {
                continue;
            }
            output_gradients[output_base + velocity_output] +=
                weight * (raw_updates[output_base + velocity_output] - target_velocity[axis]);
        }
    }
}

pub(crate) fn add_motion_memory_output_objective(
    config: &NpaConfig,
    mesh_output_gradients: &[f32],
    memory_gain: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(velocity_channels) = growth_3d_velocity_channels(config.state_dims) else {
        return;
    };
    if config.spatial_dims != 3
        || output_dims == 0
        || mesh_output_gradients.len() != output_gradients.len()
        || !mesh_output_gradients.len().is_multiple_of(output_dims)
        || memory_gain <= 0.0
        || !memory_gain.is_finite()
    {
        return;
    }
    for (axis, velocity_channel) in velocity_channels.enumerate().take(config.spatial_dims) {
        let velocity_output = config.spatial_dims + velocity_channel;
        if velocity_output >= output_dims {
            continue;
        }
        for row_base in (0..mesh_output_gradients.len()).step_by(output_dims) {
            output_gradients[row_base + velocity_output] +=
                memory_gain * mesh_output_gradients[row_base + axis];
        }
    }
}

pub(crate) fn add_liveness_phase_memory_output_objective(
    config: &NpaConfig,
    liveness_output_gradients: &[f32],
    memory_gain: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(phase_channel) = growth_3d_phase_channel(config.state_dims) else {
        return;
    };
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let phase_output = config.spatial_dims + phase_channel;
    if output_dims == 0
        || liveness_output >= output_dims
        || phase_output >= output_dims
        || liveness_output_gradients.len() != output_gradients.len()
        || !liveness_output_gradients.len().is_multiple_of(output_dims)
        || memory_gain <= 0.0
        || !memory_gain.is_finite()
    {
        return;
    }

    for row_base in (0..liveness_output_gradients.len()).step_by(output_dims) {
        output_gradients[row_base + phase_output] +=
            memory_gain * liveness_output_gradients[row_base + liveness_output];
    }
}

pub(crate) fn add_extent_motion_memory_output_objective(
    config: &NpaConfig,
    extent_motion_output_gradients: &[f32],
    temporal_extent_motion_output_gradients: &[f32],
    memory_gain: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(velocity_channels) = growth_3d_velocity_channels(config.state_dims) else {
        return;
    };
    if config.spatial_dims != 3
        || output_dims == 0
        || extent_motion_output_gradients.len() != temporal_extent_motion_output_gradients.len()
        || extent_motion_output_gradients.len() != output_gradients.len()
        || !extent_motion_output_gradients
            .len()
            .is_multiple_of(output_dims)
        || memory_gain <= 0.0
        || !memory_gain.is_finite()
    {
        return;
    }

    for (axis, velocity_channel) in velocity_channels.enumerate().take(config.spatial_dims) {
        let velocity_output = config.spatial_dims + velocity_channel;
        if velocity_output >= output_dims {
            continue;
        }
        for row_base in (0..extent_motion_output_gradients.len()).step_by(output_dims) {
            let target = extent_motion_output_gradients[row_base + axis]
                + temporal_extent_motion_output_gradients[row_base + axis];
            output_gradients[row_base + velocity_output] += memory_gain * target;
        }
    }
}

pub(crate) fn growth_3d_velocity_output_channels(config: &NpaConfig) -> Vec<usize> {
    growth_3d_velocity_channels(config.state_dims)
        .map(|channels| {
            channels
                .map(|channel| config.spatial_dims + channel)
                .filter(|output| *output < config.update_dims())
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn add_growth_phase_output_objective(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    phase_gain: f32,
    front_radius: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(phase_channel) = growth_3d_phase_channel(config.state_dims) else {
        return;
    };
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || phase_gain <= 0.0
        || !phase_gain.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
    {
        return;
    }
    let phase_output = config.spatial_dims + phase_channel;
    if phase_output >= output_dims {
        return;
    }

    let rows = positions.len();
    let schedule = step_fraction.clamp(0.0, 1.0);
    if schedule <= 0.0 {
        return;
    }
    let target_active =
        ((rows as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
    let active_count = (0..rows)
        .filter(|row| states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > -1.0)
        .count();
    let deficit = target_active.saturating_sub(active_count);
    let front_weights =
        local_front_weights_with_min_candidates(config, positions, states, front_radius, deficit);

    for row in 0..rows {
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let phase = states[state_base + phase_channel];
        let raw = raw_updates[output_base + phase_output];

        let target_phase = if liveness > -1.0 {
            schedule
        } else {
            let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
            if front_weight > 0.0 {
                (0.5 * schedule * front_weight).min(schedule)
            } else if phase > 0.05 {
                0.0
            } else {
                continue;
            }
        };
        let target_update = (target_phase - phase).clamp(-1.0, 1.0);
        output_gradients[output_base + phase_output] += phase_gain * (raw - target_update);
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn add_material_visible_liveness_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    material_liveness_gain: f32,
    surface_threshold: f32,
    max_liveness_update: f32,
    front_radius: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_output = config.spatial_dims + material_channel;
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || liveness_output >= output_dims
        || material_output >= output_dims
        || material_liveness_gain <= 0.0
        || !material_liveness_gain.is_finite()
        || surface_threshold <= 0.0
        || !surface_threshold.is_finite()
        || front_radius <= 0.0
        || !front_radius.is_finite()
        || weight <= 0.0
        || !weight.is_finite()
    {
        return;
    }
    let max_liveness_update = if max_liveness_update.is_finite() && max_liveness_update > 0.0 {
        max_liveness_update
    } else {
        f32::INFINITY
    };
    let front_weights = local_front_weights(config, positions, states, front_radius);
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness > -1.0 {
            continue;
        }
        let front_weight = front_weights.get(row).copied().unwrap_or(0.0);
        if front_weight <= 0.0 {
            continue;
        }
        let output_base = row * output_dims;
        let predicted_liveness = liveness + raw_updates[output_base + liveness_output];
        if predicted_liveness > -1.0 {
            continue;
        }
        let material_opacity = states[state_base + material_channel];
        let predicted_material = material_opacity + raw_updates[output_base + material_output];
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > surface_threshold {
            continue;
        }
        let surface_weight = (1.0 - projection.distance / surface_threshold).clamp(0.0, 1.0);
        if surface_weight <= 0.0 {
            continue;
        }
        let material_weight = if material_opacity > material_visible_threshold
            || predicted_material > material_visible_threshold
        {
            ((predicted_material.max(material_opacity) - material_visible_threshold) / 4.0)
                .clamp(0.25, 1.0)
        } else {
            0.25
        };
        let target_update = (material_liveness_gain
            * front_weight
            * surface_weight
            * material_weight
            * (0.0 - liveness))
            .clamp(0.0, max_liveness_update);
        if target_update <= 0.0 {
            continue;
        }
        let raw = raw_updates[output_base + liveness_output];
        output_gradients[output_base + liveness_output] += weight * (raw - target_update);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_mesh_geometry_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    extent_gain: f32,
    surface_gain: f32,
    surface_escape_gain: f32,
    seed_scale: f32,
    max_update_norm: f32,
    front_radius: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.spatial_dims == 0
        || config.spatial_dims > 3
        || output_dims < config.spatial_dims
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || weight <= 0.0
        || !weight.is_finite()
        || ((coverage_gain <= 0.0 || !coverage_gain.is_finite())
            && (extent_gain <= 0.0 || !extent_gain.is_finite())
            && (surface_gain <= 0.0 || !surface_gain.is_finite()))
    {
        return;
    }

    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let mut geometry_weights = vec![0.0_f32; positions.len()];
    for row in 0..positions.len() {
        let active = config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
            || states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > -1.0;
        geometry_weights[row] = if active {
            1.0
        } else {
            front_weights
                .as_ref()
                .and_then(|weights| weights.get(row))
                .copied()
                .unwrap_or(0.0)
        };
    }
    let coverage_updates = render_proxy_weighted_target_coverage_updates(
        target,
        positions,
        &geometry_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    );
    let surface_updates = render_proxy_surface_projection_updates(
        config,
        target,
        positions,
        states,
        surface_gain,
        surface_escape_gain,
        max_update_norm,
    );
    let extent_updates = render_proxy_target_extent_updates(
        config,
        target,
        positions,
        &geometry_weights,
        extent_gain,
        max_update_norm,
    );
    let expansion_updates = render_proxy_local_front_expansion_updates(
        config,
        positions,
        states,
        &geometry_weights,
        coverage_gain.max(surface_gain) * DIRECT_LOCAL_FRONT_EXPANSION_GAIN_FRACTION,
        max_update_norm,
    );

    for row in 0..positions.len() {
        let output_base = row * output_dims;
        let active = config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
            || states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > -1.0;
        let front_weight = geometry_weights[row];
        if front_weight <= 0.0 {
            continue;
        }
        let mut target_update = [0.0_f32; 3];
        if active {
            for axis in 0..config.spatial_dims {
                target_update[axis] = coverage_updates
                    .get(row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0)
                    + surface_updates
                        .get(row)
                        .map(|update| update[axis])
                        .unwrap_or(0.0)
                    + extent_updates
                        .get(row)
                        .map(|update| update[axis])
                        .unwrap_or(0.0);
            }
        } else {
            let projection = target.project(position3(positions[row]));
            let front_gain = coverage_gain.max(surface_gain);
            for axis in 0..config.spatial_dims {
                target_update[axis] = front_weight * front_gain * projection.residual[axis]
                    + extent_updates
                        .get(row)
                        .map(|update| update[axis])
                        .unwrap_or(0.0)
                    + expansion_updates
                        .get(row)
                        .map(|update| update[axis])
                        .unwrap_or(0.0);
            }
        }
        clamp_vector3(&mut target_update, max_update_norm);
        if target_update
            .iter()
            .take(config.spatial_dims)
            .all(|value| *value == 0.0)
        {
            continue;
        }
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - target_update[axis]);
        }
    }
}

pub(crate) fn render_proxy_local_front_expansion_updates(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    row_weights: &[f32],
    expansion_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0_f32; 3]; positions.len()];
    if positions.is_empty()
        || row_weights.len() < positions.len()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || expansion_gain <= 0.0
        || !expansion_gain.is_finite()
    {
        return updates;
    }

    let active_rows = (0..positions.len())
        .filter(|row| states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > -1.0)
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return updates;
    }

    for row in 0..positions.len() {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        let row_weight = row_weights[row];
        if liveness > -1.0 || row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }
        let mut nearest = None::<(usize, f32)>;
        for &active_row in &active_rows {
            let mut distance2 = 0.0_f32;
            for axis in 0..config.spatial_dims {
                let delta = positions[row][axis] - positions[active_row][axis];
                distance2 += delta * delta;
            }
            if distance2.is_finite()
                && nearest
                    .map(|(_, best_distance2)| distance2 < best_distance2)
                    .unwrap_or(true)
            {
                nearest = Some((active_row, distance2));
            }
        }
        let Some((active_row, distance2)) = nearest else {
            continue;
        };
        if distance2 <= 1.0e-12 {
            continue;
        }
        let distance = distance2.sqrt();
        for axis in 0..config.spatial_dims {
            updates[row][axis] =
                expansion_gain * row_weight * (positions[row][axis] - positions[active_row][axis])
                    / distance;
        }
        clamp_vector3(&mut updates[row], max_update_norm);
    }

    updates
}

pub(crate) fn render_proxy_target_extent_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    row_weights: &[f32],
    extent_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0_f32; 3]; positions.len()];
    if positions.is_empty()
        || row_weights.len() < positions.len()
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || extent_gain <= 0.0
        || !extent_gain.is_finite()
    {
        return updates;
    }

    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_rows = 0usize;
    for (row, position) in positions.iter().enumerate() {
        if row_weights[row] <= 1.0e-3 || !row_weights[row].is_finite() {
            continue;
        }
        active_rows += 1;
        for axis in 0..config.spatial_dims {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_rows == 0 {
        return updates;
    }

    let (target_min, target_max) = target.bounds();
    for (row, position) in positions.iter().enumerate() {
        let row_weight = row_weights[row];
        if row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }
        for axis in 0..config.spatial_dims {
            let active_extent = (active_max[axis] - active_min[axis]).max(1.0e-4);
            let t = ((position[axis] - active_min[axis]) / active_extent).clamp(0.0, 1.0);
            let min_weight = (1.0 - t).powi(3);
            let max_weight = t.powi(3);
            let residual = min_weight * (target_min[axis] - position[axis])
                + max_weight * (target_max[axis] - position[axis]);
            updates[row][axis] += extent_gain * row_weight * residual;
        }
        clamp_vector3(&mut updates[row], max_update_norm);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visibility_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    opacity_gain: f32,
    material_liveness_gain: f32,
    material_tail_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    max_opacity_update: f32,
    material_suppression_update_multiplier: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    step_fraction: f32,
    max_liveness_update: f32,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let material_output = config.spatial_dims + material_channel;
    if material_output >= output_dims
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || weight <= 0.0
        || !weight.is_finite()
        || ((opacity_gain <= 0.0 || !opacity_gain.is_finite())
            && (material_liveness_gain <= 0.0 || !material_liveness_gain.is_finite())
            && (material_tail_gain <= 0.0 || !material_tail_gain.is_finite()))
    {
        return;
    }

    let positive_cap = if max_opacity_update.is_finite() && max_opacity_update > 0.0 {
        max_opacity_update
    } else {
        f32::INFINITY
    };
    let suppression_cap =
        material_suppression_max_update(max_opacity_update, material_suppression_update_multiplier);
    let negative_cap = if suppression_cap.is_finite() && suppression_cap > 0.0 {
        suppression_cap
    } else {
        f32::INFINITY
    };
    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let liveness_enabled = liveness_output < output_dims
        && material_liveness_gain > 0.0
        && material_liveness_gain.is_finite();
    let max_liveness_update = if max_liveness_update.is_finite() && max_liveness_update > 0.0 {
        max_liveness_update
    } else {
        f32::INFINITY
    };
    let schedule = step_fraction.clamp(0.0, 1.0);
    let predicted_liveness = (0..positions.len())
        .map(|row| {
            let state_liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            if liveness_output < output_dims {
                state_liveness + raw_updates[row * output_dims + liveness_output]
            } else {
                state_liveness
            }
        })
        .collect::<Vec<_>>();
    let predicted_active_count = predicted_liveness
        .iter()
        .filter(|liveness| **liveness > -1.0)
        .count();
    let liveness_target_count =
        ((positions.len() as f32) * temporal_activation_target_fraction(schedule)).ceil() as usize;
    let mut liveness_deficit = liveness_target_count.saturating_sub(predicted_active_count);
    let mut liveness_candidates = Vec::<(usize, f32, f32)>::new();
    let mut material_candidate_weights = vec![0.0_f32; positions.len()];
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let front_weight = front_weights
            .as_ref()
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(0.0);
        let activation_weight = activation_candidate_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        material_candidate_weights[row] = if liveness > -1.0 {
            1.0
        } else {
            activation_weight * front_weight.max((predicted_liveness[row] + 1.0).clamp(0.0, 1.0))
        };
    }
    let material_coverage_updates = material_target_coverage_opacity_updates_weighted(
        config,
        target,
        positions,
        states,
        Some(&material_candidate_weights),
        opacity_gain,
        coverage_samples,
        seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        max_opacity_update,
    );
    let material_strata_updates = material_surface_strata_opacity_updates_weighted(
        config,
        target,
        positions,
        states,
        Some(&material_candidate_weights),
        opacity_gain,
        coverage_samples,
        seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        max_opacity_update,
    );
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let front_weight = front_weights
            .as_ref()
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(0.0);
        let candidate_weight = material_candidate_weights[row];
        let material_candidate = candidate_weight > 0.0;
        let material_index = state_base + material_channel;
        let material_opacity = states[material_index];
        let projection = target.project(position3(*position));
        let surface_weight =
            soft_material_assignment_weight(projection.distance, strict_threshold, soft_threshold);
        let activation_surface_weight = if liveness <= -1.0 {
            surface_weight.max(0.5 * candidate_weight)
        } else {
            surface_weight
        };
        let mut material_delta = 0.0_f32;
        if opacity_gain > 0.0 && opacity_gain.is_finite() && material_candidate {
            if activation_surface_weight > 0.0 {
                material_delta += opacity_gain
                    * activation_surface_weight
                    * candidate_weight
                    * (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_opacity);
            }
            material_delta += material_coverage_updates.get(row).copied().unwrap_or(0.0);
            material_delta += material_strata_updates.get(row).copied().unwrap_or(0.0);
        }
        if material_liveness_gain > 0.0
            && material_liveness_gain.is_finite()
            && liveness <= -1.0
            && predicted_liveness[row] <= -1.0
            && front_weight <= 0.0
            && material_opacity > GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
        {
            material_delta -= material_liveness_gain
                * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT);
        }
        if material_tail_gain > 0.0
            && material_tail_gain.is_finite()
            && material_opacity > GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
            && projection.distance.is_finite()
            && projection.distance > GROWTH_3D_SURFACE_MAX_DISTANCE
        {
            let escape = (projection.distance / GROWTH_3D_SURFACE_MAX_DISTANCE - 1.0).max(0.0);
            material_delta -= material_tail_gain
                * escape.min(8.0)
                * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT);
        }
        if liveness_enabled
            && liveness_deficit > 0
            && material_delta > 0.0
            && liveness <= -1.0
            && predicted_liveness[row] <= -1.0
            && front_weight > 0.0
        {
            if activation_surface_weight > 0.0 {
                let score =
                    (front_weight * activation_surface_weight * candidate_weight).clamp(0.0, 1.0);
                if score > 0.0 {
                    liveness_candidates.push((row, score, liveness));
                }
            }
        }
        if material_delta == 0.0 {
            continue;
        }
        let capped_delta = material_delta.clamp(-negative_cap, positive_cap);
        let output_index = row * output_dims + material_output;
        let raw = raw_updates[output_index];
        output_gradients[output_index] += weight * (raw - capped_delta);
    }

    if liveness_enabled && liveness_deficit > 0 && !liveness_candidates.is_empty() {
        liveness_candidates.sort_by(
            |(_, lhs_score, lhs_liveness), (_, rhs_score, rhs_liveness)| {
                rhs_score
                    .partial_cmp(lhs_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        rhs_liveness
                            .partial_cmp(lhs_liveness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            },
        );
        let target_liveness = temporal_activation_candidate_liveness_target(schedule);
        for (row, score, liveness) in liveness_candidates.into_iter().take(liveness_deficit) {
            let output_index = row * output_dims + liveness_output;
            let raw = raw_updates[output_index];
            let target_update = (material_liveness_gain * score * (target_liveness - liveness))
                .clamp(0.0, max_liveness_update);
            if target_update > 0.0 {
                output_gradients[output_index] += weight * (raw - target_update);
            }
            liveness_deficit = liveness_deficit.saturating_sub(1);
            if liveness_deficit == 0 {
                break;
            }
        }
    }
}

pub(crate) fn add_gaussian_scale_budget_state_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
    state_adjoint: &mut [f32],
) {
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || render_cfg.gaussian_decode_mode != GaussianDecodeMode::GaussianSh0LearnedScale
        || config.state_dims < 5
    {
        return;
    }
    let scale_channel = config.state_dims - 5;
    for particle_row in 0..trace.particle_count {
        let state_base = particle_row * trace.state_dims;
        if state_base + config.state_dims > trace.states.len()
            || state_base + scale_channel >= state_adjoint.len()
        {
            continue;
        }
        let state = &trace.states[state_base..state_base + config.state_dims];
        state_adjoint[state_base + scale_channel] +=
            gaussian_scale_budget_logit_gradient(state, render_cfg, scale_budget_weight);
    }
}

pub(crate) fn gaussian_scale_budget_logit_gradient(
    state: &[f32],
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
) -> f32 {
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || render_cfg.gaussian_decode_mode != GaussianDecodeMode::GaussianSh0LearnedScale
        || state.len() < 5
    {
        return 0.0;
    }
    let expected_scale = render_cfg
        .sigma
        .clamp(render_cfg.min_sigma, render_cfg.max_sigma)
        .max(1.0e-8);
    let scale_logit = state[state.len() - 5].clamp(-8.0, 8.0);
    let scale = (render_cfg.sigma * scale_logit.exp())
        .clamp(render_cfg.min_sigma, render_cfg.max_sigma)
        .max(1.0e-8);
    let loss = scale_budget_loss_for_scale(scale, expected_scale);
    if !loss.is_finite() || loss <= 0.0 {
        return 0.0;
    }
    let oversize_ratio = scale / expected_scale - 1.0;
    2.0 * scale_budget_weight * oversize_ratio * scale / expected_scale
}
