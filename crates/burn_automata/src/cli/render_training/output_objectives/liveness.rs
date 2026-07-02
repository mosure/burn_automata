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
            material_visible_surface_materialization_weight(projection.distance, seed_scale);
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
