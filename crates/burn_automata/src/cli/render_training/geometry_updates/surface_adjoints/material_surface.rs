#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_surface_row_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut row_weights = vec![0.0; positions.len()];
    if positions.is_empty()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len() * config.state_dims
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
    {
        return row_weights;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return row_weights;
    };
    let output_dims = config.update_dims();
    let material_output = config.spatial_dims + material_channel;
    if raw_updates.is_some_and(|updates| updates.len() < positions.len() * output_dims)
        || material_output >= output_dims
    {
        return row_weights;
    }

    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        let predicted_material = raw_updates
            .and_then(|updates| updates.get(row * output_dims + material_output))
            .map(|update| material_opacity + *update)
            .unwrap_or(material_opacity);
        let visible_logit = material_opacity.max(predicted_material);
        if visible_logit <= material_visible_threshold {
            continue;
        }
        let material_weight = ((visible_logit - material_visible_threshold) / 4.0).clamp(0.25, 1.0);
        row_weights[row] = if liveness > -1.0 {
            material_weight
        } else {
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
            material_weight * front_weight * activation_weight
        };
    }
    row_weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_surface_candidate_row_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut row_weights = material_visible_surface_row_weights(
        config,
        positions,
        states,
        raw_updates,
        front_radius,
        activation_candidate_weights,
    );
    if positions.is_empty()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
    {
        return row_weights;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return row_weights;
    };
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims
        || raw_updates.is_some_and(|updates| updates.len() < positions.len() * output_dims)
    {
        return row_weights;
    }

    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let target_span =
        (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_visible_threshold).max(1.0e-6);
    for (row, position) in positions.iter().enumerate() {
        if row_weights[row] >= 1.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let predicted_liveness = raw_updates
            .and_then(|updates| updates.get(output_base + liveness_output))
            .map(|update| liveness + *update)
            .unwrap_or(liveness);
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
        let surface_weight = frontier_material_assignment_weight(
            projection.distance,
            strict_threshold,
            soft_threshold,
            frontier_threshold,
        );
        if surface_weight <= 0.0 {
            continue;
        }
        let material = states[state_base + material_channel];
        let material_progress =
            ((material - material_visible_threshold) / target_span).clamp(0.0, 1.0);
        let candidate_floor = 0.25 + 0.25 * material_progress;
        row_weights[row] = row_weights[row].max(
            (candidate_floor * activity_weight * surface_weight)
                .clamp(0.0, 1.0)
                .min(0.5),
        );
    }
    row_weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_surface_coverage_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
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
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<[f32; 3]> {
    let row_weights = material_surface_candidate_row_weights(
        config,
        target,
        positions,
        states,
        raw_updates,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
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
pub(crate) fn material_visible_surface_approach_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0; 3]; positions.len()];
    if positions.is_empty()
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len() * config.state_dims
        || surface_gain <= 0.0
        || !surface_gain.is_finite()
    {
        return updates;
    }
    let row_weights = material_surface_candidate_row_weights(
        config,
        target,
        positions,
        states,
        raw_updates,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
    for (row, position) in positions.iter().enumerate() {
        let row_weight = row_weights[row];
        if row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }

        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() {
            continue;
        }
        let surface_weight = material_visible_surface_approach_weight(
            projection.distance,
            seed_scale,
            surface_escape_gain,
        );
        for axis in 0..config.spatial_dims {
            updates[row][axis] =
                surface_gain * row_weight * surface_weight * projection.residual[axis];
        }
        clamp_update_row(&mut updates, row, max_update_norm);
    }
    updates
}

pub(crate) fn material_visible_surface_approach_weight(
    distance: f32,
    seed_scale: f32,
    surface_escape_gain: f32,
) -> f32 {
    let escape_weight = surface_escape_weight(
        distance,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
        surface_escape_gain,
    );
    if !distance.is_finite() || !seed_scale.is_finite() || seed_scale <= 0.0 {
        return escape_weight;
    }

    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    if distance <= strict_threshold || distance >= frontier_threshold {
        return escape_weight;
    }

    let band_weight = if distance <= soft_threshold {
        (distance - strict_threshold) / (soft_threshold - strict_threshold).max(1.0e-6)
    } else {
        1.0 - (distance - soft_threshold) / (frontier_threshold - soft_threshold).max(1.0e-6)
    }
    .clamp(0.0, 1.0);

    escape_weight.max(1.0 + 3.0 * band_weight)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_approach_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if weight <= 0.0
        || !weight.is_finite()
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
    {
        return;
    }
    let updates = material_visible_surface_approach_updates(
        config,
        target,
        positions,
        states,
        Some(raw_updates),
        surface_gain,
        surface_escape_gain,
        max_update_norm,
        seed_scale,
        front_radius,
        activation_candidate_weights,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_coverage_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
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
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if weight <= 0.0
        || !weight.is_finite()
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
    {
        return;
    }
    let updates = material_visible_surface_coverage_updates(
        config,
        target,
        positions,
        states,
        Some(raw_updates),
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
        front_radius,
        activation_candidate_weights,
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
