#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

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
    let candidate_row_weights =
        dormant_candidate_row_weights(config, states, positions.len(), &row_weights);
    let normalizer = coverage_update_weight_normalizer_for_row_weights(
        &coverage_updates,
        config.spatial_dims,
        max_update_norm,
        &candidate_row_weights,
    );

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
    let candidate_row_weights =
        dormant_candidate_row_weights(config, states, positions.len(), &row_weights);
    let normalizer = coverage_update_weight_normalizer_for_row_weights(
        &coverage_updates,
        config.spatial_dims,
        max_update_norm,
        &candidate_row_weights,
    );

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

#[cfg(test)]
pub(crate) fn coverage_update_weight_normalizer(
    updates: &[[f32; 3]],
    spatial_dims: usize,
    max_update_norm: f32,
) -> f32 {
    coverage_update_weight_normalizer_for_row_weights(updates, spatial_dims, max_update_norm, &[])
}

fn coverage_update_weight_normalizer_for_row_weights(
    updates: &[[f32; 3]],
    spatial_dims: usize,
    max_update_norm: f32,
    row_weights: &[f32],
) -> f32 {
    let observed = updates
        .iter()
        .enumerate()
        .filter(|(row, _)| {
            row_weights
                .get(*row)
                .map(|weight| *weight > 1.0e-3 && weight.is_finite())
                .unwrap_or(row_weights.is_empty())
        })
        .map(|(_, update)| vector_update_norm(update, spatial_dims))
        .filter(|norm| norm.is_finite())
        .fold(0.0_f32, f32::max);
    let capped = if max_update_norm > 0.0 && max_update_norm.is_finite() {
        observed.min(max_update_norm)
    } else {
        observed
    };
    capped.max(1.0e-6)
}

fn vector_update_norm(update: &[f32; 3], spatial_dims: usize) -> f32 {
    update
        .iter()
        .take(spatial_dims.min(3))
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn dormant_candidate_row_weights(
    config: &NpaConfig,
    states: &[f32],
    rows: usize,
    row_weights: &[f32],
) -> Vec<f32> {
    let mut candidate_weights = vec![0.0_f32; rows];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL || states.len() < rows * config.state_dims {
        return candidate_weights;
    }
    for row in 0..rows {
        let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            candidate_weights[row] = row_weights.get(row).copied().unwrap_or(0.0);
        }
    }
    candidate_weights
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
