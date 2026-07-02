#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::super::*;

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
