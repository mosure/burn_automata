#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

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
