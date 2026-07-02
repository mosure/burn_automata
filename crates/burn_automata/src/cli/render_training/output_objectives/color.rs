#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_color_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    color_gain: f32,
    seed_scale: f32,
    max_color_update: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if config.state_dims < 6
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || positions.is_empty()
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || raw_updates.len() < positions.len().saturating_mul(output_dims)
        || output_gradients.len() < positions.len().saturating_mul(output_dims)
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
        || color_gain <= 0.0
        || !color_gain.is_finite()
        || seed_scale <= 0.0
        || !seed_scale.is_finite()
    {
        return;
    }

    let color_state = config.state_dims - 3;
    let color_output = config.spatial_dims + color_state;
    if color_output + 2 >= output_dims {
        return;
    }
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let max_color_update = if max_color_update.is_finite() && max_color_update > 0.0 {
        max_color_update
    } else {
        f32::INFINITY
    };
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let predicted_liveness = if liveness_output < output_dims {
            liveness + raw_updates[output_base + liveness_output]
        } else {
            liveness
        };
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

        let row_weight = color_gain * activity_weight * surface_weight;
        for channel in 0..3 {
            let target_state = 2.0 * projection.color[channel].clamp(0.0, 1.0) - 1.0;
            let state_index = state_base + color_state + channel;
            let output_index = output_base + color_output + channel;
            let target_update =
                (target_state - states[state_index]).clamp(-max_color_update, max_color_update);
            if target_update.abs() <= 1.0e-8 || !target_update.is_finite() {
                continue;
            }
            output_gradients[output_index] +=
                row_weight * (raw_updates[output_index] - target_update);
        }
    }
}

pub(crate) fn growth_3d_color_output_channels(config: &NpaConfig) -> Option<[usize; 3]> {
    (config.state_dims >= 6)
        .then(|| {
            let color_state = config.state_dims - 3;
            [
                config.spatial_dims + color_state,
                config.spatial_dims + color_state + 1,
                config.spatial_dims + color_state + 2,
            ]
        })
        .filter(|channels| {
            let output_dims = config.update_dims();
            channels.iter().all(|channel| *channel < output_dims)
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_boosted_surface_color_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    color_gain: f32,
    seed_scale: f32,
    max_color_update: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    output_gradient_rms_cap: f32,
    output_gradients: &mut [f32],
) -> Option<[usize; 3]> {
    add_surface_color_output_objective(
        config,
        target,
        positions,
        states,
        raw_updates,
        color_gain,
        seed_scale,
        max_color_update,
        front_radius,
        activation_candidate_weights,
        output_gradients,
    );
    let color_outputs = growth_3d_color_output_channels(config)?;
    boost_sparse_output_channel_rms(
        output_gradients,
        config.update_dims(),
        color_outputs,
        output_gradient_rms_cap,
        16.0,
    );
    Some(color_outputs)
}
