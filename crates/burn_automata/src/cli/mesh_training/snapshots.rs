#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn mesh_rollout_snapshot_steps(
    rollout_steps: usize,
    temporal_samples: usize,
) -> Vec<usize> {
    let samples = temporal_samples.max(1);
    if samples == 1 {
        return vec![rollout_steps];
    }
    if rollout_steps == 0 {
        return vec![0];
    }
    let mut steps = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let step = sample_idx * rollout_steps / (samples - 1);
        if steps.last().copied() != Some(step) {
            steps.push(step);
        }
    }
    if steps.last().copied() != Some(rollout_steps) {
        steps.push(rollout_steps);
    }
    steps
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn append_mesh_rollout_snapshot_rows(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &MeshFieldRolloutBatchConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    max_rows: usize,
    features: &mut Vec<f32>,
    target_update: &mut Vec<f32>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let row_budget = cfg.particle_count.min(max_rows);
    if row_budget == 0 {
        return Ok(0);
    }
    let step = model.step_cpu(positions, states, 1, cfg.particle_count, grid, 1.0, None)?;
    let mut rollout_target_update = mesh_field_target_update_for_rows(
        &model.config,
        target,
        positions,
        states,
        cfg.motion_gain,
        cfg.max_update_norm,
        cfg.color_gain,
        cfg.aux_state_gain,
        cfg.opacity_gain,
        cfg.front_opacity_gain,
        cfg.front_radius,
        cfg.front_max_opacity_update,
        cfg.front_motion_gate,
    );
    add_target_coverage_updates_for_rows(
        &model.config,
        target,
        positions,
        &mut rollout_target_update,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.seed_scale,
        cfg.max_update_norm,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        cfg.front_radius,
    );
    add_target_extent_updates_for_rows(
        &model.config,
        target,
        positions,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        &mut rollout_target_update,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.front_radius,
    );
    if cfg.preserve_opacity_update && model.config.state_dims > 3 {
        let output_dims = model.config.update_dims();
        for row in 0..cfg.particle_count.min(positions.len()) {
            let update_base = row * output_dims + model.config.spatial_dims + 3;
            let state_base = row * model.config.state_dims + 3;
            if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                rollout_target_update[update_base] = step.ds[state_base];
            }
            if let Some(channel) = growth_3d_material_opacity_channel(model.config.state_dims)
                && channel != 3
            {
                let update_base = row * output_dims + model.config.spatial_dims + channel;
                let state_base = row * model.config.state_dims + channel;
                if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                    rollout_target_update[update_base] = step.ds[state_base];
                }
            }
        }
    }
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let row_indices = mesh_rollout_row_indices(
        &rollout_target_update,
        output_dims,
        cfg.particle_count,
        row_budget,
    );
    for row in row_indices {
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        features
            .extend_from_slice(&step.perception.features[feature_base..feature_base + input_dims]);
        target_update
            .extend_from_slice(&rollout_target_update[update_base..update_base + output_dims]);
    }
    Ok(row_budget)
}

pub(crate) fn mesh_rollout_row_indices(
    target_update: &[f32],
    output_dims: usize,
    particle_count: usize,
    row_budget: usize,
) -> Vec<usize> {
    let rows = particle_count.min(row_budget);
    if rows >= particle_count {
        return (0..particle_count).collect();
    }
    if rows == 0 || output_dims == 0 {
        return Vec::new();
    }

    let spread_budget = (rows / 4).max(1).min(rows);
    let mut selected = vec![false; particle_count];
    let mut row_indices = Vec::with_capacity(rows);
    for row in spread_row_indices(particle_count, spread_budget) {
        if row < particle_count && !selected[row] {
            selected[row] = true;
            row_indices.push(row);
        }
    }

    let mut scored_rows = (0..particle_count)
        .map(|row| {
            let base = row * output_dims;
            let score = target_update
                .get(base..base + output_dims)
                .unwrap_or(&[])
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value * value)
                .sum::<f32>();
            (row, score)
        })
        .collect::<Vec<_>>();
    scored_rows.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    for (row, score) in scored_rows {
        if row_indices.len() >= rows {
            break;
        }
        if score <= 0.0 || selected[row] {
            continue;
        }
        selected[row] = true;
        row_indices.push(row);
    }
    if row_indices.len() < rows {
        for row in spread_row_indices(particle_count, particle_count) {
            if row_indices.len() >= rows {
                break;
            }
            if !selected[row] {
                selected[row] = true;
                row_indices.push(row);
            }
        }
    }
    row_indices
}

pub(crate) fn mesh_field_target_update_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    motion_gain: f32,
    max_update_norm: f32,
    color_gain: f32,
    aux_state_gain: f32,
    opacity_gain: f32,
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
    front_motion_gate: bool,
) -> Vec<f32> {
    let rows = positions.len();
    let output_dims = config.update_dims();
    let mut target_update = vec![0.0; rows * output_dims];
    let front_targets = local_front_opacity_targets(
        config,
        positions,
        states,
        front_opacity_gain,
        front_radius,
        front_max_opacity_update,
    );
    let front_weights = if front_motion_gate {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let target_radius = target
        .vertices
        .iter()
        .map(|vertex| {
            (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt()
        })
        .fold(1.0e-4_f32, f32::max);
    for (row, position) in positions.iter().enumerate() {
        let projection = target.project([position[0], position[1], position[2]]);
        let update_base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        for axis in 0..3 {
            target_update[update_base + axis] =
                front_weight * motion_gain * projection.residual[axis];
        }
        let update_norm = (target_update[update_base].powi(2)
            + target_update[update_base + 1].powi(2)
            + target_update[update_base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && update_norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / update_norm;
            for axis in 0..3 {
                target_update[update_base + axis] *= scale;
            }
        }

        let state_base = row * config.state_dims;
        if config.state_dims >= 3 {
            for axis in 0..3 {
                let target_coordinate = projection.closest[axis] / target_radius.max(1.0e-4);
                target_update[update_base + config.spatial_dims + axis] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_COORDINATE_GAIN
                    * (target_coordinate - states[state_base + axis]);
            }
        }
        if config.state_dims > 3 {
            target_update[update_base + config.spatial_dims + 3] = front_targets[row];
        }
        if let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims) {
            let current_opacity = states[state_base + opacity_channel];
            let surface_band = (target_radius * 0.10).max(0.04);
            let surface_weight = (1.0 - projection.distance / surface_band).clamp(0.0, 1.0);
            let target_opacity = GROWTH_3D_INACTIVE_OPACITY_LOGIT
                + surface_weight
                    * (UV_TORUS_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
            let direct_opacity_update =
                front_weight * opacity_gain * (target_opacity - current_opacity);
            target_update[update_base + config.spatial_dims + opacity_channel] +=
                direct_opacity_update;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = [
                projection.color[0] - 0.5,
                projection.color[1] - 0.5,
                projection.color[2] - 0.5,
            ];
            for channel in 0..3 {
                let current_tail = states[state_base + tail + channel];
                target_update[update_base + config.spatial_dims + tail + channel] =
                    front_weight * color_gain * (target_tail[channel] - current_tail);
            }
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            for axis in 0..3 {
                let channel = UV_TORUS_NORMAL_STATE_OFFSET + axis;
                let current = states[state_base + channel];
                target_update[update_base + config.spatial_dims + channel] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_ORIENTATION_GAIN
                    * (projection.normal[axis] - current);
            }
            let current_signed_distance =
                states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
            target_update
                [update_base + config.spatial_dims + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_SIGNED_DISTANCE_GAIN
                    * (projection.signed_distance - current_signed_distance);
        }
    }
    target_update
}

pub(crate) fn local_front_opacity_targets(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
) -> Vec<f32> {
    let rows = positions.len();
    let mut updates = vec![0.0; rows];
    if config.state_dims <= 3
        || rows == 0
        || front_opacity_gain <= 0.0
        || front_radius <= 0.0
        || front_max_opacity_update <= 0.0
    {
        return updates;
    }

    let dormant_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let current_opacity = states[state_base + 3];
        let mut target_opacity = if front_weights[row] >= 1.0 {
            UV_TORUS_FIELD_OPACITY_TARGET
        } else {
            dormant_target
        };

        if front_weights[row] > 0.0 && front_weights[row] < 1.0 {
            target_opacity = dormant_target
                + front_weights[row] * (UV_TORUS_FIELD_OPACITY_TARGET - dormant_target);
        }

        let delta = front_opacity_gain * (target_opacity - current_opacity);
        updates[row] = delta.clamp(-front_max_opacity_update, front_max_opacity_update);
    }

    updates
}
