#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::*;

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
