#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

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
        let material_surface_weight = surface_weight;
        let liveness_surface_weight = if liveness <= -1.0 {
            surface_weight.max(0.5 * candidate_weight)
        } else {
            surface_weight
        };
        let mut material_delta = 0.0_f32;
        if opacity_gain > 0.0 && opacity_gain.is_finite() && material_candidate {
            if material_surface_weight > 0.0 {
                material_delta += opacity_gain
                    * material_surface_weight
                    * candidate_weight
                    * (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_opacity);
                material_delta += material_surface_weight
                    * material_coverage_updates.get(row).copied().unwrap_or(0.0);
                material_delta += material_surface_weight
                    * material_strata_updates.get(row).copied().unwrap_or(0.0);
            }
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
            if liveness_surface_weight > 0.0 {
                let score =
                    (front_weight * liveness_surface_weight * candidate_weight).clamp(0.0, 1.0);
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
