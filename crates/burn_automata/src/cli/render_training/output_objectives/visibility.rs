#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(clippy::too_many_arguments)]
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
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
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
    let material_normal_updates = material_surface_normal_opacity_updates_weighted(
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
        let predicted_live = liveness > -1.0 || predicted_liveness[row] > -1.0;
        let pending_liveness = liveness <= -1.0 && predicted_liveness[row] <= -1.0;
        let material_candidate = candidate_weight > 0.0;
        let material_index = state_base + material_channel;
        let material_opacity = states[material_index];
        let output_base = row * output_dims;
        let predicted_material = material_opacity + raw_updates[output_base + material_output];
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
                material_delta += material_surface_weight
                    * material_normal_updates.get(row).copied().unwrap_or(0.0);
            }
        }
        if pending_liveness {
            let precursor_delta = material_precursor_ceiling() - material_opacity;
            if predicted_material > material_visible_threshold {
                material_delta = if material_delta == 0.0 {
                    precursor_delta
                } else {
                    material_delta.min(precursor_delta)
                };
            } else if material_delta > 0.0 {
                material_delta = material_delta.min(precursor_delta.max(0.0));
            }
        }
        if material_liveness_gain > 0.0
            && material_liveness_gain.is_finite()
            && pending_liveness
            && material_opacity > material_visible_threshold
        {
            material_delta -=
                material_liveness_gain * (material_opacity - material_visible_threshold);
        }
        if material_liveness_gain > 0.0
            && material_liveness_gain.is_finite()
            && pending_liveness
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
            && (material_delta > 0.0
                || material_opacity > material_visible_threshold
                || predicted_material > material_visible_threshold)
            && !predicted_live
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
