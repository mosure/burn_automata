use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_temporal_materialization_output_objective_with_candidate_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    material_gain: f32,
    front_radius: f32,
    candidate_weights: &[f32],
    seed_scale: f32,
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
            let surface_weight =
                surface_precursor_material_weight(target, positions[row], seed_scale);
            if surface_weight <= 0.0 {
                return None;
            }
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
            let score = (front_weight * candidate_weight * surface_weight).clamp(0.0, 1.0);
            let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
            let predicted_liveness = liveness + raw_updates[output_base + liveness_output];
            let target_update = materialization_target_update_with_liveness_gate(
                material,
                target_material,
                max_material_update,
                liveness,
                predicted_liveness,
            );
            (score > 0.0 && score.is_finite() && target_update > 0.0 && target_update.is_finite())
                .then_some((row, score, material, target_update))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(_, lhs_score, lhs_material, _), (_, rhs_score, rhs_material, _)| {
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
    for (row, score, _material, target_update) in candidates.into_iter().take(deficit) {
        let output_index = row * output_dims + material_output;
        let raw = raw_updates[output_index];
        output_gradients[output_index] += material_gain * score * (raw - target_update);
    }
}

pub(crate) fn add_material_coverage_materialization_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    step_fraction: f32,
    material_gain: f32,
    candidate_weights: &[f32],
    seed_scale: f32,
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
        || positions.len() < candidate_weights.len()
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
        let surface_weight = surface_precursor_material_weight(target, positions[row], seed_scale);
        if surface_weight <= 0.0 {
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
        let target_update = materialization_target_update_with_liveness_gate(
            material,
            target_material,
            max_material_update,
            liveness,
            predicted_liveness,
        );
        if target_update <= 0.0 || !target_update.is_finite() {
            continue;
        }
        output_gradients[output_base + material_output] += material_gain
            * candidate_weight
            * surface_weight
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

        let surface_weight = surface_precursor_material_weight(target, *position, seed_scale);
        if surface_weight <= 0.0 {
            continue;
        }

        let material = states[state_base + material_channel];
        let predicted_material = material + raw_updates[output_base + material_output];
        if predicted_material >= GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET {
            continue;
        }
        let material_target = if liveness > -1.0 || predicted_liveness > -1.0 {
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET
        } else {
            material_precursor_ceiling()
        };
        let target_update = (surface_weight * activity_weight * (material_target - material))
            .max(0.0)
            .clamp(0.0, max_material_update);
        if target_update <= 0.0 {
            continue;
        }
        let raw = raw_updates[output_base + material_output];
        output_gradients[output_base + material_output] += material_gain * (raw - target_update);
    }
}

fn surface_precursor_material_weight(
    target: &TriangleMeshTarget,
    position: [f32; 4],
    seed_scale: f32,
) -> f32 {
    let projection = target.project(position3(position));
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_opacity_frontier_coverage_threshold(seed_scale);
    frontier_material_assignment_weight(
        projection.distance,
        strict_threshold,
        soft_threshold,
        frontier_threshold,
    )
}

pub(crate) fn material_precursor_ceiling() -> f32 {
    GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 0.75
}

pub(crate) fn materialization_target_update_with_liveness_gate(
    material: f32,
    target_material: f32,
    max_material_update: f32,
    liveness: f32,
    predicted_liveness: f32,
) -> f32 {
    let target_material = if liveness > -1.0 || predicted_liveness > -1.0 {
        target_material
    } else {
        target_material.min(material_precursor_ceiling())
    };
    (target_material - material)
        .max(0.0)
        .clamp(0.0, max_material_update)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_strict_surface_materialization_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    opacity_gain: f32,
    seed_scale: f32,
    max_opacity_update: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let material_output = config.spatial_dims + material_channel;
    if material_output >= output_dims
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || opacity_gain <= 0.0
        || !opacity_gain.is_finite()
    {
        return;
    }

    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let max_update = if max_opacity_update.is_finite() && max_opacity_update > 0.0 {
        max_opacity_update
    } else {
        f32::INFINITY
    };
    let visible_gate = -1.0_f32;
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        if states[state_base + GROWTH_3D_LIVENESS_CHANNEL] <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > strict_threshold {
            continue;
        }
        let material_opacity = states[state_base + material_channel];
        if material_opacity >= GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET {
            continue;
        }
        let target_gap = (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_opacity).max(0.0);
        if target_gap <= 0.0 {
            continue;
        }
        let gate_weight = if material_opacity < visible_gate {
            1.0
        } else {
            0.5
        };
        let surface_weight = (1.0 - projection.distance / strict_threshold).clamp(0.25, 1.0);
        let target_update =
            (opacity_gain * gate_weight * surface_weight * target_gap).clamp(0.0, max_update);
        if target_update <= 0.0 {
            continue;
        }
        let output_index = row * output_dims + material_output;
        let raw = raw_updates[output_index];
        output_gradients[output_index] += raw - target_update;
    }
}
