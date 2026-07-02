#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::super::*;

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
