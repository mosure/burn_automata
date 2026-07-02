#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_target_coverage_opacity_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    material_target_coverage_opacity_updates_weighted(
        config,
        target,
        positions,
        states,
        None,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_update,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_target_coverage_opacity_updates_weighted(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    row_weights: Option<&[f32]>,
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || row_weights.is_some_and(|weights| weights.len() < positions.len())
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
        || !target_opacity_logit.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let candidate_rows = positions
        .iter()
        .enumerate()
        .filter_map(|(row, _)| {
            if let Some(weights) = row_weights {
                return (weights[row] > 1.0e-3 && weights[row].is_finite()).then_some(row);
            }
            let liveness = states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
            (liveness > -1.0).then_some(row)
        })
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return updates;
    }

    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    let frontier_threshold2 = frontier_threshold * frontier_threshold;
    let sample_count = coverage_samples.max(candidate_rows.len().max(512));
    let mut assigned_weights = vec![0.0_f32; positions.len()];
    let mut assigned_counts = vec![0usize; positions.len()];
    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = candidate_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in &candidate_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() || best_distance2 > frontier_threshold2 {
            continue;
        }
        let distance = best_distance2.sqrt();
        assigned_weights[best_row] += frontier_material_assignment_weight(
            distance,
            threshold,
            soft_threshold,
            frontier_threshold,
        );
        assigned_counts[best_row] += 1;
    }

    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    for &row in &candidate_rows {
        let count = assigned_counts[row];
        if count == 0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let material_index = state_base + material_channel;
        let assignment_weight = (assigned_weights[row] / count as f32).clamp(0.0, 1.0);
        if assignment_weight <= 0.0 {
            continue;
        }
        let row_weight = row_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 {
            continue;
        }
        let material_opacity = states[material_index];
        updates[row] = (material_opacity_gain
            * row_weight
            * assignment_weight
            * (target_opacity_logit - material_opacity))
            .clamp(-max_update, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_target_coverage_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if state_adjoint.len() < states.len() {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let updates = material_target_coverage_opacity_updates(
        config,
        target,
        positions,
        states,
        material_opacity_gain,
        coverage_samples,
        seed_scale,
        target_opacity_logit,
        max_adjoint,
    );
    for (row, update) in updates.iter().copied().enumerate() {
        if update == 0.0 {
            continue;
        }
        state_adjoint[row * config.state_dims + material_channel] -= update;
    }
}
