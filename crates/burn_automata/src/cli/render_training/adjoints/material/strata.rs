#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_surface_strata_opacity_updates(
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
    material_surface_strata_opacity_updates_weighted(
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
pub(crate) fn material_surface_strata_opacity_updates_weighted(
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

    let sample_count = coverage_samples.max(candidate_rows.len().max(512));
    let bin_count = sample_count
        .min((candidate_rows.len().saturating_mul(2)).clamp(32, 128))
        .max(1);
    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let threshold2 = threshold * threshold;
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);
    let frontier_threshold2 = frontier_threshold * frontier_threshold;
    let mut bin_samples = vec![0usize; bin_count];
    let mut bin_material_covered = vec![0usize; bin_count];
    let mut bin_candidate = vec![None::<(usize, f32)>; bin_count];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        bin_samples[bin] += 1;

        let mut best_active = None::<(usize, f32)>;
        let mut best_material_distance2 = f32::MAX;
        for &row in &candidate_rows {
            if row >= positions.len() {
                continue;
            }
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if !distance2.is_finite() {
                continue;
            }
            if best_active.is_none_or(|(_, best_distance2)| distance2 < best_distance2) {
                best_active = Some((row, distance2));
            }
            let material_opacity = states[row * config.state_dims + material_channel];
            if material_opacity > -1.0 {
                best_material_distance2 = best_material_distance2.min(distance2);
            }
        }

        if best_material_distance2 <= threshold2 {
            bin_material_covered[bin] += 1;
        }
        if let Some((row, distance2)) = best_active
            && distance2 <= frontier_threshold2
            && bin_candidate[bin].is_none_or(|(_, current_distance2)| distance2 < current_distance2)
        {
            bin_candidate[bin] = Some((row, distance2));
        }
    }

    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let target_bin_fraction = GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        .max(GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE)
        .clamp(0.0, 1.0);
    for bin in 0..bin_count {
        let samples = bin_samples[bin];
        if samples == 0 {
            continue;
        }
        let covered_fraction = bin_material_covered[bin] as f32 / samples as f32;
        let deficit = (target_bin_fraction - covered_fraction).max(0.0);
        if deficit <= 0.0 {
            continue;
        }
        let Some((row, distance2)) = bin_candidate[bin] else {
            continue;
        };
        let state_base = row * config.state_dims;
        let current_opacity = states[state_base + material_channel];
        let distance = distance2.sqrt();
        let surface_weight = frontier_material_assignment_weight(
            distance,
            threshold,
            soft_threshold,
            frontier_threshold,
        );
        let opacity_gap = (target_opacity_logit - current_opacity).max(0.0);
        let row_weight = row_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 || surface_weight <= 0.0 || opacity_gap <= 0.0 {
            continue;
        }
        updates[row] += material_opacity_gain * row_weight * deficit * surface_weight * opacity_gap;
        updates[row] = updates[row].clamp(-max_update, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_surface_strata_state_adjoint(
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
    let updates = material_surface_strata_opacity_updates(
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
