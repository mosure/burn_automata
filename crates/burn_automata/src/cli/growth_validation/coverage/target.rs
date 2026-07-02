use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct TargetCoverageStats {
    pub(crate) mean_distance: f32,
    pub(crate) max_distance: f32,
    pub(crate) covered_fraction: f32,
}

pub(crate) fn target_coverage_threshold(seed_scale: f32) -> f32 {
    (seed_scale.max(1.0e-4) * 0.18).max(0.04)
}

pub(crate) fn target_coverage_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    if positions.is_empty() {
        return TargetCoverageStats {
            mean_distance: f32::MAX,
            max_distance: f32::MAX,
            covered_fraction: 0.0,
        };
    }
    let samples = samples.max(1);
    let mut sum_distance = 0.0_f32;
    let mut max_distance = 0.0_f32;
    let mut covered = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_distance2 = f32::MAX;
        for position in positions {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            best_distance2 = best_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        let distance = best_distance2.sqrt();
        sum_distance += distance;
        max_distance = max_distance.max(distance);
        if distance <= threshold {
            covered += 1;
        }
    }

    TargetCoverageStats {
        mean_distance: sum_distance / samples as f32,
        max_distance,
        covered_fraction: covered as f32 / samples as f32,
    }
}

pub(crate) fn active_target_coverage_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    target_coverage_stats(&active_positions, target, samples, threshold)
}

pub(crate) fn material_visible_target_coverage_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return target_coverage_stats(&[], target, samples, threshold);
    };
    let visible_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let state_base = idx * state_dims;
            let material_opacity = states[state_base + material_channel];
            (material_opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    target_coverage_stats(&visible_positions, target, samples, threshold)
}

pub(crate) fn active_strict_surface_materialization_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dStrictSurfaceMaterializationReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return Growth3dStrictSurfaceMaterializationReport::default();
    };
    if state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len().saturating_mul(state_dims)
        || threshold <= 0.0
        || !threshold.is_finite()
    {
        return Growth3dStrictSurfaceMaterializationReport::default();
    }

    let visible_gate = -1.0_f32;
    let mut active_strict_count = 0usize;
    let mut materialized_count = 0usize;
    let mut material_sum = 0.0_f32;
    let mut margin_sum = 0.0_f32;
    let mut max_visible_margin = 0.0_f32;

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * state_dims;
        if states[state_base + GROWTH_3D_LIVENESS_CHANNEL] <= visible_gate {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > threshold {
            continue;
        }
        let material_opacity = states[state_base + material_channel];
        if !material_opacity.is_finite() {
            continue;
        }
        active_strict_count += 1;
        material_sum += material_opacity;
        if material_opacity > visible_gate {
            materialized_count += 1;
        }
        let margin = (visible_gate - material_opacity).max(0.0);
        margin_sum += margin;
        max_visible_margin = max_visible_margin.max(margin);
    }

    if active_strict_count == 0 {
        return Growth3dStrictSurfaceMaterializationReport {
            active_strict_count: 0,
            materialized_count: 0,
            materialized_fraction: 0.0,
            mean_material_opacity: f32::NEG_INFINITY,
            mean_visible_margin: f32::MAX,
            max_visible_margin: f32::MAX,
        };
    }

    Growth3dStrictSurfaceMaterializationReport {
        active_strict_count,
        materialized_count,
        materialized_fraction: materialized_count as f32 / active_strict_count as f32,
        mean_material_opacity: material_sum / active_strict_count as f32,
        mean_visible_margin: margin_sum / active_strict_count as f32,
        max_visible_margin,
    }
}
