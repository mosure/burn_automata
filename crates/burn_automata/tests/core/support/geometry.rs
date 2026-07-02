use super::super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceStats {
    pub(crate) mean: f32,
    pub(crate) max: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetCoverageStats {
    pub(crate) mean: f32,
    pub(crate) max: f32,
    pub(crate) covered_fraction: f32,
}

pub(crate) fn mesh_surface_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
) -> SurfaceStats {
    surface_stats(positions, |position| {
        target
            .project([position[0], position[1], position[2]])
            .signed_distance
            .abs()
    })
}

pub(crate) fn mesh_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> SurfaceStats {
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let distance = target
            .project([position[0], position[1], position[2]])
            .signed_distance
            .abs();
        max = max.max(distance);
        sum += distance;
        count += 1;
    }
    SurfaceStats {
        mean: if count > 0 {
            sum / count as f32
        } else {
            f32::INFINITY
        },
        max: if count > 0 { max } else { f32::INFINITY },
    }
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
    let samples = samples.max(1);
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;
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
        assert!(distance.is_finite());
        sum += distance;
        max = max.max(distance);
        if distance <= threshold {
            covered += 1;
        }
    }
    TargetCoverageStats {
        mean: sum / samples as f32,
        max,
        covered_fraction: covered as f32 / samples as f32,
    }
}

fn surface_stats(positions: &[[f32; 4]], mut error: impl FnMut([f32; 4]) -> f32) -> SurfaceStats {
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;
    for position in positions {
        let value = error(*position);
        assert!(value.is_finite());
        sum += value;
        max = max.max(value);
    }
    SurfaceStats {
        mean: sum / positions.len().max(1) as f32,
        max,
    }
}
