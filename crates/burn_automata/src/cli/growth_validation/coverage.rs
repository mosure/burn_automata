#![allow(clippy::too_many_arguments)]

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

pub(crate) fn active_surface_coverage_profile(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_coverage_profile(&active_positions, target, samples, threshold, bins)
}

pub(crate) fn material_visible_surface_coverage_profile(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return surface_coverage_profile(&[], target, samples, threshold, bins);
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
    surface_coverage_profile(&visible_positions, target, samples, threshold, bins)
}

pub(crate) fn active_surface_normal_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_normal_coverage_report(&active_positions, target, samples, threshold)
}

pub(crate) fn material_visible_surface_normal_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return surface_normal_coverage_report(&[], target, samples, threshold);
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
    surface_normal_coverage_report(&visible_positions, target, samples, threshold)
}

pub(crate) fn surface_normal_coverage_report(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let samples = samples.max(1);
    let directions = normal_coverage_directions();
    let normal_bins = directions.len();
    let mut target_bin_samples = vec![0usize; normal_bins];
    let mut bin_covered = vec![0usize; normal_bins];
    let mut covered_samples = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let bin = normal_direction_bin(sample.normal, &directions);
        target_bin_samples[bin] += 1;
        let mut best_distance2 = f32::MAX;
        for position in positions {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            best_distance2 = best_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        if !positions.is_empty() && best_distance2.sqrt() <= threshold {
            covered_samples += 1;
            bin_covered[bin] += 1;
        }
    }

    let target_bins = target_bin_samples
        .iter()
        .filter(|samples| **samples > 0)
        .count();
    let covered_target_bins = target_bin_samples
        .iter()
        .zip(bin_covered.iter())
        .filter(|(samples, covered)| **samples > 0 && **covered > 0)
        .count();
    let target_bin_sample_fractions = target_bin_samples
        .iter()
        .map(|count| *count as f32 / samples as f32)
        .collect::<Vec<_>>();
    let bin_covered_fractions = target_bin_samples
        .iter()
        .zip(bin_covered.iter())
        .map(|(samples, covered)| {
            if *samples == 0 {
                0.0
            } else {
                *covered as f32 / *samples as f32
            }
        })
        .collect::<Vec<_>>();
    let active_bin_fractions = target_bin_samples
        .iter()
        .zip(bin_covered_fractions.iter())
        .filter_map(|(samples, fraction)| (*samples > 0).then_some(*fraction))
        .collect::<Vec<_>>();
    let min_bin_covered_fraction = active_bin_fractions
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bin_covered_fraction = active_bin_fractions.iter().copied().fold(0.0, f32::max);
    let mean_bin_covered_fraction = if active_bin_fractions.is_empty() {
        0.0
    } else {
        active_bin_fractions.iter().copied().sum::<f32>() / active_bin_fractions.len() as f32
    };

    SurfaceNormalCoverageReport {
        samples,
        normal_bins,
        threshold,
        target_bins,
        covered_target_bins,
        covered_target_bin_fraction: if target_bins == 0 {
            0.0
        } else {
            covered_target_bins as f32 / target_bins as f32
        },
        covered_sample_fraction: covered_samples as f32 / samples as f32,
        min_bin_covered_fraction: if min_bin_covered_fraction.is_finite() {
            min_bin_covered_fraction
        } else {
            0.0
        },
        mean_bin_covered_fraction,
        max_bin_covered_fraction,
        target_bin_sample_fractions,
        bin_covered_fractions,
    }
}

pub(crate) fn surface_coverage_profile(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let samples = samples.max(1);
    let bins = bins.max(1).min(samples);
    let mut bin_samples = vec![0usize; bins];
    let mut bin_covered = vec![0usize; bins];
    let mut assigned_counts = vec![0usize; positions.len()];
    let mut covered_assigned_counts = vec![0usize; positions.len()];
    let mut covered = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bins / samples).min(bins - 1);
        bin_samples[bin] += 1;
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if positions.is_empty() || !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_distance2.sqrt() <= threshold {
            covered += 1;
            bin_covered[bin] += 1;
            covered_assigned_counts[best_row] += 1;
        }
    }

    let bin_covered_fractions = bin_samples
        .iter()
        .zip(bin_covered.iter())
        .map(|(samples, covered)| {
            if *samples == 0 {
                0.0
            } else {
                *covered as f32 / *samples as f32
            }
        })
        .collect::<Vec<_>>();
    let empty_bins = bin_covered.iter().filter(|covered| **covered == 0).count();
    let covered_bins = bins.saturating_sub(empty_bins);
    let min_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let mean_bin_covered_fraction =
        bin_covered_fractions.iter().copied().sum::<f32>() / bins as f32;
    let assigned_particles = assigned_counts.iter().filter(|count| **count > 0).count();
    let covered_assigned_particles = covered_assigned_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let max_assigned_samples = assigned_counts.iter().copied().max().unwrap_or(0);
    let max_covered_assigned_samples = covered_assigned_counts.iter().copied().max().unwrap_or(0);

    SurfaceCoverageProfileReport {
        samples,
        bins,
        threshold,
        covered_fraction: covered as f32 / samples as f32,
        covered_bin_fraction: covered_bins as f32 / bins as f32,
        empty_bins,
        min_bin_covered_fraction: if min_bin_covered_fraction.is_finite() {
            min_bin_covered_fraction
        } else {
            0.0
        },
        mean_bin_covered_fraction,
        max_bin_covered_fraction,
        assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            assigned_particles as f32 / positions.len() as f32
        },
        covered_assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            covered_assigned_particles as f32 / positions.len() as f32
        },
        max_assigned_sample_fraction: max_assigned_samples as f32 / samples as f32,
        max_covered_assigned_sample_fraction: max_covered_assigned_samples as f32 / samples as f32,
        bin_covered_fractions,
    }
}

pub(crate) fn torus_angular_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    scale: f32,
    threshold: f32,
    ring_bins: usize,
    tube_bins: usize,
) -> TorusAngularCoverageReport {
    let ring_bins = ring_bins.max(1);
    let tube_bins = tube_bins.max(1);
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    if active_positions.is_empty() {
        return TorusAngularCoverageReport {
            ring_bins,
            tube_bins,
            threshold,
            covered_joint_bins: 0,
            covered_ring_bins: 0,
            covered_tube_bins: 0,
            joint_coverage_fraction: 0.0,
            ring_coverage_fraction: 0.0,
            tube_coverage_fraction: 0.0,
            max_ring_gap_bins: ring_bins,
            max_tube_gap_bins: tube_bins,
            mean_distance: f32::MAX,
            max_distance: f32::MAX,
        };
    }
    let mut joint_covered = vec![false; ring_bins * tube_bins];
    let mut ring_covered = vec![false; ring_bins];
    let mut tube_covered = vec![false; tube_bins];
    let mut sum_distance = 0.0_f32;
    let mut max_distance = 0.0_f32;

    for ring in 0..ring_bins {
        let theta = std::f32::consts::TAU * (ring as f32 + 0.5) / ring_bins as f32;
        let theta_cos = theta.cos();
        let theta_sin = theta.sin();
        for tube in 0..tube_bins {
            let phi = std::f32::consts::TAU * (tube as f32 + 0.5) / tube_bins as f32;
            let radial = major + minor * phi.cos();
            let sample = [radial * theta_cos, radial * theta_sin, minor * phi.sin()];
            let distance = nearest_position3_distance(sample, &active_positions);
            sum_distance += distance;
            max_distance = max_distance.max(distance);
            if distance <= threshold {
                joint_covered[ring * tube_bins + tube] = true;
                ring_covered[ring] = true;
                tube_covered[tube] = true;
            }
        }
    }

    let total_bins = ring_bins * tube_bins;
    let covered_joint_bins = joint_covered.iter().filter(|covered| **covered).count();
    let covered_ring_bins = ring_covered.iter().filter(|covered| **covered).count();
    let covered_tube_bins = tube_covered.iter().filter(|covered| **covered).count();
    TorusAngularCoverageReport {
        ring_bins,
        tube_bins,
        threshold,
        covered_joint_bins,
        covered_ring_bins,
        covered_tube_bins,
        joint_coverage_fraction: covered_joint_bins as f32 / total_bins.max(1) as f32,
        ring_coverage_fraction: covered_ring_bins as f32 / ring_bins as f32,
        tube_coverage_fraction: covered_tube_bins as f32 / tube_bins as f32,
        max_ring_gap_bins: max_circular_false_run(&ring_covered),
        max_tube_gap_bins: max_circular_false_run(&tube_covered),
        mean_distance: sum_distance / total_bins.max(1) as f32,
        max_distance,
    }
}

pub(crate) fn nearest_position3_distance(sample: [f32; 3], positions: &[[f32; 4]]) -> f32 {
    if positions.is_empty() {
        return f32::MAX;
    }
    positions
        .iter()
        .map(|position| {
            let dx = sample[0] - position[0];
            let dy = sample[1] - position[1];
            let dz = sample[2] - position[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f32::MAX, f32::min)
}

pub(crate) fn max_circular_false_run(values: &[bool]) -> usize {
    if values.is_empty() || values.iter().all(|value| *value) {
        return 0;
    }
    if values.iter().all(|value| !*value) {
        return values.len();
    }
    let mut max_run = 0usize;
    let mut run = 0usize;
    for idx in 0..values.len() * 2 {
        if values[idx % values.len()] {
            max_run = max_run.max(run);
            run = 0;
        } else {
            run += 1;
            max_run = max_run.max(run.min(values.len()));
        }
    }
    max_run.min(values.len())
}

pub(crate) fn growth_3d_extent_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dExtentReport {
    let (target_bounds_min, target_bounds_max) = target.bounds();
    let target_extent = [
        target_bounds_max[0] - target_bounds_min[0],
        target_bounds_max[1] - target_bounds_min[1],
        target_bounds_max[2] - target_bounds_min[2],
    ];
    let target_max_radius = target
        .vertices
        .iter()
        .map(|position| {
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt()
        })
        .fold(0.0_f32, f32::max);

    let mut active_bounds_min = [f32::MAX; 3];
    let mut active_bounds_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..3 {
            active_bounds_min[axis] = active_bounds_min[axis].min(position[axis]);
            active_bounds_max[axis] = active_bounds_max[axis].max(position[axis]);
        }
        final_active_max_radius = final_active_max_radius.max(
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt(),
        );
    }

    if active_count == 0 {
        active_bounds_min = [0.0; 3];
        active_bounds_max = [0.0; 3];
    }
    let final_active_extent = [
        active_bounds_max[0] - active_bounds_min[0],
        active_bounds_max[1] - active_bounds_min[1],
        active_bounds_max[2] - active_bounds_min[2],
    ];
    let axis_extent_ratio = [
        final_active_extent[0] / target_extent[0].max(1.0e-6),
        final_active_extent[1] / target_extent[1].max(1.0e-6),
        final_active_extent[2] / target_extent[2].max(1.0e-6),
    ];
    let min_axis_extent_ratio = axis_extent_ratio
        .iter()
        .copied()
        .fold(f32::MAX, f32::min)
        .min(1.0e6);
    let target_diag = (target_extent[0] * target_extent[0]
        + target_extent[1] * target_extent[1]
        + target_extent[2] * target_extent[2])
        .sqrt();
    let active_diag = (final_active_extent[0] * final_active_extent[0]
        + final_active_extent[1] * final_active_extent[1]
        + final_active_extent[2] * final_active_extent[2])
        .sqrt();

    Growth3dExtentReport {
        target_bounds_min,
        target_bounds_max,
        final_active_bounds_min: active_bounds_min,
        final_active_bounds_max: active_bounds_max,
        target_extent,
        final_active_extent,
        axis_extent_ratio,
        min_axis_extent_ratio,
        bbox_diagonal_ratio: active_diag / target_diag.max(1.0e-6),
        target_max_radius,
        final_active_max_radius,
        max_radius_ratio: final_active_max_radius / target_max_radius.max(1.0e-6),
    }
}

pub(crate) fn growth_3d_catalog_sanity_report(
    target: MeshTargetArg,
    render_loss: &MultiViewRenderLossReport,
) -> Growth3dCatalogSanityReport {
    let (max_total_loss, min_density_psnr_db, min_color_psnr_db, min_depth_psnr_db) = match target {
        MeshTargetArg::Torus => (0.90, 0.95, 16.0, 14.8),
        MeshTargetArg::Teapot => (0.85, 0.95, 18.0, 18.0),
    };
    let passed = render_loss.total_loss <= max_total_loss
        && render_loss.density_psnr_db >= min_density_psnr_db
        && render_loss.color_psnr_db >= min_color_psnr_db
        && render_loss.depth_psnr_db >= min_depth_psnr_db;
    Growth3dCatalogSanityReport {
        passed,
        max_total_loss,
        min_density_psnr_db,
        min_color_psnr_db,
        min_depth_psnr_db,
        total_loss: render_loss.total_loss,
        density_psnr_db: render_loss.density_psnr_db,
        color_psnr_db: render_loss.color_psnr_db,
        depth_psnr_db: render_loss.depth_psnr_db,
    }
}
