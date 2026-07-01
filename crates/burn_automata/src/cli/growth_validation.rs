#![allow(clippy::too_many_arguments)]

use super::prelude::*;

const GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION: f32 = 0.65;
const GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE: f32 = 0.45;
const GROWTH_3D_MIN_BBOX_DIAGONAL_RATIO: f32 = 0.20;
const GROWTH_3D_MIN_AXIS_EXTENT_RATIO: f32 = 0.05;
pub(crate) const GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION: f32 = 0.75;
pub(crate) const GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE: f32 = 0.45;

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_strict_checks_report(
    position_features: bool,
    local_conditionless_lineage: bool,
    seed_coordinate_scaffold: bool,
    non_opacity_seed_abs_max: f32,
    final_opacity: Growth3dOpacityStats,
    initial_color_state: Growth3dColorStateReport,
    final_color_state: Growth3dColorStateReport,
    permutation_consistency: &Growth3dPermutationReport,
    activation: &Growth3dActivationReport,
    initial_active_surface: Growth3dSurfaceStats,
    final_active_surface: Growth3dSurfaceStats,
    final_active_surface_tail: Growth3dSurfaceTailReport,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    final_material_visible_target_coverage: TargetCoverageStats,
    final_surface_normal_coverage: &SurfaceNormalCoverageReport,
    final_material_visible_surface_normal_coverage: &SurfaceNormalCoverageReport,
    torus_angular_coverage: Option<&TorusAngularCoverageReport>,
    final_gaussian_volume: GaussianVolumeStats,
    motion: &Growth3dMotionReport,
    front: &Growth3dFrontReport,
    temporal: &Growth3dTemporalReport,
    extent: Growth3dExtentReport,
    mean_final_displacement: f32,
    seed_scale: f32,
    particle_count: usize,
    render_loss_passed: bool,
) -> Growth3dStrictChecksReport {
    let no_position_features = !position_features;
    let no_seed_coordinate_scaffold = !seed_coordinate_scaffold;
    let neutral_non_opacity_seed_state = non_opacity_seed_abs_max <= 1.0e-6;
    let sparse_active_seed =
        activation.active_seed_count > 0 && activation.active_seed_count <= particle_count / 8;
    let active_count_growth = activation.final_active_count > activation.active_seed_count * 4;
    let newly_activated_fraction = activation.newly_activated_fraction >= 0.50;
    let active_front_expanded =
        activation.final_active_max_radius > growth_3d_seed_radius(seed_scale);
    let active_extent_growth = extent.bbox_diagonal_ratio >= GROWTH_3D_MIN_BBOX_DIAGONAL_RATIO
        && extent.min_axis_extent_ratio >= GROWTH_3D_MIN_AXIS_EXTENT_RATIO;
    let nonzero_motion = motion.peak_mean_dx > 0.01;
    let sustained_motion =
        motion.active_step_fraction >= 0.50 && motion.sustained_step_fraction >= 0.25;
    let local_front_coherent = front.passed;
    let temporal_activation_progressive = temporal.progressive_activation;
    let temporal_geometry_progressive = temporal.geometry_progressive;
    let mean_displacement_growth = mean_final_displacement > growth_3d_seed_radius(seed_scale);
    let bounded_final_opacity =
        final_opacity.finite && final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT;
    let color_state_emerged = initial_color_state.available
        && final_color_state.available
        && initial_color_state.finite
        && final_color_state.finite
        && initial_color_state.active_max_abs <= 1.0e-6
        && final_color_state.active_mean_abs >= initial_color_state.active_mean_abs + 0.02
        && final_color_state.active_max_abs >= 0.05
        && final_color_state.active_channel_stddev_mean >= 0.02;
    let permutation_consistent = permutation_consistency.passed;
    let surface_mean_improved =
        final_active_surface.mean_distance < initial_active_surface.mean_distance * 0.85;
    let surface_max_bounded = final_active_surface.max_distance < GROWTH_3D_SURFACE_MAX_DISTANCE;
    let surface_tail_bounded = final_active_surface_tail.p99_distance
        < GROWTH_3D_SURFACE_MAX_DISTANCE
        && final_active_surface_tail.over_threshold_fraction <= 0.005
        && final_active_surface_tail.opacity_weighted_over_threshold_fraction <= 0.005;
    let target_coverage_mean_improved =
        final_target_coverage.mean_distance < initial_target_coverage.mean_distance * 0.85;
    let target_coverage_max_bounded = final_target_coverage.max_distance < seed_scale;
    let target_coverage_fraction = final_target_coverage.covered_fraction >= 0.60;
    let material_visible_target_coverage_fraction =
        final_material_visible_target_coverage.covered_fraction >= 0.60;
    let surface_normal_coverage = final_surface_normal_coverage.covered_target_bin_fraction
        >= GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION
        && final_surface_normal_coverage.mean_bin_covered_fraction
            >= GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE;
    let material_visible_surface_normal_coverage = final_material_visible_surface_normal_coverage
        .covered_target_bin_fraction
        >= GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION
        && final_material_visible_surface_normal_coverage.mean_bin_covered_fraction
            >= GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE;
    let torus_angular_coverage = torus_angular_coverage.is_none_or(|coverage| {
        coverage.joint_coverage_fraction >= 0.60
            && coverage.tube_coverage_fraction >= 0.75
            && coverage.max_tube_gap_bins <= coverage.tube_bins / 4
    });
    let gaussian_scale_budget = final_gaussian_volume.scale_budget_loss.is_finite()
        && final_gaussian_volume.scale_budget_loss <= ROBUST_3D_MAX_SCALE_BUDGET_LOSS
        && final_gaussian_volume.oversize_fraction <= ROBUST_3D_MAX_OVERSIZE_FRACTION;

    let checks = [
        ("no_position_features", no_position_features),
        ("local_conditionless_lineage", local_conditionless_lineage),
        ("no_seed_coordinate_scaffold", no_seed_coordinate_scaffold),
        (
            "neutral_non_opacity_seed_state",
            neutral_non_opacity_seed_state,
        ),
        ("sparse_active_seed", sparse_active_seed),
        ("active_count_growth", active_count_growth),
        ("newly_activated_fraction", newly_activated_fraction),
        ("active_front_expanded", active_front_expanded),
        ("active_extent_growth", active_extent_growth),
        ("nonzero_motion", nonzero_motion),
        ("sustained_motion", sustained_motion),
        ("local_front_coherent", local_front_coherent),
        (
            "temporal_activation_progressive",
            temporal_activation_progressive,
        ),
        (
            "temporal_geometry_progressive",
            temporal_geometry_progressive,
        ),
        ("mean_displacement_growth", mean_displacement_growth),
        ("bounded_final_opacity", bounded_final_opacity),
        ("material_visible_particles_live", true),
        ("color_state_emerged", color_state_emerged),
        ("permutation_consistent", permutation_consistent),
        ("surface_mean_improved", surface_mean_improved),
        ("surface_max_bounded", surface_max_bounded),
        ("surface_tail_bounded", surface_tail_bounded),
        (
            "target_coverage_mean_improved",
            target_coverage_mean_improved,
        ),
        ("target_coverage_max_bounded", target_coverage_max_bounded),
        ("target_coverage_fraction", target_coverage_fraction),
        (
            "material_visible_target_coverage_fraction",
            material_visible_target_coverage_fraction,
        ),
        ("surface_normal_coverage", surface_normal_coverage),
        (
            "material_visible_surface_normal_coverage",
            material_visible_surface_normal_coverage,
        ),
        ("torus_angular_coverage", torus_angular_coverage),
        ("gaussian_scale_budget", gaussian_scale_budget),
        ("render_loss_passed", render_loss_passed),
    ];
    let failure_reasons = checks
        .iter()
        .filter_map(|(name, passed)| (!*passed).then_some(*name))
        .collect::<Vec<_>>();
    let passed = failure_reasons.is_empty();

    Growth3dStrictChecksReport {
        passed,
        no_position_features,
        local_conditionless_lineage,
        no_seed_coordinate_scaffold,
        neutral_non_opacity_seed_state,
        sparse_active_seed,
        active_count_growth,
        newly_activated_fraction,
        active_front_expanded,
        active_extent_growth,
        nonzero_motion,
        sustained_motion,
        local_front_coherent,
        temporal_activation_progressive,
        temporal_geometry_progressive,
        mean_displacement_growth,
        bounded_final_opacity,
        material_visible_particles_live: true,
        color_state_emerged,
        permutation_consistent,
        surface_mean_improved,
        surface_max_bounded,
        surface_tail_bounded,
        material_visible_surface_tail_bounded: true,
        target_coverage_mean_improved,
        target_coverage_max_bounded,
        target_coverage_fraction,
        material_visible_target_coverage_fraction,
        surface_normal_coverage,
        material_visible_surface_normal_coverage,
        torus_angular_coverage,
        gaussian_scale_budget,
        render_loss_passed,
        failure_reasons,
        surface_coverage_profile: true,
        material_visible_surface_coverage_profile: true,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_strict_score_report(
    checks: &Growth3dStrictChecksReport,
    initial_active_surface: Growth3dSurfaceStats,
    final_active_surface: Growth3dSurfaceStats,
    final_active_surface_tail: Growth3dSurfaceTailReport,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    final_material_visible_target_coverage: TargetCoverageStats,
    final_surface_normal_coverage: &SurfaceNormalCoverageReport,
    final_material_visible_surface_normal_coverage: &SurfaceNormalCoverageReport,
    extent: Growth3dExtentReport,
    seed_scale: f32,
    render_loss: &MultiViewRenderLossReport,
    final_gaussian_volume: GaussianVolumeStats,
) -> Growth3dStrictScoreReport {
    let surface_mean_ratio = if initial_active_surface.mean_distance.is_finite()
        && initial_active_surface.mean_distance > 1.0e-6
    {
        final_active_surface.mean_distance / initial_active_surface.mean_distance
    } else {
        f32::INFINITY
    };
    let target_coverage_mean_ratio = if initial_target_coverage.mean_distance.is_finite()
        && initial_target_coverage.mean_distance > 1.0e-6
    {
        final_target_coverage.mean_distance / initial_target_coverage.mean_distance
    } else {
        f32::INFINITY
    };

    let hard_failures = [
        checks.no_position_features,
        checks.local_conditionless_lineage,
        checks.no_seed_coordinate_scaffold,
        checks.neutral_non_opacity_seed_state,
        checks.sparse_active_seed,
        checks.active_count_growth,
        checks.newly_activated_fraction,
        checks.active_front_expanded,
        checks.active_extent_growth,
        checks.nonzero_motion,
        checks.sustained_motion,
        checks.local_front_coherent,
        checks.temporal_activation_progressive,
        checks.temporal_geometry_progressive,
        checks.mean_displacement_growth,
        checks.bounded_final_opacity,
        checks.material_visible_particles_live,
        checks.color_state_emerged,
        checks.permutation_consistent,
        checks.surface_coverage_profile,
        checks.material_visible_surface_coverage_profile,
        checks.torus_angular_coverage,
        checks.gaussian_scale_budget,
        checks.material_visible_surface_tail_bounded,
    ]
    .into_iter()
    .filter(|passed| !passed)
    .count() as f32;
    let hard_failure_penalty = hard_failures * 10.0;
    let surface_mean_penalty = (surface_mean_ratio - 0.85).max(0.0);
    let surface_max_penalty =
        (final_active_surface.max_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let surface_tail_p99_penalty =
        (final_active_surface_tail.p99_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let surface_tail_fraction_penalty = ((final_active_surface_tail.over_threshold_fraction
        - 0.005)
        .max(0.0)
        + (final_active_surface_tail.opacity_weighted_over_threshold_fraction - 0.005).max(0.0))
        * 10.0;
    let target_coverage_mean_penalty = (target_coverage_mean_ratio - 0.85).max(0.0);
    let target_coverage_max_penalty = (final_target_coverage.max_distance - seed_scale).max(0.0);
    let target_coverage_fraction_penalty = (0.60 - final_target_coverage.covered_fraction).max(0.0);
    let material_visible_target_coverage_penalty =
        (0.60 - final_material_visible_target_coverage.covered_fraction).max(0.0);
    let active_extent_bbox_penalty =
        (GROWTH_3D_MIN_BBOX_DIAGONAL_RATIO - extent.bbox_diagonal_ratio).max(0.0);
    let active_extent_min_axis_penalty =
        (GROWTH_3D_MIN_AXIS_EXTENT_RATIO - extent.min_axis_extent_ratio).max(0.0);
    let surface_normal_bin_penalty = (GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION
        - final_surface_normal_coverage.covered_target_bin_fraction)
        .max(0.0);
    let surface_normal_mean_penalty = (GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE
        - final_surface_normal_coverage.mean_bin_covered_fraction)
        .max(0.0);
    let material_visible_surface_normal_bin_penalty = (GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION
        - final_material_visible_surface_normal_coverage.covered_target_bin_fraction)
        .max(0.0);
    let material_visible_surface_normal_mean_penalty =
        (GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE
            - final_material_visible_surface_normal_coverage.mean_bin_covered_fraction)
            .max(0.0);
    let gaussian_scale_budget_penalty =
        (final_gaussian_volume.scale_budget_loss - ROBUST_3D_MAX_SCALE_BUDGET_LOSS).max(0.0);
    let gaussian_oversize_penalty =
        (final_gaussian_volume.oversize_fraction - ROBUST_3D_MAX_OVERSIZE_FRACTION).max(0.0) * 10.0;
    let render_density_penalty = ((10.0 - render_loss.density_psnr_db).max(0.0)) / 10.0;
    let render_color_penalty = ((12.0 - render_loss.color_psnr_db).max(0.0)) / 12.0;
    let render_depth_penalty = ((14.0 - render_loss.depth_psnr_db).max(0.0)) / 14.0;
    let score = hard_failure_penalty
        + surface_mean_penalty
        + surface_max_penalty
        + surface_tail_p99_penalty
        + surface_tail_fraction_penalty
        + target_coverage_mean_penalty
        + target_coverage_max_penalty
        + target_coverage_fraction_penalty
        + material_visible_target_coverage_penalty
        + active_extent_bbox_penalty
        + active_extent_min_axis_penalty
        + surface_normal_bin_penalty
        + surface_normal_mean_penalty
        + material_visible_surface_normal_bin_penalty
        + material_visible_surface_normal_mean_penalty
        + gaussian_scale_budget_penalty
        + gaussian_oversize_penalty
        + render_density_penalty
        + render_color_penalty
        + render_depth_penalty;

    Growth3dStrictScoreReport {
        score,
        hard_failure_penalty,
        temporal_activation_schedule_error: 0.0,
        temporal_activation_schedule_penalty: 0.0,
        material_visible_inactive_fraction: 0.0,
        material_visible_inactive_fraction_penalty: 0.0,
        material_visible_max_inactive_opacity: f32::NEG_INFINITY,
        material_visible_max_inactive_opacity_penalty: 0.0,
        surface_mean_ratio,
        surface_mean_penalty,
        surface_max_distance: final_active_surface.max_distance,
        surface_max_penalty,
        surface_tail_p99_distance: final_active_surface_tail.p99_distance,
        surface_tail_p99_penalty,
        surface_tail_over_threshold_fraction: final_active_surface_tail.over_threshold_fraction,
        surface_tail_fraction_penalty,
        material_visible_surface_tail_p99_distance: final_active_surface_tail.p99_distance,
        material_visible_surface_tail_p99_penalty: 0.0,
        material_visible_surface_tail_over_threshold_fraction: final_active_surface_tail
            .over_threshold_fraction,
        material_visible_surface_tail_fraction_penalty: 0.0,
        target_coverage_mean_ratio,
        target_coverage_mean_penalty,
        target_coverage_max_distance: final_target_coverage.max_distance,
        target_coverage_max_penalty,
        target_coverage_fraction: final_target_coverage.covered_fraction,
        target_coverage_fraction_penalty,
        material_visible_target_coverage_fraction: final_material_visible_target_coverage
            .covered_fraction,
        material_visible_target_coverage_penalty,
        active_extent_bbox_ratio: extent.bbox_diagonal_ratio,
        active_extent_bbox_penalty,
        active_extent_min_axis_ratio: extent.min_axis_extent_ratio,
        active_extent_min_axis_penalty,
        surface_covered_bin_fraction: 1.0,
        surface_bin_penalty: 0.0,
        surface_mean_bin_covered_fraction: 1.0,
        surface_coverage_mean_penalty: 0.0,
        material_visible_surface_covered_bin_fraction: 1.0,
        material_visible_surface_bin_penalty: 0.0,
        material_visible_surface_mean_bin_covered_fraction: 1.0,
        material_visible_surface_mean_penalty: 0.0,
        surface_normal_covered_bin_fraction: final_surface_normal_coverage
            .covered_target_bin_fraction,
        surface_normal_bin_penalty,
        surface_normal_mean_bin_covered_fraction: final_surface_normal_coverage
            .mean_bin_covered_fraction,
        surface_normal_mean_penalty,
        material_visible_surface_normal_covered_bin_fraction:
            final_material_visible_surface_normal_coverage.covered_target_bin_fraction,
        material_visible_surface_normal_bin_penalty,
        material_visible_surface_normal_mean_bin_covered_fraction:
            final_material_visible_surface_normal_coverage.mean_bin_covered_fraction,
        material_visible_surface_normal_mean_penalty,
        gaussian_scale_budget_loss: final_gaussian_volume.scale_budget_loss,
        gaussian_scale_budget_penalty,
        gaussian_oversize_fraction: final_gaussian_volume.oversize_fraction,
        gaussian_oversize_penalty,
        render_density_psnr_db: render_loss.density_psnr_db,
        render_density_penalty,
        render_color_psnr_db: render_loss.color_psnr_db,
        render_color_penalty,
        render_depth_psnr_db: render_loss.depth_psnr_db,
        render_depth_penalty,
    }
}

pub(crate) fn growth_3d_validation_report(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut primary_cfg = cfg.clone();
    primary_cfg.extra_seeds.clear();
    let mut primary = growth_3d_validation_report_single(model_path, target_arg, primary_cfg)?;
    let mut seed_reports = Vec::with_capacity(seeds.len());
    seed_reports.push(growth_3d_robustness_seed_report(&primary));
    for seed in seeds.iter().skip(1) {
        let mut seed_cfg = cfg.clone();
        seed_cfg.seed = *seed;
        seed_cfg.extra_seeds.clear();
        let report = growth_3d_validation_report_single(model_path, target_arg, seed_cfg)?;
        seed_reports.push(growth_3d_robustness_seed_report(&report));
    }
    primary.robustness = growth_3d_robustness_report(seed_reports);
    Ok(primary)
}

pub(crate) fn growth_3d_fail_on_validation_passed(report: &CliGrowth3dValidationReport) -> bool {
    if report.robustness.seed_count > 1 {
        report.robustness.all_gate_passed
    } else {
        report.gate_passed
    }
}

pub(crate) fn growth_3d_validation_report_single(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let manifest = crate::import::load_manifest(model_path)?;
    if manifest.config.spatial_dims != 3 || manifest.config.state_dims <= 3 {
        return Err(std::io::Error::other(format!(
            "growth 3D validation requires spatial_dims=3 and state_dims>3; got spatial_dims={} state_dims={}",
            manifest.config.spatial_dims, manifest.config.state_dims
        ))
        .into());
    }
    let source = manifest.source.clone();
    let source_text = source.as_deref().unwrap_or_default();
    let local_conditionless_lineage = local_conditionless_lineage(source_text);
    let position_features = manifest.config.position_features;
    let seed_coordinate_scaffold = growth_3d_seed_has_coordinate_scaffold(cfg.seed_mode);
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let target = mesh_target_for_arg(target_arg, cfg.seed_scale);
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        update_prob: 1.0,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let rollout_steps = rollout_cfg.steps;
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );

    let mut active_seed_count = 0usize;
    let mut seed_active = Vec::with_capacity(rollout_cfg.particle_count);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }
    let non_opacity_seed_abs_max =
        growth_3d_non_scaffold_seed_abs_max(model.config.state_dims, cfg.seed_mode, &seed_states);

    let trace = run_rollout(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let final_material_opacity = growth_3d_material_opacity_stats(&trace.states, trace.state_dims);
    let final_material_liveness =
        growth_3d_material_liveness_report(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let permutation_consistency =
        growth_3d_permutation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let seed_perturbation =
        growth_3d_seed_perturbation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let initial_surface = growth_3d_surface_stats(&seed_positions, &target);
    let final_surface = growth_3d_surface_stats(&trace.positions, &target);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
    );
    let final_active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, &target);
    let initial_active_surface_tail = growth_3d_active_surface_tail_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let final_active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let initial_material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let final_material_visible_surface_tail = growth_3d_material_visible_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let target_coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let coverage_samples = cfg.particle_count.max(512);
    let initial_target_coverage = target_coverage_stats(
        &seed_positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_target_coverage = target_coverage_stats(
        &trace.positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_active_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_active_target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_material_visible_target_coverage = material_visible_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_material_visible_target_coverage = material_visible_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_active_surface_coverage_profile = active_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
        64,
    );
    let final_material_visible_surface_coverage_profile = material_visible_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
        64,
    );
    let final_active_surface_normal_coverage = active_surface_normal_coverage_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_material_visible_surface_normal_coverage =
        material_visible_surface_normal_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            &target,
            coverage_samples,
            target_coverage_threshold,
        );
    let torus_angular_coverage = (target_arg == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            target_coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let extent =
        growth_3d_extent_report(&trace.positions, &trace.states, trace.state_dims, &target);
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = coverage_samples;
    }
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, &target, render_cfg)?;
    let initial_trace = crate::RolloutTrace {
        positions: seed_positions.clone(),
        states: seed_states.clone(),
        batch_size: rollout_cfg.batch_size,
        particle_count: rollout_cfg.particle_count,
        state_dims: model.config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let initial_gaussian_volume = gaussian_volume_stats_for_trace(&initial_trace, render_cfg);
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&trace, render_cfg);
    let catalog_sanity = growth_3d_catalog_sanity_report(target_arg, &render_loss);
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let temporal = growth_3d_temporal_report(
        &model,
        &grid,
        &target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        target_coverage_threshold,
    )?;
    let front = growth_3d_front_report(
        &model,
        &grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let mut strict_checks = growth_3d_strict_checks_report(
        position_features,
        local_conditionless_lineage,
        seed_coordinate_scaffold,
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        final_material_visible_target_coverage,
        &final_active_surface_normal_coverage,
        &final_material_visible_surface_normal_coverage,
        torus_angular_coverage.as_ref(),
        final_gaussian_volume,
        &motion,
        &front,
        &temporal,
        extent,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particle_count,
        render_loss.passed,
    );
    apply_material_liveness_strict_check(&mut strict_checks, final_material_liveness);
    apply_material_visible_surface_tail_strict_check(
        &mut strict_checks,
        final_material_visible_surface_tail,
    );
    apply_surface_profile_strict_check(
        &mut strict_checks,
        &final_active_surface_coverage_profile,
        &final_material_visible_surface_coverage_profile,
    );
    let strict_passed = strict_checks.passed;
    let mut strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        final_material_visible_target_coverage,
        &final_active_surface_normal_coverage,
        &final_material_visible_surface_normal_coverage,
        extent,
        cfg.seed_scale,
        &render_loss,
        final_gaussian_volume,
    );
    apply_temporal_activation_strict_score(&mut strict_score, &temporal, rollout_steps);
    apply_material_liveness_strict_score(&mut strict_score, final_material_liveness);
    apply_material_visible_surface_tail_strict_score(
        &mut strict_score,
        final_material_visible_surface_tail,
    );
    apply_surface_profile_strict_score(
        &mut strict_score,
        &final_active_surface_coverage_profile,
        &final_material_visible_surface_coverage_profile,
    );
    let catalog_gate_passed = !position_features
        && local_conditionless_lineage
        && !seed_coordinate_scaffold
        && non_opacity_seed_abs_max <= 1.0e-6
        && final_opacity.finite
        && final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT
        && final_material_liveness.passed
        && strict_checks.color_state_emerged
        && strict_checks.permutation_consistent
        && strict_checks.gaussian_scale_budget
        && strict_checks.material_visible_surface_tail_bounded
        && strict_checks.material_visible_target_coverage_fraction
        && strict_checks.surface_coverage_profile
        && strict_checks.material_visible_surface_coverage_profile
        && strict_checks.material_visible_surface_normal_coverage
        && activation.active_seed_count > 0
        && activation.active_seed_count <= cfg.particle_count / 8
        && activation.final_active_count > activation.active_seed_count * 4
        && activation.newly_activated_fraction >= 0.50
        && activation.final_active_max_radius > growth_3d_seed_radius(cfg.seed_scale)
        && motion.peak_mean_dx > 0.01
        && motion.active_step_fraction >= 0.50
        && motion.sustained_step_fraction >= 0.25
        && front.passed
        && mean_final_displacement > growth_3d_seed_radius(cfg.seed_scale)
        && catalog_sanity.passed;
    let gate_passed = match cfg.gate {
        Growth3dValidationGateArg::Strict => strict_passed,
        Growth3dValidationGateArg::CatalogSanity => catalog_gate_passed,
    };

    Ok(CliGrowth3dValidationReport {
        target: target_arg,
        model: model_path.display().to_string(),
        source,
        position_features,
        local_conditionless_lineage,
        seed_coordinate_scaffold,
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        seed_mode: cfg.seed_mode,
        non_opacity_seed_abs_max,
        initial_color_state,
        final_color_state,
        permutation_consistency,
        seed_perturbation,
        mean_final_displacement,
        final_opacity,
        final_material_opacity,
        final_material_liveness,
        activation,
        initial_surface,
        final_surface,
        initial_active_surface,
        final_active_surface,
        initial_active_surface_tail,
        final_active_surface_tail,
        initial_material_visible_surface_tail,
        final_material_visible_surface_tail,
        target_coverage_threshold,
        initial_target_coverage,
        final_target_coverage,
        initial_active_target_coverage,
        final_active_target_coverage,
        initial_material_visible_target_coverage,
        final_material_visible_target_coverage,
        final_active_surface_coverage_profile,
        final_material_visible_surface_coverage_profile,
        final_active_surface_normal_coverage,
        final_material_visible_surface_normal_coverage,
        torus_angular_coverage,
        extent,
        motion,
        temporal,
        front,
        max_motion_per_step: motion.peak_mean_dx,
        render_loss,
        initial_gaussian_volume,
        final_gaussian_volume,
        strict_checks,
        strict_score,
        catalog_sanity,
        robustness: growth_3d_empty_robustness_report(cfg.seed),
        gate: cfg.gate,
        gate_passed,
        strict_passed,
    })
}

pub(crate) fn growth_3d_robustness_seed_report(
    report: &CliGrowth3dValidationReport,
) -> Growth3dRobustnessSeedReport {
    Growth3dRobustnessSeedReport {
        seed: report.seed,
        gate_passed: report.gate_passed,
        strict_passed: report.strict_passed,
        catalog_sanity_passed: report.catalog_sanity.passed,
        strict_score: report.strict_score.score,
        no_seed_coordinate_scaffold: report.strict_checks.no_seed_coordinate_scaffold,
        render_loss: report.render_loss.total_loss,
        density_psnr_db: report.render_loss.density_psnr_db,
        color_psnr_db: report.render_loss.color_psnr_db,
        depth_psnr_db: report.render_loss.depth_psnr_db,
        active_seed_count: report.activation.active_seed_count,
        final_active_count: report.activation.final_active_count,
        newly_activated_fraction: report.activation.newly_activated_fraction,
        active_extent_growth: report.strict_checks.active_extent_growth,
        active_extent_bbox_ratio: report.extent.bbox_diagonal_ratio,
        active_extent_min_axis_ratio: report.extent.min_axis_extent_ratio,
        final_opacity_max: report.final_opacity.max,
        material_visible_particles_live: report.final_material_liveness.passed,
        inactive_material_visible_fraction: report
            .final_material_liveness
            .inactive_material_visible_fraction,
        max_inactive_material_opacity: report.final_material_liveness.max_inactive_material_opacity,
        color_state_emerged: report.strict_checks.color_state_emerged,
        final_active_color_state_mean_abs: report.final_color_state.active_mean_abs,
        final_active_color_state_stddev_mean: report.final_color_state.active_channel_stddev_mean,
        permutation_consistent: report.permutation_consistency.passed,
        permutation_max_position_error: report.permutation_consistency.max_position_error,
        permutation_max_state_error: report.permutation_consistency.max_state_error,
        gaussian_scale_budget: report.strict_checks.gaussian_scale_budget,
        gaussian_scale_budget_loss: report.final_gaussian_volume.scale_budget_loss,
        gaussian_oversize_fraction: report.final_gaussian_volume.oversize_fraction,
        seed_perturbation_stable: report.seed_perturbation.passed,
        perturbed_newly_activated_fraction: report
            .seed_perturbation
            .perturbed_newly_activated_fraction,
        perturbed_active_count_ratio: report.seed_perturbation.active_count_ratio,
        perturbed_peak_motion_ratio: report.seed_perturbation.peak_motion_ratio,
        local_front_coherent: report.front.passed,
        front_local_newly_activated_fraction: report.front.local_newly_activated_fraction,
        front_max_nearest_previous_active_distance: report
            .front
            .max_nearest_previous_active_distance,
        temporal_activation_progressive: report.temporal.progressive_activation,
        temporal_geometry_progressive: report.temporal.geometry_progressive,
        final_active_target_coverage_fraction: report.final_active_target_coverage.covered_fraction,
        final_material_visible_target_coverage_fraction: report
            .final_material_visible_target_coverage
            .covered_fraction,
        surface_coverage_profile: report.strict_checks.surface_coverage_profile,
        final_active_surface_covered_bin_fraction: report
            .final_active_surface_coverage_profile
            .covered_bin_fraction,
        final_active_surface_mean_bin_covered_fraction: report
            .final_active_surface_coverage_profile
            .mean_bin_covered_fraction,
        material_visible_surface_coverage_profile: report
            .strict_checks
            .material_visible_surface_coverage_profile,
        final_material_visible_surface_covered_bin_fraction: report
            .final_material_visible_surface_coverage_profile
            .covered_bin_fraction,
        final_material_visible_surface_mean_bin_covered_fraction: report
            .final_material_visible_surface_coverage_profile
            .mean_bin_covered_fraction,
        surface_normal_coverage: report.strict_checks.surface_normal_coverage,
        final_active_surface_normal_covered_bin_fraction: report
            .final_active_surface_normal_coverage
            .covered_target_bin_fraction,
        final_active_surface_normal_mean_bin_covered_fraction: report
            .final_active_surface_normal_coverage
            .mean_bin_covered_fraction,
        material_visible_surface_normal_coverage: report
            .strict_checks
            .material_visible_surface_normal_coverage,
        final_material_visible_surface_normal_covered_bin_fraction: report
            .final_material_visible_surface_normal_coverage
            .covered_target_bin_fraction,
        final_material_visible_surface_normal_mean_bin_covered_fraction: report
            .final_material_visible_surface_normal_coverage
            .mean_bin_covered_fraction,
        final_active_surface_max: report.final_active_surface.max_distance,
        material_visible_surface_tail_bounded: report
            .strict_checks
            .material_visible_surface_tail_bounded,
        final_material_visible_surface_tail_p99_distance: report
            .final_material_visible_surface_tail
            .p99_distance,
        final_material_visible_surface_tail_over_threshold_fraction: report
            .final_material_visible_surface_tail
            .over_threshold_fraction,
        final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction: report
            .final_material_visible_surface_tail
            .opacity_weighted_over_threshold_fraction,
        failure_reasons: report.strict_checks.failure_reasons.clone(),
    }
}

pub(crate) fn growth_3d_robustness_report(
    seeds: Vec<Growth3dRobustnessSeedReport>,
) -> Growth3dRobustnessReport {
    let seed_count = seeds.len();
    let all_gate_passed = seed_count > 0 && seeds.iter().all(|seed| seed.gate_passed);
    let all_catalog_sanity_passed =
        seed_count > 0 && seeds.iter().all(|seed| seed.catalog_sanity_passed);
    let all_strict_passed = seed_count > 0 && seeds.iter().all(|seed| seed.strict_passed);
    let all_temporal_activation_progressive = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.temporal_activation_progressive);
    let all_temporal_geometry_progressive =
        seed_count > 0 && seeds.iter().all(|seed| seed.temporal_geometry_progressive);
    let all_local_front_coherent =
        seed_count > 0 && seeds.iter().all(|seed| seed.local_front_coherent);
    let all_no_seed_coordinate_scaffold =
        seed_count > 0 && seeds.iter().all(|seed| seed.no_seed_coordinate_scaffold);
    let all_active_extent_growth =
        seed_count > 0 && seeds.iter().all(|seed| seed.active_extent_growth);
    let all_bounded_final_opacity = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.final_opacity_max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
    let all_material_visible_particles_live = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_particles_live);
    let all_color_state_emerged =
        seed_count > 0 && seeds.iter().all(|seed| seed.color_state_emerged);
    let all_permutation_consistent =
        seed_count > 0 && seeds.iter().all(|seed| seed.permutation_consistent);
    let all_seed_perturbation_stable =
        seed_count > 0 && seeds.iter().all(|seed| seed.seed_perturbation_stable);
    let all_surface_coverage_profile =
        seed_count > 0 && seeds.iter().all(|seed| seed.surface_coverage_profile);
    let all_material_visible_surface_coverage_profile = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_coverage_profile);
    let all_surface_normal_coverage =
        seed_count > 0 && seeds.iter().all(|seed| seed.surface_normal_coverage);
    let all_material_visible_surface_normal_coverage = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_normal_coverage);
    let all_material_visible_surface_tail_bounded = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.material_visible_surface_tail_bounded);
    let worst_strict_score = seeds
        .iter()
        .map(|seed| seed.strict_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_render_loss = seeds
        .iter()
        .map(|seed| seed.render_loss)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_density_psnr_db = seeds
        .iter()
        .map(|seed| seed.density_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_color_psnr_db = seeds
        .iter()
        .map(|seed| seed.color_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_depth_psnr_db = seeds
        .iter()
        .map(|seed| seed.depth_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .min()
        .unwrap_or(0);
    let max_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .max()
        .unwrap_or(0);
    let min_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .min()
        .unwrap_or(0);
    let max_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .max()
        .unwrap_or(0);
    let min_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_active_growth_ratio = seeds
        .iter()
        .map(|seed| seed.final_active_count as f32 / seed.active_seed_count.max(1) as f32)
        .fold(f32::INFINITY, f32::min);
    let min_active_extent_bbox_ratio = seeds
        .iter()
        .map(|seed| seed.active_extent_bbox_ratio)
        .fold(f32::INFINITY, f32::min);
    let min_active_extent_min_axis_ratio = seeds
        .iter()
        .map(|seed| seed.active_extent_min_axis_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_final_opacity = seeds
        .iter()
        .map(|seed| seed.final_opacity_max)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_inactive_material_visible_fraction = seeds
        .iter()
        .map(|seed| seed.inactive_material_visible_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_inactive_material_opacity = seeds
        .iter()
        .map(|seed| seed.max_inactive_material_opacity)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_final_active_color_state_mean_abs = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_mean_abs)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_color_state_stddev_mean = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_stddev_mean)
        .fold(f32::INFINITY, f32::min);
    let max_permutation_position_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_position_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_permutation_state_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_state_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_gaussian_scale_budget_loss = seeds
        .iter()
        .map(|seed| seed.gaussian_scale_budget_loss)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_gaussian_oversize_fraction = seeds
        .iter()
        .map(|seed| seed.gaussian_oversize_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.perturbed_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_front_nearest_previous_active_distance = seeds
        .iter()
        .map(|seed| seed.front_max_nearest_previous_active_distance)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_front_local_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.front_local_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let max_final_material_visible_surface_tail_p99_distance = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_tail_p99_distance)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_final_material_visible_surface_tail_over_threshold_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_tail_over_threshold_fraction)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction = seeds
        .iter()
        .map(|seed| {
            seed.final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let min_final_active_target_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_target_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_target_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_target_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_normal_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_normal_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_surface_normal_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_surface_normal_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_normal_covered_bin_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_normal_covered_bin_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_material_visible_surface_normal_mean_bin_covered_fraction = seeds
        .iter()
        .map(|seed| seed.final_material_visible_surface_normal_mean_bin_covered_fraction)
        .fold(f32::INFINITY, f32::min);
    Growth3dRobustnessReport {
        seed_count,
        all_gate_passed,
        all_catalog_sanity_passed,
        all_strict_passed,
        all_temporal_activation_progressive,
        all_temporal_geometry_progressive,
        all_local_front_coherent,
        all_no_seed_coordinate_scaffold,
        all_active_extent_growth,
        all_bounded_final_opacity,
        all_material_visible_particles_live,
        all_color_state_emerged,
        all_permutation_consistent,
        all_seed_perturbation_stable,
        all_surface_coverage_profile,
        all_material_visible_surface_coverage_profile,
        all_surface_normal_coverage,
        all_material_visible_surface_normal_coverage,
        all_material_visible_surface_tail_bounded,
        worst_strict_score: if seed_count == 0 {
            f32::INFINITY
        } else {
            worst_strict_score
        },
        max_render_loss: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_render_loss
        },
        min_density_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_density_psnr_db
        },
        min_color_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_color_psnr_db
        },
        min_depth_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_depth_psnr_db
        },
        min_active_seed_count,
        max_active_seed_count,
        min_final_active_count,
        max_final_active_count,
        min_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_newly_activated_fraction
        },
        min_active_growth_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_growth_ratio
        },
        min_active_extent_bbox_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_extent_bbox_ratio
        },
        min_active_extent_min_axis_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_extent_min_axis_ratio
        },
        max_final_opacity: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_opacity
        },
        max_inactive_material_visible_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_inactive_material_visible_fraction
        },
        max_inactive_material_opacity: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_inactive_material_opacity
        },
        min_final_active_color_state_mean_abs: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_mean_abs
        },
        min_final_active_color_state_stddev_mean: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_stddev_mean
        },
        max_permutation_position_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_position_error
        },
        max_permutation_state_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_state_error
        },
        max_gaussian_scale_budget_loss: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_gaussian_scale_budget_loss
        },
        max_gaussian_oversize_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_gaussian_oversize_fraction
        },
        min_perturbed_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_newly_activated_fraction
        },
        min_perturbed_active_count_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_active_count_ratio
        },
        max_perturbed_active_count_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_active_count_ratio
        },
        min_perturbed_peak_motion_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_peak_motion_ratio
        },
        max_perturbed_peak_motion_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_peak_motion_ratio
        },
        max_front_nearest_previous_active_distance: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_front_nearest_previous_active_distance
        },
        min_front_local_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_front_local_newly_activated_fraction
        },
        max_final_material_visible_surface_tail_p99_distance: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_material_visible_surface_tail_p99_distance
        },
        max_final_material_visible_surface_tail_over_threshold_fraction: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_material_visible_surface_tail_over_threshold_fraction
        },
        max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction:
            if seed_count == 0 {
                f32::INFINITY
            } else {
                max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction
            },
        min_final_active_target_coverage_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_target_coverage_fraction
        },
        min_final_material_visible_target_coverage_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_target_coverage_fraction
        },
        min_final_active_surface_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_covered_bin_fraction
        },
        min_final_active_surface_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_mean_bin_covered_fraction
        },
        min_final_material_visible_surface_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_covered_bin_fraction
        },
        min_final_material_visible_surface_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_mean_bin_covered_fraction
        },
        min_final_active_surface_normal_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_normal_covered_bin_fraction
        },
        min_final_active_surface_normal_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_surface_normal_mean_bin_covered_fraction
        },
        min_final_material_visible_surface_normal_covered_bin_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_normal_covered_bin_fraction
        },
        min_final_material_visible_surface_normal_mean_bin_covered_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_material_visible_surface_normal_mean_bin_covered_fraction
        },
        seeds,
    }
}

pub(crate) fn growth_3d_empty_robustness_report(seed: u64) -> Growth3dRobustnessReport {
    growth_3d_robustness_report(vec![Growth3dRobustnessSeedReport {
        seed,
        gate_passed: false,
        strict_passed: false,
        catalog_sanity_passed: false,
        strict_score: f32::INFINITY,
        no_seed_coordinate_scaffold: false,
        render_loss: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        active_seed_count: 0,
        final_active_count: 0,
        newly_activated_fraction: 0.0,
        active_extent_growth: false,
        active_extent_bbox_ratio: 0.0,
        active_extent_min_axis_ratio: 0.0,
        final_opacity_max: f32::INFINITY,
        material_visible_particles_live: false,
        inactive_material_visible_fraction: 1.0,
        max_inactive_material_opacity: f32::INFINITY,
        color_state_emerged: false,
        final_active_color_state_mean_abs: f32::NAN,
        final_active_color_state_stddev_mean: f32::NAN,
        permutation_consistent: false,
        permutation_max_position_error: f32::INFINITY,
        permutation_max_state_error: f32::INFINITY,
        gaussian_scale_budget: false,
        gaussian_scale_budget_loss: f32::INFINITY,
        gaussian_oversize_fraction: f32::INFINITY,
        seed_perturbation_stable: false,
        perturbed_newly_activated_fraction: 0.0,
        perturbed_active_count_ratio: 0.0,
        perturbed_peak_motion_ratio: 0.0,
        local_front_coherent: false,
        front_local_newly_activated_fraction: 0.0,
        front_max_nearest_previous_active_distance: f32::INFINITY,
        temporal_activation_progressive: false,
        temporal_geometry_progressive: false,
        final_active_target_coverage_fraction: 0.0,
        final_material_visible_target_coverage_fraction: 0.0,
        surface_coverage_profile: false,
        final_active_surface_covered_bin_fraction: 0.0,
        final_active_surface_mean_bin_covered_fraction: 0.0,
        material_visible_surface_coverage_profile: false,
        final_material_visible_surface_covered_bin_fraction: 0.0,
        final_material_visible_surface_mean_bin_covered_fraction: 0.0,
        surface_normal_coverage: false,
        final_active_surface_normal_covered_bin_fraction: 0.0,
        final_active_surface_normal_mean_bin_covered_fraction: 0.0,
        material_visible_surface_normal_coverage: false,
        final_material_visible_surface_normal_covered_bin_fraction: 0.0,
        final_material_visible_surface_normal_mean_bin_covered_fraction: 0.0,
        final_active_surface_max: f32::INFINITY,
        material_visible_surface_tail_bounded: false,
        final_material_visible_surface_tail_p99_distance: f32::INFINITY,
        final_material_visible_surface_tail_over_threshold_fraction: 1.0,
        final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction: 1.0,
        failure_reasons: Vec::new(),
    }])
}

pub(crate) fn growth_3d_seed_has_coordinate_scaffold(seed_mode: ParticleSeed) -> bool {
    growth_3d_seed_writes_coordinate_scaffold(seed_mode)
}

pub(crate) fn growth_3d_non_scaffold_seed_abs_max(
    state_dims: usize,
    seed_mode: ParticleSeed,
    seed_states: &[f32],
) -> f32 {
    let material_opacity_channel = growth_3d_material_opacity_channel(state_dims);
    let allow_coordinate_scaffold = growth_3d_seed_has_coordinate_scaffold(seed_mode);
    let mut abs_max = 0.0_f32;
    for state in seed_states.chunks_exact(state_dims) {
        for (channel, value) in state.iter().enumerate() {
            if channel == GROWTH_3D_LIVENESS_CHANNEL
                || Some(channel) == material_opacity_channel
                || (allow_coordinate_scaffold && channel < 3)
            {
                continue;
            }
            abs_max = abs_max.max(value.abs());
        }
    }
    abs_max
}

pub(crate) fn growth_3d_motion_report(mean_dx: &[f32]) -> Growth3dMotionReport {
    if mean_dx.is_empty() {
        return Growth3dMotionReport {
            first_step_mean_dx: 0.0,
            peak_mean_dx: 0.0,
            peak_step: 0,
            final_step_mean_dx: 0.0,
            mean_dx: 0.0,
            late_mean_dx: 0.0,
            late_to_peak_ratio: 0.0,
            active_step_fraction: 0.0,
            sustained_step_fraction: 0.0,
        };
    }

    let first_step_mean_dx = mean_dx[0];
    let final_step_mean_dx = mean_dx[mean_dx.len() - 1];
    let mut peak_mean_dx = 0.0_f32;
    let mut peak_step = 0usize;
    let mut sum = 0.0_f32;
    for (step, value) in mean_dx.iter().copied().enumerate() {
        sum += value;
        if value > peak_mean_dx {
            peak_mean_dx = value;
            peak_step = step;
        }
    }
    let mean = sum / mean_dx.len() as f32;
    let late_start = mean_dx.len() * 3 / 4;
    let late_slice = &mean_dx[late_start..];
    let late_mean_dx = late_slice.iter().copied().sum::<f32>() / late_slice.len().max(1) as f32;
    let active_threshold = 1.0e-3;
    let sustained_threshold = (peak_mean_dx * 0.05).max(active_threshold);
    let active_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > active_threshold)
        .count();
    let sustained_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > sustained_threshold)
        .count();

    Growth3dMotionReport {
        first_step_mean_dx,
        peak_mean_dx,
        peak_step,
        final_step_mean_dx,
        mean_dx: mean,
        late_mean_dx,
        late_to_peak_ratio: if peak_mean_dx > 1.0e-8 {
            late_mean_dx / peak_mean_dx
        } else {
            0.0
        },
        active_step_fraction: active_steps as f32 / mean_dx.len() as f32,
        sustained_step_fraction: sustained_steps as f32 / mean_dx.len() as f32,
    }
}

#[derive(Clone)]
pub(crate) struct Growth3dFrontSnapshot {
    pub(crate) positions: Vec<[f32; 4]>,
    pub(crate) active: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_front_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    final_trace: &crate::RolloutTrace,
) -> Result<Growth3dFrontReport, Box<dyn std::error::Error>> {
    let max_allowed_distance = growth_3d_front_distance_threshold(rollout_cfg.seed_scale);
    let mut snapshots = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        let snapshot = if steps == 0 {
            growth_3d_front_snapshot(seed_positions, seed_states, model.config.state_dims)
        } else if steps == rollout_cfg.steps {
            growth_3d_front_snapshot(
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            growth_3d_front_snapshot(&trace.positions, &trace.states, trace.state_dims)
        };
        snapshots.push(snapshot);
    }

    let mut transition_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut local_newly_activated_count = 0usize;
    let mut finite = true;
    let mut sum_nearest = 0.0_f32;
    let mut max_nearest = 0.0_f32;

    for pair in snapshots.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if previous.positions.len() != current.positions.len()
            || previous.active.len() != current.active.len()
        {
            finite = false;
            continue;
        }
        let previous_active_positions = previous
            .positions
            .iter()
            .zip(previous.active.iter())
            .filter_map(|(position, active)| (*active).then_some(*position))
            .collect::<Vec<_>>();
        if previous_active_positions.is_empty() {
            continue;
        }
        let mut transition_newly_activated = 0usize;
        for idx in 0..current.active.len() {
            if !current.active[idx] || previous.active[idx] {
                continue;
            }
            transition_newly_activated += 1;
            newly_activated_count += 1;
            let distance =
                nearest_position_distance(current.positions[idx], &previous_active_positions);
            finite &= distance.is_finite();
            sum_nearest += distance;
            max_nearest = max_nearest.max(distance);
            if distance <= max_allowed_distance {
                local_newly_activated_count += 1;
            }
        }
        if transition_newly_activated > 0 {
            transition_count += 1;
        }
    }

    let local_newly_activated_fraction = if newly_activated_count > 0 {
        local_newly_activated_count as f32 / newly_activated_count as f32
    } else {
        0.0
    };
    let mean_nearest_previous_active_distance = if newly_activated_count > 0 {
        sum_nearest / newly_activated_count as f32
    } else {
        f32::INFINITY
    };
    let passed = finite
        && newly_activated_count > 0
        && transition_count >= 2
        && local_newly_activated_fraction >= 0.90
        && mean_nearest_previous_active_distance <= max_allowed_distance * 0.75;

    Ok(Growth3dFrontReport {
        transition_count,
        newly_activated_count,
        local_newly_activated_count,
        local_newly_activated_fraction,
        mean_nearest_previous_active_distance,
        max_nearest_previous_active_distance: if newly_activated_count > 0 {
            max_nearest
        } else {
            f32::INFINITY
        },
        max_allowed_distance,
        finite,
        passed,
    })
}

pub(crate) fn growth_3d_front_snapshot(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
) -> Growth3dFrontSnapshot {
    let active = positions
        .iter()
        .enumerate()
        .map(|(idx, _)| state_dims > 3 && states[idx * state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    Growth3dFrontSnapshot {
        positions: positions.to_vec(),
        active,
    }
}

pub(crate) fn nearest_position_distance(position: [f32; 4], candidates: &[[f32; 4]]) -> f32 {
    candidates
        .iter()
        .map(|candidate| {
            ((position[0] - candidate[0]).powi(2)
                + (position[1] - candidate[1]).powi(2)
                + (position[2] - candidate[2]).powi(2))
            .sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

pub(crate) fn growth_3d_front_distance_threshold(seed_scale: f32) -> f32 {
    growth_3d_seed_radius(seed_scale) * 2.5
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_temporal_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    seed_active: &[bool],
    active_seed_count: usize,
    final_trace: &crate::RolloutTrace,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Result<Growth3dTemporalReport, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        if steps == 0 {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                seed_positions,
                seed_states,
                model.config.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else if steps == rollout_cfg.steps {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &trace.positions,
                &trace.states,
                trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        }
    }

    let first_growth_step = samples
        .iter()
        .find(|sample| {
            sample.active_count > active_seed_count
                && sample.active_count >= active_seed_count.saturating_mul(2).max(1)
        })
        .map(|sample| sample.steps);
    let half_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.50)
        .map(|sample| sample.steps);
    let full_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.95)
        .map(|sample| sample.steps);
    let activation_span_steps =
        if let (Some(first), Some(full)) = (first_growth_step, full_activation_step) {
            full.saturating_sub(first)
        } else {
            0
        };
    let progressive_activation = match (
        first_growth_step,
        half_activation_step,
        full_activation_step,
    ) {
        (Some(first), Some(half), Some(full)) => {
            first < half && half < full && activation_span_steps >= rollout_cfg.steps / 4
        }
        _ => false,
    };
    let (surface_mean_ratio, target_coverage_mean_ratio, target_coverage_fraction_delta) =
        match (samples.first(), samples.last()) {
            (Some(initial), Some(final_sample)) => {
                let surface_mean_ratio = if initial.active_surface.mean_distance.is_finite()
                    && initial.active_surface.mean_distance > 1.0e-6
                {
                    final_sample.active_surface.mean_distance / initial.active_surface.mean_distance
                } else {
                    f32::INFINITY
                };
                let target_coverage_mean_ratio =
                    if initial.target_coverage.mean_distance.is_finite()
                        && initial.target_coverage.mean_distance > 1.0e-6
                    {
                        final_sample.target_coverage.mean_distance
                            / initial.target_coverage.mean_distance
                    } else {
                        f32::INFINITY
                    };
                let target_coverage_fraction_delta = final_sample.target_coverage.covered_fraction
                    - initial.target_coverage.covered_fraction;
                (
                    surface_mean_ratio,
                    target_coverage_mean_ratio,
                    target_coverage_fraction_delta,
                )
            }
            _ => (f32::INFINITY, f32::INFINITY, 0.0),
        };
    let geometry_progressive = target_coverage_mean_ratio < 0.85
        && target_coverage_fraction_delta >= 0.10
        && surface_mean_ratio < 0.95;

    Ok(Growth3dTemporalReport {
        samples,
        first_growth_step,
        half_activation_step,
        full_activation_step,
        activation_span_steps,
        progressive_activation,
        surface_mean_ratio,
        target_coverage_mean_ratio,
        target_coverage_fraction_delta,
        geometry_progressive,
    })
}

pub(crate) fn growth_3d_temporal_sample_steps(steps: usize) -> Vec<usize> {
    let mut samples = vec![0, steps];
    let mut step = 1usize;
    while step < steps {
        samples.push(step);
        step = step.saturating_mul(2);
        if step == 0 {
            break;
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn growth_3d_temporal_sample_report(
    steps: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    seed_positions: &[[f32; 4]],
    seed_active: &[bool],
    target: &TriangleMeshTarget,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Growth3dTemporalSampleReport {
    let mut active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut active_radius_sum = 0.0_f32;
    let mut active_max_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity > -1.0 {
            active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            active_radius_sum += radius;
            active_max_radius = active_max_radius.max(radius);
        }
    }

    Growth3dTemporalSampleReport {
        steps,
        active_count,
        active_fraction: active_count as f32 / positions.len().max(1) as f32,
        newly_activated_count,
        final_active_mean_radius: if active_count > 0 {
            active_radius_sum / active_count as f32
        } else {
            0.0
        },
        final_active_max_radius: active_max_radius,
        mean_displacement: growth_3d_mean_displacement(seed_positions, positions),
        active_surface: growth_3d_active_surface_stats(positions, states, state_dims, target),
        target_coverage: target_coverage_stats(
            positions,
            target,
            coverage_samples,
            coverage_threshold,
        ),
    }
}

pub(crate) fn growth_3d_activation_report(
    trace: &crate::RolloutTrace,
    seed_active: &[bool],
    active_seed_count: usize,
) -> Growth3dActivationReport {
    let mut final_active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut final_active_radius_sum = 0.0_f32;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in trace.positions.iter().enumerate() {
        let opacity = trace.states[idx * trace.state_dims + 3];
        if opacity > -1.0 {
            final_active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            final_active_radius_sum += radius;
            final_active_max_radius = final_active_max_radius.max(radius);
        }
    }
    let inactive_seed_count = trace.particle_count.saturating_sub(active_seed_count);
    Growth3dActivationReport {
        active_seed_count,
        inactive_seed_count,
        final_active_count,
        newly_activated_count,
        newly_activated_fraction: newly_activated_count as f32 / inactive_seed_count.max(1) as f32,
        final_active_mean_radius: final_active_radius_sum / final_active_count.max(1) as f32,
        final_active_max_radius,
    }
}

pub(crate) fn growth_3d_opacity_stats(states: &[f32], state_dims: usize) -> Growth3dOpacityStats {
    growth_3d_channel_opacity_stats(states, state_dims, GROWTH_3D_LIVENESS_CHANNEL)
}

pub(crate) fn growth_3d_material_opacity_stats(
    states: &[f32],
    state_dims: usize,
) -> Growth3dOpacityStats {
    let Some(channel) = growth_3d_material_opacity_channel(state_dims) else {
        return growth_3d_channel_opacity_stats(states, state_dims, GROWTH_3D_LIVENESS_CHANNEL);
    };
    growth_3d_channel_opacity_stats(states, state_dims, channel)
}

pub(crate) fn growth_3d_material_liveness_report(
    states: &[f32],
    state_dims: usize,
) -> Growth3dMaterialLivenessReport {
    let threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return Growth3dMaterialLivenessReport {
            material_visible_count: 0,
            inactive_material_visible_count: 0,
            inactive_material_visible_fraction: 0.0,
            inactive_material_logit_threshold: threshold,
            max_inactive_material_opacity: f32::NEG_INFINITY,
            passed: true,
        };
    };
    if state_dims <= GROWTH_3D_LIVENESS_CHANNEL || states.is_empty() {
        return Growth3dMaterialLivenessReport {
            material_visible_count: 0,
            inactive_material_visible_count: 0,
            inactive_material_visible_fraction: 0.0,
            inactive_material_logit_threshold: threshold,
            max_inactive_material_opacity: f32::NEG_INFINITY,
            passed: true,
        };
    }

    let mut material_visible_count = 0usize;
    let mut inactive_material_visible_count = 0usize;
    let mut max_inactive_material_opacity = f32::NEG_INFINITY;
    for state in states.chunks_exact(state_dims) {
        let material_opacity = state[material_channel];
        let liveness = state[GROWTH_3D_LIVENESS_CHANNEL];
        if material_opacity > threshold {
            material_visible_count += 1;
            if liveness <= -1.0 {
                inactive_material_visible_count += 1;
                max_inactive_material_opacity = max_inactive_material_opacity.max(material_opacity);
            }
        }
    }
    let inactive_material_visible_fraction =
        inactive_material_visible_count as f32 / material_visible_count.max(1) as f32;
    Growth3dMaterialLivenessReport {
        material_visible_count,
        inactive_material_visible_count,
        inactive_material_visible_fraction,
        inactive_material_logit_threshold: threshold,
        max_inactive_material_opacity,
        passed: inactive_material_visible_count == 0,
    }
}

pub(crate) fn apply_material_liveness_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    checks.material_visible_particles_live = material_liveness.passed;
    if !material_liveness.passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_particles_live")
    {
        checks
            .failure_reasons
            .push("material_visible_particles_live");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_liveness_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    let inactive_fraction_penalty = material_liveness.inactive_material_visible_fraction * 10.0;
    let max_inactive_opacity = material_liveness.max_inactive_material_opacity;
    let max_inactive_opacity_penalty = if max_inactive_opacity.is_finite() {
        ((max_inactive_opacity - material_liveness.inactive_material_logit_threshold).max(0.0))
            / 10.0
    } else {
        0.0
    };
    score.material_visible_inactive_fraction = material_liveness.inactive_material_visible_fraction;
    score.material_visible_inactive_fraction_penalty = inactive_fraction_penalty;
    score.material_visible_max_inactive_opacity = max_inactive_opacity;
    score.material_visible_max_inactive_opacity_penalty = max_inactive_opacity_penalty;
    score.score += inactive_fraction_penalty + max_inactive_opacity_penalty;
}

pub(crate) fn apply_temporal_activation_strict_score(
    score: &mut Growth3dStrictScoreReport,
    temporal: &Growth3dTemporalReport,
    rollout_steps: usize,
) {
    let schedule_error = temporal_activation_schedule_error(temporal, rollout_steps);
    let penalty = schedule_error * TEMPORAL_ACTIVATION_SCORE_WEIGHT;
    score.temporal_activation_schedule_error = schedule_error;
    score.temporal_activation_schedule_penalty = penalty;
    score.score += penalty;
}

pub(crate) fn apply_material_visible_surface_tail_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let passed = material_visible_surface_tail.p99_distance < GROWTH_3D_SURFACE_MAX_DISTANCE
        && material_visible_surface_tail.over_threshold_fraction <= 0.005
        && material_visible_surface_tail.opacity_weighted_over_threshold_fraction <= 0.005;
    checks.material_visible_surface_tail_bounded = passed;
    if !passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_tail_bounded")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_tail_bounded");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_visible_surface_tail_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let p99_penalty =
        (material_visible_surface_tail.p99_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let fraction_penalty = ((material_visible_surface_tail.over_threshold_fraction - 0.005)
        .max(0.0)
        + (material_visible_surface_tail.opacity_weighted_over_threshold_fraction - 0.005)
            .max(0.0))
        * 10.0;
    score.material_visible_surface_tail_p99_distance = material_visible_surface_tail.p99_distance;
    score.material_visible_surface_tail_p99_penalty = p99_penalty;
    score.material_visible_surface_tail_over_threshold_fraction =
        material_visible_surface_tail.over_threshold_fraction;
    score.material_visible_surface_tail_fraction_penalty = fraction_penalty;
    score.score += p99_penalty + fraction_penalty;
}

pub(crate) fn apply_surface_profile_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_passed = surface_profile_passes_strict_coverage(active_profile);
    let material_visible_passed = surface_profile_passes_strict_coverage(material_visible_profile);
    checks.surface_coverage_profile = active_passed;
    checks.material_visible_surface_coverage_profile = material_visible_passed;
    if !active_passed && !checks.failure_reasons.contains(&"surface_coverage_profile") {
        checks.failure_reasons.push("surface_coverage_profile");
    }
    if !material_visible_passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_coverage_profile")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_coverage_profile");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn surface_profile_passes_strict_coverage(
    profile: &SurfaceCoverageProfileReport,
) -> bool {
    profile.covered_bin_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        && profile.mean_bin_covered_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
}

pub(crate) fn apply_surface_profile_strict_score(
    score: &mut Growth3dStrictScoreReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_bin_penalty =
        (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION - active_profile.covered_bin_fraction).max(0.0);
    let active_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - active_profile.mean_bin_covered_fraction)
        .max(0.0);
    let material_bin_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        - material_visible_profile.covered_bin_fraction)
        .max(0.0);
    let material_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - material_visible_profile.mean_bin_covered_fraction)
        .max(0.0);
    score.surface_covered_bin_fraction = active_profile.covered_bin_fraction;
    score.surface_bin_penalty = active_bin_penalty;
    score.surface_mean_bin_covered_fraction = active_profile.mean_bin_covered_fraction;
    score.surface_coverage_mean_penalty = active_mean_penalty;
    score.material_visible_surface_covered_bin_fraction =
        material_visible_profile.covered_bin_fraction;
    score.material_visible_surface_bin_penalty = material_bin_penalty;
    score.material_visible_surface_mean_bin_covered_fraction =
        material_visible_profile.mean_bin_covered_fraction;
    score.material_visible_surface_mean_penalty = material_mean_penalty;
    score.score +=
        active_bin_penalty + active_mean_penalty + material_bin_penalty + material_mean_penalty;
}

pub(crate) fn growth_3d_channel_opacity_stats(
    states: &[f32],
    state_dims: usize,
    channel: usize,
) -> Growth3dOpacityStats {
    if state_dims <= channel || states.is_empty() {
        return Growth3dOpacityStats {
            finite: false,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: f32::NAN,
            active_min: f32::INFINITY,
            active_max: f32::NEG_INFINITY,
            active_mean: f32::NAN,
            active_count: 0,
            max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
        };
    }

    let mut finite = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    let mut active_min = f32::INFINITY;
    let mut active_max = f32::NEG_INFINITY;
    let mut active_sum = 0.0_f32;
    let mut active_count = 0usize;
    for state in states.chunks_exact(state_dims) {
        let opacity = state[channel];
        finite &= opacity.is_finite();
        min = min.min(opacity);
        max = max.max(opacity);
        sum += opacity;
        count += 1;
        if opacity > -1.0 {
            active_min = active_min.min(opacity);
            active_max = active_max.max(opacity);
            active_sum += opacity;
            active_count += 1;
        }
    }

    Growth3dOpacityStats {
        finite,
        min,
        max,
        mean: sum / count.max(1) as f32,
        active_min: if active_count == 0 {
            f32::INFINITY
        } else {
            active_min
        },
        active_max: if active_count == 0 {
            f32::NEG_INFINITY
        } else {
            active_max
        },
        active_mean: if active_count == 0 {
            f32::NAN
        } else {
            active_sum / active_count as f32
        },
        active_count,
        max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
    }
}

pub(crate) fn growth_3d_color_state_report(
    states: &[f32],
    state_dims: usize,
) -> Growth3dColorStateReport {
    if state_dims < 6 || states.is_empty() {
        return Growth3dColorStateReport {
            available: false,
            finite: false,
            count: 0,
            active_count: 0,
            mean_abs: f32::NAN,
            max_abs: f32::NAN,
            active_mean_abs: f32::NAN,
            active_max_abs: f32::NAN,
            active_channel_stddev: [f32::NAN; 3],
            active_channel_stddev_mean: f32::NAN,
        };
    }

    let tail = state_dims - 3;
    let mut finite = true;
    let mut count = 0usize;
    let mut active_count = 0usize;
    let mut sum_abs = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut active_sum_abs = 0.0_f32;
    let mut active_max_abs = 0.0_f32;
    let mut active_sum = [0.0_f32; 3];
    let mut active_sum_sq = [0.0_f32; 3];

    for state in states.chunks_exact(state_dims) {
        count += 1;
        let mut particle_max_abs = 0.0_f32;
        for channel in 0..3 {
            let value = state[tail + channel];
            finite &= value.is_finite();
            particle_max_abs = particle_max_abs.max(value.abs());
        }
        sum_abs += particle_max_abs;
        max_abs = max_abs.max(particle_max_abs);

        if state[3] > -1.0 {
            active_count += 1;
            active_sum_abs += particle_max_abs;
            active_max_abs = active_max_abs.max(particle_max_abs);
            for channel in 0..3 {
                let value = state[tail + channel];
                active_sum[channel] += value;
                active_sum_sq[channel] += value * value;
            }
        }
    }

    let mut active_channel_stddev = [f32::NAN; 3];
    if active_count > 0 {
        for channel in 0..3 {
            let mean = active_sum[channel] / active_count as f32;
            let variance = (active_sum_sq[channel] / active_count as f32 - mean * mean).max(0.0);
            active_channel_stddev[channel] = variance.sqrt();
        }
    }
    let active_channel_stddev_mean = if active_count > 0 {
        active_channel_stddev.iter().sum::<f32>() / 3.0
    } else {
        f32::NAN
    };

    Growth3dColorStateReport {
        available: true,
        finite,
        count,
        active_count,
        mean_abs: sum_abs / count.max(1) as f32,
        max_abs,
        active_mean_abs: if active_count > 0 {
            active_sum_abs / active_count as f32
        } else {
            f32::NAN
        },
        active_max_abs: if active_count > 0 {
            active_max_abs
        } else {
            f32::NAN
        },
        active_channel_stddev,
        active_channel_stddev_mean,
    }
}

pub(crate) fn growth_3d_permutation_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dPermutationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(2, 256);
    let steps = cfg.steps.min(8);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut order = (0..particle_count).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x9a55_19e3_7ac3);
    order.shuffle(&mut rng);

    let mut shuffled_positions = vec![[0.0; 4]; particle_count];
    let mut shuffled_states = vec![0.0; states.len()];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        shuffled_positions[shuffled_idx] = positions[source_idx];
        let src = source_idx * model.config.state_dims;
        let dst = shuffled_idx * model.config.state_dims;
        shuffled_states[dst..dst + model.config.state_dims]
            .copy_from_slice(&states[src..src + model.config.state_dims]);
    }

    let shuffled = run_rollout_from_state(
        model,
        grid,
        shuffled_positions,
        shuffled_states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut inverse_order = vec![0usize; particle_count];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        inverse_order[source_idx] = shuffled_idx;
    }

    let mut max_position_error = 0.0_f32;
    let mut sum_position_error = 0.0_f32;
    let mut max_state_error = 0.0_f32;
    let mut sum_state_error = 0.0_f32;
    let mut state_count = 0usize;

    for (source_idx, &shuffled_idx) in inverse_order.iter().enumerate() {
        let base_position = base.positions[source_idx];
        let shuffled_position = shuffled.positions[shuffled_idx];
        let position_error = ((base_position[0] - shuffled_position[0]).powi(2)
            + (base_position[1] - shuffled_position[1]).powi(2)
            + (base_position[2] - shuffled_position[2]).powi(2))
        .sqrt();
        max_position_error = max_position_error.max(position_error);
        sum_position_error += position_error;

        let base_state = source_idx * model.config.state_dims;
        let shuffled_state = shuffled_idx * model.config.state_dims;
        for channel in 0..model.config.state_dims {
            let state_error = (base.states[base_state + channel]
                - shuffled.states[shuffled_state + channel])
                .abs();
            max_state_error = max_state_error.max(state_error);
            sum_state_error += state_error;
            state_count += 1;
        }
    }

    let mean_position_error = sum_position_error / particle_count.max(1) as f32;
    let mean_state_error = sum_state_error / state_count.max(1) as f32;
    let passed = max_position_error <= 1.0e-3 && max_state_error <= 1.0e-3;

    Ok(Growth3dPermutationReport {
        particle_count,
        steps,
        max_position_error,
        mean_position_error,
        max_state_error,
        mean_state_error,
        passed,
    })
}

pub(crate) fn growth_3d_seed_perturbation_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dSeedPerturbationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(32, 512);
    let steps = cfg.steps.clamp(1, 32);
    let jitter_radius = (growth_3d_seed_radius(cfg.seed_scale) * 0.10).max(cfg.seed_scale * 0.002);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let mut seed_active = Vec::with_capacity(particle_count);
    let mut active_seed_count = 0usize;
    for state in states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }

    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut perturbed_positions = positions;
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x005e_ed93_7d3d);
    for position in &mut perturbed_positions {
        for value in position.iter_mut().take(3) {
            *value += rng.random_range(-jitter_radius..=jitter_radius);
        }
    }
    let perturbed = run_rollout_from_state(
        model,
        grid,
        perturbed_positions,
        states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let base_activation = growth_3d_activation_report(&base, &seed_active, active_seed_count);
    let perturbed_activation =
        growth_3d_activation_report(&perturbed, &seed_active, active_seed_count);
    let base_motion = growth_3d_motion_report(&base.mean_dx);
    let perturbed_motion = growth_3d_motion_report(&perturbed.mean_dx);
    let base_color = growth_3d_color_state_report(&base.states, base.state_dims);
    let perturbed_color = growth_3d_color_state_report(&perturbed.states, perturbed.state_dims);

    let active_count_ratio = finite_ratio(
        perturbed_activation.final_active_count as f32,
        base_activation.final_active_count.max(1) as f32,
    );
    let final_active_max_radius_ratio = finite_ratio(
        perturbed_activation.final_active_max_radius,
        base_activation.final_active_max_radius,
    );
    let peak_motion_ratio = finite_ratio(perturbed_motion.peak_mean_dx, base_motion.peak_mean_dx);
    let color_state_mean_abs_ratio =
        finite_ratio(perturbed_color.active_mean_abs, base_color.active_mean_abs);

    let base_growth = base_activation.final_active_count > active_seed_count.max(1) * 2
        && base_activation.newly_activated_fraction >= 0.25
        && base_motion.peak_mean_dx > 1.0e-3;
    let perturbed_growth = perturbed_activation.final_active_count > active_seed_count.max(1) * 2
        && perturbed_activation.newly_activated_fraction >= 0.25
        && perturbed_motion.peak_mean_dx > 1.0e-3;
    let comparable_growth = (0.50..=2.00).contains(&active_count_ratio)
        && (0.50..=2.00).contains(&final_active_max_radius_ratio)
        && (0.25..=4.00).contains(&peak_motion_ratio);
    let passed = base_growth && perturbed_growth && comparable_growth;

    Ok(Growth3dSeedPerturbationReport {
        particle_count,
        steps,
        jitter_radius,
        seed: cfg.seed,
        active_seed_count,
        base_final_active_count: base_activation.final_active_count,
        perturbed_final_active_count: perturbed_activation.final_active_count,
        active_count_ratio,
        base_newly_activated_fraction: base_activation.newly_activated_fraction,
        perturbed_newly_activated_fraction: perturbed_activation.newly_activated_fraction,
        base_final_active_max_radius: base_activation.final_active_max_radius,
        perturbed_final_active_max_radius: perturbed_activation.final_active_max_radius,
        final_active_max_radius_ratio,
        base_peak_mean_dx: base_motion.peak_mean_dx,
        perturbed_peak_mean_dx: perturbed_motion.peak_mean_dx,
        peak_motion_ratio,
        base_color_state_mean_abs: base_color.active_mean_abs,
        perturbed_color_state_mean_abs: perturbed_color.active_mean_abs,
        color_state_mean_abs_ratio,
        passed,
    })
}

pub(crate) fn finite_ratio(numerator: f32, denominator: f32) -> f32 {
    if !numerator.is_finite() || !denominator.is_finite() {
        return f32::NAN;
    }
    if denominator.abs() <= 1.0e-8 {
        if numerator.abs() <= 1.0e-8 {
            1.0
        } else if numerator.is_sign_positive() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    } else {
        numerator / denominator
    }
}

pub(crate) fn run_rollout_from_state(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    mut positions: Vec<[f32; 4]>,
    mut states: Vec<f32>,
    batch_size: usize,
    particle_count: usize,
    steps: usize,
    dt: f32,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    let mut mean_dx = Vec::with_capacity(steps);
    for _ in 0..steps {
        let step = model.step_cpu(
            &positions,
            &states,
            batch_size,
            particle_count,
            grid,
            dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok(crate::RolloutTrace {
        positions,
        states,
        batch_size,
        particle_count,
        state_dims: model.config.state_dims,
        steps,
        mean_dx,
    })
}

pub(crate) fn growth_3d_surface_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    for position in positions {
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
    }
    Growth3dSurfaceStats {
        mean_distance: sum_distance / positions.len().max(1) as f32,
        max_distance,
    }
}

pub(crate) fn growth_3d_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
        count += 1;
    }
    Growth3dSurfaceStats {
        mean_distance: if count > 0 {
            sum_distance / count as f32
        } else {
            f32::INFINITY
        },
        max_distance: if count > 0 {
            max_distance
        } else {
            f32::INFINITY
        },
    }
}

pub(crate) fn growth_3d_active_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    if state_dims > 3 {
        for (idx, position) in positions.iter().enumerate() {
            let opacity_logit = states[idx * state_dims + 3];
            if opacity_logit <= -1.0 {
                continue;
            }
            let projection = target.project([position[0], position[1], position[2]]);
            let distance = projection.distance;
            let weight = sigmoid_unit(opacity_logit);
            max_distance = max_distance.max(distance);
            if distance >= threshold {
                over_threshold_count += 1;
                weighted_over_threshold_sum += weight;
            }
            weighted_sum += distance * weight;
            weight_sum += weight;
            distances.push(distance);
        }
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn growth_3d_material_visible_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return empty_growth_3d_surface_tail_report(threshold);
    };
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let material_logit = states[idx * state_dims + material_channel];
        if material_logit <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        let distance = projection.distance;
        let weight = sigmoid_unit(material_logit);
        max_distance = max_distance.max(distance);
        if distance >= threshold {
            over_threshold_count += 1;
            weighted_over_threshold_sum += weight;
        }
        weighted_sum += distance * weight;
        weight_sum += weight;
        distances.push(distance);
    }

    if distances.is_empty() {
        return empty_growth_3d_surface_tail_report(threshold);
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

pub(crate) fn empty_growth_3d_surface_tail_report(threshold: f32) -> Growth3dSurfaceTailReport {
    Growth3dSurfaceTailReport {
        count: 0,
        threshold,
        p95_distance: f32::INFINITY,
        p99_distance: f32::INFINITY,
        max_distance: f32::INFINITY,
        over_threshold_count: 0,
        over_threshold_fraction: 0.0,
        opacity_weighted_mean_distance: f32::INFINITY,
        opacity_weighted_over_threshold_fraction: 0.0,
    }
}

pub(crate) fn percentile_from_sorted(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() as f32 * clamped).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

pub(crate) fn sigmoid_unit(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn sigmoid_unit_derivative(value: f32) -> f32 {
    let sigmoid = sigmoid_unit(value);
    sigmoid * (1.0 - sigmoid)
}

pub(crate) fn growth_3d_mean_displacement(
    initial: &[[f32; 4]],
    final_positions: &[[f32; 4]],
) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}

pub(crate) fn mesh_rollout_report_for_cases(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cases: &[MeshRolloutCaseConfig],
) -> Result<MeshRolloutReport, Box<dyn std::error::Error>> {
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_mean_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_mean_surface_distance = 0.0_f32;
    let mut max_target_coverage_distance = 0.0_f32;
    let mut sum_mean_target_coverage_distance = 0.0_f32;
    let mut min_target_coverage_fraction = 1.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_mean_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = mesh_rollout_case_report(&trace, target, *case);
        max_initial_surface_distance =
            max_initial_surface_distance.max(report.max_initial_surface_distance);
        sum_mean_initial_surface_distance += report.mean_initial_surface_distance;
        max_surface_distance = max_surface_distance.max(report.max_surface_distance);
        sum_mean_surface_distance += report.mean_surface_distance;
        max_target_coverage_distance =
            max_target_coverage_distance.max(report.max_target_coverage_distance);
        sum_mean_target_coverage_distance += report.mean_target_coverage_distance;
        min_target_coverage_fraction =
            min_target_coverage_fraction.min(report.target_coverage_fraction);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        sum_mean_color_target_error += report.mean_color_target_error;
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);

        let case_passed = report.finite
            && report.max_initial_surface_distance >= 0.08
            && report.first_motion_per_step >= 1.0e-3
            && report.max_motion_per_step >= 1.0e-3
            && report.mean_surface_improvement_ratio >= 0.15
            && report.max_surface_distance <= 0.36
            && report.mean_surface_distance <= 0.16
            && report.mean_target_coverage_distance <= 0.20
            && report.max_target_coverage_distance <= 0.72
            && report.target_coverage_fraction >= 0.60
            && report.max_color_target_error <= 0.42
            && report.mean_color_target_error <= 0.16
            && report.max_opacity_target_error <= 2.0e-2
            && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT;
        passed &= case_passed;
        case_reports.push(report);
    }

    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }
    Ok(MeshRolloutReport {
        passed,
        max_initial_surface_distance,
        mean_initial_surface_distance: sum_mean_initial_surface_distance
            / cases.len().max(1) as f32,
        max_surface_distance,
        mean_surface_distance: sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement: sum_mean_initial_surface_distance / cases.len().max(1) as f32
            - sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement_ratio: if sum_mean_initial_surface_distance > 0.0 {
            1.0 - sum_mean_surface_distance / sum_mean_initial_surface_distance
        } else {
            0.0
        },
        max_target_coverage_distance,
        mean_target_coverage_distance: sum_mean_target_coverage_distance
            / cases.len().max(1) as f32,
        min_target_coverage_fraction,
        max_color_target_error,
        mean_color_target_error: sum_mean_color_target_error / cases.len().max(1) as f32,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

pub(crate) fn mesh_rollout_case_report(
    trace: &crate::RolloutTrace,
    target: &TriangleMeshTarget,
    case: MeshRolloutCaseConfig,
) -> MeshRolloutCaseReport {
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        case.seed_scale,
    );
    let expected_final_opacity_logit = UV_TORUS_FIELD_OPACITY_TARGET;
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_surface_distance = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let initial_projection = target.project([
            initial_position[0],
            initial_position[1],
            initial_position[2],
        ]);
        max_initial_surface_distance =
            max_initial_surface_distance.max(initial_projection.distance);
        sum_initial_surface_distance += initial_projection.distance;

        let projection = target.project([position[0], position[1], position[2]]);
        max_surface_distance = max_surface_distance.max(projection.distance);
        sum_surface_distance += projection.distance;

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = projection.color;
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());
    let mean_initial_surface_distance =
        sum_initial_surface_distance / trace.positions.len().max(1) as f32;
    let mean_surface_distance = sum_surface_distance / trace.positions.len().max(1) as f32;
    let coverage_threshold = target_coverage_threshold(case.seed_scale);
    let coverage = target_coverage_stats(
        &trace.positions,
        target,
        trace.particle_count.max(512),
        coverage_threshold,
    );

    MeshRolloutCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        max_initial_surface_distance,
        mean_initial_surface_distance,
        max_surface_distance,
        mean_surface_distance,
        mean_surface_improvement: mean_initial_surface_distance - mean_surface_distance,
        mean_surface_improvement_ratio: if mean_initial_surface_distance > 0.0 {
            1.0 - mean_surface_distance / mean_initial_surface_distance
        } else {
            0.0
        },
        target_coverage_threshold: coverage_threshold,
        max_target_coverage_distance: coverage.max_distance,
        mean_target_coverage_distance: coverage.mean_distance,
        target_coverage_fraction: coverage.covered_fraction,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}

pub(crate) fn torus_robustness_report(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    torus_robustness_report_for_cases(model, grid, TORUS_ROBUSTNESS_CASES)
}

pub(crate) fn torus_robustness_report_for_cases(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cases: &[TorusRobustnessCaseConfig],
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    let opacity_update_index = model.config.spatial_dims + 3;
    let trained_opacity_delta = model.weights.b2[opacity_update_index];
    let field_mode = model.config.position_features;
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_target_position_error = 0.0_f32;
    let mut sum_mean_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = torus_robustness_case_report(&trace, *case);
        max_target_position_error = max_target_position_error.max(report.max_target_position_error);
        sum_mean_target_position_error += report.mean_target_position_error;
        max_torus_surface_error = max_torus_surface_error.max(report.max_torus_surface_error);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);
        let case_passed = if field_mode {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_torus_surface_error <= 1.2e-1
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 2.5e-1
                && report.max_opacity_target_error <= 2.0
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        } else {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_target_position_error <= 8.0e-2
                && report.max_torus_surface_error <= 8.0e-2
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 3.0e-2
                && report.max_opacity_target_error <= 1.0e-2
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        };
        passed &= case_passed;
        case_reports.push(report);
    }

    if !field_mode {
        passed &= (trained_opacity_delta - UV_TORUS_OPACITY_GROWTH_DELTA).abs() <= 1.0e-3;
    }
    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }

    Ok(TorusRobustnessReport {
        passed,
        target_opacity_delta: if field_mode {
            UV_TORUS_FIELD_OPACITY_GAIN
        } else {
            UV_TORUS_OPACITY_GROWTH_DELTA
        },
        trained_opacity_delta,
        target_motion_gain: UV_TORUS_MOTION_GAIN,
        target_residual_decay: UV_TORUS_RESIDUAL_DECAY,
        max_target_position_error,
        mean_target_position_error: sum_mean_target_position_error / cases.len().max(1) as f32,
        max_torus_surface_error,
        max_color_target_error,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

pub(crate) fn torus_robustness_case_report(
    trace: &crate::RolloutTrace,
    case: TorusRobustnessCaseConfig,
) -> TorusRobustnessCaseReport {
    let major = case.seed_scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let field_mode = case.seed_mode == ParticleSeed::TorusFieldDense3d;
    let morphogen_mode = case.seed_mode == ParticleSeed::TorusMorphogenDense3d;
    let target_mesh = if field_mode || morphogen_mode {
        Some(uv_torus_mesh_target(major))
    } else {
        None
    };
    let expected_final_opacity_logit = if field_mode {
        UV_TORUS_FIELD_OPACITY_TARGET
    } else {
        UV_TORUS_INITIAL_OPACITY_LOGIT + UV_TORUS_OPACITY_GROWTH_DELTA * case.steps as f32
    };
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        major,
    );
    let mut max_initial_target_position_error = 0.0_f32;
    let mut sum_initial_target_position_error = 0.0_f32;
    let mut max_target_position_error = 0.0_f32;
    let mut sum_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut sum_torus_surface_error = 0.0_f32;
    let mut min_final_radial = f32::MAX;
    let mut max_final_radial = f32::MIN;
    let mut max_final_abs_z = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let indexed_target =
            uv_torus_sample(idx % case.particle_count.max(1), case.particle_count, major).position;
        let initial_target = if field_mode || morphogen_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([
                    initial_position[0],
                    initial_position[1],
                    initial_position[2],
                ])
                .closest
        } else {
            indexed_target
        };
        let target = if field_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([position[0], position[1], position[2]])
                .closest
        } else if morphogen_mode {
            initial_target
        } else {
            indexed_target
        };
        let initial_target_position_error = ((initial_position[0] - target[0]).powi(2)
            + (initial_position[1] - target[1]).powi(2)
            + (initial_position[2] - target[2]).powi(2))
        .sqrt();
        let initial_target_position_error = if field_mode || morphogen_mode {
            ((initial_position[0] - initial_target[0]).powi(2)
                + (initial_position[1] - initial_target[1]).powi(2)
                + (initial_position[2] - initial_target[2]).powi(2))
            .sqrt()
        } else {
            initial_target_position_error
        };
        max_initial_target_position_error =
            max_initial_target_position_error.max(initial_target_position_error);
        sum_initial_target_position_error += initial_target_position_error;

        let target_position_error = ((position[0] - target[0]).powi(2)
            + (position[1] - target[1]).powi(2)
            + (position[2] - target[2]).powi(2))
        .sqrt();
        max_target_position_error = max_target_position_error.max(target_position_error);
        sum_target_position_error += target_position_error;

        let torus_surface_error =
            uv_torus_surface_error([position[0], position[1], position[2]], major);
        max_torus_surface_error = max_torus_surface_error.max(torus_surface_error);
        sum_torus_surface_error += torus_surface_error;
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        min_final_radial = min_final_radial.min(radial);
        max_final_radial = max_final_radial.max(radial);
        max_final_abs_z = max_final_abs_z.max(position[2].abs());

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = uv_torus_position_color(target, major);
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());

    TorusRobustnessCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        torus_inner_radius: major - minor,
        torus_outer_radius: major + minor,
        max_initial_target_position_error,
        mean_initial_target_position_error: sum_initial_target_position_error
            / trace.positions.len().max(1) as f32,
        max_target_position_error,
        mean_target_position_error: sum_target_position_error / trace.positions.len().max(1) as f32,
        max_torus_surface_error,
        mean_torus_surface_error: sum_torus_surface_error / trace.positions.len().max(1) as f32,
        min_final_radial,
        max_final_radial,
        max_final_abs_z,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}
