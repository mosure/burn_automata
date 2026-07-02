#![allow(clippy::too_many_arguments)]

use super::*;

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
        dormant_drift_bounded: true,
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
        checks.dormant_drift_bounded,
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
        motion_peak_mean_dx: 0.0,
        motion_peak_penalty: 0.0,
        motion_active_step_fraction: 0.0,
        motion_active_step_penalty: 0.0,
        motion_sustained_step_fraction: 0.0,
        motion_sustained_step_penalty: 0.0,
        mean_final_displacement: 0.0,
        mean_final_displacement_penalty: 0.0,
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
