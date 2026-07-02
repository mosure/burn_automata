use super::*;

#[test]
fn growth_3d_strict_checks_report_torus_angular_diagnostics_without_generic_rejection() {
    let activation = Growth3dActivationReport {
        active_seed_count: 4,
        inactive_seed_count: 124,
        final_active_count: 64,
        newly_activated_count: 60,
        newly_activated_fraction: 0.75,
        final_active_mean_radius: 0.25,
        final_active_max_radius: 0.30,
    };
    let initial_surface = Growth3dSurfaceStats {
        mean_distance: 1.0,
        max_distance: 1.0,
    };
    let final_surface = Growth3dSurfaceStats {
        mean_distance: 0.5,
        max_distance: 0.2,
    };
    let initial_coverage = TargetCoverageStats {
        mean_distance: 1.0,
        max_distance: 1.0,
        covered_fraction: 0.0,
    };
    let final_coverage = TargetCoverageStats {
        mean_distance: 0.5,
        max_distance: 0.3,
        covered_fraction: 0.75,
    };
    let motion = growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]);
    let front = passing_growth_3d_front_report();
    let temporal = Growth3dTemporalReport {
        samples: Vec::new(),
        first_growth_step: Some(2),
        half_activation_step: Some(8),
        full_activation_step: Some(16),
        activation_span_steps: 14,
        progressive_activation: true,
        surface_mean_ratio: 0.5,
        target_coverage_mean_ratio: 0.5,
        target_coverage_fraction_delta: 0.75,
        geometry_progressive: true,
    };
    let missing_tube_support = TorusAngularCoverageReport {
        ring_bins: 24,
        tube_bins: 16,
        threshold: 0.0972,
        covered_joint_bins: 187,
        covered_ring_bins: 24,
        covered_tube_bins: 9,
        joint_coverage_fraction: 0.486_979_16,
        ring_coverage_fraction: 1.0,
        tube_coverage_fraction: 0.5625,
        max_ring_gap_bins: 0,
        max_tube_gap_bins: 7,
        mean_distance: 0.159,
        max_distance: 0.420,
    };
    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        passing_growth_3d_opacity_stats(),
        neutral_growth_3d_color_state_report(),
        emerged_growth_3d_color_state_report(),
        &passing_growth_3d_permutation_report(),
        &activation,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        Some(&missing_tube_support),
        GaussianVolumeStats::default(),
        &motion,
        &front,
        &temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.torus_angular_coverage);
    assert!(checks.passed);
    assert!(!checks.failure_reasons.contains(&"torus_angular_coverage"));

    let full_tube_support = TorusAngularCoverageReport {
        covered_joint_bins: 288,
        covered_tube_bins: 16,
        joint_coverage_fraction: 0.75,
        tube_coverage_fraction: 1.0,
        max_tube_gap_bins: 0,
        ..missing_tube_support
    };
    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        passing_growth_3d_opacity_stats(),
        neutral_growth_3d_color_state_report(),
        emerged_growth_3d_color_state_report(),
        &passing_growth_3d_permutation_report(),
        &activation,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        Some(&full_tube_support),
        GaussianVolumeStats::default(),
        &motion,
        &front,
        &temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(checks.torus_angular_coverage);
    assert!(checks.passed);
}

#[test]
fn growth_3d_strict_score_reports_torus_angular_coverage_without_score_pressure() {
    let checks = passing_growth_3d_strict_checks();
    let initial_surface = Growth3dSurfaceStats {
        mean_distance: 0.2,
        max_distance: 0.2,
    };
    let final_surface = Growth3dSurfaceStats {
        mean_distance: 0.16,
        max_distance: 0.3,
    };
    let initial_coverage = TargetCoverageStats {
        mean_distance: 1.0,
        max_distance: 1.0,
        covered_fraction: 0.1,
    };
    let final_coverage = TargetCoverageStats {
        mean_distance: 0.8,
        max_distance: 0.7,
        covered_fraction: 0.6,
    };
    let render = synthetic_render_loss(0.0, 10.0, 12.0, 14.0);
    let missing_tube_support = TorusAngularCoverageReport {
        ring_bins: 24,
        tube_bins: 16,
        threshold: 0.0972,
        covered_joint_bins: 187,
        covered_ring_bins: 24,
        covered_tube_bins: 9,
        joint_coverage_fraction: 0.486_979_16,
        ring_coverage_fraction: 1.0,
        tube_coverage_fraction: 0.5625,
        max_ring_gap_bins: 0,
        max_tube_gap_bins: 7,
        mean_distance: 0.159,
        max_distance: 0.420,
    };

    let baseline = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &render,
        GaussianVolumeStats::default(),
        None,
    );
    let angular = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &render,
        GaussianVolumeStats::default(),
        Some(&missing_tube_support),
    );

    assert!(angular.torus_angular_joint_coverage_penalty > 0.0);
    assert!(angular.torus_angular_tube_coverage_penalty > 0.0);
    assert!(angular.torus_angular_tube_gap_penalty > 0.0);
    assert_eq!(angular.score, baseline.score);
}

#[test]
fn growth_3d_strict_checks_reject_missing_surface_normal_coverage() {
    let activation = Growth3dActivationReport {
        active_seed_count: 4,
        inactive_seed_count: 124,
        final_active_count: 64,
        newly_activated_count: 60,
        newly_activated_fraction: 0.75,
        final_active_mean_radius: 0.25,
        final_active_max_radius: 0.30,
    };
    let initial_surface = Growth3dSurfaceStats {
        mean_distance: 1.0,
        max_distance: 1.0,
    };
    let final_surface = Growth3dSurfaceStats {
        mean_distance: 0.5,
        max_distance: 0.2,
    };
    let initial_coverage = TargetCoverageStats {
        mean_distance: 1.0,
        max_distance: 1.0,
        covered_fraction: 0.0,
    };
    let final_coverage = TargetCoverageStats {
        mean_distance: 0.5,
        max_distance: 0.3,
        covered_fraction: 0.75,
    };
    let missing_normal_support = SurfaceNormalCoverageReport {
        covered_target_bin_fraction: 0.40,
        mean_bin_covered_fraction: 0.30,
        ..passing_surface_normal_coverage_report()
    };
    let temporal = Growth3dTemporalReport {
        samples: Vec::new(),
        first_growth_step: Some(2),
        half_activation_step: Some(8),
        full_activation_step: Some(16),
        activation_span_steps: 14,
        progressive_activation: true,
        surface_mean_ratio: 0.5,
        target_coverage_mean_ratio: 0.5,
        target_coverage_fraction_delta: 0.75,
        geometry_progressive: true,
    };

    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        passing_growth_3d_opacity_stats(),
        neutral_growth_3d_color_state_report(),
        emerged_growth_3d_color_state_report(),
        &passing_growth_3d_permutation_report(),
        &activation,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &missing_normal_support,
        &passing_surface_normal_coverage_report(),
        None,
        GaussianVolumeStats::default(),
        &growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]),
        &passing_growth_3d_front_report(),
        &temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.surface_normal_coverage);
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"surface_normal_coverage"));

    let score = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &missing_normal_support,
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        GaussianVolumeStats::default(),
        None,
    );
    assert!(score.surface_normal_bin_penalty > 0.0);
    assert!(score.surface_normal_mean_penalty > 0.0);
}

#[test]
fn growth_3d_strict_checks_reject_gaussian_scale_budget_abuse() {
    let activation = Growth3dActivationReport {
        active_seed_count: 4,
        inactive_seed_count: 124,
        final_active_count: 64,
        newly_activated_count: 60,
        newly_activated_fraction: 0.75,
        final_active_mean_radius: 0.25,
        final_active_max_radius: 0.30,
    };
    let initial_surface = Growth3dSurfaceStats {
        mean_distance: 1.0,
        max_distance: 1.0,
    };
    let final_surface = Growth3dSurfaceStats {
        mean_distance: 0.5,
        max_distance: 0.2,
    };
    let initial_coverage = TargetCoverageStats {
        mean_distance: 1.0,
        max_distance: 1.0,
        covered_fraction: 0.0,
    };
    let final_coverage = TargetCoverageStats {
        mean_distance: 0.5,
        max_distance: 0.3,
        covered_fraction: 0.75,
    };
    let temporal = Growth3dTemporalReport {
        samples: Vec::new(),
        first_growth_step: Some(2),
        half_activation_step: Some(8),
        full_activation_step: Some(16),
        activation_span_steps: 14,
        progressive_activation: true,
        surface_mean_ratio: 0.5,
        target_coverage_mean_ratio: 0.5,
        target_coverage_fraction_delta: 0.75,
        geometry_progressive: true,
    };
    let oversized = GaussianVolumeStats {
        scale_budget_loss: ROBUST_3D_MAX_SCALE_BUDGET_LOSS + 0.1,
        oversize_fraction: ROBUST_3D_MAX_OVERSIZE_FRACTION + 0.01,
        ..GaussianVolumeStats::default()
    };

    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        passing_growth_3d_opacity_stats(),
        neutral_growth_3d_color_state_report(),
        emerged_growth_3d_color_state_report(),
        &passing_growth_3d_permutation_report(),
        &activation,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        None,
        oversized,
        &growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]),
        &passing_growth_3d_front_report(),
        &temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.gaussian_scale_budget);
    assert!(checks.failure_reasons.contains(&"gaussian_scale_budget"));

    let score = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        oversized,
        None,
    );
    assert!(score.gaussian_scale_budget_penalty > 0.0);
    assert!(score.gaussian_oversize_penalty > 0.0);
}
