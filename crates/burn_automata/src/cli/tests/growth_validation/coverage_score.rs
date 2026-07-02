use super::*;

#[test]
fn growth_3d_strict_checks_reject_transparent_target_coverage() {
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
    let final_active_coverage = TargetCoverageStats {
        mean_distance: 0.5,
        max_distance: 0.3,
        covered_fraction: 0.75,
    };
    let final_material_visible_coverage = TargetCoverageStats {
        mean_distance: 0.9,
        max_distance: 0.8,
        covered_fraction: 0.25,
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
        true,
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
        final_active_coverage,
        final_material_visible_coverage,
        &passing_surface_normal_coverage_report(),
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

    assert!(checks.target_coverage_fraction);
    assert!(!checks.material_visible_target_coverage_fraction);
    assert!(!checks.passed);
    assert!(
        checks
            .failure_reasons
            .contains(&"material_visible_target_coverage_fraction")
    );

    let score = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_active_coverage,
        final_material_visible_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        GaussianVolumeStats::default(),
        None,
    );
    assert!(score.material_visible_target_coverage_penalty > 0.0);
}

#[test]
fn growth_3d_strict_checks_reject_transparent_normal_support() {
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
    let missing_visible_normal_support = SurfaceNormalCoverageReport {
        covered_target_bin_fraction: 0.40,
        mean_bin_covered_fraction: 0.20,
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
        true,
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
        &missing_visible_normal_support,
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

    assert!(checks.surface_normal_coverage);
    assert!(!checks.material_visible_surface_normal_coverage);
    assert!(!checks.passed);
    assert!(
        checks
            .failure_reasons
            .contains(&"material_visible_surface_normal_coverage")
    );

    let score = growth_3d_strict_score_report(
        &checks,
        initial_surface,
        final_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &missing_visible_normal_support,
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        GaussianVolumeStats::default(),
        None,
    );
    assert!(score.material_visible_surface_normal_bin_penalty > 0.0);
    assert!(score.material_visible_surface_normal_mean_penalty > 0.0);
}

#[test]
fn growth_3d_strict_score_counts_shape_render_failures_as_hard_gates() {
    let mut checks = passing_growth_3d_strict_checks();
    checks.passed = false;
    checks.target_coverage_fraction = false;
    checks.material_visible_target_coverage_fraction = false;
    checks.surface_normal_coverage = false;
    checks.material_visible_surface_normal_coverage = false;
    checks.render_loss_passed = false;
    checks.failure_reasons = vec![
        "target_coverage_fraction",
        "material_visible_target_coverage_fraction",
        "surface_normal_coverage",
        "material_visible_surface_normal_coverage",
        "render_loss_passed",
    ];

    let score = strict_score_for_gate_checks(&checks);

    assert!(
        score.hard_failure_penalty >= 50.0,
        "strict score should treat every failed shape/render promotion gate as hard; score={}",
        score.hard_failure_penalty
    );
    assert_eq!(score.target_coverage_fraction_penalty, 0.0);
    assert_eq!(score.material_visible_target_coverage_penalty, 0.0);
    assert_eq!(score.surface_normal_bin_penalty, 0.0);
    assert_eq!(score.material_visible_surface_normal_bin_penalty, 0.0);
    assert_eq!(score.render_density_penalty, 0.0);
    assert!(
        score.score >= score.hard_failure_penalty,
        "hard gate failures must dominate even when continuous distances sit exactly on threshold"
    );
}

#[test]
fn growth_3d_strict_score_tracks_distance_to_gate() {
    let checks = passing_growth_3d_strict_checks();
    let perfect_render = synthetic_render_loss(0.0, 10.0, 12.0, 14.0);
    let perfect = growth_3d_strict_score_report(
        &checks,
        Growth3dSurfaceStats {
            mean_distance: 0.2,
            max_distance: 0.2,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.16,
            max_distance: 0.3,
        },
        passing_growth_3d_surface_tail_report(),
        TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.1,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.6,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.6,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &perfect_render,
        GaussianVolumeStats::default(),
        None,
    );
    assert_eq!(perfect.score, 0.0);

    let weak_render = synthetic_render_loss(1.0, 1.0, 10.0, 10.0);
    let weak = growth_3d_strict_score_report(
        &checks,
        Growth3dSurfaceStats {
            mean_distance: 0.2,
            max_distance: 0.2,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.22,
            max_distance: 0.5,
        },
        Growth3dSurfaceTailReport {
            p95_distance: 0.45,
            p99_distance: 0.5,
            max_distance: 0.5,
            over_threshold_count: 16,
            over_threshold_fraction: 0.10,
            opacity_weighted_over_threshold_fraction: 0.08,
            ..passing_growth_3d_surface_tail_report()
        },
        TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.1,
        },
        TargetCoverageStats {
            mean_distance: 0.9,
            max_distance: 0.8,
            covered_fraction: 0.4,
        },
        TargetCoverageStats {
            mean_distance: 0.9,
            max_distance: 0.8,
            covered_fraction: 0.4,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &weak_render,
        GaussianVolumeStats::default(),
        None,
    );
    assert!(weak.score > perfect.score);
    assert!(weak.surface_mean_penalty > 0.0);
    assert!(weak.surface_max_penalty > 0.0);
    assert!(weak.target_coverage_fraction_penalty > 0.0);
    assert!(weak.render_density_penalty > 0.0);

    let surface_max_only = growth_3d_strict_score_report(
        &checks,
        Growth3dSurfaceStats {
            mean_distance: 0.2,
            max_distance: 0.2,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.16,
            max_distance: GROWTH_3D_SURFACE_MAX_DISTANCE + 0.05,
        },
        passing_growth_3d_surface_tail_report(),
        TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.1,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.6,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.6,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &perfect_render,
        GaussianVolumeStats::default(),
        None,
    );
    assert!(surface_max_only.surface_max_penalty > 0.0);
    assert_eq!(surface_max_only.score, surface_max_only.surface_max_penalty);
}

fn strict_score_for_gate_checks(checks: &Growth3dStrictChecksReport) -> Growth3dStrictScoreReport {
    growth_3d_strict_score_report(
        checks,
        Growth3dSurfaceStats {
            mean_distance: 0.2,
            max_distance: 0.2,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.16,
            max_distance: 0.3,
        },
        passing_growth_3d_surface_tail_report(),
        TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.1,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.60,
        },
        TargetCoverageStats {
            mean_distance: 0.8,
            max_distance: 0.7,
            covered_fraction: 0.60,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        GaussianVolumeStats::default(),
        None,
    )
}
