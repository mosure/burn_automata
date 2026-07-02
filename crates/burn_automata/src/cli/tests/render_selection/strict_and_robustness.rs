use super::*;

#[test]
fn growth_3d_strict_checks_reject_sparse_surface_profile_coverage() {
    let mut checks = passing_growth_3d_strict_checks();
    let sparse_active = SurfaceCoverageProfileReport {
        covered_bin_fraction: 0.25,
        mean_bin_covered_fraction: 0.20,
        empty_bins: 48,
        ..passing_surface_coverage_profile_report()
    };
    let sparse_material = SurfaceCoverageProfileReport {
        covered_bin_fraction: 0.30,
        mean_bin_covered_fraction: 0.25,
        empty_bins: 45,
        ..passing_surface_coverage_profile_report()
    };

    apply_surface_profile_strict_check(&mut checks, &sparse_active, &sparse_material);

    assert!(!checks.surface_coverage_profile);
    assert!(!checks.material_visible_surface_coverage_profile);
    assert!(checks.failure_reasons.contains(&"surface_coverage_profile"));
    assert!(
        checks
            .failure_reasons
            .contains(&"material_visible_surface_coverage_profile")
    );

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
    let mut score = growth_3d_strict_score_report(
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
        GaussianVolumeStats::default(),
    );
    apply_surface_profile_strict_score(&mut score, &sparse_active, &sparse_material);

    assert!(score.hard_failure_penalty >= 20.0);
    assert!(score.surface_bin_penalty > 0.0);
    assert!(score.surface_coverage_mean_penalty > 0.0);
    assert!(score.material_visible_surface_bin_penalty > 0.0);
    assert!(score.material_visible_surface_mean_penalty > 0.0);
}

#[test]
fn render_selection_score_penalizes_material_visible_normal_regression() {
    let baseline = vec![RenderSelectionBaselineCase {
        seed: 7,
        active_surface_max: 0.12,
        target_coverage_fraction: 0.82,
        material_visible_target_mean_distance: 0.05,
        material_visible_target_max_distance: 0.12,
        material_visible_target_coverage_fraction: 0.82,
        material_visible_inactive_fraction: 0.0,
        material_visible_max_inactive_opacity: f32::NEG_INFINITY,
        surface_covered_bin_fraction: 0.90,
        surface_mean_bin_covered_fraction: 0.80,
        material_visible_surface_covered_bin_fraction: 0.90,
        material_visible_surface_mean_bin_covered_fraction: 0.80,
        surface_normal_covered_bin_fraction: 0.80,
        surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_normal_covered_bin_fraction: 0.80,
        material_visible_surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_tail_p99_distance: 0.20,
        material_visible_surface_tail_over_threshold_fraction: 0.0,
        active_extent_bbox_ratio: 0.35,
        active_extent_min_axis_ratio: 0.15,
        final_active_count: 64,
        newly_activated_fraction: 0.75,
        front_local_newly_activated_fraction: 0.70,
        front_liveness: LocalFrontLivenessProgress::default(),
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
    }];
    let case = RenderSelectionCaseMetrics {
        render_loss: synthetic_render_loss(1.0, 20.0, 20.0, 20.0),
        active_surface: Growth3dSurfaceStats {
            mean_distance: 0.05,
            max_distance: 0.12,
        },
        target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        material_visible_target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        strict_surface_materialization: passing_strict_surface_materialization_report(),
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        final_color_state: emerged_growth_3d_color_state_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: SurfaceNormalCoverageReport {
            covered_target_bin_fraction: 0.35,
            mean_bin_covered_fraction: 0.25,
            ..passing_surface_normal_coverage_report()
        },
        material_visible_surface_tail: passing_growth_3d_surface_tail_report(),
        extent: passing_growth_3d_extent_report(),
        final_active_count: 64,
        newly_activated_fraction: 0.75,
        front_local_newly_activated_fraction: 0.70,
        front_liveness: LocalFrontLivenessProgress::default(),
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
        score: 1.0,
        failure_reasons: Vec::new(),
    };

    let scored = render_selection_case_score_with_baseline(7, &case, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > case.score + 8.0,
        "material-visible normal regression should add a meaningful selection penalty"
    );
}

#[test]
fn render_selection_score_penalizes_material_visible_tail_regression() {
    let baseline = vec![RenderSelectionBaselineCase {
        seed: 7,
        active_surface_max: 0.12,
        target_coverage_fraction: 0.82,
        material_visible_target_mean_distance: 0.05,
        material_visible_target_max_distance: 0.12,
        material_visible_target_coverage_fraction: 0.82,
        material_visible_inactive_fraction: 0.0,
        material_visible_max_inactive_opacity: f32::NEG_INFINITY,
        surface_covered_bin_fraction: 0.90,
        surface_mean_bin_covered_fraction: 0.80,
        material_visible_surface_covered_bin_fraction: 0.90,
        material_visible_surface_mean_bin_covered_fraction: 0.80,
        surface_normal_covered_bin_fraction: 0.80,
        surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_normal_covered_bin_fraction: 0.80,
        material_visible_surface_normal_mean_bin_covered_fraction: 0.70,
        material_visible_surface_tail_p99_distance: 0.20,
        material_visible_surface_tail_over_threshold_fraction: 0.0,
        active_extent_bbox_ratio: 0.35,
        active_extent_min_axis_ratio: 0.15,
        final_active_count: 64,
        newly_activated_fraction: 0.75,
        front_local_newly_activated_fraction: 0.70,
        front_liveness: LocalFrontLivenessProgress::default(),
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
    }];
    let case = RenderSelectionCaseMetrics {
        render_loss: synthetic_render_loss(1.0, 20.0, 20.0, 20.0),
        active_surface: Growth3dSurfaceStats {
            mean_distance: 0.05,
            max_distance: 0.12,
        },
        target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        material_visible_target_coverage: TargetCoverageStats {
            mean_distance: 0.05,
            max_distance: 0.12,
            covered_fraction: 0.82,
        },
        strict_surface_materialization: passing_strict_surface_materialization_report(),
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        final_color_state: emerged_growth_3d_color_state_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_tail: Growth3dSurfaceTailReport {
            p99_distance: 0.80,
            over_threshold_fraction: 0.25,
            opacity_weighted_over_threshold_fraction: 0.20,
            ..passing_growth_3d_surface_tail_report()
        },
        extent: passing_growth_3d_extent_report(),
        final_active_count: 64,
        newly_activated_fraction: 0.75,
        front_local_newly_activated_fraction: 0.70,
        front_liveness: LocalFrontLivenessProgress::default(),
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: true,
        temporal_geometry_progressive: true,
        score: 1.0,
        failure_reasons: Vec::new(),
    };

    let scored = render_selection_case_score_with_baseline(7, &case, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > case.score + 8.0,
        "material-visible tail regression should add a meaningful selection penalty"
    );
}

#[test]
fn growth_3d_robustness_report_aggregates_surface_normal_coverage() {
    let report = growth_3d_robustness_report(vec![
        robustness_seed_report_with_surface_normal_coverage(11, true, 0.80, 0.70),
        robustness_seed_report_with_surface_normal_coverage(19, false, 0.35, 0.25),
    ]);

    assert!(!report.all_surface_normal_coverage);
    assert!(!report.all_material_visible_surface_normal_coverage);
    assert!(!report.all_surface_coverage_profile);
    assert!(!report.all_material_visible_surface_coverage_profile);
    assert_eq!(
        report.min_final_active_surface_normal_covered_bin_fraction,
        0.35
    );
    assert_eq!(
        report.min_final_active_surface_normal_mean_bin_covered_fraction,
        0.25
    );
    assert_eq!(report.min_final_active_surface_covered_bin_fraction, 0.35);
    assert_eq!(
        report.min_final_active_surface_mean_bin_covered_fraction,
        0.25
    );
    assert_eq!(
        report.min_final_material_visible_surface_covered_bin_fraction,
        0.35
    );
    assert_eq!(
        report.min_final_material_visible_surface_mean_bin_covered_fraction,
        0.25
    );
    assert_eq!(
        report.min_final_material_visible_surface_normal_covered_bin_fraction,
        0.35
    );
    assert_eq!(
        report.min_final_material_visible_surface_normal_mean_bin_covered_fraction,
        0.25
    );
    assert_eq!(report.seeds[1].seed, 19);
    assert!(!report.seeds[1].surface_coverage_profile);
    assert!(!report.seeds[1].material_visible_surface_coverage_profile);
    assert!(!report.seeds[1].surface_normal_coverage);
    assert!(!report.seeds[1].material_visible_surface_normal_coverage);
}
