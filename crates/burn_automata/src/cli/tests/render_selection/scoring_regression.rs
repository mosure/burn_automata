use super::*;

#[test]
fn render_selection_score_penalizes_surface_normal_coverage_regression() {
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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: SurfaceNormalCoverageReport {
            covered_target_bin_fraction: 0.40,
            mean_bin_covered_fraction: 0.20,
            ..passing_surface_normal_coverage_report()
        },
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
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
        scored.score > case.score + 7.0,
        "normal-bin coverage regression should add a meaningful selection penalty"
    );
}

#[test]
fn render_selection_score_penalizes_material_visible_coverage_regression() {
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
            mean_distance: 0.40,
            max_distance: 0.90,
            covered_fraction: 0.20,
        },
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
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
        scored.score > case.score + 5.0,
        "material-visible target coverage regression should add a meaningful selection penalty"
    );
}

#[test]
fn render_selection_score_rewards_lower_material_visible_target_distance() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.material_visible_target_coverage = TargetCoverageStats {
        mean_distance: 0.50,
        max_distance: 1.00,
        covered_fraction: 0.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.material_visible_target_coverage = TargetCoverageStats {
        mean_distance: 0.20,
        max_distance: 0.50,
        covered_fraction: 0.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(better_score.score < weak_score.score);
    let expected_improvement = (0.50 - 0.20) * MATERIAL_VISIBLE_TARGET_MEAN_DISTANCE_SCORE_WEIGHT
        + (1.00 - 0.50) * MATERIAL_VISIBLE_TARGET_MAX_DISTANCE_SCORE_WEIGHT;
    assert!(
        (weak_score.score - better_score.score - expected_improvement).abs() < 1.0e-6,
        "material-visible surface approach should be visible to selection before coverage threshold flips"
    );
}

#[test]
fn render_selection_score_penalizes_material_visible_target_distance_regression() {
    let baseline_case = render_selection_case_with_front_liveness_margin(0.0);
    let baseline = vec![render_selection_baseline_case_from_metrics(
        7,
        &baseline_case,
    )];
    let mut regressed = baseline_case;
    regressed.material_visible_target_coverage = TargetCoverageStats {
        mean_distance: 0.20,
        max_distance: 0.40,
        covered_fraction: 0.82,
    };

    let scored = render_selection_case_score_with_baseline(7, &regressed, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > regressed.score,
        "material-visible particles moving away from the target should count as morphology regression even before coverage fraction changes"
    );
}

#[test]
fn render_selection_score_penalizes_material_visible_liveness_regression() {
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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: Growth3dMaterialLivenessReport {
            material_visible_count: 16,
            inactive_material_visible_count: 4,
            inactive_material_visible_fraction: 0.25,
            inactive_material_logit_threshold: 1.0,
            max_inactive_material_opacity: 6.0,
            passed: false,
        },
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
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
        scored.score > case.score + 2.0,
        "inactive material-visible regression should add a meaningful selection penalty"
    );
}

#[test]
fn render_selection_score_penalizes_growth_timing_regression() {
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
        final_active_count: 96,
        newly_activated_fraction: 0.65,
        front_local_newly_activated_fraction: 0.45,
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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        surface_coverage_profile: passing_surface_coverage_profile_report(),
        material_visible_surface_coverage_profile: passing_surface_coverage_profile_report(),
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_tail: passing_growth_3d_surface_tail_report(),
        extent: passing_growth_3d_extent_report(),
        final_active_count: 48,
        newly_activated_fraction: 0.10,
        front_local_newly_activated_fraction: 0.05,
        front_liveness: LocalFrontLivenessProgress::default(),
        extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_extent_front_liveness: LocalFrontLivenessProgress::default(),
        temporal_activation_schedule_error: 0.0,
        temporal_activation_progressive: false,
        temporal_geometry_progressive: false,
        score: 1.0,
        failure_reasons: Vec::new(),
    };

    let scored = render_selection_case_score_with_baseline(7, &case, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > case.score + 30.0,
        "growth timing regression should dominate otherwise good render/geometry metrics"
    );
}

#[test]
fn render_selection_score_penalizes_surface_profile_regression() {
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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
        surface_coverage_profile: SurfaceCoverageProfileReport {
            covered_bin_fraction: 0.40,
            mean_bin_covered_fraction: 0.35,
            empty_bins: 38,
            ..passing_surface_coverage_profile_report()
        },
        material_visible_surface_coverage_profile: SurfaceCoverageProfileReport {
            covered_bin_fraction: 0.30,
            mean_bin_covered_fraction: 0.25,
            empty_bins: 45,
            ..passing_surface_coverage_profile_report()
        },
        surface_normal_coverage: passing_surface_normal_coverage_report(),
        material_visible_surface_normal_coverage: passing_surface_normal_coverage_report(),
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
        scored.score > case.score + 15.0,
        "surface-strata coverage regression should add a meaningful selection penalty"
    );
}
