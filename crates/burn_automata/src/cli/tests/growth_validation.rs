use super::*;

#[test]
fn growth_3d_validation_rejects_static_local_artifact() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let path = bin_temp_path("static_local_growth3d.bpk");
    let manifest = BpkModelManifest::from_model(
        &model,
        grid,
        Some(format!(
            "ablation-rust:{UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
        )),
    );
    crate::import::save_manifest(&path, &manifest).unwrap();

    let mut validation_cfg = growth_validation_test_config(ParticleSeed::TorusGrowth3d);
    validation_cfg.extra_seeds = vec![43, 42, 44];
    let report = growth_3d_validation_report(&path, MeshTargetArg::Torus, validation_cfg).unwrap();
    std::fs::remove_file(&path).ok();

    assert!(report.local_conditionless_lineage);
    assert!(matches!(report.gate, Growth3dValidationGateArg::Strict));
    assert!(!report.gate_passed);
    assert!(!report.strict_passed);
    assert_eq!(
        report.activation.final_active_count,
        report.activation.active_seed_count
    );
    assert_eq!(report.activation.newly_activated_count, 0);
    assert_eq!(report.max_motion_per_step, 0.0);
    assert!(report.final_opacity.finite);
    assert_eq!(
        report.final_opacity.max_allowed,
        GROWTH_3D_MAX_FINAL_OPACITY_LOGIT
    );
    assert!(report.final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
    assert!(report.strict_checks.bounded_final_opacity);
    assert_eq!(report.robustness.seed_count, 3);
    assert_eq!(
        report
            .robustness
            .seeds
            .iter()
            .map(|seed_report| seed_report.seed)
            .collect::<Vec<_>>(),
        vec![42, 43, 44]
    );
    assert!(!report.robustness.all_gate_passed);
    assert!(!report.robustness.all_temporal_activation_progressive);
    assert_eq!(report.robustness.min_newly_activated_fraction, 0.0);
    assert_eq!(report.robustness.min_active_growth_ratio, 1.0);
    assert_eq!(
        report.robustness.min_active_seed_count,
        report.robustness.min_final_active_count
    );
    assert!(report.robustness.all_bounded_final_opacity);
    assert!(!report.robustness.all_color_state_emerged);
    assert!(report.robustness.all_permutation_consistent);
    assert!(!report.seed_perturbation.passed);
    assert!(!report.robustness.all_seed_perturbation_stable);
    assert_eq!(
        report.robustness.min_perturbed_newly_activated_fraction,
        0.0
    );
    assert_eq!(report.robustness.min_perturbed_active_count_ratio, 1.0);
    assert_eq!(report.robustness.max_perturbed_active_count_ratio, 1.0);
    assert!(report.robustness.max_final_opacity <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
    assert_eq!(report.robustness.min_final_active_color_state_mean_abs, 0.0);
    assert_eq!(
        report.robustness.min_final_active_color_state_stddev_mean,
        0.0
    );
    assert!(report.robustness.max_permutation_position_error <= 1.0e-6);
    assert!(report.robustness.worst_strict_score.is_finite());
    assert!(!growth_3d_fail_on_validation_passed(&report));
}

#[test]
fn growth_3d_catalog_sanity_thresholds_match_active_catalog_floor() {
    for (target, max_total_loss, min_density, min_color, min_depth) in [
        (MeshTargetArg::Torus, 0.90, 0.95, 16.0, 14.8),
        (MeshTargetArg::Teapot, 0.85, 0.95, 18.0, 18.0),
    ] {
        let exact = synthetic_render_loss(max_total_loss, min_density, min_color, min_depth);
        let exact_report = growth_3d_catalog_sanity_report(target, &exact);
        assert!(exact_report.passed, "{target:?} should pass at threshold");
        assert_eq!(exact_report.max_total_loss, max_total_loss);
        assert_eq!(exact_report.min_density_psnr_db, min_density);
        assert_eq!(exact_report.min_color_psnr_db, min_color);
        assert_eq!(exact_report.min_depth_psnr_db, min_depth);

        let weak = synthetic_render_loss(
            max_total_loss + 1.0e-3,
            min_density - 1.0e-3,
            min_color - 1.0e-3,
            min_depth - 1.0e-3,
        );
        assert!(
            !growth_3d_catalog_sanity_report(target, &weak).passed,
            "{target:?} should fail below threshold"
        );
    }
}

#[test]
fn growth_3d_strict_checks_accept_one_eighth_active_seed_boundary() {
    let activation = Growth3dActivationReport {
        active_seed_count: 8,
        inactive_seed_count: 56,
        final_active_count: 48,
        newly_activated_count: 40,
        newly_activated_fraction: 40.0 / 56.0,
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
        GaussianVolumeStats::default(),
        &growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]),
        &passing_growth_3d_front_report(),
        &temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        64,
        true,
    );

    assert!(checks.sparse_active_seed);
    assert!(!checks.failure_reasons.contains(&"sparse_active_seed"));
    assert!(checks.passed);
}

#[test]
fn growth_3d_strict_checks_reject_seed_coordinate_scaffold() {
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

    let checks = growth_3d_strict_checks_report(
        false,
        true,
        true,
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

    assert!(!checks.no_seed_coordinate_scaffold);
    assert!(!checks.passed);
    assert!(
        checks
            .failure_reasons
            .contains(&"no_seed_coordinate_scaffold")
    );
}

#[test]
fn growth_3d_strict_checks_reject_tiny_active_extent() {
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
    let tiny_extent = Growth3dExtentReport {
        final_active_bounds_min: [-0.02, -0.02, -0.005],
        final_active_bounds_max: [0.02, 0.02, 0.005],
        final_active_extent: [0.04, 0.04, 0.01],
        axis_extent_ratio: [0.02, 0.02, 0.005],
        min_axis_extent_ratio: 0.005,
        bbox_diagonal_ratio: 0.03,
        final_active_max_radius: 0.03,
        max_radius_ratio: 0.03,
        ..passing_growth_3d_extent_report()
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
        GaussianVolumeStats::default(),
        &growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]),
        &passing_growth_3d_front_report(),
        &temporal,
        tiny_extent,
        0.25,
        0.72,
        128,
        true,
    );

    assert!(!checks.active_extent_growth);
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"active_extent_growth"));

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
        tiny_extent,
        0.72,
        &synthetic_render_loss(0.0, 10.0, 12.0, 14.0),
        GaussianVolumeStats::default(),
    );
    assert!(score.active_extent_bbox_penalty > 0.0);
    assert!(score.active_extent_min_axis_penalty > 0.0);
}

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
    );
    assert!(score.material_visible_surface_normal_bin_penalty > 0.0);
    assert!(score.material_visible_surface_normal_mean_penalty > 0.0);
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
    );
    assert!(surface_max_only.surface_max_penalty > 0.0);
    assert_eq!(surface_max_only.score, surface_max_only.surface_max_penalty);
}

#[test]
fn growth_3d_strict_checks_require_sustained_motion() {
    let sustained_motion = growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]);
    assert!(sustained_motion.active_step_fraction >= 0.50);
    assert!(sustained_motion.sustained_step_fraction >= 0.25);

    let one_shot_motion = growth_3d_motion_report(&[0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(one_shot_motion.active_step_fraction < 0.50);
    assert!(one_shot_motion.sustained_step_fraction < 0.25);

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
    let bulk_temporal = Growth3dTemporalReport {
        samples: Vec::new(),
        first_growth_step: Some(8),
        half_activation_step: Some(8),
        full_activation_step: Some(8),
        activation_span_steps: 0,
        progressive_activation: false,
        surface_mean_ratio: 1.0,
        target_coverage_mean_ratio: 1.0,
        target_coverage_fraction_delta: 0.0,
        geometry_progressive: false,
    };
    let staged_temporal = Growth3dTemporalReport {
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
    let local_front = passing_growth_3d_front_report();

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
        GaussianVolumeStats::default(),
        &one_shot_motion,
        &local_front,
        &staged_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.sustained_motion);
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"sustained_motion"));

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
        GaussianVolumeStats::default(),
        &sustained_motion,
        &local_front,
        &bulk_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(checks.sustained_motion);
    assert!(!checks.temporal_activation_progressive);
    assert!(!checks.passed);
    assert!(
        checks
            .failure_reasons
            .contains(&"temporal_activation_progressive")
    );
    assert!(
        checks
            .failure_reasons
            .contains(&"temporal_geometry_progressive")
    );

    let high_opacity = Growth3dOpacityStats {
        max: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT + 1.0,
        active_max: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT + 1.0,
        ..passing_growth_3d_opacity_stats()
    };
    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        high_opacity,
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
        GaussianVolumeStats::default(),
        &sustained_motion,
        &local_front,
        &staged_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.bounded_final_opacity);
    assert!(checks.failure_reasons.contains(&"bounded_final_opacity"));

    let checks = growth_3d_strict_checks_report(
        false,
        true,
        false,
        0.0,
        passing_growth_3d_opacity_stats(),
        neutral_growth_3d_color_state_report(),
        neutral_growth_3d_color_state_report(),
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
        GaussianVolumeStats::default(),
        &sustained_motion,
        &local_front,
        &staged_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.color_state_emerged);
    assert!(checks.failure_reasons.contains(&"color_state_emerged"));

    let non_local_front = Growth3dFrontReport {
        passed: false,
        local_newly_activated_fraction: 0.25,
        mean_nearest_previous_active_distance: 0.7,
        max_nearest_previous_active_distance: 1.1,
        ..passing_growth_3d_front_report()
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
        GaussianVolumeStats::default(),
        &sustained_motion,
        &non_local_front,
        &staged_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );
    assert!(!checks.local_front_coherent);
    assert!(checks.failure_reasons.contains(&"local_front_coherent"));
}

#[test]
fn growth_3d_strict_checks_reject_surface_max_escape() {
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
    let escaped_surface = Growth3dSurfaceStats {
        mean_distance: 0.5,
        max_distance: GROWTH_3D_SURFACE_MAX_DISTANCE + 0.01,
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
    let sustained_motion = Growth3dMotionReport {
        first_step_mean_dx: 0.02,
        peak_mean_dx: 0.08,
        peak_step: 4,
        final_step_mean_dx: 0.03,
        mean_dx: 0.04,
        late_mean_dx: 0.03,
        late_to_peak_ratio: 0.375,
        active_step_fraction: 0.75,
        sustained_step_fraction: 0.50,
    };
    let progressive_temporal = Growth3dTemporalReport {
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
        escaped_surface,
        passing_growth_3d_surface_tail_report(),
        initial_coverage,
        final_coverage,
        final_coverage,
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        None,
        GaussianVolumeStats::default(),
        &sustained_motion,
        &passing_growth_3d_front_report(),
        &progressive_temporal,
        passing_growth_3d_extent_report(),
        0.25,
        0.72,
        128,
        true,
    );

    assert!(!checks.surface_max_bounded);
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"surface_max_bounded"));
}

#[test]
fn growth_3d_strict_checks_reject_missing_torus_angular_coverage() {
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
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"torus_angular_coverage"));

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
    );
    assert!(score.gaussian_scale_budget_penalty > 0.0);
    assert!(score.gaussian_oversize_penalty > 0.0);
}
