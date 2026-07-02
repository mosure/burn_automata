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
fn growth_3d_validation_rejects_mismatched_target_lineage() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let path = bin_temp_path("mismatched_target_lineage_growth3d.bpk");
    let manifest = BpkModelManifest::from_model(
        &model,
        grid,
        Some(format!(
            "ablation-rust:{UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
        )),
    );
    crate::import::save_manifest(&path, &manifest).unwrap();

    let report = growth_3d_validation_report(
        &path,
        MeshTargetArg::Teapot,
        growth_validation_test_config(ParticleSeed::TeapotLocalSubstrateGrowth3d),
    )
    .unwrap();
    std::fs::remove_file(&path).ok();

    assert!(report.local_conditionless_lineage);
    assert!(!report.target_conditionless_lineage);
    assert!(!report.strict_checks.target_conditionless_lineage);
    assert!(!report.strict_passed);
    assert!(
        report
            .strict_checks
            .failure_reasons
            .contains(&"target_conditionless_lineage"),
        "target-mismatched local artifacts should fail strict validation explicitly"
    );
    assert!(!report.robustness.all_target_conditionless_lineage);
    assert!(
        report
            .robustness
            .seeds
            .iter()
            .all(|seed| !seed.target_conditionless_lineage)
    );
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
        None,
    );
    assert!(score.active_extent_bbox_penalty > 0.0);
    assert!(score.active_extent_min_axis_penalty > 0.0);
}
