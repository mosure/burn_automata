use super::*;

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
        true,
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
        true,
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
fn growth_3d_strict_checks_reject_nonlocal_dormant_drift() {
    let mut checks = passing_growth_3d_strict_checks();
    let drift = Growth3dDormantDriftReport {
        sampled_steps: 4,
        checked_rows: 32,
        drifting_rows: 3,
        drifting_fraction: 3.0 / 32.0,
        mean_dormant_displacement: 0.03,
        max_dormant_displacement: 0.25,
        max_allowed_displacement: 0.18,
        finite: true,
        passed: false,
    };

    apply_dormant_drift_strict_check(&mut checks, drift);

    assert!(!checks.dormant_drift_bounded);
    assert!(!checks.passed);
    assert!(checks.failure_reasons.contains(&"dormant_drift_bounded"));

    let score = growth_3d_strict_score_report(
        &checks,
        Growth3dSurfaceStats {
            mean_distance: 1.0,
            max_distance: 1.0,
        },
        Growth3dSurfaceStats {
            mean_distance: 0.5,
            max_distance: 0.2,
        },
        passing_growth_3d_surface_tail_report(),
        TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.0,
        },
        TargetCoverageStats {
            mean_distance: 0.5,
            max_distance: 0.3,
            covered_fraction: 0.75,
        },
        TargetCoverageStats {
            mean_distance: 0.5,
            max_distance: 0.3,
            covered_fraction: 0.75,
        },
        &passing_surface_normal_coverage_report(),
        &passing_surface_normal_coverage_report(),
        passing_growth_3d_extent_report(),
        0.72,
        &synthetic_render_loss(0.0, 20.0, 20.0, 20.0),
        GaussianVolumeStats::default(),
        None,
    );

    assert!(
        score.hard_failure_penalty >= 10.0,
        "nonlocal dormant drift must contribute a hard strict-score penalty"
    );
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
fn morphogenesis_dynamics_strict_score_tracks_motion_progress() {
    let checks = passing_growth_3d_strict_checks();
    let render = synthetic_render_loss(0.0, 10.0, 12.0, 14.0);
    let mut weak = growth_3d_strict_score_report(
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
        &render,
        GaussianVolumeStats::default(),
        None,
    );
    let mut strong = growth_3d_strict_score_report(
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
        &render,
        GaussianVolumeStats::default(),
        None,
    );

    apply_morphogenesis_dynamics_strict_score(
        &mut weak,
        &growth_3d_motion_report(&[0.001, 0.002, 0.001, 0.0]),
        0.04,
        0.72,
    );
    apply_morphogenesis_dynamics_strict_score(
        &mut strong,
        &growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]),
        0.20,
        0.72,
    );

    assert!(weak.motion_peak_penalty > 0.0);
    assert!(weak.mean_final_displacement_penalty > 0.0);
    assert_eq!(strong.motion_peak_penalty, 0.0);
    assert_eq!(strong.mean_final_displacement_penalty, 0.0);
    assert!(
        strong.score < weak.score,
        "strict-score ranking should prefer candidates with stronger realized morphogenesis dynamics"
    );
}
