use super::*;

#[test]
fn render_proxy_gradient_rows_cover_full_cloud_instead_of_prefix_only() {
    assert_eq!(
        render_proxy_gradient_row_indices(1024, 8),
        vec![0, 128, 256, 384, 512, 640, 768, 896]
    );
    assert_eq!(render_proxy_gradient_row_indices(4, 8), vec![0, 1, 2, 3]);
    assert_eq!(render_proxy_gradient_row_indices(1024, 1), vec![0]);
}

#[test]
fn trajectory_render_sample_indices_cover_late_rollout_evenly() {
    assert_eq!(trajectory_render_sample_indices(0, 4), Vec::<usize>::new());
    assert_eq!(trajectory_render_sample_indices(8, 0), Vec::<usize>::new());
    assert_eq!(trajectory_render_sample_indices(8, 3), vec![1, 4, 7]);
    assert_eq!(trajectory_render_sample_indices(4, 16), vec![0, 1, 2, 3]);
}

#[test]
fn trajectory_liveness_sample_indices_cover_early_rollout() {
    assert_eq!(
        trajectory_liveness_sample_indices(0, 4),
        Vec::<usize>::new()
    );
    assert_eq!(
        trajectory_liveness_sample_indices(4, 2),
        vec![0, 1, 2, 3],
        "short rollouts should expose every temporal transition to liveness scheduling"
    );
    let long = trajectory_liveness_sample_indices(32, 4);
    assert_eq!(&long[..2], &[0, 1]);
    assert_eq!(long.last().copied(), Some(31));
    assert!(
        long.len() <= TEMPORAL_LIVENESS_TRAJECTORY_SAMPLE_CAP + 2,
        "long rollout liveness sampling should stay bounded"
    );
}

#[test]
fn temporal_activation_allowed_fraction_matches_progressive_growth_gate() {
    assert!(
        (0.20..0.30).contains(&temporal_activation_target_fraction(0.25)),
        "quarter-rollout activation should start expanding the local front without waking the whole cloud"
    );
    assert!(
        (0.48..0.52).contains(&temporal_activation_target_fraction(0.50)),
        "mid-rollout activation target should align with the strict half-activation gate"
    );
    assert!(
        temporal_activation_allowed_fraction(1.0) < 1.0,
        "the temporal schedule should not treat all-particle activation as a valid final shortcut"
    );
    assert_eq!(temporal_activation_allowed_fraction(1.0), 0.95);
}

#[test]
fn direct_trajectory_geometry_weight_ramps_without_disabling_late_support() {
    assert!((direct_trajectory_geometry_weight(0.0) - 0.5).abs() <= 1.0e-6);
    assert!((direct_trajectory_geometry_weight(0.5) - 0.75).abs() <= 1.0e-6);
    assert!((direct_trajectory_geometry_weight(1.0) - 1.0).abs() <= 1.0e-6);
}

#[test]
fn temporal_activation_schedule_error_penalizes_burst_growth() {
    let sample = |steps: usize, active_fraction: f32| Growth3dTemporalSampleReport {
        steps,
        active_count: (active_fraction * 32.0).round() as usize,
        active_fraction,
        newly_activated_count: 0,
        final_active_mean_radius: 0.0,
        final_active_max_radius: 0.0,
        mean_displacement: 0.0,
        active_surface: Growth3dSurfaceStats {
            mean_distance: 1.0,
            max_distance: 1.0,
        },
        target_coverage: TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.0,
        },
    };
    let report = |fractions: [f32; 4]| Growth3dTemporalReport {
        samples: vec![
            sample(0, fractions[0]),
            sample(1, fractions[1]),
            sample(2, fractions[2]),
            sample(4, fractions[3]),
        ],
        first_growth_step: None,
        half_activation_step: None,
        full_activation_step: None,
        activation_span_steps: 0,
        progressive_activation: false,
        surface_mean_ratio: 1.0,
        target_coverage_mean_ratio: 1.0,
        target_coverage_fraction_delta: 0.0,
        geometry_progressive: false,
    };
    let abrupt = report([0.03, 0.03, 0.53, 1.0]);
    let staged = report([0.03, 0.08, 0.25, 0.95]);

    assert!(
        temporal_activation_schedule_error(&abrupt, 4)
            > temporal_activation_schedule_error(&staged, 4),
        "selection should distinguish burst activation from staged rollout growth"
    );
}

#[test]
fn mesh_rollout_snapshot_steps_include_initial_and_final_when_temporal() {
    assert_eq!(mesh_rollout_snapshot_steps(8, 1), vec![8]);
    assert_eq!(mesh_rollout_snapshot_steps(8, 3), vec![0, 4, 8]);
    assert_eq!(mesh_rollout_snapshot_steps(8, 4), vec![0, 2, 5, 8]);
    assert_eq!(mesh_rollout_snapshot_steps(0, 4), vec![0]);
}

#[test]
fn mesh_rollout_row_indices_keep_sparse_high_signal_rows() {
    let output_dims = 6;
    let particle_count = 32;
    let row_budget = 6;
    let mut target_update = vec![0.0_f32; particle_count * output_dims];
    target_update[17 * output_dims + 3] = 2.0;
    target_update[23 * output_dims] = -1.5;

    let rows = mesh_rollout_row_indices(&target_update, output_dims, particle_count, row_budget);

    assert_eq!(rows.len(), row_budget);
    assert!(
        rows.contains(&17) && rows.contains(&23),
        "sparse front/material rows should not be lost to uniform spread sampling: {rows:?}"
    );
    assert_eq!(
        rows.iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        rows.len()
    );
}

#[test]
fn render_selection_metrics_average_base_and_selection_seed_with_morphology_penalty() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = local_growth_student_model(config, 17, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
    let render = RenderLossConfig {
        image_size: 8,
        target_samples: 128,
        world_scale: 1.44,
        ..RenderLossConfig::default()
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 128,
        rollout_steps: 2,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.1,
        perception_position_gain: 0.05,
        max_update_norm: 0.1,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 0,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: 0.0,
        full_coverage_adjoint: false,
        surface_gain: 0.0,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: 0.0,
        material_liveness_gain: 0.0,
        material_tail_gain: 0.0,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: 0.0,
        scale_budget_weight: 0.0,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::Proxy,
        direct_selection_seed_training: false,
        seed: 11,
        selection_seed: Some(19),
        selection_seeds: vec![23, 11, 19],
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render,
        sgd: SgdConfig {
            learning_rate: 1.0e-4,
            grad_clip_norm: 0.1,
            weight_decay: 0.0,
        },
    };

    let base =
        render_selection_case_metrics(&model, &grid, &target, &cfg, render, cfg.seed).unwrap();
    let heldout = render_selection_case_metrics(&model, &grid, &target, &cfg, render, 19).unwrap();
    let extra = render_selection_case_metrics(&model, &grid, &target, &cfg, render, 23).unwrap();
    let baseline = render_selection_baseline(&model, &grid, &target, &cfg, render).unwrap();
    let selection =
        render_selection_metrics(&model, &grid, &target, &cfg, render, Some(&baseline)).unwrap();

    assert!((selection.base_report.total_loss - base.render_loss.total_loss).abs() <= 1.0e-6);
    assert!(
        (selection.render_loss
            - (base.render_loss.total_loss
                + heldout.render_loss.total_loss
                + extra.render_loss.total_loss)
                / 3.0)
            .abs()
            <= 1.0e-6
    );
    assert_eq!(render_proxy_selection_seeds(&cfg), vec![cfg.seed, 19, 23]);
    let base_scored = render_selection_case_score_with_baseline(cfg.seed, &base, Some(&baseline));
    let heldout_scored = render_selection_case_score_with_baseline(19, &heldout, Some(&baseline));
    let extra_scored = render_selection_case_score_with_baseline(23, &extra, Some(&baseline));
    let expected_score = base_scored
        .score
        .max(heldout_scored.score)
        .max(extra_scored.score);
    assert!(
        (selection.score - expected_score).abs() <= 1.0e-5,
        "selection score {} expected worst-case {} from base {} heldout {}",
        selection.score,
        expected_score,
        base_scored.score,
        heldout_scored.score
    );
    let candidate_scores = [
        (cfg.seed, base_scored.score),
        (19, heldout_scored.score),
        (23, extra_scored.score),
    ];
    assert!(
        candidate_scores.iter().any(|(seed, score)| {
            *seed == selection.worst_seed && (*score - expected_score).abs() <= 1.0e-5
        }),
        "worst seed {} should be one of the max-score candidates {:?}",
        selection.worst_seed,
        candidate_scores
    );
    assert!(
        !selection.worst_failure_reasons.is_empty(),
        "worst selection seed should expose strict failure reasons"
    );
    assert!(
        selection
            .worst_failure_reasons
            .contains(&"torus_angular_coverage"),
        "torus render-proxy selection must preserve angular-support blockers"
    );
    assert!(
        (selection.density_psnr_db
            - (base.render_loss.density_psnr_db
                + heldout.render_loss.density_psnr_db
                + extra.render_loss.density_psnr_db)
                / 3.0)
            .abs()
            <= 1.0e-6
    );
    assert_eq!(
        selection.active_surface_max,
        base.active_surface
            .max_distance
            .max(heldout.active_surface.max_distance)
            .max(extra.active_surface.max_distance)
    );
    assert_eq!(
        selection.target_coverage_fraction,
        base.target_coverage
            .covered_fraction
            .min(heldout.target_coverage.covered_fraction)
            .min(extra.target_coverage.covered_fraction)
    );
    assert_eq!(
        selection.material_visible_target_coverage_fraction,
        base.material_visible_target_coverage
            .covered_fraction
            .min(heldout.material_visible_target_coverage.covered_fraction)
            .min(extra.material_visible_target_coverage.covered_fraction)
    );
    assert_eq!(
        selection.surface_normal_covered_bin_fraction,
        base.surface_normal_coverage
            .covered_target_bin_fraction
            .min(heldout.surface_normal_coverage.covered_target_bin_fraction)
            .min(extra.surface_normal_coverage.covered_target_bin_fraction)
    );
    assert_eq!(
        selection.surface_normal_mean_bin_covered_fraction,
        base.surface_normal_coverage
            .mean_bin_covered_fraction
            .min(heldout.surface_normal_coverage.mean_bin_covered_fraction)
            .min(extra.surface_normal_coverage.mean_bin_covered_fraction)
    );
    assert_eq!(
        selection.min_final_active_count,
        base.final_active_count
            .min(heldout.final_active_count)
            .min(extra.final_active_count)
    );
    assert_eq!(
        selection.min_newly_activated_fraction,
        base.newly_activated_fraction
            .min(heldout.newly_activated_fraction)
            .min(extra.newly_activated_fraction)
    );
    assert_eq!(
        selection.min_front_local_newly_activated_fraction,
        base.front_local_newly_activated_fraction
            .min(heldout.front_local_newly_activated_fraction)
            .min(extra.front_local_newly_activated_fraction)
    );
    assert_eq!(
        selection.max_front_liveness_margin,
        base.front_liveness
            .weighted_activation_margin
            .max(heldout.front_liveness.weighted_activation_margin)
            .max(extra.front_liveness.weighted_activation_margin)
    );
    assert_eq!(
        selection.min_front_liveness_candidate_count,
        base.front_liveness
            .candidate_count
            .min(heldout.front_liveness.candidate_count)
            .min(extra.front_liveness.candidate_count)
    );
    assert_eq!(
        selection.max_temporal_front_liveness_margin,
        base.temporal_front_liveness
            .weighted_activation_margin
            .max(heldout.temporal_front_liveness.weighted_activation_margin)
            .max(extra.temporal_front_liveness.weighted_activation_margin)
    );
    assert_eq!(
        selection.min_temporal_front_liveness_candidate_count,
        base.temporal_front_liveness
            .candidate_count
            .min(heldout.temporal_front_liveness.candidate_count)
            .min(extra.temporal_front_liveness.candidate_count)
    );
    assert_eq!(
        selection.all_temporal_activation_progressive,
        base.temporal_activation_progressive
            && heldout.temporal_activation_progressive
            && extra.temporal_activation_progressive
    );
    assert_eq!(
        selection.all_temporal_geometry_progressive,
        base.temporal_geometry_progressive
            && heldout.temporal_geometry_progressive
            && extra.temporal_geometry_progressive
    );
    assert!(
        selection.score >= selection.render_loss,
        "morphology penalty should never reduce the render objective"
    );
    assert!(
        selection.morphology_non_regressed,
        "unchanged model should not regress against its own baseline"
    );
}

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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
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
        material_opacity: passing_growth_3d_opacity_stats(),
        material_liveness: passing_material_liveness_report(),
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

#[test]
fn render_selection_candidate_requires_morphology_and_bounded_render_regression() {
    assert!(render_selection_candidate_beats(
        0.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(!render_selection_candidate_beats(
        0.5, 1.0, false, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(!render_selection_candidate_beats(
        1.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
    ));
    assert!(
        !render_selection_candidate_beats(0.98, 1.0, true, 0.91, 0.9, 2.0, 1.5),
        "weak strict score improvement should not spend render slack"
    );
    assert!(
        render_selection_candidate_beats(0.5, 1.0, true, 0.92, 0.9, 1.35, 1.5),
        "material strict score improvement can accept bounded render/density slack"
    );
    assert!(
        !render_selection_candidate_beats(0.5, 1.0, true, 0.95, 0.9, 2.0, 1.5),
        "strict score improvement should not accept large render loss regression"
    );
    assert!(
        !render_selection_candidate_beats(0.5, 1.0, true, 0.92, 0.9, 1.2, 1.5),
        "strict score improvement should not accept large density PSNR regression"
    );
}

#[test]
fn render_selection_candidate_can_retain_bounded_liveness_precursor_progress() {
    let best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 5.686245);
    let improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 3.6128867);

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded local-front liveness progress"
    );

    let mut morphology_regressed = improved.clone();
    morphology_regressed.morphology_non_regressed = false;
    assert!(!render_selection_candidate_metrics_beats(
        &morphology_regressed,
        &best
    ));

    let mut render_regressed = improved.clone();
    render_regressed.render_loss = 0.930;
    assert!(!render_selection_candidate_metrics_beats(
        &render_regressed,
        &best
    ));

    let mut weak_front_progress = improved;
    weak_front_progress.max_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_front_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_temporal_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_front_liveness_margin = 5.686245;
    best.min_temporal_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_temporal_front_liveness_margin = 3.6128867;
    improved.min_temporal_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded temporal-front liveness progress"
    );

    let mut weak_temporal_progress = improved;
    weak_temporal_progress.max_temporal_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_temporal_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_extent_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_extent_front_liveness_margin = 5.686245;
    best.min_extent_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_extent_front_liveness_margin = 3.6128867;
    improved.min_extent_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should be able to accumulate bounded extent-front liveness progress"
    );

    let mut morphology_regressed = improved.clone();
    morphology_regressed.morphology_non_regressed = false;
    assert!(!render_selection_candidate_metrics_beats(
        &morphology_regressed,
        &best
    ));

    let mut weak_extent_progress = improved;
    weak_extent_progress.max_extent_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_extent_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_bounded_temporal_extent_front_liveness_progress() {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_extent_front_liveness_margin = 5.686245;
    best.min_temporal_extent_front_liveness_candidate_count = 12;
    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.max_temporal_extent_front_liveness_margin = 3.6128867;
    improved.min_temporal_extent_front_liveness_candidate_count = 12;

    assert!(
        !render_selection_candidate_beats(
            improved.score,
            best.score,
            improved.morphology_non_regressed,
            improved.render_loss,
            best.render_loss,
            improved.density_psnr_db,
            best.density_psnr_db,
        ),
        "the scalar strict selector should still reject weak score improvements with tiny density regression"
    );
    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "metric-aware selection should retain bounded temporal extent-front liveness progress"
    );

    let mut timing_regressed = improved.clone();
    timing_regressed.max_temporal_activation_schedule_error = best
        .max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
        + 0.01;
    timing_regressed.morphology_non_regressed = false;
    assert!(
        !render_selection_candidate_metrics_beats(&timing_regressed, &best),
        "bounded temporal extent-front progress should not hide temporal activation regression"
    );

    let mut weak_temporal_extent_progress = improved;
    weak_temporal_extent_progress.max_temporal_extent_front_liveness_margin = 5.66;
    assert!(!render_selection_candidate_metrics_beats(
        &weak_temporal_extent_progress,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_carry_bounded_temporal_front_precursor_through_morphology_failure()
 {
    let mut best = render_selection_metrics_with_liveness(107.10512, 0.925306, 0.3583453, 0.0);
    best.max_temporal_front_liveness_margin = 5.686245;
    best.min_temporal_front_liveness_candidate_count = 12;
    best.max_temporal_activation_schedule_error = 0.20;

    let mut improved =
        render_selection_metrics_with_liveness(107.10004, 0.92530984, 0.3582421, 0.0);
    improved.morphology_non_regressed = false;
    improved.max_temporal_front_liveness_margin = 3.6128867;
    improved.min_temporal_front_liveness_candidate_count = 12;
    improved.max_temporal_activation_schedule_error = 0.20;
    improved.active_surface_max = 0.25;

    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "bounded temporal-front liveness progress should survive temporary strict morphology failure"
    );

    let mut activated_burst = improved.clone();
    activated_burst.min_final_active_count = 32;
    activated_burst.min_newly_activated_fraction = 1.0;
    activated_burst.min_front_local_newly_activated_fraction = 1.0;
    assert!(
        !render_selection_candidate_metrics_beats(&activated_burst, &best),
        "temporal-front precursor selection must not carry an all-active burst through morphology failure"
    );

    let mut escaped = improved.clone();
    escaped.active_surface_max = GROWTH_3D_SURFACE_MAX_DISTANCE + 0.10;
    assert!(!render_selection_candidate_metrics_beats(&escaped, &best));

    let mut timing_regressed = improved.clone();
    timing_regressed.max_temporal_activation_schedule_error = best
        .max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK
        + 0.01;
    assert!(!render_selection_candidate_metrics_beats(
        &timing_regressed,
        &best
    ));

    let mut render_regressed = improved;
    render_regressed.render_loss = 0.930;
    assert!(!render_selection_candidate_metrics_beats(
        &render_regressed,
        &best
    ));
}

#[test]
fn render_selection_candidate_can_retain_local_activation_breakthrough() {
    let best = render_selection_metrics_with_liveness(107.08897, 0.92548585, 0.35743254, 1.5105798);
    let mut activated = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    activated.morphology_non_regressed = false;
    activated.min_final_active_count = 32;
    activated.min_newly_activated_fraction = 1.0;
    activated.min_front_local_newly_activated_fraction = 1.0;
    activated.active_surface_max = 0.25;
    activated.all_temporal_activation_progressive = true;

    assert!(
        render_selection_candidate_metrics_beats(&activated, &best),
        "a bounded local-front activation breakthrough with improved render metrics should be retained for continued training"
    );

    let mut escaped = activated.clone();
    escaped.active_surface_max = GROWTH_3D_SURFACE_MAX_DISTANCE + 0.10;
    assert!(!render_selection_candidate_metrics_beats(&escaped, &best));

    let mut nonlocal = activated;
    nonlocal.min_front_local_newly_activated_fraction = 0.0;
    assert!(!render_selection_candidate_metrics_beats(&nonlocal, &best));
}

#[test]
fn render_selection_training_progress_can_continue_non_promotable_refinement() {
    let mut previous = render_selection_metrics_with_liveness(102.6, 0.838, 0.35, 1.5105798);
    previous.active_surface_max = 0.43;
    previous.min_active_extent_bbox_ratio = 0.35;
    previous.min_active_extent_min_axis_ratio = 0.22;
    previous.min_final_active_count = 54;
    previous.min_newly_activated_fraction = 0.18;
    previous.min_front_local_newly_activated_fraction = 0.92;
    previous.max_temporal_activation_schedule_error = 0.12;
    previous.surface_covered_bin_fraction = 0.06;
    previous.surface_normal_covered_bin_fraction = 0.23;
    previous.material_visible_surface_covered_bin_fraction = 0.06;
    previous.material_visible_surface_normal_covered_bin_fraction = 0.23;
    previous.material_visible_target_mean_distance = 0.86;

    let mut continued = previous.clone();
    continued.morphology_non_regressed = false;
    continued.score = 110.5;
    continued.render_loss = 0.790;
    continued.density_psnr_db = 0.39;
    continued.active_surface_max = 0.61;
    continued.min_active_extent_bbox_ratio = 0.95;
    continued.min_active_extent_min_axis_ratio = 0.88;
    continued.min_final_active_count = 205;
    continued.min_newly_activated_fraction = 0.79;
    continued.min_front_local_newly_activated_fraction = 0.62;
    continued.max_temporal_activation_schedule_error = 0.138;
    continued.surface_covered_bin_fraction = 0.31;
    continued.surface_normal_covered_bin_fraction = 0.54;
    continued.material_visible_surface_covered_bin_fraction = 0.25;
    continued.material_visible_surface_normal_covered_bin_fraction = 0.50;
    continued.material_visible_target_mean_distance = 0.89;

    assert!(
        !render_selection_candidate_metrics_beats(&continued, &previous),
        "continuation should not be promoted as a strict selected checkpoint"
    );
    assert!(
        render_selection_training_progress_beats(&continued, &previous),
        "bounded render, coverage, extent, and activation progress should continue training even before strict gates pass"
    );

    let mut bursty = continued.clone();
    bursty.active_surface_max = 0.90;
    bursty.max_temporal_activation_schedule_error = 0.27;
    bursty.material_visible_surface_tail_over_threshold_fraction = 0.125;
    bursty.min_front_local_newly_activated_fraction = 0.48;
    assert!(
        !render_selection_training_progress_beats(&bursty, &previous),
        "training continuation must still reject global activation/projection shortcuts"
    );
}

#[test]
fn render_selection_training_progress_rejects_morphology_only_continuation() {
    let previous = render_selection_metrics_with_liveness(125.047, 0.87312, 0.6845, 0.0);
    let mut unchanged = previous.clone();
    unchanged.morphology_non_regressed = true;

    assert!(
        !render_selection_training_progress_beats(&unchanged, &previous),
        "line search should not continue from a morphology-only no-op candidate"
    );

    let mut render_only = previous.clone();
    render_only.morphology_non_regressed = true;
    render_only.render_loss = previous.render_loss - 0.010;
    render_only.density_psnr_db = previous.density_psnr_db + 0.10;
    assert!(
        !render_selection_training_progress_beats(&render_only, &previous),
        "render-only improvement should not count as rollout training progress without geometry/material/activation progress"
    );

    let mut coverage_progress = render_only;
    coverage_progress.surface_covered_bin_fraction = previous.surface_covered_bin_fraction + 0.06;
    assert!(
        render_selection_training_progress_beats(&coverage_progress, &previous),
        "bounded morphology-preserving render plus coverage progress should still continue training"
    );
}

#[test]
fn render_selection_morphology_recovery_requires_strict_score_improvement() {
    let mut regressed = render_selection_metrics_with_liveness(125.047, 0.87312, 0.6845, 0.0);
    regressed.morphology_non_regressed = false;

    let mut same_score_recovery = regressed.clone();
    same_score_recovery.morphology_non_regressed = true;
    same_score_recovery.render_loss = regressed.render_loss - 0.001;
    same_score_recovery.density_psnr_db = regressed.density_psnr_db + 0.001;
    assert!(
        !render_selection_morphology_recovery_beats(&same_score_recovery, &regressed),
        "line search should not accept morphology recovery without strict-score improvement"
    );

    let mut strict_recovery = same_score_recovery;
    strict_recovery.score = regressed.score - 0.01;
    assert!(
        render_selection_morphology_recovery_beats(&strict_recovery, &regressed),
        "bounded morphology recovery with strict-score and render non-regression should stay eligible"
    );

    let mut render_regressed = strict_recovery;
    render_regressed.render_loss = regressed.render_loss + 0.02;
    assert!(
        !render_selection_morphology_recovery_beats(&render_regressed, &regressed),
        "strict-score recovery should not hide render regression"
    );
}

#[test]
fn render_selection_candidate_can_retain_bounded_material_precursor() {
    let mut best = render_selection_metrics_with_liveness(77.0, 0.9188, 0.398, 0.0);
    best.min_final_active_count = 32;
    best.material_active_mean_opacity = -3.60;
    best.material_visible_count = 1;

    let mut improved = best.clone();
    improved.material_active_mean_opacity = -3.40;
    improved.render_loss = 0.91881;
    improved.density_psnr_db = 0.39799;

    assert!(
        render_selection_candidate_metrics_beats(&improved, &best),
        "bounded material opacity precursor progress should be retained before visibility coverage gates flip"
    );

    let mut tail_regressed = improved.clone();
    tail_regressed.material_visible_surface_tail_over_threshold_fraction = 0.02;
    assert!(!render_selection_candidate_metrics_beats(
        &tail_regressed,
        &best
    ));

    let mut activation_regressed = improved;
    activation_regressed.min_final_active_count = 16;
    assert!(!render_selection_candidate_metrics_beats(
        &activation_regressed,
        &best
    ));
}

#[test]
fn render_selection_rejects_bursty_activation_breakthrough_timing_regression() {
    let mut best = render_selection_metrics_with_liveness(81.15844, 0.925548, 0.3570, 1.5105798);
    best.max_temporal_activation_schedule_error = 0.18607144;
    best.all_temporal_activation_progressive = false;

    let mut bursty = render_selection_metrics_with_liveness(75.13471, 0.91643715, 0.3985, 0.0);
    bursty.morphology_non_regressed = false;
    bursty.min_final_active_count = 32;
    bursty.min_newly_activated_fraction = 1.0;
    bursty.min_front_local_newly_activated_fraction = 1.0;
    bursty.active_surface_max = 0.25;
    bursty.max_temporal_activation_schedule_error = 0.34142855;

    assert!(
        !render_selection_candidate_metrics_beats(&bursty, &best),
        "render and activation breakthroughs must not hide a worse growth schedule"
    );

    bursty.max_temporal_activation_schedule_error = best.max_temporal_activation_schedule_error
        + TEMPORAL_ACTIVATION_SELECTION_REGRESSION_SLACK;
    assert!(
        !render_selection_candidate_metrics_beats(&bursty, &best),
        "timing-neutral all-active bursts still fail unless the temporal rollout is progressive"
    );

    bursty.all_temporal_activation_progressive = true;
    assert!(
        render_selection_candidate_metrics_beats(&bursty, &best),
        "temporally progressive activation breakthroughs remain valid"
    );
}

#[test]
fn render_selection_candidate_can_refine_after_activation_breakthrough() {
    let mut best = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    best.morphology_non_regressed = false;
    best.min_final_active_count = 32;
    best.min_newly_activated_fraction = 1.0;
    best.min_front_local_newly_activated_fraction = 1.0;
    best.active_surface_max = 0.25;
    best.all_temporal_activation_progressive = true;

    let mut refined = best.clone();
    refined.score = 66.73771;
    refined.render_loss = 0.91643715;
    refined.density_psnr_db = 0.39857513;

    assert!(
        render_selection_candidate_metrics_beats(&refined, &best),
        "after a retained activation breakthrough, continued bounded render/strict-score refinement should not be blocked by the initial breakthrough morphology flag"
    );

    let mut lost_activation = refined;
    lost_activation.min_newly_activated_fraction = 0.50;
    assert!(!render_selection_candidate_metrics_beats(
        &lost_activation,
        &best
    ));
}

#[test]
fn render_selection_rejects_post_activation_refinement_timing_regression() {
    let mut best = render_selection_metrics_with_liveness(76.6387, 0.9238937, 0.36455846, 0.0);
    best.morphology_non_regressed = false;
    best.min_final_active_count = 32;
    best.min_newly_activated_fraction = 1.0;
    best.min_front_local_newly_activated_fraction = 1.0;
    best.active_surface_max = 0.25;
    best.max_temporal_activation_schedule_error = 0.20;
    best.all_temporal_activation_progressive = true;

    let mut refined = best.clone();
    refined.score = 66.73771;
    refined.render_loss = 0.91643715;
    refined.density_psnr_db = 0.39857513;
    refined.max_temporal_activation_schedule_error = 0.35;
    refined.all_temporal_activation_progressive = false;

    assert!(
        !render_selection_candidate_metrics_beats(&refined, &best),
        "post-activation refinement should also preserve temporal growth timing"
    );
}

#[test]
fn local_front_liveness_progress_measures_dormant_activation_margin() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.1_f32, 0.0, 0.0, 0.0],
        [0.5_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = -3.0;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let progress = local_front_liveness_progress(&config, &positions, &states, 0.2);

    assert_eq!(progress.candidate_count, 1);
    assert!(
        (progress.weighted_activation_margin - 2.0).abs() < 1.0e-5,
        "near dormant front particle is two logits below the active threshold"
    );
}

#[test]
fn render_selection_score_rewards_lower_local_front_liveness_margin() {
    let weak = render_selection_case_with_front_liveness_margin(8.0);
    let better = render_selection_case_with_front_liveness_margin(2.0);

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold local-front liveness progress before strict activation gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.temporal_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.temporal_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold temporal-front liveness progress before strict activation gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_extent_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see sub-threshold extent-front liveness progress before strict active-extent gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_extent_front_liveness_margin() {
    let mut weak = render_selection_case_with_front_liveness_margin(0.0);
    weak.temporal_extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 8.0,
    };
    let mut better = render_selection_case_with_front_liveness_margin(0.0);
    better.temporal_extent_front_liveness = LocalFrontLivenessProgress {
        candidate_count: 4,
        weighted_activation_margin: 2.0,
    };

    let weak_score = render_selection_case_score_with_baseline(7, &weak, None);
    let better_score = render_selection_case_score_with_baseline(7, &better, None);

    assert!(weak_score.morphology_non_regressed);
    assert!(better_score.morphology_non_regressed);
    assert!(
        better_score.score < weak_score.score,
        "line search should see temporal extent-front progress before strict active-extent gates flip"
    );
    assert!(
        (weak_score.score - better_score.score - 6.0 * LOCAL_FRONT_LIVENESS_SCORE_WEIGHT).abs()
            < 1.0e-6
    );
}

#[test]
fn render_selection_score_rewards_lower_temporal_activation_schedule_error() {
    let mut abrupt = render_selection_case_with_front_liveness_margin(0.0);
    abrupt.temporal_activation_schedule_error = 0.40;
    let mut staged = render_selection_case_with_front_liveness_margin(0.0);
    staged.temporal_activation_schedule_error = 0.05;

    let abrupt_score = render_selection_case_score_with_baseline(7, &abrupt, None);
    let staged_score = render_selection_case_score_with_baseline(7, &staged, None);

    assert!(abrupt_score.morphology_non_regressed);
    assert!(staged_score.morphology_non_regressed);
    assert!(
        staged_score.score < abrupt_score.score,
        "selection should prefer rollouts whose activation follows the schedule"
    );
    assert!(
        (abrupt_score.score - staged_score.score - 0.35 * TEMPORAL_ACTIVATION_SCORE_WEIGHT).abs()
            < 1.0e-5
    );
}

#[test]
fn render_selection_score_penalizes_active_extent_regression() {
    let baseline_case = render_selection_case_with_front_liveness_margin(0.0);
    let baseline = vec![render_selection_baseline_case_from_metrics(
        7,
        &baseline_case,
    )];
    let mut collapsed = render_selection_case_with_front_liveness_margin(0.0);
    collapsed.extent.bbox_diagonal_ratio = 0.12;
    collapsed.extent.min_axis_extent_ratio = 0.02;

    let scored = render_selection_case_score_with_baseline(7, &collapsed, Some(&baseline));

    assert!(!scored.morphology_non_regressed);
    assert!(
        scored.score > collapsed.score + 1.0,
        "active extent regression should contribute to selection penalty"
    );
}

#[test]
fn render_training_objective_config_records_hybrid_3d_weights() {
    let config = NpaConfig::growing_3dgs();
    let render = RenderLossConfig {
        density_weight: 1.25,
        color_weight: 0.75,
        depth_weight: 1.5,
        ..RenderLossConfig::default()
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 128,
        rollout_steps: 2,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.1,
        perception_position_gain: 0.05,
        max_update_norm: 0.1,
        trajectory_supervision: true,
        trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
        trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
        trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
        liveness_gain: ROBUST_3D_LIVENESS_GAIN,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: ROBUST_3D_COVERAGE_GAIN,
        coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
        coverage_mode: CoverageUpdateModeArg::SlicedOt,
        coverage_softness: 0.0,
        coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
        extent_gain: ROBUST_3D_EXTENT_GAIN,
        full_coverage_adjoint: false,
        surface_gain: ROBUST_3D_SURFACE_GAIN,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: ROBUST_3D_OPACITY_GAIN,
        material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: ROBUST_3D_SCALE_GAIN,
        scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 11,
        selection_seed: Some(19),
        selection_seeds: vec![23],
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render,
        sgd: SgdConfig {
            learning_rate: 1.0e-4,
            grad_clip_norm: 0.1,
            weight_decay: 0.0,
        },
    };

    let objective = render_training_objective_config(&cfg, render);

    assert_eq!(config.spatial_dims, 3);
    assert_eq!(objective.render_density_weight, 1.25);
    assert!(objective.coverage_gain > 0.0);
    assert!(objective.coverage_samples >= 1024);
    assert!(objective.coverage_normal_weight > 0.0);
    assert!(objective.coverage_repulsion_gain > 0.0);
    assert!(objective.extent_gain > 0.0);
    assert!(objective.surface_gain > 0.0);
    assert!(objective.trajectory_render_gain > 0.0);
    assert!(objective.trajectory_mesh_gain > 0.0);
    assert!(objective.liveness_gain > 0.0);
    assert_eq!(objective.phase_gain, ROBUST_3D_PHASE_GAIN);
    assert!(objective.liveness_front_radius > 0.0);
    assert_eq!(
        direct_terminal_liveness_gain(&cfg),
        ROBUST_3D_LIVENESS_GAIN * DIRECT_TRAJECTORY_TERMINAL_LIVENESS_WEIGHT,
        "direct rollout training should include a small terminal active-count anchor without replacing trajectory timing"
    );
    let mut terminal_only_cfg = cfg.clone();
    terminal_only_cfg.trajectory_supervision = false;
    assert_eq!(
        direct_terminal_liveness_gain(&terminal_only_cfg),
        ROBUST_3D_LIVENESS_GAIN
    );
    assert_eq!(
        objective.liveness_update_multiplier,
        ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER
    );
    assert!(objective.opacity_gain > 0.0);
    assert_eq!(
        objective.material_liveness_gain,
        ROBUST_3D_MATERIAL_LIVENESS_GAIN
    );
    assert_eq!(objective.material_tail_gain, ROBUST_3D_MATERIAL_TAIL_GAIN);
    assert_eq!(
        objective.material_suppression_update_multiplier,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER
    );
    assert_eq!(
        objective.material_max_opacity_update,
        ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE
    );
    assert!(objective.gaussian_scale_gain > 0.0);
    assert!(objective.gaussian_scale_budget_weight > 0.0);
}

#[test]
fn direct_growth_phase_gain_has_floor_without_forcing_liveness() {
    let config = NpaConfig::growing_3dgs();
    let render = RenderLossConfig::default();
    let mut cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 128,
        rollout_steps: 2,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.1,
        perception_position_gain: 0.05,
        max_update_norm: 0.1,
        trajectory_supervision: true,
        trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
        trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
        trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
        liveness_gain: ROBUST_3D_LIVENESS_GAIN,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: ROBUST_3D_COVERAGE_GAIN,
        coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
        coverage_mode: CoverageUpdateModeArg::SlicedOt,
        coverage_softness: 0.0,
        coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
        extent_gain: ROBUST_3D_EXTENT_GAIN,
        full_coverage_adjoint: false,
        surface_gain: ROBUST_3D_SURFACE_GAIN,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: ROBUST_3D_OPACITY_GAIN,
        material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: ROBUST_3D_SCALE_GAIN,
        scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 11,
        selection_seed: Some(19),
        selection_seeds: vec![23],
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render,
        sgd: SgdConfig {
            learning_rate: 1.0e-4,
            grad_clip_norm: 0.1,
            weight_decay: 0.0,
        },
    };

    assert_eq!(config.spatial_dims, 3);
    assert_eq!(direct_growth_phase_gain(&cfg), ROBUST_3D_PHASE_GAIN);

    cfg.liveness_gain = 0.0;
    assert_eq!(
        direct_growth_phase_gain(&cfg),
        0.0,
        "disabling liveness training should still disable phase progression pressure"
    );
}

#[test]
fn material_suppression_cap_is_stronger_than_growth_opacity_cap() {
    let cap =
        material_suppression_max_update(0.05, ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER);

    assert_eq!(cap, 0.05 * ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER);
    assert!(cap > 0.05);
    assert_eq!(material_suppression_max_update(0.05, 3.5), 0.175);
}

#[test]
fn liveness_cap_is_stronger_than_growth_opacity_cap() {
    let cap = liveness_max_update(0.05, ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER);

    assert_eq!(cap, 0.05 * ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER);
    assert!(cap > 0.05);
    assert_eq!(liveness_max_update(0.05, 3.5), 0.175);
}

#[test]
fn robust_liveness_cap_can_cross_activation_threshold_within_short_rollout() {
    let cap = liveness_max_update(0.05, ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER);
    let dormant_to_active_span = -1.0 - GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    assert!(
        cap * 8.0 >= dormant_to_active_span,
        "robust 3D local-front liveness targets should be able to cross the active threshold within an 8-step local wave"
    );
    assert!(
        liveness_max_update(0.05, 5.0) * 16.0 < dormant_to_active_span,
        "legacy cap could not cross within a 16-step horizon, which was too slow for staged local growth"
    );
}
