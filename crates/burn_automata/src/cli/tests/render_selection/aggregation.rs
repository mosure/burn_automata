use super::*;

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
