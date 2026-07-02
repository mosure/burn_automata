use super::*;

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
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
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
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
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
