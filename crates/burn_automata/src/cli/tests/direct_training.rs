use super::*;

#[test]
fn render_direct_rollout_backend_applies_mlp_gradients() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 19, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 32,
        rollout_steps: 2,
        gradient_particles: 32,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 1.0,
        max_update_norm: 0.05,
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
        surface_escape_gain: 0.0,
        opacity_gain: 0.0,
        material_liveness_gain: 0.0,
        material_tail_gain: 0.0,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: 0.0,
        scale_budget_weight: 0.0,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: true,
        direct_line_search_scales: vec![0.5, 1.0, 2.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 13,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.72,
        seed_mode: ParticleSeed::UniformCircle,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 128,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };
    let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let gradient = RenderProxyGradientRows {
        row_indices: (0..cfg.particles).collect(),
        gradients: vec![[1.0, 0.25, -0.5]; cfg.particles],
        opacity_gradients: vec![0.0; cfg.particles],
        scale_gradients: vec![0.0; cfg.particles],
        color_gradients: vec![[0.1, -0.2, 0.05]; cfg.particles],
    };
    let report = render_direct_rollout_training_step(
        &mut model,
        &grid,
        &target,
        &trace,
        &trajectory,
        &gradient,
        &cfg,
        cfg.seed,
    )
    .unwrap();

    assert_eq!(report.rows, cfg.particles * cfg.rollout_steps);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert_eq!(report.best_loss, report.initial_loss.min(report.final_loss));
    assert_eq!(report.history[0].loss, report.final_loss);
    assert!(report.history[0].grad_norm.is_finite());
    assert!(report.history[0].grad_norm > 0.0);
    assert_ne!(model.weights.w2, before);
}

#[test]
fn render_proxy_history_records_direct_line_search_candidates() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 21, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 32,
        rollout_steps: 2,
        gradient_particles: 32,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 1.0,
        max_update_norm: 0.05,
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
        surface_escape_gain: 0.0,
        opacity_gain: 0.0,
        material_liveness_gain: 0.0,
        material_tail_gain: 0.0,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: 0.0,
        scale_budget_weight: 0.0,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: true,
        direct_line_search_scales: vec![0.5, 1.0, 2.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 17,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.72,
        seed_mode: ParticleSeed::UniformCircle,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 64,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();
    let history = &report.history[0];
    let candidates = &history.direct_line_search_candidates;

    assert_eq!(candidates.len(), 3);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.scale)
            .collect::<Vec<_>>(),
        vec![0.5, 1.0, 2.0]
    );
    assert!(candidates.iter().all(|candidate| candidate.inner_step == 0));
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.render_loss.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.score.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.train_final_loss.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.train_grad_norm.is_finite())
    );
    let selected_count = candidates
        .iter()
        .filter(|candidate| candidate.selected_checkpoint || candidate.selected_progress)
        .count();
    assert!(selected_count <= 1);
    let serialized = serde_json::to_value(history).unwrap();
    assert_eq!(
        serialized["direct_line_search_candidates"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn direct_rollout_objective_diagnostics_reports_channel_pressure() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = local_growth_student_model(config, 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 32,
        rollout_steps: 4,
        gradient_particles: 8,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
        trajectory_render_samples: 0,
        liveness_gain: 1.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: 20.0,
        coverage_gain: ROBUST_3D_COVERAGE_GAIN,
        coverage_samples: 32,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: ROBUST_3D_EXTENT_GAIN,
        full_coverage_adjoint: false,
        surface_gain: ROBUST_3D_SURFACE_GAIN,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: ROBUST_3D_OPACITY_GAIN,
        material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        material_tail_gain: 0.0,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: 0.0,
        scale_budget_weight: 0.0,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: 0.125,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 37,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusLocalSubstrateGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: 1.08,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };
    let (_, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let diagnostics = direct_rollout_objective_diagnostics(&model, &target, &trajectory, &cfg)
        .expect("diagnostics should evaluate direct objective pressure");

    assert_eq!(diagnostics.snapshots, cfg.rollout_steps);
    assert_eq!(diagnostics.rows, cfg.particles * cfg.rollout_steps);
    assert!(
        diagnostics.temporal_liveness_rms > 0.0,
        "temporal liveness objective should be visible before MLP backprop"
    );
    assert!(
        diagnostics.terminal_liveness_state_rms > 0.0,
        "terminal liveness anchor should be visible in state-adjoint diagnostics"
    );
    assert!(diagnostics.terminal_liveness_state_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_liveness_rms > 0.0,
        "mesh-motion liveness objective should activate moving local-front rows"
    );
    assert!(diagnostics.mesh_motion_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.target_coverage_liveness_rms > 0.0,
        "target-coverage liveness objective should activate coverage-critical local-front rows"
    );
    assert!(diagnostics.target_coverage_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_liveness_rms > 0.0,
        "material-coverage liveness objective should activate local-front rows that improve visible support"
    );
    assert!(diagnostics.material_coverage_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.phase_rms > 0.0,
        "phase objective should be visible before MLP backprop"
    );
    assert!(
        diagnostics.liveness_phase_memory_rms > 0.0,
        "liveness pressure should also supervise recurrent phase-memory outputs"
    );
    assert!(diagnostics.liveness_phase_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_rms > 0.0,
        "trajectory mesh objective should produce motion-channel pressure"
    );
    assert!(
        diagnostics.extent_front_motion_rms > 0.0,
        "extent-front objective should produce outward dormant-front motion pressure"
    );
    assert!(diagnostics.extent_front_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.temporal_extent_motion_rms > 0.0,
        "temporal extent objective should produce scheduled expansion pressure"
    );
    assert!(diagnostics.temporal_extent_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.extent_motion_memory_rms > 0.0,
        "extent-front motion should also supervise recurrent velocity-memory outputs"
    );
    assert!(diagnostics.extent_motion_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_motion_rms > 0.0,
        "material-coverage motion objective should produce visible-support motion pressure"
    );
    assert!(diagnostics.material_coverage_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_surface_motion_rms > 0.0,
        "material visible-surface objective should produce active/front surface motion pressure"
    );
    assert!(diagnostics.material_surface_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.residual_velocity_rms > 0.0,
        "mesh residual objective should supervise velocity state outputs directly"
    );
    assert!(diagnostics.residual_velocity_nonzero_fraction > 0.0);
    assert!(
        diagnostics.motion_memory_rms > 0.0,
        "trajectory mesh objective should also supervise velocity-memory state outputs"
    );
    assert!(
        diagnostics.material_coverage_motion_memory_rms > 0.0,
        "material-coverage motion should also supervise recurrent velocity-memory outputs"
    );
    assert!(diagnostics.material_coverage_motion_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_materialization_rms > 0.0,
        "material-coverage candidates should train render-material outputs directly"
    );
    assert!(diagnostics.material_coverage_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.temporal_materialization_rms > 0.0,
        "temporal materialization should train material outputs for scheduled local-front growth"
    );
    assert!(diagnostics.temporal_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.active_surface_materialization_rms > 0.0,
        "active surface materialization should train near-surface active material outputs"
    );
    assert!(diagnostics.active_surface_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_visibility_rms > 0.0,
        "material visibility objective should produce material-channel pressure"
    );
    assert!(diagnostics.combined_pre_cap_rms >= diagnostics.combined_post_cap_rms);
    assert!(diagnostics.mesh_motion_post_cap_rms > 0.0);
    assert!(diagnostics.mesh_motion_post_cap_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_post_cap_rms >= cfg.direct_output_gradient_rms_cap * 0.5,
        "combined direct gradients should preserve spatial motion pressure instead of letting liveness/material channels dominate"
    );
    assert!(diagnostics.residual_velocity_post_cap_rms > 0.0);
    assert!(diagnostics.residual_velocity_post_cap_nonzero_fraction > 0.0);
    assert!(diagnostics.motion_memory_post_cap_rms > 0.0);
    assert!(diagnostics.motion_memory_post_cap_nonzero_fraction > 0.0);
    assert!(diagnostics.liveness_post_cap_rms > 0.0);
    assert!(diagnostics.phase_post_cap_rms > 0.0);
    assert!(diagnostics.material_post_cap_rms > 0.0);
}

#[test]
fn direct_rollout_training_honors_supervised_steps_per_round() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 23, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 3,
        particles: 16,
        rollout_steps: 1,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 16,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: 0.0,
        full_coverage_adjoint: false,
        surface_gain: 0.0,
        surface_escape_gain: 0.0,
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
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 29,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: 1.08,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 5.0e-4,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };
    let baseline = render_selection_baseline(&model, &grid, &target, &cfg, cfg.render).unwrap();
    let (report, step_scale, line_search_candidates) = render_direct_rollout_training_steps(
        &mut model, &grid, &target, &cfg, 0, cfg.render, &baseline,
    )
    .unwrap();

    assert_eq!(step_scale, 1.0);
    assert!(line_search_candidates.is_empty());
    assert_eq!(report.steps, cfg.supervised_steps_per_round);
    assert_eq!(report.history.len(), cfg.supervised_steps_per_round);
    assert_eq!(
        report.rows,
        cfg.supervised_steps_per_round * cfg.particles * cfg.rollout_steps
    );
    assert_eq!(report.history[0].step, 1);
    assert_eq!(report.history[2].step, 3);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(report.best_loss.is_finite());
    assert!(report.history.iter().all(|entry| entry.loss.is_finite()));
    assert_ne!(model.weights.w2, before);
}

#[test]
fn direct_multiseed_training_reports_actual_averaged_model_loss() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 27, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 16,
        rollout_steps: 2,
        gradient_particles: 8,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.05,
        trajectory_render_samples: 2,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.1,
        coverage_samples: 32,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.1,
        extent_gain: 0.0,
        full_coverage_adjoint: true,
        surface_gain: 0.1,
        surface_escape_gain: 0.0,
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
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: true,
        seed: 31,
        selection_seed: Some(37),
        selection_seeds: vec![43],
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: 1.08,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 5.0e-4,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };
    let seeds = render_direct_rollout_training_seeds(&cfg, 0);
    let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let gradient = render_position_gradient(&trace, &target, cfg.render, &cfg).unwrap();

    let report = render_direct_rollout_multiseed_training_step(
        &mut model,
        &grid,
        &target,
        &cfg,
        0,
        &trace,
        &trajectory,
        &gradient,
    )
    .unwrap();
    let actual_final_loss = render_direct_rollout_average_loss_for_seeds(
        &model, &grid, &target, &cfg, cfg.render, &seeds,
    )
    .unwrap();

    assert_eq!(seeds.len(), 3);
    assert_eq!(report.steps, 1);
    assert_eq!(report.history.len(), 1);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(
        (report.final_loss - actual_final_loss).abs() <= 1.0e-6,
        "multiseed report should evaluate the averaged model that is kept"
    );
    assert_eq!(report.history[0].loss, report.final_loss);
    assert_eq!(report.best_loss, report.initial_loss.min(report.final_loss));
    assert_ne!(model.weights.w2, before);
}

#[test]
fn render_proxy_history_records_direct_rollout_inner_steps() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 3,
        particles: 16,
        rollout_steps: 1,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 16,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: 0.0,
        full_coverage_adjoint: false,
        surface_gain: 0.0,
        surface_escape_gain: 0.0,
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
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 37,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: 1.08,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 5.0e-4,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();
    let round = &report.history[0];

    assert_eq!(report.history.len(), 1);
    assert_eq!(round.train_step_count, report.supervised_steps_per_round);
    assert_eq!(
        round.train_loss_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(
        round.train_grad_norm_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(
        round.train_grad_scale_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(round.supervised_loss, round.train_final_loss);
    assert_eq!(
        round.train_loss_history.last().copied().unwrap(),
        round.train_final_loss
    );
    assert_eq!(
        round.train_grad_norm_history.last().copied().unwrap(),
        round.train_grad_norm
    );
    assert_eq!(
        round.train_grad_scale_history.last().copied().unwrap(),
        round.train_grad_scale
    );
    assert!(round.train_loss_history.iter().all(|loss| loss.is_finite()));
    assert!(round.before_selection_loss.is_finite());
    assert!(round.before_selection_score.is_finite());
    assert!(round.before_selection_density_psnr_db.is_finite());
    assert!(round.selection_loss.is_finite());
    assert!(round.selection_score.is_finite());
    assert!(round.selection_density_psnr_db.is_finite());
    assert!(round.train_phase_output_delta_norm.is_finite());
    assert!(round.selection_min_final_active_count <= report.final_gaussian_volume.particles);
    assert!(round.selection_min_newly_activated_fraction.is_finite());
    assert!((0.0..=1.0).contains(&round.selection_min_newly_activated_fraction));
    assert!(
        round
            .selection_min_front_local_newly_activated_fraction
            .is_finite()
    );
    assert!((0.0..=1.0).contains(&round.selection_min_front_local_newly_activated_fraction));
    assert!(
        round.selection_worst_failure_reasons.is_empty()
            || round
                .selection_worst_failure_reasons
                .iter()
                .all(|reason| !reason.is_empty())
    );
}

#[test]
fn render_proxy_training_rolls_back_rejected_round_before_next_round() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 41, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let initial_model = model.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 2,
        supervised_steps_per_round: 1,
        particles: 16,
        rollout_steps: 1,
        gradient_particles: 4,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 16,
        coverage_mode: CoverageUpdateModeArg::HardNearest,
        coverage_softness: 0.0,
        coverage_repulsion_gain: 0.0,
        coverage_gap_gain: 0.0,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: 0.0,
        extent_gain: 0.0,
        full_coverage_adjoint: false,
        surface_gain: 0.0,
        surface_escape_gain: 0.0,
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
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 43,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 32,
            world_scale: 1.08,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1000.0,
            grad_clip_norm: 100.0,
            weight_decay: 0.0,
        },
    };
    let expected_round_one_trace = render_training_trace(&initial_model, &grid, &cfg, 1).unwrap();
    let expected_round_one_loss =
        mesh_multiview_render_loss_from_trace(&expected_round_one_trace, &target, cfg.render)
            .unwrap()
            .total_loss;

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();

    assert_eq!(report.history.len(), 2);
    assert!(
        report.history[0].rolled_back_to_best_checkpoint,
        "aggressive rejected first round should be rolled back before continuing"
    );
    assert!(!report.history[0].selected_checkpoint);
    assert_eq!(
        report.history[0].train_step_scale, 0.0,
        "rolled-back rounds should report the applied checkpoint scale, not the attempted update scale"
    );
    assert!(
        (report.history[1].before_loss - expected_round_one_loss).abs() <= 1.0e-6,
        "round two should start from the best checkpoint, not the rejected round-one weights"
    );
}

#[test]
fn terminal_position_adjoint_combines_render_and_coverage_gradients() {
    let config = NpaConfig::growing_3dgs();
    let trace = crate::RolloutTrace {
        positions: vec![[0.0; 4]; 3],
        states: vec![0.0; 3 * config.state_dims],
        batch_size: 1,
        particle_count: 3,
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![1],
        gradients: vec![[0.5, -0.25, 0.1]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![0.0],
        color_gradients: vec![[0.0; 3]],
    };
    let mut coverage = vec![[0.0; 3]; 3];
    coverage[1] = [0.1, 0.05, -0.2];
    coverage[2] = [0.2, 0.0, -0.1];

    let adjoint =
        terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 2.0, true, 1);

    assert_eq!(adjoint[0], [0.0; 4]);
    assert!((adjoint[1][0] - 0.9).abs() <= 1.0e-6);
    assert!((adjoint[1][1] + 0.55).abs() <= 1.0e-6);
    assert!((adjoint[1][2] - 0.4).abs() <= 1.0e-6);
    assert!((adjoint[2][0] + 0.2).abs() <= 1.0e-6);
    assert_eq!(adjoint[2][1], 0.0);
    assert!((adjoint[2][2] - 0.1).abs() <= 1.0e-6);
    assert_eq!(adjoint[2][3], 0.0);

    let sampled_only =
        terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 2.0, false, 1);
    assert_eq!(sampled_only[0], [0.0; 4]);
    assert!((sampled_only[1][0] - 0.9).abs() <= 1.0e-6);
    assert!((sampled_only[1][1] + 0.55).abs() <= 1.0e-6);
    assert!((sampled_only[1][2] - 0.4).abs() <= 1.0e-6);
    assert_eq!(sampled_only[2], [0.0; 4]);
}

#[test]
fn output_gradient_channel_rms_cap_balances_dominant_channels() {
    let mut gradients = vec![
        10.0_f32, 0.1, -4.0, -10.0, 0.1, 0.0, 10.0, 0.1, 4.0, -10.0, 0.1, 0.0,
    ];

    let capped = cap_output_gradient_channel_rms(&mut gradients, 3, 2.0);

    assert_eq!(capped, 2);
    for output in 0..3 {
        let rms = ((0..4)
            .map(|row| gradients[row * 3 + output].powi(2))
            .sum::<f32>()
            / 4.0)
            .sqrt();
        assert!(rms <= 2.0 + 1.0e-6, "output={output} rms={rms}");
    }
    for row in 0..4 {
        assert_eq!(
            gradients[row * 3 + 1],
            0.1,
            "low-RMS channel should not be rescaled"
        );
    }
    assert_eq!(gradients[0].abs(), 2.0);
    assert_eq!(gradients[3].abs(), 2.0);

    let before = gradients.clone();
    assert_eq!(cap_output_gradient_channel_rms(&mut gradients, 3, 0.0), 0);
    assert_eq!(gradients, before);
}

#[test]
fn sparse_output_gradient_rms_boosts_nonzero_geometry_channels() {
    let mut gradients = vec![0.0_f32; 4 * 3];
    gradients[0] = -0.001;
    gradients[6] = -0.003;
    gradients[4] = 1.0;

    let boosted = boost_sparse_output_channel_rms(&mut gradients, 3, 0..2, 0.01, 16.0);

    assert_eq!(boosted, 1);
    let x_rms = ((gradients[0].powi(2) + gradients[6].powi(2)) / 2.0).sqrt();
    assert!((x_rms - 0.01).abs() <= 1.0e-6, "x_rms={x_rms}");
    assert_eq!(
        gradients[4], 1.0,
        "already-strong nonzero channels should not be scaled up"
    );
    assert_eq!(
        gradients[2], 0.0,
        "zero-only channels outside the requested range should stay untouched"
    );
}

#[test]
fn output_gradient_liveness_cap_preserves_sparse_temporal_signal() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let mut gradients = vec![0.0_f32; 4 * output_dims];
    for row in 0..4 {
        gradients[row * output_dims] = 10.0;
        gradients[row * output_dims + liveness_output] = 8.0;
    }

    let capped = cap_output_gradient_channel_rms_with_liveness_cap(
        &config,
        &mut gradients,
        output_dims,
        2.0,
        6.0,
    );

    assert_eq!(capped, 2);
    let motion_rms = ((0..4)
        .map(|row| gradients[row * output_dims].powi(2))
        .sum::<f32>()
        / 4.0)
        .sqrt();
    let liveness_rms = ((0..4)
        .map(|row| gradients[row * output_dims + liveness_output].powi(2))
        .sum::<f32>()
        / 4.0)
        .sqrt();

    assert!(motion_rms <= 2.0 + 1.0e-6);
    assert!(
        (liveness_rms - 6.0).abs() <= 1.0e-6,
        "liveness output should use the larger temporal-growth cap instead of the default render cap"
    );
}

#[test]
fn direct_rollout_gradient_normalization_averages_by_rows() {
    let mut gradients = SupervisedGradients {
        w1: vec![4.0, -8.0],
        b1: vec![2.0],
        w2: vec![12.0, -16.0],
        b2: vec![20.0],
        features: vec![0.0; 4 * 3],
    };

    normalize_supervised_gradients_by_rows(&mut gradients, 3);

    assert_eq!(gradients.w1, vec![1.0, -2.0]);
    assert_eq!(gradients.b1, vec![0.5]);
    assert_eq!(gradients.w2, vec![3.0, -4.0]);
    assert_eq!(gradients.b2, vec![5.0]);
    assert_eq!(gradients.features.len(), 12);
}

#[test]
fn direct_rollout_gradient_normalization_keeps_sparse_rollout_signal_sublinear() {
    let mut gradients = SupervisedGradients {
        w1: vec![4.0, -8.0],
        b1: vec![2.0],
        w2: vec![12.0, -16.0],
        b2: vec![20.0],
        features: vec![0.0; 4 * 3],
    };

    normalize_direct_rollout_gradients(&mut gradients, 3);

    let expected_scale = 1.0_f32 / 4.0_f32.powf(DIRECT_ROLLOUT_GRADIENT_ROW_NORMALIZATION_EXPONENT);
    assert!((gradients.w1[0] - 4.0 * expected_scale).abs() <= 1.0e-6);
    assert!((gradients.w1[1] + 8.0 * expected_scale).abs() <= 1.0e-6);
    assert!((gradients.b1[0] - 2.0 * expected_scale).abs() <= 1.0e-6);
    assert!((gradients.w2[0] - 12.0 * expected_scale).abs() <= 1.0e-6);
    assert!((gradients.w2[1] + 16.0 * expected_scale).abs() <= 1.0e-6);
    assert!((gradients.b2[0] - 20.0 * expected_scale).abs() <= 1.0e-6);
    assert!(
        expected_scale > 0.25 && expected_scale < 1.0,
        "direct rollout gradients should be stronger than full row averaging but still sublinear"
    );
    assert_eq!(gradients.features.len(), 12);
}

#[test]
fn terminal_full_coverage_adjoint_carries_normal_deficit_to_non_gradient_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.04],
            [0.08, 0.0, 0.04],
            [0.0, 0.08, 0.04],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let positions = vec![
        [0.010, 0.010, 0.0, 1.0],
        [0.020, 0.010, 0.0, 1.0],
        [0.010, 0.020, 0.0, 1.0],
        [0.030, 0.010, 0.0, 1.0],
        [0.010, 0.030, 0.0, 1.0],
        [0.020, 0.020, 0.0, 1.0],
        [0.035, 0.015, 0.0, 1.0],
        [0.015, 0.035, 0.0, 1.0],
    ];
    let states = vec![0.0; positions.len() * config.state_dims];
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states: states.clone(),
        batch_size: 1,
        particle_count: positions.len(),
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0],
        gradients: vec![[0.0; 3]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![0.0],
        color_gradients: vec![[0.0; 3]],
    };
    let coverage = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        1.0,
        CoverageUpdateModeArg::SlicedOt,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    );

    let adjoint =
        terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 0.0, true, 1);
    let sampled_only =
        terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 0.0, false, 1);
    let non_gradient_rows_with_normal = adjoint
        .iter()
        .enumerate()
        .filter(|(row, update)| *row != 0 && update[2] < -1.0e-3)
        .count();

    assert!(
        non_gradient_rows_with_normal >= 3,
        "normal-deficit coverage should reach non-gradient rows through full-cloud adjoints: coverage={coverage:?} adjoint={adjoint:?}"
    );
    assert!(
        sampled_only
            .iter()
            .enumerate()
            .filter(|(row, _)| *row != 0)
            .all(|(_, update)| update == &[0.0; 4]),
        "sparse-row adjoint mode should not update unsampled rows: {sampled_only:?}"
    );
}

#[test]
fn surface_position_adjoint_moves_only_active_particles_toward_mesh() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let mut states = vec![0.0; 2 * config.state_dims];
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![[0.0; 4]; 2];

    add_surface_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        0.0,
        &mut adjoint,
    );

    assert!(adjoint[0][0].abs() <= 1.0e-6);
    assert!(adjoint[0][1].abs() <= 1.0e-6);
    assert!(adjoint[0][2] > 0.49 && adjoint[0][2] < 0.51);
    assert_eq!(adjoint[1], [0.0; 4]);
}

#[test]
fn surface_projection_updates_boost_escaped_active_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 0.5, 0.0],
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 2.0, 0.0],
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 2.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let base = render_proxy_surface_projection_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        0.0,
        f32::INFINITY,
    );
    let boosted = render_proxy_surface_projection_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1.0,
        f32::INFINITY,
    );

    assert!((boosted[0][2] - base[0][2]).abs() <= 1.0e-6);
    assert!(
        boosted[1][2] < base[1][2],
        "escaped active particle should receive stronger pull toward the surface"
    );
    assert_eq!(boosted[2], [0.0; 3]);
}

#[test]
fn material_visible_surface_approach_updates_pull_visible_active_particles_toward_mesh() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 0.5, 0.0, 0.25, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "render-visible active material should receive generic projection motion toward the mesh"
    );
    assert!(
        updates[0][0] <= 0.25 + 1.0e-6,
        "material-visible projection update should respect max_update_norm"
    );
}

#[test]
fn material_visible_surface_approach_updates_do_not_move_far_dormant_material() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 0.5, 0.0, 0.25, 1.0, 0.20, None,
    );

    assert!(
        updates[1][0] > 1.0e-4,
        "visible material in the local front should receive bounded projection motion"
    );
    assert_eq!(
        updates[2], [0.0; 3],
        "far dormant visible material should not receive global target assignment motion"
    );
}

#[test]
fn material_surface_candidate_approach_moves_active_rows_before_visibility() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.18, -0.1, 0.0], [0.18, 0.1, 0.0], [0.18, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [1.6_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 1.0, 0.0, 1.0, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "active near-surface material candidate should get projection motion before it is visible"
    );
    assert_eq!(
        updates[1], [0.0; 3],
        "dormant material candidate outside the bounded frontier should not get global projection motion"
    );
}

#[test]
fn material_surface_candidate_approach_moves_frontier_rows_before_visibility() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.80, -0.1, 0.0], [0.80, 0.1, 0.0], [0.80, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [-0.80_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 1.0, 0.0, 1.0, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "active material candidate inside the bounded frontier should get projection motion before strict material coverage"
    );
    assert_eq!(
        updates[1], [0.0; 3],
        "active rows outside the bounded frontier should not get global material projection motion"
    );
}

#[test]
fn material_surface_candidate_coverage_uses_predicted_active_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-0.18, -0.1, 0.0],
            [-0.18, 0.1, 0.0],
            [-0.18, 0.0, 0.2],
            [0.18, -0.1, 0.0],
            [0.18, 0.1, 0.0],
            [0.18, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![[-0.18_f32, 0.0, 0.05, 0.0], [-0.12_f32, 0.0, 0.05, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[liveness_output] = 8.0;
    let weights = material_surface_candidate_row_weights(
        &config,
        &target,
        &positions,
        &states,
        Some(&raw_updates),
        1.0,
        0.0,
        None,
    );

    assert!(
        weights[0] > 0.0,
        "predicted-active near-surface row should become eligible for material surface coverage"
    );
    assert_eq!(
        weights[1], 0.0,
        "dormant non-predicted row should remain ineligible without local-front pressure"
    );
}

#[test]
fn material_visible_surface_approach_output_objective_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; output_dims];

    add_material_visible_surface_approach_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.5,
        0.0,
        0.25,
        1.0,
        0.20,
        None,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[0] < -1.0e-4,
        "positive target x motion should train the x output upward under SGD"
    );
    assert_eq!(
        output_gradients[config.spatial_dims + material_channel],
        0.0,
        "surface approach objective should not directly train material opacity"
    );
}

#[test]
fn material_visible_surface_position_adjoint_tracks_visible_local_front_only() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut adjoint = vec![[0.0_f32; 4]; positions.len()];

    add_material_visible_surface_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        0.0,
        1.0,
        0.20,
        &mut adjoint,
    );

    assert!(
        adjoint[1][0] < -1.0e-4,
        "local-front visible material should receive position adjoint opposite the target motion"
    );
    assert_eq!(
        adjoint[2], [0.0; 4],
        "far dormant visible material should not receive nonlocal surface adjoint"
    );
}

#[test]
fn material_visible_surface_row_weights_include_local_front_but_not_far_dormant() {
    let config = NpaConfig::growing_3dgs();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for row in 0..positions.len() {
        states[row * config.state_dims + material_channel] =
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    }

    let weights =
        material_visible_surface_row_weights(&config, &positions, &states, None, 0.20, None);

    assert_eq!(weights[0], 1.0);
    assert!(
        weights[1] > 0.0,
        "visible material inside the local front should be eligible"
    );
    assert_eq!(
        weights[2], 0.0,
        "far dormant visible material should not become globally eligible"
    );
}

#[test]
fn material_visible_surface_coverage_updates_move_visible_rows_to_uncovered_bins() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.02_f32, -0.04, 0.05, 0.0],
        [-1.00_f32, 0.04, 0.05, 0.0],
        [-0.98_f32, 0.0, 0.08, 0.0],
        [-1.04_f32, 0.0, 0.02, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for row in 0..positions.len() {
        states[row * config.state_dims + material_channel] =
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    }

    let updates = material_visible_surface_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        None,
        1.0,
        512,
        10.0,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        0.20,
        None,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.15),
        "material-visible coverage should relocate redundant visible rows toward uncovered support: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn material_visible_surface_coverage_output_objective_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; output_dims];

    add_material_visible_surface_coverage_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        512,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.20,
        None,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[0] < -1.0e-4,
        "positive material-visible coverage target motion should train x output upward"
    );
}

#[test]
fn material_visible_surface_coverage_position_adjoint_tracks_visible_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut adjoint = vec![[0.0_f32; 4]; positions.len()];

    add_material_visible_surface_coverage_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.20,
        &mut adjoint,
    );

    assert!(
        adjoint[0][0] < -1.0e-4 || adjoint[1][0] < -1.0e-4,
        "visible active/local-front support should receive adjoint opposite material-visible coverage target motion: {adjoint:?}"
    );
    assert_eq!(
        adjoint[2], [0.0; 4],
        "far dormant visible row outside the nearest local shell should not receive nonlocal material-visible coverage adjoint"
    );
}

#[test]
fn growth_3d_validation_rejects_shortcut_lineage() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let path = bin_temp_path("shortcut_growth3d.bpk");
    let manifest = BpkModelManifest::from_model(
        &model,
        grid,
        Some("render-proxy-rust:Torus:field-baseline".to_string()),
    );
    crate::import::save_manifest(&path, &manifest).unwrap();

    let report = growth_3d_validation_report(
        &path,
        MeshTargetArg::Torus,
        growth_validation_test_config(ParticleSeed::TorusGrowth3d),
    )
    .unwrap();
    std::fs::remove_file(&path).ok();

    assert!(!report.local_conditionless_lineage);
    assert!(!report.gate_passed);
    assert!(!report.strict_passed);
}
