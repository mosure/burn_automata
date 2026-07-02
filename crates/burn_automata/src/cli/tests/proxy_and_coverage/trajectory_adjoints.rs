use super::*;

#[test]
fn trajectory_liveness_adjoint_does_not_require_trajectory_render_samples() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let trajectory = vec![
        RenderTrajectorySnapshot {
            positions: positions.clone(),
            states: states.clone(),
            features: Vec::new(),
            step_fraction: 0.5,
        },
        RenderTrajectorySnapshot {
            positions: positions.clone(),
            states: states.clone(),
            features: Vec::new(),
            step_fraction: 1.0,
        },
    ];
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states,
        batch_size: 1,
        particle_count: positions.len(),
        state_dims: config.state_dims,
        steps: trajectory.len(),
        mean_dx: Vec::new(),
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: positions.len(),
        rollout_steps: trajectory.len(),
        gradient_particles: 1,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.25,
        liveness_front_radius: 0.20,
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
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: false,
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let adjoints = trajectory_render_adjoints(&config, &target, &trajectory, &trace, &cfg).unwrap();
    let final_adjoint = adjoints[1]
        .as_ref()
        .expect("liveness-only sample should produce an adjoint");
    let near_liveness = final_adjoint.state[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    let far_liveness = final_adjoint.state[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    let max_position_adjoint = final_adjoint
        .position
        .iter()
        .flat_map(|row| row.iter().take(config.spatial_dims))
        .fold(0.0_f32, |max_value, value| max_value.max(value.abs()));

    assert!(
        near_liveness < 0.0,
        "near-front inactive particle should receive liveness pressure even when trajectory render gain and samples are zero"
    );
    assert_eq!(far_liveness, 0.0);
    assert_eq!(max_position_adjoint, 0.0);
    assert_eq!(final_adjoint.weight, 1.0);
}
#[test]
fn trajectory_liveness_adjoint_suppresses_over_fast_activation_without_render_gain() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]; 10];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    for row in 0..9 {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = if row < 2 {
            0.75
        } else {
            -0.20 + row as f32 * 0.01
        };
    }
    let trajectory = vec![RenderTrajectorySnapshot {
        positions: positions.clone(),
        states: states.clone(),
        features: Vec::new(),
        step_fraction: 0.50,
    }];
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 10,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 10,
        rollout_steps: 1,
        gradient_particles: 1,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 1,
        liveness_gain: 0.25,
        liveness_front_radius: 0.20,
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
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: false,
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let adjoints = trajectory_render_adjoints(&config, &target, &trajectory, &trace, &cfg).unwrap();
    let adjoint = adjoints[0]
        .as_ref()
        .expect("liveness schedule should create a snapshot adjoint");
    let liveness_adjoint =
        |row: usize| adjoint.state[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];

    assert!(liveness_adjoint(2) > 0.0);
    assert_eq!(liveness_adjoint(0), 0.0);
}
#[test]
fn trajectory_mesh_adjoint_suppresses_escaped_visible_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let positions = vec![[3.0_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let trajectory = vec![RenderTrajectorySnapshot {
        positions: positions.clone(),
        states: states.clone(),
        features: Vec::new(),
        step_fraction: 1.0,
    }];
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 1,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 1,
        rollout_steps: 1,
        gradient_particles: 1,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 1.0,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.5,
        trajectory_render_samples: 0,
        liveness_gain: 0.25,
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
        surface_gain: 1.0,
        surface_escape_gain: 1.0,
        opacity_gain: 0.5,
        material_liveness_gain: 0.5,
        material_tail_gain: 0.5,
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
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: false,
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let adjoints = trajectory_render_adjoints(&config, &target, &trajectory, &trace, &cfg).unwrap();
    let adjoint = adjoints[0]
        .as_ref()
        .expect("mesh trajectory state escape should create an adjoint");

    assert!(
        adjoint.state[GROWTH_3D_LIVENESS_CHANNEL] > 0.0,
        "escaped active particle should receive trajectory liveness suppression"
    );
    assert!(
        adjoint.state[opacity_channel] > 0.0,
        "escaped active particle should receive trajectory material-opacity suppression"
    );
}
#[test]
fn trajectory_mesh_adjoint_does_not_require_trajectory_render_samples() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let positions = vec![[2.0_f32, 0.0, 0.0, 0.0], [2.1_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for row in 0..positions.len() {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    }
    let trajectory = vec![RenderTrajectorySnapshot {
        positions: positions.clone(),
        states: states.clone(),
        features: Vec::new(),
        step_fraction: 1.0,
    }];
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states,
        batch_size: 1,
        particle_count: positions.len(),
        state_dims: config.state_dims,
        steps: trajectory.len(),
        mean_dx: Vec::new(),
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: positions.len(),
        rollout_steps: trajectory.len(),
        gradient_particles: 1,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 1.0,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.5,
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
        surface_gain: 1.0,
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
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: false,
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let adjoints = trajectory_render_adjoints(&config, &target, &trajectory, &trace, &cfg).unwrap();
    let final_adjoint = adjoints[0]
        .as_ref()
        .expect("mesh-only sample should produce an adjoint");
    let max_position_adjoint = final_adjoint
        .position
        .iter()
        .flat_map(|row| row.iter().take(config.spatial_dims))
        .fold(0.0_f32, |max_value, value| max_value.max(value.abs()));
    let max_state_adjoint = final_adjoint
        .state
        .iter()
        .fold(0.0_f32, |max_value, value| max_value.max(value.abs()));

    assert!(
        max_position_adjoint > 0.0,
        "mesh trajectory gain should emit position adjoints even when trajectory render gain and samples are zero"
    );
    assert_eq!(max_state_adjoint, 0.0);
    assert_eq!(final_adjoint.weight, 1.0);
}
#[test]
fn render_proxy_trajectory_batch_applies_bounded_coverage_updates() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = local_growth_student_model(config, 17, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
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
        motion_gain: 0.0,
        perception_position_gain: 0.05,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.25,
        coverage_samples: 128,
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
        training_backend: RenderTrainingBackendArg::Proxy,
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: false,
        seed: 11,
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
            learning_rate: 1.0e-4,
            grad_clip_norm: 0.1,
            weight_decay: 0.0,
        },
    };
    let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let gradient = RenderProxyGradientRows {
        row_indices: (0..cfg.particles).collect(),
        gradients: vec![[0.0; 3]; cfg.particles],
        opacity_gradients: vec![0.0; cfg.particles],
        scale_gradients: vec![0.0; cfg.particles],
        color_gradients: vec![[0.0; 3]; cfg.particles],
    };
    let batch =
        render_proxy_supervised_batch(&model, &grid, &target, &trace, &trajectory, &gradient, &cfg)
            .unwrap();
    let rows = batch.features.len() / model.config.perception_dims();
    assert_eq!(rows, cfg.particles * cfg.rollout_steps);
    let baseline = model.forward_update_from_features(&batch.features).unwrap();
    let output_dims = model.config.update_dims();
    let mut changed_motion_rows = 0usize;
    for row in 0..rows {
        let base = row * output_dims;
        let delta = ((batch.target_update[base] - baseline[base]).powi(2)
            + (batch.target_update[base + 1] - baseline[base + 1]).powi(2)
            + (batch.target_update[base + 2] - baseline[base + 2]).powi(2))
        .sqrt();
        if delta > 0.0 {
            changed_motion_rows += 1;
            assert!(delta <= cfg.max_update_norm + 1.0e-5);
        }
    }
    assert!(changed_motion_rows > 0);
}
