use super::*;

#[test]
fn render_proxy_batch_applies_material_target_coverage_updates() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 13);
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 1,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0],
        gradients: vec![[0.0; 3]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![0.0],
        color_gradients: vec![[0.0; 3]],
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
        trajectory_supervision: false,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 0.0,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: 0.0,
        coverage_samples: 64,
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
        opacity_gain: 1.0,
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
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 1.0,
        seed_mode: ParticleSeed::UniformCircle,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let batch = render_proxy_supervised_batch(&model, &grid, &target, &trace, &[], &gradient, &cfg)
        .unwrap();
    let baseline = model.forward_update_from_features(&batch.features).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let delta = batch.target_update[material_output] - baseline[material_output];

    assert!(
        delta > 0.0,
        "target-assigned surface particle should receive a positive material output target"
    );
    assert!(
        delta <= cfg.material_max_opacity_update + 1.0e-6,
        "material coverage target should respect material_max_opacity_update"
    );
}
#[test]
fn render_proxy_batch_applies_local_liveness_front_updates() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 17);
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [0.08_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let step = model
        .step_cpu(&positions, &states, 1, positions.len(), &grid, 1.0, None)
        .unwrap();
    let trajectory = vec![RenderTrajectorySnapshot {
        positions: positions.clone(),
        states: states.clone(),
        features: step.perception.features,
        step_fraction: 0.5,
    }];
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 2,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0, 1],
        gradients: vec![[0.0; 3]; 2],
        opacity_gradients: vec![0.0; 2],
        scale_gradients: vec![0.0; 2],
        color_gradients: vec![[0.0; 3]; 2],
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 2,
        rollout_steps: 1,
        gradient_particles: 2,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 1.0,
        trajectory_supervision: true,
        trajectory_render_gain: 0.0,
        trajectory_mesh_gain: 0.0,
        trajectory_render_samples: 0,
        liveness_gain: 1.0,
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
        training_backend: RenderTrainingBackendArg::Proxy,
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

    let batch =
        render_proxy_supervised_batch(&model, &grid, &target, &trace, &trajectory, &gradient, &cfg)
            .unwrap();
    let baseline = model.forward_update_from_features(&batch.features).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let dormant_row_base = config.update_dims();
    let delta = batch.target_update[dormant_row_base + liveness_output]
        - baseline[dormant_row_base + liveness_output];

    assert!(
        delta > 0.0,
        "dormant local-front particle should receive positive liveness growth target"
    );
    assert!(
        delta > cfg.max_opacity_update,
        "liveness growth should be able to move faster than material opacity"
    );
    assert!(
        delta
            <= liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier) + 1.0e-6,
        "liveness front target should respect the liveness update cap"
    );
}
#[test]
fn render_proxy_batch_uses_stronger_cap_for_inactive_material_suppression() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let target = mesh_target_for_arg(MeshTargetArg::Teapot, 0.72);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states: states.clone(),
        batch_size: 1,
        particle_count: 1,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0],
        gradients: vec![[0.0; 3]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![0.0],
        color_gradients: vec![[0.0; 3]],
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Teapot,
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
        trajectory_supervision: false,
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
        material_liveness_gain: 1.0,
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
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TeapotGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let batch = render_proxy_supervised_batch(&model, &grid, &target, &trace, &[], &gradient, &cfg)
        .unwrap();
    let baseline = model.forward_update_from_features(&batch.features).unwrap();
    let output = config.spatial_dims + material_channel;
    let delta = batch.target_update[output] - baseline[output];

    assert!(
        delta < -cfg.max_opacity_update,
        "inactive material suppression should not be limited by the positive opacity-growth cap"
    );
    assert!(
        delta
            >= -material_suppression_max_update(
                cfg.material_max_opacity_update,
                cfg.material_suppression_update_multiplier,
            ) - 1.0e-6,
        "inactive material suppression should still be bounded by its own cap"
    );
}
#[test]
fn render_proxy_batch_activates_near_surface_material_visible_rows() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let target = mesh_target_for_arg(MeshTargetArg::Teapot, 0.72);
    let sample = target.surface_sample(0);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ]];
    let mut states = vec![0.0; config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 1,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0],
        gradients: vec![[0.0; 3]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![0.0],
        color_gradients: vec![[0.0; 3]],
    };
    let cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Teapot,
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
        trajectory_supervision: false,
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
        material_liveness_gain: 1.0,
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
        seed: 1,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TeapotGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig::default(),
    };

    let batch = render_proxy_supervised_batch(&model, &grid, &target, &trace, &[], &gradient, &cfg)
        .unwrap();
    let baseline = model.forward_update_from_features(&batch.features).unwrap();
    let output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let delta = batch.target_update[output] - baseline[output];

    assert!(
        delta > 0.0,
        "near-surface dormant material-visible row should receive positive liveness target"
    );
    assert!(
        delta > cfg.max_opacity_update,
        "material-visible liveness should be able to move faster than material opacity"
    );
    assert!(
        delta
            <= liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier) + 1.0e-6,
        "material-visible liveness target should respect the liveness update cap"
    );
}
