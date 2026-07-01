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
fn local_growth_student_model_wires_phase_controller() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 19, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let phase_output = config.spatial_dims + phase_channel;

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let far_base = input_dims;
    features[far_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let update = model.forward_update_from_features(&features).unwrap();

    assert!(
        update[phase_output] > 0.0,
        "near-front local liveness contrast should advance the phase state"
    );
    assert!(
        update[phase_output] > update[output_dims + phase_output].abs() + 1.0,
        "phase controller should dominate the random initialized far-row phase baseline"
    );
}

#[test]
fn local_growth_student_model_uses_phase_for_material_maturation() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 29, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[phase_channel] = 0.0;
    let mature_base = input_dims;
    features[mature_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[mature_base + phase_channel] = 0.75;

    let update = model.forward_update_from_features(&features).unwrap();
    let immature_material = update[material_output];
    let mature_material = update[output_dims + material_output];

    assert!(
        mature_material > immature_material + 0.15,
        "mature local phase should produce stronger material opacity growth"
    );
}

#[test]
fn local_growth_student_model_uses_phase_to_boost_local_front_liveness() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let phase_liveness_hidden = 25;

    assert_eq!(
        model.weights.b1[phase_liveness_hidden], -4.0,
        "phase liveness bridge should require local-front contrast before it activates"
    );
    assert_eq!(
        model.weights.w1
            [phase_liveness_hidden * input_dims + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[phase_liveness_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        -1.0
    );
    assert_eq!(
        model.weights.w1[phase_liveness_hidden * input_dims + phase_channel],
        1.0
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + phase_liveness_hidden],
        LOCAL_GROWTH_PHASE_LIVENESS_GAIN
    );

    let mut features = vec![0.0_f32; 3 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[phase_channel] = 0.0;
    let phased_front_base = input_dims;
    features[phased_front_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[phased_front_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[phased_front_base + phase_channel] = 0.75;
    let far_base = 2 * input_dims;
    features[far_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + phase_channel] = 0.75;

    let update = model.forward_update_from_features(&features).unwrap();
    let unphased_front_liveness = update[liveness_output];
    let phased_front_liveness = update[output_dims + liveness_output];
    let far_liveness = update[2 * output_dims + liveness_output];

    assert!(
        phased_front_liveness > unphased_front_liveness + 0.02,
        "phase memory should make a local-front dormant row easier to activate"
    );
    assert!(
        far_liveness < unphased_front_liveness * 0.25,
        "phase without local-front liveness contrast must not globally activate dormant rows"
    );
}

#[test]
fn local_growth_student_model_materializes_active_rows_without_waking_dormant_material() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 33, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let active_material_hidden = 26;

    assert_eq!(model.weights.b1[active_material_hidden], 1.0);
    assert_eq!(
        model.weights.w1[active_material_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[active_material_hidden * input_dims + material_channel],
        -0.25
    );
    assert_eq!(
        model.weights.w2[material_output * config.hidden_dims + active_material_hidden],
        LOCAL_GROWTH_ACTIVE_MATERIAL_GAIN
    );

    let mut features = vec![0.0_f32; 4 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let active_low_base = input_dims;
    features[active_low_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_low_base + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let active_mid_base = 2 * input_dims;
    features[active_mid_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_mid_base + material_channel] = 0.0;
    let active_high_base = 3 * input_dims;
    features[active_high_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_high_base + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let update = model.forward_update_from_features(&features).unwrap();
    let dormant_material = update[material_output];
    let active_low_material = update[output_dims + material_output];
    let active_mid_material = update[2 * output_dims + material_output];
    let active_high_material = update[3 * output_dims + material_output];

    assert!(
        active_low_material > dormant_material + 0.5,
        "newly active low-material rows should receive a strong materialization update"
    );
    assert!(
        active_mid_material > dormant_material + 0.2,
        "already live material rows should keep materializing until visible"
    );
    assert!(
        active_high_material < active_mid_material,
        "materialization bridge should damp itself once material opacity is high"
    );
    assert!(
        dormant_material < active_low_material * 0.25,
        "dormant rows must not become material-visible without liveness"
    );
}

#[test]
fn local_growth_student_model_sustains_active_liveness_without_global_activation() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 37, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let active_liveness_low_hidden = 17;
    let active_liveness_high_hidden = 18;

    assert_eq!(
        model.weights.b1[active_liveness_low_hidden], 1.0,
        "active liveness low hidden should gate on liveness + 1"
    );
    assert_eq!(
        model.weights.w1[active_liveness_low_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[active_liveness_high_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + active_liveness_low_hidden],
        LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + active_liveness_high_hidden],
        -2.0 * LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN
    );

    let mut features = vec![0.0_f32; 3 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let active_base = input_dims;
    features[active_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let saturated_base = 2 * input_dims;
    features[saturated_base + GROWTH_3D_LIVENESS_CHANNEL] = 2.0;
    features[saturated_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 2.0;

    let update = model.forward_update_from_features(&features).unwrap();
    let dormant_update = update[liveness_output];
    let active_update = update[output_dims + liveness_output];
    let saturated_update = update[2 * output_dims + liveness_output];

    assert!(
        active_update > dormant_update + LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN * 0.5,
        "active liveness should be sustained without globally activating dormant substrate rows"
    );
    assert!(
        saturated_update < active_update,
        "bounded active liveness controller should push back once liveness is already high"
    );
}

#[test]
fn local_growth_student_model_wires_velocity_memory_to_motion_and_damping() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 41, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let velocity_channels = growth_3d_velocity_channels(config.state_dims).unwrap();
    let velocity_channels = velocity_channels.collect::<Vec<_>>();

    for (axis, &velocity_channel) in velocity_channels.iter().enumerate() {
        let pos_hidden = 19 + axis * 2;
        let neg_hidden = pos_hidden + 1;
        assert_eq!(
            model.weights.w1[pos_hidden * input_dims + velocity_channel],
            1.0
        );
        assert_eq!(
            model.weights.w1[neg_hidden * input_dims + velocity_channel],
            -1.0
        );
        assert_eq!(
            model.weights.w2[axis * config.hidden_dims + pos_hidden],
            LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN
        );
        assert_eq!(
            model.weights.w2[axis * config.hidden_dims + neg_hidden],
            -LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN
        );
    }

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[velocity_channels[0]] = 0.25;
    let negative_base = input_dims;
    features[negative_base + velocity_channels[1]] = -0.5;
    let update = model.forward_update_from_features(&features).unwrap();

    assert!(
        update[0] > 0.20,
        "positive velocity memory should drive same-axis motion"
    );
    assert!(
        update[config.spatial_dims + velocity_channels[0]]
            < -LOCAL_GROWTH_VELOCITY_DAMPING_GAIN * 0.20,
        "positive velocity memory should decay through its state update"
    );
    assert!(
        update[output_dims + 1] < -0.40,
        "negative velocity memory should drive opposite-axis motion"
    );
    assert!(
        update[output_dims + config.spatial_dims + velocity_channels[1]]
            > LOCAL_GROWTH_VELOCITY_DAMPING_GAIN * 0.40,
        "negative velocity memory should damp back toward zero"
    );
}

#[test]
fn material_liveness_adjoint_suppresses_dormant_visible_material() {
    let config = NpaConfig::growing_3dgs();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut states = vec![0.0; 3 * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_material_liveness_state_adjoint(&config, &states, 0.5, 0.1, &mut adjoint);

    assert!(
        adjoint[material_channel] > 0.0,
        "dormant render-visible material should be suppressed"
    );
    assert_eq!(
        adjoint[config.state_dims + material_channel],
        0.0,
        "live render-visible material should not be suppressed by this consistency term"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + material_channel],
        0.0,
        "already-inactive dormant material should not be pushed further"
    );
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.1 + 1.0e-6),
        "material-liveness adjoints should respect max_adjoint"
    );
}

#[test]
fn material_visible_liveness_adjoint_activates_near_surface_material_only() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Teapot, 0.72);
    let sample = target.surface_sample(0);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [10.0_f32, 10.0, 10.0, 0.0],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut adjoint = vec![0.0; states.len()];

    add_material_visible_liveness_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        target_coverage_threshold(0.72),
        0.1,
        &mut adjoint,
    );

    assert!(
        adjoint[GROWTH_3D_LIVENESS_CHANNEL] < 0.0,
        "near-surface dormant material-visible particles should be trained live"
    );
    assert_eq!(
        adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "off-surface dormant material-visible particles should be left to material-tail suppression"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "already-live material-visible particles should not receive this liveness correction"
    );
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.1 + 1.0e-6),
        "material-visible liveness adjoints should respect max_adjoint"
    );
}

#[test]
fn direct_rollout_training_activates_near_surface_material_visible_particles() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Teapot, 0.72);
    let sample = target.surface_sample(0);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let before_liveness_bias = model.weights.b2[config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL];
    let before_material_bias = model.weights.b2[config.spatial_dims + material_channel];
    let positions = vec![[
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ]];
    let mut states = vec![0.0; config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let features = vec![0.0; config.perception_dims()];
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states: states.clone(),
        batch_size: 1,
        particle_count: 1,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let trajectory = vec![RenderTrajectorySnapshot {
        positions,
        states,
        features,
        step_fraction: 1.0,
    }];
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
        max_update_norm: 0.0,
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
        max_opacity_update: 0.1,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: false,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 17,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 8,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0,
            grad_clip_norm: 100.0,
            weight_decay: 0.0,
        },
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

    assert_eq!(report.steps, 1);
    assert_eq!(report.rows, 1);
    assert!(
        model.weights.b2[config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] > before_liveness_bias,
        "direct rollout training should increase liveness output for near-surface material-visible dormant particles"
    );
    assert!(
        model.weights.b2[config.spatial_dims + material_channel] < before_material_bias,
        "direct rollout training should also suppress material opacity while the row is still dormant"
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

#[test]
fn material_visible_surface_tail_adjoint_suppresses_off_surface_material() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let sample = target.surface_sample(0);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [2.0_f32, 0.0, 0.0, 0.0],
        [2.2_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_material_visible_surface_tail_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        10.0,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
        0.07,
        &mut adjoint,
    );

    assert_eq!(
        adjoint[material_channel], 0.0,
        "on-surface visible material should not be suppressed by the tail term"
    );
    assert!(
        (adjoint[config.state_dims + material_channel] - 0.07).abs() <= 1.0e-6,
        "off-surface visible material should receive clamped suppression"
    );
    assert_eq!(
        adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "material tail pressure must not alter liveness directly"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + material_channel],
        0.0,
        "already-inactive off-surface material should not be pushed further"
    );
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.07 + 1.0e-6),
        "material-tail adjoints should respect max_adjoint"
    );
}

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

#[test]
fn soft_chamfer_coverage_distributes_symmetric_target_pressure() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[-0.02, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let hard = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::HardNearest,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let soft = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let soft_nonzero = soft
        .iter()
        .filter(|update| update.iter().any(|value| value.abs() > 1.0e-6))
        .count();

    assert!(hard.iter().flatten().all(|value| value.is_finite()));
    assert_eq!(soft_nonzero, 2);
    assert!(soft[0][0] > 0.0);
    assert!(soft[1][0] < 0.0);
}

#[test]
fn weighted_target_coverage_updates_include_local_front_rows() {
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
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [0.72_f32, 0.0, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0, 0.0],
    ];
    let max_update_norm = 0.25;
    let updates = render_proxy_weighted_target_coverage_updates(
        &target,
        &positions,
        &[1.0, 0.5, 0.0],
        1.0,
        256,
        max_update_norm,
        CoverageUpdateModeArg::HardNearest,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let front_norm = (updates[1][0].powi(2) + updates[1][1].powi(2) + updates[1][2].powi(2)).sqrt();

    assert!(
        updates[1][0] > 0.0,
        "weighted local-front row should receive pressure toward the uncovered target lobe"
    );
    assert!(front_norm <= max_update_norm + 1.0e-6);
    assert_eq!(
        updates[2],
        [0.0, 0.0, 0.0],
        "zero-weight row should remain untouched"
    );
}

#[test]
fn soft_chamfer_coverage_respects_update_clamp() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[3.0, 4.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let max_update_norm = 0.05;

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        max_update_norm,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let norm = (updates[0][0].powi(2) + updates[0][1].powi(2) + updates[0][2].powi(2)).sqrt();

    assert!(norm <= max_update_norm + 1.0e-6);
    assert!(norm > 0.0);
}

#[test]
fn soft_chamfer_repulsion_adds_tangent_spread_pressure() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[-0.005, 0.0, 0.0, 0.0], [0.005, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let no_repulsion = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let with_repulsion = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        1.0,
        0.0,
        0.1,
        0.0,
        1.0,
    );

    assert!(no_repulsion[0][0] > 0.0);
    assert!(no_repulsion[1][0] < 0.0);
    assert!(with_repulsion[0][0] < no_repulsion[0][0]);
    assert!(with_repulsion[1][0] > no_repulsion[1][0]);
    assert!(with_repulsion[0][2].abs() <= 1.0e-6);
    assert!(with_repulsion[1][2].abs() <= 1.0e-6);
}

#[test]
fn gap_farthest_coverage_avoids_symmetric_residual_cancellation() {
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
    let positions = vec![[0.0, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let hard = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let gap = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::GapFarthest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let hard_norm = (hard[0][0].powi(2) + hard[0][1].powi(2) + hard[0][2].powi(2)).sqrt();
    let gap_norm = (gap[0][0].powi(2) + gap[0][1].powi(2) + gap[0][2].powi(2)).sqrt();

    assert!(hard.iter().flatten().all(|value| value.is_finite()));
    assert!(gap.iter().flatten().all(|value| value.is_finite()));
    assert!(
        gap_norm > hard_norm + 0.1,
        "gap mode should keep a directional worst-gap signal instead of averaging it away: hard={hard:?} gap={gap:?}"
    );
}

#[test]
fn gap_farthest_coverage_balances_uncovered_bins_across_donors() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )
    .unwrap();
    let positions = vec![[-1.0, -0.04, 0.05, 0.0], [-0.95, 0.04, 0.05, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let gap = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        10.0,
        CoverageUpdateModeArg::GapFarthest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert!(
        gap.iter().all(|update| update[0] > 0.1),
        "balanced gap mode should spread uncovered right-side bins across available donors: {gap:?}"
    );
    assert!(gap.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_strata_coverage_moves_redundant_rows_to_empty_surface_bins() {
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
    let positions = vec![
        [-1.02_f32, -0.04, 0.05, 0.0],
        [-1.00_f32, 0.04, 0.05, 0.0],
        [-0.98_f32, 0.0, 0.08, 0.0],
        [-1.04_f32, 0.0, 0.02, 0.0],
    ];
    let active_rows = vec![0, 1, 2, 3];
    let mut updates = vec![[0.0_f32; 3]; positions.len()];

    add_surface_strata_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.5,
        10.0,
        &mut updates,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.15),
        "strata coverage should move at least one redundant left-patch row toward the uncovered right patch: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_gap_relocation_can_use_low_assignment_donors() {
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )
    .unwrap();
    let positions = vec![[-1.0, 0.0, 0.05, 0.0], [0.0, 0.0, 0.05, 0.0]];
    let active_rows = vec![0, 1];
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.0,
        1.0,
        10.0,
        &mut updates,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.1),
        "a nonzero-assigned donor should be allowed to move toward the uncovered right mode: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn sliced_ot_coverage_balances_separated_surface_modes() {
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
    let positions = vec![[-0.05, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::SlicedOt,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert!(
        updates[0][0] < 0.0,
        "left-ranked particle should be pulled toward the left target mode: {updates:?}"
    );
    assert!(
        updates[1][0] > 0.0,
        "right-ranked particle should be pulled toward the right target mode: {updates:?}"
    );
}

#[test]
fn sliced_ot_coverage_pushes_collapsed_torus_centerline_into_tube() {
    let config = NpaConfig::growing_3dgs();
    let scale = 0.54_f32;
    let target = mesh_target_for_arg(MeshTargetArg::Torus, scale);
    let ring_count = 16usize;
    let positions = (0..ring_count)
        .map(|idx| {
            let theta = std::f32::consts::TAU * idx as f32 / ring_count as f32;
            [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    let states = vec![0.0; positions.len() * config.state_dims];

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        2048,
        f32::INFINITY,
        CoverageUpdateModeArg::SlicedOt,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
    );

    assert_eq!(
        sliced_ot_directions().len(),
        normal_coverage_directions().len()
    );
    let tube_pressure = updates
        .iter()
        .enumerate()
        .map(|(idx, update)| {
            let theta = std::f32::consts::TAU * idx as f32 / ring_count as f32;
            let radial = update[0] * theta.cos() + update[1] * theta.sin();
            radial.abs() + update[2].abs()
        })
        .sum::<f32>();

    assert!(
        tube_pressure > 1.0e-3,
        "collapsed centerline should receive tube-plane pressure toward the full surface support: {updates:?}"
    );
}
