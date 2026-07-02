use super::*;

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
