use super::*;

#[test]
fn material_output_only_gradients_freeze_hidden_and_motion_rows() {
    let config = NpaConfig::growing_3dgs();
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let mut gradients = zero_supervised_gradients(&model);
    gradients.w1.fill(1.0);
    gradients.b1.fill(1.0);
    gradients.w2.fill(1.0);
    gradients.b2.fill(1.0);

    retain_material_output_gradients(&model, &mut gradients).unwrap();

    assert!(gradients.w1.iter().all(|value| *value == 0.0));
    assert!(gradients.b1.iter().all(|value| *value == 0.0));
    let material_channel = growth_3d_material_opacity_channel(model.config.state_dims).unwrap();
    let material_output = model.config.spatial_dims + material_channel;
    for output in 0..model.config.update_dims() {
        let row = &gradients.w2
            [output * model.config.hidden_dims..(output + 1) * model.config.hidden_dims];
        if output == material_output {
            assert!(row.iter().all(|value| *value == 1.0));
            assert_eq!(gradients.b2[output], 1.0);
        } else {
            assert!(row.iter().all(|value| *value == 0.0));
            assert_eq!(gradients.b2[output], 0.0);
        }
    }
}

#[test]
fn learned_scale_render_gradient_writes_scale_state_channel() {
    let config = NpaConfig::growing_3dgs();
    let trace = crate::RolloutTrace {
        positions: vec![[0.0; 4]; 2],
        states: vec![0.0; 2 * config.state_dims],
        batch_size: 1,
        particle_count: 2,
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![1],
        gradients: vec![[0.0; 3]],
        opacity_gradients: vec![0.0],
        scale_gradients: vec![2.0],
        color_gradients: vec![[0.0; 3]],
    };

    let state_adjoint = terminal_render_state_adjoint(
        &config,
        &trace,
        &gradient,
        0.0,
        0.5,
        0.0,
        0.0,
        ROBUST_3D_LIVENESS_FRONT_RADIUS,
        1.0,
        0.05,
        RenderLossConfig {
            gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
            ..RenderLossConfig::default()
        },
        1,
    );

    let scale_channel = config.state_dims - 5;
    assert_eq!(state_adjoint[config.state_dims + scale_channel], 1.0);
    assert_eq!(state_adjoint[scale_channel], 0.0);
}

#[test]
fn finite_difference_render_gradient_trains_color_channels() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let tail = config.state_dims - 3;
    let mut positions = Vec::new();
    let mut states = vec![0.0; 8 * config.state_dims];
    for idx in 0..8 {
        let sample = target.surface_sample(idx);
        positions.push([
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ]);
        let base = idx * config.state_dims;
        states[base + opacity_channel] = 6.0;
        states[base + tail] = -0.4;
        states[base + tail + 1] = 0.1;
        states[base + tail + 2] = 0.2;
    }
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 8,
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let render = RenderLossConfig {
        image_size: 16,
        target_samples: 8,
        world_scale: 1.08,
        density_weight: 0.0,
        color_weight: 1.0,
        depth_weight: 0.0,
        ..RenderLossConfig::default()
    };
    let base_cfg = RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 8,
        rollout_steps: 1,
        gradient_particles: 1,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 1.0,
        perception_position_gain: 0.0,
        max_update_norm: 0.05,
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
        training_backend: RenderTrainingBackendArg::DirectRollout,
        direct_selection_seed_training: false,
        seed: 7,
        selection_seed: None,
        selection_seeds: Vec::new(),
        seed_scale: 0.54,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render,
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    };
    let analytic = render_position_gradient(&trace, &target, render, &base_cfg).unwrap();
    let finite_cfg = RenderProxyTrainingConfig {
        gradient_mode: RenderGradientModeArg::FiniteDiff,
        ..base_cfg
    };
    let finite = render_position_gradient(&trace, &target, render, &finite_cfg).unwrap();

    assert!(finite.color_gradients[0][0].abs() > 1.0e-4);
    for channel in 0..3 {
        let expected = analytic.color_gradients[0][channel];
        let actual = finite.color_gradients[0][channel];
        assert!(
            (actual - expected).abs() <= 2.5e-2 + expected.abs() * 0.20,
            "channel={channel} finite={actual} analytic={expected}"
        );
    }
}

#[test]
fn learned_scale_budget_adjoint_penalizes_oversized_particles() {
    let config = NpaConfig::growing_3dgs();
    let scale_channel = config.state_dims - 5;
    let mut states = vec![0.0; 2 * config.state_dims];
    states[scale_channel] = 1.0;
    states[config.state_dims + scale_channel] = 1.0;
    let trace = crate::RolloutTrace {
        positions: vec![[0.0; 4]; 2],
        states,
        batch_size: 1,
        particle_count: 2,
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: Vec::new(),
        gradients: Vec::new(),
        opacity_gradients: Vec::new(),
        scale_gradients: Vec::new(),
        color_gradients: Vec::new(),
    };

    let state_adjoint = terminal_render_state_adjoint(
        &config,
        &trace,
        &gradient,
        0.0,
        0.0,
        0.5,
        0.0,
        ROBUST_3D_LIVENESS_FRONT_RADIUS,
        1.0,
        0.05,
        RenderLossConfig {
            gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
            sigma: 1.0,
            min_sigma: 0.25,
            max_sigma: 6.0,
            ..RenderLossConfig::default()
        },
        0,
    );

    assert!(state_adjoint[scale_channel] > 0.0);
    assert!(state_adjoint[config.state_dims + scale_channel] > 0.0);
}

#[test]
fn liveness_front_adjoint_pushes_near_front_without_global_activation() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_liveness_front_state_adjoint(
        &config,
        &positions,
        &states,
        0.25,
        0.20,
        1.0,
        0.05,
        &mut adjoint,
    );

    assert!(
        adjoint[GROWTH_3D_LIVENESS_CHANNEL] < 0.0,
        "active seed should receive bounded liveness reinforcement"
    );
    assert!(
        adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] < 0.0,
        "near-front inactive particle should receive negative state adjoint to train positive liveness update"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "far dormant particle should not receive global activation pressure"
    );
}

#[test]
fn local_front_weights_adapt_to_sparse_dormant_shell_without_global_front() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.60_f32, 0.0, 0.0, 0.0],
        [0.95_f32, 0.0, 0.0, 0.0],
        [1.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let weights = local_front_weights(&config, &positions, &states, 0.20);

    assert_eq!(weights[0], 1.0);
    assert!(
        weights[1] > 0.0,
        "sparse clouds should expand the training front to the nearest dormant shell"
    );
    assert_eq!(
        weights[2], 0.0,
        "adaptive sparse-front radius should not make every dormant particle local"
    );
    assert_eq!(weights[3], 0.0);
}

#[test]
fn local_front_candidate_budget_scales_for_larger_clouds_without_global_default() {
    assert_eq!(default_local_front_candidate_count(0), 0);
    assert_eq!(default_local_front_candidate_count(10), 1);
    assert_eq!(default_local_front_candidate_count(64), 4);
    assert_eq!(default_local_front_candidate_count(1024), 64);
    assert_eq!(
        default_local_front_candidate_count(8192),
        DEFAULT_LOCAL_FRONT_MAX_CANDIDATES,
        "larger clouds should train a bounded shell instead of silently staying capped at eight rows"
    );
}

#[test]
fn temporal_local_front_weights_can_expand_to_activation_deficit() {
    let config = NpaConfig::growing_3dgs();
    let positions = (0..10)
        .map(|row| [row as f32 * 0.10, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let narrow = local_front_weights(&config, &positions, &states, 0.05);
    let expanded = local_front_weights_with_min_candidates(&config, &positions, &states, 0.05, 4);

    assert!(
        narrow[1] > 0.0 && narrow[2] == 0.0,
        "default local front should stay narrow for generic mesh/material objectives"
    );
    assert!(
        (1..=4).all(|row| expanded[row] > 0.0),
        "temporal activation should expose the nearest dormant shell needed by the deficit"
    );
    assert!(
        expanded[4] >= 0.25,
        "the outer requested temporal shell should keep enough weight to survive gradient balancing"
    );
    assert_eq!(
        expanded[5], 0.0,
        "temporal shell expansion should still leave farther dormant rows untouched"
    );
}

#[test]
fn temporal_front_candidate_budget_scales_but_stays_bounded() {
    assert_eq!(temporal_front_candidate_count(0, 64), 0);
    assert_eq!(temporal_front_candidate_count(64, 64), 16);
    assert_eq!(
        temporal_front_candidate_count(128, 128),
        64,
        "short 3D rollout probes need enough temporal candidates to grow beyond the initial seed shell"
    );
    assert_eq!(temporal_front_candidate_count(1024, 1024), 512);
    assert_eq!(temporal_front_candidate_count(8192, 8192), 4096);
    assert_eq!(
        temporal_front_candidate_count(8192, 7),
        7,
        "the temporal shell should never request more candidates than the current activation deficit"
    );
}

#[test]
fn terminal_state_adjoint_includes_temporal_activation_schedule() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let trace = crate::RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: 3,
        state_dims: config.state_dims,
        steps: 1,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: Vec::new(),
        gradients: Vec::new(),
        opacity_gradients: Vec::new(),
        scale_gradients: Vec::new(),
        color_gradients: Vec::new(),
    };

    let state_adjoint = terminal_render_state_adjoint(
        &config,
        &trace,
        &gradient,
        0.0,
        0.0,
        0.0,
        0.25,
        0.20,
        1.0,
        0.05,
        RenderLossConfig::default(),
        0,
    );

    let near = state_adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    assert!(
        near <= -0.09,
        "terminal liveness adjoint should include both front reinforcement and temporal activation pressure"
    );
    assert_eq!(
        state_adjoint[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "terminal activation schedule must remain local-front only"
    );
}

#[test]
fn temporal_activation_schedule_suppresses_weak_overactive_rows() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]; 10];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    for row in 0..9 {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = if row < 2 {
            0.75
        } else {
            -0.20 + row as f32 * 0.01
        };
    }
    let mut adjoint = vec![0.0; states.len()];

    add_temporal_activation_schedule_state_adjoint(
        &config,
        &positions,
        &states,
        0.25,
        0.20,
        0.50,
        0.05,
        &mut adjoint,
    );

    let liveness_adjoint =
        |row: usize| adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    assert!(
        liveness_adjoint(2) > 0.0,
        "weakest active row should be trained back below the progressive activation schedule"
    );
    assert_eq!(
        liveness_adjoint(0),
        0.0,
        "strong seed/core rows should be preserved when suppressing over-fast activation"
    );
    assert_eq!(
        liveness_adjoint(9),
        0.0,
        "inactive rows should not receive suppression"
    );
}

#[test]
fn temporal_activation_schedule_boosts_underactive_local_front_only() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let mut adjoint = vec![0.0; states.len()];

    add_temporal_activation_schedule_state_adjoint(
        &config,
        &positions,
        &states,
        0.25,
        0.20,
        0.50,
        0.05,
        &mut adjoint,
    );

    let liveness_adjoint =
        |row: usize| adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    assert!(
        liveness_adjoint(1) < 0.0,
        "under-active snapshots should train nearby dormant front particles toward activation"
    );
    assert_eq!(
        liveness_adjoint(2),
        0.0,
        "far dormant particles should not receive global activation pressure"
    );
    assert_eq!(
        liveness_adjoint(3),
        0.0,
        "only local-front candidates should be used to satisfy the temporal lower bound"
    );
}

#[test]
fn temporal_liveness_output_objective_boosts_underactive_local_front() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.20,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[liveness_output], 0.0);
    assert!(
        output_gradients[output_dims + liveness_output] < -1.0,
        "under-active snapshots should directly train the next liveness update upward for local-front rows"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "far dormant rows should not receive global liveness output pressure"
    );
}

#[test]
fn temporal_liveness_output_objective_can_gate_activation_by_mesh_motion() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.12_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];
    let candidate_weights = vec![0.0, 0.0, 1.0, 0.0];

    add_temporal_liveness_output_objective_with_candidate_weights(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.5,
        1.0,
        0.20,
        Some(&candidate_weights),
        &mut output_gradients,
    );

    assert_eq!(output_gradients[liveness_output], 0.0);
    assert_eq!(
        output_gradients[output_dims + liveness_output],
        0.0,
        "local-front rows without mesh-motion pressure should not be activated by the coupled direct objective"
    );
    assert!(
        output_gradients[2 * output_dims + liveness_output] < 0.0,
        "local-front rows with mesh-motion pressure should receive activation pressure"
    );
    assert_eq!(
        output_gradients[3 * output_dims + liveness_output],
        0.0,
        "far dormant rows should remain unguided"
    );
}

#[test]
fn temporal_liveness_output_objective_prioritizes_stronger_mesh_motion() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.12_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];
    let candidate_weights = vec![0.0, 0.05, 1.0, 0.0];

    add_temporal_liveness_output_objective_with_candidate_weights(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.25,
        1.0,
        0.20,
        Some(&candidate_weights),
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[output_dims + liveness_output],
        0.0,
        "weaker mesh-motion local-front row should not consume the single activation deficit"
    );
    assert!(
        output_gradients[2 * output_dims + liveness_output] < 0.0,
        "stronger mesh-motion local-front row should be activated first"
    );
}

#[test]
fn mesh_motion_liveness_output_objective_activates_moving_dormant_rows() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.16_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let motion_weights = vec![1.0, 0.75, 0.0];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_motion_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        &motion_weights,
        1.0,
        0.25,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[liveness_output], 0.0,
        "already-live rows should not receive mesh-motion activation pressure"
    );
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "dormant rows with mesh-motion pressure should train liveness updates upward"
    );
    assert!(
        output_gradients[output_dims + liveness_output].abs() <= 0.75 * 0.25 + 1.0e-6,
        "mesh-motion liveness targets should respect the liveness update cap and candidate weight"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "dormant rows without mesh-motion pressure should remain untouched"
    );
}

#[test]
fn mesh_motion_liveness_output_objective_skips_already_predicted_live_rows() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![[0.08_f32, 0.0, 0.0, 0.0]];
    let states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; config.state_dims];
    let mut raw_updates = vec![0.0_f32; output_dims];
    raw_updates[liveness_output] = 10.0;
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_motion_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        &[1.0],
        1.0,
        0.25,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[liveness_output], 0.0,
        "rows already predicted to become live should not get extra activation pressure"
    );
}

#[test]
fn extent_front_liveness_candidate_weights_prioritize_active_bounds_expansion() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [-0.08_f32, 0.0, 0.0, 0.0],
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.90_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let weights =
        extent_front_liveness_candidate_weights(&config, &target, &positions, &states, 0.20);

    assert_eq!(weights[0], 0.0, "active seed row should not be reactivated");
    assert!(
        weights[1] > 0.0 && weights[2] > 0.0,
        "local dormant rows that expand active x bounds should receive extent-front priority: {weights:?}"
    );
    assert_eq!(
        weights[3], 0.0,
        "rows inside current active x bounds should not receive extent-front priority"
    );
    assert_eq!(
        weights[4], 0.0,
        "far dormant rows should remain gated by the local front"
    );
}

#[test]
fn extent_front_liveness_candidate_weights_respect_target_bounds() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [1.0_f32, 0.0, 0.0, 0.0],
        [1.08_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let weights =
        extent_front_liveness_candidate_weights(&config, &target, &positions, &states, 0.20);

    assert_eq!(
        weights[1], 0.0,
        "rows beyond a fully covered high target bound should not be prioritized"
    );
    assert!(
        weights[2] > 0.0,
        "rows expanding toward an uncovered low target bound should stay eligible"
    );
}

#[test]
fn temporal_extent_motion_updates_expand_boundary_front_without_center_bias() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [-0.08_f32, 0.0, 0.0, 0.0],
        [0.90_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let updates = render_proxy_temporal_extent_motion_updates(
        &config, &target, &positions, &states, 1.0, 0.20, 1.0, 0.25,
    );

    assert_eq!(
        updates[0][0], 0.0,
        "an active particle exactly at the target center must not get arbitrary one-sided drift"
    );
    assert!(
        updates[1][0] > 0.0 && updates[2][0] < 0.0,
        "local dormant boundary rows should expand symmetrically toward target bounds: {updates:?}"
    );
    assert_eq!(
        updates[3],
        [0.0, 0.0, 0.0],
        "far dormant particles should stay gated by local front"
    );
    assert!(updates.iter().all(|update| {
        (update[0] * update[0] + update[1] * update[1] + update[2] * update[2]).sqrt()
            <= 0.25 + 1.0e-6
    }));
}

#[test]
fn temporal_extent_motion_output_objective_trains_outward_boundary_motion() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [-0.08_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_extent_motion_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.20,
        1.0,
        0.25,
        1.0,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[0], 0.0);
    assert!(
        output_gradients[output_dims] < 0.0,
        "positive-x front row should train a positive x update"
    );
    assert!(
        output_gradients[2 * output_dims] > 0.0,
        "negative-x front row should train a negative x update"
    );
}

#[test]
fn mesh_motion_candidate_weights_track_nonzero_motion_channels() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let mut output_gradients = vec![0.0_f32; 3 * output_dims];
    output_gradients[output_dims] = 1.0e-13;
    output_gradients[2 * output_dims + 1] = -1.0e-3;
    output_gradients[2 * output_dims + config.spatial_dims] = 4.0;

    let weights = mesh_motion_candidate_weights(&config, output_dims, &output_gradients);

    assert_eq!(weights, vec![0.0, 0.0, 1.0]);
}

#[test]
fn mesh_motion_candidate_weights_scale_by_relative_motion_strength() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let mut output_gradients = vec![0.0_f32; 3 * output_dims];
    output_gradients[0] = 0.01;
    output_gradients[output_dims + 1] = -0.25;
    output_gradients[2 * output_dims + 2] = 1.0;

    let weights = mesh_motion_candidate_weights(&config, output_dims, &output_gradients);

    assert_eq!(weights[2], 1.0);
    assert!(
        weights[0] > 0.0 && weights[0] < weights[1] && weights[1] < weights[2],
        "candidate weights should preserve relative mesh-motion strength"
    );
}

#[test]
fn mesh_motion_candidate_weights_floor_keeps_growth_local() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.16_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let motion_weights = vec![0.0_f32; positions.len()];

    let weights = mesh_motion_candidate_weights_with_local_front_floor(
        &config,
        &positions,
        &states,
        0.20,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &motion_weights,
    );

    assert_eq!(
        weights[0], 0.0,
        "already-active rows do not need candidate pressure"
    );
    assert!(
        weights[1] >= DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR
            && weights[2] >= DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        "dormant local-front rows should remain eligible even before mesh gradients reach them"
    );
    assert_eq!(
        weights[3], 0.0,
        "far dormant rows should not receive global activation pressure"
    );
}

#[test]
fn target_coverage_liveness_candidate_weights_prioritize_local_front_coverage_rows() {
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
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;

    let weights = target_coverage_liveness_candidate_weights(
        &config,
        &target,
        &positions,
        &states,
        0.20,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert_eq!(weights[0], 0.0, "active seed row should not be reactivated");
    assert!(
        weights[1] > 0.0,
        "dormant local-front row should receive liveness priority when target coverage needs it: {weights:?}"
    );
    assert_eq!(
        weights[2], 0.0,
        "coverage pressure should not globally activate far dormant rows"
    );
}

#[test]
fn target_coverage_liveness_objective_activates_coverage_front_rows() {
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
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let weights = target_coverage_liveness_candidate_weights(
        &config,
        &target,
        &positions,
        &states,
        0.20,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_candidate_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        &weights,
        1.0,
        0.25,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[liveness_output], 0.0);
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "local-front coverage candidate should train a positive liveness update"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "far dormant rows should stay untouched"
    );
}

#[test]
fn material_coverage_liveness_candidate_weights_prioritize_local_front_rows() {
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
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let activation_weights = vec![0.0_f32, 1.0, 1.0];

    let weights = material_coverage_liveness_candidate_weights(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        &activation_weights,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert_eq!(
        weights[0], 0.0,
        "active visible row should not be reactivated"
    );
    assert!(
        weights[1] > 0.0,
        "dormant local-front row should receive material-coverage liveness priority: {weights:?}"
    );
    assert_eq!(
        weights[2], 0.0,
        "material coverage should not globally activate far dormant rows"
    );
}

#[test]
fn material_coverage_candidates_train_liveness_and_material_updates() {
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
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let activation_weights = vec![0.0_f32, 1.0, 1.0];
    let weights = material_coverage_liveness_candidate_weights(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        &activation_weights,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let mut liveness_gradients = vec![0.0_f32; raw_updates.len()];
    add_candidate_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        &weights,
        1.0,
        0.25,
        &mut liveness_gradients,
    );

    let mut material_gradients = vec![0.0_f32; raw_updates.len()];
    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.0,
        256,
        1.0,
        0.25,
        1.0,
        0.20,
        Some(&weights),
        0.5,
        0.25,
        1.0,
        &mut material_gradients,
    );

    assert!(
        liveness_gradients[output_dims + liveness_output] < 0.0,
        "material-coverage candidate should train a positive liveness update"
    );
    assert!(
        material_gradients[output_dims + material_output] < 0.0,
        "material-coverage candidate should train a positive material-opacity update"
    );
    assert_eq!(liveness_gradients[2 * output_dims + liveness_output], 0.0);
    assert_eq!(material_gradients[2 * output_dims + material_output], 0.0);
}

#[test]
fn material_coverage_materialization_output_objective_promotes_candidate_rows_only() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let rows = 3;
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; rows * config.state_dims];
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut raw_updates = vec![0.0_f32; rows * output_dims];
    raw_updates[liveness_output] = 1.2;
    let candidate_weights = vec![1.0_f32, 0.0, 1.0];
    let mut gradients = vec![0.0_f32; rows * output_dims];

    add_material_coverage_materialization_output_objective(
        &config,
        &states,
        &raw_updates,
        0.75,
        1.0,
        &candidate_weights,
        0.25,
        &mut gradients,
    );

    assert!(
        gradients[material_output] < 0.0,
        "coverage candidate should train a positive material-opacity update"
    );
    assert!(
        gradients[material_output].abs() <= 0.25 + 1.0e-6,
        "materialization should respect the material update cap"
    );
    assert_eq!(
        gradients[output_dims + material_output],
        0.0,
        "noncandidate row should not receive materialization pressure"
    );
    assert_eq!(
        gradients[2 * output_dims + material_output],
        0.0,
        "already visible candidate should not receive more material pressure"
    );
}

#[test]
fn material_target_coverage_opacity_updates_include_bounded_frontier_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.0, -0.1, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.80_f32, 0.0, 0.0, 0.0], [1.60, 0.0, 0.0, 0.0]];
    let states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    let row_weights = vec![1.0_f32, 1.0];

    let updates = material_target_coverage_opacity_updates_weighted(
        &config,
        &target,
        &positions,
        &states,
        Some(&row_weights),
        1.0,
        128,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.25,
    );

    assert!(
        updates[0] > 0.0,
        "near-frontier rows should receive weak material pressure before strict coverage"
    );
    assert!(
        updates[0] <= 0.25 + 1.0e-6,
        "frontier material pressure should respect the update cap"
    );
    assert_eq!(
        updates[1], 0.0,
        "far rows outside the bounded frontier must not receive nonlocal material pressure"
    );
    assert_eq!(
        states[material_channel], GROWTH_3D_INACTIVE_OPACITY_LOGIT,
        "test sanity: material should start inactive"
    );
}

#[test]
fn material_surface_strata_opacity_updates_include_bounded_frontier_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.0, -0.1, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[0.80_f32, 0.0, 0.0, 0.0], [1.60, 0.0, 0.0, 0.0]];
    let states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    let row_weights = vec![1.0_f32, 1.0];

    let updates = material_surface_strata_opacity_updates_weighted(
        &config,
        &target,
        &positions,
        &states,
        Some(&row_weights),
        1.0,
        128,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.25,
    );

    assert!(
        updates[0] > 0.0,
        "frontier rows assigned to an uncovered stratum should receive material pressure"
    );
    assert_eq!(
        updates[1], 0.0,
        "strata material pressure should remain bounded to the mesh frontier"
    );
}

#[test]
fn material_coverage_front_motion_updates_move_potential_local_front_rows() {
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
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let candidate_weights = vec![0.0_f32, 1.0, 1.0];

    let updates = material_coverage_front_motion_updates(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        &candidate_weights,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let local_front_norm = updates[1]
        .iter()
        .take(config.spatial_dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        local_front_norm > 1.0e-5,
        "material-coverage motion should move the dormant local-front row: {updates:?}"
    );
    assert!(
        updates[1][0] > 0.0,
        "uncovered right lobe should pull the local-front row outward: {:?}",
        updates[1]
    );
    assert_eq!(
        updates[2],
        [0.0, 0.0, 0.0],
        "far dormant rows must not receive nonlocal coverage motion"
    );
}

#[test]
fn material_coverage_front_motion_output_objective_uses_training_gradient_sign() {
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
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [-0.92_f32, 0.0, 0.0, 0.0],
        [0.92_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let candidate_weights = vec![0.0_f32, 1.0, 1.0];
    let updates = material_coverage_front_motion_updates(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        &candidate_weights,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_coverage_front_motion_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        &candidate_weights,
        1.0,
        256,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        &mut output_gradients,
    );

    let local_front_base = output_dims;
    let local_front_dot = (0..config.spatial_dims)
        .map(|axis| output_gradients[local_front_base + axis] * updates[1][axis])
        .sum::<f32>();
    assert!(
        local_front_dot < 0.0,
        "training gradient should be raw_update - desired_update for the local front row"
    );
    let far_base = 2 * output_dims;
    for axis in 0..config.spatial_dims {
        assert_eq!(output_gradients[far_base + axis], 0.0);
    }
}

#[test]
fn temporal_liveness_candidate_floor_uses_expanded_local_shell() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let positions = (0..10)
        .map(|row| [row as f32 * 0.08, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let base_weights = vec![0.0_f32; positions.len()];

    let fixed = mesh_motion_candidate_weights_with_local_front_floor(
        &config,
        &positions,
        &states,
        0.05,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &base_weights,
    );
    let temporal = temporal_liveness_candidate_weights_with_local_front_floor(
        &config,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.05,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &fixed,
    );

    assert_eq!(
        fixed[2], 0.0,
        "the generic local-front floor should stay at its one-row default for this tiny cloud"
    );
    assert!(
        (1..=3).all(|row| temporal[row] >= DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR),
        "temporal activation should expose the nearest expanded shell when the rollout is under-active"
    );
    assert_eq!(
        temporal[4], 0.0,
        "the expanded temporal shell should still stay bounded"
    );
}

#[test]
fn motion_memory_output_objective_mirrors_mesh_motion_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_channels = growth_3d_velocity_channels(config.state_dims).unwrap();
    let velocity_outputs = velocity_channels
        .map(|channel| config.spatial_dims + channel)
        .collect::<Vec<_>>();
    let mut mesh_output_gradients = vec![0.0_f32; 2 * output_dims];
    mesh_output_gradients[1] = -0.25;
    mesh_output_gradients[output_dims + 2] = 0.5;
    mesh_output_gradients[output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] = 9.0;
    let mut output_gradients = vec![0.0_f32; mesh_output_gradients.len()];

    add_motion_memory_output_objective(
        &config,
        &mesh_output_gradients,
        DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[velocity_outputs[0]], 0.0);
    assert_eq!(
        output_gradients[velocity_outputs[1]],
        -0.25 * DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[2]],
        0.5 * DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        0.0,
        "non-motion mesh gradients should not leak into velocity memory"
    );
}

#[test]
fn material_coverage_motion_memory_mirrors_local_front_motion_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_output_channels(&config);
    let mut material_motion_gradients = vec![0.0_f32; 3 * output_dims];
    material_motion_gradients[output_dims] = -0.4;
    material_motion_gradients[output_dims + 1] = 0.2;
    material_motion_gradients[2 * output_dims + 2] = 0.75;
    material_motion_gradients[2 * output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        8.0;
    let mut output_gradients = vec![0.0_f32; material_motion_gradients.len()];

    add_motion_memory_output_objective(
        &config,
        &material_motion_gradients,
        DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[velocity_outputs[0]], 0.0);
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        -0.4 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[1]],
        0.2 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[2]],
        0.75 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[0]],
        0.0,
        "non-spatial material/liveness pressure must not leak into velocity memory"
    );
}

#[test]
fn extent_motion_memory_mirrors_front_and_temporal_extent_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_output_channels(&config);
    let mut extent_motion_gradients = vec![0.0_f32; 2 * output_dims];
    let mut temporal_extent_motion_gradients = vec![0.0_f32; 2 * output_dims];
    extent_motion_gradients[0] = -0.2;
    extent_motion_gradients[output_dims + 1] = 0.3;
    temporal_extent_motion_gradients[2] = -0.4;
    temporal_extent_motion_gradients[output_dims] = 0.1;
    temporal_extent_motion_gradients
        [output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] = 9.0;
    let mut output_gradients = vec![0.0_f32; extent_motion_gradients.len()];

    add_extent_motion_memory_output_objective(
        &config,
        &extent_motion_gradients,
        &temporal_extent_motion_gradients,
        DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[velocity_outputs[0]],
        -0.2 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(output_gradients[velocity_outputs[1]], 0.0);
    assert_eq!(
        output_gradients[velocity_outputs[2]],
        -0.4 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        0.1 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[1]],
        0.3 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[2]],
        0.0,
        "non-spatial temporal/liveness pressure must not leak into velocity memory"
    );
}

#[test]
fn liveness_phase_memory_mirrors_liveness_pressure_only() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let phase_output = config.spatial_dims + growth_3d_phase_channel(config.state_dims).unwrap();
    let mut liveness_gradients = vec![0.0_f32; 3 * output_dims];
    liveness_gradients[liveness_output] = -0.8;
    liveness_gradients[output_dims + liveness_output] = 0.4;
    liveness_gradients[2 * output_dims] = -1.0;
    liveness_gradients[2 * output_dims + config.spatial_dims + 1] = 0.25;
    let mut output_gradients = vec![0.0_f32; liveness_gradients.len()];

    add_liveness_phase_memory_output_objective(
        &config,
        &liveness_gradients,
        DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[phase_output],
        -0.8 * DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        "activation pressure should train phase memory upward under SGD"
    );
    assert_eq!(
        output_gradients[output_dims + phase_output],
        0.4 * DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        "suppression pressure should also be mirrored so phase can decay"
    );
    assert_eq!(
        output_gradients[2 * output_dims + phase_output],
        0.0,
        "non-liveness output pressure must not leak into phase memory"
    );
}

#[test]
fn mesh_residual_velocity_objective_targets_active_and_local_front_velocity() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_channels(config.state_dims)
        .unwrap()
        .map(|channel| config.spatial_dims + channel)
        .collect::<Vec<_>>();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_residual_velocity_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.0,
        0.0,
        0.25,
        0.20,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[velocity_outputs[0]] < -1.0e-4,
        "active row should train positive x velocity toward mesh residual"
    );
    assert!(
        output_gradients[output_dims + velocity_outputs[0]] < -1.0e-4,
        "dormant local-front row should also train velocity"
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[0]],
        0.0,
        "far dormant row should not receive residual velocity pressure"
    );
    assert_eq!(output_gradients[velocity_outputs[1]], 0.0);
    assert_eq!(output_gradients[velocity_outputs[2]], 0.0);
}

#[test]
fn growth_phase_output_objective_targets_active_and_local_front_rows() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let phase_output = config.spatial_dims + phase_channel;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[2 * config.state_dims + phase_channel] = 0.25;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_growth_phase_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.5,
        1.0,
        0.20,
        &mut output_gradients,
    );

    assert!(
        output_gradients[phase_output] < 0.0,
        "active rows should be trained to advance the local phase state"
    );
    assert!(
        output_gradients[output_dims + phase_output] < 0.0,
        "dormant local-front rows should receive phase precursor pressure"
    );
    assert!(
        output_gradients[2 * output_dims + phase_output] > 0.0,
        "far dormant phase leakage should be suppressed"
    );
}

#[test]
fn temporal_liveness_output_objective_bounds_nearest_shell_expansion() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = (0..10)
        .map(|row| [row as f32 * 0.10, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.5,
        1.0,
        0.05,
        &mut output_gradients,
    );

    assert!(
        (1..=3).all(|row| output_gradients[row * output_dims + liveness_output] < 0.0),
        "under-active temporal objectives should train a bounded nearest shell instead of the whole schedule deficit"
    );
    assert_eq!(
        output_gradients[4 * output_dims + liveness_output],
        0.0,
        "rows outside the bounded nearest shell should remain untouched unless they predict nonlocal activation"
    );
}

#[test]
fn temporal_liveness_output_objective_suppresses_nonlocal_liveness_drift() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [1.25_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[2 * output_dims + liveness_output] = 1.0;
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.5,
        1.0,
        0.20,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "local front row should still receive positive-activation training"
    );
    assert!(
        output_gradients[2 * output_dims + liveness_output] > 0.0,
        "far dormant rows with positive liveness drift should be trained back toward dormancy"
    );
}

#[test]
fn temporal_liveness_output_objective_suppresses_newly_predicted_burst_rows() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]; 10];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    for row in 2..positions.len() {
        raw_updates[row * output_dims + liveness_output] = 8.5;
    }
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_liveness_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.25,
        1.0,
        0.20,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[liveness_output], 0.0,
        "already-active seed rows should be preserved before newly predicted burst rows"
    );
    assert_eq!(output_gradients[output_dims + liveness_output], 0.0);
    assert!(
        (2..positions.len()).any(|row| output_gradients[row * output_dims + liveness_output] > 0.0),
        "newly predicted burst rows should receive positive gradients that suppress their liveness update"
    );
}

#[test]
fn temporal_activation_jump_adjoint_retimes_late_burst_to_previous_front() {
    let config = NpaConfig::growing_3dgs();
    let positions = (0..10)
        .map(|row| [row as f32 * 0.04, 0.0_f32, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut previous_states =
        vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    let mut current_states = previous_states.clone();
    for row in 0..5 {
        previous_states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.5;
    }
    for row in 0..10 {
        current_states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = if row < 5 {
            0.5
        } else {
            -0.2 + row as f32 * 0.01
        };
    }
    let mut previous_adjoint = vec![0.0; previous_states.len()];
    let mut current_adjoint = vec![0.0; current_states.len()];

    add_temporal_activation_jump_state_adjoint(
        &config,
        &positions,
        &previous_states,
        &current_states,
        1.0,
        0.20,
        0.50,
        0.60,
        0.50,
        &mut previous_adjoint,
        &mut current_adjoint,
    );

    let previous_liveness =
        |row: usize| previous_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    let current_liveness =
        |row: usize| current_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL];
    assert_eq!(
        previous_liveness(0),
        0.0,
        "already-active core rows should not receive burst retiming pressure"
    );
    assert!(
        previous_liveness(5) < 0.0,
        "a particle that appears in a late burst should be trained to activate at the previous local front"
    );
    assert!(
        current_liveness(5) > 0.0,
        "the later burst snapshot should also receive suppression for the same weakly active row"
    );
    assert_eq!(
        previous_liveness(9),
        0.0,
        "non-front dormant rows should not get global activation pressure from burst retiming"
    );
}

#[test]
fn liveness_front_temporal_targets_grow_local_front_and_suppress_overactive_rows() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.9;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = -0.2;
    states[3 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.8;

    let updates =
        liveness_front_temporal_target_updates(&config, &positions, &states, 1.0, 0.20, 0.1, 0.05);

    assert_eq!(
        updates[0], 0.0,
        "strong seed/core liveness should be preserved by early temporal suppression"
    );
    assert!(
        updates[1] > 0.0,
        "dormant particle near the active front should receive local growth pressure"
    );
    assert!(
        updates[2] < 0.0,
        "weak overactive far row should be suppressed under the early activation schedule"
    );
    assert!(
        updates[3] < 0.0,
        "stricter early temporal scheduling should suppress the second excess active row"
    );
    assert!(
        updates.iter().all(|value| value.abs() <= 0.05 + 1.0e-6),
        "liveness target updates should respect max_update"
    );
}

#[test]
fn surface_escape_state_adjoint_suppresses_only_escaped_active_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [2.0_f32, 0.0, 0.0, 0.0],
        [2.0_f32, 0.1, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut adjoint = vec![0.0; states.len()];

    add_surface_escape_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        0.5,
        0.25,
        0.05,
        &mut adjoint,
    );

    assert_eq!(adjoint[GROWTH_3D_LIVENESS_CHANNEL], 0.0);
    assert_eq!(adjoint[opacity_channel], 0.0);
    assert!(
        adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > 0.0,
        "escaped active particles should receive positive liveness suppression"
    );
    assert!(
        adjoint[config.state_dims + opacity_channel] > 0.0,
        "escaped active particles should receive positive material-opacity suppression"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "escaped dormant particles should not be suppressed again"
    );
    assert_eq!(adjoint[2 * config.state_dims + opacity_channel], 0.0);
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.05 + 1.0e-6),
        "surface escape state adjoints should respect max_adjoint"
    );
}
