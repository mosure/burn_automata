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
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
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
fn learned_scale_budget_output_objective_penalizes_predicted_oversize() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let scale_channel = config.state_dims - 5;
    let scale_output = config.spatial_dims + scale_channel;
    let states = vec![0.0; 2 * config.state_dims];
    let mut raw_updates = vec![0.0; 2 * output_dims];
    raw_updates[scale_output] = 1.0;
    raw_updates[output_dims + scale_output] = -1.0;
    let mut output_gradients = vec![0.0; 2 * output_dims];
    let render = RenderLossConfig {
        gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
        sigma: 1.0,
        min_sigma: 0.25,
        max_sigma: 6.0,
        ..RenderLossConfig::default()
    };

    let detected_output = add_gaussian_scale_budget_output_objective(
        &config,
        &states,
        &raw_updates,
        render,
        0.5,
        0.25,
        &mut output_gradients,
    );

    assert_eq!(detected_output, Some(scale_output));
    assert!(
        output_gradients[scale_output] > raw_updates[scale_output],
        "oversized predicted scale should train a negative scale update"
    );
    assert_eq!(
        output_gradients[output_dims + scale_output],
        0.0,
        "undersized predicted scale should not be penalized by the oversize budget"
    );

    let mut fixed_scale_gradients = vec![0.0; 2 * output_dims];
    assert_eq!(
        add_gaussian_scale_budget_output_objective(
            &config,
            &states,
            &raw_updates,
            RenderLossConfig::default(),
            0.5,
            0.25,
            &mut fixed_scale_gradients,
        ),
        None
    );
    assert!(fixed_scale_gradients.iter().all(|value| *value == 0.0));
}
