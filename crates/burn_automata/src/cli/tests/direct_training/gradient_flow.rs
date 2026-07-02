use super::*;

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
fn terminal_render_adjoint_weighting_blocks_far_dormant_shortcut_gradients() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let trace = crate::RolloutTrace {
        positions: positions.clone(),
        states,
        batch_size: 1,
        particle_count: positions.len(),
        state_dims: config.state_dims,
        steps: 0,
        mean_dx: Vec::new(),
    };
    let gradient = RenderProxyGradientRows {
        row_indices: vec![0, 1, 2],
        gradients: vec![[0.5, 0.0, 0.0]; 3],
        opacity_gradients: vec![1.0; 3],
        scale_gradients: vec![0.0; 3],
        color_gradients: vec![[0.25, 0.0, 0.0]; 3],
    };
    let coverage = vec![[0.0; 3]; positions.len()];
    let weights = terminal_render_locality_weights(&config, &trace.positions, &trace.states, 0.15);

    assert_eq!(weights[0], 1.0);
    assert!(
        weights[1] > 0.0,
        "nearest dormant shell should remain trainable as local growth front"
    );
    assert_eq!(
        weights[2], 0.0,
        "far dormant particles must not receive terminal render shortcuts"
    );

    let position_adjoint = terminal_render_position_adjoint_weighted(
        &config,
        &trace,
        &gradient,
        &coverage,
        2.0,
        false,
        3,
        Some(&weights),
    );
    assert!(
        position_adjoint[0][0] > position_adjoint[1][0] && position_adjoint[1][0] > 0.0,
        "active rows receive full render motion and front rows receive attenuated local motion"
    );
    assert_eq!(
        position_adjoint[2], [0.0; 4],
        "far dormant row should not be pulled toward a terminal target seat"
    );

    let state_adjoint = terminal_render_state_adjoint_weighted(
        &config,
        &trace,
        &gradient,
        1.0,
        0.0,
        0.0,
        0.0,
        0.15,
        1.0,
        0.0,
        RenderLossConfig::default(),
        3,
        Some(&weights),
    );
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let tail = config.state_dims - 3;
    assert!(state_adjoint[material_channel] > 0.0);
    assert!(state_adjoint[config.state_dims + material_channel] > 0.0);
    assert_eq!(state_adjoint[2 * config.state_dims + material_channel], 0.0);
    assert!(state_adjoint[tail] > 0.0);
    assert!(state_adjoint[config.state_dims + tail] > 0.0);
    assert_eq!(state_adjoint[2 * config.state_dims + tail], 0.0);
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
fn output_gradient_state_caps_preserve_materialization_signal() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_output =
        config.spatial_dims + growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut gradients = vec![0.0_f32; 4 * output_dims];
    for row in 0..4 {
        gradients[row * output_dims] = 10.0;
        gradients[row * output_dims + liveness_output] = 8.0;
        gradients[row * output_dims + material_output] = 7.0;
    }

    let capped = cap_output_gradient_channel_rms_with_state_caps(
        &config,
        &mut gradients,
        output_dims,
        2.0,
        6.0,
        5.0,
    );

    assert_eq!(capped, 3);
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
    let material_rms = ((0..4)
        .map(|row| gradients[row * output_dims + material_output].powi(2))
        .sum::<f32>()
        / 4.0)
        .sqrt();

    assert!(motion_rms <= 2.0 + 1.0e-6);
    assert!(
        (liveness_rms - 6.0).abs() <= 1.0e-6,
        "liveness cap should remain independently configurable"
    );
    assert!(
        (material_rms - 5.0).abs() <= 1.0e-6,
        "material cap should keep visible-material supervision from being capped like motion"
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
