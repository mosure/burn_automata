use super::*;

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
