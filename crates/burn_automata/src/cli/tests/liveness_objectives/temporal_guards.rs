use super::*;

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
