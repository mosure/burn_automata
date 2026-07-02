use super::*;

#[test]
fn temporal_materialization_output_objective_grows_only_local_front_candidates() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.10_f32, 0.0, 0.0, 0.0],
        [0.75_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let candidate_weights = vec![0.0_f32, 1.0, 0.0, 1.0];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_temporal_materialization_output_objective_with_candidate_weights(
        &config,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.20,
        &candidate_weights,
        0.25,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "under-active local-front candidate should train material output upward"
    );
    assert!(
        output_gradients[output_dims + material_output].abs() <= 0.25 + 1.0e-6,
        "temporal materialization should respect the material update cap"
    );
    assert_eq!(
        output_gradients[2 * output_dims + material_output],
        0.0,
        "local-front row without candidate weight should not be materialized"
    );
    assert_eq!(
        output_gradients[3 * output_dims + material_output],
        0.0,
        "far dormant row should not receive global materialization pressure"
    );
}

#[test]
fn temporal_materialization_target_follows_rollout_schedule() {
    let early = temporal_materialization_target_logit(0.0);
    let mid = temporal_materialization_target_logit(0.5);
    let late = temporal_materialization_target_logit(1.0);

    assert_eq!(early, GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0);
    assert!(mid > early);
    assert!(late > mid);
    assert_eq!(late, GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET);
}

#[test]
fn active_surface_materialization_promotes_active_surface_rows_only() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![
        [0.0_f32, 0.0, 0.01, 0.0],
        [0.0_f32, 0.0, 0.90, 0.0],
        [0.04_f32, 0.0, 0.01, 0.0],
        [0.80_f32, 0.0, 0.01, 0.0],
        [0.0_f32, 0.0, 1.40, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[3 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[2 * output_dims + liveness_output] = 8.0;
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_active_surface_materialization_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.10,
        0.0,
        None,
        &mut output_gradients,
    );

    assert!(
        output_gradients[material_output] < 0.0,
        "active near-surface row should train material output upward"
    );
    assert_eq!(
        output_gradients[4 * output_dims + material_output],
        0.0,
        "active row outside the bounded surface frontier should not receive material pressure"
    );
    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "active row inside the bounded surface frontier should materialize before coverage takes over"
    );
    assert!(
        output_gradients[2 * output_dims + material_output] < 0.0,
        "predicted-active near-surface row should materialize before visible-only coverage takes over"
    );
    assert_eq!(
        output_gradients[3 * output_dims + material_output],
        0.0,
        "dormant non-front row should not receive global material pressure"
    );
    assert!(
        output_gradients[material_output].abs() <= 0.10 + 1.0e-6,
        "active surface materialization should respect the material update cap"
    );
}

#[test]
fn active_surface_materialization_respects_local_front_candidate_weights() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![
        [0.0_f32, 0.0, 0.01, 0.0],
        [0.08_f32, 0.0, 0.01, 0.0],
        [0.10_f32, 0.0, 0.01, 0.0],
        [0.80_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    for row in 1..positions.len() {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
            GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    }
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let candidate_weights = vec![0.0_f32, 1.0, 0.0, 1.0];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_active_surface_materialization_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.10,
        0.20,
        Some(&candidate_weights),
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "local-front row with candidate weight should receive material pressure"
    );
    assert_eq!(
        output_gradients[2 * output_dims + material_output],
        0.0,
        "local-front row with zero candidate weight should not materialize"
    );
    assert_eq!(
        output_gradients[3 * output_dims + material_output],
        0.0,
        "far dormant row should remain excluded even with candidate weight"
    );
}

#[test]
fn material_visible_liveness_output_objective_activates_local_front_material_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![
        [0.0_f32, 0.0, 0.01, 0.0],
        [0.08_f32, 0.0, 0.01, 0.0],
        [0.8_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[2 * output_dims + material_output] = 8.5;
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visible_liveness_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        material_training_soft_coverage_threshold(1.0),
        0.1,
        0.20,
        1.0,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[liveness_output], 0.0,
        "already-live rows should not receive material-visible liveness pressure"
    );
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "near-front surface row should train liveness output upward before material visibility is learned"
    );
    assert!(
        output_gradients[output_dims + liveness_output].abs() <= 0.1 + 1.0e-6,
        "material-visible liveness output should respect max_liveness_update"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "far dormant predicted-material row should remain local-front gated"
    );
}
