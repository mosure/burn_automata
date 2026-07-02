use super::*;

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
    assert!(
        weights[1] > 0.99,
        "strongest local coverage deficit should normalize to full candidate weight: {weights:?}"
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
    assert!(
        weights[1] > 0.99,
        "strongest local material-coverage deficit should normalize to full candidate weight: {weights:?}"
    );
    assert_eq!(
        weights[2], 0.0,
        "material coverage should not globally activate far dormant rows"
    );
}

#[test]
fn coverage_liveness_weight_normalizer_uses_observed_deficit_scale() {
    let updates = [[0.05_f32, 0.0, 0.0], [0.10, 0.0, 0.0]];

    let observed = coverage_update_weight_normalizer(&updates, 3, 1.0);
    assert!(
        (observed - 0.10).abs() < 1.0e-6,
        "normalizer should use observed coverage pressure when the global cap is much larger"
    );

    let capped = coverage_update_weight_normalizer(&updates, 3, 0.075);
    assert!(
        (capped - 0.075).abs() < 1.0e-6,
        "normalizer should still respect a smaller explicit motion cap"
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
