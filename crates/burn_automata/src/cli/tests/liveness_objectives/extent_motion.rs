use super::*;

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
