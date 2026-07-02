use super::*;

#[test]
fn mesh_geometry_output_objective_targets_active_surface_motion() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [0.2_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_geometry_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        64,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.1,
        0.0,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[0] < 0.0,
        "active row should train immediate x motion toward target support"
    );
    assert!(
        output_gradients[0].abs() <= 0.1 + 1.0e-6,
        "geometry output target should respect max_update_norm"
    );
    assert_eq!(
        output_gradients[output_dims], 0.0,
        "dormant row should not receive mesh geometry motion pressure"
    );
}

#[test]
fn render_proxy_target_extent_updates_expand_weighted_active_bounds() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [-0.10_f32, 0.0, 0.0, 0.0],
        [0.10_f32, 0.0, 0.0, 0.0],
        [0.12_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let row_weights = vec![1.0, 1.0, 0.5, 0.0];

    let updates =
        render_proxy_target_extent_updates(&config, &target, &positions, &row_weights, 0.25, 1.0);

    assert!(
        updates[0][0] < -1.0e-4,
        "weighted min-x boundary should expand toward target min x"
    );
    assert!(
        updates[1][0] > 1.0e-4,
        "weighted max-side active row should expand toward target max x"
    );
    assert!(
        updates[2][0] > 1.0e-4,
        "near-front weighted row should share extent pressure"
    );
    assert_eq!(
        updates[3], [0.0; 3],
        "zero-weight far dormant row should not receive extent pressure"
    );
}

#[test]
fn extent_front_motion_updates_push_dormant_front_outward() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.12_f32, 0.0, 0.0, 0.0],
        [-0.12_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for row in 1..positions.len() {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
            GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    }

    let updates = render_proxy_extent_front_motion_updates(
        &config, &target, &positions, &states, 0.20, 0.25, 1.0,
    );

    assert!(
        updates[1][0] > 1.0e-4,
        "positive-x dormant local-front row should move toward remaining target max x"
    );
    assert!(
        updates[2][0] < -1.0e-4,
        "negative-x dormant local-front row should move toward remaining target min x"
    );
    assert_eq!(
        updates[0], [0.0; 3],
        "already-active seed row should not receive extent-front dormant motion"
    );
    assert_eq!(
        updates[3], [0.0; 3],
        "far dormant row outside the local front should not receive extent-front motion"
    );
}

#[test]
fn extent_front_motion_output_objective_trains_outward_motion_without_target_seats() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -0.1, 0.0], [1.0, -0.1, 0.0], [0.0, 0.1, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.12_f32, 0.0, 0.0, 0.0],
        [-0.12_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for row in 1..positions.len() {
        states[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
            GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    }
    let output_dims = config.update_dims();
    let raw_updates = vec![0.0; positions.len() * output_dims];
    let mut output_gradients = vec![0.0; positions.len() * output_dims];

    add_extent_front_motion_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.20,
        0.25,
        1.0,
        1.0,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[0], 0.0,
        "active row should not receive dormant extent-front output pressure"
    );
    assert!(
        output_gradients[output_dims] < -1.0e-4,
        "positive-x dormant front row should train positive x motion"
    );
    assert!(
        output_gradients[2 * output_dims] > 1.0e-4,
        "negative-x dormant front row should train negative x motion"
    );
}

#[test]
fn local_front_expansion_updates_push_dormant_front_outward() {
    let config = NpaConfig::growing_3dgs();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [-0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[3 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let row_weights = vec![1.0, 0.75, 0.5, 0.0];

    let updates = render_proxy_local_front_expansion_updates(
        &config,
        &positions,
        &states,
        &row_weights,
        0.2,
        1.0,
    );

    assert_eq!(
        updates[0], [0.0; 3],
        "already-active seed should not receive front expansion"
    );
    assert!(
        updates[1][0] > 0.0,
        "positive-x dormant front should expand away from active support"
    );
    assert!(
        updates[2][0] < 0.0,
        "negative-x dormant front should expand away from active support"
    );
    assert_eq!(
        updates[3], [0.0; 3],
        "zero-weight far dormant row should not receive front expansion"
    );
    assert!(updates[1][0].abs() > updates[2][0].abs());
}

#[test]
fn mesh_geometry_output_objective_targets_local_front_motion() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_geometry_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        64,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.1,
        0.20,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims] < 0.0,
        "near-front dormant row should train local motion toward target support"
    );
    assert_eq!(
        output_gradients[2 * output_dims],
        0.0,
        "far dormant row should stay unguided until the local front reaches it"
    );
}
