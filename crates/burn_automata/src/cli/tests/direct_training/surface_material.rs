use super::*;

#[test]
fn surface_position_adjoint_moves_only_active_particles_toward_mesh() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let mut states = vec![0.0; 2 * config.state_dims];
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![[0.0; 4]; 2];

    add_surface_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        0.0,
        &mut adjoint,
    );

    assert!(adjoint[0][0].abs() <= 1.0e-6);
    assert!(adjoint[0][1].abs() <= 1.0e-6);
    assert!(adjoint[0][2] > 0.49 && adjoint[0][2] < 0.51);
    assert_eq!(adjoint[1], [0.0; 4]);
}
#[test]
fn surface_projection_updates_boost_escaped_active_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 0.5, 0.0],
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 2.0, 0.0],
        [0.0, 0.0, GROWTH_3D_SURFACE_MAX_DISTANCE * 2.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let base = render_proxy_surface_projection_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        0.0,
        f32::INFINITY,
    );
    let boosted = render_proxy_surface_projection_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1.0,
        f32::INFINITY,
    );

    assert!((boosted[0][2] - base[0][2]).abs() <= 1.0e-6);
    assert!(
        boosted[1][2] < base[1][2],
        "escaped active particle should receive stronger pull toward the surface"
    );
    assert_eq!(boosted[2], [0.0; 3]);
}
#[test]
fn material_visible_surface_approach_updates_pull_visible_active_particles_toward_mesh() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 0.5, 0.0, 0.25, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "render-visible active material should receive generic projection motion toward the mesh"
    );
    assert!(
        updates[0][0] <= 0.25 + 1.0e-6,
        "material-visible projection update should respect max_update_norm"
    );
}
#[test]
fn material_visible_surface_approach_updates_do_not_move_far_dormant_material() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 0.5, 0.0, 0.25, 1.0, 0.20, None,
    );

    assert!(
        updates[1][0] > 1.0e-4,
        "visible material in the local front should receive bounded projection motion"
    );
    assert_eq!(
        updates[2], [0.0; 3],
        "far dormant visible material should not receive global target assignment motion"
    );
}
#[test]
fn material_surface_candidate_approach_moves_active_rows_before_visibility() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.18, -0.1, 0.0], [0.18, 0.1, 0.0], [0.18, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [1.6_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 1.0, 0.0, 1.0, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "active near-surface material candidate should get projection motion before it is visible"
    );
    assert_eq!(
        updates[1], [0.0; 3],
        "dormant material candidate outside the bounded frontier should not get global projection motion"
    );
}
#[test]
fn material_surface_candidate_approach_moves_frontier_rows_before_visibility() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[0.80, -0.1, 0.0], [0.80, 0.1, 0.0], [0.80, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0], [-0.80_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_visible_surface_approach_updates(
        &config, &target, &positions, &states, None, 1.0, 0.0, 1.0, 1.0, 0.20, None,
    );

    assert!(
        updates[0][0] > 1.0e-4,
        "active material candidate inside the bounded frontier should get projection motion before strict material coverage"
    );
    assert_eq!(
        updates[1], [0.0; 3],
        "active rows outside the bounded frontier should not get global material projection motion"
    );
}
#[test]
fn material_surface_candidate_coverage_uses_predicted_active_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-0.18, -0.1, 0.0],
            [-0.18, 0.1, 0.0],
            [-0.18, 0.0, 0.2],
            [0.18, -0.1, 0.0],
            [0.18, 0.1, 0.0],
            [0.18, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let positions = vec![[-0.18_f32, 0.0, 0.05, 0.0], [-0.12_f32, 0.0, 0.05, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[liveness_output] = 8.0;
    let weights = material_surface_candidate_row_weights(
        &config,
        &target,
        &positions,
        &states,
        Some(&raw_updates),
        1.0,
        0.0,
        None,
    );

    assert!(
        weights[0] > 0.0,
        "predicted-active near-surface row should become eligible for material surface coverage"
    );
    assert_eq!(
        weights[1], 0.0,
        "dormant non-predicted row should remain ineligible without local-front pressure"
    );
}
#[test]
fn material_visible_surface_approach_output_objective_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; output_dims];

    add_material_visible_surface_approach_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.5,
        0.0,
        0.25,
        1.0,
        0.20,
        None,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[0] < -1.0e-4,
        "positive target x motion should train the x output upward under SGD"
    );
    assert_eq!(
        output_gradients[config.spatial_dims + material_channel],
        0.0,
        "surface approach objective should not directly train material opacity"
    );
}
#[test]
fn material_visible_surface_position_adjoint_tracks_visible_local_front_only() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut adjoint = vec![[0.0_f32; 4]; positions.len()];

    add_material_visible_surface_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        0.0,
        1.0,
        0.20,
        &mut adjoint,
    );

    assert!(
        adjoint[1][0] < -1.0e-4,
        "local-front visible material should receive position adjoint opposite the target motion"
    );
    assert_eq!(
        adjoint[2], [0.0; 4],
        "far dormant visible material should not receive nonlocal surface adjoint"
    );
}
#[test]
fn material_visible_surface_row_weights_include_local_front_but_not_far_dormant() {
    let config = NpaConfig::growing_3dgs();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for row in 0..positions.len() {
        states[row * config.state_dims + material_channel] =
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    }

    let weights =
        material_visible_surface_row_weights(&config, &positions, &states, None, 0.20, None);

    assert_eq!(weights[0], 1.0);
    assert!(
        weights[1] > 0.0,
        "visible material inside the local front should be eligible"
    );
    assert_eq!(
        weights[2], 0.0,
        "far dormant visible material should not become globally eligible"
    );
}
#[test]
fn material_visible_surface_coverage_updates_move_visible_rows_to_uncovered_bins() {
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
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.02_f32, -0.04, 0.05, 0.0],
        [-1.00_f32, 0.04, 0.05, 0.0],
        [-0.98_f32, 0.0, 0.08, 0.0],
        [-1.04_f32, 0.0, 0.02, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for row in 0..positions.len() {
        states[row * config.state_dims + material_channel] =
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    }

    let updates = material_visible_surface_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        None,
        1.0,
        512,
        10.0,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        0.20,
        None,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.15),
        "material-visible coverage should relocate redundant visible rows toward uncovered support: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}
#[test]
fn material_visible_surface_coverage_output_objective_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; output_dims];

    add_material_visible_surface_coverage_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        512,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.20,
        None,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[0] < -1.0e-4,
        "positive material-visible coverage target motion should train x output upward"
    );
}
#[test]
fn material_visible_surface_coverage_position_adjoint_tracks_visible_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let mut adjoint = vec![[0.0_f32; 4]; positions.len()];

    add_material_visible_surface_coverage_position_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        0.25,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.20,
        &mut adjoint,
    );

    assert!(
        adjoint[0][0] < -1.0e-4 || adjoint[1][0] < -1.0e-4,
        "visible active/local-front support should receive adjoint opposite material-visible coverage target motion: {adjoint:?}"
    );
    assert_eq!(
        adjoint[2], [0.0; 4],
        "far dormant visible row outside the nearest local shell should not receive nonlocal material-visible coverage adjoint"
    );
}
