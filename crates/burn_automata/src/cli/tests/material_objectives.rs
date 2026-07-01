use super::*;

#[test]
fn surface_escape_liveness_output_objective_suppresses_escaped_active_particles() {
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
    let output_dims = config.update_dims();
    let raw_updates = vec![0.0; positions.len() * output_dims];
    let mut output_gradients = vec![0.0; raw_updates.len()];
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;

    add_surface_escape_liveness_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.25,
        0.05,
        1.0,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[liveness_output], 0.0);
    assert!(
        output_gradients[output_dims + liveness_output] > 0.0,
        "escaped active particles should train negative liveness updates"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "escaped dormant particles should not receive active-surface suppression"
    );
    assert!(
        output_gradients
            .iter()
            .all(|value| value.abs() <= 0.05 + 1.0e-6),
        "surface escape liveness objective should respect max update"
    );
}

#[test]
fn surface_material_opacity_adjoint_targets_visible_surface_particles() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let sample = target.surface_sample(0);
    let near_surface = [
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ];
    let positions = vec![
        near_surface,
        [2.0_f32, 0.0, 0.0, 0.0],
        near_surface,
        near_surface,
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    states[material_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[3 * config.state_dims + material_channel] =
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET + 2.0;
    let mut adjoint = vec![0.0; states.len()];

    add_surface_material_opacity_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        0.5,
        0.2,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.1,
        &mut adjoint,
    );

    assert!(
        adjoint[material_channel] < 0.0,
        "active low-opacity surface particles should be promoted"
    );
    assert_eq!(
        adjoint[GROWTH_3D_LIVENESS_CHANNEL], 0.0,
        "material opacity pressure must not alter liveness directly"
    );
    assert_eq!(
        adjoint[config.state_dims + material_channel],
        0.0,
        "far active particles should not receive surface material pressure"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + material_channel],
        0.0,
        "dormant surface particles should not receive material pressure"
    );
    assert!(
        adjoint[3 * config.state_dims + material_channel] > 0.0,
        "oversaturated visible particles should receive damping pressure"
    );
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.1 + 1.0e-6),
        "surface material opacity adjoints should respect max_adjoint"
    );
}

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

#[test]
fn material_target_coverage_updates_promote_assigned_surface_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [0.0_f32, 0.0, 0.01, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
        [0.0_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_target_coverage_opacity_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        64,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );

    assert!(
        (updates[0] - 0.05).abs() <= 1.0e-6,
        "assigned live surface row should receive clamped visible-material promotion"
    );
    assert_eq!(
        updates[1], 0.0,
        "far live row should not be made visible without covering target samples"
    );
    assert_eq!(
        updates[2], 0.0,
        "dormant row should not be made render-visible by material coverage"
    );
}

#[test]
fn material_target_coverage_updates_promote_soft_assigned_approaching_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let seed_scale = 1.0;
    let strict = target_coverage_threshold(seed_scale);
    let soft = material_training_soft_coverage_threshold(seed_scale);
    assert!(strict < 0.30 && soft > 0.30);
    let positions = vec![[0.0_f32, 0.0, 0.30, 0.0], [0.0_f32, 0.0, soft + 0.20, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_target_coverage_opacity_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        64,
        seed_scale,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );

    assert!(
        updates[0] > 0.0,
        "approaching active row inside the soft material radius should receive visibility pressure"
    );
    assert_eq!(
        updates[1], 0.0,
        "rows outside the soft material radius should not be made visible globally"
    );
}

#[test]
fn weighted_material_target_coverage_updates_can_promote_local_front_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.8_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    for state in states.chunks_exact_mut(config.state_dims) {
        state[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }

    let active_only = material_target_coverage_opacity_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        64,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );
    let weighted = material_target_coverage_opacity_updates_weighted(
        &config,
        &target,
        &positions,
        &states,
        Some(&[0.5, 0.0]),
        1.0,
        64,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );

    assert_eq!(
        active_only[0], 0.0,
        "active-only material coverage should not globally wake dormant rows"
    );
    assert!(
        weighted[0] > 0.0,
        "weighted local-front row should receive bounded material coverage pressure"
    );
    assert_eq!(
        weighted[1], 0.0,
        "zero-weight far row should remain untouched"
    );
}

#[test]
fn material_target_coverage_adjoint_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_material_target_coverage_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        64,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
        &mut adjoint,
    );

    assert!(
        (adjoint[material_channel] + 0.05).abs() <= 1.0e-6,
        "positive supervised material update should become a negative state adjoint"
    );
    assert_eq!(
        adjoint[GROWTH_3D_LIVENESS_CHANNEL], 0.0,
        "material target coverage must not directly change liveness"
    );
}

#[test]
fn material_visibility_output_objective_promotes_approaching_active_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![[0.0_f32, 0.0, 0.30, 0.0], [0.0_f32, 0.0, 0.90, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.0,
        0.0,
        64,
        1.0,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.0,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[material_output] < 0.0,
        "approaching active row should train immediate material output upward"
    );
    assert_eq!(
        output_gradients[output_dims + material_output],
        0.0,
        "far active row should not receive global material output pressure"
    );
}

#[test]
fn material_visibility_output_objective_promotes_local_front_rows() {
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
        [0.8_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.0,
        64,
        1.0,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "near-front dormant row should train immediate material output upward"
    );
    assert_eq!(
        output_gradients[2 * output_dims + material_output],
        0.0,
        "far dormant row should not receive global material visibility pressure"
    );
    assert!(
        output_gradients[output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] < 0.0,
        "near-front material promotion should also train bounded liveness output"
    );
}

#[test]
fn material_visibility_output_objective_couples_local_front_to_mesh_motion() {
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
        [0.08_f32, 0.0, 0.01, 0.0],
        [0.09_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];
    let activation_weights = vec![1.0_f32, 0.0, 1.0];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.0,
        64,
        1.0,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        Some(&activation_weights),
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[output_dims + material_output],
        0.0,
        "local-front row without mesh-motion activation weight should not be material-promoted"
    );
    assert_eq!(
        output_gradients[output_dims + liveness_output],
        0.0,
        "local-front row without mesh-motion activation weight should not be preactivated"
    );
    assert!(
        output_gradients[2 * output_dims + material_output] < 0.0,
        "equally local front row with mesh-motion activation weight should be material-promoted"
    );
    assert!(
        output_gradients[2 * output_dims + liveness_output] < 0.0,
        "mesh-coupled material promotion should also provide bounded liveness pressure"
    );
}

#[test]
fn material_visibility_output_objective_preactivates_offsurface_local_front_rows() {
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
        [0.0_f32, 0.0, 0.90, 0.0],
        [0.08_f32, 0.0, 0.90, 0.0],
        [0.8_f32, 0.0, 0.90, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.0,
        64,
        1.0,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "off-surface local-front row should receive material preactivation while it is being moved"
    );
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "off-surface local-front material preactivation should also provide bounded liveness pressure"
    );
    assert_eq!(
        output_gradients[2 * output_dims + material_output],
        0.0,
        "off-surface far dormant row should not receive material preactivation"
    );
    assert_eq!(
        output_gradients[2 * output_dims + liveness_output],
        0.0,
        "off-surface far dormant row should not receive liveness preactivation"
    );
}

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
        output_gradients[output_dims + material_output],
        0.0,
        "active row outside the soft surface band should not receive material pressure"
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

#[test]
fn material_visibility_output_objective_respects_material_update_cap() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut low_cap_gradients = vec![0.0_f32; raw_updates.len()];
    let mut high_cap_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.0,
        0.0,
        64,
        1.0,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.0,
        None,
        1.0,
        0.1,
        1.0,
        &mut low_cap_gradients,
    );
    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.0,
        0.0,
        64,
        1.0,
        ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.0,
        None,
        1.0,
        0.1,
        1.0,
        &mut high_cap_gradients,
    );

    assert!(
        high_cap_gradients[material_output].abs() > low_cap_gradients[material_output].abs() * 4.0,
        "material-specific cap should allow a substantially stronger visibility target"
    );
    assert!(
        (low_cap_gradients[material_output] + 0.05).abs() <= 1.0e-6,
        "low cap should bound the target material output delta"
    );
}

#[test]
fn material_visibility_output_objective_suppresses_dormant_visible_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_material_visibility_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        0.0,
        1.0,
        0.0,
        0,
        0.54,
        0.05,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.0,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[material_output] > 0.0,
        "dormant material-visible row should train immediate material output downward"
    );
    assert!(
        output_gradients[material_output]
            <= material_suppression_max_update(
                0.05,
                ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER
            ) + 1.0e-6,
        "suppression objective should respect the material suppression cap"
    );
}

#[test]
fn material_surface_strata_updates_promote_undercovered_visible_bins() {
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
        [-1.0_f32, 0.0, 0.05, 0.0],
        [1.0_f32, 0.0, 0.05, 0.0],
        [1.0_f32, 0.0, 0.05, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_surface_strata_opacity_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );

    assert_eq!(
        updates[0], 0.0,
        "already material-visible covered strata should not be boosted"
    );
    assert!(
        updates[1] > 0.0,
        "live row on an undercovered surface stratum should receive material opacity pressure: {updates:?}"
    );
    assert!(
        updates[1] <= 0.05 + 1.0e-6,
        "strata material updates should respect max_update: {updates:?}"
    );
    assert_eq!(
        updates[2], 0.0,
        "dormant row must not receive material visibility pressure"
    );
}

#[test]
fn weighted_material_surface_strata_updates_promote_local_front_bins() {
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
    let positions = vec![[-1.0_f32, 0.0, 0.05, 0.0], [1.0_f32, 0.0, 0.05, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let active_only = material_surface_strata_opacity_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );
    let weighted = material_surface_strata_opacity_updates_weighted(
        &config,
        &target,
        &positions,
        &states,
        Some(&[1.0, 0.5]),
        1.0,
        512,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
    );

    assert_eq!(
        active_only[1], 0.0,
        "active-only strata coverage should not wake dormant undercovered bins"
    );
    assert!(
        weighted[1] > 0.0,
        "weighted local-front row should receive material pressure for an undercovered bin"
    );
    assert!(
        weighted[1] <= 0.05 + 1.0e-6,
        "weighted strata update should respect max_update"
    );
}

#[test]
fn material_surface_strata_adjoint_uses_training_gradient_sign() {
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
    let positions = vec![[-1.0_f32, 0.0, 0.05, 0.0], [1.0_f32, 0.0, 0.05, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_material_surface_strata_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        1.0,
        GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
        0.05,
        &mut adjoint,
    );

    assert_eq!(adjoint[material_channel], 0.0);
    assert!(
        adjoint[config.state_dims + material_channel] < 0.0,
        "positive material stratum update should become negative state adjoint"
    );
    assert_eq!(adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL], 0.0);
}
