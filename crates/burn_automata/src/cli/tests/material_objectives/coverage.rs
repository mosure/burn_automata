use super::*;

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
