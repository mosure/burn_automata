use super::*;

#[test]
fn material_surface_normal_updates_promote_undercovered_visible_normal_bins() {
    let config = NpaConfig::growing_3dgs();
    let target = two_opposed_normal_patch_target();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let updates = material_surface_normal_opacity_updates(
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
        "already material-visible normal bin should not be boosted"
    );
    assert!(
        updates[1] > 0.0,
        "live row in the missing normal bin should receive material opacity pressure: {updates:?}"
    );
    assert!(
        updates[1] <= 0.05 + 1.0e-6,
        "normal-bin material updates should respect max_update"
    );
    assert_eq!(
        updates[2], 0.0,
        "inactive row must not receive unweighted normal material pressure"
    );
}

#[test]
fn weighted_material_surface_normal_updates_promote_local_front_bins() {
    let config = NpaConfig::growing_3dgs();
    let target = two_opposed_normal_patch_target();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[-1.0_f32, 0.0, 0.0, 0.0], [1.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;

    let active_only = material_surface_normal_opacity_updates(
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
    let weighted = material_surface_normal_opacity_updates_weighted(
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
        "active-only normal coverage should not wake dormant rows"
    );
    assert!(
        weighted[1] > 0.0,
        "weighted local-front row should receive material pressure for an undercovered normal bin"
    );
    assert!(weighted[1] <= 0.05 + 1.0e-6);
}

#[test]
fn material_surface_normal_adjoint_uses_training_gradient_sign() {
    let config = NpaConfig::growing_3dgs();
    let target = two_opposed_normal_patch_target();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let positions = vec![[-1.0_f32, 0.0, 0.0, 0.0], [1.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
    states[config.state_dims + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let mut adjoint = vec![0.0; states.len()];

    add_material_surface_normal_state_adjoint(
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
        "positive material normal-bin update should become negative state adjoint"
    );
}

#[test]
fn material_visibility_output_objective_promotes_undercovered_normal_bins() {
    let config = NpaConfig::growing_3dgs();
    let target = two_opposed_normal_patch_target();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![[-1.0_f32, 0.0, 0.0, 0.0], [1.0_f32, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;
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
        512,
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

    assert_eq!(output_gradients[material_output], 0.0);
    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "visibility objective should train material output upward for the undercovered normal bin"
    );
}

fn two_opposed_normal_patch_target() -> TriangleMeshTarget {
    TriangleMeshTarget::new(
        vec![
            [-1.0, -0.2, -0.2],
            [-1.0, 0.2, -0.2],
            [-1.0, 0.0, 0.2],
            [1.0, -0.2, -0.2],
            [1.0, 0.0, 0.2],
            [1.0, 0.2, -0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap()
}
