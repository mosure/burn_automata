use super::*;

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
