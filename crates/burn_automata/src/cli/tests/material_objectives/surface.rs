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
