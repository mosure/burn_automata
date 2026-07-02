use super::*;

#[test]
fn surface_escape_state_adjoint_suppresses_only_escaped_active_particles() {
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
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let mut adjoint = vec![0.0; states.len()];

    add_surface_escape_state_adjoint(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        0.5,
        0.25,
        0.05,
        &mut adjoint,
    );

    assert_eq!(adjoint[GROWTH_3D_LIVENESS_CHANNEL], 0.0);
    assert_eq!(adjoint[opacity_channel], 0.0);
    assert!(
        adjoint[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] > 0.0,
        "escaped active particles should receive positive liveness suppression"
    );
    assert!(
        adjoint[config.state_dims + opacity_channel] > 0.0,
        "escaped active particles should receive positive material-opacity suppression"
    );
    assert_eq!(
        adjoint[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        0.0,
        "escaped dormant particles should not be suppressed again"
    );
    assert_eq!(adjoint[2 * config.state_dims + opacity_channel], 0.0);
    assert!(
        adjoint.iter().all(|value| value.abs() <= 0.05 + 1.0e-6),
        "surface escape state adjoints should respect max_adjoint"
    );
}
