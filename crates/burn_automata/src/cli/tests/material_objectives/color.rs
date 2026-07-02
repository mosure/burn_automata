use super::*;

#[test]
fn surface_color_output_objective_trains_active_rows_toward_mesh_color() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap()
    .with_vertex_colors(vec![[1.0, 0.0, 0.0]; 3])
    .unwrap();
    let output_dims = config.update_dims();
    let color_state = config.state_dims - 3;
    let color_output = config.spatial_dims + color_state;
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.0_f32, 0.0, 1.5, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_surface_color_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.2,
        0.0,
        None,
        &mut output_gradients,
    );

    assert!(
        output_gradients[color_output] < 0.0,
        "red channel should train upward toward the mesh color"
    );
    assert!(
        output_gradients[color_output + 1] > 0.0 && output_gradients[color_output + 2] > 0.0,
        "green/blue channels should train downward toward the mesh color"
    );
    assert!(
        output_gradients[color_output].abs() <= 0.2 + 1.0e-6,
        "color target update should respect the configured cap"
    );
    assert_eq!(
        output_gradients[output_dims + color_output],
        0.0,
        "off-surface active rows should not receive color pressure"
    );
}

#[test]
fn surface_color_output_objective_requires_local_front_for_dormant_rows() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap()
    .with_vertex_colors(vec![[0.0, 1.0, 0.0]; 3])
    .unwrap();
    let output_dims = config.update_dims();
    let color_state = config.state_dims - 3;
    let color_output = config.spatial_dims + color_state;
    let positions = vec![
        [0.0_f32, 0.0, 0.01, 0.0],
        [0.05_f32, 0.0, 0.01, 0.0],
        [0.8_f32, 0.0, 0.01, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let candidate_weights = vec![0.0_f32, 1.0, 1.0];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_surface_color_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        0.2,
        0.20,
        Some(&candidate_weights),
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + color_output + 1] < 0.0,
        "near dormant local-front candidate should train green color upward"
    );
    assert_eq!(
        output_gradients[2 * output_dims + color_output + 1],
        0.0,
        "far dormant rows should not be globally pre-colored"
    );
}

#[test]
fn surface_color_output_objective_uses_tail_plus_half_rgb_convention() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap()
    .with_vertex_colors(vec![[1.0, 0.0, 0.5]; 3])
    .unwrap();
    let output_dims = config.update_dims();
    let color_state = config.state_dims - 3;
    let color_output = config.spatial_dims + color_state;
    let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; config.state_dims];
    let raw_updates = vec![0.0_f32; output_dims];
    let mut output_gradients = vec![0.0_f32; output_dims];

    add_surface_color_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        1.0,
        1.0,
        0.0,
        None,
        &mut output_gradients,
    );

    assert!((output_gradients[color_output] + 0.5).abs() <= 1.0e-6);
    assert!((output_gradients[color_output + 1] - 0.5).abs() <= 1.0e-6);
    assert!(output_gradients[color_output + 2].abs() <= 1.0e-6);
}
