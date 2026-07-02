use super::*;

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
fn material_visibility_output_objective_keeps_dormant_front_material_below_visible_threshold() {
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
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.08_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.2;
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
        1.0,
        0.0,
        64,
        1.0,
        0.25,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[material_output] > 0.0,
        "dormant render-visible local-front material should be suppressed until liveness catches up"
    );
    assert!(
        output_gradients[liveness_output] < 0.0,
        "suppressed dormant material row should still train liveness upward"
    );
    assert!(
        output_gradients[output_dims + material_output] < 0.0,
        "active neighboring surface row may train visible material immediately"
    );
}

#[test]
fn material_visibility_output_objective_suppresses_predicted_dormant_material_visibility() {
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
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.08_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[output_dims + material_output] = 1.25;
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
        64,
        1.0,
        0.25,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        1.0,
        0.1,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] > 1.0,
        "pending dormant row whose predicted material crosses visible threshold should be trained back toward inactive baseline, not just the precursor ceiling"
    );
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "predicted material visibility should still train local-front liveness upward"
    );
}

#[test]
fn material_visibility_output_objective_suppresses_predicted_dormant_material_above_precursor() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.08_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[output_dims + material_output] = 0.9;
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
        64,
        1.0,
        0.25,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        1.0,
        0.5,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] > 0.0,
        "pending dormant material predicted above the precursor ceiling should be trained back before it reaches visible opacity"
    );
}

#[test]
fn material_visibility_output_objective_activates_predicted_visible_material_without_global_deficit()
 {
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
    let positions = vec![[0.0_f32, 0.0, 0.01, 0.0], [0.08_f32, 0.0, 0.01, 0.0]];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    for state in states.chunks_exact_mut(config.state_dims) {
        state[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    }
    let mut raw_updates = vec![0.0_f32; positions.len() * output_dims];
    raw_updates[output_dims + material_output] = 1.25;
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
        64,
        1.0,
        0.25,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        0.20,
        None,
        0.0,
        0.5,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[output_dims + material_output] > 1.0,
        "predicted visible material should still be suppressed when global activation count is already on schedule"
    );
    assert!(
        output_gradients[output_dims + liveness_output] < 0.0,
        "predicted visible material should train local-front liveness even without a global activation deficit"
    );
}

#[test]
fn material_visibility_output_objective_does_not_materialize_offsurface_local_front_rows() {
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

    assert_eq!(
        output_gradients[output_dims + material_output],
        0.0,
        "off-surface local-front row should not become render-visible before reaching the surface band"
    );
    assert_eq!(
        output_gradients[output_dims + liveness_output],
        0.0,
        "off-surface local-front row should rely on geometry/liveness objectives rather than material preactivation"
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
