use super::*;

#[test]
fn temporal_liveness_candidate_floor_uses_expanded_local_shell() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let positions = (0..10)
        .map(|row| [row as f32 * 0.08, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let base_weights = vec![0.0_f32; positions.len()];

    let fixed = mesh_motion_candidate_weights_with_local_front_floor(
        &config,
        &positions,
        &states,
        0.05,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &base_weights,
    );
    let temporal = temporal_liveness_candidate_weights_with_local_front_floor(
        &config,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.05,
        DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR,
        &fixed,
    );

    assert_eq!(
        fixed[2], 0.0,
        "the generic local-front floor should stay at its one-row default for this tiny cloud"
    );
    assert!(
        (1..=3).all(|row| temporal[row] >= DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR),
        "temporal activation should expose the nearest expanded shell when the rollout is under-active"
    );
    assert_eq!(
        temporal[4], 0.0,
        "the expanded temporal shell should still stay bounded"
    );
}

#[test]
fn motion_memory_output_objective_mirrors_mesh_motion_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_channels = growth_3d_velocity_channels(config.state_dims).unwrap();
    let velocity_outputs = velocity_channels
        .map(|channel| config.spatial_dims + channel)
        .collect::<Vec<_>>();
    let mut mesh_output_gradients = vec![0.0_f32; 2 * output_dims];
    mesh_output_gradients[1] = -0.25;
    mesh_output_gradients[output_dims + 2] = 0.5;
    mesh_output_gradients[output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] = 9.0;
    let mut output_gradients = vec![0.0_f32; mesh_output_gradients.len()];

    add_motion_memory_output_objective(
        &config,
        &mesh_output_gradients,
        DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[velocity_outputs[0]], 0.0);
    assert_eq!(
        output_gradients[velocity_outputs[1]],
        -0.25 * DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[2]],
        0.5 * DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        0.0,
        "non-motion mesh gradients should not leak into velocity memory"
    );
}
#[test]
fn material_coverage_motion_memory_mirrors_local_front_motion_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_output_channels(&config);
    let mut material_motion_gradients = vec![0.0_f32; 3 * output_dims];
    material_motion_gradients[output_dims] = -0.4;
    material_motion_gradients[output_dims + 1] = 0.2;
    material_motion_gradients[2 * output_dims + 2] = 0.75;
    material_motion_gradients[2 * output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        8.0;
    let mut output_gradients = vec![0.0_f32; material_motion_gradients.len()];

    add_motion_memory_output_objective(
        &config,
        &material_motion_gradients,
        DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(output_gradients[velocity_outputs[0]], 0.0);
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        -0.4 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[1]],
        0.2 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[2]],
        0.75 * DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[0]],
        0.0,
        "non-spatial material/liveness pressure must not leak into velocity memory"
    );
}
#[test]
fn extent_motion_memory_mirrors_front_and_temporal_extent_pressure() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_output_channels(&config);
    let mut extent_motion_gradients = vec![0.0_f32; 2 * output_dims];
    let mut temporal_extent_motion_gradients = vec![0.0_f32; 2 * output_dims];
    extent_motion_gradients[0] = -0.2;
    extent_motion_gradients[output_dims + 1] = 0.3;
    temporal_extent_motion_gradients[2] = -0.4;
    temporal_extent_motion_gradients[output_dims] = 0.1;
    temporal_extent_motion_gradients
        [output_dims + config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL] = 9.0;
    let mut output_gradients = vec![0.0_f32; extent_motion_gradients.len()];

    add_extent_motion_memory_output_objective(
        &config,
        &extent_motion_gradients,
        &temporal_extent_motion_gradients,
        DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[velocity_outputs[0]],
        -0.2 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(output_gradients[velocity_outputs[1]], 0.0);
    assert_eq!(
        output_gradients[velocity_outputs[2]],
        -0.4 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[0]],
        0.1 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[1]],
        0.3 * DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION
    );
    assert_eq!(
        output_gradients[output_dims + velocity_outputs[2]],
        0.0,
        "non-spatial temporal/liveness pressure must not leak into velocity memory"
    );
}

#[test]
fn liveness_phase_memory_mirrors_liveness_pressure_only() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let phase_output = config.spatial_dims + growth_3d_phase_channel(config.state_dims).unwrap();
    let mut liveness_gradients = vec![0.0_f32; 3 * output_dims];
    liveness_gradients[liveness_output] = -0.8;
    liveness_gradients[output_dims + liveness_output] = 0.4;
    liveness_gradients[2 * output_dims] = -1.0;
    liveness_gradients[2 * output_dims + config.spatial_dims + 1] = 0.25;
    let mut output_gradients = vec![0.0_f32; liveness_gradients.len()];

    add_liveness_phase_memory_output_objective(
        &config,
        &liveness_gradients,
        DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        &mut output_gradients,
    );

    assert_eq!(
        output_gradients[phase_output],
        -0.8 * DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        "activation pressure should train phase memory upward under SGD"
    );
    assert_eq!(
        output_gradients[output_dims + phase_output],
        0.4 * DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION,
        "suppression pressure should also be mirrored so phase can decay"
    );
    assert_eq!(
        output_gradients[2 * output_dims + phase_output],
        0.0,
        "non-liveness output pressure must not leak into phase memory"
    );
}

#[test]
fn mesh_residual_velocity_objective_targets_active_and_local_front_velocity() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.2]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let output_dims = config.update_dims();
    let velocity_outputs = growth_3d_velocity_channels(config.state_dims)
        .unwrap()
        .map(|channel| config.spatial_dims + channel)
        .collect::<Vec<_>>();
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.80_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; positions.len() * config.state_dims];
    states[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_mesh_residual_velocity_output_objective(
        &config,
        &target,
        &positions,
        &states,
        &raw_updates,
        1.0,
        0.0,
        0.0,
        0.25,
        0.20,
        1.0,
        &mut output_gradients,
    );

    assert!(
        output_gradients[velocity_outputs[0]] < -1.0e-4,
        "active row should train positive x velocity toward mesh residual"
    );
    assert!(
        output_gradients[output_dims + velocity_outputs[0]] < -1.0e-4,
        "dormant local-front row should also train velocity"
    );
    assert_eq!(
        output_gradients[2 * output_dims + velocity_outputs[0]],
        0.0,
        "far dormant row should not receive residual velocity pressure"
    );
    assert_eq!(output_gradients[velocity_outputs[1]], 0.0);
    assert_eq!(output_gradients[velocity_outputs[2]], 0.0);
}

#[test]
fn growth_phase_output_objective_targets_active_and_local_front_rows() {
    let config = NpaConfig::growing_3dgs();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let phase_output = config.spatial_dims + phase_channel;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.9_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; positions.len() * config.state_dims];
    states[GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    states[2 * config.state_dims + phase_channel] = 0.25;
    let raw_updates = vec![0.0_f32; positions.len() * output_dims];
    let mut output_gradients = vec![0.0_f32; raw_updates.len()];

    add_growth_phase_output_objective(
        &config,
        &positions,
        &states,
        &raw_updates,
        0.5,
        1.0,
        0.20,
        &mut output_gradients,
    );

    assert!(
        output_gradients[phase_output] < 0.0,
        "active rows should be trained to advance the local phase state"
    );
    assert!(
        output_gradients[output_dims + phase_output] < 0.0,
        "dormant local-front rows should receive phase precursor pressure"
    );
    assert!(
        output_gradients[2 * output_dims + phase_output] > 0.0,
        "far dormant phase leakage should be suppressed"
    );
}
