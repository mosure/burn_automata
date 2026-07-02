use super::*;

#[test]
fn mesh_target_update_trains_oriented_state_from_neutral_seed() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; config.state_dims];
    let update = mesh_field_target_update_for_rows(
        &config, &target, &positions, &states, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let base = config.spatial_dims;
    let coordinate_norm =
        (update[base].powi(2) + update[base + 1].powi(2) + update[base + 2].powi(2)).sqrt();
    let normal_update = [
        update[base + UV_TORUS_NORMAL_STATE_OFFSET],
        update[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
        update[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
    ];
    let normal_norm =
        (normal_update[0].powi(2) + normal_update[1].powi(2) + normal_update[2].powi(2)).sqrt();
    assert!(coordinate_norm > 1.0e-4);
    assert!(normal_norm > 1.0e-4);
    assert!(update[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET].abs() > 1.0e-4);
}

#[test]
fn mesh_target_update_can_disable_projection_aux_state_targets() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
    let states = vec![0.0; config.state_dims];
    let update = mesh_field_target_update_for_rows(
        &config, &target, &positions, &states, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false,
    );
    let base = config.spatial_dims;

    for channel in [
        0,
        1,
        2,
        UV_TORUS_NORMAL_STATE_OFFSET,
        UV_TORUS_NORMAL_STATE_OFFSET + 1,
        UV_TORUS_NORMAL_STATE_OFFSET + 2,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
    ] {
        assert_eq!(update[base + channel], 0.0);
    }
}

#[test]
fn torus_morphogen_supervision_writes_oriented_mesh_channels() {
    let config = NpaConfig::growing_3dgs();
    let rows = 32;
    let batch = torus_morphogen_supervised_batch(&config, rows);
    let input_dims = config.perception_dims();
    let blur_offset = config.state_dims;

    for row in 0..rows {
        let base = row * input_dims;
        let normal = [
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET],
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let signed_distance = batch.features[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
        let normal_len =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((normal_len - 1.0).abs() < 1.0e-4);
        assert!(signed_distance.is_finite());
        assert!(signed_distance.abs() <= 1.5);
        for channel in 0..config.state_dims {
            assert_eq!(
                batch.features[base + channel],
                batch.features[base + blur_offset + channel]
            );
        }
    }
}
