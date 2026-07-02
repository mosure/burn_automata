use super::*;

#[test]
fn local_growth_student_opacity_controller_expands_sparse_growth_front() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let (initial_positions, initial_states) = seed_particles_scaled(
        1,
        128,
        config.state_dims,
        config.spatial_dims,
        RolloutConfig::default().seed,
        ParticleSeed::TorusGrowth3d,
        UV_TORUS_FIELD_SCALE,
    );
    let initial_step = model
        .step_cpu(
            &initial_positions,
            &initial_states,
            1,
            128,
            &grid,
            1.0,
            None,
        )
        .unwrap();
    let mut max_inactive_opacity_ds = f32::MIN;
    for row in 0..128 {
        if initial_states[row * config.state_dims + 3] <= -1.0 {
            max_inactive_opacity_ds =
                max_inactive_opacity_ds.max(initial_step.ds[row * config.state_dims + 3]);
        }
    }
    assert!(
        max_inactive_opacity_ds > 0.1,
        "inactive particles on the active front should receive positive local opacity updates, max={max_inactive_opacity_ds}"
    );
    let trace = run_rollout(
        &model,
        &grid,
        &RolloutConfig {
            particle_count: 128,
            steps: 64,
            update_prob: 1.0,
            seed_scale: UV_TORUS_FIELD_SCALE,
            ..RolloutConfig::default()
        },
        ParticleSeed::TorusGrowth3d,
    )
    .unwrap();

    let active_threshold = -1.0_f32;
    let initial_active = initial_states
        .chunks_exact(config.state_dims)
        .filter(|state| state[3] > active_threshold)
        .count();
    let final_active = trace
        .states
        .chunks_exact(config.state_dims)
        .filter(|state| state[3] > active_threshold)
        .count();
    let max_opacity = trace
        .states
        .chunks_exact(config.state_dims)
        .map(|state| state[3])
        .fold(f32::MIN, f32::max);
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let initial_material_mean = initial_states
        .chunks_exact(config.state_dims)
        .map(|state| state[material_channel])
        .sum::<f32>()
        / 128.0;
    let final_material_mean = trace
        .states
        .chunks_exact(config.state_dims)
        .map(|state| state[material_channel])
        .sum::<f32>()
        / trace.particle_count as f32;

    assert!(
        final_active > initial_active,
        "front controller should activate more particles, initial={initial_active} final={final_active}"
    );
    assert!(
        final_active < trace.particle_count,
        "front controller should not activate the whole cloud in one global sweep, final={final_active}"
    );
    assert!(
        max_opacity < UV_TORUS_FIELD_OPACITY_TARGET + 0.5,
        "front opacity should remain bounded, max opacity={max_opacity}"
    );
    assert!(
        final_material_mean > initial_material_mean + 0.25,
        "material opacity should rise with the local growth front, initial={initial_material_mean} final={final_material_mean}"
    );
}

#[test]
fn active_opacity_retime_leaves_dormant_particles_untouched() {
    let config = NpaConfig::growing_3dgs();
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let gain = 0.035;
    retime_growth_3d_active_opacity_model(&mut model, Some(32), gain).unwrap();

    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let opacity_out = config.spatial_dims + 3;
    let mut features = vec![0.0_f32; 3 * input_dims];
    features[3] = -3.0;
    features[input_dims + 3] = -0.5;
    features[2 * input_dims + 3] = 2.0;
    let update = model.forward_update_from_features(&features).unwrap();

    assert!(update[opacity_out].abs() < 1.0e-6);
    assert!((update[output_dims + opacity_out] - gain * 0.5).abs() < 1.0e-6);
    assert!((update[2 * output_dims + opacity_out] - gain).abs() < 1.0e-6);
}

#[test]
fn opacity_bias_retime_only_offsets_opacity_output_bias() {
    let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
    let before = model.weights.b2.clone();
    let opacity_out = model.config.spatial_dims + 3;
    add_growth_3d_opacity_update_bias(&mut model, 0.0125).unwrap();
    for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
        if idx == opacity_out {
            assert!((current - initial - 0.0125).abs() <= 1.0e-7);
        } else {
            assert_eq!(current, initial);
        }
    }

    let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
    assert!(add_growth_3d_opacity_update_bias(&mut position_field, 0.01).is_err());
}

#[test]
fn material_opacity_bias_retime_only_offsets_material_output_bias() {
    let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
    let before = model.weights.b2.clone();
    let material_channel = growth_3d_material_opacity_channel(model.config.state_dims).unwrap();
    let material_opacity_out = model.config.spatial_dims + material_channel;
    let liveness_opacity_out = model.config.spatial_dims + 3;
    add_growth_3d_material_opacity_update_bias(&mut model, 0.0125).unwrap();
    for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
        if idx == material_opacity_out {
            assert!((current - initial - 0.0125).abs() <= 1.0e-7);
        } else {
            assert_eq!(current, initial);
        }
    }
    assert_eq!(
        model.weights.b2[liveness_opacity_out],
        before[liveness_opacity_out]
    );

    let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
    assert!(add_growth_3d_material_opacity_update_bias(&mut position_field, 0.01).is_err());
}

#[test]
fn local_front_opacity_targets_activate_only_near_active_neighbors() {
    let config = NpaConfig::growing_3dgs();
    let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; 3 * config.state_dims];
    states[3] = 0.0;
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];

    let updates = local_front_opacity_targets(
        &config,
        &positions,
        &states,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
    );

    assert!(
        updates[1] > 0.0,
        "inactive particle near an active neighbor should receive positive opacity update"
    );
    assert!(
        updates[2].abs() < 1.0e-6,
        "far inactive particle should stay dormant until the front reaches it"
    );
}

#[test]
fn front_motion_gate_suppresses_far_dormant_mesh_targets() {
    let config = NpaConfig::growing_3dgs();
    let mut states = vec![0.0; 3 * config.state_dims];
    states[3] = 0.0;
    states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let positions = vec![
        [0.0_f32, 0.0, 0.0, 0.0],
        [0.08_f32, 0.0, 0.0, 0.0],
        [0.8_f32, 0.0, 0.0, 0.0],
    ];
    let output_dims = config.update_dims();
    let target = uv_torus_mesh_target(0.72);

    let ungated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        f32::INFINITY,
        0.0,
        1.0,
        0.0,
        0.0,
        0.20,
        0.0,
        false,
    );
    let gated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        f32::INFINITY,
        0.0,
        1.0,
        0.0,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        true,
    );
    let opacity_gated = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
        UV_TORUS_FIELD_OPACITY_GAIN,
        LOCAL_GROWTH_FRONT_OPACITY_GAIN,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        true,
    );

    let far_base = 2 * output_dims;
    let far_ungated_motion =
        (ungated[far_base].powi(2) + ungated[far_base + 1].powi(2) + ungated[far_base + 2].powi(2))
            .sqrt();
    let far_gated_motion =
        (gated[far_base].powi(2) + gated[far_base + 1].powi(2) + gated[far_base + 2].powi(2))
            .sqrt();
    let near_base = output_dims;
    let near_gated_motion =
        (gated[near_base].powi(2) + gated[near_base + 1].powi(2) + gated[near_base + 2].powi(2))
            .sqrt();
    let opacity_out = config.spatial_dims + 3;
    let far_gated_opacity = opacity_gated[far_base + opacity_out];
    let near_gated_opacity = opacity_gated[near_base + opacity_out];

    assert!(
        far_ungated_motion > 1.0e-4,
        "fixture should have a nonzero target motion without front gating"
    );
    assert!(
        far_gated_motion < 1.0e-6,
        "far dormant particle should not receive target motion before the active front reaches it"
    );
    assert!(
        near_gated_motion > 1.0e-4,
        "near-front inactive particle should still receive gated target motion"
    );
    assert!(
        far_gated_opacity.abs() < 1.0e-6,
        "far dormant particle should not receive direct opacity target before the active front reaches it"
    );
    assert!(
        near_gated_opacity > 0.0,
        "near-front inactive particle should receive front-gated opacity growth"
    );
}

#[test]
fn mesh_opacity_targets_surface_material_instead_of_whole_domain() {
    let config = NpaConfig::growing_3dgs();
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ],
        [0.0_f32, 0.0, 0.0, 0.0],
    ];
    let mut states = vec![0.0; 2 * config.state_dims];
    let material_opacity_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    states[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[material_opacity_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    states[config.state_dims + 3] = 0.0;
    states[config.state_dims + material_opacity_channel] = 0.0;

    let updates = mesh_field_target_update_for_rows(
        &config,
        &target,
        &positions,
        &states,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
        UV_TORUS_FIELD_OPACITY_GAIN,
        0.0,
        0.20,
        LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        false,
    );
    let opacity_out = config.spatial_dims + material_opacity_channel;

    assert!(
        updates[opacity_out] > 0.0,
        "near-surface dormant material should receive positive render opacity pressure"
    );
    assert!(
        updates[config.update_dims() + opacity_out] < 0.0,
        "off-surface active material should be suppressed instead of making the whole substrate visible"
    );
}
