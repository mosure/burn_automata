use super::*;

#[test]
fn uv_torus_dense_3d_seed_uses_random_cloud_with_target_residuals() {
    let particles = 512;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        31,
        ParticleSeed::UvTorusDense3d,
        scale,
    );
    let dense_radius = uv_torus_dense_seed_radius(scale);
    let mut max_target_error = 0.0_f32;
    let mut max_residual_error = 0.0_f32;
    let mut max_color_error = 0.0_f32;
    let mut max_position_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let position_radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        max_position_radius = max_position_radius.max(position_radius);

        let state_base = idx * state_dims;
        let target = uv_torus_sample(idx, particles, scale).position;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let target_error = ((reconstructed[0] - target[0]).powi(2)
            + (reconstructed[1] - target[1]).powi(2)
            + (reconstructed[2] - target[2]).powi(2))
        .sqrt();
        max_target_error = max_target_error.max(target_error);
        let residual_error = ((states[state_base] - (target[0] - position[0])).powi(2)
            + (states[state_base + 1] - (target[1] - position[1])).powi(2)
            + (states[state_base + 2] - (target[2] - position[2])).powi(2))
        .sqrt();
        max_residual_error = max_residual_error.max(residual_error);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(target, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        max_color_error = max_color_error.max(color_error);
    }

    assert!(max_position_radius <= dense_radius + 1.0e-6);
    assert!(max_target_error <= 2.0e-5);
    assert!(max_residual_error <= 2.0e-5);
    assert!(max_color_error <= 1.0e-6);
}
#[test]
fn teapot_morphogen_dense_3d_seed_uses_mesh_projected_seed_frame() {
    let particles = 256;
    let state_dims = 16;
    let scale = 0.72;
    let target = TriangleMeshTarget::utah_teapot(scale).unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        73,
        ParticleSeed::TeapotMorphogenDense3d,
        scale,
    );
    let mut max_projected_error = 0.0_f32;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;

    for (idx, position) in positions.iter().enumerate() {
        min_x = min_x.min(position[0]);
        max_x = max_x.max(position[0]);
        let state_base = idx * state_dims;
        let projection = target.project([position[0], position[1], position[2]]);
        let projected = projection.closest;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let projected_error = ((reconstructed[0] - projected[0]).powi(2)
            + (reconstructed[1] - projected[1]).powi(2)
            + (reconstructed[2] - projected[2]).powi(2))
        .sqrt();
        max_projected_error = max_projected_error.max(projected_error);

        let actual_normal = [
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        assert!((dot3(actual_normal, actual_normal).sqrt() - 1.0).abs() < 1.0e-5);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);

        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = projection.color;
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }

    assert!(max_projected_error <= 2.0e-5);
    assert!(
        max_x - min_x > 1.2 * scale,
        "teapot dense seed should cover body, spout, and handle envelope"
    );
}
#[test]
fn teapot_field_dense_3d_seed_is_neutral_not_projected_or_precolored() {
    let particles = 256;
    let state_dims = 16;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        73,
        ParticleSeed::TeapotFieldDense3d,
        scale,
    );
    let mut max_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        max_radius = max_radius.max(radius);

        let state_base = idx * state_dims;
        assert_eq!(states[state_base], 0.0);
        assert_eq!(states[state_base + 1], 0.0);
        assert_eq!(states[state_base + 2], 0.0);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET], 0.0);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1], 0.0);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2], 0.0);
        assert_eq!(
            states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
            0.0
        );
        assert_eq!(states[state_base + state_dims - 3], 0.0);
        assert_eq!(states[state_base + state_dims - 2], 0.0);
        assert_eq!(states[state_base + state_dims - 1], 0.0);
    }

    assert!(max_radius <= scale + 1.0e-6);
}
#[test]
fn growth_3d_seeds_are_compact_neutral_and_not_target_assigned() {
    let particles = 512;
    let state_dims = 16;
    let scale = 0.72;
    for seed_mode in [
        ParticleSeed::Growth3d,
        ParticleSeed::TorusGrowth3d,
        ParticleSeed::TeapotGrowth3d,
    ] {
        let (positions, states) =
            seed_particles_scaled(1, particles, state_dims, 3, 73, seed_mode, scale);
        let mut max_radius = 0.0_f32;
        let mut min_radius = f32::MAX;
        let mut active_count = 0usize;
        let mut inactive_count = 0usize;

        for (idx, position) in positions.iter().enumerate() {
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            max_radius = max_radius.max(radius);
            min_radius = min_radius.min(radius);

            let state_base = idx * state_dims;
            let opacity = states[state_base + 3];
            let material_opacity_channel = growth_3d_material_opacity_channel(state_dims);
            let domain_radius = growth_3d_domain_radius(scale).max(1.0e-4);
            for channel in 0..state_dims {
                if channel < 3 {
                    assert!(
                        (states[state_base + channel] - position[channel] / domain_radius).abs()
                            < 1.0e-6,
                        "{seed_mode:?} channel {channel} should store normalized seed-frame coordinate, not a target assignment"
                    );
                    continue;
                }
                if channel == 3 || Some(channel) == material_opacity_channel {
                    continue;
                }
                assert_eq!(
                    states[state_base + channel],
                    0.0,
                    "{seed_mode:?} should not seed particle identity or target channel {channel}"
                );
            }
            if radius <= growth_3d_active_core_radius(scale) {
                active_count += 1;
                assert!(
                    (opacity - GROWTH_3D_ACTIVE_OPACITY_LOGIT).abs() < 1.0e-6,
                    "{seed_mode:?} active opacity {opacity}"
                );
            } else {
                inactive_count += 1;
                assert!(
                    (opacity - GROWTH_3D_INACTIVE_OPACITY_LOGIT).abs() < 1.0e-6,
                    "{seed_mode:?} inactive opacity {opacity}"
                );
            }
        }

        assert!(max_radius <= growth_3d_seed_radius(scale) + 1.0e-6);
        assert!(
            min_radius < growth_3d_seed_radius(scale) * 0.35,
            "growth seed should include the compact core, got min radius {min_radius}"
        );
        assert!(
            max_radius > growth_3d_seed_radius(scale) * 0.85,
            "growth seed should fill most of the compact ball, got max radius {max_radius}"
        );
        assert!(
            active_count > 0,
            "{seed_mode:?} has no active core particles"
        );
        assert!(
            inactive_count > active_count * 4,
            "{seed_mode:?} should start from a sparse active core, active={active_count} inactive={inactive_count}"
        );
    }
}

#[test]
fn generic_3d_growth_seeds_match_legacy_alias_topology() {
    let particles = 256;
    let state_dims = 16;
    let scale = 0.72;
    for (generic, legacy) in [
        (ParticleSeed::Growth3d, ParticleSeed::TorusGrowth3d),
        (
            ParticleSeed::SubstrateGrowth3d,
            ParticleSeed::TorusSubstrateGrowth3d,
        ),
        (
            ParticleSeed::LocalGrowth3d,
            ParticleSeed::TorusLocalGrowth3d,
        ),
        (
            ParticleSeed::LocalSubstrateGrowth3d,
            ParticleSeed::TorusLocalSubstrateGrowth3d,
        ),
    ] {
        let (generic_positions, generic_states) =
            seed_particles_scaled(1, particles, state_dims, 3, 0x3d5eed, generic, scale);
        let (legacy_positions, legacy_states) =
            seed_particles_scaled(1, particles, state_dims, 3, 0x3d5eed, legacy, scale);

        assert_eq!(
            generic_positions, legacy_positions,
            "{generic:?} should keep the same topology as its legacy alias"
        );
        assert_eq!(
            generic_states, legacy_states,
            "{generic:?} should only rename the seed family, not change initialized state"
        );
    }
}
#[test]
fn morphogen_seed_envelope_sampler_mixes_generic_callbacks() {
    let envelope = MorphogenSeedEnvelope {
        core_radius: 0.25,
        bounds_min: [3.0, -1.0, -1.0],
        bounds_max: [4.0, 1.0, 1.0],
        near_surface_jitter: 0.05,
    };
    let mut rng = StdRng::seed_from_u64(0x51eed);
    let mut saw_core = false;
    let mut saw_volume = false;
    let mut saw_surface = false;
    let mut saw_bounds = false;

    for _ in 0..512 {
        let position = morphogen_seed_envelope_position(
            &mut rng,
            envelope,
            |_| [0.0, 0.0, 0.0],
            |_| [1.0, 0.0, 0.0],
            |_| [2.0, 0.0, 0.0],
            |_| [1.0, 0.0, 0.0],
        );
        saw_core |= position[0].abs() <= 1.0e-6;
        saw_volume |= (position[0] - 1.0).abs() <= 1.0e-6;
        saw_surface |= (1.95..=2.05).contains(&position[0]);
        saw_bounds |= (3.0..=4.0).contains(&position[0]);
    }

    assert!(saw_core, "generic envelope never sampled core callback");
    assert!(saw_volume, "generic envelope never sampled volume callback");
    assert!(
        saw_surface,
        "generic envelope never sampled near-surface callback"
    );
    assert!(saw_bounds, "generic envelope never sampled bounds callback");
}
#[test]
fn normalized_seed_scale_preserves_hashgrid_occupancy_for_scaled_3d_seeds() {
    let (config, base_grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let particles = 8192;
    let state_dims = config.state_dims;
    let reference_scale = 0.72;
    let baseline = seed_occupancy_stats(
        &base_grid,
        particles,
        state_dims,
        reference_scale,
        reference_scale,
        false,
    );

    for scale in [0.04_f32, 0.16, 1.2] {
        let grid = config.hashgrid_for_seed_scale(&base_grid, scale, reference_scale);
        let stats =
            seed_occupancy_stats(&grid, particles, state_dims, scale, reference_scale, false);
        assert_eq!(
            stats, baseline,
            "scale-normalized hashgrid should preserve occupancy at scale {scale}"
        );
    }

    let unnormalized_small = seed_occupancy_stats(
        &base_grid,
        particles,
        state_dims,
        0.04,
        reference_scale,
        false,
    );
    assert!(
        unnormalized_small.1 > baseline.1 * 40,
        "fixed eps should expose the dense-cell failure mode: baseline={baseline:?} fixed={unnormalized_small:?}"
    );
    assert!(
        unnormalized_small.0 < baseline.0 / 100,
        "fixed eps should collapse the small seed into far fewer cells: baseline={baseline:?} fixed={unnormalized_small:?}"
    );
}
#[test]
fn torus_field_dense_3d_seed_is_neutral_not_index_assigned() {
    let particles = 128;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        43,
        ParticleSeed::TorusFieldDense3d,
        scale,
    );
    let dense_radius = uv_torus_dense_seed_radius(scale);

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        assert!(radius <= dense_radius + 1.0e-6);

        let state_base = idx * state_dims;
        assert_eq!(states[state_base], 0.0);
        assert_eq!(states[state_base + 1], 0.0);
        assert_eq!(states[state_base + 2], 0.0);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        assert_eq!(states[state_base + state_dims - 3], 0.0);
        assert_eq!(states[state_base + state_dims - 2], 0.0);
        assert_eq!(states[state_base + state_dims - 1], 0.0);
    }
}
#[test]
fn torus_morphogen_dense_3d_seed_uses_projected_seed_frame_not_index() {
    let particles = 128;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        43,
        ParticleSeed::TorusMorphogenDense3d,
        scale,
    );
    let mut max_projected_error = 0.0_f32;
    let mut max_index_error = 0.0_f32;
    let mut min_radial = f32::MAX;
    let mut max_radial = f32::MIN;

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        assert!(radius <= uv_torus_outer_radius(scale) * 1.9);
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        min_radial = min_radial.min(radial);
        max_radial = max_radial.max(radial);

        let state_base = idx * state_dims;
        let projected = uv_torus_project_position([position[0], position[1], position[2]], scale);
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let projected_error = ((reconstructed[0] - projected[0]).powi(2)
            + (reconstructed[1] - projected[1]).powi(2)
            + (reconstructed[2] - projected[2]).powi(2))
        .sqrt();
        max_projected_error = max_projected_error.max(projected_error);

        let indexed = uv_torus_sample(idx, particles, scale).position;
        let index_error = ((reconstructed[0] - indexed[0]).powi(2)
            + (reconstructed[1] - indexed[1]).powi(2)
            + (reconstructed[2] - indexed[2]).powi(2))
        .sqrt();
        max_index_error = max_index_error.max(index_error);

        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(projected, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }

    assert!(max_projected_error <= 2.0e-5);
    assert!(
        min_radial < scale * 0.35 && max_radial > uv_torus_outer_radius(scale) * 0.75,
        "morphogen seed should cover both core and torus target envelope"
    );
    assert!(
        max_index_error >= 0.1,
        "morphogen seed unexpectedly matched indexed target error {max_index_error}"
    );
}
#[test]
fn torus_morphogen_seed_initializes_orientation_channels_when_available() {
    let particles = 64;
    let state_dims = 16;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        47,
        ParticleSeed::TorusMorphogenDense3d,
        scale,
    );
    assert!(uv_torus_orientation_state_available(state_dims));

    for (idx, position) in positions.iter().enumerate() {
        let state_base = idx * state_dims;
        let source = [position[0], position[1], position[2]];
        let actual_normal = [
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let expected_normal = uv_torus_outward_normal(source, scale);
        let normal_len = dot3(actual_normal, actual_normal).sqrt();
        assert!((normal_len - 1.0).abs() < 1.0e-5);
        assert!(dot3(actual_normal, expected_normal) > 0.999);
        assert!(
            (states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET]
                - uv_torus_signed_distance(source, scale))
            .abs()
                < 1.0e-6
        );

        let projected = uv_torus_project_position(source, scale);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(projected, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }
}
#[test]
fn uv_torus_zero_update_artifact_roundtrips_and_preserves_seed() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let manifest = BpkModelManifest::from_model(
        &model,
        grid.clone(),
        Some("unit-test:uv-torus-3d".to_string()),
    );
    let path = temp_path("uv_torus_3d.bpk");

    burn_automata::import::save_manifest(&path, &manifest).unwrap();
    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    fs::remove_file(&path).ok();
    let loaded_model = loaded.into_model();

    let cfg = RolloutConfig {
        particle_count: 128,
        steps: 4,
        update_prob: 1.0,
        seed_scale: 0.72,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&loaded_model, &grid, &cfg, ParticleSeed::UvTorus3d).unwrap();
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        cfg.particle_count,
        loaded_model.config.state_dims,
        loaded_model.config.spatial_dims,
        cfg.seed,
        ParticleSeed::UvTorus3d,
        cfg.seed_scale,
    );

    assert!(trace.mean_dx.iter().all(|value| value.abs() < 1.0e-8));
    assert_eq!(trace.positions, seed_positions);
    assert_eq!(trace.states, seed_states);
}
#[test]
fn uv_torus_opacity_growth_model_increases_visibility_state() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let mut weights = NpaWeights::zeros(&config);
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
    let model = NpaModel {
        config: config.clone(),
        weights,
    };
    let cfg = RolloutConfig {
        particle_count: 128,
        steps: 10,
        update_prob: 1.0,
        seed_scale: 0.72,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&model, &grid, &cfg, ParticleSeed::UvTorus3d).unwrap();
    let expected =
        UV_TORUS_INITIAL_OPACITY_LOGIT + UV_TORUS_OPACITY_GROWTH_DELTA * cfg.steps as f32 * cfg.dt;

    for state in trace.states.chunks_exact(config.state_dims) {
        assert!(
            (state[3] - expected).abs() < 1.0e-5,
            "opacity state {}, expected {expected}",
            state[3]
        );
    }
    assert!(trace.mean_dx.iter().all(|value| value.abs() < 1.0e-8));
}
