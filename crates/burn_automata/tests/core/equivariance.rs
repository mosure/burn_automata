use super::*;

#[test]
fn scale_equivariant_cpu_step_preserves_scaled_rollout() {
    let (mut config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    config.equivariance = EquivarianceMode::ParticleDensityAndScale;
    config.hidden_dims = 8;
    let model = NpaModel::seeded(config, 17);
    let particles = 32;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        23,
        ParticleSeed::UniformCircle,
        0.2,
    );

    let scale = 1.7;
    let mut scaled_grid = grid.clone();
    scaled_grid.eps *= scale;
    let scaled_positions = positions
        .iter()
        .map(|position| {
            [
                position[0] * scale,
                position[1] * scale,
                position[2] * scale,
                position[3],
            ]
        })
        .collect::<Vec<_>>();

    let base = model
        .step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)
        .unwrap();
    let scaled = model
        .step_cpu(
            &scaled_positions,
            &states,
            1,
            particles,
            &scaled_grid,
            1.0,
            None,
        )
        .unwrap();

    for (base, scaled) in base.next_positions.iter().zip(scaled.next_positions.iter()) {
        for axis in 0..model.config.spatial_dims {
            let normalized = scaled[axis] / scale;
            assert!(
                (base[axis] - normalized).abs() < 2.0e-4,
                "position axis {axis}: base {}, scaled/scale {}",
                base[axis],
                normalized
            );
        }
    }
    for (base, scaled) in base.next_states.iter().zip(scaled.next_states.iter()) {
        assert!(
            (base - scaled).abs() < 2.0e-4,
            "state base {base}, scaled {scaled}"
        );
    }
}
#[test]
fn scale_equivariant_seed_hashgrid_scales_particle_eps_only() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let scaled = config.hashgrid_for_seed_scale(&grid, 0.25, 0.72);
    assert!((scaled.eps - grid.eps * 0.25 / 0.72).abs() < 1.0e-7);
    assert_eq!(scaled.grid_size, grid.grid_size);
    assert_eq!(scaled.mode, grid.mode);

    let (texture_config, texture_grid) = NpaConfig::for_preset(AutomataPreset::Texture2d);
    let texture_scaled = texture_config.hashgrid_for_seed_scale(&texture_grid, 0.25, 1.0);
    assert_eq!(texture_scaled.eps, texture_grid.eps);

    let mut non_equivariant = config;
    non_equivariant.equivariance = EquivarianceMode::None;
    let unchanged = non_equivariant.hashgrid_for_seed_scale(&grid, 0.25, 0.72);
    assert_eq!(unchanged.eps, grid.eps);
}
