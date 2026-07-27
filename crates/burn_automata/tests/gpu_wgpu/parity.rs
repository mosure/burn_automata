use super::common::*;

#[test]
fn wgpu_step_matches_cpu_oracle_for_growing_2d() -> Result<(), Box<dyn std::error::Error>> {
    assert_preset_parity(AutomataPreset::Growing2d, 64, 7)
}

#[test]
fn wgpu_step_matches_cpu_oracle_for_texture_2d() -> Result<(), Box<dyn std::error::Error>> {
    assert_preset_parity(AutomataPreset::Texture2d, 64, 9)
}

#[test]
fn wgpu_step_matches_cpu_oracle_for_growing_3d_gs() -> Result<(), Box<dyn std::error::Error>> {
    assert_preset_parity(AutomataPreset::Growing3dGs, 48, 11)
}

#[test]
fn wgpu_step_matches_cpu_oracle_with_position_features() -> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let particles = 64;
    let seed_scale = 0.72;
    let config = NpaConfig::torus_field_3dgs();
    let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config, 45);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        17,
        ParticleSeed::TorusFieldDense3d,
        seed_scale,
    );

    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let gpu = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU position-feature parity test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    assert!(
        max_pos <= 5.0e-3,
        "position-feature max position error {max_pos}"
    );
    assert!(
        max_state <= 5.0e-3,
        "position-feature max state error {max_state}"
    );
    Ok(())
}

#[test]
fn wgpu_mesh_surface_anchor_matches_cpu_and_blocks_position_updates()
-> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let particles = 2;
    let mut config = burn_automata::mesh3d_model_config(64);
    config.alpha = 1.0;
    let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
    let mut weights = NpaWeights::zeros(&config);
    weights.b2[0] = 1.0;
    let model = NpaModel { config, weights };
    let positions = vec![[0.0, 0.0, 0.0, 1.0], [0.3, 0.0, 0.0, 0.0]];
    let mut states = vec![0.0; particles * model.config.state_dims];
    for state in states.chunks_exact_mut(model.config.state_dims) {
        state[burn_automata::rollout::UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = 0.04;
    }

    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let gpu = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU mesh-anchor parity test: {message}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    assert_eq!(cpu.next_positions[0], positions[0]);
    assert_eq!(gpu.next_positions[0], positions[0]);
    assert!(cpu.next_positions[1][0] > positions[1][0]);
    assert!(gpu.next_positions[1][0] > positions[1][0]);
    assert!(
        max_position_abs_error(&cpu.next_positions, &gpu.next_positions) <= 2.5e-3,
        "mesh-anchor CPU/WGPU position mismatch"
    );
    assert!(
        max_abs_error(&cpu.next_states, &gpu.next_states) <= 2.5e-3,
        "mesh-anchor CPU/WGPU state mismatch"
    );
    Ok(())
}

#[test]
fn wgpu_step_matches_cpu_oracle_for_teapot_morphogen_seed() -> Result<(), Box<dyn std::error::Error>>
{
    let _wgpu_guard = wgpu_test_guard();
    let particles = 64;
    let seed_scale = 0.72;
    let config = NpaConfig::growing_3dgs();
    let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config, 57);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        19,
        ParticleSeed::TeapotMorphogenDense3d,
        seed_scale,
    );

    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let gpu = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU teapot-seed parity test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    assert!(
        max_pos <= 5.0e-3,
        "teapot-seed max position error {max_pos}"
    );
    assert!(
        max_state <= 5.0e-3,
        "teapot-seed max state error {max_state}"
    );
    Ok(())
}

#[test]
fn wgpu_step_matches_cpu_oracle_for_point_mnist_shape() -> Result<(), Box<dyn std::error::Error>> {
    assert_preset_parity(AutomataPreset::PointMnist, 64, 13)
}

#[test]
fn wgpu_scale_equivariant_auto_mode_preserves_scaled_rollout()
-> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Growing2d;
    let particles = 64;
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 19);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        43,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let scale = 1.5;
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
    let base = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU scale equivariance test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };
    let scaled = burn_automata::gpu::step_wgpu_blocking(
        &model,
        &scaled_positions,
        &states,
        1,
        particles,
        &scaled_grid,
        1.0,
    )?;

    let max_pos = base
        .next_positions
        .iter()
        .zip(scaled.next_positions.iter())
        .flat_map(|(base, scaled)| {
            (0..model.config.spatial_dims)
                .map(move |axis| (base[axis] - scaled[axis] / scale).abs())
        })
        .fold(0.0, f32::max);
    let max_state = max_abs_error(&base.next_states, &scaled.next_states);
    eprintln!("WGPU scale equivariance: max_pos={max_pos:.8} max_state={max_state:.8}");
    assert!(
        max_pos <= 5.0e-3,
        "max scaled position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 5.0e-3,
        "max scaled state abs error {max_state} exceeded tolerance"
    );
    Ok(())
}

#[test]
fn wgpu_normalized_seed_scale_preserves_3d_torus_morphogen_rollout()
-> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let particles = 128;
    let reference_scale = 0.72;
    let scaled_seed = 0.25;
    let scale = scaled_seed / reference_scale;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = uv_torus_growth_model(config);
    let (base_positions, base_states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        37,
        ParticleSeed::TorusMorphogenDense3d,
        reference_scale,
    );
    let (scaled_positions, scaled_states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        37,
        ParticleSeed::TorusMorphogenDense3d,
        scaled_seed,
    );
    let scaled_grid = model
        .config
        .hashgrid_for_seed_scale(&grid, scaled_seed, reference_scale);

    let base = match burn_automata::gpu::step_wgpu_blocking(
        &model,
        &base_positions,
        &base_states,
        1,
        particles,
        &grid,
        1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU normalized seed-scale test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };
    let scaled = burn_automata::gpu::step_wgpu_blocking(
        &model,
        &scaled_positions,
        &scaled_states,
        1,
        particles,
        &scaled_grid,
        1.0,
    )?;

    let max_pos = base
        .next_positions
        .iter()
        .zip(scaled.next_positions.iter())
        .flat_map(|(base, scaled)| {
            (0..model.config.spatial_dims)
                .map(move |axis| (base[axis] - scaled[axis] / scale).abs())
        })
        .fold(0.0, f32::max);
    let mut max_state = 0.0_f32;
    for particle in 0..particles {
        let state_base = particle * model.config.state_dims;
        for channel in 0..model.config.state_dims {
            let base_value = base.next_states[state_base + channel];
            let scaled_value = scaled.next_states[state_base + channel];
            let normalized_scaled = match channel {
                0..=2 | 7 => scaled_value / scale,
                _ => scaled_value,
            };
            max_state = max_state.max((base_value - normalized_scaled).abs());
        }
    }
    eprintln!(
        "WGPU normalized 3D seed scale: eps={} max_pos={max_pos:.8} max_state={max_state:.8}",
        scaled_grid.eps
    );
    assert!((scaled_grid.eps - grid.eps * scale).abs() < 1.0e-7);
    assert!(
        max_pos <= 1.5e-2,
        "max normalized 3d position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 5.0e-3,
        "max normalized 3d state abs error {max_state} exceeded tolerance"
    );
    Ok(())
}
