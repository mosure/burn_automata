use super::common::*;

#[test]
fn wgpu_update_probability_zero_keeps_state_fixed() -> Result<(), Box<dyn std::error::Error>> {
    let preset = AutomataPreset::Growing2d;
    let particles = 64;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        53,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state_with_update_prob(
        &model, &positions, &states, 1, particles, &grid, 1.0, 0.0, 53,
    )?;
    executor.step_state(&mut gpu_state)?;
    let output = executor.read_state(&gpu_state)?;

    assert_eq!(output.next_positions, positions);
    assert_eq!(output.next_states, states);
    Ok(())
}

#[test]
fn wgpu_update_probability_half_matches_cpu_mask_oracle() -> Result<(), Box<dyn std::error::Error>>
{
    let preset = AutomataPreset::Growing2d;
    let particles = 64;
    let seed = 53;
    let update_prob = 0.5;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        seed,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let mask = gpu_update_mask(particles, update_prob, 0, seed);
    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, Some(&mask))?;
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state_with_update_prob(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        update_prob,
        seed,
    )?;
    executor.step_state(&mut gpu_state)?;
    let gpu = executor.read_state(&gpu_state)?;

    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    assert!(
        max_pos <= 2.5e-3,
        "update_prob=0.5 max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "update_prob=0.5 max state abs error {max_state} exceeded tolerance"
    );
    Ok(())
}
