use super::common::*;

#[test]
fn wgpu_persistent_state_matches_cpu_rollout_for_3d() -> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Growing3dGs;
    let particles = 48;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (mut cpu_positions, mut cpu_states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        31,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state(
        &model,
        &cpu_positions,
        &cpu_states,
        1,
        particles,
        &grid,
        1.0,
    )?;

    for _ in 0..3 {
        let cpu = model.step_cpu(&cpu_positions, &cpu_states, 1, particles, &grid, 1.0, None)?;
        cpu_positions = cpu.next_positions;
        cpu_states = cpu.next_states;
        executor.step_state(&mut gpu_state)?;
    }
    let gpu = executor.read_state(&gpu_state)?;

    let max_pos = max_position_abs_error(&cpu_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu_states, &gpu.next_states);
    eprintln!("Growing3dGs persistent rollout: max_pos={max_pos:.8} max_state={max_state:.8}");
    assert!(
        max_pos <= 5.0e-3,
        "max persistent 3d position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 5.0e-3,
        "max persistent 3d state abs error {max_state} exceeded tolerance"
    );
    Ok(())
}

#[test]
fn wgpu_persistent_state_accepts_model_weight_updates() -> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Growing3dGs;
    let particles = 48;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config.clone(), 42);
    let updated_model = NpaModel::seeded(config, 43);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        31,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state =
        executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;

    executor.update_state_model(&mut gpu_state, &updated_model, &grid, 1.0, 1.0, 31)?;
    executor.step_state(&mut gpu_state)?;
    executor.wait_idle()?;

    Ok(())
}
