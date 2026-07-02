use super::common::*;

#[test]
fn wgpu_neighbor_modes_match_cpu_oracle_for_2d() -> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Texture2d;
    let particles = 96;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        41,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };

    for mode in [
        burn_automata::gpu::WgpuNeighborMode::LinkedList,
        burn_automata::gpu::WgpuNeighborMode::FixedCellBuckets {
            capacity: particles,
        },
        burn_automata::gpu::WgpuNeighborMode::TiledFixedCellBuckets {
            capacity: particles,
        },
        burn_automata::gpu::WgpuNeighborMode::SortedCells,
        burn_automata::gpu::WgpuNeighborMode::Auto,
    ] {
        let mut state = executor.create_state_with_neighbor_mode(
            &model, &positions, &states, 1, particles, &grid, 1.0, mode,
        )?;
        executor.step_state(&mut state)?;
        let report = executor.neighbor_report(&state);
        let gpu = executor.read_state(&state)?;
        let overflow = executor.read_grid_overflow(&state)?;
        let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
        let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
        let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
        eprintln!(
            "Texture2d {mode:?} resolved={:?} cap={} overflow={overflow} max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8}",
            report.mode, report.bucket_capacity
        );
        assert_eq!(overflow, 0, "{mode:?} bucket overflowed");
        assert!(
            max_pos <= 2.5e-3,
            "{mode:?} max position abs error {max_pos} exceeded tolerance"
        );
        assert!(
            max_state <= 2.5e-3,
            "{mode:?} max state abs error {max_state} exceeded tolerance"
        );
        assert!(
            max_density <= 2.5e-3,
            "{mode:?} max density abs error {max_density} exceeded tolerance"
        );
    }
    Ok(())
}

#[test]
fn wgpu_bvh_mode_matches_cpu_oracle_for_clamped_2d_and_3d() -> Result<(), Box<dyn std::error::Error>>
{
    let _wgpu_guard = wgpu_test_guard();
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };

    for (preset, particles, seed) in [
        (AutomataPreset::Growing2d, 96usize, 43u64),
        (AutomataPreset::Growing3dGs, 96usize, 47u64),
    ] {
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
        let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
        for mode in [
            burn_automata::gpu::WgpuNeighborMode::Bvh { leaf_size: 16 },
            burn_automata::gpu::WgpuNeighborMode::GpuBvh { leaf_size: 16 },
            burn_automata::gpu::WgpuNeighborMode::GpuLbvh { leaf_size: 16 },
            burn_automata::gpu::WgpuNeighborMode::GpuMortonLbvh { leaf_size: 16 },
        ] {
            let mut state = executor.create_state_with_neighbor_mode(
                &model, &positions, &states, 1, particles, &grid, 1.0, mode,
            )?;
            executor.step_state(&mut state)?;
            let gpu = executor.read_state(&state)?;
            let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
            let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
            let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
            eprintln!(
                "{preset:?} {mode:?}: max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8}"
            );
            assert!(
                max_pos <= 2.5e-3,
                "{preset:?} {mode:?} max position abs error {max_pos} exceeded tolerance"
            );
            assert!(
                max_state <= 2.5e-3,
                "{preset:?} {mode:?} max state abs error {max_state} exceeded tolerance"
            );
            assert!(
                max_density <= 2.5e-3,
                "{preset:?} {mode:?} max density abs error {max_density} exceeded tolerance"
            );
        }
    }
    Ok(())
}

#[test]
fn wgpu_bvh_persistent_state_matches_cpu_rollout_for_3d() -> Result<(), Box<dyn std::error::Error>>
{
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
        53,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state_with_neighbor_mode(
        &model,
        &cpu_positions,
        &cpu_states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::GpuLbvh { leaf_size: 16 },
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
    eprintln!("Growing3dGs BVH persistent rollout: max_pos={max_pos:.8} max_state={max_state:.8}");
    assert!(
        max_pos <= 5.0e-3,
        "BVH persistent 3d max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 5.0e-3,
        "BVH persistent 3d max state abs error {max_state} exceeded tolerance"
    );
    Ok(())
}

#[test]
fn wgpu_particle_hashgrid_handles_shifted_3d_fixed_buckets()
-> Result<(), Box<dyn std::error::Error>> {
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Growing3dGs;
    let particles = 128;
    let seed_scale = 1.2;
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (mut positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        73,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    for position in &mut positions {
        position[0] += 4.0;
        position[1] -= 3.0;
        position[2] += 2.0;
    }

    let cpu = model.step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)?;
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut state = executor.create_state_with_neighbor_mode(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::FixedCellBuckets { capacity: 16 },
    )?;
    executor.step_state(&mut state)?;
    let overflow = executor.read_grid_overflow(&state)?;
    let gpu = executor.read_state(&state)?;
    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
    eprintln!(
        "shifted Growing3dGs fixed-bucket particle hash: overflow={overflow} max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8}"
    );
    assert_eq!(overflow, 0, "shifted particle hashgrid bucket overflowed");
    assert!(
        max_pos <= 2.5e-3,
        "shifted particle hashgrid max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "shifted particle hashgrid max state abs error {max_state} exceeded tolerance"
    );
    assert!(
        max_density <= 2.5e-3,
        "shifted particle hashgrid max density abs error {max_density} exceeded tolerance"
    );

    let mut tiled_state = executor.create_state_with_neighbor_mode(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::TiledFixedCellBuckets { capacity: 16 },
    )?;
    executor.step_state(&mut tiled_state)?;
    let overflow = executor.read_grid_overflow(&tiled_state)?;
    let gpu = executor.read_state(&tiled_state)?;
    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
    eprintln!(
        "shifted Growing3dGs tiled fixed-bucket particle hash: overflow={overflow} max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8}"
    );
    assert_eq!(
        overflow, 0,
        "shifted particle hashgrid tiled bucket overflowed"
    );
    assert!(
        max_pos <= 2.5e-3,
        "shifted particle hashgrid tiled max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "shifted particle hashgrid tiled max state abs error {max_state} exceeded tolerance"
    );
    assert!(
        max_density <= 2.5e-3,
        "shifted particle hashgrid tiled max density abs error {max_density} exceeded tolerance"
    );

    let mut sorted_state = executor.create_state_with_neighbor_mode(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::SortedCells,
    )?;
    executor.step_state(&mut sorted_state)?;
    let overflow = executor.read_grid_overflow(&sorted_state)?;
    let gpu = executor.read_state(&sorted_state)?;
    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
    eprintln!(
        "shifted Growing3dGs sorted particle hash: overflow={overflow} max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8}"
    );
    assert_eq!(overflow, 0, "sorted particle hashgrid should not overflow");
    assert!(
        max_pos <= 2.5e-3,
        "shifted particle hashgrid sorted max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "shifted particle hashgrid sorted max state abs error {max_state} exceeded tolerance"
    );
    assert!(
        max_density <= 2.5e-3,
        "shifted particle hashgrid sorted max density abs error {max_density} exceeded tolerance"
    );
    Ok(())
}

#[test]
fn wgpu_fixed_bucket_overflow_counter_reports_saturation() -> Result<(), Box<dyn std::error::Error>>
{
    let _wgpu_guard = wgpu_test_guard();
    let preset = AutomataPreset::Growing2d;
    let particles = 96;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        41,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut state = executor.create_state_with_neighbor_mode(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::FixedCellBuckets { capacity: 1 },
    )?;
    executor.step_state(&mut state)?;
    let overflow = executor.read_grid_overflow(&state)?;
    eprintln!("Growing2d fixed bucket capacity=1 overflow={overflow}");
    assert!(overflow > 0, "expected fixed bucket capacity=1 to overflow");
    Ok(())
}
