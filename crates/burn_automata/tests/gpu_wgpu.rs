#![cfg(feature = "gpu_wgpu")]

use burn_automata::{
    AutomataError, AutomataPreset, NpaConfig, NpaModel, NpaWeights, ParticleSeed,
    rollout::{
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_MOTION_GAIN, UV_TORUS_OPACITY_GROWTH_DELTA, UV_TORUS_RESIDUAL_DECAY,
        seed_particles_scaled, uv_torus_position_color, uv_torus_sample,
    },
};

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
fn wgpu_step_matches_cpu_oracle_for_teapot_morphogen_seed() -> Result<(), Box<dyn std::error::Error>>
{
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
fn wgpu_persistent_state_matches_cpu_rollout_for_3d() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn wgpu_state_writes_gaussian_buffers_on_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let preset = AutomataPreset::Growing3dGs;
    let particles = 64;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        37,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state =
        executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;
    let gaussian_buffers = executor.create_gaussian_buffers(particles)?;
    executor.step_state_into_gaussians(&mut gpu_state, &gaussian_buffers.refs())?;

    let gpu = executor.read_state(&gpu_state)?;
    let gaussian = executor.read_gaussian_buffers(&gaussian_buffers)?;
    assert_eq!(gaussian.position_visibility.len(), particles * 4);
    assert_eq!(
        gaussian.spherical_harmonic.len(),
        particles * burn_automata::gpu::GAUSSIAN_SH_COEFF_COUNT
    );
    assert_eq!(gaussian.rotation.len(), particles * 4);
    assert_eq!(gaussian.scale_opacity.len(), particles * 4);

    for idx in 0..particles {
        let position_base = idx * 4;
        for axis in 0..3 {
            let expected = gpu.next_positions[idx][axis];
            let actual = gaussian.position_visibility[position_base + axis];
            assert!(
                (expected - actual).abs() <= 1.0e-6,
                "gaussian position axis {axis} at {idx}: expected {expected}, got {actual}"
            );
        }
        assert_eq!(gaussian.position_visibility[position_base + 3], 1.0);
        assert_eq!(gaussian.rotation[position_base], 1.0);
        assert_eq!(gaussian.rotation[position_base + 1], 0.0);
        assert_eq!(gaussian.rotation[position_base + 2], 0.0);
        assert_eq!(gaussian.rotation[position_base + 3], 0.0);
        for axis in 0..3 {
            assert!(gaussian.scale_opacity[position_base + axis].is_finite());
            assert!(gaussian.scale_opacity[position_base + axis] > 0.0);
        }
        let opacity = gaussian.scale_opacity[position_base + 3];
        assert!(
            (0.05..=0.95).contains(&opacity),
            "opacity {opacity} at {idx}"
        );
    }
    assert!(
        gaussian
            .spherical_harmonic
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn wgpu_batched_gaussian_steps_match_repeated_steps_with_stochastic_updates()
-> Result<(), Box<dyn std::error::Error>> {
    let preset = AutomataPreset::Growing2d;
    let particles = 96;
    let steps = 4;
    let seed = 71;
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
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };

    let mut repeated_state = executor.create_state_with_update_prob(
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
    let repeated_gaussians = executor.create_gaussian_buffers(particles)?;
    for step_idx in 0..steps {
        if step_idx + 1 == steps {
            executor.step_state_into_gaussians(&mut repeated_state, &repeated_gaussians.refs())?;
        } else {
            executor.step_state(&mut repeated_state)?;
        }
    }

    let mut batched_state = executor.create_state_with_update_prob(
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
    let batched_gaussians = executor.create_gaussian_buffers(particles)?;
    let batched_bind_group =
        executor.create_gaussian_bind_group(&batched_gaussians.refs(), particles)?;
    let completed = executor.step_state_many_into_gaussian_bind_group(
        &mut batched_state,
        &batched_bind_group,
        steps,
    )?;
    assert_eq!(completed, steps);

    let repeated = executor.read_state(&repeated_state)?;
    let batched = executor.read_state(&batched_state)?;
    let repeated_gaussian = executor.read_gaussian_buffers(&repeated_gaussians)?;
    let batched_gaussian = executor.read_gaussian_buffers(&batched_gaussians)?;

    let max_pos = max_position_abs_error(&repeated.next_positions, &batched.next_positions);
    let max_state = max_abs_error(&repeated.next_states, &batched.next_states);
    assert!(
        max_pos <= 2.5e-6,
        "batched stochastic rollout position drift {max_pos}"
    );
    assert!(
        max_state <= 2.5e-6,
        "batched stochastic rollout state drift {max_state}"
    );
    let max_gaussian_position = max_abs_error(
        &repeated_gaussian.position_visibility,
        &batched_gaussian.position_visibility,
    );
    let max_gaussian_sh = max_abs_error(
        &repeated_gaussian.spherical_harmonic,
        &batched_gaussian.spherical_harmonic,
    );
    let max_gaussian_rotation =
        max_abs_error(&repeated_gaussian.rotation, &batched_gaussian.rotation);
    let max_gaussian_scale_opacity = max_abs_error(
        &repeated_gaussian.scale_opacity,
        &batched_gaussian.scale_opacity,
    );
    assert!(
        max_gaussian_position <= 5.0e-6,
        "batched gaussian position drift {max_gaussian_position}"
    );
    assert!(
        max_gaussian_sh <= 5.0e-6,
        "batched gaussian SH drift {max_gaussian_sh}"
    );
    assert!(
        max_gaussian_rotation <= 5.0e-6,
        "batched gaussian rotation drift {max_gaussian_rotation}"
    );
    assert!(
        max_gaussian_scale_opacity <= 5.0e-6,
        "batched gaussian scale/opacity drift {max_gaussian_scale_opacity}"
    );
    Ok(())
}

#[test]
fn wgpu_uv_torus_seed_writes_stationary_gaussians_on_gpu() -> Result<(), Box<dyn std::error::Error>>
{
    const SH_C0: f32 = 0.282_094_8;

    let particles = 256;
    let seed_scale = 0.72;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        37,
        ParticleSeed::UvTorus3d,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state_with_update_prob(
        &model, &positions, &states, 1, particles, &grid, 1.0, 1.0, 37,
    )?;
    let gaussian_buffers = executor.create_gaussian_buffers(particles)?;
    executor.step_state_into_gaussians(&mut gpu_state, &gaussian_buffers.refs())?;

    let gpu = executor.read_state(&gpu_state)?;
    let gaussian = executor.read_gaussian_buffers(&gaussian_buffers)?;
    let max_pos = max_position_abs_error(&positions, &gpu.next_positions);
    let max_state = max_abs_error(&states, &gpu.next_states);
    assert!(
        max_pos <= 1.0e-6,
        "stationary torus position drift {max_pos}"
    );
    assert!(
        max_state <= 1.0e-6,
        "stationary torus state drift {max_state}"
    );

    let major = seed_scale * UV_TORUS_INITIAL_SCALE;
    let minor = major * UV_TORUS_MINOR_RATIO;
    let expected_opacity = 1.0 / (1.0 + (-UV_TORUS_INITIAL_OPACITY_LOGIT).exp());
    let mut max_torus_error = 0.0_f32;
    let mut max_color_error = 0.0_f32;
    for (idx, expected_position) in positions.iter().enumerate().take(particles) {
        let base4 = idx * 4;
        let x = gaussian.position_visibility[base4];
        let y = gaussian.position_visibility[base4 + 1];
        let z = gaussian.position_visibility[base4 + 2];
        let radial = (x * x + y * y).sqrt();
        max_torus_error =
            max_torus_error.max((((radial - major).powi(2) + z.powi(2)).sqrt() - minor).abs());

        for (axis, expected) in expected_position.iter().enumerate().take(3) {
            assert!((gaussian.position_visibility[base4 + axis] - expected).abs() <= 1.0e-6);
            assert!(gaussian.scale_opacity[base4 + axis] > 0.0);
        }
        assert!((gaussian.scale_opacity[base4 + 3] - expected_opacity).abs() <= 1.0e-5);

        let sh_base = idx * burn_automata::gpu::GAUSSIAN_SH_COEFF_COUNT;
        let expected_target = uv_torus_sample(idx, particles, seed_scale).position;
        let expected_rgb = uv_torus_position_color(expected_target, seed_scale);
        for (channel, expected) in expected_rgb.iter().enumerate() {
            let value =
                (gaussian.spherical_harmonic[sh_base + channel] * SH_C0 + 0.5).clamp(0.0, 1.0);
            assert!(value.is_finite());
            max_color_error = max_color_error.max((value - expected).abs());
        }
    }

    assert!(
        max_torus_error <= 2.0e-5,
        "gpu gaussian torus surface error {max_torus_error}"
    );
    assert!(
        max_color_error <= 1.0e-5,
        "gpu color error {max_color_error}"
    );
    Ok(())
}

#[test]
fn wgpu_uv_torus_growth_step_writes_moving_gaussians_on_gpu()
-> Result<(), Box<dyn std::error::Error>> {
    let particles = 512;
    let seed_scale = 0.72;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = uv_torus_growth_model(config);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        37,
        ParticleSeed::UvTorusDense3d,
        seed_scale,
    );
    let executor = match new_executor_or_skip()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let mut gpu_state = executor.create_state_with_update_prob(
        &model, &positions, &states, 1, particles, &grid, 1.0, 1.0, 37,
    )?;
    let gaussian_buffers = executor.create_gaussian_buffers(particles)?;
    executor.step_state_into_gaussians(&mut gpu_state, &gaussian_buffers.refs())?;

    let gpu = executor.read_state(&gpu_state)?;
    let gaussian = executor.read_gaussian_buffers(&gaussian_buffers)?;
    let max_motion = positions
        .iter()
        .zip(gpu.next_positions.iter())
        .map(|(before, after)| {
            ((after[0] - before[0]).powi(2)
                + (after[1] - before[1]).powi(2)
                + (after[2] - before[2]).powi(2))
            .sqrt()
        })
        .fold(0.0_f32, f32::max);
    assert!(max_motion >= 1.0e-3, "uv torus GPU motion was {max_motion}");

    for idx in 0..particles {
        let base4 = idx * 4;
        for axis in 0..3 {
            assert!(
                (gaussian.position_visibility[base4 + axis] - gpu.next_positions[idx][axis]).abs()
                    <= 1.0e-6
            );
        }
    }
    Ok(())
}

fn uv_torus_growth_model(config: NpaConfig) -> NpaModel {
    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    for axis in 0..3 {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.w1[pos_hidden * input_dims + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + axis] = -1.0;
        weights.w2[axis * config.hidden_dims + pos_hidden] = UV_TORUS_MOTION_GAIN;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -UV_TORUS_MOTION_GAIN;
        let residual_out = config.spatial_dims + axis;
        weights.w2[residual_out * config.hidden_dims + pos_hidden] = -UV_TORUS_RESIDUAL_DECAY;
        weights.w2[residual_out * config.hidden_dims + neg_hidden] = UV_TORUS_RESIDUAL_DECAY;
    }
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
    NpaModel { config, weights }
}

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

#[test]
fn wgpu_neighbor_modes_match_cpu_oracle_for_2d() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn wgpu_scale_equivariant_auto_mode_preserves_scaled_rollout()
-> Result<(), Box<dyn std::error::Error>> {
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

fn assert_preset_parity(
    preset: AutomataPreset,
    particles: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let gpu = match burn_automata::gpu::step_wgpu_blocking(
        &model, &positions, &states, 1, particles, &grid, 1.0,
    ) {
        Ok(output) => output,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU parity test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };

    let max_pos = max_position_abs_error(&cpu.next_positions, &gpu.next_positions);
    let max_state = max_abs_error(&cpu.next_states, &gpu.next_states);
    let max_density = max_abs_error(&cpu.perception.density, &gpu.density);
    let density_index = max_abs_error_index(&cpu.perception.density, &gpu.density);
    eprintln!(
        "{preset:?}: max_pos={max_pos:.8} max_state={max_state:.8} max_density={max_density:.8} density_index={density_index:?}"
    );

    assert!(
        max_pos <= 2.5e-3,
        "max position abs error {max_pos} exceeded tolerance"
    );
    assert!(
        max_state <= 2.5e-3,
        "max state abs error {max_state} exceeded tolerance"
    );
    assert!(
        max_density <= 2.5e-3,
        "max density abs error {max_density} exceeded tolerance"
    );
    Ok(())
}

fn new_executor_or_skip()
-> Result<Option<burn_automata::gpu::WgpuAutomataExecutor>, Box<dyn std::error::Error>> {
    match burn_automata::gpu::WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => Ok(Some(executor)),
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping WGPU test: {message}");
            Ok(None)
        }
        Err(err) => Err(err.into()),
    }
}

fn is_missing_wgpu(message: &str) -> bool {
    message.contains("no WGPU adapter") || message.contains("failed to create WGPU device")
}

fn max_position_abs_error(lhs: &[[f32; 4]], rhs: &[[f32; 4]]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs.iter()).map(|(a, b)| (a - b).abs()))
        .fold(0.0, f32::max)
}

fn gpu_update_mask(count: usize, update_prob: f32, step: u32, seed: u64) -> Vec<f32> {
    let seed = (seed as u32) ^ ((seed >> 32) as u32);
    (0..count)
        .map(|idx| {
            let random = gpu_random01(idx as u32, step, seed);
            f32::from(random < update_prob)
        })
        .collect()
}

fn gpu_random01(particle: u32, step: u32, seed: u32) -> f32 {
    let mixed = hash_u32(particle ^ hash_u32(step.wrapping_add(0x9e37_79b9)) ^ seed);
    ((mixed >> 8) as f32) * (1.0 / 16_777_216.0)
}

fn hash_u32(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^ (x >> 15)
}

fn max_abs_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn max_abs_error_index(lhs: &[f32], rhs: &[f32]) -> Option<(usize, f32, f32)> {
    lhs.iter()
        .zip(rhs.iter())
        .enumerate()
        .max_by(|(_, (lhs_a, rhs_a)), (_, (lhs_b, rhs_b))| {
            ((*lhs_a - *rhs_a).abs())
                .partial_cmp(&(*lhs_b - *rhs_b).abs())
                .unwrap()
        })
        .map(|(idx, (lhs, rhs))| (idx, *lhs, *rhs))
}
