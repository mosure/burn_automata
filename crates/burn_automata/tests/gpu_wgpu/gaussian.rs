use super::common::*;

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
