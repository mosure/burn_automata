use super::common::*;

#[test]
fn burn_wgpu_writes_bevy_planar_gaussian_storage_buffers() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = bevy_test_guard();
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
        43,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => executor,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping Bevy gaussian GPU-link test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };
    let mut gpu_state =
        executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;
    let storage = create_planar_storage(&executor, particles)?;
    let gaussian_refs = gaussian_storage_buffer_refs(&storage);
    executor.step_state_into_gaussians(&mut gpu_state, &gaussian_refs)?;

    let gpu = executor.read_state(&gpu_state)?;
    let readback = executor.read_gaussian_buffer_refs(&gaussian_refs, storage.count)?;

    assert_eq!(storage.count, particles);
    assert_eq!(readback.position_visibility.len(), particles * 4);
    assert_eq!(
        readback.spherical_harmonic.len(),
        particles * GAUSSIAN_SH_COEFF_COUNT
    );
    for idx in 0..particles {
        let base = idx * 4;
        for axis in 0..3 {
            assert!(
                (readback.position_visibility[base + axis] - gpu.next_positions[idx][axis]).abs()
                    <= 1.0e-6
            );
        }
        assert_eq!(readback.position_visibility[base + 3], 1.0);
        if model.config.spatial_dims == 2 {
            assert_eq!(readback.scale_opacity[base + 3], 1.0);
        } else {
            assert!(readback.scale_opacity[base + 3] >= 0.05);
            assert!(readback.scale_opacity[base + 3] <= 0.95);
        }

        let state_base = idx * model.config.state_dims;
        let color_base = state_base + model.config.state_dims - 3;
        let expected_color = [
            (gpu.next_states[color_base] + 0.5).clamp(0.0, 1.0),
            (gpu.next_states[color_base + 1] + 0.5).clamp(0.0, 1.0),
            (gpu.next_states[color_base + 2] + 0.5).clamp(0.0, 1.0),
        ];
        let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
        for (channel, expected) in expected_color.iter().enumerate() {
            let decoded = 0.5 + SH_C0 * readback.spherical_harmonic[sh_base + channel];
            assert!(
                (decoded - *expected).abs() <= 2.0e-6,
                "particle {idx} channel {channel}: decoded {decoded} != expected {}",
                expected
            );
        }
    }
    assert!(
        readback
            .spherical_harmonic
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}
