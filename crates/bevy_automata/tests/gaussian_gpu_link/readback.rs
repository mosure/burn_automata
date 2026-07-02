use super::{fixtures::*, prelude::*};

pub(crate) fn automata_gaussian_readback(
    particles: usize,
    steps: usize,
) -> Result<(WgpuGaussianReadback, usize), Box<dyn std::error::Error>> {
    let (model, grid, seed_scale) = lizard_or_seeded_model()?;
    let spatial_dims = model.config.spatial_dims;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        42,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => executor,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping automata gaussian readback test: {message}");
            return Err(std::io::Error::other(message).into());
        }
        Err(err) => return Err(err.into()),
    };
    let mut state = executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;
    let gaussian_buffers = executor.create_gaussian_buffers(particles)?;
    for _ in 0..steps.max(1) {
        executor.step_state_into_gaussians(&mut state, &gaussian_buffers.refs())?;
    }
    Ok((
        executor.read_gaussian_buffers(&gaussian_buffers)?,
        spatial_dims,
    ))
}

pub(crate) fn assert_compact_automata_gaussian_readback(
    readback: &WgpuGaussianReadback,
    particles: usize,
    spatial_dims: usize,
) {
    assert_eq!(readback.position_visibility.len(), particles * 4);
    assert_eq!(
        readback.spherical_harmonic.len(),
        particles * GAUSSIAN_SH_COEFF_COUNT
    );
    assert_eq!(readback.rotation.len(), particles * 4);
    assert_eq!(readback.scale_opacity.len(), particles * 4);

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut min_scale = f32::INFINITY;
    let mut max_scale = f32::NEG_INFINITY;
    for idx in 0..particles {
        let base = idx * 4;
        for axis in 0..3 {
            let position = readback.position_visibility[base + axis];
            assert!(position.is_finite(), "non-finite position at {idx}:{axis}");
            min[axis] = min[axis].min(position);
            max[axis] = max[axis].max(position);
            let scale = readback.scale_opacity[base + axis];
            assert!(scale.is_finite(), "non-finite scale at {idx}:{axis}");
            min_scale = min_scale.min(scale);
            max_scale = max_scale.max(scale);
        }
        assert_eq!(readback.position_visibility[base + 3], 1.0);
        if spatial_dims == 2 {
            assert_eq!(readback.scale_opacity[base + 3], 1.0);
        } else {
            assert!((0.05..=0.95).contains(&readback.scale_opacity[base + 3]));
        }
    }

    let width = max[0] - min[0];
    let height = max[1] - min[1];
    assert!(
        width > 0.05 && height > 0.05,
        "collapsed automata bounds: min={min:?} max={max:?}"
    );
    assert!(
        width < 1.25 && height < 1.25,
        "automata bounds too large: min={min:?} max={max:?}"
    );
    assert!(
        min_scale >= 0.0001 && max_scale <= 0.08,
        "unexpected gaussian scale range: min={min_scale} max={max_scale}"
    );
}

pub(crate) fn planar_cloud_from_readback(
    readback: &WgpuGaussianReadback,
    particles: usize,
) -> PlanarGaussian3d {
    let gaussians = (0..particles)
        .map(|idx| {
            let base = idx * 4;
            let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
            let mut coefficients = [0.0; GAUSSIAN_SH_COEFF_COUNT];
            coefficients.copy_from_slice(
                &readback.spherical_harmonic[sh_base..sh_base + GAUSSIAN_SH_COEFF_COUNT],
            );
            Gaussian3d {
                position_visibility: [
                    readback.position_visibility[base],
                    readback.position_visibility[base + 1],
                    readback.position_visibility[base + 2],
                    readback.position_visibility[base + 3],
                ]
                .into(),
                spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
                rotation: [
                    readback.rotation[base],
                    readback.rotation[base + 1],
                    readback.rotation[base + 2],
                    readback.rotation[base + 3],
                ]
                .into(),
                scale_opacity: [
                    readback.scale_opacity[base],
                    readback.scale_opacity[base + 1],
                    readback.scale_opacity[base + 2],
                    readback.scale_opacity[base + 3],
                ]
                .into(),
            }
        })
        .collect::<Vec<_>>();
    gaussians.into()
}

pub(crate) fn create_planar_storage(
    executor: &WgpuAutomataExecutor,
    count: usize,
) -> Result<PlanarStorageGaussian3d, Box<dyn std::error::Error>> {
    let device = executor.device();
    let storage_usage = BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE;
    let position_visibility = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_position_visibility"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let spherical_harmonic = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_spherical_harmonic"),
        size: byte_len::<f32>(count * GAUSSIAN_SH_COEFF_COUNT)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let rotation = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_rotation"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let scale_opacity = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_scale_opacity"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let draw_indirect_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_draw_indirect"),
        size: 16,
        usage: BufferUsages::INDIRECT
            | BufferUsages::COPY_DST
            | BufferUsages::COPY_SRC
            | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    Ok(PlanarStorageGaussian3d {
        position_visibility: position_visibility.into(),
        spherical_harmonic: spherical_harmonic.into(),
        rotation: rotation.into(),
        scale_opacity: scale_opacity.into(),
        count,
        draw_indirect_buffer: draw_indirect_buffer.into(),
    })
}

pub(crate) fn byte_len<T>(len: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| std::io::Error::other("buffer byte length overflow"))? as u64)
}
