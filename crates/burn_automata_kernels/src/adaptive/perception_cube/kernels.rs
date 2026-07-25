use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use cubecl::{CubeDim, CubeLaunch, cube, prelude::*};

use super::super::{
    AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig, AdaptivePerceptionSemantics,
};
use super::grid::{self, GRID_HEIGHT, GRID_WIDTH};

const CUBE_UNITS: u32 = 256;
const SPARSE_GRID_MIN_PARTICLES: usize = 128;

pub(super) struct ForwardRaw<R: CubeRuntime> {
    pub(super) features: CubeTensor<R>,
    pub(super) density: CubeTensor<R>,
    pub(super) coarse_density: CubeTensor<R>,
    pub(super) raw_state_gradient: CubeTensor<R>,
    pub(super) state_gradient_inverse: CubeTensor<R>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_forward<R: CubeRuntime>(
    positions: CubeTensor<R>,
    states: CubeTensor<R>,
    represented_measure: CubeTensor<R>,
    bandwidth: CubeTensor<R>,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
) -> ForwardRaw<R> {
    let position_dims = positions.shape().dims::<3>();
    let state_dims = states.shape().dims::<3>();
    let batches = position_dims[0];
    let particles = position_dims[1];
    let channels = state_dims[2];
    let dtype = positions.dtype;
    let client = positions.client.clone();
    let device = positions.device.clone();
    let features = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particles, config.feature_dims(channels)]),
        dtype,
    );
    let density = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particles]),
        dtype,
    );
    let coarse_density = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particles]),
        dtype,
    );
    let batch_measure =
        empty_device_dtype(client.clone(), device.clone(), Shape::new([batches]), dtype);
    let raw_state_gradient = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particles, channels, 2]),
        dtype,
    );
    let state_gradient_inverse = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([batches, particles, 4]),
        dtype,
    );
    let sparse_grid = (particles >= SPARSE_GRID_MIN_PARTICLES)
        .then(|| grid::launch(positions.clone(), config.max_bandwidth));

    adaptive_batch_measure_kernel::launch(
        &client,
        elementwise_cube_count(batches),
        CubeDim::new_1d(CUBE_UNITS),
        AddressType::U32,
        represented_measure.clone().into_tensor_arg(),
        batch_measure.clone().into_tensor_arg(),
        dtype.into(),
    );
    if let Some(grid) = sparse_grid {
        adaptive_density_sparse_kernel::launch(
            &client,
            elementwise_cube_count(batches * particles),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.clone().into_tensor_arg(),
            represented_measure.clone().into_tensor_arg(),
            bandwidth.clone().into_tensor_arg(),
            grid.offsets.clone().into_tensor_arg(),
            grid.permutation.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            coarse_density.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
        adaptive_forward_metadata_sparse_kernel::launch(
            &client,
            elementwise_cube_count(batches * particles),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.clone().into_tensor_arg(),
            represented_measure.clone().into_tensor_arg(),
            bandwidth.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            batch_measure.into_tensor_arg(),
            grid.offsets.clone().into_tensor_arg(),
            grid.permutation.clone().into_tensor_arg(),
            features.clone().into_tensor_arg(),
            state_gradient_inverse.clone().into_tensor_arg(),
            channels,
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
        adaptive_forward_channel_sparse_kernel::launch(
            &client,
            elementwise_cube_count(batches * particles * channels),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.into_tensor_arg(),
            states.into_tensor_arg(),
            represented_measure.into_tensor_arg(),
            bandwidth.into_tensor_arg(),
            density.clone().into_tensor_arg(),
            state_gradient_inverse.clone().into_tensor_arg(),
            grid.offsets.into_tensor_arg(),
            grid.permutation.into_tensor_arg(),
            features.clone().into_tensor_arg(),
            raw_state_gradient.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
    } else {
        adaptive_density_tiled_kernel::launch(
            &client,
            CubeCount::Static(
                particles.div_ceil(CUBE_UNITS as usize) as u32,
                batches as u32,
                1,
            ),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.clone().into_tensor_arg(),
            represented_measure.clone().into_tensor_arg(),
            bandwidth.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            coarse_density.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
        adaptive_forward_metadata_kernel::launch(
            &client,
            elementwise_cube_count(batches * particles),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.clone().into_tensor_arg(),
            represented_measure.clone().into_tensor_arg(),
            bandwidth.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            batch_measure.into_tensor_arg(),
            features.clone().into_tensor_arg(),
            state_gradient_inverse.clone().into_tensor_arg(),
            channels,
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
        adaptive_forward_channel_kernel::launch(
            &client,
            elementwise_cube_count(batches * particles * channels),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.into_tensor_arg(),
            states.into_tensor_arg(),
            represented_measure.into_tensor_arg(),
            bandwidth.into_tensor_arg(),
            density.clone().into_tensor_arg(),
            state_gradient_inverse.clone().into_tensor_arg(),
            features.clone().into_tensor_arg(),
            raw_state_gradient.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
    }

    ForwardRaw {
        features,
        density,
        coarse_density,
        raw_state_gradient,
        state_gradient_inverse,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_state_adjoint<R: CubeRuntime>(
    positions: CubeTensor<R>,
    states: CubeTensor<R>,
    represented_measure: CubeTensor<R>,
    bandwidth: CubeTensor<R>,
    feature_grad: CubeTensor<R>,
    density: CubeTensor<R>,
    raw_state_gradient: CubeTensor<R>,
    state_gradient_inverse: CubeTensor<R>,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
) -> CubeTensor<R> {
    let state_dims = states.shape().dims::<3>();
    let batches = state_dims[0];
    let particles = state_dims[1];
    let channels = state_dims[2];
    let dtype = states.dtype;
    let client = states.client.clone();
    let device = states.device.clone();
    let raw_state_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particles, channels, 2]),
        dtype,
    );
    let state_grad = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([batches, particles, channels]),
        dtype,
    );
    let sparse_grid = (particles >= SPARSE_GRID_MIN_PARTICLES)
        .then(|| grid::launch(positions.clone(), config.max_bandwidth));
    let total = batches * particles * channels;
    adaptive_precompute_state_adjoint_kernel::launch(
        &client,
        elementwise_cube_count(total),
        CubeDim::new_1d(CUBE_UNITS),
        AddressType::U32,
        raw_state_gradient.into_tensor_arg(),
        state_gradient_inverse.into_tensor_arg(),
        bandwidth.clone().into_tensor_arg(),
        feature_grad.clone().into_tensor_arg(),
        raw_state_adjoint.clone().into_tensor_arg(),
        adaptive_args::<R>(options, config, semantics, dtype),
        dtype.into(),
    );
    if let Some(grid) = sparse_grid {
        adaptive_state_output_sparse_kernel::launch(
            &client,
            elementwise_cube_count(total),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.into_tensor_arg(),
            states.into_tensor_arg(),
            represented_measure.into_tensor_arg(),
            bandwidth.into_tensor_arg(),
            density.into_tensor_arg(),
            feature_grad.into_tensor_arg(),
            raw_state_adjoint.into_tensor_arg(),
            grid.offsets.into_tensor_arg(),
            grid.permutation.into_tensor_arg(),
            state_grad.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
    } else {
        adaptive_state_output_kernel::launch(
            &client,
            elementwise_cube_count(total),
            CubeDim::new_1d(CUBE_UNITS),
            AddressType::U32,
            positions.into_tensor_arg(),
            states.into_tensor_arg(),
            represented_measure.into_tensor_arg(),
            bandwidth.into_tensor_arg(),
            density.into_tensor_arg(),
            feature_grad.into_tensor_arg(),
            raw_state_adjoint.into_tensor_arg(),
            state_grad.clone().into_tensor_arg(),
            adaptive_args::<R>(options, config, semantics, dtype),
            dtype.into(),
        );
    }
    state_grad
}

fn elementwise_cube_count(elements: usize) -> CubeCount {
    CubeCount::Static(elements.div_ceil(CUBE_UNITS as usize) as u32, 1, 1)
}

#[derive(Clone, CubeLaunch, CubeType)]
struct AdaptiveArgs {
    eps0: InputScalar,
    reference_measure: InputScalar,
    pair_scale_power: InputScalar,
    max_bandwidth: InputScalar,
    shepard_epsilon: InputScalar,
    moment_regularization: InputScalar,
    moment_condition_limit: InputScalar,
    grid_width: u32,
    grid_height: u32,
    semantics: u32,
    scale_equivariance: u32,
    particle_density_equivariance: u32,
    log_norm_grad: u32,
    log_norm_density_grad: u32,
    normalized_log_gradients: u32,
    position_features: u32,
}

fn adaptive_args<R: CubeRuntime>(
    options: AdaptiveNpaPerceptionOptions,
    config: AdaptivePerceptionConfig,
    semantics: AdaptivePerceptionSemantics,
    dtype: DType,
) -> AdaptiveArgsLaunch<R> {
    AdaptiveArgsLaunch::new(
        InputScalar::new(options.eps0, dtype),
        InputScalar::new(config.reference_measure, dtype),
        InputScalar::new(config.pair_scale_power, dtype),
        InputScalar::new(config.max_bandwidth, dtype),
        InputScalar::new(config.shepard_epsilon, dtype),
        InputScalar::new(config.moment_regularization, dtype),
        InputScalar::new(config.moment_condition_limit, dtype),
        GRID_WIDTH,
        GRID_HEIGHT,
        match semantics {
            AdaptivePerceptionSemantics::NpaCompatible => 0,
            AdaptivePerceptionSemantics::NormalizedAdaptive => 1,
        },
        u32::from(options.scale_equivariance),
        u32::from(options.particle_density_equivariance),
        u32::from(options.log_norm_grad),
        u32::from(options.log_norm_density_grad),
        u32::from(config.log_normalize_gradients),
        u32::from(options.position_features),
    )
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_batch_measure_kernel<F: Float>(
    represented_measure: &Tensor<F>,
    batch_measure: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let batch = ABSOLUTE_POS;
    if batch >= represented_measure.shape(0) {
        terminate!();
    }
    let particles = represented_measure.shape(1);
    let mut total = F::new(0.0_f32);
    let mut particle = 0usize;
    while particle < particles {
        total += measure_value::<F>(represented_measure, batch, particle);
        particle += 1usize;
    }
    batch_measure[batch] = total;
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_density_tiled_kernel<F: Float>(
    positions: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &mut Tensor<F>,
    coarse_density: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = positions.shape(1);
    let unit = UNIT_POS as usize;
    let batch = CUBE_POS_Y as usize;
    let particle = CUBE_POS_X as usize * 256usize + unit;
    let active = batch < positions.shape(0) && particle < particles;
    let mut xi = F::new(0.0_f32);
    let mut yi = F::new(0.0_f32);
    let mut hi = F::new(1.0_f32);
    if active {
        xi = position_value::<F>(positions, batch, particle, 0usize);
        yi = position_value::<F>(positions, batch, particle, 1usize);
        hi = material_value::<F>(bandwidth, batch, particle);
    }

    let mut tile_x = SharedMemory::<F>::new(256usize);
    let mut tile_y = SharedMemory::<F>::new(256usize);
    let mut tile_h = SharedMemory::<F>::new(256usize);
    let mut tile_measure = SharedMemory::<F>::new(256usize);
    let mut rho = if normalized_semantics(args) {
        args.shepard_epsilon.get::<F>()
    } else {
        F::new(0.0_f32)
    };
    let mut coarse_rho = F::new(0.0_f32);
    let mut tile_start = 0usize;
    while tile_start < particles {
        let source = tile_start + unit;
        if source < particles {
            tile_x[unit] = position_value::<F>(positions, batch, source, 0usize);
            tile_y[unit] = position_value::<F>(positions, batch, source, 1usize);
            tile_h[unit] = material_value::<F>(bandwidth, batch, source);
            tile_measure[unit] = measure_value::<F>(represented_measure, batch, source);
        } else {
            tile_x[unit] = F::new(0.0_f32);
            tile_y[unit] = F::new(0.0_f32);
            tile_h[unit] = F::new(1.0_f32);
            tile_measure[unit] = F::new(0.0_f32);
        }
        sync_cube();

        if active {
            let mut tile_len = 256usize;
            if tile_start + tile_len > particles {
                tile_len = particles - tile_start;
            }
            let mut local = 0usize;
            while local < tile_len {
                let dx = tile_x[local] - xi;
                let dy = tile_y[local] - yi;
                let pair_h = pair_bandwidth::<F>(hi, tile_h[local], args);
                let contribution = tile_measure[local]
                    * perception_kernel_2d::<F>(dx * dx + dy * dy, pair_h, args);
                rho += contribution;
                if coarse_source::<F>(tile_measure[local], tile_h[local], args) {
                    coarse_rho += contribution;
                }
                local += 1usize;
            }
        }
        sync_cube();
        tile_start += 256usize;
    }
    if active {
        write_material::<F>(density, batch, particle, rho);
        write_material::<F>(coarse_density, batch, particle, coarse_rho);
    }
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_density_sparse_kernel<F: Float>(
    positions: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    density: &mut Tensor<F>,
    coarse_density: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = positions.shape(1);
    let index = ABSOLUTE_POS;
    if index >= positions.shape(0) * particles {
        terminate!();
    }
    let batch = index / particles;
    let particle = index - batch * particles;
    let xi = position_value::<F>(positions, batch, particle, 0usize);
    let yi = position_value::<F>(positions, batch, particle, 1usize);
    let hi = material_value::<F>(bandwidth, batch, particle);
    let (cell_x, cell_y) = adaptive_grid_cell::<F>(xi, yi, args);
    let mut rho = if normalized_semantics(args) {
        args.shepard_epsilon.get::<F>()
    } else {
        F::new(0.0_f32)
    };
    let mut coarse_rho = F::new(0.0_f32);

    let mut row: usize = 0usize;
    while row < 3usize {
        let neighbor_y = cell_y + row as i32 - 1i32;
        if neighbor_y >= 0i32 && neighbor_y < args.grid_height as i32 {
            let first_x = clamp(cell_x - 1i32, 0i32, args.grid_width as i32 - 1i32);
            let last_x = clamp(cell_x + 1i32, 0i32, args.grid_width as i32 - 1i32);
            let first_cell = neighbor_y as usize * args.grid_width as usize + first_x as usize;
            let last_cell = neighbor_y as usize * args.grid_width as usize + last_x as usize;
            let mut slot = adaptive_grid_offset(offsets, batch, first_cell);
            let end = adaptive_grid_offset(offsets, batch, last_cell + 1usize);
            while slot < end {
                let source = adaptive_grid_particle(permutation, batch, slot);
                let dx = position_value::<F>(positions, batch, source, 0usize) - xi;
                let dy = position_value::<F>(positions, batch, source, 1usize) - yi;
                let source_h = material_value::<F>(bandwidth, batch, source);
                let pair_h = pair_bandwidth::<F>(hi, source_h, args);
                let source_measure = measure_value::<F>(represented_measure, batch, source);
                let contribution =
                    source_measure * perception_kernel_2d::<F>(dx * dx + dy * dy, pair_h, args);
                rho += contribution;
                if coarse_source::<F>(source_measure, source_h, args) {
                    coarse_rho += contribution;
                }
                slot += 1usize;
            }
        }
        row += 1usize;
    }
    write_material::<F>(density, batch, particle, rho);
    write_material::<F>(coarse_density, batch, particle, coarse_rho);
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_forward_metadata_kernel<F: Float>(
    positions: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    batch_measure: &Tensor<F>,
    features: &mut Tensor<F>,
    inverse: &mut Tensor<F>,
    #[comptime] state_dims: usize,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = positions.shape(1);
    let index = ABSOLUTE_POS;
    if index >= positions.shape(0) * particles {
        terminate!();
    }
    let batch = index / particles;
    let particle = index - batch * particles;
    let xi = position_value::<F>(positions, batch, particle, 0usize);
    let yi = position_value::<F>(positions, batch, particle, 1usize);
    let hi = material_value::<F>(bandwidth, batch, particle);
    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    let mut density_x = F::new(0.0_f32);
    let mut density_y = F::new(0.0_f32);
    let total_measure = batch_measure[batch];
    let mean_measure = total_measure / F::cast_from(particles as f32);

    let mut source = 0usize;
    while source < particles {
        if source != particle {
            let dx = position_value::<F>(positions, batch, source, 0usize) - xi;
            let dy = position_value::<F>(positions, batch, source, 1usize) - yi;
            let r2 = dx * dx + dy * dy;
            let pair_h =
                pair_bandwidth::<F>(hi, material_value::<F>(bandwidth, batch, source), args);
            let measure = measure_value::<F>(represented_measure, batch, source);
            let (gx, gy, dgx, dgy) = if normalized_semantics(args) {
                let (gx, gy) = normalized_kernel_gradient_2d::<F>(dx, dy, r2, pair_h, measure);
                (gx, gy, gx, gy)
            } else {
                let volume =
                    measure * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
                let (gx, gy) = spiky_gradient_2d::<F>(dx, dy, r2, pair_h, volume);
                let density_weight = if args.particle_density_equivariance != 0u32 {
                    measure * reciprocal_finite::<F>(total_measure)
                } else {
                    measure * reciprocal_finite::<F>(mean_measure)
                };
                let (dgx, dgy) = spiky_gradient_2d::<F>(dx, dy, r2, pair_h, density_weight);
                (gx, gy, dgx, dgy)
            };
            m00 += dx * gx;
            m01 += dx * gy;
            m11 += dy * gy;
            density_x += dgx;
            density_y += dgy;
        }
        source += 1usize;
    }
    let (inv00, inv01, inv10, inv11) = perception_inverse_2d::<F>(m00, m01, m11, args);
    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    inverse[inverse_base] = inv00;
    inverse[inverse_base + inverse.stride(2)] = inv01;
    inverse[inverse_base + 2usize * inverse.stride(2)] = inv10;
    inverse[inverse_base + 3usize * inverse.stride(2)] = inv11;

    let mut density_scale = if normalized_semantics(args) {
        hi * reciprocal_finite::<F>(material_value::<F>(density, batch, particle))
    } else {
        F::new(1.0_f32)
    };
    if !normalized_semantics(args) && args.scale_equivariance != 0u32 {
        density_scale = (hi / args.eps0.get::<F>()).powf(F::new(3.0_f32));
    }
    density_x *= density_scale;
    density_y *= density_scale;
    if perception_density_log_normalize(args) {
        (density_x, density_y) = log_normalize_2::<F>(density_x, density_y);
    }
    let density_cursor = state_dims * 4usize;
    write_feature::<F>(features, batch, particle, density_cursor, density_x);
    write_feature::<F>(
        features,
        batch,
        particle,
        density_cursor + 1usize,
        density_y,
    );
    if args.position_features != 0u32 {
        write_feature::<F>(features, batch, particle, density_cursor + 2usize, xi);
        write_feature::<F>(features, batch, particle, density_cursor + 3usize, yi);
    }
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_forward_metadata_sparse_kernel<F: Float>(
    positions: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    batch_measure: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    features: &mut Tensor<F>,
    inverse: &mut Tensor<F>,
    #[comptime] state_dims: usize,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = positions.shape(1);
    let index = ABSOLUTE_POS;
    if index >= positions.shape(0) * particles {
        terminate!();
    }
    let batch = index / particles;
    let particle = index - batch * particles;
    let xi = position_value::<F>(positions, batch, particle, 0usize);
    let yi = position_value::<F>(positions, batch, particle, 1usize);
    let hi = material_value::<F>(bandwidth, batch, particle);
    let (cell_x, cell_y) = adaptive_grid_cell::<F>(xi, yi, args);
    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    let mut density_x = F::new(0.0_f32);
    let mut density_y = F::new(0.0_f32);
    let total_measure = batch_measure[batch];
    let mean_measure = total_measure / F::cast_from(particles as f32);

    let mut row: usize = 0usize;
    while row < 3usize {
        let neighbor_y = cell_y + row as i32 - 1i32;
        if neighbor_y >= 0i32 && neighbor_y < args.grid_height as i32 {
            let first_x = clamp(cell_x - 1i32, 0i32, args.grid_width as i32 - 1i32);
            let last_x = clamp(cell_x + 1i32, 0i32, args.grid_width as i32 - 1i32);
            let first_cell = neighbor_y as usize * args.grid_width as usize + first_x as usize;
            let last_cell = neighbor_y as usize * args.grid_width as usize + last_x as usize;
            let mut slot = adaptive_grid_offset(offsets, batch, first_cell);
            let end = adaptive_grid_offset(offsets, batch, last_cell + 1usize);
            while slot < end {
                let source = adaptive_grid_particle(permutation, batch, slot);
                if source != particle {
                    let dx = position_value::<F>(positions, batch, source, 0usize) - xi;
                    let dy = position_value::<F>(positions, batch, source, 1usize) - yi;
                    let r2 = dx * dx + dy * dy;
                    let pair_h = pair_bandwidth::<F>(
                        hi,
                        material_value::<F>(bandwidth, batch, source),
                        args,
                    );
                    let measure = measure_value::<F>(represented_measure, batch, source);
                    let (gx, gy, dgx, dgy) = if normalized_semantics(args) {
                        let (gx, gy) =
                            normalized_kernel_gradient_2d::<F>(dx, dy, r2, pair_h, measure);
                        (gx, gy, gx, gy)
                    } else {
                        let volume = measure
                            * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
                        let (gx, gy) = spiky_gradient_2d::<F>(dx, dy, r2, pair_h, volume);
                        let density_weight = if args.particle_density_equivariance != 0u32 {
                            measure * reciprocal_finite::<F>(total_measure)
                        } else {
                            measure * reciprocal_finite::<F>(mean_measure)
                        };
                        let (dgx, dgy) = spiky_gradient_2d::<F>(dx, dy, r2, pair_h, density_weight);
                        (gx, gy, dgx, dgy)
                    };
                    m00 += dx * gx;
                    m01 += dx * gy;
                    m11 += dy * gy;
                    density_x += dgx;
                    density_y += dgy;
                }
                slot += 1usize;
            }
        }
        row += 1usize;
    }
    let (inv00, inv01, inv10, inv11) = perception_inverse_2d::<F>(m00, m01, m11, args);
    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    inverse[inverse_base] = inv00;
    inverse[inverse_base + inverse.stride(2)] = inv01;
    inverse[inverse_base + 2usize * inverse.stride(2)] = inv10;
    inverse[inverse_base + 3usize * inverse.stride(2)] = inv11;

    let mut density_scale = if normalized_semantics(args) {
        hi * reciprocal_finite::<F>(material_value::<F>(density, batch, particle))
    } else {
        F::new(1.0_f32)
    };
    if !normalized_semantics(args) && args.scale_equivariance != 0u32 {
        density_scale = (hi / args.eps0.get::<F>()).powf(F::new(3.0_f32));
    }
    density_x *= density_scale;
    density_y *= density_scale;
    if perception_density_log_normalize(args) {
        (density_x, density_y) = log_normalize_2::<F>(density_x, density_y);
    }
    let density_cursor = state_dims * 4usize;
    write_feature::<F>(features, batch, particle, density_cursor, density_x);
    write_feature::<F>(
        features,
        batch,
        particle,
        density_cursor + 1usize,
        density_y,
    );
    if args.position_features != 0u32 {
        write_feature::<F>(features, batch, particle, density_cursor + 2usize, xi);
        write_feature::<F>(features, batch, particle, density_cursor + 3usize, yi);
    }
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_forward_channel_kernel<F: Float>(
    positions: &Tensor<F>,
    states: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    inverse: &Tensor<F>,
    features: &mut Tensor<F>,
    raw_state_gradient: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = states.shape(1);
    let state_dims = states.shape(2);
    let index = ABSOLUTE_POS;
    if index >= states.shape(0) * particles * state_dims {
        terminate!();
    }
    let batch = index / (particles * state_dims);
    let local = index - batch * particles * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let state_i = state_value::<F>(states, batch, particle, channel);
    let xi = position_value::<F>(positions, batch, particle, 0usize);
    let yi = position_value::<F>(positions, batch, particle, 1usize);
    let hi = material_value::<F>(bandwidth, batch, particle);
    let mut blur = if normalized_semantics(args) {
        args.shepard_epsilon.get::<F>() * state_i
    } else {
        F::new(0.0_f32)
    };
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);

    let mut source = 0usize;
    while source < particles {
        let dx = position_value::<F>(positions, batch, source, 0usize) - xi;
        let dy = position_value::<F>(positions, batch, source, 1usize) - yi;
        let r2 = dx * dx + dy * dy;
        let pair_h = pair_bandwidth::<F>(hi, material_value::<F>(bandwidth, batch, source), args);
        let measure = measure_value::<F>(represented_measure, batch, source);
        let state_j = state_value::<F>(states, batch, source, channel);
        if normalized_semantics(args) {
            blur += measure * normalized_kernel_2d::<F>(r2, pair_h) * state_j;
        } else {
            let volume =
                measure * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
            blur += volume * poly6_2d::<F>(r2, pair_h) * state_j;
        }
        if source != particle {
            let (gx, gy) = if normalized_semantics(args) {
                normalized_kernel_gradient_2d::<F>(dx, dy, r2, pair_h, measure)
            } else {
                let volume =
                    measure * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
                spiky_gradient_2d::<F>(dx, dy, r2, pair_h, volume)
            };
            let difference = state_j - state_i;
            raw_x += difference * gx;
            raw_y += difference * gy;
        }
        source += 1usize;
    }
    if normalized_semantics(args) {
        blur *= reciprocal_finite::<F>(material_value::<F>(density, batch, particle));
    }
    write_feature::<F>(features, batch, particle, channel, state_i);
    write_feature::<F>(features, batch, particle, state_dims + channel, blur);
    let raw_base = batch * raw_state_gradient.stride(0)
        + particle * raw_state_gradient.stride(1)
        + channel * raw_state_gradient.stride(2);
    raw_state_gradient[raw_base] = raw_x;
    raw_state_gradient[raw_base + raw_state_gradient.stride(3)] = raw_y;

    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    let inv00 = inverse[inverse_base];
    let inv01 = inverse[inverse_base + inverse.stride(2)];
    let inv10 = inverse[inverse_base + 2usize * inverse.stride(2)];
    let inv11 = inverse[inverse_base + 3usize * inverse.stride(2)];
    let mut scale = if normalized_semantics(args) {
        hi
    } else {
        F::new(1.0_f32)
    };
    if !normalized_semantics(args) && args.scale_equivariance != 0u32 {
        scale = hi / args.eps0.get::<F>();
    }
    let corrected_x = (raw_x * inv00 + raw_y * inv10) * scale;
    let corrected_y = (raw_x * inv01 + raw_y * inv11) * scale;
    let (out_x, out_y) = if perception_state_log_normalize(args) {
        log_normalize_2::<F>(corrected_x, corrected_y)
    } else {
        (corrected_x, corrected_y)
    };
    let cursor = state_dims * 2usize + channel * 2usize;
    write_feature::<F>(features, batch, particle, cursor, out_x);
    write_feature::<F>(features, batch, particle, cursor + 1usize, out_y);
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_forward_channel_sparse_kernel<F: Float>(
    positions: &Tensor<F>,
    states: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    inverse: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    features: &mut Tensor<F>,
    raw_state_gradient: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = states.shape(1);
    let state_dims = states.shape(2);
    let index = ABSOLUTE_POS;
    if index >= states.shape(0) * particles * state_dims {
        terminate!();
    }
    let batch = index / (particles * state_dims);
    let local = index - batch * particles * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let state_i = state_value::<F>(states, batch, particle, channel);
    let xi = position_value::<F>(positions, batch, particle, 0usize);
    let yi = position_value::<F>(positions, batch, particle, 1usize);
    let hi = material_value::<F>(bandwidth, batch, particle);
    let (cell_x, cell_y) = adaptive_grid_cell::<F>(xi, yi, args);
    let mut blur = if normalized_semantics(args) {
        args.shepard_epsilon.get::<F>() * state_i
    } else {
        F::new(0.0_f32)
    };
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);

    let mut row: usize = 0usize;
    while row < 3usize {
        let neighbor_y = cell_y + row as i32 - 1i32;
        if neighbor_y >= 0i32 && neighbor_y < args.grid_height as i32 {
            let first_x = clamp(cell_x - 1i32, 0i32, args.grid_width as i32 - 1i32);
            let last_x = clamp(cell_x + 1i32, 0i32, args.grid_width as i32 - 1i32);
            let first_cell = neighbor_y as usize * args.grid_width as usize + first_x as usize;
            let last_cell = neighbor_y as usize * args.grid_width as usize + last_x as usize;
            let mut slot = adaptive_grid_offset(offsets, batch, first_cell);
            let end = adaptive_grid_offset(offsets, batch, last_cell + 1usize);
            while slot < end {
                let source = adaptive_grid_particle(permutation, batch, slot);
                let dx = position_value::<F>(positions, batch, source, 0usize) - xi;
                let dy = position_value::<F>(positions, batch, source, 1usize) - yi;
                let r2 = dx * dx + dy * dy;
                let pair_h =
                    pair_bandwidth::<F>(hi, material_value::<F>(bandwidth, batch, source), args);
                let measure = measure_value::<F>(represented_measure, batch, source);
                let state_j = state_value::<F>(states, batch, source, channel);
                if normalized_semantics(args) {
                    blur += measure * normalized_kernel_2d::<F>(r2, pair_h) * state_j;
                } else {
                    let volume = measure
                        * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
                    blur += volume * poly6_2d::<F>(r2, pair_h) * state_j;
                }
                if source != particle {
                    let (gx, gy) = if normalized_semantics(args) {
                        normalized_kernel_gradient_2d::<F>(dx, dy, r2, pair_h, measure)
                    } else {
                        let volume = measure
                            * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
                        spiky_gradient_2d::<F>(dx, dy, r2, pair_h, volume)
                    };
                    let difference = state_j - state_i;
                    raw_x += difference * gx;
                    raw_y += difference * gy;
                }
                slot += 1usize;
            }
        }
        row += 1usize;
    }
    if normalized_semantics(args) {
        blur *= reciprocal_finite::<F>(material_value::<F>(density, batch, particle));
    }
    write_feature::<F>(features, batch, particle, channel, state_i);
    write_feature::<F>(features, batch, particle, state_dims + channel, blur);
    let raw_base = batch * raw_state_gradient.stride(0)
        + particle * raw_state_gradient.stride(1)
        + channel * raw_state_gradient.stride(2);
    raw_state_gradient[raw_base] = raw_x;
    raw_state_gradient[raw_base + raw_state_gradient.stride(3)] = raw_y;

    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    let inv00 = inverse[inverse_base];
    let inv01 = inverse[inverse_base + inverse.stride(2)];
    let inv10 = inverse[inverse_base + 2usize * inverse.stride(2)];
    let inv11 = inverse[inverse_base + 3usize * inverse.stride(2)];
    let mut scale = if normalized_semantics(args) {
        hi
    } else {
        F::new(1.0_f32)
    };
    if !normalized_semantics(args) && args.scale_equivariance != 0u32 {
        scale = hi / args.eps0.get::<F>();
    }
    let corrected_x = (raw_x * inv00 + raw_y * inv10) * scale;
    let corrected_y = (raw_x * inv01 + raw_y * inv11) * scale;
    let (out_x, out_y) = if perception_state_log_normalize(args) {
        log_normalize_2::<F>(corrected_x, corrected_y)
    } else {
        (corrected_x, corrected_y)
    };
    let cursor = state_dims * 2usize + channel * 2usize;
    write_feature::<F>(features, batch, particle, cursor, out_x);
    write_feature::<F>(features, batch, particle, cursor + 1usize, out_y);
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_precompute_state_adjoint_kernel<F: Float>(
    raw_state_gradient: &Tensor<F>,
    inverse: &Tensor<F>,
    bandwidth: &Tensor<F>,
    feature_grad: &Tensor<F>,
    raw_state_adjoint: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = raw_state_gradient.shape(1);
    let state_dims = raw_state_gradient.shape(2);
    let index = ABSOLUTE_POS;
    if index >= raw_state_gradient.shape(0) * particles * state_dims {
        terminate!();
    }
    let batch = index / (particles * state_dims);
    let local = index - batch * particles * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let raw_base = batch * raw_state_gradient.stride(0)
        + particle * raw_state_gradient.stride(1)
        + channel * raw_state_gradient.stride(2);
    let raw_x = raw_state_gradient[raw_base];
    let raw_y = raw_state_gradient[raw_base + raw_state_gradient.stride(3)];
    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    let inv00 = inverse[inverse_base];
    let inv01 = inverse[inverse_base + inverse.stride(2)];
    let inv10 = inverse[inverse_base + 2usize * inverse.stride(2)];
    let inv11 = inverse[inverse_base + 3usize * inverse.stride(2)];
    let mut scale = if normalized_semantics(args) {
        material_value::<F>(bandwidth, batch, particle)
    } else {
        F::new(1.0_f32)
    };
    if !normalized_semantics(args) && args.scale_equivariance != 0u32 {
        scale = material_value::<F>(bandwidth, batch, particle) / args.eps0.get::<F>();
    }
    let input_x = (raw_x * inv00 + raw_y * inv10) * scale;
    let input_y = (raw_x * inv01 + raw_y * inv11) * scale;
    let cursor = state_dims * 2usize + channel * 2usize;
    let grad_x = feature_value::<F>(feature_grad, batch, particle, cursor);
    let grad_y = feature_value::<F>(feature_grad, batch, particle, cursor + 1usize);
    let (mut corrected_x, mut corrected_y) = if perception_state_log_normalize(args) {
        log_normalize_adjoint_2::<F>(input_x, input_y, grad_x, grad_y)
    } else {
        (grad_x, grad_y)
    };
    corrected_x *= scale;
    corrected_y *= scale;
    let output_base = batch * raw_state_adjoint.stride(0)
        + particle * raw_state_adjoint.stride(1)
        + channel * raw_state_adjoint.stride(2);
    raw_state_adjoint[output_base] = corrected_x * inv00 + corrected_y * inv01;
    raw_state_adjoint[output_base + raw_state_adjoint.stride(3)] =
        corrected_x * inv10 + corrected_y * inv11;
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_state_output_kernel<F: Float>(
    positions: &Tensor<F>,
    states: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    feature_grad: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    state_grad: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = states.shape(1);
    let state_dims = states.shape(2);
    let index = ABSOLUTE_POS;
    if index >= states.shape(0) * particles * state_dims {
        terminate!();
    }
    let batch = index / (particles * state_dims);
    let local = index - batch * particles * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let xp = position_value::<F>(positions, batch, particle, 0usize);
    let yp = position_value::<F>(positions, batch, particle, 1usize);
    let hp = material_value::<F>(bandwidth, batch, particle);
    let mut output = feature_value::<F>(feature_grad, batch, particle, channel);
    let blur_cursor = state_dims;

    let mut query = 0usize;
    while query < particles {
        let xq = position_value::<F>(positions, batch, query, 0usize);
        let yq = position_value::<F>(positions, batch, query, 1usize);
        let pair_h = pair_bandwidth::<F>(material_value::<F>(bandwidth, batch, query), hp, args);
        let dx = xp - xq;
        let dy = yp - yq;
        let r2 = dx * dx + dy * dy;
        let blur_weight = if normalized_semantics(args) {
            measure_value::<F>(represented_measure, batch, particle)
                * normalized_kernel_2d::<F>(r2, pair_h)
                * reciprocal_finite::<F>(material_value::<F>(density, batch, query))
        } else {
            measure_value::<F>(represented_measure, batch, particle)
                * reciprocal_finite::<F>(material_value::<F>(density, batch, particle))
                * poly6_2d::<F>(r2, pair_h)
        };
        output +=
            feature_value::<F>(feature_grad, batch, query, blur_cursor + channel) * blur_weight;
        if normalized_semantics(args) && query == particle {
            output += feature_value::<F>(feature_grad, batch, query, blur_cursor + channel)
                * args.shepard_epsilon.get::<F>()
                * reciprocal_finite::<F>(material_value::<F>(density, batch, query));
        }
        if query != particle {
            output += raw_pair_contribution::<F>(
                positions,
                represented_measure,
                bandwidth,
                density,
                raw_state_adjoint,
                batch,
                query,
                particle,
                channel,
                args,
            );
        } else {
            let mut source = 0usize;
            while source < particles {
                if source != particle {
                    output -= raw_pair_contribution::<F>(
                        positions,
                        represented_measure,
                        bandwidth,
                        density,
                        raw_state_adjoint,
                        batch,
                        particle,
                        source,
                        channel,
                        args,
                    );
                }
                source += 1usize;
            }
        }
        query += 1usize;
    }
    let output_base = batch * state_grad.stride(0)
        + particle * state_grad.stride(1)
        + channel * state_grad.stride(2);
    state_grad[output_base] = output;
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_state_output_sparse_kernel<F: Float>(
    positions: &Tensor<F>,
    states: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    feature_grad: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    state_grad: &mut Tensor<F>,
    args: &AdaptiveArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particles = states.shape(1);
    let state_dims = states.shape(2);
    let index = ABSOLUTE_POS;
    if index >= states.shape(0) * particles * state_dims {
        terminate!();
    }
    let batch = index / (particles * state_dims);
    let local = index - batch * particles * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let xp = position_value::<F>(positions, batch, particle, 0usize);
    let yp = position_value::<F>(positions, batch, particle, 1usize);
    let hp = material_value::<F>(bandwidth, batch, particle);
    let (cell_x, cell_y) = adaptive_grid_cell::<F>(xp, yp, args);
    let mut output = feature_value::<F>(feature_grad, batch, particle, channel);
    let blur_cursor = state_dims;

    let mut row: usize = 0usize;
    while row < 3usize {
        let neighbor_y = cell_y + row as i32 - 1i32;
        if neighbor_y >= 0i32 && neighbor_y < args.grid_height as i32 {
            let first_x = clamp(cell_x - 1i32, 0i32, args.grid_width as i32 - 1i32);
            let last_x = clamp(cell_x + 1i32, 0i32, args.grid_width as i32 - 1i32);
            let first_cell = neighbor_y as usize * args.grid_width as usize + first_x as usize;
            let last_cell = neighbor_y as usize * args.grid_width as usize + last_x as usize;
            let mut slot = adaptive_grid_offset(offsets, batch, first_cell);
            let end = adaptive_grid_offset(offsets, batch, last_cell + 1usize);
            while slot < end {
                let query = adaptive_grid_particle(permutation, batch, slot);
                let dx = xp - position_value::<F>(positions, batch, query, 0usize);
                let dy = yp - position_value::<F>(positions, batch, query, 1usize);
                let pair_h =
                    pair_bandwidth::<F>(material_value::<F>(bandwidth, batch, query), hp, args);
                let blur_weight = if normalized_semantics(args) {
                    measure_value::<F>(represented_measure, batch, particle)
                        * normalized_kernel_2d::<F>(dx * dx + dy * dy, pair_h)
                        * reciprocal_finite::<F>(material_value::<F>(density, batch, query))
                } else {
                    measure_value::<F>(represented_measure, batch, particle)
                        * reciprocal_finite::<F>(material_value::<F>(density, batch, particle))
                        * poly6_2d::<F>(dx * dx + dy * dy, pair_h)
                };
                output += feature_value::<F>(feature_grad, batch, query, blur_cursor + channel)
                    * blur_weight;
                if normalized_semantics(args) && query == particle {
                    output += feature_value::<F>(feature_grad, batch, query, blur_cursor + channel)
                        * args.shepard_epsilon.get::<F>()
                        * reciprocal_finite::<F>(material_value::<F>(density, batch, query));
                }
                if query != particle {
                    output += raw_pair_contribution::<F>(
                        positions,
                        represented_measure,
                        bandwidth,
                        density,
                        raw_state_adjoint,
                        batch,
                        query,
                        particle,
                        channel,
                        args,
                    );
                    output -= raw_pair_contribution::<F>(
                        positions,
                        represented_measure,
                        bandwidth,
                        density,
                        raw_state_adjoint,
                        batch,
                        particle,
                        query,
                        channel,
                        args,
                    );
                }
                slot += 1usize;
            }
        }
        row += 1usize;
    }
    let output_base = batch * state_grad.stride(0)
        + particle * state_grad.stride(1)
        + channel * state_grad.stride(2);
    state_grad[output_base] = output;
}

#[cube]
fn raw_pair_contribution<F: Float>(
    positions: &Tensor<F>,
    represented_measure: &Tensor<F>,
    bandwidth: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    batch: usize,
    target: usize,
    source: usize,
    channel: usize,
    args: &AdaptiveArgs,
) -> F {
    let dx = position_value::<F>(positions, batch, source, 0usize)
        - position_value::<F>(positions, batch, target, 0usize);
    let dy = position_value::<F>(positions, batch, source, 1usize)
        - position_value::<F>(positions, batch, target, 1usize);
    let pair_h = pair_bandwidth::<F>(
        material_value::<F>(bandwidth, batch, target),
        material_value::<F>(bandwidth, batch, source),
        args,
    );
    let measure = measure_value::<F>(represented_measure, batch, source);
    let (gx, gy) = if normalized_semantics(args) {
        normalized_kernel_gradient_2d::<F>(dx, dy, dx * dx + dy * dy, pair_h, measure)
    } else {
        let volume = measure * reciprocal_finite::<F>(material_value::<F>(density, batch, source));
        spiky_gradient_2d::<F>(dx, dy, dx * dx + dy * dy, pair_h, volume)
    };
    let base = batch * raw_state_adjoint.stride(0)
        + target * raw_state_adjoint.stride(1)
        + channel * raw_state_adjoint.stride(2);
    raw_state_adjoint[base] * gx + raw_state_adjoint[base + raw_state_adjoint.stride(3)] * gy
}

#[cube]
fn adaptive_grid_cell<F: Float>(x: F, y: F, args: &AdaptiveArgs) -> (i32, i32) {
    let cell_size = args.max_bandwidth.get::<F>();
    let width = args.grid_width as i32;
    let height = args.grid_height as i32;
    let mut cell_x = i32::cast_from((x / cell_size).floor()) + width / 2i32;
    let mut cell_y = i32::cast_from((y / cell_size).floor()) + height / 2i32;
    cell_x = clamp(cell_x, 0i32, width - 1i32);
    cell_y = clamp(cell_y, 0i32, height - 1i32);
    (cell_x, cell_y)
}

#[cube]
fn adaptive_grid_offset(offsets: &Tensor<u32>, batch: usize, cell: usize) -> usize {
    offsets[batch * offsets.stride(0) + cell * offsets.stride(1)] as usize
}

#[cube]
fn adaptive_grid_particle(permutation: &Tensor<u32>, batch: usize, slot: usize) -> usize {
    permutation[batch * permutation.stride(0) + slot * permutation.stride(1)] as usize
}

#[cube]
fn pair_bandwidth<F: Float>(lhs: F, rhs: F, args: &AdaptiveArgs) -> F {
    let power = args.pair_scale_power.get::<F>();
    ((lhs.powf(power) + rhs.powf(power)) * F::new(0.5_f32)).powf(F::new(1.0_f32) / power)
}

#[cube]
fn coarse_source<F: Float>(measure: F, bandwidth: F, args: &AdaptiveArgs) -> bool {
    let tolerance = F::new(1.0_f32 + 32.0_f32 * f32::EPSILON);
    let reference_measure = args.reference_measure.get::<F>();
    if reference_measure > F::new(0.0_f32) {
        measure > reference_measure * tolerance
    } else {
        bandwidth > args.eps0.get::<F>() * tolerance
    }
}

#[cube]
fn normalized_semantics(args: &AdaptiveArgs) -> bool {
    args.semantics == 1u32
}

#[cube]
fn perception_state_log_normalize(args: &AdaptiveArgs) -> bool {
    if normalized_semantics(args) {
        args.normalized_log_gradients != 0u32
    } else {
        args.log_norm_grad != 0u32
    }
}

#[cube]
fn perception_density_log_normalize(args: &AdaptiveArgs) -> bool {
    if normalized_semantics(args) {
        args.normalized_log_gradients != 0u32
    } else {
        args.log_norm_density_grad != 0u32
    }
}

#[cube]
fn perception_kernel_2d<F: Float>(distance2: F, bandwidth: F, args: &AdaptiveArgs) -> F {
    if normalized_semantics(args) {
        normalized_kernel_2d::<F>(distance2, bandwidth)
    } else {
        poly6_2d::<F>(distance2, bandwidth)
    }
}

#[cube]
fn normalized_kernel_2d<F: Float>(distance2: F, bandwidth: F) -> F {
    let bandwidth2 = bandwidth * bandwidth;
    let mut output = F::new(0.0_f32);
    if distance2 < bandwidth2 {
        let shoulder = F::new(1.0_f32) - distance2 / bandwidth2;
        output = shoulder * shoulder * shoulder / bandwidth2;
    }
    output
}

#[cube]
fn normalized_kernel_gradient_2d<F: Float>(
    dx: F,
    dy: F,
    distance2: F,
    bandwidth: F,
    coefficient: F,
) -> (F, F) {
    let bandwidth2 = bandwidth * bandwidth;
    let mut gx = F::new(0.0_f32);
    let mut gy = F::new(0.0_f32);
    if distance2 > F::new(0.0_f32) && distance2 < bandwidth2 {
        let shoulder = F::new(1.0_f32) - distance2 / bandwidth2;
        let scale = (F::new(0.0_f32) - F::new(6.0_f32)) * coefficient * shoulder * shoulder
            / (bandwidth2 * bandwidth2);
        gx = scale * dx;
        gy = scale * dy;
    }
    (gx, gy)
}

#[cube]
fn poly6_2d<F: Float>(distance2: F, bandwidth: F) -> F {
    let bandwidth2 = bandwidth * bandwidth;
    let mut output = F::new(0.0_f32);
    if distance2 < bandwidth2 {
        let normalization =
            F::new(4.0_f32) / (F::new(core::f32::consts::PI) * bandwidth.powf(F::new(8.0_f32)));
        output = normalization * (bandwidth2 - distance2).powf(F::new(3.0_f32));
    }
    output
}

#[cube]
fn spiky_gradient_2d<F: Float>(dx: F, dy: F, distance2: F, bandwidth: F, coefficient: F) -> (F, F) {
    let mut gx = F::new(0.0_f32);
    let mut gy = F::new(0.0_f32);
    if distance2 > F::new(0.0_f32) && distance2 < bandwidth * bandwidth {
        let distance = distance2.sqrt();
        let scale = coefficient * F::new(30.0_f32) * (bandwidth - distance).powf(F::new(2.0_f32))
            / (F::new(core::f32::consts::PI) * bandwidth.powf(F::new(5.0_f32)) * distance);
        gx = scale * dx;
        gy = scale * dy;
    }
    (gx, gy)
}

#[cube]
fn safe_inverse_2d<F: Float>(m00: F, m01: F, m11: F) -> (F, F, F, F) {
    let determinant = m00 * m11 - m01 * m01;
    let mut i00 = F::new(1.0_f32);
    let mut i01 = F::new(0.0_f32);
    let mut i10 = F::new(0.0_f32);
    let mut i11 = F::new(1.0_f32);
    if determinant.abs() >= F::new(1.0e-3_f32) {
        let reciprocal = F::new(1.0_f32) / determinant;
        i00 = m11 * reciprocal;
        i01 = (F::new(0.0_f32) - m01) * reciprocal;
        i10 = i01;
        i11 = m00 * reciprocal;
    }
    (i00, i01, i10, i11)
}

#[cube]
fn perception_inverse_2d<F: Float>(m00: F, m01: F, m11: F, args: &AdaptiveArgs) -> (F, F, F, F) {
    let (mut i00, mut i01, mut i10, mut i11) = safe_inverse_2d::<F>(m00, m01, m11);
    if normalized_semantics(args) {
        let trace = m00.abs() + m11.abs();
        let diagonal = args.moment_regularization.get::<F>()
            * (trace / F::new(2.0_f32)).max(F::new(1.0e-8_f32));
        let a = if m00 < F::new(0.0_f32) {
            m00 - diagonal
        } else {
            m00 + diagonal
        };
        let d = if m11 < F::new(0.0_f32) {
            m11 - diagonal
        } else {
            m11 + diagonal
        };
        let scale = F::new(1.0_f32) / (trace / F::new(2.0_f32)).max(F::new(1.0e-6_f32));
        i00 = scale;
        i01 = F::new(0.0_f32);
        i10 = F::new(0.0_f32);
        i11 = scale;

        let determinant = a * d - m01 * m01;
        if determinant.abs() >= F::new(1.0e-12_f32) {
            let reciprocal = F::new(1.0_f32) / determinant;
            let candidate_i00 = d * reciprocal;
            let candidate_i01 = (F::new(0.0_f32) - m01) * reciprocal;
            let candidate_i10 = candidate_i01;
            let candidate_i11 = a * reciprocal;
            let matrix_norm = (a * a + F::new(2.0_f32) * m01 * m01 + d * d).sqrt();
            let inverse_norm = (candidate_i00 * candidate_i00
                + candidate_i01 * candidate_i01
                + candidate_i10 * candidate_i10
                + candidate_i11 * candidate_i11)
                .sqrt();
            if matrix_norm * inverse_norm <= args.moment_condition_limit.get::<F>() {
                i00 = candidate_i00;
                i01 = candidate_i01;
                i10 = candidate_i10;
                i11 = candidate_i11;
            }
        }
    }
    (i00, i01, i10, i11)
}

#[cube]
fn log_normalize_2<F: Float>(x: F, y: F) -> (F, F) {
    let norm = (x * x + y * y + F::new(1.0e-12_f32))
        .sqrt()
        .max(F::new(1.0e-6_f32));
    let scale = norm.log1p() / norm;
    (x * scale, y * scale)
}

#[cube]
fn log_normalize_adjoint_2<F: Float>(x: F, y: F, adj_x: F, adj_y: F) -> (F, F) {
    let norm = (x * x + y * y + F::new(1.0e-12_f32))
        .sqrt()
        .max(F::new(1.0e-6_f32));
    let log = norm.log1p();
    let scale = log / norm;
    let dscale = (norm / (F::new(1.0_f32) + norm) - log) / (norm * norm);
    let radial = dscale * (x * adj_x + y * adj_y) / norm;
    (scale * adj_x + radial * x, scale * adj_y + radial * y)
}

#[cube]
fn reciprocal_finite<F: Float>(value: F) -> F {
    let mut output = F::new(0.0_f32);
    if value.abs() > F::new(1.175_494_4e-38_f32) {
        output = F::new(1.0_f32) / value;
    }
    output
}

#[cube]
fn position_value<F: Float>(tensor: &Tensor<F>, batch: usize, particle: usize, axis: usize) -> F {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1) + axis * tensor.stride(2)]
}

#[cube]
fn state_value<F: Float>(tensor: &Tensor<F>, batch: usize, particle: usize, channel: usize) -> F {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1) + channel * tensor.stride(2)]
}

#[cube]
fn material_value<F: Float>(tensor: &Tensor<F>, batch: usize, particle: usize) -> F {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1)]
}

#[cube]
fn measure_value<F: Float>(tensor: &Tensor<F>, batch: usize, particle: usize) -> F {
    material_value::<F>(tensor, batch, particle)
}

#[cube]
fn feature_value<F: Float>(tensor: &Tensor<F>, batch: usize, particle: usize, feature: usize) -> F {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1) + feature * tensor.stride(2)]
}

#[cube]
fn write_material<F: Float>(tensor: &mut Tensor<F>, batch: usize, particle: usize, value: F) {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1)] = value;
}

#[cube]
fn write_feature<F: Float>(
    tensor: &mut Tensor<F>,
    batch: usize,
    particle: usize,
    feature: usize,
    value: F,
) {
    tensor[batch * tensor.stride(0) + particle * tensor.stride(1) + feature * tensor.stride(2)] =
        value;
}
