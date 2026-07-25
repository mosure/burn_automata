use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use cubecl::{CubeDim, CubeLaunch, calculate_cube_count_elemwise, cube, prelude::*};

pub(super) const GRID_WIDTH: u32 = 32;
pub(super) const GRID_HEIGHT: u32 = 32;

#[derive(Clone)]
pub(super) struct GridRaw<R: CubeRuntime> {
    pub(super) offsets: CubeTensor<R>,
    pub(super) permutation: CubeTensor<R>,
}

pub(super) fn launch<R: CubeRuntime>(positions: CubeTensor<R>, cell_size: f32) -> GridRaw<R> {
    let [batches, particles, _] = positions.shape().dims::<3>();
    let cells = GRID_WIDTH as usize * GRID_HEIGHT as usize;
    let client = positions.client.clone();
    let device = positions.device.clone();
    let dtype = positions.dtype;
    let counts = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, cells]),
        DType::U32,
    );
    let offsets = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, cells + 1]),
        DType::U32,
    );
    let permutation = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([batches, particles]),
        DType::U32,
    );
    let cell_units = batches * cells;
    let cell_cube_dim = CubeDim::new(&client, cell_units);
    grid_zero_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, cell_units, cell_cube_dim),
        cell_cube_dim,
        AddressType::U32,
        counts.clone().into_tensor_arg(),
    );

    let particle_units = batches * particles;
    let particle_cube_dim = CubeDim::new(&client, particle_units);
    let particle_cube_count =
        calculate_cube_count_elemwise(&client, particle_units, particle_cube_dim);
    grid_count_kernel::launch(
        &client,
        particle_cube_count.clone(),
        particle_cube_dim,
        AddressType::U32,
        positions.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        GridArgsLaunch::new(InputScalar::new(cell_size, dtype), GRID_WIDTH, GRID_HEIGHT),
        dtype.into(),
    );

    let batch_cube_dim = CubeDim::new(&client, batches);
    grid_scan_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, batches, batch_cube_dim),
        batch_cube_dim,
        AddressType::U32,
        counts.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
    );
    grid_scatter_kernel::launch(
        &client,
        particle_cube_count,
        particle_cube_dim,
        AddressType::U32,
        positions.into_tensor_arg(),
        counts.into_tensor_arg(),
        permutation.clone().into_tensor_arg(),
        GridArgsLaunch::new(InputScalar::new(cell_size, dtype), GRID_WIDTH, GRID_HEIGHT),
        dtype.into(),
    );

    GridRaw {
        offsets,
        permutation,
    }
}

#[derive(Clone, CubeLaunch, CubeType)]
struct GridArgs {
    cell_size: InputScalar,
    width: u32,
    height: u32,
}

#[cube]
fn grid_cell<F: Float>(x: F, y: F, args: &GridArgs) -> (i32, i32) {
    let width = args.width as i32;
    let height = args.height as i32;
    let cell_size = args.cell_size.get::<F>();
    let mut cell_x = i32::cast_from((x / cell_size).floor()) + width / 2i32;
    let mut cell_y = i32::cast_from((y / cell_size).floor()) + height / 2i32;
    cell_x = clamp(cell_x, 0i32, width - 1i32);
    cell_y = clamp(cell_y, 0i32, height - 1i32);
    (cell_x, cell_y)
}

#[cube]
fn position_value<F: Float>(
    positions: &Tensor<F>,
    batch: usize,
    particle: usize,
    axis: usize,
) -> F {
    positions
        [batch * positions.stride(0) + particle * positions.stride(1) + axis * positions.stride(2)]
}

#[cube(launch, address_type = "dynamic")]
fn grid_zero_kernel(counts: &mut Tensor<Atomic<u32>>) {
    let index = ABSOLUTE_POS;
    if index >= counts.len() {
        terminate!();
    }
    counts[index].store(0u32);
}

#[cube(launch, address_type = "dynamic")]
fn grid_count_kernel<F: Float>(
    positions: &Tensor<F>,
    counts: &mut Tensor<Atomic<u32>>,
    args: &GridArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particles = positions.shape(1);
    if index >= positions.shape(0) * particles {
        terminate!();
    }
    let batch = index / particles;
    let particle = index - batch * particles;
    let (cell_x, cell_y) = grid_cell::<F>(
        position_value::<F>(positions, batch, particle, 0usize),
        position_value::<F>(positions, batch, particle, 1usize),
        args,
    );
    let cell = cell_y as usize * args.width as usize + cell_x as usize;
    counts[batch * counts.stride(0) + cell * counts.stride(1)].fetch_add(1u32);
}

#[cube(launch, address_type = "dynamic")]
fn grid_scan_kernel(counts: &mut Tensor<Atomic<u32>>, offsets: &mut Tensor<u32>) {
    let batch = ABSOLUTE_POS;
    if batch >= counts.shape(0) {
        terminate!();
    }
    let cells = counts.shape(1);
    let mut sum = 0u32;
    offsets[batch * offsets.stride(0)] = 0u32;
    let mut cell = 0usize;
    while cell < cells {
        let count_index = batch * counts.stride(0) + cell * counts.stride(1);
        let count = counts[count_index].load();
        counts[count_index].store(sum);
        sum += count;
        offsets[batch * offsets.stride(0) + (cell + 1usize) * offsets.stride(1)] = sum;
        cell += 1usize;
    }
}

#[cube(launch, address_type = "dynamic")]
fn grid_scatter_kernel<F: Float>(
    positions: &Tensor<F>,
    cursors: &mut Tensor<Atomic<u32>>,
    permutation: &mut Tensor<u32>,
    args: &GridArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particles = positions.shape(1);
    if index >= positions.shape(0) * particles {
        terminate!();
    }
    let batch = index / particles;
    let particle = index - batch * particles;
    let (cell_x, cell_y) = grid_cell::<F>(
        position_value::<F>(positions, batch, particle, 0usize),
        position_value::<F>(positions, batch, particle, 1usize),
        args,
    );
    let cell = cell_y as usize * args.width as usize + cell_x as usize;
    let slot =
        cursors[batch * cursors.stride(0) + cell * cursors.stride(1)].fetch_add(1u32) as usize;
    permutation[batch * permutation.stride(0) + slot * permutation.stride(1)] = particle as u32;
}
