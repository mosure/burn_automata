//! Direct WGPU inference kernels for Neural Particle Automata.
//!
//! The current path builds a linked-cell hashgrid with GPU atomics, then keeps
//! density, SPH perception, MLP update, and Euler integration on WGPU storage
//! buffers. The convenience API reads outputs back for parity tests and CLI
//! reporting.

use std::borrow::Cow;
use std::sync::mpsc;

use burn_automata_kernels::{Boundary, HashGridConfig, HashGridMode, build_hashgrid};
use wgpu::util::DeviceExt;

use crate::{AutomataError, AutomataResult, NpaModel};

const WORKGROUP_SIZE: u32 = 128;
const PARAM_COUNT: usize = 36;
const MAX_STATE_DIMS: usize = 24;
const MAX_HIDDEN_DIMS: usize = 256;
const MAX_FEATURE_DIMS: usize = 128;
const MAX_OUTPUT_DIMS: usize = 32;
pub const GAUSSIAN_SH_COEFF_COUNT: usize = 48;

const PARAM_TOTAL: usize = 0;
const PARAM_PARTICLE_COUNT: usize = 1;
const PARAM_STATE_DIMS: usize = 2;
const PARAM_HIDDEN_DIMS: usize = 3;
const PARAM_SPATIAL_DIMS: usize = 4;
const PARAM_FEATURE_DIMS: usize = 5;
const PARAM_OUTPUT_DIMS: usize = 6;
const PARAM_GRID_X: usize = 7;
const PARAM_GRID_Y: usize = 8;
const PARAM_GRID_Z: usize = 9;
const PARAM_CELL_COUNT: usize = 10;
const PARAM_PERIODIC: usize = 11;
const PARAM_LOG_GRAD: usize = 12;
const PARAM_LOG_DENSITY_GRAD: usize = 13;
const PARAM_POSITION_FEATURES: usize = 14;
const PARAM_EPS: usize = 15;
const PARAM_ALPHA: usize = 17;
const PARAM_DT: usize = 18;
const PARAM_SMOOTH_COEF: usize = 19;
const PARAM_SPIKY_COEF: usize = 20;
const PARAM_DENSITY_SCALE: usize = 21;
const PARAM_GRAD_SCALE: usize = 22;
const PARAM_BUCKET_CAPACITY: usize = 23;
const PARAM_MOTION_EPS: usize = 24;
const PARAM_UPDATE_PROB: usize = 25;
const PARAM_STEP_INDEX: usize = 26;
const PARAM_RANDOM_SEED: usize = 27;
const PARAM_PARTICLE_GRID: usize = 28;
const PARAM_NEIGHBOR_LAYOUT: usize = 29;
const PARAM_BVH_BUILD_LEVEL: usize = 30;
const PARAM_BVH_LEAF_COUNT: usize = 31;
const PARAM_BVH_SORT_COUNT: usize = 32;
const PARAM_BVH_SORT_K: usize = 33;
const PARAM_BVH_SORT_J: usize = 34;
const BVH_HEADER_U32: usize = 4;
const BVH_NODE_U32: usize = 9;
const DEFAULT_MAX_STORAGE_BUFFER_BINDING_BYTES: usize = 128 * 1024 * 1024;
const DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32: usize =
    DEFAULT_MAX_STORAGE_BUFFER_BINDING_BYTES / std::mem::size_of::<u32>();

#[derive(Clone, Debug)]
pub struct WgpuStepOutput {
    pub next_positions: Vec<[f32; 4]>,
    pub next_states: Vec<f32>,
    pub density: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct WgpuGaussianReadback {
    pub position_visibility: Vec<f32>,
    pub spherical_harmonic: Vec<f32>,
    pub rotation: Vec<f32>,
    pub scale_opacity: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WgpuNeighborMode {
    LinkedList,
    FixedCellBuckets {
        capacity: usize,
    },
    TiledFixedCellBuckets {
        capacity: usize,
    },
    SortedCells,
    Bvh {
        leaf_size: usize,
    },
    GpuBvh {
        leaf_size: usize,
    },
    GpuLbvh {
        leaf_size: usize,
    },
    GpuMortonLbvh {
        leaf_size: usize,
    },
    #[default]
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuNeighborReport {
    pub mode: WgpuNeighborMode,
    pub bucket_capacity: usize,
    pub grid_storage_len: usize,
    pub grid_clear_len: usize,
}

pub struct WgpuGaussianBufferRefs<'a> {
    pub position_visibility: &'a wgpu::Buffer,
    pub spherical_harmonic: &'a wgpu::Buffer,
    pub rotation: &'a wgpu::Buffer,
    pub scale_opacity: &'a wgpu::Buffer,
}

pub struct WgpuOwnedGaussianBuffers {
    pub position_visibility: wgpu::Buffer,
    pub spherical_harmonic: wgpu::Buffer,
    pub rotation: wgpu::Buffer,
    pub scale_opacity: wgpu::Buffer,
    pub count: usize,
}

impl WgpuOwnedGaussianBuffers {
    pub fn refs(&self) -> WgpuGaussianBufferRefs<'_> {
        WgpuGaussianBufferRefs {
            position_visibility: &self.position_visibility,
            spherical_harmonic: &self.spherical_harmonic,
            rotation: &self.rotation,
            scale_opacity: &self.scale_opacity,
        }
    }
}

pub struct WgpuGaussianBindGroup {
    bind_group: wgpu::BindGroup,
    count: usize,
}

pub struct WgpuAutomataState {
    pub total: usize,
    pub particle_count: usize,
    pub batch_size: usize,
    spatial_dims: usize,
    bvh_leaf_count: usize,
    bvh_levels: usize,
    bvh_sort_count: usize,
    position_f32_len: usize,
    state_f32_len: usize,
    grid_storage_len: usize,
    grid_clear_len: usize,
    cell_count: usize,
    bucket_capacity: usize,
    neighbor_mode: WgpuNeighborMode,
    weights_f32_len: usize,
    current: usize,
    step_index: u32,
    params_buffer: wgpu::Buffer,
    positions_buffers: [wgpu::Buffer; 2],
    states_buffers: [wgpu::Buffer; 2],
    weights_buffer: wgpu::Buffer,
    linked_grid_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    density_buffer: wgpu::Buffer,
    grid_bind_groups: [wgpu::BindGroup; 2],
    step_bind_groups: [wgpu::BindGroup; 2],
    gaussian_source_bind_groups: [wgpu::BindGroup; 2],
    step_index_copy_buffers: Vec<wgpu::Buffer>,
}

pub struct WgpuAutomataExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    grid_bind_group_layout: wgpu::BindGroupLayout,
    gaussian_source_bind_group_layout: wgpu::BindGroupLayout,
    gaussian_bind_group_layout: wgpu::BindGroupLayout,
    clear_pipeline: wgpu::ComputePipeline,
    bin_pipeline: wgpu::ComputePipeline,
    scan_counts_pipeline: wgpu::ComputePipeline,
    scan_block_sums_pipeline: wgpu::ComputePipeline,
    add_block_offsets_pipeline: wgpu::ComputePipeline,
    scatter_sorted_pipeline: wgpu::ComputePipeline,
    bvh_init_pipeline: wgpu::ComputePipeline,
    bvh_reduce_pipeline: wgpu::ComputePipeline,
    morton_sort_init_pipeline: wgpu::ComputePipeline,
    morton_sort_step_pipeline: wgpu::ComputePipeline,
    density_pipeline: wgpu::ComputePipeline,
    tiled_density_pipeline: wgpu::ComputePipeline,
    bvh_density_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    tiled_update_pipeline: wgpu::ComputePipeline,
    bvh_update_pipeline: wgpu::ComputePipeline,
    gaussian_pipeline: wgpu::ComputePipeline,
}

mod executor;

#[allow(clippy::too_many_arguments)]
pub fn step_wgpu_blocking(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
) -> AutomataResult<WgpuStepOutput> {
    pollster::block_on(step_wgpu(
        model,
        positions,
        states,
        batch_size,
        particle_count,
        grid,
        dt,
    ))
}

#[allow(clippy::too_many_arguments)]
pub async fn step_wgpu(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
) -> AutomataResult<WgpuStepOutput> {
    let executor = WgpuAutomataExecutor::new().await?;
    executor.step(
        model,
        positions,
        states,
        batch_size,
        particle_count,
        grid,
        dt,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_gpu_step(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
    update_prob: f32,
) -> AutomataResult<()> {
    validate_gpu_model_config(model, grid)?;
    if batch_size != 1 {
        return Err(AutomataError::InvalidArgument(
            "WGPU step currently supports batch_size=1; hashgrid GPU batching is a follow-up"
                .to_owned(),
        ));
    }
    if particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "particle_count must be greater than zero".to_owned(),
        ));
    }
    if positions.len() != batch_size * particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "positions len {} does not match batch_size * particle_count {}",
            positions.len(),
            batch_size * particle_count
        )));
    }
    if states.len() != positions.len() * model.config.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "states len {} does not match positions * state_dims {}",
            states.len(),
            positions.len() * model.config.state_dims
        )));
    }
    if !dt.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "dt must be finite, got {dt}"
        )));
    }
    if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "update_prob must be finite and in [0, 1], got {update_prob}"
        )));
    }
    u32_checked(positions.len(), "positions len")?;
    u32_checked(particle_count, "particle_count")?;
    model.weights.validate(&model.config)
}

fn validate_gpu_model_config(model: &NpaModel, grid: &HashGridConfig) -> AutomataResult<()> {
    model.validate()?;
    grid.validate()?;
    if grid.dim != model.config.spatial_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "grid dim {} does not match model spatial dims {}",
            grid.dim, model.config.spatial_dims
        )));
    }
    if !model.config.state_grad || !model.config.density_grad {
        return Err(AutomataError::InvalidArgument(
            "WGPU step currently expects state_grad=true and density_grad=true".to_owned(),
        ));
    }
    if model.config.state_dims > MAX_STATE_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "state_dims {} exceeds WGPU shader max {MAX_STATE_DIMS}",
            model.config.state_dims
        )));
    }
    if model.config.hidden_dims > MAX_HIDDEN_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "hidden_dims {} exceeds WGPU shader max {MAX_HIDDEN_DIMS}",
            model.config.hidden_dims
        )));
    }
    if model.config.perception_dims() > MAX_FEATURE_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "perception_dims {} exceeds WGPU shader max {MAX_FEATURE_DIMS}",
            model.config.perception_dims()
        )));
    }
    if model.config.update_dims() > MAX_OUTPUT_DIMS {
        return Err(AutomataError::InvalidArgument(format!(
            "update_dims {} exceeds WGPU shader max {MAX_OUTPUT_DIMS}",
            model.config.update_dims()
        )));
    }
    u32_checked(grid.cell_count(), "cell_count")?;
    model.weights.validate(&model.config)
}

#[allow(clippy::too_many_arguments)]
fn gpu_params(
    model: &NpaModel,
    total: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
    bucket_capacity: usize,
    neighbor_mode: WgpuNeighborMode,
    update_prob: f32,
    seed: u64,
) -> AutomataResult<[u32; PARAM_COUNT]> {
    let mut params = [0; PARAM_COUNT];
    params[PARAM_TOTAL] = u32_checked(total, "total")?;
    params[PARAM_PARTICLE_COUNT] = u32_checked(particle_count, "particle_count")?;
    params[PARAM_STATE_DIMS] = u32_checked(model.config.state_dims, "state_dims")?;
    params[PARAM_HIDDEN_DIMS] = u32_checked(model.config.hidden_dims, "hidden_dims")?;
    params[PARAM_SPATIAL_DIMS] = u32_checked(model.config.spatial_dims, "spatial_dims")?;
    params[PARAM_FEATURE_DIMS] = u32_checked(model.config.perception_dims(), "feature_dims")?;
    params[PARAM_OUTPUT_DIMS] = u32_checked(model.config.update_dims(), "output_dims")?;
    params[PARAM_GRID_X] = u32_checked(grid.grid_size[0], "grid_size[0]")?;
    params[PARAM_GRID_Y] = u32_checked(grid.grid_size[1], "grid_size[1]")?;
    params[PARAM_GRID_Z] = u32_checked(grid.grid_size[2], "grid_size[2]")?;
    params[PARAM_CELL_COUNT] = u32_checked(grid.cell_count(), "cell_count")?;
    params[PARAM_PERIODIC] = u32::from(grid.boundary == Boundary::Periodic);
    params[PARAM_LOG_GRAD] = u32::from(model.config.log_norm_grad);
    params[PARAM_LOG_DENSITY_GRAD] = u32::from(model.config.log_norm_density_grad);
    params[PARAM_POSITION_FEATURES] = u32::from(model.config.position_features);
    params[PARAM_EPS] = grid.eps.to_bits();
    params[PARAM_ALPHA] = model.config.alpha.to_bits();
    params[PARAM_DT] = dt.to_bits();
    params[PARAM_SMOOTH_COEF] = smoothing_poly6_normalization(grid).to_bits();
    params[PARAM_SPIKY_COEF] = gradient_spiky_normalization(grid).to_bits();
    params[PARAM_DENSITY_SCALE] = density_gradient_scale(model, grid, particle_count).to_bits();
    params[PARAM_GRAD_SCALE] = state_gradient_scale(model, grid).to_bits();
    params[PARAM_BUCKET_CAPACITY] = u32_checked(bucket_capacity, "bucket_capacity")?;
    params[PARAM_MOTION_EPS] = model.config.motion_eps(grid.eps).to_bits();
    params[PARAM_UPDATE_PROB] = update_prob.to_bits();
    params[PARAM_STEP_INDEX] = 0;
    params[PARAM_RANDOM_SEED] = (seed as u32) ^ ((seed >> 32) as u32);
    params[PARAM_PARTICLE_GRID] = u32::from(grid.mode == HashGridMode::Particle);
    params[PARAM_NEIGHBOR_LAYOUT] = neighbor_layout_code(neighbor_mode);
    Ok(params)
}

fn neighbor_layout_code(mode: WgpuNeighborMode) -> u32 {
    match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { .. } => 1,
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => 1,
        WgpuNeighborMode::SortedCells => 2,
        WgpuNeighborMode::Bvh { .. } => 3,
        WgpuNeighborMode::GpuBvh { .. } => 3,
        WgpuNeighborMode::GpuLbvh { .. } => 4,
        WgpuNeighborMode::GpuMortonLbvh { .. } => 5,
        WgpuNeighborMode::Auto => 0,
    }
}

fn is_bvh_neighbor_mode(mode: WgpuNeighborMode) -> bool {
    matches!(
        mode,
        WgpuNeighborMode::Bvh { .. }
            | WgpuNeighborMode::GpuBvh { .. }
            | WgpuNeighborMode::GpuLbvh { .. }
            | WgpuNeighborMode::GpuMortonLbvh { .. }
    )
}

fn resolve_bucket_capacity(
    grid: &HashGridConfig,
    particle_count: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    let capacity = match mode {
        WgpuNeighborMode::LinkedList => 0,
        WgpuNeighborMode::FixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::TiledFixedCellBuckets { capacity } => capacity,
        WgpuNeighborMode::Bvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "BVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "BVH WGPU mode currently supports clamped/non-periodic grids only".to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::GpuBvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU BVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU BVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::GpuLbvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU LBVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU LBVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            scan_block_count(grid.cell_count())?;
            leaf_size
        }
        WgpuNeighborMode::GpuMortonLbvh { leaf_size } => {
            if leaf_size == 0 {
                return Err(AutomataError::InvalidArgument(
                    "GPU Morton LBVH leaf_size must be greater than zero".to_owned(),
                ));
            }
            if grid.boundary == Boundary::Periodic {
                return Err(AutomataError::InvalidArgument(
                    "GPU Morton LBVH WGPU mode currently supports clamped/non-periodic grids only"
                        .to_owned(),
                ));
            }
            leaf_size
        }
        WgpuNeighborMode::SortedCells => 0,
        WgpuNeighborMode::Auto => {
            if grid.mode == HashGridMode::Particle {
                return Ok(0);
            }
            if grid.dim == 2 && grid.boundary == Boundary::Periodic {
                return Ok(0);
            }
            if grid.dim == 2 && particle_count <= grid.cell_count().saturating_mul(4) {
                return Ok(0);
            }
            if grid.dim == 2 {
                return Ok(particle_count);
            }
            if grid.dim == 3 && particle_count <= grid.cell_count() {
                return Ok(0);
            }
            let average = particle_count.div_ceil(grid.cell_count().max(1));
            let multiplier = match (grid.dim, grid.boundary) {
                (2, _) => 32,
                _ => 8,
            };
            average
                .saturating_mul(multiplier)
                .max(grid.max_particles_per_block)
                .min(particle_count)
        }
    };
    u32_checked(capacity, "bucket_capacity")?;
    let resolved = resolved_neighbor_mode(mode, capacity);
    let storage_len =
        grid_storage_len_for_mode(grid.cell_count(), particle_count, capacity, resolved)?;
    ensure_grid_storage_binding_limit(storage_len, resolved)?;
    Ok(capacity)
}

fn resolve_neighbor_mode_for_state(
    grid: &HashGridConfig,
    particle_count: usize,
    positions: &[[f32; 4]],
    requested: WgpuNeighborMode,
) -> AutomataResult<(usize, WgpuNeighborMode)> {
    if requested != WgpuNeighborMode::Auto {
        let capacity = resolve_bucket_capacity(grid, particle_count, requested)?;
        return Ok((capacity, resolved_neighbor_mode(requested, capacity)));
    }

    let (nonempty_cells, max_occupancy) =
        initial_cell_occupancy_stats(grid, particle_count, positions)?;

    if grid.dim == 2 {
        if let Some(capacity) = adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
        {
            return Ok((
                capacity,
                WgpuNeighborMode::TiledFixedCellBuckets { capacity },
            ));
        }
        return exact_neighbor_fallback_mode(grid, particle_count);
    }

    if grid.mode == HashGridMode::Particle {
        if grid.dim == 3 && particle_count <= 2048 && max_occupancy >= 96 {
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if should_use_tiled_particle_grid(grid, particle_count, nonempty_cells, max_occupancy) {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                return Ok((
                    capacity,
                    WgpuNeighborMode::TiledFixedCellBuckets { capacity },
                ));
            }
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        if max_occupancy >= 96 {
            if let Some(capacity) =
                adaptive_fixed_bucket_capacity(grid, max_occupancy, particle_count)?
            {
                return Ok((capacity, WgpuNeighborMode::FixedCellBuckets { capacity }));
            }
            return exact_neighbor_fallback_mode(grid, particle_count);
        }
        return Ok((0, WgpuNeighborMode::LinkedList));
    }

    let capacity = resolve_bucket_capacity(grid, particle_count, requested)?;
    Ok((capacity, resolved_neighbor_mode(requested, capacity)))
}

fn initial_cell_occupancy_stats(
    grid: &HashGridConfig,
    particle_count: usize,
    positions: &[[f32; 4]],
) -> AutomataResult<(usize, usize)> {
    let snapshot = build_hashgrid(positions, 1, particle_count, grid)?;
    let mut nonempty_cells = 0usize;
    let max_occupancy = snapshot
        .bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .inspect(|occupancy| {
            if *occupancy > 0 {
                nonempty_cells += 1;
            }
        })
        .max()
        .unwrap_or(0);
    Ok((nonempty_cells, max_occupancy))
}

fn should_use_tiled_particle_grid(
    grid: &HashGridConfig,
    particle_count: usize,
    nonempty_cells: usize,
    max_occupancy: usize,
) -> bool {
    if grid.dim != 3 || max_occupancy < 64 {
        return false;
    }
    nonempty_cells.saturating_mul(32) <= particle_count.max(1)
}

fn adaptive_tiled_bucket_capacity(
    max_occupancy: usize,
    particle_count: usize,
) -> AutomataResult<usize> {
    let with_headroom = max_occupancy
        .saturating_mul(2)
        .saturating_add(64)
        .max(max_occupancy.saturating_add(16));
    let capacity = with_headroom.next_power_of_two().min(particle_count.max(1));
    u32_checked(capacity, "adaptive tiled bucket_capacity")?;
    Ok(capacity)
}

fn adaptive_fixed_bucket_capacity(
    grid: &HashGridConfig,
    max_occupancy: usize,
    particle_count: usize,
) -> AutomataResult<Option<usize>> {
    let target = adaptive_tiled_bucket_capacity(max_occupancy, particle_count)?;
    let Some(max_safe_capacity) =
        max_fixed_bucket_capacity_for_binding_limit(grid.cell_count(), particle_count)?
    else {
        return Ok(None);
    };
    if target <= max_safe_capacity {
        return Ok(Some(target));
    }
    if max_safe_capacity < max_occupancy {
        return Ok(None);
    }

    let reduced_power_of_two = previous_power_of_two(max_safe_capacity);
    let reduced = if reduced_power_of_two >= max_occupancy {
        reduced_power_of_two
    } else {
        max_safe_capacity
    };
    u32_checked(reduced, "adaptive reduced bucket_capacity")?;
    Ok(Some(reduced))
}

fn max_fixed_bucket_capacity_for_binding_limit(
    cell_count: usize,
    particle_count: usize,
) -> AutomataResult<Option<usize>> {
    if cell_count == 0 {
        return Ok(None);
    }
    let overhead = cell_count
        .checked_add(1)
        .and_then(|value| {
            value.checked_add(active_grid_storage_len(cell_count, particle_count).ok()?)
        })
        .ok_or_else(|| {
            AutomataError::InvalidArgument("fixed bucket storage overhead overflow".to_owned())
        })?;
    if overhead >= DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32 {
        return Ok(None);
    }
    Ok(Some(
        (DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32 - overhead) / cell_count,
    ))
}

fn exact_neighbor_fallback_mode(
    grid: &HashGridConfig,
    particle_count: usize,
) -> AutomataResult<(usize, WgpuNeighborMode)> {
    for mode in [WgpuNeighborMode::SortedCells, WgpuNeighborMode::LinkedList] {
        let storage_len = grid_storage_len_for_mode(grid.cell_count(), particle_count, 0, mode)?;
        if grid_storage_binding_len_fits(storage_len) {
            return Ok((0, mode));
        }
    }
    Err(AutomataError::InvalidArgument(format!(
        "no exact WGPU neighbor layout fits storage buffer binding limit for {} cells and {} particles",
        grid.cell_count(),
        particle_count
    )))
}

fn previous_power_of_two(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

fn resolved_neighbor_mode(requested: WgpuNeighborMode, bucket_capacity: usize) -> WgpuNeighborMode {
    match requested {
        WgpuNeighborMode::Auto if bucket_capacity == 0 => WgpuNeighborMode::LinkedList,
        WgpuNeighborMode::Auto => WgpuNeighborMode::FixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::FixedCellBuckets { .. } => WgpuNeighborMode::FixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => WgpuNeighborMode::TiledFixedCellBuckets {
            capacity: bucket_capacity,
        },
        WgpuNeighborMode::Bvh { .. } => WgpuNeighborMode::Bvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuBvh { .. } => WgpuNeighborMode::GpuBvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuLbvh { .. } => WgpuNeighborMode::GpuLbvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::GpuMortonLbvh { .. } => WgpuNeighborMode::GpuMortonLbvh {
            leaf_size: bucket_capacity,
        },
        WgpuNeighborMode::SortedCells => WgpuNeighborMode::SortedCells,
        WgpuNeighborMode::LinkedList => WgpuNeighborMode::LinkedList,
    }
}

fn grid_storage_len_for_mode(
    cell_count: usize,
    particle_count: usize,
    bucket_capacity: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    if matches!(mode, WgpuNeighborMode::SortedCells) {
        return sorted_grid_storage_len(cell_count, particle_count);
    }
    if matches!(mode, WgpuNeighborMode::GpuLbvh { .. }) {
        return sorted_grid_storage_len(cell_count, particle_count)?
            .checked_add(bvh_grid_storage_len(particle_count, bucket_capacity)?)
            .ok_or_else(|| {
                AutomataError::InvalidArgument("GPU LBVH storage length overflow".to_owned())
            });
    }
    if matches!(mode, WgpuNeighborMode::GpuMortonLbvh { .. }) {
        return morton_bvh_grid_storage_len(particle_count, bucket_capacity);
    }
    if is_bvh_neighbor_mode(mode) {
        return bvh_grid_storage_len(particle_count, bucket_capacity);
    }
    grid_storage_payload_len(cell_count, particle_count, bucket_capacity)?
        .checked_add(active_grid_storage_len(cell_count, particle_count)?)
        .ok_or_else(|| AutomataError::InvalidArgument("grid storage length overflow".to_owned()))
}

fn ensure_grid_storage_binding_limit(
    storage_len: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<()> {
    if grid_storage_binding_len_fits(storage_len) {
        return Ok(());
    }
    Err(AutomataError::InvalidArgument(format!(
        "WGPU neighbor storage for {mode:?} requires {} u32 values ({:.3} MiB), exceeding the conservative {} MiB storage buffer binding limit",
        storage_len,
        storage_len as f64 * std::mem::size_of::<u32>() as f64 / (1024.0 * 1024.0),
        DEFAULT_MAX_STORAGE_BUFFER_BINDING_BYTES / (1024 * 1024)
    )))
}

fn grid_storage_binding_len_fits(storage_len: usize) -> bool {
    storage_len <= DEFAULT_MAX_STORAGE_BUFFER_BINDING_U32
}

fn grid_storage_payload_len(
    cell_count: usize,
    particle_count: usize,
    bucket_capacity: usize,
) -> AutomataResult<usize> {
    if bucket_capacity == 0 {
        cell_count.checked_add(particle_count).ok_or_else(|| {
            AutomataError::InvalidArgument("grid storage length overflow".to_owned())
        })
    } else {
        cell_count
            .checked_mul(bucket_capacity)
            .and_then(|slots| slots.checked_add(cell_count))
            .and_then(|with_counts| with_counts.checked_add(1))
            .ok_or_else(|| {
                AutomataError::InvalidArgument("grid storage length overflow".to_owned())
            })
    }
}

fn grid_clear_len(cell_count: usize, bucket_capacity: usize) -> AutomataResult<usize> {
    if bucket_capacity == 0 {
        Ok(cell_count)
    } else {
        cell_count
            .checked_add(1)
            .ok_or_else(|| AutomataError::InvalidArgument("grid clear length overflow".to_owned()))
    }
}

fn grid_clear_len_for_mode(
    cell_count: usize,
    bucket_capacity: usize,
    mode: WgpuNeighborMode,
) -> AutomataResult<usize> {
    if is_bvh_neighbor_mode(mode) {
        if matches!(mode, WgpuNeighborMode::GpuLbvh { .. }) {
            return Ok(cell_count);
        }
        return Ok(0);
    }
    grid_clear_len(cell_count, bucket_capacity)
}

fn active_grid_storage_len(cell_count: usize, particle_count: usize) -> AutomataResult<usize> {
    Ok(cell_count.min(particle_count))
}

fn sorted_grid_storage_len(cell_count: usize, particle_count: usize) -> AutomataResult<usize> {
    let block_count = scan_block_count(cell_count)?;
    cell_count
        .checked_add(cell_count.checked_add(1).ok_or_else(|| {
            AutomataError::InvalidArgument("sorted offsets length overflow".to_owned())
        })?)
        .and_then(|with_offsets| with_offsets.checked_add(particle_count))
        .and_then(|with_particles| with_particles.checked_add(block_count))
        .ok_or_else(|| {
            AutomataError::InvalidArgument("sorted grid storage length overflow".to_owned())
        })
}

fn bvh_grid_storage_len(particle_count: usize, leaf_size: usize) -> AutomataResult<usize> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    let leaf_count = bvh_leaf_count_pow2(particle_count, leaf_size)?;
    let gpu_node_count = leaf_count
        .checked_mul(2)
        .and_then(|nodes| nodes.checked_sub(1));
    let cpu_node_count = particle_count.saturating_mul(2).saturating_sub(1);
    let node_count = gpu_node_count
        .map(|count| count.max(cpu_node_count))
        .ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?;
    BVH_HEADER_U32
        .checked_add(node_count.checked_mul(BVH_NODE_U32).ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?)
        .and_then(|with_nodes| with_nodes.checked_add(particle_count))
        .ok_or_else(|| AutomataError::InvalidArgument("BVH storage length overflow".to_owned()))
}

fn morton_bvh_grid_storage_len(particle_count: usize, leaf_size: usize) -> AutomataResult<usize> {
    bvh_sort_count_pow2(particle_count)?
        .checked_mul(2)
        .and_then(|sort_storage| {
            sort_storage.checked_add(bvh_grid_storage_len(particle_count, leaf_size).ok()?)
        })
        .ok_or_else(|| {
            AutomataError::InvalidArgument("GPU Morton LBVH storage length overflow".to_owned())
        })
}

fn bvh_leaf_count_pow2(particle_count: usize, leaf_size: usize) -> AutomataResult<usize> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    Ok(particle_count
        .div_ceil(leaf_size)
        .max(1)
        .next_power_of_two())
}

fn bvh_sort_count_pow2(particle_count: usize) -> AutomataResult<usize> {
    if particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH sort particle_count must be greater than zero".to_owned(),
        ));
    }
    Ok(particle_count.next_power_of_two())
}

fn bvh_level_count(leaf_count_pow2: usize) -> usize {
    if leaf_count_pow2 <= 1 {
        return 0;
    }
    leaf_count_pow2.trailing_zeros() as usize
}

fn scan_block_count(cell_count: usize) -> AutomataResult<usize> {
    let count = cell_count.div_ceil(256);
    if count > 256 {
        return Err(AutomataError::InvalidArgument(format!(
            "sorted WGPU scan supports at most 65536 cells, got {cell_count}"
        )));
    }
    Ok(count)
}

fn smoothing_poly6_normalization(grid: &HashGridConfig) -> f32 {
    if grid.dim == 2 {
        4.0 / (std::f32::consts::PI * grid.eps.powi(8))
    } else {
        315.0 / (64.0 * std::f32::consts::PI * grid.eps.powi(9))
    }
}

fn gradient_spiky_normalization(grid: &HashGridConfig) -> f32 {
    if grid.dim == 2 {
        10.0 / (std::f32::consts::PI * grid.eps.powi(5))
    } else {
        15.0 / (std::f32::consts::PI * grid.eps.powi(6))
    }
}

fn state_gradient_scale(model: &NpaModel, grid: &HashGridConfig) -> f32 {
    if model.config.scale_equivariant() {
        grid.eps / model.config.eps0.max(f32::MIN_POSITIVE)
    } else {
        1.0
    }
}

fn density_gradient_scale(model: &NpaModel, grid: &HashGridConfig, particle_count: usize) -> f32 {
    let scale = if model.config.scale_equivariant() {
        (grid.eps / model.config.eps0.max(f32::MIN_POSITIVE)).powi(1 + grid.dim as i32)
    } else {
        1.0
    };
    if model.config.particle_density_equivariant() {
        scale / particle_count.max(1) as f32
    } else {
        scale
    }
}

fn flatten_positions(positions: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * 4);
    for position in positions {
        out.extend_from_slice(position);
    }
    out
}

#[derive(Clone, Debug)]
struct CpuBvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left_or_start: u32,
    right_or_count: u32,
    leaf: bool,
}

fn build_bvh_storage_u32(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    leaf_size: usize,
) -> AutomataResult<Vec<u32>> {
    if leaf_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "BVH leaf_size must be greater than zero".to_owned(),
        ));
    }
    if !(spatial_dims == 2 || spatial_dims == 3) {
        return Err(AutomataError::InvalidArgument(format!(
            "BVH spatial_dims must be 2 or 3, got {spatial_dims}"
        )));
    }
    let mut indices = (0..positions.len()).collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(positions.len().saturating_mul(2).saturating_sub(1));
    let mut ordered = Vec::with_capacity(positions.len());
    if !positions.is_empty() {
        build_bvh_node(
            positions,
            spatial_dims,
            leaf_size,
            &mut indices,
            &mut nodes,
            &mut ordered,
        )?;
    }
    let index_base = BVH_HEADER_U32
        .checked_add(nodes.len().checked_mul(BVH_NODE_U32).ok_or_else(|| {
            AutomataError::InvalidArgument("BVH node storage length overflow".to_owned())
        })?)
        .ok_or_else(|| AutomataError::InvalidArgument("BVH index base overflow".to_owned()))?;
    let mut storage = Vec::with_capacity(index_base + ordered.len());
    storage.push(u32_checked(nodes.len(), "BVH node_count")?);
    storage.push(u32_checked(index_base, "BVH index_base")?);
    storage.push(0);
    storage.push(u32_checked(leaf_size, "BVH leaf_size")?);
    for node in &nodes {
        storage.push(node.min[0].to_bits());
        storage.push(node.min[1].to_bits());
        storage.push(node.min[2].to_bits());
        storage.push(node.max[0].to_bits());
        storage.push(node.max[1].to_bits());
        storage.push(node.max[2].to_bits());
        storage.push(node.left_or_start);
        storage.push(node.right_or_count);
        storage.push(u32::from(node.leaf));
    }
    for index in ordered {
        storage.push(u32_checked(index, "BVH particle index")?);
    }
    Ok(storage)
}

fn build_bvh_node(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    leaf_size: usize,
    indices: &mut [usize],
    nodes: &mut Vec<CpuBvhNode>,
    ordered: &mut Vec<usize>,
) -> AutomataResult<usize> {
    let (min, max) = bvh_bounds(positions, spatial_dims, indices);
    let node_index = nodes.len();
    nodes.push(CpuBvhNode {
        min,
        max,
        left_or_start: 0,
        right_or_count: 0,
        leaf: false,
    });
    if indices.len() <= leaf_size {
        let start = ordered.len();
        ordered.extend_from_slice(indices);
        nodes[node_index] = CpuBvhNode {
            min,
            max,
            left_or_start: u32_checked(start, "BVH leaf start")?,
            right_or_count: u32_checked(indices.len(), "BVH leaf count")?,
            leaf: true,
        };
        return Ok(node_index);
    }

    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if spatial_dims == 3 && extent[2] > extent[0].max(extent[1]) {
        2
    } else if extent[1] > extent[0] {
        1
    } else {
        0
    };
    indices.sort_by(|lhs, rhs| {
        positions[*lhs][axis]
            .partial_cmp(&positions[*rhs][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.cmp(rhs))
    });
    let mid = indices.len() / 2;
    let (left_indices, right_indices) = indices.split_at_mut(mid);
    let left = build_bvh_node(
        positions,
        spatial_dims,
        leaf_size,
        left_indices,
        nodes,
        ordered,
    )?;
    let right = build_bvh_node(
        positions,
        spatial_dims,
        leaf_size,
        right_indices,
        nodes,
        ordered,
    )?;
    nodes[node_index] = CpuBvhNode {
        min,
        max,
        left_or_start: u32_checked(left, "BVH left node")?,
        right_or_count: u32_checked(right, "BVH right node")?,
        leaf: false,
    };
    Ok(node_index)
}

fn bvh_bounds(
    positions: &[[f32; 4]],
    spatial_dims: usize,
    indices: &[usize],
) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &index in indices {
        for axis in 0..spatial_dims {
            let value = positions[index][axis];
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    if spatial_dims == 2 {
        min[2] = 0.0;
        max[2] = 0.0;
    }
    (min, max)
}

fn packed_weights(model: &NpaModel) -> Vec<f32> {
    let mut out = Vec::with_capacity(
        model.weights.w1.len()
            + model.weights.b1.len()
            + model.weights.w2.len()
            + model.weights.b2.len(),
    );
    out.extend_from_slice(&model.weights.w1);
    out.extend_from_slice(&model.weights.b1);
    out.extend_from_slice(&model.weights.w2);
    out.extend_from_slice(&model.weights.b2);
    out
}

fn unflatten_positions(values: &[f32]) -> AutomataResult<Vec<[f32; 4]>> {
    if !values.len().is_multiple_of(4) {
        return Err(AutomataError::InvalidArgument(format!(
            "position readback length {} is not divisible by 4",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(4)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
        .collect())
}

fn u32_checked(value: usize, label: &str) -> AutomataResult<u32> {
    u32::try_from(value).map_err(|_| {
        AutomataError::InvalidArgument(format!("{label} value {value} exceeds u32::MAX"))
    })
}

fn dispatch_groups(total: usize) -> AutomataResult<u32> {
    let total = u32_checked(total, "dispatch total")?;
    Ok(total.div_ceil(WORKGROUP_SIZE))
}

fn byte_len<T>(len: usize) -> AutomataResult<wgpu::BufferAddress> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| AutomataError::InvalidArgument("buffer byte length overflow".to_owned()))?;
    Ok(bytes as wgpu::BufferAddress)
}

fn storage_buffer_f32(device: &wgpu::Device, label: &'static str, values: &[f32]) -> wgpu::Buffer {
    storage_buffer_f32_with_usage(device, label, values, wgpu::BufferUsages::STORAGE)
}

fn storage_buffer_f32_with_usage(
    device: &wgpu::Device,
    label: &'static str,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

fn uniform_buffer_u32(device: &wgpu::Device, label: &'static str, values: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn staging_read_buffer(
    device: &wgpu::Device,
    label: &'static str,
    f32_len: usize,
) -> AutomataResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<f32>(f32_len)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

fn staging_read_buffer_u32(
    device: &wgpu::Device,
    label: &'static str,
    u32_len: usize,
) -> AutomataResult<wgpu::Buffer> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len::<u32>(u32_len)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn gaussian_count_too_small(gaussian: &WgpuGaussianBufferRefs<'_>, count: usize) -> bool {
    let required_vec4 = byte_len::<f32>(count * 4).unwrap_or(wgpu::BufferAddress::MAX);
    let required_sh =
        byte_len::<f32>(count * GAUSSIAN_SH_COEFF_COUNT).unwrap_or(wgpu::BufferAddress::MAX);
    gaussian.position_visibility.size() < required_vec4
        || gaussian.spherical_harmonic.size() < required_sh
        || gaussian.rotation.size() < required_vec4
        || gaussian.scale_opacity.size() < required_vec4
}

fn read_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    f32_len: usize,
) -> AutomataResult<Vec<f32>> {
    read_mapped_buffer(device, buffer, f32_len)
}

fn read_u32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    u32_len: usize,
) -> AutomataResult<Vec<u32>> {
    read_mapped_buffer(device, buffer, u32_len)
}

fn read_mapped_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> AutomataResult<Vec<T>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU poll failed: {err}")))?;
    receiver
        .recv()
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU map callback failed: {err}")))?
        .map_err(|err| AutomataError::InvalidArgument(format!("WGPU buffer map failed: {err}")))?;

    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, T>(&mapped)
        .iter()
        .take(len)
        .copied()
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests;
