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
const MAX_STATE_DIMS: usize = 16;
const MAX_HIDDEN_DIMS: usize = 256;
const MAX_FEATURE_DIMS: usize = 96;
const MAX_OUTPUT_DIMS: usize = 20;
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

impl WgpuAutomataExecutor {
    pub fn new_blocking() -> AutomataResult<Self> {
        pollster::block_on(Self::new())
    }

    pub async fn new() -> AutomataResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!("no WGPU adapter available: {err}"))
            })?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!("failed to create WGPU device: {err}"))
            })?;
        Self::from_device_queue(device, queue)
    }

    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> AutomataResult<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burn_automata_gpu_step"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("gpu_step.wgsl"))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("burn_automata_step_bind_group_layout"),
            entries: &[
                uniform_layout_entry(0),
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, true),
                storage_layout_entry(4, false),
                storage_layout_entry(5, false),
                storage_layout_entry(6, false),
                storage_layout_entry(7, false),
            ],
        });
        let grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_grid_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0),
                    storage_layout_entry(1, true),
                    storage_layout_entry(4, false),
                    storage_layout_entry(8, false),
                ],
            });
        let gaussian_source_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_gaussian_source_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, false),
                ],
            });
        let gaussian_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_gaussian_bind_group_layout"),
                entries: &[
                    storage_layout_entry(0, false),
                    storage_layout_entry(1, false),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("burn_automata_step_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("burn_automata_grid_pipeline_layout"),
            bind_group_layouts: &[Some(&grid_bind_group_layout)],
            immediate_size: 0,
        });
        let gaussian_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("burn_automata_gaussian_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&gaussian_source_bind_group_layout),
                    Some(&gaussian_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_clear_grid"),
            layout: Some(&grid_pipeline_layout),
            module: &shader,
            entry_point: Some("clear_grid_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_bin_particles"),
            layout: Some(&grid_pipeline_layout),
            module: &shader,
            entry_point: Some("bin_particles_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scan_counts_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_scan_counts"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("scan_counts_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let scan_block_sums_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_scan_block_sums"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("scan_block_sums_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let add_block_offsets_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_add_block_offsets"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("add_block_offsets_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let scatter_sorted_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_scatter_sorted_particles"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_sorted_particles_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bvh_init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_bvh_init"),
            layout: Some(&grid_pipeline_layout),
            module: &shader,
            entry_point: Some("bvh_init_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bvh_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_reduce"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_reduce_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let morton_sort_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_morton_sort_init"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("morton_sort_init_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let morton_sort_step_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_morton_sort_step"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("morton_sort_step_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_density"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("density_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let tiled_density_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_tiled_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tiled_density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bvh_density_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_update"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let tiled_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_tiled_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tiled_update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bvh_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let gaussian_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn_automata_write_gaussians"),
            layout: Some(&gaussian_pipeline_layout),
            module: &shader,
            entry_point: Some("write_gaussian_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            grid_bind_group_layout,
            gaussian_source_bind_group_layout,
            gaussian_bind_group_layout,
            clear_pipeline,
            bin_pipeline,
            scan_counts_pipeline,
            scan_block_sums_pipeline,
            add_block_offsets_pipeline,
            scatter_sorted_pipeline,
            bvh_init_pipeline,
            bvh_reduce_pipeline,
            morton_sort_init_pipeline,
            morton_sort_step_pipeline,
            density_pipeline,
            tiled_density_pipeline,
            bvh_density_pipeline,
            update_pipeline,
            tiled_update_pipeline,
            bvh_update_pipeline,
            gaussian_pipeline,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn wait_idle(&self) -> AutomataResult<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| AutomataError::InvalidArgument(format!("WGPU poll failed: {err}")))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            WgpuNeighborMode::Auto,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_update_prob(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode_and_update_prob(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            WgpuNeighborMode::Auto,
            update_prob,
            seed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_neighbor_mode(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode_and_update_prob(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            1.0,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_neighbor_mode_and_update_prob(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<WgpuAutomataState> {
        validate_gpu_step(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            update_prob,
        )?;

        let total = positions.len();
        let (bucket_capacity, resolved_neighbor_mode) =
            resolve_neighbor_mode_for_state(grid, particle_count, positions, neighbor_mode)?;
        let bvh_leaf_count = match resolved_neighbor_mode {
            WgpuNeighborMode::GpuBvh { .. }
            | WgpuNeighborMode::GpuLbvh { .. }
            | WgpuNeighborMode::GpuMortonLbvh { .. } => {
                bvh_leaf_count_pow2(total, bucket_capacity)?
            }
            _ => 0,
        };
        let bvh_levels = bvh_level_count(bvh_leaf_count);
        let bvh_sort_count = match resolved_neighbor_mode {
            WgpuNeighborMode::GpuMortonLbvh { .. } => bvh_sort_count_pow2(total)?,
            _ => 0,
        };
        let mut params = gpu_params(
            model,
            total,
            particle_count,
            grid,
            dt,
            bucket_capacity,
            resolved_neighbor_mode,
            update_prob,
            seed,
        )?;
        params[PARAM_BVH_LEAF_COUNT] = u32_checked(bvh_leaf_count, "BVH leaf count")?;
        params[PARAM_BVH_SORT_COUNT] = u32_checked(bvh_sort_count, "BVH sort count")?;
        let position_values = flatten_positions(positions);
        let weights = packed_weights(model);
        let weights_f32_len = weights.len();
        let position_f32_len = position_values.len();
        let state_f32_len = states.len();
        let grid_storage_len = grid_storage_len_for_mode(
            grid.cell_count(),
            total,
            bucket_capacity,
            resolved_neighbor_mode,
        )?;
        let grid_clear_len =
            grid_clear_len_for_mode(grid.cell_count(), bucket_capacity, resolved_neighbor_mode)?;
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let params_buffer = uniform_buffer_u32(&self.device, "burn_automata_state_params", &params);
        let positions_a = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_positions_a",
            &position_values,
            state_usage,
        );
        let positions_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_positions_b"),
            size: byte_len::<f32>(position_f32_len)?,
            usage: state_usage,
            mapped_at_creation: false,
        });
        let states_a = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_states_a",
            states,
            state_usage,
        );
        let states_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_states_b"),
            size: byte_len::<f32>(state_f32_len)?,
            usage: state_usage,
            mapped_at_creation: false,
        });
        let weights_buffer = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_weights",
            &weights,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let linked_grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_linked_grid"),
            size: byte_len::<u32>(grid_storage_len)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let WgpuNeighborMode::Bvh { leaf_size } = resolved_neighbor_mode {
            let storage = build_bvh_storage_u32(positions, model.config.spatial_dims, leaf_size)?;
            if storage.len() > grid_storage_len {
                return Err(AutomataError::InvalidArgument(format!(
                    "BVH storage len {} exceeds allocated grid storage len {}",
                    storage.len(),
                    grid_storage_len
                )));
            }
            self.queue
                .write_buffer(&linked_grid_buffer, 0, bytemuck::cast_slice(&storage));
        }
        let indirect_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_indirect_args"),
            size: byte_len::<u32>(3)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_density"),
            size: byte_len::<f32>(total)?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let positions_buffers = [positions_a, positions_b];
        let states_buffers = [states_a, states_b];
        let grid_bind_groups = std::array::from_fn(|current| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_state_grid_bind_group"),
                layout: &self.grid_bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(1, &positions_buffers[current]),
                    bind_entry(4, &linked_grid_buffer),
                    bind_entry(8, &indirect_buffer),
                ],
            })
        });
        let step_bind_groups = std::array::from_fn(|current| {
            let next = 1 - current;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_state_step_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(1, &positions_buffers[current]),
                    bind_entry(2, &states_buffers[current]),
                    bind_entry(3, &weights_buffer),
                    bind_entry(4, &linked_grid_buffer),
                    bind_entry(5, &positions_buffers[next]),
                    bind_entry(6, &states_buffers[next]),
                    bind_entry(7, &density_buffer),
                ],
            })
        });
        let gaussian_source_bind_groups = std::array::from_fn(|current| {
            let next = 1 - current;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_gaussian_source_bind_group"),
                layout: &self.gaussian_source_bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(5, &positions_buffers[next]),
                    bind_entry(6, &states_buffers[next]),
                ],
            })
        });
        Ok(WgpuAutomataState {
            total,
            particle_count,
            batch_size,
            spatial_dims: model.config.spatial_dims,
            bvh_leaf_count,
            bvh_levels,
            bvh_sort_count,
            position_f32_len,
            state_f32_len,
            grid_storage_len,
            grid_clear_len,
            cell_count: grid.cell_count(),
            bucket_capacity,
            neighbor_mode: resolved_neighbor_mode,
            weights_f32_len,
            current: 0,
            step_index: 0,
            params_buffer,
            positions_buffers,
            states_buffers,
            weights_buffer,
            linked_grid_buffer,
            indirect_buffer,
            density_buffer,
            grid_bind_groups,
            step_bind_groups,
            gaussian_source_bind_groups,
            step_index_copy_buffers: Vec::new(),
        })
    }

    pub fn update_state_model(
        &self,
        state: &mut WgpuAutomataState,
        model: &NpaModel,
        grid: &HashGridConfig,
        dt: f32,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<()> {
        validate_gpu_model_config(model, grid)?;
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
        let weights = packed_weights(model);
        if weights.len() != state.weights_f32_len {
            return Err(AutomataError::InvalidArgument(format!(
                "updated model weight len {} != resident weight len {}",
                weights.len(),
                state.weights_f32_len
            )));
        }
        let mut params = gpu_params(
            model,
            state.total,
            state.particle_count,
            grid,
            dt,
            state.bucket_capacity,
            state.neighbor_mode,
            update_prob,
            seed,
        )?;
        params[PARAM_BVH_LEAF_COUNT] = u32_checked(state.bvh_leaf_count, "BVH leaf count")?;
        params[PARAM_BVH_SORT_COUNT] = u32_checked(state.bvh_sort_count, "BVH sort count")?;
        self.queue
            .write_buffer(&state.params_buffer, 0, bytemuck::cast_slice(&params));
        self.queue
            .write_buffer(&state.weights_buffer, 0, bytemuck::cast_slice(&weights));
        Ok(())
    }

    pub fn neighbor_report(&self, state: &WgpuAutomataState) -> WgpuNeighborReport {
        WgpuNeighborReport {
            bucket_capacity: state.bucket_capacity,
            grid_storage_len: state.grid_storage_len,
            grid_clear_len: state.grid_clear_len,
            mode: state.neighbor_mode,
        }
    }

    pub fn create_gaussian_buffers(
        &self,
        count: usize,
    ) -> AutomataResult<WgpuOwnedGaussianBuffers> {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        Ok(WgpuOwnedGaussianBuffers {
            position_visibility: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("burn_automata_gaussian_position_visibility"),
                size: byte_len::<f32>(count * 4)?,
                usage,
                mapped_at_creation: false,
            }),
            spherical_harmonic: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("burn_automata_gaussian_spherical_harmonic"),
                size: byte_len::<f32>(count * GAUSSIAN_SH_COEFF_COUNT)?,
                usage,
                mapped_at_creation: false,
            }),
            rotation: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("burn_automata_gaussian_rotation"),
                size: byte_len::<f32>(count * 4)?,
                usage,
                mapped_at_creation: false,
            }),
            scale_opacity: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("burn_automata_gaussian_scale_opacity"),
                size: byte_len::<f32>(count * 4)?,
                usage,
                mapped_at_creation: false,
            }),
            count,
        })
    }

    pub fn create_gaussian_bind_group(
        &self,
        gaussian: &WgpuGaussianBufferRefs<'_>,
        count: usize,
    ) -> AutomataResult<WgpuGaussianBindGroup> {
        if gaussian_count_too_small(gaussian, count) {
            return Err(AutomataError::InvalidArgument(
                "gaussian buffers are smaller than the requested bind group count".to_owned(),
            ));
        }
        Ok(WgpuGaussianBindGroup {
            bind_group: self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_gaussian_bind_group"),
                layout: &self.gaussian_bind_group_layout,
                entries: &[
                    bind_entry(0, gaussian.position_visibility),
                    bind_entry(1, gaussian.spherical_harmonic),
                    bind_entry(2, gaussian.rotation),
                    bind_entry(3, gaussian.scale_opacity),
                ],
            }),
            count,
        })
    }

    pub fn step_state(&self, state: &mut WgpuAutomataState) -> AutomataResult<()> {
        self.write_step_index(state);
        self.rebuild_bvh_if_needed(state)?;
        self.build_gpu_bvh_if_needed(state)?;
        let bind_group = &state.step_bind_groups[state.current];
        let grid_bind_group = &state.grid_bind_groups[state.current];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_step_encoder"),
            });
        self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
        self.encode_update_pass(&mut encoder, state, bind_group)?;
        self.queue.submit(Some(encoder.finish()));
        state.current = 1 - state.current;
        state.step_index = state.step_index.wrapping_add(1);
        Ok(())
    }

    pub fn step_state_into_gaussians(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBufferRefs<'_>,
    ) -> AutomataResult<()> {
        let gaussian_bind_group = self.create_gaussian_bind_group(gaussian, state.total)?;
        self.step_state_into_gaussian_bind_group(state, &gaussian_bind_group)
    }

    pub fn step_state_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
    ) -> AutomataResult<()> {
        if gaussian.count < state.total {
            return Err(AutomataError::InvalidArgument(format!(
                "gaussian bind group count {} is smaller than automata particle count {}",
                gaussian.count, state.total
            )));
        }
        self.write_step_index(state);
        self.rebuild_bvh_if_needed(state)?;
        self.build_gpu_bvh_if_needed(state)?;
        let bind_group = &state.step_bind_groups[state.current];
        let grid_bind_group = &state.grid_bind_groups[state.current];
        let gaussian_source_bind_group = &state.gaussian_source_bind_groups[state.current];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_gaussian_step_encoder"),
            });
        self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
        self.encode_update_pass(&mut encoder, state, bind_group)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_write_gaussians_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gaussian_pipeline);
            pass.set_bind_group(0, gaussian_source_bind_group, &[]);
            pass.set_bind_group(1, &gaussian.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        state.current = 1 - state.current;
        state.step_index = state.step_index.wrapping_add(1);
        Ok(())
    }

    pub fn step_state_many_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        steps: usize,
    ) -> AutomataResult<usize> {
        let steps = steps.max(1);
        if gaussian.count < state.total {
            return Err(AutomataError::InvalidArgument(format!(
                "gaussian bind group count {} is smaller than automata particle count {}",
                gaussian.count, state.total
            )));
        }

        if steps == 1 || is_bvh_neighbor_mode(state.neighbor_mode) {
            for step_idx in 0..steps {
                if step_idx + 1 == steps {
                    self.step_state_into_gaussian_bind_group(state, gaussian)?;
                } else {
                    self.step_state(state)?;
                }
            }
            return Ok(steps);
        }

        self.ensure_step_index_copy_buffers(state, steps)?;
        for step_idx in 0..steps {
            let step_index = [state
                .step_index
                .wrapping_add(u32_checked(step_idx, "batched step index")?)];
            self.queue.write_buffer(
                &state.step_index_copy_buffers[step_idx],
                0,
                bytemuck::cast_slice(&step_index),
            );
        }

        let mut current = state.current;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_gaussian_batched_step_encoder"),
            });
        let step_index_offset =
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        for step_idx in 0..steps {
            encoder.copy_buffer_to_buffer(
                &state.step_index_copy_buffers[step_idx],
                0,
                &state.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            let bind_group = &state.step_bind_groups[current];
            let grid_bind_group = &state.grid_bind_groups[current];
            self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
            self.encode_update_pass(&mut encoder, state, bind_group)?;
            if step_idx + 1 == steps {
                let gaussian_source_bind_group = &state.gaussian_source_bind_groups[current];
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_write_gaussians_batched_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gaussian_pipeline);
                pass.set_bind_group(0, gaussian_source_bind_group, &[]);
                pass.set_bind_group(1, &gaussian.bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
            current = 1 - current;
        }
        self.queue.submit(Some(encoder.finish()));
        state.current = current;
        state.step_index = state
            .step_index
            .wrapping_add(u32_checked(steps, "batched step count")?);
        Ok(steps)
    }

    pub fn read_state(&self, state: &WgpuAutomataState) -> AutomataResult<WgpuStepOutput> {
        let out_positions_staging = staging_read_buffer(
            &self.device,
            "burn_automata_state_positions_staging",
            state.position_f32_len,
        )?;
        let out_states_staging = staging_read_buffer(
            &self.device,
            "burn_automata_state_states_staging",
            state.state_f32_len,
        )?;
        let density_staging = staging_read_buffer(
            &self.device,
            "burn_automata_state_density_staging",
            state.total,
        )?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &state.positions_buffers[state.current],
            0,
            &out_positions_staging,
            0,
            byte_len::<f32>(state.position_f32_len)?,
        );
        encoder.copy_buffer_to_buffer(
            &state.states_buffers[state.current],
            0,
            &out_states_staging,
            0,
            byte_len::<f32>(state.state_f32_len)?,
        );
        encoder.copy_buffer_to_buffer(
            &state.density_buffer,
            0,
            &density_staging,
            0,
            byte_len::<f32>(state.total)?,
        );
        self.queue.submit(Some(encoder.finish()));

        let out_positions_flat =
            read_f32_buffer(&self.device, &out_positions_staging, state.position_f32_len)?;
        let next_states = read_f32_buffer(&self.device, &out_states_staging, state.state_f32_len)?;
        let density = read_f32_buffer(&self.device, &density_staging, state.total)?;
        Ok(WgpuStepOutput {
            next_positions: unflatten_positions(&out_positions_flat)?,
            next_states,
            density,
        })
    }

    pub fn read_grid_overflow(&self, state: &WgpuAutomataState) -> AutomataResult<u32> {
        if state.bucket_capacity == 0 || is_bvh_neighbor_mode(state.neighbor_mode) {
            return Ok(0);
        }
        let overflow_index = state
            .cell_count
            .checked_mul(state.bucket_capacity)
            .and_then(|slots| slots.checked_add(state.cell_count))
            .ok_or_else(|| {
                AutomataError::InvalidArgument("grid overflow index overflow".to_owned())
            })?;
        let overflow_staging =
            staging_read_buffer_u32(&self.device, "burn_automata_state_grid_overflow_staging", 1)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_grid_overflow_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &state.linked_grid_buffer,
            byte_len::<u32>(overflow_index)?,
            &overflow_staging,
            0,
            byte_len::<u32>(1)?,
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(read_u32_buffer(&self.device, &overflow_staging, 1)?
            .into_iter()
            .next()
            .unwrap_or(0))
    }

    pub fn read_gaussian_buffers(
        &self,
        buffers: &WgpuOwnedGaussianBuffers,
    ) -> AutomataResult<WgpuGaussianReadback> {
        self.read_gaussian_buffer_refs(&buffers.refs(), buffers.count)
    }

    pub fn read_gaussian_buffer_refs(
        &self,
        buffers: &WgpuGaussianBufferRefs<'_>,
        count: usize,
    ) -> AutomataResult<WgpuGaussianReadback> {
        Ok(WgpuGaussianReadback {
            position_visibility: self.read_storage_f32(
                buffers.position_visibility,
                count * 4,
                "burn_automata_read_position_visibility",
            )?,
            spherical_harmonic: self.read_storage_f32(
                buffers.spherical_harmonic,
                count * GAUSSIAN_SH_COEFF_COUNT,
                "burn_automata_read_spherical_harmonic",
            )?,
            rotation: self.read_storage_f32(
                buffers.rotation,
                count * 4,
                "burn_automata_read_rotation",
            )?,
            scale_opacity: self.read_storage_f32(
                buffers.scale_opacity,
                count * 4,
                "burn_automata_read_scale_opacity",
            )?,
        })
    }

    fn read_storage_f32(
        &self,
        source: &wgpu::Buffer,
        f32_len: usize,
        label: &'static str,
    ) -> AutomataResult<Vec<f32>> {
        let staging = staging_read_buffer(&self.device, label, f32_len)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_read_storage_encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len::<f32>(f32_len)?);
        self.queue.submit(Some(encoder.finish()));
        read_f32_buffer(&self.device, &staging, f32_len)
    }

    fn write_step_index(&self, state: &WgpuAutomataState) {
        let step_index = [state.step_index];
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&step_index),
        );
    }

    fn ensure_step_index_copy_buffers(
        &self,
        state: &mut WgpuAutomataState,
        count: usize,
    ) -> AutomataResult<()> {
        while state.step_index_copy_buffers.len() < count {
            state
                .step_index_copy_buffers
                .push(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("burn_automata_step_index_copy"),
                    size: byte_len::<u32>(1)?,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        Ok(())
    }

    fn rebuild_bvh_if_needed(&self, state: &WgpuAutomataState) -> AutomataResult<()> {
        let WgpuNeighborMode::Bvh { leaf_size } = state.neighbor_mode else {
            return Ok(());
        };
        let positions =
            self.read_positions_buffer(&state.positions_buffers[state.current], state)?;
        let storage = build_bvh_storage_u32(&positions, state.spatial_dims, leaf_size)?;
        if storage.len() > state.grid_storage_len {
            return Err(AutomataError::InvalidArgument(format!(
                "BVH storage len {} exceeds allocated grid storage len {}",
                storage.len(),
                state.grid_storage_len
            )));
        }
        self.queue
            .write_buffer(&state.linked_grid_buffer, 0, bytemuck::cast_slice(&storage));
        Ok(())
    }

    fn build_gpu_bvh_if_needed(&self, state: &WgpuAutomataState) -> AutomataResult<()> {
        if !matches!(
            state.neighbor_mode,
            WgpuNeighborMode::GpuBvh { .. }
                | WgpuNeighborMode::GpuLbvh { .. }
                | WgpuNeighborMode::GpuMortonLbvh { .. }
        ) {
            return Ok(());
        }
        if state.bvh_leaf_count == 0 {
            return Err(AutomataError::InvalidArgument(
                "GPU BVH state has zero leaf count".to_owned(),
            ));
        }
        let grid_bind_group = &state.grid_bind_groups[state.current];
        if matches!(state.neighbor_mode, WgpuNeighborMode::GpuLbvh { .. }) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_gpu_lbvh_sort_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_clear_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.clear_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.grid_clear_len)?, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_bin_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bin_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
            let scan_groups = u32_checked(scan_block_count(state.cell_count)?, "scan block count")?;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_scan_counts_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_counts_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(scan_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_scan_block_sums_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_block_sums_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_add_block_offsets_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.add_block_offsets_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.cell_count)?, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_gpu_lbvh_scatter_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_sorted_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        if matches!(state.neighbor_mode, WgpuNeighborMode::GpuMortonLbvh { .. }) {
            self.build_morton_sort(state, grid_bind_group)?;
        }
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_gpu_bvh_init_encoder"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_gpu_bvh_init_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bvh_init_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(
                dispatch_groups(state.total.max(state.bvh_leaf_count))?,
                1,
                1,
            );
            drop(pass);
            self.queue.submit(Some(encoder.finish()));
        }

        for level in 0..state.bvh_levels {
            let level_u32 = [u32_checked(level, "BVH build level")?];
            self.queue.write_buffer(
                &state.params_buffer,
                (PARAM_BVH_BUILD_LEVEL * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                bytemuck::cast_slice(&level_u32),
            );
            let nodes_this_level = state.bvh_leaf_count >> (level + 1);
            if nodes_this_level == 0 {
                continue;
            }
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_gpu_bvh_reduce_encoder"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_gpu_bvh_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bvh_reduce_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(nodes_this_level)?, 1, 1);
            drop(pass);
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }

    fn build_morton_sort(
        &self,
        state: &WgpuAutomataState,
        grid_bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        if state.bvh_sort_count == 0 {
            return Err(AutomataError::InvalidArgument(
                "GPU Morton LBVH state has zero sort count".to_owned(),
            ));
        }
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_morton_sort_init_encoder"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_morton_sort_init_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.morton_sort_init_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.bvh_sort_count)?, 1, 1);
            drop(pass);
            self.queue.submit(Some(encoder.finish()));
        }

        let mut k = 2usize;
        while k <= state.bvh_sort_count {
            let mut j = k / 2;
            while j > 0 {
                let sort_params = [
                    u32_checked(k, "Morton sort k")?,
                    u32_checked(j, "Morton sort j")?,
                ];
                debug_assert_eq!(PARAM_BVH_SORT_J, PARAM_BVH_SORT_K + 1);
                self.queue.write_buffer(
                    &state.params_buffer,
                    (PARAM_BVH_SORT_K * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                    bytemuck::cast_slice(&sort_params),
                );
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("burn_automata_morton_sort_step_encoder"),
                        });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_morton_sort_step_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.morton_sort_step_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.bvh_sort_count)?, 1, 1);
                drop(pass);
                self.queue.submit(Some(encoder.finish()));
                j /= 2;
            }
            k *= 2;
        }
        Ok(())
    }

    fn read_positions_buffer(
        &self,
        source: &wgpu::Buffer,
        state: &WgpuAutomataState,
    ) -> AutomataResult<Vec<[f32; 4]>> {
        let staging = staging_read_buffer(
            &self.device,
            "burn_automata_bvh_positions_staging",
            state.position_f32_len,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_bvh_positions_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            source,
            0,
            &staging,
            0,
            byte_len::<f32>(state.position_f32_len)?,
        );
        self.queue.submit(Some(encoder.finish()));
        let flat = read_f32_buffer(&self.device, &staging, state.position_f32_len)?;
        unflatten_positions(&flat)
    }

    fn encode_grid_density_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        grid_bind_group: &wgpu::BindGroup,
        bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        if is_bvh_neighbor_mode(state.neighbor_mode) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_bvh_density_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bvh_density_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            return Ok(());
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_clear_grid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.grid_clear_len)?, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_bin_particles_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
        }
        if matches!(state.neighbor_mode, WgpuNeighborMode::SortedCells) {
            let scan_groups = u32_checked(scan_block_count(state.cell_count)?, "scan block count")?;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_scan_counts_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_counts_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(scan_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_scan_block_sums_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_block_sums_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_add_block_offsets_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.add_block_offsets_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.cell_count)?, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("burn_automata_scatter_sorted_particles_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_sorted_pipeline);
                pass.set_bind_group(0, grid_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_density_pass"),
                timestamp_writes: None,
            });
            let tiled = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::TiledFixedCellBuckets { .. }
            );
            pass.set_pipeline(if tiled {
                &self.tiled_density_pipeline
            } else {
                &self.density_pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }

    fn encode_update_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_update_pass"),
                timestamp_writes: None,
            });
            let tiled = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::TiledFixedCellBuckets { .. }
            );
            let bvh = is_bvh_neighbor_mode(state.neighbor_mode);
            pass.set_pipeline(if bvh {
                &self.bvh_update_pipeline
            } else if tiled {
                &self.tiled_update_pipeline
            } else {
                &self.update_pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
    ) -> AutomataResult<WgpuStepOutput> {
        validate_gpu_step(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            1.0,
        )?;

        let total = positions.len();
        let (bucket_capacity, resolved_neighbor_mode) = resolve_neighbor_mode_for_state(
            grid,
            particle_count,
            positions,
            WgpuNeighborMode::Auto,
        )?;
        let grid_clear_len =
            grid_clear_len_for_mode(grid.cell_count(), bucket_capacity, resolved_neighbor_mode)?;
        let params = gpu_params(
            model,
            total,
            particle_count,
            grid,
            dt,
            bucket_capacity,
            resolved_neighbor_mode,
            1.0,
            0,
        )?;
        let position_values = flatten_positions(positions);
        let weights = packed_weights(model);

        let params_buffer = uniform_buffer_u32(&self.device, "params", &params);
        let positions_buffer = storage_buffer_f32(&self.device, "positions", &position_values);
        let states_buffer = storage_buffer_f32(&self.device, "states", states);
        let weights_buffer = storage_buffer_f32(&self.device, "weights", &weights);
        let linked_grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linked_grid"),
            size: byte_len::<u32>(grid_storage_len_for_mode(
                grid.cell_count(),
                total,
                bucket_capacity,
                resolved_neighbor_mode,
            )?)?,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let indirect_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect_args"),
            size: byte_len::<u32>(3)?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let out_positions_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_out_positions"),
            size: byte_len::<f32>(position_values.len())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_states_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_out_states"),
            size: byte_len::<f32>(states.len())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_density"),
            size: byte_len::<f32>(total)?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("burn_automata_grid_bind_group"),
            layout: &self.grid_bind_group_layout,
            entries: &[
                bind_entry(0, &params_buffer),
                bind_entry(1, &positions_buffer),
                bind_entry(4, &linked_grid_buffer),
                bind_entry(8, &indirect_buffer),
            ],
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("burn_automata_step_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                bind_entry(0, &params_buffer),
                bind_entry(1, &positions_buffer),
                bind_entry(2, &states_buffer),
                bind_entry(3, &weights_buffer),
                bind_entry(4, &linked_grid_buffer),
                bind_entry(5, &out_positions_buffer),
                bind_entry(6, &out_states_buffer),
                bind_entry(7, &density_buffer),
            ],
        });

        let out_positions_staging = staging_read_buffer(
            &self.device,
            "burn_automata_out_positions_staging",
            position_values.len(),
        )?;
        let out_states_staging = staging_read_buffer(
            &self.device,
            "burn_automata_out_states_staging",
            states.len(),
        )?;
        let density_staging =
            staging_read_buffer(&self.device, "burn_automata_density_staging", total)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_step_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_clear_grid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(grid_clear_len)?, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_bin_particles_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(total)?, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_density_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(total)?, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(total)?, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &out_positions_buffer,
            0,
            &out_positions_staging,
            0,
            byte_len::<f32>(position_values.len())?,
        );
        encoder.copy_buffer_to_buffer(
            &out_states_buffer,
            0,
            &out_states_staging,
            0,
            byte_len::<f32>(states.len())?,
        );
        encoder.copy_buffer_to_buffer(
            &density_buffer,
            0,
            &density_staging,
            0,
            byte_len::<f32>(total)?,
        );
        self.queue.submit(Some(encoder.finish()));

        let out_positions_flat =
            read_f32_buffer(&self.device, &out_positions_staging, position_values.len())?;
        let next_states = read_f32_buffer(&self.device, &out_states_staging, states.len())?;
        let density = read_f32_buffer(&self.device, &density_staging, total)?;

        Ok(WgpuStepOutput {
            next_positions: unflatten_positions(&out_positions_flat)?,
            next_states,
            density,
        })
    }
}

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
mod tests {
    use super::*;
    use crate::{AutomataPreset, NpaConfig, ParticleSeed, rollout::seed_particles_scaled};

    #[test]
    fn auto_bucket_capacity_helper_keeps_particle_hash_linked_list() {
        let grid = HashGridConfig::growing_2d();
        let particle_count = grid.cell_count() * 8;
        let capacity =
            resolve_bucket_capacity(&grid, particle_count, WgpuNeighborMode::Auto).unwrap();

        assert_eq!(capacity, 0);
    }

    #[test]
    fn auto_bucket_capacity_helper_keeps_periodic_2d_linked_list() {
        let grid = HashGridConfig::texture_2d();
        let capacity =
            resolve_bucket_capacity(&grid, grid.cell_count() * 64, WgpuNeighborMode::Auto).unwrap();

        assert_eq!(capacity, 0);
    }

    #[test]
    fn adaptive_auto_keeps_sparse_particle_grid_linked_list() {
        let grid = HashGridConfig::growing_3dgs();
        let positions = (0..128)
            .map(|idx| {
                let x = (idx % 16) as f32 * grid.eps;
                let y = ((idx / 16) % 8) as f32 * grid.eps;
                [x, y, 0.0, 0.0]
            })
            .collect::<Vec<_>>();

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            positions.len(),
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();

        assert_eq!(capacity, 0);
        assert_eq!(mode, WgpuNeighborMode::LinkedList);
    }

    #[test]
    fn adaptive_auto_uses_tiled_buckets_for_2d_particle_grid_cells() {
        let grid = HashGridConfig::growing_2d();
        let positions = vec![[0.0, 0.0, 0.0, 0.0]; 128];

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            positions.len(),
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();

        assert!(capacity >= 128);
        assert_eq!(mode, WgpuNeighborMode::TiledFixedCellBuckets { capacity });
    }

    #[test]
    fn adaptive_auto_keeps_large_2d_tiled_storage_under_binding_limit() {
        let particle_count = 32_768;
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let (positions, _) = seed_particles_scaled(
            1,
            particle_count,
            config.state_dims,
            config.spatial_dims,
            0,
            ParticleSeed::UniformCircle,
            0.2,
        );

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            particle_count,
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();
        let storage_len =
            grid_storage_len_for_mode(grid.cell_count(), particle_count, capacity, mode).unwrap();

        assert!(
            grid_storage_binding_len_fits(storage_len),
            "mode={mode:?} capacity={capacity} storage_len={storage_len}"
        );
        if let WgpuNeighborMode::TiledFixedCellBuckets { capacity } = mode {
            assert!(
                capacity < 8192,
                "adaptive capacity should be reduced below the previously crashing 8192"
            );
        }
    }

    #[test]
    fn adaptive_auto_uses_tiled_buckets_for_periodic_2d_grid() {
        let grid = HashGridConfig::texture_2d();
        let positions = (0..512)
            .map(|idx| {
                let x = ((idx % 32) as f32 / 31.0) * 2.0 - 1.0;
                let y = ((idx / 32) as f32 / 15.0) * 2.0 - 1.0;
                [x, y, 0.0, 0.0]
            })
            .collect::<Vec<_>>();

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            positions.len(),
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();

        assert!(capacity > 0);
        assert_eq!(mode, WgpuNeighborMode::TiledFixedCellBuckets { capacity });
    }

    #[test]
    fn adaptive_auto_uses_sorted_cells_for_small_collapsed_3d_particle_grid_cells() {
        let grid = HashGridConfig::growing_3dgs();
        let positions = vec![[0.0, 0.0, 0.0, 0.0]; 128];

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            positions.len(),
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();

        assert_eq!(capacity, 0);
        assert_eq!(mode, WgpuNeighborMode::SortedCells);
    }

    #[test]
    fn adaptive_auto_falls_back_when_3d_fixed_buckets_cannot_fit_binding_limit() {
        let grid = HashGridConfig::growing_3dgs();
        let particle_count = 8192;
        let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

        let (capacity, mode) = resolve_neighbor_mode_for_state(
            &grid,
            particle_count,
            &positions,
            WgpuNeighborMode::Auto,
        )
        .unwrap();
        let storage_len =
            grid_storage_len_for_mode(grid.cell_count(), particle_count, capacity, mode).unwrap();

        assert_eq!(capacity, 0);
        assert_eq!(mode, WgpuNeighborMode::SortedCells);
        assert!(grid_storage_binding_len_fits(storage_len));
    }
}
