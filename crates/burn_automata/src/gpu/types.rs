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
    CooperativeSortedCells,
    SubgroupCooperativeSortedCells,
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
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) count: usize,
}

pub struct WgpuAutomataState {
    pub total: usize,
    pub particle_count: usize,
    pub batch_size: usize,
    pub(super) spatial_dims: usize,
    pub(super) bvh_leaf_count: usize,
    pub(super) bvh_levels: usize,
    pub(super) bvh_sort_count: usize,
    pub(super) position_f32_len: usize,
    pub(super) state_f32_len: usize,
    pub(super) grid_storage_len: usize,
    pub(super) grid_clear_len: usize,
    pub(super) cell_count: usize,
    pub(super) bucket_capacity: usize,
    pub(super) neighbor_mode: WgpuNeighborMode,
    pub(super) weights_f32_len: usize,
    pub(super) current: usize,
    pub(super) step_index: u32,
    pub(super) params_buffer: wgpu::Buffer,
    pub(super) positions_buffers: [wgpu::Buffer; 2],
    pub(super) states_buffers: [wgpu::Buffer; 2],
    pub(super) weights_buffer: wgpu::Buffer,
    pub(super) linked_grid_buffer: wgpu::Buffer,
    pub(super) indirect_buffer: wgpu::Buffer,
    pub(super) density_buffer: wgpu::Buffer,
    pub(super) grid_bind_groups: [wgpu::BindGroup; 2],
    pub(super) step_bind_groups: [wgpu::BindGroup; 2],
    pub(super) gaussian_source_bind_groups: [wgpu::BindGroup; 2],
    pub(super) step_index_copy_buffers: Vec<wgpu::Buffer>,
}

pub struct WgpuAutomataExecutor {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) grid_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) gaussian_source_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) gaussian_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) subgroup_cooperative_supported: bool,
    pub(super) clear_pipeline: wgpu::ComputePipeline,
    pub(super) bin_pipeline: wgpu::ComputePipeline,
    pub(super) scan_counts_pipeline: wgpu::ComputePipeline,
    pub(super) scan_block_sums_pipeline: wgpu::ComputePipeline,
    pub(super) add_block_offsets_pipeline: wgpu::ComputePipeline,
    pub(super) scatter_sorted_pipeline: wgpu::ComputePipeline,
    pub(super) bvh_init_pipeline: wgpu::ComputePipeline,
    pub(super) bvh_reduce_pipeline: wgpu::ComputePipeline,
    pub(super) morton_sort_init_pipeline: wgpu::ComputePipeline,
    pub(super) morton_sort_step_pipeline: wgpu::ComputePipeline,
    pub(super) density_pipeline: wgpu::ComputePipeline,
    pub(super) tiled_density_pipeline: wgpu::ComputePipeline,
    pub(super) bvh_density_pipeline: wgpu::ComputePipeline,
    pub(super) update_pipeline: wgpu::ComputePipeline,
    pub(super) tiled_update_pipeline: wgpu::ComputePipeline,
    pub(super) bvh_update_pipeline: wgpu::ComputePipeline,
    pub(super) cooperative_density_pipeline: wgpu::ComputePipeline,
    pub(super) cooperative_update_pipeline: wgpu::ComputePipeline,
    pub(super) subgroup_cooperative_density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) subgroup_cooperative_update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) gaussian_pipeline: wgpu::ComputePipeline,
}
