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
    pub support_bin_count: usize,
    pub support_bin_capacity: usize,
    pub requested_support_bin_count: usize,
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

/// Resident particle-state PCA settings used by the Gaussian viewer export.
///
/// Basis fitting follows the rolling Oja update used by `burn_jepa`: projection
/// and display normalization run every frame, while mean and basis statistics
/// are refreshed on a bounded cadence without host readback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WgpuStatePcaConfig {
    pub update_every: usize,
    pub warmup_iterations: usize,
    pub update_iterations: usize,
    pub warmup_learning_rate: f32,
    pub learning_rate: f32,
    pub mean_momentum: f32,
    pub display_momentum: f32,
    pub display_clip_sigma: f32,
    pub display_std_floor: f32,
    pub epsilon: f32,
}

impl Default for WgpuStatePcaConfig {
    fn default() -> Self {
        Self {
            update_every: 8,
            warmup_iterations: 8,
            update_iterations: 2,
            warmup_learning_rate: 0.5,
            learning_rate: 0.12,
            mean_momentum: 0.2,
            display_momentum: 0.2,
            display_clip_sigma: 2.5,
            display_std_floor: 1.0e-3,
            epsilon: 1.0e-6,
        }
    }
}

pub struct WgpuStatePca {
    pub(super) config: WgpuStatePcaConfig,
    pub(super) state_dims: usize,
    pub(super) particle_capacity: usize,
    pub(super) partial_capacity: usize,
    pub(super) observed_frames: usize,
    pub(super) last_particle_count: usize,
    pub(super) update_count: usize,
    pub(super) initialized: bool,
    pub(super) force_update: bool,
    pub(super) params: [u32; 12],
    pub(super) params_buffer: wgpu::Buffer,
    pub(super) data_buffer: wgpu::Buffer,
    pub(super) mean_offset: usize,
    pub(super) components_offset: usize,
    pub(super) display_center_offset: usize,
    pub(super) display_spread_offset: usize,
    pub(super) bind_group: wgpu::BindGroup,
}

impl WgpuStatePca {
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    pub fn force_update(&mut self) {
        self.force_update = true;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WgpuStatePcaSnapshot {
    pub mean: Vec<f32>,
    /// Row-major `[state_dims, 3]` projection basis.
    pub components: Vec<f32>,
    pub display_center: [f32; 3],
    pub display_spread: [f32; 3],
    pub update_count: usize,
}

pub const WGPU_MATERIAL_UPDATE_MASK_MEMBERS: usize = 6;

/// Declared continuous bandwidth range used to construct conservative device
/// support bins. Bins accelerate candidate lookup only; exact pair bandwidth
/// remains authoritative in the perception shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WgpuSupportBinConfig {
    pub min_bandwidth: f32,
    pub max_bandwidth: f32,
    pub ratio: f32,
    /// Bypass the runtime density gate. Intended for parity tests and explicit
    /// benchmarking; production adaptive states should leave this false.
    pub force: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct WgpuMaterialUpdateMask {
    pub particle_ids: [u64; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
    pub weights: [f32; WGPU_MATERIAL_UPDATE_MASK_MEMBERS],
}

impl WgpuMaterialUpdateMask {
    pub fn single(particle_id: u64) -> Self {
        let mut particle_ids = [0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS];
        let mut weights = [0.0; WGPU_MATERIAL_UPDATE_MASK_MEMBERS];
        particle_ids[0] = particle_id;
        weights[0] = 1.0;
        Self {
            particle_ids,
            weights,
        }
    }
}

/// Immutable material metadata consumed by represented-measure GPU rollouts.
///
/// Fixed-resolution NPA states use the executor's unit-measure defaults. Adaptive
/// callers provide physical represented measure and conservative covariance so
/// the simulation and Gaussian renderer share one material-scale definition.
pub struct WgpuMaterialStateInit<'a> {
    pub represented_measure: &'a [f32],
    /// Stable material identities used by the stochastic update schedule.
    /// Adaptive topology may reorder rows, so row indices are not stable keys.
    pub particle_ids: Option<&'a [u64]>,
    /// Optional fine-lineage masks for material aggregates. The GPU evaluates
    /// each child Bernoulli draw and restricts it with these weights.
    pub update_masks: Option<&'a [WgpuMaterialUpdateMask]>,
    /// Per-particle SPH support radius used by adaptive perception.
    pub bandwidth: &'a [f32],
    /// Optional conservative execution bins for continuous bandwidths.
    pub support_bins: Option<WgpuSupportBinConfig>,
    pub covariance: &'a [[f32; 9]],
    pub state_jacobian: &'a [f32],
    /// Optional compact recurrent closure state, laid out `[row, state]`.
    /// Ordinary and legacy material states leave this absent.
    pub closure_mode: Option<&'a [f32]>,
    /// Persistent four-child affine-null orientation anchor, laid out
    /// `[row, 4]`. It is required whenever recurrent closure is active.
    pub closure_basis: Option<&'a [f32]>,
    /// Unit geometry phase paired with the recurrent closure coefficient,
    /// laid out `[row, 2]`.
    pub closure_phase: Option<&'a [f32]>,
    /// Isotropic rendered scale at the beginning of a topology transition.
    pub render_from_scale: &'a [f32],
    /// Continuous physical support requested by the adaptive controller.
    pub render_target_footprint: &'a [f32],
    pub display_scale_per_footprint: f32,
    pub render_transition_steps: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WgpuAdaptiveLocalRuleMode {
    #[default]
    Disabled,
    Residual,
    NormalizedPrimary,
    CoarseReplacement,
    CompatibleResidual,
    NormalizedExposureResidual,
}

impl WgpuAdaptiveLocalRuleMode {
    pub(crate) const fn as_u32(self) -> u32 {
        match self {
            Self::Disabled => 0,
            Self::Residual => 1,
            Self::NormalizedPrimary => 2,
            Self::CoarseReplacement => 3,
            Self::CompatibleResidual => 4,
            Self::NormalizedExposureResidual => 5,
        }
    }

    pub(crate) const fn uses_normalized_local_pass(self) -> bool {
        matches!(
            self,
            Self::Residual
                | Self::NormalizedPrimary
                | Self::CoarseReplacement
                | Self::NormalizedExposureResidual
        )
    }
}

pub struct WgpuAutomataState {
    /// Active rows dispatched by dynamics and rendering kernels.
    pub total: usize,
    pub particle_count: usize,
    /// Resident rows allocated per trajectory. Ordinary states keep this equal
    /// to `particle_count`; adaptive states may activate reserved rows without
    /// reallocating or synchronizing through the host.
    pub particle_capacity: usize,
    pub(super) allocation_total: usize,
    pub batch_size: usize,
    pub(super) spatial_dims: usize,
    pub(super) hidden_dims: usize,
    pub(super) feature_dims: usize,
    pub(super) output_dims: usize,
    pub(super) bvh_leaf_count: usize,
    pub(super) bvh_levels: usize,
    pub(super) bvh_sort_count: usize,
    pub(super) position_f32_len: usize,
    pub(super) state_f32_len: usize,
    pub(super) material_enabled: bool,
    pub(super) mean_represented_measure: f32,
    pub(super) max_material_bandwidth: f32,
    pub(super) support_bin_count: usize,
    pub(super) support_bin_capacity: usize,
    pub(super) requested_support_bin_count: usize,
    pub(super) support_bin_min: f32,
    pub(super) support_bin_max: f32,
    pub(super) support_bin_ratio: f32,
    pub(super) support_bins_forced: bool,
    pub(super) display_scale_per_footprint: f32,
    pub(super) render_transition_steps: u32,
    pub(super) render_transition_start_step: u32,
    pub(super) adaptive_local_rule_mode: WgpuAdaptiveLocalRuleMode,
    pub(super) adaptive_local_hidden_start: u32,
    pub(super) adaptive_local_residual_scale: f32,
    pub(super) adaptive_base_footprint: f32,
    pub(super) adaptive_reference_footprint: f32,
    pub(super) adaptive_shepard_epsilon: f32,
    pub(super) adaptive_moment_regularization: f32,
    pub(super) adaptive_moment_condition_limit: f32,
    pub(super) adaptive_max_neighbors: u32,
    pub(super) adaptive_pair_scale_power: f32,
    pub(super) expected_coarse_update_mask: bool,
    pub(super) adaptive_closure_enabled: bool,
    pub(super) adaptive_closure_hidden_dims: u32,
    pub(super) adaptive_closure_basis_enabled: bool,
    pub(super) adaptive_closure_basis_hidden_dims: u32,
    pub(super) dt: f32,
    pub(super) update_prob: f32,
    pub(super) grid_storage_len: usize,
    pub(super) grid_clear_len: usize,
    pub(super) cell_count: usize,
    pub(super) spatial_cell_count: usize,
    pub(super) bucket_capacity: usize,
    pub(super) neighbor_mode: WgpuNeighborMode,
    pub(super) fused_sorted_grid_enabled: bool,
    pub(super) stable_sorted_cells_enabled: bool,
    pub(super) weights_f32_len: usize,
    pub(super) current: usize,
    pub(super) step_index: u32,
    pub(super) lane_seeds: Vec<u32>,
    pub(super) params_buffer: wgpu::Buffer,
    pub(super) positions_buffers: [wgpu::Buffer; 2],
    pub(super) states_buffers: [wgpu::Buffer; 2],
    pub(super) weights_buffer: wgpu::Buffer,
    pub(super) linked_grid_buffer: wgpu::Buffer,
    pub(super) indirect_buffer: wgpu::Buffer,
    pub(super) density_buffer: wgpu::Buffer,
    pub(super) diagnostics_f32_len: usize,
    pub(super) material_buffer: wgpu::Buffer,
    pub(super) grid_bind_groups: [wgpu::BindGroup; 2],
    pub(super) step_bind_groups: [wgpu::BindGroup; 2],
    pub(super) gaussian_source_bind_groups: [wgpu::BindGroup; 2],
    pub(super) step_indices_buffer: Option<wgpu::Buffer>,
    pub(super) step_indices_capacity: usize,
}

pub(crate) struct WgpuCoupledFineRecenter {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_groups: [[wgpu::BindGroup; 2]; 2],
    pub(super) total_students: usize,
    pub(super) batch_size: usize,
    pub(super) fine_count: usize,
    pub(super) student_count: usize,
    pub(super) state_dims: usize,
}

pub(crate) struct WgpuPersistentModeRestriction {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_groups: [[wgpu::BindGroup; 2]; 2],
    pub(super) internal_count: usize,
    pub(super) active_count: usize,
    pub(super) state_dims: usize,
}

pub(crate) struct WgpuActiveQuadratureProlongation {
    pub(super) pipeline: wgpu::ComputePipeline,
    /// Indexed as `[mode_current][active_current]`.
    pub(super) bind_groups: [[wgpu::BindGroup; 2]; 2],
    pub(super) blend_pipeline: wgpu::ComputePipeline,
    /// Indexed as `[source_current][candidate_current]`.
    pub(super) blend_bind_groups: [[wgpu::BindGroup; 2]; 2],
    pub(super) mode_count: usize,
    pub(super) active_count: usize,
    pub(super) state_dims: usize,
    pub(super) spatial_dims: usize,
}

pub(crate) struct WgpuCoupledFineSnapshot {
    pub fine_base_update: Vec<f32>,
    pub fine_positions: Vec<[f32; 4]>,
    pub fine_states: Vec<f32>,
    pub student_positions: Vec<[f32; 4]>,
    pub student_states: Vec<f32>,
    pub student_diagnostics: WgpuAdaptiveDiagnostics,
}

pub(crate) struct WgpuPendingCoupledFineSnapshot {
    pub(super) staging: wgpu::Buffer,
    pub(super) fine_position_start: usize,
    pub(super) fine_state_start: usize,
    pub(super) student_position_start: usize,
    pub(super) student_state_start: usize,
    pub(super) student_diagnostics_start: usize,
    pub(super) readback_len: usize,
}

pub(crate) struct WgpuTeacherSnapshot {
    pub positions: Vec<[f32; 4]>,
    pub states: Vec<f32>,
    pub base_features: Vec<f32>,
    pub normalized_features: Vec<f32>,
    pub base_update: Vec<f32>,
    pub observed_spacing: Vec<f32>,
    pub accepted_degree: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct WgpuAdaptiveDiagnostics {
    #[allow(dead_code)]
    pub base_features: Vec<f32>,
    pub normalized_features: Vec<f32>,
    pub base_update: Vec<f32>,
    pub model_update: Vec<f32>,
    pub observed_spacing: Vec<f32>,
    pub accepted_degree: Vec<usize>,
    #[allow(dead_code)]
    pub coarse_exposure: Vec<f32>,
    pub feature_dims: usize,
    pub output_dims: usize,
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
    pub(super) stable_sort_cells_pipeline: wgpu::ComputePipeline,
    pub(super) fused_sorted_grid_pipeline: wgpu::ComputePipeline,
    pub(super) bvh_init_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) bvh_reduce_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) morton_sort_init_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) morton_sort_step_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) tiled_density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) bvh_density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) tiled_update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) bvh_update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) cooperative_density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) cooperative_update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) adaptive_local_pipeline: wgpu::ComputePipeline,
    pub(super) resident_bootstrap_split_pipeline: wgpu::ComputePipeline,
    pub(super) paired_local_detail_topology_pipeline: wgpu::ComputePipeline,
    pub(super) continuous_local_detail_topology_pipeline: wgpu::ComputePipeline,
    pub(super) subgroup_cooperative_density_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) subgroup_cooperative_update_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) subgroup_adaptive_local_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) gaussian_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) pca_pipelines: std::sync::OnceLock<WgpuPcaPipelines>,
    pub(super) persistent_mode_restriction_pipeline:
        std::sync::OnceLock<WgpuPersistentModePipeline>,
}

pub(super) struct WgpuPcaPipelines {
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) partial_mean: wgpu::ComputePipeline,
    pub(super) finalize_mean: wgpu::ComputePipeline,
    pub(super) project_update: wgpu::ComputePipeline,
    pub(super) oja_candidate: wgpu::ComputePipeline,
    pub(super) stabilize_basis: wgpu::ComputePipeline,
    pub(super) display_stats: wgpu::ComputePipeline,
    pub(super) write_gaussian: wgpu::ComputePipeline,
}

pub(super) struct WgpuPersistentModePipeline {
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) pipeline: wgpu::ComputePipeline,
}
