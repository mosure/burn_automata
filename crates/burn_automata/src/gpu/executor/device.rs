#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
    pub fn new_blocking() -> AutomataResult<Self> {
        pollster::block_on(Self::new())
    }

    pub(crate) fn new_restriction_blocking() -> AutomataResult<Self> {
        pollster::block_on(Self::new_for_profile(ExecutorProfile::RestrictionSubgroup))
    }

    pub async fn new() -> AutomataResult<Self> {
        Self::new_for_profile(ExecutorProfile::Full).await
    }

    async fn new_for_profile(profile: ExecutorProfile) -> AutomataResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!("no WGPU adapter available: {err}"))
            })?;
        let adapter_features = adapter.features();
        let adapter_info = adapter.get_info();
        let required_features =
            subgroup_cooperative_required_features(adapter_features, &adapter_info);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("burn_automata_wgpu_device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|err| {
                AutomataError::InvalidArgument(format!("failed to create WGPU device: {err}"))
            })?;
        Self::from_device_queue_with_profile(device, queue, profile)
    }

    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> AutomataResult<Self> {
        Self::from_device_queue_with_profile(device, queue, ExecutorProfile::Full)
    }

    fn from_device_queue_with_profile(
        device: wgpu::Device,
        queue: wgpu::Queue,
        profile: ExecutorProfile,
    ) -> AutomataResult<Self> {
        let subgroup_cooperative_supported =
            supports_subgroup_cooperative_sorted_cells(device.features(), &device.adapter_info());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burn_automata_gpu_step"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{}",
                include_str!("../../gpu_step.wgsl"),
                burn_automata_kernels::PAIRED_LOCAL_DETAIL_TOPOLOGY_WGSL,
            ))),
        });
        let fused_sorted_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burn_automata_fused_sorted_grid"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{}\n{}\n{}",
                include_str!("../../gpu_step.wgsl"),
                burn_automata_kernels::PAIRED_LOCAL_DETAIL_TOPOLOGY_WGSL,
                burn_automata_kernels::FUSED_SORTED_GRID_WGSL,
                burn_automata_kernels::STABLE_SORTED_CELLS_WGSL,
            ))),
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
                storage_layout_entry(9, false),
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
                    storage_layout_entry(9, false),
                ],
            });
        let gaussian_source_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_gaussian_source_bind_group_layout"),
                entries: &[
                    uniform_layout_entry(0),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, false),
                    storage_layout_entry(9, false),
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
        let stable_sort_cells_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_stable_sort_cell_particles"),
                layout: Some(&grid_pipeline_layout),
                module: &fused_sorted_grid_shader,
                entry_point: Some("stable_sort_cell_particles_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fused_sorted_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_fused_sorted_grid"),
                layout: Some(&grid_pipeline_layout),
                module: &fused_sorted_grid_shader,
                entry_point: Some("fused_sorted_grid_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let full = profile == ExecutorProfile::Full;
        let bvh_init_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_init"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_init_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let bvh_reduce_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_reduce"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_reduce_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let morton_sort_init_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_morton_sort_init"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("morton_sort_init_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let morton_sort_step_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_morton_sort_step"),
                layout: Some(&grid_pipeline_layout),
                module: &shader,
                entry_point: Some("morton_sort_step_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let density_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let tiled_density_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_tiled_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tiled_density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let bvh_density_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let cooperative_density_pipeline = (full || !subgroup_cooperative_supported).then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_cooperative_density"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cooperative_density_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let update_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let adaptive_local_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_adaptive_local_residual"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("adaptive_local_residual_cooperative_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let resident_bootstrap_split_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_resident_bootstrap_split"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("resident_bootstrap_split_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let paired_local_detail_topology_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_paired_local_detail_topology"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("paired_local_detail_topology_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let continuous_local_detail_topology_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_continuous_local_detail_topology"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("continuous_local_detail_topology_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let tiled_update_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_tiled_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tiled_update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let bvh_update_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_bvh_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("bvh_update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let cooperative_update_pipeline = (full || !subgroup_cooperative_supported).then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_cooperative_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cooperative_update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });
        let (
            subgroup_cooperative_density_pipeline,
            subgroup_cooperative_update_pipeline,
            subgroup_adaptive_local_pipeline,
        ) = create_subgroup_cooperative_pipelines(
            &device,
            &pipeline_layout,
            subgroup_cooperative_supported,
        );
        let gaussian_pipeline = full.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_write_gaussians"),
                layout: Some(&gaussian_pipeline_layout),
                module: &shader,
                entry_point: Some("write_gaussian_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        });

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            grid_bind_group_layout,
            gaussian_source_bind_group_layout,
            gaussian_bind_group_layout,
            subgroup_cooperative_supported,
            clear_pipeline,
            bin_pipeline,
            scan_counts_pipeline,
            scan_block_sums_pipeline,
            add_block_offsets_pipeline,
            scatter_sorted_pipeline,
            stable_sort_cells_pipeline,
            fused_sorted_grid_pipeline,
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
            cooperative_density_pipeline,
            cooperative_update_pipeline,
            adaptive_local_pipeline,
            resident_bootstrap_split_pipeline,
            paired_local_detail_topology_pipeline,
            continuous_local_detail_topology_pipeline,
            subgroup_cooperative_density_pipeline,
            subgroup_cooperative_update_pipeline,
            subgroup_adaptive_local_pipeline,
            gaussian_pipeline,
            persistent_mode_restriction_pipeline: std::sync::OnceLock::new(),
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

    pub fn subgroup_cooperative_supported(&self) -> bool {
        self.subgroup_cooperative_supported
    }
}
