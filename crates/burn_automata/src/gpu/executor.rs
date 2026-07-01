#![allow(clippy::too_many_arguments)]

use super::*;

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../gpu_step.wgsl"))),
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
