#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
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
}
