#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
    pub(super) fn write_step_index(&self, state: &WgpuAutomataState) {
        let step_index = [state.step_index];
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&step_index),
        );
    }

    pub(super) fn ensure_step_index_copy_buffers(
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

    pub(super) fn rebuild_bvh_if_needed(&self, state: &WgpuAutomataState) -> AutomataResult<()> {
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

    pub(super) fn build_gpu_bvh_if_needed(&self, state: &WgpuAutomataState) -> AutomataResult<()> {
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
}
