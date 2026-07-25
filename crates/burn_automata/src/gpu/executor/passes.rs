#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
    pub(super) fn encode_grid_density_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        grid_bind_group: &wgpu::BindGroup,
        bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        if is_bvh_neighbor_mode(state.neighbor_mode) {
            let pipeline = required_pipeline(&self.bvh_density_pipeline, "BVH density")?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_bvh_density_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            return Ok(());
        }
        if state.fused_sorted_grid_enabled {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_fused_sorted_grid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fused_sorted_grid_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(
                u32_checked(state.batch_size, "fused sorted-grid batches")?,
                1,
                1,
            );
        } else {
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
            if matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SortedCells
                    | WgpuNeighborMode::CooperativeSortedCells
                    | WgpuNeighborMode::SubgroupCooperativeSortedCells
            ) {
                let scan_groups =
                    u32_checked(scan_block_count(state.cell_count)?, "scan block count")?;
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
        }
        if state.stable_sorted_cells_enabled
            && matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SortedCells
                    | WgpuNeighborMode::CooperativeSortedCells
                    | WgpuNeighborMode::SubgroupCooperativeSortedCells
            )
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_stable_sort_cell_particles_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.stable_sort_cells_pipeline);
            pass.set_bind_group(0, grid_bind_group, &[]);
            pass.dispatch_workgroups(
                u32_checked(state.cell_count, "stable sorted-cell count")?,
                1,
                1,
            );
        }
        // The fused adaptive-local pass computes ordinary SPH density while
        // traversing the same sorted support for normalized residual features.
        // Avoid a redundant third neighborhood traversal on adaptive states.
        if state.adaptive_local_rule_mode.uses_normalized_local_pass() {
            return Ok(());
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
            let cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::CooperativeSortedCells
            );
            let subgroup_cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SubgroupCooperativeSortedCells
            );
            let pipeline = if tiled {
                required_pipeline(&self.tiled_density_pipeline, "tiled density")?
            } else if cooperative {
                required_pipeline(&self.cooperative_density_pipeline, "cooperative density")?
            } else if subgroup_cooperative {
                self.subgroup_cooperative_density_pipeline
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "WGPU subgroup cooperative density pipeline is unavailable".to_owned(),
                        )
                    })?
            } else {
                required_pipeline(&self.density_pipeline, "density")?
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else if cooperative || subgroup_cooperative {
                pass.dispatch_workgroups(
                    u32_checked(state.particle_count, "cooperative density particles")?,
                    u32_checked(state.batch_size, "cooperative density batches")?,
                    1,
                );
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }

    pub(super) fn encode_adaptive_local_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        if state.adaptive_local_rule_mode.uses_normalized_local_pass() {
            if state.spatial_dims != 2
                || !matches!(
                    state.neighbor_mode,
                    WgpuNeighborMode::CooperativeSortedCells
                        | WgpuNeighborMode::SubgroupCooperativeSortedCells
                )
            {
                return Err(AutomataError::InvalidArgument(
                    "adaptive normalized local rules currently require 2D sorted-cell WGPU inference"
                        .to_owned(),
                ));
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_adaptive_local_residual_pass"),
                timestamp_writes: None,
            });
            let adaptive_pipeline = if matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SubgroupCooperativeSortedCells
            ) {
                self.subgroup_adaptive_local_pipeline
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "WGPU subgroup adaptive-local pipeline is unavailable".to_owned(),
                        )
                    })?
            } else {
                &self.adaptive_local_pipeline
            };
            pass.set_pipeline(adaptive_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                u32_checked(state.particle_count, "adaptive local residual particles")?,
                u32_checked(state.batch_size, "adaptive local residual batches")?,
                1,
            );
        }
        Ok(())
    }

    pub(super) fn encode_update_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        bind_group: &wgpu::BindGroup,
    ) -> AutomataResult<()> {
        self.encode_adaptive_local_pass(encoder, state, bind_group)?;
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
            let cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::CooperativeSortedCells
            );
            let subgroup_cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SubgroupCooperativeSortedCells
            );
            let pipeline = if bvh {
                required_pipeline(&self.bvh_update_pipeline, "BVH update")?
            } else if tiled {
                required_pipeline(&self.tiled_update_pipeline, "tiled update")?
            } else if cooperative {
                required_pipeline(&self.cooperative_update_pipeline, "cooperative update")?
            } else if subgroup_cooperative {
                self.subgroup_cooperative_update_pipeline
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "WGPU subgroup cooperative update pipeline is unavailable".to_owned(),
                        )
                    })?
            } else {
                required_pipeline(&self.update_pipeline, "update")?
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else if cooperative || subgroup_cooperative {
                pass.dispatch_workgroups(
                    u32_checked(state.particle_count, "cooperative update particles")?,
                    u32_checked(state.batch_size, "cooperative update batches")?,
                    1,
                );
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }
}
