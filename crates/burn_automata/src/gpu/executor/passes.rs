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
        if matches!(
            state.neighbor_mode,
            WgpuNeighborMode::SortedCells
                | WgpuNeighborMode::CooperativeSortedCells
                | WgpuNeighborMode::SubgroupCooperativeSortedCells
        ) {
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
            let cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::CooperativeSortedCells
            );
            let subgroup_cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SubgroupCooperativeSortedCells
            );
            let pipeline = if tiled {
                &self.tiled_density_pipeline
            } else if cooperative {
                &self.cooperative_density_pipeline
            } else if subgroup_cooperative {
                self.subgroup_cooperative_density_pipeline
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "WGPU subgroup cooperative density pipeline is unavailable".to_owned(),
                        )
                    })?
            } else {
                &self.density_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else if cooperative || subgroup_cooperative {
                pass.dispatch_workgroups(
                    u32_checked(state.total, "cooperative density groups")?,
                    1,
                    1,
                );
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }

    pub(super) fn encode_update_pass(
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
            let cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::CooperativeSortedCells
            );
            let subgroup_cooperative = matches!(
                state.neighbor_mode,
                WgpuNeighborMode::SubgroupCooperativeSortedCells
            );
            let pipeline = if bvh {
                &self.bvh_update_pipeline
            } else if tiled {
                &self.tiled_update_pipeline
            } else if cooperative {
                &self.cooperative_update_pipeline
            } else if subgroup_cooperative {
                self.subgroup_cooperative_update_pipeline
                    .as_ref()
                    .ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "WGPU subgroup cooperative update pipeline is unavailable".to_owned(),
                        )
                    })?
            } else {
                &self.update_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            if tiled {
                pass.dispatch_workgroups_indirect(&state.indirect_buffer, 0);
            } else if cooperative || subgroup_cooperative {
                pass.dispatch_workgroups(
                    u32_checked(state.total, "cooperative update groups")?,
                    1,
                    1,
                );
            } else {
                pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
            }
        }
        Ok(())
    }
}
