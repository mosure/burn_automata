#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
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
