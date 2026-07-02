#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
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
}
