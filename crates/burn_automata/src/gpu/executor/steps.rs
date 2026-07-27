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

    pub fn step_state_many(
        &self,
        state: &mut WgpuAutomataState,
        steps: usize,
    ) -> AutomataResult<usize> {
        let steps = steps.max(1);
        if steps == 1 || is_bvh_neighbor_mode(state.neighbor_mode) {
            for _ in 0..steps {
                self.step_state(state)?;
            }
            return Ok(steps);
        }

        self.prepare_step_indices(state, steps)?;
        let mut current = state.current;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_batched_step_encoder"),
            });
        let step_index_offset =
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        for step_idx in 0..steps {
            encoder.copy_buffer_to_buffer(
                state.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument("step-index buffer is unavailable".to_owned())
                })?,
                byte_len::<u32>(step_idx)?,
                &state.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            let bind_group = &state.step_bind_groups[current];
            let grid_bind_group = &state.grid_bind_groups[current];
            self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
            self.encode_update_pass(&mut encoder, state, bind_group)?;
            current = 1 - current;
        }
        self.queue.submit(Some(encoder.finish()));
        state.current = current;
        state.step_index = state
            .step_index
            .wrapping_add(u32_checked(steps, "batched step count")?);
        Ok(steps)
    }

    pub fn write_state_into_gaussian_bind_group(
        &self,
        state: &WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
    ) -> AutomataResult<()> {
        self.write_state_into_gaussian_bind_group_impl(state, gaussian, None)
    }

    pub fn write_state_pca_into_gaussian_bind_group(
        &self,
        state: &WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<()> {
        self.write_state_into_gaussian_bind_group_impl(state, gaussian, Some(pca))
    }

    fn write_state_into_gaussian_bind_group_impl(
        &self,
        state: &WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        pca: Option<&mut WgpuStatePca>,
    ) -> AutomataResult<()> {
        if gaussian.count < state.total {
            return Err(AutomataError::InvalidArgument(format!(
                "gaussian bind group count {} is smaller than automata particle count {}",
                gaussian.count, state.total
            )));
        }
        self.write_step_index(state);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_write_resident_gaussians_encoder"),
            });
        self.encode_gaussian_export(&mut encoder, state, 1 - state.current, gaussian, pca)?;
        self.queue.submit(Some(encoder.finish()));
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
        self.step_state_into_gaussian_bind_group_impl(state, gaussian, None)
    }

    pub fn step_state_pca_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<()> {
        self.step_state_into_gaussian_bind_group_impl(state, gaussian, Some(pca))
    }

    fn step_state_into_gaussian_bind_group_impl(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        pca: Option<&mut WgpuStatePca>,
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_gaussian_step_encoder"),
            });
        self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
        self.encode_update_pass(&mut encoder, state, bind_group)?;
        self.encode_gaussian_export(&mut encoder, state, state.current, gaussian, pca)?;
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
        self.step_state_many_into_gaussian_bind_group_impl(state, gaussian, steps, None)
    }

    pub fn step_state_many_pca_into_gaussian_bind_group(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        steps: usize,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<usize> {
        self.step_state_many_into_gaussian_bind_group_impl(state, gaussian, steps, Some(pca))
    }

    fn step_state_many_into_gaussian_bind_group_impl(
        &self,
        state: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
        steps: usize,
        mut pca: Option<&mut WgpuStatePca>,
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
                    self.step_state_into_gaussian_bind_group_impl(
                        state,
                        gaussian,
                        pca.as_deref_mut(),
                    )?;
                } else {
                    self.step_state(state)?;
                }
            }
            return Ok(steps);
        }

        self.prepare_step_indices(state, steps)?;

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
                state.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument("step-index buffer is unavailable".to_owned())
                })?,
                byte_len::<u32>(step_idx)?,
                &state.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            let bind_group = &state.step_bind_groups[current];
            let grid_bind_group = &state.grid_bind_groups[current];
            self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
            self.encode_update_pass(&mut encoder, state, bind_group)?;
            if step_idx + 1 == steps {
                self.encode_gaussian_export(
                    &mut encoder,
                    state,
                    current,
                    gaussian,
                    pca.as_deref_mut(),
                )?;
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

    fn encode_gaussian_export(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        source_index: usize,
        gaussian: &WgpuGaussianBindGroup,
        pca: Option<&mut WgpuStatePca>,
    ) -> AutomataResult<()> {
        if let Some(pca) = pca {
            return self.encode_state_pca_into_gaussians(
                encoder,
                state,
                source_index,
                gaussian,
                pca,
            );
        }
        let gaussian_pipeline = required_pipeline(&self.gaussian_pipeline, "Gaussian output")?;
        let gaussian_source_bind_group = &state.gaussian_source_bind_groups[source_index];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("burn_automata_write_gaussians_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(gaussian_pipeline);
        pass.set_bind_group(0, gaussian_source_bind_group, &[]);
        pass.set_bind_group(1, &gaussian.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_groups(state.total)?, 1, 1);
        Ok(())
    }
}
