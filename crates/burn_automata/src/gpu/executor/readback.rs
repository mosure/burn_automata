#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
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
}
