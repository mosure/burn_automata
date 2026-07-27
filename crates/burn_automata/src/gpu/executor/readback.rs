#![allow(clippy::too_many_arguments)]

use super::state::{
    MATERIAL_CLOSURE_BASIS_CAPACITY, MATERIAL_CLOSURE_BASIS_OFFSET, MATERIAL_CLOSURE_MODE_OFFSET,
    MATERIAL_CLOSURE_PHASE_CAPACITY, MATERIAL_CLOSURE_PHASE_OFFSET, MATERIAL_STRIDE,
};
use super::*;

impl WgpuAutomataExecutor {
    pub fn read_positions_states(
        &self,
        state: &WgpuAutomataState,
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>)> {
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_state_read_positions_states_encoder"),
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
        self.queue.submit(Some(encoder.finish()));

        let positions =
            read_f32_buffer(&self.device, &out_positions_staging, state.position_f32_len)?;
        let states = read_f32_buffer(&self.device, &out_states_staging, state.state_f32_len)?;
        Ok((unflatten_positions(&positions)?, states))
    }

    pub(crate) fn read_material_closure_state(
        &self,
        state: &WgpuAutomataState,
    ) -> AutomataResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if !state.material_enabled || state.total == 0 {
            return Err(AutomataError::InvalidArgument(
                "closure-mode readback requires a non-empty material state".to_owned(),
            ));
        }
        let state_dims = state.state_f32_len / state.total;
        let packed = self.read_storage_f32(
            &state.material_buffer,
            state.total * MATERIAL_STRIDE,
            "burn_automata_material_closure_staging",
        )?;
        let mut closure_mode = Vec::with_capacity(state.total * state_dims);
        let mut closure_basis = Vec::with_capacity(state.total * MATERIAL_CLOSURE_BASIS_CAPACITY);
        let mut closure_phase = Vec::with_capacity(state.total * MATERIAL_CLOSURE_PHASE_CAPACITY);
        for row in 0..state.total {
            let start = row * MATERIAL_STRIDE + MATERIAL_CLOSURE_MODE_OFFSET;
            closure_mode.extend_from_slice(&packed[start..start + state_dims]);
            let basis_start = row * MATERIAL_STRIDE + MATERIAL_CLOSURE_BASIS_OFFSET;
            closure_basis.extend_from_slice(
                &packed[basis_start..basis_start + MATERIAL_CLOSURE_BASIS_CAPACITY],
            );
            let phase_start = row * MATERIAL_STRIDE + MATERIAL_CLOSURE_PHASE_OFFSET;
            closure_phase.extend_from_slice(
                &packed[phase_start..phase_start + MATERIAL_CLOSURE_PHASE_CAPACITY],
            );
        }
        Ok((closure_mode, closure_basis, closure_phase))
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

    pub(crate) fn read_local_detail_topology_accept_count(
        &self,
        state: &WgpuAutomataState,
    ) -> AutomataResult<u32> {
        let staging = staging_read_buffer(
            &self.device,
            "burn_automata_local_detail_topology_accept_count_staging",
            1,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_local_detail_topology_accept_count_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &state.material_buffer,
            byte_len::<f32>(state.allocation_total * MATERIAL_STRIDE)?,
            &staging,
            0,
            byte_len::<f32>(1)?,
        );
        self.queue.submit(Some(encoder.finish()));
        let value = read_f32_buffer(&self.device, &staging, 1)?
            .into_iter()
            .next()
            .unwrap_or(0.0);
        if !value.is_finite() || value < 0.0 || value > u32::MAX as f32 {
            return Err(AutomataError::InvalidModel(format!(
                "paired topology accept counter is invalid: {value}"
            )));
        }
        Ok(value.round() as u32)
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

    pub(super) fn read_storage_f32(
        &self,
        source: &wgpu::Buffer,
        f32_len: usize,
        label: &'static str,
    ) -> AutomataResult<Vec<f32>> {
        self.read_storage_f32_range(source, 0, f32_len, label)
    }

    pub(super) fn read_storage_f32_range(
        &self,
        source: &wgpu::Buffer,
        source_offset: usize,
        f32_len: usize,
        label: &'static str,
    ) -> AutomataResult<Vec<f32>> {
        let staging = staging_read_buffer(&self.device, label, f32_len)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_read_storage_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            source,
            byte_len::<f32>(source_offset)?,
            &staging,
            0,
            byte_len::<f32>(f32_len)?,
        );
        self.queue.submit(Some(encoder.finish()));
        read_f32_buffer(&self.device, &staging, f32_len)
    }
}
