#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
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
}
