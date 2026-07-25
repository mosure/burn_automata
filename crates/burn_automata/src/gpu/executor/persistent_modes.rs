use wgpu::util::DeviceExt;

use super::*;

pub(super) fn create_persistent_mode_restriction_pipeline(
    device: &wgpu::Device,
) -> WgpuPersistentModePipeline {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("burn_automata_persistent_mode_restriction_layout"),
        entries: &[
            uniform_layout_entry(0),
            storage_layout_entry(1, true),
            storage_layout_entry(2, true),
            storage_layout_entry(3, false),
            storage_layout_entry(4, false),
            storage_layout_entry(5, true),
            storage_layout_entry(6, true),
            storage_layout_entry(7, true),
            storage_layout_entry(8, false),
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("burn_automata_persistent_mode_restriction"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            burn_automata_kernels::PERSISTENT_MODE_RESTRICT_WGSL,
        )),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("burn_automata_persistent_mode_restriction_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("burn_automata_persistent_mode_restriction_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("persistent_restrict_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    WgpuPersistentModePipeline {
        bind_group_layout: layout,
        pipeline,
    }
}

impl WgpuAutomataExecutor {
    pub(crate) fn create_persistent_mode_restriction(
        &self,
        internal: &WgpuAutomataState,
        active: &WgpuAutomataState,
        mode_offsets: &[u32],
        mode_rows: &[u32],
        mode_weights: &[f32],
    ) -> AutomataResult<WgpuPersistentModeRestriction> {
        validate_persistent_shapes(internal, active, mode_offsets, mode_rows, mode_weights)?;
        let pipeline = self
            .persistent_mode_restriction_pipeline
            .get_or_init(|| create_persistent_mode_restriction_pipeline(&self.device));
        let state_dims = internal.state_f32_len / internal.total;
        let params = [
            u32_checked(internal.total, "persistent internal row count")?,
            u32_checked(active.total, "persistent active row count")?,
            u32_checked(state_dims, "persistent state dimensions")?,
            u32_checked(internal.spatial_dims, "persistent spatial dimensions")?,
        ];
        let params_buffer = uniform_buffer_u32(
            &self.device,
            "burn_automata_persistent_mode_restriction_params",
            &params,
        );
        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("burn_automata_persistent_mode_offsets"),
                contents: bytemuck::cast_slice(mode_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mode_data = mode_rows
            .iter()
            .copied()
            .zip(mode_weights.iter().copied())
            .flat_map(|(row, weight)| [row, weight.to_bits()])
            .collect::<Vec<_>>();
        let mode_data_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("burn_automata_persistent_mode_data"),
                contents: bytemuck::cast_slice(&mode_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_groups = std::array::from_fn(|internal_current| {
            std::array::from_fn(|active_current| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("burn_automata_persistent_mode_restriction_bind_group"),
                    layout: &pipeline.bind_group_layout,
                    entries: &[
                        bind_entry(0, &params_buffer),
                        bind_entry(1, &internal.positions_buffers[internal_current]),
                        bind_entry(2, &internal.states_buffers[internal_current]),
                        bind_entry(3, &active.positions_buffers[active_current]),
                        bind_entry(4, &active.states_buffers[active_current]),
                        bind_entry(5, &offsets_buffer),
                        bind_entry(6, &mode_data_buffer),
                        bind_entry(7, &internal.material_buffer),
                        bind_entry(8, &active.material_buffer),
                    ],
                })
            })
        });
        Ok(WgpuPersistentModeRestriction {
            pipeline: pipeline.pipeline.clone(),
            bind_groups,
            internal_count: internal.total,
            active_count: active.total,
            state_dims,
        })
    }

    pub(crate) fn restrict_persistent_modes(
        &self,
        restriction: &WgpuPersistentModeRestriction,
        internal: &WgpuAutomataState,
        active: &WgpuAutomataState,
    ) -> AutomataResult<()> {
        validate_persistent_restriction(restriction, internal, active)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_persistent_mode_restriction_encoder"),
            });
        encode_persistent_restriction(&mut encoder, restriction, internal, active)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub(crate) fn restrict_persistent_modes_into_gaussians(
        &self,
        restriction: &WgpuPersistentModeRestriction,
        internal: &WgpuAutomataState,
        active: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
    ) -> AutomataResult<()> {
        validate_persistent_restriction(restriction, internal, active)?;
        if gaussian.count < active.total {
            return Err(AutomataError::InvalidArgument(format!(
                "gaussian bind group count {} is smaller than persistent active row count {}",
                gaussian.count, active.total,
            )));
        }
        active.step_index = internal.step_index;
        self.write_step_index(active);
        let gaussian_pipeline = required_pipeline(&self.gaussian_pipeline, "Gaussian output")?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_persistent_mode_gaussian_encoder"),
            });
        encode_persistent_restriction(&mut encoder, restriction, internal, active)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_persistent_mode_gaussian_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(gaussian_pipeline);
            pass.set_bind_group(
                0,
                &active.gaussian_source_bind_groups[1 - active.current],
                &[],
            );
            pass.set_bind_group(1, &gaussian.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(active.total)?, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub(crate) fn step_persistent_modes_into_gaussians(
        &self,
        restriction: &WgpuPersistentModeRestriction,
        internal: &mut WgpuAutomataState,
        active: &mut WgpuAutomataState,
        gaussian: &WgpuGaussianBindGroup,
    ) -> AutomataResult<()> {
        validate_persistent_restriction(restriction, internal, active)?;
        if gaussian.count < active.total {
            return Err(AutomataError::InvalidArgument(format!(
                "gaussian bind group count {} is smaller than persistent active row count {}",
                gaussian.count, active.total,
            )));
        }
        let gaussian_pipeline = required_pipeline(&self.gaussian_pipeline, "Gaussian output")?;
        self.write_step_index(internal);
        self.rebuild_bvh_if_needed(internal)?;
        self.build_gpu_bvh_if_needed(internal)?;

        let source_current = internal.current;
        let restricted_internal_current = 1 - source_current;
        active.step_index = internal.step_index.wrapping_add(1);
        self.write_step_index(active);
        let bind_group = &internal.step_bind_groups[source_current];
        let grid_bind_group = &internal.grid_bind_groups[source_current];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_persistent_mode_fused_step_encoder"),
            });
        self.encode_grid_density_passes(&mut encoder, internal, grid_bind_group, bind_group)?;
        self.encode_update_pass(&mut encoder, internal, bind_group)?;
        encode_persistent_restriction_for_indices(
            &mut encoder,
            restriction,
            restricted_internal_current,
            active.current,
        )?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_persistent_mode_fused_gaussian_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(gaussian_pipeline);
            pass.set_bind_group(
                0,
                &active.gaussian_source_bind_groups[1 - active.current],
                &[],
            );
            pass.set_bind_group(1, &gaussian.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(active.total)?, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        internal.current = restricted_internal_current;
        internal.step_index = internal.step_index.wrapping_add(1);
        Ok(())
    }
}

fn encode_persistent_restriction(
    encoder: &mut wgpu::CommandEncoder,
    restriction: &WgpuPersistentModeRestriction,
    internal: &WgpuAutomataState,
    active: &WgpuAutomataState,
) -> AutomataResult<()> {
    validate_persistent_restriction(restriction, internal, active)?;
    encode_persistent_restriction_for_indices(
        encoder,
        restriction,
        internal.current,
        active.current,
    )
}

pub(super) fn encode_persistent_restriction_for_indices(
    encoder: &mut wgpu::CommandEncoder,
    restriction: &WgpuPersistentModeRestriction,
    internal_current: usize,
    active_current: usize,
) -> AutomataResult<()> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("burn_automata_persistent_mode_restriction_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&restriction.pipeline);
    pass.set_bind_group(
        0,
        &restriction.bind_groups[internal_current][active_current],
        &[],
    );
    pass.dispatch_workgroups(dispatch_groups(restriction.active_count)?, 1, 1);
    Ok(())
}

fn validate_persistent_shapes(
    internal: &WgpuAutomataState,
    active: &WgpuAutomataState,
    mode_offsets: &[u32],
    mode_rows: &[u32],
    mode_weights: &[f32],
) -> AutomataResult<()> {
    let state_dims = internal.state_f32_len / internal.total;
    let valid_offsets = mode_offsets.len() == active.total + 1
        && mode_offsets.first().copied() == Some(0)
        && mode_offsets.last().copied().map(|value| value as usize) == Some(mode_rows.len())
        && offsets_have_only_trailing_empty_rows(mode_offsets);
    if internal.batch_size != 1
        || active.batch_size != 1
        || active.state_f32_len / active.total != state_dims
        || state_dims > 24
        || !valid_offsets
        || mode_rows.len() != internal.total
        || mode_weights.len() != mode_rows.len()
        || mode_rows.iter().any(|row| *row as usize >= internal.total)
        || mode_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "persistent mode restriction has incompatible state or mapping shapes".to_owned(),
        ));
    }
    for range in mode_offsets.windows(2) {
        if range[0] == range[1] {
            continue;
        }
        let sum = mode_weights[range[0] as usize..range[1] as usize]
            .iter()
            .sum::<f32>();
        if (sum - 1.0).abs() > 1.0e-4 {
            return Err(AutomataError::InvalidArgument(
                "persistent mode restriction weights must sum to one per active row".to_owned(),
            ));
        }
    }
    Ok(())
}

fn offsets_have_only_trailing_empty_rows(offsets: &[u32]) -> bool {
    let mut reached_padding = false;
    offsets.windows(2).all(|window| {
        if window[0] > window[1] || (reached_padding && window[0] != window[1]) {
            return false;
        }
        reached_padding |= window[0] == window[1];
        true
    })
}

fn validate_persistent_restriction(
    restriction: &WgpuPersistentModeRestriction,
    internal: &WgpuAutomataState,
    active: &WgpuAutomataState,
) -> AutomataResult<()> {
    if internal.total != restriction.internal_count
        || active.total != restriction.active_count
        || internal.state_f32_len / internal.total != restriction.state_dims
        || active.state_f32_len / active.total != restriction.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "persistent mode restriction no longer matches its resident states".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::offsets_have_only_trailing_empty_rows;

    #[test]
    fn persistent_offsets_allow_only_a_trailing_inert_capacity_tail() {
        assert!(offsets_have_only_trailing_empty_rows(&[0, 2, 4]));
        assert!(offsets_have_only_trailing_empty_rows(&[0, 2, 4, 4, 4]));
        assert!(!offsets_have_only_trailing_empty_rows(&[0, 2, 2, 4]));
        assert!(!offsets_have_only_trailing_empty_rows(&[0, 3, 2]));
    }
}
