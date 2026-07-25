use wgpu::util::DeviceExt;

use super::*;

impl WgpuAutomataExecutor {
    pub(crate) fn create_active_quadrature_prolongation(
        &self,
        modes: &WgpuAutomataState,
        active: &WgpuAutomataState,
        mode_active_rows: &[u32],
        mode_offsets: &[[f32; 4]],
    ) -> AutomataResult<WgpuActiveQuadratureProlongation> {
        validate_shapes(modes, active, mode_active_rows, mode_offsets)?;
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_active_quadrature_prolong_layout"),
                entries: &[
                    uniform_layout_entry(0),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, false),
                    storage_layout_entry(5, false),
                    storage_layout_entry(6, true),
                    storage_layout_entry(7, true),
                ],
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("burn_automata_active_quadrature_prolong"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    burn_automata_kernels::ACTIVE_QUADRATURE_PROLONG_WGSL,
                )),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("burn_automata_active_quadrature_prolong_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_active_quadrature_prolong_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("active_quadrature_prolong_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let state_dims = modes.state_f32_len / modes.total;
        let params = [
            u32_checked(modes.total, "active quadrature mode count")?,
            u32_checked(active.total, "active quadrature leaf count")?,
            u32_checked(state_dims, "active quadrature state dimensions")?,
            u32_checked(modes.spatial_dims, "active quadrature spatial dimensions")?,
        ];
        let params_buffer = uniform_buffer_u32(
            &self.device,
            "burn_automata_active_quadrature_prolong_params",
            &params,
        );
        let active_rows_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("burn_automata_active_quadrature_rows"),
                    contents: bytemuck::cast_slice(mode_active_rows),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("burn_automata_active_quadrature_offsets"),
                contents: bytemuck::cast_slice(mode_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_groups = std::array::from_fn(|mode_current| {
            std::array::from_fn(|active_current| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("burn_automata_active_quadrature_prolong_bind_group"),
                    layout: &layout,
                    entries: &[
                        bind_entry(0, &params_buffer),
                        bind_entry(1, &active.positions_buffers[active_current]),
                        bind_entry(2, &active.states_buffers[active_current]),
                        bind_entry(3, &active.material_buffer),
                        bind_entry(4, &modes.positions_buffers[mode_current]),
                        bind_entry(5, &modes.states_buffers[mode_current]),
                        bind_entry(6, &active_rows_buffer),
                        bind_entry(7, &offsets_buffer),
                    ],
                })
            })
        });
        let blend_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("burn_automata_active_quadrature_blend_layout"),
                entries: &[
                    uniform_layout_entry(0),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, false),
                    storage_layout_entry(5, true),
                ],
            });
        let blend_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("burn_automata_active_quadrature_blend"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    burn_automata_kernels::ACTIVE_QUADRATURE_BLEND_WGSL,
                )),
            });
        let blend_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("burn_automata_active_quadrature_blend_pipeline_layout"),
                    bind_group_layouts: &[Some(&blend_layout)],
                    immediate_size: 0,
                });
        let blend_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("burn_automata_active_quadrature_blend_pipeline"),
                    layout: Some(&blend_pipeline_layout),
                    module: &blend_shader,
                    entry_point: Some("active_quadrature_blend_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let blend_bind_groups = std::array::from_fn(|source_current| {
            std::array::from_fn(|candidate_current| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("burn_automata_active_quadrature_blend_bind_group"),
                    layout: &blend_layout,
                    entries: &[
                        bind_entry(0, &active.params_buffer),
                        bind_entry(1, &active.positions_buffers[source_current]),
                        bind_entry(2, &active.states_buffers[source_current]),
                        bind_entry(3, &active.positions_buffers[candidate_current]),
                        bind_entry(4, &active.states_buffers[candidate_current]),
                        bind_entry(5, &active.material_buffer),
                    ],
                })
            })
        });
        Ok(WgpuActiveQuadratureProlongation {
            pipeline,
            bind_groups,
            blend_pipeline,
            blend_bind_groups,
            mode_count: modes.total,
            active_count: active.total,
            state_dims,
            spatial_dims: modes.spatial_dims,
        })
    }

    pub(crate) fn step_active_quadrature_many(
        &self,
        prolongation: &WgpuActiveQuadratureProlongation,
        restriction: &WgpuPersistentModeRestriction,
        modes: &mut WgpuAutomataState,
        active: &mut WgpuAutomataState,
        steps: usize,
    ) -> AutomataResult<usize> {
        validate_prolongation(prolongation, modes, active)?;
        if steps == 0 {
            return Ok(0);
        }
        if is_bvh_neighbor_mode(modes.neighbor_mode) {
            return Err(AutomataError::InvalidArgument(
                "active quadrature rollout requires a hash-grid neighbor mode".to_owned(),
            ));
        }
        self.prepare_step_indices(modes, steps)?;
        self.prepare_step_indices(active, steps)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_active_quadrature_step_encoder"),
            });
        let step_index_offset =
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let mut mode_current = modes.current;
        let mut active_current = active.current;
        for step in 0..steps {
            encoder.copy_buffer_to_buffer(
                modes.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "active quadrature step-index buffer is unavailable".to_owned(),
                    )
                })?,
                byte_len::<u32>(step)?,
                &modes.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                active.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "active quadrature material step-index buffer is unavailable".to_owned(),
                    )
                })?,
                byte_len::<u32>(step)?,
                &active.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            encode_prolongation(&mut encoder, prolongation, mode_current, active_current)?;
            let bind_group = &modes.step_bind_groups[mode_current];
            let grid_bind_group = &modes.grid_bind_groups[mode_current];
            self.encode_grid_density_passes(&mut encoder, modes, grid_bind_group, bind_group)?;
            self.encode_update_pass(&mut encoder, modes, bind_group)?;
            mode_current = 1 - mode_current;
            let active_candidate = 1 - active_current;
            super::persistent_modes::encode_persistent_restriction_for_indices(
                &mut encoder,
                restriction,
                mode_current,
                active_candidate,
            )?;
            encode_blend(&mut encoder, prolongation, active_current, active_candidate)?;
            active_current = active_candidate;
        }
        self.queue.submit(Some(encoder.finish()));
        modes.current = mode_current;
        active.current = active_current;
        let steps = u32_checked(steps, "active quadrature step count")?;
        modes.step_index = modes.step_index.wrapping_add(steps);
        active.step_index = active.step_index.wrapping_add(steps);
        self.write_step_index(active);
        Ok(steps as usize)
    }
}

fn encode_prolongation(
    encoder: &mut wgpu::CommandEncoder,
    prolongation: &WgpuActiveQuadratureProlongation,
    mode_current: usize,
    active_current: usize,
) -> AutomataResult<()> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("burn_automata_active_quadrature_prolong_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&prolongation.pipeline);
    pass.set_bind_group(
        0,
        &prolongation.bind_groups[mode_current][active_current],
        &[],
    );
    pass.dispatch_workgroups(dispatch_groups(prolongation.mode_count)?, 1, 1);
    Ok(())
}

fn encode_blend(
    encoder: &mut wgpu::CommandEncoder,
    prolongation: &WgpuActiveQuadratureProlongation,
    source_current: usize,
    candidate_current: usize,
) -> AutomataResult<()> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("burn_automata_active_quadrature_blend_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&prolongation.blend_pipeline);
    pass.set_bind_group(
        0,
        &prolongation.blend_bind_groups[source_current][candidate_current],
        &[],
    );
    pass.dispatch_workgroups(dispatch_groups(prolongation.active_count)?, 1, 1);
    Ok(())
}

fn validate_shapes(
    modes: &WgpuAutomataState,
    active: &WgpuAutomataState,
    mode_active_rows: &[u32],
    mode_offsets: &[[f32; 4]],
) -> AutomataResult<()> {
    if modes.batch_size != 1
        || active.batch_size != 1
        || modes.spatial_dims != active.spatial_dims
        || modes.state_f32_len / modes.total != active.state_f32_len / active.total
        || mode_active_rows.len() != modes.total
        || mode_offsets.len() != modes.total
        || mode_active_rows
            .iter()
            .any(|row| *row as usize >= active.total)
        || mode_offsets
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(AutomataError::InvalidArgument(
            "active quadrature prolongation has incompatible state or mapping shapes".to_owned(),
        ));
    }
    Ok(())
}

fn validate_prolongation(
    prolongation: &WgpuActiveQuadratureProlongation,
    modes: &WgpuAutomataState,
    active: &WgpuAutomataState,
) -> AutomataResult<()> {
    if prolongation.mode_count != modes.total
        || prolongation.active_count != active.total
        || prolongation.state_dims != modes.state_f32_len / modes.total
        || prolongation.state_dims != active.state_f32_len / active.total
        || prolongation.spatial_dims != modes.spatial_dims
        || prolongation.spatial_dims != active.spatial_dims
    {
        return Err(AutomataError::InvalidArgument(
            "active quadrature prolongation no longer matches its resident states".to_owned(),
        ));
    }
    Ok(())
}
