use wgpu::util::DeviceExt;

use super::*;

impl WgpuAutomataExecutor {
    pub(crate) fn create_coupled_fine_recenter(
        &self,
        fine: &WgpuAutomataState,
        student: &WgpuAutomataState,
        member_offsets: &[u32],
        member_leaves: &[u32],
        closure_enabled: bool,
    ) -> AutomataResult<WgpuCoupledFineRecenter> {
        validate_coupled_shapes(fine, student, member_offsets, member_leaves)?;
        if closure_enabled
            && member_offsets
                .windows(2)
                .any(|window| !matches!(window[1] - window[0], 1 | 4))
        {
            return Err(AutomataError::InvalidArgument(
                "compact closure recenter requires one- or four-child material rows".to_owned(),
            ));
        }
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("burn_automata_coupled_fine_recenter_layout"),
                    entries: &[
                        uniform_layout_entry(0),
                        storage_layout_entry(1, false),
                        storage_layout_entry(2, false),
                        storage_layout_entry(3, true),
                        storage_layout_entry(4, true),
                        storage_layout_entry(5, true),
                        storage_layout_entry(6, true),
                        storage_layout_entry(7, true),
                        storage_layout_entry(8, true),
                    ],
                });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("burn_automata_coupled_fine_recenter"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    burn_automata_kernels::COUPLED_FINE_RECENTER_WGSL,
                )),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("burn_automata_coupled_fine_recenter_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("burn_automata_coupled_fine_recenter_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("coupled_recenter_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let state_dims = fine.state_f32_len / fine.total;
        let params = [
            u32_checked(fine.particle_count, "coupled fine particle count")?,
            u32_checked(student.particle_count, "coupled student particle count")?,
            u32_checked(state_dims, "coupled state dimensions")?,
            u32_checked(student.total, "coupled student total")?,
            u32::from(closure_enabled),
            0,
            0,
            0,
        ];
        let params_buffer = uniform_buffer_u32(
            &self.device,
            "burn_automata_coupled_fine_recenter_params",
            &params,
        );
        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("burn_automata_coupled_fine_member_offsets"),
                contents: bytemuck::cast_slice(member_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let leaves_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("burn_automata_coupled_fine_member_leaves"),
                contents: bytemuck::cast_slice(member_leaves),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_groups = std::array::from_fn(|fine_current| {
            std::array::from_fn(|student_current| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("burn_automata_coupled_fine_recenter_bind_group"),
                    layout: &bind_group_layout,
                    entries: &[
                        bind_entry(0, &params_buffer),
                        bind_entry(1, &fine.positions_buffers[fine_current]),
                        bind_entry(2, &fine.states_buffers[fine_current]),
                        bind_entry(3, &student.positions_buffers[student_current]),
                        bind_entry(4, &student.states_buffers[student_current]),
                        bind_entry(5, &offsets_buffer),
                        bind_entry(6, &leaves_buffer),
                        bind_entry(7, &student.material_buffer),
                        bind_entry(8, &fine.material_buffer),
                    ],
                })
            })
        });
        Ok(WgpuCoupledFineRecenter {
            pipeline,
            bind_groups,
            total_students: student.total,
            batch_size: fine.batch_size,
            fine_count: fine.particle_count,
            student_count: student.particle_count,
            state_dims,
        })
    }

    pub(crate) fn step_coupled_fine_states_many(
        &self,
        coupling: &WgpuCoupledFineRecenter,
        fine: &mut WgpuAutomataState,
        student: &mut WgpuAutomataState,
        steps: usize,
    ) -> AutomataResult<usize> {
        validate_coupling(coupling, fine, student)?;
        if steps == 0 {
            return Ok(0);
        }
        if is_bvh_neighbor_mode(fine.neighbor_mode) || is_bvh_neighbor_mode(student.neighbor_mode) {
            return Err(AutomataError::InvalidArgument(
                "batched coupled-fine rollout requires hash-grid neighbor modes".to_owned(),
            ));
        }
        self.prepare_step_indices(fine, steps)?;
        self.prepare_step_indices(student, steps)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_coupled_fine_step_encoder"),
            });
        let step_index_offset =
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let mut fine_current = fine.current;
        let mut student_current = student.current;
        for step in 0..steps {
            encoder.copy_buffer_to_buffer(
                fine.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "fine step-index buffer is unavailable".to_owned(),
                    )
                })?,
                byte_len::<u32>(step)?,
                &fine.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                student.step_indices_buffer.as_ref().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "student step-index buffer is unavailable".to_owned(),
                    )
                })?,
                byte_len::<u32>(step)?,
                &student.params_buffer,
                step_index_offset,
                std::mem::size_of::<u32>() as wgpu::BufferAddress,
            );
            encode_recenter_current(&mut encoder, coupling, fine_current, student_current)?;
            encode_state_step_current(self, &mut encoder, fine, fine_current)?;
            encode_state_step_current(self, &mut encoder, student, student_current)?;
            fine_current = 1 - fine_current;
            student_current = 1 - student_current;
        }
        self.queue.submit(Some(encoder.finish()));
        fine.current = fine_current;
        student.current = student_current;
        let steps = u32_checked(steps, "coupled-fine step count")?;
        fine.step_index = fine.step_index.wrapping_add(steps);
        student.step_index = student.step_index.wrapping_add(steps);
        Ok(steps as usize)
    }

    pub(crate) fn enqueue_coupled_fine_snapshot(
        &self,
        coupling: &WgpuCoupledFineRecenter,
        fine: &mut WgpuAutomataState,
        student: &mut WgpuAutomataState,
        base_footprint: f32,
        student_config: AdaptivePerceptionConfig,
        advance: bool,
    ) -> AutomataResult<WgpuPendingCoupledFineSnapshot> {
        validate_coupling(coupling, fine, student)?;
        let fine_saved = self.begin_base_update_diagnostics(fine)?;
        let student_saved =
            match self.begin_adaptive_diagnostics(student, base_footprint, student_config) {
                Ok(saved) => saved,
                Err(error) => {
                    self.restore_adaptive_diagnostics(fine, fine_saved);
                    return Err(error);
                }
            };
        let result: AutomataResult<_> = (|| {
            if advance {
                self.advance_state_during_diagnostics(fine);
                self.advance_state_during_diagnostics(student);
            }
            self.write_step_index(fine);
            self.rebuild_bvh_if_needed(fine)?;
            self.build_gpu_bvh_if_needed(fine)?;
            self.write_step_index(student);
            self.rebuild_bvh_if_needed(student)?;
            self.build_gpu_bvh_if_needed(student)?;

            let fine_update_len = fine.total * fine.output_dims;
            let fine_position_start = fine_update_len;
            let fine_state_start = fine_position_start
                .checked_add(fine.position_f32_len)
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "coupled-fine snapshot readback size overflow".to_owned(),
                    )
                })?;
            let student_position_start = fine_state_start
                .checked_add(fine.state_f32_len)
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "coupled-fine snapshot readback size overflow".to_owned(),
                    )
                })?;
            let student_state_start = student_position_start + student.position_f32_len;
            let student_diagnostics_start = student_state_start
                .checked_add(student.state_f32_len)
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "coupled-fine snapshot readback size overflow".to_owned(),
                    )
                })?;
            let readback_len = student_diagnostics_start
                .checked_add(student.diagnostics_f32_len)
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "coupled-fine diagnostic readback size overflow".to_owned(),
                    )
                })?;
            let diagnostic_update_start = fine
                .total
                .checked_add(2 * fine.total * fine.feature_dims)
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "coupled-fine diagnostic offset overflow".to_owned(),
                    )
                })?;
            let staging = staging_read_buffer(
                &self.device,
                "burn_automata_coupled_fine_snapshot_staging",
                readback_len,
            )?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_coupled_fine_snapshot_encoder"),
                });
            encode_recenter(&mut encoder, coupling, fine, student)?;
            encode_state_step(self, &mut encoder, fine)?;
            encode_state_step(self, &mut encoder, student)?;
            encoder.copy_buffer_to_buffer(
                &fine.density_buffer,
                byte_len::<f32>(diagnostic_update_start)?,
                &staging,
                0,
                byte_len::<f32>(fine_update_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &fine.positions_buffers[fine.current],
                0,
                &staging,
                byte_len::<f32>(fine_position_start)?,
                byte_len::<f32>(fine.position_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &fine.states_buffers[fine.current],
                0,
                &staging,
                byte_len::<f32>(fine_state_start)?,
                byte_len::<f32>(fine.state_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &student.positions_buffers[student.current],
                0,
                &staging,
                byte_len::<f32>(student_position_start)?,
                byte_len::<f32>(student.position_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &student.states_buffers[student.current],
                0,
                &staging,
                byte_len::<f32>(student_state_start)?,
                byte_len::<f32>(student.state_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &student.density_buffer,
                0,
                &staging,
                byte_len::<f32>(student_diagnostics_start)?,
                byte_len::<f32>(student.diagnostics_f32_len)?,
            );
            self.queue.submit(Some(encoder.finish()));
            Ok(WgpuPendingCoupledFineSnapshot {
                staging,
                fine_position_start,
                fine_state_start,
                student_position_start,
                student_state_start,
                student_diagnostics_start,
                readback_len,
            })
        })();
        self.restore_adaptive_diagnostics(fine, fine_saved);
        self.restore_adaptive_diagnostics(student, student_saved);
        if result.is_ok() && advance {
            finish_state_step(fine);
            finish_state_step(student);
        }
        result
    }

    pub(crate) fn read_coupled_fine_snapshot(
        &self,
        pending: WgpuPendingCoupledFineSnapshot,
        student: &WgpuAutomataState,
    ) -> AutomataResult<WgpuCoupledFineSnapshot> {
        let values = read_f32_buffer(&self.device, &pending.staging, pending.readback_len)?;
        Ok(WgpuCoupledFineSnapshot {
            fine_base_update: values[..pending.fine_position_start].to_vec(),
            fine_positions: unflatten_positions(
                &values[pending.fine_position_start..pending.fine_state_start],
            )?,
            fine_states: values[pending.fine_state_start..pending.student_position_start].to_vec(),
            student_positions: unflatten_positions(
                &values[pending.student_position_start..pending.student_state_start],
            )?,
            student_states: values[pending.student_state_start..pending.student_diagnostics_start]
                .to_vec(),
            student_diagnostics: self.decode_adaptive_diagnostics(
                student,
                &values[pending.student_diagnostics_start..],
            )?,
        })
    }
}

fn encode_recenter(
    encoder: &mut wgpu::CommandEncoder,
    coupling: &WgpuCoupledFineRecenter,
    fine: &WgpuAutomataState,
    student: &WgpuAutomataState,
) -> AutomataResult<()> {
    encode_recenter_current(encoder, coupling, fine.current, student.current)
}

fn encode_recenter_current(
    encoder: &mut wgpu::CommandEncoder,
    coupling: &WgpuCoupledFineRecenter,
    fine_current: usize,
    student_current: usize,
) -> AutomataResult<()> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("burn_automata_coupled_fine_recenter_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&coupling.pipeline);
    pass.set_bind_group(0, &coupling.bind_groups[fine_current][student_current], &[]);
    pass.dispatch_workgroups(dispatch_groups(coupling.total_students)?, 1, 1);
    drop(pass);
    Ok(())
}

fn encode_state_step(
    executor: &WgpuAutomataExecutor,
    encoder: &mut wgpu::CommandEncoder,
    state: &WgpuAutomataState,
) -> AutomataResult<()> {
    encode_state_step_current(executor, encoder, state, state.current)
}

fn encode_state_step_current(
    executor: &WgpuAutomataExecutor,
    encoder: &mut wgpu::CommandEncoder,
    state: &WgpuAutomataState,
    current: usize,
) -> AutomataResult<()> {
    let bind_group = &state.step_bind_groups[current];
    let grid_bind_group = &state.grid_bind_groups[current];
    executor.encode_grid_density_passes(encoder, state, grid_bind_group, bind_group)?;
    executor.encode_update_pass(encoder, state, bind_group)
}

fn finish_state_step(state: &mut WgpuAutomataState) {
    state.current = 1 - state.current;
    state.step_index = state.step_index.wrapping_add(1);
}

fn validate_coupled_shapes(
    fine: &WgpuAutomataState,
    student: &WgpuAutomataState,
    member_offsets: &[u32],
    member_leaves: &[u32],
) -> AutomataResult<()> {
    if fine.batch_size != student.batch_size
        || fine.state_f32_len / fine.total != student.state_f32_len / student.total
        || member_offsets.len() != student.total + 1
        || member_offsets.first().copied() != Some(0)
        || member_offsets.last().copied().map(|value| value as usize) != Some(member_leaves.len())
        || member_leaves.len() != fine.total
        || member_leaves
            .iter()
            .any(|leaf| *leaf as usize >= fine.total)
    {
        return Err(AutomataError::InvalidArgument(
            "coupled fine/student WGPU mapping has an incompatible shape".to_string(),
        ));
    }
    Ok(())
}

fn validate_coupling(
    coupling: &WgpuCoupledFineRecenter,
    fine: &WgpuAutomataState,
    student: &WgpuAutomataState,
) -> AutomataResult<()> {
    if fine.batch_size != coupling.batch_size
        || student.batch_size != coupling.batch_size
        || fine.particle_count != coupling.fine_count
        || student.particle_count != coupling.student_count
        || fine.state_f32_len / fine.total != coupling.state_dims
        || student.state_f32_len / student.total != coupling.state_dims
        || student.total != coupling.total_students
    {
        return Err(AutomataError::InvalidArgument(
            "coupled fine/student WGPU state changed after coupling creation".to_string(),
        ));
    }
    Ok(())
}
