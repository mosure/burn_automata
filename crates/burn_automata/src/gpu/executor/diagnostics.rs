use super::*;

pub(super) struct AdaptiveDiagnosticState {
    step_index: u32,
    local_rule_mode: WgpuAdaptiveLocalRuleMode,
    local_hidden_start: u32,
    local_residual_scale: f32,
    base_footprint: f32,
    shepard_epsilon: f32,
    moment_regularization: f32,
    moment_condition_limit: f32,
    max_neighbors: u32,
    pair_scale_power: f32,
}

pub(crate) struct WgpuPendingAdaptiveDiagnostics {
    staging: wgpu::Buffer,
    readback_len: usize,
    saved: AdaptiveDiagnosticState,
}

impl WgpuAutomataExecutor {
    pub(crate) fn step_state_many_with_paired_local_detail(
        &self,
        state: &mut WgpuAutomataState,
        steps: usize,
        topology_step_offsets: &[usize],
        split_radius_scale: f32,
        merge_detail_scale: f32,
        min_relative_gain: f32,
    ) -> AutomataResult<usize> {
        let steps = steps.max(1);
        if !matches!(
            state.neighbor_mode,
            WgpuNeighborMode::CooperativeSortedCells
                | WgpuNeighborMode::SubgroupCooperativeSortedCells
        ) || topology_step_offsets
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || topology_step_offsets
                .iter()
                .any(|offset| *offset == 0 || *offset > steps)
        {
            return Err(AutomataError::InvalidArgument(
                "batched paired topology requires sorted in-range step offsets and cooperative sorted cells"
                    .to_owned(),
            ));
        }
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_SPLIT_RADIUS_SCALE,
            split_radius_scale.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_MERGE_DETAIL_SCALE,
            merge_detail_scale.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN,
            min_relative_gain.to_bits(),
        );
        self.prepare_step_indices(state, steps)?;
        let mut topology_modes = vec![0_u32; steps];
        for offset in topology_step_offsets {
            topology_modes[*offset - 1] = 2;
        }
        let topology_modes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_paired_topology_step_modes"),
            size: byte_len::<u32>(topology_modes.len())?,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &topology_modes_buffer,
            0,
            bytemuck::cast_slice(&topology_modes),
        );

        const COMMAND_CHUNK_STEPS: usize = 64;

        let stable_sort_was_enabled = state.stable_sorted_cells_enabled;
        let mut current = state.current;
        let step_index_offset =
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let topology_mode_offset = (PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY
            * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        for chunk_start in (0..steps).step_by(COMMAND_CHUNK_STEPS) {
            let chunk_end = (chunk_start + COMMAND_CHUNK_STEPS).min(steps);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_batched_paired_topology_encoder"),
                });
            for (step_index, topology_mode) in topology_modes[chunk_start..chunk_end]
                .iter()
                .copied()
                .enumerate()
                .map(|(offset, mode)| (chunk_start + offset, mode))
            {
                encoder.copy_buffer_to_buffer(
                    state.step_indices_buffer.as_ref().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "step-index buffer is unavailable".to_owned(),
                        )
                    })?,
                    byte_len::<u32>(step_index)?,
                    &state.params_buffer,
                    step_index_offset,
                    std::mem::size_of::<u32>() as wgpu::BufferAddress,
                );
                encoder.copy_buffer_to_buffer(
                    &topology_modes_buffer,
                    byte_len::<u32>(step_index)?,
                    &state.params_buffer,
                    topology_mode_offset,
                    std::mem::size_of::<u32>() as wgpu::BufferAddress,
                );
                state.stable_sorted_cells_enabled =
                    stable_sort_was_enabled || (state.material_enabled && topology_mode == 2);
                let bind_group = &state.step_bind_groups[current];
                let grid_bind_group = &state.grid_bind_groups[current];
                self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
                self.encode_update_pass(&mut encoder, state, bind_group)?;
                current = 1 - current;
                if topology_mode == 2 {
                    let topology_bind_group = &state.step_bind_groups[current];
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("burn_automata_batched_paired_topology_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.paired_local_detail_topology_pipeline);
                    pass.set_bind_group(0, topology_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    drop(pass);
                    current = 1 - current;
                }
            }
            self.queue.submit(Some(encoder.finish()));
        }
        state.stable_sorted_cells_enabled = stable_sort_was_enabled;
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 0);
        state.current = current;
        state.step_index = state
            .step_index
            .wrapping_add(u32_checked(steps, "batched paired-topology step count")?);
        Ok(steps)
    }

    pub(super) fn begin_base_update_diagnostics(
        &self,
        state: &mut WgpuAutomataState,
    ) -> AutomataResult<AdaptiveDiagnosticState> {
        if !state.material_enabled
            || state.spatial_dims != 2
            || state.diagnostics_f32_len <= state.total
            || state.adaptive_local_rule_mode != WgpuAdaptiveLocalRuleMode::Disabled
        {
            return Err(AutomataError::InvalidArgument(
                "base-update diagnostics require a 2D material state without an adaptive local rule"
                    .to_owned(),
            ));
        }
        let saved = adaptive_diagnostic_state(state);
        self.write_param_u32(state, PARAM_ADAPTIVE_DIAGNOSTICS_ENABLED, 1);
        self.write_param_u32(state, PARAM_DT, 0.0_f32.to_bits());
        self.write_param_u32(state, PARAM_UPDATE_PROB, 1.0_f32.to_bits());
        Ok(saved)
    }

    pub(super) fn advance_state_during_diagnostics(&self, state: &WgpuAutomataState) {
        self.write_param_u32(state, PARAM_DT, state.dt.to_bits());
        self.write_param_u32(state, PARAM_UPDATE_PROB, state.update_prob.to_bits());
    }

    /// Evaluates adaptive perception at the resident final state without
    /// advancing dynamics. Ordinary rollout keeps diagnostics disabled; this
    /// one bounded pass replaces the CPU neighborhood reconstruction used by
    /// learned hierarchy restriction.
    pub(crate) fn capture_adaptive_diagnostics(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>, WgpuAdaptiveDiagnostics)> {
        self.capture_adaptive_diagnostics_with(state, base_footprint, config, |executor, state| {
            executor.read_positions_states_adaptive_diagnostics(state)
        })
    }

    /// Captures only controller inputs for a state whose host-side particles
    /// are already current, avoiding redundant position and state readback.
    pub(crate) fn capture_adaptive_diagnostics_only(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
    ) -> AutomataResult<WgpuAdaptiveDiagnostics> {
        let pending =
            self.begin_capture_adaptive_diagnostics_only(state, base_footprint, config)?;
        self.finish_capture_adaptive_diagnostics_only(state, pending)
    }

    /// Queues an adaptive diagnostic pass and its readback copy without waiting
    /// for the device. Callers can enqueue independent cut sizes back-to-back so
    /// the GPU does not drain between every diagnostic batch.
    pub(crate) fn begin_capture_adaptive_diagnostics_only(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
    ) -> AutomataResult<WgpuPendingAdaptiveDiagnostics> {
        let saved = self.begin_adaptive_diagnostics(state, base_footprint, config)?;
        let result = (|| {
            self.step_state(state)?;
            self.enqueue_adaptive_diagnostics_readback(state)
        })();
        match result {
            Ok((staging, readback_len)) => Ok(WgpuPendingAdaptiveDiagnostics {
                staging,
                readback_len,
                saved,
            }),
            Err(error) => {
                self.restore_adaptive_diagnostics(state, saved);
                Err(error)
            }
        }
    }

    pub(crate) fn finish_capture_adaptive_diagnostics_only(
        &self,
        state: &mut WgpuAutomataState,
        pending: WgpuPendingAdaptiveDiagnostics,
    ) -> AutomataResult<WgpuAdaptiveDiagnostics> {
        let result = read_f32_buffer(&self.device, &pending.staging, pending.readback_len)
            .and_then(|values| self.decode_adaptive_diagnostics(state, &values));
        self.restore_adaptive_diagnostics(state, pending.saved);
        result
    }

    /// Applies the fixed-budget adaptive Target2D topology operation without
    /// synchronizing resident particle state through the host.
    pub(crate) fn step_state_capturing_local_detail(
        &self,
        state: &mut WgpuAutomataState,
    ) -> AutomataResult<()> {
        if !matches!(
            state.neighbor_mode,
            WgpuNeighborMode::CooperativeSortedCells
                | WgpuNeighborMode::SubgroupCooperativeSortedCells
        ) {
            return Err(AutomataError::InvalidArgument(
                "paired topology feature capture requires cooperative sorted cells".to_owned(),
            ));
        }
        let stable_sort_was_enabled = state.stable_sorted_cells_enabled;
        self.set_stable_sorted_cells_enabled(state, true);
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 2);
        let result = self.step_state(state);
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 0);
        self.set_stable_sorted_cells_enabled(state, stable_sort_was_enabled);
        result
    }

    /// Populates reserved rows with canonical 2D children without reading or
    /// recreating resident state. The caller advances the active prefix only
    /// after this ordered queue submission has completed logically.
    pub(crate) fn apply_resident_bootstrap_splits(
        &self,
        state: &mut WgpuAutomataState,
        event_count: usize,
        bandwidth_exponent: f32,
        render_exponent: f32,
    ) -> AutomataResult<()> {
        const MAX_EVENTS: usize = 256;
        let next_count = state
            .particle_count
            .checked_add(event_count.saturating_mul(3))
            .ok_or_else(|| {
                AutomataError::InvalidArgument("resident bootstrap count overflow".to_owned())
            })?;
        if state.batch_size != 1
            || state.spatial_dims != 2
            || !state.material_enabled
            || !(1..=MAX_EVENTS).contains(&event_count)
            || next_count > state.particle_capacity
            || !bandwidth_exponent.is_finite()
            || bandwidth_exponent < 0.0
            || !render_exponent.is_finite()
            || render_exponent <= 0.0
        {
            return Err(AutomataError::InvalidArgument(format!(
                "resident bootstrap requires one 2D material trajectory, 1..={MAX_EVENTS} events within capacity, and finite non-negative scale exponents",
            )));
        }
        self.write_param_u32(
            state,
            PARAM_RESIDENT_BOOTSTRAP_EVENT_COUNT,
            u32_checked(event_count, "resident bootstrap event count")?,
        );
        self.write_param_u32(
            state,
            PARAM_RESIDENT_BOOTSTRAP_BANDWIDTH_EXPONENT,
            bandwidth_exponent.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_RESIDENT_BOOTSTRAP_RENDER_EXPONENT,
            render_exponent.to_bits(),
        );
        let bind_group = &state.step_bind_groups[state.current];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_resident_bootstrap_split_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("burn_automata_resident_bootstrap_split_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_bootstrap_split_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        state.current = 1 - state.current;
        Ok(())
    }

    /// Applies the fixed-budget adaptive Target2D topology operation without
    /// synchronizing resident particle state through the host.
    pub(crate) fn apply_paired_local_detail_topology(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
        split_radius_scale: f32,
        merge_detail_scale: f32,
        min_relative_gain: f32,
        detail_already_captured: bool,
    ) -> AutomataResult<()> {
        if state.batch_size != 1 || state.spatial_dims != 2 || !state.material_enabled {
            return Err(AutomataError::InvalidArgument(
                "resident paired local-detail topology requires one 2D material trajectory"
                    .to_owned(),
            ));
        }
        if !split_radius_scale.is_finite()
            || split_radius_scale < 0.0
            || !merge_detail_scale.is_finite()
            || merge_detail_scale < 0.0
            || !min_relative_gain.is_finite()
            || !(0.0..=1.0).contains(&min_relative_gain)
        {
            return Err(AutomataError::InvalidArgument(
                "resident paired local-detail topology scales must be finite and non-negative, and relative gain must be in [0, 1]"
                    .to_owned(),
            ));
        }
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_SPLIT_RADIUS_SCALE,
            split_radius_scale.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_MERGE_DETAIL_SCALE,
            merge_detail_scale.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN,
            min_relative_gain.to_bits(),
        );
        self.apply_local_detail_topology_kernel(
            state,
            base_footprint,
            config,
            detail_already_captured,
            &self.paired_local_detail_topology_pipeline,
            "burn_automata_paired_local_detail_topology_encoder",
            "burn_automata_paired_local_detail_topology_pass",
        )
    }

    /// Relocates fixed graded material slots using resident local-detail
    /// diagnostics. The kernel keeps row count and material metadata fixed and
    /// conserves represented centroid/state without a host synchronization.
    pub(crate) fn apply_continuous_local_detail_topology(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
        min_relative_gain: f32,
        event_budget: usize,
        detail_already_captured: bool,
    ) -> AutomataResult<()> {
        if state.batch_size != 1 || state.spatial_dims != 2 || !state.material_enabled {
            return Err(AutomataError::InvalidArgument(
                "resident continuous local-detail topology requires one 2D material trajectory"
                    .to_owned(),
            ));
        }
        if !min_relative_gain.is_finite() || !(0.0..=1.0).contains(&min_relative_gain) {
            return Err(AutomataError::InvalidArgument(
                "resident continuous local-detail relative gain must be finite and in [0, 1]"
                    .to_owned(),
            ));
        }
        if !(1..=crate::adaptive::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES).contains(&event_budget) {
            return Err(AutomataError::InvalidArgument(format!(
                "resident continuous local-detail topology event budget must be in 1..={}",
                crate::adaptive::CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES,
            )));
        }
        self.write_param_u32(
            state,
            PARAM_PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN,
            min_relative_gain.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_CONTINUOUS_TOPOLOGY_EVENT_BUDGET,
            event_budget as u32,
        );
        self.apply_local_detail_topology_kernel(
            state,
            base_footprint,
            config,
            detail_already_captured,
            &self.continuous_local_detail_topology_pipeline,
            "burn_automata_continuous_local_detail_topology_encoder",
            "burn_automata_continuous_local_detail_topology_pass",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_local_detail_topology_kernel(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
        detail_already_captured: bool,
        pipeline: &wgpu::ComputePipeline,
        encoder_label: &'static str,
        pass_label: &'static str,
    ) -> AutomataResult<()> {
        if detail_already_captured {
            return self.submit_local_detail_topology(state, pipeline, encoder_label, pass_label);
        }
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 1);

        let stable_sort_was_enabled = state.stable_sorted_cells_enabled;
        self.set_stable_sorted_cells_enabled(state, true);
        let result = (|| {
            let saved = self.begin_adaptive_diagnostics(state, base_footprint, config)?;
            let encoded = (|| {
                self.write_step_index(state);
                self.rebuild_bvh_if_needed(state)?;
                self.build_gpu_bvh_if_needed(state)?;
                let bind_group = &state.step_bind_groups[state.current];
                let grid_bind_group = &state.grid_bind_groups[state.current];
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(encoder_label),
                        });
                self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
                self.encode_adaptive_local_pass(&mut encoder, state, bind_group)?;
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(pass_label),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                state.current = 1 - state.current;
                Ok(())
            })();
            self.restore_adaptive_diagnostics(state, saved);
            encoded
        })();
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 0);
        self.set_stable_sorted_cells_enabled(state, stable_sort_was_enabled);
        result
    }

    fn submit_local_detail_topology(
        &self,
        state: &mut WgpuAutomataState,
        pipeline: &wgpu::ComputePipeline,
        encoder_label: &'static str,
        pass_label: &'static str,
    ) -> AutomataResult<()> {
        let bind_group = &state.step_bind_groups[state.current];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(encoder_label),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pass_label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        state.current = 1 - state.current;
        Ok(())
    }

    pub(crate) fn capture_teacher_snapshot(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
    ) -> AutomataResult<WgpuTeacherSnapshot> {
        let saved = self.begin_adaptive_diagnostics(state, base_footprint, config)?;
        let result: AutomataResult<_> = (|| {
            self.write_step_index(state);
            self.rebuild_bvh_if_needed(state)?;
            self.build_gpu_bvh_if_needed(state)?;

            let state_start = state.position_f32_len;
            let base_start = state_start + state.state_f32_len;
            let feature_len = state.total * state.feature_dims;
            let normalized_start = base_start + feature_len;
            let update_start = normalized_start + feature_len;
            let update_len = state.total * state.output_dims;
            let model_update_start = update_start + update_len;
            let spacing_start = model_update_start + update_len;
            let degree_start = spacing_start + state.total;
            let readback_len = degree_start.checked_add(state.total).ok_or_else(|| {
                AutomataError::InvalidArgument("teacher snapshot readback size overflow".to_owned())
            })?;
            let diagnostic_start = state.total;
            let diagnostic_len = readback_len - base_start;
            let staging = staging_read_buffer(
                &self.device,
                "burn_automata_teacher_snapshot_staging",
                readback_len,
            )?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("burn_automata_teacher_snapshot_encoder"),
                });
            let bind_group = &state.step_bind_groups[state.current];
            let grid_bind_group = &state.grid_bind_groups[state.current];
            self.encode_grid_density_passes(&mut encoder, state, grid_bind_group, bind_group)?;
            self.encode_update_pass(&mut encoder, state, bind_group)?;
            let next = 1 - state.current;
            encoder.copy_buffer_to_buffer(
                &state.positions_buffers[next],
                0,
                &staging,
                0,
                byte_len::<f32>(state.position_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &state.states_buffers[next],
                0,
                &staging,
                byte_len::<f32>(state_start)?,
                byte_len::<f32>(state.state_f32_len)?,
            );
            encoder.copy_buffer_to_buffer(
                &state.density_buffer,
                byte_len::<f32>(diagnostic_start)?,
                &staging,
                byte_len::<f32>(base_start)?,
                byte_len::<f32>(diagnostic_len)?,
            );
            self.queue.submit(Some(encoder.finish()));
            state.current = next;
            state.step_index = state.step_index.wrapping_add(1);
            Ok((staging, state_start, base_start, readback_len))
        })();
        self.restore_adaptive_diagnostics(state, saved);
        let (staging, state_start, base_start, readback_len) = result?;
        let values = read_f32_buffer(&self.device, &staging, readback_len)?;
        let feature_len = state.total * state.feature_dims;
        let normalized_start = base_start + feature_len;
        let update_start = normalized_start + feature_len;
        let model_update_start = update_start + state.total * state.output_dims;
        let spacing_start = model_update_start + state.total * state.output_dims;
        let degree_start = spacing_start + state.total;
        Ok(WgpuTeacherSnapshot {
            positions: unflatten_positions(&values[..state_start])?,
            states: values[state_start..base_start].to_vec(),
            base_features: values[base_start..normalized_start].to_vec(),
            normalized_features: values[normalized_start..update_start].to_vec(),
            base_update: values[update_start..model_update_start].to_vec(),
            observed_spacing: values[spacing_start..degree_start].to_vec(),
            accepted_degree: values[degree_start..]
                .iter()
                .map(|value| value.max(0.0).round() as usize)
                .collect(),
        })
    }

    fn capture_adaptive_diagnostics_with<T>(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
        readback: impl FnOnce(&Self, &WgpuAutomataState) -> AutomataResult<T>,
    ) -> AutomataResult<T> {
        let saved = self.begin_adaptive_diagnostics(state, base_footprint, config)?;

        let result = (|| {
            self.step_state(state)?;
            readback(self, state)
        })();

        self.restore_adaptive_diagnostics(state, saved);
        result
    }

    pub(super) fn begin_adaptive_diagnostics(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        config: AdaptivePerceptionConfig,
    ) -> AutomataResult<AdaptiveDiagnosticState> {
        config.validate()?;
        if !state.material_enabled
            || state.spatial_dims != 2
            || state.diagnostics_f32_len <= state.total
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive diagnostics require a 2D material WGPU state".to_owned(),
            ));
        }

        let saved = adaptive_diagnostic_state(state);
        let (diagnostic_mode, diagnostic_hidden_start) = match state.adaptive_local_rule_mode {
            WgpuAdaptiveLocalRuleMode::Disabled => {
                (WgpuAdaptiveLocalRuleMode::Residual, state.hidden_dims)
            }
            WgpuAdaptiveLocalRuleMode::Residual => (
                WgpuAdaptiveLocalRuleMode::Residual,
                state.adaptive_local_hidden_start as usize,
            ),
            WgpuAdaptiveLocalRuleMode::NormalizedExposureResidual => (
                WgpuAdaptiveLocalRuleMode::NormalizedExposureResidual,
                state.adaptive_local_hidden_start as usize,
            ),
            WgpuAdaptiveLocalRuleMode::CoarseReplacement => (
                WgpuAdaptiveLocalRuleMode::CoarseReplacement,
                state.adaptive_local_hidden_start as usize,
            ),
            WgpuAdaptiveLocalRuleMode::CompatibleResidual => (
                WgpuAdaptiveLocalRuleMode::CompatibleResidual,
                state.adaptive_local_hidden_start as usize,
            ),
            WgpuAdaptiveLocalRuleMode::NormalizedPrimary => {
                (WgpuAdaptiveLocalRuleMode::NormalizedPrimary, 0)
            }
        };
        self.configure_state_adaptive_local_rule(
            state,
            diagnostic_mode,
            diagnostic_hidden_start,
            state.adaptive_local_residual_scale,
            base_footprint,
            state.adaptive_reference_footprint,
            config.shepard_epsilon,
            config.moment_regularization,
            config.moment_condition_limit,
            match config.graph_policy {
                burn_automata_kernels::AdaptiveGraphPolicy::RawSupport => 0,
                burn_automata_kernels::AdaptiveGraphPolicy::DirectedTopK => config.max_neighbors,
                burn_automata_kernels::AdaptiveGraphPolicy::MutualTopK => {
                    return Err(AutomataError::InvalidArgument(
                        "adaptive WGPU diagnostics do not support mutual-top-k perception"
                            .to_owned(),
                    ));
                }
            },
            config.pair_scale_power,
        )?;

        self.write_param_u32(state, PARAM_ADAPTIVE_DIAGNOSTICS_ENABLED, 1);
        self.write_param_u32(state, PARAM_DT, 0.0_f32.to_bits());
        self.write_param_u32(state, PARAM_UPDATE_PROB, 1.0_f32.to_bits());
        let spacing = [
            config.min_bandwidth.to_bits(),
            config.max_bandwidth.to_bits(),
            config.spacing_target_neighbors.to_bits(),
            u32_checked(
                config.spacing_root_iterations,
                "adaptive spacing root iterations",
            )?,
        ];
        debug_assert_eq!(PARAM_ADAPTIVE_SPACING_MAX, PARAM_ADAPTIVE_SPACING_MIN + 1);
        debug_assert_eq!(
            PARAM_ADAPTIVE_SPACING_TARGET,
            PARAM_ADAPTIVE_SPACING_MIN + 2
        );
        debug_assert_eq!(
            PARAM_ADAPTIVE_SPACING_ROOT_ITERATIONS,
            PARAM_ADAPTIVE_SPACING_MIN + 3,
        );
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_SPACING_MIN * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&spacing),
        );
        Ok(saved)
    }

    pub(super) fn restore_adaptive_diagnostics(
        &self,
        state: &mut WgpuAutomataState,
        saved: AdaptiveDiagnosticState,
    ) {
        self.write_param_u32(state, PARAM_ADAPTIVE_DIAGNOSTICS_ENABLED, 0);
        self.write_param_u32(state, PARAM_PAIRED_TOPOLOGY_DIAGNOSTICS_ONLY, 0);
        self.write_param_u32(state, PARAM_DT, state.dt.to_bits());
        self.write_param_u32(state, PARAM_UPDATE_PROB, state.update_prob.to_bits());
        let adaptive = [
            saved.local_hidden_start,
            saved.local_residual_scale.to_bits(),
            saved.base_footprint.to_bits(),
            saved.shepard_epsilon.to_bits(),
            saved.moment_regularization.to_bits(),
            saved.moment_condition_limit.to_bits(),
            saved.max_neighbors,
        ];
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_LOCAL_HIDDEN_START * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&adaptive),
        );
        self.write_param_u32(
            state,
            PARAM_ADAPTIVE_PAIR_SCALE_POWER,
            saved.pair_scale_power.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_ADAPTIVE_LOCAL_RULE_MODE,
            saved.local_rule_mode.as_u32(),
        );
        state.adaptive_local_rule_mode = saved.local_rule_mode;
        state.adaptive_local_hidden_start = saved.local_hidden_start;
        state.adaptive_local_residual_scale = saved.local_residual_scale;
        state.adaptive_base_footprint = saved.base_footprint;
        state.adaptive_shepard_epsilon = saved.shepard_epsilon;
        state.adaptive_moment_regularization = saved.moment_regularization;
        state.adaptive_moment_condition_limit = saved.moment_condition_limit;
        state.adaptive_max_neighbors = saved.max_neighbors;
        state.adaptive_pair_scale_power = saved.pair_scale_power;
        state.step_index = saved.step_index;
    }

    fn read_positions_states_adaptive_diagnostics(
        &self,
        state: &WgpuAutomataState,
    ) -> AutomataResult<(Vec<[f32; 4]>, Vec<f32>, WgpuAdaptiveDiagnostics)> {
        let state_start = state.position_f32_len;
        let diagnostics_start = state_start
            .checked_add(state.state_f32_len)
            .ok_or_else(|| {
                AutomataError::InvalidArgument("adaptive readback size overflow".to_owned())
            })?;
        let readback_len = diagnostics_start
            .checked_add(state.diagnostics_f32_len)
            .ok_or_else(|| {
                AutomataError::InvalidArgument("adaptive readback size overflow".to_owned())
            })?;
        let staging = staging_read_buffer(
            &self.device,
            "burn_automata_adaptive_diagnostic_readback",
            readback_len,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_adaptive_diagnostics_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &state.positions_buffers[state.current],
            0,
            &staging,
            0,
            byte_len::<f32>(state.position_f32_len)?,
        );
        encoder.copy_buffer_to_buffer(
            &state.states_buffers[state.current],
            0,
            &staging,
            byte_len::<f32>(state_start)?,
            byte_len::<f32>(state.state_f32_len)?,
        );
        encoder.copy_buffer_to_buffer(
            &state.density_buffer,
            0,
            &staging,
            byte_len::<f32>(diagnostics_start)?,
            byte_len::<f32>(state.diagnostics_f32_len)?,
        );
        self.queue.submit(Some(encoder.finish()));

        let values = read_f32_buffer(&self.device, &staging, readback_len)?;
        Ok((
            unflatten_positions(&values[..state_start])?,
            values[state_start..diagnostics_start].to_vec(),
            self.decode_adaptive_diagnostics(state, &values[diagnostics_start..])?,
        ))
    }

    fn enqueue_adaptive_diagnostics_readback(
        &self,
        state: &WgpuAutomataState,
    ) -> AutomataResult<(wgpu::Buffer, usize)> {
        let staging = staging_read_buffer(
            &self.device,
            "burn_automata_adaptive_diagnostics_staging",
            state.diagnostics_f32_len,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("burn_automata_adaptive_diagnostics_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &state.density_buffer,
            0,
            &staging,
            0,
            byte_len::<f32>(state.diagnostics_f32_len)?,
        );
        self.queue.submit(Some(encoder.finish()));
        Ok((staging, state.diagnostics_f32_len))
    }

    pub(super) fn decode_adaptive_diagnostics(
        &self,
        state: &WgpuAutomataState,
        values: &[f32],
    ) -> AutomataResult<WgpuAdaptiveDiagnostics> {
        let total = state.total;
        let feature_values = total * state.feature_dims;
        let update_values = total * state.output_dims;
        let base_start = total;
        let normalized_start = base_start + feature_values;
        let update_start = normalized_start + feature_values;
        let model_update_start = update_start + update_values;
        let spacing_start = model_update_start + update_values;
        let degree_start = spacing_start + total;
        let coarse_exposure_start = degree_start + total;
        if coarse_exposure_start + total != values.len() {
            return Err(AutomataError::InvalidModel(format!(
                "adaptive diagnostic layout requires {} values, got {}",
                coarse_exposure_start + total,
                values.len(),
            )));
        }
        let accepted_degree = values[degree_start..coarse_exposure_start]
            .iter()
            .map(|value| value.max(0.0).round() as usize)
            .collect();
        Ok(WgpuAdaptiveDiagnostics {
            base_features: values[base_start..normalized_start].to_vec(),
            normalized_features: values[normalized_start..update_start].to_vec(),
            base_update: values[update_start..model_update_start].to_vec(),
            model_update: values[model_update_start..spacing_start].to_vec(),
            observed_spacing: values[spacing_start..degree_start].to_vec(),
            accepted_degree,
            coarse_exposure: values[coarse_exposure_start..].to_vec(),
            feature_dims: state.feature_dims,
            output_dims: state.output_dims,
        })
    }

    pub(super) fn write_param_u32(&self, state: &WgpuAutomataState, index: usize, value: u32) {
        self.queue.write_buffer(
            &state.params_buffer,
            (index * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::bytes_of(&value),
        );
    }
}

fn adaptive_diagnostic_state(state: &WgpuAutomataState) -> AdaptiveDiagnosticState {
    AdaptiveDiagnosticState {
        step_index: state.step_index,
        local_rule_mode: state.adaptive_local_rule_mode,
        local_hidden_start: state.adaptive_local_hidden_start,
        local_residual_scale: state.adaptive_local_residual_scale,
        base_footprint: state.adaptive_base_footprint,
        shepard_epsilon: state.adaptive_shepard_epsilon,
        moment_regularization: state.adaptive_moment_regularization,
        moment_condition_limit: state.adaptive_moment_condition_limit,
        max_neighbors: state.adaptive_max_neighbors,
        pair_scale_power: state.adaptive_pair_scale_power,
    }
}
