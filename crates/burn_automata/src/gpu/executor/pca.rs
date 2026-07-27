use std::borrow::Cow;

use super::{device::create_pca_pipelines, *};

const PCA_COMPONENTS: usize = 3;
const PCA_REDUCTION_WORKGROUP_SIZE: usize = 128;
const PCA_REDUCTION_ITEMS_PER_THREAD: usize = 4;
const PCA_BASIS_WORKGROUP_SIZE: usize = 32;

impl WgpuAutomataExecutor {
    /// Allocates a resident rolling PCA projector for one automata state shape.
    ///
    /// The projector owns only bounded statistics and projection scratch. It
    /// reads particle state directly from the resident ping-pong buffers.
    pub fn create_state_pca(
        &self,
        state: &WgpuAutomataState,
        config: WgpuStatePcaConfig,
    ) -> AutomataResult<WgpuStatePca> {
        self.pca_pipelines()?;
        validate_pca_config(config)?;
        let state_dims = state_dims(state)?;
        if !(PCA_COMPONENTS..=PCA_BASIS_WORKGROUP_SIZE).contains(&state_dims) {
            return Err(AutomataError::InvalidArgument(format!(
                "particle-state RGB PCA requires {PCA_COMPONENTS}..={PCA_BASIS_WORKGROUP_SIZE} channels, got {state_dims}"
            )));
        }
        let particle_capacity = state.allocation_total;
        let partial_capacity = pca_partial_count(particle_capacity);
        let params_buffer =
            uniform_buffer_u32(&self.device, "burn_automata_state_pca_params", &[0; 12]);
        let mean_offset = state_dims * partial_capacity;
        let components_offset = mean_offset + state_dims;
        let projected_offset = components_offset + state_dims * PCA_COMPONENTS;
        let candidate_offset = projected_offset + particle_capacity * PCA_COMPONENTS;
        let display_center_offset = candidate_offset + state_dims * PCA_COMPONENTS;
        let display_spread_offset = display_center_offset + PCA_COMPONENTS;
        let mut data = vec![0.0; display_spread_offset + PCA_COMPONENTS];
        data[components_offset..projected_offset]
            .copy_from_slice(&initial_pca_components(state_dims));
        data[display_spread_offset..display_spread_offset + PCA_COMPONENTS].fill(1.0);
        let data_buffer = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_pca_data",
            &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let pipelines = self.pca_pipelines()?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("burn_automata_state_pca_bind_group"),
            layout: &pipelines.bind_group_layout,
            entries: &[bind_entry(0, &params_buffer), bind_entry(1, &data_buffer)],
        });
        Ok(WgpuStatePca {
            config,
            state_dims,
            particle_capacity,
            partial_capacity,
            observed_frames: 0,
            last_particle_count: 0,
            update_count: 0,
            initialized: false,
            force_update: true,
            params: [0; 12],
            params_buffer,
            data_buffer,
            mean_offset,
            components_offset,
            display_center_offset,
            display_spread_offset,
            bind_group,
        })
    }

    pub(super) fn encode_state_pca_into_gaussians(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &WgpuAutomataState,
        gaussian_source_index: usize,
        gaussian: &WgpuGaussianBindGroup,
        pca: &mut WgpuStatePca,
    ) -> AutomataResult<()> {
        validate_pca_state(state, gaussian, pca)?;
        let source = state
            .gaussian_source_bind_groups
            .get(gaussian_source_index)
            .ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "invalid PCA Gaussian source index {gaussian_source_index}"
                ))
            })?;
        let pipelines = self.pca_pipelines()?;
        let active_partials = pca_partial_count(state.total);
        let should_fit = pca.force_update
            || !pca.initialized
            || pca.last_particle_count != state.total
            || pca.observed_frames.is_multiple_of(pca.config.update_every);
        let iterations = if should_fit {
            if pca.initialized {
                pca.config.update_iterations
            } else {
                pca.config.warmup_iterations
            }
        } else {
            0
        };
        let learning_rate = if pca.initialized {
            pca.config.learning_rate
        } else {
            pca.config.warmup_learning_rate
        };
        let params = [
            u32_checked(active_partials, "PCA partial count")?,
            u32_checked(pca.partial_capacity, "PCA partial capacity")?,
            0,
            u32::from(pca.initialized),
            learning_rate.to_bits(),
            pca.config.mean_momentum.to_bits(),
            pca.config.display_momentum.to_bits(),
            pca.config.display_clip_sigma.to_bits(),
            pca.config.epsilon.to_bits(),
            pca.config.display_std_floor.to_bits(),
            u32_checked(pca.particle_capacity, "PCA particle capacity")?,
            0,
        ];
        if params != pca.params {
            self.queue
                .write_buffer(&pca.params_buffer, 0, bytemuck::cast_slice(&params));
            pca.params = params;
        }

        let mut dispatch =
            |pipeline: &wgpu::ComputePipeline, label: &'static str, groups: (u32, u32, u32)| {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, source, &[]);
                pass.set_bind_group(1, &gaussian.bind_group, &[]);
                pass.set_bind_group(2, &pca.bind_group, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            };

        if should_fit {
            dispatch(
                &pipelines.partial_mean,
                "burn_automata_state_pca_partial_mean_pass",
                (
                    u32_checked(active_partials, "PCA partial dispatch")?,
                    u32_checked(pca.state_dims, "PCA feature dispatch")?,
                    1,
                ),
            );
            dispatch(
                &pipelines.finalize_mean,
                "burn_automata_state_pca_finalize_mean_pass",
                (u32_checked(pca.state_dims, "PCA mean dispatch")?, 1, 1),
            );
            for _ in 0..iterations {
                dispatch(
                    &pipelines.project_update,
                    "burn_automata_state_pca_project_update_pass",
                    (dispatch_groups(state.total)?, 1, 1),
                );
                dispatch(
                    &pipelines.oja_candidate,
                    "burn_automata_state_pca_oja_candidate_pass",
                    (
                        u32_checked(pca.state_dims, "PCA candidate feature dispatch")?,
                        PCA_COMPONENTS as u32,
                        1,
                    ),
                );
                dispatch(
                    &pipelines.stabilize_basis,
                    "burn_automata_state_pca_stabilize_basis_pass",
                    (1, 1, 1),
                );
            }
            pca.initialized = true;
            pca.force_update = false;
            pca.update_count = pca.update_count.saturating_add(1);
        }
        dispatch(
            &pipelines.project_update,
            "burn_automata_state_pca_project_display_pass",
            (dispatch_groups(state.total)?, 1, 1),
        );
        dispatch(
            &pipelines.display_stats,
            "burn_automata_state_pca_display_stats_pass",
            (PCA_COMPONENTS as u32, 1, 1),
        );
        dispatch(
            &pipelines.write_gaussian,
            "burn_automata_state_pca_write_gaussians_pass",
            (dispatch_groups(state.total)?, 1, 1),
        );
        pca.observed_frames = pca.observed_frames.wrapping_add(1);
        pca.last_particle_count = state.total;
        Ok(())
    }

    /// Reads the compact PCA state for tests and diagnostics. Viewer execution
    /// never calls this method.
    pub fn read_state_pca_snapshot(
        &self,
        pca: &WgpuStatePca,
    ) -> AutomataResult<WgpuStatePcaSnapshot> {
        let mean = self.read_storage_f32_range(
            &pca.data_buffer,
            pca.mean_offset,
            pca.state_dims,
            "burn_automata_state_pca_mean_staging",
        )?;
        let components = self.read_storage_f32_range(
            &pca.data_buffer,
            pca.components_offset,
            pca.state_dims * PCA_COMPONENTS,
            "burn_automata_state_pca_components_staging",
        )?;
        let center = self.read_storage_f32_range(
            &pca.data_buffer,
            pca.display_center_offset,
            PCA_COMPONENTS,
            "burn_automata_state_pca_display_center_staging",
        )?;
        let spread = self.read_storage_f32_range(
            &pca.data_buffer,
            pca.display_spread_offset,
            PCA_COMPONENTS,
            "burn_automata_state_pca_display_spread_staging",
        )?;
        Ok(WgpuStatePcaSnapshot {
            mean,
            components,
            display_center: [center[0], center[1], center[2]],
            display_spread: [spread[0], spread[1], spread[2]],
            update_count: pca.update_count,
        })
    }

    fn pca_pipelines(&self) -> AutomataResult<&WgpuPcaPipelines> {
        if self.gaussian_pipeline.is_none() {
            return Err(AutomataError::InvalidArgument(
                "WGPU executor was initialized without particle-state PCA support".to_owned(),
            ));
        }
        Ok(self.pca_pipelines.get_or_init(|| {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("burn_automata_state_pca"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                        "{}\n{}\n{}",
                        include_str!("../../gpu_step.wgsl"),
                        include_str!("../../gpu_pca.wgsl"),
                        burn_automata_kernels::PAIRED_LOCAL_DETAIL_TOPOLOGY_WGSL,
                    ))),
                });
            create_pca_pipelines(
                &self.device,
                &shader,
                &self.gaussian_source_bind_group_layout,
                &self.gaussian_bind_group_layout,
            )
        }))
    }
}

fn validate_pca_config(config: WgpuStatePcaConfig) -> AutomataResult<()> {
    let probabilities = [
        ("mean_momentum", config.mean_momentum),
        ("display_momentum", config.display_momentum),
    ];
    if config.update_every == 0
        || config.warmup_iterations == 0
        || config.update_iterations == 0
        || !config.warmup_learning_rate.is_finite()
        || config.warmup_learning_rate <= 0.0
        || config.warmup_learning_rate > 1.0
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.learning_rate > 1.0
        || probabilities
            .iter()
            .any(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !config.display_clip_sigma.is_finite()
        || config.display_clip_sigma <= 0.0
        || !config.display_std_floor.is_finite()
        || config.display_std_floor <= 0.0
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
    {
        return Err(AutomataError::InvalidArgument(format!(
            "invalid particle-state PCA config: {config:?}"
        )));
    }
    Ok(())
}

fn validate_pca_state(
    state: &WgpuAutomataState,
    gaussian: &WgpuGaussianBindGroup,
    pca: &WgpuStatePca,
) -> AutomataResult<()> {
    if gaussian.count < state.total {
        return Err(AutomataError::InvalidArgument(format!(
            "gaussian bind group count {} is smaller than PCA particle count {}",
            gaussian.count, state.total
        )));
    }
    let state_dims = state_dims(state)?;
    if state_dims != pca.state_dims || state.allocation_total > pca.particle_capacity {
        return Err(AutomataError::InvalidArgument(format!(
            "PCA shape {}x{} does not cover resident state {}x{}",
            pca.particle_capacity, pca.state_dims, state.allocation_total, state_dims
        )));
    }
    Ok(())
}

fn state_dims(state: &WgpuAutomataState) -> AutomataResult<usize> {
    state
        .state_f32_len
        .checked_div(state.total)
        .filter(|dims| *dims > 0)
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "particle-state PCA requires a non-empty resident state".to_owned(),
            )
        })
}

fn pca_partial_count(particle_count: usize) -> usize {
    particle_count
        .div_ceil(PCA_REDUCTION_WORKGROUP_SIZE * PCA_REDUCTION_ITEMS_PER_THREAD)
        .max(1)
}

fn initial_pca_components(state_dims: usize) -> Vec<f32> {
    let mut columns = [[0.0; PCA_BASIS_WORKGROUP_SIZE]; PCA_COMPONENTS];
    for component in 0..PCA_COMPONENTS {
        let (previous_columns, remaining_columns) = columns.split_at_mut(component);
        let column = &mut remaining_columns[0][..state_dims];
        for (feature, value) in column.iter_mut().enumerate() {
            let coordinate = (feature + 1) as f32;
            let frequency = (component * 17 + 5) as f32;
            *value = (coordinate * frequency * 0.754_877_7).sin()
                + (coordinate * (frequency + 11.0) * 0.569_840_3).cos();
        }
        for previous in previous_columns {
            let dot = column
                .iter()
                .zip(previous.iter())
                .map(|(value, previous)| value * previous)
                .sum::<f32>();
            for (value, previous) in column.iter_mut().zip(previous.iter()) {
                *value -= dot * previous;
            }
        }
        let norm = column.iter().map(|value| value.powi(2)).sum::<f32>().sqrt();
        for value in column {
            *value /= norm;
        }
    }
    let mut components = vec![0.0; state_dims * PCA_COMPONENTS];
    for (component, column) in columns.iter().enumerate() {
        for (feature, value) in column.iter().take(state_dims).enumerate() {
            components[feature * PCA_COMPONENTS + component] = *value;
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_components_are_orthonormal() {
        for state_dims in [3, 12, 24] {
            let components = initial_pca_components(state_dims);
            for lhs in 0..PCA_COMPONENTS {
                for rhs in 0..PCA_COMPONENTS {
                    let dot = (0..state_dims)
                        .map(|feature| {
                            components[feature * PCA_COMPONENTS + lhs]
                                * components[feature * PCA_COMPONENTS + rhs]
                        })
                        .sum::<f32>();
                    let expected = if lhs == rhs { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1.0e-5, "{state_dims}: {dot}");
                }
            }
        }
    }
}
