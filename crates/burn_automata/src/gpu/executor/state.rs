#![allow(clippy::too_many_arguments)]

use super::*;

impl WgpuAutomataExecutor {
    pub fn create_state(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            WgpuNeighborMode::Auto,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_update_prob(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode_and_update_prob(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            WgpuNeighborMode::Auto,
            update_prob,
            seed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_neighbor_mode(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_with_neighbor_mode_and_update_prob(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            1.0,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_state_with_neighbor_mode_and_update_prob(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_impl(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            update_prob,
            seed,
            None,
            None,
            None,
        )
    }

    /// Creates a resident represented-measure state for adaptive NPA inference.
    #[allow(clippy::too_many_arguments)]
    pub fn create_material_state_with_neighbor_mode_and_update_prob(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
        material: WgpuMaterialStateInit<'_>,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_impl(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            update_prob,
            seed,
            None,
            None,
            Some(material),
        )
    }

    /// Creates an adaptive material state with a larger resident row capacity.
    ///
    /// Only `particle_count` rows are active. The remaining rows are storage
    /// reserve for device-side topology and are never dispatched until the
    /// active count is advanced explicitly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_material_state_with_capacity(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        particle_count: usize,
        particle_capacity: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
        material: WgpuMaterialStateInit<'_>,
    ) -> AutomataResult<WgpuAutomataState> {
        self.create_state_impl(
            model,
            positions,
            states,
            1,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            update_prob,
            seed,
            None,
            Some(particle_capacity),
            Some(material),
        )
    }

    /// Creates a batched material state with one independent logical hash-grid
    /// partition and stochastic seed per trajectory.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_batched_material_state(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seeds: &[u64],
        material: WgpuMaterialStateInit<'_>,
    ) -> AutomataResult<WgpuAutomataState> {
        if seeds.len() != batch_size {
            return Err(AutomataError::InvalidArgument(format!(
                "batched WGPU state has {} lanes but {} seeds",
                batch_size,
                seeds.len(),
            )));
        }
        self.create_state_impl(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            neighbor_mode,
            update_prob,
            seeds.first().copied().unwrap_or_default(),
            Some(seeds),
            None,
            Some(material),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_state_impl(
        &self,
        model: &NpaModel,
        positions: &[[f32; 4]],
        states: &[f32],
        batch_size: usize,
        particle_count: usize,
        grid: &HashGridConfig,
        dt: f32,
        neighbor_mode: WgpuNeighborMode,
        update_prob: f32,
        seed: u64,
        independent_lane_seeds: Option<&[u64]>,
        particle_capacity: Option<usize>,
        material: Option<WgpuMaterialStateInit<'_>>,
    ) -> AutomataResult<WgpuAutomataState> {
        validate_gpu_step(
            model,
            positions,
            states,
            batch_size,
            particle_count,
            grid,
            dt,
            update_prob,
        )?;

        let total = positions.len();
        let particle_capacity = particle_capacity.unwrap_or(particle_count);
        if particle_capacity < particle_count {
            return Err(AutomataError::InvalidArgument(format!(
                "resident particle capacity {particle_capacity} is below active particle count {particle_count}",
            )));
        }
        let allocation_total = batch_size.checked_mul(particle_capacity).ok_or_else(|| {
            AutomataError::InvalidArgument("resident particle capacity overflow".to_owned())
        })?;
        let material = material_state_values(
            material,
            total,
            particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
        )?;

        let (bucket_capacity, resolved_neighbor_mode) =
            resolve_neighbor_mode_for_state(grid, particle_count, positions, neighbor_mode)?;
        let resolved_neighbor_mode = promote_auto_subgroup_mode(
            neighbor_mode,
            resolved_neighbor_mode,
            self.subgroup_cooperative_supported,
        );
        if batch_size > 1 && is_bvh_neighbor_mode(resolved_neighbor_mode) {
            return Err(AutomataError::InvalidArgument(
                "batched WGPU trajectories require a hash-grid neighbor mode".to_owned(),
            ));
        }
        if allocation_total != total && is_bvh_neighbor_mode(resolved_neighbor_mode) {
            return Err(AutomataError::InvalidArgument(
                "resident row reserve currently requires a hash-grid neighbor mode".to_owned(),
            ));
        }
        if resolved_neighbor_mode == WgpuNeighborMode::SubgroupCooperativeSortedCells
            && !self.subgroup_cooperative_supported
        {
            return Err(AutomataError::InvalidArgument(
                "WGPU subgroup cooperative sorted cells requires fixed 32-wide subgroup support; current adapter/device does not expose it"
                    .to_owned(),
            ));
        }
        let bvh_leaf_count = match resolved_neighbor_mode {
            WgpuNeighborMode::GpuBvh { .. }
            | WgpuNeighborMode::GpuLbvh { .. }
            | WgpuNeighborMode::GpuMortonLbvh { .. } => {
                bvh_leaf_count_pow2(total, bucket_capacity)?
            }
            _ => 0,
        };
        let bvh_levels = bvh_level_count(bvh_leaf_count);
        let bvh_sort_count = match resolved_neighbor_mode {
            WgpuNeighborMode::GpuMortonLbvh { .. } => bvh_sort_count_pow2(total)?,
            _ => 0,
        };
        let mut params = gpu_params(
            model,
            total,
            batch_size,
            particle_count,
            grid,
            dt,
            bucket_capacity,
            resolved_neighbor_mode,
            update_prob,
            seed,
        )?;
        params[PARAM_MATERIAL_ENABLED] = u32::from(material.enabled);
        params[PARAM_MEAN_REPRESENTED_MEASURE] = material.mean_measure.to_bits();
        params[PARAM_DISPLAY_SCALE_PER_FOOTPRINT] = material.display_scale.to_bits();
        params[PARAM_RENDER_TRANSITION_STEPS] = material.render_transition_steps;
        params[PARAM_ADAPTIVE_REFERENCE_FOOTPRINT] = 1.0_f32.to_bits();
        params[PARAM_ADAPTIVE_PAIR_SCALE_POWER] = 8.0_f32.to_bits();
        params[PARAM_MAX_MATERIAL_BANDWIDTH] = if material.enabled {
            material.max_bandwidth
        } else {
            grid.eps
        }
        .to_bits();
        params[PARAM_RENDER_TRANSITION_START_STEP] = 0;
        params[PARAM_RESIDENT_CAPACITY] =
            u32_checked(allocation_total, "resident allocation total")?;
        params[PARAM_BVH_LEAF_COUNT] = u32_checked(bvh_leaf_count, "BVH leaf count")?;
        params[PARAM_BVH_SORT_COUNT] = u32_checked(bvh_sort_count, "BVH sort count")?;
        let lane_seed_values =
            gpu_lane_seeds(batch_size, particle_count, seed, independent_lane_seeds)?;
        params[PARAM_LANE_SEEDS_START..PARAM_LANE_SEEDS_START + lane_seed_values.len()]
            .copy_from_slice(&lane_seed_values);
        let mut position_values = flatten_positions(positions);
        let weights = packed_weights(model);
        let weights_f32_len = weights.len();
        let position_f32_len = position_values.len();
        let state_f32_len = states.len();
        position_values.resize(allocation_total * 4, 0.0);
        let mut state_values = states.to_vec();
        state_values.resize(allocation_total * model.config.state_dims, 0.0);
        let spatial_cell_count = grid.cell_count();
        let requested_support_bin_count = material.support_bin_count;
        let support_capacity_layout = resolve_support_bin_grid_layout(
            spatial_cell_count,
            batch_size,
            allocation_total,
            bucket_capacity,
            resolved_neighbor_mode,
            if material.enabled {
                requested_support_bin_count
            } else {
                1
            },
        )?;
        let support_bin_capacity = support_capacity_layout.support_bin_count;
        let support_bin_count = if support_bin_capacity > 1
            && (material.support_bins_forced
                || should_activate_support_bins(
                    grid,
                    particle_count,
                    positions,
                    &material.bandwidth,
                    material.support_bin_min,
                    material.support_bin_max,
                    material.support_bin_ratio,
                )) {
            support_bin_capacity
        } else {
            1
        };
        let cell_count = spatial_cell_count
            .checked_mul(batch_size)
            .and_then(|value| value.checked_mul(support_bin_count))
            .ok_or_else(|| {
                AutomataError::InvalidArgument("active support-bin cell count overflow".to_owned())
            })?;
        params[PARAM_CELL_COUNT] = u32_checked(cell_count, "support-bin cell count")?;
        params[PARAM_SUPPORT_BIN_COUNT] = u32_checked(support_bin_count, "support bin count")?;
        params[PARAM_SPATIAL_CELL_COUNT] = u32_checked(spatial_cell_count, "spatial cell count")?;
        params[PARAM_SUPPORT_BIN_MIN] = material.support_bin_min.to_bits();
        params[PARAM_SUPPORT_BIN_MAX] = material.support_bin_max.to_bits();
        params[PARAM_SUPPORT_BIN_RATIO] = material.support_bin_ratio.to_bits();
        let grid_storage_len = support_capacity_layout.storage_len;
        let grid_clear_len =
            grid_clear_len_for_mode(cell_count, bucket_capacity, resolved_neighbor_mode)?;
        let fused_sorted_grid_enabled = spatial_cell_count
            .checked_mul(support_bin_count)
            .is_some_and(|cells| cells <= burn_automata_kernels::FUSED_SORTED_GRID_MAX_CELLS)
            && matches!(
                resolved_neighbor_mode,
                WgpuNeighborMode::SortedCells
                    | WgpuNeighborMode::CooperativeSortedCells
                    | WgpuNeighborMode::SubgroupCooperativeSortedCells
            );
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let params_buffer = uniform_buffer_u32(&self.device, "burn_automata_state_params", &params);
        let positions_a = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_positions_a",
            &position_values,
            state_usage,
        );
        let positions_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_positions_b"),
            size: byte_len::<f32>(position_values.len())?,
            usage: state_usage,
            mapped_at_creation: false,
        });
        let states_a = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_states_a",
            &state_values,
            state_usage,
        );
        let states_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_states_b"),
            size: byte_len::<f32>(state_values.len())?,
            usage: state_usage,
            mapped_at_creation: false,
        });
        let mut resident_weights =
            Vec::with_capacity(weights.len() + 2 * MAX_CLOSURE_WEIGHT_FLOATS);
        resident_weights.extend_from_slice(&weights);
        resident_weights.resize(weights.len() + 2 * MAX_CLOSURE_WEIGHT_FLOATS, 0.0);
        let weights_buffer = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_weights",
            &resident_weights,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let linked_grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_linked_grid"),
            size: byte_len::<u32>(grid_storage_len)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let WgpuNeighborMode::Bvh { leaf_size } = resolved_neighbor_mode {
            let storage = build_bvh_storage_u32(positions, model.config.spatial_dims, leaf_size)?;
            if storage.len() > grid_storage_len {
                return Err(AutomataError::InvalidArgument(format!(
                    "BVH storage len {} exceeds allocated grid storage len {}",
                    storage.len(),
                    grid_storage_len
                )));
            }
            self.queue
                .write_buffer(&linked_grid_buffer, 0, bytemuck::cast_slice(&storage));
        }
        let indirect_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_indirect_args"),
            size: byte_len::<u32>(3)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let diagnostics_f32_len = if material.enabled && model.config.spatial_dims == 2 {
            adaptive_diagnostics_f32_len(
                total,
                model.config.perception_dims(),
                model.config.update_dims(),
            )?
        } else {
            total
        };
        let diagnostics_capacity_f32_len = if material.enabled && model.config.spatial_dims == 2 {
            adaptive_diagnostics_f32_len(
                allocation_total,
                model.config.perception_dims(),
                model.config.update_dims(),
            )?
        } else {
            allocation_total
        };
        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("burn_automata_state_density"),
            size: byte_len::<f32>(diagnostics_capacity_f32_len)?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let material_values = pack_material_values(&material, allocation_total);
        let material_buffer = storage_buffer_f32_with_usage(
            &self.device,
            "burn_automata_state_material",
            &material_values,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let positions_buffers = [positions_a, positions_b];
        let states_buffers = [states_a, states_b];
        let grid_bind_groups = std::array::from_fn(|current| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_state_grid_bind_group"),
                layout: &self.grid_bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(1, &positions_buffers[current]),
                    bind_entry(4, &linked_grid_buffer),
                    bind_entry(8, &indirect_buffer),
                    bind_entry(9, &material_buffer),
                ],
            })
        });
        let step_bind_groups = std::array::from_fn(|current| {
            let next = 1 - current;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_state_step_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(1, &positions_buffers[current]),
                    bind_entry(2, &states_buffers[current]),
                    bind_entry(3, &weights_buffer),
                    bind_entry(4, &linked_grid_buffer),
                    bind_entry(5, &positions_buffers[next]),
                    bind_entry(6, &states_buffers[next]),
                    bind_entry(7, &density_buffer),
                    bind_entry(9, &material_buffer),
                ],
            })
        });
        let gaussian_source_bind_groups = std::array::from_fn(|current| {
            let next = 1 - current;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("burn_automata_gaussian_source_bind_group"),
                layout: &self.gaussian_source_bind_group_layout,
                entries: &[
                    bind_entry(0, &params_buffer),
                    bind_entry(5, &positions_buffers[next]),
                    bind_entry(6, &states_buffers[next]),
                    bind_entry(9, &material_buffer),
                ],
            })
        });
        Ok(WgpuAutomataState {
            total,
            particle_count,
            particle_capacity,
            allocation_total,
            batch_size,
            spatial_dims: model.config.spatial_dims,
            hidden_dims: model.config.hidden_dims,
            feature_dims: model.config.perception_dims(),
            output_dims: model.config.update_dims(),
            bvh_leaf_count,
            bvh_levels,
            bvh_sort_count,
            position_f32_len,
            state_f32_len,
            material_enabled: material.enabled,
            mean_represented_measure: material.mean_measure,
            max_material_bandwidth: if material.enabled {
                material.max_bandwidth
            } else {
                grid.eps
            },
            support_bin_count,
            support_bin_capacity,
            requested_support_bin_count,
            support_bin_min: material.support_bin_min,
            support_bin_max: material.support_bin_max,
            support_bin_ratio: material.support_bin_ratio,
            support_bins_forced: material.support_bins_forced,
            display_scale_per_footprint: material.display_scale,
            render_transition_steps: material.render_transition_steps,
            render_transition_start_step: 0,
            adaptive_local_rule_mode: WgpuAdaptiveLocalRuleMode::Disabled,
            adaptive_local_hidden_start: 0,
            adaptive_local_residual_scale: 0.0,
            adaptive_base_footprint: 1.0,
            adaptive_reference_footprint: 1.0,
            adaptive_shepard_epsilon: 1.0e-8,
            adaptive_moment_regularization: 1.0e-4,
            adaptive_moment_condition_limit: 1.0e5,
            adaptive_max_neighbors: 0,
            adaptive_pair_scale_power: 8.0,
            expected_coarse_update_mask: false,
            adaptive_closure_enabled: false,
            adaptive_closure_hidden_dims: 0,
            adaptive_closure_basis_enabled: false,
            adaptive_closure_basis_hidden_dims: 0,
            dt,
            update_prob,
            grid_storage_len,
            grid_clear_len,
            cell_count,
            spatial_cell_count,
            bucket_capacity,
            neighbor_mode: resolved_neighbor_mode,
            fused_sorted_grid_enabled,
            stable_sorted_cells_enabled: material.enabled,
            weights_f32_len,
            current: 0,
            step_index: 0,
            lane_seeds: lane_seed_values,
            params_buffer,
            positions_buffers,
            states_buffers,
            weights_buffer,
            linked_grid_buffer,
            indirect_buffer,
            density_buffer,
            diagnostics_f32_len,
            material_buffer,
            grid_bind_groups,
            step_bind_groups,
            gaussian_source_bind_groups,
            step_indices_buffer: None,
            step_indices_capacity: 0,
        })
    }

    pub fn update_state_model(
        &self,
        state: &mut WgpuAutomataState,
        model: &NpaModel,
        grid: &HashGridConfig,
        dt: f32,
        update_prob: f32,
        seed: u64,
    ) -> AutomataResult<()> {
        validate_gpu_model_config(model, grid)?;
        if !dt.is_finite() {
            return Err(AutomataError::InvalidArgument(format!(
                "dt must be finite, got {dt}"
            )));
        }
        if !(0.0..=1.0).contains(&update_prob) || !update_prob.is_finite() {
            return Err(AutomataError::InvalidArgument(format!(
                "update_prob must be finite and in [0, 1], got {update_prob}"
            )));
        }
        let weights = packed_weights(model);
        if weights.len() != state.weights_f32_len {
            return Err(AutomataError::InvalidArgument(format!(
                "updated model weight len {} != resident weight len {}",
                weights.len(),
                state.weights_f32_len
            )));
        }
        let mut params = gpu_params(
            model,
            state.total,
            state.batch_size,
            state.particle_count,
            grid,
            dt,
            state.bucket_capacity,
            state.neighbor_mode,
            update_prob,
            seed,
        )?;
        params[PARAM_MATERIAL_ENABLED] = u32::from(state.material_enabled);
        params[PARAM_MEAN_REPRESENTED_MEASURE] = state.mean_represented_measure.to_bits();
        params[PARAM_DISPLAY_SCALE_PER_FOOTPRINT] = state.display_scale_per_footprint.to_bits();
        params[PARAM_RENDER_TRANSITION_STEPS] = state.render_transition_steps;
        params[PARAM_RENDER_TRANSITION_START_STEP] = state.render_transition_start_step;
        params[PARAM_ADAPTIVE_LOCAL_HIDDEN_START] = state.adaptive_local_hidden_start;
        params[PARAM_ADAPTIVE_LOCAL_RESIDUAL_SCALE] = state.adaptive_local_residual_scale.to_bits();
        params[PARAM_ADAPTIVE_BASE_FOOTPRINT] = state.adaptive_base_footprint.to_bits();
        params[PARAM_ADAPTIVE_REFERENCE_FOOTPRINT] = state.adaptive_reference_footprint.to_bits();
        params[PARAM_ADAPTIVE_SHEPARD_EPSILON] = state.adaptive_shepard_epsilon.to_bits();
        params[PARAM_ADAPTIVE_MOMENT_REGULARIZATION] =
            state.adaptive_moment_regularization.to_bits();
        params[PARAM_ADAPTIVE_MOMENT_CONDITION_LIMIT] =
            state.adaptive_moment_condition_limit.to_bits();
        params[PARAM_ADAPTIVE_MAX_NEIGHBORS] = state.adaptive_max_neighbors;
        params[PARAM_ADAPTIVE_PAIR_SCALE_POWER] = state.adaptive_pair_scale_power.to_bits();
        params[PARAM_MAX_MATERIAL_BANDWIDTH] = state.max_material_bandwidth.to_bits();
        params[PARAM_CELL_COUNT] = u32_checked(state.cell_count, "support-bin cell count")?;
        params[PARAM_SUPPORT_BIN_COUNT] =
            u32_checked(state.support_bin_count, "support bin count")?;
        params[PARAM_SPATIAL_CELL_COUNT] =
            u32_checked(state.spatial_cell_count, "spatial cell count")?;
        params[PARAM_SUPPORT_BIN_MIN] = state.support_bin_min.to_bits();
        params[PARAM_SUPPORT_BIN_MAX] = state.support_bin_max.to_bits();
        params[PARAM_SUPPORT_BIN_RATIO] = state.support_bin_ratio.to_bits();
        params[PARAM_ADAPTIVE_LOCAL_RULE_MODE] = state.adaptive_local_rule_mode.as_u32();
        params[PARAM_EXPECTED_COARSE_UPDATE_MASK] = u32::from(state.expected_coarse_update_mask);
        params[PARAM_ADAPTIVE_CLOSURE_ENABLED] = u32::from(state.adaptive_closure_enabled);
        params[PARAM_ADAPTIVE_CLOSURE_HIDDEN_DIMS] = state.adaptive_closure_hidden_dims;
        params[PARAM_ADAPTIVE_CLOSURE_BASIS_ENABLED] =
            u32::from(state.adaptive_closure_basis_enabled);
        params[PARAM_ADAPTIVE_CLOSURE_BASIS_HIDDEN_DIMS] = state.adaptive_closure_basis_hidden_dims;
        params[PARAM_BVH_LEAF_COUNT] = u32_checked(state.bvh_leaf_count, "BVH leaf count")?;
        params[PARAM_BVH_SORT_COUNT] = u32_checked(state.bvh_sort_count, "BVH sort count")?;
        params[PARAM_RESIDENT_CAPACITY] =
            u32_checked(state.allocation_total, "resident allocation total")?;
        params[PARAM_STEP_INDEX] = state.step_index;
        state.lane_seeds = gpu_lane_seeds(state.batch_size, state.particle_count, seed, None)?;
        params[PARAM_LANE_SEEDS_START..PARAM_LANE_SEEDS_START + state.lane_seeds.len()]
            .copy_from_slice(&state.lane_seeds);
        self.queue
            .write_buffer(&state.params_buffer, 0, bytemuck::cast_slice(&params));
        self.queue
            .write_buffer(&state.weights_buffer, 0, bytemuck::cast_slice(&weights));
        state.dt = dt;
        state.update_prob = update_prob;
        Ok(())
    }

    /// Activates a larger prefix of an already allocated single-trajectory
    /// material state. Device topology must populate the new rows before this
    /// method is called; no resident buffers or bind groups are recreated.
    pub(crate) fn activate_reserved_material_rows(
        &self,
        state: &mut WgpuAutomataState,
        model: &NpaModel,
        grid: &HashGridConfig,
        particle_count: usize,
        mean_represented_measure: f32,
        max_material_bandwidth: f32,
    ) -> AutomataResult<()> {
        if state.batch_size != 1
            || !state.material_enabled
            || particle_count < state.particle_count
            || particle_count > state.particle_capacity
            || !mean_represented_measure.is_finite()
            || mean_represented_measure <= 0.0
            || !max_material_bandwidth.is_finite()
            || max_material_bandwidth <= 0.0
            || is_bvh_neighbor_mode(state.neighbor_mode)
        {
            return Err(AutomataError::InvalidArgument(format!(
                "reserved material activation requires one non-BVH trajectory with active {} <= requested {particle_count} <= capacity {}, finite material calibration",
                state.particle_count, state.particle_capacity,
            )));
        }
        state.particle_count = particle_count;
        state.total = particle_count;
        state.position_f32_len = particle_count.checked_mul(4).ok_or_else(|| {
            AutomataError::InvalidArgument("active position size overflow".to_owned())
        })?;
        state.state_f32_len = particle_count
            .checked_mul(model.config.state_dims)
            .ok_or_else(|| {
                AutomataError::InvalidArgument("active state size overflow".to_owned())
            })?;
        state.diagnostics_f32_len = if state.spatial_dims == 2 {
            adaptive_diagnostics_f32_len(particle_count, state.feature_dims, state.output_dims)?
        } else {
            particle_count
        };
        state.mean_represented_measure = mean_represented_measure;
        state.max_material_bandwidth = max_material_bandwidth;
        self.write_param_u32(
            state,
            PARAM_TOTAL,
            u32_checked(particle_count, "active rows")?,
        );
        self.write_param_u32(
            state,
            PARAM_PARTICLE_COUNT,
            u32_checked(particle_count, "active particles")?,
        );
        self.write_param_u32(
            state,
            PARAM_MEAN_REPRESENTED_MEASURE,
            mean_represented_measure.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_MAX_MATERIAL_BANDWIDTH,
            max_material_bandwidth.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_DENSITY_SCALE,
            density_gradient_scale(model, grid, particle_count).to_bits(),
        );
        Ok(())
    }

    pub(crate) fn configure_state_adaptive_local_rule(
        &self,
        state: &mut WgpuAutomataState,
        mode: WgpuAdaptiveLocalRuleMode,
        local_hidden_start: usize,
        residual_scale: f32,
        base_footprint: f32,
        reference_footprint: f32,
        shepard_epsilon: f32,
        moment_regularization: f32,
        moment_condition_limit: f32,
        max_neighbors: usize,
        pair_scale_power: f32,
    ) -> AutomataResult<()> {
        let invalid_hidden_start = match mode {
            WgpuAdaptiveLocalRuleMode::Disabled => true,
            WgpuAdaptiveLocalRuleMode::Residual => {
                local_hidden_start == 0 || local_hidden_start > state.hidden_dims
            }
            WgpuAdaptiveLocalRuleMode::NormalizedExposureResidual => {
                local_hidden_start == 0 || local_hidden_start > state.hidden_dims
            }
            WgpuAdaptiveLocalRuleMode::CoarseReplacement => {
                local_hidden_start == 0 || local_hidden_start > state.hidden_dims
            }
            WgpuAdaptiveLocalRuleMode::CompatibleResidual => {
                local_hidden_start == 0 || local_hidden_start > state.hidden_dims
            }
            WgpuAdaptiveLocalRuleMode::NormalizedPrimary => local_hidden_start != 0,
        };
        if invalid_hidden_start
            || state.hidden_dims > MAX_HIDDEN_DIMS
            || !residual_scale.is_finite()
            || residual_scale < 0.0
            || !base_footprint.is_finite()
            || base_footprint <= 0.0
            || !reference_footprint.is_finite()
            || reference_footprint <= 0.0
            || !shepard_epsilon.is_finite()
            || shepard_epsilon <= 0.0
            || !moment_regularization.is_finite()
            || moment_regularization < 0.0
            || !moment_condition_limit.is_finite()
            || moment_condition_limit < 1.0
            || !pair_scale_power.is_finite()
            || pair_scale_power < 1.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive GPU local-rule parameters are invalid".to_string(),
            ));
        }
        state.adaptive_local_rule_mode = mode;
        state.adaptive_local_hidden_start =
            u32_checked(local_hidden_start, "adaptive local hidden start")?;
        state.adaptive_local_residual_scale = residual_scale;
        state.adaptive_base_footprint = base_footprint;
        state.adaptive_reference_footprint = reference_footprint;
        state.adaptive_shepard_epsilon = shepard_epsilon;
        state.adaptive_moment_regularization = moment_regularization;
        state.adaptive_moment_condition_limit = moment_condition_limit;
        state.adaptive_max_neighbors = u32_checked(max_neighbors, "adaptive max neighbors")?;
        state.adaptive_pair_scale_power = pair_scale_power;
        let values = [
            state.adaptive_local_hidden_start,
            residual_scale.to_bits(),
            base_footprint.to_bits(),
            shepard_epsilon.to_bits(),
            moment_regularization.to_bits(),
            moment_condition_limit.to_bits(),
            state.adaptive_max_neighbors,
        ];
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_LOCAL_HIDDEN_START * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&values),
        );
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_REFERENCE_FOOTPRINT * std::mem::size_of::<u32>())
                as wgpu::BufferAddress,
            bytemuck::cast_slice(&[reference_footprint.to_bits()]),
        );
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_PAIR_SCALE_POWER * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&[pair_scale_power.to_bits()]),
        );
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_LOCAL_RULE_MODE * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&[mode.as_u32()]),
        );
        Ok(())
    }

    pub(crate) fn configure_state_adaptive_closure_rule(
        &self,
        state: &mut WgpuAutomataState,
        rule: &NpaModel,
    ) -> AutomataResult<()> {
        rule.validate()?;
        if !state.material_enabled
            || rule.config.spatial_dims != state.spatial_dims
            || rule.config.perception_dims() != state.feature_dims
            || rule.config.update_dims() != state.output_dims
            || rule.config.hidden_dims > MAX_HIDDEN_DIMS
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive GPU closure rule shape mismatch: material={}, spatial {} != {}, features {} != {}, output {} != {}, hidden {} > {}",
                state.material_enabled,
                rule.config.spatial_dims,
                state.spatial_dims,
                rule.config.perception_dims(),
                state.feature_dims,
                rule.config.update_dims(),
                state.output_dims,
                rule.config.hidden_dims,
                MAX_HIDDEN_DIMS,
            )));
        }
        let weights = packed_weights(rule);
        if weights.len() > MAX_CLOSURE_WEIGHT_FLOATS {
            return Err(AutomataError::InvalidArgument(
                "adaptive GPU closure rule exceeds the resident weight capacity".to_owned(),
            ));
        }
        self.queue.write_buffer(
            &state.weights_buffer,
            byte_len::<f32>(packed_weight_len(
                state.feature_dims,
                state.hidden_dims,
                state.output_dims,
            )?)?,
            bytemuck::cast_slice(&weights),
        );
        state.adaptive_closure_enabled = true;
        state.adaptive_closure_hidden_dims =
            u32_checked(rule.config.hidden_dims, "adaptive closure hidden width")?;
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_CLOSURE_ENABLED * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&[1_u32, state.adaptive_closure_hidden_dims]),
        );
        Ok(())
    }

    pub(crate) fn configure_state_adaptive_closure_basis_rule(
        &self,
        state: &mut WgpuAutomataState,
        rule: &NpaModel,
    ) -> AutomataResult<()> {
        rule.validate()?;
        if !state.adaptive_closure_enabled
            || !state.material_enabled
            || rule.config.spatial_dims != state.spatial_dims
            || rule.config.perception_dims() != state.feature_dims
            || rule.config.update_dims() != state.output_dims
            || rule.config.hidden_dims > MAX_HIDDEN_DIMS
        {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive GPU closure-basis rule shape mismatch: closure={}, material={}, spatial {} != {}, features {} != {}, output {} != {}, hidden {} > {}",
                state.adaptive_closure_enabled,
                state.material_enabled,
                rule.config.spatial_dims,
                state.spatial_dims,
                rule.config.perception_dims(),
                state.feature_dims,
                rule.config.update_dims(),
                state.output_dims,
                rule.config.hidden_dims,
                MAX_HIDDEN_DIMS,
            )));
        }
        let weights = packed_weights(rule);
        if weights.len() > MAX_CLOSURE_WEIGHT_FLOATS {
            return Err(AutomataError::InvalidArgument(
                "adaptive GPU closure-basis rule exceeds the resident weight capacity".to_owned(),
            ));
        }
        let primary_len =
            packed_weight_len(state.feature_dims, state.hidden_dims, state.output_dims)?;
        let closure_len = packed_weight_len(
            state.feature_dims,
            state.adaptive_closure_hidden_dims as usize,
            state.output_dims,
        )?;
        self.queue.write_buffer(
            &state.weights_buffer,
            byte_len::<f32>(primary_len + closure_len)?,
            bytemuck::cast_slice(&weights),
        );
        state.adaptive_closure_basis_enabled = true;
        state.adaptive_closure_basis_hidden_dims = u32_checked(
            rule.config.hidden_dims,
            "adaptive closure-basis hidden width",
        )?;
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_ADAPTIVE_CLOSURE_BASIS_ENABLED * std::mem::size_of::<u32>())
                as wgpu::BufferAddress,
            bytemuck::cast_slice(&[1_u32, state.adaptive_closure_basis_hidden_dims]),
        );
        Ok(())
    }

    pub(crate) fn configure_state_adaptive_integration(
        &self,
        state: &mut WgpuAutomataState,
        base_footprint: f32,
        expected_coarse_update_mask: bool,
    ) -> AutomataResult<()> {
        if !state.material_enabled || !base_footprint.is_finite() || base_footprint <= 0.0 {
            return Err(AutomataError::InvalidArgument(
                "adaptive GPU integration requires material metadata and a positive base footprint"
                    .to_owned(),
            ));
        }
        state.adaptive_base_footprint = base_footprint;
        state.expected_coarse_update_mask = expected_coarse_update_mask;
        self.write_param_u32(
            state,
            PARAM_ADAPTIVE_BASE_FOOTPRINT,
            base_footprint.to_bits(),
        );
        self.write_param_u32(
            state,
            PARAM_EXPECTED_COARSE_UPDATE_MASK,
            u32::from(expected_coarse_update_mask),
        );
        Ok(())
    }

    pub(crate) fn configure_state_adaptive_reference_footprint(
        &self,
        state: &mut WgpuAutomataState,
        reference_footprint: f32,
    ) -> AutomataResult<()> {
        if !state.material_enabled || !reference_footprint.is_finite() || reference_footprint <= 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive GPU material conditioning requires a positive reference footprint"
                    .to_owned(),
            ));
        }
        state.adaptive_reference_footprint = reference_footprint;
        self.write_param_u32(
            state,
            PARAM_ADAPTIVE_REFERENCE_FOOTPRINT,
            reference_footprint.to_bits(),
        );
        Ok(())
    }

    /// Replaces material metadata without reallocating the resident rollout.
    /// Particle count and render calibration remain fixed; topology changes that
    /// alter the count create a new state instead.
    pub(crate) fn update_state_particles(
        &self,
        state: &mut WgpuAutomataState,
        positions: &[[f32; 4]],
        particle_states: &[f32],
    ) -> AutomataResult<()> {
        if positions.len() != state.total || particle_states.len() != state.state_f32_len {
            return Err(AutomataError::InvalidArgument(format!(
                "WGPU in-place particle update has {} positions and {} state values; expected {} and {}",
                positions.len(),
                particle_states.len(),
                state.total,
                state.state_f32_len,
            )));
        }
        self.queue.write_buffer(
            &state.positions_buffers[state.current],
            0,
            bytemuck::cast_slice(positions),
        );
        self.queue.write_buffer(
            &state.states_buffers[state.current],
            0,
            bytemuck::cast_slice(particle_states),
        );
        Ok(())
    }

    pub fn update_state_material(
        &self,
        state: &mut WgpuAutomataState,
        material: WgpuMaterialStateInit<'_>,
    ) -> AutomataResult<()> {
        self.update_state_material_impl(state, material, None)
    }

    /// Updates material metadata and re-evaluates the support-bin execution
    /// policy from a host-synchronized particle snapshot. The resident grid
    /// buffer is allocated for the declared bin capacity at state creation, so
    /// this only switches active metadata and never reallocates device storage.
    pub fn update_state_material_with_support_policy(
        &self,
        state: &mut WgpuAutomataState,
        positions: &[[f32; 4]],
        grid: &HashGridConfig,
        material: WgpuMaterialStateInit<'_>,
    ) -> AutomataResult<()> {
        if positions.len() != state.total
            || grid.cell_count() != state.spatial_cell_count
            || grid.dim != state.spatial_dims
        {
            return Err(AutomataError::InvalidArgument(
                "WGPU support-bin policy refresh requires the resident particle layout and grid"
                    .to_owned(),
            ));
        }
        self.update_state_material_impl(state, material, Some((positions, grid)))
    }

    fn update_state_material_impl(
        &self,
        state: &mut WgpuAutomataState,
        material: WgpuMaterialStateInit<'_>,
        support_policy: Option<(&[[f32; 4]], &HashGridConfig)>,
    ) -> AutomataResult<()> {
        let material = material_state_values(
            Some(material),
            state.total,
            state.particle_count,
            state.state_f32_len / state.total,
            state.spatial_dims,
        )?;
        if !material.enabled
            || (material.mean_measure - state.mean_represented_measure).abs()
                > 1.0e-5 * state.mean_represented_measure.abs().max(1.0e-12)
            || material.display_scale.to_bits() != state.display_scale_per_footprint.to_bits()
            || material.render_transition_steps != state.render_transition_steps
            || material.support_bin_count != state.requested_support_bin_count
            || material.support_bin_min.to_bits() != state.support_bin_min.to_bits()
            || material.support_bin_max.to_bits() != state.support_bin_max.to_bits()
            || material.support_bin_ratio.to_bits() != state.support_bin_ratio.to_bits()
            || material.support_bins_forced != state.support_bins_forced
        {
            return Err(AutomataError::InvalidArgument(
                "WGPU in-place material update changed resident calibration".to_owned(),
            ));
        }
        if let Some((positions, grid)) = support_policy {
            let support_bin_count = if state.support_bin_capacity > 1
                && (state.support_bins_forced
                    || should_activate_support_bins(
                        grid,
                        state.particle_count,
                        positions,
                        &material.bandwidth,
                        state.support_bin_min,
                        state.support_bin_max,
                        state.support_bin_ratio,
                    )) {
                state.support_bin_capacity
            } else {
                1
            };
            let cell_count = state
                .spatial_cell_count
                .checked_mul(state.batch_size)
                .and_then(|value| value.checked_mul(support_bin_count))
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "refreshed support-bin cell count overflow".to_owned(),
                    )
                })?;
            state.support_bin_count = support_bin_count;
            state.cell_count = cell_count;
            state.grid_clear_len =
                grid_clear_len_for_mode(cell_count, state.bucket_capacity, state.neighbor_mode)?;
            state.fused_sorted_grid_enabled = state
                .spatial_cell_count
                .checked_mul(support_bin_count)
                .is_some_and(|cells| cells <= burn_automata_kernels::FUSED_SORTED_GRID_MAX_CELLS)
                && matches!(
                    state.neighbor_mode,
                    WgpuNeighborMode::SortedCells
                        | WgpuNeighborMode::CooperativeSortedCells
                        | WgpuNeighborMode::SubgroupCooperativeSortedCells
                );
            self.write_param_u32(
                state,
                PARAM_CELL_COUNT,
                u32_checked(cell_count, "refreshed support-bin cell count")?,
            );
            self.write_param_u32(
                state,
                PARAM_SUPPORT_BIN_COUNT,
                u32_checked(support_bin_count, "refreshed support-bin count")?,
            );
        }
        let material_values = pack_material_values(&material, state.allocation_total);
        self.queue.write_buffer(
            &state.material_buffer,
            0,
            bytemuck::cast_slice(&material_values),
        );
        state.max_material_bandwidth = material.max_bandwidth;
        self.write_param_u32(
            state,
            PARAM_MAX_MATERIAL_BANDWIDTH,
            material.max_bandwidth.to_bits(),
        );
        Ok(())
    }

    /// Restores an absolute stochastic rollout step after resident-state
    /// recreation (for example after a topology event).
    pub fn set_state_step_index(
        &self,
        state: &mut WgpuAutomataState,
        step_index: u32,
    ) -> AutomataResult<()> {
        state.step_index = step_index;
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_STEP_INDEX * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(&[step_index]),
        );
        Ok(())
    }

    /// Starts a topology-render interpolation without resetting the stochastic
    /// update sequence.
    pub fn begin_state_render_transition(
        &self,
        state: &mut WgpuAutomataState,
        start_step: u32,
    ) -> AutomataResult<()> {
        state.render_transition_start_step = start_step;
        self.queue.write_buffer(
            &state.params_buffer,
            (PARAM_RENDER_TRANSITION_START_STEP * std::mem::size_of::<u32>())
                as wgpu::BufferAddress,
            bytemuck::cast_slice(&[start_step]),
        );
        Ok(())
    }

    pub fn neighbor_report(&self, state: &WgpuAutomataState) -> WgpuNeighborReport {
        WgpuNeighborReport {
            bucket_capacity: state.bucket_capacity,
            grid_storage_len: state.grid_storage_len,
            grid_clear_len: state.grid_clear_len,
            mode: state.neighbor_mode,
            support_bin_count: state.support_bin_count,
            support_bin_capacity: state.support_bin_capacity,
            requested_support_bin_count: state.requested_support_bin_count,
        }
    }

    pub(crate) fn max_independent_trajectory_lanes(&self) -> usize {
        MAX_LANE_SEEDS.min(self.device.limits().max_compute_workgroups_per_dimension as usize)
    }

    pub(crate) fn set_stable_sorted_cells_enabled(
        &self,
        state: &mut WgpuAutomataState,
        enabled: bool,
    ) {
        state.stable_sorted_cells_enabled = enabled;
    }

    #[cfg(test)]
    pub(crate) fn set_fused_sorted_grid_enabled_for_test(
        &self,
        state: &mut WgpuAutomataState,
        enabled: bool,
    ) {
        state.fused_sorted_grid_enabled = enabled;
    }
}

fn adaptive_diagnostics_f32_len(
    total: usize,
    feature_dims: usize,
    output_dims: usize,
) -> AutomataResult<usize> {
    let update_width = output_dims.checked_mul(2).ok_or_else(|| {
        AutomataError::InvalidArgument("adaptive diagnostic output width overflow".to_owned())
    })?;
    let row_width = feature_dims
        .checked_mul(2)
        .and_then(|value| value.checked_add(update_width))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            AutomataError::InvalidArgument("adaptive diagnostic row width overflow".to_owned())
        })?;
    total.checked_mul(row_width).ok_or_else(|| {
        AutomataError::InvalidArgument("adaptive diagnostic buffer length overflow".to_owned())
    })
}

fn gpu_lane_seeds(
    batch_size: usize,
    particle_count: usize,
    seed: u64,
    independent_seeds: Option<&[u64]>,
) -> AutomataResult<Vec<u32>> {
    if batch_size > MAX_LANE_SEEDS {
        return Err(AutomataError::InvalidArgument(format!(
            "WGPU independent trajectory batch has {batch_size} lanes; maximum is {MAX_LANE_SEEDS}",
        )));
    }
    if let Some(seeds) = independent_seeds
        && seeds.len() != batch_size
    {
        return Err(AutomataError::InvalidArgument(format!(
            "WGPU lane seed count {} does not match batch size {}",
            seeds.len(),
            batch_size,
        )));
    }
    let base_seed = gpu_random_seed(seed);
    (0..batch_size)
        .map(|lane| {
            if let Some(seeds) = independent_seeds {
                Ok(gpu_random_seed(seeds[lane]))
            } else {
                let particle_offset = lane.checked_mul(particle_count).ok_or_else(|| {
                    AutomataError::InvalidArgument("WGPU lane particle offset overflow".to_owned())
                })?;
                Ok(base_seed ^ u32_checked(particle_offset, "WGPU lane particle offset")?)
            }
        })
        .collect()
}

pub(super) const MATERIAL_JACOBIAN_CAPACITY: usize = MAX_STATE_DIMS * 3;
pub(super) const MATERIAL_CLOSURE_MODE_CAPACITY: usize = MAX_STATE_DIMS;
pub(super) const MATERIAL_CLOSURE_MODE_OFFSET: usize =
    13 + 2 * WGPU_MATERIAL_UPDATE_MASK_MEMBERS + MATERIAL_JACOBIAN_CAPACITY;
pub(super) const MATERIAL_CLOSURE_BASIS_CAPACITY: usize = 4;
pub(super) const MATERIAL_CLOSURE_BASIS_OFFSET: usize =
    MATERIAL_CLOSURE_MODE_OFFSET + MATERIAL_CLOSURE_MODE_CAPACITY;
pub(super) const MATERIAL_CLOSURE_PHASE_CAPACITY: usize = 2;
pub(super) const MATERIAL_CLOSURE_PHASE_OFFSET: usize =
    MATERIAL_CLOSURE_BASIS_OFFSET + MATERIAL_CLOSURE_BASIS_CAPACITY;
pub(super) const MATERIAL_STRIDE: usize =
    MATERIAL_CLOSURE_PHASE_OFFSET + MATERIAL_CLOSURE_PHASE_CAPACITY;
const MAX_CLOSURE_WEIGHT_FLOATS: usize = MAX_HIDDEN_DIMS * MAX_FEATURE_DIMS
    + MAX_HIDDEN_DIMS
    + MAX_OUTPUT_DIMS * MAX_HIDDEN_DIMS
    + MAX_OUTPUT_DIMS;

fn packed_weight_len(
    feature_dims: usize,
    hidden_dims: usize,
    output_dims: usize,
) -> AutomataResult<usize> {
    hidden_dims
        .checked_mul(feature_dims)
        .and_then(|value| value.checked_add(hidden_dims))
        .and_then(|value| {
            output_dims
                .checked_mul(hidden_dims)
                .and_then(|tail| value.checked_add(tail))
        })
        .and_then(|value| value.checked_add(output_dims))
        .ok_or_else(|| AutomataError::InvalidArgument("packed NPA weight size overflow".to_owned()))
}

struct MaterialStateValues {
    enabled: bool,
    measure: Vec<f32>,
    bandwidth: Vec<f32>,
    covariance: Vec<f32>,
    state_jacobian: Vec<f32>,
    closure_mode: Vec<f32>,
    closure_basis: Vec<f32>,
    closure_phase: Vec<f32>,
    update_masks: Vec<WgpuMaterialUpdateMask>,
    state_jacobian_dims: usize,
    render_from: Vec<f32>,
    render_target: Vec<f32>,
    mean_measure: f32,
    display_scale: f32,
    render_transition_steps: u32,
    support_bin_count: usize,
    support_bin_min: f32,
    support_bin_max: f32,
    support_bin_ratio: f32,
    support_bins_forced: bool,
    max_bandwidth: f32,
}

fn material_state_values(
    material: Option<WgpuMaterialStateInit<'_>>,
    total: usize,
    particle_count: usize,
    state_dims: usize,
    spatial_dims: usize,
) -> AutomataResult<MaterialStateValues> {
    let jacobian_dims = state_dims.checked_mul(spatial_dims).ok_or_else(|| {
        AutomataError::InvalidArgument("material state-Jacobian dimensions overflow".to_owned())
    })?;
    if jacobian_dims > MATERIAL_JACOBIAN_CAPACITY {
        return Err(AutomataError::InvalidArgument(format!(
            "material state-Jacobian width {jacobian_dims} exceeds {MATERIAL_JACOBIAN_CAPACITY}",
        )));
    }
    let Some(material) = material else {
        return Ok(MaterialStateValues {
            enabled: false,
            measure: vec![1.0; total],
            bandwidth: vec![1.0; total],
            covariance: vec![0.0; total * 9],
            state_jacobian: vec![0.0; total * jacobian_dims],
            closure_mode: vec![0.0; total * state_dims],
            closure_basis: vec![0.0; total * MATERIAL_CLOSURE_BASIS_CAPACITY],
            closure_phase: vec![0.0; total * MATERIAL_CLOSURE_PHASE_CAPACITY],
            update_masks: (0..total)
                .map(|index| WgpuMaterialUpdateMask::single((index % particle_count.max(1)) as u64))
                .collect(),
            state_jacobian_dims: jacobian_dims,
            render_from: vec![1.0; total],
            render_target: vec![1.0; total],
            mean_measure: 1.0,
            display_scale: 1.0,
            render_transition_steps: 0,
            support_bin_count: 1,
            support_bin_min: 1.0,
            support_bin_max: 1.0,
            support_bin_ratio: 2.0,
            support_bins_forced: false,
            max_bandwidth: 1.0,
        });
    };
    if material.represented_measure.len() != total
        || material
            .particle_ids
            .is_some_and(|particle_ids| particle_ids.len() != total)
        || material
            .update_masks
            .is_some_and(|update_masks| update_masks.len() != total)
        || material.bandwidth.len() != total
        || material.covariance.len() != total
        || material.state_jacobian.len() != total * jacobian_dims
        || material
            .closure_mode
            .is_some_and(|closure_mode| closure_mode.len() != total * state_dims)
        || material.closure_basis.is_some_and(|closure_basis| {
            closure_basis.len() != total * MATERIAL_CLOSURE_BASIS_CAPACITY
        })
        || material.closure_phase.is_some_and(|closure_phase| {
            closure_phase.len() != total * MATERIAL_CLOSURE_PHASE_CAPACITY
        })
        || material.render_from_scale.len() != total
        || material.render_target_footprint.len() != total
        || material
            .represented_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || material
            .bandwidth
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || material
            .covariance
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || material
            .state_jacobian
            .iter()
            .any(|value| !value.is_finite())
        || material
            .closure_mode
            .is_some_and(|closure_mode| closure_mode.iter().any(|value| !value.is_finite()))
        || material
            .closure_basis
            .is_some_and(|closure_basis| closure_basis.iter().any(|value| !value.is_finite()))
        || material
            .closure_phase
            .is_some_and(|closure_phase| closure_phase.iter().any(|value| !value.is_finite()))
        || material
            .render_from_scale
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || material
            .render_target_footprint
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || !material.display_scale_per_footprint.is_finite()
        || material.display_scale_per_footprint <= 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "WGPU material state has invalid shape or non-finite metadata".to_owned(),
        ));
    }
    // Topology can reorder unequal represented measures while conserving their
    // sum. Accumulate calibration in f64 so an algebraically conservative
    // same-count split/merge exchange does not trip an order-sensitive f32
    // mean check when the resident buffers are updated in place.
    let mean = (material
        .represented_measure
        .iter()
        .map(|measure| f64::from(*measure))
        .sum::<f64>()
        / total.max(1) as f64) as f32;
    let min_bandwidth = material
        .bandwidth
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bandwidth = material.bandwidth.iter().copied().fold(0.0_f32, f32::max);
    let (
        support_bin_count,
        support_bin_min,
        support_bin_max,
        support_bin_ratio,
        support_bins_forced,
    ) = if let Some(config) = material.support_bins {
        let bins =
            AdaptiveSupportBins::new(config.min_bandwidth, config.max_bandwidth, config.ratio)
                .map_err(|error| AutomataError::InvalidArgument(error.to_string()))?;
        if min_bandwidth < config.min_bandwidth || max_bandwidth > config.max_bandwidth {
            return Err(AutomataError::InvalidArgument(format!(
                "WGPU material bandwidth range {min_bandwidth}..{max_bandwidth} is outside declared support bins {}..{}",
                config.min_bandwidth, config.max_bandwidth,
            )));
        }
        (
            bins.len(),
            config.min_bandwidth,
            config.max_bandwidth,
            config.ratio,
            config.force,
        )
    } else {
        (1, min_bandwidth, max_bandwidth, 2.0, false)
    };
    if !mean.is_finite() || mean <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "WGPU material state has invalid mean represented measure".to_owned(),
        ));
    }
    let update_masks = material.update_masks.map_or_else(
        || {
            material.particle_ids.map_or_else(
                || {
                    (0..total)
                        .map(|index| {
                            WgpuMaterialUpdateMask::single((index % particle_count.max(1)) as u64)
                        })
                        .collect()
                },
                |particle_ids| {
                    particle_ids
                        .iter()
                        .copied()
                        .map(WgpuMaterialUpdateMask::single)
                        .collect()
                },
            )
        },
        <[WgpuMaterialUpdateMask]>::to_vec,
    );
    if update_masks.iter().any(|mask| {
        let sum = mask.weights.iter().sum::<f32>();
        mask.weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || !sum.is_finite()
            || (sum - 1.0).abs() > 1.0e-4
    }) {
        return Err(AutomataError::InvalidArgument(
            "WGPU material update-mask weights must be finite, non-negative, and sum to one"
                .to_owned(),
        ));
    }
    Ok(MaterialStateValues {
        enabled: true,
        measure: material.represented_measure.to_vec(),
        bandwidth: material.bandwidth.to_vec(),
        covariance: material.covariance.iter().flatten().copied().collect(),
        state_jacobian: material.state_jacobian.to_vec(),
        closure_mode: material
            .closure_mode
            .map_or_else(|| vec![0.0; total * state_dims], <[f32]>::to_vec),
        closure_basis: material.closure_basis.map_or_else(
            || vec![0.0; total * MATERIAL_CLOSURE_BASIS_CAPACITY],
            <[f32]>::to_vec,
        ),
        closure_phase: material.closure_phase.map_or_else(
            || vec![0.0; total * MATERIAL_CLOSURE_PHASE_CAPACITY],
            <[f32]>::to_vec,
        ),
        update_masks,
        state_jacobian_dims: jacobian_dims,
        render_from: material.render_from_scale.to_vec(),
        render_target: material.render_target_footprint.to_vec(),
        mean_measure: mean,
        display_scale: material.display_scale_per_footprint,
        render_transition_steps: material.render_transition_steps,
        support_bin_count,
        support_bin_min,
        support_bin_max,
        support_bin_ratio,
        support_bins_forced,
        max_bandwidth,
    })
}

fn pack_material_values(material: &MaterialStateValues, capacity: usize) -> Vec<f32> {
    debug_assert!(capacity >= material.measure.len());
    debug_assert_eq!(material.covariance.len(), material.measure.len() * 9);
    debug_assert_eq!(material.bandwidth.len(), material.measure.len());
    debug_assert_eq!(material.render_from.len(), material.measure.len());
    debug_assert_eq!(material.render_target.len(), material.measure.len());
    debug_assert_eq!(material.update_masks.len(), material.measure.len());
    debug_assert_eq!(
        material.state_jacobian.len(),
        material.measure.len() * material.state_jacobian_dims,
    );
    debug_assert_eq!(
        material.closure_mode.len(),
        material.measure.len() * (material.closure_mode.len() / material.measure.len().max(1)),
    );
    debug_assert_eq!(
        material.closure_basis.len(),
        material.measure.len() * MATERIAL_CLOSURE_BASIS_CAPACITY,
    );
    debug_assert_eq!(
        material.closure_phase.len(),
        material.measure.len() * MATERIAL_CLOSURE_PHASE_CAPACITY,
    );
    let mut packed = Vec::with_capacity(capacity * MATERIAL_STRIDE + 1);
    for index in 0..material.measure.len() {
        packed.push(material.measure[index]);
        packed.extend_from_slice(&material.covariance[index * 9..(index + 1) * 9]);
        packed.push(material.render_from[index]);
        packed.push(material.render_target[index]);
        packed.push(material.bandwidth[index]);
        packed.extend(
            material.update_masks[index]
                .particle_ids
                .iter()
                .map(|id| f32::from_bits((*id as u32) ^ ((*id >> 32) as u32))),
        );
        packed.extend_from_slice(&material.update_masks[index].weights);
        packed.extend_from_slice(
            &material.state_jacobian
                [index * material.state_jacobian_dims..(index + 1) * material.state_jacobian_dims],
        );
        packed.resize(
            packed.len() + MATERIAL_JACOBIAN_CAPACITY - material.state_jacobian_dims,
            0.0,
        );
        let closure_dims = material.closure_mode.len() / material.measure.len().max(1);
        packed.extend_from_slice(
            &material.closure_mode[index * closure_dims..(index + 1) * closure_dims],
        );
        packed.resize(
            packed.len() + MATERIAL_CLOSURE_MODE_CAPACITY - closure_dims,
            0.0,
        );
        packed.extend_from_slice(
            &material.closure_basis[index * MATERIAL_CLOSURE_BASIS_CAPACITY
                ..(index + 1) * MATERIAL_CLOSURE_BASIS_CAPACITY],
        );
        packed.extend_from_slice(
            &material.closure_phase[index * MATERIAL_CLOSURE_PHASE_CAPACITY
                ..(index + 1) * MATERIAL_CLOSURE_PHASE_CAPACITY],
        );
    }
    packed.resize(capacity * MATERIAL_STRIDE, 0.0);
    // One persistent device-side diagnostic word follows the full resident
    // material allocation so activating reserve rows cannot overwrite it.
    // material payload. Paired topology increments it only when a proposed
    // budget-neutral exchange passes the configured gain margin.
    packed.push(0.0);
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_material_buffer_reserves_local_coarse_exposure() {
        let total = 7;
        let feature_dims = 11;
        let output_dims = 5;
        assert_eq!(
            adaptive_diagnostics_f32_len(total, feature_dims, output_dims).unwrap(),
            total * (2 * feature_dims + 2 * output_dims + 4),
        );
    }
}
