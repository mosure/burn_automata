//! Device-resident, budget-neutral topology changes for adaptive Target2D.

use super::*;

pub(super) struct BurnAdaptiveTopology {
    coarse_indices: Tensor1Int,
    fine_indices: Tensor1Int,
    represented_measure: Tensor1,
    active_particle_count: usize,
    coarse_particle_count: usize,
    fine_particle_count: usize,
    total_measure: f32,
    split_radius: f32,
    fine_footprint_squared: f32,
    merge_detail_scale: f32,
    min_relative_gain: f32,
    events_per_interval: usize,
    continuous: bool,
    enabled: bool,
    start_step: usize,
    end_step: usize,
    interval_steps: usize,
}

impl BurnAdaptiveTopology {
    pub(super) fn new(
        config: &AdaptiveTarget2dBurnConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Self> {
        let topology = config.topology;
        if !topology.split_radius_scale.is_finite()
            || topology.split_radius_scale < 0.0
            || !topology.merge_detail_scale.is_finite()
            || topology.merge_detail_scale < 0.0
            || !topology.min_relative_gain.is_finite()
            || !(0.0..=1.0).contains(&topology.min_relative_gain)
        {
            return Err(AutomataError::InvalidArgument(
                "adaptive Target2D topology scales must be finite and non-negative".to_string(),
            ));
        }

        let continuous = config.material.seed_layout
            == crate::adaptive::AdaptiveMaterialSeedLayout::GradedContinuous;
        let (coarse_indices, fine_indices) = if continuous {
            let total = config.material.represented_measure.iter().sum::<f32>();
            let mean = total / config.material.active_particle_count() as f32;
            let tolerance = 2.0e-4 * mean;
            (
                config
                    .material
                    .represented_measure
                    .iter()
                    .enumerate()
                    .filter_map(|(index, measure)| {
                        (*measure > mean + tolerance).then_some(index as i64)
                    })
                    .collect::<Vec<_>>(),
                config
                    .material
                    .represented_measure
                    .iter()
                    .enumerate()
                    .filter_map(|(index, measure)| {
                        (*measure + tolerance < mean).then_some(index as i64)
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            let fine_units = match config.material.fine_units.as_deref() {
                Some(fine_units) => fine_units,
                None if !topology.enabled => &[],
                None => {
                    return Err(AutomataError::InvalidArgument(
                        "paired hard topology requires a canonical integral material layout"
                            .to_owned(),
                    ));
                }
            };
            (
                fine_units
                    .iter()
                    .enumerate()
                    .filter_map(|(index, units)| (*units == 4).then_some(index as i64))
                    .collect::<Vec<_>>(),
                fine_units
                    .iter()
                    .enumerate()
                    .filter_map(|(index, units)| (*units == 1).then_some(index as i64))
                    .collect::<Vec<_>>(),
            )
        };
        let fine_footprint =
            (config.material.fine_measure / std::f32::consts::PI).sqrt();
        let enabled = topology.enabled
            && !coarse_indices.is_empty()
            && if continuous {
                !fine_indices.is_empty()
            } else {
                fine_indices.len() >= 4 && fine_footprint > 0.0
            };
        let coarse_tensor_values = if coarse_indices.is_empty() {
            vec![0]
        } else {
            coarse_indices.clone()
        };
        let fine_tensor_values = if fine_indices.is_empty() {
            vec![0]
        } else {
            fine_indices.clone()
        };

        Ok(Self {
            coarse_particle_count: coarse_indices.len(),
            fine_particle_count: fine_indices.len(),
            coarse_indices: Tensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(coarse_tensor_values.clone(), [coarse_tensor_values.len()]),
                device,
            ),
            fine_indices: Tensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(fine_tensor_values.clone(), [fine_tensor_values.len()]),
                device,
            ),
            represented_measure: Tensor::<BurnBackend, 1>::from_data(
                TensorData::new(
                    config.material.represented_measure.clone(),
                    [config.material.active_particle_count()],
                ),
                device,
            ),
            active_particle_count: config.material.active_particle_count(),
            total_measure: config.material.represented_measure.iter().sum(),
            split_radius: (1.5_f32).sqrt()
                * fine_footprint
                * topology.split_radius_scale,
            fine_footprint_squared: fine_footprint * fine_footprint,
            merge_detail_scale: topology.merge_detail_scale,
            min_relative_gain: topology.min_relative_gain,
            events_per_interval: topology.events_per_interval,
            continuous,
            enabled,
            start_step: topology.start_step,
            end_step: topology.end_step,
            interval_steps: topology.interval_steps.max(1),
        })
    }

    pub(super) fn should_apply(&self, completed_steps: usize) -> bool {
        self.enabled
            && completed_steps >= self.start_step
            && (self.end_step == 0 || completed_steps <= self.end_step)
            && completed_steps.is_multiple_of(self.interval_steps)
    }

    fn next_event_after(&self, before: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let lower = before.saturating_add(1).max(self.start_step).max(1);
        let next = lower
            .div_ceil(self.interval_steps)
            .saturating_mul(self.interval_steps);
        (self.end_step == 0 || next <= self.end_step).then_some(next)
    }

    fn should_apply_between(&self, before: usize, after: usize) -> bool {
        after > before
            && self
                .next_event_after(before)
                .is_some_and(|event| event <= after)
    }

    /// Caps a rollout chunk at the earliest scheduled event in any batch row.
    ///
    /// Persistent-pool rows have different ages. Without this split, applying
    /// an event after a fixed TBPTT chunk delays it by up to `chunk_steps - 1`
    /// and trains dynamics that differ from exact-boundary inference.
    pub(super) fn steps_until_next_event(&self, ages: &[usize], max_steps: usize) -> usize {
        ages.iter()
            .filter_map(|age| {
                self.next_event_after(*age)
                    .map(|event| event.saturating_sub(*age))
            })
            .filter(|steps| *steps > 0)
            .fold(max_steps, usize::min)
    }

    pub(super) fn scheduled_event_delay_steps(
        &self,
        before: &[usize],
        after: &[usize],
    ) -> Vec<Option<usize>> {
        debug_assert_eq!(before.len(), after.len());
        before
            .iter()
            .copied()
            .zip(after.iter().copied())
            .map(|(before, after)| {
                self.next_event_after(before)
                    .filter(|event| *event <= after)
                    .map(|event| after - event)
            })
            .collect()
    }

    pub(super) fn scheduled_event_rows(
        &self,
        before: &[usize],
        after: &[usize],
    ) -> Vec<bool> {
        debug_assert_eq!(before.len(), after.len());
        before
            .iter()
            .copied()
            .zip(after.iter().copied())
            .map(|(before, after)| self.should_apply_between(before, after))
            .collect()
    }

    pub(super) fn apply_scheduled(
        &self,
        positions: Tensor3,
        states: Tensor3,
        detail: Tensor2,
        before: &[usize],
        after: &[usize],
    ) -> (Tensor3, Tensor3, usize) {
        let batch_size = positions.shape().dims::<3>()[0];
        debug_assert_eq!(before.len(), batch_size);
        debug_assert_eq!(after.len(), batch_size);
        let gates = self
            .scheduled_event_rows(before, after)
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        let active_batches = gates.iter().filter(|gate| **gate > 0.0).count();
        if active_batches == 0 {
            return (positions, states, 0);
        }
        let device = positions.device();
        let batch_gate = Tensor::<BurnBackend, 1>::from_data(
            TensorData::new(gates, [batch_size]),
            &device,
        );
        self.apply_with_batch_gate(
            positions,
            states,
            detail,
            batch_gate,
            active_batches,
        )
    }

    /// Exchanges one four-unit coarse slot with four one-unit fine slots in
    /// each batch row. Row material metadata stays static while its spatial
    /// allocation moves. The hard selection is intentionally outside the
    /// autodiff graph and is only called at detached TBPTT boundaries.
    pub(super) fn apply(
        &self,
        positions: Tensor3,
        states: Tensor3,
        detail: Tensor2,
    ) -> (Tensor3, Tensor3, usize) {
        let batch_size = positions.shape().dims::<3>()[0];
        let batch_gate = Tensor::<BurnBackend, 1>::ones([batch_size], &positions.device());
        self.apply_with_batch_gate(positions, states, detail, batch_gate, batch_size)
    }

    fn apply_with_batch_gate(
        &self,
        positions: Tensor3,
        states: Tensor3,
        detail: Tensor2,
        batch_gate: Tensor1,
        active_batches: usize,
    ) -> (Tensor3, Tensor3, usize) {
        if !self.enabled {
            return (positions, states, 0);
        }
        let [batch_size, particle_count, position_dims] = positions.shape().dims::<3>();
        let state_dims = states.shape().dims::<3>()[2];
        debug_assert_eq!(particle_count, self.active_particle_count);
        debug_assert_eq!(position_dims, 2);
        debug_assert_eq!(detail.shape().dims::<2>(), [batch_size, particle_count]);
        debug_assert!(self.coarse_particle_count > 0);
        debug_assert!(self.fine_particle_count >= if self.continuous { 1 } else { 4 });

        if self.continuous {
            return self.apply_continuous(
                positions,
                states,
                detail,
                batch_gate,
                active_batches,
            );
        }

        let original_positions = positions.clone();
        let original_states = states.clone();
        let coarse_local = detail
            .clone()
            .select(1, self.coarse_indices.clone())
            .argmax(1)
            .squeeze_dim::<1>(1);
        let coarse_rows = self.coarse_indices.clone().select(0, coarse_local);
        let coarse_positions =
            gather_dynamic_rows(positions.clone(), coarse_rows.clone(), position_dims);
        let coarse_states = gather_dynamic_rows(states.clone(), coarse_rows.clone(), state_dims);

        let fine_positions = positions
            .clone()
            .select(1, self.fine_indices.clone());
        let fine_detail = detail.select(1, self.fine_indices.clone());
        let anchor_local = fine_detail
            .clone()
            .argmin(1)
            .squeeze_dim::<1>(1);
        let anchor_rows = self.fine_indices.clone().select(0, anchor_local);
        let anchor_positions =
            gather_dynamic_rows(positions.clone(), anchor_rows, position_dims);
        let anchor_positions = anchor_positions
            .reshape([batch_size, 1, position_dims])
            .expand([batch_size, self.fine_particle_count, position_dims]);
        let offset = fine_positions.clone() - anchor_positions;
        let distance_squared = offset.clone().mul(offset).sum_dim(2).squeeze_dim::<2>(2);
        let merge_score = distance_squared
            .div_scalar(self.fine_footprint_squared.max(f32::MIN_POSITIVE))
            .add(fine_detail.mul_scalar(self.merge_detail_scale))
            .neg();
        let merge_local = if self.fine_particle_count == 4 {
            Tensor::<BurnBackend, 1, Int>::arange(0..4, &positions.device())
                .reshape([1, 4])
                .expand([batch_size, 4])
        } else {
            merge_score.topk_with_indices(4, 1).1
        };
        let merge_rows = self
            .fine_indices
            .clone()
            .select(0, merge_local.reshape([batch_size * 4]))
            .reshape([batch_size, 4]);
        let merge_positions =
            gather_dynamic_row_set(positions.clone(), merge_rows.clone(), position_dims);
        let merge_states = gather_dynamic_row_set(states.clone(), merge_rows.clone(), state_dims);
        let merged_position = merge_positions.mean_dim(1);
        let merged_state = merge_states.mean_dim(1);

        let split_offsets = tensor3(
            vec![
                -self.split_radius,
                0.0,
                self.split_radius,
                0.0,
                0.0,
                -self.split_radius,
                0.0,
                self.split_radius,
            ],
            [1, 4, 2],
            &positions.device(),
        )
        .expand([batch_size, 4, 2]);
        let split_positions = coarse_positions
            .reshape([batch_size, 1, position_dims])
            .expand([batch_size, 4, position_dims])
            + split_offsets;
        let split_states = coarse_states
            .reshape([batch_size, 1, state_dims])
            .expand([batch_size, 4, state_dims]);

        let positions = replace_dynamic_row_set(
            positions,
            coarse_rows.clone().reshape([batch_size, 1]),
            merged_position,
        );
        let positions = replace_dynamic_row_set(positions, merge_rows.clone(), split_positions);
        let states = replace_dynamic_row_set(
            states,
            coarse_rows.reshape([batch_size, 1]),
            merged_state,
        );
        let states = replace_dynamic_row_set(states, merge_rows, split_states);
        let position_gate = batch_gate
            .clone()
            .reshape([batch_size, 1, 1])
            .expand([batch_size, particle_count, position_dims]);
        let state_gate = batch_gate
            .reshape([batch_size, 1, 1])
            .expand([batch_size, particle_count, state_dims]);
        let positions =
            original_positions.clone() + (positions - original_positions).mul(position_gate);
        let states = original_states.clone() + (states - original_states).mul(state_gate);
        (positions, states, active_batches)
    }

    fn apply_continuous(
        &self,
        positions: Tensor3,
        states: Tensor3,
        detail: Tensor2,
        batch_gate: Tensor1,
        active_batches: usize,
    ) -> (Tensor3, Tensor3, usize) {
        let [batch_size, particle_count, position_dims] = positions.shape().dims::<3>();
        let state_dims = states.shape().dims::<3>()[2];
        let exchange_count = self
            .events_per_interval
            .min(self.coarse_particle_count)
            .min(self.fine_particle_count);
        if exchange_count == 0 || self.min_relative_gain >= 1.0 {
            return (positions, states, 0);
        }
        let coarse_detail =
            stable_local_detail_rank(detail.clone().select(1, self.coarse_indices.clone()));
        let fine_detail =
            stable_local_detail_rank(detail.select(1, self.fine_indices.clone()));
        let coarse_rank = stable_local_detail_score(
            coarse_detail.clone(),
            self.coarse_indices.clone(),
            particle_count,
            true,
        );
        let fine_rank = stable_local_detail_score(
            fine_detail.clone(),
            self.fine_indices.clone(),
            particle_count,
            false,
        );
        let coarse_local = coarse_rank
            .topk_with_indices(exchange_count, 1)
            .1;
        let fine_local = fine_rank
            .topk_with_indices(exchange_count, 1)
            .1;
        let coarse_rows = self
            .coarse_indices
            .clone()
            .select(0, coarse_local.clone().reshape([batch_size * exchange_count]))
            .reshape([batch_size, exchange_count]);
        let fine_rows = self
            .fine_indices
            .clone()
            .select(0, fine_local.clone().reshape([batch_size * exchange_count]))
            .reshape([batch_size, exchange_count]);
        let selected_coarse_detail = coarse_detail.gather(1, coarse_local);
        let selected_fine_detail = fine_detail.gather(1, fine_local);
        let comparison_scale = selected_coarse_detail
            .clone()
            .abs()
            .max_pair(selected_fine_detail.clone().abs())
            .clamp_min(f32::MIN_POSITIVE);
        let accepted = (selected_coarse_detail
            - selected_fine_detail
            - comparison_scale.mul_scalar(self.min_relative_gain))
        .greater_elem(0.0)
        .float()
        .mul(
            batch_gate
                .reshape([batch_size, 1])
                .expand([batch_size, exchange_count]),
        );
        let coarse_positions =
            gather_dynamic_row_set(positions.clone(), coarse_rows.clone(), position_dims);
        let fine_positions =
            gather_dynamic_row_set(positions.clone(), fine_rows.clone(), position_dims);
        let coarse_states =
            gather_dynamic_row_set(states.clone(), coarse_rows.clone(), state_dims);
        let fine_states =
            gather_dynamic_row_set(states.clone(), fine_rows.clone(), state_dims);
        let correction_scale = (self
            .represented_measure
            .clone()
            .select(0, coarse_rows.clone().reshape([batch_size * exchange_count]))
            .reshape([batch_size, exchange_count])
            - self
                .represented_measure
                .clone()
                .select(0, fine_rows.clone().reshape([batch_size * exchange_count]))
                .reshape([batch_size, exchange_count]))
        .div_scalar(self.total_measure.max(f32::MIN_POSITIVE))
        .mul(accepted.clone());
        let state_correction = (coarse_states.clone() - fine_states.clone())
            .mul(
                correction_scale
                    .reshape([batch_size, exchange_count, 1])
                    .expand([batch_size, exchange_count, state_dims]),
            )
            .sum_dim(1);

        let original_positions = positions;
        let position_gate = accepted
            .clone()
            .reshape([batch_size, exchange_count, 1])
            .expand([batch_size, exchange_count, position_dims]);
        let coarse_position_replacement = coarse_positions.clone()
            + (fine_positions.clone() - coarse_positions.clone()).mul(position_gate.clone());
        let fine_position_replacement = fine_positions.clone()
            + (coarse_positions - fine_positions).mul(position_gate);
        let positions = replace_dynamic_row_set(
            original_positions.clone(),
            coarse_rows.clone(),
            coarse_position_replacement,
        );
        let positions = replace_dynamic_row_set(
            positions,
            fine_rows.clone(),
            fine_position_replacement,
        );
        let positions = project_positions_to_original_moments(
            original_positions,
            positions,
            self.represented_measure.clone(),
            self.total_measure,
        );
        let state_gate = accepted
            .reshape([batch_size, exchange_count, 1])
            .expand([batch_size, exchange_count, state_dims]);
        let coarse_state_replacement = coarse_states.clone()
            + (fine_states.clone() - coarse_states.clone()).mul(state_gate.clone());
        let fine_state_replacement =
            fine_states.clone() + (coarse_states - fine_states).mul(state_gate);
        let states = replace_dynamic_row_set(
            states,
            coarse_rows,
            coarse_state_replacement,
        );
        let states = replace_dynamic_row_set(
            states,
            fine_rows,
            fine_state_replacement,
        ) + state_correction
            .reshape([batch_size, 1, state_dims])
            .expand([batch_size, particle_count, state_dims]);
        (positions, states, active_batches * exchange_count)
    }
}

pub(super) fn stable_local_detail_rank(detail: Tensor2) -> Tensor2 {
    detail
        .mul_scalar(256.0)
        .add_scalar(0.5)
        .floor()
        .div_scalar(256.0)
}

pub(super) fn stable_local_detail_score(
    ranked_detail: Tensor2,
    rows: Tensor1Int,
    particle_count: usize,
    prefer_high_detail: bool,
) -> Tensor2 {
    let [batch_size, candidate_count] = ranked_detail.shape().dims::<2>();
    // WGPU resolves equal 1/256-ranked detail in favor of the lower global
    // row. Keep the perturbation below one quarter of a rank interval so it
    // cannot reorder distinct ranks.
    let tie_step = 1.0 / (1024.0 * (particle_count + 1) as f32);
    let lower_row_bias = rows
        .float()
        .neg()
        .add_scalar(particle_count as f32)
        .mul_scalar(tie_step)
        .reshape([1, candidate_count])
        .expand([batch_size, candidate_count]);
    let detail_score = if prefer_high_detail {
        ranked_detail
    } else {
        ranked_detail.neg()
    };
    detail_score + lower_row_bias
}

/// Applies a per-batch lower-triangular affine map that restores the weighted
/// centroid and 2D second moment of `original`. The hard topology selection is
/// detached, but this projection remains differentiable for the next TBPTT
/// chunk.
fn project_positions_to_original_moments(
    original: Tensor3,
    swapped: Tensor3,
    represented_measure: Tensor1,
    total_measure: f32,
) -> Tensor3 {
    let [batch_size, particle_count, position_dims] = original.shape().dims::<3>();
    debug_assert_eq!(position_dims, 2);
    let weights = represented_measure
        .reshape([1, particle_count, 1])
        .expand([batch_size, particle_count, 1]);
    let old_mean = original
        .clone()
        .mul(weights.clone())
        .sum_dim(1)
        .div_scalar(total_measure.max(f32::MIN_POSITIVE));
    let swapped_mean = swapped
        .clone()
        .mul(weights.clone())
        .sum_dim(1)
        .div_scalar(total_measure.max(f32::MIN_POSITIVE));
    let old_centered = original - old_mean.clone().expand([batch_size, particle_count, 2]);
    let swapped_centered =
        swapped - swapped_mean.expand([batch_size, particle_count, position_dims]);
    let covariance = |centered: Tensor3| {
        let x = centered.clone().narrow(2, 0, 1);
        let y = centered.narrow(2, 1, 1);
        let xx = x
            .clone()
            .mul(x.clone())
            .mul(weights.clone())
            .sum_dim(1)
            .div_scalar(total_measure.max(f32::MIN_POSITIVE));
        let xy = x
            .mul(y.clone())
            .mul(weights.clone())
            .sum_dim(1)
            .div_scalar(total_measure.max(f32::MIN_POSITIVE));
        let yy = y
            .clone()
            .mul(y)
            .mul(weights.clone())
            .sum_dim(1)
            .div_scalar(total_measure.max(f32::MIN_POSITIVE));
        (xx, xy, yy)
    };
    let (old_xx, old_xy, old_yy) = covariance(old_centered);
    let (new_xx, new_xy, new_yy) = covariance(swapped_centered.clone());
    let covariance_floor = 1.0e-12;
    let old_l00 = old_xx.clamp_min(covariance_floor).sqrt();
    let new_l00 = new_xx.clamp_min(covariance_floor).sqrt();
    let old_l10 = old_xy.div(old_l00.clone());
    let new_l10 = new_xy.div(new_l00.clone());
    let old_l11 = (old_yy - old_l10.clone().mul(old_l10.clone()))
        .clamp_min(covariance_floor)
        .sqrt();
    let new_l11 = (new_yy - new_l10.clone().mul(new_l10.clone()))
        .clamp_min(covariance_floor)
        .sqrt();
    let affine_00 = old_l00.div(new_l00.clone());
    let affine_10 = old_l10.div(new_l00.clone())
        - old_l11
            .clone()
            .mul(new_l10)
            .div(new_l00.mul(new_l11.clone()));
    let affine_11 = old_l11.div(new_l11);
    let centered_x = swapped_centered.clone().narrow(2, 0, 1);
    let centered_y = swapped_centered.narrow(2, 1, 1);
    let projected_x = old_mean.clone().narrow(2, 0, 1)
        + centered_x.clone().mul(affine_00);
    let projected_y =
        old_mean.narrow(2, 1, 1) + centered_x.mul(affine_10) + centered_y.mul(affine_11);
    Tensor::cat(vec![projected_x, projected_y], 2)
}

fn gather_dynamic_rows(values: Tensor3, rows: Tensor1Int, channels: usize) -> Tensor2 {
    let batch_size = values.shape().dims::<3>()[0];
    values
        .gather(
            1,
            rows.reshape([batch_size, 1, 1])
                .expand([batch_size, 1, channels]),
        )
        .squeeze_dim::<2>(1)
}

fn gather_dynamic_row_set(values: Tensor3, rows: Tensor2Int, channels: usize) -> Tensor3 {
    let [batch_size, selected_rows] = rows.shape().dims::<2>();
    values.gather(
        1,
        rows.reshape([batch_size, selected_rows, 1])
            .expand([batch_size, selected_rows, channels]),
    )
}

fn replace_dynamic_row_set(
    values: Tensor3,
    rows: Tensor2Int,
    replacements: Tensor3,
) -> Tensor3 {
    let [batch_size, particle_count, channels] = values.shape().dims::<3>();
    let selected_rows = rows.shape().dims::<2>()[1];
    let device = values.device();
    let offsets = Tensor::<BurnBackend, 1, Int>::arange(0..batch_size as i64, &device)
        .mul_scalar(particle_count as i64)
        .reshape([batch_size, 1])
        .expand([batch_size, selected_rows]);
    let flat_rows = (rows + offsets).reshape([batch_size * selected_rows]);
    let flat = values.reshape([batch_size * particle_count, channels]);
    let replacements = replacements.reshape([batch_size * selected_rows, channels]);
    let delta = replacements - flat.clone().select(0, flat_rows.clone());
    flat.select_assign(0, flat_rows, delta, IndexingUpdateOp::Add)
        .reshape([batch_size, particle_count, channels])
}
