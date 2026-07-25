use std::time::Instant;

use rand::{SeedableRng, rngs::StdRng, seq::index};

use super::{
    AdaptiveMultiscaleDatasetReport, AdaptiveMultiscaleRuleStrategy,
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig, normalize_positive_weights,
};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        ADAPTIVE_CONTROLLER_OUTPUT_DIMS, AdaptiveHierarchyMember, AdaptiveMaterialView,
        AdaptiveNpaConfig, AdaptiveParticleSet, AdaptiveProxyHierarchy, BudgetAllocation,
        allocate_resolution_budget,
        features::{
            closure_recurrent_auxiliary_dims, closure_recurrent_features_for_rows,
            controller_features_for_rows, local_residual_auxiliary_dims,
            local_residual_features_for_rows, local_residual_gate, local_rule_perception,
            material_detail_values, proxy_context,
        },
        material_footprint_radius,
        perception::rule_perception_pair,
    },
    rollout::{seed_particles_scaled, stochastic_mask},
};
use burn_automata_kernels::{
    AdaptiveGraphMetrics, AdaptivePerceptionOutput, AdaptivePerceptionPair, HashGridConfig,
    euler_step,
};

#[cfg(feature = "gpu_wgpu")]
mod gpu;
#[cfg(feature = "gpu_wgpu")]
pub use gpu::adaptive_multiscale_training_batch_wgpu_with_executor;

pub fn adaptive_multiscale_training_batch(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    validate(teacher, teacher_grid, adaptive, config)?;
    let started = Instant::now();
    let mut builder = MultiscaleDatasetBuilder::new(teacher, adaptive, config)?;
    collect_cpu_teacher_snapshots(teacher, teacher_grid, adaptive, config, |snapshot| {
        builder.append(snapshot)
    })?;
    builder.finish(started)
}

#[derive(Clone)]
struct FineTeacherSnapshot {
    rollout_index: usize,
    #[cfg(test)]
    #[allow(dead_code)]
    step_index: usize,
    positions: Vec<[f32; 4]>,
    states: Vec<f32>,
    raw_update: Vec<f32>,
    teacher_dx: Vec<f32>,
    teacher_ds: Vec<f32>,
    state_jacobian: Vec<f32>,
    captured_perception: Option<FineTeacherPerception>,
}

#[derive(Clone)]
struct FineTeacherPerception {
    base_features: Vec<f32>,
    normalized_features: Vec<f32>,
    observed_spacing: Vec<f32>,
    accepted_degree: Vec<usize>,
}

struct PreparedFineTeacherSnapshot {
    rollout_index: usize,
    fine: AdaptiveParticleSet,
    hierarchy: AdaptiveProxyHierarchy,
    detail: Vec<f32>,
    raw_update: Vec<f32>,
    teacher_dx: Vec<f32>,
    teacher_ds: Vec<f32>,
    captured_perception: Option<FineTeacherPerception>,
}

struct PreparedMultiscaleCut {
    rollout_index: usize,
    view: AdaptiveMaterialView,
    restricted_target: Vec<f32>,
    closure_mode_target_update: Vec<f32>,
    closure_basis_target_update: Vec<f32>,
    closure_mode_active: Vec<bool>,
}

struct PreparedCutTrainingRows {
    material_count: usize,
    proxy_nodes: usize,
    counterfactual_error_sum: f64,
    counterfactual_error_rows: usize,
    footprints: Vec<f32>,
    local_features: Vec<f32>,
    closure_features: Vec<f32>,
    proxy_features: Vec<f32>,
    target_update: Vec<f32>,
    closure_mode_target_update: Vec<f32>,
    closure_basis_target_update: Vec<f32>,
    closure_mode_row_weights: Vec<f32>,
    deployment_features: Vec<f32>,
    deployment_target_update: Vec<f32>,
    deployment_row_weights: Vec<f32>,
    deployment_residual_gate: Vec<f32>,
    controller_input: Vec<f32>,
    controller_targets: Vec<f32>,
    row_weights: Vec<f32>,
}

struct MultiscaleDatasetBuilder<'a> {
    teacher: &'a NpaModel,
    adaptive: &'a AdaptiveNpaConfig,
    config: &'a AdaptiveMultiscaleTrainingConfig,
    split_label_ratio: f32,
    merge_label_ratio: f32,
    input_dims: usize,
    output_dims: usize,
    local_input_dims: usize,
    closure_input_dims: usize,
    local_features: Vec<f32>,
    closure_features: Vec<f32>,
    proxy_features: Vec<f32>,
    target_update: Vec<f32>,
    closure_mode_target_update: Vec<f32>,
    closure_basis_target_update: Vec<f32>,
    closure_mode_row_weights: Vec<f32>,
    deployment_features: Vec<f32>,
    deployment_target_update: Vec<f32>,
    deployment_row_weights: Vec<f32>,
    deployment_residual_gate: Vec<f32>,
    controller_input: Vec<f32>,
    controller_targets: Vec<f32>,
    row_weights: Vec<f32>,
    all_footprints: Vec<f32>,
    sample_rngs: Vec<StdRng>,
    proxy_nodes: usize,
    generated_cuts: usize,
    generated_snapshots: usize,
    minimum_material_leaves: usize,
    maximum_material_leaves: usize,
    counterfactual_error_sum: f64,
    counterfactual_error_rows: usize,
}

impl<'a> MultiscaleDatasetBuilder<'a> {
    fn new(
        teacher: &'a NpaModel,
        adaptive: &'a AdaptiveNpaConfig,
        config: &'a AdaptiveMultiscaleTrainingConfig,
    ) -> AutomataResult<Self> {
        let (split_label_ratio, merge_label_ratio) = config.controller_label_ratios(adaptive)?;
        let input_dims = teacher.config.perception_dims();
        let output_dims = teacher.config.update_dims();
        let snapshots = snapshot_steps(config.rollout_steps, config.temporal_samples).len();
        let expected_rows =
            config.rollouts * snapshots * config.cut_leaf_counts.len() * config.rows_per_cut;
        let local_input_dims =
            input_dims + local_residual_auxiliary_dims(adaptive, teacher.config.state_dims);
        let closure_input_dims =
            input_dims + closure_recurrent_auxiliary_dims(adaptive, teacher.config.state_dims);
        let sample_rngs = (0..config.rollouts)
            .map(|rollout_index| {
                let seed = config
                    .seed
                    .wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                StdRng::seed_from_u64(seed ^ 0xa4a4_5eed)
            })
            .collect();
        Ok(Self {
            teacher,
            adaptive,
            config,
            split_label_ratio,
            merge_label_ratio,
            input_dims,
            output_dims,
            local_input_dims,
            closure_input_dims,
            local_features: Vec::with_capacity(expected_rows * local_input_dims),
            closure_features: Vec::with_capacity(
                usize::from(adaptive.closure_recurrent_mode) * expected_rows * closure_input_dims,
            ),
            proxy_features: Vec::with_capacity(
                if adaptive.proxy.enabled && adaptive.proxy.context_scale > 0.0 {
                    expected_rows * input_dims
                } else {
                    0
                },
            ),
            target_update: Vec::with_capacity(expected_rows * output_dims),
            closure_mode_target_update: Vec::with_capacity(
                usize::from(adaptive.closure_recurrent_mode) * expected_rows * output_dims,
            ),
            closure_basis_target_update: Vec::with_capacity(
                usize::from(adaptive.closure_recurrent_mode) * expected_rows * output_dims,
            ),
            closure_mode_row_weights: Vec::with_capacity(
                usize::from(adaptive.closure_recurrent_mode) * expected_rows,
            ),
            deployment_features: Vec::with_capacity(expected_rows * input_dims),
            deployment_target_update: Vec::with_capacity(expected_rows * output_dims),
            deployment_row_weights: Vec::with_capacity(expected_rows),
            deployment_residual_gate: Vec::with_capacity(expected_rows),
            controller_input: Vec::with_capacity(
                expected_rows * crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS,
            ),
            controller_targets: Vec::with_capacity(expected_rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS),
            row_weights: Vec::with_capacity(expected_rows),
            all_footprints: Vec::new(),
            sample_rngs,
            proxy_nodes: 0,
            generated_cuts: 0,
            generated_snapshots: 0,
            minimum_material_leaves: usize::MAX,
            maximum_material_leaves: 0,
            counterfactual_error_sum: 0.0,
            counterfactual_error_rows: 0,
        })
    }

    fn prepare_snapshot(
        &self,
        snapshot: FineTeacherSnapshot,
    ) -> AutomataResult<PreparedFineTeacherSnapshot> {
        let rollout_index = snapshot.rollout_index;
        if rollout_index >= self.sample_rngs.len() {
            return Err(AutomataError::InvalidModel(format!(
                "multiscale snapshot rollout {} exceeds configured rollout count {}",
                rollout_index,
                self.sample_rngs.len(),
            )));
        }
        let raw_update = snapshot.raw_update;
        let teacher_dx = snapshot.teacher_dx;
        let teacher_ds = snapshot.teacher_ds;
        let state_jacobian = snapshot.state_jacobian;
        let captured_perception = snapshot.captured_perception;
        let mut fine = AdaptiveParticleSet::from_equal_measure(
            snapshot.positions,
            snapshot.states,
            self.teacher.config.spatial_dims,
            self.teacher.config.state_dims,
            self.config.total_measure,
            self.config.bandwidth,
        )?;
        if state_jacobian.len() != fine.state_jacobian.len() {
            return Err(AutomataError::InvalidModel(format!(
                "multiscale snapshot has {} state-Jacobian values, expected {}",
                state_jacobian.len(),
                fine.state_jacobian.len(),
            )));
        }
        fine.state_jacobian = state_jacobian;
        fine.validate()?;
        let hierarchy = AdaptiveProxyHierarchy::build(&fine, self.adaptive.proxy.branch_factor)?;
        let detail = material_detail_values(
            &fine,
            &raw_update,
            self.output_dims,
            self.adaptive.base_rule_footprint().recip(),
        );
        Ok(PreparedFineTeacherSnapshot {
            rollout_index,
            fine,
            hierarchy,
            detail,
            raw_update,
            teacher_dx,
            teacher_ds,
            captured_perception,
        })
    }

    fn prepare_cut(
        &self,
        snapshot: &PreparedFineTeacherSnapshot,
        requested_leaves: usize,
    ) -> AutomataResult<PreparedMultiscaleCut> {
        let full_resolution = requested_leaves == snapshot.fine.len();
        let mut view = if full_resolution {
            AdaptiveMaterialView {
                particles: snapshot.fine.clone(),
                members: (0..snapshot.fine.len())
                    .map(AdaptiveHierarchyMember::Leaf)
                    .collect(),
                fine_to_material: (0..snapshot.fine.len()).collect(),
            }
        } else {
            snapshot.hierarchy.material_cut(
                &snapshot.fine,
                requested_leaves,
                &snapshot.detail,
                self.output_dims
                    + self.teacher.config.state_dims
                    + self.teacher.config.spatial_dims,
            )?
        };
        super::apply_multiscale_material_bandwidth(
            self.adaptive,
            self.config,
            &mut view.particles,
        )?;
        let (closure_mode_target_update, closure_basis_target_update, closure_mode_active) =
            if self.adaptive.closure_recurrent_mode {
                closure_mode_step_target(
                    self.adaptive,
                    self.config.dt,
                    &snapshot.fine,
                    &snapshot.hierarchy,
                    &mut view,
                    &snapshot.teacher_dx,
                    &snapshot.teacher_ds,
                    self.output_dims,
                )?
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
        let restricted_dx = snapshot.hierarchy.restrict_values(
            &snapshot.fine,
            &view.members,
            &snapshot.teacher_dx,
            self.teacher.config.spatial_dims,
        )?;
        let restricted_ds = snapshot.hierarchy.restrict_values(
            &snapshot.fine,
            &view.members,
            &snapshot.teacher_ds,
            self.teacher.config.state_dims,
        )?;
        // The identity cut has the teacher's native rows and ordering. Preserve
        // its raw update exactly instead of round-tripping bounded motion
        // through an approximate inverse.
        let restricted_target = if full_resolution {
            snapshot.raw_update.clone()
        } else {
            raw_update_from_restricted_step(
                &restricted_dx,
                &restricted_ds,
                &view.particles.bandwidth,
                self.teacher,
            )
        };
        Ok(PreparedMultiscaleCut {
            rollout_index: snapshot.rollout_index,
            view,
            restricted_target,
            closure_mode_target_update,
            closure_basis_target_update,
            closure_mode_active,
        })
    }

    fn append(&mut self, snapshot: FineTeacherSnapshot) -> AutomataResult<()> {
        let prepared = self.prepare_snapshot(snapshot)?;
        self.generated_snapshots += 1;
        for &requested_leaves in &self.config.cut_leaf_counts {
            let cut = self.prepare_cut(&prepared, requested_leaves)?;
            let (perception_pair, base_update) = if cut.view.particles.len() == prepared.fine.len()
                && let Some(captured) = &prepared.captured_perception
            {
                captured_fine_perception_pair(
                    captured,
                    &cut.view.members,
                    &prepared.raw_update,
                    self.teacher.config.state_dims,
                    self.teacher.config.spatial_dims,
                    self.input_dims,
                    self.output_dims,
                )?
            } else {
                let pair = rule_perception_pair(self.adaptive, self.teacher, &cut.view.particles)?;
                let base_update = self
                    .teacher
                    .forward_update_from_features(&pair.npa_compatible.features)?;
                (pair, base_update)
            };
            self.append_cut(cut, perception_pair, base_update)?;
        }
        Ok(())
    }

    fn append_cut(
        &mut self,
        cut: PreparedMultiscaleCut,
        perception_pair: AdaptivePerceptionPair,
        base_update: Vec<f32>,
    ) -> AutomataResult<()> {
        let selected = self.sample_cut_rows(cut.rollout_index, cut.view.particles.len());
        let rows = self.prepare_cut_training_rows(cut, perception_pair, base_update, &selected)?;
        self.append_prepared_cut(rows);
        Ok(())
    }

    fn sample_cut_rows(&mut self, rollout_index: usize, material_count: usize) -> Vec<usize> {
        let selected_count = self.config.rows_per_cut.min(material_count);
        let mut selected = index::sample(
            &mut self.sample_rngs[rollout_index],
            material_count,
            selected_count,
        )
        .into_vec();
        selected.sort_unstable();
        selected
    }

    fn prepare_cut_training_rows(
        &self,
        cut: PreparedMultiscaleCut,
        perception_pair: AdaptivePerceptionPair,
        base_update: Vec<f32>,
        selected: &[usize],
    ) -> AutomataResult<PreparedCutTrainingRows> {
        let material_count = cut.view.particles.len();
        if perception_pair.normalized.features.len() != material_count * self.input_dims
            || perception_pair.npa_compatible.features.len() != material_count * self.input_dims
            || base_update.len() != material_count * self.output_dims
            || cut.restricted_target.len() != material_count * self.output_dims
            || !(cut.closure_mode_target_update.is_empty()
                || (cut.closure_mode_target_update.len() == material_count * self.output_dims
                    && cut.closure_mode_active.len() == material_count))
            || selected.iter().any(|row| *row >= material_count)
        {
            return Err(AutomataError::InvalidModel(
                "prepared multiscale cut has an incompatible perception shape".to_string(),
            ));
        }
        let normalized = &perception_pair.normalized;
        let residual_perception = local_rule_perception(self.adaptive, &perception_pair);
        let local_features_for_selected = local_residual_features_for_rows(
            self.adaptive,
            &cut.view.particles,
            residual_perception,
            selected,
        )?;
        let recurrent_features = self
            .adaptive
            .closure_recurrent_mode
            .then(|| {
                closure_recurrent_features_for_rows(
                    self.adaptive,
                    &cut.view.particles,
                    normalized,
                    selected,
                )
            })
            .transpose()?;
        let proxy = if self.adaptive.proxy.enabled && self.adaptive.proxy.context_scale > 0.0 {
            Some(
                proxy_context(self.adaptive, &cut.view.particles)?.ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "multiscale training requires the proxy branch".to_string(),
                    )
                })?,
            )
        } else {
            None
        };
        let control = controller_features_for_rows(
            self.adaptive,
            &cut.view.particles,
            normalized,
            &base_update,
            selected,
        );
        let risk = per_row_update_error(
            &cut.restricted_target,
            &base_update,
            material_count,
            self.output_dims,
        );
        let counterfactual_error_sum = risk.iter().map(|value| *value as f64).sum::<f64>();
        let counterfactual_error_rows = risk.len();
        let allocation = controller_budget_allocation(
            self.adaptive,
            &risk,
            &cut.view.particles.represented_measure,
            self.teacher.config.spatial_dims,
        )?;
        let footprints = cut
            .view
            .particles
            .represented_measure
            .iter()
            .map(|measure| material_footprint_radius(*measure, self.teacher.config.spatial_dims))
            .collect::<Vec<_>>();
        let mean_measure = self.config.total_measure / material_count as f32;
        let mut rows = PreparedCutTrainingRows {
            material_count,
            proxy_nodes: proxy.as_ref().map_or(0, |proxy| proxy.node_count),
            counterfactual_error_sum,
            counterfactual_error_rows,
            footprints,
            local_features: Vec::with_capacity(selected.len() * self.local_input_dims),
            closure_features: Vec::with_capacity(
                usize::from(self.adaptive.closure_recurrent_mode)
                    * selected.len()
                    * self.closure_input_dims,
            ),
            proxy_features: Vec::with_capacity(if proxy.is_some() {
                selected.len() * self.input_dims
            } else {
                0
            }),
            target_update: Vec::with_capacity(selected.len() * self.output_dims),
            closure_mode_target_update: Vec::with_capacity(
                usize::from(self.adaptive.closure_recurrent_mode)
                    * selected.len()
                    * self.output_dims,
            ),
            closure_basis_target_update: Vec::with_capacity(
                usize::from(self.adaptive.closure_recurrent_mode)
                    * selected.len()
                    * self.output_dims,
            ),
            closure_mode_row_weights: Vec::with_capacity(
                usize::from(self.adaptive.closure_recurrent_mode) * selected.len(),
            ),
            deployment_features: Vec::with_capacity(selected.len() * self.input_dims),
            deployment_target_update: Vec::with_capacity(selected.len() * self.output_dims),
            deployment_row_weights: Vec::with_capacity(selected.len()),
            deployment_residual_gate: Vec::with_capacity(selected.len()),
            controller_input: Vec::with_capacity(
                selected.len() * crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS,
            ),
            controller_targets: Vec::with_capacity(
                selected.len() * ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
            ),
            row_weights: Vec::with_capacity(selected.len()),
        };
        for (selected_index, &row) in selected.iter().enumerate() {
            rows.local_features.extend_from_slice(
                &local_features_for_selected[selected_index * self.local_input_dims
                    ..(selected_index + 1) * self.local_input_dims],
            );
            if let Some(recurrent) = &recurrent_features {
                rows.closure_features.extend_from_slice(
                    &recurrent[selected_index * self.closure_input_dims
                        ..(selected_index + 1) * self.closure_input_dims],
                );
            }
            if let Some(proxy) = &proxy {
                rows.proxy_features.extend_from_slice(
                    &proxy.perception.features[row * self.input_dims..(row + 1) * self.input_dims],
                );
            }
            let gate =
                local_residual_gate(self.adaptive, &cut.view.particles, residual_perception, row);
            for channel in 0..self.output_dims {
                let index = row * self.output_dims + channel;
                rows.target_update.push(
                    if self.adaptive.local_rule_semantics
                        == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                    {
                        cut.restricted_target[index] - base_update[index]
                    } else if gate.abs() > 1.0e-6 {
                        (cut.restricted_target[index] - base_update[index]) / gate
                    } else {
                        0.0
                    },
                );
            }
            rows.deployment_features.extend_from_slice(
                &perception_pair.npa_compatible.features
                    [row * self.input_dims..(row + 1) * self.input_dims],
            );
            if self.adaptive.closure_recurrent_mode {
                rows.closure_mode_target_update.extend_from_slice(
                    &cut.closure_mode_target_update
                        [row * self.output_dims..(row + 1) * self.output_dims],
                );
                rows.closure_basis_target_update.extend_from_slice(
                    &cut.closure_basis_target_update
                        [row * self.output_dims..(row + 1) * self.output_dims],
                );
                rows.closure_mode_row_weights.push(
                    f32::from(cut.closure_mode_active[row])
                        * cut.view.particles.represented_measure[row]
                        / mean_measure.max(f32::MIN_POSITIVE),
                );
            }
            rows.deployment_target_update.extend_from_slice(
                &cut.restricted_target[row * self.output_dims..(row + 1) * self.output_dims],
            );
            rows.controller_input
                .extend_from_slice(&control[selected_index]);
            let current = cut.view.particles.footprint(row);
            let desired = allocation.desired_footprint[row];
            let target_bandwidth = self.config.bandwidth.max(2.0 * desired).clamp(
                self.adaptive.perception.min_bandwidth,
                self.adaptive.perception.max_bandwidth,
            );
            rows.controller_targets.extend_from_slice(&[
                (desired / self.adaptive.reference_footprint).ln(),
                (target_bandwidth / normalized.observed_spacing[row].max(f32::MIN_POSITIVE))
                    .ln()
                    .clamp(-1.5, 1.5),
                f32::from(desired < current * self.split_label_ratio),
                f32::from(desired > current * self.merge_label_ratio),
            ]);
            let measure_weight =
                cut.view.particles.represented_measure[row] / mean_measure.max(f32::MIN_POSITIVE);
            rows.row_weights.push(
                if self.adaptive.local_rule_semantics
                    == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                {
                    measure_weight
                } else {
                    measure_weight * (gate.powi(2) + self.config.residual_coordinate_weight)
                },
            );
            rows.deployment_row_weights.push(measure_weight);
            rows.deployment_residual_gate.push(gate);
        }
        Ok(rows)
    }

    fn append_prepared_cut(&mut self, rows: PreparedCutTrainingRows) {
        self.minimum_material_leaves = self.minimum_material_leaves.min(rows.material_count);
        self.maximum_material_leaves = self.maximum_material_leaves.max(rows.material_count);
        self.generated_cuts += 1;
        self.proxy_nodes += rows.proxy_nodes;
        self.counterfactual_error_sum += rows.counterfactual_error_sum;
        self.counterfactual_error_rows += rows.counterfactual_error_rows;
        self.all_footprints.extend(rows.footprints);
        self.local_features.extend(rows.local_features);
        self.closure_features.extend(rows.closure_features);
        self.proxy_features.extend(rows.proxy_features);
        self.target_update.extend(rows.target_update);
        self.closure_mode_target_update
            .extend(rows.closure_mode_target_update);
        self.closure_basis_target_update
            .extend(rows.closure_basis_target_update);
        self.closure_mode_row_weights
            .extend(rows.closure_mode_row_weights);
        self.deployment_features.extend(rows.deployment_features);
        self.deployment_target_update
            .extend(rows.deployment_target_update);
        self.deployment_row_weights
            .extend(rows.deployment_row_weights);
        self.deployment_residual_gate
            .extend(rows.deployment_residual_gate);
        self.controller_input.extend(rows.controller_input);
        self.controller_targets.extend(rows.controller_targets);
        self.row_weights.extend(rows.row_weights);
    }

    fn finish(mut self, started: Instant) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
        let rows = self.row_weights.len();
        normalize_positive_weights(&mut self.row_weights, "multiscale row")?;
        normalize_positive_weights(&mut self.deployment_row_weights, "deployment row")?;
        if !self.closure_mode_row_weights.is_empty() {
            normalize_positive_weights(&mut self.closure_mode_row_weights, "closure-mode row")?;
        }
        let footprint_mean = mean(&self.all_footprints);
        let footprint_variance = self
            .all_footprints
            .iter()
            .map(|value| (*value - footprint_mean).powi(2))
            .sum::<f32>()
            / self.all_footprints.len().max(1) as f32;
        let report = AdaptiveMultiscaleDatasetReport {
            rollouts: self.config.rollouts,
            snapshots: self.generated_snapshots,
            cuts: self.generated_cuts,
            rows,
            minimum_material_leaves: self.minimum_material_leaves,
            maximum_material_leaves: self.maximum_material_leaves,
            minimum_footprint: self
                .all_footprints
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min),
            maximum_footprint: self
                .all_footprints
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            footprint_coefficient_of_variation: footprint_variance.sqrt()
                / footprint_mean.max(f32::MIN_POSITIVE),
            mean_proxy_nodes: self.proxy_nodes as f32 / self.generated_cuts.max(1) as f32,
            mean_counterfactual_error: (self.counterfactual_error_sum
                / self.counterfactual_error_rows.max(1) as f64)
                as f32,
            mean_teacher_update_error: 0.0,
            teacher_update_p99_absolute: super::absolute_percentile(
                &self.deployment_target_update,
                0.99,
            ),
            maximum_teacher_update_absolute: self
                .deployment_target_update
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
            closure_target_p99_absolute: super::absolute_percentile(
                &self.closure_mode_target_update,
                0.99,
            ),
            maximum_closure_target_absolute: self
                .closure_mode_target_update
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
            closure_basis_target_p99_absolute: super::absolute_percentile(
                &self.closure_basis_target_update,
                0.99,
            ),
            maximum_closure_basis_target_absolute: self
                .closure_basis_target_update
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
            ..AdaptiveMultiscaleDatasetReport::default()
        };
        let report = AdaptiveMultiscaleDatasetReport {
            generation_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            ..report
        };
        let batch = AdaptiveMultiscaleTrainingBatch {
            local_features: self.local_features,
            closure_features: self.closure_features,
            proxy_features: self.proxy_features,
            target_update: self.target_update,
            closure_mode_target_update: self.closure_mode_target_update,
            closure_basis_target_update: self.closure_basis_target_update,
            closure_mode_row_weights: self.closure_mode_row_weights,
            deployment_features: self.deployment_features,
            deployment_target_update: self.deployment_target_update,
            deployment_row_weights: self.deployment_row_weights,
            deployment_residual_gate: self.deployment_residual_gate,
            controller_features: self.controller_input,
            controller_targets: self.controller_targets,
            row_weights: self.row_weights,
            rows,
            report,
        };
        batch.validate(self.input_dims, self.output_dims)?;
        Ok(batch)
    }
}

#[allow(clippy::too_many_arguments)]
fn captured_fine_perception_pair(
    captured: &FineTeacherPerception,
    members: &[AdaptiveHierarchyMember],
    raw_update: &[f32],
    state_dims: usize,
    spatial_dims: usize,
    feature_dims: usize,
    output_dims: usize,
) -> AutomataResult<(AdaptivePerceptionPair, Vec<f32>)> {
    let rows = members.len();
    if captured.base_features.len() != rows * feature_dims
        || captured.normalized_features.len() != rows * feature_dims
        || captured.observed_spacing.len() != rows
        || captured.accepted_degree.len() != rows
        || raw_update.len() != rows * output_dims
    {
        return Err(AutomataError::InvalidModel(
            "captured fine perception has an incompatible shape".to_string(),
        ));
    }
    let source_rows = members
        .iter()
        .map(|member| match member {
            AdaptiveHierarchyMember::Leaf(index) => Ok(*index),
            AdaptiveHierarchyMember::Proxy(_) => Err(AutomataError::InvalidModel(
                "full-resolution material cut unexpectedly contains a proxy".to_string(),
            )),
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let base_features = reorder_rows(&captured.base_features, feature_dims, &source_rows);
    let normalized_features =
        reorder_rows(&captured.normalized_features, feature_dims, &source_rows);
    let observed_spacing = source_rows
        .iter()
        .map(|&row| captured.observed_spacing[row])
        .collect::<Vec<_>>();
    let accepted_degree = source_rows
        .iter()
        .map(|&row| captured.accepted_degree[row])
        .collect::<Vec<_>>();
    let base = captured_perception_output(
        base_features,
        vec![0.0; rows],
        observed_spacing.clone(),
        accepted_degree.clone(),
        state_dims,
        spatial_dims,
        feature_dims,
    );
    let normalized = captured_perception_output(
        normalized_features,
        vec![0.0; rows],
        observed_spacing,
        accepted_degree,
        state_dims,
        spatial_dims,
        feature_dims,
    );
    Ok((
        AdaptivePerceptionPair {
            normalized,
            npa_compatible: base,
        },
        reorder_rows(raw_update, output_dims, &source_rows),
    ))
}

pub(super) fn captured_perception_output(
    features: Vec<f32>,
    coarse_exposure: Vec<f32>,
    observed_spacing: Vec<f32>,
    accepted_degree: Vec<usize>,
    state_dims: usize,
    spatial_dims: usize,
    feature_dims: usize,
) -> AdaptivePerceptionOutput {
    let rows = observed_spacing.len();
    debug_assert_eq!(coarse_exposure.len(), rows);
    let gradient_start = 2 * state_dims;
    let gradient_dims = state_dims * spatial_dims;
    let occupancy_start = gradient_start + gradient_dims;
    let mut normalized_state = Vec::with_capacity(rows * state_dims);
    let mut state_gradient = Vec::with_capacity(rows * gradient_dims);
    let mut occupancy_gradient = Vec::with_capacity(rows * spatial_dims);
    for row in features.chunks_exact(feature_dims) {
        normalized_state.extend_from_slice(&row[..state_dims]);
        state_gradient.extend_from_slice(&row[gradient_start..gradient_start + gradient_dims]);
        occupancy_gradient.extend_from_slice(&row[occupancy_start..occupancy_start + spatial_dims]);
    }
    AdaptivePerceptionOutput {
        features,
        normalized_state,
        state_gradient,
        occupancy_gradient,
        partition: vec![0.0; rows],
        coarse_exposure,
        observed_spacing,
        moment_condition: vec![0.0; rows],
        moment_fallback: vec![false; rows],
        accepted_degree,
        graph: AdaptiveGraphMetrics::default(),
        feature_dims,
    }
}

fn reorder_rows(values: &[f32], row_dims: usize, source_rows: &[usize]) -> Vec<f32> {
    let mut reordered = Vec::with_capacity(source_rows.len() * row_dims);
    for &source in source_rows {
        reordered.extend_from_slice(&values[source * row_dims..(source + 1) * row_dims]);
    }
    reordered
}

fn collect_cpu_teacher_snapshots(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    mut append: impl FnMut(FineTeacherSnapshot) -> AutomataResult<()>,
) -> AutomataResult<()> {
    let snapshots = snapshot_steps(config.rollout_steps, config.temporal_samples);
    for rollout_index in 0..config.rollouts {
        let rollout_seed = config
            .seed
            .wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (mut positions, mut states) = seed_particles_scaled(
            1,
            config.fine_particle_count,
            teacher.config.state_dims,
            teacher.config.spatial_dims,
            rollout_seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
        );
        let mut mask_rng = StdRng::seed_from_u64(rollout_seed ^ 0x5eed);
        let mut snapshot_cursor = 0;
        for step_index in 0..=config.rollout_steps {
            let teacher_step = teacher.step_cpu(
                &positions,
                &states,
                1,
                config.fine_particle_count,
                teacher_grid,
                config.dt,
                None,
            )?;
            if snapshots.get(snapshot_cursor).copied() == Some(step_index) {
                let raw_update =
                    teacher.forward_update_from_features(&teacher_step.perception.features)?;
                let fine_particles = AdaptiveParticleSet::from_equal_measure(
                    positions.clone(),
                    states.clone(),
                    teacher.config.spatial_dims,
                    teacher.config.state_dims,
                    config.total_measure,
                    config.bandwidth,
                )?;
                let normalized = rule_perception_pair(adaptive, teacher, &fine_particles)?;
                let gradient_scale = if teacher.config.scale_equivariant() {
                    teacher_grid.eps / teacher.config.eps0.max(f32::MIN_POSITIVE)
                } else {
                    1.0
                };
                let mut state_jacobian = Vec::with_capacity(
                    config.fine_particle_count
                        * teacher.config.state_dims
                        * teacher.config.spatial_dims,
                );
                for gradient in teacher_step
                    .perception
                    .state_gradient
                    .chunks_exact(teacher.config.state_dims * teacher.config.spatial_dims)
                {
                    state_jacobian.extend(
                        crate::adaptive::perception::decode_physical_state_gradient(
                            gradient,
                            teacher.config.state_dims,
                            teacher.config.spatial_dims,
                            gradient_scale,
                            teacher.config.log_norm_grad,
                        ),
                    );
                }
                let teacher_dx = teacher_step
                    .dx
                    .iter()
                    .flat_map(|value| value[..teacher.config.spatial_dims].iter().copied())
                    .collect::<Vec<_>>();
                append(FineTeacherSnapshot {
                    rollout_index,
                    #[cfg(test)]
                    step_index,
                    positions: positions.clone(),
                    states: states.clone(),
                    raw_update,
                    teacher_dx,
                    teacher_ds: teacher_step.ds.clone(),
                    state_jacobian,
                    captured_perception: Some(FineTeacherPerception {
                        base_features: teacher_step.perception.features.clone(),
                        normalized_features: normalized.normalized.features,
                        observed_spacing: normalized.normalized.observed_spacing,
                        accepted_degree: normalized.normalized.accepted_degree,
                    }),
                })?;
                snapshot_cursor += 1;
            }
            if step_index == config.rollout_steps {
                break;
            }
            let update_mask = stochastic_mask(
                config.fine_particle_count,
                config.update_prob,
                &mut mask_rng,
            );
            (positions, states) = euler_step(
                &positions,
                &states,
                &teacher_step.dx,
                &teacher_step.ds,
                1,
                config.fine_particle_count,
                teacher.config.state_dims,
                teacher_grid,
                config.dt,
                Some(&update_mask),
            )?;
        }
    }
    Ok(())
}

pub(super) fn controller_budget_allocation(
    adaptive: &AdaptiveNpaConfig,
    risk: &[f32],
    represented_measure: &[f32],
    spatial_dims: usize,
) -> AutomataResult<BudgetAllocation> {
    // Every static hierarchy cut supervises the same runtime budget. Using the
    // cut's current leaf count here teaches a different normalization problem
    // at every scale and leaves the controller out of distribution when a
    // coarse seed must refine toward `target_leaves`.
    allocate_resolution_budget(
        risk,
        represented_measure,
        spatial_dims,
        2.0,
        adaptive.reference_footprint,
        adaptive.min_footprint,
        adaptive.max_footprint,
        adaptive.target_leaves,
    )
}

fn per_row_update_error(
    target: &[f32],
    prediction: &[f32],
    rows: usize,
    output_dims: usize,
) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            ((0..output_dims)
                .map(|channel| {
                    let index = row * output_dims + channel;
                    (target[index] - prediction[index]).powi(2)
                })
                .sum::<f32>()
                / output_dims as f32)
                .sqrt()
                .max(1.0e-6)
        })
        .collect()
}

pub(super) fn raw_update_from_restricted_step(
    dx: &[f32],
    ds: &[f32],
    bandwidth: &[f32],
    teacher: &NpaModel,
) -> Vec<f32> {
    let rows = bandwidth.len();
    let spatial_dims = teacher.config.spatial_dims;
    let output_dims = teacher.config.update_dims();
    let mut output = vec![0.0; rows * output_dims];
    for row in 0..rows {
        let spatial = &dx[row * spatial_dims..(row + 1) * spatial_dims];
        let norm = spatial
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let scale = (teacher.config.alpha * teacher.config.motion_eps(bandwidth[row]))
            .max(f32::MIN_POSITIVE);
        let normalized = (norm / scale).clamp(0.0, 0.999);
        let raw_norm = normalized / (1.0 - normalized).max(1.0e-4);
        for axis in 0..spatial_dims {
            output[row * output_dims + axis] = if norm > 1.0e-12 {
                spatial[axis] * raw_norm / norm
            } else {
                0.0
            };
        }
        output[row * output_dims + spatial_dims..(row + 1) * output_dims].copy_from_slice(
            &ds[row * teacher.config.state_dims..(row + 1) * teacher.config.state_dims],
        );
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn closure_mode_step_target(
    adaptive: &AdaptiveNpaConfig,
    dt: f32,
    fine: &AdaptiveParticleSet,
    hierarchy: &AdaptiveProxyHierarchy,
    view: &mut AdaptiveMaterialView,
    teacher_dx: &[f32],
    teacher_ds: &[f32],
    output_dims: usize,
) -> AutomataResult<(Vec<f32>, Vec<f32>, Vec<bool>)> {
    if !dt.is_finite()
        || dt <= 0.0
        || teacher_dx.len() != fine.len() * fine.spatial_dims
        || teacher_ds.len() != fine.len() * fine.state_dims
        || output_dims != fine.spatial_dims + fine.state_dims
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive closure teacher step shape mismatch".to_owned(),
        ));
    }
    let (current, _) = super::super::closure::restrict_first_closure_mode_for_members(
        fine,
        hierarchy,
        &view.members,
    )?;
    view.particles.closure_mode.clone_from(&current.values);
    view.particles.closure_basis.clone_from(&current.basis);
    view.particles.closure_phase.clone_from(&current.phase);
    view.particles.validate()?;

    let mut next = fine.clone();
    for row in 0..next.len() {
        for axis in 0..next.spatial_dims {
            let index = row * next.spatial_dims + axis;
            next.positions[row][axis] = (next.positions[row][axis] + dt * teacher_dx[index])
                .clamp(adaptive.domain_min[axis], adaptive.domain_max[axis]);
        }
        for channel in 0..next.state_dims {
            let index = row * next.state_dims + channel;
            next.states[index] += dt * teacher_ds[index];
        }
    }
    let (next_mode, _) = super::super::closure::restrict_first_closure_mode_for_members_oriented(
        &next,
        hierarchy,
        &view.members,
        Some(&current.basis),
    )?;
    if current.active != next_mode.active || current.values.len() != next_mode.values.len() {
        return Err(AutomataError::InvalidModel(
            "fixed adaptive cut changed closure-mode topology during one teacher step".to_owned(),
        ));
    }
    let mut target = vec![0.0; view.particles.len() * output_dims];
    let mut basis_target = vec![0.0; view.particles.len() * output_dims];
    for row in 0..view.particles.len() {
        if !current.active[row] {
            continue;
        }
        for axis in 0..fine.spatial_dims {
            target[row * output_dims + axis] =
                (next_mode.phase[row * 2 + axis] - current.phase[row * 2 + axis]) / dt;
        }
        for channel in 0..fine.state_dims {
            let mode_index = row * fine.state_dims + channel;
            target[row * output_dims + fine.spatial_dims + channel] =
                (next_mode.values[mode_index] - current.values[mode_index]) / dt;
        }
        for component in 0..4 {
            basis_target[row * output_dims + component] =
                (next_mode.basis[row * 4 + component] - current.basis[row * 4 + component]) / dt;
        }
    }
    Ok((target, basis_target, current.active))
}

pub(super) fn controller_target(
    adaptive: &AdaptiveNpaConfig,
    particles: &AdaptiveParticleSet,
    observed_spacing: &[f32],
    desired_footprint: &[f32],
    teacher_bandwidth: f32,
    split_label_ratio: f32,
    merge_label_ratio: f32,
) -> Vec<f32> {
    let mut targets = Vec::with_capacity(particles.len() * ADAPTIVE_CONTROLLER_OUTPUT_DIMS);
    for row in 0..particles.len() {
        let current = particles.footprint(row);
        let desired = desired_footprint[row];
        let desired_log = (desired / adaptive.reference_footprint).ln();
        let target_bandwidth = teacher_bandwidth.max(2.0 * desired).clamp(
            adaptive.perception.min_bandwidth,
            adaptive.perception.max_bandwidth,
        );
        let zeta = (target_bandwidth / observed_spacing[row].max(f32::MIN_POSITIVE))
            .ln()
            .clamp(-1.5, 1.5);
        let split_label = f32::from(desired < current * split_label_ratio);
        let merge_label = f32::from(desired > current * merge_label_ratio);
        targets.extend_from_slice(&[desired_log, zeta, split_label, merge_label]);
    }
    targets
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn snapshot_steps(rollout_steps: usize, temporal_samples: usize) -> Vec<usize> {
    if temporal_samples == 1 || rollout_steps == 0 {
        return vec![rollout_steps];
    }
    let mut steps = Vec::with_capacity(temporal_samples);
    for sample_index in 0..temporal_samples {
        let step = sample_index * rollout_steps / (temporal_samples - 1);
        if steps.last().copied() != Some(step) {
            steps.push(step);
        }
    }
    steps
}

fn validate(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &AdaptiveNpaConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<()> {
    teacher.validate()?;
    teacher_grid.validate().map_err(AutomataError::from)?;
    adaptive.validate()?;
    let cuts_valid = !config.cut_leaf_counts.is_empty()
        && config
            .cut_leaf_counts
            .iter()
            .all(|count| *count > 0 && *count <= config.fine_particle_count);
    let proxy_configuration_valid = match config.rule_strategy {
        AdaptiveMultiscaleRuleStrategy::Residual => {
            (adaptive.local_rule_semantics == crate::adaptive::AdaptiveLocalRuleSemantics::Residual
                && adaptive.proxy.enabled)
                || (adaptive.local_rule_semantics
                    == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                    && !adaptive.proxy.enabled)
        }
        AdaptiveMultiscaleRuleStrategy::CoarseReplacement => {
            !adaptive.proxy.enabled
                && adaptive.local_rule_semantics
                    == crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement
        }
        AdaptiveMultiscaleRuleStrategy::FullNormalized => !adaptive.proxy.enabled,
    };
    if teacher.config.spatial_dims != adaptive.spatial_dims
        || teacher_grid.dim != teacher.config.spatial_dims
        || adaptive.perception.feature_dims(teacher.config.state_dims)
            != teacher.config.perception_dims()
        || !proxy_configuration_valid
        || config.fine_particle_count == 0
        || !cuts_valid
        || (adaptive.closure_recurrent_mode
            && !config
                .cut_leaf_counts
                .iter()
                .any(|count| *count < config.fine_particle_count))
        || config.rollouts == 0
        || config.temporal_samples == 0
        || config.rows_per_cut == 0
        || config.validation_rollouts == 0
        || config.steps == 0
        || config.report_interval == 0
        || config.controller_steps == 0
        || (config.gradient_reduction_chunk_rows != 0
            && (config.gradient_reduction_chunk_rows < 128
                || !config.gradient_reduction_chunk_rows.is_power_of_two()))
        || !config.residual_coordinate_weight.is_finite()
        || config.residual_coordinate_weight < 0.0
        || !config.local_residual_training_scale.is_finite()
        || config.local_residual_training_scale < 0.0
        || !config.proxy_residual_training_scale.is_finite()
        || config.proxy_residual_training_scale < 0.0
        || !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.update_prob.is_finite()
        || !(0.0..=1.0).contains(&config.update_prob)
        || !config.seed_scale.is_finite()
        || config.seed_scale <= 0.0
        || !config.total_measure.is_finite()
        || config.total_measure <= 0.0
        || !config.bandwidth.is_finite()
        || config.bandwidth < adaptive.perception.min_bandwidth
        || config.bandwidth > adaptive.perception.max_bandwidth
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive multiscale training config".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, adaptive::AdaptiveRulePerception, upstream_growing_2d_hashgrid};

    #[test]
    fn multiscale_batch_contains_conservative_mixed_resolution_and_proxy_rows() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.proxy.enabled = true;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        assert!(batch.rows > 0);
        assert_eq!(batch.report.minimum_material_leaves, 8);
        assert_eq!(batch.report.maximum_material_leaves, 32);
        assert!(batch.report.maximum_footprint > batch.report.minimum_footprint);
        assert!(batch.report.mean_proxy_nodes > 0.0);
        assert!(batch.report.mean_counterfactual_error > 0.0);
    }

    #[test]
    fn recurrent_closure_batch_contains_basis_phase_and_state_teacher_derivatives() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 17);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.proxy.enabled = true;
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = true;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 2,
            rollouts: 1,
            temporal_samples: 3,
            rows_per_cut: 32,
            validation_rollouts: 1,
            update_prob: 1.0,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let output_dims = teacher.config.update_dims();
        assert_eq!(
            batch.closure_mode_target_update.len(),
            batch.rows * output_dims
        );
        assert_eq!(
            batch.closure_basis_target_update.len(),
            batch.rows * output_dims
        );
        assert_eq!(batch.closure_mode_row_weights.len(), batch.rows);
        assert!(
            batch
                .closure_mode_row_weights
                .iter()
                .any(|weight| *weight > 0.0)
        );
        assert!(batch.closure_mode_row_weights.contains(&0.0));
        assert!(
            batch
                .closure_mode_target_update
                .chunks_exact(output_dims)
                .zip(&batch.closure_mode_row_weights)
                .any(|(target, weight)| {
                    *weight > 0.0
                        && target[..teacher.config.spatial_dims]
                            .iter()
                            .any(|value| value.abs() > 1.0e-7)
                })
        );
        assert!(
            batch
                .closure_mode_target_update
                .chunks_exact(output_dims)
                .zip(&batch.closure_mode_row_weights)
                .any(|(target, weight)| {
                    *weight > 0.0
                        && target[teacher.config.spatial_dims..]
                            .iter()
                            .any(|value| value.abs() > 1.0e-7)
                })
        );
        assert!(
            batch
                .closure_basis_target_update
                .chunks_exact(output_dims)
                .zip(&batch.closure_mode_row_weights)
                .any(|(target, weight)| {
                    *weight > 0.0 && target[..4].iter().any(|value| value.abs() > 1.0e-7)
                })
        );
    }

    #[test]
    fn compatible_residual_dataset_accepts_static_and_recurrent_closure_moments() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 23);
        let grid = upstream_growing_2d_hashgrid();
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.local_rule_semantics =
            crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.proxy.enabled = false;
        adaptive.closure_moment_features = true;
        adaptive.closure_recurrent_mode = false;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![16, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 1,
            rows_per_cut: 16,
            validation_rollouts: 1,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };

        validate(&teacher, &grid, &adaptive, &config).unwrap();

        adaptive.closure_recurrent_mode = true;
        validate(&teacher, &grid, &adaptive, &config).unwrap();
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    #[ignore = "requires a WGPU device"]
    fn resident_teacher_dataset_matches_cpu_reference_without_stochastic_masking() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = true;
        adaptive.proxy.context_scale = 0.0;
        adaptive.min_leaves = 8;
        adaptive.max_leaves = 64;
        adaptive.target_leaves = 32;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 16, 32],
            rollout_steps: 2,
            rollouts: 2,
            temporal_samples: 3,
            rows_per_cut: 32,
            validation_rollouts: 1,
            update_prob: 1.0,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let grid = upstream_growing_2d_hashgrid();
        let mut expected_snapshots = Vec::new();
        collect_cpu_teacher_snapshots(&teacher, &grid, &adaptive, &config, |snapshot| {
            expected_snapshots.push(snapshot);
            Ok(())
        })
        .unwrap();
        let executor = crate::gpu::WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let mut actual_snapshots = Vec::new();
        gpu::collect_teacher_snapshots(
            &executor,
            &teacher,
            &grid,
            &adaptive,
            &config,
            |snapshot| {
                actual_snapshots.push(snapshot);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual_snapshots.len(), expected_snapshots.len());
        let max_error = |actual: &[f32], expected: &[f32]| {
            actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max)
        };
        let max_error_detail = |actual: &[f32], expected: &[f32]| {
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .max_by(
                    |(_, (actual_lhs, expected_lhs)), (_, (actual_rhs, expected_rhs))| {
                        (*actual_lhs - *expected_lhs)
                            .abs()
                            .total_cmp(&(*actual_rhs - *expected_rhs).abs())
                    },
                )
                .map(|(index, (actual, expected))| (index, *actual, *expected))
                .unwrap_or((0, 0.0, 0.0))
        };
        for expected in &expected_snapshots {
            let actual = actual_snapshots
                .iter()
                .find(|snapshot| {
                    snapshot.rollout_index == expected.rollout_index
                        && snapshot.step_index == expected.step_index
                })
                .expect("resident teacher snapshot identity");
            let position_error = actual
                .positions
                .iter()
                .zip(&expected.positions)
                .flat_map(|(actual, expected)| {
                    actual
                        .iter()
                        .zip(expected)
                        .map(|(actual, expected)| (actual - expected).abs())
                })
                .fold(0.0_f32, f32::max);
            let state_error = max_error(&actual.states, &expected.states);
            let raw_error = max_error(&actual.raw_update, &expected.raw_update);
            let dx_error = max_error(&actual.teacher_dx, &expected.teacher_dx);
            let ds_error = max_error(&actual.teacher_ds, &expected.teacher_ds);
            let jacobian_error = max_error(&actual.state_jacobian, &expected.state_jacobian);
            assert!(
                position_error <= 2.0e-3
                    && state_error <= 2.0e-3
                    && raw_error <= 2.0e-3
                    && dx_error <= 2.0e-3
                    && ds_error <= 2.0e-3
                    && jacobian_error <= 2.0e-3,
                "resident teacher snapshot rollout={} step={} mismatch: position={position_error:.3e} state={state_error:.3e} raw={raw_error:.3e} dx={dx_error:.3e} ds={ds_error:.3e} jacobian={jacobian_error:.3e}",
                expected.rollout_index,
                expected.step_index,
            );
        }
        let perception_probe = MultiscaleDatasetBuilder::new(&teacher, &adaptive, &config).unwrap();
        let prepared_probe = perception_probe
            .prepare_snapshot(actual_snapshots[0].clone())
            .unwrap();
        for &leaf_count in &config.cut_leaf_counts {
            let cut = perception_probe
                .prepare_cut(&prepared_probe, leaf_count)
                .unwrap();
            let cpu = rule_perception_pair(&adaptive, &teacher, &cut.view.particles).unwrap();
            let gpu = perception_probe
                .capture_cut_perceptions(&executor, &grid, &[&cut])
                .unwrap()
                .pop()
                .unwrap()
                .0;
            let normalized_error = max_error(&gpu.normalized.features, &cpu.normalized.features);
            let base_error = max_error(&gpu.npa_compatible.features, &cpu.npa_compatible.features);
            let spacing_error = max_error(
                &gpu.normalized.observed_spacing,
                &cpu.normalized.observed_spacing,
            );
            let degree_mismatches = gpu
                .normalized
                .accepted_degree
                .iter()
                .zip(&cpu.normalized.accepted_degree)
                .filter(|(actual, expected)| actual != expected)
                .count();
            eprintln!(
                "cut perception parity leaves={leaf_count}: normalized={normalized_error:.3e} base={base_error:.3e} spacing={spacing_error:.3e} degree_mismatches={degree_mismatches}",
            );
        }
        for snapshots in actual_snapshots.chunks(config.rollouts) {
            let prepared = snapshots
                .iter()
                .cloned()
                .map(|snapshot| perception_probe.prepare_snapshot(snapshot))
                .collect::<AutomataResult<Vec<_>>>()
                .unwrap();
            for &leaf_count in &config.cut_leaf_counts {
                let cuts = prepared
                    .iter()
                    .map(|snapshot| perception_probe.prepare_cut(snapshot, leaf_count))
                    .collect::<AutomataResult<Vec<_>>>()
                    .unwrap();
                let cut_refs = cuts.iter().collect::<Vec<_>>();
                let gpu = perception_probe
                    .capture_cut_perceptions(&executor, &grid, &cut_refs)
                    .unwrap();
                for (lane, (cut, (gpu, _))) in cuts.iter().zip(gpu).enumerate() {
                    let cpu =
                        rule_perception_pair(&adaptive, &teacher, &cut.view.particles).unwrap();
                    let (max_index, _, _) =
                        max_error_detail(&gpu.normalized.features, &cpu.normalized.features);
                    let max_row = max_index / cpu.normalized.feature_dims;
                    eprintln!(
                        "batched cut parity step={} lane={lane} leaves={leaf_count}: normalized={:.3e} base={:.3e} max_row={max_row} cpu_condition={:.3e} cpu_fallback={} fallback_rows={}",
                        snapshots[0].step_index,
                        max_error(&gpu.normalized.features, &cpu.normalized.features),
                        max_error(&gpu.npa_compatible.features, &cpu.npa_compatible.features),
                        cpu.normalized.moment_condition[max_row],
                        cpu.normalized.moment_fallback[max_row],
                        cpu.normalized
                            .moment_fallback
                            .iter()
                            .filter(|value| **value)
                            .count(),
                    );
                }
            }
        }
        for snapshot in &actual_snapshots {
            let prepared = perception_probe.prepare_snapshot(snapshot.clone()).unwrap();
            let cpu = rule_perception_pair(&adaptive, &teacher, &prepared.fine).unwrap();
            let captured = prepared
                .captured_perception
                .as_ref()
                .expect("resident snapshot perception");
            eprintln!(
                "fine capture parity rollout={} step={}: normalized={:.3e} base={:.3e} spacing={:.3e} degree_mismatches={}",
                prepared.rollout_index,
                snapshot.step_index,
                max_error(&captured.normalized_features, &cpu.normalized.features),
                max_error(&captured.base_features, &cpu.npa_compatible.features),
                max_error(&captured.observed_spacing, &cpu.normalized.observed_spacing),
                captured
                    .accepted_degree
                    .iter()
                    .zip(&cpu.normalized.accepted_degree)
                    .filter(|(actual, expected)| actual != expected)
                    .count(),
            );
            let coarse_cut = perception_probe
                .prepare_cut(&prepared, config.cut_leaf_counts[0])
                .unwrap();
            let coarse_cpu =
                rule_perception_pair(&adaptive, &teacher, &coarse_cut.view.particles).unwrap();
            let coarse_gpu = perception_probe
                .capture_cut_perceptions(&executor, &grid, &[&coarse_cut])
                .unwrap()
                .pop()
                .unwrap()
                .0;
            eprintln!(
                "single coarse parity rollout={} step={}: normalized={:.3e} base={:.3e}",
                prepared.rollout_index,
                snapshot.step_index,
                max_error(
                    &coarse_gpu.normalized.features,
                    &coarse_cpu.normalized.features
                ),
                max_error(
                    &coarse_gpu.npa_compatible.features,
                    &coarse_cpu.npa_compatible.features
                ),
            );
        }
        let mut expected_builder =
            MultiscaleDatasetBuilder::new(&teacher, &adaptive, &config).unwrap();
        for mut snapshot in actual_snapshots.clone() {
            snapshot.captured_perception = None;
            expected_builder.append(snapshot).unwrap();
        }
        let expected = expected_builder.finish(Instant::now()).unwrap();
        let mut actual_builder =
            MultiscaleDatasetBuilder::new(&teacher, &adaptive, &config).unwrap();
        for snapshots in actual_snapshots.chunks(config.rollouts) {
            actual_builder
                .append_wgpu_group(&executor, &grid, snapshots.to_vec())
                .unwrap();
        }
        let actual = actual_builder.finish(Instant::now()).unwrap();

        assert_eq!(actual.rows, expected.rows);
        assert_eq!(actual.report.snapshots, expected.report.snapshots);
        assert_eq!(actual.report.cuts, expected.report.cuts);
        assert_eq!(
            actual.report.minimum_material_leaves,
            expected.report.minimum_material_leaves,
        );
        assert_eq!(
            actual.report.maximum_material_leaves,
            expected.report.maximum_material_leaves,
        );
        let max_target_error = max_error(
            &actual.deployment_target_update,
            &expected.deployment_target_update,
        );
        let local_feature_error = max_error(&actual.local_features, &expected.local_features);
        let residual_target_error = max_error(&actual.target_update, &expected.target_update);
        let deployment_feature_error =
            max_error(&actual.deployment_features, &expected.deployment_features);
        let controller_feature_error =
            max_error(&actual.controller_features, &expected.controller_features);
        eprintln!(
            "resident teacher batch parity: restricted_target={max_target_error:.3e} local_features={local_feature_error:.3e} residual_target={residual_target_error:.3e} deployment_features={deployment_feature_error:.3e} controller_features={controller_feature_error:.3e}",
        );
        eprintln!(
            "max mismatch detail: local={:?} residual={:?} deployment={:?} controller={:?}",
            max_error_detail(&actual.local_features, &expected.local_features),
            max_error_detail(&actual.target_update, &expected.target_update),
            max_error_detail(&actual.deployment_features, &expected.deployment_features),
            max_error_detail(&actual.controller_features, &expected.controller_features),
        );
        let local_width = actual.local_features.len() / actual.rows;
        let (local_index, _, _) =
            max_error_detail(&actual.local_features, &expected.local_features);
        let local_column = local_index % local_width;
        let actual_value = actual.local_features[local_index];
        let nearest_expected_row = expected
            .local_features
            .chunks_exact(local_width)
            .enumerate()
            .min_by(|(_, lhs), (_, rhs)| {
                (lhs[local_column] - actual_value)
                    .abs()
                    .total_cmp(&(rhs[local_column] - actual_value).abs())
            })
            .map(|(row, _)| row)
            .unwrap_or_default();
        eprintln!(
            "local mismatch row={} column={} width={} input_dims={} nearest_expected_row={nearest_expected_row}",
            local_index / local_width,
            local_column,
            local_width,
            teacher.config.perception_dims(),
        );
        assert!(
            max_target_error < 2.0e-3,
            "resident teacher target error {max_target_error:.3e}",
        );
        assert!(
            local_feature_error < 2.0e-3
                && residual_target_error < 2.0e-3
                && deployment_feature_error < 2.0e-3
                && controller_feature_error < 2.0e-3,
            "resident teacher objective mismatch: local={local_feature_error:.3e} residual={residual_target_error:.3e} deployment={deployment_feature_error:.3e} controller={controller_feature_error:.3e}",
        );
    }

    #[test]
    fn fine_equal_measure_cut_has_near_zero_frozen_base_residual() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        adaptive.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        adaptive.proxy.enabled = true;
        adaptive.min_leaves = 8;
        adaptive.max_leaves = 64;
        adaptive.target_leaves = 32;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };
        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();
        let residual_rms = (batch
            .target_update
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / batch.target_update.len() as f32)
            .sqrt();
        assert!(residual_rms < 1.0e-4, "residual RMS {residual_rms:.3e}");
    }

    #[test]
    fn disabled_proxy_does_not_materialize_zero_feature_rows() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.proxy.enabled = true;
        adaptive.proxy.context_scale = 0.0;
        adaptive.min_leaves = 8;
        adaptive.max_leaves = 64;
        adaptive.target_leaves = 32;
        adaptive.min_footprint = 0.003;
        adaptive.max_footprint = 0.2;
        let config = AdaptiveMultiscaleTrainingConfig {
            fine_particle_count: 32,
            cut_leaf_counts: vec![8, 32],
            rollout_steps: 1,
            rollouts: 1,
            temporal_samples: 2,
            rows_per_cut: 32,
            validation_rollouts: 1,
            steps: 1,
            report_interval: 1,
            controller_steps: 1,
            ..AdaptiveMultiscaleTrainingConfig::default()
        };

        let batch = adaptive_multiscale_training_batch(
            &teacher,
            &upstream_growing_2d_hashgrid(),
            &adaptive,
            &config,
        )
        .unwrap();

        assert!(batch.proxy_features.is_empty());
        batch
            .validate(
                teacher.config.perception_dims(),
                teacher.config.update_dims(),
            )
            .unwrap();
    }

    #[test]
    fn controller_event_labels_are_independent_from_runtime_hysteresis() {
        let radius = 0.01_f32;
        let count = 4;
        let particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.0; 4]; count],
            vec![0.0; count],
            2,
            1,
            count as f32 * std::f32::consts::PI * radius.powi(2),
            0.1,
        )
        .unwrap();
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = radius;
        adaptive.split_ratio = 0.78;
        adaptive.merge_ratio = 1.15;
        let targets = controller_target(
            &adaptive,
            &particles,
            &vec![0.1; count],
            &[radius * 0.85, radius * 1.17, radius, radius],
            0.1,
            0.90,
            1.20,
        );

        assert_eq!(targets[2], 1.0);
        assert_eq!(targets[3], 0.0);
        assert_eq!(targets[ADAPTIVE_CONTROLLER_OUTPUT_DIMS + 2], 0.0);
        assert_eq!(targets[ADAPTIVE_CONTROLLER_OUTPUT_DIMS + 3], 0.0);
    }

    #[test]
    fn controller_cuts_are_supervised_against_the_runtime_global_budget() {
        let radius = 0.02_f32;
        let current_leaves = 8;
        let target_leaves = 32;
        let particles = AdaptiveParticleSet::from_equal_measure(
            vec![[0.0; 4]; current_leaves],
            vec![0.0; current_leaves],
            2,
            1,
            current_leaves as f32 * std::f32::consts::PI * radius.powi(2),
            0.1,
        )
        .unwrap();
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = radius;
        adaptive.min_footprint = radius * 0.25;
        adaptive.max_footprint = radius * 4.0;
        adaptive.target_leaves = target_leaves;
        let allocation = controller_budget_allocation(
            &adaptive,
            &vec![1.0; current_leaves],
            &particles.represented_measure,
            particles.spatial_dims,
        )
        .unwrap();

        assert!((allocation.expected_leaf_count - target_leaves as f32).abs() < 0.01);
        assert!(
            allocation
                .desired_footprint
                .iter()
                .all(|desired| (*desired / radius - 0.5).abs() < 1.0e-3)
        );
    }
}
