use std::{collections::BTreeSet, time::Instant};

#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
use burn::tensor::backend::Backend;
#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
use burn_automata_kernels::AdaptiveMergeCostCubeBackend;
#[cfg(any(
    feature = "backend_cuda",
    feature = "backend_wgpu",
    all(test, feature = "gpu_wgpu")
))]
use burn_automata_kernels::HashGridConfig;

#[cfg(all(
    feature = "gpu_wgpu",
    any(test, feature = "backend_cuda", feature = "backend_wgpu")
))]
use crate::gpu::{WgpuAutomataExecutor, WgpuMaterialStateInit, WgpuNeighborMode};

use super::{
    AdaptiveRestrictionDatasetConfig, AdaptiveRestrictionDatasetReport,
    AdaptiveRestrictionLabelTarget, AdaptiveRestrictionSelectionReport,
    AdaptiveRestrictionTrainingBatch,
};
#[cfg(all(
    feature = "gpu_wgpu",
    any(feature = "backend_cuda", feature = "backend_wgpu")
))]
use crate::adaptive::task_merge_oracle::target_render_merge_costs_burn_batch_with_hierarchies;
use crate::{
    AutomataError, AutomataResult, ParticleSeed,
    adaptive::{
        ADAPTIVE_CONTROLLER_OUTPUT_DIMS, AdaptiveNpaModel, AdaptiveProxyHierarchy,
        AdaptiveRolloutConfig, advance_adaptive_rollout,
        restriction::{
            level_one_restriction_features, level_one_restriction_features_from_precomputed,
        },
        seed_adaptive_particles_scaled,
        task_merge_oracle::target_render_merge_costs,
    },
    target2d::{Target2dLossConfig, TargetImage2d},
};

#[derive(Clone)]
struct RestrictionSnapshot {
    particles: crate::adaptive::AdaptiveParticleSet,
    perception: Option<RestrictionSnapshotPerception>,
}

#[derive(Clone)]
struct RestrictionSnapshotPerception {
    normalized_features: Vec<f32>,
    base_update: Vec<f32>,
    observed_spacing: Vec<f32>,
    accepted_degree: Vec<usize>,
    feature_dims: usize,
}

struct PreparedRestrictionSnapshot {
    snapshot: RestrictionSnapshot,
    hierarchy: AdaptiveProxyHierarchy,
}

pub fn adaptive_restriction_training_batch(
    model: &AdaptiveNpaModel,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    config: &AdaptiveRestrictionDatasetConfig,
) -> AutomataResult<(
    AdaptiveRestrictionTrainingBatch,
    AdaptiveRestrictionDatasetReport,
)> {
    adaptive_restriction_training_batch_with(
        model,
        target,
        render_config,
        config,
        "cpu-reference",
        collect_restriction_snapshots_cpu,
        |snapshots,
         target_leaves,
         target,
         render_config,
         fine_measure,
         render_decoder,
         compactness,
         label_target| {
            snapshots
                .iter()
                .map(|prepared| {
                    target_render_merge_costs(
                        &prepared.snapshot.particles,
                        target_leaves,
                        target,
                        render_config,
                        fine_measure,
                        render_decoder,
                        compactness,
                        label_target,
                    )
                })
                .collect()
        },
    )
}

#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
pub fn adaptive_restriction_training_batch_burn<B: Backend + AdaptiveMergeCostCubeBackend>(
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    config: &AdaptiveRestrictionDatasetConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<(
    AdaptiveRestrictionTrainingBatch,
    AdaptiveRestrictionDatasetReport,
)> {
    #[cfg(feature = "gpu_wgpu")]
    {
        let executor = WgpuAutomataExecutor::new_restriction_blocking()?;
        adaptive_restriction_training_batch_burn_with_executor::<B>(
            &executor,
            model,
            grid,
            target,
            render_config,
            config,
            device,
            backend,
        )
    }
    #[cfg(not(feature = "gpu_wgpu"))]
    {
        let _ = (model, grid, target, render_config, config, device, backend);
        Err(AutomataError::InvalidArgument(
            "device restriction labels require gpu_wgpu for resident trajectory generation"
                .to_string(),
        ))
    }
}

#[cfg(all(
    feature = "gpu_wgpu",
    any(feature = "backend_cuda", feature = "backend_wgpu")
))]
#[allow(clippy::too_many_arguments)]
pub fn adaptive_restriction_training_batch_burn_with_executor<
    B: Backend + AdaptiveMergeCostCubeBackend,
>(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    config: &AdaptiveRestrictionDatasetConfig,
    device: &B::Device,
    backend: &str,
) -> AutomataResult<(
    AdaptiveRestrictionTrainingBatch,
    AdaptiveRestrictionDatasetReport,
)> {
    let label_backend = format!("{backend}+wgpu-resident-rollout");
    adaptive_restriction_training_batch_with(
        model,
        target,
        render_config,
        config,
        &label_backend,
        |model, config, cut_steps| {
            collect_restriction_snapshots_wgpu(executor, model, grid, config, cut_steps)
        },
        |snapshots,
         target_leaves,
         target,
         render_config,
         fine_measure,
         render_decoder,
         compactness,
         label_target| {
            let inputs = snapshots
                .iter()
                .map(|prepared| (&prepared.snapshot.particles, &prepared.hierarchy))
                .collect::<Vec<_>>();
            target_render_merge_costs_burn_batch_with_hierarchies::<B>(
                &inputs,
                target_leaves,
                target,
                render_config,
                fine_measure,
                render_decoder,
                compactness,
                label_target,
                device,
            )
        },
    )
}

fn adaptive_restriction_training_batch_with<F, G>(
    model: &AdaptiveNpaModel,
    target: &TargetImage2d,
    render_config: Target2dLossConfig,
    config: &AdaptiveRestrictionDatasetConfig,
    label_backend: &str,
    mut collect_snapshots: G,
    mut merge_costs_batch: F,
) -> AutomataResult<(
    AdaptiveRestrictionTrainingBatch,
    AdaptiveRestrictionDatasetReport,
)>
where
    F: FnMut(
        &[PreparedRestrictionSnapshot],
        usize,
        &TargetImage2d,
        Target2dLossConfig,
        f32,
        super::AdaptiveRenderDecoder,
        f32,
        AdaptiveRestrictionLabelTarget,
    ) -> AutomataResult<Vec<Vec<f32>>>,
    G: FnMut(
        &AdaptiveNpaModel,
        &AdaptiveRestrictionDatasetConfig,
        &[usize],
    ) -> AutomataResult<Vec<RestrictionSnapshot>>,
{
    validate_dataset_config(model, config)?;
    let started = Instant::now();
    let fine_count = model.config.bootstrap_fine_leaf_count();
    let fine_measure = config.total_measure / fine_count as f32;
    let cut_steps = config
        .cut_steps
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut oracle_rank_targets = Vec::new();
    let mut oracle_cost_utility_targets = Vec::new();
    let mut snapshots = 0usize;
    let mut groups_per_snapshot = None;
    let mut merges_per_snapshot = None;

    let rollout_model = restriction_rollout_model(model);
    let rollout_snapshots = collect_snapshots(&rollout_model, config, &cut_steps)?;
    let expected_snapshots = config.seeds.len() * cut_steps.len();
    if rollout_snapshots.len() != expected_snapshots {
        return Err(AutomataError::InvalidModel(format!(
            "restriction rollout produced {} snapshots, expected {expected_snapshots}",
            rollout_snapshots.len(),
        )));
    }
    let prepared = rollout_snapshots
        .into_iter()
        .map(|snapshot| {
            let hierarchy = AdaptiveProxyHierarchy::build(&snapshot.particles, 4)?;
            Ok(PreparedRestrictionSnapshot {
                snapshot,
                hierarchy,
            })
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let merge_cost_rows = merge_costs_batch(
        &prepared,
        model.config.target_leaves,
        target,
        render_config,
        fine_measure,
        config.render_decoder,
        config.render_compactness,
        config.label_target,
    )?;
    if merge_cost_rows.len() != prepared.len() {
        return Err(AutomataError::InvalidModel(format!(
            "restriction oracle returned {} snapshots, expected {}",
            merge_cost_rows.len(),
            prepared.len(),
        )));
    }
    for (prepared, merge_costs) in prepared.into_iter().zip(merge_cost_rows) {
        let RestrictionSnapshot {
            particles,
            perception,
        } = prepared.snapshot;
        let hierarchy = prepared.hierarchy;
        let merge_mask =
            hierarchy.level_one_merge_mask(&particles, model.config.target_leaves, &merge_costs)?;
        let rank_targets = signed_rank_targets(&merge_costs);
        let cost_utility_targets = robust_cost_utility_targets(&merge_costs);
        let group_features = if let Some(perception) = perception {
            level_one_restriction_features_from_precomputed(
                &rollout_model,
                &particles,
                &hierarchy,
                &perception.normalized_features,
                &perception.base_update,
                &perception.observed_spacing,
                &perception.accepted_degree,
                perception.feature_dims,
            )?
        } else {
            level_one_restriction_features(&rollout_model, &particles, &hierarchy)?
        };
        if group_features.len() != merge_mask.len() {
            return Err(AutomataError::InvalidModel(
                "restriction feature and oracle group counts differ".to_string(),
            ));
        }
        let group_count = group_features.len();
        let merge_count = merge_mask.iter().filter(|selected| **selected).count();
        if groups_per_snapshot
            .replace(group_count)
            .is_some_and(|old| old != group_count)
            || merges_per_snapshot
                .replace(merge_count)
                .is_some_and(|old| old != merge_count)
        {
            return Err(AutomataError::InvalidModel(
                "restriction snapshots have inconsistent group shapes".to_string(),
            ));
        }
        for (((feature, merge), rank_target), cost_utility_target) in group_features
            .into_iter()
            .zip(merge_mask)
            .zip(rank_targets)
            .zip(cost_utility_targets)
        {
            features.extend_from_slice(&feature);
            targets.extend_from_slice(&[0.0, 0.0, 0.0, if merge { 1.0 } else { 0.0 }]);
            oracle_rank_targets.push(rank_target);
            oracle_cost_utility_targets.push(cost_utility_target);
        }
        snapshots += 1;
    }

    let groups_per_snapshot = groups_per_snapshot.unwrap_or_default();
    let merges_per_snapshot = merges_per_snapshot.unwrap_or_default();
    let rows = snapshots * groups_per_snapshot;
    let batch = AdaptiveRestrictionTrainingBatch {
        controller: super::AdaptiveControllerTrainingBatch {
            features,
            targets,
            rows,
        },
        oracle_rank_targets,
        oracle_cost_utility_targets,
        snapshots,
        groups_per_snapshot,
        merges_per_snapshot,
    };
    batch.validate()?;
    Ok((
        batch,
        AdaptiveRestrictionDatasetReport {
            label_backend: label_backend.to_string(),
            seeds: config.seeds.len(),
            snapshots,
            rows,
            groups_per_snapshot,
            merges_per_snapshot,
            positive_fraction: merges_per_snapshot as f32 / groups_per_snapshot as f32,
            generation_ms: started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

fn signed_rank_targets(costs: &[f32]) -> Vec<f32> {
    if costs.len() <= 1 {
        return vec![0.0; costs.len()];
    }
    let mut ranked = (0..costs.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|lhs, rhs| {
        costs[*lhs]
            .total_cmp(&costs[*rhs])
            .then_with(|| lhs.cmp(rhs))
    });
    let denominator = (costs.len() - 1) as f32;
    let mut targets = vec![0.0_f32; costs.len()];
    for (rank, index) in ranked.into_iter().enumerate() {
        targets[index] = 1.0 - 2.0 * rank as f32 / denominator;
    }
    targets
}

fn robust_cost_utility_targets(costs: &[f32]) -> Vec<f32> {
    if costs.len() <= 1 {
        return vec![0.0; costs.len()];
    }
    let mut sorted = costs.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let last = sorted.len() - 1;
    let low = sorted[last / 20];
    let high = sorted[(19 * last).div_ceil(20)];
    let span = high - low;
    if !span.is_finite() || span <= f32::EPSILON * low.abs().max(high.abs()).max(1.0) {
        return signed_rank_targets(costs);
    }
    costs
        .iter()
        .map(|cost| (1.0 - 2.0 * (*cost - low) / span).clamp(-1.0, 1.0))
        .collect()
}

fn restriction_rollout_model(model: &AdaptiveNpaModel) -> AdaptiveNpaModel {
    let mut rollout_model = model.clone();
    rollout_model.config.hierarchical_restriction_step = 0;
    rollout_model
}

fn collect_restriction_snapshots_cpu(
    model: &AdaptiveNpaModel,
    config: &AdaptiveRestrictionDatasetConfig,
    cut_steps: &[usize],
) -> AutomataResult<Vec<RestrictionSnapshot>> {
    let fine_count = model.config.bootstrap_fine_leaf_count();
    let mut snapshots = Vec::with_capacity(config.seeds.len() * cut_steps.len());
    for seed in config.seeds.iter().copied() {
        let mut particles = seed_adaptive_particles_scaled(
            model,
            fine_count,
            seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
            config.total_measure,
            config.bandwidth,
        )?;
        let mut completed_steps = 0usize;
        for cut_step in cut_steps.iter().copied() {
            let trace = advance_adaptive_rollout(
                model,
                particles,
                AdaptiveRolloutConfig {
                    steps: cut_step - completed_steps,
                    dt: 1.0,
                    update_prob: config.update_prob,
                    seed,
                    bandwidth_adaptation_enabled: config.bandwidth_adaptation_enabled,
                    topology_enabled: false,
                    snapshot_interval: cut_step - completed_steps,
                },
                completed_steps,
            )?;
            particles = trace.particles;
            completed_steps = cut_step;
            snapshots.push(RestrictionSnapshot {
                particles: particles.clone(),
                perception: None,
            });
        }
    }
    Ok(snapshots)
}

#[cfg(all(
    feature = "gpu_wgpu",
    any(test, feature = "backend_cuda", feature = "backend_wgpu")
))]
fn collect_restriction_snapshots_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveRestrictionDatasetConfig,
    cut_steps: &[usize],
) -> AutomataResult<Vec<RestrictionSnapshot>> {
    let fine_count = model.config.bootstrap_fine_leaf_count();
    let seeds_per_dispatch = restriction_seeds_per_dispatch(executor, fine_count)?;
    let mut snapshots = Vec::with_capacity(config.seeds.len() * cut_steps.len());
    for seeds in config.seeds.chunks(seeds_per_dispatch) {
        let mut chunk = config.clone();
        chunk.seeds = seeds.to_vec();
        snapshots.extend(collect_restriction_snapshot_chunk_wgpu(
            executor, model, grid, &chunk, cut_steps,
        )?);
    }
    Ok(snapshots)
}

#[cfg(all(
    feature = "gpu_wgpu",
    any(test, feature = "backend_cuda", feature = "backend_wgpu")
))]
fn restriction_seeds_per_dispatch(
    executor: &WgpuAutomataExecutor,
    particles_per_seed: usize,
) -> AutomataResult<usize> {
    let max_workgroups = executor
        .device()
        .limits()
        .max_compute_workgroups_per_dimension as usize;
    restriction_seeds_per_dispatch_for_limits(
        particles_per_seed,
        max_workgroups,
        executor.max_independent_trajectory_lanes(),
    )
}

#[cfg(any(
    test,
    all(
        feature = "gpu_wgpu",
        any(feature = "backend_cuda", feature = "backend_wgpu")
    )
))]
fn restriction_seeds_per_dispatch_for_limits(
    particles_per_seed: usize,
    max_workgroups: usize,
    max_lanes: usize,
) -> AutomataResult<usize> {
    if particles_per_seed == 0 || particles_per_seed > max_workgroups {
        return Err(AutomataError::InvalidArgument(format!(
            "restriction trajectory has {particles_per_seed} particles, exceeding the WGPU dispatch limit {max_workgroups}",
        )));
    }
    Ok(max_lanes.max(1))
}

#[cfg(all(
    feature = "gpu_wgpu",
    any(test, feature = "backend_cuda", feature = "backend_wgpu")
))]
fn collect_restriction_snapshot_chunk_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveRestrictionDatasetConfig,
    cut_steps: &[usize],
) -> AutomataResult<Vec<RestrictionSnapshot>> {
    let mut gpu_model = model.clone();
    gpu_model.config.coarse_dynamics = crate::adaptive::AdaptiveCoarseDynamics::RepresentedMeasure;
    let fine_count = gpu_model.config.bootstrap_fine_leaf_count();
    let particle_sets = config
        .seeds
        .iter()
        .copied()
        .map(|seed| {
            seed_adaptive_particles_scaled(
                &gpu_model,
                fine_count,
                seed,
                ParticleSeed::UniformCircle,
                config.seed_scale,
                config.total_measure,
                config.bandwidth,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let total_particles = fine_count * particle_sets.len();
    let mut positions = Vec::with_capacity(total_particles);
    let mut states = Vec::with_capacity(total_particles * gpu_model.rule.config.state_dims);
    let mut represented_measure = Vec::with_capacity(total_particles);
    let mut covariance = Vec::with_capacity(total_particles);
    let mut state_jacobian = Vec::with_capacity(
        total_particles * gpu_model.rule.config.state_dims * gpu_model.rule.config.spatial_dims,
    );
    let mut render_scale = Vec::with_capacity(total_particles);
    let display_scale_per_footprint =
        crate::adaptive::adaptive_display_scale_per_footprint(&gpu_model);
    for particles in &particle_sets {
        positions.extend_from_slice(&particles.positions);
        states.extend_from_slice(&particles.states);
        represented_measure.extend_from_slice(&particles.represented_measure);
        covariance.extend_from_slice(&particles.covariance);
        state_jacobian.extend_from_slice(&particles.state_jacobian);
        render_scale.extend(particles.represented_measure.iter().map(|measure| {
            gpu_model
                .config
                .render_footprint(crate::adaptive::material_footprint_radius(*measure, 2))
                * display_scale_per_footprint
        }));
    }
    let gpu_rule = gpu_model.gpu_inference_rule()?;
    let mut resident = executor.create_batched_material_state(
        &gpu_rule.rule,
        &positions,
        &states,
        particle_sets.len(),
        fine_count,
        grid,
        1.0,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        config.update_prob,
        &config.seeds,
        WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &particle_sets
                .iter()
                .flat_map(|particles| particles.bandwidth.iter().copied())
                .collect::<Vec<_>>(),
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: Some(
                &particle_sets
                    .iter()
                    .flat_map(|particles| particles.closure_mode.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            closure_basis: Some(
                &particle_sets
                    .iter()
                    .flat_map(|particles| particles.closure_basis.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            closure_phase: Some(
                &particle_sets
                    .iter()
                    .flat_map(|particles| particles.closure_phase.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint,
            render_transition_steps: gpu_model.config.render_transition_steps,
        },
    )?;
    if let Some(local_hidden_start) = gpu_rule.local_hidden_start {
        let max_neighbors = match gpu_model.config.perception.graph_policy {
            burn_automata_kernels::AdaptiveGraphPolicy::RawSupport => 0,
            burn_automata_kernels::AdaptiveGraphPolicy::DirectedTopK => {
                gpu_model.config.perception.max_neighbors
            }
            burn_automata_kernels::AdaptiveGraphPolicy::MutualTopK => {
                return Err(AutomataError::InvalidArgument(
                    "batched WGPU restriction rollout does not support mutual-top-k perception"
                        .to_string(),
                ));
            }
        };
        executor.configure_state_adaptive_local_rule(
            &mut resident,
            gpu_rule.local_rule_mode,
            local_hidden_start,
            gpu_model.config.local_residual_scale,
            gpu_model.config.base_rule_footprint(),
            gpu_model.config.reference_footprint,
            gpu_model.config.perception.shepard_epsilon,
            gpu_model.config.perception.moment_regularization,
            gpu_model.config.perception.moment_condition_limit,
            max_neighbors,
            gpu_model.config.perception.pair_scale_power,
        )?;
    }
    if let Some(closure) = &gpu_model.closure_mode_rule {
        executor.configure_state_adaptive_closure_rule(&mut resident, closure)?;
    }
    if let Some(closure) = &gpu_model.closure_basis_rule {
        executor.configure_state_adaptive_closure_basis_rule(&mut resident, closure)?;
    }

    let mut per_seed_snapshots = vec![Vec::with_capacity(cut_steps.len()); particle_sets.len()];
    let mut completed_steps = 0usize;
    for cut_step in cut_steps.iter().copied() {
        executor.step_state_many(&mut resident, cut_step - completed_steps)?;
        completed_steps = cut_step;
        let (positions, states, diagnostics) = executor.capture_adaptive_diagnostics(
            &mut resident,
            gpu_model.config.base_rule_footprint(),
            gpu_model.config.perception,
        )?;
        for (seed_index, particles) in particle_sets.iter().enumerate() {
            let particle_start = seed_index * fine_count;
            let state_start = particle_start * particles.state_dims;
            let mut snapshot = particles.clone();
            snapshot.positions = positions[particle_start..particle_start + fine_count].to_vec();
            snapshot.states =
                states[state_start..state_start + fine_count * particles.state_dims].to_vec();
            snapshot.validate()?;
            let feature_start = particle_start * diagnostics.feature_dims;
            let feature_end = feature_start + fine_count * diagnostics.feature_dims;
            let update_start = particle_start * diagnostics.output_dims;
            let update_end = update_start + fine_count * diagnostics.output_dims;
            let particle_end = particle_start + fine_count;
            per_seed_snapshots[seed_index].push(RestrictionSnapshot {
                particles: snapshot,
                perception: Some(RestrictionSnapshotPerception {
                    normalized_features: diagnostics.normalized_features
                        [feature_start..feature_end]
                        .to_vec(),
                    base_update: diagnostics.base_update[update_start..update_end].to_vec(),
                    observed_spacing: diagnostics.observed_spacing[particle_start..particle_end]
                        .to_vec(),
                    accepted_degree: diagnostics.accepted_degree[particle_start..particle_end]
                        .to_vec(),
                    feature_dims: diagnostics.feature_dims,
                }),
            });
        }
    }
    Ok(per_seed_snapshots.into_iter().flatten().collect())
}

pub fn validate_adaptive_restriction_selection(
    controller: &crate::adaptive::AdaptiveController,
    batch: &AdaptiveRestrictionTrainingBatch,
) -> AutomataResult<AdaptiveRestrictionSelectionReport> {
    batch.validate()?;
    let raw = controller.forward_raw(&batch.controller.features)?;
    let merge_scores = raw
        .chunks_exact(ADAPTIVE_CONTROLLER_OUTPUT_DIMS)
        .map(|row| row[3])
        .collect::<Vec<_>>();
    validate_adaptive_restriction_selection_from_merge_scores(&merge_scores, batch)
}

pub(super) fn validate_adaptive_restriction_selection_from_merge_scores(
    merge_scores: &[f32],
    batch: &AdaptiveRestrictionTrainingBatch,
) -> AutomataResult<AdaptiveRestrictionSelectionReport> {
    batch.validate()?;
    if merge_scores.len() != batch.controller.rows
        || merge_scores.iter().any(|score| !score.is_finite())
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive restriction merge-score shape mismatch".to_owned(),
        ));
    }
    let mut true_positive = 0usize;
    let mut true_negative = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    let mut exact_cuts = 0usize;
    let mut normalized_cost_regret_sum = 0.0_f32;
    let mut worst_normalized_cost_regret = 0.0_f32;
    for snapshot in 0..batch.snapshots {
        let row_start = snapshot * batch.groups_per_snapshot;
        let mut ranked = (0..batch.groups_per_snapshot).collect::<Vec<_>>();
        ranked.sort_unstable_by(|lhs, rhs| {
            merge_scores[row_start + *rhs]
                .total_cmp(&merge_scores[row_start + *lhs])
                .then_with(|| lhs.cmp(rhs))
        });
        let mut selected = vec![false; batch.groups_per_snapshot];
        for group in ranked.into_iter().take(batch.merges_per_snapshot) {
            selected[group] = true;
        }
        let mut oracle_utility = 0.0_f32;
        let mut predicted_utility = 0.0_f32;
        for (group, predicted) in selected.iter().copied().enumerate() {
            let row = row_start + group;
            let expected =
                batch.controller.targets[row * ADAPTIVE_CONTROLLER_OUTPUT_DIMS + 3] >= 0.5;
            let utility = batch.oracle_cost_utility_targets[row];
            if expected {
                oracle_utility += utility;
            }
            if predicted {
                predicted_utility += utility;
            }
        }
        let normalized_cost_regret = ((oracle_utility - predicted_utility)
            / (2.0 * batch.merges_per_snapshot.max(1) as f32))
            .max(0.0);
        normalized_cost_regret_sum += normalized_cost_regret;
        worst_normalized_cost_regret = worst_normalized_cost_regret.max(normalized_cost_regret);
        let mut exact = true;
        for (group, predicted) in selected.into_iter().enumerate() {
            let expected = batch.controller.targets
                [(row_start + group) * ADAPTIVE_CONTROLLER_OUTPUT_DIMS + 3]
                >= 0.5;
            match (predicted, expected) {
                (true, true) => true_positive += 1,
                (false, false) => true_negative += 1,
                (true, false) => {
                    false_positive += 1;
                    exact = false;
                }
                (false, true) => {
                    false_negative += 1;
                    exact = false;
                }
            }
        }
        exact_cuts += usize::from(exact);
    }
    let rows = batch.controller.rows.max(1);
    Ok(AdaptiveRestrictionSelectionReport {
        snapshots: batch.snapshots,
        rows: batch.controller.rows,
        accuracy: (true_positive + true_negative) as f32 / rows as f32,
        precision: true_positive as f32 / (true_positive + false_positive).max(1) as f32,
        recall: true_positive as f32 / (true_positive + false_negative).max(1) as f32,
        intersection_over_union: true_positive as f32
            / (true_positive + false_positive + false_negative).max(1) as f32,
        exact_cut_fraction: exact_cuts as f32 / batch.snapshots as f32,
        mean_normalized_cost_regret: normalized_cost_regret_sum / batch.snapshots as f32,
        worst_normalized_cost_regret,
    })
}

fn validate_dataset_config(
    model: &AdaptiveNpaModel,
    config: &AdaptiveRestrictionDatasetConfig,
) -> AutomataResult<()> {
    model.validate()?;
    let fine = model.config.bootstrap_fine_leaf_count();
    let target = model.config.target_leaves;
    if model.config.spatial_dims != 2
        || fine == 0
        || target < fine.div_ceil(4)
        || target >= fine
        || !(fine - target).is_multiple_of(3)
        || config.seeds.is_empty()
        || config.cut_steps.is_empty()
        || config.cut_steps.contains(&0)
        || !config.update_prob.is_finite()
        || !(0.0..=1.0).contains(&config.update_prob)
        || !config.seed_scale.is_finite()
        || config.seed_scale <= 0.0
        || !config.total_measure.is_finite()
        || config.total_measure <= 0.0
        || !config.bandwidth.is_finite()
        || config.bandwidth <= 0.0
        || !config.render_decoder.supports_restriction_labels()
        || !config.render_compactness.is_finite()
        || !(0.0..=1.0).contains(&config.render_compactness)
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive restriction dataset requires a reachable 2D first-level cut, non-empty positive cut steps/seeds, valid rollout scalars, a supported isotropic/diagnostic decoder, and compactness in [0,1]"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::{ADAPTIVE_CONTROLLER_INPUT_DIMS, AdaptiveController};

    #[test]
    fn oracle_rank_targets_preserve_cost_order_and_tie_breaking() {
        let targets = signed_rank_targets(&[4.0, 1.0, 1.0, 3.0]);
        assert_eq!(targets[0], -1.0);
        assert_eq!(targets[1], 1.0);
        assert!((targets[2] - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((targets[3] + 1.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn cost_utility_targets_preserve_severity_and_clip_outliers() {
        let mut costs = (0..100).map(|value| value as f32).collect::<Vec<_>>();
        costs[99] = 10_000.0;
        let targets = robust_cost_utility_targets(&costs);
        assert!(targets.iter().all(|target| (-1.0..=1.0).contains(target)));
        assert_eq!(targets[0], 1.0);
        assert_eq!(targets[99], -1.0);
        assert!(targets[10] > targets[50]);
        assert!(targets[50] > targets[90]);
        assert_ne!(targets[10] - targets[11], targets[50] - targets[90]);

        assert_eq!(
            robust_cost_utility_targets(&[2.0, 2.0, 2.0]),
            signed_rank_targets(&[2.0, 2.0, 2.0])
        );
    }

    #[test]
    fn restriction_seed_batch_respects_wgpu_dispatch_limit() {
        assert_eq!(
            restriction_seeds_per_dispatch_for_limits(4_096, 65_535, 32).unwrap(),
            32,
        );
        assert!(restriction_seeds_per_dispatch_for_limits(65_536, 65_535, 32).is_err());
    }

    #[test]
    fn selection_validation_uses_exact_budget_top_k() {
        let mut controller = AdaptiveController::seeded(4, 7);
        controller.weights.input_weights.fill(0.0);
        controller
            .weights
            .input_bias
            .copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        controller.weights.output_weights.fill(0.0);
        controller.weights.output_weights[3 * 4] = 1.0;
        let mut features = vec![0.0; 4 * ADAPTIVE_CONTROLLER_INPUT_DIMS];
        for (row, value) in [4.0, 1.0, 3.0, 2.0].into_iter().enumerate() {
            features[row * ADAPTIVE_CONTROLLER_INPUT_DIMS] = value;
        }
        controller.weights.input_weights[0] = 1.0;
        let batch = AdaptiveRestrictionTrainingBatch {
            controller: super::super::AdaptiveControllerTrainingBatch {
                features,
                targets: vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                ],
                rows: 4,
            },
            oracle_rank_targets: vec![1.0, -1.0, 0.5, -0.5],
            oracle_cost_utility_targets: vec![1.0, -1.0, 0.5, -0.5],
            snapshots: 1,
            groups_per_snapshot: 4,
            merges_per_snapshot: 2,
        };
        let report = validate_adaptive_restriction_selection(&controller, &batch).unwrap();
        assert_eq!(report.accuracy, 1.0);
        assert_eq!(report.intersection_over_union, 1.0);
        assert_eq!(report.exact_cut_fraction, 1.0);
        assert_eq!(report.mean_normalized_cost_regret, 0.0);
        assert_eq!(report.worst_normalized_cost_regret, 0.0);

        let wrong = validate_adaptive_restriction_selection_from_merge_scores(
            &[1.0, 4.0, 2.0, 3.0],
            &batch,
        )
        .unwrap();
        assert!((wrong.mean_normalized_cost_regret - 0.75).abs() <= 1.0e-6);
        assert_eq!(
            wrong.worst_normalized_cost_regret,
            wrong.mean_normalized_cost_regret
        );
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    #[ignore = "requires a WGPU device"]
    fn logical_lane_batch_matches_independent_rollouts() {
        use crate::{NpaConfig, NpaModel};

        let count = 64;
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let footprint = crate::adaptive::material_footprint_radius(total_measure / count as f32, 2);
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = footprint;
        adaptive.base_rule_footprint = footprint;
        adaptive.min_footprint = 0.5 * footprint;
        adaptive.max_footprint = 2.0 * footprint;
        adaptive.min_leaves = 16;
        adaptive.target_leaves = 58;
        adaptive.max_leaves = count;
        adaptive.initial_leaves = count;
        adaptive.bootstrap_fine_leaves = count;
        let model =
            AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 7), adaptive, 11)
                .unwrap();
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let grid = HashGridConfig::growing_2d();
        let config = AdaptiveRestrictionDatasetConfig {
            seeds: vec![41, 42],
            cut_steps: vec![3],
            update_prob: 1.0,
            total_measure,
            ..AdaptiveRestrictionDatasetConfig::default()
        };
        let batched = collect_restriction_snapshots_wgpu(
            &executor,
            &model,
            &grid,
            &config,
            &config.cut_steps,
        )
        .unwrap();
        let mut independent = Vec::new();
        for seed in config.seeds.iter().copied() {
            let single = AdaptiveRestrictionDatasetConfig {
                seeds: vec![seed],
                ..config.clone()
            };
            independent.extend(
                collect_restriction_snapshots_wgpu(
                    &executor,
                    &model,
                    &grid,
                    &single,
                    &single.cut_steps,
                )
                .unwrap(),
            );
        }
        let max_position_error = batched
            .iter()
            .zip(&independent)
            .flat_map(|(batched, independent)| {
                batched
                    .particles
                    .positions
                    .iter()
                    .zip(&independent.particles.positions)
                    .flat_map(|(batched, independent)| {
                        (0..2).map(|axis| (batched[axis] - independent[axis]).abs())
                    })
            })
            .fold(0.0_f32, f32::max);
        let max_state_error = batched
            .iter()
            .zip(&independent)
            .flat_map(|(batched, independent)| {
                batched
                    .particles
                    .states
                    .iter()
                    .zip(&independent.particles.states)
                    .map(|(batched, independent)| (batched - independent).abs())
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_position_error < 5.0e-5,
            "logical-lane position error {max_position_error}"
        );
        assert!(
            max_state_error < 5.0e-5,
            "logical-lane state error {max_state_error}"
        );
        let max_perception_error = batched
            .iter()
            .zip(&independent)
            .map(|(batched, independent)| {
                let batched = batched.perception.as_ref().unwrap();
                let independent = independent.perception.as_ref().unwrap();
                batched
                    .normalized_features
                    .iter()
                    .zip(&independent.normalized_features)
                    .chain(batched.base_update.iter().zip(&independent.base_update))
                    .chain(
                        batched
                            .observed_spacing
                            .iter()
                            .zip(&independent.observed_spacing),
                    )
                    .map(|(batched, independent)| (batched - independent).abs())
                    .fold(0.0_f32, f32::max)
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_perception_error < 5.0e-5,
            "logical-lane perception error {max_perception_error}"
        );
        for (batched, independent) in batched.iter().zip(&independent) {
            assert_eq!(
                batched.perception.as_ref().unwrap().accepted_degree,
                independent.perception.as_ref().unwrap().accepted_degree,
            );
        }
    }
}
