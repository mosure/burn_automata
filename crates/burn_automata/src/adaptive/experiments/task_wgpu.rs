use std::time::Instant;

use burn_automata_kernels::HashGridConfig;

use super::AdaptiveTaskRestrictionPolicy;
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        AdaptiveCoarseDynamics, AdaptiveHierarchyRestrictionPolicy, AdaptiveNpaModel,
        AdaptiveParticleSet, AdaptiveProxyHierarchy, AdaptiveRolloutConfig, AdaptiveRolloutTrace,
        AdaptiveSnapshot, AdaptiveStepMetrics, AdaptiveTopologyControl,
        adaptive_display_scale_per_footprint, material_footprint_radius,
        restriction::level_one_restriction_features_from_precomputed,
        rollout::apply_adaptive_topology_at_step_with_control, scale::material_scale_metrics,
        seed::restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy,
        seed_adaptive_particles_scaled,
    },
    gpu::{WgpuAutomataExecutor, WgpuAutomataState, WgpuNeighborMode},
    rollout::{RolloutConfig, RolloutTrace, seed_particles_scaled},
};

#[derive(Clone)]
pub(super) struct BatchedAdaptiveTaskTrace {
    pub fixed: AdaptiveRolloutTrace,
    pub topology: Option<AdaptiveRolloutTrace>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_fixed_rollouts_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &NpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_count: usize,
    steps: usize,
    update_prob: f32,
    seed_scale: f32,
) -> AutomataResult<Vec<RolloutTrace>> {
    if model.config.position_features {
        return Err(AutomataError::InvalidArgument(
            "spatially batched task evaluation does not support absolute position features"
                .to_owned(),
        ));
    }
    let seeds_per_dispatch = seeds_per_wgpu_dispatch(executor, particle_count)?;
    let mut traces = Vec::with_capacity(seeds.len());
    for chunk in seeds.chunks(seeds_per_dispatch) {
        traces.extend(run_fixed_rollout_chunk_wgpu(
            executor,
            model,
            grid,
            chunk,
            particle_count,
            steps,
            update_prob,
            seed_scale,
        )?);
    }
    Ok(traces)
}

#[allow(clippy::too_many_arguments)]
fn run_fixed_rollout_chunk_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &NpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_count: usize,
    steps: usize,
    update_prob: f32,
    seed_scale: f32,
) -> AutomataResult<Vec<RolloutTrace>> {
    let mut positions = Vec::with_capacity(seeds.len() * particle_count);
    let mut states = Vec::with_capacity(seeds.len() * particle_count * model.config.state_dims);
    for seed in seeds.iter().copied() {
        let (lane_positions, lane_states) = seed_particles_scaled(
            1,
            particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            seed,
            ParticleSeed::UniformCircle,
            seed_scale,
        );
        positions.extend(lane_positions);
        states.extend(lane_states);
    }
    let total = positions.len();
    let represented_measure = vec![1.0; total];
    let covariance = vec![[0.0; 9]; total];
    let render_scale = vec![1.0; total];
    let mut resident = executor.create_batched_material_state(
        model,
        &positions,
        &states,
        seeds.len(),
        particle_count,
        grid,
        1.0,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        update_prob,
        seeds,
        crate::gpu::WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &vec![grid.eps; total],
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &vec![0.0; total * model.config.state_dims * model.config.spatial_dims],
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    executor.step_state_many(&mut resident, steps)?;
    let (positions, states) = executor.read_positions_states(&resident)?;
    let state_len = particle_count * model.config.state_dims;
    let mut traces = Vec::with_capacity(seeds.len());
    for lane in 0..seeds.len() {
        let particle_start = lane * particle_count;
        traces.push(RolloutTrace {
            positions: positions[particle_start..particle_start + particle_count].to_vec(),
            states: states[lane * state_len..(lane + 1) * state_len].to_vec(),
            batch_size: 1,
            particle_count,
            state_dims: model.config.state_dims,
            steps,
            mean_dx: Vec::new(),
        });
    }
    Ok(traces)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_task_quality_rollouts_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_count: usize,
    rollout: AdaptiveRolloutConfig,
    topology_control: AdaptiveTopologyControl,
    restriction_policy: AdaptiveTaskRestrictionPolicy,
    seed_scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<Vec<BatchedAdaptiveTaskTrace>> {
    if rollout.bandwidth_adaptation_enabled {
        return Err(AutomataError::InvalidArgument(
            "batched WGPU task rollout does not support bandwidth adaptation".to_owned(),
        ));
    }
    if model.rule.config.position_features {
        return Err(AutomataError::InvalidArgument(
            "spatially batched task evaluation does not support absolute position features"
                .to_owned(),
        ));
    }
    let topology_steps = rollout.topology_enabled.then(|| {
        (1..=rollout.steps)
            .filter(|step| {
                let hierarchical_restriction = model.config.hierarchical_restriction_step == *step
                    && particle_count == model.config.bootstrap_fine_leaf_count()
                    && particle_count > model.config.target_leaves;
                hierarchical_restriction || model.config.is_topology_step(*step, particle_count)
            })
            .collect::<Vec<_>>()
    });
    let persistent_post_cut = rollout.topology_enabled
        && model.config.coarse_dynamics == AdaptiveCoarseDynamics::PersistentFineQuadrature
        && model.config.hierarchical_restriction_step > 0
        && model.config.hierarchical_restriction_step < rollout.steps;
    let requires_individual_topology = persistent_post_cut
        || topology_steps
            .as_deref()
            .is_some_and(|steps| steps != [rollout.steps]);
    if requires_individual_topology {
        return seeds
            .iter()
            .copied()
            .map(|seed| {
                let fixed = run_task_quality_rollout_wgpu(
                    executor,
                    model,
                    grid,
                    particle_count,
                    AdaptiveRolloutConfig {
                        seed,
                        topology_enabled: false,
                        ..rollout
                    },
                    topology_control,
                    restriction_policy,
                    seed_scale,
                    total_measure,
                    bandwidth,
                )?;
                let topology = run_task_quality_rollout_wgpu(
                    executor,
                    model,
                    grid,
                    particle_count,
                    AdaptiveRolloutConfig { seed, ..rollout },
                    topology_control,
                    restriction_policy,
                    seed_scale,
                    total_measure,
                    bandwidth,
                )?;
                Ok(BatchedAdaptiveTaskTrace {
                    fixed,
                    topology: Some(topology),
                })
            })
            .collect();
    }
    let seeds_per_dispatch = seeds_per_wgpu_dispatch(executor, particle_count)?;
    let mut traces = Vec::with_capacity(seeds.len());
    for chunk in seeds.chunks(seeds_per_dispatch) {
        traces.extend(run_task_quality_rollout_chunk_wgpu(
            executor,
            model,
            grid,
            chunk,
            particle_count,
            rollout,
            topology_control,
            restriction_policy,
            seed_scale,
            total_measure,
            bandwidth,
        )?);
    }
    Ok(traces)
}

#[allow(clippy::too_many_arguments)]
fn run_task_quality_rollout_chunk_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_count: usize,
    rollout: AdaptiveRolloutConfig,
    topology_control: AdaptiveTopologyControl,
    restriction_policy: AdaptiveTaskRestrictionPolicy,
    seed_scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<Vec<BatchedAdaptiveTaskTrace>> {
    if restriction_policy == AdaptiveTaskRestrictionPolicy::TargetRenderOracle {
        return Err(AutomataError::InvalidArgument(
            "target-render-oracle restriction is a bounded CPU reference".to_owned(),
        ));
    }
    let mut gpu_model = model.clone();
    gpu_model.config.coarse_dynamics = AdaptiveCoarseDynamics::RepresentedMeasure;
    gpu_model.config.hierarchical_restriction_policy = match restriction_policy {
        AdaptiveTaskRestrictionPolicy::DynamicsDetail => {
            AdaptiveHierarchyRestrictionPolicy::DynamicsDetail
        }
        AdaptiveTaskRestrictionPolicy::LearnedController => {
            AdaptiveHierarchyRestrictionPolicy::LearnedController
        }
        AdaptiveTaskRestrictionPolicy::TargetRenderOracle => unreachable!(),
    };
    gpu_model.validate()?;

    let particle_sets = seed_task_particle_sets_wgpu(
        executor,
        &gpu_model,
        grid,
        seeds,
        particle_count,
        restriction_policy,
        seed_scale,
        total_measure,
        bandwidth,
    )?;
    let mut resident = create_task_resident_state(
        executor,
        &gpu_model,
        grid,
        seeds,
        &particle_sets,
        particle_count,
        rollout.dt,
        rollout.update_prob,
    )?;
    let started = Instant::now();
    executor.step_state_many(&mut resident, rollout.steps)?;
    let capture_restriction = rollout.topology_enabled
        && restriction_policy == AdaptiveTaskRestrictionPolicy::LearnedController;
    let (positions, states, diagnostics) = if capture_restriction {
        let (positions, states, diagnostics) = executor.capture_adaptive_diagnostics(
            &mut resident,
            gpu_model.config.base_rule_footprint(),
            gpu_model.config.perception,
        )?;
        (positions, states, Some(diagnostics))
    } else {
        let (positions, states) = executor.read_positions_states(&resident)?;
        (positions, states, None)
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let state_len = particle_count * gpu_model.rule.config.state_dims;
    let mut out = Vec::with_capacity(seeds.len());
    for lane in 0..seeds.len() {
        let particle_start = lane * particle_count;
        let mut particles = particle_sets[lane].clone();
        particles.positions = positions[particle_start..particle_start + particle_count].to_vec();
        particles.states = states[lane * state_len..(lane + 1) * state_len].to_vec();
        particles.validate()?;
        let initial = particle_sets[lane].clone();
        let fixed = adaptive_trace(
            initial.clone(),
            particles.clone(),
            rollout.steps,
            None,
            elapsed_ms,
            gpu_model.config.reference_footprint,
        );
        let topology = if rollout.topology_enabled {
            let update = if let Some(diagnostics) = diagnostics.as_ref() {
                apply_precomputed_learned_restriction(
                    &gpu_model,
                    &mut particles,
                    diagnostics,
                    lane,
                    particle_count,
                    rollout.steps,
                )?
            } else {
                apply_adaptive_topology_at_step_with_control(
                    &gpu_model,
                    &mut particles,
                    rollout.steps,
                    rollout.steps,
                    topology_control,
                )?
            };
            Some(adaptive_trace(
                initial,
                particles,
                rollout.steps,
                Some((update.split_events, update.merge_events, update.elapsed_ms)),
                elapsed_ms,
                gpu_model.config.reference_footprint,
            ))
        } else {
            None
        };
        out.push(BatchedAdaptiveTaskTrace { fixed, topology });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn seed_task_particle_sets_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_count: usize,
    restriction_policy: AdaptiveTaskRestrictionPolicy,
    seed_scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<Vec<AdaptiveParticleSet>> {
    let fine_count = model.config.bootstrap_fine_leaf_count();
    let device_target_cut = restriction_policy == AdaptiveTaskRestrictionPolicy::LearnedController
        && model.config.hierarchical_bootstrap_seed
        && particle_count == model.config.target_leaves
        && fine_count > particle_count;
    if !device_target_cut {
        return seeds
            .iter()
            .copied()
            .map(|seed| {
                seed_adaptive_particles_scaled(
                    model,
                    particle_count,
                    seed,
                    ParticleSeed::UniformCircle,
                    seed_scale,
                    total_measure,
                    bandwidth,
                )
            })
            .collect();
    }

    let mut fine_sets = seeds
        .iter()
        .copied()
        .map(|seed| {
            seed_adaptive_particles_scaled(
                model,
                fine_count,
                seed,
                ParticleSeed::UniformCircle,
                seed_scale,
                total_measure,
                bandwidth,
            )
        })
        .collect::<AutomataResult<Vec<_>>>()?;
    let mut resident = create_task_resident_state(
        executor, model, grid, seeds, &fine_sets, fine_count, 1.0, 1.0,
    )?;
    let diagnostics = executor.capture_adaptive_diagnostics_only(
        &mut resident,
        model.config.base_rule_footprint(),
        model.config.perception,
    )?;
    for (lane, particles) in fine_sets.iter_mut().enumerate() {
        apply_precomputed_learned_restriction(model, particles, &diagnostics, lane, fine_count, 0)?;
    }
    Ok(fine_sets)
}

#[allow(clippy::too_many_arguments)]
fn create_task_resident_state(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    seeds: &[u64],
    particle_sets: &[AdaptiveParticleSet],
    particle_count: usize,
    dt: f32,
    update_prob: f32,
) -> AutomataResult<WgpuAutomataState> {
    let total = seeds.len() * particle_count;
    let mut positions = Vec::with_capacity(total);
    let mut states = Vec::with_capacity(total * model.rule.config.state_dims);
    let mut represented_measure = Vec::with_capacity(total);
    let mut particle_ids = Vec::with_capacity(total);
    let mut update_masks = Vec::with_capacity(total);
    let mut covariance = Vec::with_capacity(total);
    let mut state_jacobian =
        Vec::with_capacity(total * model.rule.config.state_dims * model.rule.config.spatial_dims);
    let mut render_scale = Vec::with_capacity(total);
    let display_scale_per_footprint = adaptive_display_scale_per_footprint(model);
    for particles in particle_sets {
        positions.extend_from_slice(&particles.positions);
        states.extend_from_slice(&particles.states);
        represented_measure.extend_from_slice(&particles.represented_measure);
        particle_ids.extend_from_slice(&particles.particle_id);
        update_masks.extend(super::super::gpu::material_update_masks(particles)?);
        covariance.extend_from_slice(&particles.covariance);
        state_jacobian.extend_from_slice(&particles.state_jacobian);
        render_scale.extend(particles.represented_measure.iter().map(|measure| {
            model
                .config
                .render_footprint(material_footprint_radius(*measure, 2))
                * display_scale_per_footprint
        }));
    }
    let gpu_rule = model.gpu_inference_rule()?;
    let mut resident = executor.create_batched_material_state(
        &gpu_rule.rule,
        &positions,
        &states,
        seeds.len(),
        particle_count,
        grid,
        dt,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        update_prob,
        seeds,
        crate::gpu::WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: Some(&particle_ids),
            update_masks: Some(&update_masks),
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
            render_transition_steps: model.config.render_transition_steps,
        },
    )?;
    if let Some(local_hidden_start) = gpu_rule.local_hidden_start {
        let max_neighbors = match model.config.perception.graph_policy {
            burn_automata_kernels::AdaptiveGraphPolicy::RawSupport => 0,
            burn_automata_kernels::AdaptiveGraphPolicy::DirectedTopK => {
                model.config.perception.max_neighbors
            }
            burn_automata_kernels::AdaptiveGraphPolicy::MutualTopK => {
                return Err(AutomataError::InvalidArgument(
                    "batched WGPU task rollout does not support mutual-top-k perception".to_owned(),
                ));
            }
        };
        executor.configure_state_adaptive_local_rule(
            &mut resident,
            gpu_rule.local_rule_mode,
            local_hidden_start,
            model.config.local_residual_scale,
            model.config.base_rule_footprint(),
            model.config.reference_footprint,
            model.config.perception.shepard_epsilon,
            model.config.perception.moment_regularization,
            model.config.perception.moment_condition_limit,
            max_neighbors,
            model.config.perception.pair_scale_power,
        )?;
    }
    if let Some(closure) = &model.closure_mode_rule {
        executor.configure_state_adaptive_closure_rule(&mut resident, closure)?;
    }
    if let Some(closure) = &model.closure_basis_rule {
        executor.configure_state_adaptive_closure_basis_rule(&mut resident, closure)?;
    }
    executor.configure_state_adaptive_integration(
        &mut resident,
        model.config.base_rule_footprint(),
        model.config.expected_coarse_update_mask,
    )?;
    Ok(resident)
}

fn apply_precomputed_learned_restriction(
    model: &AdaptiveNpaModel,
    particles: &mut AdaptiveParticleSet,
    diagnostics: &crate::gpu::WgpuAdaptiveDiagnostics,
    lane: usize,
    particle_count: usize,
    step: usize,
) -> AutomataResult<crate::adaptive::AdaptiveTopologyUpdate> {
    let started = Instant::now();
    let initial_leaf_count = particles.len();
    let feature_start = lane * particle_count * diagnostics.feature_dims;
    let feature_end = feature_start + particle_count * diagnostics.feature_dims;
    let update_start = lane * particle_count * diagnostics.output_dims;
    let update_end = update_start + particle_count * diagnostics.output_dims;
    let row_start = lane * particle_count;
    let row_end = row_start + particle_count;
    let hierarchy = AdaptiveProxyHierarchy::build(particles, 2 * particles.spatial_dims)?;
    let features = level_one_restriction_features_from_precomputed(
        model,
        particles,
        &hierarchy,
        &diagnostics.normalized_features[feature_start..feature_end],
        &diagnostics.base_update[update_start..update_end],
        &diagnostics.observed_spacing[row_start..row_end],
        &diagnostics.accepted_degree[row_start..row_end],
        diagnostics.feature_dims,
    )?;
    let controller = model.restriction_controller.as_ref().ok_or_else(|| {
        AutomataError::InvalidModel(
            "learned hierarchy restriction requires restriction_controller".to_owned(),
        )
    })?;
    let merge_costs = controller
        .forward(&features)
        .into_iter()
        .map(|output| -output.merge_probability)
        .collect::<Vec<_>>();
    *particles = restrict_adaptive_particles_to_target_by_merge_cost_with_hierarchy(
        model,
        particles,
        &hierarchy,
        &merge_costs,
    )?;
    particles.validate()?;
    let merge_events = initial_leaf_count
        .saturating_sub(particles.len())
        .div_ceil(2 * particles.spatial_dims - 1);
    Ok(crate::adaptive::AdaptiveTopologyUpdate {
        step,
        initial_leaf_count,
        final_leaf_count: particles.len(),
        split_events: 0,
        merge_events,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn adaptive_trace(
    initial: AdaptiveParticleSet,
    particles: AdaptiveParticleSet,
    steps: usize,
    topology: Option<(usize, usize, f64)>,
    elapsed_ms: f64,
    reference_footprint: f32,
) -> AdaptiveRolloutTrace {
    let (split_events, merge_events, topology_ms) = topology.unwrap_or((0, 0, 0.0));
    let mut metrics = step_metrics(
        &particles,
        steps,
        split_events,
        merge_events,
        topology_ms,
        reference_footprint,
    );
    metrics.total_ms = elapsed_ms;
    AdaptiveRolloutTrace {
        metrics: vec![metrics],
        snapshots: vec![
            AdaptiveSnapshot {
                step: 0,
                particles: initial,
            },
            AdaptiveSnapshot {
                step: steps,
                particles: particles.clone(),
            },
        ],
        particles,
        steps,
    }
}

fn seeds_per_wgpu_dispatch(
    executor: &WgpuAutomataExecutor,
    particle_count: usize,
) -> AutomataResult<usize> {
    let max_workgroups = executor
        .device()
        .limits()
        .max_compute_workgroups_per_dimension as usize;
    if particle_count > max_workgroups {
        return Err(AutomataError::InvalidArgument(format!(
            "task trajectory has {particle_count} particles, exceeding the WGPU dispatch limit {max_workgroups}",
        )));
    }
    Ok(executor.max_independent_trajectory_lanes())
}

pub(super) fn run_fixed_rollout_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &NpaModel,
    grid: &HashGridConfig,
    config: &RolloutConfig,
) -> AutomataResult<RolloutTrace> {
    if config.batch_size != 1 {
        return Err(AutomataError::InvalidArgument(
            "adaptive task-quality WGPU controls require batch_size=1".to_string(),
        ));
    }
    let (positions, states) = seed_particles_scaled(
        1,
        config.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        config.seed,
        ParticleSeed::UniformCircle,
        config.seed_scale,
    );
    let mut resident = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        1,
        config.particle_count,
        grid,
        config.dt,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        config.update_prob,
        config.seed,
    )?;
    executor.step_state_many(&mut resident, config.steps)?;
    let (positions, states) = executor.read_positions_states(&resident)?;
    Ok(RolloutTrace {
        positions,
        states,
        batch_size: 1,
        particle_count: config.particle_count,
        state_dims: model.config.state_dims,
        steps: config.steps,
        mean_dx: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_task_quality_rollout_wgpu(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    particle_count: usize,
    rollout: AdaptiveRolloutConfig,
    topology_control: AdaptiveTopologyControl,
    restriction_policy: AdaptiveTaskRestrictionPolicy,
    seed_scale: f32,
    total_measure: f32,
    bandwidth: f32,
) -> AutomataResult<AdaptiveRolloutTrace> {
    if rollout.bandwidth_adaptation_enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive task-quality WGPU rollout does not support bandwidth adaptation".to_string(),
        ));
    }
    if restriction_policy == AdaptiveTaskRestrictionPolicy::TargetRenderOracle {
        return Err(AutomataError::InvalidArgument(
            "target-render-oracle restriction is a bounded CPU reference; use learned-controller or dynamics-detail for WGPU evaluation"
                .to_string(),
        ));
    }
    // The clone prevents the evaluation-only restriction policy from changing
    // the serialized experiment model. Coarse dynamics are never rewritten:
    // active and persistent quadrature now both have resident WGPU execution.
    let mut gpu_model = model.clone();
    gpu_model.config.hierarchical_restriction_policy = match restriction_policy {
        AdaptiveTaskRestrictionPolicy::DynamicsDetail => {
            AdaptiveHierarchyRestrictionPolicy::DynamicsDetail
        }
        AdaptiveTaskRestrictionPolicy::LearnedController => {
            AdaptiveHierarchyRestrictionPolicy::LearnedController
        }
        AdaptiveTaskRestrictionPolicy::TargetRenderOracle => unreachable!(),
    };
    gpu_model.validate()?;
    let particles = seed_adaptive_particles_scaled(
        &gpu_model,
        particle_count,
        rollout.seed,
        ParticleSeed::UniformCircle,
        seed_scale,
        total_measure,
        bandwidth,
    )?;
    let initial = particles.clone();
    let started = Instant::now();
    let mut state = executor.create_adaptive_state(
        &gpu_model,
        particles,
        grid,
        rollout.dt,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        rollout.update_prob,
        rollout.seed,
    )?;
    let report = executor.step_adaptive_state_many_with_topology_control(
        &mut state,
        rollout.steps,
        rollout.topology_enabled,
        topology_control,
    )?;
    executor.synchronize_adaptive_particles(&mut state)?;

    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let mut metrics = Vec::with_capacity(report.topology_updates.len().max(1));
    for update in report.topology_updates {
        metrics.push(step_metrics(
            &state.particles,
            update.step,
            update.split_events,
            update.merge_events,
            update.elapsed_ms,
            gpu_model.config.reference_footprint,
        ));
    }
    if metrics.is_empty() {
        let mut final_metrics = step_metrics(
            &state.particles,
            rollout.steps,
            0,
            0,
            0.0,
            gpu_model.config.reference_footprint,
        );
        final_metrics.total_ms = elapsed_ms;
        metrics.push(final_metrics);
    } else if let Some(last) = metrics.last_mut() {
        last.total_ms = elapsed_ms;
    }
    Ok(AdaptiveRolloutTrace {
        particles: state.particles.clone(),
        steps: rollout.steps,
        metrics,
        snapshots: vec![
            AdaptiveSnapshot {
                step: 0,
                particles: initial,
            },
            AdaptiveSnapshot {
                step: rollout.steps,
                particles: state.particles,
            },
        ],
    })
}

fn step_metrics(
    particles: &AdaptiveParticleSet,
    step: usize,
    split_events: usize,
    merge_events: usize,
    elapsed_ms: f64,
    reference_footprint: f32,
) -> AdaptiveStepMetrics {
    let footprints = particles
        .represented_measure
        .iter()
        .map(|measure| material_footprint_radius(*measure, particles.spatial_dims))
        .collect::<Vec<_>>();
    let mean_footprint = footprints.iter().sum::<f32>() / footprints.len().max(1) as f32;
    let variance = footprints
        .iter()
        .map(|value| (*value - mean_footprint).powi(2))
        .sum::<f32>()
        / footprints.len().max(1) as f32;
    let scale_metrics = material_scale_metrics(particles, reference_footprint);
    AdaptiveStepMetrics {
        step,
        leaf_count: particles.len(),
        total_measure: particles.total_measure(),
        mean_footprint,
        min_footprint: footprints.iter().copied().fold(f32::INFINITY, f32::min),
        max_footprint: footprints.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        footprint_coefficient_of_variation: variance.sqrt() / mean_footprint.max(f32::MIN_POSITIVE),
        occupied_material_scale_bins: scale_metrics.occupied_sixty_fourth_octave_bins,
        fractional_material_scale_fraction: scale_metrics.fractional_octave_fraction,
        dyadic_scale_quantization_rmse_octaves: scale_metrics.dyadic_quantization_rmse_octaves,
        mean_bandwidth: particles.bandwidth.iter().sum::<f32>()
            / particles.bandwidth.len().max(1) as f32,
        split_events,
        merge_events,
        topology_ms: elapsed_ms,
        total_ms: elapsed_ms,
        ..AdaptiveStepMetrics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AutomataPreset, NpaConfig, NpaWeights,
        gpu::{WGPU_MATERIAL_UPDATE_MASK_MEMBERS, WgpuMaterialUpdateMask},
    };

    fn gpu_hash_u32(value: u32) -> u32 {
        let mut value = (value ^ 61) ^ (value >> 16);
        value = value.wrapping_add(value << 3);
        value ^= value >> 4;
        value = value.wrapping_mul(0x27d4_eb2d);
        value ^ (value >> 15)
    }

    fn gpu_update_draw(particle: u32, step: u32, seed: u32, probability: f32) -> bool {
        let mixed = gpu_hash_u32(particle ^ gpu_hash_u32(step.wrapping_add(0x9e37_79b9)) ^ seed);
        (mixed >> 8) as f32 * (1.0 / 16_777_216.0) < probability
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn logical_seed_batch_matches_independent_wgpu_rollouts() {
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let model = NpaModel::seeded(config, 42);
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let seeds = [53, 71];
        let particle_count = 128;
        let steps = 3;
        let update_prob = 0.5;
        let batched = run_fixed_rollouts_wgpu(
            &executor,
            &model,
            &grid,
            &seeds,
            particle_count,
            steps,
            update_prob,
            0.2,
        )
        .unwrap();

        for (seed, batched) in seeds.into_iter().zip(batched) {
            let independent = run_fixed_rollout_wgpu(
                &executor,
                &model,
                &grid,
                &RolloutConfig {
                    batch_size: 1,
                    particle_count,
                    steps,
                    dt: 1.0,
                    update_prob,
                    seed,
                    seed_scale: 0.2,
                },
            )
            .unwrap();
            let max_position = batched
                .positions
                .iter()
                .zip(&independent.positions)
                .flat_map(|(lhs, rhs)| (0..2).map(move |axis| (lhs[axis] - rhs[axis]).abs()))
                .fold(0.0_f32, f32::max);
            let max_state = batched
                .states
                .iter()
                .zip(&independent.states)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_position <= 5.0e-5 && max_state <= 5.0e-5,
                "seed={seed} batch mismatch: position={max_position} state={max_state}",
            );
        }
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn fused_sorted_grid_matches_legacy_scan_rollout() {
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let model = NpaModel::seeded(config, 42);
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let seeds = [53, 71];
        let particle_count = 512;
        let mut positions = Vec::new();
        let mut states = Vec::new();
        for seed in seeds {
            let (lane_positions, lane_states) = seed_particles_scaled(
                1,
                particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                seed,
                ParticleSeed::UniformCircle,
                0.2,
            );
            positions.extend(lane_positions);
            states.extend(lane_states);
        }
        let material_count = positions.len();
        let represented_measure = vec![1.0; material_count];
        let covariance = vec![[0.0; 9]; material_count];
        let render_scale = vec![1.0; material_count];
        let create = || {
            executor
                .create_batched_material_state(
                    &model,
                    &positions,
                    &states,
                    seeds.len(),
                    particle_count,
                    &grid,
                    1.0,
                    WgpuNeighborMode::SubgroupCooperativeSortedCells,
                    0.5,
                    &seeds,
                    crate::gpu::WgpuMaterialStateInit {
                        represented_measure: &represented_measure,
                        particle_ids: None,
                        update_masks: None,
                        bandwidth: &vec![grid.eps; material_count],
                        support_bins: None,
                        covariance: &covariance,
                        state_jacobian: &vec![
                            0.0;
                            material_count
                                * model.config.state_dims
                                * model.config.spatial_dims
                        ],
                        closure_mode: None,
                        closure_basis: None,
                        closure_phase: None,
                        render_from_scale: &render_scale,
                        render_target_footprint: &render_scale,
                        display_scale_per_footprint: 1.0,
                        render_transition_steps: 0,
                    },
                )
                .unwrap()
        };
        let mut fused = create();
        let mut legacy = create();
        executor.set_fused_sorted_grid_enabled_for_test(&mut legacy, false);
        executor.step_state_many(&mut fused, 3).unwrap();
        executor.step_state_many(&mut legacy, 3).unwrap();
        let (fused_positions, fused_states) = executor.read_positions_states(&fused).unwrap();
        let (legacy_positions, legacy_states) = executor.read_positions_states(&legacy).unwrap();
        let max_position = fused_positions
            .iter()
            .zip(&legacy_positions)
            .flat_map(|(lhs, rhs)| (0..2).map(move |axis| (lhs[axis] - rhs[axis]).abs()))
            .fold(0.0_f32, f32::max);
        let max_state = fused_states
            .iter()
            .zip(&legacy_states)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_position <= 5.0e-5 && max_state <= 5.0e-5,
            "fused sorted-grid mismatch: position={max_position} state={max_state}",
        );
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn adaptive_diagnostics_match_cpu_restriction_inputs() {
        let (rule_config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let rule = NpaModel::seeded(rule_config, 42);
        let particle_count = 128;
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let footprint = material_footprint_radius(total_measure / particle_count as f32, 2);
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.reference_footprint = footprint;
        adaptive.base_rule_footprint = footprint;
        adaptive.min_footprint = 0.5 * footprint;
        adaptive.max_footprint = 4.0 * footprint;
        adaptive.min_leaves = particle_count / 4;
        adaptive.target_leaves = particle_count - 30;
        adaptive.max_leaves = particle_count;
        adaptive.initial_leaves = particle_count;
        adaptive.bootstrap_fine_leaves = particle_count;
        adaptive.hierarchical_restriction_policy =
            AdaptiveHierarchyRestrictionPolicy::LearnedController;
        adaptive.perception.max_neighbors = 32;
        adaptive.perception.spacing_target_neighbors = 16.0;
        let model = AdaptiveNpaModel::seeded(rule, adaptive, 11).unwrap();
        let particles = seed_adaptive_particles_scaled(
            &model,
            particle_count,
            71,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        let display_scale = adaptive_display_scale_per_footprint(&model);
        let render_scale = particles
            .represented_measure
            .iter()
            .map(|measure| {
                model
                    .config
                    .render_footprint(material_footprint_radius(*measure, 2))
                    * display_scale
            })
            .collect::<Vec<_>>();
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let gpu_rule = model.gpu_inference_rule().unwrap();
        let create = |support_bins| {
            executor.create_batched_material_state(
                &gpu_rule.rule,
                &particles.positions,
                &particles.states,
                1,
                particle_count,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                0.5,
                &[71],
                crate::gpu::WgpuMaterialStateInit {
                    represented_measure: &particles.represented_measure,
                    particle_ids: Some(&particles.particle_id),
                    update_masks: None,
                    bandwidth: &particles.bandwidth,
                    support_bins,
                    covariance: &particles.covariance,
                    state_jacobian: &particles.state_jacobian,
                    closure_mode: Some(&particles.closure_mode),
                    closure_basis: Some(&particles.closure_basis),
                    closure_phase: Some(&particles.closure_phase),
                    render_from_scale: &render_scale,
                    render_target_footprint: &render_scale,
                    display_scale_per_footprint: display_scale,
                    render_transition_steps: 0,
                },
            )
        };
        let mut global_resident = create(None).unwrap();
        let mut resident = create(Some(crate::gpu::WgpuSupportBinConfig {
            min_bandwidth: model.config.perception.min_bandwidth,
            max_bandwidth: model.config.perception.max_bandwidth,
            ratio: model.config.perception.support_bin_ratio,
            force: true,
        }))
        .unwrap();
        let neighbor_report = executor.neighbor_report(&resident);
        assert_eq!(neighbor_report.requested_support_bin_count, 4);
        assert_eq!(neighbor_report.support_bin_count, 4);
        let (_, _, diagnostics) = executor
            .capture_adaptive_diagnostics(
                &mut resident,
                model.config.base_rule_footprint(),
                model.config.perception,
            )
            .unwrap();
        let (_, _, global_diagnostics) = executor
            .capture_adaptive_diagnostics(
                &mut global_resident,
                model.config.base_rule_footprint(),
                model.config.perception,
            )
            .unwrap();
        let cpu = crate::adaptive::perception::rule_perception_pair(
            &model.config,
            &model.rule,
            &particles,
        )
        .unwrap();
        let cpu_update = model
            .rule
            .forward_update_from_features(&cpu.npa_compatible.features)
            .unwrap();
        let max_error = |lhs: &[f32], rhs: &[f32]| {
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f32, f32::max)
        };
        let binned_global_base_error = max_error(
            &diagnostics.base_features,
            &global_diagnostics.base_features,
        );
        let binned_global_normalized_error = max_error(
            &diagnostics.normalized_features,
            &global_diagnostics.normalized_features,
        );
        let binned_global_update_error =
            max_error(&diagnostics.base_update, &global_diagnostics.base_update);
        let binned_global_spacing_error = max_error(
            &diagnostics.observed_spacing,
            &global_diagnostics.observed_spacing,
        );
        assert_eq!(
            diagnostics.accepted_degree,
            global_diagnostics.accepted_degree
        );
        assert!(
            binned_global_base_error <= 1.0e-4
                && binned_global_normalized_error <= 1.0e-4
                && binned_global_update_error <= 1.0e-4
                && binned_global_spacing_error <= 1.0e-4,
            "support-bin diagnostic mismatch: base={binned_global_base_error} normalized={binned_global_normalized_error} update={binned_global_update_error} spacing={binned_global_spacing_error}",
        );
        let base_feature_error =
            max_error(&diagnostics.base_features, &cpu.npa_compatible.features);
        let normalized_feature_error =
            max_error(&diagnostics.normalized_features, &cpu.normalized.features);
        let normalized_feature_rmse = (diagnostics
            .normalized_features
            .iter()
            .zip(&cpu.normalized.features)
            .map(|(gpu, cpu)| (gpu - cpu).powi(2))
            .sum::<f32>()
            / diagnostics.normalized_features.len() as f32)
            .sqrt();
        let (normalized_max_index, normalized_gpu, normalized_cpu) = diagnostics
            .normalized_features
            .iter()
            .zip(&cpu.normalized.features)
            .enumerate()
            .max_by(|(_, (lhs_a, rhs_a)), (_, (lhs_b, rhs_b))| {
                (*lhs_a - *rhs_a).abs().total_cmp(&(*lhs_b - *rhs_b).abs())
            })
            .map(|(index, (gpu, cpu))| (index, *gpu, *cpu))
            .unwrap();
        let normalized_row = normalized_max_index / cpu.normalized.feature_dims;
        let normalized_channel = normalized_max_index % cpu.normalized.feature_dims;
        let update_error = max_error(&diagnostics.base_update, &cpu_update);
        let spacing_error = max_error(
            &diagnostics.observed_spacing,
            &cpu.normalized.observed_spacing,
        );
        assert_eq!(diagnostics.accepted_degree, cpu.normalized.accepted_degree);
        // Corrected occupancy gradients can cancel strongly across neighbors;
        // gate their aggregate fidelity and a bounded tail separately.
        assert!(
            base_feature_error <= 2.0e-3
                && normalized_feature_error <= 5.0e-3
                && normalized_feature_rmse <= 2.0e-4
                && update_error <= 2.0e-3
                && spacing_error <= 2.0e-3,
            "adaptive diagnostic mismatch: base={base_feature_error} normalized={normalized_feature_error} normalized_rmse={normalized_feature_rmse} update={update_error} spacing={spacing_error} normalized_row={normalized_row} normalized_channel={normalized_channel} gpu={normalized_gpu} cpu={normalized_cpu} moment_condition={} moment_fallback={}",
            cpu.normalized.moment_condition[normalized_row],
            cpu.normalized.moment_fallback[normalized_row],
        );

        let hierarchy = AdaptiveProxyHierarchy::build(&particles, 4).unwrap();
        let expected_features = crate::adaptive::restriction::level_one_restriction_features(
            &model, &particles, &hierarchy,
        )
        .unwrap();
        let actual_features = level_one_restriction_features_from_precomputed(
            &model,
            &particles,
            &hierarchy,
            &diagnostics.normalized_features,
            &diagnostics.base_update,
            &diagnostics.observed_spacing,
            &diagnostics.accepted_degree,
            diagnostics.feature_dims,
        )
        .unwrap();
        let restriction_feature_error = expected_features
            .iter()
            .zip(&actual_features)
            .flat_map(|(expected, actual)| {
                expected
                    .iter()
                    .zip(actual)
                    .map(|(expected, actual)| (expected - actual).abs())
            })
            .fold(0.0_f32, f32::max);
        let controller = model.restriction_controller.as_ref().unwrap();
        let costs = |features: &[[f32; crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS]]| {
            controller
                .forward(features)
                .into_iter()
                .map(|output| -output.merge_probability)
                .collect::<Vec<_>>()
        };
        let expected_mask = hierarchy
            .level_one_merge_mask(
                &particles,
                model.config.target_leaves,
                &costs(&expected_features),
            )
            .unwrap();
        let actual_mask = hierarchy
            .level_one_merge_mask(
                &particles,
                model.config.target_leaves,
                &costs(&actual_features),
            )
            .unwrap();
        assert_eq!(expected_mask, actual_mask);
        assert!(
            restriction_feature_error <= 3.0e-3,
            "restriction feature error {restriction_feature_error}",
        );

        assert!(model.config.hierarchical_bootstrap_seed);
        let expected_cut = seed_adaptive_particles_scaled(
            &model,
            model.config.target_leaves,
            71,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        let actual_cut = seed_task_particle_sets_wgpu(
            &executor,
            &model,
            &grid,
            &[71],
            model.config.target_leaves,
            AdaptiveTaskRestrictionPolicy::LearnedController,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap()
        .pop()
        .unwrap();
        assert_eq!(actual_cut.particle_id, expected_cut.particle_id);
        assert_eq!(
            actual_cut.represented_measure,
            expected_cut.represented_measure
        );
        assert_eq!(actual_cut.positions, expected_cut.positions);
        assert_eq!(actual_cut.states, expected_cut.states);
        assert_eq!(actual_cut.covariance, expected_cut.covariance);
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn normalized_primary_gpu_update_matches_cpu_on_unequal_measures() {
        let (rule_config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let mut rule = NpaModel::seeded(rule_config, 101);
        rule.weights.b2[0] = 0.125;
        let particle_count = 128;
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mean_measure = total_measure / particle_count as f32;
        let reference_footprint = material_footprint_radius(mean_measure, 2);
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive;
        adaptive.local_residual_scale = 0.0;
        adaptive.reference_footprint = reference_footprint;
        adaptive.base_rule_footprint = reference_footprint;
        adaptive.min_footprint = 0.5 * reference_footprint;
        adaptive.max_footprint = 2.0 * reference_footprint;
        adaptive.min_leaves = particle_count / 2;
        adaptive.target_leaves = particle_count;
        adaptive.max_leaves = particle_count;
        adaptive.initial_leaves = particle_count;
        adaptive.bootstrap_fine_leaves = particle_count;
        adaptive.perception.graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        let model = AdaptiveNpaModel::seeded(rule, adaptive, 103).unwrap();
        let mut particles = seed_adaptive_particles_scaled(
            &model,
            particle_count,
            107,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        for index in 0..particle_count {
            let ratio: f32 = if index % 2 == 0 { 0.5 } else { 1.5 };
            particles.represented_measure[index] = mean_measure * ratio;
            particles.bandwidth[index] = grid.eps * ratio.sqrt();
        }

        let display_scale = adaptive_display_scale_per_footprint(&model);
        let render_scale = particles
            .represented_measure
            .iter()
            .map(|measure| {
                model
                    .config
                    .render_footprint(material_footprint_radius(*measure, 2))
                    * display_scale
            })
            .collect::<Vec<_>>();
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let gpu_rule = model.gpu_inference_rule().unwrap();
        let mut resident = executor
            .create_batched_material_state(
                &gpu_rule.rule,
                &particles.positions,
                &particles.states,
                1,
                particle_count,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                1.0,
                &[107],
                crate::gpu::WgpuMaterialStateInit {
                    represented_measure: &particles.represented_measure,
                    particle_ids: Some(&particles.particle_id),
                    update_masks: None,
                    bandwidth: &particles.bandwidth,
                    support_bins: Some(crate::gpu::WgpuSupportBinConfig {
                        min_bandwidth: model.config.perception.min_bandwidth,
                        max_bandwidth: model.config.perception.max_bandwidth,
                        ratio: model.config.perception.support_bin_ratio,
                        force: true,
                    }),
                    covariance: &particles.covariance,
                    state_jacobian: &particles.state_jacobian,
                    closure_mode: Some(&particles.closure_mode),
                    closure_basis: Some(&particles.closure_basis),
                    closure_phase: Some(&particles.closure_phase),
                    render_from_scale: &render_scale,
                    render_target_footprint: &render_scale,
                    display_scale_per_footprint: display_scale,
                    render_transition_steps: 0,
                },
            )
            .unwrap();
        let neighbor_report = executor.neighbor_report(&resident);
        assert_eq!(neighbor_report.requested_support_bin_count, 4);
        assert_eq!(neighbor_report.support_bin_count, 4);
        executor
            .configure_state_adaptive_local_rule(
                &mut resident,
                gpu_rule.local_rule_mode,
                gpu_rule.local_hidden_start.unwrap(),
                model.config.local_residual_scale,
                model.config.base_rule_footprint(),
                model.config.reference_footprint,
                model.config.perception.shepard_epsilon,
                model.config.perception.moment_regularization,
                model.config.perception.moment_condition_limit,
                0,
                model.config.perception.pair_scale_power,
            )
            .unwrap();
        let diagnostics = executor
            .capture_adaptive_diagnostics_only(
                &mut resident,
                model.config.base_rule_footprint(),
                model.config.perception,
            )
            .unwrap();
        let cpu = crate::adaptive::perception::rule_perception_pair(
            &model.config,
            &model.rule,
            &particles,
        )
        .unwrap();
        let expected_update = model
            .rule
            .forward_update_from_features(&cpu.normalized.features)
            .unwrap();
        let incompatible_update = model
            .rule
            .forward_update_from_features(&cpu.npa_compatible.features)
            .unwrap();
        let max_error = |lhs: &[f32], rhs: &[f32]| {
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f32, f32::max)
        };
        let feature_error = max_error(&diagnostics.normalized_features, &cpu.normalized.features);
        let update_error = max_error(&diagnostics.model_update, &expected_update);
        let incompatible_gap = max_error(&incompatible_update, &expected_update);
        assert!(
            incompatible_gap > 1.0e-3,
            "fixture does not distinguish normalized and compatible updates: {incompatible_gap}"
        );
        assert!(
            feature_error <= 5.0e-3 && update_error <= 5.0e-3,
            "normalized-primary mismatch: feature={feature_error} update={update_error} incompatible_gap={incompatible_gap}",
        );
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn compatible_residual_gpu_update_matches_cpu_local_exposure() {
        let (rule_config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        let rule = NpaModel {
            weights: NpaWeights::zeros(&rule_config),
            config: rule_config,
        };
        let particle_count = 128;
        let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
        let mean_measure = total_measure / particle_count as f32;
        let reference_footprint = material_footprint_radius(mean_measure, 2);
        let mut adaptive = crate::adaptive::AdaptiveNpaConfig::growing_2d();
        adaptive.rule_perception = crate::adaptive::AdaptiveRulePerception::NpaCompatible;
        adaptive.local_rule_semantics =
            crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual;
        adaptive.local_residual_scale = 1.0;
        adaptive.reference_footprint = reference_footprint;
        adaptive.base_rule_footprint = reference_footprint;
        adaptive.min_footprint = 0.5 * reference_footprint;
        adaptive.max_footprint = 2.0 * reference_footprint;
        adaptive.min_leaves = particle_count / 2;
        adaptive.target_leaves = particle_count;
        adaptive.max_leaves = particle_count;
        adaptive.initial_leaves = particle_count;
        adaptive.bootstrap_fine_leaves = particle_count;
        adaptive.proxy.enabled = false;
        adaptive.perception.graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
        let mut model = AdaptiveNpaModel::seeded(rule, adaptive, 157).unwrap();
        model.enable_zero_local_residual_rule().unwrap();
        let local = model.local_residual_rule.as_mut().unwrap();
        local.weights.b2[0] = 0.5;
        local.weights.b2[2] = -0.25;

        let mut particles = seed_adaptive_particles_scaled(
            &model,
            particle_count,
            163,
            ParticleSeed::UniformCircle,
            0.2,
            total_measure,
            grid.eps,
        )
        .unwrap();
        for index in 0..particle_count {
            if index % 4 == 0 {
                particles.represented_measure[index] = 4.0 * mean_measure;
                particles.bandwidth[index] = 2.0 * grid.eps;
            } else {
                particles.represented_measure[index] = mean_measure;
                particles.bandwidth[index] = grid.eps;
            }
        }
        let cpu_perception = crate::adaptive::perception::rule_perception_pair(
            &model.config,
            &model.rule,
            &particles,
        )
        .unwrap();
        assert!(
            cpu_perception
                .npa_compatible
                .coarse_exposure
                .iter()
                .any(|value| *value > 0.0 && *value < 1.0)
        );
        let expected =
            crate::adaptive::dynamics::local_raw_update(&model, &particles, &cpu_perception)
                .unwrap()
                .combined;

        let executor = WgpuAutomataExecutor::new_blocking().unwrap();
        let mut state = executor
            .create_adaptive_state(
                &model,
                particles,
                &grid,
                1.0,
                WgpuNeighborMode::CooperativeSortedCells,
                1.0,
                163,
            )
            .unwrap();
        let diagnostics = executor
            .capture_adaptive_diagnostics_only(
                &mut state.resident,
                model.config.base_rule_footprint(),
                model.config.perception,
            )
            .unwrap();
        let max_error = diagnostics
            .model_update
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let output_dims = model.rule.config.update_dims();
        let inferred_exposure = diagnostics
            .model_update
            .chunks_exact(output_dims)
            .map(|row| row[0] / 0.5)
            .collect::<Vec<_>>();
        let exposure_error = inferred_exposure
            .iter()
            .zip(&cpu_perception.npa_compatible.coarse_exposure)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let stored_exposure_error = diagnostics
            .coarse_exposure
            .iter()
            .zip(&cpu_perception.npa_compatible.coarse_exposure)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= 5.0e-3,
            "compatible residual WGPU update diverged from CPU local exposure: update={max_error} inferred_exposure={exposure_error} stored_exposure={stored_exposure_error} gpu={:?} cpu={:?}",
            &inferred_exposure[..8],
            &cpu_perception.npa_compatible.coarse_exposure[..8],
        );
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn expected_coarse_mask_is_fractional_for_pair_merges_on_wgpu() {
        let (config, mut grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        grid.eps = 0.25;
        let mut weights = NpaWeights::zeros(&config);
        weights.b2[config.spatial_dims] = 1.0;
        let rule = NpaModel { config, weights };
        let positions = vec![[-0.4, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]];
        let states = vec![0.0; 2 * rule.config.state_dims];
        let mut particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            states,
            2,
            rule.config.state_dims,
            2.0 * std::f32::consts::PI,
            grid.eps,
        )
        .unwrap();
        particles.represented_measure[1] = 2.0 * std::f32::consts::PI;
        particles.render_footprint[1] =
            material_footprint_radius(particles.represented_measure[1], particles.spatial_dims);

        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let render = particles.render_footprint.clone();
        let mut resident = executor
            .create_batched_material_state(
                &rule,
                &particles.positions,
                &particles.states,
                1,
                2,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                0.5,
                &[17],
                crate::gpu::WgpuMaterialStateInit {
                    represented_measure: &particles.represented_measure,
                    particle_ids: Some(&particles.particle_id),
                    update_masks: None,
                    bandwidth: &particles.bandwidth,
                    support_bins: None,
                    covariance: &particles.covariance,
                    state_jacobian: &particles.state_jacobian,
                    closure_mode: Some(&particles.closure_mode),
                    closure_basis: Some(&particles.closure_basis),
                    closure_phase: Some(&particles.closure_phase),
                    render_from_scale: &render,
                    render_target_footprint: &render,
                    display_scale_per_footprint: 1.0,
                    render_transition_steps: 0,
                },
            )
            .unwrap();
        executor
            .configure_state_adaptive_integration(&mut resident, 1.0, true)
            .unwrap();
        executor.step_state(&mut resident).unwrap();
        let (_, next_states) = executor.read_positions_states(&resident).unwrap();
        let native = next_states[0];
        let coarse = next_states[rule.config.state_dims];
        assert!(native == 0.0 || native == 1.0, "native gate was {native}");
        assert!((coarse - 0.5).abs() <= 1.0e-6, "coarse gate was {coarse}");
    }

    #[test]
    #[ignore = "device parity test; run explicitly with --ignored"]
    fn lineage_mask_restricts_child_draws_on_wgpu() {
        let (config, mut grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        grid.eps = 0.25;
        let mut weights = NpaWeights::zeros(&config);
        weights.b2[config.spatial_dims] = 1.0;
        let rule = NpaModel { config, weights };
        let probability = 0.5;
        let seed = 17_u32;

        let mut active = Vec::with_capacity(4);
        let mut inactive = Vec::with_capacity(4);
        for particle_id in 0_u32..256 {
            if gpu_update_draw(particle_id, 0, seed, probability) {
                active.push(particle_id as u64);
            } else {
                inactive.push(particle_id as u64);
            }
            if active.len() >= 2 && inactive.len() >= 2 {
                break;
            }
        }
        assert!(active.len() >= 2 && inactive.len() >= 2);

        let mut particle_ids = [0_u64; WGPU_MATERIAL_UPDATE_MASK_MEMBERS];
        let mut mask_weights = [0.0_f32; WGPU_MATERIAL_UPDATE_MASK_MEMBERS];
        particle_ids[..4].copy_from_slice(&[active[0], active[1], inactive[0], inactive[1]]);
        mask_weights[..4].fill(0.25);
        let update_mask = [WgpuMaterialUpdateMask {
            particle_ids,
            weights: mask_weights,
        }];

        let positions = [[0.0, 0.0, 0.0, 0.0]];
        let states = vec![0.0; rule.config.state_dims];
        let represented_measure = [4.0 * std::f32::consts::PI];
        let bandwidth = [grid.eps];
        let covariance = [[0.0; 9]];
        let state_jacobian = vec![0.0; rule.config.state_dims * rule.config.spatial_dims];
        let render = [material_footprint_radius(
            represented_measure[0],
            rule.config.spatial_dims,
        )];
        let executor = WgpuAutomataExecutor::new_restriction_blocking().unwrap();
        let mut resident = executor
            .create_batched_material_state(
                &rule,
                &positions,
                &states,
                1,
                1,
                &grid,
                1.0,
                WgpuNeighborMode::SubgroupCooperativeSortedCells,
                probability,
                &[seed as u64],
                crate::gpu::WgpuMaterialStateInit {
                    represented_measure: &represented_measure,
                    particle_ids: None,
                    update_masks: Some(&update_mask),
                    bandwidth: &bandwidth,
                    support_bins: None,
                    covariance: &covariance,
                    state_jacobian: &state_jacobian,
                    closure_mode: None,
                    closure_basis: None,
                    closure_phase: None,
                    render_from_scale: &render,
                    render_target_footprint: &render,
                    display_scale_per_footprint: 1.0,
                    render_transition_steps: 0,
                },
            )
            .unwrap();
        executor
            .configure_state_adaptive_integration(&mut resident, 1.0, false)
            .unwrap();
        executor.step_state(&mut resident).unwrap();
        let (_, next_states) = executor.read_positions_states(&resident).unwrap();
        assert!(
            (next_states[0] - 0.5).abs() <= 1.0e-6,
            "lineage-restricted gate was {}",
            next_states[0]
        );
    }
}
