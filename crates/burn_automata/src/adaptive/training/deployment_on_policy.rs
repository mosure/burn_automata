use burn_automata_kernels::HashGridConfig;

#[cfg(feature = "gpu_wgpu")]
use super::normalize_positive_weights;
use super::{AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig};
use crate::{AdaptiveNpaModel, AutomataError, AutomataResult};

#[cfg(feature = "gpu_wgpu")]
use {
    super::AdaptiveMultiscaleDatasetReport,
    crate::{
        ParticleSeed,
        adaptive::{
            AdaptiveParticleSet,
            features::{local_residual_auxiliary_dims, local_residual_features, proxy_context},
            material_footprint_radius,
            perception::rule_perception_pair,
            seed_adaptive_particles_scaled,
        },
        gpu::{WgpuAutomataExecutor, WgpuNeighborMode},
    },
    rand::{SeedableRng, rngs::StdRng, seq::index},
    std::time::Instant,
};

/// Generates deployment-policy replay with the same resident WGPU rollout used
/// by inference. Exact local/proxy labels are evaluated only at sampled
/// snapshots instead of recomputing the full CPU policy at every dynamics step.
#[cfg(feature = "gpu_wgpu")]
pub fn adaptive_deployment_on_policy_batch_wgpu(
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    validate(model, config)?;
    let started = Instant::now();
    let executor = WgpuAutomataExecutor::new_blocking()?;
    let snapshots = collect_snapshots(&executor, model, grid, config, round)?;
    build_batch(model, config, round, snapshots, started)
}

#[cfg(not(feature = "gpu_wgpu"))]
pub fn adaptive_deployment_on_policy_batch_wgpu(
    _model: &AdaptiveNpaModel,
    _grid: &HashGridConfig,
    _config: &AdaptiveMultiscaleTrainingConfig,
    _round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    Err(AutomataError::InvalidArgument(
        "resident adaptive replay generation requires gpu_wgpu".to_string(),
    ))
}

#[cfg(feature = "gpu_wgpu")]
pub(super) fn collect_snapshots(
    executor: &WgpuAutomataExecutor,
    model: &AdaptiveNpaModel,
    grid: &HashGridConfig,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<Vec<AdaptiveParticleSet>> {
    let snapshot_count = config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1;
    let mut snapshots = Vec::with_capacity(config.on_policy_rollouts * snapshot_count);
    for rollout_index in 0..config.on_policy_rollouts {
        let rollout_seed = rollout_seed(config.seed, round, rollout_index);
        let initial_count = model.config.initial_leaf_count();
        let particles = seed_adaptive_particles_scaled(
            model,
            initial_count,
            rollout_seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
            config.total_measure,
            config.bandwidth,
        )?;
        let mut state = executor.create_adaptive_state(
            model,
            particles,
            grid,
            config.dt,
            WgpuNeighborMode::CooperativeSortedCells,
            config.update_prob,
            rollout_seed,
        )?;
        snapshots.push(state.particles.clone());
        let mut completed = 0;
        while completed < config.on_policy_rollout_steps {
            let segment = config
                .on_policy_snapshot_interval
                .min(config.on_policy_rollout_steps - completed);
            executor.step_adaptive_state_many_with_topology_control(
                &mut state,
                segment,
                true,
                config.on_policy_topology_control,
            )?;
            completed += segment;
            if completed.is_multiple_of(config.on_policy_snapshot_interval)
                || completed == config.on_policy_rollout_steps
            {
                executor.synchronize_adaptive_particles(&mut state)?;
                snapshots.push(state.particles.clone());
            }
        }
    }
    Ok(snapshots)
}

#[cfg(feature = "gpu_wgpu")]
fn build_batch(
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
    snapshots: Vec<AdaptiveParticleSet>,
    started: Instant,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    let input_dims = model.rule.config.perception_dims();
    let local_input_dims =
        input_dims + local_residual_auxiliary_dims(&model.config, model.rule.config.state_dims);
    let output_dims = model.rule.config.update_dims();
    let expected_rows = snapshots.len()
        * config
            .on_policy_rows_per_snapshot
            .min(model.config.max_leaves);
    let mut local_features = Vec::with_capacity(expected_rows * local_input_dims);
    let mut proxy_features = Vec::with_capacity(if model.config.proxy.context_scale > 0.0 {
        expected_rows * input_dims
    } else {
        0
    });
    let mut deployment_features = Vec::with_capacity(expected_rows * input_dims);
    let mut deployment_row_weights = Vec::with_capacity(expected_rows);
    let mut deployment_residual_gate = Vec::with_capacity(expected_rows);
    let mut row_weights = Vec::with_capacity(expected_rows);
    let mut footprints = Vec::new();
    let mut proxy_nodes = 0usize;
    let mut minimum_material_leaves = usize::MAX;
    let mut maximum_material_leaves = 0usize;

    for (snapshot_index, particles) in snapshots.iter().enumerate() {
        let count = particles.len();
        minimum_material_leaves = minimum_material_leaves.min(count);
        maximum_material_leaves = maximum_material_leaves.max(count);
        let perception = rule_perception_pair(&model.config, &model.rule, particles)?;
        let closure_features =
            local_residual_features(&model.config, particles, &perception.normalized)?;
        let proxy = if model.config.proxy.enabled && model.config.proxy.context_scale > 0.0 {
            Some(proxy_context(&model.config, particles)?.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "adaptive deployment replay requires the proxy branch".to_string(),
                )
            })?)
        } else {
            None
        };
        proxy_nodes += proxy.as_ref().map_or(0, |proxy| proxy.node_count);
        footprints.extend(
            particles
                .represented_measure
                .iter()
                .map(|measure| material_footprint_radius(*measure, particles.spatial_dims)),
        );
        let selected_count = config.on_policy_rows_per_snapshot.min(count);
        let mut rng = StdRng::seed_from_u64(
            config.seed
                ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (snapshot_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        );
        let selected = index::sample(&mut rng, count, selected_count);
        let mean_measure = config.total_measure / count as f32;
        for row in selected.iter() {
            local_features.extend_from_slice(
                &closure_features[row * local_input_dims..(row + 1) * local_input_dims],
            );
            deployment_features.extend_from_slice(
                &perception.npa_compatible.features[row * input_dims..(row + 1) * input_dims],
            );
            if let Some(proxy) = &proxy {
                proxy_features.extend_from_slice(
                    &proxy.perception.features[row * input_dims..(row + 1) * input_dims],
                );
            }
            let measure_weight =
                particles.represented_measure[row] / mean_measure.max(f32::MIN_POSITIVE);
            let gate = model.config.residual_gate(material_footprint_radius(
                particles.represented_measure[row],
                particles.spatial_dims,
            ));
            deployment_row_weights.push(measure_weight);
            deployment_residual_gate.push(gate);
            row_weights.push(measure_weight * (gate.powi(2) + config.residual_coordinate_weight));
        }
    }
    normalize_positive_weights(&mut deployment_row_weights, "deployment replay row")?;
    normalize_positive_weights(&mut row_weights, "deployment replay residual row")?;
    let rows = deployment_row_weights.len();
    let footprint_mean = mean(&footprints);
    let footprint_variance = footprints
        .iter()
        .map(|value| (*value - footprint_mean).powi(2))
        .sum::<f32>()
        / footprints.len().max(1) as f32;
    let batch = AdaptiveMultiscaleTrainingBatch {
        local_features,
        closure_features: Vec::new(),
        proxy_features,
        target_update: vec![0.0; rows * output_dims],
        closure_mode_target_update: Vec::new(),
        closure_basis_target_update: Vec::new(),
        closure_mode_row_weights: Vec::new(),
        deployment_features,
        deployment_target_update: vec![0.0; rows * output_dims],
        deployment_row_weights,
        deployment_residual_gate,
        controller_features: vec![0.0; rows * crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS],
        controller_targets: vec![0.0; rows * crate::adaptive::ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
        row_weights,
        rows,
        report: AdaptiveMultiscaleDatasetReport {
            rollouts: config.on_policy_rollouts,
            snapshots: snapshots.len(),
            cuts: snapshots.len(),
            rows,
            minimum_material_leaves,
            maximum_material_leaves,
            minimum_footprint: footprints.iter().copied().fold(f32::INFINITY, f32::min),
            maximum_footprint: footprints.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            footprint_coefficient_of_variation: footprint_variance.sqrt()
                / footprint_mean.max(f32::MIN_POSITIVE),
            mean_proxy_nodes: proxy_nodes as f32 / snapshots.len().max(1) as f32,
            mean_counterfactual_error: 0.0,
            mean_teacher_update_error: 0.0,
            generation_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
            ..AdaptiveMultiscaleDatasetReport::default()
        },
    };
    batch.validate(input_dims, output_dims)?;
    Ok(batch)
}

#[cfg(feature = "gpu_wgpu")]
fn validate(
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<()> {
    model.validate()?;
    if !model.uses_deployment_rule()
        || config.on_policy_rollouts == 0
        || config.on_policy_rollout_steps == 0
        || config.on_policy_snapshot_interval == 0
        || config.on_policy_rows_per_snapshot == 0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid resident adaptive deployment replay configuration".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu_wgpu")]
fn rollout_seed(seed: u64, round: usize, rollout_index: usize) -> u64 {
    if rollout_index == 0 {
        seed
    } else {
        seed.wrapping_add((round as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03))
            .wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

#[cfg(feature = "gpu_wgpu")]
fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}
