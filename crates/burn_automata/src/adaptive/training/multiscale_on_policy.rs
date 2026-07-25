use std::time::Instant;

use rand::{SeedableRng, rngs::StdRng, seq::index};

use super::{
    AdaptiveMultiscaleDatasetReport, AdaptiveMultiscaleRuleStrategy,
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig, AdaptiveReplayBackend,
    AdaptiveReplayTeacher,
    multiscale_dataset::{controller_budget_allocation, controller_target},
    normalize_positive_weights,
};
use crate::{
    AutomataError, AutomataResult, NpaModel, ParticleSeed,
    adaptive::{
        AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveRolloutConfig, AdaptiveTopologyControl,
        features::{
            closure_recurrent_auxiliary_dims, closure_recurrent_features_for_rows,
            controller_features_for_rows, local_detail_risk, local_residual_auxiliary_dims,
            local_residual_features, local_residual_features_for_rows, local_residual_gate,
            local_rule_perception, proxy_context,
        },
        material_footprint_radius,
        perception::rule_perception_pair,
        refinement::adaptive_refinement_defect,
        rollout::advance_adaptive_rollout_with_topology_control,
        seed_adaptive_particles_scaled,
    },
};
use burn_automata_kernels::{
    AdaptiveGraphPolicy, AdaptiveNpaPerceptionOptions, HashGridConfig, adaptive_perceive_pair,
};

pub fn adaptive_multiscale_on_policy_batch(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    validate(teacher, grid, model, config)?;
    if config.on_policy_teacher == AdaptiveReplayTeacher::CoupledFine {
        return super::recurrent_replay::adaptive_coupled_fine_replay_batch(
            teacher, grid, model, config, round,
        );
    }
    let started = Instant::now();
    let snapshots = label_snapshots(
        teacher,
        model,
        config,
        config.on_policy_teacher,
        collect_snapshots(grid, model, config, round)?,
    )?;
    build_batch(teacher, model, config, round, snapshots, started)
}

#[cfg(feature = "gpu_wgpu")]
pub fn adaptive_multiscale_on_policy_batch_wgpu_with_executor(
    executor: &crate::gpu::WgpuAutomataExecutor,
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    validate(teacher, grid, model, config)?;
    if config.on_policy_replay_backend != AdaptiveReplayBackend::WgpuResident {
        return Err(AutomataError::InvalidArgument(
            "resident on-policy replay requires on_policy_replay_backend = wgpu-resident"
                .to_string(),
        ));
    }
    if config.on_policy_teacher == AdaptiveReplayTeacher::CoupledFine {
        return super::recurrent_replay::adaptive_coupled_fine_replay_batch_wgpu_with_executor(
            executor, teacher, grid, model, config, round,
        );
    }
    if model.config.proxy.enabled
        && model.config.proxy.context_scale > 0.0
        && !model.uses_deployment_rule()
    {
        return Err(AutomataError::InvalidArgument(
            "exact resident replay requires proxy.context_scale = 0 unless a deployable rule already contains that context"
                .to_string(),
        ));
    }
    let started = Instant::now();
    let snapshots = label_snapshots(
        teacher,
        model,
        config,
        config.on_policy_teacher,
        super::deployment_on_policy::collect_snapshots(executor, model, grid, config, round)?,
    )?;
    build_batch(teacher, model, config, round, snapshots, started)
}

fn label_snapshots(
    teacher: &NpaModel,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    teacher_kind: AdaptiveReplayTeacher,
    particles: Vec<AdaptiveParticleSet>,
) -> AutomataResult<Vec<OnPolicySnapshot>> {
    let quadrature_teacher = (teacher_kind == AdaptiveReplayTeacher::MarkovQuadrature)
        .then(|| markov_quadrature_teacher(teacher, model));
    particles
        .into_iter()
        .enumerate()
        .map(|(snapshot_index, particles)| {
            let teacher_update = quadrature_teacher
                .as_ref()
                .map(|teacher| {
                    super::super::dynamics::fine_quadrature_raw_update(teacher, &particles)
                        .map(|update| update.combined)
                })
                .transpose()?;
            Ok(OnPolicySnapshot {
                particles,
                teacher_update,
                closure_mode_target_update: None,
                closure_basis_target_update: None,
                captured_perception: None,
                rollout_index: snapshot_index
                    / (config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1),
                step: (snapshot_index
                    % (config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1))
                    * config.on_policy_snapshot_interval,
            })
        })
        .collect()
}

fn markov_quadrature_teacher(teacher: &NpaModel, model: &AdaptiveNpaModel) -> AdaptiveNpaModel {
    let mut oracle = model.clone();
    oracle.rule = teacher.clone();
    oracle.config.coarse_dynamics = super::super::AdaptiveCoarseDynamics::FineQuadrature;
    oracle.config.local_residual_scale = 0.0;
    oracle.config.local_rule_semantics = super::super::AdaptiveLocalRuleSemantics::Residual;
    oracle.config.proxy.context_scale = 0.0;
    oracle.deployment_rule = None;
    oracle.deployment_local_rule = None;
    oracle
}

pub(super) struct OnPolicySnapshot {
    pub particles: AdaptiveParticleSet,
    pub teacher_update: Option<Vec<f32>>,
    pub closure_mode_target_update: Option<Vec<f32>>,
    pub closure_basis_target_update: Option<Vec<f32>>,
    pub captured_perception: Option<OnPolicyCapturedPerception>,
    pub rollout_index: usize,
    pub step: usize,
}

pub(super) struct OnPolicyCapturedPerception {
    pub perception: burn_automata_kernels::AdaptivePerceptionPair,
    pub base_update: Vec<f32>,
    pub model_update: Vec<f32>,
}

fn collect_snapshots(
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<Vec<AdaptiveParticleSet>> {
    match config.on_policy_replay_backend {
        AdaptiveReplayBackend::CpuReference => collect_cpu_snapshots(model, config, round),
        AdaptiveReplayBackend::WgpuResident => {
            #[cfg(feature = "gpu_wgpu")]
            {
                if model.config.proxy.enabled
                    && model.config.proxy.context_scale > 0.0
                    && !model.uses_deployment_rule()
                {
                    return Err(AutomataError::InvalidArgument(
                        "exact resident replay requires proxy.context_scale = 0 unless a deployable rule already contains that context"
                            .to_string(),
                    ));
                }
                let executor = crate::gpu::WgpuAutomataExecutor::new_blocking()?;
                super::deployment_on_policy::collect_snapshots(
                    &executor, model, grid, config, round,
                )
            }
            #[cfg(not(feature = "gpu_wgpu"))]
            {
                let _ = (grid, model, config, round);
                Err(AutomataError::InvalidArgument(
                    "resident exact adaptive replay requires gpu_wgpu".to_string(),
                ))
            }
        }
    }
}

fn collect_cpu_snapshots(
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
) -> AutomataResult<Vec<AdaptiveParticleSet>> {
    let snapshot_count = config.on_policy_rollout_steps / config.on_policy_snapshot_interval + 1;
    let mut snapshots = Vec::with_capacity(config.on_policy_rollouts * snapshot_count);
    for rollout_index in 0..config.on_policy_rollouts {
        let rollout_seed = rollout_seed(config.seed, round, rollout_index);
        let initial_count = model.config.initial_leaf_count();
        let mut particles = seed_adaptive_particles_scaled(
            model,
            initial_count,
            rollout_seed,
            ParticleSeed::UniformCircle,
            config.seed_scale,
            config.total_measure,
            config.bandwidth,
        )?;
        for step in 0..=config.on_policy_rollout_steps {
            if step.is_multiple_of(config.on_policy_snapshot_interval) {
                snapshots.push(particles.clone());
            }
            if step == config.on_policy_rollout_steps {
                break;
            }
            particles = advance_adaptive_rollout_with_topology_control(
                model,
                particles,
                AdaptiveRolloutConfig {
                    steps: 1,
                    dt: config.dt,
                    update_prob: config.update_prob,
                    seed: rollout_seed,
                    bandwidth_adaptation_enabled: true,
                    topology_enabled: true,
                    snapshot_interval: 1,
                },
                step,
                config.on_policy_topology_control,
            )?
            .particles;
        }
    }
    Ok(snapshots)
}

pub(super) fn build_batch(
    teacher: &NpaModel,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
    round: usize,
    snapshots: Vec<OnPolicySnapshot>,
    started: Instant,
) -> AutomataResult<AdaptiveMultiscaleTrainingBatch> {
    let (split_label_ratio, merge_label_ratio) = config.controller_label_ratios(&model.config)?;
    let input_dims = teacher.config.perception_dims();
    let local_input_dims =
        input_dims + local_residual_auxiliary_dims(&model.config, teacher.config.state_dims);
    let output_dims = teacher.config.update_dims();
    let expected_rows = snapshots.len()
        * config
            .on_policy_rows_per_snapshot
            .min(model.config.max_leaves);
    let mut local_features = Vec::with_capacity(expected_rows * local_input_dims);
    let closure_input_dims =
        input_dims + closure_recurrent_auxiliary_dims(&model.config, teacher.config.state_dims);
    let mut closure_features = Vec::with_capacity(
        usize::from(model.config.closure_recurrent_mode) * expected_rows * closure_input_dims,
    );
    let mut proxy_features = Vec::with_capacity(if model.config.proxy.context_scale > 0.0 {
        expected_rows * input_dims
    } else {
        0
    });
    let mut target_update = Vec::with_capacity(expected_rows * output_dims);
    let mut closure_mode_target_update = Vec::with_capacity(
        usize::from(model.config.closure_recurrent_mode) * expected_rows * output_dims,
    );
    let mut closure_basis_target_update = Vec::with_capacity(
        usize::from(model.config.closure_recurrent_mode) * expected_rows * output_dims,
    );
    let mut closure_mode_row_weights =
        Vec::with_capacity(usize::from(model.config.closure_recurrent_mode) * expected_rows);
    let mut deployment_features = Vec::with_capacity(expected_rows * input_dims);
    let mut deployment_target_update = Vec::with_capacity(expected_rows * output_dims);
    let mut deployment_row_weights = Vec::with_capacity(expected_rows);
    let mut deployment_residual_gate = Vec::with_capacity(expected_rows);
    let mut controller_input =
        Vec::with_capacity(expected_rows * crate::adaptive::ADAPTIVE_CONTROLLER_INPUT_DIMS);
    let mut controller_targets =
        Vec::with_capacity(expected_rows * crate::adaptive::ADAPTIVE_CONTROLLER_OUTPUT_DIMS);
    let mut row_weights = Vec::with_capacity(expected_rows);
    let mut footprints = Vec::new();
    let mut proxy_nodes = 0usize;
    let mut minimum_material_leaves = usize::MAX;
    let mut maximum_material_leaves = 0usize;
    let mut counterfactual_error_sum = 0.0_f64;
    let mut counterfactual_error_rows = 0usize;
    let mut teacher_error_sum = 0.0_f64;
    let mut teacher_error_rows = 0usize;
    let mut maximum_particle_state_absolute = 0.0_f32;
    let mut maximum_closure_mode_absolute = 0.0_f32;
    let mut rollout_maximum_particle_state_absolute = vec![0.0_f32; config.on_policy_rollouts];
    let mut rollout_maximum_teacher_update_absolute = vec![0.0_f32; config.on_policy_rollouts];
    let mut rollout_maximum_closure_target_absolute = vec![0.0_f32; config.on_policy_rollouts];
    let mut rollout_peak_particle_state_step = vec![0; config.on_policy_rollouts];
    let mut rollout_peak_teacher_update_step = vec![0; config.on_policy_rollouts];
    let mut rollout_peak_closure_target_step = vec![0; config.on_policy_rollouts];

    for (snapshot_index, snapshot) in snapshots.iter().enumerate() {
        let particles = &snapshot.particles;
        let state_maximum = particles
            .states
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        maximum_particle_state_absolute = maximum_particle_state_absolute.max(
            particles
                .states
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
        );
        maximum_closure_mode_absolute = maximum_closure_mode_absolute.max(
            particles
                .closure_mode
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
        );
        if let (Some(maximum), Some(peak_step)) = (
            rollout_maximum_particle_state_absolute.get_mut(snapshot.rollout_index),
            rollout_peak_particle_state_step.get_mut(snapshot.rollout_index),
        ) && state_maximum > *maximum
        {
            *maximum = state_maximum;
            *peak_step = snapshot.step;
        }
        let count = particles.len();
        minimum_material_leaves = minimum_material_leaves.min(count);
        maximum_material_leaves = maximum_material_leaves.max(count);
        let computed_perception;
        let model_perception = if let Some(captured) = &snapshot.captured_perception {
            &captured.perception
        } else {
            computed_perception = rule_perception_pair(&model.config, &model.rule, particles)?;
            &computed_perception
        };
        let normalized = &model_perception.normalized;
        let residual_perception = local_rule_perception(&model.config, model_perception);
        let computed_base_update;
        let base_update = if let Some(captured) = &snapshot.captured_perception {
            captured.base_update.as_slice()
        } else {
            let runtime_features = match model.config.rule_perception {
                crate::adaptive::AdaptiveRulePerception::NpaCompatible => {
                    &model_perception.npa_compatible.features
                }
                crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive => {
                    &model_perception.normalized.features
                }
            };
            computed_base_update = model.rule.forward_update_from_features(runtime_features)?;
            computed_base_update.as_slice()
        };
        let selected_count = config.on_policy_rows_per_snapshot.min(count);
        let mut sample_rng = StdRng::seed_from_u64(
            config.seed
                ^ (round as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)
                ^ (snapshot_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let selected = index::sample(&mut sample_rng, count, selected_count);
        let selected_rows = selected.iter().collect::<Vec<_>>();
        let selected_closure_features = local_residual_features_for_rows(
            &model.config,
            particles,
            residual_perception,
            &selected_rows,
        )?;
        let selected_recurrent_features = model
            .config
            .closure_recurrent_mode
            .then(|| {
                closure_recurrent_features_for_rows(
                    &model.config,
                    particles,
                    normalized,
                    &selected_rows,
                )
            })
            .transpose()?;
        let proxy = if model.config.proxy.enabled && model.config.proxy.context_scale > 0.0 {
            Some(proxy_context(&model.config, particles)?.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "adaptive on-policy training requires the proxy branch".to_string(),
                )
            })?)
        } else {
            None
        };
        proxy_nodes += proxy.as_ref().map_or(0, |proxy| proxy.node_count);
        let teacher_update = if let Some(update) = &snapshot.teacher_update {
            if update.len() != count * output_dims {
                return Err(AutomataError::InvalidArgument(
                    "coupled fine replay teacher update shape mismatch".to_string(),
                ));
            }
            update.clone()
        } else {
            let teacher_bandwidth = vec![config.bandwidth; count];
            let teacher_perception = adaptive_perceive_pair(
                &particles.positions,
                &particles.states,
                &particles.represented_measure,
                &teacher_bandwidth,
                1,
                count,
                particles.state_dims,
                model.config.perception,
                AdaptiveGraphPolicy::RawSupport,
                AdaptiveNpaPerceptionOptions {
                    eps0: teacher.config.eps0,
                    scale_equivariance: teacher.config.scale_equivariant(),
                    particle_density_equivariance: teacher.config.particle_density_equivariant(),
                    log_norm_grad: teacher.config.log_norm_grad,
                    log_norm_density_grad: teacher.config.log_norm_density_grad,
                    position_features: teacher.config.position_features,
                },
            )?;
            teacher.forward_update_from_features(&teacher_perception.npa_compatible.features)?
        };
        let closure_teacher = if model.config.closure_recurrent_mode {
            let update = snapshot
                .closure_mode_target_update
                .as_ref()
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "recurrent closure replay snapshot has no closure teacher target"
                            .to_owned(),
                    )
                })?;
            if update.len() != count * output_dims {
                return Err(AutomataError::InvalidArgument(
                    "recurrent closure replay target shape mismatch".to_owned(),
                ));
            }
            Some(update)
        } else {
            None
        };
        let teacher_maximum = teacher_update
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        if let (Some(maximum), Some(peak_step)) = (
            rollout_maximum_teacher_update_absolute.get_mut(snapshot.rollout_index),
            rollout_peak_teacher_update_step.get_mut(snapshot.rollout_index),
        ) && teacher_maximum > *maximum
        {
            *maximum = teacher_maximum;
            *peak_step = snapshot.step;
        }
        if let (Some(closure_teacher), Some(maximum), Some(peak_step)) = (
            closure_teacher,
            rollout_maximum_closure_target_absolute.get_mut(snapshot.rollout_index),
            rollout_peak_closure_target_step.get_mut(snapshot.rollout_index),
        ) {
            let closure_maximum = closure_teacher
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            if closure_maximum > *maximum {
                *maximum = closure_maximum;
                *peak_step = snapshot.step;
            }
        }
        let computed_model_update;
        let model_update = if let Some(captured) = &snapshot.captured_perception {
            captured.model_update.as_slice()
        } else if config.rule_strategy == AdaptiveMultiscaleRuleStrategy::FullNormalized {
            base_update
        } else {
            let closure_features =
                local_residual_features(&model.config, particles, residual_perception)?;
            let local_prediction = model
                .local_residual_rule
                .as_ref()
                .expect("local residual rule validated")
                .forward_update_from_features(&closure_features)?;
            let proxy_prediction = if let Some(proxy) = &proxy {
                model
                    .proxy_rule
                    .as_ref()
                    .expect("proxy rule validated")
                    .forward_update_from_features(&proxy.perception.features)?
            } else {
                vec![0.0; count * output_dims]
            };
            computed_model_update = match config.rule_strategy {
                AdaptiveMultiscaleRuleStrategy::CoarseReplacement => (0..count * output_dims)
                    .map(|index| {
                        let row = index / output_dims;
                        if model
                            .config
                            .is_coarse_rule_footprint(particles.footprint(row))
                        {
                            local_prediction[index]
                        } else {
                            base_update[index]
                        }
                    })
                    .collect::<Vec<_>>(),
                AdaptiveMultiscaleRuleStrategy::Residual => (0..count * output_dims)
                    .map(|index| {
                        let row = index / output_dims;
                        let gate =
                            local_residual_gate(&model.config, particles, residual_perception, row);
                        base_update[index]
                            + gate
                                * (local_prediction[index]
                                    + model.config.proxy.context_scale * proxy_prediction[index])
                    })
                    .collect::<Vec<_>>(),
                AdaptiveMultiscaleRuleStrategy::FullNormalized => unreachable!(),
            };
            computed_model_update.as_slice()
        };
        for row in 0..count {
            let error = (0..output_dims)
                .map(|channel| {
                    let index = row * output_dims + channel;
                    (model_update[index] - teacher_update[index]).powi(2)
                })
                .sum::<f32>()
                / output_dims as f32;
            teacher_error_sum += error.sqrt().max(1.0e-6) as f64;
            teacher_error_rows += 1;
        }
        let risk = match config.on_policy_topology_control {
            AdaptiveTopologyControl::Learned
            | AdaptiveTopologyControl::LearnedRefinementDefect
            | AdaptiveTopologyControl::RefinementDefectOracle => {
                adaptive_refinement_defect(model, particles)?
            }
            AdaptiveTopologyControl::LocalDetailOracle
            | AdaptiveTopologyControl::PairedLocalDetail
            | AdaptiveTopologyControl::ContinuousLocalDetail => {
                local_detail_risk(particles, normalized)
            }
        };
        counterfactual_error_sum += risk.iter().map(|value| *value as f64).sum::<f64>();
        counterfactual_error_rows += risk.len();
        let control = controller_features_for_rows(
            &model.config,
            particles,
            normalized,
            base_update,
            &selected_rows,
        );
        let allocation = controller_budget_allocation(
            &model.config,
            &risk,
            &particles.represented_measure,
            particles.spatial_dims,
        )?;
        let control_target = controller_target(
            &model.config,
            particles,
            &normalized.observed_spacing,
            &allocation.desired_footprint,
            config.bandwidth,
            split_label_ratio,
            merge_label_ratio,
        );
        footprints.extend(
            particles
                .represented_measure
                .iter()
                .map(|measure| material_footprint_radius(*measure, particles.spatial_dims)),
        );
        let mean_measure = config.total_measure / count as f32;
        for (selected_index, &row) in selected_rows.iter().enumerate() {
            local_features.extend_from_slice(
                &selected_closure_features
                    [selected_index * local_input_dims..(selected_index + 1) * local_input_dims],
            );
            if let Some(recurrent) = &selected_recurrent_features {
                closure_features.extend_from_slice(
                    &recurrent[selected_index * closure_input_dims
                        ..(selected_index + 1) * closure_input_dims],
                );
            }
            deployment_features.extend_from_slice(
                &model_perception.npa_compatible.features[row * input_dims..(row + 1) * input_dims],
            );
            deployment_target_update
                .extend_from_slice(&teacher_update[row * output_dims..(row + 1) * output_dims]);
            if let Some(proxy) = &proxy {
                proxy_features.extend_from_slice(
                    &proxy.perception.features[row * input_dims..(row + 1) * input_dims],
                );
            }
            let gate = local_residual_gate(&model.config, particles, residual_perception, row);
            for channel in 0..output_dims {
                let target_index = row * output_dims + channel;
                target_update.push(
                    if model.config.local_rule_semantics
                        == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                    {
                        teacher_update[target_index] - base_update[target_index]
                    } else if gate.abs() > 1.0e-6 {
                        (teacher_update[target_index] - base_update[target_index]) / gate
                    } else {
                        0.0
                    },
                );
            }
            controller_input.extend_from_slice(&control[selected_index]);
            controller_targets.extend_from_slice(
                &control_target[row * crate::adaptive::ADAPTIVE_CONTROLLER_OUTPUT_DIMS
                    ..(row + 1) * crate::adaptive::ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
            );
            let measure_weight =
                particles.represented_measure[row] / mean_measure.max(f32::MIN_POSITIVE);
            if let Some(closure_teacher) = closure_teacher {
                closure_mode_target_update.extend_from_slice(
                    &closure_teacher[row * output_dims..(row + 1) * output_dims],
                );
                let basis_teacher =
                    snapshot
                        .closure_basis_target_update
                        .as_ref()
                        .ok_or_else(|| {
                            AutomataError::InvalidModel(
                                "recurrent closure snapshot is missing basis targets".to_owned(),
                            )
                        })?;
                closure_basis_target_update
                    .extend_from_slice(&basis_teacher[row * output_dims..(row + 1) * output_dims]);
                closure_mode_row_weights.push(
                    measure_weight
                        * f32::from(
                            model
                                .config
                                .is_coarse_rule_footprint(particles.footprint(row)),
                        ),
                );
            }
            row_weights.push(
                if model.config.local_rule_semantics
                    == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                {
                    measure_weight
                } else {
                    measure_weight * (gate.powi(2) + config.residual_coordinate_weight)
                },
            );
            deployment_row_weights.push(measure_weight);
            deployment_residual_gate.push(gate);
        }
    }

    normalize_positive_weights(&mut row_weights, "on-policy row")?;
    normalize_positive_weights(&mut deployment_row_weights, "on-policy deployment row")?;
    if !closure_mode_row_weights.is_empty() {
        normalize_positive_weights(&mut closure_mode_row_weights, "on-policy closure-mode row")?;
    }
    let rows = row_weights.len();
    let footprint_mean = mean(&footprints);
    let footprint_variance = footprints
        .iter()
        .map(|value| (*value - footprint_mean).powi(2))
        .sum::<f32>()
        / footprints.len().max(1) as f32;
    let teacher_update_p99_absolute = super::absolute_percentile(&deployment_target_update, 0.99);
    let maximum_teacher_update_absolute = deployment_target_update
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let closure_target_p99_absolute = super::absolute_percentile(&closure_mode_target_update, 0.99);
    let maximum_closure_target_absolute = closure_mode_target_update
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let closure_basis_target_p99_absolute =
        super::absolute_percentile(&closure_basis_target_update, 0.99);
    let maximum_closure_basis_target_absolute = closure_basis_target_update
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let batch = AdaptiveMultiscaleTrainingBatch {
        local_features,
        closure_features,
        proxy_features,
        target_update,
        closure_mode_target_update,
        closure_basis_target_update,
        closure_mode_row_weights,
        deployment_features,
        deployment_target_update,
        deployment_row_weights,
        deployment_residual_gate,
        controller_features: controller_input,
        controller_targets,
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
            mean_counterfactual_error: (counterfactual_error_sum
                / counterfactual_error_rows.max(1) as f64)
                as f32,
            mean_teacher_update_error: (teacher_error_sum / teacher_error_rows.max(1) as f64)
                as f32,
            maximum_particle_state_absolute,
            maximum_closure_mode_absolute,
            teacher_update_p99_absolute,
            maximum_teacher_update_absolute,
            closure_target_p99_absolute,
            maximum_closure_target_absolute,
            closure_basis_target_p99_absolute,
            maximum_closure_basis_target_absolute,
            rollout_maximum_particle_state_absolute,
            rollout_peak_particle_state_step,
            rollout_maximum_teacher_update_absolute,
            rollout_peak_teacher_update_step,
            rollout_maximum_closure_target_absolute,
            rollout_peak_closure_target_step,
            generation_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        },
    };
    batch.validate(input_dims, output_dims)?;
    Ok(batch)
}

fn validate(
    teacher: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<()> {
    teacher.validate()?;
    grid.validate().map_err(AutomataError::from)?;
    model.validate()?;
    if teacher.config != model.rule.config
        || config.on_policy_rollouts == 0
        || config.on_policy_cut_steps.is_empty()
        || config
            .on_policy_cut_steps
            .iter()
            .any(|step| u32::try_from(*step).is_err())
        || config.on_policy_rollout_steps == 0
        || config.on_policy_snapshot_interval == 0
        || config.on_policy_rows_per_snapshot == 0
        || (model.config.closure_recurrent_mode
            && config.on_policy_teacher != AdaptiveReplayTeacher::CoupledFine)
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive multiscale on-policy configuration".to_string(),
        ));
    }
    Ok(())
}

fn rollout_seed(seed: u64, round: usize, rollout_index: usize) -> u64 {
    if rollout_index == 0 {
        seed
    } else {
        seed.wrapping_add((round as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03))
            .wrapping_add((rollout_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NpaConfig, adaptive::AdaptiveNpaConfig};

    #[test]
    fn markov_quadrature_labels_are_deterministic_and_shaped() {
        let teacher = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 7);
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.min_leaves = 16;
        adaptive.initial_leaves = 16;
        adaptive.target_leaves = 16;
        adaptive.max_leaves = 64;
        adaptive.bootstrap_fine_leaves = 64;
        adaptive.bootstrap_end_step = 1;
        let model = AdaptiveNpaModel::seeded(teacher.clone(), adaptive, 11).unwrap();
        let particles = seed_adaptive_particles_scaled(
            &model,
            16,
            13,
            ParticleSeed::UniformCircle,
            0.2,
            std::f32::consts::PI * 0.2_f32.powi(2),
            0.1,
        )
        .unwrap();
        assert_eq!(particles.bootstrap_templates.len(), particles.len());

        let first = label_snapshots(
            &teacher,
            &model,
            &AdaptiveMultiscaleTrainingConfig::default(),
            AdaptiveReplayTeacher::MarkovQuadrature,
            vec![particles.clone()],
        )
        .unwrap();
        let second = label_snapshots(
            &teacher,
            &model,
            &AdaptiveMultiscaleTrainingConfig::default(),
            AdaptiveReplayTeacher::MarkovQuadrature,
            vec![particles],
        )
        .unwrap();
        let first_update = first[0].teacher_update.as_ref().unwrap();
        assert_eq!(first_update.len(), 16 * teacher.config.update_dims());
        assert_eq!(first_update, second[0].teacher_update.as_ref().unwrap());
    }
}
