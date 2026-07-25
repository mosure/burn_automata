use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[cfg(feature = "gpu_wgpu")]
use super::gap_decomposition::run_gap_decomposition_wgpu;
#[cfg(feature = "gpu_wgpu")]
use super::task_wgpu::{
    run_fixed_rollout_wgpu, run_fixed_rollouts_wgpu, run_task_quality_rollout_wgpu,
    run_task_quality_rollouts_wgpu,
};
use super::{
    AdaptiveBaseTrainingPhaseReport, AdaptiveBaseTrainingReport, AdaptiveClosureAuditConfig,
    AdaptiveControllerValidationReport, AdaptiveDynamicsSemantics, AdaptiveExperimentConfig,
    AdaptiveExperimentGates, AdaptiveExperimentReport, AdaptiveGraphExperimentRow,
    AdaptiveMultiscaleExperimentReport, AdaptiveOperatorExperimentReport,
    AdaptiveRestrictionExperimentReport, AdaptiveRolloutExperimentReport,
    AdaptiveTaskQualityReport, AdaptiveTaskQualityValidationReport, AdaptiveTopologyAuditConfig,
    AdaptiveTopologyAuditReport, AdaptiveTopologyExperimentReport, AdaptiveTrainingBackend,
    density::task_density_alignment, graph::run_graph_experiment,
    operator::run_operator_experiment, scaling::run_scaling_experiment,
    task_cut::run_task_quality_rollout, topology::run_topology_experiment,
};
use crate::adaptive::{
    AdaptiveHierarchyRestrictionPolicy, AdaptiveModelArtifact, AdaptiveNpaModel,
    AdaptiveRenderDecoder, AdaptiveReplayBackend, AdaptiveTopologyControl,
    adaptive_deployment_on_policy_batch_wgpu, adaptive_multiscale_training_batch,
    adaptive_oracle_training_batch, adaptive_restriction_training_batch,
    adaptive_rule_distillation_batch, adaptive_rule_on_policy_batch,
    audit_adaptive_closure_identifiability, audit_adaptive_closure_identifiability_wgpu,
    load_adaptive_model, run_adaptive_rollout, save_adaptive_model,
    validate_adaptive_restriction_selection,
};
use crate::import::{BpkModelManifest, load_manifest, save_manifest};
use crate::rollout::{RolloutConfig, run_rollout_with_stable_material_masks};
use crate::{
    AutomataError, AutomataResult, NpaConfig, NpaModel, ParticleSeed, Target2dGpuBackend,
    Target2dGpuCheckpointConfig, load_target_image_2d_upstream, train_target_2d_gpu,
};
use burn_automata_kernels::HashGridConfig;

#[cfg(feature = "gpu_wgpu")]
type RestrictionRolloutExecutor = crate::gpu::WgpuAutomataExecutor;
#[cfg(not(feature = "gpu_wgpu"))]
struct RestrictionRolloutExecutor;

pub fn run_adaptive_topology_audit(
    config: &AdaptiveTopologyAuditConfig,
) -> AutomataResult<AdaptiveTopologyAuditReport> {
    Ok(AdaptiveTopologyAuditReport {
        schema_version: 1,
        seed: config.seed,
        topology_config: config.topology,
        topology: run_topology_experiment(config.topology, config.seed)?,
    })
}

pub fn run_adaptive_closure_audit(
    config: &AdaptiveClosureAuditConfig,
) -> AutomataResult<crate::adaptive::AdaptiveClosureIdentifiabilityReport> {
    let manifest = load_manifest(&config.base_model)?;
    let grid = manifest.hashgrid.clone();
    let teacher = manifest.into_model();
    match config.backend {
        crate::adaptive::AdaptiveClosureAuditBackend::CpuReference => {
            audit_adaptive_closure_identifiability(&teacher, &grid, &config.adaptive, &config.audit)
        }
        crate::adaptive::AdaptiveClosureAuditBackend::WgpuResident => {
            audit_adaptive_closure_identifiability_wgpu(
                &teacher,
                &grid,
                &config.adaptive,
                &config.audit,
            )
        }
    }
}

pub fn run_adaptive_experiment_suite(
    config: &AdaptiveExperimentConfig,
) -> AutomataResult<AdaptiveExperimentReport> {
    let suite_started = Instant::now();
    config.adaptive.validate()?;
    config.rollout.rollout.validate()?;
    validate_experiment_alignment(config)?;
    let initial_base_model_source = config.base_model.as_ref().map_or_else(
        || "deterministic-seeded-rule".to_string(),
        |path| path.display().to_string(),
    );
    let (mut rule, rule_grid) = if let Some(path) = &config.base_model {
        let manifest = load_manifest(path)?;
        let grid = manifest.hashgrid.clone();
        (manifest.into_model(), grid)
    } else {
        let npa_config = if config.adaptive.spatial_dims == 2 {
            NpaConfig::growing_2d()
        } else {
            NpaConfig::growing_3dgs()
        };
        let grid = if config.adaptive.spatial_dims == 2 {
            HashGridConfig::growing_2d()
        } else {
            HashGridConfig::growing_3dgs()
        };
        (NpaModel::upstream_seeded(npa_config, config.seed), grid)
    };
    let base_training = train_fresh_base(config, &mut rule, &rule_grid)?;
    let base_model_source = if base_training.is_some() {
        format!(
            "fresh deterministic seed trained by {} Target2D phases",
            config.base_training.phases.len()
        )
    } else {
        initial_base_model_source
    };
    let teacher_rule = rule.clone();
    let full_normalized_training = config.multiscale_training.enabled
        && config.multiscale_training.rule_strategy
            == crate::adaptive::AdaptiveMultiscaleRuleStrategy::FullNormalized;
    let mut model = if let Some(path) = &config.adaptive_checkpoint {
        let mut checkpoint = load_adaptive_model(path)?.model;
        if !full_normalized_training && !same_npa_rule(&checkpoint.rule, &rule) {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive checkpoint {} does not contain the configured frozen base rule",
                path.display()
            )));
        }
        checkpoint.config = config.adaptive.clone();
        if checkpoint.config.closure_recurrent_mode {
            // This is also the function-preserving schema migration for
            // checkpoints whose recurrent rules predate transported closure
            // context. Existing input columns and trained outputs are retained.
            checkpoint.enable_zero_closure_mode_rule()?;
        } else if !checkpoint.config.closure_recurrent_mode {
            checkpoint.closure_mode_rule = None;
            checkpoint.closure_basis_rule = None;
        }
        if checkpoint.config.compatible_residual_material_features {
            checkpoint.enable_material_conditioned_compatible_residual_rule()?;
        }
        if checkpoint.config.hierarchical_restriction_policy
            == AdaptiveHierarchyRestrictionPolicy::LearnedController
            && checkpoint
                .restriction_controller
                .as_ref()
                .is_none_or(|controller| {
                    controller.hidden_dims != checkpoint.config.controller_hidden_dims
                })
        {
            checkpoint.enable_seeded_restriction_controller(config.seed ^ 0x7265_7374_7269_6374)?;
        }
        checkpoint.validate()?;
        checkpoint
    } else {
        AdaptiveNpaModel::seeded(rule, config.adaptive.clone(), config.seed ^ 0xa4a4)?
    };
    let local_rule_training = config.multiscale_training.enabled
        && matches!(
            config.multiscale_training.rule_strategy,
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::Residual
                | crate::adaptive::AdaptiveMultiscaleRuleStrategy::CoarseReplacement
        );
    if local_rule_training && model.local_residual_rule.is_none() {
        match config.multiscale_training.rule_strategy {
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::Residual => {
                model.enable_zero_local_residual_rule()?;
            }
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::CoarseReplacement => {
                model.enable_base_initialized_local_rule()?;
            }
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::FullNormalized => unreachable!(),
        }
    }
    if local_rule_training {
        let base_hidden = model.rule.config.hidden_dims;
        let local_hidden = config
            .multiscale_training
            .resolved_local_residual_hidden_dims(base_hidden);
        if local_hidden < base_hidden || base_hidden + local_hidden + 1 > 320 {
            return Err(AutomataError::InvalidArgument(format!(
                "adaptive local rule hidden width {local_hidden} must be >= {base_hidden} and keep packed width <= 320",
            )));
        }
        model.expand_local_residual_rule(local_hidden, config.seed ^ 0x10ca_1eaf)?;
    }

    let closure_identifiability = config
        .closure_identifiability
        .enabled
        .then(|| {
            audit_adaptive_closure_identifiability(
                &teacher_rule,
                &rule_grid,
                &config.adaptive,
                &config.closure_identifiability,
            )
        })
        .transpose()?;

    let gpu_rollout_executor = create_rollout_executor(
        config,
        config.restriction_training.enabled
            || config.task_quality.enabled
            || (config.multiscale_training.enabled
                && (config.multiscale_training.on_policy_replay_backend
                    == AdaptiveReplayBackend::WgpuResident
                    || config.multiscale_training.deployment_replay_backend
                        == AdaptiveReplayBackend::WgpuResident)),
        config.multiscale_training.enabled
            && config.multiscale_training.on_policy_replay_backend
                == AdaptiveReplayBackend::WgpuResident,
    )?;

    let (training, validation, multiscale_training) = if config.multiscale_training.enabled {
        let training_batch = multiscale_training_batch(
            &teacher_rule,
            &rule_grid,
            &config.adaptive,
            &config.multiscale_training,
            gpu_rollout_executor.as_ref(),
        )?;
        let mut validation_config = config.multiscale_training.clone();
        validation_config.seed ^= 0xd157_111a_7100;
        validation_config.rollouts = validation_config.validation_rollouts;
        let validation_batch = multiscale_training_batch(
            &teacher_rule,
            &rule_grid,
            &config.adaptive,
            &validation_config,
            gpu_rollout_executor.as_ref(),
        )?;
        let freeze_multiscale_rule = config.multiscale_training.freeze_training_policy
            || config.multiscale_training.freeze_multiscale_rule;
        let freeze_local_residual_rule =
            freeze_multiscale_rule || config.multiscale_training.freeze_local_residual_rule;
        let freeze_closure_rule =
            freeze_multiscale_rule || config.multiscale_training.freeze_closure_rule;
        let freeze_controller = config.multiscale_training.freeze_training_policy
            || config.multiscale_training.freeze_controller;
        if freeze_controller && config.adaptive_checkpoint.is_none() {
            return Err(AutomataError::InvalidArgument(
                "freezing the adaptive controller requires adaptive_checkpoint".to_string(),
            ));
        }
        let on_policy_only_rule_training =
            config.multiscale_training.multiscale_on_policy_only_replay
                && config.multiscale_training.on_policy_rounds > 0;
        let rule_training = if freeze_local_residual_rule || on_policy_only_rule_training {
            crate::adaptive::AdaptiveMultiscaleRuleTrainingReport::default()
        } else {
            train_multiscale_rule(
                config.backend,
                &mut model,
                &training_batch,
                &config.multiscale_training,
            )?
        };
        let closure_training = if config.adaptive.closure_recurrent_mode
            && !freeze_closure_rule
            && !on_policy_only_rule_training
        {
            train_closure_mode(
                config.backend,
                &mut model,
                &training_batch,
                &config.multiscale_training,
            )?
        } else {
            crate::adaptive::AdaptiveClosureModeTrainingReport::default()
        };
        let controller_training_batch = crate::adaptive::AdaptiveControllerTrainingBatch {
            features: training_batch.controller_features.clone(),
            targets: training_batch.controller_targets.clone(),
            rows: training_batch.rows,
        };
        let controller_validation_batch = crate::adaptive::AdaptiveControllerTrainingBatch {
            features: validation_batch.controller_features.clone(),
            targets: validation_batch.controller_targets.clone(),
            rows: validation_batch.rows,
        };
        let controller_training_config = crate::adaptive::AdaptiveControllerTrainConfig {
            enabled: true,
            steps: config.multiscale_training.controller_steps,
            report_interval: config.multiscale_training.report_interval,
            gradient_reduction_chunk_rows: config.multiscale_training.gradient_reduction_chunk_rows,
            optimizer_batch_rows: 0,
            restriction_rank_boundary_emphasis: 0.0,
            restriction_rank_boundary_width: 0.125,
            restriction_topk_loss_weight: 0.0,
            restriction_topk_temperature: 0.25,
            restriction_cost_utility_weight: 0.0,
            optimizer: config.multiscale_training.controller_optimizer,
        };
        let controller_training = if freeze_controller {
            crate::adaptive::AdaptiveControllerTrainingReport::default()
        } else {
            train_controller(
                config.backend,
                &mut model.controller,
                &controller_training_batch,
                controller_training_config,
            )?
        };
        let training_dataset = training_batch.report.clone();
        let mut on_policy_datasets =
            Vec::with_capacity(config.multiscale_training.on_policy_rounds);
        let mut on_policy_training =
            Vec::with_capacity(config.multiscale_training.on_policy_rounds);
        let mut on_policy_closure_training =
            Vec::with_capacity(config.multiscale_training.on_policy_rounds);
        let mut on_policy_controller_training =
            Vec::with_capacity(config.multiscale_training.on_policy_rounds);
        let mut deployment_replay = training_batch;
        let mut controller_replay = controller_training_batch;
        let exact_on_policy_rounds = config.multiscale_training.on_policy_rounds;
        for round in 0..exact_on_policy_rounds {
            let on_policy = multiscale_on_policy_batch(
                &teacher_rule,
                &rule_grid,
                &model,
                &config.multiscale_training,
                round,
                gpu_rollout_executor.as_ref(),
            )?;
            on_policy_datasets.push(on_policy.report.clone());
            update_multiscale_replay(
                &mut deployment_replay,
                &on_policy,
                config.multiscale_training.multiscale_on_policy_only_replay && round == 0,
            )?;
            if !freeze_controller {
                let on_policy_controller = crate::adaptive::AdaptiveControllerTrainingBatch {
                    features: on_policy.controller_features.clone(),
                    targets: on_policy.controller_targets.clone(),
                    rows: on_policy.rows,
                };
                update_controller_replay(
                    &mut controller_replay,
                    &on_policy_controller,
                    config.multiscale_training.controller_on_policy_only_replay && round == 0,
                );
                on_policy_controller_training.push(train_controller(
                    config.backend,
                    &mut model.controller,
                    &controller_replay,
                    controller_training_config,
                )?);
            }
            if !freeze_multiscale_rule && config.multiscale_training.on_policy_steps > 0 {
                let mut round_config = config.multiscale_training.clone();
                round_config.steps = round_config.on_policy_steps;
                if !freeze_local_residual_rule {
                    on_policy_training.push(train_multiscale_rule(
                        config.backend,
                        &mut model,
                        &deployment_replay,
                        &round_config,
                    )?);
                }
                if config.adaptive.closure_recurrent_mode && !freeze_closure_rule {
                    on_policy_closure_training.push(train_closure_mode(
                        config.backend,
                        &mut model,
                        &deployment_replay,
                        &round_config,
                    )?);
                }
            }
        }
        let deployment_training = if config.multiscale_training.deployment_enabled {
            train_deployment_rule(
                config.backend,
                &mut model,
                &deployment_replay,
                &config.multiscale_training,
            )?
        } else {
            crate::adaptive::AdaptiveDeploymentRuleTrainingReport::default()
        };
        let mut deployment_on_policy_datasets =
            Vec::with_capacity(config.multiscale_training.deployment_on_policy_rounds);
        let mut deployment_on_policy_training =
            Vec::with_capacity(config.multiscale_training.deployment_on_policy_rounds);
        for round in 0..config.multiscale_training.deployment_on_policy_rounds {
            let replay_round = exact_on_policy_rounds + round + 1_000;
            let on_policy = if config.multiscale_training.deployment_replay_backend
                == AdaptiveReplayBackend::WgpuResident
            {
                adaptive_deployment_on_policy_batch_wgpu(
                    &model,
                    &rule_grid,
                    &config.multiscale_training,
                    replay_round,
                )?
            } else {
                crate::adaptive::adaptive_multiscale_on_policy_batch(
                    &teacher_rule,
                    &rule_grid,
                    &model,
                    &config.multiscale_training,
                    replay_round,
                )?
            };
            deployment_on_policy_datasets.push(on_policy.report.clone());
            update_multiscale_replay(
                &mut deployment_replay,
                &on_policy,
                config.multiscale_training.deployment_on_policy_only_replay && round == 0,
            )?;
            let mut round_config = config.multiscale_training.clone();
            round_config.deployment_steps = round_config.resolved_deployment_on_policy_steps();
            deployment_on_policy_training.push(train_deployment_rule(
                config.backend,
                &mut model,
                &deployment_replay,
                &round_config,
            )?);
        }
        let heldout_on_policy = if exact_on_policy_rounds > 0 {
            let mut heldout_config = config.multiscale_training.clone();
            heldout_config.seed ^= 0x6865_6c64_6f75_745f;
            heldout_config.on_policy_rollouts = heldout_config.validation_rollouts.max(1);
            Some(multiscale_on_policy_batch(
                &teacher_rule,
                &rule_grid,
                &model,
                &heldout_config,
                exact_on_policy_rounds
                    + config.multiscale_training.deployment_on_policy_rounds
                    + 10_000,
                gpu_rollout_executor.as_ref(),
            )?)
        } else {
            None
        };
        let heldout_on_policy_validation = heldout_on_policy
            .as_ref()
            .map(|batch| validate_multiscale_rule(config.backend, &model, batch))
            .transpose()?;
        let heldout_on_policy_closure_validation = if config.adaptive.closure_recurrent_mode {
            heldout_on_policy
                .as_ref()
                .map(|batch| crate::adaptive::adaptive_closure_mode_validation(&model, batch))
                .transpose()?
        } else {
            None
        };
        let heldout_on_policy_dataset = heldout_on_policy.map(|batch| batch.report);
        let controller_validation =
            validate_controller(config.backend, &model, &controller_validation_batch)?;
        let heldout_validation =
            validate_multiscale_rule(config.backend, &model, &validation_batch)?;
        let heldout_closure_validation = if config.adaptive.closure_recurrent_mode {
            crate::adaptive::adaptive_closure_mode_validation(&model, &validation_batch)?
        } else {
            crate::adaptive::AdaptiveClosureModeValidationReport::default()
        };
        let heldout_deployment_validation = if config.multiscale_training.deployment_enabled {
            crate::adaptive::adaptive_deployment_rule_validation(
                &model,
                &validation_batch,
                &config.multiscale_training,
            )?
        } else {
            crate::adaptive::AdaptiveDeploymentRuleValidationReport::default()
        };
        let report = AdaptiveMultiscaleExperimentReport {
            training_dataset,
            validation_dataset: validation_batch.report,
            training: rule_training,
            closure_training,
            on_policy_datasets,
            on_policy_training,
            on_policy_closure_training,
            on_policy_controller_training,
            heldout_validation,
            heldout_closure_validation,
            heldout_on_policy_dataset,
            heldout_on_policy_validation,
            heldout_on_policy_closure_validation,
            deployment_training,
            deployment_on_policy_datasets,
            deployment_on_policy_training,
            heldout_deployment_validation,
        };
        (controller_training, controller_validation, Some(report))
    } else {
        let training_batch = adaptive_oracle_training_batch(config.training_data)?;
        let training = if config.training.enabled {
            train_controller(
                config.backend,
                &mut model.controller,
                &training_batch,
                config.training,
            )?
        } else {
            crate::adaptive::AdaptiveControllerTrainingReport::default()
        };
        let validation_batch =
            adaptive_oracle_training_batch(crate::adaptive::AdaptiveOracleDatasetConfig {
                seed: config.training_data.seed ^ 0x9e37_79b9,
                rows: config.training_data.rows.clamp(256, 16_384),
                ..config.training_data
            })?;
        let validation = validate_controller(config.backend, &model, &validation_batch)?;
        (training, validation, None)
    };
    let rule_distillation = if config.rule_distillation.enabled {
        let validation_config = crate::adaptive::AdaptiveRuleDistillationConfig {
            seed: config.rule_distillation.seed ^ 0xd157_111a_7100,
            rollouts: config.rule_distillation.validation_rollouts,
            ..config.rule_distillation
        };
        let rule_validation_batch = adaptive_rule_distillation_batch(
            &teacher_rule,
            &rule_grid,
            &config.adaptive,
            validation_config,
        )?;
        let source_validation = validate_rule(&model.rule, &rule_validation_batch)?;
        let rule_training_batch = adaptive_rule_distillation_batch(
            &teacher_rule,
            &rule_grid,
            &config.adaptive,
            config.rule_distillation,
        )?;
        let training = train_rule(
            config.backend,
            &mut model.rule,
            &rule_training_batch,
            config.rule_distillation,
        )?;
        let offline_validation = validate_rule(&model.rule, &rule_validation_batch)?;
        let mut on_policy_training = Vec::with_capacity(config.rule_distillation.on_policy_rounds);
        for round in 0..config.rule_distillation.on_policy_rounds {
            let round_config = crate::adaptive::AdaptiveRuleDistillationConfig {
                seed: config
                    .rule_distillation
                    .seed
                    .wrapping_add((round as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
                steps: config.rule_distillation.on_policy_steps,
                ..config.rule_distillation
            };
            let on_policy = adaptive_rule_on_policy_batch(
                &teacher_rule,
                &model.rule,
                &rule_grid,
                &config.adaptive,
                round_config,
            )?;
            let replay = concatenate_rule_batches(&rule_training_batch, &on_policy);
            on_policy_training.push(train_rule(
                config.backend,
                &mut model.rule,
                &replay,
                round_config,
            )?);
        }
        let trained_validation = validate_rule(&model.rule, &rule_validation_batch)?;
        Some(crate::adaptive::AdaptiveRuleDistillationReport {
            source_validation,
            offline_validation,
            trained_validation,
            training,
            on_policy_training,
        })
    } else {
        None
    };

    let restriction_training = if config.restriction_training.enabled {
        crate::adaptive::validate_adaptive_restriction_training_memory_plan(
            &model,
            &config.restriction_training.training_data,
            config.restriction_training.training,
        )?;
        eprintln!(
            "adaptive restriction: generating {} train and {} validation trajectory snapshots on {:?}",
            config.restriction_training.training_data.seeds.len()
                * config.restriction_training.training_data.cut_steps.len(),
            config.restriction_training.validation_data.seeds.len()
                * config.restriction_training.validation_data.cut_steps.len(),
            config.backend,
        );
        let target = crate::target2d::load_target_image_2d_upstream(
            &config.task_quality.target_image,
            0.05,
            config.multiscale_training.fine_particle_count,
            None,
        )?;
        let render_config = crate::target2d::Target2dLossConfig {
            image_size: config.task_quality.image_size,
            ..crate::target2d::Target2dLossConfig::default()
        };
        let (training_batch, training_dataset) = restriction_training_batch(
            config.backend,
            gpu_rollout_executor.as_ref(),
            &model,
            &rule_grid,
            &target,
            render_config,
            &config.restriction_training.training_data,
        )?;
        eprintln!(
            "adaptive restriction: train labels backend={} rows={} generation_ms={:.2}",
            training_dataset.label_backend, training_dataset.rows, training_dataset.generation_ms,
        );
        let (validation_batch, validation_dataset) = restriction_training_batch(
            config.backend,
            gpu_rollout_executor.as_ref(),
            &model,
            &rule_grid,
            &target,
            render_config,
            &config.restriction_training.validation_data,
        )?;
        eprintln!(
            "adaptive restriction: validation labels backend={} rows={} generation_ms={:.2}",
            validation_dataset.label_backend,
            validation_dataset.rows,
            validation_dataset.generation_ms,
        );
        let controller = model.restriction_controller.as_mut().ok_or_else(|| {
            AutomataError::InvalidModel(
                "restriction training requires restriction_controller".to_string(),
            )
        })?;
        let training = train_restriction_controller(
            config.backend,
            controller,
            &training_batch,
            &validation_batch,
            config.restriction_training.training,
        )?;
        let training_selection =
            validate_adaptive_restriction_selection(controller, &training_batch)?;
        let heldout_selection =
            validate_adaptive_restriction_selection(controller, &validation_batch)?;
        Some(AdaptiveRestrictionExperimentReport {
            training_dataset,
            validation_dataset,
            training,
            training_selection,
            heldout_selection,
        })
    } else {
        None
    };

    let operator = run_operator_experiment(config.operator, 2)?;
    let operator_3d = run_operator_experiment(config.operator, 3)?;
    let topology = run_topology_experiment(config.topology, config.seed)?;
    let graph = run_graph_experiment(&config.graph, config.seed)?;
    let scaling = run_scaling_experiment(&config.scaling, config.seed)?;
    let rollout = if config.rollout.enabled {
        Some(run_rollout_experiment(&model, config)?)
    } else {
        None
    };
    let (task_quality, task_quality_validation) = if config.task_quality.enabled {
        let reference_model = task_reference_model(config)?;
        let reference_manifest = load_manifest(reference_model)?;
        let reference_grid = reference_manifest.hashgrid.clone();
        let reference_rule = reference_manifest.into_model();
        let primary = run_task_quality_experiment(
            &reference_rule,
            &teacher_rule,
            &reference_grid,
            &model,
            config,
            gpu_rollout_executor.as_ref(),
        )?;
        let validation = run_task_quality_validation(
            &reference_rule,
            &teacher_rule,
            &reference_grid,
            &model,
            config,
            &primary,
            gpu_rollout_executor.as_ref(),
        )?;
        (Some(primary), validation)
    } else {
        (None, None)
    };
    let gate_failures = validate_experiment_gates(
        config.gates,
        base_training.as_ref(),
        [&operator, &operator_3d],
        &topology,
        &graph,
        &scaling,
        &training,
        &validation,
        rule_distillation.as_ref(),
        multiscale_training.as_ref(),
        restriction_training.as_ref(),
        !config.multiscale_training.freeze_training_policy
            && !config.multiscale_training.freeze_multiscale_rule
            && !config.multiscale_training.freeze_local_residual_rule
            && config.adaptive.local_residual_scale > 0.0,
        task_quality.as_ref(),
        task_quality_validation.as_ref(),
        rollout.as_ref(),
    );
    let gates_passed = gate_failures.is_empty();

    let source = Some(format!("budgeted-adaptive-npa from {base_model_source}"));
    let artifact = if base_training.is_some() && multiscale_training.is_some() {
        AdaptiveModelArtifact::fresh_task_trained(model, source)?
    } else if multiscale_training.is_some() {
        AdaptiveModelArtifact::task_trained(model, source)?
    } else {
        AdaptiveModelArtifact::new(model, source)?
    };
    let model_output = if gates_passed {
        config.model_output.clone()
    } else {
        failed_output_path(&config.model_output)
    };
    let model_sha256 = save_adaptive_model(&model_output, &artifact)?;
    let report = AdaptiveExperimentReport {
        schema_version: 19,
        gates_passed,
        gate_failures: gate_failures.clone(),
        paper_scope: paper_scope(
            config,
            rule_distillation.is_some(),
            multiscale_training.is_some(),
        ),
        base_model_source,
        base_training,
        rule_perception: config.adaptive.rule_perception,
        operator,
        operator_3d,
        topology,
        graph,
        scaling,
        training,
        validation,
        rule_distillation,
        multiscale_training,
        closure_identifiability,
        restriction_training,
        task_quality,
        task_quality_validation,
        rollout,
        model_output: model_output.display().to_string(),
        model_sha256,
        total_elapsed_ms: suite_started.elapsed().as_secs_f64() * 1_000.0,
    };
    let report_output = if gates_passed {
        config.report_output.clone()
    } else {
        failed_output_path(&config.report_output)
    };
    if let Some(parent) = report_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&report_output, serde_json::to_vec_pretty(&report)?)?;
    if !gates_passed {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive experiment gates failed: {}; diagnostic model {}; report {}",
            gate_failures.join("; "),
            model_output.display(),
            report_output.display(),
        )));
    }
    Ok(report)
}

fn train_fresh_base(
    config: &AdaptiveExperimentConfig,
    rule: &mut NpaModel,
    rule_grid: &HashGridConfig,
) -> AutomataResult<Option<AdaptiveBaseTrainingReport>> {
    let base = &config.base_training;
    if !base.enabled {
        return Ok(None);
    }
    let target = load_target_image_2d_upstream(
        &base.target_image,
        base.target_threshold,
        base.target_points,
        base.target_image_size,
    )?;
    let backend = match config.backend {
        AdaptiveTrainingBackend::Wgpu => Target2dGpuBackend::Wgpu,
        AdaptiveTrainingBackend::Cuda => Target2dGpuBackend::Cuda,
        AdaptiveTrainingBackend::NdArray => {
            return Err(AutomataError::InvalidArgument(
                "fresh adaptive base training requires backend=wgpu or backend=cuda".to_string(),
            ));
        }
    };
    let mut phases = Vec::with_capacity(base.phases.len());
    for phase in &base.phases {
        let checkpoint = base.checkpoint_root.as_ref().map(|root| {
            let phase_root = root.join(checkpoint_phase_slug(&phase.name));
            Target2dGpuCheckpointConfig {
                current_model_output: phase_root.join("current.bpk"),
                best_model_output: phase_root.join("best.bpk"),
                metadata_output: phase_root.join("metadata.json"),
                training_state_output: None,
                resume_training_state: None,
                resume_model_sha256: None,
                curriculum_resume: false,
                include_particle_pool: false,
                source: format!("fresh adaptive base phase {}", phase.name),
                interval_steps: base.checkpoint_interval_steps,
                interval_duration: (base.checkpoint_interval_seconds > 0)
                    .then(|| Duration::from_secs(base.checkpoint_interval_seconds)),
            }
        });
        let training = train_target_2d_gpu(
            backend,
            rule,
            rule_grid,
            target.clone(),
            phase.training.clone(),
            phase.loss.unwrap_or(base.loss),
            checkpoint.as_ref(),
        )
        .map_err(|error| {
            AutomataError::InvalidArgument(format!(
                "fresh adaptive base phase `{}` failed: {error}",
                phase.name
            ))
        })?;
        phases.push(AdaptiveBaseTrainingPhaseReport {
            name: phase.name.clone(),
            particle_count: phase.training.particle_count,
            training,
        });
    }
    if let Some(path) = &base.model_output {
        save_manifest(
            path,
            &BpkModelManifest::from_model(
                rule,
                rule_grid.clone(),
                Some("fresh multiscale adaptive Target2D base".to_string()),
            ),
        )?;
    }
    Ok(Some(AdaptiveBaseTrainingReport {
        initializer: "deterministic-upstream-compatible-random".to_string(),
        target_image: base.target_image.display().to_string(),
        reference_model: base
            .reference_model
            .as_ref()
            .map(|path| path.display().to_string()),
        target_points: target.point_count(),
        phases,
        model_output: base
            .model_output
            .as_ref()
            .map(|path| path.display().to_string()),
    }))
}

fn checkpoint_phase_slug(name: &str) -> String {
    let slug = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if slug.is_empty() {
        "phase".to_string()
    } else {
        slug
    }
}

fn task_reference_model(config: &AdaptiveExperimentConfig) -> AutomataResult<&Path> {
    config
        .task_quality
        .reference_model
        .as_deref()
        .or(config.base_training.reference_model.as_deref())
        .or(config.base_model.as_deref())
        .ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive task evaluation requires task_quality.reference_model, base_training.reference_model, or base_model"
                    .to_string(),
            )
        })
}

fn evaluation_regular_base(
    config: &AdaptiveExperimentConfig,
    model: &AdaptiveNpaModel,
    reference: &NpaModel,
) -> AutomataResult<NpaModel> {
    if let Some(path) = &config.base_model {
        let base = load_manifest(path)?.into_model();
        if base.config == model.rule.config {
            return Ok(base);
        }
        if adaptive_rule_matches_reference_architecture(model, &base)? {
            return crate::adaptive::model::input_expanded_compatible_rule(
                &base,
                model.rule.config.auxiliary_input_dims,
            );
        }
        return Err(AutomataError::InvalidModel(format!(
            "configured regular base {} and adaptive artifact use different NPA architectures",
            path.display(),
        )));
    }
    if model.config.rule_perception == crate::adaptive::AdaptiveRulePerception::NpaCompatible {
        return Ok(model.rule.clone());
    }
    if same_npa_rule(&model.rule, reference) {
        return Ok(reference.clone());
    }
    Err(AutomataError::InvalidArgument(
        "evaluating a trained normalized-adaptive rule requires base_model so the immutable regular NPA baseline is not confused with the trained adaptive rule"
            .to_string(),
    ))
}

fn multiscale_training_batch(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    adaptive: &crate::adaptive::AdaptiveNpaConfig,
    config: &crate::adaptive::AdaptiveMultiscaleTrainingConfig,
    executor: Option<&RestrictionRolloutExecutor>,
) -> AutomataResult<crate::adaptive::AdaptiveMultiscaleTrainingBatch> {
    if config.on_policy_replay_backend == AdaptiveReplayBackend::WgpuResident {
        #[cfg(feature = "gpu_wgpu")]
        {
            let executor = executor.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "resident multiscale teacher collection requires a WGPU executor".to_string(),
                )
            })?;
            return crate::adaptive::adaptive_multiscale_training_batch_wgpu_with_executor(
                executor,
                teacher,
                teacher_grid,
                adaptive,
                config,
            );
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            let _ = executor;
            return Err(AutomataError::InvalidArgument(
                "resident multiscale teacher collection requires gpu_wgpu".to_string(),
            ));
        }
    }
    adaptive_multiscale_training_batch(teacher, teacher_grid, adaptive, config)
}

fn multiscale_on_policy_batch(
    teacher: &NpaModel,
    teacher_grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &crate::adaptive::AdaptiveMultiscaleTrainingConfig,
    round: usize,
    executor: Option<&RestrictionRolloutExecutor>,
) -> AutomataResult<crate::adaptive::AdaptiveMultiscaleTrainingBatch> {
    if config.on_policy_replay_backend == AdaptiveReplayBackend::WgpuResident {
        #[cfg(feature = "gpu_wgpu")]
        {
            let executor = executor.ok_or_else(|| {
                AutomataError::InvalidArgument(
                    "resident multiscale replay requires a WGPU executor".to_string(),
                )
            })?;
            return crate::adaptive::adaptive_multiscale_on_policy_batch_wgpu_with_executor(
                executor,
                teacher,
                teacher_grid,
                model,
                config,
                round,
            );
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            let _ = executor;
            return Err(AutomataError::InvalidArgument(
                "resident multiscale replay requires gpu_wgpu".to_string(),
            ));
        }
    }
    crate::adaptive::adaptive_multiscale_on_policy_batch(
        teacher,
        teacher_grid,
        model,
        config,
        round,
    )
}

fn create_rollout_executor(
    config: &AdaptiveExperimentConfig,
    needed: bool,
    _full_profile: bool,
) -> AutomataResult<Option<RestrictionRolloutExecutor>> {
    if !needed || config.backend == AdaptiveTrainingBackend::NdArray {
        return Ok(None);
    }
    #[cfg(feature = "gpu_wgpu")]
    {
        let started = Instant::now();
        let executor = if _full_profile {
            crate::gpu::WgpuAutomataExecutor::new_blocking()?
        } else {
            crate::gpu::WgpuAutomataExecutor::new_restriction_blocking()?
        };
        eprintln!(
            "adaptive GPU rollout: {} WGPU executor initialization_ms={:.2}",
            if _full_profile { "full" } else { "minimal" },
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        Ok(Some(executor))
    }
    #[cfg(not(feature = "gpu_wgpu"))]
    {
        Err(AutomataError::InvalidArgument(
            "CUDA/WGPU adaptive rollout requires the gpu_wgpu feature".to_string(),
        ))
    }
}

pub fn evaluate_adaptive_task_quality(
    trained_model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
) -> AutomataResult<AdaptiveTaskQualityReport> {
    if !config.task_quality.enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive task evaluation requires [task_quality].enabled = true".to_string(),
        ));
    }
    let manifest = load_manifest(task_reference_model(config)?)?;
    let teacher_grid = manifest.hashgrid.clone();
    let teacher = manifest.into_model();
    let mut model = trained_model.clone();
    model.config = config.adaptive.clone();
    model.validate()?;
    if !adaptive_rule_matches_reference_architecture(&model, &teacher)? {
        return Err(AutomataError::InvalidModel(
            "adaptive evaluation artifact and configured teacher use different NPA rules"
                .to_string(),
        ));
    }
    let regular_base = evaluation_regular_base(config, &model, &teacher)?;
    let executor = create_rollout_executor(config, true, false)?;
    run_task_quality_experiment(
        &teacher,
        &regular_base,
        &teacher_grid,
        &model,
        config,
        executor.as_ref(),
    )
}

pub fn evaluate_adaptive_task_quality_validation(
    trained_model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    seeds: &[u64],
) -> AutomataResult<AdaptiveTaskQualityValidationReport> {
    if !config.task_quality.enabled {
        return Err(AutomataError::InvalidArgument(
            "adaptive task evaluation requires [task_quality].enabled = true".to_string(),
        ));
    }
    let seeds = unique_seeds(seeds);
    if seeds.is_empty() {
        return Err(AutomataError::InvalidArgument(
            "adaptive parity evaluation requires at least one seed".to_string(),
        ));
    }
    let manifest = load_manifest(task_reference_model(config)?)?;
    let teacher_grid = manifest.hashgrid.clone();
    let teacher = manifest.into_model();
    let mut model = trained_model.clone();
    model.config = config.adaptive.clone();
    model.validate()?;
    if !adaptive_rule_matches_reference_architecture(&model, &teacher)? {
        return Err(AutomataError::InvalidModel(
            "adaptive evaluation artifact and configured teacher use different NPA rules"
                .to_string(),
        ));
    }
    let regular_base = evaluation_regular_base(config, &model, &teacher)?;
    let executor = create_rollout_executor(config, true, false)?;

    let mut precomputed = (0..seeds.len()).map(|_| None).collect::<Vec<_>>();
    #[cfg(feature = "gpu_wgpu")]
    if let Some(executor) = executor.as_ref() {
        precomputed = precompute_task_quality_wgpu(
            executor,
            &teacher,
            &regular_base,
            &teacher_grid,
            &model,
            config,
            &seeds,
        )?
        .into_iter()
        .map(Some)
        .collect();
    }

    let mut rows = Vec::with_capacity(seeds.len());
    for (index, seed) in seeds.iter().copied().enumerate() {
        let mut seed_config = config.clone();
        seed_config.task_quality.seed = seed;
        rows.push(run_task_quality_experiment_with_precomputed(
            TaskQualityContext {
                teacher: &teacher,
                regular_base: &regular_base,
                teacher_grid: &teacher_grid,
                model: &model,
                config: &seed_config,
                executor: executor.as_ref(),
            },
            precomputed[index].take(),
            index < config.task_quality.structural_audit_seeds,
        )?);
    }
    let mut report = summarize_task_quality_validation(rows);
    report.gap_decomposition = maybe_run_gap_decomposition(
        executor.as_ref(),
        &regular_base,
        &teacher_grid,
        &model,
        config,
        &seeds,
    )?;
    Ok(report)
}

fn run_task_quality_validation(
    reference_rule: &NpaModel,
    regular_base_rule: &NpaModel,
    reference_grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    primary: &AdaptiveTaskQualityReport,
    executor: Option<&RestrictionRolloutExecutor>,
) -> AutomataResult<Option<AdaptiveTaskQualityValidationReport>> {
    let seeds = unique_seeds(&config.task_quality.validation_seeds);
    if seeds.is_empty() {
        return Ok(None);
    }

    let mut precomputed = (0..seeds.len()).map(|_| None).collect::<Vec<_>>();
    #[cfg(feature = "gpu_wgpu")]
    if let Some(executor) = executor {
        precomputed = precompute_task_quality_wgpu(
            executor,
            reference_rule,
            regular_base_rule,
            reference_grid,
            model,
            config,
            &seeds,
        )?
        .into_iter()
        .map(Some)
        .collect();
    }

    let mut rows = Vec::with_capacity(seeds.len());
    for (index, seed) in seeds.iter().copied().enumerate() {
        if seed == primary.seed {
            rows.push(primary.clone());
            continue;
        }
        let mut seed_config = config.clone();
        seed_config.task_quality.seed = seed;
        rows.push(run_task_quality_experiment_with_precomputed(
            TaskQualityContext {
                teacher: reference_rule,
                regular_base: regular_base_rule,
                teacher_grid: reference_grid,
                model,
                config: &seed_config,
                executor,
            },
            precomputed[index].take(),
            index < config.task_quality.structural_audit_seeds,
        )?);
    }

    let mut report = summarize_task_quality_validation(rows);
    report.gap_decomposition = maybe_run_gap_decomposition(
        executor,
        regular_base_rule,
        reference_grid,
        model,
        config,
        &seeds,
    )?;
    Ok(Some(report))
}

fn maybe_run_gap_decomposition(
    executor: Option<&RestrictionRolloutExecutor>,
    regular_base: &NpaModel,
    grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    seeds: &[u64],
) -> AutomataResult<Option<super::AdaptiveGapDecompositionReport>> {
    if !config.task_quality.gap_decomposition.enabled {
        return Ok(None);
    }
    #[cfg(feature = "gpu_wgpu")]
    {
        let executor = executor.ok_or_else(|| {
            AutomataError::InvalidArgument(
                "adaptive gap decomposition requires a WGPU rollout executor".to_owned(),
            )
        })?;
        run_gap_decomposition_wgpu(executor, regular_base, grid, model, config, seeds).map(Some)
    }
    #[cfg(not(feature = "gpu_wgpu"))]
    {
        let _ = (executor, regular_base, grid, model, seeds);
        Err(AutomataError::InvalidArgument(
            "adaptive gap decomposition requires the gpu_wgpu feature".to_owned(),
        ))
    }
}

fn unique_seeds(seeds: &[u64]) -> Vec<u64> {
    let mut unique = Vec::with_capacity(seeds.len());
    for seed in seeds.iter().copied() {
        if !unique.contains(&seed) {
            unique.push(seed);
        }
    }
    unique
}

fn summarize_task_quality_validation(
    rows: Vec<AdaptiveTaskQualityReport>,
) -> AdaptiveTaskQualityValidationReport {
    debug_assert!(!rows.is_empty());
    let count = rows.len() as f32;
    let mean_adaptive = rows
        .iter()
        .map(|row| row.adaptive_target_composited_psnr_db)
        .sum::<f32>()
        / count;
    let mean_teacher = rows
        .iter()
        .map(|row| row.teacher_target_composited_psnr_db)
        .sum::<f32>()
        / count;
    let teacher_gains = rows
        .iter()
        .map(|row| row.adaptive_target_composited_psnr_db - row.teacher_target_composited_psnr_db)
        .collect::<Vec<_>>();
    let mean_regular = rows
        .iter()
        .map(|row| row.regular_base_target_composited_psnr_db)
        .sum::<f32>()
        / count;
    let regular_gains = rows
        .iter()
        .map(|row| row.adaptive_over_regular_base_psnr_gain_db)
        .collect::<Vec<_>>();
    let mean_regular_matched = rows
        .iter()
        .map(|row| row.regular_matched_budget_target_composited_psnr_db)
        .sum::<f32>()
        / count;
    let mean_regular_material_matched = rows
        .iter()
        .map(|row| row.regular_material_matched_budget_target_composited_psnr_db)
        .sum::<f32>()
        / count;
    let matched_budget_gains = rows
        .iter()
        .map(|row| row.adaptive_over_regular_matched_budget_psnr_gain_db)
        .collect::<Vec<_>>();
    let material_matched_budget_gains = rows
        .iter()
        .map(|row| row.adaptive_over_regular_material_matched_budget_psnr_gain_db)
        .collect::<Vec<_>>();
    let topology_gains = rows
        .iter()
        .map(|row| {
            row.adaptive_target_composited_psnr_db
                - row.adaptive_budget_fixed_target_composited_psnr_db
        })
        .collect::<Vec<_>>();
    let maximum_leaf_relative_error = rows
        .iter()
        .map(|row| {
            row.adaptive_final_particles
                .abs_diff(row.adaptive_target_particles.max(1)) as f64
                / row.adaptive_target_particles.max(1) as f64
        })
        .fold(0.0_f64, f64::max);
    let structural_audit_seeds = rows
        .iter()
        .filter(|row| row.structural_audit_performed)
        .count();
    let minimum_controller_correlation = rows
        .iter()
        .filter(|row| row.structural_audit_performed)
        .map(|row| row.controller_oracle_refinement_scale_correlation)
        .fold(f32::INFINITY, f32::min);
    let maximum_measure_relative_drift = rows
        .iter()
        .map(|row| row.measure_relative_drift)
        .fold(0.0_f64, f64::max);
    let minimum_final_occupied_material_scale_bins = rows
        .iter()
        .map(|row| row.final_occupied_material_scale_bins)
        .min()
        .unwrap_or_default();
    let minimum_final_fractional_material_scale_fraction = rows
        .iter()
        .map(|row| row.final_fractional_material_scale_fraction)
        .fold(f32::INFINITY, f32::min);
    let minimum_final_dyadic_scale_quantization_rmse_octaves = rows
        .iter()
        .map(|row| row.final_dyadic_scale_quantization_rmse_octaves)
        .fold(f32::INFINITY, f32::min);
    let mean_adaptive_rollout_elapsed_ms = rows
        .iter()
        .map(|row| row.adaptive_rollout_elapsed_ms)
        .sum::<f64>()
        / rows.len() as f64;
    let mean_adaptive_topology_elapsed_ms = rows
        .iter()
        .map(|row| row.adaptive_topology_elapsed_ms)
        .sum::<f64>()
        / rows.len() as f64;
    let maximum_topology_update_elapsed_ms = rows
        .iter()
        .map(|row| row.maximum_topology_update_elapsed_ms)
        .fold(0.0_f64, f64::max);

    AdaptiveTaskQualityValidationReport {
        structural_audit_seeds,
        rows,
        mean_adaptive_target_composited_psnr_db: mean_adaptive,
        mean_teacher_target_composited_psnr_db: mean_teacher,
        mean_adaptive_over_teacher_psnr_gain_db: teacher_gains.iter().sum::<f32>() / count,
        worst_adaptive_over_teacher_psnr_gain_db: teacher_gains
            .into_iter()
            .fold(f32::INFINITY, f32::min),
        mean_regular_base_target_composited_psnr_db: mean_regular,
        mean_adaptive_over_regular_base_psnr_gain_db: regular_gains.iter().sum::<f32>() / count,
        worst_adaptive_over_regular_base_psnr_gain_db: regular_gains
            .into_iter()
            .fold(f32::INFINITY, f32::min),
        mean_regular_matched_budget_target_composited_psnr_db: mean_regular_matched,
        mean_adaptive_over_regular_matched_budget_psnr_gain_db: matched_budget_gains
            .iter()
            .sum::<f32>()
            / count,
        worst_adaptive_over_regular_matched_budget_psnr_gain_db: matched_budget_gains
            .into_iter()
            .fold(f32::INFINITY, f32::min),
        mean_regular_material_matched_budget_target_composited_psnr_db:
            mean_regular_material_matched,
        mean_adaptive_over_regular_material_matched_budget_psnr_gain_db:
            material_matched_budget_gains.iter().sum::<f32>() / count,
        worst_adaptive_over_regular_material_matched_budget_psnr_gain_db:
            material_matched_budget_gains
                .into_iter()
                .fold(f32::INFINITY, f32::min),
        mean_adaptive_over_budget_fixed_psnr_gain_db: topology_gains.iter().sum::<f32>() / count,
        worst_adaptive_over_budget_fixed_psnr_gain_db: topology_gains
            .into_iter()
            .fold(f32::INFINITY, f32::min),
        minimum_controller_oracle_refinement_scale_correlation: minimum_controller_correlation,
        maximum_measure_relative_drift,
        maximum_leaf_relative_error,
        minimum_final_occupied_material_scale_bins,
        minimum_final_fractional_material_scale_fraction,
        minimum_final_dyadic_scale_quantization_rmse_octaves,
        mean_adaptive_rollout_elapsed_ms,
        mean_adaptive_topology_elapsed_ms,
        maximum_topology_update_elapsed_ms,
        gap_decomposition: None,
    }
}

fn failed_output_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("adaptive");
    let extension = path.extension().and_then(|extension| extension.to_str());
    let file_name = extension.map_or_else(
        || format!("{stem}.failed"),
        |extension| format!("{stem}.failed.{extension}"),
    );
    path.with_file_name(file_name)
}

/// Applies the parity and conservation gates used by both the full experiment
/// suite and the standalone artifact evaluator.
pub fn validate_adaptive_task_quality_validation_gates(
    gates: AdaptiveExperimentGates,
    validation: Option<&AdaptiveTaskQualityValidationReport>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let gap_gates_enabled = gates
        .min_gap_final_selected_mode_vs_fine_control_db
        .is_finite()
        || gates
            .min_gap_post_cut_recurrent_change_vs_fine_control_db
            .is_finite()
        || gates.max_gap_controller_target_render_regret_db.is_finite();
    let validation_seed_count = validation.map_or(0, |report| report.rows.len());
    if validation_seed_count < gates.min_task_quality_validation_seeds {
        failures.push(format!(
            "task parity validation seeds {validation_seed_count} < {}",
            gates.min_task_quality_validation_seeds,
        ));
    }
    let Some(validation) = validation else {
        if gap_gates_enabled {
            failures.push(
                "gap decomposition report is required by configured long-horizon gates".to_owned(),
            );
        }
        return failures;
    };
    if validation.mean_adaptive_over_teacher_psnr_gain_db
        < gates.min_validation_mean_adaptive_over_teacher_psnr_gain_db
    {
        failures.push(format!(
            "validation mean adaptive/upstream-teacher PSNR gain {:.3} dB < {:.3} dB",
            validation.mean_adaptive_over_teacher_psnr_gain_db,
            gates.min_validation_mean_adaptive_over_teacher_psnr_gain_db,
        ));
    }
    if validation.worst_adaptive_over_teacher_psnr_gain_db
        < gates.min_validation_worst_adaptive_over_teacher_psnr_gain_db
    {
        failures.push(format!(
            "validation worst adaptive/upstream-teacher PSNR gain {:.3} dB < {:.3} dB",
            validation.worst_adaptive_over_teacher_psnr_gain_db,
            gates.min_validation_worst_adaptive_over_teacher_psnr_gain_db,
        ));
    }
    if validation.mean_adaptive_over_regular_base_psnr_gain_db
        < gates.min_validation_mean_adaptive_over_regular_base_psnr_gain_db
    {
        failures.push(format!(
            "validation mean adaptive/regular-base PSNR gain {:.3} dB < {:.3} dB",
            validation.mean_adaptive_over_regular_base_psnr_gain_db,
            gates.min_validation_mean_adaptive_over_regular_base_psnr_gain_db,
        ));
    }
    if validation.worst_adaptive_over_regular_base_psnr_gain_db
        < gates.min_validation_worst_adaptive_over_regular_base_psnr_gain_db
    {
        failures.push(format!(
            "validation worst adaptive/regular-base PSNR gain {:.3} dB < {:.3} dB",
            validation.worst_adaptive_over_regular_base_psnr_gain_db,
            gates.min_validation_worst_adaptive_over_regular_base_psnr_gain_db,
        ));
    }
    if validation.mean_adaptive_over_regular_matched_budget_psnr_gain_db
        < gates.min_validation_mean_adaptive_over_regular_matched_budget_psnr_gain_db
    {
        failures.push(format!(
            "validation mean adaptive/matched-budget regular PSNR gain {:.3} dB < {:.3} dB",
            validation.mean_adaptive_over_regular_matched_budget_psnr_gain_db,
            gates.min_validation_mean_adaptive_over_regular_matched_budget_psnr_gain_db,
        ));
    }
    if validation.worst_adaptive_over_regular_matched_budget_psnr_gain_db
        < gates.min_validation_worst_adaptive_over_regular_matched_budget_psnr_gain_db
    {
        failures.push(format!(
            "validation worst adaptive/matched-budget regular PSNR gain {:.3} dB < {:.3} dB",
            validation.worst_adaptive_over_regular_matched_budget_psnr_gain_db,
            gates.min_validation_worst_adaptive_over_regular_matched_budget_psnr_gain_db,
        ));
    }
    if validation.mean_adaptive_over_regular_material_matched_budget_psnr_gain_db
        < gates.min_validation_mean_adaptive_over_regular_material_matched_budget_psnr_gain_db
    {
        failures.push(format!(
            "validation mean adaptive/material-matched regular PSNR gain {:.3} dB < {:.3} dB",
            validation.mean_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            gates.min_validation_mean_adaptive_over_regular_material_matched_budget_psnr_gain_db,
        ));
    }
    if validation.worst_adaptive_over_regular_material_matched_budget_psnr_gain_db
        < gates.min_validation_worst_adaptive_over_regular_material_matched_budget_psnr_gain_db
    {
        failures.push(format!(
            "validation worst adaptive/material-matched regular PSNR gain {:.3} dB < {:.3} dB",
            validation.worst_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            gates.min_validation_worst_adaptive_over_regular_material_matched_budget_psnr_gain_db,
        ));
    }
    if validation.mean_adaptive_over_budget_fixed_psnr_gain_db
        < gates.min_validation_mean_adaptive_over_budget_fixed_psnr_gain_db
    {
        failures.push(format!(
            "validation mean topology PSNR gain {:.3} dB < {:.3} dB",
            validation.mean_adaptive_over_budget_fixed_psnr_gain_db,
            gates.min_validation_mean_adaptive_over_budget_fixed_psnr_gain_db,
        ));
    }
    if validation.worst_adaptive_over_budget_fixed_psnr_gain_db
        < gates.min_validation_worst_adaptive_over_budget_fixed_psnr_gain_db
    {
        failures.push(format!(
            "validation worst topology PSNR gain {:.3} dB < {:.3} dB",
            validation.worst_adaptive_over_budget_fixed_psnr_gain_db,
            gates.min_validation_worst_adaptive_over_budget_fixed_psnr_gain_db,
        ));
    }
    if validation.minimum_controller_oracle_refinement_scale_correlation
        < gates.min_validation_controller_oracle_refinement_scale_correlation
    {
        failures.push(format!(
            "validation minimum controller/oracle correlation {:.4} < {:.4}",
            validation.minimum_controller_oracle_refinement_scale_correlation,
            gates.min_validation_controller_oracle_refinement_scale_correlation,
        ));
    }
    if validation.maximum_measure_relative_drift > gates.max_validation_measure_relative_drift {
        failures.push(format!(
            "validation maximum measure drift {:.3e} > {:.3e}",
            validation.maximum_measure_relative_drift, gates.max_validation_measure_relative_drift,
        ));
    }
    if validation.maximum_leaf_relative_error > gates.max_validation_leaf_relative_error {
        failures.push(format!(
            "validation maximum leaf-budget error {:.3} > {:.3}",
            validation.maximum_leaf_relative_error, gates.max_validation_leaf_relative_error,
        ));
    }
    if gates.require_active_leaf_dynamics
        && validation.rows.iter().any(|row| {
            row.dynamics_semantics != AdaptiveDynamicsSemantics::ActiveLeaves
                || row.adaptive_dynamics_particles != row.adaptive_final_particles
        })
    {
        failures.push(
            "validation requires active-leaf recurrent dynamics, but at least one trajectory retains hidden fine modes"
                .to_owned(),
        );
    }
    let maximum_interaction_ratio = validation
        .rows
        .iter()
        .map(|row| {
            row.adaptive_interaction_particles as f64 / row.adaptive_final_particles.max(1) as f64
        })
        .fold(0.0_f64, f64::max);
    if maximum_interaction_ratio > gates.max_validation_interaction_particle_ratio {
        failures.push(format!(
            "validation maximum interaction/active particle ratio {maximum_interaction_ratio:.3} > {:.3}",
            gates.max_validation_interaction_particle_ratio,
        ));
    }
    if validation.mean_adaptive_rollout_elapsed_ms
        > gates.max_validation_mean_adaptive_rollout_elapsed_ms
    {
        failures.push(format!(
            "validation mean adaptive rollout {:.2} ms > {:.2} ms",
            validation.mean_adaptive_rollout_elapsed_ms,
            gates.max_validation_mean_adaptive_rollout_elapsed_ms,
        ));
    }
    if validation.mean_adaptive_topology_elapsed_ms
        > gates.max_validation_mean_adaptive_topology_elapsed_ms
    {
        failures.push(format!(
            "validation mean adaptive topology {:.2} ms > {:.2} ms",
            validation.mean_adaptive_topology_elapsed_ms,
            gates.max_validation_mean_adaptive_topology_elapsed_ms,
        ));
    }
    if validation.maximum_topology_update_elapsed_ms
        > gates.max_validation_topology_update_elapsed_ms
    {
        failures.push(format!(
            "validation maximum topology update {:.2} ms > {:.2} ms",
            validation.maximum_topology_update_elapsed_ms,
            gates.max_validation_topology_update_elapsed_ms,
        ));
    }
    match validation.gap_decomposition.as_ref() {
        None if gap_gates_enabled => failures.push(
            "gap decomposition report is required by configured long-horizon gates".to_owned(),
        ),
        Some(gap) => {
            if gap
                .mean_final_selected_mode_gap_vs_fine_control_db
                .is_none_or(|value| value < gates.min_gap_final_selected_mode_vs_fine_control_db)
            {
                failures.push(format!(
                    "final selected-mode/fine-control PSNR gap {:?} < {:.3} dB",
                    gap.mean_final_selected_mode_gap_vs_fine_control_db,
                    gates.min_gap_final_selected_mode_vs_fine_control_db,
                ));
            }
            if gap
                .mean_post_cut_recurrent_gap_change_vs_fine_control_db
                .is_none_or(|value| {
                    value < gates.min_gap_post_cut_recurrent_change_vs_fine_control_db
                })
            {
                failures.push(format!(
                    "post-cut recurrent/fine-control PSNR change {:?} < {:.3} dB",
                    gap.mean_post_cut_recurrent_gap_change_vs_fine_control_db,
                    gates.min_gap_post_cut_recurrent_change_vs_fine_control_db,
                ));
            }
            if gap
                .mean_final_controller_target_render_regret_db
                .is_none_or(|value| value > gates.max_gap_controller_target_render_regret_db)
            {
                failures.push(format!(
                    "controller/target-render final regret {:?} > {:.3} dB",
                    gap.mean_final_controller_target_render_regret_db,
                    gates.max_gap_controller_target_render_regret_db,
                ));
            }
        }
        None => {}
    }
    failures
}

#[allow(clippy::too_many_arguments)]
fn validate_experiment_gates(
    gates: AdaptiveExperimentGates,
    base_training: Option<&AdaptiveBaseTrainingReport>,
    operators: [&AdaptiveOperatorExperimentReport; 2],
    topology: &AdaptiveTopologyExperimentReport,
    graph: &[AdaptiveGraphExperimentRow],
    scaling: &super::AdaptiveScalingExperimentReport,
    training: &crate::adaptive::AdaptiveControllerTrainingReport,
    validation: &AdaptiveControllerValidationReport,
    rule_distillation: Option<&crate::adaptive::AdaptiveRuleDistillationReport>,
    multiscale_training: Option<&AdaptiveMultiscaleExperimentReport>,
    restriction_training: Option<&AdaptiveRestrictionExperimentReport>,
    validate_multiscale_rule: bool,
    task_quality: Option<&AdaptiveTaskQualityReport>,
    task_quality_validation: Option<&AdaptiveTaskQualityValidationReport>,
    rollout: Option<&AdaptiveRolloutExperimentReport>,
) -> Vec<String> {
    let mut failures = Vec::new();
    if gates.require_fresh_base_training && base_training.is_none() {
        failures.push("fresh base training was required but did not run".to_string());
    }
    for operator in operators {
        if operator.adaptive_constant_max_error > gates.max_operator_constant_error {
            failures.push(format!(
                "{}D constant error {:.3e} > {:.3e}",
                operator.spatial_dims,
                operator.adaptive_constant_max_error,
                gates.max_operator_constant_error
            ));
        }
        if operator.adaptive_affine_gradient_mean_error > gates.max_operator_affine_error {
            failures.push(format!(
                "{}D affine error {:.3e} > {:.3e}",
                operator.spatial_dims,
                operator.adaptive_affine_gradient_mean_error,
                gates.max_operator_affine_error
            ));
        }
        if operator.moment_fallback_fraction > gates.max_operator_fallback_fraction {
            failures.push(format!(
                "{}D moment fallback {:.3e} > {:.3e}",
                operator.spatial_dims,
                operator.moment_fallback_fraction,
                gates.max_operator_fallback_fraction
            ));
        }
    }
    let topology_error = [
        topology.max_measure_relative_error,
        topology.max_centroid_l2_error,
        topology.max_second_moment_relative_error,
        topology.max_extensive_relative_error,
        topology.max_determinant_scale_relative_error,
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    if topology_error > gates.max_topology_relative_error {
        failures.push(format!(
            "topology error {topology_error:.3e} > {:.3e}",
            gates.max_topology_relative_error
        ));
    }
    if topology.spd_failures > gates.max_topology_spd_failures {
        failures.push(format!(
            "topology SPD failures {} > {}",
            topology.spd_failures, gates.max_topology_spd_failures
        ));
    }
    if topology.events_per_second < gates.min_topology_events_per_second {
        failures.push(format!(
            "topology throughput {:.0} < {:.0} events/s",
            topology.events_per_second, gates.min_topology_events_per_second
        ));
    }
    if validation.mean_squared_error > gates.max_validation_mse {
        failures.push(format!(
            "validation MSE {:.4} > {:.4}",
            validation.mean_squared_error, gates.max_validation_mse
        ));
    }
    if validation.desired_scale_correlation < gates.min_desired_scale_correlation {
        failures.push(format!(
            "scale correlation {:.4} < {:.4}",
            validation.desired_scale_correlation, gates.min_desired_scale_correlation
        ));
    }
    if training.steps > 0 && training.rows_per_second < gates.min_controller_rows_per_second {
        failures.push(format!(
            "controller throughput {:.0} < {:.0} rows/s",
            training.rows_per_second, gates.min_controller_rows_per_second
        ));
    }
    if let Some(distillation) = rule_distillation {
        if distillation
            .trained_validation
            .normalized_mean_squared_error
            > gates.max_rule_normalized_mean_squared_error
        {
            failures.push(format!(
                "adaptive rule normalized MSE {:.4} > {:.4}",
                distillation
                    .trained_validation
                    .normalized_mean_squared_error,
                gates.max_rule_normalized_mean_squared_error
            ));
        }
        if distillation.trained_validation.update_correlation < gates.min_rule_update_correlation {
            failures.push(format!(
                "adaptive rule update correlation {:.4} < {:.4}",
                distillation.trained_validation.update_correlation,
                gates.min_rule_update_correlation
            ));
        }
    }
    if let Some(multiscale) = multiscale_training {
        let heldout = &multiscale.heldout_validation;
        if validate_multiscale_rule
            && heldout.normalized_mean_squared_error
                > gates.max_multiscale_normalized_mean_squared_error
        {
            failures.push(format!(
                "held-out multiscale normalized MSE {:.4} > {:.4}",
                heldout.normalized_mean_squared_error,
                gates.max_multiscale_normalized_mean_squared_error
            ));
        }
        if validate_multiscale_rule
            && heldout.update_correlation < gates.min_multiscale_update_correlation
        {
            failures.push(format!(
                "held-out multiscale update correlation {:.4} < {:.4}",
                heldout.update_correlation, gates.min_multiscale_update_correlation
            ));
        }
        if validate_multiscale_rule
            && heldout.proxy_relative_mse_gain < gates.min_proxy_relative_mse_gain
        {
            failures.push(format!(
                "held-out proxy MSE gain {:.4} < {:.4}",
                heldout.proxy_relative_mse_gain, gates.min_proxy_relative_mse_gain
            ));
        }
        if multiscale
            .validation_dataset
            .footprint_coefficient_of_variation
            < gates.min_multiscale_dataset_footprint_coefficient_of_variation
        {
            failures.push(format!(
                "held-out multiscale footprint CV {:.4} < {:.4}",
                multiscale
                    .validation_dataset
                    .footprint_coefficient_of_variation,
                gates.min_multiscale_dataset_footprint_coefficient_of_variation
            ));
        }
        if let Some(closure) = &multiscale.heldout_on_policy_closure_validation {
            if closure.normalized_root_mean_squared_error
                > gates.max_recurrent_closure_normalized_root_mean_squared_error
            {
                failures.push(format!(
                    "held-out recurrent closure NRMSE {:.4} > {:.4}",
                    closure.normalized_root_mean_squared_error,
                    gates.max_recurrent_closure_normalized_root_mean_squared_error,
                ));
            }
            if closure.update_correlation < gates.min_recurrent_closure_update_correlation {
                failures.push(format!(
                    "held-out recurrent closure correlation {:.4} < {:.4}",
                    closure.update_correlation, gates.min_recurrent_closure_update_correlation,
                ));
            }
        }
    }
    if let Some(restriction) = restriction_training {
        if restriction.heldout_selection.accuracy < gates.min_restriction_heldout_accuracy {
            failures.push(format!(
                "held-out restriction accuracy {:.4} < {:.4}",
                restriction.heldout_selection.accuracy, gates.min_restriction_heldout_accuracy,
            ));
        }
        if restriction.heldout_selection.intersection_over_union
            < gates.min_restriction_heldout_intersection_over_union
        {
            failures.push(format!(
                "held-out restriction IoU {:.4} < {:.4}",
                restriction.heldout_selection.intersection_over_union,
                gates.min_restriction_heldout_intersection_over_union,
            ));
        }
    }
    if let Some(quality) = task_quality {
        if quality.adaptive_target_composited_psnr_db < gates.min_adaptive_target_psnr_db {
            failures.push(format!(
                "adaptive target PSNR {:.3} dB < {:.3} dB",
                quality.adaptive_target_composited_psnr_db, gates.min_adaptive_target_psnr_db
            ));
        }
        if quality.adaptive_teacher_psnr_gap_db > gates.max_adaptive_teacher_psnr_gap_db {
            failures.push(format!(
                "adaptive/teacher target PSNR gap {:.3} dB > {:.3} dB",
                quality.adaptive_teacher_psnr_gap_db, gates.max_adaptive_teacher_psnr_gap_db
            ));
        }
        if quality.adaptive_fine_fixed_teacher_psnr_gap_db
            > gates.max_fine_fixed_teacher_psnr_gap_db
        {
            failures.push(format!(
                "fine fixed/teacher target PSNR gap {:.3} dB > {:.3} dB",
                quality.adaptive_fine_fixed_teacher_psnr_gap_db,
                gates.max_fine_fixed_teacher_psnr_gap_db,
            ));
        }
        let adaptive_gain = quality.adaptive_target_composited_psnr_db
            - quality.adaptive_budget_fixed_target_composited_psnr_db;
        if adaptive_gain < gates.min_adaptive_over_budget_fixed_psnr_gain_db {
            failures.push(format!(
                "adaptive topology PSNR gain {adaptive_gain:.3} dB < {:.3} dB",
                gates.min_adaptive_over_budget_fixed_psnr_gain_db,
            ));
        }
        if quality.bandwidth_adaptation_target_psnr_gain_db
            < gates.min_bandwidth_adaptation_psnr_gain_db
        {
            failures.push(format!(
                "adaptive bandwidth PSNR gain {:.3} dB < {:.3} dB",
                quality.bandwidth_adaptation_target_psnr_gain_db,
                gates.min_bandwidth_adaptation_psnr_gain_db,
            ));
        }
        if gates.require_task_quality_bandwidth_adaptation_active
            && !quality.bandwidth_adaptation_active
        {
            failures.push(
                "task quality required active bandwidth adaptation, but the configured rule does not support it"
                    .to_string(),
            );
        }
        if quality.adaptive_over_regular_base_psnr_gain_db
            < gates.min_adaptive_over_regular_base_psnr_gain_db
        {
            failures.push(format!(
                "adaptive/regular-base PSNR gain {:.3} dB < {:.3} dB",
                quality.adaptive_over_regular_base_psnr_gain_db,
                gates.min_adaptive_over_regular_base_psnr_gain_db,
            ));
        }
        if quality.adaptive_over_regular_matched_budget_psnr_gain_db
            < gates.min_adaptive_over_regular_matched_budget_psnr_gain_db
        {
            failures.push(format!(
                "adaptive/matched-budget regular PSNR gain {:.3} dB < {:.3} dB",
                quality.adaptive_over_regular_matched_budget_psnr_gain_db,
                gates.min_adaptive_over_regular_matched_budget_psnr_gain_db,
            ));
        }
        if quality.adaptive_over_regular_material_matched_budget_psnr_gain_db
            < gates.min_adaptive_over_regular_material_matched_budget_psnr_gain_db
        {
            failures.push(format!(
                "adaptive/material-matched regular PSNR gain {:.3} dB < {:.3} dB",
                quality.adaptive_over_regular_material_matched_budget_psnr_gain_db,
                gates.min_adaptive_over_regular_material_matched_budget_psnr_gain_db,
            ));
        }
        if quality.deployment_over_training_policy_psnr_gain_db
            < gates.min_deployment_over_training_policy_psnr_gain_db
        {
            failures.push(format!(
                "adaptive deployment/training-policy PSNR gain {:.3} dB < {:.3} dB",
                quality.deployment_over_training_policy_psnr_gain_db,
                gates.min_deployment_over_training_policy_psnr_gain_db,
            ));
        }
        let target_leaves = quality.adaptive_target_particles.max(1);
        let leaf_relative_error =
            quality.adaptive_final_particles.abs_diff(target_leaves) as f64 / target_leaves as f64;
        if leaf_relative_error > gates.max_task_quality_leaf_relative_error {
            failures.push(format!(
                "task rollout leaf-budget error {leaf_relative_error:.3} > {:.3}",
                gates.max_task_quality_leaf_relative_error,
            ));
        }
        if gates.require_active_leaf_dynamics
            && (quality.dynamics_semantics != AdaptiveDynamicsSemantics::ActiveLeaves
                || quality.adaptive_dynamics_particles != quality.adaptive_final_particles)
        {
            failures.push(
                "task quality requires active-leaf recurrent dynamics without hidden fine modes"
                    .to_owned(),
            );
        }
        if quality.final_footprint_coefficient_of_variation
            < gates.min_task_quality_footprint_coefficient_of_variation
        {
            failures.push(format!(
                "task rollout footprint CV {:.3} < {:.3}",
                quality.final_footprint_coefficient_of_variation,
                gates.min_task_quality_footprint_coefficient_of_variation,
            ));
        }
        let footprint_ratio =
            quality.final_max_footprint / quality.final_min_footprint.max(f32::MIN_POSITIVE);
        if footprint_ratio < gates.min_task_quality_footprint_ratio {
            failures.push(format!(
                "task rollout footprint ratio {:.2} < {:.2}",
                footprint_ratio, gates.min_task_quality_footprint_ratio,
            ));
        }
        if quality.final_occupied_material_scale_bins
            < gates.min_task_quality_occupied_material_scale_bins
        {
            failures.push(format!(
                "task rollout material-scale bins {} < {}",
                quality.final_occupied_material_scale_bins,
                gates.min_task_quality_occupied_material_scale_bins,
            ));
        }
        if quality.final_fractional_material_scale_fraction
            < gates.min_task_quality_fractional_material_scale_fraction
        {
            failures.push(format!(
                "task rollout off-dyadic material fraction {:.3} < {:.3}",
                quality.final_fractional_material_scale_fraction,
                gates.min_task_quality_fractional_material_scale_fraction,
            ));
        }
        let topology_events = quality.steady_split_events + quality.steady_merge_events;
        if topology_events < gates.min_task_quality_topology_events {
            failures.push(format!(
                "task rollout topology events {topology_events} < {}",
                gates.min_task_quality_topology_events,
            ));
        }
        if quality.detail_density_correlation < gates.min_task_quality_detail_density_correlation {
            failures.push(format!(
                "task detail-density correlation {:.4} < {:.4}",
                quality.detail_density_correlation,
                gates.min_task_quality_detail_density_correlation,
            ));
        }
        if quality.high_to_low_detail_footprint_ratio
            > gates.max_task_quality_high_to_low_detail_footprint_ratio
        {
            failures.push(format!(
                "task high/low-detail footprint ratio {:.4} > {:.4}",
                quality.high_to_low_detail_footprint_ratio,
                gates.max_task_quality_high_to_low_detail_footprint_ratio,
            ));
        }
        if quality.refinement_defect_density_correlation
            < gates.min_task_quality_refinement_defect_density_correlation
        {
            failures.push(format!(
                "task base-refinement density correlation {:.4} < {:.4}",
                quality.refinement_defect_density_correlation,
                gates.min_task_quality_refinement_defect_density_correlation,
            ));
        }
        if quality.low_to_high_refinement_defect_footprint_ratio
            < gates.min_task_quality_low_to_high_refinement_defect_footprint_ratio
        {
            failures.push(format!(
                "task low/high-refinement-defect footprint ratio {:.4} < {:.4}",
                quality.low_to_high_refinement_defect_footprint_ratio,
                gates.min_task_quality_low_to_high_refinement_defect_footprint_ratio,
            ));
        }
        if quality.adaptive_refinement_defect_relative_gain
            < gates.min_task_quality_refinement_defect_relative_gain
        {
            failures.push(format!(
                "task adaptive refinement-defect gain {:.4} < {:.4}",
                quality.adaptive_refinement_defect_relative_gain,
                gates.min_task_quality_refinement_defect_relative_gain,
            ));
        }
        if quality.controller_oracle_refinement_scale_correlation
            < gates.min_task_quality_controller_oracle_refinement_scale_correlation
        {
            failures.push(format!(
                "task controller/oracle refinement-scale correlation {:.4} < {:.4}",
                quality.controller_oracle_refinement_scale_correlation,
                gates.min_task_quality_controller_oracle_refinement_scale_correlation,
            ));
        }
    }
    failures.extend(validate_adaptive_task_quality_validation_gates(
        gates,
        task_quality_validation,
    ));
    if gates.require_graph_cap_and_search_parity {
        for row in graph.iter().filter(|row| row.neighbor_cap > 0) {
            if row.degree_max > row.neighbor_cap {
                failures.push(format!(
                    "{}D n={} {} k={} produced degree {}",
                    row.spatial_dims, row.particles, row.policy, row.neighbor_cap, row.degree_max
                ));
            }
        }
        for hashed in graph
            .iter()
            .filter(|row| row.search == "spatial-hash" && row.policy == "raw-support")
        {
            if let Some(oracle) = graph.iter().find(|row| {
                row.search == "all-pairs"
                    && row.spatial_dims == hashed.spatial_dims
                    && row.particles == hashed.particles
            }) && (oracle.raw_messages != hashed.raw_messages
                || oracle.accepted_messages != hashed.accepted_messages)
            {
                failures.push(format!(
                    "{}D n={} spatial search messages differ from all-pairs",
                    hashed.spatial_dims, hashed.particles
                ));
            }
        }
    }
    validate_sparse_quality_gates(gates, scaling, &mut failures);
    if let Some(rollout) = rollout {
        if rollout.target_leaf_relative_error > gates.max_rollout_target_relative_error {
            failures.push(format!(
                "rollout target gap {:.3} > {:.3}",
                rollout.target_leaf_relative_error, gates.max_rollout_target_relative_error
            ));
        }
        if rollout.measure_relative_drift > gates.max_rollout_measure_relative_drift {
            failures.push(format!(
                "rollout measure drift {:.3e} > {:.3e}",
                rollout.measure_relative_drift, gates.max_rollout_measure_relative_drift
            ));
        }
        if rollout.final_footprint_coefficient_of_variation
            < gates.min_rollout_footprint_coefficient_of_variation
        {
            failures.push(format!(
                "rollout footprint CV {:.3} < {:.3}",
                rollout.final_footprint_coefficient_of_variation,
                gates.min_rollout_footprint_coefficient_of_variation
            ));
        }
        if rollout.final_occupied_material_scale_bins
            < gates.min_rollout_occupied_material_scale_bins
        {
            failures.push(format!(
                "rollout material-scale bins {} < {}",
                rollout.final_occupied_material_scale_bins,
                gates.min_rollout_occupied_material_scale_bins,
            ));
        }
        if rollout.final_fractional_material_scale_fraction
            < gates.min_rollout_fractional_material_scale_fraction
        {
            failures.push(format!(
                "rollout off-dyadic material fraction {:.3} < {:.3}",
                rollout.final_fractional_material_scale_fraction,
                gates.min_rollout_fractional_material_scale_fraction,
            ));
        }
        if rollout.particle_steps_per_second < gates.min_rollout_particle_steps_per_second {
            failures.push(format!(
                "rollout throughput {:.0} < {:.0} particle-steps/s",
                rollout.particle_steps_per_second, gates.min_rollout_particle_steps_per_second
            ));
        }
    }
    failures
}

fn validate_sparse_quality_gates(
    gates: AdaptiveExperimentGates,
    scaling: &super::AdaptiveScalingExperimentReport,
    failures: &mut Vec<String>,
) {
    let maximum_cap = scaling
        .quality_rows
        .iter()
        .map(|row| row.spacing_cap_ratio)
        .reduce(f32::max);
    for row in &scaling.quality_rows {
        let conservation = row
            .measure_relative_error
            .max(row.centroid_l2_error)
            .max(row.field_integral_relative_error);
        if !conservation.is_finite() || conservation > gates.max_sparse_conservation_error {
            failures.push(format!(
                "{} {} cap {:.1} sparse conservation {:.3e} > {:.3e}",
                row.solid,
                row.allocation,
                row.spacing_cap_ratio,
                conservation,
                gates.max_sparse_conservation_error
            ));
        }
        if row.allocation == "clearance-oracle"
            && row.protected_band_nrmse > gates.max_sparse_protected_band_nrmse
        {
            failures.push(format!(
                "{} cap {:.1} protected-band NRMSE {:.3e} > {:.3e}",
                row.solid,
                row.spacing_cap_ratio,
                row.protected_band_nrmse,
                gates.max_sparse_protected_band_nrmse
            ));
        }
        if row.allocation == "clearance-oracle"
            && row.boundary_hd95_voxels > gates.max_sparse_boundary_hd95_voxels
        {
            failures.push(format!(
                "{} cap {:.1} boundary HD95 {:.3} > {:.3} voxels",
                row.solid,
                row.spacing_cap_ratio,
                row.boundary_hd95_voxels,
                gates.max_sparse_boundary_hd95_voxels
            ));
        }
        if row.allocation == "uniform-matched-count" {
            let matched = scaling.quality_rows.iter().find(|candidate| {
                candidate.allocation == "clearance-oracle"
                    && candidate.solid == row.solid
                    && candidate.sample == row.sample
                    && (candidate.spacing_cap_ratio - row.spacing_cap_ratio).abs() <= f32::EPSILON
            });
            if matched.is_none_or(|matched| matched.retained_leaves != row.retained_leaves) {
                failures.push(format!(
                    "{} cap {:.1} sample {} uniform control is not count matched",
                    row.solid, row.spacing_cap_ratio, row.sample
                ));
            }
        }
        if row.allocation == "clearance-oracle"
            && maximum_cap.is_some_and(|cap| (row.spacing_cap_ratio - cap).abs() <= f32::EPSILON)
            && row.count_reduction < gates.min_sparse_largest_cap_count_reduction
        {
            failures.push(format!(
                "{} largest-cap count reduction {:.3} < {:.3}",
                row.solid, row.count_reduction, gates.min_sparse_largest_cap_count_reduction
            ));
        }
    }
}

fn validate_experiment_alignment(config: &AdaptiveExperimentConfig) -> AutomataResult<()> {
    let data = config.training_data;
    let adaptive = &config.adaptive;
    let same = |lhs: f32, rhs: f32| (lhs - rhs).abs() <= 1.0e-6 * lhs.abs().max(rhs.abs()).max(1.0);
    if data.spatial_dims != adaptive.spatial_dims
        || data.target_leaf_count != adaptive.target_leaves
        || !same(data.reference_footprint, adaptive.reference_footprint)
        || !same(data.min_footprint, adaptive.min_footprint)
        || !same(data.max_footprint, adaptive.max_footprint)
        || !same(data.total_measure, config.rollout.total_measure)
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive training-data, model-budget, and rollout measure/footprint contracts must match"
                .to_string(),
        ));
    }
    if config.rule_distillation.enabled
        && adaptive.rule_perception != crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive rule distillation is only valid for normalized-adaptive rule perception"
                .to_string(),
        ));
    }
    if config.base_training.enabled {
        if config.base_model.is_some()
            || config.adaptive.spatial_dims != 2
            || config.base_training.target_image.as_os_str().is_empty()
            || config.base_training.phases.is_empty()
            || config.base_training.target_points == 0
            || !config.base_training.target_threshold.is_finite()
            || config.base_training.target_threshold < 0.0
        {
            return Err(AutomataError::InvalidArgument(
                "fresh adaptive base training requires no base_model, a 2D target image, at least one phase, positive target_points, and a non-negative threshold"
                    .to_string(),
            ));
        }
        for phase in &config.base_training.phases {
            if phase.name.trim().is_empty() || phase.training.particle_count == 0 {
                return Err(AutomataError::InvalidArgument(
                    "fresh adaptive base phases require a name and positive particle count"
                        .to_string(),
                ));
            }
        }
    }
    if config.multiscale_training.enabled {
        match config.multiscale_training.rule_strategy {
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::Residual => {
                if adaptive.rule_perception
                    != crate::adaptive::AdaptiveRulePerception::NpaCompatible
                    || !((adaptive.local_rule_semantics
                        == crate::adaptive::AdaptiveLocalRuleSemantics::Residual
                        && adaptive.proxy.enabled)
                        || (adaptive.local_rule_semantics
                            == crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual
                            && !adaptive.proxy.enabled))
                {
                    return Err(AutomataError::InvalidArgument(
                        "residual multiscale training requires an NPA-compatible task base and either the normalized local/proxy policy or the canonical compatible-residual policy"
                            .to_string(),
                    ));
                }
            }
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::CoarseReplacement => {
                if adaptive.rule_perception
                    != crate::adaptive::AdaptiveRulePerception::NpaCompatible
                    || adaptive.proxy.enabled
                    || adaptive.local_rule_semantics
                        != crate::adaptive::AdaptiveLocalRuleSemantics::CoarseReplacement
                {
                    return Err(AutomataError::InvalidArgument(
                        "coarse-replacement multiscale training requires an NPA-compatible task base, local_rule_semantics=coarse-replacement, and proxy.enabled=false"
                            .to_string(),
                    ));
                }
            }
            crate::adaptive::AdaptiveMultiscaleRuleStrategy::FullNormalized => {
                if adaptive.rule_perception
                    != crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive
                    || adaptive.proxy.enabled
                    || adaptive.local_residual_scale != 0.0
                    || adaptive.closure_moment_features
                {
                    return Err(AutomataError::InvalidArgument(
                        "full-normalized multiscale training requires normalized-adaptive perception, proxy.enabled=false, local_residual_scale=0, and closure_moment_features=false"
                            .to_string(),
                    ));
                }
            }
        }
        if config.rule_distillation.enabled {
            return Err(AutomataError::InvalidArgument(
                "legacy rule distillation and multiscale task training are mutually exclusive"
                    .to_string(),
            ));
        }
        if config.multiscale_training.deployment_enabled
            && config.multiscale_training.deployment_target
                == crate::adaptive::AdaptiveDeploymentTarget::RestrictedFineTeacher
            && config.multiscale_training.deployment_on_policy_rounds > 0
            && config.multiscale_training.deployment_replay_backend
                != crate::adaptive::AdaptiveReplayBackend::CpuReference
        {
            return Err(AutomataError::InvalidArgument(
                "restricted-fine-teacher deployment DAgger requires cpu-reference replay"
                    .to_string(),
            ));
        }
        if !same(
            config.multiscale_training.total_measure,
            config.rollout.total_measure,
        ) {
            return Err(AutomataError::InvalidArgument(
                "adaptive multiscale training and rollout total measure must match".to_string(),
            ));
        }
    }
    if config.task_quality.enabled
        && (config.task_quality.target_image.as_os_str().is_empty()
            || config.task_quality.image_size == 0
            || config.task_quality.rollout_steps == 0
            || config.task_quality.structural_audit_seeds == 0
            || !config.task_quality.update_prob.is_finite()
            || !(0.0..=1.0).contains(&config.task_quality.update_prob)
            || !config.task_quality.render_compactness.is_finite()
            || !(0.0..=1.0).contains(&config.task_quality.render_compactness))
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive task-quality evaluation requires a target image, positive image/rollout/audit sizes, and a valid update probability"
                .to_string(),
        ));
    }
    if config.task_quality.enabled {
        let _ = task_reference_model(config)?;
        if config.task_quality.restriction_policy
            == super::AdaptiveTaskRestrictionPolicy::LearnedController
            && config.adaptive.hierarchical_restriction_policy
                != AdaptiveHierarchyRestrictionPolicy::LearnedController
        {
            return Err(AutomataError::InvalidArgument(
                "learned task restriction requires adaptive.hierarchical_restriction_policy=learned-controller"
                    .to_string(),
            ));
        }
        if config.task_quality.restriction_policy
            == super::AdaptiveTaskRestrictionPolicy::TargetRenderOracle
        {
            let fine = config.adaptive.bootstrap_fine_leaf_count();
            let target = config.adaptive.target_leaves;
            if !config
                .task_quality
                .render_decoder
                .supports_restriction_labels()
                || config.adaptive.initial_leaf_count() != fine
                || config.adaptive.hierarchical_restriction_step == 0
                || config.adaptive.hierarchical_restriction_step > config.task_quality.rollout_steps
                || target < fine.div_ceil(4)
                || target >= fine
                || !(fine - target).is_multiple_of(3)
            {
                return Err(AutomataError::InvalidArgument(
                    "target-render-oracle requires a supported isotropic/diagnostic decoder, a full fine initial population, a scheduled cut within the quality rollout, and a first-level 4-to-1 reachable target budget"
                        .to_string(),
                ));
            }
        }
    }
    if config
        .gates
        .require_task_quality_bandwidth_adaptation_active
        && (!config.task_quality.enabled
            || !config.task_quality.bandwidth_adaptation_enabled
            || !config.adaptive.supports_bandwidth_adaptation())
    {
        return Err(AutomataError::InvalidArgument(
            "requiring active task-quality bandwidth adaptation needs task quality enabled, bandwidth_adaptation_enabled=true, and normalized-adaptive rule perception"
                .to_string(),
        ));
    }
    if config.restriction_training.enabled {
        let training = &config.restriction_training;
        if !config.task_quality.enabled
            || config.adaptive.hierarchical_restriction_policy
                != AdaptiveHierarchyRestrictionPolicy::LearnedController
            || training
                .training_data
                .seeds
                .iter()
                .any(|seed| training.validation_data.seeds.contains(seed))
        {
            return Err(AutomataError::InvalidArgument(
                "restriction training requires task-quality evaluation, learned-controller runtime restriction, and disjoint train/validation seeds"
                    .to_string(),
            ));
        }
        for dataset in [&training.training_data, &training.validation_data] {
            if !same(dataset.seed_scale, config.multiscale_training.seed_scale)
                || !same(
                    dataset.total_measure,
                    config.multiscale_training.total_measure,
                )
                || !same(dataset.bandwidth, config.multiscale_training.bandwidth)
                || !same(dataset.update_prob, config.task_quality.update_prob)
                || dataset.render_decoder != config.task_quality.render_decoder
                || !same(
                    dataset.render_compactness,
                    config.task_quality.render_compactness,
                )
                || dataset.label_target != training.training_data.label_target
                || dataset
                    .cut_steps
                    .iter()
                    .any(|step| *step > config.task_quality.rollout_steps)
            {
                return Err(AutomataError::InvalidArgument(
                    "restriction training data must match the multiscale seed/measure/bandwidth and task-quality rollout/render configuration"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn train_controller(
    backend: AdaptiveTrainingBackend,
    controller: &mut crate::adaptive::AdaptiveController,
    batch: &crate::adaptive::AdaptiveControllerTrainingBatch,
    config: crate::adaptive::AdaptiveControllerTrainConfig,
) -> AutomataResult<crate::adaptive::AdaptiveControllerTrainingReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            #[cfg(feature = "backend_ndarray")]
            {
                crate::adaptive::train_adaptive_controller_ndarray(controller, batch, config)
            }
            #[cfg(not(feature = "backend_ndarray"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive ndarray training requires backend_ndarray".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                crate::adaptive::train_adaptive_controller_wgpu(controller, batch, config)
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive WGPU training requires backend_wgpu".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                crate::adaptive::train_adaptive_controller_cuda(controller, batch, config)
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive CUDA training requires backend_cuda".to_string(),
                ))
            }
        }
    }
}

fn train_restriction_controller(
    backend: AdaptiveTrainingBackend,
    controller: &mut crate::adaptive::AdaptiveController,
    batch: &crate::adaptive::AdaptiveRestrictionTrainingBatch,
    validation_batch: &crate::adaptive::AdaptiveRestrictionTrainingBatch,
    config: crate::adaptive::AdaptiveControllerTrainConfig,
) -> AutomataResult<crate::adaptive::AdaptiveControllerTrainingReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            #[cfg(feature = "backend_ndarray")]
            {
                crate::adaptive::train_adaptive_restriction_controller_ndarray(
                    controller,
                    batch,
                    validation_batch,
                    config,
                )
            }
            #[cfg(not(feature = "backend_ndarray"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive ndarray restriction training requires backend_ndarray".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                crate::adaptive::train_adaptive_restriction_controller_wgpu(
                    controller,
                    batch,
                    validation_batch,
                    config,
                )
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive WGPU restriction training requires backend_wgpu".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                crate::adaptive::train_adaptive_restriction_controller_cuda(
                    controller,
                    batch,
                    validation_batch,
                    config,
                )
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive CUDA restriction training requires backend_cuda".to_string(),
                ))
            }
        }
    }
}

fn restriction_training_batch(
    backend: AdaptiveTrainingBackend,
    rollout_executor: Option<&RestrictionRolloutExecutor>,
    model: &AdaptiveNpaModel,
    _grid: &HashGridConfig,
    target: &crate::target2d::TargetImage2d,
    render_config: crate::target2d::Target2dLossConfig,
    config: &crate::adaptive::AdaptiveRestrictionDatasetConfig,
) -> AutomataResult<(
    crate::adaptive::AdaptiveRestrictionTrainingBatch,
    crate::adaptive::AdaptiveRestrictionDatasetReport,
)> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            let _ = rollout_executor;
            adaptive_restriction_training_batch(model, target, render_config, config)
        }
        AdaptiveTrainingBackend::Wgpu => {
            #[cfg(all(feature = "backend_wgpu", feature = "gpu_wgpu"))]
            {
                let rollout_executor = rollout_executor.ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "adaptive WGPU restriction labels require a resident rollout executor"
                            .to_string(),
                    )
                })?;
                crate::adaptive::adaptive_restriction_training_batch_burn_with_executor::<
                    burn::backend::Wgpu<f32>,
                >(
                    rollout_executor,
                    model,
                    _grid,
                    target,
                    render_config,
                    config,
                    &Default::default(),
                    "burn-wgpu",
                )
            }
            #[cfg(not(all(feature = "backend_wgpu", feature = "gpu_wgpu")))]
            {
                let _ = rollout_executor;
                Err(AutomataError::InvalidArgument(
                    "adaptive WGPU restriction labels require backend_wgpu and gpu_wgpu"
                        .to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Cuda => {
            #[cfg(all(feature = "backend_cuda", feature = "gpu_wgpu"))]
            {
                let rollout_executor = rollout_executor.ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "adaptive CUDA restriction labels require a resident rollout executor"
                            .to_string(),
                    )
                })?;
                crate::adaptive::adaptive_restriction_training_batch_burn_with_executor::<
                    burn::backend::Cuda<f32>,
                >(
                    rollout_executor,
                    model,
                    _grid,
                    target,
                    render_config,
                    config,
                    &Default::default(),
                    "burn-cuda",
                )
            }
            #[cfg(not(all(feature = "backend_cuda", feature = "gpu_wgpu")))]
            {
                let _ = rollout_executor;
                Err(AutomataError::InvalidArgument(
                    "adaptive CUDA restriction labels require backend_cuda and gpu_wgpu"
                        .to_string(),
                ))
            }
        }
    }
}

fn train_rule(
    backend: AdaptiveTrainingBackend,
    rule: &mut NpaModel,
    batch: &crate::adaptive::AdaptiveRuleTrainingBatch,
    config: crate::adaptive::AdaptiveRuleDistillationConfig,
) -> AutomataResult<crate::adaptive::AdaptiveRuleTrainingReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            #[cfg(feature = "backend_ndarray")]
            {
                crate::adaptive::train_adaptive_rule_ndarray(rule, batch, config)
            }
            #[cfg(not(feature = "backend_ndarray"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive ndarray rule training requires backend_ndarray".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                crate::adaptive::train_adaptive_rule_wgpu(rule, batch, config)
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive WGPU rule training requires backend_wgpu".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                crate::adaptive::train_adaptive_rule_cuda(rule, batch, config)
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive CUDA rule training requires backend_cuda".to_string(),
                ))
            }
        }
    }
}

fn train_multiscale_rule(
    backend: AdaptiveTrainingBackend,
    model: &mut AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    config: &crate::adaptive::AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<crate::adaptive::AdaptiveMultiscaleRuleTrainingReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            crate::adaptive::train_adaptive_multiscale_rule_ndarray(model, batch, config)
        }
        AdaptiveTrainingBackend::Wgpu => {
            crate::adaptive::train_adaptive_multiscale_rule_wgpu(model, batch, config)
        }
        AdaptiveTrainingBackend::Cuda => {
            crate::adaptive::train_adaptive_multiscale_rule_cuda(model, batch, config)
        }
    }
}

fn validate_multiscale_rule(
    backend: AdaptiveTrainingBackend,
    model: &AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<crate::adaptive::AdaptiveMultiscaleRuleValidationReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            crate::adaptive::adaptive_multiscale_rule_validation_ndarray(model, batch)
        }
        AdaptiveTrainingBackend::Wgpu => {
            crate::adaptive::adaptive_multiscale_rule_validation_wgpu(model, batch)
        }
        AdaptiveTrainingBackend::Cuda => {
            crate::adaptive::adaptive_multiscale_rule_validation_cuda(model, batch)
        }
    }
}

fn train_closure_mode(
    backend: AdaptiveTrainingBackend,
    model: &mut AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    config: &crate::adaptive::AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<crate::adaptive::AdaptiveClosureModeTrainingReport> {
    let mut config = config.clone();
    if let Some(optimizer) = config.closure_optimizer {
        config.optimizer = optimizer;
    }
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            crate::adaptive::train_adaptive_closure_mode_rule_ndarray(model, batch, &config)
        }
        AdaptiveTrainingBackend::Wgpu => {
            crate::adaptive::train_adaptive_closure_mode_rule_wgpu(model, batch, &config)
        }
        AdaptiveTrainingBackend::Cuda => {
            crate::adaptive::train_adaptive_closure_mode_rule_cuda(model, batch, &config)
        }
    }
}

fn train_deployment_rule(
    backend: AdaptiveTrainingBackend,
    model: &mut AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    config: &crate::adaptive::AdaptiveMultiscaleTrainingConfig,
) -> AutomataResult<crate::adaptive::AdaptiveDeploymentRuleTrainingReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            crate::adaptive::train_adaptive_deployment_rule_ndarray(model, batch, config)
        }
        AdaptiveTrainingBackend::Wgpu => {
            crate::adaptive::train_adaptive_deployment_rule_wgpu(model, batch, config)
        }
        AdaptiveTrainingBackend::Cuda => {
            crate::adaptive::train_adaptive_deployment_rule_cuda(model, batch, config)
        }
    }
}

fn validate_rule(
    rule: &NpaModel,
    batch: &crate::adaptive::AdaptiveRuleTrainingBatch,
) -> AutomataResult<crate::adaptive::AdaptiveRuleValidationReport> {
    batch.validate(rule.config.perception_dims(), rule.config.update_dims())?;
    let prediction = rule.forward_update_from_features(&batch.features)?;
    let mut squared_error = 0.0_f64;
    let mut squared_target = 0.0_f64;
    for (&predicted, &target) in prediction.iter().zip(&batch.target_update) {
        squared_error += (predicted - target).powi(2) as f64;
        squared_target += target.powi(2) as f64;
    }
    let values = prediction.len().max(1) as f64;
    Ok(crate::adaptive::AdaptiveRuleValidationReport {
        rows: batch.rows,
        mean_squared_error: (squared_error / values) as f32,
        normalized_mean_squared_error: (squared_error / squared_target.max(f64::MIN_POSITIVE))
            as f32,
        update_correlation: pearson(prediction.into_iter(), batch.target_update.iter().copied()),
        target_root_mean_square: (squared_target / values).sqrt() as f32,
    })
}

fn concatenate_rule_batches(
    first: &crate::adaptive::AdaptiveRuleTrainingBatch,
    second: &crate::adaptive::AdaptiveRuleTrainingBatch,
) -> crate::adaptive::AdaptiveRuleTrainingBatch {
    let mut features = Vec::with_capacity(first.features.len() + second.features.len());
    features.extend_from_slice(&first.features);
    features.extend_from_slice(&second.features);
    let mut target_update =
        Vec::with_capacity(first.target_update.len() + second.target_update.len());
    target_update.extend_from_slice(&first.target_update);
    target_update.extend_from_slice(&second.target_update);
    crate::adaptive::AdaptiveRuleTrainingBatch {
        features,
        target_update,
        rows: first.rows + second.rows,
        generation_elapsed_ms: first.generation_elapsed_ms + second.generation_elapsed_ms,
    }
}

fn same_npa_rule(lhs: &NpaModel, rhs: &NpaModel) -> bool {
    lhs.config == rhs.config
        && lhs.weights.w1 == rhs.weights.w1
        && lhs.weights.b1 == rhs.weights.b1
        && lhs.weights.w2 == rhs.weights.w2
        && lhs.weights.b2 == rhs.weights.b2
}

fn adaptive_rule_matches_reference_architecture(
    model: &AdaptiveNpaModel,
    reference: &NpaModel,
) -> AutomataResult<bool> {
    let mut canonical = model.rule.config.clone();
    if model.config.material_scale_conditioning {
        canonical.auxiliary_input_dims =
            canonical
                .auxiliary_input_dims
                .checked_sub(1)
                .ok_or_else(|| {
                    AutomataError::InvalidModel(
                        "material-scale-conditioned rule is missing its auxiliary input".to_owned(),
                    )
                })?;
    }
    Ok(canonical == reference.config)
}

fn same_adaptive_rollout_model(lhs: &AdaptiveNpaModel, rhs: &AdaptiveNpaModel) -> bool {
    lhs.config == rhs.config
        && same_npa_rule(&lhs.rule, &rhs.rule)
        && same_optional_npa_rule(&lhs.local_residual_rule, &rhs.local_residual_rule)
        && same_optional_npa_rule(&lhs.proxy_rule, &rhs.proxy_rule)
        && same_optional_npa_rule(&lhs.deployment_rule, &rhs.deployment_rule)
        && same_optional_npa_rule(&lhs.deployment_local_rule, &rhs.deployment_local_rule)
        && same_optional_npa_rule(&lhs.closure_mode_rule, &rhs.closure_mode_rule)
        && same_optional_npa_rule(&lhs.closure_basis_rule, &rhs.closure_basis_rule)
        && lhs.controller == rhs.controller
        && lhs.restriction_controller == rhs.restriction_controller
}

#[cfg(feature = "gpu_wgpu")]
fn same_wgpu_task_dynamics(lhs: &AdaptiveNpaModel, rhs: &AdaptiveNpaModel) -> AutomataResult<bool> {
    let mut lhs_config = lhs.config.clone();
    let mut rhs_config = rhs.config.clone();
    // The resident task evaluator does not execute the hierarchy-context
    // branch. A disabled context gain is therefore identical whether the
    // serialized training branch remains attached or was removed for an
    // ablation control.
    if lhs_config.proxy.context_scale.abs() <= f32::MIN_POSITIVE {
        lhs_config.proxy.enabled = false;
    }
    if rhs_config.proxy.context_scale.abs() <= f32::MIN_POSITIVE {
        rhs_config.proxy.enabled = false;
    }
    let lhs_rule = lhs.gpu_inference_rule()?;
    let rhs_rule = rhs.gpu_inference_rule()?;
    Ok(lhs_config == rhs_config
        && lhs_rule.local_hidden_start == rhs_rule.local_hidden_start
        && same_npa_rule(&lhs_rule.rule, &rhs_rule.rule)
        && same_optional_npa_rule(&lhs.closure_mode_rule, &rhs.closure_mode_rule)
        && same_optional_npa_rule(&lhs.closure_basis_rule, &rhs.closure_basis_rule)
        && lhs.controller == rhs.controller
        && lhs.restriction_controller == rhs.restriction_controller)
}

#[cfg(feature = "gpu_wgpu")]
fn timed_wgpu_precompute<T>(
    label: &str,
    operation: impl FnOnce() -> AutomataResult<T>,
) -> AutomataResult<T> {
    let started = Instant::now();
    let output = operation()?;
    eprintln!(
        "adaptive WGPU precompute {label}: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(output)
}

fn same_optional_npa_rule(lhs: &Option<NpaModel>, rhs: &Option<NpaModel>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => same_npa_rule(lhs, rhs),
        (None, None) => true,
        _ => false,
    }
}

fn append_controller_batch(
    first: &mut crate::adaptive::AdaptiveControllerTrainingBatch,
    second: &crate::adaptive::AdaptiveControllerTrainingBatch,
) {
    first.features.extend_from_slice(&second.features);
    first.targets.extend_from_slice(&second.targets);
    first.rows += second.rows;
}

fn update_controller_replay(
    replay: &mut crate::adaptive::AdaptiveControllerTrainingBatch,
    batch: &crate::adaptive::AdaptiveControllerTrainingBatch,
    reset: bool,
) {
    if reset {
        *replay = batch.clone();
    } else {
        append_controller_batch(replay, batch);
    }
}

fn append_multiscale_batch(
    first: &mut crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    second: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
) -> AutomataResult<()> {
    let first_has_closure = !first.closure_mode_target_update.is_empty()
        || !first.closure_basis_target_update.is_empty()
        || !first.closure_mode_row_weights.is_empty();
    let second_has_closure = !second.closure_mode_target_update.is_empty()
        || !second.closure_basis_target_update.is_empty()
        || !second.closure_mode_row_weights.is_empty();
    if first_has_closure != second_has_closure {
        return Err(AutomataError::InvalidArgument(
            "cannot combine adaptive replay batches with different closure-mode schemas".to_owned(),
        ));
    }
    first
        .local_features
        .extend_from_slice(&second.local_features);
    first
        .closure_features
        .extend_from_slice(&second.closure_features);
    first
        .proxy_features
        .extend_from_slice(&second.proxy_features);
    first.target_update.extend_from_slice(&second.target_update);
    first
        .closure_mode_target_update
        .extend_from_slice(&second.closure_mode_target_update);
    first
        .closure_basis_target_update
        .extend_from_slice(&second.closure_basis_target_update);
    first
        .closure_mode_row_weights
        .extend_from_slice(&second.closure_mode_row_weights);
    first
        .deployment_features
        .extend_from_slice(&second.deployment_features);
    first
        .deployment_target_update
        .extend_from_slice(&second.deployment_target_update);
    first
        .deployment_row_weights
        .extend_from_slice(&second.deployment_row_weights);
    first
        .deployment_residual_gate
        .extend_from_slice(&second.deployment_residual_gate);
    first
        .controller_features
        .extend_from_slice(&second.controller_features);
    first
        .controller_targets
        .extend_from_slice(&second.controller_targets);
    first.row_weights.extend_from_slice(&second.row_weights);
    first.rows += second.rows;
    first.report = second.report.clone();
    Ok(())
}

fn update_multiscale_replay(
    replay: &mut crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    batch: &crate::adaptive::AdaptiveMultiscaleTrainingBatch,
    reset: bool,
) -> AutomataResult<()> {
    if reset {
        *replay = batch.clone();
        Ok(())
    } else {
        append_multiscale_batch(replay, batch)
    }
}

fn validate_controller(
    backend: AdaptiveTrainingBackend,
    model: &AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveControllerTrainingBatch,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    match backend {
        AdaptiveTrainingBackend::NdArray => {
            #[cfg(feature = "backend_ndarray")]
            {
                crate::adaptive::validate_adaptive_controller_ndarray(
                    &model.controller,
                    batch,
                    model.config.split_probability,
                    model.config.merge_probability,
                )
            }
            #[cfg(not(feature = "backend_ndarray"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive NdArray controller validation requires backend_ndarray".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                crate::adaptive::validate_adaptive_controller_wgpu(
                    &model.controller,
                    batch,
                    model.config.split_probability,
                    model.config.merge_probability,
                )
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive WGPU controller validation requires backend_wgpu".to_string(),
                ))
            }
        }
        AdaptiveTrainingBackend::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                crate::adaptive::validate_adaptive_controller_cuda(
                    &model.controller,
                    batch,
                    model.config.split_probability,
                    model.config.merge_probability,
                )
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(AutomataError::InvalidArgument(
                    "adaptive CUDA controller validation requires backend_cuda".to_string(),
                ))
            }
        }
    }
}

#[cfg(all(test, feature = "backend_ndarray"))]
fn validate_controller_reference(
    model: &AdaptiveNpaModel,
    batch: &crate::adaptive::AdaptiveControllerTrainingBatch,
) -> AutomataResult<AdaptiveControllerValidationReport> {
    let prediction = model.controller.forward_raw(&batch.features)?;
    let mut channel_squared_error = [0.0_f64; 4];
    let mut positives = [0usize; 2];
    let mut true_positives = [0usize; 2];
    let mut predicted_positives = [0usize; 2];
    for row in 0..batch.rows {
        for (channel, squared_error) in channel_squared_error.iter_mut().enumerate() {
            let index = row * 4 + channel;
            let predicted = if channel >= 2 {
                logistic(prediction[index])
            } else {
                prediction[index]
            };
            let difference = predicted - batch.targets[index];
            *squared_error += (difference * difference) as f64;
            if channel >= 2 {
                let event = channel - 2;
                let target_positive = batch.targets[index] >= 0.5;
                let threshold = if event == 0 {
                    model.config.split_probability
                } else {
                    model.config.merge_probability
                };
                let predicted_positive = predicted >= threshold;
                positives[event] += usize::from(target_positive);
                predicted_positives[event] += usize::from(predicted_positive);
                true_positives[event] += usize::from(target_positive && predicted_positive);
            }
        }
    }
    let channel_mean_squared_error =
        channel_squared_error.map(|value| (value / batch.rows as f64) as f32);
    let mean_squared_error = channel_mean_squared_error.iter().sum::<f32>() / 4.0;
    let predicted_scale = prediction.chunks_exact(4).map(|row| row[0]);
    let target_scale = batch.targets.chunks_exact(4).map(|row| row[0]);
    let desired_scale_correlation = pearson(predicted_scale, target_scale);
    Ok(AdaptiveControllerValidationReport {
        rows: batch.rows,
        mean_squared_error,
        channel_mean_squared_error,
        desired_scale_correlation,
        event_positive_fraction: positives.map(|value| value as f32 / batch.rows as f32),
        event_precision: std::array::from_fn(|event| {
            true_positives[event] as f32 / predicted_positives[event].max(1) as f32
        }),
        event_recall: std::array::from_fn(|event| {
            true_positives[event] as f32 / positives[event].max(1) as f32
        }),
    })
}

#[cfg(all(test, feature = "backend_ndarray"))]
fn logistic(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn run_rollout_experiment(
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
) -> AutomataResult<AdaptiveRolloutExperimentReport> {
    if config.rollout.particles < model.config.min_leaves
        || config.rollout.particles > model.config.max_leaves
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive rollout experiment particles {} outside configured leaf range {}..={} ",
            config.rollout.particles, model.config.min_leaves, model.config.max_leaves
        )));
    }
    let particles = crate::adaptive::seed_adaptive_particles_scaled(
        model,
        config.rollout.particles,
        config.seed,
        ParticleSeed::UniformCircle,
        config.rollout.seed_scale,
        config.rollout.total_measure,
        model.rule.config.eps0.clamp(
            model.config.perception.min_bandwidth,
            model.config.perception.max_bandwidth,
        ),
    )?;
    let initial_leaves = particles.len();
    let initial_measure = particles.total_measure();
    let started = Instant::now();
    let trace = run_adaptive_rollout(model, particles, config.rollout.rollout)?;
    let elapsed = started.elapsed();
    let final_measure = trace.particles.total_measure();
    let total_split_events = trace.metrics.iter().map(|row| row.split_events).sum();
    let total_merge_events = trace.metrics.iter().map(|row| row.merge_events).sum();
    let mean_accepted_messages = trace
        .metrics
        .iter()
        .map(|row| row.accepted_messages as f64)
        .sum::<f64>()
        / trace.metrics.len().max(1) as f64;
    let moment_fallback_fraction = trace
        .metrics
        .iter()
        .map(|row| row.moment_fallback_fraction as f64)
        .sum::<f64>()
        / trace.metrics.len().max(1) as f64;
    let particle_steps = trace
        .metrics
        .iter()
        .map(|row| row.leaf_count)
        .sum::<usize>();
    let target_leaves = model.config.target_leaves;
    let final_leaves = trace.particles.len();
    let minimum_leaves = trace
        .metrics
        .iter()
        .map(|row| row.leaf_count)
        .chain(std::iter::once(initial_leaves))
        .min()
        .unwrap_or(initial_leaves);
    let maximum_leaves = trace
        .metrics
        .iter()
        .map(|row| row.leaf_count)
        .chain(std::iter::once(initial_leaves))
        .max()
        .unwrap_or(initial_leaves);
    let final_footprints = trace
        .particles
        .represented_measure
        .iter()
        .map(|measure| {
            crate::adaptive::material_footprint_radius(*measure, model.config.spatial_dims)
        })
        .collect::<Vec<_>>();
    let final_mean_footprint =
        final_footprints.iter().sum::<f32>() / final_footprints.len().max(1) as f32;
    let final_footprint_coefficient_of_variation = (final_footprints
        .iter()
        .map(|value| (value - final_mean_footprint).powi(2))
        .sum::<f32>()
        / final_footprints.len().max(1) as f32)
        .sqrt()
        / final_mean_footprint.max(f32::MIN_POSITIVE);
    let final_scale_metrics = trace.metrics.last();
    Ok(AdaptiveRolloutExperimentReport {
        bandwidth_adaptation_requested: config.rollout.rollout.bandwidth_adaptation_enabled,
        bandwidth_adaptation_active: config.rollout.rollout.bandwidth_adaptation_enabled
            && model.config.supports_bandwidth_adaptation(),
        topology_enabled: config.rollout.rollout.topology_enabled,
        initial_leaves,
        final_leaves,
        target_leaves,
        target_leaf_relative_error: final_leaves.abs_diff(target_leaves) as f64
            / target_leaves as f64,
        minimum_leaves,
        maximum_leaves,
        final_min_footprint: final_footprints
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        final_max_footprint: final_footprints
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        final_mean_footprint,
        final_footprint_coefficient_of_variation,
        final_occupied_material_scale_bins: final_scale_metrics
            .map_or(0, |metrics| metrics.occupied_material_scale_bins),
        final_fractional_material_scale_fraction: final_scale_metrics
            .map_or(0.0, |metrics| metrics.fractional_material_scale_fraction),
        final_dyadic_scale_quantization_rmse_octaves: final_scale_metrics.map_or(0.0, |metrics| {
            metrics.dyadic_scale_quantization_rmse_octaves
        }),
        final_min_generation: trace
            .particles
            .generation
            .iter()
            .copied()
            .min()
            .unwrap_or_default(),
        final_max_generation: trace
            .particles
            .generation
            .iter()
            .copied()
            .max()
            .unwrap_or_default(),
        initial_measure,
        final_measure,
        measure_relative_drift: (final_measure - initial_measure).abs()
            / initial_measure.abs().max(f64::MIN_POSITIVE),
        total_split_events,
        total_merge_events,
        mean_event_state_transfer_rms: event_transfer_mean(&trace.metrics),
        max_event_state_transfer_rms: trace
            .metrics
            .iter()
            .map(|row| row.max_event_state_transfer_rms)
            .fold(0.0_f32, f32::max),
        max_split_probability: trace
            .metrics
            .iter()
            .map(|row| row.max_split_probability)
            .fold(0.0_f32, f32::max),
        max_merge_probability: trace
            .metrics
            .iter()
            .map(|row| row.max_merge_probability)
            .fold(0.0_f32, f32::max),
        max_compatible_merge_probability: trace
            .metrics
            .iter()
            .map(|row| row.max_compatible_merge_probability)
            .fold(0.0_f32, f32::max),
        max_eligible_split_candidates: trace
            .metrics
            .iter()
            .map(|row| row.eligible_split_candidates)
            .max()
            .unwrap_or_default(),
        max_eligible_merge_clusters: trace
            .metrics
            .iter()
            .map(|row| row.eligible_merge_clusters)
            .max()
            .unwrap_or_default(),
        minimum_desired_footprint_ratio: trace
            .metrics
            .iter()
            .filter(|row| row.min_desired_footprint_ratio > 0.0)
            .map(|row| row.min_desired_footprint_ratio)
            .fold(f32::INFINITY, f32::min),
        maximum_desired_footprint_ratio: trace
            .metrics
            .iter()
            .map(|row| row.max_desired_footprint_ratio)
            .fold(0.0_f32, f32::max),
        mean_accepted_messages,
        moment_fallback_fraction,
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
        particle_steps_per_second: particle_steps as f64
            / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        mean_perception_ms: mean_metric(&trace.metrics, |row| row.perception_ms),
        mean_controller_ms: mean_metric(&trace.metrics, |row| row.controller_ms),
        mean_local_rule_ms: mean_metric(&trace.metrics, |row| row.local_rule_ms),
        mean_proxy_rule_ms: mean_metric(&trace.metrics, |row| row.proxy_rule_ms),
        mean_integration_ms: mean_metric(&trace.metrics, |row| row.integration_ms),
        mean_topology_ms: mean_metric(&trace.metrics, |row| row.topology_ms),
        mean_total_step_ms: mean_metric(&trace.metrics, |row| row.total_ms),
    })
}

#[derive(Default)]
struct TaskQualityPrecomputed {
    teacher: Option<crate::rollout::RolloutTrace>,
    regular_base: Option<crate::rollout::RolloutTrace>,
    regular_budget: Option<crate::rollout::RolloutTrace>,
    adaptive_fine_fixed: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_budget_frozen_base: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_budget_local_only: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_budget_fixed: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_budget_fixed_no_bandwidth: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_training_policy: Option<crate::adaptive::AdaptiveRolloutTrace>,
    adaptive_deployment: Option<crate::adaptive::AdaptiveRolloutTrace>,
}

struct TaskQualityModels {
    training_policy: AdaptiveNpaModel,
    frozen_base: AdaptiveNpaModel,
    local_only: AdaptiveNpaModel,
    deployment: AdaptiveNpaModel,
}

fn task_quality_models(model: &AdaptiveNpaModel, regular_base: &NpaModel) -> TaskQualityModels {
    let mut training_policy = model.clone();
    training_policy.deployment_rule = None;
    training_policy.deployment_local_rule = None;
    let mut frozen_base = training_policy.clone();
    frozen_base.rule = regular_base.clone();
    disable_local_residual(&mut frozen_base);
    frozen_base.proxy_rule = None;
    frozen_base.config.proxy.enabled = false;
    let mut local_only = training_policy.clone();
    local_only.proxy_rule = None;
    local_only.config.proxy.enabled = false;
    let mut deployment = model.clone();
    if deployment.deployment_rule.is_some() {
        disable_local_residual(&mut deployment);
        deployment.proxy_rule = None;
        deployment.config.proxy.enabled = false;
    }
    TaskQualityModels {
        training_policy,
        frozen_base,
        local_only,
        deployment,
    }
}

fn disable_local_residual(model: &mut AdaptiveNpaModel) {
    model.local_residual_rule = None;
    model.config.compatible_residual_material_features = false;
    model.closure_mode_rule = None;
    model.closure_basis_rule = None;
    model.config.closure_recurrent_mode = false;
}

#[cfg(feature = "gpu_wgpu")]
fn precompute_task_quality_wgpu(
    executor: &RestrictionRolloutExecutor,
    teacher: &NpaModel,
    regular_base: &NpaModel,
    teacher_grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    seeds: &[u64],
) -> AutomataResult<Vec<TaskQualityPrecomputed>> {
    let total_started = Instant::now();
    let quality = &config.task_quality;
    let fine_particles = task_quality_fine_particle_count(model, config);
    let target_particles = model.config.target_leaves;
    let initial_particles = model.config.initial_leaf_count();
    let fixed = |fixed_model: &NpaModel, particle_count: usize| {
        run_fixed_rollouts_wgpu(
            executor,
            fixed_model,
            teacher_grid,
            seeds,
            particle_count,
            quality.rollout_steps,
            quality.update_prob,
            config.multiscale_training.seed_scale,
        )
    };
    let teacher_traces = timed_wgpu_precompute("teacher-4096", || fixed(teacher, fine_particles))?;
    let regular_base_traces = if same_npa_rule(teacher, regular_base) {
        teacher_traces.clone()
    } else {
        timed_wgpu_precompute("regular-base-4096", || fixed(regular_base, fine_particles))?
    };
    let regular_budget_traces = if target_particles == fine_particles {
        regular_base_traces.clone()
    } else {
        timed_wgpu_precompute("regular-budget", || fixed(regular_base, target_particles))?
    };

    let task_models = task_quality_models(model, regular_base);
    let topology_control = quality.topology_control;
    let target_render_oracle =
        quality.restriction_policy == super::AdaptiveTaskRestrictionPolicy::TargetRenderOracle;
    let rollout =
        |topology_enabled, bandwidth_adaptation_enabled| crate::adaptive::AdaptiveRolloutConfig {
            steps: quality.rollout_steps,
            dt: 1.0,
            update_prob: quality.update_prob,
            seed: seeds.first().copied().unwrap_or(quality.seed),
            bandwidth_adaptation_enabled,
            topology_enabled,
            snapshot_interval: quality.rollout_steps,
        };
    let adaptive = |evaluation_model: &AdaptiveNpaModel,
                    particle_count: usize,
                    topology_enabled: bool,
                    bandwidth_adaptation_enabled: bool| {
        run_task_quality_rollouts_wgpu(
            executor,
            evaluation_model,
            teacher_grid,
            seeds,
            particle_count,
            rollout(topology_enabled, bandwidth_adaptation_enabled),
            topology_control,
            if topology_enabled {
                quality.restriction_policy
            } else {
                super::AdaptiveTaskRestrictionPolicy::DynamicsDetail
            },
            config.multiscale_training.seed_scale,
            config.multiscale_training.total_measure,
            config.multiscale_training.bandwidth,
        )
    };

    let training_initial = if target_render_oracle {
        None
    } else {
        Some(timed_wgpu_precompute("adaptive-training-topology", || {
            adaptive(
                &task_models.training_policy,
                initial_particles,
                true,
                quality.bandwidth_adaptation_enabled,
            )
        })?)
    };
    let training_fine = if initial_particles == fine_particles && !target_render_oracle {
        None
    } else {
        Some(timed_wgpu_precompute("adaptive-training-fine", || {
            adaptive(
                &task_models.training_policy,
                fine_particles,
                false,
                quality.bandwidth_adaptation_enabled,
            )
        })?)
    };
    let frozen_budget = timed_wgpu_precompute("adaptive-frozen-budget", || {
        adaptive(
            &task_models.frozen_base,
            target_particles,
            false,
            quality.bandwidth_adaptation_enabled,
        )
    })?;
    let local_budget =
        if same_wgpu_task_dynamics(&task_models.frozen_base, &task_models.local_only)? {
            eprintln!("adaptive WGPU precompute adaptive-local-budget: reused frozen-budget");
            frozen_budget.clone()
        } else {
            timed_wgpu_precompute("adaptive-local-budget", || {
                adaptive(
                    &task_models.local_only,
                    target_particles,
                    false,
                    quality.bandwidth_adaptation_enabled,
                )
            })?
        };
    let training_budget =
        if same_wgpu_task_dynamics(&task_models.local_only, &task_models.training_policy)? {
            eprintln!("adaptive WGPU precompute adaptive-training-budget: reused local-budget");
            local_budget.clone()
        } else if same_wgpu_task_dynamics(&task_models.frozen_base, &task_models.training_policy)? {
            eprintln!("adaptive WGPU precompute adaptive-training-budget: reused frozen-budget");
            frozen_budget.clone()
        } else {
            timed_wgpu_precompute("adaptive-training-budget", || {
                adaptive(
                    &task_models.training_policy,
                    target_particles,
                    false,
                    quality.bandwidth_adaptation_enabled,
                )
            })?
        };
    let training_budget_no_bandwidth =
        if quality.bandwidth_adaptation_enabled && model.config.supports_bandwidth_adaptation() {
            timed_wgpu_precompute("adaptive-training-budget-no-bandwidth", || {
                adaptive(&task_models.training_policy, target_particles, false, false)
            })?
        } else {
            training_budget.clone()
        };
    let deployment =
        if same_adaptive_rollout_model(&task_models.training_policy, &task_models.deployment) {
            None
        } else {
            Some(timed_wgpu_precompute("adaptive-deployment", || {
                adaptive(
                    &task_models.deployment,
                    initial_particles,
                    true,
                    quality.bandwidth_adaptation_enabled,
                )
            })?)
        };

    eprintln!(
        "adaptive WGPU precompute total: {:.3}s",
        total_started.elapsed().as_secs_f64()
    );

    let expected = seeds.len();
    let lengths = [
        teacher_traces.len(),
        regular_base_traces.len(),
        regular_budget_traces.len(),
        training_initial.as_ref().map_or(expected, Vec::len),
        frozen_budget.len(),
        local_budget.len(),
        training_budget.len(),
        training_budget_no_bandwidth.len(),
        training_fine.as_ref().map_or(expected, Vec::len),
        deployment.as_ref().map_or(expected, Vec::len),
    ];
    if lengths.iter().any(|length| *length != expected) {
        return Err(AutomataError::InvalidArgument(format!(
            "batched task rollout count mismatch: expected {expected}, got {lengths:?}",
        )));
    }

    (0..expected)
        .map(|index| {
            let training_topology = training_initial
                .as_ref()
                .map(|traces| {
                    traces[index].topology.clone().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "batched training-policy rollout omitted final topology".to_owned(),
                        )
                    })
                })
                .transpose()?;
            let deployment_topology = if let Some(deployment) = &deployment {
                Some(deployment[index].topology.clone().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "batched deployment rollout omitted final topology".to_owned(),
                    )
                })?)
            } else {
                None
            };
            Ok(TaskQualityPrecomputed {
                teacher: Some(teacher_traces[index].clone()),
                regular_base: Some(regular_base_traces[index].clone()),
                regular_budget: Some(regular_budget_traces[index].clone()),
                adaptive_fine_fixed: Some(training_fine.as_ref().map_or_else(
                    || {
                        training_initial
                            .as_ref()
                            .expect("non-oracle fine trace reuses the topology batch")[index]
                            .fixed
                            .clone()
                    },
                    |traces| traces[index].fixed.clone(),
                )),
                adaptive_budget_frozen_base: Some(frozen_budget[index].fixed.clone()),
                adaptive_budget_local_only: Some(local_budget[index].fixed.clone()),
                adaptive_budget_fixed: Some(training_budget[index].fixed.clone()),
                adaptive_budget_fixed_no_bandwidth: Some(
                    training_budget_no_bandwidth[index].fixed.clone(),
                ),
                adaptive_training_policy: training_topology,
                adaptive_deployment: deployment_topology,
            })
        })
        .collect()
}

fn run_task_quality_experiment(
    teacher: &NpaModel,
    regular_base: &NpaModel,
    teacher_grid: &HashGridConfig,
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
    _executor: Option<&RestrictionRolloutExecutor>,
) -> AutomataResult<AdaptiveTaskQualityReport> {
    run_task_quality_experiment_with_precomputed(
        TaskQualityContext {
            teacher,
            regular_base,
            teacher_grid,
            model,
            config,
            executor: _executor,
        },
        None,
        true,
    )
}

#[derive(Clone, Copy)]
struct TaskQualityContext<'a> {
    teacher: &'a NpaModel,
    regular_base: &'a NpaModel,
    teacher_grid: &'a HashGridConfig,
    model: &'a AdaptiveNpaModel,
    config: &'a AdaptiveExperimentConfig,
    executor: Option<&'a RestrictionRolloutExecutor>,
}

fn run_task_quality_experiment_with_precomputed(
    context: TaskQualityContext<'_>,
    mut precomputed: Option<TaskQualityPrecomputed>,
    structural_audit: bool,
) -> AutomataResult<AdaptiveTaskQualityReport> {
    let TaskQualityContext {
        teacher,
        regular_base,
        teacher_grid,
        model,
        config,
        executor: _executor,
    } = context;
    let started = Instant::now();
    let quality = &config.task_quality;
    let fine_particles = task_quality_fine_particle_count(model, config);
    let target = crate::target2d::load_target_image_2d_upstream(
        &quality.target_image,
        0.05,
        fine_particles,
        None,
    )?;
    let render_config = crate::target2d::Target2dLossConfig {
        image_size: quality.image_size,
        ..crate::target2d::Target2dLossConfig::default()
    };
    let target_render = crate::target2d::render_target_2d_splat(&target, render_config)?;
    let fixed_rollout = |fixed_model: &NpaModel, particle_count: usize| {
        let rollout = RolloutConfig {
            batch_size: 1,
            particle_count,
            steps: quality.rollout_steps,
            dt: 1.0,
            update_prob: quality.update_prob,
            seed: quality.seed,
            seed_scale: config.multiscale_training.seed_scale,
        };
        #[cfg(feature = "gpu_wgpu")]
        if let Some(executor) = _executor {
            return run_fixed_rollout_wgpu(executor, fixed_model, teacher_grid, &rollout);
        }
        run_rollout_with_stable_material_masks(
            fixed_model,
            teacher_grid,
            &rollout,
            ParticleSeed::UniformCircle,
        )
    };
    let teacher_trace = match precomputed.as_mut().and_then(|value| value.teacher.take()) {
        Some(trace) => trace,
        None => fixed_rollout(teacher, fine_particles)?,
    };
    let target_center = target.mean_position();
    let teacher_render = crate::target2d::render_rollout_2d_splat(
        &teacher_trace.positions,
        &teacher_trace.states,
        teacher.config.state_dims,
        target.pixel_size,
        render_config,
        Some(target_center),
        1.0,
    )?;
    let regular_base_trace = match precomputed
        .as_mut()
        .and_then(|value| value.regular_base.take())
    {
        Some(trace) => trace,
        None if same_npa_rule(teacher, regular_base) => teacher_trace.clone(),
        None => fixed_rollout(regular_base, fine_particles)?,
    };
    let regular_base_render = crate::target2d::render_rollout_2d_splat(
        &regular_base_trace.positions,
        &regular_base_trace.states,
        model.rule.config.state_dims,
        target.pixel_size,
        render_config,
        Some(target_center),
        1.0,
    )?;
    let fine_measure = config.multiscale_training.total_measure / fine_particles as f32;
    let (regular_matched_budget_render, regular_material_matched_budget_render) =
        if model.config.target_leaves == fine_particles {
            (regular_base_render.clone(), regular_base_render.clone())
        } else {
            let trace = match precomputed
                .as_mut()
                .and_then(|value| value.regular_budget.take())
            {
                Some(trace) => trace,
                None => fixed_rollout(regular_base, model.config.target_leaves)?,
            };
            let fixed_footprint_render = crate::target2d::render_rollout_2d_splat(
                &trace.positions,
                &trace.states,
                model.rule.config.state_dims,
                target.pixel_size,
                render_config,
                Some(target_center),
                1.0,
            )?;
            let budget_measure =
                config.multiscale_training.total_measure / model.config.target_leaves as f32;
            let footprint_scale = crate::adaptive::material_footprint_radius(budget_measure, 2)
                / crate::adaptive::material_footprint_radius(fine_measure, 2);
            let material_footprint_render = crate::target2d::render_rollout_2d_splat(
                &trace.positions,
                &trace.states,
                model.rule.config.state_dims,
                target.pixel_size * footprint_scale,
                render_config,
                Some(target_center),
                1.0,
            )?;
            (fixed_footprint_render, material_footprint_render)
        };

    let task_models = task_quality_models(model, regular_base);
    let training_policy_model = &task_models.training_policy;
    let frozen_base_model = &task_models.frozen_base;
    let local_only_model = &task_models.local_only;
    let deployment_model = &task_models.deployment;

    let evaluate_adaptive =
        |evaluation_model: &AdaptiveNpaModel,
         particle_count: usize,
         topology_enabled: bool,
         bandwidth_adaptation_enabled: bool,
         precomputed_trace: Option<crate::adaptive::AdaptiveRolloutTrace>| {
            let initial_measure = precomputed_trace
                .as_ref()
                .and_then(|trace| trace.snapshots.first())
                .map(|snapshot| snapshot.particles.total_measure());
            let initial = if precomputed_trace.is_none() {
                Some(crate::adaptive::seed_adaptive_particles_scaled(
                    evaluation_model,
                    particle_count,
                    quality.seed,
                    ParticleSeed::UniformCircle,
                    config.multiscale_training.seed_scale,
                    config.multiscale_training.total_measure,
                    config.multiscale_training.bandwidth,
                )?)
            } else {
                None
            };
            let initial_measure = initial_measure
                .or_else(|| initial.as_ref().map(|particles| particles.total_measure()))
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "adaptive task trace omitted its initial material snapshot".to_owned(),
                    )
                })?;
            let topology_control = match (topology_enabled, quality.topology_control) {
                (false, _) | (true, AdaptiveTopologyControl::Learned) => {
                    AdaptiveTopologyControl::Learned
                }
                (true, AdaptiveTopologyControl::LearnedRefinementDefect) => {
                    AdaptiveTopologyControl::LearnedRefinementDefect
                }
                (true, AdaptiveTopologyControl::LocalDetailOracle) => {
                    AdaptiveTopologyControl::LocalDetailOracle
                }
                (true, AdaptiveTopologyControl::PairedLocalDetail) => {
                    AdaptiveTopologyControl::PairedLocalDetail
                }
                (true, AdaptiveTopologyControl::ContinuousLocalDetail) => {
                    AdaptiveTopologyControl::ContinuousLocalDetail
                }
                (true, AdaptiveTopologyControl::RefinementDefectOracle) => {
                    AdaptiveTopologyControl::RefinementDefectOracle
                }
            };
            let rollout = crate::adaptive::AdaptiveRolloutConfig {
                steps: quality.rollout_steps,
                dt: 1.0,
                update_prob: quality.update_prob,
                seed: quality.seed,
                bandwidth_adaptation_enabled,
                topology_enabled,
                snapshot_interval: quality.rollout_steps,
            };
            let restriction_policy = if topology_enabled {
                quality.restriction_policy
            } else {
                super::AdaptiveTaskRestrictionPolicy::DynamicsDetail
            };
            #[cfg(feature = "gpu_wgpu")]
            let trace = if let Some(trace) = precomputed_trace {
                trace
            } else if let Some(executor) = _executor
                && restriction_policy != super::AdaptiveTaskRestrictionPolicy::TargetRenderOracle
            {
                run_task_quality_rollout_wgpu(
                    executor,
                    evaluation_model,
                    teacher_grid,
                    particle_count,
                    rollout,
                    topology_control,
                    restriction_policy,
                    config.multiscale_training.seed_scale,
                    config.multiscale_training.total_measure,
                    config.multiscale_training.bandwidth,
                )?
            } else {
                run_task_quality_rollout(
                    evaluation_model,
                    initial.ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "adaptive CPU rollout omitted its initial particles".to_owned(),
                        )
                    })?,
                    rollout,
                    topology_control,
                    restriction_policy,
                    quality.render_decoder,
                    quality.render_compactness,
                    &target,
                    render_config,
                    fine_measure,
                )?
            };
            #[cfg(not(feature = "gpu_wgpu"))]
            let trace = if let Some(trace) = precomputed_trace {
                trace
            } else {
                run_task_quality_rollout(
                    evaluation_model,
                    initial.ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "adaptive CPU rollout omitted its initial particles".to_owned(),
                        )
                    })?,
                    rollout,
                    topology_control,
                    restriction_policy,
                    quality.render_decoder,
                    quality.render_compactness,
                    &target,
                    render_config,
                    fine_measure,
                )?
            };
            let (render, render_primitives) = match quality.render_decoder {
                AdaptiveRenderDecoder::IsotropicMaterialGaussian => (
                    crate::target2d::render_adaptive_rollout_2d_runtime_isotropic_splat(
                        &trace.particles,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                    )?,
                    trace.particles.len(),
                ),
                AdaptiveRenderDecoder::MomentGaussian => (
                    crate::target2d::render_adaptive_rollout_2d_splat(
                        &trace.particles,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                    )?,
                    trace.particles.len(),
                ),
                AdaptiveRenderDecoder::AffineMomentGaussian => (
                    crate::target2d::render_adaptive_rollout_2d_affine_splat(
                        &trace.particles,
                        &trace.particles.state_jacobian,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                    )?,
                    trace.particles.len(),
                ),
                AdaptiveRenderDecoder::CompactMomentGaussian => (
                    crate::target2d::render_adaptive_rollout_2d_compact_splat(
                        &trace.particles,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                        quality.render_compactness,
                    )?,
                    trace.particles.len(),
                ),
                AdaptiveRenderDecoder::CanonicalAffineQuadrature => {
                    crate::target2d::render_adaptive_rollout_2d_quadrature_splat(
                        &trace.particles,
                        &trace.particles.state_jacobian,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                    )?
                }
                AdaptiveRenderDecoder::RetainedFineQuadrature => {
                    crate::target2d::render_adaptive_rollout_2d_retained_quadrature_splat(
                        &trace.particles,
                        fine_measure,
                        target.pixel_size,
                        render_config,
                        Some(target_center),
                    )?
                }
                AdaptiveRenderDecoder::PersistentModeQuadrature => {
                    let quadrature = crate::adaptive::dynamics::persistent_quadrature_particle_set(
                        evaluation_model,
                        &trace.particles,
                    )?;
                    let primitive_count = quadrature.len();
                    (
                        crate::target2d::render_adaptive_rollout_2d_splat(
                            &quadrature,
                            fine_measure,
                            target.pixel_size,
                            render_config,
                            Some(target_center),
                        )?,
                        primitive_count,
                    )
                }
            };
            Ok::<_, AutomataError>((trace, render, initial_measure, render_primitives))
        };
    let adaptive_fine_fixed_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_fine_fixed.take());
    let (_, adaptive_fine_fixed_render, _, _) = evaluate_adaptive(
        training_policy_model,
        fine_particles,
        false,
        quality.bandwidth_adaptation_enabled,
        adaptive_fine_fixed_precomputed,
    )?;
    let adaptive_budget_frozen_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_budget_frozen_base.take());
    let (_, adaptive_budget_frozen_base_render, _, _) = evaluate_adaptive(
        frozen_base_model,
        model.config.target_leaves,
        false,
        quality.bandwidth_adaptation_enabled,
        adaptive_budget_frozen_precomputed,
    )?;
    let adaptive_budget_local_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_budget_local_only.take());
    let (_, adaptive_budget_local_only_render, _, _) = evaluate_adaptive(
        local_only_model,
        model.config.target_leaves,
        false,
        quality.bandwidth_adaptation_enabled,
        adaptive_budget_local_precomputed,
    )?;
    let adaptive_budget_fixed_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_budget_fixed.take());
    let (adaptive_budget_fixed_trace, adaptive_budget_fixed_render, _, _) = evaluate_adaptive(
        training_policy_model,
        model.config.target_leaves,
        false,
        quality.bandwidth_adaptation_enabled,
        adaptive_budget_fixed_precomputed,
    )?;
    let adaptive_budget_fixed_no_bandwidth_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_budget_fixed_no_bandwidth.take());
    let (_, adaptive_budget_fixed_no_bandwidth_render, _, _) = evaluate_adaptive(
        training_policy_model,
        model.config.target_leaves,
        false,
        false,
        adaptive_budget_fixed_no_bandwidth_precomputed,
    )?;
    let adaptive_initial_particles = model.config.initial_leaf_count();
    let adaptive_training_precomputed = precomputed
        .as_mut()
        .and_then(|value| value.adaptive_training_policy.take());
    let adaptive_training_policy = evaluate_adaptive(
        training_policy_model,
        adaptive_initial_particles,
        true,
        quality.bandwidth_adaptation_enabled,
        adaptive_training_precomputed,
    )?;
    let adaptive_training_policy_render = adaptive_training_policy.1.clone();
    let (adaptive_trace, adaptive_render, initial_measure, adaptive_render_primitives) =
        if same_adaptive_rollout_model(training_policy_model, deployment_model) {
            adaptive_training_policy
        } else {
            let adaptive_deployment_precomputed = precomputed
                .as_mut()
                .and_then(|value| value.adaptive_deployment.take());
            evaluate_adaptive(
                deployment_model,
                adaptive_initial_particles,
                true,
                quality.bandwidth_adaptation_enabled,
                adaptive_deployment_precomputed,
            )?
        };
    let (dynamics_semantics, adaptive_dynamics_particles, adaptive_interaction_particles) =
        match model.config.coarse_dynamics {
            crate::adaptive::AdaptiveCoarseDynamics::RepresentedMeasure => (
                AdaptiveDynamicsSemantics::ActiveLeaves,
                adaptive_trace.particles.len(),
                adaptive_trace.particles.len(),
            ),
            crate::adaptive::AdaptiveCoarseDynamics::FineQuadrature => (
                AdaptiveDynamicsSemantics::ActiveLeaves,
                adaptive_trace.particles.len(),
                crate::adaptive::dynamics::quadrature_particle_count(
                    model,
                    &adaptive_trace.particles,
                    false,
                )?,
            ),
            crate::adaptive::AdaptiveCoarseDynamics::PersistentFineQuadrature => {
                let modes = crate::adaptive::dynamics::quadrature_particle_count(
                    model,
                    &adaptive_trace.particles,
                    true,
                )?;
                (
                    AdaptiveDynamicsSemantics::PersistentHiddenFineModes,
                    modes,
                    modes,
                )
            }
        };
    let footprints = adaptive_trace
        .particles
        .represented_measure
        .iter()
        .map(|measure| {
            crate::adaptive::material_footprint_radius(*measure, model.config.spatial_dims)
        })
        .collect::<Vec<_>>();
    let footprint_mean = mean_f32(&footprints);
    let footprint_variance = footprints
        .iter()
        .map(|value| (*value - footprint_mean).powi(2))
        .sum::<f32>()
        / footprints.len().max(1) as f32;
    let render_footprints = &adaptive_trace.particles.render_footprint;
    let render_to_material = render_footprints
        .iter()
        .zip(&footprints)
        .map(|(render, material)| render / material.max(f32::MIN_POSITIVE))
        .collect::<Vec<_>>();
    let render_targets = adaptive_trace
        .particles
        .represented_measure
        .iter()
        .map(|measure| {
            model
                .config
                .render_footprint(crate::adaptive::material_footprint_radius(
                    *measure,
                    model.config.spatial_dims,
                ))
        })
        .collect::<Vec<_>>();
    let max_render_target_relative_error = render_footprints
        .iter()
        .zip(&render_targets)
        .map(|(render, target)| (render - target).abs() / target.max(f32::MIN_POSITIVE))
        .fold(0.0_f32, f32::max);
    let reference_footprint = model.config.reference_footprint;
    let scale_tolerance = reference_footprint * 1.0e-4;
    let total_represented_measure = adaptive_trace
        .particles
        .represented_measure
        .iter()
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let mut scale_counts = [0_usize; 3];
    let mut scale_measure = [0.0_f32; 3];
    for (footprint, measure) in footprints
        .iter()
        .zip(&adaptive_trace.particles.represented_measure)
    {
        let scale = if *footprint < reference_footprint - scale_tolerance {
            0
        } else if *footprint > reference_footprint + scale_tolerance {
            2
        } else {
            1
        };
        scale_counts[scale] += 1;
        scale_measure[scale] += *measure;
    }
    let teacher_target_psnr = psnr(composited_mse(&teacher_render, &target_render));
    let regular_base_target_psnr = psnr(composited_mse(&regular_base_render, &target_render));
    let regular_matched_budget_target_psnr = psnr(composited_mse(
        &regular_matched_budget_render,
        &target_render,
    ));
    let regular_material_matched_budget_target_psnr = psnr(composited_mse(
        &regular_material_matched_budget_render,
        &target_render,
    ));
    let adaptive_fine_fixed_target_psnr =
        psnr(composited_mse(&adaptive_fine_fixed_render, &target_render));
    let adaptive_budget_fixed_target_psnr = psnr(composited_mse(
        &adaptive_budget_fixed_render,
        &target_render,
    ));
    let adaptive_budget_fixed_no_bandwidth_target_psnr = psnr(composited_mse(
        &adaptive_budget_fixed_no_bandwidth_render,
        &target_render,
    ));
    let adaptive_budget_frozen_base_target_psnr = psnr(composited_mse(
        &adaptive_budget_frozen_base_render,
        &target_render,
    ));
    let adaptive_budget_local_only_target_psnr = psnr(composited_mse(
        &adaptive_budget_local_only_render,
        &target_render,
    ));
    let adaptive_target_psnr = psnr(composited_mse(&adaptive_render, &target_render));
    let adaptive_training_policy_target_psnr = psnr(composited_mse(
        &adaptive_training_policy_render,
        &target_render,
    ));
    let final_measure = adaptive_trace.particles.total_measure();
    let (density_alignment, budget_fixed_density_alignment) = if structural_audit {
        (
            task_density_alignment(model, &adaptive_trace.particles)?,
            task_density_alignment(model, &adaptive_budget_fixed_trace.particles)?,
        )
    } else {
        (Default::default(), Default::default())
    };
    let adaptive_refinement_defect_relative_gain = (budget_fixed_density_alignment
        .mean_refinement_defect
        - density_alignment.mean_refinement_defect)
        / budget_fixed_density_alignment
            .mean_refinement_defect
            .max(f32::MIN_POSITIVE);
    let final_scale_metrics = adaptive_trace.metrics.last();
    Ok(AdaptiveTaskQualityReport {
        seed: quality.seed,
        structural_audit_performed: structural_audit,
        topology_control: quality.topology_control,
        rollout_steps: quality.rollout_steps,
        bandwidth_adaptation_enabled: quality.bandwidth_adaptation_enabled,
        bandwidth_adaptation_active: quality.bandwidth_adaptation_enabled
            && model.config.supports_bandwidth_adaptation(),
        teacher_particles: fine_particles,
        adaptive_initial_particles,
        adaptive_target_particles: model.config.target_leaves,
        adaptive_final_particles: adaptive_trace.particles.len(),
        adaptive_min_particles: adaptive_trace
            .metrics
            .iter()
            .map(|metrics| metrics.leaf_count)
            .min()
            .unwrap_or(adaptive_initial_particles),
        adaptive_max_particles: adaptive_trace
            .metrics
            .iter()
            .map(|metrics| metrics.leaf_count)
            .max()
            .unwrap_or(adaptive_initial_particles),
        render_decoder: quality.render_decoder,
        restriction_policy: quality.restriction_policy,
        adaptive_render_primitives,
        adaptive_dynamics_particles,
        dynamics_semantics,
        adaptive_interaction_particles,
        teacher_target_composited_psnr_db: teacher_target_psnr,
        regular_base_target_composited_psnr_db: regular_base_target_psnr,
        regular_matched_budget_target_composited_psnr_db: regular_matched_budget_target_psnr,
        regular_material_matched_budget_target_composited_psnr_db:
            regular_material_matched_budget_target_psnr,
        adaptive_fine_fixed_target_composited_psnr_db: adaptive_fine_fixed_target_psnr,
        adaptive_fine_fixed_teacher_composited_psnr_db: psnr(composited_mse(
            &adaptive_fine_fixed_render,
            &teacher_render,
        )),
        adaptive_fine_fixed_teacher_psnr_gap_db: teacher_target_psnr
            - adaptive_fine_fixed_target_psnr,
        adaptive_budget_frozen_base_target_composited_psnr_db:
            adaptive_budget_frozen_base_target_psnr,
        adaptive_budget_local_only_target_composited_psnr_db:
            adaptive_budget_local_only_target_psnr,
        adaptive_budget_fixed_target_composited_psnr_db: adaptive_budget_fixed_target_psnr,
        adaptive_budget_fixed_no_bandwidth_target_composited_psnr_db:
            adaptive_budget_fixed_no_bandwidth_target_psnr,
        bandwidth_adaptation_target_psnr_gain_db: adaptive_budget_fixed_target_psnr
            - adaptive_budget_fixed_no_bandwidth_target_psnr,
        adaptive_budget_fixed_teacher_composited_psnr_db: psnr(composited_mse(
            &adaptive_budget_fixed_render,
            &teacher_render,
        )),
        local_residual_target_psnr_gain_db: adaptive_budget_local_only_target_psnr
            - adaptive_budget_frozen_base_target_psnr,
        proxy_residual_target_psnr_gain_db: adaptive_budget_fixed_target_psnr
            - adaptive_budget_local_only_target_psnr,
        adaptive_target_composited_psnr_db: adaptive_target_psnr,
        adaptive_training_policy_target_composited_psnr_db: adaptive_training_policy_target_psnr,
        adaptive_over_regular_base_psnr_gain_db: adaptive_target_psnr - regular_base_target_psnr,
        adaptive_over_regular_matched_budget_psnr_gain_db: adaptive_target_psnr
            - regular_matched_budget_target_psnr,
        adaptive_over_regular_material_matched_budget_psnr_gain_db: adaptive_target_psnr
            - regular_material_matched_budget_target_psnr,
        deployment_over_training_policy_psnr_gain_db: adaptive_target_psnr
            - adaptive_training_policy_target_psnr,
        adaptive_teacher_composited_psnr_db: psnr(composited_mse(
            &adaptive_render,
            &teacher_render,
        )),
        adaptive_teacher_psnr_gap_db: teacher_target_psnr - adaptive_target_psnr,
        teacher_target_density_psnr_db: psnr(density_mse(&teacher_render, &target_render)),
        adaptive_target_density_psnr_db: psnr(density_mse(&adaptive_render, &target_render)),
        final_min_footprint: footprints.iter().copied().fold(f32::INFINITY, f32::min),
        final_max_footprint: footprints.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        final_min_render_footprint: render_footprints
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        final_max_render_footprint: render_footprints
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        final_mean_render_to_material_footprint_ratio: mean_f32(&render_to_material),
        final_max_render_target_relative_error: max_render_target_relative_error,
        final_footprint_coefficient_of_variation: footprint_variance.sqrt()
            / footprint_mean.max(f32::MIN_POSITIVE),
        maximum_footprint_coefficient_of_variation: adaptive_trace
            .metrics
            .iter()
            .map(|metrics| metrics.footprint_coefficient_of_variation)
            .fold(0.0_f32, f32::max),
        final_occupied_material_scale_bins: final_scale_metrics
            .map_or(0, |metrics| metrics.occupied_material_scale_bins),
        final_fractional_material_scale_fraction: final_scale_metrics
            .map_or(0.0, |metrics| metrics.fractional_material_scale_fraction),
        final_dyadic_scale_quantization_rmse_octaves: final_scale_metrics.map_or(0.0, |metrics| {
            metrics.dyadic_scale_quantization_rmse_octaves
        }),
        fine_leaf_count: scale_counts[0],
        reference_leaf_count: scale_counts[1],
        coarse_leaf_count: scale_counts[2],
        fine_represented_measure_fraction: scale_measure[0] / total_represented_measure,
        reference_represented_measure_fraction: scale_measure[1] / total_represented_measure,
        coarse_represented_measure_fraction: scale_measure[2] / total_represented_measure,
        total_split_events: adaptive_trace
            .metrics
            .iter()
            .map(|metrics| metrics.split_events)
            .sum(),
        total_merge_events: adaptive_trace
            .metrics
            .iter()
            .map(|metrics| metrics.merge_events)
            .sum(),
        bootstrap_split_events: adaptive_trace
            .metrics
            .iter()
            .filter(|metrics| metrics.step <= model.config.bootstrap_end_step)
            .map(|metrics| metrics.split_events)
            .sum(),
        restriction_merge_events: adaptive_trace
            .metrics
            .iter()
            .filter(|metrics| model.config.is_scheduled_restriction_step(metrics.step))
            .map(|metrics| metrics.merge_events)
            .sum(),
        steady_split_events: adaptive_trace
            .metrics
            .iter()
            .filter(|metrics| {
                metrics.step > model.config.bootstrap_end_step
                    && !model.config.is_scheduled_restriction_step(metrics.step)
            })
            .map(|metrics| metrics.split_events)
            .sum(),
        steady_merge_events: adaptive_trace
            .metrics
            .iter()
            .filter(|metrics| {
                metrics.step > model.config.bootstrap_end_step
                    && !model.config.is_scheduled_restriction_step(metrics.step)
            })
            .map(|metrics| metrics.merge_events)
            .sum(),
        mean_event_state_transfer_rms: event_transfer_mean(&adaptive_trace.metrics),
        max_event_state_transfer_rms: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.max_event_state_transfer_rms)
            .fold(0.0_f32, f32::max),
        max_split_probability: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.max_split_probability)
            .fold(0.0_f32, f32::max),
        max_merge_probability: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.max_merge_probability)
            .fold(0.0_f32, f32::max),
        max_compatible_merge_probability: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.max_compatible_merge_probability)
            .fold(0.0_f32, f32::max),
        max_eligible_split_candidates: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.eligible_split_candidates)
            .max()
            .unwrap_or_default(),
        max_eligible_merge_clusters: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.eligible_merge_clusters)
            .max()
            .unwrap_or_default(),
        mean_proxy_messages: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.proxy_messages as f32)
            .sum::<f32>()
            / adaptive_trace.metrics.len().max(1) as f32,
        measure_relative_drift: (final_measure - initial_measure).abs()
            / initial_measure.abs().max(f64::MIN_POSITIVE),
        detail_density_correlation: density_alignment.state_detail_correlation,
        high_to_low_detail_footprint_ratio: density_alignment
            .high_to_low_state_detail_footprint_ratio,
        refinement_defect_density_correlation: density_alignment.refinement_defect_correlation,
        low_to_high_refinement_defect_footprint_ratio: density_alignment
            .low_to_high_refinement_defect_footprint_ratio,
        budget_fixed_mean_refinement_defect: budget_fixed_density_alignment.mean_refinement_defect,
        adaptive_mean_refinement_defect: density_alignment.mean_refinement_defect,
        adaptive_refinement_defect_relative_gain,
        controller_oracle_refinement_scale_correlation: density_alignment
            .controller_oracle_scale_correlation,
        oracle_min_desired_footprint_ratio: density_alignment.oracle_min_desired_ratio,
        oracle_max_desired_footprint_ratio: density_alignment.oracle_max_desired_ratio,
        controller_min_desired_footprint_ratio: density_alignment.controller_min_desired_ratio,
        controller_max_desired_footprint_ratio: density_alignment.controller_max_desired_ratio,
        minimum_desired_footprint_ratio: adaptive_trace
            .metrics
            .iter()
            .filter(|row| row.min_desired_footprint_ratio > 0.0)
            .map(|row| row.min_desired_footprint_ratio)
            .fold(f32::INFINITY, f32::min),
        maximum_desired_footprint_ratio: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.max_desired_footprint_ratio)
            .fold(0.0_f32, f32::max),
        adaptive_rollout_elapsed_ms: adaptive_trace
            .metrics
            .last()
            .map_or(0.0, |row| row.total_ms),
        adaptive_topology_elapsed_ms: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.topology_ms)
            .sum(),
        maximum_topology_update_elapsed_ms: adaptive_trace
            .metrics
            .iter()
            .map(|row| row.topology_ms)
            .fold(0.0_f64, f64::max),
        adaptive_topology_updates: adaptive_trace
            .metrics
            .iter()
            .filter(|row| row.topology_ms > 0.0)
            .map(|row| super::AdaptiveTopologyTimingRow {
                step: row.step,
                split_events: row.split_events,
                merge_events: row.merge_events,
                elapsed_ms: row.topology_ms,
            })
            .collect(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn task_quality_fine_particle_count(
    model: &AdaptiveNpaModel,
    config: &AdaptiveExperimentConfig,
) -> usize {
    if config.multiscale_training.enabled {
        config.multiscale_training.fine_particle_count
    } else {
        model.config.bootstrap_fine_leaf_count()
    }
}

fn mean_metric(
    metrics: &[crate::adaptive::AdaptiveStepMetrics],
    value: impl Fn(&crate::adaptive::AdaptiveStepMetrics) -> f64,
) -> f64 {
    metrics.iter().map(value).sum::<f64>() / metrics.len().max(1) as f64
}

fn event_transfer_mean(metrics: &[crate::adaptive::AdaptiveStepMetrics]) -> f32 {
    let (weighted_sum, events) = metrics.iter().fold((0.0_f32, 0_usize), |acc, row| {
        let row_events = row.split_events + row.merge_events;
        (
            acc.0 + row.mean_event_state_transfer_rms * row_events as f32,
            acc.1 + row_events,
        )
    });
    weighted_sum / events.max(1) as f32
}

fn composited_mse(
    prediction: &crate::target2d::Target2dRenderedSplat,
    target: &crate::target2d::Target2dRenderedSplat,
) -> f32 {
    let pixels = prediction.density.len().min(target.density.len());
    let mut squared_error = 0.0;
    for pixel in 0..pixels {
        let prediction_alpha = prediction.density[pixel].clamp(0.0, 1.0);
        let target_alpha = target.density[pixel].clamp(0.0, 1.0);
        for channel in 0..3 {
            let index = pixel * 3 + channel;
            let prediction_value = (prediction.rgb[index] + 1.0 - prediction_alpha).clamp(0.0, 1.0);
            let target_value = (target.rgb[index] + 1.0 - target_alpha).clamp(0.0, 1.0);
            squared_error += (prediction_value - target_value).powi(2);
        }
    }
    squared_error / (pixels * 3).max(1) as f32
}

fn density_mse(
    prediction: &crate::target2d::Target2dRenderedSplat,
    target: &crate::target2d::Target2dRenderedSplat,
) -> f32 {
    prediction
        .density
        .iter()
        .zip(&target.density)
        .map(|(prediction, target)| (prediction.clamp(0.0, 1.0) - target.clamp(0.0, 1.0)).powi(2))
        .sum::<f32>()
        / prediction.density.len().max(1) as f32
}

fn psnr(mse: f32) -> f32 {
    -10.0 * mse.max(1.0e-12).log10()
}

fn mean_f32(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn paper_scope(
    config: &AdaptiveExperimentConfig,
    rule_was_distilled: bool,
    multiscale_was_trained: bool,
) -> String {
    if config.adaptive.closure_recurrent_mode {
        return "experimental recurrent-closure candidate: compact causal geometry/state reconstruction with parity-tested resident WGPU replay; this is not a promoted adaptive NPA or a paper result until task-quality and long-rollout gates pass"
            .to_owned();
    }
    let compatible_controller_only = multiscale_was_trained
        && config.adaptive.rule_perception
            == crate::adaptive::AdaptiveRulePerception::NpaCompatible
        && config.adaptive.local_residual_scale == 0.0
        && (config.multiscale_training.freeze_training_policy
            || config.multiscale_training.freeze_multiscale_rule
            || config.multiscale_training.freeze_local_residual_rule);
    let rule = if compatible_controller_only {
        "frozen regular NPA rule with learned topology event gates and deterministic refinement-defect allocation"
    } else if multiscale_was_trained
        && config.multiscale_training.rule_strategy
            == crate::adaptive::AdaptiveMultiscaleRuleStrategy::FullNormalized
    {
        "task-trained shared normalized-adaptive NPA rule"
    } else if multiscale_was_trained
        && config.multiscale_training.rule_strategy
            == crate::adaptive::AdaptiveMultiscaleRuleStrategy::CoarseReplacement
    {
        "frozen native NPA rule plus task-trained normalized coarse-leaf replacement"
    } else if multiscale_was_trained {
        "task-trained normalized local rule plus nonmaterial proxy residual"
    } else {
        match config.adaptive.rule_perception {
            crate::adaptive::AdaptiveRulePerception::NpaCompatible => {
                "fixed-bandwidth NPA-compatible represented-measure rule with exact uniform-limit parity"
            }
            crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive if rule_was_distilled => {
                "functionally distilled normalized-adaptive rule"
            }
            crate::adaptive::AdaptiveRulePerception::NormalizedAdaptive => {
                "untrained normalized-adaptive rule semantics"
            }
        }
    };
    if multiscale_was_trained {
        if compatible_controller_only {
            return format!(
                "task-trained adaptive lizard: {rule}, exact matched-seed hierarchical bootstrap, conservative represented-measure topology, particle-ID-preserving balanced exchanges, and matched-mask adaptive/regular task-quality validation"
            );
        }
        return format!(
            "task-trained multiscale lizard: {rule}, conservative mixed-resolution material cuts, measure-weighted restriction-evolution supervision, and a counterfactual-error resolution controller; task quality, bandwidth ablation, and long-rollout gates are reported separately from the paper's equal-size bandwidth pilot"
        );
    }
    format!(
        "validated foundation only: 2D/3D normalized variable-support oracle, hard graph budgets, canonical conservative events, globally normalized material budget, supervised manufactured resolution controller, {rule}, and material-footprint rendering; this is not a task-trained adaptive NPA"
    )
}

fn pearson(lhs: impl Iterator<Item = f32>, rhs: impl Iterator<Item = f32>) -> f32 {
    let pairs = lhs.zip(rhs).collect::<Vec<_>>();
    if pairs.is_empty() {
        return 0.0;
    }
    let lhs_mean = pairs.iter().map(|pair| pair.0).sum::<f32>() / pairs.len() as f32;
    let rhs_mean = pairs.iter().map(|pair| pair.1).sum::<f32>() / pairs.len() as f32;
    let covariance = pairs
        .iter()
        .map(|pair| (pair.0 - lhs_mean) * (pair.1 - rhs_mean))
        .sum::<f32>();
    let lhs_norm = pairs
        .iter()
        .map(|pair| (pair.0 - lhs_mean).powi(2))
        .sum::<f32>()
        .sqrt();
    let rhs_norm = pairs
        .iter()
        .map(|pair| (pair.1 - rhs_mean).powi(2))
        .sum::<f32>()
        .sqrt();
    covariance / (lhs_norm * rhs_norm).max(1.0e-12)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "backend_ndarray")]
    use super::{
        AdaptiveTrainingBackend, task_quality_models, validate_controller,
        validate_controller_reference,
    };
    use super::{
        adaptive_rule_matches_reference_architecture, append_controller_batch, paper_scope,
        update_controller_replay, validate_adaptive_task_quality_validation_gates,
        validate_experiment_alignment,
    };
    use crate::AdaptiveExperimentConfig;
    use crate::adaptive::{
        ADAPTIVE_CONTROLLER_INPUT_DIMS, ADAPTIVE_CONTROLLER_OUTPUT_DIMS,
        AdaptiveControllerTrainingBatch, AdaptiveLocalRuleSemantics,
        AdaptiveMultiscaleRuleStrategy, AdaptiveRulePerception,
    };
    use crate::adaptive::{AdaptiveCoarseDynamics, AdaptiveNpaConfig, AdaptiveNpaModel};
    use crate::{NpaConfig, NpaModel};

    fn experiment_config() -> AdaptiveExperimentConfig {
        toml::from_str(
            r#"
report_output = "artifacts/report.json"
model_output = "artifacts/model.bpk"
"#,
        )
        .unwrap()
    }

    #[test]
    fn material_scale_conditioning_preserves_reference_architecture_compatibility() {
        let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 13);
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.coarse_dynamics = AdaptiveCoarseDynamics::RepresentedMeasure;
        let mut model = AdaptiveNpaModel::seeded(base.clone(), config, 29).unwrap();
        model.enable_material_scale_conditioning().unwrap();

        assert!(adaptive_rule_matches_reference_architecture(&model, &base).unwrap());
        assert_eq!(model.rule.config.auxiliary_input_dims, 1);

        let mut incompatible = base.clone();
        incompatible.config.hidden_dims += 1;
        assert!(!adaptive_rule_matches_reference_architecture(&model, &incompatible).unwrap());
    }

    #[test]
    fn full_normalized_training_contract_is_accepted() {
        let mut config = experiment_config();
        config.multiscale_training.enabled = true;
        config.multiscale_training.rule_strategy = AdaptiveMultiscaleRuleStrategy::FullNormalized;
        config.adaptive.rule_perception = AdaptiveRulePerception::NormalizedAdaptive;
        config.adaptive.proxy.enabled = false;
        config.adaptive.local_residual_scale = 0.0;
        config.adaptive.closure_moment_features = false;

        validate_experiment_alignment(&config).unwrap();
    }

    #[test]
    fn compatible_residual_training_accepts_static_and_recurrent_closure_state() {
        let mut config = experiment_config();
        config.multiscale_training.enabled = true;
        config.multiscale_training.rule_strategy = AdaptiveMultiscaleRuleStrategy::Residual;
        config.adaptive.rule_perception = AdaptiveRulePerception::NpaCompatible;
        config.adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CompatibleResidual;
        config.adaptive.proxy.enabled = false;
        config.adaptive.closure_moment_features = true;
        config.adaptive.closure_recurrent_mode = false;

        validate_experiment_alignment(&config).unwrap();

        config.adaptive.closure_recurrent_mode = true;
        validate_experiment_alignment(&config).unwrap();
    }

    #[test]
    fn recurrent_closure_report_is_explicitly_experimental() {
        let mut config = experiment_config();
        config.adaptive.closure_moment_features = true;
        config.adaptive.closure_recurrent_mode = true;

        let scope = paper_scope(&config, false, true);

        assert!(scope.contains("experimental recurrent-closure candidate"));
        assert!(scope.contains("resident WGPU replay"));
    }

    #[test]
    fn active_bandwidth_gate_rejects_compatible_fixed_bandwidth_rule() {
        let mut config = experiment_config();
        config.task_quality.enabled = true;
        config.task_quality.target_image =
            "assets/reference_targets/lizard_upstream_120.png".into();
        config.task_quality.reference_model = Some("models/catalog/growing/lizard.bpk".into());
        config.task_quality.bandwidth_adaptation_enabled = true;
        config
            .gates
            .require_task_quality_bandwidth_adaptation_active = true;

        let error = validate_experiment_alignment(&config).unwrap_err();
        assert!(error.to_string().contains("normalized-adaptive"));
    }

    #[test]
    fn standalone_validation_requires_the_configured_seed_count() {
        let mut config = experiment_config();
        config.gates.min_task_quality_validation_seeds = 1;

        let failures = validate_adaptive_task_quality_validation_gates(config.gates, None);

        assert_eq!(failures, ["task parity validation seeds 0 < 1"]);
    }

    #[test]
    fn standalone_validation_requires_long_horizon_report_when_gated() {
        let mut config = experiment_config();
        config.gates.min_gap_final_selected_mode_vs_fine_control_db = -0.25;

        let failures = validate_adaptive_task_quality_validation_gates(config.gates, None);

        assert_eq!(
            failures,
            ["gap decomposition report is required by configured long-horizon gates"]
        );
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn backend_controller_validation_matches_cpu_reference() {
        let mut adaptive = AdaptiveNpaConfig::growing_2d();
        adaptive.controller_hidden_dims = 17;
        adaptive.split_probability = 0.37;
        adaptive.merge_probability = 0.61;
        let model = AdaptiveNpaModel::seeded(
            NpaModel::upstream_seeded(NpaConfig::growing_2d(), 13),
            adaptive,
            29,
        )
        .unwrap();
        let rows = 257;
        let features = (0..rows * ADAPTIVE_CONTROLLER_INPUT_DIMS)
            .map(|index| ((index * 37 % 211) as f32 - 105.0) / 71.0)
            .collect();
        let targets = (0..rows)
            .flat_map(|row| {
                [
                    ((row * 17 % 101) as f32 - 50.0) / 31.0,
                    ((row * 11 % 67) as f32 - 33.0) / 29.0,
                    f32::from(row % 3 == 0),
                    f32::from(row % 5 <= 1),
                ]
            })
            .collect();
        let batch = AdaptiveControllerTrainingBatch {
            features,
            targets,
            rows,
        };

        let reference = validate_controller_reference(&model, &batch).unwrap();
        let backend =
            validate_controller(AdaptiveTrainingBackend::NdArray, &model, &batch).unwrap();

        assert_eq!(backend.rows, reference.rows);
        assert!((backend.mean_squared_error - reference.mean_squared_error).abs() < 2.0e-5);
        assert!(
            (backend.desired_scale_correlation - reference.desired_scale_correlation).abs()
                < 2.0e-5
        );
        for channel in 0..ADAPTIVE_CONTROLLER_OUTPUT_DIMS {
            assert!(
                (backend.channel_mean_squared_error[channel]
                    - reference.channel_mean_squared_error[channel])
                    .abs()
                    < 2.0e-5
            );
        }
        for event in 0..2 {
            assert!(
                (backend.event_positive_fraction[event] - reference.event_positive_fraction[event])
                    .abs()
                    < 1.0e-6
            );
            assert!(
                (backend.event_precision[event] - reference.event_precision[event]).abs() < 1.0e-6
            );
            assert!((backend.event_recall[event] - reference.event_recall[event]).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn task_quality_frozen_base_disables_material_residual_contract() {
        let rule = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 13);
        let mut config = AdaptiveNpaConfig::growing_2d();
        config.local_rule_semantics =
            crate::adaptive::AdaptiveLocalRuleSemantics::CompatibleResidual;
        config.compatible_residual_material_features = true;
        config.closure_moment_features = true;
        config.closure_recurrent_mode = true;
        let model = AdaptiveNpaModel::seeded(rule.clone(), config, 29).unwrap();

        let controls = task_quality_models(&model, &rule);

        controls.frozen_base.validate().unwrap();
        assert!(controls.frozen_base.local_residual_rule.is_none());
        assert!(
            !controls
                .frozen_base
                .config
                .compatible_residual_material_features
        );
        assert!(controls.frozen_base.closure_mode_rule.is_none());
        assert!(!controls.frozen_base.config.closure_recurrent_mode);
    }

    #[test]
    fn on_policy_controller_replay_preserves_offline_rows() {
        let batch = |rows: usize, value: f32| AdaptiveControllerTrainingBatch {
            features: vec![value; rows * ADAPTIVE_CONTROLLER_INPUT_DIMS],
            targets: vec![value; rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
            rows,
        };
        let mut replay = batch(2, 1.0);
        let on_policy = batch(1, 2.0);
        append_controller_batch(&mut replay, &on_policy);

        replay.validate().unwrap();
        assert_eq!(replay.rows, 3);
        assert!(
            replay.features[..2 * ADAPTIVE_CONTROLLER_INPUT_DIMS]
                .iter()
                .all(|value| *value == 1.0)
        );
        assert!(
            replay.features[2 * ADAPTIVE_CONTROLLER_INPUT_DIMS..]
                .iter()
                .all(|value| *value == 2.0)
        );
    }

    #[test]
    fn on_policy_only_controller_replay_discards_static_rows_once() {
        let batch = |rows: usize, value: f32| AdaptiveControllerTrainingBatch {
            features: vec![value; rows * ADAPTIVE_CONTROLLER_INPUT_DIMS],
            targets: vec![value; rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS],
            rows,
        };
        let mut replay = batch(2, 1.0);
        update_controller_replay(&mut replay, &batch(3, 2.0), true);
        update_controller_replay(&mut replay, &batch(1, 3.0), false);

        replay.validate().unwrap();
        assert_eq!(replay.rows, 4);
        assert!(
            replay.features[..3 * ADAPTIVE_CONTROLLER_INPUT_DIMS]
                .iter()
                .all(|value| *value == 2.0)
        );
        assert!(
            replay.features[3 * ADAPTIVE_CONTROLLER_INPUT_DIMS..]
                .iter()
                .all(|value| *value == 3.0)
        );
    }
}
