use crate::cli::commands::hyper_support::{
    attach_condition_features, load_condition_image_2d, load_hyper_2d, save_hyper_2d,
    write_pretty_json,
};
use crate::cli::prelude::*;

use super::super::sources::Hyper2dScratchSource;
use super::super::{
    DinoConditionFeatureCacheConfig, build_condition_feature_cache,
    default_dino_cache_write_interval_batches, default_dino_feature_batch_size,
};
use super::{
    DirectBasisAdapterBankIndexedEntry, DirectBasisAdapterBankLoadEntry, DirectBasisTargetConfig,
    EvalConfig, config_value_enum, direct_basis_adapter_bank_selection_manifest, eval_indices,
    evaluate_direct_basis_example, load_direct_basis_adapter_bank, parse_direct_basis_split,
    read_direct_basis_adapter_bank_selection_manifest, replay_direct_basis_adapter_bank_selection,
    resolve_direct_basis_artifact_path, select_direct_basis_adapter_bank_oracle_entries,
};

mod types;

use types::*;

pub(crate) fn run_train_hyper_2d_adapter_bank(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dAdapterBank {
        config,
        preset,
        shared_base,
        adapter_bank,
        output_dir,
        report_output,
        hyper_output,
        source_limit,
        train_limit,
        holdout_limit,
        condition_encoder,
        dino_model,
        dino_image_size,
        condition_token_grid_width,
        condition_token_grid_height,
        backend,
        hyper_hidden,
        hyper_output_scale,
        hyper_seed,
        steps,
        report_interval,
        example_batch_size,
        learning_rate,
        weight_decay,
        grad_clip_norm,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
        vector_eval_examples,
        rollout_eval_examples,
        rollout_particles,
        rollout_steps,
        update_prob,
        eval_seed,
        seed_scale,
        seed_mode,
        target_points,
        target_image_size,
        target_threshold,
        target_loss_image_size,
        target_splat_sigma,
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
        target_displacement_regularizer_weight,
        target_overflow_regularizer_weight,
        target_bound_regularizer_weight,
    } = command
    else {
        unreachable!("run_train_hyper_2d_adapter_bank called with wrong command variant");
    };

    let experiment_config_path = config;
    let experiment_config_report = experiment_config_path
        .as_ref()
        .map(|path| path.display().to_string());
    let experiment_config = load_adapter_bank_experiment_config(experiment_config_path.as_deref())?;
    let AdapterBankExperimentConfig {
        preset: config_preset,
        input: config_input,
        selection: config_selection,
        output: config_output,
        condition: config_condition,
        training: config_training,
        eval: config_eval,
        target: config_target,
    } = experiment_config;
    let AdapterBankInputExperimentConfig {
        shared_base: config_shared_base,
        adapter_bank: config_adapter_bank,
        initial_hyper: config_initial_hyper,
        psnr_gate_report: config_psnr_gate_report,
        source_limit: config_source_limit,
        train_limit: config_train_limit,
        holdout_limit: config_holdout_limit,
    } = config_input;
    let AdapterBankSelectionExperimentConfig {
        selection_seed: config_selection_seed,
        selection_manifest: config_selection_manifest,
    } = config_selection;
    let AdapterBankOutputExperimentConfig {
        output_dir: config_output_dir,
        report_output: config_report_output,
        hyper_output: config_hyper_output,
    } = config_output;
    let AdapterBankConditionExperimentConfig {
        encoder: config_condition_encoder,
        dino_model: config_dino_model,
        dino_image_size: config_dino_image_size,
        dino_batch_size: config_dino_batch_size,
        dino_cache_write_interval_batches: config_dino_cache_write_interval_batches,
        feature_cache: config_condition_feature_cache,
        token_grid_width: config_condition_token_grid_width,
        token_grid_height: config_condition_token_grid_height,
    } = config_condition;
    let AdapterBankTrainingExperimentConfig {
        backend: config_backend,
        objective: config_objective,
        hidden: config_hyper_hidden,
        output_scale: config_hyper_output_scale,
        linear_output: config_linear_output,
        canonicalize_adapters: config_canonicalize_adapters,
        flow_hidden: config_flow_hidden,
        flow_sample_steps: config_flow_sample_steps,
        flow_source_scale: config_flow_source_scale,
        flow_sample_seed: config_flow_sample_seed,
        flow_hidden_activation: config_flow_hidden_activation,
        flow_init: config_flow_init,
        flow_loss: config_flow_loss,
        flow_hard_sample_weight: config_flow_hard_sample_weight,
        flow_hard_sample_psnr_threshold_db: config_flow_hard_sample_psnr_threshold_db,
        diagnostic_vector_examples: config_diagnostic_vector_examples,
        loss_eval_batch_size: config_loss_eval_batch_size,
        system_memory_budget_gb: config_system_memory_budget_gb,
        seed: config_hyper_seed,
        steps: config_steps,
        report_interval: config_report_interval,
        example_batch_size: config_example_batch_size,
        learning_rate: config_learning_rate,
        weight_decay: config_weight_decay,
        grad_clip_norm: config_grad_clip_norm,
        adam_beta1: config_adam_beta1,
        adam_beta2: config_adam_beta2,
        adam_epsilon: config_adam_epsilon,
    } = config_training;
    let AdapterBankEvalExperimentConfig {
        vector_examples: config_vector_eval_examples,
        rollout_examples: config_rollout_eval_examples,
        particles: config_rollout_particles,
        steps: config_rollout_steps,
        update_prob: config_update_prob,
        seed: config_eval_seed,
        seed_scale: config_seed_scale,
        seed_mode: config_seed_mode,
    } = config_eval;
    let AdapterBankTargetExperimentConfig {
        points: config_target_points,
        image_size: config_target_image_size,
        threshold: config_target_threshold,
        loss_image_size: config_target_loss_image_size,
        splat_sigma: config_target_splat_sigma,
        splat_loss_weight: config_target_splat_loss_weight,
        color_loss_weight: config_target_color_loss_weight,
        density_loss_weight: config_target_density_loss_weight,
        displacement_regularizer_weight: config_target_displacement_regularizer_weight,
        overflow_regularizer_weight: config_target_overflow_regularizer_weight,
        bound_regularizer_weight: config_target_bound_regularizer_weight,
    } = config_target;

    let preset_arg = config_value_enum("preset", config_preset, preset)?;
    let preset: AutomataPreset = preset_arg.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-hyper2d-adapter-bank currently supports growing-2d adapter banks",
        )
        .into());
    }
    let shared_base = config_shared_base.or(shared_base).ok_or_else(|| {
        std::io::Error::other(
            "train-hyper2d-adapter-bank requires --shared-base or input.shared_base",
        )
    })?;
    let adapter_bank = config_adapter_bank.or(adapter_bank).ok_or_else(|| {
        std::io::Error::other(
            "train-hyper2d-adapter-bank requires --adapter-bank or input.adapter_bank",
        )
    })?;
    let output_dir = config_output_dir.unwrap_or(output_dir);
    let report_output = config_report_output
        .or(report_output)
        .unwrap_or_else(|| output_dir.join("report.json"));
    let hyper_output = config_hyper_output
        .or(hyper_output)
        .unwrap_or_else(|| output_dir.join("hyper_2d.json"));
    let source_limit = config_source_limit.unwrap_or(source_limit);
    let train_limit = config_train_limit.unwrap_or(train_limit);
    let holdout_limit = config_holdout_limit.unwrap_or(holdout_limit);
    let selection_seed = config_selection_seed;
    let condition_encoder: ConditionEncoder2d = config_value_enum(
        "condition.encoder",
        config_condition_encoder,
        condition_encoder,
    )?
    .into();
    let dino_model = config_dino_model.or(dino_model);
    let dino_image_size = config_dino_image_size.unwrap_or(dino_image_size);
    let dino_batch_size = config_dino_batch_size.unwrap_or_else(default_dino_feature_batch_size);
    let dino_cache_write_interval_batches = config_dino_cache_write_interval_batches
        .unwrap_or(default_dino_cache_write_interval_batches());
    let condition_feature_cache = config_condition_feature_cache;
    let condition_token_grid_width =
        config_condition_token_grid_width.unwrap_or(condition_token_grid_width);
    let condition_token_grid_height =
        config_condition_token_grid_height.unwrap_or(condition_token_grid_height);
    let backend = config_value_enum("training.backend", config_backend, backend)?;
    let objective = parse_adapter_bank_training_objective(
        config_objective.as_deref().unwrap_or("static-vector-mse"),
    )?;
    let hyper_hidden = config_hyper_hidden.unwrap_or(hyper_hidden);
    let hyper_output_scale = config_hyper_output_scale.unwrap_or(hyper_output_scale);
    let linear_output = config_linear_output.unwrap_or(false);
    let canonicalize_adapters = config_canonicalize_adapters.unwrap_or(false);
    let flow_hidden = config_flow_hidden.unwrap_or(hyper_hidden);
    let flow_sample_steps = config_flow_sample_steps.unwrap_or(16);
    let flow_hidden_activation = parse_adapter_bank_flow_hidden_activation(
        config_flow_hidden_activation.as_deref().unwrap_or("relu"),
    )?;
    let flow_init = parse_adapter_bank_flow_init(config_flow_init.as_deref().unwrap_or("random"))?;
    let flow_loss =
        parse_adapter_bank_flow_loss(config_flow_loss.as_deref().unwrap_or("velocity-mse"))?;
    let flow_hard_sample_weight = config_flow_hard_sample_weight.unwrap_or(1.0);
    let flow_hard_sample_psnr_threshold_db =
        config_flow_hard_sample_psnr_threshold_db.unwrap_or(26.0);
    if !flow_hard_sample_weight.is_finite() || flow_hard_sample_weight <= 0.0 {
        return Err(std::io::Error::other(
            "training.flow_hard_sample_weight must be finite and positive",
        )
        .into());
    }
    if !flow_hard_sample_psnr_threshold_db.is_finite() {
        return Err(std::io::Error::other(
            "training.flow_hard_sample_psnr_threshold_db must be finite",
        )
        .into());
    }
    if config_initial_hyper.is_some() && flow_init != AdapterBankFlowInit::FromHyper {
        return Err(std::io::Error::other(
            "input.initial_hyper is only valid with training.flow_init = \"from-hyper\"",
        )
        .into());
    }
    let hyper_seed = config_hyper_seed.unwrap_or(hyper_seed);
    let flow_sample_seed = config_flow_sample_seed.unwrap_or(hyper_seed ^ 0x9e37_79b9_7f4a_7c15);
    let steps = config_steps.unwrap_or(steps);
    let report_interval = config_report_interval.unwrap_or(report_interval);
    let example_batch_size = config_example_batch_size.unwrap_or(example_batch_size);
    let loss_eval_batch_size = config_loss_eval_batch_size.unwrap_or(512);
    let system_memory_budget_gb = config_system_memory_budget_gb;
    let learning_rate = config_learning_rate.unwrap_or(learning_rate);
    let weight_decay = config_weight_decay.unwrap_or(weight_decay);
    let grad_clip_norm = config_grad_clip_norm.unwrap_or(grad_clip_norm);
    let adam_beta1 = config_adam_beta1.unwrap_or(adam_beta1);
    let adam_beta2 = config_adam_beta2.unwrap_or(adam_beta2);
    let adam_epsilon = config_adam_epsilon.unwrap_or(adam_epsilon);
    let vector_eval_examples = config_vector_eval_examples.unwrap_or(vector_eval_examples);
    let diagnostic_vector_examples =
        config_diagnostic_vector_examples.unwrap_or_else(|| vector_eval_examples.clamp(1, 16));
    let rollout_eval_examples = config_rollout_eval_examples.unwrap_or(rollout_eval_examples);
    let rollout_particles = config_rollout_particles.unwrap_or(rollout_particles);
    let rollout_steps = config_rollout_steps.unwrap_or(rollout_steps);
    let update_prob = config_update_prob.unwrap_or(update_prob);
    let eval_seed = config_eval_seed.unwrap_or(eval_seed);
    let seed_scale = config_seed_scale.or(seed_scale);
    let seed_mode = config_value_enum("eval.seed_mode", config_seed_mode, seed_mode)?;
    let target_points = config_target_points.unwrap_or(target_points);
    let target_image_size = config_target_image_size.or(target_image_size);
    let target_threshold = config_target_threshold.unwrap_or(target_threshold);
    let target_loss_image_size = config_target_loss_image_size.unwrap_or(target_loss_image_size);
    let target_splat_sigma = config_target_splat_sigma.unwrap_or(target_splat_sigma);
    let target_splat_loss_weight =
        config_target_splat_loss_weight.unwrap_or(target_splat_loss_weight);
    let target_color_loss_weight =
        config_target_color_loss_weight.unwrap_or(target_color_loss_weight);
    let target_density_loss_weight =
        config_target_density_loss_weight.unwrap_or(target_density_loss_weight);
    let target_displacement_regularizer_weight = config_target_displacement_regularizer_weight
        .unwrap_or(target_displacement_regularizer_weight);
    let target_overflow_regularizer_weight =
        config_target_overflow_regularizer_weight.unwrap_or(target_overflow_regularizer_weight);
    let target_bound_regularizer_weight =
        config_target_bound_regularizer_weight.unwrap_or(target_bound_regularizer_weight);

    if hyper_hidden == 0 {
        return Err(std::io::Error::other("hyper hidden dimensions must be non-zero").into());
    }
    if rollout_eval_examples > 0 && (rollout_particles == 0 || rollout_steps == 0) {
        return Err(std::io::Error::other(
            "rollout eval requires non-zero particles and rollout steps",
        )
        .into());
    }

    let base_manifest = crate::import::load_manifest(&shared_base)?;
    let base_model = base_manifest.clone().into_model();
    if base_model.config.spatial_dims != 2 {
        return Err(std::io::Error::other("adapter-bank HyperNPA requires a 2D base model").into());
    }
    let bank = load_direct_basis_adapter_bank(&adapter_bank)?;
    let adapter_rank = bank.adapter_rank;
    let adapter_alpha = bank.adapter_alpha;
    let (selected_entries, selection_report) = select_adapter_bank_entries(
        bank.entries,
        source_limit,
        train_limit,
        holdout_limit,
        selection_seed,
        config_selection_manifest.as_deref(),
    )?;
    let selected_sources = selected_entries
        .iter()
        .map(|entry| {
            let condition_path = resolve_direct_basis_artifact_path(
                adapter_bank.parent().unwrap_or_else(|| Path::new("")),
                &entry.condition,
            );
            Ok(Hyper2dScratchSource {
                slug: entry.slug.clone(),
                title: entry.title.clone(),
                group: entry.group.clone(),
                condition_path,
                particles: None,
                seed_scale: None,
                update_prob: None,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let condition_features = build_condition_feature_cache(
        &selected_sources,
        condition_encoder,
        DinoConditionFeatureCacheConfig {
            model: dino_model.as_ref(),
            image_size: dino_image_size,
            batch_size: dino_batch_size,
            cache_write_interval_batches: dino_cache_write_interval_batches,
            token_grid_width: condition_token_grid_width,
            token_grid_height: condition_token_grid_height,
            cache_path: condition_feature_cache.as_deref(),
        },
    )?;
    let mut examples = load_conditioned_adapter_bank_examples(
        &adapter_bank,
        &base_manifest,
        selected_entries,
        Some(&condition_features),
        canonicalize_adapters,
    )?;
    let sample_weights = apply_psnr_sample_weights(
        &mut examples,
        config_psnr_gate_report.as_deref(),
        flow_hard_sample_weight,
        flow_hard_sample_psnr_threshold_db,
    )?;
    let memory = vec![check_process_memory_budget(
        "adapter-bank examples loaded",
        system_memory_budget_gb,
    )?];
    let train_examples = examples
        .iter()
        .filter(|example| example.split.is_train())
        .cloned()
        .collect::<Vec<_>>();
    let holdout_examples = examples
        .iter()
        .filter(|example| !example.split.is_train())
        .cloned()
        .collect::<Vec<_>>();
    if train_examples.is_empty() {
        return Err(
            std::io::Error::other("adapter bank selection produced no train examples").into(),
        );
    }

    let adapter_bias_correction = train_examples
        .iter()
        .any(|example| example.target_has_bias_correction);
    let target_stats = target_vector_stats(&train_examples, hyper_output_scale)?;
    let output_scale = target_stats.output_scale;
    let flow_source_scale =
        config_flow_source_scale.unwrap_or_else(|| target_stats.mean_rms.max(1.0e-6));
    let hyper_config = HyperNpa2dConfig {
        condition_encoder,
        condition_feature_dims: condition_feature_dims_for_encoder(
            condition_encoder,
            condition_token_grid_width,
            condition_token_grid_height,
        )?,
        condition_token_grid_width,
        condition_token_grid_height,
        hidden_dims: hyper_hidden,
        adapter_rank,
        adapter_alpha,
        adapter_bias_correction,
        output_activation: if linear_output {
            HyperNpa2dOutputActivation::Linear
        } else {
            HyperNpa2dOutputActivation::Tanh
        },
        output_scale,
    };
    let mut hyper = HyperNpa2d::seeded(base_model.config.clone(), hyper_config, hyper_seed)?;
    let initial_hyper_report = config_initial_hyper
        .as_ref()
        .map(|path| path.display().to_string());
    if flow_init == AdapterBankFlowInit::FromHyper {
        let initial_hyper_path = config_initial_hyper.as_ref().ok_or_else(|| {
            std::io::Error::other(
                "training.flow_init = \"from-hyper\" requires input.initial_hyper",
            )
        })?;
        load_adapter_bank_initial_flow(
            &mut hyper,
            initial_hyper_path,
            crate::HyperNpa2dFlowConfig {
                hidden_dims: flow_hidden,
                sample_steps: flow_sample_steps,
                source_scale: flow_source_scale,
                sample_seed: flow_sample_seed,
                hidden_activation: flow_hidden_activation,
            },
        )?;
    }
    let train_config = AdapterBankTrainConfig {
        objective,
        steps,
        report_interval,
        example_batch_size,
        diagnostic_vector_examples,
        loss_eval_batch_size,
        system_memory_budget_gb,
        seed: hyper_seed,
        optimizer: AdamWConfig {
            learning_rate,
            weight_decay,
            grad_clip_norm,
            beta1: adam_beta1,
            beta2: adam_beta2,
            epsilon: adam_epsilon,
        },
        flow: AdapterBankFlowTrainConfig {
            hidden_dims: flow_hidden,
            sample_steps: flow_sample_steps,
            source_scale: flow_source_scale,
            sample_seed: flow_sample_seed,
            hidden_activation: flow_hidden_activation,
            init: flow_init,
            loss: flow_loss,
            sample_weights,
        },
    };
    validate_training_config(train_config)?;
    let training = match backend {
        Hyper2dAdapterBankBackendArg::BurnWgpu => train_adapter_bank_burn_wgpu(
            &mut hyper,
            &train_examples,
            Some(&holdout_examples),
            train_config,
        )?,
        Hyper2dAdapterBankBackendArg::Cpu => {
            train_adapter_bank_cpu(&mut hyper, &train_examples, train_config)?
        }
        Hyper2dAdapterBankBackendArg::LinearSolve => {
            train_adapter_bank_linear_solve(&mut hyper, &train_examples, train_config)?
        }
    };
    save_hyper_2d(&hyper_output, &hyper)?;

    let train_vector_metrics =
        vector_metrics(&hyper, &train_examples, vector_eval_examples, eval_seed)?;
    let holdout_vector_metrics = if holdout_examples.is_empty() {
        None
    } else {
        Some(vector_metrics(
            &hyper,
            &holdout_examples,
            vector_eval_examples,
            eval_seed,
        )?)
    };
    let loss_config = super::super::super::target2d::target2d_loss_config(
        target_loss_image_size,
        target_splat_sigma,
        true,
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
        Target2dLossConfig::default().background_density_loss_weight,
        Target2dLossConfig::default().foreground_density_loss_weight,
        target_displacement_regularizer_weight,
        target_overflow_regularizer_weight,
        target_bound_regularizer_weight,
    )?;
    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let rollout_eval = evaluate_rollout_splits(
        &base_model,
        &hyper,
        &train_examples,
        &holdout_examples,
        AdapterBankRolloutEvalConfig {
            target: DirectBasisTargetConfig {
                threshold: target_threshold,
                points: target_points,
                image_size: target_image_size,
            },
            rollout: EvalConfig {
                particle_count: rollout_particles,
                rollout_steps,
                update_prob,
                seed: eval_seed,
                seed_scale,
                seed_mode,
            },
            loss: loss_config,
            requested_examples_per_split: rollout_eval_examples,
        },
    )?;

    let report = AdapterBankConditionedTrainingReport {
        experiment_config: experiment_config_report,
        preset,
        shared_base: shared_base.display().to_string(),
        adapter_bank: adapter_bank.display().to_string(),
        adapter_bank_base_model: bank.base_model,
        output_dir: output_dir.display().to_string(),
        report_output: report_output.display().to_string(),
        hyper_output: hyper_output.display().to_string(),
        backend,
        npa_config: base_model.config.clone(),
        hashgrid: base_manifest.hashgrid.clone(),
        hyper_config,
        generator_architecture: match objective {
            AdapterBankTrainingObjective::StaticVectorMse => "two-layer-mlp-tanh-lora-vector",
            AdapterBankTrainingObjective::RectifiedFlow => {
                "conditioned-rectified-flow-lora-vector-field"
            }
        },
        generator_objective: match objective {
            AdapterBankTrainingObjective::StaticVectorMse => "supervised-static-lora-vector-mse",
            AdapterBankTrainingObjective::RectifiedFlow => "rectified-flow-lora-vector",
        },
        adapter_rank,
        adapter_alpha,
        adapter_parameter_count: hyper.adapter_parameter_count(),
        condition_encoder: condition_encoder_report_label(
            condition_encoder,
            condition_token_grid_width,
            condition_token_grid_height,
        ),
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        source_limit,
        train_limit,
        holdout_limit,
        selection: selection_report,
        target_stats,
        requested_training: AdapterBankTrainingSettingsReport {
            objective: objective.label(),
            steps,
            report_interval,
            example_batch_size,
            diagnostic_vector_examples,
            loss_eval_batch_size,
            system_memory_budget_gb,
            seed: hyper_seed,
            optimizer: train_config.optimizer,
            flow: (objective == AdapterBankTrainingObjective::RectifiedFlow).then_some(
                AdapterBankFlowTrainingSettingsReport {
                    hidden_dims: flow_hidden,
                    sample_steps: flow_sample_steps,
                    source_scale: flow_source_scale,
                    sample_seed: flow_sample_seed,
                    hidden_activation: flow_hidden_activation,
                    init: flow_init.label(),
                    loss: flow_loss.label(),
                    sample_weights,
                    initial_hyper: initial_hyper_report,
                },
            ),
        },
        adapter_target_canonicalization: if canonicalize_adapters {
            "balanced-signed-energy-sorted-low-rank-factors"
        } else {
            "raw-stored-low-rank-factors"
        },
        memory,
        training,
        train_vector_metrics,
        holdout_vector_metrics,
        rollout_particles,
        rollout_steps,
        target_points,
        target_loss_config: loss_config,
        rollout_eval,
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} hyper={} backend={:?}",
        report_output.display(),
        report.train_examples,
        report.holdout_examples,
        hyper_output.display(),
        backend
    );
    Ok(())
}

fn load_adapter_bank_experiment_config(
    path: Option<&Path>,
) -> Result<AdapterBankExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(AdapterBankExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse adapter-bank experiment config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn select_adapter_bank_entries(
    entries: Vec<DirectBasisAdapterBankLoadEntry>,
    source_limit: usize,
    train_limit: usize,
    holdout_limit: usize,
    selection_seed: Option<u64>,
    selection_manifest_path: Option<&Path>,
) -> Result<
    (
        Vec<DirectBasisAdapterBankLoadEntry>,
        AdapterBankSelectionReport,
    ),
    Box<dyn std::error::Error>,
> {
    let limited_entries = entries
        .into_iter()
        .take(if source_limit == 0 {
            usize::MAX
        } else {
            source_limit
        })
        .collect::<Vec<_>>();
    if selection_seed.is_some() || selection_manifest_path.is_some() {
        return select_adapter_bank_entries_seeded(
            limited_entries,
            train_limit,
            holdout_limit,
            selection_seed.unwrap_or(0),
            selection_manifest_path,
        );
    }
    let mut selected = Vec::new();
    let mut train_count = 0usize;
    let mut holdout_count = 0usize;
    for entry in limited_entries {
        let split = parse_direct_basis_split(&entry.split)?;
        if split.is_train() {
            if train_limit > 0 && train_count >= train_limit {
                continue;
            }
            train_count += 1;
        } else {
            if holdout_limit > 0 && holdout_count >= holdout_limit {
                continue;
            }
            holdout_count += 1;
        }
        selected.push(entry);
    }
    if selected.is_empty() {
        return Err(std::io::Error::other("adapter bank selection produced no examples").into());
    }
    Ok((
        selected.clone(),
        AdapterBankSelectionReport {
            selection_seed: None,
            selection_manifest: None,
            replayed_manifest: false,
            train_selected: train_count,
            holdout_selected: holdout_count,
        },
    ))
}

fn select_adapter_bank_entries_seeded(
    entries: Vec<DirectBasisAdapterBankLoadEntry>,
    train_limit: usize,
    holdout_limit: usize,
    selection_seed: u64,
    selection_manifest_path: Option<&Path>,
) -> Result<
    (
        Vec<DirectBasisAdapterBankLoadEntry>,
        AdapterBankSelectionReport,
    ),
    Box<dyn std::error::Error>,
> {
    let mut train_entries = Vec::<DirectBasisAdapterBankIndexedEntry>::new();
    let mut holdout_entries = Vec::<DirectBasisAdapterBankIndexedEntry>::new();
    for entry in entries {
        let split = parse_direct_basis_split(&entry.split)?;
        if split.is_train() {
            train_entries.push((train_entries.len(), entry));
        } else {
            holdout_entries.push((holdout_entries.len(), entry));
        }
    }
    let (selected_train, selected_holdout, replayed_manifest) =
        if let Some(path) = selection_manifest_path.filter(|path| path.exists()) {
            let manifest = read_direct_basis_adapter_bank_selection_manifest(path)?;
            (
                replay_direct_basis_adapter_bank_selection(&train_entries, &manifest.train)?,
                replay_direct_basis_adapter_bank_selection(&holdout_entries, &manifest.holdout)?,
                true,
            )
        } else {
            (
                select_direct_basis_adapter_bank_oracle_entries(
                    &train_entries,
                    train_limit,
                    selection_seed,
                ),
                select_direct_basis_adapter_bank_oracle_entries(
                    &holdout_entries,
                    holdout_limit,
                    selection_seed ^ 0x90_1d_2d,
                ),
                false,
            )
        };
    let manifest = direct_basis_adapter_bank_selection_manifest(
        selection_seed,
        &selected_train,
        &selected_holdout,
    );
    if !replayed_manifest && let Some(path) = selection_manifest_path {
        write_pretty_json(path, &manifest)?;
    }
    let selected = selected_train
        .iter()
        .chain(selected_holdout.iter())
        .map(|(_, entry)| entry.clone())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(std::io::Error::other("adapter bank selection produced no examples").into());
    }
    Ok((
        selected,
        AdapterBankSelectionReport {
            selection_seed: Some(selection_seed),
            selection_manifest: selection_manifest_path.map(|path| path.display().to_string()),
            replayed_manifest,
            train_selected: manifest.train.len(),
            holdout_selected: manifest.holdout.len(),
        },
    ))
}

fn load_conditioned_adapter_bank_examples(
    adapter_bank_path: &Path,
    base_manifest: &BpkModelManifest,
    entries: Vec<DirectBasisAdapterBankLoadEntry>,
    condition_features: Option<&crate::cli::commands::hyper_support::Hyper2dConditionFeatureCache>,
    canonicalize_adapters: bool,
) -> Result<Vec<AdapterBankConditionedExample>, Box<dyn std::error::Error>> {
    let anchor = adapter_bank_path.parent().unwrap_or_else(|| Path::new(""));
    let mut examples = Vec::with_capacity(entries.len());
    for entry in entries {
        let split = parse_direct_basis_split(&entry.split)?;
        let condition_path = resolve_direct_basis_artifact_path(anchor, &entry.condition);
        let adapter_path = resolve_direct_basis_artifact_path(anchor, &entry.adapter_output);
        let adapter_manifest = crate::import::load_adapter_manifest(&adapter_path)?;
        adapter_manifest.validate(base_manifest)?;
        let target_adapter = if canonicalize_adapters {
            adapter_manifest
                .adapter
                .canonicalized(&base_manifest.config)?
        } else {
            adapter_manifest.adapter
        };
        let condition = condition_for_adapter_bank_example(&condition_path, condition_features)?;
        let target_vector = target_adapter.to_parameter_vector();
        let target_has_bias_correction = target_adapter.has_bias_correction();
        let source = Hyper2dScratchSource {
            slug: entry.slug,
            title: entry.title,
            group: entry.group,
            condition_path,
            particles: None,
            seed_scale: None,
            update_prob: None,
        };
        examples.push(AdapterBankConditionedExample {
            source,
            split,
            condition,
            target_vector,
            target_has_bias_correction,
            target_source_width: entry.target_source_width,
            target_source_height: entry.target_source_height,
            target_points: entry.target_points,
            last_train_loss: entry.last_train_loss,
            sample_weight: 1.0,
        });
    }
    Ok(examples)
}

#[derive(Deserialize)]
struct AdapterBankPsnrGateReportLoad {
    entries: Vec<AdapterBankPsnrGateEntryLoad>,
}

#[derive(Deserialize)]
struct AdapterBankPsnrGateEntryLoad {
    slug: String,
    kind: String,
    render_rgb_psnr_db: f32,
}

fn apply_psnr_sample_weights(
    examples: &mut [AdapterBankConditionedExample],
    psnr_gate_report: Option<&Path>,
    hard_weight: f32,
    psnr_threshold_db: f32,
) -> Result<AdapterBankSampleWeights, Box<dyn std::error::Error>> {
    if hard_weight <= 1.0 && psnr_gate_report.is_none() {
        return Ok(AdapterBankSampleWeights::default());
    }
    let report_path = psnr_gate_report.ok_or_else(|| {
        std::io::Error::other(
            "training.flow_hard_sample_weight > 1 requires input.psnr_gate_report",
        )
    })?;
    let text = std::fs::read_to_string(report_path)?;
    let report: AdapterBankPsnrGateReportLoad = serde_json::from_str(&text)?;
    let hard_slugs = report
        .entries
        .into_iter()
        .filter(|entry| entry.kind == "hyper" && entry.render_rgb_psnr_db < psnr_threshold_db)
        .map(|entry| entry.slug)
        .collect::<std::collections::HashSet<_>>();
    let mut hard_examples = 0usize;
    for example in examples {
        if hard_slugs.contains(&example.source.slug) {
            example.sample_weight = hard_weight;
            hard_examples += 1;
        } else {
            example.sample_weight = 1.0;
        }
    }
    Ok(AdapterBankSampleWeights {
        enabled: true,
        hard_weight,
        psnr_threshold_db,
        hard_examples,
    })
}

fn condition_for_adapter_bank_example(
    condition_path: &Path,
    condition_features: Option<&crate::cli::commands::hyper_support::Hyper2dConditionFeatureCache>,
) -> Result<ConditionImage2d, Box<dyn std::error::Error>> {
    if let Some(values) = condition_features.and_then(|features| features.get(condition_path)) {
        return Ok(ConditionImage2d::from_luma(1, 1, vec![0.0])?
            .with_dino_vits_features(values.clone())?);
    }
    attach_condition_features(
        load_condition_image_2d(condition_path)?,
        condition_path,
        condition_features,
    )
}

fn target_vector_stats(
    examples: &[AdapterBankConditionedExample],
    requested_output_scale: f32,
) -> Result<AdapterBankTargetVectorStats, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err(std::io::Error::other("target vector stats require examples").into());
    }
    if !requested_output_scale.is_finite() || requested_output_scale < 0.0 {
        return Err(
            std::io::Error::other("--hyper-output-scale must be finite and non-negative").into(),
        );
    }
    let parameters = examples[0].target_vector.len();
    let mut sum_sq = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_abs = 0.0_f32;
    let mut values = 0usize;
    for example in examples {
        if example.target_vector.len() != parameters {
            return Err(std::io::Error::other("adapter parameter counts differ").into());
        }
        for value in &example.target_vector {
            let abs = value.abs();
            sum_sq += (*value as f64) * (*value as f64);
            sum_abs += abs as f64;
            max_abs = max_abs.max(abs);
            values += 1;
        }
    }
    let auto_scale = (max_abs * 1.2).max((sum_sq / values as f64).sqrt() as f32 * 4.0);
    let output_scale = if requested_output_scale > 0.0 {
        requested_output_scale
    } else {
        auto_scale.max(1.0e-3)
    };
    let outside = examples
        .iter()
        .flat_map(|example| example.target_vector.iter())
        .filter(|value| value.abs() > output_scale)
        .count();
    Ok(AdapterBankTargetVectorStats {
        examples: examples.len(),
        parameters_per_adapter: parameters,
        mean_rms: (sum_sq / values as f64).sqrt() as f32,
        mean_abs: (sum_abs / values as f64) as f32,
        max_abs,
        output_scale,
        target_values_outside_output_scale_fraction: outside as f32 / values as f32,
    })
}

fn vector_metrics(
    hyper: &HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    requested_examples: usize,
    seed: u64,
) -> Result<AdapterBankVectorMetricsReport, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err(std::io::Error::other("vector metrics require examples").into());
    }
    let indices = eval_indices(examples.len(), requested_examples, seed);
    let output_scale = hyper.config.output_scale;
    let parameters = hyper.adapter_parameter_count();
    let mut sum_sq_err = 0.0_f64;
    let mut sum_abs_err = 0.0_f64;
    let mut max_abs_err = 0.0_f32;
    let mut sum_target_sq = 0.0_f64;
    let mut sum_prediction_sq = 0.0_f64;
    let mut target_max_abs = 0.0_f32;
    let mut prediction_max_abs = 0.0_f32;
    let mut cosine_sum = 0.0_f32;
    let mut near_scale = 0usize;
    let mut outside_scale = 0usize;
    let mut values = 0usize;
    for &idx in &indices {
        let example = &examples[idx];
        let predicted = hyper.predict_adapter_vector(&example.condition)?;
        if predicted.len() != parameters || example.target_vector.len() != parameters {
            return Err(std::io::Error::other("adapter vector length mismatch").into());
        }
        let mut dot = 0.0_f64;
        let mut pred_sq = 0.0_f64;
        let mut target_sq = 0.0_f64;
        for (actual, expected) in predicted.iter().zip(&example.target_vector) {
            let diff = actual - expected;
            let abs_diff = diff.abs();
            sum_sq_err += (diff as f64) * (diff as f64);
            sum_abs_err += abs_diff as f64;
            max_abs_err = max_abs_err.max(abs_diff);
            sum_target_sq += (*expected as f64) * (*expected as f64);
            sum_prediction_sq += (*actual as f64) * (*actual as f64);
            target_max_abs = target_max_abs.max(expected.abs());
            prediction_max_abs = prediction_max_abs.max(actual.abs());
            if actual.abs() >= output_scale * 0.98 {
                near_scale += 1;
            }
            if expected.abs() > output_scale {
                outside_scale += 1;
            }
            dot += (*actual as f64) * (*expected as f64);
            pred_sq += (*actual as f64) * (*actual as f64);
            target_sq += (*expected as f64) * (*expected as f64);
            values += 1;
        }
        let denom = (pred_sq * target_sq).sqrt();
        if denom > f64::MIN_POSITIVE {
            cosine_sum += (dot / denom) as f32;
        }
    }
    let mse = (sum_sq_err / values as f64) as f32;
    let rmse = mse.sqrt();
    let target_rms = (sum_target_sq / values as f64).sqrt() as f32;
    let prediction_rms = (sum_prediction_sq / values as f64).sqrt() as f32;
    Ok(AdapterBankVectorMetricsReport {
        examples: indices.len(),
        parameters_per_adapter: parameters,
        mse,
        rmse,
        normalized_rmse_to_target_rms: (target_rms > 0.0).then_some(rmse / target_rms),
        mean_abs_error: (sum_abs_err / values as f64) as f32,
        max_abs_error: max_abs_err,
        target_rms,
        prediction_rms,
        target_max_abs,
        prediction_max_abs,
        mean_cosine_similarity: cosine_sum / indices.len() as f32,
        prediction_values_near_output_scale_fraction: near_scale as f32 / values as f32,
        target_values_outside_output_scale_fraction: outside_scale as f32 / values as f32,
    })
}

fn train_adapter_bank_cpu(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let adapter_examples = examples
        .iter()
        .map(|example| {
            Ok(HyperAdapterExample2d {
                condition: example.condition.clone(),
                target_adapter: target_adapter_from_vector(hyper, example)?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let initial_loss = hyper_adapter_regression_loss(hyper, &adapter_examples)?;
    let mut best_loss = initial_loss;
    let mut best_step = 0usize;
    let mut best_hyper = hyper.clone();
    let mut final_loss = initial_loss;
    let mut history = Vec::new();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_batch_size(config.example_batch_size, adapter_examples.len());
    let sgd = SgdConfig {
        learning_rate: config.optimizer.learning_rate,
        weight_decay: config.optimizer.weight_decay,
        grad_clip_norm: config.optimizer.grad_clip_norm,
    };
    for step in 1..=config.steps {
        let step_started = Instant::now();
        let indices = sample_indices(adapter_examples.len(), batch_size, &mut rng);
        let batch = indices
            .iter()
            .map(|idx| adapter_examples[*idx].clone())
            .collect::<Vec<_>>();
        let step_report = hyper_adapter_regression_train_step(hyper, &batch, sgd)?;
        let step_elapsed = step_started.elapsed();
        if step == config.steps || step.is_multiple_of(config.report_interval.max(1)) {
            final_loss = hyper_adapter_regression_loss(hyper, &adapter_examples)?;
            if final_loss < best_loss {
                best_loss = final_loss;
                best_step = step;
                best_hyper = hyper.clone();
            }
            history.push(AdapterBankTrainingHistoryEntry {
                step,
                loss: final_loss,
                grad_norm: step_report.grad_norm,
                grad_scale: step_report.grad_scale,
                examples_seen: indices.len(),
                adapter_values_per_sec: (indices.len() * hyper.adapter_parameter_count()) as f64
                    / step_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                validation_loss: None,
                memory: check_process_memory_budget(
                    "cpu adapter-bank training report",
                    config.system_memory_budget_gb,
                )?,
                elapsed_ms: step_elapsed.as_secs_f64() * 1000.0,
                train_vector_metrics: None,
                validation_vector_metrics: None,
                flow_optimizer: None,
            });
        }
    }
    if best_loss < final_loss {
        *hyper = best_hyper;
        final_loss = best_loss;
    }
    Ok(AdapterBankTrainingPhaseReport {
        backend: "cpu_reference".to_string(),
        device: "host".to_string(),
        selection_metric: "train_adapter_vector_mse".to_string(),
        initial_loss,
        initial_validation_loss: None,
        final_loss,
        final_validation_loss: None,
        best_loss,
        best_validation_loss: None,
        best_step,
        history,
        memory: vec![capture_process_memory("cpu adapter-bank training complete")],
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        vector_selection: None,
    })
}

fn train_adapter_bank_linear_solve(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let adapter_examples = examples
        .iter()
        .map(|example| {
            Ok(HyperAdapterExample2d {
                condition: example.condition.clone(),
                target_adapter: target_adapter_from_vector(hyper, example)?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let initial_loss = hyper_adapter_regression_loss(hyper, &adapter_examples)?;
    if config.objective == AdapterBankTrainingObjective::RectifiedFlow {
        return train_adapter_bank_flow_linear_solve(
            hyper,
            examples,
            &adapter_examples,
            config,
            initial_loss,
            started,
        );
    }
    let input_dims = hyper.config.condition_feature_dims;
    let hidden_dims = hyper.config.hidden_dims;
    let output_dims = hyper.adapter_parameter_count();
    let rows = examples.len();
    let input_cols = input_dims + 1;
    let mut inputs = vec![0.0_f64; rows * input_cols];
    let mut target_pre = vec![0.0_f64; rows * output_dims];
    for (row, example) in examples.iter().enumerate() {
        let input = hyper.condition_input_vector(&example.condition)?;
        if input.len() != input_dims || example.target_vector.len() != output_dims {
            return Err(
                std::io::Error::other("linear-solve adapter-bank tensor shape mismatch").into(),
            );
        }
        let input_base = row * input_cols;
        for (idx, value) in input.iter().copied().enumerate() {
            inputs[input_base + idx] = f64::from(value);
        }
        inputs[input_base + input_dims] = 1.0;
        let target_base = row * output_dims;
        for (idx, value) in example.target_vector.iter().copied().enumerate() {
            target_pre[target_base + idx] = match hyper.config.output_activation {
                HyperNpa2dOutputActivation::Tanh => {
                    let normalized = (f64::from(value) / f64::from(hyper.config.output_scale))
                        .clamp(-0.999_999, 0.999_999);
                    normalized.atanh()
                }
                HyperNpa2dOutputActivation::Linear => f64::from(value),
            };
        }
    }

    let mut weights = crate::HyperNpa2dWeights::zeros(input_dims, hidden_dims, output_dims);
    let mut precise_weights =
        crate::HyperNpa2dPreciseWeights::zeros(input_dims, hidden_dims, output_dims);
    let backend = if hidden_dims >= rows {
        let hidden_margin = 16.0_f64;
        let mut hidden_targets = vec![-hidden_margin; rows * rows];
        for row in 0..rows {
            hidden_targets[row * rows + row] = hidden_margin;
        }
        let gram = gram_matrix(&inputs, rows, input_cols);
        let inverse = invert_with_jitter(&gram, rows)?;
        let temp = matmul(&inverse, rows, rows, &hidden_targets, rows, rows);
        let coeff = matmul_transpose_left(&inputs, rows, input_cols, &temp, rows, rows);
        for hidden in 0..rows {
            for input in 0..input_dims {
                let value = coeff[input * rows + hidden];
                weights.w1[hidden * input_dims + input] = value as f32;
                precise_weights.w1[hidden * input_dims + input] = value;
            }
            let value = coeff[input_dims * rows + hidden];
            weights.b1[hidden] = value as f32;
            precise_weights.b1[hidden] = value;
        }
        for output in 0..output_dims {
            for row in 0..rows {
                let value = target_pre[row * output_dims + output] / hidden_margin;
                weights.w2[output * hidden_dims + row] = value as f32;
                precise_weights.w2[output * hidden_dims + row] = value;
            }
        }
        "linear_solve_exemplar_condition_interpolator"
    } else {
        if hidden_dims < input_dims * 2 {
            return Err(std::io::Error::other(format!(
                "linear-solve adapter-bank backend requires hidden >= rows ({rows}) or hidden >= {}, got {hidden_dims}",
                input_dims * 2
            ))
            .into());
        }
        let design_cols = hidden_dims + 1;
        let mut design = vec![0.0_f64; rows * design_cols];
        for row in 0..rows {
            let design_base = row * design_cols;
            let input_base = row * input_cols;
            for idx in 0..input_dims {
                let value = inputs[input_base + idx];
                design[design_base + idx] = value.max(0.0);
                design[design_base + input_dims + idx] = (-value).max(0.0);
            }
            design[design_base + hidden_dims] = 1.0;
        }
        let gram = gram_matrix(&design, rows, design_cols);
        let inverse = invert_with_jitter(&gram, rows)?;
        let temp = matmul(&inverse, rows, rows, &target_pre, rows, output_dims);
        let coeff = matmul_transpose_left(&design, rows, design_cols, &temp, rows, output_dims);
        for idx in 0..input_dims {
            weights.w1[idx * input_dims + idx] = 1.0;
            weights.w1[(input_dims + idx) * input_dims + idx] = -1.0;
            precise_weights.w1[idx * input_dims + idx] = 1.0;
            precise_weights.w1[(input_dims + idx) * input_dims + idx] = -1.0;
        }
        for output in 0..output_dims {
            for hidden in 0..hidden_dims {
                let value = coeff[hidden * output_dims + output];
                weights.w2[output * hidden_dims + hidden] = value as f32;
                precise_weights.w2[output * hidden_dims + hidden] = value;
            }
            let value = coeff[hidden_dims * output_dims + output];
            weights.b2[output] = value as f32;
            precise_weights.b2[output] = value;
        }
        "linear_solve_positive_negative_condition_interpolator"
    };
    hyper.weights = weights;
    hyper.precise_weights = Some(precise_weights);
    hyper.validate()?;
    let final_loss = hyper_adapter_regression_loss(hyper, &adapter_examples)?;
    let elapsed = started.elapsed();
    Ok(AdapterBankTrainingPhaseReport {
        backend: backend.to_string(),
        device: "host-f64-solve".to_string(),
        selection_metric: "train_adapter_vector_mse".to_string(),
        initial_loss,
        initial_validation_loss: None,
        final_loss,
        final_validation_loss: None,
        best_loss: final_loss,
        best_validation_loss: None,
        best_step: config.steps,
        history: vec![AdapterBankTrainingHistoryEntry {
            step: config.steps,
            loss: final_loss,
            grad_norm: 0.0,
            grad_scale: 1.0,
            examples_seen: examples.len(),
            adapter_values_per_sec: (examples.len() * output_dims) as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            validation_loss: None,
            memory: check_process_memory_budget(
                "linear-solve adapter-bank training complete",
                config.system_memory_budget_gb,
            )?,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            train_vector_metrics: None,
            validation_vector_metrics: None,
            flow_optimizer: None,
        }],
        memory: vec![capture_process_memory(
            "linear-solve adapter-bank training complete",
        )],
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        vector_selection: None,
    })
}

fn train_adapter_bank_flow_linear_solve(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    adapter_examples: &[HyperAdapterExample2d],
    config: AdapterBankTrainConfig,
    initial_loss: f32,
    started: Instant,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    let output_dims = hyper.adapter_parameter_count();
    let hidden_dims = config.flow.hidden_dims;
    let weights = linear_solve_rectified_flow_condition_weights(hyper, examples, hidden_dims)?;
    hyper.set_flow(crate::HyperNpa2dFlow {
        config: crate::HyperNpa2dFlowConfig {
            hidden_dims,
            sample_steps: config.flow.sample_steps,
            source_scale: config.flow.source_scale,
            sample_seed: config.flow.sample_seed,
            hidden_activation: config.flow.hidden_activation,
        },
        weights,
    })?;
    let final_loss = hyper_adapter_regression_loss(hyper, adapter_examples)?;
    let elapsed = started.elapsed();
    Ok(AdapterBankTrainingPhaseReport {
        backend: "linear_solve_rectified_flow_condition_interpolator".to_string(),
        device: "host-f64-solve".to_string(),
        selection_metric: "train_adapter_vector_mse".to_string(),
        initial_loss,
        initial_validation_loss: None,
        final_loss,
        final_validation_loss: None,
        best_loss: final_loss,
        best_validation_loss: None,
        best_step: config.steps,
        history: vec![AdapterBankTrainingHistoryEntry {
            step: config.steps,
            loss: final_loss,
            grad_norm: 0.0,
            grad_scale: 1.0,
            examples_seen: examples.len(),
            adapter_values_per_sec: (examples.len() * output_dims) as f64
                / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            validation_loss: None,
            memory: check_process_memory_budget(
                "linear-solve rectified-flow adapter-bank training complete",
                config.system_memory_budget_gb,
            )?,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            train_vector_metrics: None,
            validation_vector_metrics: None,
            flow_optimizer: None,
        }],
        memory: vec![capture_process_memory(
            "linear-solve rectified-flow adapter-bank training complete",
        )],
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        vector_selection: None,
    })
}

fn linear_solve_rectified_flow_condition_weights(
    hyper: &HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    hidden_dims: usize,
) -> Result<crate::HyperNpa2dFlowWeights, Box<dyn std::error::Error>> {
    let input_dims = hyper.config.condition_feature_dims;
    let output_dims = hyper.adapter_parameter_count();
    let rows = examples.len();
    if hidden_dims < rows {
        return Err(std::io::Error::other(format!(
            "linear-solve rectified-flow backend requires flow_hidden >= rows ({rows}), got {hidden_dims}"
        ))
        .into());
    }
    let condition_cols = input_dims + 1;
    let flow_input_dims = input_dims
        .checked_add(1)
        .and_then(|dims| dims.checked_add(output_dims))
        .ok_or_else(|| std::io::Error::other("linear-solve flow input dimensions overflow"))?;
    let mut inputs = vec![0.0_f64; rows * condition_cols];
    for (row, example) in examples.iter().enumerate() {
        let input = hyper.condition_input_vector(&example.condition)?;
        if input.len() != input_dims || example.target_vector.len() != output_dims {
            return Err(std::io::Error::other("linear-solve flow tensor shape mismatch").into());
        }
        let input_base = row * condition_cols;
        for (idx, value) in input.iter().copied().enumerate() {
            inputs[input_base + idx] = f64::from(value);
        }
        inputs[input_base + input_dims] = 1.0;
    }

    let hidden_margin = 16.0_f64;
    let mut hidden_targets = vec![-hidden_margin; rows * rows];
    for row in 0..rows {
        hidden_targets[row * rows + row] = hidden_margin;
    }
    let gram = gram_matrix(&inputs, rows, condition_cols);
    let inverse = invert_with_jitter(&gram, rows)?;
    let temp = matmul(&inverse, rows, rows, &hidden_targets, rows, rows);
    let coeff = matmul_transpose_left(&inputs, rows, condition_cols, &temp, rows, rows);

    let mut weights = crate::HyperNpa2dFlowWeights {
        w1: vec![0.0; hidden_dims * flow_input_dims],
        b1: vec![0.0; hidden_dims],
        w2: vec![0.0; output_dims * hidden_dims],
        b2: vec![0.0; output_dims],
    };
    for hidden in 0..rows {
        let flow_base = hidden * flow_input_dims;
        for input in 0..input_dims {
            weights.w1[flow_base + input] = coeff[input * rows + hidden] as f32;
        }
        weights.b1[hidden] = coeff[input_dims * rows + hidden] as f32;
    }
    for output in 0..output_dims {
        for (row, example) in examples.iter().enumerate() {
            let value = example.target_vector[output] as f64 / hidden_margin;
            weights.w2[output * hidden_dims + row] = value as f32;
        }
    }
    Ok(weights)
}

fn gram_matrix(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * rows];
    for row in 0..rows {
        for col in row..rows {
            let mut sum = 0.0;
            for idx in 0..cols {
                sum += values[row * cols + idx] * values[col * cols + idx];
            }
            out[row * rows + col] = sum;
            out[col * rows + row] = sum;
        }
    }
    out
}

fn matmul(
    left: &[f64],
    left_rows: usize,
    inner: usize,
    right: &[f64],
    right_rows: usize,
    right_cols: usize,
) -> Vec<f64> {
    debug_assert_eq!(inner, right_rows);
    let mut out = vec![0.0; left_rows * right_cols];
    for row in 0..left_rows {
        for idx in 0..inner {
            let left_value = left[row * inner + idx];
            if left_value == 0.0 {
                continue;
            }
            for col in 0..right_cols {
                out[row * right_cols + col] += left_value * right[idx * right_cols + col];
            }
        }
    }
    out
}

fn matmul_transpose_left(
    left: &[f64],
    left_rows: usize,
    left_cols: usize,
    right: &[f64],
    right_rows: usize,
    right_cols: usize,
) -> Vec<f64> {
    debug_assert_eq!(left_rows, right_rows);
    let mut out = vec![0.0; left_cols * right_cols];
    for row in 0..left_rows {
        for left_col in 0..left_cols {
            let left_value = left[row * left_cols + left_col];
            if left_value == 0.0 {
                continue;
            }
            for right_col in 0..right_cols {
                out[left_col * right_cols + right_col] +=
                    left_value * right[row * right_cols + right_col];
            }
        }
    }
    out
}

fn invert_with_jitter(matrix: &[f64], size: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    for jitter in [0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6] {
        let mut candidate = matrix.to_vec();
        if jitter > 0.0 {
            for idx in 0..size {
                candidate[idx * size + idx] += jitter;
            }
        }
        if let Some(inverse) = invert_square(candidate, size) {
            return Ok(inverse);
        }
    }
    Err(std::io::Error::other("linear-solve adapter-bank Gram matrix is singular").into())
}

fn invert_square(mut matrix: Vec<f64>, size: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; size * size];
    for idx in 0..size {
        inverse[idx * size + idx] = 1.0;
    }
    for col in 0..size {
        let mut pivot = col;
        let mut pivot_abs = matrix[col * size + col].abs();
        for row in (col + 1)..size {
            let value_abs = matrix[row * size + col].abs();
            if value_abs > pivot_abs {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= 1.0e-14 {
            return None;
        }
        if pivot != col {
            for idx in 0..size {
                matrix.swap(col * size + idx, pivot * size + idx);
                inverse.swap(col * size + idx, pivot * size + idx);
            }
        }
        let pivot_value = matrix[col * size + col];
        for idx in 0..size {
            matrix[col * size + idx] /= pivot_value;
            inverse[col * size + idx] /= pivot_value;
        }
        for row in 0..size {
            if row == col {
                continue;
            }
            let factor = matrix[row * size + col];
            if factor == 0.0 {
                continue;
            }
            for idx in 0..size {
                matrix[row * size + idx] -= factor * matrix[col * size + idx];
                inverse[row * size + idx] -= factor * inverse[col * size + idx];
            }
        }
    }
    Some(inverse)
}

fn load_adapter_bank_initial_flow(
    hyper: &mut HyperNpa2d,
    path: &Path,
    flow_config: crate::HyperNpa2dFlowConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let loaded = load_hyper_2d(path)?;
    if loaded.npa_config != hyper.npa_config {
        return Err(std::io::Error::other(format!(
            "input.initial_hyper {} NPA config does not match the current shared base",
            path.display()
        ))
        .into());
    }
    if loaded.config.condition_encoder != hyper.config.condition_encoder
        || loaded.config.condition_feature_dims != hyper.config.condition_feature_dims
        || loaded.config.condition_token_grid_width != hyper.config.condition_token_grid_width
        || loaded.config.condition_token_grid_height != hyper.config.condition_token_grid_height
    {
        return Err(std::io::Error::other(format!(
            "input.initial_hyper {} condition config does not match the current run",
            path.display()
        ))
        .into());
    }
    if loaded.adapter_parameter_count() != hyper.adapter_parameter_count() {
        return Err(std::io::Error::other(format!(
            "input.initial_hyper {} adapter parameter count {} does not match current count {}",
            path.display(),
            loaded.adapter_parameter_count(),
            hyper.adapter_parameter_count()
        ))
        .into());
    }
    let flow = loaded.flow.ok_or_else(|| {
        std::io::Error::other(format!(
            "input.initial_hyper {} does not contain a flow head",
            path.display()
        ))
    })?;
    if flow.config.hidden_dims != flow_config.hidden_dims {
        return Err(std::io::Error::other(format!(
            "input.initial_hyper {} flow hidden dims {} do not match requested {}",
            path.display(),
            flow.config.hidden_dims,
            flow_config.hidden_dims
        ))
        .into());
    }
    hyper.set_flow(crate::HyperNpa2dFlow {
        config: flow_config,
        weights: flow.weights,
    })?;
    Ok(())
}

#[cfg(feature = "backend_wgpu")]
fn train_adapter_bank_burn_wgpu(
    hyper: &mut HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    validation_examples: Option<&[AdapterBankConditionedExample]>,
    config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    burn_wgpu::train_adapter_bank_burn_wgpu(hyper, examples, validation_examples, config)
}

#[cfg(not(feature = "backend_wgpu"))]
fn train_adapter_bank_burn_wgpu(
    _hyper: &mut HyperNpa2d,
    _examples: &[AdapterBankConditionedExample],
    _validation_examples: Option<&[AdapterBankConditionedExample]>,
    _config: AdapterBankTrainConfig,
) -> Result<AdapterBankTrainingPhaseReport, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "train-hyper2d-adapter-bank backend burn-wgpu requires the backend_wgpu feature",
    )
    .into())
}

fn evaluate_rollout_splits(
    base: &NpaModel,
    hyper: &HyperNpa2d,
    train_examples: &[AdapterBankConditionedExample],
    holdout_examples: &[AdapterBankConditionedExample],
    config: AdapterBankRolloutEvalConfig,
) -> Result<AdapterBankRolloutEvalReport, Box<dyn std::error::Error>> {
    let requested_examples_per_split = config.requested_examples_per_split;
    if requested_examples_per_split == 0 {
        return Ok(AdapterBankRolloutEvalReport {
            requested_examples_per_split,
            train_summary: None,
            holdout_summary: None,
            entries: Vec::new(),
        });
    }
    let mut entries = Vec::new();
    let train_entries = evaluate_rollout_examples(
        base,
        hyper,
        train_examples,
        config,
        requested_examples_per_split,
    )?;
    let train_summary = rollout_summary(&train_entries);
    entries.extend(train_entries);
    let holdout_entries = evaluate_rollout_examples(
        base,
        hyper,
        holdout_examples,
        config,
        requested_examples_per_split,
    )?;
    let holdout_summary = rollout_summary(&holdout_entries);
    entries.extend(holdout_entries);
    Ok(AdapterBankRolloutEvalReport {
        requested_examples_per_split,
        train_summary,
        holdout_summary,
        entries,
    })
}

fn evaluate_rollout_examples(
    base: &NpaModel,
    hyper: &HyperNpa2d,
    examples: &[AdapterBankConditionedExample],
    config: AdapterBankRolloutEvalConfig,
    requested_examples: usize,
) -> Result<Vec<AdapterBankRolloutEntry>, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(Vec::new());
    }
    let indices = eval_indices(examples.len(), requested_examples, config.rollout.seed);
    let mut entries = Vec::with_capacity(indices.len());
    for &idx in &indices {
        let example = &examples[idx];
        let target = super::super::super::target2d::load_target_image_2d_adaptive(
            &example.source.condition_path,
            config.target.threshold,
            config.target.points,
            config.target.image_size,
        )?;
        let static_example = super::DirectBasisExample {
            source: example.source.clone(),
            split: example.split,
            bank_split_index: None,
            target: target.clone(),
            adapter: target_adapter_from_vector(hyper, example)?,
            last_train_loss: example.last_train_loss,
        };
        let zero_example = super::DirectBasisExample {
            source: example.source.clone(),
            split: example.split,
            bank_split_index: None,
            target: target.clone(),
            adapter: zero_adapter_for_hyper(hyper)?,
            last_train_loss: None,
        };
        let predicted_adapter = hyper.predict_adapter(&example.condition)?;
        let hyper_example = super::DirectBasisExample {
            source: example.source.clone(),
            split: example.split,
            bank_split_index: None,
            target,
            adapter: predicted_adapter,
            last_train_loss: None,
        };
        let split_seed = config.rollout.seed.wrapping_add(idx as u64);
        let static_loss = evaluate_direct_basis_example(
            base,
            &static_example,
            &upstream_growing_2d_hashgrid(),
            EvalConfig {
                seed: split_seed,
                ..config.rollout
            },
            config.loss,
        )?;
        let zero_loss = evaluate_direct_basis_example(
            base,
            &zero_example,
            &upstream_growing_2d_hashgrid(),
            EvalConfig {
                seed: split_seed,
                ..config.rollout
            },
            config.loss,
        )?;
        let hyper_loss = evaluate_direct_basis_example(
            base,
            &hyper_example,
            &upstream_growing_2d_hashgrid(),
            EvalConfig {
                seed: split_seed,
                ..config.rollout
            },
            config.loss,
        )?;
        let predicted_vector = hyper.predict_adapter_vector(&example.condition)?;
        let (vector_mse, vector_cosine) =
            vector_pair_metrics(&predicted_vector, &example.target_vector)?;
        let zero_total = zero_loss.total_loss;
        let static_total = static_loss.total_loss;
        let hyper_total = hyper_loss.total_loss;
        let ratio = if static_total.abs() > f32::MIN_POSITIVE {
            hyper_total / static_total
        } else {
            f32::INFINITY
        };
        let zero_ratio = if zero_total.abs() > f32::MIN_POSITIVE {
            hyper_total / zero_total
        } else {
            f32::INFINITY
        };
        entries.push(AdapterBankRolloutEntry {
            slug: example.source.slug.clone(),
            split: example.split.label(),
            condition: example.source.condition_path.display().to_string(),
            target_source_width: example.target_source_width,
            target_source_height: example.target_source_height,
            target_points: example.target_points,
            zero_adapter_loss: zero_loss,
            static_adapter_loss: static_loss,
            hyper_adapter_loss: hyper_loss,
            hyper_gap_to_static: hyper_total - static_total,
            hyper_ratio_to_static: ratio,
            hyper_gap_to_zero: hyper_total - zero_total,
            hyper_ratio_to_zero: zero_ratio,
            adapter_vector_mse: vector_mse,
            adapter_vector_cosine_similarity: vector_cosine,
        });
    }
    Ok(entries)
}

fn rollout_summary(entries: &[AdapterBankRolloutEntry]) -> Option<AdapterBankRolloutSummary> {
    if entries.is_empty() {
        return None;
    }
    let mut mean_static = 0.0;
    let mut mean_zero = 0.0;
    let mut mean_hyper = 0.0;
    let mut mean_gap = 0.0;
    let mut mean_ratio = 0.0;
    let mut max_ratio = 0.0_f32;
    let mut mean_zero_gap = 0.0;
    let mut mean_zero_ratio = 0.0;
    let mut max_zero_ratio = 0.0_f32;
    for entry in entries {
        mean_zero += entry.zero_adapter_loss.total_loss;
        mean_static += entry.static_adapter_loss.total_loss;
        mean_hyper += entry.hyper_adapter_loss.total_loss;
        mean_gap += entry.hyper_gap_to_static;
        mean_ratio += entry.hyper_ratio_to_static;
        max_ratio = max_ratio.max(entry.hyper_ratio_to_static);
        mean_zero_gap += entry.hyper_gap_to_zero;
        mean_zero_ratio += entry.hyper_ratio_to_zero;
        max_zero_ratio = max_zero_ratio.max(entry.hyper_ratio_to_zero);
    }
    let scale = 1.0 / entries.len() as f32;
    Some(AdapterBankRolloutSummary {
        examples: entries.len(),
        mean_zero_loss: mean_zero * scale,
        mean_static_loss: mean_static * scale,
        mean_hyper_loss: mean_hyper * scale,
        mean_gap_to_static: mean_gap * scale,
        mean_ratio_to_static: mean_ratio * scale,
        max_ratio_to_static: max_ratio,
        mean_gap_to_zero: mean_zero_gap * scale,
        mean_ratio_to_zero: mean_zero_ratio * scale,
        max_ratio_to_zero: max_zero_ratio,
    })
}

fn zero_adapter_for_hyper(
    hyper: &HyperNpa2d,
) -> Result<NpaLowRankAdapter, Box<dyn std::error::Error>> {
    if hyper.config.adapter_bias_correction {
        let values = vec![0.0; hyper.adapter_parameter_count()];
        Ok(
            NpaLowRankAdapter::from_parameter_vector_with_bias_correction(
                &hyper.npa_config,
                hyper.config.adapter_rank,
                hyper.config.adapter_alpha,
                values,
                true,
            )?,
        )
    } else {
        Ok(NpaLowRankAdapter::zeros(
            &hyper.npa_config,
            hyper.config.adapter_rank,
            hyper.config.adapter_alpha,
        ))
    }
}

fn vector_pair_metrics(
    predicted: &[f32],
    target: &[f32],
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    if predicted.len() != target.len() {
        return Err(std::io::Error::other("adapter vector metric length mismatch").into());
    }
    let mut sum_sq = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut pred_sq = 0.0_f64;
    let mut target_sq = 0.0_f64;
    for (actual, expected) in predicted.iter().zip(target) {
        let diff = actual - expected;
        sum_sq += (diff as f64) * (diff as f64);
        dot += (*actual as f64) * (*expected as f64);
        pred_sq += (*actual as f64) * (*actual as f64);
        target_sq += (*expected as f64) * (*expected as f64);
    }
    let denom = (pred_sq * target_sq).sqrt();
    let cosine = if denom > f64::MIN_POSITIVE {
        (dot / denom) as f32
    } else {
        0.0
    };
    Ok(((sum_sq / predicted.len() as f64) as f32, cosine))
}

fn target_adapter_from_vector(
    hyper: &HyperNpa2d,
    example: &AdapterBankConditionedExample,
) -> Result<NpaLowRankAdapter, Box<dyn std::error::Error>> {
    NpaLowRankAdapter::from_parameter_vector_with_bias_correction(
        &hyper.npa_config,
        hyper.config.adapter_rank,
        hyper.config.adapter_alpha,
        example.target_vector.clone(),
        hyper.config.adapter_bias_correction,
    )
    .map_err(|err| err.into())
}

fn capture_process_memory(label: impl Into<String>) -> AdapterBankMemorySnapshot {
    let mut snapshot = AdapterBankMemorySnapshot {
        label: label.into(),
        rss_bytes: None,
        peak_rss_bytes: None,
        swap_bytes: None,
    };
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return snapshot;
    };
    for line in status.lines() {
        if let Some(bytes) = parse_status_kib(line, "VmRSS:") {
            snapshot.rss_bytes = Some(bytes);
        } else if let Some(bytes) = parse_status_kib(line, "VmHWM:") {
            snapshot.peak_rss_bytes = Some(bytes);
        } else if let Some(bytes) = parse_status_kib(line, "VmSwap:") {
            snapshot.swap_bytes = Some(bytes);
        }
    }
    snapshot
}

fn check_process_memory_budget(
    label: impl Into<String>,
    budget_gb: Option<f32>,
) -> Result<AdapterBankMemorySnapshot, Box<dyn std::error::Error>> {
    let snapshot = capture_process_memory(label);
    if let (Some(rss), Some(budget_gb)) = (snapshot.rss_bytes, budget_gb)
        && budget_gb.is_finite()
        && budget_gb > 0.0
    {
        let budget_bytes = (f64::from(budget_gb) * 1024.0 * 1024.0 * 1024.0) as u64;
        if rss > budget_bytes {
            return Err(std::io::Error::other(format!(
                "adapter-bank training exceeded system memory budget at {}: rss={:.2} GiB budget={budget_gb:.2} GiB",
                snapshot.label,
                rss as f64 / 1024.0 / 1024.0 / 1024.0
            ))
            .into());
        }
    }
    Ok(snapshot)
}

fn parse_status_kib(line: &str, key: &str) -> Option<u64> {
    let rest = line.strip_prefix(key)?;
    let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
    Some(kib * 1024)
}

fn validate_training_config(
    config: AdapterBankTrainConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.example_batch_size == 0 {
        return Err(std::io::Error::other("example batch size must be non-zero").into());
    }
    if config.loss_eval_batch_size == 0 {
        return Err(std::io::Error::other("loss eval batch size must be non-zero").into());
    }
    if matches!(
        config.objective,
        AdapterBankTrainingObjective::RectifiedFlow
    ) && (config.flow.hidden_dims == 0
        || config.flow.sample_steps == 0
        || !config.flow.source_scale.is_finite()
        || config.flow.source_scale < 0.0)
    {
        return Err(std::io::Error::other(
            "rectified-flow training requires flow_hidden, flow_sample_steps, and finite non-negative flow_source_scale",
        )
        .into());
    }
    if config.flow.loss == AdapterBankFlowLoss::SampledAdapterMse && config.flow.source_scale != 0.0
    {
        return Err(std::io::Error::other(
            "sampled-adapter flow loss currently requires flow_source_scale = 0.0",
        )
        .into());
    }
    if config.flow.sample_weights.enabled
        && config.flow.loss != AdapterBankFlowLoss::SampledAdapterMse
    {
        return Err(std::io::Error::other(
            "flow hard-sample weighting currently requires flow_loss = \"sampled-adapter-mse\"",
        )
        .into());
    }
    if let Some(memory_budget_gb) = config.system_memory_budget_gb
        && (!memory_budget_gb.is_finite() || memory_budget_gb < 0.0)
    {
        return Err(
            std::io::Error::other("system memory budget must be finite and non-negative").into(),
        );
    }
    let optimizer = config.optimizer;
    if !optimizer.learning_rate.is_finite() || optimizer.learning_rate <= 0.0 {
        return Err(std::io::Error::other("learning rate must be finite and positive").into());
    }
    if !optimizer.weight_decay.is_finite() || optimizer.weight_decay < 0.0 {
        return Err(std::io::Error::other("weight decay must be finite and non-negative").into());
    }
    if !optimizer.grad_clip_norm.is_finite() || optimizer.grad_clip_norm < 0.0 {
        return Err(std::io::Error::other("grad clip norm must be finite and non-negative").into());
    }
    if !optimizer.beta1.is_finite() || !(0.0..1.0).contains(&optimizer.beta1) {
        return Err(std::io::Error::other("adam beta1 must be finite and in [0, 1)").into());
    }
    if !optimizer.beta2.is_finite() || !(0.0..1.0).contains(&optimizer.beta2) {
        return Err(std::io::Error::other("adam beta2 must be finite and in [0, 1)").into());
    }
    if !optimizer.epsilon.is_finite() || optimizer.epsilon <= 0.0 {
        return Err(std::io::Error::other("adam epsilon must be finite and positive").into());
    }
    Ok(())
}

impl AdapterBankTrainingObjective {
    const fn label(self) -> &'static str {
        match self {
            Self::StaticVectorMse => "static-vector-mse",
            Self::RectifiedFlow => "rectified-flow",
        }
    }
}

impl AdapterBankFlowInit {
    const fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::LinearSolveConditionWarmstart => "linear-solve-condition-warmstart",
            Self::FromHyper => "from-hyper",
        }
    }
}

impl AdapterBankFlowLoss {
    const fn label(self) -> &'static str {
        match self {
            Self::VelocityMse => "velocity-mse",
            Self::SampledAdapterMse => "sampled-adapter-mse",
        }
    }
}

fn parse_adapter_bank_training_objective(
    value: &str,
) -> Result<AdapterBankTrainingObjective, Box<dyn std::error::Error>> {
    match value {
        "static-vector-mse" | "static" | "mse" => Ok(AdapterBankTrainingObjective::StaticVectorMse),
        "rectified-flow" | "flow" | "rectified-flow-lora-vector" => {
            Ok(AdapterBankTrainingObjective::RectifiedFlow)
        }
        other => Err(std::io::Error::other(format!(
            "unknown training.objective {other:?}; expected static-vector-mse or rectified-flow"
        ))
        .into()),
    }
}

fn parse_adapter_bank_flow_init(
    value: &str,
) -> Result<AdapterBankFlowInit, Box<dyn std::error::Error>> {
    match value {
        "random" | "seeded-random" => Ok(AdapterBankFlowInit::Random),
        "linear-solve-condition-warmstart"
        | "linear-solve"
        | "condition-interpolator"
        | "linear-solve-condition" => Ok(AdapterBankFlowInit::LinearSolveConditionWarmstart),
        "from-hyper" | "hyper" | "checkpoint" | "resume" => Ok(AdapterBankFlowInit::FromHyper),
        other => Err(std::io::Error::other(format!(
            "unknown training.flow_init {other:?}; expected random, linear-solve-condition-warmstart, or from-hyper"
        ))
        .into()),
    }
}

fn parse_adapter_bank_flow_loss(
    value: &str,
) -> Result<AdapterBankFlowLoss, Box<dyn std::error::Error>> {
    match value {
        "velocity-mse" | "velocity" | "rectified-flow-velocity" => {
            Ok(AdapterBankFlowLoss::VelocityMse)
        }
        "sampled-adapter-mse"
        | "sampled-adapter"
        | "sampled"
        | "final-adapter"
        | "inference-adapter-mse" => Ok(AdapterBankFlowLoss::SampledAdapterMse),
        other => Err(std::io::Error::other(format!(
            "unknown training.flow_loss {other:?}; expected velocity-mse or sampled-adapter-mse"
        ))
        .into()),
    }
}

fn parse_adapter_bank_flow_hidden_activation(
    value: &str,
) -> Result<HyperNpa2dFlowActivation, Box<dyn std::error::Error>> {
    match value {
        "relu" => Ok(HyperNpa2dFlowActivation::Relu),
        "leaky-relu" | "leaky_relu" | "leaky" => Ok(HyperNpa2dFlowActivation::LeakyRelu),
        other => Err(std::io::Error::other(format!(
            "unknown training.flow_hidden_activation {other:?}; expected relu or leaky-relu"
        ))
        .into()),
    }
}

fn normalized_batch_size(requested: usize, examples_len: usize) -> usize {
    if requested == 0 {
        examples_len
    } else {
        requested.min(examples_len).max(1)
    }
}

fn sample_indices(examples_len: usize, batch_size: usize, rng: &mut StdRng) -> Vec<usize> {
    if batch_size >= examples_len {
        return (0..examples_len).collect();
    }
    let mut indices = std::collections::BTreeSet::new();
    while indices.len() < batch_size {
        indices.insert(rng.random_range(0..examples_len));
    }
    indices.into_iter().collect()
}

#[allow(clippy::too_many_arguments, dead_code)]
fn append_rectified_flow_training_row(
    features: &[f32],
    target: &[f32],
    condition_dims: usize,
    output_dims: usize,
    idx: usize,
    source_scale: f32,
    flow_sample_seed: u64,
    sample_seed: u64,
    input_values: &mut Vec<f32>,
    velocity_values: &mut Vec<f32>,
) {
    let feature_start = idx * condition_dims;
    let target_start = idx * output_dims;
    let condition = &features[feature_start..feature_start + condition_dims];
    let target = &target[target_start..target_start + output_dims];
    let mut rng = StdRng::seed_from_u64(
        flow_sample_seed ^ sample_seed ^ ((idx as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
    );
    let t = rng.random_range(0.0..=1.0);
    input_values.extend_from_slice(condition);
    input_values.push(t);
    for &target_value in target {
        let source = if source_scale == 0.0 {
            0.0
        } else {
            rng.random_range(-source_scale..=source_scale)
        };
        let state = source.mul_add(1.0 - t, target_value * t);
        input_values.push(state);
        velocity_values.push(target_value - source);
    }
}

fn condition_encoder_label(encoder: ConditionEncoder2d) -> &'static str {
    match encoder {
        ConditionEncoder2d::SummaryTokens => "summary-pooled-token-grid-v1",
        ConditionEncoder2d::DinoVitsClsPatchMean => "dino-vits-cls-patch-mean-v1",
        ConditionEncoder2d::DinoVitsPatchStats => "dino-vits-patch-stats-v1",
        ConditionEncoder2d::DinoVitsTokenGrid => "dino-vits-token-grid-v1",
    }
}

fn condition_encoder_report_label(
    encoder: ConditionEncoder2d,
    token_grid_width: usize,
    token_grid_height: usize,
) -> String {
    match encoder {
        ConditionEncoder2d::DinoVitsTokenGrid => {
            let token_grid_width = if token_grid_width == 0 {
                crate::DEFAULT_DINO_VITS_TOKEN_GRID_WIDTH
            } else {
                token_grid_width
            };
            let token_grid_height = if token_grid_height == 0 {
                crate::DEFAULT_DINO_VITS_TOKEN_GRID_HEIGHT
            } else {
                token_grid_height
            };
            format!("dino-vits-token-grid-{token_grid_width}x{token_grid_height}-v1")
        }
        _ => condition_encoder_label(encoder).to_string(),
    }
}

#[cfg(feature = "backend_wgpu")]
mod burn_wgpu;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::commands::hyper_e2e::Hyper2dE2eSplit;

    #[test]
    fn adapter_bank_config_accepts_nested_toml() {
        let config: AdapterBankExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [input]
            shared_base = "base.bpk"
            adapter_bank = "bank.json"
            source_limit = 16

            [training]
            backend = "burn-wgpu"
            objective = "rectified-flow"
            hidden = 64
            output_scale = 0.05
            canonicalize_adapters = true
            flow_hidden = 128
            flow_sample_steps = 8
            flow_source_scale = 0.25
            flow_hidden_activation = "leaky-relu"
            flow_init = "linear-solve-condition-warmstart"
            diagnostic_vector_examples = 16
            loss_eval_batch_size = 128
            system_memory_budget_gb = 12.5

            [condition]
            encoder = "dino-vits-token-grid"
            dino_model = "models/dino/dino_vits.mpk"
            dino_image_size = 518
            dino_batch_size = 4
            dino_cache_write_interval_batches = 8
            feature_cache = "artifacts/dino_features.json"
            token_grid_width = 8
            token_grid_height = 8

            [eval]
            seed_mode = "uniform-circle"
            "#,
        )
        .unwrap();
        assert_eq!(config.input.source_limit, Some(16));
        assert_eq!(config.training.objective.as_deref(), Some("rectified-flow"));
        assert_eq!(config.training.hidden, Some(64));
        assert_eq!(config.training.canonicalize_adapters, Some(true));
        assert_eq!(config.training.flow_hidden, Some(128));
        assert_eq!(config.training.flow_sample_steps, Some(8));
        assert_eq!(config.training.flow_source_scale, Some(0.25));
        assert_eq!(
            config.training.flow_hidden_activation.as_deref(),
            Some("leaky-relu")
        );
        assert_eq!(
            config.training.flow_init.as_deref(),
            Some("linear-solve-condition-warmstart")
        );
        assert_eq!(config.training.diagnostic_vector_examples, Some(16));
        assert_eq!(config.training.loss_eval_batch_size, Some(128));
        assert_eq!(config.training.system_memory_budget_gb, Some(12.5));
        assert_eq!(config.condition.dino_batch_size, Some(4));
        assert_eq!(config.condition.dino_cache_write_interval_batches, Some(8));
        assert_eq!(config.condition.token_grid_width, Some(8));
        assert_eq!(config.condition.token_grid_height, Some(8));
        assert_eq!(config.eval.seed_mode.as_deref(), Some("uniform-circle"));
    }

    #[test]
    fn flow_init_parser_accepts_warmstart_aliases() {
        assert_eq!(
            parse_adapter_bank_flow_init("random").unwrap(),
            AdapterBankFlowInit::Random
        );
        assert_eq!(
            parse_adapter_bank_flow_init("linear-solve").unwrap(),
            AdapterBankFlowInit::LinearSolveConditionWarmstart
        );
        assert_eq!(
            parse_adapter_bank_flow_init("from-hyper").unwrap(),
            AdapterBankFlowInit::FromHyper
        );
        assert!(parse_adapter_bank_flow_init("elementwise-mean").is_err());
    }

    #[test]
    fn flow_loss_parser_accepts_sampled_adapter() {
        assert_eq!(
            parse_adapter_bank_flow_loss("velocity-mse").unwrap(),
            AdapterBankFlowLoss::VelocityMse
        );
        assert_eq!(
            parse_adapter_bank_flow_loss("sampled-adapter-mse").unwrap(),
            AdapterBankFlowLoss::SampledAdapterMse
        );
        assert!(parse_adapter_bank_flow_loss("image").is_err());
    }

    #[test]
    fn flow_hidden_activation_parser_accepts_leaky_relu() {
        assert_eq!(
            parse_adapter_bank_flow_hidden_activation("relu").unwrap(),
            HyperNpa2dFlowActivation::Relu
        );
        assert_eq!(
            parse_adapter_bank_flow_hidden_activation("leaky-relu").unwrap(),
            HyperNpa2dFlowActivation::LeakyRelu
        );
        assert!(parse_adapter_bank_flow_hidden_activation("gelu").is_err());
    }

    #[test]
    fn zero_source_rectified_flow_row_matches_target_velocity() {
        let features = vec![0.25, -0.5];
        let target = vec![1.0, -2.0, 0.5];
        let mut input = Vec::new();
        let mut velocity = Vec::new();
        append_rectified_flow_training_row(
            &features,
            &target,
            2,
            3,
            0,
            0.0,
            17,
            29,
            &mut input,
            &mut velocity,
        );
        assert_eq!(&input[..2], &features);
        let t = input[2];
        assert!((0.0..=1.0).contains(&t));
        assert_eq!(velocity, target);
        for (state, target_value) in input[3..].iter().zip(target) {
            assert!((*state - target_value * t).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn noisy_rectified_flow_row_is_bounded_and_consistent() {
        let features = vec![0.25, -0.5];
        let target = vec![1.0, -2.0, 0.5];
        let mut input = Vec::new();
        let mut velocity = Vec::new();
        append_rectified_flow_training_row(
            &features,
            &target,
            2,
            3,
            0,
            0.125,
            17,
            29,
            &mut input,
            &mut velocity,
        );
        let t = input[2];
        for ((state, target_value), velocity_value) in input[3..].iter().zip(target).zip(velocity) {
            let source = target_value - velocity_value;
            assert!(source.abs() <= 0.125);
            let expected_state = source.mul_add(1.0 - t, target_value * t);
            assert!((*state - expected_state).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn target_vector_stats_auto_scale_covers_train_range() {
        let example = AdapterBankConditionedExample {
            source: Hyper2dScratchSource {
                slug: "a".to_string(),
                title: None,
                group: None,
                condition_path: PathBuf::from("a.png"),
                particles: None,
                seed_scale: None,
                update_prob: None,
            },
            split: Hyper2dE2eSplit::Train,
            condition: ConditionImage2d::from_rgb(1, 1, vec![0.0, 0.0, 0.0]).unwrap(),
            target_vector: vec![-0.25, 0.5],
            target_has_bias_correction: false,
            target_source_width: 1,
            target_source_height: 1,
            target_points: 1,
            last_train_loss: None,
            sample_weight: 1.0,
        };
        let stats = target_vector_stats(&[example], 0.0).unwrap();
        assert!(stats.output_scale > 0.5);
        assert_eq!(stats.target_values_outside_output_scale_fraction, 0.0);
    }
}
