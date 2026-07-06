use crate::cli::commands::hyper_support::{
    attach_condition_features, load_condition_image_2d, save_hyper_2d, write_pretty_json,
};
use crate::cli::prelude::*;

use super::super::sources::Hyper2dScratchSource;
use super::super::{
    DinoConditionFeatureCacheConfig, Hyper2dE2eSplit, build_condition_feature_cache,
    default_dino_cache_write_interval_batches, default_dino_feature_batch_size,
};
use super::{
    DirectBasisAdapterBankLoadEntry, DirectBasisTargetConfig, EvalConfig, config_value_enum,
    eval_indices, evaluate_direct_basis_example, load_direct_basis_adapter_bank,
    parse_direct_basis_split, resolve_direct_basis_artifact_path,
};

#[derive(Clone)]
struct AdapterBankConditionedExample {
    source: Hyper2dScratchSource,
    split: Hyper2dE2eSplit,
    condition: ConditionImage2d,
    target_vector: Vec<f32>,
    target_has_bias_correction: bool,
    target_source_width: usize,
    target_source_height: usize,
    target_points: usize,
    last_train_loss: Option<f32>,
}

#[derive(Clone, Copy)]
struct AdapterBankRolloutEvalConfig {
    target: DirectBasisTargetConfig,
    rollout: EvalConfig,
    loss: Target2dLossConfig,
    requested_examples_per_split: usize,
}

#[derive(Clone, Copy)]
struct AdapterBankTrainConfig {
    objective: AdapterBankTrainingObjective,
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    loss_eval_batch_size: usize,
    system_memory_budget_gb: Option<f32>,
    seed: u64,
    optimizer: AdamWConfig,
    flow: AdapterBankFlowTrainConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterBankTrainingObjective {
    StaticVectorMse,
    RectifiedFlow,
}

#[derive(Clone, Copy)]
struct AdapterBankFlowTrainConfig {
    hidden_dims: usize,
    sample_steps: usize,
    source_scale: f32,
    sample_seed: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankExperimentConfig {
    preset: Option<String>,
    input: AdapterBankInputExperimentConfig,
    output: AdapterBankOutputExperimentConfig,
    condition: AdapterBankConditionExperimentConfig,
    training: AdapterBankTrainingExperimentConfig,
    eval: AdapterBankEvalExperimentConfig,
    target: AdapterBankTargetExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankInputExperimentConfig {
    shared_base: Option<PathBuf>,
    adapter_bank: Option<PathBuf>,
    source_limit: Option<usize>,
    train_limit: Option<usize>,
    holdout_limit: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankOutputExperimentConfig {
    output_dir: Option<PathBuf>,
    report_output: Option<PathBuf>,
    hyper_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankConditionExperimentConfig {
    encoder: Option<String>,
    dino_model: Option<PathBuf>,
    dino_image_size: Option<usize>,
    dino_batch_size: Option<usize>,
    dino_cache_write_interval_batches: Option<usize>,
    feature_cache: Option<PathBuf>,
    token_grid_width: Option<usize>,
    token_grid_height: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankTrainingExperimentConfig {
    backend: Option<String>,
    objective: Option<String>,
    hidden: Option<usize>,
    output_scale: Option<f32>,
    linear_output: Option<bool>,
    canonicalize_adapters: Option<bool>,
    flow_hidden: Option<usize>,
    flow_sample_steps: Option<usize>,
    flow_source_scale: Option<f32>,
    flow_sample_seed: Option<u64>,
    loss_eval_batch_size: Option<usize>,
    system_memory_budget_gb: Option<f32>,
    seed: Option<u64>,
    steps: Option<usize>,
    report_interval: Option<usize>,
    example_batch_size: Option<usize>,
    learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
    adam_beta1: Option<f32>,
    adam_beta2: Option<f32>,
    adam_epsilon: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankEvalExperimentConfig {
    vector_examples: Option<usize>,
    rollout_examples: Option<usize>,
    particles: Option<usize>,
    steps: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AdapterBankTargetExperimentConfig {
    points: Option<usize>,
    image_size: Option<usize>,
    threshold: Option<f32>,
    loss_image_size: Option<usize>,
    splat_sigma: Option<f32>,
    splat_loss_weight: Option<f32>,
    color_loss_weight: Option<f32>,
    density_loss_weight: Option<f32>,
    displacement_regularizer_weight: Option<f32>,
    overflow_regularizer_weight: Option<f32>,
    bound_regularizer_weight: Option<f32>,
}

#[derive(Serialize)]
struct AdapterBankConditionedTrainingReport {
    experiment_config: Option<String>,
    preset: AutomataPreset,
    shared_base: String,
    adapter_bank: String,
    adapter_bank_base_model: String,
    output_dir: String,
    report_output: String,
    hyper_output: String,
    backend: Hyper2dAdapterBankBackendArg,
    npa_config: NpaConfig,
    hashgrid: burn_automata_kernels::HashGridConfig,
    hyper_config: HyperNpa2dConfig,
    generator_architecture: &'static str,
    generator_objective: &'static str,
    adapter_rank: usize,
    adapter_alpha: f32,
    adapter_parameter_count: usize,
    condition_encoder: String,
    train_examples: usize,
    holdout_examples: usize,
    source_limit: usize,
    train_limit: usize,
    holdout_limit: usize,
    target_stats: AdapterBankTargetVectorStats,
    requested_training: AdapterBankTrainingSettingsReport,
    adapter_target_canonicalization: &'static str,
    memory: Vec<AdapterBankMemorySnapshot>,
    training: AdapterBankTrainingPhaseReport,
    train_vector_metrics: AdapterBankVectorMetricsReport,
    holdout_vector_metrics: Option<AdapterBankVectorMetricsReport>,
    rollout_particles: usize,
    rollout_steps: usize,
    target_points: usize,
    target_loss_config: Target2dLossConfig,
    rollout_eval: AdapterBankRolloutEvalReport,
}

#[derive(Serialize)]
struct AdapterBankTrainingSettingsReport {
    objective: &'static str,
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    loss_eval_batch_size: usize,
    system_memory_budget_gb: Option<f32>,
    seed: u64,
    optimizer: AdamWConfig,
    flow: Option<AdapterBankFlowTrainingSettingsReport>,
}

#[derive(Serialize)]
struct AdapterBankFlowTrainingSettingsReport {
    hidden_dims: usize,
    sample_steps: usize,
    source_scale: f32,
    sample_seed: u64,
}

#[derive(Clone, Serialize)]
struct AdapterBankTrainingPhaseReport {
    backend: String,
    device: String,
    selection_metric: String,
    initial_loss: f32,
    initial_validation_loss: Option<f32>,
    final_loss: f32,
    final_validation_loss: Option<f32>,
    best_loss: f32,
    best_validation_loss: Option<f32>,
    best_step: usize,
    history: Vec<AdapterBankTrainingHistoryEntry>,
    memory: Vec<AdapterBankMemorySnapshot>,
    elapsed_ms: f64,
}

#[derive(Clone, Serialize)]
struct AdapterBankTrainingHistoryEntry {
    step: usize,
    loss: f32,
    grad_norm: f32,
    grad_scale: f32,
    examples_seen: usize,
    adapter_values_per_sec: f64,
    validation_loss: Option<f32>,
    memory: AdapterBankMemorySnapshot,
    elapsed_ms: f64,
}

#[derive(Clone, Serialize)]
struct AdapterBankMemorySnapshot {
    label: String,
    rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    swap_bytes: Option<u64>,
}

#[derive(Clone, Copy, Serialize)]
struct AdapterBankTargetVectorStats {
    examples: usize,
    parameters_per_adapter: usize,
    mean_rms: f32,
    mean_abs: f32,
    max_abs: f32,
    output_scale: f32,
    target_values_outside_output_scale_fraction: f32,
}

#[derive(Clone, Copy, Serialize)]
struct AdapterBankVectorMetricsReport {
    examples: usize,
    parameters_per_adapter: usize,
    mse: f32,
    rmse: f32,
    normalized_rmse_to_target_rms: Option<f32>,
    mean_abs_error: f32,
    max_abs_error: f32,
    target_rms: f32,
    prediction_rms: f32,
    target_max_abs: f32,
    prediction_max_abs: f32,
    mean_cosine_similarity: f32,
    prediction_values_near_output_scale_fraction: f32,
    target_values_outside_output_scale_fraction: f32,
}

#[derive(Serialize)]
struct AdapterBankRolloutEvalReport {
    requested_examples_per_split: usize,
    train_summary: Option<AdapterBankRolloutSummary>,
    holdout_summary: Option<AdapterBankRolloutSummary>,
    entries: Vec<AdapterBankRolloutEntry>,
}

#[derive(Clone, Copy, Serialize)]
struct AdapterBankRolloutSummary {
    examples: usize,
    mean_zero_loss: f32,
    mean_static_loss: f32,
    mean_hyper_loss: f32,
    mean_gap_to_static: f32,
    mean_ratio_to_static: f32,
    max_ratio_to_static: f32,
    mean_gap_to_zero: f32,
    mean_ratio_to_zero: f32,
    max_ratio_to_zero: f32,
}

#[derive(Serialize)]
struct AdapterBankRolloutEntry {
    slug: String,
    split: &'static str,
    condition: String,
    target_source_width: usize,
    target_source_height: usize,
    target_points: usize,
    zero_adapter_loss: Target2dLossReport,
    static_adapter_loss: Target2dLossReport,
    hyper_adapter_loss: Target2dLossReport,
    hyper_gap_to_static: f32,
    hyper_ratio_to_static: f32,
    hyper_gap_to_zero: f32,
    hyper_ratio_to_zero: f32,
    adapter_vector_mse: f32,
    adapter_vector_cosine_similarity: f32,
}

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
        output: config_output,
        condition: config_condition,
        training: config_training,
        eval: config_eval,
        target: config_target,
    } = experiment_config;
    let AdapterBankInputExperimentConfig {
        shared_base: config_shared_base,
        adapter_bank: config_adapter_bank,
        source_limit: config_source_limit,
        train_limit: config_train_limit,
        holdout_limit: config_holdout_limit,
    } = config_input;
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
    let selected_entries =
        select_adapter_bank_entries(bank.entries, source_limit, train_limit, holdout_limit)?;
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
    let examples = load_conditioned_adapter_bank_examples(
        &adapter_bank,
        &base_manifest,
        selected_entries,
        Some(&condition_features),
        canonicalize_adapters,
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
    let train_config = AdapterBankTrainConfig {
        objective,
        steps,
        report_interval,
        example_batch_size,
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
        target_splat_loss_weight,
        target_color_loss_weight,
        target_density_loss_weight,
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
        target_stats,
        requested_training: AdapterBankTrainingSettingsReport {
            objective: objective.label(),
            steps,
            report_interval,
            example_batch_size,
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
) -> Result<Vec<DirectBasisAdapterBankLoadEntry>, Box<dyn std::error::Error>> {
    let mut selected = Vec::new();
    let mut train_count = 0usize;
    let mut holdout_count = 0usize;
    for entry in entries {
        if source_limit > 0 && selected.len() >= source_limit {
            break;
        }
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
    Ok(selected)
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
        });
    }
    Ok(examples)
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
        }],
        memory: vec![capture_process_memory(
            "linear-solve adapter-bank training complete",
        )],
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    })
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
            target: target.clone(),
            adapter: target_adapter_from_vector(hyper, example)?,
            last_train_loss: example.last_train_loss,
        };
        let zero_example = super::DirectBasisExample {
            source: example.source.clone(),
            split: example.split,
            target: target.clone(),
            adapter: zero_adapter_for_hyper(hyper)?,
            last_train_loss: None,
        };
        let predicted_adapter = hyper.predict_adapter(&example.condition)?;
        let hyper_example = super::DirectBasisExample {
            source: example.source.clone(),
            split: example.split,
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
        || config.flow.source_scale <= 0.0)
    {
        return Err(std::io::Error::other(
            "rectified-flow training requires flow_hidden, flow_sample_steps, and positive finite flow_source_scale",
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
        assert_eq!(config.training.loss_eval_batch_size, Some(128));
        assert_eq!(config.training.system_memory_budget_gb, Some(12.5));
        assert_eq!(config.condition.dino_batch_size, Some(4));
        assert_eq!(config.condition.dino_cache_write_interval_batches, Some(8));
        assert_eq!(config.condition.token_grid_width, Some(8));
        assert_eq!(config.condition.token_grid_height, Some(8));
        assert_eq!(config.eval.seed_mode.as_deref(), Some("uniform-circle"));
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
        };
        let stats = target_vector_stats(&[example], 0.0).unwrap();
        assert!(stats.output_scale > 0.5);
        assert_eq!(stats.target_values_outside_output_scale_fraction, 0.0);
    }
}
