use crate::cli::prelude::*;

use super::shared_basis::{
    add_adapter_l2_gradients, add_scaled_model_gradients, normalized_example_batch_size,
    sample_example_indices, zero_model_gradients,
};
use super::sources::{
    OmniSvgSourceConfig, ScratchSourceResolveConfig, resolve_scratch_sources, sanitize_slug,
};
use super::{Hyper2dE2eSplit, resolve_e2e_splits};
use crate::cli::commands::hyper_support::write_pretty_json;
use std::collections::HashMap;

mod conditioned;
mod oracle;
mod psnr_gate;

pub(in crate::cli::commands::hyper_e2e) use crate::hyper::e2e::{
    PerceptionRolloutBackend, Target2dLossBackend,
};
use crate::hyper::e2e_training::dense;
use crate::hyper::e2e_training::{
    DirectBasisStepStats, DirectBasisTrainConfig, DirectBasisTrainingExample,
};
pub(crate) use conditioned::run_train_hyper_2d_adapter_bank;
pub(crate) use psnr_gate::run_validate_hyper_2d_psnr_gate;

use oracle::evaluate_direct_basis_oracles;

const DEFAULT_WGPU_VRAM_BUDGET_GB: f32 = 64.0;
const WGPU_VRAM_ESTIMATE_MULTIPLIER: u64 = 160;

#[derive(Clone)]
struct DirectBasisExample {
    source: super::sources::Hyper2dScratchSource,
    split: Hyper2dE2eSplit,
    bank_split_index: Option<usize>,
    target: TargetImage2d,
    adapter: NpaLowRankAdapter,
    last_train_loss: Option<f32>,
}

impl DirectBasisExample {
    fn to_training_example(&self) -> DirectBasisTrainingExample {
        DirectBasisTrainingExample {
            target: self.target.clone(),
            adapter: self.adapter.clone(),
            last_train_loss: self.last_train_loss,
            particle_count: self.source.particles,
            update_prob: self.source.update_prob,
            seed_scale: self.source.seed_scale,
        }
    }

    fn sync_from_training_example(&mut self, trained: &DirectBasisTrainingExample) {
        self.adapter = trained.adapter.clone();
        self.last_train_loss = trained.last_train_loss;
    }
}

fn direct_basis_training_examples(
    examples: &[DirectBasisExample],
) -> Vec<DirectBasisTrainingExample> {
    examples
        .iter()
        .map(DirectBasisExample::to_training_example)
        .collect()
}

fn sync_direct_basis_training_examples(
    examples: &mut [DirectBasisExample],
    trained: &[DirectBasisTrainingExample],
) {
    for (example, trained) in examples.iter_mut().zip(trained) {
        example.sync_from_training_example(trained);
    }
}

struct DirectBasisPhaseReport {
    history: Vec<CliHyper2dDirectBasisHistoryEntry>,
    best_loss: Option<f32>,
    best_step: usize,
}

#[derive(Clone, Debug, Serialize)]
struct DirectBasisWgpuMemoryPreflightReport {
    training_requested: bool,
    train_examples: usize,
    holdout_examples: usize,
    max_training_particles: usize,
    max_dense_train_particles: usize,
    max_phase_batch_size: usize,
    rollout_steps: usize,
    tbptt_chunk_steps: usize,
    target_pixels: usize,
    estimated_graph_bytes: u64,
    estimated_target_cache_bytes: u64,
    estimated_peak_bytes: u64,
    memory_budget_bytes: Option<u64>,
    estimated_vram_bytes: u64,
    gpu_memory_budget_bytes: Option<u64>,
    vram_estimate_multiplier: u64,
    dense_train_particle_cap_passed: bool,
    memory_budget_passed: bool,
    gpu_memory_budget_passed: bool,
}

#[derive(Clone)]
struct DirectBasisOracleConfig {
    backend: DirectBasisOracleBackendArg,
    gpu_device: String,
    resume_existing: bool,
    gpu_parallel_jobs: usize,
    train_examples: usize,
    holdout_examples: usize,
    epochs: usize,
    repetitions: usize,
    report_interval: usize,
    batch_size: usize,
    pool_size: usize,
    rollout_step_min: usize,
    tbptt_chunk_steps: usize,
    loss_on_final_chunk_only: bool,
    use_particle_pool: bool,
    inject_seed_interval: usize,
    brush_size: f32,
    learning_rate: f32,
    weight_decay: f32,
    grad_clip_norm: f32,
    seed: u64,
}

#[derive(Serialize)]
struct DirectBasisAdapterBankManifest {
    base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    entries: Vec<CliHyper2dDirectBasisAdapterReport>,
}

#[derive(Deserialize)]
struct DirectBasisAdapterBankLoadManifest {
    base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    entries: Vec<DirectBasisAdapterBankLoadEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct DirectBasisAdapterBankLoadEntry {
    slug: String,
    split: String,
    title: Option<String>,
    group: Option<String>,
    condition: String,
    adapter_output: String,
    #[serde(default)]
    target_source_width: usize,
    #[serde(default)]
    target_source_height: usize,
    #[serde(default)]
    target_points: usize,
    #[serde(default)]
    last_train_loss: Option<f32>,
}

type DirectBasisAdapterBankIndexedEntry = (usize, DirectBasisAdapterBankLoadEntry);
type DirectBasisAdapterBankSplitEntries = (
    Vec<DirectBasisAdapterBankIndexedEntry>,
    Vec<DirectBasisAdapterBankIndexedEntry>,
);
type DirectBasisAdapterBankSelectedEntries = (
    Vec<DirectBasisAdapterBankIndexedEntry>,
    Vec<DirectBasisAdapterBankIndexedEntry>,
    DirectBasisAdapterBankSelectionReport,
);

#[derive(Serialize)]
struct DirectBasisOracleValidationReport {
    preset: AutomataPreset,
    shared_base: String,
    adapter_bank: String,
    report_output: String,
    adapter_bank_base_model: String,
    adapter_rank: usize,
    adapter_alpha: f32,
    npa_config: NpaConfig,
    hashgrid: burn_automata_kernels::HashGridConfig,
    target_loss_config: Target2dLossConfig,
    target_threshold: f32,
    target_points_fallback: usize,
    target_image_size: Option<usize>,
    train_examples: usize,
    holdout_examples: usize,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    eval_seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    per_parameter_grad_normalization: bool,
    selection: DirectBasisAdapterBankSelectionReport,
    oracle_validation: Option<CliHyper2dDirectBasisOracleReport>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisAdapterBankSelectionExperimentConfig {
    selection_seed: Option<u64>,
    selection_manifest: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DirectBasisAdapterBankSelectionManifest {
    selection_seed: u64,
    train: Vec<DirectBasisAdapterBankSelectionEntry>,
    holdout: Vec<DirectBasisAdapterBankSelectionEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DirectBasisAdapterBankSelectionEntry {
    split: String,
    slug: String,
    bank_split_index: usize,
}

#[derive(Clone, Debug, Serialize)]
struct DirectBasisAdapterBankSelectionReport {
    selection_seed: u64,
    selection_manifest: Option<String>,
    replayed_manifest: bool,
    train_requested: usize,
    holdout_requested: usize,
    train_selected: usize,
    holdout_selected: usize,
    train: Vec<DirectBasisAdapterBankSelectionEntry>,
    holdout: Vec<DirectBasisAdapterBankSelectionEntry>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisExperimentConfig {
    preset: Option<String>,
    source: DirectBasisSourceExperimentConfig,
    split: DirectBasisSplitExperimentConfig,
    output: DirectBasisOutputExperimentConfig,
    training: DirectBasisTrainingExperimentConfig,
    gpu: DirectBasisGpuExperimentConfig,
    adapter: DirectBasisAdapterExperimentConfig,
    rollout: DirectBasisRolloutExperimentConfig,
    target: DirectBasisTargetExperimentConfig,
    optimizer: DirectBasisOptimizerExperimentConfig,
    train_refine: DirectBasisAdapterPhaseExperimentConfig,
    holdout_adapter: DirectBasisAdapterPhaseExperimentConfig,
    eval: DirectBasisEvalExperimentConfig,
    oracle: DirectBasisOracleExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOracleValidationExperimentConfig {
    preset: Option<String>,
    input: DirectBasisOracleValidationInputConfig,
    output: DirectBasisOracleValidationOutputConfig,
    selection: DirectBasisAdapterBankSelectionExperimentConfig,
    rollout: DirectBasisRolloutExperimentConfig,
    target: DirectBasisTargetExperimentConfig,
    optimizer: DirectBasisOptimizerExperimentConfig,
    oracle: DirectBasisOracleExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOracleValidationInputConfig {
    shared_base: Option<PathBuf>,
    adapter_bank: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOracleValidationOutputConfig {
    report_output: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisSourceExperimentConfig {
    target_images: Option<Vec<PathBuf>>,
    target_image_dirs: Option<Vec<PathBuf>>,
    target_image_recursive: Option<bool>,
    image_extensions: Option<Vec<String>>,
    catalog: Option<PathBuf>,
    catalog_thumbnail_dir: Option<PathBuf>,
    catalog_group: Option<String>,
    catalog_targets: Option<Vec<String>>,
    catalog_limit: Option<usize>,
    source_limit: Option<usize>,
    omnisvg: DirectBasisOmniSvgExperimentConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOmniSvgExperimentConfig {
    dataset: Option<String>,
    split: Option<String>,
    cache_dir: Option<PathBuf>,
    offset: Option<usize>,
    limit: Option<usize>,
    page_size: Option<usize>,
    download: Option<bool>,
    refresh: Option<bool>,
    token_env: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisSplitExperimentConfig {
    holdout_targets: Option<Vec<String>>,
    holdout_stride: Option<usize>,
    holdout_offset: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOutputExperimentConfig {
    output_dir: Option<PathBuf>,
    report_output: Option<PathBuf>,
    shared_base_output: Option<PathBuf>,
    adapter_bank_output: Option<PathBuf>,
    adapter_output_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisTrainingExperimentConfig {
    device: Option<String>,
    steps: Option<usize>,
    report_interval: Option<usize>,
    example_batch_size: Option<usize>,
    tbptt_chunk_steps: Option<usize>,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    eval_interval: Option<usize>,
    eval_batch_size: Option<usize>,
    max_dense_train_particles: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisGpuExperimentConfig {
    backend: Option<String>,
    max_dense_chunk_floats: Option<usize>,
    max_splat_chunk_floats: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisAdapterExperimentConfig {
    rank: Option<usize>,
    alpha: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisRolloutExperimentConfig {
    particles: Option<usize>,
    steps: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    base_seed: Option<u64>,
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisTargetExperimentConfig {
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

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOptimizerExperimentConfig {
    per_parameter_grad_normalization: Option<bool>,
    base_learning_rate: Option<f32>,
    base_weight_decay: Option<f32>,
    base_grad_clip_norm: Option<f32>,
    adapter_learning_rate: Option<f32>,
    adapter_weight_decay: Option<f32>,
    adapter_grad_clip_norm: Option<f32>,
    adapter_l2: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisAdapterPhaseExperimentConfig {
    steps: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisEvalExperimentConfig {
    examples: Option<usize>,
    seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DirectBasisOracleExperimentConfig {
    backend: Option<String>,
    gpu_device: Option<String>,
    resume_existing: Option<bool>,
    gpu_parallel_jobs: Option<usize>,
    train_examples: Option<usize>,
    holdout_examples: Option<usize>,
    epochs: Option<usize>,
    repetitions: Option<usize>,
    report_interval: Option<usize>,
    batch_size: Option<usize>,
    pool_size: Option<usize>,
    rollout_step_min: Option<usize>,
    tbptt_chunk_steps: Option<usize>,
    loss_on_final_chunk_only: Option<bool>,
    use_particle_pool: Option<bool>,
    inject_seed_interval: Option<usize>,
    brush_size: Option<f32>,
    learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
    seed: Option<u64>,
}

fn load_direct_basis_experiment_config(
    path: Option<&Path>,
) -> Result<DirectBasisExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(DirectBasisExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse direct-basis experiment config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn load_direct_basis_oracle_validation_experiment_config(
    path: Option<&Path>,
) -> Result<DirectBasisOracleValidationExperimentConfig, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(DirectBasisOracleValidationExperimentConfig::default());
    };
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse direct-basis oracle validation config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn config_value_enum<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    match value {
        Some(value) => T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in direct-basis TOML config: {err}"
            ))
            .into()
        }),
        None => Ok(fallback),
    }
}

fn config_value_enum_option<T: ValueEnum>(
    field: &str,
    value: Option<String>,
    fallback: Option<T>,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    match value {
        Some(value) => Ok(Some(T::from_str(&value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in direct-basis TOML config: {err}"
            ))
        })?)),
        None => Ok(fallback),
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_train_hyper_2d_direct_basis(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dDirectBasis {
        config,
        preset,
        target_images,
        target_image_dirs,
        target_image_recursive,
        image_extensions,
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        omnisvg_dataset,
        omnisvg_split,
        omnisvg_cache_dir,
        omnisvg_offset,
        omnisvg_limit,
        omnisvg_page_size,
        omnisvg_download,
        omnisvg_refresh,
        omnisvg_token_env,
        source_limit,
        holdout_targets,
        holdout_stride,
        holdout_offset,
        output_dir,
        report_output,
        shared_base_output,
        adapter_bank_output,
        adapter_output_dir,
        training_device,
        gpu_backend,
        adapter_rank,
        adapter_alpha,
        steps,
        report_interval,
        example_batch_size,
        tbptt_chunk_steps,
        rollout_particles,
        rollout_steps,
        update_prob,
        seed,
        base_seed,
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
        per_parameter_grad_normalization,
        base_learning_rate,
        base_weight_decay,
        base_grad_clip_norm,
        adapter_learning_rate,
        adapter_weight_decay,
        adapter_grad_clip_norm,
        adapter_l2,
        holdout_adapter_steps,
        holdout_adapter_batch_size,
        train_adapter_refine_steps,
        train_adapter_refine_batch_size,
        train_adapter_refine_learning_rate,
        train_adapter_refine_weight_decay,
        train_adapter_refine_grad_clip_norm,
        holdout_adapter_learning_rate,
        holdout_adapter_weight_decay,
        holdout_adapter_grad_clip_norm,
        eval_examples,
        eval_interval,
        eval_batch_size,
        eval_seed,
        system_memory_budget_gb,
        gpu_memory_budget_gb,
        max_dense_train_particles,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
        oracle_train_examples,
        oracle_holdout_examples,
        oracle_epochs,
        oracle_repetitions,
        oracle_report_interval,
        oracle_batch_size,
        oracle_pool_size,
        oracle_learning_rate,
        oracle_weight_decay,
        oracle_grad_clip_norm,
        oracle_seed,
    } = command
    else {
        unreachable!("run_train_hyper_2d_direct_basis called with the wrong command variant");
    };

    let experiment_config_path = config;
    let experiment_config_report = experiment_config_path
        .as_ref()
        .map(|path| path.display().to_string());
    let experiment_config = load_direct_basis_experiment_config(experiment_config_path.as_deref())?;
    let DirectBasisExperimentConfig {
        preset: config_preset,
        source: config_source,
        split: config_split,
        output: config_output,
        training: config_training,
        gpu: config_gpu,
        adapter: config_adapter,
        rollout: config_rollout,
        target: config_target,
        optimizer: config_optimizer,
        train_refine: config_train_refine,
        holdout_adapter: config_holdout_adapter,
        eval: config_eval,
        oracle: config_oracle,
    } = experiment_config;
    let DirectBasisSourceExperimentConfig {
        target_images: config_target_images,
        target_image_dirs: config_target_image_dirs,
        target_image_recursive: config_target_image_recursive,
        image_extensions: config_image_extensions,
        catalog: config_catalog,
        catalog_thumbnail_dir: config_catalog_thumbnail_dir,
        catalog_group: config_catalog_group,
        catalog_targets: config_catalog_targets,
        catalog_limit: config_catalog_limit,
        source_limit: config_source_limit,
        omnisvg: config_omnisvg,
    } = config_source;
    let DirectBasisOmniSvgExperimentConfig {
        dataset: config_omnisvg_dataset,
        split: config_omnisvg_split,
        cache_dir: config_omnisvg_cache_dir,
        offset: config_omnisvg_offset,
        limit: config_omnisvg_limit,
        page_size: config_omnisvg_page_size,
        download: config_omnisvg_download,
        refresh: config_omnisvg_refresh,
        token_env: config_omnisvg_token_env,
    } = config_omnisvg;
    let DirectBasisSplitExperimentConfig {
        holdout_targets: config_holdout_targets,
        holdout_stride: config_holdout_stride,
        holdout_offset: config_holdout_offset,
    } = config_split;
    let DirectBasisOutputExperimentConfig {
        output_dir: config_output_dir,
        report_output: config_report_output,
        shared_base_output: config_shared_base_output,
        adapter_bank_output: config_adapter_bank_output,
        adapter_output_dir: config_adapter_output_dir,
    } = config_output;
    let DirectBasisTrainingExperimentConfig {
        device: config_training_device,
        steps: config_steps,
        report_interval: config_report_interval,
        example_batch_size: config_example_batch_size,
        tbptt_chunk_steps: config_tbptt_chunk_steps,
        system_memory_budget_gb: config_system_memory_budget_gb,
        gpu_memory_budget_gb: config_gpu_memory_budget_gb,
        eval_interval: config_eval_interval,
        eval_batch_size: config_eval_batch_size,
        max_dense_train_particles: config_max_dense_train_particles,
    } = config_training;
    let DirectBasisGpuExperimentConfig {
        backend: config_gpu_backend,
        max_dense_chunk_floats: config_max_dense_chunk_floats,
        max_splat_chunk_floats: config_max_splat_chunk_floats,
    } = config_gpu;
    let DirectBasisAdapterExperimentConfig {
        rank: config_adapter_rank,
        alpha: config_adapter_alpha,
    } = config_adapter;
    let DirectBasisRolloutExperimentConfig {
        particles: config_rollout_particles,
        steps: config_rollout_steps,
        update_prob: config_update_prob,
        seed: config_seed,
        base_seed: config_base_seed,
        seed_scale: config_seed_scale,
        seed_mode: config_seed_mode,
    } = config_rollout;
    let DirectBasisTargetExperimentConfig {
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
    let DirectBasisOptimizerExperimentConfig {
        per_parameter_grad_normalization: config_per_parameter_grad_normalization,
        base_learning_rate: config_base_learning_rate,
        base_weight_decay: config_base_weight_decay,
        base_grad_clip_norm: config_base_grad_clip_norm,
        adapter_learning_rate: config_adapter_learning_rate,
        adapter_weight_decay: config_adapter_weight_decay,
        adapter_grad_clip_norm: config_adapter_grad_clip_norm,
        adapter_l2: config_adapter_l2,
    } = config_optimizer;
    let DirectBasisAdapterPhaseExperimentConfig {
        steps: config_train_adapter_refine_steps,
        batch_size: config_train_adapter_refine_batch_size,
        learning_rate: config_train_adapter_refine_learning_rate,
        weight_decay: config_train_adapter_refine_weight_decay,
        grad_clip_norm: config_train_adapter_refine_grad_clip_norm,
    } = config_train_refine;
    let DirectBasisAdapterPhaseExperimentConfig {
        steps: config_holdout_adapter_steps,
        batch_size: config_holdout_adapter_batch_size,
        learning_rate: config_holdout_adapter_learning_rate,
        weight_decay: config_holdout_adapter_weight_decay,
        grad_clip_norm: config_holdout_adapter_grad_clip_norm,
    } = config_holdout_adapter;
    let DirectBasisEvalExperimentConfig {
        examples: config_eval_examples,
        seed: config_eval_seed,
    } = config_eval;
    let DirectBasisOracleExperimentConfig {
        backend: config_oracle_backend,
        gpu_device: config_oracle_gpu_device,
        resume_existing: config_oracle_resume_existing,
        gpu_parallel_jobs: config_oracle_gpu_parallel_jobs,
        train_examples: config_oracle_train_examples,
        holdout_examples: config_oracle_holdout_examples,
        epochs: config_oracle_epochs,
        repetitions: config_oracle_repetitions,
        report_interval: config_oracle_report_interval,
        batch_size: config_oracle_batch_size,
        pool_size: config_oracle_pool_size,
        rollout_step_min: config_oracle_rollout_step_min,
        tbptt_chunk_steps: config_oracle_tbptt_chunk_steps,
        loss_on_final_chunk_only: config_oracle_loss_on_final_chunk_only,
        use_particle_pool: config_oracle_use_particle_pool,
        inject_seed_interval: config_oracle_inject_seed_interval,
        brush_size: config_oracle_brush_size,
        learning_rate: config_oracle_learning_rate,
        weight_decay: config_oracle_weight_decay,
        grad_clip_norm: config_oracle_grad_clip_norm,
        seed: config_oracle_seed,
    } = config_oracle;

    let preset = config_value_enum("preset", config_preset, preset)?;
    let target_images = config_target_images.unwrap_or(target_images);
    let target_image_dirs = config_target_image_dirs.unwrap_or(target_image_dirs);
    let target_image_recursive = config_target_image_recursive.unwrap_or(target_image_recursive);
    let image_extensions = config_image_extensions.unwrap_or(image_extensions);
    let catalog = config_catalog.or(catalog);
    let catalog_thumbnail_dir = config_catalog_thumbnail_dir.unwrap_or(catalog_thumbnail_dir);
    let catalog_group =
        config_value_enum_option("source.catalog_group", config_catalog_group, catalog_group)?;
    let catalog_targets = config_catalog_targets.unwrap_or(catalog_targets);
    let catalog_limit = config_catalog_limit.unwrap_or(catalog_limit);
    let omnisvg_dataset = config_value_enum_option(
        "source.omnisvg.dataset",
        config_omnisvg_dataset,
        omnisvg_dataset,
    )?;
    let omnisvg_split = config_omnisvg_split.unwrap_or(omnisvg_split);
    let omnisvg_cache_dir = config_omnisvg_cache_dir.unwrap_or(omnisvg_cache_dir);
    let omnisvg_offset = config_omnisvg_offset.unwrap_or(omnisvg_offset);
    let omnisvg_limit = config_omnisvg_limit.unwrap_or(omnisvg_limit);
    let omnisvg_page_size = config_omnisvg_page_size.unwrap_or(omnisvg_page_size);
    let omnisvg_download = config_omnisvg_download.unwrap_or(omnisvg_download);
    let omnisvg_refresh = config_omnisvg_refresh.unwrap_or(omnisvg_refresh);
    let omnisvg_token_env = config_omnisvg_token_env.unwrap_or(omnisvg_token_env);
    let source_limit = config_source_limit.unwrap_or(source_limit);
    let holdout_targets = config_holdout_targets.unwrap_or(holdout_targets);
    let holdout_stride = config_holdout_stride.unwrap_or(holdout_stride);
    let holdout_offset = config_holdout_offset.unwrap_or(holdout_offset);
    let output_dir = config_output_dir.unwrap_or(output_dir);
    let report_output = config_report_output.or(report_output);
    let shared_base_output = config_shared_base_output.or(shared_base_output);
    let adapter_bank_output = config_adapter_bank_output.or(adapter_bank_output);
    let adapter_output_dir = config_adapter_output_dir.or(adapter_output_dir);
    let training_device =
        config_value_enum("training.device", config_training_device, training_device)?;
    let gpu_backend = config_value_enum("gpu.backend", config_gpu_backend, gpu_backend)?;
    let adapter_rank = config_adapter_rank.unwrap_or(adapter_rank);
    let adapter_alpha = config_adapter_alpha.unwrap_or(adapter_alpha);
    let steps = config_steps.unwrap_or(steps);
    let report_interval = config_report_interval.unwrap_or(report_interval);
    let example_batch_size = config_example_batch_size.unwrap_or(example_batch_size);
    let tbptt_chunk_steps = config_tbptt_chunk_steps.unwrap_or(tbptt_chunk_steps);
    let system_memory_budget_gb = config_system_memory_budget_gb
        .or(system_memory_budget_gb)
        .or(Some(24.0));
    let gpu_memory_budget_gb = config_gpu_memory_budget_gb
        .or(gpu_memory_budget_gb)
        .or(Some(DEFAULT_WGPU_VRAM_BUDGET_GB));
    let max_dense_train_particles =
        config_max_dense_train_particles.unwrap_or(max_dense_train_particles);
    let max_dense_chunk_floats = config_max_dense_chunk_floats.unwrap_or(max_dense_chunk_floats);
    let max_splat_chunk_floats = config_max_splat_chunk_floats.unwrap_or(max_splat_chunk_floats);
    let rollout_particles = config_rollout_particles.unwrap_or(rollout_particles);
    let rollout_steps = config_rollout_steps.unwrap_or(rollout_steps);
    let update_prob = config_update_prob.unwrap_or(update_prob);
    let seed = config_seed.unwrap_or(seed);
    let base_seed = config_base_seed.unwrap_or(base_seed);
    let seed_scale = config_seed_scale.or(seed_scale);
    let seed_mode = config_value_enum("rollout.seed_mode", config_seed_mode, seed_mode)?;
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
    let per_parameter_grad_normalization =
        config_per_parameter_grad_normalization.unwrap_or(per_parameter_grad_normalization);
    let base_learning_rate = config_base_learning_rate.unwrap_or(base_learning_rate);
    let base_weight_decay = config_base_weight_decay.unwrap_or(base_weight_decay);
    let base_grad_clip_norm = config_base_grad_clip_norm.unwrap_or(base_grad_clip_norm);
    let adapter_learning_rate = config_adapter_learning_rate.unwrap_or(adapter_learning_rate);
    let adapter_weight_decay = config_adapter_weight_decay.unwrap_or(adapter_weight_decay);
    let adapter_grad_clip_norm = config_adapter_grad_clip_norm.unwrap_or(adapter_grad_clip_norm);
    let adapter_l2 = config_adapter_l2.unwrap_or(adapter_l2);
    let holdout_adapter_steps = config_holdout_adapter_steps.unwrap_or(holdout_adapter_steps);
    let holdout_adapter_batch_size =
        config_holdout_adapter_batch_size.unwrap_or(holdout_adapter_batch_size);
    let train_adapter_refine_steps =
        config_train_adapter_refine_steps.unwrap_or(train_adapter_refine_steps);
    let train_adapter_refine_batch_size =
        config_train_adapter_refine_batch_size.unwrap_or(train_adapter_refine_batch_size);
    let train_adapter_refine_learning_rate =
        config_train_adapter_refine_learning_rate.or(train_adapter_refine_learning_rate);
    let train_adapter_refine_weight_decay =
        config_train_adapter_refine_weight_decay.or(train_adapter_refine_weight_decay);
    let train_adapter_refine_grad_clip_norm =
        config_train_adapter_refine_grad_clip_norm.or(train_adapter_refine_grad_clip_norm);
    let holdout_adapter_learning_rate =
        config_holdout_adapter_learning_rate.or(holdout_adapter_learning_rate);
    let holdout_adapter_weight_decay =
        config_holdout_adapter_weight_decay.or(holdout_adapter_weight_decay);
    let holdout_adapter_grad_clip_norm =
        config_holdout_adapter_grad_clip_norm.or(holdout_adapter_grad_clip_norm);
    let eval_examples = config_eval_examples.unwrap_or(eval_examples);
    let eval_interval = config_eval_interval
        .or(eval_interval)
        .unwrap_or(report_interval);
    let eval_batch_size = config_eval_batch_size.unwrap_or(eval_batch_size);
    let eval_seed = config_eval_seed.unwrap_or(eval_seed);
    let oracle_train_examples = config_oracle_train_examples.unwrap_or(oracle_train_examples);
    let oracle_holdout_examples = config_oracle_holdout_examples.unwrap_or(oracle_holdout_examples);
    let oracle_epochs = config_oracle_epochs.unwrap_or(oracle_epochs);
    let oracle_repetitions = config_oracle_repetitions.unwrap_or(oracle_repetitions);
    let oracle_report_interval = config_oracle_report_interval.unwrap_or(oracle_report_interval);
    let oracle_batch_size = config_oracle_batch_size.unwrap_or(oracle_batch_size);
    let oracle_pool_size = config_oracle_pool_size.unwrap_or(oracle_pool_size);
    let oracle_rollout_step_min =
        config_oracle_rollout_step_min.unwrap_or(rollout_steps.clamp(1, 32));
    let oracle_tbptt_chunk_steps = config_oracle_tbptt_chunk_steps.unwrap_or(rollout_steps.max(1));
    let oracle_loss_on_final_chunk_only = config_oracle_loss_on_final_chunk_only.unwrap_or(true);
    let oracle_use_particle_pool = config_oracle_use_particle_pool.unwrap_or(true);
    let oracle_inject_seed_interval = config_oracle_inject_seed_interval.unwrap_or(16);
    let oracle_brush_size = config_oracle_brush_size.unwrap_or(0.1);
    let oracle_learning_rate = config_oracle_learning_rate.unwrap_or(oracle_learning_rate);
    let oracle_weight_decay = config_oracle_weight_decay.unwrap_or(oracle_weight_decay);
    let oracle_grad_clip_norm = config_oracle_grad_clip_norm.unwrap_or(oracle_grad_clip_norm);
    let oracle_seed = config_oracle_seed.unwrap_or(oracle_seed);
    let oracle_backend = config_value_enum(
        "oracle.backend",
        config_oracle_backend,
        DirectBasisOracleBackendArg::Cpu,
    )?;
    let oracle_gpu_device = config_oracle_gpu_device.unwrap_or_else(|| "cuda:0".to_string());
    let oracle_resume_existing = config_oracle_resume_existing.unwrap_or(false);
    let oracle_gpu_parallel_jobs = config_oracle_gpu_parallel_jobs.unwrap_or(1);

    let preset_arg = preset;
    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-hyper2d-direct-basis currently supports the growing-2d target image objective",
        )
        .into());
    }
    validate_direct_basis_args(DirectBasisArgCheck {
        adapter_rank,
        adapter_alpha,
        rollout_particles,
        rollout_steps,
        update_prob,
        tbptt_chunk_steps,
        eval_batch_size,
        system_memory_budget_gb,
        gpu_memory_budget_gb,
        max_dense_train_particles,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
        base_learning_rate,
        adapter_learning_rate,
        train_adapter_refine_learning_rate,
        holdout_adapter_learning_rate,
        adapter_l2,
    })?;

    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let oracle_config = DirectBasisOracleConfig {
        backend: oracle_backend,
        gpu_device: oracle_gpu_device,
        resume_existing: oracle_resume_existing,
        gpu_parallel_jobs: oracle_gpu_parallel_jobs,
        train_examples: oracle_train_examples,
        holdout_examples: oracle_holdout_examples,
        epochs: oracle_epochs,
        repetitions: oracle_repetitions,
        report_interval: oracle_report_interval,
        batch_size: oracle_batch_size,
        pool_size: oracle_pool_size,
        rollout_step_min: oracle_rollout_step_min,
        tbptt_chunk_steps: oracle_tbptt_chunk_steps,
        loss_on_final_chunk_only: oracle_loss_on_final_chunk_only,
        use_particle_pool: oracle_use_particle_pool,
        inject_seed_interval: oracle_inject_seed_interval,
        brush_size: oracle_brush_size,
        learning_rate: oracle_learning_rate,
        weight_decay: oracle_weight_decay,
        grad_clip_norm: oracle_grad_clip_norm,
        seed: oracle_seed,
    };
    validate_oracle_config(&oracle_config)?;
    let report_output = report_output.unwrap_or_else(|| output_dir.join("report.json"));
    let shared_base_output =
        shared_base_output.unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    let adapter_bank_output =
        adapter_bank_output.unwrap_or_else(|| output_dir.join("adapter_bank.json"));
    let adapter_output_dir = adapter_output_dir.unwrap_or_else(|| output_dir.join("adapters"));
    let omnisvg_source = omnisvg_dataset.map(|dataset| OmniSvgSourceConfig {
        dataset,
        split: &omnisvg_split,
        cache_dir: &omnisvg_cache_dir,
        offset: omnisvg_offset,
        limit: omnisvg_limit,
        page_size: omnisvg_page_size,
        download: omnisvg_download,
        refresh: omnisvg_refresh,
        token_env: &omnisvg_token_env,
    });

    let mut sources = resolve_scratch_sources(ScratchSourceResolveConfig {
        preset: preset_arg,
        target_images: &target_images,
        target_image_dirs: &target_image_dirs,
        target_image_recursive,
        image_extensions: &image_extensions,
        catalog: catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit,
        omnisvg: omnisvg_source,
    })?;
    if source_limit > 0 && sources.len() > source_limit {
        sources.truncate(source_limit);
    }
    let splits = resolve_e2e_splits(&sources, &holdout_targets, holdout_stride, holdout_offset)?;

    let hashgrid = upstream_growing_2d_hashgrid();
    let base_config = NpaConfig::growing_2d();
    let motion_scale = base_config.alpha * base_config.motion_eps(hashgrid.eps);
    let loss_config = super::super::target2d::target2d_loss_config(
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
    let base_sgd = SgdConfig {
        learning_rate: base_learning_rate,
        weight_decay: base_weight_decay,
        grad_clip_norm: base_grad_clip_norm,
    };
    let adapter_sgd = SgdConfig {
        learning_rate: adapter_learning_rate,
        weight_decay: adapter_weight_decay,
        grad_clip_norm: adapter_grad_clip_norm,
    };
    let holdout_adapter_sgd = SgdConfig {
        learning_rate: holdout_adapter_learning_rate.unwrap_or(adapter_sgd.learning_rate),
        weight_decay: holdout_adapter_weight_decay.unwrap_or(adapter_sgd.weight_decay),
        grad_clip_norm: holdout_adapter_grad_clip_norm.unwrap_or(adapter_sgd.grad_clip_norm),
    };
    let train_refine_adapter_sgd = SgdConfig {
        learning_rate: train_adapter_refine_learning_rate
            .unwrap_or(holdout_adapter_sgd.learning_rate),
        weight_decay: train_adapter_refine_weight_decay.unwrap_or(holdout_adapter_sgd.weight_decay),
        grad_clip_norm: train_adapter_refine_grad_clip_norm
            .unwrap_or(holdout_adapter_sgd.grad_clip_norm),
    };
    if training_device != TrainingDeviceArg::Cpu {
        let request = GpuDirectBasisRunRequest {
            experiment_config: experiment_config_report,
            preset,
            requested_training_device: training_device,
            target_images: &target_images,
            target_image_dirs: &target_image_dirs,
            target_image_recursive,
            image_extensions,
            catalog: catalog.as_ref(),
            catalog_group,
            catalog_targets,
            omnisvg: super::omnisvg_source_report(omnisvg_source),
            source_limit,
            holdout_targets,
            holdout_stride,
            holdout_offset,
            output_dir: &output_dir,
            report_output: &report_output,
            shared_base_output: &shared_base_output,
            adapter_bank_output: &adapter_bank_output,
            adapter_output_dir: &adapter_output_dir,
            sources: &sources,
            splits: &splits,
            hashgrid,
            loss_config,
            adapter_rank,
            adapter_alpha,
            steps,
            report_interval,
            example_batch_size,
            tbptt_chunk_steps,
            rollout_particles,
            rollout_steps,
            update_prob,
            seed,
            base_seed,
            seed_scale,
            seed_mode,
            target_points,
            target_image_size,
            target_threshold,
            per_parameter_grad_normalization,
            base_sgd,
            adapter_sgd,
            train_refine_adapter_sgd,
            holdout_adapter_sgd,
            adapter_l2,
            holdout_adapter_steps,
            holdout_adapter_batch_size,
            train_adapter_refine_steps,
            train_adapter_refine_batch_size,
            eval_examples,
            eval_interval,
            eval_batch_size,
            eval_seed,
            system_memory_budget_gb,
            gpu_memory_budget_gb,
            max_dense_train_particles,
            max_dense_chunk_floats,
            max_splat_chunk_floats,
            oracle_config,
        };
        return match gpu_backend {
            Hyper2dDirectBasisGpuBackendArg::BurnWgpu => run_burn_wgpu_direct_basis(request),
        };
    }
    let mut base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), base_seed);
    base.validate()?;
    let examples = load_direct_basis_examples(
        &sources,
        &splits,
        &base.config,
        adapter_rank,
        adapter_alpha,
        seed,
        DirectBasisTargetConfig {
            threshold: target_threshold,
            points: target_points,
            image_size: target_image_size,
        },
    )?;
    let (mut train_examples, mut holdout_examples): (Vec<_>, Vec<_>) = examples
        .into_iter()
        .partition(|example| example.split == Hyper2dE2eSplit::Train);
    if train_examples.is_empty() {
        return Err(
            std::io::Error::other("train-hyper2d-direct-basis requires train examples").into(),
        );
    }

    let train_config = DirectBasisTrainConfig {
        steps,
        report_interval,
        example_batch_size,
        tbptt_chunk_steps,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: base_config.stopgrad_pos,
        stopgrad_state: base_config.stopgrad_state,
        rollout_particles,
        rollout_step_min: rollout_steps,
        rollout_steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale,
        loss_config,
        target2d_loss_backend: Target2dLossBackend::Auto,
        perception_backend: PerceptionRolloutBackend::Auto,
        per_parameter_grad_normalization,
        base_sgd,
        adapter_sgd,
        adapter_l2_weight: adapter_l2,
        update_base: true,
        eval_examples,
        eval_interval,
        eval_batch_size,
        eval_seed,
        system_memory_budget_gb,
        gpu_memory_budget_gb,
        max_dense_train_particles,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
    };
    let initial_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed,
    )?;
    let initial_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed ^ 0x90_1d_2d,
    )?;
    let train_phase =
        train_direct_basis_phase(&mut base, &mut train_examples, &hashgrid, train_config)?;
    let train_refine_batch_size = if train_adapter_refine_batch_size == 0 {
        example_batch_size
    } else {
        train_adapter_refine_batch_size
    };
    let train_refine_config = DirectBasisTrainConfig {
        steps: train_adapter_refine_steps,
        example_batch_size: train_refine_batch_size,
        adapter_sgd: train_refine_adapter_sgd,
        update_base: false,
        seed: seed ^ 0x7a_1d_2d,
        eval_seed: eval_seed ^ 0x7a_1d_2d,
        ..train_config
    };
    let train_refine_phase = train_direct_basis_phase(
        &mut base,
        &mut train_examples,
        &hashgrid,
        train_refine_config,
    )?;
    let holdout_config = DirectBasisTrainConfig {
        steps: holdout_adapter_steps,
        example_batch_size: holdout_adapter_batch_size,
        adapter_sgd: holdout_adapter_sgd,
        update_base: false,
        seed: seed ^ 0x90_1d_2d,
        eval_seed: eval_seed ^ 0x90_1d_2d,
        ..train_config
    };
    let holdout_phase =
        train_direct_basis_phase(&mut base, &mut holdout_examples, &hashgrid, holdout_config)?;
    let (best_train_loss, best_train_step) = match train_refine_phase.best_loss {
        Some(loss) => (
            Some(loss),
            train_config.steps + train_refine_phase.best_step,
        ),
        None => (train_phase.best_loss, train_phase.best_step),
    };
    let final_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed,
    )?;
    let final_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &hashgrid,
        train_config,
        eval_examples,
        eval_seed ^ 0x90_1d_2d,
    )?;

    let base_manifest = BpkModelManifest::from_model(
        &base,
        hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-direct-basis:sources={}:steps={steps}",
            train_examples.len()
        )),
    );
    crate::import::save_manifest(&shared_base_output, &base_manifest)?;
    let adapter_reports = save_direct_basis_adapters(
        &base_manifest,
        &shared_base_output,
        &adapter_output_dir,
        train_examples
            .iter()
            .chain(holdout_examples.iter())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let adapter_bank = DirectBasisAdapterBankManifest {
        base_model: shared_base_output.display().to_string(),
        adapter_rank,
        adapter_alpha,
        entries: adapter_reports.clone(),
    };
    write_pretty_json(&adapter_bank_output, &adapter_bank)?;
    let oracle_model_dir = output_dir.join("oracle_models");
    let oracle_validation = evaluate_direct_basis_oracles(
        &base,
        &train_examples,
        &holdout_examples,
        &hashgrid,
        train_config,
        oracle_config,
        Some(&oracle_model_dir),
    )?;

    let report = CliHyper2dDirectBasisTrainingReport {
        experiment_config: experiment_config_report,
        preset,
        target_images: target_images
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_dirs: target_image_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_recursive,
        image_extensions,
        catalog: catalog.as_ref().map(|path| path.display().to_string()),
        catalog_group,
        catalog_targets,
        omnisvg: super::omnisvg_source_report(omnisvg_source),
        source_limit,
        holdout_targets,
        holdout_stride,
        holdout_offset,
        output_dir: output_dir.display().to_string(),
        report_output: report_output.display().to_string(),
        shared_base_output: shared_base_output.display().to_string(),
        adapter_bank_output: adapter_bank_output.display().to_string(),
        adapter_output_dir: adapter_output_dir.display().to_string(),
        requested_training_device: training_device,
        training_device: TrainingDeviceArg::Cpu,
        gpu_training: None,
        npa_config: base.config.clone(),
        hashgrid,
        target_loss_config: loss_config,
        adapter_rank,
        adapter_alpha,
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        steps,
        report_interval,
        example_batch_size: normalized_example_batch_size(example_batch_size, train_examples.len()),
        tbptt_chunk_steps,
        rollout_particles,
        rollout_steps,
        update_prob,
        seed,
        seed_scale,
        seed_mode,
        per_parameter_grad_normalization,
        base_sgd,
        adapter_sgd,
        train_refine_adapter_sgd,
        holdout_adapter_sgd,
        adapter_l2_weight: adapter_l2,
        train_adapter_refine_steps,
        train_adapter_refine_batch_size: normalized_example_batch_size(
            train_refine_batch_size,
            train_examples.len(),
        ),
        holdout_adapter_steps,
        holdout_adapter_batch_size: normalized_example_batch_size(
            holdout_adapter_batch_size,
            holdout_examples.len().max(1),
        ),
        eval_examples,
        eval_interval,
        eval_batch_size,
        system_memory_budget_gb,
        gpu_memory_budget_gb,
        max_dense_train_particles,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
        initial_train_loss,
        final_train_loss,
        initial_holdout_loss,
        final_holdout_loss,
        best_train_loss,
        best_train_step,
        history: train_phase.history,
        train_refine_history: train_refine_phase.history,
        holdout_history: holdout_phase.history,
        oracle_validation,
        adapters: adapter_reports,
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} shared_base={} adapter_bank={}",
        report_output.display(),
        report.train_examples,
        report.holdout_examples,
        shared_base_output.display(),
        adapter_bank_output.display()
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_validate_hyper_2d_direct_basis_oracles(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::ValidateHyper2dDirectBasisOracles {
        config,
        preset,
        shared_base,
        adapter_bank,
        report_output,
        rollout_particles,
        rollout_steps,
        update_prob,
        eval_seed,
        seed_scale,
        seed_mode,
        per_parameter_grad_normalization,
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
        oracle_train_examples,
        oracle_holdout_examples,
        oracle_epochs,
        oracle_repetitions,
        oracle_report_interval,
        oracle_batch_size,
        oracle_pool_size,
        oracle_learning_rate,
        oracle_weight_decay,
        oracle_grad_clip_norm,
        oracle_seed,
    } = command
    else {
        unreachable!(
            "run_validate_hyper_2d_direct_basis_oracles called with the wrong command variant"
        );
    };

    let experiment_config =
        load_direct_basis_oracle_validation_experiment_config(config.as_deref())?;
    let DirectBasisOracleValidationExperimentConfig {
        preset: config_preset,
        input: config_input,
        output: config_output,
        selection: config_selection,
        rollout: config_rollout,
        target: config_target,
        optimizer: config_optimizer,
        oracle: config_oracle,
    } = experiment_config;
    let DirectBasisOracleValidationInputConfig {
        shared_base: config_shared_base,
        adapter_bank: config_adapter_bank,
    } = config_input;
    let DirectBasisOracleValidationOutputConfig {
        report_output: config_report_output,
    } = config_output;
    let DirectBasisAdapterBankSelectionExperimentConfig {
        selection_seed: config_selection_seed,
        selection_manifest: config_selection_manifest,
    } = config_selection;
    let DirectBasisRolloutExperimentConfig {
        particles: config_rollout_particles,
        steps: config_rollout_steps,
        update_prob: config_update_prob,
        seed: config_eval_seed,
        base_seed: _,
        seed_scale: config_seed_scale,
        seed_mode: config_seed_mode,
    } = config_rollout;
    let DirectBasisTargetExperimentConfig {
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
    let DirectBasisOptimizerExperimentConfig {
        per_parameter_grad_normalization: config_per_parameter_grad_normalization,
        base_learning_rate: _,
        base_weight_decay: _,
        base_grad_clip_norm: _,
        adapter_learning_rate: _,
        adapter_weight_decay: _,
        adapter_grad_clip_norm: _,
        adapter_l2: _,
    } = config_optimizer;
    let DirectBasisOracleExperimentConfig {
        backend: config_oracle_backend,
        gpu_device: config_oracle_gpu_device,
        resume_existing: config_oracle_resume_existing,
        gpu_parallel_jobs: config_oracle_gpu_parallel_jobs,
        train_examples: config_oracle_train_examples,
        holdout_examples: config_oracle_holdout_examples,
        epochs: config_oracle_epochs,
        repetitions: config_oracle_repetitions,
        report_interval: config_oracle_report_interval,
        batch_size: config_oracle_batch_size,
        pool_size: config_oracle_pool_size,
        rollout_step_min: config_oracle_rollout_step_min,
        tbptt_chunk_steps: config_oracle_tbptt_chunk_steps,
        loss_on_final_chunk_only: config_oracle_loss_on_final_chunk_only,
        use_particle_pool: config_oracle_use_particle_pool,
        inject_seed_interval: config_oracle_inject_seed_interval,
        brush_size: config_oracle_brush_size,
        learning_rate: config_oracle_learning_rate,
        weight_decay: config_oracle_weight_decay,
        grad_clip_norm: config_oracle_grad_clip_norm,
        seed: config_oracle_seed,
    } = config_oracle;

    let preset = config_value_enum("preset", config_preset, preset)?;
    let shared_base = config_shared_base.or(shared_base).ok_or_else(|| {
        std::io::Error::other(
            "validate-hyper2d-direct-basis-oracles requires --shared-base or input.shared_base",
        )
    })?;
    let adapter_bank = config_adapter_bank.or(adapter_bank).ok_or_else(|| {
        std::io::Error::other(
            "validate-hyper2d-direct-basis-oracles requires --adapter-bank or input.adapter_bank",
        )
    })?;
    let report_output = config_report_output.unwrap_or(report_output);
    let rollout_particles = config_rollout_particles.unwrap_or(rollout_particles);
    let rollout_steps = config_rollout_steps.unwrap_or(rollout_steps);
    let update_prob = config_update_prob.unwrap_or(update_prob);
    let eval_seed = config_eval_seed.unwrap_or(eval_seed);
    let seed_scale = config_seed_scale.or(seed_scale);
    let seed_mode = config_value_enum("rollout.seed_mode", config_seed_mode, seed_mode)?;
    let per_parameter_grad_normalization =
        config_per_parameter_grad_normalization.unwrap_or(per_parameter_grad_normalization);
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
    let oracle_train_examples = config_oracle_train_examples.unwrap_or(oracle_train_examples);
    let oracle_holdout_examples = config_oracle_holdout_examples.unwrap_or(oracle_holdout_examples);
    let oracle_epochs = config_oracle_epochs.unwrap_or(oracle_epochs);
    let oracle_repetitions = config_oracle_repetitions.unwrap_or(oracle_repetitions);
    let oracle_report_interval = config_oracle_report_interval.unwrap_or(oracle_report_interval);
    let oracle_batch_size = config_oracle_batch_size.unwrap_or(oracle_batch_size);
    let oracle_pool_size = config_oracle_pool_size.unwrap_or(oracle_pool_size);
    let oracle_rollout_step_min =
        config_oracle_rollout_step_min.unwrap_or(rollout_steps.clamp(1, 32));
    let oracle_tbptt_chunk_steps = config_oracle_tbptt_chunk_steps.unwrap_or(rollout_steps.max(1));
    let oracle_loss_on_final_chunk_only = config_oracle_loss_on_final_chunk_only.unwrap_or(true);
    let oracle_use_particle_pool = config_oracle_use_particle_pool.unwrap_or(true);
    let oracle_inject_seed_interval = config_oracle_inject_seed_interval.unwrap_or(16);
    let oracle_brush_size = config_oracle_brush_size.unwrap_or(0.1);
    let oracle_learning_rate = config_oracle_learning_rate.unwrap_or(oracle_learning_rate);
    let oracle_weight_decay = config_oracle_weight_decay.unwrap_or(oracle_weight_decay);
    let oracle_grad_clip_norm = config_oracle_grad_clip_norm.unwrap_or(oracle_grad_clip_norm);
    let oracle_seed = config_oracle_seed.unwrap_or(oracle_seed);
    let oracle_backend = config_value_enum(
        "oracle.backend",
        config_oracle_backend,
        DirectBasisOracleBackendArg::Cpu,
    )?;
    let oracle_gpu_device = config_oracle_gpu_device.unwrap_or_else(|| "cuda:0".to_string());
    let oracle_resume_existing = config_oracle_resume_existing.unwrap_or(false);
    let oracle_gpu_parallel_jobs = config_oracle_gpu_parallel_jobs.unwrap_or(1);
    let selection_seed = config_selection_seed.unwrap_or(oracle_seed);

    let preset: AutomataPreset = preset.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "validate-hyper2d-direct-basis-oracles currently supports growing-2d",
        )
        .into());
    }
    let seed_mode: ParticleSeed = seed_mode.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let loss_config = super::super::target2d::target2d_loss_config(
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
    let oracle_config = DirectBasisOracleConfig {
        backend: oracle_backend,
        gpu_device: oracle_gpu_device,
        resume_existing: oracle_resume_existing,
        gpu_parallel_jobs: oracle_gpu_parallel_jobs,
        train_examples: oracle_train_examples,
        holdout_examples: oracle_holdout_examples,
        epochs: oracle_epochs,
        repetitions: oracle_repetitions,
        report_interval: oracle_report_interval,
        batch_size: oracle_batch_size,
        pool_size: oracle_pool_size,
        rollout_step_min: oracle_rollout_step_min,
        tbptt_chunk_steps: oracle_tbptt_chunk_steps,
        loss_on_final_chunk_only: oracle_loss_on_final_chunk_only,
        use_particle_pool: oracle_use_particle_pool,
        inject_seed_interval: oracle_inject_seed_interval,
        brush_size: oracle_brush_size,
        learning_rate: oracle_learning_rate,
        weight_decay: oracle_weight_decay,
        grad_clip_norm: oracle_grad_clip_norm,
        seed: oracle_seed,
    };
    validate_direct_basis_args(DirectBasisArgCheck {
        adapter_rank: 1,
        adapter_alpha: 1.0,
        rollout_particles,
        rollout_steps,
        update_prob,
        tbptt_chunk_steps: 1,
        eval_batch_size: 1,
        system_memory_budget_gb: Some(24.0),
        gpu_memory_budget_gb: Some(DEFAULT_WGPU_VRAM_BUDGET_GB),
        max_dense_train_particles: 2048,
        max_dense_chunk_floats: 4 * 1024 * 1024,
        max_splat_chunk_floats: 4 * 1024 * 1024,
        base_learning_rate: 0.0,
        adapter_learning_rate: 0.0,
        train_adapter_refine_learning_rate: None,
        holdout_adapter_learning_rate: None,
        adapter_l2: 0.0,
    })?;
    validate_oracle_config(&oracle_config)?;

    let base_manifest = crate::import::load_manifest(&shared_base)?;
    let hashgrid = base_manifest.hashgrid.clone();
    let base = base_manifest.clone().into_model();
    base.validate()?;
    let bank = load_direct_basis_adapter_bank(&adapter_bank)?;
    let bank_base_model = bank.base_model.clone();
    let bank_adapter_rank = bank.adapter_rank;
    let bank_adapter_alpha = bank.adapter_alpha;
    let (train_entries, holdout_entries) = split_direct_basis_adapter_bank_entries(bank.entries)?;
    let total_train_examples = train_entries.len();
    let total_holdout_examples = holdout_entries.len();
    let (selected_train_entries, selected_holdout_entries, selection_report) =
        select_direct_basis_adapter_bank_entries_with_manifest(
            &train_entries,
            &holdout_entries,
            oracle_train_examples,
            oracle_holdout_examples,
            selection_seed,
            config_selection_manifest.as_deref(),
        )?;
    let train_examples = load_direct_basis_examples_from_adapter_bank_entries(
        &adapter_bank,
        &base,
        DirectBasisTargetConfig {
            threshold: target_threshold,
            points: target_points,
            image_size: target_image_size,
        },
        selected_train_entries,
    )?;
    let holdout_examples = load_direct_basis_examples_from_adapter_bank_entries(
        &adapter_bank,
        &base,
        DirectBasisTargetConfig {
            threshold: target_threshold,
            points: target_points,
            image_size: target_image_size,
        },
        selected_holdout_entries,
    )?;
    if total_train_examples == 0 {
        return Err(std::io::Error::other(
            "validate-hyper2d-direct-basis-oracles requires at least one train example",
        )
        .into());
    }
    let eval_config = DirectBasisTrainConfig {
        steps: 0,
        report_interval: 1,
        example_batch_size: 1,
        tbptt_chunk_steps: 1,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: base.config.stopgrad_pos,
        stopgrad_state: base.config.stopgrad_state,
        rollout_particles,
        rollout_step_min: rollout_steps,
        rollout_steps,
        update_prob,
        seed: eval_seed,
        seed_scale,
        seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale: base.config.alpha * base.config.motion_eps(hashgrid.eps),
        loss_config,
        target2d_loss_backend: Target2dLossBackend::Auto,
        perception_backend: PerceptionRolloutBackend::Auto,
        per_parameter_grad_normalization,
        base_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: false,
        eval_examples: 0,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed,
        system_memory_budget_gb: Some(24.0),
        gpu_memory_budget_gb: Some(DEFAULT_WGPU_VRAM_BUDGET_GB),
        max_dense_train_particles: 2048,
        max_dense_chunk_floats: 4 * 1024 * 1024,
        max_splat_chunk_floats: 4 * 1024 * 1024,
    };
    let oracle_model_dir = report_output.with_file_name("oracle_models");
    let oracle_validation = evaluate_direct_basis_oracles(
        &base,
        &train_examples,
        &holdout_examples,
        &hashgrid,
        eval_config,
        oracle_config,
        Some(&oracle_model_dir),
    )?;
    let report = DirectBasisOracleValidationReport {
        preset,
        shared_base: shared_base.display().to_string(),
        adapter_bank: adapter_bank.display().to_string(),
        report_output: report_output.display().to_string(),
        adapter_bank_base_model: bank_base_model,
        adapter_rank: bank_adapter_rank,
        adapter_alpha: bank_adapter_alpha,
        npa_config: base.config.clone(),
        hashgrid,
        target_loss_config: loss_config,
        target_threshold,
        target_points_fallback: target_points,
        target_image_size,
        train_examples: total_train_examples,
        holdout_examples: total_holdout_examples,
        rollout_particles,
        rollout_steps,
        update_prob,
        eval_seed,
        seed_scale,
        seed_mode,
        per_parameter_grad_normalization,
        selection: selection_report,
        oracle_validation,
    };
    write_pretty_json(&report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} oracle_train={} oracle_holdout={}",
        report_output.display(),
        report.train_examples,
        report.holdout_examples,
        report
            .oracle_validation
            .as_ref()
            .map_or(0, |oracle| oracle.train_examples),
        report
            .oracle_validation
            .as_ref()
            .map_or(0, |oracle| oracle.holdout_examples)
    );
    Ok(())
}

struct GpuDirectBasisRunRequest<'a> {
    experiment_config: Option<String>,
    preset: AutomataPreset,
    requested_training_device: TrainingDeviceArg,
    target_images: &'a [PathBuf],
    target_image_dirs: &'a [PathBuf],
    target_image_recursive: bool,
    image_extensions: Vec<String>,
    catalog: Option<&'a PathBuf>,
    catalog_group: Option<Hyper2dCatalogGroupArg>,
    catalog_targets: Vec<String>,
    omnisvg: Option<CliOmniSvgSourceReport>,
    source_limit: usize,
    holdout_targets: Vec<String>,
    holdout_stride: usize,
    holdout_offset: usize,
    output_dir: &'a Path,
    report_output: &'a Path,
    shared_base_output: &'a Path,
    adapter_bank_output: &'a Path,
    adapter_output_dir: &'a Path,
    sources: &'a [super::sources::Hyper2dScratchSource],
    splits: &'a [Hyper2dE2eSplit],
    hashgrid: burn_automata_kernels::HashGridConfig,
    loss_config: Target2dLossConfig,
    adapter_rank: usize,
    adapter_alpha: f32,
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    tbptt_chunk_steps: usize,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    base_seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    target_points: usize,
    target_image_size: Option<usize>,
    target_threshold: f32,
    per_parameter_grad_normalization: bool,
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
    train_refine_adapter_sgd: SgdConfig,
    holdout_adapter_sgd: SgdConfig,
    adapter_l2: f32,
    holdout_adapter_steps: usize,
    holdout_adapter_batch_size: usize,
    train_adapter_refine_steps: usize,
    train_adapter_refine_batch_size: usize,
    eval_examples: usize,
    eval_interval: usize,
    eval_batch_size: usize,
    eval_seed: u64,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    max_dense_train_particles: usize,
    max_dense_chunk_floats: usize,
    max_splat_chunk_floats: usize,
    oracle_config: DirectBasisOracleConfig,
}

fn validate_direct_basis_burn_wgpu_preflight(
    train_examples: &[DirectBasisExample],
    holdout_examples: &[DirectBasisExample],
    base_config: &NpaConfig,
    train_config: DirectBasisTrainConfig,
    train_refine_config: DirectBasisTrainConfig,
    holdout_config: DirectBasisTrainConfig,
) -> Result<DirectBasisWgpuMemoryPreflightReport, Box<dyn std::error::Error>> {
    let training_requested = train_config.steps > 0
        || train_refine_config.steps > 0
        || (!holdout_examples.is_empty() && holdout_config.steps > 0);
    let max_training_particles = [
        (train_config.steps > 0)
            .then(|| direct_basis_max_particles(train_examples, train_config.rollout_particles)),
        (train_refine_config.steps > 0).then(|| {
            direct_basis_max_particles(train_examples, train_refine_config.rollout_particles)
        }),
        (holdout_config.steps > 0).then(|| {
            direct_basis_max_particles(holdout_examples, holdout_config.rollout_particles)
        }),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(train_config.rollout_particles);
    let max_phase_batch_size = [
        (train_config.steps > 0).then(|| {
            normalized_example_batch_size(train_config.example_batch_size, train_examples.len())
        }),
        (train_refine_config.steps > 0).then(|| {
            normalized_example_batch_size(
                train_refine_config.example_batch_size,
                train_examples.len(),
            )
        }),
        (holdout_config.steps > 0).then(|| {
            normalized_example_batch_size(
                holdout_config.example_batch_size,
                holdout_examples.len().max(1),
            )
        }),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(0);
    let target_pixels = train_config
        .loss_config
        .image_size
        .saturating_mul(train_config.loss_config.image_size);
    let estimated_graph_bytes =
        estimate_direct_basis_wgpu_graph_bytes(DirectBasisWgpuGraphEstimateConfig {
            batch_size: max_phase_batch_size,
            particles: max_training_particles,
            state_dims: base_config.state_dims,
            target_pixels,
            rollout_steps: train_config.rollout_steps,
            tbptt_chunk_steps: train_config.tbptt_chunk_steps,
            max_dense_chunk_floats: train_config.max_dense_chunk_floats,
            max_splat_chunk_floats: train_config.max_splat_chunk_floats,
        });
    let estimated_target_cache_bytes = estimate_direct_basis_target_cache_bytes(
        train_examples.len().saturating_add(holdout_examples.len()),
        target_pixels,
    );
    let estimated_peak_bytes = estimated_graph_bytes.saturating_add(estimated_target_cache_bytes);
    let memory_budget_bytes = train_config
        .system_memory_budget_gb
        .map(memory_budget_gb_to_bytes);
    let estimated_vram_bytes =
        estimate_direct_basis_wgpu_vram_bytes(estimated_graph_bytes, estimated_target_cache_bytes);
    let gpu_memory_budget_bytes = train_config
        .gpu_memory_budget_gb
        .map(memory_budget_gb_to_bytes);
    let dense_train_particle_cap_passed =
        !training_requested || max_training_particles <= train_config.max_dense_train_particles;
    let memory_budget_passed = !training_requested
        || memory_budget_bytes
            .map(|budget| estimated_peak_bytes <= budget)
            .unwrap_or(true);
    let gpu_memory_budget_passed = !training_requested
        || gpu_memory_budget_bytes
            .map(|budget| estimated_vram_bytes <= budget)
            .unwrap_or(true);
    let report = DirectBasisWgpuMemoryPreflightReport {
        training_requested,
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        max_training_particles,
        max_dense_train_particles: train_config.max_dense_train_particles,
        max_phase_batch_size,
        rollout_steps: train_config.rollout_steps,
        tbptt_chunk_steps: train_config
            .tbptt_chunk_steps
            .min(train_config.rollout_steps)
            .max(1),
        target_pixels,
        estimated_graph_bytes,
        estimated_target_cache_bytes,
        estimated_peak_bytes,
        memory_budget_bytes,
        estimated_vram_bytes,
        gpu_memory_budget_bytes,
        vram_estimate_multiplier: WGPU_VRAM_ESTIMATE_MULTIPLIER,
        dense_train_particle_cap_passed,
        memory_budget_passed,
        gpu_memory_budget_passed,
    };
    if !report.dense_train_particle_cap_passed {
        return Err(std::io::Error::other(format!(
            "Burn direct-basis dense training is capped at {} particles for this config; requested {}. \
Increase max_dense_train_particles only with tight TBPTT/chunk caps, or use staged lower-particle training.",
            report.max_dense_train_particles, report.max_training_particles
        ))
        .into());
    }
    if !report.memory_budget_passed {
        return Err(std::io::Error::other(format!(
            "Burn/WGPU direct-basis preflight estimated {:.2} GiB peak system memory, above the configured {:.2} GiB budget. \
Reduce example_batch_size, rollout_particles, tbptt_chunk_steps, or target loss image size.",
            bytes_to_gib(report.estimated_peak_bytes),
            bytes_to_gib(report.memory_budget_bytes.unwrap_or(0))
        ))
        .into());
    }
    if !report.gpu_memory_budget_passed {
        return Err(std::io::Error::other(format!(
            "Burn/WGPU direct-basis preflight estimated {:.2} GiB peak VRAM, above the configured {:.2} GiB GPU budget. \
Reduce example_batch_size, rollout_particles, tbptt_chunk_steps, or target loss image size. \
This guard is intentionally conservative because dense Burn/WGPU autodiff can retain backend tensors across TBPTT chunks.",
            bytes_to_gib(report.estimated_vram_bytes),
            bytes_to_gib(report.gpu_memory_budget_bytes.unwrap_or(0))
        ))
        .into());
    }
    Ok(report)
}

fn direct_basis_max_particles(examples: &[DirectBasisExample], fallback: usize) -> usize {
    examples
        .iter()
        .map(|example| example.source.particles.unwrap_or(fallback))
        .max()
        .unwrap_or(fallback)
}

struct DirectBasisWgpuGraphEstimateConfig {
    batch_size: usize,
    particles: usize,
    state_dims: usize,
    target_pixels: usize,
    rollout_steps: usize,
    tbptt_chunk_steps: usize,
    max_dense_chunk_floats: usize,
    max_splat_chunk_floats: usize,
}

fn estimate_direct_basis_wgpu_graph_bytes(config: DirectBasisWgpuGraphEstimateConfig) -> u64 {
    let DirectBasisWgpuGraphEstimateConfig {
        batch_size,
        particles,
        state_dims,
        target_pixels,
        rollout_steps,
        tbptt_chunk_steps,
        max_dense_chunk_floats,
        max_splat_chunk_floats,
    } = config;
    let batch = batch_size.max(1) as u128;
    let particles = particles.max(1) as u128;
    let state_dims = state_dims.max(1) as u128;
    let tbptt = tbptt_chunk_steps.max(1).min(rollout_steps.max(1)) as u128;
    let dense_query_rows = estimate_dense_query_chunk_rows(
        batch_size,
        particles as usize,
        state_dims as usize,
        max_dense_chunk_floats,
    ) as u128;
    let splat_pixel_rows = estimate_splat_pixel_chunk_rows(
        batch_size,
        particles as usize,
        target_pixels,
        max_splat_chunk_floats,
    ) as u128;
    let bytes_per_float = std::mem::size_of::<f32>() as u128;
    let dense_tile_bytes = batch
        .saturating_mul(dense_query_rows)
        .saturating_mul(particles)
        .saturating_mul(state_dims)
        .saturating_mul(bytes_per_float)
        .saturating_mul(24);
    let dense_output_bytes = batch
        .saturating_mul(particles)
        .saturating_mul(state_dims.saturating_mul(4).saturating_add(8))
        .saturating_mul(bytes_per_float)
        .saturating_mul(4);
    let dense_graph_bytes = dense_tile_bytes
        .saturating_add(dense_output_bytes)
        .saturating_mul(tbptt);
    let splat_graph_bytes = batch
        .saturating_mul(splat_pixel_rows)
        .saturating_mul(particles)
        .saturating_mul(bytes_per_float)
        .saturating_mul(16);
    dense_graph_bytes
        .saturating_add(splat_graph_bytes)
        .min(u64::MAX as u128) as u64
}

fn estimate_dense_query_chunk_rows(
    batches: usize,
    rows: usize,
    state_dims: usize,
    max_floats: usize,
) -> usize {
    let denominator = batches
        .max(1)
        .saturating_mul(rows.max(1))
        .saturating_mul(state_dims.max(1))
        .saturating_mul(2)
        .max(1);
    (max_floats / denominator).max(1).min(rows.max(1))
}

fn estimate_splat_pixel_chunk_rows(
    batches: usize,
    particles: usize,
    pixels: usize,
    max_floats: usize,
) -> usize {
    let denominator = batches
        .max(1)
        .saturating_mul(particles.max(1))
        .saturating_mul(2)
        .max(1);
    (max_floats / denominator).max(1).min(pixels.max(1))
}

fn estimate_direct_basis_target_cache_bytes(target_count: usize, target_pixels: usize) -> u64 {
    let pixels = target_pixels.max(1) as u128;
    (target_count.max(1) as u128)
        .saturating_mul(pixels)
        .saturating_mul(5)
        .saturating_mul(std::mem::size_of::<f32>() as u128)
        .min(u64::MAX as u128) as u64
}

fn estimate_direct_basis_wgpu_vram_bytes(graph_bytes: u64, target_cache_bytes: u64) -> u64 {
    graph_bytes
        .saturating_mul(WGPU_VRAM_ESTIMATE_MULTIPLIER)
        .saturating_add(target_cache_bytes.saturating_mul(2))
}

fn memory_budget_gb_to_bytes(gb: f32) -> u64 {
    (gb as f64 * 1024.0 * 1024.0 * 1024.0).round() as u64
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn run_burn_wgpu_direct_basis(
    request: GpuDirectBasisRunRequest<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), request.base_seed);
    base.validate()?;
    let examples = load_direct_basis_examples(
        request.sources,
        request.splits,
        &base.config,
        request.adapter_rank,
        request.adapter_alpha,
        request.seed,
        DirectBasisTargetConfig {
            threshold: request.target_threshold,
            points: request.target_points,
            image_size: request.target_image_size,
        },
    )?;
    let (mut train_examples, mut holdout_examples): (Vec<_>, Vec<_>) = examples
        .into_iter()
        .partition(|example| example.split == Hyper2dE2eSplit::Train);
    if train_examples.is_empty() {
        return Err(
            std::io::Error::other("train-hyper2d-direct-basis requires train examples").into(),
        );
    }

    let train_config = DirectBasisTrainConfig {
        steps: request.steps,
        report_interval: request.report_interval,
        example_batch_size: request.example_batch_size,
        tbptt_chunk_steps: request.tbptt_chunk_steps,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: base.config.stopgrad_pos,
        stopgrad_state: base.config.stopgrad_state,
        rollout_particles: request.rollout_particles,
        rollout_step_min: request.rollout_steps,
        rollout_steps: request.rollout_steps,
        update_prob: request.update_prob,
        seed: request.seed,
        seed_scale: request.seed_scale,
        seed_mode: request.seed_mode,
        grid_eps: request.hashgrid.eps,
        motion_scale: base.config.alpha * base.config.motion_eps(request.hashgrid.eps),
        loss_config: request.loss_config,
        target2d_loss_backend: Target2dLossBackend::Auto,
        perception_backend: PerceptionRolloutBackend::Auto,
        per_parameter_grad_normalization: request.per_parameter_grad_normalization,
        base_sgd: request.base_sgd,
        adapter_sgd: request.adapter_sgd,
        adapter_l2_weight: request.adapter_l2,
        update_base: true,
        eval_examples: request.eval_examples,
        eval_interval: request.eval_interval,
        eval_batch_size: request.eval_batch_size,
        eval_seed: request.eval_seed,
        system_memory_budget_gb: request.system_memory_budget_gb,
        gpu_memory_budget_gb: request.gpu_memory_budget_gb,
        max_dense_train_particles: request.max_dense_train_particles,
        max_dense_chunk_floats: request.max_dense_chunk_floats,
        max_splat_chunk_floats: request.max_splat_chunk_floats,
    };
    let train_refine_batch_size = if request.train_adapter_refine_batch_size == 0 {
        request.example_batch_size
    } else {
        request.train_adapter_refine_batch_size
    };
    let train_refine_config = DirectBasisTrainConfig {
        steps: request.train_adapter_refine_steps,
        example_batch_size: train_refine_batch_size,
        adapter_sgd: request.train_refine_adapter_sgd,
        update_base: false,
        seed: request.seed ^ 0x7a_1d_2d,
        eval_seed: request.eval_seed ^ 0x7a_1d_2d,
        ..train_config
    };
    let holdout_config = DirectBasisTrainConfig {
        steps: request.holdout_adapter_steps,
        example_batch_size: request.holdout_adapter_batch_size,
        adapter_sgd: request.holdout_adapter_sgd,
        update_base: false,
        seed: request.seed ^ 0x90_1d_2d,
        eval_seed: request.eval_seed ^ 0x90_1d_2d,
        ..train_config
    };
    let memory_preflight = validate_direct_basis_burn_wgpu_preflight(
        &train_examples,
        &holdout_examples,
        &base.config,
        train_config,
        train_refine_config,
        holdout_config,
    )?;
    println!(
        "burn-wgpu direct-basis preflight particles={} batch={} tbptt={} estimated_system_peak_gib={:.2} system_budget_gib={} estimated_vram_gib={:.2} gpu_budget_gib={}",
        memory_preflight.max_training_particles,
        memory_preflight.max_phase_batch_size,
        memory_preflight.tbptt_chunk_steps,
        bytes_to_gib(memory_preflight.estimated_peak_bytes),
        memory_preflight
            .memory_budget_bytes
            .map(|bytes| format!("{:.2}", bytes_to_gib(bytes)))
            .unwrap_or_else(|| "unbounded".to_string()),
        bytes_to_gib(memory_preflight.estimated_vram_bytes),
        memory_preflight
            .gpu_memory_budget_bytes
            .map(|bytes| format!("{:.2}", bytes_to_gib(bytes)))
            .unwrap_or_else(|| "unbounded".to_string())
    );
    let initial_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed,
    )?;
    let initial_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed ^ 0x90_1d_2d,
    )?;
    let mut train_training_examples = direct_basis_training_examples(&train_examples);
    let mut holdout_training_examples = direct_basis_training_examples(&holdout_examples);
    let mut burn_report = dense::train_direct_basis_burn_wgpu(
        &mut base,
        &mut train_training_examples,
        &mut holdout_training_examples,
        train_config,
        train_refine_config,
        holdout_config,
        None,
    )?;
    sync_direct_basis_training_examples(&mut train_examples, &train_training_examples);
    sync_direct_basis_training_examples(&mut holdout_examples, &holdout_training_examples);
    if let Some(metrics) = burn_report.metrics.as_object_mut() {
        metrics.insert(
            "memory_preflight".to_string(),
            serde_json::to_value(&memory_preflight)?,
        );
    }
    let final_train_loss = evaluate_direct_basis_examples(
        &base,
        &train_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed,
    )?;
    let final_holdout_loss = evaluate_direct_basis_examples(
        &base,
        &holdout_examples,
        &request.hashgrid,
        train_config,
        request.eval_examples,
        request.eval_seed ^ 0x90_1d_2d,
    )?;

    let base_manifest = BpkModelManifest::from_model(
        &base,
        request.hashgrid.clone(),
        Some(format!(
            "trained-rust:hyper2d-direct-basis:burn-wgpu:sources={}:steps={}",
            train_examples.len(),
            request.steps
        )),
    );
    crate::import::save_manifest(request.shared_base_output, &base_manifest)?;
    let adapter_reports = save_direct_basis_adapters(
        &base_manifest,
        request.shared_base_output,
        request.adapter_output_dir,
        train_examples
            .iter()
            .chain(holdout_examples.iter())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let adapter_bank = DirectBasisAdapterBankManifest {
        base_model: request.shared_base_output.display().to_string(),
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        entries: adapter_reports.clone(),
    };
    write_pretty_json(request.adapter_bank_output, &adapter_bank)?;
    let oracle_model_dir = request.output_dir.join("oracle_models");
    let oracle_validation = evaluate_direct_basis_oracles(
        &base,
        &train_examples,
        &holdout_examples,
        &request.hashgrid,
        train_config,
        request.oracle_config,
        Some(&oracle_model_dir),
    )?;
    let report = CliHyper2dDirectBasisTrainingReport {
        experiment_config: request.experiment_config.clone(),
        preset: request.preset,
        target_images: request
            .target_images
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_dirs: request
            .target_image_dirs
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        target_image_recursive: request.target_image_recursive,
        image_extensions: request.image_extensions,
        catalog: request.catalog.map(|path| path.display().to_string()),
        catalog_group: request.catalog_group,
        catalog_targets: request.catalog_targets,
        omnisvg: request.omnisvg,
        source_limit: request.source_limit,
        holdout_targets: request.holdout_targets,
        holdout_stride: request.holdout_stride,
        holdout_offset: request.holdout_offset,
        output_dir: request.output_dir.display().to_string(),
        report_output: request.report_output.display().to_string(),
        shared_base_output: request.shared_base_output.display().to_string(),
        adapter_bank_output: request.adapter_bank_output.display().to_string(),
        adapter_output_dir: request.adapter_output_dir.display().to_string(),
        requested_training_device: request.requested_training_device,
        training_device: TrainingDeviceArg::Gpu,
        gpu_training: Some(CliHyper2dDirectBasisGpuTrainingReport {
            backend: burn_report.backend.to_string(),
            device: burn_report.device,
            metrics: burn_report.metrics,
        }),
        npa_config: base.config.clone(),
        hashgrid: request.hashgrid,
        target_loss_config: request.loss_config,
        adapter_rank: request.adapter_rank,
        adapter_alpha: request.adapter_alpha,
        train_examples: train_examples.len(),
        holdout_examples: holdout_examples.len(),
        steps: request.steps,
        report_interval: request.report_interval,
        example_batch_size: normalized_example_batch_size(
            request.example_batch_size,
            train_examples.len(),
        ),
        tbptt_chunk_steps: request.tbptt_chunk_steps,
        rollout_particles: request.rollout_particles,
        rollout_steps: request.rollout_steps,
        update_prob: request.update_prob,
        seed: request.seed,
        seed_scale: request.seed_scale,
        seed_mode: request.seed_mode,
        per_parameter_grad_normalization: request.per_parameter_grad_normalization,
        base_sgd: request.base_sgd,
        adapter_sgd: request.adapter_sgd,
        train_refine_adapter_sgd: request.train_refine_adapter_sgd,
        holdout_adapter_sgd: request.holdout_adapter_sgd,
        adapter_l2_weight: request.adapter_l2,
        train_adapter_refine_steps: request.train_adapter_refine_steps,
        train_adapter_refine_batch_size: normalized_example_batch_size(
            train_refine_batch_size,
            train_examples.len(),
        ),
        holdout_adapter_steps: request.holdout_adapter_steps,
        holdout_adapter_batch_size: normalized_example_batch_size(
            request.holdout_adapter_batch_size,
            holdout_examples.len().max(1),
        ),
        eval_examples: request.eval_examples,
        eval_interval: request.eval_interval,
        eval_batch_size: request.eval_batch_size,
        system_memory_budget_gb: request.system_memory_budget_gb,
        gpu_memory_budget_gb: request.gpu_memory_budget_gb,
        max_dense_train_particles: request.max_dense_train_particles,
        max_dense_chunk_floats: request.max_dense_chunk_floats,
        max_splat_chunk_floats: request.max_splat_chunk_floats,
        initial_train_loss,
        final_train_loss,
        initial_holdout_loss,
        final_holdout_loss,
        best_train_loss: burn_report.best_train_loss,
        best_train_step: burn_report.best_train_step,
        history: burn_report.history,
        train_refine_history: burn_report.train_refine_history,
        holdout_history: burn_report.holdout_history,
        oracle_validation,
        adapters: adapter_reports,
    };
    write_pretty_json(request.report_output, &report)?;
    println!(
        "wrote {} train={} holdout={} shared_base={} adapter_bank={} backend=burn-wgpu",
        request.report_output.display(),
        report.train_examples,
        report.holdout_examples,
        request.shared_base_output.display(),
        request.adapter_bank_output.display()
    );
    Ok(())
}

struct DirectBasisArgCheck {
    adapter_rank: usize,
    adapter_alpha: f32,
    rollout_particles: usize,
    rollout_steps: usize,
    update_prob: f32,
    tbptt_chunk_steps: usize,
    eval_batch_size: usize,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    max_dense_train_particles: usize,
    max_dense_chunk_floats: usize,
    max_splat_chunk_floats: usize,
    base_learning_rate: f32,
    adapter_learning_rate: f32,
    train_adapter_refine_learning_rate: Option<f32>,
    holdout_adapter_learning_rate: Option<f32>,
    adapter_l2: f32,
}

fn validate_direct_basis_args(
    config: DirectBasisArgCheck,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.adapter_rank == 0 || !config.adapter_alpha.is_finite() || config.adapter_alpha <= 0.0
    {
        return Err(std::io::Error::other(
            "adapter rank must be non-zero and adapter alpha must be finite and positive",
        )
        .into());
    }
    if config.rollout_particles == 0 || config.rollout_steps == 0 {
        return Err(std::io::Error::other(
            "rollout particles and rollout steps must be greater than zero",
        )
        .into());
    }
    if config.tbptt_chunk_steps == 0 {
        return Err(std::io::Error::other("TBPTT chunk steps must be greater than zero").into());
    }
    if config.eval_batch_size == 0 {
        return Err(std::io::Error::other("eval batch size must be greater than zero").into());
    }
    if let Some(budget_gb) = config.system_memory_budget_gb
        && (!budget_gb.is_finite() || budget_gb <= 0.0)
    {
        return Err(std::io::Error::other(
            "system memory budget must be finite and greater than zero",
        )
        .into());
    }
    if let Some(budget_gb) = config.gpu_memory_budget_gb
        && (!budget_gb.is_finite() || budget_gb <= 0.0)
    {
        return Err(std::io::Error::other(
            "GPU memory budget must be finite and greater than zero",
        )
        .into());
    }
    if config.max_dense_train_particles == 0
        || config.max_dense_chunk_floats == 0
        || config.max_splat_chunk_floats == 0
    {
        return Err(std::io::Error::other(
            "dense training particle and chunk-float caps must be greater than zero",
        )
        .into());
    }
    if !(0.0..=1.0).contains(&config.update_prob) || !config.update_prob.is_finite() {
        return Err(
            std::io::Error::other("update probability must be finite and in [0, 1]").into(),
        );
    }
    if !config.base_learning_rate.is_finite()
        || config.base_learning_rate < 0.0
        || !config.adapter_learning_rate.is_finite()
        || config.adapter_learning_rate < 0.0
        || config
            .train_adapter_refine_learning_rate
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || config
            .holdout_adapter_learning_rate
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(std::io::Error::other("learning rates must be finite and non-negative").into());
    }
    if !config.adapter_l2.is_finite() || config.adapter_l2 < 0.0 {
        return Err(
            std::io::Error::other("adapter L2 weight must be finite and non-negative").into(),
        );
    }
    Ok(())
}

fn validate_oracle_config(
    config: &DirectBasisOracleConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.train_examples == 0 && config.holdout_examples == 0 {
        return Ok(());
    }
    if config.epochs == 0
        || config.repetitions == 0
        || config.batch_size == 0
        || config.pool_size == 0
        || config.rollout_step_min == 0
        || config.tbptt_chunk_steps == 0
        || config.inject_seed_interval == 0
    {
        return Err(std::io::Error::other(
            "oracle epochs, repetitions, batch size, pool size, rollout step minimum, TBPTT chunk steps, and seed-injection interval must be greater than zero",
        )
        .into());
    }
    if config.use_particle_pool && config.pool_size < config.batch_size {
        return Err(std::io::Error::other(
            "oracle pool_size must be at least batch_size when particle-pool training is enabled",
        )
        .into());
    }
    if !config.learning_rate.is_finite()
        || config.learning_rate < 0.0
        || !config.weight_decay.is_finite()
        || config.weight_decay < 0.0
        || !config.grad_clip_norm.is_finite()
        || config.grad_clip_norm < 0.0
        || !config.brush_size.is_finite()
        || config.brush_size < 0.0
    {
        return Err(std::io::Error::other(
            "oracle optimizer settings must be finite and non-negative",
        )
        .into());
    }
    if matches!(config.backend, DirectBasisOracleBackendArg::Cuda)
        && config.gpu_device.trim().is_empty()
    {
        return Err(std::io::Error::other("oracle.gpu_device must not be empty").into());
    }
    if config.gpu_parallel_jobs == 0 {
        return Err(
            std::io::Error::other("oracle.gpu_parallel_jobs must be greater than zero").into(),
        );
    }
    if config.gpu_parallel_jobs > 64 {
        return Err(std::io::Error::other(
            "oracle.gpu_parallel_jobs must be <= 64 to avoid accidental oversubscription",
        )
        .into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DirectBasisTargetConfig {
    threshold: f32,
    points: usize,
    image_size: Option<usize>,
}

fn load_direct_basis_examples(
    sources: &[super::sources::Hyper2dScratchSource],
    splits: &[Hyper2dE2eSplit],
    npa_config: &NpaConfig,
    adapter_rank: usize,
    adapter_alpha: f32,
    seed: u64,
    target_config: DirectBasisTargetConfig,
) -> Result<Vec<DirectBasisExample>, Box<dyn std::error::Error>> {
    if sources.len() != splits.len() {
        return Err(std::io::Error::other("source split count does not match sources").into());
    }
    let mut examples = Vec::with_capacity(sources.len());
    for (idx, (source, split)) in sources.iter().zip(splits).enumerate() {
        let target = super::super::target2d::load_target_image_2d_adaptive(
            &source.condition_path,
            target_config.threshold,
            target_config.points,
            target_config.image_size,
        )?;
        let adapter = NpaLowRankAdapter::seeded_zero_delta(
            npa_config,
            adapter_rank,
            adapter_alpha,
            seed.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7)),
        );
        examples.push(DirectBasisExample {
            source: source.clone(),
            split: *split,
            bank_split_index: None,
            target,
            adapter,
            last_train_loss: None,
        });
    }
    Ok(examples)
}

fn split_direct_basis_adapter_bank_entries(
    entries: Vec<DirectBasisAdapterBankLoadEntry>,
) -> Result<DirectBasisAdapterBankSplitEntries, Box<dyn std::error::Error>> {
    let mut train_entries = Vec::new();
    let mut holdout_entries = Vec::new();
    for entry in entries {
        match parse_direct_basis_split(&entry.split)? {
            Hyper2dE2eSplit::Train => train_entries.push((train_entries.len(), entry)),
            Hyper2dE2eSplit::Holdout => holdout_entries.push((holdout_entries.len(), entry)),
        }
    }
    if train_entries.is_empty() && holdout_entries.is_empty() {
        return Err(std::io::Error::other("adapter bank has no train or holdout entries").into());
    }
    Ok((train_entries, holdout_entries))
}

fn select_direct_basis_adapter_bank_oracle_entries(
    entries: &[DirectBasisAdapterBankIndexedEntry],
    requested_examples: usize,
    seed: u64,
) -> Vec<DirectBasisAdapterBankIndexedEntry> {
    if requested_examples == 0 {
        return Vec::new();
    }
    let alpha_capable = entries
        .iter()
        .filter(|(_, entry)| {
            !Path::new(&entry.condition)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
                })
        })
        .collect::<Vec<_>>();
    eval_indices(alpha_capable.len(), requested_examples, seed)
        .into_iter()
        .map(|idx| alpha_capable[idx].clone())
        .collect()
}

fn select_direct_basis_adapter_bank_entries_with_manifest(
    train_entries: &[DirectBasisAdapterBankIndexedEntry],
    holdout_entries: &[DirectBasisAdapterBankIndexedEntry],
    train_requested: usize,
    holdout_requested: usize,
    selection_seed: u64,
    selection_manifest_path: Option<&Path>,
) -> Result<DirectBasisAdapterBankSelectedEntries, Box<dyn std::error::Error>> {
    let (train_selected, holdout_selected, replayed_manifest) =
        if let Some(path) = selection_manifest_path.filter(|path| path.exists()) {
            let manifest = read_direct_basis_adapter_bank_selection_manifest(path)?;
            (
                replay_direct_basis_adapter_bank_selection(train_entries, &manifest.train)?,
                replay_direct_basis_adapter_bank_selection(holdout_entries, &manifest.holdout)?,
                true,
            )
        } else {
            (
                select_direct_basis_adapter_bank_oracle_entries(
                    train_entries,
                    train_requested,
                    selection_seed,
                ),
                select_direct_basis_adapter_bank_oracle_entries(
                    holdout_entries,
                    holdout_requested,
                    selection_seed ^ 0x90_1d_2d,
                ),
                false,
            )
        };
    let manifest = direct_basis_adapter_bank_selection_manifest(
        selection_seed,
        &train_selected,
        &holdout_selected,
    );
    if !replayed_manifest && let Some(path) = selection_manifest_path {
        write_pretty_json(path, &manifest)?;
    }
    let report = DirectBasisAdapterBankSelectionReport {
        selection_seed,
        selection_manifest: selection_manifest_path.map(|path| path.display().to_string()),
        replayed_manifest,
        train_requested,
        holdout_requested,
        train_selected: train_selected.len(),
        holdout_selected: holdout_selected.len(),
        train: manifest.train,
        holdout: manifest.holdout,
    };
    Ok((train_selected, holdout_selected, report))
}

fn read_direct_basis_adapter_bank_selection_manifest(
    path: &Path,
) -> Result<DirectBasisAdapterBankSelectionManifest, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let manifest: DirectBasisAdapterBankSelectionManifest = serde_json::from_str(&text)?;
    Ok(manifest)
}

fn direct_basis_adapter_bank_selection_manifest(
    selection_seed: u64,
    train: &[DirectBasisAdapterBankIndexedEntry],
    holdout: &[DirectBasisAdapterBankIndexedEntry],
) -> DirectBasisAdapterBankSelectionManifest {
    DirectBasisAdapterBankSelectionManifest {
        selection_seed,
        train: direct_basis_adapter_bank_selection_entries(train),
        holdout: direct_basis_adapter_bank_selection_entries(holdout),
    }
}

fn direct_basis_adapter_bank_selection_entries(
    entries: &[DirectBasisAdapterBankIndexedEntry],
) -> Vec<DirectBasisAdapterBankSelectionEntry> {
    entries
        .iter()
        .map(
            |(bank_split_index, entry)| DirectBasisAdapterBankSelectionEntry {
                split: entry.split.clone(),
                slug: entry.slug.clone(),
                bank_split_index: *bank_split_index,
            },
        )
        .collect()
}

fn replay_direct_basis_adapter_bank_selection(
    entries: &[DirectBasisAdapterBankIndexedEntry],
    selection: &[DirectBasisAdapterBankSelectionEntry],
) -> Result<Vec<DirectBasisAdapterBankIndexedEntry>, Box<dyn std::error::Error>> {
    let mut by_key = HashMap::<(String, String), Vec<DirectBasisAdapterBankIndexedEntry>>::new();
    for entry in entries {
        by_key
            .entry((entry.1.split.clone(), entry.1.slug.clone()))
            .or_default()
            .push(entry.clone());
    }
    let mut selected = Vec::with_capacity(selection.len());
    for row in selection {
        let key = (row.split.clone(), row.slug.clone());
        let candidates = by_key.get(&key).ok_or_else(|| {
            std::io::Error::other(format!(
                "selection manifest row {}:{} is not present in the adapter bank",
                row.split, row.slug
            ))
        })?;
        if let Some(exact) = candidates
            .iter()
            .find(|(bank_split_index, _)| *bank_split_index == row.bank_split_index)
        {
            selected.push(exact.clone());
            continue;
        }
        if candidates.len() == 1 {
            selected.push(candidates[0].clone());
            continue;
        }
        return Err(std::io::Error::other(format!(
            "selection manifest row {}:{} is ambiguous without bank_split_index {}",
            row.split, row.slug, row.bank_split_index
        ))
        .into());
    }
    Ok(selected)
}

fn load_direct_basis_examples_from_adapter_bank_entries(
    adapter_bank_path: &Path,
    base: &NpaModel,
    target_config: DirectBasisTargetConfig,
    entries: Vec<DirectBasisAdapterBankIndexedEntry>,
) -> Result<Vec<DirectBasisExample>, Box<dyn std::error::Error>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let anchor = adapter_bank_path.parent().unwrap_or_else(|| Path::new(""));
    let mut examples = Vec::with_capacity(entries.len());
    let mut metadata_mismatches = DirectBasisAdapterBankReloadMismatchSummary::default();
    for (bank_split_index, entry) in entries {
        let split = parse_direct_basis_split(&entry.split)?;
        let condition_path = resolve_direct_basis_artifact_path(anchor, &entry.condition);
        let adapter_path = resolve_direct_basis_artifact_path(anchor, &entry.adapter_output);
        let adapter_manifest = crate::import::load_adapter_manifest(&adapter_path)?;
        adapter_manifest.adapter.validate(&base.config)?;
        let target = super::super::target2d::load_target_image_2d_adaptive(
            &condition_path,
            target_config.threshold,
            target_config.points,
            target_config.image_size,
        )?;
        let source = super::sources::Hyper2dScratchSource {
            slug: entry.slug,
            title: entry.title,
            group: entry.group,
            condition_path,
            particles: None,
            seed_scale: None,
            update_prob: None,
        };
        if entry.target_source_width > 0 && entry.target_source_width != target.source_width {
            metadata_mismatches.record(
                "source_width",
                &source.slug,
                entry.target_source_width,
                target.source_width,
            );
        }
        if entry.target_source_height > 0 && entry.target_source_height != target.source_height {
            metadata_mismatches.record(
                "source_height",
                &source.slug,
                entry.target_source_height,
                target.source_height,
            );
        }
        if entry.target_points > 0 && entry.target_points != target.point_count() {
            metadata_mismatches.record(
                "target_points",
                &source.slug,
                entry.target_points,
                target.point_count(),
            );
        }
        examples.push(DirectBasisExample {
            source,
            split,
            bank_split_index: Some(bank_split_index),
            target,
            adapter: adapter_manifest.adapter,
            last_train_loss: entry.last_train_loss,
        });
    }
    metadata_mismatches.emit();
    Ok(examples)
}

#[derive(Default)]
struct DirectBasisAdapterBankReloadMismatchSummary {
    source_width: usize,
    source_height: usize,
    target_points: usize,
    examples: Vec<String>,
}

impl DirectBasisAdapterBankReloadMismatchSummary {
    fn record(&mut self, field: &'static str, slug: &str, stored: usize, reloaded: usize) {
        match field {
            "source_width" => self.source_width += 1,
            "source_height" => self.source_height += 1,
            "target_points" => self.target_points += 1,
            _ => {}
        }
        if self.examples.len() < 6 {
            self.examples
                .push(format!("{slug}:{field} {stored}->{reloaded}"));
        }
    }

    fn emit(&self) {
        let total = self.source_width + self.source_height + self.target_points;
        if total == 0 {
            return;
        }
        eprintln!(
            "warning: adapter-bank metadata differs from reloaded validation targets \
             (source_width={}, source_height={}, target_points={}, examples=[{}])",
            self.source_width,
            self.source_height,
            self.target_points,
            self.examples.join(", ")
        );
    }
}

fn load_direct_basis_adapter_bank(
    adapter_bank_path: &Path,
) -> Result<DirectBasisAdapterBankLoadManifest, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(adapter_bank_path)?;
    let bank: DirectBasisAdapterBankLoadManifest = serde_json::from_str(&text)?;
    if bank.adapter_rank == 0 || !bank.adapter_alpha.is_finite() || bank.adapter_alpha <= 0.0 {
        return Err(std::io::Error::other(
            "adapter bank rank must be non-zero and alpha must be finite and positive",
        )
        .into());
    }
    Ok(bank)
}

fn resolve_direct_basis_artifact_path(anchor: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() || path.exists() {
        path
    } else {
        anchor.join(path)
    }
}

fn parse_direct_basis_split(label: &str) -> Result<Hyper2dE2eSplit, Box<dyn std::error::Error>> {
    match label {
        "train" => Ok(Hyper2dE2eSplit::Train),
        "holdout" => Ok(Hyper2dE2eSplit::Holdout),
        other => Err(std::io::Error::other(format!(
            "unknown direct-basis adapter split {other:?}"
        ))
        .into()),
    }
}

fn train_direct_basis_phase(
    base: &mut NpaModel,
    examples: &mut [DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectBasisTrainConfig,
) -> Result<DirectBasisPhaseReport, Box<dyn std::error::Error>> {
    if examples.is_empty() || config.steps == 0 {
        return Ok(DirectBasisPhaseReport {
            history: Vec::new(),
            best_loss: None,
            best_step: 0,
        });
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let batch_size = normalized_example_batch_size(config.example_batch_size, examples.len());
    let report_interval = config.report_interval.max(1);
    let mut best_loss = None::<f32>;
    let mut best_step = 0usize;
    let mut history = Vec::new();
    for step in 1..=config.steps {
        let indices = sample_example_indices(examples.len(), batch_size, &mut rng);
        let stats = direct_basis_train_step(base, examples, hashgrid, &indices, config, step)?;
        if step == config.steps || step.is_multiple_of(report_interval) {
            let eval_loss = evaluate_direct_basis_examples(
                base,
                examples,
                hashgrid,
                config,
                config.eval_examples,
                config.eval_seed,
            )?;
            if let Some(summary) = eval_loss
                && best_loss.is_none_or(|loss| summary.mean_total_loss < loss)
            {
                best_loss = Some(summary.mean_total_loss);
                best_step = step;
            }
            history.push(CliHyper2dDirectBasisHistoryEntry {
                step,
                loss: stats.loss,
                eval_loss,
                base_grad_norm: stats.base_grad_norm,
                base_grad_scale: stats.base_grad_scale,
                mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                max_adapter_grad_norm: stats.max_adapter_grad_norm,
                examples_seen: stats.examples_seen,
                particle_steps_per_sec: stats.particle_steps_per_sec,
                elapsed_ms: stats.elapsed_ms,
            });
        }
    }
    Ok(DirectBasisPhaseReport {
        history,
        best_loss,
        best_step,
    })
}

fn direct_basis_train_step(
    base: &mut NpaModel,
    examples: &mut [DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    indices: &[usize],
    config: DirectBasisTrainConfig,
    step: usize,
) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
    if indices.is_empty() {
        return Err(std::io::Error::other("direct basis step requires examples").into());
    }
    let start = Instant::now();
    let mut base_grads = zero_model_gradients(base);
    let example_scale = 1.0 / indices.len() as f32;
    let mut loss_sum = 0.0_f32;
    let mut adapter_grad_sum = 0.0_f32;
    let mut adapter_grad_max = 0.0_f32;
    let mut particle_steps = 0.0_f64;
    for &idx in indices {
        let example = examples
            .get_mut(idx)
            .ok_or_else(|| std::io::Error::other("direct basis index is out of range"))?;
        let adapted = example.adapter.apply_to_model(base)?;
        let particle_count = example.source.particles.unwrap_or(config.rollout_particles);
        let update_prob = example.source.update_prob.unwrap_or(config.update_prob);
        let seed_scale = example.source.seed_scale.unwrap_or(config.seed_scale);
        let (loss, full_grads) = target_2d_rollout_loss_with_gradients(
            &adapted,
            hashgrid,
            &example.target,
            RolloutConfig {
                batch_size: 1,
                particle_count,
                steps: config.rollout_steps,
                update_prob,
                seed: config
                    .seed
                    .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9))
                    .wrapping_add(idx as u64),
                seed_scale,
                ..RolloutConfig::default()
            },
            config.seed_mode,
            config.loss_config,
            config.per_parameter_grad_normalization,
        )?;
        loss_sum += loss.total_loss;
        example.last_train_loss = Some(loss.total_loss);
        let mut adapter_grads =
            project_low_rank_adapter_gradients(base, &example.adapter, &full_grads)?;
        add_adapter_l2_gradients(
            &example.adapter,
            &mut adapter_grads,
            config.adapter_l2_weight,
        );
        let adapter_step =
            apply_sgd_adapter_gradients(&mut example.adapter, &adapter_grads, config.adapter_sgd)?;
        adapter_grad_sum += adapter_step.grad_norm;
        adapter_grad_max = adapter_grad_max.max(adapter_step.grad_norm);
        if config.update_base {
            add_scaled_model_gradients(&mut base_grads, &full_grads, example_scale);
        }
        particle_steps += particle_count as f64 * config.rollout_steps as f64;
    }
    let (base_grad_norm, base_grad_scale) = if config.update_base {
        let step_report = apply_sgd_gradients(base, &base_grads, config.base_sgd)?;
        (step_report.grad_norm, step_report.grad_scale)
    } else {
        (0.0, 1.0)
    };
    let elapsed = start.elapsed();
    Ok(DirectBasisStepStats {
        loss: loss_sum / indices.len() as f32,
        base_grad_norm,
        base_grad_scale,
        mean_adapter_grad_norm: adapter_grad_sum / indices.len() as f32,
        max_adapter_grad_norm: adapter_grad_max,
        examples_seen: indices.len(),
        particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    })
}

fn evaluate_direct_basis_examples(
    base: &NpaModel,
    examples: &[DirectBasisExample],
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: DirectBasisTrainConfig,
    requested_examples: usize,
    seed: u64,
) -> Result<Option<CliHyper2dDirectBasisLossSummary>, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(None);
    }
    let indices = eval_indices(examples.len(), requested_examples, seed);
    let mut total = CliHyper2dDirectBasisLossSummary {
        examples: indices.len(),
        mean_total_loss: 0.0,
        max_total_loss: 0.0,
        mean_splat_loss: 0.0,
        mean_color_loss: 0.0,
        mean_density_loss: 0.0,
    };
    for &idx in &indices {
        let example = &examples[idx];
        let loss = evaluate_direct_basis_example(
            base,
            example,
            hashgrid,
            EvalConfig {
                particle_count: example.source.particles.unwrap_or(config.rollout_particles),
                rollout_steps: config.rollout_steps,
                update_prob: example.source.update_prob.unwrap_or(config.update_prob),
                seed: seed.wrapping_add(idx as u64),
                seed_scale: example.source.seed_scale.unwrap_or(config.seed_scale),
                seed_mode: config.seed_mode,
            },
            config.loss_config,
        )?;
        total.mean_total_loss += loss.total_loss;
        total.max_total_loss = total.max_total_loss.max(loss.total_loss);
        total.mean_splat_loss += loss.splat_loss;
        total.mean_color_loss += loss.color_loss;
        total.mean_density_loss += loss.density_loss;
    }
    let scale = 1.0 / indices.len() as f32;
    total.mean_total_loss *= scale;
    total.mean_splat_loss *= scale;
    total.mean_color_loss *= scale;
    total.mean_density_loss *= scale;
    Ok(Some(total))
}

fn eval_indices(examples_len: usize, requested_examples: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..examples_len).collect::<Vec<_>>();
    if requested_examples == 0 || requested_examples >= examples_len {
        return indices;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices.truncate(requested_examples);
    indices.sort_unstable();
    indices
}

fn evaluate_direct_basis_example(
    base: &NpaModel,
    example: &DirectBasisExample,
    hashgrid: &burn_automata_kernels::HashGridConfig,
    config: EvalConfig,
    loss_config: Target2dLossConfig,
) -> Result<Target2dLossReport, Box<dyn std::error::Error>> {
    let model = example.adapter.apply_to_model(base)?;
    let trace = run_rollout(
        &model,
        hashgrid,
        &RolloutConfig {
            batch_size: 1,
            particle_count: config.particle_count,
            steps: config.rollout_steps,
            update_prob: config.update_prob,
            seed: config.seed,
            seed_scale: config.seed_scale,
            ..RolloutConfig::default()
        },
        config.seed_mode,
    )?;
    Ok(target_2d_loss_with_adjoint(
        &trace.positions,
        &trace.states,
        trace.batch_size,
        trace.particle_count,
        trace.state_dims,
        &example.target,
        loss_config,
        trace.mean_dx.iter().copied().sum(),
        trace.steps,
    )?
    .report)
}

#[derive(Clone, Copy)]
struct EvalConfig {
    particle_count: usize,
    rollout_steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
}

fn save_direct_basis_adapters(
    base_manifest: &BpkModelManifest,
    base_model_path: &Path,
    adapter_dir: &Path,
    examples: &[&DirectBasisExample],
) -> Result<Vec<CliHyper2dDirectBasisAdapterReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::with_capacity(examples.len());
    for example in examples {
        let slug = sanitize_slug(&example.source.slug);
        let adapter_path = adapter_dir.join(format!("{slug}.adapter.json"));
        let adapter_manifest = BpkAdapterManifest::from_adapter(
            base_manifest,
            Some(base_model_path.display().to_string()),
            example.adapter.clone(),
            Some(format!(
                "hyper2d-direct-basis:{}",
                example.source.condition_path.display()
            )),
        )?;
        crate::import::save_adapter_manifest(&adapter_path, &adapter_manifest)?;
        reports.push(CliHyper2dDirectBasisAdapterReport {
            slug: example.source.slug.clone(),
            split: example.split.label(),
            title: example.source.title.clone(),
            group: example.source.group.clone(),
            condition: example.source.condition_path.display().to_string(),
            adapter_output: adapter_path.display().to_string(),
            target_source_width: example.target.source_width,
            target_source_height: example.target.source_height,
            target_points: example.target.point_count(),
            last_train_loss: example.last_train_loss,
            adapter_parameter_count: example.adapter.parameter_count(),
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_indices_are_sorted_and_bounded() {
        assert_eq!(eval_indices(3, 0, 1), vec![0, 1, 2]);
        let indices = eval_indices(10, 4, 9);

        assert_eq!(indices.len(), 4);
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(indices.iter().all(|idx| *idx < 10));
    }

    #[test]
    fn zero_oracle_selection_loads_no_entries() {
        let entries = vec![(0, test_adapter_bank_entry("a", "holdout"))];
        assert!(select_direct_basis_adapter_bank_oracle_entries(&entries, 0, 7).is_empty());
    }

    #[test]
    fn oracle_selection_excludes_alpha_less_jpeg_targets() {
        let png = test_adapter_bank_entry("alpha", "train");
        let mut jpeg = test_adapter_bank_entry("opaque", "train");
        jpeg.condition = "opaque.JPG".to_string();
        let selected =
            select_direct_basis_adapter_bank_oracle_entries(&[(0, png), (1, jpeg)], 2, 7);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].1.slug, "alpha");
    }

    #[test]
    fn adapter_bank_selection_manifest_replays_split_slug_rows() {
        let train = vec![
            (0, test_adapter_bank_entry("same", "train")),
            (1, test_adapter_bank_entry("same", "train")),
            (2, test_adapter_bank_entry("other", "train")),
        ];
        let holdout = vec![(0, test_adapter_bank_entry("same", "holdout"))];
        let manifest = DirectBasisAdapterBankSelectionManifest {
            selection_seed: 7,
            train: vec![DirectBasisAdapterBankSelectionEntry {
                split: "train".to_string(),
                slug: "same".to_string(),
                bank_split_index: 1,
            }],
            holdout: vec![DirectBasisAdapterBankSelectionEntry {
                split: "holdout".to_string(),
                slug: "same".to_string(),
                bank_split_index: 0,
            }],
        };

        let selected_train =
            replay_direct_basis_adapter_bank_selection(&train, &manifest.train).unwrap();
        let selected_holdout =
            replay_direct_basis_adapter_bank_selection(&holdout, &manifest.holdout).unwrap();

        assert_eq!(selected_train.len(), 1);
        assert_eq!(selected_train[0].0, 1);
        assert_eq!(selected_holdout.len(), 1);
        assert_eq!(selected_holdout[0].1.split, "holdout");
    }

    #[test]
    fn adapter_bank_selection_manifest_rejects_missing_rows() {
        let entries = vec![(0, test_adapter_bank_entry("a", "train"))];
        let selection = vec![DirectBasisAdapterBankSelectionEntry {
            split: "train".to_string(),
            slug: "missing".to_string(),
            bank_split_index: 0,
        }];

        let err = replay_direct_basis_adapter_bank_selection(&entries, &selection)
            .unwrap_err()
            .to_string();

        assert!(err.contains("not present"));
    }

    #[test]
    fn direct_basis_arg_validation_rejects_empty_rollouts() {
        let err = validate_direct_basis_args(DirectBasisArgCheck {
            adapter_rank: 1,
            adapter_alpha: 1.0,
            rollout_particles: 0,
            rollout_steps: 1,
            update_prob: 0.5,
            tbptt_chunk_steps: 1,
            eval_batch_size: 1,
            system_memory_budget_gb: Some(24.0),
            gpu_memory_budget_gb: Some(DEFAULT_WGPU_VRAM_BUDGET_GB),
            max_dense_train_particles: 2048,
            max_dense_chunk_floats: 4 * 1024 * 1024,
            max_splat_chunk_floats: 4 * 1024 * 1024,
            base_learning_rate: 1.0e-4,
            adapter_learning_rate: 1.0e-3,
            train_adapter_refine_learning_rate: None,
            holdout_adapter_learning_rate: None,
            adapter_l2: 0.0,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("rollout particles"));
    }

    #[test]
    fn burn_wgpu_preflight_accepts_2048_particle_tiled_training() {
        let base_config = NpaConfig::growing_2d();
        let train_example = test_direct_basis_example(2048, &base_config);
        let config = DirectBasisTrainConfig {
            tbptt_chunk_steps: 1,
            loss_config: Target2dLossConfig {
                image_size: 128,
                ..Target2dLossConfig::default()
            },
            max_dense_train_particles: 2048,
            max_dense_chunk_floats: 512 * 1024,
            max_splat_chunk_floats: 512 * 1024,
            ..test_direct_basis_train_config(1, 2048)
        };
        let report = validate_direct_basis_burn_wgpu_preflight(
            &[train_example],
            &[],
            &base_config,
            config,
            DirectBasisTrainConfig { steps: 0, ..config },
            DirectBasisTrainConfig { steps: 0, ..config },
        )
        .unwrap();

        assert!(report.training_requested);
        assert!(report.dense_train_particle_cap_passed);
        assert!(report.memory_budget_passed);
        assert!(report.gpu_memory_budget_passed);
        assert_eq!(report.max_training_particles, 2048);
        assert_eq!(report.max_dense_train_particles, 2048);
        assert_eq!(report.tbptt_chunk_steps, 1);
    }

    #[test]
    fn burn_wgpu_preflight_rejects_2048_when_particle_cap_is_1024() {
        let base_config = NpaConfig::growing_2d();
        let train_example = test_direct_basis_example(2048, &base_config);
        let config = DirectBasisTrainConfig {
            max_dense_train_particles: 1024,
            ..test_direct_basis_train_config(1, 2048)
        };
        let err = validate_direct_basis_burn_wgpu_preflight(
            &[train_example],
            &[],
            &base_config,
            config,
            DirectBasisTrainConfig { steps: 0, ..config },
            DirectBasisTrainConfig { steps: 0, ..config },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("capped at 1024 particles"));
    }

    #[test]
    fn burn_wgpu_preflight_accepts_staged_384_particle_training() {
        let base_config = NpaConfig::growing_2d();
        let train_example = test_direct_basis_example(384, &base_config);
        let config = DirectBasisTrainConfig {
            loss_config: Target2dLossConfig {
                image_size: 96,
                ..Target2dLossConfig::default()
            },
            max_dense_chunk_floats: 512 * 1024,
            max_splat_chunk_floats: 512 * 1024,
            ..test_direct_basis_train_config(1, 384)
        };
        let report = validate_direct_basis_burn_wgpu_preflight(
            &[train_example],
            &[],
            &base_config,
            config,
            DirectBasisTrainConfig { steps: 0, ..config },
            DirectBasisTrainConfig { steps: 0, ..config },
        )
        .unwrap();

        assert!(report.dense_train_particle_cap_passed);
        assert!(report.memory_budget_passed);
        assert!(report.gpu_memory_budget_passed);
        assert_eq!(report.max_training_particles, 384);
    }

    #[test]
    fn burn_wgpu_preflight_rejects_unsafe_512_particle_batch_vram() {
        let base_config = NpaConfig::growing_2d();
        let train_examples = (0..4)
            .map(|_| test_direct_basis_example(512, &base_config))
            .collect::<Vec<_>>();
        let config = DirectBasisTrainConfig {
            example_batch_size: 4,
            gpu_memory_budget_gb: Some(0.1),
            ..test_direct_basis_train_config(1, 512)
        };
        let err = validate_direct_basis_burn_wgpu_preflight(
            &train_examples,
            &[],
            &base_config,
            config,
            DirectBasisTrainConfig { steps: 0, ..config },
            DirectBasisTrainConfig { steps: 0, ..config },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("peak VRAM"));
    }

    #[test]
    fn burn_wgpu_preflight_rejects_unsafe_512_particle_refine_vram() {
        let base_config = NpaConfig::growing_2d();
        let train_example = test_direct_basis_example(512, &base_config);
        let config = DirectBasisTrainConfig {
            loss_config: Target2dLossConfig {
                image_size: 96,
                ..Target2dLossConfig::default()
            },
            gpu_memory_budget_gb: Some(0.1),
            ..test_direct_basis_train_config(1, 512)
        };
        let err = validate_direct_basis_burn_wgpu_preflight(
            &[train_example],
            &[],
            &base_config,
            config,
            DirectBasisTrainConfig { steps: 0, ..config },
            DirectBasisTrainConfig { steps: 0, ..config },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("peak VRAM"));
    }

    #[test]
    fn burn_wgpu_preflight_allows_2048_particle_validation_only() {
        let base_config = NpaConfig::growing_2d();
        let train_example = test_direct_basis_example(2048, &base_config);
        let config = DirectBasisTrainConfig {
            steps: 0,
            ..test_direct_basis_train_config(1, 2048)
        };
        let report = validate_direct_basis_burn_wgpu_preflight(
            &[train_example],
            &[],
            &base_config,
            config,
            config,
            config,
        )
        .unwrap();

        assert!(!report.training_requested);
        assert!(report.dense_train_particle_cap_passed);
        assert!(report.memory_budget_passed);
        assert!(report.gpu_memory_budget_passed);
    }

    #[test]
    fn direct_basis_experiment_config_accepts_cli_enum_names() {
        let config: DirectBasisExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            catalog_group = "growing"

            [source.omnisvg]
            dataset = "illustration"

            [training]
            device = "gpu"

            [gpu]
            backend = "burn-wgpu"

            [rollout]
            seed_mode = "uniform-circle"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config_value_enum("preset", config.preset, PresetArg::Texture2d).unwrap(),
            PresetArg::Growing2d
        ));
        assert!(matches!(
            config_value_enum_option::<Hyper2dCatalogGroupArg>(
                "source.catalog_group",
                config.source.catalog_group,
                None,
            )
            .unwrap(),
            Some(Hyper2dCatalogGroupArg::Growing)
        ));
        assert!(matches!(
            config_value_enum_option::<OmniSvgDatasetArg>(
                "source.omnisvg.dataset",
                config.source.omnisvg.dataset,
                None,
            )
            .unwrap(),
            Some(OmniSvgDatasetArg::MmsvgIllustration)
        ));
        assert!(matches!(
            config_value_enum(
                "training.device",
                config.training.device,
                TrainingDeviceArg::Cpu,
            )
            .unwrap(),
            TrainingDeviceArg::Gpu
        ));
        assert!(matches!(
            config_value_enum(
                "gpu.backend",
                config.gpu.backend,
                Hyper2dDirectBasisGpuBackendArg::BurnWgpu,
            )
            .unwrap(),
            Hyper2dDirectBasisGpuBackendArg::BurnWgpu
        ));
        assert!(
            config_value_enum::<Hyper2dDirectBasisGpuBackendArg>(
                "gpu.backend",
                Some("legacy-upstream-python".to_string()),
                Hyper2dDirectBasisGpuBackendArg::BurnWgpu,
            )
            .is_err()
        );
        assert!(
            config_value_enum::<Hyper2dDirectBasisGpuBackendArg>(
                "gpu.backend",
                Some("upstream-python".to_string()),
                Hyper2dDirectBasisGpuBackendArg::BurnWgpu,
            )
            .is_err()
        );
        assert!(matches!(
            config_value_enum(
                "rollout.seed_mode",
                config.rollout.seed_mode,
                SeedModeArg::Uniform
            )
            .unwrap(),
            SeedModeArg::UniformCircle
        ));
    }

    #[test]
    fn direct_basis_experiment_config_accepts_nested_toml() {
        let config: DirectBasisExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["target.png"]

            [output]
            output_dir = "artifacts/hyper2d_direct_basis"

            [training]
            device = "gpu"

            [gpu]
            backend = "burn-wgpu"

            [rollout]
            seed_mode = "uniform-circle"
            "#,
        )
        .unwrap();

        assert_eq!(config.preset.as_deref(), Some("growing-2d"));
        assert_eq!(
            config.source.target_images.as_deref(),
            Some(&[PathBuf::from("target.png")][..])
        );
        assert_eq!(config.training.device.as_deref(), Some("gpu"));
        assert_eq!(config.gpu.backend.as_deref(), Some("burn-wgpu"));
        assert_eq!(config.rollout.seed_mode.as_deref(), Some("uniform-circle"));
    }

    #[test]
    fn oracle_backend_parser_accepts_burn_names_only() {
        assert_eq!(
            config_value_enum(
                "oracle.backend",
                Some("burn-wgpu".to_string()),
                DirectBasisOracleBackendArg::Cpu,
            )
            .unwrap(),
            DirectBasisOracleBackendArg::Wgpu
        );
        assert_eq!(
            config_value_enum(
                "oracle.backend",
                Some("burn-cuda".to_string()),
                DirectBasisOracleBackendArg::Cpu,
            )
            .unwrap(),
            DirectBasisOracleBackendArg::Cuda
        );
        assert!(
            config_value_enum::<DirectBasisOracleBackendArg>(
                "oracle.backend",
                Some("legacy-upstream-python".to_string()),
                DirectBasisOracleBackendArg::Cpu,
            )
            .is_err()
        );
        assert!(
            config_value_enum::<DirectBasisOracleBackendArg>(
                "oracle.backend",
                Some("gpu".to_string()),
                DirectBasisOracleBackendArg::Cpu,
            )
            .is_err()
        );
    }

    #[test]
    fn oracle_config_rejects_invalid_parallel_jobs() {
        let mut config = DirectBasisOracleConfig {
            backend: DirectBasisOracleBackendArg::Cuda,
            gpu_device: "cuda:0".to_string(),
            resume_existing: false,
            gpu_parallel_jobs: 0,
            train_examples: 1,
            holdout_examples: 0,
            epochs: 1,
            repetitions: 1,
            report_interval: 1,
            batch_size: 1,
            pool_size: 1,
            rollout_step_min: 1,
            tbptt_chunk_steps: 1,
            loss_on_final_chunk_only: true,
            use_particle_pool: true,
            inject_seed_interval: 16,
            brush_size: 0.1,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            grad_clip_norm: 1.0,
            seed: 42,
        };
        assert!(validate_oracle_config(&config).is_err());
        config.gpu_parallel_jobs = 1;
        assert!(validate_oracle_config(&config).is_ok());
        config.gpu_parallel_jobs = 65;
        assert!(validate_oracle_config(&config).is_err());
    }

    fn test_adapter_bank_entry(slug: &str, split: &str) -> DirectBasisAdapterBankLoadEntry {
        DirectBasisAdapterBankLoadEntry {
            slug: slug.to_string(),
            split: split.to_string(),
            title: None,
            group: None,
            condition: format!("{slug}.png"),
            adapter_output: format!("{slug}.adapter.json"),
            target_source_width: 0,
            target_source_height: 0,
            target_points: 0,
            last_train_loss: None,
        }
    }

    fn test_direct_basis_train_config(
        steps: usize,
        rollout_particles: usize,
    ) -> DirectBasisTrainConfig {
        DirectBasisTrainConfig {
            steps,
            report_interval: 1,
            example_batch_size: 1,
            tbptt_chunk_steps: 4,
            loss_on_final_chunk_only: false,
            use_particle_pool: false,
            pool_size: 0,
            inject_seed_interval: 0,
            brush_size: 0.0,
            stopgrad_pos: true,
            stopgrad_state: false,
            rollout_particles,
            rollout_step_min: 8,
            rollout_steps: 8,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.5,
            seed_mode: ParticleSeed::UniformCircle,
            grid_eps: 0.125,
            motion_scale: 1.0,
            loss_config: Target2dLossConfig {
                image_size: 32,
                ..Target2dLossConfig::default()
            },
            target2d_loss_backend: Target2dLossBackend::Dense,
            perception_backend: PerceptionRolloutBackend::Dense,
            per_parameter_grad_normalization: true,
            base_sgd: SgdConfig {
                learning_rate: 1.0e-4,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
            adapter_sgd: SgdConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
            adapter_l2_weight: 0.0,
            update_base: true,
            eval_examples: 1,
            eval_interval: 1,
            eval_batch_size: 1,
            eval_seed: 42,
            system_memory_budget_gb: Some(24.0),
            gpu_memory_budget_gb: Some(DEFAULT_WGPU_VRAM_BUDGET_GB),
            max_dense_train_particles: 2048,
            max_dense_chunk_floats: 4 * 1024 * 1024,
            max_splat_chunk_floats: 4 * 1024 * 1024,
        }
    }

    fn test_direct_basis_example(particles: usize, base_config: &NpaConfig) -> DirectBasisExample {
        DirectBasisExample {
            source: super::super::sources::Hyper2dScratchSource {
                slug: format!("test-{particles}"),
                title: None,
                group: None,
                condition_path: PathBuf::from("test.png"),
                particles: Some(particles),
                seed_scale: None,
                update_prob: None,
            },
            split: Hyper2dE2eSplit::Train,
            bank_split_index: None,
            target: TargetImage2d {
                source_width: 1,
                source_height: 1,
                positions: vec![[0.0, 0.0]],
                colors: vec![[1.0, 1.0, 1.0]],
                pixel_size: 1.0,
                threshold: 0.0,
                aabb: [-1.0, -1.0, 1.0, 1.0],
            },
            adapter: NpaLowRankAdapter::seeded_zero_delta(base_config, 1, 1.0, 42),
            last_train_loss: None,
        }
    }
}
