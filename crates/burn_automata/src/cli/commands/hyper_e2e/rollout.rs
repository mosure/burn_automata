#[cfg(feature = "dino")]
use crate::cli::commands::hyper_support::load_condition_image_2d;
use crate::cli::commands::hyper_support::write_pretty_json;
use crate::cli::prelude::*;
use crate::hyper::condition::DINO_VITS_EMBED_DIMS;
use std::time::Instant;

use super::direct_basis::{
    BurnE2eRolloutExample, BurnE2eRolloutOutput, BurnE2eRolloutTrainConfig, E2eLrSchedule,
    train_e2e_rollout_burn_cuda, train_e2e_rollout_burn_wgpu,
};
use super::sources::{
    Hyper2dScratchSource, OmniSvgSourceConfig, ScratchSourceResolveConfig, resolve_scratch_sources,
};
use super::{Hyper2dE2eSplit, omnisvg_source_report, resolve_e2e_splits};

const DEFAULT_OUTPUT_DIR: &str = "artifacts/hyper2d_e2e_rollout";
const DEFAULT_DINO_IMAGE_SIZE: usize = 518;
const DEFAULT_DINO_PATCH_SIZE: usize = 14;
const DEFAULT_MAX_DENSE_TRAIN_PARTICLES: usize = 512;
const DEFAULT_DEVICE_CONDITION_CACHE_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutExperimentConfig {
    preset: Option<String>,
    source: RolloutSourceConfig,
    split: RolloutSplitConfig,
    output: RolloutOutputConfig,
    condition: RolloutConditionConfig,
    model: RolloutModelConfig,
    training: RolloutTrainingConfig,
    gpu: RolloutGpuConfig,
    adapter: RolloutAdapterConfig,
    rollout: RolloutRuntimeConfig,
    target: RolloutTargetConfig,
    optimizer: RolloutOptimizerConfig,
    validation: RolloutValidationConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutSourceConfig {
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
    omnisvg: RolloutOmniSvgConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutOmniSvgConfig {
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
struct RolloutSplitConfig {
    holdout_targets: Option<Vec<String>>,
    holdout_stride: Option<usize>,
    holdout_offset: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutOutputConfig {
    output_dir: Option<PathBuf>,
    report_output: Option<PathBuf>,
    shared_base_output: Option<PathBuf>,
    hyper_output: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutConditionConfig {
    encoder: Option<String>,
    dino_model: Option<PathBuf>,
    dino_image_size: Option<usize>,
    dino_patch_size: Option<usize>,
    dino_batch_size: Option<usize>,
    online: Option<bool>,
    feature_cache: Option<PathBuf>,
    token_grid_width: Option<usize>,
    token_grid_height: Option<usize>,
    token_attention_heads: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutModelConfig {
    shared_base: Option<PathBuf>,
    shared_base_trainable: Option<bool>,
    shared_base_init: Option<String>,
    hidden_dims: Option<usize>,
    output_activation: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutTrainingConfig {
    backend: Option<String>,
    objective: Option<String>,
    steps: Option<usize>,
    report_interval: Option<usize>,
    example_batch_size: Option<usize>,
    tbptt_chunk_steps: Option<usize>,
    max_dense_train_particles: Option<usize>,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutGpuConfig {
    backend: Option<String>,
    max_dense_chunk_floats: Option<usize>,
    max_splat_chunk_floats: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutAdapterConfig {
    rank: Option<usize>,
    alpha: Option<f32>,
    generator: Option<String>,
    flow_hidden: Option<usize>,
    flow_sample_steps: Option<usize>,
    flow_source_scale: Option<f32>,
    init_scale: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutRuntimeConfig {
    particles: Option<usize>,
    steps: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    seed_scale: Option<f32>,
    seed_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutTargetConfig {
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
struct RolloutOptimizerConfig {
    learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
    adam_beta1: Option<f32>,
    adam_beta2: Option<f32>,
    adam_epsilon: Option<f32>,
    per_parameter_grad_normalization: Option<bool>,
    lr_schedule: Option<String>,
    min_lr_scale: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutValidationConfig {
    examples: Option<usize>,
    particles: Option<usize>,
    steps: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    oracle_report: Option<PathBuf>,
    psnr_threshold_db: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum RolloutConditionEncoder {
    DinoVitsFullTokens,
    DinoVitsTokenGrid,
}

impl RolloutConditionEncoder {
    const fn label(self) -> &'static str {
        match self {
            Self::DinoVitsFullTokens => "dino-vits-full-tokens",
            Self::DinoVitsTokenGrid => "dino-vits-token-grid",
        }
    }
}

#[derive(Clone, Serialize)]
struct E2eRolloutReport {
    experiment_config: String,
    status: &'static str,
    implementation_status: &'static str,
    preset: AutomataPreset,
    source: E2eRolloutSourceReport,
    split: E2eRolloutSplitReport,
    output: E2eRolloutOutputReport,
    condition: E2eRolloutConditionReport,
    model: E2eRolloutModelReport,
    training: E2eRolloutTrainingReport,
    adapter: E2eRolloutAdapterReport,
    rollout: E2eRolloutRuntimeReport,
    target: E2eRolloutTargetReport,
    optimizer: E2eRolloutOptimizerReport,
    validation: E2eRolloutValidationReport,
    blockers: Vec<String>,
    selected_sources: Vec<E2eRolloutSourceEntry>,
    training_result: Option<E2eRolloutTrainingResultReport>,
}

#[derive(Clone, Serialize)]
struct E2eRolloutSourceReport {
    requested_source_limit: usize,
    selected_sources: usize,
    train_examples: usize,
    holdout_examples: usize,
    omnisvg: Option<CliOmniSvgSourceReport>,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutSplitReport {
    holdout_targets: Vec<String>,
    holdout_stride: usize,
    holdout_offset: usize,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutOutputReport {
    output_dir: String,
    report_output: String,
    shared_base_output: String,
    hyper_output: String,
    checkpoint_dir: String,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutConditionReport {
    encoder: &'static str,
    online_dino: bool,
    disk_feature_cache: Option<String>,
    dino_model: Option<String>,
    dino_image_size: usize,
    dino_patch_size: usize,
    dino_batch_size: usize,
    patch_grid_width: usize,
    patch_grid_height: usize,
    token_count: usize,
    embed_dims: usize,
    flattened_feature_dims: usize,
    selected_feature_cache_bytes_f32: usize,
    selected_feature_cache_gib_f32: f64,
    dino_batch_input_bytes_f32: usize,
    dino_batch_input_gib_f32: f64,
    projected_condition_load_peak_bytes_f32: usize,
    projected_condition_load_peak_gib_f32: f64,
    token_attention_heads: usize,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutModelReport {
    shared_base: Option<String>,
    shared_base_trainable: bool,
    shared_base_init: String,
    hidden_dims: usize,
    output_activation: String,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutTrainingReport {
    backend: String,
    gpu_backend: String,
    objective: String,
    steps: usize,
    report_interval: usize,
    example_batch_size: usize,
    tbptt_chunk_steps: usize,
    max_dense_train_particles: usize,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    max_dense_chunk_floats: usize,
    max_splat_chunk_floats: usize,
    seed: u64,
    adapter_vector_mse_primary_objective: bool,
    trains_shared_base_from_step_zero: bool,
    trains_hypernet_from_step_zero: bool,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutAdapterReport {
    rank: usize,
    alpha: f32,
    generator: String,
    flow_hidden: usize,
    flow_sample_steps: usize,
    flow_source_scale: f32,
    init_scale: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutRuntimeReport {
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: Option<f32>,
    seed_mode: ParticleSeed,
    estimated_pair_interactions_per_example: usize,
    estimated_pair_interactions_per_batch: usize,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutTargetReport {
    points: usize,
    image_size: Option<usize>,
    threshold: f32,
    loss_image_size: usize,
    splat_sigma: f32,
    splat_loss_weight: f32,
    color_loss_weight: f32,
    density_loss_weight: f32,
    displacement_regularizer_weight: f32,
    overflow_regularizer_weight: f32,
    bound_regularizer_weight: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutOptimizerReport {
    learning_rate: f32,
    weight_decay: f32,
    grad_clip_norm: f32,
    adam_beta1: f32,
    adam_beta2: f32,
    adam_epsilon: f32,
    per_parameter_grad_normalization: bool,
    lr_schedule: String,
    min_lr_scale: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutValidationReport {
    examples: usize,
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    oracle_report: Option<String>,
    psnr_threshold_db: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutTrainingResultReport {
    backend: String,
    device: String,
    final_loss: Option<f32>,
    history: serde_json::Value,
    metrics: serde_json::Value,
    quality_validation: Option<serde_json::Value>,
    shared_base_output: String,
    hyper_output: String,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutSourceEntry {
    slug: String,
    title: Option<String>,
    group: Option<String>,
    split: &'static str,
    condition_path: String,
    particles: Option<usize>,
    seed_scale: Option<f32>,
    update_prob: Option<f32>,
}

pub(crate) fn run_train_hyper_2d_e2e_rollout(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dE2eRollout { config } = command else {
        unreachable!("run_train_hyper_2d_e2e_rollout called with wrong command variant");
    };
    let experiment_config = load_rollout_experiment_config(&config)?;
    let mut report = build_e2e_rollout_report(&config, &experiment_config)?;
    write_pretty_json(Path::new(&report.output.report_output), &report)?;

    if report.training.steps > 0 {
        let training = run_burn_e2e_rollout_training(&report)?;
        let shared_base_output = report.output.shared_base_output.clone();
        let hyper_output = report.output.hyper_output.clone();
        let quality_validation = training
            .quality_validation
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;
        report.training_result = Some(E2eRolloutTrainingResultReport {
            backend: training.backend,
            device: training.device,
            final_loss: training.final_loss,
            history: serde_json::to_value(training.history)?,
            metrics: training.metrics,
            quality_validation,
            shared_base_output,
            hyper_output,
        });
        write_pretty_json(Path::new(&report.output.report_output), &report)?;
        eprintln!(
            "hyper2d e2e rollout training wrote {} and {}",
            report.output.shared_base_output, report.output.hyper_output
        );
        return Ok(());
    }

    eprintln!(
        "hyper2d e2e rollout preflight wrote {} ({} sources, {} train, {} holdout)",
        report.output.report_output,
        report.source.selected_sources,
        report.source.train_examples,
        report.source.holdout_examples
    );
    Ok(())
}

fn load_rollout_experiment_config(
    path: &Path,
) -> Result<RolloutExperimentConfig, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!(
            "failed to parse HyperNPA e2e rollout config {}: {err}",
            path.display()
        ))
        .into()
    })
}

fn build_e2e_rollout_report(
    config_path: &Path,
    config: &RolloutExperimentConfig,
) -> Result<E2eRolloutReport, Box<dyn std::error::Error>> {
    let preset_arg = parse_value_enum("preset", config.preset.as_deref(), PresetArg::Growing2d)?;
    let preset: AutomataPreset = preset_arg.into();
    if preset != AutomataPreset::Growing2d {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e-rollout currently supports growing-2d HyperNPA only",
        )
        .into());
    }

    let output_dir = config
        .output
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    let report_output = config
        .output
        .report_output
        .clone()
        .unwrap_or_else(|| output_dir.join("report.json"));
    let shared_base_output = config
        .output
        .shared_base_output
        .clone()
        .unwrap_or_else(|| output_dir.join("shared_base.bpk"));
    let hyper_output = config
        .output
        .hyper_output
        .clone()
        .unwrap_or_else(|| output_dir.join("hyper_2d.json"));
    let checkpoint_dir = config
        .output
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| output_dir.join("checkpoints"));

    let dino_image_size = config
        .condition
        .dino_image_size
        .unwrap_or(DEFAULT_DINO_IMAGE_SIZE);
    let dino_patch_size = config
        .condition
        .dino_patch_size
        .unwrap_or(DEFAULT_DINO_PATCH_SIZE);
    let encoder = parse_rollout_condition_encoder(config.condition.encoder.as_deref())?;
    let patch_grid = dino_patch_grid(dino_image_size, dino_patch_size)?;
    let (token_grid_width, token_grid_height) = match encoder {
        RolloutConditionEncoder::DinoVitsFullTokens => (patch_grid, patch_grid),
        RolloutConditionEncoder::DinoVitsTokenGrid => (
            config.condition.token_grid_width.unwrap_or(patch_grid),
            config.condition.token_grid_height.unwrap_or(patch_grid),
        ),
    };
    if token_grid_width == 0 || token_grid_height == 0 {
        return Err(std::io::Error::other("DINO token grid dimensions must be positive").into());
    }
    if token_grid_width > patch_grid || token_grid_height > patch_grid {
        return Err(std::io::Error::other(format!(
            "DINO token grid {token_grid_width}x{token_grid_height} exceeds full patch grid {patch_grid}x{patch_grid}"
        ))
        .into());
    }
    let token_count = 1 + token_grid_width * token_grid_height;
    let flattened_feature_dims = token_count * DINO_VITS_EMBED_DIMS;
    let online_dino = config.condition.online.unwrap_or(true);
    if !online_dino {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e-rollout requires condition.online = true; cached feature-vector regression belongs in train-hyper2d-adapter-bank",
        )
        .into());
    }
    if config.condition.feature_cache.is_some() {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e-rollout does not accept condition.feature_cache because the e2e path must condition on online DINO tokens",
        )
        .into());
    }

    let backend = config
        .training
        .backend
        .clone()
        .unwrap_or_else(|| "gpu".to_string());
    if matches!(backend.as_str(), "cpu" | "host" | "ndarray") {
        return Err(
            std::io::Error::other("train-hyper2d-e2e-rollout requires a GPU Burn backend").into(),
        );
    }
    let gpu_backend = config
        .gpu
        .backend
        .clone()
        .or_else(|| config.training.backend.clone())
        .unwrap_or_else(|| "burn-wgpu".to_string());
    if !matches!(
        gpu_backend.as_str(),
        "burn-wgpu" | "wgpu" | "burn-cuda" | "cuda"
    ) {
        return Err(std::io::Error::other(format!(
            "unsupported gpu.backend {gpu_backend:?}; expected burn-wgpu or burn-cuda"
        ))
        .into());
    }
    let objective = config
        .training
        .objective
        .clone()
        .unwrap_or_else(|| "target2d-rollout-image-loss".to_string());
    if !matches!(
        objective.as_str(),
        "target2d-rollout-image-loss" | "rollout-image-loss" | "e2e-rollout-loss"
    ) {
        return Err(std::io::Error::other(format!(
            "training.objective {objective:?} is not valid for train-hyper2d-e2e-rollout; use target2d-rollout-image-loss"
        ))
        .into());
    }

    let target_images = config.source.target_images.clone().unwrap_or_default();
    let target_image_dirs = config.source.target_image_dirs.clone().unwrap_or_default();
    let image_extensions = config
        .source
        .image_extensions
        .clone()
        .unwrap_or_else(default_image_extensions);
    let catalog_group = parse_value_enum_option::<Hyper2dCatalogGroupArg>(
        "source.catalog_group",
        config.source.catalog_group.as_deref(),
    )?;
    let catalog_targets = config.source.catalog_targets.clone().unwrap_or_default();
    let catalog_thumbnail_dir = config
        .source
        .catalog_thumbnail_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/catalog_thumbnails"));
    let omnisvg_dataset = parse_value_enum_option::<OmniSvgDatasetArg>(
        "source.omnisvg.dataset",
        config.source.omnisvg.dataset.as_deref(),
    )?;
    let omnisvg_split = config.source.omnisvg.split.as_deref().unwrap_or("train");
    let omnisvg_cache_dir = config
        .source
        .omnisvg
        .cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/omnisvg"));
    let omnisvg_token_env = config
        .source
        .omnisvg
        .token_env
        .as_deref()
        .unwrap_or("HF_TOKEN");
    let omnisvg = omnisvg_dataset.map(|dataset| OmniSvgSourceConfig {
        dataset,
        split: omnisvg_split,
        cache_dir: &omnisvg_cache_dir,
        offset: config.source.omnisvg.offset.unwrap_or(0),
        limit: config.source.omnisvg.limit.unwrap_or(128),
        page_size: config.source.omnisvg.page_size.unwrap_or(100),
        download: config.source.omnisvg.download.unwrap_or(false),
        refresh: config.source.omnisvg.refresh.unwrap_or(false),
        token_env: omnisvg_token_env,
    });
    let mut sources = resolve_scratch_sources(ScratchSourceResolveConfig {
        preset: preset_arg,
        target_images: &target_images,
        target_image_dirs: &target_image_dirs,
        target_image_recursive: config.source.target_image_recursive.unwrap_or(false),
        image_extensions: &image_extensions,
        catalog: config.source.catalog.as_ref(),
        catalog_thumbnail_dir: &catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: &catalog_targets,
        catalog_limit: config.source.catalog_limit.unwrap_or(0),
        omnisvg,
    })?;
    let source_limit = config.source.source_limit.unwrap_or(0);
    if source_limit > 0 {
        sources.truncate(source_limit);
    }
    let holdout_targets = config.split.holdout_targets.clone().unwrap_or_default();
    let holdout_stride = config.split.holdout_stride.unwrap_or(0);
    let holdout_offset = config.split.holdout_offset.unwrap_or(0);
    let splits = resolve_e2e_splits(&sources, &holdout_targets, holdout_stride, holdout_offset)?;
    let train_examples = splits.iter().filter(|split| split.is_train()).count();
    let holdout_examples = splits.len() - train_examples;

    let steps = config.training.steps.unwrap_or(0);
    let report_interval = config.training.report_interval.unwrap_or(25).max(1);
    let example_batch_size = config.training.example_batch_size.unwrap_or(1).max(1);
    let tbptt_chunk_steps = config.training.tbptt_chunk_steps.unwrap_or(8).max(1);
    let max_dense_train_particles = config
        .training
        .max_dense_train_particles
        .unwrap_or(DEFAULT_MAX_DENSE_TRAIN_PARTICLES);
    let rollout_particles = config.rollout.particles.unwrap_or(512).max(1);
    let rollout_steps = config.rollout.steps.unwrap_or(32).max(1);
    if steps > 0 && rollout_particles > max_dense_train_particles {
        return Err(std::io::Error::other(format!(
            "rollout.particles={rollout_particles} exceeds training.max_dense_train_particles={max_dense_train_particles}; keep 2048-particle runs validation-only until the tiled/fused backward path lands"
        ))
        .into());
    }
    let seed_mode = parse_value_enum(
        "rollout.seed_mode",
        config.rollout.seed_mode.as_deref(),
        SeedModeArg::UniformCircle,
    )?;
    let seed_mode: ParticleSeed = seed_mode.into();
    let pair_interactions = rollout_particles
        .checked_mul(rollout_particles)
        .ok_or_else(|| std::io::Error::other("rollout particle pair count overflowed"))?;
    let batch_pair_interactions = pair_interactions
        .checked_mul(example_batch_size)
        .ok_or_else(|| std::io::Error::other("rollout batch particle pair count overflowed"))?;

    let adapter_rank = config.adapter.rank.unwrap_or(16).max(1);
    let adapter_alpha = config.adapter.alpha.unwrap_or(adapter_rank as f32);
    if adapter_alpha <= 0.0 {
        return Err(std::io::Error::other("adapter.alpha must be positive").into());
    }
    let lr_schedule = parse_e2e_lr_schedule(config.optimizer.lr_schedule.as_deref())?;
    let lr_schedule_label = lr_schedule.as_str().to_string();
    let min_lr_scale = config.optimizer.min_lr_scale.unwrap_or(1.0).clamp(0.0, 1.0);
    let target_report = E2eRolloutTargetReport {
        points: config.target.points.unwrap_or(4096).max(1),
        image_size: config.target.image_size,
        threshold: config.target.threshold.unwrap_or(0.05),
        loss_image_size: config.target.loss_image_size.unwrap_or(128).max(1),
        splat_sigma: config.target.splat_sigma.unwrap_or(1.0),
        splat_loss_weight: config.target.splat_loss_weight.unwrap_or(2.0),
        color_loss_weight: config.target.color_loss_weight.unwrap_or(5.0),
        density_loss_weight: config.target.density_loss_weight.unwrap_or(1.0),
        displacement_regularizer_weight: config
            .target
            .displacement_regularizer_weight
            .unwrap_or(0.01),
        overflow_regularizer_weight: config.target.overflow_regularizer_weight.unwrap_or(100.0),
        bound_regularizer_weight: config.target.bound_regularizer_weight.unwrap_or(100.0),
    };
    let validation_particles = config.validation.particles.unwrap_or(2048).max(1);
    let validation_steps = config.validation.steps.unwrap_or(64).max(1);
    let blockers = Vec::new();

    let dino_batch_size = config.condition.dino_batch_size.unwrap_or(1).max(1);
    let selected_feature_cache_bytes_f32 = sources
        .len()
        .saturating_mul(flattened_feature_dims)
        .saturating_mul(std::mem::size_of::<f32>());
    let dino_batch_input_bytes_f32 = dino_batch_size
        .saturating_mul(dino_image_size)
        .saturating_mul(dino_image_size)
        .saturating_mul(3)
        .saturating_mul(std::mem::size_of::<f32>());
    let projected_condition_load_peak_bytes_f32 =
        if selected_feature_cache_bytes_f32 > DEFAULT_DEVICE_CONDITION_CACHE_MAX_BYTES {
            selected_feature_cache_bytes_f32.saturating_add(dino_batch_input_bytes_f32)
        } else {
            selected_feature_cache_bytes_f32
                .saturating_mul(2)
                .saturating_add(dino_batch_input_bytes_f32)
        };

    Ok(E2eRolloutReport {
        experiment_config: config_path.display().to_string(),
        status: if blockers.is_empty() {
            "preflight-ok"
        } else {
            "blocked"
        },
        implementation_status: "burn_generated_adapter_rollout_loss_v1",
        preset,
        source: E2eRolloutSourceReport {
            requested_source_limit: source_limit,
            selected_sources: sources.len(),
            train_examples,
            holdout_examples,
            omnisvg: omnisvg_source_report(omnisvg),
        },
        split: E2eRolloutSplitReport {
            holdout_targets,
            holdout_stride,
            holdout_offset,
        },
        output: E2eRolloutOutputReport {
            output_dir: display_path(&output_dir),
            report_output: display_path(&report_output),
            shared_base_output: display_path(&shared_base_output),
            hyper_output: display_path(&hyper_output),
            checkpoint_dir: display_path(&checkpoint_dir),
        },
        condition: E2eRolloutConditionReport {
            encoder: encoder.label(),
            online_dino,
            disk_feature_cache: config
                .condition
                .feature_cache
                .as_ref()
                .map(|path| display_path(path)),
            dino_model: config
                .condition
                .dino_model
                .as_ref()
                .map(|path| display_path(path)),
            dino_image_size,
            dino_patch_size,
            dino_batch_size,
            patch_grid_width: token_grid_width,
            patch_grid_height: token_grid_height,
            token_count,
            embed_dims: DINO_VITS_EMBED_DIMS,
            flattened_feature_dims,
            selected_feature_cache_bytes_f32,
            selected_feature_cache_gib_f32: bytes_to_gib(selected_feature_cache_bytes_f32),
            dino_batch_input_bytes_f32,
            dino_batch_input_gib_f32: bytes_to_gib(dino_batch_input_bytes_f32),
            projected_condition_load_peak_bytes_f32,
            projected_condition_load_peak_gib_f32: bytes_to_gib(
                projected_condition_load_peak_bytes_f32,
            ),
            token_attention_heads: config.condition.token_attention_heads.unwrap_or(4).max(1),
        },
        model: E2eRolloutModelReport {
            shared_base: config
                .model
                .shared_base
                .as_ref()
                .map(|path| display_path(path)),
            shared_base_trainable: config.model.shared_base_trainable.unwrap_or(true),
            shared_base_init: config
                .model
                .shared_base_init
                .clone()
                .unwrap_or_else(|| "random".to_string()),
            hidden_dims: config.model.hidden_dims.unwrap_or(128).max(1),
            output_activation: config
                .model
                .output_activation
                .clone()
                .unwrap_or_else(|| "tanh".to_string()),
        },
        training: E2eRolloutTrainingReport {
            backend,
            gpu_backend,
            objective,
            steps,
            report_interval,
            example_batch_size,
            tbptt_chunk_steps,
            max_dense_train_particles,
            system_memory_budget_gb: config.training.system_memory_budget_gb,
            gpu_memory_budget_gb: config.training.gpu_memory_budget_gb,
            max_dense_chunk_floats: config.gpu.max_dense_chunk_floats.unwrap_or(1_048_576),
            max_splat_chunk_floats: config.gpu.max_splat_chunk_floats.unwrap_or(1_048_576),
            seed: config.training.seed.unwrap_or(42),
            adapter_vector_mse_primary_objective: false,
            trains_shared_base_from_step_zero: config.model.shared_base_trainable.unwrap_or(true),
            trains_hypernet_from_step_zero: true,
        },
        adapter: E2eRolloutAdapterReport {
            rank: adapter_rank,
            alpha: adapter_alpha,
            generator: config
                .adapter
                .generator
                .clone()
                .unwrap_or_else(|| "token-aware-rectified-flow".to_string()),
            flow_hidden: config.adapter.flow_hidden.unwrap_or(512).max(1),
            flow_sample_steps: config.adapter.flow_sample_steps.unwrap_or(16).max(1),
            flow_source_scale: config.adapter.flow_source_scale.unwrap_or(1.0),
            init_scale: config.adapter.init_scale.unwrap_or(1.0e-3),
        },
        rollout: E2eRolloutRuntimeReport {
            particles: rollout_particles,
            steps: rollout_steps,
            update_prob: config.rollout.update_prob.unwrap_or(0.5),
            seed: config.rollout.seed.unwrap_or(42),
            seed_scale: config.rollout.seed_scale,
            seed_mode,
            estimated_pair_interactions_per_example: pair_interactions,
            estimated_pair_interactions_per_batch: batch_pair_interactions,
        },
        target: target_report,
        optimizer: E2eRolloutOptimizerReport {
            learning_rate: config.optimizer.learning_rate.unwrap_or(1.0e-4),
            weight_decay: config.optimizer.weight_decay.unwrap_or(0.0),
            grad_clip_norm: config.optimizer.grad_clip_norm.unwrap_or(1.0),
            adam_beta1: config.optimizer.adam_beta1.unwrap_or(0.9),
            adam_beta2: config.optimizer.adam_beta2.unwrap_or(0.999),
            adam_epsilon: config.optimizer.adam_epsilon.unwrap_or(1.0e-8),
            per_parameter_grad_normalization: config
                .optimizer
                .per_parameter_grad_normalization
                .unwrap_or(true),
            lr_schedule: lr_schedule_label,
            min_lr_scale,
        },
        validation: E2eRolloutValidationReport {
            examples: config.validation.examples.unwrap_or(16),
            particles: validation_particles,
            steps: validation_steps,
            update_prob: config.validation.update_prob.unwrap_or(0.5),
            seed: config.validation.seed.unwrap_or(42),
            oracle_report: config
                .validation
                .oracle_report
                .as_ref()
                .map(|path| display_path(path)),
            psnr_threshold_db: config.validation.psnr_threshold_db.unwrap_or(26.0),
        },
        blockers,
        selected_sources: source_entries(&sources, &splits),
        training_result: None,
    })
}

fn run_burn_e2e_rollout_training(
    report: &E2eRolloutReport,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    ensure_training_backend_available(report)?;
    check_condition_preload_memory_budget(report)?;
    let load_started = Instant::now();
    let mut examples = load_burn_e2e_rollout_examples(report)?;
    let example_condition_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let (mut train_examples, mut holdout_examples): (Vec<_>, Vec<_>) = examples
        .drain(..)
        .partition(|example| report_source_split(report, &example.slug) == Some("train"));
    if train_examples.is_empty() {
        return Err(std::io::Error::other("HyperNPA e2e rollout has no train examples").into());
    }
    let (npa_config, hashgrid) = NpaConfig::for_preset(report.preset);
    let mut base = if let Some(path) = &report.model.shared_base {
        crate::import::load_manifest(path)?.into_model()
    } else {
        match report.model.shared_base_init.as_str() {
            "upstream-seeded" | "upstream_seeded" => {
                NpaModel::upstream_seeded(npa_config.clone(), report.training.seed)
            }
            "zero" | "zeros" => NpaModel {
                config: npa_config.clone(),
                weights: NpaWeights::zeros(&npa_config),
            },
            _ => NpaModel::upstream_seeded(npa_config.clone(), report.training.seed),
        }
    };
    base.validate()?;
    let loss_config = super::super::target2d::target2d_loss_config(
        report.target.loss_image_size,
        report.target.splat_sigma,
        true,
        report.target.splat_loss_weight,
        report.target.color_loss_weight,
        report.target.density_loss_weight,
        Target2dLossConfig::default().background_density_loss_weight,
        Target2dLossConfig::default().foreground_density_loss_weight,
        report.target.displacement_regularizer_weight,
        report.target.overflow_regularizer_weight,
        report.target.bound_regularizer_weight,
    )?;
    let train_config = BurnE2eRolloutTrainConfig {
        steps: report.training.steps,
        report_interval: report.training.report_interval,
        example_batch_size: report.training.example_batch_size,
        tbptt_chunk_steps: report.training.tbptt_chunk_steps,
        rollout_particles: report.rollout.particles,
        rollout_steps: report.rollout.steps,
        update_prob: report.rollout.update_prob,
        seed: report.training.seed,
        seed_scale: report
            .rollout
            .seed_scale
            .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(report.preset)),
        seed_mode: report.rollout.seed_mode,
        grid_eps: hashgrid.eps,
        motion_scale: npa_config.alpha * npa_config.motion_eps(hashgrid.eps),
        loss_config,
        per_parameter_grad_normalization: report.optimizer.per_parameter_grad_normalization,
        shared_base_trainable: report.model.shared_base_trainable,
        base_optimizer: adamw_from_report(report),
        generator_optimizer: adamw_from_report(report),
        lr_schedule: parse_e2e_lr_schedule(Some(&report.optimizer.lr_schedule))?,
        min_lr_scale: report.optimizer.min_lr_scale,
        adapter_rank: report.adapter.rank,
        adapter_alpha: report.adapter.alpha,
        generator_hidden_dims: report.adapter.flow_hidden,
        token_attention_heads: report.condition.token_attention_heads,
        generator_sample_steps: report.adapter.flow_sample_steps,
        generator_output_scale: report.adapter.flow_source_scale,
        generator_init_scale: report.adapter.init_scale,
        stopgrad_pos: npa_config.stopgrad_pos,
        stopgrad_state: npa_config.stopgrad_state,
        system_memory_budget_gb: report.training.system_memory_budget_gb,
        gpu_memory_budget_gb: report.training.gpu_memory_budget_gb,
        max_dense_train_particles: report.training.max_dense_train_particles,
        max_dense_chunk_floats: report.training.max_dense_chunk_floats,
        max_splat_chunk_floats: report.training.max_splat_chunk_floats,
        validation_examples: report.validation.examples,
        validation_particles: report.validation.particles,
        validation_steps: report.validation.steps,
        validation_update_prob: report.validation.update_prob,
        validation_seed: report.validation.seed,
        validation_psnr_threshold_db: report.validation.psnr_threshold_db,
    };
    let train_started = Instant::now();
    let mut output = match report.training.gpu_backend.as_str() {
        "burn-cuda" | "cuda" => train_e2e_rollout_burn_cuda(
            &mut base,
            &mut train_examples,
            &mut holdout_examples,
            train_config,
        )?,
        _ => train_e2e_rollout_burn_wgpu(
            &mut base,
            &mut train_examples,
            &mut holdout_examples,
            train_config,
        )?,
    };
    let burn_training_ms = train_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(metrics) = output.metrics.as_object_mut() {
        metrics.insert(
            "example_condition_load_ms".to_string(),
            serde_json::json!(example_condition_load_ms),
        );
        metrics.insert(
            "burn_training_ms".to_string(),
            serde_json::json!(burn_training_ms),
        );
        metrics.insert(
            "end_to_end_training_command_ms".to_string(),
            serde_json::json!(example_condition_load_ms + burn_training_ms),
        );
    }
    let manifest = BpkModelManifest::from_model(
        &base,
        hashgrid,
        Some(format!(
            "trained-rust:hyper2d-e2e-rollout:sources={}:steps={}",
            train_examples.len(),
            report.training.steps
        )),
    );
    crate::import::save_manifest(&report.output.shared_base_output, &manifest)?;
    write_pretty_json(Path::new(&report.output.hyper_output), &output.generator)?;
    Ok(output)
}

fn ensure_training_backend_available(
    report: &E2eRolloutReport,
) -> Result<(), Box<dyn std::error::Error>> {
    match report.training.gpu_backend.as_str() {
        "burn-cuda" | "cuda" => {
            #[cfg(feature = "backend_cuda")]
            {
                Ok(())
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(std::io::Error::other(
                    "train-hyper2d-e2e-rollout config requests burn-cuda, but this binary was not built with backend_cuda; rebuild with --no-default-features --features cli,backend_ndarray,backend_cuda,dino",
                )
                .into())
            }
        }
        "burn-wgpu" | "wgpu" => {
            #[cfg(feature = "backend_wgpu")]
            {
                Ok(())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(std::io::Error::other(
                    "train-hyper2d-e2e-rollout config requests burn-wgpu, but this binary was not built with backend_wgpu",
                )
                .into())
            }
        }
        other => Err(std::io::Error::other(format!(
            "unsupported HyperNPA e2e rollout backend {other:?}"
        ))
        .into()),
    }
}

fn check_condition_preload_memory_budget(
    report: &E2eRolloutReport,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(budget_gb) = report.training.system_memory_budget_gb else {
        return Ok(());
    };
    let budget_bytes = (budget_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    let projected_bytes = report
        .condition
        .projected_condition_load_peak_bytes_f32
        .saturating_add(1024 * 1024 * 1024);
    if projected_bytes > budget_bytes {
        return Err(std::io::Error::other(format!(
            "projected HyperNPA e2e condition preload peak is {:.2} GiB plus 1.00 GiB overhead, above system_memory_budget_gb={:.2}; lower source_limit/dino_batch_size or raise the budget intentionally",
            report.condition.projected_condition_load_peak_gib_f32,
            budget_gb,
        ))
        .into());
    }
    Ok(())
}

fn load_burn_e2e_rollout_examples(
    report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    load_burn_e2e_rollout_examples_with_online_dino(report)
}

#[cfg(feature = "dino")]
fn load_burn_e2e_rollout_examples_with_online_dino(
    report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    let dino_model = report.condition.dino_model.as_ref().ok_or_else(|| {
        std::io::Error::other("condition.dino_model is required for e2e DINO training")
    })?;
    let encoder = super::dino::DinoVitsConditionEncoder::load(
        Path::new(dino_model),
        report.condition.dino_image_size,
    )?;
    let expected_feature_dims = report
        .condition
        .token_count
        .saturating_mul(report.condition.embed_dims);
    let batch_size = report.condition.dino_batch_size.max(1);
    let batch_count = report.selected_sources.len().div_ceil(batch_size);
    let mut examples = Vec::with_capacity(report.selected_sources.len());
    eprintln!(
        "encoding {} online DINO conditions in {} batches of up to {} (feature cache {:.2} GiB f32, input batch {:.2} GiB f32)",
        report.selected_sources.len(),
        batch_count,
        batch_size,
        report.condition.selected_feature_cache_gib_f32,
        report.condition.dino_batch_input_gib_f32,
    );
    for (batch_idx, entries) in report.selected_sources.chunks(batch_size).enumerate() {
        let conditions = entries
            .iter()
            .map(|entry| load_condition_image_2d(Path::new(&entry.condition_path)))
            .collect::<Result<Vec<_>, _>>()?;
        let features = encoder.encode_batch(
            &conditions,
            ConditionEncoder2d::DinoVitsTokenGrid,
            report.condition.patch_grid_width,
            report.condition.patch_grid_height,
        )?;
        if features.len() != entries.len() {
            return Err(
                std::io::Error::other("DINO feature count does not match source batch").into(),
            );
        }
        for (entry, condition_features) in entries.iter().zip(features) {
            if condition_features.len() != expected_feature_dims {
                return Err(std::io::Error::other(format!(
                    "DINO feature length for {} is {}; expected {}",
                    entry.slug,
                    condition_features.len(),
                    expected_feature_dims
                ))
                .into());
            }
            let target = super::super::target2d::load_target_image_2d_adaptive(
                Path::new(&entry.condition_path),
                report.target.threshold,
                report.target.points,
                report.target.image_size,
            )?;
            examples.push(BurnE2eRolloutExample {
                slug: entry.slug.clone(),
                target,
                condition_features,
                token_count: report.condition.token_count,
                embed_dims: report.condition.embed_dims,
                particle_count: entry.particles.unwrap_or(report.rollout.particles),
                update_prob: entry.update_prob.unwrap_or(report.rollout.update_prob),
                seed_scale: entry.seed_scale.unwrap_or_else(|| {
                    report
                        .rollout
                        .seed_scale
                        .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(report.preset))
                }),
            });
        }
        if (batch_idx + 1).is_multiple_of(10) || batch_idx + 1 == batch_count {
            eprintln!(
                "encoded online DINO batch {}/{} ({} examples)",
                batch_idx + 1,
                batch_count,
                examples.len()
            );
        }
    }
    Ok(examples)
}

#[cfg(not(feature = "dino"))]
fn load_burn_e2e_rollout_examples_with_online_dino(
    _report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("train-hyper2d-e2e-rollout with steps > 0 requires the dino feature")
            .into(),
    )
}

fn report_source_split<'a>(report: &'a E2eRolloutReport, slug: &str) -> Option<&'a str> {
    report
        .selected_sources
        .iter()
        .find(|entry| entry.slug == slug)
        .map(|entry| entry.split)
}

fn adamw_from_report(report: &E2eRolloutReport) -> AdamWConfig {
    AdamWConfig {
        learning_rate: report.optimizer.learning_rate,
        weight_decay: report.optimizer.weight_decay,
        grad_clip_norm: report.optimizer.grad_clip_norm,
        beta1: report.optimizer.adam_beta1,
        beta2: report.optimizer.adam_beta2,
        epsilon: report.optimizer.adam_epsilon,
    }
}

fn parse_rollout_condition_encoder(
    value: Option<&str>,
) -> Result<RolloutConditionEncoder, Box<dyn std::error::Error>> {
    match value.unwrap_or("dino-vits-full-tokens") {
        "dino-vits-full-tokens" | "dino-full-tokens" | "dino-tokens-full" => {
            Ok(RolloutConditionEncoder::DinoVitsFullTokens)
        }
        "dino-vits-token-grid" | "dino-token-grid" | "dino-tokens" => {
            Ok(RolloutConditionEncoder::DinoVitsTokenGrid)
        }
        other => Err(std::io::Error::other(format!(
            "unknown condition.encoder {other:?}; expected dino-vits-full-tokens or dino-vits-token-grid"
        ))
        .into()),
    }
}

fn parse_e2e_lr_schedule(value: Option<&str>) -> Result<E2eLrSchedule, Box<dyn std::error::Error>> {
    match value.unwrap_or("constant") {
        "constant" | "none" => Ok(E2eLrSchedule::Constant),
        "cosine" | "cosine_decay" | "cosine-decay" => Ok(E2eLrSchedule::Cosine),
        "linear" | "linear_decay" | "linear-decay" => Ok(E2eLrSchedule::Linear),
        other => Err(std::io::Error::other(format!(
            "invalid optimizer.lr_schedule `{other}`; use constant, cosine, or linear"
        ))
        .into()),
    }
}

fn dino_patch_grid(
    image_size: usize,
    patch_size: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    if image_size == 0 || patch_size == 0 {
        return Err(std::io::Error::other(
            "condition.dino_image_size and condition.dino_patch_size must be positive",
        )
        .into());
    }
    if !image_size.is_multiple_of(patch_size) {
        return Err(std::io::Error::other(format!(
            "condition.dino_image_size={image_size} must be divisible by patch size {patch_size}"
        ))
        .into());
    }
    Ok(image_size / patch_size)
}

fn parse_value_enum<T: ValueEnum>(
    field: &str,
    value: Option<&str>,
    fallback: T,
) -> Result<T, Box<dyn std::error::Error>> {
    match value {
        Some(value) => T::from_str(value, true).map_err(|err| {
            std::io::Error::other(format!(
                "invalid {field} `{value}` in HyperNPA e2e rollout config: {err}"
            ))
            .into()
        }),
        None => Ok(fallback),
    }
}

fn parse_value_enum_option<T: ValueEnum>(
    field: &str,
    value: Option<&str>,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    value
        .map(|value| {
            T::from_str(value, true).map_err(|err| {
                std::io::Error::other(format!(
                    "invalid {field} `{value}` in HyperNPA e2e rollout config: {err}"
                ))
            })
        })
        .transpose()
        .map_err(Into::into)
}

fn default_image_extensions() -> Vec<String> {
    ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
        .into_iter()
        .map(ToString::to_string)
        .collect()
}

fn source_entries(
    sources: &[Hyper2dScratchSource],
    splits: &[Hyper2dE2eSplit],
) -> Vec<E2eRolloutSourceEntry> {
    sources
        .iter()
        .zip(splits.iter())
        .map(|(source, split)| E2eRolloutSourceEntry {
            slug: source.slug.clone(),
            title: source.title.clone(),
            group: source.group.clone(),
            split: split.label(),
            condition_path: display_path(&source.condition_path),
            particles: source.particles,
            seed_scale: source.seed_scale,
            update_prob: source.update_prob,
        })
        .collect()
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dino_full_token_dims_match_vits_518() {
        let grid = dino_patch_grid(518, 14).unwrap();
        assert_eq!(grid, 37);
        let token_count = 1 + grid * grid;
        assert_eq!(token_count, 1370);
        assert_eq!(token_count * DINO_VITS_EMBED_DIMS, 526_080);
    }

    #[test]
    fn rollout_config_rejects_adapter_vector_objective() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["target.png"]

            [condition]
            encoder = "dino-vits-full-tokens"

            [training]
            objective = "rectified-flow"
            "#,
        )
        .unwrap();

        let err = match build_e2e_rollout_report(Path::new("inline.toml"), &config) {
            Ok(_) => panic!("adapter-vector objective should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("training.objective"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn verified_rollout_configs_parse() {
        for (name, expected_steps) in [
            ("smoke_lizard_dino_online.toml", 1),
            ("bench_omnisvg_8_b4_p128.toml", 200),
            ("production_omnisvg_10k_rank16_cuda.toml", 3000),
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("configs/verified/2d/hyper_e2e")
                .join(name);
            let text = std::fs::read_to_string(path).unwrap();
            let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
            assert_eq!(
                config.condition.encoder.as_deref(),
                Some("dino-vits-full-tokens")
            );
            assert_eq!(config.training.steps, Some(expected_steps));
            assert_eq!(config.adapter.flow_source_scale, Some(1.0));
            assert_eq!(config.adapter.init_scale, Some(1.0e-3));
        }
    }
}
