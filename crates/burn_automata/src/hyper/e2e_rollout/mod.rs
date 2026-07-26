#[cfg(any(feature = "dino", test))]
use crate::TargetImage2d;
use crate::hyper::NpaParameterRowLayout2d;
use crate::hyper::condition::DINO_VITS_EMBED_DIMS;
use crate::hyper::e2e::{
    DEFAULT_E2E_HYPER_ADAPTER_CHUNK_SIZE, E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK,
    E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL, E2E_HYPER_ADAPTER_FACTORIZED,
    E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW, E2E_HYPER_ARCH_MODULE_TOKEN_DECODER,
    E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2, E2E_HYPER_ARCH_SAMPLE_ID_TABLE,
    E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW, E2E_HYPER_ATTENTION_SOFTMAX, E2E_HYPER_ATTENTION_TANH_EXP,
    E2eHyperGeneratorKind, E2eHyperNpa2d, PerceptionRolloutBackend, Target2dLossBackend,
    save_e2e_hyper_npa_2d,
};
use crate::hyper::e2e_training::dense::{train_e2e_rollout_burn_cuda, train_e2e_rollout_burn_wgpu};
use crate::hyper::e2e_training::{
    BurnE2eRolloutExample, BurnE2eRolloutOutput, BurnE2eRolloutQualityEntry,
    BurnE2eRolloutTrainConfig, E2eAdapterTeacherObjective, E2eCreditAssignment, E2eLrSchedule,
    E2eTbpttLossMode, MAX_VALIDATION_HORIZONS,
};
use crate::{
    AdamWConfig, AutomataPreset, BpkModelManifest, NpaConfig, NpaLowRankAdapter, NpaModel,
    NpaWeights, ParticleSeed, Target2dLossConfig,
};
#[cfg(feature = "dino")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "dino")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};

pub(crate) mod sources;

use sources::{
    Hyper2dScratchSource, OmniSvgSourceConfig, ScratchSourceResolveConfig, resolve_scratch_sources,
};

const DEFAULT_OUTPUT_DIR: &str = "artifacts/hyper2d_e2e_rollout";
const DEFAULT_DINO_IMAGE_SIZE: usize = 224;
const DEFAULT_DINO_PATCH_SIZE: usize = 14;
const DEFAULT_MAX_DENSE_TRAIN_PARTICLES: usize = 512;
const DEFAULT_DEVICE_CONDITION_CACHE_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024;
const DEFAULT_DEVICE_TARGET_CACHE_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024;
const UPSTREAM_GROWING_TRAJECTORIES_PER_TARGET: usize = 10_000 * 3 * 8;
const MAX_STABILITY_EXAMPLES: usize = 16;
const MAX_STABILITY_PARTICLES: usize = 4096;
const MAX_STABILITY_STEPS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Hyper2dE2eSplit {
    Train,
    Holdout,
}

impl Hyper2dE2eSplit {
    fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Holdout => "holdout",
        }
    }

    fn is_train(self) -> bool {
        self == Self::Train
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum E2eCatalogGroup {
    Growing,
    Texture,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum OmniSvgDataset {
    MmsvgIllustration,
    MmsvgIcon,
}

impl OmniSvgDataset {
    fn dataset_id(self) -> &'static str {
        match self {
            Self::MmsvgIllustration => "OmniSVG/MMSVG-Illustration",
            Self::MmsvgIcon => "OmniSVG/MMSVG-Icon",
        }
    }

    fn cache_slug(self) -> &'static str {
        match self {
            Self::MmsvgIllustration => "mmsvg-illustration",
            Self::MmsvgIcon => "mmsvg-icon",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct E2eOmniSvgSourceReport {
    dataset: OmniSvgDataset,
    dataset_id: String,
    split: String,
    cache_dir: String,
    offset: usize,
    limit: usize,
    page_size: usize,
    download: bool,
    refresh: bool,
    token_env: String,
}

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
    gates: RolloutGateConfig,
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
    checkpoint_interval_steps: Option<usize>,
    checkpoint_interval_seconds: Option<usize>,
    resume_checkpoint: Option<PathBuf>,
    auto_resume: Option<bool>,
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
    rgb_channels: Option<bool>,
    rgb_channel_scale: Option<f32>,
    alpha_channel: Option<bool>,
    alpha_channel_scale: Option<f32>,
    patch_pixels: Option<bool>,
    sample_id_vocab_size: Option<usize>,
    sample_ids: Vec<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutModelConfig {
    shared_base: Option<PathBuf>,
    hyper: Option<PathBuf>,
    adapter_bank: Option<PathBuf>,
    shared_base_trainable: Option<bool>,
    shared_base_train_start_step: Option<usize>,
    shared_base_init: Option<String>,
    hidden_dims: Option<usize>,
    output_activation: Option<String>,
    oracle_model_dir: Option<PathBuf>,
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
    loss_on_final_chunk_only: Option<bool>,
    tbptt_loss_mode: Option<String>,
    tbptt_intermediate_loss_weight: Option<f32>,
    tbptt_final_loss_weight: Option<f32>,
    credit_assignment: Option<String>,
    max_full_bptt_particle_steps: Option<usize>,
    use_particle_pool: Option<bool>,
    pool_slots_per_example: Option<usize>,
    rollouts_per_example: Option<usize>,
    target_mean_trajectories_per_example: Option<u64>,
    sampling_uniform_fraction: Option<f32>,
    sampling_priority_ema_beta: Option<f32>,
    sampling_priority_min_weight: Option<f32>,
    sampling_priority_max_weight: Option<f32>,
    sampling_priority_update_interval: Option<usize>,
    pool_capacity: Option<usize>,
    inject_seed_interval: Option<usize>,
    seed_replacements_per_interval: Option<usize>,
    seed_trajectory_interval: Option<usize>,
    brush_size: Option<f32>,
    pre_rollout_step_min: Option<usize>,
    pre_rollout_steps: Option<usize>,
    target2d_loss_backend: Option<String>,
    perception_backend: Option<String>,
    max_dense_train_particles: Option<usize>,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    seed: Option<u64>,
    adapter_teacher_weight: Option<f32>,
    adapter_teacher_objective: Option<String>,
    adapter_teacher_probe_rollout_steps: Option<usize>,
    task_loss_weight: Option<f32>,
    flow_matching_weight: Option<f32>,
    flow_match_inference_source: Option<bool>,
    flow_train_sample_steps: Option<usize>,
    flow_self_rectification_weight: Option<f32>,
    curriculum_resume: Option<bool>,
    amortization_enabled: Option<bool>,
    amortization_substrate_steps: Option<usize>,
    amortization_residual_scale: Option<f32>,
    amortization_residual_anneal_steps: Option<usize>,
    amortization_hyper_only_fraction: Option<f32>,
    amortization_distillation_weight: Option<f32>,
    amortization_distillation_objective: Option<String>,
    amortization_distillation_probe_rollout_steps: Option<usize>,
    amortization_initialize_from_teacher: Option<bool>,
    amortization_initialize_from_generator: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutGpuConfig {
    backend: Option<String>,
    max_dense_chunk_floats: Option<usize>,
    max_splat_chunk_floats: Option<usize>,
    condition_device_cache_max_bytes: Option<usize>,
    target_device_cache_max_bytes: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutAdapterConfig {
    rank: Option<usize>,
    alpha: Option<f32>,
    generator: Option<String>,
    flow_hidden: Option<usize>,
    flow_layers: Option<usize>,
    flow_ffn_dims: Option<usize>,
    flow_sample_steps: Option<usize>,
    flow_source_seed: Option<u64>,
    flow_source_scale: Option<f32>,
    flow_default_endpoint_rms: Option<f32>,
    init_scale: Option<f32>,
    condition_init_scale: Option<f32>,
    output_init_scale: Option<f32>,
    adapter_chunk_size: Option<usize>,
    attention_normalization: Option<String>,
    parameterization: Option<String>,
    spatial_condition_control: Option<bool>,
    spatial_condition_control_scale: Option<f32>,
    spatial_condition_control_sigma: Option<f32>,
    spatial_condition_state_control: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutRuntimeConfig {
    particles: Option<usize>,
    step_min: Option<usize>,
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
    composited_rgb_loss_weight: Option<f32>,
    displacement_regularizer_weight: Option<f32>,
    overflow_regularizer_weight: Option<f32>,
    bound_regularizer_weight: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutOptimizerConfig {
    learning_rate: Option<f32>,
    base_learning_rate: Option<f32>,
    generator_learning_rate: Option<f32>,
    amortization_learning_rate: Option<f32>,
    weight_decay: Option<f32>,
    base_weight_decay: Option<f32>,
    generator_weight_decay: Option<f32>,
    grad_clip_norm: Option<f32>,
    base_grad_clip_norm: Option<f32>,
    generator_grad_clip_norm: Option<f32>,
    adam_beta1: Option<f32>,
    adam_beta2: Option<f32>,
    adam_epsilon: Option<f32>,
    per_parameter_grad_normalization: Option<bool>,
    base_per_parameter_grad_normalization: Option<bool>,
    generator_per_parameter_grad_normalization: Option<bool>,
    amortization_grad_normalization: Option<bool>,
    lr_schedule: Option<String>,
    warmup_steps: Option<usize>,
    min_lr_scale: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutValidationConfig {
    split: Option<String>,
    initial_examples: Option<usize>,
    examples: Option<usize>,
    interval: Option<usize>,
    particles: Option<usize>,
    steps: Option<usize>,
    horizons: Option<Vec<usize>>,
    selection_horizon_min_steps: Option<usize>,
    update_prob: Option<f32>,
    seed: Option<u64>,
    oracle_report: Option<PathBuf>,
    psnr_threshold_db: Option<f32>,
    final_examples: Option<usize>,
    final_particles: Option<usize>,
    final_steps: Option<usize>,
    final_horizons: Option<Vec<usize>>,
    final_selection_horizon_min_steps: Option<usize>,
    stability_examples: Option<usize>,
    stability_particles: Option<usize>,
    stability_reference_steps: Option<usize>,
    stability_steps: Option<usize>,
    stability_tail_steps: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RolloutGateConfig {
    min_median_particle_steps_per_sec: Option<f64>,
    max_quality_validation_evaluations: Option<usize>,
    max_quality_validation_elapsed_fraction: Option<f64>,
    min_final_mean_render_rgb_psnr_db: Option<f32>,
    min_final_p10_composited_rgb_psnr_db: Option<f32>,
    min_final_condition_shuffle_composited_psnr_gap_db: Option<f32>,
    min_final_generated_adapter_composited_psnr_gain_db: Option<f32>,
    max_final_p90_gap_to_matched_oracle_db: Option<f32>,
    require_validation_interval_at_least_report_interval: Option<bool>,
    fail_on_violation: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum RolloutConditionEncoder {
    DinoVitsFullTokens,
    DinoVitsTokenGrid,
    SampleIdOneHot,
}

impl RolloutConditionEncoder {
    const fn label(self) -> &'static str {
        match self {
            Self::DinoVitsFullTokens => "dino-vits-full-tokens",
            Self::DinoVitsTokenGrid => "dino-vits-token-grid",
            Self::SampleIdOneHot => "sample-id-onehot",
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
    warnings: Vec<String>,
    output: E2eRolloutOutputReport,
    condition: E2eRolloutConditionReport,
    model: E2eRolloutModelReport,
    training: E2eRolloutTrainingReport,
    adapter: E2eRolloutAdapterReport,
    rollout: E2eRolloutRuntimeReport,
    target: E2eRolloutTargetReport,
    optimizer: E2eRolloutOptimizerReport,
    validation: E2eRolloutValidationReport,
    gates: E2eRolloutGateReport,
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
    omnisvg: Option<E2eOmniSvgSourceReport>,
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
    checkpoint_interval_steps: usize,
    checkpoint_interval_seconds: usize,
    resume_checkpoint: Option<String>,
    auto_resume: bool,
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
    device_cache_max_bytes: usize,
    device_cache_max_gib: f64,
    device_cache_plan: &'static str,
    token_attention_heads: usize,
    feature_normalization: &'static str,
    rgb_channels: bool,
    rgb_channel_scale: f32,
    alpha_channel: bool,
    alpha_channel_scale: f32,
    patch_pixels: bool,
    sample_id_vocab_size: Option<usize>,
    sample_ids: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutModelReport {
    shared_base: Option<String>,
    hyper: Option<String>,
    adapter_bank: Option<String>,
    shared_base_trainable: bool,
    shared_base_train_start_step: usize,
    shared_base_init: String,
    hidden_dims: usize,
    output_activation: String,
    oracle_model_dir: Option<String>,
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
    tbptt_state_detach_active: bool,
    effective_tbptt_chunk_steps: Option<usize>,
    rollout_gradient_horizon_max_steps: Option<usize>,
    loss_on_final_chunk_only: bool,
    tbptt_loss_mode: String,
    tbptt_intermediate_loss_weight: f32,
    tbptt_final_loss_weight: f32,
    credit_assignment: String,
    max_full_bptt_particle_steps: usize,
    use_particle_pool: bool,
    pool_slots_per_example: usize,
    rollouts_per_example: usize,
    target_mean_trajectories_per_example: Option<u64>,
    sampling_uniform_fraction: f32,
    sampling_priority_ema_beta: f32,
    sampling_priority_min_weight: f32,
    sampling_priority_max_weight: f32,
    sampling_priority_update_interval: usize,
    planned_rollout_trajectories: usize,
    planned_mean_trajectories_per_train_example: f64,
    upstream_growing_reference_trajectories_per_target: usize,
    upstream_growing_trajectory_exposure_fraction: f64,
    pool_capacity: usize,
    inject_seed_interval: usize,
    seed_replacements_per_interval: usize,
    seed_trajectory_interval: usize,
    brush_size: f32,
    pre_rollout_step_min: usize,
    pre_rollout_steps: usize,
    target2d_loss_backend: String,
    perception_backend: String,
    max_dense_train_particles: usize,
    system_memory_budget_gb: Option<f32>,
    gpu_memory_budget_gb: Option<f32>,
    max_dense_chunk_floats: usize,
    max_splat_chunk_floats: usize,
    condition_device_cache_max_bytes: usize,
    target_device_cache_max_bytes: usize,
    seed: u64,
    adapter_teacher_weight: f32,
    adapter_teacher_objective: String,
    adapter_teacher_probe_rollout_steps: usize,
    task_loss_weight: f32,
    flow_matching_weight: f32,
    flow_match_inference_source: bool,
    flow_train_sample_steps: usize,
    flow_self_rectification_weight: f32,
    curriculum_resume: bool,
    amortization_enabled: bool,
    amortization_substrate_steps: usize,
    amortization_residual_scale: f32,
    amortization_residual_anneal_steps: usize,
    amortization_hyper_only_fraction: f32,
    amortization_distillation_weight: f32,
    amortization_distillation_objective: String,
    amortization_distillation_probe_rollout_steps: usize,
    amortization_initialize_from_teacher: bool,
    amortization_initialize_from_generator: bool,
    amortization_parameter_count: usize,
    amortization_optimizer_bytes_f32: usize,
    trains_shared_base_from_step_zero: bool,
    trains_hypernet_from_step_zero: bool,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutAdapterReport {
    rank: usize,
    alpha: f32,
    output_bias: bool,
    generator: String,
    flow_hidden: usize,
    flow_layers: usize,
    flow_ffn_dims: usize,
    flow_sample_steps: usize,
    flow_source_seed: u64,
    flow_source_scale: f32,
    flow_default_endpoint_rms: f32,
    init_scale: f32,
    condition_init_scale: f32,
    output_init_scale: f32,
    adapter_chunk_size: usize,
    attention_normalization: String,
    parameterization: String,
    spatial_condition_control: bool,
    spatial_condition_control_scale: f32,
    spatial_condition_control_sigma: f32,
    spatial_condition_state_control: bool,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutRuntimeReport {
    particles: usize,
    step_min: usize,
    steps: usize,
    sampled_training_steps: bool,
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
    composited_rgb_loss_weight: f32,
    displacement_regularizer_weight: f32,
    overflow_regularizer_weight: f32,
    bound_regularizer_weight: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutOptimizerReport {
    learning_rate: f32,
    base_learning_rate: f32,
    generator_learning_rate: f32,
    amortization_learning_rate: f32,
    weight_decay: f32,
    base_weight_decay: f32,
    generator_weight_decay: f32,
    grad_clip_norm: f32,
    base_grad_clip_norm: f32,
    generator_grad_clip_norm: f32,
    adam_beta1: f32,
    adam_beta2: f32,
    adam_epsilon: f32,
    per_parameter_grad_normalization: bool,
    base_per_parameter_grad_normalization: bool,
    generator_per_parameter_grad_normalization: bool,
    amortization_grad_normalization: bool,
    lr_schedule: String,
    warmup_steps: usize,
    min_lr_scale: f32,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutValidationReport {
    split: String,
    initial_examples: usize,
    examples: usize,
    interval: usize,
    particles: usize,
    steps: usize,
    horizons: Vec<usize>,
    selection_horizon_min_steps: usize,
    quality_scale: bool,
    training_backward_safe: bool,
    update_prob: f32,
    seed: u64,
    oracle_report: Option<String>,
    psnr_threshold_db: f32,
    final_examples: usize,
    final_particles: usize,
    final_steps: usize,
    final_horizons: Vec<usize>,
    final_selection_horizon_min_steps: usize,
    stability_examples: usize,
    stability_particles: usize,
    stability_reference_steps: usize,
    stability_steps: usize,
    stability_tail_steps: usize,
    stability_evaluation_mode: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutGateReport {
    min_median_particle_steps_per_sec: Option<f64>,
    max_quality_validation_evaluations: Option<usize>,
    max_quality_validation_elapsed_fraction: Option<f64>,
    min_final_mean_render_rgb_psnr_db: Option<f32>,
    min_final_p10_composited_rgb_psnr_db: Option<f32>,
    min_final_condition_shuffle_composited_psnr_gap_db: Option<f32>,
    min_final_generated_adapter_composited_psnr_gain_db: Option<f32>,
    max_final_p90_gap_to_matched_oracle_db: Option<f32>,
    require_validation_interval_at_least_report_interval: bool,
    fail_on_violation: bool,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutGateResultReport {
    gate: &'static str,
    passed: bool,
    observed: serde_json::Value,
    threshold: serde_json::Value,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
struct E2eRolloutTrainingResultReport {
    backend: String,
    device: String,
    final_loss: Option<f32>,
    history: serde_json::Value,
    metrics: serde_json::Value,
    quality_validation: Option<serde_json::Value>,
    amortization_quality_validation: Option<serde_json::Value>,
    stability_validation: Option<serde_json::Value>,
    gates_passed: bool,
    gates: Vec<E2eRolloutGateResultReport>,
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

pub fn run_train_hyper_2d_e2e_rollout_config_path(
    config: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = config.as_ref();
    let experiment_config = load_rollout_experiment_config(config)?;
    let mut report = build_e2e_rollout_report(config, &experiment_config)?;
    write_pretty_json(Path::new(&report.output.report_output), &report)?;

    if report.training.steps > 0 {
        let training = run_burn_e2e_rollout_training(&report)?;
        let gate_results = evaluate_e2e_rollout_gates(&report, &training);
        let gates_passed = gate_results.iter().all(|gate| gate.passed);
        let should_fail_on_gate_violation = report.gates.fail_on_violation && !gates_passed;
        let shared_base_output = report.output.shared_base_output.clone();
        let hyper_output = report.output.hyper_output.clone();
        let quality_validation = training
            .quality_validation
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;
        let amortization_quality_validation = training
            .amortization_quality_validation
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;
        let stability_validation = training
            .stability_validation
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
            amortization_quality_validation,
            stability_validation,
            gates_passed,
            gates: gate_results,
            shared_base_output,
            hyper_output,
        });
        write_pretty_json(Path::new(&report.output.report_output), &report)?;
        if should_fail_on_gate_violation {
            return Err(std::io::Error::other(format!(
                "HyperNPA e2e rollout gates failed; see {}",
                report.output.report_output
            ))
            .into());
        }
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
    let preset = parse_preset(config.preset.as_deref())?;
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
        .unwrap_or_else(|| output_dir.join("hyper_2d.bpk"));
    if has_json_extension(&hyper_output) {
        return Err(std::io::Error::other(format!(
            "trained E2E HyperNPA artifacts are binary; use a .bpk hyper_output path instead of {}",
            hyper_output.display()
        ))
        .into());
    }
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
        RolloutConditionEncoder::SampleIdOneHot => (1, 1),
    };
    if !matches!(encoder, RolloutConditionEncoder::SampleIdOneHot) {
        if token_grid_width == 0 || token_grid_height == 0 {
            return Err(
                std::io::Error::other("DINO token grid dimensions must be positive").into(),
            );
        }
        if token_grid_width > patch_grid || token_grid_height > patch_grid {
            return Err(std::io::Error::other(format!(
                "DINO token grid {token_grid_width}x{token_grid_height} exceeds full patch grid {patch_grid}x{patch_grid}"
            ))
            .into());
        }
    }
    let online_dino = match encoder {
        RolloutConditionEncoder::SampleIdOneHot => false,
        RolloutConditionEncoder::DinoVitsFullTokens
        | RolloutConditionEncoder::DinoVitsTokenGrid => config.condition.online.unwrap_or(true),
    };
    if !online_dino && !matches!(encoder, RolloutConditionEncoder::SampleIdOneHot) {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e-rollout requires condition.online = true for DINO encoders; cached feature-vector regression belongs in train-hyper2d-adapter-bank",
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
        "target2d-rollout-image-loss"
            | "rollout-image-loss"
            | "e2e-rollout-loss"
            | "conditional-row-flow-e2e"
            | "conditional-row-flow-matching"
    ) {
        return Err(std::io::Error::other(format!(
            "training.objective {objective:?} is not valid for train-hyper2d-e2e-rollout"
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
    let catalog_group = parse_catalog_group(
        "source.catalog_group",
        config.source.catalog_group.as_deref(),
    )?;
    let catalog_targets = config.source.catalog_targets.clone().unwrap_or_default();
    let catalog_thumbnail_dir = config
        .source
        .catalog_thumbnail_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/catalog_thumbnails"));
    let omnisvg_dataset = parse_omnisvg_dataset(
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
        preset,
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
    let alpha_channel = config
        .condition
        .alpha_channel
        .unwrap_or(!matches!(encoder, RolloutConditionEncoder::SampleIdOneHot));
    let rgb_channels = config.condition.rgb_channels.unwrap_or(false);
    let rgb_channel_scale = config.condition.rgb_channel_scale.unwrap_or(1.0);
    if !rgb_channel_scale.is_finite() || rgb_channel_scale <= 0.0 {
        return Err(std::io::Error::other(
            "condition.rgb_channel_scale must be positive and finite",
        )
        .into());
    }
    let alpha_channel_scale = config.condition.alpha_channel_scale.unwrap_or(1.0);
    if !alpha_channel_scale.is_finite() || alpha_channel_scale <= 0.0 {
        return Err(std::io::Error::other(
            "condition.alpha_channel_scale must be positive and finite",
        )
        .into());
    }
    let patch_pixels = config.condition.patch_pixels.unwrap_or(false);
    if patch_pixels {
        if matches!(encoder, RolloutConditionEncoder::SampleIdOneHot) {
            return Err(std::io::Error::other(
                "condition.patch_pixels requires a DINO token-grid encoder",
            )
            .into());
        }
        if token_grid_width != patch_grid || token_grid_height != patch_grid {
            return Err(std::io::Error::other(format!(
                "condition.patch_pixels requires the native DINO patch grid {patch_grid}x{patch_grid}, got {token_grid_width}x{token_grid_height}"
            ))
            .into());
        }
        if !rgb_channels && !alpha_channel {
            return Err(std::io::Error::other(
                "condition.patch_pixels requires rgb_channels and/or alpha_channel",
            )
            .into());
        }
    }
    let (sample_id_vocab_size, sample_ids) = if encoder == RolloutConditionEncoder::SampleIdOneHot {
        let vocab_size = config
            .condition
            .sample_id_vocab_size
            .unwrap_or_else(|| sources.len().max(1));
        if vocab_size == 0 {
            return Err(
                std::io::Error::other("condition.sample_id_vocab_size must be positive").into(),
            );
        }
        let ids = if config.condition.sample_ids.is_empty() {
            (0..sources.len()).collect::<Vec<_>>()
        } else {
            if config.condition.sample_ids.len() != sources.len() {
                return Err(std::io::Error::other(format!(
                    "condition.sample_ids has {} entries but the resolved source set has {}",
                    config.condition.sample_ids.len(),
                    sources.len()
                ))
                .into());
            }
            config.condition.sample_ids.clone()
        };
        if let Some(id) = ids.iter().copied().find(|id| *id >= vocab_size) {
            return Err(std::io::Error::other(format!(
                "condition sample ID {id} is outside sample_id_vocab_size={vocab_size}"
            ))
            .into());
        }
        if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(std::io::Error::other(
                "condition.sample_ids must contain unique adapter-table columns",
            )
            .into());
        }
        (Some(vocab_size), ids)
    } else {
        if config.condition.sample_id_vocab_size.is_some()
            || !config.condition.sample_ids.is_empty()
        {
            return Err(std::io::Error::other(
                "condition.sample_id_vocab_size/sample_ids require encoder=sample-id-onehot",
            )
            .into());
        }
        (None, Vec::new())
    };
    let patch_pixel_dims = if patch_pixels {
        let pixels_per_patch = (dino_image_size / token_grid_width)
            .saturating_mul(dino_image_size / token_grid_height);
        pixels_per_patch.saturating_mul(
            3usize.saturating_mul(usize::from(rgb_channels)) + usize::from(alpha_channel),
        )
    } else {
        3 * usize::from(rgb_channels) + usize::from(alpha_channel)
    };
    let (token_count, embed_dims) = match encoder {
        RolloutConditionEncoder::DinoVitsFullTokens
        | RolloutConditionEncoder::DinoVitsTokenGrid => (
            1 + token_grid_width * token_grid_height,
            DINO_VITS_EMBED_DIMS + patch_pixel_dims,
        ),
        RolloutConditionEncoder::SampleIdOneHot => (
            1,
            sample_id_vocab_size.expect("sample-ID encoder has a vocabulary"),
        ),
    };
    let flattened_feature_dims = token_count * embed_dims;

    let mut steps = config.training.steps.unwrap_or(0);
    let report_interval = config.training.report_interval.unwrap_or(25).max(1);
    let example_batch_size = config.training.example_batch_size.unwrap_or(1).max(1);
    let tbptt_chunk_steps = config.training.tbptt_chunk_steps.unwrap_or(8).max(1);
    let loss_on_final_chunk_only = config.training.loss_on_final_chunk_only.unwrap_or(false);
    let tbptt_loss_mode = parse_e2e_tbptt_loss_mode(
        config.training.tbptt_loss_mode.as_deref(),
        loss_on_final_chunk_only,
    )?;
    let tbptt_intermediate_loss_weight = config
        .training
        .tbptt_intermediate_loss_weight
        .unwrap_or(0.25)
        .max(0.0);
    let tbptt_final_loss_weight = config
        .training
        .tbptt_final_loss_weight
        .unwrap_or(1.0)
        .max(0.0);
    let credit_assignment =
        parse_e2e_credit_assignment(config.training.credit_assignment.as_deref())?;
    let adapter_teacher_weight = config.training.adapter_teacher_weight.unwrap_or(0.0);
    let adapter_teacher_objective =
        parse_e2e_adapter_teacher_objective(config.training.adapter_teacher_objective.as_deref())?;
    let task_loss_weight = config.training.task_loss_weight.unwrap_or(1.0);
    let flow_matching_weight = config.training.flow_matching_weight.unwrap_or(0.0);
    let flow_match_inference_source = config.training.flow_match_inference_source.unwrap_or(false);
    let flow_self_rectification_weight = config
        .training
        .flow_self_rectification_weight
        .unwrap_or(0.0);
    let curriculum_resume = config.training.curriculum_resume.unwrap_or(false);
    let amortization_enabled = config.training.amortization_enabled.unwrap_or(false);
    let amortization_substrate_steps = config
        .training
        .amortization_substrate_steps
        .unwrap_or(0)
        .min(steps);
    let amortization_residual_scale = config.training.amortization_residual_scale.unwrap_or(1.0);
    let amortization_residual_anneal_steps = config
        .training
        .amortization_residual_anneal_steps
        .unwrap_or(steps.max(1));
    let amortization_hyper_only_fraction = config
        .training
        .amortization_hyper_only_fraction
        .unwrap_or(0.25);
    let amortization_distillation_weight = config
        .training
        .amortization_distillation_weight
        .unwrap_or(0.1);
    let amortization_distillation_objective = parse_e2e_adapter_teacher_objective(
        config
            .training
            .amortization_distillation_objective
            .as_deref(),
    )?;
    let amortization_distillation_probe_rollout_steps = config
        .training
        .amortization_distillation_probe_rollout_steps
        .unwrap_or(0);
    let amortization_initialize_from_teacher = config
        .training
        .amortization_initialize_from_teacher
        .unwrap_or(false);
    let amortization_initialize_from_generator = config
        .training
        .amortization_initialize_from_generator
        .unwrap_or(false);
    let amortization_flow_supervision = amortization_enabled
        && (amortization_distillation_weight > 0.0 || flow_self_rectification_weight > 0.0);
    if objective == "conditional-row-flow-matching"
        && (task_loss_weight != 0.0
            || (flow_matching_weight <= 0.0
                && adapter_teacher_weight <= 0.0
                && !amortization_flow_supervision))
    {
        return Err(std::io::Error::other(
            "conditional-row-flow-matching requires task_loss_weight=0 and positive flow-matching, adapter-endpoint, or amortization-table supervision",
        )
        .into());
    }
    if objective == "conditional-row-flow-e2e" && task_loss_weight <= 0.0 {
        return Err(std::io::Error::other(
            "conditional-row-flow-e2e requires a positive task_loss_weight",
        )
        .into());
    }
    if !adapter_teacher_weight.is_finite()
        || adapter_teacher_weight < 0.0
        || !task_loss_weight.is_finite()
        || task_loss_weight < 0.0
        || !flow_matching_weight.is_finite()
        || flow_matching_weight < 0.0
        || !flow_self_rectification_weight.is_finite()
        || flow_self_rectification_weight < 0.0
        || !amortization_residual_scale.is_finite()
        || !(0.0..=1.0).contains(&amortization_residual_scale)
        || !amortization_hyper_only_fraction.is_finite()
        || !(0.0..=1.0).contains(&amortization_hyper_only_fraction)
        || !amortization_distillation_weight.is_finite()
        || amortization_distillation_weight < 0.0
    {
        return Err(std::io::Error::other(
            "training task/adapter-teacher/flow/amortization weights must be finite and non-negative, and amortization_residual_scale/amortization_hyper_only_fraction must be in 0..=1",
        )
        .into());
    }
    let detached_substrate_only = amortization_enabled
        && amortization_substrate_steps == steps
        && credit_assignment == E2eCreditAssignment::DetachedTbptt;
    let detached_behavioral_amortization = amortization_enabled
        && objective == "conditional-row-flow-e2e"
        && task_loss_weight > 0.0
        && credit_assignment == E2eCreditAssignment::DetachedTbptt;
    if amortization_enabled
        && credit_assignment != E2eCreditAssignment::FullBptt
        && !detached_substrate_only
        && !detached_behavioral_amortization
    {
        return Err(std::io::Error::other(
            "training amortization requires full-bptt, detached behavioral conditional-row-flow-e2e training, or an endpoint-only detached substrate run",
        )
        .into());
    }
    if amortization_enabled
        && task_loss_weight <= 0.0
        && !(objective == "conditional-row-flow-matching" && amortization_flow_supervision)
    {
        return Err(std::io::Error::other(
            "training amortization without task loss requires conditional-row-flow-matching and positive amortization distillation or self-rectification supervision",
        )
        .into());
    }
    if amortization_substrate_steps > 0 && !amortization_enabled {
        return Err(std::io::Error::other(
            "training.amortization_substrate_steps requires amortization_enabled=true",
        )
        .into());
    }
    let has_adapter_endpoints =
        config.model.oracle_model_dir.is_some() || config.model.adapter_bank.is_some();
    if objective == "conditional-row-flow-matching"
        && amortization_flow_supervision
        && config.output.resume_checkpoint.is_none()
        && !config.output.auto_resume.unwrap_or(false)
        && !amortization_initialize_from_teacher
        && !amortization_initialize_from_generator
    {
        return Err(std::io::Error::other(
            "rollout-free amortization flow supervision requires output.resume_checkpoint, output.auto_resume=true, or explicit teacher/generator endpoint initialization",
        )
        .into());
    }
    if config.model.oracle_model_dir.is_some() && config.model.adapter_bank.is_some() {
        return Err(std::io::Error::other(
            "model.oracle_model_dir and model.adapter_bank are mutually exclusive endpoint sources",
        )
        .into());
    }
    if amortization_initialize_from_teacher && amortization_initialize_from_generator {
        return Err(std::io::Error::other(
            "training amortization initialization must choose either teacher or generator endpoints",
        )
        .into());
    }
    if amortization_initialize_from_teacher && (!amortization_enabled || !has_adapter_endpoints) {
        return Err(std::io::Error::other(
            "training.amortization_initialize_from_teacher requires amortization_enabled=true and model.oracle_model_dir or model.adapter_bank",
        )
        .into());
    }
    if amortization_initialize_from_generator && !amortization_enabled {
        return Err(std::io::Error::other(
            "training.amortization_initialize_from_generator requires amortization_enabled=true",
        )
        .into());
    }
    if adapter_teacher_weight > 0.0 && !has_adapter_endpoints {
        return Err(std::io::Error::other(
            "training.adapter_teacher_weight requires model.oracle_model_dir or model.adapter_bank",
        )
        .into());
    }
    if (flow_matching_weight > 0.0 || adapter_teacher_weight > 0.0)
        && config.model.shared_base_trainable.unwrap_or(true)
    {
        return Err(std::io::Error::other(
            "adapter endpoint targets are defined relative to a fixed shared base; set model.shared_base_trainable=false while flow or adapter-endpoint supervision is non-zero",
        )
        .into());
    }
    let max_full_bptt_particle_steps = config
        .training
        .max_full_bptt_particle_steps
        .unwrap_or(1_048_576)
        .max(1);
    if tbptt_loss_mode == E2eTbpttLossMode::EndpointWeighted
        && tbptt_intermediate_loss_weight == 0.0
        && tbptt_final_loss_weight == 0.0
    {
        return Err(std::io::Error::other(
            "training.tbptt_loss_mode endpoint-weighted requires a non-zero intermediate or final loss weight",
        )
        .into());
    }
    let use_particle_pool = config.training.use_particle_pool.unwrap_or(false);
    let pool_slots_per_example = config.training.pool_slots_per_example.unwrap_or(1).max(1);
    let rollouts_per_example = config.training.rollouts_per_example.unwrap_or(1).max(1);
    let sampling_uniform_fraction = config.training.sampling_uniform_fraction.unwrap_or(0.75);
    if !sampling_uniform_fraction.is_finite() || !(0.0..=1.0).contains(&sampling_uniform_fraction) {
        return Err(std::io::Error::other(
            "training.sampling_uniform_fraction must be finite and in 0..=1",
        )
        .into());
    }
    let sampling_priority_ema_beta = config.training.sampling_priority_ema_beta.unwrap_or(0.95);
    if !sampling_priority_ema_beta.is_finite() || !(0.0..1.0).contains(&sampling_priority_ema_beta)
    {
        return Err(std::io::Error::other(
            "training.sampling_priority_ema_beta must be finite and in 0..1",
        )
        .into());
    }
    let sampling_priority_min_weight = config.training.sampling_priority_min_weight.unwrap_or(0.5);
    let sampling_priority_max_weight = config.training.sampling_priority_max_weight.unwrap_or(4.0);
    if !sampling_priority_min_weight.is_finite()
        || !sampling_priority_max_weight.is_finite()
        || sampling_priority_min_weight <= 0.0
        || sampling_priority_max_weight < sampling_priority_min_weight
    {
        return Err(std::io::Error::other(
            "training priority weights must be finite, positive, and max >= min",
        )
        .into());
    }
    let sampling_priority_update_interval = config
        .training
        .sampling_priority_update_interval
        .unwrap_or(32)
        .max(1);
    let pool_capacity = config.training.pool_capacity.unwrap_or(2048).max(1);
    let rollout_replicas = rollouts_per_example;
    let effective_rollout_batch_size = example_batch_size.saturating_mul(rollout_replicas);
    let target_mean_trajectories_per_example = config.training.target_mean_trajectories_per_example;
    if let Some(target_exposure) = target_mean_trajectories_per_example {
        if config.training.steps.is_some() {
            return Err(std::io::Error::other(
                "configure either training.steps or training.target_mean_trajectories_per_example, not both",
            )
            .into());
        }
        steps = optimizer_steps_for_exposure(
            target_exposure,
            train_examples,
            effective_rollout_batch_size,
        )?;
    }
    let planned_rollout_trajectories = steps.saturating_mul(effective_rollout_batch_size);
    let planned_mean_trajectories_per_train_example =
        planned_rollout_trajectories as f64 / train_examples.max(1) as f64;
    let upstream_growing_trajectory_exposure_fraction = planned_mean_trajectories_per_train_example
        / UPSTREAM_GROWING_TRAJECTORIES_PER_TARGET as f64;
    if use_particle_pool && pool_capacity < effective_rollout_batch_size {
        return Err(std::io::Error::other(format!(
            "training.pool_capacity={pool_capacity} must be at least example_batch_size*rollouts_per_example={effective_rollout_batch_size}"
        ))
        .into());
    }
    if use_particle_pool && pool_slots_per_example < rollouts_per_example {
        return Err(std::io::Error::other(format!(
            "training.pool_slots_per_example={pool_slots_per_example} must be at least rollouts_per_example={rollouts_per_example} so every trajectory has persistent identity-local state"
        ))
        .into());
    }
    let inject_seed_interval = config.training.inject_seed_interval.unwrap_or(64).max(1);
    let seed_replacements_per_interval = config
        .training
        .seed_replacements_per_interval
        .unwrap_or(1)
        .min(effective_rollout_batch_size);
    let seed_trajectory_interval = config
        .training
        .seed_trajectory_interval
        .unwrap_or(128)
        .max(1);
    let brush_size = config.training.brush_size.unwrap_or(0.0).max(0.0);
    let pre_rollout_steps = config.training.pre_rollout_steps.unwrap_or(0);
    let pre_rollout_step_min = config
        .training
        .pre_rollout_step_min
        .unwrap_or(pre_rollout_steps);
    if pre_rollout_step_min > pre_rollout_steps {
        return Err(std::io::Error::other(format!(
            "training.pre_rollout_step_min ({pre_rollout_step_min}) exceeds pre_rollout_steps ({pre_rollout_steps})"
        ))
        .into());
    }
    let max_dense_train_particles = config
        .training
        .max_dense_train_particles
        .unwrap_or(DEFAULT_MAX_DENSE_TRAIN_PARTICLES);
    let rollout_particles = config.rollout.particles.unwrap_or(512).max(1);
    let rollout_steps = config.rollout.steps.unwrap_or(32).max(1);
    let rollout_step_min = config.rollout.step_min.unwrap_or(rollout_steps).max(1);
    if rollout_step_min > rollout_steps {
        return Err(std::io::Error::other(format!(
            "rollout.step_min={rollout_step_min} must be <= rollout.steps={rollout_steps}"
        ))
        .into());
    }
    let configured_full_bptt_particle_steps = effective_rollout_batch_size
        .saturating_mul(rollout_particles)
        .saturating_mul(rollout_steps);
    if steps > 0
        && credit_assignment == E2eCreditAssignment::FullBptt
        && configured_full_bptt_particle_steps > max_full_bptt_particle_steps
    {
        return Err(std::io::Error::other(format!(
            "full-bptt preflight rejected batch*particles*steps={configured_full_bptt_particle_steps}, above training.max_full_bptt_particle_steps={max_full_bptt_particle_steps}; lower batch/particles/horizon or raise the cap only after profiling memory"
        ))
        .into());
    }
    let configured_tbptt_chunk_particle_steps = effective_rollout_batch_size
        .saturating_mul(rollout_particles)
        .saturating_mul(tbptt_chunk_steps.min(rollout_steps));
    if steps > 0
        && credit_assignment == E2eCreditAssignment::DetachedTbptt
        && configured_tbptt_chunk_particle_steps > max_full_bptt_particle_steps
    {
        return Err(std::io::Error::other(format!(
            "detached-tbptt preflight rejected batch*particles*chunk_steps={configured_tbptt_chunk_particle_steps}, above training.max_full_bptt_particle_steps={max_full_bptt_particle_steps}; lower batch/particles/chunk size or raise the cap only after profiling memory"
        ))
        .into());
    }
    let sampled_training_steps = rollout_step_min != rollout_steps;
    let effective_tbptt_chunk_steps = credit_assignment.effective_tbptt_chunk_steps(
        task_loss_weight,
        tbptt_chunk_steps,
        rollout_steps,
    );
    let rollout_gradient_horizon_max_steps = credit_assignment.rollout_gradient_horizon_max_steps(
        task_loss_weight,
        tbptt_chunk_steps,
        rollout_step_min,
        rollout_steps,
    );
    if steps > 0 && rollout_particles > max_dense_train_particles {
        return Err(std::io::Error::other(format!(
            "rollout.particles={rollout_particles} exceeds training.max_dense_train_particles={max_dense_train_particles}; raise the cap only for a profiled fused perception/Target2D backward configuration"
        ))
        .into());
    }
    let seed_mode = parse_seed_mode(config.rollout.seed_mode.as_deref())?;
    let pair_interactions = rollout_particles
        .checked_mul(rollout_particles)
        .ok_or_else(|| std::io::Error::other("rollout particle pair count overflowed"))?;
    let batch_pair_interactions = pair_interactions
        .checked_mul(effective_rollout_batch_size)
        .ok_or_else(|| std::io::Error::other("rollout batch particle pair count overflowed"))?;

    let adapter_rank = config.adapter.rank.unwrap_or(16).max(1);
    let adapter_alpha = config.adapter.alpha.unwrap_or(adapter_rank as f32);
    if adapter_alpha <= 0.0 {
        return Err(std::io::Error::other("adapter.alpha must be positive").into());
    }
    let lr_schedule = parse_e2e_lr_schedule(config.optimizer.lr_schedule.as_deref())?;
    let lr_schedule_label = lr_schedule.as_str().to_string();
    let lr_warmup_steps = config.optimizer.warmup_steps.unwrap_or(0);
    let min_lr_scale = config.optimizer.min_lr_scale.unwrap_or(1.0).clamp(0.0, 1.0);
    let learning_rate = config.optimizer.learning_rate.unwrap_or(1.0e-4);
    let base_learning_rate = config.optimizer.base_learning_rate.unwrap_or(learning_rate);
    let generator_learning_rate = config
        .optimizer
        .generator_learning_rate
        .unwrap_or(learning_rate);
    let amortization_learning_rate = config
        .optimizer
        .amortization_learning_rate
        .unwrap_or(2.0e-6);
    if amortization_enabled
        && (!amortization_learning_rate.is_finite() || amortization_learning_rate <= 0.0)
    {
        return Err(std::io::Error::other(
            "optimizer.amortization_learning_rate must be positive and finite when amortization is enabled",
        )
        .into());
    }
    let target2d_loss_backend = config
        .training
        .target2d_loss_backend
        .as_deref()
        .map(Target2dLossBackend::parse)
        .transpose()?
        .unwrap_or_default();
    let perception_backend = config
        .training
        .perception_backend
        .as_deref()
        .map(PerceptionRolloutBackend::parse)
        .transpose()?
        .unwrap_or_default();
    if steps > 0 && rollout_particles >= 2048 {
        if target2d_loss_backend == Target2dLossBackend::Dense {
            return Err(std::io::Error::other(
                "quality-scale HyperNPA training requires target2d_loss_backend=auto or tiled-adjoint",
            )
            .into());
        }
        if perception_backend == PerceptionRolloutBackend::Dense {
            return Err(std::io::Error::other(
                "quality-scale HyperNPA training requires perception_backend=auto or tiled-adjoint",
            )
            .into());
        }
    }
    let weight_decay = config.optimizer.weight_decay.unwrap_or(0.0);
    let base_weight_decay = config.optimizer.base_weight_decay.unwrap_or(weight_decay);
    let generator_weight_decay = config
        .optimizer
        .generator_weight_decay
        .unwrap_or(weight_decay);
    let grad_clip_norm = config.optimizer.grad_clip_norm.unwrap_or(1.0);
    let base_grad_clip_norm = config
        .optimizer
        .base_grad_clip_norm
        .unwrap_or(grad_clip_norm);
    let generator_grad_clip_norm = config
        .optimizer
        .generator_grad_clip_norm
        .unwrap_or(grad_clip_norm);
    let target_report = E2eRolloutTargetReport {
        points: config.target.points.unwrap_or(4096).max(1),
        image_size: config.target.image_size,
        threshold: config.target.threshold.unwrap_or(0.05),
        loss_image_size: config.target.loss_image_size.unwrap_or(128).max(1),
        splat_sigma: config.target.splat_sigma.unwrap_or(1.0),
        splat_loss_weight: config.target.splat_loss_weight.unwrap_or(2.0),
        color_loss_weight: config.target.color_loss_weight.unwrap_or(5.0),
        density_loss_weight: config.target.density_loss_weight.unwrap_or(1.0),
        composited_rgb_loss_weight: config.target.composited_rgb_loss_weight.unwrap_or(0.0),
        displacement_regularizer_weight: config
            .target
            .displacement_regularizer_weight
            .unwrap_or(0.01),
        overflow_regularizer_weight: config.target.overflow_regularizer_weight.unwrap_or(100.0),
        bound_regularizer_weight: config.target.bound_regularizer_weight.unwrap_or(100.0),
    };
    let validation_particles = config.validation.particles.unwrap_or(2048).max(1);
    let requested_validation_steps = config.validation.steps.unwrap_or(64).max(1);
    let validation_horizons = normalize_validation_horizons(
        config.validation.horizons.as_deref(),
        requested_validation_steps,
    )?;
    let validation_steps = *validation_horizons
        .last()
        .expect("normalized validation horizons are non-empty");
    let validation_selection_horizon_min_steps = config
        .validation
        .selection_horizon_min_steps
        .unwrap_or(validation_steps);
    if validation_selection_horizon_min_steps == 0
        || validation_selection_horizon_min_steps > validation_steps
    {
        return Err(std::io::Error::other(format!(
            "validation.selection_horizon_min_steps must be in 1..={validation_steps}"
        ))
        .into());
    }
    let final_validation_particles = config
        .validation
        .final_particles
        .unwrap_or(validation_particles)
        .max(1);
    let requested_final_validation_steps = config
        .validation
        .final_steps
        .unwrap_or(validation_steps)
        .max(1);
    let final_validation_horizons = normalize_validation_horizons(
        config.validation.final_horizons.as_deref(),
        requested_final_validation_steps,
    )?;
    let final_validation_steps = *final_validation_horizons
        .last()
        .expect("normalized final validation horizons are non-empty");
    let final_validation_selection_horizon_min_steps = config
        .validation
        .final_selection_horizon_min_steps
        .unwrap_or(validation_selection_horizon_min_steps);
    if final_validation_selection_horizon_min_steps == 0
        || final_validation_selection_horizon_min_steps > final_validation_steps
    {
        return Err(std::io::Error::other(format!(
            "validation.final_selection_horizon_min_steps must be in 1..={final_validation_steps}"
        ))
        .into());
    }
    let validation_interval = config.validation.interval.unwrap_or(report_interval).max(1);
    let checkpoint_interval_steps = config
        .output
        .checkpoint_interval_steps
        .unwrap_or(validation_interval)
        .max(1);
    let checkpoint_interval_seconds = config
        .output
        .checkpoint_interval_seconds
        .unwrap_or(1800)
        .max(1);
    let validation_examples = config.validation.examples.unwrap_or(16);
    let validation_split = config.validation.split.as_deref().unwrap_or("auto");
    if !matches!(validation_split, "auto" | "train" | "holdout") {
        return Err(
            std::io::Error::other("validation.split must be auto, train, or holdout").into(),
        );
    }
    if validation_split == "holdout" && holdout_examples == 0 {
        return Err(std::io::Error::other(
            "validation.split=holdout requires at least one held-out example",
        )
        .into());
    }
    let final_validation_examples = config
        .validation
        .final_examples
        .unwrap_or(validation_examples);
    let initial_validation_examples = config
        .validation
        .initial_examples
        .unwrap_or(validation_examples);
    let stability_examples = config.validation.stability_examples.unwrap_or(0);
    let stability_particles = config
        .validation
        .stability_particles
        .unwrap_or(MAX_STABILITY_PARTICLES)
        .max(1);
    let stability_reference_steps = config
        .validation
        .stability_reference_steps
        .unwrap_or(final_validation_steps)
        .max(1);
    let stability_steps = config
        .validation
        .stability_steps
        .unwrap_or(MAX_STABILITY_STEPS)
        .max(1);
    let stability_tail_steps = config.validation.stability_tail_steps.unwrap_or(256).max(1);
    if stability_examples > MAX_STABILITY_EXAMPLES {
        return Err(std::io::Error::other(format!(
            "validation.stability_examples={stability_examples} exceeds the bounded no-grad limit {MAX_STABILITY_EXAMPLES}"
        ))
        .into());
    }
    if stability_examples > holdout_examples {
        return Err(std::io::Error::other(format!(
            "validation.stability_examples={stability_examples} requires at least that many held-out examples, but the split resolves {holdout_examples}"
        ))
        .into());
    }
    if stability_examples > 0 && stability_particles > MAX_STABILITY_PARTICLES {
        return Err(std::io::Error::other(format!(
            "validation.stability_particles={stability_particles} exceeds the bounded no-grad limit {MAX_STABILITY_PARTICLES}"
        ))
        .into());
    }
    if stability_examples > 0 && stability_steps > MAX_STABILITY_STEPS {
        return Err(std::io::Error::other(format!(
            "validation.stability_steps={stability_steps} exceeds the bounded no-grad limit {MAX_STABILITY_STEPS}"
        ))
        .into());
    }
    if stability_examples > 0 && stability_reference_steps >= stability_steps {
        return Err(std::io::Error::other(format!(
            "validation.stability_reference_steps={stability_reference_steps} must be less than stability_steps={stability_steps}"
        ))
        .into());
    }
    if stability_examples > 0 && stability_tail_steps > stability_reference_steps {
        return Err(std::io::Error::other(format!(
            "validation.stability_tail_steps={stability_tail_steps} must be no greater than stability_reference_steps={stability_reference_steps}"
        ))
        .into());
    }
    let validation_quality_scale = validation_particles >= 2048
        || validation_steps >= 32
        || validation_examples >= 16
        || initial_validation_examples >= 16;
    let validation_training_backward_safe = validation_particles <= max_dense_train_particles;
    let shared_base_train_start_step = config.model.shared_base_train_start_step.unwrap_or(0);
    let blockers = Vec::new();
    let mut warnings = Vec::new();
    if steps > 0
        && validation_quality_scale
        && config.validation.psnr_threshold_db.unwrap_or(26.0) < 26.0
    {
        warnings.push(
            "validation.psnr_threshold_db is below the 26dB quality-parity target".to_string(),
        );
    }
    if steps > 0 && (rollout_particles < 512 || rollout_steps < 16) {
        warnings.push(format!(
            "training rollout scale is curriculum/diagnostic only: particles={rollout_particles}, step_min={rollout_step_min}, steps={rollout_steps}; use high-particle validation for quality claims"
        ));
    }
    if steps > 0 && upstream_growing_trajectory_exposure_fraction < 0.1 {
        warnings.push(format!(
            "planned mean trajectory exposure is {planned_mean_trajectories_per_train_example:.1} per training image ({:.3}% of the upstream growing reference {UPSTREAM_GROWING_TRAJECTORIES_PER_TARGET}); treat this as a curriculum/ablation run, not per-target parity evidence",
            upstream_growing_trajectory_exposure_fraction * 100.0,
        ));
    }
    if steps > 0 && validation_particles > max_dense_train_particles {
        warnings.push(format!(
            "validation uses {validation_particles} particles above dense backward cap {max_dense_train_particles}; this is validation-only and must not be used for dense autodiff training"
        ));
    }

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
    let condition_device_cache_max_bytes = config
        .gpu
        .condition_device_cache_max_bytes
        .unwrap_or(DEFAULT_DEVICE_CONDITION_CACHE_MAX_BYTES);
    let target_device_cache_max_bytes = config
        .gpu
        .target_device_cache_max_bytes
        .unwrap_or(DEFAULT_DEVICE_TARGET_CACHE_MAX_BYTES);
    let uses_on_demand_dino =
        online_dino && !matches!(encoder, RolloutConditionEncoder::SampleIdOneHot);
    let cache_complete_condition_set_on_device = uses_on_demand_dino
        && condition_device_cache_max_bytes > 0
        && selected_feature_cache_bytes_f32 <= condition_device_cache_max_bytes;
    let projected_condition_load_peak_bytes_f32 = if cache_complete_condition_set_on_device {
        selected_feature_cache_bytes_f32.saturating_add(dino_batch_input_bytes_f32)
    } else if uses_on_demand_dino {
        dino_batch_input_bytes_f32
    } else if selected_feature_cache_bytes_f32 > condition_device_cache_max_bytes {
        selected_feature_cache_bytes_f32.saturating_add(dino_batch_input_bytes_f32)
    } else {
        selected_feature_cache_bytes_f32
            .saturating_mul(2)
            .saturating_add(dino_batch_input_bytes_f32)
    };
    let adapter_generator_kind = E2eHyperGeneratorKind::parse(config.adapter.generator.as_deref())?;
    if objective == "conditional-row-flow-e2e"
        && adapter_generator_kind != E2eHyperGeneratorKind::ConditionalRowFlow
    {
        return Err(std::io::Error::other(
            "conditional-row-flow-e2e requires adapter.generator=conditional-row-flow",
        )
        .into());
    }
    let adapter_generator = adapter_generator_kind.artifact_architecture().to_string();
    if amortization_enabled && adapter_generator_kind != E2eHyperGeneratorKind::ConditionalRowFlow {
        return Err(std::io::Error::other(
            "training amortization is only supported by adapter.generator=conditional-row-flow",
        )
        .into());
    }
    match adapter_generator_kind {
        E2eHyperGeneratorKind::SampleIdTable => warnings.push(
            "sample-id-table is a memorization/substrate control and cannot support unseen-image HyperNPA generalization claims"
                .to_string(),
        ),
        E2eHyperGeneratorKind::ModuleTokenDecoderV2 => warnings.push(
            "module-token-decoder-v2 is retained for artifact compatibility; use module-token-decoder for the maintained multi-head generalized path"
                .to_string(),
        ),
        _ => {}
    }
    let attention_normalization = match config.adapter.attention_normalization.as_deref() {
        None if adapter_generator_kind == E2eHyperGeneratorKind::ModuleTokenDecoder => {
            E2E_HYPER_ATTENTION_SOFTMAX
        }
        None | Some(E2E_HYPER_ATTENTION_TANH_EXP) => E2E_HYPER_ATTENTION_TANH_EXP,
        Some(E2E_HYPER_ATTENTION_SOFTMAX) => E2E_HYPER_ATTENTION_SOFTMAX,
        Some(other) => {
            return Err(std::io::Error::other(format!(
                "unsupported adapter.attention_normalization {other:?}; expected tanh-exp or softmax"
            ))
            .into());
        }
    };
    let adapter_parameterization = match config.adapter.parameterization.as_deref() {
        None if adapter_generator_kind == E2eHyperGeneratorKind::ConditionalRowFlow => {
            E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
        }
        None | Some(E2E_HYPER_ADAPTER_FACTORIZED) => E2E_HYPER_ADAPTER_FACTORIZED,
        Some(E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL)
        | Some(E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK)
            if adapter_generator_kind == E2eHyperGeneratorKind::ConditionalRowFlow =>
        {
            E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
        }
        Some(E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK) => E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK,
        Some(other) => {
            return Err(std::io::Error::other(format!(
                "unsupported adapter.parameterization {other:?}; expected factorized, canonical-full-rank, or dense-npa-row-residual"
            ))
            .into());
        }
    };
    if adapter_generator_kind == E2eHyperGeneratorKind::ConditionalRowFlow
        && adapter_parameterization != E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
    {
        return Err(std::io::Error::other(
            "conditional-row-flow requires adapter.parameterization = dense-npa-row-residual",
        )
        .into());
    }
    if flow_matching_weight > 0.0
        && adapter_generator_kind != E2eHyperGeneratorKind::ConditionalRowFlow
    {
        return Err(std::io::Error::other(
            "training.flow_matching_weight is only supported by conditional-row-flow",
        )
        .into());
    }
    if flow_self_rectification_weight > 0.0
        && adapter_generator_kind != E2eHyperGeneratorKind::ConditionalRowFlow
    {
        return Err(std::io::Error::other(
            "training.flow_self_rectification_weight is only supported by conditional-row-flow",
        )
        .into());
    }
    if flow_matching_weight > 0.0 && !has_adapter_endpoints {
        return Err(std::io::Error::other(
            "training.flow_matching_weight requires model.oracle_model_dir or model.adapter_bank endpoint targets",
        )
        .into());
    }
    let (adapter_npa_config, _) = NpaConfig::for_preset(preset);
    if matches!(
        adapter_parameterization,
        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK | E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
    ) {
        crate::hyper::adapter_layout::CanonicalFullRankLora2d::new_with_output_bias(
            &adapter_npa_config,
            adapter_rank,
            adapter_alpha,
            false,
        )?;
    }
    let amortization_parameter_count = if amortization_enabled {
        let layout = NpaParameterRowLayout2d::new(&adapter_npa_config);
        train_examples
            .saturating_mul(layout.row_count())
            .saturating_mul(layout.max_row_dims())
    } else {
        0
    };
    let amortization_optimizer_bytes_f32 = amortization_parameter_count
        .saturating_mul(3)
        .saturating_mul(std::mem::size_of::<f32>());
    if amortization_enabled {
        warnings.push(format!(
            "training-only amortization residuals allocate {amortization_parameter_count} parameters and {} bytes including AdamW state; validation and serialized inference exclude this table",
            amortization_optimizer_bytes_f32,
        ));
        if let Some(budget_gb) = config.training.gpu_memory_budget_gb {
            let budget_bytes = (budget_gb.max(0.0) as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
            if amortization_optimizer_bytes_f32 > budget_bytes {
                return Err(std::io::Error::other(format!(
                    "amortization table plus AdamW state requires {amortization_optimizer_bytes_f32} bytes, above gpu_memory_budget_gb={budget_gb}"
                ))
                .into());
            }
        }
    }
    let adapter_init_scale = config.adapter.init_scale.unwrap_or(1.0e-3);
    let adapter_condition_init_scale = config
        .adapter
        .condition_init_scale
        .unwrap_or(adapter_init_scale);
    let adapter_output_init_scale = config
        .adapter
        .output_init_scale
        .unwrap_or(adapter_init_scale);
    let flow_default_endpoint_rms = config.adapter.flow_default_endpoint_rms.unwrap_or(0.02);
    if !flow_default_endpoint_rms.is_finite() || flow_default_endpoint_rms <= 0.0 {
        return Err(std::io::Error::other(
            "adapter.flow_default_endpoint_rms must be positive and finite",
        )
        .into());
    }
    let row_flow_selected = adapter_generator_kind == E2eHyperGeneratorKind::ConditionalRowFlow;
    let token_attention_heads = config
        .condition
        .token_attention_heads
        .unwrap_or(if row_flow_selected { 12 } else { 4 })
        .max(1);
    let generator_hidden_dims = config
        .adapter
        .flow_hidden
        .unwrap_or(if row_flow_selected { 768 } else { 512 })
        .max(1);
    let generator_layers = config
        .adapter
        .flow_layers
        .unwrap_or(if row_flow_selected { 12 } else { 1 })
        .max(1);
    let generator_ffn_dims = config
        .adapter
        .flow_ffn_dims
        .unwrap_or(if row_flow_selected {
            4 * generator_hidden_dims
        } else {
            generator_hidden_dims
        })
        .max(generator_hidden_dims);
    let generator_sample_steps = config
        .adapter
        .flow_sample_steps
        .unwrap_or(if row_flow_selected { 8 } else { 16 })
        .max(1);
    let generator_train_sample_steps = config
        .training
        .flow_train_sample_steps
        .unwrap_or(generator_sample_steps)
        .max(1);
    if generator_train_sample_steps > generator_sample_steps {
        return Err(std::io::Error::other(format!(
            "training.flow_train_sample_steps={generator_train_sample_steps} cannot exceed adapter.flow_sample_steps={generator_sample_steps}"
        ))
        .into());
    }
    if row_flow_selected && generator_train_sample_steps < generator_sample_steps {
        warnings.push(format!(
            "rectified-flow training uses {generator_train_sample_steps} Heun steps while validation and serialized inference use {generator_sample_steps}; validation remains the fidelity gate"
        ));
    }
    let generator_source_seed = config.adapter.flow_source_seed.unwrap_or(42);
    if matches!(
        adapter_generator_kind,
        E2eHyperGeneratorKind::ModuleTokenDecoder | E2eHyperGeneratorKind::ConditionalRowFlow
    ) && !generator_hidden_dims.is_multiple_of(token_attention_heads)
    {
        return Err(std::io::Error::other(format!(
            "adapter.flow_hidden={generator_hidden_dims} must be divisible by condition.token_attention_heads={token_attention_heads} for {adapter_generator}"
        ))
        .into());
    }
    if row_flow_selected && config.adapter.spatial_condition_control.unwrap_or(false) {
        return Err(std::io::Error::other(
            "conditional-row-flow emits a static controller and does not support per-step spatial condition control",
        )
        .into());
    }
    if !adapter_condition_init_scale.is_finite()
        || adapter_condition_init_scale <= 0.0
        || !adapter_output_init_scale.is_finite()
        || adapter_output_init_scale < 0.0
    {
        return Err(std::io::Error::other(
            "adapter condition_init_scale must be positive and output_init_scale must be non-negative; both must be finite",
        )
        .into());
    }
    if adapter_generator == E2E_HYPER_ARCH_SAMPLE_ID_TABLE
        && encoder != RolloutConditionEncoder::SampleIdOneHot
    {
        return Err(std::io::Error::other(
            "adapter.generator = sample-id-table requires condition.encoder = sample-id-onehot",
        )
        .into());
    }
    if adapter_generator == E2E_HYPER_ARCH_SAMPLE_ID_TABLE
        && config.adapter.spatial_condition_control.unwrap_or(false)
    {
        return Err(std::io::Error::other(
            "sample-ID adapter tables do not support per-step spatial condition control",
        )
        .into());
    }
    if config
        .adapter
        .spatial_condition_state_control
        .unwrap_or(false)
        && !config.adapter.spatial_condition_control.unwrap_or(false)
    {
        return Err(std::io::Error::other(
            "adapter.spatial_condition_state_control requires spatial_condition_control=true",
        )
        .into());
    }
    let adapter_chunk_size = config
        .adapter
        .adapter_chunk_size
        .unwrap_or(DEFAULT_E2E_HYPER_ADAPTER_CHUNK_SIZE)
        .max(1);
    let condition_feature_normalization = if encoder == RolloutConditionEncoder::SampleIdOneHot {
        "sample-id-onehot"
    } else if matches!(
        adapter_generator.as_str(),
        E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW
            | E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW
            | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER_V2
            | E2E_HYPER_ARCH_MODULE_TOKEN_DECODER
    ) {
        "per-token-preserved"
    } else {
        "flattened-l2"
    };
    let auto_resume = config.output.auto_resume.unwrap_or(false);
    let effective_resume_checkpoint = config.output.resume_checkpoint.clone().or_else(|| {
        let state = checkpoint_dir.join("current_training_state.mpk");
        (auto_resume && state.is_file()).then_some(checkpoint_dir.clone())
    });
    if curriculum_resume && effective_resume_checkpoint.is_none() {
        return Err(std::io::Error::other(
            "training.curriculum_resume=true requires output.resume_checkpoint or an auto-resume checkpoint",
        )
        .into());
    }
    if curriculum_resume {
        warnings.push(
            "curriculum resume restores model, optimizer, training-only endpoint state, and compatible pool/sampler state; incompatible runtime state is reset explicitly and reported"
                .to_string(),
        );
    }

    Ok(E2eRolloutReport {
        experiment_config: config_path.display().to_string(),
        status: if blockers.is_empty() {
            "preflight-ok"
        } else {
            "blocked"
        },
        implementation_status: if row_flow_selected {
            "burn_conditional_row_rectified_flow_v5"
        } else {
            "burn_static_adapter_full_bptt_alpha_aware_v4_multihead"
        },
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
        warnings,
        output: E2eRolloutOutputReport {
            output_dir: display_path(&output_dir),
            report_output: display_path(&report_output),
            shared_base_output: display_path(&shared_base_output),
            hyper_output: display_path(&hyper_output),
            checkpoint_dir: display_path(&checkpoint_dir),
            checkpoint_interval_steps,
            checkpoint_interval_seconds,
            resume_checkpoint: effective_resume_checkpoint
                .as_ref()
                .map(|path| display_path(path)),
            auto_resume,
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
            embed_dims,
            flattened_feature_dims,
            selected_feature_cache_bytes_f32,
            selected_feature_cache_gib_f32: bytes_to_gib(selected_feature_cache_bytes_f32),
            dino_batch_input_bytes_f32,
            dino_batch_input_gib_f32: bytes_to_gib(dino_batch_input_bytes_f32),
            projected_condition_load_peak_bytes_f32,
            projected_condition_load_peak_gib_f32: bytes_to_gib(
                projected_condition_load_peak_bytes_f32,
            ),
            device_cache_max_bytes: condition_device_cache_max_bytes,
            device_cache_max_gib: bytes_to_gib(condition_device_cache_max_bytes),
            device_cache_plan: if cache_complete_condition_set_on_device {
                "complete-set-device-resident"
            } else if uses_on_demand_dino {
                "on-demand-device-encode"
            } else {
                "precomputed-feature-input"
            },
            token_attention_heads,
            feature_normalization: condition_feature_normalization,
            rgb_channels,
            rgb_channel_scale,
            alpha_channel,
            alpha_channel_scale,
            patch_pixels,
            sample_id_vocab_size,
            sample_ids,
        },
        model: E2eRolloutModelReport {
            shared_base: config
                .model
                .shared_base
                .as_ref()
                .map(|path| display_path(path)),
            hyper: config.model.hyper.as_ref().map(|path| display_path(path)),
            adapter_bank: config
                .model
                .adapter_bank
                .as_ref()
                .map(|path| display_path(path)),
            shared_base_trainable: config.model.shared_base_trainable.unwrap_or(true),
            shared_base_train_start_step,
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
            oracle_model_dir: config
                .model
                .oracle_model_dir
                .as_ref()
                .map(|path| display_path(path)),
        },
        training: E2eRolloutTrainingReport {
            backend,
            gpu_backend,
            objective,
            steps,
            report_interval,
            example_batch_size,
            tbptt_chunk_steps,
            tbptt_state_detach_active: effective_tbptt_chunk_steps.is_some(),
            effective_tbptt_chunk_steps,
            rollout_gradient_horizon_max_steps,
            loss_on_final_chunk_only,
            tbptt_loss_mode: tbptt_loss_mode.as_str().to_string(),
            tbptt_intermediate_loss_weight,
            tbptt_final_loss_weight,
            credit_assignment: credit_assignment.as_str().to_string(),
            max_full_bptt_particle_steps,
            use_particle_pool,
            pool_slots_per_example,
            rollouts_per_example,
            target_mean_trajectories_per_example,
            sampling_uniform_fraction,
            sampling_priority_ema_beta,
            sampling_priority_min_weight,
            sampling_priority_max_weight,
            sampling_priority_update_interval,
            planned_rollout_trajectories,
            planned_mean_trajectories_per_train_example,
            upstream_growing_reference_trajectories_per_target:
                UPSTREAM_GROWING_TRAJECTORIES_PER_TARGET,
            upstream_growing_trajectory_exposure_fraction,
            pool_capacity,
            inject_seed_interval,
            seed_replacements_per_interval,
            seed_trajectory_interval,
            brush_size,
            pre_rollout_step_min,
            pre_rollout_steps,
            target2d_loss_backend: target2d_loss_backend.as_str().to_string(),
            perception_backend: perception_backend.as_str().to_string(),
            max_dense_train_particles,
            system_memory_budget_gb: config.training.system_memory_budget_gb,
            gpu_memory_budget_gb: config.training.gpu_memory_budget_gb,
            max_dense_chunk_floats: config.gpu.max_dense_chunk_floats.unwrap_or(1_048_576),
            max_splat_chunk_floats: config.gpu.max_splat_chunk_floats.unwrap_or(1_048_576),
            condition_device_cache_max_bytes,
            target_device_cache_max_bytes,
            seed: config.training.seed.unwrap_or(42),
            adapter_teacher_weight,
            adapter_teacher_objective: adapter_teacher_objective.as_str().to_string(),
            adapter_teacher_probe_rollout_steps: config
                .training
                .adapter_teacher_probe_rollout_steps
                .unwrap_or(0),
            task_loss_weight,
            flow_matching_weight,
            flow_match_inference_source,
            flow_train_sample_steps: generator_train_sample_steps,
            flow_self_rectification_weight,
            curriculum_resume,
            amortization_enabled,
            amortization_substrate_steps,
            amortization_residual_scale,
            amortization_residual_anneal_steps,
            amortization_hyper_only_fraction,
            amortization_distillation_weight,
            amortization_distillation_objective: amortization_distillation_objective
                .as_str()
                .to_string(),
            amortization_distillation_probe_rollout_steps,
            amortization_initialize_from_teacher,
            amortization_initialize_from_generator,
            amortization_parameter_count,
            amortization_optimizer_bytes_f32,
            trains_shared_base_from_step_zero: config.model.shared_base_trainable.unwrap_or(true)
                && shared_base_train_start_step == 0,
            trains_hypernet_from_step_zero: amortization_substrate_steps == 0,
        },
        adapter: E2eRolloutAdapterReport {
            rank: adapter_rank,
            alpha: adapter_alpha,
            output_bias: false,
            generator: adapter_generator,
            flow_hidden: generator_hidden_dims,
            flow_layers: generator_layers,
            flow_ffn_dims: generator_ffn_dims,
            flow_sample_steps: generator_sample_steps,
            flow_source_seed: generator_source_seed,
            flow_source_scale: config.adapter.flow_source_scale.unwrap_or(1.0),
            flow_default_endpoint_rms,
            init_scale: adapter_init_scale,
            condition_init_scale: adapter_condition_init_scale,
            output_init_scale: adapter_output_init_scale,
            adapter_chunk_size,
            attention_normalization: attention_normalization.to_string(),
            parameterization: adapter_parameterization.to_string(),
            spatial_condition_control: config.adapter.spatial_condition_control.unwrap_or(false),
            spatial_condition_control_scale: config
                .adapter
                .spatial_condition_control_scale
                .unwrap_or(0.1),
            spatial_condition_control_sigma: config
                .adapter
                .spatial_condition_control_sigma
                .unwrap_or(0.25),
            spatial_condition_state_control: config
                .adapter
                .spatial_condition_state_control
                .unwrap_or(false),
        },
        rollout: E2eRolloutRuntimeReport {
            particles: rollout_particles,
            step_min: rollout_step_min,
            steps: rollout_steps,
            sampled_training_steps,
            update_prob: config.rollout.update_prob.unwrap_or(0.5),
            seed: config.rollout.seed.unwrap_or(42),
            seed_scale: config.rollout.seed_scale,
            seed_mode,
            estimated_pair_interactions_per_example: pair_interactions,
            estimated_pair_interactions_per_batch: batch_pair_interactions,
        },
        target: target_report,
        optimizer: E2eRolloutOptimizerReport {
            learning_rate,
            base_learning_rate,
            generator_learning_rate,
            amortization_learning_rate,
            weight_decay,
            base_weight_decay,
            generator_weight_decay,
            grad_clip_norm,
            base_grad_clip_norm,
            generator_grad_clip_norm,
            adam_beta1: config.optimizer.adam_beta1.unwrap_or(0.9),
            adam_beta2: config.optimizer.adam_beta2.unwrap_or(0.999),
            adam_epsilon: config.optimizer.adam_epsilon.unwrap_or(1.0e-8),
            per_parameter_grad_normalization: config
                .optimizer
                .per_parameter_grad_normalization
                .unwrap_or(true),
            base_per_parameter_grad_normalization: config
                .optimizer
                .base_per_parameter_grad_normalization
                .unwrap_or_else(|| {
                    config
                        .optimizer
                        .per_parameter_grad_normalization
                        .unwrap_or(true)
                }),
            generator_per_parameter_grad_normalization: config
                .optimizer
                .generator_per_parameter_grad_normalization
                .unwrap_or(false),
            amortization_grad_normalization: config
                .optimizer
                .amortization_grad_normalization
                .unwrap_or(true),
            lr_schedule: lr_schedule_label,
            warmup_steps: lr_warmup_steps,
            min_lr_scale,
        },
        validation: E2eRolloutValidationReport {
            split: validation_split.to_string(),
            initial_examples: initial_validation_examples,
            examples: validation_examples,
            interval: validation_interval,
            particles: validation_particles,
            steps: validation_steps,
            horizons: validation_horizons,
            selection_horizon_min_steps: validation_selection_horizon_min_steps,
            quality_scale: validation_quality_scale,
            training_backward_safe: validation_training_backward_safe,
            update_prob: config.validation.update_prob.unwrap_or(0.5),
            seed: config.validation.seed.unwrap_or(42),
            oracle_report: config
                .validation
                .oracle_report
                .as_ref()
                .map(|path| display_path(path)),
            psnr_threshold_db: config.validation.psnr_threshold_db.unwrap_or(26.0),
            final_examples: final_validation_examples,
            final_particles: final_validation_particles,
            final_steps: final_validation_steps,
            final_horizons: final_validation_horizons,
            final_selection_horizon_min_steps: final_validation_selection_horizon_min_steps,
            stability_examples,
            stability_particles,
            stability_reference_steps,
            stability_steps,
            stability_tail_steps,
            stability_evaluation_mode: "final-only-detached-generated-adapter",
        },
        gates: E2eRolloutGateReport {
            min_median_particle_steps_per_sec: config.gates.min_median_particle_steps_per_sec,
            max_quality_validation_evaluations: config.gates.max_quality_validation_evaluations,
            max_quality_validation_elapsed_fraction: config
                .gates
                .max_quality_validation_elapsed_fraction,
            min_final_mean_render_rgb_psnr_db: config.gates.min_final_mean_render_rgb_psnr_db,
            min_final_p10_composited_rgb_psnr_db: config.gates.min_final_p10_composited_rgb_psnr_db,
            min_final_condition_shuffle_composited_psnr_gap_db: config
                .gates
                .min_final_condition_shuffle_composited_psnr_gap_db,
            min_final_generated_adapter_composited_psnr_gain_db: config
                .gates
                .min_final_generated_adapter_composited_psnr_gain_db,
            max_final_p90_gap_to_matched_oracle_db: config
                .gates
                .max_final_p90_gap_to_matched_oracle_db,
            require_validation_interval_at_least_report_interval: config
                .gates
                .require_validation_interval_at_least_report_interval
                .unwrap_or(false),
            fail_on_violation: config.gates.fail_on_violation.unwrap_or(true),
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
    let teacher_train_examples = if let Some(adapter_bank) = report.model.adapter_bank.as_deref() {
        attach_adapter_bank(&base, &mut train_examples, report, Path::new(adapter_bank))?;
        train_examples.len()
    } else if let Some(oracle_model_dir) = report.model.oracle_model_dir.as_deref() {
        attach_exact_oracle_adapters(
            &base,
            &mut train_examples,
            Path::new(oracle_model_dir),
            report.adapter.rank,
            report.adapter.alpha,
        )?;
        train_examples.len()
    } else {
        0
    };
    ensure_holdout_teacher_free(&holdout_examples)?;
    let initial_generator = report
        .model
        .hyper
        .as_deref()
        .map(crate::load_e2e_hyper_npa_2d)
        .transpose()?;
    if let Some(expected) = initial_generator
        .as_ref()
        .and_then(|generator| generator.shared_base_sha256.as_deref())
    {
        let shared_base_path = report.model.shared_base.as_deref().ok_or_else(|| {
            std::io::Error::other(
                "model.hyper warm start carries a shared-base checksum, so model.shared_base must point to its paired BPK",
            )
        })?;
        let shared_base_bytes = std::fs::read(shared_base_path)?;
        let actual = crate::import::bpk_payload_sha256(&shared_base_bytes)?;
        if actual != expected {
            return Err(std::io::Error::other(format!(
                "model.hyper shared-base checksum {expected} does not match model.shared_base {actual}"
            ))
            .into());
        }
    }
    let loss_config = target2d_loss_config(
        report.target.loss_image_size,
        report.target.splat_sigma,
        true,
        report.target.splat_loss_weight,
        report.target.color_loss_weight,
        report.target.density_loss_weight,
        Target2dLossConfig::default().background_density_loss_weight,
        Target2dLossConfig::default().foreground_density_loss_weight,
        report.target.composited_rgb_loss_weight,
        report.target.displacement_regularizer_weight,
        report.target.overflow_regularizer_weight,
        report.target.bound_regularizer_weight,
    )?;
    let checkpoint_dir: &'static str =
        Box::leak(report.output.checkpoint_dir.clone().into_boxed_str());
    let resume_checkpoint: Option<&'static str> = report
        .output
        .resume_checkpoint
        .clone()
        .map(|path| &*Box::leak(path.into_boxed_str()));
    let mut validation_horizons = [0usize; MAX_VALIDATION_HORIZONS];
    validation_horizons[..report.validation.horizons.len()]
        .copy_from_slice(&report.validation.horizons);
    let mut final_validation_horizons = [0usize; MAX_VALIDATION_HORIZONS];
    final_validation_horizons[..report.validation.final_horizons.len()]
        .copy_from_slice(&report.validation.final_horizons);
    let train_config = BurnE2eRolloutTrainConfig {
        steps: report.training.steps,
        report_interval: report.training.report_interval,
        example_batch_size: report.training.example_batch_size,
        tbptt_chunk_steps: report.training.tbptt_chunk_steps,
        loss_on_final_chunk_only: report.training.loss_on_final_chunk_only,
        tbptt_loss_mode: parse_e2e_tbptt_loss_mode(
            Some(&report.training.tbptt_loss_mode),
            report.training.loss_on_final_chunk_only,
        )?,
        tbptt_intermediate_loss_weight: report.training.tbptt_intermediate_loss_weight,
        tbptt_final_loss_weight: report.training.tbptt_final_loss_weight,
        credit_assignment: parse_e2e_credit_assignment(Some(&report.training.credit_assignment))?,
        max_full_bptt_particle_steps: report.training.max_full_bptt_particle_steps,
        use_particle_pool: report.training.use_particle_pool,
        pool_slots_per_example: report.training.pool_slots_per_example,
        rollouts_per_example: report.training.rollouts_per_example,
        sampling_uniform_fraction: report.training.sampling_uniform_fraction,
        sampling_priority_ema_beta: report.training.sampling_priority_ema_beta,
        sampling_priority_min_weight: report.training.sampling_priority_min_weight,
        sampling_priority_max_weight: report.training.sampling_priority_max_weight,
        sampling_priority_update_interval: report.training.sampling_priority_update_interval,
        pool_capacity: report.training.pool_capacity,
        inject_seed_interval: report.training.inject_seed_interval,
        seed_replacements_per_interval: report.training.seed_replacements_per_interval,
        seed_trajectory_interval: report.training.seed_trajectory_interval,
        brush_size: report.training.brush_size,
        pre_rollout_step_min: report.training.pre_rollout_step_min,
        pre_rollout_steps: report.training.pre_rollout_steps,
        rollout_particles: report.rollout.particles,
        rollout_step_min: report.rollout.step_min,
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
        target2d_loss_backend: Target2dLossBackend::parse(&report.training.target2d_loss_backend)?,
        perception_backend: PerceptionRolloutBackend::parse(&report.training.perception_backend)?,
        per_parameter_grad_normalization: report.optimizer.per_parameter_grad_normalization,
        base_per_parameter_grad_normalization: report
            .optimizer
            .base_per_parameter_grad_normalization,
        generator_per_parameter_grad_normalization: report
            .optimizer
            .generator_per_parameter_grad_normalization,
        adapter_teacher_weight: report.training.adapter_teacher_weight,
        adapter_teacher_objective: parse_e2e_adapter_teacher_objective(Some(
            &report.training.adapter_teacher_objective,
        ))?,
        adapter_teacher_probe_rollout_steps: report.training.adapter_teacher_probe_rollout_steps,
        task_loss_weight: report.training.task_loss_weight,
        shared_base_trainable: report.model.shared_base_trainable,
        shared_base_train_start_step: report.model.shared_base_train_start_step,
        base_optimizer: base_adamw_from_report(report),
        generator_optimizer: generator_adamw_from_report(report),
        lr_schedule: parse_e2e_lr_schedule(Some(&report.optimizer.lr_schedule))?,
        lr_warmup_steps: report.optimizer.warmup_steps,
        min_lr_scale: report.optimizer.min_lr_scale,
        adapter_rank: report.adapter.rank,
        adapter_alpha: report.adapter.alpha,
        generator_kind: E2eHyperGeneratorKind::parse(Some(&report.adapter.generator))?,
        adapter_chunk_size: report.adapter.adapter_chunk_size,
        generator_hidden_dims: report.adapter.flow_hidden,
        generator_layers: report.adapter.flow_layers,
        generator_ffn_dims: report.adapter.flow_ffn_dims,
        token_attention_heads: report.condition.token_attention_heads,
        softmax_token_attention: report.adapter.attention_normalization
            == E2E_HYPER_ATTENTION_SOFTMAX,
        canonical_full_rank_lora: matches!(
            report.adapter.parameterization.as_str(),
            E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK | E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
        ),
        adapter_output_bias: report.adapter.output_bias,
        generator_sample_steps: report.adapter.flow_sample_steps,
        generator_train_sample_steps: report.training.flow_train_sample_steps,
        generator_source_seed: report.adapter.flow_source_seed,
        generator_default_endpoint_rms: report.adapter.flow_default_endpoint_rms,
        flow_matching_weight: report.training.flow_matching_weight,
        flow_match_inference_source: report.training.flow_match_inference_source,
        flow_self_rectification_weight: report.training.flow_self_rectification_weight,
        amortization_enabled: report.training.amortization_enabled,
        amortization_substrate_steps: report.training.amortization_substrate_steps,
        amortization_substrate_only: false,
        amortization_residual_scale: report.training.amortization_residual_scale,
        amortization_residual_anneal_steps: report.training.amortization_residual_anneal_steps,
        amortization_hyper_only_fraction: report.training.amortization_hyper_only_fraction,
        amortization_distillation_weight: report.training.amortization_distillation_weight,
        amortization_distillation_objective: parse_e2e_adapter_teacher_objective(Some(
            &report.training.amortization_distillation_objective,
        ))?,
        amortization_distillation_probe_rollout_steps: report
            .training
            .amortization_distillation_probe_rollout_steps,
        amortization_initialize_from_teacher: report.training.amortization_initialize_from_teacher,
        amortization_initialize_from_generator: report
            .training
            .amortization_initialize_from_generator,
        amortization_learning_rate: report.optimizer.amortization_learning_rate,
        amortization_grad_normalization: report.optimizer.amortization_grad_normalization,
        generator_output_scale: report.adapter.flow_source_scale,
        generator_init_scale: report.adapter.init_scale,
        generator_condition_init_scale: report.adapter.condition_init_scale,
        generator_output_init_scale: report.adapter.output_init_scale,
        stopgrad_pos: npa_config.stopgrad_pos,
        stopgrad_state: npa_config.stopgrad_state,
        system_memory_budget_gb: report.training.system_memory_budget_gb,
        gpu_memory_budget_gb: report.training.gpu_memory_budget_gb,
        max_dense_train_particles: report.training.max_dense_train_particles,
        max_dense_chunk_floats: report.training.max_dense_chunk_floats,
        max_splat_chunk_floats: report.training.max_splat_chunk_floats,
        condition_device_cache_max_bytes: report.training.condition_device_cache_max_bytes,
        target_device_cache_max_bytes: report.training.target_device_cache_max_bytes,
        dino_image_size: report.condition.dino_image_size,
        dino_batch_size: report.condition.dino_batch_size,
        dino_token_grid_width: report.condition.patch_grid_width,
        dino_token_grid_height: report.condition.patch_grid_height,
        dino_l2_normalize_features: report.condition.feature_normalization == "flattened-l2",
        dino_rgb_channels: report.condition.rgb_channels,
        dino_rgb_channel_scale: report.condition.rgb_channel_scale,
        dino_alpha_channel: report.condition.alpha_channel,
        dino_alpha_channel_scale: report.condition.alpha_channel_scale,
        dino_patch_pixels: report.condition.patch_pixels,
        spatial_condition_control: report.adapter.spatial_condition_control,
        spatial_condition_control_scale: report.adapter.spatial_condition_control_scale,
        spatial_condition_control_sigma: report.adapter.spatial_condition_control_sigma,
        spatial_condition_state_control: report.adapter.spatial_condition_state_control,
        checkpoint_dir: Some(checkpoint_dir),
        checkpoint_interval_steps: report.output.checkpoint_interval_steps,
        checkpoint_interval_seconds: report.output.checkpoint_interval_seconds,
        resume_checkpoint,
        curriculum_resume: report.training.curriculum_resume,
        checkpoint_condition_encoder: Some(report.condition.encoder),
        validation_split: if report.validation.split == "train" {
            "train"
        } else if report.validation.split == "holdout" {
            "holdout"
        } else {
            "auto"
        },
        initial_validation_examples: report.validation.initial_examples,
        validation_examples: report.validation.examples,
        validation_interval: report.validation.interval,
        validation_particles: report.validation.particles,
        validation_steps: report.validation.steps,
        validation_horizons,
        validation_horizon_count: report.validation.horizons.len(),
        validation_selection_horizon_min_steps: report.validation.selection_horizon_min_steps,
        validation_update_prob: report.validation.update_prob,
        validation_seed: report.validation.seed,
        validation_psnr_threshold_db: report.validation.psnr_threshold_db,
        final_validation_examples: report.validation.final_examples,
        final_validation_particles: report.validation.final_particles,
        final_validation_steps: report.validation.final_steps,
        final_validation_horizons,
        final_validation_horizon_count: report.validation.final_horizons.len(),
        final_validation_selection_horizon_min_steps: report
            .validation
            .final_selection_horizon_min_steps,
        stability_examples: report.validation.stability_examples,
        stability_particles: report.validation.stability_particles,
        stability_reference_steps: report.validation.stability_reference_steps,
        stability_steps: report.validation.stability_steps,
        stability_tail_steps: report.validation.stability_tail_steps,
    };
    let train_started = Instant::now();
    let mut output = match report.training.gpu_backend.as_str() {
        "burn-cuda" | "cuda" => train_e2e_rollout_burn_cuda(
            &mut base,
            &mut train_examples,
            &mut holdout_examples,
            train_config,
            initial_generator.as_ref(),
        )?,
        _ => train_e2e_rollout_burn_wgpu(
            &mut base,
            &mut train_examples,
            &mut holdout_examples,
            train_config,
            initial_generator.as_ref(),
        )?,
    };
    output.generator.condition_encoder = Some(report.condition.encoder.to_string());
    output.generator.condition_token_count = Some(report.condition.token_count);
    output.generator.condition_embed_dims = Some(report.condition.embed_dims);
    output.generator.condition_token_grid_width = Some(report.condition.patch_grid_width);
    output.generator.condition_token_grid_height = Some(report.condition.patch_grid_height);
    if !output.generator.is_conditional_row_flow() {
        output.generator.adapter_chunk_size = Some(report.adapter.adapter_chunk_size);
    }
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
        metrics.insert(
            "teacher_train_examples".to_string(),
            serde_json::json!(teacher_train_examples),
        );
        metrics.insert(
            "adapter_endpoint_source".to_string(),
            serde_json::json!(if report.model.adapter_bank.is_some() {
                "training-only-adapter-bank"
            } else if report.model.oracle_model_dir.is_some() {
                "exact-oracle-models"
            } else {
                "none"
            }),
        );
        metrics.insert(
            "adapter_bank_serialized_for_inference".to_string(),
            serde_json::json!(false),
        );
        metrics.insert("teacher_holdout_examples".to_string(), serde_json::json!(0));
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
    let hyper_sha256 = save_e2e_hyper_npa_2d(&report.output.hyper_output, &output.generator)?;
    let hyper_artifact_bytes = std::fs::metadata(&report.output.hyper_output)?.len();
    if let Some(metrics) = output.metrics.as_object_mut() {
        metrics.insert(
            "hyper_artifact_format".to_string(),
            serde_json::json!("bpk"),
        );
        metrics.insert(
            "hyper_artifact_bytes".to_string(),
            serde_json::json!(hyper_artifact_bytes),
        );
        metrics.insert(
            "hyper_artifact_sha256".to_string(),
            serde_json::json!(hyper_sha256),
        );
    }
    Ok(output)
}

fn attach_exact_oracle_adapters(
    base: &NpaModel,
    examples: &mut [BurnE2eRolloutExample],
    oracle_model_dir: &Path,
    rank: usize,
    alpha: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    for example in examples {
        let path = oracle_model_dir.join(format!("{}.bpk", example.slug));
        let target = crate::import::load_manifest(&path)?.into_model();
        let adapter = exact_teacher_adapter(base, &target, rank, alpha)?;
        example.teacher_adapter = Some(adapter.to_parameter_vector());
    }
    Ok(())
}

fn attach_adapter_bank(
    base: &NpaModel,
    examples: &mut [BurnE2eRolloutExample],
    report: &E2eRolloutReport,
    adapter_bank_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let bank = crate::load_e2e_hyper_npa_2d(adapter_bank_path)?;
    if !bank.is_sample_id_table() {
        return Err(std::io::Error::other(format!(
            "model.adapter_bank must contain a sample-ID adapter table, got {}",
            bank.architecture
        ))
        .into());
    }
    validate_upstream_adapter_bank_contract(&bank)?;
    if let Some(expected) = bank.shared_base_sha256.as_deref() {
        let shared_base_path = report.model.shared_base.as_deref().ok_or_else(|| {
            std::io::Error::other(
                "model.adapter_bank carries a shared-base checksum, so model.shared_base must point to its paired BPK",
            )
        })?;
        let actual = crate::import::bpk_payload_sha256(&std::fs::read(shared_base_path)?)?;
        if actual != expected {
            return Err(std::io::Error::other(format!(
                "model.adapter_bank shared-base checksum {expected} does not match model.shared_base {actual}"
            ))
            .into());
        }
    }
    let vocabulary = bank.embed_dims()?;
    let identity_by_slug = adapter_bank_identity_by_slug(adapter_bank_path, vocabulary)?;
    let layout = NpaParameterRowLayout2d::new(&base.config);
    for example in examples {
        let identity = identity_by_slug
            .get(&example.slug)
            .copied()
            .ok_or_else(|| {
                std::io::Error::other(format!(
                    "adapter-bank example {} is absent from the bank source manifest",
                    example.slug
                ))
            })?;
        let mut condition = vec![0.0; vocabulary];
        condition[identity] = 1.0;
        let bank_adapter = bank.predict_adapter(&base.config, &condition)?;
        let canonical =
            layout.packed_to_canonical_adapter(&layout.adapter_to_packed(&bank_adapter)?)?;
        example.teacher_adapter = Some(canonical.to_parameter_vector());
    }
    Ok(())
}

fn validate_upstream_adapter_bank_contract(bank: &E2eHyperNpa2d) -> crate::AutomataResult<()> {
    if bank.adapter_output_bias_enabled() {
        return Err(crate::AutomataError::InvalidModel(
            "model.adapter_bank enables the per-image NPA output bias, which is incompatible with the official Growing 2D zero-output-bias contract; regenerate the adapter bank with adapter.output_bias=false"
                .to_string(),
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
struct AdapterBankSourceManifest {
    selected_sources: Vec<AdapterBankSourceIdentity>,
}

#[derive(Deserialize)]
struct AdapterBankSourceIdentity {
    slug: String,
}

fn adapter_bank_identity_by_slug(
    adapter_bank_path: &Path,
    vocabulary: usize,
) -> Result<BTreeMap<String, usize>, Box<dyn std::error::Error>> {
    let parent = adapter_bank_path.parent().ok_or_else(|| {
        std::io::Error::other(format!(
            "model.adapter_bank {} has no parent directory",
            adapter_bank_path.display()
        ))
    })?;
    let report_path = [
        parent.join("report.json"),
        parent
            .parent()
            .map(|root| root.join("report.json"))
            .unwrap_or_default(),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| {
        std::io::Error::other(format!(
            "model.adapter_bank {} requires its training report.json to preserve sample-ID row provenance",
            adapter_bank_path.display()
        ))
    })?;
    let manifest: AdapterBankSourceManifest =
        serde_json::from_slice(&std::fs::read(&report_path)?)?;
    if manifest.selected_sources.len() != vocabulary {
        return Err(std::io::Error::other(format!(
            "adapter-bank source manifest {} contains {} identities but the table vocabulary has {vocabulary}",
            report_path.display(),
            manifest.selected_sources.len(),
        ))
        .into());
    }
    let mut identities = BTreeMap::new();
    for (identity, source) in manifest.selected_sources.into_iter().enumerate() {
        if identities.insert(source.slug.clone(), identity).is_some() {
            return Err(std::io::Error::other(format!(
                "adapter-bank source manifest {} repeats slug {}",
                report_path.display(),
                source.slug,
            ))
            .into());
        }
    }
    Ok(identities)
}

fn ensure_holdout_teacher_free(
    examples: &[BurnE2eRolloutExample],
) -> Result<(), Box<dyn std::error::Error>> {
    let leaked = examples
        .iter()
        .filter(|example| example.teacher_adapter.is_some())
        .map(|example| example.slug.as_str())
        .collect::<Vec<_>>();
    if leaked.is_empty() {
        return Ok(());
    }
    Err(std::io::Error::other(format!(
        "holdout examples must not carry oracle adapter targets: {}",
        leaked.join(", ")
    ))
    .into())
}

fn exact_teacher_adapter(
    base: &NpaModel,
    target: &NpaModel,
    rank: usize,
    alpha: f32,
) -> crate::AutomataResult<crate::NpaLowRankAdapter> {
    let mut adapter = crate::NpaLowRankAdapter::exact_model_delta(base, target, rank, alpha)?;
    adapter.b1_delta = target
        .weights
        .b1
        .iter()
        .zip(&base.weights.b1)
        .map(|(target, base)| target - base)
        .collect();
    adapter.b2_delta = target
        .weights
        .b2
        .iter()
        .zip(&base.weights.b2)
        .map(|(target, base)| target - base)
        .collect();
    adapter.b1_delta_correction.clear();
    adapter.b2_delta_correction.clear();
    adapter.validate(&base.config)?;
    Ok(adapter)
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
    let pool_state_buffer_bytes = if report.training.use_particle_pool {
        report
            .training
            .pool_capacity
            .saturating_mul(report.rollout.particles)
            .saturating_mul(NpaConfig::growing_2d().state_dims)
            .saturating_mul(std::mem::size_of::<f32>())
    } else {
        0
    };
    const WGPU_MAX_SINGLE_POOL_BUFFER_BYTES: usize = 1usize << 31;
    if matches!(report.training.gpu_backend.as_str(), "burn-wgpu" | "wgpu")
        && pool_state_buffer_bytes >= WGPU_MAX_SINGLE_POOL_BUFFER_BYTES
    {
        return Err(std::io::Error::other(format!(
            "HyperNPA WGPU particle-pool state requires one {:.2} GiB buffer, but the current monolithic pool must stay below 2.00 GiB; reduce pool_capacity/pool_slots_per_example or implement sharded pool storage",
            bytes_to_gib(pool_state_buffer_bytes),
        ))
        .into());
    }
    let particle_pool_bytes = if report.training.use_particle_pool {
        report
            .training
            .pool_capacity
            .saturating_mul(report.rollout.particles)
            .saturating_mul(NpaConfig::growing_2d().state_dims.saturating_add(2))
            .saturating_mul(std::mem::size_of::<f32>())
    } else {
        0
    };
    let table_parameter_bytes = if report.adapter.generator == E2E_HYPER_ARCH_SAMPLE_ID_TABLE {
        let config = NpaConfig::growing_2d();
        NpaLowRankAdapter::parameter_count_for_config(&config, report.adapter.rank)
            .saturating_mul(report.source.selected_sources)
            .saturating_mul(std::mem::size_of::<f32>())
    } else {
        0
    };
    if let Some(budget_gb) = report.training.system_memory_budget_gb {
        let budget_bytes = (budget_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
        let projected_bytes = report
            .condition
            .projected_condition_load_peak_bytes_f32
            .saturating_add(table_parameter_bytes)
            .saturating_add(particle_pool_bytes)
            .saturating_add(1024 * 1024 * 1024);
        if projected_bytes > budget_bytes {
            return Err(std::io::Error::other(format!(
                "projected HyperNPA e2e host peak is {:.2} GiB including condition staging, adapter-table weights, {:.2} GiB particle-pool checkpoint readback, and 1.00 GiB overhead, above system_memory_budget_gb={budget_gb:.2}",
                bytes_to_gib(projected_bytes),
                bytes_to_gib(particle_pool_bytes),
            ))
            .into());
        }
    }
    if let Some(budget_gb) = report.training.gpu_memory_budget_gb {
        let budget_bytes = (budget_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
        let table_training_bytes = table_parameter_bytes.saturating_mul(4);
        let persistent_training_bytes = table_training_bytes.saturating_add(particle_pool_bytes);
        if persistent_training_bytes > budget_bytes {
            return Err(std::io::Error::other(format!(
                "HyperNPA persistent GPU state requires at least {:.2} GiB ({:.2} GiB adapter weights/gradients/AdamW plus {:.2} GiB particle pool), above gpu_memory_budget_gb={budget_gb:.2}; reduce source count/rank/pool depth or raise the budget intentionally",
                bytes_to_gib(persistent_training_bytes),
                bytes_to_gib(table_training_bytes),
                bytes_to_gib(particle_pool_bytes),
            ))
            .into());
        }
        if report.training.steps > 0 {
            let projection = projected_active_training_gpu_memory(report);
            if projection.total_bytes > budget_bytes {
                return Err(std::io::Error::other(format!(
                    "projected active HyperNPA GPU peak is {:.2} GiB ({:.2} GiB rollout graph, {:.2} GiB row-flow graph, {:.2} GiB trainable state, {:.2} GiB persistent caches/pool, and {:.2} GiB runtime reserve), above gpu_memory_budget_gb={budget_gb:.2}; reduce example_batch_size*rollouts_per_example, row-flow condition batch/capacity, rollout horizon, or use detached TBPTT",
                    bytes_to_gib(projection.total_bytes),
                    bytes_to_gib(projection.rollout_graph_bytes),
                    bytes_to_gib(projection.row_flow_graph_bytes),
                    bytes_to_gib(projection.trainable_state_bytes),
                    bytes_to_gib(projection.persistent_bytes),
                    bytes_to_gib(projection.runtime_reserve_bytes),
                ))
                .into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct E2eGpuMemoryProjection {
    rollout_graph_bytes: usize,
    row_flow_graph_bytes: usize,
    trainable_state_bytes: usize,
    persistent_bytes: usize,
    runtime_reserve_bytes: usize,
    total_bytes: usize,
}

fn projected_active_training_gpu_memory(report: &E2eRolloutReport) -> E2eGpuMemoryProjection {
    // Measured on the canonical growing-2D Burn/CUDA graph. This includes the
    // retained perception VJP state and the ordinary Burn autodiff graph.
    const ROLLOUT_BYTES_PER_PARTICLE_STEP: usize = 2_304;
    // Conservative batch-row workspace for retained targets, masks, losses,
    // and allocator pages that scale with independent trajectories. Older
    // dense graphs required a separate 32-trajectory hard cap. The fused
    // endpoint graph measured 27.4 GiB at 32 x 4096 x 96 in July 2026, while
    // this projection remains intentionally conservative at 58.5 GiB.
    const ROLLOUT_WORKSPACE_BYTES_PER_TRAJECTORY: usize = 384 * 1024 * 1024;
    // CubeCL's pooled allocator must overlap live graph generations and can
    // require a large contiguous page near the graph peak. A flat reserve hid
    // this scaling and admitted a 64 x 4096 x 96 graph that exhausted a 96 GiB
    // device. Reserve 50% of active graph storage, with a 4 GiB floor.
    const MIN_RUNTIME_RESERVE_BYTES: usize = 4 * 1024 * 1024 * 1024;
    const ACTIVE_GRAPH_RESERVE_DIVISOR: usize = 2;
    const TRAINABLE_TENSOR_COPIES: usize = 5; // value, tracked value, grad, Adam m, Adam v

    let rollout_batch = report
        .training
        .example_batch_size
        .saturating_mul(report.training.rollouts_per_example);
    let active_rollout_steps = if report.training.credit_assignment == "full-bptt" {
        report.rollout.steps
    } else {
        report.training.tbptt_chunk_steps.min(report.rollout.steps)
    };
    let rollout_graph_bytes = if is_rollout_free_amortization_distillation(report) {
        0
    } else {
        rollout_batch
            .saturating_mul(report.rollout.particles)
            .saturating_mul(active_rollout_steps)
            .saturating_mul(ROLLOUT_BYTES_PER_PARTICLE_STEP)
            .saturating_add(rollout_batch.saturating_mul(ROLLOUT_WORKSPACE_BYTES_PER_TRAJECTORY))
    };

    let row_flow_selected = matches!(
        report.adapter.generator.as_str(),
        E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW | E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW
    );
    let amortization_substrate_only = report.training.amortization_enabled
        && report.training.steps <= report.training.amortization_substrate_steps;
    let (row_flow_graph_bytes, row_flow_state_bytes) = if row_flow_selected {
        let npa = NpaConfig {
            hidden_dims: report.model.hidden_dims,
            ..NpaConfig::growing_2d()
        };
        let layout = NpaParameterRowLayout2d::new(&npa);
        let rows = layout.row_count();
        let width = report.adapter.flow_hidden;
        let heads = report.condition.token_attention_heads;
        let condition_tokens = report.condition.token_count;
        let ffn_dims = report.adapter.flow_ffn_dims;
        let velocity_evaluations = report
            .adapter
            .flow_sample_steps
            .saturating_mul(2)
            .saturating_add(usize::from(report.training.flow_matching_weight > 0.0))
            .saturating_add(usize::from(
                report.training.flow_self_rectification_weight > 0.0,
            ));
        let attention_values = heads
            .saturating_mul(rows)
            .saturating_mul(rows.saturating_add(condition_tokens));
        let block_values = rows.saturating_mul(
            width
                .saturating_mul(12)
                .saturating_add(ffn_dims.saturating_mul(2)),
        );
        let graph_bytes = if amortization_substrate_only {
            0
        } else {
            report
                .training
                .example_batch_size
                .saturating_mul(report.adapter.flow_layers)
                .saturating_mul(velocity_evaluations)
                .saturating_mul(attention_values.saturating_add(block_values))
                .saturating_mul(std::mem::size_of::<f32>())
        };

        let per_layer_parameters = width
            .saturating_mul(width)
            .saturating_mul(17)
            .saturating_add(width.saturating_mul(ffn_dims).saturating_mul(2));
        let fixed_parameters = report
            .condition
            .embed_dims
            .saturating_mul(width)
            .saturating_add(layout.max_row_dims().saturating_mul(width))
            .saturating_add(rows.saturating_mul(width))
            .saturating_add(width.saturating_mul(width).saturating_mul(2));
        let parameter_bytes = report
            .adapter
            .flow_layers
            .saturating_mul(per_layer_parameters)
            .saturating_add(fixed_parameters)
            .saturating_mul(std::mem::size_of::<f32>());
        (
            graph_bytes,
            parameter_bytes.saturating_mul(if amortization_substrate_only {
                1
            } else {
                TRAINABLE_TENSOR_COPIES
            }),
        )
    } else {
        (0, 0)
    };
    let trainable_state_bytes =
        row_flow_state_bytes.saturating_add(report.training.amortization_optimizer_bytes_f32);

    let particle_pool_bytes = if report.training.use_particle_pool {
        report
            .training
            .pool_capacity
            .saturating_mul(report.rollout.particles)
            .saturating_mul(NpaConfig::growing_2d().state_dims.saturating_add(2))
            .saturating_mul(std::mem::size_of::<f32>())
    } else {
        0
    };
    let condition_cache_bytes =
        if report.condition.device_cache_plan == "complete-set-device-resident" {
            report.condition.selected_feature_cache_bytes_f32
        } else {
            report.condition.dino_batch_input_bytes_f32
        };
    let target_values_per_example = report
        .target
        .loss_image_size
        .saturating_mul(report.target.loss_image_size)
        .saturating_mul(5)
        .saturating_add(2)
        .saturating_add(report.target.points.saturating_mul(2));
    let target_bytes_per_example =
        target_values_per_example.saturating_mul(std::mem::size_of::<f32>());
    let complete_target_cache_bytes = report
        .source
        .train_examples
        .saturating_mul(target_bytes_per_example);
    let target_cache_bytes = if report.training.target_device_cache_max_bytes > 0
        && complete_target_cache_bytes <= report.training.target_device_cache_max_bytes
    {
        complete_target_cache_bytes
    } else {
        report
            .training
            .example_batch_size
            .saturating_mul(target_bytes_per_example)
    };
    let persistent_bytes = particle_pool_bytes
        .saturating_add(condition_cache_bytes)
        .saturating_add(target_cache_bytes);
    let active_graph_bytes = rollout_graph_bytes.saturating_add(row_flow_graph_bytes);
    let runtime_reserve_bytes = MIN_RUNTIME_RESERVE_BYTES.max(
        active_graph_bytes.saturating_add(ACTIVE_GRAPH_RESERVE_DIVISOR - 1)
            / ACTIVE_GRAPH_RESERVE_DIVISOR,
    );
    let total_bytes = rollout_graph_bytes
        .saturating_add(row_flow_graph_bytes)
        .saturating_add(trainable_state_bytes)
        .saturating_add(persistent_bytes)
        .saturating_add(runtime_reserve_bytes);
    E2eGpuMemoryProjection {
        rollout_graph_bytes,
        row_flow_graph_bytes,
        trainable_state_bytes,
        persistent_bytes,
        runtime_reserve_bytes,
        total_bytes,
    }
}

fn is_rollout_free_amortization_distillation(report: &E2eRolloutReport) -> bool {
    report.training.task_loss_weight == 0.0
        && report.training.adapter_teacher_weight == 0.0
        && report.training.flow_matching_weight == 0.0
        && report.training.amortization_enabled
        && (report.training.amortization_distillation_weight > 0.0
            || report.training.flow_self_rectification_weight > 0.0)
}

fn load_burn_e2e_rollout_examples(
    report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    match parse_rollout_condition_encoder(Some(report.condition.encoder))? {
        RolloutConditionEncoder::DinoVitsFullTokens
        | RolloutConditionEncoder::DinoVitsTokenGrid => {
            load_burn_e2e_rollout_examples_with_online_dino(report)
        }
        RolloutConditionEncoder::SampleIdOneHot => {
            load_burn_e2e_rollout_examples_with_sample_ids(report)
        }
    }
}

#[cfg(feature = "dino")]
fn load_burn_e2e_rollout_examples_with_sample_ids(
    report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    if report.condition.token_count != 1 {
        return Err(std::io::Error::other(format!(
            "sample-id-onehot condition expects token_count=1, got {}",
            report.condition.token_count
        ))
        .into());
    }
    let embed_dims = report.condition.embed_dims;
    if report.condition.sample_ids.len() != report.selected_sources.len() {
        return Err(std::io::Error::other(format!(
            "sample-id-onehot report has {} mapped IDs for {} selected sources",
            report.condition.sample_ids.len(),
            report.selected_sources.len()
        ))
        .into());
    }
    let mut examples = Vec::with_capacity(report.selected_sources.len());
    eprintln!(
        "building {} sample-id one-hot HyperNPA conditions (embed_dims={embed_dims})",
        report.selected_sources.len()
    );
    for (idx, entry) in report.selected_sources.iter().enumerate() {
        let mut condition_features = vec![0.0_f32; embed_dims];
        let sample_id = report.condition.sample_ids[idx];
        let value = condition_features.get_mut(sample_id).ok_or_else(|| {
            std::io::Error::other(format!(
                "sample ID {sample_id} is outside condition embed_dims={embed_dims}"
            ))
        })?;
        *value = 1.0;
        let target = load_target_image_2d_adaptive(
            Path::new(&entry.condition_path),
            report.target.threshold,
            report.target.points,
            report.target.image_size,
        )?;
        examples.push(BurnE2eRolloutExample {
            slug: entry.slug.clone(),
            target,
            condition_path: Some(PathBuf::from(&entry.condition_path)),
            dino_model_path: None,
            condition_features,
            token_count: report.condition.token_count,
            embed_dims,
            particle_count: entry.particles.unwrap_or(report.rollout.particles),
            update_prob: entry.update_prob.unwrap_or(report.rollout.update_prob),
            seed_scale: entry.seed_scale.unwrap_or_else(|| {
                report
                    .rollout
                    .seed_scale
                    .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(report.preset))
            }),
            teacher_adapter: None,
        });
    }
    Ok(examples)
}

#[cfg(not(feature = "dino"))]
fn load_burn_e2e_rollout_examples_with_sample_ids(
    _report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "sample-id-onehot HyperNPA e2e training currently requires the dino feature for image loading",
    )
    .into())
}

#[cfg(feature = "dino")]
fn load_burn_e2e_rollout_examples_with_online_dino(
    report: &E2eRolloutReport,
) -> Result<Vec<BurnE2eRolloutExample>, Box<dyn std::error::Error>> {
    let _dino_model = report.condition.dino_model.as_ref().ok_or_else(|| {
        std::io::Error::other("condition.dino_model is required for e2e DINO training")
    })?;
    let expected_feature_dims = report
        .condition
        .token_count
        .saturating_mul(report.condition.embed_dims);
    if report.condition.device_cache_plan == "complete-set-device-resident" {
        eprintln!(
            "deferring image decode; {} DINO conditions will be streamed through GPU DINO into a {:.2} GiB bounded device token cache",
            report.selected_sources.len(),
            report.condition.selected_feature_cache_gib_f32,
        );
    } else {
        eprintln!(
            "deferring {} DINO conditions to on-demand GPU batches (encoded set {:.2} GiB exceeds the bounded cache plan; input batch {:.2} GiB)",
            report.selected_sources.len(),
            report.condition.selected_feature_cache_gib_f32,
            report.condition.dino_batch_input_gib_f32,
        );
    }
    if expected_feature_dims == 0 {
        return Err(std::io::Error::other("DINO feature dimensions must be non-zero").into());
    }
    let total = report.selected_sources.len();
    let progress_interval = (total / 100).clamp(1, 1_000);
    let completed = AtomicUsize::new(0);
    let default_seed_scale = report
        .rollout
        .seed_scale
        .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(report.preset));
    let dino_model = report.condition.dino_model.clone();
    let examples = report
        .selected_sources
        .par_iter()
        .map(|entry| -> Result<BurnE2eRolloutExample, String> {
            let path = Path::new(&entry.condition_path);
            let target = load_target_image_2d_adaptive(
                path,
                report.target.threshold,
                report.target.points,
                report.target.image_size,
            )
            .map_err(|err| {
                format!(
                    "failed to extract target image {} for online DINO HyperNPA: {err}",
                    path.display()
                )
            })?;
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done == total || done.is_multiple_of(progress_interval) {
                eprintln!("loaded online DINO target {done}/{total}");
            }
            Ok(BurnE2eRolloutExample {
                slug: entry.slug.clone(),
                target,
                condition_path: Some(PathBuf::from(&entry.condition_path)),
                dino_model_path: dino_model.as_ref().map(PathBuf::from),
                condition_features: Vec::new(),
                token_count: report.condition.token_count,
                embed_dims: report.condition.embed_dims,
                particle_count: entry.particles.unwrap_or(report.rollout.particles),
                update_prob: entry.update_prob.unwrap_or(report.rollout.update_prob),
                seed_scale: entry.seed_scale.unwrap_or(default_seed_scale),
                teacher_adapter: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(std::io::Error::other)?;
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

fn base_adamw_from_report(report: &E2eRolloutReport) -> AdamWConfig {
    AdamWConfig {
        learning_rate: report.optimizer.base_learning_rate,
        weight_decay: report.optimizer.base_weight_decay,
        grad_clip_norm: report.optimizer.base_grad_clip_norm,
        beta1: report.optimizer.adam_beta1,
        beta2: report.optimizer.adam_beta2,
        epsilon: report.optimizer.adam_epsilon,
    }
}

fn generator_adamw_from_report(report: &E2eRolloutReport) -> AdamWConfig {
    AdamWConfig {
        learning_rate: report.optimizer.generator_learning_rate,
        weight_decay: report.optimizer.generator_weight_decay,
        grad_clip_norm: report.optimizer.generator_grad_clip_norm,
        beta1: report.optimizer.adam_beta1,
        beta2: report.optimizer.adam_beta2,
        epsilon: report.optimizer.adam_epsilon,
    }
}

fn evaluate_e2e_rollout_gates(
    report: &E2eRolloutReport,
    training: &BurnE2eRolloutOutput,
) -> Vec<E2eRolloutGateResultReport> {
    let mut results = Vec::new();

    if let Some(threshold) = report.gates.min_median_particle_steps_per_sec {
        let observed = metric_f64(&training.metrics, "median_reported_particle_steps_per_sec");
        results.push(gate_result(
            "min_median_particle_steps_per_sec",
            observed.is_some_and(|value| value >= threshold),
            json_number_or_null(observed),
            serde_json::json!(threshold),
            match observed {
                Some(value) => format!(
                    "median reported particle throughput {value:.3} particle-steps/s; required >= {threshold:.3}"
                ),
                None => {
                    "median reported particle throughput was not present in training metrics"
                        .to_string()
                }
            },
        ));
    }

    if let Some(threshold) = report.gates.max_quality_validation_evaluations {
        let observed = metric_usize(&training.metrics, "quality_validation_evaluations");
        results.push(gate_result(
            "max_quality_validation_evaluations",
            observed.is_some_and(|value| value <= threshold),
            observed
                .map(|value| serde_json::json!(value))
                .unwrap_or(serde_json::Value::Null),
            serde_json::json!(threshold),
            match observed {
                Some(value) => {
                    format!("quality validation ran {value} times; required <= {threshold}")
                }
                None => "quality validation evaluation count was not present in training metrics"
                    .to_string(),
            },
        ));
    }

    if let Some(threshold) = report.gates.max_quality_validation_elapsed_fraction {
        let elapsed_ms = metric_f64(&training.metrics, "quality_validation_elapsed_ms");
        let training_ms = metric_f64(&training.metrics, "burn_training_ms");
        let observed = elapsed_ms
            .zip(training_ms)
            .and_then(|(elapsed_ms, training_ms)| {
                (training_ms > 0.0).then_some(elapsed_ms / training_ms)
            });
        results.push(gate_result(
            "max_quality_validation_elapsed_fraction",
            observed.is_some_and(|value| value <= threshold),
            json_number_or_null(observed),
            serde_json::json!(threshold),
            match observed {
                Some(value) => format!(
                    "quality validation consumed {:.3}% of Burn training time; required <= {:.3}%",
                    value * 100.0,
                    threshold * 100.0
                ),
                None => "quality validation or Burn training elapsed metrics were not present"
                    .to_string(),
            },
        ));
    }

    if let Some(threshold) = report.gates.min_final_mean_render_rgb_psnr_db {
        let observed = training
            .quality_validation
            .as_ref()
            .map(|quality| quality.mean_render_rgb_psnr_db);
        results.push(gate_result(
            "min_final_mean_render_rgb_psnr_db",
            observed.is_some_and(|value| value >= threshold),
            observed
                .map(|value| serde_json::json!(value))
                .unwrap_or(serde_json::Value::Null),
            serde_json::json!(threshold),
            match observed {
                Some(value) => {
                    format!(
                        "final mean render RGB PSNR {value:.3} dB; required >= {threshold:.3} dB"
                    )
                }
                None => "final quality validation was not present".to_string(),
            },
        ));
    }

    if let Some(threshold) = report.gates.min_final_p10_composited_rgb_psnr_db {
        let observed = training
            .quality_validation
            .as_ref()
            .map(|quality| quality.p10_composited_rgb_psnr_db);
        results.push(minimum_quality_gate(
            "min_final_p10_composited_rgb_psnr_db",
            observed,
            threshold,
            "final p10 composited RGB PSNR",
            "dB",
        ));
    }

    if let Some(threshold) = report
        .gates
        .min_final_condition_shuffle_composited_psnr_gap_db
    {
        let observed = training
            .quality_validation
            .as_ref()
            .and_then(|quality| quality.condition_shuffle_composited_psnr_gap_db);
        results.push(minimum_quality_gate(
            "min_final_condition_shuffle_composited_psnr_gap_db",
            observed,
            threshold,
            "final correct-vs-shuffled condition composited PSNR gap",
            "dB",
        ));
    }

    if let Some(threshold) = report
        .gates
        .min_final_generated_adapter_composited_psnr_gain_db
    {
        let observed = training
            .quality_validation
            .as_ref()
            .map(|quality| quality.generated_adapter_composited_psnr_gain_db);
        results.push(minimum_quality_gate(
            "min_final_generated_adapter_composited_psnr_gain_db",
            observed,
            threshold,
            "final generated-NPA gain over the shared trunk",
            "dB",
        ));
    }

    if let Some(threshold) = report.gates.max_final_p90_gap_to_matched_oracle_db {
        results.push(matched_oracle_gate(report, training, threshold));
    }

    if report
        .gates
        .require_validation_interval_at_least_report_interval
    {
        let passed = report.validation.interval >= report.training.report_interval;
        results.push(gate_result(
            "require_validation_interval_at_least_report_interval",
            passed,
            serde_json::json!(report.validation.interval),
            serde_json::json!(report.training.report_interval),
            format!(
                "validation interval {} must be >= report interval {}",
                report.validation.interval, report.training.report_interval
            ),
        ));
    }

    results
}

fn minimum_quality_gate(
    gate: &'static str,
    observed: Option<f32>,
    threshold: f32,
    label: &str,
    unit: &str,
) -> E2eRolloutGateResultReport {
    gate_result(
        gate,
        observed.is_some_and(|value| value >= threshold),
        observed
            .map(|value| serde_json::json!(value))
            .unwrap_or(serde_json::Value::Null),
        serde_json::json!(threshold),
        observed.map_or_else(
            || format!("{label} was not present in final quality validation"),
            |value| format!("{label} {value:.3} {unit}; required >= {threshold:.3} {unit}"),
        ),
    )
}

fn matched_oracle_gate(
    report: &E2eRolloutReport,
    training: &BurnE2eRolloutOutput,
    threshold: f32,
) -> E2eRolloutGateResultReport {
    let failed = |message: String, observed: serde_json::Value| {
        gate_result(
            "max_final_p90_gap_to_matched_oracle_db",
            false,
            observed,
            serde_json::json!(threshold),
            message,
        )
    };
    let Some(current) = training.quality_validation.as_ref() else {
        return failed(
            "final quality validation was not present for matched-oracle comparison".to_string(),
            serde_json::Value::Null,
        );
    };
    let Some(path) = report.validation.oracle_report.as_deref() else {
        return failed(
            "validation.oracle_report is required by the matched-oracle gate".to_string(),
            serde_json::Value::Null,
        );
    };
    let value = match std::fs::read_to_string(path)
        .map_err(|err| err.to_string())
        .and_then(|text| {
            serde_json::from_str::<serde_json::Value>(&text).map_err(|err| err.to_string())
        }) {
        Ok(value) => value,
        Err(err) => {
            return failed(
                format!("failed to read matched oracle report {path}: {err}"),
                serde_json::Value::Null,
            );
        }
    };
    let oracle = value
        .pointer("/training_result/quality_validation")
        .filter(|value| value.is_object())
        .or_else(|| {
            value
                .pointer("/quality_validation")
                .filter(|value| value.is_object())
        });
    let Some(oracle) = oracle else {
        return failed(
            format!("matched oracle report {path} has no quality validation object"),
            serde_json::Value::Null,
        );
    };
    let comparison = match compare_matched_oracle_quality(
        current.particle_count,
        current.rollout_steps,
        current.seed,
        current.update_prob,
        &current.entries,
        oracle,
    ) {
        Ok(comparison) => comparison,
        Err(err) => {
            return failed(
                format!("matched oracle report {path} is invalid: {err}"),
                serde_json::Value::Null,
            );
        }
    };
    let passed = comparison.contract_matched
        && comparison.all_oracles_matched
        && comparison
            .p90_gap_db
            .is_some_and(|value| value <= threshold);
    gate_result(
        "max_final_p90_gap_to_matched_oracle_db",
        passed,
        serde_json::json!({
            "report": path,
            "contract_matched": comparison.contract_matched,
            "all_oracles_matched": comparison.all_oracles_matched,
            "oracle_entries": comparison.oracle_entries,
            "matched_entries": comparison.matched_entries,
            "mean_gap_db": comparison.mean_gap_db,
            "p90_gap_db": comparison.p90_gap_db,
        }),
        serde_json::json!(threshold),
        if let Some(p90_gap) = comparison.p90_gap_db {
            format!(
                "matched-oracle p90 PSNR gap {p90_gap:.3} dB over {} samples; required <= {threshold:.3} dB with identical particles, horizon, seed, and update probability",
                comparison.matched_entries,
            )
        } else {
            "matched-oracle comparison produced no matched per-sample PSNR values".to_string()
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MatchedOracleComparison {
    contract_matched: bool,
    all_oracles_matched: bool,
    oracle_entries: usize,
    matched_entries: usize,
    mean_gap_db: Option<f32>,
    p90_gap_db: Option<f32>,
}

fn compare_matched_oracle_quality(
    particle_count: usize,
    rollout_steps: usize,
    seed: u64,
    update_prob: f32,
    entries: &[BurnE2eRolloutQualityEntry],
    oracle: &serde_json::Value,
) -> Result<MatchedOracleComparison, &'static str> {
    let oracle_particles = oracle
        .get("particle_count")
        .and_then(serde_json::Value::as_u64);
    let oracle_steps = oracle
        .get("rollout_steps")
        .and_then(serde_json::Value::as_u64);
    let oracle_seed = oracle.get("seed").and_then(serde_json::Value::as_u64);
    let oracle_update_prob = oracle
        .get("update_prob")
        .and_then(serde_json::Value::as_f64);
    let contract_matched = oracle_particles == Some(particle_count as u64)
        && oracle_steps == Some(rollout_steps as u64)
        && oracle_seed == Some(seed)
        && oracle_update_prob.is_some_and(|value| (value - update_prob as f64).abs() <= 1.0e-6);
    let oracle_entries = oracle
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .ok_or("missing per-sample entries")?;
    let mut gaps = Vec::with_capacity(oracle_entries.len());
    let mut malformed_entries = 0usize;
    for oracle_entry in oracle_entries {
        let Some(slug) = oracle_entry.get("slug").and_then(serde_json::Value::as_str) else {
            malformed_entries += 1;
            continue;
        };
        let Some(oracle_psnr) = oracle_entry
            .get("composited_rgb_psnr_db")
            .and_then(serde_json::Value::as_f64)
        else {
            malformed_entries += 1;
            continue;
        };
        if let Some(generated) = entries.iter().find(|entry| entry.slug == slug) {
            gaps.push(oracle_psnr as f32 - generated.composited_rgb_psnr_db);
        }
    }
    gaps.sort_by(f32::total_cmp);
    Ok(MatchedOracleComparison {
        contract_matched,
        all_oracles_matched: !oracle_entries.is_empty()
            && malformed_entries == 0
            && gaps.len() == oracle_entries.len(),
        oracle_entries: oracle_entries.len(),
        matched_entries: gaps.len(),
        mean_gap_db: (!gaps.is_empty()).then(|| gaps.iter().sum::<f32>() / gaps.len() as f32),
        p90_gap_db: percentile(&gaps, 0.9),
    })
}

fn percentile(sorted: &[f32], quantile: f32) -> Option<f32> {
    if sorted.is_empty() {
        return None;
    }
    let position = quantile.clamp(0.0, 1.0) * sorted.len().saturating_sub(1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let blend = position - lower as f32;
    Some(sorted[lower] * (1.0 - blend) + sorted[upper] * blend)
}

fn gate_result(
    gate: &'static str,
    passed: bool,
    observed: serde_json::Value,
    threshold: serde_json::Value,
    message: String,
) -> E2eRolloutGateResultReport {
    E2eRolloutGateResultReport {
        gate,
        passed,
        observed,
        threshold,
        message,
    }
}

fn metric_f64(metrics: &serde_json::Value, key: &str) -> Option<f64> {
    metrics.get(key).and_then(serde_json::Value::as_f64)
}

fn metric_usize(metrics: &serde_json::Value, key: &str) -> Option<usize> {
    metrics
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn json_number_or_null(value: Option<f64>) -> serde_json::Value {
    value
        .map(|value| serde_json::json!(value))
        .unwrap_or(serde_json::Value::Null)
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
        "sample-id-onehot" | "sample-id" | "onehot-sample-id" | "learned-id" => {
            Ok(RolloutConditionEncoder::SampleIdOneHot)
        }
        other => Err(std::io::Error::other(format!(
            "unknown condition.encoder {other:?}; expected dino-vits-full-tokens, dino-vits-token-grid, or sample-id-onehot"
        ))
        .into()),
    }
}

fn normalize_validation_horizons(
    configured: Option<&[usize]>,
    requested_steps: usize,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut horizons = configured.unwrap_or_default().to_vec();
    if horizons.contains(&0) {
        return Err(std::io::Error::other(
            "validation.horizons must contain only non-zero rollout steps",
        )
        .into());
    }
    horizons.push(requested_steps.max(1));
    horizons.sort_unstable();
    horizons.dedup();
    if horizons.len() > MAX_VALIDATION_HORIZONS {
        return Err(std::io::Error::other(format!(
            "validation.horizons supports at most {MAX_VALIDATION_HORIZONS} distinct steps, got {}",
            horizons.len()
        ))
        .into());
    }
    Ok(horizons)
}

fn optimizer_steps_for_exposure(
    trajectories_per_example: u64,
    train_examples: usize,
    effective_batch_size: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let requested = (trajectories_per_example as u128)
        .checked_mul(train_examples.max(1) as u128)
        .ok_or_else(|| std::io::Error::other("trajectory exposure count overflowed"))?;
    let steps = requested.div_ceil(effective_batch_size.max(1) as u128);
    usize::try_from(steps)
        .map_err(|_| std::io::Error::other("optimizer step count exceeds usize").into())
}

fn parse_e2e_lr_schedule(value: Option<&str>) -> Result<E2eLrSchedule, Box<dyn std::error::Error>> {
    match value.unwrap_or("constant") {
        "constant" | "none" => Ok(E2eLrSchedule::Constant),
        "cosine" | "cosine_decay" | "cosine-decay" => Ok(E2eLrSchedule::Cosine),
        "linear" | "linear_decay" | "linear-decay" => Ok(E2eLrSchedule::Linear),
        "upstream-growing" | "upstream_growing" => Ok(E2eLrSchedule::UpstreamGrowing),
        other => Err(std::io::Error::other(format!(
            "invalid optimizer.lr_schedule `{other}`; use constant, cosine, linear, or upstream-growing"
        ))
        .into()),
    }
}

fn parse_e2e_tbptt_loss_mode(
    value: Option<&str>,
    legacy_final_only: bool,
) -> Result<E2eTbpttLossMode, Box<dyn std::error::Error>> {
    let Some(value) = value else {
        return Ok(if legacy_final_only {
            E2eTbpttLossMode::FinalOnly
        } else {
            E2eTbpttLossMode::AllChunks
        });
    };
    match value {
        "all-chunks" | "all_chunks" | "all" => Ok(E2eTbpttLossMode::AllChunks),
        "final-only" | "final_only" | "final" => Ok(E2eTbpttLossMode::FinalOnly),
        "endpoint-weighted" | "endpoint_weighted" | "weighted" => {
            Ok(E2eTbpttLossMode::EndpointWeighted)
        }
        other => Err(std::io::Error::other(format!(
            "invalid training.tbptt_loss_mode `{other}`; use all-chunks, final-only, or endpoint-weighted"
        ))
        .into()),
    }
}

fn parse_e2e_credit_assignment(
    value: Option<&str>,
) -> Result<E2eCreditAssignment, Box<dyn std::error::Error>> {
    match value.unwrap_or("full-bptt") {
        "full-bptt" | "full_bptt" | "full" => Ok(E2eCreditAssignment::FullBptt),
        "detached-tbptt" | "detached_tbptt" | "tbptt" => Ok(E2eCreditAssignment::DetachedTbptt),
        other => Err(std::io::Error::other(format!(
            "invalid training.credit_assignment `{other}`; use full-bptt or detached-tbptt"
        ))
        .into()),
    }
}

fn parse_e2e_adapter_teacher_objective(
    value: Option<&str>,
) -> Result<E2eAdapterTeacherObjective, Box<dyn std::error::Error>> {
    match value.unwrap_or("parameter-mse") {
        "parameter-mse" | "parameter_mse" => Ok(E2eAdapterTeacherObjective::ParameterMse),
        "functional-mse" | "functional_mse" => Ok(E2eAdapterTeacherObjective::FunctionalMse),
        "hybrid" | "functional-plus-parameter-mse" | "functional_plus_parameter_mse" => {
            Ok(E2eAdapterTeacherObjective::Hybrid)
        }
        other => Err(std::io::Error::other(format!(
            "unsupported training.adapter_teacher_objective {other:?}; expected parameter-mse, functional-mse, or hybrid"
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

fn parse_preset(value: Option<&str>) -> Result<AutomataPreset, Box<dyn std::error::Error>> {
    match value.unwrap_or("growing-2d") {
        "growing-2d" | "growing2d" => Ok(AutomataPreset::Growing2d),
        "texture-2d" | "texture2d" => Ok(AutomataPreset::Texture2d),
        "growing-3d-gs" | "growing3dgs" | "growing-3dgs" => Ok(AutomataPreset::Growing3dGs),
        "point-mnist" | "pointmnist" => Ok(AutomataPreset::PointMnist),
        other => Err(std::io::Error::other(format!(
            "invalid preset `{other}` in HyperNPA e2e rollout config"
        ))
        .into()),
    }
}

fn preset_name(preset: AutomataPreset) -> &'static str {
    match preset {
        AutomataPreset::Growing2d => "growing-2d",
        AutomataPreset::Texture2d => "texture-2d",
        AutomataPreset::Growing3dGs => "growing-3d-gs",
        AutomataPreset::PointMnist => "point-mnist",
    }
}

fn parse_catalog_group(
    field: &str,
    value: Option<&str>,
) -> Result<Option<E2eCatalogGroup>, Box<dyn std::error::Error>> {
    value
        .map(|value| match value {
            "growing" => Ok(E2eCatalogGroup::Growing),
            "texture" => Ok(E2eCatalogGroup::Texture),
            "all" => Ok(E2eCatalogGroup::All),
            other => Err(std::io::Error::other(format!(
                "invalid {field} `{other}` in HyperNPA e2e rollout config; use growing, texture, or all"
            ))),
        })
        .transpose()
        .map_err(Into::into)
}

fn parse_omnisvg_dataset(
    field: &str,
    value: Option<&str>,
) -> Result<Option<OmniSvgDataset>, Box<dyn std::error::Error>> {
    value
        .map(|value| match value {
            "mmsvg-illustration" | "mmsvg_illustration" | "illustration" => {
                Ok(OmniSvgDataset::MmsvgIllustration)
            }
            "mmsvg-icon" | "mmsvg_icon" | "icon" => Ok(OmniSvgDataset::MmsvgIcon),
            other => Err(std::io::Error::other(format!(
                "invalid {field} `{other}` in HyperNPA e2e rollout config; use mmsvg-illustration or mmsvg-icon"
            ))),
        })
        .transpose()
        .map_err(Into::into)
}

fn parse_seed_mode(value: Option<&str>) -> Result<ParticleSeed, Box<dyn std::error::Error>> {
    match value.unwrap_or("uniform-circle") {
        "gaussian" => Ok(ParticleSeed::Gaussian),
        "uniform" => Ok(ParticleSeed::Uniform),
        "uniform-circle" | "circle" => Ok(ParticleSeed::UniformCircle),
        "uv-torus-3d" | "torus" => Ok(ParticleSeed::UvTorus3d),
        "uv-torus-dense-3d" | "torus-dense" | "dense-torus" => Ok(ParticleSeed::UvTorusDense3d),
        "growth-3d" | "growth" | "random-ball-growth-3d" => Ok(ParticleSeed::Growth3d),
        "substrate-growth-3d" | "substrate-growth" | "growth-substrate" => {
            Ok(ParticleSeed::SubstrateGrowth3d)
        }
        "local-growth-3d" | "local-growth" | "growth-local" => Ok(ParticleSeed::LocalGrowth3d),
        "local-substrate-growth-3d" | "local-substrate" | "growth-local-substrate" => {
            Ok(ParticleSeed::LocalSubstrateGrowth3d)
        }
        "torus-field-dense-3d" | "torus-field" | "field-torus" => {
            Ok(ParticleSeed::TorusFieldDense3d)
        }
        "teapot-field-dense-3d" | "teapot-field" | "field-teapot" => {
            Ok(ParticleSeed::TeapotFieldDense3d)
        }
        "torus-growth-3d" | "torus-growth" | "growth-torus" => Ok(ParticleSeed::TorusGrowth3d),
        "teapot-growth-3d" | "teapot-growth" | "growth-teapot" => Ok(ParticleSeed::TeapotGrowth3d),
        "torus-substrate-growth-3d" | "torus-substrate" | "substrate-torus" => {
            Ok(ParticleSeed::TorusSubstrateGrowth3d)
        }
        "teapot-substrate-growth-3d" | "teapot-substrate" | "substrate-teapot" => {
            Ok(ParticleSeed::TeapotSubstrateGrowth3d)
        }
        "torus-local-growth-3d" | "torus-local-growth" | "local-growth-torus" => {
            Ok(ParticleSeed::TorusLocalGrowth3d)
        }
        "teapot-local-growth-3d" | "teapot-local-growth" | "local-growth-teapot" => {
            Ok(ParticleSeed::TeapotLocalGrowth3d)
        }
        "torus-local-substrate-growth-3d" | "torus-local-substrate" | "local-substrate-torus" => {
            Ok(ParticleSeed::TorusLocalSubstrateGrowth3d)
        }
        "teapot-local-substrate-growth-3d"
        | "teapot-local-substrate"
        | "local-substrate-teapot" => Ok(ParticleSeed::TeapotLocalSubstrateGrowth3d),
        "torus-morphogen-dense-3d" | "torus-morphogen" | "morphogen-torus" => {
            Ok(ParticleSeed::TorusMorphogenDense3d)
        }
        "teapot-morphogen-dense-3d" | "teapot-morphogen" | "morphogen-teapot" => {
            Ok(ParticleSeed::TeapotMorphogenDense3d)
        }
        other => Err(std::io::Error::other(format!(
            "invalid rollout.seed_mode `{other}` in HyperNPA e2e rollout config"
        ))
        .into()),
    }
}

fn omnisvg_source_report(
    config: Option<OmniSvgSourceConfig<'_>>,
) -> Option<E2eOmniSvgSourceReport> {
    config.map(|config| E2eOmniSvgSourceReport {
        dataset: config.dataset,
        dataset_id: config.dataset.dataset_id().to_string(),
        split: config.split.to_string(),
        cache_dir: config.cache_dir.display().to_string(),
        offset: config.offset,
        limit: config.limit,
        page_size: config.page_size,
        download: config.download,
        refresh: config.refresh,
        token_env: config.token_env.to_string(),
    })
}

fn resolve_e2e_splits(
    sources: &[Hyper2dScratchSource],
    holdout_targets: &[String],
    holdout_stride: usize,
    holdout_offset: usize,
) -> Result<Vec<Hyper2dE2eSplit>, Box<dyn std::error::Error>> {
    if holdout_stride == 0 && holdout_offset != 0 {
        return Err(std::io::Error::other("holdout_offset requires holdout_stride > 0").into());
    }
    let requested = holdout_targets
        .iter()
        .map(|target| sources::sanitize_slug(target))
        .collect::<BTreeSet<_>>();
    let matched = sources
        .iter()
        .filter(|source| requested.contains(&sources::sanitize_slug(&source.slug)))
        .map(|source| sources::sanitize_slug(&source.slug))
        .collect::<BTreeSet<_>>();
    if let Some(missing) = requested.iter().find(|target| !matched.contains(*target)) {
        return Err(std::io::Error::other(format!(
            "holdout target {missing} did not match any selected source"
        ))
        .into());
    }

    let splits = sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            let explicit_holdout = requested.contains(&sources::sanitize_slug(&source.slug));
            let strided_holdout =
                holdout_stride > 0 && idx % holdout_stride == holdout_offset % holdout_stride;
            if explicit_holdout || strided_holdout {
                Hyper2dE2eSplit::Holdout
            } else {
                Hyper2dE2eSplit::Train
            }
        })
        .collect::<Vec<_>>();
    if !splits.iter().any(|split| split.is_train()) {
        return Err(std::io::Error::other(
            "HyperNPA e2e rollout split produced no training targets",
        )
        .into());
    }
    Ok(splits)
}

#[allow(clippy::too_many_arguments)]
fn target2d_loss_config(
    image_size: usize,
    splat_sigma: f32,
    center_loss: bool,
    splat_loss_weight: f32,
    color_loss_weight: f32,
    density_loss_weight: f32,
    background_density_loss_weight: f32,
    foreground_density_loss_weight: f32,
    composited_rgb_loss_weight: f32,
    displacement_regularizer_weight: f32,
    overflow_regularizer_weight: f32,
    bound_regularizer_weight: f32,
) -> Result<Target2dLossConfig, Box<dyn std::error::Error>> {
    if image_size == 0 {
        return Err(
            std::io::Error::other("target loss image_size must be greater than zero").into(),
        );
    }
    for (name, value) in [
        ("splat_sigma", splat_sigma),
        ("splat_loss_weight", splat_loss_weight),
        ("color_loss_weight", color_loss_weight),
        ("density_loss_weight", density_loss_weight),
        (
            "background_density_loss_weight",
            background_density_loss_weight,
        ),
        (
            "foreground_density_loss_weight",
            foreground_density_loss_weight,
        ),
        ("composited_rgb_loss_weight", composited_rgb_loss_weight),
        (
            "displacement_regularizer_weight",
            displacement_regularizer_weight,
        ),
        ("overflow_regularizer_weight", overflow_regularizer_weight),
        ("bound_regularizer_weight", bound_regularizer_weight),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(
                std::io::Error::other(format!("{name} must be finite and non-negative")).into(),
            );
        }
    }
    Ok(Target2dLossConfig {
        image_size,
        sigma: splat_sigma,
        center: center_loss,
        splat_loss_weight,
        color_loss_weight,
        density_loss_weight,
        background_density_loss_weight,
        foreground_density_loss_weight,
        composited_rgb_loss_weight,
        displacement_regularizer_weight,
        overflow_regularizer_weight,
        bound_regularizer_weight,
        ..Target2dLossConfig::default()
    })
}

#[cfg(feature = "dino")]
fn load_target_image_2d_adaptive(
    path: &Path,
    threshold: f32,
    target_points: usize,
    image_size: Option<usize>,
) -> Result<TargetImage2d, Box<dyn std::error::Error>> {
    Ok(crate::load_target_image_2d_upstream(
        path,
        threshold,
        target_points,
        image_size,
    )?)
}

fn write_pretty_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
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

fn has_json_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper::e2e::{E2eHyperNpa2d, E2eHyperNpa2dWeights};

    fn tiny_test_hyper() -> E2eHyperNpa2d {
        E2eHyperNpa2d {
            version: 1,
            architecture: "token_attention_pool_rectified_flow_generated_lora".to_string(),
            backend: Some("test".to_string()),
            condition_encoder: Some("sample-id-onehot".to_string()),
            condition_token_count: Some(1),
            condition_embed_dims: Some(1),
            condition_token_grid_width: Some(1),
            condition_token_grid_height: Some(1),
            condition_image_size: None,
            condition_alpha_mode: None,
            condition_rgb_channels: None,
            condition_rgb_channel_scale: None,
            condition_alpha_channel: None,
            condition_alpha_channel_scale: None,
            condition_patch_pixels: None,
            condition_l2_normalize_features: None,
            condition_resize_mode: None,
            condition_application: None,
            shared_base_sha256: None,
            hidden_dims: 1,
            token_attention_heads: 1,
            attention_normalization: None,
            output_dims: 1,
            sample_steps: 1,
            output_scale: 1.0,
            adapter_rank: None,
            adapter_alpha: None,
            adapter_parameterization: None,
            adapter_output_bias: None,
            adapter_chunk_size: None,
            spatial_condition_control: None,
            spatial_condition_control_scale: None,
            spatial_condition_control_sigma: None,
            spatial_condition_state_control: None,
            row_flow: None,
            weights: E2eHyperNpa2dWeights {
                token_w: vec![0.0],
                token_b: vec![0.0],
                token_gate_w: vec![0.0],
                token_gate_b: vec![0.0],
                state_w: vec![0.0],
                time_w: vec![0.0],
                output_w: vec![0.0],
                output_b: vec![0.0],
                condition_control_w: Vec::new(),
                condition_control_b: Vec::new(),
                condition_control_state_w: Vec::new(),
                row_flow: Vec::new(),
            },
        }
    }

    fn tiny_rollout_example(
        slug: &str,
        teacher_adapter: Option<Vec<f32>>,
    ) -> BurnE2eRolloutExample {
        BurnE2eRolloutExample {
            slug: slug.to_string(),
            target: TargetImage2d {
                source_width: 1,
                source_height: 1,
                positions: vec![[0.0, 0.0]],
                colors: vec![[1.0, 1.0, 1.0]],
                pixel_size: 1.0,
                threshold: 0.05,
                aabb: [-1.0, -1.0, 1.0, 1.0],
            },
            condition_path: None,
            dino_model_path: None,
            condition_features: vec![1.0],
            token_count: 1,
            embed_dims: 1,
            particle_count: 1,
            update_prob: 0.5,
            seed_scale: 1.0,
            teacher_adapter,
        }
    }

    #[test]
    fn adapter_bank_rows_are_resolved_from_recorded_slug_order() {
        let root = std::env::temp_dir().join(format!(
            "burn-automata-adapter-bank-manifest-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed"),
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("report.json"),
            br#"{
                "selected_sources": [
                    {"slug": "second"},
                    {"slug": "first"}
                ]
            }"#,
        )
        .unwrap();

        let identities = adapter_bank_identity_by_slug(&root.join("hyper_2d.bpk"), 2).unwrap();
        assert_eq!(identities.get("second"), Some(&0));
        assert_eq!(identities.get("first"), Some(&1));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn upstream_adapter_bank_rejects_legacy_output_bias() {
        let mut bank = tiny_test_hyper();
        let err = validate_upstream_adapter_bank_contract(&bank).unwrap_err();
        assert!(
            err.to_string().contains("zero-output-bias contract"),
            "unexpected error: {err}"
        );

        bank.adapter_output_bias = Some(false);
        validate_upstream_adapter_bank_contract(&bank).unwrap();
    }

    #[test]
    fn dino_full_token_dims_match_vits_224() {
        let grid = dino_patch_grid(224, 14).unwrap();
        assert_eq!(grid, 16);
        let token_count = 1 + grid * grid;
        assert_eq!(token_count, 257);
        assert_eq!(token_count * DINO_VITS_EMBED_DIMS, 98_688);
    }

    #[test]
    fn adapter_teacher_objective_parser_is_explicit() {
        assert_eq!(
            parse_e2e_adapter_teacher_objective(Some("parameter-mse")).unwrap(),
            E2eAdapterTeacherObjective::ParameterMse
        );
        assert_eq!(
            parse_e2e_adapter_teacher_objective(Some("functional-mse")).unwrap(),
            E2eAdapterTeacherObjective::FunctionalMse
        );
        assert_eq!(
            parse_e2e_adapter_teacher_objective(Some("hybrid")).unwrap(),
            E2eAdapterTeacherObjective::Hybrid
        );
        assert!(parse_e2e_adapter_teacher_objective(Some("raw-lora")).is_err());
    }

    #[test]
    fn exact_teacher_adapter_materializes_target_model() {
        let config = NpaConfig::growing_2d();
        let base = NpaModel::upstream_seeded(config.clone(), 7);
        let target = NpaModel::upstream_seeded(config.clone(), 19);
        let rank = config.perception_dims().max(config.update_dims());
        let adapter = exact_teacher_adapter(&base, &target, rank, rank as f32).unwrap();
        let materialized = adapter.apply_to_model(&base).unwrap();
        let assert_close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6)
            );
        };
        assert_close(&materialized.weights.w1, &target.weights.w1);
        assert_close(&materialized.weights.b1, &target.weights.b1);
        assert_close(&materialized.weights.w2, &target.weights.w2);
        assert_close(&materialized.weights.b2, &target.weights.b2);
    }

    #[test]
    fn holdout_examples_reject_oracle_adapter_targets() {
        let clean = vec![tiny_rollout_example("rose", None)];
        ensure_holdout_teacher_free(&clean).unwrap();

        let leaked = vec![tiny_rollout_example("tropical_fish", Some(vec![1.0]))];
        let err = ensure_holdout_teacher_free(&leaked).unwrap_err();
        assert!(err.to_string().contains("tropical_fish"));
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
    fn validation_horizons_are_sorted_deduplicated_and_include_final_steps() {
        assert_eq!(normalize_validation_horizons(None, 64).unwrap(), vec![64]);
        assert_eq!(
            normalize_validation_horizons(Some(&[512, 96, 512, 256]), 1024).unwrap(),
            vec![96, 256, 512, 1024]
        );
        assert!(normalize_validation_horizons(Some(&[0, 64]), 128).is_err());
        assert!(normalize_validation_horizons(Some(&[1, 2, 3, 4, 5, 6, 7, 8]), 9).is_err());
    }

    #[test]
    fn exposure_targets_resolve_to_optimizer_steps_without_rounding_down() {
        assert_eq!(
            optimizer_steps_for_exposure(240_000, 16, 128).unwrap(),
            30_000
        );
        assert_eq!(
            optimizer_steps_for_exposure(320_000, 16, 128).unwrap(),
            40_000
        );
        assert_eq!(
            optimizer_steps_for_exposure(40_000, 16, 64).unwrap(),
            10_000
        );
        assert_eq!(
            optimizer_steps_for_exposure(8_000, 900, 128).unwrap(),
            56_250
        );
        assert_eq!(
            optimizer_steps_for_exposure(24_000, 900, 64).unwrap(),
            337_500
        );
        assert_eq!(
            optimizer_steps_for_exposure(8_000, 900, 32).unwrap(),
            225_000
        );
        assert_eq!(optimizer_steps_for_exposure(1, 3, 2).unwrap(), 2);
    }

    #[test]
    fn verified_rollout_configs_parse() {
        for (
            name,
            expected_steps,
            expected_validation_interval,
            expected_backend,
            expected_perception_backend,
        ) in [
            ("bench_omnisvg_8_b4_p128.toml", 200, 200, "dense", "dense"),
            (
                "bench_omnisvg_8_b4_p128_tiled.toml",
                200,
                200,
                "tiled-adjoint",
                "dense",
            ),
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("configs/verified/2d/hypernpa/benchmarks")
                .join(name);
            let text = std::fs::read_to_string(path).unwrap();
            let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
            assert_eq!(
                config.condition.encoder.as_deref(),
                Some("dino-vits-full-tokens")
            );
            assert_eq!(config.training.steps, Some(expected_steps));
            assert_eq!(
                config.training.target2d_loss_backend.as_deref(),
                Some(expected_backend)
            );
            assert_eq!(
                config.training.perception_backend.as_deref(),
                Some(expected_perception_backend)
            );
            assert_eq!(
                config.training.credit_assignment.as_deref(),
                Some("full-bptt")
            );
            assert_eq!(
                config.training.max_full_bptt_particle_steps,
                Some(1_048_576)
            );
            assert_eq!(config.model.shared_base_train_start_step, Some(0));
            assert_eq!(
                config.validation.interval,
                Some(expected_validation_interval)
            );
            assert_ne!(config.training.use_particle_pool, Some(true));
            assert_eq!(
                config.adapter.generator.as_deref(),
                Some("module-token-decoder")
            );
            assert_eq!(config.condition.alpha_channel, Some(true));
            assert_eq!(
                config.gpu.condition_device_cache_max_bytes,
                Some(DEFAULT_DEVICE_CONDITION_CACHE_MAX_BYTES)
            );
            assert_eq!(config.gates.fail_on_violation, Some(true));
            assert!(config.optimizer.base_learning_rate.is_some());
            assert!(config.optimizer.generator_learning_rate.is_some());
            assert!(config.optimizer.base_grad_clip_norm.is_some());
            assert!(config.optimizer.generator_grad_clip_norm.is_some());
            assert_eq!(config.adapter.flow_source_scale, Some(1.0));
            assert_eq!(config.adapter.init_scale, Some(1.0e-3));
        }
    }

    #[test]
    fn verified_conditional_row_flow_configs_preserve_endpoint_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs/verified/2d/hypernpa/flow");
        for (name, steps, batch, width, layers, heads, ffn_dims, sample_steps) in [
            ("smoke_conditional_row_flow.toml", 1, 2, 64, 2, 4, 128, 2),
            (
                "production_contract_growing_catalog_row_flow_pretrain.toml",
                100_000,
                10,
                768,
                12,
                12,
                3_072,
                8,
            ),
        ] {
            let text = std::fs::read_to_string(root.join(name)).unwrap();
            let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
            assert_eq!(config.training.steps, Some(steps));
            assert_eq!(config.training.example_batch_size, Some(batch));
            assert_eq!(config.training.task_loss_weight, Some(0.0));
            assert_eq!(config.training.flow_matching_weight, Some(1.0));
            assert_eq!(
                config.training.objective.as_deref(),
                Some("conditional-row-flow-matching")
            );
            assert_eq!(config.model.shared_base_trainable, Some(false));
            assert_eq!(
                config.model.oracle_model_dir.as_deref(),
                Some(Path::new("models/catalog/growing"))
            );
            assert_eq!(
                config.condition.encoder.as_deref(),
                Some("dino-vits-full-tokens")
            );
            assert_eq!(config.condition.dino_image_size, Some(224));
            assert_eq!(config.condition.dino_patch_size, Some(14));
            assert_eq!(config.condition.rgb_channels, Some(true));
            assert_eq!(config.condition.alpha_channel, Some(true));
            assert_eq!(config.condition.token_attention_heads, Some(heads));
            assert_eq!(
                config.adapter.generator.as_deref(),
                Some("conditional-row-flow")
            );
            assert_eq!(
                config.adapter.parameterization.as_deref(),
                Some("dense-npa-row-residual")
            );
            assert_eq!(config.adapter.flow_hidden, Some(width));
            assert_eq!(config.adapter.flow_layers, Some(layers));
            assert_eq!(config.adapter.flow_ffn_dims, Some(ffn_dims));
            assert_eq!(config.adapter.flow_sample_steps, Some(sample_steps));
        }
    }

    #[test]
    fn verified_amortized_row_flow_smoke_crosses_substrate_boundary() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs/verified/2d/hypernpa/flow/smoke_conditional_row_flow_amortized.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
        assert_eq!(config.training.steps, Some(2));
        assert_eq!(config.training.amortization_enabled, Some(true));
        assert_eq!(config.training.amortization_substrate_steps, Some(1));
        assert_eq!(config.training.flow_train_sample_steps, Some(1));
        assert_eq!(config.adapter.flow_sample_steps, Some(2));
        assert_eq!(config.training.flow_self_rectification_weight, Some(1.0));
        assert_eq!(config.training.rollouts_per_example, Some(2));
    }

    #[test]
    fn verified_teacher_free_row_flow_configs_preserve_e2e_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs/verified/2d/hypernpa/e2e");
        for (name, source_limit, trainable, final_particles) in [
            ("smoke_conditional_row_flow_e2e.toml", 4, true, None),
            (
                "production_omnisvg_1k_conditional_row_flow_e2e_cuda.toml",
                1000,
                true,
                Some(4096),
            ),
        ] {
            let text = std::fs::read_to_string(root.join(name)).unwrap();
            let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
            assert_eq!(config.source.source_limit, Some(source_limit));
            assert_eq!(
                config.training.objective.as_deref(),
                Some("conditional-row-flow-e2e")
            );
            assert_eq!(config.training.task_loss_weight, Some(1.0));
            assert_eq!(config.training.flow_matching_weight, Some(0.0));
            assert_eq!(config.training.flow_self_rectification_weight, Some(0.05));
            assert_eq!(config.model.shared_base_trainable, Some(trainable));
            assert!(config.model.oracle_model_dir.is_none());
            assert_eq!(
                config.condition.encoder.as_deref(),
                Some("dino-vits-full-tokens")
            );
            assert_eq!(config.condition.dino_image_size, Some(224));
            assert_eq!(config.condition.dino_patch_size, Some(14));
            assert_eq!(
                config.adapter.generator.as_deref(),
                Some("conditional-row-flow")
            );
            assert_eq!(
                config.adapter.parameterization.as_deref(),
                Some("dense-npa-row-residual")
            );
            assert_eq!(config.adapter.flow_source_scale, Some(1.0e-3));
            assert_eq!(config.validation.final_particles, final_particles);
            if final_particles.is_some() {
                assert_eq!(config.training.example_batch_size, Some(8));
                assert_eq!(config.training.rollouts_per_example, Some(4));
                assert_eq!(config.training.pool_slots_per_example, Some(8));
                assert_eq!(config.training.pool_capacity, Some(8_000));
                assert_eq!(
                    config.training.credit_assignment.as_deref(),
                    Some("full-bptt")
                );
                assert_eq!(
                    config.training.max_full_bptt_particle_steps,
                    Some(33_554_432)
                );
                assert_eq!(config.training.gpu_memory_budget_gb, Some(90.0));
                assert_eq!(config.adapter.flow_sample_steps, Some(4));
                assert_eq!(config.training.inject_seed_interval, Some(4));
                assert_eq!(config.training.seed_replacements_per_interval, Some(2));
                assert_eq!(config.training.seed_trajectory_interval, Some(16));
                assert_eq!(config.model.shared_base_train_start_step, Some(1_000));
                assert_eq!(config.training.curriculum_resume, Some(true));
                assert_eq!(config.training.max_dense_train_particles, Some(4096));
                assert_eq!(
                    config.training.target2d_loss_backend.as_deref(),
                    Some("tiled-adjoint")
                );
                assert_eq!(config.training.perception_backend.as_deref(), Some("auto"));
                assert_eq!(config.rollout.particles, Some(4096));
                assert_eq!(config.rollout.step_min, Some(32));
                assert_eq!(config.rollout.steps, Some(96));
                assert_eq!(
                    config.training.example_batch_size.unwrap()
                        * config.training.rollouts_per_example.unwrap()
                        * config.rollout.particles.unwrap(),
                    8 * 4 * 4096,
                );
                assert_eq!(config.target.composited_rgb_loss_weight, Some(5.0));
                assert_eq!(config.optimizer.base_learning_rate, Some(1.0e-5));
                assert_eq!(
                    config.gpu.condition_device_cache_max_bytes,
                    Some(1_342_177_280)
                );
                assert!(config.training.pool_capacity.unwrap() >= 900 * 8);
                assert_eq!(
                    config.validation.final_horizons.as_deref(),
                    Some(&[96, 256, 512][..])
                );
                assert_eq!(config.validation.stability_examples, Some(16));
                assert_eq!(config.validation.stability_particles, Some(4096));
                assert_eq!(config.validation.stability_reference_steps, Some(512));
                assert_eq!(config.validation.stability_steps, Some(4096));
                assert_eq!(config.validation.stability_tail_steps, Some(256));
                assert_eq!(
                    config.gates.min_final_p10_composited_rgb_psnr_db,
                    Some(26.0)
                );
                assert_eq!(
                    config
                        .gates
                        .min_final_condition_shuffle_composited_psnr_gap_db,
                    Some(1.0)
                );
                assert_eq!(
                    config
                        .gates
                        .min_final_generated_adapter_composited_psnr_gain_db,
                    Some(1.0)
                );
            }
        }
    }

    #[test]
    fn matched_oracle_percentile_uses_interpolated_tail() {
        assert_eq!(percentile(&[], 0.9), None);
        assert_eq!(percentile(&[2.0], 0.9), Some(2.0));
        assert_eq!(percentile(&[0.0, 1.0, 2.0], 0.5), Some(1.0));
        assert!((percentile(&[0.0, 1.0, 2.0], 0.9).unwrap() - 1.8).abs() < 1.0e-6);
    }

    #[test]
    fn matched_oracle_comparison_requires_identical_contract_and_sample_ids() {
        let entry = |slug: &str, psnr: f32| BurnE2eRolloutQualityEntry {
            slug: slug.to_string(),
            total_loss: 0.0,
            splat_loss: 0.0,
            color_loss: 0.0,
            density_loss: 0.0,
            render_rgb_mse: 0.0,
            render_rgb_psnr_db: psnr,
            composited_rgb_mse: 0.0,
            composited_rgb_psnr_db: psnr,
            teacher_adapter_composited_rgb_psnr_db: None,
            gap_to_teacher_adapter_db: None,
            foreground_rgb_mse: 0.0,
            foreground_rgb_psnr_db: psnr,
            density_mse: 0.0,
            density_psnr_db: 0.0,
            density_soft_iou: 0.0,
            passed: true,
        };
        let entries = [entry("a", 25.0), entry("b", 27.0)];
        let oracle = serde_json::json!({
            "particle_count": 4096,
            "rollout_steps": 512,
            "seed": 42,
            "update_prob": 0.5,
            "entries": [
                {"slug": "a", "composited_rgb_psnr_db": 26.0},
                {"slug": "b", "composited_rgb_psnr_db": 28.5},
            ],
        });
        let comparison =
            compare_matched_oracle_quality(4096, 512, 42, 0.5, &entries, &oracle).unwrap();
        assert!(comparison.contract_matched);
        assert!(comparison.all_oracles_matched);
        assert_eq!(comparison.matched_entries, 2);
        assert!((comparison.mean_gap_db.unwrap() - 1.25).abs() < 1.0e-6);
        assert!((comparison.p90_gap_db.unwrap() - 1.45).abs() < 1.0e-6);

        let mismatched =
            compare_matched_oracle_quality(2048, 512, 42, 0.5, &entries[..1], &oracle).unwrap();
        assert!(!mismatched.contract_matched);
        assert!(!mismatched.all_oracles_matched);
    }

    #[test]
    fn verified_quality_scale_throughput_config_preserves_hot_path_contract() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs/verified/2d/hypernpa/benchmarks")
            .join("throughput_omnisvg_64_b64_p1024_s96_cuda.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();

        assert_eq!(config.source.source_limit, Some(64));
        assert_eq!(
            config.condition.encoder.as_deref(),
            Some("dino-vits-full-tokens")
        );
        assert_eq!(config.condition.dino_batch_size, Some(32));
        assert_eq!(config.training.example_batch_size, Some(64));
        assert_eq!(
            config.training.credit_assignment.as_deref(),
            Some("full-bptt")
        );
        assert_eq!(
            config.training.max_full_bptt_particle_steps,
            Some(6_291_456)
        );
        assert_eq!(
            config.training.target2d_loss_backend.as_deref(),
            Some("tiled-adjoint")
        );
        assert_eq!(config.training.perception_backend.as_deref(), Some("auto"));
        assert_eq!(config.training.use_particle_pool, Some(true));
        assert_eq!(config.training.pool_capacity, Some(256));
        assert_eq!(config.gpu.backend.as_deref(), Some("burn-cuda"));
        assert_eq!(config.rollout.particles, Some(1024));
        assert_eq!(config.rollout.step_min, Some(96));
        assert_eq!(config.rollout.steps, Some(96));
        assert_eq!(
            config.gates.min_median_particle_steps_per_sec,
            Some(15_000_000.0)
        );
        assert_eq!(config.gates.fail_on_violation, Some(true));
    }

    #[test]
    fn verified_catalog_quality_configs_preserve_teacher_and_eval_contracts() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs/verified/2d/hypernpa/published");
        for name in [
            "quality_growing_catalog_pretrain.toml",
            "quality_growing_catalog_refine.toml",
        ] {
            let text = std::fs::read_to_string(root.join(name)).unwrap();
            let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
            assert_eq!(
                config.model.oracle_model_dir.as_deref(),
                Some(Path::new("models/catalog/growing"))
            );
            assert_eq!(config.training.task_loss_weight, Some(0.0));
            assert_eq!(config.training.adapter_teacher_weight, Some(1000.0));
            assert_eq!(config.adapter.rank, Some(82));
            assert_eq!(config.validation.selection_horizon_min_steps, Some(256));
            assert_eq!(
                config.adapter.generator.as_deref(),
                Some("token-attention-pool")
            );
        }

        let text = std::fs::read_to_string(root.join("quality_growing_catalog_nonbase_eval.toml"))
            .unwrap();
        let config: RolloutExperimentConfig = toml::from_str(&text).unwrap();
        assert_eq!(config.validation.particles, Some(4096));
        assert_eq!(config.validation.steps, Some(1024));
        assert_eq!(
            config.validation.horizons.as_deref(),
            Some([96, 256, 512, 1024].as_slice())
        );
        assert_eq!(config.validation.psnr_threshold_db, Some(26.0));
        assert_eq!(config.validation.examples, Some(2));
        assert_eq!(config.validation.selection_horizon_min_steps, Some(256));
    }

    #[test]
    fn rollout_report_records_sampled_training_steps() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [output]
            checkpoint_interval_steps = 3

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 10
            report_interval = 5
            example_batch_size = 1
            tbptt_chunk_steps = 4
            loss_on_final_chunk_only = false
            tbptt_loss_mode = "endpoint-weighted"
            tbptt_intermediate_loss_weight = 0.25
            tbptt_final_loss_weight = 1.0
            credit_assignment = "detached-tbptt"
            use_particle_pool = true
            pool_slots_per_example = 2
            inject_seed_interval = 64
            brush_size = 0.1
            pre_rollout_step_min = 1
            pre_rollout_steps = 3
            target2d_loss_backend = "tiled-adjoint"
            perception_backend = "tiled-adjoint"

            [gpu]
            backend = "burn-wgpu"

            [optimizer]
            warmup_steps = 100

            [rollout]
            particles = 512
            step_min = 8
            steps = 16

            [validation]
            examples = 1
            interval = 10
            particles = 512
            steps = 16
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.rollout.step_min, 8);
        assert_eq!(report.rollout.steps, 16);
        assert!(report.rollout.sampled_training_steps);
        assert!(!report.training.loss_on_final_chunk_only);
        assert_eq!(report.training.tbptt_loss_mode, "endpoint-weighted");
        assert_eq!(report.training.tbptt_intermediate_loss_weight, 0.25);
        assert_eq!(report.training.tbptt_final_loss_weight, 1.0);
        assert_eq!(report.training.credit_assignment, "detached-tbptt");
        assert!(report.training.tbptt_state_detach_active);
        assert_eq!(report.training.effective_tbptt_chunk_steps, Some(4));
        assert_eq!(report.training.rollout_gradient_horizon_max_steps, Some(4));
        assert_eq!(report.optimizer.warmup_steps, 100);
        assert!(report.training.use_particle_pool);
        assert_eq!(report.training.pool_slots_per_example, 2);
        assert_eq!(report.training.planned_rollout_trajectories, 10);
        assert_eq!(
            report.training.planned_mean_trajectories_per_train_example,
            10.0
        );
        assert_eq!(
            report
                .training
                .upstream_growing_reference_trajectories_per_target,
            240_000
        );
        assert!(
            (report
                .training
                .upstream_growing_trajectory_exposure_fraction
                - 10.0 / 240_000.0)
                .abs()
                < f64::EPSILON
        );
        assert_eq!(report.training.inject_seed_interval, 64);
        assert_eq!(report.training.brush_size, 0.1);
        assert_eq!(report.training.pre_rollout_step_min, 1);
        assert_eq!(report.training.pre_rollout_steps, 3);
        assert_eq!(report.output.checkpoint_interval_steps, 3);
        assert_eq!(report.training.target2d_loss_backend, "tiled-adjoint");
        assert_eq!(report.training.perception_backend, "tiled-adjoint");
    }

    #[test]
    fn rollout_report_accepts_resumed_rollout_free_amortization_distillation() {
        let mut config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [output]
            resume_checkpoint = "artifacts/test/checkpoints"

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true
            token_attention_heads = 4

            [model]
            shared_base_trainable = false

            [training]
            backend = "gpu"
            objective = "conditional-row-flow-matching"
            steps = 10
            example_batch_size = 1
            credit_assignment = "full-bptt"
            task_loss_weight = 0.0
            flow_matching_weight = 0.0
            flow_self_rectification_weight = 0.1
            amortization_enabled = true
            amortization_distillation_weight = 1.0

            [gpu]
            backend = "burn-wgpu"

            [adapter]
            rank = 66
            alpha = 66.0
            generator = "conditional-row-flow"
            parameterization = "dense-npa-row-residual"
            flow_hidden = 64
            flow_layers = 2
            flow_ffn_dims = 128
            flow_sample_steps = 2

            [optimizer]
            generator_learning_rate = 1.0e-4
            amortization_learning_rate = 1.0e-3

            [rollout]
            particles = 32
            steps = 2

            [validation]
            examples = 1
            interval = 10
            particles = 32
            steps = 2
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.training.task_loss_weight, 0.0);
        assert!(report.training.amortization_enabled);
        assert_eq!(report.training.amortization_distillation_weight, 1.0);
        assert_eq!(report.training.flow_self_rectification_weight, 0.1);

        config.output.resume_checkpoint = None;
        let error = match build_e2e_rollout_report(Path::new("inline.toml"), &config) {
            Ok(_) => panic!("fresh zero endpoint tables must not be flow-distillation targets"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("requires output.resume_checkpoint"));
    }

    #[test]
    fn detached_tbptt_supports_amortized_behavioral_flow_training() {
        let mut config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true
            token_attention_heads = 4

            [training]
            backend = "gpu"
            objective = "conditional-row-flow-e2e"
            steps = 10
            example_batch_size = 1
            credit_assignment = "detached-tbptt"
            tbptt_chunk_steps = 4
            task_loss_weight = 1.0
            amortization_enabled = true
            amortization_substrate_steps = 10
            amortization_initialize_from_generator = true
            amortization_distillation_objective = "hybrid"
            amortization_distillation_probe_rollout_steps = 3

            [gpu]
            backend = "burn-wgpu"

            [adapter]
            rank = 66
            alpha = 66.0
            generator = "conditional-row-flow"
            parameterization = "dense-npa-row-residual"
            flow_hidden = 64
            flow_layers = 2
            flow_ffn_dims = 128
            flow_sample_steps = 2

            [optimizer]
            generator_learning_rate = 1.0e-4
            amortization_learning_rate = 1.0e-3

            [rollout]
            particles = 32
            steps = 8

            [validation]
            examples = 1
            interval = 10
            particles = 32
            steps = 8
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert!(report.training.amortization_enabled);
        assert!(report.training.amortization_initialize_from_generator);
        assert_eq!(
            report.training.amortization_distillation_objective,
            "functional-plus-parameter-mse"
        );
        assert_eq!(
            report
                .training
                .amortization_distillation_probe_rollout_steps,
            3
        );
        assert_eq!(report.training.credit_assignment, "detached-tbptt");
        assert_eq!(report.training.effective_tbptt_chunk_steps, Some(4));

        config.training.amortization_substrate_steps = Some(9);
        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert!(!report.training.trains_hypernet_from_step_zero);
        assert_eq!(report.training.amortization_substrate_steps, 9);
    }

    #[test]
    fn full_bptt_report_uses_the_sampled_rollout_as_gradient_horizon() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "sample-id-onehot"

            [training]
            steps = 1
            credit_assignment = "full-bptt"
            tbptt_chunk_steps = 1
            max_full_bptt_particle_steps = 4096

            [rollout]
            particles = 16
            step_min = 8
            steps = 16
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.training.credit_assignment, "full-bptt");
        assert!(!report.training.tbptt_state_detach_active);
        assert_eq!(report.training.tbptt_chunk_steps, 1);
        assert_eq!(report.training.effective_tbptt_chunk_steps, None);
        assert_eq!(report.training.rollout_gradient_horizon_max_steps, Some(15));
    }

    #[test]
    fn full_bptt_preflight_rejects_unbounded_graphs() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"
            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]
            [condition]
            encoder = "sample-id-onehot"
            [training]
            steps = 10
            example_batch_size = 64
            credit_assignment = "full-bptt"
            max_full_bptt_particle_steps = 1000
            [rollout]
            particles = 512
            steps = 64
            "#,
        )
        .unwrap();
        let err = build_e2e_rollout_report(Path::new("inline.toml"), &config)
            .err()
            .expect("oversized full-BPTT graph should fail preflight");
        assert!(err.to_string().contains("full-bptt preflight rejected"));
    }

    #[test]
    fn detached_tbptt_preflight_bounds_the_active_chunk_graph() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"
            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]
            [condition]
            encoder = "sample-id-onehot"
            [training]
            steps = 10
            example_batch_size = 64
            rollouts_per_example = 4
            credit_assignment = "detached-tbptt"
            tbptt_chunk_steps = 32
            max_full_bptt_particle_steps = 1000
            [rollout]
            particles = 4096
            steps = 256
            "#,
        )
        .unwrap();
        let err = build_e2e_rollout_report(Path::new("inline.toml"), &config)
            .err()
            .expect("oversized detached-TBPTT chunk should fail preflight");
        assert!(
            err.to_string()
                .contains("detached-tbptt preflight rejected")
        );
    }

    #[test]
    fn rollout_report_supports_sample_id_onehot_conditioning() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = [
                "assets/reference_targets/lizard_upstream_120.png",
                "assets/catalog_thumbnails/turtle.png",
            ]

            [condition]
            encoder = "sample-id-onehot"
            online = false

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 0
            example_batch_size = 2

            [gpu]
            backend = "burn-wgpu"
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.condition.encoder, "sample-id-onehot");
        assert!(!report.condition.online_dino);
        assert_eq!(report.condition.token_count, 1);
        assert_eq!(report.condition.embed_dims, 2);
        assert_eq!(report.condition.flattened_feature_dims, 2);
        assert_eq!(report.condition.sample_id_vocab_size, Some(2));
        assert_eq!(report.condition.sample_ids, [0, 1]);
        assert_eq!(report.source.train_examples, 2);
        assert!(report.output.hyper_output.ends_with("hyper_2d.bpk"));
    }

    #[test]
    fn rollout_report_maps_a_source_subset_to_existing_adapter_table_columns() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = [
                "assets/reference_targets/lizard_upstream_120.png",
                "assets/catalog_thumbnails/turtle.png",
            ]

            [condition]
            encoder = "sample-id-onehot"
            sample_id_vocab_size = 16
            sample_ids = [5, 11]

            [adapter]
            generator = "sample-id-table"

            [training]
            steps = 0
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.condition.embed_dims, 16);
        assert_eq!(report.condition.sample_id_vocab_size, Some(16));
        assert_eq!(report.condition.sample_ids, [5, 11]);
    }

    #[test]
    fn rollout_report_supports_sample_id_adapter_table_control() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = [
                "assets/reference_targets/lizard_upstream_120.png",
                "assets/catalog_thumbnails/turtle.png",
            ]

            [condition]
            encoder = "sample-id-onehot"
            online = false

            [adapter]
            generator = "sample-id-table"

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 0

            [gpu]
            backend = "burn-wgpu"
            "#,
        )
        .unwrap();

        let mut report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.adapter.generator, E2E_HYPER_ARCH_SAMPLE_ID_TABLE);
        assert_eq!(report.condition.feature_normalization, "sample-id-onehot");

        report.source.selected_sources = 100_000;
        report.training.gpu_memory_budget_gb = Some(1.0);
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("persistent GPU state requires"));
        assert!(err.contains("adapter weights/gradients/AdamW"));

        report.source.selected_sources = 2;
        report.training.gpu_memory_budget_gb = None;
        report.training.system_memory_budget_gb = Some(1.0);
        report.training.use_particle_pool = true;
        report.training.pool_capacity = 1_024;
        report.rollout.particles = 1_024;
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("particle-pool checkpoint readback"));

        report.training.system_memory_budget_gb = None;
        report.training.pool_capacity = 8_192;
        report.rollout.particles = 4_096;
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must stay below 2.00 GiB"));
        assert!(err.contains("sharded pool storage"));
    }

    #[test]
    fn active_gpu_preflight_accounts_for_row_flow_and_rollout_graphs() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"
            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]
            [condition]
            encoder = "sample-id-onehot"
            [training]
            steps = 1
            example_batch_size = 1
            [validation]
            split = "train"
            [rollout]
            particles = 512
            steps = 96
            "#,
        )
        .unwrap();
        let mut report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.validation.split, "train");
        report.training.gpu_memory_budget_gb = Some(90.0);
        report.training.credit_assignment = "full-bptt".to_string();
        report.rollout.particles = 4096;
        report.training.example_batch_size = 8;
        report.training.rollouts_per_example = 8;
        report.training.use_particle_pool = false;
        report.adapter.generator = E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW.to_string();
        report.adapter.flow_hidden = 768;
        report.adapter.flow_layers = 12;
        report.adapter.flow_ffn_dims = 3072;
        report.adapter.flow_sample_steps = 4;
        report.condition.embed_dims = 1168;
        report.condition.token_count = 257;
        report.condition.token_attention_heads = 12;
        report.training.flow_self_rectification_weight = 0.05;

        let oversized = projected_active_training_gpu_memory(&report);
        assert!(bytes_to_gib(oversized.total_bytes) > 90.0);
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("projected active HyperNPA GPU peak"));

        report.training.rollouts_per_example = 4;
        let bounded = projected_active_training_gpu_memory(&report);
        assert!(bytes_to_gib(bounded.total_bytes) < 90.0);
        check_condition_preload_memory_budget(&report).unwrap();

        report.training.amortization_enabled = true;
        report.training.amortization_substrate_steps = report.training.steps;
        report.training.amortization_optimizer_bytes_f32 = 128 * 1024 * 1024;
        let substrate_only = projected_active_training_gpu_memory(&report);
        assert_eq!(substrate_only.row_flow_graph_bytes, 0);
        assert!(substrate_only.trainable_state_bytes < bounded.trainable_state_bytes);
        assert!(substrate_only.total_bytes < bounded.total_bytes);

        report.training.rollouts_per_example = 6;
        let fused_substrate_48 = projected_active_training_gpu_memory(&report);
        assert!(bytes_to_gib(fused_substrate_48.total_bytes) < 90.0);
        check_condition_preload_memory_budget(&report).unwrap();

        report.training.task_loss_weight = 0.0;
        report.training.amortization_substrate_steps = 0;
        report.training.amortization_distillation_weight = 1.0;
        report.training.rollouts_per_example = 64;
        let rollout_free_distillation = projected_active_training_gpu_memory(&report);
        assert_eq!(rollout_free_distillation.rollout_graph_bytes, 0);
        check_condition_preload_memory_budget(&report).unwrap();

        report.training.task_loss_weight = 1.0;
        report.training.amortization_distillation_weight = 0.1;
        report.training.rollouts_per_example = 4;
        report.training.amortization_enabled = false;
        report.training.amortization_substrate_steps = 0;
        report.training.amortization_optimizer_bytes_f32 = 0;

        report.training.gpu_memory_budget_gb = Some(40.0);
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("projected active HyperNPA GPU peak"));
        assert!(err.contains("row-flow graph"));
        report.training.gpu_memory_budget_gb = Some(90.0);

        report.rollout.particles = 2048;
        report.training.rollouts_per_example = 6;
        let projected_48 = projected_active_training_gpu_memory(&report);
        assert!(bytes_to_gib(projected_48.total_bytes) < 90.0);
        check_condition_preload_memory_budget(&report).unwrap();

        report.training.rollouts_per_example = 8;
        let trajectory_oversized = projected_active_training_gpu_memory(&report);
        assert!(bytes_to_gib(trajectory_oversized.total_bytes) > 90.0);
        let err = check_condition_preload_memory_budget(&report)
            .unwrap_err()
            .to_string();
        assert!(err.contains("projected active HyperNPA GPU peak"));
    }

    #[test]
    fn rollout_report_supports_spatial_token_flow_generator() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            dino_model = "assets/models/dino_vits14.mpk"
            dino_image_size = 224
            dino_patch_size = 14
            online = true

            [adapter]
            generator = "spatial-token-flow"
            adapter_chunk_size = 32

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 0

            [gpu]
            backend = "burn-wgpu"
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.adapter.generator, E2E_HYPER_ARCH_SPATIAL_TOKEN_FLOW);
        assert_eq!(
            report.adapter.parameterization,
            E2E_HYPER_ADAPTER_FACTORIZED
        );
        assert_eq!(report.adapter.adapter_chunk_size, 32);
        assert_eq!(report.condition.token_count, 257);
        assert_eq!(
            report.condition.feature_normalization,
            "per-token-preserved"
        );
    }

    #[test]
    fn rollout_report_supports_canonical_full_rank_lora_and_split_initialization() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            dino_model = "assets/models/dino_vits14.mpk"
            dino_image_size = 224
            dino_patch_size = 14
            online = true

            [adapter]
            rank = 82
            alpha = 82.0
            generator = "module-token-decoder"
            parameterization = "canonical-full-rank"
            attention_normalization = "softmax"
            condition_init_scale = 0.1
            output_init_scale = 0.001

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 0

            [gpu]
            backend = "burn-wgpu"
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(
            report.adapter.parameterization,
            E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
        );
        assert_eq!(
            report.adapter.attention_normalization,
            E2E_HYPER_ATTENTION_SOFTMAX
        );
        assert_eq!(report.adapter.condition_init_scale, 0.1);
        assert_eq!(report.adapter.output_init_scale, 0.001);
    }

    #[test]
    fn rollout_report_rejects_json_hyper_weight_output() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [output]
            hyper_output = "artifacts/sandbox/hyper_2d.json"

            [condition]
            encoder = "sample-id-onehot"
            online = false

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 0

            [gpu]
            backend = "burn-wgpu"
            "#,
        )
        .unwrap();

        let err = match build_e2e_rollout_report(Path::new("inline.toml"), &config) {
            Ok(_) => panic!("json hyper output should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("artifacts are binary"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rollout_report_rejects_step_min_above_steps() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 10
            report_interval = 5
            example_batch_size = 1

            [gpu]
            backend = "burn-wgpu"

            [rollout]
            particles = 512
            step_min = 17
            steps = 16
            "#,
        )
        .unwrap();

        let err = match build_e2e_rollout_report(Path::new("inline.toml"), &config) {
            Ok(_) => panic!("rollout.step_min above rollout.steps should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("rollout.step_min"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rollout_report_accepts_condition_auxiliaries_with_detached_tbptt() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true

            [model]
            adapter_bank = "artifacts/test/teacher_bank.bpk"
            shared_base_trainable = false

            [training]
            backend = "gpu"
            objective = "conditional-row-flow-e2e"
            steps = 10
            report_interval = 5
            example_batch_size = 1
            credit_assignment = "detached-tbptt"
            tbptt_chunk_steps = 8
            task_loss_weight = 1.0
            adapter_teacher_weight = 1.0
            adapter_teacher_objective = "hybrid"
            flow_matching_weight = 0.1
            flow_self_rectification_weight = 0.01

            [gpu]
            backend = "burn-wgpu"

            [adapter]
            rank = 66
            alpha = 66.0
            generator = "conditional-row-flow"
            parameterization = "dense-npa-row-residual"
            flow_hidden = 144
            flow_layers = 2
            flow_ffn_dims = 288

            [rollout]
            particles = 128
            step_min = 16
            steps = 32
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert_eq!(report.training.credit_assignment, "detached-tbptt");
        assert_eq!(report.training.effective_tbptt_chunk_steps, Some(8));
        assert_eq!(report.training.adapter_teacher_weight, 1.0);
        assert_eq!(report.training.flow_matching_weight, 0.1);
        assert_eq!(report.training.flow_self_rectification_weight, 0.01);
    }

    #[test]
    fn rollout_report_warns_for_curriculum_training_and_low_quality_gate() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 10
            report_interval = 5
            example_batch_size = 1
            max_dense_train_particles = 512

            [gpu]
            backend = "burn-wgpu"

            [rollout]
            particles = 128
            steps = 4

            [validation]
            examples = 16
            interval = 10
            particles = 2048
            steps = 64
            psnr_threshold_db = 8.0
            "#,
        )
        .unwrap();

        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        assert!(report.validation.quality_scale);
        assert!(!report.validation.training_backward_safe);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("below the 26dB"))
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("curriculum/diagnostic"))
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("validation-only"))
        );
    }

    #[test]
    fn rollout_gate_evaluator_reports_configured_passes() {
        let config: RolloutExperimentConfig = toml::from_str(
            r#"
            preset = "growing-2d"

            [source]
            target_images = ["assets/reference_targets/lizard_upstream_120.png"]

            [condition]
            encoder = "dino-vits-full-tokens"
            online = true

            [training]
            backend = "gpu"
            objective = "target2d-rollout-image-loss"
            steps = 10
            report_interval = 5
            example_batch_size = 1

            [gpu]
            backend = "burn-wgpu"

            [rollout]
            particles = 128
            steps = 4

            [validation]
            examples = 1
            interval = 10
            particles = 128
            steps = 4

            [gates]
            min_median_particle_steps_per_sec = 100.0
            max_quality_validation_evaluations = 3
            max_quality_validation_elapsed_fraction = 0.25
            require_validation_interval_at_least_report_interval = true
            fail_on_violation = true
            "#,
        )
        .unwrap();
        let report = build_e2e_rollout_report(Path::new("inline.toml"), &config).unwrap();
        let training = BurnE2eRolloutOutput {
            backend: "test".to_string(),
            device: "test-device".to_string(),
            metrics: serde_json::json!({
                "median_reported_particle_steps_per_sec": 250.0,
                "quality_validation_evaluations": 2,
                "quality_validation_elapsed_ms": 10.0,
                "burn_training_ms": 100.0,
            }),
            history: Vec::new(),
            final_loss: Some(1.0),
            generator: tiny_test_hyper(),
            quality_validation: None,
            amortization_quality_validation: None,
            stability_validation: None,
        };

        let gates = evaluate_e2e_rollout_gates(&report, &training);
        assert_eq!(gates.len(), 4);
        assert!(gates.iter().all(|gate| gate.passed), "{gates:#?}");
    }
}
