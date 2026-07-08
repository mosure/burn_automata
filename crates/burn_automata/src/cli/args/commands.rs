use std::path::PathBuf;

use super::values::*;
use crate::{
    cli::targets::{
        DEFAULT_3D_MESH_FIELD_SCALE, LOCAL_GROWTH_EXPANSION_GAIN, LOCAL_GROWTH_OPACITY_GAIN,
    },
    mesh_objective::{
        ROBUST_3D_COVERAGE_GAIN, ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
        ROBUST_3D_COVERAGE_REPULSION_GAIN, ROBUST_3D_COVERAGE_SAMPLES,
        ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP, ROBUST_3D_EXTENT_GAIN,
        ROBUST_3D_LIVENESS_FRONT_RADIUS, ROBUST_3D_LIVENESS_GAIN,
        ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER, ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE, ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        ROBUST_3D_MATERIAL_TAIL_GAIN, ROBUST_3D_OPACITY_GAIN, ROBUST_3D_SCALE_BUDGET_WEIGHT,
        ROBUST_3D_SCALE_GAIN, ROBUST_3D_SURFACE_ESCAPE_GAIN, ROBUST_3D_SURFACE_GAIN,
        ROBUST_3D_TRAJECTORY_MESH_GAIN, ROBUST_3D_TRAJECTORY_RENDER_GAIN,
        ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
    },
};
use clap::{ArgAction, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about)]
pub(crate) struct CliArgs {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Command {
    Infer {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 1024)]
        particles: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value = "auto")]
        neighbor_mode: NeighborModeArg,
        #[arg(long)]
        bucket_capacity: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "/tmp/burn_automata_rollout.json")]
        output: PathBuf,
    },
    Train {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value = "/tmp/burn_automata_training_report.json")]
        output: PathBuf,
        #[arg(long)]
        model_output: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value = "adam-w")]
        optimizer: TrainingOptimizerArg,
        #[arg(long, default_value = "auto")]
        training_device: TrainingDeviceArg,
        #[arg(long, default_value_t = 0.9)]
        adam_beta1: f32,
        #[arg(long, default_value_t = 0.999)]
        adam_beta2: f32,
        #[arg(long, default_value_t = 1e-8)]
        adam_epsilon: f32,
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        #[arg(long, default_value_t = 1)]
        report_interval: usize,
        #[arg(long)]
        target_model: Option<PathBuf>,
        #[arg(long)]
        target_seed: Option<u64>,
        #[arg(long)]
        zero_update: bool,
        #[arg(long, default_value_t = 7)]
        student_seed: u64,
        #[arg(long, default_value = "rollout")]
        batch_source: TrainingBatchArg,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 16)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 5)]
        temporal_samples: usize,
        #[arg(long, default_value_t = 1.0)]
        rollout_update_prob: f32,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
    },
    #[command(name = "eval-target2d", alias = "eval-target-2d")]
    EvalTarget2d {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        target_image: PathBuf,
        #[arg(long)]
        reference_model: Option<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 256)]
        image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        splat_sigma: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        center_loss: bool,
        #[arg(long, default_value_t = 2.0)]
        splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        density_loss_weight: f32,
        #[arg(long, default_value_t = 0.0)]
        background_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.0)]
        foreground_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        bound_regularizer_weight: f32,
        #[arg(long)]
        render_output_dir: Option<PathBuf>,
        #[arg(long)]
        wgpu_render_diagnostic: bool,
        #[arg(long, default_value = "/tmp/burn_automata_target2d_eval.json")]
        output: PathBuf,
    },
    #[command(
        name = "validate-npa2d-parity",
        alias = "validate-npa-2d-parity",
        alias = "validate-target2d-parity"
    )]
    ValidateNpa2dParity {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = "artifacts/npa2d_parity/report.json")]
        output: PathBuf,
    },
    #[command(name = "train-target2d", alias = "train-target-2d", hide = true)]
    TrainTarget2d {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        experimental: bool,
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        target_image: Option<PathBuf>,
        #[arg(
            long,
            default_value = "/tmp/burn_automata_target2d_training_report.json"
        )]
        output: PathBuf,
        #[arg(long, default_value = "cpu")]
        training_device: TrainingDeviceArg,
        #[arg(long, default_value = "burn-wgpu")]
        gpu_backend: DirectBasisOracleBackendArg,
        #[arg(long)]
        model_output: Option<PathBuf>,
        #[arg(long)]
        checkpoint_model_output: Option<PathBuf>,
        #[arg(long)]
        checkpoint_best_model_output: Option<PathBuf>,
        #[arg(long)]
        checkpoint_report_output: Option<PathBuf>,
        #[arg(long, default_value_t = 900)]
        checkpoint_interval_seconds: u64,
        #[arg(long, default_value_t = 0)]
        checkpoint_interval_steps: usize,
        #[arg(long)]
        reference_model: Option<PathBuf>,
        #[arg(long, default_value_t = 10000)]
        epochs: usize,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        #[arg(long, default_value_t = 100)]
        report_interval: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 512)]
        pool_size: usize,
        #[arg(long, default_value_t = 4096)]
        particles: usize,
        #[arg(long, default_value_t = 32)]
        step_min: usize,
        #[arg(long, default_value_t = 96)]
        step_max: usize,
        #[arg(long, default_value_t = 16)]
        inject_seed_interval: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 42)]
        student_seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 0.1)]
        brush_size: f32,
        #[arg(long, default_value_t = 5.0e-4)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value_t = 0.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.9)]
        adam_beta1: f32,
        #[arg(long, default_value_t = 0.999)]
        adam_beta2: f32,
        #[arg(long, default_value_t = 1.0e-8)]
        adam_epsilon: f32,
        #[arg(long = "scheduler-milestone", value_delimiter = ',')]
        scheduler_milestones: Vec<usize>,
        #[arg(long, default_value_t = 0.3)]
        scheduler_gamma: f32,
        #[arg(long, action = ArgAction::SetFalse)]
        per_parameter_grad_normalization: bool,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 256)]
        image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        splat_sigma: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        center_loss: bool,
        #[arg(long, default_value_t = 2.0)]
        splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        density_loss_weight: f32,
        #[arg(long, default_value_t = 0.0)]
        background_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.0)]
        foreground_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        bound_regularizer_weight: f32,
    },
    #[command(name = "eval-dynamics2d", alias = "eval-dynamics-2d")]
    EvalDynamics2d {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        target_model: PathBuf,
        #[arg(long, default_value_t = 4096)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        render_sigma_px: f32,
        #[arg(long, default_value = "/tmp/burn_automata_dynamics2d_eval.json")]
        output: PathBuf,
    },
    #[command(name = "train-hyper2d", alias = "train-hyper-2d", hide = true)]
    TrainHyper2d {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        condition: Option<PathBuf>,
        #[arg(long)]
        target_model: Option<PathBuf>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, default_value = "assets/catalog_thumbnails")]
        catalog_thumbnail_dir: PathBuf,
        #[arg(long)]
        catalog_group: Option<Hyper2dCatalogGroupArg>,
        #[arg(long = "catalog-target", value_delimiter = ',')]
        catalog_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        catalog_limit: usize,
        #[arg(long, default_value_t = 0)]
        holdout_stride: usize,
        #[arg(long, default_value_t = 0)]
        holdout_offset: usize,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long)]
        hyper_input: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper_2d.json")]
        hyper_output: PathBuf,
        #[arg(long, default_value = "artifacts/hyper_2d_training_report.json")]
        report_output: PathBuf,
        #[arg(long)]
        adapter_output: Option<PathBuf>,
        #[arg(long)]
        materialized_output: Option<PathBuf>,
        #[arg(long)]
        generated_output_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 512)]
        rows: usize,
        #[arg(long, default_value_t = 1.0e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value_t = 66)]
        adapter_rank: usize,
        #[arg(long, default_value_t = 66.0)]
        adapter_alpha: f32,
        #[arg(long, default_value_t = 512)]
        hyper_hidden: usize,
        #[arg(long, default_value_t = 8.0)]
        hyper_output_scale: f32,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_WIDTH)]
        condition_token_grid_width: usize,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_HEIGHT)]
        condition_token_grid_height: usize,
        #[arg(long, default_value_t = 42)]
        hyper_seed: u64,
        #[arg(long, default_value_t = 1)]
        adapter_bootstrap_steps: usize,
        #[arg(long)]
        adapter_bootstrap_learning_rate: Option<f32>,
        #[arg(long)]
        adapter_bootstrap_grad_clip_norm: Option<f32>,
        #[arg(long)]
        rollout_particles: Option<usize>,
        #[arg(long, default_value_t = 16)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long)]
        rollout_update_prob: Option<f32>,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
    },
    #[command(
        name = "train-hyper2d-e2e",
        alias = "train-hyper-2d-e2e",
        alias = "train-hypernpa2d-e2e",
        hide = true
    )]
    TrainHyper2dE2e {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long = "target-image", value_delimiter = ',')]
        target_images: Vec<PathBuf>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, default_value = "assets/catalog_thumbnails")]
        catalog_thumbnail_dir: PathBuf,
        #[arg(long)]
        catalog_group: Option<Hyper2dCatalogGroupArg>,
        #[arg(long = "catalog-target", value_delimiter = ',')]
        catalog_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        catalog_limit: usize,
        #[arg(long)]
        omnisvg_dataset: Option<OmniSvgDatasetArg>,
        #[arg(long, default_value = "train")]
        omnisvg_split: String,
        #[arg(long, default_value = "data/omnisvg")]
        omnisvg_cache_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        omnisvg_offset: usize,
        #[arg(long, default_value_t = 128)]
        omnisvg_limit: usize,
        #[arg(long, default_value_t = 100)]
        omnisvg_page_size: usize,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        omnisvg_download: bool,
        #[arg(long)]
        omnisvg_refresh: bool,
        #[arg(long, default_value = "HF_TOKEN")]
        omnisvg_token_env: String,
        #[arg(long = "holdout-target", value_delimiter = ',')]
        holdout_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        holdout_stride: usize,
        #[arg(long, default_value_t = 0)]
        holdout_offset: usize,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        fit_holdout_static_oracles: bool,
        #[arg(long, default_value = "artifacts/hyper2d_e2e")]
        output_dir: PathBuf,
        #[arg(long)]
        report_output: Option<PathBuf>,
        #[arg(long)]
        scratch_catalog_output: Option<PathBuf>,
        #[arg(long)]
        shared_base_output: Option<PathBuf>,
        #[arg(long)]
        hyper_output: Option<PathBuf>,
        #[arg(long)]
        generated_output_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 10000)]
        target_epochs: usize,
        #[arg(long, default_value_t = 3)]
        target_repetitions: usize,
        #[arg(long, default_value_t = 100)]
        target_report_interval: usize,
        #[arg(long, default_value_t = 8)]
        target_batch_size: usize,
        #[arg(long, default_value_t = 512)]
        target_pool_size: usize,
        #[arg(long, default_value_t = 4096)]
        target_particles: usize,
        #[arg(long, default_value_t = 32)]
        target_step_min: usize,
        #[arg(long, default_value_t = 96)]
        target_step_max: usize,
        #[arg(long, default_value_t = 16)]
        target_inject_seed_interval: usize,
        #[arg(long, default_value_t = 0.5)]
        target_update_prob: f32,
        #[arg(long, default_value_t = 42)]
        target_seed: u64,
        #[arg(long, default_value_t = 42)]
        student_seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 0.1)]
        target_brush_size: f32,
        #[arg(long, default_value_t = 5.0e-4)]
        target_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        target_weight_decay: f32,
        #[arg(long, default_value_t = 0.0)]
        target_grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.9)]
        target_adam_beta1: f32,
        #[arg(long, default_value_t = 0.999)]
        target_adam_beta2: f32,
        #[arg(long, default_value_t = 1.0e-8)]
        target_adam_epsilon: f32,
        #[arg(long = "target-scheduler-milestone", value_delimiter = ',')]
        target_scheduler_milestones: Vec<usize>,
        #[arg(long, default_value_t = 0.3)]
        target_scheduler_gamma: f32,
        #[arg(long, action = ArgAction::SetFalse)]
        target_per_parameter_grad_normalization: bool,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 256)]
        target_loss_image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        target_splat_sigma: f32,
        #[arg(long, default_value_t = 2.0)]
        target_splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        target_color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        target_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        target_displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_bound_regularizer_weight: f32,
        #[arg(long, default_value_t = 16)]
        adapter_rank: usize,
        #[arg(long, default_value_t = 16.0)]
        adapter_alpha: f32,
        #[arg(long, default_value_t = 512)]
        adapter_rows: usize,
        #[arg(long, default_value_t = 256)]
        adapter_train_steps: usize,
        #[arg(long, default_value_t = 1.0e-3)]
        adapter_learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        adapter_grad_clip_norm: f32,
        #[arg(long)]
        adapter_rollout_particles: Option<usize>,
        #[arg(long, default_value_t = 64)]
        adapter_rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        adapter_rollouts: usize,
        #[arg(long, default_value = "summary-tokens")]
        condition_encoder: Hyper2dConditionEncoderArg,
        #[arg(long)]
        dino_model: Option<PathBuf>,
        #[arg(long, default_value_t = 518)]
        dino_image_size: usize,
        #[arg(long, default_value_t = 128)]
        shared_fit_steps: usize,
        #[arg(long, default_value_t = 16)]
        shared_fit_report_interval: usize,
        #[arg(long, default_value_t = 0)]
        shared_fit_example_batch_size: usize,
        #[arg(long, default_value_t = 0.0)]
        shared_fit_adapter_l2: f32,
        #[arg(long, default_value_t = 42)]
        shared_fit_seed: u64,
        #[arg(long, default_value_t = 1.0e-4)]
        shared_fit_base_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        shared_fit_base_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        shared_fit_base_grad_clip_norm: f32,
        #[arg(long, default_value_t = 1.0e-3)]
        shared_fit_adapter_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        shared_fit_adapter_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        shared_fit_adapter_grad_clip_norm: f32,
        #[arg(long, default_value_t = 512)]
        hyper_steps: usize,
        #[arg(long, default_value_t = 1.0e-3)]
        hyper_learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        hyper_grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        hyper_weight_decay: f32,
        #[arg(long, default_value_t = 512)]
        hyper_hidden: usize,
        #[arg(long, default_value_t = 8.0)]
        hyper_output_scale: f32,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_WIDTH)]
        condition_token_grid_width: usize,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_HEIGHT)]
        condition_token_grid_height: usize,
        #[arg(long, default_value_t = 42)]
        hyper_seed: u64,
        #[arg(long, default_value_t = 0)]
        flow_steps: usize,
        #[arg(long, default_value_t = 512)]
        flow_rows: usize,
        #[arg(long)]
        flow_rollout_particles: Option<usize>,
        #[arg(long, default_value_t = 64)]
        flow_rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        flow_rollouts: usize,
        #[arg(long, default_value_t = 0)]
        direct_finetune_steps: usize,
        #[arg(long, default_value_t = 16)]
        direct_finetune_report_interval: usize,
        #[arg(long)]
        direct_finetune_rollout_particles: Option<usize>,
        #[arg(long, default_value_t = 64)]
        direct_finetune_rollout_steps: usize,
        #[arg(long, default_value_t = 42)]
        direct_finetune_seed: u64,
        #[arg(long, default_value_t = 1.0e-4)]
        direct_finetune_hyper_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        direct_finetune_hyper_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        direct_finetune_hyper_grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        direct_finetune_adapter_l2: f32,
        #[arg(long)]
        eval_particles: Option<usize>,
        #[arg(long)]
        eval_steps: Option<usize>,
        #[arg(long, default_value_t = 42)]
        eval_seed: u64,
        #[arg(long)]
        quality_max_static_ratio: Option<f32>,
        #[arg(long)]
        quality_max_hyper_static_ratio: Option<f32>,
        #[arg(long)]
        quality_max_hyper_target_ratio: Option<f32>,
    },
    #[command(
        name = "train-hyper2d-e2e-rollout",
        alias = "train-hyper-2d-e2e-rollout",
        alias = "train-hypernpa2d-e2e-rollout",
        alias = "train-hypernpa2d",
        alias = "train-hyper-npa-2d"
    )]
    TrainHyper2dE2eRollout {
        #[arg(long)]
        config: PathBuf,
    },
    #[command(
        name = "train-hyper2d-direct-basis",
        alias = "train-hyper-2d-direct-basis",
        alias = "train-hyper2d-image-lora-suite",
        hide = true
    )]
    TrainHyper2dDirectBasis {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long = "target-image", value_delimiter = ',')]
        target_images: Vec<PathBuf>,
        #[arg(long = "target-image-dir", value_delimiter = ',')]
        target_image_dirs: Vec<PathBuf>,
        #[arg(long)]
        target_image_recursive: bool,
        #[arg(
            long = "image-extension",
            value_delimiter = ',',
            default_value = "png,jpg,jpeg,webp,bmp,tif,tiff"
        )]
        image_extensions: Vec<String>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, default_value = "assets/catalog_thumbnails")]
        catalog_thumbnail_dir: PathBuf,
        #[arg(long)]
        catalog_group: Option<Hyper2dCatalogGroupArg>,
        #[arg(long = "catalog-target", value_delimiter = ',')]
        catalog_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        catalog_limit: usize,
        #[arg(long)]
        omnisvg_dataset: Option<OmniSvgDatasetArg>,
        #[arg(long, default_value = "train")]
        omnisvg_split: String,
        #[arg(long, default_value = "data/omnisvg")]
        omnisvg_cache_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        omnisvg_offset: usize,
        #[arg(long, default_value_t = 128)]
        omnisvg_limit: usize,
        #[arg(long, default_value_t = 100)]
        omnisvg_page_size: usize,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        omnisvg_download: bool,
        #[arg(long)]
        omnisvg_refresh: bool,
        #[arg(long, default_value = "HF_TOKEN")]
        omnisvg_token_env: String,
        #[arg(long, default_value_t = 0)]
        source_limit: usize,
        #[arg(long = "holdout-target", value_delimiter = ',')]
        holdout_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        holdout_stride: usize,
        #[arg(long, default_value_t = 0)]
        holdout_offset: usize,
        #[arg(long, default_value = "artifacts/hyper2d_direct_basis")]
        output_dir: PathBuf,
        #[arg(long)]
        report_output: Option<PathBuf>,
        #[arg(long)]
        shared_base_output: Option<PathBuf>,
        #[arg(long)]
        adapter_bank_output: Option<PathBuf>,
        #[arg(long)]
        adapter_output_dir: Option<PathBuf>,
        #[arg(long, default_value = "gpu")]
        training_device: TrainingDeviceArg,
        #[arg(long, default_value = "burn-wgpu")]
        gpu_backend: Hyper2dDirectBasisGpuBackendArg,
        #[arg(long, default_value_t = 16)]
        adapter_rank: usize,
        #[arg(long, default_value_t = 16.0)]
        adapter_alpha: f32,
        #[arg(long, default_value_t = 1024)]
        steps: usize,
        #[arg(long, default_value_t = 16)]
        report_interval: usize,
        #[arg(long, default_value_t = 2)]
        example_batch_size: usize,
        #[arg(long, default_value_t = 8)]
        tbptt_chunk_steps: usize,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 42)]
        base_seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 128)]
        target_loss_image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        target_splat_sigma: f32,
        #[arg(long, default_value_t = 2.0)]
        target_splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        target_color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        target_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        target_displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_bound_regularizer_weight: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        per_parameter_grad_normalization: bool,
        #[arg(long, default_value_t = 1.0e-4)]
        base_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        base_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        base_grad_clip_norm: f32,
        #[arg(long, default_value_t = 1.0e-3)]
        adapter_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        adapter_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        adapter_grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        adapter_l2: f32,
        #[arg(long, default_value_t = 0)]
        holdout_adapter_steps: usize,
        #[arg(long, default_value_t = 8)]
        holdout_adapter_batch_size: usize,
        #[arg(long, default_value_t = 0)]
        train_adapter_refine_steps: usize,
        #[arg(long, default_value_t = 0)]
        train_adapter_refine_batch_size: usize,
        #[arg(long)]
        train_adapter_refine_learning_rate: Option<f32>,
        #[arg(long)]
        train_adapter_refine_weight_decay: Option<f32>,
        #[arg(long)]
        train_adapter_refine_grad_clip_norm: Option<f32>,
        #[arg(long)]
        holdout_adapter_learning_rate: Option<f32>,
        #[arg(long)]
        holdout_adapter_weight_decay: Option<f32>,
        #[arg(long)]
        holdout_adapter_grad_clip_norm: Option<f32>,
        #[arg(long, default_value_t = 16)]
        eval_examples: usize,
        #[arg(long)]
        eval_interval: Option<usize>,
        #[arg(long, default_value_t = 1)]
        eval_batch_size: usize,
        #[arg(long, default_value_t = 42)]
        eval_seed: u64,
        #[arg(long)]
        system_memory_budget_gb: Option<f32>,
        #[arg(long)]
        gpu_memory_budget_gb: Option<f32>,
        #[arg(long, default_value_t = 2048)]
        max_dense_train_particles: usize,
        #[arg(long, default_value_t = 4 * 1024 * 1024)]
        max_dense_chunk_floats: usize,
        #[arg(long, default_value_t = 4 * 1024 * 1024)]
        max_splat_chunk_floats: usize,
        #[arg(long, default_value_t = 0)]
        oracle_train_examples: usize,
        #[arg(long, default_value_t = 0)]
        oracle_holdout_examples: usize,
        #[arg(long, default_value_t = 256)]
        oracle_epochs: usize,
        #[arg(long, default_value_t = 1)]
        oracle_repetitions: usize,
        #[arg(long, default_value_t = 64)]
        oracle_report_interval: usize,
        #[arg(long, default_value_t = 4)]
        oracle_batch_size: usize,
        #[arg(long, default_value_t = 128)]
        oracle_pool_size: usize,
        #[arg(long, default_value_t = 5.0e-4)]
        oracle_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        oracle_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        oracle_grad_clip_norm: f32,
        #[arg(long, default_value_t = 42)]
        oracle_seed: u64,
    },
    #[command(
        name = "train-hyper2d-adapter-bank",
        alias = "train-hyper-2d-adapter-bank",
        alias = "train-hyper2d-conditioned-adapters",
        hide = true
    )]
    TrainHyper2dAdapterBank {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        shared_base: Option<PathBuf>,
        #[arg(long)]
        adapter_bank: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper2d_adapter_bank")]
        output_dir: PathBuf,
        #[arg(long)]
        report_output: Option<PathBuf>,
        #[arg(long)]
        hyper_output: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        source_limit: usize,
        #[arg(long, default_value_t = 0)]
        train_limit: usize,
        #[arg(long, default_value_t = 0)]
        holdout_limit: usize,
        #[arg(long, default_value = "summary-tokens")]
        condition_encoder: Hyper2dConditionEncoderArg,
        #[arg(long)]
        dino_model: Option<PathBuf>,
        #[arg(long, default_value_t = 518)]
        dino_image_size: usize,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_WIDTH)]
        condition_token_grid_width: usize,
        #[arg(long, default_value_t = crate::DEFAULT_CONDITION_TOKEN_GRID_HEIGHT)]
        condition_token_grid_height: usize,
        #[arg(long, default_value = "burn-wgpu")]
        backend: Hyper2dAdapterBankBackendArg,
        #[arg(long, default_value_t = 512)]
        hyper_hidden: usize,
        #[arg(long, default_value_t = 0.0)]
        hyper_output_scale: f32,
        #[arg(long, default_value_t = 42)]
        hyper_seed: u64,
        #[arg(long, default_value_t = 512)]
        steps: usize,
        #[arg(long, default_value_t = 32)]
        report_interval: usize,
        #[arg(long, default_value_t = 256)]
        example_batch_size: usize,
        #[arg(long, default_value_t = 1.0e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.9)]
        adam_beta1: f32,
        #[arg(long, default_value_t = 0.999)]
        adam_beta2: f32,
        #[arg(long, default_value_t = 1.0e-8)]
        adam_epsilon: f32,
        #[arg(long, default_value_t = 0)]
        vector_eval_examples: usize,
        #[arg(long, default_value_t = 8)]
        rollout_eval_examples: usize,
        #[arg(long, default_value_t = 2048)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        eval_seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 128)]
        target_loss_image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        target_splat_sigma: f32,
        #[arg(long, default_value_t = 2.0)]
        target_splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        target_color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        target_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        target_displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_bound_regularizer_weight: f32,
    },
    #[command(
        name = "validate-hyper2d-direct-basis-oracles",
        alias = "validate-hyper-2d-direct-basis-oracles",
        hide = true
    )]
    ValidateHyper2dDirectBasisOracles {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        shared_base: Option<PathBuf>,
        #[arg(long)]
        adapter_bank: Option<PathBuf>,
        #[arg(
            long,
            default_value = "artifacts/hyper2d_direct_basis_oracles/report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        eval_seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        per_parameter_grad_normalization: bool,
        #[arg(long, default_value_t = 4096)]
        target_points: usize,
        #[arg(long)]
        target_image_size: Option<usize>,
        #[arg(long, default_value_t = 0.05)]
        target_threshold: f32,
        #[arg(long, default_value_t = 128)]
        target_loss_image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        target_splat_sigma: f32,
        #[arg(long, default_value_t = 2.0)]
        target_splat_loss_weight: f32,
        #[arg(long, default_value_t = 5.0)]
        target_color_loss_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        target_density_loss_weight: f32,
        #[arg(long, default_value_t = 0.01)]
        target_displacement_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_overflow_regularizer_weight: f32,
        #[arg(long, default_value_t = 100.0)]
        target_bound_regularizer_weight: f32,
        #[arg(long, default_value_t = 8)]
        oracle_train_examples: usize,
        #[arg(long, default_value_t = 8)]
        oracle_holdout_examples: usize,
        #[arg(long, default_value_t = 256)]
        oracle_epochs: usize,
        #[arg(long, default_value_t = 1)]
        oracle_repetitions: usize,
        #[arg(long, default_value_t = 64)]
        oracle_report_interval: usize,
        #[arg(long, default_value_t = 4)]
        oracle_batch_size: usize,
        #[arg(long, default_value_t = 128)]
        oracle_pool_size: usize,
        #[arg(long, default_value_t = 5.0e-4)]
        oracle_learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        oracle_weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        oracle_grad_clip_norm: f32,
        #[arg(long, default_value_t = 42)]
        oracle_seed: u64,
    },
    #[command(
        name = "validate-hyper2d-psnr-gate",
        alias = "validate-hyper-2d-psnr-gate",
        hide = true
    )]
    ValidateHyper2dPsnrGate {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long)]
        adapter_bank: Option<PathBuf>,
        #[arg(long)]
        oracle_report: Option<PathBuf>,
        #[arg(long)]
        hyper: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper2d_psnr_gate/report.json")]
        output: PathBuf,
        #[arg(long, default_value = "artifacts/hyper2d_psnr_gate/generated")]
        generated_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, default_value_t = 2048)]
        particles: usize,
        #[arg(long = "step", value_delimiter = ',', default_values_t = [32_usize, 64_usize])]
        steps: Vec<usize>,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 128)]
        image_size: usize,
        #[arg(long, default_value_t = 1.0)]
        render_sigma_px: f32,
        #[arg(long, default_value_t = 26.0)]
        min_render_rgb_psnr_db: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        fail_on_threshold: bool,
    },
    #[command(name = "report-hyper2d", alias = "report-hyper-2d")]
    ReportHyper2d {
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        oracle_report: Option<PathBuf>,
        #[arg(long)]
        psnr_report: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper2d_report")]
        output_dir: PathBuf,
        #[arg(long)]
        summary_output: Option<PathBuf>,
        #[arg(long)]
        markdown_output: Option<PathBuf>,
        #[arg(long)]
        latex_output: Option<PathBuf>,
        #[arg(long)]
        require_quality_ready: bool,
    },
    #[command(name = "infer-hyper2d", alias = "infer-hyper-2d")]
    InferHyper2d {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        condition: PathBuf,
        #[arg(long)]
        hyper: PathBuf,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper_2d_infer_report.json")]
        report_output: PathBuf,
        #[arg(long)]
        adapter_output: Option<PathBuf>,
        #[arg(long)]
        materialized_output: Option<PathBuf>,
        #[arg(long)]
        rollout_output: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long)]
        particles: Option<usize>,
        #[arg(long, default_value_t = 1.0)]
        update_prob: f32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value = "auto")]
        neighbor_mode: NeighborModeArg,
        #[arg(long)]
        bucket_capacity: Option<usize>,
        #[arg(long, default_value_t = 256)]
        min_particles: usize,
        #[arg(long, default_value_t = 4096)]
        max_particles: usize,
        #[arg(long, default_value_t = 0.05)]
        min_seed_scale: f32,
        #[arg(long, default_value_t = 1.0)]
        max_seed_scale: f32,
    },
    #[command(name = "eval-hyper2d", alias = "eval-hyper-2d")]
    EvalHyper2d {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        condition: Option<PathBuf>,
        #[arg(long)]
        target_model: Option<PathBuf>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, default_value = "assets/catalog_thumbnails")]
        catalog_thumbnail_dir: PathBuf,
        #[arg(long)]
        catalog_group: Option<Hyper2dCatalogGroupArg>,
        #[arg(long = "catalog-target", value_delimiter = ',')]
        catalog_targets: Vec<String>,
        #[arg(long, default_value_t = 0)]
        catalog_limit: usize,
        #[arg(long, default_value_t = 0)]
        holdout_stride: usize,
        #[arg(long, default_value_t = 0)]
        holdout_offset: usize,
        #[arg(long)]
        hyper: PathBuf,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/hyper_2d_eval_report.json")]
        report_output: PathBuf,
        #[arg(long)]
        generated_output_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 512)]
        rows: usize,
        #[arg(long)]
        rollout_particles: Option<usize>,
        #[arg(long, default_value_t = 16)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long)]
        rollout_update_prob: Option<f32>,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        image_metrics: bool,
        #[arg(long, default_value_t = 64)]
        image_metric_size: usize,
        #[arg(long, default_value_t = 32)]
        image_metric_steps: usize,
        #[arg(long)]
        image_metric_particles: Option<usize>,
        #[arg(long)]
        image_metric_update_prob: Option<f32>,
        #[arg(long, default_value_t = 1.25)]
        image_metric_sigma: f32,
        #[arg(long, default_value_t = 0.05)]
        image_metric_threshold: f32,
        #[arg(long)]
        dynamics_metrics: bool,
        #[arg(long, default_value_t = 1024)]
        dynamics_metric_particles: usize,
        #[arg(long, default_value_t = 32)]
        dynamics_metric_steps: usize,
        #[arg(long)]
        dynamics_metric_update_prob: Option<f32>,
        #[arg(long, default_value_t = 64)]
        dynamics_metric_image_size: usize,
        #[arg(long, default_value_t = 1.25)]
        dynamics_metric_sigma: f32,
    },
    TrainTorus3d {
        #[arg(long, default_value = "artifacts/legacy_uv_torus_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_uv_torus_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 512)]
        steps: usize,
    },
    TrainTorusMorphogen3d {
        #[arg(long, default_value = "artifacts/legacy_uv_torus_morphogen_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_uv_torus_morphogen_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 96)]
        steps: usize,
        #[arg(long, default_value = "rollout-local")]
        training_mode: MeshTrainingModeArg,
        #[arg(long, default_value_t = 2048)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 128)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
    },
    TrainTeapotMorphogen3d {
        #[arg(long, default_value = "artifacts/legacy_teapot_morphogen_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_teapot_morphogen_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value = "rollout-local")]
        training_mode: MeshTrainingModeArg,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 64)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
    },
    #[command(name = "ablate-local-3d", alias = "ablate-local3d")]
    AblateLocal3d {
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "/tmp/burn_automata_conditionless_local_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/conditionless_local_3d_ablation_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 64)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 4)]
        temporal_samples: usize,
        #[arg(long, default_value_t = 4)]
        training_rounds: usize,
        #[arg(long, default_value_t = DEFAULT_3D_MESH_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 0x010c_a13d)]
        student_seed: u64,
        #[arg(long, default_value_t = 6.0e-5)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.08)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long)]
        motion_gain: Option<f32>,
        #[arg(long, default_value_t = 0.08)]
        max_update_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        density_gain: f32,
        #[arg(long, default_value_t = LOCAL_GROWTH_EXPANSION_GAIN)]
        expansion_gain: f32,
        #[arg(long, default_value_t = 0.35)]
        coverage_gain: f32,
        #[arg(long, default_value_t = 4096)]
        coverage_samples: usize,
        #[arg(long, default_value = "sliced-ot")]
        coverage_mode: CoverageUpdateModeArg,
        #[arg(long, default_value_t = 0.0)]
        coverage_softness: f32,
        #[arg(long, default_value_t = 0.2)]
        coverage_repulsion_gain: f32,
        #[arg(long)]
        coverage_gap_gain: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        coverage_repulsion_radius: f32,
        #[arg(long, default_value_t = 0.0)]
        coverage_normal_weight: f32,
        #[arg(long, default_value_t = 0.2)]
        extent_gain: f32,
        #[arg(long)]
        color_gain: Option<f32>,
        #[arg(long, default_value_t = 0.5)]
        aux_state_gain: f32,
        #[arg(long, default_value_t = 0.02)]
        opacity_gain: f32,
        #[arg(long, default_value_t = 0.06)]
        front_opacity_gain: f32,
        #[arg(long, default_value_t = 0.24)]
        front_radius: f32,
        #[arg(long, default_value_t = 0.2)]
        front_max_opacity_update: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        front_motion_gate: bool,
        #[arg(long, default_value_t = false, action = ArgAction::Set)]
        preserve_opacity_update: bool,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "render-loss-3d", alias = "render3d")]
    RenderLoss3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long, default_value = "artifacts/render_loss_3d_report.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 0x0051_a73d)]
        seed: u64,
        #[arg(long = "extra-seed", value_delimiter = ',')]
        extra_seeds: Vec<u64>,
        #[arg(long, default_value_t = DEFAULT_3D_MESH_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long, default_value_t = 0.75)]
        min_sigma: f32,
        #[arg(long, default_value_t = 5.0)]
        max_sigma: f32,
        #[arg(long, default_value = "fixed-sh0")]
        gaussian_decode_mode: RenderGaussianDecodeModeArg,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "validate-growth3d", alias = "validate-3d-growth")]
    ValidateGrowth3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long, default_value = "artifacts/growth_3d_validation_report.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 256)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long = "extra-seed", value_delimiter = ',')]
        extra_seeds: Vec<u64>,
        #[arg(long, default_value_t = DEFAULT_3D_MESH_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long, default_value_t = 0.75)]
        min_sigma: f32,
        #[arg(long, default_value_t = 5.0)]
        max_sigma: f32,
        #[arg(long, default_value = "fixed-sh0")]
        gaussian_decode_mode: RenderGaussianDecodeModeArg,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long, default_value = "strict")]
        gate: Growth3dValidationGateArg,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "retime-growth3d", alias = "retime-growth-3d")]
    RetimeGrowth3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = LOCAL_GROWTH_OPACITY_GAIN)]
        front_gain: f32,
        #[arg(long)]
        hidden: Option<usize>,
        #[arg(long)]
        skip_front_retime: bool,
        #[arg(long)]
        active_opacity_gain: Option<f32>,
        #[arg(long)]
        active_opacity_hidden: Option<usize>,
        #[arg(long)]
        opacity_bias: Option<f32>,
        #[arg(long)]
        material_opacity_bias: Option<f32>,
        #[arg(long)]
        alpha: Option<f32>,
    },
    #[command(name = "train-render3d", alias = "train-render-3d")]
    TrainRender3d {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/render_trained_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/render_trained_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 32)]
        supervised_steps_per_round: usize,
        #[arg(long, default_value_t = 512)]
        particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 64)]
        gradient_particles: usize,
        #[arg(long, default_value = "analytic")]
        gradient_mode: RenderGradientModeArg,
        #[arg(long, default_value_t = 1.0e-3)]
        finite_diff_eps: f32,
        #[arg(long, default_value_t = 0.35)]
        motion_gain: f32,
        #[arg(long, default_value_t = 0.05)]
        perception_position_gain: f32,
        #[arg(long, default_value_t = 1.0)]
        max_update_norm: f32,
        #[arg(long, default_value_t = true)]
        trajectory_supervision: bool,
        #[arg(long, default_value_t = ROBUST_3D_TRAJECTORY_RENDER_GAIN)]
        trajectory_render_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_TRAJECTORY_MESH_GAIN)]
        trajectory_mesh_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_TRAJECTORY_RENDER_SAMPLES)]
        trajectory_render_samples: usize,
        #[arg(long, default_value_t = ROBUST_3D_LIVENESS_GAIN)]
        liveness_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_LIVENESS_FRONT_RADIUS)]
        liveness_front_radius: f32,
        #[arg(long, default_value_t = ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER)]
        liveness_update_multiplier: f32,
        #[arg(long, default_value_t = ROBUST_3D_COVERAGE_GAIN)]
        coverage_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_COVERAGE_SAMPLES)]
        coverage_samples: usize,
        #[arg(long, default_value = "sliced-ot")]
        coverage_mode: CoverageUpdateModeArg,
        #[arg(long, default_value_t = 0.0)]
        coverage_softness: f32,
        #[arg(long, default_value_t = ROBUST_3D_COVERAGE_REPULSION_GAIN)]
        coverage_repulsion_gain: f32,
        #[arg(long)]
        coverage_gap_gain: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        coverage_repulsion_radius: f32,
        #[arg(long, default_value_t = ROBUST_3D_COVERAGE_NORMAL_WEIGHT)]
        coverage_normal_weight: f32,
        #[arg(long, default_value_t = ROBUST_3D_EXTENT_GAIN)]
        extent_gain: f32,
        #[arg(long)]
        full_coverage_adjoint: bool,
        #[arg(long)]
        no_full_coverage_adjoint: bool,
        #[arg(long, default_value_t = ROBUST_3D_SURFACE_GAIN)]
        surface_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_SURFACE_ESCAPE_GAIN)]
        surface_escape_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_OPACITY_GAIN)]
        opacity_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_MATERIAL_LIVENESS_GAIN)]
        material_liveness_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_MATERIAL_TAIL_GAIN)]
        material_tail_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER)]
        material_suppression_update_multiplier: f32,
        #[arg(long, default_value_t = ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE)]
        material_max_opacity_update: f32,
        #[arg(long, default_value_t = ROBUST_3D_SCALE_GAIN)]
        scale_gain: f32,
        #[arg(long, default_value_t = ROBUST_3D_SCALE_BUDGET_WEIGHT)]
        scale_budget_weight: f32,
        #[arg(long, default_value_t = 0.05)]
        max_opacity_update: f32,
        #[arg(long, default_value_t = 5.0e-4)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP)]
        direct_output_gradient_rms_cap: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        direct_line_search: bool,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "0.0625,0.125,0.25,0.5,1,2,4,8,16,32"
        )]
        direct_line_search_scales: Vec<f32>,
        #[arg(long)]
        direct_material_output_only: bool,
        #[arg(long, default_value = "direct-rollout")]
        training_backend: RenderTrainingBackendArg,
        #[arg(long, default_value = "adapter")]
        weight_update_mode: RenderWeightUpdateModeArg,
        #[arg(long, default_value_t = 8)]
        adapter_rank: usize,
        #[arg(long, default_value_t = 8.0)]
        adapter_alpha: f32,
        #[arg(long, default_value_t = 0x00ad_a973)]
        adapter_seed: u64,
        #[arg(long)]
        direct_selection_seed_training: bool,
        #[arg(long)]
        no_direct_selection_seed_training: bool,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 0x0051_a73d)]
        selection_seed: u64,
        #[arg(long = "extra-selection-seed", value_delimiter = ',')]
        extra_selection_seeds: Vec<u64>,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long, default_value_t = 0.75)]
        min_sigma: f32,
        #[arg(long, default_value_t = 5.0)]
        max_sigma: f32,
        #[arg(long, default_value = "fixed-sh0")]
        gaussian_decode_mode: RenderGaussianDecodeModeArg,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(
        name = "train-render3d-adapters",
        alias = "train-render-3d-adapters",
        alias = "train-render3d-lora-suite"
    )]
    TrainRender3dAdapters {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long)]
        shared_base_output: Option<PathBuf>,
        #[arg(long)]
        shared_base_cycles: Option<usize>,
        #[arg(long, default_value_t = 0x005a_173d)]
        shared_base_seed: u64,
        #[arg(long, default_value = "many")]
        target_set: MeshTargetSetArg,
        #[arg(long, value_delimiter = ',')]
        targets: Vec<MeshTargetArg>,
        #[arg(long, value_delimiter = ',')]
        holdout_targets: Vec<MeshTargetArg>,
        #[arg(long, default_value_t = 0)]
        auto_holdout_stride: usize,
        #[arg(long, default_value_t = 3)]
        auto_holdout_offset: usize,
        #[arg(long, default_value = "artifacts/render_3d_adapter_suite")]
        output_dir: PathBuf,
        #[arg(long, default_value = "artifacts/render_3d_adapter_suite_report.json")]
        report_output: PathBuf,
        #[arg(long)]
        adapter_bank_output: Option<PathBuf>,
        #[arg(long)]
        skip_shared_base_eval: bool,
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 32)]
        supervised_steps_per_round: usize,
        #[arg(long, default_value_t = 512)]
        particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 64)]
        gradient_particles: usize,
        #[arg(long, default_value = "analytic")]
        gradient_mode: RenderGradientModeArg,
        #[arg(long, default_value_t = 1.0e-3)]
        finite_diff_eps: f32,
        #[arg(long, default_value_t = 0.35)]
        motion_gain: f32,
        #[arg(long, default_value_t = 0.05)]
        perception_position_gain: f32,
        #[arg(long, default_value_t = 1.0)]
        max_update_norm: f32,
        #[arg(long, default_value_t = true)]
        trajectory_supervision: bool,
        #[arg(long, default_value = "direct-rollout")]
        training_backend: RenderTrainingBackendArg,
        #[arg(long, default_value_t = 8)]
        adapter_rank: usize,
        #[arg(long, default_value_t = 8.0)]
        adapter_alpha: f32,
        #[arg(long, default_value_t = 0x00ad_a973)]
        adapter_seed: u64,
        #[arg(long, default_value_t = 5.0e-4)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP)]
        direct_output_gradient_rms_cap: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        direct_line_search: bool,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "0.0625,0.125,0.25,0.5,1,2,4,8,16,32"
        )]
        direct_line_search_scales: Vec<f32>,
        #[arg(long)]
        direct_material_output_only: bool,
        #[arg(long)]
        direct_selection_seed_training: bool,
        #[arg(long)]
        no_direct_selection_seed_training: bool,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 0x0051_a73d)]
        selection_seed: u64,
        #[arg(long = "extra-selection-seed", value_delimiter = ',')]
        extra_selection_seeds: Vec<u64>,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long, default_value_t = 0.75)]
        min_sigma: f32,
        #[arg(long, default_value_t = 5.0)]
        max_sigma: f32,
        #[arg(long, default_value = "fixed-sh0")]
        gaussian_decode_mode: RenderGaussianDecodeModeArg,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long)]
        fail_on_validation: bool,
    },
    Import {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Bench {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        particles: usize,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        repeats: usize,
        #[arg(long, default_value_t = 1.0)]
        update_prob: f32,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value = "auto")]
        neighbor_mode: NeighborModeArg,
        #[arg(long)]
        bucket_capacity: Option<usize>,
        #[arg(long)]
        profile: bool,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        normalize_seed_scale: bool,
        #[arg(long, alias = "no-normalize-seed-scale")]
        fixed_eps: bool,
        #[arg(long)]
        reference_seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "seed")]
        geometry: BenchGeometryArg,
        #[arg(long)]
        gaussian: bool,
        #[arg(long)]
        step_timing: bool,
    },
    #[command(name = "bench-training", alias = "training-bench")]
    BenchTraining {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long)]
        target_model: Option<PathBuf>,
        #[arg(long, default_value_t = 32768)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 2)]
        warmup_steps: usize,
        #[arg(long, default_value_t = 0)]
        report_interval: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value = "adam-w")]
        optimizer: TrainingOptimizerArg,
        #[arg(long, default_value = "auto")]
        training_device: TrainingDeviceArg,
        #[arg(long, default_value_t = 0.9)]
        adam_beta1: f32,
        #[arg(long, default_value_t = 0.999)]
        adam_beta2: f32,
        #[arg(long, default_value_t = 1e-8)]
        adam_epsilon: f32,
        #[arg(long, default_value_t = 7)]
        student_seed: u64,
        #[arg(long, default_value = "features")]
        batch_source: TrainingBatchArg,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 16)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 5)]
        temporal_samples: usize,
        #[arg(long, default_value_t = 1.0)]
        rollout_update_prob: f32,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "/tmp/burn_automata_training_bench.json")]
        output: PathBuf,
    },
    #[command(name = "bench-spatial", alias = "spatial-bench")]
    BenchSpatial {
        #[arg(long, default_value = "growing-3d-gs")]
        preset: PresetArg,
        #[arg(long, default_value_t = 8192)]
        particles: usize,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        normalize_seed_scale: bool,
        #[arg(long)]
        fixed_eps: bool,
        #[arg(long)]
        reference_seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "seed")]
        geometry: BenchGeometryArg,
        #[arg(long, default_value = "all")]
        strategy: SpatialStrategyArg,
        #[arg(long, default_value_t = 16)]
        bvh_leaf_size: usize,
        #[arg(long, default_value = "2,2,1")]
        tile_size: String,
    },
    #[command(name = "materialize-adapter", alias = "materialize-lora")]
    MaterializeAdapter {
        #[arg(long)]
        base_model: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    #[command(name = "build-exact-adapter-bank", alias = "exact-oracle-adapter-bank")]
    BuildExactAdapterBank {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long)]
        source_adapter_bank: Option<PathBuf>,
        #[arg(long)]
        oracle_report: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/exact_adapter_bank")]
        output_dir: PathBuf,
        #[arg(long)]
        adapter_bank_output: Option<PathBuf>,
        #[arg(long)]
        rank: Option<usize>,
        #[arg(long)]
        alpha: Option<f32>,
        #[arg(long)]
        force_split: Option<String>,
    },
    Manifest {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value = "/tmp/burn_automata_seed_model.bpk")]
        output: PathBuf,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_help_shows_canonical_hypernpa_trainer_only() {
        let mut command = <CliArgs as clap::CommandFactory>::command();
        let help = command.render_long_help().to_string();

        assert!(help.contains("train-hyper2d-e2e-rollout"));
        assert!(!help.contains("train-target2d"));
        assert!(!help.contains("train-hyper2d-direct-basis"));
        assert!(!help.contains("train-hyper2d-adapter-bank"));
        assert!(!help.contains("validate-hyper2d-direct-basis-oracles"));
        assert!(!help.contains("validate-hyper2d-psnr-gate"));
    }

    #[test]
    fn canonical_hypernpa_alias_parses_to_e2e_rollout_trainer() {
        let args = CliArgs::try_parse_from([
            "burn_automata",
            "train-hypernpa2d",
            "--config",
            "configs/verified/2d/hyper_e2e/smoke_lizard_dino_online.toml",
        ])
        .unwrap();

        let Command::TrainHyper2dE2eRollout { config } = args.command else {
            panic!("train-hypernpa2d should parse to TrainHyper2dE2eRollout");
        };
        assert_eq!(
            config,
            PathBuf::from("configs/verified/2d/hyper_e2e/smoke_lizard_dino_online.toml")
        );
    }

    #[test]
    fn hidden_legacy_target2d_command_remains_explicitly_callable() {
        let args = CliArgs::try_parse_from([
            "burn_automata",
            "train-target2d",
            "--experimental",
            "--target-image",
            "assets/catalog_thumbnails/lizard.png",
        ])
        .unwrap();

        assert!(matches!(
            args.command,
            Command::TrainTarget2d {
                experimental: true,
                ..
            }
        ));
    }
}
