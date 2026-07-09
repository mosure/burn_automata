use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use super::e2e::{E2eHyperNpa2d, PerceptionRolloutBackend, Target2dLossBackend};
use crate::{
    AdamWConfig, NpaConfig, NpaLowRankAdapter, ParticleSeed, SgdConfig, Target2dLossConfig,
    TargetImage2d,
};

pub(crate) mod dense;

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct DirectBasisTrainingExample {
    pub(crate) target: TargetImage2d,
    pub(crate) adapter: NpaLowRankAdapter,
    pub(crate) last_train_loss: Option<f32>,
    pub(crate) particle_count: Option<usize>,
    pub(crate) update_prob: Option<f32>,
    pub(crate) seed_scale: Option<f32>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct DirectBasisTrainConfig {
    pub(crate) steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) example_batch_size: usize,
    pub(crate) tbptt_chunk_steps: usize,
    pub(crate) loss_on_final_chunk_only: bool,
    pub(crate) use_particle_pool: bool,
    pub(crate) pool_size: usize,
    pub(crate) inject_seed_interval: usize,
    pub(crate) brush_size: f32,
    pub(crate) stopgrad_pos: bool,
    pub(crate) stopgrad_state: bool,
    pub(crate) rollout_particles: usize,
    pub(crate) rollout_step_min: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) grid_eps: f32,
    pub(crate) motion_scale: f32,
    pub(crate) loss_config: Target2dLossConfig,
    pub(crate) target2d_loss_backend: Target2dLossBackend,
    pub(crate) perception_backend: PerceptionRolloutBackend,
    pub(crate) per_parameter_grad_normalization: bool,
    pub(crate) base_sgd: SgdConfig,
    pub(crate) adapter_sgd: SgdConfig,
    pub(crate) adapter_l2_weight: f32,
    pub(crate) update_base: bool,
    pub(crate) eval_examples: usize,
    pub(crate) eval_interval: usize,
    pub(crate) eval_batch_size: usize,
    pub(crate) eval_seed: u64,
    pub(crate) system_memory_budget_gb: Option<f32>,
    pub(crate) gpu_memory_budget_gb: Option<f32>,
    pub(crate) max_dense_train_particles: usize,
    pub(crate) max_dense_chunk_floats: usize,
    pub(crate) max_splat_chunk_floats: usize,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct Target2dBurnCheckpointConfig {
    pub(crate) current_model_output: PathBuf,
    pub(crate) best_model_output: PathBuf,
    pub(crate) metadata_output: PathBuf,
    pub(crate) model_config: NpaConfig,
    pub(crate) hashgrid: burn_automata_kernels::HashGridConfig,
    pub(crate) source: String,
    pub(crate) interval_steps: usize,
    pub(crate) interval_duration: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DirectBasisStepStats {
    pub(crate) loss: f32,
    pub(crate) base_grad_norm: f32,
    pub(crate) base_grad_scale: f32,
    pub(crate) mean_adapter_grad_norm: f32,
    pub(crate) max_adapter_grad_norm: f32,
    pub(crate) examples_seen: usize,
    pub(crate) particle_steps_per_sec: f64,
    pub(crate) elapsed_ms: f64,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Hyper2dDirectBasisLossSummary {
    pub(crate) examples: usize,
    pub(crate) mean_total_loss: f32,
    pub(crate) max_total_loss: f32,
    pub(crate) mean_splat_loss: f32,
    pub(crate) mean_color_loss: f32,
    pub(crate) mean_density_loss: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Hyper2dDirectBasisHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) eval_loss: Option<Hyper2dDirectBasisLossSummary>,
    pub(crate) base_grad_norm: f32,
    pub(crate) base_grad_scale: f32,
    pub(crate) mean_adapter_grad_norm: f32,
    pub(crate) max_adapter_grad_norm: f32,
    pub(crate) examples_seen: usize,
    pub(crate) particle_steps_per_sec: f64,
    pub(crate) elapsed_ms: f64,
}

pub(crate) struct BurnWgpuDirectBasisOutput {
    pub(crate) backend: &'static str,
    pub(crate) device: String,
    pub(crate) metrics: serde_json::Value,
    pub(crate) history: Vec<Hyper2dDirectBasisHistoryEntry>,
    pub(crate) train_refine_history: Vec<Hyper2dDirectBasisHistoryEntry>,
    pub(crate) holdout_history: Vec<Hyper2dDirectBasisHistoryEntry>,
    pub(crate) best_train_loss: Option<f32>,
    pub(crate) best_train_step: usize,
}

pub(crate) struct BurnDenseOracleBatchOutput {
    pub(crate) backend: &'static str,
    pub(crate) device: String,
    pub(crate) metrics: serde_json::Value,
    pub(crate) history: Vec<Hyper2dDirectBasisHistoryEntry>,
    pub(crate) per_model_history: Vec<Vec<Hyper2dDirectBasisHistoryEntry>>,
    pub(crate) best_train_loss: Vec<Option<f32>>,
    pub(crate) best_train_step: Vec<usize>,
}

#[allow(dead_code)]
pub(crate) struct BurnE2eRolloutExample {
    pub(crate) slug: String,
    pub(crate) target: TargetImage2d,
    pub(crate) condition_features: Vec<f32>,
    pub(crate) token_count: usize,
    pub(crate) embed_dims: usize,
    pub(crate) particle_count: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed_scale: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum E2eLrSchedule {
    #[default]
    Constant,
    Cosine,
    Linear,
}

impl E2eLrSchedule {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Cosine => "cosine",
            Self::Linear => "linear",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum E2eTbpttLossMode {
    #[default]
    AllChunks,
    FinalOnly,
    EndpointWeighted,
}

impl E2eTbpttLossMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::AllChunks => "all-chunks",
            Self::FinalOnly => "final-only",
            Self::EndpointWeighted => "endpoint-weighted",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct BurnE2eRolloutTrainConfig {
    pub(crate) steps: usize,
    pub(crate) report_interval: usize,
    pub(crate) example_batch_size: usize,
    pub(crate) tbptt_chunk_steps: usize,
    pub(crate) loss_on_final_chunk_only: bool,
    pub(crate) tbptt_loss_mode: E2eTbpttLossMode,
    pub(crate) tbptt_intermediate_loss_weight: f32,
    pub(crate) tbptt_final_loss_weight: f32,
    pub(crate) use_particle_pool: bool,
    pub(crate) pool_slots_per_example: usize,
    pub(crate) inject_seed_interval: usize,
    pub(crate) brush_size: f32,
    pub(crate) pre_rollout_steps: usize,
    pub(crate) rollout_particles: usize,
    pub(crate) rollout_step_min: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) grid_eps: f32,
    pub(crate) motion_scale: f32,
    pub(crate) loss_config: Target2dLossConfig,
    pub(crate) target2d_loss_backend: Target2dLossBackend,
    pub(crate) perception_backend: PerceptionRolloutBackend,
    pub(crate) per_parameter_grad_normalization: bool,
    pub(crate) shared_base_trainable: bool,
    pub(crate) shared_base_train_start_step: usize,
    pub(crate) base_optimizer: AdamWConfig,
    pub(crate) generator_optimizer: AdamWConfig,
    pub(crate) lr_schedule: E2eLrSchedule,
    pub(crate) min_lr_scale: f32,
    pub(crate) adapter_rank: usize,
    pub(crate) adapter_alpha: f32,
    pub(crate) spatial_token_generator: bool,
    pub(crate) adapter_chunk_size: usize,
    pub(crate) generator_hidden_dims: usize,
    pub(crate) token_attention_heads: usize,
    pub(crate) generator_sample_steps: usize,
    pub(crate) generator_output_scale: f32,
    pub(crate) generator_init_scale: f32,
    pub(crate) stopgrad_pos: bool,
    pub(crate) stopgrad_state: bool,
    pub(crate) system_memory_budget_gb: Option<f32>,
    pub(crate) gpu_memory_budget_gb: Option<f32>,
    pub(crate) max_dense_train_particles: usize,
    pub(crate) max_dense_chunk_floats: usize,
    pub(crate) max_splat_chunk_floats: usize,
    pub(crate) condition_device_cache_max_bytes: usize,
    pub(crate) validation_examples: usize,
    pub(crate) validation_interval: usize,
    pub(crate) validation_particles: usize,
    pub(crate) validation_steps: usize,
    pub(crate) validation_update_prob: f32,
    pub(crate) validation_seed: u64,
    pub(crate) validation_psnr_threshold_db: f32,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct BurnE2eRolloutHistoryEntry {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) learning_rate_scale: f32,
    pub(crate) base_learning_rate: f32,
    pub(crate) generator_learning_rate: f32,
    pub(crate) holdout_mean_psnr_db: Option<f32>,
    pub(crate) holdout_mean_loss: Option<f32>,
    pub(crate) base_grad_norm: f32,
    pub(crate) base_grad_scale: f32,
    pub(crate) generator_grad_norm: f32,
    pub(crate) generator_grad_scale: f32,
    pub(crate) examples_seen: usize,
    pub(crate) pool_seed_replacements: usize,
    pub(crate) particle_steps_per_sec: f64,
    pub(crate) dense_pair_interactions_per_sec: f64,
    pub(crate) elapsed_ms: f64,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct BurnE2eRolloutQualityEntry {
    pub(crate) slug: String,
    pub(crate) total_loss: f32,
    pub(crate) splat_loss: f32,
    pub(crate) color_loss: f32,
    pub(crate) density_loss: f32,
    pub(crate) render_rgb_mse: f32,
    pub(crate) render_rgb_psnr_db: f32,
    pub(crate) passed: bool,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct BurnE2eRolloutQualityReport {
    pub(crate) split: &'static str,
    pub(crate) examples: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed: u64,
    pub(crate) psnr_threshold_db: f32,
    pub(crate) passed: bool,
    pub(crate) mean_passed: bool,
    pub(crate) all_examples_passed: bool,
    pub(crate) elapsed_ms: f64,
    pub(crate) particle_steps: f64,
    pub(crate) particle_steps_per_sec: f64,
    pub(crate) dense_pair_interactions_per_sec: f64,
    pub(crate) adapter_batches: usize,
    pub(crate) mean_total_loss: f32,
    pub(crate) mean_splat_loss: f32,
    pub(crate) mean_color_loss: f32,
    pub(crate) mean_density_loss: f32,
    pub(crate) mean_render_rgb_mse: f32,
    pub(crate) mean_render_rgb_psnr_db: f32,
    pub(crate) min_render_rgb_psnr_db: f32,
    pub(crate) max_render_rgb_psnr_db: f32,
    pub(crate) mean_condition_shuffle_render_rgb_psnr_db: Option<f32>,
    pub(crate) condition_shuffle_psnr_gap_db: Option<f32>,
    pub(crate) entries: Vec<BurnE2eRolloutQualityEntry>,
}

pub(crate) struct BurnE2eRolloutOutput {
    pub(crate) backend: String,
    pub(crate) device: String,
    pub(crate) metrics: serde_json::Value,
    pub(crate) history: Vec<BurnE2eRolloutHistoryEntry>,
    pub(crate) final_loss: Option<f32>,
    pub(crate) generator: E2eHyperNpa2d,
    pub(crate) quality_validation: Option<BurnE2eRolloutQualityReport>,
}
