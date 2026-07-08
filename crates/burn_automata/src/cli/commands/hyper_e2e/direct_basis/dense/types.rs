use super::super::Target2dLossBackend;

pub(in crate::cli::commands::hyper_e2e::direct_basis) struct BurnWgpuDirectBasisOutput {
    pub(in crate::cli::commands::hyper_e2e::direct_basis) backend: &'static str,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) device: String,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) metrics: serde_json::Value,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) history:
        Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) train_refine_history:
        Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) holdout_history:
        Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) best_train_loss: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) best_train_step: usize,
}

pub(in crate::cli::commands::hyper_e2e::direct_basis) struct BurnDenseOracleBatchOutput {
    pub(in crate::cli::commands::hyper_e2e::direct_basis) backend: &'static str,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) device: String,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) metrics: serde_json::Value,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) history:
        Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) per_model_history:
        Vec<Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) best_train_loss: Vec<Option<f32>>,
    pub(in crate::cli::commands::hyper_e2e::direct_basis) best_train_step: Vec<usize>,
}

#[allow(dead_code)]
pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutExample {
    pub(in crate::cli::commands::hyper_e2e) slug: String,
    pub(in crate::cli::commands::hyper_e2e) target: crate::TargetImage2d,
    pub(in crate::cli::commands::hyper_e2e) condition_features: Vec<f32>,
    pub(in crate::cli::commands::hyper_e2e) token_count: usize,
    pub(in crate::cli::commands::hyper_e2e) embed_dims: usize,
    pub(in crate::cli::commands::hyper_e2e) particle_count: usize,
    pub(in crate::cli::commands::hyper_e2e) update_prob: f32,
    pub(in crate::cli::commands::hyper_e2e) seed_scale: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::cli::commands::hyper_e2e) enum E2eLrSchedule {
    #[default]
    Constant,
    Cosine,
    Linear,
}

impl E2eLrSchedule {
    pub(in crate::cli::commands::hyper_e2e) fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Cosine => "cosine",
            Self::Linear => "linear",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutTrainConfig {
    pub(in crate::cli::commands::hyper_e2e) steps: usize,
    pub(in crate::cli::commands::hyper_e2e) report_interval: usize,
    pub(in crate::cli::commands::hyper_e2e) example_batch_size: usize,
    pub(in crate::cli::commands::hyper_e2e) tbptt_chunk_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) loss_on_final_chunk_only: bool,
    pub(in crate::cli::commands::hyper_e2e) rollout_particles: usize,
    pub(in crate::cli::commands::hyper_e2e) rollout_step_min: usize,
    pub(in crate::cli::commands::hyper_e2e) rollout_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) update_prob: f32,
    pub(in crate::cli::commands::hyper_e2e) seed: u64,
    pub(in crate::cli::commands::hyper_e2e) seed_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) seed_mode: crate::ParticleSeed,
    pub(in crate::cli::commands::hyper_e2e) grid_eps: f32,
    pub(in crate::cli::commands::hyper_e2e) motion_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) loss_config: crate::Target2dLossConfig,
    pub(in crate::cli::commands::hyper_e2e) target2d_loss_backend: Target2dLossBackend,
    pub(in crate::cli::commands::hyper_e2e) per_parameter_grad_normalization: bool,
    pub(in crate::cli::commands::hyper_e2e) shared_base_trainable: bool,
    pub(in crate::cli::commands::hyper_e2e) shared_base_train_start_step: usize,
    pub(in crate::cli::commands::hyper_e2e) base_optimizer: crate::AdamWConfig,
    pub(in crate::cli::commands::hyper_e2e) generator_optimizer: crate::AdamWConfig,
    pub(in crate::cli::commands::hyper_e2e) lr_schedule: E2eLrSchedule,
    pub(in crate::cli::commands::hyper_e2e) min_lr_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) adapter_rank: usize,
    pub(in crate::cli::commands::hyper_e2e) adapter_alpha: f32,
    pub(in crate::cli::commands::hyper_e2e) generator_hidden_dims: usize,
    pub(in crate::cli::commands::hyper_e2e) token_attention_heads: usize,
    pub(in crate::cli::commands::hyper_e2e) generator_sample_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) generator_output_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) generator_init_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) stopgrad_pos: bool,
    pub(in crate::cli::commands::hyper_e2e) stopgrad_state: bool,
    pub(in crate::cli::commands::hyper_e2e) system_memory_budget_gb: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e) gpu_memory_budget_gb: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e) max_dense_train_particles: usize,
    pub(in crate::cli::commands::hyper_e2e) max_dense_chunk_floats: usize,
    pub(in crate::cli::commands::hyper_e2e) max_splat_chunk_floats: usize,
    pub(in crate::cli::commands::hyper_e2e) condition_device_cache_max_bytes: usize,
    pub(in crate::cli::commands::hyper_e2e) validation_examples: usize,
    pub(in crate::cli::commands::hyper_e2e) validation_interval: usize,
    pub(in crate::cli::commands::hyper_e2e) validation_particles: usize,
    pub(in crate::cli::commands::hyper_e2e) validation_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) validation_update_prob: f32,
    pub(in crate::cli::commands::hyper_e2e) validation_seed: u64,
    pub(in crate::cli::commands::hyper_e2e) validation_psnr_threshold_db: f32,
}

#[derive(Clone, serde::Serialize)]
pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutHistoryEntry {
    pub(in crate::cli::commands::hyper_e2e) step: usize,
    pub(in crate::cli::commands::hyper_e2e) loss: f32,
    pub(in crate::cli::commands::hyper_e2e) learning_rate_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) base_learning_rate: f32,
    pub(in crate::cli::commands::hyper_e2e) generator_learning_rate: f32,
    pub(in crate::cli::commands::hyper_e2e) holdout_mean_psnr_db: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e) holdout_mean_loss: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e) base_grad_norm: f32,
    pub(in crate::cli::commands::hyper_e2e) base_grad_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) generator_grad_norm: f32,
    pub(in crate::cli::commands::hyper_e2e) generator_grad_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) examples_seen: usize,
    pub(in crate::cli::commands::hyper_e2e) particle_steps_per_sec: f64,
    pub(in crate::cli::commands::hyper_e2e) dense_pair_interactions_per_sec: f64,
    pub(in crate::cli::commands::hyper_e2e) elapsed_ms: f64,
}

#[derive(Clone, serde::Serialize)]
pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutQualityEntry {
    pub(in crate::cli::commands::hyper_e2e) slug: String,
    pub(in crate::cli::commands::hyper_e2e) total_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) splat_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) color_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) density_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) render_rgb_mse: f32,
    pub(in crate::cli::commands::hyper_e2e) render_rgb_psnr_db: f32,
    pub(in crate::cli::commands::hyper_e2e) passed: bool,
}

#[derive(Clone, serde::Serialize)]
pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutQualityReport {
    pub(in crate::cli::commands::hyper_e2e) split: &'static str,
    pub(in crate::cli::commands::hyper_e2e) examples: usize,
    pub(in crate::cli::commands::hyper_e2e) particle_count: usize,
    pub(in crate::cli::commands::hyper_e2e) rollout_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) update_prob: f32,
    pub(in crate::cli::commands::hyper_e2e) seed: u64,
    pub(in crate::cli::commands::hyper_e2e) psnr_threshold_db: f32,
    pub(in crate::cli::commands::hyper_e2e) passed: bool,
    pub(in crate::cli::commands::hyper_e2e) mean_passed: bool,
    pub(in crate::cli::commands::hyper_e2e) all_examples_passed: bool,
    pub(in crate::cli::commands::hyper_e2e) elapsed_ms: f64,
    pub(in crate::cli::commands::hyper_e2e) particle_steps: f64,
    pub(in crate::cli::commands::hyper_e2e) particle_steps_per_sec: f64,
    pub(in crate::cli::commands::hyper_e2e) dense_pair_interactions_per_sec: f64,
    pub(in crate::cli::commands::hyper_e2e) adapter_batches: usize,
    pub(in crate::cli::commands::hyper_e2e) mean_total_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) mean_splat_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) mean_color_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) mean_density_loss: f32,
    pub(in crate::cli::commands::hyper_e2e) mean_render_rgb_mse: f32,
    pub(in crate::cli::commands::hyper_e2e) mean_render_rgb_psnr_db: f32,
    pub(in crate::cli::commands::hyper_e2e) min_render_rgb_psnr_db: f32,
    pub(in crate::cli::commands::hyper_e2e) max_render_rgb_psnr_db: f32,
    pub(in crate::cli::commands::hyper_e2e) entries: Vec<BurnE2eRolloutQualityEntry>,
}

pub(in crate::cli::commands::hyper_e2e) struct BurnE2eRolloutOutput {
    pub(in crate::cli::commands::hyper_e2e) backend: String,
    pub(in crate::cli::commands::hyper_e2e) device: String,
    pub(in crate::cli::commands::hyper_e2e) metrics: serde_json::Value,
    pub(in crate::cli::commands::hyper_e2e) history: Vec<BurnE2eRolloutHistoryEntry>,
    pub(in crate::cli::commands::hyper_e2e) final_loss: Option<f32>,
    pub(in crate::cli::commands::hyper_e2e) generator: serde_json::Value,
    pub(in crate::cli::commands::hyper_e2e) quality_validation: Option<BurnE2eRolloutQualityReport>,
}
