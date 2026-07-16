use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use super::super::e2e::{PerceptionRolloutBackend, Target2dLossBackend};
use crate::{
    AdamWConfig, NpaConfig, NpaLowRankAdapter, ParticleSeed, SgdConfig, Target2dLossConfig,
    TargetImage2d,
};

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

#[derive(Clone)]
pub(crate) struct Target2dOracleTrainPlan {
    pub(crate) train: DirectBasisTrainConfig,
    pub(crate) steps_per_repetition: usize,
    pub(crate) repetitions: usize,
    pub(crate) optimizer: AdamWConfig,
    pub(crate) scheduler_milestones: Vec<usize>,
    pub(crate) scheduler_gamma: f32,
}

impl Target2dOracleTrainPlan {
    pub(crate) fn total_steps(&self) -> usize {
        self.steps_per_repetition.saturating_mul(self.repetitions)
    }
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
