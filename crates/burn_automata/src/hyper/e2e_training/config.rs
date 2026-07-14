use std::path::PathBuf;

use super::super::e2e::{E2eHyperGeneratorKind, PerceptionRolloutBackend, Target2dLossBackend};
use crate::{AdamWConfig, ParticleSeed, Target2dLossConfig, TargetImage2d};

pub(crate) const MAX_VALIDATION_HORIZONS: usize = 8;

#[allow(dead_code)]
pub(crate) struct BurnE2eRolloutExample {
    pub(crate) slug: String,
    pub(crate) target: TargetImage2d,
    pub(crate) condition_path: Option<PathBuf>,
    pub(crate) dino_model_path: Option<PathBuf>,
    pub(crate) condition_features: Vec<f32>,
    pub(crate) token_count: usize,
    pub(crate) embed_dims: usize,
    pub(crate) particle_count: usize,
    pub(crate) update_prob: f32,
    pub(crate) seed_scale: f32,
    pub(crate) teacher_adapter: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum E2eLrSchedule {
    #[default]
    Constant,
    Cosine,
    Linear,
    UpstreamGrowing,
}

impl E2eLrSchedule {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Cosine => "cosine",
            Self::Linear => "linear",
            Self::UpstreamGrowing => "upstream-growing",
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum E2eCreditAssignment {
    #[default]
    FullBptt,
    DetachedTbptt,
}

impl E2eCreditAssignment {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::FullBptt => "full-bptt",
            Self::DetachedTbptt => "detached-tbptt",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum E2eAdapterTeacherObjective {
    #[default]
    ParameterMse,
    FunctionalMse,
    Hybrid,
}

impl E2eAdapterTeacherObjective {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ParameterMse => "parameter-mse",
            Self::FunctionalMse => "functional-mse",
            Self::Hybrid => "functional-plus-parameter-mse",
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
    pub(crate) credit_assignment: E2eCreditAssignment,
    pub(crate) max_full_bptt_particle_steps: usize,
    pub(crate) use_particle_pool: bool,
    pub(crate) pool_slots_per_example: usize,
    pub(crate) rollouts_per_example: usize,
    pub(crate) sampling_uniform_fraction: f32,
    pub(crate) sampling_priority_ema_beta: f32,
    pub(crate) sampling_priority_min_weight: f32,
    pub(crate) sampling_priority_max_weight: f32,
    pub(crate) sampling_priority_update_interval: usize,
    pub(crate) pool_capacity: usize,
    pub(crate) inject_seed_interval: usize,
    pub(crate) seed_replacements_per_interval: usize,
    pub(crate) seed_trajectory_interval: usize,
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
    pub(crate) base_per_parameter_grad_normalization: bool,
    pub(crate) generator_per_parameter_grad_normalization: bool,
    pub(crate) adapter_teacher_weight: f32,
    pub(crate) adapter_teacher_objective: E2eAdapterTeacherObjective,
    pub(crate) adapter_teacher_probe_rollout_steps: usize,
    pub(crate) task_loss_weight: f32,
    pub(crate) shared_base_trainable: bool,
    pub(crate) shared_base_train_start_step: usize,
    pub(crate) base_optimizer: AdamWConfig,
    pub(crate) generator_optimizer: AdamWConfig,
    pub(crate) lr_schedule: E2eLrSchedule,
    pub(crate) lr_warmup_steps: usize,
    pub(crate) min_lr_scale: f32,
    pub(crate) adapter_rank: usize,
    pub(crate) adapter_alpha: f32,
    pub(crate) generator_kind: E2eHyperGeneratorKind,
    pub(crate) adapter_chunk_size: usize,
    pub(crate) generator_hidden_dims: usize,
    pub(crate) generator_layers: usize,
    pub(crate) generator_ffn_dims: usize,
    pub(crate) token_attention_heads: usize,
    pub(crate) softmax_token_attention: bool,
    pub(crate) canonical_full_rank_lora: bool,
    pub(crate) generator_sample_steps: usize,
    pub(crate) generator_source_seed: u64,
    pub(crate) flow_matching_weight: f32,
    pub(crate) flow_self_rectification_weight: f32,
    pub(crate) generator_output_scale: f32,
    pub(crate) generator_init_scale: f32,
    pub(crate) generator_condition_init_scale: f32,
    pub(crate) generator_output_init_scale: f32,
    pub(crate) stopgrad_pos: bool,
    pub(crate) stopgrad_state: bool,
    pub(crate) system_memory_budget_gb: Option<f32>,
    pub(crate) gpu_memory_budget_gb: Option<f32>,
    pub(crate) max_dense_train_particles: usize,
    pub(crate) max_dense_chunk_floats: usize,
    pub(crate) max_splat_chunk_floats: usize,
    pub(crate) condition_device_cache_max_bytes: usize,
    pub(crate) target_device_cache_max_bytes: usize,
    pub(crate) dino_image_size: usize,
    pub(crate) dino_batch_size: usize,
    pub(crate) dino_token_grid_width: usize,
    pub(crate) dino_token_grid_height: usize,
    pub(crate) dino_l2_normalize_features: bool,
    pub(crate) dino_rgb_channels: bool,
    pub(crate) dino_rgb_channel_scale: f32,
    pub(crate) dino_alpha_channel: bool,
    pub(crate) dino_alpha_channel_scale: f32,
    pub(crate) spatial_condition_control: bool,
    pub(crate) spatial_condition_control_scale: f32,
    pub(crate) spatial_condition_control_sigma: f32,
    pub(crate) spatial_condition_state_control: bool,
    pub(crate) checkpoint_dir: Option<&'static str>,
    pub(crate) checkpoint_interval_steps: usize,
    pub(crate) checkpoint_interval_seconds: usize,
    pub(crate) resume_checkpoint: Option<&'static str>,
    pub(crate) checkpoint_condition_encoder: Option<&'static str>,
    pub(crate) initial_validation_examples: usize,
    pub(crate) validation_examples: usize,
    pub(crate) validation_interval: usize,
    pub(crate) validation_particles: usize,
    pub(crate) validation_steps: usize,
    pub(crate) validation_horizons: [usize; MAX_VALIDATION_HORIZONS],
    pub(crate) validation_horizon_count: usize,
    pub(crate) validation_selection_horizon_min_steps: usize,
    pub(crate) validation_update_prob: f32,
    pub(crate) validation_seed: u64,
    pub(crate) validation_psnr_threshold_db: f32,
    pub(crate) final_validation_examples: usize,
    pub(crate) final_validation_particles: usize,
    pub(crate) final_validation_steps: usize,
    pub(crate) final_validation_horizons: [usize; MAX_VALIDATION_HORIZONS],
    pub(crate) final_validation_horizon_count: usize,
    pub(crate) final_validation_selection_horizon_min_steps: usize,
    pub(crate) stability_examples: usize,
    pub(crate) stability_particles: usize,
    pub(crate) stability_reference_steps: usize,
    pub(crate) stability_steps: usize,
    pub(crate) stability_tail_steps: usize,
}
