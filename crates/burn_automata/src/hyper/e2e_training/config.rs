use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use super::super::e2e::{E2eHyperGeneratorKind, PerceptionRolloutBackend, Target2dLossBackend};
use crate::{
    AdamWConfig, AutomataResult, ParticleSeed, Target2dLossConfig, TargetImage2d,
    load_target_image_2d_upstream,
};

pub(crate) const MAX_VALIDATION_HORIZONS: usize = 8;

#[derive(Clone)]
pub(crate) enum BurnE2eRolloutTarget {
    #[allow(dead_code)]
    Materialized(TargetImage2d),
    Image {
        path: PathBuf,
        threshold: f32,
        target_points: usize,
        image_size: Option<usize>,
        cache: Arc<OnceLock<TargetImage2d>>,
    },
}

impl BurnE2eRolloutTarget {
    #[allow(dead_code)]
    pub(crate) fn materialized(target: TargetImage2d) -> Self {
        Self::Materialized(target)
    }

    pub(crate) fn image(
        path: PathBuf,
        threshold: f32,
        target_points: usize,
        image_size: Option<usize>,
    ) -> Self {
        Self::Image {
            path,
            threshold,
            target_points,
            image_size,
            cache: Arc::new(OnceLock::new()),
        }
    }

    pub(crate) fn load(&self) -> AutomataResult<TargetImage2d> {
        match self {
            Self::Materialized(target) => Ok(target.clone()),
            Self::Image {
                path,
                threshold,
                target_points,
                image_size,
                cache,
            } => {
                if let Some(target) = cache.get() {
                    return Ok(target.clone());
                }
                let target =
                    load_target_image_2d_upstream(path, *threshold, *target_points, *image_size)?;
                let _ = cache.set(target);
                Ok(cache
                    .get()
                    .expect("target cache is initialized before access")
                    .clone())
            }
        }
    }

    pub(crate) fn point_count_hint(&self) -> usize {
        match self {
            Self::Materialized(target) => target.point_count(),
            Self::Image {
                target_points,
                cache,
                ..
            } => cache
                .get()
                .map_or(*target_points, TargetImage2d::point_count),
        }
    }

    pub(crate) fn is_loaded(&self) -> bool {
        match self {
            Self::Materialized(_) => true,
            Self::Image { cache, .. } => cache.get().is_some(),
        }
    }
}

#[allow(dead_code)]
pub(crate) struct BurnE2eRolloutExample {
    pub(crate) slug: String,
    pub(crate) target: BurnE2eRolloutTarget,
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn lazy_rollout_target_materializes_once_and_shares_the_cache() {
        let path = std::env::temp_dir().join(format!(
            "burn-automata-lazy-target-{}-{}.png",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = std::fs::remove_file(&path);
        RgbaImage::from_pixel(8, 8, Rgba([32, 64, 128, 255]))
            .save(&path)
            .unwrap();

        let target = BurnE2eRolloutTarget::image(path.clone(), 0.05, 64, Some(8));
        let shared = target.clone();
        assert!(!target.is_loaded());
        let first = target.load().unwrap();
        assert!(target.is_loaded());
        assert_eq!(first.point_count(), 64);
        assert_eq!(target.point_count_hint(), 64);

        std::fs::remove_file(&path).unwrap();
        let second = shared.load().unwrap();
        assert_eq!(first.positions, second.positions);
        assert_eq!(first.colors, second.colors);
    }
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

    pub(crate) fn effective_tbptt_chunk_steps(
        self,
        task_loss_weight: f32,
        configured_chunk_steps: usize,
        rollout_steps: usize,
    ) -> Option<usize> {
        (task_loss_weight > 0.0 && self == Self::DetachedTbptt)
            .then(|| configured_chunk_steps.max(1).min(rollout_steps.max(1)))
    }

    pub(crate) fn rollout_gradient_horizon_max_steps(
        self,
        task_loss_weight: f32,
        configured_chunk_steps: usize,
        rollout_step_min: usize,
        rollout_steps: usize,
    ) -> Option<usize> {
        if task_loss_weight <= 0.0 {
            return None;
        }
        let rollout_steps = rollout_steps.max(1);
        let rollout_step_min = rollout_step_min.max(1).min(rollout_steps);
        let sampled_max = if rollout_step_min < rollout_steps {
            rollout_steps - 1
        } else {
            rollout_steps
        };
        Some(match self {
            Self::FullBptt => sampled_max,
            Self::DetachedTbptt => sampled_max.min(configured_chunk_steps.max(1)),
        })
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
    pub(crate) log1p_trajectory_loss: bool,
    pub(crate) trajectory_tail_fraction: f32,
    pub(crate) trajectory_tail_weight: f32,
    pub(crate) trajectory_tail_warmup_steps: usize,
    pub(crate) trajectory_tail_per_identity: bool,
    pub(crate) identity_tail_fraction: f32,
    pub(crate) identity_tail_weight: f32,
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
    pub(crate) sampling_active_window_size: usize,
    pub(crate) sampling_active_window_steps: usize,
    pub(crate) pool_capacity: usize,
    pub(crate) inject_seed_interval: usize,
    pub(crate) seed_replacements_per_interval: usize,
    pub(crate) seed_trajectory_interval: usize,
    pub(crate) brush_size: f32,
    pub(crate) pre_rollout_step_min: usize,
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
    pub(crate) adapter_output_bias: bool,
    pub(crate) generator_sample_steps: usize,
    pub(crate) generator_train_sample_steps: usize,
    pub(crate) generator_source_seed: u64,
    pub(crate) generator_default_endpoint_rms: f32,
    pub(crate) generator_cross_gate_init: f32,
    pub(crate) flow_matching_weight: f32,
    pub(crate) flow_match_inference_source: bool,
    pub(crate) flow_self_rectification_weight: f32,
    pub(crate) amortization_enabled: bool,
    pub(crate) amortization_substrate_steps: usize,
    pub(crate) amortization_substrate_only: bool,
    pub(crate) amortization_residual_scale: f32,
    pub(crate) amortization_residual_anneal_steps: usize,
    pub(crate) amortization_hyper_only_fraction: f32,
    pub(crate) amortization_distillation_weight: f32,
    pub(crate) amortization_distillation_objective: E2eAdapterTeacherObjective,
    pub(crate) amortization_distillation_probe_rollout_steps: usize,
    pub(crate) amortization_initialize_from_teacher: bool,
    pub(crate) amortization_initialize_from_generator: bool,
    pub(crate) amortization_learning_rate: f32,
    pub(crate) amortization_grad_normalization: bool,
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
    pub(crate) dino_patch_pixels: bool,
    pub(crate) spatial_condition_control: bool,
    pub(crate) spatial_condition_control_scale: f32,
    pub(crate) spatial_condition_control_sigma: f32,
    pub(crate) spatial_condition_state_control: bool,
    pub(crate) checkpoint_dir: Option<&'static str>,
    pub(crate) checkpoint_interval_steps: usize,
    pub(crate) checkpoint_interval_seconds: usize,
    pub(crate) resume_checkpoint: Option<&'static str>,
    pub(crate) curriculum_resume: bool,
    pub(crate) curriculum_reset_endpoint_optimizer: bool,
    pub(crate) checkpoint_condition_encoder: Option<&'static str>,
    pub(crate) validation_split: &'static str,
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
    pub(crate) validation_seed_count: usize,
    pub(crate) validation_psnr_threshold_db: f32,
    pub(crate) final_validation_examples: usize,
    pub(crate) final_validation_particles: usize,
    pub(crate) final_validation_steps: usize,
    pub(crate) final_validation_horizons: [usize; MAX_VALIDATION_HORIZONS],
    pub(crate) final_validation_horizon_count: usize,
    pub(crate) final_validation_selection_horizon_min_steps: usize,
    pub(crate) final_validation_seed_count: usize,
    pub(crate) stability_examples: usize,
    pub(crate) stability_particles: usize,
    pub(crate) stability_reference_steps: usize,
    pub(crate) stability_steps: usize,
    pub(crate) stability_tail_steps: usize,
}

impl BurnE2eRolloutTrainConfig {
    pub(crate) fn rollout_free_amortization(self) -> bool {
        self.task_loss_weight == 0.0
            && self.adapter_teacher_weight == 0.0
            && self.flow_matching_weight == 0.0
            && self.amortization_enabled
            && (self.amortization_distillation_weight > 0.0
                || self.flow_self_rectification_weight > 0.0)
    }
}
