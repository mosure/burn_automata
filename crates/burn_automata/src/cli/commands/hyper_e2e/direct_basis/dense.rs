pub(super) struct BurnWgpuDirectBasisOutput {
    pub(super) backend: &'static str,
    pub(super) device: String,
    pub(super) metrics: serde_json::Value,
    pub(super) history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) train_refine_history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) holdout_history: Vec<super::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) best_train_loss: Option<f32>,
    pub(super) best_train_step: usize,
}

pub(super) struct BurnDenseOracleBatchOutput {
    pub(super) backend: &'static str,
    pub(super) device: String,
    pub(super) metrics: serde_json::Value,
    pub(super) history: Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>,
    pub(super) per_model_history: Vec<Vec<crate::cli::reports::CliHyper2dDirectBasisHistoryEntry>>,
    pub(super) best_train_loss: Vec<Option<f32>>,
    pub(super) best_train_step: Vec<usize>,
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
    pub(in crate::cli::commands::hyper_e2e) rollout_particles: usize,
    pub(in crate::cli::commands::hyper_e2e) rollout_steps: usize,
    pub(in crate::cli::commands::hyper_e2e) update_prob: f32,
    pub(in crate::cli::commands::hyper_e2e) seed: u64,
    pub(in crate::cli::commands::hyper_e2e) seed_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) seed_mode: crate::ParticleSeed,
    pub(in crate::cli::commands::hyper_e2e) grid_eps: f32,
    pub(in crate::cli::commands::hyper_e2e) motion_scale: f32,
    pub(in crate::cli::commands::hyper_e2e) loss_config: crate::Target2dLossConfig,
    pub(in crate::cli::commands::hyper_e2e) per_parameter_grad_normalization: bool,
    pub(in crate::cli::commands::hyper_e2e) shared_base_trainable: bool,
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
    pub(in crate::cli::commands::hyper_e2e) validation_examples: usize,
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

macro_rules! dense_direct_basis_backend {
    (
        $module:ident,
        $feature:meta,
        $inner_backend:ty,
        $backend_name:expr,
        $device_label:expr,
        $log_backend:expr
    ) => {
#[cfg($feature)]
#[allow(dead_code)]
mod $module {
    use std::{fs, process::Command, time::Instant};

    use burn::{
        backend::Autodiff,
        tensor::{Device, Int, Tensor, TensorData, activation::relu},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use serde::Serialize;
    use serde_json::json;

    use super::{
        BurnDenseOracleBatchOutput, BurnE2eRolloutExample, BurnE2eRolloutHistoryEntry,
        BurnE2eRolloutOutput, BurnE2eRolloutQualityEntry, BurnE2eRolloutQualityReport,
        BurnE2eRolloutTrainConfig, BurnWgpuDirectBasisOutput, E2eLrSchedule,
    };
    use super::super::{DirectBasisExample, DirectBasisStepStats, DirectBasisTrainConfig};
    use crate::cli::reports::{
        CliHyper2dDirectBasisHistoryEntry, CliHyper2dDirectBasisLossSummary,
    };
    use crate::{
        AdamWConfig, AutomataError, AutomataResult, NpaConfig, NpaLowRankAdapter, NpaModel,
        NpaWeights,
        SgdConfig,
        rollout::{seed_particles_scaled, stochastic_mask},
        target2d::{render_target_2d_splat, target_2d_foreground_mask},
    };

    type InnerBackend = $inner_backend;
    type BurnBackend = Autodiff<InnerBackend>;
    type BurnDevice = Device<BurnBackend>;
    type Tensor1 = Tensor<BurnBackend, 1>;
    type Tensor2 = Tensor<BurnBackend, 2>;
    type Tensor3 = Tensor<BurnBackend, 3>;
    type Tensor4 = Tensor<BurnBackend, 4>;
    type Tensor1Int = Tensor<BurnBackend, 1, Int>;
    type Tensor1Inner = Tensor<InnerBackend, 1>;
    type Tensor2Inner = Tensor<InnerBackend, 2>;
    type Tensor3Inner = Tensor<InnerBackend, 3>;

    const BACKEND: &str = $backend_name;
    const DEVICE_LABEL: &str = $device_label;
    const LOG_BACKEND: &str = $log_backend;
    const EPSILON: f32 = 1.0e-6;

    #[derive(Clone)]
    struct BurnBaseParams {
        w1: Tensor2,
        b1: Tensor2,
        w2: Tensor2,
        b2: Tensor2,
    }

    struct BurnBaseBatch {
        w1: Tensor3,
        b1: Tensor3,
        w2: Tensor3,
        b2: Tensor3,
    }

    #[derive(Clone)]
    struct BurnAdapterParams {
        rank: usize,
        alpha: f32,
        w1_down: Tensor2,
        w1_up: Tensor2,
        w2_down: Tensor2,
        w2_up: Tensor2,
        b1_delta: Tensor2,
        b2_delta: Tensor2,
    }

    struct BurnAdapterBatch {
        rank: usize,
        alpha: f32,
        w1_down: Tensor3,
        w1_up: Tensor3,
        w2_down: Tensor3,
        w2_up: Tensor3,
        b1_delta: Tensor3,
        b2_delta: Tensor3,
    }

    struct BurnBaseAdamWState {
        step: usize,
        w1_m: Tensor2Inner,
        w1_v: Tensor2Inner,
        b1_m: Tensor2Inner,
        b1_v: Tensor2Inner,
        w2_m: Tensor2Inner,
        w2_v: Tensor2Inner,
        b2_m: Tensor2Inner,
        b2_v: Tensor2Inner,
    }

    struct BurnAdapterAdamWState {
        step: usize,
        w1_down_m: Tensor2Inner,
        w1_down_v: Tensor2Inner,
        w1_up_m: Tensor2Inner,
        w1_up_v: Tensor2Inner,
        w2_down_m: Tensor2Inner,
        w2_down_v: Tensor2Inner,
        w2_up_m: Tensor2Inner,
        w2_up_v: Tensor2Inner,
        b1_delta_m: Tensor2Inner,
        b1_delta_v: Tensor2Inner,
        b2_delta_m: Tensor2Inner,
        b2_delta_v: Tensor2Inner,
    }

    #[derive(Clone)]
    struct BurnE2eGeneratorParams {
        token_w: Tensor2,
        token_b: Tensor2,
        token_gate_w: Tensor2,
        token_gate_b: Tensor2,
        state_w: Tensor2,
        time_w: Tensor2,
        output_w: Tensor2,
        output_b: Tensor2,
        hidden_dims: usize,
        token_attention_heads: usize,
        output_dims: usize,
        output_scale: f32,
        sample_steps: usize,
    }

    struct BurnE2eGeneratorAdamWState {
        step: usize,
        token_w_m: Tensor2Inner,
        token_w_v: Tensor2Inner,
        token_b_m: Tensor2Inner,
        token_b_v: Tensor2Inner,
        token_gate_w_m: Tensor2Inner,
        token_gate_w_v: Tensor2Inner,
        token_gate_b_m: Tensor2Inner,
        token_gate_b_v: Tensor2Inner,
        state_w_m: Tensor2Inner,
        state_w_v: Tensor2Inner,
        time_w_m: Tensor2Inner,
        time_w_v: Tensor2Inner,
        output_w_m: Tensor2Inner,
        output_w_v: Tensor2Inner,
        output_b_m: Tensor2Inner,
        output_b_v: Tensor2Inner,
    }

    struct BurnTargetExample {
        target_rgb: Tensor2,
        target_density: Tensor2,
        target_foreground: Tensor2,
        target_foreground_scale: f32,
        target_mean: Tensor2,
        target_positions: Tensor2,
        pixel_xy: Tensor2,
        pixel_size: f32,
        target_points: usize,
        particle_count: usize,
        update_prob: f32,
        seed_scale: f32,
    }

    struct BurnPoolBatch {
        pool_indices: Vec<usize>,
        x: Tensor3,
        s: Tensor3,
    }

    struct BurnHostParticlePool {
        positions: Vec<f32>,
        states: Vec<f32>,
        pool_size: usize,
        particle_count: usize,
        state_dims: usize,
    }

    enum BurnE2eConditionValues {
        Device(Tensor3),
        HostRows(Vec<Vec<f32>>),
    }

    struct BurnE2eConditionCache {
        values: BurnE2eConditionValues,
        examples: usize,
        token_count: usize,
        embed_dims: usize,
        device: BurnDevice,
    }

    struct BurnE2eSelectedCheckpoint {
        step: usize,
        train_loss: f32,
        selection_score: f32,
        holdout_mean_psnr_db: Option<f32>,
        holdout_mean_loss: Option<f32>,
        params: BurnBaseParams,
        generator: BurnE2eGeneratorParams,
    }

    const DEVICE_CONDITION_CACHE_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024;

    struct BurnLossTensors {
        total: Tensor1,
        splat: Tensor1,
        color: Tensor1,
        density: Tensor1,
    }

    #[derive(Clone)]
    struct BurnLossBatchTensors {
        total: Tensor1,
        splat: Tensor1,
        color: Tensor1,
        density: Tensor1,
    }

    #[derive(Clone, Copy, Default)]
    struct BurnLossScalars {
        total: f32,
        splat: f32,
        color: f32,
        density: f32,
    }

    #[derive(Clone, Serialize)]
    struct ProcessMemorySnapshot {
        label: String,
        rss_bytes: Option<u64>,
        budget_bytes: Option<u64>,
    }

    #[derive(Clone, Serialize)]
    struct GpuMemorySnapshot {
        label: String,
        used_bytes: Option<u64>,
        total_bytes: Option<u64>,
        budget_bytes: Option<u64>,
    }

    #[derive(Clone, Serialize)]
    struct SampleUpdateStats {
        examples: usize,
        total_updates: usize,
        min_updates: usize,
        max_updates: usize,
        mean_updates: f32,
        zero_update_examples: usize,
    }

    pub(crate) fn train_direct_basis_burn_dense(
        base: &mut NpaModel,
        train_examples: &mut [DirectBasisExample],
        holdout_examples: &mut [DirectBasisExample],
        train_config: DirectBasisTrainConfig,
        train_refine_config: DirectBasisTrainConfig,
        holdout_config: DirectBasisTrainConfig,
        checkpoint: Option<&super::super::Target2dBurnCheckpointConfig>,
    ) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn dense direct-basis training currently supports 2D",
            )
            .into());
        }
        let mut memory_snapshots = Vec::new();
        let mut gpu_memory_snapshots = Vec::new();
        memory_snapshots.push(check_process_memory_budget("start", train_config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("start", train_config)?);
        let device = BurnDevice::default();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut train_adapters = train_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let train_targets = burn_targets(train_examples, train_config, &device)?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_tensor_cache",
            train_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_train_tensor_cache",
            train_config,
        )?);
        let mut checkpoint_state = checkpoint.map(BurnDenseCheckpointState::new);
        let train_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_config,
            true,
            "train",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_phase",
            train_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("after_train_phase", train_config)?);
        params.write_to_model(base)?;
        let train_refine_phase = run_phase(
            &mut params,
            &mut train_adapters,
            &train_targets,
            train_refine_config,
            false,
            "train-refine",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_train_refine_phase",
            train_refine_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_train_refine_phase",
            train_refine_config,
        )?);
        write_adapters(train_examples, &train_adapters)?;

        let mut holdout_adapters = holdout_examples
            .iter()
            .map(|example| BurnAdapterParams::from_adapter(&example.adapter, base, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let holdout_targets = burn_targets(holdout_examples, holdout_config, &device)?;
        memory_snapshots.push(check_process_memory_budget(
            "after_holdout_tensor_cache",
            holdout_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_holdout_tensor_cache",
            holdout_config,
        )?);
        let holdout_phase = run_phase(
            &mut params,
            &mut holdout_adapters,
            &holdout_targets,
            holdout_config,
            false,
            "holdout",
            checkpoint_state.as_mut(),
        )?;
        memory_snapshots.push(check_process_memory_budget(
            "after_holdout_phase",
            holdout_config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "after_holdout_phase",
            holdout_config,
        )?);
        write_adapters(holdout_examples, &holdout_adapters)?;

        let particle_pool_metrics = json!({
            "enabled": train_config.use_particle_pool,
            "size": train_config.pool_size,
            "inject_seed_interval": train_config.inject_seed_interval,
            "brush_size": train_config.brush_size,
        });
        let checkpoint_selection_metrics = json!({
            "mode": if train_config.use_particle_pool {
                "restore_best_reported_geometry_score_including_phase_initial_state"
            } else {
                "restore_best_reported_eval_loss_including_phase_initial_state"
            },
            "train_best_geometry_score": train_phase.best_geometry_score,
        });
        let mut metrics = json!({
            "backend": format!("{BACKEND}_e2e_rollout"),
            "device": DEVICE_LABEL,
            "objective": "target2d_pixel_splat_loss_full_image",
            "perception": "dense_compact_sph_blur_state_grad_density_grad_hybrid_moment_log_norm",
            "adapter_cache": adapter_cache_metrics(
                base,
                &params,
                &train_adapters,
                &holdout_adapters,
                &train_targets,
                &holdout_targets,
            )?,
            "train_examples": train_examples.len(),
            "holdout_examples": holdout_examples.len(),
            "train_steps": train_config.steps,
            "train_adapter_refine_steps": train_refine_config.steps,
            "holdout_adapter_steps": holdout_config.steps,
            "train_final_dense_loss": train_phase.history.last().map(|entry| entry.loss),
            "train_refine_final_dense_loss": train_refine_phase.history.last().map(|entry| entry.loss),
            "holdout_final_dense_loss": holdout_phase.history.last().map(|entry| entry.loss),
            "checkpoint_selection": checkpoint_selection_metrics,
            "optimizer": "adamw",
            "optimizer_cli_fields": "base/adapter learning_rate, weight_decay, grad_clip_norm",
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.999,
            "adamw_epsilon": 1.0e-8,
            "adapter_gradient_scale": "unaverage_batch_loss_for_per_sample_adapter_adamw",
            "batching": "homogeneous_particle_count_batched_rollout_perception_splat_loss",
            "training_graph": "tbptt_chunked_rollout_state_detach",
            "tbptt_chunk_steps": train_config.tbptt_chunk_steps,
            "loss_on_final_chunk_only": train_config.loss_on_final_chunk_only,
            "particle_pool": particle_pool_metrics,
            "eval_interval": train_config.eval_interval,
            "eval_batch_size": train_config.eval_batch_size,
            "max_dense_train_particles": train_config.max_dense_train_particles,
            "max_dense_chunk_floats": train_config.max_dense_chunk_floats,
            "max_splat_chunk_floats": train_config.max_splat_chunk_floats,
            "system_memory_budget_gb": train_config.system_memory_budget_gb,
            "gpu_memory_budget_gb": train_config.gpu_memory_budget_gb,
            "process_memory_snapshots": memory_snapshots,
            "gpu_memory_snapshots": gpu_memory_snapshots,
            "evaluation": "bounded_tbptt_chunked_loss_vectors_state_detach",
            "train_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_config.steps,
                train_config.example_batch_size,
                train_examples.len(),
            ),
            "train_adapter_update_coverage": train_phase.sample_updates,
            "train_refine_mean_adapter_updates_per_sample": mean_updates_per_sample(
                train_refine_config.steps,
                train_refine_config.example_batch_size,
                train_examples.len(),
            ),
            "train_refine_adapter_update_coverage": train_refine_phase.sample_updates,
            "holdout_mean_adapter_updates_per_sample": mean_updates_per_sample(
                holdout_config.steps,
                holdout_config.example_batch_size,
                holdout_examples.len(),
            ),
            "holdout_adapter_update_coverage": holdout_phase.sample_updates,
        });
        metrics["model_checkpoints"] = checkpoint_state
            .as_ref()
            .map(BurnDenseCheckpointState::report_json)
            .unwrap_or(serde_json::Value::Null);
        let (best_train_loss, best_train_step) =
            best_training_checkpoint(train_config.steps, &train_phase, &train_refine_phase);
        Ok(BurnWgpuDirectBasisOutput {
            backend: BACKEND,
            device: DEVICE_LABEL.to_string(),
            metrics,
            history: train_phase.history,
            train_refine_history: train_refine_phase.history,
            holdout_history: holdout_phase.history,
            best_train_loss,
            best_train_step,
        })
    }

    pub(crate) fn train_e2e_rollout_burn_dense(
        base: &mut NpaModel,
        train_examples: &mut [BurnE2eRolloutExample],
        holdout_examples: &mut [BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
        if base.config.spatial_dims != 2 {
            return Err(std::io::Error::other(
                "Burn dense HyperNPA e2e rollout training currently supports 2D",
            )
            .into());
        }
        if train_examples.is_empty() {
            return Err(std::io::Error::other(
                "Burn dense HyperNPA e2e rollout training requires train examples",
            )
            .into());
        }
        if config.rollout_particles > config.max_dense_train_particles {
            return Err(std::io::Error::other(format!(
                "rollout_particles={} exceeds max_dense_train_particles={}",
                config.rollout_particles, config.max_dense_train_particles
            ))
            .into());
        }
        let started = Instant::now();
        let device = BurnDevice::default();
        let npa_config = base.config.clone();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut base_optimizer = BurnBaseAdamWState::zeros_like(&params);
        let mut generator = BurnE2eGeneratorParams::seeded(base, train_examples, config, &device)?;
        let mut generator_optimizer = BurnE2eGeneratorAdamWState::new(&generator);
        let train_targets = burn_e2e_targets(train_examples, config, &device)?;
        let train_conditions = BurnE2eConditionCache::from_examples_drain(train_examples, &device)?;
        let holdout_conditions =
            BurnE2eConditionCache::from_examples_drain(holdout_examples, &device)?;
        let train_condition_cache_bytes = train_conditions.feature_bytes();
        let holdout_condition_cache_bytes = holdout_conditions.feature_bytes();
        let condition_cache_bytes =
            train_condition_cache_bytes.saturating_add(holdout_condition_cache_bytes);
        check_process_memory_budget("e2e_rollout:start", direct_config_view(config))?;
        check_gpu_memory_budget("e2e_rollout:start", direct_config_view(config))?;
        let initial_quality_validation = evaluate_e2e_rollout_quality(
            &params.detached(),
            &generator.detached(),
            &npa_config,
            train_examples,
            holdout_examples,
            &train_conditions,
            &holdout_conditions,
            config,
            &device,
        )?;
        if let Some(quality) = &initial_quality_validation {
            eprintln!(
                "hyper2d e2e rollout initial {} quality mean_psnr={:.3}dB mean_loss={:.6e}",
                quality.split, quality.mean_render_rgb_psnr_db, quality.mean_total_loss
            );
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let batch_size = normalized_batch_size(config.example_batch_size, train_examples.len());
        let mut history = Vec::new();
        let mut final_loss = None;
        let mut best_checkpoint = None::<BurnE2eSelectedCheckpoint>;
        for step in 1..=config.steps {
            let indices = sample_indices(train_examples.len(), batch_size, &mut rng);
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let collect_metrics =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let lr_scale = e2e_lr_scale(config, step);
            let step_config = e2e_config_with_lr_scale(config, lr_scale);
            let mut stats = train_e2e_homogeneous_step_tbptt(
                &mut params,
                &mut generator,
                &mut base_optimizer,
                &mut generator_optimizer,
                &npa_config,
                &train_conditions,
                &train_targets,
                &indices,
                step_config,
                step_seed,
                collect_metrics,
            )?;
            stats.step = step;
            stats.learning_rate_scale = lr_scale;
            stats.base_learning_rate = step_config.base_optimizer.learning_rate;
            stats.generator_learning_rate = step_config.generator_optimizer.learning_rate;
            if collect_metrics {
                final_loss = Some(stats.loss);
                let checkpoint_quality = evaluate_e2e_rollout_quality(
                    &params.detached(),
                    &generator.detached(),
                    &npa_config,
                    train_examples,
                    holdout_examples,
                    &train_conditions,
                    &holdout_conditions,
                    config,
                    &device,
                )?;
                let (holdout_mean_psnr_db, holdout_mean_loss, selection_score) =
                    if let Some(quality) = &checkpoint_quality {
                        stats.holdout_mean_psnr_db = Some(quality.mean_render_rgb_psnr_db);
                        stats.holdout_mean_loss = Some(quality.mean_total_loss);
                        (
                            Some(quality.mean_render_rgb_psnr_db),
                            Some(quality.mean_total_loss),
                            quality.mean_render_rgb_psnr_db,
                        )
                    } else {
                        (None, None, -stats.loss)
                    };
                eprintln!(
                    "hyper2d e2e rollout step {step}/{} loss={:.6e} lr_scale={:.3e} holdout_psnr={} base_grad={:.6e} generator_grad={:.6e} particle_steps/s={:.3e}",
                    config.steps,
                    stats.loss,
                    stats.learning_rate_scale,
                    format_optional_f32(holdout_mean_psnr_db),
                    stats.base_grad_norm,
                    stats.generator_grad_norm,
                    stats.particle_steps_per_sec,
                );
                if selection_score.is_finite()
                    && best_checkpoint
                        .as_ref()
                        .is_none_or(|checkpoint| selection_score > checkpoint.selection_score)
                {
                    best_checkpoint = Some(BurnE2eSelectedCheckpoint {
                        step,
                        train_loss: stats.loss,
                        selection_score,
                        holdout_mean_psnr_db,
                        holdout_mean_loss,
                        params: params.detached(),
                        generator: generator.detached(),
                    });
                }
                history.push(stats);
            }
        }
        let selected_checkpoint_step = best_checkpoint.as_ref().map(|checkpoint| checkpoint.step);
        let selected_checkpoint_loss =
            best_checkpoint.as_ref().map(|checkpoint| checkpoint.train_loss);
        let selected_checkpoint_score = best_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.selection_score);
        let selected_checkpoint_holdout_psnr_db = best_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.holdout_mean_psnr_db);
        let selected_checkpoint_holdout_loss = best_checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint.holdout_mean_loss);
        let selected_checkpoint_source = if selected_checkpoint_holdout_psnr_db.is_some() {
            "best_reported_holdout_psnr"
        } else if selected_checkpoint_step.is_some() {
            "best_reported_loss"
        } else {
            "final"
        };
        if let Some(best_checkpoint) = best_checkpoint {
            params = best_checkpoint.params;
            generator = best_checkpoint.generator;
        }
        params.write_to_model(base)?;
        let quality_validation = evaluate_e2e_rollout_quality(
            &params.detached(),
            &generator.detached(),
            &npa_config,
            train_examples,
            holdout_examples,
            &train_conditions,
            &holdout_conditions,
            config,
            &device,
        )?;
        let generator_json = generator.to_json()?;
        let (min_reported_particle_steps_per_sec, median_reported_particle_steps_per_sec, max_reported_particle_steps_per_sec) =
            reported_particle_step_speed_summary(&history);
        let (first_reported_loss, best_reported_loss, best_reported_step, final_reported_loss) =
            reported_loss_summary(&history);
        let reported_loss_delta =
            first_reported_loss.zip(final_reported_loss).map(|(first, final_loss)| final_loss - first);
        let best_reported_loss_delta =
            first_reported_loss.zip(best_reported_loss).map(|(first, best)| best - first);
        let train_condition_cache_storage = train_conditions.storage_label();
        let holdout_condition_cache_storage = holdout_conditions.storage_label();
        let condition_features_uploaded_as_resident_device_cache =
            train_conditions.is_device_resident() && holdout_conditions.is_device_resident();
        let mut metrics = serde_json::Map::new();
        metrics.insert("backend".to_string(), json!(format!("{BACKEND}_e2e_rollout")));
        metrics.insert("device".to_string(), json!(DEVICE_LABEL));
        metrics.insert(
            "objective".to_string(),
            json!("target2d_rollout_image_loss_generated_lora"),
        );
        metrics.insert(
            "conditioner".to_string(),
            json!("token_attention_pool_rectified_flow_generated_lora"),
        );
        metrics.insert("adapter_rank".to_string(), json!(config.adapter_rank));
        metrics.insert("adapter_alpha".to_string(), json!(config.adapter_alpha));
        metrics.insert(
            "generator_hidden_dims".to_string(),
            json!(config.generator_hidden_dims),
        );
        metrics.insert(
            "token_attention_heads".to_string(),
            json!(config.token_attention_heads),
        );
        metrics.insert(
            "generator_sample_steps".to_string(),
            json!(config.generator_sample_steps),
        );
        metrics.insert("generator_output_dims".to_string(), json!(generator.output_dims));
        metrics.insert(
            "generator_output_scale".to_string(),
            json!(config.generator_output_scale),
        );
        metrics.insert(
            "condition_token_count".to_string(),
            json!(train_conditions.token_count),
        );
        metrics.insert(
            "condition_embed_dims".to_string(),
            json!(train_conditions.embed_dims),
        );
        metrics.insert(
            "train_condition_cache_bytes_f32".to_string(),
            json!(train_condition_cache_bytes),
        );
        metrics.insert(
            "holdout_condition_cache_bytes_f32".to_string(),
            json!(holdout_condition_cache_bytes),
        );
        metrics.insert(
            "condition_cache_bytes_f32".to_string(),
            json!(condition_cache_bytes),
        );
        metrics.insert(
            "condition_cache_gib_f32".to_string(),
            json!(bytes_to_gib(condition_cache_bytes as u64)),
        );
        metrics.insert(
            "train_condition_cache_storage".to_string(),
            json!(train_condition_cache_storage),
        );
        metrics.insert(
            "holdout_condition_cache_storage".to_string(),
            json!(holdout_condition_cache_storage),
        );
        metrics.insert(
            "cpu_condition_features_drained_from_examples".to_string(),
            json!(true),
        );
        metrics.insert(
            "cpu_condition_features_uploaded_as_resident_device_cache".to_string(),
            json!(condition_features_uploaded_as_resident_device_cache),
        );
        metrics.insert("train_examples".to_string(), json!(train_examples.len()));
        metrics.insert("holdout_examples".to_string(), json!(holdout_examples.len()));
        metrics.insert("steps".to_string(), json!(config.steps));
        metrics.insert(
            "example_batch_size".to_string(),
            json!(config.example_batch_size),
        );
        metrics.insert("rollout_particles".to_string(), json!(config.rollout_particles));
        metrics.insert("rollout_steps".to_string(), json!(config.rollout_steps));
        metrics.insert(
            "tbptt_chunk_steps".to_string(),
            json!(config.tbptt_chunk_steps),
        );
        metrics.insert(
            "loss_on_final_chunk_only".to_string(),
            json!(false),
        );
        metrics.insert(
            "max_dense_train_particles".to_string(),
            json!(config.max_dense_train_particles),
        );
        metrics.insert(
            "training_graph".to_string(),
            json!("generated_adapter_tbptt_chunked_rollout_state_detach"),
        );
        metrics.insert(
            "shared_base_trainable".to_string(),
            json!(config.shared_base_trainable),
        );
        metrics.insert("lr_schedule".to_string(), json!(config.lr_schedule.as_str()));
        metrics.insert("min_lr_scale".to_string(), json!(config.min_lr_scale));
        metrics.insert(
            "selected_checkpoint_source".to_string(),
            json!(selected_checkpoint_source),
        );
        metrics.insert(
            "selected_checkpoint_step".to_string(),
            json!(selected_checkpoint_step),
        );
        metrics.insert(
            "selected_checkpoint_loss".to_string(),
            json!(selected_checkpoint_loss),
        );
        metrics.insert(
            "selected_checkpoint_score".to_string(),
            json!(selected_checkpoint_score),
        );
        metrics.insert(
            "selected_checkpoint_holdout_psnr_db".to_string(),
            json!(selected_checkpoint_holdout_psnr_db),
        );
        metrics.insert(
            "selected_checkpoint_holdout_loss".to_string(),
            json!(selected_checkpoint_holdout_loss),
        );
        metrics.insert(
            "min_reported_particle_steps_per_sec".to_string(),
            json!(min_reported_particle_steps_per_sec),
        );
        metrics.insert(
            "median_reported_particle_steps_per_sec".to_string(),
            json!(median_reported_particle_steps_per_sec),
        );
        metrics.insert(
            "max_reported_particle_steps_per_sec".to_string(),
            json!(max_reported_particle_steps_per_sec),
        );
        metrics.insert("first_reported_loss".to_string(), json!(first_reported_loss));
        metrics.insert("best_reported_loss".to_string(), json!(best_reported_loss));
        metrics.insert("best_reported_step".to_string(), json!(best_reported_step));
        metrics.insert("final_reported_loss".to_string(), json!(final_reported_loss));
        metrics.insert("reported_loss_delta".to_string(), json!(reported_loss_delta));
        metrics.insert(
            "best_reported_loss_delta".to_string(),
            json!(best_reported_loss_delta),
        );
        metrics.insert(
            "initial_quality_validation".to_string(),
            json!(initial_quality_validation.clone()),
        );
        metrics.insert(
            "quality_validation".to_string(),
            json!(quality_validation.clone()),
        );
        metrics.insert(
            "elapsed_ms".to_string(),
            json!(started.elapsed().as_secs_f64() * 1000.0),
        );
        let metrics = serde_json::Value::Object(metrics);
        Ok(BurnE2eRolloutOutput {
            backend: format!("{BACKEND}_e2e_rollout"),
            device: DEVICE_LABEL.to_string(),
            metrics,
            history,
            final_loss,
            generator: generator_json,
            quality_validation,
        })
    }

    pub(crate) fn train_oracle_models_burn_dense(
        models: &mut [NpaModel],
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
    ) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
        if models.is_empty() || examples.is_empty() {
            return Err(std::io::Error::other(
                "Burn dense oracle model batch requires at least one model/example",
            )
            .into());
        }
        if models.len() != examples.len() {
            return Err(std::io::Error::other(format!(
                "Burn dense oracle model batch length mismatch: models={} examples={}",
                models.len(),
                examples.len()
            ))
            .into());
        }
        if models
            .iter()
            .any(|model| model.config.spatial_dims != 2 || model.config != models[0].config)
        {
            return Err(std::io::Error::other(
                "Burn dense oracle model batch requires matching 2D NPA model configs",
            )
            .into());
        }

        let mut memory_snapshots = Vec::new();
        let mut gpu_memory_snapshots = Vec::new();
        memory_snapshots.push(check_process_memory_budget("oracle_batch:start", config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("oracle_batch:start", config)?);
        let device = BurnDevice::default();
        let mut params = models
            .iter()
            .map(|model| BurnBaseParams::from_model(model, &device))
            .collect::<AutomataResult<Vec<_>>>()?;
        let targets = burn_targets(examples, config, &device)?;
        let indices = (0..targets.len()).collect::<Vec<_>>();
        let Some(particle_count) = homogeneous_particle_count(&targets, &indices) else {
            return Err(std::io::Error::other(
                "Burn dense vectorized oracle batch requires homogeneous particle counts",
            )
            .into());
        };
        memory_snapshots.push(check_process_memory_budget(
            "oracle_batch:after_target_cache",
            config,
        )?);
        gpu_memory_snapshots.push(check_gpu_memory_budget(
            "oracle_batch:after_target_cache",
            config,
        )?);

        let mut optimizers = params
            .iter()
            .map(BurnBaseAdamWState::zeros_like)
            .collect::<Vec<_>>();
        let mut history = Vec::new();
        let mut per_model_history = vec![Vec::new(); models.len()];
        let mut best_train_loss = vec![None::<f32>; models.len()];
        let mut best_train_step = vec![0usize; models.len()];

        for step in 1..=config.steps {
            let should_report =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9));
            let stats = train_oracle_model_batch_step_tbptt(
                &mut params,
                &mut optimizers,
                &targets,
                particle_count,
                config,
                step_seed,
                should_report,
            )?;
            if should_report {
                let mean_loss = stats
                    .per_model_loss
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_loss.len().max(1) as f32;
                let mean_base_grad_norm = stats
                    .per_model_base_grad_norm
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_base_grad_norm.len().max(1) as f32;
                let mean_base_grad_scale = stats
                    .per_model_base_grad_scale
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / stats.per_model_base_grad_scale.len().max(1) as f32;
                println!(
                    "{LOG_BACKEND} oracle-model-batch train step {step}/{} loss={mean_loss:.6} models={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                    config.steps,
                    models.len(),
                    stats.particle_steps_per_sec,
                    stats.elapsed_ms
                );
                history.push(CliHyper2dDirectBasisHistoryEntry {
                    step,
                    loss: mean_loss,
                    eval_loss: None,
                    base_grad_norm: mean_base_grad_norm,
                    base_grad_scale: mean_base_grad_scale,
                    mean_adapter_grad_norm: 0.0,
                    max_adapter_grad_norm: 0.0,
                    examples_seen: models.len(),
                    particle_steps_per_sec: stats.particle_steps_per_sec,
                    elapsed_ms: stats.elapsed_ms,
                });
                for (idx, loss) in stats.per_model_loss.iter().copied().enumerate() {
                    if best_train_loss[idx].is_none_or(|best| loss < best) {
                        best_train_loss[idx] = Some(loss);
                        best_train_step[idx] = step;
                    }
                    per_model_history[idx].push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss,
                        eval_loss: None,
                        base_grad_norm: stats.per_model_base_grad_norm[idx],
                        base_grad_scale: stats.per_model_base_grad_scale[idx],
                        mean_adapter_grad_norm: 0.0,
                        max_adapter_grad_norm: 0.0,
                        examples_seen: 1,
                        particle_steps_per_sec: stats.particle_steps_per_sec
                            / models.len().max(1) as f64,
                        elapsed_ms: stats.elapsed_ms,
                    });
                }
                let _ = check_process_memory_budget(
                    &format!("oracle_batch:report_step:{step}"),
                    config,
                )?;
                let _ = check_gpu_memory_budget(
                    &format!("oracle_batch:report_step:{step}"),
                    config,
                )?;
            }
        }

        for (model, params) in models.iter_mut().zip(&params) {
            params.write_to_model(model)?;
        }
        memory_snapshots.push(check_process_memory_budget("oracle_batch:end", config)?);
        gpu_memory_snapshots.push(check_gpu_memory_budget("oracle_batch:end", config)?);
        let metrics = json!({
            "backend": BACKEND,
            "device": DEVICE_LABEL,
            "objective": "target2d_pixel_splat_loss_full_image",
            "mode": "vectorized_independent_oracle_models",
            "model_batch_size": models.len(),
            "optimizer_state": "separate_adamw_state_per_oracle_model",
            "parameter_sharing": false,
            "rollout_batch_size_per_model": 1,
            "particle_count": particle_count,
            "steps": config.steps,
            "rollout_steps": config.rollout_steps,
            "tbptt_chunk_steps": config.tbptt_chunk_steps,
            "loss_on_final_chunk_only": config.loss_on_final_chunk_only,
            "max_dense_chunk_floats": config.max_dense_chunk_floats,
            "max_splat_chunk_floats": config.max_splat_chunk_floats,
            "system_memory_budget_gb": config.system_memory_budget_gb,
            "gpu_memory_budget_gb": config.gpu_memory_budget_gb,
            "process_memory_snapshots": memory_snapshots,
            "gpu_memory_snapshots": gpu_memory_snapshots,
        });
        Ok(BurnDenseOracleBatchOutput {
            backend: BACKEND,
            device: DEVICE_LABEL.to_string(),
            metrics,
            history,
            per_model_history,
            best_train_loss,
            best_train_step,
        })
    }

    struct BurnPhaseReport {
        history: Vec<CliHyper2dDirectBasisHistoryEntry>,
        best_loss: Option<f32>,
        best_step: usize,
        best_geometry_score: Option<f32>,
        sample_updates: SampleUpdateStats,
    }

    #[derive(Clone, Serialize)]
    struct BurnDenseCheckpointEvent {
        kind: &'static str,
        phase: String,
        step: usize,
        elapsed_seconds: f64,
        train_loss: Option<f32>,
        eval_loss: Option<f32>,
        geometry_score: Option<f32>,
        model_output: String,
        sha256: Option<String>,
    }

    struct BurnDenseCheckpointWrite {
        kind: &'static str,
        output: std::path::PathBuf,
        phase: String,
        step: usize,
        train_loss: Option<f32>,
        eval_loss: Option<f32>,
        geometry_score: Option<f32>,
    }

    struct BurnDenseCheckpointState<'a> {
        config: &'a super::super::Target2dBurnCheckpointConfig,
        started: Instant,
        last_current_write: Instant,
        current_writes: usize,
        best_writes: usize,
        events: Vec<BurnDenseCheckpointEvent>,
    }

    impl<'a> BurnDenseCheckpointState<'a> {
        fn new(config: &'a super::super::Target2dBurnCheckpointConfig) -> Self {
            let now = Instant::now();
            Self {
                config,
                started: now,
                last_current_write: now,
                current_writes: 0,
                best_writes: 0,
                events: Vec::new(),
            }
        }

        fn should_write_current(&self, step: usize) -> bool {
            let step_due =
                self.config.interval_steps > 0 && step.is_multiple_of(self.config.interval_steps);
            let time_due = self
                .config
                .interval_duration
                .is_some_and(|interval| self.last_current_write.elapsed() >= interval);
            step_due || time_due
        }

        fn write_current(
            &mut self,
            params: &BurnBaseParams,
            phase: &str,
            step: usize,
            train_loss: Option<f32>,
            eval_loss: Option<f32>,
            geometry_score: Option<f32>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.write_model(params, BurnDenseCheckpointWrite {
                kind: "current",
                output: self.config.current_model_output.clone(),
                phase: phase.to_string(),
                step,
                train_loss,
                eval_loss,
                geometry_score,
            })?;
            self.current_writes = self.current_writes.saturating_add(1);
            self.last_current_write = Instant::now();
            self.write_metadata()?;
            Ok(())
        }

        fn write_best(
            &mut self,
            params: &BurnBaseParams,
            phase: &str,
            step: usize,
            train_loss: Option<f32>,
            eval_loss: Option<f32>,
            geometry_score: Option<f32>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.write_model(params, BurnDenseCheckpointWrite {
                kind: "best",
                output: self.config.best_model_output.clone(),
                phase: phase.to_string(),
                step,
                train_loss,
                eval_loss,
                geometry_score,
            })?;
            self.best_writes = self.best_writes.saturating_add(1);
            self.write_metadata()?;
            Ok(())
        }

        fn write_model(
            &mut self,
            params: &BurnBaseParams,
            request: BurnDenseCheckpointWrite,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut model = NpaModel {
                config: self.config.model_config.clone(),
                weights: NpaWeights::zeros(&self.config.model_config),
            };
            params.write_to_model(&mut model)?;
            let source = Some(format!(
                "{}:checkpoint:{}:phase={}:step={}",
                self.config.source, request.kind, request.phase, request.step
            ));
            let manifest =
                crate::import::BpkModelManifest::from_model(&model, self.config.hashgrid.clone(), source);
            let sha256 = atomic_save_manifest(&request.output, &manifest)?;
            let event = BurnDenseCheckpointEvent {
                kind: request.kind,
                phase: request.phase.clone(),
                step: request.step,
                elapsed_seconds: self.started.elapsed().as_secs_f64(),
                train_loss: request.train_loss,
                eval_loss: request.eval_loss,
                geometry_score: request.geometry_score,
                model_output: request.output.display().to_string(),
                sha256,
            };
            self.events.push(event.clone());
            println!(
                "{LOG_BACKEND} direct-basis checkpoint {} phase={} step={} model={}",
                request.kind,
                request.phase,
                request.step,
                request.output.display()
            );
            Ok(())
        }

        fn write_metadata(&self) -> Result<(), Box<dyn std::error::Error>> {
            let report = self.report_json();
            atomic_write_json(&self.config.metadata_output, &report)
        }

        fn report_json(&self) -> serde_json::Value {
            json!({
                "current_model_output": self.config.current_model_output.display().to_string(),
                "best_model_output": self.config.best_model_output.display().to_string(),
                "metadata_output": self.config.metadata_output.display().to_string(),
                "interval_steps": self.config.interval_steps,
                "interval_seconds": self.config.interval_duration.map(|duration| duration.as_secs()),
                "current_writes": self.current_writes,
                "best_writes": self.best_writes,
                "elapsed_seconds": self.started.elapsed().as_secs_f64(),
                "events": &self.events,
            })
        }
    }

    fn atomic_save_manifest(
        path: &std::path::Path,
        manifest: &crate::import::BpkModelManifest,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let tmp_path = atomic_temp_path(path);
        let sha256 = crate::import::save_manifest(&tmp_path, manifest)?;
        fs::rename(&tmp_path, path)?;
        Ok(sha256)
    }

    fn atomic_write_json(
        path: &std::path::Path,
        value: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = atomic_temp_path(path);
        fs::write(&tmp_path, serde_json::to_string_pretty(value)?)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    fn atomic_temp_path(path: &std::path::Path) -> std::path::PathBuf {
        let extension = path.extension().and_then(|value| value.to_str()).unwrap_or("json");
        let file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("checkpoint");
        path.with_file_name(format!(".{file_name}.tmp.{extension}"))
    }

    #[derive(Clone, Copy, Debug, Serialize)]
    struct BurnGeometrySummary {
        examples: usize,
        mean_score: f32,
        mean_foreground_iou: f32,
        mean_target_recall: f32,
        mean_generated_precision: f32,
        mean_bbox_iou: f32,
        mean_lit_pixel_ratio: f32,
        mean_bbox_width_ratio: f32,
        mean_bbox_area_ratio: f32,
    }

    fn run_phase(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        update_base: bool,
        phase_label: &str,
        mut checkpoint_state: Option<&mut BurnDenseCheckpointState<'_>>,
    ) -> Result<BurnPhaseReport, Box<dyn std::error::Error>> {
        if targets.is_empty() || config.steps == 0 {
            return Ok(BurnPhaseReport {
                history: Vec::new(),
                best_loss: None,
                best_step: 0,
                best_geometry_score: None,
                sample_updates: sample_update_stats(&vec![0; targets.len()]),
            });
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut sampler =
            PhaseBatchSampler::new(targets.len(), config.example_batch_size, &mut rng);
        let homogeneous_pool_particle_count = if config.use_particle_pool {
            let particle_count = targets[0].particle_count;
            if targets
                .iter()
                .any(|target| target.particle_count != particle_count)
            {
                return Err(std::io::Error::other(
                    "Burn target2d particle-pool training requires homogeneous particle counts",
                )
                .into());
            }
            Some(particle_count)
        } else {
            None
        };
        let mut particle_pool = homogeneous_pool_particle_count.map(|particle_count| {
            BurnHostParticlePool::new(
                config.pool_size.max(config.example_batch_size).max(1),
                particle_count,
                16,
                targets[0].seed_scale,
                config,
            )
        });
        let mut sample_update_counts = vec![0usize; targets.len()];
        let mut history = Vec::new();
        let mut best_loss = None;
        let mut best_step = 0;
        let mut best_geometry_score = None;
        let mut best_params = None::<BurnBaseParams>;
        let mut best_adapters = None::<Vec<BurnAdapterParams>>;
        if config.eval_interval > 0
            && let Some(eval_loss) = evaluate_targets(
                params,
                adapters,
                targets,
                config,
                config.eval_examples,
                config.eval_seed,
            )?
        {
            best_loss = Some(eval_loss.mean_total_loss);
            best_geometry_score = if config.use_particle_pool {
                evaluate_target_geometry(
                    params,
                    adapters,
                    targets,
                    config,
                    config.eval_examples,
                    config.eval_seed,
                )?
                .map(|geometry| geometry.mean_score)
            } else {
                None
            };
            best_params = update_base.then(|| params.clone());
            best_adapters = Some(adapters.to_vec());
            if let Some(checkpoint_state) = checkpoint_state.as_deref_mut() {
                checkpoint_state.write_best(
                    params,
                    phase_label,
                    0,
                    None,
                    Some(eval_loss.mean_total_loss),
                    best_geometry_score,
                )?;
                checkpoint_state.write_current(
                    params,
                    phase_label,
                    0,
                    None,
                    Some(eval_loss.mean_total_loss),
                    best_geometry_score,
                )?;
            }
        }
        let mut base_optimizer = BurnBaseAdamWState::zeros_like(params);
        let mut adapter_optimizers = adapters
            .iter()
            .map(BurnAdapterAdamWState::zeros_like)
            .collect::<Vec<_>>();
        for step in 1usize..=config.steps {
            let should_report =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let should_eval = config.eval_interval > 0
                && (step == config.steps || step.is_multiple_of(config.eval_interval.max(1)));
            let started = Instant::now();
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9));
            let stats = if let (Some(pool), Some(particle_count)) =
                (particle_pool.as_mut(), homogeneous_pool_particle_count)
            {
                let replace_seed = step.is_multiple_of(config.inject_seed_interval.max(1));
                let device = &targets[0].target_rgb.device();
                let pool_batch = pool.sample_batch(
                    &mut rng,
                    config.example_batch_size.max(1),
                    replace_seed,
                    targets[0].seed_scale,
                    config,
                    device,
                );
                let indices = (0..pool_batch.pool_indices.len())
                    .map(|local| local % targets.len())
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return Err(std::io::Error::other("Burn direct-basis pool batch was empty")
                        .into());
                }
                for &idx in &indices {
                    sample_update_counts[idx] = sample_update_counts[idx].saturating_add(1);
                }
                let stats = train_homogeneous_step_tbptt(
                    params,
                    adapters,
                    &mut base_optimizer,
                    &mut adapter_optimizers,
                    targets,
                    &indices,
                    particle_count,
                    config,
                    step_seed,
                    update_base,
                    should_report,
                    Some((pool_batch.x, pool_batch.s)),
                    Some((pool, pool_batch.pool_indices)),
                )?;
                stats
            } else {
                let indices = sampler.next_batch(&mut rng);
                if indices.is_empty() {
                    return Err(
                        std::io::Error::other("Burn direct-basis batch was empty").into(),
                    );
                };
                for &idx in &indices {
                    sample_update_counts[idx] = sample_update_counts[idx].saturating_add(1);
                }
                let stats = train_step_tbptt(
                    params,
                    adapters,
                    &mut base_optimizer,
                    &mut adapter_optimizers,
                    targets,
                    &indices,
                    config,
                    step_seed,
                    update_base,
                    should_report,
                )?;
                stats
            };
            let elapsed = started.elapsed();
            if should_report {
                let eval_loss = if should_eval {
                    evaluate_targets(
                        params,
                        adapters,
                        targets,
                        config,
                        config.eval_examples,
                        config.eval_seed + step as u64,
                    )?
                } else {
                    None
                };
                if let Some(eval_loss) = eval_loss {
                    let geometry = if config.use_particle_pool {
                        evaluate_target_geometry(
                            params,
                            adapters,
                            targets,
                            config,
                            config.eval_examples,
                            config.eval_seed + step as u64,
                        )?
                    } else {
                        None
                    };
                    let is_better = if let Some(geometry) = geometry {
                        best_geometry_score
                            .is_none_or(|best| geometry.mean_score > best)
                    } else {
                        best_loss.is_none_or(|best| eval_loss.mean_total_loss < best)
                    };
                    if is_better {
                        best_loss = Some(eval_loss.mean_total_loss);
                        best_step = step;
                        if let Some(geometry) = geometry {
                            best_geometry_score = Some(geometry.mean_score);
                        }
                        best_params = update_base.then(|| params.clone());
                        best_adapters = Some(adapters.to_vec());
                        if let Some(checkpoint_state) = checkpoint_state.as_deref_mut() {
                            checkpoint_state.write_best(
                                params,
                                phase_label,
                                step,
                                Some(stats.loss),
                                Some(eval_loss.mean_total_loss),
                                best_geometry_score,
                            )?;
                        }
                    }
                    println!(
                        "{LOG_BACKEND} direct-basis {phase_label} step {step}/{} loss={:.6} eval_mean={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                        config.steps,
                        stats.loss,
                        eval_loss.mean_total_loss,
                        stats.examples_seen,
                        stats.particle_steps_per_sec,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    history.push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss: stats.loss,
                        eval_loss: Some(eval_loss),
                        base_grad_norm: stats.base_grad_norm,
                        base_grad_scale: stats.base_grad_scale,
                        mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                        max_adapter_grad_norm: stats.max_adapter_grad_norm,
                        examples_seen: stats.examples_seen,
                        particle_steps_per_sec: stats.particle_steps_per_sec,
                        elapsed_ms: stats.elapsed_ms,
                    });
                } else {
                    println!(
                        "{LOG_BACKEND} direct-basis {phase_label} step {step}/{} loss={:.6} examples={} particle_steps_per_sec={:.0} elapsed_ms={:.1}",
                        config.steps,
                        stats.loss,
                        stats.examples_seen,
                        stats.particle_steps_per_sec,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    history.push(CliHyper2dDirectBasisHistoryEntry {
                        step,
                        loss: stats.loss,
                        eval_loss: None,
                        base_grad_norm: stats.base_grad_norm,
                        base_grad_scale: stats.base_grad_scale,
                        mean_adapter_grad_norm: stats.mean_adapter_grad_norm,
                        max_adapter_grad_norm: stats.max_adapter_grad_norm,
                        examples_seen: stats.examples_seen,
                        particle_steps_per_sec: stats.particle_steps_per_sec,
                        elapsed_ms: stats.elapsed_ms,
                    });
                }
                let _ = check_process_memory_budget(
                    &format!("{phase_label}:report_step:{step}"),
                    config,
                )?;
                let _ =
                    check_gpu_memory_budget(&format!("{phase_label}:report_step:{step}"), config)?;
            }
            if let Some(checkpoint_state) = checkpoint_state.as_deref_mut()
                && step != config.steps
                && checkpoint_state.should_write_current(step)
            {
                checkpoint_state.write_current(
                    params,
                    phase_label,
                    step,
                    Some(stats.loss),
                    None,
                    best_geometry_score,
                )?;
            }
        }
        if let Some(saved) = best_params {
            *params = saved;
        }
        if let Some(saved) = best_adapters {
            adapters.clone_from_slice(&saved);
        }
        Ok(BurnPhaseReport {
            history,
            best_loss,
            best_step,
            best_geometry_score,
            sample_updates: sample_update_stats(&sample_update_counts),
        })
    }

    fn best_training_checkpoint(
        train_steps: usize,
        train_phase: &BurnPhaseReport,
        train_refine_phase: &BurnPhaseReport,
    ) -> (Option<f32>, usize) {
        let train_best = train_phase
            .best_loss
            .map(|loss| (loss, train_phase.best_step));
        let refine_best = train_refine_phase
            .best_loss
            .map(|loss| (loss, train_steps + train_refine_phase.best_step));
        match (train_best, refine_best) {
            (Some(train), Some(refine)) => {
                if refine.0 < train.0 {
                    (Some(refine.0), refine.1)
                } else {
                    (Some(train.0), train.1)
                }
            }
            (Some(train), None) => (Some(train.0), train.1),
            (None, Some(refine)) => (Some(refine.0), refine.1),
            (None, None) => (None, 0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn train_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        if indices.is_empty() {
            return Err(std::io::Error::other("Burn direct-basis batch was empty").into());
        }
        if let Some(particle_count) = homogeneous_particle_count(targets, indices) {
            return train_homogeneous_step_tbptt(
                params,
                adapters,
                base_optimizer,
                adapter_optimizers,
                targets,
                indices,
                particle_count,
                config,
                step_seed,
                update_base,
                collect_metrics,
                None,
                None,
            );
        }
        train_mixed_step_tbptt(
            params,
            adapters,
            base_optimizer,
            adapter_optimizers,
            targets,
            indices,
            config,
            step_seed,
            update_base,
            collect_metrics,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn train_homogeneous_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
        pool_update: Option<(&mut BurnHostParticlePool, Vec<usize>)>,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let device = &targets[indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(targets, indices, particle_count, config, step_seed, device)
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let mut remaining_steps = rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let final_chunk = remaining_steps <= chunk_steps;
            let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
            let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
            let (next_x, next_s, displacement) = rollout_batch_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rng,
                steps,
                displacement,
            );
            if config.loss_on_final_chunk_only && !final_chunk {
                x = detach3(next_x);
                s = detach3(next_s);
                particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
            let loss = target_splat_loss_batch(
                &next_x,
                &next_s,
                targets,
                indices,
                config,
                &adapter_batch,
                displacement,
            );
            if let Some(loss_sum) = loss_sum.as_mut() {
                *loss_sum += loss_scalars(&loss)?.total * indices.len() as f32;
            }
            let grad_stats = apply_chunk_gradients(
                params,
                adapters,
                base_optimizer,
                adapter_optimizers,
                indices,
                loss.total,
                config,
                update_base,
                indices.len() as f32,
                collect_metrics,
            )?;
            if collect_metrics {
                base_grad_norm_sum += grad_stats.base_grad_norm;
                base_grad_scale_sum += grad_stats.base_grad_scale;
                adapter_grad_sum += grad_stats.adapter_grad_sum;
                adapter_grad_max = adapter_grad_max.max(grad_stats.adapter_grad_max);
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        if let Some((pool, pool_indices)) = pool_update {
            pool.update_batch(&pool_indices, x, s)?;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_chunk_count = if config.loss_on_final_chunk_only {
            1
        } else {
            chunk_count
        };
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / loss_chunk_count as f32
            }),
            base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
            base_grad_scale: if collect_metrics {
                base_grad_scale_sum / grad_metric_chunks as f32
            } else {
                1.0
            },
            mean_adapter_grad_norm: if collect_metrics {
                adapter_grad_sum / (indices.len() * grad_metric_chunks).max(1) as f32
            } else {
                0.0
            },
            max_adapter_grad_norm: adapter_grad_max,
            examples_seen: indices.len(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_mixed_step_tbptt(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        step_seed: u64,
        update_base: bool,
        collect_metrics: bool,
    ) -> Result<DirectBasisStepStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        for &idx in indices {
            let target = &targets[idx];
            let device = &target.target_rgb.device();
            let (mut x, mut s) = seed_tensors(
                target.particle_count,
                config,
                target.seed_scale,
                step_seed.wrapping_add(idx as u64),
                device,
            );
            let mut rng = StdRng::seed_from_u64(step_seed.wrapping_add(idx as u64) ^ 0x005e_ed2d);
            let mut remaining_steps = rollout_steps;
            while remaining_steps > 0 {
                let steps = remaining_steps.min(chunk_steps);
                let final_chunk = remaining_steps <= chunk_steps;
                let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                let (next_x, next_s, displacement) = rollout_single_chunk(
                    params,
                    &adapters[idx],
                    target,
                    x,
                    s,
                    config,
                    &mut rng,
                    steps,
                    displacement,
                );
                if config.loss_on_final_chunk_only && !final_chunk {
                    x = detach2(next_x);
                    s = detach2(next_s);
                    particle_steps += target.particle_count as f64 * steps as f64;
                    remaining_steps -= steps;
                    continue;
                }
                let loss = target_splat_loss(
                    &next_x,
                    &next_s,
                    target,
                    config,
                    &adapters[idx],
                    displacement,
                );
                if let Some(loss_sum) = loss_sum.as_mut() {
                    *loss_sum += loss_scalars(&loss)?.total;
                }
                let scaled_total = loss.total.div_scalar(indices.len() as f32);
                let single_index = [idx];
                let grad_stats = apply_chunk_gradients(
                    params,
                    adapters,
                    base_optimizer,
                    adapter_optimizers,
                    &single_index,
                    scaled_total,
                    config,
                    update_base,
                    indices.len() as f32,
                    collect_metrics,
                )?;
                if collect_metrics {
                    base_grad_norm_sum += grad_stats.base_grad_norm;
                    base_grad_scale_sum += grad_stats.base_grad_scale;
                    adapter_grad_sum += grad_stats.adapter_grad_sum;
                    adapter_grad_max = adapter_grad_max.max(grad_stats.adapter_grad_max);
                    grad_metric_chunks += 1;
                }
                x = detach2(next_x);
                s = detach2(next_s);
                particle_steps += target.particle_count as f64 * steps as f64;
                remaining_steps -= steps;
            }
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_chunk_count = if config.loss_on_final_chunk_only {
            1
        } else {
            chunk_count
        };
        Ok(DirectBasisStepStats {
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / loss_chunk_count as f32
            }),
            base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
            base_grad_scale: if collect_metrics {
                base_grad_scale_sum / grad_metric_chunks as f32
            } else {
                1.0
            },
            mean_adapter_grad_norm: if collect_metrics {
                adapter_grad_sum / grad_metric_chunks as f32
            } else {
                0.0
            },
            max_adapter_grad_norm: adapter_grad_max,
            examples_seen: indices.len(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_e2e_homogeneous_step_tbptt(
        params: &mut BurnBaseParams,
        generator: &mut BurnE2eGeneratorParams,
        base_optimizer: &mut BurnBaseAdamWState,
        generator_optimizer: &mut BurnE2eGeneratorAdamWState,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
    ) -> Result<BurnE2eRolloutHistoryEntry, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Err(std::io::Error::other(
                "Burn HyperNPA e2e rollout batches require homogeneous particle counts",
            )
            .into());
        };
        let started = Instant::now();
        let direct_config = direct_config_view(config);
        let device = &targets[indices[0]].target_rgb.device();
        let (mut x, mut s) = seed_batch_tensors(
            targets,
            indices,
            particle_count,
            direct_config,
            step_seed,
            device,
        );
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let chunk_steps = tbptt_chunk_steps(direct_config);
        let chunk_count = direct_config.rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut generator_grad_norm_sum = 0.0_f32;
        let mut generator_grad_scale_sum = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let condition = conditions.select(indices)?;
        let mut remaining_steps = direct_config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let adapter_batch = generator.adapter_batch(condition.clone(), npa_config, config);
            let displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
            let (next_x, next_s, displacement) = rollout_batch_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                steps,
                displacement,
            );
            let loss = target_splat_loss_batch_vector(
                &next_x,
                &next_s,
                targets,
                indices,
                direct_config,
                &adapter_batch,
                displacement,
            );
            if let Some(loss_sum) = loss_sum.as_mut() {
                for scalars in loss_vector_scalars(loss.clone())? {
                    *loss_sum += scalars.total;
                }
            }
            let mut grads = loss.total.sum().div_scalar(indices.len() as f32).backward();
            let (base_grad_norm, base_grad_scale) = if config.shared_base_trainable {
                params.apply_adamw(
                    &mut grads,
                    base_optimizer,
                    config.base_optimizer,
                    config.per_parameter_grad_normalization,
                    collect_metrics,
                )?
            } else {
                (0.0, 1.0)
            };
            let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
                &mut grads,
                generator_optimizer,
                config.generator_optimizer,
                config.per_parameter_grad_normalization,
                collect_metrics,
            )?;
            if collect_metrics {
                base_grad_norm_sum += base_grad_norm;
                base_grad_scale_sum += base_grad_scale;
                generator_grad_norm_sum += generator_grad_norm;
                generator_grad_scale_sum += generator_grad_scale;
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let particle_steps_per_sec =
            particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok(BurnE2eRolloutHistoryEntry {
            step: 0,
            loss: loss_sum.map_or(0.0, |value| {
                value / indices.len() as f32 / chunk_count as f32
            }),
            learning_rate_scale: 1.0,
            base_learning_rate: config.base_optimizer.learning_rate,
            generator_learning_rate: config.generator_optimizer.learning_rate,
            holdout_mean_psnr_db: None,
            holdout_mean_loss: None,
            base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
            base_grad_scale: base_grad_scale_sum / grad_metric_chunks as f32,
            generator_grad_norm: generator_grad_norm_sum / grad_metric_chunks as f32,
            generator_grad_scale: generator_grad_scale_sum / grad_metric_chunks as f32,
            examples_seen: indices.len(),
            particle_steps_per_sec,
            dense_pair_interactions_per_sec: particle_steps_per_sec * particle_count as f64,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    struct OracleModelBatchStepStats {
        per_model_loss: Vec<f32>,
        per_model_base_grad_norm: Vec<f32>,
        per_model_base_grad_scale: Vec<f32>,
        particle_steps_per_sec: f64,
        elapsed_ms: f64,
    }

    fn sampled_training_rollout_steps(config: DirectBasisTrainConfig, seed: u64) -> usize {
        let max_steps = config.rollout_steps.max(1);
        let min_steps = config.rollout_step_min.max(1).min(max_steps);
        if min_steps == max_steps {
            return max_steps;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ 0x6d2b_79f5);
        rng.random_range(min_steps..max_steps)
    }

    #[allow(clippy::too_many_arguments)]
    fn train_oracle_model_batch_step_tbptt(
        params: &mut [BurnBaseParams],
        optimizers: &mut [BurnBaseAdamWState],
        targets: &[BurnTargetExample],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
    ) -> Result<OracleModelBatchStepStats, Box<dyn std::error::Error>> {
        if params.is_empty() || params.len() != optimizers.len() || params.len() != targets.len() {
            return Err(
                std::io::Error::other("Burn oracle model batch length mismatch").into(),
            );
        }
        let started = Instant::now();
        let device = &targets[0].target_rgb.device();
        let indices = (0..targets.len()).collect::<Vec<_>>();
        let (mut x, mut s) =
            seed_batch_tensors(targets, &indices, particle_count, config, step_seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(step_seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let chunk_steps = tbptt_chunk_steps(config);
        let rollout_steps = sampled_training_rollout_steps(config, step_seed);
        let chunk_count = rollout_steps.div_ceil(chunk_steps).max(1);
        let mut loss_sums = collect_metrics.then(|| vec![0.0_f32; params.len()]);
        let mut grad_norm_sums = vec![0.0_f32; params.len()];
        let mut grad_scale_sums = vec![0.0_f32; params.len()];
        let mut grad_metric_chunks = 0usize;
        let mut particle_steps = 0.0_f64;
        let mut remaining_steps = rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let param_batch = BurnBaseBatch::from_params(params);
            let displacement = Tensor::<BurnBackend, 1>::zeros([params.len()], device);
            let (next_x, next_s, displacement) = rollout_oracle_model_batch_chunk(
                &param_batch,
                targets,
                &indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            let loss = target_splat_loss_batch_vector_base_only(
                &next_x,
                &next_s,
                targets,
                &indices,
                config,
                displacement,
            );
            if let Some(loss_sums) = loss_sums.as_mut() {
                for (idx, scalars) in loss_vector_scalars(loss.clone())?.into_iter().enumerate() {
                    loss_sums[idx] += scalars.total;
                }
            }
            let mut grads = loss.total.sum().backward();
            for (idx, (param, optimizer)) in params
                .iter_mut()
                .zip(&mut *optimizers)
                .enumerate()
            {
                let (grad_norm, grad_scale) = param.apply_adamw(
                    &mut grads,
                    optimizer,
                    adamw_from_sgd(config.base_sgd),
                    config.per_parameter_grad_normalization,
                    collect_metrics,
                )?;
                if collect_metrics {
                    grad_norm_sums[idx] += grad_norm;
                    grad_scale_sums[idx] += grad_scale;
                }
            }
            if collect_metrics {
                grad_metric_chunks += 1;
            }
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps += params.len() as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let per_model_loss = loss_sums
            .unwrap_or_else(|| vec![0.0; params.len()])
            .into_iter()
            .map(|loss| loss / chunk_count as f32)
            .collect::<Vec<_>>();
        Ok(OracleModelBatchStepStats {
            per_model_loss,
            per_model_base_grad_norm: grad_norm_sums
                .into_iter()
                .map(|value| value / grad_metric_chunks as f32)
                .collect(),
            per_model_base_grad_scale: grad_scale_sums
                .into_iter()
                .map(|value| {
                    if collect_metrics {
                        value / grad_metric_chunks as f32
                    } else {
                        1.0
                    }
                })
                .collect(),
            particle_steps_per_sec: particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    struct ChunkGradStats {
        base_grad_norm: f32,
        base_grad_scale: f32,
        adapter_grad_sum: f32,
        adapter_grad_max: f32,
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_chunk_gradients(
        params: &mut BurnBaseParams,
        adapters: &mut [BurnAdapterParams],
        base_optimizer: &mut BurnBaseAdamWState,
        adapter_optimizers: &mut [BurnAdapterAdamWState],
        indices: &[usize],
        loss_total: Tensor1,
        config: DirectBasisTrainConfig,
        update_base: bool,
        adapter_gradient_scale: f32,
        collect_metrics: bool,
    ) -> AutomataResult<ChunkGradStats> {
        let mut grads = loss_total.backward();
        let (base_grad_norm, base_grad_scale) = if update_base {
            params.apply_adamw(
                &mut grads,
                base_optimizer,
                adamw_from_sgd(config.base_sgd),
                config.per_parameter_grad_normalization,
                collect_metrics,
            )?
        } else {
            (0.0, 1.0)
        };
        let mut adapter_grad_sum = 0.0_f32;
        let mut adapter_grad_max = 0.0_f32;
        for &idx in indices {
            let (grad_norm, _) = adapters[idx].apply_adamw(
                &mut grads,
                &mut adapter_optimizers[idx],
                adamw_from_sgd(config.adapter_sgd),
                config.per_parameter_grad_normalization,
                adapter_gradient_scale,
                collect_metrics,
            )?;
            if collect_metrics {
                adapter_grad_sum += grad_norm;
                adapter_grad_max = adapter_grad_max.max(grad_norm);
            }
        }
        Ok(ChunkGradStats {
            base_grad_norm,
            base_grad_scale,
            adapter_grad_sum,
            adapter_grad_max,
        })
    }

    fn tbptt_chunk_steps(config: DirectBasisTrainConfig) -> usize {
        config
            .tbptt_chunk_steps
            .max(1)
            .min(config.rollout_steps.max(1))
    }

    fn evaluate_targets(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_examples: usize,
        seed: u64,
    ) -> Result<Option<CliHyper2dDirectBasisLossSummary>, Box<dyn std::error::Error>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let mut indices = (0..targets.len()).collect::<Vec<_>>();
        if requested_examples > 0 && requested_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            indices.truncate(requested_examples);
            indices.sort_unstable();
        }
        let mut summary = CliHyper2dDirectBasisLossSummary {
            examples: indices.len(),
            mean_total_loss: 0.0,
            max_total_loss: 0.0,
            mean_splat_loss: 0.0,
            mean_color_loss: 0.0,
            mean_density_loss: 0.0,
        };
        let eval_batch_size = normalized_eval_batch_size(config.eval_batch_size, indices.len());
        for chunk in indices.chunks(eval_batch_size) {
            if homogeneous_particle_count(targets, chunk).is_some() {
                let loss = batch_example_eval_loss(params, adapters, targets, chunk, config, seed)?;
                for scalars in loss_vector_scalars(loss)? {
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            } else {
                for &idx in chunk {
                    let loss = example_eval_loss_bounded(
                        params,
                        &adapters[idx],
                        &targets[idx],
                        config,
                        seed.wrapping_add(idx as u64),
                    );
                    let scalars = loss_scalars(&loss)?;
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            }
        }
        let scale = 1.0 / indices.len() as f32;
        summary.mean_total_loss *= scale;
        summary.mean_splat_loss *= scale;
        summary.mean_color_loss *= scale;
        summary.mean_density_loss *= scale;
        Ok(Some(summary))
    }

    fn evaluate_target_geometry(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_examples: usize,
        seed: u64,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let mut indices = (0..targets.len()).collect::<Vec<_>>();
        if requested_examples > 0 && requested_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            indices.truncate(requested_examples);
            indices.sort_unstable();
        }
        let eval_batch_size = normalized_eval_batch_size(config.eval_batch_size, indices.len());
        let mut total = BurnGeometrySummary {
            examples: 0,
            mean_score: 0.0,
            mean_foreground_iou: 0.0,
            mean_target_recall: 0.0,
            mean_generated_precision: 0.0,
            mean_bbox_iou: 0.0,
            mean_lit_pixel_ratio: 0.0,
            mean_bbox_width_ratio: 0.0,
            mean_bbox_area_ratio: 0.0,
        };
        for chunk in indices.chunks(eval_batch_size) {
            if homogeneous_particle_count(targets, chunk).is_none() {
                continue;
            }
            let Some(summary) =
                batch_example_geometry(params, adapters, targets, chunk, config, seed)?
            else {
                continue;
            };
            let weight = summary.examples as f32;
            total.examples += summary.examples;
            total.mean_score += summary.mean_score * weight;
            total.mean_foreground_iou += summary.mean_foreground_iou * weight;
            total.mean_target_recall += summary.mean_target_recall * weight;
            total.mean_generated_precision += summary.mean_generated_precision * weight;
            total.mean_bbox_iou += summary.mean_bbox_iou * weight;
            total.mean_lit_pixel_ratio += summary.mean_lit_pixel_ratio * weight;
            total.mean_bbox_width_ratio += summary.mean_bbox_width_ratio * weight;
            total.mean_bbox_area_ratio += summary.mean_bbox_area_ratio * weight;
        }
        if total.examples == 0 {
            return Ok(None);
        }
        let scale = 1.0 / total.examples as f32;
        total.mean_score *= scale;
        total.mean_foreground_iou *= scale;
        total.mean_target_recall *= scale;
        total.mean_generated_precision *= scale;
        total.mean_bbox_iou *= scale;
        total.mean_lit_pixel_ratio *= scale;
        total.mean_bbox_width_ratio *= scale;
        total.mean_bbox_area_ratio *= scale;
        Ok(Some(total))
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_e2e_rollout_quality(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        train_examples: &[BurnE2eRolloutExample],
        holdout_examples: &[BurnE2eRolloutExample],
        train_conditions: &BurnE2eConditionCache,
        holdout_conditions: &BurnE2eConditionCache,
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
    ) -> Result<Option<BurnE2eRolloutQualityReport>, Box<dyn std::error::Error>> {
        if config.validation_examples == 0 {
            return Ok(None);
        }
        let (split, examples, conditions) = if holdout_examples.is_empty() {
            ("train", train_examples, train_conditions)
        } else {
            ("holdout", holdout_examples, holdout_conditions)
        };
        if examples.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let mut indices = (0..examples.len()).collect::<Vec<_>>();
        if config.validation_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(config.validation_seed);
            indices.shuffle(&mut rng);
            indices.truncate(config.validation_examples);
            indices.sort_unstable();
        }
        let eval_config = validation_direct_config(config);
        let targets = burn_e2e_targets_with_runtime(
            examples,
            config,
            device,
            Some(config.validation_particles),
            Some(config.validation_update_prob),
        )?;
        let eval_batch_size = normalized_eval_batch_size(eval_config.eval_batch_size, indices.len());
        let mut entries = Vec::with_capacity(indices.len());
        let mut adapter_batches = 0usize;
        for chunk in indices.chunks(eval_batch_size) {
            adapter_batches += 1;
            let quality = batch_e2e_eval_quality(
                params,
                generator,
                npa_config,
                conditions,
                &targets,
                chunk,
                config,
                eval_config,
                config.validation_seed,
                device,
            )?;
            let losses = loss_vector_scalars(quality.loss)?;
            let mses = tensor1_vec(quality.render_rgb_mse.inner())?;
            if losses.len() != chunk.len() || mses.len() != chunk.len() {
                return Err(std::io::Error::other(
                    "HyperNPA e2e quality readback length mismatch",
                )
                .into());
            }
            for ((&idx, loss), render_rgb_mse) in chunk.iter().zip(losses).zip(mses) {
                let render_rgb_mse =
                    finite_scalar("HyperNPA e2e render RGB MSE", render_rgb_mse)?;
                let render_rgb_psnr_db = psnr_db_from_mse(render_rgb_mse);
                entries.push(BurnE2eRolloutQualityEntry {
                    slug: examples[idx].slug.clone(),
                    total_loss: loss.total,
                    splat_loss: loss.splat,
                    color_loss: loss.color,
                    density_loss: loss.density,
                    render_rgb_mse,
                    render_rgb_psnr_db,
                    passed: render_rgb_psnr_db >= config.validation_psnr_threshold_db,
                });
            }
        }
        let examples_count = entries.len();
        if examples_count == 0 {
            return Ok(None);
        }
        let mut mean_total_loss = 0.0_f32;
        let mut mean_splat_loss = 0.0_f32;
        let mut mean_color_loss = 0.0_f32;
        let mut mean_density_loss = 0.0_f32;
        let mut mean_render_rgb_mse = 0.0_f32;
        let mut mean_render_rgb_psnr_db = 0.0_f32;
        let mut min_render_rgb_psnr_db = f32::INFINITY;
        let mut max_render_rgb_psnr_db = f32::NEG_INFINITY;
        for entry in &entries {
            mean_total_loss += entry.total_loss;
            mean_splat_loss += entry.splat_loss;
            mean_color_loss += entry.color_loss;
            mean_density_loss += entry.density_loss;
            mean_render_rgb_mse += entry.render_rgb_mse;
            mean_render_rgb_psnr_db += entry.render_rgb_psnr_db;
            min_render_rgb_psnr_db = min_render_rgb_psnr_db.min(entry.render_rgb_psnr_db);
            max_render_rgb_psnr_db = max_render_rgb_psnr_db.max(entry.render_rgb_psnr_db);
        }
        let scale = 1.0 / examples_count as f32;
        mean_total_loss *= scale;
        mean_splat_loss *= scale;
        mean_color_loss *= scale;
        mean_density_loss *= scale;
        mean_render_rgb_mse *= scale;
        mean_render_rgb_psnr_db *= scale;
        let mean_passed = mean_render_rgb_psnr_db >= config.validation_psnr_threshold_db;
        let all_examples_passed = entries.iter().all(|entry| entry.passed);
        let passed = mean_passed;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let particle_steps =
            examples_count as f64 * config.validation_particles as f64 * config.validation_steps as f64;
        let particle_steps_per_sec =
            particle_steps / (elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
        let dense_pair_interactions_per_sec =
            particle_steps_per_sec * config.validation_particles as f64;
        Ok(Some(BurnE2eRolloutQualityReport {
            split,
            examples: examples_count,
            particle_count: config.validation_particles,
            rollout_steps: config.validation_steps,
            update_prob: config.validation_update_prob,
            seed: config.validation_seed,
            psnr_threshold_db: config.validation_psnr_threshold_db,
            passed,
            mean_passed,
            all_examples_passed,
            elapsed_ms,
            particle_steps,
            particle_steps_per_sec,
            dense_pair_interactions_per_sec,
            adapter_batches,
            mean_total_loss,
            mean_splat_loss,
            mean_color_loss,
            mean_density_loss,
            mean_render_rgb_mse,
            mean_render_rgb_psnr_db,
            min_render_rgb_psnr_db,
            max_render_rgb_psnr_db,
            entries,
        }))
    }

    struct BurnE2eQualityBatchTensors {
        loss: BurnLossBatchTensors,
        render_rgb_mse: Tensor1,
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_e2e_eval_quality(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        targets: &[BurnTargetExample],
        indices: &[usize],
        generator_config: BurnE2eRolloutTrainConfig,
        eval_config: DirectBasisTrainConfig,
        seed: u64,
        device: &BurnDevice,
    ) -> Result<BurnE2eQualityBatchTensors, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Err(std::io::Error::other(
                "HyperNPA e2e quality validation requires homogeneous particle counts",
            )
            .into());
        };
        let condition = conditions.select(indices)?;
        let adapter_batch = generator.adapter_batch(condition, npa_config, generator_config);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, eval_config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(eval_config);
        let mut remaining_steps = eval_config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                eval_config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        Ok(target_splat_quality_batch_vector(
            &x,
            &s,
            targets,
            indices,
            eval_config,
            &adapter_batch,
            displacement,
        ))
    }

    fn normalized_eval_batch_size(requested: usize, examples: usize) -> usize {
        if requested == 0 {
            examples.max(1)
        } else {
            requested.min(examples).max(1)
        }
    }

    fn e2e_lr_scale(config: BurnE2eRolloutTrainConfig, step: usize) -> f32 {
        if config.steps <= 1 {
            return 1.0;
        }
        let min_scale = config.min_lr_scale.clamp(0.0, 1.0);
        let progress = step.saturating_sub(1) as f32 / config.steps.saturating_sub(1) as f32;
        let raw_scale = match config.lr_schedule {
            E2eLrSchedule::Constant => 1.0,
            E2eLrSchedule::Linear => 1.0 - progress,
            E2eLrSchedule::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()),
        };
        min_scale + (1.0 - min_scale) * raw_scale.clamp(0.0, 1.0)
    }

    fn e2e_config_with_lr_scale(
        mut config: BurnE2eRolloutTrainConfig,
        lr_scale: f32,
    ) -> BurnE2eRolloutTrainConfig {
        let lr_scale = lr_scale.clamp(0.0, 1.0);
        config.base_optimizer.learning_rate *= lr_scale;
        config.generator_optimizer.learning_rate *= lr_scale;
        config
    }

    fn reported_particle_step_speed_summary(
        history: &[BurnE2eRolloutHistoryEntry],
    ) -> (f64, f64, f64) {
        let mut speeds = history
            .iter()
            .map(|entry| entry.particle_steps_per_sec)
            .filter(|speed| speed.is_finite())
            .collect::<Vec<_>>();
        speeds.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        let min = speeds.first().copied().unwrap_or_default();
        let median = speeds.get(speeds.len() / 2).copied().unwrap_or_default();
        let max = speeds.last().copied().unwrap_or_default();
        (min, median, max)
    }

    fn reported_loss_summary(
        history: &[BurnE2eRolloutHistoryEntry],
    ) -> (Option<f32>, Option<f32>, usize, Option<f32>) {
        let first = history
            .iter()
            .find(|entry| entry.loss.is_finite())
            .map(|entry| entry.loss);
        let final_loss = history
            .iter()
            .rev()
            .find(|entry| entry.loss.is_finite())
            .map(|entry| entry.loss);
        let Some(best) = history
            .iter()
            .filter(|entry| entry.loss.is_finite())
            .min_by(|lhs, rhs| lhs.loss.total_cmp(&rhs.loss))
        else {
            return (first, None, 0, final_loss);
        };
        (first, Some(best.loss), best.step, final_loss)
    }

    fn format_optional_f32(value: Option<f32>) -> String {
        value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
    }

    fn psnr_db_from_mse(mse: f32) -> f32 {
        let mse = mse.max(1.0e-12);
        finite_scalar("HyperNPA e2e render RGB PSNR", 10.0 * (1.0 / mse).log10()).unwrap_or(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_single_chunk(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        mut x: Tensor2,
        mut s: Tensor2,
        config: DirectBasisTrainConfig,
        rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor2, Tensor2, Tensor1) {
        for _ in 0..steps {
            let features = rollout_dense_perception(&x, &s, config);
            let update = params.forward_adapter(features, adapter, config);
            let dx_raw = update.clone().narrow(1, 0, 2);
            let ds = update.narrow(1, 2, s.shape().dims::<2>()[1]);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(1)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([target.particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(1)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .mean();
            displacement = displacement + dx_norm;
            let mask = tensor(
                stochastic_mask(target.particle_count, target.update_prob, rng),
                [target.particle_count, 1],
                &target.target_rgb.device(),
            );
            let state_dims = s.shape().dims::<2>()[1];
            x = x + dx.mul(mask.clone().expand([target.particle_count, 2]));
            s = s + ds.mul(mask.expand([target.particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_batch_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        for _ in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let update = params.forward_adapter_batch(features, adapter_batch);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([indices.len(), particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .mean();
            displacement = displacement + dx_norm;
            let mask = tensor3(
                batch_masks(targets, indices, particle_count, rng),
                [indices.len(), particle_count, 1],
                &targets[indices[0]].target_rgb.device(),
            );
            x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
            s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_batch_eval_chunk(
        params: &BurnBaseParams,
        adapter_batch: &BurnAdapterBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rngs: &mut [StdRng],
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        for _ in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let update = params.forward_adapter_batch(features, adapter_batch);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([indices.len(), particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .reshape([indices.len(), particle_count])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            displacement = displacement + dx_norm;
            let mask = tensor3(
                batch_masks_with_rngs(targets, indices, particle_count, rngs),
                [indices.len(), particle_count, 1],
                &targets[indices[0]].target_rgb.device(),
            );
            x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
            s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_oracle_model_batch_chunk(
        params: &BurnBaseBatch,
        targets: &[BurnTargetExample],
        indices: &[usize],
        mut x: Tensor3,
        mut s: Tensor3,
        config: DirectBasisTrainConfig,
        particle_count: usize,
        rngs: &mut [StdRng],
        steps: usize,
        mut displacement: Tensor1,
    ) -> (Tensor3, Tensor3, Tensor1) {
        for _ in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let update = params.forward(features);
            let state_dims = s.shape().dims::<3>()[2];
            let dx_raw = update.clone().narrow(2, 0, 2);
            let ds = update.narrow(2, 2, state_dims);
            let norm = dx_raw
                .clone()
                .mul(dx_raw.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .add_scalar(1.0)
                .expand([indices.len(), particle_count, 2]);
            let dx = dx_raw.mul_scalar(config.motion_scale).div(norm);
            let dx_norm = dx
                .clone()
                .mul(dx.clone())
                .sum_dim(2)
                .add_scalar(EPSILON * EPSILON)
                .sqrt()
                .reshape([indices.len(), particle_count])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            displacement = displacement + dx_norm;
            let mask = tensor3(
                batch_masks_with_rngs(targets, indices, particle_count, rngs),
                [indices.len(), particle_count, 1],
                &targets[indices[0]].target_rgb.device(),
            );
            x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
            s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
        }
        (x, s, displacement)
    }

    fn example_eval_loss_bounded(
        params: &BurnBaseParams,
        adapter: &BurnAdapterParams,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> BurnLossTensors {
        let device = &target.target_rgb.device();
        let (mut x, mut s) = seed_tensors(
            target.particle_count,
            config,
            target.seed_scale,
            seed,
            device,
        );
        let mut rng = StdRng::seed_from_u64(seed ^ 0x005e_ed2d);
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_single_chunk(
                params,
                adapter,
                target,
                x,
                s,
                config,
                &mut rng,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach2(x);
                s = detach2(s);
                displacement = detach1(displacement);
            }
        }
        target_splat_loss(&x, &s, target, config, adapter, displacement)
    }

    fn batch_example_eval_loss(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> Result<BurnLossBatchTensors, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Err(std::io::Error::other(
                "Burn eval batch path requires homogeneous particle counts",
            )
            .into());
        };
        let device = &targets[indices[0]].target_rgb.device();
        let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        Ok(target_splat_loss_batch_vector(
            &x,
            &s,
            targets,
            indices,
            config,
            &adapter_batch,
            displacement,
        ))
    }

    fn batch_example_geometry(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        seed: u64,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        let Some(particle_count) = homogeneous_particle_count(targets, indices) else {
            return Ok(None);
        };
        let device = &targets[indices[0]].target_rgb.device();
        let adapter_batch = BurnAdapterBatch::from_indices(adapters, indices);
        let (mut x, mut s) =
            seed_batch_tensors(targets, indices, particle_count, config, seed, device);
        let mut rngs = indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(config);
        let mut remaining_steps = config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                indices,
                x,
                s,
                config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }

        let centered = if config.loss_config.center {
            let target_mean = stack_target_mean(targets, indices);
            x.clone() - x.clone().mean_dim(1).expand([indices.len(), particle_count, 2])
                + target_mean.expand([indices.len(), particle_count, 2])
        } else {
            x.clone()
        };
        let state_dims = s.shape().dims::<3>()[2];
        let colors = s.narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (_, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let target_density = stack_target_density(targets, indices);
        geometry_summary_from_density(
            tensor3_vec(density.inner())?,
            tensor3_vec(target_density.inner())?,
            indices.len(),
            config.loss_config.image_size,
        )
    }

    fn homogeneous_particle_count(
        targets: &[BurnTargetExample],
        indices: &[usize],
    ) -> Option<usize> {
        let mut iter = indices.iter().map(|idx| targets[*idx].particle_count);
        let first = iter.next()?;
        iter.all(|count| count == first).then_some(first)
    }

    fn geometry_summary_from_density(
        density: Vec<f32>,
        target_density: Vec<f32>,
        batches: usize,
        image_size: usize,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        let pixels = image_size * image_size;
        if batches == 0 {
            return Ok(None);
        }
        if density.len() != batches * pixels || target_density.len() != batches * pixels {
            return Err(std::io::Error::other(format!(
                "Burn geometry density shape mismatch: density={} target={} expected={}",
                density.len(),
                target_density.len(),
                batches * pixels
            ))
            .into());
        }
        let mut summary = BurnGeometrySummary {
            examples: batches,
            mean_score: 0.0,
            mean_foreground_iou: 0.0,
            mean_target_recall: 0.0,
            mean_generated_precision: 0.0,
            mean_bbox_iou: 0.0,
            mean_lit_pixel_ratio: 0.0,
            mean_bbox_width_ratio: 0.0,
            mean_bbox_area_ratio: 0.0,
        };
        for batch in 0..batches {
            let start = batch * pixels;
            let end = start + pixels;
            let generated = &density[start..end];
            let target = &target_density[start..end];
            let threshold = target
                .iter()
                .copied()
                .fold(0.0_f32, |max_value, value| max_value.max(value))
                .mul_add(0.05, 0.0)
                .max(1.0e-6);
            let (lit_pixels, bbox) = density_lit_stats(generated, image_size, threshold)?;
            let (target_lit_pixels, target_bbox) =
                density_lit_stats(target, image_size, threshold)?;
            let lit_ratio = lit_pixels as f32 / target_lit_pixels.max(1) as f32;
            let iou = bbox_iou(bbox, target_bbox).unwrap_or(0.0);
            let width_ratio = bbox_width_ratio(bbox, target_bbox).unwrap_or(0.0);
            let area_ratio = bbox_area_ratio(bbox, target_bbox).unwrap_or(0.0);
            let overlap = density_overlap_stats(generated, target, threshold)?;
            let score = 1.5 * overlap.iou
                + 0.5 * overlap.target_recall
                + 0.25 * overlap.generated_precision
                + 0.25 * iou
                - 0.25 * (lit_ratio - 1.0).abs()
                - 0.35 * (width_ratio - 1.0).abs()
                - 0.15 * (area_ratio - 1.0).abs();
            summary.mean_score += score;
            summary.mean_foreground_iou += overlap.iou;
            summary.mean_target_recall += overlap.target_recall;
            summary.mean_generated_precision += overlap.generated_precision;
            summary.mean_bbox_iou += iou;
            summary.mean_lit_pixel_ratio += lit_ratio;
            summary.mean_bbox_width_ratio += width_ratio;
            summary.mean_bbox_area_ratio += area_ratio;
        }
        let scale = 1.0 / batches as f32;
        summary.mean_score *= scale;
        summary.mean_foreground_iou *= scale;
        summary.mean_target_recall *= scale;
        summary.mean_generated_precision *= scale;
        summary.mean_bbox_iou *= scale;
        summary.mean_lit_pixel_ratio *= scale;
        summary.mean_bbox_width_ratio *= scale;
        summary.mean_bbox_area_ratio *= scale;
        Ok(Some(summary))
    }

    #[derive(Clone, Copy)]
    struct BurnDensityOverlapStats {
        iou: f32,
        target_recall: f32,
        generated_precision: f32,
    }

    fn density_overlap_stats(
        generated: &[f32],
        target: &[f32],
        threshold: f32,
    ) -> Result<BurnDensityOverlapStats, Box<dyn std::error::Error>> {
        if generated.len() != target.len() {
            return Err(std::io::Error::other("Burn geometry density overlap shape mismatch").into());
        }
        let mut generated_count = 0usize;
        let mut target_count = 0usize;
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (&generated_density, &target_density) in generated.iter().zip(target) {
            let generated_hit = generated_density >= threshold;
            let target_hit = target_density >= threshold;
            generated_count += usize::from(generated_hit);
            target_count += usize::from(target_hit);
            intersection += usize::from(generated_hit && target_hit);
            union += usize::from(generated_hit || target_hit);
        }
        Ok(BurnDensityOverlapStats {
            iou: intersection as f32 / union.max(1) as f32,
            target_recall: intersection as f32 / target_count.max(1) as f32,
            generated_precision: intersection as f32 / generated_count.max(1) as f32,
        })
    }

    fn density_lit_stats(
        density: &[f32],
        image_size: usize,
        threshold: f32,
    ) -> Result<(usize, Option<[usize; 4]>), Box<dyn std::error::Error>> {
        if density.len() != image_size * image_size {
            return Err(std::io::Error::other("Burn geometry density shape mismatch").into());
        }
        let mut lit_pixels = 0usize;
        let mut min_x = image_size;
        let mut min_y = image_size;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for y in 0..image_size {
            for x in 0..image_size {
                if density[y * image_size + x] < threshold {
                    continue;
                }
                lit_pixels += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        Ok((
            lit_pixels,
            (lit_pixels > 0).then_some([min_x, min_y, max_x, max_y]),
        ))
    }

    fn bbox_iou(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        let left = left?;
        let right = right?;
        let x0 = left[0].max(right[0]);
        let y0 = left[1].max(right[1]);
        let x1 = left[2].min(right[2]);
        let y1 = left[3].min(right[3]);
        let intersection = if x1 >= x0 && y1 >= y0 {
            bbox_area([x0, y0, x1, y1])
        } else {
            0.0
        };
        let union = bbox_area(left) + bbox_area(right) - intersection;
        Some(intersection / union.max(f32::MIN_POSITIVE))
    }

    fn bbox_width_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        Some(bbox_width(left?) / bbox_width(right?).max(f32::MIN_POSITIVE))
    }

    fn bbox_area_ratio(left: Option<[usize; 4]>, right: Option<[usize; 4]>) -> Option<f32> {
        Some(bbox_area(left?) / bbox_area(right?).max(f32::MIN_POSITIVE))
    }

    fn bbox_width(bbox: [usize; 4]) -> f32 {
        bbox[2].saturating_sub(bbox[0]).saturating_add(1) as f32
    }

    fn bbox_height(bbox: [usize; 4]) -> f32 {
        bbox[3].saturating_sub(bbox[1]).saturating_add(1) as f32
    }

    fn bbox_area(bbox: [usize; 4]) -> f32 {
        bbox_width(bbox) * bbox_height(bbox)
    }

    fn rollout_dense_perception(
        x: &Tensor2,
        s: &Tensor2,
        config: DirectBasisTrainConfig,
    ) -> Tensor2 {
        let feature_x = if config.stopgrad_pos {
            detach2(x.clone())
        } else {
            x.clone()
        };
        let feature_s = if config.stopgrad_state {
            detach2(s.clone())
        } else {
            s.clone()
        };
        dense_perception(&feature_x, &feature_s, config)
    }

    fn rollout_dense_perception_batch(
        x: &Tensor3,
        s: &Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let feature_x = if config.stopgrad_pos {
            detach3(x.clone())
        } else {
            x.clone()
        };
        let feature_s = if config.stopgrad_state {
            detach3(s.clone())
        } else {
            s.clone()
        };
        dense_perception_batch(&feature_x, &feature_s, config)
    }

    fn dense_perception(x: &Tensor2, s: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let density = dense_particle_density(x, config);
        let chunk_size = dense_query_chunk_size(1, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_chunk(x, s, &density, config, start, len));
        }
        Tensor::cat(chunks, 0)
    }

    fn dense_perception_chunk(
        x: &Tensor2,
        s: &Tensor2,
        density: &Tensor2,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let xi = x
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, 2]);
        let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density.clone().transpose().recip().expand([len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff
            .clone()
            .mul(spiky_mag.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let density_grad = log_normalize_vectors(
            grad.clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(volume_j.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let state_grad = state_diff
            .unsqueeze_dim::<4>(3)
            .expand([len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<4>(2)
                    .expand([len, rows, state_dims, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let state_grad = apply_moment_correction_2d(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(0, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            1,
        )
    }

    fn dense_perception_batch(x: &Tensor3, s: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let density = dense_particle_density_batch(x, config);
        let chunk_size =
            dense_query_chunk_size(batches, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_batch_chunk(
                x, s, &density, config, start, len,
            ));
        }
        Tensor::cat(chunks, 1)
    }

    fn dense_perception_batch_chunk(
        x: &Tensor3,
        s: &Tensor3,
        density: &Tensor3,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let xi = x
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, 2]);
        let xj = x
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density
            .clone()
            .swap_dims(1, 2)
            .recip()
            .expand([batches, len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff.clone().mul(
            spiky_mag
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let density_grad = log_normalize_vectors_batch(
            grad.clone()
                .sum_dim(2)
                .squeeze_dim::<3>(2)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(
            volume_j
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let state_grad = state_diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<5>(3)
                    .expand([batches, len, rows, state_dims, 2]),
            )
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let state_grad = apply_moment_correction_2d_batch(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient_batch(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(1, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            2,
        )
    }

    fn background_density_term(density: Tensor2, foreground: Tensor2) -> Tensor2 {
        let background = foreground.mul_scalar(-1.0).add_scalar(1.0);
        let leak = density.mul(background);
        leak.clone().mul(leak)
    }

    fn background_density_term_batch(density: Tensor3, foreground: Tensor3) -> Tensor3 {
        let background = foreground.mul_scalar(-1.0).add_scalar(1.0);
        let leak = density.mul(background);
        leak.clone().mul(leak)
    }

    fn foreground_density_term(
        density: Tensor2,
        target_density: Tensor2,
        foreground: Tensor2,
        foreground_scale: f32,
    ) -> Tensor1 {
        l1l2_tensor((density - target_density).mul(foreground))
            .mean()
            .mul_scalar(foreground_scale)
    }

    fn foreground_density_term_batch(
        density: Tensor3,
        target_density: Tensor3,
        foreground: Tensor3,
        foreground_scales: Tensor3,
    ) -> Tensor3 {
        l1l2_tensor3((density - target_density).mul(foreground))
            .mul(foreground_scales)
    }

    fn target_splat_loss(
        x: &Tensor2,
        s: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterParams,
        displacement: Tensor1,
    ) -> BurnLossTensors {
        let particle_count = x.shape().dims::<2>()[0];
        let state_dims = s.shape().dims::<2>()[1];
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(0).expand([particle_count, 2])
                + target.target_mean.clone().expand([particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(1, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) = splat_render(&centered, &colors, target, config, particle_count);
        let background_density_loss =
            background_density_term(density.clone(), target.target_foreground.clone()).mean();
        let foreground_density_loss = foreground_density_term(
            density.clone(),
            target.target_density.clone(),
            target.target_foreground.clone(),
            target.target_foreground_scale,
        );
        let density_diff = density - target.target_density.clone();
        let density_term = l1l2_tensor(density_diff);
        let density_loss = density_term.clone().mean();
        let color_gate = target_2d_detached_color_gate2(density_term).expand([
            config.loss_config.image_size * config.loss_config.image_size,
            3,
        ]);
        let color_loss = l1l2_tensor(rgb - target.target_rgb.clone())
            .mul(color_gate)
            .mean();
        let shape_chamfer_loss = target_shape_chamfer_loss(&centered, target, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound = relu(x.clone().abs().add_scalar(-1.0));
        let bound_loss = bound.mean();
        let overflow = relu(s.clone().abs().add_scalar(-1.0));
        let overflow_loss = overflow.mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total + adapter.l2_loss().mul_scalar(config.adapter_l2_weight);
        }
        BurnLossTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_splat_loss_batch(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnLossTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let background_density_loss =
            background_density_term_batch(density.clone(), stack_target_foreground(targets, indices))
                .mean();
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            stack_target_foreground_scales(targets, indices),
        )
        .mean();
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term.clone().mean();
        let color_gate = target_2d_detached_color_gate3(density_term).expand([
            batches,
            config.loss_config.image_size * config.loss_config.image_size,
            3,
        ]);
        let color_loss = l1l2_tensor3(rgb - stack_target_rgb(targets, indices))
            .mul(color_gate)
            .mean();
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config).mean();
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0)).mean();
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0)).mean();
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total + adapter.l2_loss().mul_scalar(config.adapter_l2_weight);
        }
        BurnLossTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_splat_loss_batch_vector(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnLossBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            stack_target_foreground(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            stack_target_foreground_scales(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb - stack_target_rgb(targets, indices))
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }
        BurnLossBatchTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_splat_quality_batch_vector(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> BurnE2eQualityBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let target_rgb = stack_target_rgb(targets, indices);
        let rgb_diff = rgb - target_rgb;
        let render_rgb_mse = rgb_diff
            .clone()
            .mul(rgb_diff.clone())
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            stack_target_foreground(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            stack_target_foreground_scales(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb_diff)
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }
        BurnE2eQualityBatchTensors {
            loss: BurnLossBatchTensors {
                total,
                splat,
                color: color_loss,
                density: density_loss,
            },
            render_rgb_mse,
        }
    }

    fn target_splat_loss_batch_vector_base_only(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        displacement: Tensor1,
    ) -> BurnLossBatchTensors {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let target_mean = stack_target_mean(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            stack_target_foreground(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            stack_target_foreground_scales(targets, indices),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let density_diff = density - target_density;
        let density_term = l1l2_tensor3(density_diff);
        let density_loss = density_term
            .clone()
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let color_gate = target_2d_detached_color_gate3(density_term).expand([batches, pixels, 3]);
        let color_loss = l1l2_tensor3(rgb - stack_target_rgb(targets, indices))
            .mul(color_gate)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let shape_chamfer_loss =
            target_shape_chamfer_loss_batch_vector(&centered, targets, indices, config);
        let splat = color_loss
            .clone()
            .mul_scalar(config.loss_config.color_loss_weight)
            + density_loss
                .clone()
                .mul_scalar(config.loss_config.density_loss_weight)
            + background_density_loss
                .clone()
                .mul_scalar(config.loss_config.background_density_loss_weight)
            + foreground_density_loss
                .clone()
                .mul_scalar(config.loss_config.foreground_density_loss_weight);
        let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * 2])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
            .reshape([batches, particle_count * state_dims])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let total = splat
            .clone()
            .mul_scalar(config.loss_config.splat_loss_weight)
            + shape_chamfer_loss.mul_scalar(config.loss_config.shape_chamfer_loss_weight)
            + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight)
            + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight)
            + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
        BurnLossBatchTensors {
            total,
            splat,
            color: color_loss,
            density: density_loss,
        }
    }

    fn target_shape_chamfer_loss(
        x: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
    ) -> Tensor1 {
        if config.loss_config.shape_chamfer_loss_weight <= 0.0 {
            return Tensor::<BurnBackend, 1>::zeros([1], &target.target_rgb.device());
        }
        let particle_count = x.shape().dims::<2>()[0];
        let target_count = target.target_positions.shape().dims::<2>()[0];
        if particle_count == 0 || target_count == 0 {
            return Tensor::<BurnBackend, 1>::zeros([1], &target.target_rgb.device());
        }
        let particle_i = x
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([particle_count, target_count, 2]);
        let target_j = target
            .target_positions
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([particle_count, target_count, 2]);
        let diff = particle_i - target_j;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        let particle_to_target = dist2.clone().min_dim(1).mean();
        let target_to_particle = dist2.min_dim(0).mean();
        particle_to_target + target_to_particle
    }

    fn target_shape_chamfer_loss_batch_vector(
        x: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
    ) -> Tensor1 {
        if config.loss_config.shape_chamfer_loss_weight <= 0.0 {
            return Tensor::<BurnBackend, 1>::zeros(
                [indices.len()],
                &targets[indices[0]].target_rgb.device(),
            );
        }
        Tensor::cat(
            indices
                .iter()
                .enumerate()
                .map(|(local, idx)| {
                    let x_local = x.clone().narrow(0, local, 1).squeeze_dim::<2>(0);
                    target_shape_chamfer_loss(&x_local, &targets[*idx], config)
                })
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn splat_render(
        x: &Tensor2,
        colors: &Tensor2,
        target: &BurnTargetExample,
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor2, Tensor2) {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions(x, config);
        let sigma =
            (config.loss_config.sigma * config.loss_config.image_size as f32 * target.pixel_size
                / (config.loss_config.hi - config.loss_config.lo))
                .max(EPSILON);
        let denom =
            splat_particle_denominator(&particle_pixels, target, particle_count, sigma, config);
        let norm_scale = (config.loss_config.image_size as f32 * target.pixel_size
            / (config.loss_config.hi - config.loss_config.lo))
            .powi(2);
        let output_scale = target.target_points as f32 / particle_count.max(1) as f32;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut rgbs = Vec::new();
        let mut densities = Vec::new();
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(&particle_pixels, target, particle_count, sigma, start, len);
            let weights = g
                .div(denom.clone().expand([len, particle_count]))
                .mul_scalar(output_scale * norm_scale);
            densities.push(weights.clone().sum_dim(1));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 0), Tensor::cat(densities, 0))
    }

    fn splat_render_batch(
        x: &Tensor3,
        colors: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        particle_count: usize,
    ) -> (Tensor3, Tensor3) {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let particle_pixels = particle_pixel_positions_batch(x, config);
        let sigma = stack_pixel_sizes(targets, indices)
            .mul_scalar(config.loss_config.sigma * config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo)
            .clamp_min(EPSILON);
        let denom = splat_particle_denominator_batch(
            &particle_pixels,
            targets,
            indices,
            particle_count,
            sigma.clone(),
            config,
        );
        let norm_scale = stack_pixel_sizes(targets, indices)
            .mul_scalar(config.loss_config.image_size as f32)
            .div_scalar(config.loss_config.hi - config.loss_config.lo);
        let norm_scale = norm_scale.clone().mul(norm_scale);
        let output_scale =
            stack_target_point_counts(targets, indices).div_scalar(particle_count.max(1) as f32);
        let chunk_size = splat_pixel_chunk_size(
            batches,
            particle_count,
            pixels,
            config.max_splat_chunk_floats,
        );
        let mut rgbs = Vec::new();
        let mut densities = Vec::new();
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g = splat_gaussian_batch_chunk(
                &particle_pixels,
                targets,
                indices,
                particle_count,
                sigma.clone(),
                start,
                len,
            );
            let weights = g
                .div(denom.clone().expand([batches, len, particle_count]))
                .mul(norm_scale.clone().expand([batches, len, particle_count]))
                .mul(output_scale.clone().expand([batches, len, particle_count]));
            densities.push(weights.clone().sum_dim(2));
            rgbs.push(weights.matmul(colors.clone()));
        }
        (Tensor::cat(rgbs, 1), Tensor::cat(densities, 1))
    }

    fn dense_particle_density(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let rows = x.shape().dims::<2>()[0];
        let chunk_size = dense_query_chunk_size(1, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(0, start, len)
                .unsqueeze_dim::<3>(1)
                .expand([len, rows, 2]);
            let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(1)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 0)
    }

    fn dense_particle_density_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let chunk_size = dense_query_chunk_size(batches, rows, 1, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            let xi = x
                .clone()
                .narrow(1, start, len)
                .unsqueeze_dim::<4>(2)
                .expand([batches, len, rows, 2]);
            let xj = x
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, rows, 2]);
            let diff = xj - xi;
            let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
            let eps = config.grid_eps.max(EPSILON);
            let compact = relu(dist2.mul_scalar(-1.0).add_scalar(eps * eps));
            let compact2 = compact.clone().mul(compact.clone());
            chunks.push(
                compact2
                    .mul(compact)
                    .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)))
                    .sum_dim(2)
                    .clamp_min(EPSILON),
            );
        }
        Tensor::cat(chunks, 1)
    }

    fn splat_particle_denominator(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        config: DirectBasisTrainConfig,
    ) -> Tensor2 {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size =
            splat_pixel_chunk_size(1, particle_count, pixels, config.max_splat_chunk_floats);
        let mut denom = None::<Tensor2>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g =
                splat_gaussian_chunk(particle_pixels, target, particle_count, sigma, start, len);
            let contribution = g.sum_dim(0);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 2>::zeros([1, particle_count], &target.target_rgb.device())
            })
            .add_scalar(EPSILON)
    }

    fn splat_particle_denominator_batch(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let chunk_size = splat_pixel_chunk_size(
            batches,
            particle_count,
            pixels,
            config.max_splat_chunk_floats,
        );
        let mut denom = None::<Tensor3>;
        for (start, len) in chunks_for(pixels, chunk_size) {
            let g = splat_gaussian_batch_chunk(
                particle_pixels,
                targets,
                indices,
                particle_count,
                sigma.clone(),
                start,
                len,
            );
            let contribution = g.sum_dim(1);
            denom = Some(match denom {
                Some(value) => value + contribution,
                None => contribution,
            });
        }
        denom
            .unwrap_or_else(|| {
                Tensor::<BurnBackend, 3>::zeros(
                    [batches, 1, particle_count],
                    &targets[indices[0]].target_rgb.device(),
                )
            })
            .add_scalar(EPSILON)
    }

    fn splat_gaussian_chunk(
        particle_pixels: &Tensor2,
        target: &BurnTargetExample,
        particle_count: usize,
        sigma: f32,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let pixel_i = target
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(2).squeeze_dim::<2>(2);
        dist2.mul_scalar(-0.5 / (sigma * sigma)).exp()
    }

    fn splat_gaussian_batch_chunk(
        particle_pixels: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        sigma: Tensor3,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let batches = indices.len();
        let pixel_i = targets[indices[0]]
            .pixel_xy
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, particle_count, 2]);
        let particle_j =
            particle_pixels
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batches, len, particle_count, 2]);
        let diff = pixel_i - particle_j;
        let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
        let sigma2 = sigma
            .clone()
            .mul(sigma)
            .expand([batches, len, particle_count]);
        dist2.mul_scalar(-0.5).div(sigma2).exp()
    }

    fn chunks_for(total: usize, chunk_size: usize) -> impl Iterator<Item = (usize, usize)> {
        let chunk_size = chunk_size.max(1);
        (0..total)
            .step_by(chunk_size)
            .map(move |start| (start, (total - start).min(chunk_size)))
    }

    fn dense_query_chunk_size(
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

    fn splat_pixel_chunk_size(
        batches: usize,
        particle_count: usize,
        pixels: usize,
        max_floats: usize,
    ) -> usize {
        let denominator = batches
            .max(1)
            .saturating_mul(particle_count.max(1))
            .saturating_mul(2)
            .max(1);
        (max_floats / denominator).max(1).min(pixels.max(1))
    }

    fn particle_pixel_positions(x: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(1, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(1, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 1)
    }

    fn particle_pixel_positions_batch(x: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let size = config.loss_config.image_size as f32;
        let world_scale = (size - 1.0) / (config.loss_config.hi - config.loss_config.lo);
        let px = x
            .clone()
            .narrow(2, 0, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(world_scale);
        let py = x
            .clone()
            .narrow(2, 1, 1)
            .add_scalar(-config.loss_config.lo)
            .mul_scalar(-world_scale)
            .add_scalar(size - 1.0);
        Tensor::cat(vec![px, py], 2)
    }

    fn l1l2_tensor(value: Tensor2) -> Tensor2 {
        value.clone().abs() + value.clone().mul(value)
    }

    fn l1l2_tensor3(value: Tensor3) -> Tensor3 {
        value.clone().abs() + value.clone().mul(value)
    }

    fn log_normalize_vectors(values: Tensor2) -> Tensor2 {
        let dims = values.shape().dims::<2>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(1)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values * norm.clone().log1p().div(norm).expand([dims[0], dims[1]])
    }

    fn log_normalize_vectors_batch(values: Tensor3) -> Tensor3 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2]])
    }

    fn log_normalize_state_gradient(values: Tensor3) -> Tensor2 {
        let dims = values.shape().dims::<3>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(2)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2]]))
        .reshape([dims[0], dims[1] * dims[2]])
    }

    fn log_normalize_state_gradient_batch(values: Tensor4) -> Tensor3 {
        let dims = values.shape().dims::<4>();
        let norm = values
            .clone()
            .mul(values.clone())
            .sum_dim(3)
            .add_scalar(EPSILON * EPSILON)
            .sqrt()
            .clamp_min(EPSILON);
        (values
            * norm
                .clone()
                .log1p()
                .div(norm)
                .expand([dims[0], dims[1], dims[2], dims[3]]))
        .reshape([dims[0], dims[1], dims[2] * dims[3]])
    }

    fn apply_moment_correction_2d(
        state_gradient: Tensor3,
        diff: Tensor3,
        volume_grad: Tensor3,
    ) -> Tensor3 {
        let dims = state_gradient.shape().dims::<3>();
        let query_rows = dims[0];
        let state_dims = dims[1];
        let neighbor_rows = diff.shape().dims::<3>()[1];
        let moment = diff
            .unsqueeze_dim::<4>(3)
            .expand([query_rows, neighbor_rows, 2, 2])
            .mul(
                volume_grad
                    .unsqueeze_dim::<4>(2)
                    .expand([query_rows, neighbor_rows, 2, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let a = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 0, 1)
            .reshape([query_rows, 1]);
        let b = moment
            .clone()
            .narrow(1, 0, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let d = moment
            .narrow(1, 1, 1)
            .narrow(2, 1, 1)
            .reshape([query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones = Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device());
        let zeros = Tensor::<BurnBackend, 2>::zeros([query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros.clone());
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 2>::ones([query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(2, 0, 1);
        let gy = state_gradient.narrow(2, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<3>(1)
            .expand([query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 2)
    }

    fn apply_moment_correction_2d_batch(
        state_gradient: Tensor4,
        diff: Tensor4,
        volume_grad: Tensor4,
    ) -> Tensor4 {
        let dims = state_gradient.shape().dims::<4>();
        let batches = dims[0];
        let query_rows = dims[1];
        let state_dims = dims[2];
        let neighbor_rows = diff.shape().dims::<4>()[2];
        let moment = diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, query_rows, neighbor_rows, 2, 2])
            .mul(volume_grad.unsqueeze_dim::<5>(3).expand([
                batches,
                query_rows,
                neighbor_rows,
                2,
                2,
            ]))
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let a = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 0, 1)
            .reshape([batches, query_rows, 1]);
        let b = moment
            .clone()
            .narrow(2, 0, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let d = moment
            .narrow(2, 1, 1)
            .narrow(3, 1, 1)
            .reshape([batches, query_rows, 1]);
        let det = a.clone().mul(d.clone()) - b.clone().mul(b.clone());
        let near_singular = det.clone().abs().lower_elem(1.0e-3);
        let ones =
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device());
        let zeros =
            Tensor::<BurnBackend, 3>::zeros([batches, query_rows, 1], &state_gradient.device());
        let inv_det = det.mask_where(near_singular.clone(), ones.clone()).recip();
        let inv00 = d
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), ones);
        let inv01 = b
            .mul_scalar(-1.0)
            .mul(inv_det.clone())
            .mask_where(near_singular.clone(), zeros);
        let inv11 = a.mul(inv_det).mask_where(
            near_singular,
            Tensor::<BurnBackend, 3>::ones([batches, query_rows, 1], &state_gradient.device()),
        );
        let gx = state_gradient.clone().narrow(3, 0, 1);
        let gy = state_gradient.narrow(3, 1, 1);
        let inv00 = inv00
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv01 = inv01
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let inv11 = inv11
            .unsqueeze_dim::<4>(2)
            .expand([batches, query_rows, state_dims, 1]);
        let corrected_x = gx.clone().mul(inv00) + gy.clone().mul(inv01.clone());
        let corrected_y = gx.mul(inv01) + gy.mul(inv11);
        Tensor::cat(vec![corrected_x, corrected_y], 3)
    }

    #[derive(Clone, Copy)]
    struct AdamWBiasCorrection {
        beta1: f32,
        beta2: f32,
    }

    impl BurnBaseAdamWState {
        fn zeros_like(params: &BurnBaseParams) -> Self {
            Self {
                step: 0,
                w1_m: params.w1.clone().inner().zeros_like(),
                w1_v: params.w1.clone().inner().zeros_like(),
                b1_m: params.b1.clone().inner().zeros_like(),
                b1_v: params.b1.clone().inner().zeros_like(),
                w2_m: params.w2.clone().inner().zeros_like(),
                w2_v: params.w2.clone().inner().zeros_like(),
                b2_m: params.b2.clone().inner().zeros_like(),
                b2_v: params.b2.clone().inner().zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    impl BurnAdapterAdamWState {
        fn zeros_like(params: &BurnAdapterParams) -> Self {
            Self {
                step: 0,
                w1_down_m: params.w1_down.clone().inner().zeros_like(),
                w1_down_v: params.w1_down.clone().inner().zeros_like(),
                w1_up_m: params.w1_up.clone().inner().zeros_like(),
                w1_up_v: params.w1_up.clone().inner().zeros_like(),
                w2_down_m: params.w2_down.clone().inner().zeros_like(),
                w2_down_v: params.w2_down.clone().inner().zeros_like(),
                w2_up_m: params.w2_up.clone().inner().zeros_like(),
                w2_up_v: params.w2_up.clone().inner().zeros_like(),
                b1_delta_m: params.b1_delta.clone().inner().zeros_like(),
                b1_delta_v: params.b1_delta.clone().inner().zeros_like(),
                b2_delta_m: params.b2_delta.clone().inner().zeros_like(),
                b2_delta_v: params.b2_delta.clone().inner().zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    impl BurnE2eGeneratorAdamWState {
        fn new(params: &BurnE2eGeneratorParams) -> Self {
            Self {
                step: 0,
            token_w_m: params.token_w.clone().inner().zeros_like(),
            token_w_v: params.token_w.clone().inner().zeros_like(),
            token_b_m: params.token_b.clone().inner().zeros_like(),
            token_b_v: params.token_b.clone().inner().zeros_like(),
            token_gate_w_m: params.token_gate_w.clone().inner().zeros_like(),
            token_gate_w_v: params.token_gate_w.clone().inner().zeros_like(),
            token_gate_b_m: params.token_gate_b.clone().inner().zeros_like(),
            token_gate_b_v: params.token_gate_b.clone().inner().zeros_like(),
            state_w_m: params.state_w.clone().inner().zeros_like(),
            state_w_v: params.state_w.clone().inner().zeros_like(),
                time_w_m: params.time_w.clone().inner().zeros_like(),
                time_w_v: params.time_w.clone().inner().zeros_like(),
                output_w_m: params.output_w.clone().inner().zeros_like(),
                output_w_v: params.output_w.clone().inner().zeros_like(),
                output_b_m: params.output_b.clone().inner().zeros_like(),
                output_b_v: params.output_b.clone().inner().zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }
    }

    impl BurnE2eGeneratorParams {
        fn seeded(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let first = examples.first().ok_or_else(|| {
                AutomataError::InvalidArgument("HyperNPA e2e generator requires examples".into())
            })?;
            let embed_dims = first.embed_dims;
            let token_count = first.token_count;
            if embed_dims == 0 || token_count == 0 {
                return Err(AutomataError::InvalidArgument(
                    "condition token dimensions must be positive".to_string(),
                ));
            }
            if examples.iter().any(|example| {
                example.embed_dims != embed_dims
                    || example.token_count != token_count
                    || example.condition_features.len() != token_count * embed_dims
            }) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e examples must have homogeneous condition token shapes"
                        .to_string(),
                ));
            }
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            let hidden_dims = config.generator_hidden_dims.max(1);
            let token_attention_heads = config.token_attention_heads.max(1);
            let mut rng = StdRng::seed_from_u64(config.seed ^ 0xa11c_e2e0_7a5e);
            let token_w = tracked_tensor(
                seeded_values(
                    hidden_dims * embed_dims,
                    config.generator_init_scale / (embed_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [hidden_dims, embed_dims],
                device,
            );
            let token_b = tracked_tensor(vec![0.0; hidden_dims], [1, hidden_dims], device);
            let token_gate_w = tracked_tensor(
                seeded_values(
                    token_attention_heads * hidden_dims,
                    config.generator_init_scale / (hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [token_attention_heads, hidden_dims],
                device,
            );
            let token_gate_b = tracked_tensor(
                vec![0.0; token_attention_heads],
                [1, token_attention_heads],
                device,
            );
            let state_w = tracked_tensor(
                seeded_values(
                    hidden_dims * output_dims,
                    config.generator_init_scale / (output_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [hidden_dims, output_dims],
                device,
            );
            let time_w = tracked_tensor(
                seeded_values(hidden_dims, config.generator_init_scale, &mut rng),
                [hidden_dims, 1],
                device,
            );
            let output_w = tracked_tensor(
                seeded_values(
                    output_dims * hidden_dims,
                    config.generator_init_scale / (hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [output_dims, hidden_dims],
                device,
            );
            let output_b = tracked_tensor(vec![0.0; output_dims], [1, output_dims], device);
            Ok(Self {
                token_w,
                token_b,
                token_gate_w,
                token_gate_b,
                state_w,
                time_w,
                output_w,
                output_b,
                hidden_dims,
                token_attention_heads,
                output_dims,
                output_scale: config.generator_output_scale,
                sample_steps: config.generator_sample_steps.max(1),
            })
        }

        fn adapter_batch(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            config: BurnE2eRolloutTrainConfig,
        ) -> BurnAdapterBatch {
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let tokens = dims[1];
            let embed_dims = dims[2];
            let device = condition.device();
            let token_w = self
                .token_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, embed_dims, self.hidden_dims]);
            let token_b = self
                .token_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, tokens, self.hidden_dims]);
            let token_hidden = relu(condition.matmul(token_w) + token_b);
            let mean_pooled = token_hidden
                .clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .div_scalar(tokens.max(1) as f32);
            let heads = self.token_attention_heads.max(1);
            let gate_w = self
                .token_gate_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.hidden_dims, heads]);
            let gate_b = self
                .token_gate_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, tokens, heads]);
            let attention_weights = (token_hidden.clone().matmul(gate_w) + gate_b).tanh().exp();
            let attention_denominator = attention_weights
                .clone()
                .sum_dim(1)
                .add_scalar(EPSILON)
                .expand([batches, tokens, heads]);
            let attention_weights = attention_weights.div(attention_denominator);
            let attended = attention_weights
                .swap_dims(1, 2)
                .matmul(token_hidden)
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .div_scalar(heads as f32);
            let pooled = (mean_pooled + attended).div_scalar(2.0);
            let mut vector = Tensor::<BurnBackend, 2>::zeros([batches, self.output_dims], &device);
            for step in 0..self.sample_steps {
                let t = if self.sample_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.sample_steps - 1) as f32
                };
                let state_hidden = vector.clone().matmul(self.state_w.clone().transpose());
                let time_hidden = self
                    .time_w
                    .clone()
                    .transpose()
                    .mul_scalar(t)
                    .expand([batches, self.hidden_dims]);
                let hidden = relu(
                    pooled.clone()
                        + state_hidden
                        + time_hidden
                        + self.token_b.clone().expand([batches, self.hidden_dims]),
                );
                let velocity = hidden.matmul(self.output_w.clone().transpose())
                    + self.output_b.clone().expand([batches, self.output_dims]);
                vector = vector + velocity.div_scalar(self.sample_steps as f32);
            }
            let vector = vector.tanh().mul_scalar(self.output_scale);
            BurnAdapterBatch::from_parameter_vector(
                vector,
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            )
        }

        fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnE2eGeneratorAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.token_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_w.clone().inner().zeros_like()),
                self.token_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_b.clone().inner().zeros_like()),
                self.token_gate_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_gate_w.clone().inner().zeros_like()),
                self.token_gate_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.token_gate_b.clone().inner().zeros_like()),
                self.state_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.state_w.clone().inner().zeros_like()),
                self.time_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.time_w.clone().inner().zeros_like()),
                self.output_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.output_w.clone().inner().zeros_like()),
                self.output_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.output_b.clone().inner().zeros_like()),
            ];
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.token_w = track(apply_adamw_tensor(
                self.token_w.clone().inner(),
                tensors.remove(0),
                &mut state.token_w_m,
                &mut state.token_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.token_b = track(apply_adamw_tensor(
                self.token_b.clone().inner(),
                tensors.remove(0),
                &mut state.token_b_m,
                &mut state.token_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.token_gate_w = track(apply_adamw_tensor(
                self.token_gate_w.clone().inner(),
                tensors.remove(0),
                &mut state.token_gate_w_m,
                &mut state.token_gate_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.token_gate_b = track(apply_adamw_tensor(
                self.token_gate_b.clone().inner(),
                tensors.remove(0),
                &mut state.token_gate_b_m,
                &mut state.token_gate_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.state_w = track(apply_adamw_tensor(
                self.state_w.clone().inner(),
                tensors.remove(0),
                &mut state.state_w_m,
                &mut state.state_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.time_w = track(apply_adamw_tensor(
                self.time_w.clone().inner(),
                tensors.remove(0),
                &mut state.time_w_m,
                &mut state.time_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.output_w = track(apply_adamw_tensor(
                self.output_w.clone().inner(),
                tensors.remove(0),
                &mut state.output_w_m,
                &mut state.output_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.output_b = track(apply_adamw_tensor(
                self.output_b.clone().inner(),
                tensors.remove(0),
                &mut state.output_b_m,
                &mut state.output_b_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }

        fn to_json(&self) -> AutomataResult<serde_json::Value> {
            Ok(json!({
                "version": 1,
                "architecture": "token_attention_pool_rectified_flow_generated_lora",
                "backend": format!("{BACKEND}_e2e_rollout"),
                "device": DEVICE_LABEL,
                "hidden_dims": self.hidden_dims,
                "token_attention_heads": self.token_attention_heads,
                "output_dims": self.output_dims,
                "sample_steps": self.sample_steps,
                "output_scale": self.output_scale,
                "weights": {
                    "token_w": tensor_vec(self.token_w.clone().inner())?,
                    "token_b": tensor_vec(self.token_b.clone().inner())?,
                    "token_gate_w": tensor_vec(self.token_gate_w.clone().inner())?,
                    "token_gate_b": tensor_vec(self.token_gate_b.clone().inner())?,
                    "state_w": tensor_vec(self.state_w.clone().inner())?,
                    "time_w": tensor_vec(self.time_w.clone().inner())?,
                    "output_w": tensor_vec(self.output_w.clone().inner())?,
                    "output_b": tensor_vec(self.output_b.clone().inner())?,
                }
            }))
        }

        fn detached(&self) -> Self {
            Self {
                token_w: detach2(self.token_w.clone()),
                token_b: detach2(self.token_b.clone()),
                token_gate_w: detach2(self.token_gate_w.clone()),
                token_gate_b: detach2(self.token_gate_b.clone()),
                state_w: detach2(self.state_w.clone()),
                time_w: detach2(self.time_w.clone()),
                output_w: detach2(self.output_w.clone()),
                output_b: detach2(self.output_b.clone()),
                hidden_dims: self.hidden_dims,
                token_attention_heads: self.token_attention_heads,
                output_dims: self.output_dims,
                output_scale: self.output_scale,
                sample_steps: self.sample_steps,
            }
        }
    }

    fn next_adamw_bias_correction(step: &mut usize, cfg: AdamWConfig) -> AdamWBiasCorrection {
        *step = step.saturating_add(1);
        let step_i32 = (*step).min(i32::MAX as usize) as i32;
        AdamWBiasCorrection {
            beta1: 1.0 - cfg.beta1.powi(step_i32),
            beta2: 1.0 - cfg.beta2.powi(step_i32),
        }
    }

    impl BurnBaseParams {
        fn from_model(model: &NpaModel, device: &BurnDevice) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                w1: tracked_tensor(
                    model.weights.w1.clone(),
                    [config.hidden_dims, config.perception_dims()],
                    device,
                ),
                b1: tracked_tensor(model.weights.b1.clone(), [1, config.hidden_dims], device),
                w2: tracked_tensor(
                    model.weights.w2.clone(),
                    [config.update_dims(), config.hidden_dims],
                    device,
                ),
                b2: tracked_tensor(model.weights.b2.clone(), [1, config.update_dims()], device),
            })
        }

        fn forward_adapter(
            &self,
            features: Tensor2,
            adapter: &BurnAdapterParams,
            _config: DirectBasisTrainConfig,
        ) -> Tensor2 {
            let rows = features.shape().dims::<2>()[0];
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone()
                + adapter
                    .w1_up
                    .clone()
                    .matmul(adapter.w1_down.clone())
                    .mul_scalar(scale);
            let w2 = self.w2.clone()
                + adapter
                    .w2_up
                    .clone()
                    .matmul(adapter.w2_down.clone())
                    .mul_scalar(scale);
            let b1 = self.b1.clone() + adapter.b1_delta.clone();
            let b2 = self.b2.clone() + adapter.b2_delta.clone();
            let hidden_dims = b1.shape().dims::<2>()[1];
            let output_dims = b2.shape().dims::<2>()[1];
            relu(features.matmul(w1.transpose()) + b1.expand([rows, hidden_dims]))
                .matmul(w2.transpose())
                + b2.expand([rows, output_dims])
        }

        fn forward_adapter_batch(&self, features: Tensor3, adapter: &BurnAdapterBatch) -> Tensor3 {
            let dims = features.shape().dims::<3>();
            let batches = dims[0];
            let rows = dims[1];
            let scale = adapter.alpha / adapter.rank.max(1) as f32;
            let w1 = self.w1.clone().unsqueeze_dim::<3>(0).expand([
                batches,
                self.w1.shape().dims::<2>()[0],
                self.w1.shape().dims::<2>()[1],
            ]) + adapter
                .w1_up
                .clone()
                .matmul(adapter.w1_down.clone())
                .mul_scalar(scale);
            let w2 = self.w2.clone().unsqueeze_dim::<3>(0).expand([
                batches,
                self.w2.shape().dims::<2>()[0],
                self.w2.shape().dims::<2>()[1],
            ]) + adapter
                .w2_up
                .clone()
                .matmul(adapter.w2_down.clone())
                .mul_scalar(scale);
            let hidden_dims = self.b1.shape().dims::<2>()[1];
            let output_dims = self.b2.shape().dims::<2>()[1];
            let b1 = self
                .b1
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, rows, hidden_dims])
                + adapter
                    .b1_delta
                    .clone()
                    .expand([batches, rows, hidden_dims]);
            let b2 = self
                .b2
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, rows, output_dims])
                + adapter
                    .b2_delta
                    .clone()
                    .expand([batches, rows, output_dims]);
            relu(features.matmul(w1.swap_dims(1, 2)) + b1).matmul(w2.swap_dims(1, 2)) + b2
        }

        fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnBaseAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.w1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1.clone().inner().zeros_like()),
                self.b1
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1.clone().inner().zeros_like()),
                self.w2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2.clone().inner().zeros_like()),
                self.b2
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2.clone().inner().zeros_like()),
            ];
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1 = track(apply_adamw_tensor(
                self.w1.clone().inner(),
                tensors.remove(0),
                &mut state.w1_m,
                &mut state.w1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1 = track(apply_adamw_tensor(
                self.b1.clone().inner(),
                tensors.remove(0),
                &mut state.b1_m,
                &mut state.b1_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2 = track(apply_adamw_tensor(
                self.w2.clone().inner(),
                tensors.remove(0),
                &mut state.w2_m,
                &mut state.w2_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b2 = track(apply_adamw_tensor(
                self.b2.clone().inner(),
                tensors.remove(0),
                &mut state.b2_m,
                &mut state.b2_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }

        fn write_to_model(&self, model: &mut NpaModel) -> AutomataResult<()> {
            model.weights = NpaWeights {
                w1: tensor_vec(self.w1.clone().inner())?,
                b1: tensor_vec(self.b1.clone().inner())?,
                w2: tensor_vec(self.w2.clone().inner())?,
                b2: tensor_vec(self.b2.clone().inner())?,
            };
            model.validate()
        }

        fn detached(&self) -> Self {
            Self {
                w1: detach2(self.w1.clone()),
                b1: detach2(self.b1.clone()),
                w2: detach2(self.w2.clone()),
                b2: detach2(self.b2.clone()),
            }
        }
    }

    impl BurnBaseBatch {
        fn from_params(params: &[BurnBaseParams]) -> Self {
            Self {
                w1: stack_base_tensor(params, |param| &param.w1),
                b1: stack_base_tensor(params, |param| &param.b1),
                w2: stack_base_tensor(params, |param| &param.w2),
                b2: stack_base_tensor(params, |param| &param.b2),
            }
        }

        fn forward(&self, features: Tensor3) -> Tensor3 {
            let dims = features.shape().dims::<3>();
            let batches = dims[0];
            let rows = dims[1];
            let hidden_dims = self.b1.shape().dims::<3>()[2];
            let output_dims = self.b2.shape().dims::<3>()[2];
            let b1 = self.b1.clone().expand([batches, rows, hidden_dims]);
            let b2 = self.b2.clone().expand([batches, rows, output_dims]);
            relu(features.matmul(self.w1.clone().swap_dims(1, 2)) + b1)
                .matmul(self.w2.clone().swap_dims(1, 2))
                + b2
        }
    }

    impl BurnAdapterParams {
        fn from_adapter(
            adapter: &NpaLowRankAdapter,
            model: &NpaModel,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let config = &model.config;
            Ok(Self {
                rank: adapter.rank,
                alpha: adapter.alpha,
                w1_down: tracked_tensor(
                    adapter.w1_down.clone(),
                    [adapter.rank, config.perception_dims()],
                    device,
                ),
                w1_up: tracked_tensor(
                    adapter.w1_up.clone(),
                    [config.hidden_dims, adapter.rank],
                    device,
                ),
                w2_down: tracked_tensor(
                    adapter.w2_down.clone(),
                    [adapter.rank, config.hidden_dims],
                    device,
                ),
                w2_up: tracked_tensor(
                    adapter.w2_up.clone(),
                    [config.update_dims(), adapter.rank],
                    device,
                ),
                b1_delta: tracked_tensor(adapter.b1_delta.clone(), [1, config.hidden_dims], device),
                b2_delta: tracked_tensor(
                    adapter.b2_delta.clone(),
                    [1, config.update_dims()],
                    device,
                ),
            })
        }

        fn to_adapter(&self) -> AutomataResult<NpaLowRankAdapter> {
            Ok(NpaLowRankAdapter {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: tensor_vec(self.w1_down.clone().inner())?,
                w1_up: tensor_vec(self.w1_up.clone().inner())?,
                w2_down: tensor_vec(self.w2_down.clone().inner())?,
                w2_up: tensor_vec(self.w2_up.clone().inner())?,
                b1_delta: tensor_vec(self.b1_delta.clone().inner())?,
                b2_delta: tensor_vec(self.b2_delta.clone().inner())?,
                b1_delta_correction: Vec::new(),
                b2_delta_correction: Vec::new(),
            })
        }

        fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter has parameters").div_scalar(6.0)
        }

        fn apply_adamw(
            &mut self,
            grads: &mut <BurnBackend as burn::tensor::backend::AutodiffBackend>::Gradients,
            state: &mut BurnAdapterAdamWState,
            cfg: AdamWConfig,
            normalize: bool,
            gradient_scale: f32,
            collect_metrics: bool,
        ) -> AutomataResult<(f32, f32)> {
            let mut tensors = vec![
                self.w1_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_down.clone().inner().zeros_like()),
                self.w1_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w1_up.clone().inner().zeros_like()),
                self.w2_down
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_down.clone().inner().zeros_like()),
                self.w2_up
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.w2_up.clone().inner().zeros_like()),
                self.b1_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b1_delta.clone().inner().zeros_like()),
                self.b2_delta
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.b2_delta.clone().inner().zeros_like()),
            ];
            if gradient_scale != 1.0 {
                for tensor in &mut tensors {
                    *tensor = tensor.clone().mul_scalar(gradient_scale);
                }
            }
            let (norm, scale, scale_tensor) =
                prepare_grad_group(&mut tensors, cfg.grad_clip_norm, normalize, collect_metrics)?;
            let bias = state.next_bias_correction(cfg);
            self.w1_down = track(apply_adamw_tensor(
                self.w1_down.clone().inner(),
                tensors.remove(0),
                &mut state.w1_down_m,
                &mut state.w1_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w1_up = track(apply_adamw_tensor(
                self.w1_up.clone().inner(),
                tensors.remove(0),
                &mut state.w1_up_m,
                &mut state.w1_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_down = track(apply_adamw_tensor(
                self.w2_down.clone().inner(),
                tensors.remove(0),
                &mut state.w2_down_m,
                &mut state.w2_down_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.w2_up = track(apply_adamw_tensor(
                self.w2_up.clone().inner(),
                tensors.remove(0),
                &mut state.w2_up_m,
                &mut state.w2_up_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b1_delta = track(apply_adamw_tensor(
                self.b1_delta.clone().inner(),
                tensors.remove(0),
                &mut state.b1_delta_m,
                &mut state.b1_delta_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.b2_delta = track(apply_adamw_tensor(
                self.b2_delta.clone().inner(),
                tensors.remove(0),
                &mut state.b2_delta_m,
                &mut state.b2_delta_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }
    }

    impl BurnAdapterBatch {
        fn from_indices(adapters: &[BurnAdapterParams], indices: &[usize]) -> Self {
            let first = &adapters[indices[0]];
            Self {
                rank: first.rank,
                alpha: first.alpha,
                w1_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_down),
                w1_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w1_up),
                w2_down: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_down),
                w2_up: stack_adapter_tensor(adapters, indices, |adapter| &adapter.w2_up),
                b1_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b1_delta),
                b2_delta: stack_adapter_tensor(adapters, indices, |adapter| &adapter.b2_delta),
            }
        }

        fn from_parameter_vector(
            vector: Tensor2,
            config: &NpaConfig,
            rank: usize,
            alpha: f32,
        ) -> Self {
            let batches = vector.shape().dims::<2>()[0];
            let input_dims = config.perception_dims();
            let hidden_dims = config.hidden_dims;
            let output_dims = config.update_dims();
            let mut offset = 0usize;
            let mut take = |len: usize| {
                let out = vector.clone().narrow(1, offset, len);
                offset += len;
                out
            };
            Self {
                rank,
                alpha,
                w1_down: take(rank * input_dims).reshape([batches, rank, input_dims]),
                w1_up: take(hidden_dims * rank).reshape([batches, hidden_dims, rank]),
                w2_down: take(rank * hidden_dims).reshape([batches, rank, hidden_dims]),
                w2_up: take(output_dims * rank).reshape([batches, output_dims, rank]),
                b1_delta: take(hidden_dims).reshape([batches, 1, hidden_dims]),
                b2_delta: take(output_dims).reshape([batches, 1, output_dims]),
            }
        }

        fn l2_loss(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let value = tensor.clone().mul(tensor).mean();
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter batch has parameters").div_scalar(6.0)
        }

        fn l2_loss_vector(&self) -> Tensor1 {
            let terms = vec![
                self.w1_down.clone(),
                self.w1_up.clone(),
                self.w2_down.clone(),
                self.w2_up.clone(),
                self.b1_delta.clone(),
                self.b2_delta.clone(),
            ];
            let mut total = None::<Tensor1>;
            for tensor in terms {
                let dims = tensor.shape().dims::<3>();
                let value = tensor
                    .clone()
                    .mul(tensor)
                    .reshape([dims[0], dims[1] * dims[2]])
                    .mean_dim(1)
                    .squeeze_dim::<1>(1);
                total = Some(match total {
                    Some(total) => total + value,
                    None => value,
                });
            }
            total.expect("adapter batch has parameters").div_scalar(6.0)
        }
    }

    fn stack_adapter_tensor(
        adapters: &[BurnAdapterParams],
        indices: &[usize],
        select: impl Fn(&BurnAdapterParams) -> &Tensor2,
    ) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| select(&adapters[*idx]).clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_base_tensor(
        params: &[BurnBaseParams],
        select: impl Fn(&BurnBaseParams) -> &Tensor2,
    ) -> Tensor3 {
        Tensor::cat(
            params
                .iter()
                .map(|param| select(param).clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn burn_targets(
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let pixel_xy = tensor(
            pixel_xy_values(config.loss_config.image_size),
            [pixels, 2],
            device,
        );
        examples
            .iter()
            .map(|example| {
                let render = render_target_2d_splat(&example.target, config.loss_config)?;
                let foreground = target_2d_foreground_mask(&example.target, config.loss_config)?;
                let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
                let target_mean = example.target.mean_position();
                let target_positions = example
                    .target
                    .positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                Ok(BurnTargetExample {
                    target_rgb: tensor(render.rgb, [pixels, 3], device),
                    target_density: tensor(render.density, [pixels, 1], device),
                    target_foreground: tensor(foreground, [pixels, 1], device),
                    target_foreground_scale: foreground_scale,
                    target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], device),
                    target_positions: tensor(
                        target_positions,
                        [example.target.positions.len(), 2],
                        device,
                    ),
                    pixel_xy: pixel_xy.clone(),
                    pixel_size: example.target.pixel_size,
                    target_points: example.target.point_count(),
                    particle_count: example.source.particles.unwrap_or(config.rollout_particles),
                    update_prob: example.source.update_prob.unwrap_or(config.update_prob),
                    seed_scale: example.source.seed_scale.unwrap_or(config.seed_scale),
                })
            })
            .collect()
    }

    fn burn_e2e_targets(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        burn_e2e_targets_with_runtime(examples, config, device, None, None)
    }

    fn burn_e2e_targets_with_runtime(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
        particle_count: Option<usize>,
        update_prob: Option<f32>,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let direct_config = direct_config_view(config);
        let pixels = direct_config.loss_config.image_size * direct_config.loss_config.image_size;
        let pixel_xy = tensor(
            pixel_xy_values(direct_config.loss_config.image_size),
            [pixels, 2],
            device,
        );
        examples
            .iter()
            .map(|example| {
                let render = render_target_2d_splat(&example.target, direct_config.loss_config)?;
                let foreground = target_2d_foreground_mask(&example.target, direct_config.loss_config)?;
                let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
                let target_mean = example.target.mean_position();
                let target_positions = example
                    .target
                    .positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                Ok(BurnTargetExample {
                    target_rgb: tensor(render.rgb, [pixels, 3], device),
                    target_density: tensor(render.density, [pixels, 1], device),
                    target_foreground: tensor(foreground, [pixels, 1], device),
                    target_foreground_scale: foreground_scale,
                    target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], device),
                    target_positions: tensor(
                        target_positions,
                        [example.target.positions.len(), 2],
                        device,
                    ),
                    pixel_xy: pixel_xy.clone(),
                    pixel_size: example.target.pixel_size,
                    target_points: example.target.point_count(),
                    particle_count: particle_count.unwrap_or(example.particle_count).max(1),
                    update_prob: update_prob.unwrap_or(example.update_prob),
                    seed_scale: example.seed_scale,
                })
            })
            .collect()
    }

    fn direct_config_view(config: BurnE2eRolloutTrainConfig) -> DirectBasisTrainConfig {
        DirectBasisTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            example_batch_size: config.example_batch_size,
            tbptt_chunk_steps: config.tbptt_chunk_steps,
            loss_on_final_chunk_only: false,
            use_particle_pool: false,
            pool_size: 0,
            inject_seed_interval: 0,
            brush_size: 0.0,
            stopgrad_pos: config.stopgrad_pos,
            stopgrad_state: config.stopgrad_state,
            rollout_particles: config.rollout_particles,
            rollout_step_min: config.rollout_steps,
            rollout_steps: config.rollout_steps,
            update_prob: config.update_prob,
            seed: config.seed,
            seed_scale: config.seed_scale,
            seed_mode: config.seed_mode,
            grid_eps: config.grid_eps,
            motion_scale: config.motion_scale,
            loss_config: config.loss_config,
            per_parameter_grad_normalization: config.per_parameter_grad_normalization,
            base_sgd: SgdConfig {
                learning_rate: config.base_optimizer.learning_rate,
                weight_decay: config.base_optimizer.weight_decay,
                grad_clip_norm: config.base_optimizer.grad_clip_norm,
            },
            adapter_sgd: SgdConfig {
                learning_rate: config.generator_optimizer.learning_rate,
                weight_decay: config.generator_optimizer.weight_decay,
                grad_clip_norm: config.generator_optimizer.grad_clip_norm,
            },
            adapter_l2_weight: 0.0,
            update_base: config.shared_base_trainable,
            eval_examples: 0,
            eval_interval: 0,
            eval_batch_size: 1,
            eval_seed: config.seed,
            system_memory_budget_gb: config.system_memory_budget_gb,
            gpu_memory_budget_gb: config.gpu_memory_budget_gb,
            max_dense_train_particles: config.max_dense_train_particles,
            max_dense_chunk_floats: config.max_dense_chunk_floats,
            max_splat_chunk_floats: config.max_splat_chunk_floats,
        }
    }

    fn validation_direct_config(config: BurnE2eRolloutTrainConfig) -> DirectBasisTrainConfig {
        let mut direct = direct_config_view(config);
        direct.rollout_particles = config.validation_particles.max(1);
        direct.rollout_steps = config.validation_steps.max(1);
        direct.update_prob = config.validation_update_prob;
        direct.seed = config.validation_seed;
        direct.eval_batch_size = if direct.rollout_particles > config.max_dense_train_particles {
            1
        } else {
            config.example_batch_size.max(1)
        };
        direct
    }

    impl BurnE2eConditionCache {
        fn from_examples_drain(
            examples: &mut [BurnE2eRolloutExample],
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            if examples.is_empty() {
                return Ok(Self {
                    values: BurnE2eConditionValues::HostRows(Vec::new()),
                    examples: 0,
                    token_count: 0,
                    embed_dims: 0,
                    device: device.clone(),
                });
            }
            let first = &examples[0];
            let token_count = first.token_count;
            let embed_dims = first.embed_dims;
            if token_count == 0 || embed_dims == 0 {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache requires non-empty token shapes".to_string(),
                ));
            }
            let row_len = token_count * embed_dims;
            let feature_bytes = examples
                .len()
                .saturating_mul(row_len)
                .saturating_mul(std::mem::size_of::<f32>());
            let use_device_cache = feature_bytes <= DEVICE_CONDITION_CACHE_MAX_BYTES;
            let mut flat_values = use_device_cache
                .then(|| Vec::with_capacity(examples.len().saturating_mul(row_len)));
            let mut rows = (!use_device_cache).then(|| Vec::with_capacity(examples.len()));
            for example in &mut *examples {
                let feature_len = example.condition_features.len();
                if example.token_count != token_count
                    || example.embed_dims != embed_dims
                    || feature_len != row_len
                {
                    return Err(AutomataError::InvalidArgument(
                        "condition token shape mismatch in HyperNPA e2e cache".to_string(),
                    ));
                }
                let condition_features = std::mem::take(&mut example.condition_features);
                if let Some(flat_values) = flat_values.as_mut() {
                    flat_values.extend(condition_features);
                } else if let Some(rows) = rows.as_mut() {
                    rows.push(condition_features);
                }
            }
            let values = if let Some(values) = flat_values {
                BurnE2eConditionValues::Device(tensor3(
                    values,
                    [examples.len(), token_count, embed_dims],
                    device,
                ))
            } else {
                BurnE2eConditionValues::HostRows(rows.unwrap_or_default())
            };
            Ok(Self {
                values,
                examples: examples.len(),
                token_count,
                embed_dims,
                device: device.clone(),
            })
        }

        fn select(&self, indices: &[usize]) -> AutomataResult<Tensor3> {
            if indices.is_empty() || self.token_count == 0 || self.embed_dims == 0 {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache select requires non-empty indices".to_string(),
                ));
            }
            if indices.iter().any(|idx| *idx >= self.examples) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache index out of bounds".to_string(),
                ));
            }
            match &self.values {
                BurnE2eConditionValues::Device(values) => {
                    let index_values = indices.iter().map(|idx| *idx as i64).collect::<Vec<_>>();
                    let index_tensor: Tensor1Int =
                        Tensor::from_data(TensorData::new(index_values, [indices.len()]), &self.device);
                    Ok(values.clone().select(0, index_tensor))
                }
                BurnE2eConditionValues::HostRows(rows) => {
                    let row_len = self.token_count * self.embed_dims;
                    let mut selected = Vec::with_capacity(indices.len() * row_len);
                    for &idx in indices {
                        selected.extend_from_slice(&rows[idx]);
                    }
                    Ok(tensor3(
                        selected,
                        [indices.len(), self.token_count, self.embed_dims],
                        &self.device,
                    ))
                }
            }
        }

        fn feature_bytes(&self) -> usize {
            self.examples
                .saturating_mul(self.token_count)
                .saturating_mul(self.embed_dims)
                .saturating_mul(std::mem::size_of::<f32>())
        }

        fn storage_label(&self) -> &'static str {
            match &self.values {
                BurnE2eConditionValues::Device(_) => "device-resident",
                BurnE2eConditionValues::HostRows(_) => "host-row-streamed",
            }
        }

        fn is_device_resident(&self) -> bool {
            matches!(self.values, BurnE2eConditionValues::Device(_))
        }
    }

    fn seed_batch_tensors(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        let mut positions = Vec::with_capacity(indices.len() * particle_count * 2);
        let mut states = Vec::with_capacity(indices.len() * particle_count * 16);
        for &idx in indices {
            let (example_positions, example_states) = seed_particles_scaled(
                1,
                particle_count,
                16,
                2,
                step_seed.wrapping_add(idx as u64),
                config.seed_mode,
                targets[idx].seed_scale,
            );
            positions.extend(
                example_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]]),
            );
            states.extend(example_states);
        }
        (
            tensor3(positions, [indices.len(), particle_count, 2], device),
            tensor3(states, [indices.len(), particle_count, 16], device),
        )
    }

    fn batch_masks(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rng: &mut StdRng,
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for &idx in indices {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                rng,
            ));
        }
        values
    }

    fn batch_masks_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rngs: &mut [StdRng],
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for (local, &idx) in indices.iter().enumerate() {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                &mut rngs[local],
            ));
        }
        values
    }

    fn stack_target_rgb(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_rgb.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_target_density(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_density.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_target_foreground(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_foreground.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_target_foreground_scales(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].target_foreground_scale)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn stack_target_mean(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_mean.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    fn stack_pixel_sizes(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].pixel_size)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn stack_target_point_counts(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].target_points as f32)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn pixel_xy_values(image_size: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(image_size * image_size * 2);
        for y in 0..image_size {
            for x in 0..image_size {
                values.push(x as f32);
                values.push(y as f32);
            }
        }
        values
    }

    fn adapter_cache_metrics(
        base: &NpaModel,
        params: &BurnBaseParams,
        train_adapters: &[BurnAdapterParams],
        holdout_adapters: &[BurnAdapterParams],
        train_targets: &[BurnTargetExample],
        holdout_targets: &[BurnTargetExample],
    ) -> AutomataResult<serde_json::Value> {
        let rank = train_adapters
            .first()
            .or_else(|| holdout_adapters.first())
            .map_or(0, |adapter| adapter.rank);
        let parameters_per_adapter = if rank == 0 {
            0
        } else {
            NpaLowRankAdapter::parameter_count_for_config(&base.config, rank)
        };
        let total_adapters = train_adapters.len() + holdout_adapters.len();
        let total_adapter_parameters = parameters_per_adapter * total_adapters;
        let train_target_points = train_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let holdout_target_points = holdout_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let train_render_pixels = train_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        let holdout_render_pixels = holdout_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        Ok(json!({
            "representation": "resident_gpu_tensor_set_per_sample",
            "readback_policy": "report_interval_scalars_and_end_of_phase_artifacts_only",
            "non_report_step_loss_readbacks": false,
            "adapter_tensors_per_sample": 6,
            "rank": rank,
            "parameters_per_adapter": parameters_per_adapter,
            "train_adapters": train_adapters.len(),
            "holdout_adapters": holdout_adapters.len(),
            "total_adapters": total_adapters,
            "total_adapter_parameters": total_adapter_parameters,
            "estimated_adapter_weight_bytes_f32": total_adapter_parameters * std::mem::size_of::<f32>(),
            "estimated_adapter_tensor_count": total_adapters * 6,
            "train_target_points": train_target_points,
            "holdout_target_points": holdout_target_points,
            "train_render_pixels": train_render_pixels,
            "holdout_render_pixels": holdout_render_pixels,
            "estimated_target_render_cache_bytes_f32": (train_render_pixels + holdout_render_pixels) * 4 * std::mem::size_of::<f32>(),
            "base_norms": base_norm_metrics(params)?,
            "train_adapter_norms": adapter_norm_metrics(train_adapters)?,
            "holdout_adapter_norms": adapter_norm_metrics(holdout_adapters)?,
        }))
    }

    fn base_norm_metrics(params: &BurnBaseParams) -> AutomataResult<serde_json::Value> {
        let w1 = tensor_l2_norm(&params.w1.clone().inner())?;
        let b1 = tensor_l2_norm(&params.b1.clone().inner())?;
        let w2 = tensor_l2_norm(&params.w2.clone().inner())?;
        let b2 = tensor_l2_norm(&params.b2.clone().inner())?;
        Ok(json!({
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "total": finite_scalar("Burn direct base norm", (w1 * w1 + b1 * b1 + w2 * w2 + b2 * b2).sqrt())?,
        }))
    }

    fn adapter_norm_metrics(adapters: &[BurnAdapterParams]) -> AutomataResult<serde_json::Value> {
        if adapters.is_empty() {
            return Ok(json!({
                "examples": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }));
        }
        let mut sum = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = 0.0_f32;
        for adapter in adapters {
            let norm = adapter_l2_norm(adapter)?;
            sum += norm;
            min = min.min(norm);
            max = max.max(norm);
        }
        Ok(json!({
            "examples": adapters.len(),
            "mean": finite_scalar("Burn direct mean adapter norm", sum / adapters.len() as f32)?,
            "min": finite_scalar("Burn direct min adapter norm", min)?,
            "max": finite_scalar("Burn direct max adapter norm", max)?,
        }))
    }

    fn adapter_l2_norm(adapter: &BurnAdapterParams) -> AutomataResult<f32> {
        let tensors = [
            adapter.w1_down.clone().inner(),
            adapter.w1_up.clone().inner(),
            adapter.w2_down.clone().inner(),
            adapter.w2_up.clone().inner(),
            adapter.b1_delta.clone().inner(),
            adapter.b2_delta.clone().inner(),
        ];
        finite_scalar(
            "Burn direct adapter norm",
            group_norm_tensor(&tensors).into_scalar(),
        )
    }

    fn mean_updates_per_sample(steps: usize, batch_size: usize, examples: usize) -> f32 {
        if examples == 0 {
            return 0.0;
        }
        steps as f32 * batch_size.min(examples).max(1) as f32 / examples as f32
    }

    fn normalized_batch_size(requested: usize, examples: usize) -> usize {
        requested.max(1).min(examples.max(1))
    }

    fn sample_indices(examples: usize, batch_size: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut indices = (0..examples).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices.truncate(batch_size.min(examples));
        indices
    }

    fn seeded_values(len: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
        let scale = scale.abs().max(f32::MIN_POSITIVE);
        (0..len)
            .map(|_| rng.random_range(-scale..scale))
            .collect::<Vec<_>>()
    }

    fn seed_tensors(
        particle_count: usize,
        config: DirectBasisTrainConfig,
        seed_scale: f32,
        seed: u64,
        device: &BurnDevice,
    ) -> (Tensor2, Tensor2) {
        let (positions, states) =
            seed_particles_scaled(1, particle_count, 16, 2, seed, config.seed_mode, seed_scale);
        let flat_positions = positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        (
            tensor(flat_positions, [particle_count, 2], device),
            tensor(states, [particle_count, 16], device),
        )
    }

    fn write_adapters(
        examples: &mut [DirectBasisExample],
        adapters: &[BurnAdapterParams],
    ) -> AutomataResult<()> {
        for (example, adapter) in examples.iter_mut().zip(adapters) {
            example.adapter = adapter.to_adapter()?;
        }
        Ok(())
    }

    impl BurnHostParticlePool {
        fn new(
            pool_size: usize,
            particle_count: usize,
            state_dims: usize,
            seed_scale: f32,
            config: DirectBasisTrainConfig,
        ) -> Self {
            let (positions, states) = seed_particles_scaled(
                pool_size,
                particle_count,
                state_dims,
                2,
                config.seed,
                config.seed_mode,
                seed_scale,
            );
            Self {
                positions: positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect(),
                states,
                pool_size,
                particle_count,
                state_dims,
            }
        }

        fn sample_batch(
            &self,
            rng: &mut StdRng,
            batch_size: usize,
            replace_seed: bool,
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> BurnPoolBatch {
            let mut pool_indices = (0..self.pool_size).collect::<Vec<_>>();
            pool_indices.shuffle(rng);
            pool_indices.truncate(batch_size.min(self.pool_size));

            let mut positions = Vec::with_capacity(pool_indices.len() * self.particle_count * 2);
            let mut states =
                Vec::with_capacity(pool_indices.len() * self.particle_count * self.state_dims);
            for pool_index in &pool_indices {
                let position_start = pool_index * self.particle_count * 2;
                let position_end = position_start + self.particle_count * 2;
                positions.extend_from_slice(&self.positions[position_start..position_end]);
                let state_start = pool_index * self.particle_count * self.state_dims;
                let state_end = state_start + self.particle_count * self.state_dims;
                states.extend_from_slice(&self.states[state_start..state_end]);
            }

            if replace_seed && !pool_indices.is_empty() {
                let seed = config.seed ^ rng.random::<u64>();
                let (seed_positions, seed_states) = seed_particles_scaled(
                    1,
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                for (particle, position) in seed_positions.iter().enumerate() {
                    let base = particle * 2;
                    positions[base] = position[0];
                    positions[base + 1] = position[1];
                }
                states[..self.particle_count * self.state_dims]
                    .copy_from_slice(&seed_states[..self.particle_count * self.state_dims]);
            }

            if config.brush_size > 0.0 {
                self.apply_brush_damage(&positions, &mut states, pool_indices.len(), config.brush_size, rng);
            }

            BurnPoolBatch {
                pool_indices,
                x: tensor3(
                    positions,
                    [batch_size.min(self.pool_size), self.particle_count, 2],
                    device,
                ),
                s: tensor3(
                    states,
                    [
                        batch_size.min(self.pool_size),
                        self.particle_count,
                        self.state_dims,
                    ],
                    device,
                ),
            }
        }

        fn apply_brush_damage(
            &self,
            positions: &[f32],
            states: &mut [f32],
            batch_size: usize,
            brush_size: f32,
            rng: &mut StdRng,
        ) {
            for batch in 0..batch_size {
                let center_idx = batch * self.particle_count + rng.random_range(0..self.particle_count);
                let center_base = center_idx * 2;
                let center_x = positions[center_base];
                let center_y = positions[center_base + 1];
                let brush2 = brush_size * brush_size;
                for particle in 0..self.particle_count {
                    let row = batch * self.particle_count + particle;
                    let position_base = row * 2;
                    let dx = positions[position_base] - center_x;
                    let dy = positions[position_base + 1] - center_y;
                    if dx * dx + dy * dy < brush2 {
                        let state_base = row * self.state_dims;
                        states[state_base..state_base + self.state_dims].fill(0.0);
                    }
                }
            }
        }

        fn update_batch(
            &mut self,
            pool_indices: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            let positions = tensor3_vec(x.inner())?;
            let states = tensor3_vec(s.inner())?;
            for (batch, pool_index) in pool_indices.iter().copied().enumerate() {
                let position_dst = pool_index * self.particle_count * 2;
                let position_src = batch * self.particle_count * 2;
                self.positions[position_dst..position_dst + self.particle_count * 2]
                    .copy_from_slice(&positions[position_src..position_src + self.particle_count * 2]);

                let state_dst = pool_index * self.particle_count * self.state_dims;
                let state_src = batch * self.particle_count * self.state_dims;
                self.states[state_dst..state_dst + self.particle_count * self.state_dims]
                    .copy_from_slice(&states[state_src..state_src + self.particle_count * self.state_dims]);
            }
            Ok(())
        }
    }

    struct PhaseBatchSampler {
        len: usize,
        batch_size: usize,
        order: Vec<usize>,
        cursor: usize,
    }

    impl PhaseBatchSampler {
        fn new(len: usize, requested: usize, rng: &mut StdRng) -> Self {
            let batch_size = if requested == 0 {
                len
            } else {
                requested.min(len)
            };
            let mut order = (0..len).collect::<Vec<_>>();
            order.shuffle(rng);
            Self {
                len,
                batch_size,
                order,
                cursor: 0,
            }
        }

        fn next_batch(&mut self, rng: &mut StdRng) -> Vec<usize> {
            if self.len == 0 || self.batch_size == 0 {
                return Vec::new();
            }
            if self.batch_size >= self.len {
                let mut indices = (0..self.len).collect::<Vec<_>>();
                indices.shuffle(rng);
                return indices;
            }

            let mut indices = Vec::with_capacity(self.batch_size);
            while indices.len() < self.batch_size {
                if self.cursor >= self.order.len() {
                    self.reshuffle_excluding(rng, &indices);
                }
                let idx = self.order[self.cursor];
                self.cursor += 1;
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            indices
        }

        fn reshuffle_excluding(&mut self, rng: &mut StdRng, exclude: &[usize]) {
            self.order = (0..self.len)
                .filter(|idx| !exclude.contains(idx))
                .collect::<Vec<_>>();
            self.order.shuffle(rng);
            self.cursor = 0;
        }
    }

    fn sample_update_stats(counts: &[usize]) -> SampleUpdateStats {
        if counts.is_empty() {
            return SampleUpdateStats {
                examples: 0,
                total_updates: 0,
                min_updates: 0,
                max_updates: 0,
                mean_updates: 0.0,
                zero_update_examples: 0,
            };
        }
        let total_updates = counts.iter().sum::<usize>();
        SampleUpdateStats {
            examples: counts.len(),
            total_updates,
            min_updates: counts.iter().copied().min().unwrap_or(0),
            max_updates: counts.iter().copied().max().unwrap_or(0),
            mean_updates: total_updates as f32 / counts.len() as f32,
            zero_update_examples: counts.iter().filter(|updates| **updates == 0).count(),
        }
    }

    fn loss_scalars(loss: &BurnLossTensors) -> AutomataResult<BurnLossScalars> {
        Ok(BurnLossScalars {
            total: finite_scalar(
                "Burn direct total loss",
                loss.total.clone().inner().into_scalar(),
            )?,
            splat: finite_scalar(
                "Burn direct splat loss",
                loss.splat.clone().inner().into_scalar(),
            )?,
            color: finite_scalar(
                "Burn direct color loss",
                loss.color.clone().inner().into_scalar(),
            )?,
            density: finite_scalar(
                "Burn direct density loss",
                loss.density.clone().inner().into_scalar(),
            )?,
        })
    }

    fn loss_vector_scalars(loss: BurnLossBatchTensors) -> AutomataResult<Vec<BurnLossScalars>> {
        let total = tensor1_vec(loss.total.inner())?;
        let splat = tensor1_vec(loss.splat.inner())?;
        let color = tensor1_vec(loss.color.inner())?;
        let density = tensor1_vec(loss.density.inner())?;
        if total.len() != splat.len() || total.len() != color.len() || total.len() != density.len()
        {
            return Err(AutomataError::InvalidArgument(
                "Burn direct vector loss readback length mismatch".to_string(),
            ));
        }
        total
            .into_iter()
            .zip(splat)
            .zip(color)
            .zip(density)
            .enumerate()
            .map(|(idx, (((total, splat), color), density))| {
                Ok(BurnLossScalars {
                    total: finite_scalar(&format!("Burn direct total loss[{idx}]"), total)?,
                    splat: finite_scalar(&format!("Burn direct splat loss[{idx}]"), splat)?,
                    color: finite_scalar(&format!("Burn direct color loss[{idx}]"), color)?,
                    density: finite_scalar(&format!("Burn direct density loss[{idx}]"), density)?,
                })
            })
            .collect()
    }

    fn prepare_grad_group(
        tensors: &mut [Tensor2Inner],
        clip_norm: f32,
        normalize: bool,
        collect_metrics: bool,
    ) -> AutomataResult<(f32, f32, Tensor1Inner)> {
        let original_norm_tensor = group_norm_tensor(tensors);
        let original_norm = if collect_metrics {
            finite_scalar(
                "Burn direct grad norm",
                original_norm_tensor.clone().into_scalar(),
            )?
        } else {
            0.0
        };
        if normalize {
            for tensor in tensors.iter_mut() {
                let dims = tensor.shape().dims::<2>();
                let norm = tensor_l2_norm_tensor(tensor).add_scalar(1.0e-8);
                *tensor = tensor.clone().div(norm.expand(dims));
            }
        }
        let clip_norm_source = if normalize {
            group_norm_tensor(tensors)
        } else {
            original_norm_tensor
        };
        let scale_tensor = if clip_norm > 0.0 {
            clip_norm_source
                .clone()
                .clamp_min(clip_norm)
                .recip()
                .mul_scalar(clip_norm)
        } else {
            clip_norm_source.zeros_like().add_scalar(1.0)
        };
        let scale = if collect_metrics {
            finite_scalar("Burn direct grad scale", scale_tensor.clone().into_scalar())?
        } else {
            1.0
        };
        Ok((original_norm, scale, scale_tensor))
    }

    fn group_norm_tensor(tensors: &[Tensor2Inner]) -> Tensor1Inner {
        let mut total = None::<Tensor1Inner>;
        for tensor in tensors {
            let value = tensor.clone().mul(tensor.clone()).sum();
            total = Some(match total {
                Some(total) => total + value,
                None => value,
            });
        }
        total.expect("gradient group has tensors").sqrt()
    }

    fn tensor_l2_norm_tensor(tensor: &Tensor2Inner) -> Tensor1Inner {
        tensor.clone().mul(tensor.clone()).sum().sqrt()
    }

    fn tensor_l2_norm(tensor: &Tensor2Inner) -> AutomataResult<f32> {
        finite_scalar(
            "Burn direct tensor norm",
            tensor_l2_norm_tensor(tensor).into_scalar(),
        )
    }

    fn adamw_from_sgd(cfg: SgdConfig) -> AdamWConfig {
        AdamWConfig {
            learning_rate: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
            grad_clip_norm: cfg.grad_clip_norm,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }

    fn apply_adamw_tensor(
        param: Tensor2Inner,
        grad: Tensor2Inner,
        moment: &mut Tensor2Inner,
        velocity: &mut Tensor2Inner,
        cfg: AdamWConfig,
        scale: Tensor1Inner,
        bias: AdamWBiasCorrection,
    ) -> Tensor2Inner {
        let dims = param.shape().dims::<2>();
        let grad = grad.mul(scale.expand(dims));
        let decayed = if cfg.weight_decay > 0.0 {
            param
                .clone()
                .mul_scalar(1.0 - cfg.learning_rate * cfg.weight_decay)
        } else {
            param.clone()
        };
        *moment = moment.clone().mul_scalar(cfg.beta1) + grad.clone().mul_scalar(1.0 - cfg.beta1);
        *velocity = velocity.clone().mul_scalar(cfg.beta2)
            + grad.clone().mul(grad).mul_scalar(1.0 - cfg.beta2);
        let normalized_step = moment
            .clone()
            .div_scalar(bias.beta1.max(f32::MIN_POSITIVE))
            .div(
                velocity
                    .clone()
                    .div_scalar(bias.beta2.max(f32::MIN_POSITIVE))
                    .sqrt()
                    .add_scalar(cfg.epsilon),
            );
        decayed - normalized_step.mul_scalar(cfg.learning_rate)
    }

    fn tracked_tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        tensor(values, shape, device).require_grad()
    }

    fn tensor(values: Vec<f32>, shape: [usize; 2], device: &BurnDevice) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
    }

    fn tensor3(values: Vec<f32>, shape: [usize; 3], device: &BurnDevice) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_data(TensorData::new(values, shape), device)
    }

    fn detach1(tensor: Tensor1) -> Tensor1 {
        Tensor::<BurnBackend, 1>::from_inner(tensor.inner())
    }

    fn detach2(tensor: Tensor2) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor.inner())
    }

    fn detach3(tensor: Tensor3) -> Tensor3 {
        Tensor::<BurnBackend, 3>::from_inner(tensor.inner())
    }

    fn target_2d_detached_color_gate2(density_term: Tensor2) -> Tensor2 {
        debug_assert_eq!(
            crate::TARGET_2D_COLOR_GATE_GRADIENT,
            crate::Target2dColorGateGradient::DetachedDensity
        );
        detach2(density_term.mul_scalar(-1.0).exp())
    }

    fn target_2d_detached_color_gate3(density_term: Tensor3) -> Tensor3 {
        debug_assert_eq!(
            crate::TARGET_2D_COLOR_GATE_GRADIENT,
            crate::Target2dColorGateGradient::DetachedDensity
        );
        detach3(density_term.mul_scalar(-1.0).exp())
    }

    fn track(tensor: Tensor2Inner) -> Tensor2 {
        Tensor::<BurnBackend, 2>::from_inner(tensor).require_grad()
    }

    fn tensor_vec(tensor: Tensor2Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    fn tensor3_vec(tensor: Tensor3Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    fn tensor1_vec(tensor: Tensor1Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    fn finite_scalar(name: &str, value: f32) -> AutomataResult<f32> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(AutomataError::InvalidArgument(format!(
                "{name} is not finite"
            )))
        }
    }

    fn check_process_memory_budget(
        label: &str,
        config: DirectBasisTrainConfig,
    ) -> Result<ProcessMemorySnapshot, Box<dyn std::error::Error>> {
        let budget_bytes = config
            .system_memory_budget_gb
            .map(memory_budget_gb_to_bytes);
        let snapshot = ProcessMemorySnapshot {
            label: label.to_string(),
            rss_bytes: current_process_rss_bytes(),
            budget_bytes,
        };
        if let (Some(rss_bytes), Some(budget_bytes)) = (snapshot.rss_bytes, snapshot.budget_bytes)
            && rss_bytes > budget_bytes
        {
            return Err(std::io::Error::other(format!(
                "Burn dense direct-basis memory budget exceeded at {label}: rss={:.2} GiB budget={:.2} GiB",
                bytes_to_gib(rss_bytes),
                bytes_to_gib(budget_bytes)
            ))
            .into());
        }
        Ok(snapshot)
    }

    fn current_process_rss_bytes() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(|line| {
            let rest = line.strip_prefix("VmRSS:")?;
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kb.saturating_mul(1024))
        })
    }

    fn check_gpu_memory_budget(
        label: &str,
        config: DirectBasisTrainConfig,
    ) -> Result<GpuMemorySnapshot, Box<dyn std::error::Error>> {
        let budget_bytes = config.gpu_memory_budget_gb.map(memory_budget_gb_to_bytes);
        let (used_bytes, total_bytes) = current_nvidia_gpu_memory_bytes();
        let snapshot = GpuMemorySnapshot {
            label: label.to_string(),
            used_bytes,
            total_bytes,
            budget_bytes,
        };
        if let (Some(used_bytes), Some(budget_bytes)) = (snapshot.used_bytes, snapshot.budget_bytes)
            && used_bytes > budget_bytes
        {
            return Err(std::io::Error::other(format!(
                "Burn dense direct-basis GPU memory budget exceeded at {label}: used={:.2} GiB budget={:.2} GiB",
                bytes_to_gib(used_bytes),
                bytes_to_gib(budget_bytes)
            ))
            .into());
        }
        Ok(snapshot)
    }

    fn current_nvidia_gpu_memory_bytes() -> (Option<u64>, Option<u64>) {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok();
        let Some(output) = output else {
            return (None, None);
        };
        if !output.status.success() {
            return (None, None);
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let Some(line) = text.lines().next() else {
            return (None, None);
        };
        let mut fields = line.split(',').map(str::trim);
        let used_mib = fields.next().and_then(|value| value.parse::<u64>().ok());
        let total_mib = fields.next().and_then(|value| value.parse::<u64>().ok());
        (
            used_mib.map(|mib| mib.saturating_mul(1024 * 1024)),
            total_mib.map(|mib| mib.saturating_mul(1024 * 1024)),
        )
    }

    fn memory_budget_gb_to_bytes(gb: f32) -> u64 {
        (gb as f64 * 1024.0 * 1024.0 * 1024.0).round() as u64
    }

    fn bytes_to_gib(bytes: u64) -> f64 {
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dense_perception_matches_reference_kernel_fixture() {
            let npa_config = NpaConfig::growing_2d();
            let grid = burn_automata_kernels::HashGridConfig::growing_2d();
            let (positions, states) = seed_particles_scaled(
                1,
                4,
                npa_config.state_dims,
                npa_config.spatial_dims,
                17,
                crate::ParticleSeed::UniformCircle,
                0.08,
            );
            let options = burn_automata_kernels::PerceptionOptions {
                state_grad: npa_config.state_grad,
                density_grad: npa_config.density_grad,
                eps0: npa_config.eps0,
                scale_equivariance: npa_config.scale_equivariant(),
                particle_density_equivariance: npa_config.particle_density_equivariant(),
                log_norm_grad: npa_config.log_norm_grad,
                log_norm_density_grad: npa_config.log_norm_density_grad,
                hybrid_state_gradient: true,
                position_features: npa_config.position_features,
            };
            let reference = burn_automata_kernels::perceive_with_options(
                &positions,
                &states,
                1,
                4,
                npa_config.state_dims,
                &grid,
                options,
            )
            .unwrap();
            let device = BurnDevice::default();
            let x = tensor(
                positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect(),
                [4, 2],
                &device,
            );
            let s = tensor(states, [4, npa_config.state_dims], &device);
            let config = DirectBasisTrainConfig {
                steps: 0,
                report_interval: 1,
                example_batch_size: 1,
                tbptt_chunk_steps: 1,
                loss_on_final_chunk_only: false,
                use_particle_pool: false,
                pool_size: 0,
                inject_seed_interval: 0,
                brush_size: 0.0,
                stopgrad_pos: npa_config.stopgrad_pos,
                stopgrad_state: npa_config.stopgrad_state,
                rollout_particles: 4,
                rollout_step_min: 1,
                rollout_steps: 1,
                update_prob: 1.0,
                seed: 17,
                seed_scale: 0.08,
                seed_mode: crate::ParticleSeed::UniformCircle,
                grid_eps: grid.eps,
                motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
                loss_config: crate::Target2dLossConfig::default(),
                per_parameter_grad_normalization: false,
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
                update_base: true,
                eval_examples: 1,
                eval_interval: 0,
                eval_batch_size: 1,
                eval_seed: 17,
                system_memory_budget_gb: None,
                gpu_memory_budget_gb: None,
                max_dense_train_particles: 4,
                max_dense_chunk_floats: 1_000_000,
                max_splat_chunk_floats: 1_000_000,
            };
            let features = tensor_vec(dense_perception(&x, &s, config).inner()).unwrap();
            let max_abs_diff = features
                .iter()
                .zip(reference.features.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_abs_diff < 2.0e-3,
                "dense Burn perception diverged from reference: max_abs_diff={max_abs_diff}"
            );
        }

        #[test]
        fn burn_target_splat_loss_matches_reference_cpu_fixture() {
            let npa_config = NpaConfig::growing_2d();
            let grid = burn_automata_kernels::HashGridConfig::growing_2d();
            let target = crate::TargetImage2d {
                source_width: 16,
                source_height: 16,
                positions: vec![[-0.35, 0.25], [0.2, 0.05], [0.45, -0.3]],
                colors: vec![[0.1, 0.8, 0.2], [0.7, 0.3, 0.1], [0.2, 0.4, 0.9]],
                pixel_size: 2.0 / 16.0,
                threshold: 0.05,
                aabb: [-1.0, 1.0, -1.0, 1.0],
            };
            let loss_config = crate::Target2dLossConfig {
                image_size: 16,
                sigma: 1.0,
                center: true,
                foreground_density_loss_weight: 0.5,
                displacement_regularizer_weight: 0.0,
                overflow_regularizer_weight: 0.0,
                bound_regularizer_weight: 0.0,
                ..crate::Target2dLossConfig::default()
            };
            let (positions, mut states) = seed_particles_scaled(
                1,
                4,
                npa_config.state_dims,
                npa_config.spatial_dims,
                29,
                crate::ParticleSeed::UniformCircle,
                0.3,
            );
            for particle in 0..4 {
                let base = particle * npa_config.state_dims + npa_config.state_dims - 3;
                states[base] = -0.3 + particle as f32 * 0.1;
                states[base + 1] = 0.2 - particle as f32 * 0.05;
                states[base + 2] = -0.1 + particle as f32 * 0.07;
            }
            let reference_output = crate::target_2d_loss_with_adjoint(
                &positions,
                &states,
                1,
                4,
                npa_config.state_dims,
                &target,
                loss_config,
                0.0,
                0,
            )
            .unwrap();
            let reference = reference_output.report;

            let device = BurnDevice::default();
            let pixels = loss_config.image_size * loss_config.image_size;
            let render = render_target_2d_splat(&target, loss_config).unwrap();
            let foreground = target_2d_foreground_mask(&target, loss_config).unwrap();
            let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
            let target_mean = target.mean_position();
            let target = BurnTargetExample {
                target_rgb: tensor(render.rgb, [pixels, 3], &device),
                target_density: tensor(render.density, [pixels, 1], &device),
                target_foreground: tensor(foreground, [pixels, 1], &device),
                target_foreground_scale: foreground_scale,
                target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], &device),
                target_positions: tensor(
                    target
                        .positions
                        .iter()
                        .flat_map(|position| [position[0], position[1]])
                        .collect(),
                    [target.positions.len(), 2],
                    &device,
                ),
                pixel_xy: tensor(pixel_xy_values(loss_config.image_size), [pixels, 2], &device),
                pixel_size: 2.0 / 16.0,
                target_points: 3,
                particle_count: 4,
                update_prob: 1.0,
                seed_scale: 0.3,
            };
            let model = NpaModel::upstream_seeded(npa_config.clone(), 29);
            let adapter = BurnAdapterParams::from_adapter(
                &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
                &model,
                &device,
            )
            .unwrap();
            let config = DirectBasisTrainConfig {
                steps: 0,
                report_interval: 1,
                example_batch_size: 1,
                tbptt_chunk_steps: 1,
                loss_on_final_chunk_only: false,
                use_particle_pool: false,
                pool_size: 0,
                inject_seed_interval: 0,
                brush_size: 0.0,
                stopgrad_pos: npa_config.stopgrad_pos,
                stopgrad_state: npa_config.stopgrad_state,
                rollout_particles: 4,
                rollout_step_min: 1,
                rollout_steps: 1,
                update_prob: 1.0,
                seed: 29,
                seed_scale: 0.3,
                seed_mode: crate::ParticleSeed::UniformCircle,
                grid_eps: grid.eps,
                motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
                loss_config,
                per_parameter_grad_normalization: false,
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
                update_base: true,
                eval_examples: 1,
                eval_interval: 0,
                eval_batch_size: 1,
                eval_seed: 29,
                system_memory_budget_gb: None,
                gpu_memory_budget_gb: None,
                max_dense_train_particles: 4,
                max_dense_chunk_floats: 1_000_000,
                max_splat_chunk_floats: 1_000_000,
            };
            let x = tracked_tensor(
                positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect(),
                [4, 2],
                &device,
            );
            let s = tracked_tensor(states, [4, npa_config.state_dims], &device);
            let loss = target_splat_loss(
                &x,
                &s,
                &target,
                config,
                &adapter,
                Tensor::<BurnBackend, 1>::zeros([1], &device),
            );

            let burn_total = loss.total.clone().inner().into_scalar();
            let burn_splat = loss.splat.clone().inner().into_scalar();
            let burn_color = loss.color.clone().inner().into_scalar();
            let burn_density = loss.density.clone().inner().into_scalar();

            assert!(
                (burn_total - reference.total_loss).abs() < 1.0e-4,
                "Burn total target2d loss diverged from CPU reference: burn={burn_total} reference={}",
                reference.total_loss
            );
            assert!(
                (burn_splat - reference.splat_loss).abs() < 1.0e-4,
                "Burn splat target2d loss diverged from CPU reference: burn={burn_splat} reference={}",
                reference.splat_loss
            );
            assert!(
                (burn_color - reference.color_loss).abs() < 1.0e-4,
                "Burn color target2d loss diverged from CPU reference: burn={burn_color} reference={}",
                reference.color_loss
            );
            assert!(
                (burn_density - reference.density_loss).abs() < 1.0e-4,
                "Burn density target2d loss diverged from CPU reference: burn={burn_density} reference={}",
                reference.density_loss
            );

            let mut grads = loss.total.backward();
            let burn_x_grad = tensor_vec(
                x.grad_remove(&mut grads)
                    .unwrap_or_else(|| x.clone().inner().zeros_like()),
            )
            .unwrap();
            let burn_s_grad = tensor_vec(
                s.grad_remove(&mut grads)
                    .unwrap_or_else(|| s.clone().inner().zeros_like()),
            )
            .unwrap();
            let max_position_grad_diff = burn_x_grad
                .chunks_exact(2)
                .zip(&reference_output.position_gradients)
                .flat_map(|(burn, reference)| {
                    [
                        (burn[0] - reference[0]).abs(),
                        (burn[1] - reference[1]).abs(),
                    ]
                })
                .fold(0.0_f32, f32::max);
            let max_state_grad_diff = burn_s_grad
                .iter()
                .zip(&reference_output.state_gradients)
                .map(|(burn, reference)| (burn - reference).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_position_grad_diff < 2.0e-3,
                "Burn target2d position gradient diverged from CPU reference: max_abs_diff={max_position_grad_diff}"
            );
            assert!(
                max_state_grad_diff < 2.0e-3,
                "Burn target2d state gradient diverged from CPU reference: max_abs_diff={max_state_grad_diff}"
            );
        }

        #[test]
        fn phase_batch_sampler_covers_all_examples_across_epoch() {
            let mut rng = StdRng::seed_from_u64(7);
            let mut sampler = PhaseBatchSampler::new(10, 3, &mut rng);
            let mut counts = [0usize; 10];
            for _ in 0..4 {
                let batch = sampler.next_batch(&mut rng);
                assert_eq!(batch.len(), 3);
                let mut sorted = batch.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), batch.len());
                for idx in batch {
                    counts[idx] += 1;
                }
            }

            assert!(counts.iter().all(|count| *count > 0));
            let stats = sample_update_stats(&counts);
            assert_eq!(stats.zero_update_examples, 0);
            assert_eq!(stats.total_updates, 12);
        }

        #[test]
        fn phase_batch_sampler_full_batch_returns_each_example_once() {
            let mut rng = StdRng::seed_from_u64(11);
            let mut sampler = PhaseBatchSampler::new(8, 0, &mut rng);
            let mut batch = sampler.next_batch(&mut rng);
            batch.sort_unstable();

            assert_eq!(batch, (0..8).collect::<Vec<_>>());
        }

        #[test]
        fn best_training_checkpoint_keeps_base_when_refine_regresses() {
            let train_phase = test_phase(Some(5.8), 300);
            let train_refine_phase = test_phase(Some(6.1), 0);

            let (loss, step) = best_training_checkpoint(300, &train_phase, &train_refine_phase);

            assert_eq!(loss, Some(5.8));
            assert_eq!(step, 300);
        }

        #[test]
        fn best_training_checkpoint_offsets_better_refine_step() {
            let train_phase = test_phase(Some(5.8), 300);
            let train_refine_phase = test_phase(Some(4.9), 120);

            let (loss, step) = best_training_checkpoint(300, &train_phase, &train_refine_phase);

            assert_eq!(loss, Some(4.9));
            assert_eq!(step, 420);
        }

        fn test_phase(best_loss: Option<f32>, best_step: usize) -> BurnPhaseReport {
            BurnPhaseReport {
                history: Vec::new(),
                best_loss,
                best_step,
                best_geometry_score: None,
                sample_updates: sample_update_stats(&[]),
            }
        }
    }
}
    };
}

dense_direct_basis_backend!(
    wgpu_imp,
    feature = "backend_wgpu",
    burn::backend::Wgpu<f32>,
    "burn_wgpu_autodiff_dense_direct_basis",
    "wgpu-default",
    "burn-wgpu"
);

dense_direct_basis_backend!(
    ndarray_imp,
    all(test, feature = "backend_ndarray"),
    burn::backend::NdArray<f32>,
    "burn_ndarray_autodiff_dense_direct_basis",
    "ndarray-default",
    "burn-ndarray"
);

dense_direct_basis_backend!(
    cuda_imp,
    feature = "backend_cuda",
    burn::backend::Cuda<f32>,
    "burn_cuda_autodiff_dense_direct_basis",
    "cuda-default",
    "burn-cuda"
);

#[cfg(feature = "backend_wgpu")]
pub(super) fn train_direct_basis_burn_wgpu(
    base: &mut super::NpaModel,
    train_examples: &mut [super::DirectBasisExample],
    holdout_examples: &mut [super::DirectBasisExample],
    train_config: super::DirectBasisTrainConfig,
    train_refine_config: super::DirectBasisTrainConfig,
    holdout_config: super::DirectBasisTrainConfig,
    checkpoint: Option<&super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    wgpu_imp::train_direct_basis_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        train_refine_config,
        holdout_config,
        checkpoint,
    )
}

#[cfg(feature = "backend_wgpu")]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_wgpu(
    base: &mut super::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    wgpu_imp::train_e2e_rollout_burn_dense(base, train_examples, holdout_examples, train_config)
}

#[cfg(not(feature = "backend_wgpu"))]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_wgpu(
    _base: &mut super::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU HyperNPA e2e rollout training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(not(feature = "backend_wgpu"))]
pub(super) fn train_direct_basis_burn_wgpu(
    _base: &mut super::NpaModel,
    _train_examples: &mut [super::DirectBasisExample],
    _holdout_examples: &mut [super::DirectBasisExample],
    _train_config: super::DirectBasisTrainConfig,
    _train_refine_config: super::DirectBasisTrainConfig,
    _holdout_config: super::DirectBasisTrainConfig,
    _checkpoint: Option<&super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU direct-basis training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu or choose the Burn/CUDA backend in a CUDA build",
    )
    .into())
}

#[cfg(feature = "backend_wgpu")]
pub(super) fn train_oracle_models_burn_wgpu(
    models: &mut [super::NpaModel],
    examples: &[super::DirectBasisExample],
    train_config: super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    wgpu_imp::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(not(feature = "backend_wgpu"))]
pub(super) fn train_oracle_models_burn_wgpu(
    _models: &mut [super::NpaModel],
    _examples: &[super::DirectBasisExample],
    _train_config: super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/WGPU vectorized oracle training requires the backend_wgpu feature; rebuild with --features cli,backend_wgpu",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(super) fn train_direct_basis_burn_cuda(
    base: &mut super::NpaModel,
    train_examples: &mut [super::DirectBasisExample],
    holdout_examples: &mut [super::DirectBasisExample],
    train_config: super::DirectBasisTrainConfig,
    train_refine_config: super::DirectBasisTrainConfig,
    holdout_config: super::DirectBasisTrainConfig,
    checkpoint: Option<&super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    cuda_imp::train_direct_basis_burn_dense(
        base,
        train_examples,
        holdout_examples,
        train_config,
        train_refine_config,
        holdout_config,
        checkpoint,
    )
}

#[cfg(feature = "backend_cuda")]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_cuda(
    base: &mut super::NpaModel,
    train_examples: &mut [BurnE2eRolloutExample],
    holdout_examples: &mut [BurnE2eRolloutExample],
    train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    cuda_imp::train_e2e_rollout_burn_dense(base, train_examples, holdout_examples, train_config)
}

#[cfg(not(feature = "backend_cuda"))]
pub(in crate::cli::commands::hyper_e2e) fn train_e2e_rollout_burn_cuda(
    _base: &mut super::NpaModel,
    _train_examples: &mut [BurnE2eRolloutExample],
    _holdout_examples: &mut [BurnE2eRolloutExample],
    _train_config: BurnE2eRolloutTrainConfig,
) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA HyperNPA e2e rollout training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(not(feature = "backend_cuda"))]
pub(super) fn train_direct_basis_burn_cuda(
    _base: &mut super::NpaModel,
    _train_examples: &mut [super::DirectBasisExample],
    _holdout_examples: &mut [super::DirectBasisExample],
    _train_config: super::DirectBasisTrainConfig,
    _train_refine_config: super::DirectBasisTrainConfig,
    _holdout_config: super::DirectBasisTrainConfig,
    _checkpoint: Option<&super::Target2dBurnCheckpointConfig>,
) -> Result<BurnWgpuDirectBasisOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA dense direct-basis training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}

#[cfg(feature = "backend_cuda")]
pub(super) fn train_oracle_models_burn_cuda(
    models: &mut [super::NpaModel],
    examples: &[super::DirectBasisExample],
    train_config: super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    cuda_imp::train_oracle_models_burn_dense(models, examples, train_config)
}

#[cfg(not(feature = "backend_cuda"))]
pub(super) fn train_oracle_models_burn_cuda(
    _models: &mut [super::NpaModel],
    _examples: &[super::DirectBasisExample],
    _train_config: super::DirectBasisTrainConfig,
) -> Result<BurnDenseOracleBatchOutput, Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "Burn/CUDA vectorized oracle training requires the backend_cuda feature; rebuild with --features cli,backend_cuda or use backend = \"burn-wgpu\"",
    )
    .into())
}
