mod backends;

pub(crate) use backends::{
    train_direct_basis_burn_cuda, train_direct_basis_burn_wgpu, train_oracle_models_burn_cuda,
    train_oracle_models_burn_wgpu,
};
pub(crate) use backends::{train_e2e_rollout_burn_cuda, train_e2e_rollout_burn_wgpu};

macro_rules! dense_direct_basis_backend {
    (
        $module:ident,
        $feature:meta,
        $perception_cube_feature:meta,
        $inner_backend:ty,
        $backend_name:expr,
        $device_label:expr,
        $log_backend:expr
    ) => {
#[cfg($feature)]
#[allow(dead_code)]
mod $module {
    use std::{
        collections::{HashMap, VecDeque},
        fs,
        path::{Path, PathBuf},
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Instant,
    };

    use burn::{
        backend::{
            Autodiff,
            autodiff::{
                checkpoint::strategy::NoCheckpointing,
                grads::Gradients,
                ops::{Backward, Ops, OpsKind},
            },
        },
        tensor::{
            Device, Distribution, IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive,
            activation::{relu, softmax}, backend::Backend,
        },
    };
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use rayon::prelude::*;
    use serde::Serialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::super::{
        BurnDenseOracleBatchOutput, BurnE2eAdapterDiagnostics, BurnE2eNearestTeacherEntry,
        BurnE2eRolloutExample, BurnE2eRolloutHorizonSummary, BurnE2eRolloutOutput,
        BurnE2eRolloutHistoryEntry, BurnE2eRolloutQualityEntry, BurnE2eRolloutQualityReport,
        BurnE2eRolloutTrainConfig,
        BurnWgpuDirectBasisOutput,
        DirectBasisStepStats, DirectBasisTrainConfig,
        DirectBasisTrainingExample as DirectBasisExample, E2eAdapterTeacherObjective,
        E2eCreditAssignment, E2eIdentitySampler, E2eLrSchedule, E2eParticlePoolSnapshot,
        E2eTbpttLossMode, E2eTensorSnapshot, E2eTrainingCheckpoint,
        E2E_TRAINING_CHECKPOINT_VERSION,
        Hyper2dDirectBasisHistoryEntry as CliHyper2dDirectBasisHistoryEntry,
        Hyper2dDirectBasisLossSummary as CliHyper2dDirectBasisLossSummary,
    };
    use crate::hyper::e2e::{
        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK, E2E_HYPER_ADAPTER_FACTORIZED,
        E2eHyperGeneratorKind, E2eHyperNpa2d, E2eHyperNpa2dWeights, PerceptionRolloutBackend,
        Target2dLossBackend, save_e2e_hyper_npa_2d,
    };
    #[cfg(feature = "dino")]
    use crate::hyper::dino::{
        DinoVitsConditionContract, DinoVitsConditionEncoderBackend,
        DinoVitsPreparedConditionBatch,
    };
    use crate::{
        AdamWConfig, AutomataError, AutomataResult, BpkModelManifest, ConditionImage2d, NpaConfig,
        NpaLowRankAdapter, NpaModel, NpaWeights,
        SgdConfig,
        rollout::{seed_particles_scaled, stochastic_mask},
        target2d::{render_target_2d_splat, target_2d_foreground_mask},
        TargetImage2d,
    };
    #[cfg($perception_cube_feature)]
    use burn_automata_kernels::{
        PerceptionCubeAdjointBackend, PerceptionCubeAdjointConfig, PerceptionCubeForwardBackend,
        PerceptionCubePreparedBackend,
    };
    #[cfg($perception_cube_feature)]
    use burn_automata_kernels::{Target2dCubeAdjointBackend, Target2dCubeLossConfig};

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
    type Tensor4Inner = Tensor<InnerBackend, 4>;
    type Tensor1IntInner = Tensor<InnerBackend, 1, Int>;
    type Tensor2IntInner = Tensor<InnerBackend, 2, Int>;

    const BACKEND: &str = $backend_name;
    const DEVICE_LABEL: &str = $device_label;
    const LOG_BACKEND: &str = $log_backend;
    const EPSILON: f32 = 1.0e-6;
    const FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT: f32 = 0.01;
    static PERCEPTION_CUBE_ADJOINT_DEVICE_HITS: AtomicUsize = AtomicUsize::new(0);
    static PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS: AtomicUsize = AtomicUsize::new(0);
    static PERCEPTION_CUBE_FORWARD_DEVICE_HITS: AtomicUsize = AtomicUsize::new(0);
    static PERCEPTION_CUBE_FORWARD_FALLBACK_HITS: AtomicUsize = AtomicUsize::new(0);
    static PERCEPTION_CUBE_PREPARED_REUSE_HITS: AtomicUsize = AtomicUsize::new(0);
    static TARGET2D_CUBE_ADJOINT_DEVICE_HITS: AtomicUsize = AtomicUsize::new(0);
    static TARGET2D_CUBE_ADJOINT_FALLBACK_HITS: AtomicUsize = AtomicUsize::new(0);
    static STOCHASTIC_MASK_UPLOAD_HITS: AtomicUsize = AtomicUsize::new(0);
    static STOCHASTIC_MASK_DEVICE_HITS: AtomicUsize = AtomicUsize::new(0);

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

    #[derive(Clone)]
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
        kind: E2eHyperGeneratorKind,
        token_w: Tensor2,
        token_b: Tensor2,
        token_gate_w: Tensor2,
        token_gate_b: Tensor2,
        state_w: Tensor2,
        time_w: Tensor2,
        output_w: Tensor2,
        output_b: Tensor2,
        condition_control_w: Tensor2,
        condition_control_b: Tensor2,
        condition_control_state_w: Tensor2,
        hidden_dims: usize,
        token_attention_heads: usize,
        softmax_token_attention: bool,
        canonical_full_rank_lora: bool,
        adapter_constants: Tensor2,
        adapter_trainable_mask: Tensor2,
        adapter_parameter_segments: Vec<(usize, usize)>,
        output_dims: usize,
        output_scale: f32,
        sample_steps: usize,
        adapter_chunk_size: usize,
        output_chunks: usize,
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
        condition_control_w_m: Tensor2Inner,
        condition_control_w_v: Tensor2Inner,
        condition_control_b_m: Tensor2Inner,
        condition_control_b_v: Tensor2Inner,
        condition_control_state_w_m: Tensor2Inner,
        condition_control_state_w_v: Tensor2Inner,
    }

    #[derive(Clone)]
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
        target_cpu: TargetImage2d,
    }

    struct BurnE2eCpuTargetInput {
        target: TargetImage2d,
        particle_count: usize,
        update_prob: f32,
        seed_scale: f32,
    }

    struct BurnE2ePreparedTargetExample {
        target_rgb: Vec<f32>,
        target_density: Vec<f32>,
        target_foreground: Vec<f32>,
        target_foreground_scale: f32,
        target_mean: [f32; 2],
        target_positions: Vec<f32>,
        pixel_size: f32,
        target_points: usize,
        particle_count: usize,
        update_prob: f32,
        seed_scale: f32,
        target_cpu: TargetImage2d,
    }

    struct BurnE2ePreparedCpuBatch {
        indices: Vec<usize>,
        targets: Vec<BurnE2ePreparedTargetExample>,
        prepared_dino: Option<BurnE2ePreparedDinoBatch>,
    }

    struct BurnE2eCpuBatchPrefetch {
        indices: Vec<usize>,
        handle: thread::JoinHandle<Result<BurnE2ePreparedCpuBatch, String>>,
    }

    #[cfg(feature = "dino")]
    type BurnE2ePreparedDinoBatch = DinoVitsPreparedConditionBatch;

    #[cfg(not(feature = "dino"))]
    struct BurnE2ePreparedDinoBatch;

    struct BurnPoolBatch {
        pool_indices: Vec<usize>,
        x: Tensor3,
        s: Tensor3,
    }

    #[derive(Clone)]
    struct BurnE2eConditionControlBatch {
        patch_hidden: Tensor3,
        update_w: Tensor2,
        update_b: Tensor2,
        state_w: Option<Tensor2>,
        grid_width: usize,
        grid_height: usize,
        sigma: f32,
        scale: f32,
    }

    impl BurnE2eConditionControlBatch {
        fn update_for_particles(&self, x: &Tensor3, state: &Tensor3) -> Tensor3 {
            let dims = x.shape().dims::<3>();
            let batches = dims[0];
            let particles = dims[1];
            let patch_tokens = self.grid_width.saturating_mul(self.grid_height);
            let hidden_dims = self.patch_hidden.shape().dims::<3>()[2];
            let update_dims = self.update_b.shape().dims::<2>()[1];
            let device = x.device();
            let centers = tensor4(
                condition_patch_centers_values(self.grid_width, self.grid_height),
                [1, 1, patch_tokens, 2],
                &device,
            );
            let particle_xy = x
                .clone()
                .unsqueeze_dim::<4>(2)
                .expand([batches, particles, patch_tokens, 2]);
            let center_xy = centers.expand([batches, particles, patch_tokens, 2]);
            let diff = particle_xy - center_xy;
            let dist2 = diff
                .clone()
                .mul(diff)
                .sum_dim(3)
                .squeeze_dim::<3>(3);
            let weights = dist2
                .mul_scalar(-1.0 / (self.sigma * self.sigma).max(EPSILON))
                .exp();
            let weights = weights.clone().div(
                weights
                    .sum_dim(2)
                    .add_scalar(EPSILON)
                    .expand([batches, particles, patch_tokens]),
            );
            let mut local = weights.matmul(self.patch_hidden.clone());
            if let Some(state_w) = &self.state_w {
                let state_dims = state.shape().dims::<3>()[2];
                let projected = state.clone().matmul(
                    state_w
                        .clone()
                        .transpose()
                        .unsqueeze_dim::<3>(0)
                        .expand([batches, state_dims, hidden_dims]),
                );
                local = relu(local + projected);
            }
            let update_w = self
                .update_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, hidden_dims, update_dims]);
            let update_b = self
                .update_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, particles, update_dims]);
            (local.matmul(update_w) + update_b).mul_scalar(self.scale)
        }

        fn select_rows(self, rows: &[usize]) -> Self {
            if rows.is_empty() {
                return self;
            }
            let device = self.patch_hidden.device();
            let indices = Tensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(
                    rows.iter().map(|row| *row as i64).collect::<Vec<_>>(),
                    [rows.len()],
                ),
                &device,
            );
            Self {
                patch_hidden: self.patch_hidden.select(0, indices),
                ..self
            }
        }

        fn select_rows_or_identity(self, rows: Option<&[usize]>) -> Self {
            match rows {
                Some(rows) => self.select_rows(rows),
                None => self,
            }
        }
    }

    struct BurnE2ePoolBatch {
        slots: Vec<usize>,
        x: Tensor3,
        s: Tensor3,
        seed_replacements: usize,
    }

    struct BurnDeviceParticlePool {
        positions: Tensor3Inner,
        states: Tensor3Inner,
        pool_size: usize,
        particle_count: usize,
        state_dims: usize,
    }

    struct BurnE2eDeviceParticlePool {
        positions: Tensor3Inner,
        states: Tensor3Inner,
        slot_examples: Vec<Option<(usize, usize)>>,
        example_slots: HashMap<(usize, usize), usize>,
        next_evict: usize,
        capacity: usize,
        particle_count: usize,
        state_dims: usize,
        slots_per_example: usize,
    }

    enum BurnE2eConditionValues {
        Device(Tensor3),
        HostRows(Vec<Vec<f32>>),
        #[cfg(feature = "dino")]
        DynamicDino(Box<BurnE2eDinoConditionSource>),
    }

    struct BurnE2eConditionCache {
        values: BurnE2eConditionValues,
        teacher_vectors: Option<Tensor2>,
        examples: usize,
        token_count: usize,
        embed_dims: usize,
        device: BurnDevice,
    }

    #[cfg(feature = "dino")]
    struct BurnE2eDinoConditionSource {
        paths: Vec<PathBuf>,
        encoder: DinoVitsConditionEncoderBackend<InnerBackend>,
        batch_size: usize,
        token_grid_width: usize,
        token_grid_height: usize,
        l2_normalize_features: bool,
        rgb_channels: bool,
        rgb_channel_scale: f32,
        alpha_channel: bool,
        alpha_channel_scale: f32,
    }

    #[cfg(feature = "dino")]
    impl BurnE2eDinoConditionSource {
        fn contract(&self) -> DinoVitsConditionContract {
            DinoVitsConditionContract::token_grid(
                self.token_grid_width,
                self.token_grid_height,
                self.l2_normalize_features,
                self.rgb_channels,
                self.rgb_channel_scale,
                self.alpha_channel,
                self.alpha_channel_scale,
            )
        }
    }

    struct BurnE2eSelectedCheckpoint {
        step: usize,
        train_loss: f32,
        selection_score: f32,
        validation_contract: Option<BurnE2eValidationContract>,
        holdout_mean_psnr_db: Option<f32>,
        holdout_mean_loss: Option<f32>,
        quality_validation: Option<BurnE2eRolloutQualityReport>,
        params: BurnBaseParams,
        generator: BurnE2eGeneratorParams,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BurnE2eValidationContract {
        examples: usize,
        particles: usize,
        horizons: Vec<usize>,
        selection_horizon_min_steps: usize,
    }

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

    struct BurnE2eStepOutput {
        history: BurnE2eRolloutHistoryEntry,
        particle_steps: u64,
        final_x: Tensor3,
        final_s: Tensor3,
        per_example_losses: Option<Vec<f32>>,
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
            "representation": "resident_gpu_tensors",
            "per_step_host_state_transfer": false,
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
        metrics["target2d_loss_backend"] = json!(train_config.target2d_loss_backend.as_str());
        metrics["target2d_loss_backend_effective"] =
            json!(target2d_loss_backend_effective(train_config).as_str());
        metrics["perception_backend"] = json!(train_config.perception_backend.as_str());
        metrics["perception_backend_effective"] =
            json!(perception_backend_effective(train_config).as_str());
        metrics["perception_sparse_grid_effective"] = json!(
            perception_backend_effective(train_config)
                == PerceptionRolloutBackend::TiledAdjoint
                && train_config.rollout_particles >= 512
                && train_config.stopgrad_pos
        );
        metrics["perception_cube_adjoint_device_hits"] =
            json!(PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_adjoint_fallback_hits"] =
            json!(PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_forward_device_hits"] =
            json!(PERCEPTION_CUBE_FORWARD_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_forward_fallback_hits"] =
            json!(PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.load(Ordering::Relaxed));
        metrics["perception_cube_prepared_reuse_hits"] =
            json!(PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed));
        metrics["target2d_cube_adjoint_device_hits"] =
            json!(TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed));
        metrics["target2d_cube_adjoint_fallback_hits"] =
            json!(TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed));
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
        initial_generator: Option<&E2eHyperNpa2d>,
    ) -> Result<BurnE2eRolloutOutput, Box<dyn std::error::Error>> {
        PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_FORWARD_DEVICE_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.store(0, Ordering::Relaxed);
        PERCEPTION_CUBE_PREPARED_REUSE_HITS.store(0, Ordering::Relaxed);
        TARGET2D_CUBE_ADJOINT_DEVICE_HITS.store(0, Ordering::Relaxed);
        TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.store(0, Ordering::Relaxed);
        STOCHASTIC_MASK_UPLOAD_HITS.store(0, Ordering::Relaxed);
        STOCHASTIC_MASK_DEVICE_HITS.store(0, Ordering::Relaxed);
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
        let resume_checkpoint = config
            .resume_checkpoint
            .map(|path| load_e2e_training_checkpoint(path, train_examples, &base.config, config))
            .transpose()?;
        let mut resumed_generator = None;
        if let Some(resume_path) = config.resume_checkpoint {
            let requested = Path::new(resume_path);
            let checkpoint_dir = if requested.is_dir() {
                requested
            } else {
                requested.parent().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "training resume checkpoint has no parent directory".to_string(),
                    )
                })?
            };
            let shared_base_path = checkpoint_dir.join("current_shared_base.bpk");
            let hyper_path = checkpoint_dir.join("current_hyper_2d.bpk");
            *base = crate::import::load_manifest(&shared_base_path)?.into_model();
            resumed_generator = Some(crate::load_e2e_hyper_npa_2d(&hyper_path)?);
        }
        let initial_generator = resumed_generator.as_ref().or(initial_generator);
        let started = Instant::now();
        let device = BurnDevice::default();
        <BurnBackend as Backend>::seed(&device, config.seed);
        let npa_config = base.config.clone();
        let mut params = BurnBaseParams::from_model(base, &device)?;
        let mut base_optimizer = if let Some(checkpoint) = &resume_checkpoint {
            BurnBaseAdamWState::restore(checkpoint, &device)?
        } else {
            BurnBaseAdamWState::zeros_like(&params)
        };
        let mut generator = BurnE2eGeneratorParams::from_seed_or_artifact(
            base,
            train_examples,
            config,
            initial_generator,
            &device,
        )?;
        let mut generator_optimizer = if let Some(checkpoint) = &resume_checkpoint {
            BurnE2eGeneratorAdamWState::restore(checkpoint, &device)?
        } else {
            BurnE2eGeneratorAdamWState::new(&generator)
        };
        let mut particle_pool = if config.use_particle_pool {
            Some(if let Some(snapshot) = resume_checkpoint
                .as_ref()
                .and_then(|checkpoint| checkpoint.particle_pool.as_ref())
            {
                BurnE2eDeviceParticlePool::restore(snapshot, config, &device)?
            } else {
                BurnE2eDeviceParticlePool::new(
                    config.pool_capacity,
                    config.rollout_particles,
                    16,
                    config.pool_slots_per_example,
                    &device,
                )
            })
        } else {
            None
        };
        let train_conditions = BurnE2eConditionCache::from_examples_drain(
            train_examples,
            &device,
            config.condition_device_cache_max_bytes,
            config,
        )?;
        let holdout_conditions =
            BurnE2eConditionCache::from_examples_drain(
                holdout_examples,
                &device,
                config.condition_device_cache_max_bytes,
                config,
            )?;
        let train_condition_cache_bytes = train_conditions.feature_bytes();
        let holdout_condition_cache_bytes = holdout_conditions.feature_bytes();
        let condition_cache_bytes =
            train_condition_cache_bytes.saturating_add(holdout_condition_cache_bytes);
        let train_condition_pairwise_l2 = train_conditions.mean_pairwise_l2()?;
        let train_teacher_pairwise_l2 = train_conditions.mean_teacher_pairwise_l2()?;
        eprintln!(
            "hyper2d condition diagnostics examples={} condition_pairwise_l2={:.6} teacher_pairwise_l2={:.6}",
            train_conditions.examples,
            train_condition_pairwise_l2.unwrap_or_default(),
            train_teacher_pairwise_l2.unwrap_or_default(),
        );
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
            BurnE2eRolloutTrainConfig {
                validation_examples: config.initial_validation_examples,
                ..config
            },
            &device,
        )?;
        if let Some(quality) = &initial_quality_validation {
            eprintln!(
                "hyper2d e2e rollout initial {} quality composited_psnr={:.3}dB p10={:.3}dB density_psnr={:.3}dB soft_iou={:.3} mean_loss={:.6e}",
                quality.split,
                quality.aggregate_composited_rgb_psnr_db,
                quality.p10_composited_rgb_psnr_db,
                quality.aggregate_density_psnr_db,
                quality.mean_density_soft_iou,
                quality.mean_total_loss,
            );
        }
        let mut quality_validation_evaluations = initial_quality_validation
            .as_ref()
            .map_or(0usize, |_| 1usize);
        let mut quality_validation_elapsed_ms = initial_quality_validation
            .as_ref()
            .map_or(0.0_f64, |quality| quality.elapsed_ms);

        let mut sampler_init_rng = StdRng::seed_from_u64(config.seed);
        let condition_batch_size =
            normalized_batch_size(config.example_batch_size, train_examples.len());
        let rollout_replicas = config.rollouts_per_example.max(1);
        let batch_size = condition_batch_size.saturating_mul(rollout_replicas);
        let mut identity_sampler = resume_checkpoint.as_ref().map_or_else(
            || {
                E2eIdentitySampler::new(
                    train_examples.len(),
                    condition_batch_size,
                    config.sampling_uniform_fraction,
                    config.sampling_priority_ema_beta,
                    config.sampling_priority_min_weight,
                    config.sampling_priority_max_weight,
                    &mut sampler_init_rng,
                )
            },
            |checkpoint| checkpoint.sampler.clone(),
        );
        let mut seed_trajectory_counts = resume_checkpoint.as_ref().map_or_else(
            || vec![0usize; train_examples.len()],
            |checkpoint| checkpoint.seed_trajectory_counts.clone(),
        );
        let completed_step = resume_checkpoint
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.completed_step);
        if completed_step > 0 {
            eprintln!(
                "hyper2d resumed exact training state at completed_step={completed_step} optimizer_steps={}/{} pending_batches={}",
                base_optimizer.step,
                generator_optimizer.step,
                resume_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.pending_batches.len()),
            );
        }
        let train_pixel_xy = burn_e2e_pixel_xy(config, &device);
        let projected_target_cache_bytes = e2e_target_cache_bytes(train_examples, config);
        let train_target_cache = if projected_target_cache_bytes
            <= config.target_device_cache_max_bytes
        {
            eprintln!(
                "hyper2d target cache examples={} bytes={} storage=device-resident",
                train_examples.len(), projected_target_cache_bytes
            );
            Some(burn_e2e_target_cache(
                train_examples,
                config,
                &train_pixel_xy,
                &device,
            )?)
        } else {
            eprintln!(
                "hyper2d target cache examples={} bytes={} exceeds limit={}; using CPU prefetch",
                train_examples.len(),
                projected_target_cache_bytes,
                config.target_device_cache_max_bytes,
            );
            None
        };
        check_process_memory_budget("e2e_rollout:after_target_cache", direct_config_view(config))?;
        check_gpu_memory_budget("e2e_rollout:after_target_cache", direct_config_view(config))?;
        let prefetch_depth = e2e_cpu_prefetch_depth(batch_size, config.steps);
        let mut prefetch_queue = VecDeque::with_capacity(prefetch_depth);
        for indices in resume_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.pending_batches.as_slice())
            .unwrap_or_default()
        {
            prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                train_examples,
                &train_conditions,
                indices.clone(),
                config,
                train_target_cache.is_some(),
            )?);
        }
        let mut next_prefetch_step = completed_step
            .saturating_add(prefetch_queue.len())
            .saturating_add(1);
        while next_prefetch_step <= config.steps && prefetch_queue.len() < prefetch_depth {
            let mut sample_rng = e2e_sampling_rng(config.seed, next_prefetch_step);
            prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                train_examples,
                &train_conditions,
                sample_rollout_indices(
                    &mut identity_sampler,
                    rollout_replicas,
                    &mut sample_rng,
                ),
                config,
                train_target_cache.is_some(),
            )?);
            next_prefetch_step += 1;
        }
        let mut history = Vec::new();
        let mut final_loss = None;
        let mut best_checkpoint = initial_quality_validation
            .as_ref()
            .filter(|_| initial_validation_is_checkpoint_comparable(config))
            .map(|quality| BurnE2eSelectedCheckpoint {
                step: completed_step,
                train_loss: quality.mean_total_loss,
                selection_score: quality.selection_psnr_db,
                validation_contract: Some(e2e_validation_contract(
                    BurnE2eRolloutTrainConfig {
                        validation_examples: config.initial_validation_examples,
                        ..config
                    },
                )),
                holdout_mean_psnr_db: Some(quality.aggregate_composited_rgb_psnr_db),
                holdout_mean_loss: Some(quality.mean_total_loss),
                quality_validation: Some(quality.clone()),
                params: params.detached(),
                generator: generator.detached(),
            });
        let mut final_checkpoint_candidate = None::<BurnE2eSelectedCheckpoint>;
        let mut early_stop_step = None::<usize>;
        let validation_interval = config.validation_interval.max(1);
        let mut last_checkpoint_at = Instant::now();
        let mut throughput_interval_started = Instant::now();
        let mut throughput_interval_particle_steps = 0u128;
        let mut total_optimizer_particle_steps = 0u128;
        let mut measured_optimizer_training_ms = 0.0_f64;
        for step in completed_step.saturating_add(1)..=config.steps {
            let prepared_batch =
                join_e2e_cpu_batch_prefetch(prefetch_queue.pop_front().ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e CPU prefetch queue was empty".to_string(),
                    )
                })?)?;
            let BurnE2ePreparedCpuBatch {
                indices,
                targets: prepared_targets,
                prepared_dino,
            } = prepared_batch;
            while next_prefetch_step <= config.steps && prefetch_queue.len() < prefetch_depth {
                let mut sample_rng = e2e_sampling_rng(config.seed, next_prefetch_step);
                prefetch_queue.push_back(spawn_e2e_cpu_batch_prefetch(
                    train_examples,
                    &train_conditions,
                    sample_rollout_indices(
                        &mut identity_sampler,
                        rollout_replicas,
                        &mut sample_rng,
                    ),
                    config,
                    train_target_cache.is_some(),
                )?);
                next_prefetch_step += 1;
            }
            let step_seed = config
                .seed
                .wrapping_add((step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let report_due =
                step == config.steps || step.is_multiple_of(config.report_interval.max(1));
            let validation_due = config.validation_examples > 0
                && (step == config.steps || step.is_multiple_of(validation_interval));
            let checkpoint_due = should_write_e2e_checkpoint(step, last_checkpoint_at, config);
            let priority_due = step
                .is_multiple_of(config.sampling_priority_update_interval.max(1));
            let collect_metrics = report_due || validation_due || checkpoint_due;
            let collect_per_example_losses = collect_metrics || priority_due;
            if config.lr_schedule == E2eLrSchedule::UpstreamGrowing
                && step > 1
                && step.saturating_sub(1).is_multiple_of(10_000)
            {
                base_optimizer = BurnBaseAdamWState::zeros_like(&params);
                generator_optimizer = BurnE2eGeneratorAdamWState::new(&generator);
                if config.use_particle_pool {
                    particle_pool = Some(BurnE2eDeviceParticlePool::new(
                        config.pool_capacity,
                        config.rollout_particles,
                        16,
                        config.pool_slots_per_example,
                        &device,
                    ));
                }
                seed_trajectory_counts.fill(0);
                eprintln!(
                    "hyper2d upstream-growing repetition reset at optimizer step {step}"
                );
            }
            let lr_scale = e2e_lr_scale(config, step);
            let mut step_config = e2e_config_with_lr_scale(config, lr_scale);
            step_config.shared_base_trainable =
                config.shared_base_trainable && step >= config.shared_base_train_start_step;
            let pool_batch = if let Some(pool) = particle_pool.as_mut() {
                let mut pool_rng = e2e_pool_rng(config.seed, step);
                let seed_replacement_rows = per_identity_seed_replacement_rows(
                    &indices,
                    &mut seed_trajectory_counts,
                    config.seed_trajectory_interval,
                );
                Some(pool.sample_batch(
                    &indices,
                    &mut pool_rng,
                    &seed_replacement_rows,
                    step_config.seed_scale,
                    direct_config_view(step_config),
                    &device,
                )?)
            } else {
                None
            };
            let seed_replacements = pool_batch
                .as_ref()
                .map_or(0usize, |batch| batch.seed_replacements);
            let (initial_state, pool_slots) = pool_batch
                .map(|batch| (Some((batch.x, batch.s)), Some(batch.slots)))
                .unwrap_or((None, None));
            let uncached_targets;
            let (step_targets, target_indices) = if let Some(cache) = train_target_cache.as_ref() {
                (cache.as_slice(), indices.clone())
            } else {
                uncached_targets = burn_e2e_prepared_targets_to_burn(
                    prepared_targets,
                    &train_pixel_xy,
                    &device,
                )?;
                let indices = (0..uncached_targets.len()).collect::<Vec<_>>();
                (uncached_targets.as_slice(), indices)
            };
            let step_output = train_e2e_homogeneous_step_tbptt(
                &mut params,
                &mut generator,
                &mut base_optimizer,
                &mut generator_optimizer,
                &npa_config,
                &train_conditions,
                &indices,
                prepared_dino.as_ref(),
                step_targets,
                &target_indices,
                step_config,
                step_seed,
                collect_metrics,
                collect_per_example_losses,
                initial_state,
            )?;
            let step_particle_steps = step_output.particle_steps as u128;
            throughput_interval_particle_steps = throughput_interval_particle_steps
                .saturating_add(step_particle_steps);
            total_optimizer_particle_steps =
                total_optimizer_particle_steps.saturating_add(step_particle_steps);
            let condition_identities = indices
                .chunks(rollout_replicas)
                .filter_map(|replicas| replicas.first().copied())
                .collect::<Vec<_>>();
            identity_sampler.record_trajectories(&condition_identities, rollout_replicas);
            if let Some(per_example_losses) = step_output.per_example_losses.as_deref() {
                identity_sampler.update_losses(&indices, per_example_losses);
            }
            if let (Some(pool), Some(pool_slots)) = (particle_pool.as_mut(), pool_slots) {
                pool.update_batch(&pool_slots, step_output.final_x, step_output.final_s)?;
            }
            let mut stats = step_output.history;
            if collect_metrics {
                sync_training_device(&device)?;
                let interval_elapsed = throughput_interval_started.elapsed();
                let interval_elapsed_ms = interval_elapsed.as_secs_f64() * 1_000.0;
                measured_optimizer_training_ms += interval_elapsed_ms;
                stats.particle_steps_per_sec = throughput_interval_particle_steps as f64
                    / interval_elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
                stats.dense_pair_interactions_per_sec =
                    stats.particle_steps_per_sec * config.rollout_particles as f64;
                stats.elapsed_ms = interval_elapsed_ms;
            }
            stats.step = step;
            stats.learning_rate_scale = lr_scale;
            stats.base_learning_rate = step_config.base_optimizer.learning_rate;
            stats.generator_learning_rate = step_config.generator_optimizer.learning_rate;
            stats.pool_seed_replacements = seed_replacements;
            if collect_metrics {
                final_loss = Some(stats.loss);
                let validation_config = validation_due.then(|| {
                    if step == config.steps {
                        e2e_final_validation_config(config)
                    } else {
                        config
                    }
                });
                let checkpoint_quality = if let Some(validation_config) = validation_config {
                    evaluate_e2e_rollout_quality(
                        &params.detached(),
                        &generator.detached(),
                        &npa_config,
                        train_examples,
                        holdout_examples,
                        &train_conditions,
                        &holdout_conditions,
                        validation_config,
                        &device,
                    )?
                } else {
                    None
                };
                if let Some(quality) = &checkpoint_quality {
                    quality_validation_evaluations =
                        quality_validation_evaluations.saturating_add(1);
                    quality_validation_elapsed_ms += quality.elapsed_ms;
                }
                let (holdout_mean_psnr_db, holdout_mean_loss, selection_score) =
                    if let Some(quality) = &checkpoint_quality {
                        stats.holdout_mean_psnr_db =
                            Some(quality.aggregate_composited_rgb_psnr_db);
                        stats.holdout_mean_loss = Some(quality.mean_total_loss);
                        (
                            Some(quality.aggregate_composited_rgb_psnr_db),
                            Some(quality.mean_total_loss),
                            quality.selection_psnr_db,
                        )
                    } else {
                        (None, None, -stats.loss)
                    };
                if report_due || validation_due {
                    let exposure = identity_sampler.exposure_stats();
                    eprintln!(
                        "hyper2d e2e rollout step {step}/{} loss={:.6e} task={:.6e} teacher={:.6e} lr_scale={:.3e} exposure_min={} exposure_mean={:.1} exposure_p90={} final_horizon_psnr={} worst_horizon_p10={} validation_due={} base_grad={:.6e} generator_grad={:.6e} particle_steps/s={:.3e} condition_ms={:.2} rollout_loss_ms={:.2} backward_ms={:.2}",
                        config.steps,
                        stats.loss,
                        stats.task_loss,
                        stats.adapter_teacher_loss,
                        stats.learning_rate_scale,
                        exposure.min,
                        exposure.mean,
                        exposure.p90,
                        format_optional_f32(holdout_mean_psnr_db),
                        checkpoint_quality
                            .as_ref()
                            .map(|quality| format!("{:.3}", quality.selection_psnr_db))
                            .unwrap_or_else(|| "n/a".to_string()),
                        validation_due,
                        stats.base_grad_norm,
                        stats.generator_grad_norm,
                        stats.particle_steps_per_sec,
                        stats.condition_adapter_ms,
                        stats.rollout_loss_ms,
                        stats.backward_update_ms,
                    );
                }
                let mut wrote_new_best_checkpoint = false;
                if selection_score.is_finite() {
                    let candidate = BurnE2eSelectedCheckpoint {
                        step,
                        train_loss: stats.loss,
                        selection_score,
                        validation_contract: checkpoint_quality
                            .as_ref()
                            .and(validation_config.map(e2e_validation_contract)),
                        holdout_mean_psnr_db,
                        holdout_mean_loss,
                        quality_validation: checkpoint_quality.clone(),
                        params: params.detached(),
                        generator: generator.detached(),
                    };
                    if step == config.steps && checkpoint_quality.is_some() {
                        final_checkpoint_candidate = Some(candidate);
                    } else if best_checkpoint.as_ref().is_none_or(|checkpoint| {
                        comparable_selection_score_is_better(
                            candidate.validation_contract.as_ref(),
                            candidate.selection_score,
                            checkpoint.validation_contract.as_ref(),
                            checkpoint.selection_score,
                        )
                    }) {
                        wrote_new_best_checkpoint = checkpoint_quality.is_some();
                        best_checkpoint = Some(candidate);
                    }
                }
                if checkpoint_due {
                    let artifact_hashes = write_e2e_rollout_checkpoint(
                        "current",
                        step,
                        &params.detached(),
                        &generator.detached(),
                        &npa_config,
                        &train_conditions,
                        config,
                    )?;
                    write_e2e_training_checkpoint(
                        step,
                        &base_optimizer,
                        &generator_optimizer,
                        &identity_sampler,
                        &seed_trajectory_counts,
                        particle_pool.as_ref(),
                        prefetch_queue
                            .iter()
                            .map(|batch| batch.indices.clone())
                            .collect(),
                        artifact_hashes.as_ref(),
                        train_examples,
                        config,
                    )?;
                    last_checkpoint_at = Instant::now();
                }
                if wrote_new_best_checkpoint {
                    if let Some(checkpoint) = &best_checkpoint {
                        let _ = write_e2e_rollout_checkpoint(
                            "best",
                            checkpoint.step,
                            &checkpoint.params,
                            &checkpoint.generator,
                            &npa_config,
                            &train_conditions,
                            config,
                        )?;
                    }
                }
                history.push(stats);
                throughput_interval_particle_steps = 0;
                throughput_interval_started = Instant::now();
                if step == config.steps
                    && let Some(quality) = &checkpoint_quality
                    && quality.passed
                {
                    early_stop_step = Some(step);
                    eprintln!(
                        "hyper2d e2e rollout reached validation PSNR threshold at step {step}: composited_psnr={:.3}dB p10={:.3}dB threshold={:.3}dB",
                        quality.aggregate_composited_rgb_psnr_db,
                        quality.p10_composited_rgb_psnr_db,
                        config.validation_psnr_threshold_db,
                    );
                    break;
                }
            }
        }
        let final_validation_config = e2e_final_validation_config(config);
        let final_validation_contract = e2e_validation_contract(final_validation_config);
        if let Some(final_candidate) = final_checkpoint_candidate {
            let mut selected = final_candidate;
            if let Some(mut prior_best) = best_checkpoint.take() {
                if prior_best.validation_contract.as_ref() != Some(&final_validation_contract) {
                    let quality = evaluate_e2e_rollout_quality(
                        &prior_best.params.detached(),
                        &prior_best.generator.detached(),
                        &npa_config,
                        train_examples,
                        holdout_examples,
                        &train_conditions,
                        &holdout_conditions,
                        final_validation_config,
                        &device,
                    )?;
                    if let Some(quality) = quality {
                        quality_validation_evaluations =
                            quality_validation_evaluations.saturating_add(1);
                        quality_validation_elapsed_ms += quality.elapsed_ms;
                        prior_best.selection_score = quality.selection_psnr_db;
                        prior_best.validation_contract = Some(final_validation_contract.clone());
                        prior_best.holdout_mean_psnr_db =
                            Some(quality.aggregate_composited_rgb_psnr_db);
                        prior_best.holdout_mean_loss = Some(quality.mean_total_loss);
                        prior_best.quality_validation = Some(quality);
                    }
                }
                if comparable_selection_score_is_better(
                    prior_best.validation_contract.as_ref(),
                    prior_best.selection_score,
                    selected.validation_contract.as_ref(),
                    selected.selection_score,
                ) {
                    selected = prior_best;
                }
            }
            best_checkpoint = Some(selected);
            if let Some(checkpoint) = &best_checkpoint {
                let _ = write_e2e_rollout_checkpoint(
                    "best",
                    checkpoint.step,
                    &checkpoint.params,
                    &checkpoint.generator,
                    &npa_config,
                    &train_conditions,
                    config,
                )?;
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
        let selected_checkpoint_quality_validation = best_checkpoint
            .as_ref()
            .filter(|checkpoint| {
                checkpoint.validation_contract.as_ref() == Some(&final_validation_contract)
            })
            .and_then(|checkpoint| checkpoint.quality_validation.clone());
        let selected_checkpoint_source = if selected_checkpoint_step == Some(0) {
            "initial_p10_composited_rgb_psnr"
        } else if selected_checkpoint_step == Some(config.steps)
            && selected_checkpoint_quality_validation.is_some()
        {
            "final_common_contract_p10_composited_rgb_psnr"
        } else if selected_checkpoint_holdout_psnr_db.is_some() {
            "best_common_contract_p10_composited_rgb_psnr"
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
        let final_quality_validation_reused_from_selected_checkpoint =
            selected_checkpoint_quality_validation.is_some();
        let quality_validation = if let Some(quality_validation) =
            selected_checkpoint_quality_validation
        {
            Some(quality_validation)
        } else {
            let quality_validation = evaluate_e2e_rollout_quality(
                &params.detached(),
                &generator.detached(),
                &npa_config,
                train_examples,
                holdout_examples,
                &train_conditions,
                &holdout_conditions,
                final_validation_config,
                &device,
            )?;
            if let Some(quality) = &quality_validation {
                quality_validation_evaluations = quality_validation_evaluations.saturating_add(1);
                quality_validation_elapsed_ms += quality.elapsed_ms;
            }
            quality_validation
        };
        let generator_hyper = generator.to_hyper(config)?;
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
        let condition_features_drained_from_examples =
            train_conditions.drained_cpu_features_from_examples()
                || holdout_conditions.drained_cpu_features_from_examples();
        let condition_features_uploaded_as_resident_device_cache =
            train_conditions.is_device_resident() && holdout_conditions.is_device_resident();
        let exposure = identity_sampler.exposure_stats();
        let upstream_reference_trajectories = 240_000.0_f64;
        let upstream_reference_particles = 4_096.0_f64;
        let upstream_equivalent_mean_trajectories =
            exposure.mean * config.rollout_particles as f64 / upstream_reference_particles;
        let mut metrics = serde_json::Map::new();
        metrics.insert("backend".to_string(), json!(format!("{BACKEND}_e2e_rollout")));
        metrics.insert("device".to_string(), json!(DEVICE_LABEL));
        metrics.insert(
            "objective".to_string(),
            json!("target2d_rollout_image_loss_generated_lora"),
        );
        metrics.insert(
            "generator_weight_warm_start".to_string(),
            json!(initial_generator.is_some()),
        );
        metrics.insert(
            "optimizer_state_resumed".to_string(),
            json!(resume_checkpoint.is_some()),
        );
        metrics.insert("resumed_from_step".to_string(), json!(completed_step));
        metrics.insert(
            "conditioner".to_string(),
            json!(generator.kind.artifact_architecture()),
        );
        metrics.insert("adapter_rank".to_string(), json!(config.adapter_rank));
        metrics.insert("adapter_alpha".to_string(), json!(config.adapter_alpha));
        metrics.insert(
            "adapter_parameterization".to_string(),
            json!(if config.canonical_full_rank_lora {
                E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
            } else {
                E2E_HYPER_ADAPTER_FACTORIZED
            }),
        );
        metrics.insert(
            "adapter_effective_output_dims".to_string(),
            json!(if config.canonical_full_rank_lora {
                crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(
                    &npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                )?
                .trainable_parameters
            } else {
                generator.output_dims
            }),
        );
        metrics.insert(
            "adapter_chunk_size".to_string(),
            json!(generator.adapter_chunk_size),
        );
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
            "generator_condition_init_scale".to_string(),
            json!(config.generator_condition_init_scale),
        );
        metrics.insert(
            "generator_output_init_scale".to_string(),
            json!(config.generator_output_init_scale),
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
            "condition_device_cache_max_bytes".to_string(),
            json!(config.condition_device_cache_max_bytes),
        );
        metrics.insert(
            "target_device_cache".to_string(),
            json!({
                "resident": train_target_cache.is_some(),
                "projected_bytes_f32": projected_target_cache_bytes,
                "max_bytes": config.target_device_cache_max_bytes,
                "step_target_upload": train_target_cache.is_none(),
            }),
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
            json!(condition_features_drained_from_examples),
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
        metrics.insert(
            "example_batch_size_effective".to_string(),
            json!(batch_size),
        );
        metrics.insert(
            "example_batch_semantics".to_string(),
            json!("independent_image_conditioned_rollouts"),
        );
        metrics.insert(
            "example_batch_parallel_samples".to_string(),
            json!(batch_size.min(train_examples.len())),
        );
        metrics.insert(
            "identity_sampling".to_string(),
            json!({
                "uniform_fraction": config.sampling_uniform_fraction,
                "priority_fraction": 1.0 - config.sampling_uniform_fraction,
                "priority_ema_beta": config.sampling_priority_ema_beta,
                "priority_min_weight": config.sampling_priority_min_weight,
                "priority_max_weight": config.sampling_priority_max_weight,
                "priority_update_interval": config.sampling_priority_update_interval,
                "trajectory_exposure": exposure,
                "upstream_reference_trajectories_per_identity": upstream_reference_trajectories,
                "upstream_reference_particles_per_trajectory": upstream_reference_particles,
                "upstream_equivalent_mean_4096_particle_trajectories_per_identity": upstream_equivalent_mean_trajectories,
                "upstream_compute_exposure_fraction": upstream_equivalent_mean_trajectories / upstream_reference_trajectories,
            }),
        );
        metrics.insert("rollout_particles".to_string(), json!(config.rollout_particles));
        metrics.insert(
            "rollout_step_min".to_string(),
            json!(config.rollout_step_min),
        );
        metrics.insert("rollout_steps".to_string(), json!(config.rollout_steps));
        metrics.insert(
            "tbptt_chunk_steps".to_string(),
            json!(config.tbptt_chunk_steps),
        );
        metrics.insert(
            "validation_interval".to_string(),
            json!(config.validation_interval),
        );
        metrics.insert(
            "quality_validation_evaluations".to_string(),
            json!(quality_validation_evaluations),
        );
        metrics.insert(
            "quality_validation_elapsed_ms".to_string(),
            json!(quality_validation_elapsed_ms),
        );
        metrics.insert(
            "loss_on_final_chunk_only".to_string(),
            json!(config.loss_on_final_chunk_only),
        );
        metrics.insert(
            "tbptt_loss_mode".to_string(),
            json!(config.tbptt_loss_mode.as_str()),
        );
        metrics.insert(
            "tbptt_intermediate_loss_weight".to_string(),
            json!(config.tbptt_intermediate_loss_weight),
        );
        metrics.insert(
            "tbptt_final_loss_weight".to_string(),
            json!(config.tbptt_final_loss_weight),
        );
        metrics.insert(
            "credit_assignment".to_string(),
            json!(config.credit_assignment.as_str()),
        );
        metrics.insert(
            "task_loss_weight".to_string(),
            json!(config.task_loss_weight),
        );
        metrics.insert(
            "adapter_teacher_weight".to_string(),
            json!(config.adapter_teacher_weight),
        );
        metrics.insert(
            "adapter_teacher_objective".to_string(),
            json!(config.adapter_teacher_objective.as_str()),
        );
        metrics.insert(
            "adapter_teacher_probe_rollout_steps".to_string(),
            json!(config.adapter_teacher_probe_rollout_steps),
        );
        metrics.insert(
            "base_per_parameter_grad_normalization".to_string(),
            json!(config.base_per_parameter_grad_normalization),
        );
        metrics.insert(
            "generator_per_parameter_grad_normalization".to_string(),
            json!(config.generator_per_parameter_grad_normalization),
        );
        metrics.insert(
            "sample_id_table_grad_normalization".to_string(),
            json!(if config.generator_kind == E2eHyperGeneratorKind::SampleIdTable
                && config.generator_per_parameter_grad_normalization
            {
                "per-adapter-component-per-identity"
            } else {
                "not-applicable"
            }),
        );
        metrics.insert(
            "max_full_bptt_particle_steps".to_string(),
            json!(config.max_full_bptt_particle_steps),
        );
        metrics.insert(
            "pre_rollout_steps".to_string(),
            json!(config.pre_rollout_steps),
        );
        metrics.insert(
            "particle_pool".to_string(),
            json!({
                "enabled": config.use_particle_pool,
                "capacity": config.pool_capacity,
                "storage_bytes_f32": config.pool_capacity
                    .saturating_mul(config.rollout_particles)
                    .saturating_mul(npa_config.state_dims.saturating_add(2))
                    .saturating_mul(std::mem::size_of::<f32>()),
                "stored_slots_per_example": config.pool_slots_per_example,
                "rollouts_sampled_per_example": config.rollouts_per_example,
                "inject_seed_interval": config.inject_seed_interval,
                "seed_replacements_per_interval": config.seed_replacements_per_interval,
                "seed_trajectory_interval_per_identity": config.seed_trajectory_interval,
                "mode": "bounded-sample-replica-keyed-device-state-pool",
                "step_readback": false,
                "position_persistence_clamp": [-1.0, 1.0],
                "state_finite_safety_clamp": [-32.0, 32.0],
            }),
        );
        metrics.insert(
            "target2d_loss_backend".to_string(),
            json!(config.target2d_loss_backend.as_str()),
        );
        metrics.insert(
            "target2d_loss_backend_effective".to_string(),
            json!(target2d_loss_backend_effective(direct_config_view(config)).as_str()),
        );
        metrics.insert(
            "perception_backend".to_string(),
            json!(config.perception_backend.as_str()),
        );
        metrics.insert(
            "perception_backend_effective".to_string(),
            json!(perception_backend_effective(direct_config_view(config)).as_str()),
        );
        metrics.insert(
            "perception_cube_adjoint_device_hits".to_string(),
            json!(PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_adjoint_fallback_hits".to_string(),
            json!(PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_forward_device_hits".to_string(),
            json!(PERCEPTION_CUBE_FORWARD_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_forward_fallback_hits".to_string(),
            json!(PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "perception_cube_prepared_reuse_hits".to_string(),
            json!(PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed)),
        );
        let retained_perception_state_bytes_per_step = batch_size
            .saturating_mul(config.rollout_particles)
            .saturating_mul(npa_config.state_dims.saturating_mul(2).saturating_add(4))
            .saturating_mul(std::mem::size_of::<f32>());
        metrics.insert(
            "perception_cube_prepared_vjp".to_string(),
            json!({
                "mode": "retained_raw_state_gradient_and_correction_inverse",
                "additional_bytes_per_rollout_step_f32": retained_perception_state_bytes_per_step,
                "additional_bytes_at_max_full_bptt_horizon_f32": retained_perception_state_bytes_per_step
                    .saturating_mul(config.rollout_steps),
                "neighbor_recompute_in_backward": false,
            }),
        );
        metrics.insert(
            "target2d_cube_adjoint_device_hits".to_string(),
            json!(TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "target2d_cube_adjoint_fallback_hits".to_string(),
            json!(TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_upload_hits".to_string(),
            json!(STOCHASTIC_MASK_UPLOAD_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_device_hits".to_string(),
            json!(STOCHASTIC_MASK_DEVICE_HITS.load(Ordering::Relaxed)),
        );
        metrics.insert(
            "stochastic_mask_backend_effective".to_string(),
            json!("device-random-training-host-seeded-eval"),
        );
        metrics.insert(
            "max_dense_train_particles".to_string(),
            json!(config.max_dense_train_particles),
        );
        metrics.insert(
            "training_graph".to_string(),
            json!(match config.credit_assignment {
                E2eCreditAssignment::FullBptt => {
                    "generated_adapter_fixed_full_rollout_single_loss_single_update"
                }
                E2eCreditAssignment::DetachedTbptt => {
                    "generated_adapter_tbptt_chunked_rollout_state_detach_with_optional_sample_keyed_pool"
                }
            }),
        );
        metrics.insert(
            "shared_base_trainable".to_string(),
            json!(config.shared_base_trainable),
        );
        metrics.insert(
            "shared_base_train_start_step".to_string(),
            json!(config.shared_base_train_start_step),
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
            "final_quality_validation_reused_from_selected_checkpoint".to_string(),
            json!(final_quality_validation_reused_from_selected_checkpoint),
        );
        metrics.insert("early_stop_step".to_string(), json!(early_stop_step));
        metrics.insert(
            "early_stop_reason".to_string(),
            json!(early_stop_step.map(|_| "validation_psnr_threshold")),
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
        metrics.insert(
            "total_optimizer_particle_steps".to_string(),
            json!(total_optimizer_particle_steps),
        );
        metrics.insert(
            "measured_optimizer_training_ms".to_string(),
            json!(measured_optimizer_training_ms),
        );
        metrics.insert(
            "measured_optimizer_particle_steps_per_sec".to_string(),
            json!(total_optimizer_particle_steps as f64
                / (measured_optimizer_training_ms / 1_000.0).max(f64::MIN_POSITIVE)),
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
            generator: generator_hyper,
            quality_validation,
        })
    }

    fn should_write_e2e_checkpoint(
        step: usize,
        last_checkpoint_at: Instant,
        config: BurnE2eRolloutTrainConfig,
    ) -> bool {
        config.checkpoint_dir.is_some()
            && (step == config.steps
                || step.is_multiple_of(config.checkpoint_interval_steps.max(1))
                || last_checkpoint_at.elapsed().as_secs()
                    >= config.checkpoint_interval_seconds.max(1) as u64)
    }

    fn initial_validation_is_checkpoint_comparable(config: BurnE2eRolloutTrainConfig) -> bool {
        config.initial_validation_examples == config.validation_examples
    }

    fn e2e_validation_contract(
        config: BurnE2eRolloutTrainConfig,
    ) -> BurnE2eValidationContract {
        let mut horizons = config.validation_horizons
            [..config.validation_horizon_count.min(config.validation_horizons.len())]
            .iter()
            .copied()
            .filter(|steps| *steps > 0)
            .collect::<Vec<_>>();
        horizons.push(config.validation_steps.max(1));
        horizons.sort_unstable();
        horizons.dedup();
        BurnE2eValidationContract {
            examples: config.validation_examples,
            particles: config.validation_particles,
            horizons,
            selection_horizon_min_steps: config.validation_selection_horizon_min_steps,
        }
    }

    fn comparable_selection_score_is_better(
        candidate_contract: Option<&BurnE2eValidationContract>,
        candidate_score: f32,
        incumbent_contract: Option<&BurnE2eValidationContract>,
        incumbent_score: f32,
    ) -> bool {
        if !candidate_score.is_finite() {
            return false;
        }
        match (candidate_contract, incumbent_contract) {
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (candidate, incumbent) if candidate == incumbent => candidate_score > incumbent_score,
            (Some(_), Some(_)) => false,
            (None, None) => unreachable!("equal optional contracts matched above"),
        }
    }

    fn e2e_final_validation_config(
        mut config: BurnE2eRolloutTrainConfig,
    ) -> BurnE2eRolloutTrainConfig {
        config.validation_examples = config.final_validation_examples;
        config.validation_particles = config.final_validation_particles;
        config.validation_steps = config.final_validation_steps;
        config.validation_horizons = config.final_validation_horizons;
        config.validation_horizon_count = config.final_validation_horizon_count;
        config.validation_selection_horizon_min_steps =
            config.final_validation_selection_horizon_min_steps;
        config
    }

    struct E2eRolloutCheckpointHashes {
        shared_base_sha256: String,
        hyper_sha256: String,
    }

    fn write_e2e_rollout_checkpoint(
        label: &str,
        step: usize,
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        train_conditions: &BurnE2eConditionCache,
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<Option<E2eRolloutCheckpointHashes>> {
        let Some(checkpoint_dir) = config.checkpoint_dir else {
            return Ok(None);
        };
        let checkpoint_dir = Path::new(checkpoint_dir);
        fs::create_dir_all(checkpoint_dir)?;
        let shared_base_output = checkpoint_dir.join(format!("{label}_shared_base.bpk"));
        let hyper_output = checkpoint_dir.join(format!("{label}_hyper_2d.bpk"));
        let metadata_output = checkpoint_dir.join(format!("{label}_metadata.json"));
        let source =
            format!("checkpoint:{BACKEND}:hyper2d-e2e-rollout:label={label}:step={step}");

        let mut model = NpaModel {
            config: npa_config.clone(),
            weights: NpaWeights::zeros(npa_config),
        };
        params.write_to_model(&mut model)?;
        let manifest = BpkModelManifest::from_model(
            &model,
            burn_automata_kernels::HashGridConfig::growing_2d(),
            Some(source.clone()),
        );
        let shared_base_sha256 = crate::import::save_manifest(&shared_base_output, &manifest)?
            .ok_or_else(|| {
                AutomataError::InvalidFormat(
                    "HyperNPA checkpoint base path did not use the BPK format".to_string(),
                )
            })?;

        let mut hyper = generator.to_hyper(config)?;
        hyper.condition_encoder = config.checkpoint_condition_encoder.map(str::to_string);
        hyper.condition_token_count = Some(train_conditions.token_count);
        hyper.condition_embed_dims = Some(train_conditions.embed_dims);
        hyper.condition_token_grid_width = Some(config.dino_token_grid_width);
        hyper.condition_token_grid_height = Some(config.dino_token_grid_height);
        hyper.shared_base_sha256 = Some(shared_base_sha256.clone());
        let hyper_sha256 = save_e2e_hyper_npa_2d(&hyper_output, &hyper)?;

        let metadata = json!({
            "label": label,
            "step": step,
            "backend": format!("{BACKEND}_e2e_rollout"),
            "device": DEVICE_LABEL,
            "source": source,
            "shared_base_output": shared_base_output,
            "shared_base_sha256": shared_base_sha256,
            "hyper_output": hyper_output,
            "hyper_sha256": hyper_sha256,
            "condition_token_count": train_conditions.token_count,
            "condition_embed_dims": train_conditions.embed_dims,
            "condition_token_grid_width": config.dino_token_grid_width,
            "condition_token_grid_height": config.dino_token_grid_height,
            "condition_image_size": config.dino_image_size,
            "condition_alpha_mode": "composite-white",
            "condition_rgb_channels": config.dino_rgb_channels,
            "condition_rgb_channel_scale": config.dino_rgb_channel_scale,
            "condition_alpha_channel": config.dino_alpha_channel,
            "condition_alpha_channel_scale": config.dino_alpha_channel_scale,
            "condition_l2_normalize_features": config.dino_l2_normalize_features,
            "condition_resize_mode": "stretch",
        });
        fs::write(&metadata_output, serde_json::to_vec_pretty(&metadata)?)?;
        eprintln!(
            "hyper2d e2e rollout checkpoint {label} step {step} wrote {} and {}",
            shared_base_output.display(),
            hyper_output.display(),
        );
        Ok(Some(E2eRolloutCheckpointHashes {
            shared_base_sha256,
            hyper_sha256,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn e2e_training_contract_sha256(
        train_examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> String {
        let mut hasher = Sha256::new();
        for example in train_examples {
            hasher.update(example.slug.as_bytes());
            hasher.update([0]);
        }
        hasher.update(
            format!(
                "backend={BACKEND};steps={};batch={};replicas={};particles={};step_min={};step_max={};update_prob={:.9};seed={};seed_mode={:?};pool={}:{}:{};seed_interval={};brush={:.9};loss={}:{}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9};adapter={}:{:.9}:{}:{};generator={}:{}:{}:{:.9}:{:.9}:{:.9};optimizer={:?}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9}:{:.9};condition={}:{}:{}:{}:{}:{}",
                config.steps,
                config.example_batch_size,
                config.rollouts_per_example,
                config.rollout_particles,
                config.rollout_step_min,
                config.rollout_steps,
                config.update_prob,
                config.seed,
                config.seed_mode,
                config.use_particle_pool,
                config.pool_capacity,
                config.pool_slots_per_example,
                config.seed_trajectory_interval,
                config.brush_size,
                config.loss_config.image_size,
                config.loss_config.center,
                config.loss_config.splat_loss_weight,
                config.loss_config.color_loss_weight,
                config.loss_config.density_loss_weight,
                config.loss_config.composited_rgb_loss_weight,
                config.loss_config.displacement_regularizer_weight,
                config.loss_config.overflow_regularizer_weight,
                config.loss_config.bound_regularizer_weight,
                config.adapter_rank,
                config.adapter_alpha,
                config.canonical_full_rank_lora,
                config.generator_kind.artifact_architecture(),
                config.generator_hidden_dims,
                config.adapter_chunk_size,
                config.generator_sample_steps,
                config.generator_output_scale,
                config.generator_condition_init_scale,
                config.generator_output_init_scale,
                config.lr_schedule,
                config.base_optimizer.learning_rate,
                config.generator_optimizer.learning_rate,
                config.base_optimizer.weight_decay,
                config.generator_optimizer.weight_decay,
                config.base_optimizer.grad_clip_norm,
                config.generator_optimizer.grad_clip_norm,
                config.base_optimizer.beta1,
                config.base_optimizer.beta2,
                config.base_optimizer.epsilon,
                config.dino_image_size,
                config.dino_token_grid_width,
                config.dino_token_grid_height,
                config.dino_rgb_channels,
                config.dino_alpha_channel,
                config.condition_device_cache_max_bytes,
            )
            .as_bytes(),
        );
        format!("{:x}", hasher.finalize())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_e2e_training_checkpoint(
        step: usize,
        base_optimizer: &BurnBaseAdamWState,
        generator_optimizer: &BurnE2eGeneratorAdamWState,
        sampler: &E2eIdentitySampler,
        seed_trajectory_counts: &[usize],
        particle_pool: Option<&BurnE2eDeviceParticlePool>,
        pending_batches: Vec<Vec<usize>>,
        artifact_hashes: Option<&E2eRolloutCheckpointHashes>,
        train_examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<()> {
        let Some(checkpoint_dir) = config.checkpoint_dir else {
            return Ok(());
        };
        let mut optimizer_tensors = base_optimizer.snapshots()?;
        optimizer_tensors.extend(generator_optimizer.snapshots()?);
        let checkpoint = E2eTrainingCheckpoint {
            version: E2E_TRAINING_CHECKPOINT_VERSION,
            backend: BACKEND.to_string(),
            contract_sha256: e2e_training_contract_sha256(train_examples, config),
            shared_base_sha256: artifact_hashes
                .map(|hashes| hashes.shared_base_sha256.clone())
                .unwrap_or_default(),
            hyper_sha256: artifact_hashes
                .map(|hashes| hashes.hyper_sha256.clone())
                .unwrap_or_default(),
            completed_step: step,
            train_examples: train_examples.len(),
            rollout_particles: config.rollout_particles,
            rollout_step_min: config.rollout_step_min,
            rollout_steps: config.rollout_steps,
            rollouts_per_example: config.rollouts_per_example,
            base_optimizer_step: base_optimizer.step,
            generator_optimizer_step: generator_optimizer.step,
            optimizer_tensors,
            sampler: sampler.clone(),
            seed_trajectory_counts: seed_trajectory_counts.to_vec(),
            pending_batches,
            particle_pool: particle_pool.map(BurnE2eDeviceParticlePool::snapshot).transpose()?,
        };
        let path = Path::new(checkpoint_dir).join("current_training_state.mpk");
        checkpoint.write_atomic(&path)?;
        eprintln!(
            "hyper2d e2e rollout checkpoint current step {step} wrote {}",
            path.display()
        );
        Ok(())
    }

    fn load_e2e_training_checkpoint(
        path: &str,
        train_examples: &[BurnE2eRolloutExample],
        npa_config: &NpaConfig,
        config: BurnE2eRolloutTrainConfig,
    ) -> AutomataResult<E2eTrainingCheckpoint> {
        let path = Path::new(path);
        let path = if path.is_dir() {
            path.join("current_training_state.mpk")
        } else {
            path.to_path_buf()
        };
        let checkpoint = E2eTrainingCheckpoint::read(&path)?;
        let checkpoint_dir = path.parent().ok_or_else(|| {
            AutomataError::InvalidArgument(format!(
                "training checkpoint {} has no parent directory",
                path.display()
            ))
        })?;
        if !checkpoint.shared_base_sha256.is_empty() {
            let artifact = checkpoint_dir.join("current_shared_base.bpk");
            let actual = crate::import::bpk_payload_sha256(&fs::read(&artifact)?)?;
            if actual != checkpoint.shared_base_sha256 {
                return Err(AutomataError::InvalidFormat(format!(
                    "training checkpoint base BPK hash mismatch for {}; state={} artifact={actual}",
                    artifact.display(),
                    checkpoint.shared_base_sha256,
                )));
            }
        }
        if !checkpoint.hyper_sha256.is_empty() {
            let artifact = checkpoint_dir.join("current_hyper_2d.bpk");
            let actual = crate::hyper::e2e::e2e_hyper_bpk_payload_sha256(&fs::read(&artifact)?)?;
            if actual != checkpoint.hyper_sha256 {
                return Err(AutomataError::InvalidFormat(format!(
                    "training checkpoint hyper BPK hash mismatch for {}; state={} artifact={actual}",
                    artifact.display(),
                    checkpoint.hyper_sha256,
                )));
            }
        }
        if checkpoint.backend != BACKEND
            || (!checkpoint.contract_sha256.is_empty()
                && checkpoint.contract_sha256
                    != e2e_training_contract_sha256(train_examples, config))
            || checkpoint.train_examples != train_examples.len()
            || checkpoint.rollout_particles != config.rollout_particles
            || checkpoint.rollout_step_min != config.rollout_step_min
            || checkpoint.rollout_steps != config.rollout_steps
            || checkpoint.rollouts_per_example != config.rollouts_per_example
            || checkpoint.seed_trajectory_counts.len() != train_examples.len()
            || npa_config.spatial_dims != 2
        {
            return Err(AutomataError::InvalidArgument(format!(
                "training checkpoint {} is incompatible with backend/data/rollout config",
                path.display()
            )));
        }
        if checkpoint.completed_step >= config.steps {
            return Err(AutomataError::InvalidArgument(format!(
                "training checkpoint completed step {} is not below configured total steps {}",
                checkpoint.completed_step, config.steps
            )));
        }
        Ok(checkpoint)
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
            BurnDeviceParticlePool::new(
                config.pool_size.max(config.example_batch_size).max(1),
                particle_count,
                16,
                targets[0].seed_scale,
                config,
                &targets[0].target_rgb.device(),
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
                )?;
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
        pool_update: Option<(&mut BurnDeviceParticlePool, Vec<usize>)>,
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
            if config.loss_on_final_chunk_only && !final_chunk {
                let detached_params = params.detached();
                let adapter_batch =
                    BurnAdapterBatch::from_indices(adapters, indices).detached();
                let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                let (next_x, next_s, _) = rollout_batch_chunk(
                    &detached_params,
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
                    None,
                );
                x = detach3(next_x);
                s = detach3(next_s);
                particle_steps += indices.len() as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
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
                None,
            );
            let loss = target_splat_loss_batch(
                &next_x,
                &next_s,
                targets,
                indices,
                config,
                &adapter_batch,
                displacement,
            )?;
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
                if config.loss_on_final_chunk_only && !final_chunk {
                    let detached_params = params.detached();
                    let detached_adapter = adapters[idx].detached();
                    let displacement = Tensor::<BurnBackend, 1>::zeros([1], device);
                    let (next_x, next_s, _) = rollout_single_chunk(
                        &detached_params,
                        &detached_adapter,
                        target,
                        x,
                        s,
                        config,
                        &mut rng,
                        steps,
                        displacement,
                    );
                    x = detach2(next_x);
                    s = detach2(next_s);
                    particle_steps += target.particle_count as f64 * steps as f64;
                    remaining_steps -= steps;
                    continue;
                }
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
    fn select_rollout_conditions(
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        rollouts_per_example: usize,
    ) -> AutomataResult<(Tensor3, Option<Vec<usize>>)> {
        let replicas = rollouts_per_example.max(1);
        if replicas == 1 || prepared_dino.is_some() || !condition_indices.len().is_multiple_of(replicas)
        {
            return conditions
                .select_prepared(condition_indices, prepared_dino)
                .map(|condition| (condition, None));
        }
        let chunks = condition_indices.chunks(replicas).collect::<Vec<_>>();
        if chunks
            .iter()
            .any(|chunk| chunk.iter().any(|identity| *identity != chunk[0]))
        {
            return conditions
                .select_prepared(condition_indices, prepared_dino)
                .map(|condition| (condition, None));
        }
        let unique = chunks.iter().map(|chunk| chunk[0]).collect::<Vec<_>>();
        let expansion = (0..unique.len())
            .flat_map(|row| std::iter::repeat_n(row, replicas))
            .collect::<Vec<_>>();
        conditions
            .select(&unique)
            .map(|condition| (condition, Some(expansion)))
    }

    #[allow(clippy::too_many_arguments)]
    fn train_e2e_homogeneous_step_tbptt(
        params: &mut BurnBaseParams,
        generator: &mut BurnE2eGeneratorParams,
        base_optimizer: &mut BurnBaseAdamWState,
        generator_optimizer: &mut BurnE2eGeneratorAdamWState,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
        collect_per_example_losses: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
    ) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
        if config.credit_assignment == E2eCreditAssignment::FullBptt {
            return train_e2e_homogeneous_step_full_bptt(
                params,
                generator,
                base_optimizer,
                generator_optimizer,
                npa_config,
                conditions,
                condition_indices,
                prepared_dino,
                targets,
                target_indices,
                config,
                step_seed,
                collect_metrics,
                collect_per_example_losses,
                initial_state,
            );
        }
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "Burn HyperNPA e2e rollout condition/target batch length mismatch",
            )
            .into());
        }
        let batch_len = condition_indices.len();
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "Burn HyperNPA e2e rollout batches require homogeneous particle counts",
            )
            .into());
        };
        let started = Instant::now();
        let direct_config = direct_config_view(config);
        let device = &targets[target_indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(
                targets,
                target_indices,
                particle_count,
                direct_config,
                step_seed,
                device,
            )
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        let mut particle_steps = 0.0_f64;
        if config.pre_rollout_steps > 0 {
            let detached_params = params.detached();
            let detached_generator = generator.detached();
            let (condition, expansion) = select_rollout_conditions(
                conditions,
                condition_indices,
                prepared_dino,
                config.rollouts_per_example,
            )?;
            let adapter_batch = detached_generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = detached_generator
                .condition_control_batch(condition.clone(), config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, _) = rollout_batch_chunk(
                &detached_params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                config.pre_rollout_steps,
                displacement,
                condition_control.as_ref(),
            );
            x = detach3(next_x);
            s = detach3(next_s);
            particle_steps +=
                batch_len as f64 * particle_count as f64 * config.pre_rollout_steps as f64;
        }
        let chunk_steps = tbptt_chunk_steps(direct_config);
        let rollout_steps = sampled_training_rollout_steps(direct_config, step_seed);
        let mut loss_sum = collect_metrics.then_some(0.0_f32);
        let mut loss_weight_sum =
            (collect_metrics || collect_per_example_losses).then_some(0.0_f32);
        let mut per_example_loss_sum =
            collect_per_example_losses.then(|| vec![0.0_f32; batch_len]);
        let mut base_grad_norm_sum = 0.0_f32;
        let mut base_grad_scale_sum = 0.0_f32;
        let mut generator_grad_norm_sum = 0.0_f32;
        let mut generator_grad_scale_sum = 0.0_f32;
        let mut grad_metric_chunks = 0usize;
        let (condition, expansion) = select_rollout_conditions(
            conditions,
            condition_indices,
            prepared_dino,
            config.rollouts_per_example,
        )?;
        let mut remaining_steps = rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            let final_chunk = remaining_steps <= chunk_steps;
            let loss_weight = e2e_chunk_loss_weight(config, final_chunk);
            if loss_weight <= 0.0 {
                let detached_params = params.detached();
                let detached_generator = generator.detached();
                let adapter_batch = detached_generator
                    .adapter_batch(condition.clone(), npa_config, config)
                    .select_rows_or_identity(expansion.as_deref());
                let condition_control = detached_generator
                    .condition_control_batch(condition.clone(), config)
                    .map(|control| control.select_rows_or_identity(expansion.as_deref()));
                let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
                let (next_x, next_s, _) = rollout_batch_chunk(
                    &detached_params,
                    &adapter_batch,
                    targets,
                    target_indices,
                    x,
                    s,
                    direct_config,
                    particle_count,
                    &mut rng,
                    steps,
                    displacement,
                    condition_control.as_ref(),
                );
                x = detach3(next_x);
                s = detach3(next_s);
                particle_steps += batch_len as f64 * particle_count as f64 * steps as f64;
                remaining_steps -= steps;
                continue;
            }
            let adapter_batch = generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = generator
                .condition_control_batch(condition.clone(), config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, displacement) = rollout_batch_chunk(
                params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                steps,
                displacement,
                condition_control.as_ref(),
            );
            let loss = target_splat_loss_batch_vector_selected(
                &next_x,
                &next_s,
                targets,
                target_indices,
                direct_config,
                &adapter_batch,
                displacement,
            )?;
            if collect_metrics || collect_per_example_losses {
                let scalars = loss_vector_scalars(loss.clone())?;
                if let Some(loss_sum) = loss_sum.as_mut() {
                    for value in &scalars {
                        *loss_sum += value.total * loss_weight;
                    }
                }
                if let Some(per_example_loss_sum) = per_example_loss_sum.as_mut() {
                    for (sum, value) in per_example_loss_sum.iter_mut().zip(&scalars) {
                        *sum += value.total * loss_weight;
                    }
                }
            }
            if let Some(loss_weight_sum) = loss_weight_sum.as_mut() {
                *loss_weight_sum += loss_weight;
            }
            let mut grads = loss
                .total
                .sum()
                .mul_scalar(loss_weight)
                .div_scalar(batch_len as f32)
                .backward();
            let (base_grad_norm, base_grad_scale) = if config.shared_base_trainable {
                params.apply_adamw(
                    &mut grads,
                    base_optimizer,
                    config.base_optimizer,
                    config.base_per_parameter_grad_normalization,
                    collect_metrics,
                )?
            } else {
                (0.0, 1.0)
            };
            let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
                &mut grads,
                generator_optimizer,
                config.generator_optimizer,
                config.generator_per_parameter_grad_normalization,
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
            particle_steps += batch_len as f64 * particle_count as f64 * steps as f64;
            remaining_steps -= steps;
        }
        let elapsed = started.elapsed();
        let grad_metric_chunks = grad_metric_chunks.max(1);
        let loss_weight_count = loss_weight_sum.unwrap_or(1.0).max(f32::MIN_POSITIVE);
        let per_example_losses = per_example_loss_sum.map(|losses| {
            losses
                .into_iter()
                .map(|loss| loss / loss_weight_count)
                .collect()
        });
        let particle_steps_per_sec =
            particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok(BurnE2eStepOutput {
            history: BurnE2eRolloutHistoryEntry {
                step: 0,
                loss: loss_sum.map_or(0.0, |value| {
                    value / batch_len as f32 / loss_weight_count
                }),
                task_loss: loss_sum.map_or(0.0, |value| {
                    value / batch_len as f32 / loss_weight_count
                }),
                adapter_teacher_loss: 0.0,
                learning_rate_scale: 1.0,
                base_learning_rate: config.base_optimizer.learning_rate,
                generator_learning_rate: config.generator_optimizer.learning_rate,
                holdout_mean_psnr_db: None,
                holdout_mean_loss: None,
                base_grad_norm: base_grad_norm_sum / grad_metric_chunks as f32,
                base_grad_scale: base_grad_scale_sum / grad_metric_chunks as f32,
                generator_grad_norm: generator_grad_norm_sum / grad_metric_chunks as f32,
                generator_grad_scale: generator_grad_scale_sum / grad_metric_chunks as f32,
                examples_seen: batch_len,
                pool_seed_replacements: 0,
                particle_steps_per_sec,
                dense_pair_interactions_per_sec: particle_steps_per_sec * particle_count as f64,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                condition_adapter_ms: 0.0,
                rollout_loss_ms: 0.0,
                backward_update_ms: 0.0,
            },
            particle_steps: particle_steps.round().max(0.0) as u64,
            final_x: x,
            final_s: s,
            per_example_losses,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_e2e_homogeneous_step_full_bptt(
        params: &mut BurnBaseParams,
        generator: &mut BurnE2eGeneratorParams,
        base_optimizer: &mut BurnBaseAdamWState,
        generator_optimizer: &mut BurnE2eGeneratorAdamWState,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        condition_indices: &[usize],
        prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        step_seed: u64,
        collect_metrics: bool,
        collect_per_example_losses: bool,
        initial_state: Option<(Tensor3, Tensor3)>,
    ) -> Result<BurnE2eStepOutput, Box<dyn std::error::Error>> {
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "Burn HyperNPA full-BPTT condition/target batch length mismatch",
            )
            .into());
        }
        let batch_len = condition_indices.len();
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "Burn HyperNPA full-BPTT requires homogeneous particle counts",
            )
            .into());
        };
        let started = Instant::now();
        let direct_config = direct_config_view(config);
        let rollout_steps = sampled_training_rollout_steps(direct_config, step_seed);
        let particle_steps = batch_len
            .saturating_mul(particle_count)
            .saturating_mul(rollout_steps);
        if particle_steps > config.max_full_bptt_particle_steps {
            return Err(std::io::Error::other(format!(
                "HyperNPA full-BPTT runtime preflight rejected {particle_steps} particle-steps, above configured cap {}",
                config.max_full_bptt_particle_steps
            ))
            .into());
        }
        let device = &targets[target_indices[0]].target_rgb.device();
        let (mut x, mut s) = initial_state.unwrap_or_else(|| {
            seed_batch_tensors(
                targets,
                target_indices,
                particle_count,
                direct_config,
                step_seed,
                device,
            )
        });
        let mut rng = StdRng::seed_from_u64(step_seed ^ 0x005e_ed2d);
        if config.pre_rollout_steps > 0 {
            let detached_params = params.detached();
            let detached_generator = generator.detached();
            let (condition, expansion) = select_rollout_conditions(
                conditions,
                condition_indices,
                prepared_dino,
                config.rollouts_per_example,
            )?;
            let adapter = detached_generator
                .adapter_batch(condition.clone(), npa_config, config)
                .select_rows_or_identity(expansion.as_deref());
            let condition_control = detached_generator
                .condition_control_batch(condition, config)
                .map(|control| control.select_rows_or_identity(expansion.as_deref()));
            let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
            let (next_x, next_s, _) = rollout_batch_chunk(
                &detached_params,
                &adapter,
                targets,
                target_indices,
                x,
                s,
                direct_config,
                particle_count,
                &mut rng,
                config.pre_rollout_steps,
                displacement,
                condition_control.as_ref(),
            );
            x = detach3(next_x);
            s = detach3(next_s);
        }

        if collect_metrics {
            sync_training_device(device)?;
        }
        let condition_started = Instant::now();
        let (condition, expansion) = select_rollout_conditions(
            conditions,
            condition_indices,
            prepared_dino,
            config.rollouts_per_example,
        )?;
        let adapter = generator
            .adapter_batch(condition.clone(), npa_config, config)
            .select_rows_or_identity(expansion.as_deref());
        let condition_control = generator
            .condition_control_batch(condition, config)
            .map(|control| control.select_rows_or_identity(expansion.as_deref()));
        let teacher_vector = conditions.select_teacher(condition_indices);
        let teacher_adapter = teacher_vector.clone().map(|teacher| {
            BurnAdapterBatch::from_parameter_vector(
                teacher,
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            )
        });
        let teacher_probe_features = if config.adapter_teacher_weight > 0.0
            && config.adapter_teacher_objective != E2eAdapterTeacherObjective::ParameterMse
        {
            let teacher_adapter = teacher_adapter
                .as_ref()
                .expect("functional teacher objective requires teacher adapters");
            let max_probe_steps = config.adapter_teacher_probe_rollout_steps;
            let probe_steps = if max_probe_steps == 0 {
                0
            } else {
                1 + step_seed as usize % max_probe_steps
            };
            let (probe_x, probe_s) = if probe_steps == 0 {
                (detach3(x.clone()), detach3(s.clone()))
            } else {
                let mut teacher_rng = rng.clone();
                let (probe_x, probe_s, _) = rollout_batch_chunk(
                    &params.detached(),
                    teacher_adapter,
                    targets,
                    target_indices,
                    detach3(x.clone()),
                    detach3(s.clone()),
                    direct_config,
                    particle_count,
                    &mut teacher_rng,
                    probe_steps,
                    Tensor::<BurnBackend, 1>::zeros([batch_len], device),
                    None,
                );
                (detach3(probe_x), detach3(probe_s))
            };
            Some(rollout_dense_perception_batch(
                &probe_x,
                &probe_s,
                direct_config,
            ))
        } else {
            None
        };
        if collect_metrics {
            sync_training_device(device)?;
        }
        let condition_adapter_ms = collect_metrics
            .then(|| condition_started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        let rollout_started = Instant::now();
        let displacement = Tensor::<BurnBackend, 1>::zeros([batch_len], device);
        let (next_x, next_s, displacement) = rollout_batch_chunk(
            params,
            &adapter,
            targets,
            target_indices,
            x,
            s,
            direct_config,
            particle_count,
            &mut rng,
            rollout_steps,
            displacement,
            condition_control.as_ref(),
        );
        let loss = target_splat_loss_batch_vector_selected(
            &next_x,
            &next_s,
            targets,
            target_indices,
            direct_config,
            &adapter,
            displacement.clone(),
        )?;
        let task_objective = loss.total.clone().sum().div_scalar(batch_len.max(1) as f32);
        let teacher_objective = teacher_vector.map(|teacher| {
            let generated_vector = adapter.to_parameter_vector();
            let parameter_delta = generated_vector - teacher.clone();
            let parameter_mse = parameter_delta.clone().mul(parameter_delta).mean();
            if config.adapter_teacher_objective == E2eAdapterTeacherObjective::ParameterMse {
                return parameter_mse;
            }

            let teacher_adapter = teacher_adapter
                .as_ref()
                .expect("functional teacher objective requires teacher adapters");
            let probes = teacher_probe_features
                .as_ref()
                .expect("functional teacher objective prepared perception probes")
                .clone();
            let generated_update = params.forward_adapter_batch(probes.clone(), &adapter);
            let teacher_update = detach3(
                params
                    .detached()
                    .forward_adapter_batch(probes, teacher_adapter),
            );
            let functional_delta = generated_update - teacher_update;
            let functional_mse = functional_delta.clone().mul(functional_delta).mean();
            if config.adapter_teacher_objective == E2eAdapterTeacherObjective::Hybrid {
                functional_mse
                    + parameter_mse.mul_scalar(FUNCTIONAL_TEACHER_PARAMETER_AUX_WEIGHT)
            } else {
                functional_mse
            }
        });
        let teacher_loss_value = if collect_metrics {
            teacher_objective
                .as_ref()
                .map(|teacher| teacher.clone().inner().into_scalar())
                .unwrap_or_default()
        } else {
            0.0
        };
        let weighted_task = task_objective.mul_scalar(config.task_loss_weight.max(0.0));
        let objective = teacher_objective.map_or(weighted_task.clone(), |teacher| {
            weighted_task + teacher.mul_scalar(config.adapter_teacher_weight.max(0.0))
        });
        if collect_metrics {
            sync_training_device(device)?;
        }
        let rollout_loss_ms = collect_metrics
            .then(|| rollout_started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        let loss_scalars = if collect_per_example_losses {
            Some(loss_vector_scalars(loss.clone())?)
        } else {
            None
        };
        let task_loss_value = loss_scalars.as_ref().map_or(0.0, |losses| {
            losses.iter().map(|value| value.total).sum::<f32>() / batch_len.max(1) as f32
        });
        let loss_value = task_loss_value * config.task_loss_weight.max(0.0)
            + teacher_loss_value * config.adapter_teacher_weight.max(0.0);
        let backward_started = Instant::now();
        let mut grads = objective.backward();
        let (base_grad_norm, base_grad_scale) = if config.shared_base_trainable {
            params.apply_adamw(
                &mut grads,
                base_optimizer,
                config.base_optimizer,
                config.base_per_parameter_grad_normalization,
                collect_metrics,
            )?
        } else {
            (0.0, 1.0)
        };
        let (generator_grad_norm, generator_grad_scale) = generator.apply_adamw(
            &mut grads,
            generator_optimizer,
            config.generator_optimizer,
            config.generator_per_parameter_grad_normalization,
            collect_metrics,
        )?;
        if collect_metrics {
            sync_training_device(device)?;
        }
        let backward_update_ms = collect_metrics
            .then(|| backward_started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or_default();
        let elapsed = started.elapsed();
        let measured_particle_steps = batch_len as f64
            * particle_count as f64
            * (rollout_steps + config.pre_rollout_steps) as f64;
        let particle_steps_per_sec =
            measured_particle_steps / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok(BurnE2eStepOutput {
            history: BurnE2eRolloutHistoryEntry {
                step: 0,
                loss: loss_value,
                task_loss: task_loss_value,
                adapter_teacher_loss: teacher_loss_value,
                learning_rate_scale: 1.0,
                base_learning_rate: config.base_optimizer.learning_rate,
                generator_learning_rate: config.generator_optimizer.learning_rate,
                holdout_mean_psnr_db: None,
                holdout_mean_loss: None,
                base_grad_norm,
                base_grad_scale,
                generator_grad_norm,
                generator_grad_scale,
                examples_seen: batch_len,
                pool_seed_replacements: 0,
                particle_steps_per_sec,
                dense_pair_interactions_per_sec: particle_steps_per_sec * particle_count as f64,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                condition_adapter_ms,
                rollout_loss_ms,
                backward_update_ms,
            },
            particle_steps: measured_particle_steps.round().max(0.0) as u64,
            final_x: detach3(next_x),
            final_s: detach3(next_s),
            per_example_losses: loss_scalars.map(|losses| {
                losses.into_iter().map(|value| value.total).collect()
            }),
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

    fn e2e_chunk_loss_weight(config: BurnE2eRolloutTrainConfig, final_chunk: bool) -> f32 {
        let mode = if config.loss_on_final_chunk_only {
            E2eTbpttLossMode::FinalOnly
        } else {
            config.tbptt_loss_mode
        };
        match mode {
            E2eTbpttLossMode::AllChunks => 1.0,
            E2eTbpttLossMode::FinalOnly => {
                if final_chunk {
                    1.0
                } else {
                    0.0
                }
            }
            E2eTbpttLossMode::EndpointWeighted => {
                if final_chunk {
                    config.tbptt_final_loss_weight.max(0.0)
                } else {
                    config.tbptt_intermediate_loss_weight.max(0.0)
                }
            }
        }
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
        let mut horizons = config.validation_horizons
            [..config.validation_horizon_count.min(config.validation_horizons.len())]
            .iter()
            .copied()
            .filter(|&steps| steps > 0)
            .collect::<Vec<_>>();
        horizons.push(config.validation_steps.max(1));
        horizons.sort_unstable();
        horizons.dedup();

        let mut final_report = None::<BurnE2eRolloutQualityReport>;
        let mut summaries = Vec::with_capacity(horizons.len());
        let mut total_elapsed_ms = 0.0_f64;
        let mut total_particle_steps = 0.0_f64;
        let mut total_adapter_batches = 0usize;
        for &steps in &horizons {
            let horizon_config = BurnE2eRolloutTrainConfig {
                validation_steps: steps,
                validation_horizon_count: 0,
                ..config
            };
            let Some(report) = evaluate_e2e_rollout_quality_single(
                params,
                generator,
                npa_config,
                train_examples,
                holdout_examples,
                train_conditions,
                holdout_conditions,
                horizon_config,
                device,
            )? else {
                return Ok(None);
            };
            total_elapsed_ms += report.elapsed_ms;
            total_particle_steps += report.particle_steps;
            total_adapter_batches = total_adapter_batches.saturating_add(report.adapter_batches);
            summaries.push(BurnE2eRolloutHorizonSummary {
                rollout_steps: steps,
                aggregate_composited_rgb_psnr_db: report.aggregate_composited_rgb_psnr_db,
                p10_composited_rgb_psnr_db: report.p10_composited_rgb_psnr_db,
                min_composited_rgb_psnr_db: report.min_composited_rgb_psnr_db,
                target_point_splat_p10_composited_rgb_psnr_db: report
                    .target_point_splat_p10_composited_rgb_psnr_db,
                p10_gap_to_target_point_splat_db: report.p10_gap_to_target_point_splat_db,
                aggregate_density_psnr_db: report.aggregate_density_psnr_db,
                mean_density_soft_iou: report.mean_density_soft_iou,
                condition_shuffle_composited_psnr_gap_db: report
                    .condition_shuffle_composited_psnr_gap_db,
                generated_adapter_composited_psnr_gain_db: report
                    .generated_adapter_composited_psnr_gain_db,
                mean_passed: report.mean_passed,
                all_examples_passed: report.all_examples_passed,
                conditional_control_passed: report.conditional_control_passed,
                passed: report.passed,
            });
            if steps == config.validation_steps.max(1) {
                final_report = Some(report);
            }
        }

        let mut report = final_report.expect("final validation horizon is always evaluated");
        let selection_summaries = summaries
            .iter()
            .filter(|summary| {
                summary.rollout_steps >= config.validation_selection_horizon_min_steps
            })
            .collect::<Vec<_>>();
        debug_assert!(!selection_summaries.is_empty());
        let selection_psnr_db = selection_summaries
            .iter()
            .map(|summary| summary.p10_composited_rgb_psnr_db)
            .fold(f32::INFINITY, f32::min);
        let peak_horizon_p10_composited_rgb_psnr_db = summaries
            .iter()
            .map(|summary| summary.p10_composited_rgb_psnr_db)
            .fold(f32::NEG_INFINITY, f32::max);
        let final_horizon_p10_composited_rgb_psnr_db = report.p10_composited_rgb_psnr_db;
        report.selection_metric = "min-horizon-p10-composited-rgb-psnr";
        report.selection_psnr_db = selection_psnr_db;
        report.selection_horizon_min_steps = config.validation_selection_horizon_min_steps;
        report.passed = selection_summaries.iter().all(|summary| summary.passed)
            && selection_psnr_db >= config.validation_psnr_threshold_db;
        report.mean_passed = selection_summaries
            .iter()
            .all(|summary| summary.mean_passed);
        report.all_examples_passed = selection_summaries
            .iter()
            .all(|summary| summary.all_examples_passed);
        report.conditional_control_passed = selection_summaries
            .iter()
            .all(|summary| summary.conditional_control_passed);
        report.elapsed_ms = total_elapsed_ms;
        report.particle_steps = total_particle_steps;
        report.particle_steps_per_sec = total_particle_steps
            / (total_elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
        report.dense_pair_interactions_per_sec =
            report.particle_steps_per_sec * config.validation_particles as f64;
        report.adapter_batches = total_adapter_batches;
        report.horizon_summaries = summaries;
        report.peak_horizon_p10_composited_rgb_psnr_db =
            peak_horizon_p10_composited_rgb_psnr_db;
        report.final_horizon_p10_composited_rgb_psnr_db =
            final_horizon_p10_composited_rgb_psnr_db;
        report.peak_to_final_p10_drop_db =
            peak_horizon_p10_composited_rgb_psnr_db - final_horizon_p10_composited_rgb_psnr_db;
        Ok(Some(report))
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_e2e_rollout_quality_single(
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
        let targets = burn_e2e_targets_for_indices_with_runtime(
            examples,
            &indices,
            config,
            device,
            Some(config.validation_particles),
            Some(config.validation_update_prob),
        )?;
        let target_indices = (0..targets.len()).collect::<Vec<_>>();
        let eval_batch_size = normalized_eval_batch_size(eval_config.eval_batch_size, indices.len());
        let mut entries = Vec::with_capacity(indices.len());
        let adapter_parameter_count = NpaLowRankAdapter::parameter_count_for_config(
            npa_config,
            config.adapter_rank,
        );
        let mut adapter_parameter_rows = Vec::with_capacity(indices.len());
        let mut adapter_batches = 0usize;
        for (condition_chunk, target_chunk) in indices
            .chunks(eval_batch_size)
            .zip(target_indices.chunks(eval_batch_size))
        {
            adapter_batches += 1;
            let quality = batch_e2e_eval_quality(
                params,
                generator,
                npa_config,
                conditions,
                &targets,
                condition_chunk,
                target_chunk,
                config,
                eval_config,
                config.validation_seed,
                device,
                E2eEvalConditionMode::Generated,
            )?;
            let adapter_values = tensor_vec(
                quality
                    .adapter_vector
                    .clone()
                    .expect("generated quality batch contains adapter vectors")
                    .inner(),
            )?;
            if adapter_values.len() != condition_chunk.len() * adapter_parameter_count {
                return Err(std::io::Error::other(
                    "HyperNPA e2e adapter diagnostics readback length mismatch",
                )
                .into());
            }
            adapter_parameter_rows.extend(
                adapter_values
                    .chunks_exact(adapter_parameter_count)
                    .map(<[f32]>::to_vec),
            );
            let losses = loss_vector_scalars(quality.loss)?;
            let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
            let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
            let foreground_rgb_mses = tensor1_vec(quality.foreground_rgb_mse.inner())?;
            let density_mses = tensor1_vec(quality.density_mse.inner())?;
            let density_soft_ious = tensor1_vec(quality.density_soft_iou.inner())?;
            if [
                losses.len(),
                render_rgb_mses.len(),
                composited_rgb_mses.len(),
                foreground_rgb_mses.len(),
                density_mses.len(),
                density_soft_ious.len(),
            ]
            .into_iter()
            .any(|len| len != condition_chunk.len())
            {
                return Err(std::io::Error::other(
                    "HyperNPA e2e quality readback length mismatch",
                )
                .into());
            }
            for local in 0..condition_chunk.len() {
                let idx = condition_chunk[local];
                let loss = losses[local];
                let render_rgb_mse =
                    finite_scalar("HyperNPA e2e render RGB MSE", render_rgb_mses[local])?;
                let render_rgb_psnr_db = psnr_db_from_mse(render_rgb_mse);
                let composited_rgb_mse = finite_scalar(
                    "HyperNPA e2e composited RGB MSE",
                    composited_rgb_mses[local],
                )?;
                let composited_rgb_psnr_db = psnr_db_from_mse(composited_rgb_mse);
                let foreground_rgb_mse = finite_scalar(
                    "HyperNPA e2e foreground RGB MSE",
                    foreground_rgb_mses[local],
                )?;
                let foreground_rgb_psnr_db = psnr_db_from_mse(foreground_rgb_mse);
                let density_mse =
                    finite_scalar("HyperNPA e2e density MSE", density_mses[local])?;
                let density_psnr_db = psnr_db_from_mse(density_mse);
                let density_soft_iou = finite_scalar(
                    "HyperNPA e2e density soft IoU",
                    density_soft_ious[local],
                )?;
                entries.push(BurnE2eRolloutQualityEntry {
                    slug: examples[idx].slug.clone(),
                    total_loss: loss.total,
                    splat_loss: loss.splat,
                    color_loss: loss.color,
                    density_loss: loss.density,
                    render_rgb_mse,
                    render_rgb_psnr_db,
                    composited_rgb_mse,
                    composited_rgb_psnr_db,
                    foreground_rgb_mse,
                    foreground_rgb_psnr_db,
                    density_mse,
                    density_psnr_db,
                    density_soft_iou,
                    passed: composited_rgb_psnr_db >= config.validation_psnr_threshold_db,
                });
            }
        }
        let (mean_condition_shuffle_render_rgb_psnr_db, condition_shuffle_composited_rgb_psnr_db) = if indices.len() > 1 {
            let mut shuffled_condition_indices = indices.clone();
            shuffled_condition_indices.rotate_left(1);
            let mut psnr_sum = 0.0_f32;
            let mut composited_mse_sum = 0.0_f32;
            let mut psnr_count = 0usize;
            for (condition_chunk, target_chunk) in shuffled_condition_indices
                .chunks(eval_batch_size)
                .zip(target_indices.chunks(eval_batch_size))
            {
                let quality = batch_e2e_eval_quality(
                    params,
                    generator,
                    npa_config,
                    conditions,
                    &targets,
                    condition_chunk,
                    target_chunk,
                    config,
                    eval_config,
                    config.validation_seed,
                    device,
                    E2eEvalConditionMode::Generated,
                )?;
                let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
                let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
                for (render_rgb_mse, composited_rgb_mse) in
                    render_rgb_mses.into_iter().zip(composited_rgb_mses)
                {
                    let render_rgb_mse =
                        finite_scalar("HyperNPA e2e shuffled-condition render RGB MSE", render_rgb_mse)?;
                    let composited_rgb_mse = finite_scalar(
                        "HyperNPA e2e shuffled-condition composited RGB MSE",
                        composited_rgb_mse,
                    )?;
                    psnr_sum += psnr_db_from_mse(render_rgb_mse);
                    composited_mse_sum += composited_rgb_mse;
                    psnr_count += 1;
                }
            }
            if psnr_count > 0 {
                (
                    Some(psnr_sum / psnr_count as f32),
                    Some(psnr_db_from_mse(composited_mse_sum / psnr_count as f32)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let mut base_only_composited_mse_sum = 0.0_f32;
        let mut base_only_density_mse_sum = 0.0_f32;
        let mut base_only_density_soft_iou_sum = 0.0_f32;
        let mut base_only_count = 0usize;
        for (condition_chunk, target_chunk) in indices
            .chunks(eval_batch_size)
            .zip(target_indices.chunks(eval_batch_size))
        {
            let quality = batch_e2e_eval_quality(
                params,
                generator,
                npa_config,
                conditions,
                &targets,
                condition_chunk,
                target_chunk,
                config,
                eval_config,
                config.validation_seed,
                device,
                E2eEvalConditionMode::BaseOnly,
            )?;
            let composited_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
            let density_mses = tensor1_vec(quality.density_mse.inner())?;
            let density_soft_ious = tensor1_vec(quality.density_soft_iou.inner())?;
            for local in 0..composited_mses.len() {
                base_only_composited_mse_sum += finite_scalar(
                    "HyperNPA e2e base-only composited RGB MSE",
                    composited_mses[local],
                )?;
                base_only_density_mse_sum += finite_scalar(
                    "HyperNPA e2e base-only density MSE",
                    density_mses[local],
                )?;
                base_only_density_soft_iou_sum += finite_scalar(
                    "HyperNPA e2e base-only density soft IoU",
                    density_soft_ious[local],
                )?;
                base_only_count += 1;
            }
        }
        let mut dino_nearest_teacher_entries = Vec::new();
        let (
            dino_nearest_teacher_render_rgb_psnr_db,
            dino_nearest_teacher_composited_rgb_psnr_db,
        ) = if split == "holdout" && train_conditions.teacher_vectors.is_some() {
            let nearest = train_conditions.nearest_rows(conditions, &indices)?;
            let mut render_psnr_sum = 0.0_f32;
            let mut composited_mse_sum = 0.0_f32;
            let mut count = 0usize;
            for start in (0..indices.len()).step_by(eval_batch_size) {
                let end = (start + eval_batch_size).min(indices.len());
                let condition_chunk = &indices[start..end];
                let target_chunk = &target_indices[start..end];
                let nearest_chunk = &nearest[start..end];
                let nearest_indices = nearest_chunk
                    .iter()
                    .map(|(idx, _)| *idx)
                    .collect::<Vec<_>>();
                let teacher_vectors = train_conditions
                    .select_teacher(&nearest_indices)
                    .expect("nearest-teacher validation requires train teacher adapters");
                let quality = batch_e2e_eval_quality(
                    params,
                    generator,
                    npa_config,
                    conditions,
                    &targets,
                    condition_chunk,
                    target_chunk,
                    config,
                    eval_config,
                    config.validation_seed,
                    device,
                    E2eEvalConditionMode::ExplicitAdapter(teacher_vectors),
                )?;
                let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
                let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
                for local in 0..condition_chunk.len() {
                    let render_rgb_mse = finite_scalar(
                        "HyperNPA e2e nearest-teacher render RGB MSE",
                        render_rgb_mses[local],
                    )?;
                    let composited_rgb_mse = finite_scalar(
                        "HyperNPA e2e nearest-teacher composited RGB MSE",
                        composited_rgb_mses[local],
                    )?;
                    let (train_idx, condition_l2_distance) = nearest_chunk[local];
                    let render_rgb_psnr_db = psnr_db_from_mse(render_rgb_mse);
                    let composited_rgb_psnr_db = psnr_db_from_mse(composited_rgb_mse);
                    render_psnr_sum += render_rgb_psnr_db;
                    composited_mse_sum += composited_rgb_mse;
                    count += 1;
                    dino_nearest_teacher_entries.push(BurnE2eNearestTeacherEntry {
                        holdout_slug: examples[condition_chunk[local]].slug.clone(),
                        train_slug: train_examples[train_idx].slug.clone(),
                        condition_l2_distance,
                        render_rgb_psnr_db,
                        composited_rgb_psnr_db,
                    });
                }
            }
            if count > 0 {
                (
                    Some(render_psnr_sum / count as f32),
                    Some(psnr_db_from_mse(composited_mse_sum / count as f32)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let examples_count = entries.len();
        if examples_count == 0 {
            return Ok(None);
        }
        let target_point_splat_mses = target_point_splat_composited_mses(
            &targets,
            eval_config,
            eval_batch_size,
            device,
        )?;
        let target_point_splat_aggregate_composited_rgb_psnr_db = psnr_db_from_mse(
            target_point_splat_mses.iter().sum::<f32>()
                / target_point_splat_mses.len().max(1) as f32,
        );
        let mut target_point_splat_psnrs = target_point_splat_mses
            .into_iter()
            .map(psnr_db_from_mse)
            .collect::<Vec<_>>();
        target_point_splat_psnrs.sort_by(f32::total_cmp);
        let target_point_splat_p10_composited_rgb_psnr_db =
            sorted_percentile(&target_point_splat_psnrs, 0.1);
        let mut mean_total_loss = 0.0_f32;
        let mut mean_splat_loss = 0.0_f32;
        let mut mean_color_loss = 0.0_f32;
        let mut mean_density_loss = 0.0_f32;
        let mut mean_render_rgb_mse = 0.0_f32;
        let mut mean_render_rgb_psnr_db = 0.0_f32;
        let mut min_render_rgb_psnr_db = f32::INFINITY;
        let mut max_render_rgb_psnr_db = f32::NEG_INFINITY;
        let mut aggregate_composited_rgb_mse = 0.0_f32;
        let mut aggregate_foreground_rgb_mse = 0.0_f32;
        let mut aggregate_density_mse = 0.0_f32;
        let mut mean_density_soft_iou = 0.0_f32;
        let mut composited_psnrs = Vec::with_capacity(examples_count);
        for entry in &entries {
            mean_total_loss += entry.total_loss;
            mean_splat_loss += entry.splat_loss;
            mean_color_loss += entry.color_loss;
            mean_density_loss += entry.density_loss;
            mean_render_rgb_mse += entry.render_rgb_mse;
            mean_render_rgb_psnr_db += entry.render_rgb_psnr_db;
            min_render_rgb_psnr_db = min_render_rgb_psnr_db.min(entry.render_rgb_psnr_db);
            max_render_rgb_psnr_db = max_render_rgb_psnr_db.max(entry.render_rgb_psnr_db);
            aggregate_composited_rgb_mse += entry.composited_rgb_mse;
            aggregate_foreground_rgb_mse += entry.foreground_rgb_mse;
            aggregate_density_mse += entry.density_mse;
            mean_density_soft_iou += entry.density_soft_iou;
            composited_psnrs.push(entry.composited_rgb_psnr_db);
        }
        let scale = 1.0 / examples_count as f32;
        mean_total_loss *= scale;
        mean_splat_loss *= scale;
        mean_color_loss *= scale;
        mean_density_loss *= scale;
        mean_render_rgb_mse *= scale;
        mean_render_rgb_psnr_db *= scale;
        aggregate_composited_rgb_mse *= scale;
        aggregate_foreground_rgb_mse *= scale;
        aggregate_density_mse *= scale;
        mean_density_soft_iou *= scale;
        composited_psnrs.sort_by(f32::total_cmp);
        let mean_composited_rgb_psnr_db = composited_psnrs.iter().sum::<f32>() * scale;
        let median_composited_rgb_psnr_db = sorted_percentile(&composited_psnrs, 0.5);
        let p10_composited_rgb_psnr_db = sorted_percentile(&composited_psnrs, 0.1);
        let min_composited_rgb_psnr_db = composited_psnrs[0];
        let max_composited_rgb_psnr_db = composited_psnrs[examples_count - 1];
        let aggregate_composited_rgb_psnr_db =
            psnr_db_from_mse(aggregate_composited_rgb_mse);
        let aggregate_foreground_rgb_psnr_db =
            psnr_db_from_mse(aggregate_foreground_rgb_mse);
        let aggregate_density_psnr_db = psnr_db_from_mse(aggregate_density_mse);
        let base_only_scale = 1.0 / base_only_count.max(1) as f32;
        let base_only_composited_rgb_psnr_db =
            psnr_db_from_mse(base_only_composited_mse_sum * base_only_scale);
        let base_only_density_psnr_db =
            psnr_db_from_mse(base_only_density_mse_sum * base_only_scale);
        let base_only_density_soft_iou = base_only_density_soft_iou_sum * base_only_scale;
        let generated_adapter_composited_psnr_gain_db =
            aggregate_composited_rgb_psnr_db - base_only_composited_rgb_psnr_db;
        let selection_psnr_db = p10_composited_rgb_psnr_db;
        let p10_gap_to_target_point_splat_db =
            target_point_splat_p10_composited_rgb_psnr_db - p10_composited_rgb_psnr_db;
        let condition_shuffle_psnr_gap_db =
            mean_condition_shuffle_render_rgb_psnr_db.map(|shuffle| mean_render_rgb_psnr_db - shuffle);
        let condition_shuffle_composited_psnr_gap_db = condition_shuffle_composited_rgb_psnr_db
            .map(|shuffle| aggregate_composited_rgb_psnr_db - shuffle);
        let mean_passed =
            aggregate_composited_rgb_psnr_db >= config.validation_psnr_threshold_db;
        let all_examples_passed = entries.iter().all(|entry| entry.passed);
        let conditional_control_passed = generated_adapter_composited_psnr_gain_db > 0.0
            && condition_shuffle_composited_psnr_gap_db.is_none_or(|gap| gap > 0.0);
        let adapter_diagnostics = adapter_diagnostics(
            &adapter_parameter_rows,
            npa_config,
            config.adapter_rank,
        )?;
        let passed = mean_passed
            && selection_psnr_db >= config.validation_psnr_threshold_db
            && conditional_control_passed;
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
            selection_metric: "p10-composited-rgb-psnr",
            selection_psnr_db,
            selection_horizon_min_steps: config.validation_steps,
            horizon_summaries: Vec::new(),
            peak_horizon_p10_composited_rgb_psnr_db: p10_composited_rgb_psnr_db,
            final_horizon_p10_composited_rgb_psnr_db: p10_composited_rgb_psnr_db,
            peak_to_final_p10_drop_db: 0.0,
            target_point_splat_aggregate_composited_rgb_psnr_db,
            target_point_splat_p10_composited_rgb_psnr_db,
            p10_gap_to_target_point_splat_db,
            aggregate_composited_rgb_mse,
            aggregate_composited_rgb_psnr_db,
            mean_composited_rgb_psnr_db,
            median_composited_rgb_psnr_db,
            p10_composited_rgb_psnr_db,
            min_composited_rgb_psnr_db,
            max_composited_rgb_psnr_db,
            aggregate_foreground_rgb_mse,
            aggregate_foreground_rgb_psnr_db,
            aggregate_density_mse,
            aggregate_density_psnr_db,
            mean_density_soft_iou,
            mean_condition_shuffle_render_rgb_psnr_db,
            condition_shuffle_psnr_gap_db,
            condition_shuffle_composited_rgb_psnr_db,
            condition_shuffle_composited_psnr_gap_db,
            base_only_composited_rgb_psnr_db,
            generated_adapter_composited_psnr_gain_db,
            base_only_density_psnr_db,
            base_only_density_soft_iou,
            dino_nearest_teacher_render_rgb_psnr_db,
            dino_nearest_teacher_composited_rgb_psnr_db,
            dino_nearest_teacher_entries,
            conditional_control_passed,
            adapter_diagnostics,
            entries,
        }))
    }

    struct BurnE2eQualityBatchTensors {
        loss: BurnLossBatchTensors,
        adapter_vector: Option<Tensor2>,
        render_rgb_mse: Tensor1,
        composited_rgb_mse: Tensor1,
        foreground_rgb_mse: Tensor1,
        density_mse: Tensor1,
        density_soft_iou: Tensor1,
    }

    #[derive(Clone)]
    enum E2eEvalConditionMode {
        Generated,
        BaseOnly,
        ExplicitAdapter(Tensor2),
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_e2e_eval_quality(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        targets: &[BurnTargetExample],
        condition_indices: &[usize],
        target_indices: &[usize],
        generator_config: BurnE2eRolloutTrainConfig,
        eval_config: DirectBasisTrainConfig,
        seed: u64,
        device: &BurnDevice,
        condition_mode: E2eEvalConditionMode,
    ) -> Result<BurnE2eQualityBatchTensors, Box<dyn std::error::Error>> {
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "HyperNPA e2e quality validation condition/target batch length mismatch",
            )
            .into());
        }
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "HyperNPA e2e quality validation requires homogeneous particle counts",
            )
            .into());
        };
        let condition = conditions.select(condition_indices)?;
        let (adapter_batch, condition_control, adapter_vector) = match condition_mode {
            E2eEvalConditionMode::Generated => {
                let adapter =
                    generator.adapter_batch(condition.clone(), npa_config, generator_config);
                let vector = adapter.to_parameter_vector();
                (
                    adapter,
                    generator.condition_control_batch(condition.clone(), generator_config),
                    Some(vector),
                )
            }
            E2eEvalConditionMode::BaseOnly => {
                let parameter_count = NpaLowRankAdapter::parameter_count_for_config(
                    npa_config,
                    generator_config.adapter_rank,
                );
                (
                    BurnAdapterBatch::from_parameter_vector(
                        Tensor::<BurnBackend, 2>::zeros(
                            [condition_indices.len(), parameter_count],
                            device,
                        ),
                        npa_config,
                        generator_config.adapter_rank,
                        generator_config.adapter_alpha,
                    ),
                    None,
                    None,
                )
            }
            E2eEvalConditionMode::ExplicitAdapter(vector) => (
                BurnAdapterBatch::from_parameter_vector(
                    vector,
                    npa_config,
                    generator_config.adapter_rank,
                    generator_config.adapter_alpha,
                ),
                None,
                None,
            ),
        };
        let (mut x, mut s) = seed_batch_tensors_with_seed_indices(
            targets,
            target_indices,
            target_indices,
            particle_count,
            eval_config,
            seed,
            device,
        );
        let mut rngs = target_indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([target_indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(eval_config);
        let mut remaining_steps = eval_config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                eval_config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
                condition_control.as_ref(),
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        let mut quality = target_splat_quality_batch_vector(
            &x,
            &s,
            targets,
            target_indices,
            eval_config,
            &adapter_batch,
            displacement,
        );
        quality.adapter_vector = adapter_vector;
        Ok(quality)
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
            E2eLrSchedule::UpstreamGrowing => {
                let phase_step = step.saturating_sub(1) % 10_000 + 1;
                let milestones_passed = phase_step.saturating_sub(1).div_euclid(2_000).min(4);
                0.3_f32.powi(milestones_passed as i32)
            }
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
        finite_scalar("HyperNPA e2e PSNR", 10.0 * (1.0 / mse).log10()).unwrap_or(0.0)
    }

    fn sorted_percentile(sorted: &[f32], quantile: f32) -> f32 {
        debug_assert!(!sorted.is_empty());
        let position = quantile.clamp(0.0, 1.0) * sorted.len().saturating_sub(1) as f32;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let blend = position - lower as f32;
        sorted[lower] * (1.0 - blend) + sorted[upper] * blend
    }

    fn adapter_diagnostics(
        rows: &[Vec<f32>],
        npa_config: &NpaConfig,
        rank: usize,
    ) -> AutomataResult<BurnE2eAdapterDiagnostics> {
        let parameter_count = NpaLowRankAdapter::parameter_count_for_config(npa_config, rank);
        if rows.is_empty()
            || rows.iter().any(|row| {
                row.len() != parameter_count || !row.iter().all(|value| value.is_finite())
            })
        {
            return Err(AutomataError::InvalidArgument(
                "adapter diagnostics require non-empty, finite, shape-consistent rows".to_string(),
            ));
        }
        let norms = rows
            .iter()
            .map(|row| row.iter().map(|value| value * value).sum::<f32>().sqrt())
            .collect::<Vec<_>>();
        let mean_l2_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        let min_l2_norm = norms.iter().copied().fold(f32::INFINITY, f32::min);
        let max_l2_norm = norms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut pairwise_distance_sum = 0.0_f32;
        let mut pairwise_cosine_sum = 0.0_f32;
        let mut min_pairwise_l2_distance = f32::INFINITY;
        let mut pairs = 0usize;
        for left in 0..rows.len() {
            for right in left + 1..rows.len() {
                let mut squared_distance = 0.0_f32;
                let mut dot = 0.0_f32;
                for parameter in 0..parameter_count {
                    let lhs = rows[left][parameter];
                    let rhs = rows[right][parameter];
                    let delta = lhs - rhs;
                    squared_distance += delta * delta;
                    dot += lhs * rhs;
                }
                let distance = squared_distance.sqrt();
                pairwise_distance_sum += distance;
                min_pairwise_l2_distance = min_pairwise_l2_distance.min(distance);
                pairwise_cosine_sum += dot / (norms[left] * norms[right]).max(EPSILON);
                pairs += 1;
            }
        }
        let layout = crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
            npa_config,
            rank,
            1,
        )?;
        let group_rms = |group| {
            let segment = layout
                .segments
                .iter()
                .find(|segment| segment.group == group)
                .expect("adapter layout contains every parameter group");
            let sum = rows
                .iter()
                .flat_map(|row| {
                    row[segment.vector_offset..segment.vector_offset + segment.len]
                        .iter()
                        .copied()
                })
                .map(|value| value * value)
                .sum::<f32>();
            (sum / (rows.len() * segment.len).max(1) as f32).sqrt()
        };
        use crate::hyper::adapter_layout::AdapterParameterGroup2d;
        Ok(BurnE2eAdapterDiagnostics {
            parameter_count,
            mean_l2_norm,
            min_l2_norm,
            max_l2_norm,
            mean_pairwise_l2_distance: (pairs > 0)
                .then_some(pairwise_distance_sum / pairs.max(1) as f32),
            min_pairwise_l2_distance: (pairs > 0).then_some(min_pairwise_l2_distance),
            mean_pairwise_cosine_similarity: (pairs > 0)
                .then_some(pairwise_cosine_sum / pairs.max(1) as f32),
            w1_down_rms: group_rms(AdapterParameterGroup2d::W1Down),
            w1_up_rms: group_rms(AdapterParameterGroup2d::W1Up),
            w2_down_rms: group_rms(AdapterParameterGroup2d::W2Down),
            w2_up_rms: group_rms(AdapterParameterGroup2d::W2Up),
            b1_rms: group_rms(AdapterParameterGroup2d::B1),
            b2_rms: group_rms(AdapterParameterGroup2d::B2),
        })
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
        let mask_stack = if target.update_prob >= 1.0 {
            None
        } else {
            Some(host_single_mask_stack(target, steps, rng))
        };
        for step in 0..steps {
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
            let state_dims = s.shape().dims::<2>()[1];
            if target.update_prob >= 1.0 {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<2>(0);
                x = x + dx.mul(mask.clone().expand([target.particle_count, 2]));
                s = s + ds.mul(mask.expand([target.particle_count, state_dims]));
            }
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
        _rng: &mut StdRng,
        steps: usize,
        mut displacement: Tensor1,
        condition_control: Option<&BurnE2eConditionControlBatch>,
    ) -> (Tensor3, Tensor3, Tensor1) {
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(device_batch_mask_stack(
                targets,
                indices,
                particle_count,
                steps,
            ))
        };
        for step in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let mut update = params.forward_adapter_batch(features, adapter_batch);
            if let Some(condition_control) = condition_control {
                update = update + condition_control.update_for_particles(&x, &s);
            }
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
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
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
        condition_control: Option<&BurnE2eConditionControlBatch>,
    ) -> (Tensor3, Tensor3, Tensor1) {
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(host_batch_mask_stack_with_rngs(
                targets,
                indices,
                particle_count,
                steps,
                rngs,
            ))
        };
        for step in 0..steps {
            let features = rollout_dense_perception_batch(&x, &s, config);
            let mut update = params.forward_adapter_batch(features, adapter_batch);
            if let Some(condition_control) = condition_control {
                update = update + condition_control.update_for_particles(&x, &s);
            }
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
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
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
        let unit_update = batch_update_prob_is_one(targets, indices);
        let mask_stack = if unit_update {
            None
        } else {
            Some(host_batch_mask_stack_with_rngs(
                targets,
                indices,
                particle_count,
                steps,
                rngs,
            ))
        };
        for step in 0..steps {
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
            if unit_update {
                x = x + dx;
                s = s + ds;
            } else {
                let mask = mask_stack
                    .as_ref()
                    .expect("non-unit update_prob should have a mask stack")
                    .clone()
                    .narrow(0, step, 1)
                    .squeeze_dim::<3>(0);
                x = x + dx.mul(mask.clone().expand([indices.len(), particle_count, 2]));
                s = s + ds.mul(mask.expand([indices.len(), particle_count, state_dims]));
            }
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
                None,
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        Ok(target_splat_loss_batch_vector_selected(
            &x,
            &s,
            targets,
            indices,
            config,
            &adapter_batch,
            displacement,
        )?)
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
                None,
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
        match perception_backend_effective(config) {
            PerceptionRolloutBackend::Dense => dense_perception(&feature_x, &feature_s, config),
            PerceptionRolloutBackend::TiledAdjoint => perception_tiled_adjoint_batch(
                feature_x.unsqueeze_dim::<3>(0),
                feature_s.unsqueeze_dim::<3>(0),
                config,
            )
            .squeeze_dim::<2>(0),
            PerceptionRolloutBackend::Auto => unreachable!("auto perception backend resolved"),
        }
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
        match perception_backend_effective(config) {
            PerceptionRolloutBackend::Dense => dense_perception_batch(&feature_x, &feature_s, config),
            PerceptionRolloutBackend::TiledAdjoint => {
                perception_tiled_adjoint_batch(feature_x, feature_s, config)
            }
            PerceptionRolloutBackend::Auto => unreachable!("auto perception backend resolved"),
        }
    }

    fn perception_backend_effective(
        config: DirectBasisTrainConfig,
    ) -> PerceptionRolloutBackend {
        match config.perception_backend {
            PerceptionRolloutBackend::Auto => perception_backend_auto(config),
            PerceptionRolloutBackend::Dense => PerceptionRolloutBackend::Dense,
            PerceptionRolloutBackend::TiledAdjoint => PerceptionRolloutBackend::TiledAdjoint,
        }
    }

    fn perception_backend_auto(config: DirectBasisTrainConfig) -> PerceptionRolloutBackend {
        #[cfg($perception_cube_feature)]
        {
            if config.rollout_particles >= 128 {
                PerceptionRolloutBackend::TiledAdjoint
            } else {
                PerceptionRolloutBackend::Dense
            }
        }
        #[cfg(not($perception_cube_feature))]
        {
            let _ = config;
            PerceptionRolloutBackend::Dense
        }
    }

    #[derive(Clone, Debug)]
    struct PerceptionPreparedState {
        density: Tensor2Inner,
        offsets: Tensor2IntInner,
        permutation: Tensor2IntInner,
        raw_state_gradient: Tensor4Inner,
        state_gradient_inverse: Tensor3Inner,
    }

    #[derive(Clone, Debug)]
    struct PerceptionAdjointState {
        x: Tensor3Inner,
        s: Tensor3Inner,
        #[cfg($perception_cube_feature)]
        prepared: Option<PerceptionPreparedState>,
        batch_size: usize,
        particle_count: usize,
        state_dims: usize,
        grid_eps: f32,
    }

    #[derive(Clone, Copy, Debug)]
    struct PerceptionAdjointOp;

    impl Backward<InnerBackend, 2> for PerceptionAdjointOp {
        type State = PerceptionAdjointState;

        fn backward(
            self,
            ops: Ops<Self::State, 2>,
            grads: &mut Gradients,
            _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer,
        ) {
            let [x_parent, s_parent] = ops.parents;
            if x_parent.is_none() && s_parent.is_none() {
                return;
            }
            let feature_grad = grads.consume::<InnerBackend>(&ops.node);
            let feature_grad_tensor =
                Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(feature_grad));
            let device = feature_grad_tensor.device();

            #[cfg($perception_cube_feature)]
            {
                let cube_config = perception_cube_adjoint_config(
                    ops.state.grid_eps,
                    x_parent.is_some(),
                    s_parent.is_some(),
                );
                let prepared_adjoint = ops.state.prepared.as_ref().and_then(|prepared| {
                    InnerBackend::perception_cube_adjoint_prepared(
                        ops.state.x.clone(),
                        ops.state.s.clone(),
                        feature_grad_tensor.clone(),
                        prepared.density.clone(),
                        prepared.offsets.clone(),
                        prepared.permutation.clone(),
                        prepared.raw_state_gradient.clone(),
                        prepared.state_gradient_inverse.clone(),
                        cube_config,
                    )
                });
                let device_adjoint = prepared_adjoint.or_else(|| {
                    InnerBackend::perception_cube_adjoint(
                        ops.state.x.clone(),
                        ops.state.s.clone(),
                        feature_grad_tensor.clone(),
                        cube_config,
                    )
                });
                if let Some(device_adjoint) = device_adjoint {
                    PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                    if ops.state.prepared.is_some() {
                        PERCEPTION_CUBE_PREPARED_REUSE_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                    let device_adjoint =
                        device_adjoint.unwrap_or_else(|err| panic!("perception cube adjoint failed: {err}"));
                    if let Some(parent) = x_parent {
                        grads.register::<InnerBackend>(
                            parent.id,
                            device_adjoint.position_grad.into_primitive().tensor(),
                        );
                    }
                    if let Some(parent) = s_parent {
                        grads.register::<InnerBackend>(
                            parent.id,
                            device_adjoint.state_grad.into_primitive().tensor(),
                        );
                    }
                    return;
                }
            }

            PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
            let feature_grad = feature_grad_tensor
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint readback failed: {err}"));
            let x_values = ops
                .state
                .x
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint position readback failed: {err}"));
            let states = ops
                .state
                .s
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint state readback failed: {err}"));
            let positions = xy_positions_to_reference_positions(&x_values);
            let grid = perception_reference_grid(ops.state.grid_eps);
            let options = perception_reference_options(ops.state.grid_eps);
            let adjoint = burn_automata_kernels::perceive_adjoint_with_options(
                &positions,
                &states,
                ops.state.batch_size,
                ops.state.particle_count,
                ops.state.state_dims,
                &grid,
                options,
                &feature_grad,
            )
            .unwrap_or_else(|err| panic!("perception adjoint failed: {err}"));

            if let Some(parent) = x_parent {
                let position_grad = adjoint
                    .position
                    .iter()
                    .flat_map(|value| [value[0], value[1]])
                    .collect::<Vec<_>>();
                let tensor = Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(
                        position_grad,
                        [ops.state.batch_size, ops.state.particle_count, 2],
                    ),
                    &device,
                )
                .into_primitive()
                .tensor();
                grads.register::<InnerBackend>(parent.id, tensor);
            }
            if let Some(parent) = s_parent {
                let tensor = Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(
                        adjoint.state,
                        [
                            ops.state.batch_size,
                            ops.state.particle_count,
                            ops.state.state_dims,
                        ],
                    ),
                    &device,
                )
                .into_primitive()
                .tensor();
                grads.register::<InnerBackend>(parent.id, tensor);
            }
        }
    }

    fn perception_tiled_adjoint_batch(
        x: Tensor3,
        s: Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batch_size = dims[0];
        let particle_count = dims[1];
        let state_dims = dims[2];
        let x_dims = x.shape().dims::<3>();
        assert_eq!(
            x_dims,
            [batch_size, particle_count, 2],
            "perception tiled-adjoint expects x shape [batch, particles, 2]"
        );
        let x_primitive = x.into_primitive().tensor();
        let s_primitive = s.into_primitive().tensor();
        let x_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
            x_primitive.primitive.clone(),
        ));
        let s_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
            s_primitive.primitive.clone(),
        ));
        #[cfg($perception_cube_feature)]
        let (output, prepared) = {
            let cube_config = perception_cube_adjoint_config(
                config.grid_eps,
                !config.stopgrad_pos,
                !config.stopgrad_state,
            );
            let prepared_forward = (config.stopgrad_pos && particle_count >= 512)
                .then(|| {
                    InnerBackend::perception_cube_forward_prepared(
                        x_inner.clone(),
                        s_inner.clone(),
                        cube_config,
                    )
                })
                .flatten();
            if let Some(device_forward) = prepared_forward {
                PERCEPTION_CUBE_FORWARD_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                let device_forward = device_forward.unwrap_or_else(|err| {
                    panic!("prepared perception cube forward failed: {err}")
                });
                let output = device_forward.features.into_primitive().tensor();
                let prepared = PerceptionPreparedState {
                    density: device_forward.density,
                    offsets: device_forward.offsets,
                    permutation: device_forward.permutation,
                    raw_state_gradient: device_forward.raw_state_gradient,
                    state_gradient_inverse: device_forward.state_gradient_inverse,
                };
                (output, Some(prepared))
            } else if let Some(device_forward) = InnerBackend::perception_cube_forward(
                x_inner.clone(),
                s_inner.clone(),
                cube_config,
            ) {
                PERCEPTION_CUBE_FORWARD_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                (
                    device_forward
                        .unwrap_or_else(|err| panic!("perception cube forward failed: {err}"))
                        .features
                        .into_primitive()
                        .tensor(),
                    None,
                )
            } else {
                PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
                (
                    dense_perception_batch_inner(&x_inner, &s_inner, config)
                        .into_primitive()
                        .tensor(),
                    None,
                )
            }
        };
        #[cfg(not($perception_cube_feature))]
        let output = dense_perception_batch_inner(&x_inner, &s_inner, config)
            .into_primitive()
            .tensor();
        let state = PerceptionAdjointState {
            x: x_inner,
            s: s_inner,
            #[cfg($perception_cube_feature)]
            prepared,
            batch_size,
            particle_count,
            state_dims,
            grid_eps: config.grid_eps,
        };
        let prep = PerceptionAdjointOp
            .prepare::<NoCheckpointing>([x_primitive.node.clone(), s_primitive.node.clone()])
            .compute_bound();
        let output = match prep.stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        };
        Tensor::<BurnBackend, 3>::from_primitive(TensorPrimitive::Float(output))
    }

    fn xy_positions_to_reference_positions(values: &[f32]) -> Vec<[f32; 4]> {
        values
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1], 0.0, 0.0])
            .collect()
    }

    fn perception_reference_grid(grid_eps: f32) -> burn_automata_kernels::HashGridConfig {
        let mut grid = crate::upstream_growing_2d_hashgrid();
        grid.eps = grid_eps.max(EPSILON);
        grid
    }

    fn perception_reference_options(_grid_eps: f32) -> burn_automata_kernels::PerceptionOptions {
        let npa = NpaConfig::growing_2d();
        burn_automata_kernels::PerceptionOptions {
            state_grad: npa.state_grad,
            density_grad: npa.density_grad,
            eps0: npa.eps0.max(f32::MIN_POSITIVE),
            scale_equivariance: npa.scale_equivariant(),
            particle_density_equivariance: npa.particle_density_equivariant(),
            log_norm_grad: npa.log_norm_grad,
            log_norm_density_grad: npa.log_norm_density_grad,
            hybrid_state_gradient: true,
            position_features: npa.position_features,
        }
    }

    #[cfg($perception_cube_feature)]
    fn perception_cube_adjoint_config(
        grid_eps: f32,
        compute_position_grad: bool,
        compute_state_grad: bool,
    ) -> PerceptionCubeAdjointConfig {
        let npa = NpaConfig::growing_2d();
        PerceptionCubeAdjointConfig {
            eps: grid_eps.max(EPSILON),
            eps0: npa.eps0.max(f32::MIN_POSITIVE),
            state_grad: npa.state_grad,
            density_grad: npa.density_grad,
            scale_equivariance: npa.scale_equivariant(),
            particle_density_equivariance: npa.particle_density_equivariant(),
            log_norm_grad: npa.log_norm_grad,
            log_norm_density_grad: npa.log_norm_density_grad,
            hybrid_state_gradient: true,
            position_features: npa.position_features,
            compute_position_grad,
            compute_state_grad,
            grid_width: 16,
            grid_height: 16,
            sparse_grid_min_particles: 512,
        }
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

    fn dense_perception_batch_inner(
        x: &Tensor3Inner,
        s: &Tensor3Inner,
        config: DirectBasisTrainConfig,
    ) -> Tensor3Inner {
        dense_perception_batch_generic::<InnerBackend>(x, s, config)
    }

    fn dense_perception_batch_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        s: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
    ) -> Tensor<B, 3> {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let density = dense_particle_density_batch_generic(x, config);
        let chunk_size =
            dense_query_chunk_size(batches, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_batch_chunk_generic(
                x, s, &density, config, start, len,
            ));
        }
        Tensor::cat(chunks, 1)
    }

    fn dense_perception_batch_chunk_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        s: &Tensor<B, 3>,
        density: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor<B, 3> {
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
        let density_grad = log_normalize_vectors_batch_generic(
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
        let state_grad =
            apply_moment_correction_2d_batch_generic::<B>(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient_batch_generic(state_grad);

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
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target
            .target_density
            .clone()
            .clamp_min(0.0)
            .clamp_max(1.0);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target.target_rgb.clone()
            + target_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_loss = composited_diff.clone().mul(composited_diff).mean();
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
                .mul_scalar(config.loss_config.foreground_density_loss_weight)
            + composited_rgb_loss.mul_scalar(config.loss_config.composited_rgb_loss_weight);
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
    ) -> AutomataResult<BurnLossTensors> {
        let per_example = target_splat_loss_batch_vector_selected(
            x,
            s,
            targets,
            indices,
            config,
            adapter,
            displacement,
        )?;
        Ok(BurnLossTensors {
            total: per_example.total.mean(),
            splat: per_example.splat.mean(),
            color: per_example.color.mean(),
            density: per_example.density.mean(),
        })
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
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target_density.clone().clamp_min(0.0).clamp_max(1.0);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target_rgb.clone()
            + target_alpha
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_loss = composited_diff
            .clone()
            .mul(composited_diff)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
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
        let color_loss = l1l2_tensor3(rgb - target_rgb)
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
                .mul_scalar(config.loss_config.foreground_density_loss_weight)
            + composited_rgb_loss.mul_scalar(config.loss_config.composited_rgb_loss_weight);
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

    fn target_splat_loss_batch_vector_selected(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossBatchTensors> {
        match target2d_loss_backend_effective(config) {
            Target2dLossBackend::Dense => Ok(target_splat_loss_batch_vector(
                x,
                s,
                targets,
                indices,
                config,
                adapter,
                displacement,
            )),
            Target2dLossBackend::TiledAdjoint => target_splat_loss_batch_vector_tiled_adjoint(
                x,
                s,
                targets,
                indices,
                config,
                adapter,
                displacement,
            ),
            Target2dLossBackend::Auto => unreachable!("auto target2d backend must be resolved"),
        }
    }

    fn target2d_loss_backend_effective(config: DirectBasisTrainConfig) -> Target2dLossBackend {
        match config.target2d_loss_backend {
            Target2dLossBackend::Auto => target2d_loss_backend_auto(config),
            Target2dLossBackend::Dense => Target2dLossBackend::Dense,
            Target2dLossBackend::TiledAdjoint => Target2dLossBackend::TiledAdjoint,
        }
    }

    fn target2d_loss_backend_auto(config: DirectBasisTrainConfig) -> Target2dLossBackend {
        #[cfg($perception_cube_feature)]
        {
            if config.rollout_particles >= 128 {
                Target2dLossBackend::TiledAdjoint
            } else {
                Target2dLossBackend::Dense
            }
        }
        #[cfg(not($perception_cube_feature))]
        {
            let _ = config;
            Target2dLossBackend::Dense
        }
    }

    #[cfg($perception_cube_feature)]
    fn target2d_cube_loss_config(value: crate::Target2dLossConfig) -> Target2dCubeLossConfig {
        Target2dCubeLossConfig {
            image_size: value.image_size,
            sigma: value.sigma,
            lo: value.lo,
            hi: value.hi,
            splat_loss_weight: value.splat_loss_weight,
            color_loss_weight: value.color_loss_weight,
            density_loss_weight: value.density_loss_weight,
            background_density_loss_weight: value.background_density_loss_weight,
            foreground_density_loss_weight: value.foreground_density_loss_weight,
            composited_rgb_loss_weight: value.composited_rgb_loss_weight,
            center: value.center,
        }
    }

    fn target_splat_loss_batch_vector_tiled_adjoint(
        x: &Tensor3,
        s: &Tensor3,
        targets: &[BurnTargetExample],
        indices: &[usize],
        config: DirectBasisTrainConfig,
        adapter: &BurnAdapterBatch,
        displacement: Tensor1,
    ) -> AutomataResult<BurnLossBatchTensors> {
        if indices.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "tiled target2d loss requires at least one target index".to_string(),
            ));
        }
        let dims = x.shape().dims::<3>();
        let batches = dims[0];
        let particle_count = dims[1];
        let state_dims = s.shape().dims::<3>()[2];
        if batches != indices.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "tiled target2d batch mismatch: tensor batch={batches} indices={}",
                indices.len()
            )));
        }
        let device = &x.device();
        #[cfg($perception_cube_feature)]
        if config.loss_config.shape_chamfer_loss_weight == 0.0 {
            let target_mean = config
                .loss_config
                .center
                .then(|| stack_target_mean(targets, indices).inner());
            let target_rgb = stack_target_rgb(targets, indices).inner();
            let target_density = stack_target_density(targets, indices).inner();
            let target_foreground = stack_target_foreground(targets, indices).inner();
            let target_foreground_scales = stack_target_foreground_scales(targets, indices).inner();
            let pixel_sizes = stack_pixel_sizes(targets, indices).inner();
            let target_point_counts = stack_target_point_counts(targets, indices).inner();
            if let Some(device_loss) = InnerBackend::target2d_cube_adjoint(
                x.clone().inner(),
                {
                    let x_inner = x.clone().inner();
                    if let Some(target_mean) = target_mean {
                        x_inner.clone()
                            - x_inner.clone().mean_dim(1).expand([batches, particle_count, 2])
                            + target_mean.expand([batches, particle_count, 2])
                    } else {
                        x_inner
                    }
                },
                s.clone().inner(),
                target_rgb,
                target_density,
                target_foreground,
                target_foreground_scales,
                pixel_sizes,
                target_point_counts,
                target2d_cube_loss_config(config.loss_config),
            ) {
                TARGET2D_CUBE_ADJOINT_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                let device_loss = device_loss?;
                let position_grad = Tensor::<BurnBackend, 3>::from_inner(device_loss.position_grad);
                let state_grad = Tensor::<BurnBackend, 3>::from_inner(device_loss.state_grad);
                let position_term = x
                    .clone()
                    .mul(position_grad)
                    .reshape([batches, particle_count * 2])
                    .sum_dim(1)
                    .squeeze_dim::<1>(1);
                let state_term = s
                    .clone()
                    .mul(state_grad)
                    .reshape([batches, particle_count * state_dims])
                    .sum_dim(1)
                    .squeeze_dim::<1>(1);
                let mut total = position_term
                    + state_term
                    + Tensor::<BurnBackend, 1>::from_inner(device_loss.constant);
                if config.loss_config.bound_regularizer_weight > 0.0 {
                    let bound_loss = relu(x.clone().abs().add_scalar(-1.0))
                        .reshape([batches, particle_count * 2])
                        .mean_dim(1)
                        .squeeze_dim::<1>(1);
                    total = total
                        + bound_loss.mul_scalar(config.loss_config.bound_regularizer_weight);
                }
                if config.loss_config.overflow_regularizer_weight > 0.0 {
                    let overflow_loss = relu(s.clone().abs().add_scalar(-1.0))
                        .reshape([batches, particle_count * state_dims])
                        .mean_dim(1)
                        .squeeze_dim::<1>(1);
                    total = total
                        + overflow_loss.mul_scalar(config.loss_config.overflow_regularizer_weight);
                }
                if config.loss_config.displacement_regularizer_weight > 0.0 {
                    total = total
                        + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight);
                }
                if config.adapter_l2_weight > 0.0 {
                    total = total
                        + adapter
                            .l2_loss_vector()
                            .mul_scalar(config.adapter_l2_weight);
                }
                return Ok(BurnLossBatchTensors {
                    total,
                    splat: Tensor::<BurnBackend, 1>::from_inner(device_loss.splat),
                    color: Tensor::<BurnBackend, 1>::from_inner(device_loss.color),
                    density: Tensor::<BurnBackend, 1>::from_inner(device_loss.density),
                });
            }
        }
        TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
        let x_values = tensor3_vec(x.clone().inner())?;
        let s_values = tensor3_vec(s.clone().inner())?;
        let displacement_values = tensor1_vec(displacement.clone().inner())?;
        if displacement_values.len() != batches {
            return Err(AutomataError::InvalidArgument(format!(
                "tiled target2d displacement batch mismatch: displacement={} batches={batches}",
                displacement_values.len()
            )));
        }

        let mut position_grad_values = vec![0.0_f32; batches * particle_count * 2];
        let mut state_grad_values = vec![0.0_f32; batches * particle_count * state_dims];
        let mut total_values = Vec::with_capacity(batches);
        let mut splat_values = Vec::with_capacity(batches);
        let mut color_values = Vec::with_capacity(batches);
        let mut density_values = Vec::with_capacity(batches);
        let mut constant_values = Vec::with_capacity(batches);

        for (batch, target_idx) in indices.iter().copied().enumerate() {
            let target = targets.get(target_idx).ok_or_else(|| {
                AutomataError::InvalidArgument(format!(
                    "tiled target2d target index {target_idx} is out of bounds"
                ))
            })?;
            let x_offset = batch * particle_count * 2;
            let s_offset = batch * particle_count * state_dims;
            let mut positions = Vec::with_capacity(particle_count);
            for particle in 0..particle_count {
                let base = x_offset + particle * 2;
                positions.push([x_values[base], x_values[base + 1], 0.0, 0.0]);
            }
            let states =
                &s_values[s_offset..s_offset + particle_count.saturating_mul(state_dims)];
            let reference = crate::target_2d_loss_with_adjoint(
                &positions,
                states,
                1,
                particle_count,
                state_dims,
                &target.target_cpu,
                config.loss_config,
                0.0,
                0,
            )?;

            let total = reference.report.total_loss;
            total_values.push(total);
            splat_values.push(reference.report.splat_loss);
            color_values.push(reference.report.color_loss);
            density_values.push(reference.report.density_loss);

            let mut dot = 0.0_f32;
            for particle in 0..particle_count {
                let pos_base = x_offset + particle * 2;
                let reference_position = reference.position_gradients[particle];
                position_grad_values[pos_base] = reference_position[0];
                position_grad_values[pos_base + 1] = reference_position[1];
                dot += x_values[pos_base] * reference_position[0]
                    + x_values[pos_base + 1] * reference_position[1];

                let state_base = s_offset + particle * state_dims;
                let reference_state = &reference.state_gradients
                    [particle * state_dims..(particle + 1) * state_dims];
                for dim in 0..state_dims {
                    state_grad_values[state_base + dim] = reference_state[dim];
                    dot += s_values[state_base + dim] * reference_state[dim];
                }
            }
            constant_values.push(total - dot);
        }

        let position_grad = tensor3(position_grad_values, [batches, particle_count, 2], device);
        let state_grad = tensor3(
            state_grad_values,
            [batches, particle_count, state_dims],
            device,
        );
        let position_term = x
            .clone()
            .mul(position_grad)
            .reshape([batches, particle_count * 2])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let state_term = s
            .clone()
            .mul(state_grad)
            .reshape([batches, particle_count * state_dims])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let mut total = position_term + state_term + tensor1(constant_values, [batches], device);
        if config.loss_config.displacement_regularizer_weight > 0.0 {
            total = total
                + displacement.mul_scalar(config.loss_config.displacement_regularizer_weight);
        }
        if config.adapter_l2_weight > 0.0 {
            total = total
                + adapter
                    .l2_loss_vector()
                    .mul_scalar(config.adapter_l2_weight);
        }

        Ok(BurnLossBatchTensors {
            total,
            splat: tensor1(splat_values, [batches], device),
            color: tensor1(color_values, [batches], device),
            density: tensor1(density_values, [batches], device),
        })
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
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
        let centered = if config.loss_config.center {
            x.clone() - x.clone().mean_dim(1).expand([batches, particle_count, 2])
                + target_mean.expand([batches, particle_count, 2])
        } else {
            x.clone()
        };
        let colors = s.clone().narrow(2, state_dims - 3, 3).add_scalar(0.5);
        let (rgb, density) =
            splat_render_batch(&centered, &colors, targets, indices, config, particle_count);
        let rgb_diff = rgb.clone() - target_rgb.clone();
        let render_rgb_mse = rgb_diff
            .clone()
            .mul(rgb_diff.clone())
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let predicted_alpha = density.clone().clamp_min(0.0).clamp_max(1.0);
        let target_alpha = target_density.clone().clamp_min(0.0).clamp_max(1.0);
        let density_diff_for_metrics = predicted_alpha.clone() - target_alpha.clone();
        let density_mse = density_diff_for_metrics
            .clone()
            .mul(density_diff_for_metrics)
            .reshape([batches, pixels])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let predicted_straight_rgb = rgb
            .clone()
            .div(
                density
                    .clone()
                    .clamp_min(EPSILON)
                    .expand([batches, pixels, 3]),
            )
            .clamp_min(0.0)
            .clamp_max(1.0);
        let target_straight_rgb = target_rgb
            .clone()
            .div(
                target_density
                    .clone()
                    .clamp_min(EPSILON)
                    .expand([batches, pixels, 3]),
            )
            .clamp_min(0.0)
            .clamp_max(1.0);
        let foreground_rgb_diff = predicted_straight_rgb - target_straight_rgb;
        let foreground_rgb_squared = foreground_rgb_diff
            .clone()
            .mul(foreground_rgb_diff)
            .mul(target_foreground.clone().expand([batches, pixels, 3]))
            .reshape([batches, pixels * 3])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let foreground_rgb_denominator = target_foreground
            .clone()
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mul_scalar(3.0)
            .clamp_min(EPSILON);
        let foreground_rgb_mse = foreground_rgb_squared.div(foreground_rgb_denominator);
        let predicted_composited_rgb = (rgb.clone()
            + predicted_alpha
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let target_composited_rgb = (target_rgb.clone()
            + target_alpha
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .expand([batches, pixels, 3]))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let composited_rgb_diff = predicted_composited_rgb - target_composited_rgb;
        let composited_rgb_mse = composited_rgb_diff
            .clone()
            .mul(composited_rgb_diff)
            .reshape([batches, pixels * 3])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let density_intersection = predicted_alpha
            .clone()
            .min_pair(target_alpha.clone())
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let density_union = predicted_alpha
            .max_pair(target_alpha)
            .reshape([batches, pixels])
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .clamp_min(EPSILON);
        let density_soft_iou = density_intersection.div(density_union);
        let background_density_loss = background_density_term_batch(
            density.clone(),
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
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
            adapter_vector: None,
            render_rgb_mse,
            composited_rgb_mse,
            foreground_rgb_mse,
            density_mse,
            density_soft_iou,
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
        let target_rgb = stack_target_rgb(targets, indices);
        let target_density = stack_target_density(targets, indices);
        let target_foreground = stack_target_foreground(targets, indices);
        let target_foreground_scales = stack_target_foreground_scales(targets, indices);
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
            target_foreground.clone(),
        )
        .reshape([batches, pixels])
        .mean_dim(1)
        .squeeze_dim::<1>(1);
        let foreground_density_loss = foreground_density_term_batch(
            density.clone(),
            target_density.clone(),
            target_foreground,
            target_foreground_scales,
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
        let color_loss = l1l2_tensor3(rgb - target_rgb)
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
        let pixel_sizes = stack_pixel_sizes(targets, indices);
        let sigma = pixel_sizes
            .clone()
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
        let norm_scale = pixel_sizes
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

    fn target_point_splat_composited_mses(
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_batch_size: usize,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<f32>> {
        let particle_count = config.rollout_particles.max(1);
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let batch_size = requested_batch_size.max(1).min(targets.len().max(1));
        let mut output = Vec::with_capacity(targets.len());
        for start in (0..targets.len()).step_by(batch_size) {
            let end = (start + batch_size).min(targets.len());
            let indices = (start..end).collect::<Vec<_>>();
            let mut positions = Vec::with_capacity(indices.len() * particle_count * 2);
            let mut colors = Vec::with_capacity(indices.len() * particle_count * 3);
            for &target_index in &indices {
                let target = &targets[target_index].target_cpu;
                let target_points = target.point_count();
                debug_assert!(target_points > 0);
                for particle in 0..particle_count {
                    let point = if particle_count <= target_points {
                        particle.saturating_mul(target_points) / particle_count
                    } else {
                        particle % target_points
                    }
                    .min(target_points - 1);
                    positions.extend_from_slice(&target.positions[point]);
                    colors.extend_from_slice(&target.colors[point]);
                }
            }
            let positions = tensor3(
                positions,
                [indices.len(), particle_count, 2],
                device,
            );
            let colors = tensor3(colors, [indices.len(), particle_count, 3], device);
            let (rgb, density) = splat_render_batch(
                &positions,
                &colors,
                targets,
                &indices,
                config,
                particle_count,
            );
            let predicted_alpha = density.clamp_min(0.0).clamp_max(1.0);
            let target_alpha = stack_target_density(targets, &indices)
                .clamp_min(0.0)
                .clamp_max(1.0);
            let predicted_composited_rgb = (rgb
                + predicted_alpha
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .expand([indices.len(), pixels, 3]))
            .clamp_min(0.0)
            .clamp_max(1.0);
            let target_composited_rgb = (stack_target_rgb(targets, &indices)
                + target_alpha
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .expand([indices.len(), pixels, 3]))
            .clamp_min(0.0)
            .clamp_max(1.0);
            let diff = predicted_composited_rgb - target_composited_rgb;
            let mses = diff
                .clone()
                .mul(diff)
                .reshape([indices.len(), pixels * 3])
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            for mse in tensor1_vec(mses.inner())? {
                output.push(finite_scalar(
                    "HyperNPA target-point splat composited RGB MSE",
                    mse,
                )?);
            }
        }
        Ok(output)
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

    fn dense_particle_density_batch_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
    ) -> Tensor<B, 3> {
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

    fn log_normalize_vectors_batch_generic<B: burn::tensor::backend::Backend>(
        values: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
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

    fn log_normalize_state_gradient_batch_generic<B: burn::tensor::backend::Backend>(
        values: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
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

    fn apply_moment_correction_2d_batch_generic<B: burn::tensor::backend::Backend>(
        state_gradient: Tensor<B, 4>,
        diff: Tensor<B, 4>,
        volume_grad: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
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
        let ones = Tensor::<B, 3>::ones([batches, query_rows, 1], &state_gradient.device());
        let zeros = Tensor::<B, 3>::zeros([batches, query_rows, 1], &state_gradient.device());
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
            Tensor::<B, 3>::ones([batches, query_rows, 1], &state_gradient.device()),
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

        fn snapshots(&self) -> AutomataResult<Vec<E2eTensorSnapshot>> {
            Ok(vec![
                tensor2_snapshot("base.w1.m", self.w1_m.clone())?,
                tensor2_snapshot("base.w1.v", self.w1_v.clone())?,
                tensor2_snapshot("base.b1.m", self.b1_m.clone())?,
                tensor2_snapshot("base.b1.v", self.b1_v.clone())?,
                tensor2_snapshot("base.w2.m", self.w2_m.clone())?,
                tensor2_snapshot("base.w2.v", self.w2_v.clone())?,
                tensor2_snapshot("base.b2.m", self.b2_m.clone())?,
                tensor2_snapshot("base.b2.v", self.b2_v.clone())?,
            ])
        }

        fn restore(
            checkpoint: &E2eTrainingCheckpoint,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            Ok(Self {
                step: checkpoint.base_optimizer_step,
                w1_m: tensor2_from_snapshot(checkpoint.tensor("base.w1.m")?, device)?,
                w1_v: tensor2_from_snapshot(checkpoint.tensor("base.w1.v")?, device)?,
                b1_m: tensor2_from_snapshot(checkpoint.tensor("base.b1.m")?, device)?,
                b1_v: tensor2_from_snapshot(checkpoint.tensor("base.b1.v")?, device)?,
                w2_m: tensor2_from_snapshot(checkpoint.tensor("base.w2.m")?, device)?,
                w2_v: tensor2_from_snapshot(checkpoint.tensor("base.w2.v")?, device)?,
                b2_m: tensor2_from_snapshot(checkpoint.tensor("base.b2.m")?, device)?,
                b2_v: tensor2_from_snapshot(checkpoint.tensor("base.b2.v")?, device)?,
            })
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
                condition_control_w_m: params
                    .condition_control_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_w_v: params
                    .condition_control_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_b_m: params
                    .condition_control_b
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_b_v: params
                    .condition_control_b
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_state_w_m: params
                    .condition_control_state_w
                    .clone()
                    .inner()
                    .zeros_like(),
                condition_control_state_w_v: params
                    .condition_control_state_w
                    .clone()
                    .inner()
                    .zeros_like(),
            }
        }

        fn next_bias_correction(&mut self, cfg: AdamWConfig) -> AdamWBiasCorrection {
            next_adamw_bias_correction(&mut self.step, cfg)
        }

        fn snapshots(&self) -> AutomataResult<Vec<E2eTensorSnapshot>> {
            let tensors = [
                ("generator.token_w.m", self.token_w_m.clone()),
                ("generator.token_w.v", self.token_w_v.clone()),
                ("generator.token_b.m", self.token_b_m.clone()),
                ("generator.token_b.v", self.token_b_v.clone()),
                ("generator.token_gate_w.m", self.token_gate_w_m.clone()),
                ("generator.token_gate_w.v", self.token_gate_w_v.clone()),
                ("generator.token_gate_b.m", self.token_gate_b_m.clone()),
                ("generator.token_gate_b.v", self.token_gate_b_v.clone()),
                ("generator.state_w.m", self.state_w_m.clone()),
                ("generator.state_w.v", self.state_w_v.clone()),
                ("generator.time_w.m", self.time_w_m.clone()),
                ("generator.time_w.v", self.time_w_v.clone()),
                ("generator.output_w.m", self.output_w_m.clone()),
                ("generator.output_w.v", self.output_w_v.clone()),
                ("generator.output_b.m", self.output_b_m.clone()),
                ("generator.output_b.v", self.output_b_v.clone()),
                ("generator.condition_control_w.m", self.condition_control_w_m.clone()),
                ("generator.condition_control_w.v", self.condition_control_w_v.clone()),
                ("generator.condition_control_b.m", self.condition_control_b_m.clone()),
                ("generator.condition_control_b.v", self.condition_control_b_v.clone()),
                (
                    "generator.condition_control_state_w.m",
                    self.condition_control_state_w_m.clone(),
                ),
                (
                    "generator.condition_control_state_w.v",
                    self.condition_control_state_w_v.clone(),
                ),
            ];
            tensors
                .into_iter()
                .map(|(name, tensor)| tensor2_snapshot(name, tensor))
                .collect()
        }

        fn restore(
            checkpoint: &E2eTrainingCheckpoint,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let tensor = |name| tensor2_from_snapshot(checkpoint.tensor(name)?, device);
            Ok(Self {
                step: checkpoint.generator_optimizer_step,
                token_w_m: tensor("generator.token_w.m")?,
                token_w_v: tensor("generator.token_w.v")?,
                token_b_m: tensor("generator.token_b.m")?,
                token_b_v: tensor("generator.token_b.v")?,
                token_gate_w_m: tensor("generator.token_gate_w.m")?,
                token_gate_w_v: tensor("generator.token_gate_w.v")?,
                token_gate_b_m: tensor("generator.token_gate_b.m")?,
                token_gate_b_v: tensor("generator.token_gate_b.v")?,
                state_w_m: tensor("generator.state_w.m")?,
                state_w_v: tensor("generator.state_w.v")?,
                time_w_m: tensor("generator.time_w.m")?,
                time_w_v: tensor("generator.time_w.v")?,
                output_w_m: tensor("generator.output_w.m")?,
                output_w_v: tensor("generator.output_w.v")?,
                output_b_m: tensor("generator.output_b.m")?,
                output_b_v: tensor("generator.output_b.v")?,
                condition_control_w_m: tensor("generator.condition_control_w.m")?,
                condition_control_w_v: tensor("generator.condition_control_w.v")?,
                condition_control_b_m: tensor("generator.condition_control_b.m")?,
                condition_control_b_v: tensor("generator.condition_control_b.v")?,
                condition_control_state_w_m: tensor(
                    "generator.condition_control_state_w.m",
                )?,
                condition_control_state_w_v: tensor(
                    "generator.condition_control_state_w.v",
                )?,
            })
        }
    }

    impl BurnE2eGeneratorParams {
        fn from_seed_or_artifact(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            initial: Option<&E2eHyperNpa2d>,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            match initial {
                Some(initial) => Self::from_artifact(base, examples, config, initial, device),
                None => Self::seeded(base, examples, config, device),
            }
        }

        fn adapter_parameterization_tensors(
            base: &NpaModel,
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<(Tensor2, Tensor2)> {
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            let (constants, mask) = if config.canonical_full_rank_lora {
                let canonical =
                    crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(
                        &base.config,
                        config.adapter_rank,
                        config.adapter_alpha,
                    )?;
                (canonical.constants, canonical.trainable_mask)
            } else {
                (vec![0.0; output_dims], vec![1.0; output_dims])
            };
            Ok((
                tensor(constants, [1, output_dims], device),
                tensor(mask, [1, output_dims], device),
            ))
        }

        fn adapter_parameter_segments(config: &NpaConfig, rank: usize) -> Vec<(usize, usize)> {
            let lengths = [
                rank * config.perception_dims(),
                config.hidden_dims * rank,
                rank * config.hidden_dims,
                config.update_dims() * rank,
                config.hidden_dims,
                config.update_dims(),
            ];
            let mut offset = 0usize;
            lengths
                .into_iter()
                .map(|len| {
                    let segment = (offset, len);
                    offset += len;
                    segment
                })
                .collect()
        }

        fn from_artifact(
            base: &NpaModel,
            examples: &[BurnE2eRolloutExample],
            config: BurnE2eRolloutTrainConfig,
            initial: &E2eHyperNpa2d,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            initial.validate()?;
            let first = examples.first().ok_or_else(|| {
                AutomataError::InvalidArgument("HyperNPA e2e generator requires examples".into())
            })?;
            if examples.iter().any(|example| {
                example.embed_dims != first.embed_dims
                    || example.token_count != first.token_count
            }) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e examples must have homogeneous condition token shapes"
                        .to_string(),
                ));
            }
            let expected_kind = config.generator_kind;
            if initial.architecture != expected_kind.artifact_architecture() {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA architecture {:?} does not match configured {:?}",
                    initial.architecture,
                    expected_kind.artifact_architecture()
                )));
            }
            if initial.uses_canonical_full_rank_lora() != config.canonical_full_rank_lora {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter parameterization {:?} does not match configured {:?}",
                    initial
                        .adapter_parameterization
                        .as_deref()
                        .unwrap_or(E2E_HYPER_ADAPTER_FACTORIZED),
                    if config.canonical_full_rank_lora {
                        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
                    } else {
                        E2E_HYPER_ADAPTER_FACTORIZED
                    }
                )));
            }
            let (adapter_constants, adapter_trainable_mask) =
                Self::adapter_parameterization_tensors(base, config, device)?;
            let initial_embed_dims = initial.embed_dims()?;
            let adding_rgb_channels = expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && !initial.condition_rgb_channels.unwrap_or(false)
                && config.dino_rgb_channels
                && first.embed_dims == initial_embed_dims + 3;
            if initial.condition_token_count.is_some_and(|value| value != first.token_count)
                || (initial_embed_dims != first.embed_dims && !adding_rgb_channels)
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA condition shape {:?}x{} does not match training {}x{}",
                    initial.condition_token_count,
                    initial.embed_dims()?,
                    first.token_count,
                    first.embed_dims
                )));
            }
            let adapter = initial.adapter_spec(&base.config)?;
            if adapter.rank != config.adapter_rank
                || (adapter.alpha - config.adapter_alpha).abs() > f32::EPSILON
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter rank/alpha {}/{} does not match configured {}/{}",
                    adapter.rank, adapter.alpha, config.adapter_rank, config.adapter_alpha
                )));
            }
            if expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && (initial.hidden_dims != config.generator_hidden_dims
                    || initial.token_attention_heads != config.token_attention_heads.max(1)
                    || initial.sample_steps != config.generator_sample_steps.max(1)
                    || (initial.output_scale - config.generator_output_scale).abs()
                        > f32::EPSILON)
            {
                return Err(AutomataError::InvalidModel(
                    "warm-start HyperNPA hidden/sample/output-scale contract does not match the configured generator"
                        .to_string(),
                ));
            }
            if expected_kind != E2eHyperGeneratorKind::SampleIdTable
                && ((!adding_rgb_channels
                    && (initial.condition_rgb_channels.unwrap_or(false)
                        != config.dino_rgb_channels
                        || (initial.condition_rgb_channel_scale.unwrap_or(1.0)
                            - config.dino_rgb_channel_scale)
                            .abs()
                            > f32::EPSILON))
                    || initial.condition_alpha_channel.unwrap_or(false)
                        != config.dino_alpha_channel
                    || (initial.condition_alpha_channel_scale.unwrap_or(1.0)
                        - config.dino_alpha_channel_scale)
                        .abs()
                        > f32::EPSILON
                    || initial.condition_l2_normalize_features.unwrap_or(true)
                        != config.dino_l2_normalize_features)
            {
                return Err(AutomataError::InvalidModel(
                    "warm-start HyperNPA DINO RGB/alpha/normalization contract does not match the configured condition pipeline"
                        .to_string(),
                ));
            }
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            if initial.output_dims != output_dims {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA output dims {} do not match model adapter dims {output_dims}",
                    initial.output_dims
                )));
            }
            if expected_kind == E2eHyperGeneratorKind::SampleIdTable {
                if first.token_count != 1 || initial.hidden_dims != 1 {
                    return Err(AutomataError::InvalidModel(
                        "sample-ID adapter table requires one token and hidden_dims=1".to_string(),
                    ));
                }
                let placeholder = |values: Vec<f32>| tracked_tensor(values, [1, 1], device);
                return Ok(Self {
                    kind: expected_kind,
                    token_w: tracked_tensor(
                        initial.weights.token_w.clone(),
                        [output_dims, first.embed_dims],
                        device,
                    ),
                    token_b: placeholder(initial.weights.token_b.clone()),
                    token_gate_w: placeholder(initial.weights.token_gate_w.clone()),
                    token_gate_b: placeholder(initial.weights.token_gate_b.clone()),
                    state_w: placeholder(initial.weights.state_w.clone()),
                    time_w: placeholder(initial.weights.time_w.clone()),
                    output_w: placeholder(initial.weights.output_w.clone()),
                    output_b: placeholder(initial.weights.output_b.clone()),
                    condition_control_w: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [base.config.update_dims(), 1],
                        device,
                    ),
                    condition_control_b: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [1, base.config.update_dims()],
                        device,
                    ),
                    condition_control_state_w: tracked_tensor(
                        vec![0.0; base.config.state_dims],
                        [1, base.config.state_dims],
                        device,
                    ),
                    hidden_dims: 1,
                    token_attention_heads: 1,
                    softmax_token_attention: false,
                    canonical_full_rank_lora: config.canonical_full_rank_lora,
                    adapter_constants,
                    adapter_trainable_mask,
                    adapter_parameter_segments: Self::adapter_parameter_segments(
                        &base.config,
                        config.adapter_rank,
                    ),
                    output_dims,
                    output_scale: 1.0,
                    sample_steps: 1,
                    adapter_chunk_size: output_dims,
                    output_chunks: 1,
                });
            }
            let adapter_chunk_size = if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                output_dims
            } else {
                initial.adapter_chunk_size_value()
            };
            if adapter_chunk_size != config.adapter_chunk_size.max(1).min(output_dims).max(1)
                && expected_kind != E2eHyperGeneratorKind::PooledFlow
            {
                return Err(AutomataError::InvalidModel(format!(
                    "warm-start HyperNPA adapter chunk size {adapter_chunk_size} does not match configured {}",
                    config.adapter_chunk_size
                )));
            }
            let output_chunks = initial.weights.output_b.len() / adapter_chunk_size;
            let expected_control = config.spatial_condition_control;
            let initial_has_control = initial.has_spatial_condition_control();
            if initial_has_control && !expected_control {
                return Err(AutomataError::InvalidModel(
                    "cannot warm-start a per-step condition-field HyperNPA as a static-adapter model"
                        .to_string(),
                ));
            }
            let initial_has_state_control =
                initial.spatial_condition_state_control.unwrap_or(false);
            if initial_has_state_control && !config.spatial_condition_state_control {
                return Err(AutomataError::InvalidModel(
                    "cannot warm-start a state-conditioned field as a state-independent field"
                        .to_string(),
                ));
            }
            let update_dims = base.config.update_dims();
            let control_w = if initial_has_control {
                initial.weights.condition_control_w.clone()
            } else if expected_control {
                let mut rng = StdRng::seed_from_u64(config.seed ^ 0xc01d_f1e1_d2d0);
                seeded_values(
                    update_dims * initial.hidden_dims,
                    config.generator_output_init_scale
                        / (initial.hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                )
            } else {
                vec![0.0; update_dims * initial.hidden_dims]
            };
            let control_b = if initial_has_control {
                initial.weights.condition_control_b.clone()
            } else {
                vec![0.0; update_dims]
            };
            let control_state_w = if initial_has_state_control {
                if initial.weights.condition_control_state_w.len()
                    != initial.hidden_dims * base.config.state_dims
                {
                    return Err(AutomataError::InvalidModel(format!(
                        "warm-start condition state projection has {} values, expected {}",
                        initial.weights.condition_control_state_w.len(),
                        initial.hidden_dims * base.config.state_dims,
                    )));
                }
                initial.weights.condition_control_state_w.clone()
            } else {
                vec![0.0; initial.hidden_dims * base.config.state_dims]
            };
            let token_w = if adding_rgb_channels {
                let has_alpha = initial.condition_alpha_channel.unwrap_or(false);
                let semantic_dims = initial_embed_dims - usize::from(has_alpha);
                let mut expanded = vec![0.0; initial.hidden_dims * first.embed_dims];
                for hidden in 0..initial.hidden_dims {
                    let old = &initial.weights.token_w
                        [hidden * initial_embed_dims..(hidden + 1) * initial_embed_dims];
                    let new = &mut expanded
                        [hidden * first.embed_dims..(hidden + 1) * first.embed_dims];
                    new[..semantic_dims].copy_from_slice(&old[..semantic_dims]);
                    if has_alpha {
                        new[first.embed_dims - 1] = old[initial_embed_dims - 1];
                    }
                }
                eprintln!(
                    "warm-starting HyperNPA with RGB token channels: condition projection {} -> {} dimensions",
                    initial_embed_dims, first.embed_dims,
                );
                expanded
            } else {
                initial.weights.token_w.clone()
            };
            Ok(Self {
                kind: expected_kind,
                token_w: tracked_tensor(token_w, [initial.hidden_dims, first.embed_dims], device),
                token_b: tracked_tensor(
                    initial.weights.token_b.clone(),
                    [1, initial.hidden_dims],
                    device,
                ),
                token_gate_w: tracked_tensor(
                    initial.weights.token_gate_w.clone(),
                    if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                        [initial.token_attention_heads, initial.hidden_dims]
                    } else {
                        [output_chunks, initial.hidden_dims]
                    },
                    device,
                ),
                token_gate_b: tracked_tensor(
                    initial.weights.token_gate_b.clone(),
                    if expected_kind == E2eHyperGeneratorKind::PooledFlow {
                        [1, initial.token_attention_heads]
                    } else {
                        [output_chunks, initial.hidden_dims]
                    },
                    device,
                ),
                state_w: tracked_tensor(
                    initial.weights.state_w.clone(),
                    [initial.hidden_dims, adapter_chunk_size],
                    device,
                ),
                time_w: tracked_tensor(
                    initial.weights.time_w.clone(),
                    [initial.hidden_dims, 1],
                    device,
                ),
                output_w: tracked_tensor(
                    initial.weights.output_w.clone(),
                    [adapter_chunk_size, initial.hidden_dims],
                    device,
                ),
                output_b: tracked_tensor(
                    initial.weights.output_b.clone(),
                    [output_chunks, adapter_chunk_size],
                    device,
                ),
                condition_control_w: tracked_tensor(
                    control_w,
                    [update_dims, initial.hidden_dims],
                    device,
                ),
                condition_control_b: tracked_tensor(control_b, [1, update_dims], device),
                condition_control_state_w: tracked_tensor(
                    control_state_w,
                    [initial.hidden_dims, base.config.state_dims],
                    device,
                ),
                hidden_dims: initial.hidden_dims,
                token_attention_heads: initial.token_attention_heads,
                softmax_token_attention: config.softmax_token_attention,
                canonical_full_rank_lora: config.canonical_full_rank_lora,
                adapter_constants,
                adapter_trainable_mask,
                adapter_parameter_segments: Self::adapter_parameter_segments(
                    &base.config,
                    config.adapter_rank,
                ),
                output_dims,
                output_scale: initial.output_scale,
                sample_steps: initial.sample_steps,
                adapter_chunk_size,
                output_chunks,
            })
        }

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
                example.embed_dims != embed_dims || example.token_count != token_count
            }) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e examples must have homogeneous condition token shapes"
                        .to_string(),
                ));
            }
            let output_dims =
                NpaLowRankAdapter::parameter_count_for_config(&base.config, config.adapter_rank);
            let (adapter_constants, adapter_trainable_mask) =
                Self::adapter_parameterization_tensors(base, config, device)?;
            let kind = config.generator_kind;
            if kind == E2eHyperGeneratorKind::SampleIdTable {
                if token_count != 1 {
                    return Err(AutomataError::InvalidArgument(
                        "sample-ID adapter table requires exactly one condition token".to_string(),
                    ));
                }
                let initial_adapter = if config.canonical_full_rank_lora {
                    vec![0.0; output_dims]
                } else {
                    NpaLowRankAdapter::seeded_zero_delta(
                        &base.config,
                        config.adapter_rank,
                        config.adapter_alpha,
                        config.seed ^ 0x5eed_10da,
                    )
                    .to_parameter_vector()
                };
                let mut table = vec![0.0; output_dims * embed_dims];
                for (output, value) in initial_adapter.into_iter().enumerate() {
                    table[output * embed_dims..(output + 1) * embed_dims].fill(value);
                }
                let placeholder = || tracked_tensor(vec![0.0], [1, 1], device);
                return Ok(Self {
                    kind,
                    token_w: tracked_tensor(table, [output_dims, embed_dims], device),
                    token_b: placeholder(),
                    token_gate_w: placeholder(),
                    token_gate_b: placeholder(),
                    state_w: placeholder(),
                    time_w: placeholder(),
                    output_w: placeholder(),
                    output_b: placeholder(),
                    condition_control_w: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [base.config.update_dims(), 1],
                        device,
                    ),
                    condition_control_b: tracked_tensor(
                        vec![0.0; base.config.update_dims()],
                        [1, base.config.update_dims()],
                        device,
                    ),
                    condition_control_state_w: tracked_tensor(
                        vec![0.0; base.config.state_dims],
                        [1, base.config.state_dims],
                        device,
                    ),
                    hidden_dims: 1,
                    token_attention_heads: 1,
                    softmax_token_attention: false,
                    canonical_full_rank_lora: config.canonical_full_rank_lora,
                    adapter_constants,
                    adapter_trainable_mask,
                    adapter_parameter_segments: Self::adapter_parameter_segments(
                        &base.config,
                        config.adapter_rank,
                    ),
                    output_dims,
                    output_scale: 1.0,
                    sample_steps: 1,
                    adapter_chunk_size: output_dims,
                    output_chunks: 1,
                });
            }
            let hidden_dims = config.generator_hidden_dims.max(1);
            let token_attention_heads = config.token_attention_heads.max(1);
            if kind == E2eHyperGeneratorKind::ModuleTokenDecoder
                && !hidden_dims.is_multiple_of(token_attention_heads)
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "module-token-decoder hidden_dims={hidden_dims} must be divisible by token_attention_heads={token_attention_heads}"
                )));
            }
            let adapter_chunk_size = match kind {
                E2eHyperGeneratorKind::PooledFlow => output_dims,
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => config
                    .adapter_chunk_size
                    .max(1)
                    .min(output_dims)
                    .max(1),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
            };
            let module_layout = kind
                .is_module_token_decoder()
                .then(|| {
                    crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                        &base.config,
                        config.adapter_rank,
                        adapter_chunk_size,
                    )
                })
                .transpose()?;
            let output_chunks = module_layout
                .as_ref()
                .map_or_else(|| output_dims.div_ceil(adapter_chunk_size), |layout| layout.chunk_count);
            let mut rng = StdRng::seed_from_u64(config.seed ^ 0xa11c_e2e0_7a5e);
            let token_w = tracked_tensor(
                seeded_values(
                    hidden_dims * embed_dims,
                    config.generator_condition_init_scale / (embed_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [hidden_dims, embed_dims],
                device,
            );
            let token_b = tracked_tensor(vec![0.0; hidden_dims], [1, hidden_dims], device);
            let (token_gate_w, token_gate_b, state_w) = match kind {
                E2eHyperGeneratorKind::PooledFlow => (
                    tracked_tensor(
                        seeded_values(
                            token_attention_heads * hidden_dims,
                            config.generator_condition_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [token_attention_heads, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        vec![0.0; token_attention_heads],
                        [1, token_attention_heads],
                        device,
                    ),
                    tracked_tensor(
                        seeded_values(
                            hidden_dims * output_dims,
                            config.generator_condition_init_scale
                                / (output_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [hidden_dims, output_dims],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => (
                    tracked_tensor(
                        seeded_values(
                            output_chunks * hidden_dims,
                            config.generator_condition_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [output_chunks, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        module_layout.as_ref().map_or_else(
                            || vec![0.0; output_chunks * hidden_dims],
                            |layout| {
                                layout.structured_query_initialization(
                                    hidden_dims,
                                    config.generator_condition_init_scale,
                                )
                            },
                        ),
                        [output_chunks, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        seeded_values(
                            hidden_dims * adapter_chunk_size,
                            config.generator_condition_init_scale
                                / (adapter_chunk_size as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [hidden_dims, adapter_chunk_size],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
            };
            let time_w = tracked_tensor(
                seeded_values(
                    hidden_dims,
                    config.generator_condition_init_scale,
                    &mut rng,
                ),
                [hidden_dims, 1],
                device,
            );
            let (output_w, output_b) = match kind {
                E2eHyperGeneratorKind::PooledFlow => (
                    tracked_tensor(
                        seeded_values(
                            output_dims * hidden_dims,
                            config.generator_output_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [output_dims, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        if config.canonical_full_rank_lora {
                            vec![0.0; output_dims]
                        } else {
                            seeded_zero_delta_output_bias(
                                &base.config,
                                config.adapter_rank,
                                config.adapter_alpha,
                                config.seed ^ 0x5eed_10da,
                                config.generator_output_scale,
                            )
                        },
                        [1, output_dims],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SpatialTokenFlow
                | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                | E2eHyperGeneratorKind::ModuleTokenDecoder => (
                    tracked_tensor(
                        seeded_values(
                            adapter_chunk_size * hidden_dims,
                            config.generator_output_init_scale
                                / (hidden_dims as f32).sqrt().max(1.0),
                            &mut rng,
                        ),
                        [adapter_chunk_size, hidden_dims],
                        device,
                    ),
                    tracked_tensor(
                        if config.canonical_full_rank_lora {
                            vec![0.0; output_chunks * adapter_chunk_size]
                        } else {
                            seeded_zero_delta_chunk_output_bias(
                                &base.config,
                                config.adapter_rank,
                                config.adapter_alpha,
                                config.seed ^ 0x5eed_10da,
                                adapter_chunk_size,
                                output_chunks,
                                module_layout.as_ref(),
                            )
                        },
                        [output_chunks, adapter_chunk_size],
                        device,
                    ),
                ),
                E2eHyperGeneratorKind::SampleIdTable => unreachable!(),
            };
            let condition_control_w = tracked_tensor(
                seeded_values(
                    base.config.update_dims() * hidden_dims,
                    config.generator_output_init_scale
                        / (hidden_dims as f32).sqrt().max(1.0),
                    &mut rng,
                ),
                [base.config.update_dims(), hidden_dims],
                device,
            );
            let condition_control_b = tracked_tensor(
                vec![0.0; base.config.update_dims()],
                [1, base.config.update_dims()],
                device,
            );
            let condition_control_state_w = tracked_tensor(
                vec![0.0; hidden_dims * base.config.state_dims],
                [hidden_dims, base.config.state_dims],
                device,
            );
            Ok(Self {
                kind,
                token_w,
                token_b,
                token_gate_w,
                token_gate_b,
                state_w,
                time_w,
                output_w,
                output_b,
                condition_control_w,
                condition_control_b,
                condition_control_state_w,
                hidden_dims,
                token_attention_heads,
                softmax_token_attention: config.softmax_token_attention,
                canonical_full_rank_lora: config.canonical_full_rank_lora,
                adapter_constants,
                adapter_trainable_mask,
                adapter_parameter_segments: Self::adapter_parameter_segments(
                    &base.config,
                    config.adapter_rank,
                ),
                output_dims,
                output_scale: config.generator_output_scale,
                sample_steps: config.generator_sample_steps.max(1),
                adapter_chunk_size,
                output_chunks,
            })
        }

        fn token_hidden_batch(&self, condition: Tensor3) -> Tensor3 {
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let tokens = dims[1];
            let embed_dims = dims[2];
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
            relu(condition.matmul(token_w) + token_b)
        }

        fn condition_control_batch(
            &self,
            condition: Tensor3,
            config: BurnE2eRolloutTrainConfig,
        ) -> Option<BurnE2eConditionControlBatch> {
            if !config.spatial_condition_control {
                return None;
            }
            let dims = condition.shape().dims::<3>();
            let tokens = dims[1];
            let grid_width = config.dino_token_grid_width.max(1);
            let grid_height = config.dino_token_grid_height.max(1);
            let patch_tokens = grid_width.saturating_mul(grid_height);
            if tokens <= 1 || patch_tokens == 0 || tokens < patch_tokens.saturating_add(1) {
                return None;
            }
            let token_hidden = self.token_hidden_batch(condition);
            Some(BurnE2eConditionControlBatch {
                patch_hidden: token_hidden.narrow(1, 1, patch_tokens),
                update_w: self.condition_control_w.clone(),
                update_b: self.condition_control_b.clone(),
                state_w: config
                    .spatial_condition_state_control
                    .then(|| self.condition_control_state_w.clone()),
                grid_width,
                grid_height,
                sigma: config.spatial_condition_control_sigma.max(1.0e-4),
                scale: config.spatial_condition_control_scale,
            })
        }

        fn apply_adapter_parameterization(&self, vector: Tensor2) -> Tensor2 {
            let batches = vector.shape().dims::<2>()[0];
            vector.mul(
                self.adapter_trainable_mask
                    .clone()
                    .expand([batches, self.output_dims]),
            ) + self
                .adapter_constants
                .clone()
                .expand([batches, self.output_dims])
        }

        fn adapter_batch(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            config: BurnE2eRolloutTrainConfig,
        ) -> BurnAdapterBatch {
            if self.kind == E2eHyperGeneratorKind::SampleIdTable {
                let dims = condition.shape().dims::<3>();
                debug_assert_eq!(dims[1], 1);
                let vector = condition
                    .squeeze_dim::<2>(1)
                    .matmul(self.token_w.clone().transpose());
                return BurnAdapterBatch::from_parameter_vector(
                    self.apply_adapter_parameterization(vector),
                    npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                );
            }
            if matches!(
                self.kind,
                E2eHyperGeneratorKind::SpatialTokenFlow
                    | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                    | E2eHyperGeneratorKind::ModuleTokenDecoder
            ) {
                let vector = self.spatial_token_adapter_vector_batch(
                    condition,
                    npa_config,
                    config.adapter_rank,
                );
                return BurnAdapterBatch::from_parameter_vector(
                    self.apply_adapter_parameterization(vector),
                    npa_config,
                    config.adapter_rank,
                    config.adapter_alpha,
                );
            }
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let tokens = dims[1];
            let device = condition.device();
            let token_hidden = self.token_hidden_batch(condition);
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
            let attention_logits = token_hidden.clone().matmul(gate_w) + gate_b;
            let attention_weights = if self.softmax_token_attention {
                softmax(attention_logits, 1)
            } else {
                let weights = attention_logits.tanh().exp();
                let denominator = weights
                    .clone()
                    .sum_dim(1)
                    .add_scalar(EPSILON)
                    .expand([batches, tokens, heads]);
                weights.div(denominator)
            };
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
                self.apply_adapter_parameterization(vector),
                npa_config,
                config.adapter_rank,
                config.adapter_alpha,
            )
        }

        fn spatial_token_adapter_vector_batch(
            &self,
            condition: Tensor3,
            npa_config: &NpaConfig,
            adapter_rank: usize,
        ) -> Tensor2 {
            let dims = condition.shape().dims::<3>();
            let batches = dims[0];
            let device = condition.device();
            let token_hidden = self.token_hidden_batch(condition);
            let mut chunks = Tensor::<BurnBackend, 3>::zeros(
                [batches, self.output_chunks, self.adapter_chunk_size],
                &device,
            );
            let query_base = self
                .token_gate_w
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.hidden_dims])
                + self
                    .token_gate_b
                    .clone()
                    .unsqueeze_dim::<3>(0)
                    .expand([batches, self.output_chunks, self.hidden_dims]);
            let shared_bias = self
                .token_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.hidden_dims]);
            let state_w = self
                .state_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.adapter_chunk_size, self.hidden_dims]);
            let output_w = self
                .output_w
                .clone()
                .transpose()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.hidden_dims, self.adapter_chunk_size]);
            let output_b = self
                .output_b
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([batches, self.output_chunks, self.adapter_chunk_size]);
            let attention_heads = if self.kind == E2eHyperGeneratorKind::ModuleTokenDecoder {
                self.token_attention_heads
            } else {
                1
            };
            let head_dims = self.hidden_dims / attention_heads;
            let attention_scale = 1.0 / (head_dims as f32).sqrt().max(1.0);
            for step in 0..self.sample_steps {
                let t = if self.sample_steps <= 1 {
                    0.0
                } else {
                    step as f32 / (self.sample_steps - 1) as f32
                };
                let time_hidden = self
                    .time_w
                    .clone()
                    .transpose()
                    .mul_scalar(t)
                    .unsqueeze_dim::<3>(0)
                    .expand([batches, self.output_chunks, self.hidden_dims]);
                let state_hidden = chunks.clone().matmul(state_w.clone());
                let query_hidden = relu(
                    query_base.clone() + shared_bias.clone() + state_hidden + time_hidden,
                );
                let attend = |query: Tensor3, tokens: Tensor3, hidden: usize| {
                    let attention_logits = query.matmul(tokens.clone().swap_dims(1, 2));
                    let attention_logits = attention_logits.mul_scalar(attention_scale);
                    let attention_weights = if self.softmax_token_attention {
                        softmax(attention_logits, 2)
                    } else {
                        let weights = attention_logits.tanh().exp();
                        let denominator = weights
                            .clone()
                            .sum_dim(2)
                            .add_scalar(EPSILON)
                            .expand([batches, self.output_chunks, dims[1]]);
                        weights.div(denominator)
                    };
                    debug_assert_eq!(tokens.shape().dims::<3>()[2], hidden);
                    attention_weights.matmul(tokens)
                };
                let attended = if attention_heads == 1 {
                    attend(query_hidden.clone(), token_hidden.clone(), self.hidden_dims)
                } else {
                    Tensor::cat(
                        (0..attention_heads)
                            .map(|head| {
                                let start = head * head_dims;
                                attend(
                                    query_hidden.clone().narrow(2, start, head_dims),
                                    token_hidden.clone().narrow(2, start, head_dims),
                                    head_dims,
                                )
                            })
                            .collect(),
                        2,
                    )
                };
                let hidden = relu(query_hidden + attended);
                let velocity = hidden.matmul(output_w.clone()) + output_b.clone();
                chunks = chunks
                    + velocity
                        .mul_scalar(self.output_scale)
                        .div_scalar(self.sample_steps as f32);
            }
            let padded = chunks.reshape([
                batches,
                self.output_chunks * self.adapter_chunk_size,
            ]);
            let vector = if self.kind.is_module_token_decoder() {
                let layout = crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                    npa_config,
                    adapter_rank,
                    self.adapter_chunk_size,
                )
                .expect("module adapter layout validated during generator construction");
                assert_eq!(layout.chunk_count, self.output_chunks);
                Tensor::cat(
                    layout
                        .segments
                        .iter()
                        .map(|segment| {
                            padded
                                .clone()
                                .narrow(1, segment.chunk_offset * self.adapter_chunk_size, segment.len)
                        })
                        .collect(),
                    1,
                )
            } else {
                padded.narrow(1, 0, self.output_dims)
            };
            vector
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
                self.condition_control_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.condition_control_w.clone().inner().zeros_like()),
                self.condition_control_b
                    .grad_remove(grads)
                    .unwrap_or_else(|| self.condition_control_b.clone().inner().zeros_like()),
                self.condition_control_state_w
                    .grad_remove(grads)
                    .unwrap_or_else(|| {
                        self.condition_control_state_w.clone().inner().zeros_like()
                    }),
            ];
            let normalize_sample_table = normalize && self.kind == E2eHyperGeneratorKind::SampleIdTable;
            let original_table_norm = (normalize_sample_table && collect_metrics)
                .then(|| group_norm_tensor(&tensors));
            if normalize_sample_table {
                tensors[0] = normalize_sample_id_table_gradient(
                    tensors[0].clone(),
                    &self.adapter_parameter_segments,
                );
            }
            let (prepared_norm, scale, scale_tensor) = prepare_grad_group(
                &mut tensors,
                cfg.grad_clip_norm,
                normalize && !normalize_sample_table,
                collect_metrics,
            )?;
            let norm = if let Some(original) = original_table_norm {
                finite_scalar(
                    "Burn sample-ID adapter table grad norm",
                    original.into_scalar(),
                )?
            } else {
                prepared_norm
            };
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
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_w = track(apply_adamw_tensor(
                self.condition_control_w.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_w_m,
                &mut state.condition_control_w_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_b = track(apply_adamw_tensor(
                self.condition_control_b.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_b_m,
                &mut state.condition_control_b_v,
                cfg,
                scale_tensor.clone(),
                bias,
            ));
            self.condition_control_state_w = track(apply_adamw_tensor(
                self.condition_control_state_w.clone().inner(),
                tensors.remove(0),
                &mut state.condition_control_state_w_m,
                &mut state.condition_control_state_w_v,
                cfg,
                scale_tensor,
                bias,
            ));
            Ok((norm, scale))
        }

        fn to_hyper(&self, config: BurnE2eRolloutTrainConfig) -> AutomataResult<E2eHyperNpa2d> {
            let image_conditioned = self.kind != E2eHyperGeneratorKind::SampleIdTable;
            Ok(E2eHyperNpa2d {
                version: 1,
                architecture: self.kind.artifact_architecture().to_string(),
                backend: Some(format!("{BACKEND}_e2e_rollout")),
                condition_encoder: None,
                condition_token_count: None,
                condition_embed_dims: None,
                condition_token_grid_width: None,
                condition_token_grid_height: None,
                condition_image_size: image_conditioned.then_some(config.dino_image_size),
                condition_alpha_mode: image_conditioned.then(|| "composite-white".to_string()),
                condition_rgb_channels: image_conditioned.then_some(config.dino_rgb_channels),
                condition_rgb_channel_scale: image_conditioned
                    .then_some(config.dino_rgb_channel_scale),
                condition_alpha_channel: image_conditioned.then_some(config.dino_alpha_channel),
                condition_alpha_channel_scale: image_conditioned
                    .then_some(config.dino_alpha_channel_scale),
                condition_l2_normalize_features: image_conditioned
                    .then_some(config.dino_l2_normalize_features),
                condition_resize_mode: image_conditioned.then(|| "stretch".to_string()),
                condition_application: Some(if config.spatial_condition_control {
                    "per-step-field"
                } else {
                    "static-adapter"
                }.to_string()),
                shared_base_sha256: None,
                hidden_dims: self.hidden_dims,
                token_attention_heads: self.token_attention_heads,
                attention_normalization: image_conditioned.then(|| {
                    if self.softmax_token_attention {
                        crate::hyper::e2e::E2E_HYPER_ATTENTION_SOFTMAX
                    } else {
                        crate::hyper::e2e::E2E_HYPER_ATTENTION_TANH_EXP
                    }
                    .to_string()
                }),
                output_dims: self.output_dims,
                sample_steps: self.sample_steps,
                output_scale: self.output_scale,
                adapter_rank: Some(config.adapter_rank),
                adapter_alpha: Some(config.adapter_alpha),
                adapter_parameterization: Some(
                    if self.canonical_full_rank_lora {
                        E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK
                    } else {
                        E2E_HYPER_ADAPTER_FACTORIZED
                    }
                    .to_string(),
                ),
                adapter_chunk_size: matches!(
                    self.kind,
                    E2eHyperGeneratorKind::SpatialTokenFlow
                        | E2eHyperGeneratorKind::ModuleTokenDecoderV2
                        | E2eHyperGeneratorKind::ModuleTokenDecoder
                )
                .then_some(self.adapter_chunk_size),
                spatial_condition_control: config.spatial_condition_control.then_some(true),
                spatial_condition_control_scale: config
                    .spatial_condition_control
                    .then_some(config.spatial_condition_control_scale),
                spatial_condition_control_sigma: config
                    .spatial_condition_control
                    .then_some(config.spatial_condition_control_sigma),
                spatial_condition_state_control: config
                    .spatial_condition_state_control
                    .then_some(true),
                weights: E2eHyperNpa2dWeights {
                    token_w: tensor_vec(self.token_w.clone().inner())?,
                    token_b: tensor_vec(self.token_b.clone().inner())?,
                    token_gate_w: tensor_vec(self.token_gate_w.clone().inner())?,
                    token_gate_b: tensor_vec(self.token_gate_b.clone().inner())?,
                    state_w: tensor_vec(self.state_w.clone().inner())?,
                    time_w: tensor_vec(self.time_w.clone().inner())?,
                    output_w: tensor_vec(self.output_w.clone().inner())?,
                    output_b: tensor_vec(self.output_b.clone().inner())?,
                    condition_control_w: if config.spatial_condition_control {
                        tensor_vec(self.condition_control_w.clone().inner())?
                    } else {
                        Vec::new()
                    },
                    condition_control_b: if config.spatial_condition_control {
                        tensor_vec(self.condition_control_b.clone().inner())?
                    } else {
                        Vec::new()
                    },
                    condition_control_state_w: if config.spatial_condition_state_control {
                        tensor_vec(self.condition_control_state_w.clone().inner())?
                    } else {
                        Vec::new()
                    },
                },
            })
        }

        fn detached(&self) -> Self {
            Self {
                kind: self.kind,
                token_w: detach2(self.token_w.clone()),
                token_b: detach2(self.token_b.clone()),
                token_gate_w: detach2(self.token_gate_w.clone()),
                token_gate_b: detach2(self.token_gate_b.clone()),
                state_w: detach2(self.state_w.clone()),
                time_w: detach2(self.time_w.clone()),
                output_w: detach2(self.output_w.clone()),
                output_b: detach2(self.output_b.clone()),
                condition_control_w: detach2(self.condition_control_w.clone()),
                condition_control_b: detach2(self.condition_control_b.clone()),
                condition_control_state_w: detach2(self.condition_control_state_w.clone()),
                hidden_dims: self.hidden_dims,
                token_attention_heads: self.token_attention_heads,
                softmax_token_attention: self.softmax_token_attention,
                canonical_full_rank_lora: self.canonical_full_rank_lora,
                adapter_constants: detach2(self.adapter_constants.clone()),
                adapter_trainable_mask: detach2(self.adapter_trainable_mask.clone()),
                adapter_parameter_segments: self.adapter_parameter_segments.clone(),
                output_dims: self.output_dims,
                output_scale: self.output_scale,
                sample_steps: self.sample_steps,
                adapter_chunk_size: self.adapter_chunk_size,
                output_chunks: self.output_chunks,
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

        fn detached(&self) -> Self {
            Self {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: detach2(self.w1_down.clone()),
                w1_up: detach2(self.w1_up.clone()),
                w2_down: detach2(self.w2_down.clone()),
                w2_up: detach2(self.w2_up.clone()),
                b1_delta: detach2(self.b1_delta.clone()),
                b2_delta: detach2(self.b2_delta.clone()),
            }
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

        fn select_rows(self, rows: &[usize]) -> Self {
            if rows.is_empty() {
                return self;
            }
            let device = self.w1_down.device();
            let indices = Tensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(
                    rows.iter().map(|row| *row as i64).collect::<Vec<_>>(),
                    [rows.len()],
                ),
                &device,
            );
            Self {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: self.w1_down.select(0, indices.clone()),
                w1_up: self.w1_up.select(0, indices.clone()),
                w2_down: self.w2_down.select(0, indices.clone()),
                w2_up: self.w2_up.select(0, indices.clone()),
                b1_delta: self.b1_delta.select(0, indices.clone()),
                b2_delta: self.b2_delta.select(0, indices),
            }
        }

        fn select_rows_or_identity(self, rows: Option<&[usize]>) -> Self {
            match rows {
                Some(rows) => self.select_rows(rows),
                None => self,
            }
        }

        fn detached(&self) -> Self {
            Self {
                rank: self.rank,
                alpha: self.alpha,
                w1_down: detach3(self.w1_down.clone()),
                w1_up: detach3(self.w1_up.clone()),
                w2_down: detach3(self.w2_down.clone()),
                w2_up: detach3(self.w2_up.clone()),
                b1_delta: detach3(self.b1_delta.clone()),
                b2_delta: detach3(self.b2_delta.clone()),
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

        fn to_parameter_vector(&self) -> Tensor2 {
            let batches = self.w1_down.shape().dims::<3>()[0];
            Tensor::cat(
                vec![
                    self.w1_down.clone().reshape([batches, self.w1_down.shape().num_elements() / batches]),
                    self.w1_up.clone().reshape([batches, self.w1_up.shape().num_elements() / batches]),
                    self.w2_down.clone().reshape([batches, self.w2_down.shape().num_elements() / batches]),
                    self.w2_up.clone().reshape([batches, self.w2_up.shape().num_elements() / batches]),
                    self.b1_delta.clone().reshape([batches, self.b1_delta.shape().num_elements() / batches]),
                    self.b2_delta.clone().reshape([batches, self.b2_delta.shape().num_elements() / batches]),
                ],
                1,
            )
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
                    particle_count: example.particle_count.unwrap_or(config.rollout_particles),
                    update_prob: example.update_prob.unwrap_or(config.update_prob),
                    seed_scale: example.seed_scale.unwrap_or(config.seed_scale),
                    target_cpu: example.target.clone(),
                })
            })
            .collect()
    }

    fn burn_e2e_targets_for_indices_with_runtime(
        examples: &[BurnE2eRolloutExample],
        indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
        particle_count: Option<usize>,
        update_prob: Option<f32>,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let direct_config = direct_config_view(config);
        let pixel_xy = burn_e2e_pixel_xy(config, device);
        indices
            .iter()
            .map(|idx| {
                examples.get(*idx).ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e target index out of bounds".to_string(),
                    )
                })
            })
            .collect::<AutomataResult<Vec<_>>>()?
            .into_iter()
            .map(|example| {
                prepare_e2e_cpu_target(
                    BurnE2eCpuTargetInput {
                        target: example.target.clone(),
                        particle_count: particle_count.unwrap_or(example.particle_count).max(1),
                        update_prob: update_prob.unwrap_or(example.update_prob),
                        seed_scale: example.seed_scale,
                    },
                    direct_config,
                )
                .map(|prepared| prepared.into_burn(&pixel_xy, device))
            })
            .collect()
    }

    fn burn_e2e_pixel_xy(config: BurnE2eRolloutTrainConfig, device: &BurnDevice) -> Tensor2 {
        let image_size = direct_config_view(config).loss_config.image_size;
        tensor(
            pixel_xy_values(image_size),
            [image_size * image_size, 2],
            device,
        )
    }

    fn e2e_target_cache_bytes(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> usize {
        let pixels = config
            .loss_config
            .image_size
            .saturating_mul(config.loss_config.image_size);
        examples.iter().fold(0usize, |bytes, example| {
            let floats = pixels
                .saturating_mul(5)
                .saturating_add(2)
                .saturating_add(example.target.point_count().saturating_mul(2));
            bytes.saturating_add(floats.saturating_mul(std::mem::size_of::<f32>()))
        })
    }

    fn burn_e2e_target_cache(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        pixel_xy: &Tensor2,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let direct_config = direct_config_view(config);
        let prepared = examples
            .par_iter()
            .map(|example| {
                prepare_e2e_cpu_target(
                    BurnE2eCpuTargetInput {
                        target: example.target.clone(),
                        particle_count: example.particle_count,
                        update_prob: example.update_prob,
                        seed_scale: example.seed_scale,
                    },
                    direct_config,
                )
            })
            .collect::<AutomataResult<Vec<_>>>()?;
        burn_e2e_prepared_targets_to_burn(prepared, pixel_xy, device)
    }

    fn spawn_e2e_cpu_batch_prefetch(
        examples: &[BurnE2eRolloutExample],
        conditions: &BurnE2eConditionCache,
        indices: Vec<usize>,
        config: BurnE2eRolloutTrainConfig,
        targets_cached: bool,
    ) -> AutomataResult<BurnE2eCpuBatchPrefetch> {
        let mut target_inputs = Vec::with_capacity(indices.len());
        if !targets_cached {
            for &idx in &indices {
                let example = examples.get(idx).ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e prefetch target index out of bounds".to_string(),
                    )
                })?;
                target_inputs.push(BurnE2eCpuTargetInput {
                    target: example.target.clone(),
                    particle_count: example.particle_count,
                    update_prob: example.update_prob,
                    seed_scale: example.seed_scale,
                });
            }
        }
        let condition_paths = conditions.dynamic_dino_paths_for_indices(&indices)?;
        let pending_indices = indices.clone();
        Ok(BurnE2eCpuBatchPrefetch {
            indices: pending_indices,
            handle: thread::spawn(move || {
                prepare_e2e_cpu_batch(indices, target_inputs, condition_paths, config)
            }),
        })
    }

    fn join_e2e_cpu_batch_prefetch(
        handle: BurnE2eCpuBatchPrefetch,
    ) -> AutomataResult<BurnE2ePreparedCpuBatch> {
        handle
            .handle
            .join()
            .map_err(|_| {
                AutomataError::InvalidArgument("HyperNPA e2e CPU prefetch panicked".to_string())
            })?
            .map_err(AutomataError::InvalidArgument)
    }

    fn prepare_e2e_cpu_batch(
        indices: Vec<usize>,
        target_inputs: Vec<BurnE2eCpuTargetInput>,
        condition_paths: Option<Vec<PathBuf>>,
        config: BurnE2eRolloutTrainConfig,
    ) -> Result<BurnE2ePreparedCpuBatch, String> {
        let direct_config = direct_config_view(config);
        let (targets, prepared_dino) = rayon::join(
            move || {
                target_inputs
                    .into_par_iter()
                    .map(|input| {
                        prepare_e2e_cpu_target(input, direct_config).map_err(|err| err.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
            move || match condition_paths {
                Some(paths) => prepare_dino_condition_batch_for_prefetch(paths, config.dino_image_size)
                    .map(Some),
                None => Ok(None),
            },
        );
        let targets = targets?;
        let prepared_dino = prepared_dino?;
        Ok(BurnE2ePreparedCpuBatch {
            indices,
            targets,
            prepared_dino,
        })
    }

    fn prepare_e2e_cpu_target(
        input: BurnE2eCpuTargetInput,
        config: DirectBasisTrainConfig,
    ) -> AutomataResult<BurnE2ePreparedTargetExample> {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let render = render_target_2d_splat(&input.target, config.loss_config)?;
        let foreground = target_2d_foreground_mask(&input.target, config.loss_config)?;
        let target_foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
        let target_mean = input.target.mean_position();
        let target_positions = input
            .target
            .positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        Ok(BurnE2ePreparedTargetExample {
            target_rgb: render.rgb,
            target_density: render.density,
            target_foreground: foreground,
            target_foreground_scale,
            target_mean,
            target_positions,
            pixel_size: input.target.pixel_size,
            target_points: input.target.point_count(),
            particle_count: input.particle_count.max(1),
            update_prob: input.update_prob,
            seed_scale: input.seed_scale,
            target_cpu: input.target,
        })
    }

    fn burn_e2e_prepared_targets_to_burn(
        targets: Vec<BurnE2ePreparedTargetExample>,
        pixel_xy: &Tensor2,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        targets
            .into_iter()
            .map(|target| Ok(target.into_burn(pixel_xy, device)))
            .collect()
    }

    impl BurnE2ePreparedTargetExample {
        fn into_burn(self, pixel_xy: &Tensor2, device: &BurnDevice) -> BurnTargetExample {
            let pixels = self.target_rgb.len() / 3;
            let target_position_count = self.target_positions.len() / 2;
            BurnTargetExample {
                target_rgb: tensor(self.target_rgb, [pixels, 3], device),
                target_density: tensor(self.target_density, [pixels, 1], device),
                target_foreground: tensor(self.target_foreground, [pixels, 1], device),
                target_foreground_scale: self.target_foreground_scale,
                target_mean: tensor(self.target_mean.to_vec(), [1, 2], device),
                target_positions: tensor(
                    self.target_positions,
                    [target_position_count, 2],
                    device,
                ),
                pixel_xy: pixel_xy.clone(),
                pixel_size: self.pixel_size,
                target_points: self.target_points,
                particle_count: self.particle_count,
                update_prob: self.update_prob,
                seed_scale: self.seed_scale,
                target_cpu: self.target_cpu,
            }
        }
    }

    fn e2e_cpu_prefetch_depth(batch_size: usize, steps: usize) -> usize {
        if steps <= 1 {
            return 1;
        }
        let depth = if batch_size >= 256 { 2 } else { 4 };
        depth.min(steps).max(1)
    }

    #[cfg(feature = "dino")]
    fn prepare_dino_condition_batch_for_prefetch(
        paths: Vec<PathBuf>,
        image_size: usize,
    ) -> Result<DinoVitsPreparedConditionBatch, String> {
        let images = paths
            .into_par_iter()
            .map(|path| load_dino_condition_image(&path).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        DinoVitsPreparedConditionBatch::from_conditions(&images, image_size)
            .map_err(|err| err.to_string())
    }

    #[cfg(not(feature = "dino"))]
    fn prepare_dino_condition_batch_for_prefetch(
        _paths: Vec<PathBuf>,
        _image_size: usize,
    ) -> Result<BurnE2ePreparedDinoBatch, String> {
        Err("DINO prefetch requires the dino feature".to_string())
    }

    fn direct_config_view(config: BurnE2eRolloutTrainConfig) -> DirectBasisTrainConfig {
        DirectBasisTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            example_batch_size: config.example_batch_size,
            tbptt_chunk_steps: config.tbptt_chunk_steps,
            loss_on_final_chunk_only: config.loss_on_final_chunk_only,
            use_particle_pool: config.use_particle_pool,
            pool_size: config.pool_slots_per_example.max(1),
            inject_seed_interval: config.inject_seed_interval,
            brush_size: config.brush_size,
            stopgrad_pos: config.stopgrad_pos,
            stopgrad_state: config.stopgrad_state,
            rollout_particles: config.rollout_particles,
            rollout_step_min: config.rollout_step_min,
            rollout_steps: config.rollout_steps,
            update_prob: config.update_prob,
            seed: config.seed,
            seed_scale: config.seed_scale,
            seed_mode: config.seed_mode,
            grid_eps: config.grid_eps,
            motion_scale: config.motion_scale,
            loss_config: config.loss_config,
            target2d_loss_backend: config.target2d_loss_backend,
            perception_backend: config.perception_backend,
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
        direct.rollout_step_min = config.validation_steps.max(1);
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
            device_cache_max_bytes: usize,
            config: BurnE2eRolloutTrainConfig,
        ) -> AutomataResult<Self> {
            if examples.is_empty() {
                return Ok(Self {
                    values: BurnE2eConditionValues::HostRows(Vec::new()),
                    teacher_vectors: None,
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
            let teacher_vectors = if examples.iter().all(|example| example.teacher_adapter.is_some()) {
                let teacher_len = examples[0]
                    .teacher_adapter
                    .as_ref()
                    .map_or(0, Vec::len);
                if teacher_len == 0
                    || examples.iter().any(|example| {
                        example.teacher_adapter.as_ref().map(Vec::len) != Some(teacher_len)
                    })
                {
                    return Err(AutomataError::InvalidArgument(
                        "HyperNPA teacher adapter vectors must have one homogeneous non-empty shape"
                            .to_string(),
                    ));
                }
                Some(tensor(
                    examples
                        .iter()
                        .flat_map(|example| example.teacher_adapter.as_ref().unwrap().iter().copied())
                        .collect(),
                    [examples.len(), teacher_len],
                    device,
                ))
            } else if examples.iter().any(|example| example.teacher_adapter.is_some()) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA examples must either all or none provide teacher adapters".to_string(),
                ));
            } else {
                None
            };
            let feature_bytes = examples
                .len()
                .saturating_mul(row_len)
                .saturating_mul(std::mem::size_of::<f32>());
            let static_rows = examples
                .iter()
                .all(|example| example.condition_features.len() == row_len);
            let dynamic_dino_rows = examples
                .iter()
                .all(|example| example.condition_features.is_empty() && example.condition_path.is_some());
            if !static_rows && !dynamic_dino_rows {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition examples must be all static feature rows or all DINO image paths".to_string(),
                ));
            }
            if dynamic_dino_rows {
                #[cfg(feature = "dino")]
                {
                    let model_path = first.dino_model_path.as_ref().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "DINO on-demand condition source requires condition.dino_model"
                                .to_string(),
                        )
                    })?;
                    let encoder =
                        DinoVitsConditionEncoderBackend::<InnerBackend>::load(
                            model_path,
                            config.dino_image_size,
                        )
                        .map_err(|err| {
                            AutomataError::InvalidArgument(format!(
                                "failed to load DINO model {}: {err}",
                                model_path.display()
                            ))
                        })?;
                    let paths = examples
                        .iter()
                        .map(|example| {
                            example.condition_path.clone().ok_or_else(|| {
                                AutomataError::InvalidArgument(
                                    "DINO condition example is missing condition_path".to_string(),
                                )
                            })
                        })
                        .collect::<AutomataResult<Vec<_>>>()?;
                    let source = BurnE2eDinoConditionSource {
                        paths,
                        encoder,
                        batch_size: config.dino_batch_size.max(1),
                        token_grid_width: config.dino_token_grid_width,
                        token_grid_height: config.dino_token_grid_height,
                        l2_normalize_features: config.dino_l2_normalize_features,
                        rgb_channels: config.dino_rgb_channels,
                        rgb_channel_scale: config.dino_rgb_channel_scale,
                        alpha_channel: config.dino_alpha_channel,
                        alpha_channel_scale: config.dino_alpha_channel_scale,
                    };
                    let values = if device_cache_max_bytes > 0
                        && feature_bytes <= device_cache_max_bytes
                    {
                        eprintln!(
                            "encoding {} DINO conditions into a bounded {:.2} GiB device token cache",
                            examples.len(),
                            feature_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        );
                        BurnE2eConditionValues::Device(source.encode_all_to_device(
                            token_count,
                            embed_dims,
                            device,
                        )?)
                    } else {
                        BurnE2eConditionValues::DynamicDino(Box::new(source))
                    };
                    return Ok(Self {
                        values,
                        teacher_vectors,
                        examples: examples.len(),
                        token_count,
                        embed_dims,
                        device: device.clone(),
                    });
                }
                #[cfg(not(feature = "dino"))]
                {
                    return Err(AutomataError::InvalidArgument(
                        "DINO on-demand condition source requires the dino feature".to_string(),
                    ));
                }
            }
            let use_device_cache =
                device_cache_max_bytes > 0 && feature_bytes <= device_cache_max_bytes;
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
                teacher_vectors,
                examples: examples.len(),
                token_count,
                embed_dims,
                device: device.clone(),
            })
        }

        fn select_teacher(&self, indices: &[usize]) -> Option<Tensor2> {
            self.teacher_vectors.as_ref().map(|teachers| {
                teachers.clone().select(
                    0,
                    Tensor::<BurnBackend, 1, Int>::from_data(
                        TensorData::new(
                            indices.iter().map(|index| *index as i64).collect::<Vec<_>>(),
                            [indices.len()],
                        ),
                        &self.device,
                    ),
                )
            })
        }

        fn mean_pairwise_l2(&self) -> AutomataResult<Option<f32>> {
            if self.examples < 2 {
                return Ok(None);
            }
            let indices = (0..self.examples).collect::<Vec<_>>();
            let values = tensor3_vec(self.select(&indices)?.inner())?;
            let row_len = self.token_count * self.embed_dims;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..self.examples {
                for rhs in lhs + 1..self.examples {
                    let lhs = &values[lhs * row_len..(lhs + 1) * row_len];
                    let rhs = &values[rhs * row_len..(rhs + 1) * row_len];
                    let distance = lhs
                        .iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| {
                            let delta = f64::from(*lhs - *rhs);
                            delta * delta
                        })
                        .sum::<f64>()
                        .sqrt();
                    sum += distance;
                    pairs += 1;
                }
            }
            Ok(Some((sum / pairs.max(1) as f64) as f32))
        }

        fn nearest_rows(
            &self,
            queries: &Self,
            query_indices: &[usize],
        ) -> AutomataResult<Vec<(usize, f32)>> {
            if self.examples == 0 {
                return Err(AutomataError::InvalidArgument(
                    "nearest-condition lookup requires non-empty reference conditions".to_string(),
                ));
            }
            if self.token_count != queries.token_count || self.embed_dims != queries.embed_dims {
                return Err(AutomataError::InvalidArgument(
                    "nearest-condition lookup requires matching token shapes".to_string(),
                ));
            }
            let reference_indices = (0..self.examples).collect::<Vec<_>>();
            let references = tensor3_vec(self.select(&reference_indices)?.inner())?;
            let query_values = tensor3_vec(queries.select(query_indices)?.inner())?;
            let row_len = self.token_count * self.embed_dims;
            Ok(query_values
                .chunks_exact(row_len)
                .map(|query| {
                    references
                        .chunks_exact(row_len)
                        .enumerate()
                        .map(|(idx, reference)| {
                            let squared = query
                                .iter()
                                .zip(reference)
                                .map(|(query, reference)| {
                                    let delta = f64::from(*query - *reference);
                                    delta * delta
                                })
                                .sum::<f64>();
                            (idx, squared)
                        })
                        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
                        .map(|(idx, squared)| (idx, squared.sqrt() as f32))
                        .expect("non-empty nearest-condition reference set")
                })
                .collect())
        }

        fn mean_teacher_pairwise_l2(&self) -> AutomataResult<Option<f32>> {
            let Some(teachers) = &self.teacher_vectors else {
                return Ok(None);
            };
            if self.examples < 2 {
                return Ok(None);
            }
            let dims = teachers.shape().dims::<2>();
            let values = tensor_vec(teachers.clone().inner())?;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..self.examples {
                for rhs in lhs + 1..self.examples {
                    let lhs = &values[lhs * dims[1]..(lhs + 1) * dims[1]];
                    let rhs = &values[rhs * dims[1]..(rhs + 1) * dims[1]];
                    sum += lhs
                        .iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| {
                            let delta = f64::from(*lhs - *rhs);
                            delta * delta
                        })
                        .sum::<f64>()
                        .sqrt();
                    pairs += 1;
                }
            }
            Ok(Some((sum / pairs.max(1) as f64) as f32))
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
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(source) => {
                    source.select(indices, self.token_count, self.embed_dims)
                }
            }
        }

        fn select_prepared(
            &self,
            indices: &[usize],
            prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        ) -> AutomataResult<Tensor3> {
            #[cfg(feature = "dino")]
            if let (BurnE2eConditionValues::DynamicDino(source), Some(prepared)) =
                (&self.values, prepared_dino)
            {
                return source.encode_preprocessed(prepared, indices.len(), self.token_count, self.embed_dims);
            }
            self.select(indices)
        }

        fn dynamic_dino_paths_for_indices(
            &self,
            indices: &[usize],
        ) -> AutomataResult<Option<Vec<PathBuf>>> {
            if indices.iter().any(|idx| *idx >= self.examples) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache index out of bounds".to_string(),
                ));
            }
            #[cfg(feature = "dino")]
            if let BurnE2eConditionValues::DynamicDino(source) = &self.values {
                return indices
                    .iter()
                    .map(|idx| {
                        source.paths.get(*idx).cloned().ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "DINO condition source index out of bounds".to_string(),
                            )
                        })
                    })
                    .collect::<AutomataResult<Vec<_>>>()
                    .map(Some);
            }
            Ok(None)
        }

        fn feature_bytes(&self) -> usize {
            #[cfg(feature = "dino")]
            if matches!(self.values, BurnE2eConditionValues::DynamicDino(_)) {
                return 0;
            }
            self.examples
                .saturating_mul(self.token_count)
                .saturating_mul(self.embed_dims)
                .saturating_mul(std::mem::size_of::<f32>())
        }

        fn storage_label(&self) -> &'static str {
            if self.examples == 0 {
                return "empty";
            }
            match &self.values {
                BurnE2eConditionValues::Device(_) => "device-resident",
                BurnE2eConditionValues::HostRows(_) => "host-row-streamed",
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(_) => "dino-on-demand-device",
            }
        }

        fn is_device_resident(&self) -> bool {
            matches!(self.values, BurnE2eConditionValues::Device(_))
        }

        fn drained_cpu_features_from_examples(&self) -> bool {
            if self.examples == 0 {
                return false;
            }
            match &self.values {
                BurnE2eConditionValues::Device(_) | BurnE2eConditionValues::HostRows(_) => true,
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(_) => false,
            }
        }
    }

    #[cfg(feature = "dino")]
    impl BurnE2eDinoConditionSource {
        fn encode_all_to_device(
            &self,
            token_count: usize,
            embed_dims: usize,
            device: &BurnDevice,
        ) -> AutomataResult<Tensor3> {
            let mut values = Tensor::<InnerBackend, 3>::zeros(
                [self.paths.len(), token_count, embed_dims],
                device,
            );
            let batches = self.paths.len().div_ceil(self.batch_size);
            for (batch, paths) in self.paths.chunks(self.batch_size).enumerate() {
                let images = paths
                    .par_iter()
                    .map(|path| load_dino_condition_image(path))
                    .collect::<AutomataResult<Vec<_>>>()?;
                let encoded = self
                    .encode_loaded(&images, token_count, embed_dims)?
                    .inner();
                let start = batch * self.batch_size;
                let slots = (start..start + paths.len()).collect::<Vec<_>>();
                let slot_indices = inner_index_tensor(&slots, device);
                values = values.select_assign(
                    0,
                    slot_indices,
                    encoded,
                    IndexingUpdateOp::Add,
                );
                let completed = batch + 1;
                if completed == batches || completed.is_multiple_of(64) {
                    eprintln!("encoded DINO device cache batch {completed}/{batches}");
                }
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(values))
        }

        fn select(
            &self,
            indices: &[usize],
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            let mut chunks = Vec::with_capacity(indices.len().div_ceil(self.batch_size));
            for chunk_indices in indices.chunks(self.batch_size) {
                let conditions = chunk_indices
                    .iter()
                    .map(|idx| {
                        let path = self.paths.get(*idx).ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "DINO condition source index out of bounds".to_string(),
                            )
                        })?;
                        load_dino_condition_image(path)
                    })
                    .collect::<AutomataResult<Vec<_>>>()?;
                let encoded = self
                    .encoder
                    .encode_batch_tensor_with_contract(&conditions, self.contract())
                    .map_err(|err| {
                        AutomataError::InvalidArgument(format!(
                            "failed to encode on-demand DINO condition batch: {err}"
                        ))
                    })?;
                chunks.push(encoded);
            }
            let encoded = if chunks.len() == 1 {
                chunks.remove(0)
            } else {
                Tensor::cat(chunks, 0)
            };
            let dims = encoded.dims();
            if dims != [indices.len(), token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "on-demand DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims,
                    indices.len(),
                    token_count,
                    embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
        }

        fn encode_preprocessed(
            &self,
            prepared: &DinoVitsPreparedConditionBatch,
            batch: usize,
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            if batch == 0 {
                return Err(AutomataError::InvalidArgument(
                    "preprocessed DINO condition batch is empty".to_string(),
                ));
            }
            let encoded = self
                .encoder
                .encode_preprocessed_batch_tensor_with_contract(prepared, self.contract())
                .map_err(|err| {
                    AutomataError::InvalidArgument(format!(
                        "failed to encode preprocessed DINO condition batch: {err}"
                    ))
                })?;
            let dims = encoded.dims();
            if dims != [batch, token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "preprocessed DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims, batch, token_count, embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
        }

        fn encode_loaded(
            &self,
            images: &[ConditionImage2d],
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            let mut chunks = Vec::with_capacity(images.len().div_ceil(self.batch_size));
            for conditions in images.chunks(self.batch_size) {
                let encoded = self
                    .encoder
                    .encode_batch_tensor_with_contract(conditions, self.contract())
                    .map_err(|err| {
                        AutomataError::InvalidArgument(format!(
                            "failed to encode preloaded DINO condition batch: {err}"
                        ))
                    })?;
                chunks.push(encoded);
            }
            let encoded = if chunks.len() == 1 {
                chunks.remove(0)
            } else {
                Tensor::cat(chunks, 0)
            };
            let dims = encoded.dims();
            if dims != [images.len(), token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "preloaded DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims,
                    images.len(),
                    token_count,
                    embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
        }
    }

    #[cfg(feature = "dino")]
    fn load_dino_condition_image(path: &Path) -> AutomataResult<ConditionImage2d> {
        crate::load_condition_image(path).map_err(|err| {
            AutomataError::InvalidArgument(format!(
                "failed to load condition image {}: {err}",
                path.display()
            ))
        })
    }

    fn seed_batch_tensors(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        seed_batch_tensors_with_seed_indices(
            targets,
            indices,
            indices,
            particle_count,
            config,
            step_seed,
            device,
        )
    }

    fn seed_batch_tensors_with_seed_indices(
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        seed_indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        debug_assert_eq!(target_indices.len(), seed_indices.len());
        let mut positions = Vec::with_capacity(target_indices.len() * particle_count * 2);
        let mut states = Vec::with_capacity(target_indices.len() * particle_count * 16);
        for (&target_idx, &seed_idx) in target_indices.iter().zip(seed_indices) {
            let (example_positions, example_states) = seed_particles_scaled(
                1,
                particle_count,
                16,
                2,
                step_seed.wrapping_add(seed_idx as u64),
                config.seed_mode,
                targets[target_idx].seed_scale,
            );
            positions.extend(
                example_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]]),
            );
            states.extend(example_states);
        }
        (
            tensor3(positions, [target_indices.len(), particle_count, 2], device),
            tensor3(states, [target_indices.len(), particle_count, 16], device),
        )
    }

    fn host_batch_mask(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rng: &mut StdRng,
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for &idx in indices {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                rng,
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn host_single_mask_stack(
        target: &BurnTargetExample,
        steps: usize,
        rng: &mut StdRng,
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(steps * target.particle_count);
        for _ in 0..steps {
            values.extend(stochastic_mask(
                target.particle_count,
                target.update_prob,
                rng,
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [steps, target.particle_count, 1],
            &target.target_rgb.device(),
        )
    }

    fn host_batch_mask_stack(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
        rng: &mut StdRng,
    ) -> Tensor4 {
        let mut values = Vec::with_capacity(steps * indices.len() * particle_count);
        for _ in 0..steps {
            for &idx in indices {
                values.extend(stochastic_mask(
                    particle_count,
                    targets[idx].update_prob,
                    rng,
                ));
            }
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor4(
            values,
            [steps, indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn device_batch_mask_stack(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
    ) -> Tensor4 {
        let device = &targets[indices[0]].target_rgb.device();
        let shape = [steps, indices.len(), particle_count, 1];
        let samples = Tensor::<BurnBackend, 4>::random(
            shape,
            Distribution::Uniform(0.0, 1.0),
            device,
        );
        STOCHASTIC_MASK_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
        if let Some(update_prob) = homogeneous_update_prob(targets, indices) {
            return samples.lower_elem(update_prob).float();
        }
        let probs = indices
            .iter()
            .map(|idx| targets[*idx].update_prob)
            .collect::<Vec<_>>();
        let probs = tensor4(probs, [1, indices.len(), 1, 1], device).expand(shape);
        samples.lower(probs).float()
    }

    fn batch_update_prob_is_one(targets: &[BurnTargetExample], indices: &[usize]) -> bool {
        indices.iter().all(|&idx| targets[idx].update_prob >= 1.0)
    }

    fn homogeneous_update_prob(targets: &[BurnTargetExample], indices: &[usize]) -> Option<f32> {
        let first = targets[*indices.first()?].update_prob;
        indices
            .iter()
            .all(|&idx| (targets[idx].update_prob - first).abs() <= f32::EPSILON)
            .then_some(first)
    }

    fn host_batch_mask_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rngs: &mut [StdRng],
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for (local, &idx) in indices.iter().enumerate() {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                &mut rngs[local],
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn host_batch_mask_stack_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
        rngs: &mut [StdRng],
    ) -> Tensor4 {
        let mut values = Vec::with_capacity(steps * indices.len() * particle_count);
        for _ in 0..steps {
            for (local, &idx) in indices.iter().enumerate() {
                values.extend(stochastic_mask(
                    particle_count,
                    targets[idx].update_prob,
                    &mut rngs[local],
                ));
            }
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor4(
            values,
            [steps, indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    fn host_batch_mask_seeded(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        seed: u64,
    ) -> Tensor3 {
        let mut rng = StdRng::seed_from_u64(seed);
        host_batch_mask(targets, indices, particle_count, &mut rng)
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

    fn condition_patch_centers_values(width: usize, height: usize) -> Vec<f32> {
        let width = width.max(1);
        let height = height.max(1);
        let mut values = Vec::with_capacity(width * height * 2);
        for y in 0..height {
            let yy = ((y as f32 + 0.5) / height as f32) * 2.0 - 1.0;
            for x in 0..width {
                let xx = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
                values.push(xx);
                values.push(yy);
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
        let batch_size = batch_size.min(examples);
        if batch_size == 0 {
            return Vec::new();
        }
        if batch_size.saturating_mul(4) > examples {
            let mut indices = (0..examples).collect::<Vec<_>>();
            indices.shuffle(rng);
            indices.truncate(batch_size);
            return indices;
        }
        let mut indices = Vec::with_capacity(batch_size);
        while indices.len() < batch_size {
            let idx = rng.random_range(0..examples);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices
    }

    fn sample_rollout_indices(
        sampler: &mut E2eIdentitySampler,
        rollout_replicas: usize,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        sampler
            .next_batch(rng)
            .into_iter()
            .flat_map(|example| std::iter::repeat_n(example, rollout_replicas.max(1)))
            .collect()
    }

    fn e2e_sampling_rng(seed: u64, step: usize) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ 0x51a9_1e5a_d00d_f00d_u64
                ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        )
    }

    fn e2e_pool_rng(seed: u64, step: usize) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ 0x9a7c_e2e0_f00d_51ce_u64
                ^ (step as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        )
    }

    fn per_identity_seed_replacement_rows(
        rollout_identities: &[usize],
        trajectory_counts: &mut [usize],
        interval: usize,
    ) -> Vec<usize> {
        let interval = interval.max(1);
        let mut rows = Vec::new();
        for (row, &identity) in rollout_identities.iter().enumerate() {
            let Some(count) = trajectory_counts.get_mut(identity) else {
                continue;
            };
            *count = count.saturating_add(1);
            if *count >= interval {
                *count %= interval;
                rows.push(row);
            }
        }
        rows
    }

    fn seeded_values(len: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
        let scale = scale.abs().max(f32::MIN_POSITIVE);
        (0..len)
            .map(|_| rng.random_range(-scale..scale))
            .collect::<Vec<_>>()
    }

    fn seeded_zero_delta_output_bias(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        seed: u64,
        output_scale: f32,
    ) -> Vec<f32> {
        let scale = output_scale.abs().max(EPSILON);
        let adapter = NpaLowRankAdapter::seeded_zero_delta(config, rank, alpha, seed);
        let mut values = adapter.to_parameter_vector();
        for value in &mut values {
            let normalized = (*value / scale).clamp(-0.95, 0.95);
            *value = normalized.atanh();
        }
        values
    }

    fn seeded_zero_delta_chunk_output_bias(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        seed: u64,
        chunk_size: usize,
        output_chunks: usize,
        module_layout: Option<&crate::hyper::adapter_layout::AdapterParameterLayout2d>,
    ) -> Vec<f32> {
        let adapter = NpaLowRankAdapter::seeded_zero_delta(config, rank, alpha, seed);
        let mut values = adapter.to_parameter_vector();
        if let Some(layout) = module_layout {
            return layout
                .pack(&values)
                .expect("module adapter layout matches seeded adapter");
        }
        values.resize(output_chunks.saturating_mul(chunk_size), 0.0);
        values
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

    impl BurnDeviceParticlePool {
        fn new(
            pool_size: usize,
            particle_count: usize,
            state_dims: usize,
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
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
            let position_values = positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect::<Vec<_>>();
            let inner_device = Device::<InnerBackend>::from(device.clone());
            Self {
                positions: Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(position_values, [pool_size, particle_count, 2]),
                    &inner_device,
                ),
                states: Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(states, [pool_size, particle_count, state_dims]),
                    &inner_device,
                ),
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
        ) -> AutomataResult<BurnPoolBatch> {
            let mut pool_indices = (0..self.pool_size).collect::<Vec<_>>();
            pool_indices.shuffle(rng);
            pool_indices.truncate(batch_size.min(self.pool_size));
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(&pool_indices, &inner_device);
            let mut x = Tensor::<BurnBackend, 3>::from_inner(
                self.positions.clone().select(0, indices.clone()),
            );
            let mut s =
                Tensor::<BurnBackend, 3>::from_inner(self.states.clone().select(0, indices));

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
                let position_values = seed_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let replacement = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(vec![0_i64], [1]),
                    device,
                );
                let new_positions = tensor3(
                    position_values,
                    [1, self.particle_count, 2],
                    device,
                );
                let position_delta = new_positions - x.clone().select(0, replacement.clone());
                x = x.select_assign(
                    0,
                    replacement.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = tensor3(
                    seed_states,
                    [1, self.particle_count, self.state_dims],
                    device,
                );
                let state_delta = new_states - s.clone().select(0, replacement.clone());
                s = s.select_assign(0, replacement, state_delta, IndexingUpdateOp::Add);
            }

            if config.brush_size > 0.0 && !pool_indices.is_empty() {
                let center_indices = (0..pool_indices.len())
                    .map(|batch| {
                        (batch * self.particle_count + rng.random_range(0..self.particle_count))
                            as i64
                    })
                    .collect::<Vec<_>>();
                let center_indices = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(center_indices, [pool_indices.len()]),
                    device,
                );
                let centers = x
                    .clone()
                    .reshape([pool_indices.len() * self.particle_count, 2])
                    .select(0, center_indices)
                    .reshape([pool_indices.len(), 1, 2])
                    .expand([pool_indices.len(), self.particle_count, 2]);
                let diff = x.clone() - centers;
                let damaged = diff
                    .clone()
                    .mul(diff)
                    .sum_dim(2)
                    .lower_elem(config.brush_size * config.brush_size)
                    .expand([pool_indices.len(), self.particle_count, self.state_dims]);
                s = s.mask_fill(damaged, 0.0);
            }

            Ok(BurnPoolBatch {
                pool_indices,
                x,
                s,
            })
        }

        fn update_batch(
            &mut self,
            pool_indices: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            if pool_indices.is_empty() {
                return Ok(());
            }
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(pool_indices, &inner_device);
            let position_delta = x.inner() - self.positions.clone().select(0, indices.clone());
            self.positions = self.positions.clone().select_assign(
                0,
                indices.clone(),
                position_delta,
                IndexingUpdateOp::Add,
            );
            let state_delta = s.inner() - self.states.clone().select(0, indices.clone());
            self.states = self.states.clone().select_assign(
                0,
                indices,
                state_delta,
                IndexingUpdateOp::Add,
            );
            Ok(())
        }
    }

    impl BurnE2eDeviceParticlePool {
        fn new(
            capacity: usize,
            particle_count: usize,
            state_dims: usize,
            slots_per_example: usize,
            device: &BurnDevice,
        ) -> Self {
            let inner_device = Device::<InnerBackend>::from(device.clone());
            Self {
                positions: Tensor::<InnerBackend, 3>::zeros(
                    [capacity, particle_count, 2],
                    &inner_device,
                ),
                states: Tensor::<InnerBackend, 3>::zeros(
                    [capacity, particle_count, state_dims],
                    &inner_device,
                ),
                slot_examples: vec![None; capacity],
                example_slots: HashMap::with_capacity(capacity),
                next_evict: 0,
                capacity,
                particle_count,
                state_dims,
                slots_per_example: slots_per_example.max(1),
            }
        }

        fn sample_batch(
            &mut self,
            example_indices: &[usize],
            rng: &mut StdRng,
            seed_replacement_rows: &[usize],
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<BurnE2ePoolBatch> {
            if example_indices.len() > self.capacity {
                return Err(AutomataError::InvalidArgument(format!(
                    "device particle pool capacity {} is smaller than batch {}",
                    self.capacity,
                    example_indices.len()
                )));
            }
            let mut slots = Vec::with_capacity(example_indices.len());
            let mut new_slots = Vec::new();
            let mut replica_choices = HashMap::<usize, Vec<usize>>::new();
            for &example in example_indices {
                let choices = replica_choices.entry(example).or_insert_with(|| {
                    let mut choices = (0..self.slots_per_example).collect::<Vec<_>>();
                    choices.shuffle(rng);
                    choices
                });
                let replica = choices.pop().ok_or_else(|| {
                    AutomataError::InvalidArgument(format!(
                        "requested more than {} rollout replicas for example {example}",
                        self.slots_per_example
                    ))
                })?;
                let key = (example, replica);
                if let Some(&slot) = self.example_slots.get(&key) {
                    slots.push(slot);
                    continue;
                }
                let slot = self.allocate_slot(&slots);
                if let Some(previous) = self.slot_examples[slot].replace(key) {
                    self.example_slots.remove(&previous);
                }
                self.example_slots.insert(key, slot);
                slots.push(slot);
                new_slots.push(slot);
            }
            if !new_slots.is_empty() {
                let seed = config.seed ^ rng.random::<u64>();
                let (positions, states) = seed_particles_scaled(
                    new_slots.len(),
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                let position_values = positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let inner_device = self.positions.device();
                let indices = inner_index_tensor(&new_slots, &inner_device);
                let new_positions = Tensor::<InnerBackend, 3>::from_data(
                        TensorData::new(
                            position_values,
                            [new_slots.len(), self.particle_count, 2],
                        ),
                        &inner_device,
                    );
                let position_delta = new_positions - self.positions.clone().select(0, indices.clone());
                self.positions = self.positions.clone().select_assign(
                    0,
                    indices.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = Tensor::<InnerBackend, 3>::from_data(
                        TensorData::new(
                            states,
                            [new_slots.len(), self.particle_count, self.state_dims],
                        ),
                        &inner_device,
                    );
                let state_delta = new_states - self.states.clone().select(0, indices.clone());
                self.states = self.states.clone().select_assign(
                    0,
                    indices,
                    state_delta,
                    IndexingUpdateOp::Add,
                );
            }

            let inner_device = self.positions.device();
            let slot_indices = inner_index_tensor(&slots, &inner_device);
            let mut x = Tensor::<BurnBackend, 3>::from_inner(
                self.positions.clone().select(0, slot_indices.clone()),
            );
            let mut s = Tensor::<BurnBackend, 3>::from_inner(
                self.states.clone().select(0, slot_indices),
            );
            let mut replacement_rows = seed_replacement_rows
                .iter()
                .copied()
                .filter(|row| *row < slots.len())
                .collect::<Vec<_>>();
            replacement_rows.sort_unstable();
            replacement_rows.dedup();
            let seed_replacements = replacement_rows.len();
            if seed_replacements > 0 {
                let seed = config.seed ^ rng.random::<u64>();
                let (positions, states) = seed_particles_scaled(
                    seed_replacements,
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                let position_values = positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let replacement = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(
                        replacement_rows
                            .iter()
                            .map(|row| *row as i64)
                            .collect::<Vec<_>>(),
                        [seed_replacements],
                    ),
                    device,
                );
                let new_positions = tensor3(
                    position_values,
                    [seed_replacements, self.particle_count, 2],
                    device,
                );
                let position_delta = new_positions - x.clone().select(0, replacement.clone());
                x = x.select_assign(
                    0,
                    replacement.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = tensor3(
                    states,
                    [seed_replacements, self.particle_count, self.state_dims],
                    device,
                );
                let state_delta = new_states - s.clone().select(0, replacement.clone());
                s = s.select_assign(
                    0,
                    replacement,
                    state_delta,
                    IndexingUpdateOp::Add,
                );
            }
            if config.brush_size > 0.0 && !slots.is_empty() {
                let center_particles = (0..slots.len())
                    .map(|_| rng.random_range(0..self.particle_count))
                    .collect::<Vec<_>>();
                let centers = gather_live_particle_centers(x.clone(), &center_particles, device)
                    .expand([slots.len(), self.particle_count, 2]);
                let diff = x.clone() - centers;
                let damaged = diff
                    .clone()
                    .mul(diff)
                    .sum_dim(2)
                    .lower_elem(config.brush_size * config.brush_size)
                    .expand([slots.len(), self.particle_count, self.state_dims]);
                s = s.mask_fill(damaged, 0.0);
            }
            Ok(BurnE2ePoolBatch {
                slots,
                x,
                s,
                seed_replacements,
            })
        }

        fn update_batch(
            &mut self,
            slots: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            if slots.is_empty() {
                return Ok(());
            }
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(slots, &inner_device);
            let persisted_x = x.inner().clamp(-1.0, 1.0);
            let position_delta =
                persisted_x - self.positions.clone().select(0, indices.clone());
            self.positions = self.positions.clone().select_assign(
                0,
                indices.clone(),
                position_delta,
                IndexingUpdateOp::Add,
            );
            // Upstream persists mature recurrent state without amplitude clipping. Keep a
            // generous finite safety bound, but do not erase valid attractor state at +/-1.
            let persisted_s = s.inner().clamp(-32.0, 32.0);
            let state_delta = persisted_s - self.states.clone().select(0, indices.clone());
            self.states = self.states.clone().select_assign(
                0,
                indices,
                state_delta,
                IndexingUpdateOp::Add,
            );
            Ok(())
        }

        fn snapshot(&self) -> AutomataResult<E2eParticlePoolSnapshot> {
            Ok(E2eParticlePoolSnapshot {
                positions: tensor3_snapshot("pool.positions", self.positions.clone())?,
                states: tensor3_snapshot("pool.states", self.states.clone())?,
                slot_examples: self.slot_examples.clone(),
                next_evict: self.next_evict,
                slots_per_example: self.slots_per_example,
            })
        }

        fn restore(
            snapshot: &E2eParticlePoolSnapshot,
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let position_shape: [usize; 3] = snapshot
                .positions
                .shape
                .clone()
                .try_into()
                .map_err(|_| {
                    AutomataError::InvalidArgument(
                        "checkpoint pool positions are not rank three".to_string(),
                    )
                })?;
            let state_shape: [usize; 3] = snapshot.states.shape.clone().try_into().map_err(|_| {
                AutomataError::InvalidArgument(
                    "checkpoint pool states are not rank three".to_string(),
                )
            })?;
            if position_shape[0] != state_shape[0]
                || position_shape[1] != state_shape[1]
                || position_shape[2] != 2
                || state_shape[2] != 16
                || position_shape[0] != snapshot.slot_examples.len()
                || position_shape[0] != config.pool_capacity
                || position_shape[1] != config.rollout_particles
                || snapshot.slots_per_example != config.pool_slots_per_example
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "checkpoint particle pool shape {:?}/{:?} is incompatible with capacity={} particles={} slots_per_example={}",
                    position_shape,
                    state_shape,
                    config.pool_capacity,
                    config.rollout_particles,
                    config.pool_slots_per_example,
                )));
            }
            let mut example_slots = HashMap::with_capacity(snapshot.slot_examples.len());
            for (slot, key) in snapshot.slot_examples.iter().enumerate() {
                if let Some(key) = key {
                    example_slots.insert(*key, slot);
                }
            }
            Ok(Self {
                positions: tensor3_from_snapshot(&snapshot.positions, device)?,
                states: tensor3_from_snapshot(&snapshot.states, device)?,
                slot_examples: snapshot.slot_examples.clone(),
                example_slots,
                next_evict: snapshot.next_evict % config.pool_capacity.max(1),
                capacity: config.pool_capacity,
                particle_count: config.rollout_particles,
                state_dims: 16,
                slots_per_example: config.pool_slots_per_example,
            })
        }

        fn allocate_slot(&mut self, protected: &[usize]) -> usize {
            if let Some(slot) = self.slot_examples.iter().position(Option::is_none) {
                return slot;
            }
            for _ in 0..self.capacity {
                let slot = self.next_evict;
                self.next_evict = (self.next_evict + 1) % self.capacity;
                if !protected.contains(&slot) {
                    return slot;
                }
            }
            unreachable!("pool capacity is validated against batch size")
        }
    }

    fn gather_live_particle_centers(
        positions: Tensor3,
        particle_indices: &[usize],
        device: &BurnDevice,
    ) -> Tensor3 {
        let indices = particle_indices
            .iter()
            .flat_map(|index| [*index as i64, *index as i64])
            .collect::<Vec<_>>();
        let indices = Tensor::<BurnBackend, 3, Int>::from_data(
            TensorData::new(indices, [particle_indices.len(), 1, 2]),
            device,
        );
        positions.gather(1, indices)
    }

    fn inner_index_tensor(indices: &[usize], device: &Device<InnerBackend>) -> Tensor1IntInner {
        Tensor::from_data(
            TensorData::new(
                indices.iter().map(|index| *index as i64).collect::<Vec<_>>(),
                [indices.len()],
            ),
            device,
        )
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

    fn normalize_sample_id_table_gradient(
        gradient: Tensor2Inner,
        segments: &[(usize, usize)],
    ) -> Tensor2Inner {
        let dims = gradient.shape().dims::<2>();
        debug_assert_eq!(
            segments.iter().map(|(_, len)| *len).sum::<usize>(),
            dims[0]
        );
        Tensor::cat(
            segments
                .iter()
                .map(|&(offset, len)| {
                    let segment = gradient.clone().narrow(0, offset, len);
                    let per_identity_norm = segment
                        .clone()
                        .mul(segment.clone())
                        .sum_dim(0)
                        .sqrt()
                        .add_scalar(1.0e-8);
                    segment.div(per_identity_norm.expand([len, dims[1]]))
                })
                .collect(),
            0,
        )
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

    fn tensor4(values: Vec<f32>, shape: [usize; 4], device: &BurnDevice) -> Tensor4 {
        Tensor::<BurnBackend, 4>::from_data(TensorData::new(values, shape), device)
    }

    fn tensor1(values: Vec<f32>, shape: [usize; 1], device: &BurnDevice) -> Tensor1 {
        Tensor::<BurnBackend, 1>::from_data(TensorData::new(values, shape), device)
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

    fn sync_training_device(device: &BurnDevice) -> Result<(), Box<dyn std::error::Error>> {
        let inner_device: Device<InnerBackend> = device.clone();
        <InnerBackend as Backend>::sync(&inner_device)?;
        Ok(())
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

    fn tensor2_snapshot(name: &str, tensor: Tensor2Inner) -> AutomataResult<E2eTensorSnapshot> {
        let shape = tensor.shape().dims::<2>();
        Ok(E2eTensorSnapshot {
            name: name.to_string(),
            shape: shape.to_vec(),
            values: tensor_vec(tensor)?,
        })
    }

    fn tensor2_from_snapshot(
        snapshot: &E2eTensorSnapshot,
        device: &BurnDevice,
    ) -> AutomataResult<Tensor2Inner> {
        let shape: [usize; 2] = snapshot.shape.clone().try_into().map_err(|_| {
            AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} is not rank two",
                snapshot.name
            ))
        })?;
        if shape[0].saturating_mul(shape[1]) != snapshot.values.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} shape {:?} does not match {} values",
                snapshot.name,
                shape,
                snapshot.values.len()
            )));
        }
        let inner_device: Device<InnerBackend> = device.clone();
        Ok(Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(snapshot.values.clone(), shape),
            &inner_device,
        ))
    }

    fn tensor3_vec(tensor: Tensor3Inner) -> AutomataResult<Vec<f32>> {
        tensor.into_data().to_vec::<f32>().map_err(|err| {
            AutomataError::InvalidArgument(format!("Burn dense tensor readback failed: {err}"))
        })
    }

    fn tensor3_snapshot(name: &str, tensor: Tensor3Inner) -> AutomataResult<E2eTensorSnapshot> {
        let shape = tensor.shape().dims::<3>();
        Ok(E2eTensorSnapshot {
            name: name.to_string(),
            shape: shape.to_vec(),
            values: tensor3_vec(tensor)?,
        })
    }

    fn tensor3_from_snapshot(
        snapshot: &E2eTensorSnapshot,
        device: &BurnDevice,
    ) -> AutomataResult<Tensor3Inner> {
        let shape: [usize; 3] = snapshot.shape.clone().try_into().map_err(|_| {
            AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} is not rank three",
                snapshot.name
            ))
        })?;
        if shape.iter().product::<usize>() != snapshot.values.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "checkpoint tensor {} shape {:?} does not match {} values",
                snapshot.name,
                shape,
                snapshot.values.len()
            )));
        }
        let inner_device: Device<InnerBackend> = device.clone();
        Ok(Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(snapshot.values.clone(), shape),
            &inner_device,
        ))
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
        fn sample_id_table_gradients_normalize_each_parameter_and_identity() {
            let device = BurnDevice::default();
            let inner_device: Device<InnerBackend> = device;
            let gradient = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(vec![3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 8.0], [4, 2]),
                &inner_device,
            );
            let normalized = normalize_sample_id_table_gradient(gradient, &[(0, 2), (2, 2)]);
            let values = normalized.into_data().to_vec::<f32>().unwrap();
            let expected = [0.6, 0.0, 0.8, 1.0, 0.0, 0.6, 0.0, 0.8];
            assert!(max_abs_difference(&values, &expected) <= 1.0e-5);
        }

        #[test]
        fn module_token_v3_burn_forward_matches_serialized_inference() {
            let device = BurnDevice::default();
            let config = NpaConfig::growing_2d();
            let rank = 2;
            let chunk_size = 16;
            let hidden_dims = 4;
            let attention_heads = 2;
            let embed_dims = 3;
            let token_count = 3;
            let layout = crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                &config,
                rank,
                chunk_size,
            )
            .unwrap();
            let token_w = vec![
                0.7, 0.1, 0.0, 0.0, 0.8, 0.2, 0.1, 0.0, 0.9, 0.3, 0.4, 0.5,
            ];
            let token_b = vec![0.01, 0.02, 0.03, 0.04];
            let token_gate_w = layout.structured_query_initialization(hidden_dims, 0.2);
            let token_gate_b = layout.structured_query_initialization(hidden_dims, 0.1);
            let state_w = vec![0.0; hidden_dims * chunk_size];
            let time_w = vec![0.0; hidden_dims];
            let output_w = (0..chunk_size * hidden_dims)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.003)
                .collect::<Vec<_>>();
            let output_b = vec![0.0; layout.padded_parameter_count()];
            let condition = vec![0.9, 0.1, 0.2, 0.1, 0.8, 0.3, 0.2, 0.3, 0.9];
            let output_dims = layout.parameter_count;

            let generator = BurnE2eGeneratorParams {
                kind: E2eHyperGeneratorKind::ModuleTokenDecoder,
                token_w: tracked_tensor(
                    token_w.clone(),
                    [hidden_dims, embed_dims],
                    &device,
                ),
                token_b: tracked_tensor(token_b.clone(), [1, hidden_dims], &device),
                token_gate_w: tracked_tensor(
                    token_gate_w.clone(),
                    [layout.chunk_count, hidden_dims],
                    &device,
                ),
                token_gate_b: tracked_tensor(
                    token_gate_b.clone(),
                    [layout.chunk_count, hidden_dims],
                    &device,
                ),
                state_w: tracked_tensor(
                    state_w.clone(),
                    [hidden_dims, chunk_size],
                    &device,
                ),
                time_w: tracked_tensor(time_w.clone(), [hidden_dims, 1], &device),
                output_w: tracked_tensor(
                    output_w.clone(),
                    [chunk_size, hidden_dims],
                    &device,
                ),
                output_b: tracked_tensor(
                    output_b.clone(),
                    [layout.chunk_count, chunk_size],
                    &device,
                ),
                condition_control_w: tracked_tensor(
                    vec![0.0; config.update_dims() * hidden_dims],
                    [config.update_dims(), hidden_dims],
                    &device,
                ),
                condition_control_b: tracked_tensor(
                    vec![0.0; config.update_dims()],
                    [1, config.update_dims()],
                    &device,
                ),
                condition_control_state_w: tracked_tensor(
                    vec![0.0; hidden_dims * config.state_dims],
                    [hidden_dims, config.state_dims],
                    &device,
                ),
                hidden_dims,
                token_attention_heads: attention_heads,
                softmax_token_attention: true,
                canonical_full_rank_lora: false,
                adapter_constants: tracked_tensor(vec![0.0; output_dims], [1, output_dims], &device),
                adapter_trainable_mask: tracked_tensor(
                    vec![1.0; output_dims],
                    [1, output_dims],
                    &device,
                ),
                adapter_parameter_segments: BurnE2eGeneratorParams::adapter_parameter_segments(
                    &config, rank,
                ),
                output_dims,
                output_scale: 1.0,
                sample_steps: 1,
                adapter_chunk_size: chunk_size,
                output_chunks: layout.chunk_count,
            };
            let burn_vector = tensor_vec(
                generator
                    .spatial_token_adapter_vector_batch(
                        tensor3(condition.clone(), [1, token_count, embed_dims], &device),
                        &config,
                        rank,
                    )
                    .inner(),
            )
            .unwrap();
            let hyper = E2eHyperNpa2d {
                version: 1,
                architecture: E2eHyperGeneratorKind::ModuleTokenDecoder
                    .artifact_architecture()
                    .to_string(),
                backend: Some("test".to_string()),
                condition_encoder: Some("dino-vits-full-tokens".to_string()),
                condition_token_count: Some(token_count),
                condition_embed_dims: Some(embed_dims),
                condition_token_grid_width: None,
                condition_token_grid_height: None,
                condition_image_size: Some(224),
                condition_alpha_mode: Some("composite-white".to_string()),
                condition_rgb_channels: Some(false),
                condition_rgb_channel_scale: Some(1.0),
                condition_alpha_channel: Some(false),
                condition_alpha_channel_scale: Some(1.0),
                condition_l2_normalize_features: Some(false),
                condition_resize_mode: Some("stretch".to_string()),
                condition_application: Some("static-adapter".to_string()),
                shared_base_sha256: None,
                hidden_dims,
                token_attention_heads: attention_heads,
                attention_normalization: Some(
                    crate::hyper::e2e::E2E_HYPER_ATTENTION_SOFTMAX.to_string(),
                ),
                output_dims,
                sample_steps: 1,
                output_scale: 1.0,
                adapter_rank: Some(rank),
                adapter_alpha: Some(rank as f32),
                adapter_parameterization: Some(E2E_HYPER_ADAPTER_FACTORIZED.to_string()),
                adapter_chunk_size: Some(chunk_size),
                spatial_condition_control: None,
                spatial_condition_control_scale: None,
                spatial_condition_control_sigma: None,
                spatial_condition_state_control: None,
                weights: E2eHyperNpa2dWeights {
                    token_w,
                    token_b,
                    token_gate_w,
                    token_gate_b,
                    state_w,
                    time_w,
                    output_w,
                    output_b,
                    condition_control_w: Vec::new(),
                    condition_control_b: Vec::new(),
                    condition_control_state_w: Vec::new(),
                },
            };
            let inference_vector = hyper
                .predict_adapter(&config, &condition)
                .unwrap()
                .to_parameter_vector();

            assert!(
                max_abs_difference(&burn_vector, &inference_vector) < 2.0e-5,
                "Burn and serialized inference module-token v3 forwards diverged"
            );
        }

        fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
            assert_eq!(left.len(), right.len());
            left.iter()
                .zip(right)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max)
        }

        fn max_abs_difference_with_index(left: &[f32], right: &[f32]) -> (usize, f32, f32, f32) {
            assert_eq!(left.len(), right.len());
            left.iter()
                .zip(right)
                .enumerate()
                .map(|(idx, (left, right))| (idx, (left - right).abs(), *left, *right))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0, 0.0, 0.0))
        }

        fn test_burn_target(device: &BurnDevice, update_prob: f32, seed_scale: f32) -> BurnTargetExample {
            let target_cpu = crate::TargetImage2d {
                source_width: 1,
                source_height: 1,
                positions: vec![[0.0, 0.0]],
                colors: vec![[1.0, 1.0, 1.0]],
                pixel_size: 2.0,
                threshold: 0.05,
                aabb: [-1.0, 1.0, -1.0, 1.0],
            };
            BurnTargetExample {
                target_rgb: Tensor::<BurnBackend, 2>::zeros([1, 3], device),
                target_density: Tensor::<BurnBackend, 2>::zeros([1, 1], device),
                target_foreground: Tensor::<BurnBackend, 2>::zeros([1, 1], device),
                target_foreground_scale: 1.0,
                target_mean: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
                target_positions: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
                pixel_xy: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
                pixel_size: 2.0,
                target_points: 1,
                particle_count: 4,
                update_prob,
                seed_scale,
                target_cpu,
            }
        }

        fn test_direct_config(particle_count: usize) -> DirectBasisTrainConfig {
            let npa_config = NpaConfig::growing_2d();
            let grid = burn_automata_kernels::HashGridConfig::growing_2d();
            DirectBasisTrainConfig {
                steps: 0,
                report_interval: 1,
                example_batch_size: 2,
                tbptt_chunk_steps: 1,
                loss_on_final_chunk_only: false,
                use_particle_pool: false,
                pool_size: 0,
                inject_seed_interval: 0,
                brush_size: 0.0,
                stopgrad_pos: npa_config.stopgrad_pos,
                stopgrad_state: npa_config.stopgrad_state,
                rollout_particles: particle_count,
                rollout_step_min: 1,
                rollout_steps: 1,
                update_prob: 1.0,
                seed: 13,
                seed_scale: 0.1,
                seed_mode: crate::ParticleSeed::UniformCircle,
                grid_eps: grid.eps,
                motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
                loss_config: crate::Target2dLossConfig::default(),
                target2d_loss_backend: Target2dLossBackend::Dense,
                perception_backend: PerceptionRolloutBackend::Dense,
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
                eval_seed: 13,
                system_memory_budget_gb: None,
                gpu_memory_budget_gb: None,
                max_dense_train_particles: particle_count,
                max_dense_chunk_floats: 1_000_000,
                max_splat_chunk_floats: 1_000_000,
            }
        }

        #[test]
        fn functional_teacher_probe_loss_distinguishes_adapter_behavior() {
            let device = BurnDevice::default();
            let config = NpaConfig::growing_2d();
            let base = NpaModel::upstream_seeded(config.clone(), 7);
            let params = BurnBaseParams::from_model(&base, &device).unwrap();
            let rank = 4;
            let mut adapter = NpaLowRankAdapter::seeded(&config, rank, rank as f32, 19);
            adapter.b2_delta.fill(0.25);
            let adapter_vector = adapter.to_parameter_vector();
            let adapter_batch = BurnAdapterBatch::from_parameter_vector(
                tensor(adapter_vector, [1, adapter.parameter_count()], &device),
                &config,
                rank,
                rank as f32,
            );
            let zero_batch = BurnAdapterBatch::from_parameter_vector(
                Tensor::<BurnBackend, 2>::zeros([1, adapter.parameter_count()], &device),
                &config,
                rank,
                rank as f32,
            );
            let feature_dims = config.perception_dims();
            let features = tensor3(
                (0..3 * feature_dims)
                    .map(|idx| (idx as f32 * 0.071).sin())
                    .collect(),
                [1, 3, feature_dims],
                &device,
            );
            let teacher = params.forward_adapter_batch(features.clone(), &adapter_batch);
            let matching = params.forward_adapter_batch(features.clone(), &adapter_batch);
            let mismatched = params.forward_adapter_batch(features, &zero_batch);
            let matching_delta = matching - teacher.clone();
            let mismatched_delta = mismatched - teacher;
            let matching_mse = matching_delta
                .clone()
                .mul(matching_delta)
                .mean()
                .inner()
                .into_scalar();
            let mismatched_mse = mismatched_delta
                .clone()
                .mul(mismatched_delta)
                .mean()
                .inner()
                .into_scalar();

            assert!(matching_mse <= 1.0e-12);
            assert!(mismatched_mse > 1.0e-8);
        }

        #[test]
        fn spatial_condition_state_projection_is_zero_warm_start_compatible() {
            let device = BurnDevice::default();
            let x = tensor3(vec![0.0, 0.0], [1, 1, 2], &device);
            let state = tensor3(vec![1.0, 0.0], [1, 1, 2], &device);
            let base = BurnE2eConditionControlBatch {
                patch_hidden: tensor3(vec![1.0, 0.0], [1, 1, 2], &device),
                update_w: tensor(vec![1.0, 0.0, 0.0, 1.0], [2, 2], &device),
                update_b: tensor(vec![0.0, 0.0], [1, 2], &device),
                state_w: None,
                grid_width: 1,
                grid_height: 1,
                sigma: 0.25,
                scale: 1.0,
            };
            let baseline = tensor3_vec(base.update_for_particles(&x, &state).inner()).unwrap();
            let zero_state = BurnE2eConditionControlBatch {
                state_w: Some(tensor(vec![0.0; 4], [2, 2], &device)),
                ..base.clone()
            };
            let zero_state_output =
                tensor3_vec(zero_state.update_for_particles(&x, &state).inner()).unwrap();
            assert_eq!(baseline, zero_state_output);

            let state_aware = BurnE2eConditionControlBatch {
                state_w: Some(tensor(vec![1.0, 0.0, 0.0, 1.0], [2, 2], &device)),
                ..base
            };
            let state_aware_output =
                tensor3_vec(state_aware.update_for_particles(&x, &state).inner()).unwrap();
            assert!(state_aware_output[0] > baseline[0]);
        }

        #[test]
        fn perception_auto_uses_fused_path_for_training_scale_particles() {
            let mut small = test_direct_config(127);
            small.perception_backend = PerceptionRolloutBackend::Auto;
            assert_eq!(perception_backend_effective(small), PerceptionRolloutBackend::Dense);

            let mut training_scale = test_direct_config(128);
            training_scale.perception_backend = PerceptionRolloutBackend::Auto;
            #[cfg($perception_cube_feature)]
            assert_eq!(
                perception_backend_effective(training_scale),
                PerceptionRolloutBackend::TiledAdjoint
            );
            #[cfg(not($perception_cube_feature))]
            assert_eq!(
                perception_backend_effective(training_scale),
                PerceptionRolloutBackend::Dense
            );
        }

        #[test]
        fn target2d_auto_uses_device_adjoint_for_training_scale_particles() {
            let mut small = test_direct_config(127);
            small.target2d_loss_backend = Target2dLossBackend::Auto;
            assert_eq!(target2d_loss_backend_effective(small), Target2dLossBackend::Dense);

            let mut training_scale = test_direct_config(128);
            training_scale.target2d_loss_backend = Target2dLossBackend::Auto;
            #[cfg($perception_cube_feature)]
            assert_eq!(
                target2d_loss_backend_effective(training_scale),
                Target2dLossBackend::TiledAdjoint
            );
            #[cfg(not($perception_cube_feature))]
            assert_eq!(
                target2d_loss_backend_effective(training_scale),
                Target2dLossBackend::Dense
            );
        }

        #[test]
        fn stochastic_rollout_step_sampler_matches_upstream_exclusive_maximum() {
            let mut config = test_direct_config(4);
            config.rollout_step_min = 2;
            config.rollout_steps = 4;
            let samples = (0..512)
                .map(|seed| sampled_training_rollout_steps(config, seed))
                .collect::<Vec<_>>();
            assert!(
                samples.iter().all(|steps| (2..4).contains(steps)),
                "sampled rollout steps escaped configured upstream range: {samples:?}"
            );
            assert!(
                samples.contains(&2) && samples.contains(&3) && !samples.contains(&4),
                "sampled rollout steps did not match the upstream-exclusive maximum: {samples:?}"
            );
        }

        #[test]
        fn brush_centers_are_gathered_from_live_particles_per_batch_row() {
            let device = BurnDevice::default();
            let positions = tensor3(
                vec![
                    -0.9, -0.8, -0.4, -0.3, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0,
                ],
                [2, 3, 2],
                &device,
            );
            let centers = gather_live_particle_centers(positions, &[2, 0], &device);
            assert_eq!(tensor3_vec(centers.inner()).unwrap(), [0.1, 0.2, 0.3, 0.4]);
        }

        #[test]
        fn e2e_generator_zero_delta_lora_init_keeps_adapter_trainable() {
            let config = NpaConfig::growing_2d();
            let rank = 4;
            let output_scale = 1.0;
            let bias = seeded_zero_delta_output_bias(&config, rank, rank as f32, 123, output_scale);
            let vector = bias
                .iter()
                .map(|value| value.tanh() * output_scale)
                .collect::<Vec<_>>();
            let adapter =
                NpaLowRankAdapter::from_parameter_vector(&config, rank, rank as f32, vector)
                    .unwrap();
            assert!(
                adapter
                    .w1_down
                    .iter()
                    .chain(adapter.w2_down.iter())
                    .any(|value| value.abs() > 1.0e-6),
                "zero-delta LoRA init must seed one side of each low-rank product"
            );
            assert!(
                adapter
                    .w1_up
                    .iter()
                    .chain(adapter.w2_up.iter())
                    .chain(adapter.b1_delta.iter())
                    .chain(adapter.b2_delta.iter())
                    .all(|value| value.abs() <= 1.0e-6),
                "zero-delta LoRA init must not perturb the base model initially"
            );
        }

        #[test]
        fn canonical_lora_batched_device_transform_matches_shared_cpu_layout() {
            let device = BurnDevice::default();
            let config = NpaConfig::growing_2d();
            let rank = config.perception_dims().max(config.update_dims());
            let canonical = crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(
                &config,
                rank,
                rank as f32,
            )
            .unwrap();
            let output_dims = canonical.constants.len();
            let first = (0..output_dims)
                .map(|index| (index as f32 * 0.013).sin() * 0.01)
                .collect::<Vec<_>>();
            let second = (0..output_dims)
                .map(|index| (index as f32 * 0.017).cos() * 0.02)
                .collect::<Vec<_>>();
            let expected = [canonical.apply(&first).unwrap(), canonical.apply(&second).unwrap()]
                .concat();
            let generated = tensor([first, second].concat(), [2, output_dims], &device);
            let mask = tensor(
                canonical.trainable_mask.clone(),
                [1, output_dims],
                &device,
            )
            .expand([2, output_dims]);
            let constants = tensor(canonical.constants, [1, output_dims], &device)
                .expand([2, output_dims]);
            let actual = tensor_vec((generated.mul(mask) + constants).inner()).unwrap();

            assert!(max_abs_difference(&actual, &expected) < 1.0e-7);
            assert_ne!(&actual[..output_dims], &actual[output_dims..]);
        }

        #[test]
        fn hyper_e2e_batch_dimension_is_sample_parallel() {
            let device = BurnDevice::default();
            let targets = vec![
                test_burn_target(&device, 1.0, 0.1),
                test_burn_target(&device, 0.0, 0.2),
            ];
            let indices = [0usize, 1usize];
            let particle_count = 4;
            let (x, s) = seed_batch_tensors(
                &targets,
                &indices,
                particle_count,
                test_direct_config(particle_count),
                77,
                &device,
            );
            assert_eq!(x.shape().dims::<3>(), [2, particle_count, 2]);
            assert_eq!(s.shape().dims::<3>(), [2, particle_count, 16]);
            let x_values = tensor3_vec(x.inner()).unwrap();
            assert_ne!(
                &x_values[0..particle_count * 2],
                &x_values[particle_count * 2..particle_count * 4],
                "seeded rollout batch collapsed two independent samples into one state"
            );

            let mask = host_batch_mask_seeded(&targets, &indices, particle_count, 123);
            assert_eq!(mask.shape().dims::<3>(), [2, particle_count, 1]);
            let mask_values = tensor3_vec(mask.inner()).unwrap();
            assert!(mask_values[0..particle_count].iter().all(|value| *value == 1.0));
            assert!(mask_values[particle_count..particle_count * 2]
                .iter()
                .all(|value| *value == 0.0));
            let device_mask = device_batch_mask_stack(&targets, &indices, particle_count, 2);
            assert_eq!(device_mask.shape().dims::<4>(), [2, 2, particle_count, 1]);
            let device_mask_values = tensor3_vec(
                device_mask
                    .reshape([2 * indices.len(), particle_count, 1])
                    .inner(),
            )
            .unwrap();
            assert!(
                device_mask_values[0..particle_count]
                    .iter()
                    .all(|value| *value == 1.0),
                "update_prob=1.0 should keep all device-mask entries active"
            );
            assert!(
                device_mask_values[particle_count..particle_count * 2]
                    .iter()
                    .all(|value| *value == 0.0),
                "update_prob=0.0 should keep all device-mask entries inactive"
            );

            let npa_config = NpaConfig::growing_2d();
            let rank = 2;
            let parameter_count = NpaLowRankAdapter::parameter_count_for_config(&npa_config, rank);
            let mut vector = Vec::with_capacity(parameter_count * 2);
            vector.extend(std::iter::repeat_n(0.0, parameter_count));
            vector.extend(std::iter::repeat_n(1.0, parameter_count));
            let adapter_batch = BurnAdapterBatch::from_parameter_vector(
                tensor(vector, [2, parameter_count], &device),
                &npa_config,
                rank,
                1.0,
            );
            assert_eq!(adapter_batch.w1_down.shape().dims::<3>()[0], 2);
            let w1_down = tensor3_vec(adapter_batch.w1_down.inner()).unwrap();
            let row_len = rank * npa_config.perception_dims();
            assert!(w1_down[0..row_len].iter().all(|value| *value == 0.0));
            assert!(w1_down[row_len..row_len * 2].iter().all(|value| *value == 1.0));

            let model = NpaModel::seeded(npa_config.clone(), 91);
            let params = BurnBaseParams::from_model(&model, &device).unwrap();
            let host_adapters = [
                NpaLowRankAdapter::seeded(&npa_config, rank, 1.0, 101),
                NpaLowRankAdapter::seeded(&npa_config, rank, 1.0, 202),
            ];
            let burn_adapters = host_adapters
                .iter()
                .map(|adapter| BurnAdapterParams::from_adapter(adapter, &model, &device).unwrap())
                .collect::<Vec<_>>();
            let batch_adapter = BurnAdapterBatch::from_indices(&burn_adapters, &[0, 1]);
            let expanded_adapter = batch_adapter.clone().select_rows(&[0, 0, 1, 1]);
            let expanded_w1 = tensor3_vec(expanded_adapter.w1_down.inner()).unwrap();
            let adapter_row = rank * npa_config.perception_dims();
            assert_eq!(&expanded_w1[..adapter_row], &expanded_w1[adapter_row..2 * adapter_row]);
            assert_eq!(
                &expanded_w1[2 * adapter_row..3 * adapter_row],
                &expanded_w1[3 * adapter_row..4 * adapter_row]
            );
            assert_ne!(&expanded_w1[..adapter_row], &expanded_w1[2 * adapter_row..3 * adapter_row]);
            let rows = 3;
            let input_dims = npa_config.perception_dims();
            let feature_values = (0..2 * rows * input_dims)
                .map(|index| index as f32 * 0.001 - 0.25)
                .collect::<Vec<_>>();
            let batch_output = params.forward_adapter_batch(
                tensor3(
                    feature_values.clone(),
                    [2, rows, input_dims],
                    &device,
                ),
                &batch_adapter,
            );
            let batch_values = tensor3_vec(batch_output.inner()).unwrap();
            let output_dims = npa_config.update_dims();
            for batch in 0..2 {
                let feature_start = batch * rows * input_dims;
                let single_output = params.forward_adapter(
                    tensor(
                        feature_values[feature_start..feature_start + rows * input_dims].to_vec(),
                        [rows, input_dims],
                        &device,
                    ),
                    &burn_adapters[batch],
                    test_direct_config(particle_count),
                );
                let single_values = tensor_vec(single_output.inner()).unwrap();
                let batch_start = batch * rows * output_dims;
                for (actual, expected) in batch_values
                    [batch_start..batch_start + rows * output_dims]
                    .iter()
                    .zip(single_values)
                {
                    assert!(
                        (actual - expected).abs() < 1.0e-5,
                        "batched per-sample LoRA output diverged from unbatched output: {actual} vs {expected}"
                    );
                }
            }
        }

        #[test]
        fn target2d_batched_loss_and_gradients_match_unbatched_mean() {
            let device = BurnDevice::default();
            let npa_config = NpaConfig::growing_2d();
            let model = NpaModel::upstream_seeded(npa_config.clone(), 31);
            let targets = vec![
                test_burn_target(&device, 1.0, 0.1),
                test_burn_target(&device, 1.0, 0.2),
            ];
            let adapters = (0..2)
                .map(|_| {
                    BurnAdapterParams::from_adapter(
                        &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
                        &model,
                        &device,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let adapter_batch = BurnAdapterBatch::from_indices(&adapters, &[0, 1]);
            let mut config = test_direct_config(4);
            config.loss_config.image_size = 1;
            let x_values = vec![
                -0.30, -0.20, 0.15, -0.10, -0.05, 0.25, 0.30, 0.20,
                -0.25, 0.30, 0.20, 0.15, -0.10, -0.20, 0.35, -0.05,
            ];
            let s_values = (0..2 * 4 * npa_config.state_dims)
                .map(|idx| (idx as f32 * 0.013).sin() * 0.25)
                .collect::<Vec<_>>();
            let x_batch = tensor3(x_values.clone(), [2, 4, 2], &device).require_grad();
            let s_batch = tensor3(
                s_values.clone(),
                [2, 4, npa_config.state_dims],
                &device,
            )
            .require_grad();
            let batch_loss = target_splat_loss_batch(
                &x_batch,
                &s_batch,
                &targets,
                &[0, 1],
                config,
                &adapter_batch,
                Tensor::<BurnBackend, 1>::zeros([2], &device),
            )
            .unwrap();
            let batch_total = loss_scalars(&batch_loss).unwrap().total;
            let mut batch_grads = batch_loss.total.backward();
            let batch_x_grad = tensor3_vec(
                x_batch
                    .grad_remove(&mut batch_grads)
                    .expect("batched position gradient"),
            )
            .unwrap();
            let batch_s_grad = tensor3_vec(
                s_batch
                    .grad_remove(&mut batch_grads)
                    .expect("batched state gradient"),
            )
            .unwrap();

            let mut unbatched_total = 0.0_f32;
            let mut expected_x_grad = Vec::with_capacity(batch_x_grad.len());
            let mut expected_s_grad = Vec::with_capacity(batch_s_grad.len());
            for batch in 0..2 {
                let x_start = batch * 4 * 2;
                let s_start = batch * 4 * npa_config.state_dims;
                let x = tracked_tensor(
                    x_values[x_start..x_start + 4 * 2].to_vec(),
                    [4, 2],
                    &device,
                );
                let s = tracked_tensor(
                    s_values[s_start..s_start + 4 * npa_config.state_dims].to_vec(),
                    [4, npa_config.state_dims],
                    &device,
                );
                let loss = target_splat_loss(
                    &x,
                    &s,
                    &targets[batch],
                    config,
                    &adapters[batch],
                    Tensor::<BurnBackend, 1>::zeros([1], &device),
                );
                unbatched_total += loss_scalars(&loss).unwrap().total;
                let mut grads = loss.total.backward();
                expected_x_grad.extend(
                    tensor_vec(x.grad_remove(&mut grads).expect("position gradient"))
                        .unwrap()
                        .into_iter()
                        .map(|value| value * 0.5),
                );
                expected_s_grad.extend(
                    tensor_vec(s.grad_remove(&mut grads).expect("state gradient"))
                        .unwrap()
                        .into_iter()
                        .map(|value| value * 0.5),
                );
            }

            assert!((batch_total - unbatched_total * 0.5).abs() < 1.0e-5);
            assert!(max_abs_difference(&batch_x_grad, &expected_x_grad) < 1.0e-5);
            assert!(max_abs_difference(&batch_s_grad, &expected_s_grad) < 1.0e-5);
        }

        #[test]
        #[cfg($perception_cube_feature)]
        fn target2d_training_loss_routes_auto_to_device_adjoint() {
            let device = BurnDevice::default();
            let npa_config = NpaConfig::growing_2d();
            let model = NpaModel::upstream_seeded(npa_config.clone(), 37);
            let targets = vec![test_burn_target(&device, 1.0, 0.1)];
            let adapters = vec![
                BurnAdapterParams::from_adapter(
                    &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
                    &model,
                    &device,
                )
                .unwrap(),
            ];
            let adapter_batch = BurnAdapterBatch::from_indices(&adapters, &[0]);
            let mut config = test_direct_config(128);
            config.loss_config.image_size = 1;
            config.target2d_loss_backend = Target2dLossBackend::Auto;
            TARGET2D_CUBE_ADJOINT_DEVICE_HITS.store(0, Ordering::Relaxed);
            TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.store(0, Ordering::Relaxed);

            let loss = target_splat_loss_batch(
                &Tensor::<BurnBackend, 3>::zeros([1, 128, 2], &device).require_grad(),
                &Tensor::<BurnBackend, 3>::zeros(
                    [1, 128, npa_config.state_dims],
                    &device,
                )
                .require_grad(),
                &targets,
                &[0],
                config,
                &adapter_batch,
                Tensor::<BurnBackend, 1>::zeros([1], &device),
            )
            .unwrap();
            let scalar = loss.total.inner().into_scalar();

            assert!(scalar.is_finite());
            assert_eq!(
                TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed),
                1
            );
            assert_eq!(
                TARGET2D_CUBE_ADJOINT_FALLBACK_HITS.load(Ordering::Relaxed),
                0
            );
        }

        #[test]
        fn direct_particle_pool_persists_state_on_backend() {
            let device = BurnDevice::default();
            let config = test_direct_config(4);
            let mut pool = BurnDeviceParticlePool::new(2, 4, 16, 0.1, config, &device);
            pool.update_batch(
                &[1],
                Tensor::<BurnBackend, 3>::full([1, 4, 2], 0.25, &device),
                Tensor::<BurnBackend, 3>::full([1, 4, 16], 0.5, &device),
            )
            .unwrap();

            let inner_device = pool.positions.device();
            let index = inner_index_tensor(&[1], &inner_device);
            assert!(
                tensor3_vec(pool.positions.clone().select(0, index.clone()))
                    .unwrap()
                    .iter()
                    .all(|value| (*value - 0.25).abs() < 1.0e-6)
            );
            assert!(
                tensor3_vec(pool.states.clone().select(0, index))
                    .unwrap()
                    .iter()
                    .all(|value| (*value - 0.5).abs() < 1.0e-6)
            );
        }

        #[test]
        fn hyper_e2e_device_particle_pool_is_bounded_and_persistent() {
            let device = BurnDevice::default();
            let particle_count = 4;
            let state_dims = 16;
            let mut pool = BurnE2eDeviceParticlePool::new(
                2,
                particle_count,
                state_dims,
                2,
                &device,
            );
            let mut rng = StdRng::seed_from_u64(7);
            let config = test_direct_config(particle_count);
            let first = pool
                .sample_batch(&[10, 10], &mut rng, &[], 0.1, config, &device)
                .unwrap();
            assert_eq!(first.slots.len(), 2);
            assert_ne!(first.slots[0], first.slots[1]);
            pool.update_batch(
                &first.slots,
                Tensor::<BurnBackend, 3>::full([2, particle_count, 2], 0.25, &device),
                Tensor::<BurnBackend, 3>::full(
                    [2, particle_count, state_dims],
                    0.5,
                    &device,
                ),
            )
            .unwrap();
            let persisted = pool
                .sample_batch(&[10, 10], &mut rng, &[], 0.1, config, &device)
                .unwrap();
            assert!(
                tensor3_vec(persisted.x.inner())
                    .unwrap()
                    .iter()
                    .all(|value| (*value - 0.25).abs() < 1.0e-6)
            );
            assert!(
                tensor3_vec(persisted.s.inner())
                    .unwrap()
                    .iter()
                    .all(|value| (*value - 0.5).abs() < 1.0e-6)
            );
            let refreshed = pool
                .sample_batch(&[10, 10], &mut rng, &[0, 1], 0.1, config, &device)
                .unwrap();
            assert_eq!(refreshed.seed_replacements, 2);
            assert!(
                tensor3_vec(refreshed.x.inner())
                    .unwrap()
                    .iter()
                    .any(|value| (*value - 0.25).abs() > 1.0e-3)
            );
            pool.update_batch(
                &first.slots,
                Tensor::<BurnBackend, 3>::full([2, particle_count, 2], 2.0, &device),
                Tensor::<BurnBackend, 3>::full(
                    [2, particle_count, state_dims],
                    -3.0,
                    &device,
                ),
            )
            .unwrap();
            let clamped = pool
                .sample_batch(&[10], &mut rng, &[], 0.1, config, &device)
                .unwrap();
            assert!(
                tensor3_vec(clamped.x.inner())
                    .unwrap()
                    .iter()
                    .all(|value| (*value - 1.0).abs() < 1.0e-6)
            );
            assert!(
                tensor3_vec(clamped.s.inner())
                    .unwrap()
                    .iter()
                    .all(|value| (*value + 3.0).abs() < 1.0e-6)
            );
            pool.sample_batch(&[12], &mut rng, &[], 0.1, config, &device)
                .unwrap();
            assert_eq!(pool.example_slots.len(), 2);
            assert!(pool.example_slots.keys().any(|(example, _)| *example == 12));
        }

        #[test]
        fn hyper_e2e_seed_replacement_cadence_is_per_identity_trajectory() {
            let mut counts = vec![0usize; 2];
            let identities = [0usize, 0, 1, 1];
            assert!(
                per_identity_seed_replacement_rows(&identities, &mut counts, 4).is_empty()
            );
            assert_eq!(counts, [2, 2]);
            assert_eq!(
                per_identity_seed_replacement_rows(&identities, &mut counts, 4),
                [1, 3]
            );
            assert_eq!(counts, [0, 0]);
        }

        fn reference_perception_state_finite_difference(
            positions: &[[f32; 4]],
            states: &[f32],
            particle_count: usize,
            state_dims: usize,
            grid_eps: f32,
            feature_adjoint: &[f32],
            state_idx: usize,
        ) -> f32 {
            let eps = 1.0e-4;
            let grid = perception_reference_grid(grid_eps);
            let options = perception_reference_options(grid_eps);
            let mut plus_states = states.to_vec();
            plus_states[state_idx] += eps;
            let plus = burn_automata_kernels::perceive_with_options(
                positions,
                &plus_states,
                1,
                particle_count,
                state_dims,
                &grid,
                options,
            )
            .unwrap();
            let mut minus_states = states.to_vec();
            minus_states[state_idx] -= eps;
            let minus = burn_automata_kernels::perceive_with_options(
                positions,
                &minus_states,
                1,
                particle_count,
                state_dims,
                &grid,
                options,
            )
            .unwrap();
            let plus_loss = plus
                .features
                .iter()
                .zip(feature_adjoint)
                .map(|(feature, adjoint)| feature * adjoint)
                .sum::<f32>();
            let minus_loss = minus
                .features
                .iter()
                .zip(feature_adjoint)
                .map(|(feature, adjoint)| feature * adjoint)
                .sum::<f32>();
            (plus_loss - minus_loss) / (2.0 * eps)
        }

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
                target2d_loss_backend: Target2dLossBackend::Dense,
                perception_backend: PerceptionRolloutBackend::Dense,
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
        fn perception_tiled_adjoint_matches_dense_vjp_fixture() {
            let npa_config = NpaConfig::growing_2d();
            let grid = burn_automata_kernels::HashGridConfig::growing_2d();
            let (positions, states) = seed_particles_scaled(
                1,
                5,
                npa_config.state_dims,
                npa_config.spatial_dims,
                23,
                crate::ParticleSeed::UniformCircle,
                0.12,
            );
            let device = BurnDevice::default();
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
                stopgrad_pos: false,
                stopgrad_state: false,
                rollout_particles: 5,
                rollout_step_min: 1,
                rollout_steps: 1,
                update_prob: 1.0,
                seed: 23,
                seed_scale: 0.12,
                seed_mode: crate::ParticleSeed::UniformCircle,
                grid_eps: grid.eps,
                motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
                loss_config: crate::Target2dLossConfig::default(),
                target2d_loss_backend: Target2dLossBackend::Dense,
                perception_backend: PerceptionRolloutBackend::Dense,
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
                eval_seed: 23,
                system_memory_budget_gb: None,
                gpu_memory_budget_gb: None,
                max_dense_train_particles: 5,
                max_dense_chunk_floats: 1_000_000,
                max_splat_chunk_floats: 1_000_000,
            };
            let position_values = positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect::<Vec<_>>();
            let reference_positions = positions.clone();
            let reference_states = states.clone();
            let x_dense = tensor3(position_values.clone(), [1, 5, 2], &device).require_grad();
            let s_dense = tensor3(states.clone(), [1, 5, npa_config.state_dims], &device)
                .require_grad();
            let dense_features = dense_perception_batch(&x_dense, &s_dense, config);
            let feature_dims = dense_features.shape().dims::<3>()[2];
            let feature_weights = (0..feature_dims * 5)
                .map(|idx| (((idx * 17) % 13) as f32 - 6.0) * 0.01)
                .collect::<Vec<_>>();
            let reference_feature_weights = feature_weights.clone();
            let weights = tensor3(feature_weights, [1, 5, feature_dims], &device);
            let dense_loss = dense_features.clone().mul(weights.clone()).sum();
            let dense_values = tensor3_vec(dense_features.inner()).unwrap();
            let mut dense_grads = dense_loss.backward();
            let dense_x_grad = tensor3_vec(
                x_dense
                    .grad_remove(&mut dense_grads)
                    .unwrap_or_else(|| x_dense.clone().inner().zeros_like()),
            )
            .unwrap();
            let dense_s_grad = tensor3_vec(
                s_dense
                    .grad_remove(&mut dense_grads)
                    .unwrap_or_else(|| s_dense.clone().inner().zeros_like()),
            )
            .unwrap();

            let x_tiled = tensor3(position_values, [1, 5, 2], &device).require_grad();
            let s_tiled = tensor3(states, [1, 5, npa_config.state_dims], &device).require_grad();
            let tiled_features = perception_tiled_adjoint_batch(x_tiled.clone(), s_tiled.clone(), config);
            let tiled_loss = tiled_features.clone().mul(weights).sum();
            let tiled_values = tensor3_vec(tiled_features.inner()).unwrap();
            let mut tiled_grads = tiled_loss.backward();
            let tiled_x_grad = tensor3_vec(
                x_tiled
                    .grad_remove(&mut tiled_grads)
                    .unwrap_or_else(|| x_tiled.clone().inner().zeros_like()),
            )
            .unwrap();
            let tiled_s_grad = tensor3_vec(
                s_tiled
                    .grad_remove(&mut tiled_grads)
                    .unwrap_or_else(|| s_tiled.clone().inner().zeros_like()),
            )
            .unwrap();

            let feature_diff = max_abs_difference(&dense_values, &tiled_values);
            let (position_grad_idx, position_grad_diff, position_dense, position_tiled) =
                max_abs_difference_with_index(&dense_x_grad, &tiled_x_grad);
            let (state_grad_idx, state_grad_diff, state_dense, state_tiled) =
                max_abs_difference_with_index(&dense_s_grad, &tiled_s_grad);
            let manual_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
                &reference_positions,
                &reference_states,
                1,
                5,
                npa_config.state_dims,
                &perception_reference_grid(grid.eps),
                perception_reference_options(grid.eps),
                &reference_feature_weights,
            )
            .unwrap();
            let manual_state = manual_adjoint.state[state_grad_idx];
            let finite_state = reference_perception_state_finite_difference(
                &reference_positions,
                &reference_states,
                5,
                npa_config.state_dims,
                grid.eps,
                &reference_feature_weights,
                state_grad_idx,
            );
            let state_grad_relative = state_grad_diff
                / state_dense
                    .abs()
                    .max(state_tiled.abs())
                    .max(1.0);
            assert!(
                feature_diff < 2.0e-3,
                "tiled perception features diverged from dense Burn features: max_abs_diff={feature_diff}"
            );
            assert!(
                position_grad_diff < 1.0e-1,
                "tiled perception position VJP diverged from dense Burn VJP: idx={position_grad_idx} dense={position_dense} tiled={position_tiled} max_abs_diff={position_grad_diff}"
            );
            assert!(
                state_grad_diff < 3.0 && state_grad_relative < 3.5e-2,
                "tiled perception state VJP diverged from dense Burn VJP: idx={state_grad_idx} dense={state_dense} tiled={state_tiled} manual={manual_state} finite={finite_state} max_abs_diff={state_grad_diff} rel_diff={state_grad_relative}"
            );
        }

        #[cfg($perception_cube_feature)]
        #[test]
        fn perception_sparse_grid_matches_reference_state_vjp_at_threshold() {
            let npa_config = NpaConfig::growing_2d();
            let grid = crate::upstream_growing_2d_hashgrid();
            let particle_count = 512;
            let (positions, states) = seed_particles_scaled(
                1,
                particle_count,
                npa_config.state_dims,
                npa_config.spatial_dims,
                91,
                crate::ParticleSeed::UniformCircle,
                0.5,
            );
            let options = perception_reference_options(grid.eps);
            let reference = burn_automata_kernels::perceive_with_options(
                &positions,
                &states,
                1,
                particle_count,
                npa_config.state_dims,
                &grid,
                options,
            )
            .unwrap();
            let feature_dims = reference.feature_dims;
            let feature_weights = (0..particle_count * feature_dims)
                .map(|idx| (((idx * 17) % 19) as f32 - 9.0) * 1.0e-4)
                .collect::<Vec<_>>();
            let reference_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
                &positions,
                &states,
                1,
                particle_count,
                npa_config.state_dims,
                &grid,
                options,
                &feature_weights,
            )
            .unwrap();

            let device = BurnDevice::default();
            let position_values = positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect::<Vec<_>>();
            let x = tensor3(position_values, [1, particle_count, 2], &device);
            let s = tensor3(
                states,
                [1, particle_count, npa_config.state_dims],
                &device,
            )
            .require_grad();
            let mut config = test_direct_config(particle_count);
            config.stopgrad_pos = true;
            config.stopgrad_state = false;
            config.grid_eps = grid.eps;
            config.perception_backend = PerceptionRolloutBackend::TiledAdjoint;
            let cube_config = perception_cube_adjoint_config(grid.eps, false, true);
            let feature_grad_inner = Tensor::<InnerBackend, 3>::from_data(
                TensorData::new(
                    feature_weights.clone(),
                    [1, particle_count, feature_dims],
                ),
                &x.clone().inner().device(),
            );
            let prepared_forward = InnerBackend::perception_cube_forward_prepared(
                x.clone().inner(),
                s.clone().inner(),
                cube_config,
            )
            .expect("prepared perception backend")
            .expect("prepared perception forward");
            let prepared_adjoint = InnerBackend::perception_cube_adjoint_prepared(
                x.clone().inner(),
                s.clone().inner(),
                feature_grad_inner.clone(),
                prepared_forward.density,
                prepared_forward.offsets,
                prepared_forward.permutation,
                prepared_forward.raw_state_gradient,
                prepared_forward.state_gradient_inverse,
                cube_config,
            )
            .expect("prepared perception adjoint backend")
            .expect("prepared perception adjoint");
            let recomputed_adjoint = InnerBackend::perception_cube_adjoint(
                x.clone().inner(),
                s.clone().inner(),
                feature_grad_inner,
                cube_config,
            )
            .expect("recomputed perception adjoint backend")
            .expect("recomputed perception adjoint");
            let prepared_state = tensor3_vec(prepared_adjoint.state_grad).unwrap();
            let recomputed_state = tensor3_vec(recomputed_adjoint.state_grad).unwrap();
            let prepared_recomputed_diff = max_abs_difference(&prepared_state, &recomputed_state);
            assert!(
                prepared_recomputed_diff < 2.0e-5,
                "retained-state perception VJP diverged from recomputed sparse VJP: max_abs_diff={prepared_recomputed_diff}"
            );
            let prepared_reuse_before =
                PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed);
            let features = perception_tiled_adjoint_batch(x, s.clone(), config);
            let weights = tensor3(
                feature_weights,
                [1, particle_count, feature_dims],
                &device,
            );
            let loss = features.clone().mul(weights).sum();
            let feature_values = tensor3_vec(features.inner()).unwrap();
            let mut grads = loss.backward();
            let state_grad = tensor3_vec(
                s.grad_remove(&mut grads)
                    .unwrap_or_else(|| s.clone().inner().zeros_like()),
            )
            .unwrap();

            let feature_diff = max_abs_difference(&feature_values, &reference.features);
            let (_, state_diff, state_reference, state_sparse) =
                max_abs_difference_with_index(&reference_adjoint.state, &state_grad);
            let state_relative = state_diff
                / state_reference.abs().max(state_sparse.abs()).max(1.0e-4);
            assert!(
                PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed)
                    > prepared_reuse_before,
                "sparse perception backward did not reuse its forward grid/density"
            );
            assert!(
                feature_diff < 1.0e-2,
                "sparse perception forward diverged from the 512-particle reference: max_abs_diff={feature_diff}"
            );
            assert!(
                state_diff < 6.0e-3 && state_relative < 5.0e-2,
                "sparse perception state VJP diverged from the 512-particle reference: reference={state_reference} sparse={state_sparse} max_abs_diff={state_diff} rel_diff={state_relative}"
            );
        }

        #[cfg($perception_cube_feature)]
        #[test]
        #[ignore = "opt-in GPU perception throughput benchmark"]
        fn benchmark_perception_sparse_grid_forward_and_state_vjp() {
            let npa_config = NpaConfig::growing_2d();
            let grid = crate::upstream_growing_2d_hashgrid();
            let device = BurnDevice::default();
            let inner_device = device.clone().into();
            for particle_count in [512usize, 1024, 2048, 4096] {
                let (positions, states) = seed_particles_scaled(
                    1,
                    particle_count,
                    npa_config.state_dims,
                    npa_config.spatial_dims,
                    7331 + particle_count as u64,
                    crate::ParticleSeed::UniformCircle,
                    0.5,
                );
                let x = tensor3(
                    positions
                        .iter()
                        .flat_map(|position| [position[0], position[1]])
                        .collect(),
                    [1, particle_count, 2],
                    &device,
                )
                .inner();
                let s = tensor3(
                    states,
                    [1, particle_count, npa_config.state_dims],
                    &device,
                )
                .inner();
                let feature_dims = npa_config.perception_dims();
                let feature_grad = tensor3(
                    (0..particle_count * feature_dims)
                        .map(|idx| (((idx * 17) % 19) as f32 - 9.0) * 1.0e-4)
                        .collect(),
                    [1, particle_count, feature_dims],
                    &device,
                )
                .inner();

                for (mode, sparse_grid_min_particles, reuse_forward) in [
                    ("all_pairs", u32::MAX, false),
                    ("sparse_grid", 1u32, false),
                    ("sparse_prepared", 1u32, true),
                ]
                {
                    let mut cube_config =
                        perception_cube_adjoint_config(grid.eps, false, true);
                    cube_config.sparse_grid_min_particles = sparse_grid_min_particles;
                    let run = || {
                        if reuse_forward {
                            let forward = InnerBackend::perception_cube_forward_prepared(
                                x.clone(),
                                s.clone(),
                                cube_config,
                            )
                            .expect("CubeCL prepared perception forward backend")
                            .expect("CubeCL prepared perception forward");
                            let adjoint = InnerBackend::perception_cube_adjoint_prepared(
                                x.clone(),
                                s.clone(),
                                feature_grad.clone(),
                                forward.density.clone(),
                                forward.offsets.clone(),
                                forward.permutation.clone(),
                                forward.raw_state_gradient.clone(),
                                forward.state_gradient_inverse.clone(),
                                cube_config,
                            )
                            .expect("CubeCL prepared perception adjoint backend")
                            .expect("CubeCL prepared perception adjoint");
                            (forward.features, adjoint)
                        } else {
                            let forward = InnerBackend::perception_cube_forward(
                                x.clone(),
                                s.clone(),
                                cube_config,
                            )
                            .expect("CubeCL perception forward backend")
                            .expect("CubeCL perception forward");
                            let adjoint = InnerBackend::perception_cube_adjoint(
                                x.clone(),
                                s.clone(),
                                feature_grad.clone(),
                                cube_config,
                            )
                            .expect("CubeCL perception adjoint backend")
                            .expect("CubeCL perception adjoint");
                            (forward.features, adjoint)
                        }
                    };
                    let warmup = run();
                    <InnerBackend as Backend>::sync(&inner_device).unwrap();
                    drop(warmup);

                    let repeats = 50usize;
                    let start = Instant::now();
                    for _ in 0..repeats {
                        let output = run();
                        <InnerBackend as Backend>::sync(&inner_device).unwrap();
                        drop(output);
                    }
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
                    println!(
                        "perception_cube_bench mode={mode} particles={particle_count} repeats={repeats} mean_forward_vjp_ms={:.6} particle_steps_per_sec={:.0}",
                        elapsed_ms / repeats as f64,
                        particle_count as f64 * repeats as f64 / (elapsed_ms / 1_000.0),
                    );
                }
            }
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
                composited_rgb_loss_weight: 0.75,
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
                target_cpu: target.clone(),
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
                target2d_loss_backend: Target2dLossBackend::Dense,
                perception_backend: PerceptionRolloutBackend::Dense,
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
            let s = tracked_tensor(states.clone(), [4, npa_config.state_dims], &device);
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

            let x3 = tensor3(
                positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect(),
                [1, 4, 2],
                &device,
            )
            .require_grad();
            let s3 = tensor3(states, [1, 4, npa_config.state_dims], &device).require_grad();
            let adapter_batch = BurnAdapterBatch::from_indices(std::slice::from_ref(&adapter), &[0]);
            let tiled_config = DirectBasisTrainConfig {
                target2d_loss_backend: Target2dLossBackend::TiledAdjoint,
                perception_backend: PerceptionRolloutBackend::Dense,
                ..config
            };
            let tiled_loss = target_splat_loss_batch_vector_selected(
                &x3,
                &s3,
                &[target],
                &[0],
                tiled_config,
                &adapter_batch,
                Tensor::<BurnBackend, 1>::zeros([1], &device),
            )
            .unwrap();
            let tiled_scalars = loss_vector_scalars(tiled_loss.clone()).unwrap();
            let tiled_scalar = tiled_scalars[0];
            assert!(
                (tiled_scalar.total - reference.total_loss).abs() < 1.0e-4,
                "tiled-adjoint total target2d loss diverged from CPU reference: tiled={} reference={}",
                tiled_scalar.total,
                reference.total_loss
            );
            assert!(
                (tiled_scalar.splat - reference.splat_loss).abs() < 1.0e-4,
                "tiled-adjoint splat target2d loss diverged from CPU reference: tiled={} reference={}",
                tiled_scalar.splat,
                reference.splat_loss
            );
            assert!(
                (tiled_scalar.color - reference.color_loss).abs() < 1.0e-4,
                "tiled-adjoint color target2d loss diverged from CPU reference: tiled={} reference={}",
                tiled_scalar.color,
                reference.color_loss
            );
            assert!(
                (tiled_scalar.density - reference.density_loss).abs() < 1.0e-4,
                "tiled-adjoint density target2d loss diverged from CPU reference: tiled={} reference={}",
                tiled_scalar.density,
                reference.density_loss
            );

            let mut grads = tiled_loss.total.sum().backward();
            let tiled_x_grad = tensor3_vec(
                x3.grad_remove(&mut grads)
                    .unwrap_or_else(|| x3.clone().inner().zeros_like()),
            )
            .unwrap();
            let tiled_s_grad = tensor3_vec(
                s3.grad_remove(&mut grads)
                    .unwrap_or_else(|| s3.clone().inner().zeros_like()),
            )
            .unwrap();
            let max_position_grad_diff = tiled_x_grad
                .chunks_exact(2)
                .zip(&reference_output.position_gradients)
                .flat_map(|(burn, reference)| {
                    [
                        (burn[0] - reference[0]).abs(),
                        (burn[1] - reference[1]).abs(),
                    ]
                })
                .fold(0.0_f32, f32::max);
            let max_state_grad_diff = tiled_s_grad
                .iter()
                .zip(&reference_output.state_gradients)
                .map(|(burn, reference)| (burn - reference).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_position_grad_diff < 1.0e-6,
                "tiled-adjoint target2d position gradient diverged from CPU reference: max_abs_diff={max_position_grad_diff}"
            );
            assert!(
                max_state_grad_diff < 1.0e-6,
                "tiled-adjoint target2d state gradient diverged from CPU reference: max_abs_diff={max_state_grad_diff}"
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
        fn checkpoint_selection_does_not_compare_different_validation_contracts() {
            let frequent = BurnE2eValidationContract {
                examples: 16,
                particles: 2_048,
                horizons: vec![512],
                selection_horizon_min_steps: 512,
            };
            let final_contract = BurnE2eValidationContract {
                examples: 16,
                particles: 4_096,
                horizons: vec![96, 256, 512, 1_024],
                selection_horizon_min_steps: 256,
            };

            assert!(!comparable_selection_score_is_better(
                Some(&frequent),
                15.049,
                Some(&final_contract),
                13.707,
            ));
            assert!(comparable_selection_score_is_better(
                Some(&frequent),
                15.049,
                None,
                -0.185,
            ));
            assert!(!comparable_selection_score_is_better(
                None,
                -0.180,
                Some(&frequent),
                15.049,
            ));
            assert!(comparable_selection_score_is_better(
                Some(&final_contract),
                13.707,
                Some(&final_contract),
                13.236,
            ));
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
    feature = "backend_wgpu",
    burn::backend::Wgpu<f32>,
    "burn_wgpu_autodiff_dense_direct_basis",
    "wgpu-default",
    "burn-wgpu"
);

dense_direct_basis_backend!(
    ndarray_imp,
    all(test, feature = "backend_ndarray"),
    any(),
    burn::backend::NdArray<f32>,
    "burn_ndarray_autodiff_dense_direct_basis",
    "ndarray-default",
    "burn-ndarray"
);

dense_direct_basis_backend!(
    cuda_imp,
    feature = "backend_cuda",
    feature = "backend_cuda",
    burn::backend::Cuda<f32>,
    "burn_cuda_autodiff_dense_direct_basis",
    "cuda-default",
    "burn-cuda"
);
