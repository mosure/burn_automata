//! Backend-generic Burn training implementation for 2D NPA and HyperNPA.
//!
//! The macro owns only backend aliases and shared state types. Training behavior
//! is split into the responsibility-focused child modules below and instantiated
//! identically for WGPU, CUDA, and the ndarray test backend.

mod backends;

#[cfg(feature = "backend_cuda")]
pub(crate) use backends::predict_conditional_row_flow_adapter_cuda;
#[cfg(feature = "backend_wgpu")]
pub(crate) use backends::predict_conditional_row_flow_adapter_wgpu;
pub(crate) use backends::{
    train_direct_basis_burn_cuda, train_direct_basis_burn_wgpu, train_oracle_models_burn_cuda,
    train_oracle_models_burn_wgpu,
};
pub(crate) use backends::{train_e2e_rollout_burn_cuda, train_e2e_rollout_burn_wgpu};

macro_rules! dense_backend_impl {
    (
        $inner_backend:ty,
        $backend_name:expr,
        $device_label:expr,
        $log_backend:expr
    ) => {
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
                activation::{gelu, relu, softmax},
                backend::Backend,
            },
        };
        use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
        use rayon::prelude::*;
        use serde::Serialize;
        use serde_json::json;
        use sha2::{Digest, Sha256};

        use super::super::{
            BurnDenseOracleBatchOutput, BurnE2eAdapterDiagnostics, BurnE2eNearestTeacherEntry,
            BurnE2eRolloutExample, BurnE2eRolloutHistoryEntry, BurnE2eRolloutHorizonSummary,
            BurnE2eRolloutOutput, BurnE2eRolloutQualityEntry, BurnE2eRolloutQualityReport,
            BurnE2eRolloutTrainConfig, BurnWgpuDirectBasisOutput, DirectBasisStepStats,
            DirectBasisTrainConfig, DirectBasisTrainingExample as DirectBasisExample,
            E2E_TRAINING_CHECKPOINT_VERSION, E2eAdapterTeacherObjective, E2eCreditAssignment,
            E2eIdentitySampler, E2eLrSchedule, E2eParticlePoolSnapshot, E2eTbpttLossMode,
            E2eTensorSnapshot, E2eTrainingCheckpoint,
            Hyper2dDirectBasisHistoryEntry as CliHyper2dDirectBasisHistoryEntry,
            Hyper2dDirectBasisLossSummary as CliHyper2dDirectBasisLossSummary,
            Target2dBurnCheckpointConfig,
        };
        #[cfg(feature = "dino")]
        use crate::ConditionImage2d;
        #[cfg(feature = "dino")]
        use crate::hyper::dino::{
            DinoVitsConditionContract, DinoVitsConditionEncoderBackend,
            DinoVitsPreparedConditionBatch,
        };
        use crate::hyper::e2e::{
            E2E_HYPER_ADAPTER_CANONICAL_FULL_RANK, E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL,
            E2E_HYPER_ADAPTER_FACTORIZED, E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW,
            E2E_HYPER_ATTENTION_SOFTMAX, E2eHyperGeneratorKind, E2eHyperNpa2d,
            E2eHyperNpa2dWeights, PerceptionRolloutBackend, Target2dLossBackend,
            save_e2e_hyper_npa_2d,
        };
        use crate::hyper::row_flow::{
            ConditionalRowFlowConfig, ConditionalRowFlowWeights, NpaParameterRowLayout2d,
        };
        use crate::{
            AdamWConfig, AutomataError, AutomataResult, BpkModelManifest, NpaConfig,
            NpaLowRankAdapter, NpaModel, NpaWeights, SgdConfig, TargetImage2d,
            rollout::{seed_particles_scaled, stochastic_mask},
            target2d::{render_target_2d_splat, target_2d_foreground_mask},
        };
        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        use burn_automata_kernels::ModulatedLayerNormCubeBackend;
        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        use burn_automata_kernels::{
            PerceptionCubeAdjointBackend, PerceptionCubeAdjointConfig,
            PerceptionCubeForwardBackend, PerceptionCubePreparedBackend,
        };
        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
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
        const PERCEPTION_CUBE_ENABLED: bool =
            cfg!(any(feature = "backend_wgpu", feature = "backend_cuda"));

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
            canonical_dense_residual: bool,
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

        struct BurnBaseBatchAdamWState {
            step: usize,
            w1_m: Tensor3Inner,
            w1_v: Tensor3Inner,
            b1_m: Tensor3Inner,
            b1_v: Tensor3Inner,
            w2_m: Tensor3Inner,
            w2_v: Tensor3Inner,
            b2_m: Tensor3Inner,
            b2_v: Tensor3Inner,
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
            row_flow: Option<BurnRowFlowParams>,
        }

        #[derive(Clone)]
        struct BurnRowFlowParams {
            config: ConditionalRowFlowConfig,
            tensors: Vec<Tensor2>,
            source_rows: Tensor3,
            row_scale: Tensor3,
            row_mask: Tensor3,
            time_frequencies: Tensor2,
        }

        #[derive(Clone)]
        struct BurnRowFlowCondition {
            key_values: Vec<Tensor3>,
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
            row_flow_m: Vec<Tensor2Inner>,
            row_flow_v: Vec<Tensor2Inner>,
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
                let particle_xy =
                    x.clone()
                        .unsqueeze_dim::<4>(2)
                        .expand([batches, particles, patch_tokens, 2]);
                let center_xy = centers.expand([batches, particles, patch_tokens, 2]);
                let diff = particle_xy - center_xy;
                let dist2 = diff.clone().mul(diff).sum_dim(3).squeeze_dim::<3>(3);
                let weights = dist2
                    .mul_scalar(-1.0 / (self.sigma * self.sigma).max(EPSILON))
                    .exp();
                let weights = weights
                    .clone()
                    .div(weights.sum_dim(2).add_scalar(EPSILON).expand([
                        batches,
                        particles,
                        patch_tokens,
                    ]));
                let mut local = weights.matmul(self.patch_hidden.clone());
                if let Some(state_w) = &self.state_w {
                    let state_dims = state.shape().dims::<3>()[2];
                    let projected = state.clone().matmul(
                        state_w.clone().transpose().unsqueeze_dim::<3>(0).expand([
                            batches,
                            state_dims,
                            hidden_dims,
                        ]),
                    );
                    local = relu(local + projected);
                }
                let update_w = self
                    .update_w
                    .clone()
                    .transpose()
                    .unsqueeze_dim::<3>(0)
                    .expand([batches, hidden_dims, update_dims]);
                let update_b = self.update_b.clone().unsqueeze_dim::<3>(0).expand([
                    batches,
                    particles,
                    update_dims,
                ]);
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

        #[path = "attention.rs"]
        mod attention;
        #[path = "entrypoints.rs"]
        pub(super) mod entrypoints;
        #[path = "evaluation.rs"]
        mod evaluation;
        #[path = "inputs.rs"]
        mod inputs;
        #[path = "layer_norm.rs"]
        mod layer_norm;
        #[path = "parameters.rs"]
        mod parameters;
        #[path = "rollout.rs"]
        mod rollout;
        #[path = "row_flow.rs"]
        mod row_flow;
        #[path = "runtime.rs"]
        mod runtime;
        #[path = "train_steps.rs"]
        mod train_steps;

        use attention::*;
        use entrypoints::*;
        use evaluation::*;
        use inputs::*;
        use layer_norm::*;
        use parameters::*;
        use rollout::*;
        use runtime::*;
        use train_steps::*;

        #[path = "perception.rs"]
        mod perception;
        #[path = "target_loss.rs"]
        mod target_loss;
        #[cfg(test)]
        #[path = "tests.rs"]
        mod tests;

        use perception::*;
        use target_loss::*;
    };
}

#[cfg(feature = "backend_wgpu")]
#[allow(dead_code)]
mod wgpu_imp;

#[cfg(all(
    test,
    feature = "backend_ndarray",
    not(any(feature = "backend_wgpu", feature = "backend_cuda"))
))]
#[allow(dead_code)]
mod ndarray_imp;

#[cfg(feature = "backend_cuda")]
#[allow(dead_code)]
mod cuda_imp;
