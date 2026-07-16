//! Burn Neural Particle Automata.
//!
//! This crate contains the model/data APIs and a deterministic CPU implementation
//! used for parity fixtures. WGPU/CubeCL kernels are intended to replace the
//! reference kernels behind the same interfaces.

pub mod burn_bridge;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
pub mod error;
#[cfg(feature = "gpu_wgpu")]
pub mod gpu;
pub mod hyper;
pub mod import;
pub mod mesh_objective;
pub mod model;
pub mod pipeline;
pub mod render_loss;
pub mod rollout;
pub mod target2d;
pub mod target_geometry;
pub mod training;
#[cfg(feature = "backend_wgpu")]
pub mod training_wgpu;

pub use burn_automata_kernels as kernels;
pub use burn_automata_kernels::GaussianDecodeMode;
pub use config::{AutomataPreset, EquivarianceMode, ModelFormat, NpaConfig};
pub use error::{AutomataError, AutomataResult};
#[cfg(feature = "backend_cuda")]
pub use hyper::generate_e2e_conditioned_npa_2d_cuda;
#[cfg(feature = "backend_wgpu")]
pub use hyper::generate_e2e_conditioned_npa_2d_wgpu;
pub use hyper::{
    AdapterParameterGroup2d, AdapterParameterLayout2d, AdapterParameterSegment2d,
    AlphaAwareImageMetrics, CONDITION_FEATURE_DIMS, CONDITION_TOKEN_FEATURE_DIMS,
    ConditionEncoder2d, ConditionImage2d, ConditionSummary2d, ConditionToken2d, ConditionedNpa2d,
    DEFAULT_CONDITION_TOKEN_GRID_HEIGHT, DEFAULT_CONDITION_TOKEN_GRID_WIDTH,
    DEFAULT_DINO_VITS_TOKEN_GRID_HEIGHT, DEFAULT_DINO_VITS_TOKEN_GRID_WIDTH,
    DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS, DINO_VITS_EMBED_DIMS,
    DINO_VITS_PATCH_STATS_FEATURE_DIMS, E2eConditionedNpa2d, E2eHyperNpa2d,
    E2eHyperNpa2dAdapterSpec, E2eHyperNpa2dWeights, HyperAdapterExample2d,
    HyperAdapterTrainingReport, HyperFlowExample2d, HyperNpa2d, HyperNpa2dConfig, HyperNpa2dFlow,
    HyperNpa2dFlowActivation, HyperNpa2dFlowConfig, HyperNpa2dFlowWeights,
    HyperNpa2dOutputActivation, HyperNpa2dPreciseWeights, HyperNpa2dWeights, ParticlePrior2d,
    ParticlePriorConfig, alpha_aware_image_metrics, condition_feature_dims_for_encoder,
    condition_feature_dims_for_token_grid, generate_conditioned_npa_2d,
    generate_e2e_conditioned_npa_2d, hyper_adapter_regression_loss,
    hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step, load_e2e_hyper_npa_2d, save_e2e_hyper_npa_2d,
};
#[cfg(feature = "dino")]
pub use hyper::{
    DINO_CONDITION_BACKGROUND_RGB, DinoVitsConditionEncoder, decode_condition_image,
    load_condition_image,
};
pub use import::{
    BpkAdapterManifest, BpkModelManifest, ExportedCheckpoint, ImportReport,
    import_exported_checkpoint, import_model, load_adapter_manifest, save_adapter_manifest,
};
pub use mesh_objective::{GaussianVolumeStats, MeshRolloutObjectiveConfig};
pub use model::{NpaLowRankAdapter, NpaModel, NpaWeights, StepOutput};
pub use pipeline::{
    AutomataPipeline, FeatureBatchConfig, RolloutBatchConfig, RolloutSupervisionConfig,
    SupervisedTarget, feature_supervised_batch, rollout_supervised_batch,
    rollout_supervised_batch_from_model,
};
pub use render_loss::{
    MultiViewRenderLossReport, MultiViewRenderPositionGradient, RenderLossConfig,
    RenderViewLossReport, RenderViewPreset, mesh_multiview_render_loss_from_trace,
    mesh_multiview_render_position_gradient_for_rows_from_trace,
    mesh_multiview_render_position_gradient_from_trace, mesh_surface_render_samples,
};
pub use rollout::{
    MorphogenSeedEnvelope, ParticleSeed, RolloutConfig, RolloutTrace,
    morphogen_seed_envelope_position, run_rollout,
};
pub use target_geometry::{
    OvoxelTarget, TargetProjection, TargetSurfaceSample, TriangleMeshTarget,
};
pub use target2d::{
    TARGET_2D_COLOR_GATE_GRADIENT, Target2dColorGateGradient, Target2dLossConfig,
    Target2dLossOutput, Target2dLossReport, Target2dTrainingConfig, Target2dTrainingHistoryEntry,
    Target2dTrainingReport, Target2dUpstreamOneStepOutput, TargetImage2d,
    TargetImage2dExtractConfig, foreground_alpha_count_upstream, load_rgba_thumbnail_upstream,
    load_target_image_2d_upstream, target_2d_loss, target_2d_loss_with_adjoint,
    target_2d_rollout_loss_with_gradients, target_2d_upstream_one_step_with_gradients,
    train_target_2d, upstream_adaptive_target_image_size, upstream_growing_2d_hashgrid,
    upstream_growing_2d_model,
};
pub use training::{
    AdamWConfig, AdamWState, LowRankAdapterGradients, SgdConfig, SupervisedBatch,
    SupervisedGradients, SupervisedOptimizerConfig, SupervisedStepReport, TrainingHistoryEntry,
    TrainingRunConfig, TrainingRunReport, apply_adamw_gradients, apply_sgd_adapter_gradients,
    apply_sgd_gradients, mlp_backward_from_output_gradients, project_low_rank_adapter_gradients,
    run_supervised_adapter_training, run_supervised_training,
    run_supervised_training_with_optimizer, supervised_adamw_train_step, supervised_adapter_loss,
    supervised_adapter_train_step, supervised_backward, supervised_loss, supervised_train_step,
};
#[cfg(feature = "backend_wgpu")]
pub use training_wgpu::run_supervised_training_wgpu;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
