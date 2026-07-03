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
pub mod target_geometry;
pub mod training;
#[cfg(feature = "backend_wgpu")]
pub mod training_wgpu;

pub use burn_automata_kernels as kernels;
pub use burn_automata_kernels::GaussianDecodeMode;
pub use config::{AutomataPreset, EquivarianceMode, ModelFormat, NpaConfig};
pub use error::{AutomataError, AutomataResult};
pub use hyper::{
    CONDITION_FEATURE_DIMS, CONDITION_TOKEN_FEATURE_DIMS, ConditionImage2d, ConditionSummary2d,
    ConditionToken2d, ConditionedNpa2d, DEFAULT_CONDITION_TOKEN_GRID_HEIGHT,
    DEFAULT_CONDITION_TOKEN_GRID_WIDTH, HyperAdapterExample2d, HyperAdapterTrainingReport,
    HyperFlowExample2d, HyperNpa2d, HyperNpa2dConfig, HyperNpa2dWeights, ParticlePrior2d,
    ParticlePriorConfig, condition_feature_dims_for_token_grid, generate_conditioned_npa_2d,
    hyper_adapter_regression_loss, hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step,
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
