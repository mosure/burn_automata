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
pub mod import;
pub mod mesh_objective;
pub mod model;
pub mod pipeline;
pub mod render_loss;
pub mod rollout;
pub mod target_geometry;
pub mod training;

pub use burn_automata_kernels as kernels;
pub use burn_automata_kernels::GaussianDecodeMode;
pub use config::{AutomataPreset, EquivarianceMode, ModelFormat, NpaConfig};
pub use error::{AutomataError, AutomataResult};
pub use import::{
    BpkModelManifest, ExportedCheckpoint, ImportReport, import_exported_checkpoint, import_model,
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
    SgdConfig, SupervisedBatch, SupervisedGradients, SupervisedStepReport, TrainingHistoryEntry,
    TrainingRunConfig, TrainingRunReport, apply_sgd_gradients, mlp_backward_from_output_gradients,
    run_supervised_training, supervised_backward, supervised_loss, supervised_train_step,
};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
