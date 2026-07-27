//! Mesh-conditioned 3D NPA training and quality evaluation.
//!
//! This module owns the shared mesh-to-model path used by CLI and interactive
//! frontends. It is distinct from the stricter conditionless 3D morphogenesis
//! research gate: imported geometry is compiled into a recurrent NPA vector
//! field through supervised mesh projections.

#[cfg(any(feature = "backend_wgpu", test))]
mod attractor;
mod config;
mod dataset;
mod evaluation;
mod render;
#[cfg(feature = "backend_wgpu")]
mod training;

pub use config::{
    Mesh3dEvaluationConfig, Mesh3dInitializationMode, Mesh3dQualityReport, Mesh3dRolloutReport,
    Mesh3dTrainingConfig, Mesh3dTrainingProgress, Mesh3dTrainingReport, Mesh3dTrainingStageReport,
    mesh3d_model_config,
};
pub use dataset::{
    mesh3d_damaged_initialization, mesh3d_supervised_batch, mesh3d_surface_initialization,
    mesh3d_volume_initialization,
};
pub use evaluation::evaluate_mesh3d_model;
pub use render::{Mesh3dGaussianGeometry, mesh3d_gaussian_geometry};
#[cfg(feature = "backend_wgpu")]
pub use training::{Mesh3dTrainingObserver, train_mesh3d_wgpu, train_mesh3d_wgpu_with_observer};
