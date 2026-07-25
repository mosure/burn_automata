mod config;
mod density;
#[cfg(feature = "gpu_wgpu")]
mod gap_decomposition;
mod graph;
mod operator;
mod runner;
mod scaling;
mod task_cut;
#[cfg(feature = "gpu_wgpu")]
mod task_wgpu;
mod topology;

pub use config::*;
pub use runner::{
    evaluate_adaptive_task_quality, evaluate_adaptive_task_quality_validation,
    run_adaptive_closure_audit, run_adaptive_experiment_suite, run_adaptive_topology_audit,
    validate_adaptive_task_quality_validation_gates,
};
