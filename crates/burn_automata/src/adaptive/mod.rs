//! Budgeted adaptive Neural Particle Automata.
//!
//! The adaptive path is intentionally separate from the hardened fixed 2D NPA
//! implementation. It reuses the learned local rule, while owning represented
//! measure, variable support, budget allocation, and conservative topology.

mod artifact;
mod budget;
mod closure;
mod config;
mod controller;
mod dynamics;
mod experiments;
mod features;
#[cfg(feature = "gpu_wgpu")]
mod gpu;
mod hierarchy;
mod integration;
mod model;
mod perception;
mod refinement;
mod render;
mod restriction;
mod rollout;
mod scale;
mod seed;
mod state;
mod task_merge_oracle;
mod topology;
mod training;
mod validation;

pub use artifact::{
    AdaptiveModelArtifact, AdaptiveTrainingStage, load_adaptive_model, save_adaptive_model,
};
pub use budget::{
    BudgetAllocation, allocate_resolution_budget, boundary_protected_spacing,
    normalize_footprint_budget, normalize_footprint_budget_bounded,
};
pub use closure::{AdaptiveClosureModeField, AdaptiveClosureReconstructionMetrics};
pub use config::{
    AdaptiveCoarseDynamics, AdaptiveHierarchyRestrictionPolicy, AdaptiveLocalRuleSemantics,
    AdaptiveMaterialSeedLayout, AdaptiveNpaConfig, AdaptiveProxyConfig,
    AdaptiveResidualGateReference, AdaptiveRestrictionArity, AdaptiveRestrictionSchedule,
    AdaptiveRolloutConfig, AdaptiveRulePerception, AdaptiveTopologyControl,
    CONTINUOUS_LOCAL_DETAIL_MAX_EXCHANGES,
};
pub use controller::{
    ADAPTIVE_CONTROLLER_CONTEXT_DIMS, ADAPTIVE_CONTROLLER_INPUT_DIMS,
    ADAPTIVE_CONTROLLER_OUTPUT_DIMS, ADAPTIVE_CONTROLLER_SCALAR_DIMS, AdaptiveController,
    AdaptiveControllerOutput, AdaptiveControllerWeights,
};
pub use experiments::{
    AdaptiveBaseTrainingConfig, AdaptiveBaseTrainingPhase, AdaptiveBaseTrainingPhaseReport,
    AdaptiveBaseTrainingReport, AdaptiveClosureAuditConfig, AdaptiveDynamicsSemantics,
    AdaptiveExperimentConfig, AdaptiveExperimentReport, AdaptiveGapDecompositionConfig,
    AdaptiveGapDecompositionReport, AdaptiveGapDecompositionRow,
    AdaptiveMultiscaleExperimentReport, AdaptiveRestrictionExperimentConfig,
    AdaptiveRestrictionExperimentReport, AdaptiveTaskQualityConfig, AdaptiveTaskQualityReport,
    AdaptiveTaskQualityValidationReport, AdaptiveTaskRestrictionPolicy,
    AdaptiveTopologyAuditConfig, AdaptiveTopologyAuditReport, AdaptiveTopologyExperimentConfig,
    AdaptiveTopologyExperimentReport, AdaptiveTrainingBackend, evaluate_adaptive_task_quality,
    evaluate_adaptive_task_quality_validation, run_adaptive_closure_audit,
    run_adaptive_experiment_suite, run_adaptive_topology_audit,
    validate_adaptive_task_quality_validation_gates,
};
#[cfg(feature = "gpu_wgpu")]
pub use gpu::{WgpuAdaptiveNpaState, WgpuAdaptiveStepReport};
pub use hierarchy::{
    AdaptiveHierarchyMember, AdaptiveMaterialView, AdaptiveProxyHierarchy, AdaptiveProxyNode,
};
pub use model::AdaptiveNpaModel;
pub(crate) use render::diagnostic_covariance_gaussian_geometry;
pub use render::{
    AdaptiveGaussianGeometry, AdaptiveRenderDecoder, adaptive_display_scale_per_footprint,
    adaptive_isotropic_gaussian_geometry,
};
pub use rollout::{
    AdaptiveRolloutTrace, AdaptiveSnapshot, AdaptiveStepMetrics, AdaptiveTopologyUpdate,
    advance_adaptive_rollout, apply_adaptive_topology_at_step, run_adaptive_rollout,
};
pub use seed::seed_adaptive_particles_scaled;
pub use state::{
    AdaptiveBootstrapChild, AdaptiveBootstrapTemplate, AdaptiveParticleSet,
    material_footprint_radius, unit_ball_measure,
};
pub use topology::{
    CanonicalMaterial, TopologyAudit, canonical_merge, canonical_split, constrained_unequal_split,
    topology_audit,
};
#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu"))]
pub use training::adaptive_restriction_training_batch_burn;
#[cfg(all(
    feature = "gpu_wgpu",
    any(feature = "backend_cuda", feature = "backend_wgpu")
))]
pub use training::adaptive_restriction_training_batch_burn_with_executor;
pub(crate) use training::validate_adaptive_restriction_training_memory_plan;
pub use training::{
    AdaptiveClosureAuditBackend, AdaptiveClosureIdentifiabilityConfig,
    AdaptiveClosureIdentifiabilityReport, AdaptiveClosureModeTrainingReport,
    AdaptiveClosureModeValidationReport, AdaptiveControllerTrainConfig,
    AdaptiveControllerTrainingBatch, AdaptiveControllerTrainingHistory,
    AdaptiveControllerTrainingReport, AdaptiveDeploymentRuleTrainingReport,
    AdaptiveDeploymentRuleValidationReport, AdaptiveDeploymentStrategy, AdaptiveDeploymentTarget,
    AdaptiveMultiscaleDatasetReport, AdaptiveMultiscaleRuleStrategy,
    AdaptiveMultiscaleRuleTrainingReport, AdaptiveMultiscaleRuleValidationReport,
    AdaptiveMultiscaleTrainingBatch, AdaptiveMultiscaleTrainingConfig, AdaptiveOracleDatasetConfig,
    AdaptiveReplayBackend, AdaptiveReplayTeacher, AdaptiveRestrictionDatasetConfig,
    AdaptiveRestrictionDatasetReport, AdaptiveRestrictionLabelTarget,
    AdaptiveRestrictionSelectionReport, AdaptiveRestrictionTrainingBatch,
    AdaptiveRuleDistillationConfig, AdaptiveRuleDistillationReport, AdaptiveRuleTrainingBatch,
    AdaptiveRuleTrainingHistory, AdaptiveRuleTrainingReport, AdaptiveRuleValidationReport,
    AdaptiveTarget2dGpuTrainingReport, AdaptiveTarget2dMaterialConfig,
    AdaptiveTarget2dMaterialLayout, AdaptiveTarget2dRuleTraining, AdaptiveTarget2dTopologyConfig,
    AdaptiveTarget2dTrainingConfig, adaptive_closure_mode_validation,
    adaptive_deployment_on_policy_batch_wgpu, adaptive_deployment_rule_validation,
    adaptive_multiscale_on_policy_batch, adaptive_multiscale_rule_validation,
    adaptive_multiscale_rule_validation_cuda, adaptive_multiscale_rule_validation_ndarray,
    adaptive_multiscale_rule_validation_wgpu, adaptive_multiscale_training_batch,
    adaptive_oracle_training_batch, adaptive_restriction_training_batch,
    adaptive_rule_distillation_batch, adaptive_rule_on_policy_batch,
    audit_adaptive_closure_identifiability, audit_adaptive_closure_identifiability_wgpu,
    train_adaptive_closure_mode_rule_cuda, train_adaptive_closure_mode_rule_ndarray,
    train_adaptive_closure_mode_rule_wgpu, train_adaptive_deployment_rule_cuda,
    train_adaptive_deployment_rule_ndarray, train_adaptive_deployment_rule_wgpu,
    train_adaptive_multiscale_rule_cuda, train_adaptive_multiscale_rule_ndarray,
    train_adaptive_multiscale_rule_wgpu, train_adaptive_target_2d_gpu,
    validate_adaptive_restriction_selection,
};
pub(crate) use training::{
    AdaptiveTarget2dSeedBank, AdaptiveTarget2dUpdateMask, build_adaptive_target2d_seed_bank,
};
#[cfg(feature = "gpu_wgpu")]
pub use training::{
    adaptive_multiscale_on_policy_batch_wgpu_with_executor,
    adaptive_multiscale_training_batch_wgpu_with_executor,
};
#[cfg(feature = "backend_cuda")]
pub use training::{
    train_adaptive_controller_cuda, train_adaptive_restriction_controller_cuda,
    train_adaptive_rule_cuda, validate_adaptive_controller_cuda,
};
#[cfg(feature = "backend_ndarray")]
pub use training::{
    train_adaptive_controller_ndarray, train_adaptive_restriction_controller_ndarray,
    train_adaptive_rule_ndarray, validate_adaptive_controller_ndarray,
};
#[cfg(feature = "backend_wgpu")]
pub use training::{
    train_adaptive_controller_wgpu, train_adaptive_restriction_controller_wgpu,
    train_adaptive_rule_wgpu, validate_adaptive_controller_wgpu,
};
#[cfg(feature = "gpu_wgpu")]
pub use validation::validate_adaptive_target2d_wgpu;
pub use validation::{
    AdaptiveResolutionValidationSummary, AdaptiveTarget2dHorizonSummary,
    AdaptiveTarget2dValidationConfig, AdaptiveTarget2dValidationReport,
    AdaptiveTarget2dValidationRow,
};

#[cfg(test)]
mod tests;
