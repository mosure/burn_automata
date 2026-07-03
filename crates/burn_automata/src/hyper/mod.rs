//! 2D conditioned Hyper-NPA building blocks.
//!
//! This module keeps the image-conditioned path separate from the fixed-model
//! rollout APIs. The first supported target is a single 2D condition that emits
//! one LoRA adapter and a particle prior for an existing base NPA.

pub mod condition;
pub mod hypernet;
pub mod inference;
pub mod prior;
pub mod training;

pub use condition::{
    CONDITION_FEATURE_DIMS, CONDITION_TOKEN_FEATURE_DIMS, ConditionImage2d, ConditionSummary2d,
    ConditionToken2d, DEFAULT_CONDITION_TOKEN_GRID_HEIGHT, DEFAULT_CONDITION_TOKEN_GRID_WIDTH,
    condition_feature_dims_for_token_grid,
};
pub use hypernet::{HyperNpa2d, HyperNpa2dConfig, HyperNpa2dWeights};
pub use inference::{ConditionedNpa2d, generate_conditioned_npa_2d};
pub use prior::{ParticlePrior2d, ParticlePriorConfig};
pub use training::{
    HyperAdapterExample2d, HyperAdapterTrainingReport, HyperFlowExample2d,
    hyper_adapter_regression_loss, hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step,
};
