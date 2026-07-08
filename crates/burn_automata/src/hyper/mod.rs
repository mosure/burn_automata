//! 2D conditioned Hyper-NPA building blocks.
//!
//! This module keeps the image-conditioned path separate from the fixed-model
//! rollout APIs. The first supported target is a single 2D condition that emits
//! one LoRA adapter and a particle prior for an existing base NPA.

pub mod condition;
#[cfg(feature = "dino")]
pub mod dino;
pub mod e2e;
pub(crate) mod e2e_training;
pub mod hypernet;
pub mod inference;
pub mod prior;
pub mod training;

pub use condition::{
    CONDITION_FEATURE_DIMS, CONDITION_TOKEN_FEATURE_DIMS, ConditionEncoder2d, ConditionImage2d,
    ConditionSummary2d, ConditionToken2d, DEFAULT_CONDITION_TOKEN_GRID_HEIGHT,
    DEFAULT_CONDITION_TOKEN_GRID_WIDTH, DEFAULT_DINO_VITS_TOKEN_GRID_HEIGHT,
    DEFAULT_DINO_VITS_TOKEN_GRID_WIDTH, DINO_VITS_CLS_PATCH_MEAN_FEATURE_DIMS,
    DINO_VITS_EMBED_DIMS, DINO_VITS_PATCH_STATS_FEATURE_DIMS, condition_feature_dims_for_encoder,
    condition_feature_dims_for_token_grid,
};
#[cfg(feature = "dino")]
pub use dino::DinoVitsConditionEncoder;
pub use e2e::{
    E2eConditionedNpa2d, E2eHyperNpa2d, E2eHyperNpa2dAdapterSpec, E2eHyperNpa2dWeights,
    PerceptionRolloutBackend, Target2dLossBackend, generate_e2e_conditioned_npa_2d,
    load_e2e_hyper_npa_2d,
};
pub use hypernet::{
    HyperNpa2d, HyperNpa2dConfig, HyperNpa2dFlow, HyperNpa2dFlowActivation, HyperNpa2dFlowConfig,
    HyperNpa2dFlowWeights, HyperNpa2dOutputActivation, HyperNpa2dPreciseWeights, HyperNpa2dWeights,
};
pub use inference::{ConditionedNpa2d, generate_conditioned_npa_2d};
pub use prior::{ParticlePrior2d, ParticlePriorConfig};
pub use training::{
    HyperAdapterExample2d, HyperAdapterTrainingReport, HyperFlowExample2d,
    hyper_adapter_regression_loss, hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step,
};
