#![allow(unused_imports)]

pub(crate) use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(feature = "gpu_wgpu")]
pub(crate) use crate::kernels::build_hashgrid;
#[cfg(feature = "backend_wgpu")]
pub(crate) use crate::run_supervised_training_wgpu;
pub(crate) use crate::{
    AutomataPreset, BpkAdapterManifest, BpkModelManifest, ConditionImage2d, ConditionSummary2d,
    FeatureBatchConfig, GaussianDecodeMode, HyperAdapterExample2d, HyperFlowExample2d, HyperNpa2d,
    HyperNpa2dConfig, NpaConfig, NpaLowRankAdapter, NpaModel, NpaWeights, ParticlePrior2d,
    ParticlePriorConfig, ParticleSeed, RolloutConfig, RolloutSupervisionConfig, SupervisedBatch,
    SupervisedTarget, condition_feature_dims_for_token_grid, feature_supervised_batch,
    generate_conditioned_npa_2d, hyper_adapter_regression_loss,
    hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step, import_model,
    kernels::perceive_adjoint_with_options,
    kernels::{PerceptionOptions, euler_step, perceive_with_options},
    mesh_objective::{
        GaussianVolumeStats, MeshRolloutObjectiveConfig, ROBUST_3D_COVERAGE_GAIN,
        ROBUST_3D_COVERAGE_NORMAL_WEIGHT, ROBUST_3D_COVERAGE_REPULSION_GAIN,
        ROBUST_3D_COVERAGE_SAMPLES, ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        ROBUST_3D_EXTENT_GAIN, ROBUST_3D_LIVENESS_FRONT_RADIUS, ROBUST_3D_LIVENESS_GAIN,
        ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER, ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE, ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        ROBUST_3D_MATERIAL_TAIL_GAIN, ROBUST_3D_MAX_OVERSIZE_FRACTION,
        ROBUST_3D_MAX_SCALE_BUDGET_LOSS, ROBUST_3D_OPACITY_GAIN, ROBUST_3D_PHASE_GAIN,
        ROBUST_3D_SCALE_BUDGET_WEIGHT, ROBUST_3D_SCALE_GAIN, ROBUST_3D_SURFACE_ESCAPE_GAIN,
        ROBUST_3D_SURFACE_GAIN, ROBUST_3D_TRAJECTORY_MESH_GAIN, ROBUST_3D_TRAJECTORY_RENDER_GAIN,
        ROBUST_3D_TRAJECTORY_RENDER_SAMPLES, scale_budget_loss_for_scale,
    },
    render_loss::{
        MultiViewRenderLossReport, RenderLossConfig, RenderViewLossReport,
        mesh_multiview_render_loss_from_trace,
        mesh_multiview_render_position_gradient_for_rows_from_trace,
    },
    rollout::{
        GROWTH_3D_INACTIVE_OPACITY_LOGIT, GROWTH_3D_LIVENESS_CHANNEL,
        GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT, GROWTH_3D_PHASE_CHANNEL,
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_MOTION_GAIN, UV_TORUS_NORMAL_STATE_OFFSET, UV_TORUS_OPACITY_GROWTH_DELTA,
        UV_TORUS_RESIDUAL_DECAY, UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
        growth_3d_material_opacity_channel, growth_3d_phase_channel, growth_3d_seed_radius,
        growth_3d_seed_writes_coordinate_scaffold, growth_3d_velocity_channels,
        seed_particles_scaled, utah_teapot_dense_seed_position, utah_teapot_tail_state_color,
        uv_torus_continuous_surface_position, uv_torus_continuous_volume_position,
        uv_torus_dense_seed_position, uv_torus_orientation_state_available,
        uv_torus_position_color, uv_torus_sample, uv_torus_surface_error,
        uv_torus_tail_state_color, uv_torus_tail_state_to_rgb,
    },
    rollout_supervised_batch_from_model, run_rollout,
    target_geometry::{TriangleMeshTarget, dot3},
    training::{
        AdamWConfig, SgdConfig, SupervisedGradients, SupervisedOptimizerConfig,
        SupervisedStepReport, TrainingHistoryEntry, TrainingRunConfig, TrainingRunReport,
        apply_sgd_adapter_gradients, apply_sgd_gradients, mlp_backward_from_output_gradients,
        project_low_rank_adapter_gradients, run_supervised_adapter_training,
        run_supervised_training, run_supervised_training_with_optimizer, supervised_adapter_loss,
    },
};
pub(crate) use clap::{ArgAction, Parser, Subcommand, ValueEnum};
pub(crate) use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
pub(crate) use serde::Serialize;

pub(crate) use super::{
    args::*, bench::*, growth_validation::*, mesh_training::*, render_training::*, reports::*,
    targets::*,
};

pub(crate) const GROWTH_3D_MAX_FINAL_OPACITY_LOGIT: f32 = 24.0;
pub(crate) const GROWTH_3D_SURFACE_MAX_DISTANCE: f32 = 0.36;
pub(crate) const RENDER_SELECTION_BAD_SCORE: f32 = 1.0e9;
pub(crate) const CATALOG_3D_APP_EVAL_SEED: u64 = 0x0051_a73d;
pub(crate) const CATALOG_3D_HELD_OUT_SEEDS: [u64; 2] = [42, 99];
pub(crate) const CATALOG_3D_VALIDATION_PARTICLES: usize = 1024;
pub(crate) const CATALOG_3D_VALIDATION_IMAGE_SIZE: usize = 48;
pub(crate) const CATALOG_3D_VALIDATION_TARGET_SAMPLES: usize = 4096;
pub(crate) const CATALOG_3D_PROMOTION_STEPS: [usize; 2] = [64, 96];
