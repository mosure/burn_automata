use burn_automata::{
    AutomataPreset, BpkModelManifest, EquivarianceMode, MorphogenSeedEnvelope, NpaConfig, NpaModel,
    NpaWeights, ParticleSeed, RenderLossConfig, RolloutBatchConfig, RolloutConfig,
    RolloutSupervisionConfig, SgdConfig, SupervisedBatch, SupervisedTarget, TrainingRunConfig,
    feature_supervised_batch,
    kernels::build_hashgrid,
    mesh_multiview_render_loss_from_trace,
    rollout::{
        GROWTH_3D_ACTIVE_OPACITY_LOGIT, GROWTH_3D_INACTIVE_OPACITY_LOGIT,
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_NORMAL_STATE_OFFSET, UV_TORUS_OPACITY_GROWTH_DELTA,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET, growth_3d_active_core_radius,
        growth_3d_domain_radius, growth_3d_material_opacity_channel, growth_3d_seed_radius,
        morphogen_seed_envelope_position, seed_particles_scaled,
        uv_torus_continuous_surface_position, uv_torus_continuous_volume_position,
        uv_torus_dense_seed_radius, uv_torus_orientation_state_available, uv_torus_outer_radius,
        uv_torus_outward_normal, uv_torus_position_color, uv_torus_project_position,
        uv_torus_sample, uv_torus_signed_distance, uv_torus_surface_error,
        uv_torus_tail_state_to_rgb,
    },
    rollout_supervised_batch, rollout_supervised_batch_from_model, run_rollout,
    run_supervised_training, supervised_backward, supervised_loss, supervised_train_step,
    target_geometry::{TriangleMeshTarget, dot3},
};
use rand::{SeedableRng, rngs::StdRng};
use std::{collections::HashSet, fs};

const CATALOG_3D_GROWTH_SEED: u64 = 0x0051_a73d;

#[path = "core/support/mod.rs"]
mod support;
use support::*;

#[path = "core/catalog_growth.rs"]
mod catalog_growth;
#[path = "core/equivariance.rs"]
mod equivariance;
#[path = "core/import_bridge.rs"]
mod import_bridge;
#[path = "core/mesh_targets.rs"]
mod mesh_targets;
#[path = "core/seed_modes.rs"]
mod seed_modes;
#[path = "core/training.rs"]
mod training;
