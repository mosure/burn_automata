use super::prelude::*;

pub(crate) const DEFAULT_GROWTH_TARGET_SEED: u64 = 42;
pub(crate) const UV_TORUS_FIELD_MOTION_GAIN: f32 = 8.0;
pub(crate) const UV_TORUS_FIELD_COLOR_GAIN: f32 = 0.16;
pub(crate) const DEFAULT_3D_FIELD_OPACITY_TARGET: f32 = 6.0;
pub(crate) const DEFAULT_3D_FIELD_OPACITY_GAIN: f32 = 0.10;
pub(crate) const UV_TORUS_FIELD_OPACITY_TARGET: f32 = DEFAULT_3D_FIELD_OPACITY_TARGET;
pub(crate) const UV_TORUS_FIELD_OPACITY_GAIN: f32 = DEFAULT_3D_FIELD_OPACITY_GAIN;
pub(crate) const GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET: f32 = 6.0;
pub(crate) const DEFAULT_3D_MESH_FIELD_SCALE: f32 = 0.72;
pub(crate) const UV_TORUS_FIELD_SCALE: f32 = DEFAULT_3D_MESH_FIELD_SCALE;
pub(crate) const UV_TORUS_RENDER_TRAINING_SCALE: f32 = 0.54;
pub(crate) const TEAPOT_RENDER_TRAINING_SCALE: f32 = DEFAULT_3D_MESH_FIELD_SCALE;
pub(crate) const TEAPOT_FIELD_MOTION_GAIN: f32 = 1.0;
pub(crate) const TEAPOT_FIELD_COLOR_GAIN: f32 = 0.4;
pub(crate) const LOCAL_TORUS_MOTION_GAIN: f32 = 0.0;
pub(crate) const LOCAL_TEAPOT_MOTION_GAIN: f32 = 0.025;
pub(crate) const LOCAL_TORUS_COLOR_GAIN: f32 = 0.12;
pub(crate) const LOCAL_TEAPOT_COLOR_GAIN: f32 = 0.20;
pub(crate) const LOCAL_GROWTH_EXPANSION_GAIN: f32 = 0.05;
pub(crate) const LOCAL_GROWTH_OPACITY_GAIN: f32 = 2.0;
pub(crate) const LOCAL_GROWTH_MATERIAL_OPACITY_GAIN: f32 = 0.16;
pub(crate) const LOCAL_GROWTH_ACTIVE_MATERIAL_GAIN: f32 = 0.50;
pub(crate) const LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN: f32 = 0.05;
pub(crate) const LOCAL_GROWTH_PHASE_GAIN: f32 = 0.20;
pub(crate) const LOCAL_GROWTH_PHASE_LIVENESS_GAIN: f32 = 0.04;
pub(crate) const LOCAL_GROWTH_PHASE_MATERIAL_GAIN: f32 = 0.35;
pub(crate) const LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN: f32 = 1.0;
pub(crate) const LOCAL_GROWTH_VELOCITY_DAMPING_GAIN: f32 = 0.15;
pub(crate) const DIRECT_GROWTH_PHASE_GAIN_FRACTION: f32 = 0.25;
pub(crate) const DIRECT_GROWTH_LIVENESS_PHASE_MEMORY_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_MOTION_MEMORY_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_RESIDUAL_VELOCITY_GAIN_FRACTION: f32 = 0.75;
pub(crate) const DIRECT_GROWTH_SPATIAL_MOTION_RMS_TARGET_FRACTION: f32 = 0.80;
pub(crate) const DIRECT_GROWTH_MESH_MOTION_LIVENESS_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_TARGET_COVERAGE_LIVENESS_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_MATERIAL_COVERAGE_LIVENESS_GAIN_FRACTION: f32 = 0.35;
pub(crate) const DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_GAIN_FRACTION: f32 = 0.35;
pub(crate) const DIRECT_GROWTH_MATERIAL_COVERAGE_MOTION_MEMORY_GAIN_FRACTION: f32 = 0.35;
pub(crate) const DIRECT_GROWTH_MATERIAL_SURFACE_MOTION_COVERAGE_GAIN_FRACTION: f32 = 0.35;
pub(crate) const DIRECT_GROWTH_MATERIAL_SURFACE_MOTION_RMS_TARGET_FRACTION: f32 = 0.75;
pub(crate) const DIRECT_GROWTH_MATERIAL_COVERAGE_MATERIALIZATION_GAIN_FRACTION: f32 = 4.0;
pub(crate) const DIRECT_GROWTH_TEMPORAL_MATERIALIZATION_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_ACTIVE_SURFACE_MATERIALIZATION_GAIN_FRACTION: f32 = 0.75;
pub(crate) const DIRECT_GROWTH_STRICT_SURFACE_MATERIALIZATION_GAIN_FRACTION: f32 = 2.0;
pub(crate) const DIRECT_GROWTH_SURFACE_COLOR_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_EXTENT_FRONT_MOTION_GAIN_FRACTION: f32 = 0.75;
pub(crate) const DIRECT_GROWTH_EXTENT_MOTION_MEMORY_GAIN_FRACTION: f32 = 0.40;
pub(crate) const DIRECT_GROWTH_EXTENT_FRONT_LIVENESS_GAIN_FRACTION: f32 = 0.50;
pub(crate) const DIRECT_GROWTH_TEMPORAL_EXTENT_MOTION_GAIN_FRACTION: f32 = 0.25;
pub(crate) const DIRECT_GROWTH_LOCAL_FRONT_LIVENESS_FLOOR: f32 = 0.25;
pub(crate) const DIRECT_ROLLOUT_GRADIENT_ROW_NORMALIZATION_EXPONENT: f32 = 0.75;
pub(crate) const DIRECT_GROWTH_MATERIAL_OUTPUT_RMS_CAP_MULTIPLIER: f32 = 2.0;
#[cfg(test)]
pub(crate) const LOCAL_GROWTH_FRONT_OPACITY_GAIN: f32 = 0.18;
#[cfg(test)]
pub(crate) const LOCAL_GROWTH_FRONT_RADIUS: f32 = 0.22;
#[cfg(test)]
pub(crate) const LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE: f32 = 0.35;
#[cfg(test)]
pub(crate) const LOCAL_GROWTH_EXTENT_GAIN: f32 = 0.10;
pub(crate) const LOCAL_GROWTH_COORDINATE_GAIN: f32 = 0.10;
pub(crate) const LOCAL_GROWTH_ORIENTATION_GAIN: f32 = 0.12;
pub(crate) const LOCAL_GROWTH_SIGNED_DISTANCE_GAIN: f32 = 0.08;
pub(crate) const UV_TORUS_TARGET_RINGS: usize = 96;
pub(crate) const UV_TORUS_TARGET_TUBES: usize = 64;
pub(crate) const TORUS_ANGULAR_COVERAGE_RINGS: usize = 24;
pub(crate) const TORUS_ANGULAR_COVERAGE_TUBES: usize = 16;
pub(crate) const UV_TORUS_TARGET_SOURCE: &str = "uv-torus-3d:mesh-ovoxel-oriented-growth";
pub(crate) const UV_TORUS_POSITION_FIELD_TARGET_SOURCE: &str =
    "uv-torus-3d:neutral-seed-position-field-growth";
pub(crate) const UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE: &str =
    "uv-torus-3d:neutral-seed-rollout-position-field-growth";
pub(crate) const UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE: &str =
    "uv-torus-3d:mesh-ovoxel-oriented-seed-frame-morphogen-baseline";
pub(crate) const UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE: &str =
    "uv-torus-3d:rollout-local-mesh-objective-morphogen";
pub(crate) const UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-random-ball-rollout-ablation";
pub(crate) const UV_TORUS_CONDITIONLESS_COMPACT_NOSCAFFOLD_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-random-ball-no-scaffold-rollout-ablation";
pub(crate) const UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-substrate-rollout-ablation";
pub(crate) const UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-substrate-no-scaffold-rollout-ablation";
pub(crate) const TEAPOT_POSITION_FIELD_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-neutral-seed-position-field-growth";
pub(crate) const TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-neutral-seed-rollout-position-field-growth";
pub(crate) const TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-seed-frame-morphogen-baseline";
pub(crate) const TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-rollout-local-mesh-objective-morphogen";
pub(crate) const TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-random-ball-rollout-ablation";
pub(crate) const TEAPOT_CONDITIONLESS_COMPACT_NOSCAFFOLD_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-random-ball-no-scaffold-rollout-ablation";
pub(crate) const TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-substrate-rollout-ablation";
pub(crate) const TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-substrate-no-scaffold-rollout-ablation";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeshTargetTrainingProfile {
    pub(crate) target: MeshTargetArg,
    pub(crate) field_scale: f32,
    pub(crate) render_training_scale: f32,
    pub(crate) field_seed_mode: ParticleSeed,
    pub(crate) conditionless_local_seed_mode: ParticleSeed,
    pub(crate) field_motion_gain: f32,
    pub(crate) field_color_gain: f32,
    pub(crate) local_motion_gain: f32,
    pub(crate) local_color_gain: f32,
    pub(crate) conditionless_local_target_source: &'static str,
    pub(crate) lineage_marker: &'static str,
}

pub(crate) fn mesh_target_training_profile(target: MeshTargetArg) -> MeshTargetTrainingProfile {
    match target {
        MeshTargetArg::Torus => MeshTargetTrainingProfile {
            target,
            field_scale: UV_TORUS_FIELD_SCALE,
            render_training_scale: UV_TORUS_RENDER_TRAINING_SCALE,
            field_seed_mode: ParticleSeed::TorusFieldDense3d,
            conditionless_local_seed_mode: ParticleSeed::TorusLocalSubstrateGrowth3d,
            field_motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
            field_color_gain: UV_TORUS_FIELD_COLOR_GAIN,
            local_motion_gain: LOCAL_TORUS_MOTION_GAIN,
            local_color_gain: LOCAL_TORUS_COLOR_GAIN,
            conditionless_local_target_source: UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE,
            lineage_marker: "uv-torus-3d",
        },
        MeshTargetArg::Teapot => MeshTargetTrainingProfile {
            target,
            field_scale: DEFAULT_3D_MESH_FIELD_SCALE,
            render_training_scale: TEAPOT_RENDER_TRAINING_SCALE,
            field_seed_mode: ParticleSeed::TeapotFieldDense3d,
            conditionless_local_seed_mode: ParticleSeed::TeapotLocalSubstrateGrowth3d,
            field_motion_gain: TEAPOT_FIELD_MOTION_GAIN,
            field_color_gain: TEAPOT_FIELD_COLOR_GAIN,
            local_motion_gain: LOCAL_TEAPOT_MOTION_GAIN,
            local_color_gain: LOCAL_TEAPOT_COLOR_GAIN,
            conditionless_local_target_source: TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
            lineage_marker: "utah-teapot-2026",
        },
    }
}

pub(crate) fn uv_torus_mesh_target(scale: f32) -> TriangleMeshTarget {
    TriangleMeshTarget::torus(
        scale.max(1.0e-4),
        scale.max(1.0e-4) * UV_TORUS_MINOR_RATIO,
        UV_TORUS_TARGET_RINGS,
        UV_TORUS_TARGET_TUBES,
    )
    .expect("uv torus target mesh generation should be valid")
}

pub(crate) fn utah_teapot_mesh_target(scale: f32) -> TriangleMeshTarget {
    TriangleMeshTarget::utah_teapot(scale.max(1.0e-4))
        .expect("canonical Utah Teapot target mesh should be valid")
}

pub(crate) fn mesh_target_for_arg(target: MeshTargetArg, scale: f32) -> TriangleMeshTarget {
    match target {
        MeshTargetArg::Torus => uv_torus_mesh_target(scale),
        MeshTargetArg::Teapot => utah_teapot_mesh_target(scale),
    }
}

pub(crate) fn mesh_target_render_training_seed_scale(target: MeshTargetArg) -> f32 {
    mesh_target_training_profile(target).render_training_scale
}

pub(crate) fn mesh_conditionless_local_target_source(target: MeshTargetArg) -> &'static str {
    mesh_target_training_profile(target).conditionless_local_target_source
}

pub(crate) fn mesh_target_lineage_marker(target: MeshTargetArg) -> &'static str {
    mesh_target_training_profile(target).lineage_marker
}

pub(crate) fn mesh_conditionless_local_target_source_for_seed(
    target: MeshTargetArg,
    seed_mode: ParticleSeed,
) -> &'static str {
    match (target, seed_mode) {
        (MeshTargetArg::Torus, ParticleSeed::TorusGrowth3d) => {
            UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE
        }
        (MeshTargetArg::Teapot, ParticleSeed::TeapotGrowth3d) => {
            TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE
        }
        (MeshTargetArg::Torus, ParticleSeed::TorusLocalGrowth3d) => {
            UV_TORUS_CONDITIONLESS_COMPACT_NOSCAFFOLD_TARGET_SOURCE
        }
        (MeshTargetArg::Teapot, ParticleSeed::TeapotLocalGrowth3d) => {
            TEAPOT_CONDITIONLESS_COMPACT_NOSCAFFOLD_TARGET_SOURCE
        }
        (MeshTargetArg::Torus, ParticleSeed::TorusLocalSubstrateGrowth3d) => {
            UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
        }
        (MeshTargetArg::Teapot, ParticleSeed::TeapotLocalSubstrateGrowth3d) => {
            TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
        }
        _ => mesh_conditionless_local_target_source(target),
    }
}

pub(crate) fn mesh_target_motion_gain(target: MeshTargetArg) -> f32 {
    mesh_target_training_profile(target).local_motion_gain
}

pub(crate) fn mesh_target_color_gain(target: MeshTargetArg) -> f32 {
    mesh_target_training_profile(target).local_color_gain
}

pub(crate) fn conditionless_local_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    mesh_target_training_profile(target).conditionless_local_seed_mode
}

pub(crate) fn conditionless_local_rollout_cases(
    target: MeshTargetArg,
    seed_scale: f32,
    rollout_particles: usize,
) -> [MeshRolloutCaseConfig; 3] {
    let particles = rollout_particles.max(128);
    let seed_mode = conditionless_local_seed_mode(target);
    [
        MeshRolloutCaseConfig {
            particle_count: particles,
            steps: 64,
            seed: 0x010c_a101,
            seed_scale: (seed_scale * 0.5).max(1.0e-4),
            seed_mode,
        },
        MeshRolloutCaseConfig {
            particle_count: particles,
            steps: 64,
            seed: 0x010c_a102,
            seed_scale: seed_scale.max(1.0e-4),
            seed_mode,
        },
        MeshRolloutCaseConfig {
            particle_count: particles,
            steps: 64,
            seed: 0x010c_a103,
            seed_scale: (seed_scale * 1.5).max(1.0e-4),
            seed_mode,
        },
    ]
}
