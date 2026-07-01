use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(feature = "gpu_wgpu")]
use burn_automata::kernels::build_hashgrid;
use burn_automata::{
    AutomataPreset, BpkModelManifest, FeatureBatchConfig, NpaConfig, NpaModel, NpaWeights,
    ParticleSeed, RolloutConfig, RolloutSupervisionConfig, SupervisedBatch, SupervisedTarget,
    feature_supervised_batch, import_model,
    kernels::perceive_adjoint_with_options,
    kernels::{PerceptionOptions, euler_step, perceive_with_options},
    render_loss::{
        MultiViewRenderLossReport, RenderLossConfig, RenderViewLossReport,
        mesh_multiview_render_loss_from_trace,
        mesh_multiview_render_position_gradient_for_rows_from_trace,
    },
    rollout::{
        GROWTH_3D_INACTIVE_OPACITY_LOGIT, GROWTH_3D_LIVENESS_CHANNEL,
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_MOTION_GAIN, UV_TORUS_NORMAL_STATE_OFFSET, UV_TORUS_OPACITY_GROWTH_DELTA,
        UV_TORUS_RESIDUAL_DECAY, UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
        growth_3d_material_opacity_channel, growth_3d_seed_radius, seed_particles_scaled,
        utah_teapot_dense_seed_position, utah_teapot_tail_state_color,
        uv_torus_continuous_surface_position, uv_torus_continuous_volume_position,
        uv_torus_dense_seed_position, uv_torus_orientation_state_available,
        uv_torus_position_color, uv_torus_sample, uv_torus_surface_error,
        uv_torus_tail_state_color, uv_torus_tail_state_to_rgb,
    },
    rollout_supervised_batch_from_model, run_rollout,
    target_geometry::{TriangleMeshTarget, dot3},
    training::{
        SgdConfig, SupervisedGradients, TrainingHistoryEntry, TrainingRunConfig, TrainingRunReport,
        apply_sgd_gradients, mlp_backward_from_output_gradients, run_supervised_training,
    },
};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

const GROWTH_3D_MAX_FINAL_OPACITY_LOGIT: f32 = 24.0;
const GROWTH_3D_SURFACE_MAX_DISTANCE: f32 = 0.36;
const RENDER_SELECTION_BAD_SCORE: f32 = 1.0e9;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Infer {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 1024)]
        particles: usize,
        #[arg(long, default_value_t = 0.5)]
        update_prob: f32,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value = "auto")]
        neighbor_mode: NeighborModeArg,
        #[arg(long)]
        bucket_capacity: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "/tmp/burn_automata_rollout.json")]
        output: PathBuf,
    },
    Train {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value = "/tmp/burn_automata_training_report.json")]
        output: PathBuf,
        #[arg(long)]
        model_output: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long, default_value_t = 1)]
        report_interval: usize,
        #[arg(long)]
        target_model: Option<PathBuf>,
        #[arg(long)]
        target_seed: Option<u64>,
        #[arg(long)]
        zero_update: bool,
        #[arg(long, default_value_t = 7)]
        student_seed: u64,
        #[arg(long, default_value = "rollout")]
        batch_source: TrainingBatchArg,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 16)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 1.0)]
        rollout_update_prob: f32,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
    },
    TrainTorus3d {
        #[arg(long, default_value = "artifacts/legacy_uv_torus_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_uv_torus_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 512)]
        steps: usize,
    },
    TrainTorusMorphogen3d {
        #[arg(long, default_value = "artifacts/legacy_uv_torus_morphogen_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_uv_torus_morphogen_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 96)]
        steps: usize,
        #[arg(long, default_value = "rollout-local")]
        training_mode: MeshTrainingModeArg,
        #[arg(long, default_value_t = 2048)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 128)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
    },
    TrainTeapotMorphogen3d {
        #[arg(long, default_value = "artifacts/legacy_teapot_morphogen_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/legacy_teapot_morphogen_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value = "rollout-local")]
        training_mode: MeshTrainingModeArg,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 64)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
    },
    #[command(name = "ablate-local-3d", alias = "ablate-local3d")]
    AblateLocal3d {
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "/tmp/burn_automata_conditionless_local_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/conditionless_local_3d_ablation_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        rows: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 1024)]
        rollout_particles: usize,
        #[arg(long, default_value_t = 64)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 4)]
        temporal_samples: usize,
        #[arg(long, default_value_t = 4)]
        training_rounds: usize,
        #[arg(long, default_value_t = UV_TORUS_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 0x10ca_13d)]
        student_seed: u64,
        #[arg(long, default_value_t = 6.0e-5)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.08)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f32,
        #[arg(long)]
        motion_gain: Option<f32>,
        #[arg(long, default_value_t = 0.08)]
        max_update_norm: f32,
        #[arg(long, default_value_t = 0.0)]
        density_gain: f32,
        #[arg(long, default_value_t = LOCAL_GROWTH_EXPANSION_GAIN)]
        expansion_gain: f32,
        #[arg(long, default_value_t = 0.35)]
        coverage_gain: f32,
        #[arg(long, default_value_t = 4096)]
        coverage_samples: usize,
        #[arg(long, default_value = "sliced-ot")]
        coverage_mode: CoverageUpdateModeArg,
        #[arg(long, default_value_t = 0.0)]
        coverage_softness: f32,
        #[arg(long, default_value_t = 0.2)]
        coverage_repulsion_gain: f32,
        #[arg(long)]
        coverage_gap_gain: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        coverage_repulsion_radius: f32,
        #[arg(long, default_value_t = 0.0)]
        coverage_normal_weight: f32,
        #[arg(long, default_value_t = 0.2)]
        extent_gain: f32,
        #[arg(long)]
        color_gain: Option<f32>,
        #[arg(long, default_value_t = 0.5)]
        aux_state_gain: f32,
        #[arg(long, default_value_t = 0.02)]
        opacity_gain: f32,
        #[arg(long, default_value_t = 0.06)]
        front_opacity_gain: f32,
        #[arg(long, default_value_t = 0.24)]
        front_radius: f32,
        #[arg(long, default_value_t = 0.2)]
        front_max_opacity_update: f32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        front_motion_gate: bool,
        #[arg(long, default_value_t = false, action = ArgAction::Set)]
        preserve_opacity_update: bool,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "render-loss-3d", alias = "render3d")]
    RenderLoss3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long, default_value = "artifacts/render_loss_3d_report.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 0x51a7_3d)]
        seed: u64,
        #[arg(long = "extra-seed", value_delimiter = ',')]
        extra_seeds: Vec<u64>,
        #[arg(long, default_value_t = UV_TORUS_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "validate-growth3d", alias = "validate-3d-growth")]
    ValidateGrowth3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long, default_value = "artifacts/growth_3d_validation_report.json")]
        output: PathBuf,
        #[arg(long, default_value_t = 256)]
        particles: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long = "extra-seed", value_delimiter = ',')]
        extra_seeds: Vec<u64>,
        #[arg(long, default_value_t = UV_TORUS_FIELD_SCALE)]
        seed_scale: f32,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long, default_value = "strict")]
        gate: Growth3dValidationGateArg,
        #[arg(long)]
        fail_on_validation: bool,
    },
    #[command(name = "retime-growth3d", alias = "retime-growth-3d")]
    RetimeGrowth3d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = LOCAL_GROWTH_OPACITY_GAIN)]
        front_gain: f32,
        #[arg(long)]
        hidden: Option<usize>,
        #[arg(long)]
        skip_front_retime: bool,
        #[arg(long)]
        active_opacity_gain: Option<f32>,
        #[arg(long)]
        active_opacity_hidden: Option<usize>,
        #[arg(long)]
        opacity_bias: Option<f32>,
        #[arg(long)]
        material_opacity_bias: Option<f32>,
        #[arg(long)]
        alpha: Option<f32>,
    },
    #[command(name = "train-render3d", alias = "train-render-3d")]
    TrainRender3d {
        #[arg(long, default_value = "torus")]
        target: MeshTargetArg,
        #[arg(long)]
        base_model: Option<PathBuf>,
        #[arg(long, default_value = "artifacts/render_trained_3d.bpk")]
        model_output: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/render_trained_3d_training_report.json"
        )]
        report_output: PathBuf,
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 32)]
        supervised_steps_per_round: usize,
        #[arg(long, default_value_t = 512)]
        particles: usize,
        #[arg(long, default_value_t = 32)]
        rollout_steps: usize,
        #[arg(long, default_value_t = 64)]
        gradient_particles: usize,
        #[arg(long, default_value = "analytic")]
        gradient_mode: RenderGradientModeArg,
        #[arg(long, default_value_t = 1.0e-3)]
        finite_diff_eps: f32,
        #[arg(long, default_value_t = 0.35)]
        motion_gain: f32,
        #[arg(long, default_value_t = 0.05)]
        perception_position_gain: f32,
        #[arg(long, default_value_t = 1.0)]
        max_update_norm: f32,
        #[arg(long, default_value_t = true)]
        trajectory_supervision: bool,
        #[arg(long, default_value_t = 0.0)]
        trajectory_render_gain: f32,
        #[arg(long, default_value_t = 0)]
        trajectory_render_samples: usize,
        #[arg(long, default_value_t = 0.0)]
        coverage_gain: f32,
        #[arg(long, default_value_t = 0)]
        coverage_samples: usize,
        #[arg(long, default_value = "hard-nearest")]
        coverage_mode: CoverageUpdateModeArg,
        #[arg(long, default_value_t = 0.0)]
        coverage_softness: f32,
        #[arg(long, default_value_t = 0.0)]
        coverage_repulsion_gain: f32,
        #[arg(long)]
        coverage_gap_gain: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        coverage_repulsion_radius: f32,
        #[arg(long, default_value_t = 0.0)]
        coverage_normal_weight: f32,
        #[arg(long)]
        full_coverage_adjoint: bool,
        #[arg(long, default_value_t = 0.0)]
        surface_gain: f32,
        #[arg(long, default_value_t = 0.0)]
        opacity_gain: f32,
        #[arg(long, default_value_t = 0.05)]
        max_opacity_update: f32,
        #[arg(long, default_value_t = 5.0e-4)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.25)]
        grad_clip_norm: f32,
        #[arg(long)]
        direct_line_search: bool,
        #[arg(long, value_delimiter = ',', default_value = "0.25,0.5,1,2,4,8,16,32")]
        direct_line_search_scales: Vec<f32>,
        #[arg(long)]
        direct_material_output_only: bool,
        #[arg(long, default_value = "direct-rollout")]
        training_backend: RenderTrainingBackendArg,
        #[arg(long)]
        direct_selection_seed_training: bool,
        #[arg(long, default_value_t = 0.72)]
        seed_scale: f32,
        #[arg(long)]
        seed_mode: Option<SeedModeArg>,
        #[arg(long, default_value_t = 0x51a7_3d)]
        selection_seed: u64,
        #[arg(long = "extra-selection-seed", value_delimiter = ',')]
        extra_selection_seeds: Vec<u64>,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
        #[arg(long, default_value_t = 0)]
        target_samples: usize,
        #[arg(long, default_value_t = 2.5)]
        sigma: f32,
        #[arg(long)]
        world_scale: Option<f32>,
        #[arg(long, default_value_t = 0.0)]
        render_opacity_logit_bias: f32,
        #[arg(long, default_value_t = 1.0)]
        density_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        color_weight: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_weight: f32,
        #[arg(long)]
        fail_on_validation: bool,
    },
    Import {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Bench {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value_t = 4096)]
        particles: usize,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        repeats: usize,
        #[arg(long, default_value_t = 1.0)]
        update_prob: f32,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value = "auto")]
        neighbor_mode: NeighborModeArg,
        #[arg(long)]
        bucket_capacity: Option<usize>,
        #[arg(long)]
        profile: bool,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        normalize_seed_scale: bool,
        #[arg(long, alias = "no-normalize-seed-scale")]
        fixed_eps: bool,
        #[arg(long)]
        reference_seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "seed")]
        geometry: BenchGeometryArg,
        #[arg(long)]
        gaussian: bool,
    },
    #[command(name = "bench-spatial", alias = "spatial-bench")]
    BenchSpatial {
        #[arg(long, default_value = "growing-3d-gs")]
        preset: PresetArg,
        #[arg(long, default_value_t = 8192)]
        particles: usize,
        #[arg(long)]
        seed_scale: Option<f32>,
        #[arg(long)]
        normalize_seed_scale: bool,
        #[arg(long)]
        fixed_eps: bool,
        #[arg(long)]
        reference_seed_scale: Option<f32>,
        #[arg(long, default_value = "uniform-circle")]
        seed_mode: SeedModeArg,
        #[arg(long, default_value = "seed")]
        geometry: BenchGeometryArg,
        #[arg(long, default_value = "all")]
        strategy: SpatialStrategyArg,
        #[arg(long, default_value_t = 16)]
        bvh_leaf_size: usize,
        #[arg(long, default_value = "2,2,1")]
        tile_size: String,
    },
    Manifest {
        #[arg(long, default_value = "growing-2d")]
        preset: PresetArg,
        #[arg(long, default_value = "/tmp/burn_automata_seed_model.bpk")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PresetArg {
    #[value(name = "growing-2d", alias = "growing2d")]
    Growing2d,
    #[value(name = "texture-2d", alias = "texture2d")]
    Texture2d,
    #[value(name = "growing-3d-gs", alias = "growing3dgs", alias = "growing-3dgs")]
    Growing3dgs,
    #[value(name = "point-mnist", alias = "pointmnist")]
    PointMnist,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum NeighborModeArg {
    Auto,
    #[value(name = "linked-list", alias = "linked")]
    LinkedList,
    #[value(name = "fixed-buckets", alias = "buckets")]
    FixedBuckets,
    #[value(name = "tiled-fixed-buckets", alias = "tiled-buckets", alias = "tiled")]
    TiledFixedBuckets,
    #[value(name = "sorted-cells", alias = "sorted")]
    SortedCells,
    #[value(name = "bvh", alias = "cpu-bvh")]
    Bvh,
    #[value(name = "gpu-bvh", alias = "fixed-gpu-bvh")]
    GpuBvh,
    #[value(name = "gpu-lbvh", alias = "lbvh", alias = "sorted-gpu-bvh")]
    GpuLbvh,
    #[value(name = "gpu-morton-lbvh", alias = "morton-lbvh")]
    GpuMortonLbvh,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SpatialStrategyArg {
    All,
    #[value(name = "hash-grid", alias = "hashgrid", alias = "grid")]
    HashGrid,
    #[value(name = "tile-blocks", alias = "tiles", alias = "tile")]
    TileBlocks,
    Bvh,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum TrainingBatchArg {
    Rollout,
    Features,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
enum MeshTrainingModeArg {
    #[value(name = "position-field", alias = "field")]
    PositionField,
    #[value(
        name = "rollout-position-field",
        alias = "rollout-field",
        alias = "field-rollout"
    )]
    RolloutPositionField,
    #[value(name = "rollout-local", alias = "rollout")]
    RolloutLocal,
    #[value(name = "projection-baseline", alias = "baseline")]
    ProjectionBaseline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
enum MeshTargetArg {
    Torus,
    Teapot,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum RenderGradientModeArg {
    Analytic,
    #[value(name = "finite-diff", alias = "finite_difference")]
    FiniteDiff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
enum RenderTrainingBackendArg {
    #[value(name = "direct-rollout", alias = "direct")]
    DirectRollout,
    #[value(name = "proxy", alias = "supervised-proxy")]
    Proxy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
enum CoverageUpdateModeArg {
    #[value(name = "hard-nearest", alias = "nearest", alias = "hard")]
    HardNearest,
    #[value(name = "soft-chamfer", alias = "soft", alias = "chamfer")]
    SoftChamfer,
    #[value(name = "gap-farthest", alias = "gap", alias = "farthest")]
    GapFarthest,
    #[value(name = "sliced-ot", alias = "sliced", alias = "ot")]
    SlicedOt,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
enum Growth3dValidationGateArg {
    Strict,
    #[value(name = "catalog-sanity", alias = "catalog")]
    CatalogSanity,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedModeArg {
    #[value(name = "gaussian")]
    Gaussian,
    #[value(name = "uniform")]
    Uniform,
    #[value(name = "uniform-circle", alias = "circle")]
    UniformCircle,
    #[value(name = "uv-torus-3d", alias = "torus")]
    UvTorus3d,
    #[value(
        name = "uv-torus-dense-3d",
        alias = "torus-dense",
        alias = "dense-torus"
    )]
    UvTorusDense3d,
    #[value(
        name = "torus-field-dense-3d",
        alias = "torus-field",
        alias = "field-torus"
    )]
    TorusFieldDense3d,
    #[value(
        name = "teapot-field-dense-3d",
        alias = "teapot-field",
        alias = "field-teapot"
    )]
    TeapotFieldDense3d,
    #[value(
        name = "torus-growth-3d",
        alias = "torus-growth",
        alias = "growth-torus"
    )]
    TorusGrowth3d,
    #[value(
        name = "teapot-growth-3d",
        alias = "teapot-growth",
        alias = "growth-teapot"
    )]
    TeapotGrowth3d,
    #[value(
        name = "torus-substrate-growth-3d",
        alias = "torus-substrate",
        alias = "substrate-torus"
    )]
    TorusSubstrateGrowth3d,
    #[value(
        name = "teapot-substrate-growth-3d",
        alias = "teapot-substrate",
        alias = "substrate-teapot"
    )]
    TeapotSubstrateGrowth3d,
    #[value(
        name = "torus-morphogen-dense-3d",
        alias = "torus-morphogen",
        alias = "morphogen-torus"
    )]
    TorusMorphogenDense3d,
    #[value(
        name = "teapot-morphogen-dense-3d",
        alias = "teapot-morphogen",
        alias = "morphogen-teapot"
    )]
    TeapotMorphogenDense3d,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchGeometryArg {
    Seed,
    Dense,
    Uniform,
    Line,
    Ring,
    Plane,
    Shell,
    Torus,
    #[value(name = "shifted-dense", alias = "dense-shifted")]
    ShiftedDense,
    #[value(name = "shifted-uniform", alias = "uniform-shifted")]
    ShiftedUniform,
}

impl From<PresetArg> for AutomataPreset {
    fn from(value: PresetArg) -> Self {
        match value {
            PresetArg::Growing2d => Self::Growing2d,
            PresetArg::Texture2d => Self::Texture2d,
            PresetArg::Growing3dgs => Self::Growing3dGs,
            PresetArg::PointMnist => Self::PointMnist,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Infer {
            preset,
            steps,
            particles,
            update_prob,
            model,
            gpu,
            neighbor_mode,
            bucket_capacity,
            seed,
            seed_scale,
            seed_mode,
            output,
        } => {
            #[cfg(not(feature = "gpu_wgpu"))]
            let _ = (neighbor_mode, bucket_capacity);
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let (config, grid) = NpaConfig::for_preset(preset);
            let (model, grid) = if let Some(path) = model {
                let manifest = burn_automata::import::load_manifest(path)?;
                let grid = manifest.hashgrid.clone();
                (manifest.into_model(), grid)
            } else {
                (NpaModel::seeded(config, 42), grid)
            };
            let cfg = RolloutConfig {
                steps,
                particle_count: particles,
                update_prob,
                seed: seed.unwrap_or_else(|| RolloutConfig::default().seed),
                seed_scale,
                ..RolloutConfig::default()
            };
            let trace = if gpu {
                #[cfg(feature = "gpu_wgpu")]
                {
                    gpu_rollout_trace(
                        &model,
                        &grid,
                        &cfg,
                        seed_mode.into(),
                        wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                    )?
                }
                #[cfg(not(feature = "gpu_wgpu"))]
                {
                    return Err(std::io::Error::other(
                        "infer --gpu requires building burn_automata with --features gpu_wgpu",
                    )
                    .into());
                }
            } else {
                run_rollout(&model, &grid, &cfg, seed_mode.into())?
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&trace)?)?;
            println!("wrote {}", output.display());
        }
        Command::Train {
            preset,
            output,
            model_output,
            rows,
            steps,
            learning_rate,
            grad_clip_norm,
            weight_decay,
            report_interval,
            target_model,
            target_seed,
            zero_update,
            student_seed,
            batch_source,
            rollout_particles,
            rollout_steps,
            rollouts,
            rollout_update_prob,
            seed_scale,
            seed_mode,
        } => {
            let preset: AutomataPreset = preset.into();
            let (preset_config, preset_grid) = NpaConfig::for_preset(preset);
            if target_model.is_some() && target_seed.is_some() {
                return Err(std::io::Error::other(
                    "--target-model and --target-seed are mutually exclusive",
                )
                .into());
            }
            if zero_update && (target_model.is_some() || target_seed.is_some()) {
                return Err(std::io::Error::other(
                    "--zero-update cannot be combined with --target-model or --target-seed",
                )
                .into());
            }
            let (config, hashgrid, target_source, teacher) = if let Some(path) = target_model {
                let manifest = burn_automata::import::load_manifest(&path)?;
                (
                    manifest.config.clone(),
                    manifest.hashgrid.clone(),
                    format!("model:{}", path.display()),
                    Some(manifest.into_model()),
                )
            } else {
                let target_seed = default_train_target_seed(preset, target_seed, zero_update);
                let teacher = target_seed.map(|seed| NpaModel::seeded(preset_config.clone(), seed));
                let target_source = train_target_source(preset, target_seed, zero_update);
                (preset_config, preset_grid, target_source, teacher)
            };
            let mut model = NpaModel::seeded(config.clone(), student_seed);
            let target = if let Some(teacher) = teacher.as_ref() {
                SupervisedTarget::Teacher(teacher)
            } else {
                SupervisedTarget::ZeroUpdate
            };
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let rollout_report = match batch_source {
                TrainingBatchArg::Features => None,
                TrainingBatchArg::Rollout => Some(CliRolloutSupervisionReport {
                    particle_count: rollout_particles,
                    rollout_steps,
                    rollouts,
                    temporal_samples: 1,
                    update_prob: rollout_update_prob,
                    seed_scale,
                    seed_mode,
                    motion_gain: None,
                    max_update_norm: None,
                    density_gain: None,
                    expansion_gain: None,
                    coverage_gain: None,
                    coverage_samples: None,
                    coverage_mode: None,
                    coverage_softness: None,
                    coverage_repulsion_gain: None,
                    coverage_gap_gain: None,
                    coverage_repulsion_radius: None,
                    coverage_normal_weight: None,
                    extent_gain: None,
                    color_gain: None,
                    aux_state_gain: None,
                    opacity_gain: None,
                    front_opacity_gain: None,
                    front_radius: None,
                    front_max_opacity_update: None,
                    front_motion_gate: None,
                    preserve_opacity_update: None,
                }),
            };
            let batch = match batch_source {
                TrainingBatchArg::Features => feature_supervised_batch(
                    &model,
                    target,
                    FeatureBatchConfig {
                        rows,
                        seed: student_seed,
                        ..FeatureBatchConfig::default()
                    },
                )?,
                TrainingBatchArg::Rollout => {
                    let rollout_model = teacher.as_ref().unwrap_or(&model);
                    rollout_supervised_batch_from_model(
                        &model,
                        rollout_model,
                        &hashgrid,
                        target,
                        RolloutSupervisionConfig {
                            max_rows: rows,
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            update_prob: rollout_update_prob,
                            seed: student_seed,
                            seed_scale,
                            seed_mode,
                            ..RolloutSupervisionConfig::default()
                        },
                    )?
                }
            };
            let cfg = SgdConfig {
                learning_rate,
                weight_decay,
                grad_clip_norm,
            };
            let report = run_supervised_training(
                &mut model,
                &batch,
                TrainingRunConfig {
                    steps,
                    report_interval,
                    sgd: cfg,
                },
            )?;
            if let Some(path) = &model_output {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let training_source = training_source_with_batch(batch_source, &target_source);
                let manifest = BpkModelManifest::from_model(
                    &model,
                    hashgrid,
                    Some(format!("trained-rust:{training_source}")),
                );
                burn_automata::import::save_manifest(path, &manifest)?;
            }
            let training_source = training_source_with_batch(batch_source, &target_source);
            let output_report = CliTrainingReport {
                preset,
                target_source: training_source,
                student_seed,
                sgd: cfg,
                report,
                model_output: model_output.as_ref().map(|path| path.display().to_string()),
                batch_source,
                rollout_supervision: rollout_report,
                mesh_rollout: None,
                render_loss: None,
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&output_report)?)?;
            println!(
                "wrote {} target={} final_loss={:.6} best_loss={:.6}",
                output.display(),
                output_report.target_source,
                output_report.report.final_loss,
                output_report.report.best_loss
            );
        }
        Command::TrainTorus3d {
            model_output,
            report_output,
            rows,
            steps,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "train-torus3d")?;
            let config = NpaConfig::torus_field_3dgs();
            let hashgrid = burn_automata::kernels::HashGridConfig::growing_3dgs();
            let mut model = torus_field_model(config.clone())?;
            let batch = torus_field_supervised_batch(&config, rows);
            let sgd = SgdConfig {
                learning_rate: 0.002,
                grad_clip_norm: 1.0,
                ..SgdConfig::default()
            };
            let report = run_supervised_training(
                &mut model,
                &batch,
                TrainingRunConfig {
                    steps,
                    report_interval: steps.max(1),
                    sgd,
                },
            )?;
            if let Some(parent) = model_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Some(parent) = report_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid.clone(),
                Some(format!("trained-rust:{UV_TORUS_TARGET_SOURCE}")),
            );
            burn_automata::import::save_manifest(&model_output, &manifest)?;
            let loaded = burn_automata::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let robustness = torus_robustness_report(&loaded_model, &loaded_hashgrid)?;
            let output_report = CliTorusTrainingReport {
                preset: AutomataPreset::Growing3dGs,
                target_source: UV_TORUS_TARGET_SOURCE.to_string(),
                student_seed: 0,
                sgd,
                report,
                model_output: Some(model_output.display().to_string()),
                robustness,
                batch_source: TrainingBatchArg::Features,
                training_mode: MeshTrainingModeArg::ProjectionBaseline,
                rollout_supervision: None,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            println!(
                "wrote {} and {} final_loss={:.6} robust={}",
                model_output.display(),
                report_output.display(),
                output_report.report.final_loss,
                output_report.robustness.passed
            );
            if !output_report.robustness.passed {
                return Err(std::io::Error::other(format!(
                    "torus robustness validation failed; see {}",
                    report_output.display()
                ))
                .into());
            }
        }
        Command::TrainTorusMorphogen3d {
            model_output,
            report_output,
            rows,
            steps,
            training_mode,
            rollout_particles,
            rollout_steps,
            rollouts,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "train-torus-morphogen3d")?;
            let hashgrid = burn_automata::kernels::HashGridConfig::growing_3dgs();
            let (_config, mut model, batch, sgd, target_source, rollout_report) =
                match training_mode {
                    MeshTrainingModeArg::PositionField => {
                        let config = NpaConfig::torus_field_3dgs();
                        (
                            config.clone(),
                            torus_field_model(config.clone())?,
                            torus_field_supervised_batch(&config, rows),
                            SgdConfig {
                                learning_rate: 2.0e-3,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            UV_TORUS_POSITION_FIELD_TARGET_SOURCE,
                            None,
                        )
                    }
                    MeshTrainingModeArg::RolloutPositionField => {
                        let config = NpaConfig::torus_field_3dgs();
                        let model = torus_field_model(config.clone())?;
                        let feature_rows = rows / 2;
                        let rollout_rows = rows.saturating_sub(feature_rows).max(1);
                        let batch = merge_supervised_batches(
                            torus_field_supervised_batch(&config, feature_rows.max(1)),
                            mesh_field_rollout_supervised_batch(
                                &model,
                                &hashgrid,
                                &uv_torus_mesh_target(UV_TORUS_FIELD_SCALE),
                                MeshFieldRolloutBatchConfig {
                                    max_rows: rollout_rows,
                                    particle_count: rollout_particles,
                                    rollout_steps,
                                    rollouts,
                                    temporal_samples: 1,
                                    seed: 0x70_75,
                                    seed_scale: UV_TORUS_FIELD_SCALE,
                                    seed_mode: ParticleSeed::TorusFieldDense3d,
                                    motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
                                    max_update_norm: f32::INFINITY,
                                    coverage_gain: 0.0,
                                    coverage_samples: 0,
                                    coverage_mode: CoverageUpdateModeArg::HardNearest,
                                    coverage_softness: 0.0,
                                    coverage_repulsion_gain: 0.0,
                                    coverage_gap_gain: 0.0,
                                    coverage_repulsion_radius: 0.0,
                                    coverage_normal_weight: 0.0,
                                    extent_gain: 0.0,
                                    color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                                    aux_state_gain: 1.0,
                                    opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                                    front_opacity_gain: 0.0,
                                    front_radius: 0.0,
                                    front_max_opacity_update: 0.0,
                                    front_motion_gate: false,
                                    preserve_opacity_update: false,
                                },
                            )?,
                        );
                        let rollout_report = CliRolloutSupervisionReport {
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            temporal_samples: 1,
                            update_prob: 1.0,
                            seed_scale: UV_TORUS_FIELD_SCALE,
                            seed_mode: ParticleSeed::TorusFieldDense3d,
                            motion_gain: Some(UV_TORUS_FIELD_MOTION_GAIN),
                            max_update_norm: Some(f32::INFINITY),
                            density_gain: Some(0.0),
                            expansion_gain: None,
                            coverage_gain: Some(0.0),
                            coverage_samples: None,
                            coverage_mode: None,
                            coverage_softness: None,
                            coverage_repulsion_gain: None,
                            coverage_gap_gain: None,
                            coverage_repulsion_radius: None,
                            coverage_normal_weight: None,
                            extent_gain: None,
                            color_gain: Some(UV_TORUS_FIELD_COLOR_GAIN),
                            aux_state_gain: Some(1.0),
                            opacity_gain: Some(UV_TORUS_FIELD_OPACITY_GAIN),
                            front_opacity_gain: None,
                            front_radius: None,
                            front_max_opacity_update: None,
                            front_motion_gate: None,
                            preserve_opacity_update: None,
                        };
                        (
                            config,
                            model,
                            batch,
                            SgdConfig {
                                learning_rate: 2.0e-3,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE,
                            Some(rollout_report),
                        )
                    }
                    MeshTrainingModeArg::RolloutLocal => {
                        let config = NpaConfig::growing_3dgs();
                        let target_mesh = uv_torus_mesh_target(UV_TORUS_FIELD_SCALE);
                        let student = local_growth_student_model_with_axis_gains(
                            config.clone(),
                            0x70_75,
                            0.0,
                            mesh_axis_expansion_gains(&target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
                        )?;
                        let rollout_report = CliRolloutSupervisionReport {
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            temporal_samples: 5,
                            update_prob: 1.0,
                            seed_scale: UV_TORUS_FIELD_SCALE,
                            seed_mode: ParticleSeed::TorusGrowth3d,
                            motion_gain: Some(LOCAL_TORUS_MOTION_GAIN),
                            max_update_norm: Some(0.06),
                            density_gain: Some(0.0),
                            expansion_gain: Some(LOCAL_GROWTH_EXPANSION_GAIN),
                            coverage_gain: Some(0.45),
                            coverage_samples: Some(4096),
                            coverage_mode: Some(CoverageUpdateModeArg::SlicedOt),
                            coverage_softness: Some(0.0),
                            coverage_repulsion_gain: Some(0.2),
                            coverage_gap_gain: Some(0.2),
                            coverage_repulsion_radius: Some(0.0),
                            coverage_normal_weight: Some(0.0),
                            extent_gain: Some(0.4),
                            color_gain: Some(LOCAL_TORUS_COLOR_GAIN),
                            aux_state_gain: Some(0.5),
                            opacity_gain: Some(0.02),
                            front_opacity_gain: Some(0.05),
                            front_radius: Some(0.24),
                            front_max_opacity_update: Some(0.16),
                            front_motion_gate: Some(true),
                            preserve_opacity_update: Some(false),
                        };
                        let batch = mesh_local_rollout_supervised_batch(
                            &student,
                            &hashgrid,
                            &target_mesh,
                            MeshFieldRolloutBatchConfig {
                                max_rows: rows,
                                particle_count: rollout_particles,
                                rollout_steps,
                                rollouts,
                                temporal_samples: 5,
                                seed: 0x70_75,
                                seed_scale: UV_TORUS_FIELD_SCALE,
                                seed_mode: ParticleSeed::TorusGrowth3d,
                                motion_gain: LOCAL_TORUS_MOTION_GAIN,
                                max_update_norm: 0.06,
                                coverage_gain: 0.45,
                                coverage_samples: 4096,
                                coverage_mode: CoverageUpdateModeArg::SlicedOt,
                                coverage_softness: 0.0,
                                coverage_repulsion_gain: 0.2,
                                coverage_gap_gain: 0.2,
                                coverage_repulsion_radius: 0.0,
                                coverage_normal_weight: 0.0,
                                extent_gain: 0.4,
                                color_gain: LOCAL_TORUS_COLOR_GAIN,
                                aux_state_gain: 0.5,
                                opacity_gain: 0.02,
                                front_opacity_gain: 0.05,
                                front_radius: 0.24,
                                front_max_opacity_update: 0.16,
                                front_motion_gate: true,
                                preserve_opacity_update: false,
                            },
                        )?;
                        (
                            config,
                            student,
                            batch,
                            SgdConfig {
                                learning_rate: 4.0e-5,
                                grad_clip_norm: 0.06,
                                ..SgdConfig::default()
                            },
                            UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE,
                            Some(rollout_report),
                        )
                    }
                    MeshTrainingModeArg::ProjectionBaseline => {
                        let config = NpaConfig::growing_3dgs();
                        (
                            config.clone(),
                            torus_morphogen_model(config.clone())?,
                            torus_morphogen_supervised_batch(&config, rows),
                            SgdConfig {
                                learning_rate: 0.0,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE,
                            None,
                        )
                    }
                };
            let report = run_supervised_training(
                &mut model,
                &batch,
                TrainingRunConfig {
                    steps,
                    report_interval: steps.max(1),
                    sgd,
                },
            )?;
            if let Some(parent) = model_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Some(parent) = report_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid.clone(),
                Some(format!("trained-rust:{target_source}")),
            );
            burn_automata::import::save_manifest(&model_output, &manifest)?;
            let loaded = burn_automata::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let robustness_cases = if loaded_model.config.position_features {
                TORUS_ROBUSTNESS_CASES
            } else {
                TORUS_MORPHOGEN_ROBUSTNESS_CASES
            };
            let robustness = torus_robustness_report_for_cases(
                &loaded_model,
                &loaded_hashgrid,
                robustness_cases,
            )?;
            let output_report = CliTorusTrainingReport {
                preset: AutomataPreset::Growing3dGs,
                target_source: target_source.to_string(),
                student_seed: 0,
                sgd,
                report,
                model_output: Some(model_output.display().to_string()),
                robustness,
                batch_source: if matches!(
                    training_mode,
                    MeshTrainingModeArg::RolloutLocal | MeshTrainingModeArg::RolloutPositionField
                ) {
                    TrainingBatchArg::Rollout
                } else {
                    TrainingBatchArg::Features
                },
                training_mode,
                rollout_supervision: rollout_report,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            println!(
                "wrote {} and {} final_loss={:.6} robust={}",
                model_output.display(),
                report_output.display(),
                output_report.report.final_loss,
                output_report.robustness.passed
            );
            if !output_report.robustness.passed {
                return Err(std::io::Error::other(format!(
                    "torus morphogen robustness validation failed; see {}",
                    report_output.display()
                ))
                .into());
            }
        }
        Command::TrainTeapotMorphogen3d {
            model_output,
            report_output,
            rows,
            steps,
            training_mode,
            rollout_particles,
            rollout_steps,
            rollouts,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "train-teapot-morphogen3d")?;
            let hashgrid = burn_automata::kernels::HashGridConfig::growing_3dgs();
            let (_config, mut model, batch, sgd, target_source, rollout_report) =
                match training_mode {
                    MeshTrainingModeArg::PositionField => {
                        let config = NpaConfig::torus_field_3dgs();
                        (
                            config.clone(),
                            teapot_field_model(config.clone())?,
                            teapot_field_supervised_batch(&config, rows),
                            SgdConfig {
                                learning_rate: 2.0e-3,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            TEAPOT_POSITION_FIELD_TARGET_SOURCE,
                            None,
                        )
                    }
                    MeshTrainingModeArg::RolloutPositionField => {
                        let config = NpaConfig::torus_field_3dgs();
                        let model = teapot_field_model(config.clone())?;
                        let feature_rows = rows / 2;
                        let rollout_rows = rows.saturating_sub(feature_rows).max(1);
                        let batch = merge_supervised_batches(
                            teapot_field_supervised_batch(&config, feature_rows.max(1)),
                            mesh_field_rollout_supervised_batch(
                                &model,
                                &hashgrid,
                                &utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE),
                                MeshFieldRolloutBatchConfig {
                                    max_rows: rollout_rows,
                                    particle_count: rollout_particles,
                                    rollout_steps,
                                    rollouts,
                                    temporal_samples: 1,
                                    seed: 0x7ea9_07d0,
                                    seed_scale: UV_TORUS_FIELD_SCALE,
                                    seed_mode: ParticleSeed::TeapotFieldDense3d,
                                    motion_gain: TEAPOT_FIELD_MOTION_GAIN,
                                    max_update_norm: f32::INFINITY,
                                    coverage_gain: 0.0,
                                    coverage_samples: 0,
                                    coverage_mode: CoverageUpdateModeArg::HardNearest,
                                    coverage_softness: 0.0,
                                    coverage_repulsion_gain: 0.0,
                                    coverage_gap_gain: 0.0,
                                    coverage_repulsion_radius: 0.0,
                                    coverage_normal_weight: 0.0,
                                    extent_gain: 0.0,
                                    color_gain: TEAPOT_FIELD_COLOR_GAIN,
                                    aux_state_gain: 1.0,
                                    opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                                    front_opacity_gain: 0.0,
                                    front_radius: 0.0,
                                    front_max_opacity_update: 0.0,
                                    front_motion_gate: false,
                                    preserve_opacity_update: false,
                                },
                            )?,
                        );
                        let rollout_report = CliRolloutSupervisionReport {
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            temporal_samples: 1,
                            update_prob: 1.0,
                            seed_scale: UV_TORUS_FIELD_SCALE,
                            seed_mode: ParticleSeed::TeapotFieldDense3d,
                            motion_gain: Some(TEAPOT_FIELD_MOTION_GAIN),
                            max_update_norm: Some(f32::INFINITY),
                            density_gain: Some(0.0),
                            expansion_gain: None,
                            coverage_gain: Some(0.0),
                            coverage_samples: None,
                            coverage_mode: None,
                            coverage_softness: None,
                            coverage_repulsion_gain: None,
                            coverage_gap_gain: None,
                            coverage_repulsion_radius: None,
                            coverage_normal_weight: None,
                            extent_gain: None,
                            color_gain: Some(TEAPOT_FIELD_COLOR_GAIN),
                            aux_state_gain: Some(1.0),
                            opacity_gain: Some(UV_TORUS_FIELD_OPACITY_GAIN),
                            front_opacity_gain: None,
                            front_radius: None,
                            front_max_opacity_update: None,
                            front_motion_gate: None,
                            preserve_opacity_update: None,
                        };
                        (
                            config,
                            model,
                            batch,
                            SgdConfig {
                                learning_rate: 2.0e-3,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE,
                            Some(rollout_report),
                        )
                    }
                    MeshTrainingModeArg::RolloutLocal => {
                        let config = NpaConfig::growing_3dgs();
                        let target_mesh = utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE);
                        let student = local_growth_student_model_with_axis_gains(
                            config.clone(),
                            0x7ea9_07d0,
                            0.0,
                            mesh_axis_expansion_gains(&target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
                        )?;
                        let rollout_report = CliRolloutSupervisionReport {
                            particle_count: rollout_particles,
                            rollout_steps,
                            rollouts,
                            temporal_samples: 4,
                            update_prob: 1.0,
                            seed_scale: UV_TORUS_FIELD_SCALE,
                            seed_mode: ParticleSeed::TeapotGrowth3d,
                            motion_gain: Some(LOCAL_TEAPOT_MOTION_GAIN),
                            max_update_norm: Some(0.06),
                            density_gain: Some(0.0),
                            expansion_gain: Some(LOCAL_GROWTH_EXPANSION_GAIN),
                            coverage_gain: Some(0.35),
                            coverage_samples: Some(4096),
                            coverage_mode: Some(CoverageUpdateModeArg::SlicedOt),
                            coverage_softness: Some(0.0),
                            coverage_repulsion_gain: Some(0.2),
                            coverage_gap_gain: Some(0.2),
                            coverage_repulsion_radius: Some(0.0),
                            coverage_normal_weight: Some(0.0),
                            extent_gain: Some(0.14),
                            color_gain: Some(LOCAL_TEAPOT_COLOR_GAIN),
                            aux_state_gain: Some(0.3),
                            opacity_gain: Some(0.12),
                            front_opacity_gain: Some(0.05),
                            front_radius: Some(0.24),
                            front_max_opacity_update: Some(0.16),
                            front_motion_gate: Some(true),
                            preserve_opacity_update: Some(false),
                        };
                        let batch = mesh_local_rollout_supervised_batch(
                            &student,
                            &hashgrid,
                            &target_mesh,
                            MeshFieldRolloutBatchConfig {
                                max_rows: rows,
                                particle_count: rollout_particles,
                                rollout_steps,
                                rollouts,
                                temporal_samples: 4,
                                seed: 0x7ea9_07d0,
                                seed_scale: UV_TORUS_FIELD_SCALE,
                                seed_mode: ParticleSeed::TeapotGrowth3d,
                                motion_gain: LOCAL_TEAPOT_MOTION_GAIN,
                                max_update_norm: 0.06,
                                coverage_gain: 0.35,
                                coverage_samples: 4096,
                                coverage_mode: CoverageUpdateModeArg::SlicedOt,
                                coverage_softness: 0.0,
                                coverage_repulsion_gain: 0.2,
                                coverage_gap_gain: 0.2,
                                coverage_repulsion_radius: 0.0,
                                coverage_normal_weight: 0.0,
                                extent_gain: 0.14,
                                color_gain: LOCAL_TEAPOT_COLOR_GAIN,
                                aux_state_gain: 0.3,
                                opacity_gain: 0.12,
                                front_opacity_gain: 0.05,
                                front_radius: 0.24,
                                front_max_opacity_update: 0.16,
                                front_motion_gate: true,
                                preserve_opacity_update: false,
                            },
                        )?;
                        (
                            config,
                            student,
                            batch,
                            SgdConfig {
                                learning_rate: 5.0e-5,
                                grad_clip_norm: 0.08,
                                ..SgdConfig::default()
                            },
                            TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE,
                            Some(rollout_report),
                        )
                    }
                    MeshTrainingModeArg::ProjectionBaseline => {
                        let config = NpaConfig::growing_3dgs();
                        (
                            config.clone(),
                            seed_frame_morphogen_model(config.clone())?,
                            teapot_morphogen_supervised_batch(&config, rows),
                            SgdConfig {
                                learning_rate: 0.0,
                                grad_clip_norm: 1.0,
                                ..SgdConfig::default()
                            },
                            TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE,
                            None,
                        )
                    }
                };
            let report = run_supervised_training(
                &mut model,
                &batch,
                TrainingRunConfig {
                    steps,
                    report_interval: steps.max(1),
                    sgd,
                },
            )?;
            if let Some(parent) = model_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Some(parent) = report_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid.clone(),
                Some(format!("trained-rust:{target_source}")),
            );
            burn_automata::import::save_manifest(&model_output, &manifest)?;
            let loaded = burn_automata::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let mesh_rollout = if matches!(
                training_mode,
                MeshTrainingModeArg::PositionField | MeshTrainingModeArg::RolloutPositionField
            ) {
                Some(mesh_rollout_report_for_cases(
                    &loaded_model,
                    &loaded_hashgrid,
                    &utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE),
                    TEAPOT_FIELD_ROLLOUT_CASES,
                )?)
            } else {
                None
            };
            let output_report = CliTrainingReport {
                preset: AutomataPreset::Growing3dGs,
                target_source: target_source.to_string(),
                student_seed: 0,
                sgd,
                report,
                model_output: Some(model_output.display().to_string()),
                batch_source: if matches!(
                    training_mode,
                    MeshTrainingModeArg::RolloutLocal | MeshTrainingModeArg::RolloutPositionField
                ) {
                    TrainingBatchArg::Rollout
                } else {
                    TrainingBatchArg::Features
                },
                rollout_supervision: rollout_report,
                mesh_rollout,
                render_loss: None,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            println!(
                "wrote {} and {} final_loss={:.6} mesh_rollout={}",
                model_output.display(),
                report_output.display(),
                output_report.report.final_loss,
                output_report
                    .mesh_rollout
                    .as_ref()
                    .map_or("skipped", |report| if report.passed {
                        "passed"
                    } else {
                        "failed"
                    })
            );
            if output_report
                .mesh_rollout
                .as_ref()
                .is_some_and(|report| !report.passed)
            {
                return Err(std::io::Error::other(format!(
                    "teapot mesh rollout validation failed; see {}",
                    report_output.display()
                ))
                .into());
            }
        }
        Command::AblateLocal3d {
            target,
            base_model,
            model_output,
            report_output,
            rows,
            steps,
            rollout_particles,
            rollout_steps,
            rollouts,
            temporal_samples,
            training_rounds,
            seed_scale,
            seed_mode,
            student_seed,
            learning_rate,
            grad_clip_norm,
            weight_decay,
            motion_gain,
            max_update_norm,
            density_gain,
            expansion_gain,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            extent_gain,
            color_gain,
            aux_state_gain,
            opacity_gain,
            front_opacity_gain,
            front_radius,
            front_max_opacity_update,
            front_motion_gate,
            preserve_opacity_update,
            fail_on_validation,
        } => {
            validate_diagnostic_3d_output_not_catalog(&model_output, "ablate-local-3d")?;
            let target_mesh = mesh_target_for_arg(target, seed_scale);
            let seed_mode = seed_mode
                .map(ParticleSeed::from)
                .unwrap_or_else(|| conditionless_local_seed_mode(target));
            let target_source = mesh_conditionless_local_target_source_for_seed(target, seed_mode);
            let (mut model, hashgrid, output_source) = if let Some(path) = base_model.as_ref() {
                load_conditionless_local_base_model(path, target_source)?
            } else {
                let config = NpaConfig::growing_3dgs();
                let hashgrid = burn_automata::kernels::HashGridConfig::growing_3dgs();
                let model = local_growth_student_model_with_axis_gains(
                    config,
                    student_seed,
                    density_gain,
                    mesh_axis_expansion_gains(&target_mesh, expansion_gain),
                )?;
                (model, hashgrid, format!("ablation-rust:{target_source}"))
            };
            let sgd = SgdConfig {
                learning_rate,
                grad_clip_norm,
                weight_decay,
            };
            let preserve_opacity_update =
                preserve_opacity_update || (opacity_gain == 0.0 && front_opacity_gain == 0.0);
            let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
            let report = run_refreshed_mesh_local_training(
                &mut model,
                &hashgrid,
                &target_mesh,
                MeshLocalTrainingConfig {
                    max_rows: rows,
                    particle_count: rollout_particles,
                    rollout_steps,
                    rollouts,
                    temporal_samples,
                    training_rounds,
                    total_steps: steps,
                    seed: student_seed ^ 0x5eed_3d,
                    seed_scale,
                    seed_mode,
                    motion_gain: motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target)),
                    max_update_norm,
                    coverage_gain,
                    coverage_samples,
                    coverage_mode,
                    coverage_softness,
                    coverage_repulsion_gain,
                    coverage_gap_gain,
                    coverage_repulsion_radius,
                    coverage_normal_weight,
                    extent_gain,
                    color_gain: color_gain.unwrap_or_else(|| mesh_target_color_gain(target)),
                    aux_state_gain,
                    opacity_gain,
                    front_opacity_gain,
                    front_radius,
                    front_max_opacity_update,
                    front_motion_gate,
                    preserve_opacity_update,
                    sgd,
                },
            )?;
            if let Some(parent) = model_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Some(parent) = report_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let manifest =
                BpkModelManifest::from_model(&model, hashgrid.clone(), Some(output_source.clone()));
            burn_automata::import::save_manifest(&model_output, &manifest)?;
            let loaded = burn_automata::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let validation_cases =
                conditionless_local_rollout_cases(target, seed_scale, rollout_particles);
            let mesh_rollout = Some(mesh_rollout_report_for_cases(
                &loaded_model,
                &loaded_hashgrid,
                &target_mesh,
                &validation_cases,
            )?);
            let render_loss = Some(mesh_render_loss_for_model(
                &loaded_model,
                &loaded_hashgrid,
                &target_mesh,
                RenderLossEvalConfig {
                    particle_count: rollout_particles,
                    steps: 64,
                    seed: 0x10ca_202,
                    extra_seeds: Vec::new(),
                    seed_scale,
                    seed_mode,
                    render: default_render_loss_config(seed_scale),
                },
            )?);
            let rollout_supervision = Some(CliRolloutSupervisionReport {
                particle_count: rollout_particles,
                rollout_steps,
                rollouts,
                temporal_samples,
                update_prob: 1.0,
                seed_scale,
                seed_mode,
                motion_gain: Some(motion_gain.unwrap_or_else(|| mesh_target_motion_gain(target))),
                max_update_norm: Some(max_update_norm),
                density_gain: Some(density_gain),
                expansion_gain: Some(expansion_gain),
                coverage_gain: Some(coverage_gain),
                coverage_samples: Some(coverage_samples),
                coverage_mode: Some(coverage_mode),
                coverage_softness: Some(coverage_softness),
                coverage_repulsion_gain: Some(coverage_repulsion_gain),
                coverage_gap_gain: Some(coverage_gap_gain),
                coverage_repulsion_radius: Some(coverage_repulsion_radius),
                coverage_normal_weight: Some(coverage_normal_weight),
                extent_gain: Some(extent_gain),
                color_gain: Some(color_gain.unwrap_or_else(|| mesh_target_color_gain(target))),
                aux_state_gain: Some(aux_state_gain),
                opacity_gain: Some(opacity_gain),
                front_opacity_gain: Some(front_opacity_gain),
                front_radius: Some(front_radius),
                front_max_opacity_update: Some(front_max_opacity_update),
                front_motion_gate: Some(front_motion_gate),
                preserve_opacity_update: Some(preserve_opacity_update),
            });
            let output_report = CliTrainingReport {
                preset: AutomataPreset::Growing3dGs,
                target_source: output_source,
                student_seed,
                sgd,
                report,
                model_output: Some(model_output.display().to_string()),
                batch_source: TrainingBatchArg::Rollout,
                rollout_supervision,
                mesh_rollout,
                render_loss,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            let mesh_status = output_report
                .mesh_rollout
                .as_ref()
                .map_or(
                    "skipped",
                    |report| if report.passed { "passed" } else { "failed" },
                );
            let render_status = output_report
                .render_loss
                .as_ref()
                .map_or(
                    "skipped",
                    |report| if report.passed { "passed" } else { "failed" },
                );
            println!(
                "wrote {} and {} final_loss={:.6} mesh_rollout={mesh_status} render_loss={render_status}",
                model_output.display(),
                report_output.display(),
                output_report.report.final_loss
            );
            if fail_on_validation
                && output_report
                    .mesh_rollout
                    .as_ref()
                    .is_some_and(|report| !report.passed)
            {
                return Err(std::io::Error::other(format!(
                    "conditionless local 3d ablation failed validation; see {}",
                    report_output.display()
                ))
                .into());
            }
            if fail_on_validation
                && output_report
                    .render_loss
                    .as_ref()
                    .is_some_and(|report| !report.passed)
            {
                return Err(std::io::Error::other(format!(
                    "conditionless local 3d render validation failed; see {}",
                    report_output.display()
                ))
                .into());
            }
        }
        Command::RenderLoss3d {
            model,
            target,
            output,
            particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            image_size,
            target_samples,
            sigma,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            fail_on_validation,
        } => {
            let manifest = burn_automata::import::load_manifest(&model)?;
            let hashgrid = manifest.hashgrid.clone();
            let loaded_model = manifest.into_model();
            let target_mesh = mesh_target_for_arg(target, seed_scale);
            let seed_mode: ParticleSeed = seed_mode.into();
            let render_loss = mesh_render_loss_for_model(
                &loaded_model,
                &hashgrid,
                &target_mesh,
                RenderLossEvalConfig {
                    particle_count: particles,
                    steps,
                    seed,
                    extra_seeds,
                    seed_scale,
                    seed_mode,
                    render: RenderLossConfig {
                        image_size,
                        sigma,
                        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                        target_samples,
                        opacity_logit_bias: render_opacity_logit_bias,
                        density_weight,
                        color_weight,
                        depth_weight,
                    },
                },
            )?;
            let output_report = CliRenderLossEvalReport {
                target,
                model: model.display().to_string(),
                particle_count: particles,
                steps,
                seed,
                seed_scale,
                seed_mode,
                render_loss,
            };
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&output_report)?)?;
            println!(
                "wrote {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={}",
                output.display(),
                output_report.render_loss.total_loss,
                output_report.render_loss.density_psnr_db,
                output_report.render_loss.color_psnr_db,
                output_report.render_loss.depth_psnr_db,
                output_report.render_loss.passed
            );
            if fail_on_validation && !output_report.render_loss.passed {
                return Err(std::io::Error::other(format!(
                    "render loss validation failed; see {}",
                    output.display()
                ))
                .into());
            }
        }
        Command::ValidateGrowth3d {
            model,
            target,
            output,
            particles,
            steps,
            seed,
            extra_seeds,
            seed_scale,
            seed_mode,
            image_size,
            target_samples,
            sigma,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            gate,
            fail_on_validation,
        } => {
            let seed_mode = seed_mode
                .map(ParticleSeed::from)
                .unwrap_or_else(|| conditionless_local_seed_mode(target));
            let report = growth_3d_validation_report(
                &model,
                target,
                Growth3dValidationConfig {
                    particle_count: particles,
                    steps,
                    seed,
                    extra_seeds,
                    seed_scale,
                    seed_mode,
                    gate,
                    render: RenderLossConfig {
                        image_size,
                        sigma,
                        world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                        target_samples,
                        opacity_logit_bias: render_opacity_logit_bias,
                        density_weight,
                        color_weight,
                        depth_weight,
                    },
                },
            )?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
            println!(
                "wrote {} gate={:?} gate_passed={} robust_gate_passed={} strict_passed={} strict_score={:.6} catalog_sanity={} render_loss={:.6} density_psnr={:.3} active={}->{} newly_activated_fraction={:.3} opacity_max={:.3}",
                output.display(),
                report.gate,
                report.gate_passed,
                report.robustness.all_gate_passed,
                report.strict_passed,
                report.strict_score.score,
                report.catalog_sanity.passed,
                report.render_loss.total_loss,
                report.render_loss.density_psnr_db,
                report.activation.active_seed_count,
                report.activation.final_active_count,
                report.activation.newly_activated_fraction,
                report.final_opacity.max,
            );
            if fail_on_validation && !growth_3d_fail_on_validation_passed(&report) {
                return Err(std::io::Error::other(format!(
                    "growth 3D validation failed; see {}",
                    output.display()
                ))
                .into());
            }
        }
        Command::RetimeGrowth3d {
            model,
            output,
            front_gain,
            hidden,
            skip_front_retime,
            active_opacity_gain,
            active_opacity_hidden,
            opacity_bias,
            material_opacity_bias,
            alpha,
        } => {
            validate_diagnostic_3d_output_not_catalog(&output, "retime-growth3d")?;
            let manifest = burn_automata::import::load_manifest(&model)?;
            let source = manifest.source.clone();
            let hashgrid = manifest.hashgrid.clone();
            let mut model_value = manifest.into_model();
            let hidden = if skip_front_retime {
                hidden
            } else {
                Some(retime_growth_3d_front_model(
                    &mut model_value,
                    hidden,
                    front_gain,
                )?)
            };
            let active_opacity_hidden = if let Some(gain) = active_opacity_gain {
                Some(retime_growth_3d_active_opacity_model(
                    &mut model_value,
                    active_opacity_hidden,
                    gain,
                )?)
            } else {
                None
            };
            if let Some(alpha) = alpha {
                if !alpha.is_finite() || alpha <= 0.0 {
                    return Err(std::io::Error::other("alpha must be positive and finite").into());
                }
                model_value.config.alpha = alpha;
            }
            if let Some(opacity_bias) = opacity_bias {
                add_growth_3d_opacity_update_bias(&mut model_value, opacity_bias)?;
            }
            if let Some(material_opacity_bias) = material_opacity_bias {
                add_growth_3d_material_opacity_update_bias(
                    &mut model_value,
                    material_opacity_bias,
                )?;
            }
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let retimed_source = Some(format!(
                "retimed-local-front:hidden={}:gain={front_gain}:alpha={}:front_retime={}:active_opacity_hidden={}:active_opacity_gain={}:opacity_bias={}:material_opacity_bias={}:base={}",
                hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                model_value.config.alpha,
                !skip_front_retime,
                active_opacity_hidden
                    .map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
                opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                material_opacity_bias
                    .map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                source.as_deref().unwrap_or("unknown")
            ));
            let retimed_manifest =
                BpkModelManifest::from_model(&model_value, hashgrid, retimed_source);
            burn_automata::import::save_manifest(&output, &retimed_manifest)?;
            println!(
                "wrote {} retimed_hidden={} front_gain={} alpha={} front_retime={} active_opacity_hidden={} active_opacity_gain={} opacity_bias={} material_opacity_bias={}",
                output.display(),
                hidden.map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                front_gain,
                model_value.config.alpha,
                !skip_front_retime,
                active_opacity_hidden
                    .map_or_else(|| "skipped".to_string(), |hidden| hidden.to_string()),
                active_opacity_gain.map_or_else(|| "skipped".to_string(), |gain| gain.to_string()),
                opacity_bias.map_or_else(|| "skipped".to_string(), |bias| bias.to_string()),
                material_opacity_bias
                    .map_or_else(|| "skipped".to_string(), |bias| bias.to_string())
            );
        }
        Command::TrainRender3d {
            target,
            base_model,
            model_output,
            report_output,
            rounds,
            supervised_steps_per_round,
            particles,
            rollout_steps,
            gradient_particles,
            gradient_mode,
            finite_diff_eps,
            motion_gain,
            perception_position_gain,
            max_update_norm,
            trajectory_supervision,
            trajectory_render_gain,
            trajectory_render_samples,
            coverage_gain,
            coverage_samples,
            coverage_mode,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_gap_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            full_coverage_adjoint,
            surface_gain,
            opacity_gain,
            max_opacity_update,
            learning_rate,
            grad_clip_norm,
            direct_line_search,
            direct_line_search_scales,
            direct_material_output_only,
            training_backend,
            direct_selection_seed_training,
            seed_scale,
            seed_mode,
            selection_seed,
            extra_selection_seeds,
            image_size,
            target_samples,
            sigma,
            world_scale,
            render_opacity_logit_bias,
            density_weight,
            color_weight,
            depth_weight,
            fail_on_validation,
        } => {
            let hashgrid = burn_automata::kernels::HashGridConfig::growing_3dgs();
            let requested_seed_mode = seed_mode.map(ParticleSeed::from);
            let target_mesh = mesh_target_for_arg(target, seed_scale);
            let (mut model, base_source, default_seed_mode) = if let Some(path) =
                base_model.as_ref()
            {
                let manifest = burn_automata::import::load_manifest(path)?;
                let base_source = manifest.source.clone();
                let model = manifest.into_model();
                let default_seed_mode = default_render_training_seed_mode(target, &model);
                (model, base_source, default_seed_mode)
            } else {
                let default_seed_mode = render_training_default_seed_mode(target);
                let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
                if !target_local_growth_seed(target, seed_mode) {
                    return Err(std::io::Error::other(format!(
                        "train-render3d without --base-model defaults to conditionless-local growth and requires a target local growth seed; got seed_mode={seed_mode:?}"
                    ))
                    .into());
                }
                let (model, source) = render_training_base_model(target, &target_mesh, seed_mode)?;
                (model, Some(source), default_seed_mode)
            };
            let seed_mode = requested_seed_mode.unwrap_or(default_seed_mode);
            let catalog_bound_output = is_catalog_model_output_path(&model_output);
            validate_catalog_bound_render_training_output(
                &model_output,
                target,
                seed_mode,
                base_source.as_deref(),
            )?;
            let coverage_gap_gain = coverage_gap_gain.unwrap_or(coverage_repulsion_gain);
            let render = RenderLossConfig {
                image_size,
                sigma,
                world_scale: world_scale.unwrap_or(seed_scale * 2.0),
                target_samples,
                opacity_logit_bias: render_opacity_logit_bias,
                density_weight,
                color_weight,
                depth_weight,
            };
            let sgd = SgdConfig {
                learning_rate,
                grad_clip_norm,
                weight_decay: 0.0,
            };
            let report = run_render_proxy_training(
                &mut model,
                &hashgrid,
                &target_mesh,
                RenderProxyTrainingConfig {
                    target,
                    rounds,
                    supervised_steps_per_round,
                    particles,
                    rollout_steps,
                    gradient_particles,
                    gradient_mode,
                    finite_diff_eps,
                    motion_gain,
                    perception_position_gain,
                    max_update_norm,
                    trajectory_supervision,
                    trajectory_render_gain,
                    trajectory_render_samples,
                    coverage_gain,
                    coverage_samples,
                    coverage_mode,
                    coverage_softness,
                    coverage_repulsion_gain,
                    coverage_gap_gain,
                    coverage_repulsion_radius,
                    coverage_normal_weight,
                    full_coverage_adjoint,
                    surface_gain,
                    opacity_gain,
                    max_opacity_update,
                    direct_line_search,
                    direct_line_search_scales: direct_line_search_scales.clone(),
                    direct_material_output_only,
                    training_backend,
                    direct_selection_seed_training,
                    seed: 0x5a17_3d,
                    selection_seed: Some(selection_seed),
                    selection_seeds: extra_selection_seeds.clone(),
                    seed_scale,
                    seed_mode,
                    render,
                    sgd,
                },
            )?;
            if let Some(parent) = model_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if let Some(parent) = report_output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid.clone(),
                Some(render_training_source(
                    target,
                    base_source.as_deref(),
                    seed_mode,
                )),
            );
            if catalog_bound_output {
                let candidate_path = catalog_bound_candidate_path(target, std::process::id());
                if let Some(parent) = candidate_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                burn_automata::import::save_manifest(&candidate_path, &manifest)?;
                let mut validation_extra_seeds = vec![selection_seed];
                validation_extra_seeds.extend(extra_selection_seeds.iter().copied());
                let validation = growth_3d_validation_report(
                    &candidate_path,
                    target,
                    Growth3dValidationConfig {
                        particle_count: particles,
                        steps: rollout_steps,
                        seed: 0x5a17_3d,
                        extra_seeds: validation_extra_seeds,
                        seed_scale,
                        seed_mode,
                        gate: Growth3dValidationGateArg::Strict,
                        render,
                    },
                )?;
                if !growth_3d_fail_on_validation_passed(&validation) {
                    std::fs::remove_file(&candidate_path).ok();
                    return Err(std::io::Error::other(format!(
                        "catalog-bound 3D render training candidate failed strict growth validation (score={:.6}, failures={:?}); refusing to overwrite {}",
                        validation.strict_score.score,
                        validation.strict_checks.failure_reasons,
                        model_output.display()
                    ))
                    .into());
                }
                burn_automata::import::save_manifest(&model_output, &manifest)?;
                std::fs::remove_file(&candidate_path).ok();
            } else {
                burn_automata::import::save_manifest(&model_output, &manifest)?;
            }
            let loaded = burn_automata::import::load_manifest(&model_output)?;
            let loaded_hashgrid = loaded.hashgrid.clone();
            let loaded_model = loaded.into_model();
            let final_render_loss = mesh_render_loss_for_model(
                &loaded_model,
                &loaded_hashgrid,
                &target_mesh,
                RenderLossEvalConfig {
                    particle_count: particles,
                    steps: rollout_steps,
                    seed: 0x5a17_3d,
                    extra_seeds: Vec::new(),
                    seed_scale,
                    seed_mode,
                    render,
                },
            )?;
            let output_report = CliRenderTrainingReport {
                target,
                base_model: base_model.as_ref().map(|path| path.display().to_string()),
                model_output: model_output.display().to_string(),
                particle_count: particles,
                rollout_steps,
                seed_scale,
                seed_mode,
                sgd,
                report,
                final_render_loss,
            };
            std::fs::write(
                &report_output,
                serde_json::to_string_pretty(&output_report)?,
            )?;
            println!(
                "wrote {} and {} render_loss={:.6} density_psnr={:.3} color_psnr={:.3} depth_psnr={:.3} passed={}",
                model_output.display(),
                report_output.display(),
                output_report.final_render_loss.total_loss,
                output_report.final_render_loss.density_psnr_db,
                output_report.final_render_loss.color_psnr_db,
                output_report.final_render_loss.depth_psnr_db,
                output_report.final_render_loss.passed
            );
            if fail_on_validation && !output_report.final_render_loss.passed {
                return Err(std::io::Error::other(format!(
                    "render-proxy training failed render validation; see {}",
                    report_output.display()
                ))
                .into());
            }
        }
        Command::Import { input, output } => {
            let report = import_model(input, output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Bench {
            preset,
            particles,
            steps,
            repeats,
            update_prob,
            gpu,
            neighbor_mode,
            bucket_capacity,
            profile,
            seed_scale,
            normalize_seed_scale,
            fixed_eps,
            reference_seed_scale,
            seed_mode,
            geometry,
            gaussian,
        } => {
            #[cfg(not(feature = "gpu_wgpu"))]
            let _ = (neighbor_mode, bucket_capacity, gaussian, repeats);
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
            let reference_seed_scale = reference_seed_scale
                .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
            let (config, base_grid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config.clone(), 42);
            let grid = if normalize_seed_scale {
                model
                    .config
                    .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
            } else {
                base_grid
            };
            let start = Instant::now();
            if gpu {
                #[cfg(feature = "gpu_wgpu")]
                {
                    let report = gpu_rollout_bench(
                        &model,
                        &grid,
                        GpuBenchConfig {
                            particles,
                            steps,
                            seed_scale,
                            update_prob,
                            seed_mode,
                            geometry,
                            neighbor_mode: wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                            gaussian_write: gaussian,
                        },
                    )?;
                    let reports = if repeats > 1 {
                        let mut reports = Vec::with_capacity(repeats);
                        reports.push(report);
                        for _ in 1..repeats {
                            reports.push(gpu_rollout_bench(
                                &model,
                                &grid,
                                GpuBenchConfig {
                                    particles,
                                    steps,
                                    seed_scale,
                                    update_prob,
                                    seed_mode,
                                    geometry,
                                    neighbor_mode: wgpu_neighbor_mode(
                                        neighbor_mode,
                                        bucket_capacity,
                                    ),
                                    gaussian_write: gaussian,
                                },
                            )?);
                        }
                        reports
                    } else {
                        vec![report]
                    };
                    let summary = summarize_gpu_reports(&reports, steps);
                    let report = summary.median_report;
                    let avg_step_ms = report.gpu_step_ms / steps.max(1) as f64;
                    println!(
                        "backend=wgpu particles={particles} steps={steps} repeats={} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} gpu_step_ms={:.6} avg_step_ms={avg_step_ms:.6} min_avg_step_ms={:.6} median_avg_step_ms={:.6} max_avg_step_ms={:.6} final_mean_displacement_per_step={:.6} final_mean_density={:.6} initial_nonempty_cells={} initial_max_cell_occupancy={} hashgrid=gpu-local hashgrid_eps={:.6} normalized_seed_scale={} reference_seed_scale={:.6} resident_state=true timing=submit_wait readback=final gaussian_write={} neighbor_mode={:?} bucket_capacity={} grid_storage_u32={} grid_clear_u32={} grid_overflow_count={}",
                        summary.repeats,
                        start.elapsed().as_secs_f64() * 1000.0,
                        report.gpu_step_ms,
                        summary.min_avg_step_ms,
                        summary.median_avg_step_ms,
                        summary.max_avg_step_ms,
                        report.final_mean_dx,
                        report.final_mean_density,
                        report.initial_nonempty_cells,
                        report.initial_max_cell_occupancy,
                        grid.eps,
                        normalize_seed_scale,
                        reference_seed_scale,
                        report.gaussian_write,
                        report.neighbor_mode,
                        report.bucket_capacity,
                        report.grid_storage_len,
                        report.grid_clear_len,
                        report.grid_overflow_count
                    );
                }
                #[cfg(not(feature = "gpu_wgpu"))]
                {
                    return Err(std::io::Error::other(
                        "bench --gpu requires building burn_automata with --features gpu_wgpu",
                    )
                    .into());
                }
            } else if profile {
                let profile = profile_rollout(
                    &model,
                    &grid,
                    CpuProfileConfig {
                        particles,
                        steps,
                        seed_scale,
                        update_prob,
                        seed_mode,
                        geometry,
                    },
                )?;
                println!(
                    "particles={particles} steps={steps} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} perceive_ms={:.6} forward_ms={:.6} integrate_ms={:.6} final_mean_dx={:.6}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    profile.perceive_ms,
                    profile.forward_ms,
                    profile.integrate_ms,
                    profile.final_mean_dx
                );
            } else {
                let trace = run_rollout(
                    &model,
                    &grid,
                    &RolloutConfig {
                        steps,
                        particle_count: particles,
                        update_prob,
                        seed_scale,
                        ..RolloutConfig::default()
                    },
                    seed_mode,
                )?;
                println!(
                    "particles={particles} steps={steps} update_prob={update_prob:.3} elapsed_ms={} final_mean_dx={:.6}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    trace.mean_dx.last().copied().unwrap_or_default()
                );
            }
        }
        Command::BenchSpatial {
            preset,
            particles,
            seed_scale,
            normalize_seed_scale,
            fixed_eps,
            reference_seed_scale,
            seed_mode,
            geometry,
            strategy,
            bvh_leaf_size,
            tile_size,
        } => {
            let preset: AutomataPreset = preset.into();
            let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
            let seed_mode: ParticleSeed = seed_mode.into();
            let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
            let reference_seed_scale = reference_seed_scale
                .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
            let (config, base_grid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config.clone(), 42);
            let grid = if normalize_seed_scale {
                model
                    .config
                    .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
            } else {
                base_grid
            };
            let (positions, _states) = bench_particles(
                &model, &grid, particles, seed_scale, seed_mode, geometry, 42,
            );
            let strategies =
                spatial_strategies(strategy, &grid, parse_tile_size(&tile_size)?, bvh_leaf_size);
            for strategy in strategies {
                let started = Instant::now();
                match burn_automata::kernels::analyze_spatial_strategy(
                    &positions, 1, particles, &grid, strategy,
                ) {
                    Ok(report) => {
                        println!(
                            "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} dim={} eps={:.6} analyze_ms={:.6} active_bins={} max_bin_occupancy={} candidates_per_particle={:.6} entries_per_particle={:.6} exact_neighbors_per_particle={:.6} node_visits_per_particle={:.6} node_count={} max_depth={} exact_neighbor_pairs={} candidate_tests={} candidate_entries_visited={}",
                            strategy_label(report.strategy),
                            report.dim,
                            report.eps,
                            started.elapsed().as_secs_f64() * 1000.0,
                            report.active_bins,
                            report.max_bin_occupancy,
                            report.candidates_per_particle(),
                            report.entries_per_particle(),
                            report.exact_neighbors_per_particle(),
                            report.node_visits_per_particle(),
                            report.node_count,
                            report.max_depth,
                            report.exact_neighbor_pairs,
                            report.candidate_tests,
                            report.candidate_entries_visited,
                        );
                    }
                    Err(err) => {
                        println!(
                            "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} error=\"{}\"",
                            strategy_label(strategy),
                            err
                        );
                    }
                }
            }
        }
        Command::Manifest { preset, output } => {
            let preset: AutomataPreset = preset.into();
            let (config, hashgrid) = NpaConfig::for_preset(preset);
            let model = NpaModel::seeded(config, 42);
            let manifest = BpkModelManifest::from_model(
                &model,
                hashgrid,
                Some(format!("seeded-rust:{preset:?}")),
            );
            burn_automata::import::save_manifest(&output, &manifest)?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}

impl From<SeedModeArg> for ParticleSeed {
    fn from(value: SeedModeArg) -> Self {
        match value {
            SeedModeArg::Gaussian => Self::Gaussian,
            SeedModeArg::Uniform => Self::Uniform,
            SeedModeArg::UniformCircle => Self::UniformCircle,
            SeedModeArg::UvTorus3d => Self::UvTorus3d,
            SeedModeArg::UvTorusDense3d => Self::UvTorusDense3d,
            SeedModeArg::TorusFieldDense3d => Self::TorusFieldDense3d,
            SeedModeArg::TeapotFieldDense3d => Self::TeapotFieldDense3d,
            SeedModeArg::TorusGrowth3d => Self::TorusGrowth3d,
            SeedModeArg::TeapotGrowth3d => Self::TeapotGrowth3d,
            SeedModeArg::TorusSubstrateGrowth3d => Self::TorusSubstrateGrowth3d,
            SeedModeArg::TeapotSubstrateGrowth3d => Self::TeapotSubstrateGrowth3d,
            SeedModeArg::TorusMorphogenDense3d => Self::TorusMorphogenDense3d,
            SeedModeArg::TeapotMorphogenDense3d => Self::TeapotMorphogenDense3d,
        }
    }
}

const DEFAULT_GROWTH_TARGET_SEED: u64 = 42;
const UV_TORUS_FIELD_MOTION_GAIN: f32 = 8.0;
const UV_TORUS_FIELD_COLOR_GAIN: f32 = 0.16;
const UV_TORUS_FIELD_OPACITY_TARGET: f32 = 6.0;
const UV_TORUS_FIELD_OPACITY_GAIN: f32 = 0.10;
const UV_TORUS_FIELD_SCALE: f32 = 0.72;
const TEAPOT_FIELD_MOTION_GAIN: f32 = 1.0;
const TEAPOT_FIELD_COLOR_GAIN: f32 = 0.4;
const LOCAL_TORUS_MOTION_GAIN: f32 = 0.0;
const LOCAL_TEAPOT_MOTION_GAIN: f32 = 0.025;
const LOCAL_TORUS_COLOR_GAIN: f32 = 0.12;
const LOCAL_TEAPOT_COLOR_GAIN: f32 = 0.20;
const LOCAL_GROWTH_EXPANSION_GAIN: f32 = 0.05;
const LOCAL_GROWTH_OPACITY_GAIN: f32 = 2.0;
const LOCAL_GROWTH_MATERIAL_OPACITY_GAIN: f32 = 0.16;
#[cfg(test)]
const LOCAL_GROWTH_FRONT_OPACITY_GAIN: f32 = 0.18;
#[cfg(test)]
const LOCAL_GROWTH_FRONT_RADIUS: f32 = 0.22;
#[cfg(test)]
const LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE: f32 = 0.35;
#[cfg(test)]
const LOCAL_GROWTH_EXTENT_GAIN: f32 = 0.10;
const LOCAL_GROWTH_COORDINATE_GAIN: f32 = 0.10;
const LOCAL_GROWTH_ORIENTATION_GAIN: f32 = 0.12;
const LOCAL_GROWTH_SIGNED_DISTANCE_GAIN: f32 = 0.08;
const UV_TORUS_TARGET_RINGS: usize = 96;
const UV_TORUS_TARGET_TUBES: usize = 64;
const TORUS_ANGULAR_COVERAGE_RINGS: usize = 24;
const TORUS_ANGULAR_COVERAGE_TUBES: usize = 16;
const UV_TORUS_TARGET_SOURCE: &str = "uv-torus-3d:mesh-ovoxel-oriented-growth";
const UV_TORUS_POSITION_FIELD_TARGET_SOURCE: &str =
    "uv-torus-3d:neutral-seed-position-field-growth";
const UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE: &str =
    "uv-torus-3d:neutral-seed-rollout-position-field-growth";
const UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE: &str =
    "uv-torus-3d:mesh-ovoxel-oriented-seed-frame-morphogen-baseline";
const UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE: &str =
    "uv-torus-3d:rollout-local-mesh-objective-morphogen";
const UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-random-ball-rollout-ablation";
const UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE: &str =
    "uv-torus-3d:conditionless-local-substrate-rollout-ablation";
const TEAPOT_POSITION_FIELD_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-neutral-seed-position-field-growth";
const TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-neutral-seed-rollout-position-field-growth";
const TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-seed-frame-morphogen-baseline";
const TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE: &str =
    "utah-teapot-2026:canonical-mesh-rollout-local-mesh-objective-morphogen";
const TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-random-ball-rollout-ablation";
const TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE: &str =
    "utah-teapot-2026:conditionless-local-substrate-rollout-ablation";

fn uv_torus_mesh_target(scale: f32) -> TriangleMeshTarget {
    TriangleMeshTarget::torus(
        scale.max(1.0e-4),
        scale.max(1.0e-4) * UV_TORUS_MINOR_RATIO,
        UV_TORUS_TARGET_RINGS,
        UV_TORUS_TARGET_TUBES,
    )
    .expect("uv torus target mesh generation should be valid")
}

fn utah_teapot_mesh_target(scale: f32) -> TriangleMeshTarget {
    TriangleMeshTarget::utah_teapot(scale.max(1.0e-4))
        .expect("canonical Utah Teapot target mesh should be valid")
}

fn mesh_target_for_arg(target: MeshTargetArg, scale: f32) -> TriangleMeshTarget {
    match target {
        MeshTargetArg::Torus => uv_torus_mesh_target(scale),
        MeshTargetArg::Teapot => utah_teapot_mesh_target(scale),
    }
}

fn mesh_conditionless_local_target_source(target: MeshTargetArg) -> &'static str {
    match target {
        MeshTargetArg::Torus => UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE,
        MeshTargetArg::Teapot => TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    }
}

fn mesh_conditionless_local_target_source_for_seed(
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
        _ => mesh_conditionless_local_target_source(target),
    }
}

fn mesh_target_motion_gain(target: MeshTargetArg) -> f32 {
    match target {
        MeshTargetArg::Torus => LOCAL_TORUS_MOTION_GAIN,
        MeshTargetArg::Teapot => LOCAL_TEAPOT_MOTION_GAIN,
    }
}

fn mesh_target_color_gain(target: MeshTargetArg) -> f32 {
    match target {
        MeshTargetArg::Torus => LOCAL_TORUS_COLOR_GAIN,
        MeshTargetArg::Teapot => LOCAL_TEAPOT_COLOR_GAIN,
    }
}

fn conditionless_local_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    match target {
        MeshTargetArg::Torus => ParticleSeed::TorusSubstrateGrowth3d,
        MeshTargetArg::Teapot => ParticleSeed::TeapotSubstrateGrowth3d,
    }
}

fn conditionless_local_rollout_cases(
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
            seed: 0x10ca_101,
            seed_scale: (seed_scale * 0.5).max(1.0e-4),
            seed_mode,
        },
        MeshRolloutCaseConfig {
            particle_count: particles,
            steps: 64,
            seed: 0x10ca_102,
            seed_scale: seed_scale.max(1.0e-4),
            seed_mode,
        },
        MeshRolloutCaseConfig {
            particle_count: particles,
            steps: 64,
            seed: 0x10ca_103,
            seed_scale: (seed_scale * 1.5).max(1.0e-4),
            seed_mode,
        },
    ]
}

#[derive(Clone)]
struct RenderLossEvalConfig {
    particle_count: usize,
    steps: usize,
    seed: u64,
    extra_seeds: Vec<u64>,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    render: RenderLossConfig,
}

fn default_render_loss_config(seed_scale: f32) -> RenderLossConfig {
    RenderLossConfig {
        world_scale: seed_scale.max(1.0e-4) * 2.0,
        target_samples: 0,
        ..RenderLossConfig::default()
    }
}

fn mesh_render_loss_for_model(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderLossEvalConfig,
) -> Result<MultiViewRenderLossReport, Box<dyn std::error::Error>> {
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particle_count;
    }
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut reports = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let trace = run_rollout(
            model,
            grid,
            &RolloutConfig {
                particle_count: cfg.particle_count,
                steps: cfg.steps,
                update_prob: 1.0,
                seed,
                seed_scale: cfg.seed_scale,
                ..RolloutConfig::default()
            },
            cfg.seed_mode,
        )?;
        reports.push(mesh_multiview_render_loss_from_trace(
            &trace, target, render_cfg,
        )?);
    }
    Ok(average_render_loss_reports(reports, render_cfg))
}

fn eval_seed_list(seed: u64, extra_seeds: &[u64]) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(extra_seeds.len() + 1);
    seeds.push(seed);
    for &extra_seed in extra_seeds {
        if !seeds.contains(&extra_seed) {
            seeds.push(extra_seed);
        }
    }
    seeds
}

fn average_render_loss_reports(
    reports: Vec<MultiViewRenderLossReport>,
    cfg: RenderLossConfig,
) -> MultiViewRenderLossReport {
    let count = reports.len().max(1) as f32;
    let first = reports
        .first()
        .cloned()
        .unwrap_or_else(|| empty_render_loss_report(cfg));
    if reports.len() <= 1 {
        return first;
    }

    let views = (0..first.views.len())
        .map(|view_idx| {
            let view = first.views[view_idx].view;
            let view_reports: Vec<&RenderViewLossReport> = reports
                .iter()
                .filter_map(|report| report.views.get(view_idx))
                .collect();
            let view_count = view_reports.len().max(1) as f32;
            let density_mse = view_reports
                .iter()
                .map(|report| report.density_mse)
                .sum::<f32>()
                / view_count;
            let color_mse = view_reports
                .iter()
                .map(|report| report.color_mse)
                .sum::<f32>()
                / view_count;
            let depth_mse = view_reports
                .iter()
                .map(|report| report.depth_mse)
                .sum::<f32>()
                / view_count;
            let nonzero_target_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_target_alpha_fraction)
                .sum::<f32>()
                / view_count;
            let nonzero_particle_alpha_fraction = view_reports
                .iter()
                .map(|report| report.nonzero_particle_alpha_fraction)
                .sum::<f32>()
                / view_count;
            RenderViewLossReport {
                view,
                total_loss: cfg.density_weight * density_mse
                    + cfg.color_weight * color_mse
                    + cfg.depth_weight * depth_mse,
                density_mse,
                color_mse,
                depth_mse,
                density_psnr_db: render_psnr_db(density_mse, 1.0),
                color_psnr_db: render_psnr_db(color_mse, 1.0),
                depth_psnr_db: render_psnr_db(depth_mse, 1.0),
                nonzero_target_alpha_fraction,
                nonzero_particle_alpha_fraction,
            }
        })
        .collect::<Vec<_>>();

    let density_mse = reports.iter().map(|report| report.density_mse).sum::<f32>() / count;
    let color_mse = reports.iter().map(|report| report.color_mse).sum::<f32>() / count;
    let depth_mse = reports.iter().map(|report| report.depth_mse).sum::<f32>() / count;
    let density_psnr_db = render_psnr_db(density_mse, 1.0);
    let color_psnr_db = render_psnr_db(color_mse, 1.0);
    let depth_psnr_db = render_psnr_db(depth_mse, 1.0);
    let nonzero_target_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_target_alpha_fraction)
        .sum::<f32>()
        / count;
    let nonzero_particle_alpha_fraction = reports
        .iter()
        .map(|report| report.nonzero_particle_alpha_fraction)
        .sum::<f32>()
        / count;
    let finite = reports.iter().all(|report| {
        report.total_loss.is_finite()
            && report.density_mse.is_finite()
            && report.color_mse.is_finite()
            && report.depth_mse.is_finite()
            && report.nonzero_particle_alpha_fraction > 0.0
    });
    MultiViewRenderLossReport {
        passed: finite
            && reports.iter().all(|report| report.passed)
            && density_psnr_db >= 10.0
            && color_psnr_db >= 12.0
            && depth_psnr_db >= 14.0,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: cfg.density_weight * density_mse
            + cfg.color_weight * color_mse
            + cfg.depth_weight * depth_mse,
        density_mse,
        color_mse,
        depth_mse,
        density_psnr_db,
        color_psnr_db,
        depth_psnr_db,
        nonzero_target_alpha_fraction,
        nonzero_particle_alpha_fraction,
        views,
    }
}

fn render_psnr_db(mse: f32, max_value: f32) -> f32 {
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * ((max_value * max_value) / mse.max(1.0e-8)).log10()
    }
}

fn empty_render_loss_report(cfg: RenderLossConfig) -> MultiViewRenderLossReport {
    MultiViewRenderLossReport {
        passed: false,
        image_size: cfg.image_size,
        target_samples: cfg.target_samples,
        total_loss: f32::INFINITY,
        density_mse: f32::INFINITY,
        color_mse: f32::INFINITY,
        depth_mse: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        nonzero_target_alpha_fraction: 0.0,
        nonzero_particle_alpha_fraction: 0.0,
        views: Vec::new(),
    }
}

#[derive(Clone)]
struct RenderProxyTrainingConfig {
    target: MeshTargetArg,
    rounds: usize,
    supervised_steps_per_round: usize,
    particles: usize,
    rollout_steps: usize,
    gradient_particles: usize,
    gradient_mode: RenderGradientModeArg,
    finite_diff_eps: f32,
    motion_gain: f32,
    perception_position_gain: f32,
    max_update_norm: f32,
    trajectory_supervision: bool,
    trajectory_render_gain: f32,
    trajectory_render_samples: usize,
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    full_coverage_adjoint: bool,
    surface_gain: f32,
    opacity_gain: f32,
    max_opacity_update: f32,
    direct_line_search: bool,
    direct_line_search_scales: Vec<f32>,
    direct_material_output_only: bool,
    training_backend: RenderTrainingBackendArg,
    direct_selection_seed_training: bool,
    seed: u64,
    selection_seed: Option<u64>,
    selection_seeds: Vec<u64>,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    render: RenderLossConfig,
    sgd: SgdConfig,
}

fn render_training_default_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    match target {
        MeshTargetArg::Torus => ParticleSeed::TorusGrowth3d,
        MeshTargetArg::Teapot => ParticleSeed::TeapotGrowth3d,
    }
}

fn render_training_base_model(
    target: MeshTargetArg,
    target_mesh: &TriangleMeshTarget,
    seed_mode: ParticleSeed,
) -> Result<(NpaModel, String), Box<dyn std::error::Error>> {
    if !target_local_growth_seed(target, seed_mode) {
        return Err(std::io::Error::other(format!(
            "default render training base requires a target local growth seed; got seed_mode={seed_mode:?}"
        ))
        .into());
    }
    let model = local_growth_student_model_with_axis_gains(
        NpaConfig::growing_3dgs(),
        0x5a17_3d,
        0.0,
        mesh_axis_expansion_gains(target_mesh, LOCAL_GROWTH_EXPANSION_GAIN),
    )?;
    let source = format!(
        "ablation-rust:{}",
        mesh_conditionless_local_target_source_for_seed(target, seed_mode)
    );
    Ok((model, source))
}

fn render_training_seed_mode(target: MeshTargetArg) -> ParticleSeed {
    match target {
        MeshTargetArg::Torus => ParticleSeed::TorusFieldDense3d,
        MeshTargetArg::Teapot => ParticleSeed::TeapotFieldDense3d,
    }
}

fn default_render_training_seed_mode(target: MeshTargetArg, model: &NpaModel) -> ParticleSeed {
    if model.config.position_features {
        render_training_seed_mode(target)
    } else {
        conditionless_local_seed_mode(target)
    }
}

fn render_proxy_selection_seeds(cfg: &RenderProxyTrainingConfig) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(cfg.selection_seeds.len() + 2);
    seeds.push(cfg.seed);
    if let Some(selection_seed) = cfg.selection_seed {
        if !seeds.contains(&selection_seed) {
            seeds.push(selection_seed);
        }
    }
    for &selection_seed in &cfg.selection_seeds {
        if !seeds.contains(&selection_seed) {
            seeds.push(selection_seed);
        }
    }
    seeds
}

fn render_training_source(
    target: MeshTargetArg,
    base_source: Option<&str>,
    seed_mode: ParticleSeed,
) -> String {
    let local_growth_seed = matches!(
        seed_mode,
        ParticleSeed::TorusGrowth3d
            | ParticleSeed::TeapotGrowth3d
            | ParticleSeed::TorusSubstrateGrowth3d
            | ParticleSeed::TeapotSubstrateGrowth3d
    );
    if let Some(source) = base_source {
        if source.starts_with("render-refined-rust:") && local_growth_seed {
            return source.to_string();
        }
        if source.contains("conditionless-local") && local_growth_seed {
            return format!("render-refined-rust:{source}");
        }
        return format!("render-proxy-rust:{target:?}:base={source}:seed={seed_mode:?}");
    }
    format!("render-proxy-rust:{target:?}:field-baseline")
}

fn is_catalog_model_output_path(path: &Path) -> bool {
    let parts = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>();
    parts
        .windows(2)
        .any(|window| window[0] == "assets" && window[1] == "models")
}

fn catalog_bound_candidate_path(target: MeshTargetArg, process_id: u32) -> PathBuf {
    let target_label = match target {
        MeshTargetArg::Torus => "torus",
        MeshTargetArg::Teapot => "teapot",
    };
    PathBuf::from("target").join(format!(
        "catalog_{target_label}_render3d_candidate_{process_id}.bpk"
    ))
}

fn target_local_growth_seed(target: MeshTargetArg, seed_mode: ParticleSeed) -> bool {
    matches!(
        (target, seed_mode),
        (MeshTargetArg::Torus, ParticleSeed::TorusGrowth3d)
            | (MeshTargetArg::Teapot, ParticleSeed::TeapotGrowth3d)
            | (MeshTargetArg::Torus, ParticleSeed::TorusSubstrateGrowth3d)
            | (MeshTargetArg::Teapot, ParticleSeed::TeapotSubstrateGrowth3d)
    )
}

fn validate_diagnostic_3d_output_not_catalog(
    model_output: &Path,
    command: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if is_catalog_model_output_path(model_output) {
        return Err(std::io::Error::other(format!(
            "{command} writes diagnostic 3D artifacts and refuses catalog-bound output {}; write to target/ or artifacts/ and promote only after validate_3d_catalog.py passes",
            model_output.display()
        ))
        .into());
    }
    Ok(())
}

fn validate_catalog_bound_render_training_output(
    model_output: &Path,
    target: MeshTargetArg,
    seed_mode: ParticleSeed,
    base_source: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !is_catalog_model_output_path(model_output) {
        return Ok(());
    }
    if !target_local_growth_seed(target, seed_mode) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires the target local growth seed; got seed_mode={seed_mode:?}",
            model_output.display()
        ))
        .into());
    }
    let source = base_source.unwrap_or_default();
    if !local_conditionless_lineage(source) {
        return Err(std::io::Error::other(format!(
            "catalog-bound 3D render training output {} requires a conditionless-local base model; source={source:?}",
            model_output.display()
        ))
        .into());
    }
    Ok(())
}

fn local_conditionless_lineage(source: &str) -> bool {
    source.contains("conditionless-local")
        && !source.contains("position-field")
        && !source.contains("seed-frame")
        && !source.contains("render-proxy-rust")
}

fn load_conditionless_local_base_model(
    path: &Path,
    target_source: &str,
) -> Result<(NpaModel, burn_automata::kernels::HashGridConfig, String), Box<dyn std::error::Error>>
{
    let manifest = burn_automata::import::load_manifest(path)?;
    if manifest.config.spatial_dims != 3 || manifest.config.state_dims <= 3 {
        return Err(std::io::Error::other(format!(
            "local 3D continuation requires spatial_dims=3 and state_dims>3; got spatial_dims={} state_dims={}",
            manifest.config.spatial_dims, manifest.config.state_dims
        ))
        .into());
    }
    if manifest.config.position_features {
        return Err(std::io::Error::other(format!(
            "local 3D continuation rejects position-feature base model {}",
            path.display()
        ))
        .into());
    }
    let source_text = manifest.source.as_deref().unwrap_or_default();
    if !local_conditionless_lineage(source_text) {
        return Err(std::io::Error::other(format!(
            "local 3D continuation rejects shortcut lineage for {}: source={source_text:?}",
            path.display()
        ))
        .into());
    }
    let source = format!("ablation-rust:{target_source}:continued-from={source_text}");
    let hashgrid = manifest.hashgrid.clone();
    Ok((manifest.into_model(), hashgrid, source))
}

fn run_render_proxy_training(
    model: &mut NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderProxyTrainingConfig,
) -> Result<RenderProxyTrainingReport, Box<dyn std::error::Error>> {
    if cfg.rounds == 0 || cfg.supervised_steps_per_round == 0 {
        return Err(std::io::Error::other(
            "render-proxy training requires non-zero rounds and supervised steps",
        )
        .into());
    }
    if !cfg.finite_diff_eps.is_finite() || cfg.finite_diff_eps <= 0.0 {
        return Err(std::io::Error::other("finite_diff_eps must be positive and finite").into());
    }
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let initial_trace = render_training_trace(model, grid, &cfg, 0)?;
    let initial_render_loss =
        mesh_multiview_render_loss_from_trace(&initial_trace, target, render_cfg)?;
    let selection_baseline = render_selection_baseline(model, grid, target, &cfg, render_cfg)?;
    let initial_selection = render_selection_metrics(
        model,
        grid,
        target,
        &cfg,
        render_cfg,
        Some(&selection_baseline),
    )?;
    let mut best_model = model.clone();
    let mut best_render_loss = initial_render_loss.clone();
    let mut best_selection_score = initial_selection.score;
    let mut best_selection_render_loss = initial_selection.render_loss;
    let mut best_selection_density_psnr_db = initial_selection.density_psnr_db;
    let mut selected_round = None;
    let mut history = Vec::with_capacity(cfg.rounds);

    for round in 0..cfg.rounds {
        let needs_trajectory = cfg.trajectory_supervision
            || cfg.training_backend == RenderTrainingBackendArg::DirectRollout;
        let (trace, trajectory) = if needs_trajectory {
            render_training_trajectory(model, grid, &cfg, round)?
        } else {
            (render_training_trace(model, grid, &cfg, round)?, Vec::new())
        };
        let before = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
        let gradient = render_position_gradient(&trace, target, render_cfg, &cfg)?;
        let gradient_rms = (gradient
            .gradients
            .iter()
            .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
            .sum::<f32>()
            / gradient.gradients.len().max(1) as f32)
            .sqrt();
        let opacity_gradient_rms = (gradient
            .opacity_gradients
            .iter()
            .map(|gradient| gradient * gradient)
            .sum::<f32>()
            / gradient.opacity_gradients.len().max(1) as f32)
            .sqrt();
        let (train_report, train_step_scale) = match cfg.training_backend {
            RenderTrainingBackendArg::Proxy => {
                let batch = render_proxy_supervised_batch(
                    model,
                    grid,
                    target,
                    &trace,
                    &trajectory,
                    &gradient,
                    &cfg,
                )?;
                (
                    run_supervised_training(
                        model,
                        &batch,
                        TrainingRunConfig {
                            steps: cfg.supervised_steps_per_round,
                            report_interval: cfg.supervised_steps_per_round,
                            sgd: cfg.sgd,
                        },
                    )?,
                    1.0,
                )
            }
            RenderTrainingBackendArg::DirectRollout => {
                if cfg.direct_line_search {
                    render_direct_rollout_training_step_with_line_search(
                        model,
                        grid,
                        target,
                        &cfg,
                        round,
                        &trace,
                        &trajectory,
                        &gradient,
                        render_cfg,
                        &selection_baseline,
                    )?
                } else {
                    let report = if cfg.direct_selection_seed_training {
                        render_direct_rollout_multiseed_training_step(
                            model,
                            grid,
                            target,
                            &cfg,
                            round,
                            &trace,
                            &trajectory,
                            &gradient,
                        )?
                    } else {
                        render_direct_rollout_training_step(
                            model,
                            grid,
                            target,
                            &trace,
                            &trajectory,
                            &gradient,
                            &cfg,
                        )?
                    };
                    (report, 1.0)
                }
            }
        };
        let after_trace = render_training_trace(model, grid, &cfg, round)?;
        let after = mesh_multiview_render_loss_from_trace(&after_trace, target, render_cfg)?;
        let selection = render_selection_metrics(
            model,
            grid,
            target,
            &cfg,
            render_cfg,
            Some(&selection_baseline),
        )?;
        if render_selection_candidate_beats(
            selection.score,
            best_selection_score,
            selection.morphology_non_regressed,
            selection.render_loss,
            best_selection_render_loss,
            selection.density_psnr_db,
            best_selection_density_psnr_db,
        ) {
            best_model = model.clone();
            best_render_loss = selection.base_report.clone();
            best_selection_score = selection.score;
            best_selection_render_loss = selection.render_loss;
            best_selection_density_psnr_db = selection.density_psnr_db;
            selected_round = Some(round);
        }
        history.push(RenderProxyTrainingHistoryEntry {
            round,
            before_loss: before.total_loss,
            after_loss: after.total_loss,
            selection_loss: selection.render_loss,
            selection_score: selection.score,
            before_density_psnr_db: before.density_psnr_db,
            after_density_psnr_db: after.density_psnr_db,
            selection_density_psnr_db: selection.density_psnr_db,
            selection_active_surface_max: selection.active_surface_max,
            selection_target_coverage_fraction: selection.target_coverage_fraction,
            selection_morphology_non_regressed: selection.morphology_non_regressed,
            selection_worst_seed: selection.worst_seed,
            selection_worst_failure_reasons: selection.worst_failure_reasons,
            before_color_psnr_db: before.color_psnr_db,
            after_color_psnr_db: after.color_psnr_db,
            before_depth_psnr_db: before.depth_psnr_db,
            after_depth_psnr_db: after.depth_psnr_db,
            supervised_loss: train_report.final_loss,
            train_grad_norm: train_report
                .history
                .last()
                .map(|entry| entry.grad_norm)
                .unwrap_or(0.0),
            train_grad_scale: train_report
                .history
                .last()
                .map(|entry| entry.grad_scale)
                .unwrap_or(1.0),
            train_step_scale,
            gradient_rms,
            opacity_gradient_rms,
        });
    }
    *model = best_model;
    let final_render_loss = best_render_loss;

    Ok(RenderProxyTrainingReport {
        rounds: cfg.rounds,
        supervised_steps_per_round: cfg.supervised_steps_per_round,
        gradient_particles: cfg.gradient_particles,
        gradient_mode: cfg.gradient_mode,
        finite_diff_eps: cfg.finite_diff_eps,
        motion_gain: cfg.motion_gain,
        perception_position_gain: cfg.perception_position_gain,
        max_update_norm: cfg.max_update_norm,
        trajectory_supervision: cfg.trajectory_supervision,
        trajectory_render_gain: cfg.trajectory_render_gain,
        trajectory_render_samples: cfg.trajectory_render_samples,
        coverage_gain: cfg.coverage_gain,
        coverage_samples: cfg.coverage_samples,
        coverage_mode: cfg.coverage_mode,
        coverage_softness: cfg.coverage_softness,
        coverage_repulsion_gain: cfg.coverage_repulsion_gain,
        coverage_gap_gain: cfg.coverage_gap_gain,
        coverage_repulsion_radius: cfg.coverage_repulsion_radius,
        coverage_normal_weight: cfg.coverage_normal_weight,
        full_coverage_adjoint: cfg.full_coverage_adjoint,
        surface_gain: cfg.surface_gain,
        opacity_gain: cfg.opacity_gain,
        max_opacity_update: cfg.max_opacity_update,
        direct_line_search: cfg.direct_line_search,
        direct_line_search_scales: sanitized_direct_line_search_scales(&cfg),
        direct_material_output_only: cfg.direct_material_output_only,
        training_backend: cfg.training_backend,
        direct_selection_seed_training: cfg.direct_selection_seed_training,
        selection_seed: cfg.selection_seed,
        selection_seeds: render_proxy_selection_seeds(&cfg),
        initial_render_loss,
        final_render_loss,
        selected_round,
        history,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_direct_rollout_training_step_with_line_search(
    model: &mut NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    trace: &burn_automata::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    render_cfg: RenderLossConfig,
    selection_baseline: &[RenderSelectionBaselineCase],
) -> Result<(TrainingRunReport, f32), Box<dyn std::error::Error>> {
    let scales = sanitized_direct_line_search_scales(cfg);
    if scales.is_empty() {
        let report = if cfg.direct_selection_seed_training {
            render_direct_rollout_multiseed_training_step(
                model, grid, target, cfg, round, trace, trajectory, gradient,
            )?
        } else {
            render_direct_rollout_training_step(
                model, grid, target, trace, trajectory, gradient, cfg,
            )?
        };
        return Ok((report, 1.0));
    }

    let base_model = model.clone();
    let initial_loss = mesh_multiview_render_loss_from_trace(trace, target, render_cfg)?.total_loss;
    let no_op_selection = render_selection_metrics(
        model,
        grid,
        target,
        cfg,
        render_cfg,
        Some(selection_baseline),
    )?;
    let mut best_model = base_model.clone();
    let mut best_report = render_direct_rollout_noop_report(initial_loss, gradient);
    let mut best_score = no_op_selection.score;
    let mut best_render_loss = no_op_selection.render_loss;
    let mut best_density_psnr_db = no_op_selection.density_psnr_db;
    let mut best_scale = 0.0_f32;
    let mut best_morphology_non_regressed = no_op_selection.morphology_non_regressed;

    for scale in scales {
        let scaled_learning_rate = cfg.sgd.learning_rate * scale;
        if !scaled_learning_rate.is_finite() {
            continue;
        }
        let mut candidate_cfg = cfg.clone();
        candidate_cfg.direct_line_search = false;
        candidate_cfg.sgd.learning_rate = scaled_learning_rate;
        let mut candidate = base_model.clone();
        let report = if cfg.direct_selection_seed_training {
            render_direct_rollout_multiseed_training_step(
                &mut candidate,
                grid,
                target,
                &candidate_cfg,
                round,
                trace,
                trajectory,
                gradient,
            )?
        } else {
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                trace,
                trajectory,
                gradient,
                &candidate_cfg,
            )?
        };
        let selection = render_selection_metrics(
            &candidate,
            grid,
            target,
            cfg,
            render_cfg,
            Some(selection_baseline),
        )?;
        let candidate_beats = render_selection_candidate_beats(
            selection.score,
            best_score,
            selection.morphology_non_regressed,
            selection.render_loss,
            best_render_loss,
            selection.density_psnr_db,
            best_density_psnr_db,
        );
        let render_non_regressed = render_selection_render_non_regressed(
            selection.render_loss,
            best_render_loss,
            selection.density_psnr_db,
            best_density_psnr_db,
        );
        if candidate_beats
            || (best_scale == 0.0
                && !best_morphology_non_regressed
                && selection.morphology_non_regressed
                && render_non_regressed
                && selection.score.is_finite())
        {
            best_model = candidate;
            best_report = report;
            best_score = selection.score;
            best_render_loss = selection.render_loss;
            best_density_psnr_db = selection.density_psnr_db;
            best_scale = scale;
            best_morphology_non_regressed = selection.morphology_non_regressed;
        }
    }

    *model = best_model;
    Ok((best_report, best_scale))
}

fn sanitized_direct_line_search_scales(cfg: &RenderProxyTrainingConfig) -> Vec<f32> {
    if !cfg.direct_line_search {
        return Vec::new();
    }
    let mut scales = Vec::with_capacity(cfg.direct_line_search_scales.len());
    for &scale in &cfg.direct_line_search_scales {
        if scale.is_finite() && scale > 0.0 && !scales.contains(&scale) {
            scales.push(scale);
        }
    }
    if scales.is_empty() {
        scales.push(1.0);
    }
    scales
}

fn render_direct_rollout_noop_report(
    initial_loss: f32,
    gradient: &RenderProxyGradientRows,
) -> TrainingRunReport {
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.color_gradients.len());
    TrainingRunReport {
        steps: 0,
        rows,
        initial_loss,
        final_loss: initial_loss,
        best_loss: initial_loss,
        history: vec![TrainingHistoryEntry {
            step: 0,
            loss: initial_loss,
            grad_norm: 0.0,
            grad_scale: 0.0,
        }],
    }
}

fn render_training_trace(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
) -> Result<burn_automata::RolloutTrace, Box<dyn std::error::Error>> {
    render_training_trace_for_seed(
        model,
        grid,
        cfg,
        cfg.seed
            .wrapping_add((round as u64).wrapping_mul(0x9e37_79b9)),
    )
}

#[derive(Clone, Debug)]
struct RenderTrajectorySnapshot {
    positions: Vec<[f32; 4]>,
    states: Vec<f32>,
    features: Vec<f32>,
    step_fraction: f32,
}

fn render_training_trajectory(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
) -> Result<(burn_automata::RolloutTrace, Vec<RenderTrajectorySnapshot>), Box<dyn std::error::Error>>
{
    let seed = cfg
        .seed
        .wrapping_add((round as u64).wrapping_mul(0x9e37_79b9));
    render_training_trajectory_for_seed(model, grid, cfg, seed)
}

fn render_training_trajectory_for_seed(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
) -> Result<(burn_automata::RolloutTrace, Vec<RenderTrajectorySnapshot>), Box<dyn std::error::Error>>
{
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particles,
        steps: cfg.rollout_steps,
        update_prob: 1.0,
        seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (mut positions, mut states) = seed_particles_scaled(
        rollout_cfg.batch_size,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );
    let mut mean_dx = Vec::with_capacity(rollout_cfg.steps);
    let mut snapshots = Vec::with_capacity(rollout_cfg.steps);

    for step_idx in 0..rollout_cfg.steps {
        let step = model.step_cpu(
            &positions,
            &states,
            rollout_cfg.batch_size,
            rollout_cfg.particle_count,
            grid,
            rollout_cfg.dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        snapshots.push(RenderTrajectorySnapshot {
            positions: positions.clone(),
            states: states.clone(),
            features: step.perception.features.clone(),
            step_fraction: (step_idx + 1) as f32 / rollout_cfg.steps.max(1) as f32,
        });
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok((
        burn_automata::RolloutTrace {
            positions,
            states,
            batch_size: rollout_cfg.batch_size,
            particle_count: rollout_cfg.particle_count,
            state_dims: model.config.state_dims,
            steps: rollout_cfg.steps,
            mean_dx,
        },
        snapshots,
    ))
}

fn render_training_trace_for_seed(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RenderProxyTrainingConfig,
    seed: u64,
) -> Result<burn_automata::RolloutTrace, Box<dyn std::error::Error>> {
    Ok(run_rollout(
        model,
        grid,
        &RolloutConfig {
            particle_count: cfg.particles,
            steps: cfg.rollout_steps,
            update_prob: 1.0,
            seed,
            seed_scale: cfg.seed_scale,
            ..RolloutConfig::default()
        },
        cfg.seed_mode,
    )?)
}

struct RenderSelectionMetrics {
    render_loss: f32,
    score: f32,
    density_psnr_db: f32,
    active_surface_max: f32,
    target_coverage_fraction: f32,
    morphology_non_regressed: bool,
    worst_seed: u64,
    worst_failure_reasons: Vec<&'static str>,
    base_report: MultiViewRenderLossReport,
}

#[derive(Clone, Copy)]
struct RenderSelectionBaselineCase {
    seed: u64,
    active_surface_max: f32,
    target_coverage_fraction: f32,
}

fn render_selection_metrics(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
    baseline: Option<&[RenderSelectionBaselineCase]>,
) -> Result<RenderSelectionMetrics, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let base_seed = selection_seeds[0];
    let base_case = render_selection_case_metrics(model, grid, target, cfg, render_cfg, base_seed)?;
    let mut render_loss = 0.0_f32;
    let mut morphology_non_regressed = true;
    let mut score = f32::NEG_INFINITY;
    let mut worst_seed = base_seed;
    let mut worst_failure_reasons = Vec::new();
    let mut density_psnr_db = 0.0_f32;
    let mut active_surface_max = f32::NEG_INFINITY;
    let mut target_coverage_fraction = f32::INFINITY;
    for seed in &selection_seeds {
        let owned_case;
        let selection_case = if *seed == base_seed {
            &base_case
        } else {
            owned_case =
                render_selection_case_metrics(model, grid, target, cfg, render_cfg, *seed)?;
            &owned_case
        };
        render_loss += selection_case.render_loss.total_loss;
        let selection_score =
            render_selection_case_score_with_baseline(*seed, selection_case, baseline);
        if !selection_score.morphology_non_regressed {
            morphology_non_regressed = false;
        }
        if selection_score.score > score {
            worst_seed = *seed;
            worst_failure_reasons = selection_case.failure_reasons.clone();
            if !selection_score.morphology_non_regressed {
                worst_failure_reasons.push("selection_morphology_non_regressed");
            }
        }
        score = score.max(selection_score.score);
        density_psnr_db += selection_case.render_loss.density_psnr_db;
        active_surface_max = active_surface_max.max(selection_case.active_surface.max_distance);
        target_coverage_fraction =
            target_coverage_fraction.min(selection_case.target_coverage.covered_fraction);
    }
    let count = selection_seeds.len().max(1) as f32;

    Ok(RenderSelectionMetrics {
        render_loss: finite_report_metric(render_loss / count, RENDER_SELECTION_BAD_SCORE),
        score: finite_report_metric(score, RENDER_SELECTION_BAD_SCORE),
        density_psnr_db: finite_report_metric(density_psnr_db / count, -RENDER_SELECTION_BAD_SCORE),
        active_surface_max: finite_report_metric(active_surface_max, RENDER_SELECTION_BAD_SCORE),
        target_coverage_fraction: finite_report_metric(target_coverage_fraction, 0.0),
        morphology_non_regressed,
        worst_seed,
        worst_failure_reasons,
        base_report: base_case.render_loss,
    })
}

fn render_selection_candidate_beats(
    selection_score: f32,
    best_selection_score: f32,
    morphology_non_regressed: bool,
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    morphology_non_regressed
        && selection_score < best_selection_score
        && render_selection_render_non_regressed(
            selection_render_loss,
            best_render_loss,
            selection_density_psnr_db,
            best_density_psnr_db,
        )
}

fn render_selection_render_non_regressed(
    selection_render_loss: f32,
    best_render_loss: f32,
    selection_density_psnr_db: f32,
    best_density_psnr_db: f32,
) -> bool {
    const LOSS_TOLERANCE: f32 = 1.0e-5;
    const DENSITY_PSNR_TOLERANCE_DB: f32 = 1.0e-4;
    selection_render_loss.is_finite()
        && best_render_loss.is_finite()
        && selection_density_psnr_db.is_finite()
        && best_density_psnr_db.is_finite()
        && selection_render_loss <= best_render_loss + LOSS_TOLERANCE
        && selection_density_psnr_db + DENSITY_PSNR_TOLERANCE_DB >= best_density_psnr_db
}

fn finite_report_metric(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn render_selection_baseline(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
) -> Result<Vec<RenderSelectionBaselineCase>, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let mut baselines = Vec::with_capacity(selection_seeds.len());
    for seed in selection_seeds {
        let selection_case =
            render_selection_case_metrics(model, grid, target, cfg, render_cfg, seed)?;
        baselines.push(RenderSelectionBaselineCase {
            seed,
            active_surface_max: selection_case.active_surface.max_distance,
            target_coverage_fraction: selection_case.target_coverage.covered_fraction,
        });
    }
    Ok(baselines)
}

fn render_selection_case_score_with_baseline(
    seed: u64,
    case: &RenderSelectionCaseMetrics,
    baseline: Option<&[RenderSelectionBaselineCase]>,
) -> RenderSelectionCaseScore {
    let mut score = finite_report_metric(case.score, RENDER_SELECTION_BAD_SCORE);
    let mut morphology_non_regressed = true;
    if let Some(baseline_case) = baseline.and_then(|cases| {
        cases
            .iter()
            .find(|baseline_case| baseline_case.seed == seed)
    }) {
        let surface_regression = if case.active_surface.max_distance.is_finite() {
            (case.active_surface.max_distance - baseline_case.active_surface_max - 0.02).max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        let coverage_regression = if case.target_coverage.covered_fraction.is_finite() {
            (baseline_case.target_coverage_fraction - case.target_coverage.covered_fraction - 0.02)
                .max(0.0)
        } else {
            RENDER_SELECTION_BAD_SCORE
        };
        if surface_regression > 0.0 || coverage_regression > 0.0 {
            morphology_non_regressed = false;
        }
        score += (surface_regression + coverage_regression) * 10.0;
    }
    RenderSelectionCaseScore {
        score,
        morphology_non_regressed,
    }
}

struct RenderSelectionCaseScore {
    score: f32,
    morphology_non_regressed: bool,
}

struct RenderSelectionCaseMetrics {
    render_loss: MultiViewRenderLossReport,
    active_surface: Growth3dSurfaceStats,
    target_coverage: TargetCoverageStats,
    score: f32,
    failure_reasons: Vec<&'static str>,
}

fn render_selection_case_metrics(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
    seed: u64,
) -> Result<RenderSelectionCaseMetrics, Box<dyn std::error::Error>> {
    let trace = render_training_trace_for_seed(model, grid, cfg, seed)?;
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particles,
        steps: cfg.rollout_steps,
        update_prob: 1.0,
        seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );
    let mut active_seed_count = 0usize;
    let mut seed_active = Vec::with_capacity(rollout_cfg.particle_count);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }
    let non_opacity_seed_abs_max =
        growth_3d_non_scaffold_seed_abs_max(model.config.state_dims, cfg.seed_mode, &seed_states);
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
    );
    let active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, target);
    let active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let coverage_samples = cfg.particles.max(512);
    let coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let initial_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        target,
        coverage_samples,
        coverage_threshold,
    );
    let torus_angular_coverage = (cfg.target == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let temporal = growth_3d_temporal_report(
        model,
        grid,
        target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        coverage_threshold,
    )?;
    let permutation_consistency =
        growth_3d_permutation_report(model, grid, &rollout_cfg, cfg.seed_mode)?;
    let front = growth_3d_front_report(
        model,
        grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let strict_checks = growth_3d_strict_checks_report(
        model.config.position_features,
        true,
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        torus_angular_coverage.as_ref(),
        &motion,
        &front,
        &temporal,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particles,
        render_loss.passed,
    );
    let strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        active_surface,
        active_surface_tail,
        initial_target_coverage,
        target_coverage,
        cfg.seed_scale,
        &render_loss,
    );
    let score = strict_score.score;
    let failure_reasons = strict_checks.failure_reasons.clone();
    Ok(RenderSelectionCaseMetrics {
        render_loss,
        active_surface,
        target_coverage,
        score,
        failure_reasons,
    })
}

#[derive(Clone)]
struct RenderProxyGradientRows {
    row_indices: Vec<usize>,
    gradients: Vec<[f32; 3]>,
    opacity_gradients: Vec<f32>,
    color_gradients: Vec<[f32; 3]>,
}

struct RenderTrajectoryAdjoint {
    state: Vec<f32>,
    position: Vec<[f32; 4]>,
    weight: f32,
}

fn spread_row_indices(items: usize, max_rows: usize) -> Vec<usize> {
    let rows = items.min(max_rows).max(1);
    if rows >= items {
        return (0..items).collect();
    }
    (0..rows)
        .map(|idx| (idx * items / rows).min(items - 1))
        .collect()
}

fn trajectory_render_sample_indices(len: usize, max_samples: usize) -> Vec<usize> {
    let samples = len.min(max_samples);
    if samples == 0 {
        return Vec::new();
    }
    let mut indices = Vec::with_capacity(samples);
    for sample in 0..samples {
        let index = ((sample + 1) * len / samples).saturating_sub(1);
        if indices.last().copied() != Some(index) {
            indices.push(index);
        }
    }
    indices
}

fn render_proxy_gradient_row_indices(particles: usize, max_rows: usize) -> Vec<usize> {
    spread_row_indices(particles, max_rows)
}

fn render_position_gradient(
    trace: &burn_automata::RolloutTrace,
    target: &TriangleMeshTarget,
    render_cfg: RenderLossConfig,
    cfg: &RenderProxyTrainingConfig,
) -> Result<RenderProxyGradientRows, Box<dyn std::error::Error>> {
    let row_indices =
        render_proxy_gradient_row_indices(trace.particle_count, cfg.gradient_particles);
    match cfg.gradient_mode {
        RenderGradientModeArg::Analytic => {
            let report = mesh_multiview_render_position_gradient_for_rows_from_trace(
                trace,
                target,
                render_cfg,
                &row_indices,
            )?;
            Ok(RenderProxyGradientRows {
                row_indices: report.row_indices,
                gradients: report.gradients,
                opacity_gradients: report.opacity_gradients,
                color_gradients: report.color_gradients,
            })
        }
        RenderGradientModeArg::FiniteDiff => {
            let mut gradient = vec![[0.0; 3]; row_indices.len()];
            let mut opacity_gradient = vec![0.0; row_indices.len()];
            let color_gradient = vec![[0.0; 3]; row_indices.len()];
            let eps = cfg.finite_diff_eps;
            for (gradient_idx, &row) in row_indices.iter().enumerate() {
                for axis in 0..3 {
                    let plus = trace_with_position_delta(trace, row, axis, eps);
                    let minus = trace_with_position_delta(trace, row, axis, -eps);
                    let plus_loss =
                        mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                            .total_loss;
                    let minus_loss =
                        mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                            .total_loss;
                    gradient[gradient_idx][axis] = (plus_loss - minus_loss) / (2.0 * eps);
                }
                if let Some(opacity_channel) = growth_3d_material_opacity_channel(trace.state_dims)
                {
                    let plus = trace_with_state_delta(trace, row, opacity_channel, eps);
                    let minus = trace_with_state_delta(trace, row, opacity_channel, -eps);
                    let plus_loss =
                        mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                            .total_loss;
                    let minus_loss =
                        mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                            .total_loss;
                    let state_logit = trace.states[row * trace.state_dims + opacity_channel]
                        + render_cfg.opacity_logit_bias;
                    let derivative = sigmoid_unit_derivative(state_logit);
                    if derivative > 1.0e-6 {
                        opacity_gradient[gradient_idx] =
                            (plus_loss - minus_loss) / (2.0 * eps * derivative);
                    }
                }
            }
            Ok(RenderProxyGradientRows {
                row_indices,
                gradients: gradient,
                opacity_gradients: opacity_gradient,
                color_gradients: color_gradient,
            })
        }
    }
}

fn trace_with_position_delta(
    trace: &burn_automata::RolloutTrace,
    row: usize,
    axis: usize,
    delta: f32,
) -> burn_automata::RolloutTrace {
    let mut perturbed = trace.clone();
    perturbed.positions[row][axis] += delta;
    perturbed
}

fn trace_with_state_delta(
    trace: &burn_automata::RolloutTrace,
    row: usize,
    channel: usize,
    delta: f32,
) -> burn_automata::RolloutTrace {
    let mut perturbed = trace.clone();
    let index = row * trace.state_dims + channel;
    if index < perturbed.states.len() {
        perturbed.states[index] += delta;
    }
    perturbed
}

fn render_proxy_supervised_batch(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    trace: &burn_automata::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    cfg: &RenderProxyTrainingConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.color_gradients.len());
    if rows == 0 {
        return Err(std::io::Error::other("render proxy gradient produced no rows").into());
    }
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();

    let mut features = Vec::new();
    let mut states = Vec::new();
    let mut positions = Vec::new();
    let mut gradient_rows = Vec::new();
    let mut weights = Vec::new();

    if cfg.trajectory_supervision && !trajectory.is_empty() {
        features.reserve(trajectory.len() * rows * input_dims);
        states.reserve(trajectory.len() * rows * model.config.state_dims);
        positions.reserve(trajectory.len() * rows);
        gradient_rows.reserve(trajectory.len() * rows);
        weights.reserve(trajectory.len() * rows);
        for snapshot in trajectory {
            for (gradient_row, &row) in gradient.row_indices.iter().enumerate().take(rows) {
                if row >= trace.particle_count {
                    return Err(std::io::Error::other(format!(
                        "render proxy gradient row {row} out of range for {} particles",
                        trace.particle_count
                    ))
                    .into());
                }
                let feature_base = row * input_dims;
                features
                    .extend_from_slice(&snapshot.features[feature_base..feature_base + input_dims]);
                let state_base = row * model.config.state_dims;
                states.extend_from_slice(
                    &snapshot.states[state_base..state_base + model.config.state_dims],
                );
                positions.push(snapshot.positions[row]);
                gradient_rows.push(gradient_row);
                weights.push(0.5 + 0.5 * snapshot.step_fraction);
            }
        }
    } else {
        let mut selected_positions = Vec::with_capacity(rows);
        let mut selected_states = Vec::with_capacity(rows * model.config.state_dims);
        for &row in gradient.row_indices.iter().take(rows) {
            if row >= trace.particle_count {
                return Err(std::io::Error::other(format!(
                    "render proxy gradient row {row} out of range for {} particles",
                    trace.particle_count
                ))
                .into());
            }
            selected_positions.push(trace.positions[row]);
            let state_base = row * model.config.state_dims;
            selected_states
                .extend_from_slice(&trace.states[state_base..state_base + model.config.state_dims]);
        }
        let step = model.step_cpu(
            &selected_positions,
            &selected_states,
            1,
            rows,
            grid,
            1.0,
            None,
        )?;
        features = step.perception.features;
        states = selected_states;
        positions = selected_positions;
        gradient_rows.extend(0..rows);
        weights.resize(rows, 1.0);
    }

    let mut target_update = model.forward_update_from_features(&features)?;
    for chunk_start in (0..positions.len()).step_by(rows) {
        let chunk_end = (chunk_start + rows).min(positions.len());
        let chunk_positions = &positions[chunk_start..chunk_end];
        let chunk_states =
            &states[chunk_start * model.config.state_dims..chunk_end * model.config.state_dims];
        let coverage_updates = render_proxy_target_coverage_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            cfg.coverage_gain,
            cfg.coverage_samples,
            cfg.max_update_norm,
            cfg.coverage_mode,
            cfg.coverage_softness,
            cfg.coverage_repulsion_gain,
            cfg.coverage_gap_gain,
            cfg.coverage_repulsion_radius,
            cfg.coverage_normal_weight,
            cfg.seed_scale,
        );
        for local_idx in 0..chunk_positions.len() {
            let row = chunk_start + local_idx;
            let gradient_row = gradient_rows[row];
            let base = row * output_dims;
            let grad = gradient.gradients[gradient_row];
            let weight = weights[row];
            let mut update = [
                -cfg.motion_gain * grad[0] * weight + coverage_updates[local_idx][0] * weight,
                -cfg.motion_gain * grad[1] * weight + coverage_updates[local_idx][1] * weight,
                -cfg.motion_gain * grad[2] * weight + coverage_updates[local_idx][2] * weight,
            ];
            let norm =
                (update[0] * update[0] + update[1] * update[1] + update[2] * update[2]).sqrt();
            if norm > cfg.max_update_norm.max(1.0e-6) {
                let scale = cfg.max_update_norm / norm;
                update[0] *= scale;
                update[1] *= scale;
                update[2] *= scale;
            }
            target_update[base] += update[0];
            target_update[base + 1] += update[1];
            target_update[base + 2] += update[2];
            if cfg.opacity_gain > 0.0 {
                let Some(opacity_channel) =
                    growth_3d_material_opacity_channel(model.config.state_dims)
                else {
                    continue;
                };
                let state_logit = states[row * model.config.state_dims + opacity_channel]
                    + cfg.render.opacity_logit_bias;
                let opacity_update = -cfg.opacity_gain
                    * gradient.opacity_gradients[gradient_row]
                    * sigmoid_unit_derivative(state_logit)
                    * weight;
                target_update[base + model.config.spatial_dims + opacity_channel] +=
                    opacity_update.clamp(-cfg.max_opacity_update, cfg.max_opacity_update);
            }
        }
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_direct_rollout_multiseed_training_step(
    model: &mut NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    round: usize,
    trace: &burn_automata::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    let round_seed = cfg
        .seed
        .wrapping_add((round as u64).wrapping_mul(0x9e37_79b9));
    let mut training_seeds = vec![round_seed];
    for seed in render_proxy_selection_seeds(cfg) {
        if !training_seeds.contains(&seed) {
            training_seeds.push(seed);
        }
    }

    let base_model = model.clone();
    let base_weights = model.weights.clone();
    let mut delta = NpaWeights::zeros(&model.config);
    let mut reports = Vec::with_capacity(training_seeds.len());
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }

    for seed in training_seeds {
        let mut candidate = base_model.clone();
        let report = if seed == round_seed {
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                trace,
                trajectory,
                gradient,
                cfg,
            )?
        } else {
            let (seed_trace, seed_trajectory) =
                render_training_trajectory_for_seed(&candidate, grid, cfg, seed)?;
            let seed_gradient = render_position_gradient(&seed_trace, target, render_cfg, cfg)?;
            render_direct_rollout_training_step(
                &mut candidate,
                grid,
                target,
                &seed_trace,
                &seed_trajectory,
                &seed_gradient,
                cfg,
            )?
        };
        accumulate_weight_delta(&mut delta, &base_weights, &candidate.weights);
        reports.push(report);
    }

    let count = reports.len().max(1) as f32;
    apply_average_weight_delta(&mut model.weights, &base_weights, &delta, count.recip());

    let rows = reports.iter().map(|report| report.rows).sum();
    let initial_loss = reports
        .iter()
        .map(|report| report.initial_loss)
        .sum::<f32>()
        / count;
    let final_loss = reports.iter().map(|report| report.final_loss).sum::<f32>() / count;
    let best_loss = reports.iter().map(|report| report.best_loss).sum::<f32>() / count;
    let grad_norm = reports
        .iter()
        .filter_map(|report| report.history.last())
        .map(|entry| entry.grad_norm)
        .sum::<f32>()
        / count;
    let grad_scale = reports
        .iter()
        .filter_map(|report| report.history.last())
        .map(|entry| entry.grad_scale)
        .sum::<f32>()
        / count;

    Ok(TrainingRunReport {
        steps: 1,
        rows,
        initial_loss,
        final_loss,
        best_loss,
        history: vec![TrainingHistoryEntry {
            step: 1,
            loss: final_loss,
            grad_norm,
            grad_scale,
        }],
    })
}

fn accumulate_weight_delta(delta: &mut NpaWeights, before: &NpaWeights, after: &NpaWeights) {
    accumulate_weight_delta_slice(&mut delta.w1, &before.w1, &after.w1);
    accumulate_weight_delta_slice(&mut delta.b1, &before.b1, &after.b1);
    accumulate_weight_delta_slice(&mut delta.w2, &before.w2, &after.w2);
    accumulate_weight_delta_slice(&mut delta.b2, &before.b2, &after.b2);
}

fn accumulate_weight_delta_slice(delta: &mut [f32], before: &[f32], after: &[f32]) {
    debug_assert_eq!(delta.len(), before.len());
    debug_assert_eq!(before.len(), after.len());
    for ((delta_value, before_value), after_value) in
        delta.iter_mut().zip(before.iter()).zip(after.iter())
    {
        *delta_value += after_value - before_value;
    }
}

fn apply_average_weight_delta(
    weights: &mut NpaWeights,
    before: &NpaWeights,
    delta: &NpaWeights,
    scale: f32,
) {
    apply_average_weight_delta_slice(&mut weights.w1, &before.w1, &delta.w1, scale);
    apply_average_weight_delta_slice(&mut weights.b1, &before.b1, &delta.b1, scale);
    apply_average_weight_delta_slice(&mut weights.w2, &before.w2, &delta.w2, scale);
    apply_average_weight_delta_slice(&mut weights.b2, &before.b2, &delta.b2, scale);
}

fn apply_average_weight_delta_slice(
    weights: &mut [f32],
    before: &[f32],
    delta: &[f32],
    scale: f32,
) {
    debug_assert_eq!(weights.len(), before.len());
    debug_assert_eq!(before.len(), delta.len());
    for ((weight, before_value), delta_value) in weights.iter_mut().zip(before.iter()).zip(delta) {
        *weight = before_value + delta_value * scale;
    }
}

fn render_direct_rollout_training_step(
    model: &mut NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    trace: &burn_automata::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    cfg: &RenderProxyTrainingConfig,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    if trajectory.is_empty() {
        return Err(std::io::Error::other(
            "direct rollout render training requires trajectory snapshots",
        )
        .into());
    }
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.color_gradients.len());
    if rows == 0 {
        return Err(std::io::Error::other("direct rollout gradient produced no rows").into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let particle_count = trace.positions.len();
    let mut accumulated_gradients = zero_supervised_gradients(model);
    accumulated_gradients
        .features
        .reserve(trajectory.len() * particle_count * input_dims);
    let mut state_adjoint = terminal_render_state_adjoint(
        &model.config,
        trace,
        gradient,
        cfg.opacity_gain,
        cfg.render.opacity_logit_bias,
        rows,
    );
    let final_coverage_updates = render_proxy_target_coverage_updates(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.max_update_norm,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.seed_scale,
    );
    let mut position_adjoint = terminal_render_position_adjoint(
        &model.config,
        trace,
        gradient,
        &final_coverage_updates,
        cfg.motion_gain,
        cfg.full_coverage_adjoint,
        rows,
    );
    add_surface_position_adjoint(
        &model.config,
        target,
        &trace.positions,
        &trace.states,
        cfg.surface_gain,
        &mut position_adjoint,
    );
    let trajectory_adjoint =
        trajectory_render_adjoints(&model.config, target, trajectory, trace, cfg)?;
    let dt = 1.0_f32;
    let perception_options = PerceptionOptions {
        state_grad: model.config.state_grad,
        density_grad: model.config.density_grad,
        eps0: model.config.eps0,
        scale_equivariance: model.config.scale_equivariant(),
        particle_density_equivariance: model.config.particle_density_equivariant(),
        log_norm_grad: model.config.log_norm_grad,
        log_norm_density_grad: model.config.log_norm_density_grad,
        hybrid_state_gradient: true,
        position_features: model.config.position_features,
    };

    for snapshot_idx in (0..trajectory.len()).rev() {
        let snapshot = &trajectory[snapshot_idx];
        if let Some(snapshot_adjoint) = trajectory_adjoint[snapshot_idx].as_ref() {
            for particle_row in 0..particle_count {
                if particle_row >= snapshot_adjoint.position.len()
                    || particle_row * model.config.state_dims + model.config.state_dims
                        > snapshot_adjoint.state.len()
                {
                    continue;
                }
                for axis in 0..model.config.spatial_dims {
                    position_adjoint[particle_row][axis] +=
                        snapshot_adjoint.weight * snapshot_adjoint.position[particle_row][axis];
                }
                clamp_position_adjoint_row(
                    &mut position_adjoint[particle_row],
                    model.config.spatial_dims,
                );
                let state_base = particle_row * model.config.state_dims;
                for channel in 0..model.config.state_dims {
                    state_adjoint[state_base + channel] +=
                        snapshot_adjoint.weight * snapshot_adjoint.state[state_base + channel];
                }
                clamp_state_adjoint_row(
                    &mut state_adjoint[state_base..state_base + model.config.state_dims],
                );
            }
        }
        let updates = model.forward_update_from_features(&snapshot.features)?;
        let step_features = snapshot.features.clone();
        let mut step_output_gradients = vec![0.0; particle_count * output_dims];
        for particle_row in 0..particle_count {
            if particle_row >= snapshot.positions.len() {
                return Err(std::io::Error::other(format!(
                    "direct rollout gradient row {particle_row} out of range for {} particles",
                    snapshot.positions.len()
                ))
                .into());
            }
            let raw_base = particle_row * output_dims;
            let output_base = particle_row * output_dims;
            accumulate_motion_output_gradient(
                &model.config,
                grid.eps,
                &updates[raw_base..raw_base + output_dims],
                [
                    position_adjoint[particle_row][0] * dt,
                    position_adjoint[particle_row][1] * dt,
                    position_adjoint[particle_row][2] * dt,
                ],
                &mut step_output_gradients[output_base..output_base + output_dims],
            );

            let state_base = particle_row * model.config.state_dims;
            let update_state_base = model.config.spatial_dims;
            for channel in 0..model.config.state_dims {
                step_output_gradients[output_base + update_state_base + channel] +=
                    state_adjoint[state_base + channel] * dt;
            }
        }

        let step_gradients =
            mlp_backward_from_output_gradients(model, &step_features, &step_output_gradients)?;
        let perception_adjoint = perceive_adjoint_with_options(
            &snapshot.positions,
            &snapshot.states,
            trace.batch_size,
            trace.particle_count,
            model.config.state_dims,
            grid,
            perception_options,
            &step_gradients.features,
        )?;
        for particle_row in 0..particle_count {
            for axis in 0..model.config.spatial_dims {
                position_adjoint[particle_row][axis] +=
                    cfg.perception_position_gain * perception_adjoint.position[particle_row][axis];
            }
            clamp_position_adjoint_row(
                &mut position_adjoint[particle_row],
                model.config.spatial_dims,
            );
            let state_base = particle_row * model.config.state_dims;
            for channel in 0..model.config.state_dims {
                state_adjoint[state_base + channel] +=
                    perception_adjoint.state[state_base + channel];
            }
            clamp_state_adjoint_row(
                &mut state_adjoint[state_base..state_base + model.config.state_dims],
            );
        }
        accumulate_supervised_gradients(&mut accumulated_gradients, &step_gradients);
    }

    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let initial_loss = mesh_multiview_render_loss_from_trace(trace, target, render_cfg)?.total_loss;
    if cfg.direct_material_output_only {
        retain_material_output_gradients(model, &mut accumulated_gradients)?;
    }
    let step = apply_sgd_gradients(model, &accumulated_gradients, cfg.sgd)?;
    Ok(TrainingRunReport {
        steps: 1,
        rows: step.rows,
        initial_loss,
        final_loss: initial_loss,
        best_loss: initial_loss,
        history: vec![TrainingHistoryEntry {
            step: 1,
            loss: initial_loss,
            grad_norm: step.grad_norm,
            grad_scale: step.grad_scale,
        }],
    })
}

fn terminal_render_state_adjoint(
    config: &NpaConfig,
    trace: &burn_automata::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    opacity_gain: f32,
    opacity_logit_bias: f32,
    rows: usize,
) -> Vec<f32> {
    let mut state_adjoint = vec![0.0; trace.states.len()];
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row * trace.state_dims + config.state_dims > trace.states.len() {
            continue;
        }
        let state_base = particle_row * trace.state_dims;
        if opacity_gain > 0.0
            && let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims)
        {
            let final_logit = trace.states[state_base + opacity_channel] + opacity_logit_bias;
            state_adjoint[state_base + opacity_channel] += opacity_gain
                * gradient.opacity_gradients[gradient_row]
                * sigmoid_unit_derivative(final_logit);
        }
        if config.state_dims >= 3 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                let state_value = trace.states[state_base + tail + channel];
                if state_value > -1.0 && state_value < 1.0 {
                    state_adjoint[state_base + tail + channel] +=
                        0.5 * gradient.color_gradients[gradient_row][channel];
                }
            }
        }
    }
    state_adjoint
}

fn terminal_render_position_adjoint(
    config: &NpaConfig,
    trace: &burn_automata::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    coverage_updates: &[[f32; 3]],
    motion_gain: f32,
    full_coverage_adjoint: bool,
    rows: usize,
) -> Vec<[f32; 4]> {
    let mut position_adjoint = vec![[0.0; 4]; trace.positions.len()];
    if full_coverage_adjoint {
        for particle_row in 0..position_adjoint.len() {
            for axis in 0..config.spatial_dims {
                let coverage = coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0);
                position_adjoint[particle_row][axis] -= motion_gain * coverage;
            }
            clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
        }
    }
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row >= position_adjoint.len() {
            continue;
        }
        for axis in 0..config.spatial_dims {
            let coverage = if full_coverage_adjoint {
                0.0
            } else {
                coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0)
            };
            position_adjoint[particle_row][axis] +=
                motion_gain * (gradient.gradients[gradient_row][axis] - coverage);
        }
        clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
    }
    position_adjoint
}

fn add_surface_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return;
    }
    for (row, position) in positions.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        if config.state_dims > 3
            && states
                .get(row * config.state_dims + 3)
                .is_some_and(|opacity| *opacity <= -1.0)
        {
            continue;
        }
        let projection = target.project(position3(*position));
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= surface_gain * projection.residual[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

fn trajectory_render_adjoints(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    trajectory: &[RenderTrajectorySnapshot],
    trace: &burn_automata::RolloutTrace,
    cfg: &RenderProxyTrainingConfig,
) -> Result<Vec<Option<RenderTrajectoryAdjoint>>, Box<dyn std::error::Error>> {
    let mut adjoints = (0..trajectory.len()).map(|_| None).collect::<Vec<_>>();
    if cfg.trajectory_render_gain <= 0.0 || cfg.trajectory_render_samples == 0 {
        return Ok(adjoints);
    }

    let indices = trajectory_render_sample_indices(trajectory.len(), cfg.trajectory_render_samples);
    if indices.is_empty() {
        return Ok(adjoints);
    }
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let sample_count = indices.len().max(1) as f32;

    for index in indices {
        let snapshot = &trajectory[index];
        let snapshot_trace = burn_automata::RolloutTrace {
            positions: snapshot.positions.clone(),
            states: snapshot.states.clone(),
            batch_size: trace.batch_size,
            particle_count: trace.particle_count,
            state_dims: trace.state_dims,
            steps: ((snapshot.step_fraction * trace.steps.max(1) as f32).round() as usize).max(1),
            mean_dx: Vec::new(),
        };
        let gradient = render_position_gradient(&snapshot_trace, target, render_cfg, cfg)?;
        let rows = gradient
            .gradients
            .len()
            .min(gradient.row_indices.len())
            .min(gradient.opacity_gradients.len())
            .min(gradient.color_gradients.len());
        if rows == 0 {
            continue;
        }
        let coverage_updates = render_proxy_target_coverage_updates(
            config,
            target,
            &snapshot_trace.positions,
            &snapshot_trace.states,
            cfg.coverage_gain,
            cfg.coverage_samples,
            cfg.max_update_norm,
            cfg.coverage_mode,
            cfg.coverage_softness,
            cfg.coverage_repulsion_gain,
            cfg.coverage_gap_gain,
            cfg.coverage_repulsion_radius,
            cfg.coverage_normal_weight,
            cfg.seed_scale,
        );
        let state = terminal_render_state_adjoint(
            config,
            &snapshot_trace,
            &gradient,
            cfg.opacity_gain,
            cfg.render.opacity_logit_bias,
            rows,
        );
        let position = terminal_render_position_adjoint(
            config,
            &snapshot_trace,
            &gradient,
            &coverage_updates,
            cfg.motion_gain,
            cfg.full_coverage_adjoint,
            rows,
        );
        let mut position = position;
        add_surface_position_adjoint(
            config,
            target,
            &snapshot_trace.positions,
            &snapshot_trace.states,
            cfg.surface_gain,
            &mut position,
        );
        let weight = cfg.trajectory_render_gain * snapshot.step_fraction.powi(2) / sample_count;
        adjoints[index] = Some(RenderTrajectoryAdjoint {
            state,
            position,
            weight,
        });
    }

    Ok(adjoints)
}

fn zero_supervised_gradients(model: &NpaModel) -> SupervisedGradients {
    SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: Vec::new(),
    }
}

fn accumulate_supervised_gradients(total: &mut SupervisedGradients, step: &SupervisedGradients) {
    add_assign_slice(&mut total.w1, &step.w1);
    add_assign_slice(&mut total.b1, &step.b1);
    add_assign_slice(&mut total.w2, &step.w2);
    add_assign_slice(&mut total.b2, &step.b2);
    total.features.extend_from_slice(&step.features);
}

fn retain_material_output_gradients(
    model: &NpaModel,
    gradients: &mut SupervisedGradients,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) else {
        return Err(std::io::Error::other(
            "material-output-only training requires a material opacity channel",
        )
        .into());
    };
    gradients.w1.fill(0.0);
    gradients.b1.fill(0.0);
    let output_dims = model.config.update_dims();
    let material_output = model.config.spatial_dims + material_channel;
    for output in 0..output_dims {
        if output == material_output {
            continue;
        }
        let start = output * model.config.hidden_dims;
        let end = start + model.config.hidden_dims;
        gradients.w2[start..end].fill(0.0);
        gradients.b2[output] = 0.0;
    }
    Ok(())
}

fn add_assign_slice(total: &mut [f32], step: &[f32]) {
    debug_assert_eq!(total.len(), step.len());
    for (dst, src) in total.iter_mut().zip(step.iter()) {
        *dst += *src;
    }
}

fn clamp_state_adjoint_row(row: &mut [f32]) {
    const MAX_STATE_ADJOINT_NORM: f32 = 10.0;
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= MAX_STATE_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_STATE_ADJOINT_NORM / norm;
    for value in row {
        *value *= scale;
    }
}

fn clamp_position_adjoint_row(row: &mut [f32; 4], spatial_dims: usize) {
    const MAX_POSITION_ADJOINT_NORM: f32 = 10.0;
    let norm = row
        .iter()
        .take(spatial_dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= MAX_POSITION_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_POSITION_ADJOINT_NORM / norm;
    for value in row.iter_mut().take(spatial_dims) {
        *value *= scale;
    }
}

fn accumulate_motion_output_gradient(
    config: &NpaConfig,
    grid_eps: f32,
    raw_update: &[f32],
    dloss_ddx: [f32; 3],
    output_gradient: &mut [f32],
) {
    let dims = config.spatial_dims;
    let motion_scale = config.alpha * config.motion_eps(grid_eps);
    let mut norm2 = 0.0_f32;
    for value in raw_update.iter().take(dims) {
        norm2 += value * value;
    }
    let norm = norm2.sqrt();
    let denom = 1.0 + norm;
    let dot = raw_update
        .iter()
        .zip(dloss_ddx.iter())
        .take(dims)
        .map(|(raw, grad)| raw * grad)
        .sum::<f32>();

    for axis in 0..dims {
        let mut grad = motion_scale * dloss_ddx[axis] / denom;
        if norm > 1.0e-6 {
            grad -= motion_scale * raw_update[axis] * dot / (norm * denom * denom);
        }
        output_gradient[axis] += grad;
    }
}

fn render_proxy_target_coverage_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let updates = vec![[0.0; 3]; rows];
    if rows == 0 || coverage_gain <= 0.0 {
        return updates;
    }

    let active_rows = (0..rows)
        .filter(|&row| config.state_dims <= 3 || states[row * config.state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return updates;
    }

    let mut updates = match coverage_mode {
        CoverageUpdateModeArg::HardNearest => render_proxy_hard_target_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            seed_scale,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
    };
    if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &active_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut updates,
        );
    }
    add_surface_gap_relocation_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        coverage_normal_weight,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_normal_coverage_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_normal_weight,
        coverage_samples,
        max_update_norm,
        &mut updates,
    );
    updates
}

fn render_proxy_hard_target_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if best_distance2.is_finite() {
            residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
            residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
            residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
            counts[best_row] += 1;
        }
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        updates[row][0] = coverage_gain * residual_sums[row][0] / count as f32;
        updates[row][1] = coverage_gain * residual_sums[row][1] / count as f32;
        updates[row][2] = coverage_gain * residual_sums[row][2] / count as f32;
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}

fn render_proxy_soft_chamfer_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let samples = coverage_samples.max(rows.max(512));
    let sigma = if coverage_softness.is_finite() && coverage_softness > 0.0 {
        coverage_softness
    } else {
        target_coverage_threshold(seed_scale) * 1.5
    }
    .max(1.0e-4);
    let inv_two_sigma2 = 0.5 / (sigma * sigma);
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let normal_cost_scale = if coverage_normal_weight.is_finite() {
        coverage_normal_weight.max(0.0) * sigma * sigma
    } else {
        0.0
    };
    let mut projected_normals = vec![[0.0_f32; 3]; rows];
    for &row in active_rows {
        let projection = target.project([positions[row][0], positions[row][1], positions[row][2]]);
        projected_normals[row] = projection.normal;
        residual_sums[row][0] += 0.5 * projection.residual[0];
        residual_sums[row][1] += 0.5 * projection.residual[1];
        residual_sums[row][2] += 0.5 * projection.residual[2];
        weights[row] += 0.5;
    }

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_score = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            best_score = best_score.min(score);
        }
        if !best_score.is_finite() {
            continue;
        }

        let mut weight_sum = 0.0_f32;
        let mut sample_weights = Vec::with_capacity(active_rows.len());
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            let weight = (-(score - best_score) * inv_two_sigma2).exp();
            weight_sum += weight;
            sample_weights.push((row, weight));
        }
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            continue;
        }

        for (row, weight) in sample_weights {
            let normalized = weight / weight_sum;
            residual_sums[row][0] += normalized * (sample.position[0] - positions[row][0]);
            residual_sums[row][1] += normalized * (sample.position[1] - positions[row][1]);
            residual_sums[row][2] += normalized * (sample.position[2] - positions[row][2]);
            weights[row] += normalized;
        }
    }

    let mut repulsion_sums = vec![[0.0_f32; 3]; rows];
    if coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite() {
        let repulsion_radius =
            if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
                coverage_repulsion_radius
            } else {
                target_coverage_threshold(seed_scale) * 2.0
            }
            .max(1.0e-4);
        for lhs_idx in 0..active_rows.len() {
            let lhs = active_rows[lhs_idx];
            for &rhs in &active_rows[lhs_idx + 1..] {
                let dx = positions[lhs][0] - positions[rhs][0];
                let dy = positions[lhs][1] - positions[rhs][1];
                let dz = positions[lhs][2] - positions[rhs][2];
                let distance2 = dx * dx + dy * dy + dz * dz;
                if distance2 <= 1.0e-12 || distance2 >= repulsion_radius * repulsion_radius {
                    continue;
                }
                let distance = distance2.sqrt();
                let strength = (1.0 - distance / repulsion_radius).powi(2);
                let force = [
                    dx * strength / distance,
                    dy * strength / distance,
                    dz * strength / distance,
                ];
                let lhs_force = tangent_component(force, projected_normals[lhs]);
                let rhs_force =
                    tangent_component([-force[0], -force[1], -force[2]], projected_normals[rhs]);
                for axis in 0..3 {
                    repulsion_sums[lhs][axis] += lhs_force[axis];
                    repulsion_sums[rhs][axis] += rhs_force[axis];
                }
            }
        }
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        updates[row][0] = coverage_gain
            * (residual_sums[row][0] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][0]);
        updates[row][1] = coverage_gain
            * (residual_sums[row][1] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][1]);
        updates[row][2] = coverage_gain
            * (residual_sums[row][2] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}

#[allow(clippy::too_many_arguments)]
fn add_surface_tangent_repulsion_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_repulsion_gain <= 0.0
        || !coverage_repulsion_gain.is_finite()
        || active_rows.len() < 2
    {
        return;
    }
    let radius = if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
        coverage_repulsion_radius
    } else {
        target_coverage_threshold(seed_scale) * 2.0
    }
    .max(1.0e-4);
    let radius2 = radius * radius;
    let mut projected_normals = vec![[0.0_f32; 3]; positions.len()];
    for &row in active_rows {
        if row < positions.len() {
            projected_normals[row] = target.project(position3(positions[row])).normal;
        }
    }
    let mut repulsion_sums = vec![[0.0_f32; 3]; positions.len()];
    let mut counts = vec![0usize; positions.len()];
    for lhs_idx in 0..active_rows.len() {
        let lhs = active_rows[lhs_idx];
        if lhs >= positions.len() {
            continue;
        }
        for &rhs in &active_rows[lhs_idx + 1..] {
            if rhs >= positions.len() {
                continue;
            }
            let dx = positions[lhs][0] - positions[rhs][0];
            let dy = positions[lhs][1] - positions[rhs][1];
            let dz = positions[lhs][2] - positions[rhs][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 <= 1.0e-12 || distance2 >= radius2 {
                continue;
            }
            let distance = distance2.sqrt();
            let strength = (1.0 - distance / radius).powi(2);
            let force = [
                dx * strength / distance,
                dy * strength / distance,
                dz * strength / distance,
            ];
            let lhs_force = tangent_component(force, projected_normals[lhs]);
            let rhs_force =
                tangent_component([-force[0], -force[1], -force[2]], projected_normals[rhs]);
            for axis in 0..3 {
                repulsion_sums[lhs][axis] += lhs_force[axis];
                repulsion_sums[rhs][axis] += rhs_force[axis];
            }
            counts[lhs] += 1;
            counts[rhs] += 1;
        }
    }
    for &row in active_rows {
        if row >= updates.len() || counts[row] == 0 {
            continue;
        }
        let scale = coverage_gain * coverage_repulsion_gain / counts[row] as f32;
        for axis in 0..3 {
            updates[row][axis] += scale * repulsion_sums[row][axis];
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_surface_gap_relocation_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    coverage_samples: usize,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_gap_gain <= 0.0
        || !coverage_gap_gain.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        score: f32,
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let threshold = target_coverage_threshold(seed_scale);
    let threshold2 = threshold * threshold;
    let normal_cost_scale = if coverage_normal_weight.is_finite() && coverage_normal_weight > 0.0 {
        coverage_normal_weight * threshold2.max(1.0e-6)
    } else {
        0.0
    };
    let projected_normals = if normal_cost_scale > 0.0 {
        let mut normals = vec![[0.0_f32; 3]; positions.len()];
        for &row in active_rows {
            if row < positions.len() {
                normals[row] = target.project(position3(positions[row])).normal;
            }
        }
        Some(normals)
    } else {
        None
    };
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; positions.len()];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_score = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_penalty = projected_normals.as_ref().map_or(0.0, |normals| {
                normal_cost_scale * (1.0 - dot3(sample.normal, normals[row]).clamp(-1.0, 1.0))
            });
            let score = distance2 + normal_penalty;
            if score < best_score {
                best_score = score;
                best_row = row;
            }
        }
        if !best_score.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_score <= threshold2 {
            continue;
        }
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            score: best_score,
        };
        if bin_candidates[bin].is_none_or(|current| best_score > current.score) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    if gaps.is_empty() {
        return;
    }
    gaps.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = gaps.len().min(active_rows.len().saturating_div(2).max(1));
    let mut relocated = 0usize;
    for gap in gaps.iter().copied() {
        if relocated >= max_relocated {
            break;
        }
        let mut best_row = gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if best_row.is_none() {
            best_row = gap_relocation_donor(
                gap.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = best_row else {
            continue;
        };
        let donor_weight = if assigned_counts[row] == 0 { 1.0 } else { 0.5 };
        let scale = 0.5 * coverage_gain * coverage_gap_gain * donor_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
        used_donors[row] = true;
        relocated += 1;
    }

    for &row in active_rows {
        if row >= positions.len() || row >= updates.len() {
            continue;
        }
        if assigned_counts[row] > 0 || used_donors[row] {
            continue;
        }
        let mut nearest_gap = gaps[0];
        let mut nearest_gap_distance2 = f32::MAX;
        for gap in &gaps {
            let dx = gap.position[0] - positions[row][0];
            let dy = gap.position[1] - positions[row][1];
            let dz = gap.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_gap_distance2 {
                nearest_gap_distance2 = distance2;
                nearest_gap = *gap;
            }
        }
        if !nearest_gap_distance2.is_finite() {
            continue;
        }
        let scale = 0.5 * coverage_gain * coverage_gap_gain;
        updates[row][0] += scale * (nearest_gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (nearest_gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (nearest_gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    require_under_assigned: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    let average_assignments = average_assignments.max(1.0);
    let under_assigned_limit = average_assignments.ceil().max(1.0);
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let under_assigned = assignments <= under_assigned_limit;
        if require_under_assigned
            && assigned_counts.get(row).copied().unwrap_or(0) > 0
            && !under_assigned
        {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let assignment_penalty = assignments / average_assignments;
        let overflow_bonus = (assignments / under_assigned_limit).max(1.0);
        let score = if require_under_assigned {
            distance2 * (1.0 + 0.25 * assignment_penalty)
        } else {
            distance2 * (1.0 + 0.25 * assignment_penalty) / overflow_bonus.sqrt()
        };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}

#[allow(clippy::too_many_arguments)]
fn add_surface_normal_coverage_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_normal_weight: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_normal_weight <= 0.0
        || !coverage_normal_weight.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct NormalGapCandidate {
        position: [f32; 3],
        distance2: f32,
    }

    let directions = normal_coverage_directions();
    let bin_count = directions.len();
    let mut active_bin_counts = vec![0usize; bin_count];
    let mut active_bins = vec![usize::MAX; positions.len()];
    for &row in active_rows {
        if row >= positions.len() {
            continue;
        }
        let projection = target.project(position3(positions[row]));
        let bin = normal_direction_bin(projection.normal, &directions);
        active_bins[row] = bin;
        active_bin_counts[bin] += 1;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let mut target_bin_counts = vec![0usize; bin_count];
    let mut bin_candidates = vec![None::<NormalGapCandidate>; bin_count];
    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = normal_direction_bin(sample.normal, &directions);
        target_bin_counts[bin] += 1;

        let mut nearest_distance2 = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            nearest_distance2 = nearest_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        if nearest_distance2.is_finite()
            && bin_candidates[bin].is_none_or(|current| nearest_distance2 > current.distance2)
        {
            bin_candidates[bin] = Some(NormalGapCandidate {
                position: sample.position,
                distance2: nearest_distance2,
            });
        }
    }

    let mut desired_bin_counts = vec![0usize; bin_count];
    for bin in 0..bin_count {
        if target_bin_counts[bin] == 0 {
            continue;
        }
        desired_bin_counts[bin] = ((target_bin_counts[bin] as f32 / sample_count as f32)
            * active_rows.len() as f32
            * 0.85)
            .ceil()
            .max(1.0) as usize;
    }

    let mut gaps = (0..bin_count)
        .filter_map(|bin| {
            if active_bin_counts[bin] < desired_bin_counts[bin] {
                bin_candidates[bin].map(|candidate| (bin, candidate))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if gaps.is_empty() {
        return;
    }
    gaps.sort_by(|lhs, rhs| {
        rhs.1
            .distance2
            .partial_cmp(&lhs.1.distance2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used_donors = vec![false; positions.len()];
    let max_relocated = gaps.len().min(active_rows.len().saturating_div(3).max(1));
    for (gap_bin, gap) in gaps.into_iter().take(max_relocated) {
        let Some(row) = normal_gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &active_bins,
            &active_bin_counts,
            &desired_bin_counts,
            &used_donors,
        ) else {
            continue;
        };
        let donor_bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        if donor_bin < active_bin_counts.len() {
            active_bin_counts[donor_bin] = active_bin_counts[donor_bin].saturating_sub(1);
        }
        active_bin_counts[gap_bin] += 1;
        let scale = 0.5 * coverage_gain * coverage_normal_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
    }
}

fn normal_gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
) -> Option<usize> {
    normal_gap_relocation_donor_with_filter(
        gap_position,
        active_rows,
        positions,
        update_len,
        active_bins,
        active_bin_counts,
        desired_bin_counts,
        used_donors,
        true,
    )
    .or_else(|| {
        normal_gap_relocation_donor_with_filter(
            gap_position,
            active_rows,
            positions,
            update_len,
            active_bins,
            active_bin_counts,
            desired_bin_counts,
            used_donors,
            false,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn normal_gap_relocation_donor_with_filter(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
    require_surplus_bin: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        let surplus = bin < active_bin_counts.len()
            && active_bin_counts[bin] > desired_bin_counts.get(bin).copied().unwrap_or(0).max(1);
        if require_surplus_bin && !surplus {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let score = if surplus { distance2 * 0.75 } else { distance2 };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}

fn normal_direction_bin(normal: [f32; 3], directions: &[[f32; 3]]) -> usize {
    let normal = normalize3_or(normal, [0.0, 0.0, 1.0]);
    let mut best_bin = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (idx, direction) in directions.iter().enumerate() {
        let score = dot3(normal, *direction);
        if score > best_dot {
            best_dot = score;
            best_bin = idx;
        }
    }
    best_bin
}

fn normalize3_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 || !norm.is_finite() {
        fallback
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

fn normal_coverage_directions() -> [[f32; 3]; 14] {
    const INV_SQRT_3: f32 = 0.577_350_26;
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
    ]
}

fn clamp_update_row(updates: &mut [[f32; 3]], row: usize, max_update_norm: f32) {
    if row >= updates.len() {
        return;
    }
    let norm = (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for axis in 0..3 {
            updates[row][axis] *= clamp;
        }
    }
}

fn render_proxy_gap_farthest_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        distance2: f32,
    }

    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(rows.max(512));
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; rows];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;

        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            distance2: best_distance2,
        };
        if bin_candidates[bin].is_none_or(|current| best_distance2 > current.distance2) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    gaps.sort_by(|lhs, rhs| {
        rhs.distance2
            .partial_cmp(&lhs.distance2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; rows];
    let max_relocated = gaps.len().min(active_rows.len().max(1));
    for candidate in gaps.into_iter().take(max_relocated) {
        let mut donor = gap_relocation_donor(
            candidate.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if donor.is_none() {
            donor = gap_relocation_donor(
                candidate.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = donor else {
            continue;
        };
        let residual = [
            candidate.position[0] - positions[row][0],
            candidate.position[1] - positions[row][1],
            candidate.position[2] - positions[row][2],
        ];
        let weight = candidate.distance2.sqrt().max(1.0e-4);
        for axis in 0..3 {
            residual_sums[row][axis] += residual[axis] * weight;
        }
        weights[row] += weight;
        used_donors[row] = true;
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            let residual = if weights[row] > 0.0 {
                residual_sums[row][axis] / weights[row] + 0.25 * projection.residual[axis]
            } else {
                0.25 * projection.residual[axis]
            };
            updates[row][axis] = coverage_gain * residual;
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }

    updates
}

fn tangent_component(vector: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let normal_norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if normal_norm2 <= 1.0e-12 {
        return vector;
    }
    let dot = vector[0] * normal[0] + vector[1] * normal[1] + vector[2] * normal[2];
    [
        vector[0] - normal[0] * dot / normal_norm2,
        vector[1] - normal[1] * dot / normal_norm2,
        vector[2] - normal[2] * dot / normal_norm2,
    ]
}

fn reference_seed_scale_for_seed_mode(preset: AutomataPreset, seed_mode: ParticleSeed) -> f32 {
    match seed_mode {
        ParticleSeed::UvTorus3d
        | ParticleSeed::UvTorusDense3d
        | ParticleSeed::TorusFieldDense3d
        | ParticleSeed::TeapotFieldDense3d
        | ParticleSeed::TorusGrowth3d
        | ParticleSeed::TeapotGrowth3d
        | ParticleSeed::TorusSubstrateGrowth3d
        | ParticleSeed::TeapotSubstrateGrowth3d
        | ParticleSeed::TorusMorphogenDense3d
        | ParticleSeed::TeapotMorphogenDense3d => UV_TORUS_FIELD_SCALE,
        _ => NpaConfig::seed_scale_for_preset(preset),
    }
}

fn default_train_target_seed(
    _preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> Option<u64> {
    if zero_update {
        None
    } else {
        Some(target_seed.unwrap_or(DEFAULT_GROWTH_TARGET_SEED))
    }
}

fn train_target_source(
    preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> String {
    match (target_seed, zero_update) {
        (Some(seed), false) => format!("seeded:{preset:?}:{seed}"),
        (None, true) => "explicit-zero-update".to_string(),
        _ => unreachable!("target seed/source selection should be normalized first"),
    }
}

fn training_source_with_batch(batch_source: TrainingBatchArg, target_source: &str) -> String {
    match batch_source {
        TrainingBatchArg::Rollout => format!("rollout-local:{target_source}"),
        TrainingBatchArg::Features => format!("feature-rows:{target_source}"),
    }
}

fn render_proxy_sliced_ot_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let samples = (0..sample_count)
        .map(|sample_idx| target.surface_sample(sample_idx).position)
        .collect::<Vec<_>>();
    let directions = sliced_ot_directions();
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];

    for direction in directions {
        let mut target_order = (0..samples.len()).collect::<Vec<_>>();
        target_order.sort_by(|&lhs, &rhs| {
            dot3(samples[lhs], direction)
                .partial_cmp(&dot3(samples[rhs], direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut active_order = active_rows.to_vec();
        active_order.sort_by(|&lhs, &rhs| {
            dot3(position3(positions[lhs]), direction)
                .partial_cmp(&dot3(position3(positions[rhs]), direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let active_len = active_order.len().max(1);
        for (rank, &row) in active_order.iter().enumerate() {
            let sample_rank = (((rank as f32 + 0.5) * sample_count as f32 / active_len as f32)
                .floor() as usize)
                .min(sample_count - 1);
            let sample = samples[target_order[sample_rank]];
            for axis in 0..3 {
                residual_sums[row][axis] += sample[axis] - positions[row][axis];
            }
            weights[row] += 1.0;
        }
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            residual_sums[row][axis] += 0.25 * projection.residual[axis];
        }
        weights[row] += 0.25;
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        for axis in 0..3 {
            updates[row][axis] = coverage_gain * residual_sums[row][axis] / weights[row];
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }
    updates
}

fn sliced_ot_directions() -> [[f32; 3]; 13] {
    const INV_SQRT_2: f32 = 0.707_106_77;
    const INV_SQRT_3: f32 = 0.577_350_26;
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [INV_SQRT_2, INV_SQRT_2, 0.0],
        [INV_SQRT_2, -INV_SQRT_2, 0.0],
        [INV_SQRT_2, 0.0, INV_SQRT_2],
        [INV_SQRT_2, 0.0, -INV_SQRT_2],
        [0.0, INV_SQRT_2, INV_SQRT_2],
        [0.0, INV_SQRT_2, -INV_SQRT_2],
        [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
    ]
}

fn position3(position: [f32; 4]) -> [f32; 3] {
    [position[0], position[1], position[2]]
}

#[derive(Serialize)]
struct CliTrainingReport {
    preset: AutomataPreset,
    target_source: String,
    student_seed: u64,
    sgd: SgdConfig,
    report: TrainingRunReport,
    model_output: Option<String>,
    batch_source: TrainingBatchArg,
    rollout_supervision: Option<CliRolloutSupervisionReport>,
    mesh_rollout: Option<MeshRolloutReport>,
    render_loss: Option<MultiViewRenderLossReport>,
}

#[derive(Serialize)]
struct CliRenderLossEvalReport {
    target: MeshTargetArg,
    model: String,
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    render_loss: MultiViewRenderLossReport,
}

#[derive(Clone)]
struct Growth3dValidationConfig {
    particle_count: usize,
    steps: usize,
    seed: u64,
    extra_seeds: Vec<u64>,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    gate: Growth3dValidationGateArg,
    render: RenderLossConfig,
}

#[derive(Serialize)]
struct CliGrowth3dValidationReport {
    target: MeshTargetArg,
    model: String,
    source: Option<String>,
    position_features: bool,
    local_conditionless_lineage: bool,
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    non_opacity_seed_abs_max: f32,
    initial_color_state: Growth3dColorStateReport,
    final_color_state: Growth3dColorStateReport,
    permutation_consistency: Growth3dPermutationReport,
    seed_perturbation: Growth3dSeedPerturbationReport,
    mean_final_displacement: f32,
    final_opacity: Growth3dOpacityStats,
    activation: Growth3dActivationReport,
    initial_surface: Growth3dSurfaceStats,
    final_surface: Growth3dSurfaceStats,
    initial_active_surface: Growth3dSurfaceStats,
    final_active_surface: Growth3dSurfaceStats,
    initial_active_surface_tail: Growth3dSurfaceTailReport,
    final_active_surface_tail: Growth3dSurfaceTailReport,
    target_coverage_threshold: f32,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    initial_active_target_coverage: TargetCoverageStats,
    final_active_target_coverage: TargetCoverageStats,
    final_active_surface_coverage_profile: SurfaceCoverageProfileReport,
    torus_angular_coverage: Option<TorusAngularCoverageReport>,
    extent: Growth3dExtentReport,
    motion: Growth3dMotionReport,
    temporal: Growth3dTemporalReport,
    front: Growth3dFrontReport,
    max_motion_per_step: f32,
    render_loss: MultiViewRenderLossReport,
    strict_checks: Growth3dStrictChecksReport,
    strict_score: Growth3dStrictScoreReport,
    catalog_sanity: Growth3dCatalogSanityReport,
    robustness: Growth3dRobustnessReport,
    gate: Growth3dValidationGateArg,
    gate_passed: bool,
    strict_passed: bool,
}

#[derive(Serialize)]
struct Growth3dStrictChecksReport {
    passed: bool,
    no_position_features: bool,
    local_conditionless_lineage: bool,
    neutral_non_opacity_seed_state: bool,
    sparse_active_seed: bool,
    active_count_growth: bool,
    newly_activated_fraction: bool,
    active_front_expanded: bool,
    nonzero_motion: bool,
    sustained_motion: bool,
    local_front_coherent: bool,
    temporal_activation_progressive: bool,
    temporal_geometry_progressive: bool,
    mean_displacement_growth: bool,
    bounded_final_opacity: bool,
    color_state_emerged: bool,
    permutation_consistent: bool,
    surface_mean_improved: bool,
    surface_max_bounded: bool,
    surface_tail_bounded: bool,
    target_coverage_mean_improved: bool,
    target_coverage_max_bounded: bool,
    target_coverage_fraction: bool,
    torus_angular_coverage: bool,
    render_loss_passed: bool,
    failure_reasons: Vec<&'static str>,
}

#[derive(Serialize)]
struct Growth3dStrictScoreReport {
    score: f32,
    hard_failure_penalty: f32,
    surface_mean_ratio: f32,
    surface_mean_penalty: f32,
    surface_max_distance: f32,
    surface_max_penalty: f32,
    surface_tail_p99_distance: f32,
    surface_tail_p99_penalty: f32,
    surface_tail_over_threshold_fraction: f32,
    surface_tail_fraction_penalty: f32,
    target_coverage_mean_ratio: f32,
    target_coverage_mean_penalty: f32,
    target_coverage_max_distance: f32,
    target_coverage_max_penalty: f32,
    target_coverage_fraction: f32,
    target_coverage_fraction_penalty: f32,
    render_density_psnr_db: f32,
    render_density_penalty: f32,
    render_color_psnr_db: f32,
    render_color_penalty: f32,
    render_depth_psnr_db: f32,
    render_depth_penalty: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dOpacityStats {
    finite: bool,
    min: f32,
    max: f32,
    mean: f32,
    active_min: f32,
    active_max: f32,
    active_mean: f32,
    active_count: usize,
    max_allowed: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dColorStateReport {
    available: bool,
    finite: bool,
    count: usize,
    active_count: usize,
    mean_abs: f32,
    max_abs: f32,
    active_mean_abs: f32,
    active_max_abs: f32,
    active_channel_stddev: [f32; 3],
    active_channel_stddev_mean: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dPermutationReport {
    particle_count: usize,
    steps: usize,
    max_position_error: f32,
    mean_position_error: f32,
    max_state_error: f32,
    mean_state_error: f32,
    passed: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dSeedPerturbationReport {
    particle_count: usize,
    steps: usize,
    jitter_radius: f32,
    seed: u64,
    active_seed_count: usize,
    base_final_active_count: usize,
    perturbed_final_active_count: usize,
    active_count_ratio: f32,
    base_newly_activated_fraction: f32,
    perturbed_newly_activated_fraction: f32,
    base_final_active_max_radius: f32,
    perturbed_final_active_max_radius: f32,
    final_active_max_radius_ratio: f32,
    base_peak_mean_dx: f32,
    perturbed_peak_mean_dx: f32,
    peak_motion_ratio: f32,
    base_color_state_mean_abs: f32,
    perturbed_color_state_mean_abs: f32,
    color_state_mean_abs_ratio: f32,
    passed: bool,
}

#[derive(Serialize)]
struct Growth3dCatalogSanityReport {
    passed: bool,
    max_total_loss: f32,
    min_density_psnr_db: f32,
    min_color_psnr_db: f32,
    min_depth_psnr_db: f32,
    total_loss: f32,
    density_psnr_db: f32,
    color_psnr_db: f32,
    depth_psnr_db: f32,
}

#[derive(Clone, Debug, Serialize)]
struct TorusAngularCoverageReport {
    ring_bins: usize,
    tube_bins: usize,
    threshold: f32,
    covered_joint_bins: usize,
    covered_ring_bins: usize,
    covered_tube_bins: usize,
    joint_coverage_fraction: f32,
    ring_coverage_fraction: f32,
    tube_coverage_fraction: f32,
    max_ring_gap_bins: usize,
    max_tube_gap_bins: usize,
    mean_distance: f32,
    max_distance: f32,
}

#[derive(Clone, Debug, Serialize)]
struct SurfaceCoverageProfileReport {
    samples: usize,
    bins: usize,
    threshold: f32,
    covered_fraction: f32,
    covered_bin_fraction: f32,
    empty_bins: usize,
    min_bin_covered_fraction: f32,
    mean_bin_covered_fraction: f32,
    max_bin_covered_fraction: f32,
    assigned_particle_fraction: f32,
    covered_assigned_particle_fraction: f32,
    max_assigned_sample_fraction: f32,
    max_covered_assigned_sample_fraction: f32,
    bin_covered_fractions: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct Growth3dRobustnessReport {
    seed_count: usize,
    all_gate_passed: bool,
    all_catalog_sanity_passed: bool,
    all_strict_passed: bool,
    all_temporal_activation_progressive: bool,
    all_temporal_geometry_progressive: bool,
    all_local_front_coherent: bool,
    all_bounded_final_opacity: bool,
    all_color_state_emerged: bool,
    all_permutation_consistent: bool,
    all_seed_perturbation_stable: bool,
    worst_strict_score: f32,
    max_render_loss: f32,
    min_density_psnr_db: f32,
    min_color_psnr_db: f32,
    min_depth_psnr_db: f32,
    min_active_seed_count: usize,
    max_active_seed_count: usize,
    min_final_active_count: usize,
    max_final_active_count: usize,
    min_newly_activated_fraction: f32,
    min_active_growth_ratio: f32,
    max_final_opacity: f32,
    min_final_active_color_state_mean_abs: f32,
    min_final_active_color_state_stddev_mean: f32,
    max_permutation_position_error: f32,
    max_permutation_state_error: f32,
    min_perturbed_newly_activated_fraction: f32,
    min_perturbed_active_count_ratio: f32,
    max_perturbed_active_count_ratio: f32,
    min_perturbed_peak_motion_ratio: f32,
    max_perturbed_peak_motion_ratio: f32,
    max_front_nearest_previous_active_distance: f32,
    min_front_local_newly_activated_fraction: f32,
    min_final_active_target_coverage_fraction: f32,
    seeds: Vec<Growth3dRobustnessSeedReport>,
}

#[derive(Clone, Debug, Serialize)]
struct Growth3dRobustnessSeedReport {
    seed: u64,
    gate_passed: bool,
    strict_passed: bool,
    catalog_sanity_passed: bool,
    strict_score: f32,
    render_loss: f32,
    density_psnr_db: f32,
    color_psnr_db: f32,
    depth_psnr_db: f32,
    active_seed_count: usize,
    final_active_count: usize,
    newly_activated_fraction: f32,
    final_opacity_max: f32,
    color_state_emerged: bool,
    final_active_color_state_mean_abs: f32,
    final_active_color_state_stddev_mean: f32,
    permutation_consistent: bool,
    permutation_max_position_error: f32,
    permutation_max_state_error: f32,
    seed_perturbation_stable: bool,
    perturbed_newly_activated_fraction: f32,
    perturbed_active_count_ratio: f32,
    perturbed_peak_motion_ratio: f32,
    local_front_coherent: bool,
    front_local_newly_activated_fraction: f32,
    front_max_nearest_previous_active_distance: f32,
    temporal_activation_progressive: bool,
    temporal_geometry_progressive: bool,
    final_active_target_coverage_fraction: f32,
    final_active_surface_max: f32,
    failure_reasons: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dMotionReport {
    first_step_mean_dx: f32,
    peak_mean_dx: f32,
    peak_step: usize,
    final_step_mean_dx: f32,
    mean_dx: f32,
    late_mean_dx: f32,
    late_to_peak_ratio: f32,
    active_step_fraction: f32,
    sustained_step_fraction: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dFrontReport {
    transition_count: usize,
    newly_activated_count: usize,
    local_newly_activated_count: usize,
    local_newly_activated_fraction: f32,
    mean_nearest_previous_active_distance: f32,
    max_nearest_previous_active_distance: f32,
    max_allowed_distance: f32,
    finite: bool,
    passed: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dExtentReport {
    target_bounds_min: [f32; 3],
    target_bounds_max: [f32; 3],
    final_active_bounds_min: [f32; 3],
    final_active_bounds_max: [f32; 3],
    target_extent: [f32; 3],
    final_active_extent: [f32; 3],
    axis_extent_ratio: [f32; 3],
    min_axis_extent_ratio: f32,
    bbox_diagonal_ratio: f32,
    target_max_radius: f32,
    final_active_max_radius: f32,
    max_radius_ratio: f32,
}

#[derive(Clone, Debug, Serialize)]
struct Growth3dTemporalReport {
    samples: Vec<Growth3dTemporalSampleReport>,
    first_growth_step: Option<usize>,
    half_activation_step: Option<usize>,
    full_activation_step: Option<usize>,
    activation_span_steps: usize,
    progressive_activation: bool,
    surface_mean_ratio: f32,
    target_coverage_mean_ratio: f32,
    target_coverage_fraction_delta: f32,
    geometry_progressive: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dTemporalSampleReport {
    steps: usize,
    active_count: usize,
    active_fraction: f32,
    newly_activated_count: usize,
    final_active_mean_radius: f32,
    final_active_max_radius: f32,
    mean_displacement: f32,
    active_surface: Growth3dSurfaceStats,
    target_coverage: TargetCoverageStats,
}

#[derive(Serialize)]
struct Growth3dActivationReport {
    active_seed_count: usize,
    inactive_seed_count: usize,
    final_active_count: usize,
    newly_activated_count: usize,
    newly_activated_fraction: f32,
    final_active_mean_radius: f32,
    final_active_max_radius: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dSurfaceStats {
    mean_distance: f32,
    max_distance: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Growth3dSurfaceTailReport {
    count: usize,
    threshold: f32,
    p95_distance: f32,
    p99_distance: f32,
    max_distance: f32,
    over_threshold_count: usize,
    over_threshold_fraction: f32,
    opacity_weighted_mean_distance: f32,
    opacity_weighted_over_threshold_fraction: f32,
}

#[derive(Serialize)]
struct CliRenderTrainingReport {
    target: MeshTargetArg,
    base_model: Option<String>,
    model_output: String,
    particle_count: usize,
    rollout_steps: usize,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    sgd: SgdConfig,
    report: RenderProxyTrainingReport,
    final_render_loss: MultiViewRenderLossReport,
}

#[derive(Clone, Debug, Serialize)]
struct RenderProxyTrainingReport {
    rounds: usize,
    supervised_steps_per_round: usize,
    gradient_particles: usize,
    gradient_mode: RenderGradientModeArg,
    finite_diff_eps: f32,
    motion_gain: f32,
    perception_position_gain: f32,
    max_update_norm: f32,
    trajectory_supervision: bool,
    trajectory_render_gain: f32,
    trajectory_render_samples: usize,
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    full_coverage_adjoint: bool,
    surface_gain: f32,
    opacity_gain: f32,
    max_opacity_update: f32,
    direct_line_search: bool,
    direct_line_search_scales: Vec<f32>,
    direct_material_output_only: bool,
    training_backend: RenderTrainingBackendArg,
    direct_selection_seed_training: bool,
    selection_seed: Option<u64>,
    selection_seeds: Vec<u64>,
    initial_render_loss: MultiViewRenderLossReport,
    final_render_loss: MultiViewRenderLossReport,
    selected_round: Option<usize>,
    history: Vec<RenderProxyTrainingHistoryEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct RenderProxyTrainingHistoryEntry {
    round: usize,
    before_loss: f32,
    after_loss: f32,
    selection_loss: f32,
    selection_score: f32,
    before_density_psnr_db: f32,
    after_density_psnr_db: f32,
    selection_density_psnr_db: f32,
    selection_active_surface_max: f32,
    selection_target_coverage_fraction: f32,
    selection_morphology_non_regressed: bool,
    selection_worst_seed: u64,
    selection_worst_failure_reasons: Vec<&'static str>,
    before_color_psnr_db: f32,
    after_color_psnr_db: f32,
    before_depth_psnr_db: f32,
    after_depth_psnr_db: f32,
    supervised_loss: f32,
    train_grad_norm: f32,
    train_grad_scale: f32,
    train_step_scale: f32,
    gradient_rms: f32,
    opacity_gradient_rms: f32,
}

#[derive(Serialize)]
struct CliTorusTrainingReport {
    preset: AutomataPreset,
    target_source: String,
    student_seed: u64,
    sgd: SgdConfig,
    report: TrainingRunReport,
    model_output: Option<String>,
    robustness: TorusRobustnessReport,
    batch_source: TrainingBatchArg,
    training_mode: MeshTrainingModeArg,
    rollout_supervision: Option<CliRolloutSupervisionReport>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct CliRolloutSupervisionReport {
    particle_count: usize,
    rollout_steps: usize,
    rollouts: usize,
    temporal_samples: usize,
    update_prob: f32,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    motion_gain: Option<f32>,
    max_update_norm: Option<f32>,
    density_gain: Option<f32>,
    expansion_gain: Option<f32>,
    coverage_gain: Option<f32>,
    coverage_samples: Option<usize>,
    coverage_mode: Option<CoverageUpdateModeArg>,
    coverage_softness: Option<f32>,
    coverage_repulsion_gain: Option<f32>,
    coverage_gap_gain: Option<f32>,
    coverage_repulsion_radius: Option<f32>,
    coverage_normal_weight: Option<f32>,
    extent_gain: Option<f32>,
    color_gain: Option<f32>,
    aux_state_gain: Option<f32>,
    opacity_gain: Option<f32>,
    front_opacity_gain: Option<f32>,
    front_radius: Option<f32>,
    front_max_opacity_update: Option<f32>,
    front_motion_gate: Option<bool>,
    preserve_opacity_update: Option<bool>,
}

#[derive(Clone, Copy)]
struct MeshRolloutCaseConfig {
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
}

#[derive(Serialize)]
struct MeshRolloutReport {
    passed: bool,
    max_initial_surface_distance: f32,
    mean_initial_surface_distance: f32,
    max_surface_distance: f32,
    mean_surface_distance: f32,
    mean_surface_improvement: f32,
    mean_surface_improvement_ratio: f32,
    max_target_coverage_distance: f32,
    mean_target_coverage_distance: f32,
    min_target_coverage_fraction: f32,
    max_color_target_error: f32,
    mean_color_target_error: f32,
    first_motion_per_step: f32,
    max_motion_per_step: f32,
    max_opacity_target_error: f32,
    min_final_opacity: f32,
    max_final_opacity: f32,
    cases: Vec<MeshRolloutCaseReport>,
}

#[derive(Serialize)]
struct MeshRolloutCaseReport {
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    max_initial_surface_distance: f32,
    mean_initial_surface_distance: f32,
    max_surface_distance: f32,
    mean_surface_distance: f32,
    mean_surface_improvement: f32,
    mean_surface_improvement_ratio: f32,
    target_coverage_threshold: f32,
    max_target_coverage_distance: f32,
    mean_target_coverage_distance: f32,
    target_coverage_fraction: f32,
    max_color_target_error: f32,
    mean_color_target_error: f32,
    first_motion_per_step: f32,
    max_motion_per_step: f32,
    expected_final_opacity_logit: f32,
    min_final_opacity_logit: f32,
    max_final_opacity_logit: f32,
    max_opacity_target_error: f32,
    finite: bool,
}

#[derive(Clone, Copy)]
struct TorusRobustnessCaseConfig {
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
}

#[derive(Serialize)]
struct TorusRobustnessReport {
    passed: bool,
    target_opacity_delta: f32,
    trained_opacity_delta: f32,
    target_motion_gain: f32,
    target_residual_decay: f32,
    max_target_position_error: f32,
    mean_target_position_error: f32,
    max_torus_surface_error: f32,
    max_color_target_error: f32,
    first_motion_per_step: f32,
    max_motion_per_step: f32,
    max_opacity_target_error: f32,
    min_final_opacity: f32,
    max_final_opacity: f32,
    cases: Vec<TorusRobustnessCaseReport>,
}

#[derive(Serialize)]
struct TorusRobustnessCaseReport {
    particle_count: usize,
    steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    torus_inner_radius: f32,
    torus_outer_radius: f32,
    max_initial_target_position_error: f32,
    mean_initial_target_position_error: f32,
    max_target_position_error: f32,
    mean_target_position_error: f32,
    max_torus_surface_error: f32,
    mean_torus_surface_error: f32,
    min_final_radial: f32,
    max_final_radial: f32,
    max_final_abs_z: f32,
    max_color_target_error: f32,
    mean_color_target_error: f32,
    first_motion_per_step: f32,
    max_motion_per_step: f32,
    expected_final_opacity_logit: f32,
    min_final_opacity_logit: f32,
    max_final_opacity_logit: f32,
    max_opacity_target_error: f32,
    finite: bool,
}

const TORUS_ROBUSTNESS_CASES: &[TorusRobustnessCaseConfig] = &[
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 3,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 17,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 97,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 8192,
        steps: 180,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
];

const TORUS_MORPHOGEN_ROBUSTNESS_CASES: &[TorusRobustnessCaseConfig] = &[
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 5,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 8192,
        steps: 200,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
];

const TEAPOT_FIELD_ROLLOUT_CASES: &[MeshRolloutCaseConfig] = &[
    MeshRolloutCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 13,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 8192,
        steps: 180,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
];

fn torus_field_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "torus field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("torus field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    let position_pairs = [
        add_identity_pair(&mut weights, position_offset, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 1, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 2, &mut hidden),
    ];
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    let tail_pairs = [
        add_identity_pair(&mut weights, tail, &mut hidden),
        add_identity_pair(&mut weights, tail + 1, &mut hidden),
        add_identity_pair(&mut weights, tail + 2, &mut hidden),
    ];

    let major = UV_TORUS_FIELD_SCALE;
    let minor = major * UV_TORUS_MINOR_RATIO;
    let outer = major + minor;
    let color_position_coeffs = [
        1.0 / (2.0 * outer),
        1.0 / (2.0 * outer),
        1.0 / (2.0 * minor.max(1.0e-4)),
    ];
    for channel in 0..3 {
        let out = config.spatial_dims + tail + channel;
        let (pos_hidden, neg_hidden) = tail_pairs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] -= UV_TORUS_FIELD_COLOR_GAIN;
        weights.w2[out * config.hidden_dims + neg_hidden] += UV_TORUS_FIELD_COLOR_GAIN;

        let axis = channel;
        let (pos_hidden, neg_hidden) = position_pairs[axis];
        let coeff = UV_TORUS_FIELD_COLOR_GAIN * color_position_coeffs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] += coeff;
        weights.w2[out * config.hidden_dims + neg_hidden] -= coeff;
    }

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    Ok(NpaModel { config, weights })
}

#[allow(dead_code)]
fn mesh_field_model(config: NpaConfig, _seed: u64) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "mesh field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("mesh field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    for axis in 0..3 {
        add_identity_pair(&mut weights, position_offset + axis, &mut hidden);
    }
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    for channel in 0..3 {
        add_identity_pair(&mut weights, tail + channel, &mut hidden);
    }

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    Ok(NpaModel { config, weights })
}

fn teapot_field_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "teapot field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("teapot field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    let position_pairs = [
        add_identity_pair(&mut weights, position_offset, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 1, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 2, &mut hidden),
    ];
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    let tail_pairs = [
        add_identity_pair(&mut weights, tail, &mut hidden),
        add_identity_pair(&mut weights, tail + 1, &mut hidden),
        add_identity_pair(&mut weights, tail + 2, &mut hidden),
    ];

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    let (bounds_min, bounds_max) = utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE).bounds();
    for channel in 0..3 {
        let out = config.spatial_dims + tail + channel;
        let (tail_pos, tail_neg) = tail_pairs[channel];
        weights.w2[out * config.hidden_dims + tail_pos] -= TEAPOT_FIELD_COLOR_GAIN;
        weights.w2[out * config.hidden_dims + tail_neg] += TEAPOT_FIELD_COLOR_GAIN;

        let range = (bounds_max[channel] - bounds_min[channel]).max(1.0e-4);
        let coeff = TEAPOT_FIELD_COLOR_GAIN / range;
        let (pos_hidden, neg_hidden) = position_pairs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] += coeff;
        weights.w2[out * config.hidden_dims + neg_hidden] -= coeff;
        weights.b2[out] += TEAPOT_FIELD_COLOR_GAIN * (-bounds_min[channel] / range - 0.5);
    }

    Ok(NpaModel { config, weights })
}

#[allow(dead_code)]
fn local_growth_student_model(
    config: NpaConfig,
    seed: u64,
    density_gain: f32,
    expansion_gain: f32,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    local_growth_student_model_with_axis_gains(config, seed, density_gain, [expansion_gain; 3])
}

fn mesh_axis_expansion_gains(target: &TriangleMeshTarget, base_gain: f32) -> [f32; 3] {
    let (bounds_min, bounds_max) = target.bounds();
    let extents = [
        (bounds_max[0] - bounds_min[0]).max(1.0e-4),
        (bounds_max[1] - bounds_min[1]).max(1.0e-4),
        (bounds_max[2] - bounds_min[2]).max(1.0e-4),
    ];
    let mean_extent = ((extents[0] + extents[1] + extents[2]) / 3.0).max(1.0e-4);
    [
        base_gain * (extents[0] / mean_extent).clamp(0.35, 2.25),
        base_gain * (extents[1] / mean_extent).clamp(0.35, 2.25),
        base_gain * (extents[2] / mean_extent).clamp(0.35, 2.25),
    ]
}

fn local_growth_student_model_with_axis_gains(
    config: NpaConfig,
    seed: u64,
    density_gain: f32,
    expansion_gains: [f32; 3],
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims <= 3 || config.hidden_dims < 16 {
        return Err(std::io::Error::other(format!(
            "local 3D growth student requires 3D config, state_dims > 3, and hidden_dims >= 16; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if config.position_features {
        return Err(std::io::Error::other(
            "local 3D growth student must not use absolute position_features",
        )
        .into());
    }

    let mut weights = NpaWeights::seeded(&config, seed);
    for value in weights.w1.iter_mut().chain(weights.w2.iter_mut()) {
        *value *= 0.02;
    }
    for value in &mut weights.b1 {
        *value = 1.0e-3;
    }
    weights.b2.fill(0.0);
    let input_dims = config.perception_dims();
    let opacity_gradient_offset = config.state_dims * 2 + 3 * config.spatial_dims;
    if expansion_gains
        .iter()
        .any(|gain| !gain.is_finite() || *gain < 0.0)
    {
        return Err(
            std::io::Error::other("expansion_gains must be finite and non-negative").into(),
        );
    }
    for axis in 0..config.spatial_dims {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.b1[pos_hidden] = 0.0;
        weights.b1[neg_hidden] = 0.0;
        weights.w1[pos_hidden * input_dims + opacity_gradient_offset + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + opacity_gradient_offset + axis] = -1.0;
        weights.w2[axis * config.hidden_dims + pos_hidden] = expansion_gains[axis];
        weights.w2[axis * config.hidden_dims + neg_hidden] = -expansion_gains[axis];
    }
    let opacity_front_hidden = config.spatial_dims * 2;
    weights.b1[opacity_front_hidden] = 0.0;
    weights.w1[opacity_front_hidden * input_dims + config.state_dims + 3] = 1.0;
    weights.w1[opacity_front_hidden * input_dims + 3] = -1.0;
    let opacity_out = config.spatial_dims + 3;
    weights.w2[opacity_out * config.hidden_dims + opacity_front_hidden] = LOCAL_GROWTH_OPACITY_GAIN;
    if let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims)
        && material_channel != 3
    {
        let material_low_hidden = 14;
        let material_high_hidden = 15;
        for hidden in [material_low_hidden, material_high_hidden] {
            weights.b1[hidden] = 0.0;
            for value in &mut weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
                *value = 0.0;
            }
            for output in 0..config.update_dims() {
                weights.w2[output * config.hidden_dims + hidden] = 0.0;
            }
        }
        weights.b1[material_low_hidden] = 1.0;
        weights.w1[material_low_hidden * input_dims + 3] = 1.0;
        weights.w1[material_high_hidden * input_dims + 3] = 1.0;
        let material_out = config.spatial_dims + material_channel;
        let material_base = material_out * config.hidden_dims;
        weights.w2[material_base + material_low_hidden] = LOCAL_GROWTH_MATERIAL_OPACITY_GAIN;
        weights.w2[material_base + material_high_hidden] = -LOCAL_GROWTH_MATERIAL_OPACITY_GAIN;
    }
    if density_gain != 0.0 {
        let density_gradient_offset = config.state_dims * 2
            + usize::from(config.state_grad) * config.state_dims * config.spatial_dims;
        for axis in 0..config.spatial_dims {
            let pos_hidden = 8 + axis * 2;
            let neg_hidden = pos_hidden + 1;
            weights.b1[pos_hidden] = 0.0;
            weights.b1[neg_hidden] = 0.0;
            weights.w1[pos_hidden * input_dims + density_gradient_offset + axis] = 1.0;
            weights.w1[neg_hidden * input_dims + density_gradient_offset + axis] = -1.0;
            weights.w2[axis * config.hidden_dims + pos_hidden] = density_gain;
            weights.w2[axis * config.hidden_dims + neg_hidden] = -density_gain;
        }
    }

    Ok(NpaModel { config, weights })
}

fn retime_growth_3d_front_model(
    model: &mut NpaModel,
    hidden: Option<usize>,
    front_gain: f32,
) -> Result<usize, Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "retime-growth3d requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "retime-growth3d only supports local conditionless 3D models",
        )
        .into());
    }
    if !front_gain.is_finite() || front_gain <= 0.0 {
        return Err(std::io::Error::other("front_gain must be positive and finite").into());
    }
    let hidden = hidden.unwrap_or(model.config.hidden_dims.saturating_sub(1));
    if hidden >= model.config.hidden_dims {
        return Err(std::io::Error::other(format!(
            "hidden index {hidden} out of range for hidden_dims={}",
            model.config.hidden_dims
        ))
        .into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    for value in &mut model.weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
        *value = 0.0;
    }
    model.weights.b1[hidden] = 0.0;
    model.weights.w1[hidden * input_dims + model.config.state_dims + 3] = 1.0;
    model.weights.w1[hidden * input_dims + 3] = -1.0;

    for output in 0..output_dims {
        model.weights.w2[output * model.config.hidden_dims + hidden] = 0.0;
    }
    let opacity_out = model.config.spatial_dims + 3;
    let opacity_base = opacity_out * model.config.hidden_dims;
    for value in &mut model.weights.w2[opacity_base..opacity_base + model.config.hidden_dims] {
        *value = 0.0;
    }
    model.weights.b2[opacity_out] = 0.0;
    model.weights.w2[opacity_base + hidden] = front_gain;

    Ok(hidden)
}

fn retime_growth_3d_active_opacity_model(
    model: &mut NpaModel,
    hidden: Option<usize>,
    active_opacity_gain: f32,
) -> Result<usize, Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "active-opacity retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "active-opacity retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !active_opacity_gain.is_finite() || active_opacity_gain <= 0.0 {
        return Err(
            std::io::Error::other("active_opacity_gain must be positive and finite").into(),
        );
    }
    let low_hidden = hidden.unwrap_or(model.config.hidden_dims.saturating_sub(3));
    let high_hidden = low_hidden + 1;
    if high_hidden >= model.config.hidden_dims {
        return Err(std::io::Error::other(format!(
            "active opacity hidden pair {low_hidden},{high_hidden} out of range for hidden_dims={}",
            model.config.hidden_dims
        ))
        .into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    for hidden in [low_hidden, high_hidden] {
        for value in &mut model.weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
            *value = 0.0;
        }
        for output in 0..output_dims {
            model.weights.w2[output * model.config.hidden_dims + hidden] = 0.0;
        }
    }

    model.weights.b1[low_hidden] = 1.0;
    model.weights.w1[low_hidden * input_dims + 3] = 1.0;
    model.weights.b1[high_hidden] = 0.0;
    model.weights.w1[high_hidden * input_dims + 3] = 1.0;

    let opacity_out = model.config.spatial_dims + 3;
    let opacity_base = opacity_out * model.config.hidden_dims;
    model.weights.w2[opacity_base + low_hidden] = active_opacity_gain;
    model.weights.w2[opacity_base + high_hidden] = -active_opacity_gain;

    Ok(low_hidden)
}

fn add_growth_3d_opacity_update_bias(
    model: &mut NpaModel,
    opacity_bias: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "opacity bias retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "opacity bias retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !opacity_bias.is_finite() {
        return Err(std::io::Error::other("opacity_bias must be finite").into());
    }
    let opacity_out = model.config.spatial_dims + 3;
    if opacity_out >= model.config.update_dims() || opacity_out >= model.weights.b2.len() {
        return Err(std::io::Error::other("opacity output index out of range").into());
    }
    model.weights.b2[opacity_out] += opacity_bias;
    Ok(())
}

fn add_growth_3d_material_opacity_update_bias(
    model: &mut NpaModel,
    opacity_bias: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "material opacity bias retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "material opacity bias retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !opacity_bias.is_finite() {
        return Err(std::io::Error::other("material opacity_bias must be finite").into());
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) else {
        return Err(std::io::Error::other("material opacity channel is unavailable").into());
    };
    let opacity_out = model.config.spatial_dims + material_channel;
    if opacity_out >= model.config.update_dims() || opacity_out >= model.weights.b2.len() {
        return Err(std::io::Error::other("material opacity output index out of range").into());
    }
    model.weights.b2[opacity_out] += opacity_bias;
    Ok(())
}

#[allow(dead_code)]
fn torus_growth_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims <= 3 || config.hidden_dims < 6 {
        return Err(std::io::Error::other(format!(
            "uv torus growth requires 3D config, state_dims > 3, and hidden_dims >= 6; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    for axis in 0..3 {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.w1[pos_hidden * input_dims + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + axis] = -1.0;

        weights.w2[axis * config.hidden_dims + pos_hidden] = UV_TORUS_MOTION_GAIN;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -UV_TORUS_MOTION_GAIN;

        let residual_out = config.spatial_dims + axis;
        weights.w2[residual_out * config.hidden_dims + pos_hidden] = -UV_TORUS_RESIDUAL_DECAY;
        weights.w2[residual_out * config.hidden_dims + neg_hidden] = UV_TORUS_RESIDUAL_DECAY;
    }
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;

    Ok(NpaModel { config, weights })
}

fn torus_morphogen_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    seed_frame_morphogen_model(config)
}

fn seed_frame_morphogen_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.position_features {
        return Err(std::io::Error::other(
            "seed-frame morphogen model must not use absolute position_features",
        )
        .into());
    }
    torus_growth_model(config)
}

#[allow(dead_code)]
fn torus_growth_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let scales = [0.56_f32, 0.72, 0.88];
    let mut rng = StdRng::seed_from_u64(0x703d_5eed);

    for row in 0..rows {
        let scale = scales[row % scales.len()];
        let sample = uv_torus_sample(row, rows, scale);
        let structured_position = [
            sample.position[0] * UV_TORUS_INITIAL_SCALE,
            sample.position[1] * UV_TORUS_INITIAL_SCALE,
            sample.position[2] * UV_TORUS_INITIAL_SCALE,
        ];
        let dense_position = uv_torus_dense_seed_position(&mut rng, scale);
        let initial_position = if row % 2 == 0 {
            structured_position
        } else {
            dense_position
        };
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        for axis in 0..3 {
            let residual = sample.position[axis] - initial_position[axis];
            features[feature_base + axis] = residual;
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual;
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual;
        }
        features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
        target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(sample.position, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

fn torus_morphogen_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_6d0f);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        uv_torus_mesh_target(scales[0]),
        uv_torus_mesh_target(scales[1]),
        uv_torus_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(target, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

fn teapot_morphogen_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_07d0);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        utah_teapot_mesh_target(scales[0]),
        utah_teapot_mesh_target(scales[1]),
        utah_teapot_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = utah_teapot_training_position(row, scale, &targets[scale_idx], &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = utah_teapot_tail_state_color(target, &targets[scale_idx]);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

fn torus_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "torus field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_f13d);
    let scales = [UV_TORUS_FIELD_SCALE];
    let targets = [uv_torus_mesh_target(scales[0])];
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = UV_TORUS_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = uv_torus_tail_state_color(target, scale);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    UV_TORUS_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

fn teapot_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "teapot field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_f13d);
    let scale = UV_TORUS_FIELD_SCALE;
    let target_mesh = utah_teapot_mesh_target(scale);
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let position = utah_teapot_training_position(row, scale, &target_mesh, &mut rng);
        let projection = target_mesh.project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = TEAPOT_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = utah_teapot_tail_state_color(target, &target_mesh);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    TEAPOT_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

#[derive(Clone, Copy)]
struct MeshFieldRolloutBatchConfig {
    max_rows: usize,
    particle_count: usize,
    rollout_steps: usize,
    rollouts: usize,
    temporal_samples: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    motion_gain: f32,
    max_update_norm: f32,
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    extent_gain: f32,
    color_gain: f32,
    aux_state_gain: f32,
    opacity_gain: f32,
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
    front_motion_gate: bool,
    preserve_opacity_update: bool,
}

#[derive(Clone, Copy)]
struct MeshLocalTrainingConfig {
    max_rows: usize,
    particle_count: usize,
    rollout_steps: usize,
    rollouts: usize,
    temporal_samples: usize,
    training_rounds: usize,
    total_steps: usize,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    motion_gain: f32,
    max_update_norm: f32,
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    extent_gain: f32,
    color_gain: f32,
    aux_state_gain: f32,
    opacity_gain: f32,
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
    front_motion_gate: bool,
    preserve_opacity_update: bool,
    sgd: SgdConfig,
}

fn merge_supervised_batches(mut lhs: SupervisedBatch, rhs: SupervisedBatch) -> SupervisedBatch {
    lhs.features.extend(rhs.features);
    lhs.target_update.extend(rhs.target_update);
    lhs
}

fn run_refreshed_mesh_local_training(
    model: &mut NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshLocalTrainingConfig,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    if cfg.total_steps == 0 {
        return Err(std::io::Error::other("local mesh training requires at least one step").into());
    }
    let rounds = cfg.training_rounds.max(1);
    let mut history = Vec::new();
    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;
    let mut best_loss = f32::MAX;
    let mut rows = cfg.max_rows;
    let mut steps_done = 0usize;

    for round in 0..rounds {
        if steps_done >= cfg.total_steps {
            break;
        }
        let remaining_steps = cfg.total_steps - steps_done;
        let rounds_left = rounds - round;
        let round_steps = remaining_steps.div_ceil(rounds_left).max(1);
        let batch = mesh_local_rollout_supervised_batch(
            model,
            grid,
            target,
            MeshFieldRolloutBatchConfig {
                max_rows: cfg.max_rows,
                particle_count: cfg.particle_count,
                rollout_steps: cfg.rollout_steps,
                rollouts: cfg.rollouts,
                temporal_samples: cfg.temporal_samples,
                seed: cfg
                    .seed
                    .wrapping_add((round as u64).wrapping_mul(0x51ed_f00d)),
                seed_scale: cfg.seed_scale,
                seed_mode: cfg.seed_mode,
                motion_gain: cfg.motion_gain,
                max_update_norm: cfg.max_update_norm,
                coverage_gain: cfg.coverage_gain,
                coverage_samples: cfg.coverage_samples,
                coverage_mode: cfg.coverage_mode,
                coverage_softness: cfg.coverage_softness,
                coverage_repulsion_gain: cfg.coverage_repulsion_gain,
                coverage_gap_gain: cfg.coverage_gap_gain,
                coverage_repulsion_radius: cfg.coverage_repulsion_radius,
                coverage_normal_weight: cfg.coverage_normal_weight,
                extent_gain: cfg.extent_gain,
                color_gain: cfg.color_gain,
                aux_state_gain: cfg.aux_state_gain,
                opacity_gain: cfg.opacity_gain,
                front_opacity_gain: cfg.front_opacity_gain,
                front_radius: cfg.front_radius,
                front_max_opacity_update: cfg.front_max_opacity_update,
                front_motion_gate: cfg.front_motion_gate,
                preserve_opacity_update: cfg.preserve_opacity_update,
            },
        )?;
        let report = run_supervised_training(
            model,
            &batch,
            TrainingRunConfig {
                steps: round_steps,
                report_interval: round_steps.max(1),
                sgd: cfg.sgd,
            },
        )?;
        initial_loss.get_or_insert(report.initial_loss);
        rows = report.rows;
        final_loss = report.final_loss;
        best_loss = best_loss.min(report.best_loss);
        for entry in report.history {
            history.push(TrainingHistoryEntry {
                step: steps_done + entry.step,
                loss: entry.loss,
                grad_norm: entry.grad_norm,
                grad_scale: entry.grad_scale,
            });
        }
        steps_done += round_steps;
    }

    Ok(TrainingRunReport {
        steps: steps_done,
        rows,
        initial_loss: initial_loss.unwrap_or(final_loss),
        final_loss,
        best_loss,
        history,
    })
}

fn mesh_field_rollout_supervised_batch(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, true)
}

fn mesh_local_rollout_supervised_batch(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, false)
}

fn mesh_rollout_supervised_batch(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
    require_position_features: bool,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    if !model.config.position_features {
        if require_position_features {
            return Err(std::io::Error::other(
                "mesh field rollout rows require position_features=true",
            )
            .into());
        }
    } else if !require_position_features {
        return Err(std::io::Error::other(
            "mesh local rollout rows require position_features=false",
        )
        .into());
    }
    if cfg.max_rows == 0 || cfg.particle_count == 0 || cfg.rollouts == 0 {
        return Err(std::io::Error::other("mesh rollout rows require non-zero sizes").into());
    }

    let mut features = Vec::new();
    let mut target_update = Vec::new();
    let mut remaining_rows = cfg.max_rows;
    let snapshot_steps = mesh_rollout_snapshot_steps(cfg.rollout_steps, cfg.temporal_samples);
    let total_snapshots = cfg.rollouts.saturating_mul(snapshot_steps.len()).max(1);
    let distributed_row_limit = cfg.max_rows.div_ceil(total_snapshots).max(1);
    for rollout_idx in 0..cfg.rollouts {
        if remaining_rows == 0 {
            break;
        }
        let (mut positions, mut states) = seed_particles_scaled(
            1,
            cfg.particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed
                .wrapping_add((rollout_idx as u64).wrapping_mul(0x9e37_79b9)),
            cfg.seed_mode,
            cfg.seed_scale,
        );
        let mut current_step = 0usize;
        for &snapshot_step in &snapshot_steps {
            while current_step < snapshot_step {
                let step =
                    model.step_cpu(&positions, &states, 1, cfg.particle_count, grid, 1.0, None)?;
                positions = step.next_positions;
                states = step.next_states;
                current_step += 1;
            }
            let row_limit = if snapshot_steps.len() == 1 {
                remaining_rows
            } else {
                remaining_rows.min(distributed_row_limit)
            };
            let rows = append_mesh_rollout_snapshot_rows(
                model,
                grid,
                target,
                &cfg,
                &positions,
                &states,
                row_limit,
                &mut features,
                &mut target_update,
            )?;
            remaining_rows = remaining_rows.saturating_sub(rows);
            if remaining_rows == 0 {
                break;
            }
        }
    }

    if features.is_empty() {
        return Err(std::io::Error::other("mesh rollout rows produced no data").into());
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}

fn mesh_rollout_snapshot_steps(rollout_steps: usize, temporal_samples: usize) -> Vec<usize> {
    let samples = temporal_samples.max(1);
    if samples == 1 {
        return vec![rollout_steps];
    }
    if rollout_steps == 0 {
        return vec![0];
    }
    let mut steps = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let step = sample_idx * rollout_steps / (samples - 1);
        if steps.last().copied() != Some(step) {
            steps.push(step);
        }
    }
    if steps.last().copied() != Some(rollout_steps) {
        steps.push(rollout_steps);
    }
    steps
}

#[allow(clippy::too_many_arguments)]
fn append_mesh_rollout_snapshot_rows(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &MeshFieldRolloutBatchConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    max_rows: usize,
    features: &mut Vec<f32>,
    target_update: &mut Vec<f32>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let row_budget = cfg.particle_count.min(max_rows);
    if row_budget == 0 {
        return Ok(0);
    }
    let step = model.step_cpu(positions, states, 1, cfg.particle_count, grid, 1.0, None)?;
    let mut rollout_target_update = mesh_field_target_update_for_rows(
        &model.config,
        target,
        positions,
        states,
        cfg.motion_gain,
        cfg.max_update_norm,
        cfg.color_gain,
        cfg.aux_state_gain,
        cfg.opacity_gain,
        cfg.front_opacity_gain,
        cfg.front_radius,
        cfg.front_max_opacity_update,
        cfg.front_motion_gate,
    );
    add_target_coverage_updates_for_rows(
        &model.config,
        target,
        positions,
        &mut rollout_target_update,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.seed_scale,
        cfg.max_update_norm,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        cfg.front_radius,
    );
    add_target_extent_updates_for_rows(
        &model.config,
        target,
        positions,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        &mut rollout_target_update,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.front_radius,
    );
    if cfg.preserve_opacity_update && model.config.state_dims > 3 {
        let output_dims = model.config.update_dims();
        for row in 0..cfg.particle_count.min(positions.len()) {
            let update_base = row * output_dims + model.config.spatial_dims + 3;
            let state_base = row * model.config.state_dims + 3;
            if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                rollout_target_update[update_base] = step.ds[state_base];
            }
            if let Some(channel) = growth_3d_material_opacity_channel(model.config.state_dims) {
                if channel != 3 {
                    let update_base = row * output_dims + model.config.spatial_dims + channel;
                    let state_base = row * model.config.state_dims + channel;
                    if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                        rollout_target_update[update_base] = step.ds[state_base];
                    }
                }
            }
        }
    }
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let row_indices = mesh_rollout_row_indices(
        &rollout_target_update,
        output_dims,
        cfg.particle_count,
        row_budget,
    );
    for row in row_indices {
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        features
            .extend_from_slice(&step.perception.features[feature_base..feature_base + input_dims]);
        target_update
            .extend_from_slice(&rollout_target_update[update_base..update_base + output_dims]);
    }
    Ok(row_budget)
}

fn mesh_rollout_row_indices(
    target_update: &[f32],
    output_dims: usize,
    particle_count: usize,
    row_budget: usize,
) -> Vec<usize> {
    let rows = particle_count.min(row_budget);
    if rows >= particle_count {
        return (0..particle_count).collect();
    }
    if rows == 0 || output_dims == 0 {
        return Vec::new();
    }

    let spread_budget = (rows / 4).max(1).min(rows);
    let mut selected = vec![false; particle_count];
    let mut row_indices = Vec::with_capacity(rows);
    for row in spread_row_indices(particle_count, spread_budget) {
        if row < particle_count && !selected[row] {
            selected[row] = true;
            row_indices.push(row);
        }
    }

    let mut scored_rows = (0..particle_count)
        .map(|row| {
            let base = row * output_dims;
            let score = target_update
                .get(base..base + output_dims)
                .unwrap_or(&[])
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value * value)
                .sum::<f32>();
            (row, score)
        })
        .collect::<Vec<_>>();
    scored_rows.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    for (row, score) in scored_rows {
        if row_indices.len() >= rows {
            break;
        }
        if score <= 0.0 || selected[row] {
            continue;
        }
        selected[row] = true;
        row_indices.push(row);
    }
    if row_indices.len() < rows {
        for row in spread_row_indices(particle_count, particle_count) {
            if row_indices.len() >= rows {
                break;
            }
            if !selected[row] {
                selected[row] = true;
                row_indices.push(row);
            }
        }
    }
    row_indices
}

fn mesh_field_target_update_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    motion_gain: f32,
    max_update_norm: f32,
    color_gain: f32,
    aux_state_gain: f32,
    opacity_gain: f32,
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
    front_motion_gate: bool,
) -> Vec<f32> {
    let rows = positions.len();
    let output_dims = config.update_dims();
    let mut target_update = vec![0.0; rows * output_dims];
    let front_targets = local_front_opacity_targets(
        config,
        positions,
        states,
        front_opacity_gain,
        front_radius,
        front_max_opacity_update,
    );
    let front_weights = if front_motion_gate {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let target_radius = target
        .vertices
        .iter()
        .map(|vertex| {
            (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt()
        })
        .fold(1.0e-4_f32, f32::max);
    for (row, position) in positions.iter().enumerate() {
        let projection = target.project([position[0], position[1], position[2]]);
        let update_base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        for axis in 0..3 {
            target_update[update_base + axis] =
                front_weight * motion_gain * projection.residual[axis];
        }
        let update_norm = (target_update[update_base].powi(2)
            + target_update[update_base + 1].powi(2)
            + target_update[update_base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && update_norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / update_norm;
            for axis in 0..3 {
                target_update[update_base + axis] *= scale;
            }
        }

        let state_base = row * config.state_dims;
        if config.state_dims >= 3 {
            for axis in 0..3 {
                let target_coordinate = projection.closest[axis] / target_radius.max(1.0e-4);
                target_update[update_base + config.spatial_dims + axis] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_COORDINATE_GAIN
                    * (target_coordinate - states[state_base + axis]);
            }
        }
        if config.state_dims > 3 {
            target_update[update_base + config.spatial_dims + 3] = front_targets[row];
        }
        if let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims) {
            let current_opacity = states[state_base + opacity_channel];
            let surface_band = (target_radius * 0.10).max(0.04);
            let surface_weight = (1.0 - projection.distance / surface_band).clamp(0.0, 1.0);
            let target_opacity = GROWTH_3D_INACTIVE_OPACITY_LOGIT
                + surface_weight
                    * (UV_TORUS_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
            let direct_opacity_update =
                front_weight * opacity_gain * (target_opacity - current_opacity);
            target_update[update_base + config.spatial_dims + opacity_channel] +=
                direct_opacity_update;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = [
                projection.color[0] - 0.5,
                projection.color[1] - 0.5,
                projection.color[2] - 0.5,
            ];
            for channel in 0..3 {
                let current_tail = states[state_base + tail + channel];
                target_update[update_base + config.spatial_dims + tail + channel] =
                    front_weight * color_gain * (target_tail[channel] - current_tail);
            }
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            for axis in 0..3 {
                let channel = UV_TORUS_NORMAL_STATE_OFFSET + axis;
                let current = states[state_base + channel];
                target_update[update_base + config.spatial_dims + channel] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_ORIENTATION_GAIN
                    * (projection.normal[axis] - current);
            }
            let current_signed_distance =
                states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
            target_update
                [update_base + config.spatial_dims + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_SIGNED_DISTANCE_GAIN
                    * (projection.signed_distance - current_signed_distance);
        }
    }
    target_update
}

fn local_front_opacity_targets(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
) -> Vec<f32> {
    let rows = positions.len();
    let mut updates = vec![0.0; rows];
    if config.state_dims <= 3
        || rows == 0
        || front_opacity_gain <= 0.0
        || front_radius <= 0.0
        || front_max_opacity_update <= 0.0
    {
        return updates;
    }

    let dormant_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let current_opacity = states[state_base + 3];
        let mut target_opacity = if front_weights[row] >= 1.0 {
            UV_TORUS_FIELD_OPACITY_TARGET
        } else {
            dormant_target
        };

        if front_weights[row] > 0.0 && front_weights[row] < 1.0 {
            target_opacity = dormant_target
                + front_weights[row] * (UV_TORUS_FIELD_OPACITY_TARGET - dormant_target);
        }

        let delta = front_opacity_gain * (target_opacity - current_opacity);
        updates[row] = delta.clamp(-front_max_opacity_update, front_max_opacity_update);
    }

    updates
}

fn local_front_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> Vec<f32> {
    let rows = positions.len();
    let mut weights = vec![0.0; rows];
    if config.state_dims <= 3 || rows == 0 || front_radius <= 0.0 {
        return weights;
    }
    let front_radius2 = front_radius * front_radius;
    let active_threshold = -1.0_f32;
    for (row, position) in positions.iter().enumerate() {
        let current_opacity = states[row * config.state_dims + 3];
        if current_opacity > active_threshold {
            weights[row] = 1.0;
            continue;
        }

        let mut nearest_active_distance2 = f32::MAX;
        for (other_row, other_position) in positions.iter().enumerate() {
            let other_opacity = states[other_row * config.state_dims + 3];
            if other_opacity <= active_threshold {
                continue;
            }
            let dx = position[0] - other_position[0];
            let dy = position[1] - other_position[1];
            let dz = position[2] - other_position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_active_distance2 {
                nearest_active_distance2 = distance2;
            }
        }
        if nearest_active_distance2 <= front_radius2 {
            weights[row] = (1.0 - (nearest_active_distance2 / front_radius2).sqrt()).max(0.0);
        }
    }
    weights
}

fn add_target_coverage_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    target_update: &mut [f32],
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    front_states: Option<&[f32]>,
    front_radius: f32,
) {
    if coverage_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let rows = positions.len();
    let output_dims = config.update_dims();
    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));

    if coverage_mode != CoverageUpdateModeArg::HardNearest {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        if eligible_rows.is_empty() {
            return;
        }
        let coverage_updates = match coverage_mode {
            CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                coverage_softness,
                coverage_repulsion_gain,
                coverage_repulsion_radius,
                coverage_normal_weight,
                seed_scale,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::HardNearest => unreachable!("handled by outer branch"),
        };
        for row in 0..rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            if front_weight <= 1.0e-3 {
                continue;
            }
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * coverage_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
        if (coverage_mode != CoverageUpdateModeArg::SoftChamfer
            && coverage_repulsion_gain > 0.0
            && coverage_repulsion_gain.is_finite())
            || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
        {
            let mut repulsion_updates = vec![[0.0; 3]; rows];
            if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
                add_surface_tangent_repulsion_to_updates(
                    target,
                    positions,
                    &eligible_rows,
                    coverage_gain,
                    coverage_repulsion_gain,
                    coverage_repulsion_radius,
                    seed_scale,
                    max_update_norm,
                    &mut repulsion_updates,
                );
            }
            add_surface_gap_relocation_to_updates(
                target,
                positions,
                &eligible_rows,
                coverage_gain,
                coverage_gap_gain,
                coverage_samples,
                coverage_normal_weight,
                seed_scale,
                max_update_norm,
                &mut repulsion_updates,
            );
            for row in 0..rows {
                let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
                if front_weight <= 1.0e-3 {
                    continue;
                }
                let base = row * output_dims;
                for axis in 0..3 {
                    target_update[base + axis] += front_weight * repulsion_updates[row][axis];
                }
                clamp_target_motion_update(target_update, base, max_update_norm);
            }
        }
        return;
    }

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            if front_weights
                .as_ref()
                .is_some_and(|weights| weights[row] <= 1.0e-3)
            {
                continue;
            }
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }

        residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
        residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
        residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
        counts[best_row] += 1;
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        let base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        let scale = coverage_gain * front_weight / count as f32;
        for axis in 0..3 {
            target_update[base + axis] += residual_sums[row][axis] * scale;
        }
        clamp_target_motion_update(target_update, base, max_update_norm);
    }
    if (coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite())
        || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
    {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        let mut repulsion_updates = vec![[0.0; 3]; rows];
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        add_surface_gap_relocation_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_gap_gain,
            coverage_samples,
            coverage_normal_weight,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        for row in eligible_rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * repulsion_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
    }
}

fn clamp_target_motion_update(target_update: &mut [f32], base: usize, max_update_norm: f32) {
    let norm = (target_update[base].powi(2)
        + target_update[base + 1].powi(2)
        + target_update[base + 2].powi(2))
    .sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for axis in 0..3 {
            target_update[base + axis] *= clamp;
        }
    }
}

fn add_target_extent_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    front_states: Option<&[f32]>,
    target_update: &mut [f32],
    extent_gain: f32,
    max_update_norm: f32,
    front_radius: f32,
) {
    if extent_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));
    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_rows = 0usize;
    for (row, position) in positions.iter().enumerate() {
        if front_weights
            .as_ref()
            .is_some_and(|weights| weights[row] <= 1.0e-3)
        {
            continue;
        }
        active_rows += 1;
        for axis in 0..3 {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_rows == 0 {
        return;
    }

    let (target_min, target_max) = target.bounds();
    let output_dims = config.update_dims();
    for (row, position) in positions.iter().enumerate() {
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        if front_weight <= 1.0e-3 {
            continue;
        }
        let base = row * output_dims;
        for axis in 0..3 {
            let active_extent = (active_max[axis] - active_min[axis]).max(1.0e-4);
            let t = ((position[axis] - active_min[axis]) / active_extent).clamp(0.0, 1.0);
            let min_weight = (1.0 - t).powi(3);
            let max_weight = t.powi(3);
            let residual = min_weight * (target_min[axis] - position[axis])
                + max_weight * (target_max[axis] - position[axis]);
            target_update[base + axis] += extent_gain * front_weight * residual;
        }
        let norm = (target_update[base].powi(2)
            + target_update[base + 1].powi(2)
            + target_update[base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                target_update[base + axis] *= clamp;
            }
        }
    }
}

fn torus_implicit_training_position(row: usize, scale: f32, rng: &mut StdRng) -> [f32; 3] {
    match row % 4 {
        0 => uv_torus_dense_seed_position(rng, scale),
        1 => {
            let surface = uv_torus_continuous_surface_position(rng, scale);
            [
                surface[0] + rng.random_range(-0.18..0.18) * scale,
                surface[1] + rng.random_range(-0.18..0.18) * scale,
                surface[2] + rng.random_range(-0.18..0.18) * scale,
            ]
        }
        2 => uv_torus_continuous_volume_position(rng, scale),
        _ => {
            let radius = scale * (1.0 + UV_TORUS_MINOR_RATIO) * 0.95;
            [
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
            ]
        }
    }
}

fn utah_teapot_training_position(
    row: usize,
    scale: f32,
    target: &TriangleMeshTarget,
    rng: &mut StdRng,
) -> [f32; 3] {
    match row % 4 {
        0 => utah_teapot_dense_seed_position(rng, target),
        1 => {
            let sample = target.surface_sample(row);
            [
                sample.position[0] + rng.random_range(-0.14..0.14) * scale,
                sample.position[1] + rng.random_range(-0.14..0.14) * scale,
                sample.position[2] + rng.random_range(-0.14..0.14) * scale,
            ]
        }
        2 => target.near_surface_query(row * 17 + 3, rng.random_range(-0.16..0.16) * scale),
        _ => [
            rng.random_range(-1.15..1.15) * scale,
            rng.random_range(-0.70..0.70) * scale,
            rng.random_range(-0.55..0.75) * scale,
        ],
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct TargetCoverageStats {
    mean_distance: f32,
    max_distance: f32,
    covered_fraction: f32,
}

fn target_coverage_threshold(seed_scale: f32) -> f32 {
    (seed_scale.max(1.0e-4) * 0.18).max(0.04)
}

fn target_coverage_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    if positions.is_empty() {
        return TargetCoverageStats {
            mean_distance: f32::MAX,
            max_distance: f32::MAX,
            covered_fraction: 0.0,
        };
    }
    let samples = samples.max(1);
    let mut sum_distance = 0.0_f32;
    let mut max_distance = 0.0_f32;
    let mut covered = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_distance2 = f32::MAX;
        for position in positions {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            best_distance2 = best_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        let distance = best_distance2.sqrt();
        sum_distance += distance;
        max_distance = max_distance.max(distance);
        if distance <= threshold {
            covered += 1;
        }
    }

    TargetCoverageStats {
        mean_distance: sum_distance / samples as f32,
        max_distance,
        covered_fraction: covered as f32 / samples as f32,
    }
}

fn active_target_coverage_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    target_coverage_stats(&active_positions, target, samples, threshold)
}

fn active_surface_coverage_profile(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_coverage_profile(&active_positions, target, samples, threshold, bins)
}

fn surface_coverage_profile(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let samples = samples.max(1);
    let bins = bins.max(1).min(samples);
    let mut bin_samples = vec![0usize; bins];
    let mut bin_covered = vec![0usize; bins];
    let mut assigned_counts = vec![0usize; positions.len()];
    let mut covered_assigned_counts = vec![0usize; positions.len()];
    let mut covered = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bins / samples).min(bins - 1);
        bin_samples[bin] += 1;
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if positions.is_empty() || !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_distance2.sqrt() <= threshold {
            covered += 1;
            bin_covered[bin] += 1;
            covered_assigned_counts[best_row] += 1;
        }
    }

    let bin_covered_fractions = bin_samples
        .iter()
        .zip(bin_covered.iter())
        .map(|(samples, covered)| {
            if *samples == 0 {
                0.0
            } else {
                *covered as f32 / *samples as f32
            }
        })
        .collect::<Vec<_>>();
    let empty_bins = bin_covered.iter().filter(|covered| **covered == 0).count();
    let covered_bins = bins.saturating_sub(empty_bins);
    let min_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let mean_bin_covered_fraction =
        bin_covered_fractions.iter().copied().sum::<f32>() / bins as f32;
    let assigned_particles = assigned_counts.iter().filter(|count| **count > 0).count();
    let covered_assigned_particles = covered_assigned_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let max_assigned_samples = assigned_counts.iter().copied().max().unwrap_or(0);
    let max_covered_assigned_samples = covered_assigned_counts.iter().copied().max().unwrap_or(0);

    SurfaceCoverageProfileReport {
        samples,
        bins,
        threshold,
        covered_fraction: covered as f32 / samples as f32,
        covered_bin_fraction: covered_bins as f32 / bins as f32,
        empty_bins,
        min_bin_covered_fraction: if min_bin_covered_fraction.is_finite() {
            min_bin_covered_fraction
        } else {
            0.0
        },
        mean_bin_covered_fraction,
        max_bin_covered_fraction,
        assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            assigned_particles as f32 / positions.len() as f32
        },
        covered_assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            covered_assigned_particles as f32 / positions.len() as f32
        },
        max_assigned_sample_fraction: max_assigned_samples as f32 / samples as f32,
        max_covered_assigned_sample_fraction: max_covered_assigned_samples as f32 / samples as f32,
        bin_covered_fractions,
    }
}

fn torus_angular_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    scale: f32,
    threshold: f32,
    ring_bins: usize,
    tube_bins: usize,
) -> TorusAngularCoverageReport {
    let ring_bins = ring_bins.max(1);
    let tube_bins = tube_bins.max(1);
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    if active_positions.is_empty() {
        return TorusAngularCoverageReport {
            ring_bins,
            tube_bins,
            threshold,
            covered_joint_bins: 0,
            covered_ring_bins: 0,
            covered_tube_bins: 0,
            joint_coverage_fraction: 0.0,
            ring_coverage_fraction: 0.0,
            tube_coverage_fraction: 0.0,
            max_ring_gap_bins: ring_bins,
            max_tube_gap_bins: tube_bins,
            mean_distance: f32::MAX,
            max_distance: f32::MAX,
        };
    }
    let mut joint_covered = vec![false; ring_bins * tube_bins];
    let mut ring_covered = vec![false; ring_bins];
    let mut tube_covered = vec![false; tube_bins];
    let mut sum_distance = 0.0_f32;
    let mut max_distance = 0.0_f32;

    for ring in 0..ring_bins {
        let theta = std::f32::consts::TAU * (ring as f32 + 0.5) / ring_bins as f32;
        let theta_cos = theta.cos();
        let theta_sin = theta.sin();
        for tube in 0..tube_bins {
            let phi = std::f32::consts::TAU * (tube as f32 + 0.5) / tube_bins as f32;
            let radial = major + minor * phi.cos();
            let sample = [radial * theta_cos, radial * theta_sin, minor * phi.sin()];
            let distance = nearest_position3_distance(sample, &active_positions);
            sum_distance += distance;
            max_distance = max_distance.max(distance);
            if distance <= threshold {
                joint_covered[ring * tube_bins + tube] = true;
                ring_covered[ring] = true;
                tube_covered[tube] = true;
            }
        }
    }

    let total_bins = ring_bins * tube_bins;
    let covered_joint_bins = joint_covered.iter().filter(|covered| **covered).count();
    let covered_ring_bins = ring_covered.iter().filter(|covered| **covered).count();
    let covered_tube_bins = tube_covered.iter().filter(|covered| **covered).count();
    TorusAngularCoverageReport {
        ring_bins,
        tube_bins,
        threshold,
        covered_joint_bins,
        covered_ring_bins,
        covered_tube_bins,
        joint_coverage_fraction: covered_joint_bins as f32 / total_bins.max(1) as f32,
        ring_coverage_fraction: covered_ring_bins as f32 / ring_bins as f32,
        tube_coverage_fraction: covered_tube_bins as f32 / tube_bins as f32,
        max_ring_gap_bins: max_circular_false_run(&ring_covered),
        max_tube_gap_bins: max_circular_false_run(&tube_covered),
        mean_distance: sum_distance / total_bins.max(1) as f32,
        max_distance,
    }
}

fn nearest_position3_distance(sample: [f32; 3], positions: &[[f32; 4]]) -> f32 {
    if positions.is_empty() {
        return f32::MAX;
    }
    positions
        .iter()
        .map(|position| {
            let dx = sample[0] - position[0];
            let dy = sample[1] - position[1];
            let dz = sample[2] - position[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f32::MAX, f32::min)
}

fn max_circular_false_run(values: &[bool]) -> usize {
    if values.is_empty() || values.iter().all(|value| *value) {
        return 0;
    }
    if values.iter().all(|value| !*value) {
        return values.len();
    }
    let mut max_run = 0usize;
    let mut run = 0usize;
    for idx in 0..values.len() * 2 {
        if values[idx % values.len()] {
            max_run = max_run.max(run);
            run = 0;
        } else {
            run += 1;
            max_run = max_run.max(run.min(values.len()));
        }
    }
    max_run.min(values.len())
}

fn growth_3d_extent_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dExtentReport {
    let (target_bounds_min, target_bounds_max) = target.bounds();
    let target_extent = [
        target_bounds_max[0] - target_bounds_min[0],
        target_bounds_max[1] - target_bounds_min[1],
        target_bounds_max[2] - target_bounds_min[2],
    ];
    let target_max_radius = target
        .vertices
        .iter()
        .map(|position| {
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt()
        })
        .fold(0.0_f32, f32::max);

    let mut active_bounds_min = [f32::MAX; 3];
    let mut active_bounds_max = [f32::MIN; 3];
    let mut active_count = 0usize;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity <= -1.0 {
            continue;
        }
        active_count += 1;
        for axis in 0..3 {
            active_bounds_min[axis] = active_bounds_min[axis].min(position[axis]);
            active_bounds_max[axis] = active_bounds_max[axis].max(position[axis]);
        }
        final_active_max_radius = final_active_max_radius.max(
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt(),
        );
    }

    if active_count == 0 {
        active_bounds_min = [0.0; 3];
        active_bounds_max = [0.0; 3];
    }
    let final_active_extent = [
        active_bounds_max[0] - active_bounds_min[0],
        active_bounds_max[1] - active_bounds_min[1],
        active_bounds_max[2] - active_bounds_min[2],
    ];
    let axis_extent_ratio = [
        final_active_extent[0] / target_extent[0].max(1.0e-6),
        final_active_extent[1] / target_extent[1].max(1.0e-6),
        final_active_extent[2] / target_extent[2].max(1.0e-6),
    ];
    let min_axis_extent_ratio = axis_extent_ratio
        .iter()
        .copied()
        .fold(f32::MAX, f32::min)
        .min(1.0e6);
    let target_diag = (target_extent[0] * target_extent[0]
        + target_extent[1] * target_extent[1]
        + target_extent[2] * target_extent[2])
        .sqrt();
    let active_diag = (final_active_extent[0] * final_active_extent[0]
        + final_active_extent[1] * final_active_extent[1]
        + final_active_extent[2] * final_active_extent[2])
        .sqrt();

    Growth3dExtentReport {
        target_bounds_min,
        target_bounds_max,
        final_active_bounds_min: active_bounds_min,
        final_active_bounds_max: active_bounds_max,
        target_extent,
        final_active_extent,
        axis_extent_ratio,
        min_axis_extent_ratio,
        bbox_diagonal_ratio: active_diag / target_diag.max(1.0e-6),
        target_max_radius,
        final_active_max_radius,
        max_radius_ratio: final_active_max_radius / target_max_radius.max(1.0e-6),
    }
}

fn growth_3d_catalog_sanity_report(
    target: MeshTargetArg,
    render_loss: &MultiViewRenderLossReport,
) -> Growth3dCatalogSanityReport {
    let (max_total_loss, min_density_psnr_db, min_color_psnr_db, min_depth_psnr_db) = match target {
        MeshTargetArg::Torus => (0.90, 0.95, 16.0, 14.8),
        MeshTargetArg::Teapot => (0.85, 0.95, 18.0, 18.0),
    };
    let passed = render_loss.total_loss <= max_total_loss
        && render_loss.density_psnr_db >= min_density_psnr_db
        && render_loss.color_psnr_db >= min_color_psnr_db
        && render_loss.depth_psnr_db >= min_depth_psnr_db;
    Growth3dCatalogSanityReport {
        passed,
        max_total_loss,
        min_density_psnr_db,
        min_color_psnr_db,
        min_depth_psnr_db,
        total_loss: render_loss.total_loss,
        density_psnr_db: render_loss.density_psnr_db,
        color_psnr_db: render_loss.color_psnr_db,
        depth_psnr_db: render_loss.depth_psnr_db,
    }
}

#[allow(clippy::too_many_arguments)]
fn growth_3d_strict_checks_report(
    position_features: bool,
    local_conditionless_lineage: bool,
    non_opacity_seed_abs_max: f32,
    final_opacity: Growth3dOpacityStats,
    initial_color_state: Growth3dColorStateReport,
    final_color_state: Growth3dColorStateReport,
    permutation_consistency: &Growth3dPermutationReport,
    activation: &Growth3dActivationReport,
    initial_active_surface: Growth3dSurfaceStats,
    final_active_surface: Growth3dSurfaceStats,
    final_active_surface_tail: Growth3dSurfaceTailReport,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    torus_angular_coverage: Option<&TorusAngularCoverageReport>,
    motion: &Growth3dMotionReport,
    front: &Growth3dFrontReport,
    temporal: &Growth3dTemporalReport,
    mean_final_displacement: f32,
    seed_scale: f32,
    particle_count: usize,
    render_loss_passed: bool,
) -> Growth3dStrictChecksReport {
    let no_position_features = !position_features;
    let neutral_non_opacity_seed_state = non_opacity_seed_abs_max <= 1.0e-6;
    let sparse_active_seed =
        activation.active_seed_count > 0 && activation.active_seed_count < particle_count / 8;
    let active_count_growth = activation.final_active_count > activation.active_seed_count * 4;
    let newly_activated_fraction = activation.newly_activated_fraction >= 0.50;
    let active_front_expanded =
        activation.final_active_max_radius > growth_3d_seed_radius(seed_scale);
    let nonzero_motion = motion.peak_mean_dx > 0.01;
    let sustained_motion =
        motion.active_step_fraction >= 0.50 && motion.sustained_step_fraction >= 0.25;
    let local_front_coherent = front.passed;
    let temporal_activation_progressive = temporal.progressive_activation;
    let temporal_geometry_progressive = temporal.geometry_progressive;
    let mean_displacement_growth = mean_final_displacement > growth_3d_seed_radius(seed_scale);
    let bounded_final_opacity =
        final_opacity.finite && final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT;
    let color_state_emerged = initial_color_state.available
        && final_color_state.available
        && initial_color_state.finite
        && final_color_state.finite
        && initial_color_state.active_max_abs <= 1.0e-6
        && final_color_state.active_mean_abs >= initial_color_state.active_mean_abs + 0.02
        && final_color_state.active_max_abs >= 0.05
        && final_color_state.active_channel_stddev_mean >= 0.02;
    let permutation_consistent = permutation_consistency.passed;
    let surface_mean_improved =
        final_active_surface.mean_distance < initial_active_surface.mean_distance * 0.85;
    let surface_max_bounded = final_active_surface.max_distance < GROWTH_3D_SURFACE_MAX_DISTANCE;
    let surface_tail_bounded = final_active_surface_tail.p99_distance
        < GROWTH_3D_SURFACE_MAX_DISTANCE
        && final_active_surface_tail.over_threshold_fraction <= 0.005
        && final_active_surface_tail.opacity_weighted_over_threshold_fraction <= 0.005;
    let target_coverage_mean_improved =
        final_target_coverage.mean_distance < initial_target_coverage.mean_distance * 0.85;
    let target_coverage_max_bounded = final_target_coverage.max_distance < seed_scale;
    let target_coverage_fraction = final_target_coverage.covered_fraction >= 0.60;
    let torus_angular_coverage = torus_angular_coverage.is_none_or(|coverage| {
        coverage.joint_coverage_fraction >= 0.60
            && coverage.tube_coverage_fraction >= 0.75
            && coverage.max_tube_gap_bins <= coverage.tube_bins / 4
    });

    let checks = [
        ("no_position_features", no_position_features),
        ("local_conditionless_lineage", local_conditionless_lineage),
        (
            "neutral_non_opacity_seed_state",
            neutral_non_opacity_seed_state,
        ),
        ("sparse_active_seed", sparse_active_seed),
        ("active_count_growth", active_count_growth),
        ("newly_activated_fraction", newly_activated_fraction),
        ("active_front_expanded", active_front_expanded),
        ("nonzero_motion", nonzero_motion),
        ("sustained_motion", sustained_motion),
        ("local_front_coherent", local_front_coherent),
        (
            "temporal_activation_progressive",
            temporal_activation_progressive,
        ),
        (
            "temporal_geometry_progressive",
            temporal_geometry_progressive,
        ),
        ("mean_displacement_growth", mean_displacement_growth),
        ("bounded_final_opacity", bounded_final_opacity),
        ("color_state_emerged", color_state_emerged),
        ("permutation_consistent", permutation_consistent),
        ("surface_mean_improved", surface_mean_improved),
        ("surface_tail_bounded", surface_tail_bounded),
        (
            "target_coverage_mean_improved",
            target_coverage_mean_improved,
        ),
        ("target_coverage_max_bounded", target_coverage_max_bounded),
        ("target_coverage_fraction", target_coverage_fraction),
        ("torus_angular_coverage", torus_angular_coverage),
        ("render_loss_passed", render_loss_passed),
    ];
    let failure_reasons = checks
        .iter()
        .filter_map(|(name, passed)| (!*passed).then_some(*name))
        .collect::<Vec<_>>();
    let passed = failure_reasons.is_empty();

    Growth3dStrictChecksReport {
        passed,
        no_position_features,
        local_conditionless_lineage,
        neutral_non_opacity_seed_state,
        sparse_active_seed,
        active_count_growth,
        newly_activated_fraction,
        active_front_expanded,
        nonzero_motion,
        sustained_motion,
        local_front_coherent,
        temporal_activation_progressive,
        temporal_geometry_progressive,
        mean_displacement_growth,
        bounded_final_opacity,
        color_state_emerged,
        permutation_consistent,
        surface_mean_improved,
        surface_max_bounded,
        surface_tail_bounded,
        target_coverage_mean_improved,
        target_coverage_max_bounded,
        target_coverage_fraction,
        torus_angular_coverage,
        render_loss_passed,
        failure_reasons,
    }
}

#[allow(clippy::too_many_arguments)]
fn growth_3d_strict_score_report(
    checks: &Growth3dStrictChecksReport,
    initial_active_surface: Growth3dSurfaceStats,
    final_active_surface: Growth3dSurfaceStats,
    final_active_surface_tail: Growth3dSurfaceTailReport,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    seed_scale: f32,
    render_loss: &MultiViewRenderLossReport,
) -> Growth3dStrictScoreReport {
    let surface_mean_ratio = if initial_active_surface.mean_distance.is_finite()
        && initial_active_surface.mean_distance > 1.0e-6
    {
        final_active_surface.mean_distance / initial_active_surface.mean_distance
    } else {
        f32::INFINITY
    };
    let target_coverage_mean_ratio = if initial_target_coverage.mean_distance.is_finite()
        && initial_target_coverage.mean_distance > 1.0e-6
    {
        final_target_coverage.mean_distance / initial_target_coverage.mean_distance
    } else {
        f32::INFINITY
    };

    let hard_failures = [
        checks.no_position_features,
        checks.local_conditionless_lineage,
        checks.neutral_non_opacity_seed_state,
        checks.sparse_active_seed,
        checks.active_count_growth,
        checks.newly_activated_fraction,
        checks.active_front_expanded,
        checks.nonzero_motion,
        checks.sustained_motion,
        checks.local_front_coherent,
        checks.temporal_activation_progressive,
        checks.temporal_geometry_progressive,
        checks.mean_displacement_growth,
        checks.bounded_final_opacity,
        checks.color_state_emerged,
        checks.permutation_consistent,
        checks.torus_angular_coverage,
    ]
    .into_iter()
    .filter(|passed| !passed)
    .count() as f32;
    let hard_failure_penalty = hard_failures * 10.0;
    let surface_mean_penalty = (surface_mean_ratio - 0.85).max(0.0);
    let surface_max_penalty =
        (final_active_surface.max_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let surface_tail_p99_penalty =
        (final_active_surface_tail.p99_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let surface_tail_fraction_penalty = ((final_active_surface_tail.over_threshold_fraction
        - 0.005)
        .max(0.0)
        + (final_active_surface_tail.opacity_weighted_over_threshold_fraction - 0.005).max(0.0))
        * 10.0;
    let target_coverage_mean_penalty = (target_coverage_mean_ratio - 0.85).max(0.0);
    let target_coverage_max_penalty = (final_target_coverage.max_distance - seed_scale).max(0.0);
    let target_coverage_fraction_penalty = (0.60 - final_target_coverage.covered_fraction).max(0.0);
    let render_density_penalty = ((10.0 - render_loss.density_psnr_db).max(0.0)) / 10.0;
    let render_color_penalty = ((12.0 - render_loss.color_psnr_db).max(0.0)) / 12.0;
    let render_depth_penalty = ((14.0 - render_loss.depth_psnr_db).max(0.0)) / 14.0;
    let score = hard_failure_penalty
        + surface_mean_penalty
        + surface_tail_p99_penalty
        + surface_tail_fraction_penalty
        + target_coverage_mean_penalty
        + target_coverage_max_penalty
        + target_coverage_fraction_penalty
        + render_density_penalty
        + render_color_penalty
        + render_depth_penalty;

    Growth3dStrictScoreReport {
        score,
        hard_failure_penalty,
        surface_mean_ratio,
        surface_mean_penalty,
        surface_max_distance: final_active_surface.max_distance,
        surface_max_penalty,
        surface_tail_p99_distance: final_active_surface_tail.p99_distance,
        surface_tail_p99_penalty,
        surface_tail_over_threshold_fraction: final_active_surface_tail.over_threshold_fraction,
        surface_tail_fraction_penalty,
        target_coverage_mean_ratio,
        target_coverage_mean_penalty,
        target_coverage_max_distance: final_target_coverage.max_distance,
        target_coverage_max_penalty,
        target_coverage_fraction: final_target_coverage.covered_fraction,
        target_coverage_fraction_penalty,
        render_density_psnr_db: render_loss.density_psnr_db,
        render_density_penalty,
        render_color_psnr_db: render_loss.color_psnr_db,
        render_color_penalty,
        render_depth_psnr_db: render_loss.depth_psnr_db,
        render_depth_penalty,
    }
}

fn growth_3d_validation_report(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let seeds = eval_seed_list(cfg.seed, &cfg.extra_seeds);
    let mut primary_cfg = cfg.clone();
    primary_cfg.extra_seeds.clear();
    let mut primary = growth_3d_validation_report_single(model_path, target_arg, primary_cfg)?;
    let mut seed_reports = Vec::with_capacity(seeds.len());
    seed_reports.push(growth_3d_robustness_seed_report(&primary));
    for seed in seeds.iter().skip(1) {
        let mut seed_cfg = cfg.clone();
        seed_cfg.seed = *seed;
        seed_cfg.extra_seeds.clear();
        let report = growth_3d_validation_report_single(model_path, target_arg, seed_cfg)?;
        seed_reports.push(growth_3d_robustness_seed_report(&report));
    }
    primary.robustness = growth_3d_robustness_report(seed_reports);
    Ok(primary)
}

fn growth_3d_fail_on_validation_passed(report: &CliGrowth3dValidationReport) -> bool {
    if report.robustness.seed_count > 1 {
        report.robustness.all_gate_passed
    } else {
        report.gate_passed
    }
}

fn growth_3d_validation_report_single(
    model_path: &PathBuf,
    target_arg: MeshTargetArg,
    cfg: Growth3dValidationConfig,
) -> Result<CliGrowth3dValidationReport, Box<dyn std::error::Error>> {
    let manifest = burn_automata::import::load_manifest(model_path)?;
    if manifest.config.spatial_dims != 3 || manifest.config.state_dims <= 3 {
        return Err(std::io::Error::other(format!(
            "growth 3D validation requires spatial_dims=3 and state_dims>3; got spatial_dims={} state_dims={}",
            manifest.config.spatial_dims, manifest.config.state_dims
        ))
        .into());
    }
    let source = manifest.source.clone();
    let source_text = source.as_deref().unwrap_or_default();
    let local_conditionless_lineage = local_conditionless_lineage(source_text);
    let position_features = manifest.config.position_features;
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let target = mesh_target_for_arg(target_arg, cfg.seed_scale);
    let rollout_cfg = RolloutConfig {
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        update_prob: 1.0,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        rollout_cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        rollout_cfg.seed,
        cfg.seed_mode,
        rollout_cfg.seed_scale,
    );

    let mut active_seed_count = 0usize;
    let mut seed_active = Vec::with_capacity(rollout_cfg.particle_count);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }
    let non_opacity_seed_abs_max =
        growth_3d_non_scaffold_seed_abs_max(model.config.state_dims, cfg.seed_mode, &seed_states);

    let trace = run_rollout(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let activation = growth_3d_activation_report(&trace, &seed_active, active_seed_count);
    let final_opacity = growth_3d_opacity_stats(&trace.states, trace.state_dims);
    let initial_color_state = growth_3d_color_state_report(&seed_states, model.config.state_dims);
    let final_color_state = growth_3d_color_state_report(&trace.states, trace.state_dims);
    let permutation_consistency =
        growth_3d_permutation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let seed_perturbation =
        growth_3d_seed_perturbation_report(&model, &grid, &rollout_cfg, cfg.seed_mode)?;
    let initial_surface = growth_3d_surface_stats(&seed_positions, &target);
    let final_surface = growth_3d_surface_stats(&trace.positions, &target);
    let initial_active_surface = growth_3d_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
    );
    let final_active_surface =
        growth_3d_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, &target);
    let initial_active_surface_tail = growth_3d_active_surface_tail_report(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let final_active_surface_tail = growth_3d_active_surface_tail_report(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        GROWTH_3D_SURFACE_MAX_DISTANCE,
    );
    let target_coverage_threshold = target_coverage_threshold(cfg.seed_scale);
    let coverage_samples = cfg.particle_count.max(512);
    let initial_target_coverage = target_coverage_stats(
        &seed_positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_target_coverage = target_coverage_stats(
        &trace.positions,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let initial_active_target_coverage = active_target_coverage_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_active_target_coverage = active_target_coverage_stats(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
    );
    let final_active_surface_coverage_profile = active_surface_coverage_profile(
        &trace.positions,
        &trace.states,
        trace.state_dims,
        &target,
        coverage_samples,
        target_coverage_threshold,
        64,
    );
    let torus_angular_coverage = (target_arg == MeshTargetArg::Torus).then(|| {
        torus_angular_coverage_report(
            &trace.positions,
            &trace.states,
            trace.state_dims,
            cfg.seed_scale,
            target_coverage_threshold,
            TORUS_ANGULAR_COVERAGE_RINGS,
            TORUS_ANGULAR_COVERAGE_TUBES,
        )
    });
    let extent =
        growth_3d_extent_report(&trace.positions, &trace.states, trace.state_dims, &target);
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = coverage_samples;
    }
    let render_loss = mesh_multiview_render_loss_from_trace(&trace, &target, render_cfg)?;
    let catalog_sanity = growth_3d_catalog_sanity_report(target_arg, &render_loss);
    let mean_final_displacement = growth_3d_mean_displacement(&seed_positions, &trace.positions);
    let motion = growth_3d_motion_report(&trace.mean_dx);
    let temporal = growth_3d_temporal_report(
        &model,
        &grid,
        &target,
        rollout_cfg.clone(),
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &seed_active,
        active_seed_count,
        &trace,
        coverage_samples,
        target_coverage_threshold,
    )?;
    let front = growth_3d_front_report(
        &model,
        &grid,
        rollout_cfg,
        cfg.seed_mode,
        &seed_positions,
        &seed_states,
        &trace,
    )?;
    let strict_checks = growth_3d_strict_checks_report(
        position_features,
        local_conditionless_lineage,
        non_opacity_seed_abs_max,
        final_opacity,
        initial_color_state,
        final_color_state,
        &permutation_consistency,
        &activation,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        torus_angular_coverage.as_ref(),
        &motion,
        &front,
        &temporal,
        mean_final_displacement,
        cfg.seed_scale,
        cfg.particle_count,
        render_loss.passed,
    );
    let strict_passed = strict_checks.passed;
    let strict_score = growth_3d_strict_score_report(
        &strict_checks,
        initial_active_surface,
        final_active_surface,
        final_active_surface_tail,
        initial_active_target_coverage,
        final_active_target_coverage,
        cfg.seed_scale,
        &render_loss,
    );
    let catalog_gate_passed = !position_features
        && local_conditionless_lineage
        && non_opacity_seed_abs_max <= 1.0e-6
        && final_opacity.finite
        && final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT
        && strict_checks.color_state_emerged
        && strict_checks.permutation_consistent
        && activation.active_seed_count > 0
        && activation.active_seed_count < cfg.particle_count / 8
        && activation.final_active_count > activation.active_seed_count * 4
        && activation.newly_activated_fraction >= 0.50
        && activation.final_active_max_radius > growth_3d_seed_radius(cfg.seed_scale)
        && motion.peak_mean_dx > 0.01
        && motion.active_step_fraction >= 0.50
        && motion.sustained_step_fraction >= 0.25
        && front.passed
        && mean_final_displacement > growth_3d_seed_radius(cfg.seed_scale)
        && catalog_sanity.passed;
    let gate_passed = match cfg.gate {
        Growth3dValidationGateArg::Strict => strict_passed,
        Growth3dValidationGateArg::CatalogSanity => catalog_gate_passed,
    };

    Ok(CliGrowth3dValidationReport {
        target: target_arg,
        model: model_path.display().to_string(),
        source,
        position_features,
        local_conditionless_lineage,
        particle_count: cfg.particle_count,
        steps: cfg.steps,
        seed: cfg.seed,
        seed_scale: cfg.seed_scale,
        seed_mode: cfg.seed_mode,
        non_opacity_seed_abs_max,
        initial_color_state,
        final_color_state,
        permutation_consistency,
        seed_perturbation,
        mean_final_displacement,
        final_opacity,
        activation,
        initial_surface,
        final_surface,
        initial_active_surface,
        final_active_surface,
        initial_active_surface_tail,
        final_active_surface_tail,
        target_coverage_threshold,
        initial_target_coverage,
        final_target_coverage,
        initial_active_target_coverage,
        final_active_target_coverage,
        final_active_surface_coverage_profile,
        torus_angular_coverage,
        extent,
        motion,
        temporal,
        front,
        max_motion_per_step: motion.peak_mean_dx,
        render_loss,
        strict_checks,
        strict_score,
        catalog_sanity,
        robustness: growth_3d_empty_robustness_report(cfg.seed),
        gate: cfg.gate,
        gate_passed,
        strict_passed,
    })
}

fn growth_3d_robustness_seed_report(
    report: &CliGrowth3dValidationReport,
) -> Growth3dRobustnessSeedReport {
    Growth3dRobustnessSeedReport {
        seed: report.seed,
        gate_passed: report.gate_passed,
        strict_passed: report.strict_passed,
        catalog_sanity_passed: report.catalog_sanity.passed,
        strict_score: report.strict_score.score,
        render_loss: report.render_loss.total_loss,
        density_psnr_db: report.render_loss.density_psnr_db,
        color_psnr_db: report.render_loss.color_psnr_db,
        depth_psnr_db: report.render_loss.depth_psnr_db,
        active_seed_count: report.activation.active_seed_count,
        final_active_count: report.activation.final_active_count,
        newly_activated_fraction: report.activation.newly_activated_fraction,
        final_opacity_max: report.final_opacity.max,
        color_state_emerged: report.strict_checks.color_state_emerged,
        final_active_color_state_mean_abs: report.final_color_state.active_mean_abs,
        final_active_color_state_stddev_mean: report.final_color_state.active_channel_stddev_mean,
        permutation_consistent: report.permutation_consistency.passed,
        permutation_max_position_error: report.permutation_consistency.max_position_error,
        permutation_max_state_error: report.permutation_consistency.max_state_error,
        seed_perturbation_stable: report.seed_perturbation.passed,
        perturbed_newly_activated_fraction: report
            .seed_perturbation
            .perturbed_newly_activated_fraction,
        perturbed_active_count_ratio: report.seed_perturbation.active_count_ratio,
        perturbed_peak_motion_ratio: report.seed_perturbation.peak_motion_ratio,
        local_front_coherent: report.front.passed,
        front_local_newly_activated_fraction: report.front.local_newly_activated_fraction,
        front_max_nearest_previous_active_distance: report
            .front
            .max_nearest_previous_active_distance,
        temporal_activation_progressive: report.temporal.progressive_activation,
        temporal_geometry_progressive: report.temporal.geometry_progressive,
        final_active_target_coverage_fraction: report.final_active_target_coverage.covered_fraction,
        final_active_surface_max: report.final_active_surface.max_distance,
        failure_reasons: report.strict_checks.failure_reasons.clone(),
    }
}

fn growth_3d_robustness_report(
    seeds: Vec<Growth3dRobustnessSeedReport>,
) -> Growth3dRobustnessReport {
    let seed_count = seeds.len();
    let all_gate_passed = seed_count > 0 && seeds.iter().all(|seed| seed.gate_passed);
    let all_catalog_sanity_passed =
        seed_count > 0 && seeds.iter().all(|seed| seed.catalog_sanity_passed);
    let all_strict_passed = seed_count > 0 && seeds.iter().all(|seed| seed.strict_passed);
    let all_temporal_activation_progressive = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.temporal_activation_progressive);
    let all_temporal_geometry_progressive =
        seed_count > 0 && seeds.iter().all(|seed| seed.temporal_geometry_progressive);
    let all_local_front_coherent =
        seed_count > 0 && seeds.iter().all(|seed| seed.local_front_coherent);
    let all_bounded_final_opacity = seed_count > 0
        && seeds
            .iter()
            .all(|seed| seed.final_opacity_max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
    let all_color_state_emerged =
        seed_count > 0 && seeds.iter().all(|seed| seed.color_state_emerged);
    let all_permutation_consistent =
        seed_count > 0 && seeds.iter().all(|seed| seed.permutation_consistent);
    let all_seed_perturbation_stable =
        seed_count > 0 && seeds.iter().all(|seed| seed.seed_perturbation_stable);
    let worst_strict_score = seeds
        .iter()
        .map(|seed| seed.strict_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_render_loss = seeds
        .iter()
        .map(|seed| seed.render_loss)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_density_psnr_db = seeds
        .iter()
        .map(|seed| seed.density_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_color_psnr_db = seeds
        .iter()
        .map(|seed| seed.color_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_depth_psnr_db = seeds
        .iter()
        .map(|seed| seed.depth_psnr_db)
        .fold(f32::INFINITY, f32::min);
    let min_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .min()
        .unwrap_or(0);
    let max_active_seed_count = seeds
        .iter()
        .map(|seed| seed.active_seed_count)
        .max()
        .unwrap_or(0);
    let min_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .min()
        .unwrap_or(0);
    let max_final_active_count = seeds
        .iter()
        .map(|seed| seed.final_active_count)
        .max()
        .unwrap_or(0);
    let min_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_active_growth_ratio = seeds
        .iter()
        .map(|seed| seed.final_active_count as f32 / seed.active_seed_count.max(1) as f32)
        .fold(f32::INFINITY, f32::min);
    let max_final_opacity = seeds
        .iter()
        .map(|seed| seed.final_opacity_max)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_final_active_color_state_mean_abs = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_mean_abs)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_color_state_stddev_mean = seeds
        .iter()
        .map(|seed| seed.final_active_color_state_stddev_mean)
        .fold(f32::INFINITY, f32::min);
    let max_permutation_position_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_position_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_permutation_state_error = seeds
        .iter()
        .map(|seed| seed.permutation_max_state_error)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.perturbed_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_active_count_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_active_count_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::INFINITY, f32::min);
    let max_perturbed_peak_motion_ratio = seeds
        .iter()
        .map(|seed| seed.perturbed_peak_motion_ratio)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_front_nearest_previous_active_distance = seeds
        .iter()
        .map(|seed| seed.front_max_nearest_previous_active_distance)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_front_local_newly_activated_fraction = seeds
        .iter()
        .map(|seed| seed.front_local_newly_activated_fraction)
        .fold(f32::INFINITY, f32::min);
    let min_final_active_target_coverage_fraction = seeds
        .iter()
        .map(|seed| seed.final_active_target_coverage_fraction)
        .fold(f32::INFINITY, f32::min);
    Growth3dRobustnessReport {
        seed_count,
        all_gate_passed,
        all_catalog_sanity_passed,
        all_strict_passed,
        all_temporal_activation_progressive,
        all_temporal_geometry_progressive,
        all_local_front_coherent,
        all_bounded_final_opacity,
        all_color_state_emerged,
        all_permutation_consistent,
        all_seed_perturbation_stable,
        worst_strict_score: if seed_count == 0 {
            f32::INFINITY
        } else {
            worst_strict_score
        },
        max_render_loss: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_render_loss
        },
        min_density_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_density_psnr_db
        },
        min_color_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_color_psnr_db
        },
        min_depth_psnr_db: if seed_count == 0 {
            f32::NEG_INFINITY
        } else {
            min_depth_psnr_db
        },
        min_active_seed_count,
        max_active_seed_count,
        min_final_active_count,
        max_final_active_count,
        min_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_newly_activated_fraction
        },
        min_active_growth_ratio: if seed_count == 0 {
            0.0
        } else {
            min_active_growth_ratio
        },
        max_final_opacity: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_final_opacity
        },
        min_final_active_color_state_mean_abs: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_mean_abs
        },
        min_final_active_color_state_stddev_mean: if seed_count == 0 {
            f32::NAN
        } else {
            min_final_active_color_state_stddev_mean
        },
        max_permutation_position_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_position_error
        },
        max_permutation_state_error: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_permutation_state_error
        },
        min_perturbed_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_newly_activated_fraction
        },
        min_perturbed_active_count_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_active_count_ratio
        },
        max_perturbed_active_count_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_active_count_ratio
        },
        min_perturbed_peak_motion_ratio: if seed_count == 0 {
            0.0
        } else {
            min_perturbed_peak_motion_ratio
        },
        max_perturbed_peak_motion_ratio: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_perturbed_peak_motion_ratio
        },
        max_front_nearest_previous_active_distance: if seed_count == 0 {
            f32::INFINITY
        } else {
            max_front_nearest_previous_active_distance
        },
        min_front_local_newly_activated_fraction: if seed_count == 0 {
            0.0
        } else {
            min_front_local_newly_activated_fraction
        },
        min_final_active_target_coverage_fraction: if seed_count == 0 {
            0.0
        } else {
            min_final_active_target_coverage_fraction
        },
        seeds,
    }
}

fn growth_3d_empty_robustness_report(seed: u64) -> Growth3dRobustnessReport {
    growth_3d_robustness_report(vec![Growth3dRobustnessSeedReport {
        seed,
        gate_passed: false,
        strict_passed: false,
        catalog_sanity_passed: false,
        strict_score: f32::INFINITY,
        render_loss: f32::INFINITY,
        density_psnr_db: f32::NEG_INFINITY,
        color_psnr_db: f32::NEG_INFINITY,
        depth_psnr_db: f32::NEG_INFINITY,
        active_seed_count: 0,
        final_active_count: 0,
        newly_activated_fraction: 0.0,
        final_opacity_max: f32::INFINITY,
        color_state_emerged: false,
        final_active_color_state_mean_abs: f32::NAN,
        final_active_color_state_stddev_mean: f32::NAN,
        permutation_consistent: false,
        permutation_max_position_error: f32::INFINITY,
        permutation_max_state_error: f32::INFINITY,
        seed_perturbation_stable: false,
        perturbed_newly_activated_fraction: 0.0,
        perturbed_active_count_ratio: 0.0,
        perturbed_peak_motion_ratio: 0.0,
        local_front_coherent: false,
        front_local_newly_activated_fraction: 0.0,
        front_max_nearest_previous_active_distance: f32::INFINITY,
        temporal_activation_progressive: false,
        temporal_geometry_progressive: false,
        final_active_target_coverage_fraction: 0.0,
        final_active_surface_max: f32::INFINITY,
        failure_reasons: Vec::new(),
    }])
}

fn growth_3d_seed_has_coordinate_scaffold(seed_mode: ParticleSeed) -> bool {
    matches!(
        seed_mode,
        ParticleSeed::TorusGrowth3d
            | ParticleSeed::TeapotGrowth3d
            | ParticleSeed::TorusSubstrateGrowth3d
            | ParticleSeed::TeapotSubstrateGrowth3d
    )
}

fn growth_3d_non_scaffold_seed_abs_max(
    state_dims: usize,
    seed_mode: ParticleSeed,
    seed_states: &[f32],
) -> f32 {
    let material_opacity_channel = growth_3d_material_opacity_channel(state_dims);
    let allow_coordinate_scaffold = growth_3d_seed_has_coordinate_scaffold(seed_mode);
    let mut abs_max = 0.0_f32;
    for state in seed_states.chunks_exact(state_dims) {
        for (channel, value) in state.iter().enumerate() {
            if channel == GROWTH_3D_LIVENESS_CHANNEL
                || Some(channel) == material_opacity_channel
                || (allow_coordinate_scaffold && channel < 3)
            {
                continue;
            }
            abs_max = abs_max.max(value.abs());
        }
    }
    abs_max
}

fn growth_3d_motion_report(mean_dx: &[f32]) -> Growth3dMotionReport {
    if mean_dx.is_empty() {
        return Growth3dMotionReport {
            first_step_mean_dx: 0.0,
            peak_mean_dx: 0.0,
            peak_step: 0,
            final_step_mean_dx: 0.0,
            mean_dx: 0.0,
            late_mean_dx: 0.0,
            late_to_peak_ratio: 0.0,
            active_step_fraction: 0.0,
            sustained_step_fraction: 0.0,
        };
    }

    let first_step_mean_dx = mean_dx[0];
    let final_step_mean_dx = mean_dx[mean_dx.len() - 1];
    let mut peak_mean_dx = 0.0_f32;
    let mut peak_step = 0usize;
    let mut sum = 0.0_f32;
    for (step, value) in mean_dx.iter().copied().enumerate() {
        sum += value;
        if value > peak_mean_dx {
            peak_mean_dx = value;
            peak_step = step;
        }
    }
    let mean = sum / mean_dx.len() as f32;
    let late_start = mean_dx.len() * 3 / 4;
    let late_slice = &mean_dx[late_start..];
    let late_mean_dx = late_slice.iter().copied().sum::<f32>() / late_slice.len().max(1) as f32;
    let active_threshold = 1.0e-3;
    let sustained_threshold = (peak_mean_dx * 0.05).max(active_threshold);
    let active_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > active_threshold)
        .count();
    let sustained_steps = mean_dx
        .iter()
        .filter(|value| value.is_finite() && **value > sustained_threshold)
        .count();

    Growth3dMotionReport {
        first_step_mean_dx,
        peak_mean_dx,
        peak_step,
        final_step_mean_dx,
        mean_dx: mean,
        late_mean_dx,
        late_to_peak_ratio: if peak_mean_dx > 1.0e-8 {
            late_mean_dx / peak_mean_dx
        } else {
            0.0
        },
        active_step_fraction: active_steps as f32 / mean_dx.len() as f32,
        sustained_step_fraction: sustained_steps as f32 / mean_dx.len() as f32,
    }
}

#[derive(Clone)]
struct Growth3dFrontSnapshot {
    positions: Vec<[f32; 4]>,
    active: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
fn growth_3d_front_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    final_trace: &burn_automata::RolloutTrace,
) -> Result<Growth3dFrontReport, Box<dyn std::error::Error>> {
    let max_allowed_distance = growth_3d_front_distance_threshold(rollout_cfg.seed_scale);
    let mut snapshots = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        let snapshot = if steps == 0 {
            growth_3d_front_snapshot(seed_positions, seed_states, model.config.state_dims)
        } else if steps == rollout_cfg.steps {
            growth_3d_front_snapshot(
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            growth_3d_front_snapshot(&trace.positions, &trace.states, trace.state_dims)
        };
        snapshots.push(snapshot);
    }

    let mut transition_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut local_newly_activated_count = 0usize;
    let mut finite = true;
    let mut sum_nearest = 0.0_f32;
    let mut max_nearest = 0.0_f32;

    for pair in snapshots.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if previous.positions.len() != current.positions.len()
            || previous.active.len() != current.active.len()
        {
            finite = false;
            continue;
        }
        let previous_active_positions = previous
            .positions
            .iter()
            .zip(previous.active.iter())
            .filter_map(|(position, active)| (*active).then_some(*position))
            .collect::<Vec<_>>();
        if previous_active_positions.is_empty() {
            continue;
        }
        let mut transition_newly_activated = 0usize;
        for idx in 0..current.active.len() {
            if !current.active[idx] || previous.active[idx] {
                continue;
            }
            transition_newly_activated += 1;
            newly_activated_count += 1;
            let distance =
                nearest_position_distance(current.positions[idx], &previous_active_positions);
            finite &= distance.is_finite();
            sum_nearest += distance;
            max_nearest = max_nearest.max(distance);
            if distance <= max_allowed_distance {
                local_newly_activated_count += 1;
            }
        }
        if transition_newly_activated > 0 {
            transition_count += 1;
        }
    }

    let local_newly_activated_fraction = if newly_activated_count > 0 {
        local_newly_activated_count as f32 / newly_activated_count as f32
    } else {
        0.0
    };
    let mean_nearest_previous_active_distance = if newly_activated_count > 0 {
        sum_nearest / newly_activated_count as f32
    } else {
        f32::INFINITY
    };
    let passed = finite
        && newly_activated_count > 0
        && transition_count >= 2
        && local_newly_activated_fraction >= 0.90
        && mean_nearest_previous_active_distance <= max_allowed_distance * 0.75;

    Ok(Growth3dFrontReport {
        transition_count,
        newly_activated_count,
        local_newly_activated_count,
        local_newly_activated_fraction,
        mean_nearest_previous_active_distance,
        max_nearest_previous_active_distance: if newly_activated_count > 0 {
            max_nearest
        } else {
            f32::INFINITY
        },
        max_allowed_distance,
        finite,
        passed,
    })
}

fn growth_3d_front_snapshot(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
) -> Growth3dFrontSnapshot {
    let active = positions
        .iter()
        .enumerate()
        .map(|(idx, _)| state_dims > 3 && states[idx * state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    Growth3dFrontSnapshot {
        positions: positions.to_vec(),
        active,
    }
}

fn nearest_position_distance(position: [f32; 4], candidates: &[[f32; 4]]) -> f32 {
    candidates
        .iter()
        .map(|candidate| {
            ((position[0] - candidate[0]).powi(2)
                + (position[1] - candidate[1]).powi(2)
                + (position[2] - candidate[2]).powi(2))
            .sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

fn growth_3d_front_distance_threshold(seed_scale: f32) -> f32 {
    growth_3d_seed_radius(seed_scale) * 2.5
}

#[allow(clippy::too_many_arguments)]
fn growth_3d_temporal_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    rollout_cfg: RolloutConfig,
    seed_mode: ParticleSeed,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
    seed_active: &[bool],
    active_seed_count: usize,
    final_trace: &burn_automata::RolloutTrace,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Result<Growth3dTemporalReport, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for steps in growth_3d_temporal_sample_steps(rollout_cfg.steps) {
        if steps == 0 {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                seed_positions,
                seed_states,
                model.config.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else if steps == rollout_cfg.steps {
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &final_trace.positions,
                &final_trace.states,
                final_trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..rollout_cfg.clone()
                },
                seed_mode,
            )?;
            samples.push(growth_3d_temporal_sample_report(
                steps,
                &trace.positions,
                &trace.states,
                trace.state_dims,
                seed_positions,
                seed_active,
                target,
                coverage_samples,
                coverage_threshold,
            ));
        }
    }

    let first_growth_step = samples
        .iter()
        .find(|sample| {
            sample.active_count > active_seed_count
                && sample.active_count >= active_seed_count.saturating_mul(2).max(1)
        })
        .map(|sample| sample.steps);
    let half_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.50)
        .map(|sample| sample.steps);
    let full_activation_step = samples
        .iter()
        .find(|sample| sample.active_fraction >= 0.95)
        .map(|sample| sample.steps);
    let activation_span_steps =
        if let (Some(first), Some(full)) = (first_growth_step, full_activation_step) {
            full.saturating_sub(first)
        } else {
            0
        };
    let progressive_activation = match (
        first_growth_step,
        half_activation_step,
        full_activation_step,
    ) {
        (Some(first), Some(half), Some(full)) => {
            first < half && half < full && activation_span_steps >= rollout_cfg.steps / 4
        }
        _ => false,
    };
    let (surface_mean_ratio, target_coverage_mean_ratio, target_coverage_fraction_delta) =
        match (samples.first(), samples.last()) {
            (Some(initial), Some(final_sample)) => {
                let surface_mean_ratio = if initial.active_surface.mean_distance.is_finite()
                    && initial.active_surface.mean_distance > 1.0e-6
                {
                    final_sample.active_surface.mean_distance / initial.active_surface.mean_distance
                } else {
                    f32::INFINITY
                };
                let target_coverage_mean_ratio =
                    if initial.target_coverage.mean_distance.is_finite()
                        && initial.target_coverage.mean_distance > 1.0e-6
                    {
                        final_sample.target_coverage.mean_distance
                            / initial.target_coverage.mean_distance
                    } else {
                        f32::INFINITY
                    };
                let target_coverage_fraction_delta = final_sample.target_coverage.covered_fraction
                    - initial.target_coverage.covered_fraction;
                (
                    surface_mean_ratio,
                    target_coverage_mean_ratio,
                    target_coverage_fraction_delta,
                )
            }
            _ => (f32::INFINITY, f32::INFINITY, 0.0),
        };
    let geometry_progressive = target_coverage_mean_ratio < 0.85
        && target_coverage_fraction_delta >= 0.10
        && surface_mean_ratio < 0.95;

    Ok(Growth3dTemporalReport {
        samples,
        first_growth_step,
        half_activation_step,
        full_activation_step,
        activation_span_steps,
        progressive_activation,
        surface_mean_ratio,
        target_coverage_mean_ratio,
        target_coverage_fraction_delta,
        geometry_progressive,
    })
}

fn growth_3d_temporal_sample_steps(steps: usize) -> Vec<usize> {
    let mut samples = vec![0, steps];
    let mut step = 1usize;
    while step < steps {
        samples.push(step);
        step = step.saturating_mul(2);
        if step == 0 {
            break;
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

#[allow(clippy::too_many_arguments)]
fn growth_3d_temporal_sample_report(
    steps: usize,
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    seed_positions: &[[f32; 4]],
    seed_active: &[bool],
    target: &TriangleMeshTarget,
    coverage_samples: usize,
    coverage_threshold: f32,
) -> Growth3dTemporalSampleReport {
    let mut active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut active_radius_sum = 0.0_f32;
    let mut active_max_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let opacity = states[idx * state_dims + 3];
        if opacity > -1.0 {
            active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            active_radius_sum += radius;
            active_max_radius = active_max_radius.max(radius);
        }
    }

    Growth3dTemporalSampleReport {
        steps,
        active_count,
        active_fraction: active_count as f32 / positions.len().max(1) as f32,
        newly_activated_count,
        final_active_mean_radius: if active_count > 0 {
            active_radius_sum / active_count as f32
        } else {
            0.0
        },
        final_active_max_radius: active_max_radius,
        mean_displacement: growth_3d_mean_displacement(seed_positions, positions),
        active_surface: growth_3d_active_surface_stats(positions, states, state_dims, target),
        target_coverage: target_coverage_stats(
            positions,
            target,
            coverage_samples,
            coverage_threshold,
        ),
    }
}

fn growth_3d_activation_report(
    trace: &burn_automata::RolloutTrace,
    seed_active: &[bool],
    active_seed_count: usize,
) -> Growth3dActivationReport {
    let mut final_active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut final_active_radius_sum = 0.0_f32;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in trace.positions.iter().enumerate() {
        let opacity = trace.states[idx * trace.state_dims + 3];
        if opacity > -1.0 {
            final_active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            final_active_radius_sum += radius;
            final_active_max_radius = final_active_max_radius.max(radius);
        }
    }
    let inactive_seed_count = trace.particle_count.saturating_sub(active_seed_count);
    Growth3dActivationReport {
        active_seed_count,
        inactive_seed_count,
        final_active_count,
        newly_activated_count,
        newly_activated_fraction: newly_activated_count as f32 / inactive_seed_count.max(1) as f32,
        final_active_mean_radius: final_active_radius_sum / final_active_count.max(1) as f32,
        final_active_max_radius,
    }
}

fn growth_3d_opacity_stats(states: &[f32], state_dims: usize) -> Growth3dOpacityStats {
    if state_dims <= 3 || states.is_empty() {
        return Growth3dOpacityStats {
            finite: false,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: f32::NAN,
            active_min: f32::INFINITY,
            active_max: f32::NEG_INFINITY,
            active_mean: f32::NAN,
            active_count: 0,
            max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
        };
    }

    let mut finite = true;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    let mut active_min = f32::INFINITY;
    let mut active_max = f32::NEG_INFINITY;
    let mut active_sum = 0.0_f32;
    let mut active_count = 0usize;
    for state in states.chunks_exact(state_dims) {
        let opacity = state[3];
        finite &= opacity.is_finite();
        min = min.min(opacity);
        max = max.max(opacity);
        sum += opacity;
        count += 1;
        if opacity > -1.0 {
            active_min = active_min.min(opacity);
            active_max = active_max.max(opacity);
            active_sum += opacity;
            active_count += 1;
        }
    }

    Growth3dOpacityStats {
        finite,
        min,
        max,
        mean: sum / count.max(1) as f32,
        active_min: if active_count == 0 {
            f32::INFINITY
        } else {
            active_min
        },
        active_max: if active_count == 0 {
            f32::NEG_INFINITY
        } else {
            active_max
        },
        active_mean: if active_count == 0 {
            f32::NAN
        } else {
            active_sum / active_count as f32
        },
        active_count,
        max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
    }
}

fn growth_3d_color_state_report(states: &[f32], state_dims: usize) -> Growth3dColorStateReport {
    if state_dims < 6 || states.is_empty() {
        return Growth3dColorStateReport {
            available: false,
            finite: false,
            count: 0,
            active_count: 0,
            mean_abs: f32::NAN,
            max_abs: f32::NAN,
            active_mean_abs: f32::NAN,
            active_max_abs: f32::NAN,
            active_channel_stddev: [f32::NAN; 3],
            active_channel_stddev_mean: f32::NAN,
        };
    }

    let tail = state_dims - 3;
    let mut finite = true;
    let mut count = 0usize;
    let mut active_count = 0usize;
    let mut sum_abs = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut active_sum_abs = 0.0_f32;
    let mut active_max_abs = 0.0_f32;
    let mut active_sum = [0.0_f32; 3];
    let mut active_sum_sq = [0.0_f32; 3];

    for state in states.chunks_exact(state_dims) {
        count += 1;
        let mut particle_max_abs = 0.0_f32;
        for channel in 0..3 {
            let value = state[tail + channel];
            finite &= value.is_finite();
            particle_max_abs = particle_max_abs.max(value.abs());
        }
        sum_abs += particle_max_abs;
        max_abs = max_abs.max(particle_max_abs);

        if state[3] > -1.0 {
            active_count += 1;
            active_sum_abs += particle_max_abs;
            active_max_abs = active_max_abs.max(particle_max_abs);
            for channel in 0..3 {
                let value = state[tail + channel];
                active_sum[channel] += value;
                active_sum_sq[channel] += value * value;
            }
        }
    }

    let mut active_channel_stddev = [f32::NAN; 3];
    if active_count > 0 {
        for channel in 0..3 {
            let mean = active_sum[channel] / active_count as f32;
            let variance = (active_sum_sq[channel] / active_count as f32 - mean * mean).max(0.0);
            active_channel_stddev[channel] = variance.sqrt();
        }
    }
    let active_channel_stddev_mean = if active_count > 0 {
        active_channel_stddev.iter().sum::<f32>() / 3.0
    } else {
        f32::NAN
    };

    Growth3dColorStateReport {
        available: true,
        finite,
        count,
        active_count,
        mean_abs: sum_abs / count.max(1) as f32,
        max_abs,
        active_mean_abs: if active_count > 0 {
            active_sum_abs / active_count as f32
        } else {
            f32::NAN
        },
        active_max_abs: if active_count > 0 {
            active_max_abs
        } else {
            f32::NAN
        },
        active_channel_stddev,
        active_channel_stddev_mean,
    }
}

fn growth_3d_permutation_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dPermutationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(2, 256);
    let steps = cfg.steps.min(8);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut order = (0..particle_count).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x9a55_19e3_7ac3);
    order.shuffle(&mut rng);

    let mut shuffled_positions = vec![[0.0; 4]; particle_count];
    let mut shuffled_states = vec![0.0; states.len()];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        shuffled_positions[shuffled_idx] = positions[source_idx];
        let src = source_idx * model.config.state_dims;
        let dst = shuffled_idx * model.config.state_dims;
        shuffled_states[dst..dst + model.config.state_dims]
            .copy_from_slice(&states[src..src + model.config.state_dims]);
    }

    let shuffled = run_rollout_from_state(
        model,
        grid,
        shuffled_positions,
        shuffled_states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut inverse_order = vec![0usize; particle_count];
    for (shuffled_idx, &source_idx) in order.iter().enumerate() {
        inverse_order[source_idx] = shuffled_idx;
    }

    let mut max_position_error = 0.0_f32;
    let mut sum_position_error = 0.0_f32;
    let mut max_state_error = 0.0_f32;
    let mut sum_state_error = 0.0_f32;
    let mut state_count = 0usize;

    for (source_idx, &shuffled_idx) in inverse_order.iter().enumerate() {
        let base_position = base.positions[source_idx];
        let shuffled_position = shuffled.positions[shuffled_idx];
        let position_error = ((base_position[0] - shuffled_position[0]).powi(2)
            + (base_position[1] - shuffled_position[1]).powi(2)
            + (base_position[2] - shuffled_position[2]).powi(2))
        .sqrt();
        max_position_error = max_position_error.max(position_error);
        sum_position_error += position_error;

        let base_state = source_idx * model.config.state_dims;
        let shuffled_state = shuffled_idx * model.config.state_dims;
        for channel in 0..model.config.state_dims {
            let state_error = (base.states[base_state + channel]
                - shuffled.states[shuffled_state + channel])
                .abs();
            max_state_error = max_state_error.max(state_error);
            sum_state_error += state_error;
            state_count += 1;
        }
    }

    let mean_position_error = sum_position_error / particle_count.max(1) as f32;
    let mean_state_error = sum_state_error / state_count.max(1) as f32;
    let passed = max_position_error <= 1.0e-3 && max_state_error <= 1.0e-3;

    Ok(Growth3dPermutationReport {
        particle_count,
        steps,
        max_position_error,
        mean_position_error,
        max_state_error,
        mean_state_error,
        passed,
    })
}

fn growth_3d_seed_perturbation_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> Result<Growth3dSeedPerturbationReport, Box<dyn std::error::Error>> {
    let particle_count = cfg.particle_count.clamp(32, 512);
    let steps = cfg.steps.clamp(1, 32);
    let jitter_radius = (growth_3d_seed_radius(cfg.seed_scale) * 0.10).max(cfg.seed_scale * 0.002);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let mut seed_active = Vec::with_capacity(particle_count);
    let mut active_seed_count = 0usize;
    for state in states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
    }

    let base = run_rollout_from_state(
        model,
        grid,
        positions.clone(),
        states.clone(),
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let mut perturbed_positions = positions;
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5eed_937d_3d);
    for position in &mut perturbed_positions {
        for value in position.iter_mut().take(3) {
            *value += rng.random_range(-jitter_radius..=jitter_radius);
        }
    }
    let perturbed = run_rollout_from_state(
        model,
        grid,
        perturbed_positions,
        states,
        1,
        particle_count,
        steps,
        cfg.dt,
    )?;

    let base_activation = growth_3d_activation_report(&base, &seed_active, active_seed_count);
    let perturbed_activation =
        growth_3d_activation_report(&perturbed, &seed_active, active_seed_count);
    let base_motion = growth_3d_motion_report(&base.mean_dx);
    let perturbed_motion = growth_3d_motion_report(&perturbed.mean_dx);
    let base_color = growth_3d_color_state_report(&base.states, base.state_dims);
    let perturbed_color = growth_3d_color_state_report(&perturbed.states, perturbed.state_dims);

    let active_count_ratio = finite_ratio(
        perturbed_activation.final_active_count as f32,
        base_activation.final_active_count.max(1) as f32,
    );
    let final_active_max_radius_ratio = finite_ratio(
        perturbed_activation.final_active_max_radius,
        base_activation.final_active_max_radius,
    );
    let peak_motion_ratio = finite_ratio(perturbed_motion.peak_mean_dx, base_motion.peak_mean_dx);
    let color_state_mean_abs_ratio =
        finite_ratio(perturbed_color.active_mean_abs, base_color.active_mean_abs);

    let base_growth = base_activation.final_active_count > active_seed_count.max(1) * 2
        && base_activation.newly_activated_fraction >= 0.25
        && base_motion.peak_mean_dx > 1.0e-3;
    let perturbed_growth = perturbed_activation.final_active_count > active_seed_count.max(1) * 2
        && perturbed_activation.newly_activated_fraction >= 0.25
        && perturbed_motion.peak_mean_dx > 1.0e-3;
    let comparable_growth = (0.50..=2.00).contains(&active_count_ratio)
        && (0.50..=2.00).contains(&final_active_max_radius_ratio)
        && (0.25..=4.00).contains(&peak_motion_ratio);
    let passed = base_growth && perturbed_growth && comparable_growth;

    Ok(Growth3dSeedPerturbationReport {
        particle_count,
        steps,
        jitter_radius,
        seed: cfg.seed,
        active_seed_count,
        base_final_active_count: base_activation.final_active_count,
        perturbed_final_active_count: perturbed_activation.final_active_count,
        active_count_ratio,
        base_newly_activated_fraction: base_activation.newly_activated_fraction,
        perturbed_newly_activated_fraction: perturbed_activation.newly_activated_fraction,
        base_final_active_max_radius: base_activation.final_active_max_radius,
        perturbed_final_active_max_radius: perturbed_activation.final_active_max_radius,
        final_active_max_radius_ratio,
        base_peak_mean_dx: base_motion.peak_mean_dx,
        perturbed_peak_mean_dx: perturbed_motion.peak_mean_dx,
        peak_motion_ratio,
        base_color_state_mean_abs: base_color.active_mean_abs,
        perturbed_color_state_mean_abs: perturbed_color.active_mean_abs,
        color_state_mean_abs_ratio,
        passed,
    })
}

fn finite_ratio(numerator: f32, denominator: f32) -> f32 {
    if !numerator.is_finite() || !denominator.is_finite() {
        return f32::NAN;
    }
    if denominator.abs() <= 1.0e-8 {
        if numerator.abs() <= 1.0e-8 {
            1.0
        } else if numerator.is_sign_positive() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    } else {
        numerator / denominator
    }
}

fn run_rollout_from_state(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    mut positions: Vec<[f32; 4]>,
    mut states: Vec<f32>,
    batch_size: usize,
    particle_count: usize,
    steps: usize,
    dt: f32,
) -> Result<burn_automata::RolloutTrace, Box<dyn std::error::Error>> {
    let mut mean_dx = Vec::with_capacity(steps);
    for _ in 0..steps {
        let step = model.step_cpu(
            &positions,
            &states,
            batch_size,
            particle_count,
            grid,
            dt,
            None,
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|delta| (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok(burn_automata::RolloutTrace {
        positions,
        states,
        batch_size,
        particle_count,
        state_dims: model.config.state_dims,
        steps,
        mean_dx,
    })
}

fn growth_3d_surface_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    for position in positions {
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
    }
    Growth3dSurfaceStats {
        mean_distance: sum_distance / positions.len().max(1) as f32,
        max_distance,
    }
}

fn growth_3d_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> Growth3dSurfaceStats {
    let mut max_distance = 0.0_f32;
    let mut sum_distance = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let projection = target.project([position[0], position[1], position[2]]);
        max_distance = max_distance.max(projection.distance);
        sum_distance += projection.distance;
        count += 1;
    }
    Growth3dSurfaceStats {
        mean_distance: if count > 0 {
            sum_distance / count as f32
        } else {
            f32::INFINITY
        },
        max_distance: if count > 0 {
            max_distance
        } else {
            f32::INFINITY
        },
    }
}

fn growth_3d_active_surface_tail_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    threshold: f32,
) -> Growth3dSurfaceTailReport {
    let mut distances = Vec::new();
    let mut max_distance = 0.0_f32;
    let mut over_threshold_count = 0usize;
    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut weighted_over_threshold_sum = 0.0_f32;

    if state_dims > 3 {
        for (idx, position) in positions.iter().enumerate() {
            let opacity_logit = states[idx * state_dims + 3];
            if opacity_logit <= -1.0 {
                continue;
            }
            let projection = target.project([position[0], position[1], position[2]]);
            let distance = projection.distance;
            let weight = sigmoid_unit(opacity_logit);
            max_distance = max_distance.max(distance);
            if distance >= threshold {
                over_threshold_count += 1;
                weighted_over_threshold_sum += weight;
            }
            weighted_sum += distance * weight;
            weight_sum += weight;
            distances.push(distance);
        }
    }

    if distances.is_empty() {
        return Growth3dSurfaceTailReport {
            count: 0,
            threshold,
            p95_distance: f32::INFINITY,
            p99_distance: f32::INFINITY,
            max_distance: f32::INFINITY,
            over_threshold_count: 0,
            over_threshold_fraction: 0.0,
            opacity_weighted_mean_distance: f32::INFINITY,
            opacity_weighted_over_threshold_fraction: 0.0,
        };
    }

    distances.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = distances.len();
    Growth3dSurfaceTailReport {
        count,
        threshold,
        p95_distance: percentile_from_sorted(&distances, 0.95),
        p99_distance: percentile_from_sorted(&distances, 0.99),
        max_distance,
        over_threshold_count,
        over_threshold_fraction: over_threshold_count as f32 / count as f32,
        opacity_weighted_mean_distance: weighted_sum / weight_sum.max(1.0e-8),
        opacity_weighted_over_threshold_fraction: weighted_over_threshold_sum
            / weight_sum.max(1.0e-8),
    }
}

fn percentile_from_sorted(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() as f32 * clamped).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

fn sigmoid_unit(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn sigmoid_unit_derivative(value: f32) -> f32 {
    let sigmoid = sigmoid_unit(value);
    sigmoid * (1.0 - sigmoid)
}

fn growth_3d_mean_displacement(initial: &[[f32; 4]], final_positions: &[[f32; 4]]) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}

fn mesh_rollout_report_for_cases(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cases: &[MeshRolloutCaseConfig],
) -> Result<MeshRolloutReport, Box<dyn std::error::Error>> {
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_mean_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_mean_surface_distance = 0.0_f32;
    let mut max_target_coverage_distance = 0.0_f32;
    let mut sum_mean_target_coverage_distance = 0.0_f32;
    let mut min_target_coverage_fraction = 1.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_mean_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = mesh_rollout_case_report(&trace, target, *case);
        max_initial_surface_distance =
            max_initial_surface_distance.max(report.max_initial_surface_distance);
        sum_mean_initial_surface_distance += report.mean_initial_surface_distance;
        max_surface_distance = max_surface_distance.max(report.max_surface_distance);
        sum_mean_surface_distance += report.mean_surface_distance;
        max_target_coverage_distance =
            max_target_coverage_distance.max(report.max_target_coverage_distance);
        sum_mean_target_coverage_distance += report.mean_target_coverage_distance;
        min_target_coverage_fraction =
            min_target_coverage_fraction.min(report.target_coverage_fraction);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        sum_mean_color_target_error += report.mean_color_target_error;
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);

        let case_passed = report.finite
            && report.max_initial_surface_distance >= 0.08
            && report.first_motion_per_step >= 1.0e-3
            && report.max_motion_per_step >= 1.0e-3
            && report.mean_surface_improvement_ratio >= 0.15
            && report.max_surface_distance <= 0.36
            && report.mean_surface_distance <= 0.16
            && report.mean_target_coverage_distance <= 0.20
            && report.max_target_coverage_distance <= 0.72
            && report.target_coverage_fraction >= 0.60
            && report.max_color_target_error <= 0.42
            && report.mean_color_target_error <= 0.16
            && report.max_opacity_target_error <= 2.0e-2
            && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT;
        passed &= case_passed;
        case_reports.push(report);
    }

    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }
    Ok(MeshRolloutReport {
        passed,
        max_initial_surface_distance,
        mean_initial_surface_distance: sum_mean_initial_surface_distance
            / cases.len().max(1) as f32,
        max_surface_distance,
        mean_surface_distance: sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement: sum_mean_initial_surface_distance / cases.len().max(1) as f32
            - sum_mean_surface_distance / cases.len().max(1) as f32,
        mean_surface_improvement_ratio: if sum_mean_initial_surface_distance > 0.0 {
            1.0 - sum_mean_surface_distance / sum_mean_initial_surface_distance
        } else {
            0.0
        },
        max_target_coverage_distance,
        mean_target_coverage_distance: sum_mean_target_coverage_distance
            / cases.len().max(1) as f32,
        min_target_coverage_fraction,
        max_color_target_error,
        mean_color_target_error: sum_mean_color_target_error / cases.len().max(1) as f32,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

fn mesh_rollout_case_report(
    trace: &burn_automata::RolloutTrace,
    target: &TriangleMeshTarget,
    case: MeshRolloutCaseConfig,
) -> MeshRolloutCaseReport {
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        case.seed_scale,
    );
    let expected_final_opacity_logit = UV_TORUS_FIELD_OPACITY_TARGET;
    let mut max_initial_surface_distance = 0.0_f32;
    let mut sum_initial_surface_distance = 0.0_f32;
    let mut max_surface_distance = 0.0_f32;
    let mut sum_surface_distance = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let initial_projection = target.project([
            initial_position[0],
            initial_position[1],
            initial_position[2],
        ]);
        max_initial_surface_distance =
            max_initial_surface_distance.max(initial_projection.distance);
        sum_initial_surface_distance += initial_projection.distance;

        let projection = target.project([position[0], position[1], position[2]]);
        max_surface_distance = max_surface_distance.max(projection.distance);
        sum_surface_distance += projection.distance;

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = projection.color;
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());
    let mean_initial_surface_distance =
        sum_initial_surface_distance / trace.positions.len().max(1) as f32;
    let mean_surface_distance = sum_surface_distance / trace.positions.len().max(1) as f32;
    let coverage_threshold = target_coverage_threshold(case.seed_scale);
    let coverage = target_coverage_stats(
        &trace.positions,
        target,
        trace.particle_count.max(512),
        coverage_threshold,
    );

    MeshRolloutCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        max_initial_surface_distance,
        mean_initial_surface_distance,
        max_surface_distance,
        mean_surface_distance,
        mean_surface_improvement: mean_initial_surface_distance - mean_surface_distance,
        mean_surface_improvement_ratio: if mean_initial_surface_distance > 0.0 {
            1.0 - mean_surface_distance / mean_initial_surface_distance
        } else {
            0.0
        },
        target_coverage_threshold: coverage_threshold,
        max_target_coverage_distance: coverage.max_distance,
        mean_target_coverage_distance: coverage.mean_distance,
        target_coverage_fraction: coverage.covered_fraction,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}

fn torus_robustness_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    torus_robustness_report_for_cases(model, grid, TORUS_ROBUSTNESS_CASES)
}

fn torus_robustness_report_for_cases(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cases: &[TorusRobustnessCaseConfig],
) -> Result<TorusRobustnessReport, Box<dyn std::error::Error>> {
    let opacity_update_index = model.config.spatial_dims + 3;
    let trained_opacity_delta = model.weights.b2[opacity_update_index];
    let field_mode = model.config.position_features;
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut max_target_position_error = 0.0_f32;
    let mut sum_mean_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut first_motion_per_step = f32::MAX;
    let mut max_motion_per_step = 0.0_f32;
    let mut max_opacity_target_error = 0.0_f32;
    let mut min_final_opacity = f32::MAX;
    let mut max_final_opacity = f32::MIN;
    let mut passed = true;

    for case in cases {
        let cfg = RolloutConfig {
            particle_count: case.particle_count,
            steps: case.steps,
            update_prob: 1.0,
            seed: case.seed,
            seed_scale: case.seed_scale,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(model, grid, &cfg, case.seed_mode)?;
        let report = torus_robustness_case_report(&trace, *case);
        max_target_position_error = max_target_position_error.max(report.max_target_position_error);
        sum_mean_target_position_error += report.mean_target_position_error;
        max_torus_surface_error = max_torus_surface_error.max(report.max_torus_surface_error);
        max_color_target_error = max_color_target_error.max(report.max_color_target_error);
        first_motion_per_step = first_motion_per_step.min(report.first_motion_per_step);
        max_motion_per_step = max_motion_per_step.max(report.max_motion_per_step);
        max_opacity_target_error = max_opacity_target_error.max(report.max_opacity_target_error);
        min_final_opacity = min_final_opacity.min(report.min_final_opacity_logit);
        max_final_opacity = max_final_opacity.max(report.max_final_opacity_logit);
        let case_passed = if field_mode {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_torus_surface_error <= 1.2e-1
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 2.5e-1
                && report.max_opacity_target_error <= 2.0
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        } else {
            report.finite
                && report.max_initial_target_position_error >= 0.12
                && report.first_motion_per_step >= 1.0e-3
                && report.max_motion_per_step >= 1.0e-3
                && report.max_target_position_error <= 8.0e-2
                && report.max_torus_surface_error <= 8.0e-2
                && report.max_final_radial >= report.torus_outer_radius * 0.80
                && report.max_final_abs_z
                    >= (report.torus_outer_radius - report.torus_inner_radius) * 0.20
                && report.max_color_target_error <= 3.0e-2
                && report.max_opacity_target_error <= 1.0e-2
                && report.min_final_opacity_logit > UV_TORUS_INITIAL_OPACITY_LOGIT
        };
        passed &= case_passed;
        case_reports.push(report);
    }

    if !field_mode {
        passed &= (trained_opacity_delta - UV_TORUS_OPACITY_GROWTH_DELTA).abs() <= 1.0e-3;
    }
    if first_motion_per_step == f32::MAX {
        first_motion_per_step = 0.0;
    }

    Ok(TorusRobustnessReport {
        passed,
        target_opacity_delta: if field_mode {
            UV_TORUS_FIELD_OPACITY_GAIN
        } else {
            UV_TORUS_OPACITY_GROWTH_DELTA
        },
        trained_opacity_delta,
        target_motion_gain: UV_TORUS_MOTION_GAIN,
        target_residual_decay: UV_TORUS_RESIDUAL_DECAY,
        max_target_position_error,
        mean_target_position_error: sum_mean_target_position_error / cases.len().max(1) as f32,
        max_torus_surface_error,
        max_color_target_error,
        first_motion_per_step,
        max_motion_per_step,
        max_opacity_target_error,
        min_final_opacity,
        max_final_opacity,
        cases: case_reports,
    })
}

fn torus_robustness_case_report(
    trace: &burn_automata::RolloutTrace,
    case: TorusRobustnessCaseConfig,
) -> TorusRobustnessCaseReport {
    let major = case.seed_scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let field_mode = case.seed_mode == ParticleSeed::TorusFieldDense3d;
    let morphogen_mode = case.seed_mode == ParticleSeed::TorusMorphogenDense3d;
    let target_mesh = if field_mode || morphogen_mode {
        Some(uv_torus_mesh_target(major))
    } else {
        None
    };
    let expected_final_opacity_logit = if field_mode {
        UV_TORUS_FIELD_OPACITY_TARGET
    } else {
        UV_TORUS_INITIAL_OPACITY_LOGIT + UV_TORUS_OPACITY_GROWTH_DELTA * case.steps as f32
    };
    let (initial_positions, _) = seed_particles_scaled(
        trace.batch_size,
        case.particle_count,
        trace.state_dims,
        3,
        case.seed,
        case.seed_mode,
        major,
    );
    let mut max_initial_target_position_error = 0.0_f32;
    let mut sum_initial_target_position_error = 0.0_f32;
    let mut max_target_position_error = 0.0_f32;
    let mut sum_target_position_error = 0.0_f32;
    let mut max_torus_surface_error = 0.0_f32;
    let mut sum_torus_surface_error = 0.0_f32;
    let mut min_final_radial = f32::MAX;
    let mut max_final_radial = f32::MIN;
    let mut max_final_abs_z = 0.0_f32;
    let mut max_color_target_error = 0.0_f32;
    let mut sum_color_target_error = 0.0_f32;
    let mut min_final_opacity_logit = f32::MAX;
    let mut max_final_opacity_logit = f32::MIN;
    let mut max_opacity_target_error = 0.0_f32;
    let mut finite = true;

    for (idx, position) in trace.positions.iter().enumerate() {
        finite &= position.iter().all(|value| value.is_finite());
        let initial_position = initial_positions[idx];
        let indexed_target =
            uv_torus_sample(idx % case.particle_count.max(1), case.particle_count, major).position;
        let initial_target = if field_mode || morphogen_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([
                    initial_position[0],
                    initial_position[1],
                    initial_position[2],
                ])
                .closest
        } else {
            indexed_target
        };
        let target = if field_mode {
            target_mesh
                .as_ref()
                .unwrap()
                .project([position[0], position[1], position[2]])
                .closest
        } else if morphogen_mode {
            initial_target
        } else {
            indexed_target
        };
        let initial_target_position_error = ((initial_position[0] - target[0]).powi(2)
            + (initial_position[1] - target[1]).powi(2)
            + (initial_position[2] - target[2]).powi(2))
        .sqrt();
        let initial_target_position_error = if field_mode || morphogen_mode {
            ((initial_position[0] - initial_target[0]).powi(2)
                + (initial_position[1] - initial_target[1]).powi(2)
                + (initial_position[2] - initial_target[2]).powi(2))
            .sqrt()
        } else {
            initial_target_position_error
        };
        max_initial_target_position_error =
            max_initial_target_position_error.max(initial_target_position_error);
        sum_initial_target_position_error += initial_target_position_error;

        let target_position_error = ((position[0] - target[0]).powi(2)
            + (position[1] - target[1]).powi(2)
            + (position[2] - target[2]).powi(2))
        .sqrt();
        max_target_position_error = max_target_position_error.max(target_position_error);
        sum_target_position_error += target_position_error;

        let torus_surface_error =
            uv_torus_surface_error([position[0], position[1], position[2]], major);
        max_torus_surface_error = max_torus_surface_error.max(torus_surface_error);
        sum_torus_surface_error += torus_surface_error;
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        min_final_radial = min_final_radial.min(radial);
        max_final_radial = max_final_radial.max(radial);
        max_final_abs_z = max_final_abs_z.max(position[2].abs());

        let state_base = idx * trace.state_dims;
        if trace.state_dims >= 6 {
            let tail = trace.state_dims - 3;
            let rgb = uv_torus_tail_state_to_rgb([
                trace.states[state_base + tail],
                trace.states[state_base + tail + 1],
                trace.states[state_base + tail + 2],
            ]);
            let expected_rgb = uv_torus_position_color(target, major);
            let color_target_error = ((rgb[0] - expected_rgb[0]).powi(2)
                + (rgb[1] - expected_rgb[1]).powi(2)
                + (rgb[2] - expected_rgb[2]).powi(2))
            .sqrt();
            max_color_target_error = max_color_target_error.max(color_target_error);
            sum_color_target_error += color_target_error;
        }

        let opacity = trace.states[state_base + 3];
        finite &= opacity.is_finite();
        min_final_opacity_logit = min_final_opacity_logit.min(opacity);
        max_final_opacity_logit = max_final_opacity_logit.max(opacity);
        max_opacity_target_error =
            max_opacity_target_error.max((opacity - expected_final_opacity_logit).abs());
    }
    finite &= trace.states.iter().all(|value| value.is_finite());
    finite &= trace.mean_dx.iter().all(|value| value.is_finite());

    TorusRobustnessCaseReport {
        particle_count: case.particle_count,
        steps: case.steps,
        seed: case.seed,
        seed_scale: case.seed_scale,
        seed_mode: case.seed_mode,
        torus_inner_radius: major - minor,
        torus_outer_radius: major + minor,
        max_initial_target_position_error,
        mean_initial_target_position_error: sum_initial_target_position_error
            / trace.positions.len().max(1) as f32,
        max_target_position_error,
        mean_target_position_error: sum_target_position_error / trace.positions.len().max(1) as f32,
        max_torus_surface_error,
        mean_torus_surface_error: sum_torus_surface_error / trace.positions.len().max(1) as f32,
        min_final_radial,
        max_final_radial,
        max_final_abs_z,
        max_color_target_error,
        mean_color_target_error: sum_color_target_error / trace.positions.len().max(1) as f32,
        first_motion_per_step: trace.mean_dx.first().copied().unwrap_or_default(),
        max_motion_per_step: trace.mean_dx.iter().copied().fold(0.0, f32::max),
        expected_final_opacity_logit,
        min_final_opacity_logit,
        max_final_opacity_logit,
        max_opacity_target_error,
        finite,
    }
}

#[cfg(feature = "gpu_wgpu")]
fn gpu_rollout_trace(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    neighbor_mode: burn_automata::gpu::WgpuNeighborMode,
) -> Result<burn_automata::RolloutTrace, Box<dyn std::error::Error>> {
    if cfg.batch_size != 1 {
        return Err(std::io::Error::other("infer --gpu currently supports batch_size=1").into());
    }
    let (mut positions, mut states) = seed_particles_scaled(
        cfg.batch_size,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let executor = burn_automata::gpu::WgpuAutomataExecutor::new_blocking()?;
    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        cfg.batch_size,
        cfg.particle_count,
        grid,
        cfg.dt,
        neighbor_mode,
        cfg.update_prob,
        cfg.seed,
    )?;
    let mut mean_dx = Vec::with_capacity(cfg.steps);
    for _ in 0..cfg.steps {
        let before = positions.clone();
        executor.step_state(&mut state)?;
        let output = executor.read_state(&state)?;
        let dx_norm = output
            .next_positions
            .iter()
            .zip(before.iter())
            .map(|(next, prev)| {
                let mut norm = 0.0;
                for axis in 0..model.config.spatial_dims {
                    let diff = next[axis] - prev[axis];
                    norm += diff * diff;
                }
                norm.sqrt()
            })
            .sum::<f32>()
            / output.next_positions.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = output.next_positions;
        states = output.next_states;
    }
    Ok(burn_automata::RolloutTrace {
        positions,
        states,
        batch_size: cfg.batch_size,
        particle_count: cfg.particle_count,
        state_dims: model.config.state_dims,
        steps: cfg.steps,
        mean_dx,
    })
}

#[derive(Clone, Copy, Debug)]
struct ProfileReport {
    perceive_ms: f64,
    forward_ms: f64,
    integrate_ms: f64,
    final_mean_dx: f32,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
struct GpuBenchReport {
    gpu_step_ms: f64,
    final_mean_dx: f32,
    final_mean_density: f32,
    initial_nonempty_cells: usize,
    initial_max_cell_occupancy: usize,
    neighbor_mode: burn_automata::gpu::WgpuNeighborMode,
    bucket_capacity: usize,
    grid_storage_len: usize,
    grid_clear_len: usize,
    grid_overflow_count: u32,
    gaussian_write: bool,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
struct GpuBenchConfig {
    particles: usize,
    steps: usize,
    seed_scale: f32,
    update_prob: f32,
    seed_mode: ParticleSeed,
    geometry: BenchGeometryArg,
    neighbor_mode: burn_automata::gpu::WgpuNeighborMode,
    gaussian_write: bool,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
struct GpuBenchSummary {
    repeats: usize,
    median_report: GpuBenchReport,
    min_avg_step_ms: f64,
    median_avg_step_ms: f64,
    max_avg_step_ms: f64,
}

#[cfg(feature = "gpu_wgpu")]
fn summarize_gpu_reports(reports: &[GpuBenchReport], steps: usize) -> GpuBenchSummary {
    let steps = steps.max(1) as f64;
    let mut sorted = reports.to_vec();
    sorted.sort_by(|lhs, rhs| {
        let lhs_step = lhs.gpu_step_ms / steps;
        let rhs_step = rhs.gpu_step_ms / steps;
        lhs_step.total_cmp(&rhs_step)
    });
    let median_index = sorted.len() / 2;
    GpuBenchSummary {
        repeats: reports.len(),
        median_report: sorted[median_index],
        min_avg_step_ms: sorted
            .first()
            .map(|report| report.gpu_step_ms / steps)
            .unwrap_or(0.0),
        median_avg_step_ms: sorted[median_index].gpu_step_ms / steps,
        max_avg_step_ms: sorted
            .last()
            .map(|report| report.gpu_step_ms / steps)
            .unwrap_or(0.0),
    }
}

#[cfg(feature = "gpu_wgpu")]
fn gpu_rollout_bench(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: GpuBenchConfig,
) -> Result<GpuBenchReport, Box<dyn std::error::Error>> {
    let (positions, states) = bench_particles(
        model,
        grid,
        cfg.particles,
        cfg.seed_scale,
        cfg.seed_mode,
        cfg.geometry,
        42,
    );
    let initial_grid = build_hashgrid(&positions, 1, cfg.particles, grid)?;
    let (initial_nonempty_cells, initial_max_cell_occupancy) =
        hashgrid_occupancy_stats(&initial_grid.bin_offsets);
    let mut report = GpuBenchReport {
        gpu_step_ms: 0.0,
        final_mean_dx: 0.0,
        final_mean_density: 0.0,
        initial_nonempty_cells,
        initial_max_cell_occupancy,
        neighbor_mode: burn_automata::gpu::WgpuNeighborMode::Auto,
        bucket_capacity: 0,
        grid_storage_len: 0,
        grid_clear_len: 0,
        grid_overflow_count: 0,
        gaussian_write: cfg.gaussian_write,
    };
    let executor = burn_automata::gpu::WgpuAutomataExecutor::new_blocking()?;
    let mut warmup_state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        1,
        cfg.particles,
        grid,
        1.0,
        cfg.neighbor_mode,
        cfg.update_prob,
        42,
    )?;
    executor.step_state(&mut warmup_state)?;
    executor.wait_idle()?;

    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        1,
        cfg.particles,
        grid,
        1.0,
        cfg.neighbor_mode,
        cfg.update_prob,
        42,
    )?;
    let neighbor = executor.neighbor_report(&state);
    let gaussian_buffers = if cfg.gaussian_write {
        Some(executor.create_gaussian_buffers(cfg.particles)?)
    } else {
        None
    };
    let gaussian_bind_group = gaussian_buffers
        .as_ref()
        .map(|buffers| executor.create_gaussian_bind_group(&buffers.refs(), cfg.particles))
        .transpose()?;
    report.neighbor_mode = neighbor.mode;
    report.bucket_capacity = neighbor.bucket_capacity;
    report.grid_storage_len = neighbor.grid_storage_len;
    report.grid_clear_len = neighbor.grid_clear_len;
    let started = Instant::now();
    for _ in 0..cfg.steps {
        if let Some(bind_group) = gaussian_bind_group.as_ref() {
            executor.step_state_into_gaussian_bind_group(&mut state, bind_group)?;
        } else {
            executor.step_state(&mut state)?;
        }
    }
    executor.wait_idle()?;
    report.gpu_step_ms = started.elapsed().as_secs_f64() * 1000.0;
    report.grid_overflow_count = executor.read_grid_overflow(&state)?;
    let output = executor.read_state(&state)?;
    report.final_mean_dx = output
        .next_positions
        .iter()
        .zip(positions.iter())
        .map(|(next, prev)| {
            let mut norm = 0.0;
            for axis in 0..model.config.spatial_dims {
                let diff = next[axis] - prev[axis];
                norm += diff * diff;
            }
            norm.sqrt()
        })
        .sum::<f32>()
        / output.next_positions.len().max(1) as f32
        / cfg.steps.max(1) as f32;
    report.final_mean_density =
        output.density.iter().copied().sum::<f32>() / output.density.len().max(1) as f32;
    Ok(report)
}

#[cfg(feature = "gpu_wgpu")]
fn wgpu_neighbor_mode(
    mode: NeighborModeArg,
    bucket_capacity: Option<usize>,
) -> burn_automata::gpu::WgpuNeighborMode {
    match mode {
        NeighborModeArg::LinkedList => burn_automata::gpu::WgpuNeighborMode::LinkedList,
        NeighborModeArg::Auto if bucket_capacity.is_none() => {
            burn_automata::gpu::WgpuNeighborMode::Auto
        }
        NeighborModeArg::Auto | NeighborModeArg::FixedBuckets => {
            if let Some(capacity) = bucket_capacity {
                burn_automata::gpu::WgpuNeighborMode::FixedCellBuckets { capacity }
            } else {
                burn_automata::gpu::WgpuNeighborMode::Auto
            }
        }
        NeighborModeArg::TiledFixedBuckets => {
            burn_automata::gpu::WgpuNeighborMode::TiledFixedCellBuckets {
                capacity: bucket_capacity.unwrap_or(256),
            }
        }
        NeighborModeArg::SortedCells => burn_automata::gpu::WgpuNeighborMode::SortedCells,
        NeighborModeArg::Bvh => burn_automata::gpu::WgpuNeighborMode::Bvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuBvh => burn_automata::gpu::WgpuNeighborMode::GpuBvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuLbvh => burn_automata::gpu::WgpuNeighborMode::GpuLbvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuMortonLbvh => burn_automata::gpu::WgpuNeighborMode::GpuMortonLbvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
    }
}

fn spatial_strategies(
    requested: SpatialStrategyArg,
    grid: &burn_automata::kernels::HashGridConfig,
    tile_size: [usize; 3],
    bvh_leaf_size: usize,
) -> Vec<burn_automata::kernels::SpatialStrategyKind> {
    use burn_automata::kernels::{Boundary, HashGridMode, SpatialStrategyKind};
    match requested {
        SpatialStrategyArg::HashGrid => vec![SpatialStrategyKind::HashGrid],
        SpatialStrategyArg::TileBlocks => vec![SpatialStrategyKind::TileBlocks { tile_size }],
        SpatialStrategyArg::Bvh => vec![SpatialStrategyKind::Bvh {
            leaf_size: bvh_leaf_size,
        }],
        SpatialStrategyArg::All => {
            let mut strategies = vec![SpatialStrategyKind::HashGrid];
            if grid.boundary != Boundary::Periodic {
                strategies.push(SpatialStrategyKind::Bvh {
                    leaf_size: bvh_leaf_size,
                });
            }
            if grid.mode != HashGridMode::Particle {
                strategies.push(SpatialStrategyKind::TileBlocks { tile_size });
            }
            strategies
        }
    }
}

fn parse_tile_size(raw: &str) -> Result<[usize; 3], Box<dyn std::error::Error>> {
    let values = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(values.len() == 2 || values.len() == 3) {
        return Err(std::io::Error::other(
            "--tile-size expects two or three comma-separated integers",
        )
        .into());
    }
    if values.iter().any(|value| *value == 0) {
        return Err(std::io::Error::other("--tile-size values must be non-zero").into());
    }
    Ok([values[0], values[1], values.get(2).copied().unwrap_or(1)])
}

fn strategy_label(strategy: burn_automata::kernels::SpatialStrategyKind) -> &'static str {
    strategy.label()
}

fn bench_particles(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    particles: usize,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    geometry: BenchGeometryArg,
    seed: u64,
) -> (Vec<[f32; 4]>, Vec<f32>) {
    let (mut positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        seed,
        seed_mode,
        seed_scale,
    );
    apply_bench_geometry(
        &mut positions,
        model.config.spatial_dims,
        particles,
        seed_scale,
        grid,
        geometry,
        seed ^ 0x9e37_79b9,
    );
    (positions, states)
}

fn apply_bench_geometry(
    positions: &mut [[f32; 4]],
    spatial_dims: usize,
    particles: usize,
    scale: f32,
    grid: &burn_automata::kernels::HashGridConfig,
    geometry: BenchGeometryArg,
    seed: u64,
) {
    if matches!(geometry, BenchGeometryArg::Seed) {
        return;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    for (idx, position) in positions.iter_mut().enumerate() {
        let local_idx = idx % particles.max(1);
        *position = match geometry {
            BenchGeometryArg::Seed => *position,
            BenchGeometryArg::Dense | BenchGeometryArg::ShiftedDense => {
                dense_ball_position(&mut rng, spatial_dims, scale)
            }
            BenchGeometryArg::Uniform | BenchGeometryArg::ShiftedUniform => {
                uniform_box_position(&mut rng, spatial_dims, scale)
            }
            BenchGeometryArg::Line => line_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Ring => ring_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Plane => plane_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Shell => shell_position(&mut rng, spatial_dims, scale),
            BenchGeometryArg::Torus => torus_position(local_idx, particles, spatial_dims, scale),
        };
        if matches!(
            geometry,
            BenchGeometryArg::ShiftedDense | BenchGeometryArg::ShiftedUniform
        ) {
            shift_outside_fixed_grid(position, spatial_dims, grid, scale);
        }
    }
}

fn dense_ball_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let r = rng.random::<f32>().sqrt() * scale;
        [r * theta.cos(), r * theta.sin(), 0.0, 0.0]
    } else {
        let dir = sphere_direction(rng);
        let r = rng.random::<f32>().cbrt() * scale;
        [dir[0] * r, dir[1] * r, dir[2] * r, 0.0]
    }
}

fn uniform_box_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    let mut position = [0.0; 4];
    for value in position.iter_mut().take(spatial_dims) {
        *value = rng.random_range(-scale..scale);
    }
    position
}

fn line_position(rng: &mut StdRng, spatial_dims: usize, scale: f32, eps: f32) -> [f32; 4] {
    let mut position = [0.0; 4];
    position[0] = rng.random_range(-scale..scale);
    for value in position.iter_mut().take(spatial_dims).skip(1) {
        *value = rng.random_range(-0.125 * eps..0.125 * eps);
    }
    position
}

fn ring_position(rng: &mut StdRng, spatial_dims: usize, scale: f32, eps: f32) -> [f32; 4] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let r = scale + rng.random_range(-0.25 * eps..0.25 * eps);
    let mut position = [r * theta.cos(), r * theta.sin(), 0.0, 0.0];
    if spatial_dims == 3 {
        position[2] = rng.random_range(-0.25 * eps..0.25 * eps);
    }
    position
}

fn plane_position(rng: &mut StdRng, spatial_dims: usize, scale: f32, eps: f32) -> [f32; 4] {
    let mut position = [0.0; 4];
    position[0] = rng.random_range(-scale..scale);
    position[1] = rng.random_range(-scale..scale);
    if spatial_dims == 3 {
        position[2] = rng.random_range(-0.125 * eps..0.125 * eps);
    }
    position
}

fn shell_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0]
    } else {
        let dir = sphere_direction(rng);
        [dir[0] * scale, dir[1] * scale, dir[2] * scale, 0.0]
    }
}

fn torus_position(local_idx: usize, particles: usize, spatial_dims: usize, scale: f32) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = std::f32::consts::TAU * local_idx as f32 / particles.max(1) as f32;
        return [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0];
    }
    let sample = uv_torus_sample(local_idx, particles, scale);
    [
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ]
}

fn sphere_direction(rng: &mut StdRng) -> [f32; 3] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let z = rng.random_range(-1.0_f32..1.0_f32);
    let r_xy = (1.0_f32 - z * z).sqrt();
    [r_xy * theta.cos(), r_xy * theta.sin(), z]
}

fn shift_outside_fixed_grid(
    position: &mut [f32; 4],
    spatial_dims: usize,
    grid: &burn_automata::kernels::HashGridConfig,
    scale: f32,
) {
    for (axis, value) in position.iter_mut().enumerate().take(spatial_dims) {
        let extent = grid.eps * grid.grid_size[axis] as f32;
        let sign = if axis == 1 { -1.0 } else { 1.0 };
        *value += sign * (extent + scale.max(grid.eps));
    }
}

#[cfg(feature = "gpu_wgpu")]
fn hashgrid_occupancy_stats(bin_offsets: &[usize]) -> (usize, usize) {
    bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold((0usize, 0usize), |(nonempty, max), count| {
            (nonempty + usize::from(count > 0), max.max(count))
        })
}

#[derive(Clone, Copy, Debug)]
struct CpuProfileConfig {
    particles: usize,
    steps: usize,
    seed_scale: f32,
    update_prob: f32,
    seed_mode: ParticleSeed,
    geometry: BenchGeometryArg,
}

fn profile_rollout(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: CpuProfileConfig,
) -> Result<ProfileReport, Box<dyn std::error::Error>> {
    let (mut positions, mut states) = bench_particles(
        model,
        grid,
        cfg.particles,
        cfg.seed_scale,
        cfg.seed_mode,
        cfg.geometry,
        42,
    );
    let mut report = ProfileReport {
        perceive_ms: 0.0,
        forward_ms: 0.0,
        integrate_ms: 0.0,
        final_mean_dx: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(42 ^ 0x5eed);
    for _ in 0..cfg.steps {
        let mask = stochastic_mask(cfg.particles, cfg.update_prob, &mut rng);
        let started = Instant::now();
        let perception = perceive_with_options(
            &positions,
            &states,
            1,
            cfg.particles,
            model.config.state_dims,
            grid,
            PerceptionOptions {
                state_grad: model.config.state_grad,
                density_grad: model.config.density_grad,
                eps0: model.config.eps0,
                scale_equivariance: model.config.scale_equivariant(),
                particle_density_equivariance: model.config.particle_density_equivariant(),
                log_norm_grad: model.config.log_norm_grad,
                log_norm_density_grad: model.config.log_norm_density_grad,
                hybrid_state_gradient: true,
                position_features: model.config.position_features,
            },
        )?;
        report.perceive_ms += started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let (dx, ds) = model.forward_from_features_with_eps(&perception.features, grid.eps)?;
        report.forward_ms += started.elapsed().as_secs_f64() * 1000.0;

        report.final_mean_dx = dx
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / dx.len().max(1) as f32;
        let started = Instant::now();
        (positions, states) = euler_step(
            &positions,
            &states,
            &dx,
            &ds,
            1,
            cfg.particles,
            model.config.state_dims,
            grid,
            1.0,
            Some(&mask),
        )?;
        report.integrate_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(report)
}

fn stochastic_mask(count: usize, update_prob: f32, rng: &mut StdRng) -> Vec<f32> {
    if update_prob >= 1.0 {
        return vec![1.0; count];
    }
    if update_prob <= 0.0 {
        return vec![0.0; count];
    }
    (0..count)
        .map(|_| f32::from(rng.random::<f32>() < update_prob))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_defaults_to_seeded_growth_target() {
        let seed = default_train_target_seed(AutomataPreset::Growing3dGs, None, false);

        assert_eq!(seed, Some(DEFAULT_GROWTH_TARGET_SEED));
        assert_eq!(
            train_target_source(AutomataPreset::Growing3dGs, seed, false),
            "seeded:Growing3dGs:42"
        );
    }

    #[test]
    fn train_source_defaults_to_rollout_local_metadata() {
        let seed = default_train_target_seed(AutomataPreset::Growing2d, None, false);
        let target_source = train_target_source(AutomataPreset::Growing2d, seed, false);

        assert_eq!(
            training_source_with_batch(TrainingBatchArg::Rollout, &target_source),
            "rollout-local:seeded:Growing2d:42"
        );
        assert_eq!(
            training_source_with_batch(TrainingBatchArg::Features, &target_source),
            "feature-rows:seeded:Growing2d:42"
        );
    }

    #[test]
    fn train_zero_update_requires_explicit_flag() {
        let seed = default_train_target_seed(AutomataPreset::Growing2d, None, true);

        assert_eq!(seed, None);
        assert_eq!(
            train_target_source(AutomataPreset::Growing2d, seed, true),
            "explicit-zero-update"
        );
    }

    #[test]
    fn mesh_training_sources_separate_rollout_local_from_projection_baseline() {
        assert!(UV_TORUS_POSITION_FIELD_TARGET_SOURCE.contains("position-field"));
        assert!(TEAPOT_POSITION_FIELD_TARGET_SOURCE.contains("position-field"));
        assert!(UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE.contains("rollout-position-field"));
        assert!(TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE.contains("rollout-position-field"));
        assert!(UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE.contains("rollout-local"));
        assert!(TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE.contains("rollout-local"));
        assert!(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("conditionless-local"));
        assert!(TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("conditionless-local"));
        assert!(UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE.contains("random-ball"));
        assert!(TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE.contains("random-ball"));
        assert!(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("substrate"));
        assert!(TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("substrate"));
        assert_eq!(
            mesh_conditionless_local_target_source_for_seed(
                MeshTargetArg::Torus,
                ParticleSeed::TorusGrowth3d,
            ),
            UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE
        );
        assert_eq!(
            mesh_conditionless_local_target_source_for_seed(
                MeshTargetArg::Torus,
                ParticleSeed::TorusSubstrateGrowth3d,
            ),
            UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE
        );
        assert!(UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE.contains("seed-frame"));
        assert!(TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE.contains("seed-frame"));
    }

    #[test]
    fn render_training_source_preserves_local_refinement_lineage() {
        let local_source = render_training_source(
            MeshTargetArg::Torus,
            Some(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE),
            ParticleSeed::TorusGrowth3d,
        );
        assert!(local_source.starts_with("render-refined-rust:"));
        assert!(local_source.contains("conditionless-local"));
        assert!(!local_source.contains("position-field"));
        assert!(!local_source.contains("render-proxy-rust"));

        let already_refined_source = render_training_source(
            MeshTargetArg::Torus,
            Some(&local_source),
            ParticleSeed::TorusGrowth3d,
        );
        assert_eq!(already_refined_source, local_source);

        let field_source = render_training_source(
            MeshTargetArg::Torus,
            Some(UV_TORUS_POSITION_FIELD_TARGET_SOURCE),
            ParticleSeed::TorusFieldDense3d,
        );
        assert!(field_source.starts_with("render-proxy-rust:"));
        assert!(field_source.contains("position-field"));

        let default_source = render_training_source(
            MeshTargetArg::Teapot,
            None,
            ParticleSeed::TeapotFieldDense3d,
        );
        assert!(default_source.contains("field-baseline"));
    }

    #[test]
    fn render_training_defaults_match_model_family() {
        assert_eq!(
            render_training_default_seed_mode(MeshTargetArg::Torus),
            ParticleSeed::TorusGrowth3d
        );
        assert_eq!(
            render_training_default_seed_mode(MeshTargetArg::Teapot),
            ParticleSeed::TeapotGrowth3d
        );

        let local_model = NpaModel::seeded(NpaConfig::growing_3dgs(), 7);
        assert_eq!(
            default_render_training_seed_mode(MeshTargetArg::Torus, &local_model),
            ParticleSeed::TorusSubstrateGrowth3d
        );
        assert_eq!(
            default_render_training_seed_mode(MeshTargetArg::Teapot, &local_model),
            ParticleSeed::TeapotSubstrateGrowth3d
        );

        let field_model = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 7);
        assert_eq!(
            default_render_training_seed_mode(MeshTargetArg::Torus, &field_model),
            ParticleSeed::TorusFieldDense3d
        );
        assert_eq!(
            default_render_training_seed_mode(MeshTargetArg::Teapot, &field_model),
            ParticleSeed::TeapotFieldDense3d
        );
    }

    #[test]
    fn render_training_base_defaults_to_conditionless_local_growth() {
        let target = uv_torus_mesh_target(UV_TORUS_FIELD_SCALE);
        let (model, source) = render_training_base_model(
            MeshTargetArg::Torus,
            &target,
            render_training_default_seed_mode(MeshTargetArg::Torus),
        )
        .unwrap();

        assert!(!model.config.position_features);
        assert!(local_conditionless_lineage(&source));
        assert!(source.starts_with("ablation-rust:"));
        assert!(source.contains("conditionless-local"));
        assert!(source.contains("random-ball"));
        assert!(!source.contains("position-field"));
        assert!(!source.contains("render-proxy-rust"));

        let err = render_training_base_model(
            MeshTargetArg::Torus,
            &target,
            ParticleSeed::TorusFieldDense3d,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("target local growth seed"));
    }

    #[test]
    fn sparse_growth_seed_modes_do_not_preload_target_state() {
        let config = NpaConfig::growing_3dgs();
        for seed_mode in [ParticleSeed::TorusGrowth3d, ParticleSeed::TeapotGrowth3d] {
            let (_positions, states) = seed_particles_scaled(
                1,
                512,
                config.state_dims,
                config.spatial_dims,
                0x5eed,
                seed_mode,
                UV_TORUS_FIELD_SCALE,
            );
            let mut active = 0usize;
            let mut inactive = 0usize;
            for state in states.chunks_exact(config.state_dims) {
                if state[3] > -1.0 {
                    active += 1;
                } else {
                    inactive += 1;
                }
            }
            let non_opacity_seed_abs_max =
                growth_3d_non_scaffold_seed_abs_max(config.state_dims, seed_mode, &states);

            assert!(active > 0, "{seed_mode:?} should seed a sparse active core");
            assert!(
                inactive > active,
                "{seed_mode:?} should leave most particles dormant"
            );
            assert_eq!(
                non_opacity_seed_abs_max, 0.0,
                "{seed_mode:?} must not preload residual, normal, color, or other target state outside the coordinate scaffold"
            );
        }
    }

    #[test]
    fn catalog_bound_render_training_requires_local_growth_lineage() {
        assert!(is_catalog_model_output_path(Path::new(
            "assets/models/teapot_growth_3d.bpk"
        )));
        assert!(!is_catalog_model_output_path(Path::new(
            "artifacts/render_trained_3d.bpk"
        )));

        validate_catalog_bound_render_training_output(
            Path::new("artifacts/render_trained_3d.bpk"),
            MeshTargetArg::Teapot,
            ParticleSeed::TeapotFieldDense3d,
            None,
        )
        .unwrap();

        let local_source =
            format!("render-refined-rust:{TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE}");
        validate_catalog_bound_render_training_output(
            Path::new("assets/models/teapot_growth_3d.bpk"),
            MeshTargetArg::Teapot,
            ParticleSeed::TeapotGrowth3d,
            Some(&local_source),
        )
        .unwrap();

        let field_seed_error = validate_catalog_bound_render_training_output(
            Path::new("assets/models/render_trained_3d.bpk"),
            MeshTargetArg::Teapot,
            ParticleSeed::TeapotFieldDense3d,
            Some(&local_source),
        )
        .unwrap_err();
        assert!(field_seed_error.to_string().contains("local growth seed"));

        let shortcut_lineage_error = validate_catalog_bound_render_training_output(
            Path::new("assets/models/render_trained_3d.bpk"),
            MeshTargetArg::Teapot,
            ParticleSeed::TeapotGrowth3d,
            Some(TEAPOT_POSITION_FIELD_TARGET_SOURCE),
        )
        .unwrap_err();
        assert!(
            shortcut_lineage_error
                .to_string()
                .contains("conditionless-local")
        );
    }

    #[test]
    fn catalog_bound_render_training_uses_target_temp_candidate_path() {
        let torus = catalog_bound_candidate_path(MeshTargetArg::Torus, 1234);
        let teapot = catalog_bound_candidate_path(MeshTargetArg::Teapot, 1234);

        assert!(torus.starts_with("target"));
        assert!(teapot.starts_with("target"));
        assert!(!is_catalog_model_output_path(&torus));
        assert!(!is_catalog_model_output_path(&teapot));
        assert!(torus.to_string_lossy().contains("torus"));
        assert!(teapot.to_string_lossy().contains("teapot"));
        assert_ne!(torus, teapot);
    }

    #[test]
    fn diagnostic_3d_outputs_refuse_catalog_paths() {
        validate_diagnostic_3d_output_not_catalog(
            Path::new("target/teapot_probe.bpk"),
            "ablate-local-3d",
        )
        .unwrap();
        validate_diagnostic_3d_output_not_catalog(
            Path::new("artifacts/teapot_probe.bpk"),
            "ablate-local-3d",
        )
        .unwrap();

        let err = validate_diagnostic_3d_output_not_catalog(
            Path::new("assets/models/teapot_probe.bpk"),
            "ablate-local-3d",
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("diagnostic 3D artifacts"));
        assert!(message.contains("validate_3d_catalog.py"));
    }

    #[test]
    fn local_3d_continuation_accepts_only_conditionless_local_lineage() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 17);
        let local_path = bin_temp_path("local_3d_continuation_ok.bpk");
        let local_manifest = BpkModelManifest::from_model(
            &model,
            grid.clone(),
            Some(format!(
                "ablation-rust:{UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
            )),
        );
        burn_automata::import::save_manifest(&local_path, &local_manifest).unwrap();

        let (_loaded, _grid, source) = load_conditionless_local_base_model(
            &local_path,
            TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
        )
        .unwrap();
        std::fs::remove_file(&local_path).ok();
        assert!(source.contains("continued-from="));
        assert!(source.contains("conditionless-local"));
        assert!(!source.contains("position-field"));
        assert!(!source.contains("seed-frame"));
        assert!(!source.contains("render-proxy-rust"));

        let shortcut_path = bin_temp_path("local_3d_continuation_shortcut.bpk");
        let shortcut_manifest = BpkModelManifest::from_model(
            &model,
            grid.clone(),
            Some(format!(
                "ablation-rust:{UV_TORUS_POSITION_FIELD_TARGET_SOURCE}"
            )),
        );
        burn_automata::import::save_manifest(&shortcut_path, &shortcut_manifest).unwrap();
        let shortcut_err = load_conditionless_local_base_model(
            &shortcut_path,
            UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE,
        )
        .unwrap_err();
        std::fs::remove_file(&shortcut_path).ok();
        assert!(shortcut_err.to_string().contains("shortcut lineage"));

        let mut position_config = config;
        position_config.position_features = true;
        let position_model = NpaModel::seeded(position_config, 19);
        let position_path = bin_temp_path("local_3d_continuation_position_features.bpk");
        let position_manifest = BpkModelManifest::from_model(
            &position_model,
            grid,
            Some(format!(
                "ablation-rust:{TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
            )),
        );
        burn_automata::import::save_manifest(&position_path, &position_manifest).unwrap();
        let position_err = load_conditionless_local_base_model(
            &position_path,
            TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
        )
        .unwrap_err();
        std::fs::remove_file(&position_path).ok();
        assert!(position_err.to_string().contains("position-feature"));
    }

    #[test]
    fn growth_3d_validation_rejects_static_local_artifact() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let path = bin_temp_path("static_local_growth3d.bpk");
        let manifest = BpkModelManifest::from_model(
            &model,
            grid,
            Some(format!(
                "ablation-rust:{UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
            )),
        );
        burn_automata::import::save_manifest(&path, &manifest).unwrap();

        let mut validation_cfg = growth_validation_test_config(ParticleSeed::TorusGrowth3d);
        validation_cfg.extra_seeds = vec![43, 42, 44];
        let report =
            growth_3d_validation_report(&path, MeshTargetArg::Torus, validation_cfg).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(report.local_conditionless_lineage);
        assert!(matches!(report.gate, Growth3dValidationGateArg::Strict));
        assert!(!report.gate_passed);
        assert!(!report.strict_passed);
        assert_eq!(
            report.activation.final_active_count,
            report.activation.active_seed_count
        );
        assert_eq!(report.activation.newly_activated_count, 0);
        assert_eq!(report.max_motion_per_step, 0.0);
        assert!(report.final_opacity.finite);
        assert_eq!(
            report.final_opacity.max_allowed,
            GROWTH_3D_MAX_FINAL_OPACITY_LOGIT
        );
        assert!(report.final_opacity.max <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
        assert!(report.strict_checks.bounded_final_opacity);
        assert_eq!(report.robustness.seed_count, 3);
        assert_eq!(
            report
                .robustness
                .seeds
                .iter()
                .map(|seed_report| seed_report.seed)
                .collect::<Vec<_>>(),
            vec![42, 43, 44]
        );
        assert!(!report.robustness.all_gate_passed);
        assert!(!report.robustness.all_temporal_activation_progressive);
        assert_eq!(report.robustness.min_newly_activated_fraction, 0.0);
        assert_eq!(report.robustness.min_active_growth_ratio, 1.0);
        assert_eq!(
            report.robustness.min_active_seed_count,
            report.robustness.min_final_active_count
        );
        assert!(report.robustness.all_bounded_final_opacity);
        assert!(!report.robustness.all_color_state_emerged);
        assert!(report.robustness.all_permutation_consistent);
        assert!(!report.seed_perturbation.passed);
        assert!(!report.robustness.all_seed_perturbation_stable);
        assert_eq!(
            report.robustness.min_perturbed_newly_activated_fraction,
            0.0
        );
        assert_eq!(report.robustness.min_perturbed_active_count_ratio, 1.0);
        assert_eq!(report.robustness.max_perturbed_active_count_ratio, 1.0);
        assert!(report.robustness.max_final_opacity <= GROWTH_3D_MAX_FINAL_OPACITY_LOGIT);
        assert_eq!(report.robustness.min_final_active_color_state_mean_abs, 0.0);
        assert_eq!(
            report.robustness.min_final_active_color_state_stddev_mean,
            0.0
        );
        assert!(report.robustness.max_permutation_position_error <= 1.0e-6);
        assert!(report.robustness.worst_strict_score.is_finite());
        assert!(!growth_3d_fail_on_validation_passed(&report));
    }

    #[test]
    fn growth_3d_catalog_sanity_thresholds_match_active_catalog_floor() {
        for (target, max_total_loss, min_density, min_color, min_depth) in [
            (MeshTargetArg::Torus, 0.90, 0.95, 16.0, 14.8),
            (MeshTargetArg::Teapot, 0.85, 0.95, 18.0, 18.0),
        ] {
            let exact = synthetic_render_loss(max_total_loss, min_density, min_color, min_depth);
            let exact_report = growth_3d_catalog_sanity_report(target, &exact);
            assert!(exact_report.passed, "{target:?} should pass at threshold");
            assert_eq!(exact_report.max_total_loss, max_total_loss);
            assert_eq!(exact_report.min_density_psnr_db, min_density);
            assert_eq!(exact_report.min_color_psnr_db, min_color);
            assert_eq!(exact_report.min_depth_psnr_db, min_depth);

            let weak = synthetic_render_loss(
                max_total_loss + 1.0e-3,
                min_density - 1.0e-3,
                min_color - 1.0e-3,
                min_depth - 1.0e-3,
            );
            assert!(
                !growth_3d_catalog_sanity_report(target, &weak).passed,
                "{target:?} should fail below threshold"
            );
        }
    }

    #[test]
    fn growth_3d_strict_score_tracks_distance_to_gate() {
        let checks = passing_growth_3d_strict_checks();
        let perfect_render = synthetic_render_loss(0.0, 10.0, 12.0, 14.0);
        let perfect = growth_3d_strict_score_report(
            &checks,
            Growth3dSurfaceStats {
                mean_distance: 0.2,
                max_distance: 0.2,
            },
            Growth3dSurfaceStats {
                mean_distance: 0.16,
                max_distance: 0.3,
            },
            passing_growth_3d_surface_tail_report(),
            TargetCoverageStats {
                mean_distance: 1.0,
                max_distance: 1.0,
                covered_fraction: 0.1,
            },
            TargetCoverageStats {
                mean_distance: 0.8,
                max_distance: 0.7,
                covered_fraction: 0.6,
            },
            0.72,
            &perfect_render,
        );
        assert_eq!(perfect.score, 0.0);

        let weak_render = synthetic_render_loss(1.0, 1.0, 10.0, 10.0);
        let weak = growth_3d_strict_score_report(
            &checks,
            Growth3dSurfaceStats {
                mean_distance: 0.2,
                max_distance: 0.2,
            },
            Growth3dSurfaceStats {
                mean_distance: 0.22,
                max_distance: 0.5,
            },
            Growth3dSurfaceTailReport {
                p95_distance: 0.45,
                p99_distance: 0.5,
                max_distance: 0.5,
                over_threshold_count: 16,
                over_threshold_fraction: 0.10,
                opacity_weighted_over_threshold_fraction: 0.08,
                ..passing_growth_3d_surface_tail_report()
            },
            TargetCoverageStats {
                mean_distance: 1.0,
                max_distance: 1.0,
                covered_fraction: 0.1,
            },
            TargetCoverageStats {
                mean_distance: 0.9,
                max_distance: 0.8,
                covered_fraction: 0.4,
            },
            0.72,
            &weak_render,
        );
        assert!(weak.score > perfect.score);
        assert!(weak.surface_mean_penalty > 0.0);
        assert!(weak.surface_max_penalty > 0.0);
        assert!(weak.target_coverage_fraction_penalty > 0.0);
        assert!(weak.render_density_penalty > 0.0);
    }

    #[test]
    fn growth_3d_strict_checks_require_sustained_motion() {
        let sustained_motion = growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]);
        assert!(sustained_motion.active_step_fraction >= 0.50);
        assert!(sustained_motion.sustained_step_fraction >= 0.25);

        let one_shot_motion = growth_3d_motion_report(&[0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(one_shot_motion.active_step_fraction < 0.50);
        assert!(one_shot_motion.sustained_step_fraction < 0.25);

        let activation = Growth3dActivationReport {
            active_seed_count: 4,
            inactive_seed_count: 124,
            final_active_count: 64,
            newly_activated_count: 60,
            newly_activated_fraction: 0.75,
            final_active_mean_radius: 0.25,
            final_active_max_radius: 0.30,
        };
        let initial_surface = Growth3dSurfaceStats {
            mean_distance: 1.0,
            max_distance: 1.0,
        };
        let final_surface = Growth3dSurfaceStats {
            mean_distance: 0.5,
            max_distance: 0.2,
        };
        let initial_coverage = TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.0,
        };
        let final_coverage = TargetCoverageStats {
            mean_distance: 0.5,
            max_distance: 0.3,
            covered_fraction: 0.75,
        };
        let bulk_temporal = Growth3dTemporalReport {
            samples: Vec::new(),
            first_growth_step: Some(8),
            half_activation_step: Some(8),
            full_activation_step: Some(8),
            activation_span_steps: 0,
            progressive_activation: false,
            surface_mean_ratio: 1.0,
            target_coverage_mean_ratio: 1.0,
            target_coverage_fraction_delta: 0.0,
            geometry_progressive: false,
        };
        let staged_temporal = Growth3dTemporalReport {
            samples: Vec::new(),
            first_growth_step: Some(2),
            half_activation_step: Some(8),
            full_activation_step: Some(16),
            activation_span_steps: 14,
            progressive_activation: true,
            surface_mean_ratio: 0.5,
            target_coverage_mean_ratio: 0.5,
            target_coverage_fraction_delta: 0.75,
            geometry_progressive: true,
        };
        let local_front = passing_growth_3d_front_report();

        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            None,
            &one_shot_motion,
            &local_front,
            &staged_temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(!checks.sustained_motion);
        assert!(!checks.passed);
        assert!(checks.failure_reasons.contains(&"sustained_motion"));

        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            None,
            &sustained_motion,
            &local_front,
            &bulk_temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(checks.sustained_motion);
        assert!(!checks.temporal_activation_progressive);
        assert!(!checks.passed);
        assert!(
            checks
                .failure_reasons
                .contains(&"temporal_activation_progressive")
        );
        assert!(
            checks
                .failure_reasons
                .contains(&"temporal_geometry_progressive")
        );

        let high_opacity = Growth3dOpacityStats {
            max: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT + 1.0,
            active_max: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT + 1.0,
            ..passing_growth_3d_opacity_stats()
        };
        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            high_opacity,
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            None,
            &sustained_motion,
            &local_front,
            &staged_temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(!checks.bounded_final_opacity);
        assert!(checks.failure_reasons.contains(&"bounded_final_opacity"));

        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            neutral_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            None,
            &sustained_motion,
            &local_front,
            &staged_temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(!checks.color_state_emerged);
        assert!(checks.failure_reasons.contains(&"color_state_emerged"));

        let non_local_front = Growth3dFrontReport {
            passed: false,
            local_newly_activated_fraction: 0.25,
            mean_nearest_previous_active_distance: 0.7,
            max_nearest_previous_active_distance: 1.1,
            ..passing_growth_3d_front_report()
        };
        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            None,
            &sustained_motion,
            &non_local_front,
            &staged_temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(!checks.local_front_coherent);
        assert!(checks.failure_reasons.contains(&"local_front_coherent"));
    }

    #[test]
    fn growth_3d_strict_checks_reject_missing_torus_angular_coverage() {
        let activation = Growth3dActivationReport {
            active_seed_count: 4,
            inactive_seed_count: 124,
            final_active_count: 64,
            newly_activated_count: 60,
            newly_activated_fraction: 0.75,
            final_active_mean_radius: 0.25,
            final_active_max_radius: 0.30,
        };
        let initial_surface = Growth3dSurfaceStats {
            mean_distance: 1.0,
            max_distance: 1.0,
        };
        let final_surface = Growth3dSurfaceStats {
            mean_distance: 0.5,
            max_distance: 0.2,
        };
        let initial_coverage = TargetCoverageStats {
            mean_distance: 1.0,
            max_distance: 1.0,
            covered_fraction: 0.0,
        };
        let final_coverage = TargetCoverageStats {
            mean_distance: 0.5,
            max_distance: 0.3,
            covered_fraction: 0.75,
        };
        let motion = growth_3d_motion_report(&[0.012, 0.013, 0.011, 0.010]);
        let front = passing_growth_3d_front_report();
        let temporal = Growth3dTemporalReport {
            samples: Vec::new(),
            first_growth_step: Some(2),
            half_activation_step: Some(8),
            full_activation_step: Some(16),
            activation_span_steps: 14,
            progressive_activation: true,
            surface_mean_ratio: 0.5,
            target_coverage_mean_ratio: 0.5,
            target_coverage_fraction_delta: 0.75,
            geometry_progressive: true,
        };
        let missing_tube_support = TorusAngularCoverageReport {
            ring_bins: 24,
            tube_bins: 16,
            threshold: 0.0972,
            covered_joint_bins: 187,
            covered_ring_bins: 24,
            covered_tube_bins: 9,
            joint_coverage_fraction: 0.486_979_16,
            ring_coverage_fraction: 1.0,
            tube_coverage_fraction: 0.5625,
            max_ring_gap_bins: 0,
            max_tube_gap_bins: 7,
            mean_distance: 0.159,
            max_distance: 0.420,
        };
        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            Some(&missing_tube_support),
            &motion,
            &front,
            &temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(!checks.torus_angular_coverage);
        assert!(!checks.passed);
        assert!(checks.failure_reasons.contains(&"torus_angular_coverage"));

        let full_tube_support = TorusAngularCoverageReport {
            covered_joint_bins: 288,
            covered_tube_bins: 16,
            joint_coverage_fraction: 0.75,
            tube_coverage_fraction: 1.0,
            max_tube_gap_bins: 0,
            ..missing_tube_support
        };
        let checks = growth_3d_strict_checks_report(
            false,
            true,
            0.0,
            passing_growth_3d_opacity_stats(),
            neutral_growth_3d_color_state_report(),
            emerged_growth_3d_color_state_report(),
            &passing_growth_3d_permutation_report(),
            &activation,
            initial_surface,
            final_surface,
            passing_growth_3d_surface_tail_report(),
            initial_coverage,
            final_coverage,
            Some(&full_tube_support),
            &motion,
            &front,
            &temporal,
            0.25,
            0.72,
            128,
            true,
        );
        assert!(checks.torus_angular_coverage);
        assert!(checks.passed);
    }

    #[test]
    fn render_proxy_gradient_rows_cover_full_cloud_instead_of_prefix_only() {
        assert_eq!(
            render_proxy_gradient_row_indices(1024, 8),
            vec![0, 128, 256, 384, 512, 640, 768, 896]
        );
        assert_eq!(render_proxy_gradient_row_indices(4, 8), vec![0, 1, 2, 3]);
        assert_eq!(render_proxy_gradient_row_indices(1024, 1), vec![0]);
    }

    #[test]
    fn trajectory_render_sample_indices_cover_late_rollout_evenly() {
        assert_eq!(trajectory_render_sample_indices(0, 4), Vec::<usize>::new());
        assert_eq!(trajectory_render_sample_indices(8, 0), Vec::<usize>::new());
        assert_eq!(trajectory_render_sample_indices(8, 3), vec![1, 4, 7]);
        assert_eq!(trajectory_render_sample_indices(4, 16), vec![0, 1, 2, 3]);
    }

    #[test]
    fn mesh_rollout_snapshot_steps_include_initial_and_final_when_temporal() {
        assert_eq!(mesh_rollout_snapshot_steps(8, 1), vec![8]);
        assert_eq!(mesh_rollout_snapshot_steps(8, 3), vec![0, 4, 8]);
        assert_eq!(mesh_rollout_snapshot_steps(8, 4), vec![0, 2, 5, 8]);
        assert_eq!(mesh_rollout_snapshot_steps(0, 4), vec![0]);
    }

    #[test]
    fn mesh_rollout_row_indices_keep_sparse_high_signal_rows() {
        let output_dims = 6;
        let particle_count = 32;
        let row_budget = 6;
        let mut target_update = vec![0.0_f32; particle_count * output_dims];
        target_update[17 * output_dims + 3] = 2.0;
        target_update[23 * output_dims] = -1.5;

        let rows =
            mesh_rollout_row_indices(&target_update, output_dims, particle_count, row_budget);

        assert_eq!(rows.len(), row_budget);
        assert!(
            rows.contains(&17) && rows.contains(&23),
            "sparse front/material rows should not be lost to uniform spread sampling: {rows:?}"
        );
        assert_eq!(
            rows.iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            rows.len()
        );
    }

    #[test]
    fn render_selection_metrics_average_base_and_selection_seed_with_morphology_penalty() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model =
            local_growth_student_model(config, 17, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
        let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
        let render = RenderLossConfig {
            image_size: 8,
            target_samples: 128,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        };
        let cfg = RenderProxyTrainingConfig {
            target: MeshTargetArg::Torus,
            rounds: 1,
            supervised_steps_per_round: 1,
            particles: 128,
            rollout_steps: 2,
            gradient_particles: 4,
            gradient_mode: RenderGradientModeArg::Analytic,
            finite_diff_eps: 1.0e-3,
            motion_gain: 0.1,
            perception_position_gain: 0.05,
            max_update_norm: 0.1,
            trajectory_supervision: true,
            trajectory_render_gain: 0.0,
            trajectory_render_samples: 0,
            coverage_gain: 0.0,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            full_coverage_adjoint: false,
            surface_gain: 0.0,
            opacity_gain: 0.0,
            max_opacity_update: 0.05,
            direct_line_search: false,
            direct_line_search_scales: vec![1.0],
            direct_material_output_only: false,
            training_backend: RenderTrainingBackendArg::Proxy,
            direct_selection_seed_training: false,
            seed: 11,
            selection_seed: Some(19),
            selection_seeds: vec![23, 11, 19],
            seed_scale: 0.72,
            seed_mode: ParticleSeed::TorusGrowth3d,
            render,
            sgd: SgdConfig {
                learning_rate: 1.0e-4,
                grad_clip_norm: 0.1,
                weight_decay: 0.0,
            },
        };

        let base =
            render_selection_case_metrics(&model, &grid, &target, &cfg, render, cfg.seed).unwrap();
        let heldout =
            render_selection_case_metrics(&model, &grid, &target, &cfg, render, 19).unwrap();
        let extra =
            render_selection_case_metrics(&model, &grid, &target, &cfg, render, 23).unwrap();
        let baseline = render_selection_baseline(&model, &grid, &target, &cfg, render).unwrap();
        let selection =
            render_selection_metrics(&model, &grid, &target, &cfg, render, Some(&baseline))
                .unwrap();

        assert!((selection.base_report.total_loss - base.render_loss.total_loss).abs() <= 1.0e-6);
        assert!(
            (selection.render_loss
                - (base.render_loss.total_loss
                    + heldout.render_loss.total_loss
                    + extra.render_loss.total_loss)
                    / 3.0)
                .abs()
                <= 1.0e-6
        );
        assert_eq!(render_proxy_selection_seeds(&cfg), vec![cfg.seed, 19, 23]);
        let expected_score = base.score.max(heldout.score).max(extra.score);
        assert!(
            (selection.score - expected_score).abs() <= 1.0e-5,
            "selection score {} expected worst-case {} from base {} heldout {}",
            selection.score,
            expected_score,
            base.score,
            heldout.score
        );
        let candidate_scores = [
            (cfg.seed, base.score),
            (19, heldout.score),
            (23, extra.score),
        ];
        assert!(
            candidate_scores.iter().any(|(seed, score)| {
                *seed == selection.worst_seed && (*score - expected_score).abs() <= 1.0e-5
            }),
            "worst seed {} should be one of the max-score candidates {:?}",
            selection.worst_seed,
            candidate_scores
        );
        assert!(
            !selection.worst_failure_reasons.is_empty(),
            "worst selection seed should expose strict failure reasons"
        );
        assert!(
            selection
                .worst_failure_reasons
                .contains(&"torus_angular_coverage"),
            "torus render-proxy selection must preserve angular-support blockers"
        );
        assert!(
            (selection.density_psnr_db
                - (base.render_loss.density_psnr_db
                    + heldout.render_loss.density_psnr_db
                    + extra.render_loss.density_psnr_db)
                    / 3.0)
                .abs()
                <= 1.0e-6
        );
        assert_eq!(
            selection.active_surface_max,
            base.active_surface
                .max_distance
                .max(heldout.active_surface.max_distance)
                .max(extra.active_surface.max_distance)
        );
        assert_eq!(
            selection.target_coverage_fraction,
            base.target_coverage
                .covered_fraction
                .min(heldout.target_coverage.covered_fraction)
                .min(extra.target_coverage.covered_fraction)
        );
        assert!(
            selection.score >= selection.render_loss,
            "morphology penalty should never reduce the render objective"
        );
        assert!(
            selection.morphology_non_regressed,
            "unchanged model should not regress against its own baseline"
        );
    }

    #[test]
    fn render_selection_candidate_requires_morphology_and_render_nonregression() {
        assert!(render_selection_candidate_beats(
            0.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
        ));
        assert!(!render_selection_candidate_beats(
            0.5, 1.0, false, 0.8, 0.9, 2.0, 1.5,
        ));
        assert!(!render_selection_candidate_beats(
            1.5, 1.0, true, 0.8, 0.9, 2.0, 1.5,
        ));
        assert!(
            !render_selection_candidate_beats(0.5, 1.0, true, 0.95, 0.9, 2.0, 1.5),
            "strict score improvement should not accept worse render loss"
        );
        assert!(
            !render_selection_candidate_beats(0.5, 1.0, true, 0.8, 0.9, 1.4, 1.5),
            "strict score improvement should not accept density PSNR regression"
        );
    }

    #[test]
    fn material_output_only_gradients_freeze_hidden_and_motion_rows() {
        let config = NpaConfig::growing_3dgs();
        let model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let mut gradients = zero_supervised_gradients(&model);
        gradients.w1.fill(1.0);
        gradients.b1.fill(1.0);
        gradients.w2.fill(1.0);
        gradients.b2.fill(1.0);

        retain_material_output_gradients(&model, &mut gradients).unwrap();

        assert!(gradients.w1.iter().all(|value| *value == 0.0));
        assert!(gradients.b1.iter().all(|value| *value == 0.0));
        let material_channel = growth_3d_material_opacity_channel(model.config.state_dims).unwrap();
        let material_output = model.config.spatial_dims + material_channel;
        for output in 0..model.config.update_dims() {
            let row = &gradients.w2
                [output * model.config.hidden_dims..(output + 1) * model.config.hidden_dims];
            if output == material_output {
                assert!(row.iter().all(|value| *value == 1.0));
                assert_eq!(gradients.b2[output], 1.0);
            } else {
                assert!(row.iter().all(|value| *value == 0.0));
                assert_eq!(gradients.b2[output], 0.0);
            }
        }
    }

    #[test]
    fn render_proxy_trajectory_batch_applies_bounded_coverage_updates() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model =
            local_growth_student_model(config, 17, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
        let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
        let cfg = RenderProxyTrainingConfig {
            target: MeshTargetArg::Torus,
            rounds: 1,
            supervised_steps_per_round: 1,
            particles: 32,
            rollout_steps: 2,
            gradient_particles: 32,
            gradient_mode: RenderGradientModeArg::Analytic,
            finite_diff_eps: 1.0e-3,
            motion_gain: 0.0,
            perception_position_gain: 0.05,
            max_update_norm: 0.05,
            trajectory_supervision: true,
            trajectory_render_gain: 0.0,
            trajectory_render_samples: 0,
            coverage_gain: 0.25,
            coverage_samples: 128,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            full_coverage_adjoint: false,
            surface_gain: 0.0,
            opacity_gain: 0.0,
            max_opacity_update: 0.05,
            direct_line_search: false,
            direct_line_search_scales: vec![1.0],
            direct_material_output_only: false,
            training_backend: RenderTrainingBackendArg::Proxy,
            direct_selection_seed_training: false,
            seed: 11,
            selection_seed: None,
            selection_seeds: Vec::new(),
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UniformCircle,
            render: RenderLossConfig {
                image_size: 8,
                target_samples: 128,
                world_scale: 1.44,
                ..RenderLossConfig::default()
            },
            sgd: SgdConfig {
                learning_rate: 1.0e-4,
                grad_clip_norm: 0.1,
                weight_decay: 0.0,
            },
        };
        let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
        let gradient = RenderProxyGradientRows {
            row_indices: (0..cfg.particles).collect(),
            gradients: vec![[0.0; 3]; cfg.particles],
            opacity_gradients: vec![0.0; cfg.particles],
            color_gradients: vec![[0.0; 3]; cfg.particles],
        };
        let batch = render_proxy_supervised_batch(
            &model,
            &grid,
            &target,
            &trace,
            &trajectory,
            &gradient,
            &cfg,
        )
        .unwrap();
        let rows = batch.features.len() / model.config.perception_dims();
        assert_eq!(rows, cfg.particles * cfg.rollout_steps);
        let baseline = model.forward_update_from_features(&batch.features).unwrap();
        let output_dims = model.config.update_dims();
        let mut changed_motion_rows = 0usize;
        for row in 0..rows {
            let base = row * output_dims;
            let delta = ((batch.target_update[base] - baseline[base]).powi(2)
                + (batch.target_update[base + 1] - baseline[base + 1]).powi(2)
                + (batch.target_update[base + 2] - baseline[base + 2]).powi(2))
            .sqrt();
            if delta > 0.0 {
                changed_motion_rows += 1;
                assert!(delta <= cfg.max_update_norm + 1.0e-5);
            }
        }
        assert!(changed_motion_rows > 0);
    }

    #[test]
    fn soft_chamfer_coverage_distributes_symmetric_target_pressure() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let positions = vec![[-0.02, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];

        let hard = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            1,
            f32::INFINITY,
            CoverageUpdateModeArg::HardNearest,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let soft = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            1,
            f32::INFINITY,
            CoverageUpdateModeArg::SoftChamfer,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let soft_nonzero = soft
            .iter()
            .filter(|update| update.iter().any(|value| value.abs() > 1.0e-6))
            .count();

        assert!(hard.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(soft_nonzero, 2);
        assert!(soft[0][0] > 0.0);
        assert!(soft[1][0] < 0.0);
    }

    #[test]
    fn soft_chamfer_coverage_respects_update_clamp() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let positions = vec![[3.0, 4.0, 0.0, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];
        let max_update_norm = 0.05;

        let updates = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            1,
            max_update_norm,
            CoverageUpdateModeArg::SoftChamfer,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let norm = (updates[0][0].powi(2) + updates[0][1].powi(2) + updates[0][2].powi(2)).sqrt();

        assert!(norm <= max_update_norm + 1.0e-6);
        assert!(norm > 0.0);
    }

    #[test]
    fn soft_chamfer_repulsion_adds_tangent_spread_pressure() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let positions = vec![[-0.005, 0.0, 0.0, 0.0], [0.005, 0.0, 0.0, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];
        let no_repulsion = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            1,
            f32::INFINITY,
            CoverageUpdateModeArg::SoftChamfer,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let with_repulsion = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            1,
            f32::INFINITY,
            CoverageUpdateModeArg::SoftChamfer,
            0.1,
            1.0,
            0.0,
            0.1,
            0.0,
            1.0,
        );

        assert!(no_repulsion[0][0] > 0.0);
        assert!(no_repulsion[1][0] < 0.0);
        assert!(with_repulsion[0][0] < no_repulsion[0][0]);
        assert!(with_repulsion[1][0] > no_repulsion[1][0]);
        assert!(with_repulsion[0][2].abs() <= 1.0e-6);
        assert!(with_repulsion[1][2].abs() <= 1.0e-6);
    }

    #[test]
    fn gap_farthest_coverage_avoids_symmetric_residual_cancellation() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![
                [-1.0, -0.1, 0.0],
                [-1.0, 0.1, 0.0],
                [-1.0, 0.0, 0.2],
                [1.0, -0.1, 0.0],
                [1.0, 0.1, 0.0],
                [1.0, 0.0, 0.2],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap();
        let positions = vec![[0.0, 0.0, 0.0, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];

        let hard = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            512,
            f32::INFINITY,
            CoverageUpdateModeArg::HardNearest,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let gap = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            512,
            f32::INFINITY,
            CoverageUpdateModeArg::GapFarthest,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let hard_norm = (hard[0][0].powi(2) + hard[0][1].powi(2) + hard[0][2].powi(2)).sqrt();
        let gap_norm = (gap[0][0].powi(2) + gap[0][1].powi(2) + gap[0][2].powi(2)).sqrt();

        assert!(hard.iter().flatten().all(|value| value.is_finite()));
        assert!(gap.iter().flatten().all(|value| value.is_finite()));
        assert!(
            gap_norm > hard_norm + 0.1,
            "gap mode should keep a directional worst-gap signal instead of averaging it away: hard={hard:?} gap={gap:?}"
        );
    }

    #[test]
    fn gap_farthest_coverage_balances_uncovered_bins_across_donors() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![
                [-1.0, -0.1, 0.0],
                [-1.0, 0.1, 0.0],
                [-1.0, 0.0, 0.2],
                [0.0, -0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2],
                [1.0, -0.1, 0.0],
                [1.0, 0.1, 0.0],
                [1.0, 0.0, 0.2],
            ],
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        .unwrap();
        let positions = vec![[-1.0, -0.04, 0.05, 0.0], [-0.95, 0.04, 0.05, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];

        let gap = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            512,
            10.0,
            CoverageUpdateModeArg::GapFarthest,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        assert!(
            gap.iter().all(|update| update[0] > 0.1),
            "balanced gap mode should spread uncovered right-side bins across available donors: {gap:?}"
        );
        assert!(gap.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn surface_gap_relocation_can_use_low_assignment_donors() {
        let target = TriangleMeshTarget::new(
            vec![
                [-1.0, -0.1, 0.0],
                [-1.0, 0.1, 0.0],
                [-1.0, 0.0, 0.2],
                [0.0, -0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2],
                [1.0, -0.1, 0.0],
                [1.0, 0.1, 0.0],
                [1.0, 0.0, 0.2],
            ],
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        .unwrap();
        let positions = vec![[-1.0, 0.0, 0.05, 0.0], [0.0, 0.0, 0.05, 0.0]];
        let active_rows = vec![0, 1];
        let mut updates = vec![[0.0; 3]; positions.len()];

        add_surface_gap_relocation_to_updates(
            &target,
            &positions,
            &active_rows,
            1.0,
            1.0,
            512,
            0.0,
            1.0,
            10.0,
            &mut updates,
        );

        assert!(
            updates.iter().any(|update| update[0] > 0.1),
            "a nonzero-assigned donor should be allowed to move toward the uncovered right mode: {updates:?}"
        );
        assert!(updates.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn sliced_ot_coverage_balances_separated_surface_modes() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![
                [-1.0, -0.1, 0.0],
                [-1.0, 0.1, 0.0],
                [-1.0, 0.0, 0.2],
                [1.0, -0.1, 0.0],
                [1.0, 0.1, 0.0],
                [1.0, 0.0, 0.2],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap();
        let positions = vec![[-0.05, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]];
        let states = vec![0.0; positions.len() * config.state_dims];

        let updates = render_proxy_target_coverage_updates(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            512,
            f32::INFINITY,
            CoverageUpdateModeArg::SlicedOt,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        assert!(
            updates[0][0] < 0.0,
            "left-ranked particle should be pulled toward the left target mode: {updates:?}"
        );
        assert!(
            updates[1][0] > 0.0,
            "right-ranked particle should be pulled toward the right target mode: {updates:?}"
        );
    }

    #[test]
    fn render_direct_rollout_backend_applies_mlp_gradients() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let mut model =
            local_growth_student_model(config, 19, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
        let before = model.weights.w2.clone();
        let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
        let cfg = RenderProxyTrainingConfig {
            target: MeshTargetArg::Torus,
            rounds: 1,
            supervised_steps_per_round: 1,
            particles: 32,
            rollout_steps: 2,
            gradient_particles: 32,
            gradient_mode: RenderGradientModeArg::Analytic,
            finite_diff_eps: 1.0e-3,
            motion_gain: 1.0,
            perception_position_gain: 1.0,
            max_update_norm: 0.05,
            trajectory_supervision: true,
            trajectory_render_gain: 0.0,
            trajectory_render_samples: 0,
            coverage_gain: 0.0,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            full_coverage_adjoint: false,
            surface_gain: 0.0,
            opacity_gain: 0.0,
            max_opacity_update: 0.05,
            direct_line_search: true,
            direct_line_search_scales: vec![0.5, 1.0, 2.0],
            direct_material_output_only: false,
            training_backend: RenderTrainingBackendArg::DirectRollout,
            direct_selection_seed_training: false,
            seed: 13,
            selection_seed: None,
            selection_seeds: Vec::new(),
            seed_scale: 0.72,
            seed_mode: ParticleSeed::UniformCircle,
            render: RenderLossConfig {
                image_size: 8,
                target_samples: 128,
                world_scale: 1.44,
                ..RenderLossConfig::default()
            },
            sgd: SgdConfig {
                learning_rate: 1.0e-3,
                grad_clip_norm: 1.0,
                weight_decay: 0.0,
            },
        };
        let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
        let gradient = RenderProxyGradientRows {
            row_indices: (0..cfg.particles).collect(),
            gradients: vec![[1.0, 0.25, -0.5]; cfg.particles],
            opacity_gradients: vec![0.0; cfg.particles],
            color_gradients: vec![[0.1, -0.2, 0.05]; cfg.particles],
        };
        let report = render_direct_rollout_training_step(
            &mut model,
            &grid,
            &target,
            &trace,
            &trajectory,
            &gradient,
            &cfg,
        )
        .unwrap();

        assert_eq!(report.rows, cfg.particles * cfg.rollout_steps);
        assert!(report.history[0].grad_norm.is_finite());
        assert!(report.history[0].grad_norm > 0.0);
        assert_ne!(model.weights.w2, before);
    }

    #[test]
    fn terminal_position_adjoint_combines_render_and_coverage_gradients() {
        let config = NpaConfig::growing_3dgs();
        let trace = burn_automata::RolloutTrace {
            positions: vec![[0.0; 4]; 3],
            states: vec![0.0; 3 * config.state_dims],
            batch_size: 1,
            particle_count: 3,
            state_dims: config.state_dims,
            steps: 0,
            mean_dx: Vec::new(),
        };
        let gradient = RenderProxyGradientRows {
            row_indices: vec![1],
            gradients: vec![[0.5, -0.25, 0.1]],
            opacity_gradients: vec![0.0],
            color_gradients: vec![[0.0; 3]],
        };
        let mut coverage = vec![[0.0; 3]; 3];
        coverage[1] = [0.1, 0.05, -0.2];
        coverage[2] = [0.2, 0.0, -0.1];

        let adjoint =
            terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 2.0, true, 1);

        assert_eq!(adjoint[0], [0.0; 4]);
        assert!((adjoint[1][0] - 0.8).abs() <= 1.0e-6);
        assert!((adjoint[1][1] + 0.6).abs() <= 1.0e-6);
        assert!((adjoint[1][2] - 0.6).abs() <= 1.0e-6);
        assert!((adjoint[2][0] + 0.4).abs() <= 1.0e-6);
        assert_eq!(adjoint[2][1], 0.0);
        assert!((adjoint[2][2] - 0.2).abs() <= 1.0e-6);
        assert_eq!(adjoint[2][3], 0.0);

        let sampled_only =
            terminal_render_position_adjoint(&config, &trace, &gradient, &coverage, 2.0, false, 1);
        assert_eq!(sampled_only[0], [0.0; 4]);
        assert!((sampled_only[1][0] - 0.8).abs() <= 1.0e-6);
        assert_eq!(sampled_only[2], [0.0; 4]);
    }

    #[test]
    fn surface_position_adjoint_moves_only_active_particles_toward_mesh() {
        let config = NpaConfig::growing_3dgs();
        let target = TriangleMeshTarget::new(
            vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let positions = vec![[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let mut states = vec![0.0; 2 * config.state_dims];
        states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        let mut adjoint = vec![[0.0; 4]; 2];

        add_surface_position_adjoint(&config, &target, &positions, &states, 0.5, &mut adjoint);

        assert!(adjoint[0][0].abs() <= 1.0e-6);
        assert!(adjoint[0][1].abs() <= 1.0e-6);
        assert!(adjoint[0][2] > 0.49 && adjoint[0][2] < 0.51);
        assert_eq!(adjoint[1], [0.0; 4]);
    }

    #[test]
    fn growth_3d_validation_rejects_shortcut_lineage() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model =
            local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN)
                .unwrap();
        let path = bin_temp_path("shortcut_growth3d.bpk");
        let manifest = BpkModelManifest::from_model(
            &model,
            grid,
            Some("render-proxy-rust:Torus:field-baseline".to_string()),
        );
        burn_automata::import::save_manifest(&path, &manifest).unwrap();

        let report = growth_3d_validation_report(
            &path,
            MeshTargetArg::Torus,
            growth_validation_test_config(ParticleSeed::TorusGrowth3d),
        )
        .unwrap();
        std::fs::remove_file(&path).ok();

        assert!(!report.local_conditionless_lineage);
        assert!(!report.gate_passed);
        assert!(!report.strict_passed);
    }

    fn growth_validation_test_config(seed_mode: ParticleSeed) -> Growth3dValidationConfig {
        Growth3dValidationConfig {
            particle_count: 256,
            steps: 4,
            seed: 42,
            extra_seeds: Vec::new(),
            seed_scale: UV_TORUS_FIELD_SCALE,
            seed_mode,
            gate: Growth3dValidationGateArg::Strict,
            render: RenderLossConfig {
                image_size: 8,
                target_samples: 64,
                world_scale: UV_TORUS_FIELD_SCALE * 2.0,
                ..RenderLossConfig::default()
            },
        }
    }

    fn synthetic_render_loss(
        total_loss: f32,
        density_psnr_db: f32,
        color_psnr_db: f32,
        depth_psnr_db: f32,
    ) -> MultiViewRenderLossReport {
        MultiViewRenderLossReport {
            passed: false,
            image_size: 48,
            target_samples: 1024,
            total_loss,
            density_mse: 0.0,
            color_mse: 0.0,
            depth_mse: 0.0,
            density_psnr_db,
            color_psnr_db,
            depth_psnr_db,
            nonzero_target_alpha_fraction: 1.0,
            nonzero_particle_alpha_fraction: 1.0,
            views: Vec::new(),
        }
    }

    fn passing_growth_3d_strict_checks() -> Growth3dStrictChecksReport {
        Growth3dStrictChecksReport {
            passed: true,
            no_position_features: true,
            local_conditionless_lineage: true,
            neutral_non_opacity_seed_state: true,
            sparse_active_seed: true,
            active_count_growth: true,
            newly_activated_fraction: true,
            active_front_expanded: true,
            nonzero_motion: true,
            sustained_motion: true,
            local_front_coherent: true,
            temporal_activation_progressive: true,
            temporal_geometry_progressive: true,
            mean_displacement_growth: true,
            bounded_final_opacity: true,
            color_state_emerged: true,
            permutation_consistent: true,
            surface_mean_improved: true,
            surface_max_bounded: true,
            surface_tail_bounded: true,
            target_coverage_mean_improved: true,
            target_coverage_max_bounded: true,
            target_coverage_fraction: true,
            torus_angular_coverage: true,
            render_loss_passed: true,
            failure_reasons: Vec::new(),
        }
    }

    fn passing_growth_3d_front_report() -> Growth3dFrontReport {
        Growth3dFrontReport {
            transition_count: 4,
            newly_activated_count: 96,
            local_newly_activated_count: 94,
            local_newly_activated_fraction: 94.0 / 96.0,
            mean_nearest_previous_active_distance: 0.08,
            max_nearest_previous_active_distance: 0.18,
            max_allowed_distance: 0.36,
            finite: true,
            passed: true,
        }
    }

    fn passing_growth_3d_opacity_stats() -> Growth3dOpacityStats {
        Growth3dOpacityStats {
            finite: true,
            min: -1.5,
            max: 1.0,
            mean: 0.0,
            active_min: -0.5,
            active_max: 1.0,
            active_mean: 0.25,
            active_count: 64,
            max_allowed: GROWTH_3D_MAX_FINAL_OPACITY_LOGIT,
        }
    }

    fn neutral_growth_3d_color_state_report() -> Growth3dColorStateReport {
        Growth3dColorStateReport {
            available: true,
            finite: true,
            count: 64,
            active_count: 4,
            mean_abs: 0.0,
            max_abs: 0.0,
            active_mean_abs: 0.0,
            active_max_abs: 0.0,
            active_channel_stddev: [0.0; 3],
            active_channel_stddev_mean: 0.0,
        }
    }

    fn emerged_growth_3d_color_state_report() -> Growth3dColorStateReport {
        Growth3dColorStateReport {
            available: true,
            finite: true,
            count: 64,
            active_count: 64,
            mean_abs: 0.12,
            max_abs: 0.31,
            active_mean_abs: 0.12,
            active_max_abs: 0.31,
            active_channel_stddev: [0.05, 0.04, 0.06],
            active_channel_stddev_mean: 0.05,
        }
    }

    fn passing_growth_3d_permutation_report() -> Growth3dPermutationReport {
        Growth3dPermutationReport {
            particle_count: 128,
            steps: 8,
            max_position_error: 1.0e-6,
            mean_position_error: 1.0e-7,
            max_state_error: 1.0e-6,
            mean_state_error: 1.0e-7,
            passed: true,
        }
    }

    fn passing_growth_3d_surface_tail_report() -> Growth3dSurfaceTailReport {
        Growth3dSurfaceTailReport {
            count: 64,
            threshold: GROWTH_3D_SURFACE_MAX_DISTANCE,
            p95_distance: 0.20,
            p99_distance: 0.30,
            max_distance: 0.30,
            over_threshold_count: 0,
            over_threshold_fraction: 0.0,
            opacity_weighted_mean_distance: 0.12,
            opacity_weighted_over_threshold_fraction: 0.0,
        }
    }

    fn bin_temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "burn_automata_bin_{}_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test"),
            name
        ));
        path
    }

    #[test]
    fn mesh_local_rollout_rows_do_not_require_position_features() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 13);
        assert!(!model.config.position_features);

        let batch = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &uv_torus_mesh_target(0.72),
            MeshFieldRolloutBatchConfig {
                max_rows: 16,
                particle_count: 32,
                rollout_steps: 2,
                rollouts: 1,
                temporal_samples: 1,
                seed: 17,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::UniformCircle,
                motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
                max_update_norm: f32::INFINITY,
                coverage_gain: 0.0,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: 0.0,
                color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                aux_state_gain: 1.0,
                opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                front_opacity_gain: 0.0,
                front_radius: 0.0,
                front_max_opacity_update: 0.0,
                front_motion_gate: false,
                preserve_opacity_update: false,
            },
        )
        .unwrap();

        assert_eq!(batch.features.len(), 16 * config.perception_dims());
        assert_eq!(batch.target_update.len(), 16 * config.update_dims());
    }

    #[test]
    fn rollout_local_growth_seed_uses_mesh_objective_not_static_residual_teacher() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let seed_scale = UV_TORUS_FIELD_SCALE;
        let particle_count = 64;
        let output_dims = config.update_dims();
        let residual_teacher = torus_morphogen_model(config.clone()).unwrap();
        let static_teacher_batch = rollout_supervised_batch_from_model(
            &residual_teacher,
            &residual_teacher,
            &grid,
            SupervisedTarget::Teacher(&residual_teacher),
            RolloutSupervisionConfig {
                max_rows: particle_count,
                particle_count,
                rollout_steps: 1,
                rollouts: 1,
                update_prob: 1.0,
                seed: 0x70_75,
                seed_scale,
                seed_mode: ParticleSeed::TorusGrowth3d,
                ..RolloutSupervisionConfig::default()
            },
        )
        .unwrap();
        let max_static_teacher_motion = static_teacher_batch
            .target_update
            .chunks_exact(output_dims)
            .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
            .fold(0.0_f32, f32::max);

        let target = uv_torus_mesh_target(seed_scale);
        let local_student = local_growth_student_model_with_axis_gains(
            config.clone(),
            0x70_75,
            0.0,
            mesh_axis_expansion_gains(&target, LOCAL_GROWTH_EXPANSION_GAIN),
        )
        .unwrap();
        let mesh_objective_batch = mesh_local_rollout_supervised_batch(
            &local_student,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                max_rows: particle_count,
                particle_count,
                rollout_steps: 1,
                rollouts: 1,
                temporal_samples: 1,
                seed: 0x70_75,
                seed_scale,
                seed_mode: ParticleSeed::TorusGrowth3d,
                motion_gain: LOCAL_TORUS_MOTION_GAIN,
                max_update_norm: 0.25,
                coverage_gain: 5.0e-2,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: LOCAL_GROWTH_EXTENT_GAIN,
                color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                aux_state_gain: 1.0,
                opacity_gain: 0.0,
                front_opacity_gain: LOCAL_GROWTH_FRONT_OPACITY_GAIN,
                front_radius: LOCAL_GROWTH_FRONT_RADIUS,
                front_max_opacity_update: LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
                front_motion_gate: true,
                preserve_opacity_update: false,
            },
        )
        .unwrap();
        let max_mesh_objective_motion = mesh_objective_batch
            .target_update
            .chunks_exact(output_dims)
            .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
            .fold(0.0_f32, f32::max);

        assert!(
            max_static_teacher_motion < max_mesh_objective_motion * 0.5,
            "residual teacher should stay weaker than rollout-local mesh supervision with seed-coordinate scaffolds, residual={max_static_teacher_motion} mesh={max_mesh_objective_motion}"
        );
        assert!(
            max_mesh_objective_motion > 1.0e-3,
            "rollout-local mesh objective should produce nonzero motion targets from neutral growth seeds, got {max_mesh_objective_motion}"
        );
    }

    #[test]
    fn mesh_local_rollout_rows_keep_full_cloud_perception_context() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 13);
        let rollout_cfg = RolloutConfig {
            particle_count: 32,
            steps: 2,
            update_prob: 1.0,
            seed: 17,
            seed_scale: 0.72,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(&model, &grid, &rollout_cfg, ParticleSeed::UniformCircle).unwrap();
        let full_step = model
            .step_cpu(
                &trace.positions,
                &trace.states,
                1,
                trace.particle_count,
                &grid,
                1.0,
                None,
            )
            .unwrap();

        let batch = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &uv_torus_mesh_target(0.72),
            MeshFieldRolloutBatchConfig {
                max_rows: 16,
                particle_count: rollout_cfg.particle_count,
                rollout_steps: rollout_cfg.steps,
                rollouts: 1,
                temporal_samples: 1,
                seed: rollout_cfg.seed,
                seed_scale: rollout_cfg.seed_scale,
                seed_mode: ParticleSeed::UniformCircle,
                motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
                max_update_norm: f32::INFINITY,
                coverage_gain: 0.0,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: 0.0,
                color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                aux_state_gain: 1.0,
                opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                front_opacity_gain: 0.0,
                front_radius: 0.0,
                front_max_opacity_update: 0.0,
                front_motion_gate: false,
                preserve_opacity_update: false,
            },
        )
        .unwrap();

        let input_dims = config.perception_dims();
        let output_dims = config.update_dims();
        let target = uv_torus_mesh_target(0.72);
        let target_update = mesh_field_target_update_for_rows(
            &config,
            &target,
            &trace.positions,
            &trace.states,
            UV_TORUS_FIELD_MOTION_GAIN,
            f32::INFINITY,
            UV_TORUS_FIELD_COLOR_GAIN,
            1.0,
            UV_TORUS_FIELD_OPACITY_GAIN,
            0.0,
            0.0,
            0.0,
            false,
        );
        let row_indices =
            mesh_rollout_row_indices(&target_update, output_dims, rollout_cfg.particle_count, 16);
        assert_eq!(row_indices.len(), 16);
        for (batch_row, full_row) in row_indices.iter().copied().enumerate() {
            let batch_base = batch_row * input_dims;
            let full_base = full_row * input_dims;
            assert_eq!(
                &batch.features[batch_base..batch_base + input_dims],
                &full_step.perception.features[full_base..full_base + input_dims]
            );
        }
    }

    #[test]
    fn mesh_local_temporal_rollout_rows_include_initial_snapshot() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 13);
        let seed = 17;
        let seed_scale = 0.72;
        let (positions, states) = seed_particles_scaled(
            1,
            16,
            config.state_dims,
            config.spatial_dims,
            seed,
            ParticleSeed::TorusGrowth3d,
            seed_scale,
        );
        let initial_step = model
            .step_cpu(&positions, &states, 1, 16, &grid, 1.0, None)
            .unwrap();
        let batch = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &uv_torus_mesh_target(seed_scale),
            MeshFieldRolloutBatchConfig {
                max_rows: 12,
                particle_count: 16,
                rollout_steps: 4,
                rollouts: 1,
                temporal_samples: 3,
                seed,
                seed_scale,
                seed_mode: ParticleSeed::TorusGrowth3d,
                motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
                max_update_norm: 0.25,
                coverage_gain: 0.0,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: 0.0,
                color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                aux_state_gain: 1.0,
                opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                front_opacity_gain: 0.0,
                front_radius: 0.0,
                front_max_opacity_update: 0.0,
                front_motion_gate: false,
                preserve_opacity_update: false,
            },
        )
        .unwrap();

        let input_dims = config.perception_dims();
        assert_eq!(batch.features.len(), 12 * input_dims);
        assert_eq!(batch.target_update.len(), 12 * config.update_dims());
        assert_eq!(
            &batch.features[..input_dims],
            &initial_step.perception.features[..input_dims],
            "first temporal batch row should come from the initial rollout snapshot"
        );
    }

    #[test]
    fn mesh_local_rollout_can_preserve_opacity_update_targets() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 13);
        let seed = 17;
        let seed_scale = 0.72;
        let particle_count = 16;
        let (positions, states) = seed_particles_scaled(
            1,
            particle_count,
            config.state_dims,
            config.spatial_dims,
            seed,
            ParticleSeed::TorusGrowth3d,
            seed_scale,
        );
        let initial_step = model
            .step_cpu(&positions, &states, 1, particle_count, &grid, 1.0, None)
            .unwrap();
        let batch = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &uv_torus_mesh_target(seed_scale),
            MeshFieldRolloutBatchConfig {
                max_rows: particle_count,
                particle_count,
                rollout_steps: 0,
                rollouts: 1,
                temporal_samples: 1,
                seed,
                seed_scale,
                seed_mode: ParticleSeed::TorusGrowth3d,
                motion_gain: 0.0,
                max_update_norm: 0.25,
                coverage_gain: 0.0,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: 0.0,
                color_gain: 0.0,
                aux_state_gain: 0.0,
                opacity_gain: 0.0,
                front_opacity_gain: 0.0,
                front_radius: 0.0,
                front_max_opacity_update: 0.0,
                front_motion_gate: false,
                preserve_opacity_update: true,
            },
        )
        .unwrap();

        let output_dims = config.update_dims();
        for row in 0..particle_count {
            let target_opacity = batch.target_update[row * output_dims + config.spatial_dims + 3];
            let current_opacity_update = initial_step.ds[row * config.state_dims + 3];
            assert!(
                (target_opacity - current_opacity_update).abs() <= 1.0e-6,
                "preserved opacity target should match current model update for row {row}: target={target_opacity} current={current_opacity_update}"
            );
        }
    }

    #[test]
    fn mesh_local_rollout_rows_can_include_target_coverage_pressure() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config.clone(), 13);
        let target = uv_torus_mesh_target(0.72);
        let base_cfg = MeshFieldRolloutBatchConfig {
            max_rows: 16,
            particle_count: 32,
            rollout_steps: 2,
            rollouts: 1,
            temporal_samples: 1,
            seed: 17,
            seed_scale: 0.72,
            seed_mode: ParticleSeed::TorusGrowth3d,
            motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
            max_update_norm: 0.25,
            coverage_gain: 0.0,
            coverage_samples: 0,
            coverage_mode: CoverageUpdateModeArg::HardNearest,
            coverage_softness: 0.0,
            coverage_repulsion_gain: 0.0,
            coverage_gap_gain: 0.0,
            coverage_repulsion_radius: 0.0,
            coverage_normal_weight: 0.0,
            extent_gain: 0.0,
            color_gain: UV_TORUS_FIELD_COLOR_GAIN,
            aux_state_gain: 1.0,
            opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
            front_opacity_gain: 0.0,
            front_radius: 0.0,
            front_max_opacity_update: 0.0,
            front_motion_gate: false,
            preserve_opacity_update: false,
        };
        let no_coverage =
            mesh_local_rollout_supervised_batch(&model, &grid, &target, base_cfg).unwrap();
        let with_coverage = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                coverage_gain: 0.15,
                ..base_cfg
            },
        )
        .unwrap();
        let with_dense_coverage = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                coverage_gain: 0.15,
                coverage_samples: 2048,
                ..base_cfg
            },
        )
        .unwrap();
        let with_soft_coverage = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                coverage_gain: 0.15,
                coverage_samples: 128,
                coverage_mode: CoverageUpdateModeArg::SoftChamfer,
                coverage_softness: 0.12,
                ..base_cfg
            },
        )
        .unwrap();
        let with_gap_coverage = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                coverage_gain: 0.15,
                coverage_samples: 2048,
                coverage_mode: CoverageUpdateModeArg::SlicedOt,
                coverage_gap_gain: 1.0,
                ..base_cfg
            },
        )
        .unwrap();
        let with_soft_gap_coverage = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &target,
            MeshFieldRolloutBatchConfig {
                coverage_gain: 0.15,
                coverage_samples: 2048,
                coverage_mode: CoverageUpdateModeArg::SoftChamfer,
                coverage_softness: 0.12,
                coverage_gap_gain: 1.0,
                ..base_cfg
            },
        )
        .unwrap();

        let output_dims = config.update_dims();
        let position_delta_sum = |lhs: &SupervisedBatch, rhs: &SupervisedBatch| {
            lhs.target_update
                .chunks_exact(output_dims)
                .zip(rhs.target_update.chunks_exact(output_dims))
                .map(|(a, b)| {
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
                })
                .sum::<f32>()
        };
        let coverage_delta = position_delta_sum(&no_coverage, &with_coverage);
        let sample_budget_delta = position_delta_sum(&with_coverage, &with_dense_coverage);
        let soft_delta = position_delta_sum(&with_coverage, &with_soft_coverage);
        let gap_delta = position_delta_sum(&with_dense_coverage, &with_gap_coverage);
        let soft_gap_delta = position_delta_sum(&with_soft_coverage, &with_soft_gap_coverage);
        assert!(
            coverage_delta > 1.0e-5,
            "coverage pressure should alter at least one position update"
        );
        assert!(
            sample_budget_delta > 1.0e-6,
            "coverage sample budget should alter the coverage pressure signal"
        );
        assert!(
            soft_delta > 1.0e-6,
            "soft-chamfer coverage should not silently match hard-nearest updates"
        );
        assert!(
            gap_delta > 1.0e-6,
            "surface-gap gain should alter uncovered-surface pressure independently of tangent repulsion"
        );
        assert!(
            soft_gap_delta > 1.0e-6,
            "surface-gap gain should also alter soft/normal-aware coverage updates"
        );
    }

    #[test]
    fn surface_tangent_repulsion_separates_close_surface_particles() {
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let tangent = if sample.normal[0].abs() < 0.9 {
            [0.0, -sample.normal[2], sample.normal[1]]
        } else {
            [-sample.normal[1], sample.normal[0], 0.0]
        };
        let tangent_norm =
            (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
        let tangent = [
            tangent[0] / tangent_norm,
            tangent[1] / tangent_norm,
            tangent[2] / tangent_norm,
        ];
        let positions = vec![
            [
                sample.position[0] - 0.01 * tangent[0],
                sample.position[1] - 0.01 * tangent[1],
                sample.position[2] - 0.01 * tangent[2],
                1.0,
            ],
            [
                sample.position[0] + 0.01 * tangent[0],
                sample.position[1] + 0.01 * tangent[1],
                sample.position[2] + 0.01 * tangent[2],
                1.0,
            ],
        ];
        let mut updates = vec![[0.0; 3]; positions.len()];
        add_surface_tangent_repulsion_to_updates(
            &target,
            &positions,
            &[0, 1],
            1.0,
            1.0,
            0.08,
            0.72,
            1.0,
            &mut updates,
        );

        let lhs_dot =
            updates[0][0] * -tangent[0] + updates[0][1] * -tangent[1] + updates[0][2] * -tangent[2];
        let rhs_dot =
            updates[1][0] * tangent[0] + updates[1][1] * tangent[1] + updates[1][2] * tangent[2];
        assert!(
            lhs_dot > 0.0 && rhs_dot > 0.0,
            "repulsion should push close particles apart along the surface tangent, updates={updates:?}"
        );
        let projected_normal = target
            .project([positions[0][0], positions[0][1], positions[0][2]])
            .normal;
        assert!(
            (updates[0][0] * projected_normal[0]
                + updates[0][1] * projected_normal[1]
                + updates[0][2] * projected_normal[2])
                .abs()
                < 1.0e-4,
            "repulsion should remove the projected normal component"
        );
    }

    #[test]
    fn surface_gap_relocation_moves_redundant_particles_to_uncovered_regions() {
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let positions = vec![
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                1.0,
            ],
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                1.0,
            ],
        ];
        let mut updates = vec![[0.0; 3]; positions.len()];
        add_surface_gap_relocation_to_updates(
            &target,
            &positions,
            &[0, 1],
            1.0,
            1.0,
            512,
            0.0,
            0.72,
            1.0,
            &mut updates,
        );

        let update_norms = updates
            .iter()
            .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
            .collect::<Vec<_>>();
        let redundant_norm = update_norms.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            redundant_norm > 0.05,
            "a redundant active particle should receive a relocation update toward an uncovered surface gap, updates={updates:?}"
        );
        assert!(
            update_norms.iter().all(|norm| *norm <= 1.0 + 1.0e-5),
            "gap relocation should respect max_update_norm, norms={update_norms:?}"
        );
    }

    #[test]
    fn surface_normal_coverage_moves_redundant_particles_to_missing_normal_bins() {
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let positions = vec![
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                1.0,
            ],
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                1.0,
            ],
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                1.0,
            ],
        ];
        let active_rows = [0, 1, 2];
        let mut updates = vec![[0.0; 3]; positions.len()];

        add_surface_normal_coverage_to_updates(
            &target,
            &positions,
            &active_rows,
            1.0,
            1.0,
            512,
            1.0,
            &mut updates,
        );

        let max_update_norm = updates
            .iter()
            .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
            .fold(0.0_f32, f32::max);
        assert!(
            max_update_norm > 0.05,
            "normal-bin coverage should relocate a redundant particle toward an under-covered normal bin, updates={updates:?}"
        );
        assert!(updates.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn surface_gap_relocation_can_use_normal_mismatch_as_uncovered_support() {
        let target = TriangleMeshTarget::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.02],
                [0.01, 0.0, 0.02],
                [0.0, 0.01, 0.02],
            ],
            vec![[0, 1, 2], [5, 4, 3]],
        )
        .unwrap();
        let positions = vec![[0.003, 0.003, 0.0, 1.0], [0.006, 0.002, 0.0, 1.0]];
        let active_rows = [0, 1];
        let mut position_only = vec![[0.0; 3]; positions.len()];
        let mut normal_aware = vec![[0.0; 3]; positions.len()];

        add_surface_gap_relocation_to_updates(
            &target,
            &positions,
            &active_rows,
            1.0,
            1.0,
            512,
            0.0,
            0.72,
            1.0,
            &mut position_only,
        );
        add_surface_gap_relocation_to_updates(
            &target,
            &positions,
            &active_rows,
            1.0,
            1.0,
            512,
            10.0,
            0.72,
            1.0,
            &mut normal_aware,
        );

        let position_only_z = position_only
            .iter()
            .map(|update| update[2].abs())
            .fold(0.0_f32, f32::max);
        let normal_aware_z = normal_aware
            .iter()
            .map(|update| update[2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            normal_aware_z > position_only_z + 1.0e-3,
            "normal-aware gap relocation should expose nearby opposite-normal support: position_only={position_only:?} normal_aware={normal_aware:?}"
        );
    }

    #[test]
    fn gap_relocation_donor_falls_back_to_overassigned_particles() {
        let active_rows = [0, 1];
        let positions = vec![[0.0, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, 1.0]];
        let assigned_counts = vec![16, 12];
        let used_donors = vec![false, false];
        let average_assignments = 8.0;
        let gap = [0.5, 0.0, 0.0];

        let under_assigned = gap_relocation_donor(
            gap,
            &active_rows,
            &positions,
            positions.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        let fallback = gap_relocation_donor(
            gap,
            &active_rows,
            &positions,
            positions.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            false,
        );

        assert_eq!(under_assigned, None);
        assert_eq!(
            fallback,
            Some(1),
            "uncovered surface patches should still get a donor when every active particle is already assigned"
        );
    }

    #[test]
    fn mesh_axis_expansion_gains_follow_target_bounds() {
        let gains = mesh_axis_expansion_gains(&uv_torus_mesh_target(0.72), 0.05);
        assert!(gains[0] > gains[2]);
        assert!(gains[1] > gains[2]);
        assert!(gains.iter().all(|gain| gain.is_finite() && *gain > 0.0));
    }

    #[test]
    fn torus_angular_coverage_distinguishes_full_support_from_arc_collapse() {
        let config = NpaConfig::growing_3dgs();
        let scale = 0.72;
        let rings = 12;
        let tubes = 8;
        let mut full_positions = Vec::new();
        for ring in 0..rings {
            for tube in 0..tubes {
                full_positions.push(torus_angular_sample_position(
                    scale, ring, rings, tube, tubes,
                ));
            }
        }
        let mut full_states = vec![0.0_f32; full_positions.len() * config.state_dims];
        for state in full_states.chunks_exact_mut(config.state_dims) {
            state[3] = 0.0;
        }
        let full = torus_angular_coverage_report(
            &full_positions,
            &full_states,
            config.state_dims,
            scale,
            1.0e-5,
            rings,
            tubes,
        );
        assert_eq!(full.covered_joint_bins, rings * tubes);
        assert_eq!(full.max_ring_gap_bins, 0);
        assert_eq!(full.max_tube_gap_bins, 0);

        let arc_positions = full_positions[..tubes].to_vec();
        let mut arc_states = vec![0.0_f32; arc_positions.len() * config.state_dims];
        for state in arc_states.chunks_exact_mut(config.state_dims) {
            state[3] = 0.0;
        }
        let arc = torus_angular_coverage_report(
            &arc_positions,
            &arc_states,
            config.state_dims,
            scale,
            0.05,
            rings,
            tubes,
        );
        assert!(arc.ring_coverage_fraction < 0.25, "{arc:?}");
        assert_eq!(arc.tube_coverage_fraction, 1.0);
        assert!(arc.max_ring_gap_bins >= rings - 2, "{arc:?}");
    }

    #[test]
    fn active_surface_tail_report_ignores_inactive_and_tracks_opacity_weighted_tail() {
        let config = NpaConfig::growing_3dgs();
        let scale = 0.72;
        let target = uv_torus_mesh_target(scale);
        let on_surface = uv_torus_sample(0, 16, scale).position;
        let positions = vec![
            [on_surface[0], on_surface[1], on_surface[2], 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [-3.0, 0.0, 0.0, 1.0],
        ];
        let mut states = vec![0.0_f32; positions.len() * config.state_dims];
        states[3] = 4.0;
        states[config.state_dims + 3] = -0.5;
        states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;

        let report = growth_3d_active_surface_tail_report(
            &positions,
            &states,
            config.state_dims,
            &target,
            GROWTH_3D_SURFACE_MAX_DISTANCE,
        );
        assert_eq!(report.count, 2);
        assert_eq!(report.over_threshold_count, 1);
        assert!((report.over_threshold_fraction - 0.5).abs() <= 1.0e-6);
        assert!(report.p95_distance >= GROWTH_3D_SURFACE_MAX_DISTANCE);
        assert!(report.p99_distance >= report.p95_distance);
        assert!(
            report.opacity_weighted_over_threshold_fraction < report.over_threshold_fraction,
            "{report:?}"
        );
    }

    fn torus_angular_sample_position(
        scale: f32,
        ring: usize,
        ring_bins: usize,
        tube: usize,
        tube_bins: usize,
    ) -> [f32; 4] {
        let major = scale.max(1.0e-4);
        let minor = major * UV_TORUS_MINOR_RATIO;
        let theta = std::f32::consts::TAU * (ring as f32 + 0.5) / ring_bins as f32;
        let phi = std::f32::consts::TAU * (tube as f32 + 0.5) / tube_bins as f32;
        let radial = major + minor * phi.cos();
        [
            radial * theta.cos(),
            radial * theta.sin(),
            minor * phi.sin(),
            1.0,
        ]
    }

    #[test]
    fn local_growth_student_opacity_controller_expands_sparse_growth_front() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model =
            local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN)
                .unwrap();
        let (initial_positions, initial_states) = seed_particles_scaled(
            1,
            128,
            config.state_dims,
            config.spatial_dims,
            RolloutConfig::default().seed,
            ParticleSeed::TorusGrowth3d,
            UV_TORUS_FIELD_SCALE,
        );
        let initial_step = model
            .step_cpu(
                &initial_positions,
                &initial_states,
                1,
                128,
                &grid,
                1.0,
                None,
            )
            .unwrap();
        let mut max_inactive_opacity_ds = f32::MIN;
        for row in 0..128 {
            if initial_states[row * config.state_dims + 3] <= -1.0 {
                max_inactive_opacity_ds =
                    max_inactive_opacity_ds.max(initial_step.ds[row * config.state_dims + 3]);
            }
        }
        assert!(
            max_inactive_opacity_ds > 0.1,
            "inactive particles on the active front should receive positive local opacity updates, max={max_inactive_opacity_ds}"
        );
        let trace = run_rollout(
            &model,
            &grid,
            &RolloutConfig {
                particle_count: 128,
                steps: 64,
                update_prob: 1.0,
                seed_scale: UV_TORUS_FIELD_SCALE,
                ..RolloutConfig::default()
            },
            ParticleSeed::TorusGrowth3d,
        )
        .unwrap();

        let active_threshold = -1.0_f32;
        let initial_active = initial_states
            .chunks_exact(config.state_dims)
            .filter(|state| state[3] > active_threshold)
            .count();
        let final_active = trace
            .states
            .chunks_exact(config.state_dims)
            .filter(|state| state[3] > active_threshold)
            .count();
        let max_opacity = trace
            .states
            .chunks_exact(config.state_dims)
            .map(|state| state[3])
            .fold(f32::MIN, f32::max);
        let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
        let initial_material_mean = initial_states
            .chunks_exact(config.state_dims)
            .map(|state| state[material_channel])
            .sum::<f32>()
            / 128.0;
        let final_material_mean = trace
            .states
            .chunks_exact(config.state_dims)
            .map(|state| state[material_channel])
            .sum::<f32>()
            / trace.particle_count as f32;

        assert!(
            final_active > initial_active,
            "front controller should activate more particles, initial={initial_active} final={final_active}"
        );
        assert!(
            final_active < trace.particle_count,
            "front controller should not activate the whole cloud in one global sweep, final={final_active}"
        );
        assert!(
            max_opacity < UV_TORUS_FIELD_OPACITY_TARGET + 0.5,
            "front opacity should remain bounded, max opacity={max_opacity}"
        );
        assert!(
            final_material_mean > initial_material_mean + 0.25,
            "material opacity should rise with the local growth front, initial={initial_material_mean} final={final_material_mean}"
        );
    }

    #[test]
    fn active_opacity_retime_leaves_dormant_particles_untouched() {
        let config = NpaConfig::growing_3dgs();
        let mut model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let gain = 0.035;
        retime_growth_3d_active_opacity_model(&mut model, Some(32), gain).unwrap();

        let input_dims = config.perception_dims();
        let output_dims = config.update_dims();
        let opacity_out = config.spatial_dims + 3;
        let mut features = vec![0.0_f32; 3 * input_dims];
        features[3] = -3.0;
        features[input_dims + 3] = -0.5;
        features[2 * input_dims + 3] = 2.0;
        let update = model.forward_update_from_features(&features).unwrap();

        assert!(update[opacity_out].abs() < 1.0e-6);
        assert!((update[output_dims + opacity_out] - gain * 0.5).abs() < 1.0e-6);
        assert!((update[2 * output_dims + opacity_out] - gain).abs() < 1.0e-6);
    }

    #[test]
    fn opacity_bias_retime_only_offsets_opacity_output_bias() {
        let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
        let before = model.weights.b2.clone();
        let opacity_out = model.config.spatial_dims + 3;
        add_growth_3d_opacity_update_bias(&mut model, 0.0125).unwrap();
        for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
            if idx == opacity_out {
                assert!((current - initial - 0.0125).abs() <= 1.0e-7);
            } else {
                assert_eq!(current, initial);
            }
        }

        let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
        assert!(add_growth_3d_opacity_update_bias(&mut position_field, 0.01).is_err());
    }

    #[test]
    fn material_opacity_bias_retime_only_offsets_material_output_bias() {
        let mut model = NpaModel::seeded(NpaConfig::growing_3dgs(), 11);
        let before = model.weights.b2.clone();
        let material_channel = growth_3d_material_opacity_channel(model.config.state_dims).unwrap();
        let material_opacity_out = model.config.spatial_dims + material_channel;
        let liveness_opacity_out = model.config.spatial_dims + 3;
        add_growth_3d_material_opacity_update_bias(&mut model, 0.0125).unwrap();
        for (idx, (&current, &initial)) in model.weights.b2.iter().zip(before.iter()).enumerate() {
            if idx == material_opacity_out {
                assert!((current - initial - 0.0125).abs() <= 1.0e-7);
            } else {
                assert_eq!(current, initial);
            }
        }
        assert_eq!(
            model.weights.b2[liveness_opacity_out],
            before[liveness_opacity_out]
        );

        let mut position_field = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 11);
        assert!(add_growth_3d_material_opacity_update_bias(&mut position_field, 0.01).is_err());
    }

    #[test]
    fn local_front_opacity_targets_activate_only_near_active_neighbors() {
        let config = NpaConfig::growing_3dgs();
        let mut states = vec![GROWTH_3D_INACTIVE_OPACITY_LOGIT; 3 * config.state_dims];
        states[3] = 0.0;
        states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        let positions = vec![
            [0.0_f32, 0.0, 0.0, 0.0],
            [0.08_f32, 0.0, 0.0, 0.0],
            [0.8_f32, 0.0, 0.0, 0.0],
        ];

        let updates = local_front_opacity_targets(
            &config,
            &positions,
            &states,
            LOCAL_GROWTH_FRONT_OPACITY_GAIN,
            0.20,
            LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
        );

        assert!(
            updates[1] > 0.0,
            "inactive particle near an active neighbor should receive positive opacity update"
        );
        assert!(
            updates[2].abs() < 1.0e-6,
            "far inactive particle should stay dormant until the front reaches it"
        );
    }

    #[test]
    fn front_motion_gate_suppresses_far_dormant_mesh_targets() {
        let config = NpaConfig::growing_3dgs();
        let mut states = vec![0.0; 3 * config.state_dims];
        states[3] = 0.0;
        states[config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        states[2 * config.state_dims + 3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        let positions = vec![
            [0.0_f32, 0.0, 0.0, 0.0],
            [0.08_f32, 0.0, 0.0, 0.0],
            [0.8_f32, 0.0, 0.0, 0.0],
        ];
        let output_dims = config.update_dims();
        let target = uv_torus_mesh_target(0.72);

        let ungated = mesh_field_target_update_for_rows(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            f32::INFINITY,
            0.0,
            1.0,
            0.0,
            0.0,
            0.20,
            0.0,
            false,
        );
        let gated = mesh_field_target_update_for_rows(
            &config,
            &target,
            &positions,
            &states,
            1.0,
            f32::INFINITY,
            0.0,
            1.0,
            0.0,
            LOCAL_GROWTH_FRONT_OPACITY_GAIN,
            0.20,
            LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
            true,
        );
        let opacity_gated = mesh_field_target_update_for_rows(
            &config,
            &target,
            &positions,
            &states,
            0.0,
            f32::INFINITY,
            0.0,
            0.0,
            UV_TORUS_FIELD_OPACITY_GAIN,
            LOCAL_GROWTH_FRONT_OPACITY_GAIN,
            0.20,
            LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
            true,
        );

        let far_base = 2 * output_dims;
        let far_ungated_motion = (ungated[far_base].powi(2)
            + ungated[far_base + 1].powi(2)
            + ungated[far_base + 2].powi(2))
        .sqrt();
        let far_gated_motion =
            (gated[far_base].powi(2) + gated[far_base + 1].powi(2) + gated[far_base + 2].powi(2))
                .sqrt();
        let near_base = output_dims;
        let near_gated_motion = (gated[near_base].powi(2)
            + gated[near_base + 1].powi(2)
            + gated[near_base + 2].powi(2))
        .sqrt();
        let opacity_out = config.spatial_dims + 3;
        let far_gated_opacity = opacity_gated[far_base + opacity_out];
        let near_gated_opacity = opacity_gated[near_base + opacity_out];

        assert!(
            far_ungated_motion > 1.0e-4,
            "fixture should have a nonzero target motion without front gating"
        );
        assert!(
            far_gated_motion < 1.0e-6,
            "far dormant particle should not receive target motion before the active front reaches it"
        );
        assert!(
            near_gated_motion > 1.0e-4,
            "near-front inactive particle should still receive gated target motion"
        );
        assert!(
            far_gated_opacity.abs() < 1.0e-6,
            "far dormant particle should not receive direct opacity target before the active front reaches it"
        );
        assert!(
            near_gated_opacity > 0.0,
            "near-front inactive particle should receive front-gated opacity growth"
        );
    }

    #[test]
    fn mesh_opacity_targets_surface_material_instead_of_whole_domain() {
        let config = NpaConfig::growing_3dgs();
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let positions = vec![
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                0.0,
            ],
            [0.0_f32, 0.0, 0.0, 0.0],
        ];
        let mut states = vec![0.0; 2 * config.state_dims];
        let material_opacity_channel =
            growth_3d_material_opacity_channel(config.state_dims).unwrap();
        states[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        states[material_opacity_channel] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        states[config.state_dims + 3] = 0.0;
        states[config.state_dims + material_opacity_channel] = 0.0;

        let updates = mesh_field_target_update_for_rows(
            &config,
            &target,
            &positions,
            &states,
            0.0,
            f32::INFINITY,
            0.0,
            0.0,
            UV_TORUS_FIELD_OPACITY_GAIN,
            0.0,
            0.20,
            LOCAL_GROWTH_FRONT_MAX_OPACITY_UPDATE,
            false,
        );
        let opacity_out = config.spatial_dims + material_opacity_channel;

        assert!(
            updates[opacity_out] > 0.0,
            "near-surface dormant material should receive positive render opacity pressure"
        );
        assert!(
            updates[config.update_dims() + opacity_out] < 0.0,
            "off-surface active material should be suppressed instead of making the whole substrate visible"
        );
    }

    #[test]
    fn target_extent_updates_push_active_bounds_outward() {
        let config = NpaConfig::growing_3dgs();
        let positions = vec![
            [-0.10_f32, 0.0, 0.0, 0.0],
            [0.10_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 0.0, 0.0, 0.0],
        ];
        let mut target_update = vec![0.0; positions.len() * config.update_dims()];
        let target = uv_torus_mesh_target(0.72);

        add_target_extent_updates_for_rows(
            &config,
            &target,
            &positions,
            None,
            &mut target_update,
            0.10,
            0.25,
            0.30,
        );

        let output_dims = config.update_dims();
        assert!(
            target_update[0] < -1.0e-4,
            "min-x active boundary should be pushed toward target min x"
        );
        assert!(
            target_update[output_dims] > 1.0e-4,
            "max-x active boundary should be pushed toward target max x"
        );
        assert!(
            target_update[2 * output_dims].abs() < target_update[output_dims].abs(),
            "center row should receive less x extent pressure than boundary row"
        );
    }

    #[test]
    fn active_target_coverage_ignores_inactive_particles() {
        let config = NpaConfig::growing_3dgs();
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let positions = vec![
            [
                sample.position[0],
                sample.position[1],
                sample.position[2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let mut states = vec![0.0; 2 * config.state_dims];
        states[3] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
        states[config.state_dims + 3] = 0.0;

        let all = target_coverage_stats(&positions, &target, 16, 0.20);
        let active =
            active_target_coverage_stats(&positions, &states, config.state_dims, &target, 16, 0.20);

        assert!(
            all.covered_fraction > active.covered_fraction,
            "inactive particle exactly on target surface should not count toward active coverage"
        );
    }

    #[test]
    fn surface_coverage_profile_reports_sparse_target_support() {
        let target = uv_torus_mesh_target(0.72);
        let sample = target.surface_sample(0);
        let positions = vec![[
            sample.position[0],
            sample.position[1],
            sample.position[2],
            0.0,
        ]];
        let sparse = surface_coverage_profile(&positions, &target, 128, 0.05, 16);
        let empty = surface_coverage_profile(&[], &target, 128, 0.05, 16);

        assert!(sparse.covered_fraction > 0.0);
        assert!(sparse.covered_bin_fraction < 1.0);
        assert!(sparse.empty_bins > 0);
        assert_eq!(empty.covered_fraction, 0.0);
        assert_eq!(empty.assigned_particle_fraction, 0.0);
    }

    #[test]
    fn mesh_local_rollout_rows_reject_position_field_models() {
        let config = NpaConfig::torus_field_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel::seeded(config, 13);
        assert!(model.config.position_features);

        let err = mesh_local_rollout_supervised_batch(
            &model,
            &grid,
            &uv_torus_mesh_target(0.72),
            MeshFieldRolloutBatchConfig {
                max_rows: 16,
                particle_count: 32,
                rollout_steps: 2,
                rollouts: 1,
                temporal_samples: 1,
                seed: 17,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::UniformCircle,
                motion_gain: UV_TORUS_FIELD_MOTION_GAIN,
                max_update_norm: f32::INFINITY,
                coverage_gain: 0.0,
                coverage_samples: 0,
                coverage_mode: CoverageUpdateModeArg::HardNearest,
                coverage_softness: 0.0,
                coverage_repulsion_gain: 0.0,
                coverage_gap_gain: 0.0,
                coverage_repulsion_radius: 0.0,
                coverage_normal_weight: 0.0,
                extent_gain: 0.0,
                color_gain: UV_TORUS_FIELD_COLOR_GAIN,
                aux_state_gain: 1.0,
                opacity_gain: UV_TORUS_FIELD_OPACITY_GAIN,
                front_opacity_gain: 0.0,
                front_radius: 0.0,
                front_max_opacity_update: 0.0,
                front_motion_gate: false,
                preserve_opacity_update: false,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("position_features=false"));
    }

    #[test]
    fn torus_robustness_report_rejects_static_opacity_only_prior() {
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
        let mut weights = NpaWeights::zeros(&config);
        weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        let model = NpaModel {
            config: config.clone(),
            weights,
        };
        let report = torus_robustness_report_for_cases(
            &model,
            &grid,
            &[TorusRobustnessCaseConfig {
                particle_count: 64,
                steps: 4,
                seed: 11,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::UvTorusDense3d,
            }],
        )
        .unwrap();

        assert!(!report.passed);
        assert!(report.max_motion_per_step <= 1.0e-6);
        assert!(report.max_target_position_error > 0.1);
        assert!(report.max_opacity_target_error <= 1.0e-5);
        assert_eq!(report.cases.len(), 1);
    }

    #[test]
    fn torus_robustness_report_accepts_residual_motion_growth_prior() {
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
        let model = torus_growth_model(config).unwrap();
        let report = torus_robustness_report_for_cases(
            &model,
            &grid,
            &[TorusRobustnessCaseConfig {
                particle_count: 128,
                steps: 180,
                seed: 11,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::UvTorusDense3d,
            }],
        )
        .unwrap();

        assert!(
            report.passed,
            "target_position={} surface={} color={} opacity={} first_motion={} max_motion={}",
            report.max_target_position_error,
            report.max_torus_surface_error,
            report.max_color_target_error,
            report.max_opacity_target_error,
            report.first_motion_per_step,
            report.max_motion_per_step
        );
        assert!(report.first_motion_per_step >= 1.0e-3);
        assert!(report.max_motion_per_step >= 1.0e-3);
        assert!(report.max_target_position_error <= 1.2e-1);
        assert_eq!(report.cases.len(), 1);
    }

    #[test]
    fn torus_robustness_report_accepts_seed_frame_morphogen_prior() {
        let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
        let model = torus_morphogen_model(config).unwrap();
        assert!(!model.config.position_features);
        let report = torus_robustness_report_for_cases(
            &model,
            &grid,
            &[TorusRobustnessCaseConfig {
                particle_count: 128,
                steps: 180,
                seed: 11,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::TorusMorphogenDense3d,
            }],
        )
        .unwrap();

        assert!(
            report.passed,
            "target_position={} surface={} color={} opacity={} first_motion={} max_motion={}",
            report.max_target_position_error,
            report.max_torus_surface_error,
            report.max_color_target_error,
            report.max_opacity_target_error,
            report.first_motion_per_step,
            report.max_motion_per_step
        );
        assert!(report.first_motion_per_step >= 1.0e-3);
        assert!(report.max_motion_per_step >= 1.0e-3);
        assert!(report.max_target_position_error <= 1.2e-1);
        assert!(report.max_color_target_error <= 2.0e-2);
        assert_eq!(report.cases.len(), 1);
    }

    #[test]
    fn mesh_rollout_report_rejects_static_teapot_field_prior() {
        let config = NpaConfig::torus_field_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let report = mesh_rollout_report_for_cases(
            &model,
            &grid,
            &utah_teapot_mesh_target(0.72),
            &[MeshRolloutCaseConfig {
                particle_count: 64,
                steps: 4,
                seed: 11,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::TeapotFieldDense3d,
            }],
        )
        .unwrap();

        assert!(!report.passed);
        assert!(report.max_motion_per_step <= 1.0e-6);
        assert!(report.min_final_opacity <= UV_TORUS_INITIAL_OPACITY_LOGIT);
        assert_eq!(report.cases.len(), 1);
    }

    #[test]
    fn mesh_rollout_report_rejects_static_conditionless_local_prior() {
        let config = NpaConfig::growing_3dgs();
        let grid = burn_automata::kernels::HashGridConfig::growing_3dgs();
        let model = NpaModel {
            config: config.clone(),
            weights: NpaWeights::zeros(&config),
        };
        let report = mesh_rollout_report_for_cases(
            &model,
            &grid,
            &uv_torus_mesh_target(0.72),
            &[MeshRolloutCaseConfig {
                particle_count: 64,
                steps: 4,
                seed: 11,
                seed_scale: 0.72,
                seed_mode: ParticleSeed::UniformCircle,
            }],
        )
        .unwrap();

        assert!(!report.passed);
        assert!(report.max_motion_per_step <= 1.0e-6);
        assert_eq!(report.cases.len(), 1);
    }

    #[test]
    fn mesh_target_update_trains_oriented_state_from_neutral_seed() {
        let config = NpaConfig::growing_3dgs();
        let target = uv_torus_mesh_target(0.72);
        let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
        let states = vec![0.0; config.state_dims];
        let update = mesh_field_target_update_for_rows(
            &config, &target, &positions, &states, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, false,
        );
        let base = config.spatial_dims;
        let coordinate_norm =
            (update[base].powi(2) + update[base + 1].powi(2) + update[base + 2].powi(2)).sqrt();
        let normal_update = [
            update[base + UV_TORUS_NORMAL_STATE_OFFSET],
            update[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            update[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let normal_norm =
            (normal_update[0].powi(2) + normal_update[1].powi(2) + normal_update[2].powi(2)).sqrt();
        assert!(coordinate_norm > 1.0e-4);
        assert!(normal_norm > 1.0e-4);
        assert!(update[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET].abs() > 1.0e-4);
    }

    #[test]
    fn mesh_target_update_can_disable_projection_aux_state_targets() {
        let config = NpaConfig::growing_3dgs();
        let target = uv_torus_mesh_target(0.72);
        let positions = vec![[0.1_f32, 0.0, 0.0, 0.0]];
        let states = vec![0.0; config.state_dims];
        let update = mesh_field_target_update_for_rows(
            &config, &target, &positions, &states, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false,
        );
        let base = config.spatial_dims;

        for channel in [
            0,
            1,
            2,
            UV_TORUS_NORMAL_STATE_OFFSET,
            UV_TORUS_NORMAL_STATE_OFFSET + 1,
            UV_TORUS_NORMAL_STATE_OFFSET + 2,
            UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
        ] {
            assert_eq!(update[base + channel], 0.0);
        }
    }

    #[test]
    fn torus_morphogen_supervision_writes_oriented_mesh_channels() {
        let config = NpaConfig::growing_3dgs();
        let rows = 32;
        let batch = torus_morphogen_supervised_batch(&config, rows);
        let input_dims = config.perception_dims();
        let blur_offset = config.state_dims;

        for row in 0..rows {
            let base = row * input_dims;
            let normal = [
                batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET],
                batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
                batch.features[base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
            ];
            let signed_distance = batch.features[base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
            let normal_len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((normal_len - 1.0).abs() < 1.0e-4);
            assert!(signed_distance.is_finite());
            assert!(signed_distance.abs() <= 1.5);
            for channel in 0..config.state_dims {
                assert_eq!(
                    batch.features[base + channel],
                    batch.features[base + blur_offset + channel]
                );
            }
        }
    }
}
