use crate::{AutomataPreset, GaussianDecodeMode, ParticleSeed};
use clap::ValueEnum;
use serde::Serialize;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum PresetArg {
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
pub(crate) enum NeighborModeArg {
    Auto,
    #[value(name = "linked-list", alias = "linked")]
    LinkedList,
    #[value(name = "fixed-buckets", alias = "buckets")]
    FixedBuckets,
    #[value(name = "tiled-fixed-buckets", alias = "tiled-buckets", alias = "tiled")]
    TiledFixedBuckets,
    #[value(name = "sorted-cells", alias = "sorted")]
    SortedCells,
    #[value(
        name = "cooperative-sorted-cells",
        alias = "cooperative-cells",
        alias = "coop"
    )]
    CooperativeSortedCells,
    #[value(
        name = "subgroup-cooperative-sorted-cells",
        alias = "subgroup-cooperative-cells",
        alias = "subgroup-coop"
    )]
    SubgroupCooperativeSortedCells,
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
pub(crate) enum SpatialStrategyArg {
    All,
    #[value(name = "hash-grid", alias = "hashgrid", alias = "grid")]
    HashGrid,
    #[value(name = "tile-blocks", alias = "tiles", alias = "tile")]
    TileBlocks,
    Bvh,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
pub(crate) enum TrainingBatchArg {
    Rollout,
    Features,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum MeshTrainingModeArg {
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
pub(crate) enum MeshTargetArg {
    Torus,
    Teapot,
    Sphere,
    Ellipsoid,
    Cube,
    Cylinder,
    Cone,
    Capsule,
    Pyramid,
    Bicone,
    Dumbbell,
    Cross,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum MeshTargetSetArg {
    #[value(name = "core")]
    Core,
    #[value(name = "primitives", alias = "primitive")]
    Primitives,
    #[value(name = "many", alias = "all")]
    Many,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
pub(crate) enum RenderGradientModeArg {
    Analytic,
    #[value(name = "finite-diff", alias = "finite_difference")]
    FiniteDiff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum RenderGaussianDecodeModeArg {
    #[value(name = "particle-point", alias = "point")]
    ParticlePoint,
    #[value(name = "fixed-sh0", alias = "fixed", alias = "sh0-fixed")]
    GaussianSh0FixedScale,
    #[value(name = "learned-sh0", alias = "learned", alias = "sh0-learned")]
    GaussianSh0LearnedScale,
    #[value(name = "oriented-sh0", alias = "oriented")]
    GaussianSh0Oriented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum RenderTrainingBackendArg {
    #[value(name = "direct-rollout", alias = "direct")]
    DirectRollout,
    #[value(name = "proxy", alias = "supervised-proxy")]
    Proxy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum RenderWeightUpdateModeArg {
    #[value(name = "adapter", alias = "lora", alias = "low-rank")]
    Adapter,
    #[value(name = "full", alias = "full-weights")]
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(crate) enum CoverageUpdateModeArg {
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
pub(crate) enum Growth3dValidationGateArg {
    Strict,
    #[value(name = "catalog-sanity", alias = "catalog")]
    CatalogSanity,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum SeedModeArg {
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
    #[value(name = "growth-3d", alias = "growth", alias = "random-ball-growth-3d")]
    Growth3d,
    #[value(
        name = "substrate-growth-3d",
        alias = "substrate-growth",
        alias = "growth-substrate"
    )]
    SubstrateGrowth3d,
    #[value(
        name = "local-growth-3d",
        alias = "local-growth",
        alias = "growth-local"
    )]
    LocalGrowth3d,
    #[value(
        name = "local-substrate-growth-3d",
        alias = "local-substrate",
        alias = "growth-local-substrate"
    )]
    LocalSubstrateGrowth3d,
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
        name = "torus-local-growth-3d",
        alias = "torus-local-growth",
        alias = "local-growth-torus"
    )]
    TorusLocalGrowth3d,
    #[value(
        name = "teapot-local-growth-3d",
        alias = "teapot-local-growth",
        alias = "local-growth-teapot"
    )]
    TeapotLocalGrowth3d,
    #[value(
        name = "torus-local-substrate-growth-3d",
        alias = "torus-local-substrate",
        alias = "local-substrate-torus"
    )]
    TorusLocalSubstrateGrowth3d,
    #[value(
        name = "teapot-local-substrate-growth-3d",
        alias = "teapot-local-substrate",
        alias = "local-substrate-teapot"
    )]
    TeapotLocalSubstrateGrowth3d,
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
pub(crate) enum BenchGeometryArg {
    Seed,
    Point,
    #[value(name = "micro-cluster", alias = "microcluster", alias = "cluster")]
    MicroCluster,
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

impl From<SeedModeArg> for ParticleSeed {
    fn from(value: SeedModeArg) -> Self {
        match value {
            SeedModeArg::Gaussian => Self::Gaussian,
            SeedModeArg::Uniform => Self::Uniform,
            SeedModeArg::UniformCircle => Self::UniformCircle,
            SeedModeArg::UvTorus3d => Self::UvTorus3d,
            SeedModeArg::UvTorusDense3d => Self::UvTorusDense3d,
            SeedModeArg::Growth3d => Self::Growth3d,
            SeedModeArg::SubstrateGrowth3d => Self::SubstrateGrowth3d,
            SeedModeArg::LocalGrowth3d => Self::LocalGrowth3d,
            SeedModeArg::LocalSubstrateGrowth3d => Self::LocalSubstrateGrowth3d,
            SeedModeArg::TorusFieldDense3d => Self::TorusFieldDense3d,
            SeedModeArg::TeapotFieldDense3d => Self::TeapotFieldDense3d,
            SeedModeArg::TorusGrowth3d => Self::TorusGrowth3d,
            SeedModeArg::TeapotGrowth3d => Self::TeapotGrowth3d,
            SeedModeArg::TorusSubstrateGrowth3d => Self::TorusSubstrateGrowth3d,
            SeedModeArg::TeapotSubstrateGrowth3d => Self::TeapotSubstrateGrowth3d,
            SeedModeArg::TorusLocalGrowth3d => Self::TorusLocalGrowth3d,
            SeedModeArg::TeapotLocalGrowth3d => Self::TeapotLocalGrowth3d,
            SeedModeArg::TorusLocalSubstrateGrowth3d => Self::TorusLocalSubstrateGrowth3d,
            SeedModeArg::TeapotLocalSubstrateGrowth3d => Self::TeapotLocalSubstrateGrowth3d,
            SeedModeArg::TorusMorphogenDense3d => Self::TorusMorphogenDense3d,
            SeedModeArg::TeapotMorphogenDense3d => Self::TeapotMorphogenDense3d,
        }
    }
}

impl From<RenderGaussianDecodeModeArg> for GaussianDecodeMode {
    fn from(value: RenderGaussianDecodeModeArg) -> Self {
        match value {
            RenderGaussianDecodeModeArg::ParticlePoint => Self::ParticlePoint,
            RenderGaussianDecodeModeArg::GaussianSh0FixedScale => Self::GaussianSh0FixedScale,
            RenderGaussianDecodeModeArg::GaussianSh0LearnedScale => Self::GaussianSh0LearnedScale,
            RenderGaussianDecodeModeArg::GaussianSh0Oriented => Self::GaussianSh0Oriented,
        }
    }
}
