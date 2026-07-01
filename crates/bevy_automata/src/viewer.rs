use std::path::Path;

#[cfg(feature = "splatting")]
use bevy::camera::ScalingMode;
#[cfg(feature = "splatting")]
use bevy::camera::primitives::Aabb;
#[cfg(feature = "splatting")]
use bevy::camera::{CameraProjection, Viewport};
use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    image::{CompressedImageFormats, ImageSampler, ImageType},
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::hover::Hovered,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_ui_widgets::{
    Slider, SliderDragState, SliderOrientation, SliderPlugin, SliderRange, SliderStep, SliderThumb,
    SliderValue, TrackClick, ValueChange, slider_self_update,
};
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use burn_automata::gpu::WgpuNeighborMode;
use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutBatchConfig, RolloutConfig,
    RolloutTrace, SgdConfig, SupervisedTarget,
    kernels::HashGridConfig,
    rollout::{growth_3d_material_opacity_channel, uv_torus_outer_radius, uv_torus_position_color},
    rollout_supervised_batch, run_rollout, supervised_backward, supervised_loss,
    supervised_train_step,
    target_geometry::TriangleMeshTarget,
};

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use bevy::render::{
    ExtractSchedule, Render, RenderApp, RenderSystems,
    render_asset::RenderAssets,
    renderer::{RenderDevice, RenderQueue},
};
use bevy::window::PrimaryWindow;
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use bevy_gaussian_splatting::PlanarStorageBindGroup;
#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
use bevy_gaussian_splatting::SphericalHarmonicCoefficients;
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use bevy_gaussian_splatting::gaussian::formats::planar_3d::PlanarStorageGaussian3d;
#[cfg(feature = "splatting")]
use bevy_gaussian_splatting::gaussian::settings::{GaussianColorSpace, RadixSortDepthBits};
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use bevy_gaussian_splatting::render::SortBindGroup;
#[cfg(feature = "splatting")]
use bevy_gaussian_splatting::sort::SortedEntriesHandle;
#[cfg(feature = "splatting")]
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin, Planar,
    PlanarGaussian3d, PlanarGaussian3dHandle,
    sort::{SortMode, SortedEntries},
};
#[cfg(feature = "splatting")]
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, PanOrbitCameraSystemSet};

const DEFAULT_LIZARD_MODEL: &str = "/tmp/burn_automata_lizard.bpk";
const DEFAULT_POLKA_MODEL: &str = "/tmp/burn_automata_polka.bpk";
const FALLBACK_POLKA_MODEL: &str = "/tmp/polka_dotted.bpk";
const BACKWARD_PROBE_PARTICLES: usize = 1024;
const TRAINING_PROBE_PARTICLES: usize = 256;
const TRAINING_INTERVAL_FRAMES: usize = 60;
const LIVE_TRAINING_TARGET: &str = "rollout teacher";
const CATALOG_DOUBLE_CLICK_SECONDS: f64 = 0.35;
const CATALOG_3D_GROWTH_SEED: u64 = 0x51a7_3d;
const AUTOMATA_UI_PANEL_WIDTH: f32 = 540.0;
#[cfg(feature = "splatting")]
const AUTOMATA_MIN_VIEWPORT_WIDTH: u32 = 256;
#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
const GAUSSIAN_SH_C0: f32 = 0.282_094_8;
#[cfg(feature = "splatting")]
const SORTED_ENTRY_MIN_CAPACITY: usize = 16_384;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
enum ModelCatalogKey {
    #[default]
    Lizard,
    Butterfly,
    Rose,
    Turtle,
    Mushroom,
    TropicalFish,
    Sun,
    Ghost,
    Frog,
    Apple,
    Polka,
    Bubbly,
    Clouds,
    Galaxy,
    Hearts,
    Rings,
    Stars,
    Grid,
    Banded,
    Tree,
    Snow,
    Digit0,
    LetterA,
    Growing2d,
    Texture2d,
    Growing3dGs,
    UvTorusMorphogen3d,
    TeapotMorphogen3d,
    PointMnist,
}

#[cfg(test)]
const VISIBLE_MODEL_CATALOG_KEYS: &[ModelCatalogKey] = &[
    ModelCatalogKey::Lizard,
    ModelCatalogKey::Butterfly,
    ModelCatalogKey::Rose,
    ModelCatalogKey::Turtle,
    ModelCatalogKey::Mushroom,
    ModelCatalogKey::TropicalFish,
    ModelCatalogKey::Sun,
    ModelCatalogKey::Ghost,
    ModelCatalogKey::Frog,
    ModelCatalogKey::Apple,
    ModelCatalogKey::Polka,
    ModelCatalogKey::Bubbly,
    ModelCatalogKey::Clouds,
    ModelCatalogKey::Galaxy,
    ModelCatalogKey::Hearts,
    ModelCatalogKey::Rings,
    ModelCatalogKey::Stars,
    ModelCatalogKey::Grid,
    ModelCatalogKey::Banded,
    ModelCatalogKey::Tree,
    ModelCatalogKey::Snow,
    ModelCatalogKey::Digit0,
    ModelCatalogKey::LetterA,
    ModelCatalogKey::Growing2d,
    ModelCatalogKey::Texture2d,
    ModelCatalogKey::Growing3dGs,
    ModelCatalogKey::PointMnist,
];

#[derive(Clone, Copy, Debug)]
enum ModelCatalogSource {
    Preset,
    Bpk {
        primary: &'static str,
        fallback: Option<&'static str>,
    },
}

#[derive(Clone, Copy, Debug)]
struct ModelCatalogEntry {
    key: ModelCatalogKey,
    title: &'static str,
    kind: &'static str,
    detail: &'static str,
    preset: AutomataPreset,
    source: ModelCatalogSource,
    particle_count: usize,
    seed_scale: f32,
    update_prob: f32,
}

const MODEL_CATALOG: &[ModelCatalogEntry] = &[
    ModelCatalogEntry {
        key: ModelCatalogKey::Lizard,
        title: "lizard",
        kind: "imported bpk",
        detail: "SelfOrg NPA rollout",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: DEFAULT_LIZARD_MODEL,
            fallback: Some("/tmp/lizard.bpk"),
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Butterfly,
        title: "butterfly",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/butterfly.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Rose,
        title: "rose",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/rose.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Turtle,
        title: "turtle",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/turtle.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Mushroom,
        title: "mushroom",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/mushroom.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::TropicalFish,
        title: "fish",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/tropical_fish.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Sun,
        title: "sun",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/sun_with_face.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Ghost,
        title: "ghost",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/ghost.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Frog,
        title: "frog",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/frog_face.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Apple,
        title: "apple",
        kind: "web bpk",
        detail: "SelfOrg growing web model",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/growing/red_apple.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Polka,
        title: "polka",
        kind: "imported bpk",
        detail: "texture NPA rollout",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: DEFAULT_POLKA_MODEL,
            fallback: Some(FALLBACK_POLKA_MODEL),
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Bubbly,
        title: "bubbly",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/bubbly_0101.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Clouds,
        title: "clouds",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/clouds.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Galaxy,
        title: "galaxy",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/galaxy.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Hearts,
        title: "hearts",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/hearts.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Rings,
        title: "rings",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/rings.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Stars,
        title: "stars",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/stars.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Grid,
        title: "grid",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/grid_0040.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Banded,
        title: "banded",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/banded_0037.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Tree,
        title: "tree",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/tree.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Snow,
        title: "snow",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/snow.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Digit0,
        title: "digit 0",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/digit_0.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::LetterA,
        title: "A",
        kind: "web bpk",
        detail: "SelfOrg texture web model",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Bpk {
            primary: "/tmp/burn_automata_catalog/texture/letter_a.bpk",
            fallback: None,
        },
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Growing2d,
        title: "growing 2d",
        kind: "seeded preset",
        detail: "local particle growth",
        preset: AutomataPreset::Growing2d,
        source: ModelCatalogSource::Preset,
        particle_count: 4096,
        seed_scale: 0.2,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Texture2d,
        title: "texture 2d",
        kind: "seeded preset",
        detail: "stationary image prior",
        preset: AutomataPreset::Texture2d,
        source: ModelCatalogSource::Preset,
        particle_count: 4096,
        seed_scale: 1.0,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::Growing3dGs,
        title: "growing 3d",
        kind: "seeded preset",
        detail: "3d gaussian field",
        preset: AutomataPreset::Growing3dGs,
        source: ModelCatalogSource::Preset,
        particle_count: 1024,
        seed_scale: 0.35,
        update_prob: 0.5,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::UvTorusMorphogen3d,
        title: "uv torus",
        kind: "validation blocked 3d",
        detail: "hidden: latest local-front torus fails tube support/depth gates",
        preset: AutomataPreset::Growing3dGs,
        source: ModelCatalogSource::Bpk {
            primary: "assets/models/uv_torus_growth_3d.bpk",
            fallback: Some("/tmp/uv_torus_growth_3d.bpk"),
        },
        particle_count: 1024,
        seed_scale: 0.54,
        update_prob: 1.0,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::TeapotMorphogen3d,
        title: "teapot",
        kind: "validation blocked 3d",
        detail: "hidden: seed-varied strict gate exposes held-out fragility",
        preset: AutomataPreset::Growing3dGs,
        source: ModelCatalogSource::Bpk {
            primary: "assets/models/teapot_growth_3d.bpk",
            fallback: Some("/tmp/teapot_growth_3d.bpk"),
        },
        particle_count: 1024,
        seed_scale: 0.72,
        update_prob: 1.0,
    },
    ModelCatalogEntry {
        key: ModelCatalogKey::PointMnist,
        title: "point mnist",
        kind: "seeded preset",
        detail: "sparse point digits",
        preset: AutomataPreset::PointMnist,
        source: ModelCatalogSource::Preset,
        particle_count: 4096,
        seed_scale: 0.55,
        update_prob: 0.5,
    },
];

#[derive(Resource, Clone, Debug)]
pub struct AutomataSettings {
    pub preset: AutomataPreset,
    pub steps_per_frame: usize,
    pub particle_count: usize,
    pub update_prob: f32,
    pub dt: f32,
    pub seed: u64,
    pub seed_scale: f32,
    pub reference_seed_scale: f32,
    pub seed_mode: ParticleSeed,
    pub render_scale: f32,
    pub render_opacity: f32,
    #[cfg(feature = "splatting")]
    pub render_sort_mode_3d: SortMode,
    #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
    pub gpu_neighbor_mode: WgpuNeighborMode,
    pub paused: bool,
    pub visualize_backward: bool,
    pub train_live: bool,
    pub training_learning_rate: f32,
    pub model_path: Option<String>,
    pub revision: u64,
}

impl Default for AutomataSettings {
    fn default() -> Self {
        let preset = AutomataPreset::Growing2d;
        let model_path = std::env::var("BURN_AUTOMATA_MODEL").ok().or_else(|| {
            Path::new(DEFAULT_LIZARD_MODEL)
                .exists()
                .then(|| DEFAULT_LIZARD_MODEL.to_string())
        });
        Self {
            preset,
            steps_per_frame: 1,
            particle_count: 4096,
            update_prob: 0.5,
            dt: 1.0,
            seed: 42,
            seed_scale: NpaConfig::seed_scale_for_preset(preset),
            reference_seed_scale: NpaConfig::seed_scale_for_preset(preset),
            seed_mode: ParticleSeed::UniformCircle,
            render_scale: 0.5,
            render_opacity: 2.0,
            #[cfg(feature = "splatting")]
            render_sort_mode_3d: SortMode::Radix,
            #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
            gpu_neighbor_mode: WgpuNeighborMode::Auto,
            paused: false,
            visualize_backward: false,
            train_live: false,
            training_learning_rate: 1.0e-3,
            model_path,
            revision: 1,
        }
    }
}

impl AutomataSettings {
    fn mark_changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

fn effective_hashgrid(runtime: &AutomataRuntime, settings: &AutomataSettings) -> HashGridConfig {
    runtime.model.config.hashgrid_for_seed_scale(
        &runtime.hashgrid,
        settings.seed_scale,
        settings.reference_seed_scale,
    )
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn automata_render_reinit_key(
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    settings: &AutomataSettings,
    neighbor_mode: WgpuNeighborMode,
) -> AutomataRenderReinitKey {
    AutomataRenderReinitKey {
        particle_count: settings.particle_count,
        seed: settings.seed,
        seed_scale_bits: settings.seed_scale.to_bits(),
        reference_seed_scale_bits: settings.reference_seed_scale.to_bits(),
        seed_mode: settings.seed_mode,
        neighbor_mode,
        model_shape: AutomataRenderModelShapeKey {
            state_dims: model.config.state_dims,
            hidden_dims: model.config.hidden_dims,
            spatial_dims: model.config.spatial_dims,
            perception_dims: model.config.perception_dims(),
            update_dims: model.config.update_dims(),
        },
        hashgrid_shape: AutomataRenderHashGridShapeKey {
            dim: hashgrid.dim,
            grid_size: hashgrid.grid_size,
        },
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn effective_gpu_neighbor_mode(
    runtime: &AutomataRuntime,
    settings: &AutomataSettings,
) -> WgpuNeighborMode {
    if settings.gpu_neighbor_mode != WgpuNeighborMode::Auto {
        return settings.gpu_neighbor_mode;
    }
    if runtime.model.config.spatial_dims == 3 && settings.particle_count <= 2048 {
        WgpuNeighborMode::SortedCells
    } else {
        WgpuNeighborMode::Auto
    }
}

#[derive(Resource, Clone, Debug)]
pub struct AutomataRuntime {
    pub model: NpaModel,
    pub hashgrid: HashGridConfig,
    pub trace: Option<RolloutTrace>,
    pub frame: usize,
    pub status: String,
    pub loaded_model_path: Option<String>,
    pub loaded_preset: Option<AutomataPreset>,
    pub backward_loss: Option<f32>,
    pub backward_grad_norm: Option<f32>,
    pub training_step: usize,
    pub training_loss: Option<f32>,
    pub training_best_loss: Option<f32>,
    pub training_grad_norm: Option<f32>,
    pub training_teacher: Option<NpaModel>,
    pub model_revision: u64,
}

impl Default for AutomataRuntime {
    fn default() -> Self {
        let (config, hashgrid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
        Self {
            model: NpaModel::seeded(config, 42),
            hashgrid,
            trace: None,
            frame: 0,
            status: "ready".to_string(),
            loaded_model_path: None,
            loaded_preset: Some(AutomataPreset::Growing2d),
            backward_loss: None,
            backward_grad_norm: None,
            training_step: 0,
            training_loss: None,
            training_best_loss: None,
            training_grad_norm: None,
            training_teacher: None,
            model_revision: 1,
        }
    }
}

#[cfg(feature = "splatting")]
#[derive(Resource, Clone, Debug, Default)]
struct AutomataCloudState {
    handle: Option<Handle<PlanarGaussian3d>>,
    particle_count: usize,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Clone)]
struct AutomataRenderConfig {
    model: NpaModel,
    hashgrid: HashGridConfig,
    reinit_key: AutomataRenderReinitKey,
    param_key: AutomataRenderParamKey,
    particle_count: usize,
    steps_per_frame: usize,
    update_prob: f32,
    dt: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    neighbor_mode: WgpuNeighborMode,
    paused: bool,
    model_revision: u64,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AutomataRenderParamKey {
    model_revision: u64,
    dt_bits: u32,
    update_prob_bits: u32,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AutomataRenderReinitKey {
    particle_count: usize,
    seed: u64,
    seed_scale_bits: u32,
    reference_seed_scale_bits: u32,
    seed_mode: ParticleSeed,
    neighbor_mode: WgpuNeighborMode,
    model_shape: AutomataRenderModelShapeKey,
    hashgrid_shape: AutomataRenderHashGridShapeKey,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AutomataRenderModelShapeKey {
    state_dims: usize,
    hidden_dims: usize,
    spatial_dims: usize,
    perception_dims: usize,
    update_dims: usize,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AutomataRenderHashGridShapeKey {
    dim: usize,
    grid_size: [usize; 3],
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Default)]
struct AutomataRenderState {
    executor: Option<burn_automata::gpu::WgpuAutomataExecutor>,
    state: Option<burn_automata::gpu::WgpuAutomataState>,
    gaussian_bind_group: Option<burn_automata::gpu::WgpuGaussianBindGroup>,
    reinit_key: AutomataRenderReinitKey,
    param_key: AutomataRenderParamKey,
    model_revision: u64,
    asset_id: Option<AssetId<PlanarGaussian3d>>,
    frame: usize,
    last_error: Option<String>,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Clone, Debug, Default)]
pub struct AutomataRenderDiagnostics {
    pub requested_particle_count: usize,
    pub gaussian_storage_count: usize,
    pub resident_particle_count: usize,
    pub frame: usize,
    pub last_error: Option<String>,
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[derive(Resource, Default)]
struct AutomataRenderBridgeInstalled;

#[derive(Component, Clone, Debug, Default)]
struct StatusLabel;

#[derive(Component, Clone, Debug, Default)]
struct SettingsLabel;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
enum AutomataSliderKind {
    #[default]
    ParticleLog2,
    StepsPerFrame,
    UpdateProb,
    DtLog2,
    RenderScaleLog2,
    RenderOpacityLog2,
    TrainingLearningRateLog2,
}

#[derive(Component, Clone, Copy, Debug, Default)]
struct AutomataSlider(AutomataSliderKind);

#[derive(Component, Clone, Copy, Debug, Default)]
struct AutomataSliderValueLabel(AutomataSliderKind);

#[derive(Component, Clone, Debug, Default)]
struct AutomataSliderThumb;

#[derive(Component, Clone, Debug, Default)]
struct AutomataSliderFill;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RunControlKind {
    #[default]
    Pause,
    Reset,
    Backward,
    Train,
}

#[derive(Component, Clone, Copy, Debug, Default)]
struct RunControlButton(RunControlKind);

#[derive(Component, Clone, Debug, Default)]
struct AutomataUiPanel;

#[derive(Component, Clone, Debug, Default)]
struct AutomataUiRoot;

#[derive(Component, Clone, Debug, Default)]
struct AutomataUiScrollArea;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ModelCatalogCard(ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ModelCatalogThumbnail(ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default)]
struct ModelCatalogTextSize(f32);

#[derive(Component, Clone, Debug, Default)]
struct CatalogPreviewRoot;

#[derive(Component, Clone, Debug, Default)]
struct CatalogPreviewTitle;

#[derive(Component, Clone, Debug, Default)]
struct CatalogPreviewDetail;

#[derive(Component, Clone, Debug, Default)]
struct CatalogPreviewImage;

#[derive(Resource, Clone, Debug, Default)]
struct CatalogPreviewState {
    open: bool,
    key: Option<ModelCatalogKey>,
    last_pressed_key: Option<ModelCatalogKey>,
    last_press_time: f64,
}

#[derive(Resource, Clone, Debug, Default)]
struct CatalogPreviewImageState {
    handle: Option<Handle<Image>>,
    key: Option<ModelCatalogKey>,
}

#[cfg(feature = "splatting")]
#[derive(Component, Clone, Debug, Default)]
struct AutomataGaussianCloud;

#[cfg(feature = "splatting")]
#[derive(Component, Clone, Copy, Debug, Default)]
struct AutomataCloudResizeCooldown(u8);

#[cfg(feature = "splatting")]
#[derive(Component, Clone, Debug, Default)]
struct AutomataCamera2d;

#[cfg(feature = "splatting")]
#[derive(Component, Clone, Debug, Default)]
struct AutomataCamera3d;

#[cfg(feature = "splatting")]
#[derive(Resource, Clone, Debug, Default)]
struct AutomataUiInputCapture {
    active: bool,
}

#[derive(Resource, Clone, Debug)]
struct AutomataUiState {
    visible: bool,
}

impl Default for AutomataUiState {
    fn default() -> Self {
        Self { visible: true }
    }
}

pub struct AutomataViewerPlugin;

impl Plugin for AutomataViewerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AutomataSettings>()
            .init_resource::<AutomataRuntime>()
            .init_resource::<AutomataUiState>()
            .init_resource::<CatalogPreviewState>()
            .init_resource::<CatalogPreviewImageState>();
        #[cfg(feature = "splatting")]
        app.init_resource::<AutomataCloudState>();
        #[cfg(feature = "splatting")]
        app.init_resource::<AutomataUiInputCapture>();
        app.add_plugins(SliderPlugin);

        app.add_systems(
            Startup,
            (scene.spawn(), load_selected_model, setup_gaussian_cloud).chain(),
        )
        .add_systems(
            Update,
            (
                load_selected_model,
                toggle_ui_visibility,
                sync_view_cameras,
                sync_automata_camera_viewports,
                sync_gaussian_cloud_asset,
                restore_resized_gaussian_cloud_visibility,
                sync_gaussian_cloud_settings,
                advance_rollout,
                sync_cpu_trace_to_gaussian_asset,
                scroll_ui_panel,
                assign_catalog_thumbnails,
                assign_catalog_text_fonts,
                update_catalog_preview_modal,
                update_catalog_card_styles,
                sync_slider_values,
                update_slider_visuals,
                update_slider_value_labels,
                update_run_control_button_styles,
                update_status_label,
                update_settings_label,
            )
                .chain(),
        );

        #[cfg(feature = "splatting")]
        app.add_systems(
            PostUpdate,
            (
                gate_camera_controls_while_using_ui.before(PanOrbitCameraSystemSet),
                pan_zoom_2d_camera.after(gate_camera_controls_while_using_ui),
            ),
        );

        #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
        install_automata_render_bridge(app);
    }

    fn finish(&self, _app: &mut App) {
        #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
        install_automata_render_bridge(_app);
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn install_automata_render_bridge(app: &mut App) {
    if app
        .world()
        .contains_resource::<AutomataRenderBridgeInstalled>()
    {
        return;
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<AutomataRenderState>()
        .init_resource::<AutomataRenderDiagnostics>()
        .add_systems(ExtractSchedule, extract_automata_render_config)
        .add_systems(
            Render,
            step_automata_into_gaussians.in_set(RenderSystems::Prepare),
        );
    app.world_mut()
        .insert_resource(AutomataRenderBridgeInstalled);
}

pub fn run() {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.015, 0.018, 0.022)));
    app.add_plugins(DefaultPlugins);
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    #[cfg(feature = "splatting")]
    app.add_plugins((GaussianSplattingPlugin, PanOrbitCameraPlugin));
    app.add_plugins(AutomataViewerPlugin);
    app.run();
}

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub fn automata_executor_from_render_device(
    render_device: &bevy::render::renderer::RenderDevice,
    render_queue: &bevy::render::renderer::RenderQueue,
) -> burn_automata::AutomataResult<burn_automata::gpu::WgpuAutomataExecutor> {
    use std::ops::Deref;

    burn_automata::gpu::WgpuAutomataExecutor::from_device_queue(
        render_device.wgpu_device().clone(),
        render_queue.0.deref().deref().clone(),
    )
}

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub fn gaussian_storage_buffer_refs(
    storage: &PlanarStorageGaussian3d,
) -> burn_automata::gpu::WgpuGaussianBufferRefs<'_> {
    burn_automata::gpu::WgpuGaussianBufferRefs {
        position_visibility: &storage.position_visibility,
        spherical_harmonic: &storage.spherical_harmonic,
        rotation: &storage.rotation,
        scale_opacity: &storage.scale_opacity,
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn write_gaussian_draw_indirect_count(
    render_queue: &RenderQueue,
    storage: &PlanarStorageGaussian3d,
    count: usize,
) {
    let instance_count = count.min(storage.count) as u32;
    let mut bytes = [0u8; 16];
    for (index, value) in [4u32, instance_count, 0u32, 0u32].iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    render_queue.write_buffer(&storage.draw_indirect_buffer, 0, &bytes);
}

fn scene() -> impl SceneList {
    bsn_list![(
        Camera2d
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None
        }
    ), controls_panel(), catalog_preview_modal()]
}

fn controls_panel() -> impl Scene {
    bsn! {
        Node {
            width: px(AUTOMATA_UI_PANEL_WIDTH),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Start,
            overflow: Overflow::scroll_y(),
            scrollbar_width: 8.0,
            padding: px(14),
            row_gap: px(8),
        }
        BackgroundColor(Color::srgba(0.035, 0.04, 0.045, 0.88))
        ScrollPosition(Vec2::ZERO)
        AutomataUiRoot
        AutomataUiPanel
        AutomataUiScrollArea
        Children [
            (
                Text("burn_automata")
                TextColor(Color::srgb(0.92, 0.95, 0.98))
            ),
            (
                Text("status loading")
                template_value(ModelCatalogTextSize(13.0))
                TextColor(Color::srgb(0.84, 0.88, 0.76))
                StatusLabel
            ),
            controls_section("run", run_controls_row()),
            controls_section("training", training_controls_row()),
            controls_section("simulation", simulation_controls_row()),
            controls_section("view", view_controls_row()),
            controls_section("model", model_controls_row()),
            (
                Text("settings loading")
                template_value(ModelCatalogTextSize(13.0))
                TextColor(Color::srgb(0.72, 0.77, 0.82))
                SettingsLabel
            ),
            (
                Node {
                    height: px(1),
                    width: percent(100),
                    margin: UiRect::vertical(px(4)),
                }
                BackgroundColor(Color::srgb(0.20, 0.23, 0.26))
            ),
        ]
    }
}

fn catalog_preview_modal() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            right: px(0),
            top: px(0),
            bottom: px(0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            padding: px(18),
        }
        Visibility::Hidden
        AutomataUiRoot
        CatalogPreviewRoot
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.48))
        Children [(
            Node {
                width: px(430),
                max_width: percent(92),
                height: px(330),
                max_height: percent(84),
                border: px(1),
                border_radius: BorderRadius::all(px(8)),
                padding: px(14),
                flex_direction: FlexDirection::Column,
                row_gap: px(10),
                align_items: AlignItems::Stretch,
            }
            BorderColor::from(Color::srgb(0.26, 0.33, 0.36))
            BackgroundColor(Color::srgb(0.035, 0.042, 0.048))
            Children [
                (
                    Node {
                        width: percent(100),
                        flex_direction: FlexDirection::Row,
                        column_gap: px(8),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::SpaceBetween,
                    }
                    Children [
                        (
                            Text("target")
                            template_value(ModelCatalogTextSize(14.0))
                            TextColor(Color::srgb(0.91, 0.95, 0.96))
                            CatalogPreviewTitle
                        ),
                        catalog_preview_close_button(),
                    ]
                ),
                (
                    Node {
                        width: percent(100),
                        height: px(232),
                        border: px(1),
                        border_radius: BorderRadius::all(px(6)),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        overflow: Overflow::clip(),
                    }
                    BorderColor::from(Color::srgb(0.16, 0.20, 0.22))
                    BackgroundColor(Color::srgb(0.015, 0.019, 0.023))
                    ImageNode::default()
                    CatalogPreviewImage
                ),
                (
                    Text("model target")
                    template_value(ModelCatalogTextSize(12.0))
                    TextColor(Color::srgb(0.63, 0.70, 0.74))
                    CatalogPreviewDetail
                ),
            ]
        )]
    }
}

fn catalog_preview_close_button() -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(64),
            height: px(28),
            border: px(1),
            border_radius: BorderRadius::all(px(6)),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        BorderColor::from(Color::srgb(0.28, 0.34, 0.37))
        BackgroundColor(Color::srgb(0.09, 0.11, 0.125))
        on(|mut event: On<Pointer<Press>>, mut preview: ResMut<CatalogPreviewState>| {
            event.trigger_mut().propagate = false;
            preview.open = false;
        })
        Children [(
            Text("close")
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.91, 0.93))
        )]
    }
}

fn controls_section(label: &'static str, row: impl Scene) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(5),
        }
        Children [
            (
                Text(label)
                TextColor(Color::srgb(0.48, 0.56, 0.62))
            ),
            row,
        ]
    }
}

fn run_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(6),
            align_items: AlignItems::Center,
        }
        Children [
            pause_button(),
            reset_button(),
            backward_button(),
            train_button(),
        ]
    }
}

fn training_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
        }
        Children [
            (
                Text("target rollout teacher | 256 rows | 60f")
                template_value(ModelCatalogTextSize(12.0))
                TextColor(Color::srgb(0.56, 0.64, 0.68))
            ),
            slider_row("train lr", "0.0010", AutomataSliderKind::TrainingLearningRateLog2, log2_slider_value(1.0e-3), -16.0, -4.0, 0.125),
        ]
    }
}

fn simulation_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            slider_row("particles", "4096", AutomataSliderKind::ParticleLog2, particle_slider_value(4096), 6.0, 14.0, 1.0),
            slider_row("steps/frame", "1", AutomataSliderKind::StepsPerFrame, 1.0, 1.0, 8.0, 1.0),
            slider_row("update prob", "0.50", AutomataSliderKind::UpdateProb, 0.5, 0.0, 1.0, 0.05),
            slider_row("dt", "1.000", AutomataSliderKind::DtLog2, 0.0, -5.0, 2.0, 0.125),
        ]
    }
}

fn view_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            slider_row("splat scale", "0.50x", AutomataSliderKind::RenderScaleLog2, -1.0, -5.0, 2.0, 0.0625),
            slider_row("splat opacity", "2.00x", AutomataSliderKind::RenderOpacityLog2, 1.0, -4.0, 1.0, 0.0625),
        ]
    }
}

fn model_controls_row() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(8),
        }
        Children [
            (
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    column_gap: px(6),
                    align_items: AlignItems::Center,
                }
                Children [
                    (
                        Text("catalog")
                        template_value(ModelCatalogTextSize(12.0))
                        TextColor(Color::srgb(0.66, 0.72, 0.76))
                    ),
                    (
                        Text("select a model; view settings persist")
                        template_value(ModelCatalogTextSize(12.0))
                        TextColor(Color::srgb(0.42, 0.49, 0.53))
                    ),
                ]
            ),
            (
                Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(8),
                    row_gap: px(8),
                    align_items: AlignItems::Stretch,
                }
                Children [
                    model_catalog_card(ModelCatalogKey::Lizard),
                    model_catalog_card(ModelCatalogKey::Butterfly),
                    model_catalog_card(ModelCatalogKey::Rose),
                    model_catalog_card(ModelCatalogKey::Turtle),
                    model_catalog_card(ModelCatalogKey::Mushroom),
                    model_catalog_card(ModelCatalogKey::TropicalFish),
                    model_catalog_card(ModelCatalogKey::Sun),
                    model_catalog_card(ModelCatalogKey::Ghost),
                    model_catalog_card(ModelCatalogKey::Frog),
                    model_catalog_card(ModelCatalogKey::Apple),
                    model_catalog_card(ModelCatalogKey::Polka),
                    model_catalog_card(ModelCatalogKey::Bubbly),
                    model_catalog_card(ModelCatalogKey::Clouds),
                    model_catalog_card(ModelCatalogKey::Galaxy),
                    model_catalog_card(ModelCatalogKey::Hearts),
                    model_catalog_card(ModelCatalogKey::Rings),
                    model_catalog_card(ModelCatalogKey::Stars),
                    model_catalog_card(ModelCatalogKey::Grid),
                    model_catalog_card(ModelCatalogKey::Banded),
                    model_catalog_card(ModelCatalogKey::Tree),
                    model_catalog_card(ModelCatalogKey::Snow),
                    model_catalog_card(ModelCatalogKey::Digit0),
                    model_catalog_card(ModelCatalogKey::LetterA),
                    model_catalog_card(ModelCatalogKey::Growing2d),
                    model_catalog_card(ModelCatalogKey::Texture2d),
                    model_catalog_card(ModelCatalogKey::Growing3dGs),
                    model_catalog_card(ModelCatalogKey::PointMnist),
                ]
            ),
        ]
    }
}

fn model_catalog_card(key: ModelCatalogKey) -> impl Scene {
    let entry = catalog_entry(key);
    let title = entry.title;
    bsn! {
        Button
        Node {
            width: px(72),
            height: px(72),
            border: px(1),
            border_radius: BorderRadius::all(px(6)),
            padding: px(6),
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(ModelCatalogCard(key))
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.24, 0.29, 0.32))
        BackgroundColor(Color::srgb(0.072, 0.084, 0.094))
        on(handle_model_catalog_press)
        Children [
            (
                Node {
                    width: px(44),
                    height: px(36),
                    border_radius: BorderRadius::all(px(6)),
                    border: px(1),
                    flex_shrink: 0.0,
                }
                BorderColor::from(Color::srgb(0.15, 0.18, 0.20))
                BackgroundColor(Color::srgb(0.018, 0.022, 0.026))
                ImageNode::default()
                template_value(ModelCatalogThumbnail(key))
            ),
            (
                Node {
                    width: percent(100),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                }
                Children [(
                    Text(title)
                    template_value(ModelCatalogTextSize(8.0))
                    TextColor(Color::srgb(0.86, 0.91, 0.93))
                )]
            ),
        ]
    }
}

fn slider_row(
    label: &'static str,
    value_text: &'static str,
    kind: AutomataSliderKind,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: px(42),
            flex_direction: FlexDirection::Row,
            column_gap: px(10),
            align_items: AlignItems::Center,
        }
        Children [
            (
                Node {
                    width: px(104),
                    align_items: AlignItems::Center,
                }
                Children [(
                    Text(label)
                    TextColor(Color::srgb(0.70, 0.77, 0.82))
                )]
            ),
            slider_widget(kind, value, min, max, step),
            (
                Node {
                    width: px(78),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::End,
                }
                Children [(
                    Text(value_text)
                    TextColor(Color::srgb(0.88, 0.92, 0.89))
                    template_value(AutomataSliderValueLabel(kind))
                )]
            ),
        ]
    }
}

fn slider_widget(
    kind: AutomataSliderKind,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
) -> impl Scene {
    bsn! {
        Node {
            height: px(22),
            flex_grow: 1.0,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Stretch,
        }
        template_value(Hovered::default())
        Slider {
            track_click: TrackClick::Drag,
            orientation: SliderOrientation::Horizontal,
        }
        SliderValue(value)
        SliderRange::new(min, max)
        SliderStep(step)
        template_value(AutomataSlider(kind))
        on(slider_self_update)
        on(handle_slider_value_change)
        Children [
            (
                Node {
                    height: px(6),
                    width: percent(100),
                    border_radius: BorderRadius::all(px(3)),
                    align_self: AlignSelf::Center,
                }
                BackgroundColor(Color::srgb(0.07, 0.085, 0.095))
                Children [(
                    Node {
                        width: percent(0),
                        height: percent(100),
                        border_radius: BorderRadius::all(px(3)),
                    }
                    BackgroundColor(Color::srgb(0.28, 0.56, 0.62))
                    AutomataSliderFill
                )]
            ),
            (
                Node {
                    position_type: PositionType::Absolute,
                    left: px(0),
                    right: px(12),
                    top: px(0),
                    bottom: px(0),
                }
                Children [(
                    SliderThumb
                    AutomataSliderThumb
                    Node {
                        width: px(12),
                        height: px(12),
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        align_self: AlignSelf::Center,
                        border_radius: BorderRadius::MAX,
                    }
                    BackgroundColor(Color::srgb(0.78, 0.88, 0.82))
                )]
            ),
        ]
    }
}

fn control_button(label: &'static str, kind: RunControlKind) -> impl Scene {
    bsn! {
        Button
        Node {
            width: px(104),
            max_width: percent(48),
            flex_grow: 1.0,
            flex_shrink: 1.0,
            height: px(30),
            border: px(1),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        template_value(RunControlButton(kind))
        template_value(Hovered::default())
        BorderColor::from(Color::srgb(0.25, 0.30, 0.34))
        BackgroundColor(Color::srgb(0.10, 0.12, 0.14))
        Children [(
            Text(label)
            template_value(ModelCatalogTextSize(12.0))
            TextColor(Color::srgb(0.86, 0.91, 0.94))
        )]
    }
}

fn pause_button() -> impl Scene {
    bsn! {
        control_button("pause", RunControlKind::Pause)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>| {
            settings.paused = !settings.paused;
        })
    }
}

fn reset_button() -> impl Scene {
    bsn! {
        control_button("reset", RunControlKind::Reset)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.mark_changed();
            runtime.trace = None;
            runtime.frame = 0;
            runtime.status = "reset requested".to_string();
        })
    }
}

fn backward_button() -> impl Scene {
    bsn! {
        control_button("backward", RunControlKind::Backward)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.visualize_backward = !settings.visualize_backward;
            if settings.visualize_backward {
                match probe_trace_for_controls(&runtime, &settings, BACKWARD_PROBE_PARTICLES) {
                    Ok(trace) => {
                        let hashgrid = effective_hashgrid(&runtime, &settings);
                        update_backward_probe(&mut runtime, &trace, &hashgrid);
                        if let (Some(loss), Some(grad_norm)) = (runtime.backward_loss, runtime.backward_grad_norm) {
                            runtime.status = format!("backward probe on | loss {loss:.5} | grad {grad_norm:.5}");
                        }
                    }
                    Err(err) => {
                        settings.visualize_backward = false;
                        runtime.backward_loss = None;
                        runtime.backward_grad_norm = None;
                        runtime.status = format!("backward probe failed: {err}");
                    }
                }
            } else {
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                runtime.status = "backward probe off".to_string();
            }
        })
    }
}

fn train_button() -> impl Scene {
    bsn! {
        control_button("train", RunControlKind::Train)
        on(|_event: On<Pointer<Press>>, mut settings: ResMut<AutomataSettings>, mut runtime: ResMut<AutomataRuntime>| {
            settings.train_live = !settings.train_live;
            if settings.train_live {
                runtime.training_teacher = Some(runtime.model.clone());
                match probe_trace_for_controls(&runtime, &settings, TRAINING_PROBE_PARTICLES) {
                    Ok(trace) => {
                        let hashgrid = effective_hashgrid(&runtime, &settings);
                        update_training_probe(
                            &mut runtime,
                            &trace,
                            &hashgrid,
                            settings.training_learning_rate,
                        );
                        if runtime.training_loss.is_none() {
                            settings.train_live = false;
                            runtime.training_teacher = None;
                        }
                    }
                    Err(err) => {
                        settings.train_live = false;
                        runtime.training_teacher = None;
                        runtime.training_loss = None;
                        runtime.training_grad_norm = None;
                        runtime.status = format!("training probe failed: {err}");
                    }
                }
            } else {
                runtime.training_teacher = None;
                runtime.status = format!("live training paused at step {}", runtime.training_step);
            }
        })
    }
}

fn toggle_ui_visibility(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<AutomataUiState>,
    mut roots: Query<&mut Visibility, With<AutomataUiRoot>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyM) {
        return;
    }

    ui_state.visible = !ui_state.visible;
    let visibility = if ui_state.visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut root in &mut roots {
        *root = visibility;
    }
}

fn handle_model_catalog_press(
    mut event: On<Pointer<Press>>,
    time: Res<Time>,
    cards: Query<&ModelCatalogCard>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
    mut preview: ResMut<CatalogPreviewState>,
) {
    event.trigger_mut().propagate = false;
    let Ok(card) = cards.get(event.entity) else {
        return;
    };
    let now = time.elapsed_secs_f64();
    let double_click = preview.last_pressed_key == Some(card.0)
        && now - preview.last_press_time <= CATALOG_DOUBLE_CLICK_SECONDS;
    preview.last_pressed_key = Some(card.0);
    preview.last_press_time = now;
    select_catalog_entry(card.0, &mut settings, &mut runtime);
    if double_click {
        preview.open = true;
        preview.key = Some(card.0);
        runtime.status = format!("previewing {} target", catalog_entry(card.0).title);
    }
}

fn handle_slider_value_change(
    value_change: On<ValueChange<f32>>,
    sliders: Query<&AutomataSlider>,
    mut settings: ResMut<AutomataSettings>,
    mut runtime: ResMut<AutomataRuntime>,
) {
    let Ok(slider) = sliders.get(value_change.source) else {
        return;
    };
    match slider.0 {
        AutomataSliderKind::ParticleLog2 => {
            if !value_change.is_final {
                return;
            }
            let next = particles_from_slider_value(value_change.value);
            if settings.particle_count != next {
                settings.particle_count = next;
                settings.mark_changed();
                runtime.trace = None;
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::StepsPerFrame => {
            let next = value_change.value.round().clamp(1.0, 8.0) as usize;
            if settings.steps_per_frame != next {
                settings.steps_per_frame = next;
            }
        }
        AutomataSliderKind::UpdateProb => {
            let next = value_change.value.clamp(0.0, 1.0);
            if (settings.update_prob - next).abs() > 1.0e-5 {
                settings.update_prob = next;
                settings.mark_changed();
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::DtLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.03125, 4.0);
            if (settings.dt - next).abs() > 1.0e-5 {
                settings.dt = next;
                settings.mark_changed();
                runtime.frame = 0;
            }
        }
        AutomataSliderKind::RenderScaleLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.125, 4.0);
            if (settings.render_scale - next).abs() > 1.0e-5 {
                settings.render_scale = next;
            }
        }
        AutomataSliderKind::RenderOpacityLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(0.0625, 2.0);
            if (settings.render_opacity - next).abs() > 1.0e-5 {
                settings.render_opacity = next;
            }
        }
        AutomataSliderKind::TrainingLearningRateLog2 => {
            let next = exp2_slider_value(value_change.value).clamp(1.0e-5, 0.1);
            if (settings.training_learning_rate - next).abs() > 1.0e-7 {
                settings.training_learning_rate = next;
            }
        }
    }
}

fn sync_slider_values(
    settings: Res<AutomataSettings>,
    mut commands: Commands,
    sliders: Query<(Entity, &AutomataSlider, Option<&SliderValue>)>,
) {
    if !settings.is_changed() {
        return;
    }
    for (entity, slider, current) in &sliders {
        let next = slider_value_for_settings(slider.0, &settings);
        if current.is_none_or(|value| (value.0 - next).abs() > 1.0e-4) {
            commands.entity(entity).insert(SliderValue(next));
        }
    }
}

#[allow(clippy::type_complexity)]
fn update_slider_visuals(
    sliders: Query<
        (
            Entity,
            &SliderValue,
            &SliderRange,
            &Hovered,
            &SliderDragState,
        ),
        (
            Or<(
                Changed<SliderValue>,
                Changed<Hovered>,
                Changed<SliderDragState>,
            )>,
            With<AutomataSlider>,
        ),
    >,
    children: Query<&Children>,
    mut thumbs: Query<(&mut Node, &mut BackgroundColor), With<AutomataSliderThumb>>,
    mut fills: Query<&mut Node, (With<AutomataSliderFill>, Without<AutomataSliderThumb>)>,
) {
    for (slider_entity, value, range, hovered, drag_state) in &sliders {
        let position = range.thumb_position(value.0).clamp(0.0, 1.0) * 100.0;
        let active = hovered.0 || drag_state.dragging;
        for child in children.iter_descendants(slider_entity) {
            if let Ok((mut node, mut background)) = thumbs.get_mut(child) {
                node.left = percent(position);
                background.0 = if active {
                    Color::srgb(0.92, 0.98, 0.90)
                } else {
                    Color::srgb(0.78, 0.88, 0.82)
                };
            }
            if let Ok(mut node) = fills.get_mut(child) {
                node.width = percent(position);
            }
        }
    }
}

fn update_slider_value_labels(
    settings: Res<AutomataSettings>,
    mut labels: Query<(&AutomataSliderValueLabel, &mut Text)>,
) {
    if !settings.is_changed() {
        return;
    }
    for (label, mut text) in &mut labels {
        text.0 = slider_label(label.0, &settings);
    }
}

fn update_run_control_button_styles(
    settings: Res<AutomataSettings>,
    mut buttons: Query<(
        &RunControlButton,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (button, hovered, mut background, mut border) in &mut buttons {
        let active = run_control_is_active(button.0, &settings);
        background.0 = match (active, hovered.0) {
            (true, true) => Color::srgb(0.19, 0.36, 0.37),
            (true, false) => Color::srgb(0.14, 0.28, 0.29),
            (false, true) => Color::srgb(0.13, 0.15, 0.17),
            (false, false) => Color::srgb(0.10, 0.12, 0.14),
        };
        *border = BorderColor::from(match (active, hovered.0) {
            (true, true) => Color::srgb(0.48, 0.86, 0.78),
            (true, false) => Color::srgb(0.36, 0.70, 0.66),
            (false, true) => Color::srgb(0.36, 0.42, 0.46),
            (false, false) => Color::srgb(0.25, 0.30, 0.34),
        });
    }
}

fn run_control_is_active(kind: RunControlKind, settings: &AutomataSettings) -> bool {
    match kind {
        RunControlKind::Pause => settings.paused,
        RunControlKind::Reset => false,
        RunControlKind::Backward => settings.visualize_backward,
        RunControlKind::Train => settings.train_live,
    }
}

fn assign_catalog_thumbnails(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    thumbnails: Query<(Entity, &ModelCatalogThumbnail), Added<ModelCatalogThumbnail>>,
) {
    for (entity, thumbnail) in &thumbnails {
        let mut image = catalog_thumbnail_image(thumbnail.0);
        image.sampler = ImageSampler::linear();
        let handle = images.add(image);
        commands.entity(entity).insert(ImageNode::new(handle));
    }
}

fn assign_catalog_text_fonts(
    mut commands: Commands,
    text_sizes: Query<(Entity, &ModelCatalogTextSize), Added<ModelCatalogTextSize>>,
) {
    for (entity, text_size) in &text_sizes {
        commands
            .entity(entity)
            .insert(TextFont::from_font_size(text_size.0));
    }
}

#[allow(clippy::too_many_arguments)]
fn update_catalog_preview_modal(
    time: Res<Time>,
    preview: Res<CatalogPreviewState>,
    ui_state: Res<AutomataUiState>,
    mut preview_image_state: ResMut<CatalogPreviewImageState>,
    mut images: ResMut<Assets<Image>>,
    mut roots: Query<&mut Visibility, With<CatalogPreviewRoot>>,
    mut titles: Query<&mut Text, With<CatalogPreviewTitle>>,
    mut details: Query<&mut Text, (With<CatalogPreviewDetail>, Without<CatalogPreviewTitle>)>,
    mut image_nodes: Query<&mut ImageNode, With<CatalogPreviewImage>>,
) {
    let visible = ui_state.visible && preview.open && preview.key.is_some();
    let visibility = if visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut root in &mut roots {
        *root = visibility;
    }
    if !visible {
        return;
    }

    let Some(key) = preview.key else {
        return;
    };
    let entry = catalog_entry(key);
    for mut title in &mut titles {
        title.0 = format!("{} target", entry.title);
    }
    for mut detail in &mut details {
        detail.0 = format!(
            "{} | {} | {}",
            entry.kind, entry.detail, entry.particle_count
        );
    }

    let needs_new_handle = preview_image_state.key != Some(key)
        || preview_image_state
            .handle
            .as_ref()
            .is_none_or(|handle| !images.contains(handle));

    if needs_new_handle {
        let mut image = catalog_preview_image(key, time.elapsed_secs());
        image.sampler = ImageSampler::linear();
        let handle = images.add(image);
        preview_image_state.handle = Some(handle.clone());
        preview_image_state.key = Some(key);
        for mut image_node in &mut image_nodes {
            image_node.image = handle.clone();
        }
        return;
    }

    if matches!(
        key,
        ModelCatalogKey::UvTorusMorphogen3d | ModelCatalogKey::TeapotMorphogen3d
    ) && let Some(handle) = preview_image_state.handle.as_ref()
        && let Some(mut image) = images.get_mut(handle)
    {
        *image = catalog_preview_image(key, time.elapsed_secs());
        image.sampler = ImageSampler::linear();
    }
}

fn update_catalog_card_styles(
    settings: Res<AutomataSettings>,
    mut cards: Query<(
        &ModelCatalogCard,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (card, hovered, mut background, mut border) in &mut cards {
        let selected = catalog_entry_matches_settings(catalog_entry(card.0), &settings);
        background.0 = if selected {
            Color::srgb(0.105, 0.15, 0.15)
        } else if hovered.0 {
            Color::srgb(0.095, 0.112, 0.122)
        } else {
            Color::srgb(0.072, 0.084, 0.094)
        };
        *border = BorderColor::from(if selected {
            Color::srgb(0.34, 0.70, 0.66)
        } else if hovered.0 {
            Color::srgb(0.32, 0.39, 0.42)
        } else {
            Color::srgb(0.24, 0.29, 0.32)
        });
    }
}

fn catalog_entry(key: ModelCatalogKey) -> &'static ModelCatalogEntry {
    MODEL_CATALOG
        .iter()
        .find(|entry| entry.key == key)
        .expect("model catalog key must have an entry")
}

fn compact_particle_count(particle_count: usize) -> String {
    if particle_count >= 1024 {
        format!("{}k", particle_count / 1024)
    } else {
        particle_count.to_string()
    }
}

fn catalog_entry_matches_settings(entry: &ModelCatalogEntry, settings: &AutomataSettings) -> bool {
    match entry.source {
        ModelCatalogSource::Preset => {
            settings.model_path.is_none() && settings.preset == entry.preset
        }
        ModelCatalogSource::Bpk { primary, fallback } => {
            let resolved = resolved_catalog_model_path(entry);
            settings.model_path.as_deref().is_some_and(|path| {
                path == primary || fallback == Some(path) || resolved.as_deref() == Some(path)
            })
        }
    }
}

fn select_catalog_entry(
    key: ModelCatalogKey,
    settings: &mut AutomataSettings,
    runtime: &mut AutomataRuntime,
) {
    let entry = catalog_entry(key);
    let next_model_path = match entry.source {
        ModelCatalogSource::Preset => None,
        ModelCatalogSource::Bpk { .. } => match resolved_catalog_model_path(entry) {
            Some(path) => Some(path),
            None => {
                runtime.status = format!(
                    "missing model file {}",
                    catalog_primary_model_path(entry).unwrap_or("unknown")
                );
                return;
            }
        },
    };

    settings.model_path = next_model_path;
    settings.preset = entry.preset;
    settings.particle_count = entry.particle_count;
    settings.seed_scale = entry.seed_scale;
    settings.reference_seed_scale = entry.seed_scale;
    settings.seed = catalog_seed(entry);
    settings.seed_mode = catalog_seed_mode(entry);
    settings.update_prob = entry.update_prob;
    if let Some(steps_per_frame) = catalog_steps_per_frame(entry) {
        settings.steps_per_frame = steps_per_frame;
    }
    settings.mark_changed();

    runtime.loaded_model_path = None;
    runtime.trace = None;
    runtime.frame = 0;
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(runtime);
    runtime.status = format!(
        "selected {} [{}]: {} | {} particles",
        entry.title,
        entry.kind,
        entry.detail,
        compact_particle_count(entry.particle_count)
    );
    if matches!(entry.source, ModelCatalogSource::Preset) {
        apply_preset(runtime, entry.preset);
        runtime.status = format!(
            "selected {} [{}]: {} | {} particles",
            entry.title,
            entry.kind,
            entry.detail,
            compact_particle_count(entry.particle_count)
        );
    }
}

fn resolved_catalog_model_path(entry: &ModelCatalogEntry) -> Option<String> {
    match entry.source {
        ModelCatalogSource::Preset => None,
        ModelCatalogSource::Bpk { primary, fallback } => {
            resolve_catalog_path(primary).or_else(|| fallback.and_then(resolve_catalog_path))
        }
    }
}

fn catalog_primary_model_path(entry: &ModelCatalogEntry) -> Option<&'static str> {
    match entry.source {
        ModelCatalogSource::Preset => None,
        ModelCatalogSource::Bpk { primary, .. } => Some(primary),
    }
}

fn resolve_catalog_path(path: &'static str) -> Option<String> {
    if Path::new(path).exists() {
        return Some(path.to_string());
    }

    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    workspace_path
        .exists()
        .then(|| workspace_path.to_string_lossy().into_owned())
}

fn catalog_seed_mode(entry: &ModelCatalogEntry) -> ParticleSeed {
    match entry.key {
        ModelCatalogKey::UvTorusMorphogen3d => ParticleSeed::TorusGrowth3d,
        ModelCatalogKey::TeapotMorphogen3d => ParticleSeed::TeapotGrowth3d,
        _ => ParticleSeed::UniformCircle,
    }
}

fn catalog_seed(entry: &ModelCatalogEntry) -> u64 {
    match entry.key {
        ModelCatalogKey::UvTorusMorphogen3d | ModelCatalogKey::TeapotMorphogen3d => {
            CATALOG_3D_GROWTH_SEED
        }
        _ => RolloutConfig::default().seed,
    }
}

fn catalog_steps_per_frame(entry: &ModelCatalogEntry) -> Option<usize> {
    match entry.key {
        ModelCatalogKey::TeapotMorphogen3d => Some(2),
        _ => None,
    }
}

fn catalog_thumbnail_png(key: ModelCatalogKey) -> &'static [u8] {
    match key {
        ModelCatalogKey::Lizard => {
            include_bytes!("../../../assets/catalog_thumbnails/lizard.png")
        }
        ModelCatalogKey::Butterfly => {
            include_bytes!("../../../assets/catalog_thumbnails/butterfly.png")
        }
        ModelCatalogKey::Rose => include_bytes!("../../../assets/catalog_thumbnails/rose.png"),
        ModelCatalogKey::Turtle => {
            include_bytes!("../../../assets/catalog_thumbnails/turtle.png")
        }
        ModelCatalogKey::Mushroom => {
            include_bytes!("../../../assets/catalog_thumbnails/mushroom.png")
        }
        ModelCatalogKey::TropicalFish => {
            include_bytes!("../../../assets/catalog_thumbnails/tropical_fish.png")
        }
        ModelCatalogKey::Sun => {
            include_bytes!("../../../assets/catalog_thumbnails/sun_with_face.png")
        }
        ModelCatalogKey::Ghost => include_bytes!("../../../assets/catalog_thumbnails/ghost.png"),
        ModelCatalogKey::Frog => {
            include_bytes!("../../../assets/catalog_thumbnails/frog_face.png")
        }
        ModelCatalogKey::Apple => {
            include_bytes!("../../../assets/catalog_thumbnails/red_apple.png")
        }
        ModelCatalogKey::Polka => {
            include_bytes!("../../../assets/catalog_thumbnails/polka_dotted_0121.png")
        }
        ModelCatalogKey::Bubbly => {
            include_bytes!("../../../assets/catalog_thumbnails/bubbly_0101.png")
        }
        ModelCatalogKey::Clouds => {
            include_bytes!("../../../assets/catalog_thumbnails/clouds.png")
        }
        ModelCatalogKey::Galaxy => {
            include_bytes!("../../../assets/catalog_thumbnails/galaxy.png")
        }
        ModelCatalogKey::Hearts => {
            include_bytes!("../../../assets/catalog_thumbnails/hearts.png")
        }
        ModelCatalogKey::Rings => include_bytes!("../../../assets/catalog_thumbnails/rings.png"),
        ModelCatalogKey::Stars => include_bytes!("../../../assets/catalog_thumbnails/stars.png"),
        ModelCatalogKey::Grid => {
            include_bytes!("../../../assets/catalog_thumbnails/grid_0040.png")
        }
        ModelCatalogKey::Banded => {
            include_bytes!("../../../assets/catalog_thumbnails/banded_0037.png")
        }
        ModelCatalogKey::Tree => include_bytes!("../../../assets/catalog_thumbnails/tree.png"),
        ModelCatalogKey::Snow => include_bytes!("../../../assets/catalog_thumbnails/snow.png"),
        ModelCatalogKey::Digit0 => {
            include_bytes!("../../../assets/catalog_thumbnails/digit_0.png")
        }
        ModelCatalogKey::LetterA => {
            include_bytes!("../../../assets/catalog_thumbnails/letter_a.png")
        }
        ModelCatalogKey::Growing2d => {
            include_bytes!("../../../assets/catalog_thumbnails/growing_2d.png")
        }
        ModelCatalogKey::Texture2d => {
            include_bytes!("../../../assets/catalog_thumbnails/texture_2d.png")
        }
        ModelCatalogKey::Growing3dGs => {
            include_bytes!("../../../assets/catalog_thumbnails/growing_3d_gs.png")
        }
        ModelCatalogKey::UvTorusMorphogen3d => {
            include_bytes!("../../../assets/catalog_thumbnails/uv_torus_morphogen_3d.png")
        }
        ModelCatalogKey::TeapotMorphogen3d => {
            include_bytes!("../../../assets/catalog_thumbnails/teapot_morphogen_3d.png")
        }
        ModelCatalogKey::PointMnist => {
            include_bytes!("../../../assets/catalog_thumbnails/point_mnist.png")
        }
    }
}

fn catalog_thumbnail_image(key: ModelCatalogKey) -> Image {
    Image::from_buffer(
        catalog_thumbnail_png(key),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::linear(),
        RenderAssetUsages::default(),
    )
    .unwrap_or_else(|_| procedural_catalog_thumbnail_image(key))
}

fn catalog_preview_image(key: ModelCatalogKey, seconds: f32) -> Image {
    match key {
        ModelCatalogKey::UvTorusMorphogen3d => procedural_uv_torus_preview_image(seconds),
        ModelCatalogKey::TeapotMorphogen3d => procedural_teapot_preview_image(seconds),
        _ => catalog_thumbnail_image(key),
    }
}

fn procedural_catalog_thumbnail_image(key: ModelCatalogKey) -> Image {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 72;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    match key {
        ModelCatalogKey::Lizard => draw_lizard_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Polka => draw_polka_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Butterfly
        | ModelCatalogKey::Rose
        | ModelCatalogKey::Turtle
        | ModelCatalogKey::Mushroom
        | ModelCatalogKey::TropicalFish
        | ModelCatalogKey::Sun
        | ModelCatalogKey::Ghost
        | ModelCatalogKey::Frog
        | ModelCatalogKey::Apple => draw_growing_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Bubbly
        | ModelCatalogKey::Clouds
        | ModelCatalogKey::Galaxy
        | ModelCatalogKey::Hearts
        | ModelCatalogKey::Rings
        | ModelCatalogKey::Stars
        | ModelCatalogKey::Grid
        | ModelCatalogKey::Banded
        | ModelCatalogKey::Tree
        | ModelCatalogKey::Snow
        | ModelCatalogKey::Digit0
        | ModelCatalogKey::LetterA => draw_texture_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Growing2d => draw_growing_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Texture2d => draw_texture_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Growing3dGs => draw_growing_3d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::UvTorusMorphogen3d => {
            draw_uv_torus_morphogen_thumbnail(&mut data, WIDTH, HEIGHT)
        }
        ModelCatalogKey::TeapotMorphogen3d => {
            draw_teapot_morphogen_thumbnail(&mut data, WIDTH, HEIGHT)
        }
        ModelCatalogKey::PointMnist => draw_point_mnist_thumbnail(&mut data, WIDTH, HEIGHT),
    }
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn procedural_uv_torus_preview_image(seconds: f32) -> Image {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 232;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    draw_uv_torus_target_preview(&mut data, WIDTH, HEIGHT, seconds * 0.45, 0.72);
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn procedural_teapot_preview_image(seconds: f32) -> Image {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 232;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    draw_teapot_target_preview(&mut data, WIDTH, HEIGHT, seconds * 0.35);
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn fill_thumbnail_background(data: &mut [u8], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let vignette = ((x as f32 / width as f32 - 0.5).abs()
                + (y as f32 / height as f32 - 0.5).abs())
                * 0.045;
            let grid = if (x / 12 + y / 12) % 2 == 0 {
                0.006
            } else {
                0.0
            };
            write_pixel(
                data,
                width,
                x as i32,
                y as i32,
                [
                    to_u8(0.020 + grid - vignette),
                    to_u8(0.026 + grid - vignette),
                    to_u8(0.032 + grid - vignette),
                    255,
                ],
            );
        }
    }
}

fn draw_lizard_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for i in 0..64 {
        let t = i as f32 / 63.0;
        let x = 14.0 + t * 56.0;
        let y = 38.0 + (t * 5.0).sin() * 4.0;
        let r = 5.2 - (t - 0.45).abs() * 4.0;
        draw_disc(data, width, height, x, y, r.max(2.0), [130, 214, 144, 210]);
    }
    for i in 0..20 {
        let t = i as f32 / 19.0;
        draw_disc(
            data,
            width,
            height,
            68.0 + t * 11.0,
            34.0 - t * 4.0,
            2.3 - t * 0.6,
            [155, 231, 164, 220],
        );
    }
    for &(x0, y0, x1, y1) in &[
        (32.0, 40.0, 23.0, 51.0),
        (39.0, 38.0, 31.0, 25.0),
        (51.0, 38.0, 60.0, 25.0),
        (56.0, 40.0, 67.0, 51.0),
    ] {
        draw_line_dots(data, width, height, x0, y0, x1, y1, 5, [95, 185, 124, 185]);
    }
    draw_disc(data, width, height, 67.0, 32.0, 1.5, [245, 252, 214, 240]);
}

fn draw_polka_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for gy in 0..5 {
        for gx in 0..7 {
            let offset = if gy % 2 == 0 { 0.0 } else { 7.0 };
            let x = 12.0 + gx as f32 * 14.0 + offset;
            let y = 9.0 + gy as f32 * 14.0;
            let r = if (gx + gy) % 3 == 0 { 4.8 } else { 3.4 };
            let color = if (gx + gy) % 2 == 0 {
                [224, 116, 116, 230]
            } else {
                [236, 221, 140, 220]
            };
            draw_disc(data, width, height, x, y, r, color);
        }
    }
}

fn draw_growing_2d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for i in 0..180 {
        let t = i as f32;
        let angle = t * 2.3999631;
        let radius = t.sqrt() * 2.1;
        let x = 48.0 + angle.cos() * radius;
        let y = 36.0 + angle.sin() * radius;
        let alpha = (225.0 - radius * 4.0).clamp(60.0, 225.0) as u8;
        draw_disc(data, width, height, x, y, 1.3, [102, 196, 210, alpha]);
    }
    draw_disc(data, width, height, 48.0, 36.0, 5.0, [228, 245, 197, 210]);
}

fn draw_texture_2d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for y in (8..height - 8).step_by(7) {
        for x in (8..width - 8).step_by(7) {
            let wave = ((x as f32 * 0.18).sin() + (y as f32 * 0.24).cos()) * 0.5;
            let color = if wave > 0.0 {
                [82, 175, 214, 190]
            } else {
                [212, 182, 108, 185]
            };
            draw_disc(
                data,
                width,
                height,
                x as f32 + wave * 2.0,
                y as f32 - wave * 1.5,
                1.8,
                color,
            );
        }
    }
}

fn draw_growing_3d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    let points = [
        (46.0, 35.0, 8.0, [108, 190, 220, 180]),
        (34.0, 30.0, 4.6, [218, 203, 122, 170]),
        (60.0, 31.0, 5.4, [125, 223, 158, 190]),
        (53.0, 47.0, 4.4, [218, 128, 128, 175]),
        (42.0, 50.0, 3.8, [180, 160, 230, 160]),
        (69.0, 43.0, 3.2, [106, 176, 224, 155]),
        (26.0, 43.0, 3.0, [160, 220, 164, 155]),
    ];
    for (x, y, radius, color) in points {
        draw_disc(data, width, height, x, y, radius, color);
        draw_disc(
            data,
            width,
            height,
            x - radius * 0.25,
            y - radius * 0.25,
            radius * 0.35,
            [250, 255, 240, 92],
        );
    }
}

fn draw_uv_torus_morphogen_thumbnail(data: &mut [u8], width: u32, height: u32) {
    draw_uv_torus_target_preview(data, width, height, 1.18, 0.72);
}

fn draw_teapot_morphogen_thumbnail(data: &mut [u8], width: u32, height: u32) {
    draw_teapot_target_preview(data, width, height, 0.78);
}

fn draw_teapot_target_preview(data: &mut [u8], width: u32, height: u32, yaw: f32) {
    if let Ok(target) = TriangleMeshTarget::utah_teapot(0.72) {
        draw_mesh_target_preview(data, width, height, yaw, &target);
        return;
    }

    let scale = width.min(height) as f32 / 96.0;
    let cx = width as f32 * 0.50;
    let cy = height as f32 * 0.54;
    let yaw_shift = yaw.sin() * 4.0 * scale;

    draw_teapot_handle(data, width, height, cx - 22.0 * scale, cy, scale, yaw_shift);
    draw_teapot_spout(
        data,
        width,
        height,
        cx + 31.0 * scale,
        cy - 6.0 * scale,
        scale,
        yaw_shift,
    );
    draw_ellipse(
        data,
        width,
        height,
        cx,
        cy,
        28.0 * scale,
        20.0 * scale,
        [86, 177, 216, 232],
    );
    draw_ellipse(
        data,
        width,
        height,
        cx - 5.0 * scale,
        cy - 6.0 * scale,
        15.0 * scale,
        9.0 * scale,
        [120, 212, 178, 150],
    );
    draw_ellipse(
        data,
        width,
        height,
        cx,
        cy - 24.0 * scale,
        17.5 * scale,
        6.0 * scale,
        [217, 198, 117, 224],
    );
    draw_disc(
        data,
        width,
        height,
        cx,
        cy - 32.0 * scale,
        4.8 * scale,
        [228, 132, 124, 228],
    );
    draw_disc(
        data,
        width,
        height,
        cx - 10.0 * scale,
        cy - 8.0 * scale,
        4.0 * scale,
        [245, 255, 226, 82],
    );
}

fn draw_mesh_target_preview(
    data: &mut [u8],
    width: u32,
    height: u32,
    yaw: f32,
    target: &TriangleMeshTarget,
) {
    let pitch = -0.52_f32;
    let projected = target
        .vertices
        .iter()
        .map(|position| teapot_preview_project(*position, yaw, pitch))
        .collect::<Vec<_>>();
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for point in &projected {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }

    let extent_x = (max_x - min_x).max(1.0e-4);
    let extent_y = (max_y - min_y).max(1.0e-4);
    let pixel_scale = (width as f32 * 0.82 / extent_x).min(height as f32 * 0.76 / extent_y);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let screen = projected
        .iter()
        .map(|point| {
            [
                width as f32 * 0.50 + (point[0] - center_x) * pixel_scale,
                height as f32 * 0.54 - (point[1] - center_y) * pixel_scale,
                point[2],
            ]
        })
        .collect::<Vec<_>>();

    let mut z_buffer = vec![f32::MIN; (width * height) as usize];
    let colors = target.colors.as_deref();
    let light = normalize3([0.32, -0.48, 0.82]);
    for (face_index, face) in target.faces.iter().enumerate() {
        let a = screen[face[0] as usize];
        let b = screen[face[1] as usize];
        let c = screen[face[2] as usize];
        let area = edge2(a, b, c);
        if area.abs() <= 1.0e-4 {
            continue;
        }
        let normal = teapot_preview_rotate(target.face_normals[face_index], yaw, pitch);
        let diffuse = dot3(normalize3(normal), light).max(0.0);
        let shade = 0.34 + diffuse * 0.76;
        let min_px = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i32;
        let max_px = a[0].max(b[0]).max(c[0]).ceil().min(width as f32 - 1.0) as i32;
        let min_py = a[1].min(b[1]).min(c[1]).floor().max(0.0) as i32;
        let max_py = a[1].max(b[1]).max(c[1]).ceil().min(height as f32 - 1.0) as i32;
        for py in min_py..=max_py {
            for px in min_px..=max_px {
                let point = [px as f32 + 0.5, py as f32 + 0.5, 0.0];
                let w0 = edge2(b, c, point) / area;
                let w1 = edge2(c, a, point) / area;
                let w2 = edge2(a, b, point) / area;
                if w0 < -1.0e-4 || w1 < -1.0e-4 || w2 < -1.0e-4 {
                    continue;
                }
                let depth = a[2] * w0 + b[2] * w1 + c[2] * w2;
                let z_index = (py as u32 * width + px as u32) as usize;
                if depth <= z_buffer[z_index] {
                    continue;
                }
                z_buffer[z_index] = depth;
                let rgb = if let Some(colors) = colors {
                    let ca = colors[face[0] as usize];
                    let cb = colors[face[1] as usize];
                    let cc = colors[face[2] as usize];
                    [
                        ca[0] * w0 + cb[0] * w1 + cc[0] * w2,
                        ca[1] * w0 + cb[1] * w1 + cc[1] * w2,
                        ca[2] * w0 + cb[2] * w1 + cc[2] * w2,
                    ]
                } else {
                    [0.50, 0.78, 0.86]
                };
                write_pixel(
                    data,
                    width,
                    px,
                    py,
                    [
                        to_u8(rgb[0] * shade),
                        to_u8(rgb[1] * shade),
                        to_u8(rgb[2] * shade),
                        255,
                    ],
                );
            }
        }
    }
}

fn teapot_preview_project(position: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let rotated = teapot_preview_rotate(position, yaw, pitch);
    [rotated[0], rotated[2], -rotated[1]]
}

fn teapot_preview_rotate(position: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (yaw_sin, yaw_cos) = yaw.sin_cos();
    let x = position[0] * yaw_cos - position[1] * yaw_sin;
    let y = position[0] * yaw_sin + position[1] * yaw_cos;
    let z = position[2];
    let (pitch_sin, pitch_cos) = pitch.sin_cos();
    [
        x,
        y * pitch_cos - z * pitch_sin,
        y * pitch_sin + z * pitch_cos,
    ]
}

fn edge2(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn draw_teapot_spout(
    data: &mut [u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    scale: f32,
    yaw_shift: f32,
) {
    for i in 0..18 {
        let t = i as f32 / 17.0;
        let radius = (5.0 - t * 2.8) * scale;
        draw_disc(
            data,
            width,
            height,
            x + t * (30.0 * scale + yaw_shift * 0.8),
            y - t * 9.0 * scale,
            radius.max(1.2 * scale),
            [88, 183, 214, 210],
        );
    }
}

fn draw_teapot_handle(
    data: &mut [u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    scale: f32,
    yaw_shift: f32,
) {
    for i in 0..42 {
        let t = i as f32 / 41.0;
        let angle = std::f32::consts::PI - 1.15 + 2.30 * t;
        let px = x + angle.cos() * (18.0 * scale + yaw_shift.abs() * 0.4) - yaw_shift;
        let py = y + angle.sin() * 21.0 * scale;
        draw_disc(
            data,
            width,
            height,
            px,
            py,
            3.6 * scale,
            [106, 197, 202, 190],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_ellipse(
    data: &mut [u8],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: [u8; 4],
) {
    let min_x = (cx - rx - 1.0).floor() as i32;
    let max_x = (cx + rx + 1.0).ceil() as i32;
    let min_y = (cy - ry - 1.0).floor() as i32;
    let max_y = (cy + ry + 1.0).ceil() as i32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f32 + 0.5 - cx) / rx.max(1.0);
            let dy = (y as f32 + 0.5 - cy) / ry.max(1.0);
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (1.0 - distance).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let mut blended = color;
                blended[3] = (color[3] as f32 * coverage.sqrt()) as u8;
                blend_pixel(data, width, height, x, y, blended);
            }
        }
    }
}

fn draw_uv_torus_target_preview(data: &mut [u8], width: u32, height: u32, yaw: f32, scale: f32) {
    let outer_radius = uv_torus_outer_radius(scale);
    let view_radius = outer_radius * 1.32;
    let aspect = width as f32 / height.max(1) as f32;
    let (yaw_sin, yaw_cos) = yaw.sin_cos();
    let pitch = -0.46_f32;
    let (pitch_sin, pitch_cos) = pitch.sin_cos();

    for py in 0..height {
        for px in 0..width {
            let sx = (((px as f32 + 0.5) / width as f32) - 0.5) * 2.0 * view_radius * aspect;
            let sy = (0.52 - ((py as f32 + 0.5) / height as f32)) * 2.0 * view_radius;
            let origin = torus_view_to_local(
                [sx, sy, outer_radius * 3.25],
                yaw_sin,
                yaw_cos,
                pitch_sin,
                pitch_cos,
            );
            let direction =
                torus_view_to_local([0.0, 0.0, -1.0], yaw_sin, yaw_cos, pitch_sin, pitch_cos);
            if let Some(position) = raymarch_torus(origin, direction, scale) {
                let normal = torus_sdf_normal(position, scale);
                let light_dir = normalize3([0.42, -0.35, 0.84]);
                let diffuse = dot3(normal, light_dir).max(0.0);
                let color = uv_torus_position_color(position, scale);
                let ambient = 0.34;
                let light = ambient + diffuse * 0.68;
                write_pixel(
                    data,
                    width,
                    px as i32,
                    py as i32,
                    [
                        to_u8(color[0] * light),
                        to_u8(color[1] * light),
                        to_u8(color[2] * light),
                        255,
                    ],
                );
            }
        }
    }
}

fn torus_view_to_local(
    value: [f32; 3],
    yaw_sin: f32,
    yaw_cos: f32,
    pitch_sin: f32,
    pitch_cos: f32,
) -> [f32; 3] {
    let x1 = value[0];
    let y1 = value[1] * pitch_cos + value[2] * pitch_sin;
    let z = -value[1] * pitch_sin + value[2] * pitch_cos;
    [x1 * yaw_cos + y1 * yaw_sin, -x1 * yaw_sin + y1 * yaw_cos, z]
}

fn raymarch_torus(origin: [f32; 3], direction: [f32; 3], scale: f32) -> Option<[f32; 3]> {
    let direction = normalize3(direction);
    let outer_radius = uv_torus_outer_radius(scale);
    let max_distance = outer_radius * 7.0;
    let hit_epsilon = outer_radius * 0.0035;
    let min_step = outer_radius * 0.0015;
    let mut distance = 0.0_f32;
    for _ in 0..112 {
        let position = [
            origin[0] + direction[0] * distance,
            origin[1] + direction[1] * distance,
            origin[2] + direction[2] * distance,
        ];
        let sdf = uv_torus_sdf(position, scale);
        if sdf.abs() <= hit_epsilon {
            return Some(position);
        }
        distance += sdf.max(min_step);
        if distance > max_distance {
            return None;
        }
    }
    None
}

fn uv_torus_sdf(position: [f32; 3], scale: f32) -> f32 {
    let major = scale.max(1.0e-4);
    let minor = major * burn_automata::rollout::UV_TORUS_MINOR_RATIO;
    let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
    ((radial - major).powi(2) + position[2].powi(2)).sqrt() - minor
}

fn torus_sdf_normal(position: [f32; 3], scale: f32) -> [f32; 3] {
    let eps = uv_torus_outer_radius(scale) * 0.0015;
    normalize3([
        uv_torus_sdf([position[0] + eps, position[1], position[2]], scale)
            - uv_torus_sdf([position[0] - eps, position[1], position[2]], scale),
        uv_torus_sdf([position[0], position[1] + eps, position[2]], scale)
            - uv_torus_sdf([position[0], position[1] - eps, position[2]], scale),
        uv_torus_sdf([position[0], position[1], position[2] + eps], scale)
            - uv_torus_sdf([position[0], position[1], position[2] - eps], scale),
    ])
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 {
        [0.0, 0.0, 1.0]
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn draw_point_mnist_thumbnail(data: &mut [u8], width: u32, height: u32) {
    let segments = [
        (34.0, 18.0, 63.0, 18.0),
        (63.0, 18.0, 62.0, 35.0),
        (41.0, 35.0, 62.0, 35.0),
        (62.0, 35.0, 63.0, 53.0),
        (34.0, 53.0, 63.0, 53.0),
    ];
    for (x0, y0, x1, y1) in segments {
        draw_line_dots(
            data,
            width,
            height,
            x0,
            y0,
            x1,
            y1,
            10,
            [230, 235, 173, 220],
        );
    }
    for i in 0..36 {
        let t = i as f32 / 35.0;
        let x = 30.0 + (t * 38.0).sin() * 3.0;
        let y = 16.0 + t * 39.0;
        draw_disc(data, width, height, x, y, 1.0, [94, 179, 210, 140]);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line_dots(
    data: &mut [u8],
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    dots: usize,
    color: [u8; 4],
) {
    for i in 0..dots {
        let t = if dots <= 1 {
            0.0
        } else {
            i as f32 / (dots - 1) as f32
        };
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        draw_disc(data, width, height, x, y, 1.7, color);
    }
}

fn draw_disc(
    data: &mut [u8],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 4],
) {
    let min_x = (cx - radius - 1.0).floor() as i32;
    let max_x = (cx + radius + 1.0).ceil() as i32;
    let min_y = (cy - radius - 1.0).floor() as i32;
    let max_y = (cy + radius + 1.0).ceil() as i32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let mut blended = color;
                blended[3] = (color[3] as f32 * coverage) as u8;
                blend_pixel(data, width, height, x, y, blended);
            }
        }
    }
}

fn write_pixel(data: &mut [u8], width: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as u32;
    let y = y as u32;
    let index = ((y * width + x) * 4) as usize;
    if index + 3 >= data.len() {
        return;
    }
    data[index..index + 4].copy_from_slice(&color);
}

fn blend_pixel(data: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
        return;
    }
    let index = (((y as u32) * width + x as u32) * 4) as usize;
    let alpha = color[3] as f32 / 255.0;
    let inverse = 1.0 - alpha;
    data[index] = (color[0] as f32 * alpha + data[index] as f32 * inverse) as u8;
    data[index + 1] = (color[1] as f32 * alpha + data[index + 1] as f32 * inverse) as u8;
    data[index + 2] = (color[2] as f32 * alpha + data[index + 2] as f32 * inverse) as u8;
    data[index + 3] = 255;
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(feature = "splatting")]
#[allow(clippy::type_complexity)]
fn sync_view_cameras(
    runtime: Res<AutomataRuntime>,
    mut camera_2d: Query<&mut Camera, (With<AutomataCamera2d>, Without<AutomataCamera3d>)>,
    mut camera_3d: Query<
        (&mut Camera, &mut PanOrbitCamera),
        (With<AutomataCamera3d>, Without<AutomataCamera2d>),
    >,
) {
    let use_2d = runtime.model.config.spatial_dims == 2;

    for mut camera in &mut camera_2d {
        camera.is_active = use_2d;
    }

    for (mut camera, mut pan_orbit) in &mut camera_3d {
        camera.is_active = !use_2d;
        pan_orbit.enabled = !use_2d;
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::type_complexity)]
fn sync_automata_camera_viewports(
    ui_state: Res<AutomataUiState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera, Or<(With<AutomataCamera2d>, With<AutomataCamera3d>)>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let viewport = automata_camera_viewport(
        window.physical_size(),
        window.scale_factor(),
        ui_state.visible,
    );
    for mut camera in &mut cameras {
        camera.viewport = viewport.clone();
    }
}

#[cfg(not(feature = "splatting"))]
fn sync_automata_camera_viewports() {}

#[cfg(feature = "splatting")]
fn automata_camera_viewport(
    physical_size: UVec2,
    scale_factor: f32,
    ui_visible: bool,
) -> Option<Viewport> {
    if !ui_visible || physical_size.x <= AUTOMATA_MIN_VIEWPORT_WIDTH {
        return None;
    }

    let panel_physical_width = (AUTOMATA_UI_PANEL_WIDTH * scale_factor.max(1.0e-4)).round() as u32;
    let right_width = physical_size.x.saturating_sub(panel_physical_width);
    if right_width < AUTOMATA_MIN_VIEWPORT_WIDTH {
        return None;
    }

    Some(Viewport {
        physical_position: UVec2::new(panel_physical_width, 0),
        physical_size: UVec2::new(right_width, physical_size.y.max(1)),
        depth: 0.0..1.0,
    })
}

fn scroll_ui_panel(
    ui_state: Res<AutomataUiState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    mut scroll_areas: Query<&mut ScrollPosition, With<AutomataUiScrollArea>>,
) {
    if !ui_state.visible {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    if !panels
        .iter()
        .any(|(node, transform)| node.contains_point(*transform, cursor))
    {
        return;
    }

    let mut scroll_delta = 0.0;
    for event in mouse_wheel.read() {
        let unit_scale = match event.unit {
            MouseScrollUnit::Line => 48.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        scroll_delta += event.y * unit_scale;
    }
    if scroll_delta == 0.0 {
        return;
    }

    for mut scroll_position in &mut scroll_areas {
        scroll_position.0.y = (scroll_position.0.y - scroll_delta).max(0.0);
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::too_many_arguments)]
fn gate_camera_controls_while_using_ui(
    ui_state: Res<AutomataUiState>,
    preview: Res<CatalogPreviewState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut capture: ResMut<AutomataUiInputCapture>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    sliders: Query<&SliderDragState, With<AutomataSlider>>,
    mut cameras: Query<(&mut PanOrbitCamera, &Camera)>,
) {
    let dragging_slider = ui_state.visible && sliders.iter().any(|state| state.dragging);
    let cursor_over_panel = ui_state.visible
        && windows
            .single()
            .ok()
            .and_then(Window::cursor_position)
            .is_some_and(|cursor| {
                panels
                    .iter()
                    .any(|(node, transform)| node.contains_point(*transform, cursor))
            });
    let mouse_just_pressed = mouse_buttons.just_pressed(MouseButton::Left)
        || mouse_buttons.just_pressed(MouseButton::Middle)
        || mouse_buttons.just_pressed(MouseButton::Right);
    let mouse_pressed = mouse_buttons.pressed(MouseButton::Left)
        || mouse_buttons.pressed(MouseButton::Middle)
        || mouse_buttons.pressed(MouseButton::Right);

    if mouse_just_pressed && cursor_over_panel {
        capture.active = true;
    } else if !mouse_pressed {
        capture.active = false;
    }

    let ui_owns_pointer = capture.active
        || dragging_slider
        || cursor_over_panel
        || (ui_state.visible && preview.open);

    for (mut pan_orbit, camera) in &mut cameras {
        pan_orbit.enabled = camera.is_active && !ui_owns_pointer;
    }
}

#[cfg(feature = "splatting")]
#[allow(clippy::too_many_arguments)]
fn pan_zoom_2d_camera(
    ui_state: Res<AutomataUiState>,
    preview: Res<CatalogPreviewState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    capture: Res<AutomataUiInputCapture>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<AutomataUiPanel>>,
    sliders: Query<&SliderDragState, With<AutomataSlider>>,
    mut last_cursor: Local<Option<Vec2>>,
    mut cameras: Query<(&Camera, &mut Projection, &mut Transform), With<AutomataCamera2d>>,
) {
    let Ok(window) = windows.single() else {
        *last_cursor = None;
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        *last_cursor = None;
        return;
    };

    let dragging_slider = ui_state.visible && sliders.iter().any(|state| state.dragging);
    let cursor_over_panel = ui_state.visible
        && panels
            .iter()
            .any(|(node, transform)| node.contains_point(*transform, cursor));
    let ui_owns_pointer = capture.active
        || dragging_slider
        || cursor_over_panel
        || (ui_state.visible && preview.open);

    let current_cursor = Vec2::new(cursor.x, -cursor.y);
    let mut wheel_delta = 0.0;
    for event in mouse_wheel.read() {
        let unit_scale = match event.unit {
            MouseScrollUnit::Line => 100.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        wheel_delta += event.y * unit_scale * 0.001;
    }

    let mut active_camera = false;
    for (camera, mut projection, mut transform) in &mut cameras {
        if !camera.is_active {
            continue;
        }
        active_camera = true;
        if ui_owns_pointer {
            continue;
        }

        let Projection::Orthographic(projection) = &mut *projection else {
            continue;
        };
        let view_size = camera.logical_viewport_size().unwrap_or(window.size());
        if view_size.x <= 0.0 || view_size.y <= 0.0 {
            continue;
        }
        projection.update(view_size.x, view_size.y);

        if wheel_delta != 0.0 {
            let previous_scale = projection.scale;
            let previous_area = projection.area;
            projection.scale = (projection.scale * (1.0 - wheel_delta)).clamp(0.05, 16.0);
            projection.update(view_size.x, view_size.y);

            let view_origin = camera
                .logical_viewport_rect()
                .map(|viewport| viewport.min)
                .unwrap_or(Vec2::ZERO);
            let cursor_ndc = ((cursor - view_origin) / view_size) * 2.0 - Vec2::ONE;
            let cursor_view = Vec2::new(cursor_ndc.x, -cursor_ndc.y);
            let previous_size = previous_area.size() / previous_scale;
            let cursor_world =
                transform.translation.truncate() + cursor_view * previous_size * previous_scale;
            let proposed_position = cursor_world - cursor_view * previous_size * projection.scale;
            transform.translation.x = proposed_position.x;
            transform.translation.y = proposed_position.y;
        }

        let dragging = (mouse_buttons.pressed(MouseButton::Left)
            || mouse_buttons.pressed(MouseButton::Middle)
            || mouse_buttons.pressed(MouseButton::Right))
            && !(mouse_buttons.just_pressed(MouseButton::Left)
                || mouse_buttons.just_pressed(MouseButton::Middle)
                || mouse_buttons.just_pressed(MouseButton::Right));
        if dragging {
            let delta_device_pixels = current_cursor - last_cursor.unwrap_or(current_cursor);
            let world_units_per_pixel = projection.area.size() / view_size;
            let proposed_position =
                transform.translation.truncate() - delta_device_pixels * world_units_per_pixel;
            transform.translation.x = proposed_position.x;
            transform.translation.y = proposed_position.y;
        }
    }

    *last_cursor = active_camera.then_some(current_cursor);
}

#[cfg(not(feature = "splatting"))]
fn sync_view_cameras() {}

fn slider_value_for_settings(kind: AutomataSliderKind, settings: &AutomataSettings) -> f32 {
    match kind {
        AutomataSliderKind::ParticleLog2 => particle_slider_value(settings.particle_count),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame as f32,
        AutomataSliderKind::UpdateProb => settings.update_prob,
        AutomataSliderKind::DtLog2 => log2_slider_value(settings.dt),
        AutomataSliderKind::RenderScaleLog2 => log2_slider_value(settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => log2_slider_value(settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            log2_slider_value(settings.training_learning_rate)
        }
    }
}

fn slider_label(kind: AutomataSliderKind, settings: &AutomataSettings) -> String {
    match kind {
        AutomataSliderKind::ParticleLog2 => settings.particle_count.to_string(),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame.to_string(),
        AutomataSliderKind::UpdateProb => format!("{:.2}", settings.update_prob),
        AutomataSliderKind::DtLog2 => format!("{:.3}", settings.dt),
        AutomataSliderKind::RenderScaleLog2 => format!("{:.2}x", settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => format!("{:.2}x", settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            format!("{:.4}", settings.training_learning_rate)
        }
    }
}

fn log2_slider_value(value: f32) -> f32 {
    value.max(f32::MIN_POSITIVE).log2()
}

fn exp2_slider_value(value: f32) -> f32 {
    2.0_f32.powf(value)
}

fn particle_slider_value(particles: usize) -> f32 {
    (particles.max(64) as f32).log2().clamp(6.0, 16.0)
}

fn particles_from_slider_value(value: f32) -> usize {
    let log2 = value.round().clamp(6.0, 16.0) as u32;
    1usize << log2
}

fn load_selected_model(mut runtime: ResMut<AutomataRuntime>, settings: Res<AutomataSettings>) {
    if let Some(model_path) = &settings.model_path {
        if runtime.loaded_model_path.as_ref() == Some(model_path) {
            return;
        }
        match burn_automata::import::load_manifest(model_path) {
            Ok(manifest) => {
                runtime.hashgrid = manifest.hashgrid.clone();
                runtime.model = manifest.into_model();
                runtime.loaded_model_path = Some(model_path.clone());
                runtime.loaded_preset = None;
                runtime.trace = None;
                runtime.frame = 0;
                runtime.status = format!("loaded model {model_path}");
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                reset_training_stats(&mut runtime);
                runtime.model_revision = runtime.model_revision.wrapping_add(1);
            }
            Err(err) => {
                runtime.status = format!("model load failed: {err}");
            }
        }
        return;
    }

    if runtime.loaded_model_path.is_none() && runtime.loaded_preset == Some(settings.preset) {
        return;
    }
    let (config, hashgrid) = NpaConfig::for_preset(settings.preset);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_preset = Some(settings.preset);
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("seeded preset {:?}", settings.preset);
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(&mut runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
fn advance_rollout(mut runtime: ResMut<AutomataRuntime>, settings: Res<AutomataSettings>) {
    if settings.paused {
        return;
    }
    if runtime.trace.is_none() || runtime.frame.is_multiple_of(60) {
        initialize_cpu_rollout(&mut runtime, &settings);
        if settings.train_live {
            let trace = runtime.trace.clone();
            if let Some(trace) = trace.as_ref() {
                let hashgrid = effective_hashgrid(&runtime, &settings);
                update_training_probe(
                    &mut runtime,
                    trace,
                    &hashgrid,
                    settings.training_learning_rate,
                );
            }
        }
        return;
    }
    let previous_frame = runtime.frame;
    runtime.frame = runtime.frame.wrapping_add(1);
    if settings.train_live
        && crossed_interval(previous_frame, runtime.frame, TRAINING_INTERVAL_FRAMES)
    {
        let trace = runtime.trace.clone();
        if let Some(trace) = trace.as_ref() {
            let hashgrid = effective_hashgrid(&runtime, &settings);
            update_training_probe(
                &mut runtime,
                trace,
                &hashgrid,
                settings.training_learning_rate,
            );
        }
    }
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn advance_rollout(mut runtime: ResMut<AutomataRuntime>, settings: Res<AutomataSettings>) {
    if settings.paused {
        return;
    }
    let previous_frame = runtime.frame;
    runtime.frame = runtime.frame.wrapping_add(settings.steps_per_frame.max(1));
    if runtime.status == "ready" {
        runtime.status = "gpu automata -> planar gaussian buffers".to_string();
    }
    let crossed_training_interval =
        crossed_interval(previous_frame, runtime.frame, TRAINING_INTERVAL_FRAMES);
    let should_probe =
        settings.visualize_backward || (settings.train_live && crossed_training_interval);
    if should_probe {
        let cfg = RolloutConfig {
            particle_count: settings
                .particle_count
                .min(BACKWARD_PROBE_PARTICLES.max(TRAINING_PROBE_PARTICLES)),
            steps: 1,
            update_prob: settings.update_prob,
            dt: settings.dt,
            seed: settings.seed,
            seed_scale: settings.seed_scale,
            ..RolloutConfig::default()
        };
        let hashgrid = effective_hashgrid(&runtime, &settings);
        if let Ok(trace) = run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode) {
            if settings.visualize_backward {
                update_backward_probe(&mut runtime, &trace, &hashgrid);
            }
            if settings.train_live {
                update_training_probe(
                    &mut runtime,
                    &trace,
                    &hashgrid,
                    settings.training_learning_rate,
                );
            }
        }
    }
}

fn crossed_interval(previous: usize, current: usize, interval: usize) -> bool {
    interval > 0 && current / interval != previous / interval
}

#[cfg(not(all(feature = "splatting", feature = "gpu_wgpu")))]
fn initialize_cpu_rollout(runtime: &mut AutomataRuntime, settings: &AutomataSettings) {
    let cfg = RolloutConfig {
        particle_count: settings.particle_count,
        steps: settings.steps_per_frame.max(1),
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        ..RolloutConfig::default()
    };
    let hashgrid = effective_hashgrid(runtime, settings);
    match run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode) {
        Ok(trace) => {
            update_backward_probe(runtime, &trace, &hashgrid);
            runtime.trace = Some(trace);
            runtime.status = "initialized CPU rollout".to_string();
        }
        Err(err) => {
            runtime.status = format!("rollout failed: {err}");
        }
    }
}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
fn sync_cpu_trace_to_gaussian_asset(
    runtime: Res<AutomataRuntime>,
    cloud_state: Res<AutomataCloudState>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let Some(trace) = runtime.trace.as_ref() else {
        return;
    };
    let Some(handle) = cloud_state.handle.as_ref() else {
        return;
    };
    let Some(mut cloud) = assets.get_mut(handle) else {
        return;
    };
    let count = trace.positions.len().min(cloud_state.particle_count);
    let gaussians = (0..count)
        .map(|idx| trace_gaussian(&runtime, trace, idx))
        .collect::<Vec<_>>();
    *cloud = gaussians.into();
}

#[cfg(any(not(feature = "splatting"), feature = "gpu_wgpu"))]
fn sync_cpu_trace_to_gaussian_asset() {}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
fn trace_gaussian(runtime: &AutomataRuntime, trace: &RolloutTrace, idx: usize) -> Gaussian3d {
    let position = trace.positions[idx];
    let state_base = idx * trace.state_dims;
    let state = &trace.states[state_base..state_base + trace.state_dims];
    let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
    let tail = trace.state_dims.saturating_sub(3);
    let color = if trace.state_dims >= 3 {
        [
            (state[tail] + 0.5).clamp(0.0, 1.0),
            (state[tail + 1] + 0.5).clamp(0.0, 1.0),
            (state[tail + 2] + 0.5).clamp(0.0, 1.0),
        ]
    } else {
        [0.82, 0.88, 0.92]
    };
    spherical_harmonic.coefficients[0] = (color[0] - 0.5) / GAUSSIAN_SH_C0;
    spherical_harmonic.coefficients[1] = (color[1] - 0.5) / GAUSSIAN_SH_C0;
    spherical_harmonic.coefficients[2] = (color[2] - 0.5) / GAUSSIAN_SH_C0;

    let scale = (runtime.hashgrid.eps * 0.12).max(0.00008);
    let opacity = if runtime.model.config.spatial_dims == 3 {
        growth_3d_material_opacity_channel(trace.state_dims)
            .map(|channel| (1.0 / (1.0 + (-state[channel]).exp())).clamp(0.001, 0.95))
            .unwrap_or(1.0)
    } else {
        1.0
    };

    Gaussian3d {
        position_visibility: [
            position[0],
            position[1],
            if runtime.model.config.spatial_dims == 3 {
                position[2]
            } else {
                0.0
            },
            1.0,
        ]
        .into(),
        spherical_harmonic,
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [scale, scale, scale, opacity].into(),
    }
}

fn update_status_label(
    runtime: Res<AutomataRuntime>,
    settings: Res<AutomataSettings>,
    diagnostics: Option<Res<DiagnosticsStore>>,
    mut labels: Query<&mut Text, With<StatusLabel>>,
) {
    let fps = diagnostics
        .as_deref()
        .and_then(|store| store.get(&FrameTimeDiagnosticsPlugin::FPS))
        .and_then(|diagnostic| diagnostic.smoothed().or_else(|| diagnostic.average()));
    let frame_text = if let Some(fps) = fps {
        format!("frame {} | fps {:.1}", runtime.frame, fps)
    } else {
        format!("frame {}", runtime.frame)
    };
    for mut text in &mut labels {
        let mut metrics = Vec::new();
        if settings.visualize_backward {
            metrics.push("backward probe on".to_string());
        }
        if let (Some(loss), Some(grad_norm)) = (runtime.backward_loss, runtime.backward_grad_norm) {
            metrics.push(format!("backward loss {:.5}", loss));
            metrics.push(format!("backward grad {:.5}", grad_norm));
        }
        if settings.train_live {
            metrics.push(format!(
                "train {} {}r/{}f",
                LIVE_TRAINING_TARGET, TRAINING_PROBE_PARTICLES, TRAINING_INTERVAL_FRAMES
            ));
            metrics.push(format!("model rev {}", runtime.model_revision));
        }
        if let (Some(loss), Some(grad_norm)) = (runtime.training_loss, runtime.training_grad_norm) {
            metrics.push(format!("train step {}", runtime.training_step));
            metrics.push(format!("train loss {:.5}", loss));
            if let Some(best) = runtime.training_best_loss {
                metrics.push(format!("best {:.5}", best));
            }
            metrics.push(format!("train grad {:.5}", grad_norm));
        }
        let metric_text = if metrics.is_empty() {
            String::new()
        } else {
            format!(" | {}", metrics.join(" | "))
        };
        text.0 = format!("{}\n{}{}", runtime.status, frame_text, metric_text);
    }
}

fn update_settings_label(
    settings: Res<AutomataSettings>,
    mut labels: Query<&mut Text, With<SettingsLabel>>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut text in &mut labels {
        let train_state = if settings.train_live {
            format!(
                "{} {}r/{}f",
                LIVE_TRAINING_TARGET, TRAINING_PROBE_PARTICLES, TRAINING_INTERVAL_FRAMES
            )
        } else {
            "off".to_string()
        };
        text.0 = format!(
            "preset: {:?} | model: {}\nparticles: {} | steps: {} | p: {:.2} | dt: {:.3}\nmodel scale: {:.3} | splat: {:.2}x | opacity: {:.2}x\nbackward: {} | train: {} | lr: {:.4}",
            settings.preset,
            model_display_name(&settings),
            settings.particle_count,
            settings.steps_per_frame,
            settings.update_prob,
            settings.dt,
            settings.seed_scale,
            settings.render_scale,
            settings.render_opacity,
            settings.visualize_backward,
            train_state,
            settings.training_learning_rate,
        );
    }
}

fn model_display_name(settings: &AutomataSettings) -> String {
    settings
        .model_path
        .as_deref()
        .and_then(|path| Path::new(path).file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("seeded")
        .to_string()
}

fn update_backward_probe(
    runtime: &mut AutomataRuntime,
    trace: &RolloutTrace,
    hashgrid: &HashGridConfig,
) {
    match zero_update_batch_from_trace(runtime, hashgrid, trace, BACKWARD_PROBE_PARTICLES) {
        Ok(batch) => match supervised_backward(&runtime.model, &batch) {
            Ok((_grads, report)) => {
                runtime.backward_loss = Some(report.loss);
                runtime.backward_grad_norm = Some(report.grad_norm);
            }
            Err(err) => {
                runtime.backward_loss = None;
                runtime.backward_grad_norm = None;
                runtime.status = format!("backward failed: {err}");
            }
        },
        Err(err) => {
            runtime.backward_loss = None;
            runtime.backward_grad_norm = None;
            runtime.status = format!("probe failed: {err}");
        }
    }
}

fn probe_trace_for_controls(
    runtime: &AutomataRuntime,
    settings: &AutomataSettings,
    max_particles: usize,
) -> burn_automata::AutomataResult<RolloutTrace> {
    let cfg = RolloutConfig {
        particle_count: settings.particle_count.min(max_particles).max(1),
        steps: 1,
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        ..RolloutConfig::default()
    };
    let hashgrid = effective_hashgrid(runtime, settings);
    run_rollout(&runtime.model, &hashgrid, &cfg, settings.seed_mode)
}

fn update_training_probe(
    runtime: &mut AutomataRuntime,
    trace: &RolloutTrace,
    hashgrid: &HashGridConfig,
    learning_rate: f32,
) {
    match training_batch_from_trace(runtime, hashgrid, trace, TRAINING_PROBE_PARTICLES) {
        Ok(batch) => {
            let rows = batch.features.len() / runtime.model.config.perception_dims();
            match supervised_train_step(
                &mut runtime.model,
                &batch,
                SgdConfig {
                    learning_rate,
                    grad_clip_norm: 1.0,
                    ..SgdConfig::default()
                },
            )
            .and_then(|report| {
                let loss = supervised_loss(&runtime.model, &batch)?;
                Ok((report, loss))
            }) {
                Ok((report, loss)) => {
                    runtime.training_step = runtime.training_step.wrapping_add(1);
                    runtime.training_loss = Some(loss);
                    runtime.training_best_loss = Some(
                        runtime
                            .training_best_loss
                            .map_or(loss, |best| best.min(loss)),
                    );
                    runtime.training_grad_norm = Some(report.grad_norm);
                    runtime.model_revision = runtime.model_revision.wrapping_add(1);
                    runtime.status = format!(
                        "live train rollout teacher | step {} | rows {} | lr {:.4} | grad scale {:.3}",
                        runtime.training_step, rows, learning_rate, report.grad_scale
                    );
                }
                Err(err) => {
                    runtime.training_loss = None;
                    runtime.training_grad_norm = None;
                    runtime.status = format!("training failed: {err}");
                }
            }
        }
        Err(err) => {
            runtime.training_loss = None;
            runtime.training_grad_norm = None;
            runtime.status = format!("training probe failed: {err}");
        }
    }
}

fn training_batch_from_trace(
    runtime: &AutomataRuntime,
    hashgrid: &HashGridConfig,
    trace: &RolloutTrace,
    max_rows: usize,
) -> burn_automata::AutomataResult<burn_automata::SupervisedBatch> {
    let target = runtime
        .training_teacher
        .as_ref()
        .map(SupervisedTarget::Teacher)
        .unwrap_or(SupervisedTarget::ZeroUpdate);
    rollout_supervised_batch(
        &runtime.model,
        hashgrid,
        trace,
        target,
        RolloutBatchConfig { max_rows, dt: 1.0 },
    )
}

fn zero_update_batch_from_trace(
    runtime: &AutomataRuntime,
    hashgrid: &HashGridConfig,
    trace: &RolloutTrace,
    max_rows: usize,
) -> burn_automata::AutomataResult<burn_automata::SupervisedBatch> {
    rollout_supervised_batch(
        &runtime.model,
        hashgrid,
        trace,
        SupervisedTarget::ZeroUpdate,
        RolloutBatchConfig { max_rows, dt: 1.0 },
    )
}

fn apply_preset(runtime: &mut AutomataRuntime, preset: AutomataPreset) {
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;
    runtime.loaded_model_path = None;
    runtime.loaded_preset = Some(preset);
    runtime.trace = None;
    runtime.frame = 0;
    runtime.status = format!("preset changed to {preset:?}");
    runtime.backward_loss = None;
    runtime.backward_grad_norm = None;
    reset_training_stats(runtime);
    runtime.model_revision = runtime.model_revision.wrapping_add(1);
}

fn reset_training_stats(runtime: &mut AutomataRuntime) {
    runtime.training_step = 0;
    runtime.training_loss = None;
    runtime.training_best_loss = None;
    runtime.training_grad_norm = None;
    runtime.training_teacher = None;
}

#[cfg(feature = "splatting")]
fn setup_gaussian_cloud(
    mut commands: Commands,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut sorted_entries: ResMut<Assets<SortedEntries>>,
    settings: Res<AutomataSettings>,
    mut cloud_state: ResMut<AutomataCloudState>,
) {
    let cloud_asset = automata_gaussian_cloud(settings.particle_count);
    let sorted_len = sorted_entry_capacity(cloud_asset.len());
    let cloud = assets.add(cloud_asset);
    let sorted = sorted_entries.add(SortedEntries::new(1, sorted_len));
    cloud_state.handle = Some(cloud.clone());
    cloud_state.particle_count = settings.particle_count;
    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        SortedEntriesHandle(sorted),
        automata_cloud_settings(&settings, 2),
        automata_cloud_aabb(&settings),
        Transform::default(),
        Visibility::default(),
        AutomataGaussianCloud,
        Name::new("automata_gaussian_cloud"),
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: ClearColorConfig::Default,
            ..default()
        },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 2.6,
            },
            near: -1000.0,
            far: 1000.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
        AutomataCamera2d,
        Name::new("pancam_locked_gaussian_camera_2d"),
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
        PanOrbitCamera {
            enabled: false,
            allow_upside_down: true,
            target_focus: Vec3::ZERO,
            target_radius: 3.0,
            zoom_lower_limit: 0.05,
            zoom_upper_limit: Some(128.0),
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..default()
        },
        AutomataCamera3d,
        Name::new("panorbit_gaussian_camera"),
    ));
}

#[cfg(not(feature = "splatting"))]
fn setup_gaussian_cloud() {}

#[cfg(feature = "splatting")]
fn automata_cloud_settings(settings: &AutomataSettings, spatial_dims: usize) -> CloudSettings {
    CloudSettings {
        global_opacity: settings.render_opacity,
        global_scale: settings.render_scale,
        opacity_adaptive_radius: true,
        sort_mode: if spatial_dims == 2 {
            SortMode::None
        } else {
            settings.render_sort_mode_3d.clone()
        },
        radix_sort_depth_bits: RadixSortDepthBits::Bits32,
        gaussian_mode: if spatial_dims == 2 {
            GaussianMode::Gaussian2d
        } else {
            GaussianMode::Gaussian3d
        },
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

#[cfg(feature = "splatting")]
fn automata_cloud_aabb(settings: &AutomataSettings) -> Aabb {
    let extent = settings
        .seed_scale
        .max(settings.reference_seed_scale)
        .max(1.6)
        * 2.25;
    Aabb::from_min_max(Vec3::splat(-extent), Vec3::splat(extent))
}

#[cfg(feature = "splatting")]
fn sync_gaussian_cloud_settings(
    settings: Res<AutomataSettings>,
    runtime: Res<AutomataRuntime>,
    mut clouds: Query<(&mut CloudSettings, &mut Aabb), With<AutomataGaussianCloud>>,
) {
    if !settings.is_changed() && !runtime.is_changed() {
        return;
    }
    let next = automata_cloud_settings(&settings, runtime.model.config.spatial_dims);
    let next_aabb = automata_cloud_aabb(&settings);
    for (mut cloud, mut aabb) in &mut clouds {
        *cloud = next.clone();
        *aabb = next_aabb;
    }
}

#[cfg(not(feature = "splatting"))]
fn sync_gaussian_cloud_settings() {}

#[cfg(feature = "splatting")]
fn sync_gaussian_cloud_asset(
    mut commands: Commands,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut sorted_entries: ResMut<Assets<SortedEntries>>,
    settings: Res<AutomataSettings>,
    mut cloud_state: ResMut<AutomataCloudState>,
    mut clouds: Query<
        (
            Entity,
            &mut PlanarGaussian3dHandle,
            &mut SortedEntriesHandle,
            &mut Visibility,
        ),
        With<AutomataGaussianCloud>,
    >,
    gaussian_cameras: Query<&Camera, With<GaussianCamera>>,
) {
    if cloud_state.handle.is_some() && cloud_state.particle_count == settings.particle_count {
        return;
    }
    let cloud_asset = automata_gaussian_cloud(settings.particle_count);
    let sorted_len = sorted_entry_capacity(cloud_asset.len());
    let cloud = assets.add(cloud_asset);
    let camera_count = active_gaussian_camera_count(&gaussian_cameras);
    let sorted = sorted_entries.add(SortedEntries::new(camera_count, sorted_len));
    cloud_state.handle = Some(cloud.clone());
    cloud_state.particle_count = settings.particle_count;
    for (entity, mut handle, mut sorted_handle, mut visibility) in &mut clouds {
        *handle = PlanarGaussian3dHandle(cloud.clone());
        *sorted_handle = SortedEntriesHandle(sorted.clone());
        *visibility = Visibility::Hidden;
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(AutomataCloudResizeCooldown(2));
        #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
        entity_commands
            .remove::<PlanarStorageBindGroup<Gaussian3d>>()
            .remove::<SortBindGroup>();
    }
}

#[cfg(feature = "splatting")]
fn sorted_entry_capacity(cloud_len: usize) -> usize {
    cloud_len.max(SORTED_ENTRY_MIN_CAPACITY)
}

#[cfg(feature = "splatting")]
fn restore_resized_gaussian_cloud_visibility(
    mut commands: Commands,
    mut clouds: Query<(Entity, &mut Visibility, &mut AutomataCloudResizeCooldown)>,
) {
    for (entity, mut visibility, mut cooldown) in &mut clouds {
        cooldown.0 = cooldown.0.saturating_sub(1);
        if cooldown.0 == 0 {
            *visibility = Visibility::Inherited;
            commands
                .entity(entity)
                .remove::<AutomataCloudResizeCooldown>();
        }
    }
}

#[cfg(not(feature = "splatting"))]
fn restore_resized_gaussian_cloud_visibility() {}

#[cfg(feature = "splatting")]
fn automata_gaussian_cloud(count: usize) -> PlanarGaussian3d {
    let gaussian = Gaussian3d {
        position_visibility: [0.0, 0.0, 0.0, 0.0].into(),
        rotation: [1.0, 0.0, 0.0, 0.0].into(),
        scale_opacity: [0.00008, 0.00008, 0.00008, 0.0].into(),
        ..Default::default()
    };
    vec![gaussian; count].into()
}

#[cfg(feature = "splatting")]
fn active_gaussian_camera_count(cameras: &Query<&Camera, With<GaussianCamera>>) -> usize {
    cameras
        .iter()
        .filter(|camera| camera.is_active)
        .count()
        .max(1)
}

#[cfg(not(feature = "splatting"))]
fn sync_gaussian_cloud_asset() {}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
fn extract_automata_render_config(
    mut commands: Commands,
    main_world: ResMut<bevy::render::MainWorld>,
) {
    let Some(settings) = main_world.get_resource::<AutomataSettings>().cloned() else {
        return;
    };
    let Some(runtime) = main_world.get_resource::<AutomataRuntime>().cloned() else {
        return;
    };
    let hashgrid = effective_hashgrid(&runtime, &settings);
    let neighbor_mode = effective_gpu_neighbor_mode(&runtime, &settings);
    let reinit_key =
        automata_render_reinit_key(&runtime.model, &hashgrid, &settings, neighbor_mode);
    let param_key = AutomataRenderParamKey {
        model_revision: runtime.model_revision,
        dt_bits: settings.dt.to_bits(),
        update_prob_bits: settings.update_prob.to_bits(),
    };
    commands.insert_resource(AutomataRenderConfig {
        model: runtime.model,
        hashgrid,
        reinit_key,
        param_key,
        particle_count: settings.particle_count,
        steps_per_frame: settings.steps_per_frame,
        update_prob: settings.update_prob,
        dt: settings.dt,
        seed: settings.seed,
        seed_scale: settings.seed_scale,
        seed_mode: settings.seed_mode,
        neighbor_mode,
        paused: settings.paused,
        model_revision: runtime.model_revision,
    });
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[allow(clippy::too_many_arguments)]
fn step_automata_into_gaussians(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    config: Option<Res<AutomataRenderConfig>>,
    mut render_state: ResMut<AutomataRenderState>,
    mut diagnostics: ResMut<AutomataRenderDiagnostics>,
    gpu_planars: Res<RenderAssets<PlanarStorageGaussian3d>>,
    cloud_handles: Query<(Entity, &PlanarGaussian3dHandle)>,
) {
    let Some(config) = config else {
        return;
    };
    diagnostics.requested_particle_count = config.particle_count;
    if config.paused {
        return;
    }
    let Some((cloud_entity, cloud_handle)) = cloud_handles.iter().next() else {
        diagnostics.last_error = Some("waiting for gaussian cloud entity".to_string());
        return;
    };
    let Some(storage) = gpu_planars.get(&cloud_handle.0) else {
        diagnostics.last_error = Some("waiting for gaussian render asset".to_string());
        return;
    };
    diagnostics.gaussian_storage_count = storage.count;
    if storage.count < config.particle_count {
        let message = format!(
            "waiting for gaussian storage resize: storage={} particles={}",
            storage.count, config.particle_count
        );
        render_state.last_error = Some(message.clone());
        diagnostics.last_error = Some(message);
        return;
    }

    let asset_id = cloud_handle.0.id();
    let asset_changed = render_state.asset_id != Some(asset_id);
    if asset_changed {
        render_state.gaussian_bind_group = None;
        commands
            .entity(cloud_entity)
            .remove::<PlanarStorageBindGroup<Gaussian3d>>()
            .remove::<SortBindGroup>();
    }
    let needs_reinit = render_state.state.is_none()
        || render_state.reinit_key != config.reinit_key
        || asset_changed;
    if needs_reinit {
        if render_state.executor.is_none() {
            match automata_executor_from_render_device(&render_device, &render_queue) {
                Ok(executor) => render_state.executor = Some(executor),
                Err(err) => {
                    let message = err.to_string();
                    render_state.last_error = Some(message.clone());
                    diagnostics.last_error = Some(message);
                    return;
                }
            }
        }
        let (positions, states) = burn_automata::rollout::seed_particles_scaled(
            1,
            config.particle_count,
            config.model.config.state_dims,
            config.model.config.spatial_dims,
            config.seed,
            config.seed_mode,
            config.seed_scale,
        );
        let Some(executor) = render_state.executor.as_ref() else {
            return;
        };
        match executor.create_state_with_neighbor_mode_and_update_prob(
            &config.model,
            &positions,
            &states,
            1,
            config.particle_count,
            &config.hashgrid,
            config.dt,
            config.neighbor_mode,
            config.update_prob,
            config.seed,
        ) {
            Ok(state) => {
                render_state.state = Some(state);
                render_state.gaussian_bind_group = None;
                render_state.reinit_key = config.reinit_key;
                render_state.param_key = config.param_key;
                render_state.model_revision = config.model_revision;
                render_state.asset_id = Some(asset_id);
                render_state.frame = 0;
                render_state.last_error = None;
                diagnostics.resident_particle_count = config.particle_count;
                diagnostics.last_error = None;
            }
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    } else if render_state.param_key != config.param_key {
        let update_result = {
            let AutomataRenderState {
                executor, state, ..
            } = &mut *render_state;
            match (executor.as_ref(), state.as_mut()) {
                (Some(executor), Some(state)) => executor.update_state_model(
                    state,
                    &config.model,
                    &config.hashgrid,
                    config.dt,
                    config.update_prob,
                    config.seed,
                ),
                _ => Ok(()),
            }
        };
        match update_result {
            Ok(()) => {
                render_state.param_key = config.param_key;
                render_state.model_revision = config.model_revision;
                render_state.last_error = None;
                diagnostics.last_error = None;
            }
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    }

    let gaussian_refs = gaussian_storage_buffer_refs(storage);
    if render_state.gaussian_bind_group.is_none() {
        let Some(executor) = render_state.executor.as_ref() else {
            return;
        };
        match executor.create_gaussian_bind_group(&gaussian_refs, storage.count) {
            Ok(bind_group) => render_state.gaussian_bind_group = Some(bind_group),
            Err(err) => {
                let message = err.to_string();
                render_state.last_error = Some(message.clone());
                diagnostics.last_error = Some(message);
                return;
            }
        }
    }
    let steps = config.steps_per_frame.max(1);
    let step_result = {
        let AutomataRenderState {
            executor,
            state,
            gaussian_bind_group,
            ..
        } = &mut *render_state;
        let Some(executor) = executor.as_ref() else {
            return;
        };
        let Some(state) = state.as_mut() else {
            return;
        };
        let Some(gaussian_bind_group) = gaussian_bind_group.as_ref() else {
            return;
        };
        executor
            .step_state_many_into_gaussian_bind_group(state, gaussian_bind_group, steps)
            .map_err(|err| err.to_string())
    };
    match step_result {
        Ok(completed) => {
            write_gaussian_draw_indirect_count(&render_queue, storage, config.particle_count);
            render_state.frame = render_state.frame.wrapping_add(completed);
            render_state.last_error = None;
            diagnostics.frame = render_state.frame;
            diagnostics.last_error = None;
        }
        Err(err) => {
            render_state.last_error = Some(err.clone());
            diagnostics.last_error = Some(err);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::hash_map::DefaultHasher,
        collections::{HashMap, HashSet},
        hash::{Hash, Hasher},
    };

    #[test]
    fn m_key_toggles_ui_visibility() {
        let mut app = App::new();
        app.init_resource::<AutomataUiState>()
            .insert_resource(ButtonInput::<KeyCode>::default())
            .add_systems(Update, toggle_ui_visibility);
        let root = app
            .world_mut()
            .spawn((AutomataUiRoot, Visibility::Inherited))
            .id();

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyM);
        app.update();

        assert!(!app.world().resource::<AutomataUiState>().visible);
        assert_eq!(
            *app.world().entity(root).get::<Visibility>().unwrap(),
            Visibility::Hidden
        );
    }

    #[test]
    fn catalog_selection_preserves_visualization_settings() {
        let mut settings = AutomataSettings {
            render_scale: 1.5,
            render_opacity: 0.375,
            steps_per_frame: 3,
            training_learning_rate: 0.004,
            model_path: Some("previous-model.bpk".to_string()),
            ..Default::default()
        };
        let mut runtime = AutomataRuntime::default();

        select_catalog_entry(ModelCatalogKey::Growing3dGs, &mut settings, &mut runtime);

        assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
        assert_eq!(settings.particle_count, 1024);
        assert_eq!(settings.steps_per_frame, 3);
        assert_eq!(settings.seed, RolloutConfig::default().seed);
        assert!((settings.reference_seed_scale - 0.35).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);

        select_catalog_entry(
            ModelCatalogKey::UvTorusMorphogen3d,
            &mut settings,
            &mut runtime,
        );

        assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
        assert_eq!(settings.particle_count, 1024);
        assert_eq!(settings.steps_per_frame, 3);
        assert_eq!(settings.seed, CATALOG_3D_GROWTH_SEED);
        assert_eq!(settings.seed_mode, ParticleSeed::TorusGrowth3d);
        assert!((settings.reference_seed_scale - 0.54).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);

        select_catalog_entry(
            ModelCatalogKey::TeapotMorphogen3d,
            &mut settings,
            &mut runtime,
        );

        assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
        assert_eq!(settings.particle_count, 1024);
        assert_eq!(settings.steps_per_frame, 2);
        assert_eq!(settings.seed, CATALOG_3D_GROWTH_SEED);
        assert_eq!(settings.seed_mode, ParticleSeed::TeapotGrowth3d);
        assert!((settings.reference_seed_scale - 0.72).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);

        select_catalog_entry(ModelCatalogKey::Texture2d, &mut settings, &mut runtime);

        assert_eq!(settings.preset, AutomataPreset::Texture2d);
        assert_eq!(settings.particle_count, 4096);
        assert_eq!(settings.steps_per_frame, 2);
        assert_eq!(settings.seed, RolloutConfig::default().seed);
        assert_eq!(settings.seed_mode, ParticleSeed::UniformCircle);
        assert!((settings.reference_seed_scale - 1.0).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);
    }

    #[test]
    fn catalog_keeps_only_latest_torus_regression_artifact() {
        let torus_entries = MODEL_CATALOG
            .iter()
            .filter(|entry| entry.title.contains("torus"))
            .collect::<Vec<_>>();
        assert_eq!(torus_entries.len(), 1);
        assert_eq!(torus_entries[0].key, ModelCatalogKey::UvTorusMorphogen3d);
        assert_eq!(
            catalog_seed_mode(torus_entries[0]),
            ParticleSeed::TorusGrowth3d
        );
        assert!(matches!(
            torus_entries[0].source,
            ModelCatalogSource::Bpk {
                primary: "assets/models/uv_torus_growth_3d.bpk",
                ..
            }
        ));
    }

    #[test]
    fn visible_catalog_hides_blocked_3d_mesh_artifacts() {
        assert!(
            !VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::UvTorusMorphogen3d),
            "torus remains registered for regression loading but must not be selectable until validation passes"
        );
        assert!(
            !VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::TeapotMorphogen3d),
            "teapot remains registered for regression loading but must not be selectable until seed-varied robust validation passes"
        );
        assert!(VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::Growing3dGs));
    }

    #[test]
    fn catalog_registers_teapot_as_blocked_growth_artifact() {
        let teapot_entries = MODEL_CATALOG
            .iter()
            .filter(|entry| entry.title.contains("teapot"))
            .collect::<Vec<_>>();
        assert_eq!(teapot_entries.len(), 1);
        assert_eq!(teapot_entries[0].key, ModelCatalogKey::TeapotMorphogen3d);
        assert_eq!(teapot_entries[0].particle_count, 1024);
        assert!(
            teapot_entries[0].kind.contains("validation blocked"),
            "teapot should stay hidden until robust held-out seed validation passes"
        );
        assert_eq!(
            catalog_seed_mode(teapot_entries[0]),
            ParticleSeed::TeapotGrowth3d
        );
        assert!(matches!(
            teapot_entries[0].source,
            ModelCatalogSource::Bpk {
                primary: "assets/models/teapot_growth_3d.bpk",
                ..
            }
        ));
    }

    #[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
    #[test]
    fn catalog_3d_default_uses_sorted_gpu_neighbor_mode() {
        let mut runtime = AutomataRuntime::default();
        let (config, hashgrid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
        runtime.model = NpaModel::seeded(config, 42);
        runtime.hashgrid = hashgrid;

        let mut settings = AutomataSettings {
            preset: AutomataPreset::Growing3dGs,
            particle_count: 1024,
            gpu_neighbor_mode: WgpuNeighborMode::Auto,
            ..Default::default()
        };
        assert_eq!(
            effective_gpu_neighbor_mode(&runtime, &settings),
            WgpuNeighborMode::SortedCells
        );

        settings.particle_count = 4096;
        assert_eq!(
            effective_gpu_neighbor_mode(&runtime, &settings),
            WgpuNeighborMode::Auto
        );

        settings.gpu_neighbor_mode = WgpuNeighborMode::LinkedList;
        assert_eq!(
            effective_gpu_neighbor_mode(&runtime, &settings),
            WgpuNeighborMode::LinkedList
        );
    }

    #[test]
    fn catalog_3d_bpk_entries_use_local_growth_seeded_models() {
        for key in [
            ModelCatalogKey::UvTorusMorphogen3d,
            ModelCatalogKey::TeapotMorphogen3d,
        ] {
            let entry = catalog_entry(key);
            let path = resolved_catalog_model_path(entry)
                .unwrap_or_else(|| panic!("missing catalog model {}", entry.title));
            let manifest = burn_automata::import::load_manifest(&path)
                .unwrap_or_else(|err| panic!("failed to load {path}: {err}"));

            assert_eq!(manifest.config.spatial_dims, 3, "{path}");
            assert!(
                !manifest.config.position_features,
                "{path} must not depend on absolute position features"
            );
            let source = manifest.source.as_deref().unwrap_or_default();
            let expected_source = match key {
                ModelCatalogKey::UvTorusMorphogen3d => {
                    "render-refined-rust:ablation-rust:uv-torus-3d:conditionless-local-random-ball-rollout-ablation"
                }
                ModelCatalogKey::TeapotMorphogen3d => {
                    "retimed-local-front:hidden=skipped:gain=2:alpha=1:front_retime=false:active_opacity_hidden=skipped:active_opacity_gain=skipped:opacity_bias=skipped:material_opacity_bias=0.55:base=render-refined-rust:ablation-rust:utah-teapot-2026:conditionless-local-random-ball-rollout-ablation"
                }
                _ => unreachable!("only 3D growth catalog entries are checked here"),
            };
            assert_eq!(
                source, expected_source,
                "{path} should point at the current reviewed latest dynamic 3D growth artifact"
            );
            assert!(
                (source.starts_with("render-refined-rust:")
                    || source.starts_with("retimed-local-front:"))
                    && source.contains("conditionless-local")
                    && !source.contains("position-field")
                    && !source.contains("seed-frame")
                    && !source.contains("render-proxy-rust"),
                "{path} must use latest local render-refinement lineage without target-assigned shortcuts, source={source}"
            );
            assert!(matches!(
                catalog_seed_mode(entry),
                ParticleSeed::TorusGrowth3d | ParticleSeed::TeapotGrowth3d
            ));
        }
    }

    #[cfg(feature = "splatting")]
    #[test]
    fn automata_camera_viewport_centers_right_pane_when_ui_visible() {
        let viewport = automata_camera_viewport(UVec2::new(1600, 900), 1.0, true)
            .expect("wide window should allocate right-pane viewport");

        assert_eq!(viewport.physical_position, UVec2::new(540, 0));
        assert_eq!(viewport.physical_size, UVec2::new(1060, 900));
        assert!(automata_camera_viewport(UVec2::new(1600, 900), 1.0, false).is_none());
        assert!(automata_camera_viewport(UVec2::new(700, 900), 1.0, true).is_none());
    }

    #[cfg(feature = "splatting")]
    #[test]
    fn automata_camera_viewport_uses_physical_scale_factor() {
        let viewport = automata_camera_viewport(UVec2::new(3200, 1800), 2.0, true)
            .expect("hidpi window should allocate right-pane viewport");

        assert_eq!(viewport.physical_position, UVec2::new(1080, 0));
        assert_eq!(viewport.physical_size, UVec2::new(2120, 1800));
    }

    #[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
    #[test]
    fn cpu_trace_gaussian_fallback_writes_visible_gaussian() {
        let runtime = AutomataRuntime::default();
        let cfg = RolloutConfig {
            particle_count: 32,
            steps: 1,
            seed_scale: 0.2,
            update_prob: 1.0,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(
            &runtime.model,
            &runtime.hashgrid,
            &cfg,
            ParticleSeed::UniformCircle,
        )
        .unwrap();
        let gaussian = trace_gaussian(&runtime, &trace, 0);
        assert_eq!(gaussian.position_visibility.visibility, 1.0);
        assert!(gaussian.scale_opacity.scale[0] > 0.0);
        assert!(gaussian.scale_opacity.opacity > 0.0);
        assert!(
            gaussian
                .spherical_harmonic
                .coefficients
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn live_training_probe_updates_convergence_metrics_and_model_revision() {
        let mut runtime = AutomataRuntime::default();
        let cfg = RolloutConfig {
            particle_count: 32,
            steps: 1,
            seed_scale: 0.2,
            ..RolloutConfig::default()
        };
        let trace = run_rollout(
            &runtime.model,
            &runtime.hashgrid,
            &cfg,
            ParticleSeed::UniformCircle,
        )
        .unwrap();
        let previous_revision = runtime.model_revision;

        let settings = AutomataSettings::default();
        let hashgrid = effective_hashgrid(&runtime, &settings);
        update_training_probe(&mut runtime, &trace, &hashgrid, 1.0e-3);

        assert_eq!(runtime.training_step, 1);
        assert!(runtime.training_loss.is_some_and(f32::is_finite));
        assert!(runtime.training_grad_norm.is_some_and(f32::is_finite));
        assert_eq!(runtime.training_best_loss, runtime.training_loss);
        assert_ne!(runtime.model_revision, previous_revision);
    }

    #[test]
    fn run_control_active_state_tracks_settings() {
        let mut settings = AutomataSettings::default();

        assert!(!run_control_is_active(RunControlKind::Pause, &settings));
        assert!(!run_control_is_active(RunControlKind::Backward, &settings));
        assert!(!run_control_is_active(RunControlKind::Train, &settings));

        settings.paused = true;
        settings.visualize_backward = true;
        settings.train_live = true;

        assert!(run_control_is_active(RunControlKind::Pause, &settings));
        assert!(run_control_is_active(RunControlKind::Backward, &settings));
        assert!(run_control_is_active(RunControlKind::Train, &settings));
        assert!(!run_control_is_active(RunControlKind::Reset, &settings));
    }

    #[test]
    fn control_probe_trace_is_bounded_and_finite() {
        let runtime = AutomataRuntime::default();
        let settings = AutomataSettings {
            particle_count: 4096,
            ..Default::default()
        };

        let trace = probe_trace_for_controls(&runtime, &settings, 64).unwrap();

        assert_eq!(trace.particle_count, 64);
        assert_eq!(trace.steps, 1);
        assert!(
            trace
                .positions
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(trace.states.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn training_probe_interval_triggers_on_crossing() {
        assert!(!crossed_interval(58, 59, TRAINING_INTERVAL_FRAMES));
        assert!(crossed_interval(59, 60, TRAINING_INTERVAL_FRAMES));
        assert!(crossed_interval(56, 64, TRAINING_INTERVAL_FRAMES));
        assert!(!crossed_interval(60, 64, TRAINING_INTERVAL_FRAMES));
    }

    #[test]
    fn model_catalog_has_unique_keys_and_paths() {
        let mut keys = HashSet::new();
        let mut paths = HashSet::new();
        for entry in MODEL_CATALOG {
            assert!(
                keys.insert(entry.key),
                "duplicate catalog key {:?}",
                entry.key
            );
            if let ModelCatalogSource::Bpk { primary, .. } = entry.source {
                assert!(
                    paths.insert(primary),
                    "duplicate catalog primary path {primary}"
                );
            }
        }
    }

    #[test]
    fn catalog_thumbnails_are_embedded_decodable_and_distinct() {
        let mut hashes: HashMap<u64, ModelCatalogKey> = HashMap::new();
        for entry in MODEL_CATALOG {
            let bytes = catalog_thumbnail_png(entry.key);
            assert!(
                bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
                "catalog thumbnail for {:?} is not a PNG",
                entry.key
            );

            let image = catalog_thumbnail_image(entry.key);
            assert_eq!(image.texture_descriptor.size.width, 96);
            assert_eq!(image.texture_descriptor.size.height, 72);
            assert_eq!(
                image.texture_descriptor.format,
                TextureFormat::Rgba8UnormSrgb
            );

            let data = image
                .data
                .as_ref()
                .expect("decoded catalog thumbnail should keep CPU pixel data");
            assert_eq!(data.len(), 96 * 72 * 4);

            let mut min_rgb = [u8::MAX; 3];
            let mut max_rgb = [u8::MIN; 3];
            for pixel in data.chunks_exact(4) {
                for channel in 0..3 {
                    min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
                    max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
                }
            }
            let dynamic_range = (0..3)
                .map(|channel| max_rgb[channel] - min_rgb[channel])
                .max()
                .unwrap_or_default();
            assert!(
                dynamic_range > 24,
                "catalog thumbnail for {:?} looks blank: min={min_rgb:?} max={max_rgb:?}",
                entry.key
            );

            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            let hash = hasher.finish();
            let duplicate = hashes.insert(hash, entry.key);
            assert!(
                duplicate.is_none(),
                "catalog thumbnail for {:?} duplicates {:?}",
                entry.key,
                duplicate
            );
        }
        assert_eq!(hashes.len(), MODEL_CATALOG.len());
    }

    #[test]
    fn uv_torus_preview_image_is_large_and_colored() {
        let image = catalog_preview_image(ModelCatalogKey::UvTorusMorphogen3d, 0.0);
        assert_eq!(image.texture_descriptor.size.width, 320);
        assert_eq!(image.texture_descriptor.size.height, 232);

        let data = image
            .data
            .as_ref()
            .expect("preview image should keep CPU pixel data");
        let mut min_rgb = [u8::MAX; 3];
        let mut max_rgb = [u8::MIN; 3];
        for pixel in data.chunks_exact(4) {
            for channel in 0..3 {
                min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
                max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
            }
        }

        assert!(max_rgb[0] - min_rgb[0] > 80, "weak red range");
        assert!(max_rgb[1] - min_rgb[1] > 80, "weak green range");
        assert!(max_rgb[2] - min_rgb[2] > 80, "weak blue range");
    }

    #[test]
    fn teapot_preview_image_is_large_and_colored() {
        let image = catalog_preview_image(ModelCatalogKey::TeapotMorphogen3d, 0.0);
        assert_eq!(image.texture_descriptor.size.width, 320);
        assert_eq!(image.texture_descriptor.size.height, 232);

        let data = image
            .data
            .as_ref()
            .expect("preview image should keep CPU pixel data");
        let mut min_rgb = [u8::MAX; 3];
        let mut max_rgb = [u8::MIN; 3];
        for pixel in data.chunks_exact(4) {
            for channel in 0..3 {
                min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
                max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
            }
        }
        let dynamic_range = (0..3)
            .map(|channel| max_rgb[channel] - min_rgb[channel])
            .max()
            .unwrap_or_default();
        assert!(
            dynamic_range > 80,
            "teapot preview looks blank: min={min_rgb:?} max={max_rgb:?}"
        );
    }

    #[cfg(feature = "splatting")]
    #[test]
    fn sorted_entry_capacity_covers_resize_handoff_without_full_slider_floor() {
        assert_eq!(sorted_entry_capacity(0), SORTED_ENTRY_MIN_CAPACITY);
        assert_eq!(sorted_entry_capacity(128), SORTED_ENTRY_MIN_CAPACITY);
        assert_eq!(sorted_entry_capacity(4096), SORTED_ENTRY_MIN_CAPACITY);
        assert_eq!(sorted_entry_capacity(65_536), 65_536);
    }

    #[cfg(feature = "splatting")]
    #[test]
    fn automata_cloud_settings_use_display_rgb_color_space() {
        let settings = AutomataSettings::default();
        let cloud_settings = automata_cloud_settings(&settings, 2);

        assert_eq!(
            cloud_settings.color_space,
            GaussianColorSpace::SrgbRec709Display
        );
        assert_eq!(cloud_settings.sort_mode, SortMode::None);

        let cloud_settings_3d = automata_cloud_settings(&settings, 3);
        assert_eq!(cloud_settings_3d.sort_mode, SortMode::Radix);
        assert_eq!(
            cloud_settings_3d.radix_sort_depth_bits,
            RadixSortDepthBits::Bits32
        );
    }
}
