use std::path::Path;

use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, ImageSampler, ImageType},
    prelude::Image,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_automata::{
    AutomataPreset, ParticleSeed, RolloutConfig,
    rollout::{uv_torus_outer_radius, uv_torus_position_color},
    target_geometry::TriangleMeshTarget,
};

use super::{AutomataRuntime, AutomataSettings, apply_preset, reset_training_stats};

pub(super) const DEFAULT_LIZARD_MODEL: &str = "/tmp/burn_automata_lizard.bpk";
const DEFAULT_POLKA_MODEL: &str = "/tmp/burn_automata_polka.bpk";
const FALLBACK_POLKA_MODEL: &str = "/tmp/polka_dotted.bpk";
pub(super) const BACKWARD_PROBE_PARTICLES: usize = 1024;
pub(super) const TRAINING_PROBE_PARTICLES: usize = 256;
pub(super) const TRAINING_INTERVAL_FRAMES: usize = 60;
pub(super) const LIVE_TRAINING_TARGET: &str = "rollout teacher";
pub(super) const CATALOG_DOUBLE_CLICK_SECONDS: f64 = 0.35;
pub(super) const CATALOG_3D_GROWTH_SEED: u64 = 0x0051_a73d;
pub(super) const AUTOMATA_UI_PANEL_WIDTH: f32 = 540.0;
#[cfg(feature = "splatting")]
pub(super) const AUTOMATA_MIN_VIEWPORT_WIDTH: u32 = 256;
#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
pub(super) const GAUSSIAN_SH_C0: f32 = 0.282_094_8;
#[cfg(feature = "splatting")]
pub(super) const SORTED_ENTRY_MIN_CAPACITY: usize = 16_384;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) enum ModelCatalogKey {
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
pub(super) const VISIBLE_MODEL_CATALOG_KEYS: &[ModelCatalogKey] = &[
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
pub(super) enum ModelCatalogSource {
    Preset,
    Bpk {
        primary: &'static str,
        fallback: Option<&'static str>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModelCatalogEntry {
    pub(super) key: ModelCatalogKey,
    pub(super) title: &'static str,
    pub(super) kind: &'static str,
    pub(super) detail: &'static str,
    pub(super) preset: AutomataPreset,
    pub(super) source: ModelCatalogSource,
    pub(super) particle_count: usize,
    pub(super) seed_scale: f32,
    pub(super) update_prob: f32,
}

pub(super) const MODEL_CATALOG: &[ModelCatalogEntry] = &[
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

pub(super) fn catalog_entry(key: ModelCatalogKey) -> &'static ModelCatalogEntry {
    MODEL_CATALOG
        .iter()
        .find(|entry| entry.key == key)
        .expect("model catalog key must have an entry")
}

pub(super) fn compact_particle_count(particle_count: usize) -> String {
    if particle_count >= 1024 {
        format!("{}k", particle_count / 1024)
    } else {
        particle_count.to_string()
    }
}

pub(super) fn catalog_entry_matches_settings(
    entry: &ModelCatalogEntry,
    settings: &AutomataSettings,
) -> bool {
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

pub(super) fn select_catalog_entry(
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

pub(super) fn resolved_catalog_model_path(entry: &ModelCatalogEntry) -> Option<String> {
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

pub(super) fn catalog_seed_mode(entry: &ModelCatalogEntry) -> ParticleSeed {
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

pub(super) fn catalog_thumbnail_png(key: ModelCatalogKey) -> &'static [u8] {
    match key {
        ModelCatalogKey::Lizard => {
            include_bytes!("../../../../assets/catalog_thumbnails/lizard.png")
        }
        ModelCatalogKey::Butterfly => {
            include_bytes!("../../../../assets/catalog_thumbnails/butterfly.png")
        }
        ModelCatalogKey::Rose => include_bytes!("../../../../assets/catalog_thumbnails/rose.png"),
        ModelCatalogKey::Turtle => {
            include_bytes!("../../../../assets/catalog_thumbnails/turtle.png")
        }
        ModelCatalogKey::Mushroom => {
            include_bytes!("../../../../assets/catalog_thumbnails/mushroom.png")
        }
        ModelCatalogKey::TropicalFish => {
            include_bytes!("../../../../assets/catalog_thumbnails/tropical_fish.png")
        }
        ModelCatalogKey::Sun => {
            include_bytes!("../../../../assets/catalog_thumbnails/sun_with_face.png")
        }
        ModelCatalogKey::Ghost => include_bytes!("../../../../assets/catalog_thumbnails/ghost.png"),
        ModelCatalogKey::Frog => {
            include_bytes!("../../../../assets/catalog_thumbnails/frog_face.png")
        }
        ModelCatalogKey::Apple => {
            include_bytes!("../../../../assets/catalog_thumbnails/red_apple.png")
        }
        ModelCatalogKey::Polka => {
            include_bytes!("../../../../assets/catalog_thumbnails/polka_dotted_0121.png")
        }
        ModelCatalogKey::Bubbly => {
            include_bytes!("../../../../assets/catalog_thumbnails/bubbly_0101.png")
        }
        ModelCatalogKey::Clouds => {
            include_bytes!("../../../../assets/catalog_thumbnails/clouds.png")
        }
        ModelCatalogKey::Galaxy => {
            include_bytes!("../../../../assets/catalog_thumbnails/galaxy.png")
        }
        ModelCatalogKey::Hearts => {
            include_bytes!("../../../../assets/catalog_thumbnails/hearts.png")
        }
        ModelCatalogKey::Rings => include_bytes!("../../../../assets/catalog_thumbnails/rings.png"),
        ModelCatalogKey::Stars => include_bytes!("../../../../assets/catalog_thumbnails/stars.png"),
        ModelCatalogKey::Grid => {
            include_bytes!("../../../../assets/catalog_thumbnails/grid_0040.png")
        }
        ModelCatalogKey::Banded => {
            include_bytes!("../../../../assets/catalog_thumbnails/banded_0037.png")
        }
        ModelCatalogKey::Tree => include_bytes!("../../../../assets/catalog_thumbnails/tree.png"),
        ModelCatalogKey::Snow => include_bytes!("../../../../assets/catalog_thumbnails/snow.png"),
        ModelCatalogKey::Digit0 => {
            include_bytes!("../../../../assets/catalog_thumbnails/digit_0.png")
        }
        ModelCatalogKey::LetterA => {
            include_bytes!("../../../../assets/catalog_thumbnails/letter_a.png")
        }
        ModelCatalogKey::Growing2d => {
            include_bytes!("../../../../assets/catalog_thumbnails/growing_2d.png")
        }
        ModelCatalogKey::Texture2d => {
            include_bytes!("../../../../assets/catalog_thumbnails/texture_2d.png")
        }
        ModelCatalogKey::Growing3dGs => {
            include_bytes!("../../../../assets/catalog_thumbnails/growing_3d_gs.png")
        }
        ModelCatalogKey::UvTorusMorphogen3d => {
            include_bytes!("../../../../assets/catalog_thumbnails/uv_torus_morphogen_3d.png")
        }
        ModelCatalogKey::TeapotMorphogen3d => {
            include_bytes!("../../../../assets/catalog_thumbnails/teapot_morphogen_3d.png")
        }
        ModelCatalogKey::PointMnist => {
            include_bytes!("../../../../assets/catalog_thumbnails/point_mnist.png")
        }
    }
}

pub(super) fn catalog_thumbnail_image(key: ModelCatalogKey) -> Image {
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

pub(super) fn catalog_preview_image(key: ModelCatalogKey, seconds: f32) -> Image {
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
