#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use burn_automata::{AutomataPreset, ParticleSeed, RolloutConfig};

use super::{AutomataRuntime, AutomataSettings, apply_preset, reset_training_stats};

pub(super) const DEFAULT_LIZARD_MODEL: &str = "models/catalog/growing/lizard.bpk";
const LEGACY_LIZARD_MODEL: &str = "/tmp/burn_automata_lizard.bpk";
const DEFAULT_POLKA_MODEL: &str = "models/catalog/texture/polka_dotted_0121.bpk";
const LEGACY_POLKA_MODEL: &str = "/tmp/burn_automata_polka.bpk";
pub(super) const BACKWARD_PROBE_PARTICLES: usize = 1024;
pub(super) const TRAINING_PROBE_PARTICLES: usize = 256;
pub(super) const TRAINING_INTERVAL_FRAMES: usize = 60;
pub(super) const LIVE_TRAINING_TARGET: &str = "rollout teacher";
pub(super) const CATALOG_DOUBLE_CLICK_SECONDS: f64 = 0.35;
pub(super) const CATALOG_3D_GROWTH_SEED: u64 = 0x0051_a73d;
pub(super) const AUTOMATA_UI_PANEL_WIDTH: f32 = 540.0;
#[cfg(feature = "splatting")]
pub(super) const AUTOMATA_MIN_VIEWPORT_WIDTH: u32 = 256;
#[cfg(feature = "splatting")]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
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
            fallback: Some(LEGACY_LIZARD_MODEL),
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
            primary: "models/catalog/growing/butterfly.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/butterfly.bpk"),
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
            primary: "models/catalog/growing/rose.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/rose.bpk"),
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
            primary: "models/catalog/growing/turtle.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/turtle.bpk"),
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
            primary: "models/catalog/growing/mushroom.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/mushroom.bpk"),
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
            primary: "models/catalog/growing/tropical_fish.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/tropical_fish.bpk"),
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
            primary: "models/catalog/growing/sun_with_face.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/sun_with_face.bpk"),
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
            primary: "models/catalog/growing/ghost.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/ghost.bpk"),
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
            primary: "models/catalog/growing/frog_face.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/frog_face.bpk"),
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
            primary: "models/catalog/growing/red_apple.bpk",
            fallback: Some("/tmp/burn_automata_catalog/growing/red_apple.bpk"),
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
            fallback: Some(LEGACY_POLKA_MODEL),
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
            primary: "models/catalog/texture/bubbly_0101.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/bubbly_0101.bpk"),
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
            primary: "models/catalog/texture/clouds.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/clouds.bpk"),
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
            primary: "models/catalog/texture/galaxy.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/galaxy.bpk"),
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
            primary: "models/catalog/texture/hearts.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/hearts.bpk"),
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
            primary: "models/catalog/texture/rings.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/rings.bpk"),
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
            primary: "models/catalog/texture/stars.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/stars.bpk"),
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
            primary: "models/catalog/texture/grid_0040.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/grid_0040.bpk"),
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
            primary: "models/catalog/texture/banded_0037.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/banded_0037.bpk"),
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
            primary: "models/catalog/texture/tree.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/tree.bpk"),
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
            primary: "models/catalog/texture/snow.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/snow.bpk"),
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
            primary: "models/catalog/texture/digit_0.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/digit_0.bpk"),
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
            primary: "models/catalog/texture/letter_a.bpk",
            fallback: Some("/tmp/burn_automata_catalog/texture/letter_a.bpk"),
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
        kind: "hidden local regression",
        detail: "strict promotion blocked: random-ball lineage and geometry gates",
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
        kind: "hidden local regression",
        detail: "strict promotion blocked: random-ball lineage and held-out geometry gates",
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
    if settings.generated_model_label.is_some() {
        return false;
    }
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

pub(super) fn catalog_entry_is_available(entry: &ModelCatalogEntry) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = entry;
        true
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        matches!(entry.source, ModelCatalogSource::Preset)
            || resolved_catalog_model_path(entry).is_some()
    }
}

pub(super) fn missing_catalog_model_status(entry: &ModelCatalogEntry) -> String {
    format!(
        "missing {} model file {}; import catalog BPKs before selection",
        entry.title,
        catalog_primary_model_path(entry).unwrap_or("unknown")
    )
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
                runtime.status = missing_catalog_model_status(entry);
                return;
            }
        },
    };

    settings.model_path = next_model_path;
    settings.adaptive_model_path = None;
    settings.generated_model_label = None;
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
    runtime.loaded_adaptive_model_path = None;
    runtime.adaptive = None;
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
            #[cfg(target_arch = "wasm32")]
            {
                let _ = fallback;
                Some(primary.to_string())
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                resolve_catalog_path(primary).or_else(|| fallback.and_then(resolve_catalog_path))
            }
        }
    }
}

fn catalog_primary_model_path(entry: &ModelCatalogEntry) -> Option<&'static str> {
    match entry.source {
        ModelCatalogSource::Preset => None,
        ModelCatalogSource::Bpk { primary, .. } => Some(primary),
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(test)]
pub(super) use super::catalog_images::catalog_thumbnail_png;
pub(super) use super::catalog_images::{catalog_preview_image, catalog_thumbnail_image};
