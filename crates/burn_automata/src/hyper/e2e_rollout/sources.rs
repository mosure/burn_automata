use super::{E2eCatalogGroup, OmniSvgDataset, preset_name};
use std::path::{Path, PathBuf};

mod omnisvg;
pub(crate) use omnisvg::OmniSvgSourceConfig;

#[derive(Clone, Debug)]
pub(crate) struct Hyper2dScratchSource {
    pub(crate) slug: String,
    pub(crate) title: Option<String>,
    pub(crate) group: Option<String>,
    pub(crate) condition_path: PathBuf,
    pub(crate) particles: Option<usize>,
    pub(crate) seed_scale: Option<f32>,
    pub(crate) update_prob: Option<f32>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct SelfOrgCatalogEntry {
    slug: String,
    title: Option<String>,
    group: String,
    preset: String,
    particles: Option<usize>,
    seed_scale: Option<f32>,
    update_prob: Option<f32>,
}

pub(crate) struct ScratchSourceResolveConfig<'a> {
    pub(crate) preset: crate::AutomataPreset,
    pub(crate) target_images: &'a [PathBuf],
    pub(crate) target_image_dirs: &'a [PathBuf],
    pub(crate) target_image_recursive: bool,
    pub(crate) image_extensions: &'a [String],
    pub(crate) catalog: Option<&'a PathBuf>,
    pub(crate) catalog_thumbnail_dir: &'a Path,
    pub(crate) catalog_group: Option<E2eCatalogGroup>,
    pub(crate) catalog_targets: &'a [String],
    pub(crate) catalog_limit: usize,
    pub(crate) omnisvg: Option<OmniSvgSourceConfig<'a>>,
}

pub(crate) fn resolve_scratch_sources(
    config: ScratchSourceResolveConfig<'_>,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    let ScratchSourceResolveConfig {
        preset,
        target_images,
        target_image_dirs,
        target_image_recursive,
        image_extensions,
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        omnisvg,
    } = config;
    let uses_direct_sources = !target_images.is_empty() || !target_image_dirs.is_empty();
    let source_mode_count = usize::from(uses_direct_sources)
        + usize::from(catalog.is_some())
        + usize::from(omnisvg.is_some());
    if source_mode_count > 1 {
        return Err(std::io::Error::other(
            "--target-image/--target-image-dir, --catalog, and --omnisvg-dataset are mutually exclusive source modes for train-hyper2d",
        )
        .into());
    }
    if let Some(catalog_path) = catalog {
        return resolve_catalog_scratch_sources(
            preset,
            catalog_path,
            catalog_thumbnail_dir,
            catalog_group,
            catalog_targets,
            catalog_limit,
        );
    }
    if let Some(omnisvg) = omnisvg {
        if catalog_group.is_some() || !catalog_targets.is_empty() || catalog_limit > 0 {
            return Err(std::io::Error::other(
                "catalog filters require --catalog for train-hyper2d",
            )
            .into());
        }
        return omnisvg::resolve_omnisvg_scratch_sources(omnisvg);
    }
    if catalog_group.is_some() || !catalog_targets.is_empty() || catalog_limit > 0 {
        return Err(
            std::io::Error::other("catalog filters require --catalog for train-hyper2d").into(),
        );
    }
    let mut sources = target_images
        .iter()
        .map(|path| {
            let slug = path_slug(path);
            Hyper2dScratchSource {
                title: Some(slug.clone()),
                slug,
                group: Some("scratch".to_string()),
                condition_path: path.clone(),
                particles: None,
                seed_scale: None,
                update_prob: None,
            }
        })
        .collect::<Vec<_>>();
    let extensions = normalized_image_extensions(image_extensions);
    for dir in target_image_dirs {
        collect_image_dir_sources(dir, dir, target_image_recursive, &extensions, &mut sources)?;
    }
    sources.sort_by(|left, right| left.condition_path.cmp(&right.condition_path));
    sources.dedup_by(|left, right| left.condition_path == right.condition_path);
    if sources.is_empty() {
        return Err(std::io::Error::other(
            "train-hyper2d requires --target-image, --target-image-dir, --catalog, or --omnisvg-dataset",
        )
        .into());
    }
    Ok(sources)
}

#[allow(dead_code)]
pub(super) fn resolve_scratch_sources_legacy(
    preset: crate::AutomataPreset,
    target_images: &[PathBuf],
    catalog: Option<&PathBuf>,
    catalog_thumbnail_dir: &Path,
    catalog_group: Option<E2eCatalogGroup>,
    catalog_targets: &[String],
    catalog_limit: usize,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    resolve_scratch_sources(ScratchSourceResolveConfig {
        preset,
        target_images,
        target_image_dirs: &[],
        target_image_recursive: false,
        image_extensions: &[],
        catalog,
        catalog_thumbnail_dir,
        catalog_group,
        catalog_targets,
        catalog_limit,
        omnisvg: None,
    })
}

fn resolve_catalog_scratch_sources(
    preset: crate::AutomataPreset,
    catalog_path: &Path,
    catalog_thumbnail_dir: &Path,
    catalog_group: Option<E2eCatalogGroup>,
    catalog_targets: &[String],
    catalog_limit: usize,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(catalog_path)?;
    let entries: Vec<SelfOrgCatalogEntry> = serde_json::from_str(&text)?;
    let mut sources = Vec::new();
    for entry in entries {
        if !catalog_targets.is_empty()
            && !catalog_targets.iter().any(|target| target == &entry.slug)
        {
            continue;
        }
        if catalog_targets.is_empty() && !catalog_entry_matches(preset, catalog_group, &entry) {
            continue;
        }
        sources.push(Hyper2dScratchSource {
            slug: entry.slug.clone(),
            title: entry.title,
            group: Some(entry.group),
            condition_path: catalog_thumbnail_dir.join(format!("{}.png", entry.slug)),
            particles: entry.particles,
            seed_scale: entry.seed_scale,
            update_prob: entry.update_prob,
        });
        if catalog_limit > 0 && sources.len() >= catalog_limit {
            break;
        }
    }
    Ok(sources)
}

fn catalog_entry_matches(
    preset: crate::AutomataPreset,
    group: Option<E2eCatalogGroup>,
    entry: &SelfOrgCatalogEntry,
) -> bool {
    match group {
        Some(E2eCatalogGroup::Growing) => entry.group == "growing",
        Some(E2eCatalogGroup::Texture) => entry.group == "texture",
        Some(E2eCatalogGroup::All) => true,
        None => entry.preset == preset_name(preset),
    }
}

fn path_slug(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_slug)
        .unwrap_or_else(|| "condition".to_string())
}

fn relative_path_slug(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let stem_path = relative.with_extension("");
    sanitize_slug(&stem_path.to_string_lossy())
}

pub(crate) fn sanitize_slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if slug.is_empty() {
        "example".to_string()
    } else {
        slug
    }
}

fn normalized_image_extensions(values: &[String]) -> std::collections::BTreeSet<String> {
    let values = if values.is_empty() {
        ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
    } else {
        values.to_vec()
    };
    values
        .into_iter()
        .map(|value| value.trim_start_matches('.').to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn collect_image_dir_sources(
    root: &Path,
    dir: &Path,
    recursive: bool,
    extensions: &std::collections::BTreeSet<String>,
    sources: &mut Vec<Hyper2dScratchSource>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries = std::fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect_image_dir_sources(root, &path, recursive, extensions, sources)?;
            }
            continue;
        }
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };
        if !extensions.contains(&extension.to_ascii_lowercase()) {
            continue;
        }
        let slug = relative_path_slug(root, &path);
        sources.push(Hyper2dScratchSource {
            title: Some(slug.clone()),
            slug,
            group: Some("image-dir".to_string()),
            condition_path: path,
            particles: None,
            seed_scale: None,
            update_prob: None,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_image_sources_reject_catalog_filters() {
        let err = resolve_scratch_sources(ScratchSourceResolveConfig {
            preset: crate::AutomataPreset::Growing2d,
            target_images: &[PathBuf::from("lizard.png")],
            target_image_dirs: &[],
            target_image_recursive: false,
            image_extensions: &[],
            catalog: None,
            catalog_thumbnail_dir: Path::new("assets/catalog_thumbnails"),
            catalog_group: Some(E2eCatalogGroup::Growing),
            catalog_targets: &[],
            catalog_limit: 0,
            omnisvg: None,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("catalog filters require --catalog"));
    }

    #[test]
    fn omnisvg_sources_reject_mixed_source_modes() {
        let err = resolve_scratch_sources(ScratchSourceResolveConfig {
            preset: crate::AutomataPreset::Growing2d,
            target_images: &[PathBuf::from("lizard.png")],
            target_image_dirs: &[],
            target_image_recursive: false,
            image_extensions: &[],
            catalog: None,
            catalog_thumbnail_dir: Path::new("assets/catalog_thumbnails"),
            catalog_group: None,
            catalog_targets: &[],
            catalog_limit: 0,
            omnisvg: Some(OmniSvgSourceConfig {
                dataset: OmniSvgDataset::MmsvgIllustration,
                split: "train",
                cache_dir: Path::new("data/omnisvg"),
                offset: 0,
                limit: 1,
                page_size: 100,
                download: false,
                refresh: false,
                token_env: "HF_TOKEN",
            }),
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("mutually exclusive source modes"));
    }

    #[test]
    fn direct_image_sources_collect_directory_images_deterministically() {
        let root =
            std::env::temp_dir().join(format!("burn_automata_sources_{}", std::process::id()));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("b.png"), []).unwrap();
        std::fs::write(nested.join("a.jpg"), []).unwrap();
        std::fs::write(nested.join("ignore.svg"), []).unwrap();

        let flat = resolve_scratch_sources(ScratchSourceResolveConfig {
            preset: crate::AutomataPreset::Growing2d,
            target_images: &[],
            target_image_dirs: std::slice::from_ref(&root),
            target_image_recursive: false,
            image_extensions: &[],
            catalog: None,
            catalog_thumbnail_dir: Path::new("assets/catalog_thumbnails"),
            catalog_group: None,
            catalog_targets: &[],
            catalog_limit: 0,
            omnisvg: None,
        })
        .unwrap();
        let recursive = resolve_scratch_sources(ScratchSourceResolveConfig {
            preset: crate::AutomataPreset::Growing2d,
            target_images: &[],
            target_image_dirs: std::slice::from_ref(&root),
            target_image_recursive: true,
            image_extensions: &[],
            catalog: None,
            catalog_thumbnail_dir: Path::new("assets/catalog_thumbnails"),
            catalog_group: None,
            catalog_targets: &[],
            catalog_limit: 0,
            omnisvg: None,
        })
        .unwrap();

        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].slug, "b");
        assert_eq!(
            recursive
                .iter()
                .map(|source| source.slug.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "nested_a"]
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn slug_sanitizer_preserves_file_safe_names() {
        assert_eq!(sanitize_slug("lizard"), "lizard");
        assert_eq!(sanitize_slug("frog face/01"), "frog_face_01");
        assert_eq!(sanitize_slug(""), "example");
    }
}
