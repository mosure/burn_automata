use crate::cli::prelude::*;

#[derive(Clone, Debug)]
pub(super) struct Hyper2dScratchSource {
    pub(super) slug: String,
    pub(super) title: Option<String>,
    pub(super) group: Option<String>,
    pub(super) condition_path: PathBuf,
    pub(super) particles: Option<usize>,
    pub(super) seed_scale: Option<f32>,
    pub(super) update_prob: Option<f32>,
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

pub(super) fn resolve_scratch_sources(
    preset: PresetArg,
    target_images: &[PathBuf],
    catalog: Option<&PathBuf>,
    catalog_thumbnail_dir: &Path,
    catalog_group: Option<Hyper2dCatalogGroupArg>,
    catalog_targets: &[String],
    catalog_limit: usize,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    if let Some(catalog_path) = catalog {
        if !target_images.is_empty() {
            return Err(std::io::Error::other(
                "--catalog cannot be combined with --target-image for train-hyper2d-e2e",
            )
            .into());
        }
        return resolve_catalog_scratch_sources(
            preset,
            catalog_path,
            catalog_thumbnail_dir,
            catalog_group,
            catalog_targets,
            catalog_limit,
        );
    }
    if catalog_group.is_some() || !catalog_targets.is_empty() || catalog_limit > 0 {
        return Err(std::io::Error::other(
            "catalog filters require --catalog for train-hyper2d-e2e",
        )
        .into());
    }
    if target_images.is_empty() {
        return Err(std::io::Error::other(
            "train-hyper2d-e2e requires --target-image or --catalog",
        )
        .into());
    }
    Ok(target_images
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
        .collect())
}

fn resolve_catalog_scratch_sources(
    preset: PresetArg,
    catalog_path: &Path,
    catalog_thumbnail_dir: &Path,
    catalog_group: Option<Hyper2dCatalogGroupArg>,
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
    preset: PresetArg,
    group: Option<Hyper2dCatalogGroupArg>,
    entry: &SelfOrgCatalogEntry,
) -> bool {
    match group {
        Some(Hyper2dCatalogGroupArg::Growing) => entry.group == "growing",
        Some(Hyper2dCatalogGroupArg::Texture) => entry.group == "texture",
        Some(Hyper2dCatalogGroupArg::All) => true,
        None => entry.preset == preset_name(preset),
    }
}

pub(super) fn preset_name(preset: PresetArg) -> &'static str {
    match preset {
        PresetArg::Growing2d => "growing-2d",
        PresetArg::Texture2d => "texture-2d",
        PresetArg::Growing3dgs => "growing-3d-gs",
        PresetArg::PointMnist => "point-mnist",
    }
}

fn path_slug(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_slug)
        .unwrap_or_else(|| "condition".to_string())
}

pub(super) fn sanitize_slug(value: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_image_sources_reject_catalog_filters() {
        let err = resolve_scratch_sources(
            PresetArg::Growing2d,
            &[PathBuf::from("lizard.png")],
            None,
            Path::new("assets/catalog_thumbnails"),
            Some(Hyper2dCatalogGroupArg::Growing),
            &[],
            0,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("catalog filters require --catalog"));
    }

    #[test]
    fn slug_sanitizer_preserves_file_safe_names() {
        assert_eq!(sanitize_slug("lizard"), "lizard");
        assert_eq!(sanitize_slug("frog face/01"), "frog_face_01");
        assert_eq!(sanitize_slug(""), "example");
    }
}
