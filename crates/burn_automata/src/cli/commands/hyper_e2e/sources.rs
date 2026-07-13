use crate::cli::prelude::*;
use crate::hyper::e2e_rollout::{E2eCatalogGroup, OmniSvgDataset, sources as shared};

pub(super) use shared::{Hyper2dScratchSource, sanitize_slug};

#[derive(Clone, Copy, Debug)]
pub(super) struct OmniSvgSourceConfig<'a> {
    pub(super) dataset: OmniSvgDatasetArg,
    pub(super) split: &'a str,
    pub(super) cache_dir: &'a Path,
    pub(super) offset: usize,
    pub(super) limit: usize,
    pub(super) page_size: usize,
    pub(super) download: bool,
    pub(super) refresh: bool,
    pub(super) token_env: &'a str,
}

pub(super) struct ScratchSourceResolveConfig<'a> {
    pub(super) preset: PresetArg,
    pub(super) target_images: &'a [PathBuf],
    pub(super) target_image_dirs: &'a [PathBuf],
    pub(super) target_image_recursive: bool,
    pub(super) image_extensions: &'a [String],
    pub(super) catalog: Option<&'a PathBuf>,
    pub(super) catalog_thumbnail_dir: &'a Path,
    pub(super) catalog_group: Option<Hyper2dCatalogGroupArg>,
    pub(super) catalog_targets: &'a [String],
    pub(super) catalog_limit: usize,
    pub(super) omnisvg: Option<OmniSvgSourceConfig<'a>>,
}

pub(super) fn resolve_scratch_sources(
    config: ScratchSourceResolveConfig<'_>,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    let omnisvg = config.omnisvg.map(|source| shared::OmniSvgSourceConfig {
        dataset: match source.dataset {
            OmniSvgDatasetArg::MmsvgIllustration => OmniSvgDataset::MmsvgIllustration,
            OmniSvgDatasetArg::MmsvgIcon => OmniSvgDataset::MmsvgIcon,
        },
        split: source.split,
        cache_dir: source.cache_dir,
        offset: source.offset,
        limit: source.limit,
        page_size: source.page_size,
        download: source.download,
        refresh: source.refresh,
        token_env: source.token_env,
    });
    let catalog_group = config.catalog_group.map(|group| match group {
        Hyper2dCatalogGroupArg::Growing => E2eCatalogGroup::Growing,
        Hyper2dCatalogGroupArg::Texture => E2eCatalogGroup::Texture,
        Hyper2dCatalogGroupArg::All => E2eCatalogGroup::All,
    });
    shared::resolve_scratch_sources(shared::ScratchSourceResolveConfig {
        preset: config.preset.into(),
        target_images: config.target_images,
        target_image_dirs: config.target_image_dirs,
        target_image_recursive: config.target_image_recursive,
        image_extensions: config.image_extensions,
        catalog: config.catalog,
        catalog_thumbnail_dir: config.catalog_thumbnail_dir,
        catalog_group,
        catalog_targets: config.catalog_targets,
        catalog_limit: config.catalog_limit,
        omnisvg,
    })
}

pub(super) const fn preset_name(preset: PresetArg) -> &'static str {
    match preset {
        PresetArg::Growing2d => "growing-2d",
        PresetArg::Texture2d => "texture-2d",
        PresetArg::Growing3dgs => "growing-3d-gs",
        PresetArg::PointMnist => "point-mnist",
    }
}
