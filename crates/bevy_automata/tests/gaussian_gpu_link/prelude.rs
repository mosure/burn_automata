pub(crate) use bevy::{
    app::SubApps,
    asset::RenderAssetUsages,
    camera::primitives::Aabb,
    camera::{RenderTarget, Viewport},
    core_pipeline::core_3d::Transparent3d,
    diagnostic::FrameTimeDiagnosticsPlugin,
    image::Image,
    prelude::*,
    render::{
        RenderPlugin,
        render_phase::ViewSortedRenderPhases,
        render_resource::{
            BufferDescriptor, BufferUsages, Extent3d, PollType, TextureDimension, TextureFormat,
            TextureUsages,
        },
        renderer::RenderDevice,
        view::{
            ExtractedView, RenderVisibleEntities,
            screenshot::{Screenshot, ScreenshotCaptured},
        },
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
pub(crate) use bevy_automata::{
    AutomataRenderDiagnostics, AutomataRuntime, AutomataSettings, AutomataViewerPlugin,
    gaussian_storage_buffer_refs,
};
pub(crate) use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin, Planar,
    PlanarGaussian3d, PlanarGaussian3dHandle, PlanarStorageBindGroup,
    SphericalHarmonicCoefficients,
    gaussian::cloud::CloudVisibilityClass,
    gaussian::formats::planar_3d::PlanarStorageGaussian3d,
    render::SortBindGroup,
    sort::{SortMode, SortTrigger, SortedEntries, SortedEntriesHandle},
};
pub(crate) use bevy_panorbit_camera::PanOrbitCameraPlugin;
pub(crate) use burn_automata::{
    AutomataError, AutomataPreset, NpaConfig, NpaModel, ParticleSeed,
    gpu::{GAUSSIAN_SH_COEFF_COUNT, WgpuAutomataExecutor, WgpuGaussianReadback, WgpuNeighborMode},
    import::load_manifest,
    kernels::HashGridConfig,
    rollout::seed_particles_scaled,
};
pub(crate) use std::{
    path::Path,
    sync::MutexGuard,
    time::{Duration, Instant},
};
