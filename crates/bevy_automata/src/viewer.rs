#[cfg(feature = "splatting")]
use bevy::camera::ScalingMode;
#[cfg(feature = "splatting")]
use bevy::camera::primitives::Aabb;
#[cfg(feature = "splatting")]
use bevy::camera::{CameraProjection, Viewport};
#[cfg(test)]
use bevy::render::render_resource::TextureFormat;
#[cfg(feature = "hyper_dino")]
use bevy::ui::Checked;
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    image::ImageSampler,
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::hover::Hovered,
    prelude::*,
    time::Real,
};
#[cfg(feature = "hyper_dino")]
use bevy_ui_widgets::Checkbox;
use bevy_ui_widgets::{
    CheckboxPlugin, Slider, SliderDragState, SliderOrientation, SliderPlugin, SliderRange,
    SliderStep, SliderThumb, SliderValue, TrackClick, ValueChange, slider_self_update,
};
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use burn_automata::gpu::WgpuNeighborMode;
#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
use burn_automata::rollout::growth_3d_material_opacity_channel;
use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutBatchConfig, RolloutConfig,
    RolloutTrace, SgdConfig, SupervisedTarget, Target2dTrainingConfig, kernels::HashGridConfig,
    rollout_supervised_batch, run_rollout, supervised_backward, supervised_loss,
    supervised_train_step,
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
#[cfg(all(
    feature = "splatting",
    any(not(feature = "gpu_wgpu"), feature = "headless", test)
))]
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

mod adaptive;
mod camera;
mod catalog;
mod catalog_images;
mod cloud;
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
mod gpu_bridge;
#[cfg(all(feature = "headless", feature = "splatting", feature = "gpu_wgpu"))]
pub mod headless;
#[cfg(feature = "hyper_dino")]
mod hyper_inference;
#[cfg(feature = "hyper_dino")]
mod image_training;
mod runtime;
mod ui;
#[cfg(target_arch = "wasm32")]
mod web;

use adaptive::*;
use camera::*;
use cloud::*;
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use gpu_bridge::*;
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
pub use gpu_bridge::{
    AutomataRenderDiagnostics, automata_executor_from_render_device, gaussian_storage_buffer_refs,
};
#[cfg(feature = "hyper_dino")]
use hyper_inference::*;
#[cfg(feature = "hyper_dino")]
use image_training::*;
use runtime::*;
use ui::*;

#[cfg(feature = "splatting")]
use catalog::AUTOMATA_MIN_VIEWPORT_WIDTH;
#[cfg(all(
    feature = "splatting",
    any(not(feature = "gpu_wgpu"), feature = "headless", test)
))]
use catalog::GAUSSIAN_SH_C0;
#[cfg(feature = "splatting")]
use catalog::SORTED_ENTRY_MIN_CAPACITY;
use catalog::{
    AUTOMATA_UI_PANEL_WIDTH, BACKWARD_PROBE_PARTICLES, CATALOG_DOUBLE_CLICK_SECONDS,
    DEFAULT_LIZARD_MODEL, LIVE_TRAINING_TARGET, ModelCatalogKey, TRAINING_INTERVAL_FRAMES,
    TRAINING_PROBE_PARTICLES, catalog_entry, catalog_entry_is_available,
    catalog_entry_matches_settings, catalog_preview_image, catalog_thumbnail_image,
    select_catalog_entry,
};
#[cfg(test)]
use catalog::{
    CATALOG_3D_GROWTH_SEED, MODEL_CATALOG, ModelCatalogSource, VISIBLE_MODEL_CATALOG_KEYS,
    catalog_seed_mode, catalog_thumbnail_png, resolved_catalog_model_path,
};

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
    pub training_rollout_reset_interval: usize,
    pub adaptive_training_enabled: bool,
    pub model_path: Option<String>,
    pub adaptive_model_path: Option<String>,
    pub adaptive_bandwidth_enabled: bool,
    pub adaptive_topology_enabled: bool,
    pub generated_model_label: Option<String>,
    pub hyper_base_model_path: Option<String>,
    pub hyper_model_path: Option<String>,
    pub hyper_dino_model_path: Option<String>,
    pub hyper_dino_image_size: usize,
    pub hyper_dino_patch_size: usize,
    pub revision: u64,
}

impl Default for AutomataSettings {
    fn default() -> Self {
        let preset = AutomataPreset::Growing2d;
        let model_path = default_model_path();
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
            training_learning_rate: Target2dTrainingConfig::default().optimizer.learning_rate,
            training_rollout_reset_interval: 100,
            adaptive_training_enabled: false,
            model_path,
            adaptive_model_path: std::env::var("BURN_AUTOMATA_ADAPTIVE_MODEL").ok(),
            adaptive_bandwidth_enabled: true,
            adaptive_topology_enabled: true,
            generated_model_label: None,
            hyper_base_model_path: env_or_workspace_path(
                "BURN_AUTOMATA_HYPER_E2E_BASE",
                "artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda/shared_base.bpk",
            ),
            hyper_model_path: env_or_workspace_path(
                "BURN_AUTOMATA_HYPER_E2E_MODEL",
                "artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda/hyper_2d.bpk",
            ),
            hyper_dino_model_path: env_or_workspace_path(
                "BURN_AUTOMATA_DINO_MODEL",
                "models/dino/dino_vits.mpk",
            ),
            hyper_dino_image_size: 224,
            hyper_dino_patch_size: 14,
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

fn env_or_workspace_path(env_key: &str, workspace_relative: &str) -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = env_key;
        Some(workspace_relative.to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(env_key).ok().or_else(|| {
            resolve_workspace_path(workspace_relative).map(|path| path.display().to_string())
        })
    }
}

fn default_model_path() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        Some(DEFAULT_LIZARD_MODEL.to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("BURN_AUTOMATA_MODEL").ok().or_else(|| {
            std::path::Path::new(DEFAULT_LIZARD_MODEL)
                .exists()
                .then(|| DEFAULT_LIZARD_MODEL.to_string())
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_workspace_path(path: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(path);
    if direct.exists() {
        return Some(direct.to_path_buf());
    }
    let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    workspace_path.exists().then_some(workspace_path)
}

#[derive(Resource, Clone, Debug)]
pub struct AutomataRuntime {
    pub model: NpaModel,
    pub hashgrid: HashGridConfig,
    pub trace: Option<RolloutTrace>,
    pub frame: usize,
    pub status: String,
    pub loaded_model_path: Option<String>,
    pub loaded_adaptive_model_path: Option<String>,
    pub adaptive: Option<AdaptiveViewerState>,
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
            loaded_adaptive_model_path: None,
            adaptive: None,
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

pub struct AutomataViewerPlugin;

impl Plugin for AutomataViewerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AutomataSettings>()
            .init_resource::<AutomataRuntime>()
            .init_resource::<AutomataUiState>()
            .init_resource::<AutomataPerformanceTelemetry>()
            .init_resource::<PerformanceUiState>()
            .init_resource::<CatalogPreviewState>()
            .init_resource::<CatalogPreviewImageState>();
        #[cfg(target_arch = "wasm32")]
        app.init_resource::<BrowserModelLoadChannel>()
            .init_resource::<BrowserModelLoadState>();
        #[cfg(feature = "hyper_dino")]
        app.init_resource::<HyperNpaImageDialogChannel>()
            .init_resource::<HyperNpaInferenceChannel>()
            .init_resource::<HyperNpaInferenceState>()
            .init_resource::<ImageTargetTrainingChannel>()
            .init_resource::<ImageTargetTrainingState>()
            .init_resource::<ImageTargetPreviewState>()
            .add_message::<OpenHyperNpaImage>()
            .add_message::<RunHyperNpaInference>()
            .add_message::<ToggleImageTargetTraining>();
        #[cfg(all(feature = "hyper_dino", target_arch = "wasm32"))]
        app.init_non_send::<BrowserTrainingWorker>();
        #[cfg(feature = "splatting")]
        app.init_resource::<AutomataCloudState>();
        #[cfg(feature = "splatting")]
        app.init_resource::<AutomataUiInputCapture>();
        app.add_plugins((SliderPlugin, CheckboxPlugin));

        app.add_systems(
            Startup,
            (
                scene.spawn(),
                load_selected_adaptive_model,
                load_selected_model,
                setup_gaussian_cloud,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                (
                    load_selected_adaptive_model,
                    load_selected_model,
                    toggle_ui_visibility,
                    sync_view_cameras,
                    sync_automata_camera_viewports,
                    sync_gaussian_cloud_asset,
                    restore_resized_gaussian_cloud_visibility,
                    sync_gaussian_cloud_settings,
                    advance_rollout,
                    advance_adaptive_viewer,
                    sync_cpu_trace_to_gaussian_asset,
                    sync_adaptive_particles_to_gaussian_asset,
                )
                    .chain(),
                (
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
                    update_performance_labels,
                    update_settings_label,
                )
                    .chain(),
            )
                .chain(),
        );

        #[cfg(feature = "hyper_dino")]
        app.add_systems(
            Update,
            (
                handle_open_hyper_npa_image_dialog,
                handle_hyper_npa_image_drop,
                poll_hyper_npa_image_sources,
                handle_run_hyper_npa_inference,
                poll_hyper_npa_inference_results,
                handle_toggle_image_target_training,
                poll_image_target_training,
                #[cfg(target_arch = "wasm32")]
                stop_stale_browser_training,
                sync_image_target_summary,
                sync_image_training_button_label,
                sync_adaptive_training_checkbox,
                sync_image_target_preview,
                update_adaptive_training_checkbox_style,
                update_hyper_image_button_styles,
                update_hyper_inference_button_styles,
                sync_hyper_inference_button_label,
            )
                .chain()
                .before(sync_gaussian_cloud_asset),
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

pub fn run() {
    run_with_settings(AutomataSettings::default());
}

pub fn run_with_settings(settings: AutomataSettings) {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.015, 0.018, 0.022)));
    app.insert_resource(settings);
    app.add_plugins(DefaultPlugins);
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    #[cfg(feature = "splatting")]
    app.add_plugins((GaussianSplattingPlugin, PanOrbitCameraPlugin));
    app.add_plugins(AutomataViewerPlugin);
    app.run();
}

#[cfg(test)]
mod tests;
