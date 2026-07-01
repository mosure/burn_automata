#[cfg(feature = "splatting")]
use bevy::camera::ScalingMode;
#[cfg(feature = "splatting")]
use bevy::camera::primitives::Aabb;
#[cfg(feature = "splatting")]
use bevy::camera::{CameraProjection, Viewport};
#[cfg(test)]
use bevy::render::render_resource::TextureFormat;
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    image::ImageSampler,
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::hover::Hovered,
    prelude::*,
};
use bevy_ui_widgets::{
    Slider, SliderDragState, SliderOrientation, SliderPlugin, SliderRange, SliderStep, SliderThumb,
    SliderValue, TrackClick, ValueChange, slider_self_update,
};
#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
use burn_automata::gpu::WgpuNeighborMode;
use burn_automata::{
    AutomataPreset, NpaConfig, NpaModel, ParticleSeed, RolloutBatchConfig, RolloutConfig,
    RolloutTrace, SgdConfig, SupervisedTarget, kernels::HashGridConfig, rollout_supervised_batch,
    run_rollout, supervised_backward, supervised_loss, supervised_train_step,
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

mod catalog;
mod ui;

use ui::*;

#[cfg(feature = "splatting")]
use catalog::AUTOMATA_MIN_VIEWPORT_WIDTH;
#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
use catalog::GAUSSIAN_SH_C0;
#[cfg(feature = "splatting")]
use catalog::SORTED_ENTRY_MIN_CAPACITY;
use catalog::{
    AUTOMATA_UI_PANEL_WIDTH, BACKWARD_PROBE_PARTICLES, CATALOG_DOUBLE_CLICK_SECONDS,
    DEFAULT_LIZARD_MODEL, LIVE_TRAINING_TARGET, ModelCatalogKey, TRAINING_INTERVAL_FRAMES,
    TRAINING_PROBE_PARTICLES, catalog_entry, catalog_entry_matches_settings, catalog_preview_image,
    catalog_thumbnail_image, select_catalog_entry,
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
    pub model_path: Option<String>,
    pub revision: u64,
}

impl Default for AutomataSettings {
    fn default() -> Self {
        let preset = AutomataPreset::Growing2d;
        let model_path = std::env::var("BURN_AUTOMATA_MODEL").ok().or_else(|| {
            std::path::Path::new(DEFAULT_LIZARD_MODEL)
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
        .and_then(|path| std::path::Path::new(path).file_name())
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
