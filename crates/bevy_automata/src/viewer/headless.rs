use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    time::Instant,
};

use bevy::{
    app::SubApps,
    asset::RenderAssetUsages,
    camera::{RenderTarget, Viewport},
    diagnostic::FrameTimeDiagnosticsPlugin,
    image::Image,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, PollType, TextureDimension, TextureFormat, TextureUsages},
        renderer::RenderDevice,
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{GaussianCamera, GaussianSplattingPlugin};
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use burn_automata::{AutomataPreset, NpaConfig, ParticleSeed, RolloutConfig};
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadlessExportConfig {
    pub output_dir: PathBuf,
    pub output_prefix: String,
    pub width: u32,
    pub height: u32,
    pub particles: usize,
    pub steps: usize,
    pub capture_every: Option<usize>,
    pub capture_steps: Vec<usize>,
    pub warmup_frames: usize,
    pub steps_per_frame: usize,
    pub preset: AutomataPreset,
    pub seed_mode: ParticleSeed,
    pub model_path: Option<PathBuf>,
    pub hyper_image_path: Option<PathBuf>,
    pub hyper_base_model_path: Option<PathBuf>,
    pub hyper_model_path: Option<PathBuf>,
    pub dino_model_path: Option<PathBuf>,
    pub dino_image_size: usize,
    pub dino_patch_size: usize,
    pub seed: u64,
    pub seed_scale: Option<f32>,
    pub update_prob: f32,
    pub dt: f32,
    pub render_scale: f32,
    pub render_opacity: f32,
}

impl Default for HeadlessExportConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("target/bevy_automata_headless"),
            output_prefix: "automata".to_string(),
            width: 512,
            height: 512,
            particles: 4096,
            steps: 128,
            capture_every: None,
            capture_steps: Vec::new(),
            warmup_frames: 8,
            steps_per_frame: 1,
            preset: AutomataPreset::Growing2d,
            seed_mode: ParticleSeed::UniformCircle,
            model_path: None,
            hyper_image_path: None,
            hyper_base_model_path: None,
            hyper_model_path: None,
            dino_model_path: None,
            dino_image_size: 518,
            dino_patch_size: 14,
            seed: RolloutConfig::default().seed,
            seed_scale: None,
            update_prob: RolloutConfig::default().update_prob,
            dt: RolloutConfig::default().dt,
            render_scale: 0.5,
            render_opacity: 2.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadlessExportReport {
    pub output_dir: PathBuf,
    pub report_path: PathBuf,
    pub config: HeadlessExportConfig,
    pub model_source: String,
    pub requested_steps: Vec<usize>,
    pub captures: Vec<HeadlessExportRecord>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadlessExportRecord {
    pub requested_step: usize,
    pub actual_step: usize,
    pub path: PathBuf,
    pub particles: usize,
    pub width: u32,
    pub height: u32,
    pub metrics: CaptureMetrics,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CaptureMetrics {
    pub width: u32,
    pub height: u32,
    pub lit_pixels: usize,
    pub max_delta: u8,
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}

impl CaptureMetrics {
    pub fn bbox_width(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_x - self.min_x + 1
        }
    }

    pub fn bbox_height(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_y - self.min_y + 1
        }
    }

    pub fn occupancy(&self) -> f32 {
        self.lit_pixels as f32 / (self.width.max(1) * self.height.max(1)) as f32
    }
}

#[derive(Resource, Default)]
struct HeadlessImageCapture {
    captured: bool,
    metrics: Option<CaptureMetrics>,
    image: Option<Image>,
}

pub fn run_headless_export(
    mut config: HeadlessExportConfig,
) -> Result<HeadlessExportReport, Box<dyn std::error::Error>> {
    validate_config(&config)?;
    if config.model_path.is_some() && config.hyper_image_path.is_some() {
        return Err(std::io::Error::other(
            "--model and --hyper-image are mutually exclusive; --hyper-image uses --hyper-base",
        )
        .into());
    }

    let started = Instant::now();
    config.output_dir = resolve_output_dir(&config.output_dir);
    std::fs::create_dir_all(&config.output_dir)?;
    let requested_steps =
        planned_capture_steps(config.steps, config.capture_every, &config.capture_steps)?;
    #[cfg(feature = "hyper_dino")]
    let mut settings = settings_from_config(&config)?;
    #[cfg(not(feature = "hyper_dino"))]
    let settings = settings_from_config(&config)?;
    #[cfg(feature = "hyper_dino")]
    let generated = generate_hyper_model_if_requested(&config, &mut settings)?;
    #[cfg(not(feature = "hyper_dino"))]
    if config.hyper_image_path.is_some() {
        return Err(std::io::Error::other(
            "headless --hyper-image requires the hyper_dino, hyper_dino_wgpu, or hyper_dino_cuda feature",
        )
        .into());
    }
    let model_source = model_source_label(&config, &settings);

    let mut apps = headless_automata_viewer(settings);
    #[cfg(feature = "hyper_dino")]
    if let Some((label, generated)) = generated {
        install_generated_model(&mut apps, label, generated);
    }

    let target = add_render_target(&mut apps, config.width, config.height);
    pump_headless_frame(&mut apps);
    assign_render_target_to_gaussian_cameras(&mut apps, target.clone());
    for _ in 0..config.warmup_frames {
        pump_headless_frame(&mut apps);
    }

    let mut captures = Vec::with_capacity(requested_steps.len());
    for requested_step in &requested_steps {
        set_paused(&mut apps, false);
        while current_frame(&apps) < *requested_step {
            pump_headless_frame(&mut apps);
        }
        assign_render_target_to_gaussian_cameras(&mut apps, target.clone());
        set_paused(&mut apps, true);
        let actual_step = current_frame(&apps);
        let path = config
            .output_dir
            .join(format!("{}_step{actual_step:06}.png", config.output_prefix));
        let metrics = capture_target_png(&mut apps, &target, &path)?;
        captures.push(HeadlessExportRecord {
            requested_step: *requested_step,
            actual_step,
            path,
            particles: config.particles,
            width: config.width,
            height: config.height,
            metrics,
        });
    }

    let report_path = config
        .output_dir
        .join(format!("{}_report.json", config.output_prefix));
    let report = HeadlessExportReport {
        output_dir: config.output_dir.clone(),
        report_path: report_path.clone(),
        config,
        model_source,
        requested_steps,
        captures,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn validate_config(config: &HeadlessExportConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.width == 0 || config.height == 0 {
        return Err(
            std::io::Error::other("headless export width and height must be positive").into(),
        );
    }
    if config.particles == 0 {
        return Err(std::io::Error::other("headless export particles must be positive").into());
    }
    if config.steps_per_frame == 0 {
        return Err(
            std::io::Error::other("headless export steps-per-frame must be positive").into(),
        );
    }
    if config.capture_every == Some(0) {
        return Err(std::io::Error::other("headless export capture-every must be positive").into());
    }
    if !(0.0..=1.0).contains(&config.update_prob) {
        return Err(std::io::Error::other("headless export update-prob must be in [0, 1]").into());
    }
    if config.dino_image_size == 0 || config.dino_patch_size == 0 {
        return Err(std::io::Error::other("DINO image and patch sizes must be positive").into());
    }
    Ok(())
}

fn planned_capture_steps(
    total_steps: usize,
    capture_every: Option<usize>,
    explicit_steps: &[usize],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut steps = BTreeSet::new();
    if explicit_steps.is_empty() {
        if let Some(stride) = capture_every {
            if stride == 0 {
                return Err(std::io::Error::other("capture-every must be positive").into());
            }
            let mut step = stride;
            while step <= total_steps {
                steps.insert(step);
                step = step.saturating_add(stride);
                if step == usize::MAX {
                    break;
                }
            }
            steps.insert(total_steps);
        } else {
            steps.insert(total_steps);
        }
    } else {
        steps.extend(
            explicit_steps
                .iter()
                .copied()
                .filter(|step| *step <= total_steps),
        );
        if steps.is_empty() {
            return Err(std::io::Error::other(
                "capture-steps did not include any step within --steps",
            )
            .into());
        }
    }
    Ok(steps.into_iter().collect())
}

fn settings_from_config(
    config: &HeadlessExportConfig,
) -> Result<AutomataSettings, Box<dyn std::error::Error>> {
    let seed_scale = config
        .seed_scale
        .unwrap_or_else(|| NpaConfig::seed_scale_for_preset(config.preset));
    let mut settings = AutomataSettings {
        preset: config.preset,
        steps_per_frame: config.steps_per_frame,
        particle_count: config.particles,
        update_prob: config.update_prob,
        dt: config.dt,
        seed: config.seed,
        seed_scale,
        reference_seed_scale: seed_scale,
        seed_mode: config.seed_mode,
        render_scale: config.render_scale,
        render_opacity: config.render_opacity,
        paused: true,
        train_live: false,
        visualize_backward: false,
        ..Default::default()
    };

    if let Some(model_path) = &config.model_path {
        settings.model_path = Some(
            resolve_existing_path(model_path, "model")?
                .display()
                .to_string(),
        );
    } else if config.hyper_image_path.is_some() {
        settings.model_path = None;
    }
    if let Some(path) = &config.hyper_base_model_path {
        settings.hyper_base_model_path = Some(resolve_optional_path(path).display().to_string());
    }
    if let Some(path) = &config.hyper_model_path {
        settings.hyper_model_path = Some(resolve_optional_path(path).display().to_string());
    }
    if let Some(path) = &config.dino_model_path {
        settings.hyper_dino_model_path = Some(resolve_optional_path(path).display().to_string());
    }
    settings.hyper_dino_image_size = config.dino_image_size;
    settings.hyper_dino_patch_size = config.dino_patch_size;
    Ok(settings)
}

#[cfg(feature = "hyper_dino")]
fn generate_hyper_model_if_requested(
    config: &HeadlessExportConfig,
    settings: &mut AutomataSettings,
) -> Result<Option<(String, GeneratedHyperNpaModel)>, Box<dyn std::error::Error>> {
    let Some(image_path) = config.hyper_image_path.as_ref() else {
        return Ok(None);
    };
    let image_path = resolve_existing_path(image_path, "condition image")?;
    let generated = generate_hyper_npa_model_from_image_path(&image_path, settings)?;
    settings.generated_model_label = Some(format!(
        "hyper {}",
        image_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
    ));
    settings.preset = AutomataPreset::Growing2d;
    settings.seed_mode = ParticleSeed::UniformCircle;
    settings.reference_seed_scale = settings.seed_scale;
    settings.model_path = None;
    Ok(Some((
        settings.generated_model_label.clone().unwrap(),
        generated,
    )))
}

#[cfg(feature = "hyper_dino")]
fn install_generated_model(apps: &mut SubApps, label: String, generated: GeneratedHyperNpaModel) {
    let world = apps.main.world_mut();
    {
        let mut settings = world.resource_mut::<AutomataSettings>();
        settings.model_path = None;
        settings.generated_model_label = Some(label);
        settings.preset = AutomataPreset::Growing2d;
        settings.seed_mode = ParticleSeed::UniformCircle;
        settings.mark_changed();
    }
    {
        let mut runtime = world.resource_mut::<AutomataRuntime>();
        runtime.model = generated.model;
        runtime.hashgrid = generated.hashgrid;
        runtime.loaded_model_path = None;
        runtime.loaded_preset = None;
        runtime.trace = None;
        runtime.frame = 0;
        runtime.backward_loss = None;
        runtime.backward_grad_norm = None;
        reset_training_stats(&mut runtime);
        runtime.model_revision = runtime.model_revision.wrapping_add(1);
        runtime.status = format!(
            "generated HyperNPA | image {}x{} | LoRA r{} a{:.1} | {} tokens x {} dims",
            generated.image_width,
            generated.image_height,
            generated.adapter_rank,
            generated.adapter_alpha,
            generated.token_count,
            generated.embed_dims
        );
    }
}

fn headless_automata_viewer(settings: AutomataSettings) -> SubApps {
    let render_plugin = RenderPlugin {
        synchronous_pipeline_compilation: true,
        ..default()
    };
    let window_plugin = WindowPlugin {
        primary_window: None,
        exit_condition: ExitCondition::DontExit,
        ..default()
    };
    let mut app = App::new();
    app.insert_resource(settings)
        .insert_resource(ClearColor(Color::srgb(0.015, 0.018, 0.022)))
        .add_plugins(
            DefaultPlugins
                .set(window_plugin)
                .set(render_plugin)
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>()
                .disable::<WinitPlugin>(),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(GaussianSplattingPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(AutomataViewerPlugin)
        .init_resource::<HeadlessImageCapture>();
    app.finish();
    app.cleanup();
    std::mem::take(app.sub_apps_mut())
}

fn add_render_target(apps: &mut SubApps, width: u32, height: u32) -> RenderTarget {
    let mut target = Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    apps.main
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(target)
        .into()
}

fn assign_render_target_to_gaussian_cameras(apps: &mut SubApps, target: RenderTarget) {
    let target_size = target.as_image().and_then(|handle| {
        apps.main
            .world()
            .resource::<Assets<Image>>()
            .get(handle)
            .map(|image| UVec2::new(image.width(), image.height()))
    });
    let viewport = target_size.map(|size| Viewport {
        physical_position: UVec2::ZERO,
        physical_size: size,
        depth: 0.0..1.0,
    });
    let updates = {
        let world = apps.main.world_mut();
        let mut cameras = world.query::<(Entity, Option<&GaussianCamera>)>();
        cameras
            .iter(world)
            .map(|(entity, gaussian)| (entity, gaussian.is_some()))
            .collect::<Vec<_>>()
    };
    for (entity, gaussian) in updates {
        let mut entity = apps.main.world_mut().entity_mut(entity);
        if gaussian {
            if let (Some(viewport), Some(mut camera)) = (&viewport, entity.get_mut::<Camera>()) {
                camera.viewport = Some(viewport.clone());
            }
            entity.insert(target.clone());
        } else if let Some(mut camera) = entity.get_mut::<Camera>() {
            camera.is_active = false;
        }
    }
}

fn capture_target_png(
    apps: &mut SubApps,
    target: &RenderTarget,
    path: &Path,
) -> Result<CaptureMetrics, Box<dyn std::error::Error>> {
    {
        let mut capture = apps.main.world_mut().resource_mut::<HeadlessImageCapture>();
        *capture = HeadlessImageCapture::default();
    }
    let image_handle = target
        .as_image()
        .ok_or_else(|| std::io::Error::other("headless render target is not an image"))?
        .clone();
    apps.main
        .world_mut()
        .spawn(Screenshot::image(image_handle))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<HeadlessImageCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
                capture.image = Some(event.image.clone());
            },
        );
    for _ in 0..12 {
        pump_headless_frame(apps);
        if apps
            .main
            .world()
            .resource::<HeadlessImageCapture>()
            .captured
        {
            break;
        }
    }
    let capture = apps.main.world().resource::<HeadlessImageCapture>();
    let image = capture
        .image
        .as_ref()
        .ok_or_else(|| std::io::Error::other("headless screenshot was not captured"))?;
    let metrics = capture
        .metrics
        .ok_or_else(|| std::io::Error::other("headless screenshot did not return metrics"))?;
    save_bevy_image_png(image, path)?;
    Ok(metrics)
}

fn save_bevy_image_png(image: &Image, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let data = image
        .data
        .as_ref()
        .ok_or_else(|| std::io::Error::other("captured image has no CPU data"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image::save_buffer_with_format(
        path,
        data,
        image.width(),
        image.height(),
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )?;
    Ok(())
}

fn capture_metrics(image: &Image) -> Option<CaptureMetrics> {
    let data = image.data.as_ref()?;
    let width = image.width();
    let height = image.height();
    let background = data.get(0..3)?;
    let mut metrics = CaptureMetrics {
        width,
        height,
        lit_pixels: 0,
        max_delta: 0,
        min_x: width,
        max_x: 0,
        min_y: height,
        max_y: 0,
    };
    for (pixel_index, rgba) in data.chunks_exact(4).enumerate() {
        let delta = rgba[0]
            .abs_diff(background[0])
            .max(rgba[1].abs_diff(background[1]))
            .max(rgba[2].abs_diff(background[2]));
        metrics.max_delta = metrics.max_delta.max(delta);
        if delta > 8 {
            let x = pixel_index as u32 % width;
            let y = pixel_index as u32 / width;
            metrics.lit_pixels += 1;
            metrics.min_x = metrics.min_x.min(x);
            metrics.max_x = metrics.max_x.max(x);
            metrics.min_y = metrics.min_y.min(y);
            metrics.max_y = metrics.max_y.max(y);
        }
    }
    Some(metrics)
}

fn pump_headless_frame(apps: &mut SubApps) {
    apps.update();
    apps.main
        .world()
        .resource::<RenderDevice>()
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
}

fn set_paused(apps: &mut SubApps, paused: bool) {
    apps.main
        .world_mut()
        .resource_mut::<AutomataSettings>()
        .paused = paused;
}

fn current_frame(apps: &SubApps) -> usize {
    apps.main.world().resource::<AutomataRuntime>().frame
}

fn resolve_output_dir(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn resolve_existing_path(path: &Path, label: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let value = path.display().to_string();
    resolve_workspace_path(&value)
        .ok_or_else(|| std::io::Error::other(format!("missing {label} at {value}")).into())
}

fn resolve_optional_path(path: &Path) -> PathBuf {
    let value = path.display().to_string();
    resolve_workspace_path(&value).unwrap_or_else(|| path.to_path_buf())
}

fn model_source_label(config: &HeadlessExportConfig, settings: &AutomataSettings) -> String {
    if let Some(path) = &config.hyper_image_path {
        format!("hyper-image:{}", path.display())
    } else if let Some(path) = &config.model_path {
        format!("model:{}", path.display())
    } else if let Some(path) = &settings.model_path {
        format!("default-model:{path}")
    } else {
        format!("preset:{:?}", config.preset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_capture_steps_default_to_final_step() {
        assert_eq!(
            planned_capture_steps(128, None, &[]).unwrap(),
            vec![128usize]
        );
    }

    #[test]
    fn planned_capture_steps_include_stride_and_final_steps() {
        assert_eq!(
            planned_capture_steps(130, Some(64), &[]).unwrap(),
            vec![64usize, 128, 130]
        );
    }

    #[test]
    fn planned_capture_steps_sort_and_clip_explicit_steps() {
        assert_eq!(
            planned_capture_steps(64, Some(8), &[64, 16, 128, 16]).unwrap(),
            vec![16usize, 64]
        );
    }
}
