use super::{
    bench::BenchViewport, capture::RenderCapture, fixtures::LIZARD_MODEL_PATH, prelude::*,
};

pub(crate) fn headless_automata_viewer(particles: usize) -> SubApps {
    let render_plugin = RenderPlugin {
        synchronous_pipeline_compilation: true,
        ..default()
    };
    let window_plugin = WindowPlugin {
        primary_window: None,
        exit_condition: ExitCondition::DontExit,
        ..default()
    };
    let mut settings = AutomataSettings {
        particle_count: particles,
        steps_per_frame: 1,
        seed_scale: 0.2,
        render_scale: 1.0,
        render_opacity: 1.0,
        ..Default::default()
    };
    if Path::new(LIZARD_MODEL_PATH).exists() {
        settings.model_path = Some(LIZARD_MODEL_PATH.to_string());
    }

    let mut app = App::new();
    app.insert_resource(settings)
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
        .init_resource::<RenderCapture>();
    app.finish();
    app.cleanup();
    std::mem::take(app.sub_apps_mut())
}

pub(crate) fn assign_render_target_to_gaussian_cameras(apps: &mut SubApps, target: RenderTarget) {
    assign_render_target_to_gaussian_cameras_with_viewport(apps, target, None);
}

pub(crate) fn assign_render_target_to_gaussian_cameras_with_viewport(
    apps: &mut SubApps,
    target: RenderTarget,
    viewport: Option<BenchViewport>,
) {
    let world = apps.main.world_mut();
    let target_size = target.as_image().and_then(|handle| {
        world
            .resource::<Assets<Image>>()
            .get(handle)
            .map(|image| UVec2::new(image.width(), image.height()))
    });
    let viewport = viewport.or_else(|| {
        target_size.map(|size| BenchViewport {
            x: 0,
            y: 0,
            width: size.x,
            height: size.y,
        })
    });
    let updates = {
        let mut cameras = world.query::<(Entity, Option<&GaussianCamera>)>();
        cameras
            .iter(world)
            .map(|(entity, gaussian)| (entity, gaussian.is_some()))
            .collect::<Vec<_>>()
    };
    for (entity, gaussian) in updates {
        let mut entity = world.entity_mut(entity);
        if gaussian {
            if let (Some(viewport), Some(mut camera)) = (viewport, entity.get_mut::<Camera>()) {
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(viewport.x, viewport.y),
                    physical_size: UVec2::new(viewport.width, viewport.height),
                    depth: 0.0..1.0,
                });
            }
            entity.insert(target.clone());
        } else if let Some(mut camera) = entity.get_mut::<Camera>() {
            camera.is_active = false;
        }
    }
}

pub(crate) fn headless_renderer() -> SubApps {
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
    app.add_plugins(
        DefaultPlugins
            .set(window_plugin)
            .set(render_plugin)
            .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
            .disable::<bevy::log::LogPlugin>()
            .disable::<WinitPlugin>(),
    )
    .add_plugins(GaussianSplattingPlugin)
    .init_resource::<RenderCapture>();
    app.finish();
    app.cleanup();
    std::mem::take(app.sub_apps_mut())
}

pub(crate) fn add_render_target(apps: &mut SubApps, width: u32, height: u32) -> RenderTarget {
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

pub(crate) fn pump_headless_frame(apps: &mut SubApps) {
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
