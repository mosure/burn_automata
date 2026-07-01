#![cfg(all(feature = "splatting", feature = "gpu_wgpu"))]

use bevy::{
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
use bevy_automata::{
    AutomataRenderDiagnostics, AutomataRuntime, AutomataSettings, AutomataViewerPlugin,
    gaussian_storage_buffer_refs,
};
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianMode, GaussianSplattingPlugin, Planar,
    PlanarGaussian3d, PlanarGaussian3dHandle, PlanarStorageBindGroup,
    SphericalHarmonicCoefficients,
    gaussian::cloud::CloudVisibilityClass,
    gaussian::formats::planar_3d::PlanarStorageGaussian3d,
    render::SortBindGroup,
    sort::{SortMode, SortTrigger, SortedEntries, SortedEntriesHandle},
};
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use burn_automata::{
    AutomataError, AutomataPreset, NpaConfig, NpaModel, ParticleSeed,
    gpu::{GAUSSIAN_SH_COEFF_COUNT, WgpuAutomataExecutor, WgpuGaussianReadback, WgpuNeighborMode},
    import::load_manifest,
    kernels::HashGridConfig,
    rollout::seed_particles_scaled,
};
use std::{
    path::Path,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

const LIZARD_MODEL_PATH: &str = "/tmp/burn_automata_lizard.bpk";
const TORUS_GROWTH_MODEL_PATH: &str = "assets/models/uv_torus_growth_3d.bpk";
const SH_C0: f32 = 0.282_094_8;
static BEVY_TEST_LOCK: Mutex<()> = Mutex::new(());

fn bevy_test_guard() -> MutexGuard<'static, ()> {
    BEVY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn burn_wgpu_writes_bevy_planar_gaussian_storage_buffers() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = bevy_test_guard();
    let preset = AutomataPreset::Growing3dGs;
    let particles = 64;
    let seed_scale = NpaConfig::seed_scale_for_preset(preset);
    let (config, grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config, 42);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        43,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => executor,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping Bevy gaussian GPU-link test: {message}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };
    let mut gpu_state =
        executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;
    let storage = create_planar_storage(&executor, particles)?;
    let gaussian_refs = gaussian_storage_buffer_refs(&storage);
    executor.step_state_into_gaussians(&mut gpu_state, &gaussian_refs)?;

    let gpu = executor.read_state(&gpu_state)?;
    let readback = executor.read_gaussian_buffer_refs(&gaussian_refs, storage.count)?;

    assert_eq!(storage.count, particles);
    assert_eq!(readback.position_visibility.len(), particles * 4);
    assert_eq!(
        readback.spherical_harmonic.len(),
        particles * GAUSSIAN_SH_COEFF_COUNT
    );
    for idx in 0..particles {
        let base = idx * 4;
        for axis in 0..3 {
            assert!(
                (readback.position_visibility[base + axis] - gpu.next_positions[idx][axis]).abs()
                    <= 1.0e-6
            );
        }
        assert_eq!(readback.position_visibility[base + 3], 1.0);
        if model.config.spatial_dims == 2 {
            assert_eq!(readback.scale_opacity[base + 3], 1.0);
        } else {
            assert!(readback.scale_opacity[base + 3] >= 0.05);
            assert!(readback.scale_opacity[base + 3] <= 0.95);
        }

        let state_base = idx * model.config.state_dims;
        let color_base = state_base + model.config.state_dims - 3;
        let expected_color = [
            (gpu.next_states[color_base] + 0.5).clamp(0.0, 1.0),
            (gpu.next_states[color_base + 1] + 0.5).clamp(0.0, 1.0),
            (gpu.next_states[color_base + 2] + 0.5).clamp(0.0, 1.0),
        ];
        let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
        for (channel, expected) in expected_color.iter().enumerate() {
            let decoded = 0.5 + SH_C0 * readback.spherical_harmonic[sh_base + channel];
            assert!(
                (decoded - *expected).abs() <= 2.0e-6,
                "particle {idx} channel {channel}: decoded {decoded} != expected {}",
                expected
            );
        }
    }
    assert!(
        readback
            .spherical_harmonic
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn bevy_gaussian_splatting_renders_headless_image() {
    let _guard = bevy_test_guard();
    let mut apps = headless_renderer();
    let target = add_render_target(&mut apps, 128, 128);
    let cloud = apps
        .main
        .world_mut()
        .resource_mut::<Assets<bevy_gaussian_splatting::PlanarGaussian3d>>()
        .add(visible_test_cloud_3d(128));

    apps.main.world_mut().spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings {
            sort_mode: SortMode::None,
            ..default()
        },
        Transform::default(),
        Visibility::default(),
    ));
    apps.main.world_mut().spawn((
        Camera3d::default(),
        target.clone(),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
    ));

    for _ in 0..4 {
        pump_headless_frame(&mut apps);
    }

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );

    for _ in 0..8 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }

    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(capture.captured, "headless screenshot was not captured");
    let metrics = capture
        .metrics
        .expect("headless gaussian render did not return image data");
    assert!(
        metrics.lit_pixels > 0,
        "headless gaussian render produced a blank image"
    );
}

#[test]
fn viewer_bridge_renders_compact_headless_capture() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let particles = 512;
    let mut apps = headless_automata_viewer(particles);
    pump_headless_frame(&mut apps);
    let target = add_render_target(&mut apps, 256, 256);

    {
        let world = apps.main.world_mut();
        let entities = {
            let mut cameras = world.query_filtered::<Entity, With<GaussianCamera>>();
            cameras.iter(world).collect::<Vec<_>>()
        };
        for entity in entities {
            world.entity_mut(entity).insert(target.clone());
        }
    }

    for _ in 0..12 {
        pump_headless_frame(&mut apps);
    }

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );

    for _ in 0..10 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }

    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(
        capture.captured,
        "viewer bridge screenshot was not captured"
    );
    let metrics = capture
        .metrics
        .expect("viewer bridge screenshot did not return image data");
    assert_compact_capture(metrics);
    Ok(())
}

#[test]
fn viewer_3d_camera_renders_static_gaussian_cloud() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let mut apps = headless_automata_viewer(256);
    let target = add_render_target(&mut apps, 256, 256);
    configure_viewer_pipeline_bench(
        &mut apps,
        &ViewerPipelineBenchConfig {
            particles: 256,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::None,
            warmup_frames: 4,
            measured_frames: 1,
            width: 256,
            height: 256,
            viewport: None,
        },
    );
    apps.main
        .world_mut()
        .resource_mut::<AutomataSettings>()
        .paused = true;

    for _ in 0..8 {
        pump_headless_frame(&mut apps);
    }
    assign_render_target_to_gaussian_cameras(&mut apps, target.clone());
    {
        let world = apps.main.world_mut();
        let cloud_handle = {
            let mut query = world.query_filtered::<&PlanarGaussian3dHandle, With<CloudSettings>>();
            query
                .single(world)
                .expect("viewer should have one gaussian cloud")
                .0
                .clone()
        };
        world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .insert(cloud_handle.id(), visible_test_cloud_3d(256))?;
    }
    for _ in 0..8 {
        pump_headless_frame(&mut apps);
    }

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );
    for _ in 0..10 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }
    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(
        capture.captured,
        "viewer 3D control screenshot was not captured"
    );
    let metrics = capture
        .metrics
        .expect("viewer 3D control screenshot did not return image data");
    if metrics.lit_pixels == 0 {
        eprintln!(
            "blank viewer 3D static control; cameras={:?}; clouds={:?}; render_world={:?}; render_visibility={:?}",
            gaussian_camera_snapshot(&mut apps),
            gaussian_cloud_snapshot(&mut apps),
            gaussian_render_world_snapshot(&mut apps),
            gaussian_render_visibility_snapshot(&mut apps)
        );
    }
    assert!(
        metrics.lit_pixels > 0,
        "viewer 3D static control rendered blank: {metrics:?}"
    );
    Ok(())
}

fn visible_test_cloud_3d(count: usize) -> PlanarGaussian3d {
    let side = (count as f32).cbrt().ceil().max(1.0) as usize;
    let mut gaussians = Vec::with_capacity(count);
    for idx in 0..count {
        let x = idx % side;
        let y = (idx / side) % side;
        let z = idx / (side * side);
        let denom = (side.saturating_sub(1)).max(1) as f32;
        let position = [
            (x as f32 / denom - 0.5) * 0.9,
            (y as f32 / denom - 0.5) * 0.9,
            (z as f32 / denom - 0.5) * 0.9,
            1.0,
        ];
        let color = [
            0.25 + 0.55 * x as f32 / denom,
            0.30 + 0.50 * y as f32 / denom,
            0.45 + 0.40 * z as f32 / denom,
        ];
        let mut coefficients = [0.0; GAUSSIAN_SH_COEFF_COUNT];
        coefficients[0] = (color[0] - 0.5) / SH_C0;
        coefficients[1] = (color[1] - 0.5) / SH_C0;
        coefficients[2] = (color[2] - 0.5) / SH_C0;
        gaussians.push(Gaussian3d {
            position_visibility: position.into(),
            spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.06, 0.06, 0.06, 0.75].into(),
        });
    }
    gaussians.into()
}

#[test]
fn viewer_particle_count_transitions_keep_gpu_buffers_coherent()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let mut apps = headless_automata_viewer(256);
    for _ in 0..8 {
        pump_headless_frame(&mut apps);
    }

    let transitions = [
        (512usize, false),
        (1024, false),
        (4096, false),
        (8192, true),
        (16384, true),
        (2048, false),
        (64, false),
    ];
    for (particles, paused) in transitions {
        {
            let world = apps.main.world_mut();
            let mut settings = world.resource_mut::<AutomataSettings>();
            settings.particle_count = particles;
            settings.paused = paused;
            settings.revision = settings.revision.wrapping_add(1);
            let mut runtime = world.resource_mut::<AutomataRuntime>();
            runtime.trace = None;
            runtime.frame = 0;
        }
        for _ in 0..8 {
            pump_headless_frame(&mut apps);
        }
        assert_viewer_cloud_capacity(&mut apps, particles);
        if !paused {
            assert_render_resize_caught_up(&apps, particles);
        }
    }

    Ok(())
}

#[test]
fn viewer_rapid_particle_count_changes_render_final_state() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = bevy_test_guard();
    let mut apps = headless_automata_viewer(256);
    let target = add_render_target(&mut apps, 256, 256);

    pump_headless_frame(&mut apps);
    assign_render_target_to_gaussian_cameras(&mut apps, target.clone());
    for _ in 0..7 {
        pump_headless_frame(&mut apps);
    }

    for particles in [512usize, 4096, 1024, 16384, 256, 8192, 64, 2048, 4096] {
        {
            let world = apps.main.world_mut();
            let mut settings = world.resource_mut::<AutomataSettings>();
            settings.particle_count = particles;
            settings.paused = false;
            settings.revision = settings.revision.wrapping_add(1);
            let mut runtime = world.resource_mut::<AutomataRuntime>();
            runtime.trace = None;
            runtime.frame = 0;
        }
        for _ in 0..2 {
            pump_headless_frame(&mut apps);
        }
    }

    for _ in 0..16 {
        pump_headless_frame(&mut apps);
    }
    assert_viewer_cloud_capacity(&mut apps, 4096);
    assert_render_resize_caught_up(&apps, 4096);

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );

    for _ in 0..10 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }

    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(
        capture.captured,
        "rapid particle-count resize screenshot was not captured"
    );
    let metrics = capture
        .metrics
        .expect("rapid particle-count resize screenshot did not return image data");
    assert_compact_capture(metrics);
    Ok(())
}

#[test]
fn viewer_model_and_particle_config_transitions_are_stable()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let mut apps = headless_automata_viewer(256);
    for _ in 0..8 {
        pump_headless_frame(&mut apps);
    }

    let mut transitions = vec![
        (
            None,
            AutomataPreset::Growing3dGs,
            1024usize,
            0.35f32,
            true,
            3usize,
        ),
        (None, AutomataPreset::Texture2d, 512, 1.0, false, 2),
        (None, AutomataPreset::PointMnist, 512, 0.55, false, 2),
        (None, AutomataPreset::Growing2d, 1024, 0.2, false, 2),
    ];
    if Path::new(LIZARD_MODEL_PATH).exists() {
        transitions.push((
            Some(LIZARD_MODEL_PATH),
            AutomataPreset::Growing2d,
            512,
            0.2,
            false,
            2,
        ));
    }
    if Path::new("/tmp/burn_automata_polka.bpk").exists() {
        transitions.push((
            Some("/tmp/burn_automata_polka.bpk"),
            AutomataPreset::Texture2d,
            512,
            1.0,
            false,
            2,
        ));
    }

    for (model_path, preset, particles, seed_scale, paused, spatial_dims) in transitions {
        {
            let world = apps.main.world_mut();
            let mut settings = world.resource_mut::<AutomataSettings>();
            settings.model_path = model_path.map(str::to_string);
            settings.preset = preset;
            settings.particle_count = particles;
            settings.seed_scale = seed_scale;
            settings.paused = paused;
            settings.revision = settings.revision.wrapping_add(1);
        }
        for _ in 0..10 {
            pump_headless_frame(&mut apps);
        }
        assert_viewer_cloud_capacity(&mut apps, particles);
        assert_runtime_spatial_dims(&mut apps, spatial_dims);
        assert_cloud_mode(&mut apps, spatial_dims);
    }

    Ok(())
}

#[test]
fn viewer_inference_gaussian_render_pipeline_has_steady_frame_benchmark()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let sample = benchmark_viewer_pipeline(ViewerPipelineBenchConfig {
        particles: 1024,
        steps_per_frame: 1,
        neighbor_mode: WgpuNeighborMode::Auto,
        sort_mode: SortMode::None,
        warmup_frames: 14,
        measured_frames: 24,
        width: 256,
        height: 256,
        viewport: None,
    })?;

    eprintln!(
        "viewer_pipeline_bench requested_neighbor={} effective_neighbor={} sort={} target={}x{} viewport={}x{} particles={} steps/frame={} frames={} median={:.3}ms median_fps={:.1} p95={:.3}ms max={:.3}ms jitter={:.2} lit_pixels={}",
        sample.requested_neighbor_mode,
        sample.neighbor_mode,
        sample.sort_mode,
        sample.target_width,
        sample.target_height,
        sample.viewport_width,
        sample.viewport_height,
        sample.particles,
        sample.steps_per_frame,
        sample.frames,
        sample.median_ms,
        sample.median_fps(),
        sample.p95_ms,
        sample.max_ms,
        sample.jitter_ratio,
        sample.lit_pixels
    );
    assert_viewer_pipeline_sample_stable(&sample);
    Ok(())
}

#[test]
#[ignore = "set BURN_AUTOMATA_VIEWER_BENCH=1 or run explicitly for the broader UI-shaped benchmark matrix"]
fn viewer_inference_gaussian_render_pipeline_benchmark_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    if std::env::var("BURN_AUTOMATA_VIEWER_BENCH").as_deref() != Ok("1") {
        eprintln!("skipping viewer benchmark matrix; set BURN_AUTOMATA_VIEWER_BENCH=1");
        return Ok(());
    }

    let configs = [
        ViewerPipelineBenchConfig {
            particles: 1024,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 256,
            height: 256,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 1024,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::SortedCells,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 256,
            height: 256,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 4096,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 320,
            height: 320,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 8192,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 384,
            height: 384,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 4096,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::SortedCells,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 320,
            height: 320,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 8192,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::SortedCells,
            sort_mode: SortMode::None,
            warmup_frames: 18,
            measured_frames: 48,
            width: 384,
            height: 384,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 4096,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::Radix,
            warmup_frames: 30,
            measured_frames: 48,
            width: 320,
            height: 320,
            viewport: None,
        },
        ViewerPipelineBenchConfig {
            particles: 4096,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::Radix,
            warmup_frames: 18,
            measured_frames: 48,
            width: 1600,
            height: 900,
            viewport: Some(BenchViewport {
                x: 540,
                y: 0,
                width: 1060,
                height: 900,
            }),
        },
        ViewerPipelineBenchConfig {
            particles: 8192,
            steps_per_frame: 1,
            neighbor_mode: WgpuNeighborMode::Auto,
            sort_mode: SortMode::Radix,
            warmup_frames: 18,
            measured_frames: 48,
            width: 1600,
            height: 900,
            viewport: Some(BenchViewport {
                x: 540,
                y: 0,
                width: 1060,
                height: 900,
            }),
        },
    ];
    let mut samples = Vec::new();
    for config in configs {
        let sample = benchmark_viewer_pipeline(config)?;
        eprintln!(
            "viewer_pipeline_bench requested_neighbor={} effective_neighbor={} sort={} target={}x{} viewport={}x{} particles={} steps/frame={} frames={} median={:.3}ms median_fps={:.1} p95={:.3}ms max={:.3}ms jitter={:.2} lit_pixels={}",
            sample.requested_neighbor_mode,
            sample.neighbor_mode,
            sample.sort_mode,
            sample.target_width,
            sample.target_height,
            sample.viewport_width,
            sample.viewport_height,
            sample.particles,
            sample.steps_per_frame,
            sample.frames,
            sample.median_ms,
            sample.median_fps(),
            sample.p95_ms,
            sample.max_ms,
            sample.jitter_ratio,
            sample.lit_pixels
        );
        assert_viewer_pipeline_sample_stable(&sample);
        samples.push(sample);
    }
    write_viewer_pipeline_benchmark_json(&samples)?;
    Ok(())
}

#[test]
fn automata_gaussian_headless_capture_is_compact() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    let particles = 1024;
    let (readback, spatial_dims) = automata_gaussian_readback(particles, 4)?;
    assert_compact_automata_gaussian_readback(&readback, particles, spatial_dims);

    let mut apps = headless_renderer();
    let target = add_render_target(&mut apps, 256, 256);
    let cloud = apps
        .main
        .world_mut()
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(planar_cloud_from_readback(&readback, particles));

    apps.main.world_mut().spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings {
            gaussian_mode: GaussianMode::Gaussian2d,
            global_scale: 1.0,
            global_opacity: 1.0,
            opacity_adaptive_radius: true,
            ..default()
        },
        Transform::default(),
        Visibility::default(),
    ));
    apps.main.world_mut().spawn((
        Camera3d::default(),
        target.clone(),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera::default(),
    ));

    for _ in 0..6 {
        pump_headless_frame(&mut apps);
    }

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );

    for _ in 0..10 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }

    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(capture.captured, "automata screenshot was not captured");
    let metrics = capture
        .metrics
        .expect("automata screenshot did not return image data");
    assert_compact_capture(metrics);
    Ok(())
}

fn headless_automata_viewer(particles: usize) -> SubApps {
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

fn automata_gaussian_readback(
    particles: usize,
    steps: usize,
) -> Result<(WgpuGaussianReadback, usize), Box<dyn std::error::Error>> {
    let (model, grid, seed_scale) = lizard_or_seeded_model()?;
    let spatial_dims = model.config.spatial_dims;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        42,
        ParticleSeed::UniformCircle,
        seed_scale,
    );
    let executor = match WgpuAutomataExecutor::new_blocking() {
        Ok(executor) => executor,
        Err(AutomataError::InvalidArgument(message)) if is_missing_wgpu(&message) => {
            eprintln!("skipping automata gaussian readback test: {message}");
            return Err(std::io::Error::other(message).into());
        }
        Err(err) => return Err(err.into()),
    };
    let mut state = executor.create_state(&model, &positions, &states, 1, particles, &grid, 1.0)?;
    let gaussian_buffers = executor.create_gaussian_buffers(particles)?;
    for _ in 0..steps.max(1) {
        executor.step_state_into_gaussians(&mut state, &gaussian_buffers.refs())?;
    }
    Ok((
        executor.read_gaussian_buffers(&gaussian_buffers)?,
        spatial_dims,
    ))
}

#[derive(Clone, Debug)]
struct ViewerPipelineBenchConfig {
    particles: usize,
    steps_per_frame: usize,
    neighbor_mode: WgpuNeighborMode,
    sort_mode: SortMode,
    warmup_frames: usize,
    measured_frames: usize,
    width: u32,
    height: u32,
    viewport: Option<BenchViewport>,
}

#[derive(Clone, Copy, Debug)]
struct BenchViewport {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug)]
struct ViewerPipelineBenchSample {
    sort_mode: &'static str,
    requested_neighbor_mode: &'static str,
    neighbor_mode: &'static str,
    particles: usize,
    steps_per_frame: usize,
    frames: usize,
    frame_delta: usize,
    target_width: u32,
    target_height: u32,
    viewport_width: u32,
    viewport_height: u32,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    jitter_ratio: f64,
    lit_pixels: usize,
}

impl ViewerPipelineBenchSample {
    fn median_fps(&self) -> f64 {
        if self.median_ms > f64::EPSILON {
            1000.0 / self.median_ms
        } else {
            0.0
        }
    }
}

fn benchmark_viewer_pipeline(
    config: ViewerPipelineBenchConfig,
) -> Result<ViewerPipelineBenchSample, Box<dyn std::error::Error>> {
    let mut apps = headless_automata_viewer(config.particles);
    let target = add_render_target(&mut apps, config.width, config.height);
    configure_viewer_pipeline_bench(&mut apps, &config);

    for _ in 0..4 {
        pump_headless_frame(&mut apps);
    }
    assign_render_target_to_gaussian_cameras_with_viewport(
        &mut apps,
        target.clone(),
        config.viewport,
    );

    for _ in 0..config.warmup_frames {
        pump_headless_frame(&mut apps);
    }
    assign_render_target_to_gaussian_cameras_with_viewport(
        &mut apps,
        target.clone(),
        config.viewport,
    );
    pump_headless_frame(&mut apps);
    assert_viewer_cloud_capacity(&mut apps, config.particles);
    assert_runtime_spatial_dims(&mut apps, 3);
    assert_cloud_mode(&mut apps, 3);
    assert_render_resize_caught_up(&apps, config.particles);
    let start_frame = render_diagnostics(&apps).frame;

    let mut durations = Vec::with_capacity(config.measured_frames);
    for _ in 0..config.measured_frames {
        durations.push(pump_headless_frame_timed(&mut apps));
    }
    let end_frame = render_diagnostics(&apps).frame;
    let (median_ms, p95_ms, max_ms) = frame_time_stats(&durations);
    let jitter_ratio = if median_ms > f64::EPSILON {
        p95_ms / median_ms
    } else {
        0.0
    };

    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<RenderCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
            },
        );
    for _ in 0..12 {
        pump_headless_frame(&mut apps);
        if apps.main.world().resource::<RenderCapture>().captured {
            break;
        }
    }
    let capture = apps.main.world().resource::<RenderCapture>();
    assert!(
        capture.captured,
        "viewer pipeline benchmark screenshot was not captured"
    );
    let metrics = capture
        .metrics
        .expect("viewer pipeline benchmark screenshot did not return image data");
    if metrics.lit_pixels == 0 {
        eprintln!(
            "blank viewer pipeline capture; diagnostics={:?}; cameras={:?}; clouds={:?}; render_world={:?}; render_visibility={:?}",
            render_diagnostics(&apps),
            gaussian_camera_snapshot(&mut apps),
            gaussian_cloud_snapshot(&mut apps),
            gaussian_render_world_snapshot(&mut apps),
            gaussian_render_visibility_snapshot(&mut apps)
        );
    }
    assert_compact_capture(metrics);

    let (viewport_width, viewport_height) = config
        .viewport
        .map(|viewport| (viewport.width, viewport.height))
        .unwrap_or((config.width, config.height));
    Ok(ViewerPipelineBenchSample {
        sort_mode: sort_mode_label(&config.sort_mode),
        requested_neighbor_mode: neighbor_mode_label(config.neighbor_mode),
        neighbor_mode: neighbor_mode_label(effective_viewer_bench_neighbor_mode(&config)),
        particles: config.particles,
        steps_per_frame: config.steps_per_frame,
        frames: config.measured_frames,
        frame_delta: end_frame.saturating_sub(start_frame),
        target_width: config.width,
        target_height: config.height,
        viewport_width,
        viewport_height,
        median_ms,
        p95_ms,
        max_ms,
        jitter_ratio,
        lit_pixels: metrics.lit_pixels,
    })
}

fn configure_viewer_pipeline_bench(apps: &mut SubApps, config: &ViewerPipelineBenchConfig) {
    let world = apps.main.world_mut();
    {
        let mut settings = world.resource_mut::<AutomataSettings>();
        settings.model_path = Path::new(TORUS_GROWTH_MODEL_PATH)
            .exists()
            .then(|| TORUS_GROWTH_MODEL_PATH.to_string());
        settings.preset = AutomataPreset::Growing3dGs;
        settings.particle_count = config.particles;
        settings.steps_per_frame = config.steps_per_frame;
        settings.update_prob = 0.5;
        settings.seed_scale = 0.55;
        settings.reference_seed_scale = settings.seed_scale;
        settings.seed_mode = ParticleSeed::TorusGrowth3d;
        settings.render_scale = 0.5;
        settings.render_opacity = 2.0;
        settings.render_sort_mode_3d = config.sort_mode.clone();
        settings.gpu_neighbor_mode = config.neighbor_mode;
        settings.paused = false;
        settings.revision = settings.revision.wrapping_add(1);
    }
    {
        let mut runtime = world.resource_mut::<AutomataRuntime>();
        runtime.loaded_model_path = None;
        runtime.loaded_preset = None;
        runtime.trace = None;
        runtime.frame = 0;
    }
}

fn sort_mode_label(sort_mode: &SortMode) -> &'static str {
    match sort_mode {
        SortMode::None => "none",
        SortMode::Radix => "radix",
    }
}

fn neighbor_mode_label(neighbor_mode: WgpuNeighborMode) -> &'static str {
    match neighbor_mode {
        WgpuNeighborMode::Auto => "auto",
        WgpuNeighborMode::LinkedList => "linked-list",
        WgpuNeighborMode::FixedCellBuckets { .. } => "fixed-buckets",
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => "tiled-fixed-buckets",
        WgpuNeighborMode::SortedCells => "sorted-cells",
        WgpuNeighborMode::Bvh { .. } => "bvh",
        WgpuNeighborMode::GpuBvh { .. } => "gpu-bvh",
        WgpuNeighborMode::GpuLbvh { .. } => "gpu-lbvh",
        WgpuNeighborMode::GpuMortonLbvh { .. } => "gpu-morton-lbvh",
    }
}

fn effective_viewer_bench_neighbor_mode(config: &ViewerPipelineBenchConfig) -> WgpuNeighborMode {
    if config.neighbor_mode == WgpuNeighborMode::Auto && config.particles <= 2048 {
        WgpuNeighborMode::SortedCells
    } else {
        config.neighbor_mode
    }
}

fn assert_viewer_pipeline_sample_stable(sample: &ViewerPipelineBenchSample) {
    assert!(
        sample.frame_delta >= sample.frames * sample.steps_per_frame,
        "render bridge advanced {} steps for {} measured frames at {} steps/frame",
        sample.frame_delta,
        sample.frames,
        sample.steps_per_frame
    );
    assert!(
        sample.median_ms <= 60.0,
        "viewer pipeline median frame time is not interactive enough: {sample:?}"
    );
    assert!(
        sample.p95_ms <= 90.0,
        "viewer pipeline p95 frame time is too high: {sample:?}"
    );
    assert!(
        sample.jitter_ratio <= 2.5 || sample.p95_ms <= sample.median_ms + 30.0,
        "viewer pipeline frame-time jitter is too high: {sample:?}"
    );
    assert!(
        sample.max_ms <= 120.0 && sample.max_ms <= sample.median_ms * 4.0 + 40.0,
        "viewer pipeline has a large steady-state frame spike: {sample:?}"
    );
    assert!(
        sample.lit_pixels > 0,
        "viewer pipeline benchmark produced a blank render: {sample:?}"
    );
}

fn pump_headless_frame_timed(apps: &mut SubApps) -> Duration {
    let start = Instant::now();
    pump_headless_frame(apps);
    start.elapsed()
}

fn frame_time_stats(durations: &[Duration]) -> (f64, f64, f64) {
    let mut frame_ms = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    frame_ms.sort_by(f64::total_cmp);
    if frame_ms.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let median = percentile_sorted(&frame_ms, 0.50);
    let p95 = percentile_sorted(&frame_ms, 0.95);
    let max = *frame_ms.last().unwrap_or(&0.0);
    (median, p95, max)
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile)
        .round()
        .clamp(0.0, (values.len() - 1) as f64) as usize;
    values[index]
}

fn write_viewer_pipeline_benchmark_json(
    samples: &[ViewerPipelineBenchSample],
) -> Result<(), Box<dyn std::error::Error>> {
    let output = workspace_target_dir().join("viewer_pipeline_bench.json");
    std::fs::create_dir_all(output.parent().expect("target path should have a parent"))?;
    let body = samples
        .iter()
        .map(|sample| {
            format!(
                "    {{\"requested_neighbor_mode\":\"{}\",\"effective_neighbor_mode\":\"{}\",\"sort_mode\":\"{}\",\"particles\":{},\"steps_per_frame\":{},\"frames\":{},\"frame_delta\":{},\"target_width\":{},\"target_height\":{},\"viewport_width\":{},\"viewport_height\":{},\"median_ms\":{:.6},\"median_fps\":{:.6},\"p95_ms\":{:.6},\"max_ms\":{:.6},\"jitter_ratio\":{:.6},\"lit_pixels\":{}}}",
                sample.requested_neighbor_mode,
                sample.neighbor_mode,
                sample.sort_mode,
                sample.particles,
                sample.steps_per_frame,
                sample.frames,
                sample.frame_delta,
                sample.target_width,
                sample.target_height,
                sample.viewport_width,
                sample.viewport_height,
                sample.median_ms,
                sample.median_fps(),
                sample.p95_ms,
                sample.max_ms,
                sample.jitter_ratio,
                sample.lit_pixels
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    std::fs::write(output, format!("{{\n  \"samples\": [\n{body}\n  ]\n}}\n"))?;
    Ok(())
}

fn workspace_target_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
}

fn lizard_or_seeded_model() -> Result<(NpaModel, HashGridConfig, f32), Box<dyn std::error::Error>> {
    if Path::new(LIZARD_MODEL_PATH).exists() {
        let manifest = load_manifest(LIZARD_MODEL_PATH)?;
        let hashgrid = manifest.hashgrid.clone();
        return Ok((manifest.into_model(), hashgrid, 0.2));
    }
    let preset = AutomataPreset::Growing2d;
    let (config, hashgrid) = NpaConfig::for_preset(preset);
    Ok((
        NpaModel::seeded(config, 42),
        hashgrid,
        NpaConfig::seed_scale_for_preset(preset),
    ))
}

fn assert_viewer_cloud_capacity(apps: &mut SubApps, expected_particles: usize) {
    let world = apps.main.world_mut();
    let pairs = {
        let mut query = world.query::<(&PlanarGaussian3dHandle, &SortedEntriesHandle)>();
        query
            .iter(world)
            .map(|(cloud, sorted)| (cloud.0.clone(), sorted.0.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(pairs.len(), 1, "expected one automata gaussian cloud");
    let (cloud_handle, sorted_handle) = &pairs[0];
    let cloud_len = world
        .resource::<Assets<PlanarGaussian3d>>()
        .get(cloud_handle)
        .expect("cloud asset should be present")
        .len();
    let sorted_count = world
        .resource::<Assets<SortedEntries>>()
        .get(sorted_handle)
        .expect("sorted entries asset should be present")
        .entry_count;
    assert_eq!(cloud_len, expected_particles);
    assert!(
        sorted_count >= cloud_len,
        "sorted entry count {sorted_count} is smaller than cloud len {cloud_len}"
    );
}

fn assert_runtime_spatial_dims(apps: &mut SubApps, expected_spatial_dims: usize) {
    let runtime = apps.main.world().resource::<AutomataRuntime>();
    assert_eq!(runtime.model.config.spatial_dims, expected_spatial_dims);
    assert_eq!(runtime.hashgrid.dim, expected_spatial_dims);
}

fn assert_cloud_mode(apps: &mut SubApps, expected_spatial_dims: usize) {
    let world = apps.main.world_mut();
    let modes = {
        let mut query = world.query::<&CloudSettings>();
        query
            .iter(world)
            .map(|settings| settings.gaussian_mode)
            .collect::<Vec<_>>()
    };
    assert_eq!(modes.len(), 1, "expected one cloud settings component");
    let expected = if expected_spatial_dims == 2 {
        GaussianMode::Gaussian2d
    } else {
        GaussianMode::Gaussian3d
    };
    assert_eq!(modes[0], expected);
}

fn assert_render_resize_caught_up(apps: &SubApps, expected_particles: usize) {
    let diagnostics = render_diagnostics(apps);
    assert_eq!(
        diagnostics.requested_particle_count, expected_particles,
        "render bridge did not receive the latest particle count"
    );
    assert_eq!(
        diagnostics.resident_particle_count, expected_particles,
        "resident automata GPU state did not resize"
    );
    assert!(
        diagnostics.gaussian_storage_count >= expected_particles,
        "gaussian storage count {} is smaller than requested particles {}",
        diagnostics.gaussian_storage_count,
        expected_particles
    );
    assert!(
        diagnostics.last_error.is_none(),
        "render bridge still reports an error after resize: {:?}",
        diagnostics.last_error
    );
}

fn render_diagnostics(apps: &SubApps) -> AutomataRenderDiagnostics {
    apps.sub_apps
        .values()
        .find_map(|sub_app| {
            sub_app
                .world()
                .get_resource::<AutomataRenderDiagnostics>()
                .cloned()
        })
        .expect("render diagnostics resource should exist")
}

fn assign_render_target_to_gaussian_cameras(apps: &mut SubApps, target: RenderTarget) {
    assign_render_target_to_gaussian_cameras_with_viewport(apps, target, None);
}

fn assign_render_target_to_gaussian_cameras_with_viewport(
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

fn gaussian_camera_snapshot(apps: &mut SubApps) -> Vec<String> {
    let world = apps.main.world_mut();
    let mut query = world.query::<(
        &Camera,
        Option<&Name>,
        Option<&Camera3d>,
        Option<&Projection>,
        Option<&GaussianCamera>,
        Option<&SortTrigger>,
        Option<&Transform>,
        Option<&RenderTarget>,
    )>();
    query
        .iter(world)
        .map(|(camera, name, camera_3d, projection, gaussian, sort_trigger, transform, target)| {
            format!(
                "{} active={} camera3d={} projection={:?} viewport={:?} gaussian_warmup={:?} sort_trigger={:?} transform={:?} target={:?}",
                name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                camera.is_active,
                camera_3d.is_some(),
                projection.map(|projection| match projection {
                    Projection::Perspective(_) => "perspective",
                    Projection::Orthographic(_) => "orthographic",
                    Projection::Custom(_) => "custom",
                }),
                camera.viewport,
                gaussian.map(|camera| camera.warmup),
                sort_trigger.map(|trigger| (trigger.camera_index, trigger.needs_sort)),
                transform.map(|transform| (transform.translation, transform.rotation)),
                target
            )
        })
        .collect()
}

fn gaussian_cloud_snapshot(apps: &mut SubApps) -> Vec<String> {
    let world = apps.main.world_mut();
    let cloud_info = {
        let mut query = world.query::<(
            &PlanarGaussian3dHandle,
            &SortedEntriesHandle,
            &CloudSettings,
            &Visibility,
            Option<&ViewVisibility>,
            Option<&Aabb>,
            Option<&Name>,
        )>();
        query
            .iter(world)
            .map(
                |(cloud, sorted, settings, visibility, view_visibility, aabb, name)| {
                    (
                        cloud.0.clone(),
                        sorted.0.clone(),
                        settings.gaussian_mode,
                        settings.sort_mode.clone(),
                        *visibility,
                        view_visibility.map(|visibility| visibility.get()),
                        aabb.map(|aabb| (aabb.min(), aabb.max())),
                        name.map(|name| name.as_str().to_string()),
                    )
                },
            )
            .collect::<Vec<_>>()
    };
    let clouds = world.resource::<Assets<PlanarGaussian3d>>();
    let sorted_entries = world.resource::<Assets<SortedEntries>>();
    cloud_info
        .into_iter()
        .map(
            |(cloud, sorted, mode, sort_mode, visibility, view_visible, aabb, name)| {
            let cloud_len = clouds.get(&cloud).map(PlanarGaussian3d::len);
            let sorted_len = sorted_entries.get(&sorted).map(|sorted| sorted.entry_count);
            format!(
                "{} mode={mode:?} sort={sort_mode:?} visibility={visibility:?} view_visible={view_visible:?} aabb={aabb:?} cloud_len={cloud_len:?} sorted_len={sorted_len:?}",
                name.as_deref().unwrap_or("<unnamed>")
            )
        })
        .collect()
}

fn gaussian_render_world_snapshot(apps: &mut SubApps) -> Vec<String> {
    apps.sub_apps
        .values_mut()
        .flat_map(|sub_app| {
            let world = sub_app.world_mut();
            let mut query = world.query::<(
                Option<&PlanarGaussian3dHandle>,
                Option<&PlanarStorageBindGroup<Gaussian3d>>,
                Option<&SortBindGroup>,
                Option<&CloudSettings>,
                Option<&Name>,
            )>();
            query
                .iter(world)
                .filter(|(handle, storage, sort, settings, _name)| {
                    handle.is_some() || storage.is_some() || sort.is_some() || settings.is_some()
                })
                .map(|(handle, storage, sort, settings, name)| {
                    format!(
                        "{} handle={} storage_bind={} sort_bind={} settings={:?}",
                        name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                        handle.is_some(),
                        storage.is_some(),
                        sort.is_some(),
                        settings
                            .map(|settings| (settings.gaussian_mode, settings.sort_mode.clone()))
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn gaussian_render_visibility_snapshot(apps: &mut SubApps) -> Vec<String> {
    apps.sub_apps
        .values_mut()
        .flat_map(|sub_app| {
            let world = sub_app.world_mut();
            let phase_counts = world
                .get_resource::<ViewSortedRenderPhases<Transparent3d>>()
                .map(|phases| {
                    phases
                        .0
                        .iter()
                        .map(|(view, phase)| (*view, phase.items.len(), phase.transient_items.len()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut query = world.query::<(
                Option<&ExtractedView>,
                Option<&GaussianCamera>,
                Option<&RenderVisibleEntities>,
                Option<&Name>,
            )>();
            query
                .iter(world)
                .filter(|(view, gaussian, visible, _name)| {
                    view.is_some() || gaussian.is_some() || visible.is_some()
                })
                .map(|(view, gaussian, visible, name)| {
                    let visible_count = visible
                        .and_then(|visible| visible.get::<CloudVisibilityClass>())
                        .map(|class| class.entities_cpu_culling.len());
                    let phase_count = view.and_then(|view| {
                        phase_counts
                            .iter()
                            .find(|(retained, _, _)| *retained == view.retained_view_entity)
                            .map(|(_, items, transient)| (*items, *transient))
                    });
                    format!(
                        "{} extracted_view={} gaussian={} visible_clouds={visible_count:?} phase={phase_count:?}",
                        name.map(|name| name.as_str()).unwrap_or("<unnamed>"),
                        view.is_some(),
                        gaussian.is_some(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assert_compact_automata_gaussian_readback(
    readback: &WgpuGaussianReadback,
    particles: usize,
    spatial_dims: usize,
) {
    assert_eq!(readback.position_visibility.len(), particles * 4);
    assert_eq!(
        readback.spherical_harmonic.len(),
        particles * GAUSSIAN_SH_COEFF_COUNT
    );
    assert_eq!(readback.rotation.len(), particles * 4);
    assert_eq!(readback.scale_opacity.len(), particles * 4);

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut min_scale = f32::INFINITY;
    let mut max_scale = f32::NEG_INFINITY;
    for idx in 0..particles {
        let base = idx * 4;
        for axis in 0..3 {
            let position = readback.position_visibility[base + axis];
            assert!(position.is_finite(), "non-finite position at {idx}:{axis}");
            min[axis] = min[axis].min(position);
            max[axis] = max[axis].max(position);
            let scale = readback.scale_opacity[base + axis];
            assert!(scale.is_finite(), "non-finite scale at {idx}:{axis}");
            min_scale = min_scale.min(scale);
            max_scale = max_scale.max(scale);
        }
        assert_eq!(readback.position_visibility[base + 3], 1.0);
        if spatial_dims == 2 {
            assert_eq!(readback.scale_opacity[base + 3], 1.0);
        } else {
            assert!((0.05..=0.95).contains(&readback.scale_opacity[base + 3]));
        }
    }

    let width = max[0] - min[0];
    let height = max[1] - min[1];
    assert!(
        width > 0.05 && height > 0.05,
        "collapsed automata bounds: min={min:?} max={max:?}"
    );
    assert!(
        width < 1.25 && height < 1.25,
        "automata bounds too large: min={min:?} max={max:?}"
    );
    assert!(
        min_scale >= 0.0001 && max_scale <= 0.08,
        "unexpected gaussian scale range: min={min_scale} max={max_scale}"
    );
}

fn assert_compact_capture(metrics: CaptureMetrics) {
    let occupancy = metrics.lit_pixels as f32 / (metrics.width * metrics.height) as f32;
    assert!(
        occupancy > 0.002,
        "automata render is too sparse or blank: occupancy={occupancy:.6}, metrics={metrics:?}"
    );
    assert!(
        occupancy < 0.45,
        "automata render covers too much of the frame: occupancy={occupancy:.6}, metrics={metrics:?}"
    );
    assert!(
        metrics.bbox_width() < metrics.width * 9 / 10
            && metrics.bbox_height() < metrics.height * 9 / 10,
        "automata render bbox is too large: {:?}",
        metrics
    );
}

fn planar_cloud_from_readback(
    readback: &WgpuGaussianReadback,
    particles: usize,
) -> PlanarGaussian3d {
    let gaussians = (0..particles)
        .map(|idx| {
            let base = idx * 4;
            let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
            let mut coefficients = [0.0; GAUSSIAN_SH_COEFF_COUNT];
            coefficients.copy_from_slice(
                &readback.spherical_harmonic[sh_base..sh_base + GAUSSIAN_SH_COEFF_COUNT],
            );
            Gaussian3d {
                position_visibility: [
                    readback.position_visibility[base],
                    readback.position_visibility[base + 1],
                    readback.position_visibility[base + 2],
                    readback.position_visibility[base + 3],
                ]
                .into(),
                spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
                rotation: [
                    readback.rotation[base],
                    readback.rotation[base + 1],
                    readback.rotation[base + 2],
                    readback.rotation[base + 3],
                ]
                .into(),
                scale_opacity: [
                    readback.scale_opacity[base],
                    readback.scale_opacity[base + 1],
                    readback.scale_opacity[base + 2],
                    readback.scale_opacity[base + 3],
                ]
                .into(),
            }
        })
        .collect::<Vec<_>>();
    gaussians.into()
}

fn create_planar_storage(
    executor: &WgpuAutomataExecutor,
    count: usize,
) -> Result<PlanarStorageGaussian3d, Box<dyn std::error::Error>> {
    let device = executor.device();
    let storage_usage = BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE;
    let position_visibility = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_position_visibility"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let spherical_harmonic = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_spherical_harmonic"),
        size: byte_len::<f32>(count * GAUSSIAN_SH_COEFF_COUNT)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let rotation = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_rotation"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let scale_opacity = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_scale_opacity"),
        size: byte_len::<f32>(count * 4)?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let draw_indirect_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("bevy_automata_draw_indirect"),
        size: 16,
        usage: BufferUsages::INDIRECT
            | BufferUsages::COPY_DST
            | BufferUsages::COPY_SRC
            | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    Ok(PlanarStorageGaussian3d {
        position_visibility: position_visibility.into(),
        spherical_harmonic: spherical_harmonic.into(),
        rotation: rotation.into(),
        scale_opacity: scale_opacity.into(),
        count,
        draw_indirect_buffer: draw_indirect_buffer.into(),
    })
}

fn byte_len<T>(len: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| std::io::Error::other("buffer byte length overflow"))? as u64)
}

fn is_missing_wgpu(message: &str) -> bool {
    message.contains("no WGPU adapter") || message.contains("failed to create WGPU device")
}

#[derive(Resource, Default)]
struct RenderCapture {
    captured: bool,
    metrics: Option<CaptureMetrics>,
}

#[derive(Clone, Copy, Debug)]
struct CaptureMetrics {
    width: u32,
    height: u32,
    lit_pixels: usize,
    max_delta: u8,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl CaptureMetrics {
    fn bbox_width(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_x - self.min_x + 1
        }
    }

    fn bbox_height(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_y - self.min_y + 1
        }
    }
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
        let lit = delta > 8;
        if lit {
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

fn headless_renderer() -> SubApps {
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
