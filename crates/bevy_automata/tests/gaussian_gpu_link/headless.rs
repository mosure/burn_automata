use super::common::*;

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
