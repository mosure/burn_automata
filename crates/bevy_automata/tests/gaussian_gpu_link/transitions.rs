use super::common::*;

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
    if Path::new(POLKA_MODEL_PATH).exists() {
        transitions.push((
            Some(POLKA_MODEL_PATH),
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
