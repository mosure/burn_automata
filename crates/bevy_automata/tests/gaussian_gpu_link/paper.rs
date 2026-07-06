use super::common::*;

#[derive(Resource, Default)]
struct PaperImageCapture {
    captured: bool,
    metrics: Option<CaptureMetrics>,
    image: Option<Image>,
}

#[derive(Clone, Debug)]
struct PaperCaptureRecord {
    label: &'static str,
    path: std::path::PathBuf,
    particles: usize,
    frames_or_steps: usize,
    metrics: CaptureMetrics,
}

#[test]
fn export_high_particle_renderer_paper_captures() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = bevy_test_guard();
    if std::env::var("BURN_AUTOMATA_PAPER_CAPTURE").as_deref() != Ok("1") {
        eprintln!("skipping paper renderer capture; set BURN_AUTOMATA_PAPER_CAPTURE=1");
        return Ok(());
    }

    let output_dir = std::env::var("BURN_AUTOMATA_PAPER_CAPTURE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| workspace_target_dir().join("paper_renderer_capture"));
    let output_dir = if output_dir.is_absolute() {
        output_dir
    } else {
        workspace_root_dir().join(output_dir)
    };
    std::fs::create_dir_all(&output_dir)?;

    let mut records = Vec::new();
    records.extend(capture_lizard_viewer_rollout(&output_dir)?);
    records.push(capture_wgpu_gaussian_readback(&output_dir)?);
    write_capture_report(&output_dir, &records)?;
    Ok(())
}

fn capture_lizard_viewer_rollout(
    output_dir: &Path,
) -> Result<Vec<PaperCaptureRecord>, Box<dyn std::error::Error>> {
    let lizard_path = workspace_path(LIZARD_MODEL_PATH);
    if !lizard_path.exists() {
        return Err(std::io::Error::other(format!(
            "missing required lizard model at {}",
            lizard_path.display()
        ))
        .into());
    }

    let particles = 4096;
    let mut apps = headless_automata_viewer(particles);
    {
        let mut settings = apps.main.world_mut().resource_mut::<AutomataSettings>();
        settings.model_path = Some(lizard_path.display().to_string());
        settings.revision = settings.revision.wrapping_add(1);
    }
    {
        let mut runtime = apps.main.world_mut().resource_mut::<AutomataRuntime>();
        runtime.loaded_model_path = None;
        runtime.loaded_preset = None;
        runtime.trace = None;
        runtime.frame = 0;
    }
    apps.main
        .world_mut()
        .insert_resource(PaperImageCapture::default());
    let target = add_render_target(&mut apps, 384, 384);
    pump_headless_frame(&mut apps);
    assign_render_target_to_gaussian_cameras(&mut apps, target.clone());

    let capture_frames = [16usize, 64, 128];
    let mut records = Vec::with_capacity(capture_frames.len());
    let mut elapsed = 0usize;
    let mut frame_paths = Vec::with_capacity(capture_frames.len());
    for frame in capture_frames {
        for _ in elapsed..frame {
            pump_headless_frame(&mut apps);
        }
        elapsed = frame;
        assign_render_target_to_gaussian_cameras(&mut apps, target.clone());
        let path = output_dir.join(format!("lizard_wgpu_gaussian_4096_frame{frame}.png"));
        let metrics = capture_target_png(&mut apps, &target, &path)?;
        assert_compact_capture(metrics);
        frame_paths.push(path.clone());
        records.push(PaperCaptureRecord {
            label: "lizard_wgpu_viewer_gaussian",
            path,
            particles,
            frames_or_steps: frame,
            metrics,
        });
    }

    save_horizontal_grid(
        &frame_paths,
        &output_dir.join("lizard_wgpu_gaussian_4096_rollout.png"),
    )?;
    Ok(records)
}

fn capture_wgpu_gaussian_readback(
    output_dir: &Path,
) -> Result<PaperCaptureRecord, Box<dyn std::error::Error>> {
    let particles = 4096;
    let steps = 4;
    let (readback, spatial_dims) = automata_gaussian_readback(particles, steps)?;
    assert_compact_automata_gaussian_readback(&readback, particles, spatial_dims);

    let mut apps = headless_renderer();
    apps.main
        .world_mut()
        .insert_resource(PaperImageCapture::default());
    let target = add_render_target(&mut apps, 384, 384);
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
    for _ in 0..8 {
        pump_headless_frame(&mut apps);
    }

    let path = output_dir.join("wgpu_gaussian_readback_4096_steps4.png");
    let metrics = capture_target_png(&mut apps, &target, &path)?;
    assert_compact_capture(metrics);
    Ok(PaperCaptureRecord {
        label: "wgpu_gaussian_readback",
        path,
        particles,
        frames_or_steps: steps,
        metrics,
    })
}

fn capture_target_png(
    apps: &mut SubApps,
    target: &RenderTarget,
    path: &Path,
) -> Result<CaptureMetrics, Box<dyn std::error::Error>> {
    {
        let mut capture = apps.main.world_mut().resource_mut::<PaperImageCapture>();
        *capture = PaperImageCapture::default();
    }
    apps.main
        .world_mut()
        .spawn(Screenshot::image(target.as_image().unwrap().clone()))
        .observe(
            |event: On<ScreenshotCaptured>, mut capture: ResMut<PaperImageCapture>| {
                capture.captured = true;
                capture.metrics = capture_metrics(&event.image);
                capture.image = Some(event.image.clone());
            },
        );
    for _ in 0..12 {
        pump_headless_frame(apps);
        if apps.main.world().resource::<PaperImageCapture>().captured {
            break;
        }
    }
    let capture = apps.main.world().resource::<PaperImageCapture>();
    let image = capture
        .image
        .as_ref()
        .ok_or_else(|| std::io::Error::other("paper screenshot was not captured"))?;
    let metrics = capture
        .metrics
        .ok_or_else(|| std::io::Error::other("paper screenshot did not return metrics"))?;
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

fn save_horizontal_grid(
    paths: &[std::path::PathBuf],
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let panels = paths
        .iter()
        .map(image::open)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|image| image.to_rgba8())
        .collect::<Vec<_>>();
    let gap = 8u32;
    let width = panels.iter().map(image::RgbaImage::width).sum::<u32>()
        + gap * panels.len().saturating_sub(1) as u32;
    let height = panels
        .iter()
        .map(image::RgbaImage::height)
        .max()
        .unwrap_or(1);
    let mut grid = image::RgbaImage::from_pixel(width, height, image::Rgba([255, 255, 255, 255]));
    let mut x = 0u32;
    for panel in panels {
        image::imageops::overlay(&mut grid, &panel, i64::from(x), 0);
        x += panel.width() + gap;
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    grid.save(output)?;
    Ok(())
}

fn write_capture_report(
    output_dir: &Path,
    records: &[PaperCaptureRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = records
        .iter()
        .map(|record| {
            let occupancy = record.metrics.lit_pixels as f32
                / (record.metrics.width * record.metrics.height) as f32;
            let display_path = record
                .path
                .strip_prefix(workspace_root_dir())
                .unwrap_or(&record.path);
            format!(
                concat!(
                    "    {{\"label\":\"{}\",\"path\":\"{}\",\"particles\":{},",
                    "\"frames_or_steps\":{},\"width\":{},\"height\":{},",
                    "\"lit_pixels\":{},\"occupancy\":{:.8},\"bbox_width\":{},",
                    "\"bbox_height\":{},\"max_delta\":{}}}"
                ),
                record.label,
                display_path.display(),
                record.particles,
                record.frames_or_steps,
                record.metrics.width,
                record.metrics.height,
                record.metrics.lit_pixels,
                occupancy,
                record.metrics.bbox_width(),
                record.metrics.bbox_height(),
                record.metrics.max_delta
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    std::fs::write(
        output_dir.join("paper_renderer_capture_report.json"),
        format!("{{\n  \"captures\": [\n{entries}\n  ]\n}}\n"),
    )?;
    Ok(())
}
