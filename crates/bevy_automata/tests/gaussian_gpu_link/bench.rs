use super::{app::*, capture::*, diagnostics::*, fixtures::*, prelude::*};

#[derive(Clone, Debug)]
pub(crate) struct ViewerPipelineBenchConfig {
    pub(crate) particles: usize,
    pub(crate) steps_per_frame: usize,
    pub(crate) neighbor_mode: WgpuNeighborMode,
    pub(crate) sort_mode: SortMode,
    pub(crate) warmup_frames: usize,
    pub(crate) measured_frames: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) viewport: Option<BenchViewport>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BenchViewport {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct ViewerPipelineBenchSample {
    pub(crate) sort_mode: &'static str,
    pub(crate) requested_neighbor_mode: &'static str,
    pub(crate) neighbor_mode: &'static str,
    pub(crate) particles: usize,
    pub(crate) steps_per_frame: usize,
    pub(crate) frames: usize,
    pub(crate) frame_delta: usize,
    pub(crate) target_width: u32,
    pub(crate) target_height: u32,
    pub(crate) viewport_width: u32,
    pub(crate) viewport_height: u32,
    pub(crate) median_ms: f64,
    pub(crate) p95_ms: f64,
    pub(crate) max_ms: f64,
    pub(crate) jitter_ratio: f64,
    pub(crate) lit_pixels: usize,
}

impl ViewerPipelineBenchSample {
    pub(crate) fn median_fps(&self) -> f64 {
        if self.median_ms > f64::EPSILON {
            1000.0 / self.median_ms
        } else {
            0.0
        }
    }
}

pub(crate) fn benchmark_viewer_pipeline(
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

pub(crate) fn configure_viewer_pipeline_bench(
    apps: &mut SubApps,
    config: &ViewerPipelineBenchConfig,
) {
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

pub(crate) fn sort_mode_label(sort_mode: &SortMode) -> &'static str {
    match sort_mode {
        SortMode::None => "none",
        SortMode::Radix => "radix",
    }
}

pub(crate) fn neighbor_mode_label(neighbor_mode: WgpuNeighborMode) -> &'static str {
    match neighbor_mode {
        WgpuNeighborMode::Auto => "auto",
        WgpuNeighborMode::LinkedList => "linked-list",
        WgpuNeighborMode::FixedCellBuckets { .. } => "fixed-buckets",
        WgpuNeighborMode::TiledFixedCellBuckets { .. } => "tiled-fixed-buckets",
        WgpuNeighborMode::SortedCells => "sorted-cells",
        WgpuNeighborMode::CooperativeSortedCells => "cooperative-sorted-cells",
        WgpuNeighborMode::Bvh { .. } => "bvh",
        WgpuNeighborMode::GpuBvh { .. } => "gpu-bvh",
        WgpuNeighborMode::GpuLbvh { .. } => "gpu-lbvh",
        WgpuNeighborMode::GpuMortonLbvh { .. } => "gpu-morton-lbvh",
    }
}

pub(crate) fn effective_viewer_bench_neighbor_mode(
    config: &ViewerPipelineBenchConfig,
) -> WgpuNeighborMode {
    if config.neighbor_mode == WgpuNeighborMode::Auto && config.particles <= 2048 {
        WgpuNeighborMode::SortedCells
    } else {
        config.neighbor_mode
    }
}

pub(crate) fn assert_viewer_pipeline_sample_stable(sample: &ViewerPipelineBenchSample) {
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

pub(crate) fn pump_headless_frame_timed(apps: &mut SubApps) -> Duration {
    let start = Instant::now();
    pump_headless_frame(apps);
    start.elapsed()
}

pub(crate) fn frame_time_stats(durations: &[Duration]) -> (f64, f64, f64) {
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

pub(crate) fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile)
        .round()
        .clamp(0.0, (values.len() - 1) as f64) as usize;
    values[index]
}

pub(crate) fn write_viewer_pipeline_benchmark_json(
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

pub(crate) fn workspace_target_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
}
