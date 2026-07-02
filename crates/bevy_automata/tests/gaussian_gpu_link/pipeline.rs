use super::common::*;

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
