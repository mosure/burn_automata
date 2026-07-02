use super::prelude::*;

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn gpu_rollout_trace(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    neighbor_mode: crate::gpu::WgpuNeighborMode,
) -> Result<crate::RolloutTrace, Box<dyn std::error::Error>> {
    if cfg.batch_size != 1 {
        return Err(std::io::Error::other("infer --gpu currently supports batch_size=1").into());
    }
    let (mut positions, mut states) = seed_particles_scaled(
        cfg.batch_size,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let executor = crate::gpu::WgpuAutomataExecutor::new_blocking()?;
    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        cfg.batch_size,
        cfg.particle_count,
        grid,
        cfg.dt,
        neighbor_mode,
        cfg.update_prob,
        cfg.seed,
    )?;
    let mut mean_dx = Vec::with_capacity(cfg.steps);
    for _ in 0..cfg.steps {
        let before = positions.clone();
        executor.step_state(&mut state)?;
        let output = executor.read_state(&state)?;
        let dx_norm = output
            .next_positions
            .iter()
            .zip(before.iter())
            .map(|(next, prev)| {
                let mut norm = 0.0_f32;
                for axis in 0..model.config.spatial_dims {
                    let diff = next[axis] - prev[axis];
                    norm += diff * diff;
                }
                norm.sqrt()
            })
            .sum::<f32>()
            / output.next_positions.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = output.next_positions;
        states = output.next_states;
    }
    Ok(crate::RolloutTrace {
        positions,
        states,
        batch_size: cfg.batch_size,
        particle_count: cfg.particle_count,
        state_dims: model.config.state_dims,
        steps: cfg.steps,
        mean_dx,
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileReport {
    pub(crate) perceive_ms: f64,
    pub(crate) forward_ms: f64,
    pub(crate) integrate_ms: f64,
    pub(crate) final_mean_dx: f32,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuBenchReport {
    pub(crate) gpu_step_ms: f64,
    pub(crate) step_min_ms: f64,
    pub(crate) step_median_ms: f64,
    pub(crate) step_p95_ms: f64,
    pub(crate) step_p99_ms: f64,
    pub(crate) step_max_ms: f64,
    pub(crate) step_jitter_ratio: f64,
    pub(crate) final_mean_dx: f32,
    pub(crate) final_mean_density: f32,
    pub(crate) initial_nonempty_cells: usize,
    pub(crate) initial_max_cell_occupancy: usize,
    pub(crate) neighbor_mode: crate::gpu::WgpuNeighborMode,
    pub(crate) bucket_capacity: usize,
    pub(crate) grid_storage_len: usize,
    pub(crate) grid_clear_len: usize,
    pub(crate) grid_overflow_count: u32,
    pub(crate) grid_max_overflow_count: u32,
    pub(crate) grid_overflowed_steps: usize,
    pub(crate) gaussian_write: bool,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuBenchConfig {
    pub(crate) particles: usize,
    pub(crate) steps: usize,
    pub(crate) seed_scale: f32,
    pub(crate) update_prob: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) geometry: BenchGeometryArg,
    pub(crate) neighbor_mode: crate::gpu::WgpuNeighborMode,
    pub(crate) gaussian_write: bool,
    pub(crate) step_timing: bool,
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuBenchSummary {
    pub(crate) repeats: usize,
    pub(crate) median_report: GpuBenchReport,
    pub(crate) min_avg_step_ms: f64,
    pub(crate) median_avg_step_ms: f64,
    pub(crate) max_avg_step_ms: f64,
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn summarize_gpu_reports(reports: &[GpuBenchReport], steps: usize) -> GpuBenchSummary {
    let steps = steps.max(1) as f64;
    let mut sorted = reports.to_vec();
    sorted.sort_by(|lhs, rhs| {
        let lhs_step = lhs.gpu_step_ms / steps;
        let rhs_step = rhs.gpu_step_ms / steps;
        lhs_step.total_cmp(&rhs_step)
    });
    let median_index = sorted.len() / 2;
    GpuBenchSummary {
        repeats: reports.len(),
        median_report: sorted[median_index],
        min_avg_step_ms: sorted
            .first()
            .map(|report| report.gpu_step_ms / steps)
            .unwrap_or(0.0),
        median_avg_step_ms: sorted[median_index].gpu_step_ms / steps,
        max_avg_step_ms: sorted
            .last()
            .map(|report| report.gpu_step_ms / steps)
            .unwrap_or(0.0),
    }
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn gpu_rollout_bench(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: GpuBenchConfig,
) -> Result<GpuBenchReport, Box<dyn std::error::Error>> {
    let (positions, states) = bench_particles(
        model,
        grid,
        cfg.particles,
        cfg.seed_scale,
        cfg.seed_mode,
        cfg.geometry,
        42,
    );
    let initial_grid = build_hashgrid(&positions, 1, cfg.particles, grid)?;
    let (initial_nonempty_cells, initial_max_cell_occupancy) =
        hashgrid_occupancy_stats(&initial_grid.bin_offsets);
    let mut report = GpuBenchReport {
        gpu_step_ms: 0.0,
        step_min_ms: 0.0,
        step_median_ms: 0.0,
        step_p95_ms: 0.0,
        step_p99_ms: 0.0,
        step_max_ms: 0.0,
        step_jitter_ratio: 0.0,
        final_mean_dx: 0.0,
        final_mean_density: 0.0,
        initial_nonempty_cells,
        initial_max_cell_occupancy,
        neighbor_mode: crate::gpu::WgpuNeighborMode::Auto,
        bucket_capacity: 0,
        grid_storage_len: 0,
        grid_clear_len: 0,
        grid_overflow_count: 0,
        grid_max_overflow_count: 0,
        grid_overflowed_steps: 0,
        gaussian_write: cfg.gaussian_write,
    };
    let executor = crate::gpu::WgpuAutomataExecutor::new_blocking()?;
    let mut warmup_state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        1,
        cfg.particles,
        grid,
        1.0,
        cfg.neighbor_mode,
        cfg.update_prob,
        42,
    )?;
    executor.step_state(&mut warmup_state)?;
    executor.wait_idle()?;

    let mut state = executor.create_state_with_neighbor_mode_and_update_prob(
        model,
        &positions,
        &states,
        1,
        cfg.particles,
        grid,
        1.0,
        cfg.neighbor_mode,
        cfg.update_prob,
        42,
    )?;
    let neighbor = executor.neighbor_report(&state);
    let gaussian_buffers = if cfg.gaussian_write {
        Some(executor.create_gaussian_buffers(cfg.particles)?)
    } else {
        None
    };
    let gaussian_bind_group = gaussian_buffers
        .as_ref()
        .map(|buffers| executor.create_gaussian_bind_group(&buffers.refs(), cfg.particles))
        .transpose()?;
    report.neighbor_mode = neighbor.mode;
    report.bucket_capacity = neighbor.bucket_capacity;
    report.grid_storage_len = neighbor.grid_storage_len;
    report.grid_clear_len = neighbor.grid_clear_len;
    if cfg.step_timing {
        let mut step_ms = Vec::with_capacity(cfg.steps);
        for _ in 0..cfg.steps {
            let started = Instant::now();
            if let Some(bind_group) = gaussian_bind_group.as_ref() {
                executor.step_state_into_gaussian_bind_group(&mut state, bind_group)?;
            } else {
                executor.step_state(&mut state)?;
            }
            executor.wait_idle()?;
            step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let overflow = executor.read_grid_overflow(&state)?;
            report.grid_max_overflow_count = report.grid_max_overflow_count.max(overflow);
            report.grid_overflowed_steps += usize::from(overflow > 0);
        }
        let stats = latency_stats(&step_ms);
        report.gpu_step_ms = stats.total_ms;
        report.step_min_ms = stats.min_ms;
        report.step_median_ms = stats.median_ms;
        report.step_p95_ms = stats.p95_ms;
        report.step_p99_ms = stats.p99_ms;
        report.step_max_ms = stats.max_ms;
        report.step_jitter_ratio = stats.jitter_ratio;
    } else {
        let started = Instant::now();
        for _ in 0..cfg.steps {
            if let Some(bind_group) = gaussian_bind_group.as_ref() {
                executor.step_state_into_gaussian_bind_group(&mut state, bind_group)?;
            } else {
                executor.step_state(&mut state)?;
            }
        }
        executor.wait_idle()?;
        report.gpu_step_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    report.grid_overflow_count = executor.read_grid_overflow(&state)?;
    report.grid_max_overflow_count = report
        .grid_max_overflow_count
        .max(report.grid_overflow_count);
    report.grid_overflowed_steps += usize::from(!cfg.step_timing && report.grid_overflow_count > 0);
    let output = executor.read_state(&state)?;
    report.final_mean_dx = output
        .next_positions
        .iter()
        .zip(positions.iter())
        .map(|(next, prev)| {
            let mut norm = 0.0;
            for axis in 0..model.config.spatial_dims {
                let diff = next[axis] - prev[axis];
                norm += diff * diff;
            }
            norm.sqrt()
        })
        .sum::<f32>()
        / output.next_positions.len().max(1) as f32
        / cfg.steps.max(1) as f32;
    report.final_mean_density =
        output.density.iter().copied().sum::<f32>() / output.density.len().max(1) as f32;
    Ok(report)
}

#[cfg(feature = "gpu_wgpu")]
#[derive(Clone, Copy, Debug, Default)]
struct LatencyStats {
    total_ms: f64,
    min_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    jitter_ratio: f64,
}

#[cfg(feature = "gpu_wgpu")]
fn latency_stats(samples: &[f64]) -> LatencyStats {
    if samples.is_empty() {
        return LatencyStats::default();
    }
    let total_ms = samples.iter().sum();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let min_ms = sorted[0];
    let median_ms = percentile_nearest_rank(&sorted, 50.0);
    let p95_ms = percentile_nearest_rank(&sorted, 95.0);
    let p99_ms = percentile_nearest_rank(&sorted, 99.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let jitter_ratio = if median_ms > 0.0 {
        max_ms / median_ms
    } else {
        0.0
    };
    LatencyStats {
        total_ms,
        min_ms,
        median_ms,
        p95_ms,
        p99_ms,
        max_ms,
        jitter_ratio,
    }
}

#[cfg(feature = "gpu_wgpu")]
fn percentile_nearest_rank(sorted_samples: &[f64], percentile: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    let percentile = percentile.clamp(0.0, 100.0);
    let rank = (percentile / 100.0 * sorted_samples.len() as f64).ceil() as usize;
    sorted_samples[rank.saturating_sub(1).min(sorted_samples.len() - 1)]
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn wgpu_neighbor_mode(
    mode: NeighborModeArg,
    bucket_capacity: Option<usize>,
) -> crate::gpu::WgpuNeighborMode {
    match mode {
        NeighborModeArg::LinkedList => crate::gpu::WgpuNeighborMode::LinkedList,
        NeighborModeArg::Auto if bucket_capacity.is_none() => crate::gpu::WgpuNeighborMode::Auto,
        NeighborModeArg::Auto | NeighborModeArg::FixedBuckets => {
            if let Some(capacity) = bucket_capacity {
                crate::gpu::WgpuNeighborMode::FixedCellBuckets { capacity }
            } else {
                crate::gpu::WgpuNeighborMode::Auto
            }
        }
        NeighborModeArg::TiledFixedBuckets => crate::gpu::WgpuNeighborMode::TiledFixedCellBuckets {
            capacity: bucket_capacity.unwrap_or(256),
        },
        NeighborModeArg::SortedCells => crate::gpu::WgpuNeighborMode::SortedCells,
        NeighborModeArg::CooperativeSortedCells => {
            crate::gpu::WgpuNeighborMode::CooperativeSortedCells
        }
        NeighborModeArg::Bvh => crate::gpu::WgpuNeighborMode::Bvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuBvh => crate::gpu::WgpuNeighborMode::GpuBvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuLbvh => crate::gpu::WgpuNeighborMode::GpuLbvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
        NeighborModeArg::GpuMortonLbvh => crate::gpu::WgpuNeighborMode::GpuMortonLbvh {
            leaf_size: bucket_capacity.unwrap_or(16),
        },
    }
}

pub(crate) fn spatial_strategies(
    requested: SpatialStrategyArg,
    grid: &crate::kernels::HashGridConfig,
    tile_size: [usize; 3],
    bvh_leaf_size: usize,
) -> Vec<crate::kernels::SpatialStrategyKind> {
    use crate::kernels::{Boundary, HashGridMode, SpatialStrategyKind};
    match requested {
        SpatialStrategyArg::HashGrid => vec![SpatialStrategyKind::HashGrid],
        SpatialStrategyArg::TileBlocks => vec![SpatialStrategyKind::TileBlocks { tile_size }],
        SpatialStrategyArg::Bvh => vec![SpatialStrategyKind::Bvh {
            leaf_size: bvh_leaf_size,
        }],
        SpatialStrategyArg::All => {
            let mut strategies = vec![SpatialStrategyKind::HashGrid];
            if grid.boundary != Boundary::Periodic {
                strategies.push(SpatialStrategyKind::Bvh {
                    leaf_size: bvh_leaf_size,
                });
            }
            if grid.mode != HashGridMode::Particle {
                strategies.push(SpatialStrategyKind::TileBlocks { tile_size });
            }
            strategies
        }
    }
}

pub(crate) fn parse_tile_size(raw: &str) -> Result<[usize; 3], Box<dyn std::error::Error>> {
    let values = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(values.len() == 2 || values.len() == 3) {
        return Err(std::io::Error::other(
            "--tile-size expects two or three comma-separated integers",
        )
        .into());
    }
    if values.contains(&0) {
        return Err(std::io::Error::other("--tile-size values must be non-zero").into());
    }
    Ok([values[0], values[1], values.get(2).copied().unwrap_or(1)])
}

pub(crate) fn strategy_label(strategy: crate::kernels::SpatialStrategyKind) -> &'static str {
    strategy.label()
}

pub(crate) fn bench_particles(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    particles: usize,
    seed_scale: f32,
    seed_mode: ParticleSeed,
    geometry: BenchGeometryArg,
    seed: u64,
) -> (Vec<[f32; 4]>, Vec<f32>) {
    let (mut positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        seed,
        seed_mode,
        seed_scale,
    );
    apply_bench_geometry(
        &mut positions,
        model.config.spatial_dims,
        particles,
        seed_scale,
        grid,
        geometry,
        seed ^ 0x9e37_79b9,
    );
    (positions, states)
}

pub(crate) fn apply_bench_geometry(
    positions: &mut [[f32; 4]],
    spatial_dims: usize,
    particles: usize,
    scale: f32,
    grid: &crate::kernels::HashGridConfig,
    geometry: BenchGeometryArg,
    seed: u64,
) {
    if matches!(geometry, BenchGeometryArg::Seed) {
        return;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    for (idx, position) in positions.iter_mut().enumerate() {
        let local_idx = idx % particles.max(1);
        *position = match geometry {
            BenchGeometryArg::Seed => *position,
            BenchGeometryArg::Point => [0.0, 0.0, 0.0, 0.0],
            BenchGeometryArg::MicroCluster => {
                micro_cluster_position(&mut rng, spatial_dims, grid.eps)
            }
            BenchGeometryArg::Dense | BenchGeometryArg::ShiftedDense => {
                dense_ball_position(&mut rng, spatial_dims, scale)
            }
            BenchGeometryArg::Uniform | BenchGeometryArg::ShiftedUniform => {
                uniform_box_position(&mut rng, spatial_dims, scale)
            }
            BenchGeometryArg::Line => line_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Ring => ring_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Plane => plane_position(&mut rng, spatial_dims, scale, grid.eps),
            BenchGeometryArg::Shell => shell_position(&mut rng, spatial_dims, scale),
            BenchGeometryArg::Torus => torus_position(local_idx, particles, spatial_dims, scale),
        };
        if matches!(
            geometry,
            BenchGeometryArg::ShiftedDense | BenchGeometryArg::ShiftedUniform
        ) {
            shift_outside_fixed_grid(position, spatial_dims, grid, scale);
        }
    }
}

pub(crate) fn dense_ball_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let r = rng.random::<f32>().sqrt() * scale;
        [r * theta.cos(), r * theta.sin(), 0.0, 0.0]
    } else {
        let dir = sphere_direction(rng);
        let r = rng.random::<f32>().cbrt() * scale;
        [dir[0] * r, dir[1] * r, dir[2] * r, 0.0]
    }
}

pub(crate) fn micro_cluster_position(rng: &mut StdRng, spatial_dims: usize, eps: f32) -> [f32; 4] {
    let mut position = [0.0; 4];
    let center = 0.25 * eps;
    let radius = 0.0625 * eps;
    for value in position.iter_mut().take(spatial_dims) {
        *value = center + rng.random_range(-radius..radius);
    }
    position
}

pub(crate) fn uniform_box_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    let mut position = [0.0; 4];
    for value in position.iter_mut().take(spatial_dims) {
        *value = rng.random_range(-scale..scale);
    }
    position
}

pub(crate) fn line_position(
    rng: &mut StdRng,
    spatial_dims: usize,
    scale: f32,
    eps: f32,
) -> [f32; 4] {
    let mut position = [0.0; 4];
    position[0] = rng.random_range(-scale..scale);
    for value in position.iter_mut().take(spatial_dims).skip(1) {
        *value = rng.random_range(-0.125 * eps..0.125 * eps);
    }
    position
}

pub(crate) fn ring_position(
    rng: &mut StdRng,
    spatial_dims: usize,
    scale: f32,
    eps: f32,
) -> [f32; 4] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let r = scale + rng.random_range(-0.25 * eps..0.25 * eps);
    let mut position = [r * theta.cos(), r * theta.sin(), 0.0, 0.0];
    if spatial_dims == 3 {
        position[2] = rng.random_range(-0.25 * eps..0.25 * eps);
    }
    position
}

pub(crate) fn plane_position(
    rng: &mut StdRng,
    spatial_dims: usize,
    scale: f32,
    eps: f32,
) -> [f32; 4] {
    let mut position = [0.0; 4];
    position[0] = rng.random_range(-scale..scale);
    position[1] = rng.random_range(-scale..scale);
    if spatial_dims == 3 {
        position[2] = rng.random_range(-0.125 * eps..0.125 * eps);
    }
    position
}

pub(crate) fn shell_position(rng: &mut StdRng, spatial_dims: usize, scale: f32) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0]
    } else {
        let dir = sphere_direction(rng);
        [dir[0] * scale, dir[1] * scale, dir[2] * scale, 0.0]
    }
}

pub(crate) fn torus_position(
    local_idx: usize,
    particles: usize,
    spatial_dims: usize,
    scale: f32,
) -> [f32; 4] {
    if spatial_dims == 2 {
        let theta = std::f32::consts::TAU * local_idx as f32 / particles.max(1) as f32;
        return [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0];
    }
    let sample = uv_torus_sample(local_idx, particles, scale);
    [
        sample.position[0],
        sample.position[1],
        sample.position[2],
        0.0,
    ]
}

pub(crate) fn sphere_direction(rng: &mut StdRng) -> [f32; 3] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let z = rng.random_range(-1.0_f32..1.0_f32);
    let r_xy = (1.0_f32 - z * z).sqrt();
    [r_xy * theta.cos(), r_xy * theta.sin(), z]
}

pub(crate) fn shift_outside_fixed_grid(
    position: &mut [f32; 4],
    spatial_dims: usize,
    grid: &crate::kernels::HashGridConfig,
    scale: f32,
) {
    for (axis, value) in position.iter_mut().enumerate().take(spatial_dims) {
        let extent = grid.eps * grid.grid_size[axis] as f32;
        let sign = if axis == 1 { -1.0 } else { 1.0 };
        *value += sign * (extent + scale.max(grid.eps));
    }
}

#[cfg(feature = "gpu_wgpu")]
pub(crate) fn hashgrid_occupancy_stats(bin_offsets: &[usize]) -> (usize, usize) {
    bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold((0usize, 0usize), |(nonempty, max), count| {
            (nonempty + usize::from(count > 0), max.max(count))
        })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CpuProfileConfig {
    pub(crate) particles: usize,
    pub(crate) steps: usize,
    pub(crate) seed_scale: f32,
    pub(crate) update_prob: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) geometry: BenchGeometryArg,
}

pub(crate) fn profile_rollout(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    cfg: CpuProfileConfig,
) -> Result<ProfileReport, Box<dyn std::error::Error>> {
    let (mut positions, mut states) = bench_particles(
        model,
        grid,
        cfg.particles,
        cfg.seed_scale,
        cfg.seed_mode,
        cfg.geometry,
        42,
    );
    let mut report = ProfileReport {
        perceive_ms: 0.0,
        forward_ms: 0.0,
        integrate_ms: 0.0,
        final_mean_dx: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(42 ^ 0x5eed);
    for _ in 0..cfg.steps {
        let mask = stochastic_mask(cfg.particles, cfg.update_prob, &mut rng);
        let started = Instant::now();
        let perception = perceive_with_options(
            &positions,
            &states,
            1,
            cfg.particles,
            model.config.state_dims,
            grid,
            PerceptionOptions {
                state_grad: model.config.state_grad,
                density_grad: model.config.density_grad,
                eps0: model.config.eps0,
                scale_equivariance: model.config.scale_equivariant(),
                particle_density_equivariance: model.config.particle_density_equivariant(),
                log_norm_grad: model.config.log_norm_grad,
                log_norm_density_grad: model.config.log_norm_density_grad,
                hybrid_state_gradient: true,
                position_features: model.config.position_features,
            },
        )?;
        report.perceive_ms += started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let (dx, ds) = model.forward_from_features_with_eps(&perception.features, grid.eps)?;
        report.forward_ms += started.elapsed().as_secs_f64() * 1000.0;

        report.final_mean_dx = dx
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / dx.len().max(1) as f32;
        let started = Instant::now();
        (positions, states) = euler_step(
            &positions,
            &states,
            &dx,
            &ds,
            1,
            cfg.particles,
            model.config.state_dims,
            grid,
            1.0,
            Some(&mask),
        )?;
        report.integrate_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(report)
}

pub(crate) fn stochastic_mask(count: usize, update_prob: f32, rng: &mut StdRng) -> Vec<f32> {
    if update_prob >= 1.0 {
        return vec![1.0; count];
    }
    if update_prob <= 0.0 {
        return vec![0.0; count];
    }
    (0..count)
        .map(|_| f32::from(rng.random::<f32>() < update_prob))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn occupancy_stats(bin_offsets: &[usize]) -> (usize, usize) {
        bin_offsets
            .windows(2)
            .map(|window| window[1] - window[0])
            .fold((0usize, 0usize), |(nonempty, max), count| {
                (nonempty + usize::from(count > 0), max.max(count))
            })
    }

    #[test]
    fn point_bench_geometry_collapses_to_one_hashgrid_cell() {
        let grid = crate::kernels::HashGridConfig::growing_2d();
        let particles = 256;
        let mut positions = vec![[1.0, -1.0, 0.5, 0.0]; particles];

        apply_bench_geometry(
            &mut positions,
            2,
            particles,
            0.5,
            &grid,
            BenchGeometryArg::Point,
            7,
        );

        assert!(positions.iter().all(|position| position[..2] == [0.0, 0.0]));
        let snapshot = crate::kernels::build_hashgrid(&positions, 1, particles, &grid).unwrap();
        assert_eq!(occupancy_stats(&snapshot.bin_offsets), (1, particles));
    }

    #[test]
    fn micro_cluster_bench_geometry_stays_inside_one_particle_cell() {
        let grid = crate::kernels::HashGridConfig::growing_3dgs();
        let particles = 512;
        let mut positions = vec![[0.0; 4]; particles];

        apply_bench_geometry(
            &mut positions,
            3,
            particles,
            0.5,
            &grid,
            BenchGeometryArg::MicroCluster,
            11,
        );

        for position in &positions {
            for coordinate in position.iter().take(3) {
                assert!(*coordinate > 0.0);
                assert!(*coordinate < grid.eps);
            }
        }
        let snapshot = crate::kernels::build_hashgrid(&positions, 1, particles, &grid).unwrap();
        assert_eq!(occupancy_stats(&snapshot.bin_offsets), (1, particles));
    }

    #[cfg(feature = "gpu_wgpu")]
    #[test]
    fn latency_stats_reports_tail_step_spikes() {
        let stats = latency_stats(&[1.0, 2.0, 3.0, 100.0]);

        assert_eq!(stats.total_ms, 106.0);
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.median_ms, 2.0);
        assert_eq!(stats.p95_ms, 100.0);
        assert_eq!(stats.p99_ms, 100.0);
        assert_eq!(stats.max_ms, 100.0);
        assert_eq!(stats.jitter_ratio, 50.0);
    }
}
