use super::super::*;
use super::*;

pub(crate) fn seed_occupancy_stats(
    grid: &burn_automata::kernels::HashGridConfig,
    particles: usize,
    state_dims: usize,
    seed_scale: f32,
    _reference_scale: f32,
    _normalize: bool,
) -> (usize, usize) {
    let (positions, _states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        42,
        ParticleSeed::TorusMorphogenDense3d,
        seed_scale,
    );
    let snapshot = build_hashgrid(&positions, 1, particles, grid).unwrap();
    snapshot
        .bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold((0usize, 0usize), |(nonempty, max), count| {
            (nonempty + usize::from(count > 0), max.max(count))
        })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GrowthValidationCase {
    pub(crate) relative_path: &'static str,
    pub(crate) target: GrowthTarget,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) seed_scale: f32,
    pub(crate) particle_count: usize,
    pub(crate) steps: usize,
}

impl GrowthValidationCase {
    pub(crate) fn torus(relative_path: &'static str) -> Self {
        Self {
            relative_path,
            target: GrowthTarget::Torus,
            seed_mode: ParticleSeed::TorusGrowth3d,
            seed_scale: 0.72,
            particle_count: 512,
            steps: 64,
        }
    }

    pub(crate) fn teapot(relative_path: &'static str) -> Self {
        Self {
            relative_path,
            target: GrowthTarget::Teapot,
            seed_mode: ParticleSeed::TeapotGrowth3d,
            seed_scale: 0.72,
            particle_count: 1024,
            steps: 64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GrowthTarget {
    Torus,
    Teapot,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CatalogRenderSanityCase {
    pub(crate) validation: GrowthValidationCase,
    pub(crate) max_total_loss: f32,
    pub(crate) min_density_psnr_db: f32,
    pub(crate) min_color_psnr_db: f32,
    pub(crate) min_depth_psnr_db: f32,
}

#[derive(Debug)]
pub(crate) struct StrictGrowthValidationReport {
    pub(crate) position_features: bool,
    pub(crate) active_seed_count: usize,
    pub(crate) final_active_count: usize,
    pub(crate) newly_activated_count: usize,
    pub(crate) newly_activated_fraction: f32,
    pub(crate) final_active_mean_radius: f32,
    pub(crate) final_active_max_radius: f32,
    pub(crate) non_opacity_seed_abs_max: f32,
    pub(crate) mean_final_displacement: f32,
    pub(crate) initial_surface: SurfaceStats,
    pub(crate) final_surface: SurfaceStats,
    pub(crate) initial_active_surface: SurfaceStats,
    pub(crate) final_active_surface: SurfaceStats,
    pub(crate) initial_target_coverage: TargetCoverageStats,
    pub(crate) final_target_coverage: TargetCoverageStats,
    pub(crate) max_motion_per_step: f32,
    pub(crate) render_density_psnr_db: f32,
    pub(crate) render_color_psnr_db: f32,
    pub(crate) render_depth_psnr_db: f32,
    pub(crate) temporal_progressive_activation: bool,
    pub(crate) temporal_geometry_progressive: MeasuredBool,
    pub(crate) front_coherence: FrontCoherenceReport,
    pub(crate) strict_passed: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ColorStateStats {
    pub(crate) active_mean_abs: f32,
    pub(crate) active_max_abs: f32,
    pub(crate) active_channel_stddev_mean: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasuredBool {
    pub(crate) passed: bool,
    pub(crate) surface_mean_ratio: f32,
    pub(crate) target_coverage_mean_ratio: f32,
    pub(crate) target_coverage_fraction_delta: f32,
}

impl MeasuredBool {
    pub(crate) fn is_finite(self) -> bool {
        self.surface_mean_ratio.is_finite()
            && self.target_coverage_mean_ratio.is_finite()
            && self.target_coverage_fraction_delta.is_finite()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FrontCoherenceReport {
    pub(crate) passed: bool,
    pub(crate) transition_count: usize,
    pub(crate) newly_activated_count: usize,
    pub(crate) local_newly_activated_fraction: f32,
    pub(crate) mean_nearest_previous_active_distance: f32,
    pub(crate) max_nearest_previous_active_distance: f32,
    pub(crate) max_allowed_distance: f32,
}

#[derive(Debug)]
pub(crate) struct CatalogRenderSanityReport {
    pub(crate) total_loss: f32,
    pub(crate) density_psnr_db: f32,
    pub(crate) color_psnr_db: f32,
    pub(crate) depth_psnr_db: f32,
}

pub(crate) fn catalog_render_sanity_report(
    case: GrowthValidationCase,
) -> CatalogRenderSanityReport {
    let manifest =
        burn_automata::import::load_manifest(workspace_path(case.relative_path)).unwrap();
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let cfg = RolloutConfig {
        particle_count: 512,
        steps: case.steps,
        update_prob: 1.0,
        seed: CATALOG_3D_GROWTH_SEED,
        seed_scale: case.seed_scale,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&model, &grid, &cfg, case.seed_mode).unwrap();
    let target = match case.target {
        GrowthTarget::Torus => TriangleMeshTarget::torus(
            case.seed_scale,
            case.seed_scale * UV_TORUS_MINOR_RATIO,
            64,
            48,
        )
        .unwrap(),
        GrowthTarget::Teapot => TriangleMeshTarget::utah_teapot(case.seed_scale).unwrap(),
    };
    let render = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 48,
            target_samples: 1024,
            world_scale: case.seed_scale * 2.0,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();

    CatalogRenderSanityReport {
        total_loss: render.total_loss,
        density_psnr_db: render.density_psnr_db,
        color_psnr_db: render.color_psnr_db,
        depth_psnr_db: render.depth_psnr_db,
    }
}

pub(crate) fn strict_growth_validation_report(
    case: GrowthValidationCase,
) -> StrictGrowthValidationReport {
    let manifest =
        burn_automata::import::load_manifest(workspace_path(case.relative_path)).unwrap();
    let position_features = manifest.config.position_features;
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let cfg = RolloutConfig {
        particle_count: case.particle_count,
        steps: case.steps,
        update_prob: 1.0,
        seed: CATALOG_3D_GROWTH_SEED,
        seed_scale: case.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        case.seed_mode,
        cfg.seed_scale,
    );
    let mut active_seed_count = 0usize;
    let mut non_opacity_seed_abs_max = 0.0_f32;
    let mut seed_active = Vec::with_capacity(cfg.particle_count);
    let material_opacity_channel = growth_3d_material_opacity_channel(model.config.state_dims);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
        for (channel, value) in state.iter().enumerate() {
            if channel != 3 && Some(channel) != material_opacity_channel && channel >= 3 {
                non_opacity_seed_abs_max = non_opacity_seed_abs_max.max(value.abs());
            }
        }
    }

    let trace = run_rollout(&model, &grid, &cfg, case.seed_mode).unwrap();
    let mut final_active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut final_active_radius_sum = 0.0_f32;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in trace.positions.iter().enumerate() {
        let opacity = trace.states[idx * model.config.state_dims + 3];
        if opacity > -1.0 {
            final_active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            final_active_radius_sum += radius;
            final_active_max_radius = final_active_max_radius.max(radius);
        }
    }
    let inactive_seed_count = cfg.particle_count.saturating_sub(active_seed_count);
    let newly_activated_fraction = newly_activated_count as f32 / inactive_seed_count.max(1) as f32;
    let final_active_mean_radius = final_active_radius_sum / final_active_count.max(1) as f32;
    let mean_final_displacement = mean_displacement(&seed_positions, &trace.positions);
    let max_motion_per_step = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
    let target = match case.target {
        GrowthTarget::Torus => TriangleMeshTarget::torus(
            case.seed_scale,
            case.seed_scale * UV_TORUS_MINOR_RATIO,
            64,
            48,
        )
        .unwrap(),
        GrowthTarget::Teapot => TriangleMeshTarget::utah_teapot(case.seed_scale).unwrap(),
    };
    let temporal_progressive_activation = temporal_progressive_activation_report(
        &model,
        &grid,
        &cfg,
        case.seed_mode,
        active_seed_count,
    );
    let front_coherence = front_coherence_report(
        &model,
        &grid,
        &cfg,
        case.seed_mode,
        &trace,
        &seed_positions,
        &seed_states,
    );
    let temporal_geometry_progressive =
        temporal_geometry_progressive_report(&model, &grid, &cfg, case.seed_mode, &target);
    let initial_surface = mesh_surface_stats(&seed_positions, &target);
    let final_surface = mesh_surface_stats(&trace.positions, &target);
    let initial_active_surface = mesh_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
    );
    let final_active_surface =
        mesh_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, &target);
    let coverage_threshold = target_coverage_threshold(case.seed_scale);
    let initial_target_coverage = target_coverage_stats(
        &seed_positions,
        &target,
        case.particle_count.max(512),
        coverage_threshold,
    );
    let final_target_coverage = target_coverage_stats(
        &trace.positions,
        &target,
        case.particle_count.max(512),
        coverage_threshold,
    );
    let render = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 32,
            target_samples: case.particle_count.max(512),
            world_scale: case.seed_scale * 2.0,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();
    let strict_passed = !position_features
        && non_opacity_seed_abs_max <= 1.0e-6
        && active_seed_count > 0
        && active_seed_count < case.particle_count / 8
        && final_active_count > active_seed_count * 4
        && newly_activated_fraction >= 0.50
        && temporal_progressive_activation
        && front_coherence.passed
        && temporal_geometry_progressive.passed
        && final_active_max_radius > growth_3d_seed_radius(case.seed_scale)
        && max_motion_per_step > 0.01
        && mean_final_displacement > growth_3d_seed_radius(case.seed_scale)
        && final_active_surface.mean < initial_active_surface.mean * 0.85
        && final_active_surface.max < 0.36
        && final_target_coverage.mean < initial_target_coverage.mean * 0.85
        && final_target_coverage.max < case.seed_scale
        && final_target_coverage.covered_fraction >= 0.60
        && render.passed;

    StrictGrowthValidationReport {
        position_features,
        active_seed_count,
        final_active_count,
        newly_activated_count,
        newly_activated_fraction,
        final_active_mean_radius,
        final_active_max_radius,
        non_opacity_seed_abs_max,
        mean_final_displacement,
        initial_surface,
        final_surface,
        initial_active_surface,
        final_active_surface,
        initial_target_coverage,
        final_target_coverage,
        max_motion_per_step,
        render_density_psnr_db: render.density_psnr_db,
        render_color_psnr_db: render.color_psnr_db,
        render_depth_psnr_db: render.depth_psnr_db,
        temporal_progressive_activation,
        temporal_geometry_progressive,
        front_coherence,
        strict_passed,
    }
}

pub(crate) fn color_state_stats(states: &[f32], state_dims: usize) -> ColorStateStats {
    assert!(
        state_dims >= 6,
        "3D growth color validation expects tail color state"
    );
    let tail = state_dims - 3;
    let mut active_count = 0usize;
    let mut active_sum_abs = 0.0_f32;
    let mut active_max_abs = 0.0_f32;
    let mut active_sum = [0.0_f32; 3];
    let mut active_sum_sq = [0.0_f32; 3];

    for state in states.chunks_exact(state_dims) {
        if state[3] <= -1.0 {
            continue;
        }
        active_count += 1;
        let mut particle_max_abs = 0.0_f32;
        for channel in 0..3 {
            let value = state[tail + channel];
            assert!(value.is_finite(), "non-finite tail color state {value}");
            particle_max_abs = particle_max_abs.max(value.abs());
            active_sum[channel] += value;
            active_sum_sq[channel] += value * value;
        }
        active_sum_abs += particle_max_abs;
        active_max_abs = active_max_abs.max(particle_max_abs);
    }

    let mut active_channel_stddev = [0.0_f32; 3];
    if active_count > 0 {
        for channel in 0..3 {
            let mean = active_sum[channel] / active_count as f32;
            let variance = (active_sum_sq[channel] / active_count as f32 - mean * mean).max(0.0);
            active_channel_stddev[channel] = variance.sqrt();
        }
    }

    ColorStateStats {
        active_mean_abs: active_sum_abs / active_count.max(1) as f32,
        active_max_abs,
        active_channel_stddev_mean: active_channel_stddev.iter().sum::<f32>() / 3.0,
    }
}

fn temporal_geometry_progressive_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    target: &TriangleMeshTarget,
) -> MeasuredBool {
    let mut samples = vec![0usize, cfg.steps];
    let mut step = 1usize;
    while step < cfg.steps {
        samples.push(step);
        step *= 2;
    }
    samples.sort_unstable();
    samples.dedup();

    let mut initial = None;
    let mut final_sample = None;
    for steps in samples {
        let (positions, states, state_dims) = if steps == 0 {
            let (positions, states) = seed_particles_scaled(
                cfg.batch_size,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            (positions, states, model.config.state_dims)
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            (trace.positions, trace.states, trace.state_dims)
        };
        let sample = (
            mesh_active_surface_stats(&positions, &states, state_dims, target),
            target_coverage_stats(
                &positions,
                target,
                cfg.particle_count.max(512),
                target_coverage_threshold(cfg.seed_scale),
            ),
        );
        if steps == 0 {
            initial = Some(sample);
        }
        if steps == cfg.steps {
            final_sample = Some(sample);
        }
    }

    let ((initial_surface, initial_coverage), (final_surface, final_coverage)) =
        match (initial, final_sample) {
            (Some(initial), Some(final_sample)) => (initial, final_sample),
            _ => {
                return MeasuredBool {
                    passed: false,
                    surface_mean_ratio: f32::INFINITY,
                    target_coverage_mean_ratio: f32::INFINITY,
                    target_coverage_fraction_delta: 0.0,
                };
            }
        };
    let surface_mean_ratio = final_surface.mean / initial_surface.mean.max(1.0e-6);
    let target_coverage_mean_ratio = final_coverage.mean / initial_coverage.mean.max(1.0e-6);
    let target_coverage_fraction_delta =
        final_coverage.covered_fraction - initial_coverage.covered_fraction;

    MeasuredBool {
        passed: target_coverage_mean_ratio < 0.85
            && target_coverage_fraction_delta >= 0.10
            && surface_mean_ratio < 0.95,
        surface_mean_ratio,
        target_coverage_mean_ratio,
        target_coverage_fraction_delta,
    }
}

fn temporal_progressive_activation_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    active_seed_count: usize,
) -> bool {
    let mut samples = vec![0usize, cfg.steps];
    let mut step = 1usize;
    while step < cfg.steps {
        samples.push(step);
        step *= 2;
    }
    samples.sort_unstable();
    samples.dedup();

    let mut first_growth_step = None;
    let mut half_activation_step = None;
    let mut full_activation_step = None;
    for steps in samples {
        let active_count = if steps == 0 {
            let (_positions, states) = seed_particles_scaled(
                cfg.batch_size,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            states
                .chunks_exact(model.config.state_dims)
                .filter(|state| state[3] > -1.0)
                .count()
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            trace
                .states
                .chunks_exact(trace.state_dims)
                .filter(|state| state[3] > -1.0)
                .count()
        };
        let active_fraction = active_count as f32 / cfg.particle_count.max(1) as f32;
        if first_growth_step.is_none()
            && active_count > active_seed_count
            && active_count >= active_seed_count.saturating_mul(2).max(1)
        {
            first_growth_step = Some(steps);
        }
        if half_activation_step.is_none() && active_fraction >= 0.50 {
            half_activation_step = Some(steps);
        }
        if full_activation_step.is_none() && active_fraction >= 0.95 {
            full_activation_step = Some(steps);
        }
    }

    match (
        first_growth_step,
        half_activation_step,
        full_activation_step,
    ) {
        (Some(first), Some(half), Some(full)) => {
            first < half && half < full && full.saturating_sub(first) >= cfg.steps / 4
        }
        _ => false,
    }
}

fn front_coherence_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    final_trace: &burn_automata::RolloutTrace,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> FrontCoherenceReport {
    let max_allowed_distance = growth_3d_seed_radius(cfg.seed_scale) * 2.5;
    let mut previous: Option<(Vec<[f32; 4]>, Vec<bool>)> = None;
    let mut transition_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut local_newly_activated_count = 0usize;
    let mut sum_nearest = 0.0_f32;
    let mut max_nearest = 0.0_f32;
    let mut finite = true;

    for steps in temporal_sample_steps(cfg.steps) {
        let (positions, states, state_dims) = if steps == 0 {
            (
                seed_positions.to_vec(),
                seed_states.to_vec(),
                model.config.state_dims,
            )
        } else if steps == cfg.steps {
            (
                final_trace.positions.clone(),
                final_trace.states.clone(),
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            (trace.positions, trace.states, trace.state_dims)
        };
        let active = active_flags(&states, state_dims);
        if let Some((previous_positions, previous_active)) = previous.take() {
            let previous_active_positions = previous_positions
                .iter()
                .zip(previous_active.iter())
                .filter_map(|(position, active)| (*active).then_some(*position))
                .collect::<Vec<_>>();
            let mut transition_newly_activated = 0usize;
            for idx in 0..active.len() {
                if !active[idx] || previous_active[idx] || previous_active_positions.is_empty() {
                    continue;
                }
                transition_newly_activated += 1;
                newly_activated_count += 1;
                let distance = nearest_distance(positions[idx], &previous_active_positions);
                finite &= distance.is_finite();
                sum_nearest += distance;
                max_nearest = max_nearest.max(distance);
                if distance <= max_allowed_distance {
                    local_newly_activated_count += 1;
                }
            }
            if transition_newly_activated > 0 {
                transition_count += 1;
            }
        }
        previous = Some((positions, active));
    }

    let local_newly_activated_fraction = if newly_activated_count > 0 {
        local_newly_activated_count as f32 / newly_activated_count as f32
    } else {
        0.0
    };
    let mean_nearest_previous_active_distance = if newly_activated_count > 0 {
        sum_nearest / newly_activated_count as f32
    } else {
        f32::INFINITY
    };
    let passed = finite
        && newly_activated_count > 0
        && transition_count >= 2
        && local_newly_activated_fraction >= 0.90
        && mean_nearest_previous_active_distance <= max_allowed_distance * 0.75;

    FrontCoherenceReport {
        passed,
        transition_count,
        newly_activated_count,
        local_newly_activated_fraction,
        mean_nearest_previous_active_distance,
        max_nearest_previous_active_distance: if newly_activated_count > 0 {
            max_nearest
        } else {
            f32::INFINITY
        },
        max_allowed_distance,
    }
}

fn temporal_sample_steps(steps: usize) -> Vec<usize> {
    let mut samples = vec![0usize, steps];
    let mut step = 1usize;
    while step < steps {
        samples.push(step);
        step = step.saturating_mul(2);
        if step == 0 {
            break;
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

fn active_flags(states: &[f32], state_dims: usize) -> Vec<bool> {
    states
        .chunks_exact(state_dims)
        .map(|state| state_dims > 3 && state[3] > -1.0)
        .collect()
}

fn nearest_distance(position: [f32; 4], candidates: &[[f32; 4]]) -> f32 {
    candidates
        .iter()
        .map(|candidate| {
            ((position[0] - candidate[0]).powi(2)
                + (position[1] - candidate[1]).powi(2)
                + (position[2] - candidate[2]).powi(2))
            .sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

fn mean_displacement(initial: &[[f32; 4]], final_positions: &[[f32; 4]]) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}
