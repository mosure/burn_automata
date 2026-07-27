use rayon::prelude::*;

use burn_automata_kernels::{GaussianDecodeMode, HashGridConfig};

#[cfg(feature = "gpu_wgpu")]
use crate::gpu::{WgpuAutomataExecutor, WgpuNeighborMode};
#[cfg(not(feature = "gpu_wgpu"))]
use crate::run_rollout_from_particles;
use crate::{
    AutomataError, AutomataResult, NpaModel, RenderLossConfig, RolloutConfig, RolloutTrace,
    TriangleMeshTarget, mesh_multiview_render_loss_from_trace,
    rollout::{GROWTH_3D_RENDER_OPACITY_CHANNEL, UV_TORUS_NORMAL_STATE_OFFSET},
};

use super::{
    Mesh3dEvaluationConfig, Mesh3dInitializationMode, Mesh3dQualityReport, Mesh3dRolloutReport,
    mesh3d_damaged_initialization, mesh3d_surface_initialization, mesh3d_volume_initialization,
};

pub fn evaluate_mesh3d_model(
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    target: &TriangleMeshTarget,
    config: &Mesh3dEvaluationConfig,
) -> AutomataResult<Mesh3dQualityReport> {
    validate_evaluation_config(model, hashgrid, config)?;
    let runner = Mesh3dEvaluationRunner::new()?;
    let mut horizons = config.rollout_steps.clone();
    horizons.sort_unstable();
    horizons.dedup();
    let target_positions = (0..config.target_samples)
        .map(|index| target.surface_sample(index).position)
        .collect::<Vec<_>>();
    let mut rollouts = Vec::with_capacity(config.seeds.len() * horizons.len() * 3);

    for &seed in &config.seeds {
        for initialization in [
            Mesh3dInitializationMode::UniformVolume,
            Mesh3dInitializationMode::MeshSurface,
            Mesh3dInitializationMode::MeshSurfaceDamaged,
        ] {
            let mut previous_positions: Option<Vec<[f32; 4]>> = None;
            for &steps in &horizons {
                let required_for_quality = match initialization {
                    Mesh3dInitializationMode::UniformVolume => false,
                    Mesh3dInitializationMode::MeshSurface => true,
                    Mesh3dInitializationMode::MeshSurfaceDamaged => {
                        steps >= config.recovery_min_steps
                    }
                };
                let rollout_config = RolloutConfig {
                    batch_size: 1,
                    particle_count: config.particle_count,
                    steps,
                    dt: 1.0,
                    update_prob: 0.5,
                    seed,
                    seed_scale: config.seed_scale,
                };
                let initial = match initialization {
                    Mesh3dInitializationMode::UniformVolume => mesh3d_volume_initialization(
                        target,
                        &model.config,
                        config.particle_count,
                        seed,
                        config.seed_scale,
                    )?,
                    Mesh3dInitializationMode::MeshSurface => mesh3d_surface_initialization(
                        target,
                        &model.config,
                        config.particle_count,
                        seed,
                    )?,
                    Mesh3dInitializationMode::MeshSurfaceDamaged => mesh3d_damaged_initialization(
                        target,
                        &model.config,
                        config.particle_count,
                        seed,
                        config.damage_radius,
                        config.damage_displacement,
                    )?,
                };
                let trace = runner.rollout(
                    model,
                    hashgrid,
                    &rollout_config,
                    initial.positions,
                    initial.states,
                )?;
                let report = rollout_report(
                    &trace,
                    Mesh3dRolloutContext {
                        target,
                        target_positions: &target_positions,
                        previous_positions: previous_positions.as_deref(),
                        config,
                        seed,
                        initialization,
                        required_for_quality,
                    },
                )?;
                previous_positions = Some(trace.positions.clone());
                rollouts.push(report);
            }
        }
    }
    let passed = rollouts.iter().any(|report| report.required_for_quality)
        && rollouts
            .iter()
            .filter(|report| report.required_for_quality)
            .all(|report| report.passed);
    Ok(Mesh3dQualityReport {
        passed,
        particle_count: config.particle_count,
        target_samples: config.target_samples,
        rollouts,
    })
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn evaluate_mesh3d_recovery_candidate(
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    target: &TriangleMeshTarget,
    config: &Mesh3dEvaluationConfig,
) -> AutomataResult<Mesh3dRolloutReport> {
    let mut selection = config.clone();
    selection.particle_count = selection.particle_count.min(4_096);
    selection.target_samples = selection.target_samples.min(2_048);
    selection.render_image_size = selection.render_image_size.min(64);
    selection.render_target_samples = selection.render_target_samples.min(4_096);
    selection.seeds.truncate(1);
    selection.rollout_steps = vec![selection.recovery_min_steps.max(1)];
    validate_evaluation_config(model, hashgrid, &selection)?;

    let seed = selection.seeds[0];
    let steps = selection.rollout_steps[0];
    let initial = mesh3d_damaged_initialization(
        target,
        &model.config,
        selection.particle_count,
        seed,
        selection.damage_radius,
        selection.damage_displacement,
    )?;
    let trace = Mesh3dEvaluationRunner::new()?.rollout(
        model,
        hashgrid,
        &RolloutConfig {
            batch_size: 1,
            particle_count: selection.particle_count,
            steps,
            dt: 1.0,
            update_prob: 0.5,
            seed,
            seed_scale: selection.seed_scale,
        },
        initial.positions,
        initial.states,
    )?;
    let target_positions = (0..selection.target_samples)
        .map(|index| target.surface_sample(index).position)
        .collect::<Vec<_>>();
    rollout_report(
        &trace,
        Mesh3dRolloutContext {
            target,
            target_positions: &target_positions,
            previous_positions: None,
            config: &selection,
            seed,
            initialization: Mesh3dInitializationMode::MeshSurfaceDamaged,
            required_for_quality: true,
        },
    )
}

struct Mesh3dEvaluationRunner {
    #[cfg(feature = "gpu_wgpu")]
    executor: WgpuAutomataExecutor,
}

impl Mesh3dEvaluationRunner {
    fn new() -> AutomataResult<Self> {
        Ok(Self {
            #[cfg(feature = "gpu_wgpu")]
            executor: WgpuAutomataExecutor::new_blocking()?,
        })
    }

    fn rollout(
        &self,
        model: &NpaModel,
        hashgrid: &HashGridConfig,
        config: &RolloutConfig,
        positions: Vec<[f32; 4]>,
        states: Vec<f32>,
    ) -> AutomataResult<RolloutTrace> {
        #[cfg(feature = "gpu_wgpu")]
        {
            let mut state = self
                .executor
                .create_state_with_neighbor_mode_and_update_prob(
                    model,
                    &positions,
                    &states,
                    config.batch_size,
                    config.particle_count,
                    hashgrid,
                    config.dt,
                    WgpuNeighborMode::LinkedList,
                    config.update_prob,
                    config.seed,
                )?;
            self.executor.step_state_many(&mut state, config.steps)?;
            let (positions, states) = self.executor.read_positions_states(&state)?;
            Ok(RolloutTrace {
                positions,
                states,
                batch_size: config.batch_size,
                particle_count: config.particle_count,
                state_dims: model.config.state_dims,
                steps: config.steps,
                mean_dx: Vec::new(),
            })
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            run_rollout_from_particles(model, hashgrid, config, positions, states)
        }
    }
}

struct Mesh3dRolloutContext<'a> {
    target: &'a TriangleMeshTarget,
    target_positions: &'a [[f32; 3]],
    previous_positions: Option<&'a [[f32; 4]]>,
    config: &'a Mesh3dEvaluationConfig,
    seed: u64,
    initialization: Mesh3dInitializationMode,
    required_for_quality: bool,
}

fn rollout_report(
    trace: &RolloutTrace,
    context: Mesh3dRolloutContext<'_>,
) -> AutomataResult<Mesh3dRolloutReport> {
    let Mesh3dRolloutContext {
        target,
        target_positions,
        previous_positions,
        config,
        seed,
        initialization,
        required_for_quality,
    } = context;
    let finite = trace
        .positions
        .iter()
        .flatten()
        .chain(trace.states.iter())
        .all(|value| value.is_finite());
    let mut surface_distances = trace
        .positions
        .par_iter()
        .map(|position| {
            target
                .project([position[0], position[1], position[2]])
                .distance
        })
        .collect::<Vec<_>>();
    surface_distances.sort_unstable_by(f32::total_cmp);
    let mean_surface_distance =
        surface_distances.iter().sum::<f32>() / surface_distances.len().max(1) as f32;
    let p95_surface_distance = percentile(&surface_distances, 0.95);
    let max_surface_distance = surface_distances.last().copied().unwrap_or_default();

    let coverage_distances = target_positions
        .par_iter()
        .map(|target_position| {
            trace
                .positions
                .iter()
                .map(|particle| {
                    let dx = particle[0] - target_position[0];
                    let dy = particle[1] - target_position[1];
                    let dz = particle[2] - target_position[2];
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f32::MAX, f32::min)
                .sqrt()
        })
        .collect::<Vec<_>>();
    let mean_coverage_distance =
        coverage_distances.iter().sum::<f32>() / coverage_distances.len().max(1) as f32;
    let coverage_threshold = config.max_mean_coverage_distance * 2.0;
    let coverage_fraction = coverage_distances
        .iter()
        .filter(|distance| **distance <= coverage_threshold)
        .count() as f32
        / coverage_distances.len().max(1) as f32;

    let mut normal_alignment = 0.0_f32;
    let mut mean_opacity = 0.0_f32;
    let damage_center = target
        .surface_sample((seed as usize).wrapping_mul(0x9e37_79b9))
        .position;
    let color_tail = trace.state_dims - 3;
    let mut particle_color_squared_error = 0.0_f64;
    let mut damage_color_squared_error = 0.0_f64;
    let mut damage_opacity = 0.0_f32;
    let mut damage_particles = 0usize;
    for (row, position) in trace.positions.iter().enumerate() {
        let projection = target.project([position[0], position[1], position[2]]);
        let state = &trace.states[row * trace.state_dims..(row + 1) * trace.state_dims];
        let normal = [
            state[UV_TORUS_NORMAL_STATE_OFFSET],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 1],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let normal_norm =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if normal_norm > 1.0e-6 {
            normal_alignment += ((normal[0] * projection.normal[0]
                + normal[1] * projection.normal[1]
                + normal[2] * projection.normal[2])
                / normal_norm)
                .abs();
        }
        let opacity = sigmoid(state[GROWTH_3D_RENDER_OPACITY_CHANNEL]);
        mean_opacity += opacity;
        let in_damage_region = {
            let dx = position[0] - damage_center[0];
            let dy = position[1] - damage_center[1];
            let dz = position[2] - damage_center[2];
            dx * dx + dy * dy + dz * dz <= config.damage_radius * config.damage_radius
        };
        for channel in 0..3 {
            let decoded = (state[color_tail + channel] + 0.5).clamp(0.0, 1.0);
            let error = f64::from(decoded - projection.color[channel]);
            particle_color_squared_error += error * error;
            if in_damage_region {
                damage_color_squared_error += error * error;
            }
        }
        if in_damage_region {
            damage_opacity += opacity;
            damage_particles += 1;
        }
    }
    normal_alignment /= trace.particle_count.max(1) as f32;
    mean_opacity /= trace.particle_count.max(1) as f32;
    let particle_color_psnr_db = psnr_from_squared_error(
        particle_color_squared_error,
        trace.particle_count.saturating_mul(3),
    );
    let damage_region_color_psnr_db = psnr_from_squared_error(
        damage_color_squared_error,
        damage_particles.saturating_mul(3),
    );
    let damage_region_mean_opacity = damage_opacity / damage_particles.max(1) as f32;
    let damage_region_particle_fraction =
        damage_particles as f32 / trace.particle_count.max(1) as f32;

    let render = mesh_multiview_render_loss_from_trace(
        trace,
        target,
        RenderLossConfig {
            image_size: config.render_image_size,
            target_samples: config.render_target_samples,
            sigma: 1.35,
            min_sigma: 0.5,
            max_sigma: 3.0,
            gaussian_decode_mode: GaussianDecodeMode::GaussianSh0Oriented,
            world_scale: 0.9,
            opacity_logit_bias: 0.0,
            density_weight: 1.0,
            color_weight: 1.0,
            depth_weight: 0.25,
        },
    )?;
    let drift_from_previous_horizon = previous_positions.map(|previous| {
        trace
            .positions
            .iter()
            .zip(previous)
            .map(|(current, previous)| {
                let dx = current[0] - previous[0];
                let dy = current[1] - previous[1];
                let dz = current[2] - previous[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum::<f32>()
            / trace.particle_count.max(1) as f32
    });
    let passed = finite
        && mean_surface_distance <= config.max_mean_surface_distance
        && p95_surface_distance <= config.max_p95_surface_distance
        && mean_coverage_distance <= config.max_mean_coverage_distance
        && coverage_fraction >= config.min_coverage_fraction
        && render.density_psnr_db >= config.min_density_psnr_db
        && render.color_psnr_db >= config.min_color_psnr_db
        && (initialization != Mesh3dInitializationMode::MeshSurfaceDamaged
            || damage_region_color_psnr_db >= config.min_damage_region_color_psnr_db)
        && drift_from_previous_horizon.is_none_or(|drift| drift <= config.max_long_horizon_drift);
    Ok(Mesh3dRolloutReport {
        initialization,
        required_for_quality,
        seed,
        steps: trace.steps,
        finite,
        mean_surface_distance,
        p95_surface_distance,
        max_surface_distance,
        mean_coverage_distance,
        coverage_fraction,
        density_psnr_db: render.density_psnr_db,
        color_psnr_db: render.color_psnr_db,
        particle_color_psnr_db,
        damage_region_color_psnr_db,
        damage_region_mean_opacity,
        damage_region_particle_fraction,
        depth_psnr_db: render.depth_psnr_db,
        mean_normal_alignment: normal_alignment,
        mean_opacity,
        drift_from_previous_horizon,
        passed,
    })
}

fn psnr_from_squared_error(squared_error: f64, values: usize) -> f32 {
    let mse = squared_error / values.max(1) as f64;
    if mse <= f64::EPSILON {
        f32::INFINITY
    } else {
        (-10.0 * mse.log10()) as f32
    }
}

fn percentile(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn validate_evaluation_config(
    model: &NpaModel,
    hashgrid: &HashGridConfig,
    config: &Mesh3dEvaluationConfig,
) -> AutomataResult<()> {
    model.validate()?;
    if model.config.spatial_dims != 3 || hashgrid.dim != 3 {
        return Err(AutomataError::InvalidArgument(
            "mesh3d evaluation requires a 3D model and hashgrid".to_string(),
        ));
    }
    if config.particle_count == 0
        || config.target_samples == 0
        || config.render_image_size == 0
        || config.render_target_samples == 0
        || config.rollout_steps.is_empty()
        || config.seeds.is_empty()
        || !config.damage_radius.is_finite()
        || config.damage_radius <= 0.0
        || !config.damage_displacement.is_finite()
        || config.damage_displacement < 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "mesh3d evaluation counts, horizons, and seeds must be non-empty".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_sorted_nearest_rank() {
        assert_eq!(percentile(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.0), 0.0);
        assert_eq!(percentile(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.5), 2.0);
        assert_eq!(percentile(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.95), 4.0);
    }
}
