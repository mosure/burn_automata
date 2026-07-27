use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use burn_automata_kernels::{HashGridConfig, PerceptionOptions, perceive_with_options};

use crate::{
    AutomataError, AutomataResult, NpaConfig, NpaParticleInitialization, ParticleSeed,
    SupervisedBatch, TriangleMeshTarget,
    rollout::{
        GROWTH_3D_LIVENESS_CHANNEL, GROWTH_3D_RENDER_OPACITY_CHANNEL, UV_TORUS_NORMAL_STATE_OFFSET,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET, position_conditioned_3d_maturity_gate,
        seed_particles_scaled,
    },
};

use super::Mesh3dTrainingConfig;

const TARGET_OPACITY_LOGIT: f32 = 4.0;

pub fn mesh3d_surface_initialization(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    particle_count: usize,
    seed: u64,
) -> AutomataResult<NpaParticleInitialization> {
    if model_config.spatial_dims != 3 || particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "mesh3d surface initialization requires a 3D model and non-zero particle count"
                .to_string(),
        ));
    }
    let mut positions = Vec::with_capacity(particle_count);
    let mut states = vec![0.0_f32; particle_count * model_config.state_dims];
    let offset = (seed as usize).wrapping_mul(0x9e37_79b9);
    for row in 0..particle_count {
        let sample = target.surface_sample(row.wrapping_add(offset));
        positions.push([
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ]);
        let state = &mut states[row * model_config.state_dims..(row + 1) * model_config.state_dims];
        write_mesh_surface_state(state, sample.normal, sample.color);
    }
    let initialization = NpaParticleInitialization { positions, states };
    initialization.validate(model_config)?;
    Ok(initialization)
}

pub fn mesh3d_volume_initialization(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    particle_count: usize,
    seed: u64,
    scale: f32,
) -> AutomataResult<NpaParticleInitialization> {
    if model_config.spatial_dims != 3 || particle_count == 0 {
        return Err(AutomataError::InvalidArgument(
            "mesh3d volume initialization requires a 3D model and non-zero particle count"
                .to_string(),
        ));
    }
    let (positions, mut states) = seed_particles_scaled(
        1,
        particle_count,
        model_config.state_dims,
        model_config.spatial_dims,
        seed,
        ParticleSeed::UniformCircle,
        scale,
    );
    write_mesh_signed_distance_state(target, model_config, &positions, &mut states);
    let initialization = NpaParticleInitialization { positions, states };
    initialization.validate(model_config)?;
    Ok(initialization)
}

pub fn mesh3d_damaged_initialization(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    particle_count: usize,
    seed: u64,
    damage_radius: f32,
    damage_displacement: f32,
) -> AutomataResult<NpaParticleInitialization> {
    if !damage_radius.is_finite()
        || damage_radius <= 0.0
        || !damage_displacement.is_finite()
        || damage_displacement < 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "mesh3d damage radius must be positive and displacement must be non-negative"
                .to_string(),
        ));
    }
    let mut initialization =
        mesh3d_surface_initialization(target, model_config, particle_count, seed)?;
    let center = target
        .surface_sample((seed as usize).wrapping_mul(0x9e37_79b9))
        .position;
    for (row, position) in initialization.positions.iter_mut().enumerate() {
        let delta = [
            position[0] - center[0],
            position[1] - center[1],
            position[2] - center[2],
        ];
        let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if distance > damage_radius {
            continue;
        }
        let state = &mut initialization.states
            [row * model_config.state_dims..(row + 1) * model_config.state_dims];
        let normal = [
            state[UV_TORUS_NORMAL_STATE_OFFSET],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 1],
            state[UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let unit = deterministic_damage_unit(row, seed);
        let displacement = damage_displacement * (0.35 + 0.65 * unit);
        for axis in 0..3 {
            position[axis] += normal[axis] * displacement;
        }
        if displacement > 0.0 {
            position[3] = 0.0;
        }
        state.fill(0.0);
        state[GROWTH_3D_RENDER_OPACITY_CHANNEL] = -4.0;
        state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = target
            .project([position[0], position[1], position[2]])
            .signed_distance;
    }
    initialization.validate(model_config)?;
    Ok(initialization)
}

fn deterministic_damage_unit(index: usize, seed: u64) -> f32 {
    let mut value = (index as u64)
        .wrapping_add(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x6a09_e667_f3bc_c909);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as u32) as f32 / u32::MAX as f32
}

pub fn mesh3d_supervised_batch(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
) -> AutomataResult<SupervisedBatch> {
    validate_dataset_config(model_config, hashgrid, config)?;
    let particle_count = config.dataset_particles;
    let trajectories = config.dataset_trajectories;
    let rows = particle_count * trajectories;
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x6d65_7368_3364);
    let mut positions = vec![[0.0_f32; 4]; rows];
    let mut states = vec![0.0_f32; rows * model_config.state_dims];
    let deployment_trajectories = ((trajectories as f32 * config.deployment_surface_fraction)
        .round() as usize)
        .min(trajectories);
    let damaged_deployment_trajectories =
        ((deployment_trajectories as f32 * config.deployment_damage_fraction).round() as usize)
            .min(deployment_trajectories.saturating_sub(1));
    let first_damaged_lane =
        deployment_trajectories.saturating_sub(damaged_deployment_trajectories);
    let mut trajectory_ages = vec![0usize; trajectories];

    for (lane, trajectory_age) in trajectory_ages
        .iter_mut()
        .enumerate()
        .take(deployment_trajectories)
    {
        let seed = config
            .seed
            .wrapping_add((lane as u64).wrapping_mul(0x9e37_79b9));
        let initialization = if lane >= first_damaged_lane {
            let damage_index = lane - first_damaged_lane;
            *trajectory_age = if damaged_deployment_trajectories <= 1 {
                0
            } else {
                damage_index * config.teacher_rollout_max_steps
                    / (damaged_deployment_trajectories - 1)
            };
            mesh3d_damaged_initialization(
                target,
                model_config,
                particle_count,
                seed,
                config.evaluation.damage_radius,
                config.evaluation.damage_displacement,
            )?
        } else {
            mesh3d_surface_initialization(target, model_config, particle_count, seed)?
        };
        let row_start = lane * particle_count;
        positions[row_start..row_start + particle_count].copy_from_slice(&initialization.positions);
        let state_start = row_start * model_config.state_dims;
        states[state_start..state_start + initialization.states.len()]
            .copy_from_slice(&initialization.states);
    }

    for row in deployment_trajectories * particle_count..rows {
        let lane = row / particle_count;
        let field_index = lane - deployment_trajectories;
        let field_trajectories = trajectories - deployment_trajectories;
        trajectory_ages[lane] = if field_trajectories <= 1 {
            config.teacher_rollout_max_steps
        } else {
            field_index * config.teacher_rollout_max_steps / (field_trajectories - 1)
        };
        let unit = rng.random::<f32>();
        let (position, state_target, anchored) = if unit < config.surface_fraction {
            let sample = target.surface_sample(row);
            (
                sample.position,
                Some((
                    sample.normal,
                    sample.color,
                    1.0_f32,
                    rng.random::<f32>() < config.surface_erasure_fraction,
                )),
                true,
            )
        } else if unit < config.surface_fraction + config.near_surface_fraction {
            let sample = target.surface_sample(row);
            let normal_offset = rng.random_range(-0.16..0.16) * config.scale;
            let tangent_jitter = [
                rng.random_range(-0.035..0.035) * config.scale,
                rng.random_range(-0.035..0.035) * config.scale,
                rng.random_range(-0.035..0.035) * config.scale,
            ];
            (
                [
                    sample.position[0] + sample.normal[0] * normal_offset + tangent_jitter[0],
                    sample.position[1] + sample.normal[1] * normal_offset + tangent_jitter[1],
                    sample.position[2] + sample.normal[2] * normal_offset + tangent_jitter[2],
                ],
                Some((
                    sample.normal,
                    sample.color,
                    rng.random_range(0.15_f32..0.85_f32),
                    false,
                )),
                false,
            )
        } else {
            let domain = config.scale * 1.18;
            (
                [
                    rng.random_range(-domain..domain),
                    rng.random_range(-domain..domain),
                    rng.random_range(-domain..domain),
                ],
                None,
                false,
            )
        };
        positions[row] = [
            position[0],
            position[1],
            position[2],
            if anchored { 1.0 } else { 0.0 },
        ];
        let state = &mut states[row * model_config.state_dims..(row + 1) * model_config.state_dims];
        if let Some((normal, color, retained_fraction, erased)) = state_target {
            if erased {
                state.fill(0.0);
                state[GROWTH_3D_RENDER_OPACITY_CHANNEL] = -4.0;
            } else {
                write_mesh_surface_state(state, normal, color);
                for value in state.iter_mut() {
                    *value *= retained_fraction;
                }
            }
        }
        state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = target.project(position).signed_distance;
    }

    apply_teacher_rollout_curriculum(
        target,
        &mut positions,
        &mut states,
        model_config,
        hashgrid,
        config,
        &trajectory_ages,
    );
    mesh3d_supervised_batch_from_particles(
        target,
        model_config,
        hashgrid,
        config,
        Mesh3dParticleBatch {
            positions,
            states,
            trajectories,
            particle_count,
        },
    )
}

pub(crate) struct Mesh3dParticleBatch {
    pub positions: Vec<[f32; 4]>,
    pub states: Vec<f32>,
    pub trajectories: usize,
    pub particle_count: usize,
}

pub(crate) fn mesh3d_supervised_batch_from_particles(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
    particles: Mesh3dParticleBatch,
) -> AutomataResult<SupervisedBatch> {
    let Mesh3dParticleBatch {
        positions,
        states,
        trajectories,
        particle_count,
    } = particles;
    let rows = trajectories.checked_mul(particle_count).ok_or_else(|| {
        AutomataError::InvalidArgument("mesh3d dataset row count overflow".to_string())
    })?;
    if positions.len() != rows || states.len() != rows * model_config.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "mesh3d particle batch has {} positions and {} state values; expected {rows} and {}",
            positions.len(),
            states.len(),
            rows * model_config.state_dims
        )));
    }
    let perception = perceive_with_options(
        &positions,
        &states,
        trajectories,
        particle_count,
        model_config.state_dims,
        hashgrid,
        PerceptionOptions {
            state_grad: model_config.state_grad,
            density_grad: model_config.density_grad,
            eps0: model_config.eps0,
            scale_equivariance: model_config.scale_equivariant(),
            particle_density_equivariance: model_config.particle_density_equivariant(),
            log_norm_grad: model_config.log_norm_grad,
            log_norm_density_grad: model_config.log_norm_density_grad,
            hybrid_state_gradient: true,
            position_features: model_config.position_features,
        },
    )?;
    let projections = positions
        .par_iter()
        .map(|position| target.project([position[0], position[1], position[2]]))
        .collect::<Vec<_>>();
    let output_dims = model_config.update_dims();
    let state_dims = model_config.state_dims;
    let mut target_update = vec![0.0_f32; rows * output_dims];
    target_update
        .par_chunks_mut(output_dims)
        .enumerate()
        .for_each(|(row, update)| {
            let state = &states[row * state_dims..(row + 1) * state_dims];
            mesh_teacher_update(
                projections[row],
                state,
                model_config,
                hashgrid,
                config,
                update,
            );
        });

    Ok(SupervisedBatch {
        features: perception.features,
        target_update,
    })
}

fn apply_teacher_rollout_curriculum(
    target: &TriangleMeshTarget,
    positions: &mut [[f32; 4]],
    states: &mut [f32],
    model_config: &NpaConfig,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
    trajectory_ages: &[usize],
) {
    let particle_count = config.dataset_particles;
    let state_dims = model_config.state_dims;
    let output_dims = model_config.update_dims();
    positions
        .par_chunks_mut(particle_count)
        .zip(states.par_chunks_mut(particle_count * state_dims))
        .enumerate()
        .for_each(|(trajectory, (positions, states))| {
            let age = trajectory_ages[trajectory];
            let mut update = vec![0.0_f32; output_dims];
            for _ in 0..age {
                for (row, position) in positions.iter_mut().enumerate() {
                    let state = &mut states[row * state_dims..(row + 1) * state_dims];
                    let projection = target.project([position[0], position[1], position[2]]);
                    let state_gate = position_conditioned_3d_maturity_gate(
                        state[GROWTH_3D_LIVENESS_CHANNEL],
                        state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
                    );
                    mesh_teacher_update(
                        projection,
                        state,
                        model_config,
                        hashgrid,
                        config,
                        &mut update,
                    );
                    let motion_gate = crate::rollout::position_conditioned_3d_motion_gate(
                        state[GROWTH_3D_LIVENESS_CHANNEL],
                        state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
                        update[model_config.spatial_dims + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
                        1.0,
                        position[3] >= 0.5,
                    );
                    let raw = &update[..3];
                    let raw_norm = raw.iter().map(|value| value * value).sum::<f32>().sqrt();
                    let motion_scale = model_config.alpha * model_config.motion_eps(hashgrid.eps)
                        / (1.0 + raw_norm);
                    for axis in 0..3 {
                        position[axis] += raw[axis] * motion_scale * motion_gate;
                    }
                    for channel in 0..state_dims {
                        state[channel] += update[3 + channel] * state_gate;
                    }
                    update.fill(0.0);
                }
            }
        });
}

fn mesh_teacher_update(
    projection: crate::TargetProjection,
    state: &[f32],
    model_config: &NpaConfig,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
    update: &mut [f32],
) {
    update.fill(0.0);
    let on_surface = projection.distance <= config.scale * 0.007;
    let residual = if on_surface {
        [0.0; 3]
    } else {
        projection.residual
    };
    let state_update = &mut update[model_config.spatial_dims..];
    for axis in 0..3 {
        state_update[UV_TORUS_NORMAL_STATE_OFFSET + axis] = config.normal_gain
            * (projection.normal[axis] - state[UV_TORUS_NORMAL_STATE_OFFSET + axis]);
    }
    let target_signed_distance = if on_surface {
        0.0
    } else {
        projection.signed_distance
    };
    state_update[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = config.signed_distance_gain
        * (target_signed_distance - state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET]);
    let target_liveness = if on_surface {
        TARGET_OPACITY_LOGIT
    } else {
        2.5
    };
    state_update[GROWTH_3D_LIVENESS_CHANNEL] =
        config.opacity_gain * (target_liveness - state[GROWTH_3D_LIVENESS_CHANNEL]);
    state_update[GROWTH_3D_RENDER_OPACITY_CHANNEL] =
        config.opacity_gain * (TARGET_OPACITY_LOGIT - state[GROWTH_3D_RENDER_OPACITY_CHANNEL]);
    let color_tail = model_config.state_dims - 3;
    for channel in 0..3 {
        let target_state = projection.color[channel] - 0.5;
        state_update[color_tail + channel] =
            config.color_gain * (target_state - state[color_tail + channel]);
    }

    let motion_gate = crate::rollout::position_conditioned_3d_motion_gate(
        state[GROWTH_3D_LIVENESS_CHANNEL],
        state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
        state_update[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
        1.0,
        false,
    );
    if motion_gate > 1.0e-4 {
        let desired_motion = bounded_mesh_motion(
            residual,
            config.max_motion_per_step,
            model_config.motion_eps(hashgrid.eps) * model_config.alpha * motion_gate,
        );
        update[..model_config.spatial_dims].copy_from_slice(&desired_motion);
    }
    for value in update {
        *value = value.clamp(-16.0, 16.0);
    }
}

pub(crate) fn write_mesh_surface_state(state: &mut [f32], normal: [f32; 3], color: [f32; 3]) {
    state.fill(0.0);
    state[GROWTH_3D_LIVENESS_CHANNEL] = TARGET_OPACITY_LOGIT;
    state[GROWTH_3D_RENDER_OPACITY_CHANNEL] = TARGET_OPACITY_LOGIT;
    state[UV_TORUS_NORMAL_STATE_OFFSET..UV_TORUS_NORMAL_STATE_OFFSET + 3].copy_from_slice(&normal);
    state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = 0.0;
    let color_tail = state.len() - 3;
    for channel in 0..3 {
        state[color_tail + channel] = color[channel] - 0.5;
    }
}

pub(crate) fn write_mesh_signed_distance_state(
    target: &TriangleMeshTarget,
    model_config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &mut [f32],
) {
    positions
        .par_iter()
        .zip(states.par_chunks_mut(model_config.state_dims))
        .for_each(|(position, state)| {
            state[UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] = target
                .project([position[0], position[1], position[2]])
                .signed_distance;
        });
}

fn bounded_mesh_motion(residual: [f32; 3], max_step: f32, motion_limit: f32) -> [f32; 3] {
    let distance =
        (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]).sqrt();
    if distance <= 1.0e-5 {
        return [0.0; 3];
    }
    let desired_step = distance.min(max_step).min(motion_limit * 0.9);
    let raw_norm = desired_step / (motion_limit - desired_step).max(1.0e-6);
    let scale = raw_norm / distance;
    [
        residual[0] * scale,
        residual[1] * scale,
        residual[2] * scale,
    ]
}

fn validate_dataset_config(
    model: &NpaConfig,
    hashgrid: &HashGridConfig,
    config: &Mesh3dTrainingConfig,
) -> AutomataResult<()> {
    if model.spatial_dims != 3
        || model.state_dims <= GROWTH_3D_RENDER_OPACITY_CHANNEL
        || model.state_dims < 13
        || !model.position_features
    {
        return Err(AutomataError::InvalidArgument(
            "mesh3d training requires a 3D position-conditioned model with at least 13 state channels"
                .to_string(),
        ));
    }
    if model.hidden_dims > 320 {
        return Err(AutomataError::InvalidArgument(format!(
            "mesh3d hidden_dims {} exceeds the resident WGPU inference limit 320",
            model.hidden_dims
        )));
    }
    if hashgrid.dim != 3 {
        return Err(AutomataError::InvalidArgument(
            "mesh3d training requires a 3D hashgrid".to_string(),
        ));
    }
    if config.dataset_particles == 0 || config.dataset_trajectories == 0 {
        return Err(AutomataError::InvalidArgument(
            "mesh3d training requires non-zero dataset particles and trajectories".to_string(),
        ));
    }
    if !config.near_surface_fraction.is_finite()
        || !config.surface_fraction.is_finite()
        || config.near_surface_fraction < 0.0
        || config.surface_fraction < 0.0
        || config.near_surface_fraction + config.surface_fraction > 1.0
        || !config.surface_erasure_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.surface_erasure_fraction)
        || !config.deployment_surface_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.deployment_surface_fraction)
        || !config.deployment_damage_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.deployment_damage_fraction)
    {
        return Err(AutomataError::InvalidArgument(
            "mesh3d surface sampling fractions must be finite, non-negative, and sum to at most one"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh3d::mesh3d_model_config;

    #[test]
    fn inverse_motion_parameterization_reconstructs_requested_step() {
        let raw = bounded_mesh_motion([0.3, 0.4, 0.0], 0.065, 0.1);
        let norm = raw.iter().map(|value| value * value).sum::<f32>().sqrt();
        let actual = 0.1 * norm / (1.0 + norm);
        assert!((actual - 0.065).abs() <= 1.0e-6);
    }

    #[test]
    fn inverse_motion_parameterization_respects_partial_motion_gate() {
        let gate = 0.25;
        let raw = bounded_mesh_motion([0.3, 0.4, 0.0], 0.065, 0.1 * gate);
        let norm = raw.iter().map(|value| value * value).sum::<f32>().sqrt();
        let actual = 0.1 * gate * norm / (1.0 + norm);
        assert!((actual - 0.0225).abs() <= 1.0e-6);
        assert!(raw.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn state_teacher_remains_a_bounded_residual_near_maturity() {
        let config = Mesh3dTrainingConfig::default();
        let model = mesh3d_model_config(64);
        let grid = HashGridConfig::growing_3dgs();
        let target = TriangleMeshTarget::utah_teapot(config.scale).unwrap();
        let sample = target.surface_sample(731);
        let projection = target.project(sample.position);
        let color_tail = model.state_dims - 3;
        let mut state = vec![0.0_f32; model.state_dims];
        state[GROWTH_3D_RENDER_OPACITY_CHANNEL] = -4.0;
        let mut growing = vec![0.0_f32; model.update_dims()];
        mesh_teacher_update(projection, &state, &model, &grid, &config, &mut growing);

        state[GROWTH_3D_LIVENESS_CHANNEL] = 3.75;
        let mut near_mature = vec![0.0_f32; model.update_dims()];
        mesh_teacher_update(projection, &state, &model, &grid, &config, &mut near_mature);

        let update_tail = model.spatial_dims + color_tail;
        assert_eq!(
            &growing[update_tail..update_tail + 3],
            &near_mature[update_tail..update_tail + 3],
            "the runtime maturity gate must not be inverted into a singular teacher target"
        );
        assert!(
            near_mature[model.spatial_dims..]
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 4.0)
        );
    }

    #[test]
    fn teapot_batch_matches_model_shapes_and_is_finite() {
        let config = Mesh3dTrainingConfig {
            dataset_particles: 32,
            dataset_trajectories: 2,
            teacher_rollout_max_steps: 0,
            deployment_surface_fraction: 0.5,
            ..Mesh3dTrainingConfig::default()
        };
        let model = mesh3d_model_config(64);
        let grid = HashGridConfig::growing_3dgs();
        let target = TriangleMeshTarget::utah_teapot(config.scale).unwrap();
        let batch = mesh3d_supervised_batch(&target, &model, &grid, &config).unwrap();
        assert_eq!(batch.features.len(), 64 * model.perception_dims(),);
        assert_eq!(batch.target_update.len(), 64 * model.update_dims());
        assert!(batch.features.iter().all(|value| value.is_finite()));
        assert!(batch.target_update.iter().all(|value| value.is_finite()));
        assert!(
            batch.target_update.iter().all(|value| value.abs() <= 16.0),
            "mesh3d teacher updates must remain bounded"
        );
        assert!(
            batch
                .features
                .chunks_exact(model.perception_dims())
                .any(|features| features[GROWTH_3D_RENDER_OPACITY_CHANNEL] <= -3.9),
            "mesh3d training batch must include exact erased surface states"
        );
        assert!(
            batch
                .target_update
                .chunks_exact(model.update_dims())
                .any(|row| row[..3].iter().any(|value| value.abs() > 1.0e-4))
        );
    }

    #[test]
    fn teapot_damage_is_localized_and_keeps_undamaged_surface_state() {
        let model = mesh3d_model_config(64);
        let target = TriangleMeshTarget::utah_teapot(0.72).unwrap();
        let pristine = mesh3d_surface_initialization(&target, &model, 2048, 42).unwrap();
        let damaged =
            mesh3d_damaged_initialization(&target, &model, 2048, 42, 0.22, 0.045).unwrap();
        let changed = pristine
            .positions
            .iter()
            .zip(&damaged.positions)
            .filter(|(before, after)| {
                before
                    .iter()
                    .zip(after.iter())
                    .take(3)
                    .any(|(before, after)| (before - after).abs() > 1.0e-6)
            })
            .count();
        assert!(changed > 16, "damage should affect a visible local patch");
        assert!(
            changed < pristine.positions.len() / 2,
            "damage should remain localized"
        );
        let unchanged = pristine
            .states
            .chunks_exact(model.state_dims)
            .zip(damaged.states.chunks_exact(model.state_dims))
            .filter(|(before, after)| before == after)
            .count();
        assert_eq!(unchanged + changed, pristine.positions.len());
    }

    #[test]
    fn teapot_erasure_damage_preserves_geometry_and_clears_a_local_state_patch() {
        let model = mesh3d_model_config(64);
        let target = TriangleMeshTarget::utah_teapot(0.72).unwrap();
        let pristine = mesh3d_surface_initialization(&target, &model, 2048, 42).unwrap();
        let damaged = mesh3d_damaged_initialization(&target, &model, 2048, 42, 0.22, 0.0).unwrap();
        assert_eq!(damaged.positions, pristine.positions);
        let erased = damaged
            .states
            .chunks_exact(model.state_dims)
            .filter(|state| {
                state[GROWTH_3D_RENDER_OPACITY_CHANNEL] == -4.0
                    && state.iter().enumerate().all(|(channel, value)| {
                        channel == GROWTH_3D_RENDER_OPACITY_CHANNEL || *value == 0.0
                    })
            })
            .count();
        assert!(erased > 16);
        assert!(erased < pristine.positions.len() / 2);
    }
}
