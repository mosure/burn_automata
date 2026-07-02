use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::target_geometry::TriangleMeshTarget;
use crate::{AutomataResult, NpaModel};
use burn_automata_kernels::HashGridConfig;

mod geometry;
#[cfg(test)]
mod tests;

pub use geometry::*;

pub const UV_TORUS_MINOR_RATIO: f32 = 0.72;
pub const UV_TORUS_INITIAL_SCALE: f32 = 0.45;
pub const UV_TORUS_DENSE_SEED_RADIUS_RATIO: f32 = 0.35;
pub const GROWTH_3D_SEED_RADIUS_RATIO: f32 = 0.20;
pub const GROWTH_3D_ACTIVE_CORE_RADIUS_RATIO: f32 = 0.30;
pub const GROWTH_3D_MIN_ACTIVE_SEED_COUNT: usize = 8;
pub const GROWTH_3D_DOMAIN_RADIUS_RATIO: f32 = 1.75;
pub const GROWTH_3D_SUBSTRATE_MAX_RADIAL_GAP: f32 = 0.075;
pub const GROWTH_3D_INACTIVE_OPACITY_LOGIT: f32 = -8.0;
pub const GROWTH_3D_SUBSTRATE_INACTIVE_OPACITY_LOGIT: f32 = -4.0;
pub const GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT: f32 = -4.0;
pub const GROWTH_3D_ACTIVE_OPACITY_LOGIT: f32 = 0.0;
pub const GROWTH_3D_LIVENESS_CHANNEL: usize = 3;
pub const GROWTH_3D_RENDER_OPACITY_CHANNEL: usize = 8;
pub const GROWTH_3D_PHASE_CHANNEL: usize = 9;
pub const GROWTH_3D_VELOCITY_STATE_OFFSET: usize = 10;
pub const UV_TORUS_MOTION_GAIN: f32 = 0.3;
pub const UV_TORUS_RESIDUAL_DECAY: f32 = 0.025;
pub const UV_TORUS_INITIAL_OPACITY_LOGIT: f32 = -2.8;
pub const UV_TORUS_OPACITY_GROWTH_DELTA: f32 = 0.08;
pub const UV_TORUS_NORMAL_STATE_OFFSET: usize = 4;
pub const UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET: usize = 7;

pub fn growth_3d_material_opacity_channel(state_dims: usize) -> Option<usize> {
    if state_dims > GROWTH_3D_RENDER_OPACITY_CHANNEL {
        Some(GROWTH_3D_RENDER_OPACITY_CHANNEL)
    } else if state_dims > GROWTH_3D_LIVENESS_CHANNEL {
        Some(GROWTH_3D_LIVENESS_CHANNEL)
    } else {
        None
    }
}

pub fn growth_3d_phase_channel(state_dims: usize) -> Option<usize> {
    (state_dims > GROWTH_3D_PHASE_CHANNEL).then_some(GROWTH_3D_PHASE_CHANNEL)
}

pub fn growth_3d_velocity_channels(state_dims: usize) -> Option<std::ops::Range<usize>> {
    (state_dims >= GROWTH_3D_VELOCITY_STATE_OFFSET + 3)
        .then_some(GROWTH_3D_VELOCITY_STATE_OFFSET..GROWTH_3D_VELOCITY_STATE_OFFSET + 3)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphogenSeedEnvelope {
    pub core_radius: f32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub near_surface_jitter: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub batch_size: usize,
    pub particle_count: usize,
    pub steps: usize,
    pub dt: f32,
    pub update_prob: f32,
    pub seed: u64,
    pub seed_scale: f32,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            particle_count: 4096,
            steps: 32,
            dt: 1.0,
            update_prob: 0.5,
            seed: 42,
            seed_scale: 0.2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleSeed {
    Gaussian,
    Uniform,
    #[default]
    UniformCircle,
    Growth3d,
    SubstrateGrowth3d,
    LocalGrowth3d,
    LocalSubstrateGrowth3d,
    UvTorus3d,
    UvTorusDense3d,
    TorusFieldDense3d,
    TeapotFieldDense3d,
    TorusGrowth3d,
    TeapotGrowth3d,
    TorusSubstrateGrowth3d,
    TeapotSubstrateGrowth3d,
    TorusLocalGrowth3d,
    TeapotLocalGrowth3d,
    TorusLocalSubstrateGrowth3d,
    TeapotLocalSubstrateGrowth3d,
    TorusMorphogenDense3d,
    TeapotMorphogenDense3d,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutTrace {
    pub positions: Vec<[f32; 4]>,
    pub states: Vec<f32>,
    pub batch_size: usize,
    pub particle_count: usize,
    pub state_dims: usize,
    pub steps: usize,
    pub mean_dx: Vec<f32>,
}

pub fn run_rollout(
    model: &NpaModel,
    grid: &HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
) -> AutomataResult<RolloutTrace> {
    let (mut positions, mut states) = seed_particles_scaled(
        cfg.batch_size,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        seed_mode,
        cfg.seed_scale,
    );
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5eed);
    let mut mean_dx = Vec::with_capacity(cfg.steps);

    for _ in 0..cfg.steps {
        let mask = stochastic_mask(
            cfg.batch_size * cfg.particle_count,
            cfg.update_prob,
            &mut rng,
        );
        let step = model.step_cpu(
            &positions,
            &states,
            cfg.batch_size,
            cfg.particle_count,
            grid,
            cfg.dt,
            Some(&mask),
        )?;
        let dx_norm = step
            .dx
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum::<f32>()
            / step.dx.len().max(1) as f32;
        mean_dx.push(dx_norm);
        positions = step.next_positions;
        states = step.next_states;
    }

    Ok(RolloutTrace {
        positions,
        states,
        batch_size: cfg.batch_size,
        particle_count: cfg.particle_count,
        state_dims: model.config.state_dims,
        steps: cfg.steps,
        mean_dx,
    })
}

pub fn seed_particles(
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    spatial_dims: usize,
    seed: u64,
    seed_mode: ParticleSeed,
) -> (Vec<[f32; 4]>, Vec<f32>) {
    seed_particles_scaled(
        batch_size,
        particle_count,
        state_dims,
        spatial_dims,
        seed,
        seed_mode,
        0.2,
    )
}

pub fn seed_particles_scaled(
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    spatial_dims: usize,
    seed: u64,
    seed_mode: ParticleSeed,
    scale: f32,
) -> (Vec<[f32; 4]>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut positions = vec![[0.0; 4]; batch_size * particle_count];
    let mut states = vec![0.0; batch_size * particle_count * state_dims];
    let teapot_target = if seed_mode == ParticleSeed::TeapotMorphogenDense3d && spatial_dims == 3 {
        Some(
            TriangleMeshTarget::utah_teapot(scale)
                .expect("canonical Utah Teapot target mesh should be valid"),
        )
    } else {
        None
    };
    for (idx, position) in positions.iter_mut().enumerate() {
        match seed_mode {
            ParticleSeed::Gaussian => {
                for v in position.iter_mut().take(spatial_dims) {
                    *v = rng.random_range(-1.0..1.0) * scale;
                }
            }
            ParticleSeed::Uniform => {
                for v in position.iter_mut().take(spatial_dims) {
                    *v = rng.random_range(-1.0..1.0) * scale;
                }
            }
            ParticleSeed::UniformCircle => {
                if spatial_dims == 2 {
                    let theta = rng.random_range(0.0..std::f32::consts::TAU);
                    let r = rng.random::<f32>().sqrt() * scale;
                    position[0] = r * theta.cos();
                    position[1] = r * theta.sin();
                } else {
                    let theta = rng.random_range(0.0..std::f32::consts::TAU);
                    let z = rng.random_range(-1.0_f32..1.0_f32);
                    let r_xy = (1.0_f32 - z * z).sqrt();
                    let r = rng.random::<f32>().cbrt() * scale;
                    position[0] = r * r_xy * theta.cos();
                    position[1] = r * r_xy * theta.sin();
                    position[2] = r * z;
                }
            }
            ParticleSeed::Growth3d
            | ParticleSeed::SubstrateGrowth3d
            | ParticleSeed::LocalGrowth3d
            | ParticleSeed::LocalSubstrateGrowth3d
            | ParticleSeed::TorusGrowth3d
            | ParticleSeed::TeapotGrowth3d
            | ParticleSeed::TorusSubstrateGrowth3d
            | ParticleSeed::TeapotSubstrateGrowth3d
            | ParticleSeed::TorusLocalGrowth3d
            | ParticleSeed::TeapotLocalGrowth3d
            | ParticleSeed::TorusLocalSubstrateGrowth3d
            | ParticleSeed::TeapotLocalSubstrateGrowth3d => {
                if spatial_dims == 3 {
                    let local_idx = idx % particle_count.max(1);
                    let seed_position = match seed_mode {
                        ParticleSeed::SubstrateGrowth3d
                        | ParticleSeed::LocalSubstrateGrowth3d
                        | ParticleSeed::TorusSubstrateGrowth3d
                        | ParticleSeed::TeapotSubstrateGrowth3d
                        | ParticleSeed::TorusLocalSubstrateGrowth3d
                        | ParticleSeed::TeapotLocalSubstrateGrowth3d => {
                            growth_3d_stratified_substrate_position(
                                local_idx,
                                particle_count,
                                seed,
                                scale,
                            )
                        }
                        _ => growth_3d_stratified_seed_position(
                            local_idx,
                            particle_count,
                            seed,
                            scale,
                        ),
                    };
                    position[0] = seed_position[0];
                    position[1] = seed_position[1];
                    position[2] = seed_position[2];
                    if state_dims >= 3 && growth_3d_seed_writes_coordinate_scaffold(seed_mode) {
                        let domain_radius = growth_3d_domain_radius(scale).max(1.0e-4);
                        let state_base = idx * state_dims;
                        states[state_base] = seed_position[0] / domain_radius;
                        states[state_base + 1] = seed_position[1] / domain_radius;
                        states[state_base + 2] = seed_position[2] / domain_radius;
                    }
                    if state_dims > GROWTH_3D_LIVENESS_CHANNEL {
                        let active = local_idx < growth_3d_active_seed_count(particle_count);
                        let liveness_logit = if active {
                            GROWTH_3D_ACTIVE_OPACITY_LOGIT
                        } else if matches!(
                            seed_mode,
                            ParticleSeed::SubstrateGrowth3d
                                | ParticleSeed::LocalSubstrateGrowth3d
                                | ParticleSeed::TorusSubstrateGrowth3d
                                | ParticleSeed::TeapotSubstrateGrowth3d
                                | ParticleSeed::TorusLocalSubstrateGrowth3d
                                | ParticleSeed::TeapotLocalSubstrateGrowth3d
                        ) {
                            GROWTH_3D_SUBSTRATE_INACTIVE_OPACITY_LOGIT
                        } else {
                            GROWTH_3D_INACTIVE_OPACITY_LOGIT
                        };
                        states[idx * state_dims + GROWTH_3D_LIVENESS_CHANNEL] = liveness_logit;
                        if state_dims > GROWTH_3D_RENDER_OPACITY_CHANNEL {
                            states[idx * state_dims + GROWTH_3D_RENDER_OPACITY_CHANNEL] = if active
                            {
                                GROWTH_3D_ACTIVE_OPACITY_LOGIT
                            } else {
                                GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
                            };
                        }
                    }
                }
            }
            ParticleSeed::TeapotMorphogenDense3d => {
                if spatial_dims == 3 {
                    let target_mesh = teapot_target
                        .as_ref()
                        .expect("teapot target mesh should be initialized for 3D teapot seed");
                    let seed_position =
                        utah_teapot_morphogen_seed_position(&mut rng, target_mesh, scale);
                    position[0] = seed_position[0];
                    position[1] = seed_position[1];
                    position[2] = seed_position[2];

                    let state_base = idx * state_dims;
                    let mut projected_color = [0.5; 3];
                    if state_dims >= 3 {
                        let source = [position[0], position[1], position[2]];
                        let projection = target_mesh.project(source);
                        projected_color = projection.color;
                        states[state_base] = projection.residual[0];
                        states[state_base + 1] = projection.residual[1];
                        states[state_base + 2] = projection.residual[2];
                        if uv_torus_orientation_state_available(state_dims) {
                            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET] =
                                projection.normal[0];
                            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] =
                                projection.normal[1];
                            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] =
                                projection.normal[2];
                            states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                                projection.signed_distance;
                        }
                    }
                    if state_dims > GROWTH_3D_LIVENESS_CHANNEL {
                        states[state_base + GROWTH_3D_LIVENESS_CHANNEL] =
                            UV_TORUS_INITIAL_OPACITY_LOGIT;
                        if state_dims > GROWTH_3D_RENDER_OPACITY_CHANNEL {
                            states[state_base + GROWTH_3D_RENDER_OPACITY_CHANNEL] =
                                UV_TORUS_INITIAL_OPACITY_LOGIT;
                        }
                    }
                    if state_dims >= 6 {
                        let tail_color = utah_teapot_tail_state_color_from_rgb(projected_color);
                        states[state_base + state_dims - 3] = tail_color[0];
                        states[state_base + state_dims - 2] = tail_color[1];
                        states[state_base + state_dims - 1] = tail_color[2];
                    }
                }
            }
            ParticleSeed::TeapotFieldDense3d => {
                if spatial_dims == 3 {
                    let dense_position = random_sphere_position(&mut rng, scale);
                    position[0] = dense_position[0];
                    position[1] = dense_position[1];
                    position[2] = dense_position[2];
                    if state_dims > GROWTH_3D_LIVENESS_CHANNEL {
                        states[idx * state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
                            UV_TORUS_INITIAL_OPACITY_LOGIT;
                        if state_dims > GROWTH_3D_RENDER_OPACITY_CHANNEL {
                            states[idx * state_dims + GROWTH_3D_RENDER_OPACITY_CHANNEL] =
                                UV_TORUS_INITIAL_OPACITY_LOGIT;
                        }
                    }
                }
            }
            ParticleSeed::UvTorus3d
            | ParticleSeed::UvTorusDense3d
            | ParticleSeed::TorusFieldDense3d
            | ParticleSeed::TorusMorphogenDense3d => {
                let local_idx = idx % particle_count.max(1);
                let sample = uv_torus_sample(local_idx, particle_count, scale);
                match seed_mode {
                    ParticleSeed::UvTorus3d => {
                        position[0] = sample.position[0] * UV_TORUS_INITIAL_SCALE;
                        position[1] = sample.position[1] * UV_TORUS_INITIAL_SCALE;
                        position[2] = sample.position[2] * UV_TORUS_INITIAL_SCALE;
                    }
                    ParticleSeed::UvTorusDense3d => {
                        let dense_position = uv_torus_dense_seed_position(&mut rng, scale);
                        position[0] = dense_position[0];
                        position[1] = dense_position[1];
                        position[2] = dense_position[2];
                    }
                    ParticleSeed::TorusFieldDense3d => {
                        let dense_position = uv_torus_dense_seed_position(&mut rng, scale);
                        position[0] = dense_position[0];
                        position[1] = dense_position[1];
                        position[2] = dense_position[2];
                    }
                    ParticleSeed::TorusMorphogenDense3d => {
                        let seed_position = uv_torus_morphogen_seed_position(&mut rng, scale);
                        position[0] = seed_position[0];
                        position[1] = seed_position[1];
                        position[2] = seed_position[2];
                    }
                    _ => unreachable!("uv torus match arm only handles torus seeds"),
                }

                let state_base = idx * state_dims;
                if spatial_dims == 3 && state_dims >= 3 {
                    match seed_mode {
                        ParticleSeed::TorusFieldDense3d => {}
                        ParticleSeed::TorusMorphogenDense3d => {
                            let source = [position[0], position[1], position[2]];
                            let target = uv_torus_project_position(source, scale);
                            states[state_base] = target[0] - position[0];
                            states[state_base + 1] = target[1] - position[1];
                            states[state_base + 2] = target[2] - position[2];
                            if uv_torus_orientation_state_available(state_dims) {
                                let normal = uv_torus_outward_normal(source, scale);
                                states[state_base + UV_TORUS_NORMAL_STATE_OFFSET] = normal[0];
                                states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = normal[1];
                                states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = normal[2];
                                states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                                    uv_torus_signed_distance(source, scale);
                            }
                        }
                        _ => {
                            states[state_base] = sample.position[0] - position[0];
                            states[state_base + 1] = sample.position[1] - position[1];
                            states[state_base + 2] = sample.position[2] - position[2];
                        }
                    }
                }
                if spatial_dims == 3 && state_dims > GROWTH_3D_LIVENESS_CHANNEL {
                    states[state_base + GROWTH_3D_LIVENESS_CHANNEL] =
                        UV_TORUS_INITIAL_OPACITY_LOGIT;
                    if state_dims > GROWTH_3D_RENDER_OPACITY_CHANNEL {
                        states[state_base + GROWTH_3D_RENDER_OPACITY_CHANNEL] =
                            UV_TORUS_INITIAL_OPACITY_LOGIT;
                    }
                }
                if spatial_dims == 3 && state_dims >= 6 {
                    match seed_mode {
                        ParticleSeed::TorusFieldDense3d => {}
                        ParticleSeed::TorusMorphogenDense3d => {
                            let target = [
                                position[0] + states[state_base],
                                position[1] + states[state_base + 1],
                                position[2] + states[state_base + 2],
                            ];
                            let tail_color = uv_torus_tail_state_color(target, scale);
                            states[state_base + state_dims - 3] = tail_color[0];
                            states[state_base + state_dims - 2] = tail_color[1];
                            states[state_base + state_dims - 1] = tail_color[2];
                        }
                        _ => {
                            let tail_color = uv_torus_tail_state_color(sample.position, scale);
                            states[state_base + state_dims - 3] = tail_color[0];
                            states[state_base + state_dims - 2] = tail_color[1];
                            states[state_base + state_dims - 1] = tail_color[2];
                        }
                    }
                }
            }
        }
    }
    (positions, states)
}

pub fn growth_3d_seed_writes_coordinate_scaffold(seed_mode: ParticleSeed) -> bool {
    matches!(
        seed_mode,
        ParticleSeed::Growth3d
            | ParticleSeed::SubstrateGrowth3d
            | ParticleSeed::TorusGrowth3d
            | ParticleSeed::TeapotGrowth3d
            | ParticleSeed::TorusSubstrateGrowth3d
            | ParticleSeed::TeapotSubstrateGrowth3d
    )
}

fn stochastic_mask(count: usize, update_prob: f32, rng: &mut StdRng) -> Vec<f32> {
    if update_prob >= 1.0 {
        return vec![1.0; count];
    }
    (0..count)
        .map(|_| f32::from(rng.random::<f32>() < update_prob))
        .collect()
}
