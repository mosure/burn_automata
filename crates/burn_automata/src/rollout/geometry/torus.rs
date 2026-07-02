use rand::Rng;

use crate::rollout::{
    MorphogenSeedEnvelope, UV_TORUS_DENSE_SEED_RADIUS_RATIO, UV_TORUS_MINOR_RATIO,
    UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET,
};

use super::helpers::random_sphere_position;
use super::teapot::morphogen_seed_envelope_position;

#[derive(Clone, Copy, Debug)]
pub struct UvTorusSample {
    pub position: [f32; 3],
    pub u: f32,
    pub v: f32,
    pub theta: f32,
    pub phi: f32,
}

pub fn uv_torus_sample(local_idx: usize, particle_count: usize, scale: f32) -> UvTorusSample {
    let particle_count = particle_count.max(1);
    let local_idx = local_idx % particle_count;
    let ring_count = (particle_count as f32).sqrt().round().max(1.0) as usize;
    let tube_count = particle_count.div_ceil(ring_count).max(1);
    let u = (local_idx % ring_count) as f32 / ring_count as f32;
    let v = (local_idx / ring_count) as f32 / tube_count as f32;
    let theta = std::f32::consts::TAU * u;
    let phi = std::f32::consts::TAU * v;
    UvTorusSample {
        position: uv_torus_parametric_position(theta, phi, scale),
        u,
        v,
        theta,
        phi,
    }
}

pub fn uv_torus_parametric_position(theta: f32, phi: f32, scale: f32) -> [f32; 3] {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let radial = major + minor * phi.cos();
    [
        radial * theta.cos(),
        radial * theta.sin(),
        minor * phi.sin(),
    ]
}

pub fn uv_torus_continuous_surface_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    uv_torus_parametric_position(
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        scale,
    )
}

pub fn uv_torus_continuous_volume_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let phi = rng.random_range(0.0..std::f32::consts::TAU);
    let tube_radius = minor * rng.random::<f32>().sqrt();
    let radial = major + tube_radius * phi.cos();
    [
        radial * theta.cos(),
        radial * theta.sin(),
        tube_radius * phi.sin(),
    ]
}

pub fn uv_torus_position_color(position: [f32; 3], scale: f32) -> [f32; 3] {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let outer = major + minor;
    [
        (position[0] / (2.0 * outer) + 0.5).clamp(0.0, 1.0),
        (position[1] / (2.0 * outer) + 0.5).clamp(0.0, 1.0),
        (position[2] / (2.0 * minor.max(1.0e-4)) + 0.5).clamp(0.0, 1.0),
    ]
}

pub fn uv_torus_project_position(position: [f32; 3], scale: f32) -> [f32; 3] {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
    let inv_radial = if radial > 1.0e-6 { 1.0 / radial } else { 0.0 };
    let dir_x = if radial > 1.0e-6 {
        position[0] * inv_radial
    } else {
        1.0
    };
    let dir_y = if radial > 1.0e-6 {
        position[1] * inv_radial
    } else {
        0.0
    };
    let tube_x = radial - major;
    let tube_z = position[2];
    let tube_len = (tube_x * tube_x + tube_z * tube_z).sqrt();
    let inv_tube = if tube_len > 1.0e-6 {
        1.0 / tube_len
    } else {
        1.0 / minor.max(1.0e-6)
    };
    let target_radial = major + minor * tube_x * inv_tube;
    [
        dir_x * target_radial,
        dir_y * target_radial,
        minor * tube_z * inv_tube,
    ]
}

pub fn uv_torus_outward_normal(position: [f32; 3], scale: f32) -> [f32; 3] {
    let major = scale.max(1.0e-4);
    let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
    let inv_radial = if radial > 1.0e-6 { 1.0 / radial } else { 0.0 };
    let dir_x = if radial > 1.0e-6 {
        position[0] * inv_radial
    } else {
        1.0
    };
    let dir_y = if radial > 1.0e-6 {
        position[1] * inv_radial
    } else {
        0.0
    };
    let tube_x = radial - major;
    let tube_z = position[2];
    let tube_len = (tube_x * tube_x + tube_z * tube_z).sqrt();
    if tube_len > 1.0e-6 {
        [
            dir_x * tube_x / tube_len,
            dir_y * tube_x / tube_len,
            tube_z / tube_len,
        ]
    } else {
        [dir_x, dir_y, 0.0]
    }
}

pub fn uv_torus_signed_distance(position: [f32; 3], scale: f32) -> f32 {
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
    ((radial - major).powi(2) + position[2].powi(2)).sqrt() - minor
}

pub fn uv_torus_surface_error(position: [f32; 3], scale: f32) -> f32 {
    uv_torus_signed_distance(position, scale).abs()
}

pub fn uv_torus_tail_state_color(position: [f32; 3], scale: f32) -> [f32; 3] {
    let rgb = uv_torus_position_color(position, scale);
    [rgb[0] - 0.5, rgb[1] - 0.5, rgb[2] - 0.5]
}

pub fn uv_torus_tail_state_to_rgb(tail: [f32; 3]) -> [f32; 3] {
    [
        (tail[0] + 0.5).clamp(0.0, 1.0),
        (tail[1] + 0.5).clamp(0.0, 1.0),
        (tail[2] + 0.5).clamp(0.0, 1.0),
    ]
}

pub fn uv_torus_outer_radius(scale: f32) -> f32 {
    scale.max(1.0e-4) * (1.0 + UV_TORUS_MINOR_RATIO)
}

pub fn uv_torus_dense_seed_radius(scale: f32) -> f32 {
    uv_torus_outer_radius(scale) * UV_TORUS_DENSE_SEED_RADIUS_RATIO
}

pub fn uv_torus_dense_seed_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    random_sphere_position(rng, uv_torus_dense_seed_radius(scale))
}

pub fn uv_torus_morphogen_seed_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let radius = uv_torus_outer_radius(scale) * 1.05;
    let envelope = MorphogenSeedEnvelope {
        core_radius: uv_torus_dense_seed_radius(scale),
        bounds_min: [-radius, -radius, -radius],
        bounds_max: [radius, radius, radius],
        near_surface_jitter: 0.18 * scale.max(1.0e-4),
    };
    morphogen_seed_envelope_position(
        rng,
        envelope,
        |rng| random_sphere_position(rng, envelope.core_radius),
        |rng| uv_torus_continuous_volume_position(rng, scale),
        |rng| uv_torus_continuous_surface_position(rng, scale),
        |surface| uv_torus_outward_normal(surface, scale),
    )
}

pub fn uv_torus_orientation_state_available(state_dims: usize) -> bool {
    let color_tail = state_dims.saturating_sub(3);
    state_dims > UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET
        && color_tail > UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET
}
