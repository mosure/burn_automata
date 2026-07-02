use rand::Rng;

use crate::rollout::{
    GROWTH_3D_ACTIVE_CORE_RADIUS_RATIO, GROWTH_3D_DOMAIN_RADIUS_RATIO,
    GROWTH_3D_MIN_ACTIVE_SEED_COUNT, GROWTH_3D_SEED_RADIUS_RATIO,
    GROWTH_3D_SUBSTRATE_MAX_RADIAL_GAP,
};

use super::helpers::random_sphere_position;

pub fn growth_3d_seed_radius(scale: f32) -> f32 {
    scale.max(1.0e-4) * GROWTH_3D_SEED_RADIUS_RATIO
}

pub fn growth_3d_domain_radius(scale: f32) -> f32 {
    scale.max(1.0e-4) * GROWTH_3D_DOMAIN_RADIUS_RATIO
}

pub fn growth_3d_active_core_radius(scale: f32) -> f32 {
    growth_3d_seed_radius(scale) * GROWTH_3D_ACTIVE_CORE_RADIUS_RATIO
}

pub fn growth_3d_active_seed_count(particle_count: usize) -> usize {
    if particle_count == 0 {
        return 0;
    }
    let active_fraction = GROWTH_3D_ACTIVE_CORE_RADIUS_RATIO.powi(3);
    let proportional_count =
        ((particle_count as f32 * active_fraction).round() as usize).clamp(1, particle_count);
    proportional_count.max(GROWTH_3D_MIN_ACTIVE_SEED_COUNT.min(particle_count))
}

pub fn growth_3d_substrate_min_level_count(scale: f32) -> usize {
    let radial_span = (growth_3d_domain_radius(scale) - growth_3d_active_core_radius(scale))
        .max(GROWTH_3D_SUBSTRATE_MAX_RADIAL_GAP);
    (radial_span / GROWTH_3D_SUBSTRATE_MAX_RADIAL_GAP)
        .ceil()
        .max(1.0) as usize
}

pub fn growth_3d_substrate_ray_count(
    inactive_count: usize,
    active_count: usize,
    scale: f32,
) -> usize {
    if inactive_count == 0 {
        return 1;
    }
    let min_level_count = growth_3d_substrate_min_level_count(scale);
    let max_connected_rays = (inactive_count / min_level_count).max(1);
    max_connected_rays.min(active_count.max(1))
}

pub fn growth_3d_seed_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    random_sphere_position(rng, growth_3d_seed_radius(scale))
}

pub fn growth_3d_stratified_seed_position(
    local_idx: usize,
    particle_count: usize,
    seed: u64,
    scale: f32,
) -> [f32; 3] {
    let particle_count = particle_count.max(1);
    let active_count = growth_3d_active_seed_count(particle_count);
    let seed_radius = growth_3d_seed_radius(scale);
    let active_radius = growth_3d_active_core_radius(scale);
    if local_idx < active_count {
        let t = (local_idx as f32 + 0.5) / active_count as f32;
        let radius = active_radius * t.cbrt();
        let direction =
            growth_3d_stratified_direction(local_idx, active_count, seed ^ 0x3d60_7a11_5eed_f00d);
        return [
            direction[0] * radius,
            direction[1] * radius,
            direction[2] * radius,
        ];
    }

    let inactive_count = particle_count.saturating_sub(active_count).max(1);
    let inactive_idx = local_idx
        .saturating_sub(active_count)
        .min(inactive_count - 1);
    let t = (inactive_idx as f32 + 0.5) / inactive_count as f32;
    let inner3 = active_radius.powi(3);
    let outer3 = seed_radius.powi(3);
    let radius = (inner3 + (outer3 - inner3).max(0.0) * t).cbrt();
    let direction =
        growth_3d_stratified_direction(inactive_idx, inactive_count, seed ^ 0x9e37_79b9_7f4a_7c15);
    [
        direction[0] * radius,
        direction[1] * radius,
        direction[2] * radius,
    ]
}

pub fn growth_3d_stratified_substrate_position(
    local_idx: usize,
    particle_count: usize,
    seed: u64,
    scale: f32,
) -> [f32; 3] {
    let particle_count = particle_count.max(1);
    let active_count = growth_3d_active_seed_count(particle_count);
    if local_idx < active_count {
        return growth_3d_stratified_seed_position(local_idx, particle_count, seed, scale);
    }

    let inactive_count = particle_count.saturating_sub(active_count).max(1);
    let inactive_idx = local_idx
        .saturating_sub(active_count)
        .min(inactive_count - 1);
    let ray_count =
        growth_3d_substrate_ray_count(inactive_count, active_count, scale).min(inactive_count);
    let level_count = inactive_count.div_ceil(ray_count).max(1);
    let ray_idx = inactive_idx % ray_count;
    let level_idx = inactive_idx / ray_count;
    let t = ((level_idx as f32 + 0.5) / level_count as f32).powf(1.5);
    let active_radius = growth_3d_active_core_radius(scale);
    let domain_radius = growth_3d_domain_radius(scale);
    let radius = active_radius + (domain_radius - active_radius).max(0.0) * t;
    let active_ray_idx = ray_idx % active_count.max(1);
    let direction = growth_3d_stratified_direction(
        active_ray_idx,
        active_count.max(1),
        seed ^ 0x3d60_7a11_5eed_f00d,
    );
    [
        direction[0] * radius,
        direction[1] * radius,
        direction[2] * radius,
    ]
}

fn growth_3d_stratified_direction(index: usize, count: usize, seed: u64) -> [f32; 3] {
    let count = count.max(1);
    let z = 1.0_f32 - 2.0 * (index as f32 + 0.5) / count as f32;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let phase = hash_unit_f32(seed) * std::f32::consts::TAU;
    let tilt_phase = hash_unit_f32(seed ^ 0xd1b5_4a32_d192_ed03) * std::f32::consts::TAU;
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let theta = index as f32 * golden_angle + phase;
    let mut direction = [radius * theta.cos(), radius * theta.sin(), z];
    let tilt = 0.35 * tilt_phase.sin();
    let cos_tilt = tilt.cos();
    let sin_tilt = tilt.sin();
    direction = [
        direction[0],
        direction[1] * cos_tilt - direction[2] * sin_tilt,
        direction[1] * sin_tilt + direction[2] * cos_tilt,
    ];
    direction
}

fn hash_unit_f32(value: u64) -> f32 {
    let mut x = value;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x >> 40) as f32) / ((1_u64 << 24) as f32)
}
