use super::*;

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

pub fn teapot_like_dense_seed_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    [
        rng.random_range(-0.98..1.12) * scale,
        rng.random_range(-0.50..0.50) * scale,
        rng.random_range(-0.38..0.68) * scale,
    ]
}

pub fn teapot_like_morphogen_seed_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    let envelope = MorphogenSeedEnvelope {
        core_radius: 0.30 * scale,
        bounds_min: [-0.98 * scale, -0.50 * scale, -0.38 * scale],
        bounds_max: [1.12 * scale, 0.50 * scale, 0.68 * scale],
        near_surface_jitter: 0.14 * scale,
    };
    morphogen_seed_envelope_position(
        rng,
        envelope,
        |rng| random_sphere_position(rng, envelope.core_radius),
        |rng| teapot_like_volume_position(rng, scale),
        |rng| teapot_like_surface_position(rng, scale),
        |surface| teapot_like_project(surface, scale).normal,
    )
}

pub fn teapot_like_project_position(position: [f32; 3], scale: f32) -> [f32; 3] {
    teapot_like_project(position, scale).closest
}

pub fn teapot_like_position_color(position: [f32; 3], scale: f32) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    [
        (position[0] / (2.2 * scale) + 0.5).clamp(0.0, 1.0),
        (position[1] / (1.1 * scale) + 0.5).clamp(0.0, 1.0),
        (position[2] / (1.3 * scale) + 0.44).clamp(0.0, 1.0),
    ]
}

pub fn teapot_like_tail_state_color(position: [f32; 3], scale: f32) -> [f32; 3] {
    let rgb = teapot_like_position_color(position, scale);
    [rgb[0] - 0.5, rgb[1] - 0.5, rgb[2] - 0.5]
}

pub fn utah_teapot_dense_seed_position<R: Rng + ?Sized>(
    rng: &mut R,
    target: &TriangleMeshTarget,
) -> [f32; 3] {
    let (bounds_min, bounds_max) = target.bounds();
    random_box_position(rng, bounds_min, bounds_max)
}

pub fn utah_teapot_morphogen_seed_position<R: Rng + ?Sized>(
    rng: &mut R,
    target: &TriangleMeshTarget,
    scale: f32,
) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    let (bounds_min, bounds_max) = target.bounds();
    let envelope = MorphogenSeedEnvelope {
        core_radius: 0.30 * scale,
        bounds_min,
        bounds_max,
        near_surface_jitter: 0.14 * scale,
    };
    morphogen_seed_envelope_position(
        rng,
        envelope,
        |rng| random_sphere_position(rng, envelope.core_radius),
        |rng| utah_teapot_dense_seed_position(rng, target),
        |rng| random_mesh_surface_position(rng, target),
        |surface| target.project(surface).normal,
    )
}

pub fn utah_teapot_tail_state_color_from_rgb(rgb: [f32; 3]) -> [f32; 3] {
    [rgb[0] - 0.5, rgb[1] - 0.5, rgb[2] - 0.5]
}

pub fn utah_teapot_tail_state_color(position: [f32; 3], target: &TriangleMeshTarget) -> [f32; 3] {
    utah_teapot_tail_state_color_from_rgb(target.project(position).color)
}

pub fn utah_teapot_project_position(position: [f32; 3], target: &TriangleMeshTarget) -> [f32; 3] {
    target.project(position).closest
}

pub fn morphogen_seed_envelope_position<R, Core, Volume, Surface, Normal>(
    rng: &mut R,
    envelope: MorphogenSeedEnvelope,
    mut core: Core,
    mut volume: Volume,
    mut surface: Surface,
    mut normal: Normal,
) -> [f32; 3]
where
    R: Rng + ?Sized,
    Core: FnMut(&mut R) -> [f32; 3],
    Volume: FnMut(&mut R) -> [f32; 3],
    Surface: FnMut(&mut R) -> [f32; 3],
    Normal: FnMut([f32; 3]) -> [f32; 3],
{
    match rng.random_range(0..4) {
        0 => core(rng),
        1 => volume(rng),
        2 => {
            let surface_position = surface(rng);
            let normal = normal(surface_position);
            let offset =
                rng.random_range(-envelope.near_surface_jitter..envelope.near_surface_jitter);
            [
                surface_position[0] + normal[0] * offset,
                surface_position[1] + normal[1] * offset,
                surface_position[2] + normal[2] * offset,
            ]
        }
        _ => random_box_position(rng, envelope.bounds_min, envelope.bounds_max),
    }
}

fn random_mesh_surface_position<R: Rng + ?Sized>(
    rng: &mut R,
    target: &TriangleMeshTarget,
) -> [f32; 3] {
    target.random_surface_sample(rng).position
}

fn teapot_like_volume_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    match rng.random_range(0..5) {
        0 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.0],
            [0.56 * scale, 0.38 * scale, 0.34 * scale],
            false,
        ),
        1 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.36 * scale],
            [0.34 * scale, 0.26 * scale, 0.10 * scale],
            false,
        ),
        2 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.55 * scale],
            [0.12 * scale, 0.12 * scale, 0.09 * scale],
            false,
        ),
        3 => random_tapered_cylinder_position(
            rng,
            [0.42 * scale, 0.0, 0.08 * scale],
            [1.05 * scale, 0.0, 0.28 * scale],
            0.13 * scale,
            0.055 * scale,
            false,
        ),
        _ => random_handle_arc_position(
            rng,
            [-0.58 * scale, 0.0, 0.02 * scale],
            0.36 * scale,
            0.065 * scale,
            std::f32::consts::PI - 1.18,
            std::f32::consts::PI + 1.18,
            false,
        ),
    }
}

fn teapot_like_surface_position(rng: &mut impl Rng, scale: f32) -> [f32; 3] {
    let scale = scale.max(1.0e-4);
    match rng.random_range(0..5) {
        0 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.0],
            [0.56 * scale, 0.38 * scale, 0.34 * scale],
            true,
        ),
        1 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.36 * scale],
            [0.34 * scale, 0.26 * scale, 0.10 * scale],
            true,
        ),
        2 => random_ellipsoid_position(
            rng,
            [0.0, 0.0, 0.55 * scale],
            [0.12 * scale, 0.12 * scale, 0.09 * scale],
            true,
        ),
        3 => random_tapered_cylinder_position(
            rng,
            [0.42 * scale, 0.0, 0.08 * scale],
            [1.05 * scale, 0.0, 0.28 * scale],
            0.13 * scale,
            0.055 * scale,
            true,
        ),
        _ => random_handle_arc_position(
            rng,
            [-0.58 * scale, 0.0, 0.02 * scale],
            0.36 * scale,
            0.065 * scale,
            std::f32::consts::PI - 1.18,
            std::f32::consts::PI + 1.18,
            true,
        ),
    }
}

#[derive(Clone, Copy, Debug)]
struct TeapotProjection {
    closest: [f32; 3],
    normal: [f32; 3],
}

fn teapot_like_project(position: [f32; 3], scale: f32) -> TeapotProjection {
    let scale = scale.max(1.0e-4);
    let candidates = [
        ellipsoid_projection(
            position,
            [0.0, 0.0, 0.0],
            [0.56 * scale, 0.38 * scale, 0.34 * scale],
        ),
        ellipsoid_projection(
            position,
            [0.0, 0.0, 0.36 * scale],
            [0.34 * scale, 0.26 * scale, 0.10 * scale],
        ),
        ellipsoid_projection(
            position,
            [0.0, 0.0, 0.55 * scale],
            [0.12 * scale, 0.12 * scale, 0.09 * scale],
        ),
        tapered_cylinder_projection(
            position,
            [0.42 * scale, 0.0, 0.08 * scale],
            [1.05 * scale, 0.0, 0.28 * scale],
            0.13 * scale,
            0.055 * scale,
        ),
        handle_arc_projection(
            position,
            [-0.58 * scale, 0.0, 0.02 * scale],
            0.36 * scale,
            0.065 * scale,
            std::f32::consts::PI - 1.18,
            std::f32::consts::PI + 1.18,
        ),
    ];
    let mut best = candidates[0];
    let mut best_distance2 = distance2(best.closest, position);
    for candidate in candidates.iter().copied().skip(1) {
        let distance2 = distance2(candidate.closest, position);
        if distance2 < best_distance2 {
            best = candidate;
            best_distance2 = distance2;
        }
    }
    TeapotProjection {
        closest: best.closest,
        normal: best.normal,
    }
}

fn ellipsoid_projection(position: [f32; 3], center: [f32; 3], radii: [f32; 3]) -> TeapotProjection {
    let local = [
        (position[0] - center[0]) / radii[0].max(1.0e-6),
        (position[1] - center[1]) / radii[1].max(1.0e-6),
        (position[2] - center[2]) / radii[2].max(1.0e-6),
    ];
    let sphere_dir = normalize_or(local, [1.0, 0.0, 0.0]);
    let closest = [
        center[0] + radii[0] * sphere_dir[0],
        center[1] + radii[1] * sphere_dir[1],
        center[2] + radii[2] * sphere_dir[2],
    ];
    let normal = normalize_or(
        [
            sphere_dir[0] / radii[0].max(1.0e-6),
            sphere_dir[1] / radii[1].max(1.0e-6),
            sphere_dir[2] / radii[2].max(1.0e-6),
        ],
        [1.0, 0.0, 0.0],
    );
    projection_from_closest(position, closest, normal)
}

fn tapered_cylinder_projection(
    position: [f32; 3],
    start: [f32; 3],
    end: [f32; 3],
    start_radius: f32,
    end_radius: f32,
) -> TeapotProjection {
    let axis = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let axis_len2 = dot3(axis, axis).max(1.0e-8);
    let t = dot3(
        [
            position[0] - start[0],
            position[1] - start[1],
            position[2] - start[2],
        ],
        axis,
    ) / axis_len2;
    let t = t.clamp(0.0, 1.0);
    let center = [
        start[0] + axis[0] * t,
        start[1] + axis[1] * t,
        start[2] + axis[2] * t,
    ];
    let axis_dir = normalize_or(axis, [1.0, 0.0, 0.0]);
    let radial = [
        position[0] - center[0],
        position[1] - center[1],
        position[2] - center[2],
    ];
    let axial = dot3(radial, axis_dir);
    let radial = [
        radial[0] - axis_dir[0] * axial,
        radial[1] - axis_dir[1] * axial,
        radial[2] - axis_dir[2] * axial,
    ];
    let normal = normalize_or(radial, [0.0, 1.0, 0.0]);
    let radius = start_radius + (end_radius - start_radius) * t;
    let closest = [
        center[0] + normal[0] * radius,
        center[1] + normal[1] * radius,
        center[2] + normal[2] * radius,
    ];
    projection_from_closest(position, closest, normal)
}

fn handle_arc_projection(
    position: [f32; 3],
    center: [f32; 3],
    major: f32,
    tube: f32,
    start_angle: f32,
    end_angle: f32,
) -> TeapotProjection {
    let angle = (position[2] - center[2]).atan2(position[0] - center[0]);
    let angle = angle.clamp(start_angle, end_angle);
    let radial = [angle.cos(), 0.0, angle.sin()];
    let centerline = [
        center[0] + radial[0] * major,
        center[1],
        center[2] + radial[2] * major,
    ];
    let tube_vec = [
        position[0] - centerline[0],
        position[1] - centerline[1],
        position[2] - centerline[2],
    ];
    let normal = normalize_or(tube_vec, radial);
    let closest = [
        centerline[0] + normal[0] * tube,
        centerline[1] + normal[1] * tube,
        centerline[2] + normal[2] * tube,
    ];
    projection_from_closest(position, closest, normal)
}

fn projection_from_closest(
    _position: [f32; 3],
    closest: [f32; 3],
    normal: [f32; 3],
) -> TeapotProjection {
    let normal = normalize_or(normal, [1.0, 0.0, 0.0]);
    TeapotProjection { closest, normal }
}

fn random_box_position<R: Rng + ?Sized>(
    rng: &mut R,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> [f32; 3] {
    [
        rng.random_range(bounds_min[0]..bounds_max[0]),
        rng.random_range(bounds_min[1]..bounds_max[1]),
        rng.random_range(bounds_min[2]..bounds_max[2]),
    ]
}

pub(super) fn random_sphere_position<R: Rng + ?Sized>(rng: &mut R, radius: f32) -> [f32; 3] {
    let direction = random_unit_vector(rng);
    let radius = rng.random::<f32>().cbrt() * radius.max(0.0);
    [
        direction[0] * radius,
        direction[1] * radius,
        direction[2] * radius,
    ]
}

fn random_unit_vector<R: Rng + ?Sized>(rng: &mut R) -> [f32; 3] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let z = rng.random_range(-1.0_f32..1.0_f32);
    let r_xy = (1.0_f32 - z * z).sqrt();
    [r_xy * theta.cos(), r_xy * theta.sin(), z]
}

fn random_ellipsoid_position<R: Rng + ?Sized>(
    rng: &mut R,
    center: [f32; 3],
    radii: [f32; 3],
    surface: bool,
) -> [f32; 3] {
    let mut direction = random_unit_vector(rng);
    if !surface {
        let radius = rng.random::<f32>().cbrt();
        direction = [
            direction[0] * radius,
            direction[1] * radius,
            direction[2] * radius,
        ];
    }
    [
        center[0] + direction[0] * radii[0],
        center[1] + direction[1] * radii[1],
        center[2] + direction[2] * radii[2],
    ]
}

fn random_tapered_cylinder_position<R: Rng + ?Sized>(
    rng: &mut R,
    start: [f32; 3],
    end: [f32; 3],
    start_radius: f32,
    end_radius: f32,
    surface: bool,
) -> [f32; 3] {
    let axis = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let axis_dir = normalize_or(axis, [1.0, 0.0, 0.0]);
    let (tangent, bitangent) = orthonormal_basis(axis_dir);
    let t = rng.random::<f32>();
    let center = [
        start[0] + axis[0] * t,
        start[1] + axis[1] * t,
        start[2] + axis[2] * t,
    ];
    let radius = start_radius + (end_radius - start_radius) * t;
    let radial = if surface {
        radius
    } else {
        radius * rng.random::<f32>().sqrt()
    };
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    [
        center[0] + tangent[0] * radial * theta.cos() + bitangent[0] * radial * theta.sin(),
        center[1] + tangent[1] * radial * theta.cos() + bitangent[1] * radial * theta.sin(),
        center[2] + tangent[2] * radial * theta.cos() + bitangent[2] * radial * theta.sin(),
    ]
}

#[allow(clippy::too_many_arguments)]
fn random_handle_arc_position<R: Rng + ?Sized>(
    rng: &mut R,
    center: [f32; 3],
    major: f32,
    tube: f32,
    start_angle: f32,
    end_angle: f32,
    surface: bool,
) -> [f32; 3] {
    let angle = rng.random_range(start_angle..end_angle);
    let radial = [angle.cos(), 0.0, angle.sin()];
    let centerline = [
        center[0] + radial[0] * major,
        center[1],
        center[2] + radial[2] * major,
    ];
    let tube_radius = if surface {
        tube
    } else {
        tube * rng.random::<f32>().sqrt()
    };
    let phi = rng.random_range(0.0..std::f32::consts::TAU);
    [
        centerline[0] + radial[0] * tube_radius * phi.cos(),
        centerline[1] + tube_radius * phi.sin(),
        centerline[2] + radial[2] * tube_radius * phi.cos(),
    ]
}

fn orthonormal_basis(axis: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let mut tangent = cross3(axis, [0.0, 0.0, 1.0]);
    if dot3(tangent, tangent) <= 1.0e-8 {
        tangent = cross3(axis, [0.0, 1.0, 0.0]);
    }
    let tangent = normalize_or(tangent, [0.0, 1.0, 0.0]);
    let bitangent = normalize_or(cross3(axis, tangent), [0.0, 0.0, 1.0]);
    (tangent, bitangent)
}

fn distance2(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    (lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)
}

fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 {
        fallback
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}
