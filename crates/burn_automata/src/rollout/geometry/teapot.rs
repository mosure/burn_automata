use rand::Rng;

use crate::rollout::MorphogenSeedEnvelope;
use crate::target_geometry::TriangleMeshTarget;

use super::helpers::{
    distance2, dot3, normalize_or, random_box_position, random_ellipsoid_position,
    random_handle_arc_position, random_sphere_position, random_tapered_cylinder_position,
};

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
    projection_from_closest(closest, normal)
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
    projection_from_closest(closest, normal)
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
    projection_from_closest(closest, normal)
}

fn projection_from_closest(closest: [f32; 3], normal: [f32; 3]) -> TeapotProjection {
    let normal = normalize_or(normal, [1.0, 0.0, 0.0]);
    TeapotProjection { closest, normal }
}
