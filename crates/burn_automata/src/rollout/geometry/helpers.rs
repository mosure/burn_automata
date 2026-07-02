use rand::Rng;

pub(super) fn random_box_position<R: Rng + ?Sized>(
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

pub(crate) fn random_sphere_position<R: Rng + ?Sized>(rng: &mut R, radius: f32) -> [f32; 3] {
    let direction = random_unit_vector(rng);
    let radius = rng.random::<f32>().cbrt() * radius.max(0.0);
    [
        direction[0] * radius,
        direction[1] * radius,
        direction[2] * radius,
    ]
}

pub(super) fn random_unit_vector<R: Rng + ?Sized>(rng: &mut R) -> [f32; 3] {
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let z = rng.random_range(-1.0_f32..1.0_f32);
    let r_xy = (1.0_f32 - z * z).sqrt();
    [r_xy * theta.cos(), r_xy * theta.sin(), z]
}

pub(super) fn random_ellipsoid_position<R: Rng + ?Sized>(
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

pub(super) fn random_tapered_cylinder_position<R: Rng + ?Sized>(
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
pub(super) fn random_handle_arc_position<R: Rng + ?Sized>(
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

pub(super) fn distance2(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    (lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)
}

pub(super) fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

pub(super) fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

pub(super) fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 {
        fallback
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}
