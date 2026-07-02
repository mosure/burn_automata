use super::{
    EPS, TriangleMeshTarget, add3, cross3, length3, normalize_or, normalize3, scale3, sub3,
};
use crate::AutomataResult;

impl TriangleMeshTarget {
    pub fn sphere(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(8);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.0],
            [scale, scale, scale],
            segments,
            segments * 2,
        );
        colored_mesh(vertices, faces)
    }

    pub fn ellipsoid(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(8);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.0],
            [0.72 * scale, 0.48 * scale, scale],
            segments,
            segments * 2,
        );
        colored_mesh(vertices, faces)
    }

    pub fn cube(scale: f32) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let vertices = vec![
            [-scale, -scale, -scale],
            [scale, -scale, -scale],
            [scale, scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
            [scale, -scale, scale],
            [scale, scale, scale],
            [-scale, scale, scale],
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        colored_mesh(vertices, faces)
    }

    pub fn cylinder(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, -scale],
            [0.0, 0.0, scale],
            0.62 * scale,
            0.62 * scale,
            segments.max(12),
        );
        colored_mesh(vertices, faces)
    }

    pub fn cone(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, -scale],
            [0.0, 0.0, scale],
            0.75 * scale,
            0.04 * scale,
            segments.max(12),
        );
        colored_mesh(vertices, faces)
    }

    pub fn capsule(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(8);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, -0.48 * scale],
            [0.48 * scale, 0.48 * scale, 0.48 * scale],
            segments,
            segments * 2,
        );
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.48 * scale],
            [0.48 * scale, 0.48 * scale, 0.48 * scale],
            segments,
            segments * 2,
        );
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, -0.48 * scale],
            [0.0, 0.0, 0.48 * scale],
            0.48 * scale,
            0.48 * scale,
            segments * 2,
        );
        colored_mesh(vertices, faces)
    }

    pub fn pyramid(scale: f32) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let vertices = vec![
            [-0.72 * scale, -0.72 * scale, -0.60 * scale],
            [0.72 * scale, -0.72 * scale, -0.60 * scale],
            [0.72 * scale, 0.72 * scale, -0.60 * scale],
            [-0.72 * scale, 0.72 * scale, -0.60 * scale],
            [0.0, 0.0, 0.88 * scale],
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ];
        colored_mesh(vertices, faces)
    }

    pub fn bicone(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(12);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, -scale],
            [0.0, 0.0, 0.0],
            0.04 * scale,
            0.72 * scale,
            segments,
        );
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, scale],
            0.72 * scale,
            0.04 * scale,
            segments,
        );
        colored_mesh(vertices, faces)
    }

    pub fn dumbbell(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(8);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [-0.54 * scale, 0.0, 0.0],
            [0.38 * scale, 0.38 * scale, 0.38 * scale],
            segments,
            segments * 2,
        );
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.54 * scale, 0.0, 0.0],
            [0.38 * scale, 0.38 * scale, 0.38 * scale],
            segments,
            segments * 2,
        );
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [-0.54 * scale, 0.0, 0.0],
            [0.54 * scale, 0.0, 0.0],
            0.18 * scale,
            0.18 * scale,
            segments * 2,
        );
        colored_mesh(vertices, faces)
    }

    pub fn cross(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(12);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for (start, end) in [
            ([-scale, 0.0, 0.0], [scale, 0.0, 0.0]),
            ([0.0, -scale, 0.0], [0.0, scale, 0.0]),
            ([0.0, 0.0, -scale], [0.0, 0.0, scale]),
        ] {
            append_tapered_cylinder(
                &mut vertices,
                &mut faces,
                start,
                end,
                0.24 * scale,
                0.24 * scale,
                segments,
            );
        }
        colored_mesh(vertices, faces)
    }

    pub fn torus(major: f32, minor: f32, rings: usize, tubes: usize) -> AutomataResult<Self> {
        let major = major.max(1.0e-4);
        let minor = minor.max(1.0e-4);
        let rings = rings.max(3);
        let tubes = tubes.max(3);
        let mut vertices = Vec::with_capacity(rings * tubes);
        let mut normals = Vec::with_capacity(rings * tubes);
        let mut colors = Vec::with_capacity(rings * tubes);
        let outer = major + minor;

        for ring in 0..rings {
            let theta = std::f32::consts::TAU * ring as f32 / rings as f32;
            let theta_cos = theta.cos();
            let theta_sin = theta.sin();
            for tube in 0..tubes {
                let phi = std::f32::consts::TAU * tube as f32 / tubes as f32;
                let phi_cos = phi.cos();
                let phi_sin = phi.sin();
                let radial = major + minor * phi_cos;
                let position = [radial * theta_cos, radial * theta_sin, minor * phi_sin];
                let normal = normalize3([theta_cos * phi_cos, theta_sin * phi_cos, phi_sin]);
                vertices.push(position);
                normals.push(normal);
                colors.push([
                    (position[0] / (2.0 * outer) + 0.5).clamp(0.0, 1.0),
                    (position[1] / (2.0 * outer) + 0.5).clamp(0.0, 1.0),
                    (position[2] / (2.0 * minor) + 0.5).clamp(0.0, 1.0),
                ]);
            }
        }

        let mut faces = Vec::with_capacity(rings * tubes * 2);
        for ring in 0..rings {
            let next_ring = (ring + 1) % rings;
            for tube in 0..tubes {
                let next_tube = (tube + 1) % tubes;
                let a = (ring * tubes + tube) as u32;
                let b = (next_ring * tubes + tube) as u32;
                let c = (next_ring * tubes + next_tube) as u32;
                let d = (ring * tubes + next_tube) as u32;
                faces.push([a, b, c]);
                faces.push([a, c, d]);
            }
        }

        let mut target = Self::new(vertices, faces)?;
        target.vertex_normals = normals;
        target.colors = Some(colors);
        Ok(target)
    }

    pub fn teapot_like(scale: f32, segments: usize) -> AutomataResult<Self> {
        let scale = scale.max(1.0e-4);
        let segments = segments.max(8);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.0],
            [0.56 * scale, 0.38 * scale, 0.34 * scale],
            segments,
            segments * 2,
        );
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.36 * scale],
            [0.34 * scale, 0.26 * scale, 0.10 * scale],
            segments / 2,
            segments * 2,
        );
        append_uv_sphere(
            &mut vertices,
            &mut faces,
            [0.0, 0.0, 0.55 * scale],
            [0.12 * scale, 0.12 * scale, 0.09 * scale],
            segments / 2,
            segments,
        );
        append_tapered_cylinder(
            &mut vertices,
            &mut faces,
            [0.42 * scale, 0.0, 0.08 * scale],
            [1.05 * scale, 0.0, 0.28 * scale],
            0.13 * scale,
            0.055 * scale,
            segments * 2,
        );
        append_torus_arc(
            &mut vertices,
            &mut faces,
            [-0.58 * scale, 0.0, 0.02 * scale],
            0.36 * scale,
            0.065 * scale,
            std::f32::consts::PI - 1.18,
            std::f32::consts::PI + 1.18,
            segments * 2,
            segments,
        );

        let colors = vertices
            .iter()
            .map(|position| teapot_like_color(*position, scale))
            .collect::<Vec<_>>();
        let mut target = Self::new(vertices, faces)?;
        target.colors = Some(colors);
        Ok(target)
    }
}

fn append_uv_sphere(
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    center: [f32; 3],
    radii: [f32; 3],
    lat_segments: usize,
    lon_segments: usize,
) {
    let lat_segments = lat_segments.max(4);
    let lon_segments = lon_segments.max(6);
    let base = vertices.len() as u32;
    for lat in 0..=lat_segments {
        let phi = std::f32::consts::PI * lat as f32 / lat_segments as f32;
        let z = phi.cos();
        let ring = phi.sin();
        for lon in 0..lon_segments {
            let theta = std::f32::consts::TAU * lon as f32 / lon_segments as f32;
            vertices.push([
                center[0] + radii[0] * ring * theta.cos(),
                center[1] + radii[1] * ring * theta.sin(),
                center[2] + radii[2] * z,
            ]);
        }
    }

    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let next_lon = (lon + 1) % lon_segments;
            let a = base + (lat * lon_segments + lon) as u32;
            let b = base + ((lat + 1) * lon_segments + lon) as u32;
            let c = base + ((lat + 1) * lon_segments + next_lon) as u32;
            let d = base + (lat * lon_segments + next_lon) as u32;
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
}

fn append_tapered_cylinder(
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    start: [f32; 3],
    end: [f32; 3],
    start_radius: f32,
    end_radius: f32,
    segments: usize,
) {
    let segments = segments.max(6);
    let base = vertices.len() as u32;
    let axis = normalize_or(sub3(end, start), [1.0, 0.0, 0.0]);
    let mut tangent = cross3(axis, [0.0, 0.0, 1.0]);
    if length3(tangent) <= EPS {
        tangent = cross3(axis, [0.0, 1.0, 0.0]);
    }
    let tangent = normalize_or(tangent, [0.0, 1.0, 0.0]);
    let bitangent = normalize_or(cross3(axis, tangent), [0.0, 0.0, 1.0]);

    for (center, radius) in [(start, start_radius), (end, end_radius)] {
        for segment in 0..segments {
            let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
            let offset = add3(
                scale3(tangent, radius * theta.cos()),
                scale3(bitangent, radius * theta.sin()),
            );
            vertices.push(add3(center, offset));
        }
    }
    let start_center = vertices.len() as u32;
    vertices.push(start);
    let end_center = vertices.len() as u32;
    vertices.push(end);

    for segment in 0..segments {
        let next = (segment + 1) % segments;
        let a = base + segment as u32;
        let b = base + next as u32;
        let c = base + segments as u32 + next as u32;
        let d = base + segments as u32 + segment as u32;
        faces.push([a, d, c]);
        faces.push([a, c, b]);
        faces.push([start_center, b, a]);
        faces.push([end_center, d, c]);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_torus_arc(
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    center: [f32; 3],
    major: f32,
    tube: f32,
    start_angle: f32,
    end_angle: f32,
    arc_segments: usize,
    tube_segments: usize,
) {
    let arc_segments = arc_segments.max(3);
    let tube_segments = tube_segments.max(6);
    let base = vertices.len() as u32;
    for arc in 0..=arc_segments {
        let t = arc as f32 / arc_segments as f32;
        let angle = start_angle + (end_angle - start_angle) * t;
        let radial = [angle.cos(), 0.0, angle.sin()];
        let centerline = add3(center, scale3(radial, major));
        for tube_idx in 0..tube_segments {
            let phi = std::f32::consts::TAU * tube_idx as f32 / tube_segments as f32;
            let offset = add3(
                scale3(radial, tube * phi.cos()),
                [0.0, tube * phi.sin(), 0.0],
            );
            vertices.push(add3(centerline, offset));
        }
    }

    for arc in 0..arc_segments {
        for tube_idx in 0..tube_segments {
            let next_tube = (tube_idx + 1) % tube_segments;
            let a = base + (arc * tube_segments + tube_idx) as u32;
            let b = base + ((arc + 1) * tube_segments + tube_idx) as u32;
            let c = base + ((arc + 1) * tube_segments + next_tube) as u32;
            let d = base + (arc * tube_segments + next_tube) as u32;
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
}

fn teapot_like_color(position: [f32; 3], scale: f32) -> [f32; 3] {
    [
        (position[0] / (2.2 * scale) + 0.5).clamp(0.0, 1.0),
        (position[1] / (1.1 * scale) + 0.5).clamp(0.0, 1.0),
        (position[2] / (1.3 * scale) + 0.44).clamp(0.0, 1.0),
    ]
}

fn colored_mesh(
    vertices: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
) -> AutomataResult<TriangleMeshTarget> {
    let target = TriangleMeshTarget::new(vertices, faces)?;
    let (bounds_min, bounds_max) = target.bounds();
    let colors = target
        .vertices
        .iter()
        .map(|position| normalized_position_color(*position, bounds_min, bounds_max))
        .collect::<Vec<_>>();
    target.with_vertex_colors(colors)
}

fn normalized_position_color(
    position: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> [f32; 3] {
    [
        normalize_bound(position[0], bounds_min[0], bounds_max[0]),
        normalize_bound(position[1], bounds_min[1], bounds_max[1]),
        normalize_bound(position[2], bounds_min[2], bounds_max[2]),
    ]
}

fn normalize_bound(value: f32, min: f32, max: f32) -> f32 {
    ((value - min) / (max - min).max(EPS)).clamp(0.0, 1.0)
}
