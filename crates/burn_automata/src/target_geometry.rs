use rand::Rng;

use crate::{AutomataError, AutomataResult};

const EPS: f32 = 1.0e-6;
const UTAH_TEAPOT_OBJ: &str = include_str!("../../../assets/meshes/utah_teapot.obj");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetSurfaceSample {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetProjection {
    pub query: [f32; 3],
    pub closest: [f32; 3],
    pub normal: [f32; 3],
    pub residual: [f32; 3],
    pub signed_distance: f32,
    pub distance: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct TriangleMeshTarget {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub vertex_normals: Vec<[f32; 3]>,
    pub face_normals: Vec<[f32; 3]>,
    pub face_areas: Vec<f32>,
    face_area_prefix: Vec<f32>,
    pub colors: Option<Vec<[f32; 3]>>,
}

#[derive(Clone, Debug)]
pub struct OvoxelTarget {
    pub coords: Vec<[u32; 4]>,
    pub vertices: Vec<[f32; 3]>,
    pub intersected: Vec<[bool; 3]>,
    pub intersection_logits: Vec<[f32; 3]>,
    pub quad_lerp: Vec<f32>,
    pub mesh: TriangleMeshTarget,
}

impl TriangleMeshTarget {
    pub fn new(vertices: Vec<[f32; 3]>, mut faces: Vec<[u32; 3]>) -> AutomataResult<Self> {
        if vertices.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "mesh target requires at least one vertex".to_string(),
            ));
        }
        if faces.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "mesh target requires at least one face".to_string(),
            ));
        }
        for (idx, face) in faces.iter().enumerate() {
            for vertex in face {
                if *vertex as usize >= vertices.len() {
                    return Err(AutomataError::InvalidArgument(format!(
                        "mesh face {idx} references missing vertex {vertex}"
                    )));
                }
            }
        }

        let volume = mesh_signed_volume(&vertices, &faces);
        if volume < -EPS {
            for face in &mut faces {
                face.swap(1, 2);
            }
        }

        let mut target = Self {
            vertices,
            faces,
            vertex_normals: Vec::new(),
            face_normals: Vec::new(),
            face_areas: Vec::new(),
            face_area_prefix: Vec::new(),
            colors: None,
        };
        target.recompute_normals();
        Ok(target)
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

    pub fn utah_teapot(scale: f32) -> AutomataResult<Self> {
        Self::from_obj_str_with_transform(UTAH_TEAPOT_OBJ, scale, |[x, y, z]| [x, z, y])
    }

    pub fn from_obj_str(obj: &str, scale: f32) -> AutomataResult<Self> {
        Self::from_obj_str_with_transform(obj, scale, |position| position)
    }

    fn from_obj_str_with_transform<F>(
        obj: &str,
        scale: f32,
        mut transform: F,
    ) -> AutomataResult<Self>
    where
        F: FnMut([f32; 3]) -> [f32; 3],
    {
        let scale = scale.max(1.0e-4);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for (line_idx, line) in obj.lines().enumerate() {
            let mut parts = line.split_whitespace();
            let Some(kind) = parts.next() else {
                continue;
            };
            match kind {
                "#" | "o" | "g" | "s" | "vn" | "vt" | "usemtl" | "mtllib" => {}
                "v" => {
                    let x = parse_obj_f32(parts.next(), line_idx, "x")?;
                    let y = parse_obj_f32(parts.next(), line_idx, "y")?;
                    let z = parse_obj_f32(parts.next(), line_idx, "z")?;
                    vertices.push(transform([x, y, z]));
                }
                "f" => {
                    let polygon = parts
                        .map(|part| parse_obj_vertex_index(part, vertices.len(), line_idx))
                        .collect::<AutomataResult<Vec<_>>>()?;
                    if polygon.len() < 3 {
                        return Err(AutomataError::InvalidArgument(format!(
                            "OBJ face on line {} has fewer than 3 vertices",
                            line_idx + 1
                        )));
                    }
                    for tri in 1..polygon.len() - 1 {
                        faces.push([polygon[0], polygon[tri], polygon[tri + 1]]);
                    }
                }
                _ => {}
            }
        }

        if vertices.is_empty() || faces.is_empty() {
            return Err(AutomataError::InvalidArgument(
                "OBJ mesh requires at least one vertex and one face".to_string(),
            ));
        }

        let (bounds_min, bounds_max) = bounds_for_vertices(&vertices);
        let center = [
            (bounds_min[0] + bounds_max[0]) * 0.5,
            (bounds_min[1] + bounds_max[1]) * 0.5,
            (bounds_min[2] + bounds_max[2]) * 0.5,
        ];
        let extent = (bounds_max[0] - bounds_min[0])
            .max(bounds_max[1] - bounds_min[1])
            .max(bounds_max[2] - bounds_min[2])
            .max(EPS);
        let mesh_scale = 2.0 * scale / extent;
        for vertex in &mut vertices {
            *vertex = scale3(sub3(*vertex, center), mesh_scale);
        }

        let (scaled_min, scaled_max) = bounds_for_vertices(&vertices);
        let colors = vertices
            .iter()
            .map(|position| normalized_position_color(*position, scaled_min, scaled_max))
            .collect::<Vec<_>>();

        Self::new(vertices, faces)?.with_vertex_colors(colors)
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

    pub fn with_vertex_colors(mut self, colors: Vec<[f32; 3]>) -> AutomataResult<Self> {
        if colors.len() != self.vertices.len() {
            return Err(AutomataError::InvalidArgument(format!(
                "mesh target expected {} vertex colors, got {}",
                self.vertices.len(),
                colors.len()
            )));
        }
        self.colors = Some(colors);
        Ok(self)
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        bounds_for_vertices(&self.vertices)
    }

    pub fn project(&self, query: [f32; 3]) -> TargetProjection {
        let mut best = ClosestPoint {
            point: self.vertices[0],
            barycentric: [1.0, 0.0, 0.0],
            face_index: 0,
            distance2: f32::MAX,
        };

        for (face_index, face) in self.faces.iter().enumerate() {
            let a = self.vertices[face[0] as usize];
            let b = self.vertices[face[1] as usize];
            let c = self.vertices[face[2] as usize];
            let candidate = closest_point_on_triangle(query, a, b, c);
            if candidate.distance2 < best.distance2 {
                best = ClosestPoint {
                    face_index,
                    ..candidate
                };
            }
        }

        let face = self.faces[best.face_index];
        let normal = self.interpolated_normal(face, best.barycentric);
        let color = self.interpolated_color(face, best.barycentric, normal);
        let query_to_closest = sub3(best.point, query);
        let closest_to_query = sub3(query, best.point);
        let signed_distance = dot3(closest_to_query, normal);
        TargetProjection {
            query,
            closest: best.point,
            normal,
            residual: query_to_closest,
            signed_distance,
            distance: best.distance2.sqrt(),
            color,
        }
    }

    pub fn surface_sample(&self, sample_index: usize) -> TargetSurfaceSample {
        let face_index = self.area_weighted_face_index(sample_index);
        let face = self.faces[face_index];
        let barycentric = low_discrepancy_triangle_barycentric(sample_index);
        self.surface_sample_on_face(face, barycentric)
    }

    pub fn random_surface_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> TargetSurfaceSample {
        let face_index = self.random_area_weighted_face_index(rng.random::<f32>());
        let u = rng.random::<f32>().clamp(EPS, 1.0 - EPS);
        let v = rng.random::<f32>().clamp(0.0, 1.0);
        let sqrt_u = u.sqrt();
        let barycentric = [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v];
        self.surface_sample_on_face(self.faces[face_index], barycentric)
    }

    fn surface_sample_on_face(&self, face: [u32; 3], barycentric: [f32; 3]) -> TargetSurfaceSample {
        let a = self.vertices[face[0] as usize];
        let b = self.vertices[face[1] as usize];
        let c = self.vertices[face[2] as usize];
        let position = add3(
            add3(scale3(a, barycentric[0]), scale3(b, barycentric[1])),
            scale3(c, barycentric[2]),
        );
        let normal = self.interpolated_normal(face, barycentric);
        let color = self.interpolated_color(face, barycentric, normal);
        TargetSurfaceSample {
            position,
            normal,
            color,
        }
    }

    fn area_weighted_face_index(&self, sample_index: usize) -> usize {
        let Some(total_area) = self.face_area_prefix.last().copied() else {
            return sample_index % self.faces.len().max(1);
        };
        if total_area <= EPS || !total_area.is_finite() {
            return sample_index % self.faces.len().max(1);
        }

        let target = radical_inverse(sample_index as u64 + 1, 2) * total_area;
        let mut lo = 0usize;
        let mut hi = self.face_area_prefix.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.face_area_prefix[mid] <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.min(self.faces.len().saturating_sub(1))
    }

    fn random_area_weighted_face_index(&self, unit: f32) -> usize {
        let Some(total_area) = self.face_area_prefix.last().copied() else {
            return 0;
        };
        if total_area <= EPS || !total_area.is_finite() {
            return 0;
        }

        let target = unit.clamp(0.0, 1.0 - EPS) * total_area;
        let mut lo = 0usize;
        let mut hi = self.face_area_prefix.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.face_area_prefix[mid] <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.min(self.faces.len().saturating_sub(1))
    }

    pub fn near_surface_query(&self, sample_index: usize, signed_offset: f32) -> [f32; 3] {
        let sample = self.surface_sample(sample_index);
        add3(sample.position, scale3(sample.normal, signed_offset))
    }

    fn recompute_normals(&mut self) {
        self.vertex_normals = vec![[0.0; 3]; self.vertices.len()];
        self.face_normals = Vec::with_capacity(self.faces.len());
        self.face_areas = Vec::with_capacity(self.faces.len());
        self.face_area_prefix = Vec::with_capacity(self.faces.len());
        let mut area_sum = 0.0_f32;

        for face in &self.faces {
            let a = self.vertices[face[0] as usize];
            let b = self.vertices[face[1] as usize];
            let c = self.vertices[face[2] as usize];
            let area_normal = cross3(sub3(b, a), sub3(c, a));
            let area2 = length3(area_normal);
            let normal = if area2 > EPS {
                scale3(area_normal, 1.0 / area2)
            } else {
                [0.0, 0.0, 1.0]
            };
            self.face_normals.push(normal);
            let area = 0.5 * area2;
            self.face_areas.push(area);
            area_sum += area.max(0.0);
            self.face_area_prefix.push(area_sum);
            for vertex in face {
                let slot = &mut self.vertex_normals[*vertex as usize];
                *slot = add3(*slot, area_normal);
            }
        }

        for normal in &mut self.vertex_normals {
            *normal = normalize_or(*normal, [0.0, 0.0, 1.0]);
        }
    }

    fn interpolated_normal(&self, face: [u32; 3], barycentric: [f32; 3]) -> [f32; 3] {
        let mut normal = [0.0; 3];
        for (axis, value) in normal.iter_mut().enumerate() {
            *value = self.vertex_normals[face[0] as usize][axis] * barycentric[0]
                + self.vertex_normals[face[1] as usize][axis] * barycentric[1]
                + self.vertex_normals[face[2] as usize][axis] * barycentric[2];
        }
        normalize_or(normal, self.face_normals[0])
    }

    fn interpolated_color(
        &self,
        face: [u32; 3],
        barycentric: [f32; 3],
        normal: [f32; 3],
    ) -> [f32; 3] {
        if let Some(colors) = &self.colors {
            let mut color = [0.0; 3];
            for (axis, value) in color.iter_mut().enumerate() {
                *value = colors[face[0] as usize][axis] * barycentric[0]
                    + colors[face[1] as usize][axis] * barycentric[1]
                    + colors[face[2] as usize][axis] * barycentric[2];
            }
            color
        } else {
            [
                (0.5 + 0.5 * normal[0]).clamp(0.0, 1.0),
                (0.5 + 0.5 * normal[1]).clamp(0.0, 1.0),
                (0.5 + 0.5 * normal[2]).clamp(0.0, 1.0),
            ]
        }
    }
}

fn low_discrepancy_triangle_barycentric(sample_index: usize) -> [f32; 3] {
    let u = radical_inverse(sample_index as u64 + 1, 3).clamp(EPS, 1.0 - EPS);
    let v = radical_inverse(sample_index as u64 + 1, 5).clamp(0.0, 1.0);
    let sqrt_u = u.sqrt();
    [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v]
}

fn radical_inverse(mut n: u64, base: u32) -> f32 {
    debug_assert!(base >= 2);
    let inv_base = 1.0 / base as f32;
    let mut inv = inv_base;
    let mut value = 0.0_f32;
    while n > 0 {
        let digit = (n % base as u64) as f32;
        value += digit * inv;
        n /= base as u64;
        inv *= inv_base;
    }
    value
}

fn parse_obj_f32(value: Option<&str>, line_idx: usize, axis: &str) -> AutomataResult<f32> {
    let Some(value) = value else {
        return Err(AutomataError::InvalidArgument(format!(
            "OBJ vertex on line {} is missing {axis}",
            line_idx + 1
        )));
    };
    value.parse::<f32>().map_err(|err| {
        AutomataError::InvalidArgument(format!(
            "OBJ vertex on line {} has invalid {axis} value `{value}`: {err}",
            line_idx + 1
        ))
    })
}

fn parse_obj_vertex_index(
    token: &str,
    vertex_count: usize,
    line_idx: usize,
) -> AutomataResult<u32> {
    let raw = token.split('/').next().unwrap_or_default();
    let index = raw.parse::<i32>().map_err(|err| {
        AutomataError::InvalidArgument(format!(
            "OBJ face on line {} has invalid vertex index `{raw}`: {err}",
            line_idx + 1
        ))
    })?;
    if index == 0 {
        return Err(AutomataError::InvalidArgument(format!(
            "OBJ face on line {} uses invalid 1-based index 0",
            line_idx + 1
        )));
    }
    let zero_based = if index > 0 {
        index - 1
    } else {
        vertex_count as i32 + index
    };
    if zero_based < 0 || zero_based as usize >= vertex_count {
        return Err(AutomataError::InvalidArgument(format!(
            "OBJ face on line {} references vertex {index} with only {vertex_count} vertices loaded",
            line_idx + 1
        )));
    }
    Ok(zero_based as u32)
}

fn bounds_for_vertices(vertices: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut bounds_min = [f32::MAX; 3];
    let mut bounds_max = [f32::MIN; 3];
    for vertex in vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex[axis]);
            bounds_max[axis] = bounds_max[axis].max(vertex[axis]);
        }
    }
    (bounds_min, bounds_max)
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

impl OvoxelTarget {
    pub fn new(
        coords: Vec<[u32; 4]>,
        vertices: Vec<[f32; 3]>,
        intersected: Vec<[bool; 3]>,
        intersection_logits: Vec<[f32; 3]>,
        quad_lerp: Vec<f32>,
        mesh: TriangleMeshTarget,
    ) -> AutomataResult<Self> {
        let voxel_count = coords.len();
        if vertices.len() != voxel_count
            || intersected.len() != voxel_count
            || intersection_logits.len() != voxel_count
            || quad_lerp.len() != voxel_count
        {
            return Err(AutomataError::InvalidArgument(format!(
                "O-Voxel target field lengths must match coords; coords={}, vertices={}, intersected={}, logits={}, quad_lerp={}",
                voxel_count,
                vertices.len(),
                intersected.len(),
                intersection_logits.len(),
                quad_lerp.len()
            )));
        }
        Ok(Self {
            coords,
            vertices,
            intersected,
            intersection_logits,
            quad_lerp,
            mesh,
        })
    }

    pub fn project(&self, query: [f32; 3]) -> TargetProjection {
        self.mesh.project(query)
    }
}

#[derive(Clone, Copy, Debug)]
struct ClosestPoint {
    point: [f32; 3],
    barycentric: [f32; 3],
    face_index: usize,
    distance2: f32,
}

fn closest_point_on_triangle(
    point: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> ClosestPoint {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(point, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return closest(a, [1.0, 0.0, 0.0], point);
    }

    let bp = sub3(point, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return closest(b, [0.0, 1.0, 0.0], point);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(EPS);
        return closest(add3(a, scale3(ab, v)), [1.0 - v, v, 0.0], point);
    }

    let cp = sub3(point, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return closest(c, [0.0, 0.0, 1.0], point);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(EPS);
        return closest(add3(a, scale3(ac, w)), [1.0 - w, 0.0, w], point);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(EPS);
        return closest(add3(b, scale3(sub3(c, b), w)), [0.0, 1.0 - w, w], point);
    }

    let denom = (va + vb + vc).max(EPS);
    let v = vb / denom;
    let w = vc / denom;
    let u = 1.0 - v - w;
    closest(
        add3(add3(scale3(a, u), scale3(b, v)), scale3(c, w)),
        [u, v, w],
        point,
    )
}

fn closest(point: [f32; 3], barycentric: [f32; 3], query: [f32; 3]) -> ClosestPoint {
    ClosestPoint {
        point,
        barycentric,
        face_index: 0,
        distance2: length2_3(sub3(point, query)),
    }
}

pub fn mesh_signed_volume(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> f32 {
    faces
        .iter()
        .map(|face| {
            let a = vertices[face[0] as usize];
            let b = vertices[face[1] as usize];
            let c = vertices[face[2] as usize];
            dot3(a, cross3(b, c)) / 6.0
        })
        .sum()
}

pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length2_3(v: [f32; 3]) -> f32 {
    dot3(v, v)
}

pub fn length3(v: [f32; 3]) -> f32 {
    length2_3(v).sqrt()
}

pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    normalize_or(v, [0.0, 0.0, 0.0])
}

pub fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = length3(v);
    if length > EPS {
        scale3(v, 1.0 / length)
    } else {
        fallback
    }
}
