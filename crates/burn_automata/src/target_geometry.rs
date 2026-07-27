use rand::Rng;

use crate::{AutomataError, AutomataResult};

mod constructors;
mod obj;

const EPS: f32 = 1.0e-6;

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
    projection_bvh: MeshProjectionBvh,
    pub colors: Option<Vec<[f32; 3]>>,
}

#[derive(Clone, Debug)]
struct MeshProjectionBvh {
    nodes: Vec<MeshProjectionBvhNode>,
    face_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct MeshProjectionBvhNode {
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    children: Option<[usize; 2]>,
    face_start: usize,
    face_count: usize,
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
            projection_bvh: MeshProjectionBvh {
                nodes: Vec::new(),
                face_indices: Vec::new(),
            },
            colors: None,
        };
        target.recompute_normals();
        target.projection_bvh = MeshProjectionBvh::build(&target.vertices, &target.faces);
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
        let best = self
            .projection_bvh
            .closest_point(query, &self.vertices, &self.faces);

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

impl MeshProjectionBvh {
    const LEAF_FACES: usize = 8;

    fn build(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Self {
        let mut face_indices = (0..faces.len()).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(faces.len().saturating_mul(2));
        Self::build_node(vertices, faces, &mut face_indices, 0, &mut nodes);
        Self {
            nodes,
            face_indices,
        }
    }

    fn build_node(
        vertices: &[[f32; 3]],
        faces: &[[u32; 3]],
        face_indices: &mut [usize],
        face_start: usize,
        nodes: &mut Vec<MeshProjectionBvhNode>,
    ) -> usize {
        let (bounds_min, bounds_max, centroid_min, centroid_max) =
            bvh_bounds(vertices, faces, face_indices);
        let node_index = nodes.len();
        nodes.push(MeshProjectionBvhNode {
            bounds_min,
            bounds_max,
            children: None,
            face_start,
            face_count: face_indices.len(),
        });
        if face_indices.len() <= Self::LEAF_FACES {
            return node_index;
        }

        let extents = [
            centroid_max[0] - centroid_min[0],
            centroid_max[1] - centroid_min[1],
            centroid_max[2] - centroid_min[2],
        ];
        let axis = if extents[1] > extents[0] && extents[1] >= extents[2] {
            1
        } else if extents[2] > extents[0] {
            2
        } else {
            0
        };
        face_indices.sort_unstable_by(|left, right| {
            triangle_centroid(vertices, faces[*left])[axis]
                .total_cmp(&triangle_centroid(vertices, faces[*right])[axis])
                .then_with(|| left.cmp(right))
        });
        let midpoint = face_indices.len() / 2;
        let (left_faces, right_faces) = face_indices.split_at_mut(midpoint);
        let left = Self::build_node(vertices, faces, left_faces, face_start, nodes);
        let right = Self::build_node(vertices, faces, right_faces, face_start + midpoint, nodes);
        nodes[node_index].children = Some([left, right]);
        nodes[node_index].face_count = 0;
        node_index
    }

    fn closest_point(
        &self,
        query: [f32; 3],
        vertices: &[[f32; 3]],
        faces: &[[u32; 3]],
    ) -> ClosestPoint {
        let mut best = ClosestPoint {
            point: vertices[0],
            barycentric: [1.0, 0.0, 0.0],
            face_index: 0,
            distance2: f32::MAX,
        };
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if point_aabb_distance2(query, node.bounds_min, node.bounds_max) > best.distance2 {
                continue;
            }
            if let Some([left, right]) = node.children {
                let left_node = self.nodes[left];
                let right_node = self.nodes[right];
                let left_distance =
                    point_aabb_distance2(query, left_node.bounds_min, left_node.bounds_max);
                let right_distance =
                    point_aabb_distance2(query, right_node.bounds_min, right_node.bounds_max);
                if left_distance <= right_distance {
                    if right_distance <= best.distance2 {
                        stack.push(right);
                    }
                    if left_distance <= best.distance2 {
                        stack.push(left);
                    }
                } else {
                    if left_distance <= best.distance2 {
                        stack.push(left);
                    }
                    if right_distance <= best.distance2 {
                        stack.push(right);
                    }
                }
                continue;
            }

            for slot in node.face_start..node.face_start + node.face_count {
                let face_index = self.face_indices[slot];
                let face = faces[face_index];
                let candidate = closest_point_on_triangle(
                    query,
                    vertices[face[0] as usize],
                    vertices[face[1] as usize],
                    vertices[face[2] as usize],
                );
                if candidate.distance2 < best.distance2 {
                    best = ClosestPoint {
                        face_index,
                        ..candidate
                    };
                }
            }
        }
        best
    }
}

fn bvh_bounds(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    face_indices: &[usize],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let mut bounds_min = [f32::MAX; 3];
    let mut bounds_max = [f32::MIN; 3];
    let mut centroid_min = [f32::MAX; 3];
    let mut centroid_max = [f32::MIN; 3];
    for &face_index in face_indices {
        let face = faces[face_index];
        let centroid = triangle_centroid(vertices, face);
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(centroid[axis]);
            for vertex in face {
                let value = vertices[vertex as usize][axis];
                bounds_min[axis] = bounds_min[axis].min(value);
                bounds_max[axis] = bounds_max[axis].max(value);
            }
        }
    }
    (bounds_min, bounds_max, centroid_min, centroid_max)
}

fn triangle_centroid(vertices: &[[f32; 3]], face: [u32; 3]) -> [f32; 3] {
    let a = vertices[face[0] as usize];
    let b = vertices[face[1] as usize];
    let c = vertices[face[2] as usize];
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn point_aabb_distance2(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut distance2 = 0.0;
    for axis in 0..3 {
        let delta = if point[axis] < min[axis] {
            min[axis] - point[axis]
        } else if point[axis] > max[axis] {
            point[axis] - max[axis]
        } else {
            0.0
        };
        distance2 += delta * delta;
    }
    distance2
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teapot_projection_bvh_matches_exhaustive_triangle_search() {
        let target = TriangleMeshTarget::utah_teapot(0.72).unwrap();
        for query_index in 0..48 {
            let query = [
                radical_inverse(query_index + 1, 2) * 2.0 - 1.0,
                radical_inverse(query_index + 1, 3) * 2.0 - 1.0,
                radical_inverse(query_index + 1, 5) * 2.0 - 1.0,
            ];
            let accelerated =
                target
                    .projection_bvh
                    .closest_point(query, &target.vertices, &target.faces);
            let exhaustive = target
                .faces
                .iter()
                .enumerate()
                .map(|(face_index, face)| {
                    let candidate = closest_point_on_triangle(
                        query,
                        target.vertices[face[0] as usize],
                        target.vertices[face[1] as usize],
                        target.vertices[face[2] as usize],
                    );
                    ClosestPoint {
                        face_index,
                        ..candidate
                    }
                })
                .min_by(|left, right| left.distance2.total_cmp(&right.distance2))
                .unwrap();
            assert!(
                (accelerated.distance2 - exhaustive.distance2).abs() <= 1.0e-6,
                "query {query_index}: accelerated={} exhaustive={}",
                accelerated.distance2,
                exhaustive.distance2
            );
            assert!(
                length3(sub3(accelerated.point, exhaustive.point)) <= 1.0e-4,
                "query {query_index}: accelerated={:?} exhaustive={:?}",
                accelerated.point,
                exhaustive.point
            );
        }
    }
}
