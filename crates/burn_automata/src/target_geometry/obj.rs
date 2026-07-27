use super::{EPS, TriangleMeshTarget, bounds_for_vertices, scale3, sub3};
use crate::{AutomataError, AutomataResult};

const UTAH_TEAPOT_OBJ: &str = include_str!("../../../../assets/meshes/utah_teapot.obj");

impl TriangleMeshTarget {
    pub fn utah_teapot(scale: f32) -> AutomataResult<Self> {
        Self::from_obj_str(UTAH_TEAPOT_OBJ, scale)
    }

    pub fn from_obj_str(obj: &str, scale: f32) -> AutomataResult<Self> {
        let z_up = obj.lines().take(32).any(|line| {
            let line = line.trim().to_ascii_lowercase();
            line.starts_with('#') && (line.contains("z-up") || line.contains("z up"))
        });
        Self::from_obj_str_with_transform(
            obj,
            scale,
            move |[x, y, z]| {
                if z_up { [x, z, y] } else { [x, y, z] }
            },
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: &str = "v 0 0 0\nv 2 0 0\nv 0 0 1\nf 1 2 3\n";

    #[test]
    fn obj_z_up_metadata_is_converted_to_bevy_y_up() {
        let z_up = TriangleMeshTarget::from_obj_str(
            &format!("# coordinate system: z-up\n{TRIANGLE}"),
            0.72,
        )
        .unwrap();
        let y_up = TriangleMeshTarget::from_obj_str(TRIANGLE, 0.72).unwrap();
        let (z_min, z_max) = z_up.bounds();
        let (y_min, y_max) = y_up.bounds();
        assert!(z_max[1] - z_min[1] > 0.5);
        assert!((z_max[2] - z_min[2]).abs() <= EPS);
        assert!(y_max[2] - y_min[2] > 0.5);
        assert!((y_max[1] - y_min[1]).abs() <= EPS);
    }
}
