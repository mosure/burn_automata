use burn_automata::rollout::{uv_torus_outer_radius, uv_torus_position_color};

use super::raster::{dot3, normalize3, to_u8, write_pixel};

pub(super) fn draw_uv_torus_morphogen_thumbnail(data: &mut [u8], width: u32, height: u32) {
    draw_uv_torus_target_preview(data, width, height, 1.18, 0.72);
}

pub(super) fn draw_uv_torus_target_preview(
    data: &mut [u8],
    width: u32,
    height: u32,
    yaw: f32,
    scale: f32,
) {
    let outer_radius = uv_torus_outer_radius(scale);
    let view_radius = outer_radius * 1.32;
    let aspect = width as f32 / height.max(1) as f32;
    let (yaw_sin, yaw_cos) = yaw.sin_cos();
    let pitch = -0.46_f32;
    let (pitch_sin, pitch_cos) = pitch.sin_cos();

    for py in 0..height {
        for px in 0..width {
            let sx = (((px as f32 + 0.5) / width as f32) - 0.5) * 2.0 * view_radius * aspect;
            let sy = (0.52 - ((py as f32 + 0.5) / height as f32)) * 2.0 * view_radius;
            let origin = torus_view_to_local(
                [sx, sy, outer_radius * 3.25],
                yaw_sin,
                yaw_cos,
                pitch_sin,
                pitch_cos,
            );
            let direction =
                torus_view_to_local([0.0, 0.0, -1.0], yaw_sin, yaw_cos, pitch_sin, pitch_cos);
            if let Some(position) = raymarch_torus(origin, direction, scale) {
                let normal = torus_sdf_normal(position, scale);
                let light_dir = normalize3([0.42, -0.35, 0.84]);
                let diffuse = dot3(normal, light_dir).max(0.0);
                let color = uv_torus_position_color(position, scale);
                let ambient = 0.34;
                let light = ambient + diffuse * 0.68;
                write_pixel(
                    data,
                    width,
                    px as i32,
                    py as i32,
                    [
                        to_u8(color[0] * light),
                        to_u8(color[1] * light),
                        to_u8(color[2] * light),
                        255,
                    ],
                );
            }
        }
    }
}

fn torus_view_to_local(
    value: [f32; 3],
    yaw_sin: f32,
    yaw_cos: f32,
    pitch_sin: f32,
    pitch_cos: f32,
) -> [f32; 3] {
    let x1 = value[0];
    let y1 = value[1] * pitch_cos + value[2] * pitch_sin;
    let z = -value[1] * pitch_sin + value[2] * pitch_cos;
    [x1 * yaw_cos + y1 * yaw_sin, -x1 * yaw_sin + y1 * yaw_cos, z]
}

fn raymarch_torus(origin: [f32; 3], direction: [f32; 3], scale: f32) -> Option<[f32; 3]> {
    let direction = normalize3(direction);
    let outer_radius = uv_torus_outer_radius(scale);
    let max_distance = outer_radius * 7.0;
    let hit_epsilon = outer_radius * 0.0035;
    let min_step = outer_radius * 0.0015;
    let mut distance = 0.0_f32;
    for _ in 0..112 {
        let position = [
            origin[0] + direction[0] * distance,
            origin[1] + direction[1] * distance,
            origin[2] + direction[2] * distance,
        ];
        let sdf = uv_torus_sdf(position, scale);
        if sdf.abs() <= hit_epsilon {
            return Some(position);
        }
        distance += sdf.max(min_step);
        if distance > max_distance {
            return None;
        }
    }
    None
}

fn uv_torus_sdf(position: [f32; 3], scale: f32) -> f32 {
    let major = scale.max(1.0e-4);
    let minor = major * burn_automata::rollout::UV_TORUS_MINOR_RATIO;
    let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
    ((radial - major).powi(2) + position[2].powi(2)).sqrt() - minor
}

fn torus_sdf_normal(position: [f32; 3], scale: f32) -> [f32; 3] {
    let eps = uv_torus_outer_radius(scale) * 0.0015;
    normalize3([
        uv_torus_sdf([position[0] + eps, position[1], position[2]], scale)
            - uv_torus_sdf([position[0] - eps, position[1], position[2]], scale),
        uv_torus_sdf([position[0], position[1] + eps, position[2]], scale)
            - uv_torus_sdf([position[0], position[1] - eps, position[2]], scale),
        uv_torus_sdf([position[0], position[1], position[2] + eps], scale)
            - uv_torus_sdf([position[0], position[1], position[2] - eps], scale),
    ])
}
