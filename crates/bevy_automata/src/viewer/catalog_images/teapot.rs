use burn_automata::target_geometry::TriangleMeshTarget;

use super::raster::{blend_pixel, dot3, draw_disc, normalize3, to_u8, write_pixel};

pub(super) fn draw_teapot_morphogen_thumbnail(data: &mut [u8], width: u32, height: u32) {
    draw_teapot_target_preview(data, width, height, 0.78);
}

pub(super) fn draw_teapot_target_preview(data: &mut [u8], width: u32, height: u32, yaw: f32) {
    if let Ok(target) = TriangleMeshTarget::utah_teapot(0.72) {
        draw_mesh_target_preview(data, width, height, yaw, &target);
        return;
    }

    let scale = width.min(height) as f32 / 96.0;
    let cx = width as f32 * 0.50;
    let cy = height as f32 * 0.54;
    let yaw_shift = yaw.sin() * 4.0 * scale;

    draw_teapot_handle(data, width, height, cx - 22.0 * scale, cy, scale, yaw_shift);
    draw_teapot_spout(
        data,
        width,
        height,
        cx + 31.0 * scale,
        cy - 6.0 * scale,
        scale,
        yaw_shift,
    );
    draw_ellipse(
        data,
        width,
        height,
        cx,
        cy,
        28.0 * scale,
        20.0 * scale,
        [86, 177, 216, 232],
    );
    draw_ellipse(
        data,
        width,
        height,
        cx - 5.0 * scale,
        cy - 6.0 * scale,
        15.0 * scale,
        9.0 * scale,
        [120, 212, 178, 150],
    );
    draw_ellipse(
        data,
        width,
        height,
        cx,
        cy - 24.0 * scale,
        17.5 * scale,
        6.0 * scale,
        [217, 198, 117, 224],
    );
    draw_disc(
        data,
        width,
        height,
        cx,
        cy - 32.0 * scale,
        4.8 * scale,
        [228, 132, 124, 228],
    );
    draw_disc(
        data,
        width,
        height,
        cx - 10.0 * scale,
        cy - 8.0 * scale,
        4.0 * scale,
        [245, 255, 226, 82],
    );
}

fn draw_mesh_target_preview(
    data: &mut [u8],
    width: u32,
    height: u32,
    yaw: f32,
    target: &TriangleMeshTarget,
) {
    let pitch = -0.52_f32;
    let projected = target
        .vertices
        .iter()
        .map(|position| teapot_preview_project(*position, yaw, pitch))
        .collect::<Vec<_>>();
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for point in &projected {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }

    let extent_x = (max_x - min_x).max(1.0e-4);
    let extent_y = (max_y - min_y).max(1.0e-4);
    let pixel_scale = (width as f32 * 0.82 / extent_x).min(height as f32 * 0.76 / extent_y);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let screen = projected
        .iter()
        .map(|point| {
            [
                width as f32 * 0.50 + (point[0] - center_x) * pixel_scale,
                height as f32 * 0.54 - (point[1] - center_y) * pixel_scale,
                point[2],
            ]
        })
        .collect::<Vec<_>>();

    let mut z_buffer = vec![f32::MIN; (width * height) as usize];
    let colors = target.colors.as_deref();
    let light = normalize3([0.32, -0.48, 0.82]);
    for (face_index, face) in target.faces.iter().enumerate() {
        let a = screen[face[0] as usize];
        let b = screen[face[1] as usize];
        let c = screen[face[2] as usize];
        let area = edge2(a, b, c);
        if area.abs() <= 1.0e-4 {
            continue;
        }
        let normal = teapot_preview_rotate(target.face_normals[face_index], yaw, pitch);
        let diffuse = dot3(normalize3(normal), light).max(0.0);
        let shade = 0.34 + diffuse * 0.76;
        let min_px = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i32;
        let max_px = a[0].max(b[0]).max(c[0]).ceil().min(width as f32 - 1.0) as i32;
        let min_py = a[1].min(b[1]).min(c[1]).floor().max(0.0) as i32;
        let max_py = a[1].max(b[1]).max(c[1]).ceil().min(height as f32 - 1.0) as i32;
        for py in min_py..=max_py {
            for px in min_px..=max_px {
                let point = [px as f32 + 0.5, py as f32 + 0.5, 0.0];
                let w0 = edge2(b, c, point) / area;
                let w1 = edge2(c, a, point) / area;
                let w2 = edge2(a, b, point) / area;
                if w0 < -1.0e-4 || w1 < -1.0e-4 || w2 < -1.0e-4 {
                    continue;
                }
                let depth = a[2] * w0 + b[2] * w1 + c[2] * w2;
                let z_index = (py as u32 * width + px as u32) as usize;
                if depth <= z_buffer[z_index] {
                    continue;
                }
                z_buffer[z_index] = depth;
                let rgb = if let Some(colors) = colors {
                    let ca = colors[face[0] as usize];
                    let cb = colors[face[1] as usize];
                    let cc = colors[face[2] as usize];
                    [
                        ca[0] * w0 + cb[0] * w1 + cc[0] * w2,
                        ca[1] * w0 + cb[1] * w1 + cc[1] * w2,
                        ca[2] * w0 + cb[2] * w1 + cc[2] * w2,
                    ]
                } else {
                    [0.50, 0.78, 0.86]
                };
                write_pixel(
                    data,
                    width,
                    px,
                    py,
                    [
                        to_u8(rgb[0] * shade),
                        to_u8(rgb[1] * shade),
                        to_u8(rgb[2] * shade),
                        255,
                    ],
                );
            }
        }
    }
}

fn teapot_preview_project(position: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let rotated = teapot_preview_rotate(position, yaw, pitch);
    [rotated[0], rotated[2], -rotated[1]]
}

fn teapot_preview_rotate(position: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (yaw_sin, yaw_cos) = yaw.sin_cos();
    let x = position[0] * yaw_cos - position[1] * yaw_sin;
    let y = position[0] * yaw_sin + position[1] * yaw_cos;
    let z = position[2];
    let (pitch_sin, pitch_cos) = pitch.sin_cos();
    [
        x,
        y * pitch_cos - z * pitch_sin,
        y * pitch_sin + z * pitch_cos,
    ]
}

fn edge2(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn draw_teapot_spout(
    data: &mut [u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    scale: f32,
    yaw_shift: f32,
) {
    for i in 0..18 {
        let t = i as f32 / 17.0;
        let radius = (5.0 - t * 2.8) * scale;
        draw_disc(
            data,
            width,
            height,
            x + t * (30.0 * scale + yaw_shift * 0.8),
            y - t * 9.0 * scale,
            radius.max(1.2 * scale),
            [88, 183, 214, 210],
        );
    }
}

fn draw_teapot_handle(
    data: &mut [u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    scale: f32,
    yaw_shift: f32,
) {
    for i in 0..42 {
        let t = i as f32 / 41.0;
        let angle = std::f32::consts::PI - 1.15 + 2.30 * t;
        let px = x + angle.cos() * (18.0 * scale + yaw_shift.abs() * 0.4) - yaw_shift;
        let py = y + angle.sin() * 21.0 * scale;
        draw_disc(
            data,
            width,
            height,
            px,
            py,
            3.6 * scale,
            [106, 197, 202, 190],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_ellipse(
    data: &mut [u8],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: [u8; 4],
) {
    let min_x = (cx - rx - 1.0).floor() as i32;
    let max_x = (cx + rx + 1.0).ceil() as i32;
    let min_y = (cy - ry - 1.0).floor() as i32;
    let max_y = (cy + ry + 1.0).ceil() as i32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f32 + 0.5 - cx) / rx.max(1.0);
            let dy = (y as f32 + 0.5 - cy) / ry.max(1.0);
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (1.0 - distance).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let mut blended = color;
                blended[3] = (color[3] as f32 * coverage.sqrt()) as u8;
                blend_pixel(data, width, height, x, y, blended);
            }
        }
    }
}
