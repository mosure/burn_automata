#![allow(clippy::too_many_arguments)]

pub(super) fn fill_thumbnail_background(data: &mut [u8], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let vignette = ((x as f32 / width as f32 - 0.5).abs()
                + (y as f32 / height as f32 - 0.5).abs())
                * 0.045;
            let grid = if (x / 12 + y / 12) % 2 == 0 {
                0.006
            } else {
                0.0
            };
            write_pixel(
                data,
                width,
                x as i32,
                y as i32,
                [
                    to_u8(0.020 + grid - vignette),
                    to_u8(0.026 + grid - vignette),
                    to_u8(0.032 + grid - vignette),
                    255,
                ],
            );
        }
    }
}

pub(super) fn draw_line_dots(
    data: &mut [u8],
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    dots: usize,
    color: [u8; 4],
) {
    for i in 0..dots {
        let t = if dots <= 1 {
            0.0
        } else {
            i as f32 / (dots - 1) as f32
        };
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        draw_disc(data, width, height, x, y, 1.7, color);
    }
}

pub(super) fn draw_disc(
    data: &mut [u8],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 4],
) {
    let min_x = (cx - radius - 1.0).floor() as i32;
    let max_x = (cx + radius + 1.0).ceil() as i32;
    let min_y = (cy - radius - 1.0).floor() as i32;
    let max_y = (cy + radius + 1.0).ceil() as i32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0);
            if coverage > 0.0 {
                let mut blended = color;
                blended[3] = (color[3] as f32 * coverage) as u8;
                blend_pixel(data, width, height, x, y, blended);
            }
        }
    }
}

pub(super) fn write_pixel(data: &mut [u8], width: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as u32;
    let y = y as u32;
    let index = ((y * width + x) * 4) as usize;
    if index + 3 >= data.len() {
        return;
    }
    data[index..index + 4].copy_from_slice(&color);
}

pub(super) fn blend_pixel(
    data: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: [u8; 4],
) {
    if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
        return;
    }
    let index = (((y as u32) * width + x as u32) * 4) as usize;
    let alpha = color[3] as f32 / 255.0;
    let inverse = 1.0 - alpha;
    data[index] = (color[0] as f32 * alpha + data[index] as f32 * inverse) as u8;
    data[index + 1] = (color[1] as f32 * alpha + data[index + 1] as f32 * inverse) as u8;
    data[index + 2] = (color[2] as f32 * alpha + data[index + 2] as f32 * inverse) as u8;
    data[index + 3] = 255;
}

pub(super) fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(super) fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 {
        [0.0, 0.0, 1.0]
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

pub(super) fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}
