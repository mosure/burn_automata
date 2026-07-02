use super::raster::{draw_disc, draw_line_dots};

pub(super) fn draw_lizard_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for i in 0..64 {
        let t = i as f32 / 63.0;
        let x = 14.0 + t * 56.0;
        let y = 38.0 + (t * 5.0).sin() * 4.0;
        let r = 5.2 - (t - 0.45).abs() * 4.0;
        draw_disc(data, width, height, x, y, r.max(2.0), [130, 214, 144, 210]);
    }
    for i in 0..20 {
        let t = i as f32 / 19.0;
        draw_disc(
            data,
            width,
            height,
            68.0 + t * 11.0,
            34.0 - t * 4.0,
            2.3 - t * 0.6,
            [155, 231, 164, 220],
        );
    }
    for &(x0, y0, x1, y1) in &[
        (32.0, 40.0, 23.0, 51.0),
        (39.0, 38.0, 31.0, 25.0),
        (51.0, 38.0, 60.0, 25.0),
        (56.0, 40.0, 67.0, 51.0),
    ] {
        draw_line_dots(data, width, height, x0, y0, x1, y1, 5, [95, 185, 124, 185]);
    }
    draw_disc(data, width, height, 67.0, 32.0, 1.5, [245, 252, 214, 240]);
}

pub(super) fn draw_polka_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for gy in 0..5 {
        for gx in 0..7 {
            let offset = if gy % 2 == 0 { 0.0 } else { 7.0 };
            let x = 12.0 + gx as f32 * 14.0 + offset;
            let y = 9.0 + gy as f32 * 14.0;
            let r = if (gx + gy) % 3 == 0 { 4.8 } else { 3.4 };
            let color = if (gx + gy) % 2 == 0 {
                [224, 116, 116, 230]
            } else {
                [236, 221, 140, 220]
            };
            draw_disc(data, width, height, x, y, r, color);
        }
    }
}

pub(super) fn draw_growing_2d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for i in 0..180 {
        let t = i as f32;
        let angle = t * 2.3999631;
        let radius = t.sqrt() * 2.1;
        let x = 48.0 + angle.cos() * radius;
        let y = 36.0 + angle.sin() * radius;
        let alpha = (225.0 - radius * 4.0).clamp(60.0, 225.0) as u8;
        draw_disc(data, width, height, x, y, 1.3, [102, 196, 210, alpha]);
    }
    draw_disc(data, width, height, 48.0, 36.0, 5.0, [228, 245, 197, 210]);
}

pub(super) fn draw_texture_2d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    for y in (8..height - 8).step_by(7) {
        for x in (8..width - 8).step_by(7) {
            let wave = ((x as f32 * 0.18).sin() + (y as f32 * 0.24).cos()) * 0.5;
            let color = if wave > 0.0 {
                [82, 175, 214, 190]
            } else {
                [212, 182, 108, 185]
            };
            draw_disc(
                data,
                width,
                height,
                x as f32 + wave * 2.0,
                y as f32 - wave * 1.5,
                1.8,
                color,
            );
        }
    }
}

pub(super) fn draw_growing_3d_thumbnail(data: &mut [u8], width: u32, height: u32) {
    let points = [
        (46.0, 35.0, 8.0, [108, 190, 220, 180]),
        (34.0, 30.0, 4.6, [218, 203, 122, 170]),
        (60.0, 31.0, 5.4, [125, 223, 158, 190]),
        (53.0, 47.0, 4.4, [218, 128, 128, 175]),
        (42.0, 50.0, 3.8, [180, 160, 230, 160]),
        (69.0, 43.0, 3.2, [106, 176, 224, 155]),
        (26.0, 43.0, 3.0, [160, 220, 164, 155]),
    ];
    for (x, y, radius, color) in points {
        draw_disc(data, width, height, x, y, radius, color);
        draw_disc(
            data,
            width,
            height,
            x - radius * 0.25,
            y - radius * 0.25,
            radius * 0.35,
            [250, 255, 240, 92],
        );
    }
}

pub(super) fn draw_point_mnist_thumbnail(data: &mut [u8], width: u32, height: u32) {
    let segments = [
        (34.0, 18.0, 63.0, 18.0),
        (63.0, 18.0, 62.0, 35.0),
        (41.0, 35.0, 62.0, 35.0),
        (62.0, 35.0, 63.0, 53.0),
        (34.0, 53.0, 63.0, 53.0),
    ];
    for (x0, y0, x1, y1) in segments {
        draw_line_dots(
            data,
            width,
            height,
            x0,
            y0,
            x1,
            y1,
            10,
            [230, 235, 173, 220],
        );
    }
    for i in 0..36 {
        let t = i as f32 / 35.0;
        let x = 30.0 + (t * 38.0).sin() * 3.0;
        let y = 16.0 + t * 39.0;
        draw_disc(data, width, height, x, y, 1.0, [94, 179, 210, 140]);
    }
}
