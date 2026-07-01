#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Splat2dConfig {
    pub image_size: usize,
    pub sigma: f32,
    pub lo: f32,
    pub hi: f32,
    pub normalize_color: bool,
    pub y_up: bool,
    pub pixel_size: f32,
}

impl Default for Splat2dConfig {
    fn default() -> Self {
        Self {
            image_size: 128,
            sigma: 1.0,
            lo: -1.0,
            hi: 1.0,
            normalize_color: true,
            y_up: true,
            pixel_size: 2.0 / 128.0,
        }
    }
}

pub fn splat_particles_2d(
    positions: &[[f32; 4]],
    colors: &[[f32; 3]],
    cfg: Splat2dConfig,
) -> Vec<[f32; 4]> {
    assert_eq!(positions.len(), colors.len());
    let size = cfg.image_size;
    let mut out = vec![[0.0; 4]; size * size];
    let mut sigma = cfg.sigma;
    if cfg.normalize_color {
        sigma = sigma * size as f32 * cfg.pixel_size / (cfg.hi - cfg.lo);
    }
    let radius = (5.0 * sigma).ceil().max(1.0) as isize;
    let norm_scale = if cfg.normalize_color {
        (size as f32 * cfg.pixel_size / (cfg.hi - cfg.lo)).powi(2)
    } else {
        1.0
    };

    for (pos, color) in positions.iter().zip(colors.iter()) {
        let px = (pos[0] - cfg.lo) / (cfg.hi - cfg.lo) * (size as f32 - 1.0);
        let mut py = (pos[1] - cfg.lo) / (cfg.hi - cfg.lo) * (size as f32 - 1.0);
        if cfg.y_up {
            py = (size as f32 - 1.0) - py;
        }
        let base_x = px.floor() as isize;
        let base_y = py.floor() as isize;
        let frac_x = px - base_x as f32;
        let frac_y = py - base_y as f32;

        let mut weights = Vec::new();
        let mut weight_sum = 0.0;
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                let x = base_x + ox;
                let y = base_y + oy;
                if x < 0 || y < 0 || x >= size as isize || y >= size as isize {
                    continue;
                }
                let dx = ox as f32 - frac_x;
                let dy = oy as f32 - frac_y;
                let w = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                weights.push((x as usize, y as usize, w));
                weight_sum += w;
            }
        }

        let denom = if cfg.normalize_color {
            weight_sum.max(1e-8)
        } else {
            1.0
        };
        for (x, y, w) in weights {
            let w = w / denom * norm_scale;
            let pixel = &mut out[y * size + x];
            pixel[0] += color[0] * w;
            pixel[1] += color[1] * w;
            pixel[2] += color[2] * w;
            pixel[3] += w;
        }
    }

    out
}
