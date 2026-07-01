#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gaussian3d {
    pub position_visibility: [f32; 4],
    pub spherical_harmonic: Vec<[f32; 3]>,
    pub rotation: [f32; 4],
    pub scale_opacity: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaussianDecodeConfig {
    pub sigma: f32,
    pub max_anisotropy: f32,
    pub opacity_scale: f32,
    pub sh_degree: usize,
}

impl Default for GaussianDecodeConfig {
    fn default() -> Self {
        Self {
            sigma: 0.02,
            max_anisotropy: 0.1,
            opacity_scale: 0.15,
            sh_degree: 1,
        }
    }
}

pub fn decode_gaussians_3d(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    cfg: GaussianDecodeConfig,
) -> Vec<Gaussian3d> {
    assert_eq!(positions.len() * state_dims, states.len());
    let sh_coeffs = (cfg.sh_degree + 1).pow(2);
    let color_dims = sh_coeffs * 3;
    let needed_tail = color_dims + 4 + 4;
    assert!(
        state_dims >= needed_tail,
        "state_dims must fit color/sh, rotation, and anisotropy/opacity tails"
    );

    positions
        .iter()
        .enumerate()
        .map(|(idx, position)| {
            let state = &states[idx * state_dims..(idx + 1) * state_dims];
            let color_start = state_dims - color_dims;
            let rot_start = color_start - 4;
            let anis_start = rot_start - 4;

            let mut spherical_harmonic = Vec::with_capacity(sh_coeffs);
            for coeff in 0..sh_coeffs {
                let base = color_start + coeff * 3;
                spherical_harmonic.push([
                    0.5 + 0.5 * state[base],
                    0.5 + 0.5 * state[base + 1],
                    0.5 + 0.5 * state[base + 2],
                ]);
            }

            let mut rotation = [
                state[rot_start],
                state[rot_start + 1],
                state[rot_start + 2],
                state[rot_start + 3],
            ];
            normalize_quat(&mut rotation);

            let ax = (state[anis_start] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let ay =
                (state[anis_start + 1] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let az =
                (state[anis_start + 2] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let opacity_log =
                (state[anis_start + 3] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;

            Gaussian3d {
                position_visibility: [position[0], position[1], position[2], 1.0],
                spherical_harmonic,
                rotation,
                scale_opacity: [
                    cfg.sigma * ax.exp(),
                    cfg.sigma * ay.exp(),
                    cfg.sigma * az.exp(),
                    cfg.opacity_scale * opacity_log.exp(),
                ],
            }
        })
        .collect()
}

fn normalize_quat(q: &mut [f32; 4]) {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm <= 1e-8 {
        *q = [1.0, 0.0, 0.0, 0.0];
    } else {
        for v in q.iter_mut() {
            *v /= norm;
        }
    }
}
