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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GaussianDecodeMode {
    ParticlePoint,
    #[default]
    GaussianSh0FixedScale,
    GaussianSh0LearnedScale,
    GaussianSh0Oriented,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaussianDecodeConfig {
    pub mode: GaussianDecodeMode,
    pub sigma: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub max_anisotropy: f32,
    pub opacity_scale: f32,
    pub min_opacity: f32,
    pub max_opacity: f32,
    pub sh_degree: usize,
}

impl Default for GaussianDecodeConfig {
    fn default() -> Self {
        Self {
            mode: GaussianDecodeMode::GaussianSh0FixedScale,
            sigma: 0.02,
            min_scale: 0.001,
            max_scale: 0.08,
            max_anisotropy: 0.1,
            opacity_scale: 0.15,
            min_opacity: 0.001,
            max_opacity: 0.95,
            sh_degree: 0,
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
    validate_decode_layout(state_dims, cfg);
    let sh_coeffs = decode_sh_coeffs(cfg);

    positions
        .iter()
        .enumerate()
        .map(|(idx, position)| {
            let state = &states[idx * state_dims..(idx + 1) * state_dims];
            let color_start = match cfg.mode {
                GaussianDecodeMode::GaussianSh0Oriented => state_dims - sh_coeffs * 3,
                _ => state_dims - 3,
            };
            let mut spherical_harmonic = decode_sh(state, color_start, sh_coeffs, cfg);
            if cfg.mode != GaussianDecodeMode::GaussianSh0Oriented {
                spherical_harmonic.truncate(1);
            }
            let rotation = match cfg.mode {
                GaussianDecodeMode::GaussianSh0Oriented => {
                    let rot_start = color_start - 4;
                    let mut rotation = [
                        state[rot_start],
                        state[rot_start + 1],
                        state[rot_start + 2],
                        state[rot_start + 3],
                    ];
                    normalize_quat(&mut rotation);
                    rotation
                }
                _ => [1.0, 0.0, 0.0, 0.0],
            };
            let scale_opacity = decode_scale_opacity(state, state_dims, cfg);

            Gaussian3d {
                position_visibility: [position[0], position[1], position[2], 1.0],
                spherical_harmonic,
                rotation,
                scale_opacity,
            }
        })
        .collect()
}

fn validate_decode_layout(state_dims: usize, cfg: GaussianDecodeConfig) {
    match cfg.mode {
        GaussianDecodeMode::ParticlePoint | GaussianDecodeMode::GaussianSh0FixedScale => {
            assert!(
                state_dims >= 3,
                "state_dims must include an RGB tail for SH0 gaussian decoding"
            );
        }
        GaussianDecodeMode::GaussianSh0LearnedScale => {
            assert!(
                state_dims >= 5,
                "state_dims must include scale, opacity, and RGB tail channels"
            );
        }
        GaussianDecodeMode::GaussianSh0Oriented => {
            let sh_coeffs = decode_sh_coeffs(cfg);
            let needed_tail = sh_coeffs * 3 + 4 + 4;
            assert!(
                state_dims >= needed_tail,
                "state_dims must fit color/sh, rotation, and anisotropy/opacity tails"
            );
        }
    }
}

fn decode_sh_coeffs(cfg: GaussianDecodeConfig) -> usize {
    match cfg.mode {
        GaussianDecodeMode::GaussianSh0Oriented => (cfg.sh_degree + 1).pow(2),
        _ => 1,
    }
}

fn decode_sh(
    state: &[f32],
    color_start: usize,
    sh_coeffs: usize,
    cfg: GaussianDecodeConfig,
) -> Vec<[f32; 3]> {
    let mut spherical_harmonic = Vec::with_capacity(sh_coeffs);
    for coeff in 0..sh_coeffs {
        let base = color_start + coeff * 3;
        let color = match cfg.mode {
            GaussianDecodeMode::GaussianSh0Oriented => [
                (state[base] + 0.5).clamp(0.0, 1.0),
                (state[base + 1] + 0.5).clamp(0.0, 1.0),
                (state[base + 2] + 0.5).clamp(0.0, 1.0),
            ],
            _ => [
                (state[base] + 0.5).clamp(0.0, 1.0),
                (state[base + 1] + 0.5).clamp(0.0, 1.0),
                (state[base + 2] + 0.5).clamp(0.0, 1.0),
            ],
        };
        spherical_harmonic.push(color);
    }
    spherical_harmonic
}

fn decode_scale_opacity(state: &[f32], state_dims: usize, cfg: GaussianDecodeConfig) -> [f32; 4] {
    match cfg.mode {
        GaussianDecodeMode::ParticlePoint => {
            let scale = cfg.min_scale.max(1.0e-6);
            [scale, scale, scale, 1.0]
        }
        GaussianDecodeMode::GaussianSh0FixedScale => {
            let scale = cfg.sigma.clamp(cfg.min_scale, cfg.max_scale);
            [
                scale,
                scale,
                scale,
                cfg.opacity_scale.clamp(cfg.min_opacity, cfg.max_opacity),
            ]
        }
        GaussianDecodeMode::GaussianSh0LearnedScale => {
            let scale_logit = state[state_dims - 5].clamp(-8.0, 8.0);
            let opacity_logit = state[state_dims - 4].clamp(-12.0, 12.0);
            let scale = (cfg.sigma * scale_logit.exp()).clamp(cfg.min_scale, cfg.max_scale);
            let opacity =
                sigmoid(opacity_logit).clamp(cfg.min_opacity, cfg.max_opacity) * cfg.opacity_scale;
            [
                scale,
                scale,
                scale,
                opacity.clamp(cfg.min_opacity, cfg.max_opacity),
            ]
        }
        GaussianDecodeMode::GaussianSh0Oriented => {
            let sh_coeffs = decode_sh_coeffs(cfg);
            let color_start = state_dims - sh_coeffs * 3;
            let rot_start = color_start - 4;
            let anis_start = rot_start - 4;
            let ax = (state[anis_start] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let ay =
                (state[anis_start + 1] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let az =
                (state[anis_start + 2] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            let opacity_log =
                (state[anis_start + 3] / cfg.max_anisotropy.max(1e-6)).tanh() * cfg.max_anisotropy;
            [
                (cfg.sigma * ax.exp()).clamp(cfg.min_scale, cfg.max_scale),
                (cfg.sigma * ay.exp()).clamp(cfg.min_scale, cfg.max_scale),
                (cfg.sigma * az.exp()).clamp(cfg.min_scale, cfg.max_scale),
                (cfg.opacity_scale * opacity_log.exp()).clamp(cfg.min_opacity, cfg.max_opacity),
            ]
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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
