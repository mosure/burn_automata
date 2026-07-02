use super::*;

pub(crate) fn safe_inverse_symmetric(matrix: &[f32], dim: usize) -> [f32; 9] {
    const TOL: f32 = 1e-3;
    let mut out = [0.0; 9];
    if dim == 2 {
        let a = matrix[0];
        let b = matrix[1];
        let d = matrix[3];
        let det = a * d - b * b;
        if det.abs() < TOL {
            out[0] = 1.0;
            out[3] = 1.0;
            return out;
        }
        let inv_det = det.recip();
        out[0] = d * inv_det;
        out[1] = -b * inv_det;
        out[2] = -b * inv_det;
        out[3] = a * inv_det;
        return out;
    }

    let a = matrix[0];
    let b = matrix[1];
    let c = matrix[2];
    let d = matrix[4];
    let e = matrix[5];
    let f = matrix[8];
    let t1 = d * f - e * e;
    let t2 = c * e - b * f;
    let t3 = b * e - c * d;
    let det = a * t1 + b * t2 + c * t3;
    if det.abs() < TOL {
        out[0] = 1.0;
        out[4] = 1.0;
        out[8] = 1.0;
        return out;
    }
    let inv_det = det.recip();
    out[0] = t1 * inv_det;
    out[1] = t2 * inv_det;
    out[2] = t3 * inv_det;
    out[3] = t2 * inv_det;
    out[4] = (a * f - c * c) * inv_det;
    out[5] = (b * c - a * e) * inv_det;
    out[6] = t3 * inv_det;
    out[7] = (b * c - a * e) * inv_det;
    out[8] = (a * d - b * b) * inv_det;
    out
}

pub(crate) fn smoothing_poly6_kernel(r2: f32, cfg: &HashGridConfig) -> f32 {
    let eps2 = cfg.eps * cfg.eps;
    if r2 >= eps2 {
        return 0.0;
    }
    let x = eps2 - r2;
    smoothing_poly6_normalization(cfg) * x * x * x
}

pub(crate) fn smoothing_poly6_delta_adjoint(
    delta: [f32; 4],
    r2: f32,
    cfg: &HashGridConfig,
    output_adjoint: f32,
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 >= eps2 {
        return out;
    }
    let x = eps2 - r2;
    let dkernel_dr2 = -3.0 * smoothing_poly6_normalization(cfg) * x * x;
    for axis in 0..cfg.dim {
        out[axis] = output_adjoint * dkernel_dr2 * 2.0 * delta[axis];
    }
    out
}

pub(crate) fn spiky_gradient_kernel(
    delta: [f32; 4],
    r2: f32,
    cfg: &HashGridConfig,
    coeff: f32,
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 <= 0.0 || r2 >= eps2 {
        return out;
    }
    let r = r2.sqrt();
    let mag = coeff * gradient_spiky_normalization(cfg) * 3.0 * (cfg.eps - r).powi(2) / r;
    for axis in 0..cfg.dim {
        out[axis] = mag * delta[axis];
    }
    out
}

pub(crate) fn spiky_gradient_delta_adjoint(
    delta: [f32; 4],
    r2: f32,
    cfg: &HashGridConfig,
    coeff: f32,
    output_adjoint: &[f32],
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let eps2 = cfg.eps * cfg.eps;
    if r2 <= 0.0 || r2 >= eps2 {
        return out;
    }
    let r = r2.sqrt();
    let norm = coeff * gradient_spiky_normalization(cfg) * 3.0;
    let scale = norm * (cfg.eps - r).powi(2) / r;
    let dscale_dr = norm * (1.0 - (cfg.eps * cfg.eps) / r2);
    let dot = output_adjoint
        .iter()
        .take(cfg.dim)
        .zip(delta.iter())
        .map(|(adjoint, value)| adjoint * value)
        .sum::<f32>();
    for axis in 0..cfg.dim {
        out[axis] = scale * output_adjoint[axis] + dscale_dr * dot * delta[axis] / r;
    }
    out
}

pub(crate) fn smoothing_poly6_normalization(cfg: &HashGridConfig) -> f32 {
    if cfg.dim == 2 {
        4.0 / (std::f32::consts::PI * cfg.eps.powi(8))
    } else {
        315.0 / (64.0 * std::f32::consts::PI * cfg.eps.powi(9))
    }
}

pub(crate) fn gradient_spiky_normalization(cfg: &HashGridConfig) -> f32 {
    if cfg.dim == 2 {
        10.0 / (std::f32::consts::PI * cfg.eps.powi(5))
    } else {
        15.0 / (std::f32::consts::PI * cfg.eps.powi(6))
    }
}

pub(crate) trait RecipFinite {
    fn recip_finite(self) -> Self;
}

impl RecipFinite for f32 {
    fn recip_finite(self) -> Self {
        if self.abs() <= 1e-20 || !self.is_finite() {
            0.0
        } else {
            self.recip()
        }
    }
}
