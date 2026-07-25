use serde::{Deserialize, Serialize};

use super::{AdaptiveNpaModel, unit_ball_measure};
use crate::{AutomataError, AutomataResult};

/// Rendering semantics used by adaptive quality experiments and restriction
/// label generation. Only [`Self::IsotropicMaterialGaussian`] is deployable;
/// the remaining variants are bounded diagnostic controls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveRenderDecoder {
    /// One isotropic Gaussian per visible material leaf. Represented measure
    /// determines its scalar target footprint; covariance stays internal.
    #[default]
    IsotropicMaterialGaussian,
    /// Evaluator-only covariance counterfactual using mean intensive state.
    MomentGaussian,
    /// Evaluator-only covariance counterfactual with affine color detail.
    AffineMomentGaussian,
    /// Evaluator-only compact covariance counterfactual.
    CompactMomentGaussian,
    /// Evaluator-only bounded quadrature reconstructed from moments.
    CanonicalAffineQuadrature,
    /// Diagnostic full-fine render ceiling; not a visible adaptive budget.
    RetainedFineQuadrature,
    /// Diagnostic persistent-mode render ceiling; not a material-leaf budget.
    PersistentModeQuadrature,
}

impl AdaptiveRenderDecoder {
    pub const fn is_deployable(self) -> bool {
        matches!(self, Self::IsotropicMaterialGaussian)
    }

    pub const fn supports_restriction_labels(self) -> bool {
        matches!(
            self,
            Self::IsotropicMaterialGaussian | Self::CompactMomentGaussian
        )
    }
}

/// Renderer-neutral Gaussian geometry derived from conservative material moments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveGaussianGeometry {
    pub scale: [f32; 3],
    /// Unit quaternion in `[w, x, y, z]` order.
    pub rotation: [f32; 4],
    /// Compensates covariance volume so integrated opacity follows represented measure.
    pub opacity: f32,
}

/// Builds the canonical isotropic render primitive for one material leaf.
///
/// Represented measure is authoritative. `render_footprint` may temporarily
/// differ from the equal-measure radius during a topology transition, so
/// opacity compensates for that display-only change. Conservative covariance
/// remains simulation metadata and cannot introduce an anisotropic rendering
/// degree of freedom on this path.
pub fn adaptive_isotropic_gaussian_geometry(
    represented_measure: f32,
    render_footprint: f32,
    spatial_dims: usize,
) -> AutomataResult<AdaptiveGaussianGeometry> {
    if !(spatial_dims == 2 || spatial_dims == 3)
        || !represented_measure.is_finite()
        || represented_measure <= 0.0
        || !render_footprint.is_finite()
        || render_footprint <= 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive isotropic Gaussian geometry requires 2D/3D positive finite material scale"
                .to_string(),
        ));
    }
    let rendered_measure =
        unit_ball_measure(spatial_dims) * render_footprint.powi(spatial_dims as i32);
    Ok(AdaptiveGaussianGeometry {
        scale: [render_footprint; 3],
        rotation: [1.0, 0.0, 0.0, 0.0],
        opacity: (represented_measure / rendered_measure.max(f32::MIN_POSITIVE)).clamp(0.001, 1.0),
    })
}

pub fn adaptive_display_scale_per_footprint(model: &AdaptiveNpaModel) -> f32 {
    let fixed_npa_display_scale = (model.rule.config.eps0 * 0.12).max(0.00008);
    fixed_npa_display_scale / model.config.base_rule_footprint()
}

/// Evaluator-only covariance decoder retained as a counterfactual control.
///
/// Runtime adaptive NPA rendering must use
/// [`adaptive_isotropic_gaussian_geometry`]. Covariance is conservative
/// simulation state and is not a learned or deployable 3DGS shape channel.
pub(crate) fn diagnostic_covariance_gaussian_geometry(
    represented_measure: f32,
    covariance: [f32; 9],
    spatial_dims: usize,
) -> AutomataResult<AdaptiveGaussianGeometry> {
    if !(spatial_dims == 2 || spatial_dims == 3)
        || !represented_measure.is_finite()
        || represented_measure <= 0.0
        || covariance.iter().any(|value| !value.is_finite())
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive Gaussian geometry requires 2D/3D finite positive material moments"
                .to_string(),
        ));
    }
    let (scale, rotation) = covariance_transform(covariance, spatial_dims);
    let scale_product = scale[..spatial_dims]
        .iter()
        .copied()
        .product::<f32>()
        .max(f32::MIN_POSITIVE);
    let calibrated_measure = unit_ball_measure(spatial_dims) * scale_product;
    Ok(AdaptiveGaussianGeometry {
        scale,
        rotation,
        opacity: (represented_measure / calibrated_measure).clamp(0.001, 1.0),
    })
}

fn covariance_transform(covariance: [f32; 9], spatial_dims: usize) -> ([f32; 3], [f32; 4]) {
    if spatial_dims == 2 {
        let a = covariance[0].max(1.0e-12);
        let b = 0.5 * (covariance[1] + covariance[3]);
        let d = covariance[4].max(1.0e-12);
        let radius = (((a - d) * 0.5).powi(2) + b * b).sqrt();
        let center = 0.5 * (a + d);
        let major = (center + radius).max(1.0e-12);
        let minor = (center - radius).max(1.0e-12);
        let angle = 0.5 * (2.0 * b).atan2(a - d);
        let half = 0.5 * angle;
        return (
            [2.0 * major.sqrt(), 2.0 * minor.sqrt(), 2.0 * minor.sqrt()],
            [half.cos(), 0.0, 0.0, half.sin()],
        );
    }

    let (eigenvalues, rotation_matrix) = symmetric_eigen_3d(covariance);
    (
        [
            2.0 * eigenvalues[0].max(1.0e-12).sqrt(),
            2.0 * eigenvalues[1].max(1.0e-12).sqrt(),
            2.0 * eigenvalues[2].max(1.0e-12).sqrt(),
        ],
        rotation_matrix_to_quaternion(rotation_matrix),
    )
}

fn symmetric_eigen_3d(mut matrix: [f32; 9]) -> ([f32; 3], [f32; 9]) {
    matrix[1] = 0.5 * (matrix[1] + matrix[3]);
    matrix[3] = matrix[1];
    matrix[2] = 0.5 * (matrix[2] + matrix[6]);
    matrix[6] = matrix[2];
    matrix[5] = 0.5 * (matrix[5] + matrix[7]);
    matrix[7] = matrix[5];
    let mut vectors = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..8 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = matrix[p * 3 + q];
            if apq.abs() <= 1.0e-10 {
                continue;
            }
            let angle = 0.5 * (2.0 * apq).atan2(matrix[q * 3 + q] - matrix[p * 3 + p]);
            let (sin, cos) = angle.sin_cos();
            for row in 0..3 {
                let rp = matrix[row * 3 + p];
                let rq = matrix[row * 3 + q];
                matrix[row * 3 + p] = cos * rp - sin * rq;
                matrix[row * 3 + q] = sin * rp + cos * rq;
            }
            for col in 0..3 {
                let pc = matrix[p * 3 + col];
                let qc = matrix[q * 3 + col];
                matrix[p * 3 + col] = cos * pc - sin * qc;
                matrix[q * 3 + col] = sin * pc + cos * qc;
            }
            for row in 0..3 {
                let vp = vectors[row * 3 + p];
                let vq = vectors[row * 3 + q];
                vectors[row * 3 + p] = cos * vp - sin * vq;
                vectors[row * 3 + q] = sin * vp + cos * vq;
            }
        }
    }
    let mut order = [0_usize, 1, 2];
    order.sort_by(|lhs, rhs| matrix[*rhs * 3 + *rhs].total_cmp(&matrix[*lhs * 3 + *lhs]));
    let values = order.map(|index| matrix[index * 3 + index]);
    let mut sorted = [0.0; 9];
    for (column, source) in order.into_iter().enumerate() {
        for row in 0..3 {
            sorted[row * 3 + column] = vectors[row * 3 + source];
        }
    }
    if determinant_3d(sorted) < 0.0 {
        for row in 0..3 {
            sorted[row * 3 + 2] = -sorted[row * 3 + 2];
        }
    }
    (values, sorted)
}

fn determinant_3d(matrix: [f32; 9]) -> f32 {
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
}

fn rotation_matrix_to_quaternion(matrix: [f32; 9]) -> [f32; 4] {
    let trace = matrix[0] + matrix[4] + matrix[8];
    let (w, x, y, z) = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * scale,
            (matrix[7] - matrix[5]) / scale,
            (matrix[2] - matrix[6]) / scale,
            (matrix[3] - matrix[1]) / scale,
        )
    } else if matrix[0] > matrix[4] && matrix[0] > matrix[8] {
        let scale = (1.0 + matrix[0] - matrix[4] - matrix[8]).sqrt() * 2.0;
        (
            (matrix[7] - matrix[5]) / scale,
            0.25 * scale,
            (matrix[1] + matrix[3]) / scale,
            (matrix[2] + matrix[6]) / scale,
        )
    } else if matrix[4] > matrix[8] {
        let scale = (1.0 + matrix[4] - matrix[0] - matrix[8]).sqrt() * 2.0;
        (
            (matrix[2] - matrix[6]) / scale,
            (matrix[1] + matrix[3]) / scale,
            0.25 * scale,
            (matrix[5] + matrix[7]) / scale,
        )
    } else {
        let scale = (1.0 + matrix[8] - matrix[0] - matrix[4]).sqrt() * 2.0;
        (
            (matrix[3] - matrix[1]) / scale,
            (matrix[2] + matrix[6]) / scale,
            (matrix[5] + matrix[7]) / scale,
            0.25 * scale,
        )
    };
    let norm = (w * w + x * x + y * y + z * z)
        .sqrt()
        .max(f32::MIN_POSITIVE);
    [w / norm, x / norm, y / norm, z / norm]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_equal_measure_maps_to_material_footprint() {
        let radius = 0.0125_f32;
        let variance = (0.5 * radius).powi(2);
        let geometry = diagnostic_covariance_gaussian_geometry(
            std::f32::consts::PI * radius.powi(2),
            [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap();
        assert_eq!(geometry.scale, [radius; 3]);
        assert!((geometry.opacity - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn isotropic_geometry_ignores_covariance_and_preserves_measure() {
        let represented_measure = 0.001_f32;
        let footprint = (represented_measure / std::f32::consts::PI).sqrt() * 1.5;
        let geometry =
            adaptive_isotropic_gaussian_geometry(represented_measure, footprint, 2).unwrap();
        assert_eq!(geometry.scale, [footprint; 3]);
        assert_eq!(geometry.rotation, [1.0, 0.0, 0.0, 0.0]);
        let reconstructed =
            std::f32::consts::PI * geometry.scale[0] * geometry.scale[1] * geometry.opacity;
        assert!((reconstructed - represented_measure).abs() < 1.0e-6);
    }

    #[test]
    fn anisotropic_moment_maps_to_rotated_measure_preserving_gaussian() {
        let represented_measure = 0.001_f32;
        let geometry = diagnostic_covariance_gaussian_geometry(
            represented_measure,
            [0.0004, 0.00015, 0.0, 0.00015, 0.0002, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap();
        assert!(geometry.scale[0] > geometry.scale[1]);
        assert!(geometry.rotation[3].abs() > 1.0e-4);
        let reconstructed =
            std::f32::consts::PI * geometry.scale[0] * geometry.scale[1] * geometry.opacity;
        assert!((reconstructed - represented_measure).abs() < 1.0e-6);
    }
}
