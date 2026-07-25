use serde::{Deserialize, Serialize};

use crate::{AutomataError, AutomataResult};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalMaterial {
    pub represented_measure: f64,
    pub position: Vec<f64>,
    /// Row-major covariance with shape q x q.
    pub covariance: Vec<f64>,
    pub extensive: Vec<f64>,
}

impl CanonicalMaterial {
    pub fn validate(&self) -> AutomataResult<()> {
        let dim = self.position.len();
        if dim == 0 || dim > 4 || self.covariance.len() != dim * dim {
            return Err(AutomataError::InvalidArgument(
                "canonical material requires q in 1..=4 and q x q covariance".to_string(),
            ));
        }
        if !self.represented_measure.is_finite() || self.represented_measure <= 0.0 {
            return Err(AutomataError::InvalidArgument(
                "canonical material measure must be finite and positive".to_string(),
            ));
        }
        if self
            .position
            .iter()
            .chain(&self.covariance)
            .chain(&self.extensive)
            .any(|value| !value.is_finite())
        {
            return Err(AutomataError::InvalidArgument(
                "canonical material contains non-finite values".to_string(),
            ));
        }
        cholesky(&self.covariance, dim).ok_or_else(|| {
            AutomataError::InvalidArgument("canonical material covariance is not SPD".to_string())
        })?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TopologyAudit {
    pub measure_relative_error: f64,
    pub centroid_l2_error: f64,
    pub second_moment_relative_error: f64,
    pub extensive_relative_error: f64,
    pub child_spd: bool,
    pub determinant_scale_relative_error: f64,
}

pub fn canonical_split(parent: &CanonicalMaterial) -> AutomataResult<Vec<CanonicalMaterial>> {
    parent.validate()?;
    let dim = parent.position.len();
    let child_count = 2 * dim;
    let covariance_scale = (child_count as f64).powf(-2.0 / dim as f64);
    let offset_scale = (dim as f64 * (1.0 - covariance_scale)).sqrt();
    let factor = cholesky(&parent.covariance, dim).expect("validated SPD covariance");
    let mut children = Vec::with_capacity(child_count);
    for axis in 0..dim {
        for sign in [-1.0_f64, 1.0] {
            let mut position = parent.position.clone();
            for row in 0..dim {
                position[row] += sign * offset_scale * factor[row * dim + axis];
            }
            children.push(CanonicalMaterial {
                represented_measure: parent.represented_measure / child_count as f64,
                position,
                covariance: parent
                    .covariance
                    .iter()
                    .map(|value| value * covariance_scale)
                    .collect(),
                extensive: parent
                    .extensive
                    .iter()
                    .map(|value| value / child_count as f64)
                    .collect(),
            });
        }
    }
    Ok(children)
}

/// Splits material into unequal, continuously weighted children while
/// preserving measure, centroid, second moment, and footprint calibration.
///
/// The event keeps the canonical `2q` child count. Child `c` receives measure
/// fraction `alpha_c` and covariance `alpha_c^(2/q) * Sigma_parent`, so its
/// determinant-derived footprint scales as `alpha_c^(1/q)`. Deterministic
/// weighted sigma points supply the remaining covariance. Equal fractions use
/// [`canonical_split`] exactly, preserving existing artifact behavior.
pub fn constrained_unequal_split(
    parent: &CanonicalMaterial,
    child_fractions: &[f64],
) -> AutomataResult<Vec<CanonicalMaterial>> {
    parent.validate()?;
    let dim = parent.position.len();
    let child_count = 2 * dim;
    if child_fractions.len() != child_count
        || child_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || *fraction <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(format!(
            "constrained unequal split requires {child_count} positive finite fractions"
        )));
    }
    let fraction_sum = child_fractions.iter().sum::<f64>();
    if (fraction_sum - 1.0).abs() > 1.0e-10 {
        return Err(AutomataError::InvalidArgument(format!(
            "constrained unequal split fractions must sum to one, got {fraction_sum}"
        )));
    }
    let equal_fraction = 1.0 / child_count as f64;
    if child_fractions
        .iter()
        .all(|fraction| (*fraction - equal_fraction).abs() <= 1.0e-14)
    {
        return canonical_split(parent);
    }

    let covariance_scales = child_fractions
        .iter()
        .map(|fraction| fraction.powf(2.0 / dim as f64))
        .collect::<Vec<_>>();
    let offset_covariance_scale = 1.0
        - child_fractions
            .iter()
            .zip(&covariance_scales)
            .map(|(fraction, covariance_scale)| fraction * covariance_scale)
            .sum::<f64>();
    if !offset_covariance_scale.is_finite() || offset_covariance_scale <= 1.0e-14 {
        return Err(AutomataError::InvalidArgument(
            "constrained unequal split has no positive offset covariance".to_string(),
        ));
    }

    let sigma_points = weighted_whitened_canonical_points(child_fractions, dim)?;
    let factor = cholesky(&parent.covariance, dim).expect("validated SPD covariance");
    let offset_scale = offset_covariance_scale.sqrt();
    let mut children = Vec::with_capacity(child_count);
    for child in 0..child_count {
        let fraction = child_fractions[child];
        let mut normalized_offset = vec![0.0; dim];
        for axis in 0..dim {
            normalized_offset[axis] = offset_scale * sigma_points[child * dim + axis];
        }
        let mut position = parent.position.clone();
        for row in 0..dim {
            position[row] += (0..dim)
                .map(|col| factor[row * dim + col] * normalized_offset[col])
                .sum::<f64>();
        }
        children.push(CanonicalMaterial {
            represented_measure: parent.represented_measure * fraction,
            position,
            covariance: parent
                .covariance
                .iter()
                .map(|value| value * covariance_scales[child])
                .collect(),
            extensive: parent
                .extensive
                .iter()
                .map(|value| value * fraction)
                .collect(),
        });
    }
    if children.iter().any(|child| child.validate().is_err()) {
        return Err(AutomataError::InvalidModel(
            "constrained unequal split produced invalid child material".to_string(),
        ));
    }
    Ok(children)
}

/// Centers and whitens the canonical +/-axis sigma points under arbitrary
/// positive weights. At equal weights the weighted mean is zero and the
/// covariance is identity, so this returns the canonical points exactly.
/// Nearby weights therefore produce nearby geometry instead of changing to an
/// unrelated orthonormal frame.
fn weighted_whitened_canonical_points(weights: &[f64], dim: usize) -> AutomataResult<Vec<f64>> {
    let count = weights.len();
    let mut points = vec![0.0_f64; count * dim];
    let canonical_radius = (dim as f64).sqrt();
    for axis in 0..dim {
        points[(2 * axis) * dim + axis] = -canonical_radius;
        points[(2 * axis + 1) * dim + axis] = canonical_radius;
    }
    let mut mean = vec![0.0_f64; dim];
    for child in 0..count {
        for axis in 0..dim {
            mean[axis] += weights[child] * points[child * dim + axis];
        }
    }
    let mut covariance = vec![0.0_f64; dim * dim];
    for child in 0..count {
        for row in 0..dim {
            let row_value = points[child * dim + row] - mean[row];
            for col in 0..dim {
                covariance[row * dim + col] +=
                    weights[child] * row_value * (points[child * dim + col] - mean[col]);
            }
        }
    }
    let factor = cholesky(&covariance, dim).ok_or_else(|| {
        AutomataError::InvalidModel(
            "constrained unequal split canonical point covariance is not SPD".to_string(),
        )
    })?;
    let mut whitened = vec![0.0_f64; count * dim];
    for child in 0..count {
        for row in 0..dim {
            let prior = (0..row)
                .map(|col| factor[row * dim + col] * whitened[child * dim + col])
                .sum::<f64>();
            whitened[child * dim + row] =
                (points[child * dim + row] - mean[row] - prior) / factor[row * dim + row];
        }
    }
    if whitened.iter().any(|value| !value.is_finite()) {
        return Err(AutomataError::InvalidModel(
            "constrained unequal split produced non-finite whitened sigma points".to_string(),
        ));
    }
    Ok(whitened)
}

pub fn canonical_merge(children: &[CanonicalMaterial]) -> AutomataResult<CanonicalMaterial> {
    let first = children.first().ok_or_else(|| {
        AutomataError::InvalidArgument("canonical merge requires children".to_string())
    })?;
    let dim = first.position.len();
    let extensive_dims = first.extensive.len();
    if children.iter().any(|child| {
        child.position.len() != dim
            || child.covariance.len() != dim * dim
            || child.extensive.len() != extensive_dims
            || child.validate().is_err()
    }) {
        return Err(AutomataError::InvalidArgument(
            "canonical merge children have incompatible shapes or invalid values".to_string(),
        ));
    }
    let represented_measure = children
        .iter()
        .map(|child| child.represented_measure)
        .sum::<f64>();
    let mut position = vec![0.0; dim];
    for child in children {
        for (axis, value) in position.iter_mut().enumerate() {
            *value += child.represented_measure * child.position[axis];
        }
    }
    position
        .iter_mut()
        .for_each(|value| *value /= represented_measure);

    let mut covariance = vec![0.0; dim * dim];
    let mut extensive = vec![0.0; extensive_dims];
    for child in children {
        for row in 0..dim {
            let row_delta = child.position[row] - position[row];
            for col in 0..dim {
                let col_delta = child.position[col] - position[col];
                covariance[row * dim + col] += child.represented_measure
                    * (child.covariance[row * dim + col] + row_delta * col_delta);
            }
        }
        for (index, value) in child.extensive.iter().enumerate() {
            extensive[index] += value;
        }
    }
    covariance
        .iter_mut()
        .for_each(|value| *value /= represented_measure);
    let parent = CanonicalMaterial {
        represented_measure,
        position,
        covariance,
        extensive,
    };
    parent.validate()?;
    Ok(parent)
}

pub fn topology_audit(
    parent: &CanonicalMaterial,
    children: &[CanonicalMaterial],
) -> AutomataResult<TopologyAudit> {
    parent.validate()?;
    let merged = canonical_merge(children)?;
    let dim = parent.position.len();
    let measure_relative_error =
        relative_error(merged.represented_measure, parent.represented_measure);
    let centroid_l2_error = merged
        .position
        .iter()
        .zip(&parent.position)
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let parent_second = uncentered_second_moment(parent);
    let merged_second = uncentered_second_moment(&merged);
    let second_moment_relative_error = vector_relative_error(&merged_second, &parent_second);
    let extensive_relative_error = vector_relative_error(&merged.extensive, &parent.extensive);
    let child_spd = children.iter().all(|child| child.validate().is_ok());
    let parent_det = determinant(&parent.covariance, dim);
    let determinant_scale_relative_error = children
        .iter()
        .map(|child| {
            let fraction = child.represented_measure / parent.represented_measure;
            let child_det = determinant(&child.covariance, dim);
            relative_error(child_det / parent_det, fraction * fraction)
        })
        .fold(0.0_f64, f64::max);
    Ok(TopologyAudit {
        measure_relative_error,
        centroid_l2_error,
        second_moment_relative_error,
        extensive_relative_error,
        child_spd,
        determinant_scale_relative_error,
    })
}

fn uncentered_second_moment(material: &CanonicalMaterial) -> Vec<f64> {
    let dim = material.position.len();
    let mut moment = vec![0.0; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            moment[row * dim + col] = material.represented_measure
                * (material.covariance[row * dim + col]
                    + material.position[row] * material.position[col]);
        }
    }
    moment
}

fn cholesky(matrix: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut factor = vec![0.0; dim * dim];
    for row in 0..dim {
        for col in 0..=row {
            let mut sum = matrix[row * dim + col];
            for k in 0..col {
                sum -= factor[row * dim + k] * factor[col * dim + k];
            }
            if row == col {
                if !sum.is_finite() || sum <= 1.0e-14 {
                    return None;
                }
                factor[row * dim + col] = sum.sqrt();
            } else {
                factor[row * dim + col] = sum / factor[col * dim + col];
            }
        }
    }
    Some(factor)
}

fn determinant(matrix: &[f64], dim: usize) -> f64 {
    let Some(factor) = cholesky(matrix, dim) else {
        return f64::NAN;
    };
    (0..dim)
        .map(|axis| factor[axis * dim + axis].powi(2))
        .product()
}

fn relative_error(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(1.0e-15)
}

fn vector_relative_error(value: &[f64], reference: &[f64]) -> f64 {
    let numerator = value
        .iter()
        .zip(reference)
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0e-15);
    numerator / denominator
}
