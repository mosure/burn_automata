use serde::{Deserialize, Serialize};

use super::unit_ball_measure;
use crate::{AutomataError, AutomataResult};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BudgetAllocation {
    pub lagrange_multiplier: f32,
    pub desired_footprint: Vec<f32>,
    pub expected_leaf_count: f32,
    pub clamped_min_fraction: f32,
    pub clamped_max_fraction: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn allocate_resolution_budget(
    error_density: &[f32],
    domain_measure: &[f32],
    dim: usize,
    error_exponent: f32,
    reference_footprint: f32,
    min_footprint: f32,
    max_footprint: f32,
    target_leaf_count: usize,
) -> AutomataResult<BudgetAllocation> {
    if error_density.is_empty() || error_density.len() != domain_measure.len() {
        return Err(AutomataError::InvalidArgument(
            "adaptive budget requires equal non-empty error/domain arrays".to_string(),
        ));
    }
    if !(dim == 2 || dim == 3 || dim == 4)
        || !error_exponent.is_finite()
        || error_exponent <= 0.0
        || !reference_footprint.is_finite()
        || !min_footprint.is_finite()
        || !max_footprint.is_finite()
        || min_footprint <= 0.0
        || reference_footprint < min_footprint
        || max_footprint < reference_footprint
        || target_leaf_count == 0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive budget dimensions, exponents, footprints, or target".to_string(),
        ));
    }
    if error_density
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || domain_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive budget inputs must be finite, with non-negative error and positive measure"
                .to_string(),
        ));
    }

    let unit_ball = unit_ball_measure(dim);
    let expected_count = |lambda: f32| {
        error_density
            .iter()
            .zip(domain_measure)
            .map(|(error, measure)| {
                let footprint =
                    pointwise_footprint(*error, lambda, dim, error_exponent, reference_footprint)
                        .clamp(min_footprint, max_footprint);
                measure / (unit_ball * footprint.powi(dim as i32))
            })
            .sum::<f32>()
    };

    let target = target_leaf_count as f32;
    let mut log_lo = -32.0_f32;
    let mut log_hi = 32.0_f32;
    for _ in 0..80 {
        let log_mid = 0.5 * (log_lo + log_hi);
        let count = expected_count(log_mid.exp());
        if count > target {
            log_lo = log_mid;
        } else {
            log_hi = log_mid;
        }
    }
    let lambda = (0.5 * (log_lo + log_hi)).exp();
    let desired_footprint = error_density
        .iter()
        .map(|error| {
            pointwise_footprint(*error, lambda, dim, error_exponent, reference_footprint)
                .clamp(min_footprint, max_footprint)
        })
        .collect::<Vec<_>>();
    let min_count = desired_footprint
        .iter()
        .filter(|value| (**value - min_footprint).abs() <= f32::EPSILON)
        .count();
    let max_count = desired_footprint
        .iter()
        .filter(|value| (**value - max_footprint).abs() <= f32::EPSILON)
        .count();
    let expected_leaf_count = domain_measure
        .iter()
        .zip(&desired_footprint)
        .map(|(measure, footprint)| measure / (unit_ball * footprint.powi(dim as i32)))
        .sum();
    Ok(BudgetAllocation {
        lagrange_multiplier: lambda,
        desired_footprint,
        expected_leaf_count,
        clamped_min_fraction: min_count as f32 / error_density.len() as f32,
        clamped_max_fraction: max_count as f32 / error_density.len() as f32,
    })
}

pub fn normalize_footprint_budget(
    proposed_footprint: &[f32],
    represented_measure: &[f32],
    dim: usize,
    min_footprint: f32,
    max_footprint: f32,
    target_leaf_count: usize,
) -> AutomataResult<BudgetAllocation> {
    if proposed_footprint.is_empty()
        || proposed_footprint.len() != represented_measure.len()
        || !(dim == 2 || dim == 3 || dim == 4)
        || !min_footprint.is_finite()
        || !max_footprint.is_finite()
        || min_footprint <= 0.0
        || max_footprint < min_footprint
        || target_leaf_count == 0
        || proposed_footprint
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || represented_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive proposed-footprint budget".to_string(),
        ));
    }
    let unit_ball = unit_ball_measure(dim);
    let expected_count = |log_scale: f32| {
        let scale = log_scale.exp();
        proposed_footprint
            .iter()
            .zip(represented_measure)
            .map(|(proposed, measure)| {
                let footprint = (proposed * scale).clamp(min_footprint, max_footprint);
                measure / (unit_ball * footprint.powi(dim as i32))
            })
            .sum::<f32>()
    };
    let target = target_leaf_count as f32;
    let mut log_lo = -32.0_f32;
    let mut log_hi = 32.0_f32;
    for _ in 0..80 {
        let log_mid = 0.5 * (log_lo + log_hi);
        if expected_count(log_mid) > target {
            log_lo = log_mid;
        } else {
            log_hi = log_mid;
        }
    }
    let log_scale = 0.5 * (log_lo + log_hi);
    let scale = log_scale.exp();
    let desired_footprint = proposed_footprint
        .iter()
        .map(|value| (value * scale).clamp(min_footprint, max_footprint))
        .collect::<Vec<_>>();
    let min_count = desired_footprint
        .iter()
        .filter(|value| (**value - min_footprint).abs() <= f32::EPSILON)
        .count();
    let max_count = desired_footprint
        .iter()
        .filter(|value| (**value - max_footprint).abs() <= f32::EPSILON)
        .count();
    Ok(BudgetAllocation {
        lagrange_multiplier: scale,
        expected_leaf_count: expected_count(log_scale),
        clamped_min_fraction: min_count as f32 / desired_footprint.len() as f32,
        clamped_max_fraction: max_count as f32 / desired_footprint.len() as f32,
        desired_footprint,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn normalize_footprint_budget_bounded(
    proposed_footprint: &[f32],
    current_footprint: &[f32],
    represented_measure: &[f32],
    dim: usize,
    min_footprint: f32,
    max_footprint: f32,
    target_leaf_count: usize,
    min_current_ratio: f32,
    max_current_ratio: f32,
) -> AutomataResult<BudgetAllocation> {
    if proposed_footprint.is_empty()
        || proposed_footprint.len() != current_footprint.len()
        || proposed_footprint.len() != represented_measure.len()
        || !(dim == 2 || dim == 3 || dim == 4)
        || !min_footprint.is_finite()
        || !max_footprint.is_finite()
        || !min_current_ratio.is_finite()
        || !max_current_ratio.is_finite()
        || min_footprint <= 0.0
        || max_footprint < min_footprint
        || min_current_ratio <= 0.0
        || min_current_ratio > 1.0
        || max_current_ratio < 1.0
        || target_leaf_count == 0
        || proposed_footprint
            .iter()
            .chain(current_footprint)
            .any(|value| !value.is_finite() || *value <= 0.0)
        || represented_measure
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AutomataError::InvalidArgument(
            "invalid bounded adaptive proposed-footprint budget".to_string(),
        ));
    }

    let row_bounds = current_footprint
        .iter()
        .map(|current| {
            (
                (current * min_current_ratio).max(min_footprint),
                (current * max_current_ratio).min(max_footprint),
            )
        })
        .collect::<Vec<_>>();
    if row_bounds
        .iter()
        .any(|(minimum, maximum)| minimum > maximum)
    {
        return Err(AutomataError::InvalidArgument(
            "current adaptive footprints lie outside the configured material range".to_string(),
        ));
    }

    let unit_ball = unit_ball_measure(dim);
    let expected_count = |log_scale: f32| {
        let scale = log_scale.exp();
        proposed_footprint
            .iter()
            .zip(represented_measure)
            .zip(&row_bounds)
            .map(|((proposed, measure), (minimum, maximum))| {
                let footprint = (proposed * scale).clamp(*minimum, *maximum);
                measure / (unit_ball * footprint.powi(dim as i32))
            })
            .sum::<f32>()
    };
    let target = target_leaf_count as f32;
    let mut log_lo = -32.0_f32;
    let mut log_hi = 32.0_f32;
    for _ in 0..80 {
        let log_mid = 0.5 * (log_lo + log_hi);
        if expected_count(log_mid) > target {
            log_lo = log_mid;
        } else {
            log_hi = log_mid;
        }
    }
    let log_scale = 0.5 * (log_lo + log_hi);
    let scale = log_scale.exp();
    let desired_footprint = proposed_footprint
        .iter()
        .zip(&row_bounds)
        .map(|(proposed, (minimum, maximum))| (proposed * scale).clamp(*minimum, *maximum))
        .collect::<Vec<_>>();
    let min_count = desired_footprint
        .iter()
        .filter(|value| (**value - min_footprint).abs() <= f32::EPSILON)
        .count();
    let max_count = desired_footprint
        .iter()
        .filter(|value| (**value - max_footprint).abs() <= f32::EPSILON)
        .count();
    Ok(BudgetAllocation {
        lagrange_multiplier: scale,
        expected_leaf_count: expected_count(log_scale),
        clamped_min_fraction: min_count as f32 / desired_footprint.len() as f32,
        clamped_max_fraction: max_count as f32 / desired_footprint.len() as f32,
        desired_footprint,
    })
}

pub fn boundary_protected_spacing(
    boundary_distance: f32,
    epsilon: f32,
    slope: f32,
    maximum: f32,
) -> f32 {
    (epsilon + slope * boundary_distance.max(0.0)).min(maximum)
}

fn pointwise_footprint(
    error: f32,
    lambda: f32,
    dim: usize,
    error_exponent: f32,
    reference: f32,
) -> f32 {
    let stabilized_error = error.max(1.0e-12);
    let ratio = dim as f32 * lambda / (error_exponent * stabilized_error);
    reference * ratio.powf(1.0 / (error_exponent + dim as f32))
}
