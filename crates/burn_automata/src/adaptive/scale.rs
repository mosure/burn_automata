use std::collections::BTreeSet;

use super::{
    AdaptiveParticleSet, CanonicalMaterial, canonical_split, constrained_unequal_split,
    material_footprint_radius,
};
use crate::{AutomataError, AutomataResult};

#[derive(Clone, Debug)]
pub(crate) struct ContinuousSplitPlan {
    pub fractions: Vec<f64>,
    pub desired_footprints: Vec<f32>,
    pub children: Vec<CanonicalMaterial>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MaterialScaleMetrics {
    pub occupied_sixty_fourth_octave_bins: usize,
    pub fractional_octave_fraction: f32,
    pub dyadic_quantization_rmse_octaves: f32,
}

pub(crate) fn continuous_split_plan(
    parent_index: usize,
    parent: &CanonicalMaterial,
    particles: &AdaptiveParticleSet,
    desired_footprint: &[f32],
    max_measure_ratio: f32,
    neighbor_count: usize,
) -> AutomataResult<ContinuousSplitPlan> {
    if parent_index >= particles.len()
        || desired_footprint.len() != particles.len()
        || !max_measure_ratio.is_finite()
        || max_measure_ratio < 1.0
        || neighbor_count == 0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid continuous split-plan inputs".to_string(),
        ));
    }
    let probes = canonical_split(parent)?;
    let child_count = probes.len();
    let equal = 1.0 / child_count as f64;
    let fractions = if max_measure_ratio <= 1.0 + f32::EPSILON {
        vec![equal; child_count]
    } else {
        let gradient = desired_log_footprint_gradient(
            parent_index,
            particles,
            desired_footprint,
            neighbor_count,
        );
        let parent_log = desired_footprint[parent_index].max(f32::MIN_POSITIVE).ln();
        let log_target_measure = probes
            .iter()
            .map(|probe| {
                let log_footprint = parent_log
                    + (0..particles.spatial_dims)
                        .map(|axis| {
                            gradient[axis]
                                * (probe.position[axis] as f32
                                    - particles.positions[parent_index][axis])
                        })
                        .sum::<f32>();
                particles.spatial_dims as f64 * f64::from(log_footprint)
            })
            .collect::<Vec<_>>();
        bounded_measure_fractions(&log_target_measure, f64::from(max_measure_ratio))?
    };
    let children = constrained_unequal_split(parent, &fractions)?;
    let desired_footprints = probes
        .iter()
        .map(|probe| {
            interpolate_log_footprint_at(
                &probe.position,
                parent_index,
                particles,
                desired_footprint,
                neighbor_count,
            )
            .exp()
        })
        .collect();
    Ok(ContinuousSplitPlan {
        fractions,
        desired_footprints,
        children,
    })
}

/// Converts arbitrary log target measures into positive fractions with a
/// bounded largest-to-smallest ratio. Clamping happens in log space so no
/// preferred absolute material scale is introduced.
pub(crate) fn bounded_measure_fractions(
    log_target_measure: &[f64],
    max_ratio: f64,
) -> AutomataResult<Vec<f64>> {
    if log_target_measure.is_empty()
        || log_target_measure.iter().any(|value| !value.is_finite())
        || !max_ratio.is_finite()
        || max_ratio < 1.0
    {
        return Err(AutomataError::InvalidArgument(
            "continuous child fractions require finite log targets and max ratio >= 1".to_string(),
        ));
    }
    let center = log_target_measure.iter().sum::<f64>() / log_target_measure.len() as f64;
    let half_range = 0.5 * max_ratio.ln();
    let mut weights = log_target_measure
        .iter()
        .map(|value| value.clamp(center - half_range, center + half_range).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(AutomataError::InvalidArgument(
            "continuous child fraction normalization failed".to_string(),
        ));
    }
    weights.iter_mut().for_each(|weight| *weight /= sum);
    // Make the algebraic sum deterministic and exactly one to floating-point
    // precision without changing positivity.
    let prefix = weights[..weights.len() - 1].iter().sum::<f64>();
    *weights.last_mut().expect("non-empty weights") = 1.0 - prefix;
    if weights.iter().any(|weight| *weight <= 0.0) {
        return Err(AutomataError::InvalidArgument(
            "continuous child fraction clamping produced a non-positive value".to_string(),
        ));
    }
    Ok(weights)
}

pub(crate) fn material_scale_metrics(
    particles: &AdaptiveParticleSet,
    reference_footprint: f32,
) -> MaterialScaleMetrics {
    if particles.is_empty() || !reference_footprint.is_finite() || reference_footprint <= 0.0 {
        return MaterialScaleMetrics::default();
    }
    let log_scales = (0..particles.len())
        .map(|index| (particles.footprint(index) / reference_footprint).log2())
        .collect::<Vec<_>>();
    let occupied_sixty_fourth_octave_bins = log_scales
        .iter()
        .map(|value| (value * 64.0).round() as i32)
        .collect::<BTreeSet<_>>()
        .len();
    let dyadic_errors = log_scales
        .iter()
        .map(|value| value - value.round())
        .collect::<Vec<_>>();
    let fractional_octave_fraction = dyadic_errors
        .iter()
        .filter(|error| error.abs() > 1.0e-3)
        .count() as f32
        / dyadic_errors.len().max(1) as f32;
    let dyadic_quantization_rmse_octaves =
        (dyadic_errors.iter().map(|error| error * error).sum::<f32>()
            / dyadic_errors.len().max(1) as f32)
            .sqrt();
    MaterialScaleMetrics {
        occupied_sixty_fourth_octave_bins,
        fractional_octave_fraction,
        dyadic_quantization_rmse_octaves,
    }
}

pub(crate) fn split_respects_scale_grading(
    parent_index: usize,
    children: &[CanonicalMaterial],
    inherited_bandwidth: f32,
    particles: &AdaptiveParticleSet,
    max_ratio: f32,
    pair_scale_power: f32,
) -> bool {
    if max_ratio <= 0.0 {
        return true;
    }
    let child_footprints = children
        .iter()
        .map(|child| {
            material_footprint_radius(child.represented_measure as f32, particles.spatial_dims)
        })
        .collect::<Vec<_>>();
    for lhs in 0..children.len() {
        for rhs in lhs + 1..children.len() {
            if scale_ratio(child_footprints[lhs], child_footprints[rhs]) > max_ratio {
                return false;
            }
        }
    }
    for (child, child_footprint) in children.iter().zip(child_footprints) {
        for other in 0..particles.len() {
            if other == parent_index {
                continue;
            }
            let distance2 = (0..particles.spatial_dims)
                .map(|axis| {
                    let delta = child.position[axis] as f32 - particles.positions[other][axis];
                    delta * delta
                })
                .sum::<f32>();
            let pair_bandwidth = power_mean(
                inherited_bandwidth,
                particles.bandwidth[other],
                pair_scale_power,
            );
            if distance2 < pair_bandwidth * pair_bandwidth
                && scale_ratio(child_footprint, particles.footprint(other)) > max_ratio
            {
                return false;
            }
        }
    }
    true
}

pub(crate) fn merge_respects_scale_grading(
    merged: &CanonicalMaterial,
    merged_bandwidth: f32,
    merged_indices: &[usize],
    particles: &AdaptiveParticleSet,
    max_ratio: f32,
    pair_scale_power: f32,
) -> bool {
    if max_ratio <= 0.0 {
        return true;
    }
    let merged_footprint =
        material_footprint_radius(merged.represented_measure as f32, particles.spatial_dims);
    for other in 0..particles.len() {
        if merged_indices.contains(&other) {
            continue;
        }
        let distance2 = (0..particles.spatial_dims)
            .map(|axis| {
                let delta = merged.position[axis] as f32 - particles.positions[other][axis];
                delta * delta
            })
            .sum::<f32>();
        let pair_bandwidth = power_mean(
            merged_bandwidth,
            particles.bandwidth[other],
            pair_scale_power,
        );
        if distance2 < pair_bandwidth * pair_bandwidth
            && scale_ratio(merged_footprint, particles.footprint(other)) > max_ratio
        {
            return false;
        }
    }
    true
}

fn desired_log_footprint_gradient(
    parent_index: usize,
    particles: &AdaptiveParticleSet,
    desired_footprint: &[f32],
    neighbor_count: usize,
) -> [f32; 3] {
    let dim = particles.spatial_dims;
    let mut neighbors = (0..particles.len())
        .filter(|index| *index != parent_index)
        .map(|index| {
            let distance2 = (0..dim)
                .map(|axis| {
                    let delta =
                        particles.positions[index][axis] - particles.positions[parent_index][axis];
                    delta * delta
                })
                .sum::<f32>();
            (distance2, particles.particle_id[index], index)
        })
        .collect::<Vec<_>>();
    neighbors.sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
    neighbors.truncate(neighbor_count);
    if neighbors.len() < dim {
        return [0.0; 3];
    }
    let reference2 = particles.footprint(parent_index).powi(2).max(1.0e-12);
    let parent_log = desired_footprint[parent_index].max(f32::MIN_POSITIVE).ln();
    let mut moment = [0.0_f32; 9];
    let mut rhs = [0.0_f32; 3];
    for (distance2, _, index) in neighbors {
        let weight = 1.0 / (distance2 + reference2);
        let delta_value = desired_footprint[index].max(f32::MIN_POSITIVE).ln() - parent_log;
        let mut delta = [0.0; 3];
        for axis in 0..dim {
            delta[axis] =
                particles.positions[index][axis] - particles.positions[parent_index][axis];
            rhs[axis] += weight * delta[axis] * delta_value;
        }
        for row in 0..dim {
            for col in 0..dim {
                moment[row * 3 + col] += weight * delta[row] * delta[col];
            }
        }
    }
    let trace = (0..dim).map(|axis| moment[axis * 3 + axis]).sum::<f32>();
    let regularization = (trace / dim as f32).max(1.0e-8) * 1.0e-4;
    for axis in 0..dim {
        moment[axis * 3 + axis] += regularization;
    }
    solve_spd_3(moment, rhs, dim).unwrap_or([0.0; 3])
}

fn interpolate_log_footprint_at(
    query: &[f64],
    parent_index: usize,
    particles: &AdaptiveParticleSet,
    desired_footprint: &[f32],
    neighbor_count: usize,
) -> f32 {
    let gradient =
        desired_log_footprint_gradient(parent_index, particles, desired_footprint, neighbor_count);
    desired_footprint[parent_index].max(f32::MIN_POSITIVE).ln()
        + (0..particles.spatial_dims)
            .map(|axis| {
                gradient[axis] * (query[axis] as f32 - particles.positions[parent_index][axis])
            })
            .sum::<f32>()
}

fn solve_spd_3(matrix: [f32; 9], rhs: [f32; 3], dim: usize) -> Option<[f32; 3]> {
    let mut lower = [0.0_f32; 9];
    for row in 0..dim {
        for col in 0..=row {
            let mut sum = matrix[row * 3 + col];
            for k in 0..col {
                sum -= lower[row * 3 + k] * lower[col * 3 + k];
            }
            if row == col {
                if !sum.is_finite() || sum <= 1.0e-12 {
                    return None;
                }
                lower[row * 3 + col] = sum.sqrt();
            } else {
                lower[row * 3 + col] = sum / lower[col * 3 + col];
            }
        }
    }
    let mut intermediate = [0.0_f32; 3];
    for row in 0..dim {
        intermediate[row] = (rhs[row]
            - (0..row)
                .map(|col| lower[row * 3 + col] * intermediate[col])
                .sum::<f32>())
            / lower[row * 3 + row];
    }
    let mut solution = [0.0_f32; 3];
    for row in (0..dim).rev() {
        solution[row] = (intermediate[row]
            - (row + 1..dim)
                .map(|col| lower[col * 3 + row] * solution[col])
                .sum::<f32>())
            / lower[row * 3 + row];
    }
    Some(solution)
}

fn power_mean(lhs: f32, rhs: f32, power: f32) -> f32 {
    ((lhs.powf(power) + rhs.powf(power)) * 0.5).powf(power.recip())
}

fn scale_ratio(lhs: f32, rhs: f32) -> f32 {
    lhs.max(rhs) / lhs.min(rhs).max(f32::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_fractions_are_positive_normalized_and_ratio_limited() {
        let fractions = bounded_measure_fractions(&[-8.0, -1.0, 3.0, 12.0], 4.0).unwrap();
        assert!((fractions.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
        assert!(fractions.iter().all(|fraction| *fraction > 0.0));
        let minimum = fractions.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = fractions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(maximum / minimum <= 4.0 + 1.0e-12);
    }

    #[test]
    fn desired_scale_gradient_produces_a_conservative_fractional_split() {
        let positions = vec![
            [0.0, 0.0, 0.0, 0.0],
            [-0.08, 0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0, 0.0],
            [0.0, -0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0, 0.0],
        ];
        let footprint = 0.04_f32;
        let total_measure =
            positions.len() as f32 * crate::adaptive::unit_ball_measure(2) * footprint.powi(2);
        let particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            vec![0.0; 5],
            2,
            1,
            total_measure,
            0.1,
        )
        .unwrap();
        let parent = CanonicalMaterial {
            represented_measure: particles.represented_measure[0] as f64,
            position: vec![0.0, 0.0],
            covariance: vec![4.0e-4, 0.0, 0.0, 4.0e-4],
            extensive: vec![1.0],
        };
        let desired = vec![0.04, 0.02, 0.08, 0.03, 0.06];
        let plan = continuous_split_plan(0, &parent, &particles, &desired, 4.0, 4).unwrap();
        let minimum = plan.fractions.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = plan
            .fractions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(maximum / minimum > 1.05);
        assert!(maximum / minimum <= 4.0 + 1.0e-12);
        let audit = crate::adaptive::topology_audit(&parent, &plan.children).unwrap();
        assert!(audit.measure_relative_error < 1.0e-12, "{audit:?}");
        assert!(audit.centroid_l2_error < 1.0e-12, "{audit:?}");
        assert!(audit.second_moment_relative_error < 1.0e-12, "{audit:?}");
        assert!(
            audit.determinant_scale_relative_error < 1.0e-11,
            "{audit:?}"
        );
    }

    #[test]
    fn constant_desired_field_preserves_exact_canonical_split() {
        let positions = vec![
            [0.0, 0.0, 0.0, 0.0],
            [-0.08, 0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0, 0.0],
            [0.0, -0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0, 0.0],
        ];
        let footprint = 0.04_f32;
        let total_measure =
            positions.len() as f32 * crate::adaptive::unit_ball_measure(2) * footprint.powi(2);
        let particles = AdaptiveParticleSet::from_equal_measure(
            positions,
            vec![0.0; 5],
            2,
            1,
            total_measure,
            0.1,
        )
        .unwrap();
        let parent = CanonicalMaterial {
            represented_measure: particles.represented_measure[0] as f64,
            position: vec![0.0, 0.0],
            covariance: vec![4.0e-4, 0.0, 0.0, 4.0e-4],
            extensive: vec![1.0],
        };
        let plan = continuous_split_plan(0, &parent, &particles, &[0.04; 5], 8.0, 4).unwrap();
        assert_eq!(plan.fractions, vec![0.25; 4]);
        assert_eq!(plan.children, canonical_split(&parent).unwrap());
    }
}
