use std::time::Instant;

use super::{
    AdaptiveScalingExperimentConfig, AdaptiveScalingExperimentReport, AdaptiveScalingExperimentRow,
    AdaptiveScalingFit,
};
use crate::{AutomataError, AutomataResult};

mod quality;

#[derive(Clone, Copy)]
pub(super) enum AnalyticSolid {
    SphereWithCavity,
    Torus,
}

impl AnalyticSolid {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::SphereWithCavity => "sphere-with-cavity",
            Self::Torus => "solid-torus",
        }
    }

    pub(super) fn clearance(
        self,
        point: [f32; 3],
        config: &AdaptiveScalingExperimentConfig,
    ) -> Option<f32> {
        match self {
            Self::SphereWithCavity => {
                let outer = norm(point);
                let cavity = norm([
                    point[0] - config.sphere_cavity_center[0],
                    point[1] - config.sphere_cavity_center[1],
                    point[2] - config.sphere_cavity_center[2],
                ]);
                (outer <= config.sphere_outer_radius && cavity >= config.sphere_cavity_radius).then(
                    || {
                        (config.sphere_outer_radius - outer)
                            .min(cavity - config.sphere_cavity_radius)
                    },
                )
            }
            Self::Torus => {
                let radial = (point[0] * point[0] + point[1] * point[1]).sqrt();
                let tube_distance =
                    ((radial - config.torus_major_radius).powi(2) + point[2] * point[2]).sqrt();
                (tube_distance <= config.torus_minor_radius)
                    .then_some(config.torus_minor_radius - tube_distance)
            }
        }
    }
}

pub(super) fn run_scaling_experiment(
    config: &AdaptiveScalingExperimentConfig,
    seed: u64,
) -> AutomataResult<AdaptiveScalingExperimentReport> {
    validate(config)?;
    let started = Instant::now();
    let mut rows = Vec::new();
    for solid in [AnalyticSolid::SphereWithCavity, AnalyticSolid::Torus] {
        for &resolution in &config.resolutions {
            rows.push(run_row(config, solid, resolution, seed));
        }
    }
    let fits = [AnalyticSolid::SphereWithCavity, AnalyticSolid::Torus]
        .into_iter()
        .map(|solid| {
            let solid_rows = rows
                .iter()
                .filter(|row| row.solid == solid.name())
                .collect::<Vec<_>>();
            let tail_start = solid_rows.len().saturating_sub(3);
            AdaptiveScalingFit {
                solid: solid.name().to_string(),
                full_count_exponent: count_exponent(&solid_rows),
                tail_count_exponent: count_exponent(&solid_rows[tail_start..]),
            }
        })
        .collect();
    let quality_rows = quality::run_sparse_quality_experiment(config, seed)?;
    Ok(AdaptiveScalingExperimentReport {
        rows,
        fits,
        quality_rows,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn run_row(
    config: &AdaptiveScalingExperimentConfig,
    solid: AnalyticSolid,
    resolution: usize,
    seed: u64,
) -> AdaptiveScalingExperimentRow {
    let voxel = 2.0 / resolution as f32;
    let protected_distance = config.protected_band_voxels * voxel;
    let mut fine_leaves = 0_usize;
    let mut protected_leaves = 0_usize;
    let mut expected_retained = 0.0_f64;
    let mut retained = vec![0_usize; config.retention_samples];
    let mut spacing_ratio_sum = 0.0_f64;
    let mut spacing_ratio_max = 0.0_f64;
    for index in 0..resolution.pow(3) {
        let x = index % resolution;
        let y = (index / resolution) % resolution;
        let z = index / (resolution * resolution);
        let point = [
            -1.0 + (x as f32 + 0.5) * voxel,
            -1.0 + (y as f32 + 0.5) * voxel,
            -1.0 + (z as f32 + 0.5) * voxel,
        ];
        let Some(clearance) = solid.clearance(point, config) else {
            continue;
        };
        fine_leaves += 1;
        let spacing = (voxel
            + (clearance - protected_distance).max(0.0) / config.transition_divisor)
            .min(config.interior_spacing_cap)
            .max(voxel);
        let spacing_ratio = (spacing / voxel) as f64;
        let probability = spacing_ratio.powi(-3).clamp(0.0, 1.0);
        expected_retained += probability;
        spacing_ratio_sum += spacing_ratio;
        spacing_ratio_max = spacing_ratio_max.max(spacing_ratio);
        if clearance <= protected_distance {
            protected_leaves += 1;
        }
        for (sample, count) in retained.iter_mut().enumerate() {
            if stateless_uniform(seed, solid as u64, resolution, index, sample) < probability {
                *count += 1;
            }
        }
    }
    let retained_mean = retained.iter().sum::<usize>() as f64 / retained.len() as f64;
    let retained_stddev = (retained
        .iter()
        .map(|count| (*count as f64 - retained_mean).powi(2))
        .sum::<f64>()
        / retained.len() as f64)
        .sqrt();
    AdaptiveScalingExperimentRow {
        solid: solid.name().to_string(),
        resolution,
        fine_leaves,
        protected_leaves,
        expected_retained_leaves: expected_retained,
        retained_mean,
        retained_stddev,
        count_reduction: fine_leaves as f64 / retained_mean.max(1.0),
        mean_spacing_ratio: spacing_ratio_sum / fine_leaves.max(1) as f64,
        max_spacing_ratio: spacing_ratio_max,
    }
}

fn count_exponent(rows: &[&AdaptiveScalingExperimentRow]) -> f64 {
    if rows.len() < 2 {
        return f64::NAN;
    }
    let mean_x = rows
        .iter()
        .map(|row| (row.resolution as f64).ln())
        .sum::<f64>()
        / rows.len() as f64;
    let mean_y = rows
        .iter()
        .map(|row| row.retained_mean.max(1.0).ln())
        .sum::<f64>()
        / rows.len() as f64;
    let covariance = rows
        .iter()
        .map(|row| {
            ((row.resolution as f64).ln() - mean_x) * (row.retained_mean.max(1.0).ln() - mean_y)
        })
        .sum::<f64>();
    let variance = rows
        .iter()
        .map(|row| ((row.resolution as f64).ln() - mean_x).powi(2))
        .sum::<f64>();
    covariance / variance.max(f64::MIN_POSITIVE)
}

pub(super) fn stateless_key(
    seed: u64,
    solid: u64,
    resolution: usize,
    index: usize,
    sample: usize,
) -> u64 {
    let mut value = seed
        ^ solid.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (resolution as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (index as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (sample as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    value
}

pub(super) fn stateless_uniform(
    seed: u64,
    solid: u64,
    resolution: usize,
    index: usize,
    sample: usize,
) -> f64 {
    (stateless_key(seed, solid, resolution, index, sample) >> 11) as f64
        * (1.0 / (1_u64 << 53) as f64)
}

fn norm(value: [f32; 3]) -> f32 {
    value.iter().map(|axis| axis * axis).sum::<f32>().sqrt()
}

fn validate(config: &AdaptiveScalingExperimentConfig) -> AutomataResult<()> {
    if config.resolutions.len() < 2
        || config.resolutions.iter().any(|value| *value < 8)
        || !config.interior_spacing_cap.is_finite()
        || config.interior_spacing_cap <= 0.0
        || !config.protected_band_voxels.is_finite()
        || config.protected_band_voxels <= 0.0
        || !config.transition_divisor.is_finite()
        || config.transition_divisor <= 0.0
        || config.retention_samples == 0
        || config.quality_resolution < 8
        || config.quality_spacing_cap_ratios.is_empty()
        || config
            .quality_spacing_cap_ratios
            .iter()
            .any(|value| !value.is_finite() || *value < 1.0)
        || !config.sphere_outer_radius.is_finite()
        || !config.sphere_cavity_radius.is_finite()
        || !config.torus_major_radius.is_finite()
        || !config.torus_minor_radius.is_finite()
        || config.sphere_outer_radius <= 0.0
        || config.sphere_cavity_radius <= 0.0
        || config.torus_major_radius <= 0.0
        || config.torus_minor_radius <= 0.0
        || config
            .sphere_cavity_center
            .iter()
            .any(|value| !value.is_finite())
        || norm(config.sphere_cavity_center) + config.sphere_cavity_radius
            >= config.sphere_outer_radius
        || config.torus_major_radius + config.torus_minor_radius >= 1.0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive fixed-world scaling config".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_world_count_audit_is_deterministic_and_protects_boundary_band() {
        let config = AdaptiveScalingExperimentConfig {
            resolutions: vec![16, 24, 32],
            ..AdaptiveScalingExperimentConfig::default()
        };
        let first = run_scaling_experiment(&config, 17).unwrap();
        let second = run_scaling_experiment(&config, 17).unwrap();
        assert_eq!(first.rows.len(), 6);
        for (lhs, rhs) in first.rows.iter().zip(&second.rows) {
            assert_eq!(lhs.retained_mean, rhs.retained_mean);
            assert!(lhs.retained_mean >= lhs.protected_leaves as f64);
            assert!(lhs.count_reduction >= 1.0);
        }
        assert!(
            first
                .fits
                .iter()
                .all(|fit| fit.full_count_exponent.is_finite()
                    && fit.tail_count_exponent.is_finite())
        );
    }
}
