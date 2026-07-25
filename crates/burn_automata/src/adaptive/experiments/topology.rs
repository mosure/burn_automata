use std::time::Instant;

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{AdaptiveTopologyExperimentConfig, AdaptiveTopologyExperimentReport};
use crate::adaptive::{
    CanonicalMaterial, canonical_split, constrained_unequal_split,
    scale::bounded_measure_fractions, topology_audit,
};
use crate::{AutomataError, AutomataResult};

pub(super) fn run_topology_experiment(
    config: AdaptiveTopologyExperimentConfig,
    seed: u64,
) -> AutomataResult<AdaptiveTopologyExperimentReport> {
    if config.samples == 0
        || !config.max_unequal_measure_ratio.is_finite()
        || config.max_unequal_measure_ratio < 1.0
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive topology experiment requires samples and unequal ratio >= 1".to_string(),
        ));
    }
    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x7a11_0ca1);
    let mut report = AdaptiveTopologyExperimentReport {
        samples: config.samples,
        ..AdaptiveTopologyExperimentReport::default()
    };
    for sample in 0..config.samples {
        let dim = 2 + sample % 3;
        let mut factor = vec![0.0_f64; dim * dim];
        for row in 0..dim {
            for col in 0..=row {
                factor[row * dim + col] = if row == col {
                    rng.random_range(0.05_f64..0.6)
                } else {
                    rng.random_range(-0.08_f64..0.08)
                };
            }
        }
        let mut covariance = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                covariance[row * dim + col] = (0..dim)
                    .map(|axis| factor[row * dim + axis] * factor[col * dim + axis])
                    .sum::<f64>();
                if row == col {
                    covariance[row * dim + col] += 1.0e-3;
                }
            }
        }
        let parent = CanonicalMaterial {
            represented_measure: rng.random_range(1.0e-3_f64..10.0),
            position: (0..dim).map(|_| rng.random_range(-1.0_f64..1.0)).collect(),
            covariance,
            extensive: (0..4).map(|_| rng.random_range(-2.0_f64..2.0)).collect(),
        };
        let children = canonical_split(&parent)?;
        let audit = topology_audit(&parent, &children)?;
        record_audit(&mut report, &audit);
        report.canonical_events += 1;

        let half_log_ratio = 0.5 * config.max_unequal_measure_ratio.ln();
        let log_targets = (0..2 * dim)
            .map(|_| rng.random_range(-half_log_ratio..=half_log_ratio))
            .collect::<Vec<_>>();
        let fractions = bounded_measure_fractions(&log_targets, config.max_unequal_measure_ratio)?;
        let minimum = fractions.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = fractions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        report.maximum_sampled_child_measure_ratio = report
            .maximum_sampled_child_measure_ratio
            .max(maximum / minimum);
        let unequal_children = constrained_unequal_split(&parent, &fractions)?;
        let unequal_audit = topology_audit(&parent, &unequal_children)?;
        record_audit(&mut report, &unequal_audit);
        report.unequal_events += 1;
    }
    let elapsed = started.elapsed();
    report.elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
    report.events_per_second = (report.canonical_events + report.unequal_events) as f64
        / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    Ok(report)
}

fn record_audit(
    report: &mut AdaptiveTopologyExperimentReport,
    audit: &crate::adaptive::TopologyAudit,
) {
    report.max_measure_relative_error = report
        .max_measure_relative_error
        .max(audit.measure_relative_error);
    report.max_centroid_l2_error = report.max_centroid_l2_error.max(audit.centroid_l2_error);
    report.max_second_moment_relative_error = report
        .max_second_moment_relative_error
        .max(audit.second_moment_relative_error);
    report.max_extensive_relative_error = report
        .max_extensive_relative_error
        .max(audit.extensive_relative_error);
    report.max_determinant_scale_relative_error = report
        .max_determinant_scale_relative_error
        .max(audit.determinant_scale_relative_error);
    report.spd_failures += usize::from(!audit.child_spd);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_unequal_events_remain_conservative() {
        let report = run_topology_experiment(
            AdaptiveTopologyExperimentConfig {
                samples: 100_000,
                max_unequal_measure_ratio: 8.0,
            },
            17,
        )
        .unwrap();
        assert_eq!(report.canonical_events, 100_000);
        assert_eq!(report.unequal_events, 100_000);
        assert_eq!(report.spd_failures, 0);
        assert!(report.maximum_sampled_child_measure_ratio <= 8.0 + 1.0e-10);
        assert!(report.max_measure_relative_error < 1.0e-12);
        assert!(report.max_centroid_l2_error < 1.0e-11);
        assert!(report.max_second_moment_relative_error < 1.0e-11);
        assert!(report.max_extensive_relative_error < 1.0e-12);
        assert!(report.max_determinant_scale_relative_error < 1.0e-10);
    }
}
