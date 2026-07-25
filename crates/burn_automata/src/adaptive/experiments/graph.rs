use std::time::Instant;

use rand::{Rng, SeedableRng, rngs::StdRng};

use burn_automata_kernels::{
    AdaptiveGraphPolicy, AdaptivePerceptionConfig, AdaptivePerceptionOutput, adaptive_perceive,
    adaptive_perceive_all_pairs,
};

use super::{AdaptiveGraphExperimentConfig, AdaptiveGraphExperimentRow};
use crate::{AutomataError, AutomataResult};

pub(super) fn run_graph_experiment(
    config: &AdaptiveGraphExperimentConfig,
    seed: u64,
) -> AutomataResult<Vec<AdaptiveGraphExperimentRow>> {
    if config.spatial_dims.is_empty()
        || config.spatial_dims.iter().any(|dim| !matches!(dim, 2 | 3))
        || config.particle_counts.is_empty()
        || config.neighbor_caps.is_empty()
        || config.particle_counts.contains(&0)
        || config.neighbor_caps.contains(&0)
        || !config.coarse_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.coarse_fraction)
        || !config.fine_bandwidth.is_finite()
        || !config.coarse_bandwidth.is_finite()
        || !config.target_fine_degree.is_finite()
        || config.fine_bandwidth <= 0.0
        || config.coarse_bandwidth < config.fine_bandwidth
        || config.target_fine_degree <= 0.0
        || config.timed_runs == 0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive graph experiment config".to_string(),
        ));
    }
    let mut rows = Vec::new();
    for &dim in &config.spatial_dims {
        for &particle_count in &config.particle_counts {
            let mut rng =
                StdRng::seed_from_u64(seed ^ particle_count as u64 ^ ((dim as u64) << 56));
            let domain_extent = graph_domain_extent(
                particle_count,
                dim,
                config.fine_bandwidth,
                config.target_fine_degree,
            );
            let positions = (0..particle_count)
                .map(|_| {
                    [
                        rng.random_range(-0.5_f32..0.5) * domain_extent,
                        rng.random_range(-0.5_f32..0.5) * domain_extent,
                        if dim == 3 {
                            rng.random_range(-0.5_f32..0.5) * domain_extent
                        } else {
                            0.0
                        },
                        0.0,
                    ]
                })
                .collect::<Vec<_>>();
            let bandwidth = (0..particle_count)
                .map(|index| {
                    if index as f32 / (particle_count as f32) < config.coarse_fraction {
                        config.coarse_bandwidth
                    } else {
                        config.fine_bandwidth
                    }
                })
                .collect::<Vec<_>>();
            let states = vec![0.0; particle_count];
            let measure = vec![1.0; particle_count];
            if particle_count <= config.all_pairs_baseline_max_particles {
                let cfg = perception_config(
                    config,
                    dim,
                    particle_count,
                    AdaptiveGraphPolicy::RawSupport,
                    particle_count.saturating_sub(1).max(1),
                );
                let measurement = benchmark(config, || {
                    adaptive_perceive_all_pairs(
                        &positions,
                        &states,
                        &measure,
                        &bandwidth,
                        1,
                        particle_count,
                        1,
                        cfg,
                    )
                })?;
                rows.push(graph_row(
                    particle_count,
                    dim,
                    "all-pairs",
                    AdaptiveGraphPolicy::RawSupport,
                    0,
                    measurement,
                ));
            }
            for (policy, cap) in std::iter::once((AdaptiveGraphPolicy::RawSupport, usize::MAX))
                .chain(config.neighbor_caps.iter().flat_map(|cap| {
                    [
                        (AdaptiveGraphPolicy::DirectedTopK, *cap),
                        (AdaptiveGraphPolicy::MutualTopK, *cap),
                    ]
                }))
            {
                let actual_cap = cap.min(particle_count.saturating_sub(1).max(1));
                let cfg = perception_config(config, dim, particle_count, policy, actual_cap);
                let measurement = benchmark(config, || {
                    adaptive_perceive(
                        &positions,
                        &states,
                        &measure,
                        &bandwidth,
                        1,
                        particle_count,
                        1,
                        cfg,
                    )
                })?;
                rows.push(graph_row(
                    particle_count,
                    dim,
                    "spatial-hash",
                    policy,
                    actual_cap,
                    measurement,
                ));
            }
        }
    }
    Ok(rows)
}

struct GraphMeasurement {
    output: AdaptivePerceptionOutput,
    mean_seconds: f64,
    stddev_seconds: f64,
    min_seconds: f64,
    timed_runs: usize,
}

fn benchmark(
    config: &AdaptiveGraphExperimentConfig,
    mut run: impl FnMut() -> burn_automata_kernels::KernelResult<AdaptivePerceptionOutput>,
) -> Result<GraphMeasurement, burn_automata_kernels::KernelError> {
    for _ in 0..config.warmup_runs {
        let _ = run()?;
    }
    let mut output = None;
    let mut seconds = Vec::with_capacity(config.timed_runs);
    for _ in 0..config.timed_runs {
        let started = Instant::now();
        output = Some(run()?);
        seconds.push(started.elapsed().as_secs_f64());
    }
    let mean_seconds = seconds.iter().sum::<f64>() / seconds.len() as f64;
    let stddev_seconds = (seconds
        .iter()
        .map(|value| (value - mean_seconds).powi(2))
        .sum::<f64>()
        / seconds.len() as f64)
        .sqrt();
    Ok(GraphMeasurement {
        output: output.expect("timed_runs is validated as non-zero"),
        mean_seconds,
        stddev_seconds,
        min_seconds: seconds.into_iter().fold(f64::INFINITY, f64::min),
        timed_runs: config.timed_runs,
    })
}

fn graph_domain_extent(
    particle_count: usize,
    dim: usize,
    bandwidth: f32,
    target_degree: f32,
) -> f32 {
    let unit_ball = match dim {
        2 => std::f32::consts::PI,
        3 => 4.0 * std::f32::consts::PI / 3.0,
        _ => unreachable!(),
    };
    bandwidth * (particle_count as f32 * unit_ball / target_degree).powf(1.0 / dim as f32)
}

fn perception_config(
    config: &AdaptiveGraphExperimentConfig,
    dim: usize,
    particle_count: usize,
    graph_policy: AdaptiveGraphPolicy,
    max_neighbors: usize,
) -> AdaptivePerceptionConfig {
    AdaptivePerceptionConfig {
        dim,
        graph_policy,
        max_neighbors: max_neighbors.min(particle_count.saturating_sub(1).max(1)),
        pair_scale_power: 8.0,
        reference_measure: 0.0,
        min_bandwidth: config.fine_bandwidth,
        max_bandwidth: config.coarse_bandwidth,
        support_bin_ratio: 2.0,
        spacing_target_neighbors: if dim == 3 { 32.0 } else { 16.0 },
        spacing_root_iterations: 12,
        shepard_epsilon: 1.0e-8,
        moment_regularization: 1.0e-4,
        moment_condition_limit: 1.0e5,
        log_normalize_gradients: false,
        include_position_features: false,
    }
}

fn graph_row(
    particles: usize,
    spatial_dims: usize,
    search: &str,
    policy: AdaptiveGraphPolicy,
    neighbor_cap: usize,
    measurement: GraphMeasurement,
) -> AdaptiveGraphExperimentRow {
    let GraphMeasurement {
        output,
        mean_seconds,
        stddev_seconds,
        min_seconds,
        timed_runs,
    } = measurement;
    AdaptiveGraphExperimentRow {
        spatial_dims,
        particles,
        search: search.to_string(),
        policy: match policy {
            AdaptiveGraphPolicy::RawSupport => "raw-support",
            AdaptiveGraphPolicy::DirectedTopK => "directed-top-k",
            AdaptiveGraphPolicy::MutualTopK => "mutual-top-k",
        }
        .to_string(),
        neighbor_cap: if policy == AdaptiveGraphPolicy::RawSupport {
            0
        } else {
            neighbor_cap
        },
        candidate_visits: output.graph.candidate_visits,
        raw_messages: output.graph.raw_messages,
        accepted_messages: output.graph.accepted_messages,
        degree_mean: output.graph.degree_mean,
        degree_p95: output.graph.degree_p95,
        degree_max: output.graph.degree_max,
        isolated_particles: output.graph.isolated_particles,
        cross_scale_fraction: output.graph.cross_scale_fraction,
        elapsed_ms: mean_seconds * 1_000.0,
        elapsed_ms_stddev: stddev_seconds * 1_000.0,
        elapsed_ms_min: min_seconds * 1_000.0,
        timed_runs,
        messages_per_second: output.graph.accepted_messages as f64
            / mean_seconds.max(f64::MIN_POSITIVE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_stress_scene_exercises_hard_neighbor_cap() {
        let rows = run_graph_experiment(
            &AdaptiveGraphExperimentConfig {
                spatial_dims: vec![2, 3],
                particle_counts: vec![256],
                neighbor_caps: vec![16],
                all_pairs_baseline_max_particles: 256,
                coarse_fraction: 0.25,
                fine_bandwidth: 0.035,
                coarse_bandwidth: 0.14,
                target_fine_degree: 48.0,
                warmup_runs: 0,
                timed_runs: 1,
            },
            42,
        )
        .unwrap();
        for dim in [2, 3] {
            let raw = rows
                .iter()
                .find(|row| {
                    row.spatial_dims == dim
                        && row.search == "spatial-hash"
                        && row.policy == "raw-support"
                })
                .unwrap();
            let capped = rows
                .iter()
                .find(|row| {
                    row.spatial_dims == dim
                        && row.policy == "directed-top-k"
                        && row.neighbor_cap == 16
                })
                .unwrap();
            assert!(raw.degree_p95 > 16, "{dim}D raw p95 did not stress top-K");
            assert_eq!(capped.degree_max, 16);
            assert!(capped.accepted_messages < raw.accepted_messages);
        }
    }
}
