use super::*;
use crate::{NpaConfig, NpaModel, ParticleSeed, rollout::seed_particles_scaled};

#[test]
fn canonical_events_preserve_measure_centroid_and_second_moment() {
    for dim in [2usize, 3, 4] {
        let mut covariance = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                covariance[row * dim + col] = if row == col {
                    0.2 + row as f64 * 0.07
                } else {
                    0.01 / (1 + row.abs_diff(col)) as f64
                };
            }
        }
        let parent = CanonicalMaterial {
            represented_measure: 3.25,
            position: (0..dim).map(|axis| axis as f64 * 0.17 - 0.2).collect(),
            covariance,
            extensive: vec![1.5, -0.75, 4.0],
        };
        let children = canonical_split(&parent).unwrap();
        assert_eq!(children.len(), 2 * dim);
        let audit = topology_audit(&parent, &children).unwrap();
        assert!(audit.measure_relative_error < 1.0e-12);
        assert!(audit.centroid_l2_error < 1.0e-12);
        assert!(audit.second_moment_relative_error < 1.0e-12);
        assert!(audit.extensive_relative_error < 1.0e-12);
        assert!(audit.determinant_scale_relative_error < 1.0e-11);
        assert!(audit.child_spd);
        let merged = canonical_merge(&children).unwrap();
        assert_eq!(merged.position.len(), dim);
    }
}

#[test]
fn unequal_events_preserve_invariants_and_continuous_footprint_scale() {
    for dim in [2usize, 3, 4] {
        let mut covariance = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                covariance[row * dim + col] = if row == col {
                    0.18 + row as f64 * 0.09
                } else {
                    0.008 / (1 + row.abs_diff(col)) as f64
                };
            }
        }
        let parent = CanonicalMaterial {
            represented_measure: 2.75,
            position: (0..dim).map(|axis| axis as f64 * 0.11 - 0.15).collect(),
            covariance,
            extensive: vec![0.75, -1.25, 3.5],
        };
        let child_count = 2 * dim;
        let normalizer = (1..=child_count).map(|value| value as f64).sum::<f64>();
        let fractions = (1..=child_count)
            .map(|value| value as f64 / normalizer)
            .collect::<Vec<_>>();
        let children = constrained_unequal_split(&parent, &fractions).unwrap();
        let audit = topology_audit(&parent, &children).unwrap();
        assert!(audit.measure_relative_error < 1.0e-12, "{audit:?}");
        assert!(audit.centroid_l2_error < 1.0e-12, "{audit:?}");
        assert!(audit.second_moment_relative_error < 1.0e-12, "{audit:?}");
        assert!(audit.extensive_relative_error < 1.0e-12, "{audit:?}");
        assert!(
            audit.determinant_scale_relative_error < 1.0e-11,
            "{audit:?}"
        );
        assert!(audit.child_spd);

        for (child, fraction) in children.iter().zip(fractions) {
            let covariance_scale = child.covariance[0] / parent.covariance[0];
            let footprint_ratio = covariance_scale.sqrt();
            assert!((footprint_ratio - fraction.powf(1.0 / dim as f64)).abs() < 1.0e-12);
        }
    }
}

#[test]
fn near_equal_unequal_split_is_geometrically_continuous_with_canonical_split() {
    let parent = CanonicalMaterial {
        represented_measure: 1.0,
        position: vec![0.15, -0.25],
        covariance: vec![0.12, 0.018, 0.018, 0.07],
        extensive: vec![0.5, -0.75],
    };
    let canonical = canonical_split(&parent).unwrap();
    let epsilon = 1.0e-7;
    let fractions = [
        0.25 + epsilon,
        0.25 - epsilon,
        0.25 + 0.5 * epsilon,
        0.25 - 0.5 * epsilon,
    ];
    let perturbed = constrained_unequal_split(&parent, &fractions).unwrap();
    let maximum_position_delta = canonical
        .iter()
        .zip(&perturbed)
        .flat_map(|(canonical, perturbed)| {
            canonical
                .position
                .iter()
                .zip(&perturbed.position)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
        })
        .fold(0.0_f64, f64::max);
    let maximum_covariance_delta = canonical
        .iter()
        .zip(&perturbed)
        .flat_map(|(canonical, perturbed)| {
            canonical
                .covariance
                .iter()
                .zip(&perturbed.covariance)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(maximum_position_delta < 1.0e-6, "{maximum_position_delta}");
    assert!(
        maximum_covariance_delta < 1.0e-6,
        "{maximum_covariance_delta}"
    );
}

#[test]
fn budget_allocator_hits_unclamped_target() {
    let rows = 512;
    let error = (0..rows)
        .map(|index| 0.25 + (index as f32 * 0.017).sin().abs())
        .collect::<Vec<_>>();
    let measure = vec![1.0 / rows as f32; rows];
    let allocation =
        allocate_resolution_budget(&error, &measure, 2, 2.0, 0.05, 0.005, 0.25, 512).unwrap();
    assert!((allocation.expected_leaf_count - 512.0).abs() < 0.1);
    assert!(
        allocation
            .desired_footprint
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn proposed_footprints_are_normalized_to_global_leaf_budget() {
    let rows = 128;
    let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
    let proposed = (0..rows)
        .map(|index| 0.005 + (index % 11) as f32 * 0.001)
        .collect::<Vec<_>>();
    let measure = vec![total_measure / rows as f32; rows];
    let allocation = normalize_footprint_budget(&proposed, &measure, 2, 0.001, 0.1, 512).unwrap();
    assert!((allocation.expected_leaf_count - 512.0).abs() < 0.1);
    assert!(
        allocation
            .desired_footprint
            .iter()
            .all(|value| (0.001..=0.1).contains(value))
    );
}

#[test]
fn bounded_footprint_projection_preserves_budget_and_limits_one_pass_scale() {
    let rows = 128;
    let current_radius = 0.02_f32;
    let total_measure = rows as f32 * std::f32::consts::PI * current_radius.powi(2);
    let current = vec![current_radius; rows];
    let proposed = (0..rows)
        .map(|index| if index % 2 == 0 { 1.0e-5 } else { 1.0 })
        .collect::<Vec<_>>();
    let measure = vec![total_measure / rows as f32; rows];
    let allocation = normalize_footprint_budget_bounded(
        &proposed, &current, &measure, 2, 0.001, 0.1, rows, 0.5, 2.0,
    )
    .unwrap();
    assert!((allocation.expected_leaf_count - rows as f32).abs() < 0.1);
    assert!(allocation.desired_footprint.iter().all(|value| {
        let ratio = value / current_radius;
        (0.5 - 1.0e-6..=2.0 + 1.0e-6).contains(&ratio)
    }));
    assert!(
        allocation
            .desired_footprint
            .iter()
            .any(|value| *value < current_radius)
    );
    assert!(
        allocation
            .desired_footprint
            .iter()
            .any(|value| *value > current_radius)
    );
}

#[test]
fn adaptive_artifact_is_binary_checksummed_and_roundtrips() {
    let model = AdaptiveNpaModel::seeded(
        NpaModel::seeded(NpaConfig::growing_2d(), 4),
        AdaptiveNpaConfig::growing_2d(),
        9,
    )
    .unwrap();
    let artifact = AdaptiveModelArtifact::new(model, Some("test".to_string())).unwrap();
    let path = std::env::temp_dir().join(format!(
        "burn_automata_adaptive_{}_{}.bpk",
        std::process::id(),
        91
    ));
    let digest = save_adaptive_model(&path, &artifact).unwrap();
    assert_eq!(digest.len(), 64);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"BANPABP1");
    assert_ne!(bytes[52], b'{');
    let restored = load_adaptive_model(&path).unwrap();
    assert_eq!(restored.model.config, artifact.model.config);
    assert_eq!(
        restored.model.controller.weights,
        artifact.model.controller.weights
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn adaptive_artifact_runtime_rebind_preserves_weights_and_stage() {
    let model = AdaptiveNpaModel::seeded(
        NpaModel::seeded(NpaConfig::growing_2d(), 4),
        AdaptiveNpaConfig::growing_2d(),
        9,
    )
    .unwrap();
    let artifact = AdaptiveModelArtifact::task_trained(model, Some("trained".to_owned())).unwrap();
    let weights = artifact.model.controller.weights.clone();
    let stage = artifact.training_stage;
    let mut runtime = artifact.model.config.clone();
    runtime.render_transition_steps = 37;
    let rebound = artifact
        .with_runtime_config(runtime.clone(), Some("validated".to_owned()))
        .unwrap();
    assert_eq!(rebound.model.config, runtime);
    assert_eq!(rebound.model.controller.weights, weights);
    assert_eq!(rebound.training_stage, stage);
    assert_eq!(rebound.source.as_deref(), Some("validated"));
}

#[test]
fn compatible_rule_rollout_matches_fixed_multi_step_at_uniform_scale() {
    let rule = NpaModel::seeded(NpaConfig::growing_2d(), 42);
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.min_leaves = 64;
    config.target_leaves = 64;
    config.max_leaves = 256;
    config.rule_graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
    config.perception.graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
    config.perception.min_bandwidth = 0.1;
    config.perception.max_bandwidth = 0.1;
    let model = AdaptiveNpaModel::seeded(rule.clone(), config, 7).unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        64,
        model.rule.config.state_dims,
        2,
        42,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let steps = 8;
    let mut fixed_positions = positions.clone();
    let mut fixed_states = states.clone();
    let material_ids = (0..64).map(|index| index as u64).collect::<Vec<_>>();
    for step in 1..=steps {
        let mask = material_ids
            .iter()
            .map(|id| f32::from(crate::rollout::stable_material_uniform(42, step, *id) < 0.5))
            .collect::<Vec<_>>();
        let fixed = rule
            .step_cpu(
                &fixed_positions,
                &fixed_states,
                1,
                64,
                &crate::upstream_growing_2d_hashgrid(),
                1.0,
                Some(&mask),
            )
            .unwrap();
        fixed_positions = fixed.next_positions;
        fixed_states = fixed.next_states;
    }
    let particles = AdaptiveParticleSet::from_equal_measure(
        positions.clone(),
        states.clone(),
        2,
        model.rule.config.state_dims,
        std::f32::consts::PI * 0.2_f32.powi(2),
        0.1,
    )
    .unwrap();
    let initial_measure = particles.total_measure();
    let trace = run_adaptive_rollout(
        &model,
        particles,
        AdaptiveRolloutConfig {
            steps,
            update_prob: 0.5,
            seed: 42,
            topology_enabled: false,
            snapshot_interval: steps,
            ..AdaptiveRolloutConfig::default()
        },
    )
    .unwrap();
    assert_eq!(trace.particles.len(), 64);
    assert!((trace.particles.total_measure() - initial_measure).abs() < 1.0e-9);
    assert!(
        trace
            .particles
            .bandwidth
            .iter()
            .all(|bandwidth| (*bandwidth - 0.1).abs() < f32::EPSILON)
    );
    assert_eq!(trace.metrics.len(), steps);
    let max_position_error = trace
        .particles
        .positions
        .iter()
        .zip(&fixed_positions)
        .flat_map(|(actual, expected)| {
            actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
        })
        .fold(0.0_f32, f32::max);
    let max_state_error = trace
        .particles
        .states
        .iter()
        .zip(&fixed_states)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_position_error < 2.0e-4, "{max_position_error}");
    assert!(max_state_error < 2.0e-4, "{max_state_error}");
}

#[test]
fn chunked_adaptive_rollout_matches_uninterrupted_dynamics() {
    let rule = NpaModel::seeded(NpaConfig::growing_2d(), 42);
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.min_leaves = 32;
    config.target_leaves = 32;
    config.max_leaves = 128;
    let model = AdaptiveNpaModel::seeded(rule, config, 7).unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        32,
        model.rule.config.state_dims,
        2,
        19,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let particles = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        model.rule.config.state_dims,
        std::f32::consts::PI * 0.2_f32.powi(2),
        0.1,
    )
    .unwrap();
    let rollout = AdaptiveRolloutConfig {
        steps: 7,
        topology_enabled: false,
        update_prob: 0.63,
        seed: 1234,
        snapshot_interval: 7,
        ..AdaptiveRolloutConfig::default()
    };
    let uninterrupted = run_adaptive_rollout(&model, particles.clone(), rollout).unwrap();
    let first = advance_adaptive_rollout(
        &model,
        particles,
        AdaptiveRolloutConfig {
            steps: 3,
            ..rollout
        },
        0,
    )
    .unwrap();
    let second = advance_adaptive_rollout(
        &model,
        first.particles,
        AdaptiveRolloutConfig {
            steps: 4,
            ..rollout
        },
        3,
    )
    .unwrap();
    assert_eq!(second.particles, uninterrupted.particles);
}

#[test]
fn zero_residual_branches_preserve_frozen_rule_and_report_nonmaterial_hubs() {
    let rule = NpaModel::seeded(NpaConfig::growing_2d(), 42);
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.min_leaves = 64;
    config.target_leaves = 64;
    config.max_leaves = 128;
    let local = AdaptiveNpaModel::seeded(rule, config, 7).unwrap();
    let mut hierarchical = local.clone();
    hierarchical.enable_zero_local_residual_rule().unwrap();
    hierarchical.enable_zero_proxy_rule().unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        64,
        local.rule.config.state_dims,
        2,
        19,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let particles = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        local.rule.config.state_dims,
        std::f32::consts::PI * 0.2_f32.powi(2),
        0.1,
    )
    .unwrap();
    let rollout = AdaptiveRolloutConfig {
        steps: 2,
        topology_enabled: false,
        bandwidth_adaptation_enabled: false,
        update_prob: 1.0,
        snapshot_interval: 2,
        ..AdaptiveRolloutConfig::default()
    };
    let local_trace = run_adaptive_rollout(&local, particles.clone(), rollout).unwrap();
    let hierarchical_trace = run_adaptive_rollout(&hierarchical, particles, rollout).unwrap();
    assert_eq!(hierarchical_trace.particles, local_trace.particles);
    assert_eq!(hierarchical_trace.metrics[0].leaf_count, 64);
    assert!(hierarchical_trace.metrics[0].proxy_nodes > 0);
    assert!(hierarchical_trace.metrics[0].proxy_messages > 0);
}

#[test]
fn adaptive_rollout_refine_coarsen_cycle_recovers_budget_and_moments() {
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.min_leaves = 64;
    config.target_leaves = 256;
    config.max_leaves = 512;
    config.topology_interval = 1;
    config.max_events_per_interval = 64;
    config.cooldown_steps = 2;
    let mut model =
        AdaptiveNpaModel::seeded(NpaModel::seeded(NpaConfig::growing_2d(), 3), config, 9).unwrap();
    model
        .controller
        .weights
        .output_weights
        .iter_mut()
        .for_each(|value| *value = 0.0);
    model.controller.weights.output_bias = vec![0.0, 0.0, 10.0, 10.0];
    let (positions, states) = seed_particles_scaled(
        1,
        64,
        model.rule.config.state_dims,
        2,
        13,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let initial = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        model.rule.config.state_dims,
        std::f32::consts::PI * 0.2_f32.powi(2),
        model.rule.config.eps0,
    )
    .unwrap();
    let initial_measure = initial.total_measure();
    let initial_centroid = weighted_centroid(&initial);
    let refined = run_adaptive_rollout(
        &model,
        initial,
        AdaptiveRolloutConfig {
            steps: 1,
            update_prob: 0.0,
            snapshot_interval: 1,
            ..AdaptiveRolloutConfig::default()
        },
    )
    .unwrap();
    assert_eq!(refined.particles.len(), 256);
    assert_eq!(refined.metrics[0].split_events, 64);

    model.config.target_leaves = 64;
    let coarsened = run_adaptive_rollout(
        &model,
        refined.particles,
        AdaptiveRolloutConfig {
            steps: 3,
            update_prob: 0.0,
            snapshot_interval: 3,
            ..AdaptiveRolloutConfig::default()
        },
    )
    .unwrap();
    assert_eq!(coarsened.particles.len(), 64);
    assert_eq!(
        coarsened
            .metrics
            .iter()
            .map(|metrics| metrics.merge_events)
            .sum::<usize>(),
        64
    );
    assert!((coarsened.particles.total_measure() - initial_measure).abs() < 1.0e-9);
    let recovered_centroid = weighted_centroid(&coarsened.particles);
    let centroid_error = initial_centroid
        .iter()
        .zip(recovered_centroid)
        .map(|(initial, recovered)| (initial - recovered).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(centroid_error < 1.0e-7, "{centroid_error}");
}

fn weighted_centroid(particles: &AdaptiveParticleSet) -> [f64; 3] {
    let total = particles.total_measure();
    let mut centroid = [0.0; 3];
    for (position, measure) in particles
        .positions
        .iter()
        .zip(&particles.represented_measure)
    {
        for axis in 0..particles.spatial_dims {
            centroid[axis] += position[axis] as f64 * *measure as f64 / total;
        }
    }
    centroid
}

#[test]
fn residual_gate_is_signed_and_tracks_hierarchy_levels() {
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.base_rule_footprint = 0.01;
    assert!(config.residual_gate(0.01).abs() <= f32::EPSILON);
    assert!((config.residual_gate(0.02) - 1.0).abs() <= 1.0e-6);
    assert!((config.residual_gate(0.04) - 2.0).abs() <= 1.0e-6);
    assert!((config.residual_gate(0.005) + 1.0).abs() <= 1.0e-6);

    config.reference_footprint = 0.02;
    config.residual_gate_reference = super::AdaptiveResidualGateReference::TargetBudget;
    assert!(config.residual_gate(0.02).abs() <= f32::EPSILON);
    assert!((config.residual_gate(0.04) - 1.0).abs() <= 1.0e-6);
    assert!((config.residual_gate(0.01) + 1.0).abs() <= 1.0e-6);
}

#[test]
fn split_state_transfer_limit_is_independent_and_backward_compatible() {
    let mut config = AdaptiveNpaConfig::growing_2d();
    config.merge_state_rms_limit = 0.05;
    assert!((config.split_state_transfer_rms_limit() - 0.05).abs() <= f32::EPSILON);

    config.split_state_transfer_rms_limit = 0.25;
    assert!((config.split_state_transfer_rms_limit() - 0.25).abs() <= f32::EPSILON);
    assert!(config.validate().is_ok());
}

#[cfg(feature = "backend_ndarray")]
#[test]
fn burn_controller_training_reduces_oracle_objective() {
    let batch = adaptive_oracle_training_batch(AdaptiveOracleDatasetConfig {
        rows: 2_048,
        // The default material measure and reference footprint represent roughly
        // 64 leaves. Keep the target there so the randomized current footprint
        // produces both split and merge examples.
        target_leaf_count: 64,
        ..AdaptiveOracleDatasetConfig::default()
    })
    .unwrap();
    let mut controller = AdaptiveController::seeded(32, 11);
    let report = train_adaptive_controller_ndarray(
        &mut controller,
        &batch,
        AdaptiveControllerTrainConfig {
            enabled: true,
            steps: 500,
            report_interval: 100,
            gradient_reduction_chunk_rows: 1_024,
            optimizer_batch_rows: 0,
            restriction_rank_boundary_emphasis: 0.0,
            restriction_rank_boundary_width: 0.125,
            restriction_topk_loss_weight: 0.0,
            restriction_topk_temperature: 0.25,
            restriction_cost_utility_weight: 0.0,
            optimizer: crate::AdamWConfig {
                learning_rate: 3.0e-3,
                grad_clip_norm: 5.0,
                ..crate::AdamWConfig::default()
            },
        },
    )
    .unwrap();
    assert!(report.final_loss < report.initial_loss * 0.35, "{report:?}");
    assert!(report.rows_per_second.is_finite() && report.rows_per_second > 0.0);
    assert!(
        report
            .event_positive_weights
            .iter()
            .all(|weight| *weight >= 1.0)
    );
    for target in batch.targets.chunks_exact(4) {
        assert!(matches!(target[2], 0.0 | 1.0));
        assert!(matches!(target[3], 0.0 | 1.0));
    }
    let prediction = controller.forward_raw(&batch.features).unwrap();
    for event in 0..2 {
        let channel = event + 2;
        let mut positives = 0usize;
        let mut true_positives = 0usize;
        for (predicted, target) in prediction
            .chunks_exact(4)
            .zip(batch.targets.chunks_exact(4))
        {
            if target[channel] >= 0.5 {
                positives += 1;
                true_positives += usize::from(predicted[channel] >= 0.0);
            }
        }
        let recall = true_positives as f32 / positives.max(1) as f32;
        assert!(positives > 0, "event {event} fixture has no positive rows");
        assert!(recall >= 0.75, "event {event} recall {recall:.3}");
    }
}

#[cfg(feature = "cli")]
#[test]
fn verified_adaptive_configs_parse_and_validate() {
    for source in [
        include_str!(
            "../../../../configs/verified/adaptive/foundation_compatibility_smoke_2d_wgpu.toml"
        ),
        include_str!(
            "../../../../configs/verified/adaptive/foundation_compatibility_full_2d_wgpu.toml"
        ),
        include_str!(
            "../../../../configs/verified/adaptive/task_multiscale_lizard_smoke_2d_wgpu.toml"
        ),
        include_str!(
            "../../../../configs/verified/adaptive/task_multiscale_lizard_full_2d_cuda.toml"
        ),
        include_str!(
            "../../../../configs/verified/adaptive/task_resident_lizard_smoke_3070_2d_wgpu.toml"
        ),
    ] {
        let config: AdaptiveExperimentConfig = toml::from_str(source).unwrap();
        config.adaptive.validate().unwrap();
        config.rollout.rollout.validate().unwrap();
        assert!(!config.graph.particle_counts.is_empty());
        assert!(config.topology.samples >= 10_000);
    }
}

#[cfg(feature = "cli")]
#[test]
fn verified_continuous_topology_audits_parse() {
    for source in [
        include_str!("../../../../configs/verified/adaptive/continuous_topology_smoke.toml"),
        include_str!("../../../../configs/verified/adaptive/continuous_topology_full.toml"),
    ] {
        let config: super::AdaptiveTopologyAuditConfig = toml::from_str(source).unwrap();
        assert!(config.topology.samples >= 10_000);
        assert!(config.topology.max_unequal_measure_ratio >= 1.0);
        assert!(!config.report_output.as_os_str().is_empty());
    }
}
