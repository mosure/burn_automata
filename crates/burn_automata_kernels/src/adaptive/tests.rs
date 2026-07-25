use super::*;
use crate::{
    Boundary, HashGridConfig, HashGridMode, PerceptionOptions, perceive_state_adjoint_with_options,
    perceive_with_options,
};

fn regular_grid_2d(side: usize, jitter: f32) -> Vec<[f32; 4]> {
    let spacing = 1.6 / (side - 1) as f32;
    (0..side * side)
        .map(|index| {
            let x = index % side;
            let y = index / side;
            let phase = index as f32 * 12.9898;
            [
                -0.8 + x as f32 * spacing + jitter * phase.sin(),
                -0.8 + y as f32 * spacing + jitter * (phase * 1.7).cos(),
                0.0,
                0.0,
            ]
        })
        .collect()
}

#[test]
fn compatible_operator_matches_fixed_npa_at_uniform_measure_and_bandwidth() {
    let particle_count = 64;
    let state_dims = 5;
    let positions = (0..particle_count)
        .map(|index| {
            let angle = index as f32 * 2.399_963_1;
            let radius = 0.015 * (index as f32).sqrt();
            [radius * angle.cos(), radius * angle.sin(), 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.173).sin())
        .collect::<Vec<_>>();
    let grid = HashGridConfig {
        dim: 2,
        boundary: Boundary::Clamped,
        mode: HashGridMode::Particle,
        grid_size: [64, 64, 1],
        eps: 0.1,
        max_particles_per_block: 128,
    };
    let fixed_options = PerceptionOptions {
        state_grad: true,
        density_grad: true,
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        hybrid_state_gradient: true,
        position_features: false,
    };
    let fixed = perceive_with_options(
        &positions,
        &states,
        1,
        particle_count,
        state_dims,
        &grid,
        fixed_options,
    )
    .unwrap();
    let adaptive = adaptive_npa_perceive_all_pairs(
        &positions,
        &states,
        &vec![0.125 / particle_count as f32; particle_count],
        &vec![0.1; particle_count],
        1,
        particle_count,
        state_dims,
        AdaptivePerceptionConfig {
            dim: 2,
            graph_policy: AdaptiveGraphPolicy::RawSupport,
            max_neighbors: particle_count,
            pair_scale_power: 8.0,
            reference_measure: 0.0,
            min_bandwidth: 0.1,
            max_bandwidth: 0.1,
            support_bin_ratio: 2.0,
            spacing_target_neighbors: 8.0,
            spacing_root_iterations: 8,
            shepard_epsilon: 1.0e-8,
            moment_regularization: 0.0,
            moment_condition_limit: 1.0e8,
            log_normalize_gradients: true,
            include_position_features: false,
        },
        AdaptiveNpaPerceptionOptions {
            eps0: fixed_options.eps0,
            scale_equivariance: fixed_options.scale_equivariance,
            particle_density_equivariance: fixed_options.particle_density_equivariance,
            log_norm_grad: fixed_options.log_norm_grad,
            log_norm_density_grad: fixed_options.log_norm_density_grad,
            position_features: fixed_options.position_features,
        },
    )
    .unwrap();
    assert_vectors_close(&adaptive.normalized_state, &fixed.blurred_state, 2.0e-5);
    assert_vectors_close(&adaptive.state_gradient, &fixed.state_gradient, 2.0e-5);
    assert_vectors_close(
        &adaptive.occupancy_gradient,
        &fixed.density_gradient,
        2.0e-5,
    );
    assert_vectors_close(&adaptive.features, &fixed.features, 2.0e-5);
    assert!(adaptive.coarse_exposure.iter().all(|value| *value == 0.0));
}

#[test]
fn compatible_coarse_exposure_is_local_and_density_weighted() {
    let positions = [
        [0.0, 0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0, 0.0],
        [0.8, 0.8, 0.0, 0.0],
    ];
    let states = vec![0.0; positions.len() * 2];
    let output = adaptive_npa_perceive_all_pairs(
        &positions,
        &states,
        &[1.0, 4.0, 1.0],
        &[0.1, 0.1, 0.1],
        1,
        positions.len(),
        2,
        AdaptivePerceptionConfig {
            dim: 2,
            graph_policy: AdaptiveGraphPolicy::RawSupport,
            max_neighbors: positions.len(),
            pair_scale_power: 8.0,
            reference_measure: 1.0,
            min_bandwidth: 0.1,
            max_bandwidth: 0.1,
            support_bin_ratio: 2.0,
            spacing_target_neighbors: 2.0,
            spacing_root_iterations: 4,
            shepard_epsilon: 1.0e-8,
            moment_regularization: 0.0,
            moment_condition_limit: 1.0e8,
            log_normalize_gradients: true,
            include_position_features: false,
        },
        AdaptiveNpaPerceptionOptions {
            eps0: 0.1,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            position_features: false,
        },
    )
    .unwrap();
    assert!(output.coarse_exposure[0] > 0.0);
    assert!(output.coarse_exposure[0] < 1.0);
    assert!(output.coarse_exposure[1] > output.coarse_exposure[0]);
    assert_eq!(output.coarse_exposure[2], 0.0);
}

fn assert_vectors_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum <= tolerance,
        "maximum error {maximum} > {tolerance}"
    );
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn assert_adjoint_matches_finite_difference(
    states: &[f32],
    analytical: &[f32],
    epsilon: f32,
    tolerance: f32,
    mut objective: impl FnMut(&[f32]) -> f32,
) {
    assert_eq!(states.len(), analytical.len());
    let mut perturbed = states.to_vec();
    let mut maximum_error = 0.0_f32;
    let mut maximum_scale = 0.0_f32;
    for index in 0..states.len() {
        perturbed[index] = states[index] + epsilon;
        let positive = objective(&perturbed);
        perturbed[index] = states[index] - epsilon;
        let negative = objective(&perturbed);
        perturbed[index] = states[index];
        let numerical = (positive - negative) / (2.0 * epsilon);
        maximum_error = maximum_error.max((analytical[index] - numerical).abs());
        maximum_scale = maximum_scale.max(analytical[index].abs().max(numerical.abs()));
    }
    let relative_error = maximum_error / maximum_scale.max(1.0);
    assert!(
        relative_error <= tolerance,
        "adjoint relative error {relative_error} (absolute {maximum_error}, scale {maximum_scale}) \
         exceeds {tolerance}"
    );
}

#[test]
fn compatible_state_adjoint_matches_fixed_npa_at_uniform_scale() {
    let particle_count = 64;
    let state_dims = 4;
    let positions = (0..particle_count)
        .map(|index| {
            let angle = index as f32 * 2.399_963_1;
            let radius = 0.015 * (index as f32).sqrt();
            [radius * angle.cos(), radius * angle.sin(), 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.173).sin())
        .collect::<Vec<_>>();
    let grid = HashGridConfig {
        dim: 2,
        boundary: Boundary::Clamped,
        mode: HashGridMode::Particle,
        grid_size: [64, 64, 1],
        eps: 0.1,
        max_particles_per_block: 128,
    };
    let fixed_options = PerceptionOptions {
        state_grad: true,
        density_grad: true,
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        hybrid_state_gradient: true,
        position_features: false,
    };
    let adaptive_config = AdaptivePerceptionConfig {
        dim: 2,
        graph_policy: AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        pair_scale_power: 8.0,
        reference_measure: 0.0,
        min_bandwidth: 0.1,
        max_bandwidth: 0.1,
        support_bin_ratio: 2.0,
        spacing_target_neighbors: 8.0,
        spacing_root_iterations: 8,
        shepard_epsilon: 1.0e-8,
        moment_regularization: 0.0,
        moment_condition_limit: 1.0e8,
        log_normalize_gradients: true,
        include_position_features: false,
    };
    let adaptive_options = AdaptiveNpaPerceptionOptions {
        eps0: fixed_options.eps0,
        scale_equivariance: fixed_options.scale_equivariance,
        particle_density_equivariance: fixed_options.particle_density_equivariance,
        log_norm_grad: fixed_options.log_norm_grad,
        log_norm_density_grad: fixed_options.log_norm_density_grad,
        position_features: fixed_options.position_features,
    };
    let feature_dims = adaptive_config.feature_dims(state_dims);
    let feature_adjoint = (0..particle_count * feature_dims)
        .map(|index| (index as f32 * 0.071).sin())
        .collect::<Vec<_>>();
    let fixed = perceive_state_adjoint_with_options(
        &positions,
        &states,
        1,
        particle_count,
        state_dims,
        &grid,
        fixed_options,
        &feature_adjoint,
    )
    .unwrap();
    let adaptive = adaptive_npa_perceive_state_adjoint_all_pairs(
        &positions,
        &states,
        &vec![0.125 / particle_count as f32; particle_count],
        &vec![0.1; particle_count],
        1,
        particle_count,
        state_dims,
        adaptive_config,
        adaptive_options,
        &feature_adjoint,
    )
    .unwrap();
    assert_vectors_close(&adaptive, &fixed, 3.0e-5);
}

#[test]
fn compatible_state_adjoint_matches_finite_difference_at_mixed_scale() {
    let positions = regular_grid_2d(4, 0.003);
    let particle_count = positions.len();
    let state_dims = 3;
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.217).sin())
        .collect::<Vec<_>>();
    let measures = (0..particle_count)
        .map(|index| 0.4 + (index % 5) as f32 * 0.17)
        .collect::<Vec<_>>();
    let bandwidth = (0..particle_count)
        .map(|index| 0.48 + (index % 3) as f32 * 0.07)
        .collect::<Vec<_>>();
    let config = AdaptivePerceptionConfig {
        graph_policy: AdaptiveGraphPolicy::DirectedTopK,
        max_neighbors: 10,
        min_bandwidth: 0.4,
        max_bandwidth: 0.7,
        spacing_target_neighbors: 4.0,
        moment_condition_limit: 1.0e8,
        ..AdaptivePerceptionConfig::growing_2d()
    };
    let options = AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let feature_adjoint = (0..particle_count * config.feature_dims(state_dims))
        .map(|index| (index as f32 * 0.113).cos())
        .collect::<Vec<_>>();
    let analytical = adaptive_npa_perceive_state_adjoint_all_pairs(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        particle_count,
        state_dims,
        config,
        options,
        &feature_adjoint,
    )
    .unwrap();
    assert_adjoint_matches_finite_difference(
        &states,
        &analytical,
        2.0e-3,
        2.0e-3,
        |candidate_states| {
            let output = adaptive_npa_perceive_all_pairs(
                &positions,
                candidate_states,
                &measures,
                &bandwidth,
                1,
                particle_count,
                state_dims,
                config,
                options,
            )
            .unwrap();
            dot(&output.features, &feature_adjoint)
        },
    );
}

#[test]
fn normalized_state_adjoint_matches_finite_difference_at_mixed_scale() {
    let positions = regular_grid_2d(4, 0.002);
    let particle_count = positions.len();
    let state_dims = 3;
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.193).cos())
        .collect::<Vec<_>>();
    let measures = (0..particle_count)
        .map(|index| 0.35 + (index % 4) as f32 * 0.23)
        .collect::<Vec<_>>();
    let bandwidth = (0..particle_count)
        .map(|index| 0.47 + (index % 3) as f32 * 0.075)
        .collect::<Vec<_>>();
    let config = AdaptivePerceptionConfig {
        graph_policy: AdaptiveGraphPolicy::DirectedTopK,
        max_neighbors: 9,
        min_bandwidth: 0.4,
        max_bandwidth: 0.7,
        spacing_target_neighbors: 4.0,
        moment_condition_limit: 1.0e8,
        ..AdaptivePerceptionConfig::growing_2d()
    };
    let feature_adjoint = (0..particle_count * config.feature_dims(state_dims))
        .map(|index| (index as f32 * 0.097).sin())
        .collect::<Vec<_>>();
    let analytical = adaptive_perceive_state_adjoint_all_pairs(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        particle_count,
        state_dims,
        config,
        &feature_adjoint,
    )
    .unwrap();
    assert_adjoint_matches_finite_difference(
        &states,
        &analytical,
        2.0e-3,
        2.0e-3,
        |candidate_states| {
            let output = adaptive_perceive_all_pairs(
                &positions,
                candidate_states,
                &measures,
                &bandwidth,
                1,
                particle_count,
                state_dims,
                config,
            )
            .unwrap();
            dot(&output.features, &feature_adjoint)
        },
    );
}

#[test]
fn adaptive_state_adjoint_spatial_hash_matches_all_pairs() {
    let positions = regular_grid_2d(6, 0.001);
    let particle_count = positions.len();
    let state_dims = 2;
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.181).sin())
        .collect::<Vec<_>>();
    let measures = (0..particle_count)
        .map(|index| 0.5 + (index % 3) as f32 * 0.2)
        .collect::<Vec<_>>();
    let bandwidth = (0..particle_count)
        .map(|index| 0.31 + (index % 4) as f32 * 0.04)
        .collect::<Vec<_>>();
    let config = AdaptivePerceptionConfig {
        graph_policy: AdaptiveGraphPolicy::MutualTopK,
        max_neighbors: 12,
        min_bandwidth: 0.25,
        max_bandwidth: 0.45,
        spacing_target_neighbors: 5.0,
        ..AdaptivePerceptionConfig::growing_2d()
    };
    let options = AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let feature_adjoint = (0..particle_count * config.feature_dims(state_dims))
        .map(|index| (index as f32 * 0.131).cos())
        .collect::<Vec<_>>();
    let hashed = adaptive_npa_perceive_state_adjoint(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        particle_count,
        state_dims,
        config,
        options,
        &feature_adjoint,
    )
    .unwrap();
    let all_pairs = adaptive_npa_perceive_state_adjoint_all_pairs(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        particle_count,
        state_dims,
        config,
        options,
        &feature_adjoint,
    )
    .unwrap();
    assert_eq!(hashed, all_pairs);
}

#[test]
fn compatible_rule_only_path_preserves_every_rule_feature() {
    let positions = regular_grid_2d(13, 0.0007);
    let count = positions.len();
    let state_dims = 4;
    let states = (0..count * state_dims)
        .map(|index| (index as f32 * 0.137).sin())
        .collect::<Vec<_>>();
    let measures = (0..count)
        .map(|index| 0.5 + (index % 5) as f32 * 0.125)
        .collect::<Vec<_>>();
    let bandwidth = (0..count)
        .map(|index| if index % 3 == 0 { 0.18 } else { 0.24 })
        .collect::<Vec<_>>();
    let config = AdaptivePerceptionConfig {
        graph_policy: AdaptiveGraphPolicy::DirectedTopK,
        max_neighbors: 32,
        min_bandwidth: 0.05,
        max_bandwidth: 0.3,
        spacing_target_neighbors: 12.0,
        ..AdaptivePerceptionConfig::growing_2d()
    };
    let options = AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let full = adaptive_perceive_pair(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        count,
        state_dims,
        config,
        AdaptiveGraphPolicy::RawSupport,
        options,
    )
    .unwrap()
    .npa_compatible;
    let mut rule_config = config;
    rule_config.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let rule_only = adaptive_npa_perceive_without_spacing(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        count,
        state_dims,
        rule_config,
        options,
    )
    .unwrap();

    assert_eq!(rule_only.features, full.features);
    assert_eq!(rule_only.normalized_state, full.normalized_state);
    assert_eq!(rule_only.state_gradient, full.state_gradient);
    assert_eq!(rule_only.occupancy_gradient, full.occupancy_gradient);
    assert_eq!(rule_only.partition, full.partition);
    assert_eq!(rule_only.moment_condition, full.moment_condition);
    assert_eq!(rule_only.moment_fallback, full.moment_fallback);
    assert_eq!(rule_only.accepted_degree, full.accepted_degree);
    let mut rule_semantics = rule_only.graph.clone();
    let mut full_semantics = full.graph.clone();
    rule_semantics.candidate_visits = 0;
    full_semantics.candidate_visits = 0;
    assert_eq!(rule_semantics, full_semantics);
    assert_eq!(rule_only.observed_spacing, bandwidth);
}

#[test]
fn normalized_shepard_reproduces_constants_across_density_interface() {
    let mut positions = regular_grid_2d(17, 0.001);
    positions.retain(|position| position[0] >= 0.0 || ((position[1] * 100.0) as i32) % 2 == 0);
    let count = positions.len();
    let states = vec![2.75; count];
    let measures = positions
        .iter()
        .map(|position| if position[0] < 0.0 { 2.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let bandwidth = positions
        .iter()
        .map(|position| if position[0] < 0.0 { 0.32 } else { 0.23 })
        .collect::<Vec<_>>();
    let output = adaptive_perceive(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        count,
        1,
        AdaptivePerceptionConfig {
            graph_policy: AdaptiveGraphPolicy::RawSupport,
            log_normalize_gradients: false,
            ..AdaptivePerceptionConfig::growing_2d()
        },
    )
    .unwrap();
    let max_error = output
        .normalized_state
        .iter()
        .map(|value| (value - 2.75).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error < 2.0e-5,
        "constant reproduction error {max_error}"
    );
}

#[test]
fn corrected_gradient_recovers_affine_field_interior() {
    let positions = regular_grid_2d(21, 0.0005);
    let count = positions.len();
    let states = positions
        .iter()
        .map(|position| 0.3 + 1.25 * position[0] - 0.7 * position[1])
        .collect::<Vec<_>>();
    let measures = vec![1.0; count];
    let bandwidth = vec![0.24; count];
    let output = adaptive_perceive(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        count,
        1,
        AdaptivePerceptionConfig {
            graph_policy: AdaptiveGraphPolicy::RawSupport,
            log_normalize_gradients: false,
            moment_regularization: 1.0e-6,
            ..AdaptivePerceptionConfig::growing_2d()
        },
    )
    .unwrap();
    let mut error_sum = 0.0;
    let mut samples = 0;
    for (index, position) in positions.iter().enumerate() {
        if position[0].abs() < 0.55 && position[1].abs() < 0.55 {
            let gx = output.state_gradient[index * 2] / bandwidth[index];
            let gy = output.state_gradient[index * 2 + 1] / bandwidth[index];
            error_sum += (gx - 1.25).abs() + (gy + 0.7).abs();
            samples += 2;
        }
    }
    let mean_error = error_sum / samples as f32;
    assert!(
        mean_error < 2.0e-3,
        "affine gradient mean error {mean_error}"
    );
}

#[test]
fn directed_topk_enforces_hard_degree_budget() {
    let positions = regular_grid_2d(20, 0.0);
    let count = positions.len();
    let output = adaptive_perceive(
        &positions,
        &vec![0.0; count * 2],
        &vec![1.0; count],
        &vec![0.4; count],
        1,
        count,
        2,
        AdaptivePerceptionConfig {
            max_neighbors: 16,
            ..AdaptivePerceptionConfig::growing_2d()
        },
    )
    .unwrap();
    assert!(output.graph.raw_messages > output.graph.accepted_messages);
    assert!(output.graph.degree_max <= 16);
    assert_eq!(
        output.graph.accepted_messages,
        output.accepted_degree.iter().sum::<usize>()
    );
}

#[test]
fn batch_rows_never_cross_connect() {
    let positions = vec![
        [0.0, 0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0, 0.0],
    ];
    let states = vec![1.0, 1.0, 9.0, 9.0];
    let output = adaptive_perceive(
        &positions,
        &states,
        &[1.0; 4],
        &[0.2; 4],
        2,
        2,
        1,
        AdaptivePerceptionConfig::growing_2d(),
    )
    .unwrap();
    assert!(
        output.normalized_state[..2]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-6)
    );
    assert!(
        output.normalized_state[2..]
            .iter()
            .all(|value| (*value - 9.0).abs() < 1e-6)
    );
}

#[test]
fn spatial_hash_matches_all_pairs_across_policies_and_dimensions() {
    for dim in [2, 3] {
        let particle_count = 96;
        let batch_size = 2;
        let total = batch_size * particle_count;
        let positions = (0..total)
            .map(|index| {
                let batch = index / particle_count;
                let local = index % particle_count;
                let phase = local as f32 * 0.754_877_7 + batch as f32 * 0.31;
                [
                    phase.sin() * 0.83,
                    (phase * 1.731).cos() * 0.79,
                    if dim == 3 {
                        (phase * 0.417).sin() * 0.74
                    } else {
                        0.0
                    },
                    0.0,
                ]
            })
            .collect::<Vec<_>>();
        let states = (0..total * 3)
            .map(|index| (index as f32 * 0.117).sin())
            .collect::<Vec<_>>();
        let measures = (0..total)
            .map(|index| 0.25 + (index % 7) as f32 * 0.17)
            .collect::<Vec<_>>();
        let bandwidth = (0..total)
            .map(|index| {
                let log_fraction = (index % 31) as f32 / 30.0;
                0.01875 * 16.0_f32.powf(log_fraction)
            })
            .collect::<Vec<_>>();

        for graph_policy in [
            AdaptiveGraphPolicy::RawSupport,
            AdaptiveGraphPolicy::DirectedTopK,
            AdaptiveGraphPolicy::MutualTopK,
        ] {
            let cfg = AdaptivePerceptionConfig {
                dim,
                graph_policy,
                max_neighbors: 12,
                min_bandwidth: 0.01875,
                max_bandwidth: 0.3,
                log_normalize_gradients: false,
                include_position_features: true,
                ..AdaptivePerceptionConfig::growing_2d()
            };
            let hashed = adaptive_perceive(
                &positions,
                &states,
                &measures,
                &bandwidth,
                batch_size,
                particle_count,
                3,
                cfg,
            )
            .unwrap();
            let all_pairs = adaptive_perceive_all_pairs(
                &positions,
                &states,
                &measures,
                &bandwidth,
                batch_size,
                particle_count,
                3,
                cfg,
            )
            .unwrap();

            assert_eq!(hashed.features, all_pairs.features);
            assert_eq!(hashed.normalized_state, all_pairs.normalized_state);
            assert_eq!(hashed.state_gradient, all_pairs.state_gradient);
            assert_eq!(hashed.occupancy_gradient, all_pairs.occupancy_gradient);
            assert_eq!(hashed.partition, all_pairs.partition);
            assert_eq!(hashed.observed_spacing, all_pairs.observed_spacing);
            assert_eq!(hashed.moment_condition, all_pairs.moment_condition);
            assert_eq!(hashed.moment_fallback, all_pairs.moment_fallback);
            assert_eq!(hashed.accepted_degree, all_pairs.accepted_degree);
            assert!(hashed.graph.candidate_visits <= all_pairs.graph.candidate_visits);
            let mut hashed_semantics = hashed.graph.clone();
            let mut all_pairs_semantics = all_pairs.graph.clone();
            hashed_semantics.candidate_visits = 0;
            all_pairs_semantics.candidate_visits = 0;
            assert_eq!(hashed_semantics, all_pairs_semantics);
            assert_eq!(hashed.feature_dims, all_pairs.feature_dims);
        }
    }
}

#[test]
fn finer_support_bins_reduce_broad_phase_work_without_changing_perception() {
    let side = 28;
    let positions = regular_grid_2d(side, 0.001);
    let count = positions.len();
    let states = (0..count * 2)
        .map(|index| (index as f32 * 0.071).sin())
        .collect::<Vec<_>>();
    let measures = vec![1.0 / count as f32; count];
    let bandwidth = (0..count)
        .map(|index| {
            let fraction = (index % side) as f32 / (side - 1) as f32;
            0.025 * 8.0_f32.powf(fraction)
        })
        .collect::<Vec<_>>();
    let base = AdaptivePerceptionConfig {
        graph_policy: AdaptiveGraphPolicy::RawSupport,
        max_neighbors: count,
        min_bandwidth: 0.025,
        max_bandwidth: 0.2,
        spacing_target_neighbors: 8.0,
        ..AdaptivePerceptionConfig::growing_2d()
    };
    let dyadic = adaptive_perceive(
        &positions, &states, &measures, &bandwidth, 1, count, 2, base,
    )
    .unwrap();
    let fine = adaptive_perceive(
        &positions,
        &states,
        &measures,
        &bandwidth,
        1,
        count,
        2,
        AdaptivePerceptionConfig {
            support_bin_ratio: 2.0_f32.sqrt(),
            ..base
        },
    )
    .unwrap();
    let all_pairs = adaptive_perceive_all_pairs(
        &positions, &states, &measures, &bandwidth, 1, count, 2, base,
    )
    .unwrap();

    assert_eq!(fine.features, dyadic.features);
    assert_eq!(fine.features, all_pairs.features);
    assert_eq!(fine.partition, dyadic.partition);
    assert_eq!(fine.accepted_degree, dyadic.accepted_degree);
    assert_eq!(fine.graph.raw_messages, dyadic.graph.raw_messages);
    eprintln!(
        "support-bin candidate visits: all-pairs={} dyadic={} sqrt2={} accepted={}",
        all_pairs.graph.candidate_visits,
        dyadic.graph.candidate_visits,
        fine.graph.candidate_visits,
        fine.graph.raw_messages,
    );
    assert!(
        fine.graph.candidate_visits < dyadic.graph.candidate_visits,
        "finer bins visited {} candidates versus {} for dyadic bins",
        fine.graph.candidate_visits,
        dyadic.graph.candidate_visits,
    );
}
