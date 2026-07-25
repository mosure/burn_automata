use std::time::Instant;

use burn_automata_kernels::{
    AdaptiveGraphPolicy, AdaptivePerceptionConfig, Boundary, HashGridConfig, HashGridMode,
    PerceptionOptions, adaptive_perceive, perceive_with_options,
};

use super::{AdaptiveOperatorExperimentConfig, AdaptiveOperatorExperimentReport};
use crate::{AutomataError, AutomataResult};

pub(super) fn run_operator_experiment(
    config: AdaptiveOperatorExperimentConfig,
    dim: usize,
) -> AutomataResult<AdaptiveOperatorExperimentReport> {
    let side = if dim == 2 {
        config.side
    } else {
        config.side_3d
    };
    if !(dim == 2 || dim == 3)
        || side < 5
        || config.sparse_side_stride == 0
        || !config.jitter.is_finite()
    {
        return Err(AutomataError::InvalidArgument(
            "adaptive operator experiment needs dimension 2/3, side >= 5, non-zero stride, and finite jitter"
                .to_string(),
        ));
    }
    let started = Instant::now();
    let spacing = 1.8 / (side - 1) as f32;
    let positions = interface_grid(side, dim, config.sparse_side_stride, config.jitter, spacing);
    let count = positions.len();
    let constant = vec![2.75; count];
    let expected_gradient = [1.25, -0.7, 0.45];
    let affine = positions
        .iter()
        .map(|position| {
            0.3 + (0..dim)
                .map(|axis| expected_gradient[axis] * position[axis])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let measures = positions
        .iter()
        .map(|position| if position[0] < 0.0 { 2.0 } else { 1.0 })
        .collect::<Vec<_>>();
    // Keep enough support in every active axis at each resolution. In particular,
    // the sparse 3D side skips rows and planes, so a fixed support can leave a
    // locally collinear stencil on smaller smoke grids.
    let fine_bandwidth = match dim {
        2 => 0.22_f32.max(1.5 * spacing),
        3 => 0.25_f32.max(1.5 * spacing),
        _ => unreachable!(),
    };
    let coarse_bandwidth = match dim {
        2 => 0.30_f32.max(1.5 * config.sparse_side_stride as f32 * spacing),
        3 => 0.34_f32.max(1.5 * config.sparse_side_stride as f32 * spacing),
        _ => unreachable!(),
    };
    let bandwidth = positions
        .iter()
        .map(|position| {
            if position[0] < 0.0 {
                coarse_bandwidth
            } else {
                fine_bandwidth
            }
        })
        .collect::<Vec<_>>();

    let fixed_grid = HashGridConfig {
        dim,
        boundary: Boundary::Clamped,
        mode: HashGridMode::Particle,
        grid_size: [64, 64, if dim == 3 { 64 } else { 1 }],
        eps: if dim == 3 { 0.30 } else { 0.26 },
        max_particles_per_block: 64,
    };
    let fixed = perceive_with_options(
        &positions,
        &constant,
        1,
        count,
        1,
        &fixed_grid,
        PerceptionOptions {
            state_grad: true,
            density_grad: true,
            eps0: fixed_grid.eps,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: false,
            log_norm_density_grad: false,
            hybrid_state_gradient: true,
            position_features: false,
        },
    )?;
    let adaptive_config = AdaptivePerceptionConfig {
        dim,
        graph_policy: AdaptiveGraphPolicy::RawSupport,
        max_neighbors: 512,
        pair_scale_power: 8.0,
        reference_measure: 0.0,
        min_bandwidth: 0.02,
        max_bandwidth: (coarse_bandwidth * 1.05).max(0.4),
        support_bin_ratio: 2.0,
        spacing_target_neighbors: if dim == 3 { 24.0 } else { 12.0 },
        spacing_root_iterations: 16,
        shepard_epsilon: 1.0e-8,
        moment_regularization: 1.0e-6,
        moment_condition_limit: 1.0e6,
        log_normalize_gradients: false,
        include_position_features: false,
    };
    let adaptive_constant = adaptive_perceive(
        &positions,
        &constant,
        &measures,
        &bandwidth,
        1,
        count,
        1,
        adaptive_config,
    )?;
    let adaptive_affine = adaptive_perceive(
        &positions,
        &affine,
        &measures,
        &bandwidth,
        1,
        count,
        1,
        adaptive_config,
    )?;

    let mut all_error = 0.0;
    let mut all_values = 0;
    let mut interface_error = 0.0;
    let mut interface_values = 0;
    for (index, position) in positions.iter().enumerate() {
        let error = (0..dim)
            .map(|axis| {
                let gradient =
                    adaptive_affine.state_gradient[index * dim + axis] / bandwidth[index];
                (gradient - expected_gradient[axis]).abs()
            })
            .sum::<f32>();
        all_error += error;
        all_values += dim;
        if position[0].abs() <= 2.0 * spacing {
            interface_error += error;
            interface_values += dim;
        }
    }
    Ok(AdaptiveOperatorExperimentReport {
        spatial_dims: dim,
        particles: count,
        fixed_constant_max_error: fixed
            .blurred_state
            .iter()
            .map(|value| (value - 2.75).abs())
            .fold(0.0_f32, f32::max),
        adaptive_constant_max_error: adaptive_constant
            .normalized_state
            .iter()
            .map(|value| (value - 2.75).abs())
            .fold(0.0_f32, f32::max),
        adaptive_affine_gradient_mean_error: all_error / all_values.max(1) as f32,
        interface_affine_gradient_mean_error: interface_error / interface_values.max(1) as f32,
        moment_fallback_fraction: adaptive_affine
            .moment_fallback
            .iter()
            .filter(|value| **value)
            .count() as f32
            / count.max(1) as f32,
        partition_min: adaptive_affine
            .partition
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        partition_max: adaptive_affine
            .partition
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn interface_grid(
    side: usize,
    dim: usize,
    sparse_stride: usize,
    jitter: f32,
    spacing: f32,
) -> Vec<[f32; 4]> {
    let z_count = if dim == 3 { side } else { 1 };
    let mut positions = Vec::new();
    for z in 0..z_count {
        for y in 0..side {
            for x in 0..side {
                if x < side / 2 && (y % sparse_stride != 0 || (dim == 3 && z % sparse_stride != 0))
                {
                    continue;
                }
                let index = (z * side + y) * side + x;
                let phase = index as f32 * 12.9898;
                positions.push([
                    -0.9 + x as f32 * spacing + jitter * phase.sin(),
                    -0.9 + y as f32 * spacing + jitter * (phase * 1.7).cos(),
                    if dim == 3 {
                        -0.9 + z as f32 * spacing + jitter * (phase * 0.73).sin()
                    } else {
                        0.0
                    },
                    0.0,
                ]);
            }
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manufactured_operator_reproduces_affine_fields_in_2d_and_3d() {
        let config = AdaptiveOperatorExperimentConfig {
            side: 17,
            side_3d: 9,
            jitter: 5.0e-4,
            sparse_side_stride: 2,
        };
        for dim in [2, 3] {
            let report = run_operator_experiment(config, dim).unwrap();
            assert!(
                report.adaptive_constant_max_error < 1.0e-5,
                "{dim}D constant error: {}",
                report.adaptive_constant_max_error
            );
            assert!(
                report.adaptive_affine_gradient_mean_error < 1.0e-3,
                "{dim}D affine error: {}",
                report.adaptive_affine_gradient_mean_error
            );
            assert_eq!(
                report.moment_fallback_fraction, 0.0,
                "{dim}D moment correction unexpectedly fell back"
            );
        }
    }
}
