//! Reference kernels and shared kernel-facing types for Neural Particle Automata.
//!
//! The CPU implementation is deliberately simple and deterministic. It is the
//! correctness oracle for future CubeCL/WGPU kernels.

pub mod config;
pub mod gaussian;
pub mod hashgrid;
pub mod reference;
pub mod spatial;
pub mod splat;
pub mod tile;

pub use config::{Boundary, HashGridConfig, HashGridMode, KernelError, KernelResult};
pub use gaussian::{Gaussian3d, GaussianDecodeConfig, GaussianDecodeMode, decode_gaussians_3d};
pub use hashgrid::{HashGridSnapshot, build_hashgrid};
pub use reference::{
    PerceptionAdjointOutput, PerceptionOptions, PerceptionOutput, euler_step, perceive,
    perceive_adjoint_with_options, perceive_state_adjoint_with_options, perceive_with_options,
};
pub use spatial::{SpatialStrategyKind, SpatialStrategyReport, analyze_spatial_strategy};
pub use splat::{Splat2dConfig, splat_particles_2d};
pub use tile::{TileAssignment, TileGridConfig, assign_tiles, tile_for_position};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashgrid_counts_particles_per_cell() {
        let cfg = HashGridConfig {
            grid_size: [4, 4, 1],
            ..HashGridConfig::growing_2d()
        };
        let positions = [[-0.9, -0.9, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0]];
        let grid = build_hashgrid(&positions, 1, 2, &cfg).unwrap();

        assert_eq!(grid.bin_offsets.len(), cfg.cell_count() + 1);
        assert_eq!(grid.permutation.len(), 2);
        assert!(grid.bin_offsets.windows(2).any(|w| w[1] > w[0]));
    }

    #[test]
    fn perceive_returns_finite_reference_features() {
        let cfg = HashGridConfig::growing_2d();
        let positions = [[-0.05, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]];
        let states = vec![1.0, 0.0, 0.0, 1.0];
        let out = perceive(&positions, &states, 1, 2, 2, &cfg, true, true).unwrap();

        assert_eq!(out.feature_dims, 2 * 2 + 2 * cfg.dim + cfg.dim);
        assert_eq!(out.features.len(), 2 * out.feature_dims);
        assert!(out.features.iter().all(|v| v.is_finite()));
        assert!(out.density.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn perceive_matches_upstream_sph_kernel_constants() {
        let cfg = HashGridConfig {
            grid_size: [4, 4, 1],
            eps: 0.1,
            ..HashGridConfig::growing_2d()
        };
        let positions = [[0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]];
        let states = vec![1.0, 0.0, 0.0, 1.0];
        let out = perceive(&positions, &states, 1, 2, 2, &cfg, true, true).unwrap();

        assert_close(out.density[0], 181.038_74, 1e-3);
        assert_close(out.density[1], 181.038_74, 1e-3);
        assert_close(out.blurred_state[0], 0.703_296_7, 1e-5);
        assert_close(out.blurred_state[1], 0.296_703_3, 1e-5);
        assert_close(out.blurred_state[2], 0.296_703_3, 1e-5);
        assert_close(out.blurred_state[3], 0.703_296_7, 1e-5);
        assert_close(out.state_gradient[0], -13.186_813, 1e-4);
        assert_close(out.state_gradient[2], 13.186_813, 1e-4);
        assert_close(out.density_gradient[0], 1_193.662, 1e-3);
        assert_close(out.density_gradient[2], -1_193.662, 1e-3);
    }

    #[test]
    fn perceive_state_adjoint_matches_finite_difference() {
        let cfg = HashGridConfig {
            dim: 3,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Particle,
            grid_size: [8, 8, 8],
            eps: 0.35,
            max_particles_per_block: 16,
        };
        let positions = [
            [0.00, 0.00, 0.00, 0.0],
            [0.12, 0.04, -0.03, 0.0],
            [-0.08, 0.10, 0.02, 0.0],
            [0.04, -0.09, 0.11, 0.0],
        ];
        let state_dims = 3;
        let states = vec![
            0.10, -0.20, 0.05, //
            0.30, 0.15, -0.10, //
            -0.25, 0.35, 0.20, //
            0.05, -0.30, 0.40,
        ];
        let options = PerceptionOptions {
            state_grad: true,
            density_grad: true,
            eps0: cfg.eps,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            hybrid_state_gradient: true,
            position_features: false,
        };
        let perception =
            perceive_with_options(&positions, &states, 1, 4, state_dims, &cfg, options).unwrap();
        let feature_adjoint = (0..perception.features.len())
            .map(|idx| ((idx as f32 + 1.0) * 0.017).sin() * 0.1)
            .collect::<Vec<_>>();
        let state_adjoint = perceive_state_adjoint_with_options(
            &positions,
            &states,
            1,
            4,
            state_dims,
            &cfg,
            options,
            &feature_adjoint,
        )
        .unwrap();

        let channel = 4;
        let eps = 1.0e-3;
        let mut plus_states = states.clone();
        plus_states[channel] += eps;
        let plus = perceive_with_options(&positions, &plus_states, 1, 4, state_dims, &cfg, options)
            .unwrap();
        let mut minus_states = states.clone();
        minus_states[channel] -= eps;
        let minus =
            perceive_with_options(&positions, &minus_states, 1, 4, state_dims, &cfg, options)
                .unwrap();
        let plus_loss = plus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let minus_loss = minus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
        assert_close(state_adjoint[channel], finite_difference, 2.0e-3);
    }

    #[test]
    fn perceive_position_adjoint_matches_finite_difference_without_hybrid_moment() {
        let cfg = HashGridConfig {
            dim: 3,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Particle,
            grid_size: [8, 8, 8],
            eps: 0.40,
            max_particles_per_block: 16,
        };
        let positions = [
            [0.00, 0.00, 0.00, 0.0],
            [0.11, 0.03, -0.02, 0.0],
            [-0.07, 0.09, 0.04, 0.0],
            [0.05, -0.08, 0.10, 0.0],
        ];
        let state_dims = 3;
        let states = vec![
            0.10, -0.20, 0.05, //
            0.30, 0.15, -0.10, //
            -0.25, 0.35, 0.20, //
            0.05, -0.30, 0.40,
        ];
        let options = PerceptionOptions {
            state_grad: true,
            density_grad: true,
            eps0: cfg.eps,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            hybrid_state_gradient: false,
            position_features: true,
        };
        let perception =
            perceive_with_options(&positions, &states, 1, 4, state_dims, &cfg, options).unwrap();
        let feature_adjoint = (0..perception.features.len())
            .map(|idx| ((idx as f32 + 3.0) * 0.013).cos() * 0.05)
            .collect::<Vec<_>>();
        let adjoint = perceive_adjoint_with_options(
            &positions,
            &states,
            1,
            4,
            state_dims,
            &cfg,
            options,
            &feature_adjoint,
        )
        .unwrap();

        let particle = 1;
        let axis = 0;
        let eps = 1.0e-4;
        let mut plus_positions = positions;
        plus_positions[particle][axis] += eps;
        let plus = perceive_with_options(&plus_positions, &states, 1, 4, state_dims, &cfg, options)
            .unwrap();
        let mut minus_positions = positions;
        minus_positions[particle][axis] -= eps;
        let minus =
            perceive_with_options(&minus_positions, &states, 1, 4, state_dims, &cfg, options)
                .unwrap();
        let plus_loss = plus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let minus_loss = minus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
        assert_close(adjoint.position[particle][axis], finite_difference, 7.5e-2);
    }

    #[test]
    fn perceive_position_adjoint_matches_finite_difference_with_hybrid_moment() {
        let cfg = HashGridConfig {
            dim: 3,
            boundary: Boundary::Clamped,
            mode: HashGridMode::Particle,
            grid_size: [8, 8, 8],
            eps: 0.40,
            max_particles_per_block: 16,
        };
        let positions = [
            [0.00, 0.00, 0.00, 0.0],
            [0.11, 0.03, -0.02, 0.0],
            [-0.07, 0.09, 0.04, 0.0],
            [0.05, -0.08, 0.10, 0.0],
        ];
        let state_dims = 3;
        let states = vec![
            0.10, -0.20, 0.05, //
            0.30, 0.15, -0.10, //
            -0.25, 0.35, 0.20, //
            0.05, -0.30, 0.40,
        ];
        let options = PerceptionOptions {
            state_grad: true,
            density_grad: true,
            eps0: cfg.eps,
            scale_equivariance: true,
            particle_density_equivariance: true,
            log_norm_grad: true,
            log_norm_density_grad: true,
            hybrid_state_gradient: true,
            position_features: true,
        };
        let perception =
            perceive_with_options(&positions, &states, 1, 4, state_dims, &cfg, options).unwrap();
        let feature_adjoint = (0..perception.features.len())
            .map(|idx| ((idx as f32 + 5.0) * 0.011).sin() * 0.03)
            .collect::<Vec<_>>();
        let adjoint = perceive_adjoint_with_options(
            &positions,
            &states,
            1,
            4,
            state_dims,
            &cfg,
            options,
            &feature_adjoint,
        )
        .unwrap();

        let particle = 2;
        let axis = 1;
        let eps = 1.0e-4;
        let mut plus_positions = positions;
        plus_positions[particle][axis] += eps;
        let plus = perceive_with_options(&plus_positions, &states, 1, 4, state_dims, &cfg, options)
            .unwrap();
        let mut minus_positions = positions;
        minus_positions[particle][axis] -= eps;
        let minus =
            perceive_with_options(&minus_positions, &states, 1, 4, state_dims, &cfg, options)
                .unwrap();
        let plus_loss = plus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let minus_loss = minus
            .features
            .iter()
            .zip(feature_adjoint.iter())
            .map(|(feature, adjoint)| feature * adjoint)
            .sum::<f32>();
        let finite_difference = (plus_loss - minus_loss) / (2.0 * eps);
        assert_close(adjoint.position[particle][axis], finite_difference, 2.5e-1);
    }

    #[test]
    fn euler_step_applies_masks_and_boundaries() {
        let cfg = HashGridConfig::growing_2d();
        let positions = [[0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]];
        let states = vec![0.0, 1.0];
        let dx = [[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]];
        let ds = vec![1.0, 1.0];
        let mask = [1.0, 0.0];
        let (next_pos, next_state) = euler_step(
            &positions,
            &states,
            &dx,
            &ds,
            1,
            2,
            1,
            &cfg,
            1.0,
            Some(&mask),
        )
        .unwrap();

        assert_eq!(next_pos[0][0], 0.5);
        assert_eq!(next_pos[1][0], 0.9);
        assert_eq!(next_state, vec![1.0, 1.0]);
    }

    #[test]
    fn euler_step_clamps_3d_opacity_logit() {
        let cfg = HashGridConfig::growing_3dgs();
        let positions = [[0.0, 0.0, 0.0, 0.0]];
        let states = vec![0.0, 0.0, 0.0, 23.0];
        let dx = [[0.0, 0.0, 0.0, 0.0]];
        let ds = vec![0.0, 0.0, 0.0, 10.0];
        let (_next_pos, next_state) =
            euler_step(&positions, &states, &dx, &ds, 1, 1, 4, &cfg, 1.0, None).unwrap();

        assert_eq!(next_state[3], 24.0);

        let states = vec![0.0, 0.0, 0.0, -7.0];
        let ds = vec![0.0, 0.0, 0.0, -10.0];
        let (_next_pos, next_state) =
            euler_step(&positions, &states, &dx, &ds, 1, 1, 4, &cfg, 1.0, None).unwrap();

        assert_eq!(next_state[3], -8.0);
    }

    #[test]
    fn splat_and_gaussian_decode_are_shape_stable() {
        let positions = [[0.0, 0.0, 0.0, 0.0]];
        let image = splat_particles_2d(
            &positions,
            &[[1.0, 0.0, 0.0]],
            Splat2dConfig {
                image_size: 8,
                ..Splat2dConfig::default()
            },
        );
        assert_eq!(image.len(), 64);
        assert!(image.iter().any(|px| px[3] > 0.0));

        let state_dims = 20;
        let mut states = vec![0.0; state_dims];
        states[state_dims - 3] = 0.4;
        states[state_dims - 2] = -0.2;
        states[state_dims - 1] = 0.1;
        let gaussians = decode_gaussians_3d(
            &[[0.0, 0.0, 0.0, 0.0]],
            &states,
            state_dims,
            GaussianDecodeConfig::default(),
        );
        assert_eq!(gaussians.len(), 1);
        assert_eq!(gaussians[0].spherical_harmonic.len(), 1);
        assert_eq!(gaussians[0].spherical_harmonic[0], [0.9, 0.3, 0.6]);
        assert!(gaussians[0].scale_opacity[3] > 0.0);
    }

    #[test]
    fn gaussian_decode_modes_cover_fixed_learned_and_oriented_scale() {
        let positions = [[0.1, -0.2, 0.3, 0.0]];

        let mut learned = vec![0.0; 8];
        learned[3] = 2.0;
        learned[4] = 1.5;
        learned[5] = 0.25;
        learned[6] = -0.25;
        learned[7] = 0.0;
        let learned_gaussian = decode_gaussians_3d(
            &positions,
            &learned,
            8,
            GaussianDecodeConfig {
                mode: GaussianDecodeMode::GaussianSh0LearnedScale,
                sigma: 0.01,
                opacity_scale: 1.0,
                ..GaussianDecodeConfig::default()
            },
        );
        assert_eq!(learned_gaussian[0].spherical_harmonic.len(), 1);
        assert_eq!(learned_gaussian[0].spherical_harmonic[0], [0.75, 0.25, 0.5]);
        assert!(learned_gaussian[0].scale_opacity[0] > 0.01);
        assert!(learned_gaussian[0].scale_opacity[3] > 0.5);
        assert_eq!(learned_gaussian[0].rotation, [1.0, 0.0, 0.0, 0.0]);

        let mut oriented = vec![0.0; 20];
        oriented[4] = 1.0;
        oriented[12] = 1.0;
        oriented[16] = 0.5;
        oriented[17] = -0.5;
        oriented[18] = 0.0;
        let oriented_gaussian = decode_gaussians_3d(
            &positions,
            &oriented,
            20,
            GaussianDecodeConfig {
                mode: GaussianDecodeMode::GaussianSh0Oriented,
                sh_degree: 1,
                ..GaussianDecodeConfig::default()
            },
        );
        assert_eq!(oriented_gaussian[0].spherical_harmonic.len(), 4);
        assert!((oriented_gaussian[0].rotation[0] - 1.0).abs() <= 1.0e-6);
        assert!(oriented_gaussian[0].scale_opacity[0].is_finite());
    }

    #[test]
    fn tile_assignment_supports_2d_and_3d_domains() {
        let cfg2 = HashGridConfig {
            grid_size: [8, 8, 1],
            mode: HashGridMode::Grid,
            ..HashGridConfig::growing_2d()
        };
        let tiles2 = TileGridConfig::from_hashgrid(&cfg2, [2, 2, 1]);
        let positions2 = [
            [-0.39, -0.39, 0.0, 0.0],
            [0.39, 0.39, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let assignment2 = assign_tiles(&positions2, 1, positions2.len(), &cfg2, &tiles2).unwrap();
        assert_eq!(tiles2.tile_grid_size(), [4, 4, 1]);
        assert_eq!(tiles2.tile_count(), 16);
        assert_eq!(tiles2.neighbor_offsets().len(), 9);
        assert_eq!(
            assignment2.tile_offsets.last().copied(),
            Some(positions2.len())
        );
        assert_eq!(assignment2.permutation.len(), positions2.len());

        let cfg3 = HashGridConfig {
            grid_size: [8, 8, 8],
            mode: HashGridMode::Grid,
            ..HashGridConfig::growing_3dgs()
        };
        let tiles3 = TileGridConfig::from_hashgrid(&cfg3, [2, 2, 2]);
        let positions3 = [
            [-0.39, -0.39, -0.39, 0.0],
            [0.39, 0.39, 0.39, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let assignment3 = assign_tiles(&positions3, 1, positions3.len(), &cfg3, &tiles3).unwrap();
        assert_eq!(tiles3.tile_grid_size(), [4, 4, 4]);
        assert_eq!(tiles3.tile_count(), 64);
        assert_eq!(tiles3.neighbor_offsets().len(), 27);
        assert_eq!(
            assignment3.tile_offsets.last().copied(),
            Some(positions3.len())
        );
        assert_eq!(assignment3.permutation.len(), positions3.len());
    }

    #[test]
    fn spatial_strategy_bvh_matches_hashgrid_neighbor_counts() {
        let cfg = HashGridConfig {
            grid_size: [32, 32, 32],
            mode: HashGridMode::Particle,
            ..HashGridConfig::growing_3dgs()
        };
        let positions = [
            [0.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.08, 0.02, 0.0],
            [-0.4, 0.1, 0.3, 0.0],
        ];
        let hash = analyze_spatial_strategy(
            &positions,
            1,
            positions.len(),
            &cfg,
            SpatialStrategyKind::HashGrid,
        )
        .unwrap();
        let bvh = analyze_spatial_strategy(
            &positions,
            1,
            positions.len(),
            &cfg,
            SpatialStrategyKind::Bvh { leaf_size: 2 },
        )
        .unwrap();

        assert_eq!(hash.exact_neighbor_pairs, bvh.exact_neighbor_pairs);
        assert!(bvh.node_count > 0);
        assert!(bvh.node_visits > 0);
        assert!(bvh.candidate_tests >= bvh.exact_neighbor_pairs);
    }

    #[test]
    fn spatial_strategy_tile_blocks_count_fixed_grid_candidates() {
        let cfg = HashGridConfig {
            grid_size: [8, 8, 1],
            mode: HashGridMode::Grid,
            ..HashGridConfig::texture_2d()
        };
        let positions = [
            [-0.45, -0.45, 0.0, 0.0],
            [-0.40, -0.45, 0.0, 0.0],
            [0.20, 0.25, 0.0, 0.0],
            [0.38, 0.37, 0.0, 0.0],
        ];
        let report = analyze_spatial_strategy(
            &positions,
            1,
            positions.len(),
            &cfg,
            SpatialStrategyKind::TileBlocks {
                tile_size: [2, 2, 1],
            },
        )
        .unwrap();

        assert_eq!(report.strategy.label(), "tile-blocks");
        assert!(report.active_bins > 0);
        assert!(report.candidate_tests >= report.exact_neighbor_pairs);
    }

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected} tolerance {tolerance}"
        );
    }
}
