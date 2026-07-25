use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{AdaptiveControllerTrainingBatch, AdaptiveOracleDatasetConfig};
use crate::adaptive::{
    ADAPTIVE_CONTROLLER_INPUT_DIMS, ADAPTIVE_CONTROLLER_OUTPUT_DIMS, allocate_resolution_budget,
    boundary_protected_spacing,
};
use crate::{AutomataError, AutomataResult};

pub fn adaptive_oracle_training_batch(
    cfg: AdaptiveOracleDatasetConfig,
) -> AutomataResult<AdaptiveControllerTrainingBatch> {
    validate_dataset_config(cfg)?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut boundary_distance = Vec::with_capacity(cfg.rows);
    let mut error_density = Vec::with_capacity(cfg.rows);
    let mut domain_measure = Vec::with_capacity(cfg.rows);
    let mut domain_boundary_distance = Vec::with_capacity(cfg.rows);
    let mut nuisance = Vec::with_capacity(cfg.rows);
    for _ in 0..cfg.rows {
        let point = [
            rng.random_range(-1.0_f32..1.0),
            rng.random_range(-1.0_f32..1.0),
            rng.random_range(-1.0_f32..1.0),
        ];
        let (distance, curvature) = manufactured_boundary_distance(point, cfg.spatial_dims);
        boundary_distance.push(distance);
        domain_boundary_distance.push(
            point[..cfg.spatial_dims]
                .iter()
                .map(|axis| 1.0 - axis.abs())
                .fold(f32::INFINITY, f32::min),
        );
        error_density
            .push(0.05 + (-distance / (2.0 * cfg.boundary_epsilon)).exp() * (1.0 + curvature));
        domain_measure.push(cfg.total_measure / cfg.rows as f32);
        nuisance.push([
            rng.random_range(-0.5_f32..0.5),
            rng.random_range(-0.35_f32..0.35),
            rng.random_range(-0.75_f32..0.75),
            rng.random_range(0.0_f32..1.0),
            rng.random_range(0.0_f32..1.0),
        ]);
    }
    let mean_error = error_density.iter().sum::<f32>() / cfg.rows as f32;
    let budget = allocate_resolution_budget(
        &error_density,
        &domain_measure,
        cfg.spatial_dims,
        2.0,
        cfg.reference_footprint,
        cfg.min_footprint,
        cfg.max_footprint,
        cfg.target_leaf_count,
    )?;
    let mut features = Vec::with_capacity(cfg.rows * ADAPTIVE_CONTROLLER_INPUT_DIMS);
    let mut targets = Vec::with_capacity(cfg.rows * ADAPTIVE_CONTROLLER_OUTPUT_DIMS);
    for row in 0..cfg.rows {
        let boundary_spacing = boundary_protected_spacing(
            boundary_distance[row],
            cfg.boundary_epsilon,
            cfg.boundary_slope,
            cfg.max_footprint,
        );
        let desired = budget.desired_footprint[row]
            .min(boundary_spacing)
            .clamp(cfg.min_footprint, cfg.max_footprint);
        let current_log = nuisance[row][0];
        let current = cfg.reference_footprint * current_log.exp();
        let spacing_ratio_log = nuisance[row][1];
        let bandwidth_spacing_log = nuisance[row][2];
        let degree_fraction = nuisance[row][3];
        let cooldown_fraction = nuisance[row][4];
        let desired_log = (desired / cfg.reference_footprint).ln();
        let mut feature = [0.0; ADAPTIVE_CONTROLLER_INPUT_DIMS];
        feature[..crate::adaptive::ADAPTIVE_CONTROLLER_SCALAR_DIMS].copy_from_slice(&[
            (error_density[row] / mean_error.max(1.0e-6)).ln_1p(),
            (domain_boundary_distance[row] / cfg.reference_footprint).ln_1p(),
            (1.5 - degree_fraction).ln(),
            spacing_ratio_log,
            current_log,
            bandwidth_spacing_log,
            degree_fraction,
            cooldown_fraction,
        ]);
        features.extend_from_slice(&feature);
        let desired_bandwidth = (2.2 * desired).clamp(cfg.min_footprint, cfg.max_footprint * 4.0);
        let observed_spacing = current * spacing_ratio_log.exp();
        let zeta = (desired_bandwidth / observed_spacing.max(1.0e-6))
            .ln()
            .clamp(-1.5, 1.5);
        let split_score = 8.0 * (current_log - desired_log) - 1.0 - 2.0 * cooldown_fraction;
        let merge_score = 8.0 * (desired_log - current_log) - 1.0 - 2.0 * cooldown_fraction;
        targets.extend_from_slice(&[
            desired_log,
            zeta,
            f32::from(split_score > 0.0),
            f32::from(merge_score > 0.0),
        ]);
    }
    let batch = AdaptiveControllerTrainingBatch {
        features,
        targets,
        rows: cfg.rows,
    };
    batch.validate()?;
    Ok(batch)
}

fn validate_dataset_config(cfg: AdaptiveOracleDatasetConfig) -> AutomataResult<()> {
    if cfg.rows == 0
        || !(cfg.spatial_dims == 2 || cfg.spatial_dims == 3)
        || cfg.target_leaf_count == 0
        || !cfg.reference_footprint.is_finite()
        || !cfg.min_footprint.is_finite()
        || !cfg.max_footprint.is_finite()
        || !cfg.total_measure.is_finite()
        || cfg.total_measure <= 0.0
        || cfg.min_footprint <= 0.0
        || cfg.reference_footprint < cfg.min_footprint
        || cfg.max_footprint < cfg.reference_footprint
        || !cfg.boundary_epsilon.is_finite()
        || cfg.boundary_epsilon <= 0.0
        || !cfg.boundary_slope.is_finite()
        || cfg.boundary_slope < 0.0
    {
        return Err(AutomataError::InvalidArgument(
            "invalid adaptive oracle dataset config".to_string(),
        ));
    }
    let unit_ball = crate::adaptive::unit_ball_measure(cfg.spatial_dims);
    let minimum_count =
        cfg.total_measure / (unit_ball * cfg.max_footprint.powi(cfg.spatial_dims as i32));
    let maximum_count =
        cfg.total_measure / (unit_ball * cfg.min_footprint.powi(cfg.spatial_dims as i32));
    if cfg.target_leaf_count as f32 + 1.0e-3 < minimum_count
        || cfg.target_leaf_count as f32 - 1.0e-3 > maximum_count
    {
        return Err(AutomataError::InvalidArgument(format!(
            "adaptive target {} is infeasible for total measure {} and footprint bounds {}..{} (feasible {:.1}..{:.1})",
            cfg.target_leaf_count,
            cfg.total_measure,
            cfg.min_footprint,
            cfg.max_footprint,
            minimum_count,
            maximum_count,
        )));
    }
    Ok(())
}

fn manufactured_boundary_distance(point: [f32; 3], dim: usize) -> (f32, f32) {
    let radial = point[..dim]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let outer = (radial - 0.82).abs();
    let cavity_center = if dim == 2 {
        [0.22, -0.12, 0.0]
    } else {
        [0.22, -0.12, 0.15]
    };
    let cavity_radial = (0..dim)
        .map(|axis| (point[axis] - cavity_center[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    let cavity = (cavity_radial - 0.23).abs();
    let distance = outer.min(cavity);
    let curvature = if cavity < outer { 1.5 } else { 0.35 };
    (distance, curvature)
}
