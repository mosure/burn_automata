use std::time::Instant;

use rayon::prelude::*;

use super::{AnalyticSolid, stateless_key, stateless_uniform};
use crate::adaptive::experiments::{AdaptiveScalingExperimentConfig, AdaptiveSparseQualityRow};
use crate::{AutomataError, AutomataResult};

#[derive(Clone, Copy)]
struct FineVoxel {
    cell: [usize; 3],
    linear: usize,
    point: [f64; 3],
    clearance: f64,
    value: f64,
    boundary: bool,
    protected: bool,
}

#[derive(Default)]
struct Cluster {
    count: usize,
    point_sum: [f64; 3],
    value_sum: f64,
}

pub(super) fn run_sparse_quality_experiment(
    config: &AdaptiveScalingExperimentConfig,
    seed: u64,
) -> AutomataResult<Vec<AdaptiveSparseQualityRow>> {
    let mut rows = Vec::new();
    let maximum_cap = config
        .quality_spacing_cap_ratios
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| AutomataError::InvalidArgument("quality cap sweep is empty".to_string()))?;
    for solid in [AnalyticSolid::SphereWithCavity, AnalyticSolid::Torus] {
        let voxels = build_voxels(config, solid);
        if voxels.is_empty() {
            return Err(AutomataError::InvalidArgument(format!(
                "{} quality voxelization is empty",
                solid.name()
            )));
        }
        let protected_leaves = voxels.iter().filter(|voxel| voxel.protected).count();
        let surface = surface_samples(solid, config);
        for &cap in &config.quality_spacing_cap_ratios {
            for sample in 0..config.retention_samples {
                let retained = adaptive_retained_indices(config, solid, &voxels, cap, seed, sample);
                rows.push(evaluate_allocation(
                    config,
                    solid,
                    &voxels,
                    &surface,
                    &retained,
                    protected_leaves,
                    cap,
                    sample,
                    "clearance-oracle",
                    true,
                )?);
                if (cap - maximum_cap).abs() <= f32::EPSILON {
                    let uniform = uniform_matched_indices(
                        config,
                        solid,
                        &voxels,
                        retained.len(),
                        seed,
                        sample,
                    );
                    rows.push(evaluate_allocation(
                        config,
                        solid,
                        &voxels,
                        &surface,
                        &uniform,
                        protected_leaves,
                        cap,
                        sample,
                        "uniform-matched-count",
                        false,
                    )?);
                }
            }
        }
    }
    Ok(rows)
}

fn build_voxels(config: &AdaptiveScalingExperimentConfig, solid: AnalyticSolid) -> Vec<FineVoxel> {
    let resolution = config.quality_resolution;
    let voxel = 2.0 / resolution as f64;
    let protected_distance = config.protected_band_voxels as f64 * voxel;
    (0..resolution.pow(3))
        .filter_map(|linear| {
            let cell = delinearize(linear, resolution);
            let point = cell_center(cell, voxel);
            let clearance = solid.clearance(point.map(|value| value as f32), config)? as f64;
            Some(FineVoxel {
                cell,
                linear,
                point,
                clearance,
                value: analytic_field(point),
                boundary: clearance <= 0.9 * voxel,
                protected: clearance <= protected_distance,
            })
        })
        .collect()
}

fn adaptive_retained_indices(
    config: &AdaptiveScalingExperimentConfig,
    solid: AnalyticSolid,
    voxels: &[FineVoxel],
    cap_ratio: f32,
    seed: u64,
    sample: usize,
) -> Vec<usize> {
    let voxel_width = 2.0 / config.quality_resolution as f64;
    let protected_distance = config.protected_band_voxels as f64 * voxel_width;
    let mut retained = voxels
        .iter()
        .enumerate()
        .filter_map(|(index, voxel)| {
            let spacing = (voxel_width
                + (voxel.clearance - protected_distance).max(0.0)
                    / config.transition_divisor as f64)
                .min(cap_ratio as f64 * voxel_width)
                .max(voxel_width);
            let probability = (spacing / voxel_width).powi(-3).clamp(0.0, 1.0);
            (voxel.protected
                || stateless_uniform(
                    seed,
                    solid as u64,
                    config.quality_resolution,
                    voxel.linear,
                    sample,
                ) < probability)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if !retained.iter().any(|index| !voxels[*index].protected)
        && let Some(index) = voxels.iter().position(|voxel| !voxel.protected)
    {
        retained.push(index);
    }
    retained
}

fn uniform_matched_indices(
    config: &AdaptiveScalingExperimentConfig,
    solid: AnalyticSolid,
    voxels: &[FineVoxel],
    count: usize,
    seed: u64,
    sample: usize,
) -> Vec<usize> {
    let mut ranked = (0..voxels.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|index| {
        stateless_key(
            seed ^ 0x756e_6966_6f72_6d21,
            solid as u64,
            config.quality_resolution,
            voxels[*index].linear,
            sample,
        )
    });
    ranked.truncate(count.clamp(1, voxels.len()));
    ranked
}

#[allow(clippy::too_many_arguments)]
fn evaluate_allocation(
    config: &AdaptiveScalingExperimentConfig,
    solid: AnalyticSolid,
    voxels: &[FineVoxel],
    surface: &[[f64; 3]],
    retained: &[usize],
    protected_leaves: usize,
    cap_ratio: f32,
    sample: usize,
    allocation: &str,
    protected_singletons: bool,
) -> AutomataResult<AdaptiveSparseQualityRow> {
    let started = Instant::now();
    let resolution = config.quality_resolution;
    let cell_count = resolution.pow(3);
    let mut all_map = vec![usize::MAX; cell_count];
    let mut interior_map = vec![usize::MAX; cell_count];
    for (cluster, &voxel_index) in retained.iter().enumerate() {
        let voxel = voxels[voxel_index];
        all_map[voxel.linear] = cluster;
        if !voxel.protected {
            interior_map[voxel.linear] = cluster;
        }
    }
    let interior_available = interior_map.iter().any(|value| *value != usize::MAX);
    let assignments = voxels
        .par_iter()
        .map(|voxel| {
            if protected_singletons && voxel.protected {
                return all_map[voxel.linear];
            }
            let primary = if protected_singletons && interior_available {
                &interior_map
            } else {
                &all_map
            };
            nearest_grid_cluster(voxel.cell, primary, resolution)
                .or_else(|| nearest_grid_cluster(voxel.cell, &all_map, resolution))
                .unwrap_or(usize::MAX)
        })
        .collect::<Vec<_>>();
    if assignments.contains(&usize::MAX) {
        return Err(AutomataError::InvalidArgument(format!(
            "{} {} allocation left an unassigned voxel",
            solid.name(),
            allocation
        )));
    }

    let mut clusters = (0..retained.len())
        .map(|_| Cluster::default())
        .collect::<Vec<_>>();
    for (voxel, &cluster_index) in voxels.iter().zip(&assignments) {
        let cluster = &mut clusters[cluster_index];
        cluster.count += 1;
        for axis in 0..3 {
            cluster.point_sum[axis] += voxel.point[axis];
        }
        cluster.value_sum += voxel.value;
    }
    let cluster_values = clusters
        .iter()
        .map(|cluster| cluster.value_sum / cluster.count.max(1) as f64)
        .collect::<Vec<_>>();

    let mut field_error2 = 0.0;
    let mut field_norm2 = 0.0;
    let mut protected_error2 = 0.0;
    let mut protected_norm2 = 0.0;
    let mut fine_centroid_sum = [0.0; 3];
    let mut fine_integral = 0.0;
    let mut fine_quadratic = 0.0;
    let mut boundary_owner = vec![false; retained.len()];
    for (voxel, &cluster_index) in voxels.iter().zip(&assignments) {
        let difference = cluster_values[cluster_index] - voxel.value;
        field_error2 += difference * difference;
        field_norm2 += voxel.value * voxel.value;
        if voxel.protected {
            protected_error2 += difference * difference;
            protected_norm2 += voxel.value * voxel.value;
        }
        if voxel.boundary {
            boundary_owner[cluster_index] = true;
        }
        for (sum, value) in fine_centroid_sum.iter_mut().zip(voxel.point) {
            *sum += value;
        }
        fine_integral += voxel.value;
        fine_quadratic += voxel.value * voxel.value;
    }
    let coarse_quadratic = clusters
        .iter()
        .zip(&cluster_values)
        .map(|(cluster, value)| cluster.count as f64 * value * value)
        .sum::<f64>();
    let coarse_centroid_sum = clusters.iter().fold([0.0; 3], |mut sum, cluster| {
        for (sum, value) in sum.iter_mut().zip(cluster.point_sum) {
            *sum += value;
        }
        sum
    });
    let coarse_integral = clusters
        .iter()
        .map(|cluster| cluster.value_sum)
        .sum::<f64>();
    let count = voxels.len() as f64;
    let centroid_l2_error = (0..3)
        .map(|axis| ((coarse_centroid_sum[axis] - fine_centroid_sum[axis]) / count).powi(2))
        .sum::<f64>()
        .sqrt();

    let protected_distance = config.protected_band_voxels as f64 * 2.0 / resolution as f64;
    let mut deep_ratios = Vec::new();
    let mut boundary_ratios = Vec::new();
    for (cluster_index, (&voxel_index, cluster)) in retained.iter().zip(&clusters).enumerate() {
        let ratio = (cluster.count as f64).cbrt();
        if voxels[voxel_index].clearance > 2.0 * protected_distance {
            deep_ratios.push(ratio);
        }
        if boundary_owner[cluster_index] {
            boundary_ratios.push(ratio);
        }
    }

    let boundary_hd95_voxels =
        boundary_hd95(voxels, retained, &boundary_owner, surface, resolution);
    Ok(AdaptiveSparseQualityRow {
        solid: solid.name().to_string(),
        allocation: allocation.to_string(),
        resolution,
        spacing_cap_ratio: cap_ratio,
        sample,
        fine_leaves: voxels.len(),
        retained_leaves: retained.len(),
        protected_leaves,
        count_reduction: voxels.len() as f64 / retained.len().max(1) as f64,
        field_nrmse: (field_error2 / field_norm2.max(f64::MIN_POSITIVE)).sqrt(),
        protected_band_nrmse: (protected_error2 / protected_norm2.max(f64::MIN_POSITIVE)).sqrt(),
        measure_relative_error: (clusters.iter().map(|cluster| cluster.count).sum::<usize>()
            as f64
            - count)
            .abs()
            / count.max(1.0),
        centroid_l2_error,
        field_integral_relative_error: (coarse_integral - fine_integral).abs()
            / fine_integral.abs().max(f64::MIN_POSITIVE),
        quadratic_integral_loss_fraction: ((fine_quadratic - coarse_quadratic)
            / fine_quadratic.max(f64::MIN_POSITIVE))
        .max(0.0),
        median_deep_footprint_ratio: median(&mut deep_ratios),
        median_boundary_footprint_ratio: median(&mut boundary_ratios),
        boundary_hd95_voxels,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn nearest_grid_cluster(cell: [usize; 3], map: &[usize], resolution: usize) -> Option<usize> {
    let mut best = None;
    let mut best_distance2 = usize::MAX;
    for radius in 0..resolution {
        visit_shell(cell, radius, resolution, |candidate| {
            let cluster = map[linearize(candidate, resolution)];
            if cluster == usize::MAX {
                return;
            }
            let distance2 = (0..3)
                .map(|axis| candidate[axis].abs_diff(cell[axis]).pow(2))
                .sum::<usize>();
            if distance2 < best_distance2
                || (distance2 == best_distance2 && best.is_none_or(|value| cluster < value))
            {
                best_distance2 = distance2;
                best = Some(cluster);
            }
        });
        if best.is_some() && best_distance2 < (radius + 1).pow(2) {
            break;
        }
    }
    best
}

fn boundary_hd95(
    voxels: &[FineVoxel],
    retained: &[usize],
    boundary_owner: &[bool],
    surface: &[[f64; 3]],
    resolution: usize,
) -> f64 {
    let mut owner_map = vec![usize::MAX; resolution.pow(3)];
    let mut site_to_surface = Vec::new();
    for (cluster, (&voxel_index, &owns_boundary)) in retained.iter().zip(boundary_owner).enumerate()
    {
        if owns_boundary {
            let voxel = voxels[voxel_index];
            owner_map[voxel.linear] = cluster;
            site_to_surface.push(voxel.clearance * resolution as f64 / 2.0);
        }
    }
    if site_to_surface.is_empty() {
        return f64::INFINITY;
    }
    let voxel_width = 2.0 / resolution as f64;
    let mut surface_to_site = surface
        .par_iter()
        .map(|point| nearest_point_distance(*point, &owner_map, resolution, voxel_width))
        .collect::<Vec<_>>();
    surface_to_site
        .iter_mut()
        .for_each(|distance| *distance /= voxel_width);
    percentile(&mut surface_to_site, 0.95).max(percentile(&mut site_to_surface, 0.95))
}

fn nearest_point_distance(
    point: [f64; 3],
    map: &[usize],
    resolution: usize,
    voxel_width: f64,
) -> f64 {
    let base = point.map(|value| {
        (((value + 1.0) / voxel_width - 0.5).round() as isize).clamp(0, resolution as isize - 1)
            as usize
    });
    let mut best_distance2 = f64::INFINITY;
    for radius in 0..resolution {
        visit_shell(base, radius, resolution, |candidate| {
            if map[linearize(candidate, resolution)] == usize::MAX {
                return;
            }
            let center = cell_center(candidate, voxel_width);
            let distance2 = (0..3)
                .map(|axis| (center[axis] - point[axis]).powi(2))
                .sum::<f64>();
            best_distance2 = best_distance2.min(distance2);
        });
        let unseen_lower_bound = (radius as f64 + 0.5) * voxel_width;
        if best_distance2.is_finite() && unseen_lower_bound * unseen_lower_bound > best_distance2 {
            break;
        }
    }
    best_distance2.sqrt()
}

fn visit_shell(
    center: [usize; 3],
    radius: usize,
    resolution: usize,
    mut visit: impl FnMut([usize; 3]),
) {
    let radius = radius as isize;
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dy.abs()).max(dz.abs()) != radius {
                    continue;
                }
                let candidate = [
                    center[0] as isize + dx,
                    center[1] as isize + dy,
                    center[2] as isize + dz,
                ];
                if candidate
                    .iter()
                    .any(|value| *value < 0 || *value >= resolution as isize)
                {
                    continue;
                }
                visit(candidate.map(|value| value as usize));
            }
        }
    }
}

fn surface_samples(
    solid: AnalyticSolid,
    config: &AdaptiveScalingExperimentConfig,
) -> Vec<[f64; 3]> {
    match solid {
        AnalyticSolid::SphereWithCavity => {
            let mut points = fibonacci_sphere(2_048, [0.0; 3], config.sphere_outer_radius as f64);
            points.extend(fibonacci_sphere(
                1_024,
                config.sphere_cavity_center.map(|value| value as f64),
                config.sphere_cavity_radius as f64,
            ));
            points
        }
        AnalyticSolid::Torus => {
            let mut points = Vec::with_capacity(64 * 48);
            for major_index in 0..64 {
                let major = std::f64::consts::TAU * major_index as f64 / 64.0;
                for minor_index in 0..48 {
                    let minor = std::f64::consts::TAU * minor_index as f64 / 48.0;
                    let radius = config.torus_major_radius as f64
                        + config.torus_minor_radius as f64 * minor.cos();
                    points.push([
                        radius * major.cos(),
                        radius * major.sin(),
                        config.torus_minor_radius as f64 * minor.sin(),
                    ]);
                }
            }
            points
        }
    }
}

fn fibonacci_sphere(count: usize, center: [f64; 3], radius: f64) -> Vec<[f64; 3]> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|index| {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
            let radial = (1.0 - z * z).sqrt();
            let angle = golden_angle * index as f64;
            [
                center[0] + radius * radial * angle.cos(),
                center[1] + radius * radial * angle.sin(),
                center[2] + radius * z,
            ]
        })
        .collect()
}

fn analytic_field(point: [f64; 3]) -> f64 {
    1.0 + 0.35 * (2.3 * point[0]).sin() + 0.25 * (3.1 * point[1]).cos()
        - 0.20 * (4.7 * point[2]).sin()
        + 0.15 * point[0] * point[1]
}

fn linearize(cell: [usize; 3], resolution: usize) -> usize {
    (cell[2] * resolution + cell[1]) * resolution + cell[0]
}

fn delinearize(linear: usize, resolution: usize) -> [usize; 3] {
    [
        linear % resolution,
        (linear / resolution) % resolution,
        linear / (resolution * resolution),
    ]
}

fn cell_center(cell: [usize; 3], voxel_width: f64) -> [f64; 3] {
    cell.map(|value| -1.0 + (value as f64 + 0.5) * voxel_width)
}

fn median(values: &mut [f64]) -> f64 {
    percentile(values, 0.5)
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_quality_audit_conserves_restricted_quantities() {
        let config = AdaptiveScalingExperimentConfig {
            resolutions: vec![16, 24],
            quality_resolution: 16,
            quality_spacing_cap_ratios: vec![2.0, 4.0],
            retention_samples: 2,
            ..AdaptiveScalingExperimentConfig::default()
        };
        let rows = run_sparse_quality_experiment(&config, 17).unwrap();
        assert_eq!(rows.len(), 12);
        for row in &rows {
            assert!(row.retained_leaves > 0);
            assert!(row.retained_leaves <= row.fine_leaves);
            assert!(row.measure_relative_error < 1.0e-12);
            assert!(row.centroid_l2_error < 1.0e-12);
            assert!(row.field_integral_relative_error < 1.0e-12);
            assert!(row.boundary_hd95_voxels.is_finite());
            if row.allocation == "clearance-oracle" {
                assert!(row.protected_band_nrmse < 1.0e-12);
                assert!((row.median_boundary_footprint_ratio - 1.0).abs() < 1.0e-12);
            }
        }
    }
}
