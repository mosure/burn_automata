use super::*;

pub(crate) fn active_surface_coverage_profile(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_coverage_profile(&active_positions, target, samples, threshold, bins)
}

pub(crate) fn material_visible_surface_coverage_profile(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return surface_coverage_profile(&[], target, samples, threshold, bins);
    };
    let visible_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let state_base = idx * state_dims;
            let material_opacity = states[state_base + material_channel];
            (material_opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_coverage_profile(&visible_positions, target, samples, threshold, bins)
}

pub(crate) fn active_surface_normal_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_normal_coverage_report(&active_positions, target, samples, threshold)
}

pub(crate) fn material_visible_surface_normal_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let Some(material_channel) = growth_3d_material_opacity_channel(state_dims) else {
        return surface_normal_coverage_report(&[], target, samples, threshold);
    };
    let visible_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let state_base = idx * state_dims;
            let material_opacity = states[state_base + material_channel];
            (material_opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    surface_normal_coverage_report(&visible_positions, target, samples, threshold)
}

pub(crate) fn surface_normal_coverage_report(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> SurfaceNormalCoverageReport {
    let samples = samples.max(1);
    let directions = normal_coverage_directions();
    let normal_bins = directions.len();
    let mut target_bin_samples = vec![0usize; normal_bins];
    let mut bin_covered = vec![0usize; normal_bins];
    let mut covered_samples = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let bin = normal_direction_bin(sample.normal, &directions);
        target_bin_samples[bin] += 1;
        let mut best_distance2 = f32::MAX;
        for position in positions {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            best_distance2 = best_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        if !positions.is_empty() && best_distance2.sqrt() <= threshold {
            covered_samples += 1;
            bin_covered[bin] += 1;
        }
    }

    let target_bins = target_bin_samples
        .iter()
        .filter(|samples| **samples > 0)
        .count();
    let covered_target_bins = target_bin_samples
        .iter()
        .zip(bin_covered.iter())
        .filter(|(samples, covered)| **samples > 0 && **covered > 0)
        .count();
    let target_bin_sample_fractions = target_bin_samples
        .iter()
        .map(|count| *count as f32 / samples as f32)
        .collect::<Vec<_>>();
    let bin_covered_fractions = target_bin_samples
        .iter()
        .zip(bin_covered.iter())
        .map(|(samples, covered)| {
            if *samples == 0 {
                0.0
            } else {
                *covered as f32 / *samples as f32
            }
        })
        .collect::<Vec<_>>();
    let active_bin_fractions = target_bin_samples
        .iter()
        .zip(bin_covered_fractions.iter())
        .filter_map(|(samples, fraction)| (*samples > 0).then_some(*fraction))
        .collect::<Vec<_>>();
    let min_bin_covered_fraction = active_bin_fractions
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bin_covered_fraction = active_bin_fractions.iter().copied().fold(0.0, f32::max);
    let mean_bin_covered_fraction = if active_bin_fractions.is_empty() {
        0.0
    } else {
        active_bin_fractions.iter().copied().sum::<f32>() / active_bin_fractions.len() as f32
    };

    SurfaceNormalCoverageReport {
        samples,
        normal_bins,
        threshold,
        target_bins,
        covered_target_bins,
        covered_target_bin_fraction: if target_bins == 0 {
            0.0
        } else {
            covered_target_bins as f32 / target_bins as f32
        },
        covered_sample_fraction: covered_samples as f32 / samples as f32,
        min_bin_covered_fraction: if min_bin_covered_fraction.is_finite() {
            min_bin_covered_fraction
        } else {
            0.0
        },
        mean_bin_covered_fraction,
        max_bin_covered_fraction,
        target_bin_sample_fractions,
        bin_covered_fractions,
    }
}

pub(crate) fn surface_coverage_profile(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
    bins: usize,
) -> SurfaceCoverageProfileReport {
    let samples = samples.max(1);
    let bins = bins.max(1).min(samples);
    let mut bin_samples = vec![0usize; bins];
    let mut bin_covered = vec![0usize; bins];
    let mut assigned_counts = vec![0usize; positions.len()];
    let mut covered_assigned_counts = vec![0usize; positions.len()];
    let mut covered = 0usize;

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bins / samples).min(bins - 1);
        bin_samples[bin] += 1;
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if positions.is_empty() || !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_distance2.sqrt() <= threshold {
            covered += 1;
            bin_covered[bin] += 1;
            covered_assigned_counts[best_row] += 1;
        }
    }

    let bin_covered_fractions = bin_samples
        .iter()
        .zip(bin_covered.iter())
        .map(|(samples, covered)| {
            if *samples == 0 {
                0.0
            } else {
                *covered as f32 / *samples as f32
            }
        })
        .collect::<Vec<_>>();
    let empty_bins = bin_covered.iter().filter(|covered| **covered == 0).count();
    let covered_bins = bins.saturating_sub(empty_bins);
    let min_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_bin_covered_fraction = bin_covered_fractions
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let mean_bin_covered_fraction =
        bin_covered_fractions.iter().copied().sum::<f32>() / bins as f32;
    let assigned_particles = assigned_counts.iter().filter(|count| **count > 0).count();
    let covered_assigned_particles = covered_assigned_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let max_assigned_samples = assigned_counts.iter().copied().max().unwrap_or(0);
    let max_covered_assigned_samples = covered_assigned_counts.iter().copied().max().unwrap_or(0);

    SurfaceCoverageProfileReport {
        samples,
        bins,
        threshold,
        covered_fraction: covered as f32 / samples as f32,
        covered_bin_fraction: covered_bins as f32 / bins as f32,
        empty_bins,
        min_bin_covered_fraction: if min_bin_covered_fraction.is_finite() {
            min_bin_covered_fraction
        } else {
            0.0
        },
        mean_bin_covered_fraction,
        max_bin_covered_fraction,
        assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            assigned_particles as f32 / positions.len() as f32
        },
        covered_assigned_particle_fraction: if positions.is_empty() {
            0.0
        } else {
            covered_assigned_particles as f32 / positions.len() as f32
        },
        max_assigned_sample_fraction: max_assigned_samples as f32 / samples as f32,
        max_covered_assigned_sample_fraction: max_covered_assigned_samples as f32 / samples as f32,
        bin_covered_fractions,
    }
}
