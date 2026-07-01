#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) const DEFAULT_LOCAL_FRONT_ROW_FRACTION: usize = 16;
pub(crate) const DEFAULT_LOCAL_FRONT_MAX_CANDIDATES: usize = 64;

pub(crate) fn default_local_front_candidate_count(rows: usize) -> usize {
    if rows == 0 {
        0
    } else {
        rows.div_ceil(DEFAULT_LOCAL_FRONT_ROW_FRACTION)
            .clamp(1, DEFAULT_LOCAL_FRONT_MAX_CANDIDATES)
    }
}

pub(crate) fn local_front_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> Vec<f32> {
    local_front_weights_with_min_candidates(config, positions, states, front_radius, 0)
}

pub(crate) fn local_front_weights_with_min_candidates(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
    min_front_candidates: usize,
) -> Vec<f32> {
    let rows = positions.len();
    let mut weights = vec![0.0; rows];
    if config.state_dims <= 3 || rows == 0 || front_radius <= 0.0 {
        return weights;
    }
    let active_threshold = -1.0_f32;
    let mut active_count = 0usize;
    let mut dormant_distances = Vec::new();
    for (row, position) in positions.iter().enumerate() {
        let current_opacity = states[row * config.state_dims + 3];
        if current_opacity > active_threshold {
            weights[row] = 1.0;
            active_count += 1;
            continue;
        }

        let mut nearest_active_distance2 = f32::MAX;
        for (other_row, other_position) in positions.iter().enumerate() {
            let other_opacity = states[other_row * config.state_dims + 3];
            if other_opacity <= active_threshold {
                continue;
            }
            let dx = position[0] - other_position[0];
            let dy = position[1] - other_position[1];
            let dz = position[2] - other_position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_active_distance2 {
                nearest_active_distance2 = distance2;
            }
        }
        if nearest_active_distance2.is_finite() {
            dormant_distances.push((row, nearest_active_distance2));
        }
    }
    let mut effective_front_radius = front_radius;
    let mut requested_front_rows = Vec::new();
    if active_count > 0 && !dormant_distances.is_empty() {
        dormant_distances.sort_by(|(_, lhs), (_, rhs)| {
            lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
        });
        let default_front = default_local_front_candidate_count(rows);
        let requested_front = min_front_candidates.min(rows / 2).max(default_front);
        let desired_front = dormant_distances.len().min(requested_front);
        if desired_front > 0 {
            if min_front_candidates > default_front {
                requested_front_rows.extend(
                    dormant_distances
                        .iter()
                        .take(desired_front)
                        .map(|(row, _)| *row),
                );
            }
            let sparse_radius = dormant_distances[desired_front - 1].1.sqrt() * 1.05;
            if sparse_radius.is_finite() {
                effective_front_radius = effective_front_radius.max(sparse_radius);
            }
        }
    }
    let front_radius2 = effective_front_radius * effective_front_radius;
    if front_radius2 <= 0.0 || !front_radius2.is_finite() {
        return weights;
    }
    for (row, nearest_active_distance2) in dormant_distances {
        if nearest_active_distance2 <= front_radius2 {
            let weight = (1.0 - (nearest_active_distance2 / front_radius2).sqrt()).max(0.0);
            weights[row] = if requested_front_rows.contains(&row) {
                weight.max(0.25)
            } else {
                weight
            };
        }
    }
    weights
}

pub(crate) fn add_target_coverage_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    target_update: &mut [f32],
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    front_states: Option<&[f32]>,
    front_radius: f32,
) {
    if coverage_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let rows = positions.len();
    let output_dims = config.update_dims();
    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));

    if coverage_mode != CoverageUpdateModeArg::HardNearest {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        if eligible_rows.is_empty() {
            return;
        }
        let coverage_updates = match coverage_mode {
            CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                coverage_softness,
                coverage_repulsion_gain,
                coverage_repulsion_radius,
                coverage_normal_weight,
                seed_scale,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::HardNearest => unreachable!("handled by outer branch"),
        };
        for row in 0..rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            if front_weight <= 1.0e-3 {
                continue;
            }
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * coverage_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
        if (coverage_mode != CoverageUpdateModeArg::SoftChamfer
            && coverage_repulsion_gain > 0.0
            && coverage_repulsion_gain.is_finite())
            || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
        {
            let mut repulsion_updates = vec![[0.0; 3]; rows];
            if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
                add_surface_tangent_repulsion_to_updates(
                    target,
                    positions,
                    &eligible_rows,
                    coverage_gain,
                    coverage_repulsion_gain,
                    coverage_repulsion_radius,
                    seed_scale,
                    max_update_norm,
                    &mut repulsion_updates,
                );
            }
            add_surface_gap_relocation_to_updates(
                target,
                positions,
                &eligible_rows,
                coverage_gain,
                coverage_gap_gain,
                coverage_samples,
                coverage_normal_weight,
                seed_scale,
                max_update_norm,
                &mut repulsion_updates,
            );
            for row in 0..rows {
                let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
                if front_weight <= 1.0e-3 {
                    continue;
                }
                let base = row * output_dims;
                for axis in 0..3 {
                    target_update[base + axis] += front_weight * repulsion_updates[row][axis];
                }
                clamp_target_motion_update(target_update, base, max_update_norm);
            }
        }
        return;
    }

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            if front_weights
                .as_ref()
                .is_some_and(|weights| weights[row] <= 1.0e-3)
            {
                continue;
            }
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }

        residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
        residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
        residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
        counts[best_row] += 1;
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        let base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        let scale = coverage_gain * front_weight / count as f32;
        for axis in 0..3 {
            target_update[base + axis] += residual_sums[row][axis] * scale;
        }
        clamp_target_motion_update(target_update, base, max_update_norm);
    }
    if (coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite())
        || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
    {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        let mut repulsion_updates = vec![[0.0; 3]; rows];
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        add_surface_gap_relocation_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_gap_gain,
            coverage_samples,
            coverage_normal_weight,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        for row in eligible_rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * repulsion_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
    }
}

pub(crate) fn clamp_target_motion_update(
    target_update: &mut [f32],
    base: usize,
    max_update_norm: f32,
) {
    let norm = (target_update[base].powi(2)
        + target_update[base + 1].powi(2)
        + target_update[base + 2].powi(2))
    .sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for axis in 0..3 {
            target_update[base + axis] *= clamp;
        }
    }
}

pub(crate) fn add_target_extent_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    front_states: Option<&[f32]>,
    target_update: &mut [f32],
    extent_gain: f32,
    max_update_norm: f32,
    front_radius: f32,
) {
    if extent_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));
    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_rows = 0usize;
    for (row, position) in positions.iter().enumerate() {
        if front_weights
            .as_ref()
            .is_some_and(|weights| weights[row] <= 1.0e-3)
        {
            continue;
        }
        active_rows += 1;
        for axis in 0..3 {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_rows == 0 {
        return;
    }

    let (target_min, target_max) = target.bounds();
    let output_dims = config.update_dims();
    for (row, position) in positions.iter().enumerate() {
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        if front_weight <= 1.0e-3 {
            continue;
        }
        let base = row * output_dims;
        for axis in 0..3 {
            let active_extent = (active_max[axis] - active_min[axis]).max(1.0e-4);
            let t = ((position[axis] - active_min[axis]) / active_extent).clamp(0.0, 1.0);
            let min_weight = (1.0 - t).powi(3);
            let max_weight = t.powi(3);
            let residual = min_weight * (target_min[axis] - position[axis])
                + max_weight * (target_max[axis] - position[axis]);
            target_update[base + axis] += extent_gain * front_weight * residual;
        }
        let norm = (target_update[base].powi(2)
            + target_update[base + 1].powi(2)
            + target_update[base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                target_update[base + axis] *= clamp;
            }
        }
    }
}

pub(crate) fn torus_implicit_training_position(
    row: usize,
    scale: f32,
    rng: &mut StdRng,
) -> [f32; 3] {
    match row % 4 {
        0 => uv_torus_dense_seed_position(rng, scale),
        1 => {
            let surface = uv_torus_continuous_surface_position(rng, scale);
            [
                surface[0] + rng.random_range(-0.18..0.18) * scale,
                surface[1] + rng.random_range(-0.18..0.18) * scale,
                surface[2] + rng.random_range(-0.18..0.18) * scale,
            ]
        }
        2 => uv_torus_continuous_volume_position(rng, scale),
        _ => {
            let radius = scale * (1.0 + UV_TORUS_MINOR_RATIO) * 0.95;
            [
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
            ]
        }
    }
}

pub(crate) fn utah_teapot_training_position(
    row: usize,
    scale: f32,
    target: &TriangleMeshTarget,
    rng: &mut StdRng,
) -> [f32; 3] {
    match row % 4 {
        0 => utah_teapot_dense_seed_position(rng, target),
        1 => {
            let sample = target.surface_sample(row);
            [
                sample.position[0] + rng.random_range(-0.14..0.14) * scale,
                sample.position[1] + rng.random_range(-0.14..0.14) * scale,
                sample.position[2] + rng.random_range(-0.14..0.14) * scale,
            ]
        }
        2 => target.near_surface_query(row * 17 + 3, rng.random_range(-0.16..0.16) * scale),
        _ => [
            rng.random_range(-1.15..1.15) * scale,
            rng.random_range(-0.70..0.70) * scale,
            rng.random_range(-0.55..0.75) * scale,
        ],
    }
}
