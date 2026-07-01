#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn render_proxy_target_coverage_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let updates = vec![[0.0; 3]; rows];
    if rows == 0 || coverage_gain <= 0.0 {
        return updates;
    }

    let active_rows = (0..rows)
        .filter(|&row| config.state_dims <= 3 || states[row * config.state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return updates;
    }

    let mut updates = match coverage_mode {
        CoverageUpdateModeArg::HardNearest => render_proxy_hard_target_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
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
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
    };
    if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &active_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut updates,
        );
    }
    add_surface_gap_relocation_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        coverage_normal_weight,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_strata_coverage_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_normal_coverage_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_normal_weight,
        coverage_samples,
        max_update_norm,
        &mut updates,
    );
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_proxy_weighted_target_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    row_weights: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let updates = vec![[0.0; 3]; rows];
    if rows == 0 || row_weights.len() < rows || coverage_gain <= 0.0 {
        return updates;
    }

    let candidate_rows = (0..rows)
        .filter(|&row| row_weights[row].is_finite() && row_weights[row] > 1.0e-3)
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return updates;
    }

    let mut updates = match coverage_mode {
        CoverageUpdateModeArg::HardNearest => render_proxy_hard_target_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
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
            &candidate_rows,
            updates,
        ),
        CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
        CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
    };
    if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &candidate_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut updates,
        );
    }
    add_surface_gap_relocation_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        coverage_normal_weight,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_strata_coverage_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_normal_coverage_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_normal_weight,
        coverage_samples,
        max_update_norm,
        &mut updates,
    );

    for (row, update) in updates.iter_mut().enumerate() {
        let weight = row_weights[row].clamp(0.0, 1.0);
        if weight >= 1.0 {
            continue;
        }
        for axis_update in update.iter_mut() {
            *axis_update *= weight;
        }
        clamp_vector3(update, max_update_norm);
    }
    updates
}

pub(crate) fn render_proxy_hard_target_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if best_distance2.is_finite() {
            residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
            residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
            residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
            counts[best_row] += 1;
        }
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        updates[row][0] = coverage_gain * residual_sums[row][0] / count as f32;
        updates[row][1] = coverage_gain * residual_sums[row][1] / count as f32;
        updates[row][2] = coverage_gain * residual_sums[row][2] / count as f32;
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}

pub(crate) fn render_proxy_soft_chamfer_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let samples = coverage_samples.max(rows.max(512));
    let sigma = if coverage_softness.is_finite() && coverage_softness > 0.0 {
        coverage_softness
    } else {
        target_coverage_threshold(seed_scale) * 1.5
    }
    .max(1.0e-4);
    let inv_two_sigma2 = 0.5 / (sigma * sigma);
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let normal_cost_scale = if coverage_normal_weight.is_finite() {
        coverage_normal_weight.max(0.0) * sigma * sigma
    } else {
        0.0
    };
    let mut projected_normals = vec![[0.0_f32; 3]; rows];
    for &row in active_rows {
        let projection = target.project([positions[row][0], positions[row][1], positions[row][2]]);
        projected_normals[row] = projection.normal;
        residual_sums[row][0] += 0.5 * projection.residual[0];
        residual_sums[row][1] += 0.5 * projection.residual[1];
        residual_sums[row][2] += 0.5 * projection.residual[2];
        weights[row] += 0.5;
    }

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_score = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            best_score = best_score.min(score);
        }
        if !best_score.is_finite() {
            continue;
        }

        let mut weight_sum = 0.0_f32;
        let mut sample_weights = Vec::with_capacity(active_rows.len());
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            let weight = (-(score - best_score) * inv_two_sigma2).exp();
            weight_sum += weight;
            sample_weights.push((row, weight));
        }
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            continue;
        }

        for (row, weight) in sample_weights {
            let normalized = weight / weight_sum;
            residual_sums[row][0] += normalized * (sample.position[0] - positions[row][0]);
            residual_sums[row][1] += normalized * (sample.position[1] - positions[row][1]);
            residual_sums[row][2] += normalized * (sample.position[2] - positions[row][2]);
            weights[row] += normalized;
        }
    }

    let mut repulsion_sums = vec![[0.0_f32; 3]; rows];
    if coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite() {
        let repulsion_radius =
            if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
                coverage_repulsion_radius
            } else {
                target_coverage_threshold(seed_scale) * 2.0
            }
            .max(1.0e-4);
        for lhs_idx in 0..active_rows.len() {
            let lhs = active_rows[lhs_idx];
            for &rhs in &active_rows[lhs_idx + 1..] {
                let dx = positions[lhs][0] - positions[rhs][0];
                let dy = positions[lhs][1] - positions[rhs][1];
                let dz = positions[lhs][2] - positions[rhs][2];
                let distance2 = dx * dx + dy * dy + dz * dz;
                if distance2 <= 1.0e-12 || distance2 >= repulsion_radius * repulsion_radius {
                    continue;
                }
                let distance = distance2.sqrt();
                let strength = (1.0 - distance / repulsion_radius).powi(2);
                let force = [
                    dx * strength / distance,
                    dy * strength / distance,
                    dz * strength / distance,
                ];
                let lhs_force = tangent_component(force, projected_normals[lhs]);
                let rhs_force =
                    tangent_component([-force[0], -force[1], -force[2]], projected_normals[rhs]);
                for axis in 0..3 {
                    repulsion_sums[lhs][axis] += lhs_force[axis];
                    repulsion_sums[rhs][axis] += rhs_force[axis];
                }
            }
        }
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        updates[row][0] = coverage_gain
            * (residual_sums[row][0] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][0]);
        updates[row][1] = coverage_gain
            * (residual_sums[row][1] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][1]);
        updates[row][2] = coverage_gain
            * (residual_sums[row][2] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}
