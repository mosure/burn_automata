#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_gap_relocation_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    coverage_samples: usize,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_gap_gain <= 0.0
        || !coverage_gap_gain.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        score: f32,
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let threshold = target_coverage_threshold(seed_scale);
    let threshold2 = threshold * threshold;
    let normal_cost_scale = if coverage_normal_weight.is_finite() && coverage_normal_weight > 0.0 {
        coverage_normal_weight * threshold2.max(1.0e-6)
    } else {
        0.0
    };
    let projected_normals = if normal_cost_scale > 0.0 {
        let mut normals = vec![[0.0_f32; 3]; positions.len()];
        for &row in active_rows {
            if row < positions.len() {
                normals[row] = target.project(position3(positions[row])).normal;
            }
        }
        Some(normals)
    } else {
        None
    };
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; positions.len()];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_score = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_penalty = projected_normals.as_ref().map_or(0.0, |normals| {
                normal_cost_scale * (1.0 - dot3(sample.normal, normals[row]).clamp(-1.0, 1.0))
            });
            let score = distance2 + normal_penalty;
            if score < best_score {
                best_score = score;
                best_row = row;
            }
        }
        if !best_score.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_score <= threshold2 {
            continue;
        }
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            score: best_score,
        };
        if bin_candidates[bin].is_none_or(|current| best_score > current.score) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    if gaps.is_empty() {
        return;
    }
    gaps.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = gaps.len().min(active_rows.len().saturating_div(2).max(1));
    let mut relocated = 0usize;
    for gap in gaps.iter().copied() {
        if relocated >= max_relocated {
            break;
        }
        let mut best_row = gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if best_row.is_none() {
            best_row = gap_relocation_donor(
                gap.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = best_row else {
            continue;
        };
        let donor_weight = if assigned_counts[row] == 0 { 1.0 } else { 0.5 };
        let scale = 0.5 * coverage_gain * coverage_gap_gain * donor_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
        used_donors[row] = true;
        relocated += 1;
    }

    for &row in active_rows {
        if row >= positions.len() || row >= updates.len() {
            continue;
        }
        if assigned_counts[row] > 0 || used_donors[row] {
            continue;
        }
        let mut nearest_gap = gaps[0];
        let mut nearest_gap_distance2 = f32::MAX;
        for gap in &gaps {
            let dx = gap.position[0] - positions[row][0];
            let dy = gap.position[1] - positions[row][1];
            let dz = gap.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_gap_distance2 {
                nearest_gap_distance2 = distance2;
                nearest_gap = *gap;
            }
        }
        if !nearest_gap_distance2.is_finite() {
            continue;
        }
        let scale = 0.5 * coverage_gain * coverage_gap_gain;
        updates[row][0] += scale * (nearest_gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (nearest_gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (nearest_gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    require_under_assigned: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    let average_assignments = average_assignments.max(1.0);
    let under_assigned_limit = average_assignments.ceil().max(1.0);
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let under_assigned = assignments <= under_assigned_limit;
        if require_under_assigned
            && assigned_counts.get(row).copied().unwrap_or(0) > 0
            && !under_assigned
        {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let assignment_penalty = assignments / average_assignments;
        let overflow_bonus = (assignments / under_assigned_limit).max(1.0);
        let score = if require_under_assigned {
            distance2 * (1.0 + 0.25 * assignment_penalty)
        } else {
            distance2 * (1.0 + 0.25 * assignment_penalty) / overflow_bonus.sqrt()
        };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}
