#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[derive(Clone, Copy)]
struct GapCandidate {
    position: [f32; 3],
    score: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_weighted_surface_gap_relocation_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    row_weights: &[f32],
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
        || row_weights.len() < positions.len()
    {
        return;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let threshold = target_coverage_threshold(seed_scale);
    let threshold2 = threshold * threshold;
    let normal_cost_scale = if coverage_normal_weight.is_finite() && coverage_normal_weight > 0.0 {
        coverage_normal_weight * threshold2.max(1.0e-6)
    } else {
        0.0
    };
    let projected_normals =
        projected_active_normals(target, positions, active_rows, normal_cost_scale);
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
            let confidence = row_weights[row].clamp(1.0e-3, 1.0);
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_penalty = projected_normals.as_ref().map_or(0.0, |normals| {
                normal_cost_scale * (1.0 - dot3(sample.normal, normals[row]).clamp(-1.0, 1.0))
            });
            let score = (distance2 + normal_penalty) / confidence.sqrt();
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

    relocate_gap_donors(
        positions,
        active_rows,
        row_weights,
        coverage_gain,
        coverage_gap_gain,
        max_update_norm,
        updates,
        &gaps,
        &assigned_counts,
        sample_count,
    );
}

fn projected_active_normals(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    normal_cost_scale: f32,
) -> Option<Vec<[f32; 3]>> {
    if normal_cost_scale <= 0.0 {
        return None;
    }

    let mut normals = vec![[0.0_f32; 3]; positions.len()];
    for &row in active_rows {
        if row < positions.len() {
            normals[row] = target.project(position3(positions[row])).normal;
        }
    }
    Some(normals)
}

#[allow(clippy::too_many_arguments)]
fn relocate_gap_donors(
    positions: &[[f32; 4]],
    active_rows: &[usize],
    row_weights: &[f32],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
    gaps: &[GapCandidate],
    assigned_counts: &[usize],
    sample_count: usize,
) {
    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = gaps.len().min(active_rows.len().saturating_div(2).max(1));
    let mut relocated = 0usize;

    for gap in gaps.iter().copied() {
        if relocated >= max_relocated {
            break;
        }
        let mut best_row = weighted_gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            assigned_counts,
            average_assignments,
            &used_donors,
            row_weights,
            true,
        );
        if best_row.is_none() {
            best_row = weighted_gap_relocation_donor(
                gap.position,
                active_rows,
                positions,
                updates.len(),
                assigned_counts,
                average_assignments,
                &used_donors,
                row_weights,
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
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
        relocated += 1;
    }

    relocate_unassigned_rows(
        positions,
        active_rows,
        row_weights,
        coverage_gain,
        coverage_gap_gain,
        max_update_norm,
        updates,
        gaps,
        assigned_counts,
        &used_donors,
    );
}

#[allow(clippy::too_many_arguments)]
fn relocate_unassigned_rows(
    positions: &[[f32; 4]],
    active_rows: &[usize],
    row_weights: &[f32],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
    gaps: &[GapCandidate],
    assigned_counts: &[usize],
    used_donors: &[bool],
) {
    for &row in active_rows {
        if row >= positions.len() || row >= updates.len() || !row_weights[row].is_finite() {
            continue;
        }
        if assigned_counts[row] > 0 || used_donors[row] || row_weights[row] <= 1.0e-3 {
            continue;
        }
        let mut nearest_gap = gaps[0];
        let mut nearest_gap_distance2 = f32::MAX;
        for gap in gaps {
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
        clamp_update_row(updates, row, max_update_norm);
    }
}

#[allow(clippy::too_many_arguments)]
fn weighted_gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    row_weights: &[f32],
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
        let confidence = row_weights.get(row).copied().unwrap_or(0.0);
        if confidence <= 1.0e-3 || !confidence.is_finite() {
            continue;
        }
        let confidence = confidence.clamp(1.0e-3, 1.0);
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let under_assigned = assignments <= under_assigned_limit || confidence >= 0.75;
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
        let confidence_bonus = confidence.sqrt();
        let score = if require_under_assigned {
            distance2 * (1.0 + 0.25 * assignment_penalty) / confidence_bonus
        } else {
            distance2 * (1.0 + 0.25 * assignment_penalty)
                / (overflow_bonus.sqrt() * confidence_bonus)
        };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}
