#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct SurfaceStrataCandidate {
    pub(crate) position: [f32; 3],
    pub(crate) score: f32,
    pub(crate) covered_fraction: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_strata_coverage_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    coverage_samples: usize,
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

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 128))
        .max(1);
    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let threshold2 = threshold * threshold;
    let mut bin_sample_counts = vec![0usize; bin_count];
    let mut bin_covered_counts = vec![0usize; bin_count];
    let mut bin_candidates = vec![None::<SurfaceStrataCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; positions.len()];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        bin_sample_counts[bin] += 1;
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_distance2 <= threshold2 {
            bin_covered_counts[bin] += 1;
        }
        let candidate = SurfaceStrataCandidate {
            position: sample.position,
            score: best_distance2,
            covered_fraction: 0.0,
        };
        if bin_candidates[bin].is_none_or(|current| best_distance2 > current.score) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut candidates = Vec::new();
    for bin in 0..bin_count {
        let samples = bin_sample_counts[bin];
        if samples == 0 {
            continue;
        }
        let covered_fraction = bin_covered_counts[bin] as f32 / samples as f32;
        if covered_fraction >= 0.60 {
            continue;
        }
        if let Some(mut candidate) = bin_candidates[bin] {
            candidate.covered_fraction = covered_fraction;
            candidate.score *= (0.60 - covered_fraction).max(0.0);
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        return;
    }
    candidates.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = candidates
        .len()
        .min(active_rows.len().saturating_mul(3).saturating_div(4).max(1));
    for candidate in candidates.into_iter().take(max_relocated) {
        let mut donor = surface_strata_relocation_donor(
            candidate.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if donor.is_none() {
            donor = surface_strata_relocation_donor(
                candidate.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = donor else {
            continue;
        };
        let donor_weight = if assigned_counts[row] == 0 { 1.0 } else { 0.6 };
        let strata_weight = (0.60 - candidate.covered_fraction).clamp(0.0, 1.0);
        let scale = coverage_gain * coverage_gap_gain * donor_weight * strata_weight;
        updates[row][0] += scale * (candidate.position[0] - positions[row][0]);
        updates[row][1] += scale * (candidate.position[1] - positions[row][1]);
        updates[row][2] += scale * (candidate.position[2] - positions[row][2]);
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn surface_strata_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    require_surplus: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    let average_assignments = average_assignments.max(1.0);
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let surplus = assignments > average_assignments.ceil().max(1.0);
        if require_surplus && assignments > 0.0 && !surplus {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let assignment_factor = if assignments == 0.0 {
            0.5
        } else if surplus {
            0.75 / (assignments / average_assignments).sqrt()
        } else {
            1.0 + 0.25 * assignments / average_assignments
        };
        let score = distance2 * assignment_factor;
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}
