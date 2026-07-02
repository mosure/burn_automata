#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct NormalGapCandidate {
    pub(crate) position: [f32; 3],
    pub(crate) distance2: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_normal_coverage_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_normal_weight: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_normal_weight <= 0.0
        || !coverage_normal_weight.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    const NORMAL_CANDIDATES_PER_BIN: usize = 8;

    let directions = normal_coverage_directions();
    let bin_count = directions.len();
    let mut active_bin_counts = vec![0usize; bin_count];
    let mut active_bins = vec![usize::MAX; positions.len()];
    for &row in active_rows {
        if row >= positions.len() {
            continue;
        }
        let projection = target.project(position3(positions[row]));
        let bin = normal_direction_bin(projection.normal, &directions);
        active_bins[row] = bin;
        active_bin_counts[bin] += 1;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let mut target_bin_counts = vec![0usize; bin_count];
    let mut bin_candidates = vec![Vec::<NormalGapCandidate>::new(); bin_count];
    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = normal_direction_bin(sample.normal, &directions);
        target_bin_counts[bin] += 1;

        let mut nearest_distance2 = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            nearest_distance2 = nearest_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        if nearest_distance2.is_finite() {
            let candidate = NormalGapCandidate {
                position: sample.position,
                distance2: nearest_distance2,
            };
            let candidates = &mut bin_candidates[bin];
            candidates.push(candidate);
            candidates.sort_by(|lhs, rhs| {
                rhs.distance2
                    .partial_cmp(&lhs.distance2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(NORMAL_CANDIDATES_PER_BIN);
        }
    }

    let mut desired_bin_counts = vec![0usize; bin_count];
    for bin in 0..bin_count {
        if target_bin_counts[bin] == 0 {
            continue;
        }
        desired_bin_counts[bin] = ((target_bin_counts[bin] as f32 / sample_count as f32)
            * active_rows.len() as f32
            * 0.85)
            .ceil()
            .max(1.0) as usize;
    }

    let mut missing = active_bin_counts
        .iter()
        .zip(desired_bin_counts.iter())
        .map(|(active, desired)| desired.saturating_sub(*active))
        .sum::<usize>();
    if missing == 0 || bin_candidates.iter().all(Vec::is_empty) {
        return;
    }

    let mut used_donors = vec![false; positions.len()];
    let mut candidate_offsets = vec![0usize; bin_count];
    let max_relocated = missing.min(active_rows.len().saturating_mul(2).saturating_div(3).max(1));
    for _ in 0..max_relocated {
        let Some((gap_bin, gap)) = normal_gap_candidate(
            &active_bin_counts,
            &desired_bin_counts,
            &bin_candidates,
            &candidate_offsets,
        ) else {
            break;
        };
        let Some(row) = normal_gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &active_bins,
            &active_bin_counts,
            &desired_bin_counts,
            &used_donors,
        ) else {
            continue;
        };
        let donor_bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        if donor_bin < active_bin_counts.len() {
            active_bin_counts[donor_bin] = active_bin_counts[donor_bin].saturating_sub(1);
        }
        active_bin_counts[gap_bin] += 1;
        let scale = 0.5 * coverage_gain * coverage_normal_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
        candidate_offsets[gap_bin] += 1;
        missing = missing.saturating_sub(1);
        if missing == 0 {
            break;
        }
    }
}

pub(crate) fn normal_gap_candidate(
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    bin_candidates: &[Vec<NormalGapCandidate>],
    candidate_offsets: &[usize],
) -> Option<(usize, NormalGapCandidate)> {
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    for bin in 0..bin_candidates.len() {
        let deficit = desired_bin_counts
            .get(bin)
            .copied()
            .unwrap_or(0)
            .saturating_sub(active_bin_counts.get(bin).copied().unwrap_or(0));
        if deficit == 0 || bin_candidates[bin].is_empty() {
            continue;
        }
        let candidate_index = candidate_offsets
            .get(bin)
            .copied()
            .unwrap_or(0)
            .min(bin_candidates[bin].len() - 1);
        let candidate = bin_candidates[bin][candidate_index];
        let score = candidate.distance2 * (deficit as f32).sqrt();
        if score > best_score {
            best_score = score;
            best = Some((bin, candidate));
        }
    }
    best
}

pub(crate) fn normal_gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
) -> Option<usize> {
    normal_gap_relocation_donor_with_filter(
        gap_position,
        active_rows,
        positions,
        update_len,
        active_bins,
        active_bin_counts,
        desired_bin_counts,
        used_donors,
        true,
    )
    .or_else(|| {
        normal_gap_relocation_donor_with_filter(
            gap_position,
            active_rows,
            positions,
            update_len,
            active_bins,
            active_bin_counts,
            desired_bin_counts,
            used_donors,
            false,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normal_gap_relocation_donor_with_filter(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
    require_surplus_bin: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        let surplus = bin < active_bin_counts.len()
            && active_bin_counts[bin] > desired_bin_counts.get(bin).copied().unwrap_or(0).max(1);
        if require_surplus_bin && !surplus {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let score = if surplus { distance2 * 0.75 } else { distance2 };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}
