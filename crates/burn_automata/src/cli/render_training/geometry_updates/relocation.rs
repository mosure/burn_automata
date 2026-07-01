#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_tangent_repulsion_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_repulsion_gain <= 0.0
        || !coverage_repulsion_gain.is_finite()
        || active_rows.len() < 2
    {
        return;
    }
    let radius = if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
        coverage_repulsion_radius
    } else {
        target_coverage_threshold(seed_scale) * 2.0
    }
    .max(1.0e-4);
    let radius2 = radius * radius;
    let mut projected_normals = vec![[0.0_f32; 3]; positions.len()];
    for &row in active_rows {
        if row < positions.len() {
            projected_normals[row] = target.project(position3(positions[row])).normal;
        }
    }
    let mut repulsion_sums = vec![[0.0_f32; 3]; positions.len()];
    let mut counts = vec![0usize; positions.len()];
    for lhs_idx in 0..active_rows.len() {
        let lhs = active_rows[lhs_idx];
        if lhs >= positions.len() {
            continue;
        }
        for &rhs in &active_rows[lhs_idx + 1..] {
            if rhs >= positions.len() {
                continue;
            }
            let dx = positions[lhs][0] - positions[rhs][0];
            let dy = positions[lhs][1] - positions[rhs][1];
            let dz = positions[lhs][2] - positions[rhs][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 <= 1.0e-12 || distance2 >= radius2 {
                continue;
            }
            let distance = distance2.sqrt();
            let strength = (1.0 - distance / radius).powi(2);
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
            counts[lhs] += 1;
            counts[rhs] += 1;
        }
    }
    for &row in active_rows {
        if row >= updates.len() || counts[row] == 0 {
            continue;
        }
        let scale = coverage_gain * coverage_repulsion_gain / counts[row] as f32;
        for axis in 0..3 {
            updates[row][axis] += scale * repulsion_sums[row][axis];
        }
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
