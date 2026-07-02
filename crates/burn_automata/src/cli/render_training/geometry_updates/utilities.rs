#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn normal_direction_bin(normal: [f32; 3], directions: &[[f32; 3]]) -> usize {
    let normal = normalize3_or(normal, [0.0, 0.0, 1.0]);
    let mut best_bin = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (idx, direction) in directions.iter().enumerate() {
        let score = dot3(normal, *direction);
        if score > best_dot {
            best_dot = score;
            best_bin = idx;
        }
    }
    best_bin
}

pub(crate) fn normalize3_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 || !norm.is_finite() {
        fallback
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

pub(crate) fn normal_coverage_directions() -> [[f32; 3]; 26] {
    const INV_SQRT_2: f32 = 0.707_106_77;
    const INV_SQRT_3: f32 = 0.577_350_26;
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [INV_SQRT_2, INV_SQRT_2, 0.0],
        [INV_SQRT_2, -INV_SQRT_2, 0.0],
        [-INV_SQRT_2, INV_SQRT_2, 0.0],
        [-INV_SQRT_2, -INV_SQRT_2, 0.0],
        [INV_SQRT_2, 0.0, INV_SQRT_2],
        [INV_SQRT_2, 0.0, -INV_SQRT_2],
        [-INV_SQRT_2, 0.0, INV_SQRT_2],
        [-INV_SQRT_2, 0.0, -INV_SQRT_2],
        [0.0, INV_SQRT_2, INV_SQRT_2],
        [0.0, INV_SQRT_2, -INV_SQRT_2],
        [0.0, -INV_SQRT_2, INV_SQRT_2],
        [0.0, -INV_SQRT_2, -INV_SQRT_2],
        [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
    ]
}

pub(crate) fn clamp_update_row(updates: &mut [[f32; 3]], row: usize, max_update_norm: f32) {
    if row >= updates.len() {
        return;
    }
    clamp_vector3(&mut updates[row], max_update_norm);
}

pub(crate) fn clamp_vector3(update: &mut [f32; 3], max_update_norm: f32) {
    let norm = (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for value in update {
            *value *= clamp;
        }
    }
}

pub(crate) fn render_proxy_gap_farthest_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        distance2: f32,
    }

    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(rows.max(512));
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; rows];

    for sample_idx in 0..sample_count {
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
        if !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;

        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            distance2: best_distance2,
        };
        if bin_candidates[bin].is_none_or(|current| best_distance2 > current.distance2) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    gaps.sort_by(|lhs, rhs| {
        rhs.distance2
            .partial_cmp(&lhs.distance2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; rows];
    let max_relocated = gaps.len().min(active_rows.len().max(1));
    for candidate in gaps.into_iter().take(max_relocated) {
        let mut donor = gap_relocation_donor(
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
            donor = gap_relocation_donor(
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
        let residual = [
            candidate.position[0] - positions[row][0],
            candidate.position[1] - positions[row][1],
            candidate.position[2] - positions[row][2],
        ];
        let weight = candidate.distance2.sqrt().max(1.0e-4);
        for axis in 0..3 {
            residual_sums[row][axis] += residual[axis] * weight;
        }
        weights[row] += weight;
        used_donors[row] = true;
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            let residual = if weights[row] > 0.0 {
                residual_sums[row][axis] / weights[row] + 0.25 * projection.residual[axis]
            } else {
                0.25 * projection.residual[axis]
            };
            updates[row][axis] = coverage_gain * residual;
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }

    updates
}

pub(crate) fn tangent_component(vector: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let normal_norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if normal_norm2 <= 1.0e-12 {
        return vector;
    }
    let dot = vector[0] * normal[0] + vector[1] * normal[1] + vector[2] * normal[2];
    [
        vector[0] - normal[0] * dot / normal_norm2,
        vector[1] - normal[1] * dot / normal_norm2,
        vector[2] - normal[2] * dot / normal_norm2,
    ]
}

pub(crate) fn reference_seed_scale_for_seed_mode(
    preset: AutomataPreset,
    seed_mode: ParticleSeed,
) -> f32 {
    match seed_mode {
        ParticleSeed::UvTorus3d
        | ParticleSeed::UvTorusDense3d
        | ParticleSeed::TorusFieldDense3d
        | ParticleSeed::TeapotFieldDense3d
        | ParticleSeed::TorusGrowth3d
        | ParticleSeed::TeapotGrowth3d
        | ParticleSeed::TorusSubstrateGrowth3d
        | ParticleSeed::TeapotSubstrateGrowth3d
        | ParticleSeed::TorusLocalGrowth3d
        | ParticleSeed::TeapotLocalGrowth3d
        | ParticleSeed::TorusLocalSubstrateGrowth3d
        | ParticleSeed::TeapotLocalSubstrateGrowth3d
        | ParticleSeed::TorusMorphogenDense3d
        | ParticleSeed::TeapotMorphogenDense3d => DEFAULT_3D_MESH_FIELD_SCALE,
        _ => NpaConfig::seed_scale_for_preset(preset),
    }
}

pub(crate) fn default_train_target_seed(
    _preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> Option<u64> {
    if zero_update {
        None
    } else {
        Some(target_seed.unwrap_or(DEFAULT_GROWTH_TARGET_SEED))
    }
}

pub(crate) fn train_target_source(
    preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> String {
    match (target_seed, zero_update) {
        (Some(seed), false) => format!("seeded:{preset:?}:{seed}"),
        (None, true) => "explicit-zero-update".to_string(),
        _ => unreachable!("target seed/source selection should be normalized first"),
    }
}

pub(crate) fn training_source_with_batch(
    batch_source: TrainingBatchArg,
    target_source: &str,
) -> String {
    match batch_source {
        TrainingBatchArg::Rollout => format!("rollout-local:{target_source}"),
        TrainingBatchArg::Features => format!("feature-rows:{target_source}"),
    }
}

pub(crate) fn render_proxy_sliced_ot_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let samples = (0..sample_count)
        .map(|sample_idx| target.surface_sample(sample_idx).position)
        .collect::<Vec<_>>();
    let directions = sliced_ot_directions();
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];

    for direction in directions {
        let mut target_order = (0..samples.len()).collect::<Vec<_>>();
        target_order.sort_by(|&lhs, &rhs| {
            dot3(samples[lhs], direction)
                .partial_cmp(&dot3(samples[rhs], direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut active_order = active_rows.to_vec();
        active_order.sort_by(|&lhs, &rhs| {
            dot3(position3(positions[lhs]), direction)
                .partial_cmp(&dot3(position3(positions[rhs]), direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let active_len = active_order.len().max(1);
        for (rank, &row) in active_order.iter().enumerate() {
            let sample_rank = (((rank as f32 + 0.5) * sample_count as f32 / active_len as f32)
                .floor() as usize)
                .min(sample_count - 1);
            let sample = samples[target_order[sample_rank]];
            for axis in 0..3 {
                residual_sums[row][axis] += sample[axis] - positions[row][axis];
            }
            weights[row] += 1.0;
        }
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            residual_sums[row][axis] += 0.25 * projection.residual[axis];
        }
        weights[row] += 0.25;
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        for axis in 0..3 {
            updates[row][axis] = coverage_gain * residual_sums[row][axis] / weights[row];
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }
    updates
}

pub(crate) fn sliced_ot_directions() -> [[f32; 3]; 26] {
    normal_coverage_directions()
}

pub(crate) fn position3(position: [f32; 4]) -> [f32; 3] {
    [position[0], position[1], position[2]]
}
