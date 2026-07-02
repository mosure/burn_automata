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
