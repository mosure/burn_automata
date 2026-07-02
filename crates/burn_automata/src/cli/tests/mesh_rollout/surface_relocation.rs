use super::*;

#[test]
fn surface_tangent_repulsion_separates_close_surface_particles() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let tangent = if sample.normal[0].abs() < 0.9 {
        [0.0, -sample.normal[2], sample.normal[1]]
    } else {
        [-sample.normal[1], sample.normal[0], 0.0]
    };
    let tangent_norm =
        (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let tangent = [
        tangent[0] / tangent_norm,
        tangent[1] / tangent_norm,
        tangent[2] / tangent_norm,
    ];
    let positions = vec![
        [
            sample.position[0] - 0.01 * tangent[0],
            sample.position[1] - 0.01 * tangent[1],
            sample.position[2] - 0.01 * tangent[2],
            1.0,
        ],
        [
            sample.position[0] + 0.01 * tangent[0],
            sample.position[1] + 0.01 * tangent[1],
            sample.position[2] + 0.01 * tangent[2],
            1.0,
        ],
    ];
    let mut updates = vec![[0.0; 3]; positions.len()];
    add_surface_tangent_repulsion_to_updates(
        &target,
        &positions,
        &[0, 1],
        1.0,
        1.0,
        0.08,
        0.72,
        1.0,
        &mut updates,
    );

    let lhs_dot =
        updates[0][0] * -tangent[0] + updates[0][1] * -tangent[1] + updates[0][2] * -tangent[2];
    let rhs_dot =
        updates[1][0] * tangent[0] + updates[1][1] * tangent[1] + updates[1][2] * tangent[2];
    assert!(
        lhs_dot > 0.0 && rhs_dot > 0.0,
        "repulsion should push close particles apart along the surface tangent, updates={updates:?}"
    );
    let projected_normal = target
        .project([positions[0][0], positions[0][1], positions[0][2]])
        .normal;
    assert!(
        (updates[0][0] * projected_normal[0]
            + updates[0][1] * projected_normal[1]
            + updates[0][2] * projected_normal[2])
            .abs()
            < 1.0e-4,
        "repulsion should remove the projected normal component"
    );
}

#[test]
fn surface_gap_relocation_moves_redundant_particles_to_uncovered_regions() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
    ];
    let mut updates = vec![[0.0; 3]; positions.len()];
    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &[0, 1],
        1.0,
        1.0,
        512,
        0.0,
        0.72,
        1.0,
        &mut updates,
    );

    let update_norms = updates
        .iter()
        .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
        .collect::<Vec<_>>();
    let redundant_norm = update_norms.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        redundant_norm > 0.05,
        "a redundant active particle should receive a relocation update toward an uncovered surface gap, updates={updates:?}"
    );
    assert!(
        update_norms.iter().all(|norm| *norm <= 1.0 + 1.0e-5),
        "gap relocation should respect max_update_norm, norms={update_norms:?}"
    );
}

#[test]
fn surface_normal_coverage_moves_redundant_particles_to_missing_normal_bins() {
    let target = uv_torus_mesh_target(0.72);
    let sample = target.surface_sample(0);
    let positions = vec![
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
        [
            sample.position[0],
            sample.position[1],
            sample.position[2],
            1.0,
        ],
    ];
    let active_rows = [0, 1, 2];
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_normal_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        1.0,
        &mut updates,
    );

    let max_update_norm = updates
        .iter()
        .map(|update| (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    assert!(
        max_update_norm > 0.05,
        "normal-bin coverage should relocate a redundant particle toward an under-covered normal bin, updates={updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_normal_coverage_fills_normal_bin_deficits_with_multiple_donors() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.04],
            [0.08, 0.0, 0.04],
            [0.0, 0.08, 0.04],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let positions = vec![
        [0.010, 0.010, 0.0, 1.0],
        [0.020, 0.010, 0.0, 1.0],
        [0.010, 0.020, 0.0, 1.0],
        [0.030, 0.010, 0.0, 1.0],
        [0.010, 0.030, 0.0, 1.0],
        [0.020, 0.020, 0.0, 1.0],
        [0.035, 0.015, 0.0, 1.0],
        [0.015, 0.035, 0.0, 1.0],
    ];
    let active_rows = (0..positions.len()).collect::<Vec<_>>();
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_normal_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        1.0,
        &mut updates,
    );

    let upward_relocations = updates.iter().filter(|update| update[2] > 1.0e-3).count();
    assert!(
        upward_relocations >= 3,
        "normal coverage should fill an opposite-normal deficit with multiple donors, updates={updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn surface_normal_coverage_report_detects_missing_opposite_normal_support() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.04],
            [0.08, 0.0, 0.04],
            [0.0, 0.08, 0.04],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let lower_only = vec![[0.02, 0.02, 0.0, 1.0], [0.04, 0.02, 0.0, 1.0]];
    let both_sides = vec![
        [0.02, 0.02, 0.0, 1.0],
        [0.04, 0.02, 0.0, 1.0],
        [0.02, 0.02, 0.04, 1.0],
        [0.04, 0.02, 0.04, 1.0],
    ];

    let missing = surface_normal_coverage_report(&lower_only, &target, 512, 0.012);
    let covered = surface_normal_coverage_report(&both_sides, &target, 512, 0.012);

    assert!(
        missing.covered_target_bin_fraction < covered.covered_target_bin_fraction,
        "normal coverage should detect that one of the target normal families is absent: missing={missing:?} covered={covered:?}"
    );
    assert!(covered.covered_target_bin_fraction >= 0.99);
    assert!(covered.mean_bin_covered_fraction > missing.mean_bin_covered_fraction);
}

#[test]
fn surface_gap_relocation_can_use_normal_mismatch_as_uncovered_support() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.02],
            [0.01, 0.0, 0.02],
            [0.0, 0.01, 0.02],
        ],
        vec![[0, 1, 2], [5, 4, 3]],
    )
    .unwrap();
    let positions = vec![[0.003, 0.003, 0.0, 1.0], [0.006, 0.002, 0.0, 1.0]];
    let active_rows = [0, 1];
    let mut position_only = vec![[0.0; 3]; positions.len()];
    let mut normal_aware = vec![[0.0; 3]; positions.len()];

    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.0,
        0.72,
        1.0,
        &mut position_only,
    );
    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        10.0,
        0.72,
        1.0,
        &mut normal_aware,
    );

    let position_only_z = position_only
        .iter()
        .map(|update| update[2].abs())
        .fold(0.0_f32, f32::max);
    let normal_aware_z = normal_aware
        .iter()
        .map(|update| update[2])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        normal_aware_z > position_only_z + 1.0e-3,
        "normal-aware gap relocation should expose nearby opposite-normal support: position_only={position_only:?} normal_aware={normal_aware:?}"
    );
}

#[test]
fn gap_relocation_donor_falls_back_to_overassigned_particles() {
    let active_rows = [0, 1];
    let positions = vec![[0.0, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, 1.0]];
    let assigned_counts = vec![16, 12];
    let used_donors = vec![false, false];
    let average_assignments = 8.0;
    let gap = [0.5, 0.0, 0.0];

    let under_assigned = gap_relocation_donor(
        gap,
        &active_rows,
        &positions,
        positions.len(),
        &assigned_counts,
        average_assignments,
        &used_donors,
        true,
    );
    let fallback = gap_relocation_donor(
        gap,
        &active_rows,
        &positions,
        positions.len(),
        &assigned_counts,
        average_assignments,
        &used_donors,
        false,
    );

    assert_eq!(under_assigned, None);
    assert_eq!(
        fallback,
        Some(1),
        "uncovered surface patches should still get a donor when every active particle is already assigned"
    );
}
