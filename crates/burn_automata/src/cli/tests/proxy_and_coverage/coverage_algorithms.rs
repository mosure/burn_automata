use super::*;

#[test]
fn soft_chamfer_coverage_distributes_symmetric_target_pressure() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[-0.02, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let hard = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::HardNearest,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let soft = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let soft_nonzero = soft
        .iter()
        .filter(|update| update.iter().any(|value| value.abs() > 1.0e-6))
        .count();

    assert!(hard.iter().flatten().all(|value| value.is_finite()));
    assert_eq!(soft_nonzero, 2);
    assert!(soft[0][0] > 0.0);
    assert!(soft[1][0] < 0.0);
}
#[test]
fn weighted_target_coverage_updates_include_local_front_rows() {
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let positions = vec![
        [-1.0_f32, 0.0, 0.0, 0.0],
        [0.72_f32, 0.0, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0, 0.0],
    ];
    let max_update_norm = 0.25;
    let updates = render_proxy_weighted_target_coverage_updates(
        &target,
        &positions,
        &[1.0, 0.5, 0.0],
        1.0,
        256,
        max_update_norm,
        CoverageUpdateModeArg::HardNearest,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let front_norm = (updates[1][0].powi(2) + updates[1][1].powi(2) + updates[1][2].powi(2)).sqrt();

    assert!(
        updates[1][0] > 0.0,
        "weighted local-front row should receive pressure toward the uncovered target lobe"
    );
    assert!(front_norm <= max_update_norm + 1.0e-6);
    assert_eq!(
        updates[2],
        [0.0, 0.0, 0.0],
        "zero-weight row should remain untouched"
    );
}
#[test]
fn soft_chamfer_coverage_respects_update_clamp() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[3.0, 4.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let max_update_norm = 0.05;

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        max_update_norm,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let norm = (updates[0][0].powi(2) + updates[0][1].powi(2) + updates[0][2].powi(2)).sqrt();

    assert!(norm <= max_update_norm + 1.0e-6);
    assert!(norm > 0.0);
}
#[test]
fn soft_chamfer_repulsion_adds_tangent_spread_pressure() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let positions = vec![[-0.005, 0.0, 0.0, 0.0], [0.005, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];
    let no_repulsion = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let with_repulsion = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        1,
        f32::INFINITY,
        CoverageUpdateModeArg::SoftChamfer,
        0.1,
        1.0,
        0.0,
        0.1,
        0.0,
        1.0,
    );

    assert!(no_repulsion[0][0] > 0.0);
    assert!(no_repulsion[1][0] < 0.0);
    assert!(with_repulsion[0][0] < no_repulsion[0][0]);
    assert!(with_repulsion[1][0] > no_repulsion[1][0]);
    assert!(with_repulsion[0][2].abs() <= 1.0e-6);
    assert!(with_repulsion[1][2].abs() <= 1.0e-6);
}
#[test]
fn gap_farthest_coverage_avoids_symmetric_residual_cancellation() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let positions = vec![[0.0, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let hard = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::HardNearest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let gap = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::GapFarthest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let hard_norm = (hard[0][0].powi(2) + hard[0][1].powi(2) + hard[0][2].powi(2)).sqrt();
    let gap_norm = (gap[0][0].powi(2) + gap[0][1].powi(2) + gap[0][2].powi(2)).sqrt();

    assert!(hard.iter().flatten().all(|value| value.is_finite()));
    assert!(gap.iter().flatten().all(|value| value.is_finite()));
    assert!(
        gap_norm > hard_norm + 0.1,
        "gap mode should keep a directional worst-gap signal instead of averaging it away: hard={hard:?} gap={gap:?}"
    );
}
#[test]
fn gap_farthest_coverage_balances_uncovered_bins_across_donors() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )
    .unwrap();
    let positions = vec![[-1.0, -0.04, 0.05, 0.0], [-0.95, 0.04, 0.05, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let gap = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        10.0,
        CoverageUpdateModeArg::GapFarthest,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert!(
        gap.iter().all(|update| update[0] > 0.1),
        "balanced gap mode should spread uncovered right-side bins across available donors: {gap:?}"
    );
    assert!(gap.iter().flatten().all(|value| value.is_finite()));
}
#[test]
fn surface_strata_coverage_moves_redundant_rows_to_empty_surface_bins() {
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let positions = vec![
        [-1.02_f32, -0.04, 0.05, 0.0],
        [-1.00_f32, 0.04, 0.05, 0.0],
        [-0.98_f32, 0.0, 0.08, 0.0],
        [-1.04_f32, 0.0, 0.02, 0.0],
    ];
    let active_rows = vec![0, 1, 2, 3];
    let mut updates = vec![[0.0_f32; 3]; positions.len()];

    add_surface_strata_coverage_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.5,
        10.0,
        &mut updates,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.15),
        "strata coverage should move at least one redundant left-patch row toward the uncovered right patch: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}
#[test]
fn surface_gap_relocation_can_use_low_assignment_donors() {
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )
    .unwrap();
    let positions = vec![[-1.0, 0.0, 0.05, 0.0], [0.0, 0.0, 0.05, 0.0]];
    let active_rows = vec![0, 1];
    let mut updates = vec![[0.0; 3]; positions.len()];

    add_surface_gap_relocation_to_updates(
        &target,
        &positions,
        &active_rows,
        1.0,
        1.0,
        512,
        0.0,
        1.0,
        10.0,
        &mut updates,
    );

    assert!(
        updates.iter().any(|update| update[0] > 0.1),
        "a nonzero-assigned donor should be allowed to move toward the uncovered right mode: {updates:?}"
    );
    assert!(updates.iter().flatten().all(|value| value.is_finite()));
}
#[test]
fn sliced_ot_coverage_balances_separated_surface_modes() {
    let config = NpaConfig::growing_3dgs();
    let target = TriangleMeshTarget::new(
        vec![
            [-1.0, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.2],
            [1.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.2],
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let positions = vec![[-0.05, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]];
    let states = vec![0.0; positions.len() * config.state_dims];

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        512,
        f32::INFINITY,
        CoverageUpdateModeArg::SlicedOt,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    assert!(
        updates[0][0] < 0.0,
        "left-ranked particle should be pulled toward the left target mode: {updates:?}"
    );
    assert!(
        updates[1][0] > 0.0,
        "right-ranked particle should be pulled toward the right target mode: {updates:?}"
    );
}
#[test]
fn sliced_ot_coverage_pushes_collapsed_torus_centerline_into_tube() {
    let config = NpaConfig::growing_3dgs();
    let scale = 0.54_f32;
    let target = mesh_target_for_arg(MeshTargetArg::Torus, scale);
    let ring_count = 16usize;
    let positions = (0..ring_count)
        .map(|idx| {
            let theta = std::f32::consts::TAU * idx as f32 / ring_count as f32;
            [scale * theta.cos(), scale * theta.sin(), 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    let states = vec![0.0; positions.len() * config.state_dims];

    let updates = render_proxy_target_coverage_updates(
        &config,
        &target,
        &positions,
        &states,
        1.0,
        2048,
        f32::INFINITY,
        CoverageUpdateModeArg::SlicedOt,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
    );

    assert_eq!(
        sliced_ot_directions().len(),
        normal_coverage_directions().len()
    );
    let tube_pressure = updates
        .iter()
        .enumerate()
        .map(|(idx, update)| {
            let theta = std::f32::consts::TAU * idx as f32 / ring_count as f32;
            let radial = update[0] * theta.cos() + update[1] * theta.sin();
            radial.abs() + update[2].abs()
        })
        .sum::<f32>();

    assert!(
        tube_pressure > 1.0e-3,
        "collapsed centerline should receive tube-plane pressure toward the full surface support: {updates:?}"
    );
}
