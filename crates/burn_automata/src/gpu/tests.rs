use burn_automata_kernels::HashGridConfig;

use super::{helpers::*, types::*};
use crate::{AutomataPreset, NpaConfig, ParticleSeed, rollout::seed_particles_scaled};

#[test]
fn gpu_parameter_uniform_covers_every_host_parameter() {
    let shader = include_str!("../gpu_step.wgsl");
    let expected = format!("values: array<vec4<u32>, {}>", PARAM_COUNT / 4);
    assert!(
        shader.contains(&expected),
        "WGSL parameter uniform must contain {PARAM_COUNT} host scalars"
    );
}

#[test]
fn paired_topology_composed_wgsl_is_valid() {
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../gpu_step.wgsl"),
        include_str!("../gpu_pca.wgsl"),
        burn_automata_kernels::PAIRED_LOCAL_DETAIL_TOPOLOGY_WGSL,
    );
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("paired topology WGSL parse failed: {error}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("paired topology WGSL validation failed: {error}"));
}

#[test]
fn support_bin_grid_layout_keeps_one_particle_index_and_bounded_cells() {
    let layout = resolve_support_bin_grid_layout(
        4_096,
        1,
        4_096,
        0,
        WgpuNeighborMode::CooperativeSortedCells,
        4,
    )
    .unwrap();

    assert_eq!(layout.support_bin_count, 4);
    assert_eq!(layout.cell_count, 16_384);
    assert_eq!(
        layout.storage_len,
        sorted_grid_storage_len(layout.cell_count, 4_096).unwrap(),
    );
}

#[test]
fn support_bin_grid_layout_falls_back_before_scan_surface_explodes() {
    let layout = resolve_support_bin_grid_layout(
        256,
        32,
        32 * 4_096,
        0,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
        16,
    )
    .unwrap();

    assert_eq!(layout.support_bin_count, 1);
    assert_eq!(layout.cell_count, 8_192);
}

#[test]
fn support_bins_do_not_enable_on_unconverted_sorted_traversal() {
    let layout =
        resolve_support_bin_grid_layout(256, 1, 4_096, 0, WgpuNeighborMode::SortedCells, 8)
            .unwrap();

    assert_eq!(layout.support_bin_count, 1);
}

#[test]
fn support_bins_stay_inactive_for_diffuse_multiscale_particles() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 4_096;
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        4,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.8,
    );
    let bandwidth = (0..particle_count)
        .map(|index| {
            if index < particle_count / 10 {
                0.2
            } else {
                0.025
            }
        })
        .collect::<Vec<_>>();

    assert!(!should_activate_support_bins(
        &grid,
        particle_count,
        &positions,
        &bandwidth,
        0.025,
        0.2,
        2.0,
    ));
}

#[test]
fn support_bins_activate_for_concentrated_multiscale_particles() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 4_096;
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        4,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let bandwidth = (0..particle_count)
        .map(|index| {
            if index < particle_count / 10 {
                0.2
            } else {
                0.025
            }
        })
        .collect::<Vec<_>>();

    assert!(should_activate_support_bins(
        &grid,
        particle_count,
        &positions,
        &bandwidth,
        0.025,
        0.2,
        2.0,
    ));
}

#[test]
fn support_bins_stay_inactive_for_sparse_scale_tail() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 4_096;
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        4,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let bandwidth = (0..particle_count)
        .map(|index| {
            if index < particle_count / 100 {
                0.2
            } else {
                0.025
            }
        })
        .collect::<Vec<_>>();

    assert!(!should_activate_support_bins(
        &grid,
        particle_count,
        &positions,
        &bandwidth,
        0.025,
        0.2,
        2.0,
    ));
}

#[test]
fn support_bins_stay_inactive_for_balanced_scale_histogram() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 4_096;
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        4,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let bandwidth = (0..particle_count)
        .map(|index| {
            let fraction = (index * 977 % particle_count) as f32 / (particle_count - 1) as f32;
            0.025 * 8.0_f32.powf(fraction)
        })
        .collect::<Vec<_>>();

    assert!(!should_activate_support_bins(
        &grid,
        particle_count,
        &positions,
        &bandwidth,
        0.025,
        0.2,
        2.0,
    ));
}

#[test]
fn support_bins_stay_inactive_without_scale_diversity() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 4_096;
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        4,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let bandwidth = vec![0.05; particle_count];

    assert!(!should_activate_support_bins(
        &grid,
        particle_count,
        &positions,
        &bandwidth,
        0.025,
        0.2,
        2.0,
    ));
}

#[test]
fn auto_bucket_capacity_helper_keeps_particle_hash_linked_list() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = grid.cell_count() * 8;
    let capacity = resolve_bucket_capacity(&grid, particle_count, WgpuNeighborMode::Auto).unwrap();

    assert_eq!(capacity, 0);
}

#[test]
fn auto_bucket_capacity_helper_keeps_periodic_2d_linked_list() {
    let grid = HashGridConfig::texture_2d();
    let capacity =
        resolve_bucket_capacity(&grid, grid.cell_count() * 64, WgpuNeighborMode::Auto).unwrap();

    assert_eq!(capacity, 0);
}

#[test]
fn adaptive_auto_keeps_sparse_particle_grid_linked_list() {
    let grid = HashGridConfig::growing_3dgs();
    let positions = (0..128)
        .map(|idx| {
            let x = (idx % 16) as f32 * grid.eps;
            let y = ((idx / 16) % 8) as f32 * grid.eps;
            [x, y, 0.0, 0.0]
        })
        .collect::<Vec<_>>();

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, positions.len(), &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::LinkedList);
}

#[test]
fn adaptive_auto_uses_cooperative_cells_for_2d_particle_grid_cells() {
    let grid = HashGridConfig::growing_2d();
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; 128];

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, positions.len(), &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn adaptive_auto_keeps_large_2d_tiled_storage_under_binding_limit() {
    let particle_count = 32_768;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        config.state_dims,
        config.spatial_dims,
        0,
        ParticleSeed::UniformCircle,
        0.2,
    );

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap();
    let storage_len =
        grid_storage_len_for_mode(grid.cell_count(), particle_count, capacity, mode).unwrap();

    assert!(
        grid_storage_binding_len_fits(storage_len),
        "mode={mode:?} capacity={capacity} storage_len={storage_len}"
    );
    if let WgpuNeighborMode::TiledFixedCellBuckets { capacity } = mode {
        assert!(
            capacity < 8192,
            "adaptive capacity should be reduced below the previously crashing 8192"
        );
    }
}

#[test]
fn adaptive_auto_uses_cooperative_sorted_cells_for_periodic_2d_grid() {
    let grid = HashGridConfig::texture_2d();
    let positions = (0..512)
        .map(|idx| {
            let x = ((idx % 32) as f32 / 31.0) * 2.0 - 1.0;
            let y = ((idx / 32) as f32 / 15.0) * 2.0 - 1.0;
            [x, y, 0.0, 0.0]
        })
        .collect::<Vec<_>>();

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, positions.len(), &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn auto_promotes_cooperative_cells_when_fixed_subgroups_are_supported() {
    assert_eq!(
        promote_auto_subgroup_mode(
            WgpuNeighborMode::Auto,
            WgpuNeighborMode::CooperativeSortedCells,
            true,
        ),
        WgpuNeighborMode::SubgroupCooperativeSortedCells
    );
    assert_eq!(
        promote_auto_subgroup_mode(
            WgpuNeighborMode::Auto,
            WgpuNeighborMode::CooperativeSortedCells,
            false,
        ),
        WgpuNeighborMode::CooperativeSortedCells
    );
    assert_eq!(
        promote_auto_subgroup_mode(
            WgpuNeighborMode::CooperativeSortedCells,
            WgpuNeighborMode::CooperativeSortedCells,
            true,
        ),
        WgpuNeighborMode::CooperativeSortedCells,
        "an explicit portable mode must not be rewritten"
    );
}

#[test]
fn adaptive_auto_uses_sorted_cells_for_small_collapsed_3d_particle_grid_cells() {
    let grid = HashGridConfig::growing_3dgs();
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; 128];

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, positions.len(), &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::SortedCells);
}

#[test]
fn adaptive_auto_uses_cooperative_sorted_cells_for_collapsed_3d_cells() {
    let grid = HashGridConfig::growing_3dgs();
    let particle_count = 8192;
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn explicit_subgroup_cooperative_sorted_cells_resolves() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 1024;
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

    let (capacity, mode) = resolve_neighbor_mode_for_state(
        &grid,
        particle_count,
        &positions,
        WgpuNeighborMode::SubgroupCooperativeSortedCells,
    )
    .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::SubgroupCooperativeSortedCells);
}

#[test]
fn adaptive_auto_uses_cooperative_sorted_cells_for_validated_3d_throughput_range() {
    let particle_count = 8192;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        config.state_dims,
        config.spatial_dims,
        13,
        ParticleSeed::UniformCircle,
        1.0,
    );
    let (nonempty_cells, max_occupancy) =
        initial_cell_occupancy_stats(&grid, particle_count, &positions).unwrap();

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert!(
        nonempty_cells > 4 && max_occupancy > 0,
        "expected a distributed 3D seed, got nonempty={nonempty_cells} max={max_occupancy}"
    );
    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn adaptive_auto_uses_cooperative_sorted_cells_for_collapsed_2d_cells() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 8192;
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn adaptive_auto_uses_cooperative_sorted_cells_for_validated_2d_throughput_range() {
    let particle_count = 8192;
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let (positions, _) = seed_particles_scaled(
        1,
        particle_count,
        config.state_dims,
        config.spatial_dims,
        7,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let (nonempty_cells, max_occupancy) =
        initial_cell_occupancy_stats(&grid, particle_count, &positions).unwrap();

    let (capacity, mode) =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap();

    assert!(
        nonempty_cells > 4 && max_occupancy > 0,
        "expected a distributed 2D seed, got nonempty={nonempty_cells} max={max_occupancy}"
    );
    assert_eq!(capacity, 0);
    assert_eq!(mode, WgpuNeighborMode::CooperativeSortedCells);
}

#[test]
fn adaptive_auto_rejects_unvalidated_oversized_cooperative_cells() {
    let grid = HashGridConfig::growing_2d();
    let particle_count = 16_384;
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

    let err =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap_err();

    assert!(
        err.to_string()
            .contains("cooperative sorted cells currently supports"),
        "unexpected error: {err}"
    );
}

#[test]
fn adaptive_auto_rejects_unvalidated_oversized_3d_cooperative_cells() {
    let grid = HashGridConfig::growing_3dgs();
    let particle_count = 16_384;
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; particle_count];

    let err =
        resolve_neighbor_mode_for_state(&grid, particle_count, &positions, WgpuNeighborMode::Auto)
            .unwrap_err();

    assert!(
        err.to_string()
            .contains("cooperative sorted cells currently supports"),
        "unexpected error: {err}"
    );
}

#[test]
fn fixed_buckets_reject_initial_occupancy_over_capacity() {
    let grid = HashGridConfig::growing_2d();
    let positions = vec![[0.0, 0.0, 0.0, 0.0]; 1024];

    let err = resolve_neighbor_mode_for_state(
        &grid,
        positions.len(),
        &positions,
        WgpuNeighborMode::TiledFixedCellBuckets { capacity: 512 },
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("is smaller than initial max cell occupancy"),
        "unexpected error: {err}"
    );
}
