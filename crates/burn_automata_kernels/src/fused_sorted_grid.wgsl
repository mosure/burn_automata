const FUSED_SORTED_GRID_MAX_CELLS: u32 = 256u;

var<workgroup> fused_cell_counts: array<atomic<u32>, FUSED_SORTED_GRID_MAX_CELLS>;

@compute @workgroup_size(256)
fn fused_sorted_grid_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let lane = workgroup.x;
    let local_id = local.x;
    let cells = grid_cells_per_lane();
    if (lane >= batch_size() || cells > FUSED_SORTED_GRID_MAX_CELLS) {
        return;
    }

    if (local_id < cells) {
        atomicStore(&fused_cell_counts[local_id], 0u);
    }
    workgroupBarrier();

    let particle_base = lane * particle_count();
    let cell_base = lane * cells;
    for (var particle = local_id; particle < particle_count(); particle += 256u) {
        let index = particle_base + particle;
        let cell = u32(source_cell_index(cell_coords(position(index)), index)) - cell_base;
        atomicAdd(&fused_cell_counts[cell], 1u);
    }
    workgroupBarrier();

    var count = 0u;
    if (local_id < cells) {
        count = atomicLoad(&fused_cell_counts[local_id]);
    }
    scan_values[local_id] = count;
    workgroupBarrier();

    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            scan_values[index] += scan_values[index - stride];
        }
        workgroupBarrier();
    }
    if (local_id == 0u) {
        scan_values[255u] = 0u;
    }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            let left = scan_values[index - stride];
            scan_values[index - stride] = scan_values[index];
            scan_values[index] += left;
        }
        workgroupBarrier();
    }

    if (local_id < cells) {
        atomicStore(
            &linked_grid.values[sorted_offsets_base() + cell_base + local_id],
            particle_base + scan_values[local_id],
        );
        atomicStore(&fused_cell_counts[local_id], 0u);
    }
    if (local_id == 0u && lane + 1u == batch_size()) {
        atomicStore(
            &linked_grid.values[sorted_offsets_base() + cell_count()],
            total_count(),
        );
    }
    workgroupBarrier();

    for (var particle = local_id; particle < particle_count(); particle += 256u) {
        let index = particle_base + particle;
        let cell = u32(source_cell_index(cell_coords(position(index)), index)) - cell_base;
        let slot = atomicAdd(&fused_cell_counts[cell], 1u);
        atomicStore(
            &linked_grid.values[
                sorted_indices_base() + particle_base + scan_values[cell] + slot
            ],
            index,
        );
    }
}
