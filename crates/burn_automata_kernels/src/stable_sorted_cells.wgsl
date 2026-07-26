fn stable_sorted_index(slot: u32) -> u32 {
    return atomicLoad(&linked_grid.values[sorted_indices_base() + slot]);
}

fn stable_store_sorted_index(slot: u32, value: u32) {
    atomicStore(&linked_grid.values[sorted_indices_base() + slot], value);
}

const STABLE_SORT_WORKGROUP_SIZE: u32 = 128u;
const STABLE_SORT_CAPACITY: u32 = 512u;
const STABLE_SORT_SENTINEL: u32 = 0xffffffffu;

var<workgroup> stable_sort_values: array<u32, STABLE_SORT_CAPACITY>;
var<workgroup> stable_sort_range: vec2<u32>;

fn stable_serial_insertion_sort(begin: u32, count: u32) {
    var source = 1u;
    while (source < count) {
        let value = stable_sorted_index(begin + source);
        var destination = source;
        while (destination > 0u) {
            let previous = stable_sorted_index(begin + destination - 1u);
            if (previous <= value) {
                break;
            }
            stable_store_sorted_index(begin + destination, previous);
            destination -= 1u;
        }
        stable_store_sorted_index(begin + destination, value);
        source += 1u;
    }
}

fn stable_next_power_of_two(value: u32) -> u32 {
    var result = 1u;
    while (result < value) {
        result *= 2u;
    }
    return result;
}

// One workgroup owns one cell. Normal NPA occupancies are sorted in shared
// memory with a deterministic bitonic network; the bounded serial fallback
// preserves correctness for unexpectedly concentrated cells. Regular NPA
// inference retains the throughput-optimized atomic scatter order.
@compute @workgroup_size(128)
fn stable_sort_cell_particles_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let cell = workgroup.x;
    if (cell >= cell_count()) {
        return;
    }
    if (local.x == 0u) {
        stable_sort_range = vec2<u32>(
            sorted_offset(cell),
            sorted_offset(cell + 1u),
        );
    }
    let range = workgroupUniformLoad(&stable_sort_range);
    let begin = range.x;
    let end = range.y;
    let count = end - begin;
    if (count <= 1u) {
        return;
    }

    if (count > STABLE_SORT_CAPACITY) {
        if (local.x == 0u) {
            stable_serial_insertion_sort(begin, count);
        }
        return;
    }

    let padded_count = stable_next_power_of_two(count);
    for (
        var index = local.x;
        index < padded_count;
        index += STABLE_SORT_WORKGROUP_SIZE
    ) {
        stable_sort_values[index] = STABLE_SORT_SENTINEL;
        if (index < count) {
            stable_sort_values[index] = stable_sorted_index(begin + index);
        }
    }
    workgroupBarrier();

    var sequence = 2u;
    while (sequence <= padded_count) {
        var stride = sequence / 2u;
        while (stride > 0u) {
            for (
                var index = local.x;
                index < padded_count;
                index += STABLE_SORT_WORKGROUP_SIZE
            ) {
                let partner = index ^ stride;
                if (partner > index) {
                    let lhs = stable_sort_values[index];
                    let rhs = stable_sort_values[partner];
                    let ascending = (index & sequence) == 0u;
                    if ((lhs > rhs) == ascending) {
                        stable_sort_values[index] = rhs;
                        stable_sort_values[partner] = lhs;
                    }
                }
            }
            workgroupBarrier();
            stride /= 2u;
        }
        sequence *= 2u;
    }

    for (
        var index = local.x;
        index < count;
        index += STABLE_SORT_WORKGROUP_SIZE
    ) {
        stable_store_sorted_index(begin + index, stable_sort_values[index]);
    }
}
