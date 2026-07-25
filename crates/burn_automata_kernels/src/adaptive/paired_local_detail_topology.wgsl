// Device-resident fixed-budget topology for adaptive Target2D.
//
// This fragment is concatenated after gpu_step.wgsl and therefore uses its
// bindings, parameter helpers, material layout, and diagnostic feature layout.
// One workgroup copies the resident state and performs deterministic candidate
// reductions. Global diagnostic means retain the original lane-zero summation
// order so CPU/WGPU topology decisions remain directly comparable.

const PAIRED_TOPOLOGY_WORKGROUP_SIZE: u32 = 256u;
const PAIRED_TOPOLOGY_SPLIT_SCALE_PARAM: u32 = 97u;
const PAIRED_TOPOLOGY_MERGE_DETAIL_SCALE_PARAM: u32 = 98u;
const PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN_PARAM: u32 = 102u;
const CONTINUOUS_TOPOLOGY_EVENT_BUDGET_PARAM: u32 = 103u;
const CONTINUOUS_MAX_EXCHANGES: u32 = 64u;

var<workgroup> paired_reduce_values: array<f32, 256>;
var<workgroup> paired_reduce_rows: array<u32, 256>;
var<workgroup> paired_selected_rows: array<u32, 4>;
var<workgroup> paired_invalid: atomic<u32>;
var<workgroup> paired_fine_measure: f32;
var<workgroup> paired_mean_state_detail: f32;
var<workgroup> paired_mean_occupancy_detail: f32;
var<workgroup> paired_coarse_row: u32;
var<workgroup> paired_coarse_detail: f32;
var<workgroup> paired_anchor_row: u32;
var<workgroup> paired_anchor_x: f32;
var<workgroup> paired_anchor_y: f32;
var<workgroup> paired_fine_footprint_squared: f32;
var<workgroup> continuous_total_measure: f32;
var<workgroup> continuous_mean_measure: f32;
var<workgroup> continuous_accept: u32;
var<workgroup> continuous_position_correction: array<f32, 4>;
var<workgroup> continuous_state_correction: array<f32, 24>;
var<workgroup> continuous_old_mean_x: f32;
var<workgroup> continuous_old_mean_y: f32;
var<workgroup> continuous_old_cov_xx: f32;
var<workgroup> continuous_old_cov_xy: f32;
var<workgroup> continuous_old_cov_yy: f32;
var<workgroup> continuous_affine_00: f32;
var<workgroup> continuous_affine_01: f32;
var<workgroup> continuous_affine_10: f32;
var<workgroup> continuous_affine_11: f32;
var<workgroup> continuous_exchange_count: u32;
var<workgroup> continuous_coarse_rows: array<u32, 64>;
var<workgroup> continuous_fine_rows: array<u32, 64>;
var<workgroup> continuous_coarse_details: array<f32, 64>;
var<workgroup> continuous_fine_details: array<f32, 64>;

fn paired_reduce_min(local_id: u32) {
    var stride = PAIRED_TOPOLOGY_WORKGROUP_SIZE / 2u;
    loop {
        if (local_id < stride) {
            let other_value = paired_reduce_values[local_id + stride];
            let other_row = paired_reduce_rows[local_id + stride];
            let current_value = paired_reduce_values[local_id];
            let current_row = paired_reduce_rows[local_id];
            if (other_value < current_value
                || (other_value == current_value && other_row < current_row)) {
                paired_reduce_values[local_id] = other_value;
                paired_reduce_rows[local_id] = other_row;
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
}

fn paired_reduce_max(local_id: u32) {
    var stride = PAIRED_TOPOLOGY_WORKGROUP_SIZE / 2u;
    loop {
        if (local_id < stride) {
            let other_value = paired_reduce_values[local_id + stride];
            let other_row = paired_reduce_rows[local_id + stride];
            let current_value = paired_reduce_values[local_id];
            let current_row = paired_reduce_rows[local_id];
            if (other_value > current_value
                || (other_value == current_value && other_row < current_row)) {
                paired_reduce_values[local_id] = other_value;
                paired_reduce_rows[local_id] = other_row;
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
}

fn paired_topology_detail(
    row: u32,
    mean_state_detail: f32,
    mean_occupancy_detail: f32,
) -> f32 {
    let sd = state_dims();
    let feature_base = diagnostics_normalized_feature_offset() + row * feature_dims();
    let state_gradient_base = feature_base + 2u * sd;
    var state_squared = 0.0;
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        for (var axis = 0u; axis < 2u; axis = axis + 1u) {
            let value = density.values[state_gradient_base + channel * 2u + axis];
            state_squared = state_squared + value * value;
        }
    }
    let occupancy_base = feature_base + 4u * sd;
    let occupancy_x = density.values[occupancy_base];
    let occupancy_y = density.values[occupancy_base + 1u];
    return max(
        0.25 * sqrt(state_squared) / mean_state_detail
            + sqrt(occupancy_x * occupancy_x + occupancy_y * occupancy_y)
                / mean_occupancy_detail,
        1.0e-6,
    );
}

fn stable_local_detail_rank(detail: f32) -> f32 {
    return floor(detail * 256.0 + 0.5) / 256.0;
}

fn paired_topology_is_units(measure: f32, fine_measure: f32, units: f32) -> bool {
    return abs(measure / fine_measure - units) <= 2.0e-4 * units;
}

fn paired_reallocation_gain_is_sufficient(
    split_benefit: f32,
    merge_cost: f32,
    relative_margin: f32,
) -> bool {
    if (relative_margin >= 1.0) {
        return false;
    }
    let comparison_scale =
        max(max(abs(split_benefit), abs(merge_cost)), 1.175494351e-38);
    return split_benefit
        > merge_cost + relative_margin * comparison_scale;
}

fn current_material_display_scale(row: u32) -> f32 {
    let target_scale = max(
        material_render_target_footprint(row) * display_scale_per_footprint(),
        1.0e-20,
    );
    let transition_steps = render_transition_steps();
    var progress = 1.0;
    if (transition_steps > 0u) {
        let age = step_index() - min(step_index(), render_transition_start_step());
        progress = clamp(f32(age) / f32(transition_steps), 0.0, 1.0);
        progress = progress * progress * (3.0 - 2.0 * progress);
    }
    return exp2(mix(
        log2(max(material_render_from_scale(row), 1.0e-20)),
        log2(target_scale),
        progress,
    ));
}

// Canonical 2D coarse-to-fine activation for a state with resident reserve.
// The pass copies the active prefix into the alternate ping-pong buffers, then
// replaces each selected parent with four moment-preserving children. Reserve
// rows are populated before the host advances `total_count`.
const RESIDENT_BOOTSTRAP_MAX_SPLITS: u32 = 256u;
const RESIDENT_BOOTSTRAP_EVENT_COUNT_PARAM: u32 = 105u;
const RESIDENT_BOOTSTRAP_BANDWIDTH_EXPONENT_PARAM: u32 = 106u;
const RESIDENT_BOOTSTRAP_RENDER_EXPONENT_PARAM: u32 = 107u;
var<workgroup> resident_bootstrap_selected: array<u32, 256>;

@compute @workgroup_size(256)
fn resident_bootstrap_split_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let total = total_count();
    let capacity = resident_capacity();
    let sd = state_dims();
    let requested = min(
        pu(RESIDENT_BOOTSTRAP_EVENT_COUNT_PARAM),
        RESIDENT_BOOTSTRAP_MAX_SPLITS,
    );
    let event_count = min(requested, (capacity - min(capacity, total)) / 3u);

    var row = local_id.x;
    while (row < total) {
        let position_base = row * 4u;
        for (var axis = 0u; axis < 4u; axis = axis + 1u) {
            out_positions.values[position_base + axis] =
                positions.values[position_base + axis];
        }
        let state_base = row * sd;
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            out_states.values[state_base + channel] = states.values[state_base + channel];
        }
        row = row + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
    }
    storageBarrier();
    workgroupBarrier();
    if (local_id.x != 0u || spatial_dims() != 2u || event_count == 0u) {
        return;
    }

    let bandwidth_exponent =
        bitcast<f32>(pu(RESIDENT_BOOTSTRAP_BANDWIDTH_EXPONENT_PARAM));
    let render_exponent =
        bitcast<f32>(pu(RESIDENT_BOOTSTRAP_RENDER_EXPONENT_PARAM));
    let child_bandwidth_scale = pow(0.25, bandwidth_exponent);
    let child_render_scale = pow(0.25, 0.5 * render_exponent);
    let offset_scale = sqrt(1.5);

    var accepted = 0u;
    for (var event = 0u; event < event_count; event = event + 1u) {
        var parent = NIL;
        var parent_measure = -3.402823466e+38;
        for (var candidate = 0u; candidate < total; candidate = candidate + 1u) {
            var selected = false;
            for (var prior = 0u; prior < event; prior = prior + 1u) {
                selected = selected || resident_bootstrap_selected[prior] == candidate;
            }
            if (selected) {
                continue;
            }
            let measure = particle_measure(candidate);
            if (measure > parent_measure
                || (measure == parent_measure && candidate > parent)) {
                parent = candidate;
                parent_measure = measure;
            }
        }
        if (parent == NIL || parent_measure <= 0.0) {
            break;
        }
        resident_bootstrap_selected[event] = parent;

        var parent_material: array<f32, 127>;
        let parent_material_base = parent * MATERIAL_STRIDE;
        for (var component = 0u; component < MATERIAL_STRIDE; component = component + 1u) {
            parent_material[component] =
                material_data.values[parent_material_base + component];
        }
        let parent_position_base = parent * 4u;
        let parent_x = positions.values[parent_position_base];
        let parent_y = positions.values[parent_position_base + 1u];
        let parent_z = positions.values[parent_position_base + 2u];
        let parent_w = positions.values[parent_position_base + 3u];
        let cov_xx = max(parent_material[1u], 1.0e-20);
        let cov_yx = parent_material[4u];
        let cov_yy = max(parent_material[5u], 1.0e-20);
        let l00 = sqrt(cov_xx);
        let l10 = cov_yx / max(l00, 1.0e-20);
        let l11 = sqrt(max(cov_yy - l10 * l10, 1.0e-20));
        let render_from = current_material_display_scale(parent);
        let render_target = max(parent_material[11u] * child_render_scale, 1.0e-20);
        let child_measure = 0.25 * parent_measure;
        let child_bandwidth =
            max(parent_material[MATERIAL_BANDWIDTH_OFFSET] * child_bandwidth_scale, 1.0e-20);

        for (var child = 0u; child < 4u; child = child + 1u) {
            var destination = parent;
            if (child > 0u) {
                destination = total + event * 3u + child - 1u;
            }
            let destination_material_base = destination * MATERIAL_STRIDE;
            for (var component = 0u; component < MATERIAL_STRIDE; component = component + 1u) {
                material_data.values[destination_material_base + component] =
                    parent_material[component];
            }
            material_data.values[destination_material_base] = child_measure;
            for (var component = 0u; component < 9u; component = component + 1u) {
                material_data.values[destination_material_base + 1u + component] =
                    0.25 * parent_material[1u + component];
            }
            material_data.values[destination_material_base + 10u] = render_from;
            material_data.values[destination_material_base + 11u] = render_target;
            material_data.values[
                destination_material_base + MATERIAL_BANDWIDTH_OFFSET
            ] = child_bandwidth;
            for (var member = 0u; member < MATERIAL_UPDATE_MASK_MEMBERS; member = member + 1u) {
                material_data.values[
                    destination_material_base + MATERIAL_UPDATE_KEY_OFFSET + member
                ] = 0.0;
                material_data.values[
                    destination_material_base + MATERIAL_UPDATE_WEIGHT_OFFSET + member
                ] = 0.0;
            }
            material_data.values[
                destination_material_base + MATERIAL_UPDATE_KEY_OFFSET
            ] = bitcast<f32>(destination ^ 0x9e3779b9u);
            material_data.values[
                destination_material_base + MATERIAL_UPDATE_WEIGHT_OFFSET
            ] = 1.0;

            var offset_x = 0.0;
            var offset_y = 0.0;
            if (child == 0u) {
                offset_x = -offset_scale * l00;
                offset_y = -offset_scale * l10;
            } else if (child == 1u) {
                offset_x = offset_scale * l00;
                offset_y = offset_scale * l10;
            } else if (child == 2u) {
                offset_y = -offset_scale * l11;
            } else {
                offset_y = offset_scale * l11;
            }
            let destination_position_base = destination * 4u;
            out_positions.values[destination_position_base] = parent_x + offset_x;
            out_positions.values[destination_position_base + 1u] = parent_y + offset_y;
            out_positions.values[destination_position_base + 2u] = parent_z;
            out_positions.values[destination_position_base + 3u] = parent_w;
            let parent_state_base = parent * sd;
            let destination_state_base = destination * sd;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                out_states.values[destination_state_base + channel] =
                    states.values[parent_state_base + channel];
            }
        }
        accepted = accepted + 1u;
    }
    let accept_counter = resident_capacity() * MATERIAL_STRIDE;
    material_data.values[accept_counter] =
        material_data.values[accept_counter] + f32(accepted);
}

@compute @workgroup_size(256)
fn paired_local_detail_topology_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let total = total_count();
    let sd = state_dims();

    // The ordinary step bind group points at the current buffers and a
    // separate next-buffer pair. Populate every output row before lane zero
    // overwrites the five rows participating in the conservative exchange.
    var row = local_id.x;
    while (row < total) {
        let position_base = row * 4u;
        for (var axis = 0u; axis < 4u; axis = axis + 1u) {
            out_positions.values[position_base + axis] =
                positions.values[position_base + axis];
        }
        let state_base = row * sd;
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            out_states.values[state_base + channel] = states.values[state_base + channel];
        }
        row = row + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
    }
    storageBarrier();
    workgroupBarrier();
    if (total < 5u || spatial_dims() != 2u) {
        return;
    }

    if (local_id.x == 0u) {
        atomicStore(&paired_invalid, 0u);
        var fine_measure = 3.402823466e+38;
        var state_detail_sum = 0.0;
        var occupancy_detail_sum = 0.0;
        for (var index = 0u; index < total; index = index + 1u) {
            fine_measure = min(fine_measure, particle_measure(index));
            let feature_base =
                diagnostics_normalized_feature_offset() + index * feature_dims();
            let state_gradient_base = feature_base + 2u * sd;
            var state_squared = 0.0;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                for (var axis = 0u; axis < 2u; axis = axis + 1u) {
                    let value = density.values[
                        state_gradient_base + channel * 2u + axis
                    ];
                    state_squared = state_squared + value * value;
                }
            }
            let occupancy_base = feature_base + 4u * sd;
            let occupancy_x = density.values[occupancy_base];
            let occupancy_y = density.values[occupancy_base + 1u];
            state_detail_sum = state_detail_sum + sqrt(state_squared);
            occupancy_detail_sum = occupancy_detail_sum
                + sqrt(occupancy_x * occupancy_x + occupancy_y * occupancy_y);
        }
        paired_fine_measure = fine_measure;
        paired_mean_state_detail = max(state_detail_sum / f32(total), 1.0e-6);
        paired_mean_occupancy_detail =
            max(occupancy_detail_sum / f32(total), 1.0e-6);
    }
    workgroupBarrier();

    var local_coarse_row = NIL;
    var local_coarse_detail = -3.402823466e+38;
    var index = local_id.x;
    while (index < total) {
        let measure = particle_measure(index);
        let detail =
            paired_topology_detail(
                index,
                paired_mean_state_detail,
                paired_mean_occupancy_detail,
            );
        if (paired_topology_is_units(measure, paired_fine_measure, 4.0)) {
            if (detail > local_coarse_detail
                || (detail == local_coarse_detail && index < local_coarse_row)) {
                local_coarse_detail = detail;
                local_coarse_row = index;
            }
        } else if (!paired_topology_is_units(measure, paired_fine_measure, 1.0)) {
            atomicStore(&paired_invalid, 1u);
        }
        index = index + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
    }
    paired_reduce_values[local_id.x] = local_coarse_detail;
    paired_reduce_rows[local_id.x] = local_coarse_row;
    workgroupBarrier();
    paired_reduce_max(local_id.x);
    if (local_id.x == 0u) {
        paired_coarse_row = paired_reduce_rows[0];
        paired_coarse_detail = paired_reduce_values[0];
    }
    workgroupBarrier();

    var local_anchor_row = NIL;
    var local_anchor_detail = 3.402823466e+38;
    index = local_id.x;
    while (index < total) {
        if (paired_topology_is_units(
            particle_measure(index),
            paired_fine_measure,
            1.0,
        )) {
            let detail = paired_topology_detail(
                index,
                paired_mean_state_detail,
                paired_mean_occupancy_detail,
            );
            if (detail < local_anchor_detail
                || (detail == local_anchor_detail && index < local_anchor_row)) {
                local_anchor_detail = detail;
                local_anchor_row = index;
            }
        }
        index = index + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
    }
    paired_reduce_values[local_id.x] = local_anchor_detail;
    paired_reduce_rows[local_id.x] = local_anchor_row;
    workgroupBarrier();
    paired_reduce_min(local_id.x);
    if (local_id.x == 0u) {
        paired_anchor_row = paired_reduce_rows[0];
        if (paired_coarse_row == NIL || paired_anchor_row == NIL) {
            atomicStore(&paired_invalid, 1u);
        } else {
            let anchor_base = paired_anchor_row * 4u;
            paired_anchor_x = positions.values[anchor_base];
            paired_anchor_y = positions.values[anchor_base + 1u];
            paired_fine_footprint_squared =
                max(paired_fine_measure / 3.141592653589793, 1.0e-20);
        }
    }
    workgroupBarrier();
    if (atomicLoad(&paired_invalid) != 0u) {
        return;
    }

    let merge_detail_scale = bitcast<f32>(pu(PAIRED_TOPOLOGY_MERGE_DETAIL_SCALE_PARAM));
    for (var slot = 0u; slot < 4u; slot = slot + 1u) {
        var local_best_row = NIL;
        var local_best_score = 3.402823466e+38;
        index = local_id.x;
        while (index < total) {
            if (!paired_topology_is_units(
                particle_measure(index),
                paired_fine_measure,
                1.0,
            )) {
                index = index + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
                continue;
            }
            var already_selected = false;
            for (var previous = 0u; previous < slot; previous = previous + 1u) {
                already_selected =
                    already_selected || paired_selected_rows[previous] == index;
            }
            if (already_selected) {
                index = index + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
                continue;
            }
            let position_base = index * 4u;
            let dx = positions.values[position_base] - paired_anchor_x;
            let dy = positions.values[position_base + 1u] - paired_anchor_y;
            let score = (dx * dx + dy * dy) / paired_fine_footprint_squared
                + merge_detail_scale
                    * paired_topology_detail(
                        index,
                        paired_mean_state_detail,
                        paired_mean_occupancy_detail,
                    );
            if (score < local_best_score
                || (score == local_best_score && index < local_best_row)) {
                local_best_score = score;
                local_best_row = index;
            }
            index = index + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
        }
        paired_reduce_values[local_id.x] = local_best_score;
        paired_reduce_rows[local_id.x] = local_best_row;
        workgroupBarrier();
        paired_reduce_min(local_id.x);
        if (local_id.x == 0u) {
            paired_selected_rows[slot] = paired_reduce_rows[0];
            if (paired_reduce_rows[0] == NIL) {
                atomicStore(&paired_invalid, 1u);
            }
        }
        workgroupBarrier();
        if (atomicLoad(&paired_invalid) != 0u) {
            return;
        }
    }

    if (local_id.x != 0u) {
        return;
    }
    let coarse_row = paired_coarse_row;
    var selected_rows: array<u32, 4>;
    var merge_detail = 0.0;
    for (var slot = 0u; slot < 4u; slot = slot + 1u) {
        selected_rows[slot] = paired_selected_rows[slot];
        merge_detail = merge_detail
            + paired_topology_detail(
                selected_rows[slot],
                paired_mean_state_detail,
                paired_mean_occupancy_detail,
            );
    }
    merge_detail = 0.25 * merge_detail;
    let min_relative_gain =
        bitcast<f32>(pu(PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN_PARAM));
    if (!paired_reallocation_gain_is_sufficient(
        paired_coarse_detail,
        merge_detail,
        min_relative_gain,
    )) {
        return;
    }
    let accept_counter = resident_capacity() * MATERIAL_STRIDE;
    material_data.values[accept_counter] =
        material_data.values[accept_counter] + 1.0;
    let coarse_position_base = coarse_row * 4u;
    let coarse_x = positions.values[coarse_position_base];
    let coarse_y = positions.values[coarse_position_base + 1u];
    let coarse_z = positions.values[coarse_position_base + 2u];
    let coarse_w = positions.values[coarse_position_base + 3u];
    var merged_x = 0.0;
    var merged_y = 0.0;
    var merged_z = 0.0;
    var merged_w = 0.0;
    for (var slot = 0u; slot < 4u; slot = slot + 1u) {
        let base = selected_rows[slot] * 4u;
        merged_x = merged_x + positions.values[base];
        merged_y = merged_y + positions.values[base + 1u];
        merged_z = merged_z + positions.values[base + 2u];
        merged_w = merged_w + positions.values[base + 3u];
    }
    out_positions.values[coarse_position_base] = 0.25 * merged_x;
    out_positions.values[coarse_position_base + 1u] = 0.25 * merged_y;
    out_positions.values[coarse_position_base + 2u] = 0.25 * merged_z;
    out_positions.values[coarse_position_base + 3u] = 0.25 * merged_w;

    let split_scale = bitcast<f32>(pu(PAIRED_TOPOLOGY_SPLIT_SCALE_PARAM));
    let split_radius =
        sqrt(1.5) * sqrt(paired_fine_footprint_squared) * split_scale;
    for (var slot = 0u; slot < 4u; slot = slot + 1u) {
        let row = selected_rows[slot];
        let base = row * 4u;
        var offset_x = 0.0;
        var offset_y = 0.0;
        if (slot == 0u) {
            offset_x = -split_radius;
        } else if (slot == 1u) {
            offset_x = split_radius;
        } else if (slot == 2u) {
            offset_y = -split_radius;
        } else {
            offset_y = split_radius;
        }
        out_positions.values[base] = coarse_x + offset_x;
        out_positions.values[base + 1u] = coarse_y + offset_y;
        out_positions.values[base + 2u] = coarse_z;
        out_positions.values[base + 3u] = coarse_w;
    }

    let coarse_state_base = coarse_row * sd;
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        var merged_state = 0.0;
        for (var slot = 0u; slot < 4u; slot = slot + 1u) {
            merged_state = merged_state
                + states.values[selected_rows[slot] * sd + channel];
        }
        out_states.values[coarse_state_base + channel] = 0.25 * merged_state;
        let coarse_state = states.values[coarse_state_base + channel];
        for (var slot = 0u; slot < 4u; slot = slot + 1u) {
            out_states.values[selected_rows[slot] * sd + channel] = coarse_state;
        }
    }
}

// Relocate fixed graded material slots without changing the row budget. The
// high-detail coarse slot and low-detail fine slot exchange their dynamic
// fields. A global near-identity affine projection restores the weighted
// position centroid and second moment, while one common intensive correction
// restores every extensive recurrent-state channel.
@compute @workgroup_size(256)
fn continuous_local_detail_topology_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let total = total_count();
    let sd = state_dims();
    if (total < 2u || spatial_dims() != 2u) {
        var copy_row = local_id.x;
        while (copy_row < total) {
            let position_base = copy_row * 4u;
            for (var axis = 0u; axis < 4u; axis = axis + 1u) {
                out_positions.values[position_base + axis] =
                    positions.values[position_base + axis];
            }
            let state_base = copy_row * sd;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                out_states.values[state_base + channel] =
                    states.values[state_base + channel];
            }
            copy_row = copy_row + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
        }
        return;
    }

    if (local_id.x == 0u) {
        var total_measure = 0.0;
        var state_detail_sum = 0.0;
        var occupancy_detail_sum = 0.0;
        var position_first_x = 0.0;
        var position_first_y = 0.0;
        var position_second_xx = 0.0;
        var position_second_xy = 0.0;
        var position_second_yy = 0.0;
        for (var index = 0u; index < total; index = index + 1u) {
            let measure = particle_measure(index);
            let position_base = index * 4u;
            let x = positions.values[position_base];
            let y = positions.values[position_base + 1u];
            total_measure = total_measure + measure;
            position_first_x = position_first_x + measure * x;
            position_first_y = position_first_y + measure * y;
            position_second_xx = position_second_xx + measure * x * x;
            position_second_xy = position_second_xy + measure * x * y;
            position_second_yy = position_second_yy + measure * y * y;
            let feature_base =
                diagnostics_normalized_feature_offset() + index * feature_dims();
            let state_gradient_base = feature_base + 2u * sd;
            var state_squared = 0.0;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                for (var axis = 0u; axis < 2u; axis = axis + 1u) {
                    let value = density.values[
                        state_gradient_base + channel * 2u + axis
                    ];
                    state_squared = state_squared + value * value;
                }
            }
            let occupancy_base = feature_base + 4u * sd;
            let occupancy_x = density.values[occupancy_base];
            let occupancy_y = density.values[occupancy_base + 1u];
            state_detail_sum = state_detail_sum + sqrt(state_squared);
            occupancy_detail_sum = occupancy_detail_sum
                + sqrt(occupancy_x * occupancy_x + occupancy_y * occupancy_y);
        }
        continuous_total_measure = total_measure;
        continuous_mean_measure = total_measure / f32(total);
        continuous_old_mean_x = position_first_x / total_measure;
        continuous_old_mean_y = position_first_y / total_measure;
        continuous_old_cov_xx = position_second_xx / total_measure
            - continuous_old_mean_x * continuous_old_mean_x;
        continuous_old_cov_xy = position_second_xy / total_measure
            - continuous_old_mean_x * continuous_old_mean_y;
        continuous_old_cov_yy = position_second_yy / total_measure
            - continuous_old_mean_y * continuous_old_mean_y;
        continuous_affine_00 = 1.0;
        continuous_affine_01 = 0.0;
        continuous_affine_10 = 0.0;
        continuous_affine_11 = 1.0;
        paired_mean_state_detail = max(state_detail_sum / f32(total), 1.0e-6);
        paired_mean_occupancy_detail =
            max(occupancy_detail_sum / f32(total), 1.0e-6);
        continuous_accept = 0u;
    }
    workgroupBarrier();

    if (local_id.x == 0u) {
        let requested = clamp(
            pu(CONTINUOUS_TOPOLOGY_EVENT_BUDGET_PARAM),
            1u,
            CONTINUOUS_MAX_EXCHANGES,
        );
        let measure_tolerance = 2.0e-4 * continuous_mean_measure;
        for (var slot = 0u; slot < CONTINUOUS_MAX_EXCHANGES; slot = slot + 1u) {
            continuous_coarse_rows[slot] = NIL;
            continuous_fine_rows[slot] = NIL;
            continuous_coarse_details[slot] = -3.402823466e+38;
            continuous_fine_details[slot] = 3.402823466e+38;
        }

        for (var index = 0u; index < total; index = index + 1u) {
            let measure = particle_measure(index);
            if (measure > continuous_mean_measure + measure_tolerance
                || measure + measure_tolerance < continuous_mean_measure) {
                let detail = stable_local_detail_rank(paired_topology_detail(
                    index,
                    paired_mean_state_detail,
                    paired_mean_occupancy_detail,
                ));
                if (measure > continuous_mean_measure + measure_tolerance) {
                    var insert = requested;
                    for (var slot = 0u; slot < requested; slot = slot + 1u) {
                        if (detail > continuous_coarse_details[slot]
                            || (detail == continuous_coarse_details[slot]
                                && index < continuous_coarse_rows[slot])) {
                            insert = slot;
                            break;
                        }
                    }
                    if (insert < requested) {
                        var shift = requested;
                        while (shift > insert + 1u) {
                            shift = shift - 1u;
                            continuous_coarse_details[shift] =
                                continuous_coarse_details[shift - 1u];
                            continuous_coarse_rows[shift] =
                                continuous_coarse_rows[shift - 1u];
                        }
                        continuous_coarse_details[insert] = detail;
                        continuous_coarse_rows[insert] = index;
                    }
                } else {
                    var insert = requested;
                    for (var slot = 0u; slot < requested; slot = slot + 1u) {
                        if (detail < continuous_fine_details[slot]
                            || (detail == continuous_fine_details[slot]
                                && index < continuous_fine_rows[slot])) {
                            insert = slot;
                            break;
                        }
                    }
                    if (insert < requested) {
                        var shift = requested;
                        while (shift > insert + 1u) {
                            shift = shift - 1u;
                            continuous_fine_details[shift] =
                                continuous_fine_details[shift - 1u];
                            continuous_fine_rows[shift] =
                                continuous_fine_rows[shift - 1u];
                        }
                        continuous_fine_details[insert] = detail;
                        continuous_fine_rows[insert] = index;
                    }
                }
            }
        }

        let min_relative_gain =
            bitcast<f32>(pu(PAIRED_TOPOLOGY_MIN_RELATIVE_GAIN_PARAM));
        var accepted = 0u;
        for (var slot = 0u; slot < requested; slot = slot + 1u) {
            if (continuous_coarse_rows[slot] != NIL
                && continuous_fine_rows[slot] != NIL
                && paired_reallocation_gain_is_sufficient(
                    continuous_coarse_details[slot],
                    continuous_fine_details[slot],
                    min_relative_gain,
                )) {
                continuous_coarse_rows[accepted] = continuous_coarse_rows[slot];
                continuous_fine_rows[accepted] = continuous_fine_rows[slot];
                continuous_coarse_details[accepted] =
                    continuous_coarse_details[slot];
                continuous_fine_details[accepted] = continuous_fine_details[slot];
                accepted = accepted + 1u;
            }
        }
        continuous_exchange_count = accepted;
        continuous_accept = select(0u, 1u, accepted > 0u);

        let reciprocal_total = 1.0 / continuous_total_measure;
        var swapped_first_x = continuous_old_mean_x * continuous_total_measure;
        var swapped_first_y = continuous_old_mean_y * continuous_total_measure;
        var swapped_second_xx = continuous_total_measure
            * (continuous_old_cov_xx
                + continuous_old_mean_x * continuous_old_mean_x);
        var swapped_second_xy = continuous_total_measure
            * (continuous_old_cov_xy
                + continuous_old_mean_x * continuous_old_mean_y);
        var swapped_second_yy = continuous_total_measure
            * (continuous_old_cov_yy
                + continuous_old_mean_y * continuous_old_mean_y);
        continuous_position_correction[2] = 0.0;
        continuous_position_correction[3] = 0.0;
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            continuous_state_correction[channel] = 0.0;
        }
        for (var slot = 0u; slot < accepted; slot = slot + 1u) {
            let coarse_row = continuous_coarse_rows[slot];
            let fine_row = continuous_fine_rows[slot];
            let measure_delta =
                particle_measure(coarse_row) - particle_measure(fine_row);
            let coarse_position_base = coarse_row * 4u;
            let fine_position_base = fine_row * 4u;
            let coarse_x = positions.values[coarse_position_base];
            let coarse_y = positions.values[coarse_position_base + 1u];
            let fine_x = positions.values[fine_position_base];
            let fine_y = positions.values[fine_position_base + 1u];
            swapped_first_x =
                swapped_first_x + measure_delta * (fine_x - coarse_x);
            swapped_first_y =
                swapped_first_y + measure_delta * (fine_y - coarse_y);
            swapped_second_xx = swapped_second_xx
                + measure_delta * (fine_x * fine_x - coarse_x * coarse_x);
            swapped_second_xy = swapped_second_xy
                + measure_delta * (fine_x * fine_y - coarse_x * coarse_y);
            swapped_second_yy = swapped_second_yy
                + measure_delta * (fine_y * fine_y - coarse_y * coarse_y);
            for (var axis = 2u; axis < 4u; axis = axis + 1u) {
                continuous_position_correction[axis] =
                    continuous_position_correction[axis]
                    + measure_delta
                        * (positions.values[coarse_position_base + axis]
                            - positions.values[fine_position_base + axis])
                        * reciprocal_total;
            }
            let coarse_state_base = coarse_row * sd;
            let fine_state_base = fine_row * sd;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                continuous_state_correction[channel] =
                    continuous_state_correction[channel]
                    + measure_delta
                        * (states.values[coarse_state_base + channel]
                            - states.values[fine_state_base + channel])
                        * reciprocal_total;
            }
        }

        if (accepted > 0u) {
            let swapped_mean_x = swapped_first_x * reciprocal_total;
            let swapped_mean_y = swapped_first_y * reciprocal_total;
            let swapped_cov_xx = swapped_second_xx * reciprocal_total
                - swapped_mean_x * swapped_mean_x;
            let swapped_cov_xy = swapped_second_xy * reciprocal_total
                - swapped_mean_x * swapped_mean_y;
            let swapped_cov_yy = swapped_second_yy * reciprocal_total
                - swapped_mean_y * swapped_mean_y;
            let old_scale = max(
                abs(continuous_old_cov_xx) + abs(continuous_old_cov_yy),
                1.0e-20,
            );
            let swapped_scale = max(
                abs(swapped_cov_xx) + abs(swapped_cov_yy),
                1.0e-20,
            );
            let old_l00 = sqrt(max(continuous_old_cov_xx, 0.0));
            let swapped_l00 = sqrt(max(swapped_cov_xx, 0.0));
            var old_l10 = 0.0;
            var swapped_l10 = 0.0;
            if (old_l00 > 0.0) {
                old_l10 = continuous_old_cov_xy / old_l00;
            }
            if (swapped_l00 > 0.0) {
                swapped_l10 = swapped_cov_xy / swapped_l00;
            }
            let old_l11_squared =
                continuous_old_cov_yy - old_l10 * old_l10;
            let swapped_l11_squared =
                swapped_cov_yy - swapped_l10 * swapped_l10;
            let valid_moments =
                old_l00 * old_l00 > 1.0e-8 * old_scale
                && swapped_l00 * swapped_l00 > 1.0e-8 * swapped_scale
                && old_l11_squared > 1.0e-8 * old_scale
                && swapped_l11_squared > 1.0e-8 * swapped_scale;
            if (valid_moments) {
                let old_l11 = sqrt(old_l11_squared);
                let swapped_l11 = sqrt(swapped_l11_squared);
                continuous_affine_00 = old_l00 / swapped_l00;
                continuous_affine_01 = 0.0;
                continuous_affine_10 = old_l10 / swapped_l00
                    - old_l11 * swapped_l10
                        / (swapped_l00 * swapped_l11);
                continuous_affine_11 = old_l11 / swapped_l11;
                continuous_position_correction[0] =
                    continuous_old_mean_x - swapped_mean_x;
                continuous_position_correction[1] =
                    continuous_old_mean_y - swapped_mean_y;
                let accept_counter = resident_capacity() * MATERIAL_STRIDE;
                material_data.values[accept_counter] =
                    material_data.values[accept_counter] + f32(accepted);
            } else {
                continuous_exchange_count = 0u;
                continuous_accept = 0u;
            }
        }
    }
    workgroupBarrier();

    var row = local_id.x;
    while (row < total) {
        var source_row = row;
        if (continuous_accept != 0u) {
            for (
                var slot = 0u;
                slot < continuous_exchange_count;
                slot = slot + 1u
            ) {
                if (row == continuous_coarse_rows[slot]) {
                    source_row = continuous_fine_rows[slot];
                } else if (row == continuous_fine_rows[slot]) {
                    source_row = continuous_coarse_rows[slot];
                }
            }
        }
        let position_base = row * 4u;
        let source_position_base = source_row * 4u;
        let translated_x = positions.values[source_position_base]
            + select(0.0, continuous_position_correction[0], continuous_accept != 0u);
        let translated_y = positions.values[source_position_base + 1u]
            + select(0.0, continuous_position_correction[1], continuous_accept != 0u);
        let centered_x = translated_x - continuous_old_mean_x;
        let centered_y = translated_y - continuous_old_mean_y;
        out_positions.values[position_base] = select(
            positions.values[source_position_base],
            continuous_old_mean_x
                + continuous_affine_00 * centered_x
                + continuous_affine_01 * centered_y,
            continuous_accept != 0u,
        );
        out_positions.values[position_base + 1u] = select(
            positions.values[source_position_base + 1u],
            continuous_old_mean_y
                + continuous_affine_10 * centered_x
                + continuous_affine_11 * centered_y,
            continuous_accept != 0u,
        );
        for (var axis = 2u; axis < 4u; axis = axis + 1u) {
            out_positions.values[position_base + axis] =
                positions.values[source_position_base + axis]
                + select(
                    0.0,
                    continuous_position_correction[axis],
                    continuous_accept != 0u,
                );
        }
        let state_base = row * sd;
        let source_state_base = source_row * sd;
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            var correction = 0.0;
            if (continuous_accept != 0u) {
                correction = continuous_state_correction[channel];
            }
            out_states.values[state_base + channel] =
                states.values[source_state_base + channel] + correction;
        }
        let render_from = current_material_display_scale(source_row);
        material_data.values[row * MATERIAL_STRIDE + 10u] = render_from;
        row = row + PAIRED_TOPOLOGY_WORKGROUP_SIZE;
    }
}
