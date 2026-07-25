const MAX_STATE_DIMS: u32 = 24u;
const MATERIAL_UPDATE_MASK_MEMBERS: u32 = 6u;
const MATERIAL_BANDWIDTH_OFFSET: u32 = 12u;
const MATERIAL_STATE_JACOBIAN_CAPACITY: u32 = MAX_STATE_DIMS * 3u;
const MATERIAL_STRIDE: u32 =
    13u + 2u * MATERIAL_UPDATE_MASK_MEMBERS + MATERIAL_STATE_JACOBIAN_CAPACITY;

struct Params {
    internal_count: u32,
    active_count: u32,
    state_dims: u32,
    spatial_dims: u32,
};

struct F32Buffer {
    values: array<f32>,
};

struct U32Buffer {
    values: array<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> internal_positions: F32Buffer;
@group(0) @binding(2) var<storage, read> internal_states: F32Buffer;
@group(0) @binding(3) var<storage, read_write> active_positions: F32Buffer;
@group(0) @binding(4) var<storage, read_write> active_states: F32Buffer;
@group(0) @binding(5) var<storage, read> mode_offsets: U32Buffer;
@group(0) @binding(6) var<storage, read> mode_data: U32Buffer;
@group(0) @binding(7) var<storage, read> internal_material: F32Buffer;
@group(0) @binding(8) var<storage, read_write> active_material: F32Buffer;

@compute @workgroup_size(128)
fn persistent_restrict_main(@builtin(global_invocation_id) global: vec3<u32>) {
    let active_row = global.x;
    if (active_row >= params.active_count) {
        return;
    }
    let start = mode_offsets.values[active_row];
    let end = mode_offsets.values[active_row + 1u];
    if (start >= end) {
        return;
    }

    var position = vec4<f32>(0.0);
    var state: array<f32, MAX_STATE_DIMS>;
    var bandwidth = 0.0;
    for (var channel = 0u; channel < MAX_STATE_DIMS; channel = channel + 1u) {
        state[channel] = 0.0;
    }
    for (var cursor = start; cursor < end; cursor = cursor + 1u) {
        let row = mode_data.values[2u * cursor];
        if (row >= params.internal_count) {
            continue;
        }
        let weight = bitcast<f32>(mode_data.values[2u * cursor + 1u]);
        let position_base = row * 4u;
        position = position + weight * vec4<f32>(
            internal_positions.values[position_base],
            internal_positions.values[position_base + 1u],
            internal_positions.values[position_base + 2u],
            internal_positions.values[position_base + 3u],
        );
        let state_base = row * params.state_dims;
        for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
            state[channel] = state[channel]
                + weight * internal_states.values[state_base + channel];
        }
        bandwidth = bandwidth
            + weight * internal_material.values[
                row * MATERIAL_STRIDE + MATERIAL_BANDWIDTH_OFFSET
            ];
    }

    var covariance: array<f32, 9>;
    for (var element = 0u; element < 9u; element = element + 1u) {
        covariance[element] = 0.0;
    }
    for (var cursor = start; cursor < end; cursor = cursor + 1u) {
        let row = mode_data.values[2u * cursor];
        if (row >= params.internal_count) {
            continue;
        }
        let weight = bitcast<f32>(mode_data.values[2u * cursor + 1u]);
        let position_base = row * 4u;
        let mode_position = vec3<f32>(
            internal_positions.values[position_base],
            internal_positions.values[position_base + 1u],
            internal_positions.values[position_base + 2u],
        );
        let delta = mode_position - position.xyz;
        for (var row_axis = 0u; row_axis < params.spatial_dims; row_axis = row_axis + 1u) {
            for (var col_axis = 0u; col_axis < params.spatial_dims; col_axis = col_axis + 1u) {
                let element = row_axis * 3u + col_axis;
                covariance[element] = covariance[element] + weight * (
                    internal_material.values[row * MATERIAL_STRIDE + 1u + element]
                    + delta[row_axis] * delta[col_axis]
                );
            }
        }
    }

    let active_position_base = active_row * 4u;
    active_positions.values[active_position_base] = position.x;
    active_positions.values[active_position_base + 1u] = position.y;
    active_positions.values[active_position_base + 2u] = position.z;
    active_positions.values[active_position_base + 3u] = position.w;
    let active_state_base = active_row * params.state_dims;
    for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
        active_states.values[active_state_base + channel] = state[channel];
    }
    let active_material_base = active_row * MATERIAL_STRIDE;
    for (var element = 0u; element < 9u; element = element + 1u) {
        active_material.values[active_material_base + 1u + element] = covariance[element];
    }
    active_material.values[active_material_base + MATERIAL_BANDWIDTH_OFFSET] = bandwidth;
}
