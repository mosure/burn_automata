const MAX_STATE_DIMS: u32 = 24u;
const MAX_DIMS: u32 = 3u;
const MATERIAL_UPDATE_MASK_MEMBERS: u32 = 6u;
const MATERIAL_STATE_JACOBIAN_OFFSET: u32 =
    13u + 2u * MATERIAL_UPDATE_MASK_MEMBERS;
const MATERIAL_STATE_JACOBIAN_CAPACITY: u32 = MAX_STATE_DIMS * MAX_DIMS;
const MATERIAL_STRIDE: u32 =
    MATERIAL_STATE_JACOBIAN_OFFSET + MATERIAL_STATE_JACOBIAN_CAPACITY;

struct Params {
    mode_count: u32,
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
@group(0) @binding(1) var<storage, read> active_positions: F32Buffer;
@group(0) @binding(2) var<storage, read> active_states: F32Buffer;
@group(0) @binding(3) var<storage, read> active_material: F32Buffer;
@group(0) @binding(4) var<storage, read_write> mode_positions: F32Buffer;
@group(0) @binding(5) var<storage, read_write> mode_states: F32Buffer;
@group(0) @binding(6) var<storage, read> mode_active_rows: U32Buffer;
@group(0) @binding(7) var<storage, read> mode_offsets: F32Buffer;

@compute @workgroup_size(128)
fn active_quadrature_prolong_main(@builtin(global_invocation_id) global: vec3<u32>) {
    let mode = global.x;
    if (mode >= params.mode_count) {
        return;
    }
    let active_row = mode_active_rows.values[mode];
    if (active_row >= params.active_count) {
        return;
    }

    let active_position_base = active_row * 4u;
    let mode_position_base = mode * 4u;
    let offset_base = mode * 4u;
    for (var axis = 0u; axis < 4u; axis = axis + 1u) {
        mode_positions.values[mode_position_base + axis] =
            active_positions.values[active_position_base + axis]
            + mode_offsets.values[offset_base + axis];
    }

    let active_state_base = active_row * params.state_dims;
    let mode_state_base = mode * params.state_dims;
    let active_material_base = active_row * MATERIAL_STRIDE;
    for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
        var value = active_states.values[active_state_base + channel];
        for (var axis = 0u; axis < params.spatial_dims; axis = axis + 1u) {
            value = value
                + active_material.values[
                    active_material_base
                    + MATERIAL_STATE_JACOBIAN_OFFSET
                    + channel * params.spatial_dims
                    + axis
                ] * mode_offsets.values[offset_base + axis];
        }
        mode_states.values[mode_state_base + channel] = value;
    }
}
