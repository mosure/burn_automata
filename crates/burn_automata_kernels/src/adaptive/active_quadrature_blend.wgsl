const MATERIAL_UPDATE_MASK_MEMBERS: u32 = 6u;
const MATERIAL_UPDATE_KEY_OFFSET: u32 = 13u;
const MATERIAL_UPDATE_WEIGHT_OFFSET: u32 =
    MATERIAL_UPDATE_KEY_OFFSET + MATERIAL_UPDATE_MASK_MEMBERS;
const MATERIAL_STATE_JACOBIAN_CAPACITY: u32 = 24u * 3u;
const MATERIAL_STRIDE: u32 =
    MATERIAL_UPDATE_WEIGHT_OFFSET
    + MATERIAL_UPDATE_MASK_MEMBERS
    + MATERIAL_STATE_JACOBIAN_CAPACITY;

struct ParamsBuffer {
    values: array<vec4<u32>, 24>,
};

struct F32Buffer {
    values: array<f32>,
};

@group(0) @binding(0) var<uniform> params: ParamsBuffer;
@group(0) @binding(1) var<storage, read> source_positions: F32Buffer;
@group(0) @binding(2) var<storage, read> source_states: F32Buffer;
@group(0) @binding(3) var<storage, read_write> candidate_positions: F32Buffer;
@group(0) @binding(4) var<storage, read_write> candidate_states: F32Buffer;
@group(0) @binding(5) var<storage, read> material: F32Buffer;

fn pu(index: u32) -> u32 {
    return params.values[index / 4u][index % 4u];
}

fn pf(index: u32) -> f32 {
    return bitcast<f32>(pu(index));
}

fn hash_u32(value: u32) -> u32 {
    var x = value;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn random01(particle: u32, step: u32, seed: u32) -> f32 {
    let mixed = hash_u32(particle ^ hash_u32(step + 0x9e3779b9u) ^ seed);
    return f32(mixed >> 8u) * (1.0 / 16777216.0);
}

fn is_coarse(row: u32) -> bool {
    var footprint = sqrt(max(material.values[row * MATERIAL_STRIDE], 1.0e-20)
        / 3.141592653589793);
    if (pu(4u) == 3u) {
        footprint = pow(
            max(material.values[row * MATERIAL_STRIDE], 1.0e-20)
                / (4.0 * 3.141592653589793 / 3.0),
            1.0 / 3.0,
        );
    }
    return footprint > pf(42u) * (1.0 + 32.0 * 1.1920929e-7);
}

fn update_mask(row: u32) -> f32 {
    let probability = pf(25u);
    if (probability >= 1.0) {
        return 1.0;
    }
    if (probability <= 0.0) {
        return 0.0;
    }
    if (pu(89u) != 0u && is_coarse(row)) {
        return probability;
    }
    var mask = 0.0;
    for (var member = 0u; member < MATERIAL_UPDATE_MASK_MEMBERS; member = member + 1u) {
        let weight = material.values[
            row * MATERIAL_STRIDE + MATERIAL_UPDATE_WEIGHT_OFFSET + member
        ];
        if (weight <= 0.0) {
            break;
        }
        let key = bitcast<u32>(material.values[
            row * MATERIAL_STRIDE + MATERIAL_UPDATE_KEY_OFFSET + member
        ]);
        mask = mask + weight * select(
            0.0,
            1.0,
            random01(key, pu(26u), pu(48u)) < probability,
        );
    }
    return mask;
}

@compute @workgroup_size(128)
fn active_quadrature_blend_main(@builtin(global_invocation_id) global: vec3<u32>) {
    let row = global.x;
    if (row >= pu(0u)) {
        return;
    }
    let mask = update_mask(row);
    let position_base = row * 4u;
    for (var axis = 0u; axis < 4u; axis = axis + 1u) {
        let source = source_positions.values[position_base + axis];
        let candidate = candidate_positions.values[position_base + axis];
        candidate_positions.values[position_base + axis] =
            source + mask * (candidate - source);
    }
    let state_dims = pu(2u);
    let state_base = row * state_dims;
    for (var channel = 0u; channel < state_dims; channel = channel + 1u) {
        let source = source_states.values[state_base + channel];
        let candidate = candidate_states.values[state_base + channel];
        candidate_states.values[state_base + channel] =
            source + mask * (candidate - source);
    }
}
