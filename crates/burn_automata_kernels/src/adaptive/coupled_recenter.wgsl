const MAX_STATE_DIMS: u32 = 24u;
const MATERIAL_STATE_JACOBIAN_OFFSET: u32 = 25u;
const MATERIAL_CLOSURE_MODE_OFFSET: u32 = 97u;
const MATERIAL_CLOSURE_BASIS_OFFSET: u32 = 121u;
const MATERIAL_CLOSURE_PHASE_OFFSET: u32 = 125u;
const MATERIAL_STRIDE: u32 = 127u;

struct Params {
    fine_count: u32,
    student_count: u32,
    state_dims: u32,
    total_students: u32,
    closure_enabled: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct F32Buffer {
    values: array<f32>,
};

struct U32Buffer {
    values: array<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> fine_positions: F32Buffer;
@group(0) @binding(2) var<storage, read_write> fine_states: F32Buffer;
@group(0) @binding(3) var<storage, read> student_positions: F32Buffer;
@group(0) @binding(4) var<storage, read> student_states: F32Buffer;
@group(0) @binding(5) var<storage, read> member_offsets: U32Buffer;
@group(0) @binding(6) var<storage, read> member_leaves: U32Buffer;
@group(0) @binding(7) var<storage, read> student_material: F32Buffer;
@group(0) @binding(8) var<storage, read> fine_material: F32Buffer;

fn determinant_3x3(first: vec3<f32>, second: vec3<f32>, third: vec3<f32>) -> f32 {
    return dot(first, cross(second, third));
}

@compute @workgroup_size(128)
fn coupled_recenter_main(@builtin(global_invocation_id) global: vec3<u32>) {
    let group = global.x;
    if (group >= params.total_students) {
        return;
    }
    let start = member_offsets.values[group];
    let end = member_offsets.values[group + 1u];
    if (start >= end) {
        return;
    }

    var mean_position = vec4<f32>(0.0);
    var mean_state: array<f32, MAX_STATE_DIMS>;
    for (var channel = 0u; channel < MAX_STATE_DIMS; channel = channel + 1u) {
        mean_state[channel] = 0.0;
    }
    for (var cursor = start; cursor < end; cursor = cursor + 1u) {
        let leaf = member_leaves.values[cursor];
        let position_base = leaf * 4u;
        mean_position = mean_position + vec4<f32>(
            fine_positions.values[position_base],
            fine_positions.values[position_base + 1u],
            fine_positions.values[position_base + 2u],
            fine_positions.values[position_base + 3u],
        );
        let state_base = leaf * params.state_dims;
        for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
            mean_state[channel] = mean_state[channel] + fine_states.values[state_base + channel];
        }
    }
    let inverse_count = 1.0 / f32(end - start);
    mean_position = mean_position * inverse_count;
    for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
        mean_state[channel] = mean_state[channel] * inverse_count;
    }

    let student_position_base = group * 4u;
    let student_state_base = group * params.state_dims;
    let student_position = vec4<f32>(
        student_positions.values[student_position_base],
        student_positions.values[student_position_base + 1u],
        student_positions.values[student_position_base + 2u],
        student_positions.values[student_position_base + 3u],
    );
    var detail_x: array<f32, 4>;
    var detail_y: array<f32, 4>;
    for (var cursor = start; cursor < end; cursor = cursor + 1u) {
        let leaf = member_leaves.values[cursor];
        let position_base = leaf * 4u;
        let detail_position = vec4<f32>(
            fine_positions.values[position_base],
            fine_positions.values[position_base + 1u],
            fine_positions.values[position_base + 2u],
            fine_positions.values[position_base + 3u],
        ) - mean_position;
        let recentered = student_position + detail_position;
        fine_positions.values[position_base] = recentered.x;
        fine_positions.values[position_base + 1u] = recentered.y;
        fine_positions.values[position_base + 2u] = recentered.z;
        fine_positions.values[position_base + 3u] = recentered.w;

        if (end - start == 4u) {
            detail_x[cursor - start] = detail_position.x;
            detail_y[cursor - start] = detail_position.y;
        }

        if (params.closure_enabled == 0u || end - start != 4u) {
            let state_base = leaf * params.state_dims;
            for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
                fine_states.values[state_base + channel] =
                    student_states.values[student_state_base + channel]
                    + fine_states.values[state_base + channel]
                    - mean_state[channel];
            }
        }
    }

    if (params.closure_enabled == 0u || end - start != 4u) {
        return;
    }

    let material_base = group * MATERIAL_STRIDE;
    let weight_direction = vec4<f32>(0.5);
    var null_direction = vec4<f32>(
        student_material.values[material_base + MATERIAL_CLOSURE_BASIS_OFFSET],
        student_material.values[material_base + MATERIAL_CLOSURE_BASIS_OFFSET + 1u],
        student_material.values[material_base + MATERIAL_CLOSURE_BASIS_OFFSET + 2u],
        student_material.values[material_base + MATERIAL_CLOSURE_BASIS_OFFSET + 3u],
    );
    null_direction = null_direction - dot(null_direction, weight_direction) * weight_direction;
    let null_norm = length(null_direction);
    if (null_norm <= 1.0e-10) {
        return;
    }
    null_direction = null_direction / null_norm;

    var plane0 = vec4<f32>(0.0);
    var plane1 = vec4<f32>(0.0);
    var plane_count = 0u;
    for (var seed_axis = 0u; seed_axis < 4u; seed_axis = seed_axis + 1u) {
        var candidate = vec4<f32>(0.0);
        candidate[seed_axis] = 1.0;
        candidate = candidate - dot(candidate, weight_direction) * weight_direction;
        candidate = candidate - dot(candidate, null_direction) * null_direction;
        if (plane_count > 0u) {
            candidate = candidate - dot(candidate, plane0) * plane0;
        }
        let candidate_norm = length(candidate);
        if (candidate_norm > 1.0e-10) {
            if (plane_count == 0u) {
                plane0 = candidate / candidate_norm;
            } else {
                plane1 = candidate / candidate_norm;
            }
            plane_count = plane_count + 1u;
            if (plane_count == 2u) {
                break;
            }
        }
    }
    if (plane_count != 2u) {
        return;
    }

    var intrinsic_xx = 0.0;
    var intrinsic_xy = 0.0;
    var intrinsic_yy = 0.0;
    for (var child = 0u; child < 4u; child = child + 1u) {
        let leaf = member_leaves.values[start + child];
        let fine_material_base = leaf * MATERIAL_STRIDE;
        intrinsic_xx = intrinsic_xx + 0.25 * fine_material.values[fine_material_base + 1u];
        intrinsic_xy = intrinsic_xy + 0.25 * fine_material.values[fine_material_base + 2u];
        intrinsic_yy = intrinsic_yy + 0.25 * fine_material.values[fine_material_base + 5u];
    }
    let offset_xx = student_material.values[material_base + 1u] - intrinsic_xx;
    let offset_xy = student_material.values[material_base + 2u] - intrinsic_xy;
    let offset_yy = student_material.values[material_base + 5u] - intrinsic_yy;
    let offset_det_sqrt = sqrt(max(offset_xx * offset_yy - offset_xy * offset_xy, 0.0));
    let offset_denominator = sqrt(max(offset_xx + offset_yy + 2.0 * offset_det_sqrt, 0.0));
    if (offset_denominator <= 1.0e-10) {
        return;
    }
    let covariance_sqrt_xx = (offset_xx + offset_det_sqrt) / offset_denominator;
    let covariance_sqrt_xy = offset_xy / offset_denominator;
    let covariance_sqrt_yy = (offset_yy + offset_det_sqrt) / offset_denominator;

    var phase = vec2<f32>(
        student_material.values[material_base + MATERIAL_CLOSURE_PHASE_OFFSET],
        student_material.values[material_base + MATERIAL_CLOSURE_PHASE_OFFSET + 1u],
    );
    let phase_norm = length(phase);
    if (phase_norm <= 1.0e-10) {
        return;
    }
    phase = phase / phase_norm;
    let column0 = vec3<f32>(0.25, plane0.x, plane1.x);
    let column1 = vec3<f32>(0.25, plane0.y, plane1.y);
    let column2 = vec3<f32>(0.25, plane0.z, plane1.z);
    let column3 = vec3<f32>(0.25, plane0.w, plane1.w);
    var canonical_normal = vec4<f32>(
        determinant_3x3(column1, column2, column3),
        -determinant_3x3(column0, column2, column3),
        determinant_3x3(column0, column1, column3),
        -determinant_3x3(column0, column1, column2),
    );
    let canonical_norm = length(canonical_normal);
    if (canonical_norm <= 1.0e-10) {
        return;
    }
    canonical_normal = canonical_normal / canonical_norm;
    let orientation = select(-1.0, 1.0, dot(canonical_normal, null_direction) >= 0.0);
    // Equal-measure four-child rows have weighted tangent metric 0.25 I,
    // so the inverse metric square root is exactly 2 I.
    let coordinate_xx =
        2.0 * (covariance_sqrt_xx * phase.x + covariance_sqrt_xy * phase.y);
    let coordinate_xy = 2.0 * orientation
        * (-covariance_sqrt_xx * phase.y + covariance_sqrt_xy * phase.x);
    let coordinate_yx =
        2.0 * (covariance_sqrt_xy * phase.x + covariance_sqrt_yy * phase.y);
    let coordinate_yy = 2.0 * orientation
        * (-covariance_sqrt_xy * phase.y + covariance_sqrt_yy * phase.x);
    for (var child = 0u; child < 4u; child = child + 1u) {
        let leaf = member_leaves.values[start + child];
        let offset_x = coordinate_xx * plane0[child] + coordinate_xy * plane1[child];
        let offset_y = coordinate_yx * plane0[child] + coordinate_yy * plane1[child];
        let position_base = leaf * 4u;
        fine_positions.values[position_base] = student_position.x + offset_x;
        fine_positions.values[position_base + 1u] = student_position.y + offset_y;
        detail_x[child] = offset_x;
        detail_y[child] = offset_y;
    }

    var constraint_x: array<f32, 4>;
    var constraint_y: array<f32, 4>;
    var gxx = 0.0;
    var gxy = 0.0;
    var gyy = 0.0;
    for (var child = 0u; child < 4u; child = child + 1u) {
        constraint_x[child] = 0.25 * detail_x[child];
        constraint_y[child] = 0.25 * detail_y[child];
        gxx = gxx + constraint_x[child] * constraint_x[child];
        gxy = gxy + constraint_x[child] * constraint_y[child];
        gyy = gyy + constraint_y[child] * constraint_y[child];
    }
    let determinant = gxx * gyy - gxy * gxy;
    if (abs(determinant) <= 1.0e-20) {
        return;
    }

    var state_null_direction = vec4<f32>(
        determinant_3x3(
            vec3<f32>(0.25, constraint_x[1], constraint_y[1]),
            vec3<f32>(0.25, constraint_x[2], constraint_y[2]),
            vec3<f32>(0.25, constraint_x[3], constraint_y[3]),
        ),
        -determinant_3x3(
            vec3<f32>(0.25, constraint_x[0], constraint_y[0]),
            vec3<f32>(0.25, constraint_x[2], constraint_y[2]),
            vec3<f32>(0.25, constraint_x[3], constraint_y[3]),
        ),
        determinant_3x3(
            vec3<f32>(0.25, constraint_x[0], constraint_y[0]),
            vec3<f32>(0.25, constraint_x[1], constraint_y[1]),
            vec3<f32>(0.25, constraint_x[3], constraint_y[3]),
        ),
        -determinant_3x3(
            vec3<f32>(0.25, constraint_x[0], constraint_y[0]),
            vec3<f32>(0.25, constraint_x[1], constraint_y[1]),
            vec3<f32>(0.25, constraint_x[2], constraint_y[2]),
        ),
    );
    let state_null_norm2 = dot(state_null_direction, state_null_direction);
    if (state_null_norm2 <= 1.0e-20) {
        return;
    }
    var anchor_alignment = 0.0;
    for (var child = 0u; child < 4u; child = child + 1u) {
        anchor_alignment = anchor_alignment
            + state_null_direction[child]
                * student_material.values[material_base + MATERIAL_CLOSURE_BASIS_OFFSET + child];
    }
    if (anchor_alignment < 0.0) {
        state_null_direction = -state_null_direction;
    }
    let inverse_null_norm = inverseSqrt(max(state_null_norm2, 1.0e-20));
    for (var channel = 0u; channel < params.state_dims; channel = channel + 1u) {
        let jacobian_base = material_base + MATERIAL_STATE_JACOBIAN_OFFSET + channel * 2u;
        let jacobian_x = student_material.values[jacobian_base];
        let jacobian_y = student_material.values[jacobian_base + 1u];
        let target_x = student_material.values[material_base + 1u] * jacobian_x
            + student_material.values[material_base + 2u] * jacobian_y;
        let target_y = student_material.values[material_base + 4u] * jacobian_x
            + student_material.values[material_base + 5u] * jacobian_y;
        let coefficient_x = (gyy * target_x - gxy * target_y) / determinant;
        let coefficient_y = (gxx * target_y - gxy * target_x) / determinant;
        let mean = student_states.values[student_state_base + channel];
        let mode = student_material.values[
            material_base + MATERIAL_CLOSURE_MODE_OFFSET + channel
        ];
        for (var child = 0u; child < 4u; child = child + 1u) {
            let leaf = member_leaves.values[start + child];
            fine_states.values[leaf * params.state_dims + channel] = mean
                + constraint_x[child] * coefficient_x
                + constraint_y[child] * coefficient_y
                + mode * state_null_direction[child] * inverse_null_norm;
        }
    }
}
