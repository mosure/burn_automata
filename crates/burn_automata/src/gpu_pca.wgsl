const PCA_COMPONENTS: u32 = 3u;
const PCA_WORKGROUP_SIZE: u32 = 128u;
const PCA_ITEMS_PER_THREAD: u32 = 4u;
const PCA_BASIS_WORKGROUP_SIZE: u32 = 32u;

struct PcaParamsBuffer {
    values: array<vec4<u32>, 3>,
};

@group(2) @binding(0) var<uniform> pca_params: PcaParamsBuffer;
@group(2) @binding(1) var<storage, read_write> pca_data: F32Buffer;

var<workgroup> pca_reduction: array<f32, PCA_WORKGROUP_SIZE>;
var<workgroup> pca_basis_reduction: array<f32, PCA_BASIS_WORKGROUP_SIZE>;

fn pca_param_u32(index: u32) -> u32 {
    let lane = pca_params.values[index / 4u];
    let axis = index % 4u;
    if (axis == 0u) {
        return lane.x;
    }
    if (axis == 1u) {
        return lane.y;
    }
    if (axis == 2u) {
        return lane.z;
    }
    return lane.w;
}

fn pca_param_f32(index: u32) -> f32 {
    return bitcast<f32>(pca_param_u32(index));
}

fn pca_active_partial_count() -> u32 {
    return pca_param_u32(0u);
}

fn pca_partial_capacity() -> u32 {
    return pca_param_u32(1u);
}

fn pca_initialized() -> bool {
    return pca_param_u32(3u) != 0u;
}

fn pca_learning_rate() -> f32 {
    return pca_param_f32(4u);
}

fn pca_mean_momentum() -> f32 {
    return pca_param_f32(5u);
}

fn pca_display_momentum() -> f32 {
    return pca_param_f32(6u);
}

fn pca_display_clip_sigma() -> f32 {
    return pca_param_f32(7u);
}

fn pca_epsilon() -> f32 {
    return pca_param_f32(8u);
}

fn pca_display_std_floor() -> f32 {
    return pca_param_f32(9u);
}

fn pca_particle_capacity() -> u32 {
    return pca_param_u32(10u);
}

fn pca_mean_offset() -> u32 {
    return state_dims() * pca_partial_capacity();
}

fn pca_components_offset() -> u32 {
    return pca_mean_offset() + state_dims();
}

fn pca_projected_offset() -> u32 {
    return pca_components_offset() + state_dims() * PCA_COMPONENTS;
}

fn pca_candidate_offset() -> u32 {
    return pca_projected_offset() + pca_particle_capacity() * PCA_COMPONENTS;
}

fn pca_display_center_offset() -> u32 {
    return pca_candidate_offset() + state_dims() * PCA_COMPONENTS;
}

fn pca_display_spread_offset() -> u32 {
    return pca_display_center_offset() + PCA_COMPONENTS;
}

fn pca_partial_index(feature: u32, partial: u32) -> u32 {
    return feature * pca_partial_capacity() + partial;
}

fn pca_mean_index(feature: u32) -> u32 {
    return pca_mean_offset() + feature;
}

fn pca_component_index(feature: u32, component: u32) -> u32 {
    return pca_components_offset() + feature * PCA_COMPONENTS + component;
}

fn pca_projected_index(particle: u32, component: u32) -> u32 {
    return pca_projected_offset() + particle * PCA_COMPONENTS + component;
}

fn pca_candidate_index(feature: u32, component: u32) -> u32 {
    return pca_candidate_offset() + feature * PCA_COMPONENTS + component;
}

fn pca_reduce_sum(value: f32, lane: u32) -> f32 {
    pca_reduction[lane] = value;
    workgroupBarrier();
    var stride = PCA_WORKGROUP_SIZE / 2u;
    loop {
        if (lane < stride) {
            pca_reduction[lane] = pca_reduction[lane] + pca_reduction[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
    return pca_reduction[0u];
}

fn pca_basis_reduce_sum(value: f32, lane: u32) -> f32 {
    pca_basis_reduction[lane] = value;
    workgroupBarrier();
    var stride = PCA_BASIS_WORKGROUP_SIZE / 2u;
    loop {
        if (lane < stride) {
            pca_basis_reduction[lane] =
                pca_basis_reduction[lane] + pca_basis_reduction[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
    return pca_basis_reduction[0u];
}

@compute @workgroup_size(128, 1, 1)
fn pca_partial_mean_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let feature = workgroup_id.y;
    if (feature >= state_dims()) {
        return;
    }
    let lane = local_id.x;
    let chunk_start =
        workgroup_id.x * PCA_WORKGROUP_SIZE * PCA_ITEMS_PER_THREAD
        + lane * PCA_ITEMS_PER_THREAD;
    var sum = 0.0;
    for (var item = 0u; item < PCA_ITEMS_PER_THREAD; item = item + 1u) {
        let particle = chunk_start + item;
        if (particle < total_count()) {
            sum = sum + output_state_channel(particle, feature);
        }
    }
    let reduced = pca_reduce_sum(sum, lane);
    if (lane == 0u) {
        pca_data.values[pca_partial_index(feature, workgroup_id.x)] = reduced;
    }
}

@compute @workgroup_size(128, 1, 1)
fn pca_finalize_mean_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let feature = workgroup_id.x;
    if (feature >= state_dims()) {
        return;
    }
    let lane = local_id.x;
    var sum = 0.0;
    var partial = lane;
    loop {
        if (partial >= pca_active_partial_count()) {
            break;
        }
        sum = sum + pca_data.values[pca_partial_index(feature, partial)];
        partial = partial + PCA_WORKGROUP_SIZE;
    }
    let reduced = pca_reduce_sum(sum, lane);
    if (lane == 0u) {
        let observed = reduced / max(f32(total_count()), 1.0);
        let momentum = select(1.0, pca_mean_momentum(), pca_initialized());
        let index = pca_mean_index(feature);
        pca_data.values[index] = mix(pca_data.values[index], observed, momentum);
    }
}

@compute @workgroup_size(128, 1, 1)
fn pca_project_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle = gid.x;
    if (particle >= total_count()) {
        return;
    }
    for (var component = 0u; component < PCA_COMPONENTS; component = component + 1u) {
        var projected = 0.0;
        for (var feature = 0u; feature < state_dims(); feature = feature + 1u) {
            let centered =
                output_state_channel(particle, feature) - pca_data.values[pca_mean_index(feature)];
            projected = projected
                + centered * pca_data.values[pca_component_index(feature, component)];
        }
        pca_data.values[pca_projected_index(particle, component)] = projected;
    }
}

@compute @workgroup_size(128, 1, 1)
fn pca_oja_candidate_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let feature = workgroup_id.x;
    let component = workgroup_id.y;
    if (feature >= state_dims() || component >= PCA_COMPONENTS) {
        return;
    }
    let lane = local_id.x;
    var sum = 0.0;
    var particle = lane;
    loop {
        if (particle >= total_count()) {
            break;
        }
        let centered =
            output_state_channel(particle, feature) - pca_data.values[pca_mean_index(feature)];
        sum = sum
            + centered * pca_data.values[pca_projected_index(particle, component)];
        particle = particle + PCA_WORKGROUP_SIZE;
    }
    let reduced = pca_reduce_sum(sum, lane);
    if (lane == 0u) {
        pca_data.values[pca_candidate_index(feature, component)] =
            reduced / max(f32(total_count()), 1.0);
    }
}

@compute @workgroup_size(32, 1, 1)
fn pca_stabilize_basis_main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    let feature_count = state_dims();
    for (var component = 0u; component < PCA_COMPONENTS; component = component + 1u) {
        var reference = 0.0;
        var vector = 0.0;
        if (lane < feature_count) {
            let index = pca_component_index(lane, component);
            reference = pca_data.values[index];
            vector = mix(
                reference,
                pca_data.values[pca_candidate_index(lane, component)],
                pca_learning_rate(),
            );
        }
        workgroupBarrier();

        for (var previous = 0u; previous < component; previous = previous + 1u) {
            var previous_value = 0.0;
            if (lane < feature_count) {
                previous_value =
                    pca_data.values[pca_component_index(lane, previous)];
            }
            let coefficient = pca_basis_reduce_sum(vector * previous_value, lane);
            if (lane < feature_count) {
                vector = vector - coefficient * previous_value;
            }
            workgroupBarrier();
        }

        let norm_squared = pca_basis_reduce_sum(vector * vector, lane);
        let valid = norm_squared > pca_epsilon();
        let inverse_norm = inverseSqrt(max(norm_squared, pca_epsilon()));
        if (lane < feature_count) {
            vector = select(reference, vector * inverse_norm, valid);
        }
        workgroupBarrier();

        let sign_dot = pca_basis_reduce_sum(vector * reference, lane);
        let sign = select(-1.0, 1.0, sign_dot >= 0.0);
        if (lane < feature_count) {
            pca_data.values[pca_component_index(lane, component)] = vector * sign;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(128, 1, 1)
fn pca_display_stats_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let component = workgroup_id.x;
    if (component >= PCA_COMPONENTS) {
        return;
    }
    let lane = local_id.x;
    var sum = 0.0;
    var sum_squared = 0.0;
    var particle = lane;
    loop {
        if (particle >= total_count()) {
            break;
        }
        let value = pca_data.values[pca_projected_index(particle, component)];
        sum = sum + value;
        sum_squared = sum_squared + value * value;
        particle = particle + PCA_WORKGROUP_SIZE;
    }
    let reduced_sum = pca_reduce_sum(sum, lane);
    workgroupBarrier();
    let reduced_sum_squared = pca_reduce_sum(sum_squared, lane);
    if (lane == 0u) {
        let denominator = max(f32(total_count()), 1.0);
        let center = reduced_sum / denominator;
        let variance = max(reduced_sum_squared / denominator - center * center, pca_epsilon());
        let spread = sqrt(variance) + pca_display_std_floor();
        let momentum = select(1.0, pca_display_momentum(), pca_initialized());
        let center_index = pca_display_center_offset() + component;
        let spread_index = pca_display_spread_offset() + component;
        pca_data.values[center_index] = mix(pca_data.values[center_index], center, momentum);
        pca_data.values[spread_index] = mix(pca_data.values[spread_index], spread, momentum);
    }
}

fn pca_semantic_display(value: f32, component: u32) -> f32 {
    let centered = value - pca_data.values[pca_display_center_offset() + component];
    let spread =
        pca_data.values[pca_display_spread_offset() + component] + pca_display_std_floor();
    let normalized = centered / spread / pca_display_clip_sigma();
    let signed = normalized / (sqrt(normalized * normalized + pca_epsilon()) + 1.0);
    return clamp(0.5 + 0.5 * signed, 0.0, 1.0);
}

@compute @workgroup_size(128, 1, 1)
fn write_gaussian_pca_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle = gid.x;
    if (particle >= total_count()) {
        return;
    }
    let projected_base = particle * PCA_COMPONENTS;
    let projected = vec3<f32>(
        pca_data.values[pca_projected_offset() + projected_base],
        pca_data.values[pca_projected_offset() + projected_base + 1u],
        pca_data.values[pca_projected_offset() + projected_base + 2u],
    );
    let color = vec3<f32>(
        pca_semantic_display(projected.x, 0u),
        pca_semantic_display(projected.y, 1u),
        pca_semantic_display(projected.z, 2u),
    );
    let pos = output_position(particle);
    let base4 = particle * 4u;
    gaussian_position_visibility.values[base4] = pos.x;
    gaussian_position_visibility.values[base4 + 1u] = pos.y;
    gaussian_position_visibility.values[base4 + 2u] =
        select(0.0, pos.z, spatial_dims() == 3u);
    gaussian_position_visibility.values[base4 + 3u] = 1.0;
    write_gaussian_sh0_color(particle, color);
    write_gaussian_geometry_from_output(particle);
}
