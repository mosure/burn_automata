const MAX_STATE_DIMS: u32 = 24u;
const MAX_HIDDEN_DIMS: u32 = 320u;
const MAX_FEATURE_DIMS: u32 = 192u;
const MAX_OUTPUT_DIMS: u32 = 32u;
const MAX_DIMS: u32 = 3u;
const GAUSSIAN_SH_COEFF_COUNT: u32 = 48u;
const SH_C0: f32 = 0.28209479177387814;
const TILE_SIZE: u32 = 128u;
const COOP_SIZE: u32 = 32u;
const SCAN_SIZE: u32 = 256u;
const LAYOUT_LINKED_LIST: u32 = 0u;
const LAYOUT_FIXED_BUCKETS: u32 = 1u;
const LAYOUT_SORTED_CELLS: u32 = 2u;
const LAYOUT_BVH: u32 = 3u;
const LAYOUT_SORTED_BVH: u32 = 4u;
const LAYOUT_MORTON_BVH: u32 = 5u;
const LAYOUT_COOPERATIVE_SORTED_CELLS: u32 = 6u;
const LAYOUT_SUBGROUP_COOPERATIVE_SORTED_CELLS: u32 = 7u;
const BVH_HEADER_U32: u32 = 4u;
const BVH_NODE_U32: u32 = 9u;
const BVH_STACK_SIZE: u32 = 64u;
const COOP_BLUR_BASE: u32 = 0u;
const COOP_STATE_GRAD_BASE: u32 = COOP_BLUR_BASE + MAX_STATE_DIMS;
const COOP_DENSITY_GRAD_BASE: u32 = COOP_STATE_GRAD_BASE + MAX_STATE_DIMS * MAX_DIMS;
const COOP_MOMENT_BASE: u32 = COOP_DENSITY_GRAD_BASE + MAX_DIMS;
const MAX_CLOSURE_CONTEXT_DIMS: u32 = MAX_STATE_DIMS + 6u;
const COOP_CLOSURE_BLUR_BASE: u32 = COOP_MOMENT_BASE + MAX_DIMS * MAX_DIMS;
const COOP_COMPONENTS: u32 = COOP_CLOSURE_BLUR_BASE + MAX_CLOSURE_CONTEXT_DIMS;
const ADAPTIVE_RADIX_BITS: u32 = 8u;
const ADAPTIVE_RADIX_BINS: u32 = 1u << ADAPTIVE_RADIX_BITS;
const GROWTH_3D_MIN_OPACITY_LOGIT: f32 = -8.0;
const GROWTH_3D_MAX_OPACITY_LOGIT: f32 = 24.0;
const MATERIAL_UPDATE_MASK_MEMBERS: u32 = 6u;
const MATERIAL_BANDWIDTH_OFFSET: u32 = 12u;
const MATERIAL_UPDATE_KEY_OFFSET: u32 = 13u;
const MATERIAL_UPDATE_WEIGHT_OFFSET: u32 =
    MATERIAL_UPDATE_KEY_OFFSET + MATERIAL_UPDATE_MASK_MEMBERS;
const MATERIAL_STATE_JACOBIAN_OFFSET: u32 =
    MATERIAL_UPDATE_WEIGHT_OFFSET + MATERIAL_UPDATE_MASK_MEMBERS;
const MATERIAL_STATE_JACOBIAN_CAPACITY: u32 = MAX_STATE_DIMS * MAX_DIMS;
const MATERIAL_CLOSURE_MODE_OFFSET: u32 =
    MATERIAL_STATE_JACOBIAN_OFFSET + MATERIAL_STATE_JACOBIAN_CAPACITY;
const MATERIAL_CLOSURE_MODE_CAPACITY: u32 = MAX_STATE_DIMS;
const MATERIAL_CLOSURE_BASIS_OFFSET: u32 =
    MATERIAL_CLOSURE_MODE_OFFSET + MATERIAL_CLOSURE_MODE_CAPACITY;
const MATERIAL_CLOSURE_BASIS_CAPACITY: u32 = 4u;
const MATERIAL_CLOSURE_PHASE_OFFSET: u32 =
    MATERIAL_CLOSURE_BASIS_OFFSET + MATERIAL_CLOSURE_BASIS_CAPACITY;
const MATERIAL_CLOSURE_PHASE_CAPACITY: u32 = 2u;
const MATERIAL_STRIDE: u32 =
    MATERIAL_CLOSURE_PHASE_OFFSET + MATERIAL_CLOSURE_PHASE_CAPACITY;

struct U32Buffer {
    values: array<u32>,
};

struct F32Buffer {
    values: array<f32>,
};

struct AtomicU32Buffer {
    values: array<atomic<u32>>,
};

struct ParamsBuffer {
    values: array<vec4<u32>, 27>,
};

struct MaterialGaussianGeometry {
    scale: vec3<f32>,
    rotation: vec4<f32>,
    opacity: f32,
};

@group(0) @binding(0) var<uniform> params: ParamsBuffer;
@group(0) @binding(1) var<storage, read> positions: F32Buffer;
@group(0) @binding(2) var<storage, read> states: F32Buffer;
@group(0) @binding(3) var<storage, read> weights: F32Buffer;
@group(0) @binding(4) var<storage, read_write> linked_grid: AtomicU32Buffer;
@group(0) @binding(5) var<storage, read_write> out_positions: F32Buffer;
@group(0) @binding(6) var<storage, read_write> out_states: F32Buffer;
@group(0) @binding(7) var<storage, read_write> density: F32Buffer;
@group(0) @binding(8) var<storage, read_write> indirect_args: AtomicU32Buffer;
@group(0) @binding(9) var<storage, read_write> material_data: F32Buffer;

@group(1) @binding(0) var<storage, read_write> gaussian_position_visibility: F32Buffer;
@group(1) @binding(1) var<storage, read_write> gaussian_spherical_harmonic: F32Buffer;
@group(1) @binding(2) var<storage, read_write> gaussian_rotation: F32Buffer;
@group(1) @binding(3) var<storage, read_write> gaussian_scale_opacity: F32Buffer;

const NIL: u32 = 0xffffffffu;

var<workgroup> tile_indices: array<u32, TILE_SIZE>;
var<workgroup> tile_positions: array<vec4<f32>, TILE_SIZE>;
var<workgroup> tile_density: array<f32, TILE_SIZE>;
var<workgroup> tile_states: array<f32, TILE_SIZE * MAX_STATE_DIMS>;
var<workgroup> tile_center: vec3<i32>;
var<workgroup> tile_dispatch: vec4<u32>;
var<workgroup> tile_neighbor: vec2<u32>;
var<workgroup> tile_mismatch: atomic<u32>;
var<workgroup> scan_values: array<u32, SCAN_SIZE>;
var<workgroup> coop_values: array<f32, COOP_SIZE * COOP_COMPONENTS>;
var<workgroup> coop_reduced_values: array<f32, COOP_COMPONENTS>;
var<workgroup> coop_feature: array<f32, MAX_FEATURE_DIMS>;
var<workgroup> coop_hidden: array<f32, MAX_HIDDEN_DIMS>;
var<workgroup> coop_update_values: array<f32, MAX_OUTPUT_DIMS>;
var<workgroup> cooperative_mask: f32;
var<workgroup> adaptive_radix_counts: array<atomic<u32>, ADAPTIVE_RADIX_BINS>;
var<workgroup> adaptive_update_active: atomic<u32>;
var<workgroup> adaptive_support_count: atomic<u32>;
var<workgroup> adaptive_selection_prefix: u32;
var<workgroup> adaptive_selection_rank: u32;
var<workgroup> adaptive_cutoff_distance: u32;
var<workgroup> adaptive_cutoff_index: u32;
var<workgroup> adaptive_min_cutoff_index: atomic<u32>;
var<workgroup> adaptive_retained_jacobian_enabled: u32;

fn pu(index: u32) -> u32 {
    let lane = params.values[index / 4u];
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

fn pf(index: u32) -> f32 {
    return bitcast<f32>(pu(index));
}

fn total_count() -> u32 {
    return pu(0u);
}

fn resident_capacity() -> u32 {
    return max(pu(104u), total_count());
}

fn particle_count() -> u32 {
    return pu(1u);
}

fn batch_size() -> u32 {
    return max(pu(47u), 1u);
}

fn cooperative_particle_index(workgroup: vec3<u32>) -> u32 {
    return workgroup.x + workgroup.y * particle_count();
}

fn particle_lane(index: u32) -> u32 {
    return index / max(particle_count(), 1u);
}

fn base_cell_count() -> u32 {
    return pu(91u);
}

fn support_bin_count() -> u32 {
    return max(pu(90u), 1u);
}

fn support_bin_min() -> f32 {
    return pf(92u);
}

fn support_bin_max() -> f32 {
    return pf(93u);
}

fn support_bin_ratio() -> f32 {
    return pf(94u);
}

fn grid_cells_per_lane() -> u32 {
    return base_cell_count() * support_bin_count();
}

fn state_dims() -> u32 {
    return pu(2u);
}

fn hidden_dims() -> u32 {
    return pu(3u);
}

fn spatial_dims() -> u32 {
    return pu(4u);
}

fn feature_dims() -> u32 {
    return pu(5u);
}

fn output_dims() -> u32 {
    return pu(6u);
}

fn cell_count() -> u32 {
    return pu(10u);
}

fn grid_size_axis(axis: u32) -> u32 {
    if (axis == 0u) {
        return pu(7u);
    }
    if (axis == 1u) {
        return pu(8u);
    }
    return pu(9u);
}

fn is_periodic() -> bool {
    return pu(11u) != 0u;
}

fn has_position_features() -> bool {
    return pu(14u) != 0u;
}

fn eps() -> f32 {
    return pf(15u);
}

fn alpha() -> f32 {
    return pf(17u);
}

fn dt() -> f32 {
    return pf(18u);
}

fn smooth_coef() -> f32 {
    return pf(19u);
}

fn spiky_coef() -> f32 {
    return pf(20u);
}

fn density_scale() -> f32 {
    return pf(21u);
}

fn grad_scale() -> f32 {
    return pf(22u);
}

fn bucket_capacity() -> u32 {
    return pu(23u);
}

fn motion_eps() -> f32 {
    return pf(24u);
}

fn update_prob() -> f32 {
    return pf(25u);
}

fn step_index() -> u32 {
    return pu(26u);
}

fn random_seed() -> u32 {
    return pu(27u);
}

fn is_particle_grid() -> bool {
    return pu(28u) != 0u;
}

fn neighbor_layout() -> u32 {
    return pu(29u);
}

fn bvh_build_level() -> u32 {
    return pu(30u);
}

fn bvh_leaf_count() -> u32 {
    return pu(31u);
}

fn bvh_sort_count() -> u32 {
    return pu(32u);
}

fn bvh_sort_k() -> u32 {
    return pu(33u);
}

fn bvh_sort_j() -> u32 {
    return pu(34u);
}

fn material_enabled() -> bool {
    return pu(35u) != 0u;
}

fn mean_represented_measure() -> f32 {
    return pf(36u);
}

fn display_scale_per_footprint() -> f32 {
    return pf(37u);
}

fn render_transition_steps() -> u32 {
    return pu(38u);
}

fn render_transition_start_step() -> u32 {
    return params.values[9].w;
}

fn particle_measure(index: u32) -> f32 {
    return select(1.0, material_data.values[index * MATERIAL_STRIDE], material_enabled());
}

fn material_covariance_value(index: u32, element: u32) -> f32 {
    return material_data.values[index * MATERIAL_STRIDE + 1u + element];
}

fn material_state_jacobian(index: u32, component: u32) -> f32 {
    return material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_STATE_JACOBIAN_OFFSET + component
    ];
}

fn material_closure_mode(index: u32, channel: u32) -> f32 {
    return material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_MODE_OFFSET + channel
    ];
}

fn material_closure_basis(index: u32, component: u32) -> f32 {
    return material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_BASIS_OFFSET + component
    ];
}

fn material_closure_phase(index: u32, component: u32) -> f32 {
    return material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_PHASE_OFFSET + component
    ];
}

fn material_closure_context(index: u32, component: u32) -> f32 {
    if (component < 4u) {
        return material_closure_basis(index, component);
    }
    if (component < 6u) {
        return material_closure_phase(index, component - 4u);
    }
    return material_closure_mode(index, component - 6u);
}

fn add_material_closure_mode(index: u32, channel: u32, value: f32) {
    material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_MODE_OFFSET + channel
    ] = material_closure_mode(index, channel) + value;
}

fn set_material_closure_phase(index: u32, component: u32, value: f32) {
    material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_PHASE_OFFSET + component
    ] = value;
}

fn set_material_closure_basis(index: u32, component: u32, value: f32) {
    material_data.values[
        index * MATERIAL_STRIDE + MATERIAL_CLOSURE_BASIS_OFFSET + component
    ] = value;
}

fn material_render_from_scale(index: u32) -> f32 {
    return material_data.values[index * MATERIAL_STRIDE + 10u];
}

fn material_render_target_footprint(index: u32) -> f32 {
    return material_data.values[index * MATERIAL_STRIDE + 11u];
}

fn particle_bandwidth(index: u32) -> f32 {
    return select(
        eps(),
        material_data.values[index * MATERIAL_STRIDE + MATERIAL_BANDWIDTH_OFFSET],
        material_enabled(),
    );
}

fn material_update_mask(index: u32, probability: f32, step: u32, seed: u32) -> f32 {
    var mask = 0.0;
    for (var member = 0u; member < MATERIAL_UPDATE_MASK_MEMBERS; member = member + 1u) {
        let weight = material_data.values[
            index * MATERIAL_STRIDE + MATERIAL_UPDATE_WEIGHT_OFFSET + member
        ];
        if (weight <= 0.0) {
            break;
        }
        let key = bitcast<u32>(material_data.values[
            index * MATERIAL_STRIDE + MATERIAL_UPDATE_KEY_OFFSET + member
        ]);
        mask = mask + weight * select(0.0, 1.0, random01(key, step, seed) < probability);
    }
    return mask;
}

fn max_material_bandwidth() -> f32 {
    return select(eps(), pf(86u), material_enabled());
}

fn scale_equivariant() -> bool {
    return pu(87u) != 0u;
}

fn adaptive_scale_power(value: f32, power: f32) -> f32 {
    if (power == 8.0) {
        let value2 = value * value;
        let value4 = value2 * value2;
        return value4 * value4;
    }
    if (power == 4.0) {
        let value2 = value * value;
        return value2 * value2;
    }
    if (power == 2.0) {
        return value * value;
    }
    return pow(value, power);
}

fn adaptive_inverse_scale_power(value: f32, power: f32) -> f32 {
    if (power == 8.0) {
        return sqrt(sqrt(sqrt(value)));
    }
    if (power == 4.0) {
        return sqrt(sqrt(value));
    }
    if (power == 2.0) {
        return sqrt(value);
    }
    return pow(value, 1.0 / power);
}

fn adaptive_pair_bandwidth(lhs_index: u32, rhs_index: u32) -> f32 {
    if (!material_enabled()) {
        return eps();
    }
    let lhs = particle_bandwidth(lhs_index);
    let rhs = particle_bandwidth(rhs_index);
    if (lhs == rhs) {
        return lhs;
    }
    let power = adaptive_pair_scale_power();
    return adaptive_inverse_scale_power(
        0.5 * (adaptive_scale_power(lhs, power) + adaptive_scale_power(rhs, power)),
        power,
    );
}

fn adaptive_max_pair_bandwidth(index: u32) -> f32 {
    if (!material_enabled()) {
        return eps();
    }
    let lhs = particle_bandwidth(index);
    let rhs = max_material_bandwidth();
    if (lhs == rhs) {
        return lhs;
    }
    let power = adaptive_pair_scale_power();
    return adaptive_inverse_scale_power(
        0.5 * (adaptive_scale_power(lhs, power) + adaptive_scale_power(rhs, power)),
        power,
    );
}

fn adaptive_pair_bandwidth_values(lhs: f32, rhs: f32) -> f32 {
    if (lhs == rhs) {
        return lhs;
    }
    let power = adaptive_pair_scale_power();
    return adaptive_inverse_scale_power(
        0.5 * (adaptive_scale_power(lhs, power) + adaptive_scale_power(rhs, power)),
        power,
    );
}

fn support_bin_for_bandwidth(bandwidth: f32) -> u32 {
    let count = support_bin_count();
    if (count <= 1u) {
        return 0u;
    }
    var bin = 0u;
    var upper = min(support_bin_min() * support_bin_ratio(), support_bin_max());
    while (bin + 1u < count && upper < bandwidth) {
        bin = bin + 1u;
        upper = min(upper * support_bin_ratio(), support_bin_max());
    }
    return bin;
}

fn support_bin_upper(bin: u32) -> f32 {
    if (support_bin_count() <= 1u) {
        return max_material_bandwidth();
    }
    if (bin + 1u >= support_bin_count()) {
        return support_bin_max();
    }
    var upper = min(support_bin_min() * support_bin_ratio(), support_bin_max());
    for (var index = 0u; index < bin; index = index + 1u) {
        upper = min(upper * support_bin_ratio(), support_bin_max());
    }
    return upper;
}

fn adaptive_support_cell_radius_for_bin(index: u32, source_bin: u32) -> i32 {
    let pair_bandwidth = adaptive_pair_bandwidth_values(
        particle_bandwidth(index),
        support_bin_upper(source_bin),
    );
    return max(i32(ceil(pair_bandwidth / eps())), 1);
}

fn adaptive_support_cell_radius(index: u32) -> i32 {
    return max(i32(ceil(adaptive_max_pair_bandwidth(index) / eps())), 1);
}

fn adaptive_pair_normalized_distance2(lhs_index: u32, rhs_index: u32, r2: f32) -> f32 {
    let bandwidth = adaptive_pair_bandwidth(lhs_index, rhs_index);
    return r2 / (bandwidth * bandwidth);
}

fn adaptive_kernel_value_with_bandwidth(r2: f32, bandwidth: f32) -> f32 {
    let q2 = r2 / (bandwidth * bandwidth);
    if (q2 >= 1.0) {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    var inverse_h_dim = 1.0 / (bandwidth * bandwidth);
    if (spatial_dims() == 3u) {
        inverse_h_dim = inverse_h_dim / bandwidth;
    }
    return shoulder * shoulder * shoulder * inverse_h_dim;
}

fn adaptive_kernel_gradient_with_bandwidth(
    delta: array<f32, MAX_DIMS>,
    r2: f32,
    bandwidth: f32,
    axis: u32,
) -> f32 {
    let q2 = r2 / (bandwidth * bandwidth);
    if (q2 <= 0.0 || q2 >= 1.0) {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    var inverse_h_dim = 1.0 / (bandwidth * bandwidth);
    if (spatial_dims() == 3u) {
        inverse_h_dim = inverse_h_dim / bandwidth;
    }
    return -6.0 * shoulder * shoulder * inverse_h_dim
        / (bandwidth * bandwidth) * delta[axis];
}

fn smoothing_poly6_with_bandwidth(r2: f32, bandwidth: f32) -> f32 {
    if (!material_enabled()) {
        return smoothing_poly6(r2);
    }
    let q2 = r2 / (bandwidth * bandwidth);
    if (q2 >= 1.0) {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    var normalization = 4.0 / 3.141592653589793 / (bandwidth * bandwidth);
    if (spatial_dims() == 3u) {
        normalization = 315.0 / (64.0 * 3.141592653589793)
            / (bandwidth * bandwidth * bandwidth);
    }
    return normalization * shoulder * shoulder * shoulder;
}

fn spiky_gradient_with_bandwidth(
    delta: array<f32, MAX_DIMS>,
    r2: f32,
    bandwidth: f32,
    coeff: f32,
    axis: u32,
) -> f32 {
    if (!material_enabled()) {
        return spiky_gradient(delta, r2, coeff, axis);
    }
    let bandwidth2 = bandwidth * bandwidth;
    if (r2 <= 0.0 || r2 >= bandwidth2) {
        return 0.0;
    }
    let distance = sqrt(r2);
    let bandwidth4 = bandwidth2 * bandwidth2;
    var normalization = 10.0 / 3.141592653589793 / (bandwidth4 * bandwidth);
    if (spatial_dims() == 3u) {
        normalization = 15.0 / 3.141592653589793 / (bandwidth4 * bandwidth2);
    }
    let shoulder = bandwidth - distance;
    return coeff * normalization * 3.0 * shoulder * shoulder / distance * delta[axis];
}

fn particle_density_contribution_with_bandwidth(source: u32, r2: f32, bandwidth: f32) -> f32 {
    return particle_measure(source) * smoothing_poly6_with_bandwidth(r2, bandwidth);
}

fn particle_state_gradient_scale(index: u32) -> f32 {
    if (!material_enabled() || !scale_equivariant()) {
        return grad_scale();
    }
    return grad_scale() * particle_bandwidth(index) / eps();
}

fn particle_density_gradient_scale(index: u32) -> f32 {
    if (!material_enabled() || !scale_equivariant()) {
        return density_scale();
    }
    let ratio = particle_bandwidth(index) / eps();
    let ratio2 = ratio * ratio;
    var scale = ratio2 * ratio;
    if (spatial_dims() == 3u) {
        scale = scale * ratio;
    }
    return density_scale() * scale;
}

fn particle_motion_eps(index: u32) -> f32 {
    if (!material_enabled() || !scale_equivariant()) {
        return motion_eps();
    }
    return motion_eps() * particle_bandwidth(index) / eps();
}

fn adaptive_local_hidden_start() -> u32 {
    return pu(40u);
}

fn adaptive_local_rule_mode() -> u32 {
    return pu(88u);
}

fn material_scale_conditioning_enabled(canonical_feature_dims: u32) -> bool {
    return material_enabled()
        && adaptive_local_rule_mode() == 0u
        && feature_dims() == canonical_feature_dims + 1u;
}

fn residual_material_features_enabled(canonical_feature_dims: u32) -> bool {
    let closure_dims =
        1u
        + spatial_dims() * (spatial_dims() + 1u) / 2u
        + state_dims() * spatial_dims();
    let feature_count = feature_dims();
    return material_enabled()
        && (
            adaptive_compatible_residual_enabled()
            || adaptive_normalized_exposure_residual_enabled()
        )
        && (
            feature_count == canonical_feature_dims + 2u
            || feature_count >= canonical_feature_dims + 2u + closure_dims
        );
}

fn particle_material_scale_feature(index: u32) -> f32 {
    let measure = max(particle_measure(index), 1.0e-20);
    var footprint = sqrt(measure / 3.141592653589793);
    if (spatial_dims() == 3u) {
        footprint = pow(
            measure / (4.0 * 3.141592653589793 / 3.0),
            1.0 / 3.0,
        );
    }
    return clamp(
        footprint / max(adaptive_reference_footprint(), 1.0e-20) - 1.0,
        -0.75,
        3.0,
    );
}

fn adaptive_closure_enabled() -> bool {
    return pu(95u) != 0u;
}

fn adaptive_closure_hidden_dims() -> u32 {
    return pu(96u);
}

fn adaptive_closure_basis_enabled() -> bool {
    return pu(100u) != 0u;
}

fn adaptive_closure_basis_hidden_dims() -> u32 {
    return pu(101u);
}

fn closure_weight_base() -> u32 {
    let hd = hidden_dims();
    return hd * feature_dims() + hd + output_dims() * hd + output_dims();
}

fn closure_b1_offset() -> u32 {
    return closure_weight_base() + adaptive_closure_hidden_dims() * feature_dims();
}

fn closure_w2_offset() -> u32 {
    return closure_b1_offset() + adaptive_closure_hidden_dims();
}

fn closure_b2_offset() -> u32 {
    return closure_w2_offset() + output_dims() * adaptive_closure_hidden_dims();
}

fn closure_basis_rule_weight_base() -> u32 {
    return closure_b2_offset() + output_dims();
}

fn closure_basis_rule_b1_offset() -> u32 {
    return closure_basis_rule_weight_base()
        + adaptive_closure_basis_hidden_dims() * feature_dims();
}

fn closure_basis_rule_w2_offset() -> u32 {
    return closure_basis_rule_b1_offset() + adaptive_closure_basis_hidden_dims();
}

fn closure_basis_rule_b2_offset() -> u32 {
    return closure_basis_rule_w2_offset()
        + output_dims() * adaptive_closure_basis_hidden_dims();
}

fn expected_coarse_update_mask() -> bool {
    return pu(89u) != 0u;
}

fn is_coarse_material(index: u32) -> bool {
    if (!material_enabled()) {
        return false;
    }
    var footprint = sqrt(max(particle_measure(index), 1.0e-20) / 3.141592653589793);
    if (spatial_dims() == 3u) {
        footprint = pow(
            max(particle_measure(index), 1.0e-20) / (4.0 * 3.141592653589793 / 3.0),
            1.0 / 3.0,
        );
    }
    return footprint > adaptive_base_footprint() * (1.0 + 32.0 * 1.1920929e-7);
}

fn is_coarse_density_source(index: u32) -> bool {
    if (!material_enabled()) {
        return false;
    }
    var footprint = sqrt(max(particle_measure(index), 1.0e-20) / 3.141592653589793);
    if (spatial_dims() == 3u) {
        footprint = pow(
            max(particle_measure(index), 1.0e-20) / (4.0 * 3.141592653589793 / 3.0),
            1.0 / 3.0,
        );
    }
    return footprint > adaptive_reference_footprint() * (1.0 + 32.0 * 1.1920929e-7);
}

fn adaptive_local_rule_enabled() -> bool {
    return adaptive_local_rule_mode() != 0u;
}

fn adaptive_compatible_residual_enabled() -> bool {
    return adaptive_local_rule_mode() == 4u;
}

fn adaptive_normalized_exposure_residual_enabled() -> bool {
    return adaptive_local_rule_mode() == 5u;
}

fn adaptive_compatible_residual_gate(index: u32) -> f32 {
    if (!material_enabled()) {
        return 0.0;
    }
    return adaptive_local_residual_scale() * clamp(
        density.values[diagnostics_coarse_exposure_offset() + index],
        0.0,
        1.0,
    );
}

fn adaptive_normalized_primary_enabled() -> bool {
    return adaptive_local_rule_mode() == 2u;
}

fn adaptive_coarse_replacement_enabled() -> bool {
    return adaptive_local_rule_mode() == 3u;
}

fn adaptive_local_residual_scale() -> f32 {
    return pf(41u);
}

fn adaptive_base_footprint() -> f32 {
    return pf(42u);
}

fn adaptive_pair_scale_power() -> f32 {
    return pf(85u);
}

fn adaptive_shepard_epsilon() -> f32 {
    return pf(43u);
}

fn adaptive_moment_regularization() -> f32 {
    return pf(44u);
}

fn adaptive_moment_condition_limit() -> f32 {
    return pf(45u);
}

fn adaptive_max_neighbors() -> u32 {
    return pu(46u);
}

fn adaptive_diagnostics_enabled() -> bool {
    return pu(16u) != 0u;
}

fn adaptive_spacing_min() -> f32 {
    return pf(80u);
}

fn adaptive_spacing_max() -> f32 {
    return pf(81u);
}

fn adaptive_spacing_target() -> f32 {
    return pf(82u);
}

fn adaptive_spacing_root_iterations() -> u32 {
    return pu(83u);
}

fn diagnostics_base_feature_offset() -> u32 {
    return total_count();
}

fn diagnostics_normalized_feature_offset() -> u32 {
    return diagnostics_base_feature_offset() + total_count() * feature_dims();
}

fn diagnostics_base_update_offset() -> u32 {
    return diagnostics_normalized_feature_offset() + total_count() * feature_dims();
}

fn diagnostics_model_update_offset() -> u32 {
    return diagnostics_base_update_offset() + total_count() * output_dims();
}

fn diagnostics_spacing_offset() -> u32 {
    return diagnostics_model_update_offset() + total_count() * output_dims();
}

fn diagnostics_degree_offset() -> u32 {
    return diagnostics_spacing_offset() + total_count();
}

fn diagnostics_coarse_exposure_offset() -> u32 {
    return diagnostics_degree_offset() + total_count();
}

fn adaptive_residual_gate(index: u32) -> f32 {
    let measure = max(particle_measure(index), 1.0e-20);
    var footprint = sqrt(measure / 3.141592653589793);
    if (spatial_dims() == 3u) {
        footprint = pow(measure / (4.0 * 3.141592653589793 / 3.0), 1.0 / 3.0);
    }
    return adaptive_local_residual_scale() * clamp(
        log2(footprint / max(adaptive_base_footprint(), 1.0e-20)),
        -3.0,
        3.0,
    );
}

fn adaptive_local_rule_gate(index: u32) -> f32 {
    if (adaptive_normalized_primary_enabled()) {
        return 1.0;
    }
    if (adaptive_coarse_replacement_enabled()) {
        return select(0.0, adaptive_local_residual_scale(), is_coarse_material(index));
    }
    if (adaptive_normalized_exposure_residual_enabled()) {
        let coarse_exposure = adaptive_local_residual_scale() * clamp(
            density.values[diagnostics_coarse_exposure_offset() + index],
            0.0,
            1.0,
        );
        return max(adaptive_residual_gate(index), coarse_exposure);
    }
    return adaptive_residual_gate(index);
}

fn adaptive_reference_footprint() -> f32 {
    return pf(84u);
}

fn adaptive_hidden_scale(index: u32, hidden: u32) -> f32 {
    if (adaptive_normalized_primary_enabled()) {
        return 0.0;
    }
    let start = adaptive_local_hidden_start();
    if (!material_enabled() || !adaptive_local_rule_enabled() || hidden < start) {
        return 1.0;
    }
    if (adaptive_compatible_residual_enabled()) {
        return adaptive_compatible_residual_gate(index);
    }
    return 0.0;
}

fn adaptive_residual_value(index: u32, output: u32) -> f32 {
    if (!adaptive_local_rule_enabled()) {
        return 0.0;
    }
    if (adaptive_local_rule_mode() == 1u && adaptive_local_residual_scale() == 0.0) {
        return 0.0;
    }
    if (output < spatial_dims()) {
        return out_positions.values[index * 4u + output];
    }
    return out_states.values[index * state_dims() + output - spatial_dims()];
}

fn adaptive_combined_update_value(index: u32, output: u32, base: f32) -> f32 {
    if (adaptive_compatible_residual_enabled()) {
        return base;
    }
    let local = adaptive_residual_value(index, output);
    if (adaptive_normalized_primary_enabled()) {
        return local;
    }
    if (adaptive_coarse_replacement_enabled()) {
        let replacement = (1.0 - adaptive_local_residual_scale()) * base + local;
        return select(base, replacement, is_coarse_material(index));
    }
    return base + local;
}

fn adaptive_kernel_value(r2: f32) -> f32 {
    let h = eps();
    let q2 = r2 / (h * h);
    if (q2 >= 1.0) {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    var inverse_h_dim = 1.0 / (h * h);
    if (spatial_dims() == 3u) {
        inverse_h_dim = inverse_h_dim / h;
    }
    return shoulder * shoulder * shoulder * inverse_h_dim;
}

fn adaptive_kernel_gradient(
    delta: array<f32, MAX_DIMS>,
    r2: f32,
    axis: u32,
) -> f32 {
    let h = eps();
    let q2 = r2 / (h * h);
    if (q2 <= 0.0 || q2 >= 1.0) {
        return 0.0;
    }
    let shoulder = 1.0 - q2;
    var inverse_h_dim = 1.0 / (h * h);
    if (spatial_dims() == 3u) {
        inverse_h_dim = inverse_h_dim / h;
    }
    return -6.0 * shoulder * shoulder * inverse_h_dim / (h * h) * delta[axis];
}

fn particle_volume(index: u32) -> f32 {
    return particle_measure(index) * recip_finite(density.values[index]);
}

fn density_gradient_weight(index: u32) -> f32 {
    if (!material_enabled()) {
        return 1.0;
    }
    return particle_measure(index) / max(mean_represented_measure(), 1.0e-20);
}

fn particle_density_contribution(index: u32, r2: f32) -> f32 {
    return particle_measure(index) * smoothing_poly6(r2);
}

fn is_sorted_layout() -> bool {
    return neighbor_layout() == LAYOUT_SORTED_CELLS
        || neighbor_layout() == LAYOUT_SORTED_BVH
        || neighbor_layout() == LAYOUT_COOPERATIVE_SORTED_CELLS
        || neighbor_layout() == LAYOUT_SUBGROUP_COOPERATIVE_SORTED_CELLS;
}

fn is_bvh_layout() -> bool {
    return neighbor_layout() == LAYOUT_BVH
        || neighbor_layout() == LAYOUT_SORTED_BVH
        || neighbor_layout() == LAYOUT_MORTON_BVH;
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

fn update_mask(idx: u32) -> f32 {
    let probability = update_prob();
    if (probability >= 1.0) {
        return 1.0;
    }
    if (probability <= 0.0) {
        return 0.0;
    }
    if (expected_coarse_update_mask() && is_coarse_material(idx)) {
        return probability;
    }
    let count = max(particle_count(), 1u);
    let lane = idx / count;
    if (material_enabled()) {
        return material_update_mask(idx, probability, step_index(), pu(48u + lane));
    }
    let local_particle = idx - lane * count;
    return select(
        0.0,
        1.0,
        random01(local_particle, step_index(), pu(48u + lane)) < probability,
    );
}

fn position(index: u32) -> vec4<f32> {
    let base = index * 4u;
    return vec4<f32>(
        positions.values[base],
        positions.values[base + 1u],
        positions.values[base + 2u],
        positions.values[base + 3u],
    );
}

fn output_position(index: u32) -> vec4<f32> {
    let base = index * 4u;
    return vec4<f32>(
        out_positions.values[base],
        out_positions.values[base + 1u],
        out_positions.values[base + 2u],
        out_positions.values[base + 3u],
    );
}

fn b1_offset() -> u32 {
    return hidden_dims() * feature_dims();
}

fn w2_offset() -> u32 {
    return b1_offset() + hidden_dims();
}

fn b2_offset() -> u32 {
    return w2_offset() + output_dims() * hidden_dims();
}

fn vec4_axis(value: vec4<f32>, axis: u32) -> f32 {
    if (axis == 0u) {
        return value.x;
    }
    if (axis == 1u) {
        return value.y;
    }
    if (axis == 2u) {
        return value.z;
    }
    return value.w;
}

fn vec3i_axis(value: vec3<i32>, axis: u32) -> i32 {
    if (axis == 0u) {
        return value.x;
    }
    if (axis == 1u) {
        return value.y;
    }
    return value.z;
}

fn rem_euclid_i32(value: i32, modulus: i32) -> i32 {
    let quotient = value / modulus;
    var out = value - quotient * modulus;
    if (out < 0) {
        out = out + modulus;
    } else if (out >= modulus) {
        out = out - modulus;
    }
    return out;
}

fn mix_i32(value: i32, salt: u32) -> u32 {
    var x = bitcast<u32>(value) ^ salt;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn particle_cell_hash(coords: vec3<i32>) -> u32 {
    var hash = mix_i32(coords.x, 0x9e3779b9u) ^ mix_i32(coords.y, 0x85ebca6bu);
    if (spatial_dims() == 3u) {
        hash = hash ^ mix_i32(coords.z, 0xc2b2ae35u);
    }
    return hash % base_cell_count();
}

fn cell_coord(point: vec4<f32>, axis: u32) -> i32 {
    if (is_particle_grid()) {
        return i32(floor(vec4_axis(point, axis) / eps()));
    }

    let size = i32(grid_size_axis(axis));
    let extent = eps() * f32(size);
    let half = extent * 0.5;
    var cell = i32(floor((vec4_axis(point, axis) + half) / eps()));
    if (is_periodic()) {
        cell = rem_euclid_i32(cell, size);
    } else {
        cell = clamp(cell, 0, size - 1);
    }
    return cell;
}

fn cell_coords(point: vec4<f32>) -> vec3<i32> {
    var coords = vec3<i32>(0, 0, 0);
    coords.x = cell_coord(point, 0u);
    coords.y = cell_coord(point, 1u);
    if (spatial_dims() == 3u) {
        coords.z = cell_coord(point, 2u);
    }
    return coords;
}

fn particle_candidate_matches_cell(point: vec4<f32>, coords: vec3<i32>) -> bool {
    if (!is_particle_grid()) {
        return true;
    }
    let candidate = cell_coords(point);
    if (candidate.x != coords.x || candidate.y != coords.y) {
        return false;
    }
    if (spatial_dims() == 3u && candidate.z != coords.z) {
        return false;
    }
    return true;
}

fn same_cell_coords(lhs: vec3<i32>, rhs: vec3<i32>) -> bool {
    if (lhs.x != rhs.x || lhs.y != rhs.y) {
        return false;
    }
    if (spatial_dims() == 3u && lhs.z != rhs.z) {
        return false;
    }
    return true;
}

fn cell_index_for_support_bin(
    coords: vec3<i32>,
    particle_index: u32,
    source_bin: u32,
) -> i32 {
    if (source_bin >= support_bin_count()) {
        return -1;
    }
    let lane_offset = particle_lane(particle_index) * grid_cells_per_lane();
    let support_offset = source_bin * base_cell_count();
    if (is_particle_grid()) {
        return i32(lane_offset + support_offset + particle_cell_hash(coords));
    }

    var stride = 1u;
    var hash = 0u;
    for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
        let size = i32(grid_size_axis(axis));
        var cell = vec3i_axis(coords, axis);
        if (is_periodic()) {
            cell = rem_euclid_i32(cell, size);
        } else if (cell < 0 || cell >= size) {
            return -1;
        }
        hash = hash + u32(cell) * stride;
        stride = stride * u32(size);
    }
    return i32(lane_offset + support_offset + hash);
}

fn cell_index(coords: vec3<i32>, particle_index: u32) -> i32 {
    return cell_index_for_support_bin(coords, particle_index, 0u);
}

fn source_cell_index(coords: vec3<i32>, particle_index: u32) -> i32 {
    return cell_index_for_support_bin(
        coords,
        particle_index,
        support_bin_for_bandwidth(particle_bandwidth(particle_index)),
    );
}

fn z_min() -> i32 {
    if (spatial_dims() == 3u) {
        return -1;
    }
    return 0;
}

fn z_max() -> i32 {
    if (spatial_dims() == 3u) {
        return 1;
    }
    return 0;
}

fn neighbor_delta(lhs: vec4<f32>, rhs: vec4<f32>, axis: u32) -> f32 {
    var delta = vec4_axis(rhs, axis) - vec4_axis(lhs, axis);
    if (is_periodic()) {
        let extent = f32(grid_size_axis(axis)) * eps();
        let half = extent * 0.5;
        if (delta > half) {
            delta = delta - extent;
        } else if (delta < -half) {
            delta = delta + extent;
        }
    }
    return delta;
}

fn smoothing_poly6(r2: f32) -> f32 {
    let eps2 = eps() * eps();
    if (r2 >= eps2) {
        return 0.0;
    }
    let x = eps2 - r2;
    return smooth_coef() * x * x * x;
}

fn spiky_gradient(delta: array<f32, MAX_DIMS>, r2: f32, coeff: f32, axis: u32) -> f32 {
    let eps2 = eps() * eps();
    if (r2 <= 0.0 || r2 >= eps2) {
        return 0.0;
    }
    let r = sqrt(r2);
    let e = eps() - r;
    let mag = coeff * spiky_coef() * 3.0 * e * e / r;
    return mag * delta[axis];
}

fn recip_finite(value: f32) -> f32 {
    if (abs(value) <= 1e-20) {
        return 0.0;
    }
    return 1.0 / value;
}

fn log_normalized(value: f32, norm: f32) -> f32 {
    let stable_norm = sqrt(norm * norm + 1e-12);
    return value * log(1.0 + stable_norm) / max(stable_norm, 1e-6);
}

fn wrap_axis(value: f32, axis: u32) -> f32 {
    if (!is_periodic()) {
        return value;
    }
    let extent = f32(grid_size_axis(axis)) * eps();
    let half = extent * 0.5;
    return ((value + half) % extent + extent) % extent - half;
}

fn linked_next(index: u32) -> u32 {
    return atomicLoad(&linked_grid.values[cell_count() + index]);
}

fn grid_storage_count() -> u32 {
    if (neighbor_layout() == LAYOUT_MORTON_BVH) {
        return morton_sort_storage_count() + bvh_storage_count();
    }
    if (is_sorted_layout()) {
        let sorted_count = sorted_block_sums_base() + sorted_scan_block_count();
        if (is_bvh_layout()) {
            return sorted_count + bvh_storage_count();
        }
        return sorted_count;
    }
    let cap = bucket_capacity();
    if (cap == 0u) {
        return cell_count() + total_count() + active_grid_storage_count();
    }
    return cell_count() + cell_count() * cap + 1u + active_grid_storage_count();
}

fn active_grid_base() -> u32 {
    let cap = bucket_capacity();
    if (cap == 0u) {
        return cell_count() + total_count();
    }
    return cell_count() + cell_count() * cap + 1u;
}

fn active_grid_storage_count() -> u32 {
    return min(cell_count(), total_count());
}

fn clear_grid_count() -> u32 {
    if (bucket_capacity() == 0u || is_sorted_layout()) {
        return cell_count();
    }
    return cell_count() + 1u;
}

fn sorted_offsets_base() -> u32 {
    return cell_count();
}

fn sorted_indices_base() -> u32 {
    return sorted_offsets_base() + cell_count() + 1u;
}

fn sorted_block_sums_base() -> u32 {
    return sorted_indices_base() + total_count();
}

fn sorted_scan_block_count() -> u32 {
    return (cell_count() + SCAN_SIZE - 1u) / SCAN_SIZE;
}

fn sorted_storage_count() -> u32 {
    return sorted_block_sums_base() + sorted_scan_block_count();
}

fn sorted_offset(cell: u32) -> u32 {
    return atomicLoad(&linked_grid.values[sorted_offsets_base() + cell]);
}

fn sorted_particle(slot: u32) -> u32 {
    return atomicLoad(&linked_grid.values[sorted_indices_base() + slot]);
}

fn bvh_node_count() -> u32 {
    return atomicLoad(&linked_grid.values[bvh_storage_base()]);
}

fn bvh_index_base() -> u32 {
    return atomicLoad(&linked_grid.values[bvh_storage_base() + 1u]);
}

fn bvh_param_node_count() -> u32 {
    return bvh_leaf_count() * 2u - 1u;
}

fn bvh_param_index_base() -> u32 {
    return bvh_storage_base() + BVH_HEADER_U32 + bvh_param_node_count() * BVH_NODE_U32;
}

fn bvh_storage_base() -> u32 {
    if (neighbor_layout() == LAYOUT_SORTED_BVH) {
        return sorted_storage_count();
    }
    if (neighbor_layout() == LAYOUT_MORTON_BVH) {
        return morton_sort_storage_count();
    }
    return 0u;
}

fn bvh_storage_count() -> u32 {
    return BVH_HEADER_U32 + bvh_param_node_count() * BVH_NODE_U32 + total_count();
}

fn bvh_node_base(node: u32) -> u32 {
    return bvh_storage_base() + BVH_HEADER_U32 + node * BVH_NODE_U32;
}

fn bvh_node_min(node: u32) -> vec3<f32> {
    let base = bvh_node_base(node);
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&linked_grid.values[base])),
        bitcast<f32>(atomicLoad(&linked_grid.values[base + 1u])),
        bitcast<f32>(atomicLoad(&linked_grid.values[base + 2u])),
    );
}

fn bvh_node_max(node: u32) -> vec3<f32> {
    let base = bvh_node_base(node);
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&linked_grid.values[base + 3u])),
        bitcast<f32>(atomicLoad(&linked_grid.values[base + 4u])),
        bitcast<f32>(atomicLoad(&linked_grid.values[base + 5u])),
    );
}

fn bvh_node_left_or_start(node: u32) -> u32 {
    return atomicLoad(&linked_grid.values[bvh_node_base(node) + 6u]);
}

fn bvh_node_right_or_count(node: u32) -> u32 {
    return atomicLoad(&linked_grid.values[bvh_node_base(node) + 7u]);
}

fn bvh_node_is_leaf(node: u32) -> bool {
    return atomicLoad(&linked_grid.values[bvh_node_base(node) + 8u]) != 0u;
}

fn bvh_particle(slot: u32) -> u32 {
    return atomicLoad(&linked_grid.values[bvh_index_base() + slot]);
}

fn bvh_build_source_particle(slot: u32) -> u32 {
    if (neighbor_layout() == LAYOUT_SORTED_BVH) {
        return sorted_particle(slot);
    }
    if (neighbor_layout() == LAYOUT_MORTON_BVH) {
        return morton_sort_index(slot);
    }
    return slot;
}

fn morton_sort_keys_base() -> u32 {
    return 0u;
}

fn morton_sort_indices_base() -> u32 {
    return bvh_sort_count();
}

fn morton_sort_storage_count() -> u32 {
    return bvh_sort_count() * 2u;
}

fn morton_sort_key(slot: u32) -> u32 {
    return atomicLoad(&linked_grid.values[morton_sort_keys_base() + slot]);
}

fn morton_sort_index(slot: u32) -> u32 {
    return atomicLoad(&linked_grid.values[morton_sort_indices_base() + slot]);
}

fn morton_store(slot: u32, key: u32, particle: u32) {
    atomicStore(&linked_grid.values[morton_sort_keys_base() + slot], key);
    atomicStore(&linked_grid.values[morton_sort_indices_base() + slot], particle);
}

fn morton_expand_10(value: u32) -> u32 {
    var x = value & 1023u;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn morton2_key(coords: vec3<i32>) -> u32 {
    let x = morton_expand_10(u32(max(coords.x, 0)));
    let y = morton_expand_10(u32(max(coords.y, 0)));
    return x | (y << 1u);
}

fn morton3_key(coords: vec3<i32>) -> u32 {
    let x = morton_expand_10(u32(max(coords.x, 0)));
    let y = morton_expand_10(u32(max(coords.y, 0)));
    let z = morton_expand_10(u32(max(coords.z, 0)));
    return x | (y << 1u) | (z << 2u);
}

fn morton_key_for_particle(idx: u32) -> u32 {
    let coords = cell_coords(position(idx));
    if (spatial_dims() == 3u) {
        return morton3_key(coords);
    }
    return morton2_key(coords);
}

fn bvh_store_node(
    node: u32,
    min_v: vec3<f32>,
    max_v: vec3<f32>,
    left_or_start: u32,
    right_or_count: u32,
    leaf: bool,
) {
    let base = bvh_node_base(node);
    atomicStore(&linked_grid.values[base], bitcast<u32>(min_v.x));
    atomicStore(&linked_grid.values[base + 1u], bitcast<u32>(min_v.y));
    atomicStore(&linked_grid.values[base + 2u], bitcast<u32>(min_v.z));
    atomicStore(&linked_grid.values[base + 3u], bitcast<u32>(max_v.x));
    atomicStore(&linked_grid.values[base + 4u], bitcast<u32>(max_v.y));
    atomicStore(&linked_grid.values[base + 5u], bitcast<u32>(max_v.z));
    atomicStore(&linked_grid.values[base + 6u], left_or_start);
    atomicStore(&linked_grid.values[base + 7u], right_or_count);
    atomicStore(&linked_grid.values[base + 8u], select(0u, 1u, leaf));
}

fn bvh_intersects_eps(point: vec4<f32>, node: u32) -> bool {
    let min_v = bvh_node_min(node);
    let max_v = bvh_node_max(node);
    var r2 = 0.0;
    for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
        let p = vec4_axis(point, axis);
        let lo = vec4_axis(vec4<f32>(min_v, 0.0), axis);
        let hi = vec4_axis(vec4<f32>(max_v, 0.0), axis);
        var delta = 0.0;
        if (p < lo) {
            delta = lo - p;
        } else if (p > hi) {
            delta = p - hi;
        }
        r2 = r2 + delta * delta;
    }
    return r2 < eps() * eps();
}

fn bucket_overflow_index() -> u32 {
    return cell_count() + cell_count() * bucket_capacity();
}

fn bucket_slot_index(cell: u32, slot: u32) -> u32 {
    return cell_count() + cell * bucket_capacity() + slot;
}

fn active_cell_list_offset() -> u32 {
    return 0u;
}

fn active_cell(index: u32) -> u32 {
    return atomicLoad(&linked_grid.values[active_grid_base() + active_cell_list_offset() + index]);
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn output_state_channel(index: u32, channel: u32) -> f32 {
    let sd = state_dims();
    let state_base = index * sd;
    if (channel < sd) {
        return out_states.values[state_base + channel];
    }
    if (sd > 0u) {
        return out_states.values[state_base];
    }
    return 0.0;
}

fn output_tail_state_channel(index: u32, offset_from_end: u32) -> f32 {
    let sd = state_dims();
    if (sd > offset_from_end) {
        return output_state_channel(index, sd - 1u - offset_from_end);
    }
    return output_state_channel(index, 0u);
}

fn output_material_opacity_logit(index: u32) -> f32 {
    let sd = state_dims();
    if (sd > 8u) {
        return output_state_channel(index, 8u);
    }
    return output_state_channel(index, 3u);
}

@compute @workgroup_size(128)
fn clear_grid_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= clear_grid_count()) {
        return;
    }
    if (idx == 0u) {
        atomicStore(&indirect_args.values[0u], 0u);
        atomicStore(&indirect_args.values[1u], 1u);
        atomicStore(&indirect_args.values[2u], 1u);
    }
    if (is_sorted_layout()) {
        atomicStore(&linked_grid.values[idx], 0u);
    } else if (bucket_capacity() == 0u) {
        atomicStore(&linked_grid.values[idx], NIL);
    } else if (idx < cell_count()) {
        atomicStore(&linked_grid.values[idx], 0u);
    } else {
        atomicStore(&linked_grid.values[bucket_overflow_index()], 0u);
    }
}

@compute @workgroup_size(128)
fn bin_particles_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    let cell = u32(source_cell_index(cell_coords(position(idx)), idx));
    let cap = bucket_capacity();
    if (is_sorted_layout()) {
        atomicAdd(&linked_grid.values[cell], 1u);
    } else if (cap == 0u) {
        let old_head = atomicExchange(&linked_grid.values[cell], idx);
        atomicStore(&linked_grid.values[cell_count() + idx], old_head);
    } else {
        let slot = atomicAdd(&linked_grid.values[cell], 1u);
        if (slot < cap) {
            let occupied = slot + 1u;
            let active_blocks = (occupied + TILE_SIZE - 1u) / TILE_SIZE;
            atomicMax(&indirect_args.values[1u], active_blocks);
            if (slot == 0u) {
                let active_base = active_grid_base();
                let active_slot = atomicAdd(&indirect_args.values[0u], 1u);
                atomicStore(
                    &linked_grid.values[active_base + active_slot],
                    cell,
                );
            }
            atomicStore(&linked_grid.values[bucket_slot_index(cell, slot)], idx);
        } else {
            atomicAdd(&linked_grid.values[bucket_overflow_index()], 1u);
        }
    }
}

@compute @workgroup_size(256)
fn scan_counts_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let local_id = local.x;
    let cell = workgroup.x * SCAN_SIZE + local_id;
    var value = 0u;
    if (cell < cell_count()) {
        value = atomicLoad(&linked_grid.values[cell]);
    }
    scan_values[local_id] = value;
    workgroupBarrier();

    for (var stride = 1u; stride < SCAN_SIZE; stride = stride * 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < SCAN_SIZE) {
            scan_values[index] = scan_values[index] + scan_values[index - stride];
        }
        workgroupBarrier();
    }

    if (local_id == 0u) {
        atomicStore(&linked_grid.values[sorted_block_sums_base() + workgroup.x], scan_values[SCAN_SIZE - 1u]);
        scan_values[SCAN_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    for (var stride = SCAN_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < SCAN_SIZE) {
            let t = scan_values[index - stride];
            scan_values[index - stride] = scan_values[index];
            scan_values[index] = scan_values[index] + t;
        }
        workgroupBarrier();
    }

    if (cell < cell_count()) {
        atomicStore(&linked_grid.values[sorted_offsets_base() + cell], scan_values[local_id]);
    }
}

@compute @workgroup_size(256)
fn scan_block_sums_main(@builtin(local_invocation_id) local: vec3<u32>) {
    let local_id = local.x;
    let block_count = sorted_scan_block_count();
    var value = 0u;
    if (local_id < block_count) {
        value = atomicLoad(&linked_grid.values[sorted_block_sums_base() + local_id]);
    }
    scan_values[local_id] = value;
    workgroupBarrier();

    for (var stride = 1u; stride < SCAN_SIZE; stride = stride * 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < SCAN_SIZE) {
            scan_values[index] = scan_values[index] + scan_values[index - stride];
        }
        workgroupBarrier();
    }

    if (local_id == 0u) {
        scan_values[SCAN_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    for (var stride = SCAN_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        let index = (local_id + 1u) * stride * 2u - 1u;
        if (index < SCAN_SIZE) {
            let t = scan_values[index - stride];
            scan_values[index - stride] = scan_values[index];
            scan_values[index] = scan_values[index] + t;
        }
        workgroupBarrier();
    }

    if (local_id < block_count) {
        atomicStore(&linked_grid.values[sorted_block_sums_base() + local_id], scan_values[local_id]);
    }
}

@compute @workgroup_size(128)
fn add_block_offsets_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell == 0u) {
        atomicStore(&linked_grid.values[sorted_offsets_base() + cell_count()], total_count());
    }
    if (cell >= cell_count()) {
        return;
    }
    let block = cell / SCAN_SIZE;
    let block_offset = atomicLoad(&linked_grid.values[sorted_block_sums_base() + block]);
    let local_offset = atomicLoad(&linked_grid.values[sorted_offsets_base() + cell]);
    atomicStore(&linked_grid.values[sorted_offsets_base() + cell], local_offset + block_offset);
}

@compute @workgroup_size(128)
fn scatter_sorted_particles_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    let cell = u32(source_cell_index(cell_coords(position(idx)), idx));
    let remaining = atomicSub(&linked_grid.values[cell], 1u) - 1u;
    let base = atomicLoad(&linked_grid.values[sorted_offsets_base() + cell]);
    atomicStore(&linked_grid.values[sorted_indices_base() + base + remaining], idx);
}

@compute @workgroup_size(128)
fn morton_sort_init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bvh_sort_count()) {
        return;
    }
    if (idx < total_count()) {
        morton_store(idx, morton_key_for_particle(idx), idx);
    } else {
        morton_store(idx, 0xffffffffu, NIL);
    }
}

@compute @workgroup_size(128)
fn morton_sort_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let sort_count = bvh_sort_count();
    if (i >= sort_count) {
        return;
    }
    let j = i ^ bvh_sort_j();
    if (j <= i || j >= sort_count) {
        return;
    }

    let key_i = morton_sort_key(i);
    let key_j = morton_sort_key(j);
    let particle_i = morton_sort_index(i);
    let particle_j = morton_sort_index(j);
    let ascending = (i & bvh_sort_k()) == 0u;
    let greater = key_i > key_j || (key_i == key_j && particle_i > particle_j);
    if (greater == ascending) {
        morton_store(i, key_j, particle_j);
        morton_store(j, key_i, particle_i);
    }
}

@compute @workgroup_size(128)
fn bvh_init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let leaves = bvh_leaf_count();
    let node_count = bvh_param_node_count();
    let index_base = bvh_param_index_base();
    let header_base = bvh_storage_base();
    if (idx == 0u) {
        atomicStore(&linked_grid.values[header_base], node_count);
        atomicStore(&linked_grid.values[header_base + 1u], index_base);
        atomicStore(&linked_grid.values[header_base + 2u], leaves);
        atomicStore(&linked_grid.values[header_base + 3u], bucket_capacity());
    }
    if (idx < total_count()) {
        atomicStore(&linked_grid.values[index_base + idx], bvh_build_source_particle(idx));
    }
    if (idx >= leaves) {
        return;
    }

    let leaf_size = bucket_capacity();
    let start = idx * leaf_size;
    var count = 0u;
    if (start < total_count()) {
        count = min(leaf_size, total_count() - start);
    }
    var min_v = vec3<f32>(1.0e30, 1.0e30, 1.0e30);
    var max_v = vec3<f32>(-1.0e30, -1.0e30, -1.0e30);
    for (var local = 0u; local < count; local = local + 1u) {
        let particle = bvh_build_source_particle(start + local);
        let p = position(particle);
        min_v.x = min(min_v.x, p.x);
        min_v.y = min(min_v.y, p.y);
        max_v.x = max(max_v.x, p.x);
        max_v.y = max(max_v.y, p.y);
        if (spatial_dims() == 3u) {
            min_v.z = min(min_v.z, p.z);
            max_v.z = max(max_v.z, p.z);
        }
    }
    if (spatial_dims() == 2u) {
        min_v.z = 0.0;
        max_v.z = 0.0;
    }
    if (count == 0u) {
        min_v = vec3<f32>(1.0e30, 1.0e30, 1.0e30);
        max_v = vec3<f32>(-1.0e30, -1.0e30, -1.0e30);
    }
    bvh_store_node(leaves - 1u + idx, min_v, max_v, start, count, true);
}

@compute @workgroup_size(128)
fn bvh_reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nodes_this_level = bvh_leaf_count() >> (bvh_build_level() + 1u);
    if (idx >= nodes_this_level || nodes_this_level == 0u) {
        return;
    }
    let node = nodes_this_level - 1u + idx;
    let left = node * 2u + 1u;
    let right = left + 1u;
    let left_min = bvh_node_min(left);
    let left_max = bvh_node_max(left);
    let right_min = bvh_node_min(right);
    let right_max = bvh_node_max(right);
    let min_v = vec3<f32>(
        min(left_min.x, right_min.x),
        min(left_min.y, right_min.y),
        min(left_min.z, right_min.z),
    );
    let max_v = vec3<f32>(
        max(left_max.x, right_max.x),
        max(left_max.y, right_max.y),
        max(left_max.z, right_max.z),
    );
    bvh_store_node(node, min_v, max_v, left, right, false);
}

@compute @workgroup_size(128)
fn density_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    density.values[idx] = compute_density_particle(idx);
}

fn compute_density_particle(idx: u32) -> f32 {
    let pi = position(idx);
    let center = cell_coords(pi);
    let support_cell_radius = adaptive_support_cell_radius(idx);
    var rho = 0.0;

    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        var z_radius = 0;
        if (spatial_dims() == 3u) {
            z_radius = cell_radius;
        }
        for (var dz = -z_radius; dz <= z_radius; dz = dz + 1) {
            for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
                for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index_for_support_bin(coords, idx, source_bin);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let cap = bucket_capacity();
                if (is_sorted_layout()) {
                    let start = sorted_offset(cell_u);
                    let end = sorted_offset(cell_u + 1u);
                    for (var slot = start; slot < end; slot = slot + 1u) {
                        let j = sorted_particle(slot);
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                        if (r2 < pair_bandwidth * pair_bandwidth) {
                            rho = rho + particle_density_contribution_with_bandwidth(
                                j,
                                r2,
                                pair_bandwidth,
                            );
                        }
                    }
                } else if (cap == 0u) {
                    var j = atomicLoad(&linked_grid.values[cell_u]);
                    while (j != NIL) {
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            j = linked_next(j);
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                        if (r2 < pair_bandwidth * pair_bandwidth) {
                            rho = rho + particle_density_contribution_with_bandwidth(
                                j,
                                r2,
                                pair_bandwidth,
                            );
                        }
                        j = linked_next(j);
                    }
                } else {
                    let count = min(atomicLoad(&linked_grid.values[cell_u]), cap);
                    for (var slot = 0u; slot < count; slot = slot + 1u) {
                        let j = atomicLoad(&linked_grid.values[bucket_slot_index(cell_u, slot)]);
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                        if (r2 < pair_bandwidth * pair_bandwidth) {
                            rho = rho + particle_density_contribution_with_bandwidth(
                                j,
                                r2,
                                pair_bandwidth,
                            );
                        }
                    }
                }
            }
        }
    }
    }

    return rho;
}

fn coop_component_index(component: u32, lane: u32) -> u32 {
    return component * COOP_SIZE + lane;
}

fn cache_coop_reduced_values(local_id: u32) {
    for (
        var component = local_id;
        component < COOP_COMPONENTS;
        component = component + COOP_SIZE
    ) {
        coop_reduced_values[component] = coop_values[coop_component_index(component, 0u)];
    }
    workgroupBarrier();
}

@compute @workgroup_size(32)
fn cooperative_density_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = cooperative_particle_index(workgroup);
    let local_id = local.x;
    if (idx >= total_count()) {
        return;
    }

    let pi = position(idx);
    let center = cell_coords(pi);
    var rho = 0.0;
    var coarse_rho = 0.0;

    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        var z_radius = 0;
        if (spatial_dims() == 3u) {
            z_radius = cell_radius;
        }
        for (var dz = -z_radius; dz <= z_radius; dz = dz + 1) {
            for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
                for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index_for_support_bin(coords, idx, source_bin);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let start = sorted_offset(cell_u);
                let end = sorted_offset(cell_u + 1u);
                for (var slot = start + local_id; slot < end; slot = slot + COOP_SIZE) {
                    let j = sorted_particle(slot);
                    let pj = position(j);
                    if (!particle_candidate_matches_cell(pj, coords)) {
                        continue;
                    }
                    var r2 = 0.0;
                    for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                        let delta = neighbor_delta(pi, pj, axis);
                        r2 = r2 + delta * delta;
                    }
                    let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                    if (r2 < pair_bandwidth * pair_bandwidth) {
                        let contribution = particle_density_contribution_with_bandwidth(
                            j,
                            r2,
                            pair_bandwidth,
                        );
                        rho = rho + contribution;
                        if (is_coarse_density_source(j)) {
                            coarse_rho = coarse_rho + contribution;
                        }
                    }
                }
            }
        }
    }
    }

    coop_values[local_id] = rho;
    coop_values[COOP_SIZE + local_id] = coarse_rho;
    workgroupBarrier();
    for (var stride = COOP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            coop_values[local_id] = coop_values[local_id] + coop_values[local_id + stride];
            coop_values[COOP_SIZE + local_id] =
                coop_values[COOP_SIZE + local_id] + coop_values[COOP_SIZE + local_id + stride];
        }
        workgroupBarrier();
    }
    if (local_id == 0u) {
        density.values[idx] = coop_values[0u];
        density.values[diagnostics_coarse_exposure_offset() + idx] = clamp(
            coop_values[COOP_SIZE] / max(coop_values[0u], 1.0e-20),
            0.0,
            1.0,
        );
    }
}

@compute @workgroup_size(128)
fn bvh_density_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    density.values[idx] = compute_density_bvh_particle(idx);
}

fn compute_density_bvh_particle(idx: u32) -> f32 {
    let pi = position(idx);
    let eps2 = eps() * eps();
    var rho = 0.0;
    var stack: array<u32, BVH_STACK_SIZE>;
    var stack_len = 0u;
    if (bvh_node_count() == 0u) {
        return 0.0;
    }
    stack[stack_len] = 0u;
    stack_len = stack_len + 1u;

    while (stack_len > 0u) {
        stack_len = stack_len - 1u;
        let node = stack[stack_len];
        if (!bvh_intersects_eps(pi, node)) {
            continue;
        }
        if (bvh_node_is_leaf(node)) {
            let start = bvh_node_left_or_start(node);
            let count = bvh_node_right_or_count(node);
            for (var slot = 0u; slot < count; slot = slot + 1u) {
                let j = bvh_particle(start + slot);
                let pj = position(j);
                var delta: array<f32, MAX_DIMS>;
                var r2 = 0.0;
                for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                    delta[axis] = neighbor_delta(pi, pj, axis);
                    r2 = r2 + delta[axis] * delta[axis];
                }
                if (r2 < eps2) {
                    rho = rho + particle_density_contribution(j, r2);
                }
            }
        } else {
            let left = bvh_node_left_or_start(node);
            let right = bvh_node_right_or_count(node);
            if (stack_len + 2u <= BVH_STACK_SIZE) {
                stack[stack_len] = left;
                stack[stack_len + 1u] = right;
                stack_len = stack_len + 2u;
            }
        }
    }
    return rho;
}

@compute @workgroup_size(128)
fn tiled_density_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    if (bucket_capacity() == 0u) {
        return;
    }

    let local_id = local.x;
    let block_start = workgroup.y * TILE_SIZE;
    if (local_id == 0u) {
        let loaded_target_cell = active_cell(workgroup.x);
        let loaded_target_count = min(
            atomicLoad(&linked_grid.values[loaded_target_cell]),
            bucket_capacity(),
        );
        var reference_index = 0u;
        tile_center = vec3<i32>(0, 0, 0);
        if (block_start < loaded_target_count) {
            reference_index = atomicLoad(
                &linked_grid.values[bucket_slot_index(loaded_target_cell, block_start)],
            );
            tile_center = cell_coords(position(reference_index));
        }
        tile_dispatch = vec4<u32>(
            loaded_target_cell,
            loaded_target_count,
            reference_index,
            0u,
        );
        atomicStore(&tile_mismatch, 0u);
    }
    let dispatch = workgroupUniformLoad(&tile_dispatch);
    let common_center = workgroupUniformLoad(&tile_center);
    let target_cell = dispatch.x;
    let target_count = dispatch.y;
    let reference_index = dispatch.z;
    if (block_start >= target_count) {
        return;
    }

    let target_slot = block_start + local_id;
    let has_target = target_slot < target_count;
    var idx = 0u;
    var pi = vec4<f32>(0.0);
    var center = vec3<i32>(0, 0, 0);
    if (has_target) {
        idx = atomicLoad(&linked_grid.values[bucket_slot_index(target_cell, target_slot)]);
        pi = position(idx);
        center = cell_coords(pi);
    }

    if (has_target && !same_cell_coords(center, common_center)) {
        atomicStore(&tile_mismatch, 1u);
    }
    let mismatch = workgroupUniformLoad(&tile_mismatch);
    if (mismatch != 0u) {
        if (has_target) {
            density.values[idx] = compute_density_particle(idx);
        }
        return;
    }

    let eps2 = eps() * eps();
    var rho = 0.0;

    for (var dz = z_min(); dz <= z_max(); dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coords = vec3<i32>(
                    common_center.x + dx,
                    common_center.y + dy,
                    common_center.z + dz,
                );
                if (local_id == 0u) {
                    let cell = cell_index(coords, reference_index);
                    var cell_u = NIL;
                    var count = 0u;
                    if (cell >= 0) {
                        cell_u = u32(cell);
                        count = min(
                            atomicLoad(&linked_grid.values[cell_u]),
                            bucket_capacity(),
                        );
                    }
                    tile_neighbor = vec2<u32>(cell_u, count);
                }
                let neighbor = workgroupUniformLoad(&tile_neighbor);
                let cell_u = neighbor.x;
                let count = neighbor.y;
                if (cell_u == NIL) {
                    continue;
                }
                for (var chunk = 0u; chunk < count; chunk = chunk + TILE_SIZE) {
                    let load_slot = chunk + local_id;
                    if (load_slot < count) {
                        let j = atomicLoad(&linked_grid.values[bucket_slot_index(cell_u, load_slot)]);
                        tile_indices[local_id] = j;
                        tile_positions[local_id] = position(j);
                    } else {
                        tile_indices[local_id] = NIL;
                        tile_positions[local_id] = vec4<f32>(0.0);
                    }
                    workgroupBarrier();

                    let chunk_count = min(TILE_SIZE, count - chunk);
                    if (has_target) {
                        for (var k = 0u; k < chunk_count; k = k + 1u) {
                            let j = tile_indices[k];
                            if (j == NIL) {
                                continue;
                            }
                            let pj = tile_positions[k];
                            if (!particle_candidate_matches_cell(pj, coords)) {
                                continue;
                            }
                            var delta: array<f32, MAX_DIMS>;
                            var r2 = 0.0;
                            for (var axis = 0u; axis < spatial_dims(); axis = axis + 1u) {
                                delta[axis] = neighbor_delta(pi, pj, axis);
                                r2 = r2 + delta[axis] * delta[axis];
                            }
                            if (r2 < eps2) {
                                rho = rho + particle_density_contribution(j, r2);
                            }
                        }
                    }
                    workgroupBarrier();
                }
            }
        }
    }

    if (has_target) {
        density.values[idx] = rho;
    }
}

@compute @workgroup_size(128)
fn adaptive_local_residual_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let start_hidden = adaptive_local_hidden_start();
    if (idx >= total_count() || !adaptive_local_rule_enabled()) {
        return;
    }
    let sd = state_dims();
    let dim = spatial_dims();
    let fd = feature_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;
    if (update_mask(idx) == 0.0) {
        for (var axis = 0u; axis < 4u; axis = axis + 1u) {
            out_positions.values[position_base + axis] = 0.0;
        }
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            out_states.values[state_base + channel] = 0.0;
        }
        return;
    }

    let pi = position(idx);
    let center = cell_coords(pi);
    let eps2 = eps() * eps();
    let shepard = adaptive_shepard_epsilon();
    var shepard_sum = shepard;
    var normalized_state: array<f32, MAX_STATE_DIMS>;
    var closure_blur: array<f32, MAX_CLOSURE_CONTEXT_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var occupancy_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        normalized_state[channel] = shepard * states.values[state_base + channel];
    }
    if (adaptive_closure_enabled()) {
        for (var component = 0u; component < sd + 6u; component = component + 1u) {
            closure_blur[component] = shepard * material_closure_context(idx, component);
        }
    }

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index(coords, idx);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin; slot < end; slot = slot + 1u) {
                let j = sorted_particle(slot);
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                var delta: array<f32, MAX_DIMS>;
                var r2 = 0.0;
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    delta[axis] = neighbor_delta(pi, pj, axis);
                    r2 = r2 + delta[axis] * delta[axis];
                }
                if (r2 >= eps2) {
                    continue;
                }
                let measure_j = particle_measure(j);
                let weight = measure_j * adaptive_kernel_value(r2);
                shepard_sum = shepard_sum + weight;
                let source = j * sd;
                for (var channel = 0u; channel < sd; channel = channel + 1u) {
                    normalized_state[channel] = normalized_state[channel]
                        + weight * states.values[source + channel];
                }
                if (adaptive_closure_enabled()) {
                    for (
                        var component = 0u;
                        component < sd + 6u;
                        component = component + 1u
                    ) {
                        closure_blur[component] = closure_blur[component]
                            + weight * material_closure_context(j, component);
                    }
                }
                if (j == idx || r2 <= 0.0) {
                    continue;
                }
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    let gradient = measure_j * adaptive_kernel_gradient(delta, r2, axis);
                    occupancy_grad[axis] = occupancy_grad[axis] + gradient;
                    for (var col = 0u; col < dim; col = col + 1u) {
                        moment[axis * MAX_DIMS + col] = moment[axis * MAX_DIMS + col]
                            + gradient * delta[col];
                    }
                    for (var channel = 0u; channel < sd; channel = channel + 1u) {
                        let difference = states.values[source + channel]
                            - states.values[state_base + channel];
                        state_grad[channel * MAX_DIMS + axis] =
                            state_grad[channel * MAX_DIMS + axis] + difference * gradient;
                    }
                }
            }
        }
    }

    let partition_inverse = 1.0 / max(shepard_sum, 1.0e-20);
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        normalized_state[channel] = normalized_state[channel] * partition_inverse;
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        occupancy_grad[axis] = occupancy_grad[axis] * partition_inverse;
    }

    var m00 = moment[0u];
    var m01 = moment[1u];
    var m10 = moment[MAX_DIMS];
    var m11 = moment[MAX_DIMS + 1u];
    let trace = abs(m00) + abs(m11);
    let diagonal = adaptive_moment_regularization() * max(0.5 * trace, 1.0e-8);
    m00 = m00 + select(-diagonal, diagonal, m00 >= 0.0);
    m11 = m11 + select(-diagonal, diagonal, m11 >= 0.0);
    let determinant = m00 * m11 - m01 * m10;
    var inverse = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    if (abs(determinant) >= 1.0e-12) {
        let reciprocal = 1.0 / determinant;
        inverse[0] = m11 * reciprocal;
        inverse[1] = -m01 * reciprocal;
        inverse[2] = -m10 * reciprocal;
        inverse[3] = m00 * reciprocal;
    }
    let matrix_norm = sqrt(m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11);
    let inverse_norm = sqrt(
        inverse[0] * inverse[0] + inverse[1] * inverse[1]
            + inverse[2] * inverse[2] + inverse[3] * inverse[3],
    );
    let condition = matrix_norm * inverse_norm;
    if (!(condition <= adaptive_moment_condition_limit()) || abs(determinant) < 1.0e-12) {
        let fallback = 1.0 / max(0.5 * trace, 1.0e-6);
        inverse = array<f32, 4>(fallback, 0.0, 0.0, fallback);
    }

    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        let gx = state_grad[channel * MAX_DIMS];
        let gy = state_grad[channel * MAX_DIMS + 1u];
        state_grad[channel * MAX_DIMS] = inverse[0] * gx + inverse[1] * gy;
        state_grad[channel * MAX_DIMS + 1u] = inverse[2] * gx + inverse[3] * gy;
        var norm = 0.0;
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            let value = state_grad[channel * MAX_DIMS + axis] * eps();
            state_grad[channel * MAX_DIMS + axis] = value;
            norm = norm + value * value;
        }
        norm = sqrt(norm);
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            state_grad[channel * MAX_DIMS + axis] =
                log_normalized(state_grad[channel * MAX_DIMS + axis], norm);
        }
    }
    var occupancy_norm = 0.0;
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        occupancy_grad[axis] = occupancy_grad[axis] * eps();
        occupancy_norm = occupancy_norm + occupancy_grad[axis] * occupancy_grad[axis];
    }
    occupancy_norm = sqrt(occupancy_norm);
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        occupancy_grad[axis] = log_normalized(occupancy_grad[axis], occupancy_norm);
    }

    var feature: array<f32, MAX_FEATURE_DIMS>;
    var cursor = 0u;
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        feature[cursor] = states.values[state_base + channel];
        cursor = cursor + 1u;
    }
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        feature[cursor] = normalized_state[channel];
        cursor = cursor + 1u;
    }
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            feature[cursor] = state_grad[channel * MAX_DIMS + axis];
            cursor = cursor + 1u;
        }
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        feature[cursor] = occupancy_grad[axis];
        cursor = cursor + 1u;
    }
    if (has_position_features()) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            feature[cursor] = vec4_axis(pi, axis);
            cursor = cursor + 1u;
        }
    }
    if (material_scale_conditioning_enabled(cursor)) {
        feature[cursor] = particle_material_scale_feature(idx);
        cursor = cursor + 1u;
    } else if (residual_material_features_enabled(cursor)) {
        feature[cursor] = particle_material_scale_feature(idx);
        feature[cursor + 1u] = clamp(
            density.values[diagnostics_coarse_exposure_offset() + idx],
            0.0,
            1.0,
        );
        cursor = cursor + 2u;
    }
    for (var input = cursor; input < fd; input = input + 1u) {
        feature[input] = 0.0;
    }
    let closure_dims = 1u + dim * (dim + 1u) / 2u + sd * dim;
    if (fd >= cursor + closure_dims) {
        let measure = max(particle_measure(idx), 1.0e-20);
        var footprint = sqrt(measure / 3.141592653589793);
        if (dim == 3u) {
            footprint = pow(measure / (4.0 * 3.141592653589793 / 3.0), 1.0 / 3.0);
        }
        feature[cursor] = clamp(
            log2(footprint / max(adaptive_reference_footprint(), 1.0e-20)),
            -3.0,
            3.0,
        );
        cursor = cursor + 1u;
        let footprint2 = max(footprint * footprint, 1.0e-20);
        for (var lhs = 0u; lhs < dim; lhs = lhs + 1u) {
            for (var rhs = lhs; rhs < dim; rhs = rhs + 1u) {
                feature[cursor] = clamp(
                    material_covariance_value(idx, lhs * 3u + rhs) / footprint2,
                    -8.0,
                    8.0,
                );
                cursor = cursor + 1u;
            }
        }
        for (var component = 0u; component < sd * dim; component = component + 1u) {
            let scaled = material_state_jacobian(idx, component) * footprint;
            feature[cursor] = sign(scaled) * min(log(1.0 + abs(scaled)), 8.0);
            cursor = cursor + 1u;
        }
        if (adaptive_closure_enabled() && fd >= cursor + 6u + sd) {
            for (var component = 0u; component < 4u; component = component + 1u) {
                feature[cursor + component] = material_closure_basis(idx, component);
            }
            cursor = cursor + 4u;
            for (var component = 0u; component < 2u; component = component + 1u) {
                feature[cursor + component] = material_closure_phase(idx, component);
            }
            cursor = cursor + 2u;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                feature[cursor + channel] = material_closure_mode(idx, channel);
            }
            cursor = cursor + sd;
            if (fd >= cursor + 6u + sd) {
                for (
                    var component = 0u;
                    component < 6u + sd;
                    component = component + 1u
                ) {
                    feature[cursor + component] =
                        closure_blur[component] * partition_inverse;
                }
                cursor = cursor + 6u + sd;
            }
        }
    }

    var hidden: array<f32, MAX_HIDDEN_DIMS>;
    let hd = hidden_dims();
    for (var h = start_hidden; h < hd; h = h + 1u) {
        var sum = weights.values[b1_offset() + h];
        let weight_base = h * fd;
        for (var input = 0u; input < fd; input = input + 1u) {
            sum = sum + weights.values[weight_base + input] * feature[input];
        }
        hidden[h] = max(sum, 0.0);
    }
    let gate = adaptive_local_rule_gate(idx);
    let od = output_dims();
    for (var output = 0u; output < od; output = output + 1u) {
        var value = select(
            0.0,
            weights.values[b2_offset() + output],
            adaptive_normalized_primary_enabled(),
        );
        let weight_base = w2_offset() + output * hd;
        for (var h = start_hidden; h < hd; h = h + 1u) {
            value = value + weights.values[weight_base + h] * hidden[h];
        }
        value = value * gate;
        if (output < dim) {
            out_positions.values[position_base + output] = value;
        } else {
            out_states.values[state_base + output - dim] = value;
        }
    }
    if (adaptive_closure_enabled() && is_coarse_material(idx)) {
        let closure_hidden_dims = adaptive_closure_hidden_dims();
        for (var h = 0u; h < closure_hidden_dims; h = h + 1u) {
            var sum = weights.values[closure_b1_offset() + h];
            let weight_base = closure_weight_base() + h * fd;
            for (var input = 0u; input < fd; input = input + 1u) {
                sum = sum + weights.values[weight_base + input] * feature[input];
            }
            hidden[h] = max(sum, 0.0);
        }
        let mask_dt = update_mask(idx) * dt();
        var next_phase: vec2<f32>;
        for (var axis = 0u; axis < 2u; axis = axis + 1u) {
            var value = weights.values[closure_b2_offset() + axis];
            let weight_base = closure_w2_offset() + axis * closure_hidden_dims;
            for (var h = 0u; h < closure_hidden_dims; h = h + 1u) {
                value = value + weights.values[weight_base + h] * hidden[h];
            }
            let previous = material_closure_phase(idx, axis);
            if (axis == 0u) {
                next_phase.x = previous + mask_dt * value;
            } else {
                next_phase.y = previous + mask_dt * value;
            }
        }
        let phase_norm = length(next_phase);
        let normalized_phase = next_phase / max(phase_norm, 1.0e-6);
        next_phase = select(vec2<f32>(1.0, 0.0), normalized_phase, phase_norm > 1.0e-6);
        set_material_closure_phase(idx, 0u, next_phase.x);
        set_material_closure_phase(idx, 1u, next_phase.y);
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            let output = dim + channel;
            var value = weights.values[closure_b2_offset() + output];
            let weight_base = closure_w2_offset() + output * closure_hidden_dims;
            for (var h = 0u; h < closure_hidden_dims; h = h + 1u) {
                value = value + weights.values[weight_base + h] * hidden[h];
            }
            add_material_closure_mode(idx, channel, mask_dt * value);
        }
    }
    if (adaptive_closure_basis_enabled() && is_coarse_material(idx)) {
        let basis_hidden_dims = adaptive_closure_basis_hidden_dims();
        for (var h = 0u; h < basis_hidden_dims; h = h + 1u) {
            var sum = weights.values[closure_basis_rule_b1_offset() + h];
            let weight_base = closure_basis_rule_weight_base() + h * fd;
            for (var input = 0u; input < fd; input = input + 1u) {
                sum = sum + weights.values[weight_base + input] * feature[input];
            }
            hidden[h] = max(sum, 0.0);
        }
        let mask_dt = update_mask(idx) * dt();
        let previous_basis = vec4<f32>(
            material_closure_basis(idx, 0u),
            material_closure_basis(idx, 1u),
            material_closure_basis(idx, 2u),
            material_closure_basis(idx, 3u),
        );
        var next_basis = previous_basis;
        for (var component = 0u; component < 4u; component = component + 1u) {
            var value = weights.values[closure_basis_rule_b2_offset() + component];
            let weight_base =
                closure_basis_rule_w2_offset() + component * basis_hidden_dims;
            for (var h = 0u; h < basis_hidden_dims; h = h + 1u) {
                value = value + weights.values[weight_base + h] * hidden[h];
            }
            next_basis[component] = next_basis[component] + mask_dt * value;
        }
        next_basis = next_basis - vec4<f32>(
            dot(next_basis, vec4<f32>(0.25, 0.25, 0.25, 0.25)),
        );
        let basis_norm = length(next_basis);
        next_basis = select(previous_basis, next_basis / max(basis_norm, 1.0e-6), basis_norm > 1.0e-6);
        next_basis = select(next_basis, -next_basis, dot(next_basis, previous_basis) < 0.0);
        for (var component = 0u; component < 4u; component = component + 1u) {
            set_material_closure_basis(idx, component, next_basis[component]);
        }
    }
}

fn adaptive_radix_prefix_matches(key: u32, prefix: u32, shift: u32) -> bool {
    if (shift == 32u - ADAPTIVE_RADIX_BITS) {
        return true;
    }
    let mask = 0xffffffffu << (shift + ADAPTIVE_RADIX_BITS);
    return (key & mask) == (prefix & mask);
}

fn adaptive_count_support_candidates(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    _cell_radius: i32,
) {
    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                if (j == idx) {
                    continue;
                }
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                let delta_x = neighbor_delta(pi, pj, 0u);
                let delta_y = neighbor_delta(pi, pj, 1u);
                let r2 = delta_x * delta_x + delta_y * delta_y;
                if (adaptive_pair_normalized_distance2(idx, j, r2) < 1.0) {
                    atomicAdd(&adaptive_support_count, 1u);
                }
            }
        }
    }
    }
}

fn adaptive_spacing_occupancy_cooperative(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    radius: f32,
) -> f32 {
    let radius2 = radius * radius;
    let cell_radius = max(i32(ceil(radius / eps())), 1);
    var occupancy = 0.0;
    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                if (j == idx) {
                    continue;
                }
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                let delta_x = neighbor_delta(pi, pj, 0u);
                let delta_y = neighbor_delta(pi, pj, 1u);
                let r2 = delta_x * delta_x + delta_y * delta_y;
                if (r2 < radius2) {
                    let q2 = r2 / radius2;
                    let shoulder = 1.0 - q2;
                    occupancy = occupancy + shoulder * shoulder * shoulder;
                }
            }
        }
    }
    }
    coop_values[local_id] = occupancy;
    workgroupBarrier();
    for (var stride = COOP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            coop_values[local_id] = coop_values[local_id] + coop_values[local_id + stride];
        }
        workgroupBarrier();
    }
    return workgroupUniformLoad(&coop_values[0u]);
}

fn adaptive_histogram_distance(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    _cell_radius: i32,
    prefix: u32,
    shift: u32,
) {
    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                if (j == idx) {
                    continue;
                }
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                let delta_x = neighbor_delta(pi, pj, 0u);
                let delta_y = neighbor_delta(pi, pj, 1u);
                let r2 = delta_x * delta_x + delta_y * delta_y;
                let normalized_distance2 = adaptive_pair_normalized_distance2(idx, j, r2);
                if (normalized_distance2 >= 1.0) {
                    continue;
                }
                let key = bitcast<u32>(normalized_distance2);
                if (adaptive_radix_prefix_matches(key, prefix, shift)) {
                    atomicAdd(
                        &adaptive_radix_counts[(key >> shift) & (ADAPTIVE_RADIX_BINS - 1u)],
                        1u,
                    );
                }
            }
        }
    }
    }
}

fn adaptive_histogram_index(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    _cell_radius: i32,
    distance_key: u32,
    prefix: u32,
    shift: u32,
) {
    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                if (j == idx) {
                    continue;
                }
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                let delta_x = neighbor_delta(pi, pj, 0u);
                let delta_y = neighbor_delta(pi, pj, 1u);
                let r2 = delta_x * delta_x + delta_y * delta_y;
                let normalized_distance2 = adaptive_pair_normalized_distance2(idx, j, r2);
                if (normalized_distance2 >= 1.0
                    || bitcast<u32>(normalized_distance2) != distance_key) {
                    continue;
                }
                if (adaptive_radix_prefix_matches(j, prefix, shift)) {
                    atomicAdd(
                        &adaptive_radix_counts[(j >> shift) & (ADAPTIVE_RADIX_BINS - 1u)],
                        1u,
                    );
                }
            }
        }
    }
    }
}

fn adaptive_find_min_cutoff_index(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    _cell_radius: i32,
    distance_key: u32,
) {
    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                if (j == idx) {
                    continue;
                }
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                let delta_x = neighbor_delta(pi, pj, 0u);
                let delta_y = neighbor_delta(pi, pj, 1u);
                let r2 = delta_x * delta_x + delta_y * delta_y;
                let normalized_distance2 = adaptive_pair_normalized_distance2(idx, j, r2);
                if (normalized_distance2 < 1.0
                    && bitcast<u32>(normalized_distance2) == distance_key) {
                    atomicMin(&adaptive_min_cutoff_index, j);
                }
            }
        }
    }
    }
}

fn adaptive_select_radix_bin(local_id: u32, shift: u32) {
    workgroupBarrier();
    if (local_id == 0u) {
        var count_before = 0u;
        var selected_bin = 0u;
        for (var bin = 0u; bin < ADAPTIVE_RADIX_BINS; bin = bin + 1u) {
            let count = atomicLoad(&adaptive_radix_counts[bin]);
            if (adaptive_selection_rank < count_before + count) {
                selected_bin = bin;
                break;
            }
            count_before = count_before + count;
        }
        adaptive_selection_prefix = adaptive_selection_prefix | (selected_bin << shift);
        adaptive_selection_rank = adaptive_selection_rank - count_before;
    }
    workgroupBarrier();
}

fn finish_adaptive_local_residual_cooperative(
    local_id: u32,
    idx: u32,
    start_hidden: u32,
    observed_spacing: f32,
    support_count: u32,
    max_neighbors: u32,
    pi: vec4<f32>,
) {
    let sd = state_dims();
    let dim = spatial_dims();
    let fd = feature_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;
    if (local_id == 0u) {
        density.values[idx] = coop_reduced_values[COOP_MOMENT_BASE + 8u];
        let footprint = sqrt(max(particle_measure(idx), 1.0e-20) / 3.141592653589793);
        var retained_norm_squared = 0.0;
        for (var component = 0u; component < sd * dim; component = component + 1u) {
            let value = material_state_jacobian(idx, component);
            retained_norm_squared = retained_norm_squared + value * value;
        }
        adaptive_retained_jacobian_enabled = select(
            0u,
            1u,
            footprint > 1.5 * adaptive_base_footprint()
                && retained_norm_squared > 1.0e-12,
        );
    }
    workgroupBarrier();
    let inverse_partition = 1.0 / max(
        coop_reduced_values[COOP_DENSITY_GRAD_BASE + 2u],
        1.0e-20,
    );
    var m00 = coop_reduced_values[COOP_MOMENT_BASE];
    let m01 = coop_reduced_values[COOP_MOMENT_BASE + 1u];
    let m10 = coop_reduced_values[COOP_MOMENT_BASE + MAX_DIMS];
    var m11 = coop_reduced_values[COOP_MOMENT_BASE + MAX_DIMS + 1u];
    let trace = abs(m00) + abs(m11);
    let diagonal = adaptive_moment_regularization() * max(0.5 * trace, 1.0e-8);
    m00 = m00 + select(-diagonal, diagonal, m00 >= 0.0);
    m11 = m11 + select(-diagonal, diagonal, m11 >= 0.0);
    let determinant = m00 * m11 - m01 * m10;
    var inverse = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    if (abs(determinant) >= 1.0e-12) {
        let reciprocal = 1.0 / determinant;
        inverse[0] = m11 * reciprocal;
        inverse[1] = -m01 * reciprocal;
        inverse[2] = -m10 * reciprocal;
        inverse[3] = m00 * reciprocal;
    }
    let matrix_norm = sqrt(m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11);
    let inverse_norm = sqrt(
        inverse[0] * inverse[0] + inverse[1] * inverse[1]
            + inverse[2] * inverse[2] + inverse[3] * inverse[3],
    );
    let condition = matrix_norm * inverse_norm;
    if (!(condition <= adaptive_moment_condition_limit()) || abs(determinant) < 1.0e-12) {
        let fallback = 1.0 / max(0.5 * trace, 1.0e-6);
        inverse = array<f32, 4>(fallback, 0.0, 0.0, fallback);
    }

    for (var channel = local_id; channel < sd; channel = channel + COOP_SIZE) {
        coop_feature[channel] = states.values[state_base + channel];
        coop_feature[sd + channel] =
            coop_reduced_values[COOP_BLUR_BASE + channel] * inverse_partition;
        let gx = coop_reduced_values[COOP_STATE_GRAD_BASE + channel * MAX_DIMS];
        let gy = coop_reduced_values[COOP_STATE_GRAD_BASE + channel * MAX_DIMS + 1u];
        let target_bandwidth = particle_bandwidth(idx);
        var corrected_x = target_bandwidth * (inverse[0] * gx + inverse[1] * gy);
        var corrected_y = target_bandwidth * (inverse[2] * gx + inverse[3] * gy);
        if (adaptive_retained_jacobian_enabled != 0u) {
            corrected_x = target_bandwidth
                * material_state_jacobian(idx, channel * dim);
            corrected_y = target_bandwidth
                * material_state_jacobian(idx, channel * dim + 1u);
        }
        let norm = sqrt(corrected_x * corrected_x + corrected_y * corrected_y);
        let gradient_base = 2u * sd + channel * dim;
        coop_feature[gradient_base] = select(
            corrected_x,
            log_normalized(corrected_x, norm),
            pu(12u) != 0u,
        );
        coop_feature[gradient_base + 1u] = select(
            corrected_y,
            log_normalized(corrected_y, norm),
            pu(12u) != 0u,
        );
    }
    if (local_id < dim) {
        let target_bandwidth = particle_bandwidth(idx);
        let occupancy_x = target_bandwidth * inverse_partition
            * coop_reduced_values[COOP_DENSITY_GRAD_BASE];
        let occupancy_y = target_bandwidth * inverse_partition
            * coop_reduced_values[COOP_DENSITY_GRAD_BASE + 1u];
        let norm = sqrt(occupancy_x * occupancy_x + occupancy_y * occupancy_y);
        let value = select(occupancy_x, occupancy_y, local_id == 1u);
        let occupancy_base = 2u * sd + sd * dim;
        coop_feature[occupancy_base + local_id] = select(
            value,
            log_normalized(value, norm),
            pu(12u) != 0u,
        );
    }
    if (has_position_features() && local_id < dim) {
        let position_base_feature = 2u * sd + sd * dim + dim;
        coop_feature[position_base_feature + local_id] = vec4_axis(pi, local_id);
    }
    if (local_id == 0u) {
        var cursor = 2u * sd + sd * dim + dim;
        if (has_position_features()) {
            cursor = cursor + dim;
        }
        if (material_scale_conditioning_enabled(cursor)) {
            coop_feature[cursor] = particle_material_scale_feature(idx);
            cursor = cursor + 1u;
        } else if (residual_material_features_enabled(cursor)) {
            coop_feature[cursor] = particle_material_scale_feature(idx);
            coop_feature[cursor + 1u] = clamp(
                density.values[diagnostics_coarse_exposure_offset() + idx],
                0.0,
                1.0,
            );
            cursor = cursor + 2u;
        }
        for (var input = cursor; input < fd; input = input + 1u) {
            coop_feature[input] = 0.0;
        }
        let closure_dims = 1u + dim * (dim + 1u) / 2u + sd * dim;
        if (fd >= cursor + closure_dims) {
            let measure = max(particle_measure(idx), 1.0e-20);
            let footprint = sqrt(measure / 3.141592653589793);
            coop_feature[cursor] = clamp(
                log2(footprint / max(adaptive_reference_footprint(), 1.0e-20)),
                -3.0,
                3.0,
            );
            cursor = cursor + 1u;
            let footprint2 = max(footprint * footprint, 1.0e-20);
            for (var lhs = 0u; lhs < dim; lhs = lhs + 1u) {
                for (var rhs = lhs; rhs < dim; rhs = rhs + 1u) {
                    coop_feature[cursor] = clamp(
                        material_covariance_value(idx, lhs * 3u + rhs) / footprint2,
                        -8.0,
                        8.0,
                    );
                    cursor = cursor + 1u;
                }
            }
            for (var component = 0u; component < sd * dim; component = component + 1u) {
                let scaled = material_state_jacobian(idx, component) * footprint;
                coop_feature[cursor] = sign(scaled) * min(log(1.0 + abs(scaled)), 8.0);
                cursor = cursor + 1u;
            }
            if (adaptive_closure_enabled() && fd >= cursor + 6u + sd) {
                for (var component = 0u; component < 4u; component = component + 1u) {
                    coop_feature[cursor + component] = material_closure_basis(idx, component);
                }
                cursor = cursor + 4u;
                for (var component = 0u; component < 2u; component = component + 1u) {
                    coop_feature[cursor + component] = material_closure_phase(idx, component);
                }
                cursor = cursor + 2u;
                for (var channel = 0u; channel < sd; channel = channel + 1u) {
                    coop_feature[cursor + channel] = material_closure_mode(idx, channel);
                }
                cursor = cursor + sd;
                if (fd >= cursor + 6u + sd) {
                    for (
                        var component = 0u;
                        component < 6u + sd;
                        component = component + 1u
                    ) {
                        coop_feature[cursor + component] =
                            coop_reduced_values[COOP_CLOSURE_BLUR_BASE + component]
                                * inverse_partition;
                    }
                    cursor = cursor + 6u + sd;
                }
            }
        }
    }
    workgroupBarrier();

    if (adaptive_diagnostics_enabled()) {
        for (var feature = local_id; feature < fd; feature = feature + COOP_SIZE) {
            density.values[
                diagnostics_normalized_feature_offset() + idx * fd + feature
            ] = coop_feature[feature];
        }
        if (local_id == 0u) {
            density.values[diagnostics_spacing_offset() + idx] = observed_spacing;
            let accepted = select(
                support_count,
                min(support_count, max_neighbors),
                max_neighbors > 0u,
            );
            density.values[diagnostics_degree_offset() + idx] = f32(accepted);
        }
    }
    workgroupBarrier();
    if (adaptive_diagnostics_enabled() && pu(99u) != 0u) {
        return;
    }

    let hd = hidden_dims();
    for (var h = start_hidden + local_id; h < hd; h = h + COOP_SIZE) {
        var sum = weights.values[b1_offset() + h];
        let weight_base = h * fd;
        for (var input = 0u; input < fd; input = input + 1u) {
            sum = sum + weights.values[weight_base + input] * coop_feature[input];
        }
        coop_hidden[h] = max(sum, 0.0);
    }
    workgroupBarrier();
    let gate = adaptive_local_rule_gate(idx);
    let od = output_dims();
    for (var output = local_id; output < od; output = output + COOP_SIZE) {
        var value = select(
            0.0,
            weights.values[b2_offset() + output],
            adaptive_normalized_primary_enabled(),
        );
        let weight_base = w2_offset() + output * hd;
        for (var h = start_hidden; h < hd; h = h + 1u) {
            value = value + weights.values[weight_base + h] * coop_hidden[h];
        }
        value = value * gate;
        if (output < dim) {
            out_positions.values[position_base + output] = value;
        } else {
            out_states.values[state_base + output - dim] = value;
        }
    }
    workgroupBarrier();

    if (adaptive_closure_enabled()) {
        let closure_hidden_dims = adaptive_closure_hidden_dims();
        for (var h = local_id; h < closure_hidden_dims; h = h + COOP_SIZE) {
            var sum = weights.values[closure_b1_offset() + h];
            let weight_base = closure_weight_base() + h * fd;
            for (var input = 0u; input < fd; input = input + 1u) {
                sum = sum + weights.values[weight_base + input] * coop_feature[input];
            }
            coop_hidden[h] = max(sum, 0.0);
        }
        workgroupBarrier();
        let mask_dt = update_mask(idx) * dt();
        if (is_coarse_material(idx)) {
            if (local_id == 0u) {
                var next_phase: vec2<f32>;
                for (var axis = 0u; axis < 2u; axis = axis + 1u) {
                    var value = weights.values[closure_b2_offset() + axis];
                    let weight_base = closure_w2_offset() + axis * closure_hidden_dims;
                    for (var h = 0u; h < closure_hidden_dims; h = h + 1u) {
                        value = value + weights.values[weight_base + h] * coop_hidden[h];
                    }
                    let previous = material_closure_phase(idx, axis);
                    if (axis == 0u) {
                        next_phase.x = previous + mask_dt * value;
                    } else {
                        next_phase.y = previous + mask_dt * value;
                    }
                }
                let phase_norm = length(next_phase);
                let normalized_phase = next_phase / max(phase_norm, 1.0e-6);
                next_phase = select(
                    vec2<f32>(1.0, 0.0),
                    normalized_phase,
                    phase_norm > 1.0e-6,
                );
                set_material_closure_phase(idx, 0u, next_phase.x);
                set_material_closure_phase(idx, 1u, next_phase.y);
            }
            for (var channel = local_id; channel < sd; channel = channel + COOP_SIZE) {
                let output = dim + channel;
                var value = weights.values[closure_b2_offset() + output];
                let weight_base = closure_w2_offset() + output * closure_hidden_dims;
                for (var h = 0u; h < closure_hidden_dims; h = h + 1u) {
                    value = value + weights.values[weight_base + h] * coop_hidden[h];
                }
                add_material_closure_mode(idx, channel, mask_dt * value);
            }
        }
    }
    workgroupBarrier();
    if (adaptive_closure_basis_enabled()) {
        let basis_hidden_dims = adaptive_closure_basis_hidden_dims();
        for (var h = local_id; h < basis_hidden_dims; h = h + COOP_SIZE) {
            var sum = weights.values[closure_basis_rule_b1_offset() + h];
            let weight_base = closure_basis_rule_weight_base() + h * fd;
            for (var input = 0u; input < fd; input = input + 1u) {
                sum = sum + weights.values[weight_base + input] * coop_feature[input];
            }
            coop_hidden[h] = max(sum, 0.0);
        }
        workgroupBarrier();
        if (is_coarse_material(idx) && local_id == 0u) {
            let basis_mask_dt = update_mask(idx) * dt();
            let previous_basis = vec4<f32>(
                material_closure_basis(idx, 0u),
                material_closure_basis(idx, 1u),
                material_closure_basis(idx, 2u),
                material_closure_basis(idx, 3u),
            );
            var next_basis = previous_basis;
            for (var component = 0u; component < 4u; component = component + 1u) {
                var value = weights.values[closure_basis_rule_b2_offset() + component];
                let weight_base =
                    closure_basis_rule_w2_offset() + component * basis_hidden_dims;
                for (var h = 0u; h < basis_hidden_dims; h = h + 1u) {
                    value = value + weights.values[weight_base + h] * coop_hidden[h];
                }
                next_basis[component] =
                    next_basis[component] + basis_mask_dt * value;
            }
            next_basis = next_basis - vec4<f32>(
                dot(next_basis, vec4<f32>(0.25, 0.25, 0.25, 0.25)),
            );
            let basis_norm = length(next_basis);
            next_basis = select(
                previous_basis,
                next_basis / max(basis_norm, 1.0e-6),
                basis_norm > 1.0e-6,
            );
            next_basis = select(
                next_basis,
                -next_basis,
                dot(next_basis, previous_basis) < 0.0,
            );
            for (var component = 0u; component < 4u; component = component + 1u) {
                set_material_closure_basis(idx, component, next_basis[component]);
            }
        }
    }
}

@compute @workgroup_size(32)
fn adaptive_local_residual_cooperative_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = cooperative_particle_index(workgroup);
    let local_id = local.x;
    let start_hidden = adaptive_local_hidden_start();
    if (idx >= total_count() || !adaptive_local_rule_enabled()) {
        return;
    }
    let sd = state_dims();
    let dim = spatial_dims();
    let fd = feature_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;

    let pi = position(idx);
    let center = cell_coords(pi);
    let support_cell_radius = adaptive_support_cell_radius(idx);
    let max_neighbors = adaptive_max_neighbors();
    if (local_id == 0u) {
        atomicStore(
            &adaptive_update_active,
            select(0u, 1u, update_mask(idx) != 0.0),
        );
        atomicStore(&adaptive_support_count, 0u);
        adaptive_cutoff_distance = 0xffffffffu;
        adaptive_cutoff_index = 0xffffffffu;
    }
    let update_active = workgroupUniformLoad(&adaptive_update_active) != 0u;
    let perception_active = update_active || adaptive_diagnostics_enabled();
    let paired_detail_only = adaptive_diagnostics_enabled() && pu(99u) != 0u;
    if (perception_active
        && (max_neighbors > 0u
            || (adaptive_diagnostics_enabled() && !paired_detail_only))) {
        adaptive_count_support_candidates(idx, local_id, pi, center, support_cell_radius);
    }
    workgroupBarrier();

    var observed_spacing = adaptive_spacing_max();
    if (adaptive_diagnostics_enabled() && !paired_detail_only) {
        let spacing_lo = adaptive_spacing_min();
        var spacing_hi = spacing_lo;
        var max_occupancy = adaptive_spacing_occupancy_cooperative(
            idx,
            local_id,
            pi,
            center,
            spacing_hi,
        );
        while (
            max_occupancy < adaptive_spacing_target()
                && spacing_hi < adaptive_spacing_max()
        ) {
            spacing_hi = min(2.0 * spacing_hi, adaptive_spacing_max());
            max_occupancy = adaptive_spacing_occupancy_cooperative(
                idx,
                local_id,
                pi,
                center,
                spacing_hi,
            );
        }
        if (max_occupancy >= adaptive_spacing_target()) {
            var lo = spacing_lo;
            var hi = spacing_hi;
            for (
                var iteration = 0u;
                iteration < adaptive_spacing_root_iterations();
                iteration = iteration + 1u
            ) {
                let mid = 0.5 * (lo + hi);
                let occupancy = adaptive_spacing_occupancy_cooperative(
                    idx,
                    local_id,
                    pi,
                    center,
                    mid,
                );
                if (occupancy < adaptive_spacing_target()) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            observed_spacing = 0.5 * (lo + hi);
        }
    }

    let support_count = workgroupUniformLoad(&adaptive_support_count);
    if (perception_active && max_neighbors > 0u && support_count > max_neighbors) {
        if (local_id == 0u) {
            adaptive_selection_prefix = 0u;
            adaptive_selection_rank = max_neighbors - 1u;
        }
        workgroupBarrier();
        for (var radix_pass = 0u; radix_pass < 4u; radix_pass = radix_pass + 1u) {
            for (
                var bin = local_id;
                bin < ADAPTIVE_RADIX_BINS;
                bin = bin + COOP_SIZE
            ) {
                atomicStore(&adaptive_radix_counts[bin], 0u);
            }
            workgroupBarrier();
            let shift = 24u - radix_pass * ADAPTIVE_RADIX_BITS;
            adaptive_histogram_distance(
                idx,
                local_id,
                pi,
                center,
                support_cell_radius,
                adaptive_selection_prefix,
                shift,
            );
            adaptive_select_radix_bin(local_id, shift);
        }
        if (local_id == 0u) {
            adaptive_cutoff_distance = adaptive_selection_prefix;
            adaptive_selection_prefix = 0u;
        }
        workgroupBarrier();
        let selection_rank = workgroupUniformLoad(&adaptive_selection_rank);
        if (selection_rank == 0u) {
            if (local_id == 0u) {
                atomicStore(&adaptive_min_cutoff_index, 0xffffffffu);
            }
            workgroupBarrier();
            adaptive_find_min_cutoff_index(
                idx,
                local_id,
                pi,
                center,
                support_cell_radius,
                adaptive_cutoff_distance,
            );
            workgroupBarrier();
            if (local_id == 0u) {
                adaptive_cutoff_index = atomicLoad(&adaptive_min_cutoff_index);
            }
        } else {
            for (var radix_pass = 0u; radix_pass < 4u; radix_pass = radix_pass + 1u) {
                for (
                    var bin = local_id;
                    bin < ADAPTIVE_RADIX_BINS;
                    bin = bin + COOP_SIZE
                ) {
                    atomicStore(&adaptive_radix_counts[bin], 0u);
                }
                workgroupBarrier();
                let shift = 24u - radix_pass * ADAPTIVE_RADIX_BITS;
                adaptive_histogram_index(
                    idx,
                    local_id,
                    pi,
                    center,
                    support_cell_radius,
                    adaptive_cutoff_distance,
                    adaptive_selection_prefix,
                    shift,
                );
                adaptive_select_radix_bin(local_id, shift);
            }
            if (local_id == 0u) {
                adaptive_cutoff_index = adaptive_selection_prefix;
            }
        }
    }
    workgroupBarrier();

    if (!perception_active) {
        var inactive_density = 0.0;
        for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
            let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
            for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
                for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
                let cell = cell_index_for_support_bin(coords, idx, source_bin);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let begin = sorted_offset(cell_u);
                let end = sorted_offset(cell_u + 1u);
                for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                    let j = sorted_particle(slot);
                    let pj = position(j);
                    if (!particle_candidate_matches_cell(pj, coords)) {
                        continue;
                    }
                    let delta_x = neighbor_delta(pi, pj, 0u);
                    let delta_y = neighbor_delta(pi, pj, 1u);
                    let r2 = delta_x * delta_x + delta_y * delta_y;
                    let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                    if (r2 < pair_bandwidth * pair_bandwidth) {
                        inactive_density = inactive_density
                            + particle_density_contribution_with_bandwidth(
                                j,
                                r2,
                                pair_bandwidth,
                            );
                    }
                }
            }
        }
        }
        coop_values[local_id] = inactive_density;
        workgroupBarrier();
        for (var stride = COOP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
            if (local_id < stride) {
                coop_values[local_id] = coop_values[local_id] + coop_values[local_id + stride];
            }
            workgroupBarrier();
        }
        if (local_id == 0u) {
            density.values[idx] = coop_values[0u];
            for (var axis = 0u; axis < 4u; axis = axis + 1u) {
                out_positions.values[position_base + axis] = 0.0;
            }
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                out_states.values[state_base + channel] = 0.0;
            }
        }
        return;
    }

    let shepard = adaptive_shepard_epsilon();
    var shepard_sum = select(0.0, shepard, local_id == 0u);
    var normalized_state: array<f32, MAX_STATE_DIMS>;
    var closure_blur: array<f32, MAX_CLOSURE_CONTEXT_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var occupancy_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    var rho = 0.0;
    if (local_id == 0u) {
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            normalized_state[channel] = shepard * states.values[state_base + channel];
        }
        if (adaptive_closure_enabled()) {
            for (var component = 0u; component < sd + 6u; component = component + 1u) {
                closure_blur[component] =
                    shepard * material_closure_context(idx, component);
            }
        }
    }

    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
            for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
            let coords = vec3<i32>(center.x + dx, center.y + dy, center.z);
            let cell = cell_index_for_support_bin(coords, idx, source_bin);
            if (cell < 0) {
                continue;
            }
            let cell_u = u32(cell);
            let begin = sorted_offset(cell_u);
            let end = sorted_offset(cell_u + 1u);
            for (var slot = begin + local_id; slot < end; slot = slot + COOP_SIZE) {
                let j = sorted_particle(slot);
                let pj = position(j);
                if (!particle_candidate_matches_cell(pj, coords)) {
                    continue;
                }
                var delta: array<f32, MAX_DIMS>;
                var r2 = 0.0;
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    delta[axis] = neighbor_delta(pi, pj, axis);
                    r2 = r2 + delta[axis] * delta[axis];
                }
                let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                let normalized_distance2 = r2 / (pair_bandwidth * pair_bandwidth);
                if (normalized_distance2 >= 1.0) {
                    continue;
                }
                rho = rho + particle_density_contribution_with_bandwidth(
                    j,
                    r2,
                    pair_bandwidth,
                );
                let key = bitcast<u32>(normalized_distance2);
                let selected = j == idx
                    || max_neighbors == 0u
                    || support_count <= max_neighbors
                    || key < adaptive_cutoff_distance
                    || (key == adaptive_cutoff_distance && j <= adaptive_cutoff_index);
                if (!selected) {
                    continue;
                }
                let measure_j = particle_measure(j);
                let weight = measure_j
                    * adaptive_kernel_value_with_bandwidth(r2, pair_bandwidth);
                shepard_sum = shepard_sum + weight;
                let source = j * sd;
                for (var channel = 0u; channel < sd; channel = channel + 1u) {
                    normalized_state[channel] = normalized_state[channel]
                        + weight * states.values[source + channel];
                }
                if (adaptive_closure_enabled()) {
                    for (
                        var component = 0u;
                        component < sd + 6u;
                        component = component + 1u
                    ) {
                        closure_blur[component] = closure_blur[component]
                            + weight * material_closure_context(j, component);
                    }
                }
                if (j == idx || r2 <= 0.0) {
                    continue;
                }
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    let gradient = measure_j
                        * adaptive_kernel_gradient_with_bandwidth(
                            delta,
                            r2,
                            pair_bandwidth,
                            axis,
                        );
                    occupancy_grad[axis] = occupancy_grad[axis] + gradient;
                    for (var col = 0u; col < dim; col = col + 1u) {
                        moment[axis * MAX_DIMS + col] = moment[axis * MAX_DIMS + col]
                            + gradient * delta[col];
                    }
                    for (var channel = 0u; channel < sd; channel = channel + 1u) {
                        let difference = states.values[source + channel]
                            - states.values[state_base + channel];
                        state_grad[channel * MAX_DIMS + axis] =
                            state_grad[channel * MAX_DIMS + axis] + difference * gradient;
                    }
                }
            }
        }
    }
    }

    for (var channel = 0u; channel < MAX_STATE_DIMS; channel = channel + 1u) {
        coop_values[coop_component_index(COOP_BLUR_BASE + channel, local_id)] =
            normalized_state[channel];
    }
    for (var component = 0u; component < MAX_STATE_DIMS * MAX_DIMS; component = component + 1u) {
        coop_values[coop_component_index(COOP_STATE_GRAD_BASE + component, local_id)] =
            state_grad[component];
    }
    for (var axis = 0u; axis < MAX_DIMS; axis = axis + 1u) {
        coop_values[coop_component_index(COOP_DENSITY_GRAD_BASE + axis, local_id)] =
            occupancy_grad[axis];
    }
    coop_values[coop_component_index(COOP_DENSITY_GRAD_BASE + 2u, local_id)] = shepard_sum;
    for (var component = 0u; component < MAX_DIMS * MAX_DIMS; component = component + 1u) {
        coop_values[coop_component_index(COOP_MOMENT_BASE + component, local_id)] =
            moment[component];
    }
    coop_values[coop_component_index(COOP_MOMENT_BASE + 8u, local_id)] = rho;
    for (
        var component = 0u;
        component < MAX_CLOSURE_CONTEXT_DIMS;
        component = component + 1u
    ) {
        coop_values[coop_component_index(COOP_CLOSURE_BLUR_BASE + component, local_id)] =
            closure_blur[component];
    }
    workgroupBarrier();
    reduce_coop_components(local_id, true);
    cache_coop_reduced_values(local_id);

    finish_adaptive_local_residual_cooperative(
        local_id,
        idx,
        start_hidden,
        observed_spacing,
        support_count,
        max_neighbors,
        pi,
    );
}

@compute @workgroup_size(128)
fn update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    update_particle(idx);
}

fn update_particle(idx: u32) {
    let sd = state_dims();
    let dim = spatial_dims();
    let hd = hidden_dims();
    let fd = feature_dims();
    let od = output_dims();
    let mask = update_mask(idx);
    let position_base = idx * 4u;
    let state_base = idx * sd;

    if (mask == 0.0 && !adaptive_diagnostics_enabled()) {
        for (var axis = 0u; axis < 4u; axis = axis + 1u) {
            out_positions.values[position_base + axis] = positions.values[position_base + axis];
        }
        for (var c = 0u; c < sd; c = c + 1u) {
            out_states.values[state_base + c] = states.values[state_base + c];
        }
        return;
    }

    let pi = position(idx);
    let center = cell_coords(pi);
    let eps2 = eps() * eps();

    var blur: array<f32, MAX_STATE_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var density_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;

    for (var dz = z_min(); dz <= z_max(); dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index(coords, idx);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let cap = bucket_capacity();
                if (is_sorted_layout()) {
                    let start = sorted_offset(cell_u);
                    let end = sorted_offset(cell_u + 1u);
                    for (var slot = start; slot < end; slot = slot + 1u) {
                        let j = sorted_particle(slot);
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        if (r2 < eps2) {
                            let volume_j = particle_volume(j);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(
                                        delta,
                                        r2,
                                        density_gradient_weight(j),
                                        axis,
                                    );
                                    density_grad[axis] = density_grad[axis] + grad;
                                }

                                for (var c = 0u; c < sd; c = c + 1u) {
                                    let diff = states.values[src + c] - states.values[state_base + c];
                                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, axis);
                                        state_grad[c * MAX_DIMS + axis] =
                                            state_grad[c * MAX_DIMS + axis] + diff * grad;
                                    }
                                }

                                for (var row = 0u; row < dim; row = row + 1u) {
                                    for (var col = 0u; col < dim; col = col + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, col);
                                        moment[row * MAX_DIMS + col] =
                                            moment[row * MAX_DIMS + col] + delta[row] * grad;
                                    }
                                }
                            }
                        }
                    }
                } else if (cap == 0u) {
                    var j = atomicLoad(&linked_grid.values[cell_u]);
                    while (j != NIL) {
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            j = linked_next(j);
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        if (r2 < eps2) {
                            let volume_j = particle_volume(j);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(
                                        delta,
                                        r2,
                                        density_gradient_weight(j),
                                        axis,
                                    );
                                    density_grad[axis] = density_grad[axis] + grad;
                                }

                                for (var c = 0u; c < sd; c = c + 1u) {
                                    let diff = states.values[src + c] - states.values[state_base + c];
                                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, axis);
                                        state_grad[c * MAX_DIMS + axis] =
                                            state_grad[c * MAX_DIMS + axis] + diff * grad;
                                    }
                                }

                                for (var row = 0u; row < dim; row = row + 1u) {
                                    for (var col = 0u; col < dim; col = col + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, col);
                                        moment[row * MAX_DIMS + col] =
                                            moment[row * MAX_DIMS + col] + delta[row] * grad;
                                    }
                                }
                            }
                        }
                        j = linked_next(j);
                    }
                } else {
                    let count = min(atomicLoad(&linked_grid.values[cell_u]), cap);
                    for (var slot = 0u; slot < count; slot = slot + 1u) {
                        let j = atomicLoad(&linked_grid.values[bucket_slot_index(cell_u, slot)]);
                        let pj = position(j);
                        if (!particle_candidate_matches_cell(pj, coords)) {
                            continue;
                        }
                        var delta: array<f32, MAX_DIMS>;
                        var r2 = 0.0;
                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                            delta[axis] = neighbor_delta(pi, pj, axis);
                            r2 = r2 + delta[axis] * delta[axis];
                        }
                        if (r2 < eps2) {
                            let volume_j = particle_volume(j);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(
                                        delta,
                                        r2,
                                        density_gradient_weight(j),
                                        axis,
                                    );
                                    density_grad[axis] = density_grad[axis] + grad;
                                }

                                for (var c = 0u; c < sd; c = c + 1u) {
                                    let diff = states.values[src + c] - states.values[state_base + c];
                                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, axis);
                                        state_grad[c * MAX_DIMS + axis] =
                                            state_grad[c * MAX_DIMS + axis] + diff * grad;
                                    }
                                }

                                for (var row = 0u; row < dim; row = row + 1u) {
                                    for (var col = 0u; col < dim; col = col + 1u) {
                                        let grad = spiky_gradient(delta, r2, volume_j, col);
                                        moment[row * MAX_DIMS + col] =
                                            moment[row * MAX_DIMS + col] + delta[row] * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    finish_update_particle(idx, mask, pi, blur, state_grad, density_grad, moment);
}

fn reduce_coop_components(local_id: u32, adaptive_scalars: bool) {
    let sd = state_dims();
    let dim = spatial_dims();
    for (var stride = COOP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                let dst = coop_component_index(COOP_BLUR_BASE + channel, local_id);
                coop_values[dst] = coop_values[dst] + coop_values[dst + stride];
            }
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    let component = COOP_STATE_GRAD_BASE + channel * MAX_DIMS + axis;
                    let dst = coop_component_index(component, local_id);
                    coop_values[dst] = coop_values[dst] + coop_values[dst + stride];
                }
            }
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                let dst = coop_component_index(COOP_DENSITY_GRAD_BASE + axis, local_id);
                coop_values[dst] = coop_values[dst] + coop_values[dst + stride];
            }
            for (var row = 0u; row < dim; row = row + 1u) {
                for (var col = 0u; col < dim; col = col + 1u) {
                    let component = COOP_MOMENT_BASE + row * MAX_DIMS + col;
                    let dst = coop_component_index(component, local_id);
                    coop_values[dst] = coop_values[dst] + coop_values[dst + stride];
                }
            }
            if (adaptive_scalars) {
                let partition_index =
                    coop_component_index(COOP_DENSITY_GRAD_BASE + 2u, local_id);
                coop_values[partition_index] =
                    coop_values[partition_index] + coop_values[partition_index + stride];
                let raw_density = coop_component_index(COOP_MOMENT_BASE + 8u, local_id);
                coop_values[raw_density] =
                    coop_values[raw_density] + coop_values[raw_density + stride];
                for (
                    var component = 0u;
                    component < sd + 6u;
                    component = component + 1u
                ) {
                    let closure =
                        coop_component_index(COOP_CLOSURE_BLUR_BASE + component, local_id);
                    coop_values[closure] =
                        coop_values[closure] + coop_values[closure + stride];
                }
            }
        }
        workgroupBarrier();
    }
}

fn finish_update_particle_cooperative(local_id: u32, idx: u32, mask: f32, pi: vec4<f32>) {
    let sd = state_dims();
    let dim = spatial_dims();
    let hd = hidden_dims();
    let fd = feature_dims();
    let od = output_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;

    if (local_id == 0u) {
        var inverse: array<f32, MAX_DIMS * MAX_DIMS>;
        if (dim == 2u) {
            let a = coop_reduced_values[COOP_MOMENT_BASE + 0u];
            let b = coop_reduced_values[COOP_MOMENT_BASE + 1u];
            let d = coop_reduced_values[COOP_MOMENT_BASE + 4u];
            let det = a * d - b * b;
            if (abs(det) < 1e-3) {
                inverse[0u] = 1.0;
                inverse[4u] = 1.0;
            } else {
                let inv_det = 1.0 / det;
                inverse[0u] = d * inv_det;
                inverse[1u] = -b * inv_det;
                inverse[3u] = -b * inv_det;
                inverse[4u] = a * inv_det;
            }
        } else {
            let a = coop_reduced_values[COOP_MOMENT_BASE + 0u];
            let b = coop_reduced_values[COOP_MOMENT_BASE + 1u];
            let c = coop_reduced_values[COOP_MOMENT_BASE + 2u];
            let d = coop_reduced_values[COOP_MOMENT_BASE + 4u];
            let e = coop_reduced_values[COOP_MOMENT_BASE + 5u];
            let f = coop_reduced_values[COOP_MOMENT_BASE + 8u];
            let t1 = d * f - e * e;
            let t2 = c * e - b * f;
            let t3 = b * e - c * d;
            let det = a * t1 + b * t2 + c * t3;
            if (abs(det) < 1e-3) {
                inverse[0u] = 1.0;
                inverse[4u] = 1.0;
                inverse[8u] = 1.0;
            } else {
                let inv_det = 1.0 / det;
                inverse[0u] = t1 * inv_det;
                inverse[1u] = t2 * inv_det;
                inverse[2u] = t3 * inv_det;
                inverse[3u] = t2 * inv_det;
                inverse[4u] = (a * f - c * c) * inv_det;
                inverse[5u] = (b * c - a * e) * inv_det;
                inverse[6u] = t3 * inv_det;
                inverse[7u] = (b * c - a * e) * inv_det;
                inverse[8u] = (a * d - b * b) * inv_det;
            }
        }

        var cursor = 0u;
        for (var c = 0u; c < sd; c = c + 1u) {
            coop_feature[cursor] = states.values[state_base + c];
            cursor = cursor + 1u;
        }
        for (var c = 0u; c < sd; c = c + 1u) {
            coop_feature[cursor] = coop_reduced_values[COOP_BLUR_BASE + c];
            cursor = cursor + 1u;
        }

        for (var c = 0u; c < sd; c = c + 1u) {
            var corrected: array<f32, MAX_DIMS>;
            for (var out_axis = 0u; out_axis < dim; out_axis = out_axis + 1u) {
                var value = 0.0;
                for (var in_axis = 0u; in_axis < dim; in_axis = in_axis + 1u) {
                    value = value
                        + coop_reduced_values[
                            COOP_STATE_GRAD_BASE + c * MAX_DIMS + in_axis
                        ] * inverse[in_axis * MAX_DIMS + out_axis];
                }
                corrected[out_axis] = value * particle_state_gradient_scale(idx);
            }

            var norm = 0.0;
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                norm = norm + corrected[axis] * corrected[axis];
            }
            norm = sqrt(norm);
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                var value = corrected[axis];
                if (pu(12u) != 0u) {
                    value = log_normalized(value, norm);
                }
                coop_feature[cursor] = value;
                cursor = cursor + 1u;
            }
        }

        var density_grad_local: array<f32, MAX_DIMS>;
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            density_grad_local[axis] =
                coop_reduced_values[COOP_DENSITY_GRAD_BASE + axis]
                * particle_density_gradient_scale(idx);
        }
        if (pu(13u) != 0u) {
            var norm = 0.0;
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                norm = norm + density_grad_local[axis] * density_grad_local[axis];
            }
            norm = sqrt(norm);
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                density_grad_local[axis] = log_normalized(density_grad_local[axis], norm);
            }
        }
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            coop_feature[cursor] = density_grad_local[axis];
            cursor = cursor + 1u;
        }
        if (has_position_features()) {
            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                coop_feature[cursor] = vec4_axis(pi, axis);
                cursor = cursor + 1u;
            }
        }
        if (material_scale_conditioning_enabled(cursor)) {
            coop_feature[cursor] = particle_material_scale_feature(idx);
            cursor = cursor + 1u;
        } else if (residual_material_features_enabled(cursor)) {
            coop_feature[cursor] = particle_material_scale_feature(idx);
            coop_feature[cursor + 1u] = clamp(
                density.values[diagnostics_coarse_exposure_offset() + idx],
                0.0,
                1.0,
            );
            cursor = cursor + 2u;
        }
        let closure_dims = 1u + dim * (dim + 1u) / 2u + sd * dim;
        if (fd >= cursor + closure_dims) {
            let measure = max(particle_measure(idx), 1.0e-20);
            var footprint = sqrt(measure / 3.141592653589793);
            if (dim == 3u) {
                footprint = pow(
                    measure / (4.0 * 3.141592653589793 / 3.0),
                    1.0 / 3.0,
                );
            }
            coop_feature[cursor] = clamp(
                log2(footprint / max(adaptive_reference_footprint(), 1.0e-20)),
                -3.0,
                3.0,
            );
            cursor = cursor + 1u;
            let footprint2 = max(footprint * footprint, 1.0e-20);
            for (var lhs = 0u; lhs < dim; lhs = lhs + 1u) {
                for (var rhs = lhs; rhs < dim; rhs = rhs + 1u) {
                    coop_feature[cursor] = clamp(
                        material_covariance_value(idx, lhs * 3u + rhs) / footprint2,
                        -8.0,
                        8.0,
                    );
                    cursor = cursor + 1u;
                }
            }
            for (var component = 0u; component < sd * dim; component = component + 1u) {
                let scaled = material_state_jacobian(idx, component) * footprint;
                coop_feature[cursor] =
                    sign(scaled) * min(log(1.0 + abs(scaled)), 8.0);
                cursor = cursor + 1u;
            }
            if (adaptive_closure_enabled() && fd >= cursor + 6u + sd) {
                for (var component = 0u; component < 4u; component = component + 1u) {
                    coop_feature[cursor + component] = material_closure_basis(idx, component);
                }
                cursor = cursor + 4u;
                for (var component = 0u; component < 2u; component = component + 1u) {
                    coop_feature[cursor + component] = material_closure_phase(idx, component);
                }
                cursor = cursor + 2u;
                for (var channel = 0u; channel < sd; channel = channel + 1u) {
                    coop_feature[cursor + channel] = material_closure_mode(idx, channel);
                }
                cursor = cursor + sd;
            }
        }
    }
    var base_feature_dims = 2u * sd + sd * dim + dim;
    if (has_position_features()) {
        base_feature_dims = base_feature_dims + dim;
    }
    if (material_scale_conditioning_enabled(base_feature_dims)) {
        base_feature_dims = base_feature_dims + 1u;
    } else if (residual_material_features_enabled(base_feature_dims)) {
        base_feature_dims = base_feature_dims + 2u;
    }
    let closure_dims = 1u + dim * (dim + 1u) / 2u + sd * dim;
    if (fd >= base_feature_dims + closure_dims) {
        base_feature_dims = base_feature_dims + closure_dims;
        if (adaptive_closure_enabled() && fd >= base_feature_dims + 6u + sd) {
            base_feature_dims = base_feature_dims + 6u + sd;
        }
    }
    for (
        var feature = base_feature_dims + local_id;
        feature < fd;
        feature = feature + COOP_SIZE
    ) {
        coop_feature[feature] = 0.0;
    }
    workgroupBarrier();

    if (adaptive_diagnostics_enabled()) {
        for (var feature = local_id; feature < fd; feature = feature + COOP_SIZE) {
            density.values[diagnostics_base_feature_offset() + idx * fd + feature] =
                coop_feature[feature];
        }
    }
    if (pu(99u) == 2u) {
        for (var feature = local_id; feature < fd; feature = feature + COOP_SIZE) {
            density.values[
                diagnostics_normalized_feature_offset() + idx * fd + feature
            ] = coop_feature[feature];
        }
    }

    for (var h = local_id; h < hd; h = h + COOP_SIZE) {
        var sum = weights.values[b1_offset() + h];
        let w_base = h * fd;
        for (var i = 0u; i < fd; i = i + 1u) {
            sum = sum + weights.values[w_base + i] * coop_feature[i];
        }
        coop_hidden[h] = max(sum, 0.0);
    }
    workgroupBarrier();

    for (var o = local_id; o < od; o = o + COOP_SIZE) {
        var sum = weights.values[b2_offset() + o];
        var base_sum = sum;
        let w_base = w2_offset() + o * hd;
        var base_hidden_end = hd;
        if (adaptive_local_rule_enabled() && adaptive_local_hidden_start() > 0u) {
            base_hidden_end = min(adaptive_local_hidden_start(), hd);
        }
        for (var h = 0u; h < hd; h = h + 1u) {
            let value = weights.values[w_base + h] * coop_hidden[h];
            sum = sum + value * adaptive_hidden_scale(idx, h);
            if (h < base_hidden_end) {
                base_sum = base_sum + value;
            }
        }
        if (adaptive_diagnostics_enabled()) {
            density.values[diagnostics_base_update_offset() + idx * od + o] = base_sum;
        }
        let model_update = adaptive_combined_update_value(idx, o, sum);
        if (adaptive_diagnostics_enabled()) {
            density.values[diagnostics_model_update_offset() + idx * od + o] = model_update;
        }
        coop_update_values[o] = model_update;
    }
    workgroupBarrier();

    if (local_id == 0u) {
        var update_norm = 0.0;
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            update_norm = update_norm + coop_update_values[axis] * coop_update_values[axis];
        }
        update_norm = sqrt(update_norm);
        for (var axis = 0u; axis < 4u; axis = axis + 1u) {
            var value = positions.values[position_base + axis];
            if (axis < dim) {
                value = value
                    + mask * dt() * alpha() * coop_update_values[axis] * particle_motion_eps(idx)
                        / (1.0 + update_norm);
                value = wrap_axis(value, axis);
            }
            out_positions.values[position_base + axis] = value;
        }

        let update_state_base = dim;
        for (var c = 0u; c < sd; c = c + 1u) {
            var next_state =
                states.values[state_base + c] + mask * dt() * coop_update_values[update_state_base + c];
            if (dim == 3u && sd > 3u && (c == 3u || (sd > 8u && c == 8u))) {
                next_state = clamp(
                    next_state,
                    GROWTH_3D_MIN_OPACITY_LOGIT,
                    GROWTH_3D_MAX_OPACITY_LOGIT,
                );
            }
            out_states.values[state_base + c] = next_state;
        }
    }
}

@compute @workgroup_size(32)
fn cooperative_update_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = cooperative_particle_index(workgroup);
    let local_id = local.x;
    if (idx >= total_count()) {
        return;
    }

    let sd = state_dims();
    let dim = spatial_dims();
    if (local_id == 0u) {
        cooperative_mask = update_mask(idx);
    }
    let mask = workgroupUniformLoad(&cooperative_mask);
    if (mask == 0.0 && !adaptive_diagnostics_enabled() && pu(99u) != 2u) {
        if (local_id == 0u) {
            copy_particle_to_output(idx);
        }
        return;
    }

    let pi = position(idx);
    let center = cell_coords(pi);
    let support_cell_radius = adaptive_support_cell_radius(idx);
    let state_base = idx * sd;

    var blur: array<f32, MAX_STATE_DIMS>;
    var target_state: array<f32, MAX_STATE_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var density_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    for (var c = 0u; c < sd; c = c + 1u) {
        target_state[c] = states.values[state_base + c];
    }

    for (var source_bin = 0u; source_bin < support_bin_count(); source_bin = source_bin + 1u) {
        let cell_radius = adaptive_support_cell_radius_for_bin(idx, source_bin);
        var z_radius = 0;
        if (dim == 3u) {
            z_radius = cell_radius;
        }
        for (var dz = -z_radius; dz <= z_radius; dz = dz + 1) {
            for (var dy = -cell_radius; dy <= cell_radius; dy = dy + 1) {
                for (var dx = -cell_radius; dx <= cell_radius; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index_for_support_bin(coords, idx, source_bin);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let start = sorted_offset(cell_u);
                let end = sorted_offset(cell_u + 1u);
                for (var slot = start + local_id; slot < end; slot = slot + COOP_SIZE) {
                    let j = sorted_particle(slot);
                    let pj = position(j);
                    if (!particle_candidate_matches_cell(pj, coords)) {
                        continue;
                    }
                    var delta: array<f32, MAX_DIMS>;
                    var r2 = 0.0;
                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                        delta[axis] = neighbor_delta(pi, pj, axis);
                        r2 = r2 + delta[axis] * delta[axis];
                    }
                    let pair_bandwidth = adaptive_pair_bandwidth(idx, j);
                    if (r2 < pair_bandwidth * pair_bandwidth) {
                        let volume_j = particle_volume(j);
                        let smooth_w = smoothing_poly6_with_bandwidth(r2, pair_bandwidth);
                        let src = j * sd;
                        if (idx != j && r2 > 0.0) {
                            var grad_volume: array<f32, MAX_DIMS>;
                            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                let grad_volume_axis = spiky_gradient_with_bandwidth(
                                    delta,
                                    r2,
                                    pair_bandwidth,
                                    volume_j,
                                    axis,
                                );
                                let grad_density_axis = spiky_gradient_with_bandwidth(
                                    delta,
                                    r2,
                                    pair_bandwidth,
                                    density_gradient_weight(j),
                                    axis,
                                );
                                grad_volume[axis] = grad_volume_axis;
                                density_grad[axis] = density_grad[axis]
                                    + grad_density_axis;
                            }

                            for (var c = 0u; c < sd; c = c + 1u) {
                                let state_j = states.values[src + c];
                                blur[c] = blur[c] + state_j * smooth_w * volume_j;
                                let diff = state_j - target_state[c];
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    state_grad[c * MAX_DIMS + axis] =
                                        state_grad[c * MAX_DIMS + axis] + diff * grad_volume[axis];
                                }
                            }

                            for (var row = 0u; row < dim; row = row + 1u) {
                                for (var col = 0u; col < dim; col = col + 1u) {
                                    moment[row * MAX_DIMS + col] =
                                        moment[row * MAX_DIMS + col] + delta[row] * grad_volume[col];
                                }
                            }
                        } else {
                            for (var c = 0u; c < sd; c = c + 1u) {
                                let state_j = states.values[src + c];
                                blur[c] = blur[c] + state_j * smooth_w * volume_j;
                            }
                        }
                    }
                }
            }
        }
    }
    }

    for (var c = 0u; c < sd; c = c + 1u) {
        coop_values[coop_component_index(COOP_BLUR_BASE + c, local_id)] = blur[c];
    }
    for (var c = 0u; c < sd; c = c + 1u) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            let component = c * MAX_DIMS + axis;
            coop_values[coop_component_index(COOP_STATE_GRAD_BASE + component, local_id)] =
                state_grad[component];
        }
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        coop_values[coop_component_index(COOP_DENSITY_GRAD_BASE + axis, local_id)] =
            density_grad[axis];
    }
    for (var row = 0u; row < dim; row = row + 1u) {
        for (var col = 0u; col < dim; col = col + 1u) {
            let component = row * MAX_DIMS + col;
            coop_values[coop_component_index(COOP_MOMENT_BASE + component, local_id)] =
                moment[component];
        }
    }
    workgroupBarrier();
    reduce_coop_components(local_id, false);
    cache_coop_reduced_values(local_id);

    finish_update_particle_cooperative(local_id, idx, mask, pi);
}

@compute @workgroup_size(128)
fn bvh_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    update_bvh_particle(idx);
}

fn update_bvh_particle(idx: u32) {
    let sd = state_dims();
    let dim = spatial_dims();
    let mask = update_mask(idx);
    let state_base = idx * sd;

    if (mask == 0.0) {
        copy_particle_to_output(idx);
        return;
    }

    let pi = position(idx);
    let eps2 = eps() * eps();

    var blur: array<f32, MAX_STATE_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var density_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    var stack: array<u32, BVH_STACK_SIZE>;
    var stack_len = 0u;
    if (bvh_node_count() > 0u) {
        stack[stack_len] = 0u;
        stack_len = stack_len + 1u;
    }

    while (stack_len > 0u) {
        stack_len = stack_len - 1u;
        let node = stack[stack_len];
        if (!bvh_intersects_eps(pi, node)) {
            continue;
        }
        if (bvh_node_is_leaf(node)) {
            let start = bvh_node_left_or_start(node);
            let count = bvh_node_right_or_count(node);
            for (var slot = 0u; slot < count; slot = slot + 1u) {
                let j = bvh_particle(start + slot);
                let pj = position(j);
                var delta: array<f32, MAX_DIMS>;
                var r2 = 0.0;
                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                    delta[axis] = neighbor_delta(pi, pj, axis);
                    r2 = r2 + delta[axis] * delta[axis];
                }
                if (r2 < eps2) {
                    let volume_j = particle_volume(j);
                    let smooth_w = smoothing_poly6(r2);
                    let src = j * sd;
                    for (var c = 0u; c < sd; c = c + 1u) {
                        blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                    }

                    if (idx != j) {
                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                            let grad = spiky_gradient(
                                delta,
                                r2,
                                density_gradient_weight(j),
                                axis,
                            );
                            density_grad[axis] = density_grad[axis] + grad;
                        }

                        for (var c = 0u; c < sd; c = c + 1u) {
                            let diff = states.values[src + c] - states.values[state_base + c];
                            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                let grad = spiky_gradient(delta, r2, volume_j, axis);
                                state_grad[c * MAX_DIMS + axis] =
                                    state_grad[c * MAX_DIMS + axis] + diff * grad;
                            }
                        }

                        for (var row = 0u; row < dim; row = row + 1u) {
                            for (var col = 0u; col < dim; col = col + 1u) {
                                let grad = spiky_gradient(delta, r2, volume_j, col);
                                moment[row * MAX_DIMS + col] =
                                    moment[row * MAX_DIMS + col] + delta[row] * grad;
                            }
                        }
                    }
                }
            }
        } else {
            let left = bvh_node_left_or_start(node);
            let right = bvh_node_right_or_count(node);
            if (stack_len + 2u <= BVH_STACK_SIZE) {
                stack[stack_len] = left;
                stack[stack_len + 1u] = right;
                stack_len = stack_len + 2u;
            }
        }
    }

    finish_update_particle(idx, mask, pi, blur, state_grad, density_grad, moment);
}

fn copy_particle_to_output(idx: u32) {
    let sd = state_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;
    for (var axis = 0u; axis < 4u; axis = axis + 1u) {
        out_positions.values[position_base + axis] = positions.values[position_base + axis];
    }
    for (var c = 0u; c < sd; c = c + 1u) {
        out_states.values[state_base + c] = states.values[state_base + c];
    }
}

fn finish_update_particle(
    idx: u32,
    mask: f32,
    pi: vec4<f32>,
    blur: array<f32, MAX_STATE_DIMS>,
    state_grad_in: array<f32, MAX_STATE_DIMS * MAX_DIMS>,
    density_grad_in: array<f32, MAX_DIMS>,
    moment: array<f32, MAX_DIMS * MAX_DIMS>,
) {
    let sd = state_dims();
    let dim = spatial_dims();
    let hd = hidden_dims();
    let fd = feature_dims();
    let od = output_dims();
    let position_base = idx * 4u;
    let state_base = idx * sd;
    var state_grad = state_grad_in;
    var density_grad = density_grad_in;
    var inverse: array<f32, MAX_DIMS * MAX_DIMS>;
    if (dim == 2u) {
        let a = moment[0u];
        let b = moment[1u];
        let d = moment[4u];
        let det = a * d - b * b;
        if (abs(det) < 1e-3) {
            inverse[0u] = 1.0;
            inverse[4u] = 1.0;
        } else {
            let inv_det = 1.0 / det;
            inverse[0u] = d * inv_det;
            inverse[1u] = -b * inv_det;
            inverse[3u] = -b * inv_det;
            inverse[4u] = a * inv_det;
        }
    } else {
        let a = moment[0u];
        let b = moment[1u];
        let c = moment[2u];
        let d = moment[4u];
        let e = moment[5u];
        let f = moment[8u];
        let t1 = d * f - e * e;
        let t2 = c * e - b * f;
        let t3 = b * e - c * d;
        let det = a * t1 + b * t2 + c * t3;
        if (abs(det) < 1e-3) {
            inverse[0u] = 1.0;
            inverse[4u] = 1.0;
            inverse[8u] = 1.0;
        } else {
            let inv_det = 1.0 / det;
            inverse[0u] = t1 * inv_det;
            inverse[1u] = t2 * inv_det;
            inverse[2u] = t3 * inv_det;
            inverse[3u] = t2 * inv_det;
            inverse[4u] = (a * f - c * c) * inv_det;
            inverse[5u] = (b * c - a * e) * inv_det;
            inverse[6u] = t3 * inv_det;
            inverse[7u] = (b * c - a * e) * inv_det;
            inverse[8u] = (a * d - b * b) * inv_det;
        }
    }

    for (var c = 0u; c < sd; c = c + 1u) {
        var corrected: array<f32, MAX_DIMS>;
        for (var out_axis = 0u; out_axis < dim; out_axis = out_axis + 1u) {
            var value = 0.0;
            for (var in_axis = 0u; in_axis < dim; in_axis = in_axis + 1u) {
                value = value
                    + state_grad[c * MAX_DIMS + in_axis] * inverse[in_axis * MAX_DIMS + out_axis];
            }
            corrected[out_axis] = value * particle_state_gradient_scale(idx);
        }

        var norm = 0.0;
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            norm = norm + corrected[axis] * corrected[axis];
        }
        norm = sqrt(norm);
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            var value = corrected[axis];
            if (pu(12u) != 0u) {
                value = log_normalized(value, norm);
            }
            state_grad[c * MAX_DIMS + axis] = value;
        }
    }

    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        density_grad[axis] = density_grad[axis] * particle_density_gradient_scale(idx);
    }
    if (pu(13u) != 0u) {
        var norm = 0.0;
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            norm = norm + density_grad[axis] * density_grad[axis];
        }
        norm = sqrt(norm);
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            density_grad[axis] = log_normalized(density_grad[axis], norm);
        }
    }

    var feature: array<f32, MAX_FEATURE_DIMS>;
    var cursor = 0u;
    for (var c = 0u; c < sd; c = c + 1u) {
        feature[cursor] = states.values[state_base + c];
        cursor = cursor + 1u;
    }
    for (var c = 0u; c < sd; c = c + 1u) {
        feature[cursor] = blur[c];
        cursor = cursor + 1u;
    }
    for (var c = 0u; c < sd; c = c + 1u) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            feature[cursor] = state_grad[c * MAX_DIMS + axis];
            cursor = cursor + 1u;
        }
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        feature[cursor] = density_grad[axis];
        cursor = cursor + 1u;
    }
    if (has_position_features()) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            feature[cursor] = vec4_axis(pi, axis);
            cursor = cursor + 1u;
        }
    }
    if (material_scale_conditioning_enabled(cursor)) {
        feature[cursor] = particle_material_scale_feature(idx);
        cursor = cursor + 1u;
    } else if (residual_material_features_enabled(cursor)) {
        feature[cursor] = particle_material_scale_feature(idx);
        feature[cursor + 1u] = clamp(
            density.values[diagnostics_coarse_exposure_offset() + idx],
            0.0,
            1.0,
        );
        cursor = cursor + 2u;
    }
    let closure_dims = 1u + dim * (dim + 1u) / 2u + sd * dim;
    if (fd >= cursor + closure_dims) {
        let measure = max(particle_measure(idx), 1.0e-20);
        var footprint = sqrt(measure / 3.141592653589793);
        if (dim == 3u) {
            footprint = pow(
                measure / (4.0 * 3.141592653589793 / 3.0),
                1.0 / 3.0,
            );
        }
        feature[cursor] = clamp(
            log2(footprint / max(adaptive_reference_footprint(), 1.0e-20)),
            -3.0,
            3.0,
        );
        cursor = cursor + 1u;
        let footprint2 = max(footprint * footprint, 1.0e-20);
        for (var lhs = 0u; lhs < dim; lhs = lhs + 1u) {
            for (var rhs = lhs; rhs < dim; rhs = rhs + 1u) {
                feature[cursor] = clamp(
                    material_covariance_value(idx, lhs * 3u + rhs) / footprint2,
                    -8.0,
                    8.0,
                );
                cursor = cursor + 1u;
            }
        }
        for (var component = 0u; component < sd * dim; component = component + 1u) {
            let scaled = material_state_jacobian(idx, component) * footprint;
            feature[cursor] = sign(scaled) * min(log(1.0 + abs(scaled)), 8.0);
            cursor = cursor + 1u;
        }
        if (adaptive_closure_enabled() && fd >= cursor + 6u + sd) {
            for (var component = 0u; component < 4u; component = component + 1u) {
                feature[cursor + component] = material_closure_basis(idx, component);
            }
            cursor = cursor + 4u;
            for (var component = 0u; component < 2u; component = component + 1u) {
                feature[cursor + component] = material_closure_phase(idx, component);
            }
            cursor = cursor + 2u;
            for (var channel = 0u; channel < sd; channel = channel + 1u) {
                feature[cursor + channel] = material_closure_mode(idx, channel);
            }
            cursor = cursor + sd;
        }
    }
    for (var input = cursor; input < fd; input = input + 1u) {
        feature[input] = 0.0;
    }

    var hidden: array<f32, MAX_HIDDEN_DIMS>;
    for (var h = 0u; h < hd; h = h + 1u) {
        var sum = weights.values[b1_offset() + h];
        let w_base = h * fd;
        for (var i = 0u; i < fd; i = i + 1u) {
            sum = sum + weights.values[w_base + i] * feature[i];
        }
        hidden[h] = max(sum, 0.0);
    }

    var update: array<f32, MAX_OUTPUT_DIMS>;
    for (var o = 0u; o < od; o = o + 1u) {
        var sum = weights.values[b2_offset() + o];
        let w_base = w2_offset() + o * hd;
        for (var h = 0u; h < hd; h = h + 1u) {
            sum = sum + weights.values[w_base + h] * hidden[h] * adaptive_hidden_scale(idx, h);
        }
        update[o] = adaptive_combined_update_value(idx, o, sum);
    }

    var update_norm = 0.0;
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        update_norm = update_norm + update[axis] * update[axis];
    }
    update_norm = sqrt(update_norm);
    for (var axis = 0u; axis < 4u; axis = axis + 1u) {
        var value = positions.values[position_base + axis];
        if (axis < dim) {
            value = value
                + mask * dt() * alpha() * update[axis] * particle_motion_eps(idx)
                    / (1.0 + update_norm);
            value = wrap_axis(value, axis);
        }
        out_positions.values[position_base + axis] = value;
    }

    let update_state_base = dim;
    for (var c = 0u; c < sd; c = c + 1u) {
        var next_state =
            states.values[state_base + c] + mask * dt() * update[update_state_base + c];
        if (dim == 3u && sd > 3u && (c == 3u || (sd > 8u && c == 8u))) {
            next_state = clamp(
                next_state,
                GROWTH_3D_MIN_OPACITY_LOGIT,
                GROWTH_3D_MAX_OPACITY_LOGIT,
            );
        }
        out_states.values[state_base + c] = next_state;
    }
}

@compute @workgroup_size(128)
fn tiled_update_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    if (bucket_capacity() == 0u) {
        return;
    }

    let local_id = local.x;
    let block_start = workgroup.y * TILE_SIZE;
    if (local_id == 0u) {
        let loaded_target_cell = active_cell(workgroup.x);
        let loaded_target_count = min(
            atomicLoad(&linked_grid.values[loaded_target_cell]),
            bucket_capacity(),
        );
        var reference_index = 0u;
        tile_center = vec3<i32>(0, 0, 0);
        if (block_start < loaded_target_count) {
            reference_index = atomicLoad(
                &linked_grid.values[bucket_slot_index(loaded_target_cell, block_start)],
            );
            tile_center = cell_coords(position(reference_index));
        }
        tile_dispatch = vec4<u32>(
            loaded_target_cell,
            loaded_target_count,
            reference_index,
            0u,
        );
        atomicStore(&tile_mismatch, 0u);
    }
    let dispatch = workgroupUniformLoad(&tile_dispatch);
    let common_center = workgroupUniformLoad(&tile_center);
    let target_cell = dispatch.x;
    let target_count = dispatch.y;
    let reference_index = dispatch.z;
    if (block_start >= target_count) {
        return;
    }

    let target_slot = block_start + local_id;
    let has_target = target_slot < target_count;
    var idx = 0u;
    var pi = vec4<f32>(0.0);
    var center = vec3<i32>(0, 0, 0);
    if (has_target) {
        idx = atomicLoad(&linked_grid.values[bucket_slot_index(target_cell, target_slot)]);
        pi = position(idx);
        center = cell_coords(pi);
    }

    if (has_target && !same_cell_coords(center, common_center)) {
        atomicStore(&tile_mismatch, 1u);
    }
    let mismatch = workgroupUniformLoad(&tile_mismatch);
    if (mismatch != 0u) {
        if (has_target) {
            update_particle(idx);
        }
        return;
    }

    let sd = state_dims();
    let dim = spatial_dims();
    let mask = select(0.0, update_mask(idx), has_target);
    let should_update = has_target && mask != 0.0;
    let state_base = idx * sd;
    let eps2 = eps() * eps();

    var blur: array<f32, MAX_STATE_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var density_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;

    for (var dz = z_min(); dz <= z_max(); dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coords = vec3<i32>(
                    common_center.x + dx,
                    common_center.y + dy,
                    common_center.z + dz,
                );
                if (local_id == 0u) {
                    let cell = cell_index(coords, reference_index);
                    var cell_u = NIL;
                    var count = 0u;
                    if (cell >= 0) {
                        cell_u = u32(cell);
                        count = min(
                            atomicLoad(&linked_grid.values[cell_u]),
                            bucket_capacity(),
                        );
                    }
                    tile_neighbor = vec2<u32>(cell_u, count);
                }
                let neighbor = workgroupUniformLoad(&tile_neighbor);
                let cell_u = neighbor.x;
                let count = neighbor.y;
                if (cell_u == NIL) {
                    continue;
                }
                for (var chunk = 0u; chunk < count; chunk = chunk + TILE_SIZE) {
                    let load_slot = chunk + local_id;
                    if (load_slot < count) {
                        let j = atomicLoad(&linked_grid.values[bucket_slot_index(cell_u, load_slot)]);
                        tile_indices[local_id] = j;
                        tile_positions[local_id] = position(j);
                        tile_density[local_id] = density.values[j];
                        let src = j * sd;
                        let state_tile_base = local_id * MAX_STATE_DIMS;
                        for (var c = 0u; c < sd; c = c + 1u) {
                            tile_states[state_tile_base + c] = states.values[src + c];
                        }
                    } else {
                        tile_indices[local_id] = NIL;
                        tile_positions[local_id] = vec4<f32>(0.0);
                        tile_density[local_id] = 0.0;
                    }
                    workgroupBarrier();

                    let chunk_count = min(TILE_SIZE, count - chunk);
                    if (should_update) {
                        for (var k = 0u; k < chunk_count; k = k + 1u) {
                            let j = tile_indices[k];
                            if (j == NIL) {
                                continue;
                            }
                            let pj = tile_positions[k];
                            if (!particle_candidate_matches_cell(pj, coords)) {
                                continue;
                            }
                            var delta: array<f32, MAX_DIMS>;
                            var r2 = 0.0;
                            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                delta[axis] = neighbor_delta(pi, pj, axis);
                                r2 = r2 + delta[axis] * delta[axis];
                            }
                            if (r2 < eps2) {
                                let volume_j = particle_measure(j) * recip_finite(tile_density[k]);
                                let smooth_w = smoothing_poly6(r2);
                                let state_tile_base = k * MAX_STATE_DIMS;
                                for (var c = 0u; c < sd; c = c + 1u) {
                                    blur[c] = blur[c] + tile_states[state_tile_base + c] * smooth_w * volume_j;
                                }

                                if (idx != j) {
                                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                        let grad = spiky_gradient(
                                            delta,
                                            r2,
                                            density_gradient_weight(j),
                                            axis,
                                        );
                                        density_grad[axis] = density_grad[axis] + grad;
                                    }

                                    for (var c = 0u; c < sd; c = c + 1u) {
                                        let diff = tile_states[state_tile_base + c] - states.values[state_base + c];
                                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                            let grad = spiky_gradient(delta, r2, volume_j, axis);
                                            state_grad[c * MAX_DIMS + axis] =
                                                state_grad[c * MAX_DIMS + axis] + diff * grad;
                                        }
                                    }

                                    for (var row = 0u; row < dim; row = row + 1u) {
                                        for (var col = 0u; col < dim; col = col + 1u) {
                                            let grad = spiky_gradient(delta, r2, volume_j, col);
                                            moment[row * MAX_DIMS + col] =
                                                moment[row * MAX_DIMS + col] + delta[row] * grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    workgroupBarrier();
                }
            }
        }
    }

    if (!has_target) {
        return;
    }
    if (mask == 0.0) {
        copy_particle_to_output(idx);
        return;
    }
    finish_update_particle(idx, mask, pi, blur, state_grad, density_grad, moment);
}

@compute @workgroup_size(128)
fn write_gaussian_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= total_count()) {
        return;
    }
    write_gaussian_from_output(idx);
}

fn material_gaussian_geometry(idx: u32) -> MaterialGaussianGeometry {
    let measure = max(particle_measure(idx), 1.0e-20);
    let display_scale = max(display_scale_per_footprint(), 1.0e-20);
    let target_isotropic = max(material_render_target_footprint(idx) * display_scale, 1.0e-20);
    let transition_steps = render_transition_steps();
    var progress = 1.0;
    if (transition_steps > 0u) {
        let age = step_index() - min(step_index(), render_transition_start_step());
        progress = clamp(f32(age) / f32(transition_steps), 0.0, 1.0);
        progress = progress * progress * (3.0 - 2.0 * progress);
    }
    let initial_scale = max(material_render_from_scale(idx), 1.0e-20);
    let displayed_isotropic = exp2(mix(log2(initial_scale), log2(target_isotropic), progress));
    let raw_footprint = max(displayed_isotropic / display_scale, 1.0e-20);
    var represented = 3.141592653589793 * raw_footprint * raw_footprint;
    if (spatial_dims() == 3u) {
        represented = (4.0 * 3.141592653589793 / 3.0)
            * raw_footprint * raw_footprint * raw_footprint;
    }
    return MaterialGaussianGeometry(
        vec3<f32>(displayed_isotropic),
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        clamp(measure / max(represented, 1.0e-20), 0.001, 1.0),
    );
}

fn write_gaussian_from_output(idx: u32) {
    let pos = output_position(idx);
    let base4 = idx * 4u;
    gaussian_position_visibility.values[base4] = pos.x;
    gaussian_position_visibility.values[base4 + 1u] = pos.y;
    gaussian_position_visibility.values[base4 + 2u] = select(0.0, pos.z, spatial_dims() == 3u);
    gaussian_position_visibility.values[base4 + 3u] = 1.0;

    let tail_rgb = vec3<f32>(
        output_tail_state_channel(idx, 2u),
        output_tail_state_channel(idx, 1u),
        output_tail_state_channel(idx, 0u),
    );
    let color = clamp(tail_rgb + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    write_gaussian_sh0_color(idx, color);
    write_gaussian_geometry_from_output(idx);
}

fn write_gaussian_sh0_color(idx: u32, color: vec3<f32>) {
    let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
    for (var coeff = 0u; coeff < GAUSSIAN_SH_COEFF_COUNT; coeff = coeff + 1u) {
        gaussian_spherical_harmonic.values[sh_base + coeff] = 0.0;
    }
    gaussian_spherical_harmonic.values[sh_base] = (color.x - 0.5) / SH_C0;
    gaussian_spherical_harmonic.values[sh_base + 1u] = (color.y - 0.5) / SH_C0;
    gaussian_spherical_harmonic.values[sh_base + 2u] = (color.z - 0.5) / SH_C0;
}

fn write_gaussian_geometry_from_output(idx: u32) {
    let base4 = idx * 4u;
    var material_geometry = MaterialGaussianGeometry(
        vec3<f32>(max(eps() * 0.12, 0.00008)),
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        1.0,
    );
    if (material_enabled()) {
        material_geometry = material_gaussian_geometry(idx);
    }
    gaussian_rotation.values[base4] = material_geometry.rotation.x;
    gaussian_rotation.values[base4 + 1u] = material_geometry.rotation.y;
    gaussian_rotation.values[base4 + 2u] = material_geometry.rotation.z;
    gaussian_rotation.values[base4 + 3u] = material_geometry.rotation.w;

    gaussian_scale_opacity.values[base4] = material_geometry.scale.x;
    gaussian_scale_opacity.values[base4 + 1u] = material_geometry.scale.y;
    gaussian_scale_opacity.values[base4 + 2u] = material_geometry.scale.z;
    var particle_opacity = 1.0;
    if (spatial_dims() == 3u) {
        particle_opacity = clamp(sigmoid(output_material_opacity_logit(idx)), 0.001, 0.95);
    }
    gaussian_scale_opacity.values[base4 + 3u] =
        particle_opacity * material_geometry.opacity;
}
