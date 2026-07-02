const MAX_STATE_DIMS: u32 = 24u;
const MAX_HIDDEN_DIMS: u32 = 256u;
const MAX_FEATURE_DIMS: u32 = 128u;
const MAX_OUTPUT_DIMS: u32 = 32u;
const MAX_DIMS: u32 = 3u;
const GAUSSIAN_SH_COEFF_COUNT: u32 = 48u;
const SH_C0: f32 = 0.28209479177387814;
const TILE_SIZE: u32 = 128u;
const SCAN_SIZE: u32 = 256u;
const LAYOUT_LINKED_LIST: u32 = 0u;
const LAYOUT_FIXED_BUCKETS: u32 = 1u;
const LAYOUT_SORTED_CELLS: u32 = 2u;
const LAYOUT_BVH: u32 = 3u;
const LAYOUT_SORTED_BVH: u32 = 4u;
const LAYOUT_MORTON_BVH: u32 = 5u;
const BVH_HEADER_U32: u32 = 4u;
const BVH_NODE_U32: u32 = 9u;
const BVH_STACK_SIZE: u32 = 64u;
const GROWTH_3D_MIN_OPACITY_LOGIT: f32 = -8.0;
const GROWTH_3D_MAX_OPACITY_LOGIT: f32 = 24.0;

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
    values: array<vec4<u32>, 9>,
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
var<workgroup> tile_mismatch: atomic<u32>;
var<workgroup> scan_values: array<u32, SCAN_SIZE>;

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

fn is_sorted_layout() -> bool {
    return neighbor_layout() == LAYOUT_SORTED_CELLS || neighbor_layout() == LAYOUT_SORTED_BVH;
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
    return select(0.0, 1.0, random01(idx, step_index(), random_seed()) < probability);
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
    var out = value;
    while (out < 0) {
        out = out + modulus;
    }
    while (out >= modulus) {
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
    return hash % cell_count();
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

fn cell_index(coords: vec3<i32>) -> i32 {
    if (is_particle_grid()) {
        return i32(particle_cell_hash(coords));
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
    return i32(hash);
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
    if (norm <= 1e-12) {
        return 0.0;
    }
    return value * log(1.0 + norm) / norm;
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
    let cell = u32(cell_index(cell_coords(position(idx))));
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
    let cell = u32(cell_index(cell_coords(position(idx))));
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
    let eps2 = eps() * eps();
    var rho = 0.0;

    for (var dz = z_min(); dz <= z_max(); dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index(coords);
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
                        if (r2 < eps2) {
                            rho = rho + smoothing_poly6(r2);
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
                        if (r2 < eps2) {
                            rho = rho + smoothing_poly6(r2);
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
                        if (r2 < eps2) {
                            rho = rho + smoothing_poly6(r2);
                        }
                    }
                }
            }
        }
    }

    return rho;
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
                    rho = rho + smoothing_poly6(r2);
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

    let target_cell = active_cell(workgroup.x);
    let local_id = local.x;
    let block_start = workgroup.y * TILE_SIZE;
    let target_count = min(atomicLoad(&linked_grid.values[target_cell]), bucket_capacity());
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

    if (local_id == 0u) {
        let first_idx = atomicLoad(&linked_grid.values[bucket_slot_index(target_cell, block_start)]);
        tile_center = cell_coords(position(first_idx));
        atomicStore(&tile_mismatch, 0u);
    }
    workgroupBarrier();

    if (has_target && !same_cell_coords(center, tile_center)) {
        atomicStore(&tile_mismatch, 1u);
    }
    workgroupBarrier();

    if (atomicLoad(&tile_mismatch) != 0u) {
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
                let coords = vec3<i32>(tile_center.x + dx, tile_center.y + dy, tile_center.z + dz);
                let cell = cell_index(coords);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let count = min(atomicLoad(&linked_grid.values[cell_u]), bucket_capacity());
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
                                rho = rho + smoothing_poly6(r2);
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

    if (mask == 0.0) {
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
                let cell = cell_index(coords);
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
                            let volume_j = recip_finite(density.values[j]);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(delta, r2, 1.0, axis);
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
                            let volume_j = recip_finite(density.values[j]);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(delta, r2, 1.0, axis);
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
                            let volume_j = recip_finite(density.values[j]);
                            let smooth_w = smoothing_poly6(r2);
                            let src = j * sd;
                            for (var c = 0u; c < sd; c = c + 1u) {
                                blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                            }

                            if (idx != j) {
                                for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                    let grad = spiky_gradient(delta, r2, 1.0, axis);
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
                    let volume_j = recip_finite(density.values[j]);
                    let smooth_w = smoothing_poly6(r2);
                    let src = j * sd;
                    for (var c = 0u; c < sd; c = c + 1u) {
                        blur[c] = blur[c] + states.values[src + c] * smooth_w * volume_j;
                    }

                    if (idx != j) {
                        for (var axis = 0u; axis < dim; axis = axis + 1u) {
                            let grad = spiky_gradient(delta, r2, 1.0, axis);
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
            corrected[out_axis] = value * grad_scale();
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
        density_grad[axis] = density_grad[axis] * density_scale();
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
            sum = sum + weights.values[w_base + h] * hidden[h];
        }
        update[o] = sum;
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
                + mask * dt() * alpha() * update[axis] * motion_eps() / (1.0 + update_norm);
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

    let target_cell = active_cell(workgroup.x);
    let local_id = local.x;
    let block_start = workgroup.y * TILE_SIZE;
    let target_count = min(atomicLoad(&linked_grid.values[target_cell]), bucket_capacity());
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

    if (local_id == 0u) {
        let first_idx = atomicLoad(&linked_grid.values[bucket_slot_index(target_cell, block_start)]);
        tile_center = cell_coords(position(first_idx));
        atomicStore(&tile_mismatch, 0u);
    }
    workgroupBarrier();

    if (has_target && !same_cell_coords(center, tile_center)) {
        atomicStore(&tile_mismatch, 1u);
    }
    workgroupBarrier();

    if (atomicLoad(&tile_mismatch) != 0u) {
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
                let coords = vec3<i32>(tile_center.x + dx, tile_center.y + dy, tile_center.z + dz);
                let cell = cell_index(coords);
                if (cell < 0) {
                    continue;
                }
                let cell_u = u32(cell);
                let count = min(atomicLoad(&linked_grid.values[cell_u]), bucket_capacity());
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
                                let volume_j = recip_finite(tile_density[k]);
                                let smooth_w = smoothing_poly6(r2);
                                let state_tile_base = k * MAX_STATE_DIMS;
                                for (var c = 0u; c < sd; c = c + 1u) {
                                    blur[c] = blur[c] + tile_states[state_tile_base + c] * smooth_w * volume_j;
                                }

                                if (idx != j) {
                                    for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                        let grad = spiky_gradient(delta, r2, 1.0, axis);
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

fn write_gaussian_from_output(idx: u32) {
    let pos = output_position(idx);
    let base4 = idx * 4u;
    gaussian_position_visibility.values[base4] = pos.x;
    gaussian_position_visibility.values[base4 + 1u] = pos.y;
    gaussian_position_visibility.values[base4 + 2u] = select(0.0, pos.z, spatial_dims() == 3u);
    gaussian_position_visibility.values[base4 + 3u] = 1.0;

    let sh_base = idx * GAUSSIAN_SH_COEFF_COUNT;
    for (var coeff = 0u; coeff < GAUSSIAN_SH_COEFF_COUNT; coeff = coeff + 1u) {
        gaussian_spherical_harmonic.values[sh_base + coeff] = 0.0;
    }
    let tail_rgb = vec3<f32>(
        output_tail_state_channel(idx, 2u),
        output_tail_state_channel(idx, 1u),
        output_tail_state_channel(idx, 0u),
    );
    let color = clamp(tail_rgb + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    gaussian_spherical_harmonic.values[sh_base] = (color.x - 0.5) / SH_C0;
    gaussian_spherical_harmonic.values[sh_base + 1u] = (color.y - 0.5) / SH_C0;
    gaussian_spherical_harmonic.values[sh_base + 2u] = (color.z - 0.5) / SH_C0;

    gaussian_rotation.values[base4] = 1.0;
    gaussian_rotation.values[base4 + 1u] = 0.0;
    gaussian_rotation.values[base4 + 2u] = 0.0;
    gaussian_rotation.values[base4 + 3u] = 0.0;

    let particle_scale = max(eps() * 0.12, 0.00008);
    gaussian_scale_opacity.values[base4] = particle_scale;
    gaussian_scale_opacity.values[base4 + 1u] = particle_scale;
    gaussian_scale_opacity.values[base4 + 2u] = particle_scale;
    var particle_opacity = 1.0;
    if (spatial_dims() == 3u) {
        particle_opacity = clamp(sigmoid(output_material_opacity_logit(idx)), 0.001, 0.95);
    }
    gaussian_scale_opacity.values[base4 + 3u] = particle_opacity;
}
