fn adaptive_count_support_candidates_subgroup(
    idx: u32,
    local_id: u32,
    pi: vec4<f32>,
    center: vec3<i32>,
    _cell_radius: i32,
) -> u32 {
    var local_count = 0u;
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
                    local_count = local_count + 1u;
                }
            }
        }
    }
    }
    return subgroupAdd(local_count);
}

fn adaptive_spacing_occupancy_subgroup(
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
    return subgroupAdd(occupancy);
}

@compute @workgroup_size(32)
fn subgroup_cooperative_density_main(
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
    let support_cell_radius = adaptive_support_cell_radius(idx);
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
                        if (particle_bandwidth(j) > eps() * (1.0 + 32.0 * 1.1920929e-7)) {
                            coarse_rho = coarse_rho + contribution;
                        }
                    }
                }
            }
        }
    }
    }

    let reduced = subgroupAdd(rho);
    let reduced_coarse = subgroupAdd(coarse_rho);
    if (local_id == 0u) {
        density.values[idx] = reduced;
        density.values[diagnostics_coarse_exposure_offset() + idx] = clamp(
            reduced_coarse / max(reduced, 1.0e-20),
            0.0,
            1.0,
        );
    }
}

@compute @workgroup_size(32)
fn subgroup_adaptive_local_residual_main(
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
    let position_base = idx * 4u;
    let state_base = idx * sd;
    let update_active = update_mask(idx) != 0.0;
    let perception_active = update_active || adaptive_diagnostics_enabled();

    let pi = position(idx);
    let center = cell_coords(pi);
    let support_cell_radius = adaptive_support_cell_radius(idx);
    let max_neighbors = adaptive_max_neighbors();
    if (local_id == 0u) {
        adaptive_cutoff_distance = 0xffffffffu;
        adaptive_cutoff_index = 0xffffffffu;
    }
    workgroupBarrier();
    var support_count = 0u;
    if (perception_active && (max_neighbors > 0u || adaptive_diagnostics_enabled())) {
        support_count = adaptive_count_support_candidates_subgroup(
            idx,
            local_id,
            pi,
            center,
            support_cell_radius,
        );
    }

    var observed_spacing = adaptive_spacing_max();
    if (adaptive_diagnostics_enabled()) {
        let spacing_lo = adaptive_spacing_min();
        var spacing_hi = spacing_lo;
        var max_occupancy = adaptive_spacing_occupancy_subgroup(
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
            max_occupancy = adaptive_spacing_occupancy_subgroup(
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
                let occupancy = adaptive_spacing_occupancy_subgroup(
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
        if (adaptive_selection_rank == 0u) {
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
        let reduced_density = subgroupAdd(inactive_density);
        if (local_id == 0u) {
            density.values[idx] = reduced_density;
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
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var occupancy_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    var rho = 0.0;
    if (local_id == 0u) {
        for (var channel = 0u; channel < sd; channel = channel + 1u) {
            normalized_state[channel] = shepard * states.values[state_base + channel];
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

    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        let reduced = subgroupAdd(normalized_state[channel]);
        if (local_id == 0u) {
            coop_reduced_values[COOP_BLUR_BASE + channel] = reduced;
        }
    }
    for (var channel = 0u; channel < sd; channel = channel + 1u) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            let component = channel * MAX_DIMS + axis;
            let reduced = subgroupAdd(state_grad[component]);
            if (local_id == 0u) {
                coop_reduced_values[COOP_STATE_GRAD_BASE + component] = reduced;
            }
        }
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        let reduced = subgroupAdd(occupancy_grad[axis]);
        if (local_id == 0u) {
            coop_reduced_values[COOP_DENSITY_GRAD_BASE + axis] = reduced;
        }
    }
    let reduced_partition = subgroupAdd(shepard_sum);
    if (local_id == 0u) {
        coop_reduced_values[COOP_DENSITY_GRAD_BASE + 2u] = reduced_partition;
    }
    for (var row = 0u; row < dim; row = row + 1u) {
        for (var col = 0u; col < dim; col = col + 1u) {
            let component = row * MAX_DIMS + col;
            let reduced = subgroupAdd(moment[component]);
            if (local_id == 0u) {
                coop_reduced_values[COOP_MOMENT_BASE + component] = reduced;
            }
        }
    }
    let reduced_density = subgroupAdd(rho);
    if (local_id == 0u) {
        coop_reduced_values[COOP_MOMENT_BASE + 8u] = reduced_density;
    }
    workgroupBarrier();
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

@compute @workgroup_size(32)
fn subgroup_cooperative_update_main(
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
    let mask = update_mask(idx);
    if (mask == 0.0 && !adaptive_diagnostics_enabled()) {
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
        let reduced = subgroupAdd(blur[c]);
        if (local_id == 0u) {
            coop_reduced_values[COOP_BLUR_BASE + c] = reduced;
        }
    }
    for (var c = 0u; c < sd; c = c + 1u) {
        for (var axis = 0u; axis < dim; axis = axis + 1u) {
            let component = c * MAX_DIMS + axis;
            let reduced = subgroupAdd(state_grad[component]);
            if (local_id == 0u) {
                coop_reduced_values[COOP_STATE_GRAD_BASE + component] = reduced;
            }
        }
    }
    for (var axis = 0u; axis < dim; axis = axis + 1u) {
        let reduced = subgroupAdd(density_grad[axis]);
        if (local_id == 0u) {
            coop_reduced_values[COOP_DENSITY_GRAD_BASE + axis] = reduced;
        }
    }
    for (var row = 0u; row < dim; row = row + 1u) {
        for (var col = 0u; col < dim; col = col + 1u) {
            let component = row * MAX_DIMS + col;
            let reduced = subgroupAdd(moment[component]);
            if (local_id == 0u) {
                coop_reduced_values[COOP_MOMENT_BASE + component] = reduced;
            }
        }
    }
    workgroupBarrier();

    finish_update_particle_cooperative(local_id, idx, mask, pi);
}
