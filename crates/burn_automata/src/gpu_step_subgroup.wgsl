@compute @workgroup_size(32)
fn subgroup_cooperative_density_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = workgroup.x;
    let local_id = local.x;
    if (idx >= total_count()) {
        return;
    }

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
                    if (r2 < eps2) {
                        rho = rho + smoothing_poly6(r2);
                    }
                }
            }
        }
    }

    let reduced = subgroupAdd(rho);
    if (local_id == 0u) {
        density.values[idx] = reduced;
    }
}

@compute @workgroup_size(32)
fn subgroup_cooperative_update_main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = workgroup.x;
    let local_id = local.x;
    if (idx >= total_count()) {
        return;
    }

    let sd = state_dims();
    let dim = spatial_dims();
    let mask = update_mask(idx);
    if (mask == 0.0) {
        if (local_id == 0u) {
            copy_particle_to_output(idx);
        }
        return;
    }

    let pi = position(idx);
    let center = cell_coords(pi);
    let eps2 = eps() * eps();
    let state_base = idx * sd;

    var blur: array<f32, MAX_STATE_DIMS>;
    var target_state: array<f32, MAX_STATE_DIMS>;
    var state_grad: array<f32, MAX_STATE_DIMS * MAX_DIMS>;
    var density_grad: array<f32, MAX_DIMS>;
    var moment: array<f32, MAX_DIMS * MAX_DIMS>;
    for (var c = 0u; c < sd; c = c + 1u) {
        target_state[c] = states.values[state_base + c];
    }

    for (var dz = z_min(); dz <= z_max(); dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let coords = vec3<i32>(center.x + dx, center.y + dy, center.z + dz);
                let cell = cell_index(coords);
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
                    if (r2 < eps2) {
                        let volume_j = recip_finite(density.values[j]);
                        let smooth_w = smoothing_poly6(r2);
                        let src = j * sd;
                        if (idx != j && r2 > 0.0) {
                            let r = sqrt(r2);
                            let e = eps() - r;
                            let grad_mag = spiky_coef() * 3.0 * e * e / r;
                            var grad_volume: array<f32, MAX_DIMS>;
                            for (var axis = 0u; axis < dim; axis = axis + 1u) {
                                let grad = grad_mag * delta[axis];
                                grad_volume[axis] = grad * volume_j;
                                density_grad[axis] = density_grad[axis] + grad;
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

    for (var c = 0u; c < MAX_STATE_DIMS; c = c + 1u) {
        coop_values[coop_component_index(COOP_BLUR_BASE + c, local_id)] = subgroupAdd(blur[c]);
    }
    for (var c = 0u; c < MAX_STATE_DIMS * MAX_DIMS; c = c + 1u) {
        coop_values[coop_component_index(COOP_STATE_GRAD_BASE + c, local_id)] = subgroupAdd(state_grad[c]);
    }
    for (var c = 0u; c < MAX_DIMS; c = c + 1u) {
        coop_values[coop_component_index(COOP_DENSITY_GRAD_BASE + c, local_id)] = subgroupAdd(density_grad[c]);
    }
    for (var c = 0u; c < MAX_DIMS * MAX_DIMS; c = c + 1u) {
        coop_values[coop_component_index(COOP_MOMENT_BASE + c, local_id)] = subgroupAdd(moment[c]);
    }
    workgroupBarrier();

    finish_update_particle_cooperative(local_id, idx, mask, pi);
}
