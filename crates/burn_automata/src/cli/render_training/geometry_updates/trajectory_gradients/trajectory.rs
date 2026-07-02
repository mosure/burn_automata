#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn surface_escape_weight(
    distance: f32,
    threshold: f32,
    surface_escape_gain: f32,
) -> f32 {
    if surface_escape_gain <= 0.0
        || !surface_escape_gain.is_finite()
        || !distance.is_finite()
        || !threshold.is_finite()
        || threshold <= 1.0e-6
        || distance <= threshold
    {
        return 1.0;
    }
    let escape_ratio = (distance / threshold - 1.0).max(0.0);
    (1.0 + surface_escape_gain * escape_ratio).min(8.0)
}

pub(crate) fn trajectory_render_adjoints(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    trajectory: &[RenderTrajectorySnapshot],
    trace: &crate::RolloutTrace,
    cfg: &RenderProxyTrainingConfig,
) -> Result<Vec<Option<RenderTrajectoryAdjoint>>, Box<dyn std::error::Error>> {
    let mut adjoints = (0..trajectory.len()).map(|_| None).collect::<Vec<_>>();
    let mesh_enabled = cfg.trajectory_mesh_gain > 0.0
        && cfg.trajectory_mesh_gain.is_finite()
        && (cfg.coverage_gain > 0.0 || cfg.surface_gain > 0.0);
    let liveness_enabled = cfg.liveness_gain > 0.0 && cfg.liveness_gain.is_finite();
    let render_enabled = cfg.trajectory_render_samples > 0
        && cfg.trajectory_render_gain > 0.0
        && cfg.trajectory_render_gain.is_finite();
    if !render_enabled && !mesh_enabled && !liveness_enabled {
        return Ok(adjoints);
    }

    let render_mesh_enabled = render_enabled || mesh_enabled;
    let render_mesh_sample_budget = if cfg.trajectory_render_samples > 0 {
        cfg.trajectory_render_samples
    } else {
        trajectory
            .len()
            .clamp(1, ROBUST_3D_TRAJECTORY_RENDER_SAMPLES)
    };
    let render_mesh_indices = if render_mesh_enabled {
        trajectory_render_sample_indices(trajectory.len(), render_mesh_sample_budget)
    } else {
        Vec::new()
    };
    let liveness_indices = if liveness_enabled {
        trajectory_liveness_sample_indices(trajectory.len(), render_mesh_sample_budget)
    } else {
        Vec::new()
    };
    let mut indices = render_mesh_indices.clone();
    for index in &liveness_indices {
        if !indices.contains(index) {
            indices.push(*index);
        }
    }
    indices.sort_unstable();
    if indices.is_empty() {
        return Ok(adjoints);
    }
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let render_mesh_sample_count = render_mesh_indices.len().max(1) as f32;
    let liveness_sample_count = liveness_indices.len().max(1) as f32;
    let liveness_sample_weight = 1.0 / liveness_sample_count.sqrt();
    let liveness_update_cap =
        liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier);

    for index in indices {
        let render_mesh_sampled = render_mesh_indices.contains(&index);
        let liveness_sampled = liveness_indices.contains(&index);
        let snapshot = &trajectory[index];
        let snapshot_trace = crate::RolloutTrace {
            positions: snapshot.positions.clone(),
            states: snapshot.states.clone(),
            batch_size: trace.batch_size,
            particle_count: trace.particle_count,
            state_dims: trace.state_dims,
            steps: ((snapshot.step_fraction * trace.steps.max(1) as f32).round() as usize).max(1),
            mean_dx: Vec::new(),
        };
        let mut state = vec![0.0; snapshot_trace.states.len()];
        let mut position = vec![[0.0; 4]; snapshot_trace.positions.len()];

        if render_enabled && render_mesh_sampled {
            let gradient = render_position_gradient(&snapshot_trace, target, render_cfg, cfg)?;
            let rows = gradient
                .gradients
                .len()
                .min(gradient.row_indices.len())
                .min(gradient.opacity_gradients.len())
                .min(gradient.scale_gradients.len())
                .min(gradient.color_gradients.len());
            if rows > 0 {
                let terminal_row_weights = terminal_render_locality_weights(
                    config,
                    &snapshot_trace.positions,
                    &snapshot_trace.states,
                    cfg.liveness_front_radius,
                );
                state = terminal_render_state_adjoint_weighted(
                    config,
                    &snapshot_trace,
                    &gradient,
                    cfg.opacity_gain,
                    cfg.scale_gain,
                    cfg.scale_budget_weight,
                    0.0,
                    cfg.liveness_front_radius,
                    snapshot.step_fraction,
                    cfg.material_max_opacity_update,
                    cfg.render,
                    rows,
                    Some(&terminal_row_weights),
                );
                let zero_coverage_updates = vec![[0.0_f32; 3]; snapshot_trace.positions.len()];
                position = terminal_render_position_adjoint_weighted(
                    config,
                    &snapshot_trace,
                    &gradient,
                    &zero_coverage_updates,
                    cfg.motion_gain,
                    false,
                    rows,
                    Some(&terminal_row_weights),
                );
                let render_weight = cfg.trajectory_render_gain * snapshot.step_fraction.powi(2)
                    / render_mesh_sample_count;
                scale_state_adjoint(&mut state, render_weight);
                scale_position_adjoint(&mut position, render_weight, config.spatial_dims);
            }
        }

        if mesh_enabled && render_mesh_sampled {
            let coverage_updates = render_proxy_target_coverage_updates(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.coverage_gain,
                cfg.coverage_samples,
                cfg.max_update_norm,
                cfg.coverage_mode,
                cfg.coverage_softness,
                cfg.coverage_repulsion_gain,
                cfg.coverage_gap_gain,
                cfg.coverage_repulsion_radius,
                cfg.coverage_normal_weight,
                cfg.seed_scale,
            );
            let mut mesh_position = vec![[0.0_f32; 4]; snapshot_trace.positions.len()];
            for particle_row in 0..mesh_position.len() {
                for axis in 0..config.spatial_dims {
                    mesh_position[particle_row][axis] -= coverage_updates
                        .get(particle_row)
                        .map(|update| update[axis])
                        .unwrap_or(0.0);
                }
                clamp_position_adjoint_row(&mut mesh_position[particle_row], config.spatial_dims);
            }
            add_surface_position_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.surface_gain,
                cfg.surface_escape_gain,
                &mut mesh_position,
            );
            add_material_visible_surface_position_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.surface_gain,
                cfg.surface_escape_gain,
                cfg.seed_scale,
                cfg.liveness_front_radius,
                &mut mesh_position,
            );
            add_material_visible_surface_coverage_position_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.coverage_gain,
                cfg.coverage_samples,
                cfg.max_update_norm,
                cfg.coverage_mode,
                cfg.coverage_softness,
                cfg.coverage_repulsion_gain,
                cfg.coverage_gap_gain,
                cfg.coverage_repulsion_radius,
                cfg.coverage_normal_weight,
                cfg.seed_scale,
                cfg.liveness_front_radius,
                &mut mesh_position,
            );
            let mesh_weight = cfg.trajectory_mesh_gain * snapshot.step_fraction.powi(2)
                / render_mesh_sample_count;
            scale_position_adjoint(&mut mesh_position, mesh_weight, config.spatial_dims);
            for particle_row in 0..position.len().min(mesh_position.len()) {
                for axis in 0..config.spatial_dims {
                    position[particle_row][axis] += mesh_position[particle_row][axis];
                }
                clamp_position_adjoint_row(&mut position[particle_row], config.spatial_dims);
            }
        }

        if liveness_enabled && liveness_sampled {
            add_liveness_front_state_adjoint(
                config,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.liveness_gain * liveness_sample_weight,
                cfg.liveness_front_radius,
                snapshot.step_fraction,
                liveness_update_cap,
                &mut state,
            );
            add_temporal_activation_schedule_state_adjoint(
                config,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.liveness_gain * liveness_sample_weight,
                cfg.liveness_front_radius,
                snapshot.step_fraction,
                liveness_update_cap,
                &mut state,
            );
        }

        if mesh_enabled && render_mesh_sampled {
            let mesh_weight = cfg.trajectory_mesh_gain * snapshot.step_fraction.powi(2)
                / render_mesh_sample_count;
            let mut material_state = vec![0.0_f32; snapshot_trace.states.len()];
            add_surface_material_opacity_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.opacity_gain,
                cfg.seed_scale,
                GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
                cfg.material_max_opacity_update,
                &mut material_state,
            );
            add_material_target_coverage_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.opacity_gain,
                cfg.coverage_samples,
                cfg.seed_scale,
                GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
                cfg.material_max_opacity_update,
                &mut material_state,
            );
            add_material_surface_strata_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.opacity_gain,
                cfg.coverage_samples,
                cfg.seed_scale,
                GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
                cfg.material_max_opacity_update,
                &mut material_state,
            );
            add_material_liveness_state_adjoint(
                config,
                &snapshot_trace.states,
                cfg.material_liveness_gain,
                material_suppression_max_update(
                    cfg.material_max_opacity_update,
                    cfg.material_suppression_update_multiplier,
                ),
                &mut material_state,
            );
            add_material_visible_liveness_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.material_liveness_gain,
                target_coverage_threshold(cfg.seed_scale),
                liveness_update_cap,
                &mut material_state,
            );
            add_material_visible_surface_tail_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.material_tail_gain,
                GROWTH_3D_SURFACE_MAX_DISTANCE,
                material_suppression_max_update(
                    cfg.material_max_opacity_update,
                    cfg.material_suppression_update_multiplier,
                ),
                &mut material_state,
            );
            scale_state_adjoint(&mut material_state, mesh_weight);
            for value_idx in 0..state.len().min(material_state.len()) {
                state[value_idx] += material_state[value_idx];
            }

            let mut escape_state = vec![0.0_f32; snapshot_trace.states.len()];
            add_surface_escape_state_adjoint(
                config,
                target,
                &snapshot_trace.positions,
                &snapshot_trace.states,
                cfg.surface_escape_gain,
                cfg.opacity_gain,
                cfg.liveness_gain,
                cfg.material_max_opacity_update,
                &mut escape_state,
            );
            scale_state_adjoint(&mut escape_state, mesh_weight);
            for value_idx in 0..state.len().min(escape_state.len()) {
                if escape_state[value_idx] > 0.0 {
                    state[value_idx] = state[value_idx].max(escape_state[value_idx]);
                }
            }
        }

        adjoints[index] = Some(RenderTrajectoryAdjoint {
            state,
            position,
            weight: 1.0,
        });
    }

    if liveness_enabled && liveness_indices.len() > 1 {
        for pair in liveness_indices.windows(2) {
            let previous_idx = pair[0];
            let current_idx = pair[1];
            if previous_idx >= current_idx || current_idx >= trajectory.len() {
                continue;
            }
            let previous_snapshot = &trajectory[previous_idx];
            let current_snapshot = &trajectory[current_idx];
            let (before_current, from_current) = adjoints.split_at_mut(current_idx);
            if before_current[previous_idx].is_none() {
                before_current[previous_idx] = Some(RenderTrajectoryAdjoint {
                    state: vec![0.0; previous_snapshot.states.len()],
                    position: vec![[0.0; 4]; previous_snapshot.positions.len()],
                    weight: 1.0,
                });
            }
            if from_current[0].is_none() {
                from_current[0] = Some(RenderTrajectoryAdjoint {
                    state: vec![0.0; current_snapshot.states.len()],
                    position: vec![[0.0; 4]; current_snapshot.positions.len()],
                    weight: 1.0,
                });
            }
            let previous_adjoint = before_current[previous_idx]
                .as_mut()
                .expect("previous liveness adjoint should exist");
            let current_adjoint = from_current[0]
                .as_mut()
                .expect("current liveness adjoint should exist");
            add_temporal_activation_jump_state_adjoint(
                config,
                &previous_snapshot.positions,
                &previous_snapshot.states,
                &current_snapshot.states,
                cfg.liveness_gain * liveness_sample_weight,
                cfg.liveness_front_radius,
                previous_snapshot.step_fraction,
                current_snapshot.step_fraction,
                liveness_update_cap,
                &mut previous_adjoint.state,
                &mut current_adjoint.state,
            );
        }
    }

    Ok(adjoints)
}
