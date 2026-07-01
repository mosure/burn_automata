#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn terminal_render_position_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    coverage_updates: &[[f32; 3]],
    motion_gain: f32,
    full_coverage_adjoint: bool,
    rows: usize,
) -> Vec<[f32; 4]> {
    let mut position_adjoint = vec![[0.0; 4]; trace.positions.len()];
    if full_coverage_adjoint {
        for particle_row in 0..position_adjoint.len() {
            for axis in 0..config.spatial_dims {
                let coverage = coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0);
                position_adjoint[particle_row][axis] -= coverage;
            }
            clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
        }
    }
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row >= position_adjoint.len() {
            continue;
        }
        for axis in 0..config.spatial_dims {
            let coverage = if full_coverage_adjoint {
                0.0
            } else {
                coverage_updates
                    .get(particle_row)
                    .map(|update| update[axis])
                    .unwrap_or(0.0)
            };
            position_adjoint[particle_row][axis] +=
                motion_gain * gradient.gradients[gradient_row][axis] - coverage;
        }
        clamp_position_adjoint_row(&mut position_adjoint[particle_row], config.spatial_dims);
    }
    position_adjoint
}

pub(crate) fn add_surface_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return;
    }
    let updates = render_proxy_surface_projection_updates(
        config,
        target,
        positions,
        states,
        surface_gain,
        surface_escape_gain,
        f32::INFINITY,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

pub(crate) fn add_material_visible_surface_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    seed_scale: f32,
    front_radius: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return;
    }
    let updates = material_visible_surface_approach_updates(
        config,
        target,
        positions,
        states,
        None,
        surface_gain,
        surface_escape_gain,
        f32::INFINITY,
        seed_scale,
        front_radius,
        None,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_coverage_position_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    front_radius: f32,
    position_adjoint: &mut [[f32; 4]],
) {
    let updates = material_visible_surface_coverage_updates(
        config,
        target,
        positions,
        states,
        None,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
        front_radius,
        None,
    );
    for (row, update) in updates.iter().enumerate() {
        if row >= position_adjoint.len() {
            break;
        }
        for axis in 0..config.spatial_dims {
            position_adjoint[row][axis] -= update[axis];
        }
        clamp_position_adjoint_row(&mut position_adjoint[row], config.spatial_dims);
    }
}

pub(crate) fn render_proxy_surface_projection_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0; 3]; positions.len()];
    if surface_gain <= 0.0 || !surface_gain.is_finite() {
        return updates;
    }
    for (row, position) in positions.iter().enumerate() {
        if config.state_dims > 3
            && states
                .get(row * config.state_dims + 3)
                .is_some_and(|opacity| *opacity <= -1.0)
        {
            continue;
        }
        let projection = target.project(position3(*position));
        let weight = surface_escape_weight(
            projection.distance,
            GROWTH_3D_SURFACE_MAX_DISTANCE,
            surface_escape_gain,
        );
        updates[row] = [
            surface_gain * weight * projection.residual[0],
            surface_gain * weight * projection.residual[1],
            surface_gain * weight * projection.residual[2],
        ];
        clamp_update_row(&mut updates, row, max_update_norm);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_surface_row_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut row_weights = vec![0.0; positions.len()];
    if positions.is_empty()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len() * config.state_dims
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
    {
        return row_weights;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return row_weights;
    };
    let output_dims = config.update_dims();
    let material_output = config.spatial_dims + material_channel;
    if raw_updates.is_some_and(|updates| updates.len() < positions.len() * output_dims)
        || material_output >= output_dims
    {
        return row_weights;
    }

    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        let predicted_material = raw_updates
            .and_then(|updates| updates.get(row * output_dims + material_output))
            .map(|update| material_opacity + *update)
            .unwrap_or(material_opacity);
        let visible_logit = material_opacity.max(predicted_material);
        if visible_logit <= material_visible_threshold {
            continue;
        }
        let material_weight = ((visible_logit - material_visible_threshold) / 4.0).clamp(0.25, 1.0);
        row_weights[row] = if liveness > -1.0 {
            material_weight
        } else {
            let front_weight = front_weights
                .as_ref()
                .and_then(|weights| weights.get(row))
                .copied()
                .unwrap_or(0.0);
            let activation_weight = activation_candidate_weights
                .and_then(|weights| weights.get(row))
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            material_weight * front_weight * activation_weight
        };
    }
    row_weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_surface_candidate_row_weights(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut row_weights = material_visible_surface_row_weights(
        config,
        positions,
        states,
        raw_updates,
        front_radius,
        activation_candidate_weights,
    );
    if positions.is_empty()
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len().saturating_mul(config.state_dims)
        || activation_candidate_weights.is_some_and(|weights| weights.len() < positions.len())
    {
        return row_weights;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return row_weights;
    };
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    if liveness_output >= output_dims
        || raw_updates.is_some_and(|updates| updates.len() < positions.len() * output_dims)
    {
        return row_weights;
    }

    let front_weights = if front_radius > 0.0 && front_radius.is_finite() {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    let target_span =
        (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_visible_threshold).max(1.0e-6);
    for (row, position) in positions.iter().enumerate() {
        if row_weights[row] >= 1.0 {
            continue;
        }
        let state_base = row * config.state_dims;
        let output_base = row * output_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let predicted_liveness = raw_updates
            .and_then(|updates| updates.get(output_base + liveness_output))
            .map(|update| liveness + *update)
            .unwrap_or(liveness);
        let front_weight = front_weights
            .as_ref()
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(0.0);
        let activation_weight = activation_candidate_weights
            .and_then(|weights| weights.get(row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let activity_weight = if liveness > -1.0 {
            1.0
        } else if predicted_liveness > -1.0 {
            (0.5 + 0.5 * activation_weight).clamp(0.0, 1.0)
        } else {
            front_weight * activation_weight
        };
        if activity_weight <= 1.0e-3 || !activity_weight.is_finite() {
            continue;
        }

        let projection = target.project(position3(*position));
        let surface_weight =
            soft_material_assignment_weight(projection.distance, strict_threshold, soft_threshold);
        if surface_weight <= 0.0 {
            continue;
        }
        let material = states[state_base + material_channel];
        let material_progress =
            ((material - material_visible_threshold) / target_span).clamp(0.0, 1.0);
        let candidate_floor = 0.25 + 0.25 * material_progress;
        row_weights[row] = row_weights[row].max(
            (candidate_floor * activity_weight * surface_weight)
                .clamp(0.0, 1.0)
                .min(0.5),
        );
    }
    row_weights
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_surface_coverage_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<[f32; 3]> {
    let row_weights = material_surface_candidate_row_weights(
        config,
        target,
        positions,
        states,
        raw_updates,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
    render_proxy_weighted_target_coverage_updates(
        target,
        positions,
        &row_weights,
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_surface_approach_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: Option<&[f32]>,
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
) -> Vec<[f32; 3]> {
    let mut updates = vec![[0.0; 3]; positions.len()];
    if positions.is_empty()
        || config.spatial_dims == 0
        || config.spatial_dims > 3
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.len() < positions.len() * config.state_dims
        || surface_gain <= 0.0
        || !surface_gain.is_finite()
    {
        return updates;
    }
    let row_weights = material_surface_candidate_row_weights(
        config,
        target,
        positions,
        states,
        raw_updates,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
    for (row, position) in positions.iter().enumerate() {
        let row_weight = row_weights[row];
        if row_weight <= 1.0e-3 || !row_weight.is_finite() {
            continue;
        }

        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() {
            continue;
        }
        let surface_weight = surface_escape_weight(
            projection.distance,
            GROWTH_3D_SURFACE_MAX_DISTANCE,
            surface_escape_gain,
        );
        for axis in 0..config.spatial_dims {
            updates[row][axis] =
                surface_gain * row_weight * surface_weight * projection.residual[axis];
        }
        clamp_update_row(&mut updates, row, max_update_norm);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_approach_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    surface_gain: f32,
    surface_escape_gain: f32,
    max_update_norm: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if weight <= 0.0
        || !weight.is_finite()
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
    {
        return;
    }
    let updates = material_visible_surface_approach_updates(
        config,
        target,
        positions,
        states,
        Some(raw_updates),
        surface_gain,
        surface_escape_gain,
        max_update_norm,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
    for (row, update) in updates.iter().enumerate() {
        if update
            .iter()
            .take(config.spatial_dims)
            .all(|value| value.abs() <= 1.0e-8)
        {
            continue;
        }
        let output_base = row * output_dims;
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - update[axis]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_coverage_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    front_radius: f32,
    activation_candidate_weights: Option<&[f32]>,
    weight: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    if weight <= 0.0
        || !weight.is_finite()
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
    {
        return;
    }
    let updates = material_visible_surface_coverage_updates(
        config,
        target,
        positions,
        states,
        Some(raw_updates),
        coverage_gain,
        coverage_samples,
        max_update_norm,
        coverage_mode,
        coverage_softness,
        coverage_repulsion_gain,
        coverage_gap_gain,
        coverage_repulsion_radius,
        coverage_normal_weight,
        seed_scale,
        front_radius,
        activation_candidate_weights,
    );
    for (row, update) in updates.iter().enumerate() {
        if update
            .iter()
            .take(config.spatial_dims)
            .all(|value| value.abs() <= 1.0e-8)
        {
            continue;
        }
        let output_base = row * output_dims;
        for axis in 0..config.spatial_dims {
            output_gradients[output_base + axis] +=
                weight * (raw_updates[output_base + axis] - update[axis]);
        }
    }
}

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
                state = terminal_render_state_adjoint(
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
                );
                let zero_coverage_updates = vec![[0.0_f32; 3]; snapshot_trace.positions.len()];
                position = terminal_render_position_adjoint(
                    config,
                    &snapshot_trace,
                    &gradient,
                    &zero_coverage_updates,
                    cfg.motion_gain,
                    false,
                    rows,
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
                target_coverage_threshold(cfg.seed_scale),
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

pub(crate) fn scale_state_adjoint(state: &mut [f32], weight: f32) {
    if weight == 1.0 {
        return;
    }
    for value in state {
        *value *= weight;
    }
}

pub(crate) fn scale_position_adjoint(position: &mut [[f32; 4]], weight: f32, spatial_dims: usize) {
    if weight == 1.0 {
        return;
    }
    for row in position {
        for value in row.iter_mut().take(spatial_dims) {
            *value *= weight;
        }
    }
}

pub(crate) fn zero_supervised_gradients(model: &NpaModel) -> SupervisedGradients {
    SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: Vec::new(),
    }
}

pub(crate) fn accumulate_supervised_gradients(
    total: &mut SupervisedGradients,
    step: &SupervisedGradients,
) {
    add_assign_slice(&mut total.w1, &step.w1);
    add_assign_slice(&mut total.b1, &step.b1);
    add_assign_slice(&mut total.w2, &step.w2);
    add_assign_slice(&mut total.b2, &step.b2);
    total.features.extend_from_slice(&step.features);
}

#[cfg(test)]
pub(crate) fn normalize_supervised_gradients_by_rows(
    gradients: &mut SupervisedGradients,
    input_dims: usize,
) {
    if input_dims == 0
        || gradients.features.is_empty()
        || gradients.features.len() % input_dims != 0
    {
        return;
    }
    let rows = gradients.features.len() / input_dims;
    if rows == 0 {
        return;
    }
    let scale = 1.0 / rows as f32;
    scale_slice(&mut gradients.w1, scale);
    scale_slice(&mut gradients.b1, scale);
    scale_slice(&mut gradients.w2, scale);
    scale_slice(&mut gradients.b2, scale);
}

pub(crate) fn normalize_direct_rollout_gradients(
    gradients: &mut SupervisedGradients,
    input_dims: usize,
) {
    if input_dims == 0
        || gradients.features.is_empty()
        || gradients.features.len() % input_dims != 0
    {
        return;
    }
    let rows = gradients.features.len() / input_dims;
    if rows == 0 {
        return;
    }
    let exponent = DIRECT_ROLLOUT_GRADIENT_ROW_NORMALIZATION_EXPONENT.clamp(0.0, 1.0);
    let scale = 1.0 / (rows as f32).powf(exponent);
    scale_slice(&mut gradients.w1, scale);
    scale_slice(&mut gradients.b1, scale);
    scale_slice(&mut gradients.w2, scale);
    scale_slice(&mut gradients.b2, scale);
}

pub(crate) fn retain_material_output_gradients(
    model: &NpaModel,
    gradients: &mut SupervisedGradients,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) else {
        return Err(std::io::Error::other(
            "material-output-only training requires a material opacity channel",
        )
        .into());
    };
    gradients.w1.fill(0.0);
    gradients.b1.fill(0.0);
    let output_dims = model.config.update_dims();
    let material_output = model.config.spatial_dims + material_channel;
    for output in 0..output_dims {
        if output == material_output {
            continue;
        }
        let start = output * model.config.hidden_dims;
        let end = start + model.config.hidden_dims;
        gradients.w2[start..end].fill(0.0);
        gradients.b2[output] = 0.0;
    }
    Ok(())
}

pub(crate) fn add_assign_slice(total: &mut [f32], step: &[f32]) {
    debug_assert_eq!(total.len(), step.len());
    for (dst, src) in total.iter_mut().zip(step.iter()) {
        *dst += *src;
    }
}

pub(crate) fn scale_slice(values: &mut [f32], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for value in values {
        *value *= scale;
    }
}

pub(crate) fn clamp_state_adjoint_row(row: &mut [f32]) {
    const MAX_STATE_ADJOINT_NORM: f32 = 10.0;
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= MAX_STATE_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_STATE_ADJOINT_NORM / norm;
    for value in row {
        *value *= scale;
    }
}

pub(crate) fn clamp_position_adjoint_row(row: &mut [f32; 4], spatial_dims: usize) {
    const MAX_POSITION_ADJOINT_NORM: f32 = 10.0;
    let norm = row
        .iter()
        .take(spatial_dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= MAX_POSITION_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_POSITION_ADJOINT_NORM / norm;
    for value in row.iter_mut().take(spatial_dims) {
        *value *= scale;
    }
}

pub(crate) fn accumulate_motion_output_gradient(
    config: &NpaConfig,
    grid_eps: f32,
    raw_update: &[f32],
    dloss_ddx: [f32; 3],
    output_gradient: &mut [f32],
) {
    let dims = config.spatial_dims;
    let motion_scale = config.alpha * config.motion_eps(grid_eps);
    let mut norm2 = 0.0_f32;
    for value in raw_update.iter().take(dims) {
        norm2 += value * value;
    }
    let norm = norm2.sqrt();
    let denom = 1.0 + norm;
    let dot = raw_update
        .iter()
        .zip(dloss_ddx.iter())
        .take(dims)
        .map(|(raw, grad)| raw * grad)
        .sum::<f32>();

    for axis in 0..dims {
        let mut grad = motion_scale * dloss_ddx[axis] / denom;
        if norm > 1.0e-6 {
            grad -= motion_scale * raw_update[axis] * dot / (norm * denom * denom);
        }
        output_gradient[axis] += grad;
    }
}

#[cfg(test)]
pub(crate) fn cap_output_gradient_channel_rms(
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
) -> usize {
    cap_output_gradient_channel_rms_impl(output_gradients, output_dims, rms_cap, None)
}

pub(crate) fn cap_output_gradient_channel_rms_with_liveness_cap(
    config: &NpaConfig,
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
    liveness_rms_cap: f32,
) -> usize {
    let liveness_output = if config.state_dims > GROWTH_3D_LIVENESS_CHANNEL {
        Some(config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL)
    } else {
        None
    };
    let liveness_cap = if liveness_rms_cap.is_finite() && liveness_rms_cap > rms_cap {
        Some(liveness_rms_cap)
    } else {
        None
    };
    cap_output_gradient_channel_rms_impl(
        output_gradients,
        output_dims,
        rms_cap,
        liveness_output.zip(liveness_cap),
    )
}

pub(crate) fn cap_output_gradient_channel_rms_impl(
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
    channel_override: Option<(usize, f32)>,
) -> usize {
    if output_dims == 0
        || output_gradients.is_empty()
        || output_gradients.len() % output_dims != 0
        || rms_cap <= 0.0
        || !rms_cap.is_finite()
    {
        return 0;
    }
    let rows = output_gradients.len() / output_dims;
    let mut capped = 0usize;
    for output in 0..output_dims {
        let channel_cap = channel_override
            .filter(|(channel, cap)| *channel == output && *cap > 0.0 && cap.is_finite())
            .map(|(_, cap)| cap)
            .unwrap_or(rms_cap);
        let rms = ((0..rows)
            .map(|row| {
                let value = output_gradients[row * output_dims + output];
                value * value
            })
            .sum::<f32>()
            / rows as f32)
            .sqrt();
        if !rms.is_finite() || rms <= channel_cap {
            continue;
        }
        let scale = channel_cap / rms;
        for row in 0..rows {
            output_gradients[row * output_dims + output] *= scale;
        }
        capped += 1;
    }
    capped
}

pub(crate) fn boost_sparse_output_channel_rms(
    output_gradients: &mut [f32],
    output_dims: usize,
    channels: impl IntoIterator<Item = usize>,
    target_nonzero_rms: f32,
    max_scale: f32,
) -> usize {
    if output_dims == 0
        || output_gradients.is_empty()
        || output_gradients.len() % output_dims != 0
        || target_nonzero_rms <= 0.0
        || !target_nonzero_rms.is_finite()
        || max_scale <= 1.0
        || !max_scale.is_finite()
    {
        return 0;
    }
    let rows = output_gradients.len() / output_dims;
    let mut boosted = 0usize;
    for output in channels {
        if output >= output_dims {
            continue;
        }
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for row in 0..rows {
            let value = output_gradients[row * output_dims + output];
            if value.abs() <= 1.0e-12 {
                continue;
            }
            sum += value * value;
            count += 1;
        }
        if count == 0 {
            continue;
        }
        let rms = (sum / count as f32).sqrt();
        if !rms.is_finite() || rms <= 0.0 || rms >= target_nonzero_rms {
            continue;
        }
        let scale = (target_nonzero_rms / rms).min(max_scale);
        for row in 0..rows {
            output_gradients[row * output_dims + output] *= scale;
        }
        boosted += 1;
    }
    boosted
}

pub(crate) fn add_output_gradients(target: &mut [f32], source: &[f32]) {
    debug_assert_eq!(target.len(), source.len());
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}

pub(crate) fn render_proxy_target_coverage_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let updates = vec![[0.0; 3]; rows];
    if rows == 0 || coverage_gain <= 0.0 {
        return updates;
    }

    let active_rows = (0..rows)
        .filter(|&row| config.state_dims <= 3 || states[row * config.state_dims + 3] > -1.0)
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return updates;
    }

    let mut updates = match coverage_mode {
        CoverageUpdateModeArg::HardNearest => render_proxy_hard_target_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            seed_scale,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
        CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &active_rows,
            updates,
        ),
    };
    if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &active_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut updates,
        );
    }
    add_surface_gap_relocation_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        coverage_normal_weight,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_strata_coverage_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_normal_coverage_to_updates(
        target,
        positions,
        &active_rows,
        coverage_gain,
        coverage_normal_weight,
        coverage_samples,
        max_update_norm,
        &mut updates,
    );
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_proxy_weighted_target_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    row_weights: &[f32],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let updates = vec![[0.0; 3]; rows];
    if rows == 0 || row_weights.len() < rows || coverage_gain <= 0.0 {
        return updates;
    }

    let candidate_rows = (0..rows)
        .filter(|&row| row_weights[row].is_finite() && row_weights[row] > 1.0e-3)
        .collect::<Vec<_>>();
    if candidate_rows.is_empty() {
        return updates;
    }

    let mut updates = match coverage_mode {
        CoverageUpdateModeArg::HardNearest => render_proxy_hard_target_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
        CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            coverage_softness,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            coverage_normal_weight,
            seed_scale,
            &candidate_rows,
            updates,
        ),
        CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
        CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
            target,
            positions,
            coverage_gain,
            coverage_samples,
            max_update_norm,
            &candidate_rows,
            updates,
        ),
    };
    if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &candidate_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut updates,
        );
    }
    add_surface_gap_relocation_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        coverage_normal_weight,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_strata_coverage_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_gap_gain,
        coverage_samples,
        seed_scale,
        max_update_norm,
        &mut updates,
    );
    add_surface_normal_coverage_to_updates(
        target,
        positions,
        &candidate_rows,
        coverage_gain,
        coverage_normal_weight,
        coverage_samples,
        max_update_norm,
        &mut updates,
    );

    for (row, update) in updates.iter_mut().enumerate() {
        let weight = row_weights[row].clamp(0.0, 1.0);
        if weight >= 1.0 {
            continue;
        }
        for axis_update in update.iter_mut() {
            *axis_update *= weight;
        }
        clamp_vector3(update, max_update_norm);
    }
    updates
}

pub(crate) fn render_proxy_hard_target_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if best_distance2.is_finite() {
            residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
            residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
            residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
            counts[best_row] += 1;
        }
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        updates[row][0] = coverage_gain * residual_sums[row][0] / count as f32;
        updates[row][1] = coverage_gain * residual_sums[row][1] / count as f32;
        updates[row][2] = coverage_gain * residual_sums[row][2] / count as f32;
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}

pub(crate) fn render_proxy_soft_chamfer_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    let samples = coverage_samples.max(rows.max(512));
    let sigma = if coverage_softness.is_finite() && coverage_softness > 0.0 {
        coverage_softness
    } else {
        target_coverage_threshold(seed_scale) * 1.5
    }
    .max(1.0e-4);
    let inv_two_sigma2 = 0.5 / (sigma * sigma);
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let normal_cost_scale = if coverage_normal_weight.is_finite() {
        coverage_normal_weight.max(0.0) * sigma * sigma
    } else {
        0.0
    };
    let mut projected_normals = vec![[0.0_f32; 3]; rows];
    for &row in active_rows {
        let projection = target.project([positions[row][0], positions[row][1], positions[row][2]]);
        projected_normals[row] = projection.normal;
        residual_sums[row][0] += 0.5 * projection.residual[0];
        residual_sums[row][1] += 0.5 * projection.residual[1];
        residual_sums[row][2] += 0.5 * projection.residual[2];
        weights[row] += 0.5;
    }

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_score = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            best_score = best_score.min(score);
        }
        if !best_score.is_finite() {
            continue;
        }

        let mut weight_sum = 0.0_f32;
        let mut sample_weights = Vec::with_capacity(active_rows.len());
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_alignment = dot3(sample.normal, projected_normals[row]).clamp(-1.0, 1.0);
            let score = distance2 + normal_cost_scale * (1.0 - normal_alignment);
            let weight = (-(score - best_score) * inv_two_sigma2).exp();
            weight_sum += weight;
            sample_weights.push((row, weight));
        }
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            continue;
        }

        for (row, weight) in sample_weights {
            let normalized = weight / weight_sum;
            residual_sums[row][0] += normalized * (sample.position[0] - positions[row][0]);
            residual_sums[row][1] += normalized * (sample.position[1] - positions[row][1]);
            residual_sums[row][2] += normalized * (sample.position[2] - positions[row][2]);
            weights[row] += normalized;
        }
    }

    let mut repulsion_sums = vec![[0.0_f32; 3]; rows];
    if coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite() {
        let repulsion_radius =
            if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
                coverage_repulsion_radius
            } else {
                target_coverage_threshold(seed_scale) * 2.0
            }
            .max(1.0e-4);
        for lhs_idx in 0..active_rows.len() {
            let lhs = active_rows[lhs_idx];
            for &rhs in &active_rows[lhs_idx + 1..] {
                let dx = positions[lhs][0] - positions[rhs][0];
                let dy = positions[lhs][1] - positions[rhs][1];
                let dz = positions[lhs][2] - positions[rhs][2];
                let distance2 = dx * dx + dy * dy + dz * dz;
                if distance2 <= 1.0e-12 || distance2 >= repulsion_radius * repulsion_radius {
                    continue;
                }
                let distance = distance2.sqrt();
                let strength = (1.0 - distance / repulsion_radius).powi(2);
                let force = [
                    dx * strength / distance,
                    dy * strength / distance,
                    dz * strength / distance,
                ];
                let lhs_force = tangent_component(force, projected_normals[lhs]);
                let rhs_force =
                    tangent_component([-force[0], -force[1], -force[2]], projected_normals[rhs]);
                for axis in 0..3 {
                    repulsion_sums[lhs][axis] += lhs_force[axis];
                    repulsion_sums[rhs][axis] += rhs_force[axis];
                }
            }
        }
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        updates[row][0] = coverage_gain
            * (residual_sums[row][0] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][0]);
        updates[row][1] = coverage_gain
            * (residual_sums[row][1] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][1]);
        updates[row][2] = coverage_gain
            * (residual_sums[row][2] / weights[row]
                + coverage_repulsion_gain * repulsion_sums[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            updates[row][0] *= scale;
            updates[row][1] *= scale;
            updates[row][2] *= scale;
        }
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_tangent_repulsion_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_repulsion_gain: f32,
    coverage_repulsion_radius: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_repulsion_gain <= 0.0
        || !coverage_repulsion_gain.is_finite()
        || active_rows.len() < 2
    {
        return;
    }
    let radius = if coverage_repulsion_radius.is_finite() && coverage_repulsion_radius > 0.0 {
        coverage_repulsion_radius
    } else {
        target_coverage_threshold(seed_scale) * 2.0
    }
    .max(1.0e-4);
    let radius2 = radius * radius;
    let mut projected_normals = vec![[0.0_f32; 3]; positions.len()];
    for &row in active_rows {
        if row < positions.len() {
            projected_normals[row] = target.project(position3(positions[row])).normal;
        }
    }
    let mut repulsion_sums = vec![[0.0_f32; 3]; positions.len()];
    let mut counts = vec![0usize; positions.len()];
    for lhs_idx in 0..active_rows.len() {
        let lhs = active_rows[lhs_idx];
        if lhs >= positions.len() {
            continue;
        }
        for &rhs in &active_rows[lhs_idx + 1..] {
            if rhs >= positions.len() {
                continue;
            }
            let dx = positions[lhs][0] - positions[rhs][0];
            let dy = positions[lhs][1] - positions[rhs][1];
            let dz = positions[lhs][2] - positions[rhs][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 <= 1.0e-12 || distance2 >= radius2 {
                continue;
            }
            let distance = distance2.sqrt();
            let strength = (1.0 - distance / radius).powi(2);
            let force = [
                dx * strength / distance,
                dy * strength / distance,
                dz * strength / distance,
            ];
            let lhs_force = tangent_component(force, projected_normals[lhs]);
            let rhs_force =
                tangent_component([-force[0], -force[1], -force[2]], projected_normals[rhs]);
            for axis in 0..3 {
                repulsion_sums[lhs][axis] += lhs_force[axis];
                repulsion_sums[rhs][axis] += rhs_force[axis];
            }
            counts[lhs] += 1;
            counts[rhs] += 1;
        }
    }
    for &row in active_rows {
        if row >= updates.len() || counts[row] == 0 {
            continue;
        }
        let scale = coverage_gain * coverage_repulsion_gain / counts[row] as f32;
        for axis in 0..3 {
            updates[row][axis] += scale * repulsion_sums[row][axis];
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_gap_relocation_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    coverage_samples: usize,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_gap_gain <= 0.0
        || !coverage_gap_gain.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        score: f32,
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let threshold = target_coverage_threshold(seed_scale);
    let threshold2 = threshold * threshold;
    let normal_cost_scale = if coverage_normal_weight.is_finite() && coverage_normal_weight > 0.0 {
        coverage_normal_weight * threshold2.max(1.0e-6)
    } else {
        0.0
    };
    let projected_normals = if normal_cost_scale > 0.0 {
        let mut normals = vec![[0.0_f32; 3]; positions.len()];
        for &row in active_rows {
            if row < positions.len() {
                normals[row] = target.project(position3(positions[row])).normal;
            }
        }
        Some(normals)
    } else {
        None
    };
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; positions.len()];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_score = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            let normal_penalty = projected_normals.as_ref().map_or(0.0, |normals| {
                normal_cost_scale * (1.0 - dot3(sample.normal, normals[row]).clamp(-1.0, 1.0))
            });
            let score = distance2 + normal_penalty;
            if score < best_score {
                best_score = score;
                best_row = row;
            }
        }
        if !best_score.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_score <= threshold2 {
            continue;
        }
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            score: best_score,
        };
        if bin_candidates[bin].is_none_or(|current| best_score > current.score) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    if gaps.is_empty() {
        return;
    }
    gaps.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = gaps.len().min(active_rows.len().saturating_div(2).max(1));
    let mut relocated = 0usize;
    for gap in gaps.iter().copied() {
        if relocated >= max_relocated {
            break;
        }
        let mut best_row = gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if best_row.is_none() {
            best_row = gap_relocation_donor(
                gap.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = best_row else {
            continue;
        };
        let donor_weight = if assigned_counts[row] == 0 { 1.0 } else { 0.5 };
        let scale = 0.5 * coverage_gain * coverage_gap_gain * donor_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
        used_donors[row] = true;
        relocated += 1;
    }

    for &row in active_rows {
        if row >= positions.len() || row >= updates.len() {
            continue;
        }
        if assigned_counts[row] > 0 || used_donors[row] {
            continue;
        }
        let mut nearest_gap = gaps[0];
        let mut nearest_gap_distance2 = f32::MAX;
        for gap in &gaps {
            let dx = gap.position[0] - positions[row][0];
            let dy = gap.position[1] - positions[row][1];
            let dz = gap.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_gap_distance2 {
                nearest_gap_distance2 = distance2;
                nearest_gap = *gap;
            }
        }
        if !nearest_gap_distance2.is_finite() {
            continue;
        }
        let scale = 0.5 * coverage_gain * coverage_gap_gain;
        updates[row][0] += scale * (nearest_gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (nearest_gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (nearest_gap.position[2] - positions[row][2]);
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= clamp;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    require_under_assigned: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    let average_assignments = average_assignments.max(1.0);
    let under_assigned_limit = average_assignments.ceil().max(1.0);
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let under_assigned = assignments <= under_assigned_limit;
        if require_under_assigned
            && assigned_counts.get(row).copied().unwrap_or(0) > 0
            && !under_assigned
        {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let assignment_penalty = assignments / average_assignments;
        let overflow_bonus = (assignments / under_assigned_limit).max(1.0);
        let score = if require_under_assigned {
            distance2 * (1.0 + 0.25 * assignment_penalty)
        } else {
            distance2 * (1.0 + 0.25 * assignment_penalty) / overflow_bonus.sqrt()
        };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}

#[derive(Clone, Copy)]
pub(crate) struct SurfaceStrataCandidate {
    pub(crate) position: [f32; 3],
    pub(crate) score: f32,
    pub(crate) covered_fraction: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_strata_coverage_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_gap_gain: f32,
    coverage_samples: usize,
    seed_scale: f32,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_gap_gain <= 0.0
        || !coverage_gap_gain.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 128))
        .max(1);
    let threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let threshold2 = threshold * threshold;
    let mut bin_sample_counts = vec![0usize; bin_count];
    let mut bin_covered_counts = vec![0usize; bin_count];
    let mut bin_candidates = vec![None::<SurfaceStrataCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; positions.len()];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        bin_sample_counts[bin] += 1;
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;
        if best_distance2 <= threshold2 {
            bin_covered_counts[bin] += 1;
        }
        let candidate = SurfaceStrataCandidate {
            position: sample.position,
            score: best_distance2,
            covered_fraction: 0.0,
        };
        if bin_candidates[bin].is_none_or(|current| best_distance2 > current.score) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut candidates = Vec::new();
    for bin in 0..bin_count {
        let samples = bin_sample_counts[bin];
        if samples == 0 {
            continue;
        }
        let covered_fraction = bin_covered_counts[bin] as f32 / samples as f32;
        if covered_fraction >= 0.60 {
            continue;
        }
        if let Some(mut candidate) = bin_candidates[bin] {
            candidate.covered_fraction = covered_fraction;
            candidate.score *= (0.60 - covered_fraction).max(0.0);
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        return;
    }
    candidates.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; positions.len()];
    let max_relocated = candidates
        .len()
        .min(active_rows.len().saturating_mul(3).saturating_div(4).max(1));
    for candidate in candidates.into_iter().take(max_relocated) {
        let mut donor = surface_strata_relocation_donor(
            candidate.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if donor.is_none() {
            donor = surface_strata_relocation_donor(
                candidate.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = donor else {
            continue;
        };
        let donor_weight = if assigned_counts[row] == 0 { 1.0 } else { 0.6 };
        let strata_weight = (0.60 - candidate.covered_fraction).clamp(0.0, 1.0);
        let scale = coverage_gain * coverage_gap_gain * donor_weight * strata_weight;
        updates[row][0] += scale * (candidate.position[0] - positions[row][0]);
        updates[row][1] += scale * (candidate.position[1] - positions[row][1]);
        updates[row][2] += scale * (candidate.position[2] - positions[row][2]);
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn surface_strata_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    assigned_counts: &[usize],
    average_assignments: f32,
    used_donors: &[bool],
    require_surplus: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    let average_assignments = average_assignments.max(1.0);
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let assignments = assigned_counts.get(row).copied().unwrap_or(0) as f32;
        let surplus = assignments > average_assignments.ceil().max(1.0);
        if require_surplus && assignments > 0.0 && !surplus {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let assignment_factor = if assignments == 0.0 {
            0.5
        } else if surplus {
            0.75 / (assignments / average_assignments).sqrt()
        } else {
            1.0 + 0.25 * assignments / average_assignments
        };
        let score = distance2 * assignment_factor;
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}

#[derive(Clone, Copy)]
pub(crate) struct NormalGapCandidate {
    pub(crate) position: [f32; 3],
    pub(crate) distance2: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_normal_coverage_to_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    active_rows: &[usize],
    coverage_gain: f32,
    coverage_normal_weight: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    updates: &mut [[f32; 3]],
) {
    if coverage_gain <= 0.0
        || coverage_normal_weight <= 0.0
        || !coverage_normal_weight.is_finite()
        || active_rows.is_empty()
    {
        return;
    }

    const NORMAL_CANDIDATES_PER_BIN: usize = 8;

    let directions = normal_coverage_directions();
    let bin_count = directions.len();
    let mut active_bin_counts = vec![0usize; bin_count];
    let mut active_bins = vec![usize::MAX; positions.len()];
    for &row in active_rows {
        if row >= positions.len() {
            continue;
        }
        let projection = target.project(position3(positions[row]));
        let bin = normal_direction_bin(projection.normal, &directions);
        active_bins[row] = bin;
        active_bin_counts[bin] += 1;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let mut target_bin_counts = vec![0usize; bin_count];
    let mut bin_candidates = vec![Vec::<NormalGapCandidate>::new(); bin_count];
    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let bin = normal_direction_bin(sample.normal, &directions);
        target_bin_counts[bin] += 1;

        let mut nearest_distance2 = f32::MAX;
        for &row in active_rows {
            if row >= positions.len() {
                continue;
            }
            let dx = sample.position[0] - positions[row][0];
            let dy = sample.position[1] - positions[row][1];
            let dz = sample.position[2] - positions[row][2];
            nearest_distance2 = nearest_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        if nearest_distance2.is_finite() {
            let candidate = NormalGapCandidate {
                position: sample.position,
                distance2: nearest_distance2,
            };
            let candidates = &mut bin_candidates[bin];
            candidates.push(candidate);
            candidates.sort_by(|lhs, rhs| {
                rhs.distance2
                    .partial_cmp(&lhs.distance2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(NORMAL_CANDIDATES_PER_BIN);
        }
    }

    let mut desired_bin_counts = vec![0usize; bin_count];
    for bin in 0..bin_count {
        if target_bin_counts[bin] == 0 {
            continue;
        }
        desired_bin_counts[bin] = ((target_bin_counts[bin] as f32 / sample_count as f32)
            * active_rows.len() as f32
            * 0.85)
            .ceil()
            .max(1.0) as usize;
    }

    let mut missing = active_bin_counts
        .iter()
        .zip(desired_bin_counts.iter())
        .map(|(active, desired)| desired.saturating_sub(*active))
        .sum::<usize>();
    if missing == 0 || bin_candidates.iter().all(Vec::is_empty) {
        return;
    }

    let mut used_donors = vec![false; positions.len()];
    let mut candidate_offsets = vec![0usize; bin_count];
    let max_relocated = missing.min(active_rows.len().saturating_mul(2).saturating_div(3).max(1));
    for _ in 0..max_relocated {
        let Some((gap_bin, gap)) = normal_gap_candidate(
            &active_bin_counts,
            &desired_bin_counts,
            &bin_candidates,
            &candidate_offsets,
        ) else {
            break;
        };
        let Some(row) = normal_gap_relocation_donor(
            gap.position,
            active_rows,
            positions,
            updates.len(),
            &active_bins,
            &active_bin_counts,
            &desired_bin_counts,
            &used_donors,
        ) else {
            continue;
        };
        let donor_bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        if donor_bin < active_bin_counts.len() {
            active_bin_counts[donor_bin] = active_bin_counts[donor_bin].saturating_sub(1);
        }
        active_bin_counts[gap_bin] += 1;
        let scale = 0.5 * coverage_gain * coverage_normal_weight;
        updates[row][0] += scale * (gap.position[0] - positions[row][0]);
        updates[row][1] += scale * (gap.position[1] - positions[row][1]);
        updates[row][2] += scale * (gap.position[2] - positions[row][2]);
        clamp_update_row(updates, row, max_update_norm);
        used_donors[row] = true;
        candidate_offsets[gap_bin] += 1;
        missing = missing.saturating_sub(1);
        if missing == 0 {
            break;
        }
    }
}

pub(crate) fn normal_gap_candidate(
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    bin_candidates: &[Vec<NormalGapCandidate>],
    candidate_offsets: &[usize],
) -> Option<(usize, NormalGapCandidate)> {
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    for bin in 0..bin_candidates.len() {
        let deficit = desired_bin_counts
            .get(bin)
            .copied()
            .unwrap_or(0)
            .saturating_sub(active_bin_counts.get(bin).copied().unwrap_or(0));
        if deficit == 0 || bin_candidates[bin].is_empty() {
            continue;
        }
        let candidate_index = candidate_offsets
            .get(bin)
            .copied()
            .unwrap_or(0)
            .min(bin_candidates[bin].len() - 1);
        let candidate = bin_candidates[bin][candidate_index];
        let score = candidate.distance2 * (deficit as f32).sqrt();
        if score > best_score {
            best_score = score;
            best = Some((bin, candidate));
        }
    }
    best
}

pub(crate) fn normal_gap_relocation_donor(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
) -> Option<usize> {
    normal_gap_relocation_donor_with_filter(
        gap_position,
        active_rows,
        positions,
        update_len,
        active_bins,
        active_bin_counts,
        desired_bin_counts,
        used_donors,
        true,
    )
    .or_else(|| {
        normal_gap_relocation_donor_with_filter(
            gap_position,
            active_rows,
            positions,
            update_len,
            active_bins,
            active_bin_counts,
            desired_bin_counts,
            used_donors,
            false,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normal_gap_relocation_donor_with_filter(
    gap_position: [f32; 3],
    active_rows: &[usize],
    positions: &[[f32; 4]],
    update_len: usize,
    active_bins: &[usize],
    active_bin_counts: &[usize],
    desired_bin_counts: &[usize],
    used_donors: &[bool],
    require_surplus_bin: bool,
) -> Option<usize> {
    let mut best_row = None::<usize>;
    let mut best_score = f32::MAX;
    for &row in active_rows {
        if row >= positions.len()
            || row >= update_len
            || used_donors.get(row).copied().unwrap_or(true)
        {
            continue;
        }
        let bin = active_bins.get(row).copied().unwrap_or(usize::MAX);
        let surplus = bin < active_bin_counts.len()
            && active_bin_counts[bin] > desired_bin_counts.get(bin).copied().unwrap_or(0).max(1);
        if require_surplus_bin && !surplus {
            continue;
        }
        let dx = gap_position[0] - positions[row][0];
        let dy = gap_position[1] - positions[row][1];
        let dz = gap_position[2] - positions[row][2];
        let distance2 = dx * dx + dy * dy + dz * dz;
        if !distance2.is_finite() {
            continue;
        }
        let score = if surplus { distance2 * 0.75 } else { distance2 };
        if score < best_score {
            best_score = score;
            best_row = Some(row);
        }
    }
    best_row
}

pub(crate) fn normal_direction_bin(normal: [f32; 3], directions: &[[f32; 3]]) -> usize {
    let normal = normalize3_or(normal, [0.0, 0.0, 1.0]);
    let mut best_bin = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (idx, direction) in directions.iter().enumerate() {
        let score = dot3(normal, *direction);
        if score > best_dot {
            best_dot = score;
            best_bin = idx;
        }
    }
    best_bin
}

pub(crate) fn normalize3_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = dot3(value, value).sqrt();
    if norm <= 1.0e-8 || !norm.is_finite() {
        fallback
    } else {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    }
}

pub(crate) fn normal_coverage_directions() -> [[f32; 3]; 26] {
    const INV_SQRT_2: f32 = 0.707_106_77;
    const INV_SQRT_3: f32 = 0.577_350_26;
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [INV_SQRT_2, INV_SQRT_2, 0.0],
        [INV_SQRT_2, -INV_SQRT_2, 0.0],
        [-INV_SQRT_2, INV_SQRT_2, 0.0],
        [-INV_SQRT_2, -INV_SQRT_2, 0.0],
        [INV_SQRT_2, 0.0, INV_SQRT_2],
        [INV_SQRT_2, 0.0, -INV_SQRT_2],
        [-INV_SQRT_2, 0.0, INV_SQRT_2],
        [-INV_SQRT_2, 0.0, -INV_SQRT_2],
        [0.0, INV_SQRT_2, INV_SQRT_2],
        [0.0, INV_SQRT_2, -INV_SQRT_2],
        [0.0, -INV_SQRT_2, INV_SQRT_2],
        [0.0, -INV_SQRT_2, -INV_SQRT_2],
        [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
        [-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
    ]
}

pub(crate) fn clamp_update_row(updates: &mut [[f32; 3]], row: usize, max_update_norm: f32) {
    if row >= updates.len() {
        return;
    }
    clamp_vector3(&mut updates[row], max_update_norm);
}

pub(crate) fn clamp_vector3(update: &mut [f32; 3], max_update_norm: f32) {
    let norm = (update[0].powi(2) + update[1].powi(2) + update[2].powi(2)).sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for value in update {
            *value *= clamp;
        }
    }
}

pub(crate) fn render_proxy_gap_farthest_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    #[derive(Clone, Copy)]
    struct GapCandidate {
        position: [f32; 3],
        distance2: f32,
    }

    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(rows.max(512));
    let bin_count = sample_count
        .min((active_rows.len().saturating_mul(2)).clamp(32, 512))
        .max(1);
    let mut bin_candidates = vec![None::<GapCandidate>; bin_count];
    let mut assigned_counts = vec![0usize; rows];

    for sample_idx in 0..sample_count {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = active_rows[0];
        let mut best_distance2 = f32::MAX;
        for &row in active_rows {
            let position = positions[row];
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }
        assigned_counts[best_row] += 1;

        let bin = (sample_idx * bin_count / sample_count).min(bin_count - 1);
        let candidate = GapCandidate {
            position: sample.position,
            distance2: best_distance2,
        };
        if bin_candidates[bin].is_none_or(|current| best_distance2 > current.distance2) {
            bin_candidates[bin] = Some(candidate);
        }
    }

    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];
    let mut gaps = bin_candidates
        .into_iter()
        .flatten()
        .collect::<Vec<GapCandidate>>();
    gaps.sort_by(|lhs, rhs| {
        rhs.distance2
            .partial_cmp(&lhs.distance2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let average_assignments = sample_count as f32 / active_rows.len().max(1) as f32;
    let mut used_donors = vec![false; rows];
    let max_relocated = gaps.len().min(active_rows.len().max(1));
    for candidate in gaps.into_iter().take(max_relocated) {
        let mut donor = gap_relocation_donor(
            candidate.position,
            active_rows,
            positions,
            updates.len(),
            &assigned_counts,
            average_assignments,
            &used_donors,
            true,
        );
        if donor.is_none() {
            donor = gap_relocation_donor(
                candidate.position,
                active_rows,
                positions,
                updates.len(),
                &assigned_counts,
                average_assignments,
                &used_donors,
                false,
            );
        }
        let Some(row) = donor else {
            continue;
        };
        let residual = [
            candidate.position[0] - positions[row][0],
            candidate.position[1] - positions[row][1],
            candidate.position[2] - positions[row][2],
        ];
        let weight = candidate.distance2.sqrt().max(1.0e-4);
        for axis in 0..3 {
            residual_sums[row][axis] += residual[axis] * weight;
        }
        weights[row] += weight;
        used_donors[row] = true;
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            let residual = if weights[row] > 0.0 {
                residual_sums[row][axis] / weights[row] + 0.25 * projection.residual[axis]
            } else {
                0.25 * projection.residual[axis]
            };
            updates[row][axis] = coverage_gain * residual;
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }

    updates
}

pub(crate) fn tangent_component(vector: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let normal_norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if normal_norm2 <= 1.0e-12 {
        return vector;
    }
    let dot = vector[0] * normal[0] + vector[1] * normal[1] + vector[2] * normal[2];
    [
        vector[0] - normal[0] * dot / normal_norm2,
        vector[1] - normal[1] * dot / normal_norm2,
        vector[2] - normal[2] * dot / normal_norm2,
    ]
}

pub(crate) fn reference_seed_scale_for_seed_mode(
    preset: AutomataPreset,
    seed_mode: ParticleSeed,
) -> f32 {
    match seed_mode {
        ParticleSeed::UvTorus3d
        | ParticleSeed::UvTorusDense3d
        | ParticleSeed::TorusFieldDense3d
        | ParticleSeed::TeapotFieldDense3d
        | ParticleSeed::TorusGrowth3d
        | ParticleSeed::TeapotGrowth3d
        | ParticleSeed::TorusSubstrateGrowth3d
        | ParticleSeed::TeapotSubstrateGrowth3d
        | ParticleSeed::TorusLocalGrowth3d
        | ParticleSeed::TeapotLocalGrowth3d
        | ParticleSeed::TorusLocalSubstrateGrowth3d
        | ParticleSeed::TeapotLocalSubstrateGrowth3d
        | ParticleSeed::TorusMorphogenDense3d
        | ParticleSeed::TeapotMorphogenDense3d => UV_TORUS_FIELD_SCALE,
        _ => NpaConfig::seed_scale_for_preset(preset),
    }
}

pub(crate) fn default_train_target_seed(
    _preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> Option<u64> {
    if zero_update {
        None
    } else {
        Some(target_seed.unwrap_or(DEFAULT_GROWTH_TARGET_SEED))
    }
}

pub(crate) fn train_target_source(
    preset: AutomataPreset,
    target_seed: Option<u64>,
    zero_update: bool,
) -> String {
    match (target_seed, zero_update) {
        (Some(seed), false) => format!("seeded:{preset:?}:{seed}"),
        (None, true) => "explicit-zero-update".to_string(),
        _ => unreachable!("target seed/source selection should be normalized first"),
    }
}

pub(crate) fn training_source_with_batch(
    batch_source: TrainingBatchArg,
    target_source: &str,
) -> String {
    match batch_source {
        TrainingBatchArg::Rollout => format!("rollout-local:{target_source}"),
        TrainingBatchArg::Features => format!("feature-rows:{target_source}"),
    }
}

pub(crate) fn render_proxy_sliced_ot_coverage_updates(
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    coverage_gain: f32,
    coverage_samples: usize,
    max_update_norm: f32,
    active_rows: &[usize],
    mut updates: Vec<[f32; 3]>,
) -> Vec<[f32; 3]> {
    let rows = positions.len();
    if rows == 0 || active_rows.is_empty() {
        return updates;
    }

    let sample_count = coverage_samples.max(active_rows.len().max(512));
    let samples = (0..sample_count)
        .map(|sample_idx| target.surface_sample(sample_idx).position)
        .collect::<Vec<_>>();
    let directions = sliced_ot_directions();
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut weights = vec![0.0_f32; rows];

    for direction in directions {
        let mut target_order = (0..samples.len()).collect::<Vec<_>>();
        target_order.sort_by(|&lhs, &rhs| {
            dot3(samples[lhs], direction)
                .partial_cmp(&dot3(samples[rhs], direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut active_order = active_rows.to_vec();
        active_order.sort_by(|&lhs, &rhs| {
            dot3(position3(positions[lhs]), direction)
                .partial_cmp(&dot3(position3(positions[rhs]), direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let active_len = active_order.len().max(1);
        for (rank, &row) in active_order.iter().enumerate() {
            let sample_rank = (((rank as f32 + 0.5) * sample_count as f32 / active_len as f32)
                .floor() as usize)
                .min(sample_count - 1);
            let sample = samples[target_order[sample_rank]];
            for axis in 0..3 {
                residual_sums[row][axis] += sample[axis] - positions[row][axis];
            }
            weights[row] += 1.0;
        }
    }

    for &row in active_rows {
        let projection = target.project(position3(positions[row]));
        for axis in 0..3 {
            residual_sums[row][axis] += 0.25 * projection.residual[axis];
        }
        weights[row] += 0.25;
    }

    for row in 0..rows {
        if weights[row] <= 0.0 {
            continue;
        }
        for axis in 0..3 {
            updates[row][axis] = coverage_gain * residual_sums[row][axis] / weights[row];
        }
        let norm =
            (updates[row][0].powi(2) + updates[row][1].powi(2) + updates[row][2].powi(2)).sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / norm;
            for axis in 0..3 {
                updates[row][axis] *= scale;
            }
        }
    }
    updates
}

pub(crate) fn sliced_ot_directions() -> [[f32; 3]; 26] {
    normal_coverage_directions()
}

pub(crate) fn position3(position: [f32; 4]) -> [f32; 3] {
    [position[0], position[1], position[2]]
}
