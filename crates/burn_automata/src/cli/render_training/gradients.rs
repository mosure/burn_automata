#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::*;

#[derive(Clone)]
pub(crate) struct RenderProxyGradientRows {
    pub(crate) row_indices: Vec<usize>,
    pub(crate) gradients: Vec<[f32; 3]>,
    pub(crate) opacity_gradients: Vec<f32>,
    pub(crate) scale_gradients: Vec<f32>,
    pub(crate) color_gradients: Vec<[f32; 3]>,
}

pub(crate) struct RenderTrajectoryAdjoint {
    pub(crate) state: Vec<f32>,
    pub(crate) position: Vec<[f32; 4]>,
    pub(crate) weight: f32,
}

pub(crate) fn spread_row_indices(items: usize, max_rows: usize) -> Vec<usize> {
    let rows = items.min(max_rows).max(1);
    if rows >= items {
        return (0..items).collect();
    }
    (0..rows)
        .map(|idx| (idx * items / rows).min(items - 1))
        .collect()
}

pub(crate) fn trajectory_render_sample_indices(len: usize, max_samples: usize) -> Vec<usize> {
    let samples = len.min(max_samples);
    if samples == 0 {
        return Vec::new();
    }
    let mut indices = Vec::with_capacity(samples);
    for sample in 0..samples {
        let index = ((sample + 1) * len / samples).saturating_sub(1);
        if indices.last().copied() != Some(index) {
            indices.push(index);
        }
    }
    indices
}

pub(crate) fn trajectory_liveness_sample_indices(
    len: usize,
    render_sample_budget: usize,
) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let sample_cap = len.min(
        TEMPORAL_LIVENESS_TRAJECTORY_SAMPLE_CAP
            .max(render_sample_budget)
            .max(1),
    );
    let mut indices = Vec::with_capacity(sample_cap + 2);
    indices.push(0);
    if len > 1 {
        indices.push(1);
    }
    for index in trajectory_render_sample_indices(len, sample_cap) {
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

pub(crate) fn render_proxy_gradient_row_indices(particles: usize, max_rows: usize) -> Vec<usize> {
    spread_row_indices(particles, max_rows)
}

pub(crate) fn render_position_gradient(
    trace: &crate::RolloutTrace,
    target: &TriangleMeshTarget,
    render_cfg: RenderLossConfig,
    cfg: &RenderProxyTrainingConfig,
) -> Result<RenderProxyGradientRows, Box<dyn std::error::Error>> {
    let row_indices =
        render_proxy_gradient_row_indices(trace.particle_count, cfg.gradient_particles);
    match cfg.gradient_mode {
        RenderGradientModeArg::Analytic => {
            let report = mesh_multiview_render_position_gradient_for_rows_from_trace(
                trace,
                target,
                render_cfg,
                &row_indices,
            )?;
            Ok(RenderProxyGradientRows {
                row_indices: report.row_indices,
                gradients: report.gradients,
                opacity_gradients: report.opacity_gradients,
                scale_gradients: report.scale_gradients,
                color_gradients: report.color_gradients,
            })
        }
        RenderGradientModeArg::FiniteDiff => {
            let mut gradient = vec![[0.0; 3]; row_indices.len()];
            let mut opacity_gradient = vec![0.0; row_indices.len()];
            let mut scale_gradient = vec![0.0; row_indices.len()];
            let mut color_gradient = vec![[0.0; 3]; row_indices.len()];
            let eps = cfg.finite_diff_eps;
            for (gradient_idx, &row) in row_indices.iter().enumerate() {
                for axis in 0..3 {
                    let plus = trace_with_position_delta(trace, row, axis, eps);
                    let minus = trace_with_position_delta(trace, row, axis, -eps);
                    let plus_loss =
                        mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                            .total_loss;
                    let minus_loss =
                        mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                            .total_loss;
                    gradient[gradient_idx][axis] = (plus_loss - minus_loss) / (2.0 * eps);
                }
                if let Some(opacity_channel) = growth_3d_material_opacity_channel(trace.state_dims)
                {
                    let plus = trace_with_state_delta(trace, row, opacity_channel, eps);
                    let minus = trace_with_state_delta(trace, row, opacity_channel, -eps);
                    let plus_loss =
                        mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                            .total_loss;
                    let minus_loss =
                        mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                            .total_loss;
                    let state_logit = trace.states[row * trace.state_dims + opacity_channel]
                        + render_cfg.opacity_logit_bias;
                    let derivative = sigmoid_unit_derivative(state_logit);
                    if derivative > 1.0e-6 {
                        opacity_gradient[gradient_idx] =
                            (plus_loss - minus_loss) / (2.0 * eps * derivative);
                    }
                }
                if render_cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale
                    && trace.state_dims >= 5
                {
                    let scale_channel = trace.state_dims - 5;
                    let plus = trace_with_state_delta(trace, row, scale_channel, eps);
                    let minus = trace_with_state_delta(trace, row, scale_channel, -eps);
                    let plus_loss =
                        mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                            .total_loss;
                    let minus_loss =
                        mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                            .total_loss;
                    scale_gradient[gradient_idx] = (plus_loss - minus_loss) / (2.0 * eps);
                }
                if trace.state_dims >= 3 {
                    let tail = trace.state_dims - 3;
                    for channel in 0..3 {
                        let state_value = trace.states[row * trace.state_dims + tail + channel];
                        if state_value <= -1.0 || state_value >= 1.0 {
                            continue;
                        }
                        let plus = trace_with_state_delta(trace, row, tail + channel, eps);
                        let minus = trace_with_state_delta(trace, row, tail + channel, -eps);
                        let plus_loss =
                            mesh_multiview_render_loss_from_trace(&plus, target, render_cfg)?
                                .total_loss;
                        let minus_loss =
                            mesh_multiview_render_loss_from_trace(&minus, target, render_cfg)?
                                .total_loss;
                        color_gradient[gradient_idx][channel] = (plus_loss - minus_loss) / eps;
                    }
                }
            }
            Ok(RenderProxyGradientRows {
                row_indices,
                gradients: gradient,
                opacity_gradients: opacity_gradient,
                scale_gradients: scale_gradient,
                color_gradients: color_gradient,
            })
        }
    }
}

pub(crate) fn trace_with_position_delta(
    trace: &crate::RolloutTrace,
    row: usize,
    axis: usize,
    delta: f32,
) -> crate::RolloutTrace {
    let mut perturbed = trace.clone();
    perturbed.positions[row][axis] += delta;
    perturbed
}

pub(crate) fn trace_with_state_delta(
    trace: &crate::RolloutTrace,
    row: usize,
    channel: usize,
    delta: f32,
) -> crate::RolloutTrace {
    let mut perturbed = trace.clone();
    let index = row * trace.state_dims + channel;
    if index < perturbed.states.len() {
        perturbed.states[index] += delta;
    }
    perturbed
}

pub(crate) fn render_proxy_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    trace: &crate::RolloutTrace,
    trajectory: &[RenderTrajectorySnapshot],
    gradient: &RenderProxyGradientRows,
    cfg: &RenderProxyTrainingConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    let rows = gradient
        .gradients
        .len()
        .min(gradient.row_indices.len())
        .min(gradient.opacity_gradients.len())
        .min(gradient.scale_gradients.len())
        .min(gradient.color_gradients.len());
    if rows == 0 {
        return Err(std::io::Error::other("render proxy gradient produced no rows").into());
    }
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();

    let mut features = Vec::new();
    let mut states = Vec::new();
    let mut positions = Vec::new();
    let mut gradient_rows = Vec::new();
    let mut weights = Vec::new();
    let mut step_fractions = Vec::new();

    if cfg.trajectory_supervision && !trajectory.is_empty() {
        features.reserve(trajectory.len() * rows * input_dims);
        states.reserve(trajectory.len() * rows * model.config.state_dims);
        positions.reserve(trajectory.len() * rows);
        gradient_rows.reserve(trajectory.len() * rows);
        weights.reserve(trajectory.len() * rows);
        step_fractions.reserve(trajectory.len() * rows);
        for snapshot in trajectory {
            for (gradient_row, &row) in gradient.row_indices.iter().enumerate().take(rows) {
                if row >= trace.particle_count {
                    return Err(std::io::Error::other(format!(
                        "render proxy gradient row {row} out of range for {} particles",
                        trace.particle_count
                    ))
                    .into());
                }
                let feature_base = row * input_dims;
                features
                    .extend_from_slice(&snapshot.features[feature_base..feature_base + input_dims]);
                let state_base = row * model.config.state_dims;
                states.extend_from_slice(
                    &snapshot.states[state_base..state_base + model.config.state_dims],
                );
                positions.push(snapshot.positions[row]);
                gradient_rows.push(gradient_row);
                weights.push(0.5 + 0.5 * snapshot.step_fraction);
                step_fractions.push(snapshot.step_fraction);
            }
        }
    } else {
        let mut selected_positions = Vec::with_capacity(rows);
        let mut selected_states = Vec::with_capacity(rows * model.config.state_dims);
        for &row in gradient.row_indices.iter().take(rows) {
            if row >= trace.particle_count {
                return Err(std::io::Error::other(format!(
                    "render proxy gradient row {row} out of range for {} particles",
                    trace.particle_count
                ))
                .into());
            }
            selected_positions.push(trace.positions[row]);
            let state_base = row * model.config.state_dims;
            selected_states
                .extend_from_slice(&trace.states[state_base..state_base + model.config.state_dims]);
        }
        let step = model.step_cpu(
            &selected_positions,
            &selected_states,
            1,
            rows,
            grid,
            1.0,
            None,
        )?;
        features = step.perception.features;
        states = selected_states;
        positions = selected_positions;
        gradient_rows.extend(0..rows);
        weights.resize(rows, 1.0);
        step_fractions.resize(rows, 1.0);
    }

    let mut target_update = model.forward_update_from_features(&features)?;
    for chunk_start in (0..positions.len()).step_by(rows) {
        let chunk_end = (chunk_start + rows).min(positions.len());
        let chunk_positions = &positions[chunk_start..chunk_end];
        let chunk_states =
            &states[chunk_start * model.config.state_dims..chunk_end * model.config.state_dims];
        let coverage_updates = render_proxy_target_coverage_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
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
        let surface_updates = render_proxy_surface_projection_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            cfg.surface_gain,
            cfg.surface_escape_gain,
            cfg.max_update_norm,
        );
        let material_coverage_updates = material_target_coverage_opacity_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            cfg.opacity_gain,
            cfg.coverage_samples,
            cfg.seed_scale,
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
            cfg.material_max_opacity_update,
        );
        let material_strata_updates = material_surface_strata_opacity_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            cfg.opacity_gain,
            cfg.coverage_samples,
            cfg.seed_scale,
            GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET,
            cfg.material_max_opacity_update,
        );
        let liveness_update_cap =
            liveness_max_update(cfg.max_opacity_update, cfg.liveness_update_multiplier);
        let step_fraction = step_fractions.get(chunk_start).copied().unwrap_or(1.0);
        let liveness_updates = liveness_front_temporal_target_updates(
            &model.config,
            chunk_positions,
            chunk_states,
            cfg.liveness_gain,
            cfg.liveness_front_radius,
            step_fraction,
            liveness_update_cap,
        );
        let material_liveness_updates = material_visible_liveness_target_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            cfg.material_liveness_gain,
            target_coverage_threshold(cfg.seed_scale),
            liveness_update_cap,
        );
        let raw_start = chunk_start * output_dims;
        let raw_end = chunk_end * output_dims;
        let material_surface_updates = material_visible_surface_approach_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            Some(&target_update[raw_start..raw_end]),
            cfg.surface_gain,
            cfg.surface_escape_gain,
            cfg.max_update_norm,
            cfg.seed_scale,
            cfg.liveness_front_radius,
            None,
        );
        let material_surface_coverage_updates = material_visible_surface_coverage_updates(
            &model.config,
            target,
            chunk_positions,
            chunk_states,
            Some(&target_update[raw_start..raw_end]),
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
            None,
        );
        for local_idx in 0..chunk_positions.len() {
            let row = chunk_start + local_idx;
            let gradient_row = gradient_rows[row];
            let base = row * output_dims;
            let grad = gradient.gradients[gradient_row];
            let weight = weights[row];
            let mut update = [
                -cfg.motion_gain * grad[0] * weight
                    + (coverage_updates[local_idx][0]
                        + surface_updates[local_idx][0]
                        + material_surface_updates[local_idx][0]
                        + material_surface_coverage_updates[local_idx][0])
                        * weight,
                -cfg.motion_gain * grad[1] * weight
                    + (coverage_updates[local_idx][1]
                        + surface_updates[local_idx][1]
                        + material_surface_updates[local_idx][1]
                        + material_surface_coverage_updates[local_idx][1])
                        * weight,
                -cfg.motion_gain * grad[2] * weight
                    + (coverage_updates[local_idx][2]
                        + surface_updates[local_idx][2]
                        + material_surface_updates[local_idx][2]
                        + material_surface_coverage_updates[local_idx][2])
                        * weight,
            ];
            let norm =
                (update[0] * update[0] + update[1] * update[1] + update[2] * update[2]).sqrt();
            if norm > cfg.max_update_norm.max(1.0e-6) {
                let scale = cfg.max_update_norm / norm;
                update[0] *= scale;
                update[1] *= scale;
                update[2] *= scale;
            }
            target_update[base] += update[0];
            target_update[base + 1] += update[1];
            target_update[base + 2] += update[2];
            if (cfg.liveness_gain > 0.0 || cfg.material_liveness_gain > 0.0)
                && model.config.state_dims > GROWTH_3D_LIVENESS_CHANNEL
            {
                let liveness_output = base + model.config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
                let liveness_delta = liveness_updates.get(local_idx).copied().unwrap_or(0.0)
                    + material_liveness_updates
                        .get(local_idx)
                        .copied()
                        .unwrap_or(0.0);
                target_update[liveness_output] +=
                    (liveness_delta * weight).clamp(-liveness_update_cap, liveness_update_cap);
            }
            if cfg.opacity_gain > 0.0
                || cfg.material_liveness_gain > 0.0
                || cfg.material_tail_gain > 0.0
            {
                let Some(opacity_channel) =
                    growth_3d_material_opacity_channel(model.config.state_dims)
                else {
                    continue;
                };
                let state_base = row * model.config.state_dims;
                let opacity_output = base + model.config.spatial_dims + opacity_channel;
                let projection = target.project(position3(chunk_positions[local_idx]));
                let mut material_delta = 0.0_f32;
                if cfg.opacity_gain > 0.0
                    && projection.distance.is_finite()
                    && projection.distance <= target_coverage_threshold(cfg.seed_scale)
                    && states[state_base + GROWTH_3D_LIVENESS_CHANNEL] > -1.0
                {
                    let surface_weight = (1.0
                        - projection.distance / target_coverage_threshold(cfg.seed_scale))
                    .clamp(0.0, 1.0);
                    let material_update = cfg.opacity_gain
                        * surface_weight
                        * (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET
                            - states[state_base + opacity_channel])
                        * weight;
                    material_delta += material_update;
                }
                if cfg.opacity_gain > 0.0 {
                    material_delta += material_coverage_updates
                        .get(local_idx)
                        .copied()
                        .unwrap_or(0.0)
                        * weight;
                    material_delta += material_strata_updates
                        .get(local_idx)
                        .copied()
                        .unwrap_or(0.0)
                        * weight;
                }
                let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
                let material_opacity = states[state_base + opacity_channel];
                if cfg.material_liveness_gain > 0.0
                    && liveness <= -1.0
                    && material_opacity > GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
                {
                    let material_update = -cfg.material_liveness_gain
                        * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT)
                        * weight;
                    material_delta += material_update;
                }
                if cfg.material_tail_gain > 0.0
                    && material_opacity > GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
                    && projection.distance.is_finite()
                    && projection.distance > GROWTH_3D_SURFACE_MAX_DISTANCE
                {
                    let escape =
                        (projection.distance / GROWTH_3D_SURFACE_MAX_DISTANCE - 1.0).max(0.0);
                    let material_update = -cfg.material_tail_gain
                        * escape.min(8.0)
                        * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT)
                        * weight;
                    material_delta += material_update;
                }
                if cfg.opacity_gain > 0.0 {
                    let state_logit =
                        states[state_base + opacity_channel] + cfg.render.opacity_logit_bias;
                    let opacity_update = -cfg.opacity_gain
                        * gradient.opacity_gradients[gradient_row]
                        * sigmoid_unit_derivative(state_logit)
                        * weight;
                    material_delta += opacity_update;
                }
                let positive_cap = if cfg.material_max_opacity_update.is_finite()
                    && cfg.material_max_opacity_update > 0.0
                {
                    cfg.material_max_opacity_update
                } else {
                    f32::INFINITY
                };
                let suppression_cap = material_suppression_max_update(
                    cfg.material_max_opacity_update,
                    cfg.material_suppression_update_multiplier,
                );
                let negative_cap = if suppression_cap.is_finite() && suppression_cap > 0.0 {
                    suppression_cap
                } else {
                    f32::INFINITY
                };
                target_update[opacity_output] += material_delta.clamp(-negative_cap, positive_cap);
            }
            if (cfg.scale_gain > 0.0 || cfg.scale_budget_weight > 0.0)
                && cfg.render.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale
                && model.config.state_dims >= 5
            {
                let scale_channel = model.config.state_dims - 5;
                let state_base = row * model.config.state_dims;
                let state = &states[state_base..state_base + model.config.state_dims];
                let scale_update = -weight
                    * (cfg.scale_gain * gradient.scale_gradients[gradient_row]
                        + gaussian_scale_budget_logit_gradient(
                            state,
                            cfg.render,
                            cfg.scale_budget_weight,
                        ));
                target_update[base + model.config.spatial_dims + scale_channel] +=
                    scale_update.clamp(-cfg.max_opacity_update, cfg.max_opacity_update);
            }
        }
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}
