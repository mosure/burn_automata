#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(dead_code)]
pub(crate) fn terminal_render_state_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    opacity_gain: f32,
    scale_gain: f32,
    scale_budget_weight: f32,
    liveness_gain: f32,
    liveness_front_radius: f32,
    liveness_step_fraction: f32,
    max_opacity_update: f32,
    render_cfg: RenderLossConfig,
    rows: usize,
) -> Vec<f32> {
    terminal_render_state_adjoint_weighted(
        config,
        trace,
        gradient,
        opacity_gain,
        scale_gain,
        scale_budget_weight,
        liveness_gain,
        liveness_front_radius,
        liveness_step_fraction,
        max_opacity_update,
        render_cfg,
        rows,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn terminal_render_state_adjoint_weighted(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    gradient: &RenderProxyGradientRows,
    opacity_gain: f32,
    scale_gain: f32,
    scale_budget_weight: f32,
    liveness_gain: f32,
    liveness_front_radius: f32,
    liveness_step_fraction: f32,
    max_opacity_update: f32,
    render_cfg: RenderLossConfig,
    rows: usize,
    row_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut state_adjoint = vec![0.0; trace.states.len()];
    for (gradient_row, &particle_row) in gradient.row_indices.iter().enumerate().take(rows) {
        if particle_row * trace.state_dims + config.state_dims > trace.states.len() {
            continue;
        }
        let row_weight = row_weights
            .and_then(|weights| weights.get(particle_row))
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if row_weight <= 0.0 {
            continue;
        }
        let state_base = particle_row * trace.state_dims;
        if opacity_gain > 0.0
            && let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims)
        {
            let final_logit =
                trace.states[state_base + opacity_channel] + render_cfg.opacity_logit_bias;
            state_adjoint[state_base + opacity_channel] += row_weight
                * opacity_gain
                * gradient.opacity_gradients[gradient_row]
                * sigmoid_unit_derivative(final_logit);
        }
        if scale_gain > 0.0
            && render_cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale
            && config.state_dims >= 5
        {
            let scale_channel = config.state_dims - 5;
            state_adjoint[state_base + scale_channel] +=
                row_weight * scale_gain * gradient.scale_gradients[gradient_row];
        }
        if config.state_dims >= 3 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                let state_value = trace.states[state_base + tail + channel];
                if state_value > -1.0 && state_value < 1.0 {
                    state_adjoint[state_base + tail + channel] +=
                        row_weight * gradient.color_gradients[gradient_row][channel];
                }
            }
        }
    }
    add_gaussian_scale_budget_state_adjoint(
        config,
        trace,
        render_cfg,
        scale_budget_weight,
        &mut state_adjoint,
    );
    add_liveness_front_state_adjoint(
        config,
        &trace.positions,
        &trace.states,
        liveness_gain,
        liveness_front_radius,
        liveness_step_fraction,
        max_opacity_update,
        &mut state_adjoint,
    );
    add_temporal_activation_schedule_state_adjoint(
        config,
        &trace.positions,
        &trace.states,
        liveness_gain,
        liveness_front_radius,
        liveness_step_fraction,
        max_opacity_update,
        &mut state_adjoint,
    );
    state_adjoint
}
