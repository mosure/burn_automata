#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn add_gaussian_scale_budget_state_adjoint(
    config: &NpaConfig,
    trace: &crate::RolloutTrace,
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
    state_adjoint: &mut [f32],
) {
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || render_cfg.gaussian_decode_mode != GaussianDecodeMode::GaussianSh0LearnedScale
        || config.state_dims < 5
    {
        return;
    }
    let scale_channel = config.state_dims - 5;
    for particle_row in 0..trace.particle_count {
        let state_base = particle_row * trace.state_dims;
        if state_base + config.state_dims > trace.states.len()
            || state_base + scale_channel >= state_adjoint.len()
        {
            continue;
        }
        let state = &trace.states[state_base..state_base + config.state_dims];
        state_adjoint[state_base + scale_channel] +=
            gaussian_scale_budget_logit_gradient(state, render_cfg, scale_budget_weight);
    }
}

pub(crate) fn add_gaussian_scale_budget_output_objective(
    config: &NpaConfig,
    states: &[f32],
    raw_updates: &[f32],
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
    max_scale_update: f32,
    output_gradients: &mut [f32],
) -> Option<usize> {
    let scale_output = growth_3d_scale_output_channel(config, render_cfg)?;
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || states.is_empty()
        || raw_updates.is_empty()
        || output_gradients.is_empty()
    {
        return Some(scale_output);
    }

    let scale_channel = config.state_dims - 5;
    let output_dims = config.update_dims();
    let rows = states.len() / config.state_dims;
    if raw_updates.len() < rows.saturating_mul(output_dims)
        || output_gradients.len() < rows.saturating_mul(output_dims)
    {
        return Some(scale_output);
    }

    let max_scale_update = if max_scale_update.is_finite() && max_scale_update > 0.0 {
        max_scale_update
    } else {
        f32::INFINITY
    };
    for row in 0..rows {
        let state_base = row * config.state_dims;
        let output_index = row * output_dims + scale_output;
        let predicted_logit = states[state_base + scale_channel] + raw_updates[output_index];
        let pressure = gaussian_scale_budget_logit_gradient_for_logit(
            predicted_logit,
            render_cfg,
            scale_budget_weight,
        );
        if pressure <= 0.0 || !pressure.is_finite() {
            continue;
        }
        let target_update = (-pressure).clamp(-max_scale_update, 0.0);
        output_gradients[output_index] += raw_updates[output_index] - target_update;
    }
    Some(scale_output)
}

pub(crate) fn growth_3d_scale_output_channel(
    config: &NpaConfig,
    render_cfg: RenderLossConfig,
) -> Option<usize> {
    (render_cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale
        && config.state_dims >= 5)
        .then(|| config.spatial_dims + config.state_dims - 5)
        .filter(|channel| *channel < config.update_dims())
}

pub(crate) fn gaussian_scale_budget_logit_gradient(
    state: &[f32],
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
) -> f32 {
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || render_cfg.gaussian_decode_mode != GaussianDecodeMode::GaussianSh0LearnedScale
        || state.len() < 5
    {
        return 0.0;
    }
    gaussian_scale_budget_logit_gradient_for_logit(
        state[state.len() - 5],
        render_cfg,
        scale_budget_weight,
    )
}

pub(crate) fn gaussian_scale_budget_logit_gradient_for_logit(
    scale_logit: f32,
    render_cfg: RenderLossConfig,
    scale_budget_weight: f32,
) -> f32 {
    if scale_budget_weight <= 0.0
        || !scale_budget_weight.is_finite()
        || render_cfg.gaussian_decode_mode != GaussianDecodeMode::GaussianSh0LearnedScale
    {
        return 0.0;
    }
    let expected_scale = render_cfg
        .sigma
        .clamp(render_cfg.min_sigma, render_cfg.max_sigma)
        .max(1.0e-8);
    let scale_logit = scale_logit.clamp(-8.0, 8.0);
    let scale = (render_cfg.sigma * scale_logit.exp())
        .clamp(render_cfg.min_sigma, render_cfg.max_sigma)
        .max(1.0e-8);
    let loss = scale_budget_loss_for_scale(scale, expected_scale);
    if !loss.is_finite() || loss <= 0.0 {
        return 0.0;
    }
    let oversize_ratio = scale / expected_scale - 1.0;
    2.0 * scale_budget_weight * oversize_ratio * scale / expected_scale
}
