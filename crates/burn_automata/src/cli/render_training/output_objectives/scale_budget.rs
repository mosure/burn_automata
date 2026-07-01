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
    let expected_scale = render_cfg
        .sigma
        .clamp(render_cfg.min_sigma, render_cfg.max_sigma)
        .max(1.0e-8);
    let scale_logit = state[state.len() - 5].clamp(-8.0, 8.0);
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
