use super::{EPS, RENDER_MAX_OPACITY, RENDER_MIN_OPACITY, RenderLossConfig, math::sigmoid};
use crate::{
    AutomataError, AutomataResult, RolloutTrace, rollout::growth_3d_material_opacity_channel,
};
use burn_automata_kernels::GaussianDecodeMode;

#[derive(Clone, Debug)]
pub(super) struct RenderParticleAttributes {
    pub(super) colors: Vec<[f32; 3]>,
    pub(super) opacities: Vec<f32>,
    pub(super) sigmas: Vec<f32>,
    pub(super) scale_logit_derivatives: Vec<f32>,
}

pub(super) fn state_tail_render_attributes(
    states: &[f32],
    state_dims: usize,
    cfg: RenderLossConfig,
) -> AutomataResult<RenderParticleAttributes> {
    if state_dims < 3 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss needs at least 3 state channels for color, got {state_dims}"
        )));
    }
    if cfg.gaussian_decode_mode == GaussianDecodeMode::GaussianSh0LearnedScale && state_dims < 5 {
        return Err(AutomataError::InvalidArgument(format!(
            "learned-scale render decode needs at least 5 state channels, got {state_dims}"
        )));
    }
    if !states.len().is_multiple_of(state_dims) {
        return Err(AutomataError::InvalidArgument(format!(
            "state len {} is not divisible by state_dims {state_dims}",
            states.len()
        )));
    }
    let count = states.len() / state_dims;
    let tail = state_dims - 3;
    let mut colors = Vec::with_capacity(count);
    let mut opacities = Vec::with_capacity(count);
    let mut sigmas = Vec::with_capacity(count);
    let mut scale_logit_derivatives = Vec::with_capacity(count);
    for idx in 0..count {
        let base = idx * state_dims + tail;
        colors.push([
            (states[base] + 0.5).clamp(0.0, 1.0),
            (states[base + 1] + 0.5).clamp(0.0, 1.0),
            (states[base + 2] + 0.5).clamp(0.0, 1.0),
        ]);
        let opacity = if let Some(channel) = growth_3d_material_opacity_channel(state_dims) {
            sigmoid(states[idx * state_dims + channel] + cfg.opacity_logit_bias)
                .clamp(RENDER_MIN_OPACITY, RENDER_MAX_OPACITY)
        } else {
            RENDER_MAX_OPACITY
        };
        opacities.push(opacity);
        let (sigma, derivative) = match cfg.gaussian_decode_mode {
            GaussianDecodeMode::ParticlePoint => (cfg.min_sigma.max(EPS), 0.0),
            GaussianDecodeMode::GaussianSh0FixedScale | GaussianDecodeMode::GaussianSh0Oriented => {
                (cfg.sigma.clamp(cfg.min_sigma, cfg.max_sigma), 0.0)
            }
            GaussianDecodeMode::GaussianSh0LearnedScale => {
                let channel = state_dims - 5;
                let logit = states[idx * state_dims + channel].clamp(-8.0, 8.0);
                let raw_sigma = cfg.sigma * logit.exp();
                let sigma = raw_sigma.clamp(cfg.min_sigma, cfg.max_sigma);
                let derivative = if raw_sigma > cfg.min_sigma && raw_sigma < cfg.max_sigma {
                    raw_sigma
                } else {
                    0.0
                };
                (sigma, derivative)
            }
        };
        sigmas.push(sigma);
        scale_logit_derivatives.push(derivative);
    }
    Ok(RenderParticleAttributes {
        colors,
        opacities,
        sigmas,
        scale_logit_derivatives,
    })
}

pub(super) fn validate_render_trace(
    trace: &RolloutTrace,
    cfg: RenderLossConfig,
) -> AutomataResult<()> {
    validate_render_loss_config(cfg)?;
    if trace.batch_size != 1 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss expects batch_size=1, got {}",
            trace.batch_size
        )));
    }
    if trace.positions.len() != trace.particle_count {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss expects final trace positions for one batch, got {} for {} particles",
            trace.positions.len(),
            trace.particle_count
        )));
    }
    if trace.states.len() != trace.particle_count * trace.state_dims {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss state len {} does not match particles {} * state_dims {}",
            trace.states.len(),
            trace.particle_count,
            trace.state_dims
        )));
    }
    Ok(())
}

fn validate_render_loss_config(cfg: RenderLossConfig) -> AutomataResult<()> {
    if cfg.image_size == 0 {
        return Err(AutomataError::InvalidArgument(
            "render loss image_size must be non-zero".to_string(),
        ));
    }
    if cfg.target_samples == 0 {
        return Err(AutomataError::InvalidArgument(
            "render loss target_samples must be non-zero".to_string(),
        ));
    }
    if !cfg.sigma.is_finite() || cfg.sigma <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss sigma must be finite and positive, got {}",
            cfg.sigma
        )));
    }
    if !cfg.min_sigma.is_finite() || cfg.min_sigma <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss min_sigma must be finite and positive, got {}",
            cfg.min_sigma
        )));
    }
    if !cfg.max_sigma.is_finite() || cfg.max_sigma < cfg.min_sigma {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss max_sigma must be finite and >= min_sigma, got max={} min={}",
            cfg.max_sigma, cfg.min_sigma
        )));
    }
    if !cfg.world_scale.is_finite() || cfg.world_scale <= 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss world_scale must be finite and positive, got {}",
            cfg.world_scale
        )));
    }
    if !cfg.opacity_logit_bias.is_finite() {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss opacity_logit_bias must be finite, got {}",
            cfg.opacity_logit_bias
        )));
    }
    if !cfg.density_weight.is_finite() || cfg.density_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss density_weight must be finite and non-negative, got {}",
            cfg.density_weight
        )));
    }
    if !cfg.color_weight.is_finite() || cfg.color_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss color_weight must be finite and non-negative, got {}",
            cfg.color_weight
        )));
    }
    if !cfg.depth_weight.is_finite() || cfg.depth_weight < 0.0 {
        return Err(AutomataError::InvalidArgument(format!(
            "render loss depth_weight must be finite and non-negative, got {}",
            cfg.depth_weight
        )));
    }
    Ok(())
}
