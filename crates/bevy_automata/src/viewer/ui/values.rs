use super::*;

pub(in crate::viewer) fn slider_value_for_settings(
    kind: AutomataSliderKind,
    settings: &AutomataSettings,
) -> f32 {
    match kind {
        AutomataSliderKind::ParticleLog2 => particle_slider_value(settings.particle_count),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame as f32,
        AutomataSliderKind::UpdateProb => settings.update_prob,
        AutomataSliderKind::DtLog2 => log2_slider_value(settings.dt),
        AutomataSliderKind::RenderScaleLog2 => log2_slider_value(settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => log2_slider_value(settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            log2_slider_value(settings.training_learning_rate)
        }
        AutomataSliderKind::TrainingRolloutResetInterval => {
            settings.training_rollout_reset_interval as f32
        }
    }
}

pub(in crate::viewer) fn slider_label(
    kind: AutomataSliderKind,
    settings: &AutomataSettings,
) -> String {
    match kind {
        AutomataSliderKind::ParticleLog2 => settings.particle_count.to_string(),
        AutomataSliderKind::StepsPerFrame => settings.steps_per_frame.to_string(),
        AutomataSliderKind::UpdateProb => format!("{:.2}", settings.update_prob),
        AutomataSliderKind::DtLog2 => format!("{:.3}", settings.dt),
        AutomataSliderKind::RenderScaleLog2 => format!("{:.2}x", settings.render_scale),
        AutomataSliderKind::RenderOpacityLog2 => format!("{:.2}x", settings.render_opacity),
        AutomataSliderKind::TrainingLearningRateLog2 => {
            format!("{:.4}", settings.training_learning_rate)
        }
        AutomataSliderKind::TrainingRolloutResetInterval => {
            if settings.training_rollout_reset_interval == 0 {
                "off".to_string()
            } else {
                format!("{} steps", settings.training_rollout_reset_interval)
            }
        }
    }
}

pub(in crate::viewer) fn log2_slider_value(value: f32) -> f32 {
    value.max(f32::MIN_POSITIVE).log2()
}

pub(in crate::viewer) fn exp2_slider_value(value: f32) -> f32 {
    2.0_f32.powf(value)
}

pub(in crate::viewer) fn particle_slider_value(particles: usize) -> f32 {
    (particles.max(64) as f32).log2().clamp(6.0, 16.0)
}

pub(in crate::viewer) fn particles_from_slider_value(value: f32) -> usize {
    let log2 = value.round().clamp(6.0, 16.0) as u32;
    1usize << log2
}
