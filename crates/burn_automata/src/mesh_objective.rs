use serde::{Deserialize, Serialize};

use crate::{
    GaussianDecodeMode, RenderLossConfig, RolloutTrace, rollout::growth_3d_material_opacity_channel,
};

pub const ROBUST_3D_COVERAGE_GAIN: f32 = 0.35;
pub const ROBUST_3D_COVERAGE_SAMPLES: usize = 4096;
pub const ROBUST_3D_COVERAGE_REPULSION_GAIN: f32 = 0.20;
pub const ROBUST_3D_COVERAGE_NORMAL_WEIGHT: f32 = 0.35;
pub const ROBUST_3D_EXTENT_GAIN: f32 = 0.10;
pub const ROBUST_3D_SURFACE_GAIN: f32 = 0.025;
pub const ROBUST_3D_SURFACE_ESCAPE_GAIN: f32 = 0.50;
pub const ROBUST_3D_OPACITY_GAIN: f32 = 0.10;
pub const ROBUST_3D_MATERIAL_LIVENESS_GAIN: f32 = ROBUST_3D_OPACITY_GAIN;
pub const ROBUST_3D_MATERIAL_TAIL_GAIN: f32 = ROBUST_3D_OPACITY_GAIN;
pub const ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER: f32 = 5.0;
pub const ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE: f32 = 0.75;
pub const ROBUST_3D_SCALE_GAIN: f32 = 0.05;
pub const ROBUST_3D_TRAJECTORY_RENDER_GAIN: f32 = 0.05;
pub const ROBUST_3D_TRAJECTORY_MESH_GAIN: f32 = 0.05;
pub const ROBUST_3D_TRAJECTORY_RENDER_SAMPLES: usize = 4;
pub const ROBUST_3D_LIVENESS_GAIN: f32 = 0.05;
pub const ROBUST_3D_PHASE_GAIN: f32 = 0.10;
pub const ROBUST_3D_LIVENESS_FRONT_RADIUS: f32 = 0.24;
pub const ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER: f32 = 24.0;
pub const ROBUST_3D_SCALE_BUDGET_WEIGHT: f32 = 0.05;
pub const ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP: f32 = 0.05;
pub const ROBUST_3D_MAX_SCALE_BUDGET_LOSS: f32 = 0.25;
pub const ROBUST_3D_MAX_OVERSIZE_FRACTION: f32 = 0.05;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MeshRolloutObjectiveConfig {
    pub gaussian_decode_mode: GaussianDecodeMode,
    pub render_sigma: f32,
    pub render_min_sigma: f32,
    pub render_max_sigma: f32,
    pub render_density_weight: f32,
    pub render_color_weight: f32,
    pub render_depth_weight: f32,
    pub surface_gain: f32,
    pub surface_escape_gain: f32,
    pub coverage_gain: f32,
    pub coverage_samples: usize,
    pub coverage_repulsion_gain: f32,
    pub coverage_normal_weight: f32,
    pub extent_gain: f32,
    pub trajectory_render_gain: f32,
    pub trajectory_mesh_gain: f32,
    pub trajectory_render_samples: usize,
    pub liveness_gain: f32,
    pub phase_gain: f32,
    pub liveness_front_radius: f32,
    pub liveness_update_multiplier: f32,
    pub opacity_gain: f32,
    pub material_liveness_gain: f32,
    pub material_tail_gain: f32,
    pub material_suppression_update_multiplier: f32,
    pub material_max_opacity_update: f32,
    pub gaussian_scale_gain: f32,
    pub gaussian_scale_budget_weight: f32,
}

impl MeshRolloutObjectiveConfig {
    pub fn robust_3d(render: RenderLossConfig) -> Self {
        Self {
            gaussian_decode_mode: render.gaussian_decode_mode,
            render_sigma: render.sigma,
            render_min_sigma: render.min_sigma,
            render_max_sigma: render.max_sigma,
            render_density_weight: render.density_weight,
            render_color_weight: render.color_weight,
            render_depth_weight: render.depth_weight,
            surface_gain: ROBUST_3D_SURFACE_GAIN,
            surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
            coverage_gain: ROBUST_3D_COVERAGE_GAIN,
            coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
            coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
            coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
            extent_gain: ROBUST_3D_EXTENT_GAIN,
            trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
            trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
            trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
            liveness_gain: ROBUST_3D_LIVENESS_GAIN,
            phase_gain: ROBUST_3D_PHASE_GAIN,
            liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
            liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
            opacity_gain: ROBUST_3D_OPACITY_GAIN,
            material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
            material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
            material_suppression_update_multiplier:
                ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
            material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
            gaussian_scale_gain: ROBUST_3D_SCALE_GAIN,
            gaussian_scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GaussianVolumeStats {
    pub particles: usize,
    pub visible_particles: usize,
    pub expected_scale: f32,
    pub max_expected_scale: f32,
    pub mean_scale: f32,
    pub max_scale: f32,
    pub mean_opacity: f32,
    pub max_opacity: f32,
    pub mean_visible_volume: f32,
    pub max_visible_volume: f32,
    pub oversize_fraction: f32,
    pub scale_budget_loss: f32,
}

impl GaussianVolumeStats {
    pub fn from_trace(
        trace: &RolloutTrace,
        fixed_scale: f32,
        max_expected_scale: f32,
        opacity_logit_bias: f32,
    ) -> Self {
        if trace.particle_count == 0 || trace.state_dims == 0 {
            return Self::default();
        }
        let mut visible_particles = 0usize;
        let mut scale_sum = 0.0_f32;
        let mut opacity_sum = 0.0_f32;
        let mut volume_sum = 0.0_f32;
        let mut max_scale = 0.0_f32;
        let mut max_opacity = 0.0_f32;
        let mut max_volume = 0.0_f32;
        let mut oversize = 0usize;
        let mut scale_budget_loss = 0.0_f32;
        let scale = fixed_scale.max(1.0e-8);
        let volume = scale * scale * scale;
        let expected_scale = fixed_scale.max(1.0e-8);
        let max_expected_scale = max_expected_scale.max(1.0e-8);
        let opacity_channel = growth_3d_material_opacity_channel(trace.state_dims);

        for particle in 0..trace.particle_count {
            let state_base = particle * trace.state_dims;
            let opacity = opacity_channel
                .and_then(|channel| trace.states.get(state_base + channel))
                .map(|logit| sigmoid(*logit + opacity_logit_bias).clamp(0.001, 0.95))
                .unwrap_or(1.0);
            scale_sum += scale;
            opacity_sum += opacity;
            volume_sum += volume * opacity;
            max_scale = max_scale.max(scale);
            max_opacity = max_opacity.max(opacity);
            max_volume = max_volume.max(volume * opacity);
            if opacity > 0.01 {
                visible_particles += 1;
            }
            if scale > max_expected_scale {
                oversize += 1;
            }
            scale_budget_loss += scale_budget_loss_for_scale(scale, expected_scale);
        }

        let count = trace.particle_count.max(1) as f32;
        Self {
            particles: trace.particle_count,
            visible_particles,
            expected_scale,
            max_expected_scale,
            mean_scale: scale_sum / count,
            max_scale,
            mean_opacity: opacity_sum / count,
            max_opacity,
            mean_visible_volume: volume_sum / count,
            max_visible_volume: max_volume,
            oversize_fraction: oversize as f32 / count,
            scale_budget_loss: scale_budget_loss / count,
        }
    }

    pub fn from_render_config(trace: &RolloutTrace, cfg: RenderLossConfig) -> Self {
        if trace.particle_count == 0 || trace.state_dims == 0 {
            return Self::default();
        }
        let expected_scale = cfg.sigma.clamp(cfg.min_sigma, cfg.max_sigma).max(1.0e-8);
        let max_expected_scale = (expected_scale * 2.0)
            .min(cfg.max_sigma)
            .max(expected_scale);
        let opacity_channel = growth_3d_material_opacity_channel(trace.state_dims);
        let mut visible_particles = 0usize;
        let mut scale_sum = 0.0_f32;
        let mut opacity_sum = 0.0_f32;
        let mut volume_sum = 0.0_f32;
        let mut max_scale = 0.0_f32;
        let mut max_opacity = 0.0_f32;
        let mut max_volume = 0.0_f32;
        let mut oversize = 0usize;
        let mut scale_budget_loss = 0.0_f32;

        for particle in 0..trace.particle_count {
            let state_base = particle * trace.state_dims;
            let state = &trace.states[state_base..state_base + trace.state_dims];
            let scale = render_scale_for_particle(state, cfg);
            let opacity = opacity_channel
                .and_then(|channel| trace.states.get(state_base + channel))
                .map(|logit| sigmoid(*logit + cfg.opacity_logit_bias).clamp(0.001, 0.95))
                .unwrap_or(1.0);
            let volume = scale * scale * scale * opacity;
            scale_sum += scale;
            opacity_sum += opacity;
            volume_sum += volume;
            max_scale = max_scale.max(scale);
            max_opacity = max_opacity.max(opacity);
            max_volume = max_volume.max(volume);
            if opacity > 0.01 {
                visible_particles += 1;
            }
            if scale > max_expected_scale {
                oversize += 1;
            }
            scale_budget_loss += scale_budget_loss_for_scale(scale, expected_scale);
        }

        let count = trace.particle_count.max(1) as f32;
        Self {
            particles: trace.particle_count,
            visible_particles,
            expected_scale,
            max_expected_scale,
            mean_scale: scale_sum / count,
            max_scale,
            mean_opacity: opacity_sum / count,
            max_opacity,
            mean_visible_volume: volume_sum / count,
            max_visible_volume: max_volume,
            oversize_fraction: oversize as f32 / count,
            scale_budget_loss: scale_budget_loss / count,
        }
    }
}

pub fn scale_budget_loss_for_scale(scale: f32, expected_scale: f32) -> f32 {
    if !scale.is_finite() || !expected_scale.is_finite() || expected_scale <= 0.0 {
        return f32::INFINITY;
    }
    let oversize_ratio = (scale / expected_scale - 1.0).max(0.0);
    oversize_ratio * oversize_ratio
}

fn render_scale_for_particle(state: &[f32], cfg: RenderLossConfig) -> f32 {
    match cfg.gaussian_decode_mode {
        GaussianDecodeMode::ParticlePoint => cfg.min_sigma.max(1.0e-8),
        GaussianDecodeMode::GaussianSh0FixedScale | GaussianDecodeMode::GaussianSh0Oriented => {
            cfg.sigma.clamp(cfg.min_sigma, cfg.max_sigma)
        }
        GaussianDecodeMode::GaussianSh0LearnedScale => {
            if state.len() < 5 {
                return cfg.sigma.clamp(cfg.min_sigma, cfg.max_sigma);
            }
            let logit = state[state.len() - 5].clamp(-8.0, 8.0);
            (cfg.sigma * logit.exp()).clamp(cfg.min_sigma, cfg.max_sigma)
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_3d_defaults_are_hybrid_not_point_only() {
        let render = RenderLossConfig {
            density_weight: 1.5,
            color_weight: 0.5,
            depth_weight: 2.0,
            ..RenderLossConfig::default()
        };
        let cfg = MeshRolloutObjectiveConfig::robust_3d(render);

        assert_eq!(
            cfg.gaussian_decode_mode,
            GaussianDecodeMode::GaussianSh0FixedScale
        );
        assert_eq!(cfg.render_sigma, render.sigma);
        assert_eq!(cfg.render_density_weight, 1.5);
        assert!(cfg.coverage_gain > 0.0);
        assert!(cfg.coverage_samples > 0);
        assert!(cfg.coverage_repulsion_gain > 0.0);
        assert!(cfg.coverage_normal_weight > 0.0);
        assert!(cfg.surface_gain > 0.0);
        assert!(cfg.surface_escape_gain > 0.0);
        assert!(cfg.trajectory_render_gain > 0.0);
        assert!(cfg.trajectory_mesh_gain > 0.0);
        assert!(cfg.trajectory_render_samples > 0);
        assert!(cfg.liveness_gain > 0.0);
        assert!(cfg.liveness_front_radius > 0.0);
        assert!(cfg.opacity_gain > 0.0);
        assert!(cfg.gaussian_scale_gain > 0.0);
    }

    #[test]
    fn gaussian_volume_stats_reports_oversized_visible_particles() {
        let state_dims = 9;
        let trace = RolloutTrace {
            positions: vec![[0.0; 4]; 2],
            states: vec![
                0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.0, 2.0, //
                0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.0, -8.0,
            ],
            batch_size: 1,
            particle_count: 2,
            state_dims,
            steps: 1,
            mean_dx: vec![0.0],
        };

        let stats = GaussianVolumeStats::from_trace(&trace, 0.05, 0.02, 0.0);

        assert_eq!(stats.particles, 2);
        assert_eq!(stats.visible_particles, 1);
        assert_eq!(stats.oversize_fraction, 1.0);
        assert!(stats.mean_opacity > 0.4);
        assert!(stats.max_visible_volume > stats.mean_visible_volume);
    }

    #[test]
    fn gaussian_volume_stats_uses_learned_render_scale() {
        let state_dims = 16;
        let scale_channel = state_dims - 5;
        let opacity_channel = growth_3d_material_opacity_channel(state_dims).unwrap();
        let mut states = vec![0.0; 2 * state_dims];
        states[scale_channel] = 1.5;
        states[state_dims + scale_channel] = 0.0;
        states[opacity_channel] = 8.0;
        states[state_dims + opacity_channel] = 8.0;
        let trace = RolloutTrace {
            positions: vec![[0.0; 4]; 2],
            states,
            batch_size: 1,
            particle_count: 2,
            state_dims,
            steps: 1,
            mean_dx: vec![0.0],
        };

        let stats = GaussianVolumeStats::from_render_config(
            &trace,
            RenderLossConfig {
                gaussian_decode_mode: GaussianDecodeMode::GaussianSh0LearnedScale,
                sigma: 1.0,
                min_sigma: 0.25,
                max_sigma: 6.0,
                ..RenderLossConfig::default()
            },
        );

        assert!(stats.max_scale > stats.mean_scale);
        assert!(stats.max_visible_volume > stats.mean_visible_volume);
    }
}
