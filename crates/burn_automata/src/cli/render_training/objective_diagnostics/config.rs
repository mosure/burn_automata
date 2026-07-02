use super::*;

pub(crate) fn render_training_objective_config(
    cfg: &RenderProxyTrainingConfig,
    render: RenderLossConfig,
) -> MeshRolloutObjectiveConfig {
    MeshRolloutObjectiveConfig {
        gaussian_decode_mode: render.gaussian_decode_mode,
        render_sigma: render.sigma,
        render_min_sigma: render.min_sigma,
        render_max_sigma: render.max_sigma,
        render_density_weight: render.density_weight,
        render_color_weight: render.color_weight,
        render_depth_weight: render.depth_weight,
        surface_gain: cfg.surface_gain,
        surface_escape_gain: cfg.surface_escape_gain,
        coverage_gain: cfg.coverage_gain,
        coverage_samples: cfg.coverage_samples,
        coverage_repulsion_gain: cfg.coverage_repulsion_gain,
        coverage_normal_weight: cfg.coverage_normal_weight,
        extent_gain: cfg.extent_gain,
        trajectory_render_gain: cfg.trajectory_render_gain,
        trajectory_mesh_gain: cfg.trajectory_mesh_gain,
        trajectory_render_samples: cfg.trajectory_render_samples,
        liveness_gain: cfg.liveness_gain,
        phase_gain: direct_growth_phase_gain(cfg),
        liveness_front_radius: cfg.liveness_front_radius,
        liveness_update_multiplier: cfg.liveness_update_multiplier,
        opacity_gain: cfg.opacity_gain,
        material_liveness_gain: cfg.material_liveness_gain,
        material_tail_gain: cfg.material_tail_gain,
        material_suppression_update_multiplier: cfg.material_suppression_update_multiplier,
        material_max_opacity_update: cfg.material_max_opacity_update,
        gaussian_scale_gain: cfg.scale_gain,
        gaussian_scale_budget_weight: cfg.scale_budget_weight,
    }
}

pub(crate) fn material_suppression_max_update(
    max_opacity_update: f32,
    material_suppression_update_multiplier: f32,
) -> f32 {
    if max_opacity_update.is_finite()
        && max_opacity_update > 0.0
        && material_suppression_update_multiplier.is_finite()
        && material_suppression_update_multiplier > 0.0
    {
        max_opacity_update * material_suppression_update_multiplier
    } else {
        max_opacity_update
    }
}

pub(crate) fn liveness_max_update(max_opacity_update: f32, liveness_update_multiplier: f32) -> f32 {
    if max_opacity_update.is_finite()
        && max_opacity_update > 0.0
        && liveness_update_multiplier.is_finite()
        && liveness_update_multiplier > 0.0
    {
        max_opacity_update * liveness_update_multiplier
    } else {
        max_opacity_update
    }
}
