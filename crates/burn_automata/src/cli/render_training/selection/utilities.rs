use super::*;

pub(crate) fn finite_report_metric(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

pub(crate) fn material_training_soft_coverage_threshold(seed_scale: f32) -> f32 {
    let strict = target_coverage_threshold(seed_scale).max(1.0e-6);
    (strict * 3.0).min(seed_scale.max(strict)).max(strict)
}

pub(crate) fn material_training_frontier_coverage_threshold(seed_scale: f32) -> f32 {
    let soft = material_training_soft_coverage_threshold(seed_scale);
    soft.max(seed_scale.max(soft) * 1.25).max(soft)
}

pub(crate) fn material_opacity_frontier_coverage_threshold(seed_scale: f32) -> f32 {
    let soft = material_training_soft_coverage_threshold(seed_scale);
    (soft * 1.5).max(soft)
}

pub(crate) fn direct_trajectory_geometry_weight(step_fraction: f32) -> f32 {
    let schedule = step_fraction.clamp(0.0, 1.0);
    0.5 + 0.5 * schedule
}

pub(crate) fn direct_material_surface_motion_weight(
    trajectory_mesh_gain: f32,
    coverage_gain: f32,
    step_fraction: f32,
) -> f32 {
    let schedule = direct_trajectory_geometry_weight(step_fraction);
    let trajectory_weight = finite_positive(trajectory_mesh_gain)
        .map(|gain| gain * schedule)
        .unwrap_or(0.0);
    let coverage_weight = finite_positive(coverage_gain)
        .map(|gain| gain * DIRECT_GROWTH_MATERIAL_SURFACE_MOTION_COVERAGE_GAIN_FRACTION * schedule)
        .unwrap_or(0.0);
    trajectory_weight.max(coverage_weight)
}

pub(crate) fn direct_growth_phase_gain(cfg: &RenderProxyTrainingConfig) -> f32 {
    if cfg.liveness_gain <= 0.0 || !cfg.liveness_gain.is_finite() {
        return 0.0;
    }
    (cfg.liveness_gain * DIRECT_GROWTH_PHASE_GAIN_FRACTION).max(ROBUST_3D_PHASE_GAIN)
}

fn finite_positive(value: f32) -> Option<f32> {
    (value > 0.0 && value.is_finite()).then_some(value)
}

pub(crate) fn soft_material_assignment_weight(
    distance: f32,
    strict_threshold: f32,
    soft_threshold: f32,
) -> f32 {
    if !distance.is_finite() {
        return 0.0;
    }
    let strict = strict_threshold.max(1.0e-6);
    let soft = soft_threshold.max(strict);
    if distance <= strict {
        1.0
    } else if distance >= soft {
        0.0
    } else {
        (1.0 - (distance - strict) / (soft - strict).max(1.0e-6)).clamp(0.0, 1.0)
    }
}

pub(crate) fn frontier_material_assignment_weight(
    distance: f32,
    strict_threshold: f32,
    soft_threshold: f32,
    frontier_threshold: f32,
) -> f32 {
    let soft_weight = soft_material_assignment_weight(distance, strict_threshold, soft_threshold);
    if soft_weight > 0.0 || !distance.is_finite() {
        return soft_weight;
    }
    let soft = soft_threshold.max(strict_threshold.max(1.0e-6));
    let frontier = frontier_threshold.max(soft);
    if distance >= frontier {
        return 0.0;
    }
    let falloff = 1.0 - (distance - soft) / (frontier - soft).max(1.0e-6);
    0.25 * falloff.clamp(0.0, 1.0).powi(2)
}
