#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_material_opacity_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_opacity_gain: f32,
    seed_scale: f32,
    target_opacity_logit: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
        || seed_scale <= 0.0
        || !seed_scale.is_finite()
        || !target_opacity_logit.is_finite()
    {
        return;
    }
    let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let soft_threshold = material_training_soft_coverage_threshold(seed_scale);
    let frontier_threshold = material_training_frontier_coverage_threshold(seed_scale);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        if liveness <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        let opacity_index = state_base + opacity_channel;
        let current_opacity = states[opacity_index];
        let surface_weight = frontier_material_assignment_weight(
            projection.distance,
            strict_threshold,
            soft_threshold,
            frontier_threshold,
        );
        if surface_weight <= 0.0 {
            continue;
        }
        let adjoint =
            (material_opacity_gain * surface_weight * (current_opacity - target_opacity_logit))
                .clamp(-max_adjoint, max_adjoint);
        state_adjoint[opacity_index] += adjoint;
    }
}
