#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn add_material_liveness_state_adjoint(
    config: &NpaConfig,
    states: &[f32],
    material_opacity_gain: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || states.is_empty()
        || state_adjoint.len() < states.len()
        || material_opacity_gain <= 0.0
        || !material_opacity_gain.is_finite()
    {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    for (row, state) in states.chunks_exact(config.state_dims).enumerate() {
        let liveness = state[GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = state[material_channel];
        if liveness > -1.0 || material_opacity <= GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT {
            continue;
        }
        let adjoint = (material_opacity_gain
            * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT))
            .clamp(0.0, max_adjoint);
        state_adjoint[row * config.state_dims + material_channel] += adjoint;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn material_visible_liveness_target_updates(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    surface_threshold: f32,
    max_update: f32,
) -> Vec<f32> {
    let mut updates = vec![0.0_f32; positions.len()];
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || liveness_gain <= 0.0
        || !liveness_gain.is_finite()
        || surface_threshold <= 0.0
        || !surface_threshold.is_finite()
    {
        return updates;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return updates;
    };
    let max_update = if max_update.is_finite() && max_update > 0.0 {
        max_update
    } else {
        f32::INFINITY
    };
    let material_visible_threshold = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT + 1.0;
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness = states[state_base + GROWTH_3D_LIVENESS_CHANNEL];
        let material_opacity = states[state_base + material_channel];
        if liveness > -1.0 || material_opacity <= material_visible_threshold {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > surface_threshold {
            continue;
        }
        let surface_weight = (1.0 - projection.distance / surface_threshold).clamp(0.0, 1.0);
        let material_weight =
            ((material_opacity - material_visible_threshold) / 4.0).clamp(0.25, 1.0);
        let target_liveness = 0.0_f32;
        updates[row] =
            (liveness_gain * surface_weight * material_weight * (target_liveness - liveness))
                .clamp(0.0, max_update);
    }
    updates
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_liveness_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    liveness_gain: f32,
    surface_threshold: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if state_adjoint.len() < states.len() {
        return;
    }
    let updates = material_visible_liveness_target_updates(
        config,
        target,
        positions,
        states,
        liveness_gain,
        surface_threshold,
        max_adjoint,
    );
    for (row, update) in updates.iter().copied().enumerate() {
        if update == 0.0 {
            continue;
        }
        state_adjoint[row * config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] -= update;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_material_visible_surface_tail_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    material_tail_gain: f32,
    surface_threshold: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || material_tail_gain <= 0.0
        || !material_tail_gain.is_finite()
        || surface_threshold <= 0.0
        || !surface_threshold.is_finite()
    {
        return;
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let threshold = surface_threshold.max(1.0e-6);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let material_index = state_base + material_channel;
        let material_opacity = states[material_index];
        if material_opacity <= GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance <= threshold {
            continue;
        }
        let escape = (projection.distance / threshold - 1.0).max(0.0);
        let adjoint = (material_tail_gain
            * escape.min(8.0)
            * (material_opacity - GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT))
            .clamp(0.0, max_adjoint);
        state_adjoint[material_index] += adjoint;
    }
}
