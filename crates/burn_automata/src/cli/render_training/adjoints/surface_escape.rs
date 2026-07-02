#![allow(clippy::too_many_arguments)]

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_surface_escape_state_adjoint(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    surface_escape_gain: f32,
    opacity_gain: f32,
    liveness_gain: f32,
    max_adjoint: f32,
    state_adjoint: &mut [f32],
) {
    if config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || state_adjoint.len() < states.len()
        || surface_escape_gain <= 0.0
        || !surface_escape_gain.is_finite()
        || ((opacity_gain <= 0.0 || !opacity_gain.is_finite())
            && (liveness_gain <= 0.0 || !liveness_gain.is_finite()))
    {
        return;
    }
    let max_adjoint = if max_adjoint.is_finite() && max_adjoint > 0.0 {
        max_adjoint
    } else {
        f32::INFINITY
    };
    let threshold = GROWTH_3D_SURFACE_MAX_DISTANCE.max(1.0e-6);
    let opacity_channel = growth_3d_material_opacity_channel(config.state_dims);

    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        let liveness_index = state_base + GROWTH_3D_LIVENESS_CHANNEL;
        let liveness = states[liveness_index];
        if liveness <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance <= threshold {
            continue;
        }
        let escape = (projection.distance / threshold - 1.0).max(0.0);
        let strength = (surface_escape_gain * escape).min(8.0);
        if liveness_gain > 0.0 && liveness_gain.is_finite() {
            let adjoint = (liveness_gain * strength).clamp(0.0, max_adjoint);
            state_adjoint[liveness_index] = state_adjoint[liveness_index].max(adjoint);
        }
        if opacity_gain > 0.0
            && opacity_gain.is_finite()
            && let Some(opacity_channel) = opacity_channel
        {
            let opacity_index = state_base + opacity_channel;
            if opacity_index != liveness_index {
                let adjoint = (opacity_gain * strength).clamp(0.0, max_adjoint);
                state_adjoint[opacity_index] = state_adjoint[opacity_index].max(adjoint);
            }
        }
    }
}
