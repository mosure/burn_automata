use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_strict_surface_materialization_output_objective(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    raw_updates: &[f32],
    opacity_gain: f32,
    seed_scale: f32,
    max_opacity_update: f32,
    output_gradients: &mut [f32],
) {
    let output_dims = config.update_dims();
    let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims) else {
        return;
    };
    let material_output = config.spatial_dims + material_channel;
    if material_output >= output_dims
        || config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
        || positions.is_empty()
        || states.len() < positions.len() * config.state_dims
        || raw_updates.len() < positions.len() * output_dims
        || output_gradients.len() < positions.len() * output_dims
        || opacity_gain <= 0.0
        || !opacity_gain.is_finite()
    {
        return;
    }

    let strict_threshold = target_coverage_threshold(seed_scale).max(1.0e-6);
    let max_update = if max_opacity_update.is_finite() && max_opacity_update > 0.0 {
        max_opacity_update
    } else {
        f32::INFINITY
    };
    let visible_gate = -1.0_f32;
    for (row, position) in positions.iter().enumerate() {
        let state_base = row * config.state_dims;
        if states[state_base + GROWTH_3D_LIVENESS_CHANNEL] <= -1.0 {
            continue;
        }
        let projection = target.project(position3(*position));
        if !projection.distance.is_finite() || projection.distance > strict_threshold {
            continue;
        }
        let material_opacity = states[state_base + material_channel];
        if material_opacity >= GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET {
            continue;
        }
        let target_gap = (GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET - material_opacity).max(0.0);
        if target_gap <= 0.0 {
            continue;
        }
        let gate_weight = if material_opacity < visible_gate {
            1.0
        } else {
            0.5
        };
        let surface_weight = (1.0 - projection.distance / strict_threshold).clamp(0.25, 1.0);
        let target_update =
            (opacity_gain * gate_weight * surface_weight * target_gap).clamp(0.0, max_update);
        if target_update <= 0.0 {
            continue;
        }
        let output_index = row * output_dims + material_output;
        let raw = raw_updates[output_index];
        output_gradients[output_index] += raw - target_update;
    }
}
