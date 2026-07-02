#![allow(clippy::too_many_arguments)]

use super::*;

const SEED_FRAME_RESIDUAL_MOTION_GAIN: f32 = 0.3;
const SEED_FRAME_RESIDUAL_DECAY: f32 = 0.025;
const SEED_FRAME_OPACITY_GROWTH_DELTA: f32 = 0.08;

#[allow(dead_code)]
pub(crate) fn local_growth_student_model(
    config: NpaConfig,
    seed: u64,
    density_gain: f32,
    expansion_gain: f32,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    local_growth_student_model_with_axis_gains(config, seed, density_gain, [expansion_gain; 3])
}

pub(crate) fn mesh_axis_expansion_gains(target: &TriangleMeshTarget, base_gain: f32) -> [f32; 3] {
    let (bounds_min, bounds_max) = target.bounds();
    let extents = [
        (bounds_max[0] - bounds_min[0]).max(1.0e-4),
        (bounds_max[1] - bounds_min[1]).max(1.0e-4),
        (bounds_max[2] - bounds_min[2]).max(1.0e-4),
    ];
    let mean_extent = ((extents[0] + extents[1] + extents[2]) / 3.0).max(1.0e-4);
    [
        base_gain * (extents[0] / mean_extent).clamp(0.35, 2.25),
        base_gain * (extents[1] / mean_extent).clamp(0.35, 2.25),
        base_gain * (extents[2] / mean_extent).clamp(0.35, 2.25),
    ]
}

pub(crate) fn local_growth_student_model_with_axis_gains(
    config: NpaConfig,
    seed: u64,
    density_gain: f32,
    expansion_gains: [f32; 3],
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims <= 3 || config.hidden_dims < 16 {
        return Err(std::io::Error::other(format!(
            "local 3D growth student requires 3D config, state_dims > 3, and hidden_dims >= 16; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if config.position_features {
        return Err(std::io::Error::other(
            "local 3D growth student must not use absolute position_features",
        )
        .into());
    }

    let mut weights = NpaWeights::seeded(&config, seed);
    for value in weights.w1.iter_mut().chain(weights.w2.iter_mut()) {
        *value *= 0.02;
    }
    for value in &mut weights.b1 {
        *value = 1.0e-3;
    }
    weights.b2.fill(0.0);
    let input_dims = config.perception_dims();
    let opacity_gradient_offset = config.state_dims * 2 + 3 * config.spatial_dims;
    if expansion_gains
        .iter()
        .any(|gain| !gain.is_finite() || *gain < 0.0)
    {
        return Err(
            std::io::Error::other("expansion_gains must be finite and non-negative").into(),
        );
    }
    for (axis, &expansion_gain) in expansion_gains.iter().enumerate().take(config.spatial_dims) {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.b1[pos_hidden] = 0.0;
        weights.b1[neg_hidden] = 0.0;
        weights.w1[pos_hidden * input_dims + opacity_gradient_offset + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + opacity_gradient_offset + axis] = -1.0;
        weights.w2[axis * config.hidden_dims + pos_hidden] = expansion_gain;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -expansion_gain;
    }
    let opacity_front_hidden = config.spatial_dims * 2;
    weights.b1[opacity_front_hidden] = 0.0;
    weights.w1[opacity_front_hidden * input_dims + config.state_dims + 3] = 1.0;
    weights.w1[opacity_front_hidden * input_dims + 3] = -1.0;
    let opacity_out = config.spatial_dims + 3;
    weights.w2[opacity_out * config.hidden_dims + opacity_front_hidden] = LOCAL_GROWTH_OPACITY_GAIN;
    if let Some(phase_channel) = growth_3d_phase_channel(config.state_dims) {
        let phase_hidden = opacity_front_hidden + 1;
        weights.b1[phase_hidden] = 0.0;
        for value in &mut weights.w1[phase_hidden * input_dims..(phase_hidden + 1) * input_dims] {
            *value = 0.0;
        }
        for output in 0..config.update_dims() {
            weights.w2[output * config.hidden_dims + phase_hidden] = 0.0;
        }
        weights.w1[phase_hidden * input_dims + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
            1.0;
        weights.w1[phase_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL] = -1.0;
        weights.w1[phase_hidden * input_dims + phase_channel] = -0.25;
        let phase_out = config.spatial_dims + phase_channel;
        weights.w2[phase_out * config.hidden_dims + phase_hidden] = LOCAL_GROWTH_PHASE_GAIN;
    }
    if let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims)
        && material_channel != 3
    {
        let material_low_hidden = 14;
        let material_high_hidden = 15;
        for hidden in [material_low_hidden, material_high_hidden] {
            weights.b1[hidden] = 0.0;
            for value in &mut weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
                *value = 0.0;
            }
            for output in 0..config.update_dims() {
                weights.w2[output * config.hidden_dims + hidden] = 0.0;
            }
        }
        weights.b1[material_low_hidden] = 1.0;
        weights.w1[material_low_hidden * input_dims + 3] = 1.0;
        weights.w1[material_high_hidden * input_dims + 3] = 1.0;
        let material_out = config.spatial_dims + material_channel;
        let material_base = material_out * config.hidden_dims;
        weights.w2[material_base + material_low_hidden] = LOCAL_GROWTH_MATERIAL_OPACITY_GAIN;
        weights.w2[material_base + material_high_hidden] = -LOCAL_GROWTH_MATERIAL_OPACITY_GAIN;
        if let Some(phase_channel) = growth_3d_phase_channel(config.state_dims)
            && config.hidden_dims > 16
        {
            let phase_material_hidden = 16;
            weights.b1[phase_material_hidden] = -0.15;
            for value in &mut weights.w1
                [phase_material_hidden * input_dims..(phase_material_hidden + 1) * input_dims]
            {
                *value = 0.0;
            }
            for output in 0..config.update_dims() {
                weights.w2[output * config.hidden_dims + phase_material_hidden] = 0.0;
            }
            weights.w1[phase_material_hidden * input_dims + phase_channel] = 1.0;
            weights.w2[material_base + phase_material_hidden] = LOCAL_GROWTH_PHASE_MATERIAL_GAIN;
        }
    }
    if config.hidden_dims > 18 {
        let active_liveness_low_hidden = 17;
        let active_liveness_high_hidden = 18;
        for hidden in [active_liveness_low_hidden, active_liveness_high_hidden] {
            weights.b1[hidden] = 0.0;
            for value in &mut weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
                *value = 0.0;
            }
            for output in 0..config.update_dims() {
                weights.w2[output * config.hidden_dims + hidden] = 0.0;
            }
        }
        weights.b1[active_liveness_low_hidden] = 1.0;
        weights.w1[active_liveness_low_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL] = 1.0;
        weights.w1[active_liveness_high_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL] = 1.0;
        let liveness_out = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
        let liveness_base = liveness_out * config.hidden_dims;
        weights.w2[liveness_base + active_liveness_low_hidden] = LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN;
        weights.w2[liveness_base + active_liveness_high_hidden] =
            -2.0 * LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN;
    }
    if config.hidden_dims > 24
        && let Some(velocity_channels) = growth_3d_velocity_channels(config.state_dims)
    {
        for (axis, velocity_channel) in velocity_channels.enumerate() {
            let pos_hidden = 19 + axis * 2;
            let neg_hidden = pos_hidden + 1;
            for hidden in [pos_hidden, neg_hidden] {
                weights.b1[hidden] = 0.0;
                for value in &mut weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
                    *value = 0.0;
                }
                for output in 0..config.update_dims() {
                    weights.w2[output * config.hidden_dims + hidden] = 0.0;
                }
            }
            weights.w1[pos_hidden * input_dims + velocity_channel] = 1.0;
            weights.w1[neg_hidden * input_dims + velocity_channel] = -1.0;
            weights.w2[axis * config.hidden_dims + pos_hidden] = LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN;
            weights.w2[axis * config.hidden_dims + neg_hidden] = -LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN;

            let velocity_out = config.spatial_dims + velocity_channel;
            let velocity_base = velocity_out * config.hidden_dims;
            weights.w2[velocity_base + pos_hidden] = -LOCAL_GROWTH_VELOCITY_DAMPING_GAIN;
            weights.w2[velocity_base + neg_hidden] = LOCAL_GROWTH_VELOCITY_DAMPING_GAIN;
        }
    }
    if config.hidden_dims > 25
        && let Some(phase_channel) = growth_3d_phase_channel(config.state_dims)
    {
        let phase_liveness_hidden = 25;
        weights.b1[phase_liveness_hidden] = -4.0;
        for value in &mut weights.w1
            [phase_liveness_hidden * input_dims..(phase_liveness_hidden + 1) * input_dims]
        {
            *value = 0.0;
        }
        for output in 0..config.update_dims() {
            weights.w2[output * config.hidden_dims + phase_liveness_hidden] = 0.0;
        }
        weights.w1
            [phase_liveness_hidden * input_dims + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
            1.0;
        weights.w1[phase_liveness_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL] = -1.0;
        weights.w1[phase_liveness_hidden * input_dims + phase_channel] = 1.0;
        let liveness_out = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
        weights.w2[liveness_out * config.hidden_dims + phase_liveness_hidden] =
            LOCAL_GROWTH_PHASE_LIVENESS_GAIN;
    }
    if config.hidden_dims > 26
        && let Some(material_channel) = growth_3d_material_opacity_channel(config.state_dims)
        && material_channel != GROWTH_3D_LIVENESS_CHANNEL
    {
        let active_material_hidden = 26;
        weights.b1[active_material_hidden] = 1.0;
        for value in &mut weights.w1
            [active_material_hidden * input_dims..(active_material_hidden + 1) * input_dims]
        {
            *value = 0.0;
        }
        for output in 0..config.update_dims() {
            weights.w2[output * config.hidden_dims + active_material_hidden] = 0.0;
        }
        weights.w1[active_material_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL] = 1.0;
        weights.w1[active_material_hidden * input_dims + material_channel] = -0.25;
        let material_out = config.spatial_dims + material_channel;
        weights.w2[material_out * config.hidden_dims + active_material_hidden] =
            LOCAL_GROWTH_ACTIVE_MATERIAL_GAIN;
    }
    if density_gain != 0.0 {
        let density_gradient_offset = config.state_dims * 2
            + usize::from(config.state_grad) * config.state_dims * config.spatial_dims;
        for axis in 0..config.spatial_dims {
            let pos_hidden = 8 + axis * 2;
            let neg_hidden = pos_hidden + 1;
            weights.b1[pos_hidden] = 0.0;
            weights.b1[neg_hidden] = 0.0;
            weights.w1[pos_hidden * input_dims + density_gradient_offset + axis] = 1.0;
            weights.w1[neg_hidden * input_dims + density_gradient_offset + axis] = -1.0;
            weights.w2[axis * config.hidden_dims + pos_hidden] = density_gain;
            weights.w2[axis * config.hidden_dims + neg_hidden] = -density_gain;
        }
    }

    Ok(NpaModel { config, weights })
}

pub(crate) fn retime_growth_3d_front_model(
    model: &mut NpaModel,
    hidden: Option<usize>,
    front_gain: f32,
) -> Result<usize, Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "retime-growth3d requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "retime-growth3d only supports local conditionless 3D models",
        )
        .into());
    }
    if !front_gain.is_finite() || front_gain <= 0.0 {
        return Err(std::io::Error::other("front_gain must be positive and finite").into());
    }
    let hidden = hidden.unwrap_or(model.config.hidden_dims.saturating_sub(1));
    if hidden >= model.config.hidden_dims {
        return Err(std::io::Error::other(format!(
            "hidden index {hidden} out of range for hidden_dims={}",
            model.config.hidden_dims
        ))
        .into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    for value in &mut model.weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
        *value = 0.0;
    }
    model.weights.b1[hidden] = 0.0;
    model.weights.w1[hidden * input_dims + model.config.state_dims + 3] = 1.0;
    model.weights.w1[hidden * input_dims + 3] = -1.0;

    for output in 0..output_dims {
        model.weights.w2[output * model.config.hidden_dims + hidden] = 0.0;
    }
    let opacity_out = model.config.spatial_dims + 3;
    let opacity_base = opacity_out * model.config.hidden_dims;
    for value in &mut model.weights.w2[opacity_base..opacity_base + model.config.hidden_dims] {
        *value = 0.0;
    }
    model.weights.b2[opacity_out] = 0.0;
    model.weights.w2[opacity_base + hidden] = front_gain;

    Ok(hidden)
}

pub(crate) fn retime_growth_3d_active_opacity_model(
    model: &mut NpaModel,
    hidden: Option<usize>,
    active_opacity_gain: f32,
) -> Result<usize, Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "active-opacity retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "active-opacity retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !active_opacity_gain.is_finite() || active_opacity_gain <= 0.0 {
        return Err(
            std::io::Error::other("active_opacity_gain must be positive and finite").into(),
        );
    }
    let low_hidden = hidden.unwrap_or(model.config.hidden_dims.saturating_sub(3));
    let high_hidden = low_hidden + 1;
    if high_hidden >= model.config.hidden_dims {
        return Err(std::io::Error::other(format!(
            "active opacity hidden pair {low_hidden},{high_hidden} out of range for hidden_dims={}",
            model.config.hidden_dims
        ))
        .into());
    }

    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    for hidden in [low_hidden, high_hidden] {
        for value in &mut model.weights.w1[hidden * input_dims..(hidden + 1) * input_dims] {
            *value = 0.0;
        }
        for output in 0..output_dims {
            model.weights.w2[output * model.config.hidden_dims + hidden] = 0.0;
        }
    }

    model.weights.b1[low_hidden] = 1.0;
    model.weights.w1[low_hidden * input_dims + 3] = 1.0;
    model.weights.b1[high_hidden] = 0.0;
    model.weights.w1[high_hidden * input_dims + 3] = 1.0;

    let opacity_out = model.config.spatial_dims + 3;
    let opacity_base = opacity_out * model.config.hidden_dims;
    model.weights.w2[opacity_base + low_hidden] = active_opacity_gain;
    model.weights.w2[opacity_base + high_hidden] = -active_opacity_gain;

    Ok(low_hidden)
}

pub(crate) fn add_growth_3d_opacity_update_bias(
    model: &mut NpaModel,
    opacity_bias: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "opacity bias retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "opacity bias retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !opacity_bias.is_finite() {
        return Err(std::io::Error::other("opacity_bias must be finite").into());
    }
    let opacity_out = model.config.spatial_dims + 3;
    if opacity_out >= model.config.update_dims() || opacity_out >= model.weights.b2.len() {
        return Err(std::io::Error::other("opacity output index out of range").into());
    }
    model.weights.b2[opacity_out] += opacity_bias;
    Ok(())
}

pub(crate) fn add_growth_3d_material_opacity_update_bias(
    model: &mut NpaModel,
    opacity_bias: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if model.config.spatial_dims != 3 || model.config.state_dims <= 3 {
        return Err(std::io::Error::other(
            "material opacity bias retime requires a 3D model with opacity state",
        )
        .into());
    }
    if model.config.position_features {
        return Err(std::io::Error::other(
            "material opacity bias retime only supports local conditionless 3D models",
        )
        .into());
    }
    if !opacity_bias.is_finite() {
        return Err(std::io::Error::other("material opacity_bias must be finite").into());
    }
    let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) else {
        return Err(std::io::Error::other("material opacity channel is unavailable").into());
    };
    let opacity_out = model.config.spatial_dims + material_channel;
    if opacity_out >= model.config.update_dims() || opacity_out >= model.weights.b2.len() {
        return Err(std::io::Error::other("material opacity output index out of range").into());
    }
    model.weights.b2[opacity_out] += opacity_bias;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn torus_growth_model(
    config: NpaConfig,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    residual_state_growth_model(
        config,
        "uv torus growth",
        UV_TORUS_MOTION_GAIN,
        UV_TORUS_RESIDUAL_DECAY,
        UV_TORUS_OPACITY_GROWTH_DELTA,
    )
}

fn residual_state_growth_model(
    config: NpaConfig,
    family: &str,
    motion_gain: f32,
    residual_decay: f32,
    opacity_growth_delta: f32,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims <= 3 || config.hidden_dims < 6 {
        return Err(std::io::Error::other(format!(
            "{family} requires 3D config, state_dims > 3, and hidden_dims >= 6; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    for axis in 0..3 {
        let pos_hidden = axis * 2;
        let neg_hidden = pos_hidden + 1;
        weights.w1[pos_hidden * input_dims + axis] = 1.0;
        weights.w1[neg_hidden * input_dims + axis] = -1.0;

        weights.w2[axis * config.hidden_dims + pos_hidden] = motion_gain;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -motion_gain;

        let residual_out = config.spatial_dims + axis;
        weights.w2[residual_out * config.hidden_dims + pos_hidden] = -residual_decay;
        weights.w2[residual_out * config.hidden_dims + neg_hidden] = residual_decay;
    }
    weights.b2[config.spatial_dims + 3] = opacity_growth_delta;

    Ok(NpaModel { config, weights })
}

pub(crate) fn torus_morphogen_model(
    config: NpaConfig,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    seed_frame_morphogen_model(config)
}

pub(crate) fn seed_frame_morphogen_model(
    config: NpaConfig,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.position_features {
        return Err(std::io::Error::other(
            "seed-frame morphogen model must not use absolute position_features",
        )
        .into());
    }
    residual_state_growth_model(
        config,
        "seed-frame residual growth",
        SEED_FRAME_RESIDUAL_MOTION_GAIN,
        SEED_FRAME_RESIDUAL_DECAY,
        SEED_FRAME_OPACITY_GROWTH_DELTA,
    )
}
