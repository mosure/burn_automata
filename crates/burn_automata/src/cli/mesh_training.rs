#![allow(clippy::too_many_arguments)]

use super::prelude::*;

pub(crate) const TORUS_ROBUSTNESS_CASES: &[TorusRobustnessCaseConfig] = &[
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 3,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 17,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 97,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 8192,
        steps: 180,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusFieldDense3d,
    },
];

pub(crate) const TORUS_MORPHOGEN_ROBUSTNESS_CASES: &[TorusRobustnessCaseConfig] = &[
    TorusRobustnessCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 5,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
    TorusRobustnessCaseConfig {
        particle_count: 8192,
        steps: 200,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TorusMorphogenDense3d,
    },
];

pub(crate) const TEAPOT_FIELD_ROLLOUT_CASES: &[MeshRolloutCaseConfig] = &[
    MeshRolloutCaseConfig {
        particle_count: 512,
        steps: 180,
        seed: 13,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 8192,
        steps: 180,
        seed: 131,
        seed_scale: UV_TORUS_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
];

pub(crate) fn torus_field_model(config: NpaConfig) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "torus field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("torus field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    let position_pairs = [
        add_identity_pair(&mut weights, position_offset, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 1, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 2, &mut hidden),
    ];
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    let tail_pairs = [
        add_identity_pair(&mut weights, tail, &mut hidden),
        add_identity_pair(&mut weights, tail + 1, &mut hidden),
        add_identity_pair(&mut weights, tail + 2, &mut hidden),
    ];

    let major = UV_TORUS_FIELD_SCALE;
    let minor = major * UV_TORUS_MINOR_RATIO;
    let outer = major + minor;
    let color_position_coeffs = [
        1.0 / (2.0 * outer),
        1.0 / (2.0 * outer),
        1.0 / (2.0 * minor.max(1.0e-4)),
    ];
    for channel in 0..3 {
        let out = config.spatial_dims + tail + channel;
        let (pos_hidden, neg_hidden) = tail_pairs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] -= UV_TORUS_FIELD_COLOR_GAIN;
        weights.w2[out * config.hidden_dims + neg_hidden] += UV_TORUS_FIELD_COLOR_GAIN;

        let axis = channel;
        let (pos_hidden, neg_hidden) = position_pairs[axis];
        let coeff = UV_TORUS_FIELD_COLOR_GAIN * color_position_coeffs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] += coeff;
        weights.w2[out * config.hidden_dims + neg_hidden] -= coeff;
    }

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    Ok(NpaModel { config, weights })
}

#[allow(dead_code)]
pub(crate) fn mesh_field_model(
    config: NpaConfig,
    _seed: u64,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "mesh field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("mesh field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    for axis in 0..3 {
        add_identity_pair(&mut weights, position_offset + axis, &mut hidden);
    }
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    for channel in 0..3 {
        add_identity_pair(&mut weights, tail + channel, &mut hidden);
    }

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    Ok(NpaModel { config, weights })
}

pub(crate) fn teapot_field_model(
    config: NpaConfig,
) -> Result<NpaModel, Box<dyn std::error::Error>> {
    if config.spatial_dims != 3 || config.state_dims < 6 || config.hidden_dims < 20 {
        return Err(std::io::Error::other(format!(
            "teapot field requires 3D config, state_dims >= 6, and hidden_dims >= 20; got spatial_dims={}, state_dims={}, hidden_dims={}",
            config.spatial_dims, config.state_dims, config.hidden_dims
        ))
        .into());
    }
    if !config.position_features {
        return Err(std::io::Error::other("teapot field requires position_features=true").into());
    }

    let mut weights = NpaWeights::zeros(&config);
    let input_dims = config.perception_dims();
    let position_offset = input_dims - config.spatial_dims;
    let mut hidden = 0usize;

    let add_identity_pair =
        |weights: &mut NpaWeights, input: usize, hidden: &mut usize| -> (usize, usize) {
            let pos = *hidden;
            let neg = *hidden + 1;
            weights.w1[pos * input_dims + input] = 1.0;
            weights.w1[neg * input_dims + input] = -1.0;
            *hidden += 2;
            (pos, neg)
        };

    let position_pairs = [
        add_identity_pair(&mut weights, position_offset, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 1, &mut hidden),
        add_identity_pair(&mut weights, position_offset + 2, &mut hidden),
    ];
    let opacity_pair = add_identity_pair(&mut weights, 3, &mut hidden);
    let tail = config.state_dims - 3;
    let tail_pairs = [
        add_identity_pair(&mut weights, tail, &mut hidden),
        add_identity_pair(&mut weights, tail + 1, &mut hidden),
        add_identity_pair(&mut weights, tail + 2, &mut hidden),
    ];

    let opacity_out = config.spatial_dims + 3;
    weights.b2[opacity_out] = UV_TORUS_FIELD_OPACITY_GAIN * UV_TORUS_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= UV_TORUS_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += UV_TORUS_FIELD_OPACITY_GAIN;

    let (bounds_min, bounds_max) = utah_teapot_mesh_target(UV_TORUS_FIELD_SCALE).bounds();
    for channel in 0..3 {
        let out = config.spatial_dims + tail + channel;
        let (tail_pos, tail_neg) = tail_pairs[channel];
        weights.w2[out * config.hidden_dims + tail_pos] -= TEAPOT_FIELD_COLOR_GAIN;
        weights.w2[out * config.hidden_dims + tail_neg] += TEAPOT_FIELD_COLOR_GAIN;

        let range = (bounds_max[channel] - bounds_min[channel]).max(1.0e-4);
        let coeff = TEAPOT_FIELD_COLOR_GAIN / range;
        let (pos_hidden, neg_hidden) = position_pairs[channel];
        weights.w2[out * config.hidden_dims + pos_hidden] += coeff;
        weights.w2[out * config.hidden_dims + neg_hidden] -= coeff;
        weights.b2[out] += TEAPOT_FIELD_COLOR_GAIN * (-bounds_min[channel] / range - 0.5);
    }

    Ok(NpaModel { config, weights })
}

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
    if config.spatial_dims != 3 || config.state_dims <= 3 || config.hidden_dims < 6 {
        return Err(std::io::Error::other(format!(
            "uv torus growth requires 3D config, state_dims > 3, and hidden_dims >= 6; got spatial_dims={}, state_dims={}, hidden_dims={}",
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

        weights.w2[axis * config.hidden_dims + pos_hidden] = UV_TORUS_MOTION_GAIN;
        weights.w2[axis * config.hidden_dims + neg_hidden] = -UV_TORUS_MOTION_GAIN;

        let residual_out = config.spatial_dims + axis;
        weights.w2[residual_out * config.hidden_dims + pos_hidden] = -UV_TORUS_RESIDUAL_DECAY;
        weights.w2[residual_out * config.hidden_dims + neg_hidden] = UV_TORUS_RESIDUAL_DECAY;
    }
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;

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
    torus_growth_model(config)
}

#[allow(dead_code)]
pub(crate) fn torus_growth_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let scales = [0.56_f32, 0.72, 0.88];
    let mut rng = StdRng::seed_from_u64(0x703d_5eed);

    for row in 0..rows {
        let scale = scales[row % scales.len()];
        let sample = uv_torus_sample(row, rows, scale);
        let structured_position = [
            sample.position[0] * UV_TORUS_INITIAL_SCALE,
            sample.position[1] * UV_TORUS_INITIAL_SCALE,
            sample.position[2] * UV_TORUS_INITIAL_SCALE,
        ];
        let dense_position = uv_torus_dense_seed_position(&mut rng, scale);
        let initial_position = if row % 2 == 0 {
            structured_position
        } else {
            dense_position
        };
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        for axis in 0..3 {
            let residual = sample.position[axis] - initial_position[axis];
            features[feature_base + axis] = residual;
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual;
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual;
        }
        features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
        target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(sample.position, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn torus_morphogen_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_6d0f);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        uv_torus_mesh_target(scales[0]),
        uv_torus_mesh_target(scales[1]),
        uv_torus_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = uv_torus_tail_state_color(target, scale);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn teapot_morphogen_supervised_batch(
    config: &NpaConfig,
    rows: usize,
) -> SupervisedBatch {
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_07d0);
    let scales = [0.56_f32, UV_TORUS_FIELD_SCALE, 0.88];
    let targets = [
        utah_teapot_mesh_target(scales[0]),
        utah_teapot_mesh_target(scales[1]),
        utah_teapot_mesh_target(scales[2]),
    ];

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = utah_teapot_training_position(row, scale, &targets[scale_idx], &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;
        let feature_base = row * input_dims;
        let update_base = row * output_dims;

        for axis in 0..3 {
            features[feature_base + axis] = residual[axis];
            target_update[update_base + axis] = UV_TORUS_MOTION_GAIN * residual[axis];
            target_update[update_base + config.spatial_dims + axis] =
                -UV_TORUS_RESIDUAL_DECAY * residual[axis];
        }
        if config.state_dims > 3 {
            features[feature_base + 3] = UV_TORUS_INITIAL_OPACITY_LOGIT;
            target_update[update_base + config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET] = projection.normal[0];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 1] = projection.normal[1];
            features[feature_base + UV_TORUS_NORMAL_STATE_OFFSET + 2] = projection.normal[2];
            features[feature_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                projection.signed_distance;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let tail_color = utah_teapot_tail_state_color(target, &targets[scale_idx]);
            features[feature_base + tail] = tail_color[0];
            features[feature_base + tail + 1] = tail_color[1];
            features[feature_base + tail + 2] = tail_color[2];
        }
        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn torus_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "torus field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x703d_f13d);
    let scales = [UV_TORUS_FIELD_SCALE];
    let targets = [uv_torus_mesh_target(scales[0])];
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let scale_idx = row % scales.len();
        let scale = scales[scale_idx];
        let position = torus_implicit_training_position(row, scale, &mut rng);
        let projection = targets[scale_idx].project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = UV_TORUS_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = uv_torus_tail_state_color(target, scale);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    UV_TORUS_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

pub(crate) fn teapot_field_supervised_batch(config: &NpaConfig, rows: usize) -> SupervisedBatch {
    assert!(
        config.position_features,
        "teapot field training requires position features"
    );
    let rows = rows.max(1);
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let mut features = vec![0.0; rows * input_dims];
    let mut target_update = vec![0.0; rows * output_dims];
    let mut rng = StdRng::seed_from_u64(0x7ea9_f13d);
    let scale = UV_TORUS_FIELD_SCALE;
    let target_mesh = utah_teapot_mesh_target(scale);
    let position_offset = input_dims - config.spatial_dims;

    for row in 0..rows {
        let position = utah_teapot_training_position(row, scale, &target_mesh, &mut rng);
        let projection = target_mesh.project(position);
        let target = projection.closest;
        let residual = projection.residual;

        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        let mut current_tail = [0.0_f32; 3];
        if config.state_dims > 3 {
            features[feature_base + 3] =
                rng.random_range(UV_TORUS_INITIAL_OPACITY_LOGIT..UV_TORUS_FIELD_OPACITY_TARGET);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            for channel in 0..3 {
                current_tail[channel] = rng.random_range(-0.35..0.35);
                features[feature_base + tail + channel] = current_tail[channel];
            }
        }

        let blur_offset = config.state_dims;
        for channel in 0..config.state_dims {
            features[feature_base + blur_offset + channel] = features[feature_base + channel];
        }
        for axis in 0..3 {
            features[feature_base + position_offset + axis] = position[axis];
            target_update[update_base + axis] = TEAPOT_FIELD_MOTION_GAIN * residual[axis];
        }

        if config.state_dims > 3 {
            let current_opacity = features[feature_base + 3];
            target_update[update_base + config.spatial_dims + 3] =
                UV_TORUS_FIELD_OPACITY_GAIN * (UV_TORUS_FIELD_OPACITY_TARGET - current_opacity);
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = utah_teapot_tail_state_color(target, &target_mesh);
            for channel in 0..3 {
                target_update[update_base + config.spatial_dims + tail + channel] =
                    TEAPOT_FIELD_COLOR_GAIN * (target_tail[channel] - current_tail[channel]);
            }
        }
    }

    SupervisedBatch {
        features,
        target_update,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MeshFieldRolloutBatchConfig {
    pub(crate) max_rows: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: f32,
    pub(crate) max_update_norm: f32,
    pub(crate) coverage_gain: f32,
    pub(crate) coverage_samples: usize,
    pub(crate) coverage_mode: CoverageUpdateModeArg,
    pub(crate) coverage_softness: f32,
    pub(crate) coverage_repulsion_gain: f32,
    pub(crate) coverage_gap_gain: f32,
    pub(crate) coverage_repulsion_radius: f32,
    pub(crate) coverage_normal_weight: f32,
    pub(crate) extent_gain: f32,
    pub(crate) color_gain: f32,
    pub(crate) aux_state_gain: f32,
    pub(crate) opacity_gain: f32,
    pub(crate) front_opacity_gain: f32,
    pub(crate) front_radius: f32,
    pub(crate) front_max_opacity_update: f32,
    pub(crate) front_motion_gate: bool,
    pub(crate) preserve_opacity_update: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct MeshLocalTrainingConfig {
    pub(crate) max_rows: usize,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) rollouts: usize,
    pub(crate) temporal_samples: usize,
    pub(crate) training_rounds: usize,
    pub(crate) total_steps: usize,
    pub(crate) seed: u64,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) motion_gain: f32,
    pub(crate) max_update_norm: f32,
    pub(crate) coverage_gain: f32,
    pub(crate) coverage_samples: usize,
    pub(crate) coverage_mode: CoverageUpdateModeArg,
    pub(crate) coverage_softness: f32,
    pub(crate) coverage_repulsion_gain: f32,
    pub(crate) coverage_gap_gain: f32,
    pub(crate) coverage_repulsion_radius: f32,
    pub(crate) coverage_normal_weight: f32,
    pub(crate) extent_gain: f32,
    pub(crate) color_gain: f32,
    pub(crate) aux_state_gain: f32,
    pub(crate) opacity_gain: f32,
    pub(crate) front_opacity_gain: f32,
    pub(crate) front_radius: f32,
    pub(crate) front_max_opacity_update: f32,
    pub(crate) front_motion_gate: bool,
    pub(crate) preserve_opacity_update: bool,
    pub(crate) sgd: SgdConfig,
}

pub(crate) fn merge_supervised_batches(
    mut lhs: SupervisedBatch,
    rhs: SupervisedBatch,
) -> SupervisedBatch {
    lhs.features.extend(rhs.features);
    lhs.target_update.extend(rhs.target_update);
    lhs
}

pub(crate) fn run_refreshed_mesh_local_training(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshLocalTrainingConfig,
) -> Result<TrainingRunReport, Box<dyn std::error::Error>> {
    if cfg.total_steps == 0 {
        return Err(std::io::Error::other("local mesh training requires at least one step").into());
    }
    let rounds = cfg.training_rounds.max(1);
    let mut history = Vec::new();
    let mut initial_loss = None;
    let mut final_loss = 0.0_f32;
    let mut best_loss = f32::MAX;
    let mut rows = cfg.max_rows;
    let mut steps_done = 0usize;

    for round in 0..rounds {
        if steps_done >= cfg.total_steps {
            break;
        }
        let remaining_steps = cfg.total_steps - steps_done;
        let rounds_left = rounds - round;
        let round_steps = remaining_steps.div_ceil(rounds_left).max(1);
        let batch = mesh_local_rollout_supervised_batch(
            model,
            grid,
            target,
            MeshFieldRolloutBatchConfig {
                max_rows: cfg.max_rows,
                particle_count: cfg.particle_count,
                rollout_steps: cfg.rollout_steps,
                rollouts: cfg.rollouts,
                temporal_samples: cfg.temporal_samples,
                seed: cfg
                    .seed
                    .wrapping_add((round as u64).wrapping_mul(0x51ed_f00d)),
                seed_scale: cfg.seed_scale,
                seed_mode: cfg.seed_mode,
                motion_gain: cfg.motion_gain,
                max_update_norm: cfg.max_update_norm,
                coverage_gain: cfg.coverage_gain,
                coverage_samples: cfg.coverage_samples,
                coverage_mode: cfg.coverage_mode,
                coverage_softness: cfg.coverage_softness,
                coverage_repulsion_gain: cfg.coverage_repulsion_gain,
                coverage_gap_gain: cfg.coverage_gap_gain,
                coverage_repulsion_radius: cfg.coverage_repulsion_radius,
                coverage_normal_weight: cfg.coverage_normal_weight,
                extent_gain: cfg.extent_gain,
                color_gain: cfg.color_gain,
                aux_state_gain: cfg.aux_state_gain,
                opacity_gain: cfg.opacity_gain,
                front_opacity_gain: cfg.front_opacity_gain,
                front_radius: cfg.front_radius,
                front_max_opacity_update: cfg.front_max_opacity_update,
                front_motion_gate: cfg.front_motion_gate,
                preserve_opacity_update: cfg.preserve_opacity_update,
            },
        )?;
        let report = run_supervised_training(
            model,
            &batch,
            TrainingRunConfig {
                steps: round_steps,
                report_interval: round_steps.max(1),
                sgd: cfg.sgd,
            },
        )?;
        initial_loss.get_or_insert(report.initial_loss);
        rows = report.rows;
        final_loss = report.final_loss;
        best_loss = best_loss.min(report.best_loss);
        for entry in report.history {
            history.push(TrainingHistoryEntry {
                step: steps_done + entry.step,
                loss: entry.loss,
                grad_norm: entry.grad_norm,
                grad_scale: entry.grad_scale,
            });
        }
        steps_done += round_steps;
    }

    Ok(TrainingRunReport {
        steps: steps_done,
        rows,
        initial_loss: initial_loss.unwrap_or(final_loss),
        final_loss,
        best_loss,
        history,
    })
}

pub(crate) fn mesh_field_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, true)
}

pub(crate) fn mesh_local_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    mesh_rollout_supervised_batch(model, grid, target, cfg, false)
}

pub(crate) fn mesh_rollout_supervised_batch(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: MeshFieldRolloutBatchConfig,
    require_position_features: bool,
) -> Result<SupervisedBatch, Box<dyn std::error::Error>> {
    if !model.config.position_features {
        if require_position_features {
            return Err(std::io::Error::other(
                "mesh field rollout rows require position_features=true",
            )
            .into());
        }
    } else if !require_position_features {
        return Err(std::io::Error::other(
            "mesh local rollout rows require position_features=false",
        )
        .into());
    }
    if cfg.max_rows == 0 || cfg.particle_count == 0 || cfg.rollouts == 0 {
        return Err(std::io::Error::other("mesh rollout rows require non-zero sizes").into());
    }

    let mut features = Vec::new();
    let mut target_update = Vec::new();
    let mut remaining_rows = cfg.max_rows;
    let snapshot_steps = mesh_rollout_snapshot_steps(cfg.rollout_steps, cfg.temporal_samples);
    let total_snapshots = cfg.rollouts.saturating_mul(snapshot_steps.len()).max(1);
    let distributed_row_limit = cfg.max_rows.div_ceil(total_snapshots).max(1);
    for rollout_idx in 0..cfg.rollouts {
        if remaining_rows == 0 {
            break;
        }
        let (mut positions, mut states) = seed_particles_scaled(
            1,
            cfg.particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed
                .wrapping_add((rollout_idx as u64).wrapping_mul(0x9e37_79b9)),
            cfg.seed_mode,
            cfg.seed_scale,
        );
        let mut current_step = 0usize;
        for &snapshot_step in &snapshot_steps {
            while current_step < snapshot_step {
                let step =
                    model.step_cpu(&positions, &states, 1, cfg.particle_count, grid, 1.0, None)?;
                positions = step.next_positions;
                states = step.next_states;
                current_step += 1;
            }
            let row_limit = if snapshot_steps.len() == 1 {
                remaining_rows
            } else {
                remaining_rows.min(distributed_row_limit)
            };
            let rows = append_mesh_rollout_snapshot_rows(
                model,
                grid,
                target,
                &cfg,
                &positions,
                &states,
                row_limit,
                &mut features,
                &mut target_update,
            )?;
            remaining_rows = remaining_rows.saturating_sub(rows);
            if remaining_rows == 0 {
                break;
            }
        }
    }

    if features.is_empty() {
        return Err(std::io::Error::other("mesh rollout rows produced no data").into());
    }
    Ok(SupervisedBatch {
        features,
        target_update,
    })
}

pub(crate) fn mesh_rollout_snapshot_steps(
    rollout_steps: usize,
    temporal_samples: usize,
) -> Vec<usize> {
    let samples = temporal_samples.max(1);
    if samples == 1 {
        return vec![rollout_steps];
    }
    if rollout_steps == 0 {
        return vec![0];
    }
    let mut steps = Vec::with_capacity(samples);
    for sample_idx in 0..samples {
        let step = sample_idx * rollout_steps / (samples - 1);
        if steps.last().copied() != Some(step) {
            steps.push(step);
        }
    }
    if steps.last().copied() != Some(rollout_steps) {
        steps.push(rollout_steps);
    }
    steps
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn append_mesh_rollout_snapshot_rows(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &MeshFieldRolloutBatchConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    max_rows: usize,
    features: &mut Vec<f32>,
    target_update: &mut Vec<f32>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let row_budget = cfg.particle_count.min(max_rows);
    if row_budget == 0 {
        return Ok(0);
    }
    let step = model.step_cpu(positions, states, 1, cfg.particle_count, grid, 1.0, None)?;
    let mut rollout_target_update = mesh_field_target_update_for_rows(
        &model.config,
        target,
        positions,
        states,
        cfg.motion_gain,
        cfg.max_update_norm,
        cfg.color_gain,
        cfg.aux_state_gain,
        cfg.opacity_gain,
        cfg.front_opacity_gain,
        cfg.front_radius,
        cfg.front_max_opacity_update,
        cfg.front_motion_gate,
    );
    add_target_coverage_updates_for_rows(
        &model.config,
        target,
        positions,
        &mut rollout_target_update,
        cfg.coverage_gain,
        cfg.coverage_samples,
        cfg.coverage_mode,
        cfg.coverage_softness,
        cfg.coverage_repulsion_gain,
        cfg.coverage_gap_gain,
        cfg.coverage_repulsion_radius,
        cfg.coverage_normal_weight,
        cfg.seed_scale,
        cfg.max_update_norm,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        cfg.front_radius,
    );
    add_target_extent_updates_for_rows(
        &model.config,
        target,
        positions,
        if cfg.front_motion_gate {
            Some(states)
        } else {
            None
        },
        &mut rollout_target_update,
        cfg.extent_gain,
        cfg.max_update_norm,
        cfg.front_radius,
    );
    if cfg.preserve_opacity_update && model.config.state_dims > 3 {
        let output_dims = model.config.update_dims();
        for row in 0..cfg.particle_count.min(positions.len()) {
            let update_base = row * output_dims + model.config.spatial_dims + 3;
            let state_base = row * model.config.state_dims + 3;
            if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                rollout_target_update[update_base] = step.ds[state_base];
            }
            if let Some(channel) = growth_3d_material_opacity_channel(model.config.state_dims)
                && channel != 3
            {
                let update_base = row * output_dims + model.config.spatial_dims + channel;
                let state_base = row * model.config.state_dims + channel;
                if update_base < rollout_target_update.len() && state_base < step.ds.len() {
                    rollout_target_update[update_base] = step.ds[state_base];
                }
            }
        }
    }
    let input_dims = model.config.perception_dims();
    let output_dims = model.config.update_dims();
    let row_indices = mesh_rollout_row_indices(
        &rollout_target_update,
        output_dims,
        cfg.particle_count,
        row_budget,
    );
    for row in row_indices {
        let feature_base = row * input_dims;
        let update_base = row * output_dims;
        features
            .extend_from_slice(&step.perception.features[feature_base..feature_base + input_dims]);
        target_update
            .extend_from_slice(&rollout_target_update[update_base..update_base + output_dims]);
    }
    Ok(row_budget)
}

pub(crate) fn mesh_rollout_row_indices(
    target_update: &[f32],
    output_dims: usize,
    particle_count: usize,
    row_budget: usize,
) -> Vec<usize> {
    let rows = particle_count.min(row_budget);
    if rows >= particle_count {
        return (0..particle_count).collect();
    }
    if rows == 0 || output_dims == 0 {
        return Vec::new();
    }

    let spread_budget = (rows / 4).max(1).min(rows);
    let mut selected = vec![false; particle_count];
    let mut row_indices = Vec::with_capacity(rows);
    for row in spread_row_indices(particle_count, spread_budget) {
        if row < particle_count && !selected[row] {
            selected[row] = true;
            row_indices.push(row);
        }
    }

    let mut scored_rows = (0..particle_count)
        .map(|row| {
            let base = row * output_dims;
            let score = target_update
                .get(base..base + output_dims)
                .unwrap_or(&[])
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value * value)
                .sum::<f32>();
            (row, score)
        })
        .collect::<Vec<_>>();
    scored_rows.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    for (row, score) in scored_rows {
        if row_indices.len() >= rows {
            break;
        }
        if score <= 0.0 || selected[row] {
            continue;
        }
        selected[row] = true;
        row_indices.push(row);
    }
    if row_indices.len() < rows {
        for row in spread_row_indices(particle_count, particle_count) {
            if row_indices.len() >= rows {
                break;
            }
            if !selected[row] {
                selected[row] = true;
                row_indices.push(row);
            }
        }
    }
    row_indices
}

pub(crate) fn mesh_field_target_update_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    states: &[f32],
    motion_gain: f32,
    max_update_norm: f32,
    color_gain: f32,
    aux_state_gain: f32,
    opacity_gain: f32,
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
    front_motion_gate: bool,
) -> Vec<f32> {
    let rows = positions.len();
    let output_dims = config.update_dims();
    let mut target_update = vec![0.0; rows * output_dims];
    let front_targets = local_front_opacity_targets(
        config,
        positions,
        states,
        front_opacity_gain,
        front_radius,
        front_max_opacity_update,
    );
    let front_weights = if front_motion_gate {
        Some(local_front_weights(config, positions, states, front_radius))
    } else {
        None
    };
    let target_radius = target
        .vertices
        .iter()
        .map(|vertex| {
            (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt()
        })
        .fold(1.0e-4_f32, f32::max);
    for (row, position) in positions.iter().enumerate() {
        let projection = target.project([position[0], position[1], position[2]]);
        let update_base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        for axis in 0..3 {
            target_update[update_base + axis] =
                front_weight * motion_gain * projection.residual[axis];
        }
        let update_norm = (target_update[update_base].powi(2)
            + target_update[update_base + 1].powi(2)
            + target_update[update_base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && update_norm > max_update_norm.max(1.0e-6) {
            let scale = max_update_norm / update_norm;
            for axis in 0..3 {
                target_update[update_base + axis] *= scale;
            }
        }

        let state_base = row * config.state_dims;
        if config.state_dims >= 3 {
            for axis in 0..3 {
                let target_coordinate = projection.closest[axis] / target_radius.max(1.0e-4);
                target_update[update_base + config.spatial_dims + axis] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_COORDINATE_GAIN
                    * (target_coordinate - states[state_base + axis]);
            }
        }
        if config.state_dims > 3 {
            target_update[update_base + config.spatial_dims + 3] = front_targets[row];
        }
        if let Some(opacity_channel) = growth_3d_material_opacity_channel(config.state_dims) {
            let current_opacity = states[state_base + opacity_channel];
            let surface_band = (target_radius * 0.10).max(0.04);
            let surface_weight = (1.0 - projection.distance / surface_band).clamp(0.0, 1.0);
            let target_opacity = GROWTH_3D_INACTIVE_OPACITY_LOGIT
                + surface_weight
                    * (UV_TORUS_FIELD_OPACITY_TARGET - GROWTH_3D_INACTIVE_OPACITY_LOGIT);
            let direct_opacity_update =
                front_weight * opacity_gain * (target_opacity - current_opacity);
            target_update[update_base + config.spatial_dims + opacity_channel] +=
                direct_opacity_update;
        }
        if config.state_dims >= 6 {
            let tail = config.state_dims - 3;
            let target_tail = [
                projection.color[0] - 0.5,
                projection.color[1] - 0.5,
                projection.color[2] - 0.5,
            ];
            for channel in 0..3 {
                let current_tail = states[state_base + tail + channel];
                target_update[update_base + config.spatial_dims + tail + channel] =
                    front_weight * color_gain * (target_tail[channel] - current_tail);
            }
        }
        if uv_torus_orientation_state_available(config.state_dims) {
            for axis in 0..3 {
                let channel = UV_TORUS_NORMAL_STATE_OFFSET + axis;
                let current = states[state_base + channel];
                target_update[update_base + config.spatial_dims + channel] = front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_ORIENTATION_GAIN
                    * (projection.normal[axis] - current);
            }
            let current_signed_distance =
                states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET];
            target_update
                [update_base + config.spatial_dims + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET] =
                front_weight
                    * aux_state_gain
                    * LOCAL_GROWTH_SIGNED_DISTANCE_GAIN
                    * (projection.signed_distance - current_signed_distance);
        }
    }
    target_update
}

pub(crate) fn local_front_opacity_targets(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_opacity_gain: f32,
    front_radius: f32,
    front_max_opacity_update: f32,
) -> Vec<f32> {
    let rows = positions.len();
    let mut updates = vec![0.0; rows];
    if config.state_dims <= 3
        || rows == 0
        || front_opacity_gain <= 0.0
        || front_radius <= 0.0
        || front_max_opacity_update <= 0.0
    {
        return updates;
    }

    let dormant_target = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let front_weights = local_front_weights(config, positions, states, front_radius);
    for row in 0..positions.len() {
        let state_base = row * config.state_dims;
        let current_opacity = states[state_base + 3];
        let mut target_opacity = if front_weights[row] >= 1.0 {
            UV_TORUS_FIELD_OPACITY_TARGET
        } else {
            dormant_target
        };

        if front_weights[row] > 0.0 && front_weights[row] < 1.0 {
            target_opacity = dormant_target
                + front_weights[row] * (UV_TORUS_FIELD_OPACITY_TARGET - dormant_target);
        }

        let delta = front_opacity_gain * (target_opacity - current_opacity);
        updates[row] = delta.clamp(-front_max_opacity_update, front_max_opacity_update);
    }

    updates
}

pub(crate) const DEFAULT_LOCAL_FRONT_ROW_FRACTION: usize = 16;
pub(crate) const DEFAULT_LOCAL_FRONT_MAX_CANDIDATES: usize = 64;

pub(crate) fn default_local_front_candidate_count(rows: usize) -> usize {
    if rows == 0 {
        0
    } else {
        rows.div_ceil(DEFAULT_LOCAL_FRONT_ROW_FRACTION)
            .clamp(1, DEFAULT_LOCAL_FRONT_MAX_CANDIDATES)
    }
}

pub(crate) fn local_front_weights(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
) -> Vec<f32> {
    local_front_weights_with_min_candidates(config, positions, states, front_radius, 0)
}

pub(crate) fn local_front_weights_with_min_candidates(
    config: &NpaConfig,
    positions: &[[f32; 4]],
    states: &[f32],
    front_radius: f32,
    min_front_candidates: usize,
) -> Vec<f32> {
    let rows = positions.len();
    let mut weights = vec![0.0; rows];
    if config.state_dims <= 3 || rows == 0 || front_radius <= 0.0 {
        return weights;
    }
    let active_threshold = -1.0_f32;
    let mut active_count = 0usize;
    let mut dormant_distances = Vec::new();
    for (row, position) in positions.iter().enumerate() {
        let current_opacity = states[row * config.state_dims + 3];
        if current_opacity > active_threshold {
            weights[row] = 1.0;
            active_count += 1;
            continue;
        }

        let mut nearest_active_distance2 = f32::MAX;
        for (other_row, other_position) in positions.iter().enumerate() {
            let other_opacity = states[other_row * config.state_dims + 3];
            if other_opacity <= active_threshold {
                continue;
            }
            let dx = position[0] - other_position[0];
            let dy = position[1] - other_position[1];
            let dz = position[2] - other_position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < nearest_active_distance2 {
                nearest_active_distance2 = distance2;
            }
        }
        if nearest_active_distance2.is_finite() {
            dormant_distances.push((row, nearest_active_distance2));
        }
    }
    let mut effective_front_radius = front_radius;
    let mut requested_front_rows = Vec::new();
    if active_count > 0 && !dormant_distances.is_empty() {
        dormant_distances.sort_by(|(_, lhs), (_, rhs)| {
            lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal)
        });
        let default_front = default_local_front_candidate_count(rows);
        let requested_front = min_front_candidates.min(rows / 2).max(default_front);
        let desired_front = dormant_distances.len().min(requested_front);
        if desired_front > 0 {
            if min_front_candidates > default_front {
                requested_front_rows.extend(
                    dormant_distances
                        .iter()
                        .take(desired_front)
                        .map(|(row, _)| *row),
                );
            }
            let sparse_radius = dormant_distances[desired_front - 1].1.sqrt() * 1.05;
            if sparse_radius.is_finite() {
                effective_front_radius = effective_front_radius.max(sparse_radius);
            }
        }
    }
    let front_radius2 = effective_front_radius * effective_front_radius;
    if front_radius2 <= 0.0 || !front_radius2.is_finite() {
        return weights;
    }
    for (row, nearest_active_distance2) in dormant_distances {
        if nearest_active_distance2 <= front_radius2 {
            let weight = (1.0 - (nearest_active_distance2 / front_radius2).sqrt()).max(0.0);
            weights[row] = if requested_front_rows.contains(&row) {
                weight.max(0.25)
            } else {
                weight
            };
        }
    }
    weights
}

pub(crate) fn add_target_coverage_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    target_update: &mut [f32],
    coverage_gain: f32,
    coverage_samples: usize,
    coverage_mode: CoverageUpdateModeArg,
    coverage_softness: f32,
    coverage_repulsion_gain: f32,
    coverage_gap_gain: f32,
    coverage_repulsion_radius: f32,
    coverage_normal_weight: f32,
    seed_scale: f32,
    max_update_norm: f32,
    front_states: Option<&[f32]>,
    front_radius: f32,
) {
    if coverage_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let rows = positions.len();
    let output_dims = config.update_dims();
    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));

    if coverage_mode != CoverageUpdateModeArg::HardNearest {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        if eligible_rows.is_empty() {
            return;
        }
        let coverage_updates = match coverage_mode {
            CoverageUpdateModeArg::SoftChamfer => render_proxy_soft_chamfer_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                coverage_softness,
                coverage_repulsion_gain,
                coverage_repulsion_radius,
                coverage_normal_weight,
                seed_scale,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::GapFarthest => render_proxy_gap_farthest_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::SlicedOt => render_proxy_sliced_ot_coverage_updates(
                target,
                positions,
                coverage_gain,
                coverage_samples,
                max_update_norm,
                &eligible_rows,
                vec![[0.0; 3]; rows],
            ),
            CoverageUpdateModeArg::HardNearest => unreachable!("handled by outer branch"),
        };
        for row in 0..rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            if front_weight <= 1.0e-3 {
                continue;
            }
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * coverage_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
        if (coverage_mode != CoverageUpdateModeArg::SoftChamfer
            && coverage_repulsion_gain > 0.0
            && coverage_repulsion_gain.is_finite())
            || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
        {
            let mut repulsion_updates = vec![[0.0; 3]; rows];
            if coverage_mode != CoverageUpdateModeArg::SoftChamfer {
                add_surface_tangent_repulsion_to_updates(
                    target,
                    positions,
                    &eligible_rows,
                    coverage_gain,
                    coverage_repulsion_gain,
                    coverage_repulsion_radius,
                    seed_scale,
                    max_update_norm,
                    &mut repulsion_updates,
                );
            }
            add_surface_gap_relocation_to_updates(
                target,
                positions,
                &eligible_rows,
                coverage_gain,
                coverage_gap_gain,
                coverage_samples,
                coverage_normal_weight,
                seed_scale,
                max_update_norm,
                &mut repulsion_updates,
            );
            for row in 0..rows {
                let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
                if front_weight <= 1.0e-3 {
                    continue;
                }
                let base = row * output_dims;
                for axis in 0..3 {
                    target_update[base + axis] += front_weight * repulsion_updates[row][axis];
                }
                clamp_target_motion_update(target_update, base, max_update_norm);
            }
        }
        return;
    }

    let samples = coverage_samples.max(rows.max(512));
    let mut residual_sums = vec![[0.0_f32; 3]; rows];
    let mut counts = vec![0usize; rows];

    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_row = 0usize;
        let mut best_distance2 = f32::MAX;
        for (row, position) in positions.iter().enumerate() {
            if front_weights
                .as_ref()
                .is_some_and(|weights| weights[row] <= 1.0e-3)
            {
                continue;
            }
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let distance2 = dx * dx + dy * dy + dz * dz;
            if distance2 < best_distance2 {
                best_distance2 = distance2;
                best_row = row;
            }
        }
        if !best_distance2.is_finite() {
            continue;
        }

        residual_sums[best_row][0] += sample.position[0] - positions[best_row][0];
        residual_sums[best_row][1] += sample.position[1] - positions[best_row][1];
        residual_sums[best_row][2] += sample.position[2] - positions[best_row][2];
        counts[best_row] += 1;
    }

    for row in 0..rows {
        let count = counts[row];
        if count == 0 {
            continue;
        }
        let base = row * output_dims;
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        let scale = coverage_gain * front_weight / count as f32;
        for axis in 0..3 {
            target_update[base + axis] += residual_sums[row][axis] * scale;
        }
        clamp_target_motion_update(target_update, base, max_update_norm);
    }
    if (coverage_repulsion_gain > 0.0 && coverage_repulsion_gain.is_finite())
        || (coverage_gap_gain > 0.0 && coverage_gap_gain.is_finite())
    {
        let eligible_rows = (0..rows)
            .filter(|&row| {
                front_weights
                    .as_ref()
                    .is_none_or(|weights| weights[row] > 1.0e-3)
            })
            .collect::<Vec<_>>();
        let mut repulsion_updates = vec![[0.0; 3]; rows];
        add_surface_tangent_repulsion_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_repulsion_gain,
            coverage_repulsion_radius,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        add_surface_gap_relocation_to_updates(
            target,
            positions,
            &eligible_rows,
            coverage_gain,
            coverage_gap_gain,
            coverage_samples,
            coverage_normal_weight,
            seed_scale,
            max_update_norm,
            &mut repulsion_updates,
        );
        for row in eligible_rows {
            let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
            let base = row * output_dims;
            for axis in 0..3 {
                target_update[base + axis] += front_weight * repulsion_updates[row][axis];
            }
            clamp_target_motion_update(target_update, base, max_update_norm);
        }
    }
}

pub(crate) fn clamp_target_motion_update(
    target_update: &mut [f32],
    base: usize,
    max_update_norm: f32,
) {
    let norm = (target_update[base].powi(2)
        + target_update[base + 1].powi(2)
        + target_update[base + 2].powi(2))
    .sqrt();
    if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
        let clamp = max_update_norm / norm;
        for axis in 0..3 {
            target_update[base + axis] *= clamp;
        }
    }
}

pub(crate) fn add_target_extent_updates_for_rows(
    config: &NpaConfig,
    target: &TriangleMeshTarget,
    positions: &[[f32; 4]],
    front_states: Option<&[f32]>,
    target_update: &mut [f32],
    extent_gain: f32,
    max_update_norm: f32,
    front_radius: f32,
) {
    if extent_gain <= 0.0 || positions.is_empty() {
        return;
    }

    let front_weights =
        front_states.map(|states| local_front_weights(config, positions, states, front_radius));
    let mut active_min = [f32::MAX; 3];
    let mut active_max = [f32::MIN; 3];
    let mut active_rows = 0usize;
    for (row, position) in positions.iter().enumerate() {
        if front_weights
            .as_ref()
            .is_some_and(|weights| weights[row] <= 1.0e-3)
        {
            continue;
        }
        active_rows += 1;
        for axis in 0..3 {
            active_min[axis] = active_min[axis].min(position[axis]);
            active_max[axis] = active_max[axis].max(position[axis]);
        }
    }
    if active_rows == 0 {
        return;
    }

    let (target_min, target_max) = target.bounds();
    let output_dims = config.update_dims();
    for (row, position) in positions.iter().enumerate() {
        let front_weight = front_weights.as_ref().map_or(1.0, |weights| weights[row]);
        if front_weight <= 1.0e-3 {
            continue;
        }
        let base = row * output_dims;
        for axis in 0..3 {
            let active_extent = (active_max[axis] - active_min[axis]).max(1.0e-4);
            let t = ((position[axis] - active_min[axis]) / active_extent).clamp(0.0, 1.0);
            let min_weight = (1.0 - t).powi(3);
            let max_weight = t.powi(3);
            let residual = min_weight * (target_min[axis] - position[axis])
                + max_weight * (target_max[axis] - position[axis]);
            target_update[base + axis] += extent_gain * front_weight * residual;
        }
        let norm = (target_update[base].powi(2)
            + target_update[base + 1].powi(2)
            + target_update[base + 2].powi(2))
        .sqrt();
        if max_update_norm.is_finite() && norm > max_update_norm.max(1.0e-6) {
            let clamp = max_update_norm / norm;
            for axis in 0..3 {
                target_update[base + axis] *= clamp;
            }
        }
    }
}

pub(crate) fn torus_implicit_training_position(
    row: usize,
    scale: f32,
    rng: &mut StdRng,
) -> [f32; 3] {
    match row % 4 {
        0 => uv_torus_dense_seed_position(rng, scale),
        1 => {
            let surface = uv_torus_continuous_surface_position(rng, scale);
            [
                surface[0] + rng.random_range(-0.18..0.18) * scale,
                surface[1] + rng.random_range(-0.18..0.18) * scale,
                surface[2] + rng.random_range(-0.18..0.18) * scale,
            ]
        }
        2 => uv_torus_continuous_volume_position(rng, scale),
        _ => {
            let radius = scale * (1.0 + UV_TORUS_MINOR_RATIO) * 0.95;
            [
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
                rng.random_range(-radius..radius),
            ]
        }
    }
}

pub(crate) fn utah_teapot_training_position(
    row: usize,
    scale: f32,
    target: &TriangleMeshTarget,
    rng: &mut StdRng,
) -> [f32; 3] {
    match row % 4 {
        0 => utah_teapot_dense_seed_position(rng, target),
        1 => {
            let sample = target.surface_sample(row);
            [
                sample.position[0] + rng.random_range(-0.14..0.14) * scale,
                sample.position[1] + rng.random_range(-0.14..0.14) * scale,
                sample.position[2] + rng.random_range(-0.14..0.14) * scale,
            ]
        }
        2 => target.near_surface_query(row * 17 + 3, rng.random_range(-0.16..0.16) * scale),
        _ => [
            rng.random_range(-1.15..1.15) * scale,
            rng.random_range(-0.70..0.70) * scale,
            rng.random_range(-0.55..0.75) * scale,
        ],
    }
}
