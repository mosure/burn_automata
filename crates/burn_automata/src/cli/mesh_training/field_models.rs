#![allow(clippy::too_many_arguments)]

use super::*;

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
        seed_scale: DEFAULT_3D_MESH_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 2048,
        steps: 180,
        seed: 42,
        seed_scale: DEFAULT_3D_MESH_FIELD_SCALE,
        seed_mode: ParticleSeed::TeapotFieldDense3d,
    },
    MeshRolloutCaseConfig {
        particle_count: 8192,
        steps: 180,
        seed: 131,
        seed_scale: DEFAULT_3D_MESH_FIELD_SCALE,
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
    weights.b2[opacity_out] = DEFAULT_3D_FIELD_OPACITY_GAIN * DEFAULT_3D_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= DEFAULT_3D_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += DEFAULT_3D_FIELD_OPACITY_GAIN;

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
    weights.b2[opacity_out] = DEFAULT_3D_FIELD_OPACITY_GAIN * DEFAULT_3D_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= DEFAULT_3D_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += DEFAULT_3D_FIELD_OPACITY_GAIN;

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
    weights.b2[opacity_out] = DEFAULT_3D_FIELD_OPACITY_GAIN * DEFAULT_3D_FIELD_OPACITY_TARGET;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.0] -= DEFAULT_3D_FIELD_OPACITY_GAIN;
    weights.w2[opacity_out * config.hidden_dims + opacity_pair.1] += DEFAULT_3D_FIELD_OPACITY_GAIN;

    let (bounds_min, bounds_max) = utah_teapot_mesh_target(DEFAULT_3D_MESH_FIELD_SCALE).bounds();
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
