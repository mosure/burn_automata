#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

pub(crate) fn scale_state_adjoint(state: &mut [f32], weight: f32) {
    if weight == 1.0 {
        return;
    }
    for value in state {
        *value *= weight;
    }
}

pub(crate) fn scale_position_adjoint(position: &mut [[f32; 4]], weight: f32, spatial_dims: usize) {
    if weight == 1.0 {
        return;
    }
    for row in position {
        for value in row.iter_mut().take(spatial_dims) {
            *value *= weight;
        }
    }
}

pub(crate) fn zero_supervised_gradients(model: &NpaModel) -> SupervisedGradients {
    SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: Vec::new(),
    }
}

pub(crate) fn accumulate_supervised_gradients(
    total: &mut SupervisedGradients,
    step: &SupervisedGradients,
) {
    add_assign_slice(&mut total.w1, &step.w1);
    add_assign_slice(&mut total.b1, &step.b1);
    add_assign_slice(&mut total.w2, &step.w2);
    add_assign_slice(&mut total.b2, &step.b2);
    total.features.extend_from_slice(&step.features);
}

#[cfg(test)]
pub(crate) fn normalize_supervised_gradients_by_rows(
    gradients: &mut SupervisedGradients,
    input_dims: usize,
) {
    if input_dims == 0
        || gradients.features.is_empty()
        || gradients.features.len() % input_dims != 0
    {
        return;
    }
    let rows = gradients.features.len() / input_dims;
    if rows == 0 {
        return;
    }
    let scale = 1.0 / rows as f32;
    scale_slice(&mut gradients.w1, scale);
    scale_slice(&mut gradients.b1, scale);
    scale_slice(&mut gradients.w2, scale);
    scale_slice(&mut gradients.b2, scale);
}

pub(crate) fn normalize_direct_rollout_gradients(
    gradients: &mut SupervisedGradients,
    input_dims: usize,
) {
    if input_dims == 0
        || gradients.features.is_empty()
        || gradients.features.len() % input_dims != 0
    {
        return;
    }
    let rows = gradients.features.len() / input_dims;
    if rows == 0 {
        return;
    }
    let exponent = DIRECT_ROLLOUT_GRADIENT_ROW_NORMALIZATION_EXPONENT.clamp(0.0, 1.0);
    let scale = 1.0 / (rows as f32).powf(exponent);
    scale_slice(&mut gradients.w1, scale);
    scale_slice(&mut gradients.b1, scale);
    scale_slice(&mut gradients.w2, scale);
    scale_slice(&mut gradients.b2, scale);
}

pub(crate) fn retain_material_output_gradients(
    model: &NpaModel,
    gradients: &mut SupervisedGradients,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(material_channel) = growth_3d_material_opacity_channel(model.config.state_dims) else {
        return Err(std::io::Error::other(
            "material-output-only training requires a material opacity channel",
        )
        .into());
    };
    gradients.w1.fill(0.0);
    gradients.b1.fill(0.0);
    let output_dims = model.config.update_dims();
    let material_output = model.config.spatial_dims + material_channel;
    for output in 0..output_dims {
        if output == material_output {
            continue;
        }
        let start = output * model.config.hidden_dims;
        let end = start + model.config.hidden_dims;
        gradients.w2[start..end].fill(0.0);
        gradients.b2[output] = 0.0;
    }
    Ok(())
}

pub(crate) fn add_assign_slice(total: &mut [f32], step: &[f32]) {
    debug_assert_eq!(total.len(), step.len());
    for (dst, src) in total.iter_mut().zip(step.iter()) {
        *dst += *src;
    }
}

pub(crate) fn scale_slice(values: &mut [f32], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for value in values {
        *value *= scale;
    }
}

pub(crate) fn clamp_state_adjoint_row(row: &mut [f32]) {
    const MAX_STATE_ADJOINT_NORM: f32 = 10.0;
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= MAX_STATE_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_STATE_ADJOINT_NORM / norm;
    for value in row {
        *value *= scale;
    }
}

pub(crate) fn clamp_position_adjoint_row(row: &mut [f32; 4], spatial_dims: usize) {
    const MAX_POSITION_ADJOINT_NORM: f32 = 10.0;
    let norm = row
        .iter()
        .take(spatial_dims)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= MAX_POSITION_ADJOINT_NORM || norm <= 1.0e-12 {
        return;
    }
    let scale = MAX_POSITION_ADJOINT_NORM / norm;
    for value in row.iter_mut().take(spatial_dims) {
        *value *= scale;
    }
}

pub(crate) fn accumulate_motion_output_gradient(
    config: &NpaConfig,
    grid_eps: f32,
    raw_update: &[f32],
    dloss_ddx: [f32; 3],
    output_gradient: &mut [f32],
) {
    let dims = config.spatial_dims;
    let motion_scale = config.alpha * config.motion_eps(grid_eps);
    let mut norm2 = 0.0_f32;
    for value in raw_update.iter().take(dims) {
        norm2 += value * value;
    }
    let norm = norm2.sqrt();
    let denom = 1.0 + norm;
    let dot = raw_update
        .iter()
        .zip(dloss_ddx.iter())
        .take(dims)
        .map(|(raw, grad)| raw * grad)
        .sum::<f32>();

    for axis in 0..dims {
        let mut grad = motion_scale * dloss_ddx[axis] / denom;
        if norm > 1.0e-6 {
            grad -= motion_scale * raw_update[axis] * dot / (norm * denom * denom);
        }
        output_gradient[axis] += grad;
    }
}

#[cfg(test)]
pub(crate) fn cap_output_gradient_channel_rms(
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
) -> usize {
    cap_output_gradient_channel_rms_impl(output_gradients, output_dims, rms_cap, &[])
}

#[cfg(test)]
pub(crate) fn cap_output_gradient_channel_rms_with_liveness_cap(
    config: &NpaConfig,
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
    liveness_rms_cap: f32,
) -> usize {
    cap_output_gradient_channel_rms_with_state_caps(
        config,
        output_gradients,
        output_dims,
        rms_cap,
        liveness_rms_cap,
        rms_cap,
    )
}

pub(crate) fn cap_output_gradient_channel_rms_with_state_caps(
    config: &NpaConfig,
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
    liveness_rms_cap: f32,
    material_rms_cap: f32,
) -> usize {
    let liveness_output = if config.state_dims > GROWTH_3D_LIVENESS_CHANNEL {
        Some(config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL)
    } else {
        None
    };
    let liveness_cap = if liveness_rms_cap.is_finite() && liveness_rms_cap > rms_cap {
        Some(liveness_rms_cap)
    } else {
        None
    };
    let material_output = growth_3d_material_opacity_channel(config.state_dims)
        .map(|channel| config.spatial_dims + channel)
        .filter(|channel| *channel < output_dims && Some(*channel) != liveness_output);
    let material_cap = if material_rms_cap.is_finite() && material_rms_cap > rms_cap {
        Some(material_rms_cap)
    } else {
        None
    };
    let overrides = [
        liveness_output.zip(liveness_cap),
        material_output.zip(material_cap),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    cap_output_gradient_channel_rms_impl(output_gradients, output_dims, rms_cap, &overrides)
}

pub(crate) fn cap_output_gradient_channel_rms_impl(
    output_gradients: &mut [f32],
    output_dims: usize,
    rms_cap: f32,
    channel_overrides: &[(usize, f32)],
) -> usize {
    if output_dims == 0
        || output_gradients.is_empty()
        || output_gradients.len() % output_dims != 0
        || rms_cap <= 0.0
        || !rms_cap.is_finite()
    {
        return 0;
    }
    let rows = output_gradients.len() / output_dims;
    let mut capped = 0usize;
    for output in 0..output_dims {
        let channel_cap = channel_overrides
            .iter()
            .copied()
            .find(|(channel, cap)| *channel == output && *cap > 0.0 && cap.is_finite())
            .map(|(_, cap)| cap)
            .unwrap_or(rms_cap);
        let rms = ((0..rows)
            .map(|row| {
                let value = output_gradients[row * output_dims + output];
                value * value
            })
            .sum::<f32>()
            / rows as f32)
            .sqrt();
        if !rms.is_finite() || rms <= channel_cap {
            continue;
        }
        let scale = channel_cap / rms;
        for row in 0..rows {
            output_gradients[row * output_dims + output] *= scale;
        }
        capped += 1;
    }
    capped
}

pub(crate) fn boost_sparse_output_channel_rms(
    output_gradients: &mut [f32],
    output_dims: usize,
    channels: impl IntoIterator<Item = usize>,
    target_nonzero_rms: f32,
    max_scale: f32,
) -> usize {
    if output_dims == 0
        || output_gradients.is_empty()
        || output_gradients.len() % output_dims != 0
        || target_nonzero_rms <= 0.0
        || !target_nonzero_rms.is_finite()
        || max_scale <= 1.0
        || !max_scale.is_finite()
    {
        return 0;
    }
    let rows = output_gradients.len() / output_dims;
    let mut boosted = 0usize;
    for output in channels {
        if output >= output_dims {
            continue;
        }
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for row in 0..rows {
            let value = output_gradients[row * output_dims + output];
            if value.abs() <= 1.0e-12 {
                continue;
            }
            sum += value * value;
            count += 1;
        }
        if count == 0 {
            continue;
        }
        let rms = (sum / count as f32).sqrt();
        if !rms.is_finite() || rms <= 0.0 || rms >= target_nonzero_rms {
            continue;
        }
        let scale = (target_nonzero_rms / rms).min(max_scale);
        for row in 0..rows {
            output_gradients[row * output_dims + output] *= scale;
        }
        boosted += 1;
    }
    boosted
}

pub(crate) fn add_output_gradients(target: &mut [f32], source: &[f32]) {
    debug_assert_eq!(target.len(), source.len());
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}
