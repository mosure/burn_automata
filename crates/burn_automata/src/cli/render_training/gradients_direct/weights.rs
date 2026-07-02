use super::*;

pub(crate) fn accumulate_scaled_weight_delta(
    delta: &mut NpaWeights,
    before: &NpaWeights,
    after: &NpaWeights,
    scale: f32,
) {
    accumulate_scaled_weight_delta_slice(&mut delta.w1, &before.w1, &after.w1, scale);
    accumulate_scaled_weight_delta_slice(&mut delta.b1, &before.b1, &after.b1, scale);
    accumulate_scaled_weight_delta_slice(&mut delta.w2, &before.w2, &after.w2, scale);
    accumulate_scaled_weight_delta_slice(&mut delta.b2, &before.b2, &after.b2, scale);
}

pub(crate) fn accumulate_scaled_adapter_delta(
    delta: &mut NpaLowRankAdapter,
    before: &NpaLowRankAdapter,
    after: &NpaLowRankAdapter,
    scale: f32,
) {
    accumulate_scaled_weight_delta_slice(
        &mut delta.w1_down,
        &before.w1_down,
        &after.w1_down,
        scale,
    );
    accumulate_scaled_weight_delta_slice(&mut delta.w1_up, &before.w1_up, &after.w1_up, scale);
    accumulate_scaled_weight_delta_slice(
        &mut delta.w2_down,
        &before.w2_down,
        &after.w2_down,
        scale,
    );
    accumulate_scaled_weight_delta_slice(&mut delta.w2_up, &before.w2_up, &after.w2_up, scale);
    accumulate_scaled_weight_delta_slice(
        &mut delta.b1_delta,
        &before.b1_delta,
        &after.b1_delta,
        scale,
    );
    accumulate_scaled_weight_delta_slice(
        &mut delta.b2_delta,
        &before.b2_delta,
        &after.b2_delta,
        scale,
    );
}

pub(crate) fn accumulate_scaled_weight_delta_slice(
    delta: &mut [f32],
    before: &[f32],
    after: &[f32],
    scale: f32,
) {
    debug_assert_eq!(delta.len(), before.len());
    debug_assert_eq!(before.len(), after.len());
    for ((delta_value, before_value), after_value) in
        delta.iter_mut().zip(before.iter()).zip(after.iter())
    {
        *delta_value += (after_value - before_value) * scale;
    }
}

pub(crate) fn output_channel_delta_norm(
    before: &NpaWeights,
    after: &NpaWeights,
    hidden_dims: usize,
    output: usize,
) -> f32 {
    if hidden_dims == 0 || output >= before.b2.len() || output >= after.b2.len() {
        return 0.0;
    }
    let start = output.saturating_mul(hidden_dims);
    let end = start.saturating_add(hidden_dims);
    if end > before.w2.len() || end > after.w2.len() {
        return 0.0;
    }
    let bias_delta = after.b2[output] - before.b2[output];
    let weight_delta2 = before.w2[start..end]
        .iter()
        .zip(after.w2[start..end].iter())
        .map(|(lhs, rhs)| {
            let delta = rhs - lhs;
            delta * delta
        })
        .sum::<f32>();
    (bias_delta * bias_delta + weight_delta2).sqrt()
}

pub(crate) fn spatial_output_delta_norm(
    before: &NpaWeights,
    after: &NpaWeights,
    hidden_dims: usize,
    spatial_dims: usize,
) -> f32 {
    (0..spatial_dims)
        .map(|output| output_channel_delta_norm(before, after, hidden_dims, output).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub(crate) fn apply_average_weight_delta(
    weights: &mut NpaWeights,
    before: &NpaWeights,
    delta: &NpaWeights,
    scale: f32,
) {
    apply_average_weight_delta_slice(&mut weights.w1, &before.w1, &delta.w1, scale);
    apply_average_weight_delta_slice(&mut weights.b1, &before.b1, &delta.b1, scale);
    apply_average_weight_delta_slice(&mut weights.w2, &before.w2, &delta.w2, scale);
    apply_average_weight_delta_slice(&mut weights.b2, &before.b2, &delta.b2, scale);
}

pub(crate) fn apply_average_adapter_delta(
    adapter: &mut NpaLowRankAdapter,
    before: &NpaLowRankAdapter,
    delta: &NpaLowRankAdapter,
    scale: f32,
) {
    apply_average_weight_delta_slice(&mut adapter.w1_down, &before.w1_down, &delta.w1_down, scale);
    apply_average_weight_delta_slice(&mut adapter.w1_up, &before.w1_up, &delta.w1_up, scale);
    apply_average_weight_delta_slice(&mut adapter.w2_down, &before.w2_down, &delta.w2_down, scale);
    apply_average_weight_delta_slice(&mut adapter.w2_up, &before.w2_up, &delta.w2_up, scale);
    apply_average_weight_delta_slice(
        &mut adapter.b1_delta,
        &before.b1_delta,
        &delta.b1_delta,
        scale,
    );
    apply_average_weight_delta_slice(
        &mut adapter.b2_delta,
        &before.b2_delta,
        &delta.b2_delta,
        scale,
    );
}

pub(crate) fn apply_average_weight_delta_slice(
    weights: &mut [f32],
    before: &[f32],
    delta: &[f32],
    scale: f32,
) {
    debug_assert_eq!(weights.len(), before.len());
    debug_assert_eq!(before.len(), delta.len());
    for ((weight, before_value), delta_value) in weights.iter_mut().zip(before.iter()).zip(delta) {
        *weight = before_value + delta_value * scale;
    }
}
