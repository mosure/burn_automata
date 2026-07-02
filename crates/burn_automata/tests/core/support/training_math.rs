use super::super::*;

pub(crate) fn mse(model: &NpaModel, batch: &SupervisedBatch) -> f32 {
    let (dx, ds) = model.forward_from_features(&batch.features).unwrap();
    let mut output = Vec::with_capacity(dx.len() * model.config.update_dims());
    for (row, delta) in dx.iter().enumerate() {
        output.extend_from_slice(&delta[..model.config.spatial_dims]);
        let base = row * model.config.state_dims;
        output.extend_from_slice(&ds[base..base + model.config.state_dims]);
    }
    output
        .iter()
        .zip(batch.target_update.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / output.len() as f32
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GradientParam {
    W1(usize),
    B1(usize),
    W2(usize),
    B2(usize),
}

pub(crate) fn analytic_gradient(
    grads: &burn_automata::SupervisedGradients,
    param: GradientParam,
) -> f32 {
    match param {
        GradientParam::W1(index) => grads.w1[index],
        GradientParam::B1(index) => grads.b1[index],
        GradientParam::W2(index) => grads.w2[index],
        GradientParam::B2(index) => grads.b2[index],
    }
}

pub(crate) fn finite_difference_gradient(
    model: &NpaModel,
    batch: &SupervisedBatch,
    param: GradientParam,
) -> f32 {
    let eps = 1.0e-3;
    let mut plus = model.clone();
    perturb_param(&mut plus, param, eps);
    let mut minus = model.clone();
    perturb_param(&mut minus, param, -eps);
    (supervised_loss(&plus, batch).unwrap() - supervised_loss(&minus, batch).unwrap()) / (2.0 * eps)
}

fn perturb_param(model: &mut NpaModel, param: GradientParam, delta: f32) {
    match param {
        GradientParam::W1(index) => model.weights.w1[index] += delta,
        GradientParam::B1(index) => model.weights.b1[index] += delta,
        GradientParam::W2(index) => model.weights.w2[index] += delta,
        GradientParam::B2(index) => model.weights.b2[index] += delta,
    }
}
