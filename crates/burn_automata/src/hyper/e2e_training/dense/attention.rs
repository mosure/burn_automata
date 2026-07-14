//! Tiled attention forward with an explicit Burn autodiff adjoint.

use super::*;

#[derive(Clone, Debug)]
pub(super) struct AttentionAdjointState {
    query: Tensor4Inner,
    key: Tensor4Inner,
    value: Tensor4Inner,
    scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionAdjointOp;

impl Backward<InnerBackend, 3> for AttentionAdjointOp {
    type State = AttentionAdjointState;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer,
    ) {
        let [query_parent, key_parent, value_parent] = ops.parents;
        if query_parent.is_none() && key_parent.is_none() && value_parent.is_none() {
            return;
        }
        let output_grad = Tensor::<InnerBackend, 4>::from_primitive(TensorPrimitive::Float(
            grads.consume::<InnerBackend>(&ops.node),
        ));
        let query = ops.state.query;
        let key = ops.state.key;
        let value = ops.state.value;
        let [batches, heads, query_rows, key_rows] = [
            query.shape().dims::<4>()[0],
            query.shape().dims::<4>()[1],
            query.shape().dims::<4>()[2],
            key.shape().dims::<4>()[2],
        ];
        let probabilities = softmax(
            query
                .clone()
                .matmul(key.clone().swap_dims(2, 3))
                .mul_scalar(ops.state.scale),
            3,
        );

        if let Some(parent) = value_parent {
            let value_grad = probabilities
                .clone()
                .swap_dims(2, 3)
                .matmul(output_grad.clone());
            grads.register::<InnerBackend>(parent.id, value_grad.into_primitive().tensor());
        }
        if query_parent.is_none() && key_parent.is_none() {
            return;
        }

        let probability_grad = output_grad.matmul(value.swap_dims(2, 3));
        let centered_probability_grad = probability_grad.clone()
            - probability_grad
                .mul(probabilities.clone())
                .sum_dim(3)
                .expand([batches, heads, query_rows, key_rows]);
        let score_grad = probabilities.mul(centered_probability_grad);
        if let Some(parent) = query_parent {
            let query_grad = score_grad
                .clone()
                .matmul(key.clone())
                .mul_scalar(ops.state.scale);
            grads.register::<InnerBackend>(parent.id, query_grad.into_primitive().tensor());
        }
        if let Some(parent) = key_parent {
            let key_grad = score_grad
                .swap_dims(2, 3)
                .matmul(query)
                .mul_scalar(ops.state.scale);
            grads.register::<InnerBackend>(parent.id, key_grad.into_primitive().tensor());
        }
    }
}

pub(super) fn tiled_attention_adjoint(
    query: Tensor4,
    key: Tensor4,
    value: Tensor4,
) -> Tensor4 {
    let head_dims = query.shape().dims::<4>()[3];
    let scale = (head_dims as f32).sqrt().recip();
    let query_primitive = query.into_primitive().tensor();
    let key_primitive = key.into_primitive().tensor();
    let value_primitive = value.into_primitive().tensor();
    let query_inner = Tensor::<InnerBackend, 4>::from_primitive(TensorPrimitive::Float(
        query_primitive.primitive.clone(),
    ));
    let key_inner = Tensor::<InnerBackend, 4>::from_primitive(TensorPrimitive::Float(
        key_primitive.primitive.clone(),
    ));
    let value_inner = Tensor::<InnerBackend, 4>::from_primitive(TensorPrimitive::Float(
        value_primitive.primitive.clone(),
    ));
    let output = burn::tensor::module::attention(
        query_inner.clone(),
        key_inner.clone(),
        value_inner.clone(),
        None,
        None,
        Default::default(),
    )
    .into_primitive()
    .tensor();
    let state = AttentionAdjointState {
        query: query_inner,
        key: key_inner,
        value: value_inner,
        scale,
    };
    let prep = AttentionAdjointOp
        .prepare::<NoCheckpointing>([
            query_primitive.node.clone(),
            key_primitive.node.clone(),
            value_primitive.node.clone(),
        ])
        .compute_bound();
    let output = match prep.stateful() {
        OpsKind::Tracked(prep) => prep.finish(state, output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Tensor::<BurnBackend, 4>::from_primitive(TensorPrimitive::Float(output))
}
