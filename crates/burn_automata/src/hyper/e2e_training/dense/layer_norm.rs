//! Fused modulated layer normalization with a backend-generic autodiff boundary.

use super::*;

#[derive(Clone, Debug)]
struct ModulatedLayerNormAdjointState {
    input: Tensor3Inner,
    scale: Tensor2Inner,
    stats: Tensor3Inner,
}

#[derive(Clone, Copy, Debug)]
struct ModulatedLayerNormAdjointOp;

impl Backward<InnerBackend, 3> for ModulatedLayerNormAdjointOp {
    type State = ModulatedLayerNormAdjointState;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer,
    ) {
        let [input_parent, shift_parent, scale_parent] = ops.parents;
        if input_parent.is_none() && shift_parent.is_none() && scale_parent.is_none() {
            return;
        }
        let output_grad = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
            grads.consume::<InnerBackend>(&ops.node),
        ));

        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        if let Some(output) = InnerBackend::modulated_layer_norm_cube_backward(
            ops.state.input.clone(),
            ops.state.scale.clone(),
            output_grad.clone(),
            ops.state.stats.clone(),
        ) {
            let output = output.unwrap_or_else(|err| {
                panic!("modulated layer norm CubeCL adjoint failed: {err}")
            });
            if let Some(parent) = input_parent {
                grads.register::<InnerBackend>(
                    parent.id,
                    output.input_grad.into_primitive().tensor(),
                );
            }
            if let Some(parent) = shift_parent {
                grads.register::<InnerBackend>(
                    parent.id,
                    output.shift_grad.into_primitive().tensor(),
                );
            }
            if let Some(parent) = scale_parent {
                grads.register::<InnerBackend>(
                    parent.id,
                    output.scale_grad.into_primitive().tensor(),
                );
            }
            return;
        }

        let (input_grad, shift_grad, scale_grad) = modulated_layer_norm_inner_adjoint(
            ops.state.input,
            ops.state.scale,
            output_grad,
            ops.state.stats,
        );
        if let Some(parent) = input_parent {
            grads.register::<InnerBackend>(parent.id, input_grad.into_primitive().tensor());
        }
        if let Some(parent) = shift_parent {
            grads.register::<InnerBackend>(parent.id, shift_grad.into_primitive().tensor());
        }
        if let Some(parent) = scale_parent {
            grads.register::<InnerBackend>(parent.id, scale_grad.into_primitive().tensor());
        }
    }
}

pub(super) fn modulated_layer_norm3(
    input: Tensor3,
    shift: Tensor2,
    scale: Tensor2,
) -> Tensor3 {
    let input_primitive = input.into_primitive().tensor();
    let shift_primitive = shift.into_primitive().tensor();
    let scale_primitive = scale.into_primitive().tensor();
    let input_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
        input_primitive.primitive.clone(),
    ));
    let shift_inner = Tensor::<InnerBackend, 2>::from_primitive(TensorPrimitive::Float(
        shift_primitive.primitive.clone(),
    ));
    let scale_inner = Tensor::<InnerBackend, 2>::from_primitive(TensorPrimitive::Float(
        scale_primitive.primitive.clone(),
    ));

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    let device_output = InnerBackend::modulated_layer_norm_cube_forward(
        input_inner.clone(),
        shift_inner.clone(),
        scale_inner.clone(),
    )
    .map(|output| {
        output.unwrap_or_else(|err| panic!("modulated layer norm CubeCL forward failed: {err}"))
    });
    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    let (output, stats) = if let Some(output) = device_output {
        (output.output, output.stats)
    } else {
        modulated_layer_norm_inner(
            input_inner.clone(),
            shift_inner.clone(),
            scale_inner.clone(),
        )
    };
    #[cfg(not(any(feature = "backend_wgpu", feature = "backend_cuda")))]
    let (output, stats) =
        modulated_layer_norm_inner(input_inner.clone(), shift_inner, scale_inner.clone());
    let output = output.into_primitive().tensor();
    let state = ModulatedLayerNormAdjointState {
        input: input_inner,
        scale: scale_inner,
        stats,
    };
    let prep = ModulatedLayerNormAdjointOp
        .prepare::<NoCheckpointing>([
            input_primitive.node.clone(),
            shift_primitive.node.clone(),
            scale_primitive.node.clone(),
        ])
        .compute_bound();
    let output = match prep.stateful() {
        OpsKind::Tracked(prep) => prep.finish(state, output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    Tensor::<BurnBackend, 3>::from_primitive(TensorPrimitive::Float(output))
}

fn modulated_layer_norm_inner(
    input: Tensor3Inner,
    shift: Tensor2Inner,
    scale: Tensor2Inner,
) -> (Tensor3Inner, Tensor3Inner) {
    let [batches, rows, dims] = input.shape().dims::<3>();
    let mean = input.clone().mean_dim(2);
    let centered = input - mean.clone().expand([batches, rows, dims]);
    let inv_std = centered
        .clone()
        .mul(centered.clone())
        .mean_dim(2)
        .add_scalar(EPSILON)
        .sqrt()
        .recip();
    let normalized = centered.mul(inv_std.clone().expand([batches, rows, dims]));
    let output = normalized
        .mul(
            scale
                .add_scalar(1.0)
                .unsqueeze_dim::<3>(1)
                .expand([batches, rows, dims]),
        )
        + shift
            .unsqueeze_dim::<3>(1)
            .expand([batches, rows, dims]);
    let stats = Tensor::cat(vec![mean, inv_std], 2);
    (output, stats)
}

fn modulated_layer_norm_inner_adjoint(
    input: Tensor3Inner,
    scale: Tensor2Inner,
    output_grad: Tensor3Inner,
    stats: Tensor3Inner,
) -> (Tensor3Inner, Tensor2Inner, Tensor2Inner) {
    let [batches, rows, dims] = input.shape().dims::<3>();
    let mean = stats.clone().narrow(2, 0, 1);
    let inv_std = stats.narrow(2, 1, 1);
    let normalized = (input - mean.expand([batches, rows, dims]))
        .mul(inv_std.clone().expand([batches, rows, dims]));
    let normalized_grad = output_grad.clone().mul(
        scale
            .add_scalar(1.0)
            .unsqueeze_dim::<3>(1)
            .expand([batches, rows, dims]),
    );
    let grad_mean = normalized_grad.clone().mean_dim(2);
    let grad_normalized_mean = normalized_grad
        .clone()
        .mul(normalized.clone())
        .mean_dim(2);
    let shift_grad = output_grad.clone().sum_dim(1).squeeze_dim::<2>(1);
    let scale_grad = output_grad
        .mul(normalized.clone())
        .sum_dim(1)
        .squeeze_dim::<2>(1);
    let input_grad = inv_std.expand([batches, rows, dims]).mul(
        normalized_grad
            - grad_mean.expand([batches, rows, dims])
            - normalized.mul(grad_normalized_mean.expand([batches, rows, dims])),
    );
    (input_grad, shift_grad, scale_grad)
}
