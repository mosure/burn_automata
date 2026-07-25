//! Adaptive material perception and its state-only Burn autodiff boundary.

use super::*;

#[derive(Clone, Debug)]
struct AdaptivePerceptionPreparedState {
    density: Tensor2Inner,
    raw_state_gradient: Tensor4Inner,
    state_gradient_inverse: Tensor3Inner,
}

#[derive(Clone)]
pub(super) struct AdaptivePerceptionBatchOutput {
    pub(super) features: Tensor3,
    pub(super) coarse_exposure: Tensor2,
}

#[derive(Clone, Debug)]
struct AdaptivePerceptionAdjointState {
    positions: Tensor3Inner,
    states: Tensor3Inner,
    represented_measure: Tensor2Inner,
    bandwidth: Tensor2Inner,
    prepared: AdaptivePerceptionPreparedState,
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: burn_automata_kernels::AdaptivePerceptionConfig,
    options: burn_automata_kernels::AdaptiveNpaPerceptionOptions,
    semantics: burn_automata_kernels::AdaptivePerceptionSemantics,
}

#[derive(Clone, Copy, Debug)]
struct AdaptivePerceptionAdjointOp;

impl Backward<InnerBackend, 1> for AdaptivePerceptionAdjointOp {
    type State = AdaptivePerceptionAdjointState;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer,
    ) {
        let [state_parent] = ops.parents;
        let Some(state_parent) = state_parent else {
            return;
        };
        let feature_grad = grads.consume::<InnerBackend>(&ops.node);
        let feature_grad =
            Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(feature_grad));
        let device = feature_grad.device();

        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        if let Some(device_adjoint) =
            InnerBackend::adaptive_npa_perception_cube_state_adjoint(
                ops.state.positions.clone(),
                ops.state.states.clone(),
                ops.state.represented_measure.clone(),
                ops.state.bandwidth.clone(),
                feature_grad.clone(),
                ops.state.prepared.density.clone(),
                ops.state.prepared.raw_state_gradient.clone(),
                ops.state.prepared.state_gradient_inverse.clone(),
                ops.state.config,
                ops.state.options,
                ops.state.semantics,
            )
        {
            let device_adjoint = device_adjoint
                .unwrap_or_else(|error| panic!("adaptive perception cube adjoint failed: {error}"));
            let state_grad = device_adjoint.state_grad;
            let state_grad =
                state_grad
                    .clone()
                    .mask_fill(state_grad.is_finite().bool_not(), 0.0);
            grads.register::<InnerBackend>(
                state_parent.id,
                state_grad.into_primitive().tensor(),
            );
            return;
        }

        let feature_grad = feature_grad
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|error| panic!("adaptive feature-gradient readback failed: {error}"));
        let positions = adaptive_reference_positions(&ops.state.positions);
        let states = tensor_values3(&ops.state.states, "adaptive states");
        let represented_measure =
            tensor_values2(&ops.state.represented_measure, "adaptive represented measure");
        let bandwidth = tensor_values2(&ops.state.bandwidth, "adaptive bandwidth");
        let mut state_grad = match ops.state.semantics {
            burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible => {
                burn_automata_kernels::adaptive_npa_perceive_state_adjoint_all_pairs(
                    &positions,
                    &states,
                    &represented_measure,
                    &bandwidth,
                    ops.state.batch_size,
                    ops.state.particle_count,
                    ops.state.state_dims,
                    ops.state.config,
                    ops.state.options,
                    &feature_grad,
                )
            }
            burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive => {
                burn_automata_kernels::adaptive_perceive_state_adjoint_all_pairs(
                    &positions,
                    &states,
                    &represented_measure,
                    &bandwidth,
                    ops.state.batch_size,
                    ops.state.particle_count,
                    ops.state.state_dims,
                    ops.state.config,
                    &feature_grad,
                )
            }
        }
        .unwrap_or_else(|error| panic!("adaptive perception state adjoint failed: {error}"));
        for value in &mut state_grad {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        let state_grad = Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(
                state_grad,
                [
                    ops.state.batch_size,
                    ops.state.particle_count,
                    ops.state.state_dims,
                ],
            ),
            &device,
        );
        grads.register::<InnerBackend>(state_parent.id, state_grad.into_primitive().tensor());
    }
}

pub(super) fn adaptive_npa_perception_batch(
    positions: Tensor3,
    states: Tensor3,
    represented_measure: Tensor2,
    bandwidth: Tensor2,
    config: burn_automata_kernels::AdaptivePerceptionConfig,
    options: burn_automata_kernels::AdaptiveNpaPerceptionOptions,
    semantics: burn_automata_kernels::AdaptivePerceptionSemantics,
) -> AdaptivePerceptionBatchOutput {
    let state_shape = states.shape().dims::<3>();
    let position_shape = positions.shape().dims::<3>();
    assert_eq!(
        position_shape,
        [state_shape[0], state_shape[1], 2],
        "adaptive perception expects positions [batch, particles, 2]"
    );
    assert_eq!(
        represented_measure.shape().dims::<2>(),
        [state_shape[0], state_shape[1]],
        "adaptive perception expects represented measure [batch, particles]"
    );
    assert_eq!(
        bandwidth.shape().dims::<2>(),
        [state_shape[0], state_shape[1]],
        "adaptive perception expects bandwidth [batch, particles]"
    );

    let positions = detach3(positions);
    let represented_measure = detach2(represented_measure);
    let bandwidth = detach2(bandwidth);
    let positions_primitive = positions.into_primitive().tensor();
    let states_primitive = states.into_primitive().tensor();
    let measure_primitive = represented_measure.into_primitive().tensor();
    let bandwidth_primitive = bandwidth.into_primitive().tensor();
    let positions_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
        positions_primitive.primitive.clone(),
    ));
    let states_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
        states_primitive.primitive.clone(),
    ));
    let measure_inner = Tensor::<InnerBackend, 2>::from_primitive(TensorPrimitive::Float(
        measure_primitive.primitive.clone(),
    ));
    let bandwidth_inner = Tensor::<InnerBackend, 2>::from_primitive(TensorPrimitive::Float(
        bandwidth_primitive.primitive.clone(),
    ));

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    let device_forward = InnerBackend::adaptive_npa_perception_cube_forward(
        positions_inner.clone(),
        states_inner.clone(),
        measure_inner.clone(),
        bandwidth_inner.clone(),
        config,
        options,
        semantics,
    );

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    let (output, coarse_exposure, prepared) = if let Some(device_forward) = device_forward {
        let device_forward = device_forward
            .unwrap_or_else(|error| panic!("adaptive perception cube forward failed: {error}"));
        let coarse_exposure = (device_forward.coarse_density
            / device_forward.density.clone().clamp_min(f32::MIN_POSITIVE))
        .clamp(0.0, 1.0);
        (
            device_forward.features.into_primitive().tensor(),
            coarse_exposure,
            AdaptivePerceptionPreparedState {
                density: device_forward.density,
                raw_state_gradient: device_forward.raw_state_gradient,
                state_gradient_inverse: device_forward.state_gradient_inverse,
            },
        )
    } else {
        let (features, coarse_exposure, prepared) = adaptive_reference_forward(
            (
                &positions_inner,
                &states_inner,
                &measure_inner,
                &bandwidth_inner,
            ),
            state_shape,
            config,
            options,
            semantics,
        );
        (
            features.into_primitive().tensor(),
            coarse_exposure,
            prepared,
        )
    };
    #[cfg(not(any(feature = "backend_wgpu", feature = "backend_cuda")))]
    let (output, coarse_exposure, prepared) = {
        let (features, coarse_exposure, prepared) = adaptive_reference_forward(
            (
                &positions_inner,
                &states_inner,
                &measure_inner,
                &bandwidth_inner,
            ),
            state_shape,
            config,
            options,
            semantics,
        );
        (
            features.into_primitive().tensor(),
            coarse_exposure,
            prepared,
        )
    };
    let state = AdaptivePerceptionAdjointState {
        positions: positions_inner,
        states: states_inner,
        represented_measure: measure_inner,
        bandwidth: bandwidth_inner,
        prepared,
        batch_size: state_shape[0],
        particle_count: state_shape[1],
        state_dims: state_shape[2],
        config,
        options,
        semantics,
    };
    let prep = AdaptivePerceptionAdjointOp
        .prepare::<NoCheckpointing>([states_primitive.node])
        .compute_bound();
    let output = match prep.stateful() {
        OpsKind::Tracked(prep) => prep.finish(state, output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    };
    AdaptivePerceptionBatchOutput {
        features: Tensor::<BurnBackend, 3>::from_primitive(TensorPrimitive::Float(output)),
        coarse_exposure: Tensor::<BurnBackend, 2>::from_inner(coarse_exposure),
    }
}

type AdaptiveReferenceTensors<'a> = (
    &'a Tensor3Inner,
    &'a Tensor3Inner,
    &'a Tensor2Inner,
    &'a Tensor2Inner,
);

fn adaptive_reference_forward(
    tensors: AdaptiveReferenceTensors<'_>,
    state_shape: [usize; 3],
    config: burn_automata_kernels::AdaptivePerceptionConfig,
    options: burn_automata_kernels::AdaptiveNpaPerceptionOptions,
    semantics: burn_automata_kernels::AdaptivePerceptionSemantics,
) -> (
    Tensor3Inner,
    Tensor2Inner,
    AdaptivePerceptionPreparedState,
) {
    let (positions, states, represented_measure, bandwidth) = tensors;
    let device = states.device();
    let positions_values = adaptive_reference_positions(positions);
    let state_values = tensor_values3(states, "adaptive states");
    let measure_values = tensor_values2(represented_measure, "adaptive represented measure");
    let bandwidth_values = tensor_values2(bandwidth, "adaptive bandwidth");
    let output = match semantics {
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible => {
            burn_automata_kernels::adaptive_npa_perceive_all_pairs(
                &positions_values,
                &state_values,
                &measure_values,
                &bandwidth_values,
                state_shape[0],
                state_shape[1],
                state_shape[2],
                config,
                options,
            )
        }
        burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive => {
            burn_automata_kernels::adaptive_perceive_all_pairs(
                &positions_values,
                &state_values,
                &measure_values,
                &bandwidth_values,
                state_shape[0],
                state_shape[1],
                state_shape[2],
                config,
            )
        }
    }
    .unwrap_or_else(|error| panic!("adaptive perception reference forward failed: {error}"));
    let features = Tensor::<InnerBackend, 3>::from_data(
        TensorData::new(
            output.features,
            [
                state_shape[0],
                state_shape[1],
                config.feature_dims(state_shape[2]),
            ],
        ),
        &device,
    );
    let coarse_exposure = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(output.coarse_exposure, [state_shape[0], state_shape[1]]),
        &device,
    );
    // Fallback execution does not consume prepared tensors in backward, but
    // state shapes stay valid so this path remains inspectable.
    let density = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(output.partition, [state_shape[0], state_shape[1]]),
        &device,
    );
    let raw_state_gradient =
        Tensor::<InnerBackend, 4>::zeros([state_shape[0], state_shape[1], state_shape[2], 2], &device);
    let state_gradient_inverse =
        Tensor::<InnerBackend, 3>::zeros([state_shape[0], state_shape[1], 4], &device);
    (
        features,
        coarse_exposure,
        AdaptivePerceptionPreparedState {
            density,
            raw_state_gradient,
            state_gradient_inverse,
        },
    )
}

fn adaptive_reference_positions(positions: &Tensor3Inner) -> Vec<[f32; 4]> {
    tensor_values3(positions, "adaptive positions")
        .chunks_exact(2)
        .map(|position| [position[0], position[1], 0.0, 0.0])
        .collect()
}

fn tensor_values2(tensor: &Tensor2Inner, label: &str) -> Vec<f32> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("{label} readback failed: {error}"))
}

fn tensor_values3(tensor: &Tensor3Inner, label: &str) -> Vec<f32> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("{label} readback failed: {error}"))
}
