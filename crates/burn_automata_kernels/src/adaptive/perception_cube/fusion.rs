use std::marker::PhantomData;

use burn::tensor::{
    Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend as BurnBackendTrait,
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_fusion::{
    Fusion,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

use super::{
    AdaptiveNpaPerceptionCubeAdjointOutput, AdaptiveNpaPerceptionCubeForwardOutput,
    kernels::{launch_forward, launch_state_adjoint},
};
use crate::{
    AdaptiveGraphPolicy, AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig, KernelError,
    KernelResult, adaptive::AdaptivePerceptionSemantics,
};

const FORWARD_OP: &str = "burn_automata.adaptive_npa.perception.forward.v2";
const STATE_ADJOINT_OP: &str = "burn_automata.adaptive_npa.perception.state_adjoint.v2";

type FusionBackend<R, F, I, BT> = Fusion<CubeBackend<R, F, I, BT>>;

pub(super) fn forward<R, F, I, BT>(
    positions: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    states: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    represented_measure: BurnTensor<FusionBackend<R, F, I, BT>, 2>,
    bandwidth: BurnTensor<FusionBackend<R, F, I, BT>, 2>,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
) -> KernelResult<AdaptiveNpaPerceptionCubeForwardOutput<FusionBackend<R, F, I, BT>>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let (batches, particles, state_dims) = validate_forward_shapes(
        &positions,
        &states,
        &represented_measure,
        &bandwidth,
        config,
        options,
        semantics,
    )?;
    let positions = positions.into_primitive().tensor();
    let states = states.into_primitive().tensor();
    let represented_measure = represented_measure.into_primitive().tensor();
    let bandwidth = bandwidth.into_primitive().tensor();
    let client = positions.client.clone();
    let dtype = positions.dtype;
    let features = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles, config.feature_dims(state_dims)]),
        dtype,
    );
    let density = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles]),
        dtype,
    );
    let coarse_density = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles]),
        dtype,
    );
    let raw_state_gradient = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles, state_dims, 2]),
        dtype,
    );
    let state_gradient_inverse = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles, 4]),
        dtype,
    );
    let inputs = [
        positions.clone().into_ir(),
        states.clone().into_ir(),
        represented_measure.clone().into_ir(),
        bandwidth.clone().into_ir(),
    ];
    let outputs = [
        features.clone(),
        density.clone(),
        coarse_density.clone(),
        raw_state_gradient.clone(),
        state_gradient_inverse.clone(),
    ];
    let streams =
        OperationStreams::with_inputs([&positions, &states, &represented_measure, &bandwidth]);
    let op = ForwardFusionOp::<R, F, I, BT> {
        desc: ForwardDesc {
            positions: inputs[0].clone(),
            states: inputs[1].clone(),
            represented_measure: inputs[2].clone(),
            bandwidth: inputs[3].clone(),
            features,
            density,
            coarse_density,
            raw_state_gradient,
            state_gradient_inverse,
        },
        config,
        options,
        semantics,
        _marker: PhantomData,
    };
    let [
        features,
        density,
        coarse_density,
        raw_state_gradient,
        state_gradient_inverse,
    ] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(FORWARD_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<5>();
    Ok(AdaptiveNpaPerceptionCubeForwardOutput {
        features: float_tensor(features),
        density: float_tensor(density),
        coarse_density: float_tensor(coarse_density),
        raw_state_gradient: float_tensor(raw_state_gradient),
        state_gradient_inverse: float_tensor(state_gradient_inverse),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn state_adjoint<R, F, I, BT>(
    positions: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    states: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    represented_measure: BurnTensor<FusionBackend<R, F, I, BT>, 2>,
    bandwidth: BurnTensor<FusionBackend<R, F, I, BT>, 2>,
    feature_grad: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    density: BurnTensor<FusionBackend<R, F, I, BT>, 2>,
    raw_state_gradient: BurnTensor<FusionBackend<R, F, I, BT>, 4>,
    state_gradient_inverse: BurnTensor<FusionBackend<R, F, I, BT>, 3>,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
) -> KernelResult<AdaptiveNpaPerceptionCubeAdjointOutput<FusionBackend<R, F, I, BT>>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let (batches, particles, state_dims) = validate_forward_shapes(
        &positions,
        &states,
        &represented_measure,
        &bandwidth,
        config,
        options,
        semantics,
    )?;
    let feature_dims = feature_grad.shape().dims::<3>();
    let density_dims = density.shape().dims::<2>();
    let raw_dims = raw_state_gradient.shape().dims::<4>();
    let inverse_dims = state_gradient_inverse.shape().dims::<3>();
    if feature_dims != [batches, particles, config.feature_dims(state_dims)]
        || density_dims != [batches, particles]
        || raw_dims != [batches, particles, state_dims, 2]
        || inverse_dims != [batches, particles, 4]
    {
        return Err(KernelError::InvalidArgument(format!(
            "adaptive perception prepared adjoint shape mismatch: feature_grad={feature_dims:?}, density={density_dims:?}, raw={raw_dims:?}, inverse={inverse_dims:?}",
        )));
    }

    let positions = positions.into_primitive().tensor();
    let states = states.into_primitive().tensor();
    let represented_measure = represented_measure.into_primitive().tensor();
    let bandwidth = bandwidth.into_primitive().tensor();
    let feature_grad = feature_grad.into_primitive().tensor();
    let density = density.into_primitive().tensor();
    let raw_state_gradient = raw_state_gradient.into_primitive().tensor();
    let state_gradient_inverse = state_gradient_inverse.into_primitive().tensor();
    let client = positions.client.clone();
    let dtype = positions.dtype;
    let state_grad = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particles, state_dims]),
        dtype,
    );
    let inputs = [
        positions.clone().into_ir(),
        states.clone().into_ir(),
        represented_measure.clone().into_ir(),
        bandwidth.clone().into_ir(),
        feature_grad.clone().into_ir(),
        density.clone().into_ir(),
        raw_state_gradient.clone().into_ir(),
        state_gradient_inverse.clone().into_ir(),
    ];
    let outputs = [state_grad.clone()];
    let streams = OperationStreams::with_inputs([
        &positions,
        &states,
        &represented_measure,
        &bandwidth,
        &feature_grad,
        &density,
        &raw_state_gradient,
        &state_gradient_inverse,
    ]);
    let op = StateAdjointFusionOp::<R, F, I, BT> {
        desc: StateAdjointDesc {
            positions: inputs[0].clone(),
            states: inputs[1].clone(),
            represented_measure: inputs[2].clone(),
            bandwidth: inputs[3].clone(),
            feature_grad: inputs[4].clone(),
            density: inputs[5].clone(),
            raw_state_gradient: inputs[6].clone(),
            state_gradient_inverse: inputs[7].clone(),
            state_grad,
        },
        config,
        options,
        semantics,
        _marker: PhantomData,
    };
    let [state_grad] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(STATE_ADJOINT_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<1>();
    Ok(AdaptiveNpaPerceptionCubeAdjointOutput {
        state_grad: float_tensor(state_grad),
    })
}

fn validate_forward_shapes<B: BurnBackendTrait>(
    positions: &BurnTensor<B, 3>,
    states: &BurnTensor<B, 3>,
    represented_measure: &BurnTensor<B, 2>,
    bandwidth: &BurnTensor<B, 2>,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    _semantics: AdaptivePerceptionSemantics,
) -> KernelResult<(usize, usize, usize)> {
    config.validate()?;
    options.validate()?;
    if config.dim != 2 || config.graph_policy != AdaptiveGraphPolicy::RawSupport {
        return Err(KernelError::InvalidArgument(
            "adaptive perception cube v2 requires 2D raw-support graph semantics".to_string(),
        ));
    }
    if config.include_position_features != options.position_features {
        return Err(KernelError::InvalidArgument(
            "adaptive perception position-feature options disagree".to_string(),
        ));
    }
    let position_dims = positions.shape().dims::<3>();
    let state_dims = states.shape().dims::<3>();
    let measure_dims = represented_measure.shape().dims::<2>();
    let bandwidth_dims = bandwidth.shape().dims::<2>();
    if position_dims[0] == 0
        || position_dims[1] == 0
        || position_dims[2] != 2
        || state_dims[0] != position_dims[0]
        || state_dims[1] != position_dims[1]
        || state_dims[2] == 0
        || measure_dims != [position_dims[0], position_dims[1]]
        || bandwidth_dims != measure_dims
    {
        return Err(KernelError::InvalidArgument(format!(
            "adaptive perception cube expects positions [B,N,2], states [B,N,C], and material [B,N], got {position_dims:?}, {state_dims:?}, {measure_dims:?}, {bandwidth_dims:?}",
        )));
    }
    Ok((position_dims[0], position_dims[1], state_dims[2]))
}

fn float_tensor<B: BurnBackendTrait, const D: usize>(
    primitive: B::FloatTensorPrimitive,
) -> BurnTensor<B, D> {
    BurnTensor::from_primitive(TensorPrimitive::Float(primitive))
}

#[derive(Clone, Debug)]
struct ForwardDesc {
    positions: TensorIr,
    states: TensorIr,
    represented_measure: TensorIr,
    bandwidth: TensorIr,
    features: TensorIr,
    density: TensorIr,
    coarse_density: TensorIr,
    raw_state_gradient: TensorIr,
    state_gradient_inverse: TensorIr,
}

#[derive(Debug)]
struct ForwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: ForwardDesc,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for ForwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<
            <burn_cubecl::fusion::FusionCubeRuntime<R> as burn_fusion::FusionRuntime>::FusionHandle,
        >,
    ) {
        type Raw<R, F, I, BT> = CubeBackend<R, F, I, BT>;
        let positions = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.positions);
        let states = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.states);
        let represented_measure =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.represented_measure);
        let bandwidth = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.bandwidth);
        let output = launch_forward(
            positions,
            states,
            represented_measure,
            bandwidth,
            self.config,
            self.options,
            self.semantics,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.features.id, output.features);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.density.id, output.density);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.coarse_density.id,
            output.coarse_density,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.raw_state_gradient.id,
            output.raw_state_gradient,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.state_gradient_inverse.id,
            output.state_gradient_inverse,
        );
    }
}

#[derive(Clone, Debug)]
struct StateAdjointDesc {
    positions: TensorIr,
    states: TensorIr,
    represented_measure: TensorIr,
    bandwidth: TensorIr,
    feature_grad: TensorIr,
    density: TensorIr,
    raw_state_gradient: TensorIr,
    state_gradient_inverse: TensorIr,
    state_grad: TensorIr,
}

#[derive(Debug)]
struct StateAdjointFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: StateAdjointDesc,
    config: AdaptivePerceptionConfig,
    options: AdaptiveNpaPerceptionOptions,
    semantics: AdaptivePerceptionSemantics,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for StateAdjointFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<
            <burn_cubecl::fusion::FusionCubeRuntime<R> as burn_fusion::FusionRuntime>::FusionHandle,
        >,
    ) {
        type Raw<R, F, I, BT> = CubeBackend<R, F, I, BT>;
        let positions = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.positions);
        let states = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.states);
        let represented_measure =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.represented_measure);
        let bandwidth = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.bandwidth);
        let feature_grad = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.feature_grad);
        let density = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.density);
        let raw_state_gradient =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.raw_state_gradient);
        let state_gradient_inverse =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_gradient_inverse);
        let state_grad = launch_state_adjoint(
            positions,
            states,
            represented_measure,
            bandwidth,
            feature_grad,
            density,
            raw_state_gradient,
            state_gradient_inverse,
            self.config,
            self.options,
            self.semantics,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_grad.id, state_grad);
    }
}
