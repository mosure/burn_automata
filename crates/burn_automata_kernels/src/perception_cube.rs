use std::marker::PhantomData;

use burn::tensor::{
    DType, Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
    backend::Backend as BurnBackendTrait,
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_fusion::{
    Fusion,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::{CubeDim, CubeLaunch, calculate_cube_count_elemwise, cube, prelude::*};

use crate::{KernelError, KernelResult};

const PERCEPTION_ADJOINT_OP: &str = "burn_automata.perception.adjoint.v1";
const PERCEPTION_FORWARD_OP: &str = "burn_automata.perception.forward.v1";
const LOG_NORMALIZE_EPSILON: f32 = 1.0e-6;
const MAX_SPARSE_PLANES_PER_CUBE: u32 = 4;

type FusionCubeBackend<R, F, I, BT> = Fusion<CubeBackend<R, F, I, BT>>;
type PerceptionForwardFusionOutput<R, F, I, BT> =
    PerceptionCubeForwardOutput<FusionCubeBackend<R, F, I, BT>>;
type PerceptionPreparedForwardFusionOutput<R, F, I, BT> =
    PerceptionCubePreparedForwardOutput<FusionCubeBackend<R, F, I, BT>>;
type PerceptionAdjointFusionOutput<R, F, I, BT> =
    PerceptionCubeAdjointOutput<FusionCubeBackend<R, F, I, BT>>;

#[derive(Clone, Copy, Debug)]
pub struct PerceptionCubeAdjointConfig {
    pub eps: f32,
    pub eps0: f32,
    pub state_grad: bool,
    pub density_grad: bool,
    pub scale_equivariance: bool,
    pub particle_density_equivariance: bool,
    pub log_norm_grad: bool,
    pub log_norm_density_grad: bool,
    pub hybrid_state_gradient: bool,
    pub position_features: bool,
    pub compute_position_grad: bool,
    pub compute_state_grad: bool,
    pub grid_width: u32,
    pub grid_height: u32,
    pub sparse_grid_min_particles: u32,
}

pub struct PerceptionCubeAdjointOutput<B: BurnBackendTrait> {
    pub position_grad: BurnTensor<B, 3>,
    pub state_grad: BurnTensor<B, 3>,
}

pub struct PerceptionCubeForwardOutput<B: BurnBackendTrait> {
    pub features: BurnTensor<B, 3>,
}

pub struct PerceptionCubePreparedForwardOutput<B: BurnBackendTrait> {
    pub features: BurnTensor<B, 3>,
    pub density: BurnTensor<B, 2>,
    pub offsets: BurnTensor<B, 2, burn::tensor::Int>,
    pub permutation: BurnTensor<B, 2, burn::tensor::Int>,
    pub raw_state_gradient: BurnTensor<B, 4>,
    pub state_gradient_inverse: BurnTensor<B, 3>,
}

#[allow(unused_variables)]
pub trait PerceptionCubeForwardBackend: BurnBackendTrait + Sized {
    fn perception_cube_forward(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeForwardOutput<Self>>> {
        None
    }
}

#[allow(unused_variables)]
pub trait PerceptionCubeAdjointBackend: BurnBackendTrait + Sized {
    fn perception_cube_adjoint(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        None
    }
}

#[allow(unused_variables)]
pub trait PerceptionCubePreparedBackend: BurnBackendTrait + Sized {
    fn perception_cube_forward_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubePreparedForwardOutput<Self>>> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn perception_cube_adjoint_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        offsets: BurnTensor<Self, 2, burn::tensor::Int>,
        permutation: BurnTensor<Self, 2, burn::tensor::Int>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        None
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl PerceptionCubeForwardBackend for burn::backend::Wgpu<f32> {
    fn perception_cube_forward(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeForwardOutput<Self>>> {
        Some(perception_cube_forward_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(x, s, cfg))
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl PerceptionCubeAdjointBackend for burn::backend::Wgpu<f32> {
    fn perception_cube_adjoint(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        Some(perception_cube_adjoint_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(x, s, feature_grad, cfg))
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl PerceptionCubePreparedBackend for burn::backend::Wgpu<f32> {
    fn perception_cube_forward_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubePreparedForwardOutput<Self>>> {
        Some(perception_cube_forward_prepared_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(x, s, cfg))
    }

    fn perception_cube_adjoint_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        offsets: BurnTensor<Self, 2, burn::tensor::Int>,
        permutation: BurnTensor<Self, 2, burn::tensor::Int>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        Some(perception_cube_adjoint_prepared_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(
            x,
            s,
            feature_grad,
            density,
            offsets,
            permutation,
            raw_state_gradient,
            state_gradient_inverse,
            cfg,
        ))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl PerceptionCubeForwardBackend for burn::backend::Cuda<f32> {
    fn perception_cube_forward(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeForwardOutput<Self>>> {
        Some(perception_cube_forward_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(x, s, cfg))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl PerceptionCubeAdjointBackend for burn::backend::Cuda<f32> {
    fn perception_cube_adjoint(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        Some(perception_cube_adjoint_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(x, s, feature_grad, cfg))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl PerceptionCubePreparedBackend for burn::backend::Cuda<f32> {
    fn perception_cube_forward_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubePreparedForwardOutput<Self>>> {
        Some(perception_cube_forward_prepared_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(x, s, cfg))
    }

    fn perception_cube_adjoint_prepared(
        x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        offsets: BurnTensor<Self, 2, burn::tensor::Int>,
        permutation: BurnTensor<Self, 2, burn::tensor::Int>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        cfg: PerceptionCubeAdjointConfig,
    ) -> Option<KernelResult<PerceptionCubeAdjointOutput<Self>>> {
        Some(perception_cube_adjoint_prepared_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(
            x,
            s,
            feature_grad,
            density,
            offsets,
            permutation,
            raw_state_gradient,
            state_gradient_inverse,
            cfg,
        ))
    }
}

fn perception_feature_dims(state_dims: usize, cfg: PerceptionCubeAdjointConfig) -> usize {
    state_dims * 2
        + usize::from(cfg.state_grad) * state_dims * 2
        + usize::from(cfg.density_grad) * 2
        + usize::from(cfg.position_features) * 2
}

fn perception_cube_forward_fusion<R, F, I, BT>(
    x: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    s: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    cfg: PerceptionCubeAdjointConfig,
) -> KernelResult<PerceptionForwardFusionOutput<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let x_dims = x.shape().dims::<3>();
    let s_dims = s.shape().dims::<3>();
    let batches = x_dims[0];
    let particle_count = x_dims[1];
    let state_dims = s_dims[2];
    if x_dims[2] != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "perception cube forward expects x shape [batch, particles, 2], got {x_dims:?}",
        )));
    }
    if s_dims[0] != batches || s_dims[1] != particle_count {
        return Err(KernelError::InvalidArgument(format!(
            "perception cube forward tensor shape mismatch: x={x_dims:?} s={s_dims:?}",
        )));
    }
    if !cfg.eps.is_finite() || cfg.eps <= 0.0 || !cfg.eps0.is_finite() || cfg.eps0 <= 0.0 {
        return Err(KernelError::InvalidArgument(
            "perception cube forward requires positive finite eps and eps0".to_string(),
        ));
    }
    if cfg.grid_width == 0 || cfg.grid_height == 0 {
        return Err(KernelError::InvalidArgument(
            "perception cube forward requires a non-empty 2D grid".to_string(),
        ));
    }

    let x_fusion = x.into_primitive().tensor();
    let s_fusion = s.into_primitive().tensor();
    let client = x_fusion.client.clone();
    let dtype = x_fusion.dtype;
    let features_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([
            batches,
            particle_count,
            perception_feature_dims(state_dims, cfg),
        ]),
        dtype,
    );
    let inputs = [x_fusion.clone().into_ir(), s_fusion.clone().into_ir()];
    let outputs = [features_ir.clone()];
    let streams = OperationStreams::with_inputs([&x_fusion, &s_fusion]);
    let op = PerceptionForwardFusionOp::<R, F, I, BT> {
        desc: PerceptionForwardDesc {
            x: inputs[0].clone(),
            s: inputs[1].clone(),
            features: features_ir,
        },
        cfg,
        _marker: PhantomData,
    };
    let [features_fusion] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(PERCEPTION_FORWARD_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<1>();

    Ok(PerceptionCubeForwardOutput {
        features: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(features_fusion),
        ),
    })
}

fn perception_cube_forward_prepared_fusion<R, F, I, BT>(
    x: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    s: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    cfg: PerceptionCubeAdjointConfig,
) -> KernelResult<PerceptionPreparedForwardFusionOutput<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let x_dims = x.shape().dims::<3>();
    let s_dims = s.shape().dims::<3>();
    let batches = x_dims[0];
    let particle_count = x_dims[1];
    let state_dims = s_dims[2];
    if x_dims[2] != 2 || s_dims[0] != batches || s_dims[1] != particle_count {
        return Err(KernelError::InvalidArgument(format!(
            "prepared perception forward tensor shape mismatch: x={x_dims:?} s={s_dims:?}",
        )));
    }
    if !should_use_sparse_grid(particle_count, cfg) || cfg.compute_position_grad {
        return Err(KernelError::InvalidArgument(
            "prepared perception forward requires sparse-grid state-only gradients".to_string(),
        ));
    }

    let x_fusion = x.into_primitive().tensor();
    let s_fusion = s.into_primitive().tensor();
    let client = x_fusion.client.clone();
    let dtype = x_fusion.dtype;
    let cell_count = cfg.grid_width as usize * cfg.grid_height as usize;
    let features_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([
            batches,
            particle_count,
            perception_feature_dims(state_dims, cfg),
        ]),
        dtype,
    );
    let density_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count]),
        dtype,
    );
    let offsets_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, cell_count + 1]),
        DType::U32,
    );
    let permutation_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count]),
        DType::U32,
    );
    let raw_state_gradient_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, state_dims, 2]),
        dtype,
    );
    let state_gradient_inverse_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, 4]),
        dtype,
    );
    let inputs = [x_fusion.clone().into_ir(), s_fusion.clone().into_ir()];
    let outputs = [
        features_ir.clone(),
        density_ir.clone(),
        offsets_ir.clone(),
        permutation_ir.clone(),
        raw_state_gradient_ir.clone(),
        state_gradient_inverse_ir.clone(),
    ];
    let streams = OperationStreams::with_inputs([&x_fusion, &s_fusion]);
    let op = PerceptionPreparedForwardFusionOp::<R, F, I, BT> {
        desc: PerceptionPreparedForwardDesc {
            x: inputs[0].clone(),
            s: inputs[1].clone(),
            features: features_ir,
            density: density_ir,
            offsets: offsets_ir,
            permutation: permutation_ir,
            raw_state_gradient: raw_state_gradient_ir,
            state_gradient_inverse: state_gradient_inverse_ir,
        },
        cfg,
        _marker: PhantomData,
    };
    let [
        features,
        density,
        offsets,
        permutation,
        raw_state_gradient,
        state_gradient_inverse,
    ] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(
                "burn_automata.perception.forward_prepared.v1",
                &inputs,
                &outputs,
            )),
            op,
        )
        .outputs::<6>();

    Ok(PerceptionCubePreparedForwardOutput {
        features: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(features),
        ),
        density: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 2>::from_primitive(
            TensorPrimitive::Float(density),
        ),
        offsets: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 2, burn::tensor::Int>::from_primitive(
            offsets,
        ),
        permutation:
            BurnTensor::<FusionCubeBackend<R, F, I, BT>, 2, burn::tensor::Int>::from_primitive(
                permutation,
            ),
        raw_state_gradient: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 4>::from_primitive(
            TensorPrimitive::Float(raw_state_gradient),
        ),
        state_gradient_inverse: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(state_gradient_inverse),
        ),
    })
}

fn perception_cube_adjoint_fusion<R, F, I, BT>(
    x: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    s: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    feature_grad: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    cfg: PerceptionCubeAdjointConfig,
) -> KernelResult<PerceptionAdjointFusionOutput<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let x_dims = x.shape().dims::<3>();
    let s_dims = s.shape().dims::<3>();
    let feature_dims = feature_grad.shape().dims::<3>();
    let batches = x_dims[0];
    let particle_count = x_dims[1];
    let state_dims = s_dims[2];
    if x_dims[2] != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "perception cube adjoint expects x shape [batch, particles, 2], got {x_dims:?}",
        )));
    }
    if s_dims[0] != batches || s_dims[1] != particle_count {
        return Err(KernelError::InvalidArgument(format!(
            "perception cube adjoint tensor shape mismatch: x={x_dims:?} s={s_dims:?}",
        )));
    }
    let expected_feature_dims = perception_feature_dims(state_dims, cfg);
    if feature_dims != [batches, particle_count, expected_feature_dims] {
        return Err(KernelError::InvalidArgument(format!(
            "perception cube adjoint feature grad shape mismatch: expected [{batches}, {particle_count}, {expected_feature_dims}], got {feature_dims:?}",
        )));
    }
    if !cfg.eps.is_finite() || cfg.eps <= 0.0 || !cfg.eps0.is_finite() || cfg.eps0 <= 0.0 {
        return Err(KernelError::InvalidArgument(
            "perception cube adjoint requires positive finite eps and eps0".to_string(),
        ));
    }
    if cfg.grid_width == 0 || cfg.grid_height == 0 {
        return Err(KernelError::InvalidArgument(
            "perception cube adjoint requires a non-empty 2D grid".to_string(),
        ));
    }

    let x_fusion = x.into_primitive().tensor();
    let s_fusion = s.into_primitive().tensor();
    let feature_grad_fusion = feature_grad.into_primitive().tensor();
    let client = x_fusion.client.clone();
    let dtype = x_fusion.dtype;
    let position_grad_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, 2]),
        dtype,
    );
    let state_grad_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, state_dims]),
        dtype,
    );
    let inputs = [
        x_fusion.clone().into_ir(),
        s_fusion.clone().into_ir(),
        feature_grad_fusion.clone().into_ir(),
    ];
    let outputs = [position_grad_ir.clone(), state_grad_ir.clone()];
    let streams = OperationStreams::with_inputs([&x_fusion, &s_fusion, &feature_grad_fusion]);
    let op = PerceptionAdjointFusionOp::<R, F, I, BT> {
        desc: PerceptionAdjointDesc {
            x: inputs[0].clone(),
            s: inputs[1].clone(),
            feature_grad: inputs[2].clone(),
            position_grad: position_grad_ir,
            state_grad: state_grad_ir,
        },
        cfg,
        _marker: PhantomData,
    };
    let [position_grad_fusion, state_grad_fusion] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(PERCEPTION_ADJOINT_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<2>();

    Ok(PerceptionCubeAdjointOutput {
        position_grad: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(position_grad_fusion),
        ),
        state_grad: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(state_grad_fusion),
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn perception_cube_adjoint_prepared_fusion<R, F, I, BT>(
    x: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    s: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    feature_grad: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    density: BurnTensor<FusionCubeBackend<R, F, I, BT>, 2>,
    offsets: BurnTensor<FusionCubeBackend<R, F, I, BT>, 2, burn::tensor::Int>,
    permutation: BurnTensor<FusionCubeBackend<R, F, I, BT>, 2, burn::tensor::Int>,
    raw_state_gradient: BurnTensor<FusionCubeBackend<R, F, I, BT>, 4>,
    state_gradient_inverse: BurnTensor<FusionCubeBackend<R, F, I, BT>, 3>,
    cfg: PerceptionCubeAdjointConfig,
) -> KernelResult<PerceptionAdjointFusionOutput<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let x_dims = x.shape().dims::<3>();
    let s_dims = s.shape().dims::<3>();
    let feature_dims = feature_grad.shape().dims::<3>();
    let density_dims = density.shape().dims::<2>();
    let offset_dims = offsets.shape().dims::<2>();
    let permutation_dims = permutation.shape().dims::<2>();
    let raw_state_gradient_dims = raw_state_gradient.shape().dims::<4>();
    let state_gradient_inverse_dims = state_gradient_inverse.shape().dims::<3>();
    let batches = x_dims[0];
    let particle_count = x_dims[1];
    let state_dims = s_dims[2];
    let cell_count = cfg.grid_width as usize * cfg.grid_height as usize;
    let expected_feature_dims = perception_feature_dims(state_dims, cfg);
    if x_dims[2] != 2
        || s_dims != [batches, particle_count, state_dims]
        || feature_dims != [batches, particle_count, expected_feature_dims]
        || density_dims != [batches, particle_count]
        || offset_dims != [batches, cell_count + 1]
        || permutation_dims != [batches, particle_count]
        || raw_state_gradient_dims != [batches, particle_count, state_dims, 2]
        || state_gradient_inverse_dims != [batches, particle_count, 4]
    {
        return Err(KernelError::InvalidArgument(format!(
            "prepared perception adjoint tensor shape mismatch: x={x_dims:?} s={s_dims:?} feature_grad={feature_dims:?} density={density_dims:?} offsets={offset_dims:?} permutation={permutation_dims:?} raw_state_gradient={raw_state_gradient_dims:?} state_gradient_inverse={state_gradient_inverse_dims:?}",
        )));
    }
    if !should_use_sparse_grid(particle_count, cfg) || cfg.compute_position_grad {
        return Err(KernelError::InvalidArgument(
            "prepared perception adjoint requires sparse-grid state-only gradients".to_string(),
        ));
    }

    let x_fusion = x.into_primitive().tensor();
    let s_fusion = s.into_primitive().tensor();
    let feature_grad_fusion = feature_grad.into_primitive().tensor();
    let density_fusion = density.into_primitive().tensor();
    let offsets_fusion = offsets.into_primitive();
    let permutation_fusion = permutation.into_primitive();
    let raw_state_gradient_fusion = raw_state_gradient.into_primitive().tensor();
    let state_gradient_inverse_fusion = state_gradient_inverse.into_primitive().tensor();
    let client = x_fusion.client.clone();
    let dtype = x_fusion.dtype;
    let position_grad_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, 2]),
        dtype,
    );
    let state_grad_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batches, particle_count, state_dims]),
        dtype,
    );
    let inputs = [
        x_fusion.clone().into_ir(),
        s_fusion.clone().into_ir(),
        feature_grad_fusion.clone().into_ir(),
        density_fusion.clone().into_ir(),
        offsets_fusion.clone().into_ir(),
        permutation_fusion.clone().into_ir(),
        raw_state_gradient_fusion.clone().into_ir(),
        state_gradient_inverse_fusion.clone().into_ir(),
    ];
    let outputs = [position_grad_ir.clone(), state_grad_ir.clone()];
    let streams = OperationStreams::with_inputs([
        &x_fusion,
        &s_fusion,
        &feature_grad_fusion,
        &density_fusion,
        &offsets_fusion,
        &permutation_fusion,
        &raw_state_gradient_fusion,
        &state_gradient_inverse_fusion,
    ]);
    let op = PerceptionPreparedAdjointFusionOp::<R, F, I, BT> {
        desc: PerceptionPreparedAdjointDesc {
            x: inputs[0].clone(),
            s: inputs[1].clone(),
            feature_grad: inputs[2].clone(),
            density: inputs[3].clone(),
            offsets: inputs[4].clone(),
            permutation: inputs[5].clone(),
            raw_state_gradient: inputs[6].clone(),
            state_gradient_inverse: inputs[7].clone(),
            position_grad: position_grad_ir,
            state_grad: state_grad_ir,
        },
        cfg,
        _marker: PhantomData,
    };
    let [position_grad, state_grad] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(
                "burn_automata.perception.adjoint_prepared.v1",
                &inputs,
                &outputs,
            )),
            op,
        )
        .outputs::<2>();

    Ok(PerceptionCubeAdjointOutput {
        position_grad: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(position_grad),
        ),
        state_grad: BurnTensor::<FusionCubeBackend<R, F, I, BT>, 3>::from_primitive(
            TensorPrimitive::Float(state_grad),
        ),
    })
}

#[derive(Clone, Debug)]
struct PerceptionForwardDesc {
    x: TensorIr,
    s: TensorIr,
    features: TensorIr,
}

#[derive(Debug)]
struct PerceptionForwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: PerceptionForwardDesc,
    cfg: PerceptionCubeAdjointConfig,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for PerceptionForwardFusionOp<R, F, I, BT>
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
        let x = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.x);
        let s = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.s);
        let features = launch_perception_forward(x, s, self.cfg);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.features.id, features);
    }
}

#[derive(Clone, Debug)]
struct PerceptionPreparedForwardDesc {
    x: TensorIr,
    s: TensorIr,
    features: TensorIr,
    density: TensorIr,
    offsets: TensorIr,
    permutation: TensorIr,
    raw_state_gradient: TensorIr,
    state_gradient_inverse: TensorIr,
}

#[derive(Debug)]
struct PerceptionPreparedForwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: PerceptionPreparedForwardDesc,
    cfg: PerceptionCubeAdjointConfig,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for PerceptionPreparedForwardFusionOp<R, F, I, BT>
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
        let x = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.x);
        let s = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.s);
        let output = launch_perception_forward_raw(x, s, self.cfg, true);
        let grid = output
            .grid
            .expect("prepared perception forward requires a sparse grid");
        let raw_state_gradient = output
            .raw_state_gradient
            .expect("prepared perception forward requires retained raw state gradients");
        let state_gradient_inverse = output
            .state_gradient_inverse
            .expect("prepared perception forward requires retained correction inverses");
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.features.id, output.features);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.density.id, output.density);
        handles.register_int_tensor::<Raw<R, F, I, BT>>(&self.desc.offsets.id, grid.offsets);
        handles
            .register_int_tensor::<Raw<R, F, I, BT>>(&self.desc.permutation.id, grid.permutation);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.raw_state_gradient.id,
            raw_state_gradient,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.state_gradient_inverse.id,
            state_gradient_inverse,
        );
    }
}

#[derive(Clone, Debug)]
struct PerceptionAdjointDesc {
    x: TensorIr,
    s: TensorIr,
    feature_grad: TensorIr,
    position_grad: TensorIr,
    state_grad: TensorIr,
}

#[derive(Debug)]
struct PerceptionAdjointFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: PerceptionAdjointDesc,
    cfg: PerceptionCubeAdjointConfig,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for PerceptionAdjointFusionOp<R, F, I, BT>
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
        let x = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.x);
        let s = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.s);
        let feature_grad = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.feature_grad);
        let output = launch_perception_adjoint(x, s, feature_grad, self.cfg);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.position_grad.id,
            output.position_grad,
        );
        handles
            .register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_grad.id, output.state_grad);
    }
}

#[derive(Clone, Debug)]
struct PerceptionPreparedAdjointDesc {
    x: TensorIr,
    s: TensorIr,
    feature_grad: TensorIr,
    density: TensorIr,
    offsets: TensorIr,
    permutation: TensorIr,
    raw_state_gradient: TensorIr,
    state_gradient_inverse: TensorIr,
    position_grad: TensorIr,
    state_grad: TensorIr,
}

#[derive(Debug)]
struct PerceptionPreparedAdjointFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: PerceptionPreparedAdjointDesc,
    cfg: PerceptionCubeAdjointConfig,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for PerceptionPreparedAdjointFusionOp<R, F, I, BT>
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
        let x = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.x);
        let s = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.s);
        let feature_grad = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.feature_grad);
        let density = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.density);
        let offsets = handles.get_int_tensor::<Raw<R, F, I, BT>>(&self.desc.offsets);
        let permutation = handles.get_int_tensor::<Raw<R, F, I, BT>>(&self.desc.permutation);
        let raw_state_gradient =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.raw_state_gradient);
        let state_gradient_inverse =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_gradient_inverse);
        let output = launch_perception_adjoint_prepared(
            x,
            s,
            feature_grad,
            density,
            PerceptionGridRaw {
                offsets,
                permutation,
            },
            raw_state_gradient,
            state_gradient_inverse,
            self.cfg,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.position_grad.id,
            output.position_grad,
        );
        handles
            .register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_grad.id, output.state_grad);
    }
}

struct PerceptionAdjointRawOutput<R: CubeRuntime> {
    position_grad: CubeTensor<R>,
    state_grad: CubeTensor<R>,
}

#[derive(Clone)]
struct PerceptionGridRaw<R: CubeRuntime> {
    offsets: CubeTensor<R>,
    permutation: CubeTensor<R>,
}

struct PerceptionPreparedAdjointRaw<R: CubeRuntime> {
    density: CubeTensor<R>,
    grid: PerceptionGridRaw<R>,
    raw_state_gradient: CubeTensor<R>,
    state_gradient_inverse: CubeTensor<R>,
}

struct PerceptionForwardRawOutput<R: CubeRuntime> {
    features: CubeTensor<R>,
    density: CubeTensor<R>,
    grid: Option<PerceptionGridRaw<R>>,
    raw_state_gradient: Option<CubeTensor<R>>,
    state_gradient_inverse: Option<CubeTensor<R>>,
}

fn should_use_sparse_grid(particle_count: usize, cfg: PerceptionCubeAdjointConfig) -> bool {
    cfg.sparse_grid_min_particles > 0
        && particle_count >= cfg.sparse_grid_min_particles as usize
        && cfg.grid_width > 0
        && cfg.grid_height > 0
}

fn perception_sparse_plane_dim<R: CubeRuntime>(
    client: &ComputeClient<R>,
    particle_count: usize,
    state_dims: usize,
) -> Option<u32> {
    let hardware = &client.properties().hardware;
    (particle_count >= 1024
        && hardware.plane_size_min == hardware.plane_size_max
        && hardware.plane_size_min >= state_dims as u32
        && hardware.plane_size_min <= hardware.max_units_per_cube)
        .then_some(hardware.plane_size_min)
}

fn perception_sparse_planes_per_cube(particle_count: usize) -> u32 {
    if particle_count >= 4096 {
        MAX_SPARSE_PLANES_PER_CUBE
    } else {
        1
    }
}

fn launch_perception_grid<R: CubeRuntime>(
    x: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
) -> PerceptionGridRaw<R> {
    let [batches, particle_count, _] = x.shape().dims::<3>();
    let cell_count = cfg.grid_width as usize * cfg.grid_height as usize;
    let client = x.client.clone();
    let device = x.device.clone();
    let counts = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, cell_count]),
        DType::U32,
    );
    let offsets = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, cell_count + 1]),
        DType::U32,
    );
    let permutation = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([batches, particle_count]),
        DType::U32,
    );
    let dtype = x.dtype;
    let cell_units = batches * cell_count;
    let cell_cube_dim = CubeDim::new(&client, cell_units);
    let cell_cube_count = calculate_cube_count_elemwise(&client, cell_units, cell_cube_dim);
    perception_grid_zero_kernel::launch(
        &client,
        cell_cube_count,
        cell_cube_dim,
        AddressType::U32,
        counts.clone().into_tensor_arg(),
    );
    let particle_units = batches * particle_count;
    let particle_cube_dim = CubeDim::new(&client, particle_units);
    let particle_cube_count =
        calculate_cube_count_elemwise(&client, particle_units, particle_cube_dim);
    perception_grid_count_kernel::launch(
        &client,
        particle_cube_count.clone(),
        particle_cube_dim,
        AddressType::U32,
        x.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        perception_args(cfg, dtype),
        dtype.into(),
    );

    let batch_cube_dim = CubeDim::new(&client, batches);
    let batch_cube_count = calculate_cube_count_elemwise(&client, batches, batch_cube_dim);
    perception_grid_scan_kernel::launch(
        &client,
        batch_cube_count,
        batch_cube_dim,
        AddressType::U32,
        counts.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
    );
    perception_grid_scatter_kernel::launch(
        &client,
        particle_cube_count,
        particle_cube_dim,
        AddressType::U32,
        x.into_tensor_arg(),
        counts.into_tensor_arg(),
        permutation.clone().into_tensor_arg(),
        perception_args(cfg, dtype),
        dtype.into(),
    );

    PerceptionGridRaw {
        offsets,
        permutation,
    }
}

fn launch_perception_forward<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
) -> CubeTensor<R> {
    launch_perception_forward_raw(x, s, cfg, false).features
}

fn launch_perception_forward_raw<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
    retain_adjoint_state: bool,
) -> PerceptionForwardRawOutput<R> {
    let dims = x.shape().dims::<3>();
    let batches = dims[0];
    let particle_count = dims[1];
    let state_dims = s.shape().dims::<3>()[2];
    let feature_dims = perception_feature_dims(state_dims, cfg);
    let dtype = x.dtype;
    let client = x.client.clone();
    let device = x.device.clone();
    let density = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count]),
        dtype,
    );
    let features = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, feature_dims]),
        dtype,
    );
    let raw_state_gradient = retain_adjoint_state.then(|| {
        empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([batches, particle_count, state_dims, 2]),
            dtype,
        )
    });
    let state_gradient_inverse = retain_adjoint_state.then(|| {
        empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([batches, particle_count, 4]),
            dtype,
        )
    });
    let sparse_grid =
        should_use_sparse_grid(particle_count, cfg).then(|| launch_perception_grid(x.clone(), cfg));
    let args_density = perception_args(cfg, dtype);
    if let Some(grid) = sparse_grid.as_ref() {
        let units = batches * particle_count;
        let cube_dim = CubeDim::new(&client, units);
        let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
        perception_density_sparse_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            grid.offsets.clone().into_tensor_arg(),
            grid.permutation.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    } else if particle_count >= 512 {
        let tile_size = 256usize;
        let query_blocks = particle_count.div_ceil(tile_size);
        perception_density_tiled_kernel::launch(
            &client,
            CubeCount::Static(query_blocks as u32, batches as u32, 1),
            CubeDim::new_1d(tile_size as u32),
            AddressType::U32,
            x.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    } else {
        let units = batches * particle_count;
        let cube_dim = CubeDim::new(&client, units);
        let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
        perception_density_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    }

    let args_forward = perception_args(cfg, dtype);
    let units = batches * particle_count;
    let cube_dim = CubeDim::new(&client, units);
    let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
    if let Some(grid) = sparse_grid.as_ref() {
        if let Some(plane_dim) = perception_sparse_plane_dim(&client, particle_count, state_dims) {
            let planes_per_cube = perception_sparse_planes_per_cube(particle_count);
            let raw_state_gradient_arg = raw_state_gradient
                .as_ref()
                .cloned()
                .unwrap_or_else(|| features.clone());
            let state_gradient_inverse_arg = state_gradient_inverse
                .as_ref()
                .cloned()
                .unwrap_or_else(|| density.clone());
            perception_forward_sparse_plane_kernel::launch(
                &client,
                CubeCount::Static(
                    particle_count.div_ceil(planes_per_cube as usize) as u32,
                    batches as u32,
                    1,
                ),
                CubeDim::new_2d(plane_dim, planes_per_cube),
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                features.clone().into_tensor_arg(),
                raw_state_gradient_arg.into_tensor_arg(),
                state_gradient_inverse_arg.into_tensor_arg(),
                args_forward,
                retain_adjoint_state,
                dtype.into(),
            );
        } else {
            let inverse = state_gradient_inverse.clone().unwrap_or_else(|| {
                empty_device_dtype(
                    client.clone(),
                    features.device.clone(),
                    Shape::new([batches, particle_count, 4]),
                    dtype,
                )
            });
            perception_forward_sparse_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                features.clone().into_tensor_arg(),
                inverse.clone().into_tensor_arg(),
                perception_args(cfg, dtype),
                dtype.into(),
            );
            let channel_units = batches * particle_count * state_dims;
            let channel_cube_dim = CubeDim::new(&client, channel_units);
            let channel_cube_count =
                calculate_cube_count_elemwise(&client, channel_units, channel_cube_dim);
            perception_forward_sparse_channel_kernel::launch(
                &client,
                channel_cube_count,
                channel_cube_dim,
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                inverse.into_tensor_arg(),
                features.clone().into_tensor_arg(),
                raw_state_gradient
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| features.clone())
                    .into_tensor_arg(),
                args_forward,
                retain_adjoint_state,
                dtype.into(),
            );
        }
    } else {
        perception_forward_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            AddressType::U32,
            x.into_tensor_arg(),
            s.into_tensor_arg(),
            density.clone().into_tensor_arg(),
            features.clone().into_tensor_arg(),
            args_forward,
            dtype.into(),
        );
    }
    PerceptionForwardRawOutput {
        features,
        density,
        grid: sparse_grid,
        raw_state_gradient,
        state_gradient_inverse,
    }
}

fn launch_perception_adjoint<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    feature_grad: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
) -> PerceptionAdjointRawOutput<R> {
    launch_perception_adjoint_impl(x, s, feature_grad, None, cfg)
}

#[allow(clippy::too_many_arguments)]
fn launch_perception_adjoint_prepared<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    feature_grad: CubeTensor<R>,
    density: CubeTensor<R>,
    grid: PerceptionGridRaw<R>,
    raw_state_gradient: CubeTensor<R>,
    state_gradient_inverse: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
) -> PerceptionAdjointRawOutput<R> {
    launch_perception_adjoint_impl(
        x,
        s,
        feature_grad,
        Some(PerceptionPreparedAdjointRaw {
            density,
            grid,
            raw_state_gradient,
            state_gradient_inverse,
        }),
        cfg,
    )
}

fn launch_perception_adjoint_impl<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    feature_grad: CubeTensor<R>,
    prepared: Option<PerceptionPreparedAdjointRaw<R>>,
    cfg: PerceptionCubeAdjointConfig,
) -> PerceptionAdjointRawOutput<R> {
    let dims = x.shape().dims::<3>();
    let batches = dims[0];
    let particle_count = dims[1];
    let state_dims = s.shape().dims::<3>()[2];
    let dtype = x.dtype;
    let client = x.client.clone();
    let device = x.device.clone();
    let density = prepared.as_ref().map_or_else(
        || {
            empty_device_dtype(
                client.clone(),
                device.clone(),
                Shape::new([batches, particle_count]),
                dtype,
            )
        },
        |prepared| prepared.density.clone(),
    );
    let raw_state_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, state_dims, 2]),
        dtype,
    );
    let raw_density_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, 2]),
        dtype,
    );
    let moment_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, 4]),
        dtype,
    );
    let density_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count]),
        dtype,
    );
    let position_grad = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, 2]),
        dtype,
    );
    let state_grad = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, state_dims]),
        dtype,
    );

    let has_prepared = prepared.is_some();
    let sparse_grid = prepared
        .as_ref()
        .map(|prepared| prepared.grid.clone())
        .or_else(|| {
            (should_use_sparse_grid(particle_count, cfg) && !cfg.compute_position_grad)
                .then(|| launch_perception_grid(x.clone(), cfg))
        });
    let sparse_plane_dim = sparse_grid
        .as_ref()
        .and_then(|_| perception_sparse_plane_dim(&client, particle_count, state_dims));
    let args_density = perception_args(cfg, dtype);
    if has_prepared {
        // Density and grid were retained from the matching forward pass.
    } else if let Some(grid) = sparse_grid.as_ref() {
        let units = batches * particle_count;
        let cube_dim = CubeDim::new(&client, units);
        let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
        perception_density_sparse_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            grid.offsets.clone().into_tensor_arg(),
            grid.permutation.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    } else if particle_count >= 512 {
        let tile_size = 256usize;
        let query_blocks = particle_count.div_ceil(tile_size);
        perception_density_tiled_kernel::launch(
            &client,
            CubeCount::Static(query_blocks as u32, batches as u32, 1),
            CubeDim::new_1d(tile_size as u32),
            AddressType::U32,
            x.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    } else {
        let units = batches * particle_count;
        let cube_dim = CubeDim::new(&client, units);
        let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
        perception_density_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            args_density,
            dtype.into(),
        );
    }

    let precompute_units = batches * particle_count;
    let precompute_cube_dim = CubeDim::new(&client, precompute_units);
    let precompute_cube_count =
        calculate_cube_count_elemwise(&client, precompute_units, precompute_cube_dim);
    let args_precompute = perception_args(cfg, dtype);
    if let Some(prepared) = prepared.as_ref() {
        let channel_units = batches * particle_count * state_dims;
        let channel_cube_dim = CubeDim::new(&client, channel_units);
        let channel_cube_count =
            calculate_cube_count_elemwise(&client, channel_units, channel_cube_dim);
        perception_precompute_adjoint_saved_kernel::launch(
            &client,
            channel_cube_count,
            channel_cube_dim,
            AddressType::U32,
            prepared.raw_state_gradient.clone().into_tensor_arg(),
            prepared.state_gradient_inverse.clone().into_tensor_arg(),
            feature_grad.clone().into_tensor_arg(),
            raw_state_adjoint.clone().into_tensor_arg(),
            args_precompute,
            dtype.into(),
        );
    } else if let Some(grid) = sparse_grid.as_ref() {
        if let Some(plane_dim) = sparse_plane_dim {
            let planes_per_cube = perception_sparse_planes_per_cube(particle_count);
            perception_precompute_adjoint_sparse_plane_kernel::launch(
                &client,
                CubeCount::Static(
                    particle_count.div_ceil(planes_per_cube as usize) as u32,
                    batches as u32,
                    1,
                ),
                CubeDim::new_2d(plane_dim, planes_per_cube),
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                feature_grad.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                raw_state_adjoint.clone().into_tensor_arg(),
                args_precompute,
                dtype.into(),
            );
        } else {
            let inverse = empty_device_dtype(
                client.clone(),
                device.clone(),
                Shape::new([batches, particle_count, 4]),
                dtype,
            );
            perception_inverse_sparse_kernel::launch(
                &client,
                precompute_cube_count,
                precompute_cube_dim,
                AddressType::U32,
                x.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                inverse.clone().into_tensor_arg(),
                perception_args(cfg, dtype),
                dtype.into(),
            );
            let channel_units = batches * particle_count * state_dims;
            let channel_cube_dim = CubeDim::new(&client, channel_units);
            let channel_cube_count =
                calculate_cube_count_elemwise(&client, channel_units, channel_cube_dim);
            perception_precompute_adjoint_sparse_kernel::launch(
                &client,
                channel_cube_count,
                channel_cube_dim,
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                feature_grad.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                grid.offsets.clone().into_tensor_arg(),
                grid.permutation.clone().into_tensor_arg(),
                inverse.into_tensor_arg(),
                raw_state_adjoint.clone().into_tensor_arg(),
                args_precompute,
                dtype.into(),
            );
        }
    } else {
        perception_precompute_adjoint_kernel::launch(
            &client,
            precompute_cube_count,
            precompute_cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            s.clone().into_tensor_arg(),
            feature_grad.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            raw_state_adjoint.clone().into_tensor_arg(),
            raw_density_adjoint.clone().into_tensor_arg(),
            moment_adjoint.clone().into_tensor_arg(),
            args_precompute,
            dtype.into(),
        );
    }

    if cfg.compute_position_grad {
        let density_adj_units = batches * particle_count;
        let density_adj_cube_dim = CubeDim::new(&client, density_adj_units);
        let density_adj_cube_count =
            calculate_cube_count_elemwise(&client, density_adj_units, density_adj_cube_dim);
        let args_density_adjoint = perception_args(cfg, dtype);
        perception_density_adjoint_kernel::launch(
            &client,
            density_adj_cube_count,
            density_adj_cube_dim,
            AddressType::U32,
            x.clone().into_tensor_arg(),
            s.clone().into_tensor_arg(),
            feature_grad.clone().into_tensor_arg(),
            density.clone().into_tensor_arg(),
            raw_state_adjoint.clone().into_tensor_arg(),
            moment_adjoint.clone().into_tensor_arg(),
            density_adjoint.clone().into_tensor_arg(),
            args_density_adjoint,
            dtype.into(),
        );
    }

    if cfg.compute_state_grad {
        let state_units = batches * particle_count * state_dims;
        let state_cube_dim = CubeDim::new(&client, state_units);
        let state_cube_count = calculate_cube_count_elemwise(&client, state_units, state_cube_dim);
        let args_state = perception_args(cfg, dtype);
        if let Some(grid) = sparse_grid.as_ref() {
            if let Some(plane_dim) = sparse_plane_dim {
                let planes_per_cube = perception_sparse_planes_per_cube(particle_count);
                perception_state_output_sparse_plane_kernel::launch(
                    &client,
                    CubeCount::Static(
                        particle_count.div_ceil(planes_per_cube as usize) as u32,
                        batches as u32,
                        1,
                    ),
                    CubeDim::new_2d(plane_dim, planes_per_cube),
                    AddressType::U32,
                    x.clone().into_tensor_arg(),
                    s.clone().into_tensor_arg(),
                    feature_grad.clone().into_tensor_arg(),
                    density.clone().into_tensor_arg(),
                    raw_state_adjoint.clone().into_tensor_arg(),
                    grid.offsets.clone().into_tensor_arg(),
                    grid.permutation.clone().into_tensor_arg(),
                    state_grad.clone().into_tensor_arg(),
                    args_state,
                    dtype.into(),
                );
            } else {
                perception_state_output_sparse_kernel::launch(
                    &client,
                    state_cube_count,
                    state_cube_dim,
                    AddressType::U32,
                    x.clone().into_tensor_arg(),
                    s.clone().into_tensor_arg(),
                    feature_grad.clone().into_tensor_arg(),
                    density.clone().into_tensor_arg(),
                    raw_state_adjoint.clone().into_tensor_arg(),
                    grid.offsets.clone().into_tensor_arg(),
                    grid.permutation.clone().into_tensor_arg(),
                    state_grad.clone().into_tensor_arg(),
                    args_state,
                    dtype.into(),
                );
            }
        } else {
            perception_state_output_kernel::launch(
                &client,
                state_cube_count,
                state_cube_dim,
                AddressType::U32,
                x.clone().into_tensor_arg(),
                s.clone().into_tensor_arg(),
                feature_grad.clone().into_tensor_arg(),
                density.clone().into_tensor_arg(),
                raw_state_adjoint.clone().into_tensor_arg(),
                state_grad.clone().into_tensor_arg(),
                args_state,
                dtype.into(),
            );
        }
    }

    if cfg.compute_position_grad {
        let position_units = batches * particle_count * 2;
        let position_cube_dim = CubeDim::new(&client, position_units);
        let position_cube_count =
            calculate_cube_count_elemwise(&client, position_units, position_cube_dim);
        let args_position = perception_args(cfg, dtype);
        perception_position_output_kernel::launch(
            &client,
            position_cube_count,
            position_cube_dim,
            AddressType::U32,
            x.into_tensor_arg(),
            s.into_tensor_arg(),
            feature_grad.into_tensor_arg(),
            density.into_tensor_arg(),
            raw_state_adjoint.into_tensor_arg(),
            raw_density_adjoint.into_tensor_arg(),
            moment_adjoint.into_tensor_arg(),
            density_adjoint.into_tensor_arg(),
            position_grad.clone().into_tensor_arg(),
            args_position,
            dtype.into(),
        );
    }

    PerceptionAdjointRawOutput {
        position_grad,
        state_grad,
    }
}

#[derive(Clone, CubeLaunch, CubeType)]
struct PerceptionArgs {
    eps: InputScalar,
    eps0: InputScalar,
    state_grad: u32,
    density_grad: u32,
    scale_equivariance: u32,
    particle_density_equivariance: u32,
    log_norm_grad: u32,
    log_norm_density_grad: u32,
    hybrid_state_gradient: u32,
    position_features: u32,
    output_position_grad: u32,
    grid_width: u32,
    grid_height: u32,
}

fn perception_args<R: CubeRuntime>(
    cfg: PerceptionCubeAdjointConfig,
    dtype: DType,
) -> PerceptionArgsLaunch<R> {
    PerceptionArgsLaunch::new(
        InputScalar::new(cfg.eps, dtype),
        InputScalar::new(cfg.eps0, dtype),
        u32::from(cfg.state_grad),
        u32::from(cfg.density_grad),
        u32::from(cfg.scale_equivariance),
        u32::from(cfg.particle_density_equivariance),
        u32::from(cfg.log_norm_grad),
        u32::from(cfg.log_norm_density_grad),
        u32::from(cfg.hybrid_state_gradient),
        u32::from(cfg.position_features),
        u32::from(cfg.compute_position_grad),
        cfg.grid_width,
        cfg.grid_height,
    )
}

#[cube]
fn perception_grid_cell<F: Float>(x: F, y: F, args: &PerceptionArgs) -> (i32, i32) {
    let eps = args.eps.get::<F>();
    let width = args.grid_width as i32;
    let height = args.grid_height as i32;
    let mut cell_x = i32::cast_from((x / eps).floor()) + width / 2;
    let mut cell_y = i32::cast_from((y / eps).floor()) + height / 2;
    cell_x = clamp(cell_x, 0i32, width - 1i32);
    cell_y = clamp(cell_y, 0i32, height - 1i32);
    (cell_x, cell_y)
}

#[cube]
fn perception_grid_offset(offsets: &Tensor<u32>, batch: usize, cell: usize) -> usize {
    offsets[batch * offsets.stride(0) + cell * offsets.stride(1)] as usize
}

#[cube]
fn perception_grid_particle(permutation: &Tensor<u32>, batch: usize, slot: usize) -> usize {
    permutation[batch * permutation.stride(0) + slot * permutation.stride(1)] as usize
}

#[cube(launch, address_type = "dynamic")]
fn perception_grid_zero_kernel(counts: &mut Tensor<Atomic<u32>>) {
    let index = ABSOLUTE_POS;
    if index >= counts.len() {
        terminate!();
    }
    counts[index].store(0u32);
}

#[cube(launch, address_type = "dynamic")]
fn perception_grid_count_kernel<F: Float>(
    x: &Tensor<F>,
    counts: &mut Tensor<Atomic<u32>>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let (cell_x, cell_y) = perception_grid_cell::<F>(
        x_value::<F>(x, batch, particle, 0),
        x_value::<F>(x, batch, particle, 1),
        args,
    );
    let cell = cell_y as usize * args.grid_width as usize + cell_x as usize;
    counts[batch * counts.stride(0) + cell * counts.stride(1)].fetch_add(1u32);
}

#[cube(launch, address_type = "dynamic")]
fn perception_grid_scan_kernel(counts: &mut Tensor<Atomic<u32>>, offsets: &mut Tensor<u32>) {
    let batch = ABSOLUTE_POS;
    if batch >= counts.shape(0) {
        terminate!();
    }
    let cell_count = counts.shape(1);
    let mut sum = 0u32;
    offsets[batch * offsets.stride(0)] = 0u32;
    let mut cell = 0usize;
    while cell < cell_count {
        let index = batch * counts.stride(0) + cell * counts.stride(1);
        let count = counts[index].load();
        counts[index].store(sum);
        sum += count;
        offsets[batch * offsets.stride(0) + (cell + 1usize) * offsets.stride(1)] = sum;
        cell += 1usize;
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_grid_scatter_kernel<F: Float>(
    x: &Tensor<F>,
    cursors: &mut Tensor<Atomic<u32>>,
    permutation: &mut Tensor<u32>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let (cell_x, cell_y) = perception_grid_cell::<F>(
        x_value::<F>(x, batch, particle, 0),
        x_value::<F>(x, batch, particle, 1),
        args,
    );
    let cell = cell_y as usize * args.grid_width as usize + cell_x as usize;
    let slot =
        cursors[batch * cursors.stride(0) + cell * cursors.stride(1)].fetch_add(1u32) as usize;
    permutation[batch * permutation.stride(0) + slot * permutation.stride(1)] = particle as u32;
}

#[cube]
fn poly6<F: Float>(r2: F, eps: F) -> F {
    let eps2 = eps * eps;
    let mut out = F::new(0.0_f32);
    if r2 < eps2 {
        let compact = eps2 - r2;
        out = F::new(4.0_f32) / (F::new(core::f32::consts::PI) * eps.powf(F::new(8.0_f32)))
            * compact
            * compact
            * compact;
    }
    out
}

#[cube]
fn poly6_delta_adjoint<F: Float>(dx: F, dy: F, r2: F, eps: F, output_adjoint: F) -> (F, F) {
    let eps2 = eps * eps;
    let mut out_x = F::new(0.0_f32);
    let mut out_y = F::new(0.0_f32);
    if r2 < eps2 {
        let compact = eps2 - r2;
        let norm = F::new(4.0_f32) / (F::new(core::f32::consts::PI) * eps.powf(F::new(8.0_f32)));
        let dkernel_dr2 = F::new(-3.0_f32) * norm * compact * compact;
        out_x = output_adjoint * dkernel_dr2 * F::new(2.0_f32) * dx;
        out_y = output_adjoint * dkernel_dr2 * F::new(2.0_f32) * dy;
    }
    (out_x, out_y)
}

#[cube]
fn spiky_gradient<F: Float>(dx: F, dy: F, r2: F, eps: F, coeff: F) -> (F, F) {
    let eps2 = eps * eps;
    let mut gx = F::new(0.0_f32);
    let mut gy = F::new(0.0_f32);
    if r2 > F::new(0.0_f32) && r2 < eps2 {
        let r = r2.sqrt();
        let norm = F::new(30.0_f32) / (F::new(core::f32::consts::PI) * eps.powf(F::new(5.0_f32)));
        let mag = coeff * norm * (eps - r) * (eps - r) / r;
        gx = mag * dx;
        gy = mag * dy;
    }
    (gx, gy)
}

#[cube]
fn spiky_delta_adjoint<F: Float>(
    dx: F,
    dy: F,
    r2: F,
    eps: F,
    coeff: F,
    adj_x: F,
    adj_y: F,
) -> (F, F) {
    let eps2 = eps * eps;
    let mut out_x = F::new(0.0_f32);
    let mut out_y = F::new(0.0_f32);
    if r2 > F::new(0.0_f32) && r2 < eps2 {
        let r = r2.sqrt();
        let norm =
            coeff * F::new(30.0_f32) / (F::new(core::f32::consts::PI) * eps.powf(F::new(5.0_f32)));
        let scale = norm * (eps - r) * (eps - r) / r;
        let dscale_dr = norm * (F::new(1.0_f32) - eps2 / r2);
        let dot = adj_x * dx + adj_y * dy;
        out_x = scale * adj_x + dscale_dr * dot * dx / r;
        out_y = scale * adj_y + dscale_dr * dot * dy / r;
    }
    (out_x, out_y)
}

#[cube]
fn recip_finite<F: Float>(value: F) -> F {
    let mut out = F::new(0.0_f32);
    if value.abs() > F::new(1.0e-20_f32) {
        out = F::new(1.0_f32) / value;
    }
    out
}

#[cube]
fn density_adjoint_from_volume<F: Float>(density: F, volume_adjoint: F) -> F {
    let mut out = F::new(0.0_f32);
    if density.abs() > F::new(1.0e-20_f32) {
        out = F::new(0.0_f32) - volume_adjoint / (density * density);
    }
    out
}

#[cube]
fn log_normalize_adjoint_2<F: Float>(x: F, y: F, adj_x: F, adj_y: F) -> (F, F) {
    let norm = (x * x + y * y + F::new(LOG_NORMALIZE_EPSILON * LOG_NORMALIZE_EPSILON))
        .sqrt()
        .max(F::new(LOG_NORMALIZE_EPSILON));
    let log1p = (F::new(1.0_f32) + norm).ln();
    let scale = log1p / norm;
    let dscale_dnorm = (norm / (F::new(1.0_f32) + norm) - log1p) / (norm * norm);
    let dot = x * adj_x + y * adj_y;
    let radial = dscale_dnorm * dot / norm;
    (scale * adj_x + radial * x, scale * adj_y + radial * y)
}

#[cube]
fn log_normalize_2<F: Float>(x: F, y: F) -> (F, F) {
    let norm = (x * x + y * y + F::new(LOG_NORMALIZE_EPSILON * LOG_NORMALIZE_EPSILON))
        .sqrt()
        .max(F::new(LOG_NORMALIZE_EPSILON));
    let scale = (F::new(1.0_f32) + norm).ln() / norm;
    (x * scale, y * scale)
}

#[cube]
fn inverse_2d<F: Float>(m00: F, m01: F, m11: F) -> (F, F, F, F) {
    let det = m00 * m11 - m01 * m01;
    let mut i00 = F::new(1.0_f32);
    let mut i01 = F::new(0.0_f32);
    let mut i10 = F::new(0.0_f32);
    let mut i11 = F::new(1.0_f32);
    if det.abs() >= F::new(1.0e-3_f32) {
        let inv_det = F::new(1.0_f32) / det;
        i00 = m11 * inv_det;
        i01 = F::new(0.0_f32) - m01 * inv_det;
        i10 = i01;
        i11 = m00 * inv_det;
    }
    (i00, i01, i10, i11)
}

#[cube]
fn inverse_matrix_adjoint_2d<F: Float>(
    i00: F,
    i01: F,
    i10: F,
    i11: F,
    a00: F,
    a01: F,
    a10: F,
    a11: F,
) -> (F, F, F, F) {
    let m00 =
        F::new(0.0_f32) - (i00 * a00 * i00 + i00 * a01 * i01 + i10 * a10 * i00 + i10 * a11 * i01);
    let m01 =
        F::new(0.0_f32) - (i00 * a00 * i10 + i00 * a01 * i11 + i10 * a10 * i10 + i10 * a11 * i11);
    let m10 =
        F::new(0.0_f32) - (i01 * a00 * i00 + i01 * a01 * i01 + i11 * a10 * i00 + i11 * a11 * i01);
    let m11 =
        F::new(0.0_f32) - (i01 * a00 * i10 + i01 * a01 * i11 + i11 * a10 * i10 + i11 * a11 * i11);
    (m00, m01, m10, m11)
}

#[cube(launch, address_type = "dynamic")]
fn perception_density_tiled_kernel<F: Float>(
    x: &Tensor<F>,
    density: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let particle_count = x.shape(1);
    let unit = UNIT_POS as usize;
    let batch = CUBE_POS_Y as usize;
    let particle = CUBE_POS_X as usize * 256usize + unit;
    let active = batch < x.shape(0) && particle < particle_count;
    let mut xi = F::new(0.0_f32);
    let mut yi = F::new(0.0_f32);
    if active {
        xi = x_value::<F>(x, batch, particle, 0);
        yi = x_value::<F>(x, batch, particle, 1);
    }

    let eps = args.eps.get::<F>();
    let mut rho = F::new(0.0_f32);
    let mut tile_x = SharedMemory::<F>::new(256usize);
    let mut tile_y = SharedMemory::<F>::new(256usize);
    let mut tile_start = 0usize;
    while tile_start < particle_count {
        let neighbor = tile_start + unit;
        if neighbor < particle_count {
            tile_x[unit] = x_value::<F>(x, batch, neighbor, 0);
            tile_y[unit] = x_value::<F>(x, batch, neighbor, 1);
        } else {
            tile_x[unit] = F::new(0.0_f32);
            tile_y[unit] = F::new(0.0_f32);
        }
        sync_cube();

        if active {
            let mut tile_len = 256usize;
            if tile_start + tile_len > particle_count {
                tile_len = particle_count - tile_start;
            }
            let mut local = 0usize;
            while local < tile_len {
                let dx = tile_x[local] - xi;
                let dy = tile_y[local] - yi;
                rho += poly6::<F>(dx * dx + dy * dy, eps);
                local += 1;
            }
        }
        sync_cube();
        tile_start += 256usize;
    }

    if active {
        density[batch * density.stride(0) + particle * density.stride(1)] = rho;
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_density_kernel<F: Float>(
    x: &Tensor<F>,
    density: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let base_i = batch * x.stride(0) + particle * x.stride(1);
    let xi = x[base_i];
    let yi = x[base_i + x.stride(2)];
    let eps = args.eps.get::<F>();
    let mut rho = F::new(0.0_f32);
    let mut j = 0usize;
    while j < particle_count {
        let base_j = batch * x.stride(0) + j * x.stride(1);
        let dx = x[base_j] - xi;
        let dy = x[base_j + x.stride(2)] - yi;
        rho += poly6::<F>(dx * dx + dy * dy, eps);
        j += 1;
    }
    density[batch * density.stride(0) + particle * density.stride(1)] = rho;
}

#[cube(launch, address_type = "dynamic")]
fn perception_forward_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    features: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let state_dims = s.shape(2);
    let eps = args.eps.get::<F>();

    let mut channel = 0usize;
    while channel < state_dims {
        write_feature::<F>(
            features,
            batch,
            particle,
            channel,
            state_value::<F>(s, batch, particle, channel),
        );
        channel += 1;
    }

    let blur_cursor = state_dims;
    channel = 0usize;
    while channel < state_dims {
        let mut blur = F::new(0.0_f32);
        let mut neighbor = 0usize;
        while neighbor < particle_count {
            let (_, _, r2) = delta_from::<F>(x, batch, particle, neighbor);
            let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
            blur += poly6::<F>(r2, eps) * volume * state_value::<F>(s, batch, neighbor, channel);
            neighbor += 1;
        }
        write_feature::<F>(features, batch, particle, blur_cursor + channel, blur);
        channel += 1;
    }

    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m10 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    if args.state_grad != 0 && args.hybrid_state_gradient != 0 {
        let mut neighbor = 0usize;
        while neighbor < particle_count {
            if neighbor != particle {
                let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
                let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                m00 += dx * gx;
                m01 += dx * gy;
                m10 += dy * gx;
                m11 += dy * gy;
            }
            neighbor += 1;
        }
    }
    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if args.state_grad != 0 && args.hybrid_state_gradient != 0 {
        let (a, b, c, d) = inverse_2d::<F>(m00, m01, m11);
        inv00 = a;
        inv01 = b;
        inv10 = c;
        inv11 = d;
    }

    if args.state_grad != 0 {
        let state_grad_cursor = feature_state_grad_cursor(state_dims);
        channel = 0usize;
        while channel < state_dims {
            let mut raw_x = F::new(0.0_f32);
            let mut raw_y = F::new(0.0_f32);
            let state_i = state_value::<F>(s, batch, particle, channel);
            let mut neighbor = 0usize;
            while neighbor < particle_count {
                let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
                let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                let diff = state_value::<F>(s, batch, neighbor, channel) - state_i;
                raw_x += diff * gx;
                raw_y += diff * gy;
                neighbor += 1;
            }
            let corrected_x = raw_x * inv00 + raw_y * inv10;
            let corrected_y = raw_x * inv01 + raw_y * inv11;
            let (out_x, out_y) = if args.log_norm_grad != 0 {
                log_normalize_2::<F>(corrected_x, corrected_y)
            } else {
                (corrected_x, corrected_y)
            };
            write_feature::<F>(
                features,
                batch,
                particle,
                state_grad_cursor + channel * 2,
                out_x,
            );
            write_feature::<F>(
                features,
                batch,
                particle,
                state_grad_cursor + channel * 2 + 1,
                out_y,
            );
            channel += 1;
        }
    }

    if args.density_grad != 0 {
        let density_cursor = feature_density_grad_cursor(state_dims, args);
        let mut raw_x = F::new(0.0_f32);
        let mut raw_y = F::new(0.0_f32);
        let mut neighbor = 0usize;
        while neighbor < particle_count {
            let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
            let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, F::new(1.0_f32));
            raw_x += gx;
            raw_y += gy;
            neighbor += 1;
        }
        let scale = density_gradient_scale::<F>(eps, particle_count, args);
        raw_x *= scale;
        raw_y *= scale;
        let (out_x, out_y) = if args.log_norm_density_grad != 0 {
            log_normalize_2::<F>(raw_x, raw_y)
        } else {
            (raw_x, raw_y)
        };
        write_feature::<F>(features, batch, particle, density_cursor, out_x);
        write_feature::<F>(features, batch, particle, density_cursor + 1, out_y);
    }

    if args.position_features != 0 {
        let position_cursor = feature_position_cursor(state_dims, args);
        write_feature::<F>(
            features,
            batch,
            particle,
            position_cursor,
            x_value::<F>(x, batch, particle, 0),
        );
        write_feature::<F>(
            features,
            batch,
            particle,
            position_cursor + 1,
            x_value::<F>(x, batch, particle, 1),
        );
    }
}

#[cube]
fn feature_state_grad_cursor(state_dims: usize) -> usize {
    state_dims * 2
}

#[cube]
fn feature_density_grad_cursor(state_dims: usize, args: &PerceptionArgs) -> usize {
    let mut cursor = state_dims * 2;
    if args.state_grad != 0 {
        cursor += state_dims * 2;
    }
    cursor
}

#[cube]
fn feature_position_cursor(state_dims: usize, args: &PerceptionArgs) -> usize {
    let mut cursor = state_dims * 2;
    if args.state_grad != 0 {
        cursor += state_dims * 2;
    }
    if args.density_grad != 0 {
        cursor += 2;
    }
    cursor
}

#[cube]
fn state_scale<F: Float>(eps: F, args: &PerceptionArgs) -> F {
    if args.scale_equivariance != 0 {
        eps / args.eps0.get::<F>().max(F::new(f32::MIN_POSITIVE))
    } else {
        F::new(1.0_f32)
    }
}

#[cube]
fn density_gradient_scale<F: Float>(eps: F, particle_count: usize, args: &PerceptionArgs) -> F {
    let scale = if args.scale_equivariance != 0 {
        let ratio = eps / args.eps0.get::<F>().max(F::new(f32::MIN_POSITIVE));
        ratio * ratio * ratio
    } else {
        F::new(1.0_f32)
    };
    if args.particle_density_equivariance != 0 {
        scale / F::cast_from(particle_count)
    } else {
        scale
    }
}

#[cube]
fn feature<F: Float>(feature_grad: &Tensor<F>, batch: usize, particle: usize, channel: usize) -> F {
    feature_grad[batch * feature_grad.stride(0)
        + particle * feature_grad.stride(1)
        + channel * feature_grad.stride(2)]
}

#[cube]
fn state_value<F: Float>(s: &Tensor<F>, batch: usize, particle: usize, channel: usize) -> F {
    s[batch * s.stride(0) + particle * s.stride(1) + channel * s.stride(2)]
}

#[cube]
fn x_value<F: Float>(x: &Tensor<F>, batch: usize, particle: usize, axis: usize) -> F {
    x[batch * x.stride(0) + particle * x.stride(1) + axis * x.stride(2)]
}

#[cube]
fn density_value<F: Float>(density: &Tensor<F>, batch: usize, particle: usize) -> F {
    density[batch * density.stride(0) + particle * density.stride(1)]
}

#[cube]
fn write_feature<F: Float>(
    features: &mut Tensor<F>,
    batch: usize,
    particle: usize,
    channel: usize,
    value: F,
) {
    features[batch * features.stride(0)
        + particle * features.stride(1)
        + channel * features.stride(2)] = value;
}

#[cube]
fn raw_state_adjoint_value<F: Float>(
    raw_state_adjoint: &Tensor<F>,
    batch: usize,
    particle: usize,
    channel: usize,
    axis: usize,
) -> F {
    raw_state_adjoint[batch * raw_state_adjoint.stride(0)
        + particle * raw_state_adjoint.stride(1)
        + channel * raw_state_adjoint.stride(2)
        + axis * raw_state_adjoint.stride(3)]
}

#[cube]
fn raw_density_adjoint_value<F: Float>(
    raw_density_adjoint: &Tensor<F>,
    batch: usize,
    particle: usize,
    axis: usize,
) -> F {
    raw_density_adjoint[batch * raw_density_adjoint.stride(0)
        + particle * raw_density_adjoint.stride(1)
        + axis * raw_density_adjoint.stride(2)]
}

#[cube]
fn moment_adjoint_value<F: Float>(
    moment_adjoint: &Tensor<F>,
    batch: usize,
    particle: usize,
    index: usize,
) -> F {
    moment_adjoint[batch * moment_adjoint.stride(0)
        + particle * moment_adjoint.stride(1)
        + index * moment_adjoint.stride(2)]
}

#[cube]
fn delta_from<F: Float>(x: &Tensor<F>, batch: usize, lhs: usize, rhs: usize) -> (F, F, F) {
    let dx = x_value::<F>(x, batch, rhs, 0) - x_value::<F>(x, batch, lhs, 0);
    let dy = x_value::<F>(x, batch, rhs, 1) - x_value::<F>(x, batch, lhs, 1);
    (dx, dy, dx * dx + dy * dy)
}

#[cube]
fn perception_neighbor_cell(
    cell_x: i32,
    cell_y: i32,
    neighbor_cell: usize,
    args: &PerceptionArgs,
) -> (bool, usize) {
    let offset_x = (neighbor_cell % 3usize) as i32 - 1i32;
    let offset_y = (neighbor_cell / 3usize) as i32 - 1i32;
    let x = cell_x + offset_x;
    let y = cell_y + offset_y;
    let valid = x >= 0i32 && x < args.grid_width as i32 && y >= 0i32 && y < args.grid_height as i32;
    let mut cell = 0usize;
    if valid {
        cell = y as usize * args.grid_width as usize + x as usize;
    }
    (valid, cell)
}

#[cube]
fn sparse_density_for<F: Float>(
    x: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    args: &PerceptionArgs,
) -> F {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut rho = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                rho += poly6::<F>(dx * dx + dy * dy, eps);
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    rho
}

#[cube]
fn sparse_blur_and_state_gradient_for<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    channel: usize,
    args: &PerceptionArgs,
) -> (F, F, F) {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let state_i = state_value::<F>(s, batch, particle, channel);
    let mut blur = F::new(0.0_f32);
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                let r2 = dx * dx + dy * dy;
                let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                let neighbor_state = state_value::<F>(s, batch, neighbor, channel);
                blur += poly6::<F>(r2, eps) * volume * neighbor_state;
                if args.state_grad != 0u32 && neighbor != particle {
                    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    let diff = neighbor_state - state_i;
                    raw_x += diff * gx;
                    raw_y += diff * gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    (blur, raw_x, raw_y)
}

#[cube]
fn sparse_moment_for<F: Float>(
    x: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    args: &PerceptionArgs,
) -> (F, F, F, F) {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m10 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                if neighbor != particle {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let r2 = dx * dx + dy * dy;
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    m00 += dx * gx;
                    m01 += dx * gy;
                    m10 += dy * gx;
                    m11 += dy * gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    (m00, m01, m10, m11)
}

#[cube]
fn sparse_raw_state_gradient_for<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    channel: usize,
    args: &PerceptionArgs,
) -> (F, F) {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let state_i = state_value::<F>(s, batch, particle, channel);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                if neighbor != particle {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let r2 = dx * dx + dy * dy;
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    let diff = state_value::<F>(s, batch, neighbor, channel) - state_i;
                    raw_x += diff * gx;
                    raw_y += diff * gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    (raw_x, raw_y)
}

#[cube]
fn sparse_density_gradient_for<F: Float>(
    x: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    args: &PerceptionArgs,
) -> (F, F) {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                if neighbor != particle {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let (gx, gy) =
                        spiky_gradient::<F>(dx, dy, dx * dx + dy * dy, eps, F::new(1.0_f32));
                    raw_x += gx;
                    raw_y += gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    (raw_x, raw_y)
}

#[cube]
fn sparse_moment_and_density_gradient_for<F: Float>(
    x: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    batch: usize,
    particle: usize,
    args: &PerceptionArgs,
) -> (F, F, F, F, F, F) {
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m10 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    let mut density_x = F::new(0.0_f32);
    let mut density_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                if neighbor != particle {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let r2 = dx * dx + dy * dy;
                    let (density_gx, density_gy) =
                        spiky_gradient::<F>(dx, dy, r2, eps, F::new(1.0_f32));
                    density_x += density_gx;
                    density_y += density_gy;
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    m00 += dx * gx;
                    m01 += dx * gy;
                    m10 += dy * gx;
                    m11 += dy * gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }
    (m00, m01, m10, m11, density_x, density_y)
}

#[cube(launch, address_type = "dynamic")]
fn perception_density_sparse_kernel<F: Float>(
    x: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    density: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    density[batch * density.stride(0) + particle * density.stride(1)] =
        sparse_density_for::<F>(x, offsets, permutation, batch, particle, args);
}

#[cube(launch, address_type = "dynamic")]
fn perception_forward_sparse_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    features: &mut Tensor<F>,
    inverse: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let state_dims = s.shape(2);
    let mut channel = 0usize;
    while channel < state_dims {
        write_feature::<F>(
            features,
            batch,
            particle,
            channel,
            state_value::<F>(s, batch, particle, channel),
        );
        channel += 1usize;
    }

    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if args.state_grad != 0u32 && args.hybrid_state_gradient != 0u32 {
        let (m00, m01, _, m11) =
            sparse_moment_for::<F>(x, density, offsets, permutation, batch, particle, args);
        let (a, b, c, d) = inverse_2d::<F>(m00, m01, m11);
        inv00 = a;
        inv01 = b;
        inv10 = c;
        inv11 = d;
    }
    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    inverse[inverse_base] = inv00;
    inverse[inverse_base + inverse.stride(2)] = inv01;
    inverse[inverse_base + 2usize * inverse.stride(2)] = inv10;
    inverse[inverse_base + 3usize * inverse.stride(2)] = inv11;

    if args.density_grad != 0u32 {
        let eps = args.eps.get::<F>();
        let density_cursor = feature_density_grad_cursor(state_dims, args);
        let (mut raw_x, mut raw_y) =
            sparse_density_gradient_for::<F>(x, offsets, permutation, batch, particle, args);
        let scale = density_gradient_scale::<F>(eps, particle_count, args);
        raw_x *= scale;
        raw_y *= scale;
        let (out_x, out_y) = if args.log_norm_density_grad != 0u32 {
            log_normalize_2::<F>(raw_x, raw_y)
        } else {
            (raw_x, raw_y)
        };
        write_feature::<F>(features, batch, particle, density_cursor, out_x);
        write_feature::<F>(features, batch, particle, density_cursor + 1usize, out_y);
    }

    if args.position_features != 0u32 {
        let position_cursor = feature_position_cursor(state_dims, args);
        write_feature::<F>(
            features,
            batch,
            particle,
            position_cursor,
            x_value::<F>(x, batch, particle, 0),
        );
        write_feature::<F>(
            features,
            batch,
            particle,
            position_cursor + 1usize,
            x_value::<F>(x, batch, particle, 1),
        );
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_forward_sparse_channel_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    inverse: &Tensor<F>,
    features: &mut Tensor<F>,
    raw_state_gradient: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[comptime] retain_adjoint_state: bool,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    let state_dims = s.shape(2);
    if index >= x.shape(0) * particle_count * state_dims {
        terminate!();
    }
    let batch = index / (particle_count * state_dims);
    let local = index - batch * particle_count * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let (blur, raw_x, raw_y) = sparse_blur_and_state_gradient_for::<F>(
        x,
        s,
        density,
        offsets,
        permutation,
        batch,
        particle,
        channel,
        args,
    );
    if retain_adjoint_state {
        let raw_base = batch * raw_state_gradient.stride(0)
            + particle * raw_state_gradient.stride(1)
            + channel * raw_state_gradient.stride(2);
        raw_state_gradient[raw_base] = raw_x;
        raw_state_gradient[raw_base + raw_state_gradient.stride(3)] = raw_y;
    }
    write_feature::<F>(features, batch, particle, state_dims + channel, blur);

    if args.state_grad != 0u32 {
        let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
        let inv00 = inverse[inverse_base];
        let inv01 = inverse[inverse_base + inverse.stride(2)];
        let inv10 = inverse[inverse_base + 2usize * inverse.stride(2)];
        let inv11 = inverse[inverse_base + 3usize * inverse.stride(2)];
        let corrected_x = raw_x * inv00 + raw_y * inv10;
        let corrected_y = raw_x * inv01 + raw_y * inv11;
        let (out_x, out_y) = if args.log_norm_grad != 0u32 {
            log_normalize_2::<F>(corrected_x, corrected_y)
        } else {
            (corrected_x, corrected_y)
        };
        let cursor = feature_state_grad_cursor(state_dims) + channel * 2usize;
        write_feature::<F>(features, batch, particle, cursor, out_x);
        write_feature::<F>(features, batch, particle, cursor + 1usize, out_y);
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_forward_sparse_plane_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    features: &mut Tensor<F>,
    raw_state_gradient: &mut Tensor<F>,
    state_gradient_inverse: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[comptime] retain_adjoint_state: bool,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let particle = CUBE_POS_X as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    let particle_count = x.shape(1);
    if particle >= particle_count {
        terminate!();
    }
    let state_dims = s.shape(2);
    let active_channel = lane < state_dims;

    if active_channel {
        write_feature::<F>(
            features,
            batch,
            particle,
            lane,
            state_value::<F>(s, batch, particle, lane),
        );
    }

    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if lane == 0usize {
        let (m00, m01, _, m11, mut density_x, mut density_y) =
            sparse_moment_and_density_gradient_for::<F>(
                x,
                density,
                offsets,
                permutation,
                batch,
                particle,
                args,
            );
        if args.state_grad != 0u32 && args.hybrid_state_gradient != 0u32 {
            (inv00, inv01, inv10, inv11) = inverse_2d::<F>(m00, m01, m11);
        }
        if args.density_grad != 0u32 {
            let eps = args.eps.get::<F>();
            let scale = density_gradient_scale::<F>(eps, particle_count, args);
            density_x *= scale;
            density_y *= scale;
            let (out_x, out_y) = if args.log_norm_density_grad != 0u32 {
                log_normalize_2::<F>(density_x, density_y)
            } else {
                (density_x, density_y)
            };
            let cursor = feature_density_grad_cursor(state_dims, args);
            write_feature::<F>(features, batch, particle, cursor, out_x);
            write_feature::<F>(features, batch, particle, cursor + 1usize, out_y);
        }
        if args.position_features != 0u32 {
            let cursor = feature_position_cursor(state_dims, args);
            write_feature::<F>(
                features,
                batch,
                particle,
                cursor,
                x_value::<F>(x, batch, particle, 0),
            );
            write_feature::<F>(
                features,
                batch,
                particle,
                cursor + 1usize,
                x_value::<F>(x, batch, particle, 1),
            );
        }
    }
    inv00 = plane_broadcast(inv00, 0u32);
    inv01 = plane_broadcast(inv01, 0u32);
    inv10 = plane_broadcast(inv10, 0u32);
    inv11 = plane_broadcast(inv11, 0u32);
    if retain_adjoint_state && lane == 0usize {
        let inverse_base =
            batch * state_gradient_inverse.stride(0) + particle * state_gradient_inverse.stride(1);
        state_gradient_inverse[inverse_base] = inv00;
        state_gradient_inverse[inverse_base + state_gradient_inverse.stride(2)] = inv01;
        state_gradient_inverse[inverse_base + 2usize * state_gradient_inverse.stride(2)] = inv10;
        state_gradient_inverse[inverse_base + 3usize * state_gradient_inverse.stride(2)] = inv11;
    }

    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut state_i = F::new(0.0_f32);
    if active_channel {
        state_i = state_value::<F>(s, batch, particle, lane);
    }
    let mut blur = F::new(0.0_f32);
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let mut blur_weight = F::new(0.0_f32);
                let mut gx = F::new(0.0_f32);
                let mut gy = F::new(0.0_f32);
                if lane == 0usize {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let r2 = dx * dx + dy * dy;
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    blur_weight = poly6::<F>(r2, eps) * volume;
                    if args.state_grad != 0u32 && neighbor != particle {
                        (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    }
                }
                blur_weight = plane_broadcast(blur_weight, 0u32);
                gx = plane_broadcast(gx, 0u32);
                gy = plane_broadcast(gy, 0u32);
                if active_channel {
                    let neighbor_state = state_value::<F>(s, batch, neighbor, lane);
                    blur += blur_weight * neighbor_state;
                    if args.state_grad != 0u32 && neighbor != particle {
                        let diff = neighbor_state - state_i;
                        raw_x += diff * gx;
                        raw_y += diff * gy;
                    }
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }

    if active_channel {
        if retain_adjoint_state {
            let raw_base = batch * raw_state_gradient.stride(0)
                + particle * raw_state_gradient.stride(1)
                + lane * raw_state_gradient.stride(2);
            raw_state_gradient[raw_base] = raw_x;
            raw_state_gradient[raw_base + raw_state_gradient.stride(3)] = raw_y;
        }
        write_feature::<F>(features, batch, particle, state_dims + lane, blur);
        if args.state_grad != 0u32 {
            let corrected_x = raw_x * inv00 + raw_y * inv10;
            let corrected_y = raw_x * inv01 + raw_y * inv11;
            let (out_x, out_y) = if args.log_norm_grad != 0u32 {
                log_normalize_2::<F>(corrected_x, corrected_y)
            } else {
                (corrected_x, corrected_y)
            };
            let cursor = feature_state_grad_cursor(state_dims) + lane * 2usize;
            write_feature::<F>(features, batch, particle, cursor, out_x);
            write_feature::<F>(features, batch, particle, cursor + 1usize, out_y);
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_inverse_sparse_kernel<F: Float>(
    x: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    inverse: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if args.state_grad != 0u32 && args.hybrid_state_gradient != 0u32 {
        let (m00, m01, _, m11) =
            sparse_moment_for::<F>(x, density, offsets, permutation, batch, particle, args);
        let (a, b, c, d) = inverse_2d::<F>(m00, m01, m11);
        inv00 = a;
        inv01 = b;
        inv10 = c;
        inv11 = d;
    }
    let base = batch * inverse.stride(0) + particle * inverse.stride(1);
    inverse[base] = inv00;
    inverse[base + inverse.stride(2)] = inv01;
    inverse[base + 2usize * inverse.stride(2)] = inv10;
    inverse[base + 3usize * inverse.stride(2)] = inv11;
}

#[cube(launch, address_type = "dynamic")]
fn perception_precompute_adjoint_saved_kernel<F: Float>(
    raw_state_gradient: &Tensor<F>,
    state_gradient_inverse: &Tensor<F>,
    feature_grad: &Tensor<F>,
    raw_state_adjoint: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = raw_state_gradient.shape(1);
    let state_dims = raw_state_gradient.shape(2);
    if index >= raw_state_gradient.shape(0) * particle_count * state_dims {
        terminate!();
    }
    let batch = index / (particle_count * state_dims);
    let local = index - batch * particle_count * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let raw_base = batch * raw_state_gradient.stride(0)
        + particle * raw_state_gradient.stride(1)
        + channel * raw_state_gradient.stride(2);
    let raw_x = raw_state_gradient[raw_base];
    let raw_y = raw_state_gradient[raw_base + raw_state_gradient.stride(3)];
    let inverse_base =
        batch * state_gradient_inverse.stride(0) + particle * state_gradient_inverse.stride(1);
    let inv00 = state_gradient_inverse[inverse_base];
    let inv01 = state_gradient_inverse[inverse_base + state_gradient_inverse.stride(2)];
    let inv10 = state_gradient_inverse[inverse_base + 2usize * state_gradient_inverse.stride(2)];
    let inv11 = state_gradient_inverse[inverse_base + 3usize * state_gradient_inverse.stride(2)];
    let scale = state_scale::<F>(args.eps.get::<F>(), args);
    let input_x = (raw_x * inv00 + raw_y * inv10) * scale;
    let input_y = (raw_x * inv01 + raw_y * inv11) * scale;
    let cursor = feature_state_grad_cursor(state_dims) + channel * 2usize;
    let adj_x = feature::<F>(feature_grad, batch, particle, cursor);
    let adj_y = feature::<F>(feature_grad, batch, particle, cursor + 1usize);
    let (mut corrected_adj_x, mut corrected_adj_y) = if args.log_norm_grad != 0u32 {
        log_normalize_adjoint_2::<F>(input_x, input_y, adj_x, adj_y)
    } else {
        (adj_x, adj_y)
    };
    corrected_adj_x *= scale;
    corrected_adj_y *= scale;
    let raw_adj_x = corrected_adj_x * inv00 + corrected_adj_y * inv01;
    let raw_adj_y = corrected_adj_x * inv10 + corrected_adj_y * inv11;
    let output_base = batch * raw_state_adjoint.stride(0)
        + particle * raw_state_adjoint.stride(1)
        + channel * raw_state_adjoint.stride(2);
    raw_state_adjoint[output_base] = raw_adj_x;
    raw_state_adjoint[output_base + raw_state_adjoint.stride(3)] = raw_adj_y;
}

#[cube(launch, address_type = "dynamic")]
fn perception_precompute_adjoint_sparse_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    inverse: &Tensor<F>,
    raw_state_adjoint: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    let state_dims = s.shape(2);
    if index >= x.shape(0) * particle_count * state_dims {
        terminate!();
    }
    let batch = index / (particle_count * state_dims);
    let local = index - batch * particle_count * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let (raw_x, raw_y) = sparse_raw_state_gradient_for::<F>(
        x,
        s,
        density,
        offsets,
        permutation,
        batch,
        particle,
        channel,
        args,
    );
    let inverse_base = batch * inverse.stride(0) + particle * inverse.stride(1);
    let inv00 = inverse[inverse_base];
    let inv01 = inverse[inverse_base + inverse.stride(2)];
    let inv10 = inverse[inverse_base + 2usize * inverse.stride(2)];
    let inv11 = inverse[inverse_base + 3usize * inverse.stride(2)];
    let scale = state_scale::<F>(args.eps.get::<F>(), args);
    let corrected_x = raw_x * inv00 + raw_y * inv10;
    let corrected_y = raw_x * inv01 + raw_y * inv11;
    let input_x = corrected_x * scale;
    let input_y = corrected_y * scale;
    let cursor = feature_state_grad_cursor(state_dims) + channel * 2usize;
    let adj_x = feature::<F>(feature_grad, batch, particle, cursor);
    let adj_y = feature::<F>(feature_grad, batch, particle, cursor + 1usize);
    let (mut corrected_adj_x, mut corrected_adj_y) = if args.log_norm_grad != 0u32 {
        log_normalize_adjoint_2::<F>(input_x, input_y, adj_x, adj_y)
    } else {
        (adj_x, adj_y)
    };
    corrected_adj_x *= scale;
    corrected_adj_y *= scale;
    let raw_adj_x = corrected_adj_x * inv00 + corrected_adj_y * inv01;
    let raw_adj_y = corrected_adj_x * inv10 + corrected_adj_y * inv11;
    let output_base = batch * raw_state_adjoint.stride(0)
        + particle * raw_state_adjoint.stride(1)
        + channel * raw_state_adjoint.stride(2);
    raw_state_adjoint[output_base] = raw_adj_x;
    raw_state_adjoint[output_base + raw_state_adjoint.stride(3)] = raw_adj_y;
}

#[cube(launch, address_type = "dynamic")]
fn perception_precompute_adjoint_sparse_plane_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    raw_state_adjoint: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let particle = CUBE_POS_X as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    if particle >= x.shape(1) {
        terminate!();
    }
    let state_dims = s.shape(2);
    let active_channel = lane < state_dims;

    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if lane == 0usize && args.state_grad != 0u32 && args.hybrid_state_gradient != 0u32 {
        let (m00, m01, _, m11) =
            sparse_moment_for::<F>(x, density, offsets, permutation, batch, particle, args);
        (inv00, inv01, inv10, inv11) = inverse_2d::<F>(m00, m01, m11);
    }
    inv00 = plane_broadcast(inv00, 0u32);
    inv01 = plane_broadcast(inv01, 0u32);
    inv10 = plane_broadcast(inv10, 0u32);
    inv11 = plane_broadcast(inv11, 0u32);

    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let eps = args.eps.get::<F>();
    let mut state_i = F::new(0.0_f32);
    if active_channel {
        state_i = state_value::<F>(s, batch, particle, lane);
    }
    let mut raw_x = F::new(0.0_f32);
    let mut raw_y = F::new(0.0_f32);
    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let mut gx = F::new(0.0_f32);
                let mut gy = F::new(0.0_f32);
                if lane == 0usize && neighbor != particle {
                    let dx = x_value::<F>(x, batch, neighbor, 0) - xi;
                    let dy = x_value::<F>(x, batch, neighbor, 1) - yi;
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    (gx, gy) = spiky_gradient::<F>(dx, dy, dx * dx + dy * dy, eps, volume);
                }
                gx = plane_broadcast(gx, 0u32);
                gy = plane_broadcast(gy, 0u32);
                if active_channel && neighbor != particle {
                    let diff = state_value::<F>(s, batch, neighbor, lane) - state_i;
                    raw_x += diff * gx;
                    raw_y += diff * gy;
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }

    if active_channel {
        let scale = state_scale::<F>(eps, args);
        let corrected_x = raw_x * inv00 + raw_y * inv10;
        let corrected_y = raw_x * inv01 + raw_y * inv11;
        let input_x = corrected_x * scale;
        let input_y = corrected_y * scale;
        let cursor = feature_state_grad_cursor(state_dims) + lane * 2usize;
        let adj_x = feature::<F>(feature_grad, batch, particle, cursor);
        let adj_y = feature::<F>(feature_grad, batch, particle, cursor + 1usize);
        let (mut corrected_adj_x, mut corrected_adj_y) = if args.log_norm_grad != 0u32 {
            log_normalize_adjoint_2::<F>(input_x, input_y, adj_x, adj_y)
        } else {
            (adj_x, adj_y)
        };
        corrected_adj_x *= scale;
        corrected_adj_y *= scale;
        let raw_adj_x = corrected_adj_x * inv00 + corrected_adj_y * inv01;
        let raw_adj_y = corrected_adj_x * inv10 + corrected_adj_y * inv11;
        let output_base = batch * raw_state_adjoint.stride(0)
            + particle * raw_state_adjoint.stride(1)
            + lane * raw_state_adjoint.stride(2);
        raw_state_adjoint[output_base] = raw_adj_x;
        raw_state_adjoint[output_base + raw_state_adjoint.stride(3)] = raw_adj_y;
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_state_output_sparse_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    state_grad: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    let state_dims = s.shape(2);
    if index >= x.shape(0) * particle_count * state_dims {
        terminate!();
    }
    let batch = index / (particle_count * state_dims);
    let local = index - batch * particle_count * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let eps = args.eps.get::<F>();
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let blur_cursor = state_dims;
    let volume = recip_finite::<F>(density_value::<F>(density, batch, particle));
    let mut out = feature::<F>(feature_grad, batch, particle, channel);

    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let dx = xi - x_value::<F>(x, batch, neighbor, 0);
                let dy = yi - x_value::<F>(x, batch, neighbor, 1);
                out += feature::<F>(feature_grad, batch, neighbor, blur_cursor + channel)
                    * poly6::<F>(dx * dx + dy * dy, eps)
                    * volume;
                if args.state_grad != 0u32 && neighbor != particle {
                    out += raw_state_pair_contribution::<F>(
                        x,
                        density,
                        raw_state_adjoint,
                        batch,
                        neighbor,
                        particle,
                        channel,
                        eps,
                    );
                    out -= raw_state_pair_contribution::<F>(
                        x,
                        density,
                        raw_state_adjoint,
                        batch,
                        particle,
                        neighbor,
                        channel,
                        eps,
                    );
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }

    state_grad[batch * state_grad.stride(0)
        + particle * state_grad.stride(1)
        + channel * state_grad.stride(2)] = out;
}

#[cube(launch, address_type = "dynamic")]
fn perception_state_output_sparse_plane_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    offsets: &Tensor<u32>,
    permutation: &Tensor<u32>,
    state_grad: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let particle = CUBE_POS_X as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    if particle >= x.shape(1) {
        terminate!();
    }
    let state_dims = s.shape(2);
    let active_channel = lane < state_dims;
    let eps = args.eps.get::<F>();
    let xi = x_value::<F>(x, batch, particle, 0);
    let yi = x_value::<F>(x, batch, particle, 1);
    let (cell_x, cell_y) = perception_grid_cell::<F>(xi, yi, args);
    let blur_cursor = state_dims;
    let volume_i = recip_finite::<F>(density_value::<F>(density, batch, particle));
    let mut out = F::new(0.0_f32);
    if active_channel {
        out = feature::<F>(feature_grad, batch, particle, lane);
    }

    let mut neighbor_cell = 0usize;
    while neighbor_cell < 9usize {
        let (valid, cell) = perception_neighbor_cell(cell_x, cell_y, neighbor_cell, args);
        if valid {
            let mut slot = perception_grid_offset(offsets, batch, cell);
            let end = perception_grid_offset(offsets, batch, cell + 1usize);
            while slot < end {
                let neighbor = perception_grid_particle(permutation, batch, slot);
                let mut blur_weight = F::new(0.0_f32);
                let mut incoming_gx = F::new(0.0_f32);
                let mut incoming_gy = F::new(0.0_f32);
                let mut outgoing_gx = F::new(0.0_f32);
                let mut outgoing_gy = F::new(0.0_f32);
                if lane == 0usize {
                    let dx = xi - x_value::<F>(x, batch, neighbor, 0);
                    let dy = yi - x_value::<F>(x, batch, neighbor, 1);
                    let r2 = dx * dx + dy * dy;
                    blur_weight = poly6::<F>(r2, eps) * volume_i;
                    if args.state_grad != 0u32 && neighbor != particle {
                        (incoming_gx, incoming_gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume_i);
                        let volume_neighbor =
                            recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                        (outgoing_gx, outgoing_gy) = spiky_gradient::<F>(
                            F::new(0.0_f32) - dx,
                            F::new(0.0_f32) - dy,
                            r2,
                            eps,
                            volume_neighbor,
                        );
                    }
                }
                blur_weight = plane_broadcast(blur_weight, 0u32);
                incoming_gx = plane_broadcast(incoming_gx, 0u32);
                incoming_gy = plane_broadcast(incoming_gy, 0u32);
                outgoing_gx = plane_broadcast(outgoing_gx, 0u32);
                outgoing_gy = plane_broadcast(outgoing_gy, 0u32);
                if active_channel {
                    out += feature::<F>(feature_grad, batch, neighbor, blur_cursor + lane)
                        * blur_weight;
                    if args.state_grad != 0u32 && neighbor != particle {
                        out += raw_state_adjoint_value::<F>(
                            raw_state_adjoint,
                            batch,
                            neighbor,
                            lane,
                            0,
                        ) * incoming_gx
                            + raw_state_adjoint_value::<F>(
                                raw_state_adjoint,
                                batch,
                                neighbor,
                                lane,
                                1,
                            ) * incoming_gy;
                        out -= raw_state_adjoint_value::<F>(
                            raw_state_adjoint,
                            batch,
                            particle,
                            lane,
                            0,
                        ) * outgoing_gx
                            + raw_state_adjoint_value::<F>(
                                raw_state_adjoint,
                                batch,
                                particle,
                                lane,
                                1,
                            ) * outgoing_gy;
                    }
                }
                slot += 1usize;
            }
        }
        neighbor_cell += 1usize;
    }

    if active_channel {
        state_grad[batch * state_grad.stride(0)
            + particle * state_grad.stride(1)
            + lane * state_grad.stride(2)] = out;
    }
}

#[cube]
fn raw_state_pair_contribution<F: Float>(
    x: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    batch: usize,
    query: usize,
    neighbor: usize,
    channel: usize,
    eps: F,
) -> F {
    let (dx, dy, r2) = delta_from::<F>(x, batch, query, neighbor);
    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 0) * gx
        + raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 1) * gy
}

#[cube]
fn blur_weight_adjoint<F: Float>(
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    batch: usize,
    query: usize,
    neighbor: usize,
    state_dims: usize,
) -> F {
    let blur_cursor = state_dims;
    let mut out = F::new(0.0_f32);
    let mut channel = 0usize;
    while channel < state_dims {
        out += feature::<F>(feature_grad, batch, query, blur_cursor + channel)
            * state_value::<F>(s, batch, neighbor, channel);
        channel += 1;
    }
    out
}

#[cube]
fn moment_position_delta<F: Float>(
    x: &Tensor<F>,
    density: &Tensor<F>,
    moment_adjoint: &Tensor<F>,
    batch: usize,
    query: usize,
    neighbor: usize,
    eps: F,
) -> (F, F, F) {
    let (dx, dy, r2) = delta_from::<F>(x, batch, query, neighbor);
    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
    let (vgx, vgy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
    let ma00 = moment_adjoint_value::<F>(moment_adjoint, batch, query, 0);
    let ma01 = moment_adjoint_value::<F>(moment_adjoint, batch, query, 1);
    let ma10 = moment_adjoint_value::<F>(moment_adjoint, batch, query, 2);
    let ma11 = moment_adjoint_value::<F>(moment_adjoint, batch, query, 3);
    let direct_x = ma00 * vgx + ma01 * vgy;
    let direct_y = ma10 * vgx + ma11 * vgy;
    let grad_adj_x = ma00 * dx + ma10 * dy;
    let grad_adj_y = ma01 * dx + ma11 * dy;
    let (delta_x, delta_y) =
        spiky_delta_adjoint::<F>(dx, dy, r2, eps, volume, grad_adj_x, grad_adj_y);
    let (ugx, ugy) = spiky_gradient::<F>(dx, dy, r2, eps, F::new(1.0_f32));
    (
        direct_x + delta_x,
        direct_y + delta_y,
        grad_adj_x * ugx + grad_adj_y * ugy,
    )
}

#[cube(launch, address_type = "dynamic")]
fn perception_precompute_adjoint_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &mut Tensor<F>,
    raw_density_adjoint: &mut Tensor<F>,
    moment_adjoint: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let state_dims = s.shape(2);
    let eps = args.eps.get::<F>();
    let mut m00 = F::new(0.0_f32);
    let mut m01 = F::new(0.0_f32);
    let mut m10 = F::new(0.0_f32);
    let mut m11 = F::new(0.0_f32);
    if args.state_grad != 0 && args.hybrid_state_gradient != 0 {
        let mut neighbor = 0usize;
        while neighbor < particle_count {
            if neighbor != particle {
                let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
                let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                m00 += dx * gx;
                m01 += dx * gy;
                m10 += dy * gx;
                m11 += dy * gy;
            }
            neighbor += 1;
        }
    }
    let mut inv00 = F::new(1.0_f32);
    let mut inv01 = F::new(0.0_f32);
    let mut inv10 = F::new(0.0_f32);
    let mut inv11 = F::new(1.0_f32);
    if args.state_grad != 0 && args.hybrid_state_gradient != 0 {
        let (a, b, c, d) = inverse_2d::<F>(m00, m01, m11);
        inv00 = a;
        inv01 = b;
        inv10 = c;
        inv11 = d;
    }

    let mut inv_adj00 = F::new(0.0_f32);
    let mut inv_adj01 = F::new(0.0_f32);
    let mut inv_adj10 = F::new(0.0_f32);
    let mut inv_adj11 = F::new(0.0_f32);
    if args.state_grad != 0 {
        let state_grad_cursor = feature_state_grad_cursor(state_dims);
        let scale = state_scale::<F>(eps, args);
        let mut channel = 0usize;
        while channel < state_dims {
            let mut raw_x = F::new(0.0_f32);
            let mut raw_y = F::new(0.0_f32);
            let state_i = state_value::<F>(s, batch, particle, channel);
            let mut neighbor = 0usize;
            while neighbor < particle_count {
                if neighbor != particle {
                    let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
                    let volume = recip_finite::<F>(density_value::<F>(density, batch, neighbor));
                    let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, volume);
                    let diff = state_value::<F>(s, batch, neighbor, channel) - state_i;
                    raw_x += diff * gx;
                    raw_y += diff * gy;
                }
                neighbor += 1;
            }
            let corrected_x = raw_x * inv00 + raw_y * inv10;
            let corrected_y = raw_x * inv01 + raw_y * inv11;
            let input_x = corrected_x * scale;
            let input_y = corrected_y * scale;
            let adj_x = feature::<F>(
                feature_grad,
                batch,
                particle,
                state_grad_cursor + channel * 2,
            );
            let adj_y = feature::<F>(
                feature_grad,
                batch,
                particle,
                state_grad_cursor + channel * 2 + 1,
            );
            let (mut corrected_adj_x, mut corrected_adj_y) = if args.log_norm_grad != 0 {
                log_normalize_adjoint_2::<F>(input_x, input_y, adj_x, adj_y)
            } else {
                (adj_x, adj_y)
            };
            corrected_adj_x *= scale;
            corrected_adj_y *= scale;
            let raw_adj_x = corrected_adj_x * inv00 + corrected_adj_y * inv01;
            let raw_adj_y = corrected_adj_x * inv10 + corrected_adj_y * inv11;
            raw_state_adjoint[batch * raw_state_adjoint.stride(0)
                + particle * raw_state_adjoint.stride(1)
                + channel * raw_state_adjoint.stride(2)] = raw_adj_x;
            raw_state_adjoint[batch * raw_state_adjoint.stride(0)
                + particle * raw_state_adjoint.stride(1)
                + channel * raw_state_adjoint.stride(2)
                + raw_state_adjoint.stride(3)] = raw_adj_y;
            if args.output_position_grad != 0 && args.hybrid_state_gradient != 0 {
                inv_adj00 += raw_x * corrected_adj_x;
                inv_adj01 += raw_x * corrected_adj_y;
                inv_adj10 += raw_y * corrected_adj_x;
                inv_adj11 += raw_y * corrected_adj_y;
            }
            channel += 1;
        }
    }

    let mut ma00 = F::new(0.0_f32);
    let mut ma01 = F::new(0.0_f32);
    let mut ma10 = F::new(0.0_f32);
    let mut ma11 = F::new(0.0_f32);
    if args.output_position_grad != 0 && args.state_grad != 0 && args.hybrid_state_gradient != 0 {
        let (a, b, c, d) = inverse_matrix_adjoint_2d::<F>(
            inv00, inv01, inv10, inv11, inv_adj00, inv_adj01, inv_adj10, inv_adj11,
        );
        ma00 = a;
        ma01 = b;
        ma10 = c;
        ma11 = d;
    }
    if args.output_position_grad != 0 {
        moment_adjoint[batch * moment_adjoint.stride(0) + particle * moment_adjoint.stride(1)] =
            ma00;
        moment_adjoint[batch * moment_adjoint.stride(0)
            + particle * moment_adjoint.stride(1)
            + moment_adjoint.stride(2)] = ma01;
        moment_adjoint[batch * moment_adjoint.stride(0)
            + particle * moment_adjoint.stride(1)
            + 2 * moment_adjoint.stride(2)] = ma10;
        moment_adjoint[batch * moment_adjoint.stride(0)
            + particle * moment_adjoint.stride(1)
            + 3 * moment_adjoint.stride(2)] = ma11;
    }

    let mut raw_density_x = F::new(0.0_f32);
    let mut raw_density_y = F::new(0.0_f32);
    if args.output_position_grad != 0 && args.density_grad != 0 {
        let mut neighbor = 0usize;
        while neighbor < particle_count {
            if neighbor != particle {
                let (dx, dy, r2) = delta_from::<F>(x, batch, particle, neighbor);
                let (gx, gy) = spiky_gradient::<F>(dx, dy, r2, eps, F::new(1.0_f32));
                raw_density_x += gx;
                raw_density_y += gy;
            }
            neighbor += 1;
        }
        let scale = density_gradient_scale::<F>(eps, particle_count, args);
        let input_x = raw_density_x * scale;
        let input_y = raw_density_y * scale;
        let density_cursor = feature_density_grad_cursor(state_dims, args);
        let adj_x = feature::<F>(feature_grad, batch, particle, density_cursor);
        let adj_y = feature::<F>(feature_grad, batch, particle, density_cursor + 1);
        let (mut adj_x, mut adj_y) = if args.log_norm_density_grad != 0 {
            log_normalize_adjoint_2::<F>(input_x, input_y, adj_x, adj_y)
        } else {
            (adj_x, adj_y)
        };
        adj_x *= scale;
        adj_y *= scale;
        raw_density_x = adj_x;
        raw_density_y = adj_y;
    }
    if args.output_position_grad != 0 {
        raw_density_adjoint
            [batch * raw_density_adjoint.stride(0) + particle * raw_density_adjoint.stride(1)] =
            raw_density_x;
        raw_density_adjoint[batch * raw_density_adjoint.stride(0)
            + particle * raw_density_adjoint.stride(1)
            + raw_density_adjoint.stride(2)] = raw_density_y;
    }
}

#[cube(launch, address_type = "dynamic")]
fn perception_density_adjoint_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    moment_adjoint: &Tensor<F>,
    density_adjoint: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let state_dims = s.shape(2);
    let eps = args.eps.get::<F>();
    let rho = density_value::<F>(density, batch, particle);
    let mut volume_adjoint = F::new(0.0_f32);

    let mut query = 0usize;
    while query < particle_count {
        let (dx, dy, r2) = delta_from::<F>(x, batch, query, particle);
        let kernel = poly6::<F>(r2, eps);
        volume_adjoint +=
            blur_weight_adjoint::<F>(s, feature_grad, batch, query, particle, state_dims) * kernel;
        if args.state_grad != 0 && query != particle {
            let (ugx, ugy) = spiky_gradient::<F>(dx, dy, r2, eps, F::new(1.0_f32));
            let mut channel = 0usize;
            while channel < state_dims {
                let diff = state_value::<F>(s, batch, particle, channel)
                    - state_value::<F>(s, batch, query, channel);
                let grad_adj_x =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 0)
                        * diff;
                let grad_adj_y =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 1)
                        * diff;
                volume_adjoint += grad_adj_x * ugx + grad_adj_y * ugy;
                channel += 1;
            }
            if args.hybrid_state_gradient != 0 {
                let (_, _, moment_volume_adjoint) = moment_position_delta::<F>(
                    x,
                    density,
                    moment_adjoint,
                    batch,
                    query,
                    particle,
                    eps,
                );
                volume_adjoint += moment_volume_adjoint;
            }
        }
        query += 1;
    }

    density_adjoint[batch * density_adjoint.stride(0) + particle * density_adjoint.stride(1)] =
        density_adjoint_from_volume::<F>(rho, volume_adjoint);
}

#[cube(launch, address_type = "dynamic")]
fn perception_state_output_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    state_grad: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    let state_dims = s.shape(2);
    if index >= x.shape(0) * particle_count * state_dims {
        terminate!();
    }
    let batch = index / (particle_count * state_dims);
    let local = index - batch * particle_count * state_dims;
    let particle = local / state_dims;
    let channel = local - particle * state_dims;
    let eps = args.eps.get::<F>();
    let mut out = feature::<F>(feature_grad, batch, particle, channel);

    let blur_cursor = state_dims;
    let mut query = 0usize;
    while query < particle_count {
        let (_, _, r2) = delta_from::<F>(x, batch, query, particle);
        let volume = recip_finite::<F>(density_value::<F>(density, batch, particle));
        out += feature::<F>(feature_grad, batch, query, blur_cursor + channel)
            * poly6::<F>(r2, eps)
            * volume;
        if args.state_grad != 0 {
            if query == particle {
                let mut neighbor = 0usize;
                while neighbor < particle_count {
                    if neighbor != particle {
                        out -= raw_state_pair_contribution::<F>(
                            x,
                            density,
                            raw_state_adjoint,
                            batch,
                            particle,
                            neighbor,
                            channel,
                            eps,
                        );
                    }
                    neighbor += 1;
                }
            } else {
                out += raw_state_pair_contribution::<F>(
                    x,
                    density,
                    raw_state_adjoint,
                    batch,
                    query,
                    particle,
                    channel,
                    eps,
                );
            }
        }
        query += 1;
    }

    state_grad[batch * state_grad.stride(0)
        + particle * state_grad.stride(1)
        + channel * state_grad.stride(2)] = out;
}

#[cube(launch, address_type = "dynamic")]
fn perception_position_output_kernel<F: Float>(
    x: &Tensor<F>,
    s: &Tensor<F>,
    feature_grad: &Tensor<F>,
    density: &Tensor<F>,
    raw_state_adjoint: &Tensor<F>,
    raw_density_adjoint: &Tensor<F>,
    moment_adjoint: &Tensor<F>,
    density_adjoint: &Tensor<F>,
    position_grad: &mut Tensor<F>,
    args: &PerceptionArgs,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = x.shape(1);
    if index >= x.shape(0) * particle_count * 2 {
        terminate!();
    }
    let batch = index / (particle_count * 2);
    let local = index - batch * particle_count * 2;
    let particle = local / 2;
    let axis = local - particle * 2;
    let state_dims = s.shape(2);
    let eps = args.eps.get::<F>();
    let mut out = F::new(0.0_f32);

    let mut query = 0usize;
    while query < particle_count {
        let (dx_j, dy_j, r2_j) = delta_from::<F>(x, batch, query, particle);
        let volume_j = recip_finite::<F>(density_value::<F>(density, batch, particle));
        let blur_adj_j =
            blur_weight_adjoint::<F>(s, feature_grad, batch, query, particle, state_dims)
                * volume_j;
        let (blur_dx_j, blur_dy_j) = poly6_delta_adjoint::<F>(dx_j, dy_j, r2_j, eps, blur_adj_j);
        if axis == 0 {
            out += blur_dx_j;
        } else {
            out += blur_dy_j;
        }

        let (dx_i, dy_i, r2_i) = delta_from::<F>(x, batch, particle, query);
        let volume_i = recip_finite::<F>(density_value::<F>(density, batch, query));
        let blur_adj_i =
            blur_weight_adjoint::<F>(s, feature_grad, batch, particle, query, state_dims)
                * volume_i;
        let (blur_dx_i, blur_dy_i) = poly6_delta_adjoint::<F>(dx_i, dy_i, r2_i, eps, blur_adj_i);
        if axis == 0 {
            out -= blur_dx_i;
        } else {
            out -= blur_dy_i;
        }

        if args.state_grad != 0 && query != particle {
            let mut channel = 0usize;
            while channel < state_dims {
                let diff_j = state_value::<F>(s, batch, particle, channel)
                    - state_value::<F>(s, batch, query, channel);
                let grad_adj_j_x =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 0)
                        * diff_j;
                let grad_adj_j_y =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, query, channel, 1)
                        * diff_j;
                let (state_dx_j, state_dy_j) = spiky_delta_adjoint::<F>(
                    dx_j,
                    dy_j,
                    r2_j,
                    eps,
                    volume_j,
                    grad_adj_j_x,
                    grad_adj_j_y,
                );
                if axis == 0 {
                    out += state_dx_j;
                } else {
                    out += state_dy_j;
                }

                let diff_i = state_value::<F>(s, batch, query, channel)
                    - state_value::<F>(s, batch, particle, channel);
                let grad_adj_i_x =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, particle, channel, 0)
                        * diff_i;
                let grad_adj_i_y =
                    raw_state_adjoint_value::<F>(raw_state_adjoint, batch, particle, channel, 1)
                        * diff_i;
                let (state_dx_i, state_dy_i) = spiky_delta_adjoint::<F>(
                    dx_i,
                    dy_i,
                    r2_i,
                    eps,
                    volume_i,
                    grad_adj_i_x,
                    grad_adj_i_y,
                );
                if axis == 0 {
                    out -= state_dx_i;
                } else {
                    out -= state_dy_i;
                }
                channel += 1;
            }
            if args.hybrid_state_gradient != 0 {
                let (moment_dx_j, moment_dy_j, _) = moment_position_delta::<F>(
                    x,
                    density,
                    moment_adjoint,
                    batch,
                    query,
                    particle,
                    eps,
                );
                let (moment_dx_i, moment_dy_i, _) = moment_position_delta::<F>(
                    x,
                    density,
                    moment_adjoint,
                    batch,
                    particle,
                    query,
                    eps,
                );
                if axis == 0 {
                    out += moment_dx_j - moment_dx_i;
                } else {
                    out += moment_dy_j - moment_dy_i;
                }
            }
        }

        if args.density_grad != 0 && query != particle {
            let raw_density_j_x =
                raw_density_adjoint_value::<F>(raw_density_adjoint, batch, query, 0);
            let raw_density_j_y =
                raw_density_adjoint_value::<F>(raw_density_adjoint, batch, query, 1);
            let (density_dx_j, density_dy_j) = spiky_delta_adjoint::<F>(
                dx_j,
                dy_j,
                r2_j,
                eps,
                F::new(1.0_f32),
                raw_density_j_x,
                raw_density_j_y,
            );
            let raw_density_i_x =
                raw_density_adjoint_value::<F>(raw_density_adjoint, batch, particle, 0);
            let raw_density_i_y =
                raw_density_adjoint_value::<F>(raw_density_adjoint, batch, particle, 1);
            let (density_dx_i, density_dy_i) = spiky_delta_adjoint::<F>(
                dx_i,
                dy_i,
                r2_i,
                eps,
                F::new(1.0_f32),
                raw_density_i_x,
                raw_density_i_y,
            );
            if axis == 0 {
                out += density_dx_j - density_dx_i;
            } else {
                out += density_dy_j - density_dy_i;
            }
        }

        let density_adj_j =
            density_adjoint[batch * density_adjoint.stride(0) + query * density_adjoint.stride(1)];
        let (density_kernel_dx_i, density_kernel_dy_i) =
            poly6_delta_adjoint::<F>(dx_i, dy_i, r2_i, eps, density_adj_j);
        let density_adj_i = density_adjoint
            [batch * density_adjoint.stride(0) + particle * density_adjoint.stride(1)];
        let (density_kernel_dx_j, density_kernel_dy_j) =
            poly6_delta_adjoint::<F>(dx_j, dy_j, r2_j, eps, density_adj_i);
        if axis == 0 {
            out += density_kernel_dx_j - density_kernel_dx_i;
        } else {
            out += density_kernel_dy_j - density_kernel_dy_i;
        }

        query += 1;
    }

    if args.position_features != 0 {
        let position_cursor = feature_position_cursor(state_dims, args);
        out += feature::<F>(feature_grad, batch, particle, position_cursor + axis);
    }
    position_grad[batch * position_grad.stride(0)
        + particle * position_grad.stride(1)
        + axis * position_grad.stride(2)] = out;
}
