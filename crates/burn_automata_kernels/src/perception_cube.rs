use std::marker::PhantomData;

use burn::tensor::{
    Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
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
const LOG_NORMALIZE_EPSILON: f32 = 1.0e-6;

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
}

pub struct PerceptionCubeAdjointOutput<B: BurnBackendTrait> {
    pub position_grad: BurnTensor<B, 3>,
    pub state_grad: BurnTensor<B, 3>,
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

fn perception_feature_dims(state_dims: usize, cfg: PerceptionCubeAdjointConfig) -> usize {
    state_dims * 2
        + usize::from(cfg.state_grad) * state_dims * 2
        + usize::from(cfg.density_grad) * 2
        + usize::from(cfg.position_features) * 2
}

#[allow(clippy::type_complexity)]
fn perception_cube_adjoint_fusion<R, F, I, BT>(
    x: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    s: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    feature_grad: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    cfg: PerceptionCubeAdjointConfig,
) -> KernelResult<PerceptionCubeAdjointOutput<Fusion<CubeBackend<R, F, I, BT>>>>
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
        position_grad: BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 3>::from_primitive(
            TensorPrimitive::Float(position_grad_fusion),
        ),
        state_grad: BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 3>::from_primitive(
            TensorPrimitive::Float(state_grad_fusion),
        ),
    })
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

struct PerceptionAdjointRawOutput<R: CubeRuntime> {
    position_grad: CubeTensor<R>,
    state_grad: CubeTensor<R>,
}

fn launch_perception_adjoint<R: CubeRuntime>(
    x: CubeTensor<R>,
    s: CubeTensor<R>,
    feature_grad: CubeTensor<R>,
    cfg: PerceptionCubeAdjointConfig,
) -> PerceptionAdjointRawOutput<R> {
    let dims = x.shape().dims::<3>();
    let batches = dims[0];
    let particle_count = dims[1];
    let state_dims = s.shape().dims::<3>()[2];
    let dtype = x.dtype;
    let client = x.client.clone();
    let device = x.device.clone();
    let density = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count]),
        dtype,
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

    let args_density = PerceptionArgsLaunch::new(
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
    );
    if particle_count >= 512 {
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
    let args_precompute = PerceptionArgsLaunch::new(
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
    );
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

    if cfg.compute_position_grad {
        let density_adj_units = batches * particle_count;
        let density_adj_cube_dim = CubeDim::new(&client, density_adj_units);
        let density_adj_cube_count =
            calculate_cube_count_elemwise(&client, density_adj_units, density_adj_cube_dim);
        let args_density_adjoint = PerceptionArgsLaunch::new(
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
        );
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
        let args_state = PerceptionArgsLaunch::new(
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
        );
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

    if cfg.compute_position_grad {
        let position_units = batches * particle_count * 2;
        let position_cube_dim = CubeDim::new(&client, position_units);
        let position_cube_count =
            calculate_cube_count_elemwise(&client, position_units, position_cube_dim);
        let args_position = PerceptionArgsLaunch::new(
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
        );
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
