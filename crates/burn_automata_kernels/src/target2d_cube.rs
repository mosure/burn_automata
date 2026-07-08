use std::marker::PhantomData;

use burn::tensor::{
    Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
    backend::Backend as BurnBackendTrait,
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    ops::numeric::{empty_device_dtype, zeros_client},
    tensor::CubeTensor,
};
use burn_fusion::{
    Fusion,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{KernelError, KernelResult};

const TARGET2D_ADJOINT_OP: &str = "burn_automata.target2d.adjoint.v1";
const IMAGE_EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug)]
pub struct Target2dCubeLossConfig {
    pub image_size: usize,
    pub sigma: f32,
    pub lo: f32,
    pub hi: f32,
    pub splat_loss_weight: f32,
    pub color_loss_weight: f32,
    pub density_loss_weight: f32,
    pub background_density_loss_weight: f32,
    pub foreground_density_loss_weight: f32,
    pub center: bool,
}

pub struct Target2dCubeLossOutput<B: BurnBackendTrait> {
    pub position_grad: BurnTensor<B, 3>,
    pub state_grad: BurnTensor<B, 3>,
    pub constant: BurnTensor<B, 1>,
    pub splat: BurnTensor<B, 1>,
    pub color: BurnTensor<B, 1>,
    pub density: BurnTensor<B, 1>,
}

#[allow(unused_variables)]
pub trait Target2dCubeAdjointBackend: BurnBackendTrait + Sized {
    #[allow(clippy::too_many_arguments)]
    fn target2d_cube_adjoint(
        x: BurnTensor<Self, 3>,
        centered_x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        target_rgb: BurnTensor<Self, 3>,
        target_density: BurnTensor<Self, 3>,
        target_foreground: BurnTensor<Self, 3>,
        target_foreground_scale: BurnTensor<Self, 3>,
        pixel_size: BurnTensor<Self, 3>,
        target_points: BurnTensor<Self, 3>,
        cfg: Target2dCubeLossConfig,
    ) -> Option<KernelResult<Target2dCubeLossOutput<Self>>> {
        None
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl Target2dCubeAdjointBackend for burn::backend::Wgpu<f32> {
    fn target2d_cube_adjoint(
        x: BurnTensor<Self, 3>,
        centered_x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        target_rgb: BurnTensor<Self, 3>,
        target_density: BurnTensor<Self, 3>,
        target_foreground: BurnTensor<Self, 3>,
        target_foreground_scale: BurnTensor<Self, 3>,
        pixel_size: BurnTensor<Self, 3>,
        target_points: BurnTensor<Self, 3>,
        cfg: Target2dCubeLossConfig,
    ) -> Option<KernelResult<Target2dCubeLossOutput<Self>>> {
        Some(target2d_cube_adjoint_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(
            x,
            centered_x,
            s,
            target_rgb,
            target_density,
            target_foreground,
            target_foreground_scale,
            pixel_size,
            target_points,
            cfg,
        ))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl Target2dCubeAdjointBackend for burn::backend::Cuda<f32> {
    fn target2d_cube_adjoint(
        x: BurnTensor<Self, 3>,
        centered_x: BurnTensor<Self, 3>,
        s: BurnTensor<Self, 3>,
        target_rgb: BurnTensor<Self, 3>,
        target_density: BurnTensor<Self, 3>,
        target_foreground: BurnTensor<Self, 3>,
        target_foreground_scale: BurnTensor<Self, 3>,
        pixel_size: BurnTensor<Self, 3>,
        target_points: BurnTensor<Self, 3>,
        cfg: Target2dCubeLossConfig,
    ) -> Option<KernelResult<Target2dCubeLossOutput<Self>>> {
        Some(target2d_cube_adjoint_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(
            x,
            centered_x,
            s,
            target_rgb,
            target_density,
            target_foreground,
            target_foreground_scale,
            pixel_size,
            target_points,
            cfg,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn target2d_cube_adjoint_fusion<R, F, I, BT>(
    x: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    centered_x: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    s: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    target_rgb: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    target_density: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    target_foreground: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    target_foreground_scale: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    pixel_size: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    target_points: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    cfg: Target2dCubeLossConfig,
) -> KernelResult<Target2dCubeLossOutput<Fusion<CubeBackend<R, F, I, BT>>>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let dims = x.shape().dims::<3>();
    let batches = dims[0];
    let particle_count = dims[1];
    let state_dims = s.shape().dims::<3>()[2];
    if dims[2] != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "target2d cube adjoint expects x shape [batch, particles, 2], got {:?}",
            dims
        )));
    }
    if cfg.image_size == 0 || !cfg.sigma.is_finite() || cfg.sigma <= 0.0 {
        return Err(KernelError::InvalidArgument(
            "target2d cube adjoint requires positive image_size and sigma".to_string(),
        ));
    }
    if cfg.hi <= cfg.lo || !cfg.hi.is_finite() || !cfg.lo.is_finite() {
        return Err(KernelError::InvalidArgument(
            "target2d cube adjoint requires finite lo < hi".to_string(),
        ));
    }

    let x_fusion = x.clone().into_primitive().tensor();
    let centered_x_fusion = centered_x.into_primitive().tensor();
    let s_fusion = s.clone().into_primitive().tensor();
    let target_rgb_fusion = target_rgb.into_primitive().tensor();
    let target_density_fusion = target_density.into_primitive().tensor();
    let target_foreground_fusion = target_foreground.into_primitive().tensor();
    let target_foreground_scale_fusion = target_foreground_scale.into_primitive().tensor();
    let pixel_size_fusion = pixel_size.into_primitive().tensor();
    let target_points_fusion = target_points.into_primitive().tensor();

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
    let splat_ir = TensorIr::uninit(client.create_empty_handle(), Shape::new([batches]), dtype);
    let color_ir = TensorIr::uninit(client.create_empty_handle(), Shape::new([batches]), dtype);
    let density_ir = TensorIr::uninit(client.create_empty_handle(), Shape::new([batches]), dtype);

    let inputs = [
        centered_x_fusion.clone().into_ir(),
        s_fusion.clone().into_ir(),
        target_rgb_fusion.clone().into_ir(),
        target_density_fusion.clone().into_ir(),
        target_foreground_fusion.clone().into_ir(),
        target_foreground_scale_fusion.clone().into_ir(),
        pixel_size_fusion.clone().into_ir(),
        target_points_fusion.clone().into_ir(),
    ];
    let outputs = [
        position_grad_ir.clone(),
        state_grad_ir.clone(),
        splat_ir.clone(),
        color_ir.clone(),
        density_ir.clone(),
    ];
    let streams = OperationStreams::with_inputs([
        &centered_x_fusion,
        &s_fusion,
        &target_rgb_fusion,
        &target_density_fusion,
        &target_foreground_fusion,
        &target_foreground_scale_fusion,
        &pixel_size_fusion,
        &target_points_fusion,
    ]);
    let op = Target2dAdjointFusionOp::<R, F, I, BT> {
        desc: Target2dAdjointDesc {
            centered_x: inputs[0].clone(),
            s: inputs[1].clone(),
            target_rgb: inputs[2].clone(),
            target_density: inputs[3].clone(),
            target_foreground: inputs[4].clone(),
            target_foreground_scale: inputs[5].clone(),
            pixel_size: inputs[6].clone(),
            target_points: inputs[7].clone(),
            position_grad: position_grad_ir,
            state_grad: state_grad_ir,
            splat: splat_ir,
            color: color_ir,
            density: density_ir,
        },
        cfg,
        _marker: PhantomData,
    };
    let [
        position_grad_fusion,
        state_grad_fusion,
        splat_fusion,
        color_fusion,
        density_fusion,
    ] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(TARGET2D_ADJOINT_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<5>();

    let mut position_grad = BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 3>::from_primitive(
        TensorPrimitive::Float(position_grad_fusion),
    );
    if cfg.center {
        position_grad = position_grad.clone()
            - position_grad
                .clone()
                .mean_dim(1)
                .expand([batches, particle_count, 2]);
    }
    let state_grad = BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 3>::from_primitive(
        TensorPrimitive::Float(state_grad_fusion),
    );
    let splat = BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 1>::from_primitive(
        TensorPrimitive::Float(splat_fusion),
    );
    let color = BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 1>::from_primitive(
        TensorPrimitive::Float(color_fusion),
    );
    let density = BurnTensor::<Fusion<CubeBackend<R, F, I, BT>>, 1>::from_primitive(
        TensorPrimitive::Float(density_fusion),
    );
    let dot = x
        .mul(position_grad.clone())
        .reshape([batches, particle_count * 2])
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        + s.mul(state_grad.clone())
            .reshape([batches, particle_count * state_dims])
            .sum_dim(1)
            .squeeze_dim::<1>(1);
    let constant = splat.clone().mul_scalar(cfg.splat_loss_weight as f64) - dot;

    Ok(Target2dCubeLossOutput {
        position_grad,
        state_grad,
        constant,
        splat,
        color,
        density,
    })
}

#[derive(Clone, Debug)]
struct Target2dAdjointDesc {
    centered_x: TensorIr,
    s: TensorIr,
    target_rgb: TensorIr,
    target_density: TensorIr,
    target_foreground: TensorIr,
    target_foreground_scale: TensorIr,
    pixel_size: TensorIr,
    target_points: TensorIr,
    position_grad: TensorIr,
    state_grad: TensorIr,
    splat: TensorIr,
    color: TensorIr,
    density: TensorIr,
}

#[derive(Debug)]
struct Target2dAdjointFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: Target2dAdjointDesc,
    cfg: Target2dCubeLossConfig,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for Target2dAdjointFusionOp<R, F, I, BT>
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
        let centered_x = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.centered_x);
        let s = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.s);
        let target_rgb = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target_rgb);
        let target_density =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target_density);
        let target_foreground =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target_foreground);
        let target_foreground_scale =
            handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target_foreground_scale);
        let pixel_size = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.pixel_size);
        let target_points = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target_points);
        let output = launch_target2d_adjoint(
            centered_x,
            s,
            target_rgb,
            target_density,
            target_foreground,
            target_foreground_scale,
            pixel_size,
            target_points,
            self.cfg,
        );
        handles.register_float_tensor::<Raw<R, F, I, BT>>(
            &self.desc.position_grad.id,
            output.position_grad,
        );
        handles
            .register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.state_grad.id, output.state_grad);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.splat.id, output.splat);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.color.id, output.color);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.density.id, output.density);
    }
}

struct Target2dAdjointRawOutput<R: CubeRuntime> {
    position_grad: CubeTensor<R>,
    state_grad: CubeTensor<R>,
    splat: CubeTensor<R>,
    color: CubeTensor<R>,
    density: CubeTensor<R>,
}

#[allow(clippy::too_many_arguments)]
fn launch_target2d_adjoint<R: CubeRuntime>(
    centered_x: CubeTensor<R>,
    s: CubeTensor<R>,
    target_rgb: CubeTensor<R>,
    target_density: CubeTensor<R>,
    target_foreground: CubeTensor<R>,
    target_foreground_scale: CubeTensor<R>,
    pixel_size: CubeTensor<R>,
    target_points: CubeTensor<R>,
    cfg: Target2dCubeLossConfig,
) -> Target2dAdjointRawOutput<R> {
    let dims = centered_x.shape().dims::<3>();
    let batches = dims[0];
    let particle_count = dims[1];
    let state_dims = s.shape().dims::<3>()[2];
    let pixels = cfg.image_size * cfg.image_size;
    let dtype = centered_x.dtype;
    let client = centered_x.client.clone();
    let device = centered_x.device.clone();
    let denominator = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count]),
        dtype,
    );
    let pixel_adjoint = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, pixels, 4]),
        dtype,
    );
    let pixel_loss = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, pixels, 4]),
        dtype,
    );
    let position_grad = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, 2]),
        dtype,
    );
    let state_grad = zeros_client(
        client.clone(),
        device.clone(),
        Shape::new([batches, particle_count, state_dims]),
        dtype,
    );
    let splat = empty_device_dtype(client.clone(), device.clone(), Shape::new([batches]), dtype);
    let color = empty_device_dtype(client.clone(), device.clone(), Shape::new([batches]), dtype);
    let density = empty_device_dtype(client.clone(), device.clone(), Shape::new([batches]), dtype);

    let particle_units = batches * particle_count;
    let particle_cube_dim = CubeDim::new(&client, particle_units);
    let particle_cube_count =
        calculate_cube_count_elemwise(&client, particle_units, particle_cube_dim);
    let pixel_units = batches * pixels;
    let pixel_cube_dim = CubeDim::new(&client, pixel_units);
    let pixel_cube_count = calculate_cube_count_elemwise(&client, pixel_units, pixel_cube_dim);
    let batch_cube_dim = CubeDim::new(&client, batches);
    let batch_cube_count = calculate_cube_count_elemwise(&client, batches, batch_cube_dim);

    denominator_kernel::launch(
        &client,
        particle_cube_count.clone(),
        particle_cube_dim,
        AddressType::U32,
        centered_x.clone().into_tensor_arg(),
        pixel_size.clone().into_tensor_arg(),
        denominator.clone().into_tensor_arg(),
        cfg.image_size,
        InputScalar::new(cfg.sigma, dtype),
        InputScalar::new(cfg.lo, dtype),
        InputScalar::new(cfg.hi, dtype),
        dtype.into(),
    );
    pixel_loss_kernel::launch(
        &client,
        pixel_cube_count,
        pixel_cube_dim,
        AddressType::U32,
        centered_x.clone().into_tensor_arg(),
        s.clone().into_tensor_arg(),
        denominator.clone().into_tensor_arg(),
        target_rgb.into_tensor_arg(),
        target_density.into_tensor_arg(),
        target_foreground.into_tensor_arg(),
        target_foreground_scale.into_tensor_arg(),
        pixel_size.clone().into_tensor_arg(),
        target_points.clone().into_tensor_arg(),
        pixel_adjoint.clone().into_tensor_arg(),
        pixel_loss.clone().into_tensor_arg(),
        cfg.image_size,
        InputScalar::new(cfg.sigma, dtype),
        InputScalar::new(cfg.lo, dtype),
        InputScalar::new(cfg.hi, dtype),
        InputScalar::new(cfg.splat_loss_weight, dtype),
        InputScalar::new(cfg.color_loss_weight, dtype),
        InputScalar::new(cfg.density_loss_weight, dtype),
        InputScalar::new(cfg.background_density_loss_weight, dtype),
        InputScalar::new(cfg.foreground_density_loss_weight, dtype),
        dtype.into(),
    );
    reduce_loss_kernel::launch(
        &client,
        batch_cube_count,
        batch_cube_dim,
        AddressType::U32,
        pixel_loss.clone().into_tensor_arg(),
        splat.clone().into_tensor_arg(),
        color.clone().into_tensor_arg(),
        density.clone().into_tensor_arg(),
        InputScalar::new(cfg.color_loss_weight, dtype),
        InputScalar::new(cfg.density_loss_weight, dtype),
        InputScalar::new(cfg.background_density_loss_weight, dtype),
        InputScalar::new(cfg.foreground_density_loss_weight, dtype),
        dtype.into(),
    );
    particle_adjoint_kernel::launch(
        &client,
        particle_cube_count,
        particle_cube_dim,
        AddressType::U32,
        centered_x.into_tensor_arg(),
        s.into_tensor_arg(),
        denominator.into_tensor_arg(),
        pixel_adjoint.into_tensor_arg(),
        pixel_size.into_tensor_arg(),
        target_points.into_tensor_arg(),
        position_grad.clone().into_tensor_arg(),
        state_grad.clone().into_tensor_arg(),
        cfg.image_size,
        InputScalar::new(cfg.sigma, dtype),
        InputScalar::new(cfg.lo, dtype),
        InputScalar::new(cfg.hi, dtype),
        dtype.into(),
    );

    Target2dAdjointRawOutput {
        position_grad,
        state_grad,
        splat,
        color,
        density,
    }
}

#[cube]
fn cube_l1l2<F: Float>(value: F) -> F {
    value.abs() + value * value
}

#[cube]
fn cube_l1l2_grad<F: Float>(value: F) -> F {
    let mut sign = F::new(0.0_f32);
    if value > F::new(0.0_f32) {
        sign = F::new(1.0_f32);
    }
    if value < F::new(0.0_f32) {
        sign = F::new(-1.0_f32);
    }
    sign + F::new(2.0_f32) * value
}

#[cube]
fn particle_pixel_params<F: Float>(
    x: F,
    y: F,
    pixel_size: F,
    #[comptime] image_size: usize,
    sigma_scale: F,
    lo: F,
    hi: F,
) -> (F, F, i32, i32, F, F, i32, F, F, F) {
    let size_f = F::cast_from(image_size as f32);
    let sigma = sigma_scale * size_f * pixel_size / (hi - lo);
    let radius = i32::cast_from((F::new(5.0_f32) * sigma).ceil().max(F::new(1.0_f32)));
    let px = (x - lo) / (hi - lo) * (size_f - F::new(1.0_f32));
    let py_unflipped = (y - lo) / (hi - lo) * (size_f - F::new(1.0_f32));
    let py = (size_f - F::new(1.0_f32)) - py_unflipped;
    let base_x_f = px.floor();
    let base_y_f = py.floor();
    let base_x = i32::cast_from(base_x_f);
    let base_y = i32::cast_from(base_y_f);
    let frac_x = px - base_x_f;
    let frac_y = py - base_y_f;
    let inv_two_sigma2 = F::new(1.0_f32) / (F::new(2.0_f32) * sigma * sigma);
    let inv_sigma2 = F::new(1.0_f32) / (sigma * sigma);
    let pixel_to_world = (size_f - F::new(1.0_f32)) / (hi - lo);
    (
        px,
        py,
        base_x,
        base_y,
        frac_x,
        frac_y,
        radius,
        inv_two_sigma2,
        inv_sigma2,
        pixel_to_world,
    )
}

#[cube]
fn sample_g<F: Float>(dx: F, dy: F, inv_two_sigma2: F) -> F {
    (F::new(0.0_f32) - (dx * dx + dy * dy) * inv_two_sigma2).exp()
}

#[cube(launch, address_type = "dynamic")]
fn denominator_kernel<F: Float>(
    centered_x: &Tensor<F>,
    pixel_size: &Tensor<F>,
    denominator: &mut Tensor<F>,
    #[comptime] image_size: usize,
    sigma_scale: InputScalar,
    lo: InputScalar,
    hi: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    if index >= denominator.len() {
        terminate!();
    }
    let particle_count = centered_x.shape(1);
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let x_base = batch * centered_x.stride(0) + particle * centered_x.stride(1);
    let pixel = pixel_size[batch * pixel_size.stride(0)];
    let (_, _, base_x, base_y, frac_x, frac_y, radius, inv_two_sigma2, _, _) =
        particle_pixel_params::<F>(
            centered_x[x_base],
            centered_x[x_base + centered_x.stride(2)],
            pixel,
            image_size,
            sigma_scale.get::<F>(),
            lo.get::<F>(),
            hi.get::<F>(),
        );
    let mut total = F::new(0.0_f32);
    let size_i = image_size as i32;
    let mut oy = 0 - radius;
    while oy <= radius {
        let y = base_y + oy;
        if y >= 0 && y < size_i {
            let mut ox = 0 - radius;
            while ox <= radius {
                let x = base_x + ox;
                if x >= 0 && x < size_i {
                    let dx = F::cast_from(ox as f32) - frac_x;
                    let dy = F::cast_from(oy as f32) - frac_y;
                    total += sample_g::<F>(dx, dy, inv_two_sigma2);
                }
                ox += 1;
            }
        }
        oy += 1;
    }
    denominator[index] = total + F::new(IMAGE_EPSILON);
}

#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn pixel_loss_kernel<F: Float>(
    centered_x: &Tensor<F>,
    s: &Tensor<F>,
    denominator: &Tensor<F>,
    target_rgb: &Tensor<F>,
    target_density: &Tensor<F>,
    target_foreground: &Tensor<F>,
    target_foreground_scale: &Tensor<F>,
    pixel_size: &Tensor<F>,
    target_points: &Tensor<F>,
    pixel_adjoint: &mut Tensor<F>,
    pixel_loss: &mut Tensor<F>,
    #[comptime] image_size: usize,
    sigma_scale: InputScalar,
    lo: InputScalar,
    hi: InputScalar,
    splat_loss_weight: InputScalar,
    color_loss_weight: InputScalar,
    density_loss_weight: InputScalar,
    background_density_loss_weight: InputScalar,
    foreground_density_loss_weight: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let pixels = image_size * image_size;
    if index >= centered_x.shape(0) * pixels {
        terminate!();
    }
    let batch = index / pixels;
    let pixel = index - batch * pixels;
    let px_i = pixel % image_size;
    let py_i = pixel / image_size;
    let particle_count = centered_x.shape(1);
    let state_dims = s.shape(2);
    let pixel_size_value = pixel_size[batch * pixel_size.stride(0)];
    let target_points_value = target_points[batch * target_points.stride(0)];
    let output_scale = target_points_value / F::cast_from(particle_count as f32);
    let size_f = F::cast_from(image_size as f32);
    let norm = size_f * pixel_size_value / (hi.get::<F>() - lo.get::<F>());
    let norm_scale = norm * norm * output_scale;
    let mut density = F::new(0.0_f32);
    let mut rgb0 = F::new(0.0_f32);
    let mut rgb1 = F::new(0.0_f32);
    let mut rgb2 = F::new(0.0_f32);

    let mut particle = 0usize;
    while particle < particle_count {
        let x_base = batch * centered_x.stride(0) + particle * centered_x.stride(1);
        let (_, _, base_x, base_y, frac_x, frac_y, radius, inv_two_sigma2, _, _) =
            particle_pixel_params::<F>(
                centered_x[x_base],
                centered_x[x_base + centered_x.stride(2)],
                pixel_size_value,
                image_size,
                sigma_scale.get::<F>(),
                lo.get::<F>(),
                hi.get::<F>(),
            );
        let ox = px_i as i32 - base_x;
        let oy = py_i as i32 - base_y;
        if ox >= 0 - radius && ox <= radius && oy >= 0 - radius && oy <= radius {
            let dx = F::cast_from(ox as f32) - frac_x;
            let dy = F::cast_from(oy as f32) - frac_y;
            let g = sample_g::<F>(dx, dy, inv_two_sigma2);
            let denom =
                denominator[batch * denominator.stride(0) + particle * denominator.stride(1)];
            let weight = norm_scale * g / denom;
            let state_base =
                batch * s.stride(0) + particle * s.stride(1) + (state_dims - 3) * s.stride(2);
            let c0 = s[state_base] + F::new(0.5_f32);
            let c1 = s[state_base + s.stride(2)] + F::new(0.5_f32);
            let c2 = s[state_base + 2 * s.stride(2)] + F::new(0.5_f32);
            density += weight;
            rgb0 += c0 * weight;
            rgb1 += c1 * weight;
            rgb2 += c2 * weight;
        }
        particle += 1;
    }

    let density_denom = F::cast_from(pixels as f32);
    let color_denom = F::cast_from((pixels * 3) as f32);
    let target_density_value =
        target_density[batch * target_density.stride(0) + pixel * target_density.stride(1)];
    let density_diff = density - target_density_value;
    let density_term = cube_l1l2::<F>(density_diff);
    let density_loss = density_term / density_denom;
    let fg = target_foreground
        [batch * target_foreground.stride(0) + pixel * target_foreground.stride(1)];
    let bg = F::new(1.0_f32) - fg;
    let leak = density * bg;
    let bg_loss = leak * leak / density_denom;
    let fg_scale = target_foreground_scale[batch * target_foreground_scale.stride(0)];
    let foreground_denom = density_denom / fg_scale.max(F::new(IMAGE_EPSILON));
    let fg_loss = density_term * fg / foreground_denom;
    let color_gate = (F::new(0.0_f32) - density_term).exp();

    let rgb_base = batch * target_rgb.stride(0) + pixel * target_rgb.stride(1);
    let rgb_diff0 = rgb0 - target_rgb[rgb_base];
    let rgb_diff1 = rgb1 - target_rgb[rgb_base + target_rgb.stride(2)];
    let rgb_diff2 = rgb2 - target_rgb[rgb_base + 2 * target_rgb.stride(2)];
    let color_loss = color_gate
        * (cube_l1l2::<F>(rgb_diff0) + cube_l1l2::<F>(rgb_diff1) + cube_l1l2::<F>(rgb_diff2))
        / color_denom;

    let splat_loss_weight = splat_loss_weight.get::<F>();
    let color_loss_weight = color_loss_weight.get::<F>();
    let density_loss_weight = density_loss_weight.get::<F>();
    let background_density_loss_weight = background_density_loss_weight.get::<F>();
    let foreground_density_loss_weight = foreground_density_loss_weight.get::<F>();
    let mut density_adj =
        splat_loss_weight * density_loss_weight * cube_l1l2_grad::<F>(density_diff) / density_denom;
    if background_density_loss_weight > F::new(0.0_f32) {
        density_adj +=
            splat_loss_weight * background_density_loss_weight * F::new(2.0_f32) * leak * bg
                / density_denom;
    }
    if foreground_density_loss_weight > F::new(0.0_f32) {
        density_adj += splat_loss_weight
            * foreground_density_loss_weight
            * fg
            * cube_l1l2_grad::<F>(density_diff)
            / foreground_denom;
    }
    let color_scale = splat_loss_weight * color_loss_weight * color_gate / color_denom;
    let adj_base = batch * pixel_adjoint.stride(0) + pixel * pixel_adjoint.stride(1);
    pixel_adjoint[adj_base] = color_scale * cube_l1l2_grad::<F>(rgb_diff0);
    pixel_adjoint[adj_base + pixel_adjoint.stride(2)] =
        color_scale * cube_l1l2_grad::<F>(rgb_diff1);
    pixel_adjoint[adj_base + 2 * pixel_adjoint.stride(2)] =
        color_scale * cube_l1l2_grad::<F>(rgb_diff2);
    pixel_adjoint[adj_base + 3 * pixel_adjoint.stride(2)] = density_adj;

    let loss_base = batch * pixel_loss.stride(0) + pixel * pixel_loss.stride(1);
    pixel_loss[loss_base] = color_loss;
    pixel_loss[loss_base + pixel_loss.stride(2)] = density_loss;
    pixel_loss[loss_base + 2 * pixel_loss.stride(2)] = bg_loss;
    pixel_loss[loss_base + 3 * pixel_loss.stride(2)] = fg_loss;
}

#[cube(launch, address_type = "dynamic")]
fn reduce_loss_kernel<F: Float>(
    pixel_loss: &Tensor<F>,
    splat: &mut Tensor<F>,
    color: &mut Tensor<F>,
    density: &mut Tensor<F>,
    color_loss_weight: InputScalar,
    density_loss_weight: InputScalar,
    background_density_loss_weight: InputScalar,
    foreground_density_loss_weight: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    let batch = ABSOLUTE_POS;
    if batch >= pixel_loss.shape(0) {
        terminate!();
    }
    let pixels = pixel_loss.shape(1);
    let mut color_sum = F::new(0.0_f32);
    let mut density_sum = F::new(0.0_f32);
    let mut background_sum = F::new(0.0_f32);
    let mut foreground_sum = F::new(0.0_f32);
    let mut pixel = 0usize;
    while pixel < pixels {
        let base = batch * pixel_loss.stride(0) + pixel * pixel_loss.stride(1);
        color_sum += pixel_loss[base];
        density_sum += pixel_loss[base + pixel_loss.stride(2)];
        background_sum += pixel_loss[base + 2 * pixel_loss.stride(2)];
        foreground_sum += pixel_loss[base + 3 * pixel_loss.stride(2)];
        pixel += 1;
    }
    color[batch] = color_sum;
    density[batch] = density_sum;
    splat[batch] = color_loss_weight.get::<F>() * color_sum
        + density_loss_weight.get::<F>() * density_sum
        + background_density_loss_weight.get::<F>() * background_sum
        + foreground_density_loss_weight.get::<F>() * foreground_sum;
}

#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn particle_adjoint_kernel<F: Float>(
    centered_x: &Tensor<F>,
    s: &Tensor<F>,
    denominator: &Tensor<F>,
    pixel_adjoint: &Tensor<F>,
    pixel_size: &Tensor<F>,
    target_points: &Tensor<F>,
    position_grad: &mut Tensor<F>,
    state_grad: &mut Tensor<F>,
    #[comptime] image_size: usize,
    sigma_scale: InputScalar,
    lo: InputScalar,
    hi: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let particle_count = centered_x.shape(1);
    if index >= centered_x.shape(0) * particle_count {
        terminate!();
    }
    let batch = index / particle_count;
    let particle = index - batch * particle_count;
    let x_base = batch * centered_x.stride(0) + particle * centered_x.stride(1);
    let pixel_size_value = pixel_size[batch * pixel_size.stride(0)];
    let target_points_value = target_points[batch * target_points.stride(0)];
    let output_scale = target_points_value / F::cast_from(particle_count as f32);
    let size_f = F::cast_from(image_size as f32);
    let norm = size_f * pixel_size_value / (hi.get::<F>() - lo.get::<F>());
    let norm_scale = norm * norm * output_scale;
    let (_, _, base_x, base_y, frac_x, frac_y, radius, inv_two_sigma2, inv_sigma2, pixel_to_world) =
        particle_pixel_params::<F>(
            centered_x[x_base],
            centered_x[x_base + centered_x.stride(2)],
            pixel_size_value,
            image_size,
            sigma_scale.get::<F>(),
            lo.get::<F>(),
            hi.get::<F>(),
        );
    let denom = denominator[batch * denominator.stride(0) + particle * denominator.stride(1)];
    let state_dims = s.shape(2);
    let state_base = batch * s.stride(0) + particle * s.stride(1) + (state_dims - 3) * s.stride(2);
    let c0 = s[state_base] + F::new(0.5_f32);
    let c1 = s[state_base + s.stride(2)] + F::new(0.5_f32);
    let c2 = s[state_base + 2 * s.stride(2)] + F::new(0.5_f32);
    let size_i = image_size as i32;
    let mut weighted_adjoint_sum = F::new(0.0_f32);
    let mut color_grad0 = F::new(0.0_f32);
    let mut color_grad1 = F::new(0.0_f32);
    let mut color_grad2 = F::new(0.0_f32);
    let mut oy = 0 - radius;
    while oy <= radius {
        let y = base_y + oy;
        if y >= 0 && y < size_i {
            let mut ox = 0 - radius;
            while ox <= radius {
                let x = base_x + ox;
                if x >= 0 && x < size_i {
                    let dx = F::cast_from(ox as f32) - frac_x;
                    let dy = F::cast_from(oy as f32) - frac_y;
                    let g = sample_g::<F>(dx, dy, inv_two_sigma2);
                    let pixel = y as usize * image_size + x as usize;
                    let adj_base =
                        batch * pixel_adjoint.stride(0) + pixel * pixel_adjoint.stride(1);
                    let rgb_adj0 = pixel_adjoint[adj_base];
                    let rgb_adj1 = pixel_adjoint[adj_base + pixel_adjoint.stride(2)];
                    let rgb_adj2 = pixel_adjoint[adj_base + 2 * pixel_adjoint.stride(2)];
                    let density_adj = pixel_adjoint[adj_base + 3 * pixel_adjoint.stride(2)];
                    color_grad0 += rgb_adj0 * norm_scale * g / denom;
                    color_grad1 += rgb_adj1 * norm_scale * g / denom;
                    color_grad2 += rgb_adj2 * norm_scale * g / denom;
                    let weight_adj = density_adj + rgb_adj0 * c0 + rgb_adj1 * c1 + rgb_adj2 * c2;
                    weighted_adjoint_sum += weight_adj * g;
                }
                ox += 1;
            }
        }
        oy += 1;
    }

    let mut pix_grad_x = F::new(0.0_f32);
    let mut pix_grad_y = F::new(0.0_f32);
    let mut oy2 = 0 - radius;
    while oy2 <= radius {
        let y = base_y + oy2;
        if y >= 0 && y < size_i {
            let mut ox2 = 0 - radius;
            while ox2 <= radius {
                let x = base_x + ox2;
                if x >= 0 && x < size_i {
                    let dx = F::cast_from(ox2 as f32) - frac_x;
                    let dy = F::cast_from(oy2 as f32) - frac_y;
                    let g = sample_g::<F>(dx, dy, inv_two_sigma2);
                    let pixel = y as usize * image_size + x as usize;
                    let adj_base =
                        batch * pixel_adjoint.stride(0) + pixel * pixel_adjoint.stride(1);
                    let rgb_adj0 = pixel_adjoint[adj_base];
                    let rgb_adj1 = pixel_adjoint[adj_base + pixel_adjoint.stride(2)];
                    let rgb_adj2 = pixel_adjoint[adj_base + 2 * pixel_adjoint.stride(2)];
                    let density_adj = pixel_adjoint[adj_base + 3 * pixel_adjoint.stride(2)];
                    let weight_adj = density_adj + rgb_adj0 * c0 + rgb_adj1 * c1 + rgb_adj2 * c2;
                    let g_adj =
                        norm_scale * (weight_adj / denom - weighted_adjoint_sum / (denom * denom));
                    let g_pos = g_adj * g * inv_sigma2;
                    pix_grad_x += g_pos * dx;
                    pix_grad_y += g_pos * dy;
                }
                ox2 += 1;
            }
        }
        oy2 += 1;
    }

    let pos_base = batch * position_grad.stride(0) + particle * position_grad.stride(1);
    position_grad[pos_base] = pix_grad_x * pixel_to_world;
    position_grad[pos_base + position_grad.stride(2)] =
        F::new(0.0_f32) - pix_grad_y * pixel_to_world;
    let grad_base = batch * state_grad.stride(0)
        + particle * state_grad.stride(1)
        + (state_dims - 3) * state_grad.stride(2);
    state_grad[grad_base] = color_grad0;
    state_grad[grad_base + state_grad.stride(2)] = color_grad1;
    state_grad[grad_base + 2 * state_grad.stride(2)] = color_grad2;
}
