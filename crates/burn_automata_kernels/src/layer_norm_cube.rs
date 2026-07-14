//! Fused CubeCL modulated layer-normalization forward and adjoint kernels.

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
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{KernelError, KernelResult};

const FORWARD_OP: &str = "burn_automata.modulated_layer_norm.forward.v1";
const BACKWARD_OP: &str = "burn_automata.modulated_layer_norm.backward.v1";
const ROW_THREADS: u32 = 256;

pub struct ModulatedLayerNormCubeForwardOutput<B: BurnBackendTrait> {
    pub output: BurnTensor<B, 3>,
    pub stats: BurnTensor<B, 3>,
}

pub struct ModulatedLayerNormCubeBackwardOutput<B: BurnBackendTrait> {
    pub input_grad: BurnTensor<B, 3>,
    pub shift_grad: BurnTensor<B, 2>,
    pub scale_grad: BurnTensor<B, 2>,
}

#[allow(unused_variables)]
pub trait ModulatedLayerNormCubeBackend: BurnBackendTrait + Sized {
    fn modulated_layer_norm_cube_forward(
        input: BurnTensor<Self, 3>,
        shift: BurnTensor<Self, 2>,
        scale: BurnTensor<Self, 2>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeForwardOutput<Self>>> {
        None
    }

    fn modulated_layer_norm_cube_backward(
        input: BurnTensor<Self, 3>,
        scale: BurnTensor<Self, 2>,
        output_grad: BurnTensor<Self, 3>,
        stats: BurnTensor<Self, 3>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeBackwardOutput<Self>>> {
        None
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl ModulatedLayerNormCubeBackend for burn::backend::Wgpu<f32> {
    fn modulated_layer_norm_cube_forward(
        input: BurnTensor<Self, 3>,
        shift: BurnTensor<Self, 2>,
        scale: BurnTensor<Self, 2>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeForwardOutput<Self>>> {
        Some(forward_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(input, shift, scale))
    }

    fn modulated_layer_norm_cube_backward(
        input: BurnTensor<Self, 3>,
        scale: BurnTensor<Self, 2>,
        output_grad: BurnTensor<Self, 3>,
        stats: BurnTensor<Self, 3>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeBackwardOutput<Self>>> {
        Some(backward_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(input, scale, output_grad, stats))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl ModulatedLayerNormCubeBackend for burn::backend::Cuda<f32> {
    fn modulated_layer_norm_cube_forward(
        input: BurnTensor<Self, 3>,
        shift: BurnTensor<Self, 2>,
        scale: BurnTensor<Self, 2>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeForwardOutput<Self>>> {
        Some(forward_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(input, shift, scale))
    }

    fn modulated_layer_norm_cube_backward(
        input: BurnTensor<Self, 3>,
        scale: BurnTensor<Self, 2>,
        output_grad: BurnTensor<Self, 3>,
        stats: BurnTensor<Self, 3>,
    ) -> Option<KernelResult<ModulatedLayerNormCubeBackwardOutput<Self>>> {
        Some(backward_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(input, scale, output_grad, stats))
    }
}

fn validate_forward_shapes(
    input: &[usize; 3],
    shift: &[usize; 2],
    scale: &[usize; 2],
) -> KernelResult<()> {
    if input[2] == 0 || *shift != [input[0], input[2]] || *scale != [input[0], input[2]] {
        return Err(KernelError::InvalidArgument(format!(
            "modulated layer norm expects input [batch, rows, dims] and shift/scale [batch, dims], got input={input:?} shift={shift:?} scale={scale:?}"
        )));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn forward_fusion<R, F, I, BT>(
    input: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    shift: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 2>,
    scale: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 2>,
) -> KernelResult<ModulatedLayerNormCubeForwardOutput<Fusion<CubeBackend<R, F, I, BT>>>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let input_dims = input.shape().dims::<3>();
    let shift_dims = shift.shape().dims::<2>();
    let scale_dims = scale.shape().dims::<2>();
    validate_forward_shapes(&input_dims, &shift_dims, &scale_dims)?;

    let input_fusion = input.into_primitive().tensor();
    let shift_fusion = shift.into_primitive().tensor();
    let scale_fusion = scale.into_primitive().tensor();
    let client = input_fusion.client.clone();
    let dtype = input_fusion.dtype;
    let output_ir = TensorIr::uninit(client.create_empty_handle(), Shape::new(input_dims), dtype);
    let stats_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([input_dims[0], input_dims[1], 2]),
        dtype,
    );
    let inputs = [
        input_fusion.clone().into_ir(),
        shift_fusion.clone().into_ir(),
        scale_fusion.clone().into_ir(),
    ];
    let outputs = [output_ir.clone(), stats_ir.clone()];
    let streams = OperationStreams::with_inputs([&input_fusion, &shift_fusion, &scale_fusion]);
    let op = LayerNormForwardFusionOp::<R, F, I, BT> {
        desc: LayerNormForwardDesc {
            input: inputs[0].clone(),
            shift: inputs[1].clone(),
            scale: inputs[2].clone(),
            output: output_ir,
            stats: stats_ir,
        },
        _marker: PhantomData,
    };
    let [output, stats] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(FORWARD_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<2>();
    Ok(ModulatedLayerNormCubeForwardOutput {
        output: BurnTensor::from_primitive(TensorPrimitive::Float(output)),
        stats: BurnTensor::from_primitive(TensorPrimitive::Float(stats)),
    })
}

#[allow(clippy::type_complexity)]
fn backward_fusion<R, F, I, BT>(
    input: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    scale: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 2>,
    output_grad: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
    stats: BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, 3>,
) -> KernelResult<ModulatedLayerNormCubeBackwardOutput<Fusion<CubeBackend<R, F, I, BT>>>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let input_dims = input.shape().dims::<3>();
    let scale_dims = scale.shape().dims::<2>();
    let output_grad_dims = output_grad.shape().dims::<3>();
    let stats_dims = stats.shape().dims::<3>();
    validate_forward_shapes(&input_dims, &scale_dims, &scale_dims)?;
    if output_grad_dims != input_dims || stats_dims != [input_dims[0], input_dims[1], 2] {
        return Err(KernelError::InvalidArgument(format!(
            "modulated layer norm backward shape mismatch: input={input_dims:?} output_grad={output_grad_dims:?} stats={stats_dims:?}"
        )));
    }

    let input_fusion = input.into_primitive().tensor();
    let scale_fusion = scale.into_primitive().tensor();
    let output_grad_fusion = output_grad.into_primitive().tensor();
    let stats_fusion = stats.into_primitive().tensor();
    let client = input_fusion.client.clone();
    let dtype = input_fusion.dtype;
    let input_grad_ir =
        TensorIr::uninit(client.create_empty_handle(), Shape::new(input_dims), dtype);
    let shift_grad_ir =
        TensorIr::uninit(client.create_empty_handle(), Shape::new(scale_dims), dtype);
    let scale_grad_ir =
        TensorIr::uninit(client.create_empty_handle(), Shape::new(scale_dims), dtype);
    let inputs = [
        input_fusion.clone().into_ir(),
        scale_fusion.clone().into_ir(),
        output_grad_fusion.clone().into_ir(),
        stats_fusion.clone().into_ir(),
    ];
    let outputs = [
        input_grad_ir.clone(),
        shift_grad_ir.clone(),
        scale_grad_ir.clone(),
    ];
    let streams = OperationStreams::with_inputs([
        &input_fusion,
        &scale_fusion,
        &output_grad_fusion,
        &stats_fusion,
    ]);
    let op = LayerNormBackwardFusionOp::<R, F, I, BT> {
        desc: LayerNormBackwardDesc {
            input: inputs[0].clone(),
            scale: inputs[1].clone(),
            output_grad: inputs[2].clone(),
            stats: inputs[3].clone(),
            input_grad: input_grad_ir,
            shift_grad: shift_grad_ir,
            scale_grad: scale_grad_ir,
        },
        _marker: PhantomData,
    };
    let [input_grad, shift_grad, scale_grad] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(BACKWARD_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<3>();
    Ok(ModulatedLayerNormCubeBackwardOutput {
        input_grad: BurnTensor::from_primitive(TensorPrimitive::Float(input_grad)),
        shift_grad: BurnTensor::from_primitive(TensorPrimitive::Float(shift_grad)),
        scale_grad: BurnTensor::from_primitive(TensorPrimitive::Float(scale_grad)),
    })
}

#[derive(Clone, Debug)]
struct LayerNormForwardDesc {
    input: TensorIr,
    shift: TensorIr,
    scale: TensorIr,
    output: TensorIr,
    stats: TensorIr,
}

#[derive(Debug)]
struct LayerNormForwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: LayerNormForwardDesc,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for LayerNormForwardFusionOp<R, F, I, BT>
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
        let input = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.input);
        let shift = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.shift);
        let scale = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.scale);
        let (output, stats) = launch_forward(input, shift, scale);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.output.id, output);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.stats.id, stats);
    }
}

#[derive(Clone, Debug)]
struct LayerNormBackwardDesc {
    input: TensorIr,
    scale: TensorIr,
    output_grad: TensorIr,
    stats: TensorIr,
    input_grad: TensorIr,
    shift_grad: TensorIr,
    scale_grad: TensorIr,
}

#[derive(Debug)]
struct LayerNormBackwardFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: LayerNormBackwardDesc,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for LayerNormBackwardFusionOp<R, F, I, BT>
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
        let input = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.input);
        let scale = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.scale);
        let output_grad = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.output_grad);
        let stats = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.stats);
        let (input_grad, shift_grad, scale_grad) =
            launch_backward(input, scale, output_grad, stats);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.input_grad.id, input_grad);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.shift_grad.id, shift_grad);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.scale_grad.id, scale_grad);
    }
}

fn launch_forward<R: CubeRuntime>(
    input: CubeTensor<R>,
    shift: CubeTensor<R>,
    scale: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let [batches, rows, _] = input.shape().dims::<3>();
    let dtype = input.dtype;
    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new(input.shape().dims::<3>()),
        dtype,
    );
    let stats = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([batches, rows, 2]),
        dtype,
    );
    modulated_layer_norm_forward_kernel::launch(
        &client,
        CubeCount::Static((batches * rows) as u32, 1, 1),
        CubeDim::new_1d(ROW_THREADS),
        AddressType::U32,
        input.into_tensor_arg(),
        shift.into_tensor_arg(),
        scale.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        dtype.into(),
    );
    (output, stats)
}

fn launch_backward<R: CubeRuntime>(
    input: CubeTensor<R>,
    scale: CubeTensor<R>,
    output_grad: CubeTensor<R>,
    stats: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let [batches, rows, dims] = input.shape().dims::<3>();
    let dtype = input.dtype;
    let client = input.client.clone();
    let device = input.device.clone();
    let input_grad = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new(input.shape().dims::<3>()),
        dtype,
    );
    let shift_grad = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([batches, dims]),
        dtype,
    );
    let scale_grad = empty_device_dtype(client.clone(), device, Shape::new([batches, dims]), dtype);
    modulated_layer_norm_input_adjoint_kernel::launch(
        &client,
        CubeCount::Static((batches * rows) as u32, 1, 1),
        CubeDim::new_1d(ROW_THREADS),
        AddressType::U32,
        input.clone().into_tensor_arg(),
        scale.clone().into_tensor_arg(),
        output_grad.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        input_grad.clone().into_tensor_arg(),
        dtype.into(),
    );
    let affine_units = batches * dims;
    let affine_cube_dim = CubeDim::new(&client, affine_units);
    modulated_layer_norm_affine_adjoint_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, affine_units, affine_cube_dim),
        affine_cube_dim,
        AddressType::U32,
        input.into_tensor_arg(),
        output_grad.into_tensor_arg(),
        stats.into_tensor_arg(),
        shift_grad.clone().into_tensor_arg(),
        scale_grad.clone().into_tensor_arg(),
        dtype.into(),
    );
    (input_grad, shift_grad, scale_grad)
}

#[cube]
fn reduce_row_pair<F: Float>(lhs: F, rhs: F, shared: &mut SharedMemory<F>) {
    let plane = UNIT_POS_X / PLANE_DIM;
    let plane_count = CUBE_DIM_X.div_ceil(PLANE_DIM);
    let lhs = plane_sum(lhs);
    let rhs = plane_sum(rhs);
    if UNIT_POS_PLANE == 0 {
        shared[plane as usize] = lhs;
        shared[16usize + plane as usize] = rhs;
    }
    sync_cube();
    if plane == 0 {
        let mut lhs = F::new(0.0_f32);
        let mut rhs = F::new(0.0_f32);
        if UNIT_POS_PLANE < plane_count {
            lhs = shared[UNIT_POS_PLANE as usize];
            rhs = shared[16usize + UNIT_POS_PLANE as usize];
        }
        lhs = plane_sum(lhs);
        rhs = plane_sum(rhs);
        if UNIT_POS_PLANE == 0 {
            shared[0] = lhs;
            shared[16] = rhs;
        }
    }
    sync_cube();
}

#[cube(launch, address_type = "dynamic")]
fn modulated_layer_norm_forward_kernel<F: Float>(
    input: &Tensor<F>,
    shift: &Tensor<F>,
    scale: &Tensor<F>,
    output: &mut Tensor<F>,
    stats: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let row_index = CUBE_POS_X as usize;
    let rows = input.shape(1);
    let dims = input.shape(2);
    let batch = row_index / rows;
    let row = row_index - batch * rows;
    let input_base = batch * input.stride(0) + row * input.stride(1);
    let mut sum = F::new(0.0_f32);
    let mut square_sum = F::new(0.0_f32);
    let mut dim = UNIT_POS_X as usize;
    while dim < dims {
        let value = input[input_base + dim * input.stride(2)];
        sum += value;
        square_sum += value * value;
        dim += CUBE_DIM_X as usize;
    }
    let mut shared = SharedMemory::<F>::new(32usize);
    reduce_row_pair(sum, square_sum, &mut shared);
    let dims_f = F::cast_from(dims as f32);
    let mean = shared[0] / dims_f;
    let variance = (shared[16] / dims_f - mean * mean).max(F::new(0.0_f32));
    let inv_std = F::new(1.0_f32) / (variance + F::new(1.0e-6_f32)).sqrt();
    if UNIT_POS_X == 0 {
        let stats_base = batch * stats.stride(0) + row * stats.stride(1);
        stats[stats_base] = mean;
        stats[stats_base + stats.stride(2)] = inv_std;
    }
    dim = UNIT_POS_X as usize;
    while dim < dims {
        let value = input[input_base + dim * input.stride(2)];
        let affine = batch * shift.stride(0) + dim * shift.stride(1);
        let normalized = (value - mean) * inv_std;
        let output_base = batch * output.stride(0) + row * output.stride(1);
        output[output_base + dim * output.stride(2)] =
            normalized * (F::new(1.0_f32) + scale[affine]) + shift[affine];
        dim += CUBE_DIM_X as usize;
    }
}

#[cube(launch, address_type = "dynamic")]
fn modulated_layer_norm_input_adjoint_kernel<F: Float>(
    input: &Tensor<F>,
    scale: &Tensor<F>,
    output_grad: &Tensor<F>,
    stats: &Tensor<F>,
    input_grad: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let row_index = CUBE_POS_X as usize;
    let rows = input.shape(1);
    let dims = input.shape(2);
    let batch = row_index / rows;
    let row = row_index - batch * rows;
    let input_base = batch * input.stride(0) + row * input.stride(1);
    let grad_base = batch * output_grad.stride(0) + row * output_grad.stride(1);
    let stats_base = batch * stats.stride(0) + row * stats.stride(1);
    let mean = stats[stats_base];
    let inv_std = stats[stats_base + stats.stride(2)];
    let mut grad_sum = F::new(0.0_f32);
    let mut grad_normalized_sum = F::new(0.0_f32);
    let mut dim = UNIT_POS_X as usize;
    while dim < dims {
        let affine = batch * scale.stride(0) + dim * scale.stride(1);
        let normalized = (input[input_base + dim * input.stride(2)] - mean) * inv_std;
        let grad = output_grad[grad_base + dim * output_grad.stride(2)]
            * (F::new(1.0_f32) + scale[affine]);
        grad_sum += grad;
        grad_normalized_sum += grad * normalized;
        dim += CUBE_DIM_X as usize;
    }
    let mut shared = SharedMemory::<F>::new(32usize);
    reduce_row_pair(grad_sum, grad_normalized_sum, &mut shared);
    let grad_mean = shared[0] / F::cast_from(dims as f32);
    let grad_normalized_mean = shared[16] / F::cast_from(dims as f32);
    dim = UNIT_POS_X as usize;
    while dim < dims {
        let affine = batch * scale.stride(0) + dim * scale.stride(1);
        let normalized = (input[input_base + dim * input.stride(2)] - mean) * inv_std;
        let grad = output_grad[grad_base + dim * output_grad.stride(2)]
            * (F::new(1.0_f32) + scale[affine]);
        let output_base = batch * input_grad.stride(0) + row * input_grad.stride(1);
        input_grad[output_base + dim * input_grad.stride(2)] =
            inv_std * (grad - grad_mean - normalized * grad_normalized_mean);
        dim += CUBE_DIM_X as usize;
    }
}

#[cube(launch, address_type = "dynamic")]
fn modulated_layer_norm_affine_adjoint_kernel<F: Float>(
    input: &Tensor<F>,
    output_grad: &Tensor<F>,
    stats: &Tensor<F>,
    shift_grad: &mut Tensor<F>,
    scale_grad: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    let dims = input.shape(2);
    if index >= input.shape(0) * dims {
        terminate!();
    }
    let batch = index / dims;
    let dim = index - batch * dims;
    let mut shift_sum = F::new(0.0_f32);
    let mut scale_sum = F::new(0.0_f32);
    let mut row = 0usize;
    while row < input.shape(1) {
        let input_index = batch * input.stride(0) + row * input.stride(1) + dim * input.stride(2);
        let grad_index = batch * output_grad.stride(0)
            + row * output_grad.stride(1)
            + dim * output_grad.stride(2);
        let stats_base = batch * stats.stride(0) + row * stats.stride(1);
        let normalized =
            (input[input_index] - stats[stats_base]) * stats[stats_base + stats.stride(2)];
        let grad = output_grad[grad_index];
        shift_sum += grad;
        scale_sum += grad * normalized;
        row += 1;
    }
    let affine_index = batch * shift_grad.stride(0) + dim * shift_grad.stride(1);
    shift_grad[affine_index] = shift_sum;
    scale_grad[affine_index] = scale_sum;
}
