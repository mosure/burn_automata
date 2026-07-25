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
use cubecl::prelude::*;

use crate::{KernelError, KernelResult};

const ADAPTIVE_MERGE_COST_OP: &str = "burn_automata.adaptive.merge_cost.v1";
const PRIMITIVES_PER_GROUP: usize = 5;
const PRIMITIVE_FIELDS: usize = 10;
const REDUCTION_UNITS: usize = 256;

type FusedCubeTensor<R, F, I, BT, const D: usize> = BurnTensor<Fusion<CubeBackend<R, F, I, BT>>, D>;

/// Device implementation for scoring every four-child-to-parent replacement.
///
/// `baseline` and `target` use `[batch, pixel, density/r/g/b]`. `primitives` uses
/// `[batch, group, primitive, cx/cy/inv_x2/inv_y2/sin/cos/signed_weight/r/g/b]`.
pub trait AdaptiveMergeCostCubeBackend: BurnBackendTrait + Sized {
    fn adaptive_merge_cost_cube(
        baseline: BurnTensor<Self, 3>,
        target: BurnTensor<Self, 3>,
        primitives: BurnTensor<Self, 4>,
        image_size: usize,
    ) -> Option<KernelResult<BurnTensor<Self, 2>>> {
        let _ = (baseline, target, primitives, image_size);
        None
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl AdaptiveMergeCostCubeBackend for burn::backend::Wgpu<f32> {
    fn adaptive_merge_cost_cube(
        baseline: BurnTensor<Self, 3>,
        target: BurnTensor<Self, 3>,
        primitives: BurnTensor<Self, 4>,
        image_size: usize,
    ) -> Option<KernelResult<BurnTensor<Self, 2>>> {
        Some(adaptive_merge_cost_cube_fusion::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(baseline, target, primitives, image_size))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl AdaptiveMergeCostCubeBackend for burn::backend::Cuda<f32> {
    fn adaptive_merge_cost_cube(
        baseline: BurnTensor<Self, 3>,
        target: BurnTensor<Self, 3>,
        primitives: BurnTensor<Self, 4>,
        image_size: usize,
    ) -> Option<KernelResult<BurnTensor<Self, 2>>> {
        Some(adaptive_merge_cost_cube_fusion::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(baseline, target, primitives, image_size))
    }
}

fn adaptive_merge_cost_cube_fusion<R, F, I, BT>(
    baseline: FusedCubeTensor<R, F, I, BT, 3>,
    target: FusedCubeTensor<R, F, I, BT, 3>,
    primitives: FusedCubeTensor<R, F, I, BT, 4>,
    image_size: usize,
) -> KernelResult<FusedCubeTensor<R, F, I, BT, 2>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let baseline_dims = baseline.shape().dims::<3>();
    let target_dims = target.shape().dims::<3>();
    let primitive_dims = primitives.shape().dims::<4>();
    let pixels = image_size.checked_mul(image_size).ok_or_else(|| {
        KernelError::InvalidArgument("adaptive merge-cost image size overflow".to_string())
    })?;
    if image_size == 0
        || baseline_dims[0] == 0
        || baseline_dims[1] != pixels
        || baseline_dims[2] != 4
        || target_dims != baseline_dims
        || primitive_dims[0] != baseline_dims[0]
        || primitive_dims[1] == 0
        || primitive_dims[2] != PRIMITIVES_PER_GROUP
        || primitive_dims[3] != PRIMITIVE_FIELDS
    {
        return Err(KernelError::InvalidArgument(format!(
            "adaptive merge-cost shapes must be baseline/target [batch, {pixels}, 4] and primitives [batch, groups, {PRIMITIVES_PER_GROUP}, {PRIMITIVE_FIELDS}], got {baseline_dims:?}, {target_dims:?}, {primitive_dims:?}",
        )));
    }

    let batch = primitive_dims[0];
    let groups = primitive_dims[1];
    let baseline_fusion = baseline.into_primitive().tensor();
    let target_fusion = target.into_primitive().tensor();
    let primitives_fusion = primitives.into_primitive().tensor();
    let client = baseline_fusion.client.clone();
    let dtype = baseline_fusion.dtype;
    let costs_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, groups]),
        dtype,
    );
    let inputs = [
        baseline_fusion.clone().into_ir(),
        target_fusion.clone().into_ir(),
        primitives_fusion.clone().into_ir(),
    ];
    let outputs = [costs_ir.clone()];
    let streams =
        OperationStreams::with_inputs([&baseline_fusion, &target_fusion, &primitives_fusion]);
    let op = AdaptiveMergeCostFusionOp::<R, F, I, BT> {
        desc: AdaptiveMergeCostDesc {
            baseline: inputs[0].clone(),
            target: inputs[1].clone(),
            primitives: inputs[2].clone(),
            costs: costs_ir,
        },
        image_size,
        _marker: PhantomData,
    };
    let [costs] = client
        .register(
            streams,
            OperationIr::Custom(CustomOpIr::new(ADAPTIVE_MERGE_COST_OP, &inputs, &outputs)),
            op,
        )
        .outputs::<1>();
    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(costs)))
}

#[derive(Clone, Debug)]
struct AdaptiveMergeCostDesc {
    baseline: TensorIr,
    target: TensorIr,
    primitives: TensorIr,
    costs: TensorIr,
}

#[derive(Debug)]
struct AdaptiveMergeCostFusionOp<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    desc: AdaptiveMergeCostDesc,
    image_size: usize,
    _marker: PhantomData<(R, F, I, BT)>,
}

impl<R, F, I, BT> Operation<burn_cubecl::fusion::FusionCubeRuntime<R>>
    for AdaptiveMergeCostFusionOp<R, F, I, BT>
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
        let baseline = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.baseline);
        let target = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.target);
        let primitives = handles.get_float_tensor::<Raw<R, F, I, BT>>(&self.desc.primitives);
        let costs = launch_adaptive_merge_cost(baseline, target, primitives, self.image_size);
        handles.register_float_tensor::<Raw<R, F, I, BT>>(&self.desc.costs.id, costs);
    }
}

fn launch_adaptive_merge_cost<R: CubeRuntime>(
    baseline: CubeTensor<R>,
    target: CubeTensor<R>,
    primitives: CubeTensor<R>,
    image_size: usize,
) -> CubeTensor<R> {
    let primitive_dims = primitives.shape().dims::<4>();
    let batch = primitive_dims[0];
    let groups = primitive_dims[1];
    let pixels = image_size * image_size;
    let dtype = baseline.dtype;
    let client = baseline.client.clone();
    let costs = empty_device_dtype(
        client.clone(),
        baseline.device.clone(),
        Shape::new([batch, groups]),
        dtype,
    );
    adaptive_merge_cost_kernel::launch(
        &client,
        CubeCount::Static(groups as u32, batch as u32, 1),
        CubeDim::new_1d(REDUCTION_UNITS as u32),
        AddressType::U32,
        baseline.into_tensor_arg(),
        target.into_tensor_arg(),
        primitives.into_tensor_arg(),
        costs.clone().into_tensor_arg(),
        image_size,
        InputScalar::new(1.0 / (pixels * 3) as f32, dtype),
        dtype.into(),
    );
    costs
}

#[cube(launch, address_type = "dynamic")]
fn adaptive_merge_cost_kernel<F: Float>(
    baseline: &Tensor<F>,
    target: &Tensor<F>,
    primitives: &Tensor<F>,
    costs: &mut Tensor<F>,
    #[comptime] image_size: usize,
    normalization: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    let group = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let unit = UNIT_POS as usize;
    if batch >= primitives.shape(0) || group >= primitives.shape(1) {
        terminate!();
    }

    let pixels = image_size * image_size;
    let mut squared_error = F::new(0.0_f32);
    let mut pixel = unit;
    while pixel < pixels {
        let pixel_x = F::cast_from((pixel % image_size) as f32);
        let pixel_y = F::cast_from((pixel / image_size) as f32);
        let baseline_base = batch * baseline.stride(0) + pixel * baseline.stride(1);
        let target_base = batch * target.stride(0) + pixel * target.stride(1);
        let density = RuntimeCell::<F>::new(baseline[baseline_base]);
        let rgb0 = RuntimeCell::<F>::new(baseline[baseline_base + baseline.stride(2)]);
        let rgb1 = RuntimeCell::<F>::new(baseline[baseline_base + 2usize * baseline.stride(2)]);
        let rgb2 = RuntimeCell::<F>::new(baseline[baseline_base + 3usize * baseline.stride(2)]);

        #[unroll]
        for primitive in 0usize..PRIMITIVES_PER_GROUP {
            let base = batch * primitives.stride(0)
                + group * primitives.stride(1)
                + primitive * primitives.stride(2);
            let field_stride = primitives.stride(3);
            let dx = pixel_x - primitives[base];
            let dy = pixel_y - primitives[base + field_stride];
            let inverse_sigma_x2 = primitives[base + 2usize * field_stride];
            let inverse_sigma_y2 = primitives[base + 3usize * field_stride];
            let sin = primitives[base + 4usize * field_stride];
            let cos = primitives[base + 5usize * field_stride];
            let major = cos * dx + sin * dy;
            let minor = cos * dy - sin * dx;
            let exponent = major * major * inverse_sigma_x2 + minor * minor * inverse_sigma_y2;
            if exponent <= F::new(25.0_f32) {
                let weight =
                    (F::new(-0.5_f32) * exponent).exp() * primitives[base + 6usize * field_stride];
                density.store(density.read() + weight);
                rgb0.store(rgb0.read() + weight * primitives[base + 7usize * field_stride]);
                rgb1.store(rgb1.read() + weight * primitives[base + 8usize * field_stride]);
                rgb2.store(rgb2.read() + weight * primitives[base + 9usize * field_stride]);
            }
        }

        let density = density.consume().clamp(F::new(0.0_f32), F::new(1.0_f32));
        let rgb0 = rgb0.consume();
        let rgb1 = rgb1.consume();
        let rgb2 = rgb2.consume();
        let target_density = target[target_base].clamp(F::new(0.0_f32), F::new(1.0_f32));
        let value0 = (rgb0 + F::new(1.0_f32) - density).clamp(F::new(0.0_f32), F::new(1.0_f32));
        let value1 = (rgb1 + F::new(1.0_f32) - density).clamp(F::new(0.0_f32), F::new(1.0_f32));
        let value2 = (rgb2 + F::new(1.0_f32) - density).clamp(F::new(0.0_f32), F::new(1.0_f32));
        let target0 = (target[target_base + target.stride(2)] + F::new(1.0_f32) - target_density)
            .clamp(F::new(0.0_f32), F::new(1.0_f32));
        let target1 = (target[target_base + 2usize * target.stride(2)] + F::new(1.0_f32)
            - target_density)
            .clamp(F::new(0.0_f32), F::new(1.0_f32));
        let target2 = (target[target_base + 3usize * target.stride(2)] + F::new(1.0_f32)
            - target_density)
            .clamp(F::new(0.0_f32), F::new(1.0_f32));
        let difference0 = value0 - target0;
        let difference1 = value1 - target1;
        let difference2 = value2 - target2;
        squared_error +=
            difference0 * difference0 + difference1 * difference1 + difference2 * difference2;
        pixel += REDUCTION_UNITS;
    }

    let mut shared = SharedMemory::<F>::new(REDUCTION_UNITS);
    shared[unit] = squared_error;
    sync_cube();
    let mut stride = 128usize;
    while stride > 0usize {
        if unit < stride {
            let other = shared[unit + stride];
            shared[unit] += other;
        }
        sync_cube();
        stride /= 2usize;
    }
    if unit == 0usize {
        costs[batch * costs.stride(0) + group * costs.stride(1)] =
            shared[0] * normalization.get::<F>();
    }
}
