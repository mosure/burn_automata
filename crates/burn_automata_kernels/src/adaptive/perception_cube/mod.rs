//! Backend-neutral CubeCL implementation of adaptive NPA-compatible perception.
//!
//! The prepared forward retains state-independent density and correction data so
//! TBPTT backward does not rebuild them or cross a host boundary.

mod fusion;
mod grid;
mod kernels;

use burn::tensor::{Tensor as BurnTensor, backend::Backend as BurnBackendTrait};

use super::{AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig, AdaptivePerceptionSemantics};
use crate::KernelResult;

pub struct AdaptiveNpaPerceptionCubeForwardOutput<B: BurnBackendTrait> {
    pub features: BurnTensor<B, 3>,
    pub density: BurnTensor<B, 2>,
    pub coarse_density: BurnTensor<B, 2>,
    pub raw_state_gradient: BurnTensor<B, 4>,
    pub state_gradient_inverse: BurnTensor<B, 3>,
}

pub struct AdaptiveNpaPerceptionCubeAdjointOutput<B: BurnBackendTrait> {
    pub state_grad: BurnTensor<B, 3>,
}

#[allow(unused_variables)]
pub trait AdaptiveNpaPerceptionCubeBackend: BurnBackendTrait + Sized {
    #[allow(clippy::too_many_arguments)]
    fn adaptive_npa_perception_cube_forward(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeForwardOutput<Self>>> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn adaptive_npa_perception_cube_state_adjoint(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeAdjointOutput<Self>>> {
        None
    }
}

#[cfg(feature = "cubecl_wgpu")]
impl AdaptiveNpaPerceptionCubeBackend for burn::backend::Wgpu<f32> {
    fn adaptive_npa_perception_cube_forward(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeForwardOutput<Self>>> {
        Some(fusion::forward::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(
            positions,
            states,
            represented_measure,
            bandwidth,
            config,
            options,
            semantics,
        ))
    }

    fn adaptive_npa_perception_cube_state_adjoint(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeAdjointOutput<Self>>> {
        Some(fusion::state_adjoint::<
            burn_cubecl::cubecl::wgpu::WgpuRuntime,
            f32,
            i32,
            u32,
        >(
            positions,
            states,
            represented_measure,
            bandwidth,
            feature_grad,
            density,
            raw_state_gradient,
            state_gradient_inverse,
            config,
            options,
            semantics,
        ))
    }
}

#[cfg(feature = "cubecl_cuda")]
impl AdaptiveNpaPerceptionCubeBackend for burn::backend::Cuda<f32> {
    fn adaptive_npa_perception_cube_forward(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeForwardOutput<Self>>> {
        Some(fusion::forward::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(
            positions,
            states,
            represented_measure,
            bandwidth,
            config,
            options,
            semantics,
        ))
    }

    fn adaptive_npa_perception_cube_state_adjoint(
        positions: BurnTensor<Self, 3>,
        states: BurnTensor<Self, 3>,
        represented_measure: BurnTensor<Self, 2>,
        bandwidth: BurnTensor<Self, 2>,
        feature_grad: BurnTensor<Self, 3>,
        density: BurnTensor<Self, 2>,
        raw_state_gradient: BurnTensor<Self, 4>,
        state_gradient_inverse: BurnTensor<Self, 3>,
        config: AdaptivePerceptionConfig,
        options: AdaptiveNpaPerceptionOptions,
        semantics: AdaptivePerceptionSemantics,
    ) -> Option<KernelResult<AdaptiveNpaPerceptionCubeAdjointOutput<Self>>> {
        Some(fusion::state_adjoint::<
            burn_cubecl::cubecl::cuda::CudaRuntime,
            f32,
            i32,
            u8,
        >(
            positions,
            states,
            represented_measure,
            bandwidth,
            feature_grad,
            density,
            raw_state_gradient,
            state_gradient_inverse,
            config,
            options,
            semantics,
        ))
    }
}
