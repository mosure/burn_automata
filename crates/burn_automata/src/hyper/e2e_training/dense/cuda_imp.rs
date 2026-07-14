// The shared trainer modules are deliberately monomorphized once per Burn backend.
#![allow(clippy::duplicate_mod)]

dense_backend_impl!(
    burn::backend::Cuda<f32>,
    "burn_cuda_autodiff_dense_direct_basis",
    "cuda-default",
    "burn-cuda"
);
