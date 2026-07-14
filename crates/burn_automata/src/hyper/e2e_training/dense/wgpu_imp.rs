// The shared trainer modules are deliberately monomorphized once per Burn backend.
#![allow(clippy::duplicate_mod)]

dense_backend_impl!(
    burn::backend::Wgpu<f32>,
    "burn_wgpu_autodiff_dense_direct_basis",
    "wgpu-default",
    "burn-wgpu"
);
