// The shared backend implementation must clone devices because WGPU and CUDA
// devices are not Copy; the ndarray test device happens to be Copy.
#![allow(clippy::clone_on_copy)]

dense_backend_impl!(
    burn::backend::NdArray<f32>,
    "burn_ndarray_autodiff_dense_direct_basis",
    "ndarray-default",
    "burn-ndarray"
);
