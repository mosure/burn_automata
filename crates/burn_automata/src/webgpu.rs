//! Browser-specific Burn/WGPU initialization.

use std::cell::Cell;

thread_local! {
    static INITIALIZED: Cell<bool> = const { Cell::new(false) };
}

/// Initializes the default Burn WebGPU device without blocking the browser
/// event loop.
///
/// CubeCL cannot create a browser WebGPU adapter synchronously. Browser
/// entrypoints must await this once per WebAssembly module instance before
/// constructing WGPU tensors.
#[cfg(all(target_arch = "wasm32", feature = "backend_wgpu"))]
pub async fn initialize_webgpu_backend() {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::WebGpu, init_setup_async};

    if INITIALIZED.get() {
        return;
    }
    init_setup_async::<WebGpu>(&WgpuDevice::DefaultDevice, RuntimeOptions::default()).await;
    INITIALIZED.set(true);
}
