#![cfg(feature = "gpu_wgpu")]

#[path = "gpu_wgpu/adaptive.rs"]
mod adaptive;
#[path = "gpu_wgpu/common.rs"]
mod common;
#[path = "gpu_wgpu/gaussian.rs"]
mod gaussian;
#[path = "gpu_wgpu/neighbors.rs"]
mod neighbors;
#[path = "gpu_wgpu/parity.rs"]
mod parity;
#[path = "gpu_wgpu/persistent.rs"]
mod persistent;
#[path = "gpu_wgpu/updates.rs"]
mod updates;
