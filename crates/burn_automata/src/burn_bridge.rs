//! Small Burn tensor helpers.
//!
//! The CPU rollout path in this first implementation is intentionally plain
//! Rust. These helpers keep the public library Burn-oriented and give future
//! WGPU/CubeCL kernels a stable shape contract.

use burn::tensor::{Tensor, TensorData, backend::Backend};

pub fn positions_to_tensor<B: Backend>(positions: &[[f32; 4]], device: &B::Device) -> Tensor<B, 2> {
    let flat = positions
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(flat, [positions.len(), 4]), device)
}

pub fn states_to_tensor<B: Backend>(
    states: &[f32],
    rows: usize,
    state_dims: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(states.to_vec(), [rows, state_dims]), device)
}
