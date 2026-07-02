//! Direct WGPU inference kernels for Neural Particle Automata.
//!
//! The current path builds a linked-cell hashgrid with GPU atomics, then keeps
//! density, SPH perception, MLP update, and Euler integration on WGPU storage
//! buffers. The convenience API reads outputs back for parity tests and CLI
//! reporting.

use burn_automata_kernels::HashGridConfig;

use crate::{AutomataResult, NpaModel};

mod executor;
mod helpers;
#[cfg(test)]
mod tests;
mod types;

pub use helpers::GAUSSIAN_SH_COEFF_COUNT;
pub use types::{
    WgpuAutomataExecutor, WgpuAutomataState, WgpuGaussianBindGroup, WgpuGaussianBufferRefs,
    WgpuGaussianReadback, WgpuNeighborMode, WgpuNeighborReport, WgpuOwnedGaussianBuffers,
    WgpuStepOutput,
};

#[allow(clippy::too_many_arguments)]
pub fn step_wgpu_blocking(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
) -> AutomataResult<WgpuStepOutput> {
    pollster::block_on(step_wgpu(
        model,
        positions,
        states,
        batch_size,
        particle_count,
        grid,
        dt,
    ))
}

#[allow(clippy::too_many_arguments)]
pub async fn step_wgpu(
    model: &NpaModel,
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    grid: &HashGridConfig,
    dt: f32,
) -> AutomataResult<WgpuStepOutput> {
    let executor = WgpuAutomataExecutor::new().await?;
    executor.step(
        model,
        positions,
        states,
        batch_size,
        particle_count,
        grid,
        dt,
    )
}
