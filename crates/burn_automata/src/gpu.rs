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

pub(crate) use executor::WgpuPendingAdaptiveDiagnostics;
pub use helpers::GAUSSIAN_SH_COEFF_COUNT;
pub use types::{
    WGPU_MATERIAL_UPDATE_MASK_MEMBERS, WgpuAutomataExecutor, WgpuAutomataState,
    WgpuGaussianBindGroup, WgpuGaussianBufferRefs, WgpuGaussianReadback, WgpuMaterialStateInit,
    WgpuMaterialUpdateMask, WgpuNeighborMode, WgpuNeighborReport, WgpuOwnedGaussianBuffers,
    WgpuStepOutput, WgpuSupportBinConfig,
};
pub(crate) use types::{
    WgpuActiveQuadratureProlongation, WgpuAdaptiveDiagnostics, WgpuAdaptiveLocalRuleMode,
    WgpuCoupledFineSnapshot, WgpuPersistentModeRestriction, WgpuTeacherSnapshot,
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
