//! Budgeted adaptive NPA reference kernels.
//!
//! These kernels deliberately do not reuse the fixed-support SPH implementation:
//! represented measure, interaction bandwidth, and material footprint have
//! different semantics. The deterministic implementation is the oracle for the
//! device path and for manufactured operator tests.

mod compatible;
mod config;
mod dual;
#[cfg(feature = "cubecl")]
mod merge_oracle_cube;
mod perception;
#[cfg(feature = "cubecl")]
mod perception_cube;
mod proxy;
mod scale_bins;
mod state_adjoint;
mod types;

pub const COUPLED_FINE_RECENTER_WGSL: &str = include_str!("coupled_recenter.wgsl");
pub const ACTIVE_QUADRATURE_BLEND_WGSL: &str = include_str!("active_quadrature_blend.wgsl");
pub const ACTIVE_QUADRATURE_PROLONG_WGSL: &str = include_str!("active_quadrature_prolong.wgsl");
pub const PERSISTENT_MODE_RESTRICT_WGSL: &str = include_str!("persistent_restrict.wgsl");
pub const PAIRED_LOCAL_DETAIL_TOPOLOGY_WGSL: &str =
    include_str!("paired_local_detail_topology.wgsl");

pub use compatible::{
    adaptive_npa_perceive, adaptive_npa_perceive_all_pairs, adaptive_npa_perceive_without_spacing,
};
pub use config::{
    AdaptiveGraphPolicy, AdaptiveNpaPerceptionOptions, AdaptivePerceptionConfig,
    AdaptivePerceptionSemantics,
};
pub use dual::adaptive_perceive_pair;
#[cfg(feature = "cubecl")]
pub use merge_oracle_cube::AdaptiveMergeCostCubeBackend;
pub use perception::{
    adaptive_perceive, adaptive_perceive_all_pairs, adaptive_perceive_without_spacing,
};
#[cfg(feature = "cubecl")]
pub use perception_cube::{
    AdaptiveNpaPerceptionCubeAdjointOutput, AdaptiveNpaPerceptionCubeBackend,
    AdaptiveNpaPerceptionCubeForwardOutput,
};
pub use proxy::adaptive_proxy_perceive;
pub use scale_bins::{AdaptiveSupportBins, MAX_ADAPTIVE_SUPPORT_BINS};
pub use state_adjoint::{
    adaptive_npa_perceive_state_adjoint, adaptive_npa_perceive_state_adjoint_all_pairs,
    adaptive_perceive_state_adjoint, adaptive_perceive_state_adjoint_all_pairs,
};
pub use types::{AdaptiveGraphMetrics, AdaptivePerceptionOutput, AdaptivePerceptionPair};

#[cfg(test)]
mod tests;
