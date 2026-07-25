#![allow(clippy::too_many_arguments)]

use std::borrow::Cow;

use burn_automata_kernels::{AdaptivePerceptionConfig, AdaptiveSupportBins, HashGridConfig};

use crate::{AutomataError, AutomataResult, NpaModel};

use super::helpers::*;
use super::types::*;

mod active_quadrature;
mod blocking_step;
mod coupled_fine;
mod device;
mod diagnostics;
pub(crate) use diagnostics::WgpuPendingAdaptiveDiagnostics;
mod gaussian;
mod maintenance;
mod passes;
mod persistent_modes;
mod readback;
mod state;
mod steps;
mod subgroup;

use subgroup::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutorProfile {
    Full,
    RestrictionSubgroup,
}

fn required_pipeline<'a>(
    pipeline: &'a Option<wgpu::ComputePipeline>,
    name: &str,
) -> AutomataResult<&'a wgpu::ComputePipeline> {
    pipeline.as_ref().ok_or_else(|| {
        AutomataError::InvalidArgument(format!(
            "WGPU executor was initialized without the {name} pipeline"
        ))
    })
}
