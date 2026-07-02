#![allow(clippy::too_many_arguments)]

use super::prelude::*;

const GROWTH_3D_MIN_SURFACE_NORMAL_BIN_FRACTION: f32 = 0.65;
const GROWTH_3D_MIN_SURFACE_NORMAL_MEAN_BIN_COVERAGE: f32 = 0.45;
const GROWTH_3D_MIN_BBOX_DIAGONAL_RATIO: f32 = 0.20;
const GROWTH_3D_MIN_AXIS_EXTENT_RATIO: f32 = 0.05;
pub(crate) const GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION: f32 = 0.75;
pub(crate) const GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE: f32 = 0.45;

mod coverage;
pub(crate) use coverage::*;
mod strict;
pub(crate) use strict::*;
mod validation;
pub(crate) use validation::*;
mod robustness;
pub(crate) use robustness::*;
mod dynamics;
pub(crate) use dynamics::*;
mod state_metrics;
pub(crate) use state_metrics::*;
mod rollout_cases;
pub(crate) use rollout_cases::*;
