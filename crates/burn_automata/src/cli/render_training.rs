use super::prelude::*;

pub(crate) const DIRECT_LOCAL_FRONT_EXPANSION_GAIN_FRACTION: f32 = 0.20;
const TEMPORAL_FRONT_CANDIDATE_SMALL_ROW_FRACTION: usize = 4;
const TEMPORAL_FRONT_CANDIDATE_ROW_FRACTION: usize = 2;
const TEMPORAL_FRONT_CANDIDATE_WIDE_MIN_ROWS: usize = 128;
const TEMPORAL_FRONT_CANDIDATE_MIN_CAP: usize = 32;
const TEMPORAL_FRONT_CANDIDATE_MAX_CAP: usize = 4096;
const TEMPORAL_NONLOCAL_LIVENESS_SUPPRESSION_GAIN_FRACTION: f32 = 0.35;

mod adjoints;
mod catalog_validation;
mod config;
mod direct_loop;
mod geometry_updates;
mod gradients;
mod gradients_direct;
mod loss_eval;
mod objective_diagnostics;
mod output_objectives;
mod selection;
mod selection_cases;
mod training_loop;

pub(crate) use adjoints::*;
pub(crate) use catalog_validation::*;
pub(crate) use config::*;
pub(crate) use direct_loop::*;
pub(crate) use geometry_updates::*;
pub(crate) use gradients::*;
pub(crate) use gradients_direct::*;
pub(crate) use loss_eval::*;
pub(crate) use objective_diagnostics::*;
pub(crate) use output_objectives::*;
pub(crate) use selection::*;
pub(crate) use selection_cases::*;
pub(crate) use training_loop::*;
