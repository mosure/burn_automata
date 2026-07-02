#![allow(clippy::too_many_arguments)]

use std::borrow::Cow;

use burn_automata_kernels::HashGridConfig;

use crate::{AutomataError, AutomataResult, NpaModel};

use super::helpers::*;
use super::types::*;

mod blocking_step;
mod device;
mod gaussian;
mod maintenance;
mod passes;
mod readback;
mod state;
mod steps;
mod subgroup;

use subgroup::*;
