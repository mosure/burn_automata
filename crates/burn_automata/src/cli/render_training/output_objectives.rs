#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

mod liveness;
pub(crate) use liveness::*;
mod color;
pub(crate) use color::*;
mod motion;
pub(crate) use motion::*;
mod memory;
pub(crate) use memory::*;
mod visibility;
pub(crate) use visibility::*;
mod scale_budget;
pub(crate) use scale_budget::*;
