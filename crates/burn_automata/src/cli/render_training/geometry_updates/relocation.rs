#![allow(
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use super::*;

mod gap;
pub(crate) use gap::*;
mod normal;
pub(crate) use normal::*;
mod strata;
pub(crate) use strata::*;
mod tangent;
pub(crate) use tangent::*;
