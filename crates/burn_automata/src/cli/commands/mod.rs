//! Command dispatch and command-local option validation.

mod dispatch;
mod options;

pub(crate) use dispatch::run_command;
pub(crate) use options::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};
