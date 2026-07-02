//! Command dispatch and command-local option validation.

mod basic;
mod bench_handlers;
mod dispatch;
mod mesh;
mod options;
mod render;

pub(crate) use dispatch::run_command;
pub(crate) use options::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};
