//! Command dispatch and command-local option validation.

mod basic;
mod bench_handlers;
mod dispatch;
mod dynamics2d;
mod hyper;
mod hyper_e2e;
mod hyper_support;
mod mesh;
mod options;
mod render;
mod reporting;
mod target2d;
mod training_bench;

pub(crate) use dispatch::run_command;
pub(crate) use options::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};
