use clap::Parser;

use super::{args::CliArgs, commands};

#[cfg(test)]
pub(crate) use commands::{resolve_direct_selection_seed_training, resolve_full_coverage_adjoint};

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    commands::run_command(args.command)
}
