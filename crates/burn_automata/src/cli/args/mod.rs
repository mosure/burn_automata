//! CLI argument declarations.
//!
//! The command tree is intentionally separate from clap value enums. Command
//! dispatch and tests can import the parsed surface without depending on the
//! binary entrypoint.

mod commands;
mod values;

pub(crate) use commands::{CliArgs, Command};
pub(crate) use values::*;
