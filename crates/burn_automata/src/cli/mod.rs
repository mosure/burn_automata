//! Command-line interface implementation.
//!
//! Keep the binary target as a thin launcher. The command parser, dispatch, and
//! training/evaluation helpers live in normal Rust modules here so they can be
//! checked and tested without source splicing.

mod app;
mod args;
mod bench;
mod commands;
mod growth_validation;
mod mesh_training;
mod prelude;
mod render_training;
mod reports;
mod targets;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}

#[cfg(test)]
mod tests;
