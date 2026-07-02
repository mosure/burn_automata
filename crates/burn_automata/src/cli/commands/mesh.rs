//! Mesh-oriented CLI command handlers.

mod ablate;
mod teapot;
mod torus;

pub(crate) use ablate::run_ablate_local_3d;
pub(crate) use teapot::run_train_teapot_morphogen_3d;
pub(crate) use torus::{run_train_torus_3d, run_train_torus_morphogen_3d};
