use super::{basic, bench_handlers, mesh, render};
use crate::cli::prelude::*;

pub(crate) fn run_command(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        command @ Command::Infer { .. } => basic::run_infer(command),
        command @ Command::Train { .. } => basic::run_train(command),
        command @ Command::TrainTorus3d { .. } => mesh::run_train_torus_3d(command),
        command @ Command::TrainTorusMorphogen3d { .. } => {
            mesh::run_train_torus_morphogen_3d(command)
        }
        command @ Command::TrainTeapotMorphogen3d { .. } => {
            mesh::run_train_teapot_morphogen_3d(command)
        }
        command @ Command::AblateLocal3d { .. } => mesh::run_ablate_local_3d(command),
        command @ Command::RenderLoss3d { .. } => render::run_render_loss_3d(command),
        command @ Command::ValidateGrowth3d { .. } => render::run_validate_growth_3d(command),
        command @ Command::RetimeGrowth3d { .. } => render::run_retime_growth_3d(command),
        command @ Command::TrainRender3d { .. } => render::run_train_render_3d(command),
        command @ Command::Import { .. } => basic::run_import(command),
        command @ Command::Bench { .. } => bench_handlers::run_bench(command),
        command @ Command::BenchSpatial { .. } => bench_handlers::run_bench_spatial(command),
        command @ Command::Manifest { .. } => basic::run_manifest(command),
    }
}
