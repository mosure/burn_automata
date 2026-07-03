use super::{basic, bench_handlers, dynamics2d, hyper, mesh, render, target2d, training_bench};
use crate::cli::prelude::*;

pub(crate) fn run_command(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        command @ Command::Infer { .. } => basic::run_infer(command),
        command @ Command::Train { .. } => basic::run_train(command),
        command @ Command::EvalTarget2d { .. } => target2d::run_eval_target_2d(command),
        command @ Command::TrainTarget2d { .. } => target2d::run_train_target_2d(command),
        command @ Command::EvalDynamics2d { .. } => dynamics2d::run_eval_dynamics_2d(command),
        command @ Command::TrainHyper2d { .. } => hyper::run_train_hyper_2d(command),
        command @ Command::InferHyper2d { .. } => hyper::run_infer_hyper_2d(command),
        command @ Command::EvalHyper2d { .. } => hyper::run_eval_hyper_2d(command),
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
        command @ Command::TrainRender3dAdapters { .. } => {
            render::run_train_render_3d_adapters(command)
        }
        command @ Command::Import { .. } => basic::run_import(command),
        command @ Command::Bench { .. } => bench_handlers::run_bench(command),
        command @ Command::BenchTraining { .. } => training_bench::run_bench_training(command),
        command @ Command::BenchSpatial { .. } => bench_handlers::run_bench_spatial(command),
        command @ Command::Manifest { .. } => basic::run_manifest(command),
    }
}
