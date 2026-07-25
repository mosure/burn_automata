use super::{
    adaptive, adaptive_target2d, basic, bench_handlers, dynamics2d, hyper, hyper_e2e, mesh,
    npa2d_parity, render, reporting, target2d, training_bench,
};
use crate::cli::prelude::*;

pub(crate) fn run_command(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        command @ Command::Infer { .. } => basic::run_infer(command),
        command @ Command::Train { .. } => basic::run_train(command),
        command @ Command::EvalTarget2d { .. } => target2d::run_eval_target_2d(command),
        command @ Command::ValidateNpa2dParity { .. } => {
            npa2d_parity::run_validate_npa_2d_parity(command)
        }
        command @ Command::AdaptiveNpa { .. } => adaptive::run_adaptive_npa(command),
        command @ Command::TrainAdaptiveTarget2d { .. } => {
            adaptive_target2d::run_train_adaptive_target2d(command)
        }
        command @ Command::EvalAdaptiveTarget2d { .. } => {
            adaptive_target2d::run_eval_adaptive_target2d(command)
        }
        command @ Command::AuditAdaptiveTopology { .. } => {
            adaptive::run_audit_adaptive_topology(command)
        }
        command @ Command::AuditAdaptiveClosure { .. } => {
            adaptive::run_audit_adaptive_closure(command)
        }
        command @ Command::EvalAdaptiveNpa { .. } => adaptive::run_eval_adaptive_npa(command),
        command @ Command::TrainTarget2d { .. } => {
            warn_legacy_2d("train-target2d");
            target2d::run_train_target_2d(command)
        }
        command @ Command::EvalDynamics2d { .. } => dynamics2d::run_eval_dynamics_2d(command),
        command @ Command::TrainHyper2d { .. } => {
            warn_legacy_2d("train-hyper2d");
            hyper::run_train_hyper_2d(command)
        }
        command @ Command::TrainHyper2dE2e { .. } => {
            warn_legacy_2d("train-hyper2d-e2e");
            hyper_e2e::run_train_hyper_2d_e2e(command)
        }
        command @ Command::TrainHyper2dE2eRollout { .. } => {
            hyper_e2e::run_train_hyper_2d_e2e_rollout(command)
        }
        command @ Command::TrainHyper2dDirectBasis { .. } => {
            warn_legacy_2d("train-hyper2d-direct-basis");
            hyper_e2e::run_train_hyper_2d_direct_basis(command)
        }
        command @ Command::TrainHyper2dAdapterBank { .. } => {
            warn_legacy_2d("train-hyper2d-adapter-bank");
            hyper_e2e::run_train_hyper_2d_adapter_bank(command)
        }
        command @ Command::ValidateHyper2dDirectBasisOracles { .. } => {
            warn_legacy_2d("validate-hyper2d-direct-basis-oracles");
            hyper_e2e::run_validate_hyper_2d_direct_basis_oracles(command)
        }
        command @ Command::ValidateHyper2dPsnrGate { .. } => {
            warn_legacy_2d("validate-hyper2d-psnr-gate");
            hyper_e2e::run_validate_hyper_2d_psnr_gate(command)
        }
        command @ Command::ReportHyper2d { .. } => reporting::run_report_hyper_2d(command),
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
        command @ Command::MaterializeAdapter { .. } => basic::run_materialize_adapter(command),
        command @ Command::BuildExactAdapterBank { .. } => {
            basic::run_build_exact_adapter_bank(command)
        }
        command @ Command::Manifest { .. } => basic::run_manifest(command),
    }
}

fn warn_legacy_2d(command: &str) {
    eprintln!(
        "warning: {command} is a hidden legacy 2D diagnostic. Use train-hyper2d-e2e-rollout / train-hypernpa2d for maintained online HyperNPA training, and validate-npa2d-parity for upstream parity gates."
    );
}
