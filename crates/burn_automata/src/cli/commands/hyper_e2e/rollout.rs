use crate::cli::prelude::*;

pub(crate) fn run_train_hyper_2d_e2e_rollout(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainHyper2dE2eRollout { config } = command else {
        unreachable!("run_train_hyper_2d_e2e_rollout called with wrong command variant");
    };
    crate::hyper::e2e_rollout::run_train_hyper_2d_e2e_rollout_config_path(config)
}
