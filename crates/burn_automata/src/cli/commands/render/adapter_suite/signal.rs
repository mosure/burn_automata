use crate::cli::prelude::*;

pub(super) fn adapter_suite_missing_train_signal(
    shared_base_training: &[CliRenderAdapterSuiteBaseEntry],
    entries: &[CliRenderAdapterSuiteEntry],
) -> Vec<CliRenderAdapterSuiteTrainingSignalGap> {
    let mut missing = Vec::new();
    for entry in shared_base_training {
        let rounds = render_proxy_missing_signal_rounds(&entry.report);
        if !rounds.is_empty() {
            missing.push(CliRenderAdapterSuiteTrainingSignalGap {
                phase: CliRenderAdapterSuiteTrainingPhase::SharedBase,
                cycle: Some(entry.cycle),
                target: entry.target,
                rounds,
            });
        }
    }
    for entry in entries {
        let rounds = render_proxy_missing_signal_rounds(&entry.report);
        if !rounds.is_empty() {
            missing.push(CliRenderAdapterSuiteTrainingSignalGap {
                phase: CliRenderAdapterSuiteTrainingPhase::Adapter,
                cycle: None,
                target: entry.target,
                rounds,
            });
        }
    }
    missing
}

pub(super) fn adapter_suite_missing_signal_labels(
    missing: &[CliRenderAdapterSuiteTrainingSignalGap],
) -> Vec<String> {
    missing
        .iter()
        .map(|entry| {
            let phase = match entry.phase {
                CliRenderAdapterSuiteTrainingPhase::SharedBase => "shared-base",
                CliRenderAdapterSuiteTrainingPhase::Adapter => "adapter",
            };
            match entry.cycle {
                Some(cycle) => format!(
                    "{phase}:cycle={cycle}:target={}:rounds={:?}",
                    mesh_target_slug(entry.target),
                    entry.rounds
                ),
                None => format!(
                    "{phase}:target={}:rounds={:?}",
                    mesh_target_slug(entry.target),
                    entry.rounds
                ),
            }
        })
        .collect()
}
