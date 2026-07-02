use crate::cli::prelude::*;

pub(super) fn adapter_suite_bank_entries(
    entries: &[CliRenderAdapterSuiteEntry],
) -> Vec<CliRenderAdapterBankEntry> {
    entries
        .iter()
        .map(|entry| CliRenderAdapterBankEntry {
            target: entry.target,
            split: entry.split,
            adapter_output: entry.adapter_output.clone(),
            materialized_model_output: entry.materialized_model_output.clone(),
            seed_scale: entry.seed_scale,
            seed_mode: entry.seed_mode,
            strict_passed: entry.strict_gate_summary.strict_passed,
            strict_score: entry.strict_gate_summary.strict_score,
            render_total_loss: entry.final_render_loss.total_loss,
            density_psnr_db: entry.final_render_loss.density_psnr_db,
        })
        .collect()
}
