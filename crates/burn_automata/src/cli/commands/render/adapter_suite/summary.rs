use crate::cli::prelude::*;

pub(super) fn adapter_suite_shared_base_summary(
    evaluations: &[CliRenderAdapterSuiteBaseEvalEntry],
) -> CliRenderAdapterSuiteValidationSummary {
    adapter_suite_validation_summary(evaluations.iter().map(|entry| &entry.strict_gate_summary))
}

pub(super) fn adapter_suite_adapter_summary(
    entries: &[CliRenderAdapterSuiteEntry],
) -> CliRenderAdapterSuiteValidationSummary {
    adapter_suite_validation_summary(entries.iter().map(|entry| &entry.strict_gate_summary))
}

pub(super) fn adapter_suite_split_summaries(
    shared_base_evaluations: &[CliRenderAdapterSuiteBaseEvalEntry],
    entries: &[CliRenderAdapterSuiteEntry],
) -> Vec<CliRenderAdapterSuiteSplitSummary> {
    [
        CliRenderAdapterSuiteSplit::SharedBaseTrain,
        CliRenderAdapterSuiteSplit::HoldoutAdapterOnly,
    ]
    .into_iter()
    .map(|split| CliRenderAdapterSuiteSplitSummary {
        split,
        shared_base: adapter_suite_validation_summary(
            shared_base_evaluations
                .iter()
                .filter(move |entry| entry.split == split)
                .map(|entry| &entry.strict_gate_summary),
        ),
        adapted: adapter_suite_validation_summary(
            entries
                .iter()
                .filter(move |entry| entry.split == split)
                .map(|entry| &entry.strict_gate_summary),
        ),
    })
    .collect()
}

fn adapter_suite_validation_summary<'a>(
    summaries: impl IntoIterator<Item = &'a CliRenderTrainingGateSummary>,
) -> CliRenderAdapterSuiteValidationSummary {
    let mut stats = ValidationStats::default();
    for summary in summaries {
        stats.push(summary);
    }
    stats.finish()
}

#[derive(Default)]
struct ValidationStats {
    target_count: usize,
    strict_pass_count: usize,
    gate_pass_count: usize,
    catalog_sanity_pass_count: usize,
    strict_score_sum: f32,
    max_strict_score: f32,
    render_loss_sum: f32,
    max_render_loss: f32,
    density_psnr_sum: f32,
    min_density_psnr: f32,
    color_psnr_sum: f32,
    depth_psnr_sum: f32,
    active_delta_sum: isize,
    min_active_delta: isize,
    newly_activated_sum: f32,
    min_newly_activated: f32,
    all_local_conditionless_lineage: bool,
    all_target_seed_conditionless_lineage: bool,
    all_object_agnostic_growth_seed_mode: bool,
    all_target_growth_seed_mode: bool,
    all_no_seed_coordinate_scaffold: bool,
}

impl ValidationStats {
    fn push(&mut self, summary: &CliRenderTrainingGateSummary) {
        if self.target_count == 0 {
            self.max_strict_score = f32::NEG_INFINITY;
            self.max_render_loss = f32::NEG_INFINITY;
            self.min_density_psnr = f32::INFINITY;
            self.min_active_delta = isize::MAX;
            self.min_newly_activated = f32::INFINITY;
            self.all_local_conditionless_lineage = true;
            self.all_target_seed_conditionless_lineage = true;
            self.all_object_agnostic_growth_seed_mode = true;
            self.all_target_growth_seed_mode = true;
            self.all_no_seed_coordinate_scaffold = true;
        }

        self.target_count += 1;
        self.strict_pass_count += usize::from(summary.strict_passed);
        self.gate_pass_count += usize::from(summary.gate_passed);
        self.catalog_sanity_pass_count += usize::from(summary.catalog_sanity_passed);
        self.strict_score_sum += summary.strict_score;
        self.max_strict_score = self.max_strict_score.max(summary.strict_score);
        self.render_loss_sum += summary.render_total_loss;
        self.max_render_loss = self.max_render_loss.max(summary.render_total_loss);
        self.density_psnr_sum += summary.render_density_psnr_db;
        self.min_density_psnr = self.min_density_psnr.min(summary.render_density_psnr_db);
        self.color_psnr_sum += summary.render_color_psnr_db;
        self.depth_psnr_sum += summary.render_depth_psnr_db;
        self.active_delta_sum += summary.active_count_delta;
        self.min_active_delta = self.min_active_delta.min(summary.active_count_delta);
        self.newly_activated_sum += summary.newly_activated_fraction;
        self.min_newly_activated = self
            .min_newly_activated
            .min(summary.newly_activated_fraction);
        self.all_local_conditionless_lineage &= summary.local_conditionless_lineage;
        self.all_target_seed_conditionless_lineage &= summary.target_seed_conditionless_lineage;
        self.all_object_agnostic_growth_seed_mode &= summary.object_agnostic_growth_seed_mode;
        self.all_target_growth_seed_mode &= summary.target_growth_seed_mode;
        self.all_no_seed_coordinate_scaffold &= summary.no_seed_coordinate_scaffold;
    }

    fn finish(self) -> CliRenderAdapterSuiteValidationSummary {
        if self.target_count == 0 {
            return CliRenderAdapterSuiteValidationSummary {
                target_count: 0,
                strict_pass_count: 0,
                gate_pass_count: 0,
                catalog_sanity_pass_count: 0,
                strict_pass_rate: 0.0,
                gate_pass_rate: 0.0,
                catalog_sanity_pass_rate: 0.0,
                mean_strict_score: 0.0,
                max_strict_score: 0.0,
                mean_render_loss: 0.0,
                max_render_loss: 0.0,
                mean_density_psnr_db: 0.0,
                min_density_psnr_db: 0.0,
                mean_color_psnr_db: 0.0,
                mean_depth_psnr_db: 0.0,
                mean_active_count_delta: 0.0,
                min_active_count_delta: 0,
                mean_newly_activated_fraction: 0.0,
                min_newly_activated_fraction: 0.0,
                all_local_conditionless_lineage: true,
                all_target_seed_conditionless_lineage: true,
                all_object_agnostic_growth_seed_mode: true,
                all_target_growth_seed_mode: true,
                all_no_seed_coordinate_scaffold: true,
            };
        }

        let target_count_f = self.target_count as f32;
        CliRenderAdapterSuiteValidationSummary {
            target_count: self.target_count,
            strict_pass_count: self.strict_pass_count,
            gate_pass_count: self.gate_pass_count,
            catalog_sanity_pass_count: self.catalog_sanity_pass_count,
            strict_pass_rate: self.strict_pass_count as f32 / target_count_f,
            gate_pass_rate: self.gate_pass_count as f32 / target_count_f,
            catalog_sanity_pass_rate: self.catalog_sanity_pass_count as f32 / target_count_f,
            mean_strict_score: self.strict_score_sum / target_count_f,
            max_strict_score: self.max_strict_score,
            mean_render_loss: self.render_loss_sum / target_count_f,
            max_render_loss: self.max_render_loss,
            mean_density_psnr_db: self.density_psnr_sum / target_count_f,
            min_density_psnr_db: self.min_density_psnr,
            mean_color_psnr_db: self.color_psnr_sum / target_count_f,
            mean_depth_psnr_db: self.depth_psnr_sum / target_count_f,
            mean_active_count_delta: self.active_delta_sum as f32 / target_count_f,
            min_active_count_delta: self.min_active_delta,
            mean_newly_activated_fraction: self.newly_activated_sum / target_count_f,
            min_newly_activated_fraction: self.min_newly_activated,
            all_local_conditionless_lineage: self.all_local_conditionless_lineage,
            all_target_seed_conditionless_lineage: self.all_target_seed_conditionless_lineage,
            all_object_agnostic_growth_seed_mode: self.all_object_agnostic_growth_seed_mode,
            all_target_growth_seed_mode: self.all_target_growth_seed_mode,
            all_no_seed_coordinate_scaffold: self.all_no_seed_coordinate_scaffold,
        }
    }
}
