use crate::cli::prelude::*;

use super::{growth3d::CliGrowth3dValidationReport, render_proxy::RenderProxyTrainingReport};

#[derive(Serialize)]
pub(crate) struct CliRenderTrainingReport {
    pub(crate) target: MeshTargetArg,
    pub(crate) base_model: Option<String>,
    pub(crate) model_output: String,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) sgd: SgdConfig,
    pub(crate) report: RenderProxyTrainingReport,
    pub(crate) final_render_loss: MultiViewRenderLossReport,
    pub(crate) strict_gate_summary: CliRenderTrainingGateSummary,
    pub(crate) growth_validation: CliGrowth3dValidationReport,
    pub(crate) catalog_promotion: CliCatalogPromotionSummary,
    pub(crate) catalog_promotion_validations: Vec<CliGrowth3dValidationReport>,
}

#[derive(Serialize)]
pub(crate) struct CliRenderAdapterSuiteReport {
    pub(crate) base_model_input: Option<String>,
    pub(crate) base_model: String,
    pub(crate) base_source: Option<String>,
    pub(crate) shared_base_initialized: bool,
    pub(crate) shared_base_cycles: usize,
    pub(crate) shared_base_training: Vec<CliRenderAdapterSuiteBaseEntry>,
    pub(crate) output_dir: String,
    pub(crate) target_set: MeshTargetSetArg,
    pub(crate) targets: Vec<MeshTargetArg>,
    pub(crate) shared_base_targets: Vec<MeshTargetArg>,
    pub(crate) holdout_targets: Vec<MeshTargetArg>,
    pub(crate) particle_count: usize,
    pub(crate) rollout_steps: usize,
    pub(crate) sgd: SgdConfig,
    pub(crate) adapter_rank: usize,
    pub(crate) adapter_alpha: f32,
    pub(crate) base_parameter_count: usize,
    pub(crate) materialized_parameter_count: usize,
    pub(crate) adapter_parameter_count: usize,
    pub(crate) adapter_to_full_ratio: f32,
    pub(crate) target_count: usize,
    pub(crate) shared_base_target_count: usize,
    pub(crate) holdout_target_count: usize,
    pub(crate) adapter_total_parameter_count: usize,
    pub(crate) full_bank_parameter_count: usize,
    pub(crate) shared_plus_adapter_parameter_count: usize,
    pub(crate) shared_plus_adapter_to_full_bank_ratio: f32,
    pub(crate) shared_plus_adapter_savings_ratio: f32,
    pub(crate) training_signal_passed: bool,
    pub(crate) missing_train_signal: Vec<CliRenderAdapterSuiteTrainingSignalGap>,
    pub(crate) entries: Vec<CliRenderAdapterSuiteEntry>,
}

#[derive(Serialize)]
pub(crate) struct CliRenderAdapterSuiteBaseEntry {
    pub(crate) cycle: usize,
    pub(crate) target: MeshTargetArg,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) report: RenderProxyTrainingReport,
}

#[derive(Serialize)]
pub(crate) struct CliRenderAdapterSuiteEntry {
    pub(crate) target: MeshTargetArg,
    pub(crate) split: CliRenderAdapterSuiteSplit,
    pub(crate) adapter_output: String,
    pub(crate) materialized_model_output: String,
    pub(crate) seed_scale: f32,
    pub(crate) seed_mode: ParticleSeed,
    pub(crate) report: RenderProxyTrainingReport,
    pub(crate) final_render_loss: MultiViewRenderLossReport,
    pub(crate) strict_gate_summary: CliRenderTrainingGateSummary,
    pub(crate) growth_validation: CliGrowth3dValidationReport,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CliRenderAdapterSuiteSplit {
    SharedBaseTrain,
    HoldoutAdapterOnly,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct CliRenderAdapterSuiteTrainingSignalGap {
    pub(crate) phase: CliRenderAdapterSuiteTrainingPhase,
    pub(crate) cycle: Option<usize>,
    pub(crate) target: MeshTargetArg,
    pub(crate) rounds: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CliRenderAdapterSuiteTrainingPhase {
    SharedBase,
    Adapter,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CliCatalogPromotionSummary {
    pub(crate) requested: bool,
    pub(crate) validation_count: usize,
    pub(crate) validation_passed: bool,
    pub(crate) training_signal_passed: bool,
    pub(crate) missing_train_signal_rounds: Vec<usize>,
    pub(crate) rejection_reason: Option<String>,
}

impl CliCatalogPromotionSummary {
    pub(crate) fn from_validation_and_training_result(
        requested: bool,
        validation_count: usize,
        missing_train_signal_rounds: Vec<usize>,
        rejection_reason: Option<String>,
    ) -> Self {
        let training_signal_passed = missing_train_signal_rounds.is_empty();
        Self {
            requested,
            validation_count,
            validation_passed: requested && training_signal_passed && rejection_reason.is_none(),
            training_signal_passed,
            missing_train_signal_rounds,
            rejection_reason,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CliRenderTrainingGateSummary {
    pub(crate) target: MeshTargetArg,
    pub(crate) source: Option<String>,
    pub(crate) strict_passed: bool,
    pub(crate) gate_passed: bool,
    pub(crate) catalog_sanity_passed: bool,
    pub(crate) strict_score: f32,
    pub(crate) hard_failure_penalty: f32,
    pub(crate) failure_reasons: Vec<&'static str>,
    pub(crate) no_position_features: bool,
    pub(crate) local_conditionless_lineage: bool,
    pub(crate) target_conditionless_lineage: bool,
    pub(crate) target_seed_conditionless_lineage: bool,
    pub(crate) target_growth_seed_mode: bool,
    pub(crate) no_seed_coordinate_scaffold: bool,
    pub(crate) neutral_non_opacity_seed_state: bool,
    pub(crate) active_seed_count: usize,
    pub(crate) final_active_count: usize,
    pub(crate) active_count_delta: isize,
    pub(crate) newly_activated_fraction: f32,
    pub(crate) active_extent_growth: bool,
    pub(crate) active_extent_bbox_ratio: f32,
    pub(crate) active_extent_min_axis_ratio: f32,
    pub(crate) local_newly_activated_fraction: f32,
    pub(crate) dormant_drift_bounded: bool,
    pub(crate) dormant_drift_fraction: f32,
    pub(crate) max_dormant_drift: f32,
    pub(crate) mean_final_displacement: f32,
    pub(crate) peak_motion_per_step: f32,
    pub(crate) final_motion_per_step: f32,
    pub(crate) temporal_activation_progressive: bool,
    pub(crate) temporal_geometry_progressive: bool,
    pub(crate) target_coverage_mean_improvement: f32,
    pub(crate) target_coverage_mean_ratio: f32,
    pub(crate) target_coverage_max_distance: f32,
    pub(crate) target_coverage_fraction_delta: f32,
    pub(crate) target_coverage_fraction: f32,
    pub(crate) material_visible_target_mean_distance: f32,
    pub(crate) material_visible_target_max_distance: f32,
    pub(crate) material_visible_target_coverage_fraction: f32,
    pub(crate) surface_mean_ratio: f32,
    pub(crate) surface_max_distance: f32,
    pub(crate) surface_tail_p99_distance: f32,
    pub(crate) surface_tail_over_threshold_fraction: f32,
    pub(crate) material_visible_surface_tail_p99_distance: f32,
    pub(crate) material_visible_surface_tail_over_threshold_fraction: f32,
    pub(crate) surface_covered_bin_fraction: f32,
    pub(crate) surface_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_mean_bin_covered_fraction: f32,
    pub(crate) surface_normal_covered_bin_fraction: f32,
    pub(crate) surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) material_visible_surface_normal_covered_bin_fraction: f32,
    pub(crate) material_visible_surface_normal_mean_bin_covered_fraction: f32,
    pub(crate) gaussian_scale_budget_loss: f32,
    pub(crate) gaussian_oversize_fraction: f32,
    pub(crate) render_loss_passed: bool,
    pub(crate) render_total_loss: f32,
    pub(crate) render_density_psnr_db: f32,
    pub(crate) render_color_psnr_db: f32,
    pub(crate) render_depth_psnr_db: f32,
}

impl CliRenderTrainingGateSummary {
    pub(crate) fn from_validation(report: &CliGrowth3dValidationReport) -> Self {
        Self {
            target: report.target,
            source: report.source.clone(),
            strict_passed: report.strict_passed,
            gate_passed: report.gate_passed,
            catalog_sanity_passed: report.catalog_sanity.passed,
            strict_score: report.strict_score.score,
            hard_failure_penalty: report.strict_score.hard_failure_penalty,
            failure_reasons: report.strict_checks.failure_reasons.clone(),
            no_position_features: report.strict_checks.no_position_features,
            local_conditionless_lineage: report.strict_checks.local_conditionless_lineage,
            target_conditionless_lineage: report.strict_checks.target_conditionless_lineage,
            target_seed_conditionless_lineage: report
                .strict_checks
                .target_seed_conditionless_lineage,
            target_growth_seed_mode: report.strict_checks.target_growth_seed_mode,
            no_seed_coordinate_scaffold: report.strict_checks.no_seed_coordinate_scaffold,
            neutral_non_opacity_seed_state: report.strict_checks.neutral_non_opacity_seed_state,
            active_seed_count: report.activation.active_seed_count,
            final_active_count: report.activation.final_active_count,
            active_count_delta: report.activation.final_active_count as isize
                - report.activation.active_seed_count as isize,
            newly_activated_fraction: report.activation.newly_activated_fraction,
            active_extent_growth: report.strict_checks.active_extent_growth,
            active_extent_bbox_ratio: report.extent.bbox_diagonal_ratio,
            active_extent_min_axis_ratio: report.extent.min_axis_extent_ratio,
            local_newly_activated_fraction: report.front.local_newly_activated_fraction,
            dormant_drift_bounded: report.strict_checks.dormant_drift_bounded,
            dormant_drift_fraction: report.dormant_drift.drifting_fraction,
            max_dormant_drift: report.dormant_drift.max_dormant_displacement,
            mean_final_displacement: report.mean_final_displacement,
            peak_motion_per_step: report.motion.peak_mean_dx,
            final_motion_per_step: report.motion.final_step_mean_dx,
            temporal_activation_progressive: report.temporal.progressive_activation,
            temporal_geometry_progressive: report.temporal.geometry_progressive,
            target_coverage_mean_improvement: report.initial_target_coverage.mean_distance
                - report.final_target_coverage.mean_distance,
            target_coverage_mean_ratio: report.strict_score.target_coverage_mean_ratio,
            target_coverage_max_distance: report.final_target_coverage.max_distance,
            target_coverage_fraction_delta: report.final_target_coverage.covered_fraction
                - report.initial_target_coverage.covered_fraction,
            target_coverage_fraction: report.final_target_coverage.covered_fraction,
            material_visible_target_mean_distance: report
                .final_material_visible_target_coverage
                .mean_distance,
            material_visible_target_max_distance: report
                .final_material_visible_target_coverage
                .max_distance,
            material_visible_target_coverage_fraction: report
                .final_material_visible_target_coverage
                .covered_fraction,
            surface_mean_ratio: report.strict_score.surface_mean_ratio,
            surface_max_distance: report.final_active_surface.max_distance,
            surface_tail_p99_distance: report.final_active_surface_tail.p99_distance,
            surface_tail_over_threshold_fraction: report
                .final_active_surface_tail
                .over_threshold_fraction,
            material_visible_surface_tail_p99_distance: report
                .final_material_visible_surface_tail
                .p99_distance,
            material_visible_surface_tail_over_threshold_fraction: report
                .final_material_visible_surface_tail
                .over_threshold_fraction,
            surface_covered_bin_fraction: report
                .final_active_surface_coverage_profile
                .covered_bin_fraction,
            surface_mean_bin_covered_fraction: report
                .final_active_surface_coverage_profile
                .mean_bin_covered_fraction,
            material_visible_surface_covered_bin_fraction: report
                .final_material_visible_surface_coverage_profile
                .covered_bin_fraction,
            material_visible_surface_mean_bin_covered_fraction: report
                .final_material_visible_surface_coverage_profile
                .mean_bin_covered_fraction,
            surface_normal_covered_bin_fraction: report
                .final_active_surface_normal_coverage
                .covered_target_bin_fraction,
            surface_normal_mean_bin_covered_fraction: report
                .final_active_surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_normal_covered_bin_fraction: report
                .final_material_visible_surface_normal_coverage
                .covered_target_bin_fraction,
            material_visible_surface_normal_mean_bin_covered_fraction: report
                .final_material_visible_surface_normal_coverage
                .mean_bin_covered_fraction,
            gaussian_scale_budget_loss: report.final_gaussian_volume.scale_budget_loss,
            gaussian_oversize_fraction: report.final_gaussian_volume.oversize_fraction,
            render_loss_passed: report.render_loss.passed,
            render_total_loss: report.render_loss.total_loss,
            render_density_psnr_db: report.render_loss.density_psnr_db,
            render_color_psnr_db: report.render_loss.color_psnr_db,
            render_depth_psnr_db: report.render_loss.depth_psnr_db,
        }
    }
}
