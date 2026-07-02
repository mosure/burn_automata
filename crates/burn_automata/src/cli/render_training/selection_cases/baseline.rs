use super::super::selection::RenderSelectionBaselineCase;
use super::super::*;
use super::metrics::render_selection_case_metrics;

pub(crate) fn render_selection_baseline(
    model: &NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: &RenderProxyTrainingConfig,
    render_cfg: RenderLossConfig,
) -> Result<Vec<RenderSelectionBaselineCase>, Box<dyn std::error::Error>> {
    let selection_seeds = render_proxy_selection_seeds(cfg);
    let mut baselines = Vec::with_capacity(selection_seeds.len());
    for seed in selection_seeds {
        let selection_case =
            render_selection_case_metrics(model, grid, target, cfg, render_cfg, seed)?;
        baselines.push(RenderSelectionBaselineCase {
            seed,
            active_surface_max: selection_case.active_surface.max_distance,
            target_coverage_fraction: selection_case.target_coverage.covered_fraction,
            material_visible_target_mean_distance: selection_case
                .material_visible_target_coverage
                .mean_distance,
            material_visible_target_max_distance: selection_case
                .material_visible_target_coverage
                .max_distance,
            material_visible_target_coverage_fraction: selection_case
                .material_visible_target_coverage
                .covered_fraction,
            material_visible_inactive_fraction: selection_case
                .material_liveness
                .inactive_material_visible_fraction,
            material_visible_max_inactive_opacity: selection_case
                .material_liveness
                .max_inactive_material_opacity,
            surface_covered_bin_fraction: selection_case
                .surface_coverage_profile
                .covered_bin_fraction,
            surface_mean_bin_covered_fraction: selection_case
                .surface_coverage_profile
                .mean_bin_covered_fraction,
            material_visible_surface_covered_bin_fraction: selection_case
                .material_visible_surface_coverage_profile
                .covered_bin_fraction,
            material_visible_surface_mean_bin_covered_fraction: selection_case
                .material_visible_surface_coverage_profile
                .mean_bin_covered_fraction,
            surface_normal_covered_bin_fraction: selection_case
                .surface_normal_coverage
                .covered_target_bin_fraction,
            surface_normal_mean_bin_covered_fraction: selection_case
                .surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_normal_covered_bin_fraction: selection_case
                .material_visible_surface_normal_coverage
                .covered_target_bin_fraction,
            material_visible_surface_normal_mean_bin_covered_fraction: selection_case
                .material_visible_surface_normal_coverage
                .mean_bin_covered_fraction,
            material_visible_surface_tail_p99_distance: selection_case
                .material_visible_surface_tail
                .p99_distance,
            material_visible_surface_tail_over_threshold_fraction: selection_case
                .material_visible_surface_tail
                .over_threshold_fraction,
            dormant_drift_fraction: selection_case.dormant_drift.drifting_fraction,
            max_dormant_drift: selection_case.dormant_drift.max_dormant_displacement,
            active_extent_bbox_ratio: selection_case.extent.bbox_diagonal_ratio,
            active_extent_min_axis_ratio: selection_case.extent.min_axis_extent_ratio,
            final_active_count: selection_case.final_active_count,
            newly_activated_fraction: selection_case.newly_activated_fraction,
            front_local_newly_activated_fraction: selection_case
                .front_local_newly_activated_fraction,
            front_liveness: selection_case.front_liveness,
            extent_front_liveness: selection_case.extent_front_liveness,
            temporal_front_liveness: selection_case.temporal_front_liveness,
            temporal_extent_front_liveness: selection_case.temporal_extent_front_liveness,
            temporal_activation_schedule_error: selection_case.temporal_activation_schedule_error,
            temporal_activation_progressive: selection_case.temporal_activation_progressive,
            temporal_geometry_progressive: selection_case.temporal_geometry_progressive,
        });
    }
    Ok(baselines)
}
