"""Summary projection for 3D catalog validation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checks import is_strict_seed_lineage_eligible
from .config import ValidationCase
from .expect import nested_get


SUMMARY_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gate_passed", ("gate_passed",)),
    ("strict_passed", ("strict_passed",)),
    ("catalog_sanity_passed", ("catalog_sanity", "passed")),
    ("seed_coordinate_scaffold", ("seed_coordinate_scaffold",)),
    ("strict_no_seed_coordinate_scaffold", ("strict_checks", "no_seed_coordinate_scaffold")),
    ("robustness_all_no_seed_coordinate_scaffold", ("robustness", "all_no_seed_coordinate_scaffold")),
    ("material_visible_particles_live", ("strict_checks", "material_visible_particles_live")),
    ("inactive_material_visible_fraction", ("final_material_liveness", "inactive_material_visible_fraction")),
    ("max_inactive_material_opacity", ("final_material_liveness", "max_inactive_material_opacity")),
    (
        "surface_normal_covered_bin_fraction",
        ("final_active_surface_normal_coverage", "covered_target_bin_fraction"),
    ),
    (
        "surface_normal_mean_bin_covered_fraction",
        ("final_active_surface_normal_coverage", "mean_bin_covered_fraction"),
    ),
    ("surface_covered_bin_fraction", ("final_active_surface_coverage_profile", "covered_bin_fraction")),
    ("surface_mean_bin_covered_fraction", ("final_active_surface_coverage_profile", "mean_bin_covered_fraction")),
    ("surface_empty_bins", ("final_active_surface_coverage_profile", "empty_bins")),
    (
        "material_visible_surface_covered_bin_fraction",
        ("final_material_visible_surface_coverage_profile", "covered_bin_fraction"),
    ),
    (
        "material_visible_surface_mean_bin_covered_fraction",
        ("final_material_visible_surface_coverage_profile", "mean_bin_covered_fraction"),
    ),
    ("material_visible_surface_empty_bins", ("final_material_visible_surface_coverage_profile", "empty_bins")),
    ("material_visible_surface_tail_bounded", ("strict_checks", "material_visible_surface_tail_bounded")),
    ("active_extent_growth", ("strict_checks", "active_extent_growth")),
    ("active_extent_bbox_ratio", ("extent", "bbox_diagonal_ratio")),
    ("active_extent_min_axis_ratio", ("extent", "min_axis_extent_ratio")),
    (
        "final_material_visible_surface_tail_p99_distance",
        ("final_material_visible_surface_tail", "p99_distance"),
    ),
    (
        "final_material_visible_surface_tail_over_threshold_fraction",
        ("final_material_visible_surface_tail", "over_threshold_fraction"),
    ),
    (
        "final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ("final_material_visible_surface_tail", "opacity_weighted_over_threshold_fraction"),
    ),
    (
        "strict_score_material_visible_surface_tail_p99_penalty",
        ("strict_score", "material_visible_surface_tail_p99_penalty"),
    ),
    (
        "strict_score_material_visible_surface_tail_fraction_penalty",
        ("strict_score", "material_visible_surface_tail_fraction_penalty"),
    ),
    ("strict_score_material_visible_inactive_fraction", ("strict_score", "material_visible_inactive_fraction")),
    (
        "strict_score_material_visible_inactive_fraction_penalty",
        ("strict_score", "material_visible_inactive_fraction_penalty"),
    ),
    ("strict_score_material_visible_max_inactive_opacity", ("strict_score", "material_visible_max_inactive_opacity")),
    (
        "strict_score_material_visible_max_inactive_opacity_penalty",
        ("strict_score", "material_visible_max_inactive_opacity_penalty"),
    ),
    (
        "strict_score_temporal_activation_schedule_error",
        ("strict_score", "temporal_activation_schedule_error"),
    ),
    (
        "strict_score_temporal_activation_schedule_penalty",
        ("strict_score", "temporal_activation_schedule_penalty"),
    ),
    ("strict_score_active_extent_bbox_penalty", ("strict_score", "active_extent_bbox_penalty")),
    ("strict_score_active_extent_min_axis_penalty", ("strict_score", "active_extent_min_axis_penalty")),
    ("strict_score_surface_bin_penalty", ("strict_score", "surface_bin_penalty")),
    ("strict_score_surface_coverage_mean_penalty", ("strict_score", "surface_coverage_mean_penalty")),
    ("strict_score_material_visible_surface_bin_penalty", ("strict_score", "material_visible_surface_bin_penalty")),
    (
        "strict_score_material_visible_surface_mean_penalty",
        ("strict_score", "material_visible_surface_mean_penalty"),
    ),
    ("render_total_loss", ("render_loss", "total_loss")),
    ("density_psnr_db", ("render_loss", "density_psnr_db")),
    ("color_state_mean_abs", ("final_color_state", "active_mean_abs")),
    ("color_state_stddev_mean", ("final_color_state", "active_channel_stddev_mean")),
    ("permutation_max_position_error", ("permutation_consistency", "max_position_error")),
    ("permutation_max_state_error", ("permutation_consistency", "max_state_error")),
    ("robustness_seed_count", ("robustness", "seed_count")),
    ("robustness_all_gate_passed", ("robustness", "all_gate_passed")),
    ("robustness_all_catalog_sanity_passed", ("robustness", "all_catalog_sanity_passed")),
    ("robustness_all_strict_passed", ("robustness", "all_strict_passed")),
    ("robustness_max_render_loss", ("robustness", "max_render_loss")),
    ("robustness_min_density_psnr_db", ("robustness", "min_density_psnr_db")),
    ("robustness_min_color_psnr_db", ("robustness", "min_color_psnr_db")),
    ("robustness_min_depth_psnr_db", ("robustness", "min_depth_psnr_db")),
    (
        "robustness_min_final_active_target_coverage_fraction",
        ("robustness", "min_final_active_target_coverage_fraction"),
    ),
    ("robustness_all_surface_normal_coverage", ("robustness", "all_surface_normal_coverage")),
    ("robustness_all_surface_coverage_profile", ("robustness", "all_surface_coverage_profile")),
    (
        "robustness_all_material_visible_surface_coverage_profile",
        ("robustness", "all_material_visible_surface_coverage_profile"),
    ),
    (
        "robustness_min_surface_normal_covered_bin_fraction",
        ("robustness", "min_final_active_surface_normal_covered_bin_fraction"),
    ),
    (
        "robustness_min_surface_normal_mean_bin_covered_fraction",
        ("robustness", "min_final_active_surface_normal_mean_bin_covered_fraction"),
    ),
    (
        "robustness_min_surface_covered_bin_fraction",
        ("robustness", "min_final_active_surface_covered_bin_fraction"),
    ),
    (
        "robustness_min_surface_mean_bin_covered_fraction",
        ("robustness", "min_final_active_surface_mean_bin_covered_fraction"),
    ),
    (
        "robustness_min_material_visible_surface_covered_bin_fraction",
        ("robustness", "min_final_material_visible_surface_covered_bin_fraction"),
    ),
    (
        "robustness_min_material_visible_surface_mean_bin_covered_fraction",
        ("robustness", "min_final_material_visible_surface_mean_bin_covered_fraction"),
    ),
    ("robustness_all_material_visible_particles_live", ("robustness", "all_material_visible_particles_live")),
    ("robustness_max_inactive_material_visible_fraction", ("robustness", "max_inactive_material_visible_fraction")),
    ("robustness_max_inactive_material_opacity", ("robustness", "max_inactive_material_opacity")),
    (
        "robustness_all_material_visible_surface_tail_bounded",
        ("robustness", "all_material_visible_surface_tail_bounded"),
    ),
    (
        "robustness_max_material_visible_surface_tail_p99_distance",
        ("robustness", "max_final_material_visible_surface_tail_p99_distance"),
    ),
    (
        "robustness_max_material_visible_surface_tail_over_threshold_fraction",
        ("robustness", "max_final_material_visible_surface_tail_over_threshold_fraction"),
    ),
    (
        "robustness_max_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ("robustness", "max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction"),
    ),
    ("robustness_min_active_seed_count", ("robustness", "min_active_seed_count")),
    ("robustness_max_active_seed_count", ("robustness", "max_active_seed_count")),
    ("robustness_min_final_active_count", ("robustness", "min_final_active_count")),
    ("robustness_min_newly_activated_fraction", ("robustness", "min_newly_activated_fraction")),
    ("robustness_min_active_growth_ratio", ("robustness", "min_active_growth_ratio")),
    ("robustness_all_color_state_emerged", ("robustness", "all_color_state_emerged")),
    ("robustness_all_permutation_consistent", ("robustness", "all_permutation_consistent")),
    ("seed_perturbation_passed", ("seed_perturbation", "passed")),
    ("seed_perturbation_active_count_ratio", ("seed_perturbation", "active_count_ratio")),
    ("seed_perturbation_peak_motion_ratio", ("seed_perturbation", "peak_motion_ratio")),
    ("robustness_all_seed_perturbation_stable", ("robustness", "all_seed_perturbation_stable")),
    (
        "robustness_min_perturbed_newly_activated_fraction",
        ("robustness", "min_perturbed_newly_activated_fraction"),
    ),
    ("robustness_min_perturbed_active_count_ratio", ("robustness", "min_perturbed_active_count_ratio")),
    ("robustness_max_perturbed_active_count_ratio", ("robustness", "max_perturbed_active_count_ratio")),
    ("robustness_min_perturbed_peak_motion_ratio", ("robustness", "min_perturbed_peak_motion_ratio")),
    ("robustness_max_perturbed_peak_motion_ratio", ("robustness", "max_perturbed_peak_motion_ratio")),
    ("robustness_min_color_state_mean_abs", ("robustness", "min_final_active_color_state_mean_abs")),
    ("robustness_min_color_state_stddev_mean", ("robustness", "min_final_active_color_state_stddev_mean")),
    ("robustness_max_permutation_position_error", ("robustness", "max_permutation_position_error")),
    ("robustness_max_permutation_state_error", ("robustness", "max_permutation_state_error")),
    ("final_gaussian_scale_budget_loss", ("final_gaussian_volume", "scale_budget_loss")),
    ("final_gaussian_oversize_fraction", ("final_gaussian_volume", "oversize_fraction")),
    ("robustness_max_gaussian_scale_budget_loss", ("robustness", "max_gaussian_scale_budget_loss")),
    ("robustness_max_gaussian_oversize_fraction", ("robustness", "max_gaussian_oversize_fraction")),
)


def build_report_summary(
    case: ValidationCase,
    report: dict[str, Any],
    output_path: Path,
    status: str,
) -> dict[str, Any]:
    summary = {
        "name": case.name,
        "model": str(case.model),
        "target": case.target,
        "visible_catalog_entry": case.visible_catalog_entry,
        "expected_strict_seed_lineage_eligible": case.strict_seed_lineage_eligible,
        "strict_seed_lineage_eligible": is_strict_seed_lineage_eligible(report),
        "report": str(output_path),
        "status": status,
        "failure_reasons": nested_get(report, "strict_checks", "failure_reasons") or [],
    }
    summary.update(
        {field: nested_get(report, *path) for field, path in SUMMARY_PATHS}
    )
    return summary
