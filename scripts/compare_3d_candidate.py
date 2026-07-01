#!/usr/bin/env python3
"""Compare a 3D growth candidate validation report against an active baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_ABS_TOLERANCE = 1.0e-6
MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION = 0.25
MIN_PERTURBED_ACTIVE_COUNT_RATIO = 0.5
MAX_PERTURBED_ACTIVE_COUNT_RATIO = 2.0
MIN_PERTURBED_PEAK_MOTION_RATIO = 0.25
MAX_PERTURBED_PEAK_MOTION_RATIO = 4.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reject 3D candidate artifacts that regress current local-growth "
            "validation metrics. Run validate-growth3d first and pass the "
            "resulting JSON reports here."
        )
    )
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument(
        "--require-catalog-safe",
        action="store_true",
        help="Require the candidate to pass catalog-sanity and robust gates.",
    )
    parser.add_argument(
        "--abs-tolerance",
        default=DEFAULT_ABS_TOLERANCE,
        type=float,
        help="Absolute tolerance for no-regression comparisons.",
    )
    parser.add_argument(
        "--min-render-improvement",
        default=0.0,
        type=float,
        help="Required reduction in primary render total loss.",
    )
    parser.add_argument(
        "--min-coverage-improvement",
        default=0.0,
        type=float,
        help="Required increase in primary target coverage fraction.",
    )
    args = parser.parse_args()

    baseline = load_report(args.baseline_report)
    candidate = load_report(args.candidate_report)
    failures = compare_reports(
        baseline,
        candidate,
        require_catalog_safe=args.require_catalog_safe,
        abs_tolerance=args.abs_tolerance,
        min_render_improvement=args.min_render_improvement,
        min_coverage_improvement=args.min_coverage_improvement,
    )
    summary = {
        "baseline": str(args.baseline_report),
        "candidate": str(args.candidate_report),
        "status": "ok" if not failures else "failed",
        "baseline_metrics": metrics_summary(baseline),
        "candidate_metrics": metrics_summary(candidate),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing validation report: {path}")
    return json.loads(path.read_text())


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    require_catalog_safe: bool,
    abs_tolerance: float,
    min_render_improvement: float,
    min_coverage_improvement: float,
) -> list[str]:
    failures: list[str] = []
    for field in ("target", "seed_mode", "particle_count", "steps", "seed_scale"):
        if baseline.get(field) != candidate.get(field):
            failures.append(
                f"{field}: expected {baseline.get(field)!r}, got {candidate.get(field)!r}"
            )

    for field in (
        "local_conditionless_lineage",
        "position_features",
    ):
        expected = baseline.get(field)
        observed = candidate.get(field)
        if field == "position_features":
            if observed is not False:
                failures.append(f"{field}: expected false, got {observed!r}")
        elif expected is True and observed is not True:
            failures.append(f"{field}: expected true, got {observed!r}")

    baseline_strict = baseline.get("strict_checks") or {}
    strict = candidate.get("strict_checks") or {}
    for field in (
        "no_position_features",
        "local_conditionless_lineage",
        "neutral_non_opacity_seed_state",
        "sparse_active_seed",
        "active_count_growth",
        "newly_activated_fraction",
        "active_front_expanded",
        "active_extent_growth",
        "nonzero_motion",
        "sustained_motion",
        "local_front_coherent",
        "temporal_geometry_progressive",
        "mean_displacement_growth",
        "bounded_final_opacity",
        "color_state_emerged",
        "permutation_consistent",
        "surface_tail_bounded",
        "target_coverage_mean_improved",
    ):
        if strict.get(field) is not True:
            failures.append(f"strict_checks.{field}: expected true, got {strict.get(field)!r}")
    seed_perturbation = candidate.get("seed_perturbation") or {}
    if seed_perturbation.get("passed") is not True:
        failures.append(
            f"seed_perturbation.passed: expected true, got {seed_perturbation.get('passed')!r}"
        )
    for field in (
        "material_visible_particles_live",
        "material_visible_target_coverage_fraction",
        "material_visible_surface_coverage_profile",
        "material_visible_surface_normal_coverage",
        "material_visible_surface_tail_bounded",
    ):
        compare_bool_preserves_true(
            failures,
            f"strict_checks.{field}",
            strict.get(field),
            baseline_strict.get(field),
        )

    compare_lower(
        failures,
        "strict_score.score",
        nested_get(candidate, "strict_score", "score"),
        nested_get(baseline, "strict_score", "score"),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "render_loss.total_loss",
        nested_get(candidate, "render_loss", "total_loss"),
        nested_get(baseline, "render_loss", "total_loss") - min_render_improvement,
        abs_tolerance,
    )
    compare_higher(
        failures,
        "render_loss.density_psnr_db",
        nested_get(candidate, "render_loss", "density_psnr_db"),
        nested_get(baseline, "render_loss", "density_psnr_db"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "render_loss.depth_psnr_db",
        nested_get(candidate, "render_loss", "depth_psnr_db"),
        nested_get(baseline, "render_loss", "depth_psnr_db"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_active_target_coverage.covered_fraction",
        nested_get(candidate, "final_active_target_coverage", "covered_fraction"),
        nested_get(baseline, "final_active_target_coverage", "covered_fraction")
        + min_coverage_improvement,
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_material_visible_target_coverage.covered_fraction",
        nested_get(candidate, "final_material_visible_target_coverage", "covered_fraction"),
        nested_get(baseline, "final_material_visible_target_coverage", "covered_fraction"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "extent.bbox_diagonal_ratio",
        nested_get(candidate, "extent", "bbox_diagonal_ratio"),
        nested_get(baseline, "extent", "bbox_diagonal_ratio"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "extent.min_axis_extent_ratio",
        nested_get(candidate, "extent", "min_axis_extent_ratio"),
        nested_get(baseline, "extent", "min_axis_extent_ratio"),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "final_material_liveness.inactive_material_visible_fraction",
        nested_get(candidate, "final_material_liveness", "inactive_material_visible_fraction"),
        nested_get(baseline, "final_material_liveness", "inactive_material_visible_fraction"),
        abs_tolerance,
    )
    compare_optional_lower(
        failures,
        "final_material_liveness.max_inactive_material_opacity",
        nested_get(candidate, "final_material_liveness", "max_inactive_material_opacity"),
        nested_get(baseline, "final_material_liveness", "max_inactive_material_opacity"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_active_surface_normal_coverage.covered_target_bin_fraction",
        nested_get(
            candidate,
            "final_active_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        nested_get(
            baseline,
            "final_active_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_active_surface_normal_coverage.mean_bin_covered_fraction",
        nested_get(
            candidate,
            "final_active_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "final_active_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_active_surface_coverage_profile.covered_bin_fraction",
        nested_get(candidate, "final_active_surface_coverage_profile", "covered_bin_fraction"),
        nested_get(baseline, "final_active_surface_coverage_profile", "covered_bin_fraction"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_active_surface_coverage_profile.mean_bin_covered_fraction",
        nested_get(
            candidate,
            "final_active_surface_coverage_profile",
            "mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "final_active_surface_coverage_profile",
            "mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_material_visible_surface_coverage_profile.covered_bin_fraction",
        nested_get(
            candidate,
            "final_material_visible_surface_coverage_profile",
            "covered_bin_fraction",
        ),
        nested_get(
            baseline,
            "final_material_visible_surface_coverage_profile",
            "covered_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_material_visible_surface_coverage_profile.mean_bin_covered_fraction",
        nested_get(
            candidate,
            "final_material_visible_surface_coverage_profile",
            "mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "final_material_visible_surface_coverage_profile",
            "mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_material_visible_surface_normal_coverage.covered_target_bin_fraction",
        nested_get(
            candidate,
            "final_material_visible_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        nested_get(
            baseline,
            "final_material_visible_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "final_material_visible_surface_normal_coverage.mean_bin_covered_fraction",
        nested_get(
            candidate,
            "final_material_visible_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "final_material_visible_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "final_material_visible_surface_tail.p99_distance",
        nested_get(candidate, "final_material_visible_surface_tail", "p99_distance"),
        nested_get(baseline, "final_material_visible_surface_tail", "p99_distance"),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "final_material_visible_surface_tail.over_threshold_fraction",
        nested_get(candidate, "final_material_visible_surface_tail", "over_threshold_fraction"),
        nested_get(baseline, "final_material_visible_surface_tail", "over_threshold_fraction"),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "final_material_visible_surface_tail.opacity_weighted_over_threshold_fraction",
        nested_get(
            candidate,
            "final_material_visible_surface_tail",
            "opacity_weighted_over_threshold_fraction",
        ),
        nested_get(
            baseline,
            "final_material_visible_surface_tail",
            "opacity_weighted_over_threshold_fraction",
        ),
        abs_tolerance,
    )

    if candidate.get("target") == "Torus":
        compare_higher(
            failures,
            "torus_angular_coverage.joint_coverage_fraction",
            nested_get(candidate, "torus_angular_coverage", "joint_coverage_fraction"),
            nested_get(baseline, "torus_angular_coverage", "joint_coverage_fraction"),
            abs_tolerance,
        )
        compare_higher(
            failures,
            "torus_angular_coverage.tube_coverage_fraction",
            nested_get(candidate, "torus_angular_coverage", "tube_coverage_fraction"),
            nested_get(baseline, "torus_angular_coverage", "tube_coverage_fraction"),
            abs_tolerance,
        )

    compare_lower(
        failures,
        "final_opacity.max",
        nested_get(candidate, "final_opacity", "max"),
        nested_get(baseline, "final_opacity", "max") + 0.25,
        abs_tolerance,
    )
    compare_lower(
        failures,
        "robustness.max_render_loss",
        nested_get(candidate, "robustness", "max_render_loss"),
        nested_get(baseline, "robustness", "max_render_loss"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_depth_psnr_db",
        nested_get(candidate, "robustness", "min_depth_psnr_db"),
        nested_get(baseline, "robustness", "min_depth_psnr_db"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_active_target_coverage_fraction",
        nested_get(candidate, "robustness", "min_final_active_target_coverage_fraction"),
        nested_get(baseline, "robustness", "min_final_active_target_coverage_fraction"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_material_visible_target_coverage_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_material_visible_target_coverage_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_material_visible_target_coverage_fraction",
        ),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "robustness.max_inactive_material_visible_fraction",
        nested_get(candidate, "robustness", "max_inactive_material_visible_fraction"),
        nested_get(baseline, "robustness", "max_inactive_material_visible_fraction"),
        abs_tolerance,
    )
    compare_optional_lower(
        failures,
        "robustness.max_inactive_material_opacity",
        nested_get(candidate, "robustness", "max_inactive_material_opacity"),
        nested_get(baseline, "robustness", "max_inactive_material_opacity"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_active_surface_normal_covered_bin_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_active_surface_normal_covered_bin_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_active_surface_normal_covered_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_active_surface_normal_mean_bin_covered_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_active_surface_normal_mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_active_surface_normal_mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_active_surface_covered_bin_fraction",
        nested_get(candidate, "robustness", "min_final_active_surface_covered_bin_fraction"),
        nested_get(baseline, "robustness", "min_final_active_surface_covered_bin_fraction"),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_active_surface_mean_bin_covered_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_active_surface_mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_active_surface_mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_material_visible_surface_covered_bin_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_material_visible_surface_covered_bin_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_material_visible_surface_covered_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_material_visible_surface_mean_bin_covered_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_material_visible_surface_mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_material_visible_surface_mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_material_visible_surface_normal_covered_bin_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_material_visible_surface_normal_covered_bin_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_material_visible_surface_normal_covered_bin_fraction",
        ),
        abs_tolerance,
    )
    compare_higher(
        failures,
        "robustness.min_final_material_visible_surface_normal_mean_bin_covered_fraction",
        nested_get(
            candidate,
            "robustness",
            "min_final_material_visible_surface_normal_mean_bin_covered_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "min_final_material_visible_surface_normal_mean_bin_covered_fraction",
        ),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "robustness.max_final_material_visible_surface_tail_p99_distance",
        nested_get(
            candidate,
            "robustness",
            "max_final_material_visible_surface_tail_p99_distance",
        ),
        nested_get(
            baseline,
            "robustness",
            "max_final_material_visible_surface_tail_p99_distance",
        ),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "robustness.max_final_material_visible_surface_tail_over_threshold_fraction",
        nested_get(
            candidate,
            "robustness",
            "max_final_material_visible_surface_tail_over_threshold_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "max_final_material_visible_surface_tail_over_threshold_fraction",
        ),
        abs_tolerance,
    )
    compare_lower(
        failures,
        "robustness.max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        nested_get(
            candidate,
            "robustness",
            "max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ),
        nested_get(
            baseline,
            "robustness",
            "max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ),
        abs_tolerance,
    )
    robustness = candidate.get("robustness") or {}
    baseline_robustness = baseline.get("robustness") or {}
    for field in (
        "all_material_visible_particles_live",
        "all_material_visible_surface_normal_coverage",
        "all_material_visible_surface_tail_bounded",
    ):
        compare_bool_preserves_true(
            failures,
            f"robustness.{field}",
            robustness.get(field),
            baseline_robustness.get(field),
        )
    if robustness.get("all_seed_perturbation_stable") is not True:
        failures.append(
            "robustness.all_seed_perturbation_stable: expected true, "
            f"got {robustness.get('all_seed_perturbation_stable')!r}"
        )
    compare_higher_value(
        failures,
        "robustness.min_perturbed_newly_activated_fraction",
        nested_get(candidate, "robustness", "min_perturbed_newly_activated_fraction"),
        MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION,
        abs_tolerance,
    )
    compare_higher_value(
        failures,
        "robustness.min_perturbed_active_count_ratio",
        nested_get(candidate, "robustness", "min_perturbed_active_count_ratio"),
        MIN_PERTURBED_ACTIVE_COUNT_RATIO,
        abs_tolerance,
    )
    compare_lower_value(
        failures,
        "robustness.max_perturbed_active_count_ratio",
        nested_get(candidate, "robustness", "max_perturbed_active_count_ratio"),
        MAX_PERTURBED_ACTIVE_COUNT_RATIO,
        abs_tolerance,
    )
    compare_higher_value(
        failures,
        "robustness.min_perturbed_peak_motion_ratio",
        nested_get(candidate, "robustness", "min_perturbed_peak_motion_ratio"),
        MIN_PERTURBED_PEAK_MOTION_RATIO,
        abs_tolerance,
    )
    compare_lower_value(
        failures,
        "robustness.max_perturbed_peak_motion_ratio",
        nested_get(candidate, "robustness", "max_perturbed_peak_motion_ratio"),
        MAX_PERTURBED_PEAK_MOTION_RATIO,
        abs_tolerance,
    )

    if require_catalog_safe:
        if candidate.get("strict_passed") is not True:
            failures.append(f"strict_passed: expected true, got {candidate.get('strict_passed')!r}")
        strict_checks = candidate.get("strict_checks") or {}
        if strict_checks.get("passed") is not True:
            failures.append(
                f"strict_checks.passed: expected true, got {strict_checks.get('passed')!r}"
            )
        if candidate.get("gate_passed") is not True:
            failures.append(f"gate_passed: expected true, got {candidate.get('gate_passed')!r}")
        catalog_sanity = candidate.get("catalog_sanity") or {}
        if catalog_sanity.get("passed") is not True:
            failures.append(
                f"catalog_sanity.passed: expected true, got {catalog_sanity.get('passed')!r}"
            )
        robustness = candidate.get("robustness") or {}
        for field in (
            "all_gate_passed",
            "all_catalog_sanity_passed",
            "all_strict_passed",
            "all_surface_normal_coverage",
            "all_surface_coverage_profile",
            "all_material_visible_surface_coverage_profile",
            "all_material_visible_particles_live",
            "all_material_visible_surface_normal_coverage",
            "all_material_visible_surface_tail_bounded",
        ):
            if robustness.get(field) is not True:
                failures.append(
                    f"robustness.{field}: expected true, got {robustness.get(field)!r}"
                )

    return failures


def metrics_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": report.get("model"),
        "gate_passed": report.get("gate_passed"),
        "strict_score": nested_get(report, "strict_score", "score"),
        "failure_reasons": nested_get(report, "strict_checks", "failure_reasons") or [],
        "strict_score_material_visible_inactive_fraction": nested_get(
            report, "strict_score", "material_visible_inactive_fraction"
        ),
        "strict_score_material_visible_inactive_fraction_penalty": nested_get(
            report, "strict_score", "material_visible_inactive_fraction_penalty"
        ),
        "strict_score_material_visible_max_inactive_opacity": nested_get(
            report, "strict_score", "material_visible_max_inactive_opacity"
        ),
        "strict_score_material_visible_max_inactive_opacity_penalty": nested_get(
            report, "strict_score", "material_visible_max_inactive_opacity_penalty"
        ),
        "active_extent_growth": nested_get(report, "strict_checks", "active_extent_growth"),
        "active_extent_bbox_ratio": nested_get(report, "extent", "bbox_diagonal_ratio"),
        "active_extent_min_axis_ratio": nested_get(report, "extent", "min_axis_extent_ratio"),
        "active_extent_bbox_penalty": nested_get(
            report, "strict_score", "active_extent_bbox_penalty"
        ),
        "active_extent_min_axis_penalty": nested_get(
            report, "strict_score", "active_extent_min_axis_penalty"
        ),
        "render_total_loss": nested_get(report, "render_loss", "total_loss"),
        "density_psnr_db": nested_get(report, "render_loss", "density_psnr_db"),
        "depth_psnr_db": nested_get(report, "render_loss", "depth_psnr_db"),
        "target_coverage_fraction": nested_get(
            report, "final_active_target_coverage", "covered_fraction"
        ),
        "material_visible_target_coverage_fraction": nested_get(
            report, "final_material_visible_target_coverage", "covered_fraction"
        ),
        "inactive_material_visible_fraction": nested_get(
            report, "final_material_liveness", "inactive_material_visible_fraction"
        ),
        "max_inactive_material_opacity": nested_get(
            report, "final_material_liveness", "max_inactive_material_opacity"
        ),
        "surface_normal_covered_bin_fraction": nested_get(
            report,
            "final_active_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        "surface_normal_mean_bin_covered_fraction": nested_get(
            report,
            "final_active_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        "surface_covered_bin_fraction": nested_get(
            report, "final_active_surface_coverage_profile", "covered_bin_fraction"
        ),
        "surface_mean_bin_covered_fraction": nested_get(
            report, "final_active_surface_coverage_profile", "mean_bin_covered_fraction"
        ),
        "material_visible_surface_covered_bin_fraction": nested_get(
            report,
            "final_material_visible_surface_coverage_profile",
            "covered_bin_fraction",
        ),
        "material_visible_surface_mean_bin_covered_fraction": nested_get(
            report,
            "final_material_visible_surface_coverage_profile",
            "mean_bin_covered_fraction",
        ),
        "material_visible_surface_normal_covered_bin_fraction": nested_get(
            report,
            "final_material_visible_surface_normal_coverage",
            "covered_target_bin_fraction",
        ),
        "material_visible_surface_normal_mean_bin_covered_fraction": nested_get(
            report,
            "final_material_visible_surface_normal_coverage",
            "mean_bin_covered_fraction",
        ),
        "material_visible_surface_tail_p99_distance": nested_get(
            report, "final_material_visible_surface_tail", "p99_distance"
        ),
        "material_visible_surface_tail_over_threshold_fraction": nested_get(
            report, "final_material_visible_surface_tail", "over_threshold_fraction"
        ),
        "material_visible_surface_tail_opacity_weighted_over_threshold_fraction": nested_get(
            report,
            "final_material_visible_surface_tail",
            "opacity_weighted_over_threshold_fraction",
        ),
        "torus_joint_coverage_fraction": nested_get(
            report, "torus_angular_coverage", "joint_coverage_fraction"
        ),
        "torus_tube_coverage_fraction": nested_get(
            report, "torus_angular_coverage", "tube_coverage_fraction"
        ),
        "seed_perturbation_passed": nested_get(report, "seed_perturbation", "passed"),
        "seed_perturbation_active_count_ratio": nested_get(
            report, "seed_perturbation", "active_count_ratio"
        ),
        "seed_perturbation_peak_motion_ratio": nested_get(
            report, "seed_perturbation", "peak_motion_ratio"
        ),
        "max_final_opacity": nested_get(report, "final_opacity", "max"),
        "robustness_max_render_loss": nested_get(report, "robustness", "max_render_loss"),
        "robustness_min_depth_psnr_db": nested_get(
            report, "robustness", "min_depth_psnr_db"
        ),
        "robustness_min_target_coverage_fraction": nested_get(
            report, "robustness", "min_final_active_target_coverage_fraction"
        ),
        "robustness_min_material_visible_target_coverage_fraction": nested_get(
            report,
            "robustness",
            "min_final_material_visible_target_coverage_fraction",
        ),
        "robustness_all_material_visible_particles_live": nested_get(
            report, "robustness", "all_material_visible_particles_live"
        ),
        "robustness_max_inactive_material_visible_fraction": nested_get(
            report, "robustness", "max_inactive_material_visible_fraction"
        ),
        "robustness_max_inactive_material_opacity": nested_get(
            report, "robustness", "max_inactive_material_opacity"
        ),
        "robustness_all_surface_normal_coverage": nested_get(
            report, "robustness", "all_surface_normal_coverage"
        ),
        "robustness_all_surface_coverage_profile": nested_get(
            report, "robustness", "all_surface_coverage_profile"
        ),
        "robustness_all_material_visible_surface_coverage_profile": nested_get(
            report, "robustness", "all_material_visible_surface_coverage_profile"
        ),
        "robustness_min_surface_normal_covered_bin_fraction": nested_get(
            report,
            "robustness",
            "min_final_active_surface_normal_covered_bin_fraction",
        ),
        "robustness_min_surface_normal_mean_bin_covered_fraction": nested_get(
            report,
            "robustness",
            "min_final_active_surface_normal_mean_bin_covered_fraction",
        ),
        "robustness_min_surface_covered_bin_fraction": nested_get(
            report,
            "robustness",
            "min_final_active_surface_covered_bin_fraction",
        ),
        "robustness_min_surface_mean_bin_covered_fraction": nested_get(
            report,
            "robustness",
            "min_final_active_surface_mean_bin_covered_fraction",
        ),
        "robustness_min_material_visible_surface_covered_bin_fraction": nested_get(
            report,
            "robustness",
            "min_final_material_visible_surface_covered_bin_fraction",
        ),
        "robustness_min_material_visible_surface_mean_bin_covered_fraction": nested_get(
            report,
            "robustness",
            "min_final_material_visible_surface_mean_bin_covered_fraction",
        ),
        "robustness_all_material_visible_surface_normal_coverage": nested_get(
            report, "robustness", "all_material_visible_surface_normal_coverage"
        ),
        "robustness_min_material_visible_surface_normal_covered_bin_fraction": nested_get(
            report,
            "robustness",
            "min_final_material_visible_surface_normal_covered_bin_fraction",
        ),
        "robustness_min_material_visible_surface_normal_mean_bin_covered_fraction": nested_get(
            report,
            "robustness",
            "min_final_material_visible_surface_normal_mean_bin_covered_fraction",
        ),
        "robustness_all_material_visible_surface_tail_bounded": nested_get(
            report, "robustness", "all_material_visible_surface_tail_bounded"
        ),
        "robustness_max_material_visible_surface_tail_p99_distance": nested_get(
            report,
            "robustness",
            "max_final_material_visible_surface_tail_p99_distance",
        ),
        "robustness_max_material_visible_surface_tail_over_threshold_fraction": nested_get(
            report,
            "robustness",
            "max_final_material_visible_surface_tail_over_threshold_fraction",
        ),
        "robustness_max_material_visible_surface_tail_opacity_weighted_over_threshold_fraction": nested_get(
            report,
            "robustness",
            "max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ),
        "robustness_all_seed_perturbation_stable": nested_get(
            report, "robustness", "all_seed_perturbation_stable"
        ),
        "robustness_min_perturbed_newly_activated_fraction": nested_get(
            report, "robustness", "min_perturbed_newly_activated_fraction"
        ),
        "robustness_min_perturbed_active_count_ratio": nested_get(
            report, "robustness", "min_perturbed_active_count_ratio"
        ),
        "robustness_max_perturbed_active_count_ratio": nested_get(
            report, "robustness", "max_perturbed_active_count_ratio"
        ),
        "robustness_min_perturbed_peak_motion_ratio": nested_get(
            report, "robustness", "min_perturbed_peak_motion_ratio"
        ),
        "robustness_max_perturbed_peak_motion_ratio": nested_get(
            report, "robustness", "max_perturbed_peak_motion_ratio"
        ),
    }


def compare_lower(
    failures: list[str],
    field: str,
    observed: Any,
    maximum: Any,
    tolerance: float,
) -> None:
    if not is_number(observed) or not is_number(maximum):
        failures.append(f"{field}: expected numeric <= {maximum!r}, got {observed!r}")
        return
    if float(observed) > float(maximum) + tolerance:
        failures.append(f"{field}: expected <= {float(maximum):.8g}, got {float(observed):.8g}")


def compare_lower_value(
    failures: list[str],
    field: str,
    observed: Any,
    maximum: float,
    tolerance: float,
) -> None:
    compare_lower(failures, field, observed, maximum, tolerance)


def compare_optional_lower(
    failures: list[str],
    field: str,
    observed: Any,
    maximum: Any,
    tolerance: float,
) -> None:
    if maximum is None:
        if observed is not None:
            failures.append(f"{field}: expected None, got {observed!r}")
        return
    compare_lower(failures, field, observed, maximum, tolerance)


def compare_higher(
    failures: list[str],
    field: str,
    observed: Any,
    minimum: Any,
    tolerance: float,
) -> None:
    if not is_number(observed) or not is_number(minimum):
        failures.append(f"{field}: expected numeric >= {minimum!r}, got {observed!r}")
        return
    if float(observed) + tolerance < float(minimum):
        failures.append(f"{field}: expected >= {float(minimum):.8g}, got {float(observed):.8g}")


def compare_higher_value(
    failures: list[str],
    field: str,
    observed: Any,
    minimum: float,
    tolerance: float,
) -> None:
    compare_higher(failures, field, observed, minimum, tolerance)


def compare_bool_preserves_true(
    failures: list[str],
    field: str,
    observed: Any,
    baseline: Any,
) -> None:
    if baseline is True and observed is not True:
        failures.append(f"{field}: expected true because baseline is true, got {observed!r}")


def nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def is_number(value: Any) -> bool:
    return isinstance(value, int | float)


if __name__ == "__main__":
    main()
