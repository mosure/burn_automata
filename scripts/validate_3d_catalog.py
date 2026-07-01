#!/usr/bin/env python3
"""Regenerate and assert app-scale 3D catalog validation reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_EVAL_SEED = 0x51A7_3D
HELD_OUT_SEEDS = (42, 99)
PARTICLES = 1024
IMAGE_SIZE = 48
TARGET_SAMPLES = 4096
NON_OPACITY_SEED_TOLERANCE = 1e-6
MIN_COLOR_STATE_MEAN_ABS = 0.01
MIN_COLOR_STATE_STDDEV_MEAN = 0.02
MIN_FINAL_ACTIVE_FRACTION = 0.5
MIN_NEWLY_ACTIVATED_FRACTION = 0.5
MIN_ACTIVE_GROWTH_RATIO = 4.0
MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION = 0.25
MIN_PERTURBED_ACTIVE_COUNT_RATIO = 0.5
MAX_PERTURBED_ACTIVE_COUNT_RATIO = 2.0
MIN_PERTURBED_PEAK_MOTION_RATIO = 0.25
MAX_PERTURBED_PEAK_MOTION_RATIO = 4.0
MAX_GAUSSIAN_SCALE_BUDGET_LOSS = 0.25
MAX_GAUSSIAN_OVERSIZE_FRACTION = 0.05


@dataclass(frozen=True)
class ValidationCase:
    name: str
    model: Path
    target: str
    seed_mode_cli: str
    seed_mode_report: str
    seed_scale: float
    steps: int
    output_name: str
    visible_catalog_entry: bool
    expected_source: str
    required_failure_reasons: frozenset[str]
    allowed_failure_reasons: frozenset[str]


CASES = (
    ValidationCase(
        name="teapot-catalog-64",
        model=Path("assets/models/teapot_growth_3d.bpk"),
        target="teapot",
        seed_mode_cli="teapot-growth-3d",
        seed_mode_report="TeapotGrowth3d",
        seed_scale=0.72,
        steps=64,
        output_name="teapot_growth_3d_catalog_active_validation.json",
        visible_catalog_entry=False,
        expected_source=(
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:"
            "front_retime=false:active_opacity_hidden=skipped:"
            "active_opacity_gain=skipped:opacity_bias=skipped:"
            "material_opacity_bias=0.55:base=render-refined-rust:"
            "ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset({"no_seed_coordinate_scaffold", "render_loss_passed"}),
        allowed_failure_reasons=frozenset(
            {
                "no_seed_coordinate_scaffold",
                "temporal_activation_progressive",
                "surface_max_bounded",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "material_visible_particles_live",
                "material_visible_surface_coverage_profile",
                "material_visible_surface_tail_bounded",
                "render_loss_passed",
            }
        ),
    ),
    ValidationCase(
        name="teapot-viewer-96",
        model=Path("assets/models/teapot_growth_3d.bpk"),
        target="teapot",
        seed_mode_cli="teapot-growth-3d",
        seed_mode_report="TeapotGrowth3d",
        seed_scale=0.72,
        steps=96,
        output_name="teapot_growth_3d_96step_validation.json",
        visible_catalog_entry=False,
        expected_source=(
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:"
            "front_retime=false:active_opacity_hidden=skipped:"
            "active_opacity_gain=skipped:opacity_bias=skipped:"
            "material_opacity_bias=0.55:base=render-refined-rust:"
            "ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset({"no_seed_coordinate_scaffold", "render_loss_passed"}),
        allowed_failure_reasons=frozenset(
            {
                "no_seed_coordinate_scaffold",
                "surface_max_bounded",
                "surface_tail_bounded",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "material_visible_particles_live",
                "material_visible_surface_coverage_profile",
                "material_visible_surface_tail_bounded",
                "render_loss_passed",
            }
        ),
    ),
    ValidationCase(
        name="torus-hidden-64",
        model=Path("assets/models/uv_torus_growth_3d.bpk"),
        target="torus",
        seed_mode_cli="torus-growth-3d",
        seed_mode_report="TorusGrowth3d",
        seed_scale=0.54,
        steps=64,
        output_name="uv_torus_growth_3d_catalog_active_validation.json",
        visible_catalog_entry=False,
        expected_source=(
            "render-refined-rust:ablation-rust:uv-torus-3d:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset(
            {
                "target_coverage_fraction",
                "target_coverage_max_bounded",
                "torus_angular_coverage",
                "no_seed_coordinate_scaffold",
                "temporal_activation_progressive",
                "surface_max_bounded",
                "render_loss_passed",
            }
        ),
        allowed_failure_reasons=frozenset(
            {
                "target_coverage_fraction",
                "target_coverage_max_bounded",
                "torus_angular_coverage",
                "no_seed_coordinate_scaffold",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "material_visible_target_coverage_fraction",
                "material_visible_surface_coverage_profile",
                "material_visible_surface_normal_coverage",
                "material_visible_surface_tail_bounded",
                "temporal_activation_progressive",
                "surface_max_bounded",
                "render_loss_passed",
            }
        ),
    ),
    ValidationCase(
        name="torus-hidden-96",
        model=Path("assets/models/uv_torus_growth_3d.bpk"),
        target="torus",
        seed_mode_cli="torus-growth-3d",
        seed_mode_report="TorusGrowth3d",
        seed_scale=0.54,
        steps=96,
        output_name="uv_torus_growth_3d_96step_validation.json",
        visible_catalog_entry=False,
        expected_source=(
            "render-refined-rust:ablation-rust:uv-torus-3d:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset(
            {
                "target_coverage_fraction",
                "torus_angular_coverage",
                "no_seed_coordinate_scaffold",
                "render_loss_passed",
            }
        ),
        allowed_failure_reasons=frozenset(
            {
                "surface_mean_improved",
                "surface_max_bounded",
                "target_coverage_fraction",
                "target_coverage_max_bounded",
                "torus_angular_coverage",
                "no_seed_coordinate_scaffold",
                "temporal_activation_progressive",
                "surface_tail_bounded",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "material_visible_target_coverage_fraction",
                "material_visible_surface_coverage_profile",
                "material_visible_surface_normal_coverage",
                "material_visible_surface_tail_bounded",
                "render_loss_passed",
            }
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that shipped 3D BPKs match the latest local-growth pipeline "
            "and the Bevy catalog exposure policy."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("target"))
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("target/validate_3d_catalog_summary.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in CASES:
        output_path = args.output_dir / case.output_name
        if not args.no_run:
            run_validation(args.cargo, case, output_path)
        report = load_report(output_path)
        case_failures = assert_case(case, report)
        status = "ok" if not case_failures else "failed"
        reports.append(
            {
                "name": case.name,
                "model": str(case.model),
                "target": case.target,
                "visible_catalog_entry": case.visible_catalog_entry,
                "report": str(output_path),
                "status": status,
                "gate_passed": report.get("gate_passed"),
                "strict_passed": report.get("strict_passed"),
                "catalog_sanity_passed": nested_get(report, "catalog_sanity", "passed"),
                "failure_reasons": nested_get(report, "strict_checks", "failure_reasons") or [],
                "seed_coordinate_scaffold": report.get("seed_coordinate_scaffold"),
                "strict_no_seed_coordinate_scaffold": nested_get(
                    report, "strict_checks", "no_seed_coordinate_scaffold"
                ),
                "robustness_all_no_seed_coordinate_scaffold": nested_get(
                    report, "robustness", "all_no_seed_coordinate_scaffold"
                ),
                "material_visible_particles_live": nested_get(
                    report, "strict_checks", "material_visible_particles_live"
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
                    report,
                    "final_active_surface_coverage_profile",
                    "covered_bin_fraction",
                ),
                "surface_mean_bin_covered_fraction": nested_get(
                    report,
                    "final_active_surface_coverage_profile",
                    "mean_bin_covered_fraction",
                ),
                "surface_empty_bins": nested_get(
                    report,
                    "final_active_surface_coverage_profile",
                    "empty_bins",
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
                "material_visible_surface_empty_bins": nested_get(
                    report,
                    "final_material_visible_surface_coverage_profile",
                    "empty_bins",
                ),
                "material_visible_surface_tail_bounded": nested_get(
                    report, "strict_checks", "material_visible_surface_tail_bounded"
                ),
                "active_extent_growth": nested_get(
                    report, "strict_checks", "active_extent_growth"
                ),
                "active_extent_bbox_ratio": nested_get(
                    report,
                    "extent",
                    "bbox_diagonal_ratio",
                ),
                "active_extent_min_axis_ratio": nested_get(
                    report,
                    "extent",
                    "min_axis_extent_ratio",
                ),
                "final_material_visible_surface_tail_p99_distance": nested_get(
                    report,
                    "final_material_visible_surface_tail",
                    "p99_distance",
                ),
                "final_material_visible_surface_tail_over_threshold_fraction": nested_get(
                    report,
                    "final_material_visible_surface_tail",
                    "over_threshold_fraction",
                ),
                "final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction": nested_get(
                    report,
                    "final_material_visible_surface_tail",
                    "opacity_weighted_over_threshold_fraction",
                ),
                "strict_score_material_visible_surface_tail_p99_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_surface_tail_p99_penalty",
                ),
                "strict_score_material_visible_surface_tail_fraction_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_surface_tail_fraction_penalty",
                ),
                "strict_score_material_visible_inactive_fraction": nested_get(
                    report,
                    "strict_score",
                    "material_visible_inactive_fraction",
                ),
                "strict_score_material_visible_inactive_fraction_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_inactive_fraction_penalty",
                ),
                "strict_score_material_visible_max_inactive_opacity": nested_get(
                    report,
                    "strict_score",
                    "material_visible_max_inactive_opacity",
                ),
                "strict_score_material_visible_max_inactive_opacity_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_max_inactive_opacity_penalty",
                ),
                "strict_score_temporal_activation_schedule_error": nested_get(
                    report,
                    "strict_score",
                    "temporal_activation_schedule_error",
                ),
                "strict_score_temporal_activation_schedule_penalty": nested_get(
                    report,
                    "strict_score",
                    "temporal_activation_schedule_penalty",
                ),
                "strict_score_active_extent_bbox_penalty": nested_get(
                    report,
                    "strict_score",
                    "active_extent_bbox_penalty",
                ),
                "strict_score_active_extent_min_axis_penalty": nested_get(
                    report,
                    "strict_score",
                    "active_extent_min_axis_penalty",
                ),
                "strict_score_surface_bin_penalty": nested_get(
                    report,
                    "strict_score",
                    "surface_bin_penalty",
                ),
                "strict_score_surface_coverage_mean_penalty": nested_get(
                    report,
                    "strict_score",
                    "surface_coverage_mean_penalty",
                ),
                "strict_score_material_visible_surface_bin_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_surface_bin_penalty",
                ),
                "strict_score_material_visible_surface_mean_penalty": nested_get(
                    report,
                    "strict_score",
                    "material_visible_surface_mean_penalty",
                ),
                "render_total_loss": nested_get(report, "render_loss", "total_loss"),
                "density_psnr_db": nested_get(report, "render_loss", "density_psnr_db"),
                "color_state_mean_abs": nested_get(report, "final_color_state", "active_mean_abs"),
                "color_state_stddev_mean": nested_get(
                    report, "final_color_state", "active_channel_stddev_mean"
                ),
                "permutation_max_position_error": nested_get(
                    report, "permutation_consistency", "max_position_error"
                ),
                "permutation_max_state_error": nested_get(
                    report, "permutation_consistency", "max_state_error"
                ),
                "robustness_seed_count": nested_get(report, "robustness", "seed_count"),
                "robustness_all_gate_passed": nested_get(
                    report, "robustness", "all_gate_passed"
                ),
                "robustness_all_catalog_sanity_passed": nested_get(
                    report, "robustness", "all_catalog_sanity_passed"
                ),
                "robustness_all_strict_passed": nested_get(
                    report, "robustness", "all_strict_passed"
                ),
                "robustness_max_render_loss": nested_get(
                    report, "robustness", "max_render_loss"
                ),
                "robustness_min_density_psnr_db": nested_get(
                    report, "robustness", "min_density_psnr_db"
                ),
                "robustness_min_color_psnr_db": nested_get(
                    report, "robustness", "min_color_psnr_db"
                ),
                "robustness_min_depth_psnr_db": nested_get(
                    report, "robustness", "min_depth_psnr_db"
                ),
                "robustness_min_final_active_target_coverage_fraction": nested_get(
                    report,
                    "robustness",
                    "min_final_active_target_coverage_fraction",
                ),
                "robustness_all_surface_normal_coverage": nested_get(
                    report, "robustness", "all_surface_normal_coverage"
                ),
                "robustness_all_surface_coverage_profile": nested_get(
                    report, "robustness", "all_surface_coverage_profile"
                ),
                "robustness_all_material_visible_surface_coverage_profile": nested_get(
                    report,
                    "robustness",
                    "all_material_visible_surface_coverage_profile",
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
                "robustness_all_material_visible_particles_live": nested_get(
                    report,
                    "robustness",
                    "all_material_visible_particles_live",
                ),
                "robustness_max_inactive_material_visible_fraction": nested_get(
                    report,
                    "robustness",
                    "max_inactive_material_visible_fraction",
                ),
                "robustness_max_inactive_material_opacity": nested_get(
                    report,
                    "robustness",
                    "max_inactive_material_opacity",
                ),
                "robustness_all_material_visible_surface_tail_bounded": nested_get(
                    report,
                    "robustness",
                    "all_material_visible_surface_tail_bounded",
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
                "robustness_min_active_seed_count": nested_get(
                    report, "robustness", "min_active_seed_count"
                ),
                "robustness_max_active_seed_count": nested_get(
                    report, "robustness", "max_active_seed_count"
                ),
                "robustness_min_final_active_count": nested_get(
                    report, "robustness", "min_final_active_count"
                ),
                "robustness_min_newly_activated_fraction": nested_get(
                    report, "robustness", "min_newly_activated_fraction"
                ),
                "robustness_min_active_growth_ratio": nested_get(
                    report, "robustness", "min_active_growth_ratio"
                ),
                "robustness_all_color_state_emerged": nested_get(
                    report, "robustness", "all_color_state_emerged"
                ),
                "robustness_all_permutation_consistent": nested_get(
                    report, "robustness", "all_permutation_consistent"
                ),
                "seed_perturbation_passed": nested_get(
                    report, "seed_perturbation", "passed"
                ),
                "seed_perturbation_active_count_ratio": nested_get(
                    report, "seed_perturbation", "active_count_ratio"
                ),
                "seed_perturbation_peak_motion_ratio": nested_get(
                    report, "seed_perturbation", "peak_motion_ratio"
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
                "robustness_min_color_state_mean_abs": nested_get(
                    report, "robustness", "min_final_active_color_state_mean_abs"
                ),
                "robustness_min_color_state_stddev_mean": nested_get(
                    report, "robustness", "min_final_active_color_state_stddev_mean"
                ),
                "robustness_max_permutation_position_error": nested_get(
                    report, "robustness", "max_permutation_position_error"
                ),
                "robustness_max_permutation_state_error": nested_get(
                    report, "robustness", "max_permutation_state_error"
                ),
                "final_gaussian_scale_budget_loss": nested_get(
                    report, "final_gaussian_volume", "scale_budget_loss"
                ),
                "final_gaussian_oversize_fraction": nested_get(
                    report, "final_gaussian_volume", "oversize_fraction"
                ),
                "robustness_max_gaussian_scale_budget_loss": nested_get(
                    report, "robustness", "max_gaussian_scale_budget_loss"
                ),
                "robustness_max_gaussian_oversize_fraction": nested_get(
                    report, "robustness", "max_gaussian_oversize_fraction"
                ),
            }
        )
        for failure in case_failures:
            failures.append(f"{case.name}: {failure}")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"cases": reports, "failed": len(failures)}, indent=2))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)


def run_validation(cargo: str, case: ValidationCase, output_path: Path) -> None:
    cmd = [
        cargo,
        "run",
        "-p",
        "burn_automata",
        "--release",
        "--bin",
        "burn_automata",
        "--",
        "validate-growth3d",
        "--model",
        str(case.model),
        "--target",
        case.target,
        "--seed-mode",
        case.seed_mode_cli,
        "--particles",
        str(PARTICLES),
        "--steps",
        str(case.steps),
        "--seed",
        str(APP_EVAL_SEED),
        "--extra-seed",
        ",".join(str(seed) for seed in HELD_OUT_SEEDS),
        "--seed-scale",
        str(case.seed_scale),
        "--image-size",
        str(IMAGE_SIZE),
        "--target-samples",
        str(TARGET_SAMPLES),
        "--gate",
        "catalog-sanity",
        "--output",
        str(output_path),
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing validation report: {path}")
    return json.loads(path.read_text())


def assert_case(case: ValidationCase, report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    strict_checks = report.get("strict_checks") or {}
    failure_reasons = set(strict_checks.get("failure_reasons") or [])
    catalog_sanity = report.get("catalog_sanity") or {}
    initial_color = report.get("initial_color_state") or {}
    final_color = report.get("final_color_state") or {}
    permutation = report.get("permutation_consistency") or {}
    robustness = report.get("robustness") or {}
    final_material_liveness = report.get("final_material_liveness") or {}
    final_gaussian = report.get("final_gaussian_volume") or {}
    final_surface_normal = report.get("final_active_surface_normal_coverage") or {}
    final_surface_profile = report.get("final_active_surface_coverage_profile") or {}
    final_material_visible_surface_profile = (
        report.get("final_material_visible_surface_coverage_profile") or {}
    )
    final_material_visible_surface_tail = report.get("final_material_visible_surface_tail") or {}
    extent = report.get("extent") or {}
    strict_score = report.get("strict_score") or {}

    expect_equal(failures, "model", report.get("model"), str(case.model))
    expect_equal(failures, "source", report.get("source"), case.expected_source)
    expect_equal(failures, "target", str(report.get("target")).lower(), case.target)
    expect_equal(failures, "seed_mode", report.get("seed_mode"), case.seed_mode_report)
    expect_equal(failures, "seed", report.get("seed"), APP_EVAL_SEED)
    expect_equal(failures, "particle_count", report.get("particle_count"), PARTICLES)
    expect_equal(failures, "steps", report.get("steps"), case.steps)
    expect_close(failures, "seed_scale", report.get("seed_scale"), case.seed_scale, 1e-6)

    expect_true(failures, "local_conditionless_lineage", report.get("local_conditionless_lineage"))
    expect_true(failures, "strict local_conditionless_lineage", strict_checks.get("local_conditionless_lineage"))
    expect_true(failures, "strict no_position_features", strict_checks.get("no_position_features"))
    expect_false(failures, "position_features", report.get("position_features"))
    if case.visible_catalog_entry:
        expect_false(failures, "seed_coordinate_scaffold", report.get("seed_coordinate_scaffold"))
        expect_true(
            failures,
            "strict no_seed_coordinate_scaffold",
            strict_checks.get("no_seed_coordinate_scaffold"),
        )
    else:
        expect_true(failures, "seed_coordinate_scaffold", report.get("seed_coordinate_scaffold"))
        expect_false(
            failures,
            "strict no_seed_coordinate_scaffold",
            strict_checks.get("no_seed_coordinate_scaffold"),
        )
    expect_true(
        failures,
        "strict neutral_non_opacity_seed_state",
        strict_checks.get("neutral_non_opacity_seed_state"),
    )
    expect_true(failures, "strict sparse_active_seed", strict_checks.get("sparse_active_seed"))
    expect_close(
        failures,
        "non_opacity_seed_abs_max",
        report.get("non_opacity_seed_abs_max"),
        0.0,
        NON_OPACITY_SEED_TOLERANCE,
    )

    expect_true(failures, "strict nonzero_motion", strict_checks.get("nonzero_motion"))
    expect_true(failures, "strict sustained_motion", strict_checks.get("sustained_motion"))
    expect_true(failures, "strict local_front_coherent", strict_checks.get("local_front_coherent"))
    expect_bool(
        failures,
        "strict active_extent_growth",
        strict_checks.get("active_extent_growth"),
    )
    for field in ("bbox_diagonal_ratio", "min_axis_extent_ratio", "max_radius_ratio"):
        expect_finite(failures, f"extent.{field}", extent.get(field))
    expect_true(
        failures,
        "strict temporal_geometry_progressive",
        strict_checks.get("temporal_geometry_progressive"),
    )

    expect_true(failures, "initial color state available", initial_color.get("available"))
    expect_true(failures, "final color state available", final_color.get("available"))
    expect_close(
        failures,
        "initial active color state max abs",
        initial_color.get("active_max_abs"),
        0.0,
        NON_OPACITY_SEED_TOLERANCE,
    )
    if (final_color.get("active_mean_abs") or 0.0) < MIN_COLOR_STATE_MEAN_ABS:
        failures.append(
            "final active color-state mean did not emerge "
            f"({final_color.get('active_mean_abs')!r} < {MIN_COLOR_STATE_MEAN_ABS})"
        )
    if (final_color.get("active_channel_stddev_mean") or 0.0) < MIN_COLOR_STATE_STDDEV_MEAN:
        failures.append(
            "final active color-state stddev is too uniform "
            f"({final_color.get('active_channel_stddev_mean')!r} < {MIN_COLOR_STATE_STDDEV_MEAN})"
        )
    expect_true(failures, "strict color_state_emerged", strict_checks.get("color_state_emerged"))

    expect_true(failures, "permutation consistency", permutation.get("passed"))
    expect_true(
        failures,
        "strict permutation_consistent",
        strict_checks.get("permutation_consistent"),
    )
    expect_true(
        failures,
        "strict gaussian_scale_budget",
        strict_checks.get("gaussian_scale_budget"),
    )
    expect_bool(
        failures,
        "strict surface_coverage_profile",
        strict_checks.get("surface_coverage_profile"),
    )
    expect_bool(
        failures,
        "strict material_visible_surface_coverage_profile",
        strict_checks.get("material_visible_surface_coverage_profile"),
    )
    if "material_visible_particles_live" in strict_checks:
        expect_bool(
            failures,
            "strict material_visible_particles_live",
            strict_checks.get("material_visible_particles_live"),
        )
    if final_material_liveness:
        expect_bool(
            failures,
            "final_material_liveness.passed",
            final_material_liveness.get("passed"),
        )
        expect_finite(
            failures,
            "final_material_liveness.inactive_material_visible_fraction",
            final_material_liveness.get("inactive_material_visible_fraction"),
        )
        if final_material_liveness.get("max_inactive_material_opacity") is not None:
            expect_finite(
                failures,
                "final_material_liveness.max_inactive_material_opacity",
                final_material_liveness.get("max_inactive_material_opacity"),
            )
    expect_finite(
        failures,
        "strict_score.material_visible_inactive_fraction",
        strict_score.get("material_visible_inactive_fraction"),
    )
    expect_finite(
        failures,
        "strict_score.material_visible_inactive_fraction_penalty",
        strict_score.get("material_visible_inactive_fraction_penalty"),
    )
    if strict_score.get("material_visible_max_inactive_opacity") is not None:
        expect_finite(
            failures,
            "strict_score.material_visible_max_inactive_opacity",
            strict_score.get("material_visible_max_inactive_opacity"),
        )
    expect_finite(
        failures,
        "strict_score.material_visible_max_inactive_opacity_penalty",
        strict_score.get("material_visible_max_inactive_opacity_penalty"),
    )
    if "temporal_activation_schedule_error" in strict_score:
        expect_finite(
            failures,
            "strict_score.temporal_activation_schedule_error",
            strict_score.get("temporal_activation_schedule_error"),
        )
    if "temporal_activation_schedule_penalty" in strict_score:
        expect_finite(
            failures,
            "strict_score.temporal_activation_schedule_penalty",
            strict_score.get("temporal_activation_schedule_penalty"),
        )
    for field in (
        "active_extent_bbox_ratio",
        "active_extent_bbox_penalty",
        "active_extent_min_axis_ratio",
        "active_extent_min_axis_penalty",
    ):
        expect_finite(failures, f"strict_score.{field}", strict_score.get(field))
    assert_surface_normal_coverage(
        failures,
        "final_active_surface_normal_coverage",
        final_surface_normal,
    )
    assert_surface_coverage_profile(
        failures,
        "final_active_surface_coverage_profile",
        final_surface_profile,
    )
    assert_surface_coverage_profile(
        failures,
        "final_material_visible_surface_coverage_profile",
        final_material_visible_surface_profile,
    )
    expect_finite(
        failures,
        "strict_score.surface_normal_covered_bin_fraction",
        strict_score.get("surface_normal_covered_bin_fraction"),
    )
    expect_finite(
        failures,
        "strict_score.surface_normal_mean_bin_covered_fraction",
        strict_score.get("surface_normal_mean_bin_covered_fraction"),
    )
    expect_finite(
        failures,
        "strict_score.surface_normal_bin_penalty",
        strict_score.get("surface_normal_bin_penalty"),
    )
    expect_finite(
        failures,
        "strict_score.surface_normal_mean_penalty",
        strict_score.get("surface_normal_mean_penalty"),
    )
    if "material_visible_surface_tail_bounded" in strict_checks:
        expect_bool(
            failures,
            "strict material_visible_surface_tail_bounded",
            strict_checks.get("material_visible_surface_tail_bounded"),
        )
    if final_material_visible_surface_tail:
        assert_surface_tail_report(
            failures,
            "final_material_visible_surface_tail",
            final_material_visible_surface_tail,
        )
    if "material_visible_surface_tail_p99_penalty" in strict_score:
        expect_finite(
            failures,
            "strict_score.material_visible_surface_tail_p99_penalty",
            strict_score.get("material_visible_surface_tail_p99_penalty"),
        )
    if "material_visible_surface_tail_fraction_penalty" in strict_score:
        expect_finite(
            failures,
            "strict_score.material_visible_surface_tail_fraction_penalty",
            strict_score.get("material_visible_surface_tail_fraction_penalty"),
        )
    for field in (
        "surface_covered_bin_fraction",
        "surface_bin_penalty",
        "surface_mean_bin_covered_fraction",
        "surface_coverage_mean_penalty",
        "material_visible_surface_covered_bin_fraction",
        "material_visible_surface_bin_penalty",
        "material_visible_surface_mean_bin_covered_fraction",
        "material_visible_surface_mean_penalty",
    ):
        expect_finite(failures, f"strict_score.{field}", strict_score.get(field))
    assert_gaussian_volume(failures, "final_gaussian_volume", final_gaussian)
    assert_robustness(failures, robustness)

    if case.visible_catalog_entry:
        expect_true(failures, "strict_checks.passed", strict_checks.get("passed"))
        expect_true(failures, "strict_passed", report.get("strict_passed"))
        expect_true(failures, "catalog_sanity.passed", catalog_sanity.get("passed"))
        expect_true(failures, "gate_passed", report.get("gate_passed"))
        expect_true(
            failures,
            "robustness.all_gate_passed",
            robustness.get("all_gate_passed"),
        )
        expect_true(
            failures,
            "robustness.all_catalog_sanity_passed",
            robustness.get("all_catalog_sanity_passed"),
        )
        expect_true(
            failures,
            "robustness.all_strict_passed",
            robustness.get("all_strict_passed"),
        )
        expect_true(
            failures,
            "robustness.all_no_seed_coordinate_scaffold",
            robustness.get("all_no_seed_coordinate_scaffold"),
        )
        expect_true(
            failures,
            "robustness.all_surface_normal_coverage",
            robustness.get("all_surface_normal_coverage"),
        )
        expect_true(
            failures,
            "robustness.all_surface_coverage_profile",
            robustness.get("all_surface_coverage_profile"),
        )
        expect_true(
            failures,
            "robustness.all_material_visible_surface_coverage_profile",
            robustness.get("all_material_visible_surface_coverage_profile"),
        )
        expect_le(
            failures,
            "robustness.max_render_loss",
            robustness.get("max_render_loss"),
            catalog_sanity.get("max_total_loss"),
        )
        expect_ge(
            failures,
            "robustness.min_density_psnr_db",
            robustness.get("min_density_psnr_db"),
            catalog_sanity.get("min_density_psnr_db"),
        )
        expect_ge(
            failures,
            "robustness.min_color_psnr_db",
            robustness.get("min_color_psnr_db"),
            catalog_sanity.get("min_color_psnr_db"),
        )
        expect_ge(
            failures,
            "robustness.min_depth_psnr_db",
            robustness.get("min_depth_psnr_db"),
            catalog_sanity.get("min_depth_psnr_db"),
        )
    else:
        expect_false(failures, "strict_checks.passed", strict_checks.get("passed"))
        expect_false(failures, "strict_passed", report.get("strict_passed"))

    missing_reasons = case.required_failure_reasons - failure_reasons
    if missing_reasons:
        failures.append(
            "missing expected strict blocker(s): " + ", ".join(sorted(missing_reasons))
        )
    unexpected_reasons = failure_reasons - case.allowed_failure_reasons
    if unexpected_reasons:
        failures.append(
            "unexpected strict blocker(s): " + ", ".join(sorted(unexpected_reasons))
        )

    if not case.visible_catalog_entry and strict_checks.get("passed"):
        failures.append("hidden model became strict-safe; update catalog visibility before passing")

    return failures


def assert_robustness(failures: list[str], robustness: dict[str, Any]) -> None:
    expected_seed_count = 1 + len(HELD_OUT_SEEDS)
    expect_equal(
        failures,
        "robustness.seed_count",
        robustness.get("seed_count"),
        expected_seed_count,
    )
    expected_seeds = [APP_EVAL_SEED, *HELD_OUT_SEEDS]
    observed_seeds = [seed.get("seed") for seed in robustness.get("seeds") or []]
    expect_equal(failures, "robustness.seeds", observed_seeds, expected_seeds)
    expect_true(
        failures,
        "robustness.all_color_state_emerged",
        robustness.get("all_color_state_emerged"),
    )
    expect_true(
        failures,
        "robustness.all_permutation_consistent",
        robustness.get("all_permutation_consistent"),
    )
    expect_true(
        failures,
        "robustness.all_seed_perturbation_stable",
        robustness.get("all_seed_perturbation_stable"),
    )
    expect_bool(
        failures,
        "robustness.all_strict_passed",
        robustness.get("all_strict_passed"),
    )
    expect_bool(
        failures,
        "robustness.all_no_seed_coordinate_scaffold",
        robustness.get("all_no_seed_coordinate_scaffold"),
    )
    if "all_active_extent_growth" in robustness:
        expect_bool(
            failures,
            "robustness.all_active_extent_growth",
            robustness.get("all_active_extent_growth"),
        )
    expect_bool(
        failures,
        "robustness.all_surface_normal_coverage",
        robustness.get("all_surface_normal_coverage"),
    )
    expect_bool(
        failures,
        "robustness.all_surface_coverage_profile",
        robustness.get("all_surface_coverage_profile"),
    )
    expect_bool(
        failures,
        "robustness.all_material_visible_surface_coverage_profile",
        robustness.get("all_material_visible_surface_coverage_profile"),
    )
    if "all_material_visible_particles_live" in robustness:
        expect_bool(
            failures,
            "robustness.all_material_visible_particles_live",
            robustness.get("all_material_visible_particles_live"),
        )
    if "max_inactive_material_visible_fraction" in robustness:
        expect_finite(
            failures,
            "robustness.max_inactive_material_visible_fraction",
            robustness.get("max_inactive_material_visible_fraction"),
        )
    if robustness.get("max_inactive_material_opacity") is not None:
        expect_finite(
            failures,
            "robustness.max_inactive_material_opacity",
            robustness.get("max_inactive_material_opacity"),
        )
    expect_finite(
        failures,
        "robustness.min_final_active_surface_normal_covered_bin_fraction",
        robustness.get("min_final_active_surface_normal_covered_bin_fraction"),
    )
    expect_finite(
        failures,
        "robustness.min_final_active_surface_normal_mean_bin_covered_fraction",
        robustness.get("min_final_active_surface_normal_mean_bin_covered_fraction"),
    )
    for field in (
        "min_final_active_surface_covered_bin_fraction",
        "min_final_active_surface_mean_bin_covered_fraction",
        "min_final_material_visible_surface_covered_bin_fraction",
        "min_final_material_visible_surface_mean_bin_covered_fraction",
    ):
        expect_finite(failures, f"robustness.{field}", robustness.get(field))
    if "all_material_visible_surface_tail_bounded" in robustness:
        expect_bool(
            failures,
            "robustness.all_material_visible_surface_tail_bounded",
            robustness.get("all_material_visible_surface_tail_bounded"),
        )
    for field in (
        "max_final_material_visible_surface_tail_p99_distance",
        "max_final_material_visible_surface_tail_over_threshold_fraction",
        "max_final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
    ):
        if field in robustness:
            expect_finite(failures, f"robustness.{field}", robustness.get(field))
    if (robustness.get("min_active_seed_count") or 0) <= 0:
        failures.append(
            "robustness minimum active seed count should be nonzero "
            f"({robustness.get('min_active_seed_count')!r})"
        )
    min_final_active_count = robustness.get("min_final_active_count") or 0
    if min_final_active_count < int(PARTICLES * MIN_FINAL_ACTIVE_FRACTION):
        failures.append(
            "robustness minimum final active count is too low "
            f"({min_final_active_count!r} < {int(PARTICLES * MIN_FINAL_ACTIVE_FRACTION)})"
        )
    if (
        robustness.get("min_newly_activated_fraction") or 0.0
    ) < MIN_NEWLY_ACTIVATED_FRACTION:
        failures.append(
            "robustness minimum newly activated fraction is too low "
            f"({robustness.get('min_newly_activated_fraction')!r} "
            f"< {MIN_NEWLY_ACTIVATED_FRACTION})"
        )
    if (robustness.get("min_active_growth_ratio") or 0.0) < MIN_ACTIVE_GROWTH_RATIO:
        failures.append(
            "robustness minimum active growth ratio is too low "
            f"({robustness.get('min_active_growth_ratio')!r} "
            f"< {MIN_ACTIVE_GROWTH_RATIO})"
        )
    for field in ("min_active_extent_bbox_ratio", "min_active_extent_min_axis_ratio"):
        if field in robustness:
            expect_finite(failures, f"robustness.{field}", robustness.get(field))
    if (
        robustness.get("min_perturbed_newly_activated_fraction") or 0.0
    ) < MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION:
        failures.append(
            "robustness minimum perturbed newly activated fraction is too low "
            f"({robustness.get('min_perturbed_newly_activated_fraction')!r} "
            f"< {MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION})"
        )
    min_perturbed_count_ratio = robustness.get("min_perturbed_active_count_ratio") or 0.0
    if min_perturbed_count_ratio < MIN_PERTURBED_ACTIVE_COUNT_RATIO:
        failures.append(
            "robustness minimum perturbed active-count ratio is too low "
            f"({min_perturbed_count_ratio!r} < {MIN_PERTURBED_ACTIVE_COUNT_RATIO})"
        )
    max_perturbed_count_ratio = robustness.get("max_perturbed_active_count_ratio") or 0.0
    if max_perturbed_count_ratio > MAX_PERTURBED_ACTIVE_COUNT_RATIO:
        failures.append(
            "robustness maximum perturbed active-count ratio is too high "
            f"({max_perturbed_count_ratio!r} > {MAX_PERTURBED_ACTIVE_COUNT_RATIO})"
        )
    min_perturbed_motion_ratio = robustness.get("min_perturbed_peak_motion_ratio") or 0.0
    if min_perturbed_motion_ratio < MIN_PERTURBED_PEAK_MOTION_RATIO:
        failures.append(
            "robustness minimum perturbed peak-motion ratio is too low "
            f"({min_perturbed_motion_ratio!r} < {MIN_PERTURBED_PEAK_MOTION_RATIO})"
        )
    max_perturbed_motion_ratio = robustness.get("max_perturbed_peak_motion_ratio") or 0.0
    if max_perturbed_motion_ratio > MAX_PERTURBED_PEAK_MOTION_RATIO:
        failures.append(
            "robustness maximum perturbed peak-motion ratio is too high "
            f"({max_perturbed_motion_ratio!r} > {MAX_PERTURBED_PEAK_MOTION_RATIO})"
        )
    if (robustness.get("min_final_active_color_state_mean_abs") or 0.0) < MIN_COLOR_STATE_MEAN_ABS:
        failures.append(
            "robustness min active color-state mean did not emerge "
            f"({robustness.get('min_final_active_color_state_mean_abs')!r} "
            f"< {MIN_COLOR_STATE_MEAN_ABS})"
        )
    if (
        robustness.get("min_final_active_color_state_stddev_mean") or 0.0
    ) < MIN_COLOR_STATE_STDDEV_MEAN:
        failures.append(
            "robustness min active color-state stddev is too uniform "
            f"({robustness.get('min_final_active_color_state_stddev_mean')!r} "
            f"< {MIN_COLOR_STATE_STDDEV_MEAN})"
        )
    expect_le(
        failures,
        "robustness.max_gaussian_scale_budget_loss",
        robustness.get("max_gaussian_scale_budget_loss"),
        MAX_GAUSSIAN_SCALE_BUDGET_LOSS,
    )
    expect_le(
        failures,
        "robustness.max_gaussian_oversize_fraction",
        robustness.get("max_gaussian_oversize_fraction"),
        MAX_GAUSSIAN_OVERSIZE_FRACTION,
    )
    for seed in robustness.get("seeds") or []:
        seed_label = f"robustness.seed[{seed.get('seed')!r}]"
        expect_true(
            failures,
            f"{seed_label}.gaussian_scale_budget",
            seed.get("gaussian_scale_budget"),
        )
        expect_le(
            failures,
            f"{seed_label}.gaussian_scale_budget_loss",
            seed.get("gaussian_scale_budget_loss"),
            MAX_GAUSSIAN_SCALE_BUDGET_LOSS,
        )
        expect_le(
            failures,
            f"{seed_label}.gaussian_oversize_fraction",
            seed.get("gaussian_oversize_fraction"),
            MAX_GAUSSIAN_OVERSIZE_FRACTION,
        )
        expect_bool(
            failures,
            f"{seed_label}.surface_normal_coverage",
            seed.get("surface_normal_coverage"),
        )
        expect_bool(
            failures,
            f"{seed_label}.surface_coverage_profile",
            seed.get("surface_coverage_profile"),
        )
        expect_bool(
            failures,
            f"{seed_label}.material_visible_surface_coverage_profile",
            seed.get("material_visible_surface_coverage_profile"),
        )
        if "material_visible_particles_live" in seed:
            expect_bool(
                failures,
                f"{seed_label}.material_visible_particles_live",
                seed.get("material_visible_particles_live"),
            )
        if "inactive_material_visible_fraction" in seed:
            expect_finite(
                failures,
                f"{seed_label}.inactive_material_visible_fraction",
                seed.get("inactive_material_visible_fraction"),
            )
        if seed.get("max_inactive_material_opacity") is not None:
            expect_finite(
                failures,
                f"{seed_label}.max_inactive_material_opacity",
                seed.get("max_inactive_material_opacity"),
            )
        expect_finite(
            failures,
            f"{seed_label}.final_active_surface_normal_covered_bin_fraction",
            seed.get("final_active_surface_normal_covered_bin_fraction"),
        )
        expect_finite(
            failures,
            f"{seed_label}.final_active_surface_normal_mean_bin_covered_fraction",
            seed.get("final_active_surface_normal_mean_bin_covered_fraction"),
        )
        for field in (
            "final_active_surface_covered_bin_fraction",
            "final_active_surface_mean_bin_covered_fraction",
            "final_material_visible_surface_covered_bin_fraction",
            "final_material_visible_surface_mean_bin_covered_fraction",
        ):
            expect_finite(failures, f"{seed_label}.{field}", seed.get(field))
        if "material_visible_surface_tail_bounded" in seed:
            expect_bool(
                failures,
                f"{seed_label}.material_visible_surface_tail_bounded",
                seed.get("material_visible_surface_tail_bounded"),
            )
        for field in (
            "final_material_visible_surface_tail_p99_distance",
            "final_material_visible_surface_tail_over_threshold_fraction",
            "final_material_visible_surface_tail_opacity_weighted_over_threshold_fraction",
        ):
            if field in seed:
                expect_finite(failures, f"{seed_label}.{field}", seed.get(field))


def assert_surface_normal_coverage(
    failures: list[str], prefix: str, report: dict[str, Any]
) -> None:
    expect_ge(failures, f"{prefix}.samples", report.get("samples"), 1)
    expect_ge(failures, f"{prefix}.normal_bins", report.get("normal_bins"), 1)
    expect_ge(failures, f"{prefix}.target_bins", report.get("target_bins"), 1)
    expect_finite(failures, f"{prefix}.threshold", report.get("threshold"))
    for field in (
        "covered_target_bin_fraction",
        "covered_sample_fraction",
        "min_bin_covered_fraction",
        "mean_bin_covered_fraction",
        "max_bin_covered_fraction",
    ):
        value = report.get(field)
        expect_finite(failures, f"{prefix}.{field}", value)
        expect_ge(failures, f"{prefix}.{field}", value, 0.0)
        expect_le(failures, f"{prefix}.{field}", value, 1.0)


def assert_surface_coverage_profile(
    failures: list[str], prefix: str, report: dict[str, Any]
) -> None:
    expect_ge(failures, f"{prefix}.samples", report.get("samples"), 1)
    expect_ge(failures, f"{prefix}.bins", report.get("bins"), 1)
    expect_finite(failures, f"{prefix}.threshold", report.get("threshold"))
    expect_ge(failures, f"{prefix}.empty_bins", report.get("empty_bins"), 0)
    for field in (
        "covered_fraction",
        "covered_bin_fraction",
        "min_bin_covered_fraction",
        "mean_bin_covered_fraction",
        "max_bin_covered_fraction",
        "assigned_particle_fraction",
        "covered_assigned_particle_fraction",
        "max_assigned_sample_fraction",
        "max_covered_assigned_sample_fraction",
    ):
        value = report.get(field)
        expect_finite(failures, f"{prefix}.{field}", value)
        expect_ge(failures, f"{prefix}.{field}", value, 0.0)
        expect_le(failures, f"{prefix}.{field}", value, 1.0)


def assert_surface_tail_report(
    failures: list[str], prefix: str, report: dict[str, Any]
) -> None:
    expect_ge(failures, f"{prefix}.count", report.get("count"), 0)
    expect_finite(failures, f"{prefix}.threshold", report.get("threshold"))
    for field in (
        "p95_distance",
        "p99_distance",
        "max_distance",
        "opacity_weighted_mean_distance",
    ):
        value = report.get(field)
        if value is not None:
            expect_finite(failures, f"{prefix}.{field}", value)
    for field in (
        "over_threshold_fraction",
        "opacity_weighted_over_threshold_fraction",
    ):
        value = report.get(field)
        expect_finite(failures, f"{prefix}.{field}", value)
        expect_ge(failures, f"{prefix}.{field}", value, 0.0)
        expect_le(failures, f"{prefix}.{field}", value, 1.0)


def assert_gaussian_volume(
    failures: list[str], prefix: str, volume: dict[str, Any]
) -> None:
    expect_finite(failures, f"{prefix}.expected_scale", volume.get("expected_scale"))
    expect_finite(failures, f"{prefix}.max_expected_scale", volume.get("max_expected_scale"))
    expect_finite(failures, f"{prefix}.mean_scale", volume.get("mean_scale"))
    expect_finite(failures, f"{prefix}.max_scale", volume.get("max_scale"))
    expect_finite(
        failures,
        f"{prefix}.scale_budget_loss",
        volume.get("scale_budget_loss"),
    )
    expect_finite(
        failures,
        f"{prefix}.oversize_fraction",
        volume.get("oversize_fraction"),
    )
    expect_le(
        failures,
        f"{prefix}.scale_budget_loss",
        volume.get("scale_budget_loss"),
        MAX_GAUSSIAN_SCALE_BUDGET_LOSS,
    )
    expect_le(
        failures,
        f"{prefix}.oversize_fraction",
        volume.get("oversize_fraction"),
        MAX_GAUSSIAN_OVERSIZE_FRACTION,
    )


def nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def expect_equal(failures: list[str], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{field}: expected {expected!r}, got {actual!r}")


def expect_close(
    failures: list[str],
    field: str,
    actual: Any,
    expected: float,
    tolerance: float,
) -> None:
    if not isinstance(actual, int | float) or abs(float(actual) - expected) > tolerance:
        failures.append(
            f"{field}: expected {expected!r} +/- {tolerance}, got {actual!r}"
        )


def expect_true(failures: list[str], field: str, actual: Any) -> None:
    if actual is not True:
        failures.append(f"{field}: expected true, got {actual!r}")


def expect_false(failures: list[str], field: str, actual: Any) -> None:
    if actual is not False:
        failures.append(f"{field}: expected false, got {actual!r}")


def expect_bool(failures: list[str], field: str, actual: Any) -> None:
    if not isinstance(actual, bool):
        failures.append(f"{field}: expected boolean, got {actual!r}")


def expect_le(failures: list[str], field: str, actual: Any, maximum: Any) -> None:
    if (
        not isinstance(actual, int | float)
        or not isinstance(maximum, int | float)
        or float(actual) > float(maximum)
    ):
        failures.append(f"{field}: expected <= {maximum!r}, got {actual!r}")


def expect_ge(failures: list[str], field: str, actual: Any, minimum: Any) -> None:
    if (
        not isinstance(actual, int | float)
        or not isinstance(minimum, int | float)
        or float(actual) < float(minimum)
    ):
        failures.append(f"{field}: expected >= {minimum!r}, got {actual!r}")


def expect_finite(failures: list[str], field: str, actual: Any) -> None:
    if not isinstance(actual, int | float) or not float("-inf") < float(actual) < float("inf"):
        failures.append(f"{field}: expected finite number, got {actual!r}")


if __name__ == "__main__":
    main()
