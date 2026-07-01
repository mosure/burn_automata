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
MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION = 0.5
MIN_PERTURBED_ACTIVE_COUNT_RATIO = 0.5
MAX_PERTURBED_ACTIVE_COUNT_RATIO = 2.0
MIN_PERTURBED_PEAK_MOTION_RATIO = 0.25
MAX_PERTURBED_PEAK_MOTION_RATIO = 4.0


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
            "render-refined-rust:ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset({"render_loss_passed"}),
        allowed_failure_reasons=frozenset({"render_loss_passed"}),
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
            "render-refined-rust:ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset({"render_loss_passed"}),
        allowed_failure_reasons=frozenset({"render_loss_passed"}),
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
                "torus_angular_coverage",
                "render_loss_passed",
            }
        ),
        allowed_failure_reasons=frozenset(
            {
                "target_coverage_fraction",
                "torus_angular_coverage",
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
            {"target_coverage_fraction", "torus_angular_coverage", "render_loss_passed"}
        ),
        allowed_failure_reasons=frozenset(
            {
                "surface_mean_improved",
                "target_coverage_fraction",
                "torus_angular_coverage",
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


if __name__ == "__main__":
    main()
