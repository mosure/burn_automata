"""Report assertions for app-scale 3D catalog validation."""

from __future__ import annotations

from typing import Any

from .config import (
    APP_EVAL_SEED,
    HELD_OUT_SEEDS,
    MAX_GAUSSIAN_OVERSIZE_FRACTION,
    MAX_GAUSSIAN_SCALE_BUDGET_LOSS,
    MAX_PERTURBED_ACTIVE_COUNT_RATIO,
    MAX_PERTURBED_PEAK_MOTION_RATIO,
    MIN_ACTIVE_GROWTH_RATIO,
    MIN_COLOR_STATE_MEAN_ABS,
    MIN_COLOR_STATE_STDDEV_MEAN,
    MIN_FINAL_ACTIVE_FRACTION,
    MIN_NEWLY_ACTIVATED_FRACTION,
    MIN_PERTURBED_ACTIVE_COUNT_RATIO,
    MIN_PERTURBED_NEWLY_ACTIVATED_FRACTION,
    MIN_PERTURBED_PEAK_MOTION_RATIO,
    NON_OPACITY_SEED_TOLERANCE,
    PARTICLES,
    ValidationCase,
)
from .expect import (
    expect_bool,
    expect_close,
    expect_equal,
    expect_false,
    expect_finite,
    expect_ge,
    expect_le,
    expect_true,
)


def is_strict_seed_lineage_eligible(report: dict[str, Any]) -> bool:
    strict_checks = report.get("strict_checks") or {}
    return (
        bool(report.get("local_conditionless_lineage"))
        and not bool(report.get("position_features"))
        and not bool(report.get("seed_coordinate_scaffold"))
        and bool(strict_checks.get("local_conditionless_lineage"))
        and bool(strict_checks.get("no_position_features"))
        and bool(strict_checks.get("no_seed_coordinate_scaffold"))
        and bool(strict_checks.get("neutral_non_opacity_seed_state"))
    )


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
    expect_true(
        failures,
        "strict local_conditionless_lineage",
        strict_checks.get("local_conditionless_lineage"),
    )
    expect_true(failures, "strict no_position_features", strict_checks.get("no_position_features"))
    expect_false(failures, "position_features", report.get("position_features"))
    expect_equal(
        failures,
        "strict_seed_lineage_eligible",
        is_strict_seed_lineage_eligible(report),
        case.strict_seed_lineage_eligible,
    )
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
