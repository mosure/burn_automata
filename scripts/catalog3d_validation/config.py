"""Static validation configuration for shipped 3D catalog assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    strict_seed_lineage_eligible: bool
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
        strict_seed_lineage_eligible=False,
        expected_source=(
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:"
            "front_retime=false:active_opacity_hidden=skipped:"
            "active_opacity_gain=skipped:opacity_bias=skipped:"
            "material_opacity_bias=0.55:base=render-refined-rust:"
            "ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset(
            {"no_seed_coordinate_scaffold", "render_loss_passed"}
        ),
        allowed_failure_reasons=frozenset(
            {
                "no_seed_coordinate_scaffold",
                "temporal_activation_progressive",
                "surface_max_bounded",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "dormant_drift_bounded",
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
        strict_seed_lineage_eligible=False,
        expected_source=(
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:"
            "front_retime=false:active_opacity_hidden=skipped:"
            "active_opacity_gain=skipped:opacity_bias=skipped:"
            "material_opacity_bias=0.55:base=render-refined-rust:"
            "ablation-rust:utah-teapot-2026:"
            "conditionless-local-random-ball-rollout-ablation"
        ),
        required_failure_reasons=frozenset(
            {"no_seed_coordinate_scaffold", "render_loss_passed"}
        ),
        allowed_failure_reasons=frozenset(
            {
                "no_seed_coordinate_scaffold",
                "surface_max_bounded",
                "surface_tail_bounded",
                "surface_coverage_profile",
                "surface_normal_coverage",
                "active_extent_growth",
                "dormant_drift_bounded",
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
        strict_seed_lineage_eligible=False,
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
        strict_seed_lineage_eligible=False,
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
