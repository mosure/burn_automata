"""Cargo invocation and report loading for 3D catalog validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    APP_EVAL_SEED,
    HELD_OUT_SEEDS,
    IMAGE_SIZE,
    PARTICLES,
    TARGET_SAMPLES,
    ValidationCase,
)


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
