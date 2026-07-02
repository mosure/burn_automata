#!/usr/bin/env python3
"""Run WGPU inference benchmark sweeps and save parseable reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "target" / "bench_gpu_matrix.json"
DENSITY_GEOMETRIES = ["seed", "uniform", "dense", "line", "point", "micro-cluster"]


@dataclass(frozen=True)
class Mode:
    name: str
    bucket_capacity: int | None = None

    @property
    def label(self) -> str:
        if self.bucket_capacity is None:
            return self.name
        return f"{self.name}:{self.bucket_capacity}"


QUICK_MATRIX = {
    "growing-3d-gs": {
        "particles": [4096, 16384, 32768],
        "steps": 16,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 64),
            Mode("tiled-fixed-buckets", 128),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("bvh", 32),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
            Mode("gpu-bvh", 32),
            Mode("gpu-lbvh", 32),
        ],
    },
    "texture-2d": {
        "particles": [4096, 16384, 32768],
        "steps": 16,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 128),
            Mode("tiled-fixed-buckets", 128),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
        ],
    },
    "growing-2d": {
        "particles": [1024, 4096, 8192],
        "steps": 8,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 256),
            Mode("tiled-fixed-buckets", 256),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("bvh", 32),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
            Mode("gpu-bvh", 32),
            Mode("gpu-lbvh", 32),
        ],
    },
    "point-mnist": {
        "particles": [4096, 16384],
        "steps": 16,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 128),
            Mode("tiled-fixed-buckets", 128),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
        ],
    },
}

FULL_MATRIX = {
    "growing-3d-gs": {
        "particles": [4096, 8192, 16384, 32768, 65536],
        "steps": 24,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 64),
            Mode("fixed-buckets", 128),
            Mode("tiled-fixed-buckets", 128),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("bvh", 32),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
            Mode("gpu-bvh", 32),
            Mode("gpu-lbvh", 32),
        ],
    },
    "texture-2d": {
        "particles": [1024, 4096, 8192, 16384, 32768, 65536],
        "steps": 24,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 64),
            Mode("fixed-buckets", 128),
            Mode("fixed-buckets", 256),
            Mode("tiled-fixed-buckets", 128),
            Mode("tiled-fixed-buckets", 256),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
        ],
    },
    "growing-2d": {
        "particles": [1024, 2048, 4096, 8192, 16384, 32768],
        "steps": 12,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 128),
            Mode("fixed-buckets", 256),
            Mode("fixed-buckets", 512),
            Mode("tiled-fixed-buckets", 256),
            Mode("tiled-fixed-buckets", 512),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("bvh", 32),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
            Mode("gpu-bvh", 32),
            Mode("gpu-lbvh", 32),
        ],
    },
    "point-mnist": {
        "particles": [1024, 4096, 8192, 16384, 32768],
        "steps": 24,
        "modes": [
            Mode("auto"),
            Mode("linked-list"),
            Mode("fixed-buckets", 64),
            Mode("fixed-buckets", 128),
            Mode("tiled-fixed-buckets", 128),
            Mode("sorted-cells"),
            Mode("cooperative-sorted-cells"),
            Mode("bvh", 16),
            Mode("gpu-bvh", 16),
            Mode("gpu-lbvh", 16),
            Mode("gpu-morton-lbvh", 16),
        ],
    },
}


KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=["quick", "full"], default="quick")
    parser.add_argument("--preset", action="append", dest="presets")
    parser.add_argument("--particles", type=int, action="append")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--geometry", action="append", dest="geometries", default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        help="Mode override: auto, linked-list, fixed-buckets:128, tiled-fixed-buckets:128, sorted-cells, cooperative-sorted-cells, bvh:16, gpu-bvh:16, gpu-lbvh:16, gpu-morton-lbvh:16.",
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--fail-on-overflow", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--step-timing", action="store_true")
    parser.add_argument(
        "--density-suite",
        action="store_true",
        help="Run seed, uniform, dense, line, point, and micro-cluster geometries unless --geometry is supplied.",
    )
    parser.add_argument("--extra-env", action="append", default=[])
    return parser.parse_args()


def parse_mode(value: str) -> Mode:
    if ":" not in value:
        if value in {"fixed", "fixed-buckets", "buckets"}:
            raise SystemExit(f"{value} requires a capacity, for example {value}:128")
        return Mode(value)
    name, raw_capacity = value.split(":", 1)
    if name == "fixed":
        name = "fixed-buckets"
    return Mode(name, int(raw_capacity))


def parse_cli_output(output: str) -> dict[str, str]:
    values = {}
    for key, value in KEY_VALUE_RE.findall(output.strip()):
        values[key] = value.strip()
    return values


def coerce_values(values: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = dict(values)
    for key in [
        "particles",
        "steps",
        "repeats",
        "bucket_capacity",
        "grid_storage_u32",
        "grid_clear_u32",
        "grid_overflow_count",
        "grid_max_overflow_count",
        "grid_overflowed_steps",
    ]:
        if key in out:
            out[key] = int(str(out[key]))
    for key in [
        "elapsed_ms",
        "gpu_step_ms",
        "avg_step_ms",
        "min_avg_step_ms",
        "median_avg_step_ms",
        "max_avg_step_ms",
        "final_mean_displacement_per_step",
        "final_mean_density",
        "step_min_ms",
        "step_median_ms",
        "step_p95_ms",
        "step_p99_ms",
        "step_max_ms",
        "step_jitter_ratio",
    ]:
        if key in out:
            out[key] = float(str(out[key]))
    return out


def build_binary(binary: Path) -> None:
    if binary.exists():
        return
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "burn_automata",
            "--features",
            "gpu_wgpu",
            "--bin",
            "burn_automata",
        ],
        cwd=ROOT,
        check=True,
    )


def run_case(
    binary: Path,
    preset: str,
    particles: int,
    steps: int,
    repeats: int,
    mode: Mode,
    geometry: str,
    model: Path | None,
    gaussian: bool,
    step_timing: bool,
    env: dict[str, str],
    timeout: float,
) -> dict[str, object]:
    command = [
        str(binary),
        "bench",
        "--preset",
        preset,
        "--particles",
        str(particles),
        "--steps",
        str(steps),
        "--repeats",
        str(repeats),
        "--gpu",
        "--geometry",
        geometry,
        "--neighbor-mode",
        mode.name,
    ]
    if model is not None:
        command.extend(["--model", str(model)])
    if mode.bucket_capacity is not None:
        command.extend(["--bucket-capacity", str(mode.bucket_capacity)])
    if gaussian:
        command.append("--gaussian")
    if step_timing:
        command.append("--step-timing")

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    row: dict[str, object] = {
        "preset": preset,
        "requested_particles": particles,
        "requested_steps": steps,
        "requested_mode": mode.label,
        "requested_geometry": geometry,
        "requested_model": str(model) if model is not None else "",
        "requested_gaussian": gaussian,
        "command": " ".join(command),
        "returncode": result.returncode,
        "wall_ms": wall_ms,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0:
        row.update(coerce_values(parse_cli_output(result.stdout)))
        gpu_step_ms = float(row.get("gpu_step_ms", 0.0))
        if gpu_step_ms > 0.0:
            row["particles_per_second"] = particles * steps / (gpu_step_ms / 1000.0)
            row["million_particles_per_second"] = row["particles_per_second"] / 1_000_000.0
        max_overflow = int(row.get("grid_max_overflow_count", row.get("grid_overflow_count", 0)))
        row["overflow_ok"] = max_overflow == 0
    return row


def selected_cases(args: argparse.Namespace) -> list[tuple[str, int, int, Mode, str]]:
    matrix = FULL_MATRIX if args.matrix == "full" else QUICK_MATRIX
    presets = args.presets or list(matrix.keys())
    modes_override = [parse_mode(value) for value in args.modes] if args.modes else None
    cases = []
    for preset in presets:
        if preset not in matrix:
            raise SystemExit(f"unknown preset {preset!r}; known presets: {', '.join(matrix)}")
        spec = matrix[preset]
        particles = args.particles or spec["particles"]
        steps = args.steps or spec["steps"]
        modes = modes_override or spec["modes"]
        geometries = args.geometries or (DENSITY_GEOMETRIES if args.density_suite else ["seed"])
        for particle_count in particles:
            for mode in modes:
                for geometry in geometries:
                    cases.append((preset, int(particle_count), int(steps), mode, geometry))
    return cases


def write_reports(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    csv_output = output.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in {"stdout", "stderr"}})
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nFastest non-overflow case per preset/particle count:")
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("returncode") != 0 or not row.get("overflow_ok", False):
            continue
        key = (
            str(row["preset"]),
            int(row["requested_particles"]),
            str(row.get("requested_geometry", "seed")),
            str(row.get("requested_model", "")),
        )
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        best = min(grouped[key], key=lambda row: float(row.get("avg_step_ms", float("inf"))))
        print(
            f"{key[0]:14s} {key[1]:6d} particles {key[2]:7s}  "
            f"{best['requested_mode']:18s} {float(best['avg_step_ms']):9.4f} ms/step  "
            f"{float(best.get('million_particles_per_second', 0.0)):8.3f} M particles/s  "
            f"resolved={best.get('neighbor_mode')} cap={best.get('bucket_capacity')} "
            f"model={key[3] or '<seeded>'}"
        )

    failed = [row for row in rows if row.get("returncode") != 0]
    overflowed = [row for row in rows if row.get("returncode") == 0 and not row.get("overflow_ok", False)]
    if failed:
        print(f"\nFailed cases: {len(failed)}")
    if overflowed:
        print(f"Overflowed fixed-bucket cases: {len(overflowed)}")


def main() -> int:
    args = parse_args()
    binary = args.binary or (ROOT / "target" / "release" / "burn_automata")
    if not args.no_build:
        build_binary(binary)

    env = os.environ.copy()
    for item in args.extra_env:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--extra-env expects KEY=VALUE, got {item!r}")
        env[key] = value

    rows = []
    cases = selected_cases(args)
    for index, (preset, particles, steps, mode, geometry) in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {preset} particles={particles} "
            f"steps={steps} mode={mode.label} geometry={geometry}"
        )
        try:
            row = run_case(
                binary,
                preset,
                particles,
                steps,
                args.repeats,
                mode,
                geometry,
                args.model,
                args.gaussian,
                args.step_timing,
                env,
                args.timeout,
            )
        except subprocess.TimeoutExpired as err:
            row = {
                "preset": preset,
                "requested_particles": particles,
                "requested_steps": steps,
                "requested_mode": mode.label,
                "requested_geometry": geometry,
                "returncode": -1,
                "wall_ms": args.timeout * 1000.0,
                "stdout": err.stdout or "",
                "stderr": f"timeout after {args.timeout}s",
            }
        rows.append(row)
        if row.get("returncode") == 0:
            print(
                f"    avg={float(row.get('avg_step_ms', 0.0)):.4f} ms "
                f"p99={float(row.get('step_p99_ms', 0.0)):.4f} ms "
                f"max={float(row.get('step_max_ms', 0.0)):.4f} ms "
                f"overflow={row.get('grid_max_overflow_count', row.get('grid_overflow_count', 'n/a'))} "
                f"resolved={row.get('neighbor_mode')}"
            )
        else:
            print(f"    failed: {row.get('stderr')}")

    write_reports(rows, args.output)
    print(f"\nwrote {args.output}")
    print(f"wrote {args.output.with_suffix('.csv')}")
    print_summary(rows)

    if args.fail_on_overflow:
        bad = [
            row
            for row in rows
            if row.get("returncode") != 0 or not row.get("overflow_ok", False)
        ]
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
