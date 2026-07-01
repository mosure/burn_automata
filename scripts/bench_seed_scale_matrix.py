#!/usr/bin/env python3
"""Benchmark seed-scale sensitivity for WGPU NPA inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "target" / "bench_seed_scale_matrix.json"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target" / "release" / "burn_automata")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preset", default="growing-3d-gs")
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-mode", default="torus-morphogen-dense-3d")
    parser.add_argument("--reference-seed-scale", type=float, default=0.72)
    parser.add_argument("--seed-scale", type=float, action="append", dest="seed_scales")
    parser.add_argument("--geometry", action="append", dest="geometries")
    parser.add_argument("--neighbor-mode", default="auto")
    parser.add_argument("--bucket-capacity", type=int)
    parser.add_argument("--fixed-eps", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--gaussian", action="store_true", default=True)
    return parser.parse_args()


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


def parse_cli_output(output: str) -> dict[str, str]:
    return {key: value.strip() for key, value in KEY_VALUE_RE.findall(output.strip())}


def coerce(values: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = dict(values)
    int_keys = {
        "particles",
        "steps",
        "repeats",
        "bucket_capacity",
        "grid_storage_u32",
        "grid_clear_u32",
        "grid_overflow_count",
        "initial_nonempty_cells",
        "initial_max_cell_occupancy",
    }
    float_keys = {
        "elapsed_ms",
        "gpu_step_ms",
        "avg_step_ms",
        "min_avg_step_ms",
        "median_avg_step_ms",
        "max_avg_step_ms",
        "final_mean_displacement_per_step",
        "final_mean_density",
        "hashgrid_eps",
        "reference_seed_scale",
    }
    for key in int_keys & out.keys():
        out[key] = int(str(out[key]))
    for key in float_keys & out.keys():
        out[key] = float(str(out[key]))
    return out


def run_case(args: argparse.Namespace, scale: float, geometry: str) -> dict[str, object]:
    command = [
        str(args.binary),
        "bench",
        "--preset",
        args.preset,
        "--particles",
        str(args.particles),
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--gpu",
        "--seed-mode",
        args.seed_mode,
        "--geometry",
        geometry,
        "--seed-scale",
        str(scale),
        "--reference-seed-scale",
        str(args.reference_seed_scale),
        "--neighbor-mode",
        args.neighbor_mode,
    ]
    if args.bucket_capacity is not None:
        command.extend(["--bucket-capacity", str(args.bucket_capacity)])
    if args.fixed_eps:
        command.append("--fixed-eps")
    else:
        command.append("--normalize-seed-scale")
    if args.gaussian:
        command.append("--gaussian")

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    row: dict[str, object] = {
        "seed_scale": scale,
        "geometry": geometry,
        "fixed_eps": args.fixed_eps,
        "command": " ".join(command),
        "returncode": result.returncode,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0:
        row.update(coerce(parse_cli_output(result.stdout)))
        gpu_step_ms = float(row.get("gpu_step_ms", 0.0))
        if gpu_step_ms > 0.0:
            row["million_particles_per_second"] = (
                args.particles * args.steps / (gpu_step_ms / 1000.0) / 1_000_000.0
            )
        row["overflow_ok"] = int(row.get("grid_overflow_count", 0)) == 0
    return row


def write_reports(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    csv_output = output.with_suffix(".csv")
    fieldnames = sorted({key for row in rows for key in row if key not in {"stdout", "stderr"}})
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nSeed-scale summary:")
    for row in rows:
        if row.get("returncode") != 0:
            print(f"{row['geometry']:8s} scale={row['seed_scale']:<5} failed")
            continue
        print(
            f"{str(row['geometry']):8s} scale={float(row['seed_scale']):5.2f} "
            f"avg={float(row.get('avg_step_ms', 0.0)):8.3f} ms "
            f"eps={float(row.get('hashgrid_eps', 0.0)):8.5f} "
            f"cells={int(row.get('initial_nonempty_cells', 0)):5d} "
            f"maxocc={int(row.get('initial_max_cell_occupancy', 0)):4d} "
            f"overflow={row.get('grid_overflow_count', '?')}"
        )


def main() -> int:
    args = parse_args()
    if not args.no_build:
        build_binary(args.binary)
    scales = args.seed_scales or [0.04, 0.08, 0.16, 0.32, 0.72, 1.2]
    geometries = args.geometries or ["seed", "dense", "uniform", "line", "plane", "torus"]
    rows = []
    total = len(scales) * len(geometries)
    for index, geometry in enumerate(geometries, start=1):
        for scale_index, scale in enumerate(scales, start=1):
            print(f"[{(index - 1) * len(scales) + scale_index}/{total}] {geometry} scale={scale}")
            rows.append(run_case(args, scale, geometry))
    write_reports(rows, args.output)
    print_summary(rows)
    print(f"\nWrote {args.output} and {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
