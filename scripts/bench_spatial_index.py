#!/usr/bin/env python3
"""Run CPU spatial-index strategy analyses for NPA neighbor search."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "target" / "bench_spatial_index.json"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)")

DEFAULT_PRESETS = ["growing-2d", "texture-2d", "growing-3d-gs"]
DEFAULT_PARTICLES = [1024, 4096, 8192]
DEFAULT_GEOMETRIES = ["seed", "dense", "uniform", "line", "plane", "torus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target" / "release" / "burn_automata")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preset", action="append", dest="presets")
    parser.add_argument("--particles", type=int, action="append")
    parser.add_argument("--geometry", action="append", dest="geometries")
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--bvh-leaf-size", type=int, default=16)
    parser.add_argument("--tile-size", default="2,2,1")
    parser.add_argument("--seed-mode", default="uniform-circle")
    parser.add_argument("--seed-scale", type=float)
    parser.add_argument("--reference-seed-scale", type=float)
    parser.add_argument("--fixed-eps", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
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


def parse_cli_output(output: str) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        rows.append({key: value.strip().strip('"') for key, value in KEY_VALUE_RE.findall(line)})
    return rows


def coerce(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = dict(row)
    int_keys = {
        "particles",
        "dim",
        "active_bins",
        "max_bin_occupancy",
        "node_count",
        "max_depth",
        "exact_neighbor_pairs",
        "candidate_tests",
        "candidate_entries_visited",
    }
    float_keys = {
        "eps",
        "analyze_ms",
        "candidates_per_particle",
        "entries_per_particle",
        "exact_neighbors_per_particle",
        "node_visits_per_particle",
    }
    for key in int_keys & out.keys():
        out[key] = int(str(out[key]))
    for key in float_keys & out.keys():
        out[key] = float(str(out[key]))
    return out


def run_case(
    args: argparse.Namespace,
    preset: str,
    particles: int,
    geometry: str,
) -> list[dict[str, object]]:
    command = [
        str(args.binary),
        "bench-spatial",
        "--preset",
        preset,
        "--particles",
        str(particles),
        "--geometry",
        geometry,
        "--strategy",
        args.strategy,
        "--bvh-leaf-size",
        str(args.bvh_leaf_size),
        "--tile-size",
        args.tile_size,
        "--seed-mode",
        args.seed_mode,
    ]
    if args.seed_scale is not None:
        command.extend(["--seed-scale", str(args.seed_scale)])
    if args.reference_seed_scale is not None:
        command.extend(["--reference-seed-scale", str(args.reference_seed_scale)])
    if args.fixed_eps:
        command.append("--fixed-eps")
    else:
        command.append("--normalize-seed-scale")

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    rows = []
    if result.stdout:
        for parsed in parse_cli_output(result.stdout):
            row = coerce(parsed)
            row.update(
                {
                    "requested_preset": preset,
                    "requested_particles": particles,
                    "requested_geometry": geometry,
                    "command": " ".join(command),
                    "returncode": result.returncode,
                    "wall_ms": wall_ms,
                    "stderr": result.stderr.strip(),
                }
            )
            rows.append(row)
    if not rows:
        rows.append(
            {
                "requested_preset": preset,
                "requested_particles": particles,
                "requested_geometry": geometry,
                "command": " ".join(command),
                "returncode": result.returncode,
                "wall_ms": wall_ms,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    return rows


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
    print("\nSpatial-index summary:")
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("returncode") != 0 or "error" in row:
            continue
        key = (
            str(row.get("requested_preset", row.get("preset", ""))),
            int(row.get("requested_particles", row.get("particles", 0))),
            str(row.get("requested_geometry", row.get("geometry", ""))),
        )
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        exact = grouped[key][0].get("exact_neighbors_per_particle", 0.0)
        best_candidates = min(
            grouped[key],
            key=lambda row: float(row.get("candidates_per_particle", float("inf"))),
        )
        best_time = min(
            grouped[key],
            key=lambda row: float(row.get("analyze_ms", float("inf"))),
        )
        print(
            f"{key[0]:14s} n={key[1]:5d} {key[2]:7s} "
            f"neighbors={float(exact):8.3f} "
            f"min-candidates={best_candidates.get('strategy')}:{float(best_candidates.get('candidates_per_particle', 0.0)):8.3f} "
            f"fast-analysis={best_time.get('strategy')}:{float(best_time.get('analyze_ms', 0.0)):8.3f} ms"
        )


def main() -> int:
    args = parse_args()
    if not args.no_build:
        build_binary(args.binary)
    presets = args.presets or DEFAULT_PRESETS
    particles = args.particles or DEFAULT_PARTICLES
    geometries = args.geometries or DEFAULT_GEOMETRIES
    rows: list[dict[str, object]] = []
    total = len(presets) * len(particles) * len(geometries)
    index = 0
    for preset in presets:
        for particle_count in particles:
            for geometry in geometries:
                index += 1
                print(f"[{index}/{total}] {preset} particles={particle_count} geometry={geometry}")
                try:
                    rows.extend(run_case(args, preset, particle_count, geometry))
                except subprocess.TimeoutExpired as err:
                    rows.append(
                        {
                            "requested_preset": preset,
                            "requested_particles": particle_count,
                            "requested_geometry": geometry,
                            "returncode": -1,
                            "stderr": f"timeout after {args.timeout}s: {err}",
                        }
                    )
    write_reports(rows, args.output)
    print_summary(rows)
    print(f"\nWrote {args.output} and {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

