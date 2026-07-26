#!/usr/bin/env python3
"""Validate all imported SelfOrg catalog models against the Python reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from import_selforg_catalog import import_entry, load_catalog, parameter_count, write_bpk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("scripts/reference/selforg/selforg_catalog.json"),
    )
    parser.add_argument("--web-root", type=Path, default=Path("/tmp/selforg_npa_web"))
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--psnr-threshold", type=float, default=70.0)
    parser.add_argument("--hidden-psnr-threshold", type=float, default=70.0)
    parser.add_argument("--binary", type=Path, default=Path("target/release/burn_automata"))
    parser.add_argument("--build-binary", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("/tmp/burn_automata_catalog_parity.json"))
    args = parser.parse_args()

    entries = load_catalog(args.catalog)
    if args.only:
        requested = set(args.only)
        entries = [entry for entry in entries if entry["slug"] in requested]
        missing = sorted(requested - {entry["slug"] for entry in entries})
        if missing:
            raise SystemExit(f"unknown catalog slug(s): {', '.join(missing)}")

    if args.build_binary:
        build_cmd = ["cargo", "build", "--release", "-p", "burn_automata", "--bin", "burn_automata"]
        if args.gpu:
            build_cmd[2:2] = ["--features", "gpu_wgpu"]
        subprocess.run(build_cmd, check=True)

    summary: list[dict[str, Any]] = []
    failures = 0
    with tempfile.TemporaryDirectory(prefix="burn_automata_catalog_metrics_") as tmp:
        tmp_path = Path(tmp)
        for entry in entries:
            output = Path(entry["output"])
            try:
                if args.force_import or not output.exists():
                    manifest = import_entry(args.web_root, entry)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = write_bpk(output, manifest)
                    print(
                        f"imported {entry['slug']}: {output} "
                        f"params={parameter_count(manifest)} sha256={digest}"
                    )
                metrics_path = tmp_path / f"{entry['slug']}.json"
                cmd = [
                    sys.executable,
                    "scripts/reference/selforg/validate_import_parity.py",
                    "--model",
                    str(output),
                    "--particles",
                    str(args.particles),
                    "--preset",
                    entry["preset"],
                    "--seed-scale",
                    str(entry["seed_scale"]),
                    "--steps",
                    str(args.steps),
                    "--tolerance",
                    str(args.tolerance),
                    "--psnr-threshold",
                    str(args.psnr_threshold),
                    "--hidden-psnr-threshold",
                    str(args.hidden_psnr_threshold),
                    "--binary",
                    str(args.binary),
                    "--metrics-output",
                    str(metrics_path),
                ]
                if args.gpu:
                    cmd.append("--gpu")
                    if args.binary.exists():
                        cmd.append("--use-binary-for-gpu")
                result = subprocess.run(cmd, text=True, capture_output=True)
                if result.stdout:
                    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
                if result.returncode != 0:
                    failures += 1
                    summary.append(
                        {
                            "slug": entry["slug"],
                            "title": entry["title"],
                            "group": entry["group"],
                            "model": str(output),
                            "status": "failed",
                            "returncode": result.returncode,
                        }
                    )
                    continue
                metrics = json.loads(metrics_path.read_text())
                metrics.update(
                    {
                        "slug": entry["slug"],
                        "title": entry["title"],
                        "group": entry["group"],
                        "status": "ok",
                    }
                )
                summary.append(metrics)
            except Exception as err:  # noqa: BLE001 - report every catalog item.
                failures += 1
                status = "failed" if args.require_all else "skipped"
                if status == "failed":
                    print(f"failed {entry['slug']}: {err}", file=sys.stderr)
                else:
                    print(f"skipping {entry['slug']}: {err}")
                summary.append(
                    {
                        "slug": entry["slug"],
                        "title": entry["title"],
                        "group": entry["group"],
                        "model": str(output),
                        "status": status,
                        "error": str(err),
                    }
                )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    passed = sum(1 for item in summary if item["status"] == "ok")
    skipped = sum(1 for item in summary if item["status"] == "skipped")
    print(
        json.dumps(
            {
                "catalog_entries": len(summary),
                "passed": passed,
                "skipped": skipped,
                "failed": failures if args.require_all else sum(1 for item in summary if item["status"] == "failed"),
                "summary_output": str(args.summary_output),
            },
            indent=2,
        )
    )
    if any(item["status"] == "failed" for item in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
