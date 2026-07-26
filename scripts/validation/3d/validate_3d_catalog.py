#!/usr/bin/env python3
"""Regenerate and assert app-scale 3D catalog validation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from catalog3d_validation.checks import assert_case
from catalog3d_validation.config import CASES
from catalog3d_validation.runner import load_report, run_validation
from catalog3d_validation.summary import build_report_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that shipped 3D BPKs match the latest local-growth pipeline "
            "and the Bevy catalog exposure policy."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("target"))
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("target/validate_3d_catalog_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        reports.append(build_report_summary(case, report, output_path, status))
        failures.extend(f"{case.name}: {failure}" for failure in case_failures)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"cases": reports, "failed": len(failures)}, indent=2))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
