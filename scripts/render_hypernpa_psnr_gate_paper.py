#!/usr/bin/env python3
"""Render paper figures from HyperNPA PSNR-gate artifacts.

The generated panel intentionally uses the exact same materialized direct,
HyperNPA, and oracle BPKs referenced by a PSNR-gate report. It is meant for
paper visualization of high-particle validation, not for training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repository_binary() -> list[str]:
    binary = ROOT / "target" / "release" / "burn_automata"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "--release", "-p", "burn_automata", "--features", "cli", "--"]


def infer_trace(
    model: Path,
    output: Path,
    particles: int,
    steps: int,
    update_prob: float,
    seed: int,
    seed_scale: float,
    seed_mode: str,
    reuse: bool,
) -> dict[str, Any]:
    if not reuse or not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *repository_binary(),
            "infer",
            "--model",
            str(model),
            "--preset",
            "growing-2d",
            "--particles",
            str(particles),
            "--steps",
            str(steps),
            "--update-prob",
            str(update_prob),
            "--seed",
            str(seed),
            "--seed-scale",
            str(seed_scale),
            "--seed-mode",
            seed_mode,
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    return load_json(output)


def condition_panel(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def rollout_panel(trace: dict[str, Any], size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    positions = trace["positions"]
    states = trace["states"]
    state_dims = int(trace["state_dims"])
    radius = max(1, size // 96)
    for index, position in enumerate(positions):
        x = int(round((float(position[0]) + 1.0) * 0.5 * (size - 1)))
        y = int(round((1.0 - (float(position[1]) + 1.0) * 0.5) * (size - 1)))
        state = states[index * state_dims : (index + 1) * state_dims]
        tail = state[-3:] if len(state) >= 3 else [0.0, 0.0, 0.0]
        color = tuple(int(max(0.0, min(1.0, float(value) + 0.5)) * 255) for value in tail)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 170),
        )
    return image.convert("RGB")


def label_panel(image: Image.Image, label: str, height: int = 28) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + height), "white")
    out.paste(image, (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((4, 6), label, fill=(0, 0, 0))
    return out


def concat_h(images: list[Image.Image], gap: int = 8) -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def concat_v(images: list[Image.Image], gap: int = 10) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        out.paste(image, (0, y))
        y += image.height + gap
    return out


def entries_by_slug(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in report["entries"]:
        out.setdefault(entry["slug"], {})[entry["kind"]] = entry
    return out


def seed_mode_arg(value: str) -> str:
    normalized = value.strip()
    aliases = {
        "UniformCircle": "uniform-circle",
        "uniformcircle": "uniform-circle",
        "Uniform": "uniform",
        "Gaussian": "gaussian",
    }
    return aliases.get(normalized, normalized.lower().replace("_", "-"))


def render_report(
    psnr_report: Path,
    wgpu_psnr_report: Path | None,
    output_dir: Path,
    panel_size: int,
    reuse: bool,
    keep_traces: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    report = load_json(psnr_report)
    by_slug = entries_by_slug(report)
    rows: list[Image.Image] = []
    samples: list[dict[str, Any]] = []
    particles = int(report["particle_count"])
    steps = int(report["rollout_steps"][0])
    update_prob = float(report["update_prob"])
    seed = int(report["seed"])
    seed_scale = float(report["seed_scale"])
    seed_mode = seed_mode_arg(str(report["seed_mode"]))

    ordered = sorted(
        by_slug.items(),
        key=lambda item: item[1]["hyper"]["render_rgb_psnr_db"],
    )
    for row_index, (slug, kinds) in enumerate(ordered, start=1):
        direct = kinds["direct"]
        hyper = kinds["hyper"]
        condition = resolve_path(hyper["condition"])
        target_model = resolve_path(hyper["target_model"])
        direct_model = resolve_path(direct["model"])
        hyper_model = resolve_path(hyper["model"])

        panels = [
            label_panel(condition_panel(condition, panel_size), f"{row_index:02d} condition"),
        ]
        for label, model, psnr in [
            ("HyperNPA", hyper_model, hyper["render_rgb_psnr_db"]),
            ("Direct LoRA", direct_model, direct["render_rgb_psnr_db"]),
            ("Oracle", target_model, None),
        ]:
            trace = infer_trace(
                model,
                trace_dir / f"{slug}_{label.lower().replace(' ', '_')}_p{particles}_s{steps}.json",
                particles,
                steps,
                update_prob,
                seed,
                seed_scale,
                seed_mode,
                reuse,
            )
            psnr_label = "target" if psnr is None else f"{psnr:.1f} dB"
            panels.append(label_panel(rollout_panel(trace, panel_size), f"{label} {psnr_label}"))
        rows.append(concat_h(panels))
        samples.append(
            {
                "slug": slug,
                "split": hyper["split"],
                "condition": hyper["condition"],
                "hyper_render_rgb_psnr_db": hyper["render_rgb_psnr_db"],
                "direct_render_rgb_psnr_db": direct["render_rgb_psnr_db"],
                "target_model": hyper["target_model"],
            }
        )

    all_panel = concat_v(rows)
    all_panel_path = output_dir / "exact_dino_flow_16sample_panel.png"
    all_panel.save(all_panel_path)

    half = (len(rows) + 1) // 2
    panel_paths = []
    for name, chunk in [("a", rows[:half]), ("b", rows[half:])]:
        path = output_dir / f"exact_dino_flow_16sample_panel_{name}.png"
        concat_v(chunk).save(path)
        panel_paths.append(path)

    summary = {
        "psnr_report": str(psnr_report),
        "wgpu_psnr_report": str(wgpu_psnr_report) if wgpu_psnr_report else None,
        "particles": particles,
        "steps": steps,
        "update_prob": update_prob,
        "seed": seed,
        "seed_scale": seed_scale,
        "seed_mode": seed_mode,
        "sample_count": len(samples),
        "summaries": report["summaries"],
        "wgpu_summaries": load_json(wgpu_psnr_report)["summaries"] if wgpu_psnr_report else None,
        "samples": samples,
        "panel": str(all_panel_path.relative_to(ROOT)),
        "split_panels": [str(path.relative_to(ROOT)) for path in panel_paths],
    }
    summary_path = output_dir / "exact_dino_flow_16sample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if not keep_traces and trace_dir.exists():
        shutil.rmtree(trace_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--psnr-report",
        type=Path,
        default=ROOT
        / "artifacts/hyper2d_adapter_bank_exact_oracle_10k8x8_dino_token_grid_flow_linear_solve_overfit_train_all/psnr_gate_report.json",
    )
    parser.add_argument(
        "--wgpu-psnr-report",
        type=Path,
        default=ROOT
        / "artifacts/hyper2d_adapter_bank_exact_oracle_10k8x8_dino_token_grid_flow_overfit_train_all/psnr_gate_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/hyper_npa_figures/exact_dino_flow_psnr",
    )
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--keep-traces", action="store_true")
    args = parser.parse_args()
    summary = render_report(
        args.psnr_report,
        args.wgpu_psnr_report if args.wgpu_psnr_report.exists() else None,
        args.output_dir,
        args.panel_size,
        not args.no_reuse,
        args.keep_traces,
    )
    print(
        f"wrote {summary['panel']} samples={summary['sample_count']} "
        f"particles={summary['particles']} steps={summary['steps']}"
    )


if __name__ == "__main__":
    main()
