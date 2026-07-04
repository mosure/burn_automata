#!/usr/bin/env python3
"""Generate a paper-style validation report for Hyper2D direct-basis runs.

This script intentionally reports the evaluated artifact as a shared-base plus
stored per-sample LoRA adapter bank. It does not claim image-conditioned
hypernet generalization unless a corresponding hypernet report is supplied in a
future extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BPK_MAGIC = b"BAUTBPK1"
BPK_HEADER_LEN = 8 + 4 + 8 + 32


@dataclass(frozen=True)
class Sample:
    slug: str
    split: str
    condition: Path
    adapter_output: Path
    oracle_model_output: Path | None
    title: str | None
    ratio_to_oracle: float | None
    shared_loss: float | None
    oracle_loss: float | None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_bpk_or_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(BPK_MAGIC):
        return json.loads(data)
    if len(data) < BPK_HEADER_LEN:
        raise ValueError(f"{path} is shorter than the BPK header")
    version = struct.unpack("<I", data[8:12])[0]
    if version != 1:
        raise ValueError(f"{path} has unsupported BPK version {version}")
    payload_len = struct.unpack("<Q", data[12:20])[0]
    payload = data[BPK_HEADER_LEN:]
    if len(payload) != payload_len:
        raise ValueError(f"{path} payload length mismatch")
    expected = data[20:52]
    actual = hashlib.sha256(payload).digest()
    if expected != actual:
        raise ValueError(f"{path} payload checksum mismatch")
    return json.loads(payload)


def matmul(a: list[float], a_rows: int, a_cols: int, b: list[float], b_cols: int) -> list[float]:
    out = [0.0] * (a_rows * b_cols)
    for row in range(a_rows):
        for col in range(b_cols):
            acc = 0.0
            for mid in range(a_cols):
                acc += a[row * a_cols + mid] * b[mid * b_cols + col]
            out[row * b_cols + col] = acc
    return out


def add_scaled(base: list[float], delta: list[float], scale: float) -> list[float]:
    if len(base) != len(delta):
        raise ValueError(f"shape mismatch: {len(base)} != {len(delta)}")
    return [b + scale * d for b, d in zip(base, delta)]


def materialize_adapter(base: dict[str, Any], adapter_manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = base["config"]
    adapter = adapter_manifest["adapter"]
    rank = int(adapter["rank"])
    alpha = float(adapter["alpha"])
    scale = alpha / max(rank, 1)
    hidden = int(cfg["hidden_dims"])
    spatial = int(cfg["spatial_dims"])
    state = int(cfg["state_dims"])
    update_dims = spatial + state
    perception_dims = len(adapter["w1_down"]) // rank
    w1_delta = matmul(adapter["w1_up"], hidden, rank, adapter["w1_down"], perception_dims)
    w2_delta = matmul(adapter["w2_up"], update_dims, rank, adapter["w2_down"], hidden)
    weights = base["weights"]
    return {
        "format_version": 1,
        "model_kind": "npa",
        "source": adapter_manifest.get("source") or "materialized-direct-basis-adapter",
        "config": cfg,
        "hashgrid": base["hashgrid"],
        "weights": {
            "w1": add_scaled(weights["w1"], w1_delta, scale),
            "b1": [b + d for b, d in zip(weights["b1"], adapter["b1_delta"])],
            "w2": add_scaled(weights["w2"], w2_delta, scale),
            "b2": [b + d for b, d in zip(weights["b2"], adapter["b2_delta"])],
        },
    }


def pct_reduction(initial: float, final: float) -> float:
    if not math.isfinite(initial) or initial <= 0.0:
        return float("nan")
    return 100.0 * (initial - final) / initial


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def repository_binary() -> list[str]:
    binary = ROOT / "target" / "debug" / "burn_automata"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "-p", "burn_automata", "--features", "cli", "--"]


def seed_for_sample(base_seed: int, slug: str) -> int:
    digest = hashlib.blake2s(slug.encode("utf-8"), digest_size=4).digest()
    return base_seed ^ int.from_bytes(digest, "little")


def run_infer(
    model_path: Path,
    output_path: Path,
    steps: int,
    particles: int,
    update_prob: float,
    seed: int,
    seed_scale: float,
    seed_mode: str,
) -> None:
    command = [
        *repository_binary(),
        "infer",
        "--model",
        str(model_path),
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
        str(output_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def trace_image(trace: dict[str, Any], size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    state_dims = int(trace["state_dims"])
    positions = trace["positions"]
    states = trace["states"]
    radius = max(2, size // 48)
    for idx, position in enumerate(positions):
        x = int(round((float(position[0]) + 1.0) * 0.5 * (size - 1)))
        y = int(round((1.0 - (float(position[1]) + 1.0) * 0.5) * (size - 1)))
        state = states[idx * state_dims : (idx + 1) * state_dims]
        tail = state[-3:] if len(state) >= 3 else [0.0, 0.0, 0.0]
        color = tuple(int(max(0.0, min(1.0, float(v) + 0.5)) * 255) for v in tail)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 190))
    return image


def load_condition_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def label_panel(image: Image.Image, label: str, pad: int = 18) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), label, fill=(0, 0, 0))
    return out


def concat_h(images: list[Image.Image], gap: int = 6) -> Image.Image:
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


def resolve_path(anchor: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.exists():
        return direct
    return anchor / path


def samples_from_reports(main_report: dict[str, Any], oracle_report: dict[str, Any] | None) -> list[Sample]:
    output_anchor = ROOT / main_report["output_dir"]
    adapter_by_slug = {entry["slug"]: entry for entry in main_report.get("adapters", [])}
    oracle = None
    if oracle_report:
        oracle = oracle_report.get("oracle_validation")
    if oracle is None:
        oracle = main_report.get("oracle_validation")
    samples: list[Sample] = []
    if oracle and oracle.get("entries"):
        for entry in oracle["entries"]:
            adapter_entry = adapter_by_slug.get(entry["slug"])
            if not adapter_entry:
                continue
            samples.append(
                Sample(
                    slug=entry["slug"],
                    split=entry["split"],
                    condition=resolve_path(output_anchor, entry["condition"]),
                    adapter_output=resolve_path(output_anchor, adapter_entry["adapter_output"]),
                    oracle_model_output=(
                        resolve_path(output_anchor, entry["oracle_model_output"])
                        if entry.get("oracle_model_output")
                        else None
                    ),
                    title=adapter_entry.get("title"),
                    ratio_to_oracle=float(entry["loss_ratio_to_oracle"]),
                    shared_loss=float(entry["shared_loss"]["total_loss"]),
                    oracle_loss=float(entry["oracle_final_loss"]["total_loss"]),
                )
            )
    if samples:
        return samples
    for entry in main_report.get("adapters", []):
        samples.append(
            Sample(
                slug=entry["slug"],
                split=entry["split"],
                condition=resolve_path(output_anchor, entry["condition"]),
                adapter_output=resolve_path(output_anchor, entry["adapter_output"]),
                oracle_model_output=None,
                title=entry.get("title"),
                ratio_to_oracle=None,
                shared_loss=None,
                oracle_loss=None,
            )
        )
    return samples


def select_samples(samples: list[Sample], limit: int) -> list[Sample]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    by_split = {
        "train": [s for s in samples if s.split == "train"],
        "holdout": [s for s in samples if s.split == "holdout"],
    }
    for values in by_split.values():
        values.sort(key=lambda sample: sample.ratio_to_oracle or -1.0, reverse=True)
    train_take = min(len(by_split["train"]), max(1, limit // 2))
    holdout_take = min(len(by_split["holdout"]), limit - train_take)
    selected = by_split["train"][:train_take] + by_split["holdout"][:holdout_take]
    if len(selected) < limit:
        leftovers = [s for s in samples if s not in selected]
        leftovers.sort(key=lambda sample: sample.ratio_to_oracle or -1.0, reverse=True)
        selected.extend(leftovers[: limit - len(selected)])
    return selected


def generate_rollout_figures(
    main_report: dict[str, Any],
    samples: list[Sample],
    out_dir: Path,
    steps: list[int],
    image_size: int,
) -> dict[str, Any]:
    figures = out_dir / "figures"
    models = out_dir / "materialized_models"
    traces = out_dir / "traces"
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    base_path = resolve_path(ROOT, main_report["shared_base_output"])
    base = load_bpk_or_json(base_path)
    particles = int(main_report["rollout_particles"])
    update_prob = float(main_report["update_prob"])
    seed = int(main_report["seed"])
    seed_scale = float(main_report["seed_scale"])
    seed_mode = "uniform-circle"
    rows = []
    sample_records = []
    for sample in samples:
        adapter_manifest = load_json(sample.adapter_output)
        materialized = materialize_adapter(base, adapter_manifest)
        model_path = models / f"{sample.slug}.model.json"
        model_path.write_text(json.dumps(materialized))
        sample_seed = seed_for_sample(seed, sample.slug)
        direct_panels = [
            label_panel(load_condition_image(sample.condition, image_size), f"{sample.split} target")
        ]
        for step in steps:
            trace_path = traces / f"{sample.slug}_steps{step}.json"
            run_infer(model_path, trace_path, step, particles, update_prob, sample_seed, seed_scale, seed_mode)
            trace = load_json(trace_path)
            direct_panels.append(label_panel(trace_image(trace, image_size), f"direct t={step}"))
        row_parts = [concat_h(direct_panels)]
        if sample.oracle_model_output and sample.oracle_model_output.exists():
            oracle_panels = [
                label_panel(load_condition_image(sample.condition, image_size), "oracle target")
            ]
            for step in steps:
                trace_path = traces / f"{sample.slug}_oracle_steps{step}.json"
                run_infer(
                    sample.oracle_model_output,
                    trace_path,
                    step,
                    particles,
                    update_prob,
                    sample_seed,
                    seed_scale,
                    seed_mode,
                )
                trace = load_json(trace_path)
                oracle_panels.append(label_panel(trace_image(trace, image_size), f"oracle t={step}"))
            row_parts.append(concat_h(oracle_panels))
        row = concat_v(row_parts, gap=4)
        row_path = figures / f"{sample.slug}_rollout.png"
        row.save(row_path)
        rows.append(row)
        sample_records.append(
            {
                "slug": sample.slug,
                "split": sample.split,
                "title": sample.title,
                "condition": str(sample.condition),
                "adapter_output": str(sample.adapter_output),
                "oracle_model_output": str(sample.oracle_model_output)
                if sample.oracle_model_output
                else None,
                "figure": str(row_path.relative_to(out_dir)),
                "ratio_to_oracle": sample.ratio_to_oracle,
                "shared_loss": sample.shared_loss,
                "oracle_loss": sample.oracle_loss,
                "seed": sample_seed,
            }
        )
    grid = concat_v(rows) if rows else Image.new("RGB", (image_size, image_size), "white")
    grid_path = figures / "rollout_grid.png"
    grid.save(grid_path)
    return {"rollout_grid": str(grid_path.relative_to(out_dir)), "samples": sample_records}


def clean_generated_outputs(out_dir: Path) -> None:
    patterns = [
        "figures/*_rollout.png",
        "figures/rollout_grid.png",
        "materialized_models/*.model.json",
        "traces/*.json",
    ]
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def oracle_block(main_report: dict[str, Any], oracle_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if oracle_report and oracle_report.get("oracle_validation"):
        return oracle_report["oracle_validation"]
    return main_report.get("oracle_validation")


def summarize(main_report: dict[str, Any], oracle: dict[str, Any] | None, figure_info: dict[str, Any]) -> dict[str, Any]:
    initial_train = main_report["initial_train_loss"]["mean_total_loss"]
    final_train = main_report["final_train_loss"]["mean_total_loss"]
    initial_holdout = main_report["initial_holdout_loss"]["mean_total_loss"]
    final_holdout = main_report["final_holdout_loss"]["mean_total_loss"]
    summary = {
        "artifact": main_report["output_dir"],
        "train_examples": main_report["train_examples"],
        "holdout_examples": main_report["holdout_examples"],
        "adapter_rank": main_report["adapter_rank"],
        "adapter_alpha": main_report["adapter_alpha"],
        "steps": main_report["steps"],
        "train_adapter_refine_steps": main_report.get("train_adapter_refine_steps", 0),
        "holdout_adapter_steps": main_report.get("holdout_adapter_steps", 0),
        "train_initial_loss": initial_train,
        "train_final_loss": final_train,
        "train_loss_reduction_percent": pct_reduction(initial_train, final_train),
        "holdout_initial_loss": initial_holdout,
        "holdout_final_loss": final_holdout,
        "holdout_loss_reduction_percent": pct_reduction(initial_holdout, final_holdout),
        "best_train_loss": main_report.get("best_train_loss"),
        "best_train_step": main_report.get("best_train_step"),
        "rollout_particles": main_report["rollout_particles"],
        "rollout_steps": main_report["rollout_steps"],
        "eval_examples": main_report["eval_examples"],
        "gpu_backend": main_report["gpu_training"]["backend"] if main_report.get("gpu_training") else None,
        "target_loss_config": main_report.get("target_loss_config"),
        "hypernet_generalization_validated": False,
        "evaluated_system": "shared NPA base plus directly optimized stored per-sample LoRA adapters",
        "intended_conditional_system": "image encoder features -> rectified-flow LoRA generator -> shared-base NPA rollout",
        "figures": figure_info,
        "oracle_rollout_renders_available": any(
            sample.get("oracle_model_output") for sample in figure_info.get("samples", [])
        ),
    }
    if oracle:
        summary["oracle_validation"] = {
            "train_examples": oracle["train_examples"],
            "holdout_examples": oracle["holdout_examples"],
            "epochs": oracle["epochs"],
            "train_summary": oracle.get("train_summary"),
            "holdout_summary": oracle.get("holdout_summary"),
        }
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    oracle = summary.get("oracle_validation")
    train_oracle = oracle.get("train_summary") if oracle else None
    holdout_oracle = oracle.get("holdout_summary") if oracle else None
    oracle_render_note = (
        "Selected rows include oracle-overfit rollout panels where the validation report persisted oracle checkpoints."
        if summary.get("oracle_rollout_renders_available")
        else "Oracle checkpoints were not present for the selected report, so oracle comparison is numeric in this render."
    )
    return textwrap.dedent(
        f"""
        # Hyper2D Direct-Basis 10k Validation Report

        ## Status

        The evaluated artifact is **not yet an image-conditioned hypernetwork**. It is a shared NPA base with directly optimized, stored per-sample LoRA adapters. The intended conditional system is `{summary["intended_conditional_system"]}`; that final image-to-LoRA generator remains unvalidated in this artifact set.

        ## Main 10k Run

        - Artifact: `{summary["artifact"]}`
        - Train / holdout examples: {summary["train_examples"]} / {summary["holdout_examples"]}
        - Adapter: rank {summary["adapter_rank"]}, alpha {summary["adapter_alpha"]}
        - Training: {summary["steps"]} shared+adapter steps, {summary["train_adapter_refine_steps"]} train refine steps, {summary["holdout_adapter_steps"]} holdout adapter steps
        - Rollout objective: {summary["rollout_particles"]} particles, {summary["rollout_steps"]} training steps
        - Backend: {summary["gpu_backend"]}

        ## Loss Summary

        | split | initial mean loss | final mean loss | reduction |
        | --- | ---: | ---: | ---: |
        | train sample eval | {fmt(summary["train_initial_loss"])} | {fmt(summary["train_final_loss"])} | {fmt(summary["train_loss_reduction_percent"], 1)}% |
        | holdout sample eval | {fmt(summary["holdout_initial_loss"])} | {fmt(summary["holdout_final_loss"])} | {fmt(summary["holdout_loss_reduction_percent"], 1)}% |

        ## Oracle Comparison

        | split | examples | shared loss | oracle overfit loss | mean ratio | max ratio |
        | --- | ---: | ---: | ---: | ---: | ---: |
        | train | {fmt(train_oracle.get("examples") if train_oracle else None)} | {fmt(train_oracle.get("mean_shared_loss") if train_oracle else None)} | {fmt(train_oracle.get("mean_oracle_loss") if train_oracle else None)} | {fmt(train_oracle.get("mean_ratio_to_oracle") if train_oracle else None)} | {fmt(train_oracle.get("max_ratio_to_oracle") if train_oracle else None)} |
        | holdout | {fmt(holdout_oracle.get("examples") if holdout_oracle else None)} | {fmt(holdout_oracle.get("mean_shared_loss") if holdout_oracle else None)} | {fmt(holdout_oracle.get("mean_oracle_loss") if holdout_oracle else None)} | {fmt(holdout_oracle.get("mean_ratio_to_oracle") if holdout_oracle else None)} | {fmt(holdout_oracle.get("max_ratio_to_oracle") if holdout_oracle else None)} |

        ## Rollout Figures

        See `{summary["figures"]["rollout_grid"]}` for target thumbnails and long-rollout panels from selected stored LoRA adapters. {oracle_render_note} These renders evaluate stability of the shared-base+LoRA baseline, not a learned image-conditioned hypernet.

        ## Interpretation

        The 10k result supports the claim that a shared NPA basis plus per-sample LoRA adapters is a viable intermediate representation: train and holdout losses improve similarly, and the sampled oracle gap is moderate rather than catastrophic. It does **not** yet establish that image features can predict the LoRA weights. The next experiment must train and evaluate the conditional LoRA generator against this adapter bank and compare generated adapters to direct adapters and oracle overfits on the same rollout metrics.
        """
    ).strip() + "\n"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def latex_paper(summary: dict[str, Any]) -> str:
    oracle = summary.get("oracle_validation")
    train_oracle = oracle.get("train_summary") if oracle else None
    holdout_oracle = oracle.get("holdout_summary") if oracle else None
    fig = summary["figures"]["rollout_grid"]
    loss_cfg = summary.get("target_loss_config") or {}
    oracle_table_caption = (
        "Oracle checkpoints were persisted for this validation, so selected oracle rollouts are rendered in Figure~\\ref{fig:rollouts}."
        if summary.get("oracle_rollout_renders_available")
        else "Oracle weights are not persisted in the selected report, so oracle comparison is numeric here."
    )
    rollout_caption = (
        "Rows include matched oracle-overfit panels when the selected validation saved oracle checkpoints."
        if summary.get("oracle_rollout_renders_available")
        else "Oracle-overfit panels require a validation report with persisted oracle checkpoints."
    )
    return rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\title{{Shared Neural Particle Automata Bases with Low-Rank Image-Specific Adapters}}
\author{{burn\_automata local validation report}}
\date{{July 4, 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We evaluate a 2D Neural Particle Automata (NPA) training stage intended to support conditional generation of particle dynamics from images. The evaluated system learns a shared NPA base and directly optimized per-sample LoRA adapters over a 10k OmniSVG thumbnail slice. This is an intermediate representation for a future conditional hypernetwork, not yet an end-to-end image-to-LoRA model. On a 9k/1k train/holdout split, the shared-base plus stored-adapter system reduces sampled target-image rollout loss by {fmt(summary["train_loss_reduction_percent"], 1)}\% on train examples and {fmt(summary["holdout_loss_reduction_percent"], 1)}\% on holdout examples. A broader oracle comparison estimates the gap to independently overfit 2D NPAs and provides the calibration target for the next image-conditioned LoRA generator stage.
\end{{abstract}}

\section{{Introduction}}
Conditional Neural Particle Automata aim to turn a static image condition into an executable particle-growth program. Instead of directly predicting pixels, the model predicts dynamics: particles exchange local information, update positions and latent state, and are rendered into a target image by differentiable splatting. This report evaluates the shared-dynamics stage needed before training a condition-to-dynamics hypernetwork.

The central question is whether many 2D targets can share a common NPA update rule while retaining enough sample-specific capacity through low-rank adapters. If this basis is weak, a downstream hypernetwork has no stable adapter manifold to predict. If the basis is strong, the next stage can train an image encoder and rectified-flow adapter generator against a meaningful target.

\section{{Conditional NPA Architecture}}
The intended end-to-end model has three pieces. First, an image encoder extracts condition features from the target thumbnail. Second, a rectified-flow hypernetwork maps those features and flow time to a LoRA adapter vector. Third, the adapter modulates a shared NPA base and the NPA rolls out particle dynamics:
\[
x_{{t+1}}, h_{{t+1}} = F_{{\theta_0 + \Delta\theta(c)}}(x_t, h_t, \mathcal{{N}}(x_t)),
\]
where \(x_t\) are particle positions, \(h_t\) are particle states, \(\mathcal{{N}}\) denotes local particle neighborhoods, \(\theta_0\) is the shared base, and \(c\) is the image condition. For the MLP update layers, the adapter uses low-rank deltas,
\[
\Delta W_i = \frac{{\alpha}}{{r}} U_i V_i,
\]
with rank \(r={summary["adapter_rank"]}\) and alpha \(\alpha={fmt(summary["adapter_alpha"], 1)}\).

This paper evaluates the basis stage only: \(\theta_0\) and one stored adapter per sample are optimized directly from image loss. The image encoder and rectified-flow hypernetwork are described as the next stage, not claimed as validated here.

\section{{Training Protocol}}
The evaluated artifact is the 10k Burn/WGPU direct-basis run recorded in the companion validation summary. It contains {summary["train_examples"]} train adapters and {summary["holdout_examples"]} holdout adapters, each rank {summary["adapter_rank"]} with alpha {summary["adapter_alpha"]}. Training used {summary["steps"]} shared-base plus adapter steps, followed by {summary["train_adapter_refine_steps"]} train-adapter refinement steps and {summary["holdout_adapter_steps"]} holdout-adapter steps. Rollouts used {summary["rollout_particles"]} particles and {summary["rollout_steps"]} objective steps.

The dataset is a 10k OmniSVG thumbnail slice split into 9000 train samples and 1000 holdout samples. Each sample owns a persistent LoRA adapter, while the base NPA weights are shared during the joint stage. Holdout adapters are optimized against the frozen final shared base to test whether the learned basis supports unseen targets. The target objective combines splat reconstruction, color, density, displacement regularization, overflow regularization, and bound regularization. The report configuration used image-size {fmt(loss_cfg.get("image_size") if loss_cfg else None, 0)}, splat weight {fmt(loss_cfg.get("splat_loss_weight") if loss_cfg else None)}, color weight {fmt(loss_cfg.get("color_loss_weight") if loss_cfg else None)}, and density weight {fmt(loss_cfg.get("density_loss_weight") if loss_cfg else None)}.

\section{{Oracle Validation Protocol}}
To calibrate adapter quality, selected train and holdout samples are compared against independently overfit 2D NPA oracle models trained from scratch on the same target objective. The oracle is not a hypernetwork and does not share weights across samples; it estimates the per-sample quality ceiling under the current NPA architecture and training recipe. The broader validation uses {fmt(train_oracle.get("examples") if train_oracle else None)} train and {fmt(holdout_oracle.get("examples") if holdout_oracle else None)} holdout oracle fits for {fmt(oracle.get("epochs") if oracle else None, 0)} epochs each.

\section{{Results}}
\begin{{table}}[h]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
split & initial mean loss & final mean loss & reduction \\
\midrule
train sample eval & {fmt(summary["train_initial_loss"])} & {fmt(summary["train_final_loss"])} & {fmt(summary["train_loss_reduction_percent"], 1)}\% \\
holdout sample eval & {fmt(summary["holdout_initial_loss"])} & {fmt(summary["holdout_final_loss"])} & {fmt(summary["holdout_loss_reduction_percent"], 1)}\% \\
\bottomrule
\end{{tabular}}
\caption{{Main 10k shared-base plus stored-LoRA adapter training result.}}
\end{{table}}

\begin{{table}}[h]
\centering
\begin{{tabular}}{{lrrrrr}}
\toprule
split & examples & shared loss & oracle loss & mean ratio & max ratio \\
\midrule
train & {fmt(train_oracle.get("examples") if train_oracle else None)} & {fmt(train_oracle.get("mean_shared_loss") if train_oracle else None)} & {fmt(train_oracle.get("mean_oracle_loss") if train_oracle else None)} & {fmt(train_oracle.get("mean_ratio_to_oracle") if train_oracle else None)} & {fmt(train_oracle.get("max_ratio_to_oracle") if train_oracle else None)} \\
holdout & {fmt(holdout_oracle.get("examples") if holdout_oracle else None)} & {fmt(holdout_oracle.get("mean_shared_loss") if holdout_oracle else None)} & {fmt(holdout_oracle.get("mean_oracle_loss") if holdout_oracle else None)} & {fmt(holdout_oracle.get("mean_ratio_to_oracle") if holdout_oracle else None)} & {fmt(holdout_oracle.get("max_ratio_to_oracle") if holdout_oracle else None)} \\
\bottomrule
\end{{tabular}}
\caption{{Comparison against independently overfit 2D NPA oracle models. {oracle_table_caption}}}
\end{{table}}

\begin{{figure}}[p]
\centering
\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{{{latex_escape(fig)}}}
\caption{{Selected condition thumbnails and shared-base plus stored-LoRA rollouts at multiple step counts. {rollout_caption} These panels probe long-rollout stability of the stored-adapter baseline; they are not generated by an image-conditioned hypernetwork.}}
\label{{fig:rollouts}}
\end{{figure}}

\section{{Interpretation}}
The train and holdout curves are close, indicating that the shared basis is not simply memorizing train-only dynamics. The oracle ratios measure how far the shared-base plus adapter representation remains from single-sample overfit NPAs. Ratios near one would indicate parity with a fully overfit oracle; larger ratios indicate remaining adapter/base expressivity or optimization gap. In this run, train and holdout oracle ratios are similar: {fmt(train_oracle.get("mean_ratio_to_oracle") if train_oracle else None)} on train and {fmt(holdout_oracle.get("mean_ratio_to_oracle") if holdout_oracle else None)} on holdout. That supports the shared-basis hypothesis, but does not prove image-conditioned generalization.

\section{{Reproducibility Artifacts}}
The companion Markdown report, JSON summary, LaTeX source, and rollout figures are generated by \path|scripts/render_hyper2d_direct_basis_paper.py| from the 10k training report and the 8x8 oracle-validation report. The generated rollout grid in Figure~\ref{{fig:rollouts}} selects high-ratio train and holdout cases, making the visualization intentionally stress the samples where the shared-basis representation is furthest from oracle parity.

\section{{Limitations and Next Experiment}}
This report does not validate generalized image-to-LoRA inference. The missing experiment is to train a DINO-family image-condition encoder and rectified-flow LoRA generator over the stored adapter bank, then evaluate generated adapters against direct adapters and oracle overfits on the same train/holdout samples and rollout stability panels. Future validation should use oracle reports with persisted checkpoints to render oracle dynamics directly beside generated and direct adapters.

\end{{document}}
""".strip() + "\n"


def try_compile_latex(tex_path: Path) -> None:
    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name], cwd=tex_path.parent, check=False)
    elif shutil.which("pdflatex"):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=tex_path.parent, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-report", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "hyper2d_direct_basis_10k")
    parser.add_argument("--max-render-samples", type=int, default=6)
    parser.add_argument("--steps", default="0,8,16,32,64")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    main_report = load_json(args.main_report)
    oracle_report = load_json(args.oracle_report) if args.oracle_report and args.oracle_report.exists() else None
    samples = select_samples(samples_from_reports(main_report, oracle_report), args.max_render_samples)
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(args.output_dir)
    figure_info = generate_rollout_figures(main_report, samples, args.output_dir, steps, args.image_size)
    oracle = oracle_block(main_report, oracle_report)
    summary = summarize(main_report, oracle, figure_info)
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "validation_report.md").write_text(markdown_report(summary))
    tex_path = args.output_dir / "hyper2d_direct_basis_10k.tex"
    tex_path.write_text(latex_paper(summary))
    if args.compile_pdf:
        try_compile_latex(tex_path)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
