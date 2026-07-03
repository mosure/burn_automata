#!/usr/bin/env python3
"""Train shared 2D NPA weights plus per-sample LoRA adapters on CUDA.

This is the GPU backend for burn_automata train-hyper2d-direct-basis. It uses
the upstream NPA-wave CUDA/sphops perception, rollout, and target-image loss,
but optimizes a shared update MLP together with one low-rank adapter per image.
The output is JSON containing Rust-native base weights and adapter vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--payload-output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--report-interval", type=int, default=16)
    parser.add_argument("--example-batch-size", type=int, default=8)
    parser.add_argument("--rollout-particles", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--update-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-scale", type=float, default=0.2)
    parser.add_argument("--seed-mode", default="uniform_circle")
    parser.add_argument("--inject-seed-interval", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--adapter-alpha", type=float, default=16.0)
    parser.add_argument("--target-points", type=int, default=4096)
    parser.add_argument("--target-image-size", type=int)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--splat-sigma", type=float, default=1.0)
    parser.add_argument("--splat-loss-weight", type=float, default=2.0)
    parser.add_argument("--color-loss-weight", type=float, default=5.0)
    parser.add_argument("--density-loss-weight", type=float, default=1.0)
    parser.add_argument("--displacement-regularizer-weight", type=float, default=0.01)
    parser.add_argument("--overflow-regularizer-weight", type=float, default=100.0)
    parser.add_argument("--bound-regularizer-weight", type=float, default=100.0)
    parser.add_argument("--normalize-grads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--base-weight-decay", type=float, default=0.0)
    parser.add_argument("--base-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--adapter-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--adapter-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapter-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--adapter-l2", type=float, default=0.0)
    parser.add_argument("--holdout-adapter-steps", type=int, default=0)
    parser.add_argument("--holdout-adapter-batch-size", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=42)
    return parser.parse_args()


def resolve_upstream_root(value: Path | None) -> Path:
    candidates: list[Path] = []
    if value is not None:
        candidates.append(value)
    env_root = os.environ.get("NPA_WAVE_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            ROOT / "third_party" / "NPA-wave",
            ROOT / "external" / "NPA-wave",
            Path("/tmp/NPA-wave"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "train.py").is_file() and (candidate / "sphops").is_dir():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "Could not find upstream NPA-wave checkout with train.py and sphops. "
        f"Checked: {joined}. Set --upstream-root or NPA_WAVE_ROOT."
    )


def require_cuda(device_name: str) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"PyTorch with CUDA is required; import torch failed: {exc}") from exc
    if not device_name.startswith("cuda"):
        raise SystemExit(f"GPU training requires a CUDA device, got {device_name!r}")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch reports no CUDA device; refusing to train on CPU")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return torch


def import_upstream(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        from losses import Loss
        from models.npa import NPA, Pool, step_euler
        from sphops import HashGrid
        from train import fix_seed, get_target
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Failed to import upstream CUDA training modules. "
            "Install the NPA-wave requirements, including sphops. "
            f"Original error: {exc}"
        ) from exc
    return {
        "HashGrid": HashGrid,
        "Loss": Loss,
        "NPA": NPA,
        "Pool": Pool,
        "step_euler": step_euler,
        "fix_seed": fix_seed,
        "get_target": get_target,
    }


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out != out or out in {float("inf"), float("-inf")}:
        return 0.0
    return out


def tensor_to_list(value: Any) -> list[float]:
    value = value.detach().cpu().contiguous().view(-1)
    return [float(x) for x in value.tolist()]


def grad_norm(params: list[Any], torch: Any) -> float:
    terms = []
    for param in params:
        if param.grad is not None:
            terms.append(param.grad.detach().pow(2).sum())
    if not terms:
        return 0.0
    return finite_float(torch.sqrt(torch.stack(terms).sum()).item())


def prepare_grads(params: list[Any], torch: Any, clip_norm: float, normalize: bool) -> float:
    norm = grad_norm(params, torch)
    scale = 1.0
    if normalize:
        for param in params:
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm()
            if float(param_norm.item()) > 0.0:
                param.grad.data.div_(param_norm + 1.0e-8)
        norm = grad_norm(params, torch)
    if clip_norm > 0.0 and norm > clip_norm:
        scale = clip_norm / max(norm, 1.0e-12)
    for param in params:
        if param.grad is None:
            continue
        param.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if scale != 1.0:
            param.grad.mul_(scale)
    return scale


def summarize_loss(
    loss_value: float,
    loss_info: dict[str, Any],
    image_size: int,
    particle_count: int,
    target_points: int,
) -> dict[str, Any]:
    splat = loss_info.get("splat_loss", {})
    terms = splat.get("terms", {}) if isinstance(splat, dict) else {}
    color = terms.get("color_loss", [0.0, 0.0])
    density = terms.get("density_loss", [0.0, 0.0])
    displacement = loss_info.get("displacement_regularizer", {}).get("value", 0.0)
    overflow = loss_info.get("overflow_regularizer", {}).get("value", 0.0)
    bound = loss_info.get("bound_regularizer", {}).get("value", 0.0)
    return {
        "total_loss": finite_float(loss_value),
        "splat_loss": finite_float(splat.get("value", 0.0)) if isinstance(splat, dict) else 0.0,
        "color_loss": finite_float(color[1] if isinstance(color, (list, tuple)) else color),
        "density_loss": finite_float(density[1] if isinstance(density, (list, tuple)) else density),
        "displacement_regularizer": finite_float(displacement),
        "overflow_regularizer": finite_float(overflow),
        "bound_regularizer": finite_float(bound),
        "target_points": int(target_points),
        "particle_count": int(particle_count),
        "batch_size": 1,
        "image_size": int(image_size),
    }


class LowRankAdapter:
    def __init__(self, torch: Any, device: Any, input_dims: int, hidden_dims: int, output_dims: int, rank: int, alpha: float, seed: int):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        self.rank = rank
        self.alpha = alpha
        self.w1_down = torch.nn.Parameter((torch.rand(rank, input_dims, device=device, generator=gen) * 0.02) - 0.01)
        self.w1_up = torch.nn.Parameter((torch.rand(hidden_dims, rank, device=device, generator=gen) * 0.02) - 0.01)
        self.w2_down = torch.nn.Parameter((torch.rand(rank, hidden_dims, device=device, generator=gen) * 0.02) - 0.01)
        self.w2_up = torch.nn.Parameter((torch.rand(output_dims, rank, device=device, generator=gen) * 0.02) - 0.01)
        self.b1_delta = torch.nn.Parameter(torch.zeros(hidden_dims, device=device))
        self.b2_delta = torch.nn.Parameter(torch.zeros(output_dims, device=device))

    def params(self) -> list[Any]:
        return [
            self.w1_down,
            self.w1_up,
            self.w2_down,
            self.w2_up,
            self.b1_delta,
            self.b2_delta,
        ]

    def vector(self) -> list[float]:
        values = []
        for param in self.params():
            values.extend(tensor_to_list(param))
        return values

    def l2(self, torch: Any) -> Any:
        terms = [param.pow(2).mean() for param in self.params()]
        return torch.stack(terms).mean()


class AdaptedNPA:
    def __init__(self, torch: Any, base: Any, adapter: LowRankAdapter):
        self.torch = torch
        self.base = base
        self.adapter = adapter
        self.spatial_dims = base.spatial_dims
        self.state_dims = base.state_dims
        self.alpha = base.alpha
        self.last_snapshot = None

    def __call__(self, x: Any, s: Any, grid: Any) -> tuple[Any, Any, Any, Any]:
        z, x_bin, s_bin, snapshot = self.base.perceive(x, s, grid)
        self.last_snapshot = snapshot
        scale = self.adapter.alpha / max(self.adapter.rank, 1)
        w1 = self.base.model[0].weight + scale * (self.adapter.w1_up @ self.adapter.w1_down)
        b1_base = self.base.model[0].bias
        if b1_base is None:
            b1_base = self.torch.zeros_like(self.adapter.b1_delta)
        b1 = b1_base + self.adapter.b1_delta
        w2 = self.base.model[2].weight + scale * (self.adapter.w2_up @ self.adapter.w2_down)
        b2_base = self.base.model[2].bias
        if b2_base is None:
            b2_base = self.torch.zeros_like(self.adapter.b2_delta)
        b2 = b2_base + self.adapter.b2_delta
        hidden = self.torch.nn.functional.relu(self.torch.nn.functional.linear(z, w1, b1))
        y = self.torch.nn.functional.linear(hidden, w2, b2)
        dx_bin = y[..., : self.spatial_dims]
        eps = grid.eps
        dx_bin = self.alpha * dx_bin * eps / (1.0 + dx_bin.norm(dim=-1, keepdim=True))
        ds_bin = y[..., self.spatial_dims :]
        return dx_bin, ds_bin, x_bin, s_bin

    def decode(self, s: Any) -> Any:
        return self.base.decode(s)


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources", payload)
    if not isinstance(sources, list) or not sources:
        raise SystemExit("sources JSON must contain a non-empty sources array")
    for source in sources:
        source["path"] = str(Path(source["path"]).expanduser().resolve())
    return sources


def load_examples(args: argparse.Namespace, torch: Any, modules: dict[str, Any], device: Any, base: Any, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_image_size = None
    if args.target_image_size is not None:
        target_image_size = [args.target_image_size, args.target_image_size]
    examples = []
    input_dims = int(base.model[0].weight.shape[1])
    hidden_dims = int(base.model[0].weight.shape[0])
    output_dims = int(base.model[2].weight.shape[0])
    for idx, source in enumerate(sources):
        target = modules["get_target"](
            device,
            {
                "path": source["path"],
                "aabb": [-1.0, 1.0, -1.0, 1.0],
                "image_size": target_image_size,
                "threshold": args.target_threshold,
                "num_points": args.target_points,
            },
        )
        loss_fn = modules["Loss"](
            target,
            splat_loss_weight=args.splat_loss_weight,
            displacement_regularizer_weight=args.displacement_regularizer_weight,
            overflow_regularizer_weight=args.overflow_regularizer_weight,
            bound_regularizer_weight=args.bound_regularizer_weight,
            splat_loss_kwargs={
                "density_loss_weight": args.density_loss_weight,
                "color_loss_weight": args.color_loss_weight,
                "lpips_loss_weight": 0.0,
                "grid_size": args.image_size,
                "sigma": args.splat_sigma,
                "lo": -1.0,
                "hi": 1.0,
                "center": True,
            },
        )
        adapter = LowRankAdapter(
            torch,
            device,
            input_dims,
            hidden_dims,
            output_dims,
            args.adapter_rank,
            args.adapter_alpha,
            args.seed + idx * 0x517CC1B7,
        )
        pool = modules["Pool"](
            device=device,
            pool_size=1,
            state_dims=base.state_dims,
            spatial_dims=2,
            num_particles=args.rollout_particles,
            sigma=args.seed_scale,
            noise_level=0.0,
            seed_mode=args.seed_mode,
            random_seed=args.seed + idx,
        )
        examples.append(
            {
                **source,
                "target": target,
                "loss_fn": loss_fn,
                "adapter": adapter,
                "pool": pool,
                "target_points": int(target["positions"].shape[0]),
                "last_train_loss": None,
                "optimizer": torch.optim.SGD(
                    adapter.params(),
                    lr=args.adapter_learning_rate,
                    weight_decay=args.adapter_weight_decay,
                ),
            }
        )
    return examples


def run_example_loss(args: argparse.Namespace, torch: Any, modules: dict[str, Any], base: Any, grid: Any, example: dict[str, Any], step: int, train: bool) -> tuple[Any, dict[str, Any]]:
    model = AdaptedNPA(torch, base, example["adapter"])
    pool = example["pool"]
    replace = train and (step % max(args.inject_seed_interval, 1) == 0)
    if train:
        x, s, idx = pool.sample(1, replace_seed=replace)
    else:
        eval_pool = modules["Pool"](
            device=x_device(base),
            pool_size=1,
            state_dims=base.state_dims,
            spatial_dims=2,
            num_particles=args.rollout_particles,
            sigma=args.seed_scale,
            noise_level=0.0,
            seed_mode=args.seed_mode,
            random_seed=args.eval_seed + int(example["index"]),
        )
        x, s, idx = eval_pool.sample(1, replace_seed=True)
    sum_dx = 0.0
    for _ in range(args.rollout_steps):
        x, s, stats = modules["step_euler"](
            model,
            x,
            s,
            grid,
            update_prob=args.update_prob,
            intermediate=True,
            noise_std=0.0,
        )
        sum_dx = sum_dx + stats["dx_norm"]
    x = grid.wrap_positions(x)
    out = model.decode(s)
    loss, info = example["loss_fn"](
        {"positions": x, "states": s, "outputs": out, "sum_dx": sum_dx},
        return_info=True,
        return_summary=False,
    )
    if train:
        with torch.no_grad():
            pool.update(x.detach(), s.detach(), idx)
    return loss, info


def x_device(base: Any) -> Any:
    return next(base.parameters()).device


def select_indices(examples: list[dict[str, Any]], count: int, rng: random.Random) -> list[int]:
    if not examples:
        return []
    count = len(examples) if count == 0 else min(count, len(examples))
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return indices[:count]


def evaluate(args: argparse.Namespace, torch: Any, modules: dict[str, Any], base: Any, grid: Any, examples: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    if not examples:
        return None
    rng = random.Random(seed)
    indices = select_indices(examples, args.eval_examples, rng)
    totals = []
    splats = []
    colors = []
    densities = []
    with torch.no_grad():
        for idx in indices:
            example = examples[idx]
            loss, info = run_example_loss(args, torch, modules, base, grid, example, seed + idx, False)
            summary = summarize_loss(
                loss.item(),
                info,
                args.image_size,
                args.rollout_particles,
                example["target_points"],
            )
            totals.append(summary["total_loss"])
            splats.append(summary["splat_loss"])
            colors.append(summary["color_loss"])
            densities.append(summary["density_loss"])
    return {
        "examples": len(indices),
        "mean_total_loss": statistics.mean(totals),
        "max_total_loss": max(totals),
        "mean_splat_loss": statistics.mean(splats),
        "mean_color_loss": statistics.mean(colors),
        "mean_density_loss": statistics.mean(densities),
    }


def train_phase(args: argparse.Namespace, torch: Any, modules: dict[str, Any], base: Any, grid: Any, examples: list[dict[str, Any]], steps: int, batch_size: int, update_base: bool, seed: int) -> tuple[list[dict[str, Any]], float | None, int]:
    if not examples or steps == 0:
        return [], None, 0
    rng = random.Random(seed)
    base_optimizer = torch.optim.SGD(
        base.parameters(),
        lr=args.base_learning_rate,
        weight_decay=args.base_weight_decay,
    )
    history = []
    best_loss = None
    best_step = 0
    throughputs = []
    for step in range(1, steps + 1):
        start = time.perf_counter()
        indices = select_indices(examples, batch_size, rng)
        if update_base:
            base_optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        adapter_norms = []
        for idx in indices:
            example = examples[idx]
            example["optimizer"].zero_grad(set_to_none=True)
            loss, info = run_example_loss(args, torch, modules, base, grid, example, step, True)
            if args.adapter_l2 > 0.0:
                loss = loss + args.adapter_l2 * example["adapter"].l2(torch)
            scaled_loss = loss / max(len(indices), 1)
            scaled_loss.backward()
            adapter_norm = grad_norm(example["adapter"].params(), torch)
            adapter_scale = prepare_grads(
                example["adapter"].params(),
                torch,
                args.adapter_grad_clip_norm,
                args.normalize_grads,
            )
            _ = adapter_scale
            example["optimizer"].step()
            loss_sum += finite_float(loss.detach().item())
            adapter_norms.append(adapter_norm)
            example["last_train_loss"] = finite_float(loss.detach().item())
        base_norm = grad_norm(list(base.parameters()), torch) if update_base else 0.0
        base_scale = 1.0
        if update_base:
            base_scale = prepare_grads(
                list(base.parameters()),
                torch,
                args.base_grad_clip_norm,
                args.normalize_grads,
            )
            base_optimizer.step()
        torch.cuda.synchronize(x_device(base))
        elapsed = time.perf_counter() - start
        particle_steps = len(indices) * args.rollout_particles * args.rollout_steps
        pps = particle_steps / max(elapsed, 1.0e-12)
        throughputs.append(pps)
        should_report = step == steps or (step % max(args.report_interval, 1)) == 0
        if should_report:
            eval_loss = evaluate(args, torch, modules, base, grid, examples, args.eval_seed + step)
            if eval_loss is not None and (best_loss is None or eval_loss["mean_total_loss"] < best_loss):
                best_loss = eval_loss["mean_total_loss"]
                best_step = step
            history.append(
                {
                    "step": step,
                    "loss": loss_sum / max(len(indices), 1),
                    "eval_loss": eval_loss,
                    "base_grad_norm": base_norm,
                    "base_grad_scale": base_scale,
                    "mean_adapter_grad_norm": statistics.mean(adapter_norms) if adapter_norms else 0.0,
                    "max_adapter_grad_norm": max(adapter_norms) if adapter_norms else 0.0,
                    "examples_seen": len(indices),
                    "particle_steps_per_sec": pps,
                    "elapsed_ms": elapsed * 1000.0,
                }
            )
            print(
                "gpu-hyper2d-direct-basis "
                f"step={step} loss={history[-1]['loss']:.6f} pps={pps:.3f}",
                flush=True,
            )
    return history, best_loss, best_step


def export_base(base: Any) -> dict[str, list[float]]:
    b1 = base.model[0].bias
    if b1 is None:
        b1 = base.model[0].weight.new_zeros(base.model[0].weight.shape[0])
    b2 = base.model[2].bias
    if b2 is None:
        b2 = base.model[2].weight.new_zeros(base.model[2].weight.shape[0])
    return {
        "w1": tensor_to_list(base.model[0].weight),
        "b1": tensor_to_list(b1),
        "w2": tensor_to_list(base.model[2].weight),
        "b2": tensor_to_list(b2),
    }


def export_adapters(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for example in examples:
        out.append(
            {
                "slug": example["slug"],
                "split": example["split"],
                "title": example.get("title"),
                "group": example.get("group"),
                "condition": example["path"],
                "target_source_width": int(example.get("source_width", 0)),
                "target_source_height": int(example.get("source_height", 0)),
                "target_points": int(example["target_points"]),
                "last_train_loss": example.get("last_train_loss"),
                "adapter": example["adapter"].vector(),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    args.sources_json = args.sources_json.expanduser().resolve()
    args.payload_output = args.payload_output.expanduser().resolve()
    upstream_root = resolve_upstream_root(args.upstream_root)
    torch = require_cuda(args.device)
    modules = import_upstream(upstream_root)
    modules["fix_seed"](args.base_seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    sources = load_sources(args.sources_json)
    for idx, source in enumerate(sources):
        source["index"] = idx

    base = modules["NPA"](
        spatial_dims=2,
        state_dims=16,
        hidden_dims=128,
        eps0=0.1,
        alpha=0.5,
    ).to(device)
    grid = modules["HashGrid"](
        num_particles=args.rollout_particles,
        batch_size=1,
        dim=2,
        boundary="clamped",
        mode="grid",
        grid_size=[16, 16],
        eps=0.1,
        max_particles_per_block=32,
    )
    examples = load_examples(args, torch, modules, device, base, sources)
    train_examples = [example for example in examples if example["split"] == "train"]
    holdout_examples = [example for example in examples if example["split"] == "holdout"]
    if not train_examples:
        raise SystemExit("direct basis GPU training requires at least one train example")

    total_start = time.perf_counter()
    initial_train = evaluate(args, torch, modules, base, grid, train_examples, args.eval_seed)
    initial_holdout = evaluate(args, torch, modules, base, grid, holdout_examples, args.eval_seed ^ 0x901D2D)
    history, best_loss, best_step = train_phase(
        args,
        torch,
        modules,
        base,
        grid,
        train_examples,
        args.steps,
        args.example_batch_size,
        True,
        args.seed,
    )
    holdout_history, _, _ = train_phase(
        args,
        torch,
        modules,
        base,
        grid,
        holdout_examples,
        args.holdout_adapter_steps,
        args.holdout_adapter_batch_size,
        False,
        args.seed ^ 0x901D2D,
    )
    final_train = evaluate(args, torch, modules, base, grid, train_examples, args.eval_seed)
    final_holdout = evaluate(args, torch, modules, base, grid, holdout_examples, args.eval_seed ^ 0x901D2D)
    torch.cuda.synchronize(device)

    payload = {
        "backend": "upstream_torch_cuda_sphops_direct_basis",
        "upstream_root": str(upstream_root),
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_elapsed_ms": (time.perf_counter() - total_start) * 1000.0,
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "initial_holdout_loss": initial_holdout,
        "final_holdout_loss": final_holdout,
        "best_train_loss": best_loss,
        "best_train_step": best_step,
        "history": history,
        "holdout_history": holdout_history,
        "base": export_base(base),
        "adapters": export_adapters(train_examples + holdout_examples),
    }
    args.payload_output.parent.mkdir(parents=True, exist_ok=True)
    args.payload_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.payload_output}", flush=True)


if __name__ == "__main__":
    main()
