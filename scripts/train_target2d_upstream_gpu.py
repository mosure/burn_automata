#!/usr/bin/env python3
"""Train a 2D NPA target model with the upstream CUDA/sphops pipeline.

The script is intentionally GPU-only. It exits before training if PyTorch CUDA
or the upstream sphops extension is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--step-min", type=int, default=32)
    parser.add_argument("--step-max", type=int, default=96)
    parser.add_argument("--inject-seed-interval", type=int, default=16)
    parser.add_argument("--update-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-scale", type=float, default=0.2)
    parser.add_argument("--seed-mode", default="uniform_circle")
    parser.add_argument("--brush-size", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--scheduler-milestones", default="2000,4000,6000,8000")
    parser.add_argument("--scheduler-gamma", type=float, default=0.3)
    parser.add_argument("--normalize-grads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-points", type=int, default=4096)
    parser.add_argument("--target-image-size", type=int)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--splat-sigma", type=float, default=1.0)
    parser.add_argument("--splat-loss-weight", type=float, default=2.0)
    parser.add_argument("--color-loss-weight", type=float, default=5.0)
    parser.add_argument("--density-loss-weight", type=float, default=1.0)
    parser.add_argument("--displacement-regularizer-weight", type=float, default=0.01)
    parser.add_argument("--overflow-regularizer-weight", type=float, default=100.0)
    parser.add_argument("--bound-regularizer-weight", type=float, default=100.0)
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
        raise SystemExit(
            "PyTorch with CUDA is required for target2d GPU training; "
            f"import torch failed: {exc}"
        ) from exc

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
            "Install the NPA-wave requirements, including its sphops extension. "
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


def parse_milestones(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part) for part in value.split(",") if part.strip()]


def sample_rollout_steps(step_min: int, step_max: int, torch: Any, device: Any) -> int:
    if step_min == step_max:
        return step_min
    return int(torch.randint(step_min, step_max, (1,), device=device).item())


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out != out or out in {float("inf"), float("-inf")}:
        return 0.0
    return out


def summarize_loss(
    loss_value: float,
    loss_info: dict[str, Any],
    args: argparse.Namespace,
    target_points: int,
) -> dict[str, Any]:
    splat = loss_info.get("splat_loss", {})
    terms = splat.get("terms", {}) if isinstance(splat, dict) else {}
    color = terms.get("color_loss", [args.color_loss_weight, 0.0])
    density = terms.get("density_loss", [args.density_loss_weight, 0.0])
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
        "particle_count": int(args.particles),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
    }


def grad_norm(model: Any, torch: Any) -> float:
    terms = []
    for param in model.parameters():
        if param.grad is not None:
            terms.append(param.grad.detach().pow(2).sum())
    if not terms:
        return 0.0
    return finite_float(torch.sqrt(torch.stack(terms).sum()).item())


def evaluate_seed_loss(
    model: Any,
    grid: Any,
    loss_fn: Any,
    target: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    Pool = modules["Pool"]
    step_euler = modules["step_euler"]
    pool = Pool(
        device=torch.device(args.device),
        pool_size=1,
        state_dims=model.state_dims,
        spatial_dims=2,
        num_particles=args.particles,
        sigma=args.seed_scale,
        noise_level=0.0,
        seed_mode=args.seed_mode,
        random_seed=args.seed,
    )
    x, s, _ = pool.sample(1, replace_seed=True)
    sum_dx = 0.0
    with torch.enable_grad():
        for timestep in range(args.step_max):
            x, s, stats = step_euler(
                model,
                x,
                s,
                grid,
                update_prob=args.update_prob,
                intermediate=True,
                noise_std=0.0,
            )
            sum_dx = sum_dx + stats["dx_norm"]
            if timestep % 128 == 0:
                x = grid.wrap_positions(x)
        x = grid.wrap_positions(x)
        out = model.decode(s)
        loss, info = loss_fn(
            {"positions": x, "states": s, "outputs": out, "sum_dx": sum_dx},
            return_info=True,
            return_summary=False,
        )
    return summarize_loss(loss.item(), info, args, int(target["positions"].shape[0]))


def main() -> None:
    args = parse_args()
    args.target_image = args.target_image.expanduser().resolve()
    args.checkpoint_output = args.checkpoint_output.expanduser().resolve()
    args.metrics_output = args.metrics_output.expanduser().resolve()
    if args.upstream_root is not None:
        args.upstream_root = args.upstream_root.expanduser().resolve()
    if args.step_max < args.step_min:
        raise SystemExit("--step-max must be >= --step-min")
    if args.batch_size > args.pool_size:
        raise SystemExit("--batch-size must be <= --pool-size")

    upstream_root = resolve_upstream_root(args.upstream_root)
    torch = require_cuda(args.device)
    modules = import_upstream(upstream_root)
    modules["fix_seed"](args.seed)

    device = torch.device(args.device)
    target_image_size = None
    if args.target_image_size is not None:
        target_image_size = [args.target_image_size, args.target_image_size]
    target = modules["get_target"](
        device,
        {
            "path": str(args.target_image),
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
    model = modules["NPA"](
        spatial_dims=2,
        state_dims=16,
        hidden_dims=128,
        eps0=0.1,
        alpha=0.5,
    ).to(device)
    grid = modules["HashGrid"](
        num_particles=args.particles,
        batch_size=args.batch_size,
        dim=2,
        boundary="clamped",
        mode="grid",
        grid_size=[16, 16],
        eps=0.1,
        max_particles_per_block=32,
    )
    grid_eval = modules["HashGrid"](
        num_particles=args.particles,
        batch_size=1,
        dim=2,
        boundary="clamped",
        mode="grid",
        grid_size=[16, 16],
        eps=0.1,
        max_particles_per_block=32,
    )
    initial_eval_loss = evaluate_seed_loss(
        model, grid_eval, loss_fn, target, args, torch, modules
    )
    history = []
    throughputs = []
    best_loss = None
    best_epoch = 0
    final_loss = initial_eval_loss
    total_start = time.perf_counter()
    window_start = time.perf_counter()
    window_particle_steps = 0
    total_updates = (args.epochs + 1) * args.repetitions
    update_index = 0

    for repetition in range(args.repetitions):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=parse_milestones(args.scheduler_milestones),
            gamma=args.scheduler_gamma,
        )
        pool = modules["Pool"](
            device=device,
            pool_size=args.pool_size,
            state_dims=model.state_dims,
            spatial_dims=2,
            num_particles=args.particles,
            sigma=args.seed_scale,
            noise_level=0.0,
            seed_mode=args.seed_mode,
            random_seed=args.seed,
        )
        for local_epoch in range(args.epochs + 1):
            epoch = repetition * args.epochs + local_epoch
            replace = (epoch % max(args.inject_seed_interval, 1)) == 0
            with torch.no_grad():
                x, s, idx = pool.sample(args.batch_size, replace_seed=replace)
                if args.brush_size > 0.0:
                    random_point = torch.randint(args.particles, (args.batch_size,), device=device)
                    center = x[torch.arange(args.batch_size, device=device), random_point]
                    distances = torch.norm(x - center.unsqueeze(1), dim=-1)
                    mask = distances < args.brush_size
                    s = s.masked_fill(mask.unsqueeze(-1), 0.0)

            steps = sample_rollout_steps(args.step_min, args.step_max, torch, device)
            sum_dx = 0.0
            for _ in range(steps):
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
            loss, loss_info = loss_fn(
                {"positions": x, "states": s, "outputs": out, "sum_dx": sum_dx},
                return_info=True,
                return_summary=False,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise SystemExit("Loss became NaN/Inf during CUDA training")

            loss.backward()
            pre_normalized_grad_norm = grad_norm(model, torch)
            if args.normalize_grads:
                model.normalize_grads()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.no_grad():
                pool.update(x, s, idx)

            update_index += 1
            particle_steps = args.batch_size * args.particles * steps
            window_particle_steps += particle_steps
            should_report = (
                update_index == total_updates
                or epoch % max(args.report_interval, 1) == 0
            )
            if should_report:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - window_start
                throughput = window_particle_steps / max(elapsed, 1.0e-12)
                throughputs.append(throughput)
                final_loss = summarize_loss(
                    loss.item(), loss_info, args, int(target["positions"].shape[0])
                )
                if best_loss is None or final_loss["total_loss"] < best_loss["total_loss"]:
                    best_loss = dict(final_loss)
                    best_epoch = epoch
                history.append(
                    {
                        "epoch": epoch,
                        "repetition": repetition,
                        "rollout_steps": steps,
                        "loss": final_loss,
                        "grad_norm": pre_normalized_grad_norm,
                        "grad_scale": 1.0,
                        "elapsed_ms": elapsed * 1000.0,
                        "particle_steps_per_sec": throughput,
                    }
                )
                print(
                    "gpu-target2d "
                    f"epoch={epoch} loss={final_loss['total_loss']:.6f} "
                    f"pps={throughput:.3f}",
                    flush=True,
                )
                window_start = time.perf_counter()
                window_particle_steps = 0

    torch.cuda.synchronize(device)
    final_eval_loss = evaluate_seed_loss(model, grid_eval, loss_fn, target, args, torch, modules)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_output)
    metrics = {
        "backend": "upstream_torch_cuda_sphops",
        "upstream_root": str(upstream_root),
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint_output": str(args.checkpoint_output),
        "target_image": str(args.target_image),
        "target_points": int(target["positions"].shape[0]),
        "epochs_completed": total_updates,
        "repetitions_completed": args.repetitions,
        "total_elapsed_ms": (time.perf_counter() - total_start) * 1000.0,
        "median_particle_steps_per_sec": statistics.median(throughputs) if throughputs else 0.0,
        "initial_eval_loss": initial_eval_loss,
        "final_loss": final_loss,
        "best_loss": best_loss or final_loss,
        "best_epoch": best_epoch,
        "final_eval_loss": final_eval_loss,
        "history": history,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "wrote "
        f"{args.checkpoint_output} {args.metrics_output} "
        f"final_eval={final_eval_loss['total_loss']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
