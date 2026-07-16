#!/usr/bin/env python3
"""Export a small official SelfOrg-NPA reference fixture.

This is intentionally a reference/export bridge, not a local trainer. It reads
the upstream checkout, reproduces the released target extraction formula, and
records checkpoint tensor metadata/hashes so Rust-side parity checks can verify
that they are comparing against the official baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import yaml


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_upstream_loader(upstream_root: pathlib.Path) -> Any:
    sys.path.insert(0, str(upstream_root))
    from utils import loader  # type: ignore

    return loader


def grange_np(shape: tuple[int, int], aabb: list[float]) -> np.ndarray:
    gx, gy = shape
    mins = np.asarray(aabb[0::2], dtype=np.float32)
    maxs = np.asarray(aabb[1::2], dtype=np.float32)
    sizes = maxs - mins
    xs = np.arange(gx, dtype=np.float32)
    ys = np.arange(gy, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)
    denom = np.asarray([gx, gy], dtype=np.float32)
    return mins + sizes * (grid + 0.5) / denom


def adaptive_image_size(loader: Any, path: str, threshold: float, target_points: int) -> int:
    size = 128
    for _ in range(5):
        target_image = loader.load_target_image(path, size=[size, size])
        nonzero = int(np.sum(target_image[..., 3] >= threshold))
        nonzero = max(nonzero, 1)
        size = int(round(math.sqrt(target_points / nonzero) * size))
    return max(size, 1)


def extract_target(
    upstream_root: pathlib.Path,
    cfg_target: dict[str, Any],
    target_image: pathlib.Path | None,
) -> dict[str, Any]:
    loader = load_upstream_loader(upstream_root)
    path = str(target_image) if target_image is not None else str(cfg_target["path"])
    threshold = float(cfg_target.get("threshold", 0.05))
    image_size = cfg_target.get("image_size")
    if image_size is None:
        size = adaptive_image_size(
            loader,
            path,
            threshold,
            int(cfg_target.get("num_points", 4096)),
        )
        image_size = [size, size]
    source_image = loader.load_target_image(path, size=image_size)
    target_image_rgba = source_image.transpose(1, 0, 2)[:, ::-1, :].copy()
    positions = grange_np(target_image_rgba.shape[:2], cfg_target["aabb"])
    alpha = target_image_rgba[..., 3]
    mask = alpha > threshold
    target_positions = positions[mask].astype(np.float32)
    target_colors = target_image_rgba[mask][..., :3].astype(np.float32)
    pixel_size = (cfg_target["aabb"][1] - cfg_target["aabb"][0]) / image_size[0]
    return {
        "path": path,
        "threshold": threshold,
        "image_size": list(map(int, image_size)),
        "pixel_size": float(pixel_size),
        "point_count": int(target_positions.shape[0]),
        "positions_sha256": sha256_array(target_positions),
        "colors_sha256": sha256_array(target_colors),
        "rgba_sha256": sha256_array(target_image_rgba.astype(np.float32)),
        "positions": target_positions.tolist(),
        "colors": target_colors.tolist(),
    }


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    array = tensor.numpy()
    finite = np.isfinite(array)
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": sha256_array(array),
        "finite": bool(finite.all()),
    }
    if array.size:
        summary["mean"] = float(array.mean())
        summary["std"] = float(array.std())
        summary["min"] = float(array.min())
        summary["max"] = float(array.max())
    return summary


def checkpoint_summary(path: pathlib.Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not load as a state-dict-like mapping")
    tensors = {
        name: tensor_summary(value)
        for name, value in sorted(state.items())
        if isinstance(value, torch.Tensor)
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def tensor_values(value: torch.Tensor) -> list[float]:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1).tolist()


def deterministic_values(count: int, scale: float, phase: float = 0.0) -> torch.Tensor:
    index = torch.arange(count, dtype=torch.float32)
    return scale * torch.sin(index * 0.017 + phase)


def training_step_fixture(
    upstream_root: pathlib.Path,
    config: dict[str, Any],
    target: dict[str, Any],
    device_name: str,
) -> dict[str, Any]:
    """Export one deterministic official forward/backward/AdamW update.

    This fixture deliberately supplies all floating-point inputs and parameters;
    it does not ask Rust and PyTorch to reproduce each other's RNG streams. The
    particle batch is first binned by the official hash grid, and that binned
    order is the canonical input order recorded for the parity replay.
    """

    sys.path.insert(0, str(upstream_root))
    from losses import Loss  # type: ignore
    from models.npa import NPA  # type: ignore
    from sphops import HashGrid  # type: ignore

    device = torch.device(device_name)
    npa_cfg = config["npa"]["kwargs"]
    state_dims = int(npa_cfg["state_dims"])
    hidden_dims = int(npa_cfg["hidden_dims"])
    spatial_dims = int(config["hashgrid"]["dim"])
    particle_count = 24
    batch_size = 1

    model = NPA(spatial_dims=spatial_dims, **npa_cfg).to(device)
    with torch.no_grad():
        w1 = deterministic_values(model.model[0].weight.numel(), 0.006, 0.1).reshape_as(
            model.model[0].weight
        )
        b1 = deterministic_values(model.model[0].bias.numel(), 0.02, 0.7)
        w2 = deterministic_values(model.model[2].weight.numel(), 0.006, 1.3).reshape_as(
            model.model[2].weight
        )
        model.model[0].weight.copy_(w1.to(device))
        model.model[0].bias.copy_(b1.to(device))
        model.model[2].weight.copy_(w2.to(device))

    index = torch.arange(particle_count, dtype=torch.float32)
    angle = index * (2.0 * math.pi / particle_count) + 0.07
    radius = 0.035 + 0.145 * ((index.remainder(6.0) + 1.0) / 6.0)
    positions = torch.stack(
        [radius * torch.cos(angle) - 0.08, radius * torch.sin(angle) + 0.04], dim=-1
    ).reshape(batch_size, particle_count, spatial_dims)
    state_index = torch.arange(particle_count * state_dims, dtype=torch.float32)
    states = (0.08 * torch.sin(state_index * 0.071 + 0.3)).reshape(
        batch_size, particle_count, state_dims
    )
    positions = positions.to(device)
    states = states.to(device)

    grid = HashGrid(
        num_particles=particle_count,
        batch_size=batch_size,
        **config["hashgrid"],
    )
    features, positions_binned, states_binned, snapshot = model.perceive(
        positions, states, grid
    )
    raw_update = model.model(features)
    raw_motion = raw_update[..., :spatial_dims]
    dx = (
        model.alpha
        * raw_motion
        * grid.eps
        / (1.0 + raw_motion.norm(dim=-1, keepdim=True))
    )
    ds = raw_update[..., spatial_dims:]
    update_mask = (
        (torch.arange(particle_count, device=device).remainder(3) != 0)
        .to(torch.float32)
        .reshape(batch_size, particle_count, 1)
    )
    next_positions = positions_binned + dx * update_mask
    next_states = states_binned + ds * update_mask
    mean_dx_norm = dx.norm(dim=-1).mean()

    target_tensors = {
        "positions": torch.tensor(target["positions"], dtype=torch.float32, device=device),
        "colors": torch.tensor(target["colors"], dtype=torch.float32, device=device),
        "pixel_size": float(target["pixel_size"]),
    }
    loss_fn = Loss(target_tensors, **config["train"]["loss"])
    optimizer_cfg = dict(config["train"]["optimizer"]["kwargs"])
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
    loss, loss_info = loss_fn(
        {
            "positions": next_positions,
            "states": next_states,
            "outputs": next_states,
            "sum_dx": mean_dx_norm,
        },
        return_info=True,
        return_summary=False,
    )

    initial_model = {
        "w1": tensor_values(model.model[0].weight),
        "b1": tensor_values(model.model[0].bias),
        "w2": tensor_values(model.model[2].weight),
    }
    forward = {
        "features": tensor_values(features),
        "raw_update": tensor_values(raw_update),
        "dx": tensor_values(dx),
        "ds": tensor_values(ds),
        "next_positions": tensor_values(next_positions),
        "next_states": tensor_values(next_states),
        "mean_dx_norm": float(mean_dx_norm.detach().cpu()),
    }
    loss.backward()
    raw_gradients = {
        "w1": tensor_values(model.model[0].weight.grad),
        "b1": tensor_values(model.model[0].bias.grad),
        "w2": tensor_values(model.model[2].weight.grad),
    }
    model.normalize_grads()
    normalized_gradients = {
        "w1": tensor_values(model.model[0].weight.grad),
        "b1": tensor_values(model.model[0].bias.grad),
        "w2": tensor_values(model.model[2].weight.grad),
    }
    optimizer.step()
    updated_model = {
        "w1": tensor_values(model.model[0].weight),
        "b1": tensor_values(model.model[0].bias),
        "w2": tensor_values(model.model[2].weight),
    }

    loss_components = {
        name: float(info["value"])
        for name, info in loss_info.items()
        if info is not None and "value" in info
    }
    loss_terms = {
        f"{name}.{term}": float(value)
        for name, info in loss_info.items()
        if info is not None
        for term, (_weight, value) in info.get("terms", {}).items()
    }
    permutation = tensor_values(snapshot.permutation.to(torch.float32))

    return {
        "device": str(device),
        "batch_size": batch_size,
        "particle_count": particle_count,
        "spatial_dims": spatial_dims,
        "state_dims": state_dims,
        "hidden_dims": hidden_dims,
        "perception_dims": int(model.perception_dim),
        "update_dims": spatial_dims + state_dims,
        "architecture": "linear_bias_relu_linear_no_output_bias",
        "canonical_input_order": "official_hashgrid_binned",
        "positions": tensor_values(positions_binned),
        "states": tensor_values(states_binned),
        "update_mask": tensor_values(update_mask),
        "permutation": permutation,
        "model": initial_model,
        "forward": forward,
        "loss": {
            "total": float(loss.detach().cpu()),
            "components": loss_components,
            "terms": loss_terms,
        },
        "raw_gradients": raw_gradients,
        "normalized_gradients": normalized_gradients,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(optimizer_cfg["lr"]),
            "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
            "beta1": float(optimizer.defaults["betas"][0]),
            "beta2": float(optimizer.defaults["betas"][1]),
            "epsilon": float(optimizer.defaults["eps"]),
        },
        "updated_model": updated_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=pathlib.Path, default=".cache/selforg_npa/NPA")
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--target-image", type=pathlib.Path)
    parser.add_argument("--training-step", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    config_path = args.config or upstream_root / "configs/growing.yaml"
    checkpoint_path = args.checkpoint or upstream_root / "data/pretrained/lizard.pth"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    target = extract_target(upstream_root, config["train"]["target"], args.target_image)
    report = {
        "upstream": {
            "root": str(upstream_root),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "commit": None,
        },
        "hashgrid": config["hashgrid"],
        "npa": config["npa"]["kwargs"],
        "pool": config["pool"],
        "train": {
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "step_range": config["train"]["step_range"],
            "inject_seed_interval": config["train"]["inject_seed_interval"],
            "num_repetitions": config["train"]["num_repetitions"],
            "update_prob": config["train"].get("update_prob", 0.5),
            "optimizer": config["train"]["optimizer"],
            "scheduler": config["train"].get("scheduler"),
            "loss": config["train"]["loss"],
        },
        "target": target,
        "checkpoint": checkpoint_summary(checkpoint_path),
    }
    if args.training_step:
        report["training_step"] = training_step_fixture(
            upstream_root, config, target, args.device
        )

    git_head = upstream_root / ".git" / "HEAD"
    if git_head.exists():
        head = git_head.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = upstream_root / ".git" / head.split(" ", 1)[1]
            report["upstream"]["commit"] = ref.read_text(encoding="utf-8").strip()
        else:
            report["upstream"]["commit"] = head

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output} target_points={report['target']['point_count']}")


if __name__ == "__main__":
    main()
