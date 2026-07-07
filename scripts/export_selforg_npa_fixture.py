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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=pathlib.Path, default=".cache/selforg_npa/NPA")
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--target-image", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    config_path = args.config or upstream_root / "configs/growing.yaml"
    checkpoint_path = args.checkpoint or upstream_root / "data/pretrained/lizard.pth"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

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
        "target": extract_target(upstream_root, config["train"]["target"], args.target_image),
        "checkpoint": checkpoint_summary(checkpoint_path),
    }

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
