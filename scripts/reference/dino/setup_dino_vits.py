#!/usr/bin/env python3
"""Download and import DINOv2 ViT-S/14 weights for HyperNPA conditioning.

This script intentionally writes only ignored model artifacts under
`models/dino/`. It uses the local PyTorch environment to convert the official
PyTorch checkpoint to safetensors, then invokes the `burn_dino` import tool to
produce the Burn NamedMpk checkpoint consumed by `--condition-encoder dino`.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file


DINO_VITS_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_pretrain.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/dino"))
    parser.add_argument("--url", default=DINO_VITS_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--burn-dino-manifest",
        type=Path,
        default=None,
        help="Optional path to burn_dino Cargo.toml. Auto-detected from ~/.cargo otherwise.",
    )
    return parser.parse_args()


def download(url: str, path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        print(f"using existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total_bytes:
                pct = copied * 100.0 / total_bytes
                print(f"\r  {copied / (1024 * 1024):.1f} MiB ({pct:.1f}%)", end="")
        if total_bytes:
            print()
    tmp.replace(path)


def normalize_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("model", "teacher", "state_dict"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                raw = nested
                break
    if not isinstance(raw, dict):
        raise TypeError(f"expected a checkpoint dict, got {type(raw)!r}")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        clean = str(key)
        for prefix in ("module.", "backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        state[clean] = value.detach().cpu().contiguous()
    if not state:
        raise ValueError("checkpoint did not contain tensor weights")
    return state


def convert_to_safetensors(pth_path: Path, safetensors_path: Path, refresh: bool) -> None:
    if safetensors_path.exists() and not refresh:
        print(f"using existing {safetensors_path}")
        return
    print(f"converting {pth_path} -> {safetensors_path}")
    raw = torch.load(pth_path, map_location="cpu")
    state = normalize_state_dict(raw)
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, safetensors_path)
    print(f"wrote {len(state)} tensors")


def find_burn_dino_manifest(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(explicit)
    patterns = [
        str(Path.home() / ".cargo/registry/src/*/burn_dino-0.8.0/Cargo.toml"),
        str(Path.home() / ".cargo/git/checkouts/burn_dino-*/**/Cargo.toml"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return Path(matches[-1])
    raise FileNotFoundError(
        "could not find burn_dino Cargo.toml; run `cargo fetch` first or pass "
        "--burn-dino-manifest"
    )


def import_to_burn(manifest: Path, weights: Path, output_base: Path) -> None:
    output_mpk = output_base.with_suffix(".mpk")
    if output_mpk.exists():
        print(f"using existing {output_mpk}")
        return
    print(f"importing {weights} -> {output_mpk}")
    subprocess.run(
        [
            "cargo",
            "run",
            "--manifest-path",
            str(manifest),
            "--features",
            "import",
            "--bin",
            "import",
            "--",
            "--vit-type",
            "small",
            "--weights-path",
            str(weights),
            "--output",
            str(output_base),
            "--skip-pca",
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    pth_path = output_dir / "dinov2_vits14_pretrain.pth"
    safetensors_path = output_dir / "dinov2_vits14_pretrain.safetensors"
    burn_output = output_dir / "dino_vits"
    download(args.url, pth_path, args.refresh)
    convert_to_safetensors(pth_path, safetensors_path, args.refresh)
    manifest = find_burn_dino_manifest(args.burn_dino_manifest)
    import_to_burn(manifest, safetensors_path, burn_output)
    print(f"ready: {burn_output.with_suffix('.mpk')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
