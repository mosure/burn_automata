#!/usr/bin/env python3
"""Import SelfOrg-NPA web-demo JSON models into BPK files.

The official implementation ships two PyTorch checkpoints, but the web demo
publishes many more trained 2D models as base64 JSON tensors. This importer
converts the non-equivariant web format into the same BPK manifest used by the
Rust runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BPK_MAGIC = b"BAUTBPK1"
BPK_VERSION = 1


@dataclass(frozen=True)
class Tensor:
    shape: list[int]
    values: list[float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("scripts/selforg_catalog.json"))
    parser.add_argument("--web-root", type=Path, default=Path("/tmp/selforg_npa_web"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    entries = load_catalog(args.catalog)
    if args.only:
        requested = set(args.only)
        entries = [entry for entry in entries if entry["slug"] in requested]
        missing = sorted(requested - {entry["slug"] for entry in entries})
        if missing:
            raise SystemExit(f"unknown catalog slug(s): {', '.join(missing)}")

    if args.list:
        for entry in entries:
            print(f"{entry['slug']}: {entry['source_json']}#{entry['model_key']} -> {entry['output']}")
        return

    summary = []
    for entry in entries:
        output = Path(entry["output"])
        if output.exists() and not args.force:
            summary.append({"slug": entry["slug"], "output": str(output), "status": "exists"})
            print(f"exists {entry['slug']}: {output}")
            continue
        if args.dry_run:
            summary.append({"slug": entry["slug"], "output": str(output), "status": "dry-run"})
            print(f"would import {entry['slug']}: {output}")
            continue
        manifest = import_entry(args.web_root, entry)
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = write_bpk(output, manifest)
        summary.append(
            {
                "slug": entry["slug"],
                "output": str(output),
                "status": "imported",
                "sha256": digest,
                "parameter_count": parameter_count(manifest),
            }
        )
        print(f"imported {entry['slug']}: {output} sha256={digest}")

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")


def load_catalog(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON list")
    return entries


def import_entry(web_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    source_path = web_root / entry["source_json"]
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing {source_path}; clone https://github.com/SelfOrg-NPA/SelfOrg-NPA.github.io "
            "or pass --web-root"
        )
    models = json.loads(source_path.read_text())
    model_key = entry["model_key"]
    if model_key not in models:
        raise KeyError(f"{model_key!r} not found in {source_path}")
    payload = models[model_key]
    if "W_hidden" in payload or "W_out" in payload:
        raise ValueError(
            f"{entry['slug']} uses the equivariant web architecture, which is not yet "
            "representable as a scalar MLP BPK"
        )
    if "w2.weight.T" not in payload:
        raise ValueError(f"{entry['slug']} is missing web-demo tensor w2.weight.T")

    w1 = decode_tensor(payload["w1.weight"])
    b1 = decode_tensor(payload["w1.bias"])
    w2t = decode_tensor(payload["w2.weight.T"])

    hidden_dims, perception_dims = expect_rank2(w1, "w1.weight")
    if b1.shape != [hidden_dims]:
        raise ValueError(f"w1.bias shape {b1.shape} != [{hidden_dims}]")
    w2_hidden, padded_web_update_dims = expect_rank2(w2t, "w2.weight.T")
    if w2_hidden != hidden_dims:
        raise ValueError(f"w2 hidden dim {w2_hidden} != {hidden_dims}")

    spatial_dims = 2
    state_dims = infer_state_dims(perception_dims, spatial_dims)
    unpadded_perception_dims = state_dims * (2 + spatial_dims) + spatial_dims
    update_dims = spatial_dims + state_dims
    if padded_web_update_dims < update_dims:
        raise ValueError(
            f"web w2 output dim {padded_web_update_dims} cannot fit {update_dims} update dims"
        )

    w2 = unpack_web_w2(w2t.values, hidden_dims, padded_web_update_dims, state_dims, spatial_dims)
    eps0 = float(payload["eps0"])
    alpha = float(payload["alpha"])
    config = {
        "spatial_dims": spatial_dims,
        "state_dims": state_dims,
        "hidden_dims": hidden_dims,
        "eps0": eps0,
        "alpha": alpha,
        "density_grad": True,
        "state_grad": True,
        "log_norm_grad": True,
        "log_norm_density_grad": True,
        "stopgrad_pos": True,
        "stopgrad_state": False,
        "equivariance": "ParticleDensityAndScale",
        "decoder_dims": None,
        "output_dims": None,
    }
    return {
        "format_version": 1,
        "model_kind": "npa",
        "source": f"{source_path}#{model_key}",
        "config": config,
        "hashgrid": inferred_hashgrid(eps0),
        "weights": {
            "w1": unpack_padded_w1(w1.values, hidden_dims, perception_dims, unpadded_perception_dims),
            "b1": b1.values,
            "w2": w2,
            "b2": [0.0] * update_dims,
        },
    }


def decode_tensor(spec: dict[str, Any]) -> Tensor:
    if spec.get("dtype") != "float32":
        raise ValueError(f"only float32 tensors are supported, got {spec.get('dtype')!r}")
    shape = [int(dim) for dim in spec["shape"]]
    count = math.prod(shape)
    raw = base64.b64decode(spec["data64"])
    expected = count * 4
    if len(raw) != expected:
        raise ValueError(f"tensor byte len {len(raw)} != expected {expected}")
    values = list(struct.unpack("<" + "f" * count, raw))
    return Tensor(shape, values)


def expect_rank2(tensor: Tensor, name: str) -> tuple[int, int]:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} rank {len(tensor.shape)} != 2")
    return tensor.shape[0], tensor.shape[1]


def infer_state_dims(padded_perception_dims: int, spatial_dims: int) -> int:
    for state_dims in range(1, 512):
        perception_dims = state_dims * (2 + spatial_dims) + spatial_dims
        padding = padded_perception_dims - perception_dims
        if 0 <= padding < 4:
            return state_dims
    raise ValueError(f"cannot infer state dims from perception dim {padded_perception_dims}")


def unpack_padded_w1(
    values: list[float],
    hidden_dims: int,
    padded_perception_dims: int,
    perception_dims: int,
) -> list[float]:
    out = []
    for hidden in range(hidden_dims):
        base = hidden * padded_perception_dims
        out.extend(values[base : base + perception_dims])
    return out


def unpack_web_w2(
    w2t: list[float],
    hidden_dims: int,
    padded_web_update_dims: int,
    state_dims: int,
    spatial_dims: int,
) -> list[float]:
    # The web demo stores columns as [state_delta..., dx..., padding...] so that
    # state chunks and position update fit its packed RGBA state texture. Rust
    # and the upstream PyTorch model use [dx..., state_delta...].
    web_order = list(range(state_dims, state_dims + spatial_dims)) + list(range(state_dims))
    out = []
    for output_channel in web_order:
        for hidden in range(hidden_dims):
            out.append(w2t[hidden * padded_web_update_dims + output_channel])
    return out


def inferred_hashgrid(eps0: float) -> dict[str, Any]:
    if abs(eps0 - 0.2) < 1.0e-5:
        return {
            "dim": 2,
            "boundary": "Periodic",
            "mode": "Grid",
            "grid_size": [10, 10, 1],
            "eps": eps0,
            "max_particles_per_block": 32,
        }
    return {
        "dim": 2,
        "boundary": "Clamped",
        "mode": "Grid",
        "grid_size": [16, 16, 1],
        "eps": eps0,
        "max_particles_per_block": 32,
    }


def write_bpk(path: Path, manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False).encode()
    digest = hashlib.sha256(payload).digest()
    header = BPK_MAGIC + struct.pack("<I", BPK_VERSION) + struct.pack("<Q", len(payload)) + digest
    path.write_bytes(header + payload)
    return digest.hex()


def parameter_count(manifest: dict[str, Any]) -> int:
    weights = manifest["weights"]
    return sum(len(weights[name]) for name in ("w1", "b1", "w2", "b2"))


if __name__ == "__main__":
    main()
