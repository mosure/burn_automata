#!/usr/bin/env python3
"""Export a PyTorch NPA checkpoint into burn_automata JSON interchange format.

This script expects torch to be installed. It intentionally handles only the
two-layer update MLP used by the initial Rust model schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def tensor_to_list(value: Any) -> list[float]:
    value = value.detach().cpu().contiguous().view(-1)
    return [float(x) for x in value.tolist()]


def find_linear_layers(state: dict[str, Any]) -> tuple[str, str]:
    weight_keys = [key for key, value in state.items() if key.endswith(".weight") and getattr(value, "ndim", None) == 2]
    weight_keys.sort()
    if len(weight_keys) < 2:
        raise SystemExit("checkpoint does not contain at least two 2D linear weight tensors")
    first, second = weight_keys[0], weight_keys[1]
    return first.removesuffix(".weight"), second.removesuffix(".weight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spatial-dims", type=int, default=2)
    parser.add_argument("--state-dims", type=int, default=16)
    parser.add_argument("--hidden-dims", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise SystemExit("checkpoint root is not a tensor dictionary")

    first, second = find_linear_layers(state)
    payload = {
        "config": {
            "spatial_dims": args.spatial_dims,
            "state_dims": args.state_dims,
            "hidden_dims": args.hidden_dims,
            "eps0": args.eps,
            "alpha": 0.5,
            "density_grad": True,
            "state_grad": True,
            "log_norm_grad": True,
            "log_norm_density_grad": True,
            "stopgrad_pos": True,
            "stopgrad_state": False,
            "decoder_dims": None,
            "output_dims": None,
        },
        "hashgrid": {
            "dim": args.spatial_dims,
            "boundary": "Clamped",
            "mode": "Grid",
            "grid_size": [16, 16, 1],
            "eps": args.eps,
            "max_particles_per_block": 32,
        },
        "w1": tensor_to_list(state[f"{first}.weight"]),
        "b1": tensor_to_list(state.get(f"{first}.bias", torch.zeros(args.hidden_dims))),
        "w2": tensor_to_list(state[f"{second}.weight"]),
        "b2": tensor_to_list(state.get(f"{second}.bias", torch.zeros(args.spatial_dims + args.state_dims))),
        "source": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
