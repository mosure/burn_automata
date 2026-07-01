#!/usr/bin/env python3
"""Validate Rust BPK inference against a pure Python SPH reference.

This script is intentionally dependency-free and slow. It is meant for small
particle counts that catch import/layout/kernel parity regressions, not for
throughput benchmarking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path


BPK_MAGIC = b"BAUTBPK1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--preset", default="growing-2d")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed-scale", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--psnr-threshold", type=float, default=70.0)
    parser.add_argument("--hidden-psnr-threshold", type=float, default=70.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--binary", type=Path, default=Path("target/release/burn_automata"))
    parser.add_argument("--use-binary-for-gpu", action="store_true")
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.model)
    with tempfile.TemporaryDirectory(prefix="burn_automata_parity_") as tmp:
        initial_path = Path(tmp) / "initial.json"
        rust_path = Path(tmp) / "rust_rollout.json"
        run_infer(args, initial_path, steps=0, gpu=False)
        run_infer(args, rust_path, steps=args.steps, gpu=args.gpu)
        initial = json.loads(initial_path.read_text())
        rust = json.loads(rust_path.read_text())

    positions = initial["positions"]
    states = initial["states"]
    for _ in range(args.steps):
        positions, states = step_python(
            positions,
            states,
            manifest["config"],
            manifest["hashgrid"],
            manifest["weights"],
        )
    max_pos_err = max_abs_nested(positions, rust["positions"])
    max_state_err = max_abs_flat(states, rust["states"])
    position_psnr = signal_psnr(flatten_nested(positions), flatten_nested(rust["positions"]))
    hidden_state_psnr = signal_psnr(states, rust["states"])
    tail_rgb_psnr = tail_rgb_state_psnr(states, rust["states"], manifest["config"])
    rendered_gaussian_psnr = None
    if manifest["config"]["spatial_dims"] == 2:
        py_image = rasterize_particles(
            positions,
            states,
            manifest["config"],
            manifest["hashgrid"],
            args.image_size,
            args.sigma,
        )
        rust_image = rasterize_particles(
            rust["positions"],
            rust["states"],
            manifest["config"],
            manifest["hashgrid"],
            args.image_size,
            args.sigma,
        )
        rendered_gaussian_psnr = image_psnr(py_image, rust_image)
    metrics = {
        "model": str(args.model),
        "backend": "wgpu" if args.gpu else "cpu",
        "particles": args.particles,
        "steps": args.steps,
        "max_position_abs_error": max_pos_err,
        "max_state_abs_error": max_state_err,
        "tolerance": args.tolerance,
        "position_psnr_db": position_psnr,
        "hidden_state_psnr_db": hidden_state_psnr,
        "hidden_psnr_threshold_db": args.hidden_psnr_threshold,
        "tail_rgb_psnr_db": tail_rgb_psnr,
        "rendered_gaussian_psnr_db": rendered_gaussian_psnr,
        "raster_psnr_db": rendered_gaussian_psnr,
        "psnr_threshold_db": args.psnr_threshold if rendered_gaussian_psnr is not None else None,
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    if max(max_pos_err, max_state_err) > args.tolerance:
        raise SystemExit(1)
    if hidden_state_psnr < args.hidden_psnr_threshold:
        raise SystemExit(1)
    if rendered_gaussian_psnr is not None and rendered_gaussian_psnr < args.psnr_threshold:
        raise SystemExit(1)


def run_infer(args: argparse.Namespace, output: Path, steps: int, gpu: bool) -> None:
    binary = args.binary
    cmd: list[str]
    if binary.exists() and (not gpu or args.use_binary_for_gpu):
        cmd = [str(binary)]
    else:
        cmd = ["cargo", "run", "--release", "-p", "burn_automata"]
        if gpu:
            cmd += ["--features", "gpu_wgpu"]
        cmd += ["--bin", "burn_automata", "--"]
    cmd += [
        "infer",
        "--preset",
        args.preset,
        "--model",
        str(args.model),
        "--particles",
        str(args.particles),
        "--steps",
        str(steps),
        "--update-prob",
        "1.0",
        "--output",
        str(output),
    ]
    if gpu:
        cmd += ["--gpu"]
    if args.seed_scale is not None:
        cmd += ["--seed-scale", str(args.seed_scale)]
    subprocess.run(cmd, check=True)


def load_manifest(path: Path) -> dict:
    data = path.read_bytes()
    if data.startswith(BPK_MAGIC):
        if len(data) < 52:
            raise ValueError("BPK file is shorter than its header")
        payload_len = struct.unpack("<Q", data[12:20])[0]
        payload = data[52 : 52 + payload_len]
        return json.loads(payload)
    return json.loads(data)


def step_python(
    positions: list[list[float]],
    states: list[float],
    config: dict,
    grid: dict,
    weights: dict,
) -> tuple[list[list[float]], list[float]]:
    state_dims = config["state_dims"]
    dim = grid["dim"]
    particle_count = len(positions)
    rho = density(positions, grid)
    blur, grad_s, grad_rho = perceive_second_pass(positions, states, rho, state_dims, config, grid)
    features: list[float] = []
    for idx in range(particle_count):
        state_base = idx * state_dims
        features.extend(states[state_base : state_base + state_dims])
        features.extend(blur[state_base : state_base + state_dims])
        if config["state_grad"]:
            features.extend(grad_s[idx * state_dims * dim : (idx + 1) * state_dims * dim])
        if config["density_grad"]:
            features.extend(grad_rho[idx * dim : (idx + 1) * dim])
    update = mlp(features, config, weights)
    next_positions = [row[:] for row in positions]
    next_states = states[:]
    out_dims = config["spatial_dims"] + state_dims
    eps_motion = motion_eps(config, grid)
    for idx in range(particle_count):
        base = idx * out_dims
        norm = math.sqrt(sum(update[base + axis] ** 2 for axis in range(config["spatial_dims"])))
        for axis in range(config["spatial_dims"]):
            dx = config["alpha"] * update[base + axis] * eps_motion / (1.0 + norm)
            next_positions[idx][axis] += dx
            next_positions[idx][axis] = wrap_axis(next_positions[idx][axis], axis, grid)
        state_base = idx * state_dims
        for c in range(state_dims):
            next_states[state_base + c] += update[base + config["spatial_dims"] + c]
    return next_positions, next_states


def density(positions: list[list[float]], grid: dict) -> list[float]:
    out = []
    for pi in positions:
        rho = 0.0
        for pj in positions:
            delta = neighbor_delta(pi, pj, grid)
            r2 = sum(delta[axis] * delta[axis] for axis in range(grid["dim"]))
            rho += smoothing(r2, grid)
        out.append(rho)
    return out


def perceive_second_pass(
    positions: list[list[float]],
    states: list[float],
    rho: list[float],
    state_dims: int,
    config: dict,
    grid: dict,
) -> tuple[list[float], list[float], list[float]]:
    dim = grid["dim"]
    n = len(positions)
    blur = [0.0] * (n * state_dims)
    grad_s = [0.0] * (n * state_dims * dim)
    grad_rho = [0.0] * (n * dim)
    moment = [0.0] * (n * dim * dim)
    for i, pi in enumerate(positions):
        si = i * state_dims
        for j, pj in enumerate(positions):
            delta = neighbor_delta(pi, pj, grid)
            r2 = sum(delta[axis] * delta[axis] for axis in range(dim))
            if r2 >= grid["eps"] * grid["eps"]:
                continue
            volume = 0.0 if abs(rho[j]) < 1e-20 else 1.0 / rho[j]
            smooth = smoothing(r2, grid)
            sj = j * state_dims
            for c in range(state_dims):
                blur[si + c] += states[sj + c] * smooth * volume
            if i == j:
                continue
            dens_grad = spiky(delta, r2, grid, 1.0)
            vol_grad = spiky(delta, r2, grid, volume)
            for axis in range(dim):
                grad_rho[i * dim + axis] += dens_grad[axis]
            for c in range(state_dims):
                diff = states[sj + c] - states[si + c]
                for axis in range(dim):
                    grad_s[i * state_dims * dim + c * dim + axis] += diff * vol_grad[axis]
            for row in range(dim):
                for col in range(dim):
                    moment[i * dim * dim + row * dim + col] += delta[row] * vol_grad[col]
    apply_moment(grad_s, moment, state_dims, dim)
    scale_gradients(grad_s, grad_rho, n, state_dims, config, grid)
    return blur, grad_s, grad_rho


def mlp(features: list[float], config: dict, weights: dict) -> list[float]:
    in_dims = config["state_dims"] * 2
    if config["state_grad"]:
        in_dims += config["state_dims"] * config["spatial_dims"]
    if config["density_grad"]:
        in_dims += config["spatial_dims"]
    hidden_dims = config["hidden_dims"]
    out_dims = config["spatial_dims"] + config["state_dims"]
    rows = len(features) // in_dims
    out = [0.0] * (rows * out_dims)
    for row in range(rows):
        hidden = [0.0] * hidden_dims
        f_base = row * in_dims
        for h in range(hidden_dims):
            total = weights["b1"][h]
            w_base = h * in_dims
            for i in range(in_dims):
                total += weights["w1"][w_base + i] * features[f_base + i]
            hidden[h] = max(total, 0.0)
        for o in range(out_dims):
            total = weights["b2"][o]
            w_base = o * hidden_dims
            for h in range(hidden_dims):
                total += weights["w2"][w_base + h] * hidden[h]
            out[row * out_dims + o] = total
    return out


def apply_moment(grad_s: list[float], moment: list[float], state_dims: int, dim: int) -> None:
    particles = len(moment) // (dim * dim)
    for idx in range(particles):
        inv = safe_inverse(moment[idx * dim * dim : (idx + 1) * dim * dim], dim)
        for c in range(state_dims):
            base = idx * state_dims * dim + c * dim
            raw = grad_s[base : base + dim]
            for out_axis in range(dim):
                grad_s[base + out_axis] = sum(raw[in_axis] * inv[in_axis * dim + out_axis] for in_axis in range(dim))


def scale_gradients(
    grad_s: list[float],
    grad_rho: list[float],
    particles: int,
    state_dims: int,
    config: dict,
    grid: dict,
) -> None:
    dim = grid["dim"]
    state_scale = grid["eps"] / config["eps0"] if scale_equivariant(config) else 1.0
    for idx in range(particles):
        for c in range(state_dims):
            base = idx * state_dims * dim + c * dim
            for axis in range(dim):
                grad_s[base + axis] *= state_scale
            if config["log_norm_grad"]:
                log_normalize(grad_s, base, dim)
    density_scale = (grid["eps"] / config["eps0"]) ** (1 + dim) if scale_equivariant(config) else 1.0
    if particle_density_equivariant(config):
        density_scale /= max(particles, 1)
    for idx in range(particles):
        base = idx * dim
        for axis in range(dim):
            grad_rho[base + axis] *= density_scale
        if config["log_norm_density_grad"]:
            log_normalize(grad_rho, base, dim)


def equivariance_mode(config: dict) -> str:
    mode = config.get("equivariance", "ParticleDensityAndScale")
    if isinstance(mode, str):
        return mode.replace("_", "").replace("-", "").lower()
    return "particledensityandscale"


def scale_equivariant(config: dict) -> bool:
    return equivariance_mode(config) in {"particledensityandscale", "densityandscale"}


def particle_density_equivariant(config: dict) -> bool:
    return equivariance_mode(config) in {
        "particledensity",
        "particledensityandscale",
        "densityandscale",
    }


def motion_eps(config: dict, grid: dict) -> float:
    return grid["eps"] if scale_equivariant(config) else config["eps0"]


def log_normalize(values: list[float], base: int, dim: int) -> None:
    norm = math.sqrt(sum(values[base + axis] ** 2 for axis in range(dim)))
    if norm <= 1e-12:
        for axis in range(dim):
            values[base + axis] = 0.0
        return
    scale = math.log1p(norm) / norm
    for axis in range(dim):
        values[base + axis] *= scale


def smoothing(r2: float, grid: dict) -> float:
    eps = grid["eps"]
    if r2 >= eps * eps:
        return 0.0
    coef = 4.0 / (math.pi * eps**8) if grid["dim"] == 2 else 315.0 / (64.0 * math.pi * eps**9)
    x = eps * eps - r2
    return coef * x * x * x


def spiky(delta: list[float], r2: float, grid: dict, coeff: float) -> list[float]:
    dim = grid["dim"]
    eps = grid["eps"]
    if r2 <= 0.0 or r2 >= eps * eps:
        return [0.0] * dim
    r = math.sqrt(r2)
    coef = 10.0 / (math.pi * eps**5) if dim == 2 else 15.0 / (math.pi * eps**6)
    mag = coeff * coef * 3.0 * (eps - r) ** 2 / r
    return [mag * delta[axis] for axis in range(dim)]


def safe_inverse(matrix: list[float], dim: int) -> list[float]:
    tol = 1e-3
    if dim == 2:
        a, b, d = matrix[0], matrix[1], matrix[3]
        det = a * d - b * b
        if abs(det) < tol:
            return [1.0, 0.0, 0.0, 1.0]
        inv_det = 1.0 / det
        return [d * inv_det, -b * inv_det, -b * inv_det, a * inv_det]
    a, b, c = matrix[0], matrix[1], matrix[2]
    d, e, f = matrix[4], matrix[5], matrix[8]
    t1 = d * f - e * e
    t2 = c * e - b * f
    t3 = b * e - c * d
    det = a * t1 + b * t2 + c * t3
    if abs(det) < tol:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    inv_det = 1.0 / det
    return [
        t1 * inv_det,
        t2 * inv_det,
        t3 * inv_det,
        t2 * inv_det,
        (a * f - c * c) * inv_det,
        (b * c - a * e) * inv_det,
        t3 * inv_det,
        (b * c - a * e) * inv_det,
        (a * d - b * b) * inv_det,
    ]


def neighbor_delta(lhs: list[float], rhs: list[float], grid: dict) -> list[float]:
    out = []
    for axis in range(grid["dim"]):
        d = rhs[axis] - lhs[axis]
        if grid["boundary"] == "Periodic":
            extent = grid["grid_size"][axis] * grid["eps"]
            half = extent * 0.5
            if d > half:
                d -= extent
            elif d < -half:
                d += extent
        out.append(d)
    return out


def wrap_axis(value: float, axis: int, grid: dict) -> float:
    if grid["boundary"] != "Periodic":
        return value
    extent = grid["grid_size"][axis] * grid["eps"]
    half = extent * 0.5
    return ((value + half) % extent) - half


def rasterize_particles(
    positions: list[list[float]],
    states: list[float],
    config: dict,
    grid: dict,
    image_size: int,
    sigma: float,
) -> list[float]:
    state_dims = config["state_dims"]
    width = image_size
    height = image_size
    image = [0.0] * (width * height * 3)
    weight = [0.0] * (width * height)
    extent_x = grid["grid_size"][0] * grid["eps"]
    extent_y = grid["grid_size"][1] * grid["eps"]
    half_x = extent_x * 0.5
    half_y = extent_y * 0.5
    radius = max(1, math.ceil(sigma * 3.0))
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    for idx, position in enumerate(positions):
        px = (position[0] + half_x) / extent_x * (width - 1)
        py = (position[1] + half_y) / extent_y * (height - 1)
        if not math.isfinite(px) or not math.isfinite(py):
            continue
        state_base = idx * state_dims
        color = [
            clamp01(states[state_base + state_dims - 3 + channel] + 0.5)
            for channel in range(3)
        ]
        min_x = max(0, math.floor(px - radius))
        max_x = min(width - 1, math.ceil(px + radius))
        min_y = max(0, math.floor(py - radius))
        max_y = min(height - 1, math.ceil(py + radius))
        for y in range(min_y, max_y + 1):
            dy = y - py
            for x in range(min_x, max_x + 1):
                dx = x - px
                w = math.exp(-(dx * dx + dy * dy) * inv_two_sigma2)
                pixel = y * width + x
                weight[pixel] += w
                rgb = pixel * 3
                image[rgb] += color[0] * w
                image[rgb + 1] += color[1] * w
                image[rgb + 2] += color[2] * w
    for pixel, w in enumerate(weight):
        if w <= 0.0:
            continue
        rgb = pixel * 3
        image[rgb] /= w
        image[rgb + 1] /= w
        image[rgb + 2] /= w
    return image


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def image_psnr(lhs: list[float], rhs: list[float]) -> float:
    return signal_psnr(lhs, rhs, peak=1.0)


def signal_psnr(lhs: list[float], rhs: list[float], peak: float | None = None) -> float:
    mse = sum((a - b) * (a - b) for a, b in zip(lhs, rhs)) / max(len(lhs), 1)
    if mse <= 0.0:
        return 999.0
    if peak is None:
        peak = max(1.0, max((abs(v) for v in lhs), default=0.0), max((abs(v) for v in rhs), default=0.0))
    return 20.0 * math.log10(peak / math.sqrt(mse))


def tail_rgb_state_psnr(lhs: list[float], rhs: list[float], config: dict) -> float:
    state_dims = config["state_dims"]
    lhs_rgb = []
    rhs_rgb = []
    rows = min(len(lhs), len(rhs)) // state_dims
    for row in range(rows):
        base = row * state_dims + state_dims - 3
        for channel in range(3):
            lhs_rgb.append(clamp01(lhs[base + channel] + 0.5))
            rhs_rgb.append(clamp01(rhs[base + channel] + 0.5))
    return signal_psnr(lhs_rgb, rhs_rgb, peak=1.0)


def flatten_nested(values: list[list[float]]) -> list[float]:
    return [value for row in values for value in row]


def max_abs_flat(lhs: list[float], rhs: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def max_abs_nested(lhs: list[list[float]], rhs: list[list[float]]) -> float:
    return max((abs(a - b) for row_a, row_b in zip(lhs, rhs) for a, b in zip(row_a, row_b)), default=0.0)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
