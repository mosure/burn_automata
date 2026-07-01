#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all --check
cargo check -p burn_automata_kernels -p burn_automata --all-targets
cargo check -p burn_automata --examples --benches
cargo check -p burn_automata --no-default-features --features backend_wgpu
cargo check -p burn_automata --no-default-features --features gpu_wgpu
cargo test -p burn_automata_kernels -p burn_automata -p bevy_burn
cargo test -p burn_automata --test gpu_wgpu --no-default-features --features gpu_wgpu
cargo test -p bevy_automata --no-default-features --features "splatting gpu_wgpu" --test gaussian_gpu_link
cargo check -p bevy_automata --no-default-features --features viewer
cargo check -p bevy_automata
