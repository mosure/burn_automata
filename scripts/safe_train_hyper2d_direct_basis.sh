#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 --config configs/hyper2d_direct_basis/<experiment>.toml [extra train-hyper2d-direct-basis args...]" >&2
  exit 2
fi

TIMEOUT_VALUE="${BURN_AUTOMATA_TIMEOUT:-6h}"
MEMORY_MAX="${BURN_AUTOMATA_MEMORY_MAX:-32G}"
CARGO_BIN="${CARGO:-/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo}"
RUSTC_BIN="${RUSTC:-/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc}"

cmd=(
  timeout --foreground "$TIMEOUT_VALUE"
  env "RUSTC=$RUSTC_BIN"
  "$CARGO_BIN" run --release -p burn_automata --features cli,backend_wgpu
  --bin burn_automata -- train-hyper2d-direct-basis "$@"
)

if command -v systemd-run >/dev/null 2>&1; then
  exec systemd-run --user --scope \
    -p "MemoryMax=$MEMORY_MAX" \
    -p "MemorySwapMax=0" \
    "${cmd[@]}"
fi

echo "warning: systemd-run not found; running with timeout only and no process memory cgroup" >&2
exec "${cmd[@]}"
