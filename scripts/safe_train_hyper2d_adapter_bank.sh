#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 --config configs/hyper2d_adapter_bank/<experiment>.toml [extra train-hyper2d-adapter-bank args...]" >&2
  exit 2
fi

TIMEOUT_VALUE="${BURN_AUTOMATA_TIMEOUT:-6h}"
MEMORY_MAX="${BURN_AUTOMATA_MEMORY_MAX:-24G}"
CARGO_BIN="${CARGO:-/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo}"
RUSTC_BIN="${RUSTC:-/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/rustc}"

cmd=(
  timeout --foreground "$TIMEOUT_VALUE"
  env "RUSTC=$RUSTC_BIN"
  "$CARGO_BIN" run --release -p burn_automata --features "cli dino backend_wgpu"
  --bin burn_automata -- train-hyper2d-adapter-bank "$@"
)

if command -v systemd-run >/dev/null 2>&1 \
  && systemd-run --user --scope --quiet true >/dev/null 2>&1; then
  exec systemd-run --user --scope \
    -p "MemoryMax=$MEMORY_MAX" \
    -p "MemorySwapMax=0" \
    "${cmd[@]}"
fi

if command -v prlimit >/dev/null 2>&1; then
  exec prlimit --as="$MEMORY_MAX" -- "${cmd[@]}"
fi

echo "warning: prlimit not found; running with timeout only and no process memory cap" >&2
exec "${cmd[@]}"
