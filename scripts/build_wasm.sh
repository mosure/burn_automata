#!/usr/bin/env bash
set -euo pipefail

rustup target add wasm32-unknown-unknown
export RUSTFLAGS="${RUSTFLAGS:-} --cfg getrandom_backend=\"wasm_js\""
cargo build -p bevy_automata --no-default-features --features viewer --target wasm32-unknown-unknown
