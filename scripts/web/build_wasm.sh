#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="wasm32-unknown-unknown"
PROFILE="${WASM_PROFILE:-wasm-release}"
OUT_DIR="${WEB_OUT_DIR:-${ROOT}/www}"
WASM_BINDGEN_VERSION="$(
  awk '
    $0 == "name = \"wasm-bindgen\"" { found = 1; next }
    found && /^version = / { gsub(/"/, "", $3); print $3; exit }
  ' "${ROOT}/Cargo.lock"
)"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen-cli ${WASM_BINDGEN_VERSION} is required" >&2
  exit 1
fi
if [[ "$(wasm-bindgen --version)" != "wasm-bindgen ${WASM_BINDGEN_VERSION}" ]]; then
  echo "wasm-bindgen-cli must match Cargo.lock (${WASM_BINDGEN_VERSION})" >&2
  exit 1
fi

rustup target add wasm32-unknown-unknown
rm -rf "${OUT_DIR}/pkg" "${OUT_DIR}/worker_pkg"
mkdir -p "${OUT_DIR}/pkg" "${OUT_DIR}/worker_pkg"

cargo build \
  --locked \
  --profile "${PROFILE}" \
  --target "${TARGET}" \
  -p bevy_automata \
  --no-default-features \
  --features viewer,splatting,gpu_wgpu,hyper_dino_wgpu
wasm-bindgen \
  --target web \
  --out-dir "${OUT_DIR}/pkg" \
  --out-name bevy_automata \
  "${ROOT}/target/${TARGET}/${PROFILE}/bevy_automata.wasm"

cargo build \
  --locked \
  --profile "${PROFILE}" \
  --target "${TARGET}" \
  -p burn_automata_web_worker
wasm-bindgen \
  --target web \
  --out-dir "${OUT_DIR}/worker_pkg" \
  --out-name burn_automata_web_worker \
  "${ROOT}/target/${TARGET}/${PROFILE}/burn_automata_web_worker.wasm"

printf 'viewer: %s\nworker: %s\n' \
  "$(du -h "${OUT_DIR}/pkg/bevy_automata_bg.wasm" | cut -f1)" \
  "$(du -h "${OUT_DIR}/worker_pkg/burn_automata_web_worker_bg.wasm" | cut -f1)"
