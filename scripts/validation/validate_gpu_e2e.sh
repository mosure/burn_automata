#!/usr/bin/env bash
set -euo pipefail

PARTICLES="${PARTICLES:-64}"
STEPS="${STEPS:-4}"
TOLERANCE="${TOLERANCE:-0.002}"
PSNR_THRESHOLD="${PSNR_THRESHOLD:-70}"
HIDDEN_PSNR_THRESHOLD="${HIDDEN_PSNR_THRESHOLD:-70}"
LIZARD_BPK="${LIZARD_BPK:-models/catalog/growing/lizard.bpk}"
POLKA_BPK="${POLKA_BPK:-models/catalog/texture/polka_dotted_0121.bpk}"
REQUIRE_BPK="${REQUIRE_BPK:-0}"
CATALOG_PARITY="${CATALOG_PARITY:-0}"
SELFORG_WEB_ROOT="${SELFORG_WEB_ROOT:-/tmp/selforg_npa_web}"

cargo test -p burn_automata --test gpu_wgpu --no-default-features --features "cli gpu_wgpu" -- --nocapture
cargo test -p bevy_automata --no-default-features --features "splatting gpu_wgpu" --test gaussian_gpu_link -- --nocapture

validate_bpk() {
  local model_path="$1"
  local preset="$2"
  local seed_scale="$3"
  if [[ ! -f "$model_path" ]]; then
    if [[ "$REQUIRE_BPK" == "1" ]]; then
      echo "missing required model: $model_path" >&2
      exit 1
    fi
    echo "skipping missing model: $model_path"
    return
  fi
  python3 scripts/reference/selforg/validate_import_parity.py \
    --model "$model_path" \
    --particles "$PARTICLES" \
    --preset "$preset" \
    --seed-scale "$seed_scale" \
    --gpu \
    --steps "$STEPS" \
    --tolerance "$TOLERANCE" \
    --psnr-threshold "$PSNR_THRESHOLD" \
    --hidden-psnr-threshold "$HIDDEN_PSNR_THRESHOLD"
}

validate_bpk "$LIZARD_BPK" growing-2d 0.2
validate_bpk "$POLKA_BPK" texture-2d 1.0

if [[ "$CATALOG_PARITY" == "1" ]]; then
  python3 scripts/reference/selforg/validate_catalog_parity.py \
    --web-root "$SELFORG_WEB_ROOT" \
    --gpu \
    --build-binary \
    --require-all \
    --particles "$PARTICLES" \
    --steps "$STEPS" \
    --tolerance "$TOLERANCE" \
    --psnr-threshold "$PSNR_THRESHOLD" \
    --hidden-psnr-threshold "$HIDDEN_PSNR_THRESHOLD"
fi
