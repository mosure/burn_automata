#!/usr/bin/env bash
set -euo pipefail

FEATURES="${FEATURES:-cli gpu_wgpu}"
TREE="$(cargo tree -p burn_automata --no-default-features --features "$FEATURES" -e features)"

if grep -Eq 'burn-autodiff|burn-fusion' <<<"$TREE"; then
  echo "inference feature set unexpectedly pulls autodiff/fusion dependencies" >&2
  grep -En 'burn-autodiff|burn-fusion' <<<"$TREE" >&2
  exit 1
fi

echo "inference feature set is autodiff/fusion free: $FEATURES"
