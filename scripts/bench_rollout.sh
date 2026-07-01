#!/usr/bin/env bash
set -euo pipefail

PARTICLES="${PARTICLES:-4096}"
STEPS="${STEPS:-1}"
PRESET="${PRESET:-growing-2d}"
PROFILE="${PROFILE:-1}"
RELEASE="${RELEASE:-1}"

PROFILE_ARG=()
if [[ "${PROFILE}" == "1" ]]; then
  PROFILE_ARG=(--profile)
fi

RELEASE_ARG=()
if [[ "${RELEASE}" == "1" ]]; then
  RELEASE_ARG=(--release)
fi

cargo run "${RELEASE_ARG[@]}" -p burn_automata --bin burn_automata -- bench \
  --preset "${PRESET}" \
  --particles "${PARTICLES}" \
  --steps "${STEPS}" \
  "${PROFILE_ARG[@]}"
