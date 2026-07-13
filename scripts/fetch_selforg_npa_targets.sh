#!/usr/bin/env bash
set -euo pipefail

output_dir="${SELFORG_NPA_TARGET_DIR:-.cache/selforg_npa/targets/growing}"
base_url="https://github.com/googlefonts/noto-emoji/raw/main/png/512"
mkdir -p "${output_dir}"

targets=(
  "red_apple:1f34e"
  "turtle:1f422"
  "sun_with_face:1f31e"
  "butterfly:1f98b"
  "lizard:1f98e"
  "ghost:1f47b"
  "mushroom:1f344"
  "rose:1f339"
  "tropical_fish:1f420"
  "frog_face:1f438"
)

for target in "${targets[@]}"; do
  slug="${target%%:*}"
  codepoint="${target##*:}"
  output="${output_dir}/${slug}.png"
  if [[ ! -s "${output}" ]]; then
    curl --location --fail --retry 3 --silent --show-error \
      --output "${output}" "${base_url}/emoji_u${codepoint}.png"
  fi
done

printf 'fetched %s official SelfOrg-NPA targets into %s\n' "${#targets[@]}" "${output_dir}"
