#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${ROOT}/target/burn_automata_web_models_v1.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

HYPER_DIR="artifacts/hyper2d_e2e_rollout_train_omnisvg_10k_steps3000_b16_p128s4_rank16_cosine_cuda"
REQUIRED=(
  "models/catalog/growing/lizard.bpk"
  "models/dino/dino_vits.mpk"
  "${HYPER_DIR}/shared_base.bpk"
  "${HYPER_DIR}/hyper_2d.bpk"
)

for path in "${REQUIRED[@]}"; do
  if [[ ! -f "${ROOT}/${path}" ]]; then
    echo "missing required web model: ${path}" >&2
    exit 1
  fi
done

mkdir -p \
  "${STAGE}/models/dino" \
  "${STAGE}/${HYPER_DIR}"
cp -a "${ROOT}/models/catalog" "${STAGE}/models/catalog"
cp "${ROOT}/models/dino/dino_vits.mpk" "${STAGE}/models/dino/dino_vits.mpk"
cp \
  "${ROOT}/${HYPER_DIR}/shared_base.bpk" \
  "${ROOT}/${HYPER_DIR}/hyper_2d.bpk" \
  "${STAGE}/${HYPER_DIR}/"

(
  cd "${STAGE}"
  find models artifacts -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > web_models.sha256
)

mkdir -p "$(dirname "${OUTPUT}")"
tar --sort=name --mtime="@0" --owner=0 --group=0 --numeric-owner \
  -C "${STAGE}" -cf - . \
  | gzip -n > "${OUTPUT}"
echo "${OUTPUT}"
