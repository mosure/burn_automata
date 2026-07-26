#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

fail() {
  echo "repository layout check failed: $*" >&2
  exit 1
}

mapfile -t sandbox_files < <(git ls-files configs/sandbox)
if [[ "${#sandbox_files[@]}" -ne 1 || "${sandbox_files[0]}" != "configs/sandbox/README.md" ]]; then
  fail "configs/sandbox must contain only its tracked README"
fi

mapfile -t root_scripts < <(
  git ls-files scripts |
    awk -F/ 'NF == 2 && $0 != "scripts/README.md"'
)
if [[ "${#root_scripts[@]}" -ne 0 ]]; then
  printf '%s\n' "${root_scripts[@]}" >&2
  fail "operational scripts must live in a purpose-specific subdirectory"
fi

mapfile -t root_docs < <(
  git ls-files docs |
    awk -F/ 'NF == 2 && $0 != "docs/README.md"'
)
if [[ "${#root_docs[@]}" -ne 0 ]]; then
  printf '%s\n' "${root_docs[@]}" >&2
  fail "documentation must live under architecture, development, evidence, papers, or research"
fi

if git ls-files |
  grep -Eq '(^|/)(__pycache__|target|artifacts)(/|$)|\.(aux|bbl|blg|fdb_latexmk|fls|log|out|pyc)$'; then
  fail "generated artifacts or caches are tracked"
fi

for paper in \
  docs/papers/adaptive/adaptive_npa.pdf \
  docs/papers/hypernpa/hyper_npa.pdf; do
  [[ -s "${paper}" ]] || fail "missing versioned publication: ${paper}"
done

for legacy_root in \
  configs/render3d \
  configs/render3d_adapters \
  configs/verified/adaptive \
  configs/verified/2d/hyper_e2e \
  docs/benchmarks \
  docs/hyper_npa_figures \
  docs/adaptive_npa_figures; do
  if git ls-files "${legacy_root}" | grep -q .; then
    fail "legacy tracked root remains: ${legacy_root}"
  fi
done

echo "repository layout is canonical"
