#!/usr/bin/env bash
set -euo pipefail

repo_url="${SELFORG_NPA_REPO_URL:-https://github.com/TheDevilWillBeBee/NPA.git}"
commit="${SELFORG_NPA_COMMIT:-1176ead09ef34e21814fd6b1dd29ca661aedb5da}"
cache_dir="${SELFORG_NPA_CACHE_DIR:-.cache/selforg_npa/NPA}"

if [[ ! -d "${cache_dir}/.git" ]]; then
  mkdir -p "$(dirname "${cache_dir}")"
  git clone --depth 1 "${repo_url}" "${cache_dir}"
fi

git -C "${cache_dir}" fetch --depth 1 origin "${commit}"
git -C "${cache_dir}" checkout --detach "${commit}"
git -C "${cache_dir}" rev-parse HEAD
