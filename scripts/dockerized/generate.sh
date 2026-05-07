#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-QKV.py}"
REPO_ROOT="/workspace"

cd "${REPO_ROOT}"

if [[ ! -f "${SPEC}" ]]; then
  echo "error: ISA spec not found: ${SPEC}" >&2
  exit 1
fi

python "${SPEC}"
