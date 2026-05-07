#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-QKV}"
REPO_ROOT="/workspace"
BACKEND_DIR="${REPO_ROOT}/targets/${TARGET}/backend"
OUT_DIR="${REPO_ROOT}/backends"

if [[ ! -d "${BACKEND_DIR}" ]]; then
  echo "error: generated backend not found: ${BACKEND_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

(
  cd "${BACKEND_DIR}"
  cargo build --release
)

cp "${BACKEND_DIR}/target/release/backend" "${OUT_DIR}/${TARGET}"
chmod +x "${OUT_DIR}/${TARGET}"

echo "Built backend: backends/${TARGET}"
