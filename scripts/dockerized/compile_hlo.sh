#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/dockerized/compile_hlo.sh BACKEND INPUT.hlo" >&2
  exit 1
fi

BACKEND="$1"
INPUT="$(basename "$2")"
INPUT_DIR="/workspace/kernels"
IN_FILE_NAME="${INPUT%.hlo}"

mkdir -p /workspace/asm
/workspace/backends/${BACKEND} \
  --input "${INPUT_DIR}/${INPUT}" \
  --output "/workspace/asm/compiled_${IN_FILE_NAME}_${BACKEND}.py"
