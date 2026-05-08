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
# now copy the cost model file into the correct place
BACKEND=${SPEC%.py}

BACKEND_IMPROVED=${BACKEND}_improved
MODEL_PY_TEMPLATE=${REPO_ROOT}/cost-models/${BACKEND}/model.py
MODEL_PY_PATH=${REPO_ROOT}/targets/${BACKEND_IMPROVED}/backend/python/cost/model.py

MODEL_SMOOTHE_TEMPLATE=${REPO_ROOT}/cost-models/${BACKEND}/smoothe.rs
MODEL_SMOOTHE_PATH=${REPO_ROOT}/targets/${BACKEND_IMPROVED}/backend/src/isel/extractor/smoothe.rs
cp $MODEL_SMOOTHE_TEMPLATE $MODEL_SMOOTHE_PATH
cp $MODEL_PY_TEMPLATE $MODEL_PY_PATH

