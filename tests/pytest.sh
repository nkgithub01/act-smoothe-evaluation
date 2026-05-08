#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/exec.sh \
  "PYTHONPATH=/workspace pytest tests/test_qkv.py && \
  pytest tests/test_qkv_new.py && \
  pytest tests/test_qkv_improved.py && \
  pytest tests/test_qkv_new_improved.py"
