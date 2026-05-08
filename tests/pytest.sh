#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 <QKV|QKV_new> <original|improved>" >&2
  echo "Example: $0 QKV improved" >&2
}

if [[ $# -ne 2 ]]; then
  echo "Error: wrong number of arguments." >&2
  usage
  exit 1
fi

case "$1:$2" in
  QKV:original)
    test_file="tests/test_qkv.py"
    ;;
  QKV:improved)
    test_file="tests/test_qkv_improved.py"
    ;;
  QKV_new:original)
    test_file="tests/test_qkv_new.py"
    ;;
  QKV_new:improved)
    test_file="tests/test_qkv_new_improved.py"
    ;;
  *)
    echo "Error: invalid arguments '$1' '$2'." >&2
    usage
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."

./scripts/exec.sh "PYTHONPATH=/workspace pytest ${test_file}"
