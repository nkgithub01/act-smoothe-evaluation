#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    IMAGE_NAME="act-alpha:latest-amd64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

HOST_MOUNT="$(pwd)/.."

docker run --rm --entrypoint bash \
  --gpus all \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*"
