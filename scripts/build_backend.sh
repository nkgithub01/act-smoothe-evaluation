#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_backend.sh [TARGET]

Build generated backend inside the ACT Docker image.
Default TARGET: QKV
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${1:-QKV}"
ARCH="$(uname -m)"

case "${ARCH}" in
  x86_64) IMAGE_NAME="act-alpha:latest-amd64" ;;
  arm64|aarch64) IMAGE_NAME="act-alpha:latest-arm64" ;;
  *) echo "error: unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "error: local ACT image not found: ${IMAGE_NAME}" >&2
  echo "Build it first with: docker/build.sh" >&2
  exit 1
fi

docker run -it --rm \
  --gpus all \
  --name "act-rm-$(id -un)-build-backend" \
  -v "${REPO_ROOT}:/workspace:rw" \
  -w "/workspace" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "${IMAGE_NAME}" \
  ./scripts/dockerized/build_backend.sh "${TARGET}"
