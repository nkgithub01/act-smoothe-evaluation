#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/enter_docker.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCH="$(uname -m)"
MODE=""
CMD=()

case "${ARCH}" in
  x86_64)
    IMAGE_NAME="act-alpha:latest-amd64"
    ;;
  arm64|aarch64)
    IMAGE_NAME="act-alpha:latest-arm64"
    ;;
  *)
    echo "error: unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac


if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "error: local ACT image not found: ${IMAGE_NAME}" >&2
  echo "Build it first with: docker/build.sh" >&2
  exit 1
fi

DOCKER_ARGS=(-it --rm)
if [[ ${#CMD[@]} -gt 0 ]]; then
  DOCKER_ARGS=(-i --rm)
fi

CONTAINER_NAME="act-rm-$(id -un)-${MODE}"

docker run "${DOCKER_ARGS[@]}" \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  -v "${REPO_ROOT}:/workspace:rw" \
  -w "/workspace" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "${IMAGE_NAME}" \
  "${CMD[@]}"