#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/generate.sh [ISA_SPEC.py]

Run backend generation inside the ACT Docker image.
Default ISA_SPEC.py: QKV.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEC="${1:-QKV.py}"
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
  --name "act-rm-$(id -un)-generate" \
  -v "${REPO_ROOT}:/workspace:rw" \
  -w "/workspace" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "${IMAGE_NAME}" \
  ./scripts/dockerized/generate.sh "${SPEC}"
