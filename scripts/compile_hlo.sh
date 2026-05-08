#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/compile_hlo.sh BACKEND INPUT.hlo

Compile a kernel inside the ACT Docker image.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND="$1"
INPUT="$2"
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

export LD_LIBRARY_PATH="/opt/ortools/lib:${LD_LIBRARY_PATH:-}"

docker run -it --rm \
  --gpus all \
  --name "act-rm-$(id -un)-compile-hlo" \
  -v "${REPO_ROOT}:/workspace:rw" \
  -w "/workspace" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e LD_LIBRARY_PATH \
  "${IMAGE_NAME}" \
  ./scripts/dockerized/compile_hlo.sh "${BACKEND}" "${INPUT}"
