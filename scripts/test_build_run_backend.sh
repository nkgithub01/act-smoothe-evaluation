SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR=/workspace/kernels
BACKEND=$1
KERNEL=attention.hlo
${SCRIPT_DIR}/docker.sh --compile -- ${SCRIPT_DIR}/build_backend.sh $BACKEND
