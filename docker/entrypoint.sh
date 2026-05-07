#!/bin/bash
set -e

if [ "$#" -eq 0 ]; then
    bash -l
else
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate act
    export PATH="/opt/cargo/bin:$PATH"
    exec "$@"
fi
