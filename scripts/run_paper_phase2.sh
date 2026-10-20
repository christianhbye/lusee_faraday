#!/usr/bin/env bash
# Checkpointed real-response covariance calculation with an absolute log.
set -euo pipefail

script_path=$(realpath "$0")
repo_dir=$(dirname "$(dirname "$script_path")")
log_path="$repo_dir/generated_data/paper_phase2.log"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_paper_phase2.sh"
    echo "Launch kernels, moments, numerical gates and covariance figure."
    echo "Log: $log_path"
    exit 0
fi

if [[ "${1:-}" != "--worker" ]]; then
    if [[ $# -ne 0 ]]; then
        echo "Use --help for usage." >&2
        exit 2
    fi
    mkdir -p "$repo_dir/generated_data"
    "$repo_dir/.venv/bin/python" - "$script_path" "$log_path" <<'PY'
import subprocess
import sys
from pathlib import Path

script, log = map(Path, sys.argv[1:])
with log.open("w") as output:
    process = subprocess.Popen(
        ["bash", str(script), "--worker"], cwd=script.parents[1],
        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(f"Started PID {process.pid}; log: {log}")
PY
    exit 0
fi

ulimit -v 24000000
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$repo_dir"
.venv/bin/python scripts/faraday_covariance.py kernels
.venv/bin/python scripts/faraday_covariance.py moments
.venv/bin/python scripts/faraday_covariance.py validate
.venv/bin/python scripts/faraday_covariance.py plot
