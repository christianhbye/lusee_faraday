#!/usr/bin/env bash
# Column-anchored screen through the real-instrument covariance stage.
# Separate config/record so the local-scenario fingerprints are untouched.
set -euo pipefail
repo_dir=$(dirname "$(dirname "$(realpath "$0")")")
cd "$repo_dir"
ulimit -v 24000000
export JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
cfg=configs/paper/covariance-column.json
rec=docs/faraday-covariance-column-results.json
.venv/bin/python scripts/faraday_covariance.py kernels --config $cfg --record $rec
.venv/bin/python scripts/faraday_covariance.py moments --config $cfg --record $rec
.venv/bin/python scripts/faraday_covariance.py validate --config $cfg --record $rec
echo COLUMN_RUN_DONE
