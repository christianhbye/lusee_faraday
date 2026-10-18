#!/usr/bin/env bash
# Checkpointed complete pipeline, with isolated logs and bounded memory.
set -euo pipefail

script_path=$(realpath "$0")
repo_dir=$(dirname "$(dirname "$script_path")")
log_path="$repo_dir/generated_data/paper_implementation.log"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_paper_implementation.sh [--smoke | --from STAGE]"
    echo "Launch the checkpointed complete calculation and report."
    echo "--smoke runs only the inexpensive data-free backend checks."
    echo "Stages: horizon depth kernels foreground mixed exposure leakage signal controls system local science observations figures report"
    echo "Large response/map inputs and the existing editable environment are required."
    echo "Log: $log_path"
    exit 0
fi

if [[ "${1:-}" != "--worker" ]]; then
    mode=full
    first_stage=horizon
    if [[ "${1:-}" == "--smoke" && $# -eq 1 ]]; then
        mode=smoke
    elif [[ "${1:-}" == "--from" && $# -eq 2 ]]; then
        first_stage=$2
        case "$first_stage" in
            horizon|depth|kernels|foreground|mixed|exposure|leakage|signal|controls|system|local|science|observations|figures|report) ;;
            *) echo "Unknown stage; use --help." >&2; exit 2 ;;
        esac
    elif [[ $# -ne 0 ]]; then
        echo "Use --help for usage." >&2
        exit 2
    fi
    mkdir -p "$repo_dir/generated_data"
    "$repo_dir/.venv/bin/python" - "$script_path" "$log_path" "$mode" "$first_stage" <<'PY'
import subprocess
import sys
from pathlib import Path

script, log = map(Path, sys.argv[1:3])
with log.open("a") as output:
    process = subprocess.Popen(
        ["bash", str(script), "--worker", sys.argv[3], sys.argv[4]],
        cwd=script.parents[1],
        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(f"Started PID {process.pid}; log: {log}")
PY
    exit 0
fi

exec 9>"$repo_dir/generated_data/paper_implementation.lock"
flock --nonblock 9 || { echo "The pipeline is already running." >&2; exit 1; }
ulimit -v 24000000
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$repo_dir"

if [[ "${2:-full}" == "smoke" ]]; then
    ulimit -v 16000000
    .venv/bin/python -m pytest -q -o addopts= \
        tests/test_depth_channel_covariance.py tests/test_receiver_noise.py \
        tests/test_separability.py tests/test_observing.py \
        tests/test_measurement_statistics.py tests/test_global_phase_mixture.py \
        tests/test_visibility_draws.py tests/test_moment_diagonal.py
    exit 0
fi

started=0
run_stage() {
    stage_name=$1
    shift
    if [[ "$stage_name" == "$first_stage" ]]; then started=1; fi
    if [[ $started -eq 1 ]]; then "$@"; fi
}
first_stage=${3:-horizon}
run_stage horizon .venv/bin/python scripts/faraday_horizon.py
run_stage horizon .venv/bin/python scripts/local_input_audit.py
run_stage depth .venv/bin/python scripts/depth_depolarization.py compute
run_stage depth .venv/bin/python scripts/depth_depolarization.py plot
run_stage kernels .venv/bin/python scripts/faraday_covariance.py kernels
run_stage foreground .venv/bin/python scripts/faraday_covariance.py moments
run_stage foreground .venv/bin/python scripts/faraday_covariance.py validate
run_stage mixed .venv/bin/python scripts/mixed_depth_predictions.py
run_stage mixed .venv/bin/python scripts/plot_mixed_depth.py
run_stage exposure .venv/bin/python scripts/observing_exposure.py
run_stage leakage .venv/bin/python scripts/chromatic_leakage.py
run_stage signal .venv/bin/python scripts/residual_signal.py
run_stage signal .venv/bin/python scripts/validate_global_phase_mixture_faults.py
run_stage signal .venv/bin/python scripts/validate_residual_signal.py
run_stage controls .venv/bin/python scripts/validate_native_clock_reference.py
run_stage controls .venv/bin/python scripts/validate_matched_geometry.py
run_stage controls .venv/bin/python scripts/validate_channel_mixture.py
run_stage controls .venv/bin/python scripts/validate_strong_screen.py
run_stage controls .venv/bin/python scripts/common_observing_control.py compute
run_stage controls .venv/bin/python scripts/common_observing_control.py plot
run_stage system .venv/bin/python scripts/measured_system.py
run_stage local .venv/bin/python scripts/local_medium_sensitivity.py compute
run_stage local .venv/bin/python scripts/local_medium_separability.py
run_stage local .venv/bin/python scripts/validate_local_medium_joint.py
run_stage science .venv/bin/python scripts/science_product_predictions.py
run_stage observations .venv/bin/python scripts/faraday_separability.py
run_stage figures .venv/bin/python scripts/plot_measurement_predictions.py
run_stage figures .venv/bin/python scripts/plot_local_medium_predictions.py
run_stage figures .venv/bin/python scripts/common_observing_control.py plot
run_stage report .venv/bin/python scripts/paper_products.py --verify-only
run_stage report .venv/bin/python scripts/build_paper_report.py
