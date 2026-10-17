#!/usr/bin/env bash
# Reproduce the bounded benchmark. Completed screen/fit chunks are reused only
# when their code/configuration fingerprints match. No external paper edits.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TASK_OUT="$TASK_ROOT/generated_data/gaussian_screen_inference"
case "${1:-help}" in
  run)
    mkdir -p "$TASK_OUT"
    if [[ -f "$TASK_OUT/launcher.pid" ]] && kill -0 "$(cat "$TASK_OUT/launcher.pid")" 2>/dev/null; then
      echo "A benchmark launcher is already running; use status." >&2
      exit 1
    fi
    nohup bash "$TASK_ROOT/scripts/run_gaussian_screen_inference.sh" _work > "$TASK_OUT/launcher.log" 2>&1 < /dev/null &
    task_pid=$!
    echo "$task_pid" > "$TASK_OUT/launcher.pid"
    echo "Started PID $task_pid; log: $TASK_OUT/launcher.log"
    wait "$task_pid"
    ;;
  status)
    if [[ -f "$TASK_OUT/launcher.pid" ]]; then
      ps -p "$(cat "$TASK_OUT/launcher.pid")" -o pid,etime,rss,args || true
    fi
    if [[ -f "$TASK_OUT/launcher.exit" ]]; then
      echo "Last launcher exit: $(cat "$TASK_OUT/launcher.exit")"
    fi
    if [[ -f "$TASK_OUT/launcher.log" ]]; then tail -n 12 "$TASK_OUT/launcher.log"; fi
    ;;
  _work)
    cd "$TASK_ROOT"
    ulimit -v 16000000
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=1
    trap 'echo "$?" > "$TASK_OUT/launcher.exit"' EXIT
    .venv/bin/python scripts/validate_gaussian_screen.py --ell .05 .15 .5 1.5 5 --grid-refinement 1 1 2 2 1 --samples 128 --freq 45 46.25 47.5 48.75 50 --label covariance-validation
    .venv/bin/python scripts/validate_gaussian_screen.py --ell .5 --samples 128 --seed-start 40000 --freq 20 25 30 35 --label strong-validation
    .venv/bin/python scripts/gaussian_screen_experiment.py assess
    .venv/bin/python scripts/gaussian_screen_experiment.py freeze
    .venv/bin/python scripts/validate_gaussian_screen.py --ell .5 1.5 --grid-refinement 2 --samples 1024 --seed-start 60000 --freq 45 46.25 47.5 48.75 50 --label final-validation
    .venv/bin/python scripts/validate_gaussian_screen.py --ell .5 --grid-refinement 2 --samples 1024 --seed-start 80000 --freq 45 46.25 47.5 48.75 50 --mixture --label mixture-validation
    .venv/bin/python scripts/calibrate_gaussian_screen.py
    .venv/bin/python scripts/gaussian_screen_inference.py
    ;;
  help|--help|-h)
    echo "Usage: bash scripts/run_gaussian_screen_inference.sh {run|status}"
    echo "run: capped background benchmark, checkpointed logs; waits for completion"
    echo "status: read-only launcher status and log tail"
    ;;
  *) echo "Unknown action; use --help" >&2; exit 2 ;;
esac
