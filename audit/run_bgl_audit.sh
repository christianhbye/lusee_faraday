#!/usr/bin/env bash
# The default is a read-only plan. smoke/full are explicit future run commands.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
audit_python="$repo_dir/.venv/bin/python"
audit_script="$repo_dir/audit/scripts/shuffle_control_bgl.py"
audit_mode=${1:-plan}
audit_args=(--chunk 100000 --nufft-chunk 1000000 --threads 2 --save-spectra)
case "$audit_mode" in
  plan)
    exec "$audit_python" "$audit_script" --plan "${audit_args[@]}"
    ;;
  smoke)
    audit_args+=(--nsides 512)
    ;;
  full)
    audit_args+=(--nsides 512 1024 2048)
    ;;
  *)
    echo 'Usage: bash audit/run_bgl_audit.sh [plan|smoke|full]' >&2
    exit 2
    ;;
esac
if [[ -n ${LUSEE_RESPONSE:-} ]]; then
  echo 'This launcher uses the as-built response. Unset LUSEE_RESPONSE first.' >&2
  exit 2
fi
mkdir -p "$repo_dir/generated_data"
audit_unit="lusee-bgl-v2-$audit_mode"
audit_log="$repo_dir/generated_data/shuffle_bgl_v2_$audit_mode.log"
# A service survives terminal disconnects. 8 GiB protects this 15-GiB host;
# the 16,000,000-KiB virtual-memory limit is separate from resident memory.
# MemoryHigh encourages reclamation of disk-backed weight pages.
systemd-run --user --unit="$audit_unit" \
  --description="LuSEE BGL v2 $audit_mode audit" \
  --property="WorkingDirectory=$repo_dir" \
  --property=MemoryHigh=6G --property=MemoryMax=8G \
  --property=MemorySwapMax=0 \
  --property="StandardOutput=append:$audit_log" \
  --property="StandardError=append:$audit_log" \
  --setenv=JAX_ENABLE_X64=1 --setenv=OMP_NUM_THREADS=2 \
  --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
  /bin/bash -c 'ulimit -v 16000000; exec /usr/bin/time -v "$@"' bash \
  "$audit_python" -u "$audit_script" "${audit_args[@]}"
echo "Log: $audit_log"
echo "Status: systemctl --user status $audit_unit.service"
echo "Outcome: systemctl --user show $audit_unit.service -p Result -p ExecMainStatus -p MemoryPeak"
echo "Journal: journalctl --user -u $audit_unit.service --no-pager"
