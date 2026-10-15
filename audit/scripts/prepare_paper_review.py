"""Snapshot existing evidence for paper preparation; never launch a sky job."""

from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "audit/scripts/shuffle_control_bgl.py"
spec = importlib.util.spec_from_file_location("bgl_audit_snapshot", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def artifact(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": audit.sha256(path),
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def main():
    legacy_path = ROOT / "audit/generated/shuffle_bgl_results.json"
    legacy = json.loads(legacy_path.read_text())
    runs = list(legacy["runs"].values())
    detection_path = ROOT / "generated_data/step5_detection.npz"
    with np.load(detection_path) as d:
        safe = np.flatnonzero(d["cuts"] == 27.5).item()
        detection = {
            **artifact(detection_path),
            "lunations": int(d["lunations"]),
            "absolute_depth_cut_rad_m2": float(d["cuts"][safe]),
            "meaning": "amplitude divided by five-sigma amplitude threshold",
            "rows": [
                {
                    "band_mhz": float(band),
                    "slab_threshold_ratio": float(d["ratio_slab"][ib, safe]),
                    "dispersion_threshold_ratio": float(
                        d["ratio_dispersion"][ib, safe]
                    ),
                    "retained_power_fraction": float(
                        d["power_fraction"][ib, safe]
                    ),
                }
                for ib, band in enumerate(d["bands"])
            ],
        }
    new_ledger = (
        json.loads(audit.RESULT_PATH.read_text())
        if audit.RESULT_PATH.exists()
        else {}
    )
    validation_logs = [
        ROOT / "generated_data" / name
        for name in (
            "paper_preparation_tests.log",
            "paper_preparation_mutation_branch.log",
            "paper_preparation_mutation_count.log",
        )
    ]
    content = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": git("rev-parse", "HEAD"),
        "template_branch_head": git("rev-parse", "faraday-delay-template"),
        "template_report_source": {
            "ref": "faraday-delay-template",
            "path": "report/faraday_depth_template/faraday_depth_template.tex",
        },
        "legacy_audit": {
            **artifact(legacy_path),
            "metadata": legacy["metadata"],
            "counts_by_nside": dict(Counter(r["nside"] for r in runs)),
            "missing_nsides": sorted(
                set(audit.NSIDES) - {r["nside"] for r in runs}
            ),
            "status": "legacy only; branch exponent and N_eff need correction",
            "negative_branch_issue": "both branches used positive exponent",
            "n_eff_issue": "complex cancellation included in effective count",
        },
        "saved_detection": detection,
        "current_audit_provenance": audit.fingerprint(),
        "execution_status": {
            "snapshot_regenerates_forecast": False,
            "new_audit_ledger_exists": audit.RESULT_PATH.exists(),
            "new_audit_execution_count": len(new_ledger.get("executions", [])),
        },
        "preparation_verification": {
            "logs": [artifact(p) for p in validation_logs if p.exists()],
            "test_source_sha256": audit.sha256(
                ROOT / "tests/test_shuffle_control_bgl.py"
            ),
            "launcher_sha256": audit.sha256(ROOT / "audit/run_bgl_audit.sh"),
        },
    }
    path = ROOT / "docs/paper-preparation-status.json"
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
