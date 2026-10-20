"""Validate all stable residuals against direct subtraction and PSD gates."""

import common  # noqa: F401
import json

import numpy as np

from common import REPO
from paper_common import digest, load_record, write_json
from residual_signal import setup
from lusee_faraday.separability import residual_covariance


def compute():
    config = json.loads((REPO / "configs/paper/separability.json").read_text())
    *_, key, directory = setup(config)
    record = load_record(REPO / "docs/residual-signal-results.json", key)
    rows = []
    for row in record["results"]:
        path = REPO / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("corrupt residual-signal artifact")
        checks = []
        with np.load(path) as data:
            r = data["operator"]
            for component in ("smooth", "speckle", "total"):
                raw, filtered = (
                    data["raw_" + component],
                    data["residual_" + component],
                )
                direct = residual_covariance(raw, r)
                error = float(abs(direct - filtered).max())
                raw_scale = max(float(abs(raw).max()), 1e-300)
                residual_scale = max(float(abs(filtered).max()), 1e-300)
                n = filtered.shape[0] * filtered.shape[1]
                eigen = np.linalg.eigvalsh(filtered.reshape(n, n))
                minimum = float(eigen[0])
                passed = (
                    minimum >= -1e-8 * residual_scale
                    and error <= 1e-10 * raw_scale + 1e-6 * residual_scale
                )
                checks.append(
                    dict(
                        component=component,
                        minimum_eigenvalue_k2=minimum,
                        psd_tolerance_k2=1e-8 * residual_scale,
                        direct_subtraction_error_k2=error,
                        passed=bool(passed),
                    )
                )
        rows.append(
            dict(
                path=row["path"],
                checks=checks,
                passed=all(c["passed"] for c in checks),
            )
        )
        print(path.name, rows[-1]["passed"], flush=True)
    write_json(
        REPO / "docs/residual-signal-validation.json",
        dict(
            fingerprint=key,
            input_record_sha256=digest(
                REPO / "docs/residual-signal-results.json"
            ),
            validator_sha256=digest(__file__),
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            finite_ensemble_test=json.loads(
                (
                    REPO
                    / "generated_data/global_phase_mixture_fault_checks.json"
                ).read_text()
            ),
            scope=(
                "Stable pre-covariance filtering compared with R C R^T; full "
                "residual PSD checked separately for smooth, speckle and "
                "total. "
                "The per-case numerical error remains exported and must be "
                "compared with small residual power before interpretation."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit("Retained residual covariance validation failures")


if __name__ == "__main__":
    compute()
