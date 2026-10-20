"""Check saved noise covariances, common fits and component accounting."""

import common  # noqa: F401
import argparse
import json

import numpy as np

from common import REPO
from paper_common import digest, write_json
from lusee_faraday.receiver_noise import check_spectral_covariance
from lusee_faraday.separability import residual_covariance


def compute(fault=False):
    record_path = REPO / "docs/measured-system-results.json"
    record = json.loads(record_path.read_text())
    references = {}
    for row in record["leakage"]:
        if row["response"] == "asbuilt":
            with np.load(REPO / row["path"]) as data:
                references[row["center_mhz"]] = {
                    k: data[k] for k in data.files
                }
    rows = []
    for row in record["noise"]:
        path = REPO / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("corrupt noise checkpoint")
        reference = references[row["center_mhz"]]
        r = reference["operator"]
        diagnostics = []
        with np.load(path) as data:
            thermal, amp = data["thermal_psd"], data["amplifier_psd"]
            check_spectral_covariance(thermal)
            check_spectral_covariance(amp)
            fine = reference["fine_sky_covariance"]
            expected = fine[len(fine) // 2] + thermal + amp
            system = data["system_center_psd"]
            component_error = float(
                abs(expected - system).max() / abs(system).max()
            )
            for phase in range(len(data["covariance_per_second"])):
                raw = data["covariance_per_second"][phase]
                filtered = data["residual_covariance_per_second"][phase]
                direct = (
                    np.einsum("bi,ipjq->bpjq", r, raw)
                    if fault
                    else residual_covariance(raw, r)
                )
                scale = max(float(abs(raw).max()), 1e-300)
                projection_error = float(abs(filtered - direct).max() / scale)
                n = raw.shape[0] * raw.shape[1]
                minimum_raw = float(np.linalg.eigvalsh(raw.reshape(n, n))[0])
                minimum_res = float(
                    np.linalg.eigvalsh(filtered.reshape(n, n))[0]
                )
                passed = (
                    projection_error <= 1e-9
                    and minimum_raw >= -1e-8 * scale
                    and minimum_res >= -1e-8 * scale
                    and component_error <= 1e-12
                )
                diagnostics.append(
                    dict(
                        phase_index=phase,
                        projection_error_over_raw_scale=projection_error,
                        minimum_raw_eigenvalue_k2_s=minimum_raw,
                        minimum_residual_eigenvalue_k2_s=minimum_res,
                        passed=bool(passed),
                    )
                )
        rows.append(
            dict(
                path=row["path"],
                diagnostics=diagnostics,
                component_accounting_relative_error=component_error,
                passed=all(d["passed"] for d in diagnostics),
            )
        )
        print(path.name, rows[-1]["passed"], flush=True)
    write_json(
        REPO
        / "docs"
        / (
            "measured-system-fault.json"
            if fault
            else "measured-system-validation.json"
        ),
        dict(
            input_record_sha256=digest(record_path),
            validator_sha256=digest(__file__),
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            injected_single_covariance_leg=fault,
            covariance_coefficient_units=(
                "K^2 s; divide by exposure_s to get K^2"
            ),
            voltage_spectral_covariance_units=(
                "one-sided V^2/Hz at loaded receiver input"
            ),
            scope=(
                "All nine voltage/current cases, both bands and both phases. "
                "Full raw and residual joint covariance positivity; "
                "independent R C R^T comparison to filtering coefficients; "
                "sky, Moon/loss "
                "and amplifier power added exactly once at the stored center."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject-single-leg", action="store_true")
    args = parser.parse_args()
    compute(args.inject_single_leg)
