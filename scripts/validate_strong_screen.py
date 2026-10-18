"""Strong-column frequency and real-beam scalar-floor controls."""

import common  # noqa: F401
import argparse
import json

import numpy as np

from common import CACHE_DIR, REPO
from faraday_covariance import channel_moments, projector, setup
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.spin_covariance import AngularScreen, helicity_covariance
from lusee_faraday.visibility_covariance import (
    evaluate_mode_moments,
    pair_mode_moments,
)


def compute(quick=False):
    base = json.loads((REPO / "configs/paper/covariance.json").read_text())
    _, kr, _, _, _ = setup(base)
    settings = dict(
        covariance=base,
        sigma_depth=5.0,
        mean_depth=0.01,
        theta0_rad=0.01,
        alpha=5 / 3,
        frequency_nodes=[129, 257],
        relative_tolerance=0.01,
        absolute_off_tolerance=1e-18,
        quick=quick,
    )
    provenance, key = manifest(
        settings,
        [
            "scripts/validate_strong_screen.py",
            "scripts/faraday_covariance.py",
            "scripts/paper_common.py",
            "src/lusee_faraday/spin_covariance.py",
            "src/lusee_faraday/visibility_covariance.py",
            "src/lusee_faraday/observing.py",
            "src/lusee_faraday/frequency_covariance.py",
            "src/lusee_faraday/channelization.py",
        ],
    )
    directory = CACHE_DIR / "strong_screen" / key
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for center in ([30.0] if quick else [30.0, 50.0]):
        path = directory / f"{center:g}.npz"
        if path.exists():
            saved = load_record(path.with_suffix(".json"), key)
            if digest(path) != saved["sha256"]:
                raise ValueError("corrupt strong-screen checkpoint")
            rows.append(saved["result"])
            continue
        print("strong screen", center, flush=True)
        kernel_file = kr["arrays"][f"{center:g}"]
        kernels = load_kernel(REPO / kernel_file["path"], kr["fingerprint"])[
            "stokes_kernels"
        ]
        kernels = kernels * sidereal_bin_window(kernels.shape[-2] - 1)
        angular = projector(kernels, base)
        refined = projector(kernels, base, refinement=2)
        screen = AngularScreen(
            sigma_depth=5.0, mean_depth=0.01, theta0_rad=0.01, alpha=5 / 3
        )
        coarse = channel_moments(angular, center, screen, 129, base)
        high = channel_moments(angular, center, screen, 257, base)
        check = channel_moments(refined, center, screen, 257, base)
        dimension = high.shape[0] * high.shape[1]
        diagonal = np.diag(high.reshape(dimension, dimension))
        on_scale = np.sqrt(
            np.outer(np.maximum(diagonal, 0), np.maximum(diagonal, 0))
        )
        off = angular.screen_covariance(0.0, 0.0, AngularScreen(sigma_depth=0))
        off_diag = np.tile(np.diag(off), high.shape[0])
        allowance = 0.01 * on_scale + 1e-18 * np.sqrt(
            np.outer(off_diag, off_diag)
        )
        gates = {
            name: float(
                np.max(
                    abs(high - alt).reshape(dimension, dimension)
                    / np.maximum(allowance, 1e-300)
                )
            )
            for name, alt in (("frequency", coarse), ("angular", check))
        }
        x = float(lambda_squared(center)[0])
        actual = refined.screen_covariance(x, x, screen)
        theta = refined.theta
        # Low-ell white approximation at the same intrinsic point power.
        t = projector(kernels, base, refinement=2)
        exponent = (
            4
            * screen.sigma_depth**2
            * x
            * x
            * (-np.expm1(-screen.exponent(theta)))
        )
        floor = (
            2
            * np.pi
            * np.trapezoid(
                np.sin(theta) * t.xi["plus"][0] * np.exp(-exponent), theta
            )
        )
        lmax = kernels.shape[-2] - 1
        plus = np.full(lmax + 1, float(floor.real))
        plus[:2] = 0
        h = helicity_covariance(plus, np.zeros_like(plus))
        modes = pair_mode_moments(kernels, h)
        angles = 2 * np.pi * np.asarray(base["sidereal_turns"])
        phases = np.exp(-1j * angles[:, None] * np.arange(-lmax, lmax + 1))
        white = evaluate_mode_moments(modes, phases)[
            "covariance"
        ].real.reshape(actual.shape)
        scale = np.sqrt(np.outer(np.diag(actual), np.diag(actual)))
        white_error = float(
            np.max(abs(actual - white) / np.maximum(scale, 1e-300))
        )
        q = high[3:, 1, 3:, 1]
        corr = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        passed = all(v <= 1 for v in gates.values()) and white_error < 0.01
        np.savez_compressed(
            path,
            covariance=high,
            coarse=coarse,
            angular_check=check,
            native_covariance=actual,
            native_white_approximation=white,
        )
        row = dict(
            center_mhz=center,
            passed=bool(passed),
            gates=gates,
            white_floor_worst_error_over_rms=white_error,
            minimum_central_zoom_q_correlation=float(corr.min()),
            path=str(path.relative_to(REPO)),
            sha256=digest(path),
        )
        rows.append(row)
        write_json(
            path.with_suffix(".json"),
            dict(fingerprint=key, sha256=digest(path), result=row),
        )
        print(row, flush=True)
    write_json(
        REPO / "docs/strong-screen-results.json",
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            scope=(
                "Strong foreground-column control, not a local-screen prior. "
                "Full frequency-pair solver is used when Hermite mode limits "
                "would be excessive. Same frozen loaded instrument and time "
                "average; actual folded central-parent zoom bins. Native "
                "covariance also tests the scalar white-floor normalization "
                "against all real readout kernels and cross terms."
            ),
        ),
    )
    if not all(r["passed"] for r in rows):
        raise SystemExit("Retained strong-screen gate failure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    compute(args.quick)
