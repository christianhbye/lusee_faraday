"""Time-averaged Gaussian and scale-mixture signals through the same fit."""

import common  # noqa: F401
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from common import CACHE_DIR, REPO
from faraday_covariance import projector
from mixed_depth_predictions import (
    SOURCES as MIXED_SOURCES,
    coefficients,
    compare,
    frequency_map,
    setup as mixed_setup,
)
from paper_common import digest, load_record, manifest, write_json
from paper_kernels import load as load_kernel
from lusee_faraday.config import N_TIMES, SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.depth_channel_covariance import (
    DepthChannelProjector,
    hermite_degree,
)
from lusee_faraday.depth_depolarization import DepthProfile, DepthQuadrature
from lusee_faraday.observing import sidereal_bin_window
from lusee_faraday.separability import channel_smooth_basis, residual_operator
from lusee_faraday.spin_covariance import AngularScreen

SOURCES = sorted(
    set(
        MIXED_SOURCES
        + [
            "scripts/residual_signal.py",
            "src/lusee_faraday/separability.py",
            "src/lusee_faraday/observing.py",
        ]
    )
)


def common_fit(center, degree):
    fine, w, nominal = frequency_map(center, "channels")
    fits = [
        residual_operator(channel_smooth_basis(fine, part, degree, center))
        for part in (w[:3], w[3:])
    ]
    return fine, w, nominal, block_diag(*(f["operator"] for f in fits)), fits


def setup(config):
    mixed = json.loads((REPO / config["mixed_config"]).read_text())
    base, kr, _, mixed_key, _ = mixed_setup(mixed)
    load_record(REPO / config["mixed_record"], mixed_key)
    complete = dict(
        config,
        mixed_fingerprint=mixed_key,
        bin_seconds=SIDEREAL_DAY_S / N_TIMES,
    )
    provenance, key = manifest(complete, SOURCES)
    return (
        mixed,
        base,
        kr,
        provenance,
        key,
        CACHE_DIR / "residual_signal" / key,
    )


def mixture_moments(solvers, values, probabilities):
    """Stable total-variance split, including global scale mean variation."""
    mean = sum(p * a[:, 0] for p, a in zip(probabilities, values))
    mean_coefficients = np.zeros_like(values[0])
    mean_coefficients[:, 0] = mean
    smooth = solvers[0].moments(mean_coefficients)["smooth"]
    speckle = np.zeros_like(smooth)
    for p, a, solver in zip(probabilities, values, solvers):
        delta = np.zeros_like(a)
        delta[:, 0] = a[:, 0] - mean
        speckle += p * (
            solver.moments(a)["speckle"] + solver.moments(delta)["smooth"]
        )
    return dict(smooth=smooth, speckle=speckle, total=smooth + speckle)


def compute(config, record_path):
    mixed, base, kr, provenance, key, directory = setup(config)
    directory.mkdir(parents=True, exist_ok=True)
    numerical = DepthQuadrature(**mixed["depth_numerical"])
    rows = []
    for center in mixed["channel_centers_mhz"]:
        item = kr["arrays"][f"{center:g}"]
        if digest(REPO / item["path"]) != item["sha256"]:
            raise ValueError("corrupt instrument kernels")
        kernels = load_kernel(REPO / item["path"], kr["fingerprint"])[
            "stokes_kernels"
        ]
        kernels = kernels * sidereal_bin_window(kernels.shape[-2] - 1)
        angular = projector(kernels, base)
        refined = projector(kernels, base, refinement=2)
        fine, w, nominal, r, fits = common_fit(center, config["smooth_degree"])
        for si, case in enumerate(base["screens"]):
            initial = AngularScreen(**case["parameters"])
            for model in ("gaussian", "variance_mixture"):
                scales = (
                    [1.0]
                    if model == "gaussian"
                    else config["variance_mixture_scales"]
                )
                probabilities = (
                    [1.0]
                    if model == "gaussian"
                    else config["variance_mixture_weights"]
                )
                if not np.isclose(sum(probabilities), 1):
                    raise ValueError("mixture probabilities must sum to one")
                screens = [
                    replace(initial, sigma_depth=initial.sigma_depth * s**0.5)
                    for s in scales
                ]
                degree = max(
                    hermite_degree(
                        lambda_squared(fine).max(),
                        s,
                        DepthProfile(),
                        mixed["mode_tail"],
                        mixed["max_degree"],
                    )[0]
                    for s in screens
                )
                solvers = [
                    DepthChannelProjector(angular, s, degree) for s in screens
                ]
                solvers_refined = [
                    DepthChannelProjector(refined, s, degree) for s in screens
                ]
                for pi, profile_parameters in enumerate(mixed["profiles"]):
                    tag = f"{center:g}_{si}_{pi}_{model}"
                    path = directory / (tag + ".npz")
                    if path.exists():
                        check = load_record(path.with_suffix(".json"), key)
                        if digest(path) != check["sha256"]:
                            raise ValueError("corrupt residual-signal cache")
                        rows.append(check["result"])
                        print("verified", tag, flush=True)
                        continue
                    print("computing", tag, flush=True)
                    profile = DepthProfile(**profile_parameters)
                    values, coarse = [], []
                    for s in screens:
                        a, off, *_ = coefficients(
                            center,
                            "channels",
                            s,
                            profile,
                            numerical,
                            degree,
                            base,
                            mixed,
                        )
                        values.append(a)
                        coarse.append(
                            coefficients(
                                center,
                                "channels",
                                s,
                                profile,
                                numerical,
                                degree,
                                base,
                                mixed,
                                stride=2,
                            )[0]
                        )
                    raw = mixture_moments(solvers, values, probabilities)
                    filtered_values = [r @ a for a in values]
                    residual = mixture_moments(
                        solvers, filtered_values, probabilities
                    )
                    high = mixture_moments(
                        solvers_refined, filtered_values, probabilities
                    )
                    low = mixture_moments(
                        solvers, [r @ a for a in coarse], probabilities
                    )
                    off_cov = solvers[0].moments(off)["total"]
                    gates = {
                        "angular": compare(residual, high, off_cov, mixed),
                        "frequency": compare(residual, low, off_cov, mixed),
                    }
                    errors = {
                        name: max(
                            float(abs(residual[name] - x[name]).max())
                            for x in (high, low)
                        )
                        for name in residual
                    }
                    passed = all(
                        v <= 1
                        for group in gates.values()
                        for v in group.values()
                    )
                    np.savez_compressed(
                        path,
                        nominal_mhz=nominal,
                        operator=r,
                        **{"raw_" + n: v for n, v in raw.items()},
                        **{"residual_" + n: v for n, v in residual.items()},
                        conditional_coefficients=np.stack(values),
                        residual_coefficients=np.stack(filtered_values),
                        probabilities=probabilities,
                    )
                    row = dict(
                        center_mhz=center,
                        screen_index=si,
                        profile_index=pi,
                        model=model,
                        passed=passed,
                        gates=gates,
                        maximum_residual_covariance_error_k2=errors,
                        screen=case["parameters"],
                        profile=profile_parameters,
                        fit=[
                            {
                                k: v
                                for k, v in f.items()
                                if k
                                in ("rank", "condition_number", "residual_dof")
                            }
                            for f in fits
                        ],
                        path=str(path.relative_to(REPO)),
                        sha256=digest(path),
                    )
                    write_json(
                        path.with_suffix(".json"),
                        dict(fingerprint=key, sha256=digest(path), result=row),
                    )
                    rows.append(row)
    write_json(
        record_path,
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            all_passed=all(row["passed"] for row in rows),
            scope=(
                "All profiles and local screens, shared physical-frequency "
                "fit, "
                "actual folded channels and finite sidereal averaging. The "
                "non-Gaussian global variance mixture has the same RM "
                "covariance "
                "as its Gaussian reference; its additional mean variation is "
                "retained in the speckle. Numerical errors are exported "
                "beside "
                "the residuals; tiny components can only support upper bounds."
            ),
        ),
    )
    if not all(row["passed"] for row in rows):
        raise SystemExit("Residual-signal gates failed; results retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/paper/separability.json"
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=REPO / "docs/residual-signal-results.json",
    )
    args = parser.parse_args()
    compute(json.loads(args.config.read_text()), args.record)
