"""Matched deterministic spin analyzer: spherical window versus PR 7 plane."""

import common  # noqa: F401
import argparse
from dataclasses import asdict, replace

import numpy as np
from scipy.special import roots_legendre

from common import REPO
from paper_common import manifest, write_json
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.gaussian_screen import (
    ScreenModel,
    qu_moments,
    real_covariance,
)


def spherical_window(model, order=256, angular_nodes=6000, lmax=768):
    sigma = model.weight_sigma_pc / model.distance_pc
    nodes, weights = roots_legendre(order)
    theta = (nodes + 1) * 4 * sigma
    weight = (
        weights
        * 4
        * sigma
        * np.sin(theta)
        * np.exp(-(theta**2) / (2 * sigma**2))
    )
    weight /= weight.sum()
    previous, current = np.ones_like(theta), np.cos(theta)
    fl = [weight @ previous, weight @ current]
    for mode in range(1, lmax):
        following = (
            (2 * mode + 1) * np.cos(theta) * current - mode * previous
        ) / (mode + 1)
        fl.append(weight @ following)
        previous, current = current, following
    fl = np.asarray(fl)
    separation = np.r_[0, np.geomspace(1e-10, 16 * sigma, angular_nodes)]
    previous, current = np.ones_like(separation), np.cos(separation)
    pair = previous * fl[0] ** 2 + 3 * current * fl[1] ** 2
    for mode in range(1, lmax):
        following = (
            (2 * mode + 1) * np.cos(separation) * current - mode * previous
        ) / (mode + 1)
        pair += (2 * mode + 3) * fl[mode + 1] ** 2 * following
        previous, current = current, following
    density = np.sin(separation) * pair / 2
    return separation, density


def spherical_moments(freq, model, quadrature):
    theta, density = quadrature
    x = lambda_squared(freq)
    # Chordal Gaussian is positive definite on the sphere and has the
    # same tangent-plane correlation length as the PR 7 Gaussian screen.
    theta0 = model.correlation_pc / model.distance_pc
    z = 2 * np.sin(theta / 2) ** 2 / theta0**2
    rho = np.exp(-z)
    loss = -np.expm1(-z)
    c, j = (np.zeros((len(x), len(x)), complex) for _ in range(2))
    amplitude = (
        model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    )
    intrinsic = amplitude * np.exp(2j * np.deg2rad(model.angle_deg))
    for i, u in enumerate(x):
        for k, v in enumerate(x):
            a = 4 * model.sigma_depth**2 * u * v
            conjugate = np.exp(
                -2 * model.sigma_depth**2 * (u - v) ** 2 - a * loss
            ) * -np.expm1(-a * rho)
            relation = np.exp(
                -2 * model.sigma_depth**2 * (u * u + v * v)
            ) * np.expm1(-a * rho)
            c[i, k] = (
                np.trapezoid(density * conjugate, theta)
                * np.exp(2j * model.mean_depth * (u - v))
                * intrinsic[i]
                * intrinsic[k].conjugate()
            )
            j[i, k] = (
                np.trapezoid(density * relation, theta)
                * np.exp(2j * model.mean_depth * (u + v))
                * intrinsic[i]
                * intrinsic[k]
            )
    return real_covariance(c, j)


def compute(fault=False):
    freq = np.array([20, 27.5, 35, 45, 47.5, 50.0])
    rows = []
    for width_scale in (1.0, 0.25):
        model = replace(
            ScreenModel(),
            beam_sigma_deg=width_scale,
            source_sigma_deg=1.5 * width_scale,
        )
        low_rule = spherical_window(model, lmax=1536)
        high_rule = spherical_window(
            model, order=512, angular_nodes=12000, lmax=1536
        )
        high = spherical_moments(freq, model, high_rule)
        low = spherical_moments(freq, model, low_rule)
        flat = qu_moments(freq, model)["covariance"]
        if fault:
            high = high / model.eta**2
        scale = np.sqrt(np.outer(np.diag(flat), np.diag(flat)))
        difference = float(np.max(abs(high - flat) / scale))
        refinement = float(np.max(abs(high - low) / scale))
        window_mass = float(np.trapezoid(high_rule[1], high_rule[0]))
        passed = bool(
            difference < 0.01
            and refinement < 0.001
            and abs(window_mass - 1) < 2e-6
            and high_rule[1].min() >= -1e-10 * high_rule[1].max()
        )
        rows.append(
            dict(
                model=asdict(model),
                worst_covariance_error_over_flat_rms=difference,
                worst_refinement_error_over_flat_rms=refinement,
                window_probability_mass=window_mass,
                minimum_pair_probability_density=float(high_rule[1].min()),
                passed=passed,
            )
        )
    provenance, key = manifest(
        dict(
            frequencies_mhz=freq.tolist(),
            ell_max=1536,
            low=[256, 6000],
            high=[512, 12000],
            tolerance=0.01,
            deliberate_normalization_fault=fault,
        ),
        [
            "scripts/validate_matched_geometry.py",
            "src/lusee_faraday/gaussian_screen.py",
            "src/lusee_faraday/conventions.py",
            "scripts/paper_common.py",
        ],
    )
    record = dict(
        fingerprint=key,
        provenance=provenance,
        results=rows,
        all_passed=all(r["passed"] for r in rows),
        scope=(
            "Deterministic Gaussian source and analyzer both parallel "
            "transported to the window center, cancelling their intrinsic "
            "spin phases. Exact spherical separation distribution from "
            "Legendre window convolution; no isotropic random intrinsic "
            "E/B ensemble substitution. Flat geometry is the PR 7/8 "
            "benchmark, including dilution and both complex moments. "
            "Angular widths shrink while physical screen statistics stay "
            "fixed. This validates the matched narrow-window limit, not "
            "visibility Gaussianity for the whole production sky."
        ),
    )
    path = (
        REPO
        / "docs"
        / (
            "matched-geometry-fault.json"
            if fault
            else "matched-geometry-results.json"
        )
    )
    write_json(path, record)
    print(rows, flush=True)
    if not record["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject-normalization-fault", action="store_true")
    args = parser.parse_args()
    compute(args.inject_normalization_fault)
