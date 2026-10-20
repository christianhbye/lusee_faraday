"""Independent Gaussian integration of a shared-screen channel mixture."""

import common  # noqa: F401
import argparse
from dataclasses import replace

import numpy as np
from scipy.special import roots_hermitenorm

from common import REPO
from mixed_depth_predictions import frequency_map
from paper_common import manifest, write_json
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.depth_channel_covariance import depth_phase_coefficients
from lusee_faraday.depth_depolarization import DepthProfile
from lusee_faraday.frequency_covariance import interpolated_channel_operator
from lusee_faraday.gaussian_screen import real_covariance
from lusee_faraday.spin_covariance import AngularScreen


def analytic(nodes, w, rho, profile, mixture, omit_mean_variation=False):
    source = np.array([0.7 + 0.2j, -0.4 + 0.5j])
    screen = AngularScreen(sigma_depth=0.02, mean_depth=0.01)
    values = []
    for scale in mixture:
        s = replace(screen, sigma_depth=screen.sigma_depth * np.sqrt(scale))
        values.append(
            w
            @ depth_phase_coefficients(lambda_squared(nodes), s, profile)[
                "coefficients"
            ]
        )
    degree = max(a.shape[1] for a in values)
    values = [np.pad(a, ((0, 0), (0, degree - a.shape[1]))) for a in values]
    k = np.arange(1, degree)
    spatial_c = (
        np.sum(abs(source) ** 2)
        + 2 * (source[0] * source[1].conjugate()).real * rho**k
    )
    spatial_j = np.sum(source**2) + 2 * source.prod() * rho**k
    mean = sum(a[:, 0] * source.sum() for a in values) / len(values)
    c, j = (np.zeros((len(w), len(w)), complex) for _ in range(2))
    for a in values:
        c += (a[:, 1:] * spatial_c) @ a[:, 1:].conj().T / len(values)
        j += (a[:, 1:] * spatial_j) @ a[:, 1:].T / len(values)
        delta = a[:, 0] * source.sum() - mean
        if not omit_mean_variation:
            c += np.outer(delta, delta.conj()) / len(values)
            j += np.outer(delta, delta) / len(values)
    return real_covariance(c, j), mean


def independent(nodes, w, rho, profile, order):
    g, weight = roots_hermitenorm(order)
    weight = weight / np.sqrt(2 * np.pi)
    probabilities = (weight[:, None] * weight[None, :]).ravel() / 2
    second = (rho * g[:, None] + np.sqrt(1 - rho * rho) * g[None, :]).ravel()
    first = np.broadcast_to(g[:, None], (order, order)).ravel()
    source = np.array([0.7 + 0.2j, -0.4 + 0.5j])
    x = lambda_squared(nodes)
    moments = []
    for scale in (0.5, 1.5):
        value = np.zeros((order * order, len(w)), complex)
        for position, coefficient in zip((first, second), source):
            phi = 0.01 + 0.02 * np.sqrt(scale) * position
            z = phi[:, None] * x
            phase = (
                np.exp(2j * z)
                if profile.kind == "screen"
                else np.exp(1j * z) * np.sinc(z / np.pi)
            )
            value += coefficient * (phase @ w.T)
        moments.append(value)
    values = np.r_[moments[0], moments[1]]
    weights = np.r_[probabilities, probabilities]
    mean = weights @ values
    centered = values - mean
    real = np.c_[centered.real, centered.imag]
    return (real.T * weights) @ real, mean


def compute(fault=False):
    rows = []
    selected = np.array([0, 1, 2, 3 + 64 + 32, 3 + 64 + 5])
    for center in (30.0, 50.0):
        fine, w, _ = frequency_map(center, "channels")
        nodes = np.linspace(fine[0], fine[-1], 33)
        compressed = interpolated_channel_operator(fine, nodes, w[selected])
        for rho in (0.3, 0.95):
            for profile in (DepthProfile(), DepthProfile(kind="slab")):
                expected, mu = analytic(
                    nodes, compressed, rho, profile, [0.5, 1.5], fault
                )
                measured, measured_mu = independent(
                    nodes, compressed, rho, profile, 128
                )
                coarse, _ = independent(nodes, compressed, rho, profile, 64)
                gaussian, _ = analytic(nodes, compressed, rho, profile, [1.0])
                scale = np.sqrt(np.outer(np.diag(measured), np.diag(measured)))
                error = float(
                    np.max(
                        abs(expected - measured) / np.maximum(scale, 1e-300)
                    )
                )
                refinement = float(
                    np.max(abs(coarse - measured) / np.maximum(scale, 1e-300))
                )
                mean_error = float(abs(mu - measured_mu).max())
                gaussian_mismatch = float(
                    np.max(
                        abs(gaussian - measured) / np.maximum(scale, 1e-300)
                    )
                )
                passed = (
                    error < 2e-6 and refinement < 1e-6 and mean_error < 1e-10
                )
                rows.append(
                    dict(
                        center_mhz=center,
                        rho=rho,
                        profile=profile.kind,
                        maximum_covariance_error_over_rms=error,
                        gaussian_mismatch_over_rms=gaussian_mismatch,
                        quadrature_refinement_over_rms=refinement,
                        maximum_mean_error=mean_error,
                        passed=bool(passed),
                    )
                )
    provenance, key = manifest(
        dict(
            spatial_weights=[[0.7, 0.2], [-0.4, 0.5]],
            sites=2,
            correlations=[0.3, 0.95],
            variance_scales=[0.5, 1.5],
            probabilities=[0.5, 0.5],
            selected_rows=selected.tolist(),
            frequency_nodes=33,
            gaussian_orders=[64, 128],
            covariance_tolerance=2e-6,
            omit_global_mean_variance=fault,
        ),
        [
            "scripts/validate_channel_mixture.py",
            "src/lusee_faraday/depth_channel_covariance.py",
            "src/lusee_faraday/frequency_covariance.py",
            "src/lusee_faraday/channelization.py",
            "src/lusee_faraday/gaussian_screen.py",
            "scripts/paper_common.py",
        ],
    )
    write_json(
        REPO
        / "docs"
        / (
            "channel-mixture-fault.json"
            if fault
            else "channel-mixture-validation.json"
        ),
        dict(
            fingerprint=key,
            provenance=provenance,
            results=rows,
            all_passed=all(r["passed"] for r in rows),
            scope=(
                "Independent bivariate Gaussian quadrature at two "
                "deterministic "
                "complex-polarization source positions. Shared global scale "
                "draw per realization; exact screen/slab phase laws followed "
                "by actual parent and folded zoom channel responses on a "
                "declared 33-node interpolation. Full cross-frequency "
                "covariance "
                "and complex relation enter Q/U. The Gaussian reference with "
                "identical RM covariance is retained as a mismatched control. "
                "This validates channel phase-mixture moments, not "
                "Gaussianity "
                "of the full production visibility sky."
            ),
        ),
    )
    print(rows, flush=True)
    if not all(r["passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omit-global-mean-variance", action="store_true")
    args = parser.parse_args()
    compute(args.omit_global_mean_variance)
