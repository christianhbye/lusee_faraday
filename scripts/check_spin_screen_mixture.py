"""Direct phase-average control for matched-covariance non-Gaussian screens.

This checks a bivariate screen marginal, not Gaussianity of beam-averaged
visibilities. The same global scale mixture was used by the PR 8 benchmark.
"""

import common  # noqa: F401
import json
from dataclasses import replace

import numpy as np
from scipy.special import roots_hermitenorm

from common import REPO
from paper_common import manifest, write_json
from lusee_faraday.conventions import lambda_squared
from lusee_faraday.spin_covariance import AngularScreen, screen_phase_moments


def direct_moments(x, theta, screen, scales, probabilities, order):
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2 * np.pi)
    rho = np.exp(-screen.exponent(theta))
    first_unit = nodes[:, None]
    second_unit = rho * first_unit + np.sqrt(1 - rho * rho) * nodes[None, :]
    w = weights[:, None] * weights[None, :]
    mean = np.zeros(len(x), complex)
    c, j = (np.zeros((len(x),) * 2, complex) for _ in range(2))
    variance, spatial = 0.0, 0.0
    for scale, probability in zip(scales, probabilities):
        a = screen.mean_depth + screen.sigma_depth * scale * first_unit
        b = screen.mean_depth + screen.sigma_depth * scale * second_unit
        phase_a = np.exp(2j * x[:, None, None] * a)
        phase_b = np.exp(2j * x[:, None, None] * b)
        # a is constant on the second Gaussian coordinate.
        mean += probability * (phase_a[:, :, 0] @ weights)
        # Cross moments must retain the shared first Gaussian coordinate.
        for i in range(len(x)):
            c[i] += probability * np.sum(
                w * phase_a[i] * phase_b.conj(), axis=(1, 2)
            )
            j[i] += probability * np.sum(w * phase_a[i] * phase_b, axis=(1, 2))
        variance += probability * np.sum(w * (a - screen.mean_depth) ** 2)
        spatial += probability * np.sum(
            w * (a - screen.mean_depth) * (b - screen.mean_depth)
        )
    return dict(
        mean=mean,
        second=c,
        nonconjugated_second=j,
        rm_variance=float(variance),
        rm_cross_covariance=float(spatial),
    )


def analytic_mixture(x, theta, screen, scales, probabilities):
    result = dict(
        mean=np.zeros(len(x), complex),
        second=np.zeros((len(x),) * 2, complex),
        nonconjugated_second=np.zeros((len(x),) * 2, complex),
    )
    for scale, probability in zip(scales, probabilities):
        component = replace(screen, sigma_depth=screen.sigma_depth * scale)
        phase = screen_phase_moments(x[:, None], x[None, :], theta, component)
        result["mean"] += probability * phase["mean_x"][:, 0]
        for key in ("second", "nonconjugated_second"):
            result[key] += probability * phase[key][..., 0]
    return result


def complex_record(value):
    return dict(real=value.real.tolist(), imag=value.imag.tolist())


def main():
    config = json.loads(
        (REPO / "configs/paper/screen-mixture.json").read_text()
    )
    screen = AngularScreen(**config["screen"])
    scales = np.sqrt(config["variance_scales"])
    probabilities = np.asarray(config["probabilities"])
    if (
        np.any(probabilities < 0)
        or np.any(scales < 0)
        or abs(probabilities.sum() - 1) > 1e-14
        or abs(probabilities @ scales**2 - 1) > 1e-14
    ):
        raise ValueError("mixture must preserve the Gaussian RM covariance")
    sources = [
        "scripts/common.py",
        "scripts/paper_common.py",
        "scripts/check_spin_screen_mixture.py",
        "src/lusee_faraday/spin_covariance.py",
        "src/lusee_faraday/conventions.py",
    ]
    provenance, key = manifest(config, sources)
    x = lambda_squared(config["frequencies_mhz"])
    rows = []
    for theta in config["separations_rad"]:
        direct = direct_moments(
            x, theta, screen, scales, probabilities, config["quadrature_order"]
        )
        refined = direct_moments(
            x,
            theta,
            screen,
            scales,
            probabilities,
            2 * config["quadrature_order"],
        )
        mixture = analytic_mixture(x, theta, screen, scales, probabilities)
        gaussian = analytic_mixture(x, theta, screen, [1.0], [1.0])
        errors, refinement, misspecification = {}, {}, {}
        for name in ("mean", "second", "nonconjugated_second"):
            errors[name] = float(np.max(abs(direct[name] - mixture[name])))
            refinement[name] = float(np.max(abs(direct[name] - refined[name])))
            misspecification[name] = float(
                np.max(abs(direct[name] - gaussian[name]))
            )
        expected_spatial = screen.sigma_depth**2 * np.exp(
            -screen.exponent(theta)
        )
        rm_error = max(
            abs(direct["rm_variance"] - screen.sigma_depth**2),
            abs(direct["rm_cross_covariance"] - expected_spatial),
        )
        passed = (
            max(errors.values()) < config["absolute_tolerance"]
            and max(refinement.values()) < config["absolute_tolerance"]
            and rm_error < 1e-13 * screen.sigma_depth**2
        )
        rows.append(
            dict(
                separation_rad=theta,
                passed=passed,
                rm_covariance_max_error=rm_error,
                direct_vs_mixture=errors,
                quadrature_refinement=refinement,
                direct_vs_gaussian=misspecification,
                mixture={
                    name: complex_record(value)
                    for name, value in mixture.items()
                },
            )
        )
        print(theta, "PASS" if passed else "FAIL", errors, flush=True)
    record = dict(
        fingerprint=key,
        provenance=provenance,
        results=rows,
        all_passed=all(r["passed"] for r in rows),
        scope="Independent direct Gaussian quadrature of bivariate phase "
        "marginals for a global scale mixture with matched RM covariance. "
        "Tests both frequency-moment types; does not validate full "
        "spherical realizations or visibility Gaussianity.",
    )
    write_json(REPO / "docs/spin-screen-mixture-results.json", record)
    if not record["all_passed"]:
        raise SystemExit("Retained failed mixture checks.")


if __name__ == "__main__":
    main()
