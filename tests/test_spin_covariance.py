"""Independent Wigner sums, phase averages and physical spin-spectrum gates."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from math import factorial  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import eval_legendre, roots_hermitenorm  # noqa: E402

from lusee_faraday.spin_covariance import (  # noqa: E402
    AngularScreen,
    SpinTransform,
    eb_spectra,
    helicity_covariance,
    rotated_spin_spectra,
    screen_phase_moments,
    wigner_spin2,
)


def direct_wigner(ell, m, n, theta):
    """Small-ell factorial sum, independent of Jacobi polynomials."""
    prefactor = np.sqrt(
        float(
            factorial(ell + m)
            * factorial(ell - m)
            * factorial(ell + n)
            * factorial(ell - n)
        )
    )
    value = np.zeros_like(theta)
    for k in range(2 * ell + 1):
        args = (ell + n - k, k, m - n + k, ell - m - k)
        if min(args) < 0:
            continue
        value += (
            prefactor
            * (-1.0) ** (m - n + k)
            * np.cos(theta / 2) ** (2 * ell + n - m - 2 * k)
            * np.sin(theta / 2) ** (m - n + 2 * k)
            / np.prod([float(factorial(v)) for v in args])
        )
    return value


def intrinsic(lmax):
    ell = np.arange(lmax + 1)
    power = np.exp(-ell * (ell + 1) / 40.0)
    power[:2] = 0
    power /= np.sum((2 * ell + 1) * power / (4 * np.pi))
    return eb_spectra(0.8 * power, 0.2 * power, 0.1 * power)


def test_wigner_rows_match_independent_factorial_sum():
    theta = np.array([0.0, 0.21, 0.8, 1.7, np.pi])
    plus, minus = wigner_spin2(theta, 12)
    for ell in range(2, 13):
        np.testing.assert_allclose(
            plus[ell], direct_wigner(ell, 2, 2, theta), atol=2e-12
        )
        np.testing.assert_allclose(
            minus[ell], direct_wigner(ell, 2, -2, theta), atol=2e-12
        )
    np.testing.assert_allclose(plus[2], np.cos(theta / 2) ** 4)
    np.testing.assert_allclose(minus[2], np.sin(theta / 2) ** 4)
    assert np.max(abs(plus[:2])) == 0


def test_spin_roundtrip_including_eb_relation():
    spectra = intrinsic(24)
    transform = SpinTransform.gauss_legendre(24, 64)
    xi = transform.correlations(**spectra)
    result = transform.spectra(**xi)
    for name in ("plus", "minus"):
        np.testing.assert_allclose(result[name], spectra[name], atol=2e-13)
    assert np.max(abs(result["minus"].imag)) > 0.01


def test_uniform_deterministic_rotation_mixes_eb_without_losing_power():
    transform = SpinTransform.gauss_legendre(16, 64)
    base = intrinsic(16)
    screen = AngularScreen(sigma_depth=0, mean_depth=0.03)
    x = 41.0
    result = rotated_spin_spectra(transform, base, x, x, screen)
    np.testing.assert_allclose(
        result["total"]["plus"], base["plus"], atol=2e-13
    )
    np.testing.assert_allclose(
        result["total"]["minus"],
        base["minus"] * np.exp(4j * screen.mean_depth * x),
        atol=2e-13,
    )
    assert np.max(abs(result["speckle"]["plus"])) == 0
    cross = rotated_spin_spectra(transform, base, x, 28.0, screen)
    np.testing.assert_allclose(
        cross["total"]["plus"],
        base["plus"] * np.exp(2j * 0.03 * (x - 28)),
        atol=2e-13,
    )


def test_screen_phase_kernels_match_direct_gaussian_ensemble():
    screen = AngularScreen(sigma_depth=0.012, mean_depth=0.017, theta0_rad=0.3)
    x, y = 28.0, 41.0
    fraction = 0.6
    nodes, weights = roots_hermitenorm(32)
    weights /= np.sqrt(2 * np.pi)
    for theta in (0.0, 0.21, 1.4):
        rho = np.exp(-screen.exponent(theta))
        first = screen.mean_depth + screen.sigma_depth * nodes
        second = screen.mean_depth + screen.sigma_depth * (
            rho * nodes[:, None] + np.sqrt(1 - rho * rho) * nodes[None, :]
        )
        a = np.exp(2j * x * fraction * first[:, None])
        b = np.exp(2j * y * fraction * second)
        w = weights[:, None] * weights[None, :]
        result = screen_phase_moments(x, y, theta, screen, fraction)
        np.testing.assert_allclose(
            result["second"], np.sum(w * a * b.conj()), atol=2e-13
        )
        np.testing.assert_allclose(
            result["nonconjugated_second"], np.sum(w * a * b), atol=2e-13
        )
        np.testing.assert_allclose(
            result["covariance"],
            np.sum(w * (a - result["mean_x"]) * (b - result["mean_y"]).conj()),
            atol=2e-13,
        )


def test_rotated_power_split_conservation_and_spectral_positivity():
    transform = SpinTransform.gauss_legendre(64, 256)
    base = intrinsic(16)
    result = rotated_spin_spectra(
        transform,
        base,
        100.0,
        100.0,
        AngularScreen(sigma_depth=0.005, theta0_rad=0.5),
    )
    ell = np.arange(65)
    np.testing.assert_allclose(
        np.sum((2 * ell + 1) * result["total"]["plus"] / (4 * np.pi)),
        1.0,
        atol=2e-11,
    )
    for name in ("plus", "minus"):
        np.testing.assert_allclose(
            result["total"][name],
            result["smooth"][name] + result["speckle"][name],
            atol=2e-13,
        )
    for part in result.values():
        values = np.linalg.eigvalsh(helicity_covariance(**part))
        assert values.min() > -2e-12


def test_strong_screen_recovers_scalar_low_ell_floor_and_coherence():
    theta = np.r_[
        0.0,
        np.geomspace(1e-10, 0.01, 2401),
        np.linspace(0.01, np.pi, 4001)[1:],
    ]
    transform = SpinTransform.theta_grid(theta, 96)
    screen = AngularScreen(
        sigma_depth=5.0, mean_depth=0, theta0_rad=0.1, alpha=5 / 3
    )
    base = intrinsic(16)
    phase = screen_phase_moments(100.0, 100.0, theta, screen)
    xi = transform.correlations(**base)
    spectra = rotated_spin_spectra(transform, base, 100.0, 100.0, screen)
    for ell in (20, 64, 96):
        scalar = (
            2
            * np.pi
            * np.sum(
                transform.weights
                * xi["plus"]
                * phase["second"]
                * eval_legendre(ell, np.cos(theta))
            )
        )
        np.testing.assert_allclose(
            spectra["total"]["plus"][ell], scalar, rtol=2e-6
        )
    angle = screen.coherence_angle(30.0)[0]
    from lusee_faraday.conventions import lambda_squared

    x = float(lambda_squared(30)[0])
    structure = (
        2 * screen.sigma_depth**2 * (-np.expm1(-screen.exponent(angle)))
    )
    np.testing.assert_allclose(2 * x * x * structure, 1.0)
    assert np.isinf(screen.coherence_angle(30.0, fraction=0)[0])


def test_invalid_spin_inputs_are_rejected():
    with pytest.raises(ValueError):
        AngularScreen(alpha=2.1)
    with pytest.raises(ValueError):
        wigner_spin2([-1.0], 8)
    with pytest.raises(ValueError):
        SpinTransform.gauss_legendre(12, 5)
    with pytest.raises(ValueError):
        eb_spectra([0.0, 0.0, 1.0], [0.0, 0.0, 0.1], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        SpinTransform.gauss_legendre(8).correlations(np.ones(9), np.ones(9))
    with pytest.raises(ValueError):
        AngularScreen().coherence_angle(30.0, fraction=-1)
