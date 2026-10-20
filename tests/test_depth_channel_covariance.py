"""Mixed-depth moments checked against direct Gaussian and depth integrals."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import roots_hermitenorm  # noqa: E402
from scipy.stats import poisson  # noqa: E402

from lusee_faraday.depth_channel_covariance import (  # noqa: E402
    DepthChannelProjector,
    channel_phase_coefficients,
    depth_phase_coefficients,
    hermite_degree,
    phase_moments_from_coefficients,
)
from lusee_faraday.depth_depolarization import (  # noqa: E402
    DepthConvergenceError,
    DepthProfile,
    DepthQuadrature,
    phase_moments,
)
from lusee_faraday.gaussian_screen import ScreenModel  # noqa: E402
from lusee_faraday.spin_covariance import (  # noqa: E402
    AngularScreen,
    SpinTransform,
    eb_spectra,
    screen_phase_moments,
)
from lusee_faraday.visibility_covariance import (  # noqa: E402
    AngularReadoutProjector,
    real_readout_kernels,
)


@pytest.mark.parametrize("fraction", [0.0, 0.6, 1.0])
def test_fractional_screen_matches_exact_phase_kernels(fraction):
    screen = AngularScreen(theta0_rad=0.3)
    x, theta = np.array([21.0, 43.0, 100.0]), np.array([0.0, 0.2, 1.4])
    result = depth_phase_coefficients(
        x, screen, DepthProfile(fraction=fraction)
    )
    a = result["coefficients"]
    rho = np.exp(-screen.exponent(theta))
    actual = phase_moments_from_coefficients(a, a, rho)
    expected = screen_phase_moments(
        x[:, None], x[None, :], theta, screen, fraction
    )
    for name in ("second", "nonconjugated_second", "covariance", "relation"):
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-12)
    np.testing.assert_allclose(actual["mean_x"], expected["mean_x"][:, 0])
    if fraction == 0:
        assert np.max(abs(actual["covariance"])) == 0


def test_slab_phase_moments_match_direct_bivariate_gaussian_averages():
    screen = AngularScreen(sigma_depth=0.02, mean_depth=0.017)
    x = np.array([28.0, 41.0, 100.0])
    result = depth_phase_coefficients(x, screen, DepthProfile(kind="slab"))
    a = result["coefficients"]
    nodes, w = roots_hermitenorm(96)
    w /= np.sqrt(2 * np.pi)
    first = screen.mean_depth + screen.sigma_depth * nodes[:, None]
    for rho in (0.0, 0.45, 1.0):
        second = screen.mean_depth + screen.sigma_depth * (
            rho * nodes[:, None] + np.sqrt(1 - rho * rho) * nodes[None, :]
        )

        def slab(depth):
            z = x[:, None, None] * depth
            return np.exp(1j * z) * np.sinc(z / np.pi)

        left, right = slab(first), slab(second)
        expected_c = np.einsum("iab,jab,a,b->ij", left, right.conj(), w, w)
        expected_j = np.einsum("iab,jab,a,b->ij", left, right, w, w)
        actual = phase_moments_from_coefficients(a, a, [rho])
        np.testing.assert_allclose(
            actual["second"][..., 0], expected_c, atol=2e-12
        )
        np.testing.assert_allclose(
            actual["nonconjugated_second"][..., 0], expected_j, atol=2e-12
        )
        assert result["depth_error"] < 1e-10


def test_truncated_gaussian_mean_matches_independent_adaptive_integral():
    profile = DepthProfile(kind="gaussian", center=0.3, width=0.13)
    screen = AngularScreen(sigma_depth=0.013, mean_depth=0.021)
    x = np.array([20.0, 60.0, 120.0])
    actual = depth_phase_coefficients(x, screen, profile)["coefficients"][:, 0]

    def weight(f):
        return np.exp(-0.5 * ((f - profile.center) / profile.width) ** 2)

    norm = quad(weight, 0, 1, epsabs=1e-13)[0]
    expected = [
        quad(
            lambda f: weight(f)
            * np.exp(
                2j * screen.mean_depth * v * f
                - 2 * (screen.sigma_depth * v * f) ** 2
            ),
            0,
            1,
            complex_func=True,
            epsabs=1e-13,
        )[0]
        / norm
        for v in x
    ]
    np.testing.assert_allclose(actual, expected, atol=3e-13)


def test_log_coefficients_recover_power_after_mean_underflows():
    screen = AngularScreen(sigma_depth=0.4, mean_depth=0.01)
    result = depth_phase_coefficients([70.0], screen)
    a = result["coefficients"][0]
    assert a[0] == 0
    assert np.max(abs(a)) > 0.05
    np.testing.assert_allclose(np.sum(abs(a) ** 2), 1, atol=5e-12)
    assert result["tail_bound"] <= 1e-12
    with pytest.raises(DepthConvergenceError, match="max_degree"):
        depth_phase_coefficients([70.0], screen, max_degree=100)


def real_projector():
    rng = np.random.default_rng(181)
    lmax = 4
    beam = rng.normal(size=(2, 4, 5, 9)) + 1j * rng.normal(size=(2, 4, 5, 9))
    for ell in range(5):
        start, stop = lmax - ell, lmax + ell + 1
        beam[:, :, ell, :start] = 0
        beam[:, :, ell, stop:] = 0
    beam[:, :, :2] = 0
    kernels = real_readout_kernels(beam, rng.normal(size=(2, 4)))
    phases = np.exp(-1j * np.array([0.0, 0.7])[:, None] * np.arange(-4, 5))
    power = np.array([0.0, 0.0, 0.8, 0.4, 0.2])
    intrinsic = eb_spectra(0.7 * power, 0.3 * power, 0.1 * power)
    return AngularReadoutProjector(
        kernels, phases, SpinTransform.gauss_legendre(4, 64), intrinsic
    )


def test_loaded_angular_channels_match_direct_double_depth_covariance():
    projected = real_projector()
    screen = AngularScreen(theta0_rad=0.4)
    x = np.array([20.0, 35.0, 60.0, 85.0])
    profile = DepthProfile(kind="slab")
    result = depth_phase_coefficients(x, screen, profile)
    amplitude = np.array([1.0, 1.1, 0.9, 1.2])
    w = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    channel = channel_phase_coefficients(result["coefficients"], w, amplitude)
    solver = DepthChannelProjector(projected, screen, result["degree"])
    actual = solver.moments(channel)
    # Independent explicit f,f' quadrature from Phase 1. This variable
    # remapping matches exp(-h²/2) to the declared angular covariance.
    model = ScreenModel(
        correlation_pc=1.0,
        mean_depth=screen.mean_depth,
        sigma_depth=screen.sigma_depth,
    )
    h = np.sqrt(2 * screen.exponent(projected.theta))
    direct = np.zeros((4, 4, 4, 4))
    for i in range(4):
        for j in range(4):
            phase = phase_moments(x[i], x[j], h, model, profile)
            direct[i, j] = (
                projected.phase_covariance(
                    phase["second"], phase["nonconjugated_second"]
                )
                * amplitude[i]
                * amplitude[j]
            )
    expected = np.einsum("bi,ijpq,cj->bpcq", w, direct, w)
    np.testing.assert_allclose(actual["total"], expected, atol=3e-11)
    np.testing.assert_allclose(
        actual["total"], actual["smooth"] + actual["speckle"]
    )
    for value in actual.values():
        assert np.linalg.eigvalsh(value.reshape(8, 8)).min() > -2e-11


def test_omitted_mode_bound_covers_actual_channel_error():
    projected = real_projector()
    screen = AngularScreen(theta0_rad=0.4)
    x = np.array([20.0, 40.0, 60.0])
    result = depth_phase_coefficients(x, screen)
    a = result["coefficients"]
    full = DepthChannelProjector(projected, screen, result["degree"]).moments(
        a
    )["total"]
    cut = 5
    solver = DepthChannelProjector(projected, screen, cut)
    coarse = solver.moments(a[:, : cut + 1])["total"]
    rms = np.sqrt(poisson.sf(cut, (2 * screen.sigma_depth * x) ** 2))
    bound = solver.omitted_covariance_bound(rms)
    assert np.max(abs(full - coarse)) > 1e-3
    assert np.all(abs(full - coarse) <= bound + 3e-12)


def test_unresolved_and_invalid_depth_channels_are_rejected():
    with pytest.raises(DepthConvergenceError, match="phase scale"):
        depth_phase_coefficients(
            [100.0],
            AngularScreen(mean_depth=5.0, sigma_depth=0.01),
            DepthProfile(kind="slab"),
            DepthQuadrature(order=2, max_order=8),
        )
    with pytest.raises(ValueError, match="bound settings"):
        hermite_degree(1.0, AngularScreen(), DepthProfile(), tail=-1)
    with pytest.raises(ValueError):
        channel_phase_coefficients(np.ones((3, 2)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        phase_moments_from_coefficients(
            np.ones((2, 3)), np.ones((2, 3)), [1.1]
        )
