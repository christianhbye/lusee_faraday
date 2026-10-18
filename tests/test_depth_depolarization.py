"""Independent depth integrals and the validated foreground-screen limit."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import roots_hermitenorm  # noqa: E402

from lusee_faraday.conventions import lambda_squared  # noqa: E402
from lusee_faraday.depth_depolarization import (  # noqa: E402
    DepthConvergenceError,
    DepthProfile,
    DepthQuadrature,
    depth_nodes,
    depth_qu_moments,
    phase_moments,
)
from lusee_faraday.gaussian_screen import (  # noqa: E402
    ScreenModel,
    SeparationQuadrature,
    qu_moments,
)


def response(phi, x, profile):
    """Direct emission integral at a deterministic depth; no moment code."""
    if profile.kind == "screen":
        return np.exp(2j * profile.fraction * phi * x)
    if profile.kind == "slab":
        return np.exp(1j * phi * x) * np.sinc(phi * x / np.pi)

    def density(f):
        return np.exp(-0.5 * ((f - profile.center) / profile.width) ** 2)

    norm = quad(density, 0, 1, epsabs=1e-13)[0]
    return (
        complex(
            quad(lambda f: density(f) * np.cos(2 * f * phi * x), 0, 1)[0],
            quad(lambda f: density(f) * np.sin(2 * f * phi * x), 0, 1)[0],
        )
        / norm
    )


def gaussian_depth_reference(x, y, h, model, profile, order=40):
    """Integrate directly over two correlated Gaussian depths, not f,f'."""
    z, w = roots_hermitenorm(order)
    w = w / np.sqrt(2 * np.pi)
    rho = np.exp(-0.5 * (h / model.correlation_pc) ** 2)
    phi = model.mean_depth + model.sigma_depth * z
    other = model.mean_depth + model.sigma_depth * (
        rho * z[:, None] + np.sqrt(1 - rho**2) * z[None, :]
    )
    a = np.array([response(v, x, profile) for v in phi])
    b = np.array([response(v, y, profile) for v in other.ravel()])
    b = b.reshape(other.shape)
    mx = w @ a
    my = np.sum(w[:, None] * w[None, :] * b)
    c = np.sum(w[:, None] * w[None, :] * (a[:, None] - mx) * (b - my).conj())
    j = np.sum(w[:, None] * w[None, :] * (a[:, None] - mx) * (b - my))
    return mx, my, c, j


@pytest.mark.parametrize(
    "profile",
    [
        DepthProfile(fraction=0),
        DepthProfile(kind="slab"),
        DepthProfile(kind="gaussian", center=0.07, width=0.02),
    ],
)
def test_profile_normalization_and_zero_rotation(profile):
    nodes, weights = depth_nodes(profile, 64)
    assert np.all((nodes >= 0) & (nodes <= 1))
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1, atol=2e-15)
    model = ScreenModel(mean_depth=0, sigma_depth=0)
    result = depth_qu_moments([10, 30, 50], model, profile)
    np.testing.assert_allclose(result["mean"], result["off"], atol=2e-12)
    assert np.max(abs(result["covariance"])) == 0
    np.testing.assert_allclose(result["total_power"], result["off_power"])


def test_fraction_zero_and_scaled_foreground_screen():
    freq = np.array([20.0, 30.0, 50.0])
    model = ScreenModel()
    zero = depth_qu_moments(freq, model, DepthProfile(fraction=0))
    np.testing.assert_allclose(zero["mean"], zero["off"])
    assert np.max(abs(zero["covariance"])) == 0
    for fraction in (0.6, 1.0):
        result = depth_qu_moments(freq, model, DepthProfile(fraction=fraction))
        reference = qu_moments(
            freq,
            replace(
                model,
                mean_depth=fraction * model.mean_depth,
                sigma_depth=fraction * model.sigma_depth,
            ),
        )
        for key in ("mean", "covariance", "second", "complex_relation"):
            np.testing.assert_allclose(result[key], reference[key], atol=2e-10)


@pytest.mark.parametrize(
    "profile",
    [
        DepthProfile(kind="slab"),
        DepthProfile(kind="gaussian", center=0.3, width=0.1),
    ],
)
def test_phase_moments_against_direct_bivariate_depths(profile):
    model = ScreenModel(mean_depth=0.02, sigma_depth=0.012)
    x, y, h = 22.0, 17.0, 0.4
    result = phase_moments(x, y, [h], model, profile)
    ref = gaussian_depth_reference(x, y, h, model, profile, order=20)
    for value, expected in zip(
        (
            result["mean_x"],
            result["mean_y"],
            result["covariance"][0],
            result["relation"][0],
        ),
        ref,
    ):
        np.testing.assert_allclose(value, expected, atol=2e-11)
    np.testing.assert_allclose(
        result["second"],
        result["covariance"] + result["mean_x"] * result["mean_y"].conjugate(),
        atol=2e-14,
    )


def test_deterministic_slab_and_truncated_gaussian_response():
    freq = np.array([10.0, 30.0, 50.0])
    x = lambda_squared(freq)
    model = ScreenModel(mean_depth=0.017, sigma_depth=0)
    for profile in (
        DepthProfile(kind="slab"),
        DepthProfile(kind="gaussian", center=0.04, width=0.1),
    ):
        result = depth_qu_moments(freq, model, profile)
        off = result["off"][:3] + 1j * result["off"][3:]
        expected = off * np.array(
            [response(model.mean_depth, v, profile) for v in x]
        )
        np.testing.assert_allclose(
            result["mean"][:3] + 1j * result["mean"][3:],
            expected,
            atol=2e-10,
        )
        assert np.max(abs(result["covariance"])) == 0
    # A Gaussian fully inside the interval approaches the untruncated law.
    profile = DepthProfile(kind="gaussian", center=0.5, width=0.03)
    result = phase_moments(80, 80, 0, model, profile)
    expected = np.exp(
        2j * 80 * model.mean_depth * 0.5
        - 2 * (80 * model.mean_depth * 0.03) ** 2
    )
    np.testing.assert_allclose(result["mean_x"], expected, atol=2e-12)


def test_uniform_random_screen_slab_beam_moments():
    """Independent scalar-depth ensemble includes both Q/U blocks."""
    freq = np.array([45.0, 50.0])
    model = ScreenModel(correlation_pc=np.inf)
    result = depth_qu_moments(freq, model, DepthProfile(kind="slab"))
    z, w = roots_hermitenorm(100)
    w = w / np.sqrt(2 * np.pi)
    phi = model.mean_depth + model.sigma_depth * z
    phase = phi[:, None] * lambda_squared(freq)
    off = result["off"][:2] + 1j * result["off"][2:]
    p = off * np.exp(1j * phase) * np.sinc(phase / np.pi)
    samples = np.c_[p.real, p.imag]
    mean = w @ samples
    centered = samples - mean
    covariance = (centered.T * w) @ centered
    np.testing.assert_allclose(result["mean"], mean, atol=2e-11)
    np.testing.assert_allclose(result["covariance"], covariance, atol=2e-11)


def test_joint_covariance_power_split_and_refinement():
    freq = np.array([20.0, 30.0, 50.0])
    model = ScreenModel()
    for profile in (DepthProfile(kind="slab"), DepthProfile(kind="gaussian")):
        result = depth_qu_moments(freq, model, profile)
        tight = depth_qu_moments(
            freq,
            model,
            profile,
            DepthQuadrature(
                order=32,
                max_order=512,
                epsabs=2e-12,
                epsrel=2e-8,
                separation=SeparationQuadrature(
                    epsabs=2e-13, epsrel=2e-12, upper_u=48
                ),
            ),
        )
        np.testing.assert_allclose(
            result["covariance"], tight["covariance"], atol=2e-9
        )
        assert np.linalg.eigvalsh(result["covariance"]).min() > -2e-10
        total = result["smooth_power"] + result["speckle_power"]
        np.testing.assert_allclose(result["total_power"], total)
        np.testing.assert_allclose(
            total,
            np.diag(result["second"])[:3] + np.diag(result["second"])[3:],
            atol=2e-12,
        )
        assert np.all(total <= result["off_power"] * (1 + 1e-9))
        diagonal = depth_qu_moments(freq, model, profile, diagonal_only=True)
        assert "covariance" not in diagonal
        np.testing.assert_allclose(
            diagonal["complex_covariance"],
            np.diag(result["complex_covariance"]),
        )


def test_raw_screen_relation_is_not_the_centered_relation():
    model = ScreenModel(mean_depth=0.03, sigma_depth=0.02)
    x = 15.0
    result = phase_moments(x, x, 0, model)
    raw = np.exp(4j * x * model.mean_depth - 8 * model.sigma_depth**2 * x**2)
    np.testing.assert_allclose(result["nonconjugated_second"], raw)
    np.testing.assert_allclose(result["relation"], raw - result["mean_x"] ** 2)
    assert abs(result["relation"][0] - raw) > 0.1


def test_unresolved_depth_ridge_raises():
    with pytest.raises(DepthConvergenceError, match="phase scale"):
        depth_qu_moments(
            [10.0], ScreenModel(sigma_depth=10), DepthProfile(kind="slab")
        )
    with pytest.raises(DepthConvergenceError, match="did not converge"):
        phase_moments(
            40,
            50,
            0,
            ScreenModel(sigma_depth=0.02),
            DepthProfile(kind="gaussian", width=0.003),
            DepthQuadrature(order=2, max_order=4, epsabs=1e-16, epsrel=1e-16),
        )


def test_invalid_depth_parameters():
    for kw in (
        {"kind": "layers"},
        {"fraction": -1},
        {"width": 0},
        {"center": np.nan},
    ):
        with pytest.raises(ValueError):
            DepthProfile(**kw)
    for kw in ({"order": 1}, {"max_order": 16}, {"epsabs": np.nan}):
        with pytest.raises(ValueError):
            DepthQuadrature(**kw)
    with pytest.raises(ValueError):
        phase_moments(1, 2, [-1], ScreenModel())
    with pytest.raises(ValueError):
        depth_qu_moments([0], ScreenModel())
