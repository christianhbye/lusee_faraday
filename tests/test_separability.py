"""Estimator alignment, residual covariance and noisy-intensity ratios."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.channelization import channel_operator  # noqa: E402
from lusee_faraday.separability import (  # noqa: E402
    amplitude_ratio,
    channel_smooth_basis,
    fractional_residual,
    residual_covariance,
    residual_operator,
)


def test_smooth_templates_include_actual_zoom_fold():
    fine = np.linspace(29.95, 30.05, 1025)
    w = channel_operator(fine, [30.0], "zoom")
    basis = channel_smooth_basis(fine, w, 1, center_mhz=30.0)
    np.testing.assert_allclose(basis[:, 0], 1, atol=3e-15)
    # Response evaluation and MHz subtraction have finite precision. Match
    # the independently mirrored power weights, then bound the tiny offset.
    symmetric = (w[32] + w[32, ::-1]) / 2
    coordinate = (fine - 30) / ((fine[-1] - fine[0]) / 2)
    expected = symmetric @ coordinate
    asymmetry_bound = abs(w[32] - w[32, ::-1]) @ abs(coordinate) / 2
    np.testing.assert_allclose(
        basis[32, 1], expected, atol=asymmetry_bound + 16 * np.finfo(float).eps
    )
    # The nominal Nyquist bin is -12.5 kHz, but its actual linear-template
    # response is centered at zero because of its symmetric folded image.
    assert basis[31, 1] > 0
    assert basis[33, 1] < 0
    fitted = residual_operator(basis)
    np.testing.assert_allclose(fitted["operator"] @ basis, 0, atol=2e-14)


def test_weighted_smooth_fit_uses_weighted_mean_and_excludes_flags():
    basis = np.ones((4, 1))
    weights = np.array([1.0, 0.0, 3.0, 2.0])
    values = np.array([0.0, 999.0, 4.0, 8.0])
    result = residual_operator(basis, weights)
    expected = weights @ values / weights.sum()
    residual = result["operator"] @ values
    np.testing.assert_allclose(
        residual[[0, 2, 3]], values[[0, 2, 3]] - expected
    )
    assert residual[1] == 0
    assert result["residual_dof"] == 2
    assert result["condition_number"] == 1


def test_residual_covariance_matches_independent_finite_ensemble():
    rng = np.random.default_rng(441)
    n, p, rank = 6, 2, 5
    latent = np.r_[np.eye(rank), -np.eye(rank)] * np.sqrt(rank)
    matrix = rng.normal(size=(n * p, rank))
    samples = (latent @ matrix.T).reshape(2 * rank, n, p)
    c = (matrix @ matrix.T).reshape(n, p, n, p)
    basis = np.polynomial.legendre.legvander(np.linspace(-1, 1, n), 1)
    r = residual_operator(basis)["operator"]
    direct = np.einsum("bi,nip->nbp", r, samples).reshape(2 * rank, n * p)
    expected = (direct.T @ direct / (2 * rank)).reshape(n, p, n, p)
    np.testing.assert_allclose(residual_covariance(c, r), expected, atol=3e-14)


def test_intensity_uncertainty_and_cross_covariance_match_ratio_monte_carlo():
    rng = np.random.default_rng(442)
    n = 3
    central = np.array(
        [
            [100.0, 1.0, 4.0, 0.0],
            [110.0, 5.0, 1.0, 0.0],
            [120.0, 2.0, 3.0, 0.0],
        ]
    )
    scale = np.tile([1.0, 0.01, 0.015, 0.01], n)
    factor = scale[:, None] * rng.normal(size=(4 * n, 4 * n)) / np.sqrt(4 * n)
    c = (factor @ factor.T).reshape(n, 4, n, 4)
    r = residual_operator(np.c_[np.ones(n), np.arange(n)])["operator"]
    result = fractional_residual(central, c, r)
    samples = (rng.normal(size=(100000, 4 * n)) @ factor.T).reshape(
        -1, n, 4
    ) + central
    values = (
        np.einsum("bi,nip->nbp", r, samples[:, :, 1:3])
        / samples[:, :, 0, None]
    )
    centered = values.reshape(len(values), -1) - values.reshape(
        len(values), -1
    ).mean(axis=0)
    measured = centered.T @ centered / len(values)
    predicted = result["covariance"].reshape(2 * n, 2 * n)
    covariance_scale = np.sqrt(
        np.outer(np.diag(predicted), np.diag(predicted))
    )
    assert np.max(abs(measured - predicted) / covariance_scale) < 0.02
    fixed = fractional_residual(
        central, c, r, intensity_reference=central[:, 0]
    )
    assert np.max(abs(fixed["covariance"] - result["covariance"])) > 1e-9
    assert result["maximum_intensity_fractional_rms"] < 0.02


def test_smooth_subtraction_precedes_intensity_division():
    n = 3
    s = np.array(
        [[10.0, 1.0, 2.0, 0.0], [20.0, 2.0, 4.0, 0.0], [40.0, 4.0, 8.0, 0.0]]
    )
    r = residual_operator(np.c_[np.ones(n), np.arange(n)])["operator"]
    result = fractional_residual(s, np.zeros((n, 4, n, 4)), r)
    np.testing.assert_allclose(
        result["value"], (r @ s[:, 1:3]) / s[:, 0, None]
    )
    assert np.max(abs(result["value"])) > 0.01
    np.testing.assert_allclose(r @ (s[:, 1:3] / s[:, 0, None]), 0, atol=1e-16)


def test_undefined_ratios_and_uninformative_fits_are_explicit():
    result = amplitude_ratio([0.0, 1.0, 2.0], [0.0, 0.0, 4.0])
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == 0.5
    with pytest.raises(ValueError, match="degrees of freedom"):
        residual_operator(np.eye(3))
    with pytest.raises(ValueError, match="rank deficient"):
        residual_operator(np.ones((4, 2)))
    with pytest.raises(ValueError, match="uncertainty"):
        fractional_residual(
            np.ones((3, 4)), np.eye(12).reshape(3, 4, 3, 4), np.eye(3)
        )
