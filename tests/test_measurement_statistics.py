"""Full correlated Gaussian power uncertainty and repeated-sky controls."""

import numpy as np
import pytest
from lusee_faraday.measurement_statistics import (
    gaussian_mean_square,
    repeat_difference,
)


def test_power_uncertainty_matches_exact_gaussian_fourth_moments():
    # Three-point cubature integrates each independent Gaussian through
    # fourth order; correlations and the nonzero mean remain explicit.
    from itertools import product

    values = np.array(list(product([-np.sqrt(3), 0, np.sqrt(3)], repeat=3)))
    weights = np.prod(
        np.array(list(product([1 / 6, 2 / 3, 1 / 6], repeat=3))), axis=1
    )
    transform = np.array([[1, 0.8, -0.3], [0.7, 2, 0.5], [0.1, 0.9, 1.4]])
    mean = np.array([0.5, -0.8, 1.1])
    samples = values @ transform.T + mean
    power = np.mean(samples * samples, axis=1)
    expected = gaussian_mean_square(transform @ transform.T, mean)
    np.testing.assert_allclose(
        expected["mean_k2"], weights @ power, rtol=1e-14
    )
    np.testing.assert_allclose(
        expected["sampling_rms_k2"] ** 2,
        weights @ power**2 - (weights @ power) ** 2,
        rtol=1e-14,
    )


def test_repeat_difference_cancels_fixed_sky_and_adds_independent_noise():
    ca = np.array([[1, 0.3], [0.3, 2.0]])
    cb = ca * 3
    sky = np.array([10000.0, -20000.0])
    first, second = np.array([0.1, 0.3]), np.array([-0.2, 0.1])
    result = repeat_difference(sky + first, sky + second, ca, cb)
    np.testing.assert_allclose(
        result["difference"], first - second, atol=3e-12
    )
    np.testing.assert_allclose(result["covariance"], ca + cb)
    assert result["prediction"]["mean_k2"] == 6


def test_invalid_statistics_covariance_is_rejected():
    with pytest.raises(ValueError):
        gaussian_mean_square(np.diag([1.0, -1.0]))
