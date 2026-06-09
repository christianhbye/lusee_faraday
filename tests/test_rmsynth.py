import numpy as np
import pytest
from lusee_faraday import rmsynth

C = 299792458.0


def test_lambda2_matches_formula():
    nu = np.array([10.0, 30.0, 50.0])
    expected = (C / (nu * 1e6)) ** 2
    np.testing.assert_allclose(rmsynth.lambda2(nu), expected)


def test_lambda2_inverse_square_scaling():
    assert np.isclose(rmsynth.lambda2(10) / rmsynth.lambda2(20), 4.0)


def test_faraday_resolution():
    lam2 = np.linspace(0.0, 1.0, 50)
    assert np.isclose(rmsynth.faraday_resolution(lam2), 2 * np.sqrt(3))


def test_max_scale():
    lam2 = np.array([1.0, 2.0, 3.0])
    assert np.isclose(rmsynth.max_scale(lam2), np.pi)
