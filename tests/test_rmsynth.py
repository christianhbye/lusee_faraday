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


def test_phi_grid_range_and_symmetry():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=100.0, dphi=1.0)
    assert np.isclose(phi[0], -100.0)
    assert np.isclose(phi[-1], 100.0)
    assert np.isclose(phi[len(phi) // 2], 0.0)


def test_phi_grid_spacing_respects_dphi():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=50.0, dphi=0.5)
    assert np.all(np.diff(phi) <= 0.5 + 1e-9)


def test_phi_grid_default_dphi_oversamples_resolution():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=10.0, oversample=3)
    assert np.median(np.diff(phi)) <= rmsynth.faraday_resolution(lam2) / 3
