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


def test_rmsf_peak_at_zero_is_unity():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    R = rmsynth.rmsf(lam2, np.array([0.0]))
    assert np.isclose(np.abs(R[0]), 1.0)


def test_rmsf_single_channel_is_flat():
    lam2 = np.array([100.0])
    phi = np.linspace(-50, 50, 101)
    R = rmsynth.rmsf(lam2, phi)
    np.testing.assert_allclose(np.abs(R), 1.0)


def test_rmsf_weights_normalized():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    w = np.random.default_rng(0).uniform(0.1, 1.0, lam2.size)
    R = rmsynth.rmsf(lam2, np.array([0.0]), weights=w)
    assert np.isclose(np.abs(R[0]), 1.0)


def _synthetic_pol(lam2, rm, chi0=0.3, amp=1.0):
    p = amp * np.exp(2j * (chi0 + rm * lam2))
    return p.real, p.imag


def test_faraday_spectrum_recovers_positive_rm():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    rm_true = 8.0
    Q, U = _synthetic_pol(lam2, rm_true)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    peak = phi[np.argmax(np.abs(F[0]))]
    assert abs(peak - rm_true) < rmsynth.faraday_resolution(lam2) * 0.5


def test_faraday_spectrum_recovers_negative_rm():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    rm_true = -12.0
    Q, U = _synthetic_pol(lam2, rm_true)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    peak = phi[np.argmax(np.abs(F[0]))]
    assert abs(peak - rm_true) < rmsynth.faraday_resolution(lam2) * 0.5


def test_faraday_spectrum_zero_rm_peaks_at_zero():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    Q, U = _synthetic_pol(lam2, 0.0)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    assert abs(phi[np.argmax(np.abs(F[0]))]) < rmsynth.faraday_resolution(lam2) * 0.5


def test_faraday_spectrum_shape_multi_time():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 100))
    Q = np.zeros((3, lam2.size))
    U = np.zeros((3, lam2.size))
    phi = rmsynth.phi_grid(lam2, phi_max=10.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    assert F.shape == (3, phi.size)


def test_faraday_resolution_zero_span_raises():
    with pytest.raises(ValueError):
        rmsynth.faraday_resolution(np.array([5.0, 5.0]))


def test_normalized_weights_zero_sum_raises():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 10))
    with pytest.raises(ValueError):
        rmsynth.rmsf(lam2, np.array([0.0]), weights=np.zeros(lam2.size))


def test_faraday_spectrum_1d_input_returns_2d():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 100))
    Q, U = _synthetic_pol(lam2, 5.0)
    phi = rmsynth.phi_grid(lam2, phi_max=10.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    assert F.shape == (1, phi.size)


def test_faraday_spectrum_accepts_weights():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 100))
    Q, U = _synthetic_pol(lam2, 5.0)
    phi = rmsynth.phi_grid(lam2, phi_max=20.0)
    w = np.linspace(0.5, 1.0, lam2.size)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi, weights=w)
    peak = phi[np.argmax(np.abs(F[0]))]
    assert abs(peak - 5.0) < rmsynth.faraday_resolution(lam2) * 2
