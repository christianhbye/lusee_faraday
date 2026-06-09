import numpy as np
import pytest
from lusee_faraday import freqfft


def _uniform_nu(nchan, nu0=1.0, dnu_mhz=0.0256):
    """Uniform frequency grid (MHz)."""
    return nu0 + np.arange(nchan) * dnu_mhz


def test_faraday_fft_1d_input_returns_2d():
    nu = _uniform_nu(64)
    Q = np.zeros(nu.size)
    U = np.zeros(nu.size)
    F = freqfft.faraday_fft(Q, U, nu)
    assert F.shape == (1, nu.size)


def test_faraday_fft_shape_multi_time():
    nu = _uniform_nu(64)
    Q = np.zeros((3, nu.size))
    U = np.zeros((3, nu.size))
    F = freqfft.faraday_fft(Q, U, nu)
    assert F.shape == (3, nu.size)


def test_faraday_fft_pad_lengthens_output():
    nu = _uniform_nu(64)
    Q = np.zeros(nu.size)
    U = np.zeros(nu.size)
    F = freqfft.faraday_fft(Q, U, nu, pad=2)
    assert F.shape == (1, 2 * nu.size)


def test_delay_grid_length_and_zero_centered():
    nu = _uniform_nu(64)
    tau = freqfft.delay_grid(nu)
    assert tau.size == nu.size
    assert np.isclose(tau[tau.size // 2], 0.0)


def test_delay_grid_pad_lengthens():
    nu = _uniform_nu(64)
    tau = freqfft.delay_grid(nu, pad=2)
    assert tau.size == 2 * nu.size


def test_delay_grid_spacing_matches_fft_convention():
    nu = _uniform_nu(64, dnu_mhz=0.025)
    tau = freqfft.delay_grid(nu)
    expected = np.fft.fftshift(np.fft.fftfreq(nu.size, d=0.025e6))
    np.testing.assert_allclose(tau, expected)


def test_dc_term_is_mean_polarization():
    nu = _uniform_nu(64)
    rng = np.random.default_rng(0)
    Q = rng.normal(size=nu.size)
    U = rng.normal(size=nu.size)
    F = freqfft.faraday_fft(Q, U, nu)
    tau = freqfft.delay_grid(nu)
    dc = F[0, np.argmin(np.abs(tau))]
    assert np.isclose(dc, Q.mean() + 1j * U.mean())


def test_pure_tone_peaks_at_expected_delay():
    nchan, m0, dnu_mhz = 64, 5, 0.025
    nu = _uniform_nu(nchan, dnu_mhz=dnu_mhz)
    k = np.arange(nchan)
    P = np.exp(2j * np.pi * m0 * k / nchan)
    F = freqfft.faraday_fft(P.real, P.imag, nu)
    tau = freqfft.delay_grid(nu)
    peak = tau[np.argmax(np.abs(F[0]))]
    expected = m0 / (nchan * dnu_mhz * 1e6)
    assert np.isclose(peak, expected)


def test_constant_polarization_peaks_at_zero_delay():
    nu = _uniform_nu(64)
    Q = np.full(nu.size, 2.0)
    U = np.full(nu.size, -1.0)
    F = freqfft.faraday_fft(Q, U, nu)
    tau = freqfft.delay_grid(nu)
    peak = tau[np.argmax(np.abs(F[0]))]
    assert np.isclose(peak, 0.0)


def test_uniform_weights_match_none():
    nu = _uniform_nu(64)
    rng = np.random.default_rng(1)
    Q = rng.normal(size=nu.size)
    U = rng.normal(size=nu.size)
    F_none = freqfft.faraday_fft(Q, U, nu)
    F_ones = freqfft.faraday_fft(Q, U, nu, weights=np.ones(nu.size))
    np.testing.assert_allclose(F_none, F_ones)


def test_non_uniform_nu_raises():
    nu = np.array([1.0, 2.0, 4.0])
    Q = np.zeros(nu.size)
    U = np.zeros(nu.size)
    with pytest.raises(ValueError):
        freqfft.faraday_fft(Q, U, nu)


def test_decreasing_nu_raises():
    nu = np.array([3.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        freqfft.delay_grid(nu)


def test_too_short_nu_raises():
    with pytest.raises(ValueError):
        freqfft.delay_grid(np.array([1.0]))


def test_qu_length_mismatch_raises():
    nu = _uniform_nu(64)
    Q = np.zeros(32)
    U = np.zeros(32)
    with pytest.raises(ValueError):
        freqfft.faraday_fft(Q, U, nu)


def test_weights_wrong_shape_raises():
    nu = _uniform_nu(64)
    Q = np.zeros(nu.size)
    U = np.zeros(nu.size)
    with pytest.raises(ValueError):
        freqfft.faraday_fft(Q, U, nu, weights=np.ones(nu.size + 1))


def test_weights_zero_sum_raises():
    nu = _uniform_nu(64)
    Q = np.zeros(nu.size)
    U = np.zeros(nu.size)
    with pytest.raises(ValueError):
        freqfft.faraday_fft(Q, U, nu, weights=np.zeros(nu.size))
