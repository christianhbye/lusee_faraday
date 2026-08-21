"""Radiometer noise and the matched-filter threshold (spec S4.10, S6.12)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from lusee_faraday import noise
from lusee_faraday.conventions import lambda_squared


def test_radiometer_sigma_closed_form():
    assert np.isclose(
        noise.radiometer_sigma(1.0, 563.4, 2305.0), 8.8e-4, rtol=0.02
    )


def test_add_noise_statistics():
    rng = np.random.default_rng(0)
    x = noise.add_noise(np.zeros(200_000), 2.0, rng)
    assert np.isclose(x.std(), 2.0, rtol=0.02)


def test_closed_form_reproduces_the_spec_table():
    """S4.10 corrected row: 30 MHz, zoom on 3 parents, n=1 -> 4.9e-5."""
    got = noise.closed_form_threshold(10838.0, 2305.0, 7086, 1)
    assert np.isclose(got, 4.9e-5, rtol=0.03)
    # 50 MHz parent 200 kHz row: 2.6e-5
    got50 = noise.closed_form_threshold(50176.0, 2305.0, 4086, 1)
    assert np.isclose(got50, 2.6e-5, rtol=0.03)
    # scalings: n^-1/2 and N^-1/4
    assert np.isclose(
        noise.closed_form_threshold(10838.0, 2305.0, 7086, 4) / got,
        0.5,
        rtol=1e-6,
    )


def test_matched_filter_reduces_to_the_closed_form_when_diagonal():
    n, n_lst, sigma = 7, 1024, 2.0e-4
    S = np.eye(n, dtype=complex)
    N = sigma**2 * np.eye(n)
    got = noise.matched_filter_threshold(S, N, n_nights=1, n_lst=n_lst)
    expected = noise.closed_form_threshold(
        10838.0, 2305.0, n * n_lst, 1
    )  # sigma_mode(10838, 2305) = 2.0e-4 = sigma
    assert np.isclose(got, expected, rtol=0.01)


def test_overlap_correlation_degrades_the_threshold():
    """S6.12: the 1.44x zoom overlap must show up, not be ignored."""
    from lusee_faraday import dispersion as dsp

    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    sigma = 8.8e-4
    N_corr = noise.zoom_noise_covariance(W, sigma)
    N_diag = sigma**2 * np.eye(bins.size)
    lam2b = np.asarray(lambda_squared(bins), dtype=float)
    phi = np.arange(2.0, 120.0, 4.0)
    H = np.exp(-phi / 30.0)
    S = noise.faraday_signal_covariance(phi, H, lam2b)
    a_corr = noise.matched_filter_threshold(S, N_corr, 1, 1024)
    a_diag = noise.matched_filter_threshold(S, N_diag, 1, 1024)
    ratio = a_corr / a_diag
    print(f"\noverlap degradation: {ratio:.3f}x")
    # S6.12: the 1.44x ENBW overlap must show up with the right
    # MAGNITUDE, not merely the right sign.  A covariance with a
    # 300x-too-small correlation still satisfies ratio > 1, so the
    # bare inequality is not a regression guard.  Measured: 1.160.
    assert 1.05 < ratio < 1.5, ratio


def test_zoom_noise_covariance_contract():
    """Direct contract for zoom_noise_covariance (S6.12).

    These are the properties that make the overlap-degradation ratio
    in test_overlap_correlation_degrades_the_threshold meaningful:
    unit diagonal (by construction), symmetric, PSD, and a
    nearest-neighbour correlation of order 0.15 (the real ~1.44x ENBW
    overlap) rather than ~0 (a diagonal-only covariance).
    """
    from lusee_faraday import dispersion as dsp

    _, bins, W = dsp.zoom_bin_matrix(30.0)
    sigma = 8.8e-4
    N = noise.zoom_noise_covariance(W, sigma)

    np.testing.assert_allclose(np.diag(N), sigma**2, rtol=1e-10)
    np.testing.assert_allclose(N, N.T, rtol=1e-10)

    eigvals = np.linalg.eigvalsh(N)
    assert eigvals.min() > 0.0

    rho = N / sigma**2
    nn_corr = np.diag(rho, k=1)
    # Measured range is [0.074, 0.153], median 0.152 -- of order the
    # real 1.44x ENBW overlap, not ~0 (a diagonal-only covariance).
    assert np.all(np.abs(nn_corr) > 0.05)
    assert np.all(np.abs(nn_corr) < 0.25)
    assert 0.1 < np.median(np.abs(nn_corr)) < 0.2


def test_matched_filter_monte_carlo():
    """The Fisher SNR matches the empirical score-statistic shift."""
    rng = np.random.default_rng(7)
    nb, M = 48, 3000
    lam2b = np.asarray(
        lambda_squared(np.linspace(29.99, 30.01, nb)), dtype=float
    )
    S = noise.faraday_signal_covariance(
        np.array([30.0, 60.0]), np.array([0.6, 0.4]), lam2b
    )
    sigma2 = 1e-6
    N = sigma2 * np.eye(nb)
    A2 = 4e-7  # amplitude^2, weak-signal regime
    F = np.linalg.solve(N, S)
    snr_pred = A2 * np.sqrt(np.einsum("ij,ji->", F, F).real)

    Ls = np.linalg.cholesky(S + 1e-12 * np.eye(nb))

    def draw(with_signal):
        x = (
            rng.normal(size=(nb, M)) + 1j * rng.normal(size=(nb, M))
        ) / np.sqrt(2)
        x *= np.sqrt(sigma2)
        if with_signal:
            g = (
                rng.normal(size=(nb, M)) + 1j * rng.normal(size=(nb, M))
            ) / np.sqrt(2)
            x = x + np.sqrt(A2) * (Ls @ g)
        NiSNi = np.linalg.solve(N, S) @ np.linalg.inv(N)
        return np.einsum("im,ij,jm->m", x.conj(), NiSNi, x).real

    q0, q1 = draw(False), draw(True)
    snr_emp = (q1.mean() - q0.mean()) / q0.std()
    assert np.isclose(snr_emp, snr_pred, rtol=0.15)
