import numpy as np

from lusee_faraday.beam import Beam
from lusee_faraday.fisher import (
    detection_snr,
    dispersion_column,
    faraday_column,
    fisher_matrix,
    marginal_error,
    run_forecast,
    stack_real,
)


def test_stack_real_layout():
    P = np.array([[1 + 2j, 3 + 4j]])
    np.testing.assert_array_equal(stack_real(P), [1, 3, 2, 4])


def test_orthogonal_nuisance_does_not_inflate():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)  # real quadrature
    nuisance = 1j * np.ones((1, n), dtype=complex)  # imag quadrature
    F = fisher_matrix([signal, nuisance], sig)
    F_only = fisher_matrix([signal], sig)
    # orthogonal -> marginalized error equals unmarginalized
    assert np.isclose(marginal_error(F, 0), marginal_error(F_only, 0))
    assert np.isclose(marginal_error(F_only, 0), 1.0 / np.sqrt(n))


def test_degenerate_nuisance_inflates_error():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)
    near_parallel = (1.0 + 1e-3 * 1j) * np.ones((1, n), dtype=complex)
    F = fisher_matrix([signal, near_parallel], sig)
    F_only = fisher_matrix([signal], sig)
    assert marginal_error(F, 0) > 10 * marginal_error(F_only, 0)
    # marginalizing can only reduce SNR
    assert detection_snr(F, 0) < detection_snr(F_only, 0)


def test_sigma_scaling():
    n = 8
    signal = np.ones((1, n), dtype=complex)
    F1 = fisher_matrix([signal], np.ones((1, n)))
    F2 = fisher_matrix([signal], 2 * np.ones((1, n)))
    assert np.isclose(marginal_error(F2, 0), 2 * marginal_error(F1, 0))


def test_marginal_differs_from_conditional():
    # Off-diagonal but well-conditioned F: marginalizing a correlated
    # nuisance must inflate the error above the conditional 1/sqrt(F[ii]).
    # A conditional implementation (1/sqrt(F[0,0])) would fail this.
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)
    nuisance = (0.5 + 0.5j) * np.ones((1, n), dtype=complex)
    F = fisher_matrix([signal, nuisance], sig)
    conditional = 1.0 / np.sqrt(F[0, 0])
    marg = marginal_error(F, 0)
    assert marg > conditional * (1 + 1e-6)
    np.testing.assert_allclose(marg, np.sqrt(2.0 / n), rtol=1e-9)


def _toy(nside=8, ntimes=2, nfreq=3, seed=1):
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    I = rng.normal(size=(ntimes, npix)) + 100.0
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = rng.normal(size=(ntimes, npix)) * 5.0
    beam = Beam.short_dipole(nside=nside)
    beam.precompute_weights()
    mask = np.ones(npix, dtype=bool)
    freqs = np.linspace(10.0, 50.0, nfreq)
    lam2 = (3e8 / (freqs * 1e6)) ** 2
    basis = [(rng.normal(size=npix), rng.normal(size=npix)) for _ in range(2)]
    return I, Q, U, rm, beam, mask, freqs, lam2, basis


def test_faraday_and_dispersion_columns_shape():
    I, Q, U, rm, beam, mask, freqs, lam2, _ = _toy()
    a = faraday_column(I, Q, U, rm, beam, mask, freqs)
    assert a.shape == (I.shape[0], freqs.size) and np.iscomplexobj(a)
    P_pol = np.ones((I.shape[0], freqs.size), dtype=complex)
    t = dispersion_column(P_pol, lam2)
    np.testing.assert_allclose(t, -2 * lam2[None, :] ** 2 * P_pol)


def test_run_forecast_marginalized_le_fixed():
    I, Q, U, rm, beam, mask, freqs, lam2, basis = _toy()
    sigma = np.ones((I.shape[0], freqs.size))
    out = run_forecast(I, Q, U, rm, basis, beam, mask, freqs, lam2, sigma)
    assert np.isfinite(out["snr"]) and out["snr"] > 0
    # marginalizing the sky+tau cannot increase the SNR
    assert out["snr"] <= out["snr_opt"] * (1 + 1e-9)
    assert out["n_modes"] == 2
