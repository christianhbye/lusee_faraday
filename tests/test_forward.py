import numpy as np

from lusee_faraday.beam import Beam
from lusee_faraday.forward import pol_response


def _setup(nside=8, ntimes=2, nfreq=3, seed=0):
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
    return I, Q, U, rm, beam, mask, freqs


def test_pol_response_shape_and_complex():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P = pol_response(I, Q, U, rm, beam, mask, freqs)
    assert P.shape == (I.shape[0], freqs.size)
    assert np.iscomplexobj(P)


def test_pol_response_linear_in_QU():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P0 = pol_response(I, 0 * Q, 0 * U, rm, beam, mask, freqs)
    P1 = pol_response(I, Q, U, rm, beam, mask, freqs)
    P2 = pol_response(I, 2 * Q, 2 * U, rm, beam, mask, freqs)
    np.testing.assert_allclose(P2 - P0, 2 * (P1 - P0), rtol=1e-10, atol=1e-10)


def test_alpha_zero_is_unrotated():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P_a0 = pol_response(I, Q, U, rm, beam, mask, freqs, alpha=0.0)
    P_rm0 = pol_response(I, Q, U, 0 * rm, beam, mask, freqs, alpha=1.0)
    np.testing.assert_allclose(P_a0, P_rm0, rtol=1e-12, atol=1e-12)
