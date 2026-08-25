"""Radiometer noise and the matched-filter threshold (spec S4.10, S6.12)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

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
    # a one-sided toy shape on purpose: this pins the ENBW overlap
    # ratio, which is a property of N and is blind to the sign of phi.
    S = noise.faraday_signal_covariance(phi, H, lam2b, allow_one_sided=True)
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
        np.array([30.0, 60.0]),
        np.array([0.6, 0.4]),
        lam2b,
        allow_one_sided=True,  # a toy shape, not a sky template
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


# --------------------------------------------------------------------
# The SIGN of the depth axis in the signal covariance.
#
# The observable is the complex P = Q + iU, so S_ij is the complex
# transform of the SIGNED depth distribution.  Feeding it a FOLDED
# (all-positive) template models a sky whose every column has one sign
# of RM and understates A_5sigma by ~18% on the real map.  These pin
# the convention so a regression back to folding fails loudly.


def _signed_S(phi, H, lam2b, one_sided=False):
    return noise.faraday_signal_covariance(
        np.asarray(phi), np.asarray(H), lam2b, allow_one_sided=one_sided
    )


def test_signal_covariance_is_hermitian_with_unit_diagonal():
    """Hhat sums to one, so diag(S) = 1 exactly and S = S^dagger --
    the amplitude lives in A, never in the shape matrix."""
    lam2b = np.asarray(lambda_squared(np.linspace(29.99, 30.01, 16)), float)
    phi = np.array([-45.0, -12.0, 0.5, 30.0, 60.0])
    H = np.array([0.3, 0.1, 0.05, 0.35, 0.2])
    S = _signed_S(phi, H, lam2b)
    np.testing.assert_allclose(np.diag(S).real, 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(np.diag(S).imag), 0.0, atol=1e-12)
    np.testing.assert_allclose(S, S.conj().T, atol=1e-12)


def test_folding_the_template_changes_the_covariance():
    """The regression guard.  For an ASYMMETRIC signed sky the folded
    and signed templates give materially different S, so silently
    reverting the call sites to d["H"] (folded) cannot pass."""
    lam2b = np.asarray(lambda_squared(np.linspace(29.99, 30.01, 24)), float)
    # A NEAR-symmetric signed sky, which is the regime the real map is
    # in (measured L1 asymmetry 0.118).  The strongly asymmetric case
    # below is the easy one; this is the one that actually bites.
    S_signed = _signed_S(
        [-60.0, -30.0, 30.0, 60.0], [0.22, 0.30, 0.28, 0.20], lam2b
    )
    S_folded = _signed_S([30.0, 60.0], [0.58, 0.42], lam2b, one_sided=True)
    rel = np.abs(S_signed - S_folded).max() / np.abs(S_signed).max()
    assert rel > 0.3, rel
    # The symptom: a near-symmetric signed sky has an almost real
    # covariance, and folding inflates the imaginary part ~25x.
    assert np.abs(S_folded.imag).max() > 5.0 * np.abs(S_signed.imag).max()

    # Strongly asymmetric sky: folded and signed simply disagree.
    Sa = _signed_S([30.0, 60.0, -45.0], [0.5, 0.2, 0.3], lam2b)
    Sf = _signed_S([30.0, 45.0, 60.0], [0.5, 0.3, 0.2], lam2b, one_sided=True)
    assert np.abs(Sa - Sf).max() / np.abs(Sa).max() > 0.3


def test_symmetric_signed_sky_gives_a_real_covariance():
    """Sanity anchor for the above: exact +-phi symmetry makes the
    complex transform collapse to the cosine transform."""
    lam2b = np.asarray(lambda_squared(np.linspace(29.99, 30.01, 16)), float)
    phi = np.array([-60.0, -30.0, 30.0, 60.0])
    H = np.array([0.2, 0.3, 0.3, 0.2])
    S = _signed_S(phi, H, lam2b)
    np.testing.assert_allclose(S.imag, 0.0, atol=1e-12)


def test_complex_p_noise_equals_radiometer_sigma_squared():
    """Pins the NOISE normalisation, end to end through pseudo_stokes.

    The data vector is the complex P = Q + iU (polarimeter.pseudo_stokes
    returns Q and U as independent real spectra, and the Faraday phase
    acts on Q + iU).  The naive worry is that <|n_P|^2> = 2 sigma^2,
    since Q and U each contribute -- which would raise every A_5sigma
    by sqrt(2).  It does not, and the reason is the factors of two in
    pseudo_stokes itself: with I = (XX+YY)/2 and Q = (XX-YY)/2,

        sigma_Q = sigma_U = sigma_rad / sqrt(2),
        <|n_P|^2> = sigma_Q^2 + sigma_U^2 = sigma_rad^2,

    where sigma_rad = radiometer_sigma(T_sys) and T_sys is the measured
    Stokes I, which is what noise.py's "T_sys is sky-dominated (~ Stokes
    I)" fixes.  The per-component 1/sqrt(2) exactly cancels the two
    components, so zoom_noise_covariance needs NO factor of two and
    A_5sigma is unchanged.  Q and U also come out uncorrelated, so
    there is no cross term to carry.

    Scale-invariant by construction: Q and I are both linear in the
    port covariance, so the probe-gain convention cancels in the ratio.
    """
    from lusee_faraday.polarimeter import X_VEC, Y_VEC, pseudo_stokes

    rng = np.random.default_rng(0)
    t_pol, nsamp, nreal, chunk = 1.0, 400, 12_000, 1_000

    def cg(shape):
        z = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return z / np.sqrt(2)

    acc = []
    for _ in range(nreal // chunk):
        a = np.sqrt(t_pol) * cg((chunk, nsamp))
        b = np.sqrt(t_pol) * cg((chunk, nsamp))
        v = a[..., None] * X_VEC + b[..., None] * Y_VEC
        cov = np.einsum("mna,mnb->mab", v, v.conj()) / nsamp
        acc.append(pseudo_stokes(cov / 2.0))
    S = np.concatenate(acc)
    I, Q, U = S[:, 0], S[:, 1], S[:, 2]

    sigma = noise.radiometer_sigma(I.mean(), nsamp, 1.0)
    # each Stokes component sits at sigma/sqrt(2), not sigma
    assert np.isclose(Q.std(), sigma / np.sqrt(2), rtol=0.06), Q.std() / sigma
    assert np.isclose(U.std(), sigma / np.sqrt(2), rtol=0.06), U.std() / sigma
    # and they are uncorrelated, so <|n_P|^2> is a plain sum
    assert abs(np.corrcoef(Q, U)[0, 1]) < 0.05
    # which lands exactly on sigma^2 -- the factor 2 does NOT apply
    var_p = (np.abs((Q - Q.mean()) + 1j * (U - U.mean())) ** 2).mean()
    assert np.isclose(var_p, sigma**2, rtol=0.06), var_p / sigma**2


def test_folded_template_is_refused_by_default():
    """The structural guard.  A folded (all-positive) grid is the
    shipped bug; it must raise rather than silently return an S that is
    18% optimistic.  Reproduced in fresh prototype code within an hour
    of being found, which is why it is refused and not documented."""
    lam2b = np.asarray(lambda_squared(np.linspace(29.99, 30.01, 8)), float)
    phi = np.array([0.5, 30.0, 60.0])
    H = np.array([0.2, 0.5, 0.3])
    with pytest.raises(ValueError, match="FOLDED"):
        noise.faraday_signal_covariance(phi, H, lam2b)
    # the opt-out still works, and a signed grid passes untouched
    noise.faraday_signal_covariance(phi, H, lam2b, allow_one_sided=True)
    noise.faraday_signal_covariance(np.array([-30.0, 0.5, 30.0]), H, lam2b)


def test_signed_covariance_is_hermitian_and_nearly_symmetric():
    """Folded S is Hermitian but grossly NON-symmetric; signed S is
    both.  max|S-S^T| on the real 30 MHz template: 1.08 folded, 0.19
    signed.  The residual is the sky's genuine ~12% L1 sign asymmetry,
    so 'nearly' is the strongest true statement."""
    lam2b = np.asarray(lambda_squared(np.linspace(29.99, 30.01, 24)), float)
    phi = np.concatenate(
        [-np.arange(60.0, 0.0, -5.0), np.arange(5.0, 65.0, 5.0)]
    )
    rng = np.random.default_rng(31)
    H = rng.gamma(3.0, 1.0, size=phi.size)
    S = noise.faraday_signal_covariance(phi, H, lam2b)
    assert np.abs(S - S.conj().T).max() < 1e-12  # Hermitian, always
    Sf = noise.faraday_signal_covariance(
        np.abs(phi[phi > 0]), H[phi > 0], lam2b, allow_one_sided=True
    )
    assert np.abs(S - S.T).max() < np.abs(Sf - Sf.T).max()
