"""Analytic limits of the dispersion module (spec S6.3)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp
from lusee_faraday.config import PHI_FD_POINT, fine_freqs
from lusee_faraday.conventions import faraday_phase_cosmo, lambda_squared

FREQS_30 = fine_freqs(30.0)[::64]  # 256 fine frequencies, +-0.1 MHz


def test_phi_edges_width_and_span():
    edges = dsp.phi_edges(30.0)
    dphi = np.diff(edges)
    lam2_max = float(np.asarray(lambda_squared(29.9))[0])
    assert np.allclose(dphi, np.pi / (2 * lam2_max))
    assert np.isclose(dphi[0], 0.016, atol=2e-3)  # spec S3 number
    assert edges[0] <= -2500.0 and edges[-1] >= 2500.0
    # 10 MHz: 0.0017 rad/m^2 bins (spec S3)
    assert np.isclose(np.diff(dsp.phi_edges(10.0))[0], 0.0017, atol=2e-4)


def test_delta_is_pure_winding():
    """F = delta(phi - PHI_FD_POINT) -> the repo's COSMO Faraday phase."""
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(np.array([PHI_FD_POINT]), np.array([1.0]), lam2)
    expected = faraday_phase_cosmo(np.array([PHI_FD_POINT]), FREQS_30)[0]
    np.testing.assert_allclose(P, expected, rtol=0, atol=1e-9)


def test_tophat_is_sinc_with_the_right_factor():
    """F uniform on [0, Phi] -> |sin(Phi lam2)/(Phi lam2)|, NOT sinc(2...).

    Spec S6.3: under e^{+2i phi lam2}, Int_0^1 e^{2 i f Phi lam2} df has
    modulus |sin(Phi lam2)/(Phi lam2)|.
    """
    Phi = 25.0
    n = 1 << 17
    dphi = Phi / n
    phi = (np.arange(n) + 0.5) * dphi
    F = np.full(n, 1.0 / n)  # unit total emission
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    x = Phi * lam2
    expected = np.abs(np.sin(x) / x)
    keep = expected > 0.05
    np.testing.assert_allclose(np.abs(P)[keep], expected[keep], rtol=1e-3)


def test_gaussian_is_burn():
    """F Gaussian width sigma -> |P| = exp(-2 sigma^2 lam2^2).

    sigma=0.02: the exponent is 2 sigma^2 (lam2)^2 ~ 8.0 (O(1)), keeping
    expected ~ 3e-04 far above the ~5e-16 floor set by +-8 sigma truncation.
    """
    sigma = 0.02
    n = 1 << 15
    phi = np.linspace(-8 * sigma, 8 * sigma, n)
    F = np.exp(-0.5 * (phi / sigma) ** 2)
    F /= F.sum()
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    expected = np.exp(-2.0 * sigma**2 * lam2**2)
    np.testing.assert_allclose(np.abs(P), expected, rtol=1e-3)


def test_delay_power_recovers_a_single_depth():
    """delay_power inverts transform: peak at the injected depth."""
    phi0 = 120.0
    freqs = fine_freqs(30.0)[::16]  # 1024 points
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(0.0, 300.0, 0.25)
    p = dsp.delay_power(spec, freqs, phi_out)
    assert abs(phi_out[np.argmax(p)] - phi0) < 1.0
    assert np.isclose(p.max(), 1.0, rtol=1e-6)  # unit tone, normalized
