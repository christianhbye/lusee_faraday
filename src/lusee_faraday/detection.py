"""Faraday-spectrum detection statistics.

For per-channel Stokes noise std sigma_k and weights w_k, the weighted
Faraday spectrum F(phi) = sum_k (w_k/sum w) (Q+iU)_k exp(-2i phi dl2_k).
With independent Gaussian noise on Q and U, the noise on F is complex
with per-quadrature std sigma_F = sqrt(sum (w_k sigma_k)^2) / sum w_k
(phi-independent). SNR = |F|_peak / sigma_F.
"""

import numpy as np

from .rmsynth import faraday_spectrum


def faraday_noise_std(sigma, weights=None):
    """Per-quadrature noise std of the weighted Faraday spectrum."""
    sigma = np.asarray(sigma, dtype=float)
    if weights is None:
        weights = np.ones_like(sigma)
    w = np.asarray(weights, dtype=float)
    return np.sqrt(np.sum((w * sigma) ** 2)) / w.sum()


def faraday_snr(Q, U, lam2, sigma, phi, weights=None):
    """SNR of the Faraday-spectrum peak vs the analytic noise level.

    Returns (snr, peak, noise_std).
    """
    F = faraday_spectrum(Q, U, lam2, phi, weights=weights)[0]
    peak = float(np.abs(F).max())
    noise_std = faraday_noise_std(sigma, weights=weights)
    return peak / noise_std, peak, noise_std
