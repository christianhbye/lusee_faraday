"""Rotation-measure (RM) synthesis for LuSEE Faraday detection.

Operates on a flat channel set: per-channel frequency (MHz), lambda^2
(m^2), and Stokes Q/U. The complex polarization P = Q + iU rotates as
exp(2i * RM * lambda^2); RM synthesis is the matched transform that maps
P(lambda^2) to the Faraday spectrum F(phi).
"""

import numpy as np

C = 299792458.0  # speed of light, m/s


def lambda2(nu_mhz):
    """Wavelength squared (m^2) for frequencies given in MHz."""
    nu_mhz = np.asarray(nu_mhz, dtype=float)
    return (C / (nu_mhz * 1e6)) ** 2


def faraday_resolution(lam2):
    """FWHM of the RMSF main lobe (rad/m^2), set by lambda^2 coverage."""
    lam2 = np.asarray(lam2, dtype=float)
    return 2 * np.sqrt(3) / (lam2.max() - lam2.min())


def max_scale(lam2):
    """Largest recoverable Faraday-thick scale (rad/m^2)."""
    lam2 = np.asarray(lam2, dtype=float)
    return np.pi / lam2.min()


def phi_grid(lam2, phi_max, dphi=None, oversample=3):
    """Symmetric Faraday-depth grid on [-phi_max, phi_max].

    If dphi is None it defaults to faraday_resolution / oversample.
    """
    lam2 = np.asarray(lam2, dtype=float)
    if dphi is None:
        dphi = faraday_resolution(lam2) / oversample
    # round up so the actual linspace spacing never exceeds dphi
    n = int(np.ceil(2 * phi_max / dphi)) + 1
    return np.linspace(-phi_max, phi_max, n)


def _normalized_weights(lam2, weights):
    if weights is None:
        weights = np.ones_like(lam2)
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def rmsf(lam2, phi, weights=None):
    """Rotation-measure spread function R(phi).

    R(phi) = sum_k w_k exp(-2i phi (lam2_k - lam2_ref)) / sum_k w_k
    """
    lam2 = np.asarray(lam2, dtype=float)
    phi = np.asarray(phi, dtype=float)
    w = _normalized_weights(lam2, weights)
    lam2_ref = np.sum(w * lam2)
    kernel = np.exp(-2j * np.outer(phi, lam2 - lam2_ref))
    return kernel @ w
