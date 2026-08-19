"""Pinned conventions for the LuSEE Faraday simulation.

Every convention conversion in the package funnels through this module.
The two that matter:

- Sky Q/U maps are healpy/COSMO on input; croissant consumes IAU, and
  ``U_IAU = -U_COSMO``.
- Faraday rotation is ``(Q + iU)_COSMO -> (Q + iU)_COSMO e^{+2i phi l^2}``.

Combining them gives the result the whole refactor rests on.  croissant's
harmonic dual holds ``P_MINUS`` = the spin -2 analysis of ``Q + iU`` and
``P_PLUS`` = the spin +2 analysis of ``Q - iU``, both in IAU.  Since
``(Q + iU)_IAU = conj((Q + iU)_COSMO)`` for real maps, Faraday rotation
multiplies the P_MINUS block by ``e^{-2i phi l^2}`` and the P_PLUS block
by its conjugate -- it is diagonal in the dual, so a region of constant
Faraday depth needs one component and a per-block coefficient.
"""

import numpy as np
from scipy.constants import c as C_LIGHT

PORT_NAMES = ("N", "E", "S", "W")
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
DUAL_BLOCKS = ("I", "V", "P_MINUS", "P_PLUS")


def _product_labels():
    labels = []
    for a, b in PORT_PAIRS:
        if a == b:
            labels.append(f"{a}{b}R")
        else:
            labels.extend((f"{a}{b}R", f"{a}{b}I"))
    return tuple(labels)


PRODUCT_LABELS = _product_labels()


def lambda_squared(freq_mhz):
    """Wavelength squared in m^2 for frequencies in MHz."""
    return (
        C_LIGHT / (np.atleast_1d(np.asarray(freq_mhz, dtype=float)) * 1e6)
    ) ** 2


def cosmo_to_iau_qu(Q, U):
    """healpy/COSMO (Q, U) -> IAU (Q, U)."""
    return np.asarray(Q), -np.asarray(U)


def iau_to_cosmo_qu(Q, U):
    """IAU (Q, U) -> healpy/COSMO (Q, U)."""
    return np.asarray(Q), -np.asarray(U)


def faraday_phase_cosmo(phi_fd, freq_mhz):
    """Factor multiplying ``(Q + iU)_COSMO``; shape ``(..., nfreq)``."""
    phi = np.asarray(phi_fd, dtype=float)
    lam2 = lambda_squared(freq_mhz)
    return np.exp(2j * phi[..., None] * lam2)


def dual_block_phase(phi_fd, freq_mhz):
    """Per-dual-block Faraday coefficients; shape ``(..., nfreq, 4)``.

    Blocks are ordered as :data:`DUAL_BLOCKS`.  The spin-0 blocks are
    untouched; ``P_MINUS`` picks up the conjugate of the COSMO phase and
    ``P_PLUS`` the phase itself.
    """
    cosmo = faraday_phase_cosmo(phi_fd, freq_mhz)
    ones = np.ones_like(cosmo)
    return np.stack([ones, ones, np.conj(cosmo), cosmo], axis=-1)
