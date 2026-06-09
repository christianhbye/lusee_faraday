"""Low-l spin-2 (Q, U) spherical-harmonic basis for the intrinsic
polarized-sky nuisance in the Fisher forecast.

Each element is a real (Q, U) map pair from a unit E- or B-mode a_lm.
These span the large-scale intrinsic polarization the beam can
constrain; their amplitudes are the marginalized nuisance parameters.
Absolute normalization is irrelevant: the marginalized sigma(alpha)
depends only on the subspace the basis spans.
"""

import healpy as hp
import numpy as np


def n_modes(lmax):
    """Number of real spin-2 basis maps for 2 <= l <= lmax."""
    return sum(2 * (1 + 2 * L) for L in range(2, lmax + 1))


def spin2_basis(nside, lmax):
    """List of (label, Q_map, U_map) real basis elements (RING)."""
    nalm = hp.Alm.getsize(lmax)
    T = np.zeros(nalm, dtype=complex)
    basis = []
    for L in range(2, lmax + 1):
        for M in range(L + 1):
            idx = hp.Alm.getidx(lmax, L, M)
            parts = ("re",) if M == 0 else ("re", "im")
            for mode in ("E", "B"):
                for part in parts:
                    alm = np.zeros(nalm, dtype=complex)
                    alm[idx] = 1.0 if part == "re" else 1.0j
                    if mode == "E":
                        eb = [T, alm, np.zeros(nalm, dtype=complex)]
                    else:
                        eb = [T, np.zeros(nalm, dtype=complex), alm]
                    _, Q, U = hp.alm2map(eb, nside, lmax=lmax, pol=True)
                    basis.append((f"{mode}_{L}_{M}_{part}", Q, U))
    return basis
