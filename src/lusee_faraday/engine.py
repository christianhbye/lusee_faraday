"""Harmonic contraction and spectral expansion.

The refactor rests on one separation.  Faraday rotation is diagonal in
croissant's harmonic dual, so a sky is a small set of frequency-
independent component alms plus a per-frequency, per-block coefficient
matrix, and

    V(t, p, nu) = sum_k sum_c coeff[k, nu, c] * W[k, c, t, p]

The expensive part is ``W``: one contraction per component, independent
of how many frequency channels are wanted.  The 16,384-channel fine grid
is then a single einsum.
"""

import numpy as np


def contract_blocks(beam_alm, sky_alm, phases):
    """Contract sky and pair-response alms, keeping the dual-block axis.

    This is ``croissant.polarized_convolve`` with the block axis ``c``
    retained instead of summed, because a Faraday sky needs a different
    coefficient per block.  Summing the returned ``c`` axis reproduces
    ``polarized_convolve`` exactly (see the test).

    Parameters
    ----------
    beam_alm : (npair, 4, L, 2L-1) complex
        Pair-response alms at one frequency, already in the frame the
        contraction happens in.
    sky_alm : (K, 4, L, 2L-1) complex
        Component alms in the same frame.
    phases : (ntime, 2L-1) complex
        croissant's ``exp(-i m phi)`` time phases.

    Returns
    -------
    (K, 4, ntime, npair) complex
    """
    return np.einsum(
        "kclm,tm,pclm->kctp",
        np.conj(np.asarray(sky_alm)),
        np.asarray(phases),
        np.asarray(beam_alm),
        optimize=True,
    )
