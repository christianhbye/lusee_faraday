"""Covariance assembly, entirely on luseepy's instrument physics.

This module owns the wiring, not the physics.  The order is fixed by
``lusee.FullStokesSimulatorBase.simulate``: assemble the open covariance
from the sky pair integrals plus the Moon and antenna-metal terms, apply
the receiver loading, project onto the Hermitian part, then pack the 16
real science channels.

``Z_A`` and ``Z_L`` are evaluated on whatever frequency grid the caller
passes, which for a Faraday run is the fine grid.  The fixed-beam
approximation applies to the response alms only, so receiver loading is
not smeared along with it.
"""

import numpy as np

from .conventions import PORT_PAIRS, PRODUCT_LABELS


def covariance(
    pair_integrals,
    resp,
    receiver,
    freqs_mhz,
    T_moon=250.0,
    T_ant=0.0,
):
    """Loaded, Hermitian port covariance -> ``(ntime, nfreq, 4, 4)``."""
    from lusee.Covariance import (
        apply_receiver_loading,
        assemble_open_covariance,
        project_hermitian,
    )

    freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
    ZA, _, Rmoon, Rloss, _ = resp.target_matrices(freqs)
    open_cov = assemble_open_covariance(
        np.asarray(pair_integrals),
        Rmoon,
        Rloss,
        T_moon=T_moon,
        T_ant=T_ant,
    )
    ZL = receiver.Z(freqs)
    unprojected, _ = apply_receiver_loading(open_cov, ZA, ZL)
    return np.asarray(project_hermitian(unprojected))


def blackbody_normalization(resp, receiver, freqs_mhz):
    """Covariance response to a one-kelvin blackbody enclosure."""
    from lusee.Covariance import (
        blackbody_normalization as _blackbody,
        loading_matrix,
    )

    freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
    ZA, _, _, _, _ = resp.target_matrices(freqs)
    M = loading_matrix(ZA, receiver.Z(freqs))
    return np.asarray(_blackbody(ZA, M))


def channels(covariance_matrix, products="all"):
    """Hermitian covariance -> 16 real channels plus their labels."""
    C = np.asarray(covariance_matrix)
    if products != "all":
        raise ValueError("only products='all' is supported")
    out = np.empty(C.shape[:-2] + (16,), dtype=np.float64)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            out[..., k] = C[..., a, b].real
            k += 1
        else:
            out[..., k] = C[..., a, b].real
            out[..., k + 1] = C[..., a, b].imag
            k += 2
    return out, PRODUCT_LABELS


def unpack_channels(packed):
    """16 real channels -> Hermitian covariance ``(..., 4, 4)``."""
    ch = np.asarray(packed)
    C = np.zeros(ch.shape[:-1] + (4, 4), dtype=complex)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            C[..., a, b] = ch[..., k]
            k += 1
        else:
            C[..., a, b] = ch[..., k] + 1j * ch[..., k + 1]
            C[..., b, a] = np.conj(C[..., a, b])
            k += 2
    return C
