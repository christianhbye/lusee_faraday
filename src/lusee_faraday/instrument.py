"""Covariance assembly, entirely on luseepy's instrument physics.

This module owns the wiring, not the physics.  The order is fixed by
``lusee.FullStokesSimulatorBase.simulate``: assemble the open covariance
from the sky pair integrals plus the Moon and antenna-metal terms, apply
the receiver loading, project onto the Hermitian part, then pack the 16
real science channels.

``Z_A`` and ``Z_L`` are evaluated on whatever frequency grid the caller
passes -- unless ``impedance_freq_mhz`` freezes them at one frequency.
A Faraday run must freeze them.  The antenna is near resonance at
30 MHz: one 0.5 MHz native step moves ``Z_A`` by 12%, and letting the
impedances follow the +-0.1 MHz fine grid moves the loading matrix by
11% across the band.  That is a smooth chromatic ramp of exactly the
kind the step-1 delay-space argument asserts is absent, so the
fixed-beam approximation has to cover the receiver loading too, not
only the response alms.
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
    impedance_freq_mhz=None,
):
    """Loaded, Hermitian port covariance -> ``(ntime, nfreq, 4, 4)``.

    ``impedance_freq_mhz`` evaluates ``Z_A``, ``Z_L``, ``R_moon`` and
    ``R_loss`` at that single frequency and broadcasts them along the
    ``nfreq`` axis: the fixed-beam freeze becomes visible at the call
    site, and it costs one ``target_matrices`` call rather than one over
    the whole fine grid.  An off-native ``impedance_freq_mhz`` is
    linearly interpolated between the neighbouring native channels
    rather than rejected.  That differs on purpose from
    ``response.FixedChannelKernel``, which asserts a native channel:
    freezing the *loading* somewhere between channels is a legitimate
    thing to ask for, whereas interpolating the beam would smear the
    response the fixed-beam approximation is supposed to hold fixed.
    """
    from lusee.Covariance import (
        apply_receiver_loading,
        assemble_open_covariance,
        project_hermitian,
    )

    freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
    frozen = impedance_freq_mhz is not None
    grid = (
        np.array([float(impedance_freq_mhz)], dtype=float) if frozen else freqs
    )
    ZA, _, Rmoon, Rloss, _ = resp.target_matrices(grid)
    ZL = receiver.Z(grid)
    if frozen:
        shape = (freqs.size, 4, 4)
        ZA, Rmoon, Rloss, ZL = (
            np.broadcast_to(np.asarray(m), shape)
            for m in (ZA, Rmoon, Rloss, ZL)
        )
    open_cov = assemble_open_covariance(
        np.asarray(pair_integrals),
        Rmoon,
        Rloss,
        T_moon=T_moon,
        T_ant=T_ant,
    )
    unprojected, _ = apply_receiver_loading(open_cov, ZA, ZL)
    return np.asarray(project_hermitian(unprojected))


def blackbody_normalization(resp, receiver, freqs_mhz):
    """Covariance response to a one-kelvin blackbody enclosure.

    There is no ``impedance_freq_mhz`` here: ``Z_A`` and the loading
    matrix always follow ``freqs_mhz``.  Normalizing a frozen-beam
    covariance (see :func:`covariance`) by this unfrozen blackbody would
    put the ~11% chromatic ramp the freeze removed straight back in, so
    a frozen caller must freeze this grid itself by passing a constant
    ``freqs_mhz``.
    """
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
