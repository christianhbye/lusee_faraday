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


def contract(
    pair_alms, component_alms, times, loc, lmax, sky_frame="galactic"
):
    """Rotate into a common frame and contract every component.

    Mirrors ``lusee.FullStokesCroSimulator._convolve``: the response is
    rotated from topocentric into MEPA at the first timestamp, the sky is
    rotated from its own frame into MEPA, and the remaining time
    dependence is the diagonal-in-m ``rot_alm_z`` phase.

    Parameters
    ----------
    pair_alms : (npair, 4, L, 2L-1) complex
        Response alms at one native channel, topocentric.
    component_alms : (K, 4, L, 2L-1) complex
        Frequency-independent sky components in ``sky_frame``.
    times : astropy Time array
    loc : lunarsky.MoonLocation
    lmax : int
    sky_frame : {"galactic", "mepa", "topo"}

    Returns
    -------
    (K, 4, ntime, npair) complex
    """
    import croissant as cro
    from lunarsky import LunarTopo

    beam = np.asarray(pair_alms)
    sky = np.asarray(component_alms)
    n_m = 2 * int(lmax) + 1

    if sky_frame == "topo":
        phases = np.ones((len(times), n_m), dtype=complex)
        return contract_blocks(beam, sky, phases)

    from lusee.spice_utils import ensure_lunarsky_moon_frame

    ensure_lunarsky_moon_frame()
    et = cro.rotations.jd_to_et(times[0].tdb.jd)
    topo = LunarTopo(obstime=times[0], location=loc)
    beam_rotation, beam_dl = cro.rotations.generate_euler_dl(
        int(lmax), topo, "mepa", et=et
    )
    beam_work = np.asarray(
        cro.rotations.rotate_alm(beam, beam_rotation, dl_array=beam_dl)
    )
    if sky_frame == "galactic":
        sky_work = np.asarray(cro.rotations.gal2mepa(sky, et=et))
    elif sky_frame == "mepa":
        sky_work = sky
    else:
        raise ValueError(f"unsupported sky frame {sky_frame!r}")

    elapsed = np.asarray(
        (times.tdb - times[0].tdb).to_value("s"), dtype=np.float64
    )
    phases = np.asarray(cro.simulator.rot_alm_z(int(lmax), times=elapsed))
    return contract_blocks(beam_work, sky_work, phases)


def expand(W, coeffs, chunk=None, out=None):
    """Apply per-frequency component coefficients to a contraction.

    ``V[t, f, p] = sum_k sum_c coeffs[k, f, c] * W[k, c, t, p]``.

    Chunked over frequency so a full run (1024 x 16384 x 10 complex,
    2.7 GB) can stream into a memmapped ``out``.
    """
    W = np.asarray(W)
    coeffs = np.asarray(coeffs)
    if W.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"component count mismatch: W has {W.shape[0]}, "
            f"coeffs has {coeffs.shape[0]}"
        )
    if W.shape[1] != coeffs.shape[2]:
        raise ValueError(
            f"dual-block count mismatch: W has {W.shape[1]}, "
            f"coeffs has {coeffs.shape[2]}"
        )
    ntime, npair, nfreq = W.shape[2], W.shape[3], coeffs.shape[1]
    if out is None:
        out = np.empty((ntime, nfreq, npair), dtype=complex)
    step = nfreq if chunk is None else int(chunk)
    for start in range(0, nfreq, step):
        stop = min(start + step, nfreq)
        out[:, start:stop] = np.einsum(
            "kctp,kfc->tfp",
            W,
            coeffs[:, start:stop],
            optimize=True,
        )
    return out
