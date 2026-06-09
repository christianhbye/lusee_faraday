"""Optimized Faraday rotation simulation.

Exploits the commutativity of Faraday rotation (SO(2) on Q,U) with
coordinate rotation (also SO(2) on Q,U via parallel transport) to
separate the expensive coordinate rotation from the frequency-dependent
Faraday rotation.

The approach:
1. Rotate reference sky maps (I, Q, U) and the RM map to topocentric
   once per time step -- O(N_times) rotations total.
2. For each frequency, apply power-law scaling and Faraday rotation in
   topocentric frame, then convolve with beam weights -- O(N_freq * N_pix)
   per time step but with no spherical harmonic transforms.

The beam-weighted polarization sum factorizes as:

    Rxx_pol(f) = scale_QU(f) * [A_x . cos(2*RM*lam^2) + B_x . sin(...)]

where A_x(p) = wQ_x*Q_topo + wU_x*U_topo and B_x(p) = wU_x*Q_topo -
wQ_x*U_topo are precomputed per time step.  The dot products are
evaluated as BLAS matrix-vector products in frequency batches.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import healpy as hp
import numpy as np
from lunarsky import LunarTopo

from .healpix import HealpixGrid
from .rotations import get_rot_mat, rotmat_to_eulerZYX


def precompute_rotated_maps(
    I_ref, Q_ref, U_ref, rm_gal, times, nside, site_loc,
):
    """Rotate reference sky maps and RM to topocentric for all time steps.

    Parameters
    ----------
    I_ref : ndarray, shape (npix,)
        Reference Stokes I map in Galactic frame.
    Q_ref : ndarray, shape (npix,)
        Reference Stokes Q map in Galactic frame.
    U_ref : ndarray, shape (npix,)
        Reference Stokes U map in Galactic frame.
    rm_gal : ndarray, shape (npix,)
        Rotation measure map in Galactic frame (rad/m^2).
    times : array-like
        Observation times (astropy Time objects).
    nside : int
        HEALPix nside parameter.
    site_loc : MoonLocation
        Observer location on the Moon.

    Returns
    -------
    I_topo : ndarray, shape (ntimes, npix)
    Q_topo : ndarray, shape (ntimes, npix)
    U_topo : ndarray, shape (ntimes, npix)
    rm_topo : ndarray, shape (ntimes, npix)

    """
    ntimes = len(times)
    npix = hp.nside2npix(nside)

    I_topo = np.zeros((ntimes, npix))
    Q_topo = np.zeros((ntimes, npix))
    U_topo = np.zeros((ntimes, npix))
    rm_topo = np.zeros((ntimes, npix))

    for i, t in enumerate(times):
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Rotating time step {i + 1}/{ntimes}"
            )
        topo = LunarTopo(location=site_loc, obstime=t)
        rmat = get_rot_mat(topo)
        euler = rotmat_to_eulerZYX(rmat)
        rot = hp.Rotator(
            rot=euler, deg=False, eulertype="ZYX"
        )

        # Polarized rotation of reference maps (spin-2 via SHT)
        It, Qt, Ut = rot.rotate_map_alms(
            [I_ref, Q_ref, U_ref]
        )
        I_topo[i] = It
        Q_topo[i] = Qt
        U_topo[i] = Ut

        # Scalar rotation of RM map via pixel interpolation.
        # SHT-based rotation introduces small RM errors that get
        # amplified by 2*lambda^2, so we use healpy's built-in
        # pixel-level map rotation instead.
        rm_topo[i] = rot.rotate_map_pixel(rm_gal)

    return I_topo, Q_topo, U_topo, rm_topo


def compute_vis_fast(
    I_topo,
    Q_topo,
    U_topo,
    rm_topo,
    beam,
    freqs,
    mask,
    freq_ref_I=50,
    beta_I=-2.55,
    freq_ref_QU=23e3,
    beta_QU=-2.8,
    cmb=2.725,
    batch_size=32,
):
    """Compute visibilities with Faraday rotation (optimized).

    Uses precomputed topocentric reference maps.  For each time step
    the beam-weighted Faraday-rotated visibility is decomposed as::

        R(f) = R_I(f) + scale_QU(f) * [A . cos(phi) + B . sin(phi)]

    where *A* and *B* are beam-weighted pixel arrays precomputed per
    time step, and phi(f, p) = 2 * RM(p) * lambda^2(f).

    Parameters
    ----------
    I_topo : ndarray, shape (ntimes, npix)
        Rotated reference I maps (at *freq_ref_I* MHz).
    Q_topo : ndarray, shape (ntimes, npix)
        Rotated reference Q maps (at *freq_ref_QU* MHz).
    U_topo : ndarray, shape (ntimes, npix)
        Rotated reference U maps (at *freq_ref_QU* MHz).
    rm_topo : ndarray, shape (ntimes, npix)
        Rotated RM maps (rad/m^2).
    beam : Beam
        Beam with precomputed weights.
    freqs : ndarray, shape (nfreq,)
        Simulation frequencies in MHz.
    mask : ndarray, shape (npix,)
        Boolean horizon mask.
    freq_ref_I : float
        Reference frequency for I scaling (MHz).
    beta_I : float
        Spectral index for I.
    freq_ref_QU : float
        Reference frequency for Q/U scaling (MHz).
    beta_QU : float
        Spectral index for Q/U.
    cmb : float
        CMB temperature (K).
    batch_size : int
        Number of frequencies per batch.

    Returns
    -------
    vis : ndarray, shape (ntimes, 3, nfreq)
        Visibility matrix [Rxx, Ryy, Rxy].  Rxy is stored as its
        real part only, matching the ``Simulator`` convention.

    """
    ntimes = I_topo.shape[0]
    nfreq = len(freqs)

    scale_I = (freqs / freq_ref_I) ** beta_I
    scale_QU = (freqs / freq_ref_QU) ** beta_QU
    lambda_sq = (3e8 / (freqs * 1e6)) ** 2

    w = beam.weights
    m = mask.astype(float)
    norm = (
        np.sum(w["wI_x"] * m) + np.sum(w["wI_y"] * m)
    )

    vis = np.zeros((ntimes, 3, nfreq))
    pols = ("x", "y", "xy")

    keep = mask.astype(bool)
    nkeep = int(keep.sum())

    for i in range(ntimes):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Visibilities: time step {i + 1}/{ntimes}")

        I_m = (I_topo[i] - cmb) * m
        Q_m = Q_topo[i] * m
        U_m = U_topo[i] * m
        cmb_m = cmb * m

        # I contribution (frequency-independent shape; full-pixel sums)
        I_contrib = np.zeros((3, nfreq), dtype=complex)
        for k, pol in enumerate(pols):
            sII = np.sum(w[f"wI_{pol}"] * I_m)
            sIc = np.sum(w[f"wI_{pol}"] * cmb_m)
            I_contrib[k] = scale_I * sII + sIc

        # A, B only over above-horizon pixels (below-horizon are zero)
        rm_i = rm_topo[i][keep]
        # complex: the xy cross-pol weights (wQ_xy, wU_xy) are complex
        A = np.zeros((3, nkeep), dtype=complex)
        B = np.zeros((3, nkeep), dtype=complex)
        for k, pol in enumerate(pols):
            wQ = w[f"wQ_{pol}"][keep]
            wU = w[f"wU_{pol}"][keep]
            Qk = Q_m[keep]
            Uk = U_m[keep]
            A[k] = wQ * Qk + wU * Uk
            B[k] = wU * Qk - wQ * Uk

        for f0 in range(0, nfreq, batch_size):
            f1 = min(f0 + batch_size, nfreq)
            lsq = lambda_sq[f0:f1]
            phase = 2 * rm_i[None, :] * lsq[:, None]
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            for k in range(3):
                pol_contrib = cos_p @ A[k] + sin_p @ B[k]
                I_contrib[k, f0:f1] += scale_QU[f0:f1] * pol_contrib

        vis[i] = np.real(I_contrib) / norm

    return vis


def _vis_chunk(args):
    # ProcessPoolExecutor worker target; must stay module-level (picklable)
    I_t, Q_t, U_t, rm_t, beam, freqs, mask, kwargs = args
    return compute_vis_fast(
        I_t, Q_t, U_t, rm_t, beam, freqs, mask, **kwargs
    )


def compute_vis_fast_parallel(
    I_topo,
    Q_topo,
    U_topo,
    rm_topo,
    beam,
    freqs,
    mask,
    nproc=None,
    **kwargs,
):
    """Parallel ``compute_vis_fast`` over the time axis.

    Splits the ntimes axis into ``nproc`` chunks run in separate
    processes and concatenates the results.  nproc=None or 1 runs
    serially.  Extra kwargs are forwarded to compute_vis_fast.
    """
    ntimes = I_topo.shape[0]
    if nproc in (None, 1) or ntimes < 2:
        return compute_vis_fast(
            I_topo, Q_topo, U_topo, rm_topo,
            beam, freqs, mask, **kwargs,
        )
    chunks = np.array_split(np.arange(ntimes), nproc)
    args = [
        (
            I_topo[c], Q_topo[c], U_topo[c], rm_topo[c],
            beam, freqs, mask, kwargs,
        )
        for c in chunks
        if c.size
    ]
    # forkserver: plain fork of the multi-threaded numpy/healpy process
    # can deadlock the workers
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as ex:
        results = list(ex.map(_vis_chunk, args))
    return np.concatenate(results, axis=0)
