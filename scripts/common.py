"""Shared configuration for the four-port Faraday analysis."""

import os

# jax reads this at import time, and every script in this directory
# imports common before it imports anything that pulls jax in.  Without
# it croissant and luseepy silently drop to complex64 -- croissant even
# says so on stderr -- and the covariance, the Loewdin G^{-1/2} and the
# cached zenith weights all come out at ~1e-7 precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lusee_faraday.config import (  # noqa: E402,F401
    BETA_I,
    BETA_QU,
    FINE_STEP_MHZ,
    FREQ_REF_I,
    FREQ_REF_QU,
    LUN_LAT_DEG,
    LUN_LONG_DEG,
    MAP_NSIDE,
    N_FINE,
    N_TIMES,
    PHI_FD_POINT,
    SIDEREAL_DAY_S,
    T_CMB,
    T_START_UTC,
    fine_freqs,
    lam2,
    moon_location,
    parent_centers,
    times,
)

DATA_DIR = REPO / "data"
GEN_DIR = REPO / "generated_data"
CACHE_DIR = GEN_DIR / "cache"
FIG_DIR = REPO / "report" / "figures"
for _d in (GEN_DIR, CACHE_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Four-port response artifact.  Defaults to the as-built (asymmetric,
# fully coupled) BGL_v16 model -- the same file the luseepy-version
# branch used.  Override with LUSEE_RESPONSE to run the ablations:
#   _c4sym.fits  -> C4 group-averaged (the paper's 90deg-rotation
#                   assumption made self-consistent)
#   _diagza.fits -> ZA-diagonalised (inter-port coupling removed)
RESPONSE_DIR = DATA_DIR / "BGL_v16"
RESPONSE_PATH = os.environ.get(
    "LUSEE_RESPONSE", str(RESPONSE_DIR / "lusee_bgl_v16_response_v3.fits")
)

C_LIGHT = 299792458.0


def rotation_matrices(force=False):
    """Cached galactic->response-frame rotation matrices (T, 3, 3)."""
    from lusee_faraday.conventions import topo_rotation_matrix

    cache = CACHE_DIR / "rotation_matrices.npy"
    if cache.exists() and not force:
        R = np.load(cache)
        if R.shape == (N_TIMES, 3, 3):
            return R
    loc = moon_location()
    tt = times()
    R = np.empty((N_TIMES, 3, 3))
    for i, t in enumerate(tt):
        R[i] = topo_rotation_matrix(t, loc)
        if (i + 1) % 128 == 0:
            print(f"  rotation matrices {i + 1}/{N_TIMES}", flush=True)
    np.save(cache, R)
    return R


def load_sky_maps():
    """Reference galactic maps: Haslam 408 (I), WMAP K (Q, U), RM.

    Returns dict with I408 (Tcmb subtracted), Q23, U23 (K, healpy/COSMO
    convention, K band in K_RJ), RM (rad/m^2), all RING at their NATIVE
    resolution.  Do not degrade the maps: the pixel-space engine sums
    over full-size maps (per-pixel Faraday phases do not commute with
    ud_grade averaging); harmonic paths only need ell <= 30 anyway and
    should apply that lmax to the full-size maps.  All three inputs are
    natively nside=512 RING on a common grid.
    """
    import healpy as hp
    import h5py
    from astropy.io import fits

    cache = CACHE_DIR / "sky_maps_native.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    # Haslam 408 MHz destriped/desourced (Remazeilles 2014), in K.
    # NOTE: this file is RING ordered (header ORDERING=RING) — unlike
    # WMAP, no reorder!
    with fits.open(DATA_DIR / "haslam408_dsds_Remazeilles2014.fits") as h:
        I408 = h[1].data["TEMPERATURE"].ravel().astype(np.float64)

    # WMAP K band, mK thermodynamic -> K_RJ at 23 GHz
    x = 6.62607015e-34 * 23e9 / (1.380649e-23 * T_CMB)
    fconv = x**2 * np.exp(x) / (np.exp(x) - 1) ** 2
    with fits.open(DATA_DIR / "wmap_band_iqumap_r9_9yr_K_v5.fits") as h:
        d = h["Stokes Maps"].data
        Q = d["Q_POLARISATION"].astype(np.float64) * 1e-3 * fconv
        U = d["U_POLARISATION"].astype(np.float64) * 1e-3 * fconv
    Q23 = hp.reorder(Q, n2r=True)
    U23 = hp.reorder(U, n2r=True)

    with h5py.File(DATA_DIR / "faraday2020v2.hdf5", "r") as f:
        RM = f["faraday_sky_mean"][:]  # RING already

    out = {"I408": I408 - T_CMB, "Q23": Q23, "U23": U23, "RM": RM}
    np.savez(cache, **out)
    return out


def sky_at_freq(maps, freq_mhz):
    """Scale reference maps to `freq_mhz` -> (I, Q, U) in K."""
    I = maps["I408"] * (freq_mhz / FREQ_REF_I) ** BETA_I + T_CMB
    s = (freq_mhz / FREQ_REF_QU) ** BETA_QU
    return I, maps["Q23"] * s, maps["U23"] * s
