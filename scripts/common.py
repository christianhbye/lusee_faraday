"""Shared configuration for the four-port Faraday analysis."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA_DIR = REPO / "data"
GEN_DIR = REPO / "generated_data"
CACHE_DIR = GEN_DIR / "cache"
FIG_DIR = REPO / "report" / "figures"
for _d in (GEN_DIR, CACHE_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESPONSE_PATH = (
    "/home/anze/work/lusee/Drive/Simulations/BeamModels/BGL_v16/"
    "lusee_bgl_v16_response_v3.fits"
)

# LuSEE-Night landing site = lusee.Observation defaults
LUN_LAT_DEG = -23.814
LUN_LONG_DEG = 182.258

# Time grid: 1024 samples over exactly one lunar sidereal day so that
# the time axis is periodic (easy FFT).
N_TIMES = 1024
SIDEREAL_DAY_S = 27.321661 * 86400.0
T_START_UTC = "2027-01-01 09:00:00"

# Fine frequency grid: +-4 parent bins around the center, spaced at
# 25 kHz / 2048.
FINE_STEP_MHZ = 25e-3 / 2048
N_FINE = 16384  # spans +-0.1 MHz

# Real sky maps are used at their native resolution (nside=512); no
# degradation anywhere.  Synthetic validation skies pick their own nside.
MAP_NSIDE = 512

# Sky model spectral parameters (as in the paper)
BETA_I = -2.55
FREQ_REF_I = 408.0  # MHz (Haslam)
BETA_QU = -2.8
FREQ_REF_QU = 23e3  # MHz (WMAP K band)
T_CMB = 2.7255

# Faraday depth of the single-source toy example (paper value)
PHI_FD_POINT = 250.0  # rad/m^2

C_LIGHT = 299792458.0


def times():
    """1024 astropy Times covering one lunar sidereal day."""
    import astropy.units as u
    from lunarsky.time import Time

    t0 = Time(T_START_UTC)
    dt = SIDEREAL_DAY_S / N_TIMES
    return t0 + np.arange(N_TIMES) * dt * u.s


def moon_location():
    from lunarsky import MoonLocation

    return MoonLocation.from_selenodetic(
        lon=LUN_LONG_DEG, lat=LUN_LAT_DEG, height=0.0
    )


def fine_freqs(center_mhz):
    """16384 fine frequencies (MHz) spanning +-0.1 MHz around center."""
    k = np.arange(N_FINE) - N_FINE // 2
    return center_mhz + k * FINE_STEP_MHZ


def parent_centers(center_mhz):
    """The three parent 25 kHz bins fully covered by the fine grid."""
    return np.array([center_mhz - 0.025, center_mhz, center_mhz + 0.025])


def lam2(freq_mhz):
    return (C_LIGHT / (np.asarray(freq_mhz) * 1e6)) ** 2


def rotation_matrices(force=False):
    """Cached galactic->response-frame rotation matrices (T, 3, 3)."""
    from lusee_faraday.fourport import topo_rotation_matrix

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
