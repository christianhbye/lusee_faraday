"""Pinned configuration for the LuSEE Faraday analysis.

These values were previously duplicated in ``scripts/common.py``.  They
are documented in ``AGENTS.md`` under "Pinned conventions"; do not change
one without changing that document.
"""

import numpy as np

from .conventions import lambda_squared

# LuSEE-Night landing site == lusee.Observation defaults
LUN_LAT_DEG = -23.814
LUN_LONG_DEG = 182.258

# 1024 samples over exactly one lunar sidereal day: the time axis is
# periodic, so the observation-time FFT needs no window.
N_TIMES = 1024
SIDEREAL_DAY_S = 27.321661 * 86400.0
T_START_UTC = "2027-01-01 09:00:00"

# Fine frequency grid: +-4 parent bins around the center at 25 kHz / 2048.
FINE_STEP_MHZ = 25e-3 / 2048
N_FINE = 16384

MAP_NSIDE = 512
BAND_CENTERS_MHZ = (30.0, 10.0, 50.0)

# Sky spectral parameters (as in the paper)
BETA_I = -2.55
FREQ_REF_I = 408.0  # MHz, Haslam
BETA_QU = -2.8
FREQ_REF_QU = 23e3  # MHz, WMAP K band
T_CMB = 2.7255

# Faraday depth of the single-source toy example (paper value)
PHI_FD_POINT = 250.0  # rad/m^2


def times():
    """``N_TIMES`` astropy Times covering one lunar sidereal day."""
    import astropy.units as u
    from lunarsky.time import Time

    t0 = Time(T_START_UTC)
    dt = SIDEREAL_DAY_S / N_TIMES
    return t0 + np.arange(N_TIMES) * dt * u.s


def moon_location():
    """The LuSEE-Night landing site."""
    from lunarsky import MoonLocation

    return MoonLocation.from_selenodetic(
        lon=LUN_LONG_DEG, lat=LUN_LAT_DEG, height=0.0
    )


def fine_freqs(center_mhz):
    """``N_FINE`` fine frequencies (MHz) spanning +-0.1 MHz around center."""
    k = np.arange(N_FINE) - N_FINE // 2
    return center_mhz + k * FINE_STEP_MHZ


def parent_centers(center_mhz):
    """The three parent 25 kHz bins fully covered by the fine grid."""
    return np.array([center_mhz - 0.025, center_mhz, center_mhz + 0.025])


def lam2(freq_mhz):
    """Convenience alias for :func:`conventions.lambda_squared`."""
    return lambda_squared(freq_mhz)
