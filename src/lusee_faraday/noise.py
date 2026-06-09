"""Radiometer noise for LuSEE polarized spectra.

sigma = T_sys / sqrt(dnu * dt). T_sys is sky-dominated (~ Stokes I).
"""

import numpy as np


def radiometer_sigma(T_sys, dnu_hz, dt_s):
    """Radiometer noise std (same units as T_sys)."""
    T_sys = np.asarray(T_sys, dtype=float)
    dnu_hz = np.asarray(dnu_hz, dtype=float)
    return T_sys / np.sqrt(dnu_hz * dt_s)


def add_noise(stokes, sigma, rng):
    """Add Gaussian noise of std `sigma` to a Stokes array.

    `sigma` may be a scalar or broadcastable to `stokes.shape`. `rng`
    is a numpy Generator (e.g. np.random.default_rng(seed)).
    """
    stokes = np.asarray(stokes, dtype=float)
    return stokes + rng.normal(scale=sigma, size=stokes.shape)
