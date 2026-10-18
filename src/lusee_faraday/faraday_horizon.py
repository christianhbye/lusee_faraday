"""Constant-rate Faraday horizon scenarios, with explicit phase convention.

This arithmetic does not constrain the angular screen statistics or local
synchrotron emissivity. A zero line-of-sight field gives no finite horizon.
"""

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, parsec

from .config import BETA_QU
from .conventions import lambda_squared

# SI rotation measure coefficient, converted to cm^-3, microgauss and pc.
ROTATION_PER_PC = (
    e**3 / (8 * np.pi**2 * epsilon_0 * m_e**2 * c**3) * 1e6 * 1e-10 * parsec
)


def faraday_rate(ne_cm3, b_parallel_ug):
    """Signed dPhi/dL in rad m^-2 pc^-1 for a uniform density/field."""
    ne, b = np.broadcast_arrays(
        np.asarray(ne_cm3, float), np.asarray(b_parallel_ug, float)
    )
    if not np.all(np.isfinite(ne) & (ne >= 0) & np.isfinite(b)):
        raise ValueError("density must be nonnegative; inputs must be finite")
    return ROTATION_PER_PC * ne * b


def horizon_depth(freq_mhz, ne_cm3, b_parallel_ug, phase_limit_rad=1.0):
    """Path in pc at which |2 lambda² Phi| reaches phase_limit_rad.

    Frequencies occupy the final axis. Limits of 1 and 2 rad distinguish a
    one-radian polarization phase from the framing's Phi ~ 1/lambda².
    Infinite results are meaningful and must not be replaced by a finite cap.
    """
    freq = np.atleast_1d(np.asarray(freq_mhz, float))
    if freq.ndim != 1 or not np.all(np.isfinite(freq) & (freq > 0)):
        raise ValueError("frequencies must be a positive finite MHz vector")
    if not np.isfinite(phase_limit_rad) or phase_limit_rad <= 0:
        raise ValueError("phase limit must be positive and finite")
    rate = np.abs(faraday_rate(ne_cm3, b_parallel_ug))[..., None]
    denominator = 2 * rate * lambda_squared(freq)
    depth = np.full(denominator.shape, np.inf)
    np.divide(phase_limit_rad, denominator, out=depth, where=denominator > 0)
    return depth


def unrotated_brightness(
    freq_mhz,
    horizon_pc,
    emitting_path_pc,
    emissivity_ref_k_per_pc,
    ref_mhz=30.0,
    beta=BETA_QU,
):
    """Unrotated emission within min(horizon, assumed emitting path), in K.

    A power law is an explicitly illustrative normalization, especially below
    20 MHz. This is unrotated brightness, not a local measurement.
    """
    freq = np.atleast_1d(np.asarray(freq_mhz, float))
    depth = np.asarray(horizon_pc, float)
    if not np.all(np.isfinite(freq) & (freq > 0)) or not np.all(
        (depth >= 0) & ~np.isnan(depth)
    ):
        raise ValueError("invalid frequency or horizon")
    if (
        not all(
            np.isfinite(v)
            for v in (
                emitting_path_pc,
                emissivity_ref_k_per_pc,
                ref_mhz,
                beta,
            )
        )
        or min(emitting_path_pc, ref_mhz) <= 0
        or emissivity_ref_k_per_pc < 0
    ):
        raise ValueError("invalid emitting path, emissivity or spectrum")
    return (
        np.minimum(depth, emitting_path_pc)
        * emissivity_ref_k_per_pc
        * (freq / ref_mhz) ** beta
    )
