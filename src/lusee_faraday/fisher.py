"""Fisher-matrix detection forecast for the Faraday amplitude alpha.

Data: complex polarized channels P = pQ + i*pU over (time, channel) with
independent Gaussian noise sigma on pQ and pU. Parameters: a Faraday
amplitude alpha (detection target, fiducial 1), intrinsic-sky nuisance
amplitudes (spin-2 harmonic modes), and one effective Faraday-dispersion
nuisance tau (Faraday variance). The sky-marginalized error sigma(alpha)
gives the realistic detection SNR = alpha_fid / sigma(alpha). A Fisher
forecast needs only J and N, never the data vector.
"""

import numpy as np

from .forward import pol_response


def stack_real(P):
    """Flatten complex (ntimes, nchan) to real vector [Re..., Im...]."""
    P = np.asarray(P)
    return np.concatenate([P.real.ravel(), P.imag.ravel()])


def fisher_matrix(columns, sigma):
    """F_ij = sum Re(dP_i* dP_j)/sigma^2.

    columns: list of complex (ntimes, nchan) derivative arrays.
    sigma: real (ntimes, nchan) per-channel Stokes noise (pQ and pU
    share it). Returns (nparam, nparam) real array.
    """
    sig = np.asarray(sigma, dtype=float)
    w = np.concatenate([1.0 / sig.ravel() ** 2] * 2)  # Re + Im quadratures
    J = np.column_stack([stack_real(c) for c in columns])
    return J.T @ (w[:, None] * J)


def marginal_error(F, idx, rcond=1e-12):
    """Marginalized 1-sigma error on parameter idx (others free)."""
    Cinv = np.linalg.pinv(F, rcond=rcond)
    return float(np.sqrt(Cinv[idx, idx]))


def detection_snr(F, idx, alpha_fid=1.0, rcond=1e-12):
    """Detection SNR = alpha_fid / marginalized sigma(alpha)."""
    return alpha_fid / marginal_error(F, idx, rcond=rcond)
