"""Frequency-domain FFT of the complex polarization P = Q + iU.

An alternative to lambda^2 RM synthesis (see :mod:`rmsynth`): instead of
the matched transform in lambda^2, take a plain FFT of P along the
*uniform* frequency axis. The conjugate variable is a delay tau (s), NOT
Faraday depth -- because Faraday rotation winds the P phase as
lambda^2 ~ 1/nu^2, P(nu) is a chirp in nu and the FFT does not localize
RM. This is a quick-look transform whose axis is easy to read; it mirrors
:func:`rmsynth.faraday_spectrum` so the two transforms are drop-in
comparable on the same P. ``delay_grid``/``faraday_fft`` here are the
analogues of ``phi_grid``/``faraday_spectrum`` there.
"""

import numpy as np


def _uniform_dnu_hz(nu_mhz):
    """Channel spacing (Hz) of a strictly increasing, uniform nu grid."""
    nu = np.asarray(nu_mhz, dtype=float)
    if nu.ndim != 1 or nu.size < 2:
        raise ValueError("nu_mhz must be a 1-D grid of >= 2 frequencies")
    d = np.diff(nu)
    dnu = d.mean()
    if dnu <= 0:
        raise ValueError("nu_mhz must be strictly increasing")
    if not np.allclose(d, dnu, rtol=1e-6, atol=0.0):
        raise ValueError("nu_mhz must be uniformly spaced for an FFT")
    return dnu * 1e6


def _normalized_weights(nchan, weights):
    if weights is None:
        weights = np.ones(nchan)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (nchan,):
        raise ValueError("weights must have shape (nchan,)")
    total = weights.sum()
    if total == 0:
        raise ValueError("weights must have a non-zero sum")
    return weights / total


def delay_grid(nu_mhz, pad=1):
    """Delay axis tau (s), fftshifted so tau = 0 is centered.

    Conjugate variable to frequency for the FFT in :func:`faraday_fft`.
    `pad` zero-pads to pad*nchan output bins (interpolates the spectrum).
    Requires `nu_mhz` to be uniformly spaced.
    """
    dnu_hz = _uniform_dnu_hz(nu_mhz)
    n = int(pad) * np.asarray(nu_mhz).size
    return np.fft.fftshift(np.fft.fftfreq(n, d=dnu_hz))


def faraday_fft(Q, U, nu_mhz, weights=None, pad=1):
    """Complex delay spectrum F(t, tau) from a nu-FFT of P = Q + iU.

    Q, U have shape (nchan,) or (ntimes, nchan), sampled on the uniform
    grid `nu_mhz` (MHz). Returns F of shape (ntimes, pad*nchan), fftshifted
    to align with ``delay_grid(nu_mhz, pad)``. `weights` (per-channel, e.g.
    an apodization window) are normalized to sum 1 and applied to P before
    the FFT, matching :func:`rmsynth.faraday_spectrum`'s convention.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    nchan = np.asarray(nu_mhz).size
    _uniform_dnu_hz(nu_mhz)  # validate spacing
    if Q.shape[-1] != nchan or U.shape[-1] != nchan:
        raise ValueError("Q, U last axis must match nu_mhz length")
    w = _normalized_weights(nchan, weights)
    P = (Q + 1j * U) * w  # (ntimes, nchan)
    F = np.fft.fft(P, n=int(pad) * nchan, axis=-1)  # (ntimes, n)
    return np.fft.fftshift(F, axes=-1)
