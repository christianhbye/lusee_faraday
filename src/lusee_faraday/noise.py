"""Radiometer noise for LuSEE polarized spectra.

sigma = T_sys / sqrt(dnu * dt). T_sys is sky-dominated (~ Stokes I).

radiometer_sigma / add_noise are ported verbatim from the
faraday-fisher-forecast branch (spec S4.10).  The rest is the S4.10
detectability machinery: the zoom-bin noise covariance (the 1.44x ENBW
overlap makes adjacent bins correlated), the Faraday signal covariance,
and the whitened matched-filter threshold whose diagonal limit is the
closed form.
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


def zoom_noise_covariance(W, sigma_bin):
    """Noise covariance of the zoom bins: sigma_bin^2 * overlap.

    ``W`` is the (nfine, nbin) column-normalized weight matrix
    (dispersion.zoom_bin_matrix): white fine-channel noise gives bin
    covariance ~ W.T W, normalized so the diagonal is sigma_bin^2.
    """
    W = np.asarray(W, dtype=float)
    G = W.T @ W
    d = np.sqrt(np.diag(G))
    return float(sigma_bin) ** 2 * (G / np.outer(d, d))


def faraday_signal_covariance(phi, H, lam2_bins, allow_one_sided=False):
    """Frequency covariance of a Gaussian Faraday signal of shape H.

    S_ij = sum_b Hhat_b exp(2i phi_b (lam2_i - lam2_j)); Hhat sums to
    one so diag(S) = 1 and an amplitude A means per-bin signal power
    A^2.

    ``phi`` MUST BE THE SIGNED DEPTH GRID.  The observable is the
    complex P = Q + iU, so S is the complex transform of the SIGNED
    depth distribution.  Passing a FOLDED (all-positive) template
    models a sky whose every column has one sign of RM: it understates
    A_5sigma by ~18% on the real map, and the visible symptom is an S
    that is Hermitian but grossly non-symmetric (max|S - S^T| = 1.08
    folded against 0.19 signed, with |Im S| running 54% of |S| rather
    than 9%).

    That mix-up shipped once and was reproduced in fresh prototype code
    within an hour of being found, so it is refused here rather than
    documented -- the same way ``sky.binned_screen`` refuses an
    unresolved screen.  ``allow_one_sided=True`` is the deliberate
    opt-out, for the coherence tilt (a folded shape statement with no
    signed twin, entering no verdict) and for tests that compare the
    two conventions on purpose.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    if not allow_one_sided and not np.any(phi < 0.0):
        raise ValueError(
            "phi has no negative bins, so this looks like a FOLDED "
            "template. S must be built from the SIGNED depth "
            "distribution (the observable is the complex P = Q + iU); "
            "a folded one understates A_5sigma by ~18%. Pass the "
            "signed grid, or allow_one_sided=True if the one-sided "
            "shape is deliberate."
        )
    Hhat = np.asarray(H, dtype=float).ravel()
    Hhat = Hhat / Hhat.sum()
    lam2 = np.asarray(lam2_bins, dtype=float).ravel()
    E = np.exp(2j * np.outer(lam2, phi))  # (nbin, nphi)
    return (E * Hhat[None, :]) @ E.conj().T


def matched_filter_threshold(S, N, n_nights, n_lst, snr=5.0):
    """5-sigma amplitude threshold of the whitened matched filter.

    Gaussian-signal likelihood ratio with M = n_lst independent LST
    samples and n_nights coherent nights (noise power / n):
    SNR = n * A^2 * sqrt(n_lst * tr[(N^-1 S)^2]).  With S = I and
    N = sigma^2 I this is exactly the closed form with
    N_modes = n_lst * nbin.
    """
    F = np.linalg.solve(np.asarray(N), np.asarray(S))
    fisher = np.sqrt(float(n_lst) * max(np.einsum("ij,ji->", F, F).real, 0.0))
    return float(np.sqrt(snr / (float(n_nights) * fisher)))


def closed_form_threshold(dnu_coh_hz, tau_s, n_modes, n_nights, snr=5.0):
    """A = sigma_mode * sqrt(snr / (n * sqrt(N_modes))), the corrected
    S4.10 closed form: noise per coherence cell, not per zoom bin.
    """
    sigma_mode = 1.0 / np.sqrt(float(dnu_coh_hz) * float(tau_s))
    return float(
        sigma_mode * np.sqrt(snr / (float(n_nights) * np.sqrt(n_modes)))
    )
