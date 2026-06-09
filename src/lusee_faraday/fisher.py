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
    """Marginalized 1-sigma error on parameter idx (others free).

    Uses a pseudo-inverse so a rank-deficient (degenerate) Fisher matrix
    does not raise: singular values below rcond * max(sv) are dropped,
    yielding a finite pseudo-inverse value rather than signalling
    non-identifiability. rcond=1e-12 is deliberately looser than numpy's
    default so near-degenerate parameter pairs read as unconstrained.
    """
    Cinv = np.linalg.pinv(F, rcond=rcond)
    return float(np.sqrt(Cinv[idx, idx]))


def detection_snr(F, idx, alpha_fid=1.0, rcond=1e-12):
    """Detection SNR = alpha_fid / marginalized sigma(alpha)."""
    return alpha_fid / marginal_error(F, idx, rcond=rcond)


def faraday_column(
    I_topo,
    Q_topo,
    U_topo,
    rm_topo,
    beam,
    mask,
    freqs,
    alpha_fid=1.0,
    dalpha=1e-3,
    **kwargs,
):
    """dP/dalpha via central finite difference at alpha_fid."""
    Pp = pol_response(
        I_topo,
        Q_topo,
        U_topo,
        rm_topo,
        beam,
        mask,
        freqs,
        alpha=alpha_fid + dalpha,
        **kwargs,
    )
    Pm = pol_response(
        I_topo,
        Q_topo,
        U_topo,
        rm_topo,
        beam,
        mask,
        freqs,
        alpha=alpha_fid - dalpha,
        **kwargs,
    )
    return (Pp - Pm) / (2 * dalpha)


def dispersion_column(P_pol_fid, lam2):
    """dP/dtau at tau=0 for depolarization exp(-2 tau lam2^2):
    -2 (lam2)^2 * P_pol_fid (tau is the Faraday variance)."""
    lam2 = np.asarray(lam2, dtype=float)
    return -2.0 * lam2[None, :] ** 2 * P_pol_fid


def run_forecast(
    I_topo,
    Q_topo,
    U_topo,
    rm_topo,
    basis_topo,
    beam,
    mask,
    freqs,
    lam2,
    sigma,
    alpha_fid=1.0,
    dalpha=1e-3,
    **kwargs,
):
    """Sky-marginalized Faraday detection forecast.

    basis_topo: list of (Q_basis_topo, U_basis_topo) rotated nuisance
    maps. Returns dict with sigma_alpha / snr (sky+tau marginalized) and
    sigma_alpha_opt / snr_opt (sky+tau fixed -> optimistic bound).
    """
    zeroQ = np.zeros_like(Q_topo)
    P0 = pol_response(
        I_topo,
        zeroQ,
        zeroQ,
        rm_topo,
        beam,
        mask,
        freqs,
        alpha=alpha_fid,
        **kwargs,
    )
    P_fid = pol_response(
        I_topo,
        Q_topo,
        U_topo,
        rm_topo,
        beam,
        mask,
        freqs,
        alpha=alpha_fid,
        **kwargs,
    )
    P_pol_fid = P_fid - P0  # polarized sky part only (I-leakage removed)

    a_col = faraday_column(
        I_topo,
        Q_topo,
        U_topo,
        rm_topo,
        beam,
        mask,
        freqs,
        alpha_fid=alpha_fid,
        dalpha=dalpha,
        **kwargs,
    )
    t_col = dispersion_column(P_pol_fid, lam2)
    mode_cols = [
        pol_response(
            I_topo,
            Qb,
            Ub,
            rm_topo,
            beam,
            mask,
            freqs,
            alpha=alpha_fid,
            **kwargs,
        )
        - P0
        for Qb, Ub in basis_topo
    ]

    cols = [a_col, t_col] + mode_cols  # alpha is index 0
    F = fisher_matrix(cols, sigma)
    F_opt = fisher_matrix([a_col], sigma)
    sig_a = marginal_error(F, 0)
    sig_a_opt = marginal_error(F_opt, 0)
    return {
        "sigma_alpha": sig_a,
        "snr": alpha_fid / sig_a,
        "sigma_alpha_opt": sig_a_opt,
        "snr_opt": alpha_fid / sig_a_opt,
        "n_modes": len(basis_topo),
    }
