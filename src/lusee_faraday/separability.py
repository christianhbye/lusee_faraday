"""Common smooth subtraction, residual covariance and intensity ratios."""

import numpy as np


def channel_smooth_basis(fine_mhz, operator, degree, center_mhz=None):
    """Smooth physical-frequency templates through the actual channel map.

    Folded zoom bins must use W times the fine-frequency template; a
    polynomial evaluated at nominal bin centers is a different estimator.
    """
    fine, w = np.asarray(fine_mhz, float), np.asarray(operator)
    if (
        fine.ndim != 1
        or fine.size < 2
        or not np.all(np.isfinite(fine))
        or not np.all(np.diff(fine) > 0)
        or w.ndim != 2
        or w.shape[1] != len(fine)
        or not np.isrealobj(w)
        or not np.all(np.isfinite(w))
        or not isinstance(degree, int)
        or degree < 0
    ):
        raise ValueError("invalid smooth-template coordinates or degree")
    center = (fine[0] + fine[-1]) / 2 if center_mhz is None else center_mhz
    if not np.isfinite(center):
        raise ValueError("smooth-template center must be finite")
    coordinate = (fine - center) / ((fine[-1] - fine[0]) / 2)
    return w @ np.polynomial.legendre.legvander(coordinate, degree)


def residual_operator(basis, weights=None):
    """Weighted linear least-squares residual map shared by every component.

    Zero-weight samples do not enter the fit and their output rows vanish.
    Their values must be masked in plotted/exported observations. Reject
    rank-deficient or saturated fits; neither yields an informative residual.
    """
    b = np.asarray(basis, float)
    if b.ndim != 2 or not b.shape[1]:
        raise ValueError("smooth basis must have at least one column")
    w = np.ones(b.shape[0]) if weights is None else np.asarray(weights, float)
    if (
        b.ndim != 2
        or not np.all(np.isfinite(b))
        or w.shape != (b.shape[0],)
        or not np.all(np.isfinite(w) & (w >= 0))
        or not np.any(w > 0)
    ):
        raise ValueError("invalid smooth basis or fit weights")
    selected = w > 0
    if selected.sum() <= b.shape[1]:
        raise ValueError(
            "smooth fit needs positive residual degrees of freedom"
        )
    root = np.sqrt(w / w.max())
    u, s, vt = np.linalg.svd(root[:, None] * b, full_matrices=False)
    rank = int(np.count_nonzero(s > 1e-12 * s[0]))
    if rank != b.shape[1]:
        raise ValueError("smooth fit is rank deficient")
    fit = (b @ (vt.T / s) @ u.T) * root[None, :]
    r = np.diag(selected.astype(float)) - selected[:, None] * fit
    return dict(
        operator=r,
        fit_operator=selected[:, None] * fit,
        rank=rank,
        condition_number=float(s[0] / s[-1]),
        residual_dof=int(selected.sum() - rank),
        selected=selected,
    )


def residual_covariance(covariance, operator):
    """Real bin/readout covariance -> R C R^T on both frequency legs."""
    c, r = np.asarray(covariance), np.asarray(operator)
    if (
        c.ndim != 4
        or c.shape[0] != c.shape[2]
        or r.ndim != 2
        or r.shape[1] != c.shape[0]
        or not np.isrealobj(c)
        or not np.isrealobj(r)
    ):
        raise ValueError(
            "residual operator and covariance dimensions disagree"
        )
    return np.einsum("bi,ipjq,cj->bpcq", r, c, r, optimize=True)


def fractional_residual(
    stokes,
    covariance,
    operator,
    intensity_reference=None,
    max_intensity_fractional_rms=0.05,
):
    """Subtract Q/U in K, then divide by a common intensity reference.

    With estimated I=stokes[:,0], the delta-method Jacobian retains I-Q/U
    correlations and intensity variance. A supplied fixed reference has no
    denominator noise. The linearization is rejected at low intensity SNR.
    Covariance has shape (bin,4,bin,4); output is (bin,2,bin,2).
    """
    s, c, r = np.asarray(stokes), np.asarray(covariance), np.asarray(operator)
    if (
        s.ndim != 2
        or s.shape[1] != 4
        or c.shape != (len(s), 4, len(s), 4)
        or r.shape != (len(s), len(s))
        or not np.isrealobj(s)
        or not np.isrealobj(c)
        or not np.isrealobj(r)
        or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(c))
        or not np.all(np.isfinite(r))
    ):
        raise ValueError("invalid Stokes/covariance ratio inputs")
    estimated = intensity_reference is None
    intensity = (
        s[:, 0] if estimated else np.asarray(intensity_reference, float)
    )
    if intensity.shape != (len(s),) or not np.all(
        np.isfinite(intensity) & (intensity > 0)
    ):
        raise ValueError("intensity reference must be positive and finite")
    n = len(s)
    variance = np.diag(c.reshape(4 * n, 4 * n)).reshape(n, 4)[:, 0]
    fraction = np.sqrt(np.maximum(variance, 0)) / intensity
    if (
        not np.isfinite(max_intensity_fractional_rms)
        or max_intensity_fractional_rms <= 0
        or (estimated and np.max(fraction) > max_intensity_fractional_rms)
    ):
        raise ValueError(
            "intensity uncertainty is too large for linearization"
        )
    numerator = r @ s[:, 1:3]
    jacobian = np.zeros((n, 2, n, 4))
    for component in range(2):
        jacobian[:, component, :, component + 1] = r / intensity[:, None]
        if estimated:
            jacobian[np.arange(n), component, np.arange(n), 0] = (
                -numerator[:, component] / intensity**2
            )
    result = (
        jacobian.reshape(2 * n, 4 * n)
        @ c.reshape(4 * n, 4 * n)
        @ jacobian.reshape(2 * n, 4 * n).T
    )
    return dict(
        value=numerator / intensity[:, None],
        covariance=result.reshape(n, 2, n, 2),
        jacobian=jacobian,
        intensity_reference=intensity,
        intensity_is_estimated=estimated,
        maximum_intensity_fractional_rms=float(fraction.max()),
    )


def amplitude_ratio(numerator, denominator):
    """RMS ratio, with NaN for an exactly zero denominator, never infinity."""
    a, b = np.broadcast_arrays(
        np.asarray(numerator, float), np.asarray(denominator, float)
    )
    if not np.all(np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)):
        raise ValueError("RMS values must be finite and nonnegative")
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
