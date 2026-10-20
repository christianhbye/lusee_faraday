"""Statistics of correlated measured spectra, with independent repeats."""

import numpy as np


def gaussian_mean_square(covariance, mean=None):
    """Mean and sampling RMS of x^T x/n for one correlated Gaussian draw."""
    if not np.isrealobj(covariance) or (
        mean is not None and not np.isrealobj(mean)
    ):
        raise ValueError("real observables are required; retain complex C/J")
    c = np.asarray(covariance, float)
    if (
        c.ndim != 2
        or c.shape[0] != c.shape[1]
        or not c.size
        or not np.all(np.isfinite(c))
        or not np.allclose(c, c.T, rtol=1e-10, atol=1e-14 * np.max(abs(c)))
    ):
        raise ValueError("finite symmetric covariance required")
    mu = np.zeros(len(c)) if mean is None else np.asarray(mean, float)
    if mu.shape != (len(c),) or not np.all(np.isfinite(mu)):
        raise ValueError("mean and covariance disagree")
    if np.linalg.eigvalsh(c)[0] < -1e-10 * max(abs(c).max(), 1e-300):
        raise ValueError("covariance is not positive semidefinite")
    n = len(c)
    expectation = (np.trace(c) + mu @ mu) / n
    variance = (2 * np.sum(c * c) + 4 * mu @ c @ mu) / n**2
    return dict(
        mean_k2=float(expectation),
        sampling_rms_k2=float(np.sqrt(max(variance, 0))),
    )


def repeat_difference(first, second, noise_first, noise_second):
    """One fixed sky cancels; independent estimator noise covariances add."""
    a, b, ca, cb = map(np.asarray, (first, second, noise_first, noise_second))
    if (
        a.shape != b.shape
        or ca.shape != (a.size, a.size)
        or cb.shape != ca.shape
    ):
        raise ValueError("repeat coordinates/covariances disagree")
    difference = a - b
    predicted = gaussian_mean_square(ca + cb)
    return dict(
        observed_mean_square_k2=float(np.mean(difference**2)),
        prediction=predicted,
        covariance=ca + cb,
        difference=difference,
    )
