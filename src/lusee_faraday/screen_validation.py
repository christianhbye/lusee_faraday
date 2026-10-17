"""Screen-ensemble estimates and sampling-error diagnostics."""

import numpy as np
from scipy.stats import norm


def moment_comparison(
    samples,
    predicted,
    bootstrap_seed=700001,
    bootstrap_count=512,
    family_size=1,
):
    """All cells, screen bootstrap SE, and a simultaneous normal diagnostic.

    The Bonferroni threshold is an approximation, not an exact distributional
    test. Compare bootstrap SE with influence-function SE to flag poor error
    estimation before interpreting this moment check.
    """
    y = np.asarray(samples)
    n, dim = y.shape
    if n < 8:
        raise ValueError("at least eight independent screens required")
    mean = y.mean(axis=0)
    dev = y - mean
    cov = dev.T @ dev / (n - 1)
    raw = y[:, :, None] * y[:, None, :]
    delta = y - predicted["off"]
    delta_raw = delta[:, :, None] * delta[:, None, :]
    counts = np.random.default_rng(bootstrap_seed).multinomial(
        n, np.full(n, 1 / n), size=bootstrap_count
    )
    bm = counts @ y / n
    br = (counts @ raw.reshape(n, -1) / n).reshape(-1, dim, dim)
    bc = (br - bm[:, :, None] * bm[:, None, :]) * n / (n - 1)
    bd = (counts @ delta_raw.reshape(n, -1) / n).reshape(-1, dim, dim)
    empirical = dict(
        mean=mean,
        covariance=cov,
        second=raw.mean(0),
        delta_second=delta_raw.mean(0),
    )
    pseudo_cov = n / (n - 1) * dev[:, :, None] * dev[:, None, :] - cov[
        None, :, :
    ] / (n - 1)
    influence = dict(
        mean=y, covariance=pseudo_cov, second=raw, delta_second=delta_raw
    )
    boot = dict(mean=bm, covariance=bc, second=br, delta_second=bd)
    cells = family_size * (dim + 3 * dim * dim)
    threshold = norm.ppf(1 - 0.01 / (2 * cells))
    output = dict(threshold=float(threshold), n=n, cells=int(cells))
    worst, min_ratio, max_ratio = 0.0, np.inf, 0.0
    for key, value in empirical.items():
        se = boot[key].std(0, ddof=1)
        ise = influence[key].std(0, ddof=1) / np.sqrt(n)
        error = value - predicted[key]
        z = np.divide(error, se, out=np.zeros_like(error), where=se > 0)
        if np.any((se == 0) & (abs(error) > 1e-13)):
            z[(se == 0) & (abs(error) > 1e-13)] = np.inf
        ratio = np.divide(se, ise, out=np.ones_like(se), where=ise > 0)
        worst = max(worst, float(np.max(abs(z))))
        min_ratio = min(min_ratio, float(ratio.min()))
        max_ratio = max(max_ratio, float(ratio.max()))
        output[key] = dict(
            empirical=value,
            expected=predicted[key],
            se=se,
            error=error,
            standardized=z,
        )
    output.update(
        worst_z=worst,
        bootstrap_se_ratio=[min_ratio, max_ratio],
        passed=bool(worst < threshold and min_ratio > 0.7 and max_ratio < 1.3),
    )
    return output


def noise_weighted_error(mean_error, covariance_error, total):
    """Euclidean mean error and spectral covariance error after whitening."""
    factor = np.linalg.cholesky(total)
    dm = np.linalg.solve(factor, mean_error)
    ds = np.linalg.solve(factor, covariance_error)
    ds = np.linalg.solve(factor, ds.T).T
    return float(np.linalg.norm(dm)), float(np.linalg.norm(ds, ord=2))
