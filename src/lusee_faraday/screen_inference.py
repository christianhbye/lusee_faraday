"""Small conditional Gaussian-screen inference tools.

Noise here is synthetic Gaussian readout noise in beam-averaged K, not a
radiometer forecast. Screen realization scatter is retained in every model.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import beta as beta_distribution

from .conventions import lambda_squared
from .gaussian_screen import qu_moments, real_covariance


def gaussian_nll(data, mean, covariance):
    """Negative log density with normalization, using Cholesky solves."""
    factor = cho_factor(covariance, lower=True)
    residual = np.asarray(data) - mean
    return 0.5 * (
        residual @ cho_solve(factor, residual)
        + 2 * np.log(np.diag(factor[0])).sum()
        + len(mean) * np.log(2 * np.pi)
    )


def gaussian_kl(mean, covariance, other_mean, other_covariance):
    """KL(N(mean,covariance) || N(other_mean,other_covariance))."""
    factor = cho_factor(other_covariance, lower=True)
    own = cho_factor(covariance, lower=True)
    delta = mean - other_mean
    return 0.5 * (
        np.trace(cho_solve(factor, covariance))
        + delta @ cho_solve(factor, delta)
        - len(mean)
        + 2 * np.log(np.diag(factor[0])).sum()
        - 2 * np.log(np.diag(own[0])).sum()
    )


def transform_source(
    mean,
    covariance,
    freq,
    model,
    amplitude=1.0,
    angle=0.0,
    beta_shift=0.0,
    curvature=0.0,
):
    """Change source amplitude, angle (radians) and spectral law jointly."""
    logf = np.log(np.asarray(freq) / model.ref_mhz)
    scale = amplitude * np.exp(beta_shift * logf + curvature * logf**2)
    co, si = np.cos(2 * angle), np.sin(2 * angle)
    diagonal = np.diag(scale)
    matrix = np.block(
        [[co * diagonal, -si * diagonal], [si * diagonal, co * diagonal]]
    )
    return matrix @ mean, matrix @ covariance @ matrix.T


def profile_expected_kl(truth, candidate, noise, freq, model, stage=1):
    """Finite expected distinguishability after profiling source nuisances.

    Stages 1/2/3 free amplitude+angle, then beta, then curvature. Return all
    optimizer statuses; unsuccessful fits cannot establish identifiability.
    """
    bounds = [(0.0001, model.i_ref_k / model.p_ref_k), (-np.pi / 2, np.pi / 2)]
    if stage >= 2:
        bounds.append((-4 - model.beta, -1 - model.beta))
    if stage >= 3:
        bounds.append((-2.0, 2.0))

    def objective(parameters):
        values = np.zeros(4)
        values[: len(parameters)] = parameters
        logf = np.log(freq / model.ref_mhz)
        ratio = values[0] * np.exp(values[2] * logf + values[3] * logf**2)
        if np.max(ratio) * model.p_ref_k > model.i_ref_k:
            return (
                1e6
                + 1e6 * (np.max(ratio) * model.p_ref_k - model.i_ref_k) ** 2
            )
        mean, cov = transform_source(
            candidate["mean"], candidate["covariance"], freq, model, *values
        )
        return gaussian_kl(
            truth["mean"], truth["covariance"] + noise, mean, cov + noise
        )

    results = []
    for amplitude in (0.5, 1.0, 1.5):
        start = np.zeros(len(bounds))
        start[0] = amplitude
        results.append(
            minimize(
                objective,
                start,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(ftol=1e-11, gtol=1e-7, maxiter=400),
            )
        )
    best = min(results, key=lambda r: r.fun)
    return dict(
        kl=float(best.fun),
        parameters=best.x.tolist(),
        converged=bool(best.success),
        statuses=[
            dict(
                success=bool(r.success),
                message=str(r.message),
                value=float(r.fun),
            )
            for r in results
        ],
    )


def ablate_frequency_covariance(covariance):
    """Retain same-frequency Q/U blocks; remove cross-frequency blocks."""
    n = len(covariance) // 2
    mask = np.equal.outer(np.arange(2 * n) % n, np.arange(2 * n) % n)
    return np.asarray(covariance) * mask


def binomial_interval(successes, trials, alpha=0.05):
    """Two-sided exact Clopper-Pearson interval, including boundary cases."""
    low = (
        0.0
        if successes == 0
        else beta_distribution.ppf(
            alpha / 2, successes, trials - successes + 1
        )
    )
    high = (
        1.0
        if successes == trials
        else beta_distribution.ppf(
            1 - alpha / 2, successes + 1, trials - successes
        )
    )
    return [float(low), float(high)]


def fixed_source_scores(data, moments, noise):
    """Log likelihood for rows of data across a fixed-source scale grid."""
    rows = np.atleast_2d(data)
    output = []
    for item in moments:
        total = item["covariance"] + noise
        factor = cho_factor(total, lower=True)
        residual = rows - item["mean"]
        output.append(
            -0.5
            * (
                np.sum(residual * cho_solve(factor, residual.T).T, axis=1)
                + 2 * np.log(np.diag(factor[0])).sum()
                + len(total) * np.log(2 * np.pi)
            )
        )
    return np.stack(output, axis=1)


def mixture_moments(freq, model, amplitudes=(np.sqrt(0.5), np.sqrt(1.5))):
    """Exact equal-weight global scale mixture via conditional moments."""
    predictions = [
        qu_moments(
            freq,
            replace(model, sigma_depth=model.sigma_depth * a),
            method="special",
        )
        for a in amplitudes
    ]
    mean = np.mean([p["mean"] for p in predictions], axis=0)
    second = np.mean([p["second"] for p in predictions], axis=0)
    return dict(
        mean=mean,
        covariance=second - np.outer(mean, mean),
        second=second,
        off=predictions[0]["off"],
        delta_second=np.mean([p["delta_second"] for p in predictions], axis=0),
    )


def local_stationary_template(freq, model, center_mhz):
    """Strong-rotation, small-separation stationary approximation to P.

    For x=lambda^2 and w the physical Gaussian weight width, the raw
    same-branch coherence integral is approximately
    exp(2im(x-y)-2sigma^2(x-y)^2)/(1+8sigma^2*w^2*x*y/ell^2).
    Freeze x*y=x0^2 and neglect exponentially small opposite-branch/mean
    terms only in the stated strong-rotation limit. A circular P then has
    J=0 and physical variance eta^2*P_ref^2/(1+8sigma^2*w^2*x0^2/ell^2).
    """
    x = lambda_squared(freq)
    x0 = float(lambda_squared([center_mhz])[0])
    variance = 1 / (
        1
        + 8
        * model.sigma_depth**2
        * model.weight_sigma_pc**2
        * x0**2
        / model.correlation_pc**2
    )
    intrinsic = (
        model.eta
        * model.p_ref_k
        * (np.asarray(freq) / model.ref_mhz) ** model.beta
        * np.exp(2j * np.deg2rad(model.angle_deg))
    )
    dx = x[:, None] - x[None, :]
    c = (
        variance
        * np.outer(intrinsic, intrinsic.conj())
        * np.exp(2j * model.mean_depth * dx - 2 * model.sigma_depth**2 * dx**2)
    )
    return dict(
        mean=np.zeros(2 * len(x)),
        covariance=real_covariance(c, np.zeros_like(c)),
        normalized_variance=variance,
    )


def profile_null_batch(
    data,
    freq,
    model,
    noise,
    constant_depth=False,
    free_beta=False,
    beta_bounds=(-4.0, -1.0),
    depth_bounds=(-0.1, 0.1),
):
    """Fit deterministic Q+iU source nulls; profile complex source amplitude.

    Diagonal equal-Q/U noise is required. A bounded grid initializes depth
    and beta, followed by continuous optimization. Returned residuals remain
    in the original Q/U coordinates; no hidden on/off realization is used.
    """
    freq = np.asarray(freq)
    data = np.atleast_2d(data)
    n = len(freq)
    variance = np.diag(noise)[:n]
    if not np.allclose(
        noise, np.diag(np.diag(noise)), atol=0, rtol=0
    ) or not np.array_equal(np.diag(noise)[n:], variance):
        raise ValueError("requires diagonal noise equal for Q and U")
    pol = data[:, :n] + 1j * data[:, n:]
    x = lambda_squared(freq)
    depths = (
        np.linspace(*depth_bounds, 65) if constant_depth else np.array([0.0])
    )
    betas = (
        np.linspace(*beta_bounds, 31) if free_beta else np.array([model.beta])
    )
    grid = np.array([(c, b) for c in depths for b in betas])

    def template(c, b):
        return model.eta * (freq / model.ref_mhz) ** b * np.exp(2j * c * x)

    templates = np.array([template(c, b) for c, b in grid])
    denom = np.sum(abs(templates) ** 2 / variance, axis=1)
    # Bounds enforce peak polarization and |P|<=I throughout the window.
    limits = np.minimum(
        model.i_ref_k,
        np.min(
            model.i_ref_k
            * (freq[None, :] / model.ref_mhz)
            ** (model.beta - grid[:, 1, None]),
            axis=1,
        ),
    )
    best_indices = []
    for start in range(0, len(pol), 256):
        stop = start + 256
        rows = pol[start:stop]
        cross = (rows / variance) @ templates.conj().T
        coeff = cross / denom
        coeff *= np.minimum(1, limits / np.maximum(abs(coeff), 1e-300))
        cost = (
            np.sum(abs(rows) ** 2 / variance, axis=1)[:, None]
            - 2 * np.real(cross * coeff.conj())
            + denom * abs(coeff) ** 2
        )
        best_indices.extend(np.argmin(cost, axis=1))
    residuals = []
    parameters = []
    statuses = []
    for row, index in zip(pol, best_indices):
        start = grid[index]
        bounds = []
        initial = []
        if constant_depth:
            bounds.append(depth_bounds)
            initial.append(start[0])
        if free_beta:
            bounds.append(beta_bounds)
            initial.append(start[1])

        def evaluate(values, details=False):
            c = values[0] if constant_depth else 0.0
            b = values[-1] if free_beta else model.beta
            t = template(c, b)
            coefficient = np.sum(row * t.conj() / variance) / np.sum(
                abs(t) ** 2 / variance
            )
            limit = min(
                model.i_ref_k,
                np.min(
                    model.i_ref_k * (freq / model.ref_mhz) ** (model.beta - b)
                ),
            )
            coefficient *= min(1, limit / max(abs(coefficient), 1e-300))
            residual = row - coefficient * t
            value = float(np.sum(abs(residual) ** 2 / variance))
            if details:
                return residual, [
                    float(abs(coefficient)),
                    float(0.5 * np.angle(coefficient)),
                    float(c),
                    float(b),
                ]
            return value

        if bounds:
            result = minimize(
                evaluate,
                initial,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(ftol=1e-11, gtol=1e-7, maxiter=200),
            )
            if not result.success:
                result = minimize(
                    evaluate,
                    result.x,
                    bounds=bounds,
                    method="Powell",
                    options=dict(xtol=1e-9, ftol=1e-10, maxiter=200),
                )
            values = result.x
            statuses.append(bool(result.success))
        else:
            values = []
            statuses.append(True)
        residual, parameter = evaluate(values, True)
        residuals.append(np.r_[residual.real, residual.imag])
        parameters.append(parameter)
    return dict(
        residual=np.array(residuals),
        parameters=np.array(parameters),
        converged=np.array(statuses),
    )


def covariance_discrepancy(residual, signal, noise):
    """Quadratic residual score with Wiener eigenvalue weights S/(S+N).

    Null fits and thresholds are calibrated with the identical statistic.
    This is a discrepancy, not an asserted chi-square likelihood ratio.
    """
    factor = np.linalg.cholesky(noise)
    white = np.linalg.solve(factor, signal)
    white = np.linalg.solve(factor, white.T).T
    # Algebraically W=I-(I+S_white)^-1; no eigenvalue clipping or jitter.
    weight = np.eye(len(signal)) - np.linalg.solve(
        np.eye(len(signal)) + white, np.eye(len(signal))
    )
    r = np.linalg.solve(factor, np.atleast_2d(residual).T).T
    return np.einsum("ni,ij,nj->n", r, weight, r)
