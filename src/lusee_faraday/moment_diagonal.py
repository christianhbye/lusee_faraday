"""Exact marginal variances from the complete Hermite covariance operator."""

import numpy as np


def diagonal_moments(projector, coefficients):
    """Contract the same four complex branches without allocating bin pairs."""
    a = np.asarray(coefficients)
    if a.ndim != 2 or a.shape[1] != projector.degree + 1:
        raise ValueError("coefficient and mode dimensions disagree")
    if not np.all(np.isfinite(a)):
        raise ValueError("nonfinite phase coefficients")
    weights = np.diagonal(projector.weights, axis1=-2, axis2=-1)

    def contract(selection):
        b, w = a[:, selection], weights[:, selection]
        conjugate = abs(b) ** 2
        relation = b * b
        return (
            (conjugate @ w[0]).real
            + (conjugate @ w[1]).imag
            + (relation @ w[2]).real
            + (relation @ w[3]).imag
        )

    smooth, speckle = contract(slice(0, 1)), contract(slice(1, None))
    return dict(smooth=smooth, speckle=speckle, total=smooth + speckle)


def mixture_diagonals(projector, values, probabilities):
    """Global scale mixture, retaining the variance of its phase mean."""
    mean = sum(p * a[:, 0] for p, a in zip(probabilities, values))
    mean_coefficients = np.zeros_like(values[0])
    mean_coefficients[:, 0] = mean
    smooth = diagonal_moments(projector, mean_coefficients)["smooth"]
    speckle = np.zeros_like(smooth)
    for p, a in zip(probabilities, values):
        delta = np.zeros_like(a)
        delta[:, 0] = a[:, 0] - mean
        speckle += p * (
            diagonal_moments(projector, a)["speckle"]
            + diagonal_moments(projector, delta)["smooth"]
        )
    return dict(smooth=smooth, speckle=speckle, total=smooth + speckle)
