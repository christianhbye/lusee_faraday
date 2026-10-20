"""Stable Gaussian visibility illustrations from complex phase coefficients."""

import numpy as np


def draw_depth_coefficients(projector, coefficients, seed, draws=1):
    """Draw independent Hermite visibility modes before channel/fit mixing.

    Each mode has a joint real covariance for the unit complex coefficients
    1 and i, retaining both complex moments and readout/time cross terms.
    Tiny channel residuals therefore do not depend on resolving tiny
    eigenvalues of a nearly coherent full frequency covariance. The modes
    are uncorrelated by construction and are assumed jointly Gaussian for
    this illustration; this does not establish full-sky Gaussianity.
    """
    a = np.asarray(coefficients)
    if (
        a.ndim != 2
        or a.shape[1] != projector.degree + 1
        or not np.all(np.isfinite(a))
        or not isinstance(draws, int)
        or draws < 1
    ):
        raise ValueError("invalid coefficient-draw dimensions")
    rng = np.random.default_rng(seed)
    sample = np.zeros((draws, len(a), projector.size))
    smooth = np.zeros_like(sample)
    speckle = np.zeros_like(sample)
    diagnostics = []
    for mode in range(a.shape[1]):
        basis = np.zeros((2, a.shape[1]), complex)
        basis[:, mode] = [1, 1j]
        covariance = projector.moments(basis)["total"].reshape(
            2 * projector.size, 2 * projector.size
        )
        scale = max(float(abs(covariance).max()), np.finfo(float).tiny)
        if abs(covariance - covariance.T).max() > 1e-8 * scale:
            raise ValueError("mode covariance is not symmetric")
        eigen, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        if eigen[0] < -1e-8 * scale:
            raise ValueError("mode covariance is not positive semidefinite")
        latent = (
            rng.normal(size=(draws, len(eigen)))
            * np.sqrt(np.maximum(eigen, 0))
        ) @ vectors.T
        size = projector.size
        real = latent[:, :size]
        imag = latent[:, size:]
        value = (
            a[None, :, mode, None].real * real[:, None, :]
            + a[None, :, mode, None].imag * imag[:, None, :]
        )
        sample += value
        if mode == 0:
            smooth += value
        else:
            speckle += value
        diagnostics.append(
            dict(
                mode=mode,
                minimum_eigenvalue=float(eigen[0]),
                tolerance=1e-8 * scale,
            )
        )
    return dict(
        sample=sample,
        smooth=smooth,
        speckle=speckle,
        mode_diagnostics=diagnostics,
    )
