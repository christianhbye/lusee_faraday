"""Broad frequency covariance with PSD-preserving channel compression."""

import numpy as np
from scipy.sparse import csr_matrix


def interpolated_channel_operator(fine_mhz, nodes_mhz, operator):
    """Return W T, where T linearly interpolates nodes onto fine frequencies.

    This approximates the random process, so covariance projection is a
    congruence and preserves positive semidefiniteness. Every frequency
    pair remains present. Refine nodes against the required observable;
    a broad-band smooth component must not be cut off by a coherence mask.
    """
    fine, nodes, w = map(np.asarray, (fine_mhz, nodes_mhz, operator))
    if (
        fine.ndim != 1
        or not fine.size
        or nodes.ndim != 1
        or nodes.size < 2
        or not np.all(np.isfinite(fine))
        or not np.all(np.isfinite(nodes))
        or not np.all(np.diff(nodes) > 0)
        or fine.min() < nodes[0]
        or fine.max() > nodes[-1]
        or w.ndim != 2
        or w.shape[1] != fine.size
        or not np.all(np.isfinite(w))
    ):
        raise ValueError("invalid interpolation coordinates or channel map")
    right = np.clip(
        np.searchsorted(nodes, fine, side="right"), 1, len(nodes) - 1
    )
    fraction = (fine - nodes[right - 1]) / (nodes[right] - nodes[right - 1])
    rows = np.repeat(np.arange(fine.size), 2)
    columns = np.column_stack([right - 1, right]).ravel()
    values = np.column_stack([1 - fraction, fraction]).ravel()
    interpolation = csr_matrix(
        (values, (rows, columns)), shape=(fine.size, nodes.size)
    )
    return np.asarray((interpolation.T @ w.T).T)


def project_real_frequency_kernel(nodes_mhz, operator, kernel, block_size=8):
    """Project real kernel K(nu,nu')(...readout,readout) onto output bins.

    Callback arrays have shape (ni,nj,D,D). The result is (bin,D,bin,D).
    Memory is bounded by an angular/frequency tile and the output. Time
    selections can be packed into D. Coordinates, not channel indices,
    are passed to the callback, including cross blocks far from the diagonal.
    """
    nodes, w = np.asarray(nodes_mhz), np.asarray(operator)
    if (
        nodes.ndim != 1
        or not nodes.size
        or not np.all(np.isfinite(nodes))
        or w.ndim != 2
        or w.shape[1] != nodes.size
        or not np.isrealobj(w)
        or not np.all(np.isfinite(w))
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("invalid real covariance projection")
    result = None
    dimension = None
    for i in range(0, nodes.size, block_size):
        si = slice(i, min(i + block_size, nodes.size))
        for j in range(0, nodes.size, block_size):
            sj = slice(j, min(j + block_size, nodes.size))
            value = np.asarray(kernel(nodes[si], nodes[sj]))
            if (
                value.ndim != 4
                or value.shape[:2] != (nodes[si].size, nodes[sj].size)
                or value.shape[-1] != value.shape[-2]
                or not np.isrealobj(value)
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("invalid real frequency covariance block")
            if result is None:
                dimension = value.shape[-1]
                result = np.zeros((w.shape[0], dimension) * 2)
            if value.shape[-1] != dimension:
                raise ValueError("inconsistent readout dimension")
            result += np.einsum(
                "bi,ijpq,cj->bpcq", w[:, si], value, w[:, sj], optimize=True
            )
    return result


def gaussian_visibility_draw(covariance, seed, relative_tolerance=1e-10):
    """Conditional Gaussian draw; reject negative eigenvalues above tolerance.

    This draws beam-weighted visibility statistics, not a Gaussian Faraday
    sky or a high-resolution sky realization. Reuse a draw on repeated days.
    """
    c = np.asarray(covariance)
    if (
        c.ndim != 2
        or c.shape[0] != c.shape[1]
        or not np.isrealobj(c)
        or not np.all(np.isfinite(c))
        or not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0
    ):
        raise ValueError("invalid real draw covariance")
    scale = max(np.max(abs(c)), np.finfo(float).tiny)
    if np.max(abs(c - c.T)) > relative_tolerance * scale:
        raise ValueError("draw covariance is not symmetric")
    eigen, vectors = np.linalg.eigh((c + c.T) / 2)
    if eigen[0] < -relative_tolerance * scale:
        raise ValueError("draw covariance is not positive semidefinite")
    noise = np.random.default_rng(seed).normal(size=c.shape[0])
    sample = vectors @ (np.sqrt(np.maximum(eigen, 0)) * noise)
    return dict(
        sample=sample,
        min_eigenvalue=float(eigen[0]),
        clipped_roundoff_eigenvalues=int(np.count_nonzero(eigen < 0)),
        tolerance=float(relative_tolerance * scale),
    )
