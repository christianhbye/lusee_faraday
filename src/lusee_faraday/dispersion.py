"""Faraday depth distributions and their delay-space transforms.

Owns F(phi) and its transforms (spec S4.1).  Model side: ``transform``
turns a depth distribution into P(lambda^2).  Analysis side:
``delay_power`` turns a measured/model spectrum into delay-space power
via a type-3 NUFFT on the true lambda^2 nodes -- NEVER an FFT on a
uniform nu grid; the chirp that removes is spec S4.5.

Does not import pixel_arm.
"""

import numpy as np

from .conventions import lambda_squared

# The phi grid must span the map maximum (|RM|_max = 2442 rad/m^2 in
# faraday2020v2); spec S3.
PHI_SPAN = 2500.0

# Below this many (source x target) points the exact direct sum is used;
# above it, finufft type 3.  The switch is numerical only -- both compute
# the same sum.
_DIRECT_LIMIT = 4_000_000


def phi_edges(center_mhz, span=PHI_SPAN):
    """Uniform signed depth-bin edges for a band's +-0.1 MHz window.

    Bin width pi / (2 lambda^2_max), lambda^2_max at the window's low
    edge (spec S3): half a turn of Faraday phase per bin.
    """
    lam2_max = float(np.asarray(lambda_squared(center_mhz - 0.1))[0])
    dphi = np.pi / (2.0 * lam2_max)
    n = int(np.ceil(span / dphi))
    return dphi * np.arange(-n, n + 1)


def phi_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[1:] + edges[:-1])


def transform(phi, F, lam2_targets, eps=1e-12):
    """P(lambda^2) = sum_j F_j exp(+2i phi_j lambda^2).

    ``phi`` may be bin centres or raw pixel depths (nonuniform points).
    """
    phi = np.asarray(phi, dtype=float).ravel()
    F = np.asarray(F).ravel().astype(np.complex128)
    s = 2.0 * np.asarray(lam2_targets, dtype=float).ravel()
    if phi.size * s.size <= _DIRECT_LIMIT:
        return (F[None, :] * np.exp(1j * np.outer(s, phi))).sum(axis=1)
    import finufft

    return finufft.nufft1d3(phi, F, s, isign=+1, eps=eps)


def delay_power(spectrum, freqs_mhz, phi_out, window=None, eps=1e-12):
    """|P~(phi)|^2 of a spectrum sampled at arbitrary frequencies.

    Type-3 NUFFT with nodes 2*lambda^2(freq) and targets phi; the
    window (if any) is applied across the frequency samples and the
    result is normalized by sum(window), so a unit tone at depth phi_0
    gives peak power 1 at phi_0.
    """
    s = np.asarray(spectrum, dtype=np.complex128).ravel()
    lam2 = np.asarray(lambda_squared(freqs_mhz), dtype=float).ravel()
    win = (
        np.ones(s.size)
        if window is None
        else np.asarray(window, dtype=float).ravel()
    )
    c = (win * s).astype(np.complex128)
    x = 2.0 * lam2
    t = np.asarray(phi_out, dtype=float).ravel()
    if x.size * t.size <= _DIRECT_LIMIT:
        P = (c[None, :] * np.exp(-1j * np.outer(t, x))).sum(axis=1)
    else:
        import finufft

        P = finufft.nufft1d3(x, c, t, isign=-1, eps=eps)
    return np.abs(P / win.sum()) ** 2


def _pushforward_onesided(phi_abs, w2, edges_abs, k):
    """Per-bin mass of the f^k pushforward, all depths > 0.

    CDF_n(e) = min(e / phi_n, 1)^(k+1); the per-bin mass is the CDF
    difference summed over pixels.  Sorting once gives every edge in
    O(log N): pixels with phi <= e contribute w2 fully; the rest
    contribute w2 * (e / phi)^(k+1), whose pixel sum is a suffix sum.
    """
    order = np.argsort(phi_abs)
    p = phi_abs[order]
    w = w2[order]
    q = k + 1.0
    csum_w = np.concatenate([[0.0], np.cumsum(w)])
    csum_wp = np.concatenate([[0.0], np.cumsum(w * p ** (-q))])
    total_wp = csum_wp[-1]
    e = np.clip(np.asarray(edges_abs, dtype=float), 0.0, None)
    idx = np.searchsorted(p, e, side="right")
    G = csum_w[idx] + e**q * (total_wp - csum_wp[idx])
    return np.diff(G)


def depth_distribution(phi_col, w2, edges, k=0.0):
    """|w|^2-weighted pushforward of rho(f) ~ f^k through f*phi_col.

    k = np.inf -> histogram of phi_col (all emission behind the column);
    k = 0     -> uniform slab, superposition of top-hats [0, phi_col];
    k = -1    -> all emission local, delta at phi = 0.
    k must be >= -1 (the pushforward CDF is (e/phi)^(k+1), non-integrable
    at f = 0 for k < -1).
    Spec S4.2.  Sums to w2.sum().
    """
    phi_col = np.asarray(phi_col, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float)
    H = np.zeros(edges.size - 1)
    zero_bin = np.searchsorted(edges, 0.0, side="right") - 1
    if np.isinf(k):
        H, _ = np.histogram(phi_col, bins=edges, weights=w2)
        return H
    if k < -1.0:
        raise ValueError(
            "k must be >= -1: rho ~ f^k is not integrable at f = 0"
        )
    if k == -1.0:
        H[zero_bin] = w2.sum()
        return H
    pos = phi_col > 1e-12
    neg = phi_col < -1e-12
    H[zero_bin] += w2[~(pos | neg)].sum()
    if pos.any():
        H += _pushforward_onesided(phi_col[pos], w2[pos], edges, k)
    if neg.any():
        e_abs = np.clip(-edges, 0.0, None)[::-1]
        H += _pushforward_onesided(-phi_col[neg], w2[neg], e_abs, k)[::-1]
    return H


def fold_template(centers, H):
    """Fold a signed-grid template onto |phi|; same bin width."""
    centers = np.asarray(centers, dtype=float)
    H = np.asarray(H, dtype=float)
    dphi = centers[1] - centers[0]
    n = int(np.ceil((np.abs(centers).max() + 0.5 * dphi) / dphi))
    edges = dphi * np.arange(n + 1)
    Hf, _ = np.histogram(np.abs(centers), bins=edges, weights=H)
    return 0.5 * (edges[1:] + edges[:-1]), Hf


def half_power_knee(phi_abs, H):
    """The last |phi| where H crosses half its peak (spec S4.2.2)."""
    phi_abs = np.asarray(phi_abs, dtype=float)
    H = np.asarray(H, dtype=float)
    half = 0.5 * H.max()
    above = np.nonzero(H >= half)[0]
    i = above[-1]
    if i + 1 >= H.size or H[i] == H[i + 1]:
        return float(phi_abs[i])
    f = (H[i] - half) / (H[i] - H[i + 1])
    return float(phi_abs[i] + f * (phi_abs[i + 1] - phi_abs[i]))


def mass_quantile_knee(phi_abs, H, q=0.90):
    """Depth containing a fraction ``q`` of the folded template's mass.

    The roll-off statistic of spec S4.2.2.  A CDF quantile, not a
    peak-relative threshold: it never references ``H.max()``, so the
    spike the k=0 slab piles up at the origin cannot move it, and a
    rigid map rotation -- which only permutes pixels -- leaves it
    invariant by construction.  ``half_power_knee`` is the
    peak-relative statistic this replaced; the gates print both.
    """
    phi_abs = np.asarray(phi_abs, dtype=float)
    H = np.asarray(H, dtype=float)
    cum = np.cumsum(H)
    cum = cum / cum[-1]
    return float(phi_abs[np.searchsorted(cum, float(q))])


def weighted_percentiles(values, weights, qs):
    """Weighted percentiles (values at cumulative-weight fractions)."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return np.array(
        [
            v[np.searchsorted(cw, q / 100.0, side="left")]
            for q in np.atleast_1d(qs)
        ]
    )


def bh4_window(n):
    """4-term minimum-sidelobe Blackman-Harris (peak sidelobe ~ -92 dB).

    The window step4_power_spectra.py used; the S4.8 dynamic-range
    budget is computed against exactly this.
    """
    from scipy.signal.windows import blackmanharris

    return blackmanharris(int(n), sym=False)
