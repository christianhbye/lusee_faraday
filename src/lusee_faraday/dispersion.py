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


def zoom_bin_matrix(center_mhz):
    """Fine grid, sorted zoom-bin centres, (nfine, 192) weight matrix.

    Built from luseepy's real spectrometer_response_zoom via
    channelization.zoom_weights -- the real response, not a boxcar
    (spec S4.6).  Columns are normalized bin weights.
    """
    from .channelization import (
        PARENT_HALF_WIDTH_HZ,
        zoom_frequency_grid,
        zoom_weights,
    )
    from .config import fine_freqs, parent_centers

    fine = fine_freqs(center_mhz)
    parents = parent_centers(center_mhz)
    bin_f, order = zoom_frequency_grid(parents)
    W = np.zeros((fine.size, bin_f.size))
    cache = {}
    for i, (p, kbin) in enumerate(order):
        if p not in cache:
            off = (fine - parents[p]) * 1e6
            sel = np.abs(off) <= PARENT_HALF_WIDTH_HZ + 1e-6
            cache[p] = (sel, zoom_weights(off[sel]))
        sel, Wz = cache[p]
        W[sel, i] = Wz[:, kbin]
    return fine, np.asarray(bin_f), W


def rmsf(phi0, fine_freqs_mhz, W, bin_freqs_mhz, phi_out, window=None):
    """Delay-power response of the binned system to a tone at phi0.

    The deconvolution kernel of spec S4.1: a unit Faraday tone on the
    fine grid, integrated by the true bin responses, then
    delay-transformed over the bin centres.
    """
    lam2 = np.asarray(lambda_squared(fine_freqs_mhz), dtype=float)
    tone = np.exp(2j * float(phi0) * lam2)
    binned = np.asarray(W).T @ tone
    return delay_power(binned, bin_freqs_mhz, phi_out, window=window)


def bin_envelope(phi, fine_offsets_hz, w, center_mhz):
    """|FT of the bin response| at the Faraday rate of depth phi.

    The multiplicative envelope a channel imposes in Faraday depth
    (spec S4.6): attenuation of a tone at phi integrated by one bin.
    """
    off = np.asarray(fine_offsets_hz, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    freqs = center_mhz + off * 1e-6
    dlam2 = (
        np.asarray(lambda_squared(freqs), dtype=float)
        - lambda_squared(center_mhz)[0]
    )
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    env = np.abs(np.exp(2j * np.outer(phi, dlam2)) @ w)
    return env if env.size > 1 else float(env[0])


def depth_horizon(fine_offsets_hz, w, center_mhz, level=0.5):
    """First depth where the bin envelope falls through ``level``."""
    grid = np.geomspace(0.1, 1e5, 600)
    env = bin_envelope(grid, fine_offsets_hz, w, center_mhz)
    below = np.nonzero(env < level)[0]
    if below.size == 0:
        return float(grid[-1])
    j = below[0]
    lo, hi = (0.0, grid[0]) if j == 0 else (grid[j - 1], grid[j])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bin_envelope(mid, fine_offsets_hz, w, center_mhz) >= level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
