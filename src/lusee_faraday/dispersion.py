"""Faraday depth distributions, their transforms, and the depth-space
template geometry built on top of them.

A NAMING WARNING, because this module used to get it wrong.  The axis
here is Faraday depth ``phi`` in rad/m^2, the Fourier conjugate of
``lambda^2``.  It is NOT the delay ``tau`` (in seconds) conjugate to
``nu``, which is what the refuted Step 4 at the audit-2026-08-18 tag
transformed onto.  The two are related by a MONOTONIC map,
``tau_FD = 2 phi c^2 / (pi nu^3)``, so a cut in one is exactly a cut in
the other and detection is identical either way -- but they have
different units and the same phi sits at different tau in different
bands, so they must not be called by one name.

Owns F(phi) and its transforms (spec S4.1): ``transform`` turns a
depth distribution into P(lambda^2) on the model side; ``depth_power``
turns a measured/model spectrum into Faraday-depth power via a type-3
NUFFT on the true lambda^2 nodes -- NEVER an FFT on a uniform nu grid;
the chirp that removes is spec S4.5. It also owns the shape geometry
of the template built from F(phi) -- folding, the half-power and
mass-quantile knees (S4.2, S4.2.2) -- the real channel response the
shape is measured through, via luseepy's zoom-bin machinery and
``channelization``/``config`` (S4.6: ``zoom_bin_matrix``, ``rmsf``,
``bin_envelope``, ``depth_horizon``), and the coherence and amplitude
brackets that separate the shape prediction from the amplitude this
repository does not predict (S4.4, S4.4.1: ``structure_function``,
``coherence_angle``, ``patch_counts``, ``coherence_tilt``,
``amplitude_bracket``). ``tail_gate_bins``/``tail_gate_fractions`` are
the LST-resolved tail gate's fixed |RM| binning and threshold-to-
fraction arithmetic (S6.14).

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


def depth_power(spectrum, freqs_mhz, phi_out, window=None, eps=1e-12):
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

    That suffix sum is formed as an ACTUAL suffix cumsum and not as
    ``total - prefix``.  The summands ``w * p^-q`` span a dynamic range
    of ``(p_max/p_min)^q``, so the subtractive form cancels
    catastrophically once that exceeds double precision: on the
    committed 1 rad/m^2 display grid the ratio of total to remainder
    reaches 1.5e15 at k = 8 and the subtraction returns exactly zero by
    k = 12, silently snapping every large-k answer onto the k -> inf
    histogram.  A continuous k scan walks straight into it; the
    production geometries (k = inf, 0, -1) never did, which is why it
    survived -- at k = 0 the ratio is 2.2 and the two forms agree to
    3e-15.  Summing the tail among itself is well conditioned because
    every term of ``(e/p)^q`` there lies in (0, 1).
    """
    order = np.argsort(phi_abs)
    p = phi_abs[order]
    w = w2[order]
    q = k + 1.0
    e = np.clip(np.asarray(edges_abs, dtype=float), 0.0, None)
    # ``p^-q`` and ``e^q`` are formed explicitly, so a large enough q
    # overflows to inf and then to nan (inf * 0).  Raise instead: a
    # geometry this steep is indistinguishable from k -> inf anyway,
    # and the caller should say so rather than get a nan.
    if q > 0.0 and p.size:
        pmin = float(p.min())
        emax = float(e.max()) if e.size else 0.0
        span = q * (np.log10(max(emax, 1.0)) - np.log10(max(pmin, 1e-300)))
        if span > 290.0 or -q * np.log10(max(pmin, 1e-300)) > 290.0:
            raise ValueError(
                f"k = {k} is too steep for this depth grid: (e/phi)^(k+1) "
                f"spans 10^{span:.0f} and overflows double precision. Use "
                "k = np.inf, which is the limit it is approaching."
            )
    csum_w = np.concatenate([[0.0], np.cumsum(w)])
    suffix_wp = np.concatenate([np.cumsum((w * p ** (-q))[::-1])[::-1], [0.0]])
    idx = np.searchsorted(p, e, side="right")
    G = csum_w[idx] + e**q * suffix_wp[idx]
    return np.diff(G)


def depth_distribution(phi_col, w2, edges, k=0.0):
    """|w|^2-weighted pushforward of rho(f) ~ f^k through f*phi_col.

    k = np.inf -> histogram of phi_col (all emission behind the column);
    k = 0     -> uniform slab, superposition of top-hats [0, phi_col];
    k = -1    -> all emission local, delta at phi = 0.
    k must be >= -1 (the pushforward CDF is (e/phi)^(k+1), non-integrable
    at f = 0 for k < -1).
    Every finite-k pushforward puts mass at phi = 0 (the near end of
    every column), so ``edges`` must bracket zero: ``edges[0] <= 0 <
    edges[-1]``.  Without that guard ``searchsorted(edges, 0.0,
    'right') - 1`` returns -1 for an all-positive grid and the
    zero-depth mass -- all of it, for k = -1 -- lands silently in the
    LAST bin.  Same style as the k >= -1 check: raise rather than
    return a wrong histogram.  ``k = np.inf`` is exempt: it is a plain
    ``np.histogram`` of phi_col, which drops out-of-range mass in the
    usual way and never touches the zero bin.
    Spec S4.2.  Sums to w2.sum().
    """
    phi_col = np.asarray(phi_col, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float)
    if np.isinf(k):
        H, _ = np.histogram(phi_col, bins=edges, weights=w2)
        return H
    if k < -1.0:
        raise ValueError(
            "k must be >= -1: rho ~ f^k is not integrable at f = 0"
        )
    if not (edges[0] <= 0.0 < edges[-1]):
        raise ValueError(
            "edges must bracket phi = 0 (edges[0] <= 0 < edges[-1]): "
            "the pushforward puts mass at zero depth and there is no "
            f"bin for it in [{edges[0]}, {edges[-1]}]"
        )
    H = np.zeros(edges.size - 1)
    zero_bin = np.searchsorted(edges, 0.0, side="right") - 1
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


def _uniform_abs_grid(phi_abs):
    """Bin edges of a uniform folded |phi| grid whose first edge is 0.

    ``depth_distribution`` consumes per-pixel depths; the two functions
    below consume an already-binned k -> infinity histogram, which is
    the only form the committed products carry (``step5_template.npz``
    stores templates, not the per-band ``w2``).  Both need the same
    guard, so it lives here.
    """
    phi_abs = np.asarray(phi_abs, dtype=float).ravel()
    if phi_abs.size < 2:
        raise ValueError("need at least two bins")
    d = np.diff(phi_abs)
    if not np.allclose(d, d[0], rtol=1e-9, atol=0.0):
        raise ValueError("phi_abs must be a uniform grid of bin centres")
    dphi = float(d[0])
    if dphi <= 0.0:
        raise ValueError("phi_abs must be increasing")
    first = phi_abs[0] - 0.5 * dphi
    # The pushforward drags mass to zero depth from EVERY column, so a
    # grid whose first edge sits above the origin silently loses it --
    # the same failure ``depth_distribution`` raises on, in the folded
    # form.  Tolerance is a fraction of a bin, not exact equality: the
    # committed grid is built by arithmetic on floats.
    if abs(first) > 1e-6 * dphi:
        raise ValueError(
            "phi_abs must be a folded grid whose first edge is zero "
            f"(got {first}); the pushforward puts mass at zero depth "
            "and there is no bin for it"
        )
    return np.concatenate([[0.0], phi_abs + 0.5 * dphi])


def pushforward_histogram(phi_abs, H_far, k):
    """Re-cast a ``k -> infinity`` depth histogram into geometry ``k``.

    ``H_far`` is the folded, ``|w|^2``-weighted histogram of ``|RM|`` --
    that is, ``depth_distribution(..., k=np.inf)`` folded onto ``|phi|``,
    which is exactly the ``k -> infinity`` column of the committed
    template products.  Each of its bins is one population of columns
    at depth ``phi_j``; this spreads each one over ``[0, phi_j]`` with
    the CDF of Equation (pushforward), ``min(e/phi_j, 1)^(k+1)``.

    It is the SAME operator as ``depth_distribution`` -- both call
    ``_pushforward_onesided`` -- applied to binned rather than
    per-pixel input, so the two agree to the binning of ``phi_abs``
    (measured at KS ~ 1e-3 on the committed 1 rad/m^2 display grid,
    against a 1e-2 gate).  That is what makes the geometry scan a
    re-analysis of the stored products rather than a re-run of the
    20-40 minute template job.

    Total mass is conserved: the pushforward REDISTRIBUTES the sky's
    polarized power in depth, it does not reweight it.
    """
    H_far = np.asarray(H_far, dtype=float).ravel()
    phi_abs = np.asarray(phi_abs, dtype=float).ravel()
    if phi_abs.size != H_far.size:
        raise ValueError(
            f"phi_abs ({phi_abs.size}) and H_far ({H_far.size}) must match"
        )
    if np.isinf(k):
        return H_far.copy()
    if k < -1.0:
        raise ValueError(
            "k must be >= -1: rho ~ f^k is not integrable at f = 0"
        )
    edges = _uniform_abs_grid(phi_abs)
    if k == -1.0:
        out = np.zeros_like(H_far)
        out[0] = H_far.sum()
        return out
    return _pushforward_onesided(phi_abs, H_far, edges, k)


def pushforward_signed(phi_signed, H_far, k):
    """Re-cast a SIGNED ``k -> inf`` histogram into geometry ``k``.

    ``pushforward_histogram`` requires a folded grid whose first edge is
    zero, so each sign is re-cast on its own ``|phi|`` grid and the two
    are reassembled -- exactly how ``depth_distribution`` treats the two
    signs internally.

    The signed form exists because the matched filter needs it: the
    observable is the complex ``P = Q + iU``, so its frequency
    covariance is the complex transform of the SIGNED depth
    distribution.  Re-casting the folded histogram instead would model a
    sky whose every column has one sign of RM.  Folding this result
    reproduces ``pushforward_histogram`` on the folded input, which is
    the consistency the tests pin.

    ``phi_signed`` must be a uniform grid symmetric about zero with no
    bin centred there (the committed grid is +-0.5, +-1.5, ...).
    """
    phi_signed = np.asarray(phi_signed, dtype=float).ravel()
    H_far = np.asarray(H_far, dtype=float).ravel()
    if phi_signed.size != H_far.size:
        raise ValueError(
            f"phi_signed ({phi_signed.size}) and H_far ({H_far.size}) "
            "must match"
        )
    pos, neg = phi_signed > 0, phi_signed < 0
    if pos.sum() != neg.sum():
        raise ValueError(
            "phi_signed must be symmetric about zero: got "
            f"{pos.sum()} positive and {neg.sum()} negative bins"
        )
    ph = phi_signed[pos]
    out = np.zeros_like(H_far)
    out[pos] = pushforward_histogram(ph, H_far[pos], k)
    out[neg] = pushforward_histogram(ph, H_far[neg][::-1], k)[::-1]
    return out


def retained_fraction(phi_abs, H_far, cut, k):
    """Template power fraction at ``|phi| >= cut`` under geometry ``k``.

    The systematics-cut statistic of spec S4.10, as a function of the
    emissivity geometry.  Selection is by BIN CENTRE (``phi_abs >=
    cut``), which is the convention ``scripts/step5_detection.py``
    uses; computing it any other way would let the geometry scan and
    the detection table disagree on the same number.
    """
    H = pushforward_histogram(phi_abs, H_far, k)
    total = H.sum()
    if not total > 0.0:
        raise ValueError("total mass must be positive")
    return float(H[np.asarray(phi_abs, dtype=float) >= cut].sum() / total)


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
    if not H.sum() > 0.0:
        raise ValueError("total mass must be positive")
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


def tail_gate_bins(rm_abs, nbins=2000):
    """Fixed |RM| binning for the LST-resolved tail gate (spec S6.14).

    |RM| does not change across LSTs, so it is binned once; the
    per-LST w2 is then accumulated into these bins with a streaming
    ``np.bincount`` (in the caller) instead of storing a full depth
    histogram per LST -- 2000 bins is ~2 MB per band's ``(nlst,
    nbins)`` accumulator versus up to hundreds of MB for a full
    ``phi_edges`` histogram per LST.

    Returns ``(edges, idx)``: ``edges`` has ``nbins + 1`` entries
    spanning ``[0, rm_abs.max()]``; ``idx`` maps each input pixel to
    its bin, clipped so a value at the maximum still lands in the
    last bin rather than falling out of range.
    """
    rm_abs = np.asarray(rm_abs, dtype=float).ravel()
    edges = np.linspace(0.0, rm_abs.max(), nbins + 1)
    idx = np.clip(
        np.searchsorted(edges, rm_abs, side="right") - 1, 0, nbins - 1
    )
    return edges, idx


def tail_gate_fractions(rm_bin_edges, tail_hist, threshold):
    """Per-LST tail fraction above a threshold (spec S6.14).

    ``tail_hist`` is one or more |RM|-binned w2 histograms (shape
    ``(..., nbins)``, built on ``rm_bin_edges`` from ``tail_gate_bins``
    via ``np.bincount`` per LST).  For k=inf the template mass beyond
    a depth T is exactly the w2 weight of pixels with |RM| > T, so no
    depth histogram is needed to get the fraction of mass above
    ``threshold``.

    ``threshold`` must be held FIXED across whatever leading axis
    ``tail_hist`` carries (e.g. LST, Ruling R19): computing it
    separately per row from that row's own weight forces the returned
    fraction to ~1% by the definition of the percentile -- that is
    the tautological gate this replaced, since it measures nothing
    but NumPy's percentile implementation.

    Sums row by row rather than as one vectorised ``(..., above)``
    reduction: NumPy's pairwise summation groups a masked reduction
    over a >1-D array's last axis differently from summing each row
    on its own (verified to differ at the ~1e-16 relative level).
    scripts/step5_template.py's original inline form used the
    per-row loop, and this function must reproduce that original
    row-by-row floating point sum bit-for-bit -- it is the arithmetic
    that built the committed, not-regenerated
    ``generated_data/step5_template*.npz``.
    """
    rm_bin_edges = np.asarray(rm_bin_edges, dtype=float)
    tail_hist = np.asarray(tail_hist, dtype=float)
    above = rm_bin_edges[:-1] > threshold
    flat = tail_hist.reshape(-1, tail_hist.shape[-1])
    frac = np.array([row[above].sum() / row.sum() for row in flat])
    return frac.reshape(tail_hist.shape[:-1])


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
    depth-transformed over the bin centres.

    Returns POWER, not amplitude -- an FWHM taken from this array is a
    power FWHM, a factor 0.734 below the amplitude FWHM that
    Brentjens & de Bruyn's 2 sqrt(3) / dlambda^2 rule quotes.  State
    the convention wherever the width is reported (docs section 10).

    At ``phi0 = 0`` the tone is flat and ``W`` has normalized columns,
    so ``binned`` is exactly 1.0 in every bin: the width returned
    there is set by the bin-CENTRE positions alone (192 zoom bins,
    390.625 Hz apart, spanning 74609 Hz) and carries no information
    about the bin response shapes.  The shapes enter at phi0 != 0,
    through the depth envelope of ``bin_envelope``/``depth_horizon``.
    """
    lam2 = np.asarray(lambda_squared(fine_freqs_mhz), dtype=float)
    tone = np.exp(2j * float(phi0) * lam2)
    binned = np.asarray(W).T @ tone
    return depth_power(binned, bin_freqs_mhz, phi_out, window=window)


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


def structure_function(rm_map, theta_deg, nsamp=200_000, rng=None):
    """RM structure function D(theta) by Monte-Carlo pixel pairs."""
    import healpy as hp

    rng = np.random.default_rng(0) if rng is None else rng
    rm_map = np.asarray(rm_map, dtype=float)
    nside = hp.get_nside(rm_map)
    out = np.empty(len(np.atleast_1d(theta_deg)))
    for i, th in enumerate(np.atleast_1d(theta_deg)):
        pix = rng.integers(0, rm_map.size, nsamp)
        v1 = np.array(hp.pix2vec(nside, pix))
        r = rng.normal(size=(3, nsamp))
        t = r - (r * v1).sum(axis=0) * v1
        t /= np.linalg.norm(t, axis=0)
        a = np.radians(th)
        v2 = np.cos(a) * v1 + np.sin(a) * t
        th2, ph2 = hp.vec2ang(v2.T)
        rm2 = hp.get_interp_val(rm_map, th2, ph2)
        out[i] = np.mean((rm_map[pix] - rm2) ** 2)
    return out


def coherence_angle(theta_deg, D, lam2):
    """theta_c (radians) solving 2 lam2^2 D(theta_c) = 1 (spec S4.4).

    ``D`` is monotonized (running max) before inverting, then
    log-log-interpolated for the root. If the target lies outside the
    sampled range, the result is CLAMPED to the nearest sampled
    ``theta_deg`` rather than extrapolated -- callers must check for
    this.

    **The low-end clamp OVERSTATES theta_c.** ``target <= D[0]`` means
    the root lies BELOW the grid's lower edge (D is a running max and
    the root is where D falls to ``target``), so returning ``th[0]``
    returns something LARGER than the true root: a clamped return is
    an UPPER BOUND on theta_c, not the root. Anything proportional to
    theta_c -- ``amplitude_bracket``'s ``upper``, via
    ``N_patch = omega_beam / theta_c^2`` -- inherits that as an upper
    bound too, and on the real sky the overstatement is large: with
    ``faraday2020v2`` and the 0.2-30 deg grid ``step5_template.py``
    uses, D(0.2 deg) = 96.2 (rad/m^2)^2 against targets 5.01e-5 /
    3.87e-4 / 6.19e-7 at 30 / 50 / 10 MHz, i.e. a root 1385x / 499x /
    12465x below the grid's lower edge under D ~ theta^2, and a
    correspondingly overstated ``upper``.

    Widening the grid does not fix this: at these lambda^4 the root
    sits at sub-arcsecond separations (0.52 / 1.44 / 0.058 arcsec),
    three decades below the map's own nside-512 resolution. The map
    cannot determine theta_c at all; a wider grid would extrapolate
    ``D ~ theta^2``, not measure. Quote a clamped ``upper`` as
    "not computable from this map", never as a value.
    """
    th = np.radians(np.asarray(theta_deg, dtype=float))
    D = np.maximum.accumulate(np.asarray(D, dtype=float))
    target = 1.0 / (2.0 * float(lam2) ** 2)
    if target <= D[0]:
        return float(th[0])
    if target >= D[-1]:
        return float(th[-1])
    return float(np.exp(np.interp(np.log(target), np.log(D), np.log(th))))


def patch_counts(phi_col, w2, edges, theta_c, pix_area):
    """Independent-patch count per depth bin (spec S4.4.1)."""
    phi_col = np.asarray(phi_col, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    s1, _ = np.histogram(phi_col, bins=edges, weights=w2)
    s2, _ = np.histogram(phi_col, bins=edges, weights=w2**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = np.where(s2 > 0, s1**2 / s2, 0.0)
    return np.maximum(1.0, neff * pix_area / float(theta_c) ** 2)


def coherence_tilt(H, npatch):
    """Coherent-limit template: H boosted by the patch count, then
    renormalized to H's total (the tilt is a shape statement, S4.4.1).
    """
    H = np.asarray(H, dtype=float)
    tilted = H * np.asarray(npatch, dtype=float)
    return tilted * (H.sum() / tilted.sum())


def amplitude_bracket(lam2, theta_c, omega_beam, phi_med, sigma_eff=9.8):
    """The S4.4 bracket.  Not a prediction -- two ends with reasons.

    Only ``upper`` -- the incoherent-patch estimate -- contains
    ``theta_c``.  ``lower_slab`` = 1/(|phi_med| lam2) and
    ``lower_dispersion`` = 1/(2 sigma_eff^2 lam2^2) are theta_c-FREE:
    a clamped coherence angle (see ``coherence_angle``) contaminates
    the bracket's UPPER end only, and it contaminates it badly -- on
    the real sky ``upper`` is a clamp-derived upper bound overstated
    by ~3 orders of magnitude, not a measurement.  Do not write that
    the bracket, or its lower end, "derives from theta_c".
    """
    n_patch_tot = max(1.0, float(omega_beam) / float(theta_c) ** 2)
    lam2 = float(lam2)
    return {
        "upper": 1.0 / np.sqrt(n_patch_tot),
        "lower_slab": 1.0 / (abs(float(phi_med)) * lam2),
        "lower_dispersion": 1.0 / (2.0 * float(sigma_eff) ** 2 * lam2**2),
    }
