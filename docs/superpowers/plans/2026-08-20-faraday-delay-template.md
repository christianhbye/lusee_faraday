# Faraday Delay Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the diffuse Faraday delay-space search template (normalised
shape, bracketed amplitude) and its detectability forecast, per the revised
2026-08-20 spec.

**Architecture:** A new production module `dispersion.py` owns the Faraday
depth distribution `F(phi)`, its transforms (type-3 NUFFT, never a uniform-grid
FFT), the channel-response depth horizon, and the geometry/coherence template
families. `noise.py` (ported, then extended) owns the radiometer noise and the
whitened matched-filter threshold. `response.py` gains one function that turns
the frozen-channel pair-Stokes kernel into per-pair pixel weight maps. Four
`step5_*` scripts assemble templates, envelope tables, sensitivity curves and
figures from those modules.

**Tech Stack:** numpy, healpy, finufft (already a dependency), scipy
(windows), h5py/astropy (map IO), luseepy (`spectrometer_response`,
`spectrometer_response_zoom`, `InstrumentResponse`, `Covariance`), lunarsky.

**Spec:** `docs/superpowers/specs/2026-08-20-faraday-delay-template-design.md`
(read it first — every task cites its sections as S#).

## Global Constraints

- Install with `uv add <pkg>`, NEVER `uv pip install`. Env: `uv sync --extra dev`.
- `JAX_ENABLE_X64=1` must be set before any jax import. Every test module
  starts with `import os; os.environ.setdefault("JAX_ENABLE_X64", "1")` above
  the other imports; scripts import `common` first (it does this).
- Frequencies are in MHz everywhere. `conventions.lambda_squared(freq_mhz)`
  returns m^2 (array, even for scalar input).
- Faraday convention: `(Q + iU)_COSMO * exp(+2i phi lambda^2)`. Dual blocks
  ordered `I, V, P_MINUS, P_PLUS`; pair-Stokes kernel blocks ordered
  `I, Q, U, V`.
- The RM map (`data/faraday2020v2.hdf5`, key `faraday_sky_mean`, RING,
  nside 512) is used at native resolution in production. The nside sweeps in
  the gate tests (Task 7) are the *test of pixelisation* that spec S6.1
  explicitly calls for — the only place regridding it is allowed.
- Production modules must not import `pixel_arm`.
- Formatting: `uv run black src/ tests/ scripts/` (line length 79), then
  `uv run flake8 src/`.
- Tests that need `data/` files skip when the file is missing. The `slow`
  marker means "needs the 631 MB BGL_v16 response artifact" — use it only for
  that.
- Heavy script runs go in the background under `ulimit -v 16000000` with
  absolute log paths under `generated_data/`. 12 GB is not enough.
- Script outputs: npz to `generated_data/`, figures to `report/figures/`
  (`common.GEN_DIR`, `common.FIG_DIR`).
- Commit after every task; imperative, descriptive commit messages like the
  existing history.

## File Structure

- Create `src/lusee_faraday/dispersion.py` — phi grids, `depth_distribution`
  (the `rho ~ f^k` pushforward), `transform`, `delay_power`, `bh4_window`,
  `zoom_bin_matrix`, `rmsf`, `bin_envelope`, `depth_horizon`,
  `fold_template`, `half_power_knee`, `weighted_percentiles`,
  `structure_function`, `coherence_angle`, `patch_counts`, `coherence_tilt`,
  `amplitude_bracket`.
- Create `src/lusee_faraday/noise.py` — verbatim port + `zoom_noise_covariance`,
  `faraday_signal_covariance`, `matched_filter_threshold`,
  `closed_form_threshold`.
- Modify `src/lusee_faraday/response.py` — add `pair_weight_maps`.
- Create `scripts/step5_instrument_envelope.py`, `scripts/step5_template.py`,
  `scripts/step5_sensitivity.py`, `scripts/step5_plots.py`.
- Create `tests/test_dispersion.py`, `tests/test_dispersion_geometry.py`,
  `tests/test_dispersion_gates.py`, `tests/test_noise.py`,
  `tests/test_pair_weights.py`.
- Modify `docs/measurement-model.md` (append §9–§11), `PROGRESS.md`.

---

### Task 1: `dispersion.py` core — grids, transform, delay power, analytic limits (S3, S4.1, S6.3)

**Files:**
- Create: `src/lusee_faraday/dispersion.py`
- Test: `tests/test_dispersion.py`

**Interfaces:**
- Consumes: `conventions.lambda_squared`, `conventions.faraday_phase_cosmo`,
  `config.PHI_FD_POINT`, `config.fine_freqs`.
- Produces:
  - `phi_edges(center_mhz, span=2500.0) -> (nedge,) float` — uniform signed
    bin edges, width `pi / (2 * lambda_squared(center_mhz - 0.1))`, spanning
    at least ±span.
  - `phi_centers(edges) -> (nedge-1,) float`.
  - `transform(phi, F, lam2_targets) -> (ntarget,) complex` — the model-side
    RM-synthesis sum `sum_j F_j exp(+2i phi_j lam2)`. `phi` are nonuniform
    points (bin centres OR raw pixel depths — S3 says feed pixels directly).
  - `delay_power(spectrum, freqs_mhz, phi_out, window=None) -> (nphi,) float`
    — the analysis-side type-3 NUFFT `|sum_f win_f s_f exp(-2i lam2_f phi)|^2
    / (sum win)^2`.

- [ ] **Step 1: Write the failing tests** — `tests/test_dispersion.py`:

```python
"""Analytic limits of the dispersion module (spec S6.3)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp
from lusee_faraday.config import PHI_FD_POINT, fine_freqs
from lusee_faraday.conventions import faraday_phase_cosmo, lambda_squared

FREQS_30 = fine_freqs(30.0)[::64]  # 256 fine frequencies, +-0.1 MHz


def test_phi_edges_width_and_span():
    edges = dsp.phi_edges(30.0)
    dphi = np.diff(edges)
    lam2_max = float(np.asarray(lambda_squared(29.9)))
    assert np.allclose(dphi, np.pi / (2 * lam2_max))
    assert np.isclose(dphi[0], 0.016, atol=2e-3)  # spec S3 number
    assert edges[0] <= -2500.0 and edges[-1] >= 2500.0
    # 10 MHz: 0.0017 rad/m^2 bins (spec S3)
    assert np.isclose(np.diff(dsp.phi_edges(10.0))[0], 0.0017, atol=2e-4)


def test_delta_is_pure_winding():
    """F = delta(phi - PHI_FD_POINT) -> the repo's COSMO Faraday phase."""
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(np.array([PHI_FD_POINT]), np.array([1.0]), lam2)
    expected = faraday_phase_cosmo(np.array([PHI_FD_POINT]), FREQS_30)[0]
    np.testing.assert_allclose(P, expected, rtol=0, atol=1e-9)


def test_tophat_is_sinc_with_the_right_factor():
    """F uniform on [0, Phi] -> |sin(Phi lam2)/(Phi lam2)|, NOT sinc(2...).

    Spec S6.3: under e^{+2i phi lam2}, Int_0^1 e^{2 i f Phi lam2} df has
    modulus |sin(Phi lam2)/(Phi lam2)|.
    """
    Phi = 25.0
    n = 1 << 17
    dphi = Phi / n
    phi = (np.arange(n) + 0.5) * dphi
    F = np.full(n, 1.0 / n)  # unit total emission
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    x = Phi * lam2
    expected = np.abs(np.sin(x) / x)
    keep = expected > 0.05
    np.testing.assert_allclose(np.abs(P)[keep], expected[keep], rtol=1e-3)


def test_gaussian_is_burn():
    """F Gaussian width sigma -> |P| = exp(-2 sigma^2 lam2^2)."""
    sigma = 0.05
    n = 1 << 15
    phi = np.linspace(-8 * sigma, 8 * sigma, n)
    F = np.exp(-0.5 * (phi / sigma) ** 2)
    F /= F.sum()
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    expected = np.exp(-2.0 * sigma**2 * lam2**2)
    np.testing.assert_allclose(np.abs(P), expected, rtol=1e-3)


def test_delay_power_recovers_a_single_depth():
    """delay_power inverts transform: peak at the injected depth."""
    phi0 = 120.0
    freqs = fine_freqs(30.0)[::16]  # 1024 points
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(0.0, 300.0, 0.25)
    p = dsp.delay_power(spec, freqs, phi_out)
    assert abs(phi_out[np.argmax(p)] - phi0) < 1.0
    assert np.isclose(p.max(), 1.0, rtol=1e-6)  # unit tone, normalized
```

Note the Gaussian test uses `sigma = 0.05`: at `lam2 ~ 100`,
`exp(-2 sigma^2 lam2^4)` with a large sigma would underflow; the Burn law in
`lam2` needs `2 sigma^2 lam2^2 ~ O(1)` over the fine window to be a
non-trivial check (`lam2` varies 99.2–100.5 m^2, so this probes the curve).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dispersion.py -v`
Expected: FAIL with `ModuleNotFoundError: lusee_faraday.dispersion` (or
`ImportError`).

- [ ] **Step 3: Write the implementation** — `src/lusee_faraday/dispersion.py`:

```python
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
    lam2_max = float(np.asarray(lambda_squared(center_mhz - 0.1)))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dispersion.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/dispersion.py tests/test_dispersion.py
uv run flake8 src/lusee_faraday/dispersion.py
git add src/lusee_faraday/dispersion.py tests/test_dispersion.py
git commit -m "Add dispersion.py: phi grids, RM-synthesis transform, delay power"
```

---

### Task 2: The geometry pushforward, folding, knee, weighted percentiles (S4.2, S4.2.2)

**Files:**
- Modify: `src/lusee_faraday/dispersion.py`
- Test: `tests/test_dispersion_geometry.py`

**Interfaces:**
- Produces:
  - `depth_distribution(phi_col, w2, edges, k=0.0) -> (nbin,) float` — the
    `|w|^2`-weighted pushforward of `rho(f) ~ f^k` through
    `phi(f) = f * phi_col` (S4.2). `k=np.inf` = the RM histogram (all
    emission behind the column); `k=0` = uniform slab (fiducial,
    superposition of top-hats); `k=-1` = all local (delta at phi=0).
    Sums to `w2.sum()` exactly. Exact per-bin masses via the analytic CDF
    `min(e/phi_col, 1)^(k+1)` — no `f` quadrature, no sampling noise.
  - `fold_template(centers, H) -> (phi_abs, H_folded)` — fold onto |phi|
    (positive-half grid of the same width).
  - `half_power_knee(phi_abs, H) -> float` — the last |phi| where H crosses
    half its peak, linearly interpolated (S4.2.2).
  - `weighted_percentiles(values, weights, qs) -> array` — for the
    beam-weighted p50/p90/p99/p99.9 the spec's comparisons switch to.

- [ ] **Step 1: Write the failing tests** — `tests/test_dispersion_geometry.py`:

```python
"""Geometry knob, folding, knee (spec S4.2, S4.2.2)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from lusee_faraday import dispersion as dsp


def test_k_infinite_is_the_rm_histogram():
    phi_col = np.array([0.5, 1.5, 1.6, -2.5])
    w2 = np.array([1.0, 2.0, 3.0, 4.0])
    edges = np.arange(-3.0, 3.5, 1.0)
    H = dsp.depth_distribution(phi_col, w2, edges, k=np.inf)
    expected, _ = np.histogram(phi_col, bins=edges, weights=w2)
    np.testing.assert_allclose(H, expected)


def test_k_zero_is_a_superposition_of_tophats():
    """Two pixels, closed form: uniform density on [0, phi_col]."""
    phi_col = np.array([4.0, -2.0])
    w2 = np.array([1.0, 2.0])
    edges = np.arange(-3.0, 6.0, 1.0)  # -3..5
    H = dsp.depth_distribution(phi_col, w2, edges, k=0.0)
    # pixel 1: density 1/4 on (0,4); pixel 2: density 2/2 = 1 on (-2,0)
    expected = np.array([0.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.0])
    np.testing.assert_allclose(H, expected, atol=1e-12)
    assert np.isclose(H.sum(), w2.sum())


def test_k_one_cdf_power():
    """rho ~ f: CDF (e/phi_col)^2. One pixel phi_col=2 over edges 0,1,2."""
    H = dsp.depth_distribution(
        np.array([2.0]), np.array([1.0]), np.array([0.0, 1.0, 2.0]), k=1.0
    )
    np.testing.assert_allclose(H, [0.25, 0.75])


def test_k_minus_one_is_all_local():
    edges = np.arange(-2.0, 2.5, 1.0)
    H = dsp.depth_distribution(
        np.array([100.0, -50.0]), np.array([1.0, 3.0]), edges, k=-1.0
    )
    expected = np.array([0.0, 0.0, 4.0, 0.0])  # bin (0, 1) holds phi=0+
    np.testing.assert_allclose(H, expected)


def test_support_extends_to_phi_col():
    """S6.4 extent clause: the k=0 top-hat reaches its column depth."""
    edges = dsp.phi_edges(30.0)
    H = dsp.depth_distribution(
        np.array([2000.0]), np.array([1.0]), edges, k=0.0
    )
    c = dsp.phi_centers(edges)
    assert H[np.searchsorted(edges, 1999.0) - 1] > 0
    assert H[c > 2001.0].sum() == 0.0


def test_fold_and_knee():
    edges = np.arange(-10.0, 10.5, 1.0)
    c = dsp.phi_centers(edges)
    H = np.where(np.abs(c) < 4, 1.0, 0.0)  # top-hat on |phi| < 4
    phi_abs, Hf = dsp.fold_template(c, H)
    knee = dsp.half_power_knee(phi_abs, Hf)
    assert 3.0 < knee < 4.5


def test_weighted_percentiles():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 97.0])
    p = dsp.weighted_percentiles(v, w, [50.0, 99.0])
    assert p[0] == 4.0 and p[1] == 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dispersion_geometry.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'depth_distribution'`.

- [ ] **Step 3: Implement** — append to `dispersion.py`:

```python
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
    Requires k > -1 otherwise (the pushforward CDF is (e/phi)^(k+1)).
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
    if k <= -1.0:
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


def weighted_percentiles(values, weights, qs):
    """Weighted percentiles (values at cumulative-weight fractions)."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return np.array(
        [v[np.searchsorted(cw, q / 100.0, side="left")] for q in
         np.atleast_1d(qs)]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dispersion_geometry.py tests/test_dispersion.py -v`
Expected: all PASS.

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/dispersion.py tests/test_dispersion_geometry.py
git add -A src tests
git commit -m "Add the f^k geometry pushforward, template folding and the knee"
```

---

### Task 3: The chirp test — NUFFT vs uniform-grid FFT (S4.5, S6.8) and the BH4 window

**Files:**
- Modify: `src/lusee_faraday/dispersion.py` (add `bh4_window`)
- Test: `tests/test_dispersion.py` (append)

**Interfaces:**
- Produces: `bh4_window(n) -> (n,) float` — 4-term minimum-sidelobe
  Blackman-Harris (`scipy.signal.windows.blackmanharris`), the window
  `step4_power_spectra.py` used and S4.8 budgets.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_dispersion.py`:

```python
def _fwhm(x, y):
    half = 0.5 * y.max()
    above = np.nonzero(y >= half)[0]
    return x[above[-1]] - x[above[0]]


def test_nufft_beats_fft_on_a_single_depth():
    """S6.8: the chirp is an analysis artifact; the NUFFT removes it.

    A single depth at 30 MHz, phi0 = 600 (chirp ~ 5 resolution elements
    per the spec's table): the uniform-nu FFT smears it >= 4x wider than
    the type-3 NUFFT on the same samples.
    """
    phi0 = 600.0
    freqs = fine_freqs(30.0)[::4]  # 4096 uniform samples
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    spec = np.exp(2j * phi0 * lam2)

    phi_out = np.arange(560.0, 640.0, 0.05)
    p_nufft = dsp.delay_power(spec, freqs, phi_out)
    w_nufft = _fwhm(phi_out, p_nufft)

    # FFT on the uniform nu grid; map delay bins to phi by linearizing
    # lambda^2(nu) at the band centre.
    n = freqs.size
    P = np.fft.fftshift(np.fft.fft(spec)) / n
    dnu_hz = (freqs[1] - freqs[0]) * 1e6
    bw = n * dnu_hz
    nu0 = 30e6
    lam2_0 = float(np.asarray(lambda_squared(30.0)))
    # delay bin k <-> phase rate 2*pi*k/bw <-> phi = pi*k*nu0/(2*bw*lam2_0)
    k = np.arange(n) - n // 2
    phi_fft = np.pi * k * nu0 / (2.0 * bw * lam2_0)
    p_fft = np.abs(P) ** 2
    sel = np.abs(phi_fft - phi0) < 60.0
    w_fft = _fwhm(phi_fft[sel], p_fft[sel])

    assert w_fft / w_nufft >= 4.0  # spec measured 11.80 / 2.36 = 5.0
    # and the NUFFT peak is within one bin of the truth
    assert abs(phi_out[np.argmax(p_nufft)] - phi0) < 0.5


def test_bh4_window_sidelobe_level():
    """The 4-term Blackman-Harris peak sidelobe is ~ -92 dB (2.5e-5)."""
    n = 4096
    win = dsp.bh4_window(n)
    freqs = fine_freqs(30.0)[::4]
    phi_out = np.arange(0.0, 400.0, 0.1)
    p = dsp.delay_power(np.ones(n), freqs, phi_out, window=win)
    # main lobe is at phi = 0; measure the highest sidelobe beyond it
    side = p[phi_out > 15.0].max()
    assert np.sqrt(side) < 5e-5
    assert np.sqrt(side) > 5e-7  # a window this good would be a bug
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dispersion.py -v -k "nufft or bh4"`
Expected: FAIL (`bh4_window` missing; the chirp test may import-error first).

- [ ] **Step 3: Implement** — append to `dispersion.py`:

```python
def bh4_window(n):
    """4-term minimum-sidelobe Blackman-Harris (peak sidelobe ~ -92 dB).

    The window step4_power_spectra.py used; the S4.8 dynamic-range
    budget is computed against exactly this.
    """
    from scipy.signal.windows import blackmanharris

    return blackmanharris(int(n), sym=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dispersion.py -v`
Expected: all PASS. If the chirp ratio assert fails *low*, the FFT phi
mapping is wrong — check the `phi_fft` line against S4.5's chirp table, do
not weaken the 4.0.

- [ ] **Step 5: Commit**

```bash
uv run black src tests && git add -A src tests
git commit -m "Pin the chirp as an analysis artifact and the BH4 sidelobe floor"
```

---

### Task 4: Channel-response machinery — `zoom_bin_matrix`, `rmsf`, `bin_envelope`, `depth_horizon` (S4.6, S6.7, S6.9 instrument side)

**Files:**
- Modify: `src/lusee_faraday/dispersion.py`
- Test: `tests/test_dispersion.py` (append)

**Interfaces:**
- Consumes: `channelization.parent_weights`, `channelization.zoom_weights`,
  `channelization.zoom_frequency_grid`, `config.fine_freqs`,
  `config.parent_centers`.
- Produces:
  - `zoom_bin_matrix(center_mhz) -> (fine_freqs_mhz, bin_freqs_mhz, W)` —
    `W` is `(nfine, 192)`, column-normalized true zoom responses for the 3
    parents inside the fine window, columns sorted by bin frequency.
  - `rmsf(phi0, fine_freqs_mhz, W, bin_freqs_mhz, phi_out, window=None) ->
    (nphi,) float` — the delay-power response of the channelized system to a
    unit Faraday tone at `phi0` (deconvolution kernel, S4.1/S4.6).
  - `bin_envelope(phi, fine_offsets_hz, w, center_mhz) -> float or array` —
    `|sum_f w_f exp(2i phi (lam2_f - lam2_center))|` with `w` normalized.
  - `depth_horizon(fine_offsets_hz, w, center_mhz, level=0.5) -> float` —
    the depth where the envelope first falls through `level`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_dispersion.py`:

```python
from lusee_faraday.channelization import parent_weights, zoom_weights


def _fine_offsets():
    return np.arange(-50000.0, 50000.0 + 1.0, 12.20703125)


def test_boxcar_rmsf_widths():
    """S6.7: the rectangular idealisation gives 2 sqrt(3) / Dlam2."""
    expected = {50.0: 12.0, 30.0: 2.60, 10.0: 0.096}
    for band, width in expected.items():
        freqs = fine_freqs(band)[::8]
        phi_out = np.arange(0.0, 40.0 * width, width / 50.0)
        p = dsp.delay_power(np.ones(freqs.size), freqs, phi_out)
        amp = np.sqrt(p)
        half = np.nonzero(amp >= 0.5)[0]
        fwhm = 2.0 * phi_out[half[-1]]  # symmetric about 0
        assert np.isclose(fwhm, width, rtol=0.08), (band, fwhm)


def test_depth_horizon_pins_the_s46_table():
    """S6.9 (instrument side): 50% depths of the real bin responses."""
    off = _fine_offsets()
    wp = parent_weights(off)
    wz = zoom_weights(off)[:, 0]
    parent_expect = {50.0: 58.7, 30.0: 13.3, 10.0: 2.7}
    zoom_expect = {50.0: 2830.0, 30.0: 613.0, 10.0: 24.0}
    for band in (50.0, 30.0, 10.0):
        hp_ = dsp.depth_horizon(off, wp, band)
        hz_ = dsp.depth_horizon(off, wz, band)
        assert np.isclose(hp_, parent_expect[band], rtol=0.05), (band, hp_)
        assert np.isclose(hz_, zoom_expect[band], rtol=0.05), (band, hz_)


def test_zoom_bin_matrix_shape_and_normalization():
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    assert W.shape == (fine.size, 192) and bins.size == 192
    np.testing.assert_allclose(W.sum(axis=0), 1.0, rtol=1e-9)
    assert np.all(np.diff(bins) > 0)


def test_rmsf_peaks_at_the_probe_depth_inside_the_horizon():
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    phi_out = np.arange(0.0, 200.0, 0.2)
    r = dsp.rmsf(100.0, fine, W, bins, phi_out)
    assert abs(phi_out[np.argmax(r)] - 100.0) < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dispersion.py -v -k "boxcar or horizon or zoom_bin or rmsf"`
Expected: FAIL with missing attributes.

- [ ] **Step 3: Implement** — append to `dispersion.py`:

```python
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
    dlam2 = np.asarray(lambda_squared(freqs), dtype=float) - float(
        np.asarray(lambda_squared(center_mhz))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dispersion.py -v`
Expected: all PASS. The horizon pins (58.7/13.3/2.7 parent, 2830/613/24
zoom) are quoted in the paper — if any lands outside 5%, investigate the
envelope recipe (normalization, reference `lam2` at the bin centre) before
touching the tolerance; these values must be reproducible (S5).

- [ ] **Step 5: Commit**

```bash
uv run black src tests && git add -A src tests
git commit -m "Add the real-response RMSF and the Faraday depth horizon"
```

---

### Task 5: The window dynamic-range budget (S4.8, S6.10)

**Files:**
- Test: `tests/test_dispersion.py` (append)

**Interfaces:**
- Consumes: `delay_power`, `bh4_window`, `config.fine_freqs`.

- [ ] **Step 1: Write the test** — append to `tests/test_dispersion.py`:

```python
def test_foreground_sidelobe_budget():
    """S6.10: a phi~0 leakage foreground at |P|/I = 0.15 through BH4.

    The budget: sidelobe amplitude in the roll-off region ~ 0.15 *
    2.5e-5 = 3.8e-6.  Adequate against the bracket's optimistic end
    (1e-4); inadequate against the 1e-6 floor -- both reported.
    """
    freqs = fine_freqs(30.0)[::4]  # 4096 samples
    # smooth synchrotron-sloped foreground at the PROGRESS.md level
    fg = 0.15 * (freqs / 30.0) ** (-2.5)
    win = dsp.bh4_window(freqs.size)
    phi_out = np.arange(0.0, 2500.0, 1.0)
    p = dsp.delay_power(fg.astype(complex), freqs, phi_out, window=win)
    contamination = float(np.sqrt(p[phi_out > 200.0].max()))
    # disqualification threshold: the optimistic bracket end (S4.8)
    assert contamination < 1e-5
    assert contamination > 1e-8  # sanity: the foreground exists
    print(
        f"\nsidelobe contamination amplitude {contamination:.2e}; "
        f"vs bracket ends 1e-4 (ratio {contamination / 1e-4:.2e}) "
        f"and 1e-6 (ratio {contamination / 1e-6:.2e})"
    )


def test_boxcar_would_fail_the_budget():
    """Without the window the foreground floods the roll-off region."""
    freqs = fine_freqs(30.0)[::4]
    fg = 0.15 * (freqs / 30.0) ** (-2.5)
    phi_out = np.arange(0.0, 2500.0, 1.0)
    p = dsp.delay_power(fg.astype(complex), freqs, phi_out)
    assert np.sqrt(p[phi_out > 200.0].max()) > 1e-5
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_dispersion.py -v -k budget -s`
Expected: PASS, printing the budget numbers (~3.8e-6 contamination). If the
BH4 number exceeds 1e-5 that is a *finding* (S4.8 says it would be
disqualifying at the optimistic end) — stop and report, do not tune.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dispersion.py
git commit -m "Compute the BH4 window budget against the amplitude bracket"
```

---

### Task 6: Zoom aliasing (S4.6 item 3, S6.13)

**Files:**
- Test: `tests/test_dispersion.py` (append)

- [ ] **Step 1: Write the test** — append to `tests/test_dispersion.py`:

```python
def test_zoom_aliasing_image_is_modelled():
    """S6.13: a depth beyond the zoom fold appears at the aliased
    position, and the rmsf model predicts the same image.

    The zoom delay range wraps with period dphi_wrap =
    2 pi nu / (4 lam2 * 390.625 Hz) ~ 1208 rad/m^2 at 30 MHz, so
    phi0 = 900 folds to |900 - 1208| ~ 308.
    """
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    lam2 = np.asarray(lambda_squared(fine), dtype=float)
    lam2_0 = float(np.asarray(lambda_squared(30.0)))
    phi0 = 900.0
    wrap = 2.0 * np.pi * 30e6 / (4.0 * lam2_0 * 390.625)
    phi_img = abs(phi0 - wrap)  # ~ 308

    tone = np.exp(2j * phi0 * lam2)
    binned = W.T @ tone
    win = dsp.bh4_window(bins.size)
    phi_out = np.arange(0.0, 700.0, 0.5)
    measured = dsp.delay_power(binned, bins, phi_out, window=win)
    model = dsp.rmsf(phi0, fine, W, bins, phi_out, window=win)

    peak_meas = phi_out[np.argmax(measured)]
    peak_model = phi_out[np.argmax(model)]
    assert abs(peak_meas - phi_img) < 10.0, (peak_meas, phi_img)
    assert abs(peak_meas - peak_model) < 1.0
    # the true depth itself must NOT be the peak: it is beyond the fold
    near_true = measured[np.abs(phi_out - (phi0 - 300.0)) < 5.0]
    assert measured.max() > 3.0 * near_true.max()
```

Note: `measured` and `model` share the binning code path by construction —
the *independent* pin is `phi_img` from the wrap arithmetic. If the peak is
not near 308, the folding is not where `channelization.py`'s docstring says
it is ("removed downstream"); that is the scope addition the spec's risk
list warns about — stop and report at this gate, per S4.6 item 3.

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_dispersion.py -v -k aliasing`
Expected: PASS (or a reportable finding, see the note).

- [ ] **Step 3: Commit**

```bash
git add tests/test_dispersion.py
git commit -m "Pin the zoom fold: an out-of-range depth images where the response says"
```

---

### Task 7: THE ACCEPTANCE GATES — shape invariance under refinement and null rotation (S6.1, S6.2) — **STOP POINT**

**Files:**
- Test: `tests/test_dispersion_gates.py` (create)

**Interfaces:**
- Consumes: `dispersion.depth_distribution`, `dispersion.phi_edges`,
  `dispersion.half_power_knee`, `dispersion.fold_template`;
  `data/faraday2020v2.hdf5` (skip when absent).

These are the direct rebuttals of the audit's two findings. **If either
fails, the design is refuted and nothing downstream is worth building
(spec S7 step 2). Stop and report — do not continue to Task 8.**

- [ ] **Step 1: Write the tests** — `tests/test_dispersion_gates.py`:

```python
"""Acceptance gates on the real RM map (spec S6.1, S6.2, S6.4, S6.6)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp

DATA = Path(__file__).resolve().parents[1] / "data"
RM_FILE = DATA / "faraday2020v2.hdf5"

needs_rm = pytest.mark.skipif(
    not RM_FILE.exists(), reason="needs data/faraday2020v2.hdf5"
)


def _rm_map():
    import h5py

    with h5py.File(RM_FILE, "r") as f:
        return np.asarray(f["faraday_sky_mean"][:], dtype=float)


def _rm_at_nside(rm512, nside):
    import healpy as hp

    if nside == 512:
        return rm512
    if nside < 512:
        return hp.ud_grade(rm512, nside)
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return hp.get_interp_val(rm512, th, ph)


def _normalized_template(rm, k):
    edges = dsp.phi_edges(30.0)
    w2 = np.full(rm.size, 1.0 / rm.size)
    H = dsp.depth_distribution(rm, w2, edges, k=k)
    return dsp.phi_centers(edges), H / H.sum()


def _cdf_distance(Ha, Hb):
    return float(np.abs(np.cumsum(Ha) - np.cumsum(Hb)).max())


@needs_rm
@pytest.mark.parametrize("k", [np.inf, 0.0])
def test_gate1_shape_invariance_under_refinement(k):
    """S6.1: nside 256/512/1024/2048 templates agree; the old coherent
    amplitude falls.  Tolerance: 1% Kolmogorov distance, 2% knee shift.
    """
    from lusee_faraday.conventions import lambda_squared

    rm512 = _rm_map()
    lam2_0 = float(np.asarray(lambda_squared(30.0)))
    templates, knees, coherent = {}, {}, {}
    for nside in (256, 512, 1024, 2048):
        rm = _rm_at_nside(rm512, nside)
        c, H = _normalized_template(rm, k)
        templates[nside] = H
        knees[nside] = dsp.half_power_knee(*dsp.fold_template(c, H))
        # the audit's shot-noise observable: |mean e^{2 i phi lam2}|^2
        z = np.exp(2j * rm * lam2_0)
        coherent[nside] = abs(z.mean()) ** 2
    pairs = [(256, 512), (512, 1024), (1024, 2048)]
    for a, b in pairs:
        d = _cdf_distance(templates[a], templates[b])
        assert d <= 0.01, (a, b, d)
        rel = abs(knees[a] - knees[b]) / knees[b]
        assert rel <= 0.02, (a, b, knees)
    # contrast: the coherent power is NOT invariant (it fell ~1/N_pix)
    assert coherent[2048] < 0.5 * coherent[256], coherent
    print(f"\nk={k}: knees {knees}; coherent power {coherent}")


@needs_rm
def test_gate2_shape_invariance_under_null_rotation():
    """S6.2: a rigid grid rotation is physically null.  It moved the old
    |P| by 7.2x; the normalised template must be stable.
    """
    import healpy as hp

    rm = _rm_map()
    rot = hp.Rotator(rot=(40.0, 25.0, 10.0), deg=True)
    rm_rot = rot.rotate_map_pixel(rm)
    for k in (np.inf, 0.0):
        c, H = _normalized_template(rm, k)
        _, Hr = _normalized_template(rm_rot, k)
        d = _cdf_distance(H, Hr)
        assert d <= 0.01, (k, d)
        knee = dsp.half_power_knee(*dsp.fold_template(c, H))
        knee_r = dsp.half_power_knee(*dsp.fold_template(c, Hr))
        assert abs(knee - knee_r) / knee <= 0.02


@needs_rm
def test_gate_knee_location_and_extent():
    """S6.4: knee between p50 and p99 of |RM|; support reaches max."""
    rm = _rm_map()
    c, H = _normalized_template(rm, 0.0)
    phi_abs, Hf = dsp.fold_template(c, H)
    knee = dsp.half_power_knee(phi_abs, Hf)
    p50, p99 = np.percentile(np.abs(rm), [50.0, 99.0])
    assert p50 <= knee <= p99, (knee, p50, p99)
    # extent: nonzero mass out to the map maximum (lower bound, S4.2)
    mx = np.abs(rm).max()
    assert Hf[phi_abs > 0.98 * mx].sum() > 0
```

- [ ] **Step 2: Run the gates**

Run: `uv run pytest tests/test_dispersion_gates.py -v -s`
Expected: PASS with the knee and coherent-power sequences printed
(runtime a few minutes; the nside-2048 interpolation is ~50M points).

- [ ] **Step 3: STOP-POINT review**

If gate 1 or gate 2 FAILS: **stop the plan**. Commit the failing tests with
an `xfail` marker and a message stating the design is refuted at which gate,
and report to the user (spec S7 step 2). Do not proceed to Task 8.

- [ ] **Step 4: Commit (gates green)**

```bash
uv run black tests/test_dispersion_gates.py
git add tests/test_dispersion_gates.py
git commit -m "Pass the acceptance gates: shape invariant, amplitude was the shot noise"
```

---

### Task 8: Converged-regime agreement (S6.6)

**Files:**
- Test: `tests/test_dispersion_gates.py` (append)

- [ ] **Step 1: Write the test** — append:

```python
WMAP_FILE = DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits"

needs_wmap = pytest.mark.skipif(
    not WMAP_FILE.exists(), reason="needs the WMAP K map"
)


def _wmap_qu():
    import healpy as hp
    from astropy.io import fits

    from lusee_faraday.config import T_CMB

    x = 6.62607015e-34 * 23e9 / (1.380649e-23 * T_CMB)
    fconv = x**2 * np.exp(x) / (np.exp(x) - 1) ** 2
    with fits.open(WMAP_FILE) as h:
        d = h["Stokes Maps"].data
        Q = d["Q_POLARISATION"].astype(np.float64) * 1e-3 * fconv
        U = d["U_POLARISATION"].astype(np.float64) * 1e-3 * fconv
    return hp.reorder(Q, n2r=True), hp.reorder(U, n2r=True)


@needs_rm
@needs_wmap
def test_converged_regime_points_match_direct_sum():
    """S6.6: the RM x 0.02 positive control.  In the converged regime
    the type-3 NUFFT on raw pixel depths reproduces the direct coherent
    sum to four digits, with the real polarised sky as weights.
    """
    from lusee_faraday.config import fine_freqs
    from lusee_faraday.conventions import lambda_squared

    rm = 0.02 * _rm_map()
    Q, U = _wmap_qu()
    c = (Q + 1j * U) / len(rm)
    freqs = fine_freqs(30.0)[::256]  # 64 frequencies
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    # direct chunked sum
    direct = np.zeros(lam2.size, dtype=complex)
    for i in range(0, rm.size, 500_000):
        s = slice(i, i + 500_000)
        direct += np.exp(2j * np.outer(lam2, rm[s])) @ c[s]
    nufft = dsp.transform(rm, c, lam2)
    np.testing.assert_allclose(nufft, direct, rtol=1e-4)
```

- [ ] **Step 2: Run, then commit**

Run: `uv run pytest tests/test_dispersion_gates.py -v -k converged`
Expected: PASS (this forces the finufft path: 3.1M x 64 > the direct-sum
size switch).

```bash
git add tests/test_dispersion_gates.py
git commit -m "Reproduce the converged-regime control through the NUFFT point path"
```

---

### Task 9: Coherence bracket and amplitude bracket (S4.4, S4.4.1, S6.15 machinery)

**Files:**
- Modify: `src/lusee_faraday/dispersion.py`
- Test: `tests/test_dispersion_geometry.py` (append)

**Interfaces:**
- Produces:
  - `structure_function(rm_map, theta_deg, nsamp=200_000, rng=None) ->
    (ntheta,) float` — `D_phi(theta) = <(RM(n1) - RM(n2))^2>` at angular
    separations `theta_deg`, Monte-Carlo pairs, interpolated map lookup.
  - `coherence_angle(theta_deg, D, lam2) -> float` — `theta_c` (radians)
    solving `2 lam2^2 D(theta_c) = 1` (S4.4 upper bound; monotonized,
    log-interpolated, clamped to the sampled range).
  - `patch_counts(phi_col, w2, edges, theta_c, pix_area) -> (nbin,) float`
    — per depth bin, `N_patch = max(1, N_eff * pix_area / theta_c^2)` with
    `N_eff = (sum w2)^2 / sum w2^2` over the bin's pixels.
  - `coherence_tilt(H, npatch) -> (nbin,) float` — the coherent-limit
    template `H * npatch`, normalized to `H.sum()` (S4.4.1: per depth
    cell, coherent addition boosts power by the patch count).
  - `amplitude_bracket(lam2, theta_c, omega_beam, phi_med, sigma_eff=9.8)
    -> dict(upper, lower_slab, lower_dispersion)` (S4.4; slab floor is
    `1/(phi lam2)` per the revised spec, matching test 6.3's convention).

- [ ] **Step 1: Write the failing tests** — append to
  `tests/test_dispersion_geometry.py`:

```python
import pytest

healpy = pytest.importorskip("healpy")


def test_structure_function_of_a_smooth_map_scales_as_theta_squared():
    import healpy as hp

    nside = 64
    th, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    m = np.cos(th)  # smooth dipole-like scalar
    thetas = np.array([1.0, 2.0, 4.0])
    D = dsp.structure_function(
        m, thetas, nsamp=40_000, rng=np.random.default_rng(1)
    )
    # D(theta) ~ c theta^2 for a smooth field: ratios 4 and 16
    assert np.isclose(D[1] / D[0], 4.0, rtol=0.3)
    assert np.isclose(D[2] / D[0], 16.0, rtol=0.3)


def test_coherence_angle_analytic():
    theta_deg = np.linspace(0.1, 30.0, 300)
    c = 25.0
    D = c * np.radians(theta_deg) ** 2
    lam2 = 100.0
    got = dsp.coherence_angle(theta_deg, D, lam2)
    expected = 1.0 / (lam2 * np.sqrt(2.0 * c))
    assert np.isclose(got, expected, rtol=0.02)


def test_patch_counts_and_tilt():
    phi_col = np.array([1.5, 1.6, 5.5])
    w2 = np.array([1.0, 1.0, 2.0])
    edges = np.array([0.0, 3.0, 6.0])
    npatch = dsp.patch_counts(phi_col, w2, edges, 0.01, pix_area=1e-3)
    # bin 0: N_eff = (2)^2/2 = 2 -> 2 * 1e-3 / 1e-4 = 20
    # bin 1: N_eff = 1 -> 10
    np.testing.assert_allclose(npatch, [20.0, 10.0])
    H = np.array([2.0, 2.0])
    tilt = dsp.coherence_tilt(H, npatch)
    assert np.isclose(tilt.sum(), H.sum())
    assert tilt[0] > tilt[1]  # more patches -> boosted in the coherent limit


def test_amplitude_bracket_closed_forms():
    b = dsp.amplitude_bracket(
        lam2=99.86, theta_c=0.01, omega_beam=2 * np.pi, phi_med=18.4
    )
    assert np.isclose(b["upper"], 1.0 / np.sqrt(2 * np.pi / 1e-4))
    assert np.isclose(b["lower_slab"], 1.0 / (18.4 * 99.86))
    assert np.isclose(
        b["lower_dispersion"], 1.0 / (2.0 * 9.8**2 * 99.86**2)
    )
    assert b["upper"] > b["lower_slab"] > b["lower_dispersion"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dispersion_geometry.py -v -k "structure or coherence or patch or bracket"`
Expected: FAIL with missing attributes.

- [ ] **Step 3: Implement** — append to `dispersion.py`:

```python
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
    """theta_c (radians) solving 2 lam2^2 D(theta_c) = 1 (spec S4.4)."""
    th = np.radians(np.asarray(theta_deg, dtype=float))
    D = np.maximum.accumulate(np.asarray(D, dtype=float))
    target = 1.0 / (2.0 * float(lam2) ** 2)
    if target <= D[0]:
        return float(th[0])
    if target >= D[-1]:
        return float(th[-1])
    return float(
        np.exp(np.interp(np.log(target), np.log(D), np.log(th)))
    )


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
    """The S4.4 bracket.  Not a prediction -- two ends with reasons."""
    n_patch_tot = max(1.0, float(omega_beam) / float(theta_c) ** 2)
    lam2 = float(lam2)
    return {
        "upper": 1.0 / np.sqrt(n_patch_tot),
        "lower_slab": 1.0 / (abs(float(phi_med)) * lam2),
        "lower_dispersion": 1.0 / (2.0 * float(sigma_eff) ** 2 * lam2**2),
    }
```

- [ ] **Step 4: Run tests, format, commit**

Run: `uv run pytest tests/test_dispersion_geometry.py -v`
Expected: all PASS.

```bash
uv run black src tests && git add -A src tests
git commit -m "Add the coherence tilt and the amplitude bracket machinery"
```

---

### Task 10: `response.pair_weight_maps` — the beam enters (S4.3)

**Files:**
- Modify: `src/lusee_faraday/response.py`
- Test: `tests/test_pair_weights.py` (create)

**Interfaces:**
- Consumes: `conventions.topo_rotation_matrix`,
  `response.FixedChannelKernel` (or any object with
  `.sample(theta_rad, phi_rad) -> (npair, 4, N)`, blocks I,Q,U,V).
- Produces: `pair_weight_maps(kernel, time, loc, nside) -> (npair, npix)
  float` — `|W^{P-}(n)| = 0.5 |K_Q(n) - i K_U(n)|` on the galactic RING
  grid at `nside`, zero below the horizon. This is the beam factor of the
  template weight `w(n)`; the emissivity factor multiplies in the script.

- [ ] **Step 1: Write the failing tests** — `tests/test_pair_weights.py`:

```python
"""Per-pair pixel weights from the frozen-channel kernel (spec S4.3)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lunarsky")
healpy = pytest.importorskip("healpy")

from lusee_faraday import response as rsp
from lusee_faraday.config import moon_location, times
from lusee_faraday.conventions import topo_rotation_matrix


class _SyntheticKernel:
    """One pair; K_Q = cos(theta), K_U = i sin(theta) on the upper sky."""

    def sample(self, theta_rad, phi_rad):
        n = theta_rad.size
        K = np.zeros((1, 4, n), dtype=complex)
        K[0, 1] = np.cos(theta_rad)
        K[0, 2] = 1j * np.sin(theta_rad)
        return K


def test_pair_weight_maps_geometry_and_masking():
    import healpy as hp

    loc = moon_location()
    t = times()[0]
    nside = 32
    w = rsp.pair_weight_maps(_SyntheticKernel(), t, loc, nside)
    assert w.shape == (1, hp.nside2npix(nside))
    assert np.all(w >= 0.0)
    # below-horizon pixels are exactly zero, above-horizon are not
    R = topo_rotation_matrix(t, loc)
    vec = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    z = (R @ vec)[2]
    assert np.all(w[0, z <= 0] == 0.0)
    assert np.all(w[0, z > 1e-3] > 0.0)
    # value check: 0.5 |cos(theta) - i (i sin(theta))| at one pixel
    up = np.argmax(z)  # the pixel nearest zenith
    theta = np.arccos(np.clip(z[up], -1, 1))
    expected = 0.5 * abs(np.cos(theta) - 1j * (1j * np.sin(theta)))
    assert np.isclose(w[0, up], expected, rtol=1e-12)


ARTIFACT = Path(
    os.environ.get(
        "LUSEE_RESPONSE", "data/BGL_v16/lusee_bgl_v16_response_v3.fits"
    )
)


@pytest.mark.slow
@pytest.mark.skipif(not ARTIFACT.exists(), reason="needs BGL_v16 artifact")
def test_pair_weight_maps_from_the_real_kernel():
    lusee = pytest.importorskip("lusee")  # noqa: F841

    resp = rsp.load_response(str(ARTIFACT))
    kernel = rsp.FixedChannelKernel(resp, 30.0)
    w = rsp.pair_weight_maps(kernel, times()[0], moon_location(), 64)
    assert w.shape == (10, 12 * 64**2)
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
    assert (w > 0).any(axis=1).all()  # every pair sees the sky
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pair_weights.py -v`
Expected: first test FAILS with `AttributeError: pair_weight_maps`; the
slow test runs only with the artifact present.

- [ ] **Step 3: Implement** — append to `src/lusee_faraday/response.py`:

```python
def pair_weight_maps(kernel, time, loc, nside):
    """Per-pair |W^{P-}| on the galactic HEALPix grid -> (npair, npix).

    The Faraday-active weight of spec S4.3: the pair-Stokes kernel
    couples K_Q Q + K_U U = W^- (Q + iU) + W^+ (Q - iU) with
    W^- = (K_Q - i K_U) / 2, and (Q + iU) carries e^{+2i phi lam2}.
    Zero below the horizon.  RING ordering, galactic frame.
    """
    import healpy as hp

    R = topo_rotation_matrix(time, loc)
    npix = hp.nside2npix(nside)
    vec = np.array(hp.pix2vec(nside, np.arange(npix)))
    n_resp = R @ vec
    up = n_resp[2] > 0.0
    theta = np.arccos(np.clip(n_resp[2, up], -1.0, 1.0))
    phi = np.mod(np.arctan2(n_resp[1, up], n_resp[0, up]), 2.0 * np.pi)
    K = np.asarray(kernel.sample(theta, phi))  # (npair, 4, Nup), I Q U V
    w = np.zeros((K.shape[0], npix))
    w[:, up] = 0.5 * np.abs(K[:, 1] - 1j * K[:, 2])
    return w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pair_weights.py -v` (and, with the artifact
present, `uv run pytest tests/test_pair_weights.py -v -m slow`).
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
uv run black src tests && git add -A src tests
git commit -m "Sample the frozen kernel into per-pair Faraday weight maps"
```

---

### Task 11: `noise.py` — port, covariances, matched filter (S4.10, S6.12)

**Files:**
- Create: `src/lusee_faraday/noise.py`
- Test: `tests/test_noise.py`

**Interfaces:**
- Consumes: `dispersion.zoom_bin_matrix` (for `W` in tests/scripts),
  `conventions.lambda_squared`.
- Produces:
  - `radiometer_sigma(T_sys, dnu_hz, dt_s)`, `add_noise(stokes, sigma, rng)`
    — ported VERBATIM from `faraday-fisher-forecast` (S5: a free grab).
  - `zoom_noise_covariance(W, sigma_bin) -> (nbin, nbin)` — per-bin noise
    covariance `sigma_bin^2 * rho` with `rho` the normalized overlap
    `W.T W` (the 1.44x ENBW overlap, S4.6).
  - `faraday_signal_covariance(phi, H, lam2_bins) -> (nbin, nbin) complex`
    — `S_ij = sum_b Hhat_b exp(2i phi_b (lam2_i - lam2_j))`, `Hhat`
    normalized to sum 1 so `diag(S) = 1`, `tr(S) = nbin`.
  - `matched_filter_threshold(S, N, n_nights, n_lst, snr=5.0) -> float` —
    Gaussian-signal likelihood-ratio threshold:
    `A^2 = snr / (n_nights * sqrt(n_lst * tr[(N^-1 S)^2]))`.
  - `closed_form_threshold(dnu_coh_hz, tau_s, n_modes, n_nights, snr=5.0)
    -> float` — `sigma_mode * sqrt(snr / (n_nights * sqrt(n_modes)))`,
    `sigma_mode = 1/sqrt(dnu_coh * tau)` (the corrected S4.10 closed form).

- [ ] **Step 1: Write the failing tests** — `tests/test_noise.py`:

```python
"""Radiometer noise and the matched-filter threshold (spec S4.10, S6.12)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from lusee_faraday import noise
from lusee_faraday.conventions import lambda_squared


def test_radiometer_sigma_closed_form():
    assert np.isclose(
        noise.radiometer_sigma(1.0, 563.4, 2305.0), 8.8e-4, rtol=0.02
    )


def test_add_noise_statistics():
    rng = np.random.default_rng(0)
    x = noise.add_noise(np.zeros(200_000), 2.0, rng)
    assert np.isclose(x.std(), 2.0, rtol=0.02)


def test_closed_form_reproduces_the_spec_table():
    """S4.10 corrected row: 30 MHz, zoom on 3 parents, n=1 -> 4.9e-5."""
    got = noise.closed_form_threshold(10838.0, 2305.0, 7086, 1)
    assert np.isclose(got, 4.9e-5, rtol=0.03)
    # 50 MHz parent 200 kHz row: 2.6e-5
    got50 = noise.closed_form_threshold(50176.0, 2305.0, 4086, 1)
    assert np.isclose(got50, 2.6e-5, rtol=0.03)
    # scalings: n^-1/2 and N^-1/4
    assert np.isclose(
        noise.closed_form_threshold(10838.0, 2305.0, 7086, 4) / got,
        0.5,
        rtol=1e-6,
    )


def test_matched_filter_reduces_to_the_closed_form_when_diagonal():
    n, n_lst, sigma = 7, 1024, 2.0e-4
    S = np.eye(n, dtype=complex)
    N = sigma**2 * np.eye(n)
    got = noise.matched_filter_threshold(S, N, n_nights=1, n_lst=n_lst)
    expected = noise.closed_form_threshold(
        10838.0, 2305.0, n * n_lst, 1
    )  # sigma_mode(10838, 2305) = 2.0e-4 = sigma
    assert np.isclose(got, expected, rtol=0.01)


def test_overlap_correlation_degrades_the_threshold():
    """S6.12: the 1.44x zoom overlap must show up, not be ignored."""
    from lusee_faraday import dispersion as dsp

    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    sigma = 8.8e-4
    N_corr = noise.zoom_noise_covariance(W, sigma)
    N_diag = sigma**2 * np.eye(bins.size)
    lam2b = np.asarray(lambda_squared(bins), dtype=float)
    phi = np.arange(2.0, 120.0, 4.0)
    H = np.exp(-phi / 30.0)
    S = noise.faraday_signal_covariance(phi, H, lam2b)
    a_corr = noise.matched_filter_threshold(S, N_corr, 1, 1024)
    a_diag = noise.matched_filter_threshold(S, N_diag, 1, 1024)
    assert a_corr > a_diag
    print(f"\noverlap degradation: {a_corr / a_diag:.3f}x")


def test_matched_filter_monte_carlo():
    """The Fisher SNR matches the empirical score-statistic shift."""
    rng = np.random.default_rng(7)
    nb, M = 48, 3000
    lam2b = np.asarray(
        lambda_squared(np.linspace(29.99, 30.01, nb)), dtype=float
    )
    S = noise.faraday_signal_covariance(
        np.array([30.0, 60.0]), np.array([0.6, 0.4]), lam2b
    )
    sigma2 = 1e-6
    N = sigma2 * np.eye(nb)
    A2 = 4e-7  # amplitude^2, weak-signal regime
    F = np.linalg.solve(N, S)
    snr_pred = A2 * np.sqrt(np.einsum("ij,ji->", F, F).real)

    Ls = np.linalg.cholesky(S + 1e-12 * np.eye(nb))

    def draw(with_signal):
        x = (
            rng.normal(size=(nb, M)) + 1j * rng.normal(size=(nb, M))
        ) / np.sqrt(2)
        x *= np.sqrt(sigma2)
        if with_signal:
            g = (
                rng.normal(size=(nb, M)) + 1j * rng.normal(size=(nb, M))
            ) / np.sqrt(2)
            x = x + np.sqrt(A2) * (Ls @ g)
        NiSNi = np.linalg.solve(N, S) @ np.linalg.inv(N)
        return np.einsum("im,ij,jm->m", x.conj(), NiSNi, x).real

    q0, q1 = draw(False), draw(True)
    snr_emp = (q1.mean() - q0.mean()) / q0.std()
    assert np.isclose(snr_emp, snr_pred, rtol=0.15)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_noise.py -v`
Expected: FAIL with `ModuleNotFoundError: lusee_faraday.noise`.

- [ ] **Step 3: Implement** — `src/lusee_faraday/noise.py`. The first two
  functions are the verbatim port
  (`git show faraday-fisher-forecast:src/lusee_faraday/noise.py`); then the
  additions:

```python
"""Radiometer noise for LuSEE polarized spectra.

sigma = T_sys / sqrt(dnu * dt). T_sys is sky-dominated (~ Stokes I).

radiometer_sigma / add_noise are ported verbatim from the
faraday-fisher-forecast branch (spec S4.10).  The rest is the S4.10
detectability machinery: the zoom-bin noise covariance (the 1.44x ENBW
overlap makes adjacent bins correlated), the Faraday signal covariance,
and the whitened matched-filter threshold whose diagonal limit is the
closed form.
"""

import numpy as np


def radiometer_sigma(T_sys, dnu_hz, dt_s):
    """Radiometer noise std (same units as T_sys)."""
    T_sys = np.asarray(T_sys, dtype=float)
    dnu_hz = np.asarray(dnu_hz, dtype=float)
    return T_sys / np.sqrt(dnu_hz * dt_s)


def add_noise(stokes, sigma, rng):
    """Add Gaussian noise of std `sigma` to a Stokes array.

    `sigma` may be a scalar or broadcastable to `stokes.shape`. `rng`
    is a numpy Generator (e.g. np.random.default_rng(seed)).
    """
    stokes = np.asarray(stokes, dtype=float)
    return stokes + rng.normal(scale=sigma, size=stokes.shape)


def zoom_noise_covariance(W, sigma_bin):
    """Noise covariance of the zoom bins: sigma_bin^2 * overlap.

    ``W`` is the (nfine, nbin) column-normalized weight matrix
    (dispersion.zoom_bin_matrix): white fine-channel noise gives bin
    covariance ~ W.T W, normalized so the diagonal is sigma_bin^2.
    """
    W = np.asarray(W, dtype=float)
    G = W.T @ W
    d = np.sqrt(np.diag(G))
    return float(sigma_bin) ** 2 * (G / np.outer(d, d))


def faraday_signal_covariance(phi, H, lam2_bins):
    """Frequency covariance of a Gaussian Faraday signal of shape H.

    S_ij = sum_b Hhat_b exp(2i phi_b (lam2_i - lam2_j)); Hhat sums to
    one so diag(S) = 1 and an amplitude A means per-bin signal power
    A^2.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    Hhat = np.asarray(H, dtype=float).ravel()
    Hhat = Hhat / Hhat.sum()
    lam2 = np.asarray(lam2_bins, dtype=float).ravel()
    E = np.exp(2j * np.outer(lam2, phi))  # (nbin, nphi)
    return (E * Hhat[None, :]) @ E.conj().T


def matched_filter_threshold(S, N, n_nights, n_lst, snr=5.0):
    """5-sigma amplitude threshold of the whitened matched filter.

    Gaussian-signal likelihood ratio with M = n_lst independent LST
    samples and n_nights coherent nights (noise power / n):
    SNR = n * A^2 * sqrt(n_lst * tr[(N^-1 S)^2]).  With S = I and
    N = sigma^2 I this is exactly the closed form with
    N_modes = n_lst * nbin.
    """
    F = np.linalg.solve(np.asarray(N), np.asarray(S))
    fisher = np.sqrt(
        float(n_lst) * max(np.einsum("ij,ji->", F, F).real, 0.0)
    )
    return float(np.sqrt(snr / (float(n_nights) * fisher)))


def closed_form_threshold(dnu_coh_hz, tau_s, n_modes, n_nights, snr=5.0):
    """A = sigma_mode * sqrt(snr / (n * sqrt(N_modes))), the corrected
    S4.10 closed form: noise per coherence cell, not per zoom bin.
    """
    sigma_mode = 1.0 / np.sqrt(float(dnu_coh_hz) * float(tau_s))
    return float(
        sigma_mode * np.sqrt(snr / (float(n_nights) * np.sqrt(n_modes)))
    )
```

- [ ] **Step 4: Run tests, format, commit**

Run: `uv run pytest tests/test_noise.py -v -s`
Expected: all PASS (the MC test takes ~10 s; the overlap ratio prints).

```bash
uv run black src tests && git add -A src tests
git commit -m "Port noise.py and add the whitened matched-filter threshold"
```

---

### Task 12: `scripts/step5_instrument_envelope.py` — the S4.6 table, reproducible (S5)

**Files:**
- Create: `scripts/step5_instrument_envelope.py`
- Test: `tests/test_dispersion_gates.py` (append the sky-side S6.9 pins)

**Interfaces:**
- Consumes: `dispersion.depth_horizon`, `channelization.parent_weights`,
  `channelization.zoom_weights`, `common.load_sky_maps`,
  `dispersion.weighted_percentiles`.

- [ ] **Step 1: Write the sky-side pin test** — append to
  `tests/test_dispersion_gates.py`:

```python
@needs_rm
def test_gate_envelope_orderings_against_the_sky():
    """S6.9 (sky side): the orderings the paper's claims rest on."""
    rm = np.abs(_rm_map())
    p50, p90, p99, p999 = np.percentile(rm, [50.0, 90.0, 99.0, 99.9])
    mx = rm.max()
    # pin the map percentiles themselves (loose -- conclusions, not digits)
    for got, want in [
        (p50, 18.4), (p90, 91.0), (p99, 278.0), (p999, 648.8), (mx, 2442.1)
    ]:
        assert np.isclose(got, want, rtol=0.02), (got, want)
    off = np.arange(-50000.0, 50001.0, 12.20703125)
    from lusee_faraday.channelization import parent_weights, zoom_weights

    wp, wz = parent_weights(off), zoom_weights(off)[:, 0]
    assert dsp.depth_horizon(off, wz, 50.0) > mx
    z30 = dsp.depth_horizon(off, wz, 30.0)
    assert p999 / 1.5 < z30 < p999 * 1.5
    assert dsp.depth_horizon(off, wz, 10.0) < p50
    for band in (50.0, 30.0, 10.0):
        assert dsp.depth_horizon(off, wp, band) < p90
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_dispersion_gates.py -v -k orderings`
Expected: PASS.

- [ ] **Step 3: Write the script** — `scripts/step5_instrument_envelope.py`:

```python
"""Regenerate the S4.6 envelope table from luseepy's response and the map.

The numbers quoted in the paper's instrument-methods section come from
here, not from the spec (spec S5).  Prints the parent/zoom 50% depth
horizons per band against the |RM| percentiles, and saves them to
generated_data/step5_envelope.npz for step5_plots.py.

Weighted percentiles: if generated_data/step5_template.npz exists (the
Task 13 output), its LST-mean pair weights re-weight the percentiles;
otherwise the unweighted map percentiles are printed with a note.
"""

import common  # noqa: F401  (sets JAX_ENABLE_X64, sys.path)
import numpy as np

from common import GEN_DIR, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday.channelization import parent_weights, zoom_weights

BANDS = (50.0, 30.0, 10.0)


def main():
    off = np.arange(-50000.0, 50001.0, 12.20703125)
    wp = parent_weights(off)
    wz = zoom_weights(off)[:, 0]
    horizons = {
        b: (dsp.depth_horizon(off, wp, b), dsp.depth_horizon(off, wz, b))
        for b in BANDS
    }
    rm = np.abs(load_sky_maps()["RM"])
    qs = [50.0, 90.0, 99.0, 99.9]
    tmpl = GEN_DIR / "step5_template.npz"
    if tmpl.exists():
        w2 = np.load(tmpl)["w2_mean"]  # (npix,) LST/pair-mean |w|^2
        pct = dsp.weighted_percentiles(rm, w2, qs)
        label = "beam-weighted"
        mx = rm[w2 > 0].max()
    else:
        pct = np.percentile(rm, qs)
        label = "UNWEIGHTED (run step5_template.py for the paper numbers)"
        mx = rm.max()

    print(f"|RM| percentiles ({label}):")
    for q, v in zip(qs, pct):
        print(f"  p{q:<5} {v:8.1f} rad/m^2")
    print(f"  max    {mx:8.1f}")
    print("\nband   parent horizon   zoom horizon   (50% depth, rad/m^2)")
    for b in BANDS:
        hp_, hz_ = horizons[b]
        print(f"{b:5.0f}   {hp_:14.1f}   {hz_:12.1f}")
    np.savez(
        GEN_DIR / "step5_envelope.npz",
        bands=np.array(BANDS),
        parent_horizon=np.array([horizons[b][0] for b in BANDS]),
        zoom_horizon=np.array([horizons[b][1] for b in BANDS]),
        percentiles=pct,
        percentile_qs=np.array(qs),
        rm_max=mx,
        weighted=tmpl.exists(),
    )
    print(f"\nwrote {GEN_DIR / 'step5_envelope.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script and check the table**

Run: `cd scripts && uv run python step5_instrument_envelope.py`
Expected: table matching S4.6 (2830/613/24 zoom, 58.7/13.3/2.7 parent,
p50 18.4 ... max 2442.1), npz written.

- [ ] **Step 5: Format and commit**

```bash
uv run black scripts/step5_instrument_envelope.py tests
git add -A scripts tests
git commit -m "Regenerate the depth-horizon table from committed code"
```

---

### Task 13: `scripts/step5_template.py` — templates, LST tail gate, two arms (S4.2, S4.2.2, S4.3, S4.9, S6.5, S6.11, S6.14)

**Files:**
- Create: `scripts/step5_template.py`

**Interfaces:**
- Consumes: `response.load_response`, `response.FixedChannelKernel`,
  `response.pair_weight_maps`, `response.two_port_jones_from_fits`,
  `response.pair_stokes_from_jones`, `response.sample_periodic_maps`,
  `dispersion.*`, `common.load_sky_maps`, `common.RESPONSE_PATH`,
  `config.times`, `config.moon_location`, `config.BETA_QU`,
  `config.FREQ_REF_QU`.
- Produces: `generated_data/step5_template.npz` with keys:
  `phi` (coarse |phi| grid, 1 rad/m^2), `H` shaped
  `(nband, nk, nphi)` (LST- and pair-averaged normalised templates),
  `H_coh` (same shape, coherence-tilted), `ks`, `bands`, `knee`
  `(nband, nk)`, `knee_taper` (plane-tapered, S6.5), `tail_frac_lst`
  `(nband, nlst)` (S6.14 gate curve), `lst_hours`, `w2_mean` `(npix,)`,
  `weighted_percentiles` `(4,)`, `sigma_eff`, `theta_c` `(nband,)`,
  `bracket` `(nband, 3)` [upper, lower_slab, lower_dispersion];
  and, with `--arm two-port`, `generated_data/step5_template_two_port.npz`
  with the same keys (S6.11 compares the two).

- [ ] **Step 1: Write the script** — `scripts/step5_template.py`:

```python
"""Build the diffuse Faraday delay templates (spec S4.2--S4.3, S4.9).

Per band and per geometry k, the |w|^2-weighted depth distribution of
the RM map, with w = pair beam x polarised emissivity, LST-resolved.
Outputs the normalised template family, the coherence-tilted variant
(S4.4.1), the half-power knees (plain and plane-tapered, S4.2.1), the
LST-resolved tail fraction that decides the S4.2.2 gate, and the
amplitude bracket inputs.

Heavy: run in the background under ulimit -v 16000000 with a log in
generated_data/.  ~20-40 min at --lst 128 on the as-built kernel.

Usage:
  uv run python step5_template.py [--arm four-port|two-port]
      [--lst 128] [--bands 30 50 10] [--sigma-eff 9.8]
"""

import argparse

import common  # noqa: F401
import numpy as np

import healpy as hp
from common import DATA_DIR, GEN_DIR, RESPONSE_PATH, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday import response as rsp
from lusee_faraday.config import (
    BETA_QU,
    FREQ_REF_QU,
    MAP_NSIDE,
    moon_location,
    times,
)

KS = (np.inf, 0.0, -1.0)
K_LABELS = ("inf", "0", "-1")
COARSE_DPHI = 1.0  # rad/m^2, the display/npz grid


class _TwoPortKernel:
    """Duck-typed kernel for the symmetric pseudo-dipole arm (S4.9)."""

    def __init__(self, path, freq_mhz):
        h_theta, h_phi = rsp.two_port_jones_from_fits(path, freq_mhz)
        maps = rsp.pair_stokes_from_jones(
            h_theta[:, None], h_phi[:, None], pairs=rsp.TWO_PORT_PAIRS
        )[:, 0]
        self.K = maps  # (3, 4, ntheta, nphi)
        self.theta_deg = np.arange(self.K.shape[-2], dtype=float)
        self.phi_deg = np.arange(self.K.shape[-1], dtype=float)

    def sample(self, theta_rad, phi_rad):
        return rsp.sample_periodic_maps(
            self.K, self.theta_deg, self.phi_deg, theta_rad, phi_rad
        )


def build_kernel(arm, freq_mhz):
    if arm == "two-port":
        return _TwoPortKernel(
            DATA_DIR / "hfss_lbl_3m_75deg.2port.fits", freq_mhz
        )
    resp = rsp.load_response(RESPONSE_PATH)
    return rsp.FixedChannelKernel(resp, freq_mhz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="four-port",
                    choices=["four-port", "two-port"])
    ap.add_argument("--lst", type=int, default=128)
    ap.add_argument("--bands", type=float, nargs="+",
                    default=[30.0, 50.0, 10.0])
    ap.add_argument("--sigma-eff", type=float, default=9.8)
    args = ap.parse_args()

    maps = load_sky_maps()
    rm = np.asarray(maps["RM"], dtype=float)
    loc = moon_location()
    t_all = times()
    lst_idx = np.linspace(0, len(t_all) - 1, args.lst, dtype=int)
    pix_area = hp.nside2pixarea(MAP_NSIDE)
    _, b_gal = hp.pix2ang(
        MAP_NSIDE, np.arange(rm.size), lonlat=True
    )
    taper = np.sin(np.radians(b_gal)) ** 2  # |b| taper, S4.2.1

    theta_grid = np.geomspace(0.2, 30.0, 24)
    D = dsp.structure_function(rm, theta_grid)

    coarse = np.arange(0.0, 2500.0 + COARSE_DPHI, COARSE_DPHI)
    ccent = 0.5 * (coarse[1:] + coarse[:-1])
    nb, nk = len(args.bands), len(KS)
    H_out = np.zeros((nb, nk, ccent.size))
    Hc_out = np.zeros_like(H_out)
    knee = np.zeros((nb, nk))
    knee_taper = np.zeros((nb, nk))
    tail = np.zeros((nb, args.lst))
    theta_cs = np.zeros(nb)
    bracket = np.zeros((nb, 3))
    w2_accum = np.zeros(rm.size)

    for ib, band in enumerate(args.bands):
        from lusee_faraday.conventions import lambda_squared

        lam2 = float(np.asarray(lambda_squared(band)))
        kernel = build_kernel(args.arm, band)
        p_emis = np.hypot(maps["Q23"], maps["U23"]) * (
            band / FREQ_REF_QU
        ) ** BETA_QU
        edges = dsp.phi_edges(band)
        cent = dsp.phi_centers(edges)
        theta_cs[ib] = dsp.coherence_angle(theta_grid, D, lam2)
        Hsum = np.zeros((nk, cent.size))
        Hsum_taper = np.zeros_like(Hsum)
        for il, ti in enumerate(lst_idx):
            wb = rsp.pair_weight_maps(kernel, t_all[ti], loc, MAP_NSIDE)
            w2 = ((wb * p_emis[None, :]) ** 2).sum(axis=0)  # pair-summed
            w2_accum += w2
            for ik, k in enumerate(KS):
                H = dsp.depth_distribution(rm, w2, edges, k=k)
                Hsum[ik] += H
                Hsum_taper[ik] += dsp.depth_distribution(
                    rm, w2 * taper, edges, k=k
                )
            # tail gate (S6.14): fraction of k=inf template power above
            # the running beam-weighted p99
            p99 = dsp.weighted_percentiles(np.abs(rm), w2, [99.0])[0]
            Hh = dsp.depth_distribution(rm, w2, edges, k=np.inf)
            tail[ib, il] = Hh[np.abs(cent) > p99].sum() / Hh.sum()
            print(f"band {band} LST {il + 1}/{args.lst}", flush=True)

        npatch = dsp.patch_counts(
            rm, w2_accum, edges, theta_cs[ib], pix_area
        )
        for ik in range(nk):
            pa, Hf = dsp.fold_template(cent, Hsum[ik])
            _, Hft = dsp.fold_template(cent, Hsum_taper[ik])
            _, Hfc = dsp.fold_template(
                cent, dsp.coherence_tilt(Hsum[ik], npatch)
            )
            knee[ib, ik] = dsp.half_power_knee(pa, Hf)
            knee_taper[ib, ik] = dsp.half_power_knee(pa, Hft)
            for target, src in ((H_out, Hf), (Hc_out, Hfc)):
                rb, _ = np.histogram(pa, bins=coarse, weights=src)
                target[ib, ik] = rb / max(rb.sum(), 1e-300)

        wpct = dsp.weighted_percentiles(
            np.abs(rm), w2_accum, [50.0, 90.0, 99.0, 99.9]
        )
        omega_beam = w2_accum.sum() ** 2 / (w2_accum**2).sum() * pix_area
        br = dsp.amplitude_bracket(
            lam2, theta_cs[ib], omega_beam, wpct[0], args.sigma_eff
        )
        bracket[ib] = [br["upper"], br["lower_slab"],
                       br["lower_dispersion"]]

    suffix = "" if args.arm == "four-port" else "_two_port"
    out = GEN_DIR / f"step5_template{suffix}.npz"
    np.savez(
        out,
        phi=ccent, H=H_out, H_coh=Hc_out, ks=np.array([100.0, 0.0, -1.0]),
        bands=np.array(args.bands), knee=knee, knee_taper=knee_taper,
        tail_frac_lst=tail,
        lst_hours=lst_idx * (27.321661 * 24.0 / 1024.0),
        w2_mean=w2_accum / w2_accum.sum(),
        weighted_percentiles=wpct, sigma_eff=args.sigma_eff,
        theta_c=theta_cs, bracket=bracket,
    )
    print(f"knees:\n{knee}\nplane-tapered:\n{knee_taper}")
    print(f"tail fraction: min {tail.min():.2e} max {tail.max():.2e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run at reduced cost**

Run (background, logged):

```bash
cd scripts && mkdir -p ../generated_data
ulimit -v 16000000 && nohup uv run python step5_template.py --lst 8 \
  --bands 30 > /home/christian/Documents/research/lusee/lusee_faraday/generated_data/step5_template_smoke.log 2>&1 &
```

Expected: completes; knees printed; the fiducial (k=0) 30 MHz knee lands
between the beam-weighted p50 and p99 (S6.4); `tail_frac_lst` varies with
LST and peaks near the GC-transit LSTs.

- [ ] **Step 3: Full runs, both arms**

```bash
ulimit -v 16000000 && nohup uv run python step5_template.py --lst 128 \
  > .../generated_data/step5_template.log 2>&1 &
# after it finishes:
ulimit -v 16000000 && nohup uv run python step5_template.py --lst 128 \
  --arm two-port > .../generated_data/step5_template_two_port.log 2>&1 &
```

(absolute log paths; run sequentially, each is memory-heavy). Record in the
task report: the knee per (band, k), the plane-taper shift (S6.5 reported
number), the two-arm knee shift (S6.11 reported number), and the tail-gate
verdict — the max LST tail fraction against the Task 14 threshold (S6.14
decides whether the `max phi_col` section opens).

- [ ] **Step 4: Re-run the envelope script for weighted percentiles**

Run: `cd scripts && uv run python step5_instrument_envelope.py`
Expected: now prints beam-weighted percentiles (Task 12 wired this).

- [ ] **Step 5: Format and commit**

```bash
uv run black scripts/step5_template.py
git add scripts/step5_template.py
git commit -m "Assemble the LST-resolved template family and the tail gate"
```

---

### Task 14: `scripts/step5_sensitivity.py` — matched filter, schedule, T_sys/T_sky (S4.10)

**Files:**
- Create: `scripts/step5_sensitivity.py`

**Interfaces:**
- Consumes: `noise.*`, `dispersion.zoom_bin_matrix`,
  `instrument.covariance`, `instrument.blackbody_normalization`,
  `response.load_response`, `common.RESPONSE_PATH`,
  `generated_data/step5_template.npz` (falls back to an analytic slab
  template if absent, so the script is standalone-runnable).
- Produces: `generated_data/step5_sensitivity.npz` with keys `lunations`,
  `A_mf` `(nband, nlun)`, `A_closed` `(nband, nlun)`, `bands`, `bracket`
  `(nband, 3)`, `tsys_over_tsky` `(nband,)` (nan when the artifact or
  receiver model is unavailable).

- [ ] **Step 1: Write the script** — `scripts/step5_sensitivity.py`:

```python
"""The detectability threshold (spec S4.10): matched filter + schedule.

The whitened matched filter on the zoom-bin covariance is the
deliverable; the corrected closed form is printed as the sanity check.
The schedule model carries the night-fraction and sidereal-drift
corrections (S4.10): one lunation covers ~55% of LST bins, and a given
LST bin is dark in ~0.54 of lunations.

T_sys/T_sky: computed from the luseepy loading model (moon + loss
terms at 250 K) against a 1 K blackbody scaled by the mean sky
temperature.  Amplifier noise is NOT in this chain -- pass --t-amp
with the receiver noise temperature when the collaboration provides
it; until then the printed ratio is a lower bound.
"""

import argparse

import common  # noqa: F401
import numpy as np

from common import GEN_DIR, RESPONSE_PATH, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday import noise
from lusee_faraday.config import BETA_I, FREQ_REF_I, SIDEREAL_DAY_S
from lusee_faraday.conventions import lambda_squared

TAU_S = SIDEREAL_DAY_S / 1024
COH_BW_HZ = {50.0: 50176.0, 30.0: 10838.0, 10.0: 401.0}
NIGHT_FRACTION = 0.55
DRIFT_FACTOR = 0.54


def schedule(lunations):
    """(coherent nights per LST bin, LST bins covered)."""
    n_lst = int(round(1024 * min(1.0, NIGHT_FRACTION * lunations)))
    n_coh = max(1.0, DRIFT_FACTOR * lunations)
    return n_coh, max(n_lst, 1)


def template_for(band):
    f = GEN_DIR / "step5_template.npz"
    if f.exists():
        d = np.load(f)
        ib = int(np.argmin(np.abs(d["bands"] - band)))
        return d["phi"], d["H"][ib, 1], d["bracket"][ib]  # k = 0 fiducial
    # standalone fallback: uniform slab to the map median depth
    phi = np.arange(0.0, 2500.0, 1.0)
    H = np.where(phi < 18.4, 1.0, 0.0)
    return phi, H / H.sum(), np.array([1e-4, 5.4e-4, 5.2e-7])


def tsys_over_tsky(band, t_amp_k):
    try:
        import lusee
        from lusee_faraday import instrument, response as rsp

        resp = rsp.load_response(RESPONSE_PATH)
        receiver = lusee.LuSEE_Receiver()
        bb = instrument.blackbody_normalization(
            resp, receiver, np.array([band]), impedance_freq_mhz=band
        )
        load = instrument.covariance(
            np.zeros((1, 1, 10)), resp, receiver, np.array([band]),
            T_moon=250.0, T_ant=250.0, impedance_freq_mhz=band,
        )
        t_sky = float(
            np.mean(load_sky_maps()["I408"])
            * (band / FREQ_REF_I) ** BETA_I
        )
        r = np.nanmean(
            np.abs(np.diagonal(load[0, 0]))
            / (np.abs(np.diagonal(bb[0])) * t_sky)
        )
        return float(r + t_amp_k / t_sky)
    except Exception as e:  # artifact or receiver model unavailable
        print(f"T_sys/T_sky at {band} MHz unavailable: {e}")
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=float, nargs="+", default=[30.0, 50.0])
    ap.add_argument("--lunations", type=int, default=24)
    ap.add_argument("--t-amp", type=float, default=0.0)
    args = ap.parse_args()

    lun = np.arange(1, args.lunations + 1)
    A_mf = np.zeros((len(args.bands), lun.size))
    A_cf = np.zeros_like(A_mf)
    bracket = np.zeros((len(args.bands), 3))
    ratios = np.zeros(len(args.bands))

    for ib, band in enumerate(args.bands):
        fine, bins, W = dsp.zoom_bin_matrix(band)
        sigma_bin = noise.radiometer_sigma(1.0, 563.4, TAU_S)
        N = noise.zoom_noise_covariance(W, sigma_bin)
        phi, H, bracket[ib] = template_for(band)
        lam2b = np.asarray(lambda_squared(bins), dtype=float)
        keep = H > H.max() * 1e-6
        S = noise.faraday_signal_covariance(phi[keep], H[keep], lam2b)
        n_modes_cf = 75000.0 / COH_BW_HZ[band]
        for j, L in enumerate(lun):
            n_coh, n_lst = schedule(int(L))
            A_mf[ib, j] = noise.matched_filter_threshold(
                S, N, n_coh, n_lst
            )
            A_cf[ib, j] = noise.closed_form_threshold(
                COH_BW_HZ[band], TAU_S, n_modes_cf * n_lst, n_coh
            )
        ratios[ib] = tsys_over_tsky(band, args.t_amp)
        print(
            f"{band:.0f} MHz: A(1 lun) mf {A_mf[ib, 0]:.2e} "
            f"closed {A_cf[ib, 0]:.2e}; T_sys/T_sky >= {ratios[ib]:.2f}"
        )

    np.savez(
        GEN_DIR / "step5_sensitivity.npz",
        lunations=lun, A_mf=A_mf, A_closed=A_cf,
        bands=np.array(args.bands), bracket=bracket,
        tsys_over_tsky=ratios,
    )
    print(f"wrote {GEN_DIR / 'step5_sensitivity.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd scripts && uv run python step5_sensitivity.py`
Expected: thresholds at 1 lunation within a factor ~2 of the closed form
(the matched filter knows the overlap correlation and the template shape;
exact equality is not expected), closed form ~5e-5 (30) / ~2.6e-5 (50) at
`n=1, n_lst≈563`; `T_sys/T_sky` printed per band (nan is acceptable
without the artifact, but with `data/BGL_v16` present it must print a
number — if it is materially above 1 at 50 MHz, record it: S4.10 says the
50 MHz rows scale up by it).

- [ ] **Step 3: Format and commit**

```bash
uv run black scripts/step5_sensitivity.py
git add scripts/step5_sensitivity.py
git commit -m "Compute the matched-filter threshold curve with the real schedule"
```

---

### Task 15: `scripts/step5_plots.py` — the paper figures (S5)

**Files:**
- Create: `scripts/step5_plots.py`

**Interfaces:**
- Consumes: `generated_data/step5_template.npz`,
  `step5_template_two_port.npz` (optional), `step5_envelope.npz`,
  `step5_sensitivity.npz`; `common.FIG_DIR`.
- Produces (PDF, in `report/figures/`): `step5_template_family.pdf`,
  `step5_knee_tail_lst.pdf`, `step5_envelope.pdf`,
  `step5_sensitivity.pdf`, `step5_chirp_coherence.pdf`,
  `step5_two_arm.pdf` (only if the two-port npz exists).

- [ ] **Step 1: Write the script** — `scripts/step5_plots.py`:

```python
"""Figures for the delay-template paper (spec S5).

Reads the step5_*.npz products; every figure is regenerable from
committed code plus data/.
"""

import common  # noqa: F401
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIG_DIR, GEN_DIR  # noqa: E402
from lusee_faraday import dispersion as dsp  # noqa: E402
from lusee_faraday.config import fine_freqs  # noqa: E402
from lusee_faraday.conventions import lambda_squared  # noqa: E402

K_LABELS = {0: "$k\\to\\infty$ (all far)", 1: "$k=0$ (slab, fiducial)",
            2: "$k\\to-1$ (all local)"}


def fig_template_family(d):
    bands = d["bands"]
    fig, axes = plt.subplots(
        1, len(bands), figsize=(4 * len(bands), 3.2), sharey=True
    )
    for ib, (ax, band) in enumerate(zip(np.atleast_1d(axes), bands)):
        for ik in range(d["H"].shape[1]):
            ax.plot(d["phi"], d["H"][ib, ik], label=K_LABELS[ik])
            ax.plot(d["phi"], d["H_coh"][ib, ik], ls=":", alpha=0.7)
        for q in d["weighted_percentiles"]:
            ax.axvline(q, color="0.8", lw=0.6, zorder=0)
        ax.set(xscale="log", yscale="log", xlim=(0.5, 2600),
               title=f"{band:.0f} MHz", xlabel=r"$\phi$ [rad m$^{-2}$]")
        ax.set_ylim(1e-8, None)
    np.atleast_1d(axes)[0].set_ylabel("normalised template")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_template_family.pdf")


def fig_knee_tail(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for ib, band in enumerate(d["bands"]):
        a1.plot(d["lst_hours"], d["tail_frac_lst"][ib],
                label=f"{band:.0f} MHz")
    a1.set(yscale="log", xlabel="LST [h]",
           ylabel="template power fraction above beam-weighted p99",
           title="the S4.2.2 tail gate")
    a1.legend(fontsize=7)
    x = np.arange(d["knee"].shape[1])
    for ib, band in enumerate(d["bands"]):
        a2.plot(x, d["knee"][ib], "o-", label=f"{band:.0f} MHz")
        a2.plot(x, d["knee_taper"][ib], "s--", alpha=0.6)
    a2.set(xticks=x, xticklabels=["inf", "0", "-1"], xlabel="k",
           ylabel=r"half-power knee [rad m$^{-2}$]",
           title="knee vs geometry (dashed: plane-tapered)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_knee_tail_lst.pdf")


def fig_envelope(env, d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(env["bands"]))
    ax.bar(x - 0.15, env["parent_horizon"], 0.3, label="parent horizon")
    ax.bar(x + 0.15, env["zoom_horizon"], 0.3, label="zoom horizon")
    for q, v in zip(env["percentile_qs"], env["percentiles"]):
        ax.axhline(v, color="0.7", lw=0.7)
        ax.text(2.55, v, f"p{q:g}", fontsize=6, va="bottom")
    ax.axhline(env["rm_max"], color="k", lw=0.9, ls="--", label="|RM| max")
    ax.set(yscale="log", xticks=x,
           xticklabels=[f"{b:.0f} MHz" for b in env["bands"]],
           ylabel=r"50% Faraday depth [rad m$^{-2}$]",
           title="the instrument's depth horizon vs the sky")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_envelope.pdf")


def fig_sensitivity(s):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for ib, band in enumerate(s["bands"]):
        (ln,) = ax.plot(s["lunations"], s["A_mf"][ib],
                        label=f"{band:.0f} MHz matched filter")
        ax.plot(s["lunations"], s["A_closed"][ib], ls=":",
                color=ln.get_color(), label=f"{band:.0f} MHz closed form")
        up, sl, dis = s["bracket"][ib]
        ax.axhspan(dis, up, color=ln.get_color(), alpha=0.06)
        ax.axhline(up, color=ln.get_color(), lw=0.6, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="lunations",
           ylabel="5$\\sigma$ fractional amplitude threshold",
           title="threshold vs the S4.4 amplitude bracket")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_sensitivity.pdf")


def fig_chirp_coherence():
    freqs = fine_freqs(30.0)[::4]
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    phi0 = 600.0
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(500.0, 700.0, 0.1)
    p_nufft = dsp.delay_power(spec, freqs, phi_out)
    n = freqs.size
    P = np.abs(np.fft.fftshift(np.fft.fft(spec)) / n) ** 2
    bw = (freqs[1] - freqs[0]) * 1e6 * n
    k = np.arange(n) - n // 2
    lam2_0 = float(np.asarray(lambda_squared(30.0)))
    phi_fft = np.pi * k * 30e6 / (2.0 * bw * lam2_0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(phi_out, p_nufft / p_nufft.max(), label="type-3 NUFFT")
    sel = (phi_fft > 500) & (phi_fft < 700)
    ax.plot(phi_fft[sel], P[sel] / P[sel].max(),
            label="uniform-$\\nu$ FFT (the chirp)")
    ax.set(xlabel=r"$\phi$ [rad m$^{-2}$]", ylabel="normalised power",
           title=r"a single depth at $\phi_0 = 600$, 30 MHz")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_chirp_coherence.pdf")


def fig_two_arm(d, d2):
    fig, axes = plt.subplots(
        1, len(d["bands"]), figsize=(4 * len(d["bands"]), 3.2), sharey=True
    )
    for ib, (ax, band) in enumerate(zip(np.atleast_1d(axes), d["bands"])):
        ax.plot(d["phi"], d["H"][ib, 1], label="as-built four-port")
        ax.plot(d2["phi"], d2["H"][ib, 1], label="symmetric two-port")
        ax.set(xscale="log", yscale="log", title=f"{band:.0f} MHz",
               xlabel=r"$\phi$ [rad m$^{-2}$]", ylim=(1e-8, None))
    np.atleast_1d(axes)[0].set_ylabel("normalised template (k = 0)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step5_two_arm.pdf")


def main():
    d = np.load(GEN_DIR / "step5_template.npz")
    fig_template_family(d)
    fig_knee_tail(d)
    fig_envelope(np.load(GEN_DIR / "step5_envelope.npz"), d)
    fig_sensitivity(np.load(GEN_DIR / "step5_sensitivity.npz"))
    fig_chirp_coherence()
    two = GEN_DIR / "step5_template_two_port.npz"
    if two.exists():
        fig_two_arm(d, np.load(two))
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and eyeball every figure**

Run: `cd scripts && uv run python step5_plots.py`
Expected: 5–6 PDFs in `report/figures/`. Open each; check: template
family shows the roll-off/extent ordering across k; the envelope figure
reproduces the S4.6 story; the sensitivity curve sits against the shaded
bracket; the tail-gate curve peaks at GC transit.

- [ ] **Step 3: Format and commit**

```bash
uv run black scripts/step5_plots.py
git add scripts/step5_plots.py
git commit -m "Draw the template, envelope, tail-gate and sensitivity figures"
```

---

### Task 16: Docs, PROGRESS, full-suite verification (S5 report surgery)

**Files:**
- Modify: `docs/measurement-model.md` (append after §8, before "See also")
- Modify: `PROGRESS.md`

- [ ] **Step 1: Append §9–§11 to `docs/measurement-model.md`** (insert
  before the `## See also` line):

```markdown
## 9. Bin in depth first, transform second

The old diffuse calculation summed `e^{2i phi(n) lambda^2}` pixel by
pixel; at nside 512 the phase moves ~1730 rad between neighbours and the
sum is a random walk whose amplitude falls as `1/sqrt(N_pix)` — the
2026-08-18 audit's finding. Binning the beam- and emissivity-weighted
sky by Faraday depth first gives `F(phi)`, a stable histogram whose
*shape* is invariant under refinement (`tests/test_dispersion_gates.py`
pins this at nside 256–2048 and under a null rotation); only the
normalisation carries the shot noise. The observable is then the
RM-synthesis pair `P(lambda^2) = Int F(phi) e^{2i phi lambda^2} dphi`
and, in the incoherent limit, delay power = the `|w|^2`-weighted depth
distribution. `dispersion.py` owns both directions; both use type-3
NUFFTs on the true `lambda^2` nodes, and raw pixel depths can be fed as
nonuniform points directly (the converged-regime gate does).

## 10. The three bands are three different problems

Per the spec's S4.5 table (regenerated by
`scripts/step5_instrument_envelope.py`): 30 MHz has the Faraday
resolution (RMSF 2.60 rad/m^2) and hosts the template's half-power knee;
50 MHz has coarse resolution (12.0) but the only zoom horizon that
reaches the map maximum; 10 MHz is out — its zoom horizon (24 rad/m^2)
is below the median sky depth. The chirp of `lambda^2(nu)` against a
uniform frequency grid is an analysis artifact removed by the NUFFT
(`test_nufft_beats_fft_on_a_single_depth`), not a physical wall.

## 11. Channel width is a Faraday-depth horizon

A spectrometer bin is a window in `nu`, so in depth it is a
multiplicative envelope: the instrument imposes its own roll-off on top
of the sky's. The 50% depths from luseepy's real bin responses are
2830 / 613 / 24 rad/m^2 (zoom) and 58.7 / 13.3 / 2.7 (parent) at
50 / 30 / 10 MHz, against sky percentiles p50 18.4 / p90 91 / p99.9
648.8 / max 2442. No parent bin reaches the knee; at 30 MHz the zoom
envelope lands on the sky's p99.9 and only deconvolution separates the
two — `dispersion.rmsf` (built from the real responses, folding
included) is the kernel to deconvolve against. Zoom bins overlap
(ENBW 563 Hz on 390.6 Hz spacing): adjacent bins are correlated and the
matched filter in `noise.py` carries that covariance.
```

- [ ] **Step 2: Update `PROGRESS.md`** — append a step-5 section after the
  existing step entries:

```markdown
## Step 5: the Faraday delay template (branch faraday-delay-template)
- [x] `dispersion.py` (depth distributions, NUFFT transforms, real-response
  RMSF, depth horizon, geometry knob, coherence bracket) + `noise.py`
  (ported + matched filter) + `response.pair_weight_maps`.
- [x] Acceptance gates passed: normalised template invariant under nside
  256–2048 refinement and under a null rotation (the audit's two findings,
  rebutted); converged-regime control reproduced through the NUFFT path.
- [x] `step5_instrument_envelope.py` / `step5_template.py` /
  `step5_sensitivity.py` / `step5_plots.py`; figures in report/figures/.
- [ ] Tail gate verdict (S4.2.2): <record the max LST tail fraction vs the
  threshold here after the full --lst 128 run>.
- Figure provenance: all step-5 figures regenerate from committed scripts
  on this branch; the refuted Step 2/4 figures live only at the
  audit-2026-08-18 tag. The mixed-provenance list is empty here.
```

Fill the tail-gate line with the actual Task 13/14 numbers before
committing.

- [ ] **Step 3: Full verification**

Run: `uv run pytest` (full suite, with data present also
`uv run pytest -m slow`), then `uv run black src/ tests/ scripts/ --check`
and `uv run flake8 src/`.
Expected: everything green; no formatting drift.

- [ ] **Step 4: Commit**

```bash
git add docs/measurement-model.md PROGRESS.md
git commit -m "Document the delay-template layer and record step 5"
```

---

## Self-Review (performed while writing)

- **Spec coverage:** S3 (Task 1–2), S4.1 (1–4), S4.2/S4.2.1/S4.2.2
  (2, 13), S4.3 (10, 13), S4.4/S4.4.1 (9), S4.5 (3, 16 §10), S4.6
  (4, 6, 12, 16 §11), S4.7 (no task — it is a prohibition: nothing here
  imports the polarimeter into the diffuse path), S4.8 (5), S4.9 (13),
  S4.10 (11, 14), S5 deliverables (12–16), S6.1/6.2 (7), 6.3 (1), 6.4
  (7 + 13), 6.5 (13), 6.6 (8), 6.7 (4), 6.8 (3), 6.9 (4 + 12), 6.10 (5),
  6.11 (13), 6.12 (11), 6.13 (6), 6.14 (13), 6.15 (9 + 13), S7 ordering
  respected with the stop point at Task 7.
- **Known open risk carried from the spec:** if Task 6's aliasing peak is
  not at the wrap-predicted position, or Task 7's gates fail, stop and
  report — those are spec-mandated findings, not bugs to fix silently.
- **Type consistency:** `depth_distribution(phi_col, w2, edges, k)`,
  `transform(phi, F, lam2_targets)`, `delay_power(spectrum, freqs_mhz,
  phi_out, window)`, `zoom_bin_matrix(center_mhz) -> (fine, bins, W)`,
  `matched_filter_threshold(S, N, n_nights, n_lst, snr)` are used with
  these exact signatures in every later task and script.
