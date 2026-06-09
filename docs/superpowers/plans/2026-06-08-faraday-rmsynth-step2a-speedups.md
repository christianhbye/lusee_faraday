# Faraday RM-Synthesis — Step 2a (lossless sim speedups) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the full-band Faraday sim feasible (~15–20 min on 8 cores instead of ~9 hr) with three accuracy-preserving speedups: horizon-masking before the trig, response truncation, and time-step parallelism — plus the per-mode decimation `FrequencyPlan` needs for a mixed wide/zoom grid.

**Architecture:** All changes are library-level and behaviour-preserving. `compute_vis_fast` restricts the per-frequency `cos`/`sin` to above-horizon pixels (below-horizon pixels are already zeroed in the beam-weighted sums, so this is bit-for-bit identical). `SpectrometerResponse.truncate` drops negligible response wings. `FrequencyPlan` gains per-mode decimation and an optional truncation. `compute_vis_fast_parallel` splits the time axis across processes. Each change is verified against a serial/untruncated reference.

**Tech Stack:** Python, NumPy, concurrent.futures, pytest, uv.

**Source spec:** `docs/superpowers/specs/2026-06-08-faraday-rmsynth-design.md`

---

## Verified facts this plan relies on (from prototyping)

- Horizon mask: 98560/196608 pixels above horizon (50.1%). Below-horizon pixels contribute exactly 0 to `cos_p @ A[k]` (since `A`,`B` are built from mask-zeroed maps), so restricting the trig to above-horizon pixels is identity → ~2× speedup.
- Truncation to the symmetric window holding `support` of every response column: ±16.6 kHz (0.99), ±17.9 kHz (0.999), ±22.4 kHz (0.9999). At 0.999 the truncated+renormalized `apply_wide`/`apply_narrow` reproduce the full result to <1e-4.
- `FrequencyPlan` dedup is exact on the integer-Hz lattice regardless of per-spec decimation (all offsets are multiples of 10 Hz).

## Scope

Library only; no sim run, no driver rewrite (those are Step 2b). Backward compatible: existing Step-1 `FrequencyPlan(response, specs)` and `compute_vis_fast(...)` calls keep working unchanged (new behaviour is opt-in via new args).

## File Structure

- Modify: `src/lusee_faraday/fast_sim.py` — mask the trig in `compute_vis_fast`; add `compute_vis_fast_parallel` + module-level worker.
- Modify: `src/lusee_faraday/spectrometer.py` — add `truncate`.
- Modify: `src/lusee_faraday/freqplan.py` — per-mode decimation + `support`.
- Test: `tests/test_fast_sim.py` (create), extend `tests/test_freqplan.py`, extend a spectrometer test (create `tests/test_spectrometer.py` if absent).

Conventions: readability first, sparse comments, Black line-length 79. If `uv run` touches pyproject.toml/uv.lock, `git checkout` them before committing.

---

## Task 1: Horizon-mask the trig in `compute_vis_fast`

**Files:** Modify `src/lusee_faraday/fast_sim.py`, Create `tests/test_fast_sim.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_fast_sim.py`)

These build a tiny synthetic case (small npix, stub beam) and check (a) the result matches an independent brute-force reference over all pixels, and (b) below-horizon pixel values never affect the output (the masking invariant).

```python
import types
import numpy as np
from lusee_faraday.fast_sim import compute_vis_fast


def _stub(npix, ntimes, seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"w{s}_{p}" for s in "IQU" for p in ("x", "y", "xy")]
    weights = {k: rng.normal(size=npix) for k in keys}
    beam = types.SimpleNamespace(weights=weights)
    I = rng.uniform(50, 100, (ntimes, npix))
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = rng.normal(scale=10, size=(ntimes, npix))
    mask = rng.random(npix) > 0.4
    freqs = np.array([10.0, 30.0, 50.0])
    return beam, I, Q, U, rm, mask, freqs


def _ref_vis(beam, I, Q, U, rm, mask, freqs,
             freq_ref_I=50, beta_I=-2.55,
             freq_ref_QU=23e3, beta_QU=-2.8, cmb=2.725):
    # straightforward all-pixel reference of the same math
    w = beam.weights
    m = mask.astype(float)
    norm = np.sum(w["wI_x"] * m) + np.sum(w["wI_y"] * m)
    sI = (freqs / freq_ref_I) ** beta_I
    sQU = (freqs / freq_ref_QU) ** beta_QU
    l2 = (3e8 / (freqs * 1e6)) ** 2
    nt = I.shape[0]
    vis = np.zeros((nt, 3, len(freqs)))
    for i in range(nt):
        Im = (I[i] - cmb) * m
        Qm = Q[i] * m
        Um = U[i] * m
        for k, pol in enumerate(("x", "y", "xy")):
            sII = np.sum(w[f"wI_{pol}"] * Im)
            sIc = np.sum(w[f"wI_{pol}"] * cmb * m)
            A = w[f"wQ_{pol}"] * Qm + w[f"wU_{pol}"] * Um
            B = w[f"wU_{pol}"] * Qm - w[f"wQ_{pol}"] * Um
            ph = 2 * np.outer(l2, rm[i])
            pol_c = np.cos(ph) @ A + np.sin(ph) @ B
            vis[i, k] = (sI * sII + sIc + sQU * np.real(pol_c)) / norm
    return vis


def test_compute_vis_fast_matches_reference():
    beam, I, Q, U, rm, mask, freqs = _stub(48, 4)
    got = compute_vis_fast(I, Q, U, rm, beam, freqs, mask)
    exp = _ref_vis(beam, I, Q, U, rm, mask, freqs)
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)


def test_below_horizon_pixels_do_not_affect_output():
    beam, I, Q, U, rm, mask, freqs = _stub(48, 4)
    v1 = compute_vis_fast(I, Q, U, rm, beam, freqs, mask)
    below = ~mask
    I2, Q2, U2, rm2 = I.copy(), Q.copy(), U.copy(), rm.copy()
    I2[:, below] += 1e3
    Q2[:, below] += 5.0
    U2[:, below] -= 3.0
    rm2[:, below] += 100.0
    v2 = compute_vis_fast(I2, Q2, U2, rm2, beam, freqs, mask)
    np.testing.assert_allclose(v1, v2, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run to verify the reference test passes and pin current behaviour**

Run: `uv run pytest tests/test_fast_sim.py -v`
Expected: both pass on the CURRENT implementation (the current code already masks `A`/`B`, so both hold). This pins behaviour before optimizing.

- [ ] **Step 3: Optimize `compute_vis_fast` to skip below-horizon pixels**

Replace the per-time loop body in `src/lusee_faraday/fast_sim.py` (lines ~176–220) so the trig and matmuls run only over above-horizon pixels. The `I` sums stay over all pixels (they are frequency-independent and cheap). Concretely, replace the body from `for i in range(ntimes):` through `vis[i] = np.real(I_contrib) / norm` with:

```python
    keep = mask.astype(bool)

    for i in range(ntimes):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Visibilities: time step {i + 1}/{ntimes}")

        I_m = (I_topo[i] - cmb) * m
        Q_m = Q_topo[i] * m
        U_m = U_topo[i] * m
        cmb_m = cmb * m

        # I contribution (frequency-independent shape; full-pixel sums)
        I_contrib = np.zeros((3, nfreq), dtype=complex)
        for k, pol in enumerate(pols):
            sII = np.sum(w[f"wI_{pol}"] * I_m)
            sIc = np.sum(w[f"wI_{pol}"] * cmb_m)
            I_contrib[k] = scale_I * sII + sIc

        # A, B only over above-horizon pixels (below-horizon are zero)
        rm_i = rm_topo[i][keep]
        A = np.zeros((3, keep.sum()), dtype=complex)
        B = np.zeros((3, keep.sum()), dtype=complex)
        for k, pol in enumerate(pols):
            wQ = w[f"wQ_{pol}"][keep]
            wU = w[f"wU_{pol}"][keep]
            Qk = Q_m[keep]
            Uk = U_m[keep]
            A[k] = wQ * Qk + wU * Uk
            B[k] = wU * Qk - wQ * Uk

        for f0 in range(0, nfreq, batch_size):
            f1 = min(f0 + batch_size, nfreq)
            lsq = lambda_sq[f0:f1]
            phase = 2 * rm_i[None, :] * lsq[:, None]
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            for k in range(3):
                pol_contrib = cos_p @ A[k] + sin_p @ B[k]
                I_contrib[k, f0:f1] += scale_QU[f0:f1] * pol_contrib

        vis[i] = np.real(I_contrib) / norm
```

- [ ] **Step 4: Run to verify still correct**

Run: `uv run pytest tests/test_fast_sim.py -v && uv run pytest -q`
Expected: `test_fast_sim.py` 2 passed; full suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/fast_sim.py tests/test_fast_sim.py
git commit -m "perf(fast_sim): restrict per-frequency trig to above-horizon pixels"
```

---

## Task 2: `SpectrometerResponse.truncate`

**Files:** Modify `src/lusee_faraday/spectrometer.py`, Create `tests/test_spectrometer.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_spectrometer.py`)

```python
import numpy as np
from lusee_faraday import SpectrometerResponse

SPEC_PATH = "data/spectrometer_bin_response.txt"


def _spec():
    return SpectrometerResponse.from_file(SPEC_PATH)


def test_truncate_reduces_points_and_is_symmetric():
    s = _spec()
    t = s.truncate(0.999)
    assert t.freq_offset_hz.size < s.freq_offset_hz.size
    # ~36% retained at 0.999 (verified empirically)
    assert 0.3 < t.freq_offset_hz.size / s.freq_offset_hz.size < 0.45
    assert np.isclose(t.freq_offset_hz.min(), -t.freq_offset_hz.max())


def test_truncate_renormalizes():
    t = _spec().truncate(0.999)
    assert np.isclose(t._wide_norm.sum(), 1.0)
    np.testing.assert_allclose(t._narrow_norm.sum(axis=0), 1.0)


def test_truncate_preserves_channelization_on_smooth_spectrum():
    s = _spec()
    t = s.truncate(0.999)
    # smooth spectrum: bin-center value should be ~unchanged
    full = 100.0 + 2.0 * s.freq_offset_mhz
    trunc = 100.0 + 2.0 * t.freq_offset_mhz
    assert np.isclose(s.apply_wide(full), t.apply_wide(trunc), atol=1e-3)
    np.testing.assert_allclose(
        s.apply_narrow(full), t.apply_narrow(trunc), atol=1e-3
    )


def test_truncate_preserves_faraday_depolarization():
    # the depolarization factor (the physics that matters) is preserved
    s = _spec()
    t = s.truncate(0.999)
    c = 3e8
    for center, rm in [(30.0, 20.0), (50.0, 20.0)]:
        nuf = (center + s.freq_offset_mhz) * 1e6
        nut = (center + t.freq_offset_mhz) * 1e6
        Pf = np.exp(2j * rm * (c / nuf) ** 2)
        Pt = np.exp(2j * rm * (c / nut) ** 2)
        assert abs(abs(s.apply_wide(Pf)) - abs(t.apply_wide(Pt))) < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spectrometer.py -v`
Expected: FAIL (`AttributeError: ... has no attribute 'truncate'`).

- [ ] **Step 3: Implement `truncate`** — add to `src/lusee_faraday/spectrometer.py` inside the class:

```python
    def truncate(self, support=0.999):
        """Drop response wings, keeping the smallest symmetric offset
        window that retains `support` of every channel's weight.

        Returns a new SpectrometerResponse (re-normalized in __init__).
        """
        cols = np.column_stack([self._wide_norm, self._narrow_norm])
        order = np.argsort(np.abs(self.freq_offset_hz))
        half_width = 0.0
        for k in range(cols.shape[1]):
            cum = np.cumsum(cols[order, k])
            idx = min(int(np.searchsorted(cum, support)), order.size - 1)
            half_width = max(half_width, abs(self.freq_offset_hz[order][idx]))
        keep = np.abs(self.freq_offset_hz) <= half_width
        return SpectrometerResponse(
            self.freq_offset_hz[keep], self.wide[keep], self.narrow[keep]
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spectrometer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/spectrometer.py tests/test_spectrometer.py
git commit -m "feat(spectrometer): truncate response to significant support"
```

---

## Task 3: Per-mode decimation + truncation in `FrequencyPlan`

**Files:** Modify `src/lusee_faraday/freqplan.py`, extend `tests/test_freqplan.py`

Backward compatible: `decimation` may be an int (all specs) or a dict keyed by mode; `support=1.0` (default) keeps the full response. Existing Step-1 calls are unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_freqplan.py`)

```python
def test_per_mode_decimation(spec):
    plan = FrequencyPlan(
        spec, [(30.0, "zoom"), (40.0, "wide")],
        decimation={"zoom": 10, "wide": 250},
    )
    # plan still channelizes correctly and produces 65 channels
    raw = np.ones(plan.sim_freqs().size)
    assert plan.channelize(raw).shape == (65,)
    t = plan.channel_table
    assert t["nu"].shape == (65,)


def test_support_truncation_shrinks_sim_grid(spec):
    full = FrequencyPlan(spec, [(10.0, "zoom")])
    trunc = FrequencyPlan(spec, [(10.0, "zoom")], support=0.999)
    assert trunc.sim_freqs().size < full.sim_freqs().size


def test_truncated_channelize_matches_full_on_faraday(spec):
    # the depolarization the sim must capture is preserved under truncation
    c = 3e8
    full = FrequencyPlan(spec, [(30.0, "wide")])
    trunc = FrequencyPlan(spec, [(30.0, "wide")], support=0.999)
    Pf = np.exp(2j * 20.0 * (c / (full.sim_freqs() * 1e6)) ** 2)
    Pt = np.exp(2j * 20.0 * (c / (trunc.sim_freqs() * 1e6)) ** 2)
    assert np.allclose(
        np.abs(full.channelize(Pf)), np.abs(trunc.channelize(Pt)), atol=1e-3
    )


def test_int_decimation_still_supported(spec):
    # backward compatibility with Step-1 signature
    plan = FrequencyPlan(spec, [(30.0, "zoom")], decimation=10)
    assert plan.channel_table["nu"].shape == (64,)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_freqplan.py -k "per_mode or support or truncated or int_decimation" -v`
Expected: FAIL (per-mode dict / support not handled).

- [ ] **Step 3: Reimplement `FrequencyPlan.__init__`** in `src/lusee_faraday/freqplan.py` to support per-mode decimation and truncation. Replace the existing `__init__` with:

```python
    def __init__(self, response, specs, decimation=1, support=1.0):
        """response: SpectrometerResponse. specs: list of
        (center_mhz, mode), mode in {"zoom", "wide"}.
        decimation: int applied to all specs, or a dict keyed by mode.
        support: if < 1, truncate the response to that weight fraction.
        """
        base = response.truncate(support) if support < 1.0 else response
        self.specs = [(_snap_to_lusee(c), m) for c, m in specs]
        self._resp = []
        self._off_hz = []
        abs_hz = []
        for c, mode in self.specs:
            dec = decimation[mode] if isinstance(decimation, dict) else decimation
            r = base.decimate(dec) if dec > 1 else base
            off = np.round(r.freq_offset_hz).astype(np.int64)
            self._resp.append(r)
            self._off_hz.append(off)
            abs_hz.append(np.round(c * 1e6).astype(np.int64) + off)
        self._grid_hz = np.unique(np.concatenate(abs_hz))
        self._idx = [np.searchsorted(self._grid_hz, a) for a in abs_hz]
```

And update `channelize` to use the per-spec response `self._resp` instead of the single `self.response`:

```python
    def channelize(self, raw):
        """Map a raw spectrum (..., nraw) aligned with sim_freqs() to
        the spectrometer channels (..., nchan)."""
        out = []
        for (_, mode), r, idx in zip(self.specs, self._resp, self._idx):
            window = raw[..., idx]
            if mode == "wide":
                out.append(r.apply_wide(window)[..., None])
            else:
                out.append(r.apply_narrow(window))
        return np.concatenate(out, axis=-1)
```

(Remove the now-unused `self.response` attribute. `sim_freqs` and `channel_table` are unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_freqplan.py -v`
Expected: all pass (Step-1 tests + 4 new). Step-1 tests must still pass (backward compatibility).

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/freqplan.py tests/test_freqplan.py
git commit -m "feat(freqplan): per-mode decimation and response truncation"
```

---

## Task 4: Time-parallel `compute_vis_fast_parallel`

**Files:** Modify `src/lusee_faraday/fast_sim.py`, extend `tests/test_fast_sim.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_fast_sim.py`)

```python
from lusee_faraday.fast_sim import compute_vis_fast_parallel


def test_parallel_matches_serial():
    beam, I, Q, U, rm, mask, freqs = _stub(48, 6)
    serial = compute_vis_fast(I, Q, U, rm, beam, freqs, mask)
    par = compute_vis_fast_parallel(
        I, Q, U, rm, beam, freqs, mask, nproc=2
    )
    np.testing.assert_allclose(serial, par, rtol=1e-12, atol=1e-12)


def test_parallel_nproc_one_is_serial():
    beam, I, Q, U, rm, mask, freqs = _stub(48, 3)
    serial = compute_vis_fast(I, Q, U, rm, beam, freqs, mask)
    par = compute_vis_fast_parallel(
        I, Q, U, rm, beam, freqs, mask, nproc=1
    )
    np.testing.assert_allclose(serial, par, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_fast_sim.py -k parallel -v`
Expected: FAIL (no attribute `compute_vis_fast_parallel`).

- [ ] **Step 3: Implement** — add to `src/lusee_faraday/fast_sim.py`:

```python
from concurrent.futures import ProcessPoolExecutor


def _vis_chunk(args):
    I_t, Q_t, U_t, rm_t, beam, freqs, mask, kwargs = args
    return compute_vis_fast(I_t, Q_t, U_t, rm_t, beam, freqs, mask, **kwargs)


def compute_vis_fast_parallel(
    I_topo, Q_topo, U_topo, rm_topo, beam, freqs, mask,
    nproc=None, **kwargs,
):
    """Parallel `compute_vis_fast` over the time axis.

    Splits the ntimes axis into `nproc` chunks run in separate
    processes and concatenates the results. nproc=None or 1 runs
    serially. Extra kwargs are forwarded to compute_vis_fast.
    """
    ntimes = I_topo.shape[0]
    if nproc in (None, 1) or ntimes < 2:
        return compute_vis_fast(
            I_topo, Q_topo, U_topo, rm_topo, beam, freqs, mask, **kwargs
        )
    chunks = np.array_split(np.arange(ntimes), nproc)
    args = [
        (I_topo[c], Q_topo[c], U_topo[c], rm_topo[c],
         beam, freqs, mask, kwargs)
        for c in chunks if c.size
    ]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        results = list(ex.map(_vis_chunk, args))
    return np.concatenate(results, axis=0)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fast_sim.py -v`
Expected: all pass (reference, masking-invariant, parallel, nproc=1).

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/fast_sim.py tests/test_fast_sim.py
git commit -m "perf(fast_sim): parallelize compute_vis_fast over time steps"
```

---

## Self-Review

**Spec coverage (Step-2a slice):** Masking → Task 1; truncation → Tasks 2–3; per-mode decimation (for the mixed wide/zoom grid) → Task 3; parallelism → Task 4. The grid-design exploration, the `faraday_fullband_sim.py` driver, LST tagging into outputs, and the actual sim run are Step 2b (separate plan).

**Placeholder scan:** Every code step has complete code; every command has expected output. The reference `_ref_vis` and `_stub` helpers are defined in the test file.

**Type consistency:** `compute_vis_fast(...)` signature unchanged (behaviour-preserving optimization); `compute_vis_fast_parallel(..., nproc=None, **kwargs)`; `SpectrometerResponse.truncate(support=0.999)`; `FrequencyPlan(response, specs, decimation=1, support=1.0)` with `decimation` int-or-dict. All existing Step-1 call sites (`FrequencyPlan(response, specs)`, `compute_vis_fast(...)`) remain valid — verified by keeping the Step-1 freqplan tests green.

**Accuracy note:** truncation is the only approximation; at support=0.999 it is <1e-4 vs full and is covered by tests asserting the Faraday depolarization factor is preserved. Masking and parallelism are exact (tested to 1e-12).
```
