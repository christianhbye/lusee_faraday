# Faraday RM-Synthesis — Step 1 (FrequencyPlan + noise + LST) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable infrastructure for full-band RM synthesis: a `FrequencyPlan` that switches each parent bin between zoom and wide channelization on a deduplicated raw grid, a radiometer `noise` module, and an LST-tagging helper.

**Architecture:** Three independent, fully-tested library pieces under `src/lusee_faraday/`. `FrequencyPlan` reuses the existing `SpectrometerResponse.apply_wide/apply_narrow` (proven correct) and indexes a shared integer-Hz raw grid so overlapping parent windows are computed once. The channel table derives each channel's effective frequency from the response (correct for the non-monotonic zoom-bin ordering), which both the sim and `rmsynth` consume.

**Tech Stack:** Python, NumPy, lunarsky, pytest, uv.

**Source spec:** `docs/superpowers/specs/2026-06-08-faraday-rmsynth-design.md`

---

## Verified facts this plan relies on (from prototyping)

- The raw response grid is ±50 kHz at 10 Hz (10001 pts); LuSEE parents sit on a 25 kHz lattice, so every parent's `center + offset` lands on a common integer-Hz lattice → **exact dedup, no interpolation**.
- `apply_narrow` columns are **non-monotonic in frequency** (effective offsets 0→+8→−8 kHz over column index). The channel table must therefore compute each channel's frequency as `center + weighted-mean offset of its response column`. Equivalently and DRY: `nu == channelize(sim_freqs())`.
- A prototype `channelize` reproduced per-parent `apply_wide`/`apply_narrow` to 0.0 difference; dedup reduced the grid as expected; 2D `(ntimes, nraw)` inputs work.

## Scope / deferrals

- This plan delivers library code only (no slow sim run). **Rewiring `faraday_sims.py` onto `FrequencyPlan` and writing LST tags into the sim outputs is deferred to the start of Step 2**, where the curated sim is configured and can be verified end-to-end.
- **Truncation** of the response to its significant support (a further compute optimization) is deferred; v1 uses dedup + per-parent `decimation`, which already gives large savings and reuses the exact, tested channelization. Add truncation later only if Step-2 compute demands it.
- Note for Step 3: the Step-0 calibration (`notebooks/rmsynth_calibration.py`) used `utils.freqs_zoom` (monotonic) which mis-pairs with `apply_narrow` column order; the proper analysis must use `FrequencyPlan.channel_table` frequencies instead.

## File Structure

- Create: `src/lusee_faraday/noise.py` — `radiometer_sigma`, `add_noise`.
- Create: `src/lusee_faraday/freqplan.py` — `FrequencyPlan` (+ private `_snap_to_lusee`).
- Modify: `src/lusee_faraday/rotations.py` — add `topo_euler_angles`.
- Modify: `src/lusee_faraday/__init__.py` — expose `noise` and `FrequencyPlan`.
- Test: `tests/test_noise.py`, `tests/test_freqplan.py`, `tests/test_rotations.py`.

Conventions: readability first, sparse comments, Black line-length 79. Touch only the files listed per task. If `uv run` modifies `pyproject.toml`/`uv.lock`, `git checkout` them before committing.

---

## Task 1: `noise.py`

**Files:** Create `src/lusee_faraday/noise.py`, `tests/test_noise.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_noise.py`)

```python
import numpy as np
from lusee_faraday import noise


def test_radiometer_sigma_matches_formula():
    sig = noise.radiometer_sigma(100.0, 390.625, 3600.0)
    assert np.isclose(sig, 100.0 / np.sqrt(390.625 * 3600.0))


def test_radiometer_sigma_vectorized():
    T = np.array([100.0, 200.0])
    dnu = np.array([390.625, 25000.0])
    sig = noise.radiometer_sigma(T, dnu, 3600.0)
    assert sig.shape == (2,)
    assert np.allclose(sig, T / np.sqrt(dnu * 3600.0))


def test_add_noise_statistics():
    rng = np.random.default_rng(0)
    x = np.zeros(200000)
    y = noise.add_noise(x, 2.0, rng)
    assert abs(np.std(y) - 2.0) < 0.05
    assert abs(np.mean(y)) < 0.05


def test_add_noise_per_channel_sigma_broadcasts():
    rng = np.random.default_rng(1)
    sigma = np.array([1.0, 5.0])
    y = noise.add_noise(np.zeros((50000, 2)), sigma, rng)
    assert np.isclose(np.std(y[:, 0]), 1.0, atol=0.05)
    assert np.isclose(np.std(y[:, 1]), 5.0, atol=0.1)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_noise.py -v`
Expected: FAIL (`ModuleNotFoundError`/no attribute).

- [ ] **Step 3: Implement** (`src/lusee_faraday/noise.py`)

```python
"""Radiometer noise for LuSEE polarized spectra.

sigma = T_sys / sqrt(dnu * dt). T_sys is sky-dominated (~ Stokes I).
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_noise.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/noise.py tests/test_noise.py
git commit -m "feat(noise): radiometer_sigma and add_noise"
```

---

## Task 2: `FrequencyPlan` core (sim_freqs + channelize)

**Files:** Create `src/lusee_faraday/freqplan.py`, `tests/test_freqplan.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_freqplan.py`)

```python
import numpy as np
import pytest
from lusee_faraday import SpectrometerResponse
from lusee_faraday.freqplan import FrequencyPlan, _snap_to_lusee

SPEC_PATH = "data/spectrometer_bin_response.txt"


@pytest.fixture(scope="module")
def spec():
    return SpectrometerResponse.from_file(SPEC_PATH)


def test_snap_to_lusee_rounds_to_grid():
    # 30.0 MHz is on the 25 kHz grid; a nearby value snaps to it
    assert np.isclose(_snap_to_lusee(30.0), 30.0)
    assert np.isclose(_snap_to_lusee(30.01), 30.0)


def test_sim_freqs_dedups_overlapping_parents(spec):
    specs = [(29.975, "zoom"), (30.0, "zoom")]
    plan = FrequencyPlan(spec, specs)
    sf = plan.sim_freqs()
    naive = 2 * len(spec.freq_offset_hz)
    assert sf.size < naive          # overlap removed
    assert np.all(np.diff(sf) > 0)  # sorted, unique


def test_channelize_matches_apply_narrow_and_wide(spec):
    specs = [(30.0, "zoom"), (40.0, "wide")]
    plan = FrequencyPlan(spec, specs)
    sf = plan.sim_freqs()
    raw = 1000.0 + (sf - sf.mean()) ** 2  # arbitrary smooth spectrum
    ch = plan.channelize(raw)
    # reference: evaluate each parent on its own window
    off_hz = np.round(spec.freq_offset_hz).astype(np.int64)
    ref = []
    for c, m in plan.specs:
        a = np.round(c * 1e6).astype(np.int64) + off_hz
        win = raw[np.searchsorted(sf_to_hz(sf), a)]
        if m == "wide":
            ref.append(np.atleast_1d(spec.apply_wide(win)))
        else:
            ref.append(spec.apply_narrow(win))
    ref = np.concatenate(ref)
    assert ch.shape == (65,)
    assert np.allclose(ch, ref)


def sf_to_hz(sf):
    return np.round(sf * 1e6).astype(np.int64)


def test_channelize_2d_input(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    raw = np.ones((3, plan.sim_freqs().size))
    ch = plan.channelize(raw)
    assert ch.shape == (3, 65)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_freqplan.py -v`
Expected: FAIL (import error / no attribute).

- [ ] **Step 3: Implement** (`src/lusee_faraday/freqplan.py`)

```python
"""Frequency plan: per-parent choice of zoom or wide channelization.

A plan is a list of (center_mhz, mode) specs. It builds the minimal
deduplicated raw frequency grid to simulate (`sim_freqs`) and maps a
simulated raw spectrum onto the spectrometer channels (`channelize`),
reusing SpectrometerResponse.apply_wide / apply_narrow. All parent
centers and response offsets share a common integer-Hz lattice, so
overlapping windows are deduplicated exactly without interpolation.
"""

import numpy as np

from .utils import freqs_lusee
from .rmsynth import lambda2

BIN_WIDTH_HZ = 25000.0
N_ZOOM = 64


def _snap_to_lusee(center_mhz):
    f = freqs_lusee()
    return float(f[np.argmin(np.abs(f - center_mhz))])


class FrequencyPlan:
    def __init__(self, response, specs, decimation=1):
        """response: SpectrometerResponse. specs: list of
        (center_mhz, mode) with mode in {"zoom", "wide"}.
        decimation: subsample the raw response grid."""
        self.response = (
            response.decimate(decimation) if decimation > 1 else response
        )
        self.specs = [(_snap_to_lusee(c), m) for c, m in specs]
        self._off_hz = np.round(self.response.freq_offset_hz).astype(np.int64)
        abs_hz = [
            np.round(c * 1e6).astype(np.int64) + self._off_hz
            for c, _ in self.specs
        ]
        self._grid_hz = np.unique(np.concatenate(abs_hz))
        self._idx = [np.searchsorted(self._grid_hz, a) for a in abs_hz]

    def sim_freqs(self):
        """Sorted unique absolute frequencies to simulate (MHz)."""
        return self._grid_hz * 1e-6

    def channelize(self, raw):
        """Map a raw spectrum (..., nraw) aligned with sim_freqs() to
        the spectrometer channels (..., nchan)."""
        out = []
        for (_, mode), idx in zip(self.specs, self._idx):
            window = raw[..., idx]
            if mode == "wide":
                out.append(self.response.apply_wide(window)[..., None])
            else:
                out.append(self.response.apply_narrow(window))
        return np.concatenate(out, axis=-1)

    @property
    def channel_table(self):
        """Per-channel nu (MHz), lambda2 (m^2), dnu (Hz). nu is the
        response-weighted effective frequency (correct for the
        non-monotonic zoom ordering)."""
        nu = self.channelize(self.sim_freqs())
        dnu = []
        for _, mode in self.specs:
            if mode == "wide":
                dnu.append(BIN_WIDTH_HZ)
            else:
                dnu.extend([BIN_WIDTH_HZ / N_ZOOM] * N_ZOOM)
        dnu = np.array(dnu)
        return {"nu": nu, "lambda2": lambda2(nu), "dnu": dnu}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_freqplan.py -v`
Expected: 4 passed (snap, dedup, channelize-matches, 2d). (The `channel_table` is covered in Task 3.)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/freqplan.py tests/test_freqplan.py
git commit -m "feat(freqplan): FrequencyPlan sim_freqs and channelize"
```

---

## Task 3: `FrequencyPlan.channel_table`

**Files:** Modify `tests/test_freqplan.py` (the implementation already exists from Task 2; this task adds its tests).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_freqplan.py`)

```python
def test_channel_table_sizes_and_dnu(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    t = plan.channel_table
    assert t["nu"].shape == (65,)
    assert t["lambda2"].shape == (65,)
    # first 64 are zoom sub-bins, last is the wide channel
    assert np.allclose(t["dnu"][:64], BIN_WIDTH_HZ / N_ZOOM)
    assert np.isclose(t["dnu"][64], BIN_WIDTH_HZ)


def test_channel_table_nu_equals_channelized_frequency_axis(spec):
    # invariant: channelizing the frequency grid returns the channel
    # centers, so nu is self-consistent with channelize
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    nu = plan.channel_table["nu"]
    assert np.allclose(nu, plan.channelize(plan.sim_freqs()))


def test_channel_table_zoom_is_non_monotonic(spec):
    # apply_narrow columns are FFT-ordered, so the 64 zoom centers are
    # NOT sorted in frequency -- this guards the ordering fix
    plan = FrequencyPlan(spec, [(30.0, "zoom")])
    nu = plan.channel_table["nu"]
    assert nu.size == 64
    assert not np.all(np.diff(nu) > 0)
    # all within +/- 12.5 kHz of the parent center
    assert np.all(np.abs(nu - 30.0) <= BIN_WIDTH_HZ / 2 * 1e-6 + 1e-9)


def test_channel_table_lambda2_consistent(spec):
    from lusee_faraday import rmsynth
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    t = plan.channel_table
    assert np.allclose(t["lambda2"], rmsynth.lambda2(t["nu"]))
```

Also add the imports `from lusee_faraday.freqplan import BIN_WIDTH_HZ, N_ZOOM` at the top of the test file (alongside the existing freqplan import).

- [ ] **Step 2: Run to verify pass** (implementation already present from Task 2)

Run: `uv run pytest tests/test_freqplan.py -v`
Expected: all pass (Task 2 tests + 4 new). If `test_channel_table_*` fail, fix `channel_table` in `freqplan.py` until they pass; do not weaken the tests.

- [ ] **Step 3: Commit**

```bash
git add src/lusee_faraday/freqplan.py tests/test_freqplan.py
git commit -m "test(freqplan): channel_table sizes, dnu, non-monotonic zoom nu"
```

---

## Task 4: LST helper `topo_euler_angles`

**Files:** Modify `src/lusee_faraday/rotations.py`, Create `tests/test_rotations.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_rotations.py`)

```python
import numpy as np
from lunarsky import Time, MoonLocation
import astropy.units as u
from lusee_faraday import rotations
from lusee_faraday.sky import LUSEE_LOC


def _times(n):
    t0 = Time("2027-01-01T09:00:00", location=MoonLocation(
        lat=-23.813, lon=182.258))
    return np.linspace(t0, t0 + 655.720 * 3600 * u.s, num=n, endpoint=False)


def test_topo_euler_angles_shape():
    ang = rotations.topo_euler_angles(_times(3), LUSEE_LOC)
    assert ang.shape == (3, 3)


def test_topo_euler_angles_changes_with_time():
    ang = rotations.topo_euler_angles(_times(3), LUSEE_LOC)
    # the sky orientation must differ between distinct times
    assert not np.allclose(ang[0], ang[-1])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_rotations.py -v`
Expected: FAIL (no attribute `topo_euler_angles`).

- [ ] **Step 3: Implement** — add to `src/lusee_faraday/rotations.py`.

Add this import near the top (with the existing imports):

```python
from lunarsky import LunarTopo
```

Add this function at the end of the file:

```python
def topo_euler_angles(times, location):
    """Galactic->topocentric ZYX Euler angles for each observation time.

    Returns an (ntimes, 3) array of (alpha, beta, gamma). The angles
    track the sky orientation versus lunar sidereal time and serve as
    the LST tag for later per-orientation coadd.
    """
    angles = np.empty((len(times), 3))
    for i, t in enumerate(times):
        topo = LunarTopo(location=location, obstime=t)
        angles[i] = rotmat_to_eulerZYX(get_rot_mat(topo))
    return angles
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_rotations.py -v`
Expected: 2 passed. (May take a few seconds per time due to lunarsky frame evaluation; this is expected.)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/rotations.py tests/test_rotations.py
git commit -m "feat(rotations): topo_euler_angles for LST tagging"
```

---

## Task 5: Expose `noise` and `FrequencyPlan` in package init

**Files:** Modify `src/lusee_faraday/__init__.py`

- [ ] **Step 1: Inspect current init**

Run: `uv run python -c "import lusee_faraday as ld; print([x for x in dir(ld) if not x.startswith('_')])"`
Expected: lists existing exports (incl. `rmsynth`) without `noise` or `FrequencyPlan`.

- [ ] **Step 2: Add the exports**

Add to `src/lusee_faraday/__init__.py`, matching existing style (module imports with the other `from . import ...`; class import with the other `from .X import Y`):

```python
from . import noise
from .freqplan import FrequencyPlan
```

- [ ] **Step 3: Verify**

Run: `uv run python -c "import lusee_faraday as ld; print(ld.FrequencyPlan, ld.noise.radiometer_sigma(100,390.625,3600))"`
Expected: prints the class and a finite number (~0.0843).

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (rmsynth + noise + freqplan + rotations).

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/__init__.py
git commit -m "feat: expose noise and FrequencyPlan in package init"
```

---

## Self-Review

**Spec coverage (Step-1 slice):** Zoom/wide switch + deduplicated raw grid → `FrequencyPlan` (Tasks 2–3). Channel table `(nu, lambda2, dnu)` consumed by sim + rmsynth → Task 3, with the zoom-ordering correctness fix. Radiometer noise + MC draw → `noise.py` (Task 1). LST tagging groundwork → `topo_euler_angles` (Task 4). Deferred (documented above): truncation optimization; `faraday_sims.py` rewiring + writing LST tags into outputs (→ Step 2); Step-0 calibration uses incorrect zoom frequencies (→ fixed in Step 3 via channel_table).

**Placeholder scan:** Every code step has complete code; every command has expected output. The `sf_to_hz` helper used in Task 2's test is defined within the same test file.

**Type consistency:** `FrequencyPlan(response, specs, decimation=1)`, `.sim_freqs()`, `.channelize(raw)`, `.channel_table` (dict keys `nu`/`lambda2`/`dnu`); module constants `BIN_WIDTH_HZ`, `N_ZOOM`; `radiometer_sigma(T_sys, dnu_hz, dt_s)`, `add_noise(stokes, sigma, rng)`; `topo_euler_angles(times, location)`. Names are consistent across tasks and tests. `channel_table["nu"]` is defined as `channelize(sim_freqs())`, guaranteeing sim/analysis consistency.
