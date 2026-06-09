# Faraday RM-Synthesis — Step 0 (rmsynth module + calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, model-independent RM-synthesis module and use it to run a calibration on the existing 3-band sim outputs, so the measured RMSF/sidelobes decide the adaptive grid for later steps.

**Architecture:** Pure NumPy functions in `src/lusee_faraday/rmsynth.py` operating on a flat channel set `(nu, lambda2, Q, U)`. A calibration script loads `results/faraday_sim_{10,30,50}mhz.npz`, concatenates the 64 zoom sub-bins of each band into one 192-channel comb, and runs the transform + RMSF.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, uv. Reference: Brentjens & de Bruyn (2005) RM synthesis.

**Source spec:** `docs/superpowers/specs/2026-06-08-faraday-rmsynth-design.md`

---

## File Structure

- Create: `src/lusee_faraday/rmsynth.py` — `lambda2`, `faraday_resolution`, `max_scale`, `phi_grid`, `rmsf`, `faraday_spectrum`.
- Modify: `src/lusee_faraday/__init__.py` — expose `rmsynth` for notebook use (`ld.rmsynth`).
- Create: `tests/test_rmsynth.py` — unit tests for every function.
- Create: `notebooks/rmsynth_calibration.py` — calibration script (module-style, like the existing `faraday_sims.py`), saves figures to `notebooks/results/`.

Conventions: readability first, sparse comments, Black line-length 79.

---

## Task 1: `lambda2`, `faraday_resolution`, `max_scale`

**Files:**
- Create: `src/lusee_faraday/rmsynth.py`
- Test: `tests/test_rmsynth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rmsynth.py
import numpy as np
import pytest
from lusee_faraday import rmsynth

C = 299792458.0


def test_lambda2_matches_formula():
    nu = np.array([10.0, 30.0, 50.0])
    expected = (C / (nu * 1e6)) ** 2
    np.testing.assert_allclose(rmsynth.lambda2(nu), expected)


def test_lambda2_inverse_square_scaling():
    # lambda^2 ~ 1/nu^2, so halving nu quadruples lambda^2
    assert np.isclose(rmsynth.lambda2(10) / rmsynth.lambda2(20), 4.0)


def test_faraday_resolution():
    # FWHM of RMSF = 2*sqrt(3) / (lam2_max - lam2_min)
    lam2 = np.linspace(0.0, 1.0, 50)
    assert np.isclose(rmsynth.faraday_resolution(lam2), 2 * np.sqrt(3))


def test_max_scale():
    # largest recoverable scale = pi / lam2_min
    lam2 = np.array([1.0, 2.0, 3.0])
    assert np.isclose(rmsynth.max_scale(lam2), np.pi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rmsynth.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'lusee_faraday.rmsynth' has no attribute ...`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lusee_faraday/rmsynth.py
"""Rotation-measure (RM) synthesis for LuSEE Faraday detection.

Operates on a flat channel set: per-channel frequency (MHz), lambda^2
(m^2), and Stokes Q/U. The complex polarization P = Q + iU rotates as
exp(2i * RM * lambda^2); RM synthesis is the matched transform that maps
P(lambda^2) to the Faraday spectrum F(phi).
"""

import numpy as np

C = 299792458.0  # speed of light, m/s


def lambda2(nu_mhz):
    """Wavelength squared (m^2) for frequencies given in MHz."""
    nu_mhz = np.asarray(nu_mhz, dtype=float)
    return (C / (nu_mhz * 1e6)) ** 2


def faraday_resolution(lam2):
    """FWHM of the RMSF main lobe (rad/m^2), set by lambda^2 coverage."""
    lam2 = np.asarray(lam2, dtype=float)
    return 2 * np.sqrt(3) / (lam2.max() - lam2.min())


def max_scale(lam2):
    """Largest recoverable Faraday-thick scale (rad/m^2)."""
    lam2 = np.asarray(lam2, dtype=float)
    return np.pi / lam2.min()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rmsynth.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/rmsynth.py tests/test_rmsynth.py
git commit -m "feat(rmsynth): lambda2, faraday_resolution, max_scale"
```

---

## Task 2: `phi_grid`

**Files:**
- Modify: `src/lusee_faraday/rmsynth.py`
- Test: `tests/test_rmsynth.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_rmsynth.py
def test_phi_grid_range_and_symmetry():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=100.0, dphi=1.0)
    assert np.isclose(phi[0], -100.0)
    assert np.isclose(phi[-1], 100.0)
    assert np.isclose(phi[len(phi) // 2], 0.0)


def test_phi_grid_spacing_respects_dphi():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=50.0, dphi=0.5)
    assert np.all(np.diff(phi) <= 0.5 + 1e-9)


def test_phi_grid_default_dphi_oversamples_resolution():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    phi = rmsynth.phi_grid(lam2, phi_max=10.0, oversample=3)
    assert np.median(np.diff(phi)) <= rmsynth.faraday_resolution(lam2) / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rmsynth.py -k phi_grid -v`
Expected: FAIL with `AttributeError: ... has no attribute 'phi_grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/lusee_faraday/rmsynth.py
def phi_grid(lam2, phi_max, dphi=None, oversample=3):
    """Symmetric Faraday-depth grid on [-phi_max, phi_max].

    If dphi is None it defaults to faraday_resolution / oversample.
    """
    lam2 = np.asarray(lam2, dtype=float)
    if dphi is None:
        dphi = faraday_resolution(lam2) / oversample
    # round up so the actual linspace spacing never exceeds dphi
    n = int(np.ceil(2 * phi_max / dphi)) + 1
    return np.linspace(-phi_max, phi_max, n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rmsynth.py -k phi_grid -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/rmsynth.py tests/test_rmsynth.py
git commit -m "feat(rmsynth): phi_grid"
```

---

## Task 3: `rmsf`

**Files:**
- Modify: `src/lusee_faraday/rmsynth.py`
- Test: `tests/test_rmsynth.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_rmsynth.py
def test_rmsf_peak_at_zero_is_unity():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    R = rmsynth.rmsf(lam2, np.array([0.0]))
    assert np.isclose(np.abs(R[0]), 1.0)


def test_rmsf_single_channel_is_flat():
    # one channel -> lam2 - ref = 0 -> |R| = 1 everywhere
    lam2 = np.array([100.0])
    phi = np.linspace(-50, 50, 101)
    R = rmsynth.rmsf(lam2, phi)
    np.testing.assert_allclose(np.abs(R), 1.0)


def test_rmsf_weights_normalized():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    w = np.random.default_rng(0).uniform(0.1, 1.0, lam2.size)
    R = rmsynth.rmsf(lam2, np.array([0.0]), weights=w)
    assert np.isclose(np.abs(R[0]), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rmsynth.py -k rmsf -v`
Expected: FAIL with `AttributeError: ... has no attribute 'rmsf'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/lusee_faraday/rmsynth.py
def _normalized_weights(lam2, weights):
    if weights is None:
        weights = np.ones_like(lam2)
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def rmsf(lam2, phi, weights=None):
    """Rotation-measure spread function R(phi).

    R(phi) = sum_k w_k exp(-2i phi (lam2_k - lam2_ref)) / sum_k w_k
    """
    lam2 = np.asarray(lam2, dtype=float)
    phi = np.asarray(phi, dtype=float)
    w = _normalized_weights(lam2, weights)
    lam2_ref = np.sum(w * lam2)
    kernel = np.exp(-2j * np.outer(phi, lam2 - lam2_ref))
    return kernel @ w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rmsynth.py -k rmsf -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/rmsynth.py tests/test_rmsynth.py
git commit -m "feat(rmsynth): rmsf"
```

---

## Task 4: `faraday_spectrum`

**Files:**
- Modify: `src/lusee_faraday/rmsynth.py`
- Test: `tests/test_rmsynth.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_rmsynth.py
def _synthetic_pol(lam2, rm, chi0=0.3, amp=1.0):
    p = amp * np.exp(2j * (chi0 + rm * lam2))
    return p.real, p.imag


def test_faraday_spectrum_recovers_positive_rm():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    rm_true = 8.0
    Q, U = _synthetic_pol(lam2, rm_true)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    peak = phi[np.argmax(np.abs(F[0]))]
    assert abs(peak - rm_true) < rmsynth.faraday_resolution(lam2) * 3


def test_faraday_spectrum_recovers_negative_rm():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    rm_true = -12.0
    Q, U = _synthetic_pol(lam2, rm_true)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    peak = phi[np.argmax(np.abs(F[0]))]
    assert abs(peak - rm_true) < rmsynth.faraday_resolution(lam2) * 3


def test_faraday_spectrum_zero_rm_peaks_at_zero():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 400))
    Q, U = _synthetic_pol(lam2, 0.0)
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    assert abs(phi[np.argmax(np.abs(F[0]))]) < rmsynth.faraday_resolution(lam2) * 3


def test_faraday_spectrum_shape_multi_time():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 100))
    Q = np.zeros((3, lam2.size))
    U = np.zeros((3, lam2.size))
    phi = rmsynth.phi_grid(lam2, phi_max=10.0)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    assert F.shape == (3, phi.size)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rmsynth.py -k faraday_spectrum -v`
Expected: FAIL with `AttributeError: ... has no attribute 'faraday_spectrum'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/lusee_faraday/rmsynth.py
def faraday_spectrum(Q, U, lam2, phi, weights=None):
    """Complex Faraday spectrum F(phi, t) from Stokes Q, U.

    Q, U have shape (nchan,) or (ntimes, nchan). Returns shape
    (ntimes, nphi). F(phi) = sum_k w_k (Q+iU)_k
    exp(-2i phi (lam2_k - lam2_ref)) / sum_k w_k.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    lam2 = np.asarray(lam2, dtype=float)
    phi = np.asarray(phi, dtype=float)
    w = _normalized_weights(lam2, weights)
    lam2_ref = np.sum(w * lam2)
    kernel = np.exp(-2j * np.outer(phi, lam2 - lam2_ref))  # (nphi, nchan)
    P = (Q + 1j * U) * w  # (ntimes, nchan)
    return P @ kernel.T  # (ntimes, nphi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rmsynth.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/rmsynth.py tests/test_rmsynth.py
git commit -m "feat(rmsynth): faraday_spectrum"
```

---

## Task 5: Expose `rmsynth` in package init

**Files:**
- Modify: `src/lusee_faraday/__init__.py`

- [ ] **Step 1: Inspect current init**

Run: `uv run python -c "import lusee_faraday as ld; print([x for x in dir(ld) if not x.startswith('_')])"`
Expected: lists existing exports (e.g. SpectrometerResponse, sky, utils) without `rmsynth`.

- [ ] **Step 2: Add the import**

Add to `src/lusee_faraday/__init__.py` alongside the other submodule imports:

```python
from . import rmsynth
```

- [ ] **Step 3: Verify it imports**

Run: `uv run python -c "import lusee_faraday as ld; print(ld.rmsynth.lambda2(10))"`
Expected: prints `898.755...`

- [ ] **Step 4: Commit**

```bash
git add src/lusee_faraday/__init__.py
git commit -m "feat(rmsynth): expose rmsynth in package init"
```

---

## Task 6: Calibration script on existing 3-band data

**Files:**
- Create: `notebooks/rmsynth_calibration.py`

- [ ] **Step 1: Write the calibration script**

```python
# notebooks/rmsynth_calibration.py
"""Step-0 calibration: RM synthesis on existing 3-band zoom spectra.

Combines the 64 zoom sub-bins of the 10/30/50 MHz sims into one
192-channel comb and runs RM synthesis. The RMSF reveals the sidelobe
structure of this sparse lambda^2 sampling, which sizes the adaptive
grid for the full-band sim.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lusee_faraday import rmsynth

RES = Path(__file__).resolve().parent / "results"
BANDS = [10, 30, 50]
PHI_MAX = 100.0


def load_comb():
    freqs, Q, U, Qn, Un = [], [], [], [], []
    i_gal = None
    for cf in BANDS:
        d = np.load(RES / f"faraday_sim_{cf}mhz.npz")
        freqs.append(d["freqs_zoom"])
        Q.append(d["pQ_FR_zoom"])
        U.append(d["pU_FR_zoom"])
        Qn.append(d["pQ_noFR_zoom"])
        Un.append(d["pU_noFR_zoom"])
        i_gal = int(d["i_gal"])
    freqs = np.concatenate(freqs)
    Q = np.concatenate(Q, axis=1)
    U = np.concatenate(U, axis=1)
    Qn = np.concatenate(Qn, axis=1)
    Un = np.concatenate(Un, axis=1)
    return freqs, Q, U, Qn, Un, i_gal


def main():
    freqs, Q, U, Qn, Un, i_gal = load_comb()
    lam2 = rmsynth.lambda2(freqs)
    phi = rmsynth.phi_grid(lam2, phi_max=PHI_MAX)

    res = rmsynth.faraday_resolution(lam2)
    scale = rmsynth.max_scale(lam2)
    print(f"channels: {lam2.size}")
    print(f"lambda^2 span: {lam2.min():.2f} .. {lam2.max():.2f} m^2")
    print(f"RMSF resolution (FWHM): {res:.4f} rad/m^2")
    print(f"max recoverable scale:  {scale:.4f} rad/m^2")
    print(f"phi grid: {phi.size} points over +/-{PHI_MAX}")

    R = rmsynth.rmsf(lam2, phi)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    Fn = rmsynth.faraday_spectrum(Qn, Un, lam2, phi)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].plot(phi, np.abs(R))
    ax[0].set(title="RMSF |R(phi)|", xlabel="phi [rad/m^2]", yscale="log")

    ax[1].plot(phi, np.abs(F[i_gal]), label="FR")
    ax[1].plot(phi, np.abs(Fn[i_gal]), label="no FR", ls="--")
    ax[1].set(title=f"Faraday spectrum, galaxy-up (t={i_gal})",
              xlabel="phi [rad/m^2]", yscale="log")
    ax[1].legend()

    im = ax[2].imshow(
        np.abs(F), aspect="auto", origin="lower",
        extent=[phi[0], phi[-1], 0, F.shape[0]],
    )
    ax[2].set(title="|F(phi, t)|", xlabel="phi [rad/m^2]", ylabel="time index")
    fig.colorbar(im, ax=ax[2])

    fig.tight_layout()
    out = RES / "rmsynth_calibration.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the calibration**

Run: `uv run python notebooks/rmsynth_calibration.py`
Expected: prints channel count (192), lambda^2 span (~36 .. ~899 m^2), RMSF resolution (~0.004 rad/m^2), max scale, phi grid size; saves `notebooks/results/rmsynth_calibration.png` with no error.

- [ ] **Step 3: Inspect and report for the grid decision**

Open `notebooks/results/rmsynth_calibration.png` and report:
- RMSF main-lobe width and the height/spacing of the largest sidelobes (the sparse 3-cluster comb will produce strong sidelobes — quantify them).
- Whether the FR Faraday spectrum shows power displaced from phi=0 relative to the no-FR baseline, and over what phi range (this is the beam-RM spread).
These numbers feed the Step-1/2 adaptive-grid choice (how many zoom anchors are needed to suppress sidelobes).

- [ ] **Step 4: Commit**

```bash
git add notebooks/rmsynth_calibration.py
git commit -m "feat: step-0 RM-synthesis calibration on existing 3-band sims"
```

---

## Self-Review

**Spec coverage (Step 0 slice):** `rmsynth.py` with `lambda2`/`phi_grid`/`rmsf`/`faraday_spectrum` — Tasks 1–4. Diagnostics (resolution, max_scale, phi_max) — Tasks 1–2. Calibration on existing `faraday_sim_{10,30,50}mhz.npz` zoom spectra → RMSF/sidelobe inspection → grid decision — Task 6. Uniform weights default (inverse-variance deferred to `noise.py` in Step 1) — covered by the optional `weights` argument. Out-of-scope items (FrequencyPlan, noise, LST, sim, attic) are intentionally in later plans.

**Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. Task 6 step 3 is an inspection/reporting action (no code), explicitly framed as such.

**Type consistency:** `lambda2`/`faraday_resolution`/`max_scale`/`phi_grid`/`rmsf`/`faraday_spectrum` signatures and the `_normalized_weights` helper are consistent across tasks and the calibration script. The calibration uses exactly the npz keys verified to exist (`freqs_zoom`, `pQ_FR_zoom`, `pU_FR_zoom`, `pQ_noFR_zoom`, `pU_noFR_zoom`, `i_gal`).

**Note on compute:** with ~150k phi points × 192 channels, the kernel is ~0.5 GB and the transform runs in seconds — within budget.
```
