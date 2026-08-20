# luseepy + croissant Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `lusee_faraday`'s custom simulation machinery with luseepy + croissant, keeping only the sky model, the Faraday operator, and the LuSEE-specific post-processing.

**Architecture:** Faraday rotation is exactly diagonal in croissant's harmonic dual, so a sky becomes a small set of frequency-independent component alms plus a per-frequency, per-dual-block coefficient matrix. Visibilities separate as `V(t,p,nu) = sum_k sum_c coeff[k,nu,c] * W[k,c,t,p]`, so the fine 16,384-channel axis costs `K` contractions and one einsum instead of 16,384 spherical transforms. The as-built four-port instrument goes through luseepy (`InstrumentResponse` -> `lusee.Covariance`); the symmetric pseudo-dipole arm goes through a thin `croissant.PairStokesBeam` driver. Both share one conventions module, one sky, one contraction.

**Tech Stack:** Python 3.12, numpy, jax (x64), croissant-sim (editable, `/home/christian/Documents/projects/croissant-main`), luseepy (editable, `../luseepy`), healpy, lunarsky, astropy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-luseepy-croissant-refactor-design.md` — read it before Task 1. It carries the audit findings that justify every design choice here.

## Global Constraints

- Black formatting, **line length 79**. Run `uv run black src/ tests/ scripts/` before every commit.
- flake8 clean: `uv run flake8 src/`.
- Package installs use `uv add`, never `uv pip install`.
- **NEVER modify the luseepy or croissant checkouts.** "Use luseepy infrastructure" means *import* from `lusee`. All code, tests and scripts live in this repo.
- `export JAX_ENABLE_X64=1` (or `jax.config.update("jax_enable_x64", True)`) before importing any croissant/jax path. Every test module that touches croissant must do this at import time.
- Sky Q/U maps are healpy/**COSMO** convention on input. croissant consumes **IAU** (`U_IAU = -U_COSMO`). All conversion goes through `conventions.py`.
- Faraday rotation, pinned: `(Q + iU)_COSMO -> (Q + iU)_COSMO * exp(+2i * phi_FD * lambda^2)`.
- Port order `0,1,2,3 = N,E,S,W`. 16 real channels ordered as `lusee.Covariance.default_product_labels()`.
- Response frame: `x = East, y = North, z = zenith` (proper rotation); grid `phi = 90deg - astronomical azimuth`.
- Fixed-beam approximation: the response is evaluated at **one native channel** (30/10/50 MHz). Only the Faraday phase is chromatic, so all delay-space power is Faraday-induced. This must be *asserted*, never implied by an interpolation default.
- Real sky maps stay at native `nside=512` RING. Never `ud_grade` a map that a per-pixel Faraday phase will be applied to.
- Heavy jobs run in background under `ulimit -v 16000000`, logs written to `generated_data/` with **absolute** paths.
- Run tests with `uv run pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/lusee_faraday/conventions.py` | NEW. COSMO<->IAU, Faraday phase, dual-block coefficients, port pairs, product labels, constants. |
| `src/lusee_faraday/config.py` | NEW. Site, time grid, fine frequency grid, band centers, sky spectral parameters. Absorbs `scripts/common.py`. |
| `src/lusee_faraday/response.py` | NEW. Pair-Stokes from Jones; four-port alms at one native channel; two-port `croissant.PairStokesBeam` arm. |
| `src/lusee_faraday/engine.py` | NEW. Block-resolved harmonic contraction and the spectral expansion. |
| `src/lusee_faraday/sky.py` | REWRITTEN. `FaradaySky`: component decomposition, coefficients, audit diagnostics. |
| `src/lusee_faraday/instrument.py` | NEW. luseepy covariance assembly: open covariance -> loading -> hermitian -> 16 packed channels. |
| `src/lusee_faraday/polarimeter.py` | NEW. Zenith calibration (gains + Loewdin ortho) and pseudo-Stokes. |
| `src/lusee_faraday/channelization.py` | NEW. Parent/zoom/ideal bin weights on `lusee.spectrometer_response*`, integration, zoom frequency grid. |
| `src/lusee_faraday/pixel_arm.py` | MOVED from `fourport.py`. Validation arm only; never imported by production code. |
| `src/lusee_faraday/__init__.py` | MODIFIED. New public surface. |
| DELETED | `beam.py`, `sim.py`, `fast_sim.py`, `healpix.py`, `rotations.py`, `spectrometer.py`, `utils.py`, and their tests. |

---

### Task 1: Phase-0 prep — branch, toolchain probe, regression fixture

**Files:**
- Create: `scripts/probe_toolchain.py`
- Create: `tests/fixtures/regression_baselines.json`
- Create: `tests/test_regression_fixture.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/fixtures/regression_baselines.json`, a dict of published numbers later tasks assert against.

**Context:** `generated_data/` is empty on disk (gitignored, never repopulated), so baselines come from the committed `report/report.tex` and `PROGRESS.md`, not from stored npz. The spec's Phase-0 wording assumed the npz were present; they are not.

- [ ] **Step 1: Create the working branch**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
git checkout croissant-crosscheck
git checkout -b luseepy-refactor
```

- [ ] **Step 2: Verify the croissant worktree is present**

```bash
cat .venv/lib/python3.12/site-packages/__editable__.croissant_sim-*.pth
ls -d /home/christian/Documents/projects/croissant-main
```

Expected: the `.pth` points at `/home/christian/Documents/projects/croissant-main` and that directory exists. If it is missing, restore it from the croissant repo with `git worktree add --detach /home/christian/Documents/projects/croissant-main main`, then `uv sync`.

- [ ] **Step 3: Write the toolchain probe**

This confirms the spec's claim that croissant `1c4d6c5` no longer routes a low-`lmax`, high-`nside` polarized transform through the dense engine.

```python
"""Confirm croissant's engine resolution at the resolutions we use.

The spec claims croissant 1c4d6c5 added a memory cap to
_low_pass_in_one_step, so PolarizedSky(nside=512).compute_alm(lmax=30)
takes the native transform and truncates instead of building an ~800 GB
dense operator.  Verify rather than trust.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import croissant as cro
    import healpy as hp

    nside = 512
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1, 4, npix)) * 1e-3
    sky = cro.PolarizedSky(data, np.array([30.0]), sampling="healpix")
    print("resolved engines:", sky.engine)
    print("reasons:", sky.engine_reason)
    alm = np.asarray(sky.compute_alm(lmax=30))
    print("alm shape:", alm.shape)
    assert alm.shape == (1, 4, 31, 61), alm.shape
    assert "dense" not in set(sky.engine.values()), sky.engine
    print("OK: no dense engine at nside=512, lmax=30")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the probe**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
ulimit -v 16000000
uv run python scripts/probe_toolchain.py
```

Expected: prints the resolved engines and `OK: no dense engine at nside=512, lmax=30`.

If it instead resolves to `dense` or gets OOM-killed, **stop and report**: the spec's section 3 claim is wrong and the plan needs an explicit `lmax = 3*nside - 1` + manual truncation step threaded through Tasks 4, 8 and 9.

- [ ] **Step 5: Write the regression baseline fixture**

```bash
mkdir -p tests/fixtures
```


Every number is quoted from `report/report.tex` or `PROGRESS.md`; the `source` field records which.

```json
{
  "_comment": "Published numbers the refactor must reproduce. Sources are report/report.tex and PROGRESS.md on branch croissant-crosscheck at commit afb3290. generated_data/ was empty, so these are the committed values, not re-derived.",
  "point_source_phi_fd": 250.0,
  "band_centers_mhz": [30.0, 10.0, 50.0],
  "q_oscillation_period_khz": {
    "value": 1.89,
    "rtol": 0.01,
    "source": "PROGRESS.md Step 1"
  },
  "unpolarized_transit_leakage_raw": {
    "value": 0.134,
    "rtol": 0.02,
    "source": "report.tex sec Step 0"
  },
  "unpolarized_transit_leakage_gains": {
    "value": 0.096,
    "rtol": 0.05,
    "source": "PROGRESS.md zenith-calibrated polarimeter"
  },
  "unpolarized_transit_leakage_ortho": {
    "value": 7e-4,
    "rtol": 0.5,
    "source": "PROGRESS.md zenith-calibrated polarimeter"
  },
  "zenith_null_ortho_max": {
    "value": 1e-15,
    "source": "PROGRESS.md: ortho nulls zenith Q=U=V to ~1e-16"
  },
  "zoom_recovery_real": {
    "value": 0.79,
    "rtol": 0.03,
    "source": "report.tex line 235"
  },
  "zoom_recovery_ideal": {
    "value": 0.86,
    "rtol": 0.03,
    "source": "report.tex line 235"
  },
  "parent_stokes_over_i_30mhz": {
    "q": 0.146,
    "u": -0.032,
    "atol": 2e-4,
    "source": "report.tex line 425"
  }
}
```

- [ ] **Step 6: Write a test that the fixture loads and is complete**

```python
import json
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "regression_baselines.json"
)

REQUIRED = (
    "point_source_phi_fd",
    "band_centers_mhz",
    "q_oscillation_period_khz",
    "unpolarized_transit_leakage_raw",
    "unpolarized_transit_leakage_ortho",
    "zenith_null_ortho_max",
    "zoom_recovery_real",
    "zoom_recovery_ideal",
    "parent_stokes_over_i_30mhz",
)


def test_fixture_has_every_required_baseline():
    data = json.loads(FIXTURE.read_text())
    missing = [key for key in REQUIRED if key not in data]
    assert missing == [], missing


def test_every_baseline_records_its_source():
    data = json.loads(FIXTURE.read_text())
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        assert "source" in value, key
```

- [ ] **Step 7: Run the test**

```bash
uv run pytest tests/test_regression_fixture.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
uv run black scripts/probe_toolchain.py tests/test_regression_fixture.py
git add scripts/probe_toolchain.py tests/fixtures/regression_baselines.json tests/test_regression_fixture.py
git commit -m "Pin the published numbers the refactor has to reproduce"
```

---

### Task 2: `conventions.py`

**Files:**
- Create: `src/lusee_faraday/conventions.py`
- Test: `tests/test_conventions.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PORT_NAMES: tuple[str, ...]` = `("N", "E", "S", "W")`
  - `PORT_PAIRS: tuple[tuple[int, int], ...]` — the ten `a <= b` pairs
  - `PRODUCT_LABELS: tuple[str, ...]` — 16 labels
  - `DUAL_BLOCKS: tuple[str, ...]` = `("I", "V", "P_MINUS", "P_PLUS")`
  - `lambda_squared(freq_mhz) -> np.ndarray` (m^2)
  - `cosmo_to_iau_qu(Q, U) -> tuple` and `iau_to_cosmo_qu(Q, U) -> tuple`
  - `faraday_phase_cosmo(phi_fd, freq_mhz) -> np.ndarray` — the factor multiplying `(Q + iU)_COSMO`, shape `(..., nfreq)`
  - `dual_block_phase(phi_fd, freq_mhz) -> np.ndarray` — shape `(..., nfreq, 4)`, the per-block Faraday coefficients in `DUAL_BLOCKS` order

**Why this task is alone:** the sign conventions are the single highest-risk item in the refactor. Nothing else happens in this phase.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from lusee_faraday import conventions as cv


def test_product_labels_match_luseepy():
    lusee_cov = pytest.importorskip("lusee.Covariance")

    assert cv.PRODUCT_LABELS == lusee_cov.default_product_labels()
    assert len(cv.PRODUCT_LABELS) == 16
    assert cv.PORT_PAIRS == tuple(
        (a, b) for a in range(4) for b in range(a, 4)
    )


def test_qu_convention_roundtrip_is_identity():
    rng = np.random.default_rng(0)
    Q, U = rng.normal(size=5), rng.normal(size=5)
    Q2, U2 = cv.iau_to_cosmo_qu(*cv.cosmo_to_iau_qu(Q, U))
    assert np.allclose(Q2, Q)
    assert np.allclose(U2, U)


def test_lambda_squared_at_30_mhz():
    # c / 30 MHz = 9.9931 m, so lambda^2 is just under 100 m^2.
    assert np.isclose(cv.lambda_squared(30.0)[0], 99.8617, rtol=1e-4)


def test_faraday_phase_matches_explicit_rotation():
    """The COSMO phase must reproduce an explicit (Q, U) rotation."""
    phi, freq = 250.0, np.array([30.0])
    Q, U = 0.3, -0.7
    lam2 = cv.lambda_squared(freq)
    angle = 2 * phi * lam2
    Q_rot = Q * np.cos(angle) - U * np.sin(angle)
    U_rot = Q * np.sin(angle) + U * np.cos(angle)
    got = (Q + 1j * U) * cv.faraday_phase_cosmo(phi, freq)
    assert np.allclose(got.real, Q_rot)
    assert np.allclose(got.imag, U_rot)


def test_dual_block_phase_is_conjugate_on_p_blocks():
    """P- carries exp(-2i phi lam^2) because IAU flips U."""
    phi, freq = 250.0, np.array([30.0, 10.0])
    blocks = cv.dual_block_phase(phi, freq)
    assert blocks.shape == (2, 4)
    assert np.allclose(blocks[:, 0], 1.0)  # I
    assert np.allclose(blocks[:, 1], 1.0)  # V
    assert np.allclose(blocks[:, 2], np.conj(blocks[:, 3]))
    cosmo = cv.faraday_phase_cosmo(phi, freq)
    assert np.allclose(blocks[:, 3], cosmo)      # P+ == COSMO phase
    assert np.allclose(blocks[:, 2], np.conj(cosmo))


def test_dual_block_phase_agrees_with_rotating_maps_first():
    """Rotate (Q, U) then convert to IAU == convert then apply P blocks."""
    rng = np.random.default_rng(3)
    phi = 137.0
    freq = np.array([29.9, 30.0, 30.1])
    Q, U = rng.normal(size=8), rng.normal(size=8)

    rotated = (Q + 1j * U)[:, None] * cv.faraday_phase_cosmo(phi, freq)
    Q_rot, U_rot = rotated.real, rotated.imag
    Q_iau, U_iau = cv.cosmo_to_iau_qu(Q_rot, U_rot)
    p_minus_direct = Q_iau + 1j * U_iau

    Q0_iau, U0_iau = cv.cosmo_to_iau_qu(Q, U)
    blocks = cv.dual_block_phase(phi, freq)
    p_minus_via_blocks = (Q0_iau + 1j * U0_iau)[:, None] * blocks[None, :, 2]

    assert np.allclose(p_minus_direct, p_minus_via_blocks)


def test_dual_block_phase_broadcasts_over_regions():
    phi = np.array([0.0, 100.0, -250.0])
    freq = np.array([30.0, 30.1])
    assert cv.dual_block_phase(phi, freq).shape == (3, 2, 4)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_conventions.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'lusee_faraday.conventions'`.

- [ ] **Step 3: Write the implementation**

```python
"""Pinned conventions for the LuSEE Faraday simulation.

Every convention conversion in the package funnels through this module.
The two that matter:

- Sky Q/U maps are healpy/COSMO on input; croissant consumes IAU, and
  ``U_IAU = -U_COSMO``.
- Faraday rotation is ``(Q + iU)_COSMO -> (Q + iU)_COSMO e^{+2i phi l^2}``.

Combining them gives the result the whole refactor rests on.  croissant's
harmonic dual holds ``P_MINUS`` = the spin -2 analysis of ``Q + iU`` and
``P_PLUS`` = the spin +2 analysis of ``Q - iU``, both in IAU.  Since
``(Q + iU)_IAU = conj((Q + iU)_COSMO)`` for real maps, Faraday rotation
multiplies the P_MINUS block by ``e^{-2i phi l^2}`` and the P_PLUS block
by its conjugate -- it is diagonal in the dual, so a region of constant
Faraday depth needs one component and a per-block coefficient.
"""

import numpy as np
from scipy.constants import c as C_LIGHT

PORT_NAMES = ("N", "E", "S", "W")
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
DUAL_BLOCKS = ("I", "V", "P_MINUS", "P_PLUS")


def _product_labels():
    labels = []
    for a, b in PORT_PAIRS:
        if a == b:
            labels.append(f"{a}{b}R")
        else:
            labels.extend((f"{a}{b}R", f"{a}{b}I"))
    return tuple(labels)


PRODUCT_LABELS = _product_labels()


def lambda_squared(freq_mhz):
    """Wavelength squared in m^2 for frequencies in MHz."""
    return (C_LIGHT / (np.atleast_1d(np.asarray(freq_mhz, dtype=float))
                       * 1e6)) ** 2


def cosmo_to_iau_qu(Q, U):
    """healpy/COSMO (Q, U) -> IAU (Q, U)."""
    return np.asarray(Q), -np.asarray(U)


def iau_to_cosmo_qu(Q, U):
    """IAU (Q, U) -> healpy/COSMO (Q, U)."""
    return np.asarray(Q), -np.asarray(U)


def faraday_phase_cosmo(phi_fd, freq_mhz):
    """Factor multiplying ``(Q + iU)_COSMO``; shape ``(..., nfreq)``."""
    phi = np.asarray(phi_fd, dtype=float)
    lam2 = lambda_squared(freq_mhz)
    return np.exp(2j * phi[..., None] * lam2)


def dual_block_phase(phi_fd, freq_mhz):
    """Per-dual-block Faraday coefficients; shape ``(..., nfreq, 4)``.

    Blocks are ordered as :data:`DUAL_BLOCKS`.  The spin-0 blocks are
    untouched; ``P_MINUS`` picks up the conjugate of the COSMO phase and
    ``P_PLUS`` the phase itself.
    """
    cosmo = faraday_phase_cosmo(phi_fd, freq_mhz)
    ones = np.ones_like(cosmo)
    return np.stack([ones, ones, np.conj(cosmo), cosmo], axis=-1)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_conventions.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/conventions.py tests/test_conventions.py
uv run flake8 src/lusee_faraday/conventions.py
git add src/lusee_faraday/conventions.py tests/test_conventions.py
git commit -m "Add the conventions module: COSMO/IAU and the Faraday dual-block phase"
```

---

### Task 3: `config.py`

**Files:**
- Create: `src/lusee_faraday/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `conventions.lambda_squared` (for `lam2`).
- Produces:
  - constants `LUN_LAT_DEG`, `LUN_LONG_DEG`, `N_TIMES`, `SIDEREAL_DAY_S`, `T_START_UTC`, `FINE_STEP_MHZ`, `N_FINE`, `MAP_NSIDE`, `BAND_CENTERS_MHZ`, `BETA_I`, `FREQ_REF_I`, `BETA_QU`, `FREQ_REF_QU`, `T_CMB`, `PHI_FD_POINT`
  - `times() -> astropy Time array` (length `N_TIMES`)
  - `moon_location() -> lunarsky.MoonLocation`
  - `fine_freqs(center_mhz) -> np.ndarray` (length `N_FINE`)
  - `parent_centers(center_mhz) -> np.ndarray` (length 3)

**Context:** these values already exist in `scripts/common.py` and are pinned in `AGENTS.md`. This task moves them into the package so scripts stop owning configuration. Do not change any value.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from lusee_faraday import config as cfg


def test_time_grid_spans_exactly_one_lunar_sidereal_day():
    t = cfg.times()
    assert len(t) == cfg.N_TIMES == 1024
    span = (t[-1] - t[0]).sec
    step = cfg.SIDEREAL_DAY_S / cfg.N_TIMES
    assert np.isclose(span, cfg.SIDEREAL_DAY_S - step, rtol=1e-12)


def test_fine_grid_is_uniform_and_covers_the_three_parents():
    center = 30.0
    f = cfg.fine_freqs(center)
    assert f.size == cfg.N_FINE == 16384
    d = np.diff(f)
    assert np.allclose(d, d[0], rtol=1e-12)
    assert np.isclose(d[0], 25e-3 / 2048)
    parents = cfg.parent_centers(center)
    assert np.allclose(parents, [29.975, 30.0, 30.025])
    # every parent's +-50 kHz response support sits inside the fine grid
    assert f.min() <= parents.min() - 0.05
    assert f.max() >= parents.max() + 0.05


def test_site_matches_luseepy_observation_defaults():
    import pytest

    obs_mod = pytest.importorskip("lusee.Observation")
    obs = obs_mod.Observation()
    assert np.isclose(cfg.LUN_LAT_DEG, obs.default_lun_lat_deg)
    assert np.isclose(cfg.LUN_LONG_DEG, obs.default_lun_long_deg)


def test_band_centers_are_the_three_studied_bands():
    assert cfg.BAND_CENTERS_MHZ == (30.0, 10.0, 50.0)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.config'`.

- [ ] **Step 3: Write the implementation**

```python
"""Pinned configuration for the LuSEE Faraday analysis.

These values were previously duplicated in ``scripts/common.py``.  They
are documented in ``AGENTS.md`` under "Pinned conventions"; do not change
one without changing that document.
"""

import numpy as np

from .conventions import lambda_squared

# LuSEE-Night landing site == lusee.Observation defaults
LUN_LAT_DEG = -23.814
LUN_LONG_DEG = 182.258

# 1024 samples over exactly one lunar sidereal day: the time axis is
# periodic, so the observation-time FFT needs no window.
N_TIMES = 1024
SIDEREAL_DAY_S = 27.321661 * 86400.0
T_START_UTC = "2027-01-01 09:00:00"

# Fine frequency grid: +-4 parent bins around the center at 25 kHz / 2048.
FINE_STEP_MHZ = 25e-3 / 2048
N_FINE = 16384

MAP_NSIDE = 512
BAND_CENTERS_MHZ = (30.0, 10.0, 50.0)

# Sky spectral parameters (as in the paper)
BETA_I = -2.55
FREQ_REF_I = 408.0  # MHz, Haslam
BETA_QU = -2.8
FREQ_REF_QU = 23e3  # MHz, WMAP K band
T_CMB = 2.7255

# Faraday depth of the single-source toy example (paper value)
PHI_FD_POINT = 250.0  # rad/m^2


def times():
    """``N_TIMES`` astropy Times covering one lunar sidereal day."""
    import astropy.units as u
    from lunarsky.time import Time

    t0 = Time(T_START_UTC)
    dt = SIDEREAL_DAY_S / N_TIMES
    return t0 + np.arange(N_TIMES) * dt * u.s


def moon_location():
    """The LuSEE-Night landing site."""
    from lunarsky import MoonLocation

    return MoonLocation.from_selenodetic(
        lon=LUN_LONG_DEG, lat=LUN_LAT_DEG, height=0.0
    )


def fine_freqs(center_mhz):
    """``N_FINE`` fine frequencies (MHz) spanning +-0.1 MHz around center."""
    k = np.arange(N_FINE) - N_FINE // 2
    return center_mhz + k * FINE_STEP_MHZ


def parent_centers(center_mhz):
    """The three parent 25 kHz bins fully covered by the fine grid."""
    return np.array([center_mhz - 0.025, center_mhz, center_mhz + 0.025])


def lam2(freq_mhz):
    """Convenience alias for :func:`conventions.lambda_squared`."""
    return lambda_squared(freq_mhz)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/config.py tests/test_config.py
uv run flake8 src/lusee_faraday/config.py
git add src/lusee_faraday/config.py tests/test_config.py
git commit -m "Move the pinned analysis configuration into the package"
```

---

### Task 4: `response.py` — pair-Stokes from Jones, and four-port alms

**Files:**
- Create: `src/lusee_faraday/response.py`
- Test: `tests/test_response.py`

**Interfaces:**
- Consumes: `conventions.PORT_PAIRS`.
- Produces:
  - `load_response(path) -> lusee.InstrumentResponse` — fast loader, skips slow revalidation
  - `native_channel_index(resp, freq_mhz) -> int` — raises unless the frequency is a native channel
  - `pair_stokes_from_jones(h_theta, h_phi, pairs) -> np.ndarray` shape `(npair, 4, ...)`
  - `four_port_pair_alms(resp, freq_mhz, lmax) -> np.ndarray` shape `(10, 4, lmax+1, 2*lmax+1)`, complex, physical units (m^2 scaled by `eta0 / lambda^2`)

**Context:** luseepy's `InstrumentResponse.pair_stokes_maps` defines the pair-Stokes kernel as

```
I = at*conj(bt) + ap*conj(bp)
Q = at*conj(bt) - ap*conj(bp)
U = at*conj(bp) + ap*conj(bt)
V = 1j*(ap*conj(bt) - at*conj(bp))
```

with **no** factor of one half. `pair_stokes_from_jones` must reproduce that exactly — the test pins it against luseepy directly, so the two-port arm in Task 14 inherits a convention that is already validated.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import response as rsp  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    return lusee.synthetic_four_port_response(freq_mhz=(10.0, 20.0))


def test_pair_stokes_from_jones_matches_luseepy(synthetic):
    """Our kernel formula must be luseepy's, not a near miss."""
    want = np.asarray(synthetic.all_pair_stokes_maps())
    got = rsp.pair_stokes_from_jones(
        np.asarray(synthetic.H_theta),
        np.asarray(synthetic.H_phi),
        synthetic.pairs,
    )
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_native_channel_index_rejects_off_grid_frequencies(synthetic):
    assert rsp.native_channel_index(synthetic, 20.0) == 1
    with pytest.raises(ValueError, match="native response channel"):
        rsp.native_channel_index(synthetic, 20.01)


def test_four_port_pair_alms_shape_and_pairs(synthetic):
    lmax = 8
    alms = rsp.four_port_pair_alms(synthetic, 10.0, lmax)
    assert alms.shape == (10, 4, lmax + 1, 2 * lmax + 1)
    assert np.iscomplexobj(alms)


def test_four_port_pair_alms_is_the_fixed_beam_slice(synthetic):
    """Two native channels must give different alms; picking one is a choice."""
    a10 = rsp.four_port_pair_alms(synthetic, 10.0, 4)
    a20 = rsp.four_port_pair_alms(synthetic, 20.0, 4)
    assert not np.allclose(a10, a20)


def test_four_port_pair_alms_rejects_non_native_frequency(synthetic):
    with pytest.raises(ValueError, match="native response channel"):
        rsp.four_port_pair_alms(synthetic, 15.0, 4)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_response.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.response'`.

- [ ] **Step 3: Write the implementation**

```python
"""Adapters from instrument models to harmonic pair-Stokes responses.

Two arms share this module:

- the as-built four-port instrument, read from a BGL_v16 response
  artifact through ``lusee.InstrumentResponse``;
- the symmetric pseudo-dipoles of the paper's Fig 4, built from a 2-port
  Jones FITS file (added in a later task).

Both end up as complex pair-Stokes alms in croissant's harmonic dual, so
the contraction in :mod:`lusee_faraday.engine` does not know which arm it
is serving.
"""

import numpy as np

from .conventions import PORT_PAIRS


def load_response(path):
    """Load a v3 response artifact without re-running slow validation."""
    import fitsio
    from lusee.InstrumentResponse import InstrumentResponse

    with fitsio.FITS(str(path)) as f:
        header = dict(f[0].read_header())

        def cplx(name):
            return f[f"{name}_REAL"].read() + 1j * f[f"{name}_IMAG"].read()

        return InstrumentResponse.from_arrays(
            f["FREQ"].read(),
            f["THETA"].read(),
            f["PHI"].read(),
            cplx("HTHETA"),
            cplx("HPHI"),
            cplx("ZA"),
            cplx("RSKY"),
            cplx("RMOON"),
            cplx("RLOSS"),
            validated=False,
            metadata={**header, "VALIDATED": False},
            ZLoad=cplx("ZLOAD"),
        )


def native_channel_index(resp, freq_mhz):
    """Index of ``freq_mhz`` in the response's native grid.

    The fixed-beam approximation is an assertion, not a default: an
    off-grid frequency would be silently interpolated by luseepy's
    ``FrequencyMap``, smearing the beam across the band and putting
    non-Faraday structure into delay space.
    """
    freq = np.asarray(resp.freq, dtype=float)
    idx = int(np.argmin(np.abs(freq - freq_mhz)))
    if abs(freq[idx] - freq_mhz) > 1e-9:
        raise ValueError(
            f"{freq_mhz} MHz is not a native response channel; "
            f"nearest is {freq[idx]} MHz."
        )
    return idx


def pair_stokes_from_jones(h_theta, h_phi, pairs=PORT_PAIRS):
    """Complex bare pair-Stokes maps from Jones components.

    Mirrors ``lusee.InstrumentResponse.pair_stokes_maps`` exactly, so the
    two-port arm inherits a convention already validated against luseepy.
    Input arrays are indexed ``(port, ...)``; the output is
    ``(pair, 4, ...)`` with the 4 axis in I, Q, U, V order.
    """
    at_all = np.asarray(h_theta)
    ap_all = np.asarray(h_phi)
    out = []
    for a, b in pairs:
        at, ap = at_all[a], ap_all[a]
        bt, bp = np.conj(at_all[b]), np.conj(ap_all[b])
        out.append(
            np.stack(
                [
                    at * bt + ap * bp,
                    at * bt - ap * bp,
                    at * bp + ap * bt,
                    1j * (ap * bt - at * bp),
                ],
                axis=0,
            )
        )
    return np.stack(out, axis=0)


def four_port_pair_alms(resp, freq_mhz, lmax):
    """Physical pair-Stokes alms at ONE native channel -> (10, 4, L, 2L-1).

    luseepy applies the ``eta0 / lambda^2`` scaling that turns bare m^2
    maps into the physical W kernel, so the result is directly
    contractable with a sky in kelvin.
    """
    idx = native_channel_index(resp, freq_mhz)
    freq = np.asarray(resp.freq, dtype=float)
    alms, _ = resp.pair_stokes_alms(int(lmax), np.array([freq[idx]]))
    return np.asarray(alms)[:, 0]
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_response.py -v
```

Expected: 5 passed.

If `test_pair_stokes_from_jones_matches_luseepy` fails on the V component only, the sign of `1j*(ap*conj(bt) - at*conj(bp))` is the thing to check against `luseepy/lusee/InstrumentResponse.py:621` — copy it, do not re-derive it.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/response.py tests/test_response.py
uv run flake8 src/lusee_faraday/response.py
git add src/lusee_faraday/response.py tests/test_response.py
git commit -m "Add the four-port response adapter with a fixed-beam assertion"
```

---

### Task 5: `engine.py` — the block-resolved contraction

**Files:**
- Create: `src/lusee_faraday/engine.py`
- Test: `tests/test_engine_contract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `contract_blocks(beam_alm, sky_alm, phases) -> np.ndarray` shape `(K, 4, ntime, npair)`, where `beam_alm` is `(npair, 4, L, 2L-1)`, `sky_alm` is `(K, 4, L, 2L-1)`, `phases` is `(ntime, 2L-1)`.

**Context — why this is not just a call to croissant:** `croissant.polarized_convolve` contracts `einsum("fclm,tm,pfclm->tpf")`, summing over the dual-block axis `c`. Our coefficients differ per block (the `I` block carries the Stokes-I power law, the `P` blocks carry the polarized power law times conjugate Faraday phases), so summing `c` before applying them would be wrong. `contract_blocks` performs the identical contraction with `c` retained. This is the only place the refactor does not call a croissant entry point, so it carries a test that summing our `c` axis reproduces `polarized_convolve` to machine precision.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine  # noqa: E402

LMAX = 5
L = LMAX + 1
M = 2 * LMAX + 1


def random_alm(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_shapes():
    rng = np.random.default_rng(0)
    beam = random_alm(rng, (10, 4, L, M))
    sky = random_alm(rng, (3, 4, L, M))
    phases = random_alm(rng, (7, M))
    W = engine.contract_blocks(beam, sky, phases)
    assert W.shape == (3, 4, 7, 10)


def test_block_sum_reproduces_croissant_polarized_convolve():
    """The one contract we own must agree with the library's."""
    cro = pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(1)
    n_components = 3
    beam = random_alm(rng, (10, 4, L, M))
    sky = random_alm(rng, (n_components, 4, L, M))
    phases = random_alm(rng, (7, M))

    ours = engine.contract_blocks(beam, sky, phases).sum(axis=1)

    # croissant pairs sky frequency f with beam frequency f, so tile the
    # single-frequency beam across our component axis.
    beam_tiled = np.broadcast_to(
        beam[:, None], (10, n_components, 4, L, M)
    ).copy()
    theirs = np.asarray(
        cro.polarized_convolve(beam_tiled, sky, phases)
    )  # (t, p, f)

    assert np.allclose(
        ours, np.transpose(theirs, (2, 0, 1)), rtol=1e-12, atol=1e-12
    )


def test_contraction_is_linear_in_the_sky():
    rng = np.random.default_rng(2)
    beam = random_alm(rng, (10, 4, L, M))
    a = random_alm(rng, (1, 4, L, M))
    b = random_alm(rng, (1, 4, L, M))
    phases = random_alm(rng, (4, M))
    both = engine.contract_blocks(beam, np.concatenate([a, b]), phases)
    combined = engine.contract_blocks(beam, 2.0 * a + 3.0 * b, phases)
    assert np.allclose(combined[0], 2.0 * both[0] + 3.0 * both[1])
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_engine_contract.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.engine'`.

- [ ] **Step 3: Write the implementation**

```python
"""Harmonic contraction and spectral expansion.

The refactor rests on one separation.  Faraday rotation is diagonal in
croissant's harmonic dual, so a sky is a small set of frequency-
independent component alms plus a per-frequency, per-block coefficient
matrix, and

    V(t, p, nu) = sum_k sum_c coeff[k, nu, c] * W[k, c, t, p]

The expensive part is ``W``: one contraction per component, independent
of how many frequency channels are wanted.  The 16,384-channel fine grid
is then a single einsum.
"""

import numpy as np


def contract_blocks(beam_alm, sky_alm, phases):
    """Contract sky and pair-response alms, keeping the dual-block axis.

    This is ``croissant.polarized_convolve`` with the block axis ``c``
    retained instead of summed, because a Faraday sky needs a different
    coefficient per block.  Summing the returned ``c`` axis reproduces
    ``polarized_convolve`` exactly (see the test).

    Parameters
    ----------
    beam_alm : (npair, 4, L, 2L-1) complex
        Pair-response alms at one frequency, already in the frame the
        contraction happens in.
    sky_alm : (K, 4, L, 2L-1) complex
        Component alms in the same frame.
    phases : (ntime, 2L-1) complex
        croissant's ``exp(-i m phi)`` time phases.

    Returns
    -------
    (K, 4, ntime, npair) complex
    """
    return np.einsum(
        "kclm,tm,pclm->kctp",
        np.conj(np.asarray(sky_alm)),
        np.asarray(phases),
        np.asarray(beam_alm),
        optimize=True,
    )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_engine_contract.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/engine.py tests/test_engine_contract.py
uv run flake8 src/lusee_faraday/engine.py
git add src/lusee_faraday/engine.py tests/test_engine_contract.py
git commit -m "Add the block-resolved contraction, pinned against polarized_convolve"
```

---

### Task 6: `engine.py` — frame rotation and the chunked spectral expansion

**Files:**
- Modify: `src/lusee_faraday/engine.py`
- Test: `tests/test_engine_expand.py`

**Interfaces:**
- Consumes: `engine.contract_blocks` from Task 5.
- Produces:
  - `contract(pair_alms, component_alms, times, loc, lmax, sky_frame="galactic") -> np.ndarray` shape `(K, 4, ntime, npair)`
  - `expand(W, coeffs, chunk=None, out=None) -> np.ndarray` shape `(ntime, nfreq, npair)`

**Context:** `contract` mirrors `lusee.FullStokesCroSimulator._convolve` (`luseepy/lusee/FullStokesSimulator.py:730`): rotate the beam alms from topo into MEPA at the first timestamp, rotate the sky from galactic into MEPA, then apply `croissant.simulator.rot_alm_z` phases for the elapsed time. Copy that structure; do not invent a rotation.

`expand` is the outer product that makes the fine frequency grid cheap. It must chunk over frequency and support writing into a preallocated (optionally memmapped) array, because a full run is `1024 x 16384 x 10` complex = 2.7 GB.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine  # noqa: E402

LMAX = 4
L = LMAX + 1
M = 2 * LMAX + 1


def random_alm(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_expand_matches_the_reference_einsum():
    rng = np.random.default_rng(0)
    W = random_alm(rng, (3, 4, 6, 10))
    coeffs = random_alm(rng, (3, 11, 4))
    want = np.einsum("kctp,kfc->tfp", W, coeffs)
    assert np.allclose(engine.expand(W, coeffs), want)


def test_expand_is_chunk_invariant():
    rng = np.random.default_rng(1)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (2, 17, 4))
    full = engine.expand(W, coeffs)
    for chunk in (1, 4, 16, 64):
        assert np.allclose(engine.expand(W, coeffs, chunk=chunk), full)


def test_expand_writes_into_a_preallocated_output(tmp_path):
    rng = np.random.default_rng(2)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (2, 9, 4))
    path = tmp_path / "out.dat"
    out = np.memmap(path, dtype=np.complex128, mode="w+", shape=(5, 9, 10))
    engine.expand(W, coeffs, chunk=3, out=out)
    out.flush()
    assert np.allclose(np.asarray(out), engine.expand(W, coeffs))


def test_expand_rejects_mismatched_component_counts():
    rng = np.random.default_rng(3)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (3, 9, 4))
    with pytest.raises(ValueError, match="component"):
        engine.expand(W, coeffs)


def test_contract_of_an_isotropic_sky_is_time_independent():
    """A monopole sky is rotation invariant, so W must not vary with time."""
    pytest.importorskip("croissant")
    pytest.importorskip("lunarsky")
    import jax

    jax.config.update("jax_enable_x64", True)

    from lusee_faraday import config as cfg

    rng = np.random.default_rng(4)
    beam = random_alm(rng, (10, 4, L, M))
    sky = np.zeros((1, 4, L, M), dtype=complex)
    sky[0, 0, 0, LMAX] = 1.0  # I monopole only, m = 0

    times = cfg.times()[:4]
    W = engine.contract(beam, sky, times, cfg.moon_location(), LMAX)
    assert W.shape == (1, 4, 4, 10)
    spread = np.abs(W - W[:, :, :1]).max()
    assert spread < 1e-10 * np.abs(W).max()
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_engine_expand.py -v
```

Expected: `AttributeError: module 'lusee_faraday.engine' has no attribute 'expand'`.

- [ ] **Step 3: Append the implementation to `engine.py`**

```python
def contract(pair_alms, component_alms, times, loc, lmax, sky_frame="galactic"):
    """Rotate into a common frame and contract every component.

    Mirrors ``lusee.FullStokesCroSimulator._convolve``: the response is
    rotated from topocentric into MEPA at the first timestamp, the sky is
    rotated from its own frame into MEPA, and the remaining time
    dependence is the diagonal-in-m ``rot_alm_z`` phase.

    Parameters
    ----------
    pair_alms : (npair, 4, L, 2L-1) complex
        Response alms at one native channel, topocentric.
    component_alms : (K, 4, L, 2L-1) complex
        Frequency-independent sky components in ``sky_frame``.
    times : astropy Time array
    loc : lunarsky.MoonLocation
    lmax : int
    sky_frame : {"galactic", "mepa", "topo"}

    Returns
    -------
    (K, 4, ntime, npair) complex
    """
    import croissant as cro
    import numpy as np
    from lunarsky import LunarTopo

    beam = np.asarray(pair_alms)
    sky = np.asarray(component_alms)
    n_m = 2 * int(lmax) + 1

    if sky_frame == "topo":
        phases = np.ones((len(times), n_m), dtype=complex)
        return contract_blocks(beam, sky, phases)

    from lusee.spice_utils import ensure_lunarsky_moon_frame

    ensure_lunarsky_moon_frame()
    et = cro.rotations.jd_to_et(times[0].tdb.jd)
    topo = LunarTopo(obstime=times[0], location=loc)
    beam_rotation, beam_dl = cro.rotations.generate_euler_dl(
        int(lmax), topo, "mepa", et=et
    )
    beam_work = np.asarray(
        cro.rotations.rotate_alm(beam, beam_rotation, dl_array=beam_dl)
    )
    if sky_frame == "galactic":
        sky_work = np.asarray(cro.rotations.gal2mepa(sky, et=et))
    elif sky_frame == "mepa":
        sky_work = sky
    else:
        raise ValueError(f"unsupported sky frame {sky_frame!r}")

    elapsed = np.asarray(
        (times.tdb - times[0].tdb).to_value("s"), dtype=np.float64
    )
    phases = np.asarray(cro.simulator.rot_alm_z(int(lmax), times=elapsed))
    return contract_blocks(beam_work, sky_work, phases)


def expand(W, coeffs, chunk=None, out=None):
    """Apply per-frequency component coefficients to a contraction.

    ``V[t, f, p] = sum_k sum_c coeffs[k, f, c] * W[k, c, t, p]``.

    Chunked over frequency so a full run (1024 x 16384 x 10 complex,
    2.7 GB) can stream into a memmapped ``out``.
    """
    W = np.asarray(W)
    coeffs = np.asarray(coeffs)
    if W.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"component count mismatch: W has {W.shape[0]}, "
            f"coeffs has {coeffs.shape[0]}"
        )
    if W.shape[1] != coeffs.shape[2]:
        raise ValueError(
            f"dual-block count mismatch: W has {W.shape[1]}, "
            f"coeffs has {coeffs.shape[2]}"
        )
    ntime, npair, nfreq = W.shape[2], W.shape[3], coeffs.shape[1]
    if out is None:
        out = np.empty((ntime, nfreq, npair), dtype=complex)
    step = nfreq if chunk is None else int(chunk)
    for start in range(0, nfreq, step):
        stop = min(start + step, nfreq)
        out[:, start:stop] = np.einsum(
            "kctp,kfc->tfp",
            W,
            coeffs[:, start:stop],
            optimize=True,
        )
    return out
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_engine_expand.py -v
```

Expected: 5 passed. The isotropic-sky test is the meaningful one — if `W` varies with time for a monopole sky, the rotation path is wrong and every later result will be too.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/engine.py tests/test_engine_expand.py
uv run flake8 src/lusee_faraday/engine.py
git add src/lusee_faraday/engine.py tests/test_engine_expand.py
git commit -m "Add frame rotation and the chunked spectral expansion"
```

---

### Task 7: GATE — the contraction agrees with luseepy and with the pixel arm

**Files:**
- Create: `tests/test_engine_gate.py`
- Create: `scripts/crosscheck_pixel_arm.py`

**Interfaces:**
- Consumes: `response.four_port_pair_alms`, `engine.contract`.
- Produces: nothing importable. This task's deliverable is a verdict.

**Context:** this is the spec's first real gate. If the harmonic contraction does not reproduce luseepy's own convolution here, nothing built later will be right. Two independent checks:

1. **Fast, data-free.** Our `contract` versus `lusee.FullStokesCroSimulator._convolve` on a synthetic four-port response and a random band-limited sky. Same library, so agreement should be at round-off.
2. **Slow, real artifact.** Our harmonic path versus the existing pixel-space engine in `fourport.py` on the BGL_v16 response. Different quadrature, so the tolerance is the beam band-limit, not round-off.

**If check 1 fails, stop and report — do not proceed to Task 8.**

- [ ] **Step 1: Write the data-free gate test**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import config as cfg  # noqa: E402
from lusee_faraday import engine, response as rsp  # noqa: E402

LMAX = 8
NSIDE = 16


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    cro = pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    resp = lusee.synthetic_four_port_response(
        freq_mhz=(10.0, 20.0), angular_step_deg=5.0
    )
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    data = rng.normal(size=(1, 4, npix))
    data[0, 0] = np.abs(data[0, 0]) + 5.0  # keep Stokes I positive
    sky = cro.PolarizedSky(
        data, np.array([10.0]), sampling="healpix", coord="galactic"
    )
    return lusee, resp, sky


def test_contract_matches_luseepy_full_stokes_convolve(pieces):
    lusee, resp, sky = pieces
    from lusee.FullStokesSimulator import (
        _sky_frame,
        prepare_polarized_sky_alms,
    )
    from lusee.ReceiverImpedance import JFETReceiver

    obs = lusee.Observation()
    times = cfg.times()[:5]
    sim = lusee.FullStokesCroSimulator(
        obs,
        resp,
        sky,
        JFETReceiver(),
        freq=np.array([10.0]),
        lmax=LMAX,
    )
    pair_alms, *_ = sim.prepare_pair_alms(resp)
    sky_alms = prepare_polarized_sky_alms(sky, sim.freq, sim.lmax)
    theirs = np.asarray(
        sim._convolve(pair_alms, sky_alms, _sky_frame(sky), times)
    )[:, 0, :]  # (ntime, npair)

    beam = rsp.four_port_pair_alms(resp, 10.0, LMAX)
    components = np.asarray(sky.compute_alm(lmax=LMAX))
    W = engine.contract(beam, components, times, obs.loc, LMAX)
    ours = W.sum(axis=1)[0]  # (ntime, npair)

    scale = np.abs(theirs).max()
    assert np.abs(ours - theirs).max() < 1e-10 * scale
```

- [ ] **Step 2: Run the gate test**

```bash
uv run pytest tests/test_engine_gate.py -v
```

Expected: 1 passed.

Failure triage, in order: (a) is the beam alm slice the same object luseepy builds — compare `rsp.four_port_pair_alms(resp, 10.0, LMAX)` against `pair_alms[:, 0]` directly; (b) is the rotation right — the isotropic-sky test in Task 6 should already have caught that; (c) is the phase array the same — compare `cro.simulator.rot_alm_z` output against `sim.elapsed_tdb_seconds`.

- [ ] **Step 3: Write the pixel-arm cross-check script**

```python
"""Harmonic four-port path vs the pixel-space engine in fourport.py.

Two independent quadratures of the same integral on the real BGL_v16
response.  Agreement is limited by the beam band-limit, not by
round-off, so the tolerance here is looser than the data-free gate in
tests/test_engine_gate.py.

Run:
    ulimit -v 16000000
    uv run python scripts/crosscheck_pixel_arm.py 2>&1 | tee \
        /home/christian/Documents/research/lusee/lusee_faraday/generated_data/crosscheck_pixel_arm.log
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from lusee_faraday import config as cfg  # noqa: E402
from lusee_faraday import engine, response as rsp  # noqa: E402
from lusee_faraday import fourport as fp  # noqa: E402

RESPONSE_PATH = os.environ.get(
    "LUSEE_RESPONSE",
    "data/BGL_v16/lusee_bgl_v16_response_v3.fits",
)
FREQ_MHZ = 30.0
NSIDE = 64
LMAX = 30
N_TIMES = 4


def band_limited_sky(rng, nside, lmax):
    """A smooth IQUV sky the beam's band-limit can actually represent."""
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.empty((4, npix))
    for i in range(4):
        alm = hp.synalm(
            np.exp(-np.arange(3 * nside) / 6.0), lmax=lmax, new=True
        )
        maps[i] = hp.alm2map(alm, nside, lmax=lmax)
    maps[0] = np.abs(maps[0]) + 10.0  # Stokes I positive
    maps[3] = 0.0  # no circular polarization
    return maps[None]


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import croissant as cro
    from lusee.ReceiverImpedance import JFETReceiver

    rng = np.random.default_rng(7)
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()

    data = band_limited_sky(rng, NSIDE, LMAX)
    # croissant wants IAU; the maps above are treated as COSMO.
    from lusee_faraday.conventions import cosmo_to_iau_qu

    data[0, 1], data[0, 2] = cosmo_to_iau_qu(data[0, 1], data[0, 2])
    sky = cro.PolarizedSky(
        data, np.array([FREQ_MHZ]), sampling="healpix", coord="galactic"
    )

    times = cfg.times()[:N_TIMES]
    beam = rsp.four_port_pair_alms(resp, FREQ_MHZ, LMAX)
    components = np.asarray(sky.compute_alm(lmax=LMAX))
    W = engine.contract(beam, components, times, cfg.moon_location(), LMAX)
    harmonic = W.sum(axis=1)[0]  # (ntime, npair)

    # Pixel arm: same maps, same response, no Faraday.
    kern = fp.FixedFreqKernel(resp, FREQ_MHZ, receiver)
    grid = fp.GalacticGrid(NSIDE)
    I_map = data[0, 0]
    Q_cos, U_cos = data[0, 1], -data[0, 2]  # back to COSMO for fourport
    sim = fp.SkyWaterfallSim(
        kern, grid, I_map, Q_cos, U_cos, np.zeros_like(I_map)
    )
    lam2 = cfg.lam2(np.array([FREQ_MHZ]))
    pixel = np.empty_like(harmonic)
    for i, t in enumerate(times):
        R = fp.topo_rotation_matrix(t, cfg.moon_location())
        pixel[i] = sim.pair_integrals(R, lam2, faraday=False)[:, 0]

    scale = np.abs(pixel).max()
    worst = np.abs(harmonic - pixel).max() / scale
    print(f"worst relative disagreement: {worst:.3e}")
    print("PASS" if worst < 2e-2 else "FAIL")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the pixel-arm cross-check**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
ulimit -v 16000000
uv run python scripts/crosscheck_pixel_arm.py 2>&1 | tee \
  /home/christian/Documents/research/lusee/lusee_faraday/generated_data/crosscheck_pixel_arm.log
```

Expected: `PASS`, with the worst relative disagreement at or below the `1.1e-2` that `scripts/validate_engine.py` already records for the same comparison. Anything above `2e-2` means the two arms disagree beyond the band-limit and must be investigated before continuing.

- [ ] **Step 5: Commit**

```bash
uv run black tests/test_engine_gate.py scripts/crosscheck_pixel_arm.py
git add tests/test_engine_gate.py scripts/crosscheck_pixel_arm.py
git commit -m "Gate the harmonic contraction against luseepy and the pixel arm"
```

---

### Task 8: `sky.py` — the `FaradaySky` container and its exactness

**Files:**
- Rewrite: `src/lusee_faraday/sky.py` (delete the existing contents; the old `SkyModel`, `load_wmap`, `power_law`, `point_src` go away)
- Test: `tests/test_sky_exactness.py`
- Delete: `tests/test_sky.py`

**Interfaces:**
- Consumes: `conventions.dual_block_phase`.
- Produces:
  - `class FaradaySky` with attributes `component_alms` `(K, 4, L, 2L-1)`, `phi_fd` `(K,)`, `beta` `(K, 4)`, `ref_freq_mhz` `(K, 4)`, `lmax`, `coord`
  - luseepy protocol attributes: `units = "K"`, `convention = "IAU"`, `stokes = ("I", "Q", "U", "V")`, `tangent_basis = "theta-phi"`, `frequency_units = "MHz"`, `frame`
  - `FaradaySky.coeffs(freqs_mhz) -> np.ndarray` `(K, nfreq, 4)`
  - `FaradaySky.polarized_alm_at_freq(target_freqs, lmax=None) -> np.ndarray` `(nfreq, 4, L, 2L-1)`
  - `FaradaySky.from_maps(I, Q, U, phi_fd, ...) -> FaradaySky` — one region, `K = 1`
  - `FaradaySky.uniform_screen(...)`, `FaradaySky.i_only(...)`

**Context:** input Q/U are COSMO. Hand them to `croissant.PolarizedSky(..., convention="COSMO")` and let croissant convert to IAU — do not convert by hand, because the component alms must be the IAU duals that `dual_block_phase` was derived against.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12


@pytest.fixture(scope="module")
def maps():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    Q = rng.normal(size=npix) * 0.1
    U = rng.normal(size=npix) * 0.1
    return I, Q, U


def test_uniform_screen_uses_one_component(maps):
    sky = FaradaySky.uniform_screen(*maps, phi_fd=250.0, lmax=LMAX)
    assert sky.component_alms.shape == (1, 4, LMAX + 1, 2 * LMAX + 1)
    assert sky.phi_fd.shape == (1,)


def test_coeffs_shape_and_flat_spectrum(maps):
    sky = FaradaySky.uniform_screen(*maps, phi_fd=0.0, lmax=LMAX)
    freqs = np.array([29.9, 30.0, 30.1])
    c = sky.coeffs(freqs)
    assert c.shape == (1, 3, 4)
    assert np.allclose(c, 1.0)  # no Faraday, no spectral index


def test_polarized_alm_at_freq_matches_rotating_the_maps_directly(maps):
    """The exactness claim the whole refactor rests on."""
    import croissant as cro

    from lusee_faraday.conventions import faraday_phase_cosmo

    I, Q, U = maps
    phi = 137.0
    freqs = np.array([29.9, 30.0, 30.2])

    sky = FaradaySky.uniform_screen(I, Q, U, phi_fd=phi, lmax=LMAX)
    ours = sky.polarized_alm_at_freq(freqs, lmax=LMAX)
    assert ours.shape == (3, 4, LMAX + 1, 2 * LMAX + 1)

    phase = faraday_phase_cosmo(phi, freqs)  # (nfreq,)
    direct = []
    for i, f in enumerate(freqs):
        P = (Q + 1j * U) * phase[i]
        data = np.stack([I, P.real, P.imag, np.zeros_like(I)])[None]
        rotated = cro.PolarizedSky(
            data,
            np.array([f]),
            sampling="healpix",
            coord="galactic",
            convention="COSMO",
        )
        direct.append(np.asarray(rotated.compute_alm(lmax=LMAX))[0])
    direct = np.stack(direct)

    scale = np.abs(direct).max()
    assert np.abs(ours - direct).max() < 1e-10 * scale


def test_spectral_index_scales_the_right_blocks(maps):
    I, Q, U = maps
    sky = FaradaySky.uniform_screen(
        I,
        Q,
        U,
        phi_fd=0.0,
        lmax=LMAX,
        beta_i=-2.55,
        ref_freq_i=408.0,
        beta_qu=-2.8,
        ref_freq_qu=23e3,
    )
    freqs = np.array([30.0])
    c = sky.coeffs(freqs)
    assert np.isclose(c[0, 0, 0], (30.0 / 408.0) ** -2.55)
    assert np.isclose(c[0, 0, 2], (30.0 / 23e3) ** -2.8)
    assert np.isclose(c[0, 0, 3], (30.0 / 23e3) ** -2.8)


def test_satisfies_the_luseepy_polarized_sky_protocol(maps):
    from lusee.FullStokesSimulator import _validate_polarized_sky_metadata

    sky = FaradaySky.uniform_screen(*maps, phi_fd=0.0, lmax=LMAX)
    _validate_polarized_sky_metadata(sky, require_frequency_units=True)


def test_i_only_has_no_polarized_blocks(maps):
    I, _, _ = maps
    sky = FaradaySky.i_only(I, lmax=LMAX)
    alm = sky.polarized_alm_at_freq(np.array([10.0, 30.0, 50.0]), lmax=LMAX)
    assert np.abs(alm[:, 2]).max() == 0.0
    assert np.abs(alm[:, 3]).max() == 0.0
    assert np.abs(alm[:, 0]).max() > 0.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_sky_exactness.py -v
```

Expected: `ImportError: cannot import name 'FaradaySky'`.

- [ ] **Step 3: Delete the old sky module and its tests, then write the new one**

```bash
git rm tests/test_sky.py
```

```python
"""The Faraday sky, decomposed into spectrally separable components.

Faraday rotation is diagonal in croissant's harmonic dual, so a region of
constant Faraday depth contributes one frequency-independent component
alm plus a per-frequency, per-block coefficient.  A sky is therefore

    alm(nu) = sum_k coeff[k, nu, c] * component_alms[k, c]

which is exact -- not an approximation -- whenever ``phi_FD`` is
piecewise constant, and which makes the 16,384-channel fine grid cost
one einsum rather than 16,384 spherical transforms.

Input Stokes Q/U are healpy/COSMO.  They are handed to
``croissant.PolarizedSky`` with ``convention="COSMO"`` so croissant does
the IAU conversion; the stored component alms are therefore IAU duals,
which is what :func:`lusee_faraday.conventions.dual_block_phase` was
derived against.
"""

import numpy as np

from .conventions import dual_block_phase

STOKES_IQUV = ("I", "Q", "U", "V")


def _component_alm(I, Q, U, lmax, coord):
    """One region's IAU dual alms from COSMO Stokes maps."""
    import croissant as cro

    data = np.stack(
        [
            np.asarray(I, dtype=float),
            np.asarray(Q, dtype=float),
            np.asarray(U, dtype=float),
            np.zeros_like(np.asarray(I, dtype=float)),
        ]
    )[None]
    sky = cro.PolarizedSky(
        data,
        np.array([1.0]),  # placeholder: the spectrum lives in coeffs
        sampling="healpix",
        coord=coord,
        convention="COSMO",
    )
    return np.asarray(sky.compute_alm(lmax=int(lmax)))[0]


class FaradaySky:
    """A sky whose frequency dependence separates into components."""

    units = "K"
    convention = "IAU"
    stokes = STOKES_IQUV
    tangent_basis = "theta-phi"
    frequency_units = "MHz"

    def __init__(
        self,
        component_alms,
        phi_fd,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        self.component_alms = np.asarray(component_alms)
        if self.component_alms.ndim != 4:
            raise ValueError(
                "component_alms must have shape (K, 4, L, 2L-1); got "
                f"{self.component_alms.shape}"
            )
        n_components = self.component_alms.shape[0]
        self.phi_fd = np.atleast_1d(np.asarray(phi_fd, dtype=float))
        if self.phi_fd.size != n_components:
            raise ValueError(
                f"phi_fd has {self.phi_fd.size} entries for "
                f"{n_components} components"
            )
        self.beta = (
            np.zeros((n_components, 4))
            if beta is None
            else np.asarray(beta, dtype=float)
        )
        self.ref_freq_mhz = (
            np.ones((n_components, 4))
            if ref_freq_mhz is None
            else np.asarray(ref_freq_mhz, dtype=float)
        )
        self.lmax = self.component_alms.shape[2] - 1
        self.coord = coord
        self.frame = coord

    @property
    def n_components(self):
        return self.component_alms.shape[0]

    def coeffs(self, freqs_mhz):
        """Per-frequency, per-block coefficients; shape ``(K, nfreq, 4)``."""
        freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
        scale = (
            freqs[None, :, None] / self.ref_freq_mhz[:, None, :]
        ) ** self.beta[:, None, :]
        return scale * dual_block_phase(self.phi_fd, freqs)

    def polarized_alm_at_freq(self, target_freqs, lmax=None):
        """Sky alms at each target frequency; the luseepy sky protocol."""
        target_lmax = self.lmax if lmax is None else int(lmax)
        if target_lmax > self.lmax:
            raise ValueError(
                f"requested lmax={target_lmax} exceeds the sky's "
                f"{self.lmax}"
            )
        alms = self.component_alms
        if target_lmax < self.lmax:
            # m = 0 sits at index lmax, so trimming ell also trims m
            # symmetrically from both ends.
            drop = self.lmax - target_lmax
            alms = alms[:, :, : target_lmax + 1, drop : alms.shape[3] - drop]
        return np.einsum(
            "kclm,kfc->fclm",
            alms,
            self.coeffs(target_freqs),
            optimize=True,
        )

    @classmethod
    def from_maps(
        cls,
        I,
        Q,
        U,
        phi_fd,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """One region of constant Faraday depth."""
        alm = _component_alm(I, Q, U, lmax, coord)[None]
        return cls(alm, [float(phi_fd)], beta, ref_freq_mhz, coord)

    @classmethod
    def uniform_screen(
        cls,
        I,
        Q,
        U,
        phi_fd,
        lmax,
        beta_i=0.0,
        ref_freq_i=1.0,
        beta_qu=0.0,
        ref_freq_qu=1.0,
        coord="galactic",
    ):
        """A constant Faraday depth across the whole sky."""
        beta = np.array([[beta_i, beta_i, beta_qu, beta_qu]])
        ref = np.array([[ref_freq_i, ref_freq_i, ref_freq_qu, ref_freq_qu]])
        return cls.from_maps(I, Q, U, phi_fd, lmax, beta, ref, coord)

    @classmethod
    def i_only(cls, I, lmax, beta_i=0.0, ref_freq_i=1.0, coord="galactic"):
        """Perfect depolarization: Stokes I only, no polarized blocks."""
        zeros = np.zeros_like(np.asarray(I, dtype=float))
        beta = np.array([[beta_i, beta_i, 0.0, 0.0]])
        ref = np.array([[ref_freq_i, ref_freq_i, 1.0, 1.0]])
        sky = cls.from_maps(I, zeros, zeros, 0.0, lmax, beta, ref, coord)
        sky.component_alms[:, 2:] = 0.0
        return sky
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_sky_exactness.py -v
```

Expected: 6 passed. `test_polarized_alm_at_freq_matches_rotating_the_maps_directly` is the one that matters — it is the exactness claim the architecture rests on. If it fails, the culprit is almost certainly the `P_MINUS`/`P_PLUS` phase assignment in `conventions.dual_block_phase`, not this module.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/sky.py tests/test_sky_exactness.py
uv run flake8 src/lusee_faraday/sky.py
git add -A src/lusee_faraday/sky.py tests/test_sky_exactness.py tests/test_sky.py
git commit -m "Replace SkyModel with the spectrally separable FaradaySky"
```

---

### Task 9: `sky.py` — point-source and binned-screen constructors

**Files:**
- Modify: `src/lusee_faraday/sky.py`
- Test: `tests/test_sky_constructors.py`

**Interfaces:**
- Consumes: `FaradaySky.__init__`, `_component_alm` from Task 8.
- Produces:
  - `FaradaySky.point_source(theta, phi, stokes, phi_fd, nside, lmax, ...) -> FaradaySky` — `K = len(theta)`
  - `FaradaySky.binned_screen(I, Q, U, rm_map, dphi, lmax, ...) -> FaradaySky` — `K = number of occupied phi bins`

**Context:** `binned_screen` must partition the sky, never overlap it: each component holds I, Q and U masked to its own Faraday-depth bin, so summing the components reproduces the input maps exactly. Getting this wrong double-counts Stokes I, which shows up as a factor-of-two error in the leakage numbers rather than as an obvious failure.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12


@pytest.fixture(scope="module")
def hp_module():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    return hp


def test_point_source_makes_one_component_per_source(hp_module):
    sky = FaradaySky.point_source(
        theta=np.array([0.5, 1.2]),
        phi=np.array([0.0, 2.0]),
        stokes=np.array([[1.0, -1.0, 0.0], [2.0, 0.0, 0.5]]),
        phi_fd=np.array([250.0, -30.0]),
        nside=NSIDE,
        lmax=LMAX,
    )
    assert sky.n_components == 2
    assert np.allclose(sky.phi_fd, [250.0, -30.0])


def test_point_source_components_carry_their_own_faraday_depth(hp_module):
    """Two sources with different phi must rotate at different rates."""
    sky = FaradaySky.point_source(
        theta=np.array([0.5, 1.2]),
        phi=np.array([0.0, 2.0]),
        stokes=np.array([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]),
        phi_fd=np.array([250.0, 0.0]),
        nside=NSIDE,
        lmax=LMAX,
    )
    c = sky.coeffs(np.array([30.0, 30.05]))
    assert not np.allclose(c[0, :, 2], c[0, 0, 2])  # source 0 winds
    assert np.allclose(c[1, :, 2], 1.0)  # source 1 does not


def test_binned_screen_partitions_the_sky_exactly(hp_module):
    hp = hp_module
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    Q = rng.normal(size=npix) * 0.1
    U = rng.normal(size=npix) * 0.1
    rm = rng.uniform(-40.0, 40.0, size=npix)

    sky = FaradaySky.binned_screen(I, Q, U, rm, dphi=10.0, lmax=LMAX)
    assert sky.n_components == 8  # (-40, 40) in steps of 10

    # With every phi bin forced to zero depth the sum of the components
    # must be the alm of the unpartitioned sky.
    whole = FaradaySky.uniform_screen(I, Q, U, phi_fd=0.0, lmax=LMAX)
    summed = sky.component_alms.sum(axis=0)
    scale = np.abs(whole.component_alms[0]).max()
    assert np.abs(summed - whole.component_alms[0]).max() < 1e-10 * scale


def test_binned_screen_assigns_each_component_its_bin_centre(hp_module):
    hp = hp_module
    npix = hp.nside2npix(NSIDE)
    rm = np.full(npix, 7.0)
    rm[: npix // 2] = -23.0
    I = np.ones(npix)
    z = np.zeros(npix)
    sky = FaradaySky.binned_screen(I, z, z, rm, dphi=10.0, lmax=LMAX)
    assert sky.n_components == 2
    assert np.allclose(np.sort(sky.phi_fd), [-23.0, 7.0])


def test_binned_screen_rejects_a_nonpositive_bin_width(hp_module):
    hp = hp_module
    npix = hp.nside2npix(NSIDE)
    ones = np.ones(npix)
    with pytest.raises(ValueError, match="dphi"):
        FaradaySky.binned_screen(
            ones, ones, ones, ones, dphi=0.0, lmax=LMAX
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_sky_constructors.py -v
```

Expected: `AttributeError: type object 'FaradaySky' has no attribute 'point_source'`.

- [ ] **Step 3: Append the constructors to `sky.py`**

```python
    @classmethod
    def point_source(
        cls,
        theta,
        phi,
        stokes,
        phi_fd,
        nside,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Discrete sources, each with its own Faraday depth.

        Parameters
        ----------
        theta, phi : (n_sources,) float
            Source directions in ``coord``, radians.
        stokes : (n_sources, 3) float
            Per-source I, Q, U in the healpy/COSMO convention.  The
            values land in a single HEALPix pixel each, so they carry
            the pixel's solid angle.
        phi_fd : (n_sources,) float
            Faraday depth per source, rad/m^2.
        """
        import healpy as hp

        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        stokes = np.atleast_2d(np.asarray(stokes, dtype=float))
        phi_fd = np.atleast_1d(np.asarray(phi_fd, dtype=float))
        n = theta.size
        if not (phi.size == n and stokes.shape == (n, 3) and
                phi_fd.size == n):
            raise ValueError(
                "theta, phi, stokes and phi_fd must describe the same "
                "number of sources"
            )
        npix = hp.nside2npix(int(nside))
        pix = hp.ang2pix(int(nside), theta, phi)
        alms = []
        for k in range(n):
            maps = np.zeros((3, npix))
            maps[:, pix[k]] = stokes[k]
            alms.append(_component_alm(*maps, lmax, coord))
        return cls(
            np.stack(alms), phi_fd, beta, ref_freq_mhz, coord
        )

    @classmethod
    def binned_screen(
        cls,
        I,
        Q,
        U,
        rm_map,
        dphi,
        lmax,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Partition a Faraday screen into bins of constant depth.

        Each component holds I, Q and U masked to its own bin, so the
        components partition the sky rather than overlapping it: summing
        them reproduces the input maps exactly.
        """
        dphi = float(dphi)
        if dphi <= 0:
            raise ValueError("dphi must be positive")
        I = np.asarray(I, dtype=float)
        Q = np.asarray(Q, dtype=float)
        U = np.asarray(U, dtype=float)
        rm = np.asarray(rm_map, dtype=float)
        index = np.floor(rm / dphi).astype(int)
        alms, depths = [], []
        for value in np.unique(index):
            mask = index == value
            alms.append(
                _component_alm(I * mask, Q * mask, U * mask, lmax, coord)
            )
            depths.append(float(rm[mask].mean()))
        return cls(np.stack(alms), depths, beta, ref_freq_mhz, coord)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_sky_constructors.py -v
```

Expected: 5 passed. `test_binned_screen_partitions_the_sky_exactly` is the one guarding against double-counting.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/sky.py tests/test_sky_constructors.py
uv run flake8 src/lusee_faraday/sky.py
git add src/lusee_faraday/sky.py tests/test_sky_constructors.py
git commit -m "Add point-source and binned-screen sky constructors"
```

---

### Task 10: `sky.py` — the audit's two criteria, as a guardrail

**Files:**
- Modify: `src/lusee_faraday/sky.py`
- Test: `tests/test_sky_diagnostics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `nyquist_nside(rm_map, freq_mhz, percentile=99.9) -> float`
  - `spectral_component_count(phi_min, phi_max, freqs_mhz) -> int`
  - `FaradaySky.from_rm_map(I, Q, U, rm_map, freqs_mhz, lmax, allow_pixelwise=False, max_components=4096, ...)` — chooses `dphi` from the band, refuses when the map is spatially unresolved

**Context — this is where the audit stops being a paragraph someone has to remember.** Two different criteria, both cheap, both reported:

- *Spectral*: how many constant-depth components are needed for the phase to be resolved **across the simulated band**, `dphi <= pi / (2 * span(lambda^2))`. Governs cost.
- *Spatial*: the audit's Nyquist criterion — the `nside` at which the phase is resolved **between adjacent pixels at fixed frequency**. Governs whether the answer means anything. This is the number that reads ~2.8e5 for the real RM map at 30 MHz.

`from_rm_map` raises unless `allow_pixelwise=True`, quoting both numbers and pointing at the audit.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import sky as sky_mod  # noqa: E402
from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12


def test_spectral_component_count_scales_with_the_depth_range():
    freqs = np.array([29.9, 30.1])
    few = sky_mod.spectral_component_count(-30.0, 30.0, freqs)
    many = sky_mod.spectral_component_count(-2400.0, 2400.0, freqs)
    assert 1 <= few < many
    assert many > 1000  # the full Galactic screen is expensive


def test_spectral_component_count_is_one_for_a_uniform_screen():
    freqs = np.array([29.9, 30.1])
    assert sky_mod.spectral_component_count(50.0, 50.0, freqs) == 1


def test_nyquist_nside_grows_with_frequency_and_gradient():
    pytest.importorskip("healpy")
    import healpy as hp

    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    smooth = hp.smoothing(rng.normal(size=npix), fwhm=0.5) * 10.0
    rough = rng.normal(size=npix) * 10.0
    assert (
        sky_mod.nyquist_nside(rough, 30.0)
        > sky_mod.nyquist_nside(smooth, 30.0)
    )
    # lambda^2 grows as nu^-2, so a lower frequency needs finer pixels
    assert (
        sky_mod.nyquist_nside(smooth, 10.0)
        > sky_mod.nyquist_nside(smooth, 30.0)
    )


def test_from_rm_map_refuses_an_unresolved_screen():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(1)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    z = np.zeros(npix)
    rm = rng.normal(size=npix) * 300.0  # wildly unresolved
    with pytest.raises(ValueError) as excinfo:
        FaradaySky.from_rm_map(
            I, z, z, rm, np.array([29.9, 30.1]), lmax=LMAX
        )
    message = str(excinfo.value)
    assert "nside" in message
    assert "allow_pixelwise" in message


def test_from_rm_map_accepts_a_resolved_screen():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(2)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    z = np.zeros(npix)
    rm = hp.smoothing(rng.normal(size=npix), fwhm=1.0) * 1e-3
    sky = FaradaySky.from_rm_map(
        I, z, z, rm, np.array([29.9, 30.1]), lmax=LMAX
    )
    assert sky.n_components >= 1
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_sky_diagnostics.py -v
```

Expected: `AttributeError: module 'lusee_faraday.sky' has no attribute 'spectral_component_count'`.

- [ ] **Step 3: Append the diagnostics to `sky.py`**

```python
AUDIT_REFERENCE = (
    "see the 2026-08-18 audit (commit 4b401c5): the diffuse-sky Faraday "
    "signature is HEALPix shot noise when the screen is unresolved"
)


def spectral_component_count(phi_min, phi_max, freqs_mhz):
    """Constant-depth components needed to resolve the phase in-band.

    The Faraday phase turns by ``2 * phi * lambda^2``, so across a band
    spanning ``d(lambda^2)`` two depths differing by more than
    ``pi / (2 d(lambda^2))`` are no longer coherent and need separate
    components.  This governs cost, not validity.
    """
    from .conventions import lambda_squared

    lam2 = lambda_squared(freqs_mhz)
    span = float(lam2.max() - lam2.min())
    width = float(phi_max) - float(phi_min)
    if span <= 0 or width <= 0:
        return 1
    return int(np.ceil(width / (np.pi / (2 * span))))


def nyquist_nside(rm_map, freq_mhz, percentile=99.9):
    """The nside at which the screen is resolved between adjacent pixels.

    This is the audit's criterion.  At 30 MHz the real Hutschenreuter map
    returns of order 3e5, i.e. ~1e12 pixels: the input does not determine
    the answer at any computable resolution, and no engine choice
    rescues it.
    """
    import healpy as hp

    from .conventions import lambda_squared

    rm = np.asarray(rm_map, dtype=float)
    nside0 = hp.npix2nside(rm.size)
    neighbours = hp.get_all_neighbours(nside0, np.arange(rm.size))
    valid = neighbours >= 0
    diffs = np.abs(rm[np.where(valid, neighbours, 0)] - rm[None, :])
    step = float(np.percentile(diffs[valid], percentile))
    lam2 = float(lambda_squared(freq_mhz).max())
    phase_step = 2.0 * step * lam2
    return nside0 * max(1.0, phase_step / np.pi)
```

- [ ] **Step 4: Append the gated constructor to `FaradaySky`**

```python
    @classmethod
    def from_rm_map(
        cls,
        I,
        Q,
        U,
        rm_map,
        freqs_mhz,
        lmax,
        allow_pixelwise=False,
        max_components=4096,
        beta=None,
        ref_freq_mhz=None,
        coord="galactic",
    ):
        """Build a binned screen, refusing an unresolved one.

        Reports both audit criteria and raises unless the caller has
        explicitly opted in to a screen the map cannot resolve.
        """
        import healpy as hp

        from .conventions import lambda_squared

        rm = np.asarray(rm_map, dtype=float)
        used_nside = hp.npix2nside(rm.size)
        needed_nside = nyquist_nside(rm, np.min(freqs_mhz))
        n_needed = spectral_component_count(
            float(rm.min()), float(rm.max()), freqs_mhz
        )
        if needed_nside > used_nside and not allow_pixelwise:
            raise ValueError(
                f"Faraday screen is not resolved: nside={used_nside} used, "
                f"nside~{needed_nside:.3g} needed at "
                f"{np.min(freqs_mhz):g} MHz. The pixel sum is a random "
                f"walk, not a quadrature ({AUDIT_REFERENCE}). Pass "
                "allow_pixelwise=True to build it anyway."
            )
        if n_needed > max_components and not allow_pixelwise:
            raise ValueError(
                f"screen needs {n_needed} components across the band "
                f"(cap {max_components}); pass allow_pixelwise=True or "
                "narrow the band."
            )
        span = float(np.ptp(lambda_squared(freqs_mhz)))
        dphi = np.pi / (2 * span) if span > 0 else (np.ptp(rm) or 1.0)
        return cls.binned_screen(
            I, Q, U, rm, dphi, lmax, beta, ref_freq_mhz, coord
        )
```

- [ ] **Step 5: Log croissant's resolved engine at construction**

The spec requires this: a silent fall back to a dense transform must be
visible as a log line, not as an OOM kill twenty minutes later.  Add to
the top of `sky.py`:

```python
import logging

logger = logging.getLogger(__name__)
```

and, in `_component_alm`, immediately after building the `PolarizedSky`
and before `compute_alm`:

```python
    logger.info(
        "component transform engines: %s (%s)",
        sky.engine,
        sky.engine_reason,
    )
```

Add the matching test to `tests/test_sky_diagnostics.py`:

```python
def test_component_construction_logs_the_resolved_engine(caplog):
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    npix = hp.nside2npix(NSIDE)
    ones = np.ones(npix)
    with caplog.at_level("INFO", logger="lusee_faraday.sky"):
        FaradaySky.uniform_screen(
            ones, 0 * ones, 0 * ones, phi_fd=0.0, lmax=LMAX
        )
    assert any("transform engines" in r.message for r in caplog.records)
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/test_sky_diagnostics.py tests/test_sky_exactness.py tests/test_sky_constructors.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run black src/lusee_faraday/sky.py tests/test_sky_diagnostics.py
uv run flake8 src/lusee_faraday/sky.py
git add src/lusee_faraday/sky.py tests/test_sky_diagnostics.py
git commit -m "Turn the audit's two criteria into a constructor guardrail"
```

---

### Task 11: `instrument.py` — luseepy covariance assembly

**Files:**
- Create: `src/lusee_faraday/instrument.py`
- Test: `tests/test_instrument.py`

**Interfaces:**
- Consumes: `conventions.PORT_PAIRS`, `conventions.PRODUCT_LABELS`.
- Produces:
  - `covariance(pair_integrals, resp, receiver, freqs_mhz, T_moon=250.0, T_ant=0.0) -> np.ndarray` shape `(ntime, nfreq, 4, 4)` complex, Hermitian
  - `channels(covariance_matrix, products="all") -> tuple[np.ndarray, tuple[str, ...]]` — `(ntime, nfreq, 16)` real plus labels
  - `unpack_channels(channels) -> np.ndarray` `(..., 4, 4)` complex Hermitian
  - `blackbody_normalization(resp, receiver, freqs_mhz) -> np.ndarray` `(nfreq, 4, 4)`

**Context:** every physical step here is a luseepy function; this module only wires them in the right order and evaluates `Z_A`/`Z_L` on the **fine** grid so receiver loading is not smeared by the fixed-beam approximation. Order is fixed by `lusee.FullStokesSimulatorBase.simulate` (`luseepy/lusee/FullStokesSimulator.py:442`): `assemble_open_covariance` -> `apply_receiver_loading` -> `project_hermitian` -> `pack_covariance`.

Note the axis convention: `assemble_open_covariance` expects pair integrals with shape `(ntime, nfreq, npair)`, which is exactly what `engine.expand` returns.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import instrument as inst  # noqa: E402


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = lusee.synthetic_four_port_response(freq_mhz=(10.0, 20.0))
    return resp, JFETReceiver()


def test_channels_roundtrip_through_unpack():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 4, 4)) + 1j * rng.normal(size=(3, 5, 4, 4))
    C = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    ch, labels = inst.channels(C)
    assert ch.shape == (3, 5, 16)
    assert len(labels) == 16
    assert np.allclose(inst.unpack_channels(ch), C)


def test_channel_labels_match_luseepy():
    lusee_cov = pytest.importorskip("lusee.Covariance")
    rng = np.random.default_rng(1)
    C = np.zeros((1, 1, 4, 4), dtype=complex)
    C[..., 0, 0] = 1.0
    _, labels = inst.channels(C)
    assert labels == lusee_cov.default_product_labels()


def test_covariance_is_hermitian(pieces):
    resp, receiver = pieces
    rng = np.random.default_rng(2)
    freqs = np.array([10.0, 12.0, 20.0])
    pair = rng.normal(size=(4, 3, 10)) + 1j * rng.normal(size=(4, 3, 10))
    C = inst.covariance(pair, resp, receiver, freqs)
    assert C.shape == (4, 3, 4, 4)
    assert np.allclose(C, np.conj(np.swapaxes(C, -1, -2)))


def test_covariance_is_linear_in_the_pair_integrals(pieces):
    """T_moon and T_ant are additive offsets; the sky term must be linear."""
    resp, receiver = pieces
    rng = np.random.default_rng(3)
    freqs = np.array([10.0, 20.0])
    a = rng.normal(size=(2, 2, 10)) + 1j * rng.normal(size=(2, 2, 10))
    b = rng.normal(size=(2, 2, 10)) + 1j * rng.normal(size=(2, 2, 10))
    kw = dict(T_moon=0.0, T_ant=0.0)
    Ca = inst.covariance(a, resp, receiver, freqs, **kw)
    Cb = inst.covariance(b, resp, receiver, freqs, **kw)
    Cab = inst.covariance(2 * a + 3 * b, resp, receiver, freqs, **kw)
    assert np.allclose(Cab, 2 * Ca + 3 * Cb)


def test_blackbody_normalization_shape(pieces):
    resp, receiver = pieces
    freqs = np.array([10.0, 15.0, 20.0])
    B = inst.blackbody_normalization(resp, receiver, freqs)
    assert B.shape == (3, 4, 4)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_instrument.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.instrument'`.

- [ ] **Step 3: Write the implementation**

```python
"""Covariance assembly, entirely on luseepy's instrument physics.

This module owns the wiring, not the physics.  The order is fixed by
``lusee.FullStokesSimulatorBase.simulate``: assemble the open covariance
from the sky pair integrals plus the Moon and antenna-metal terms, apply
the receiver loading, project onto the Hermitian part, then pack the 16
real science channels.

``Z_A`` and ``Z_L`` are evaluated on whatever frequency grid the caller
passes, which for a Faraday run is the fine grid.  The fixed-beam
approximation applies to the response alms only, so receiver loading is
not smeared along with it.
"""

import numpy as np

from .conventions import PORT_PAIRS, PRODUCT_LABELS


def covariance(
    pair_integrals,
    resp,
    receiver,
    freqs_mhz,
    T_moon=250.0,
    T_ant=0.0,
):
    """Loaded, Hermitian port covariance -> ``(ntime, nfreq, 4, 4)``."""
    from lusee.Covariance import (
        apply_receiver_loading,
        assemble_open_covariance,
        project_hermitian,
    )

    freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
    ZA, _, Rmoon, Rloss, _ = resp.target_matrices(freqs)
    open_cov = assemble_open_covariance(
        np.asarray(pair_integrals),
        Rmoon,
        Rloss,
        T_moon=T_moon,
        T_ant=T_ant,
    )
    ZL = receiver.Z(freqs)
    unprojected, _ = apply_receiver_loading(open_cov, ZA, ZL)
    return np.asarray(project_hermitian(unprojected))


def blackbody_normalization(resp, receiver, freqs_mhz):
    """Covariance response to a one-kelvin blackbody enclosure."""
    from lusee.Covariance import (
        blackbody_normalization as _blackbody,
        loading_matrix,
    )

    freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
    ZA, _, _, _, _ = resp.target_matrices(freqs)
    M = loading_matrix(ZA, receiver.Z(freqs))
    return np.asarray(_blackbody(ZA, M))


def channels(covariance_matrix, products="all"):
    """Hermitian covariance -> 16 real channels plus their labels."""
    C = np.asarray(covariance_matrix)
    if products != "all":
        raise ValueError("only products='all' is supported")
    out = np.empty(C.shape[:-2] + (16,), dtype=np.float64)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            out[..., k] = C[..., a, b].real
            k += 1
        else:
            out[..., k] = C[..., a, b].real
            out[..., k + 1] = C[..., a, b].imag
            k += 2
    return out, PRODUCT_LABELS


def unpack_channels(packed):
    """16 real channels -> Hermitian covariance ``(..., 4, 4)``."""
    ch = np.asarray(packed)
    C = np.zeros(ch.shape[:-1] + (4, 4), dtype=complex)
    k = 0
    for a, b in PORT_PAIRS:
        if a == b:
            C[..., a, b] = ch[..., k]
            k += 1
        else:
            C[..., a, b] = ch[..., k] + 1j * ch[..., k + 1]
            C[..., b, a] = np.conj(C[..., a, b])
            k += 2
    return C
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_instrument.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/instrument.py tests/test_instrument.py
uv run flake8 src/lusee_faraday/instrument.py
git add src/lusee_faraday/instrument.py tests/test_instrument.py
git commit -m "Wire covariance assembly onto luseepy's instrument physics"
```

---

### Task 12: `polarimeter.py` — zenith calibration and pseudo-Stokes

**Files:**
- Create: `src/lusee_faraday/polarimeter.py`
- Test: `tests/test_polarimeter.py`

**Interfaces:**
- Consumes: `instrument.covariance`, `instrument.unpack_channels`, `response.native_channel_index`.
- Produces:
  - `X_VEC`, `Y_VEC` — the raw pseudo-dipoles `X = E - W`, `Y = N - S`
  - `zenith_covariance(resp, receiver, freq_mhz) -> np.ndarray` `(4, 4)`
  - `zenith_port_weights(C0, null_q=True) -> tuple[np.ndarray, np.ndarray]`
  - `orthonormalize_xy(C0, x_vec, y_vec) -> tuple[np.ndarray, np.ndarray]`
  - `zenith_vectors(resp, receiver, freq_mhz, mode="ortho") -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(x_vec, y_vec, C0)`
  - `pseudo_stokes(C, x_vec=None, y_vec=None) -> np.ndarray` `(..., 4)`
  - `pseudo_stokes_from_channels(channels, x_vec=None, y_vec=None) -> np.ndarray`
  - `check_psd(stokes, rtol=1e-9) -> None` — raises on `sqrt(Q^2+U^2+V^2) > I`

**Context:** the algorithms are lifted from `fourport.py` (`zenith_port_weights`, `orthonormalize_xy`, `polarimeter`) with one change: `zenith_covariance` samples the response through `lusee.InstrumentResponse.pair_stokes_at`, which already applies the `eta0 / lambda^2` scaling, instead of through the bespoke `FixedFreqKernel`.

`check_psd` is a **runtime** assertion, not just a test. Any physical covariance obeys `sqrt(Q^2 + U^2 + V^2) <= I`; a single polarized source is rank-1, so equality. This invariant previously caught a real bug — the `e^{-2i chi}` coefficient of a complex cross-pair kernel is `0.5 * (K_Q + i K_U)`, not the conjugate of the `e^{+2i chi}` one.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import polarimeter as pol  # noqa: E402


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = lusee.synthetic_four_port_response(
        freq_mhz=(10.0, 20.0), angular_step_deg=5.0
    )
    return resp, JFETReceiver()


def test_pseudo_stokes_of_a_pure_x_state():
    vx = np.array([0.0, 1.0, 0.0, -1.0])
    C = np.einsum("a,b->ab", vx, vx).astype(complex)
    I, Q, U, V = pol.pseudo_stokes(C)
    assert np.isclose(I, 2.0) and np.isclose(Q, 2.0)
    assert np.isclose(U, 0.0) and np.isclose(V, 0.0)


def test_pseudo_stokes_of_a_pure_u_state():
    v = pol.X_VEC + pol.Y_VEC
    C = np.einsum("a,b->ab", v, v).astype(complex)
    I, Q, U, V = pol.pseudo_stokes(C)
    assert np.isclose(Q, 0.0, atol=1e-12)
    assert np.isclose(U, I)
    assert np.isclose(V, 0.0, atol=1e-12)


def test_check_psd_accepts_physical_and_rejects_unphysical():
    pol.check_psd(np.array([1.0, 0.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match="PSD"):
        pol.check_psd(np.array([1.0, 0.9, 0.9, 0.0]))


def test_ortho_vectors_null_the_zenith_polarization(pieces):
    resp, receiver = pieces
    x, y, C0 = pol.zenith_vectors(resp, receiver, 10.0, mode="ortho")
    I, Q, U, V = pol.pseudo_stokes(C0, x, y)
    assert abs(Q) < 1e-12 * I
    assert abs(U) < 1e-12 * I
    assert abs(V) < 1e-12 * I


def test_gains_mode_nulls_q_but_not_necessarily_u(pieces):
    resp, receiver = pieces
    x, y, C0 = pol.zenith_vectors(resp, receiver, 10.0, mode="gains")
    I, Q, _, _ = pol.pseudo_stokes(C0, x, y)
    assert abs(Q) < 1e-12 * I


def test_zenith_vectors_reject_an_unknown_mode(pieces):
    resp, receiver = pieces
    with pytest.raises(ValueError, match="mode"):
        pol.zenith_vectors(resp, receiver, 10.0, mode="magic")
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_polarimeter.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.polarimeter'`.

- [ ] **Step 3: Write the implementation**

```python
"""The four-port polarimeter and its zenith calibration.

Raw pseudo-dipoles are ``X = E - W`` and ``Y = N - S``.  As built, the
four ports are neither identical nor uncoupled, so an unpolarized source
at zenith does not give ``Q = U = V = 0``.  Two calibrations fix that:

- ``mode="gains"``: real per-port weights ``w_p ~ 1/sqrt(C0_pp)`` plus a
  common X/Y rescale.  Nulls zenith pseudo-Q exactly; a residual U
  survives through the inter-port cross-couplings, which no real
  diagonal gain can remove.
- ``mode="ortho"`` (default): Loewdin ``G^{-1/2}`` orthonormalization of
  the pair in the C0 metric.  X and Y become complex combinations of all
  four ports and zenith Q, U and V all vanish.
"""

import numpy as np

from .instrument import covariance, unpack_channels

# Ports N, E, S, W = 0, 1, 2, 3
Y_VEC = np.array([1.0, 0.0, -1.0, 0.0])  # Y = N - S
X_VEC = np.array([0.0, 1.0, 0.0, -1.0])  # X = E - W


def pseudo_stokes(C, x_vec=None, y_vec=None):
    """Pseudo-Stokes I, Q, U, V from a port covariance ``(..., 4, 4)``.

    ``XX = <|X|^2>``, ``YY = <|Y|^2>``, ``XY = <Y X*>`` and
    ``I = (XX+YY)/2``, ``Q = (XX-YY)/2``, ``U = Re XY``, ``V = Im XY``.
    """
    xv = X_VEC if x_vec is None else np.asarray(x_vec)
    yv = Y_VEC if y_vec is None else np.asarray(y_vec)
    C = np.asarray(C)
    XX = np.einsum("a,b,...ab->...", xv, np.conj(xv), C).real
    YY = np.einsum("a,b,...ab->...", yv, np.conj(yv), C).real
    XY = np.einsum("a,b,...ab->...", yv, np.conj(xv), C)
    return np.stack(
        [0.5 * (XX + YY), 0.5 * (XX - YY), XY.real, XY.imag], axis=-1
    )


def pseudo_stokes_from_channels(packed, x_vec=None, y_vec=None):
    """Pseudo-Stokes straight from the 16 packed real channels."""
    return pseudo_stokes(unpack_channels(packed), x_vec, y_vec)


def check_psd(stokes, rtol=1e-9):
    """Assert the physical bound ``sqrt(Q^2+U^2+V^2) <= I``.

    A runtime check, not just a test: this invariant caught a real sign
    bug in the complex cross-pair kernel decomposition.
    """
    s = np.asarray(stokes)
    I = s[..., 0]
    p = np.sqrt(s[..., 1] ** 2 + s[..., 2] ** 2 + s[..., 3] ** 2)
    worst = np.max(p - I * (1.0 + rtol))
    if worst > 0:
        raise ValueError(
            f"PSD violation: sqrt(Q^2+U^2+V^2) exceeds I by {worst:.3e}"
        )


def zenith_covariance(resp, receiver, freq_mhz):
    """Loaded covariance of a unit unpolarized source at exact zenith."""
    kernel = np.asarray(
        resp.pair_stokes_at(
            np.array([0.0]), np.array([0.0]), np.array([float(freq_mhz)])
        )
    )  # (npair, nfreq, 4, ndir)
    pair = kernel[:, 0, 0, 0][None, None, :]  # unpolarized: I kernel only
    C = covariance(
        pair,
        resp,
        receiver,
        np.array([float(freq_mhz)]),
        T_moon=0.0,
        T_ant=0.0,
    )
    return C[0, 0]


def zenith_port_weights(C0, null_q=True):
    """Real per-port gain weights that equalize the zenith autos."""
    C0 = np.asarray(C0)
    autos = np.diagonal(C0).real
    g = np.exp(np.mean(np.log(autos)))
    w = np.sqrt(g / autos)
    x_vec = np.array([0.0, w[1], 0.0, -w[3]])
    y_vec = np.array([w[0], 0.0, -w[2], 0.0])
    if null_q:
        XX = np.einsum("a,b,ab->", x_vec, x_vec, C0).real
        YY = np.einsum("a,b,ab->", y_vec, y_vec, C0).real
        s = (YY / XX) ** 0.25
        x_vec, y_vec = x_vec * s, y_vec / s
    return x_vec, y_vec


def orthonormalize_xy(C0, x_vec, y_vec):
    """Loewdin-orthonormalize (X, Y) in the metric of ``C0``.

    The pseudo-Stokes Q, U and V of a source with covariance ``C0``
    vanish exactly iff ``conj(x)`` and ``conj(y)`` are C0-orthogonal with
    equal C0-norms, because the polarimeter forms are ``p^H C0 p`` and
    ``q^H C0 p`` with ``p = conj(x)``, ``q = conj(y)``.  The symmetric
    ``G^{-1/2}`` transform achieves that while perturbing the input
    dipole vectors as little as possible.
    """
    C0 = np.asarray(C0)
    P = np.stack(
        [
            np.conj(np.asarray(x_vec, dtype=complex)),
            np.conj(np.asarray(y_vec, dtype=complex)),
        ],
        axis=1,
    )
    G = P.conj().T @ C0 @ P
    scale = np.sqrt(np.real(G[0, 0] * G[1, 1]))
    evals, evecs = np.linalg.eigh(G)
    if evals.min() <= 0:
        raise ValueError("X/Y are degenerate in the C0 metric")
    G_isqrt = (evecs / np.sqrt(evals)) @ evecs.conj().T
    P_new = P @ G_isqrt * np.sqrt(scale)
    return np.conj(P_new[:, 0]), np.conj(P_new[:, 1])


def zenith_vectors(resp, receiver, freq_mhz, mode="ortho"):
    """Calibrated (x_vec, y_vec, C0) for one band center."""
    if mode not in ("gains", "ortho"):
        raise ValueError(f"unknown mode {mode!r}; use 'gains' or 'ortho'")
    C0 = zenith_covariance(resp, receiver, freq_mhz)
    x_vec, y_vec = zenith_port_weights(C0, null_q=True)
    if mode == "ortho":
        x_vec, y_vec = orthonormalize_xy(C0, x_vec, y_vec)
    return x_vec, y_vec, C0
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_polarimeter.py -v
```

Expected: 6 passed.

Note on `zenith_covariance`: `pair_stokes_at` returns `(npair, nfreq, 4, ndir)`. The slice `kernel[:, 0, 0, 0]` takes the first frequency, the Stokes-I block and the single direction, giving `(npair,)`; it is then reshaped to the `(ntime=1, nfreq=1, npair)` that `instrument.covariance` expects. If the test fails on shapes, print `kernel.shape` first rather than guessing.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/polarimeter.py tests/test_polarimeter.py
uv run flake8 src/lusee_faraday/polarimeter.py
git add src/lusee_faraday/polarimeter.py tests/test_polarimeter.py
git commit -m "Add the zenith-calibrated polarimeter with a runtime PSD check"
```

---

### Task 13: `channelization.py` — parent and zoom bins

**Files:**
- Create: `src/lusee_faraday/channelization.py`
- Test: `tests/test_channelization.py`
- Delete: `src/lusee_faraday/spectrometer.py`, `tests/test_spectrometer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ZOOM_STEP_HZ = 25000.0 / 64`, `FINE_STEP_HZ = 25000.0 / 2048`
  - `zoom_bin_offsets_hz() -> np.ndarray` (64,)
  - `parent_weights(offsets_hz, notch=0) -> np.ndarray`
  - `zoom_weights(offsets_hz) -> np.ndarray` `(noff, 64)`
  - `ideal_zoom_weights(offsets_hz, fwhm_hz=ZOOM_STEP_HZ) -> np.ndarray` `(noff, 64)`
  - `integrate(waterfall, fine_freqs_mhz, parent_centers_mhz, notch=0) -> dict` with keys `"parent"`, `"zoom"`, `"ideal_zoom"`
  - `zoom_frequency_grid(parent_centers_mhz) -> tuple[np.ndarray, list[tuple[int, int]]]`

**Context:** the existing `src/lusee_faraday/spectrometer.py` loads `data/spectrometer_bin_response.txt` itself. luseepy already owns that response (`lusee.spectrometer_response`, `lusee.spectrometer_response_zoom`, backed by its own normalized data file), so this task deletes our loader and keeps only the binning helpers, which luseepy does not have. The bodies come from `fourport.py`; move them, do not rewrite them.

Zoom bins use **FFT ordering**: bin 0 is the center, 1-32 are positive offsets, 33-63 negative.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from lusee_faraday import channelization as ch


def test_zoom_bin_offsets_use_fft_ordering():
    off = ch.zoom_bin_offsets_hz()
    assert off.shape == (64,)
    assert off[0] == 0.0
    assert np.isclose(off[1], ch.ZOOM_STEP_HZ)
    assert np.isclose(off[63], -ch.ZOOM_STEP_HZ)
    assert np.isclose(off[32], 32 * ch.ZOOM_STEP_HZ)


def test_weights_are_normalized():
    pytest.importorskip("lusee")
    off = np.linspace(-50000.0, 50000.0, 4001)
    assert np.isclose(ch.parent_weights(off).sum(), 1.0)
    assert np.allclose(ch.zoom_weights(off).sum(axis=0), 1.0)
    assert np.allclose(ch.ideal_zoom_weights(off).sum(axis=0), 1.0)


def test_integrating_a_constant_spectrum_returns_the_constant():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)
    waterfall = np.full((2, fine.size, 3), 7.0)
    out = ch.integrate(waterfall, fine, np.array([30.0]))
    assert np.allclose(out["parent"], 7.0, rtol=1e-6)
    assert np.allclose(out["zoom"], 7.0, rtol=1e-6)
    assert np.allclose(out["ideal_zoom"], 7.0, rtol=1e-6)


def test_integrate_shapes():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)
    waterfall = np.zeros((5, fine.size, 16))
    parents = np.array([29.975, 30.0, 30.025])
    out = ch.integrate(waterfall, fine, parents)
    assert out["parent"].shape == (5, 3, 16)
    assert out["zoom"].shape == (5, 3, 64, 16)
    assert out["ideal_zoom"].shape == (5, 3, 64, 16)


def test_integrate_rejects_a_grid_that_does_not_cover_the_response():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(256) - 128) * (25e-3 / 2048)  # +-1.6 kHz
    waterfall = np.zeros((1, fine.size, 1))
    with pytest.raises(ValueError, match="does not cover"):
        ch.integrate(waterfall, fine, np.array([30.0]))


def test_integrate_rejects_a_nonuniform_grid():
    pytest.importorskip("lusee")
    fine = np.concatenate([np.linspace(29.9, 30.0, 100),
                           np.linspace(30.001, 30.1, 100)])
    waterfall = np.zeros((1, fine.size, 1))
    with pytest.raises(ValueError, match="uniform"):
        ch.integrate(waterfall, fine, np.array([30.0]))


def test_zoom_frequency_grid_is_contiguous_and_unique():
    parents = np.array([29.975, 30.0, 30.025])
    freqs, order = ch.zoom_frequency_grid(parents)
    assert freqs.size == 192
    assert len(order) == 192
    assert np.all(np.diff(freqs) > 0)
    steps = np.diff(freqs) * 1e6
    assert np.allclose(steps, ch.ZOOM_STEP_HZ, rtol=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_channelization.py -v
```

Expected: `ModuleNotFoundError: No module named 'lusee_faraday.channelization'`.

- [ ] **Step 3: Delete the old spectrometer module and write the new one**

```bash
git rm src/lusee_faraday/spectrometer.py tests/test_spectrometer.py
```

```python
"""LuSEE spectrometer channelization.

luseepy owns the bin response itself (``lusee.spectrometer_response`` and
``lusee.spectrometer_response_zoom``); what it does not own, and what
lives here, is the binning: which fine channels feed which parent and
zoom bins, and the ideal Gaussian comparison bins.

Zoom bins use FFT ordering -- bin 0 is the parent centre, bins 1-32 are
positive offsets, bins 33-63 negative.  The zoom FFT runs on the
critically sampled 25 kHz parent stream, so the bins carry folded
images; that folding is physical and is removed downstream, not here.
"""

import numpy as np

ZOOM_STEP_HZ = 25000.0 / 64  # 390.625 Hz
FINE_STEP_HZ = 25000.0 / 2048  # 12.20703125 Hz
PARENT_HALF_WIDTH_HZ = 50000.0


def zoom_bin_offsets_hz():
    """Nominal centre offsets of the 64 zoom bins, FFT ordering."""
    k = np.arange(64)
    return np.where(k < 32, k, k - 64) * ZOOM_STEP_HZ


def parent_weights(offsets_hz, notch=0):
    """Normalized parent-bin weights on a fine offset grid."""
    from lusee.SpectrometerResponse import spectrometer_response

    w = spectrometer_response(np.asarray(offsets_hz, dtype=float), notch)
    return w / w.sum()


def zoom_weights(offsets_hz):
    """Normalized zoom-bin weights -> ``(noffsets, 64)``."""
    from lusee.SpectrometerResponse import spectrometer_response_zoom

    off = np.asarray(offsets_hz, dtype=float)
    W = np.stack(
        [spectrometer_response_zoom(off, k) for k in range(64)], axis=-1
    )
    return W / W.sum(axis=0, keepdims=True)


def ideal_zoom_weights(offsets_hz, fwhm_hz=ZOOM_STEP_HZ):
    """Gaussian 'ideal' zoom bins at the nominal centres."""
    off = np.asarray(offsets_hz, dtype=float)
    centers = zoom_bin_offsets_hz()
    sigma = fwhm_hz / (2 * np.sqrt(2 * np.log(2)))
    W = np.exp(-0.5 * ((off[:, None] - centers[None, :]) / sigma) ** 2)
    return W / W.sum(axis=0, keepdims=True)


def integrate(waterfall, fine_freqs_mhz, parent_centers_mhz, notch=0):
    """Convolve a fine-frequency waterfall with the bin responses.

    ``waterfall`` has shape ``(..., nfine, nchan)``.  Returns a dict with
    ``parent`` ``(..., nparent, nchan)``, ``zoom`` and ``ideal_zoom``
    ``(..., nparent, 64, nchan)``.  A parent bin whose +-50 kHz response
    support is not fully covered by the fine grid raises.
    """
    fine = np.asarray(fine_freqs_mhz, dtype=float)
    df = np.diff(fine)
    if not np.allclose(df, df[0], rtol=1e-9):
        raise ValueError("fine frequency grid must be uniform")
    parents, zooms, ideals = [], [], []
    for fc in np.atleast_1d(parent_centers_mhz):
        off = (fine - fc) * 1e6
        sel = np.abs(off) <= PARENT_HALF_WIDTH_HZ + 1e-6
        covered = sel.any() and (
            off[sel].min() <= -PARENT_HALF_WIDTH_HZ + 1e-3
            and off[sel].max() >= PARENT_HALF_WIDTH_HZ - 1e-3
        )
        if not covered:
            raise ValueError(
                f"fine grid does not cover the response of bin {fc} MHz"
            )
        chunk = waterfall[..., sel, :]
        o = off[sel]
        parents.append(
            np.einsum("...fc,f->...c", chunk, parent_weights(o, notch=notch))
        )
        zooms.append(np.einsum("...fc,fz->...zc", chunk, zoom_weights(o)))
        ideals.append(
            np.einsum("...fc,fz->...zc", chunk, ideal_zoom_weights(o))
        )
    return {
        "parent": np.stack(parents, axis=-2),
        "zoom": np.stack(zooms, axis=-3),
        "ideal_zoom": np.stack(ideals, axis=-3),
    }


def zoom_frequency_grid(parent_centers_mhz):
    """Sorted zoom-bin centre frequencies (MHz) and their index map.

    Returns ``(freqs_sorted, order)`` where ``order[i] = (parent_index,
    zoom_bin)`` names the bin at ``freqs_sorted[i]``.  The grid is
    contiguous at ``ZOOM_STEP_HZ`` across adjacent parents.
    """
    offs = zoom_bin_offsets_hz()
    entries = []
    for p, fc in enumerate(np.atleast_1d(parent_centers_mhz)):
        for k in range(64):
            entries.append((fc + offs[k] * 1e-6, p, k))
    entries.sort()
    freqs = np.array([e[0] for e in entries])
    order = [(e[1], e[2]) for e in entries]
    return freqs, order
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_channelization.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/channelization.py tests/test_channelization.py
uv run flake8 src/lusee_faraday/channelization.py
git add -A src/lusee_faraday tests
git commit -m "Move channelization onto luseepy's spectrometer response"
```

---

### Task 14: `response.py` — the symmetric pseudo-dipole arm

**Files:**
- Modify: `src/lusee_faraday/response.py`
- Test: `tests/test_response_two_port.py`

**Interfaces:**
- Consumes: `response.pair_stokes_from_jones` from Task 4.
- Produces:
  - `two_port_jones_from_fits(path, freq_mhz, orientation="y") -> tuple[np.ndarray, np.ndarray]` — `(h_theta, h_phi)`, each `(2, ntheta, nphi)`, ports ordered `(X, Y)`
  - `two_port_pair_alms(h_theta, h_phi, theta_deg, phi_deg, lmax) -> np.ndarray` `(3, 4, L, 2L-1)` for pairs `((0,0), (0,1), (1,1))`
  - `TWO_PORT_PAIRS = ((0, 0), (0, 1), (1, 1))`

**Context:** this is the lineage of the paper's Fig 4 and of `scripts/compare_main_vs_asbuilt.py`'s `MainBeam`: two symmetric pseudo-dipoles, no impedance, no receiver loading, unitless. `lusee._validate_instrument_metadata` hard-requires four ports and the ten ordered pairs (`FullStokesSimulator.py:136`), so this arm cannot go through luseepy's simulator. It goes through `croissant.PairStokesBeam`, which accepts arbitrary `pairs`, and then through the same `engine.contract`.

The Y dipole is the X dipole rolled 90 degrees in phi. Rotating the antenna about z translates the tangent-basis components in phi, because `e_theta` and `e_phi` rotate with the direction — so a roll of the phi axis is the whole operation.

- [ ] **Step 1: Write the failing tests**

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import response as rsp  # noqa: E402


def analytic_short_dipoles(theta_deg, phi_deg):
    """X and Y short dipoles on a theta/phi grid, upper hemisphere only."""
    th = np.radians(theta_deg)[:, None]
    ph = np.radians(phi_deg)[None, :]
    below = np.cos(th) < 0
    hx_t = -np.cos(th) * np.cos(ph) * ~below
    hx_p = np.sin(ph) * np.ones_like(th) * ~below
    hy_t = -np.cos(th) * np.sin(ph) * ~below
    hy_p = -np.cos(ph) * np.ones_like(th) * ~below
    return (
        np.stack([hx_t, hy_t]).astype(complex),
        np.stack([hx_p, hy_p]).astype(complex),
    )


def test_two_port_pair_alms_shape():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    lmax = 8
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, lmax)
    assert alms.shape == (3, 4, lmax + 1, 2 * lmax + 1)
    assert rsp.TWO_PORT_PAIRS == ((0, 0), (0, 1), (1, 1))


def test_short_dipole_autos_have_equal_total_power():
    """X and Y are the same dipole rotated, so their I monopoles match."""
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, 8)
    lmax = 8
    xx = alms[0, 0, 0, lmax]
    yy = alms[2, 0, 0, lmax]
    assert np.isclose(xx, yy, rtol=1e-6)


def test_short_dipole_cross_monopole_vanishes():
    """Orthogonal dipoles have no monopole in their cross response."""
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    lmax = 8
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, lmax)
    xy = alms[1, 0, 0, lmax]
    xx = alms[0, 0, 0, lmax]
    assert abs(xy) < 1e-8 * abs(xx)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_response_two_port.py -v
```

Expected: `AttributeError: module 'lusee_faraday.response' has no attribute 'two_port_pair_alms'`.

- [ ] **Step 3: Append the two-port arm to `response.py`**

```python
TWO_PORT_PAIRS = ((0, 0), (0, 1), (1, 1))


def two_port_jones_from_fits(path, freq_mhz, orientation="y"):
    """Load a 2-port Jones FITS and build the orthogonal pseudo-dipole.

    The file stores only the upper hemisphere on a 1-degree grid with a
    duplicated ``phi = 360`` column.  The lower hemisphere is zero-filled.
    Rotating the antenna about z translates the tangent-basis components
    in phi, so the partner dipole is a roll of the phi axis.
    """
    from astropy.io import fits

    with fits.open(str(path)) as f:
        e_theta = f["Etheta_real"].data + 1j * f["Etheta_imag"].data
        e_phi = f["Ephi_real"].data + 1j * f["Ephi_imag"].data
        idx = int(np.argwhere(f["freq"].data == freq_mhz)[0, 0])
    e_theta = e_theta[idx][..., :-1]
    e_phi = e_phi[idx][..., :-1]
    lower = np.zeros_like(e_theta)[:-1, :]
    e_theta = np.concatenate([e_theta, lower], axis=0)
    e_phi = np.concatenate([e_phi, lower], axis=0)

    if orientation == "y":
        rolls = (270, 0)
    elif orientation == "x":
        rolls = (0, 90)
    else:
        raise ValueError("orientation must be 'x' or 'y'")
    h_theta = np.stack([np.roll(e_theta, r, axis=-1) for r in rolls])
    h_phi = np.stack([np.roll(e_phi, r, axis=-1) for r in rolls])
    return h_theta, h_phi


def two_port_pair_alms(h_theta, h_phi, theta_deg, phi_deg, lmax):
    """Pair-Stokes alms for two pseudo-dipoles -> (3, 4, L, 2L-1).

    Unitless: this arm has no impedance model and no receiver loading,
    so it is the direct analogue of the paper's Fig 4 pipeline rather
    than of the as-built four-port instrument.
    """
    import croissant as cro

    maps = pair_stokes_from_jones(h_theta, h_phi, TWO_PORT_PAIRS)
    beam = cro.PairStokesBeam(
        maps[:, None],  # (pair, freq=1, 4, ntheta, nphi)
        np.array([1.0]),
        TWO_PORT_PAIRS,
        sampling="mwss",
        frame="topo",
    )
    return np.asarray(beam.compute_alm(lmax=int(lmax)))[:, 0]
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_response_two_port.py -v
```

Expected: 3 passed.

If `PairStokesBeam` rejects the grid, the `sampling` argument is the thing to check: `"mwss"` expects a specific theta/phi layout. Print `beam.theta.shape` and compare against the grid actually passed; if they disagree, resample the Jones components onto the sampling croissant expects rather than forcing the argument.

- [ ] **Step 5: Commit**

```bash
uv run black src/lusee_faraday/response.py tests/test_response_two_port.py
uv run flake8 src/lusee_faraday/response.py
git add src/lusee_faraday/response.py tests/test_response_two_port.py
git commit -m "Add the symmetric pseudo-dipole arm through croissant"
```

---

### Task 15: Port `scripts/zenith_weights.py`

**Files:**
- Modify: `scripts/zenith_weights.py`
- Modify: `scripts/common.py`
- Test: `tests/test_zenith_weights_regression.py`

**Interfaces:**
- Consumes: `polarimeter.zenith_vectors`, `response.load_response`, `config`.
- Produces: `scripts.zenith_weights.get_weights(center_mhz, mode="ortho", force=False) -> tuple[np.ndarray, np.ndarray]` — unchanged signature, so the scripts that call it keep working.

**Context:** the current implementation builds a `fourport.FixedFreqKernel` and calls `fourport.zenith_port_weights` / `fourport.orthonormalize_xy`. Replace those three with `polarimeter.zenith_vectors`. Keep the cache file format (`generated_data/cache/zenith_weights_{C}.npz` with keys `x_gains`, `y_gains`, `x_ortho`, `y_ortho`, `C0`), the CLI, and the printed diagnostics exactly as they are — other scripts and the report's Table 1 depend on them.

Also shrink `scripts/common.py`: delete the constants and helpers that moved to `lusee_faraday.config` (`LUN_LAT_DEG`, `LUN_LONG_DEG`, `N_TIMES`, `SIDEREAL_DAY_S`, `T_START_UTC`, `FINE_STEP_MHZ`, `N_FINE`, `MAP_NSIDE`, `BETA_I`, `FREQ_REF_I`, `BETA_QU`, `FREQ_REF_QU`, `T_CMB`, `PHI_FD_POINT`, `times`, `moon_location`, `fine_freqs`, `parent_centers`, `lam2`) and re-export them from the package so existing `from common import ...` lines keep resolving:

```python
from lusee_faraday.config import (  # noqa: F401
    BETA_I,
    BETA_QU,
    FINE_STEP_MHZ,
    FREQ_REF_I,
    FREQ_REF_QU,
    LUN_LAT_DEG,
    LUN_LONG_DEG,
    MAP_NSIDE,
    N_FINE,
    N_TIMES,
    PHI_FD_POINT,
    SIDEREAL_DAY_S,
    T_CMB,
    T_START_UTC,
    fine_freqs,
    lam2,
    moon_location,
    parent_centers,
    times,
)
```

Leave `DATA_DIR`, `GEN_DIR`, `CACHE_DIR`, `FIG_DIR`, `RESPONSE_DIR`, `RESPONSE_PATH`, `rotation_matrices` and `load_sky_maps` in `common.py` — they are analysis plumbing, not package configuration.

- [ ] **Step 1: Write the failing regression test**

```python
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESPONSE = REPO / "data" / "BGL_v16" / "lusee_bgl_v16_response_v3.fits"
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)

pytestmark = pytest.mark.skipif(
    not RESPONSE.exists(), reason="BGL_v16 response artifact not present"
)


@pytest.mark.slow
@pytest.mark.parametrize("center", [30.0, 10.0, 50.0])
def test_ortho_weights_null_zenith_polarization(center):
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp

    resp = rsp.load_response(RESPONSE)
    x, y, C0 = pol.zenith_vectors(resp, JFETReceiver(), center)
    stokes = pol.pseudo_stokes(C0, x, y)
    residual = np.abs(stokes[1:]).max() / stokes[0]
    assert residual < 10 * BASELINES["zenith_null_ortho_max"]["value"]
```

Register the marker in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: needs the 631 MB BGL_v16 response artifact"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_zenith_weights_regression.py -v
```

Expected: fails on the missing `lusee_faraday.polarimeter` import path only if Task 12 was skipped; otherwise it should already pass once `load_response` handles the real artifact. If it errors on the FITS read, that is the thing to fix.

- [ ] **Step 3: Rewrite `get_weights` in `scripts/zenith_weights.py`**

Replace the body between the cache lookup and the `np.savez` call with:

```python
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp

    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()
    x_g, y_g, C0 = pol.zenith_vectors(resp, receiver, center_mhz, "gains")
    x_o, y_o, _ = pol.zenith_vectors(resp, receiver, center_mhz, "ortho")
    del resp
```

Change the module import at the top from `from lusee_faraday import fourport as fp` to nothing (the module no longer needs it). Leave everything below `np.savez(...)` untouched.

- [ ] **Step 4: Recompute the weights and compare against the cached report values**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
ulimit -v 16000000
uv run python scripts/zenith_weights.py --force 2>&1 | tee \
  /home/christian/Documents/research/lusee/lusee_faraday/generated_data/zenith_weights_refactor.log
```

Expected: for each of 10, 30 and 50 MHz the printed ortho vectors reproduce Table 1 of `report/report.tex`, and each band's ortho vectors null its own zenith leakage to ~1e-16. Compare the printed numbers against the table by eye and record any disagreement in the commit message.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_zenith_weights_regression.py -v
uv run pytest tests/ -x -q
```

Expected: the regression test passes for all three bands; the full suite is green.

- [ ] **Step 6: Commit**

```bash
uv run black scripts/zenith_weights.py scripts/common.py \
  tests/test_zenith_weights_regression.py
git add scripts/zenith_weights.py scripts/common.py pyproject.toml \
  tests/test_zenith_weights_regression.py
git commit -m "Port the zenith polarimeter calibration onto the new stack"
```

---

### Task 16: Port the point-source scripts

**Files:**
- Modify: `scripts/step1_point_source.py`
- Modify: `scripts/step1_ionly_source.py`
- Test: `tests/test_point_source_regression.py`

**Interfaces:**
- Consumes: `sky.FaradaySky.point_source`, `response.four_port_pair_alms`, `engine.contract`, `engine.expand`, `instrument.covariance`, `instrument.channels`, `polarimeter`, `channelization.integrate`, `config`.
- Produces: the same `generated_data/*.npz` files with the same keys as today, so the plot scripts need no changes.

**Context:** the physics core changes; the plotting and CLI do not. Today's computation builds a `FixedFreqKernel`, samples it along the source track, and decomposes the source Stokes vector by hand. The replacement builds a one-component `FaradaySky` per source and lets the contraction do the work — which is also where the speedup comes from, since the 16,384 fine channels become an einsum.

**Leave every `def plot*`, `def fig_*` and the `argparse` block exactly as they are.** Only the `compute()` function changes.

- [ ] **Step 1: Write the failing regression test**

```python
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)


def test_faraday_q_oscillation_period_matches_the_published_value():
    """phi_FD = 250 rad/m^2 at 30 MHz gives a 1.89 kHz beat in Q."""
    from lusee_faraday.conventions import lambda_squared

    phi = BASELINES["point_source_phi_fd"]
    # Q returns to itself when 2 phi lambda^2 advances by 2 pi.
    f0 = 30.0
    lam2 = lambda_squared(f0)[0]
    # d(lambda^2)/df = -2 lambda^2 / f, so the period in frequency is
    # delta_f = 2 pi / (2 phi * |d lambda^2 / df|) = pi f / (2 phi lam2).
    period_hz = np.pi * (f0 * 1e6) / (2 * phi * lam2)
    want = BASELINES["q_oscillation_period_khz"]
    assert np.isclose(
        period_hz / 1e3, want["value"], rtol=want["rtol"]
    )


@pytest.mark.slow
def test_polarized_source_track_is_rank_one():
    """A single polarized source must saturate the PSD bound."""
    npz = REPO / "generated_data" / "step1_point_source.npz"
    if not npz.exists():
        pytest.skip("run scripts/step1_point_source.py first")
    from lusee_faraday import polarimeter as pol

    d = np.load(npz)
    stokes = pol.pseudo_stokes_from_channels(d["products"])
    pol.check_psd(stokes)
    above = stokes[..., 0] > 0
    p = np.sqrt((stokes[..., 1:] ** 2).sum(axis=-1))[above]
    assert p.max() / stokes[..., 0][above].max() > 0.9
```

- [ ] **Step 2: Run to verify the analytic one fails, then passes**

```bash
uv run pytest tests/test_point_source_regression.py -v
```

Expected: `test_faraday_q_oscillation_period_matches_the_published_value` passes immediately (it is pure arithmetic on `conventions`); the slow one skips until the script has been run. If the period test fails, `conventions.lambda_squared` or the published 1.89 kHz is wrong — resolve that before touching the scripts.

- [ ] **Step 3: Replace `compute()` in `scripts/step1_point_source.py`**

```python
def compute():
    """Waterfall of the transiting polarized source on the new stack."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import channelization as chan
    from lusee_faraday import config as cfg
    from lusee_faraday import engine, instrument
    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp
    from lusee_faraday.sky import FaradaySky

    lmax = LMAX
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()
    beam = rsp.four_port_pair_alms(resp, CENTER_MHZ, lmax)

    # One source, one Faraday depth, unit polarized intensity.
    sky = FaradaySky.point_source(
        theta=np.array([SOURCE_THETA]),
        phi=np.array([SOURCE_PHI]),
        stokes=np.array([[1.0, -1.0, 0.0]]),
        phi_fd=np.array([cfg.PHI_FD_POINT]),
        nside=SOURCE_NSIDE,
        lmax=lmax,
    )

    times = cfg.times()
    fine = cfg.fine_freqs(CENTER_MHZ)
    W = engine.contract(
        beam, sky.component_alms, times, cfg.moon_location(), lmax
    )
    pair = engine.expand(W, sky.coeffs(fine), chunk=512)
    C = instrument.covariance(pair, resp, receiver, fine)
    products, labels = instrument.channels(C)

    stokes = pol.pseudo_stokes_from_channels(products)
    pol.check_psd(stokes)  # rank-1 source: this must hold on every sample

    binned = chan.integrate(products, fine, cfg.parent_centers(CENTER_MHZ))
    np.savez(
        GEN_DIR / "step1_point_source.npz",
        products=products,
        parent=binned["parent"],
        zoom=binned["zoom"],
        ideal_zoom=binned["ideal_zoom"],
        fine_freqs=fine,
        labels=np.array(labels),
        center_mhz=CENTER_MHZ,
    )
    return products
```

Add near the top of the file, next to `CENTER_MHZ`:

```python
LMAX = 30          # the response's band-limit; higher buys nothing
SOURCE_NSIDE = 32  # pixel 0 of nside=32 sits safely off the pole
SOURCE_THETA = np.radians(90.0 - 45.0)  # source declination on the track
SOURCE_PHI = 0.0
```

**Before running:** confirm `SOURCE_THETA` and `SOURCE_PHI` against the values the current script uses for the transiting source — read them out of the existing `compute()` before you delete it, and carry them over unchanged. The transit track must be the same one the report figures show.

- [ ] **Step 4: Replace `compute()` in `scripts/step1_ionly_source.py`**

Same structure, with the source unpolarized and the result frequency-flat:

```python
def compute():
    """Unpolarized source: pure instrumental leakage along the track."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import config as cfg
    from lusee_faraday import engine, instrument
    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp
    from lusee_faraday.sky import FaradaySky

    lmax = 30
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()
    beam = rsp.four_port_pair_alms(resp, CENTER_MHZ, lmax)

    sky = FaradaySky.point_source(
        theta=np.array([SOURCE_THETA]),
        phi=np.array([SOURCE_PHI]),
        stokes=np.array([[1.0, 0.0, 0.0]]),
        phi_fd=np.array([0.0]),
        nside=SOURCE_NSIDE,
        lmax=lmax,
    )

    times = cfg.times()
    freqs = np.array([CENTER_MHZ])  # frequency-flat by construction
    W = engine.contract(
        beam, sky.component_alms, times, cfg.moon_location(), lmax
    )
    pair = engine.expand(W, sky.coeffs(freqs))
    C = instrument.covariance(pair, resp, receiver, freqs)
    products, _ = instrument.channels(C)
    products = products[:, 0]  # drop the singleton frequency axis

    stokes = pol.pseudo_stokes_from_channels(products)
    pol.check_psd(stokes)

    np.savez(
        GEN_DIR / "step1_ionly_source.npz",
        products=products,
        center_mhz=CENTER_MHZ,
    )
    return products
```

Import `SOURCE_THETA`, `SOURCE_PHI` and `SOURCE_NSIDE` from `step1_point_source` rather than duplicating them, so the two sources cannot drift apart.

- [ ] **Step 5: Run both scripts and regenerate their figures**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
ulimit -v 16000000
uv run python scripts/step1_ionly_source.py 2>&1 | tee \
  /home/christian/Documents/research/lusee/lusee_faraday/generated_data/step1_ionly_refactor.log
uv run python scripts/step1_point_source.py 2>&1 | tee \
  /home/christian/Documents/research/lusee/lusee_faraday/generated_data/step1_point_refactor.log
uv run python scripts/step1_plots.py --calibrated
uv run python scripts/step1_ionly_source.py --calibrated
```

- [ ] **Step 6: Check the numbers against the published ones**

```bash
uv run pytest tests/test_point_source_regression.py -v -m ""
```

Then read the script output and confirm, against `tests/fixtures/regression_baselines.json`:

| Quantity | Expected |
|---|---|
| raw transit leakage, unpolarized source | 0.134 (rtol 2%) |
| ortho transit leakage, unpolarized source | ~7e-4 |
| parent-bin p at transit, polarized source | ~7e-4 (true bandwidth depolarization) |
| zoom recovery, real / ideal | 0.79 / 0.86 (rtol 3%) |

Any disagreement beyond those tolerances is a finding, not a rounding difference — stop and report it rather than adjusting the tolerance.

- [ ] **Step 7: Commit**

```bash
uv run black scripts/step1_point_source.py scripts/step1_ionly_source.py \
  tests/test_point_source_regression.py
git add scripts/step1_point_source.py scripts/step1_ionly_source.py \
  tests/test_point_source_regression.py
git commit -m "Port the point-source analyses onto the component-spectral sky"
```

---

### Task 17: Port `scripts/step_ionly.py` (the leakage reference)

**Files:**
- Modify: `scripts/step_ionly.py`
- Test: `tests/test_ionly_regression.py`

**Interfaces:**
- Consumes: `sky.FaradaySky.i_only`, `response.four_port_pair_alms`, `engine`, `instrument`, `polarimeter`, `channelization`, `config`.
- Produces: `generated_data/real{10,30,50}_ionly.npz` with the same keys as today.

**Context:** this is the analysis the spec expects to become the paper's actual result — `I -> Q,U` leakage is deterministic, dominant, and untouched by the audit. Perfect depolarization means Stokes I only, so `K = 1` and the whole band is frequency-flat by construction: one contraction per band, no fine grid needed.

The `--analyze` mode compares this reference against the full diffuse run. **That full run stays on the legacy pixel arm by decision** (its Faraday content is grid shot noise), so `--analyze` keeps reading whatever `generated_data/real{C}.npz` the legacy `step2_real_sky.py` wrote. Do not port `step2_real_sky.py`.

- [ ] **Step 1: Write the failing regression test**

```python
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)


@pytest.mark.slow
def test_parent_bin_leakage_matches_the_published_stokes_ratio():
    npz = REPO / "generated_data" / "real30_ionly.npz"
    if not npz.exists():
        pytest.skip("run scripts/step_ionly.py --centers 30 first")
    from lusee_faraday import polarimeter as pol

    d = np.load(npz)
    stokes = pol.pseudo_stokes_from_channels(d["products"])
    q = np.median(stokes[..., 1] / stokes[..., 0])
    u = np.median(stokes[..., 2] / stokes[..., 0])
    want = BASELINES["parent_stokes_over_i_30mhz"]
    assert np.isclose(q, want["q"], atol=50 * want["atol"])
    assert np.isclose(u, want["u"], atol=50 * want["atol"])


@pytest.mark.slow
def test_ionly_result_is_frequency_flat():
    """Perfect depolarization has no chromatic structure by construction."""
    npz = REPO / "generated_data" / "real30_ionly.npz"
    if not npz.exists():
        pytest.skip("run scripts/step_ionly.py --centers 30 first")
    d = np.load(npz)
    products = d["products"]
    if products.ndim < 3:
        pytest.skip("already reduced over frequency")
    spread = np.ptp(products, axis=1)
    assert np.abs(spread).max() < 1e-12 * np.abs(products).max()
```

- [ ] **Step 2: Run to verify it skips cleanly**

```bash
uv run pytest tests/test_ionly_regression.py -v -m ""
```

Expected: 2 skipped (the npz do not exist yet).

- [ ] **Step 3: Replace the computation in `scripts/step_ionly.py`**

Keep the CLI, the `--analyze` branch and every plotting function. Replace the per-band computation with:

```python
def compute_band(center_mhz):
    """Perfect-depolarization reference at one band centre."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import config as cfg
    from lusee_faraday import engine, instrument
    from lusee_faraday import response as rsp
    from lusee_faraday.sky import FaradaySky

    lmax = 30
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()
    beam = rsp.four_port_pair_alms(resp, center_mhz, lmax)

    maps = load_sky_maps()
    sky = FaradaySky.i_only(
        maps["I408"],
        lmax=lmax,
        beta_i=cfg.BETA_I,
        ref_freq_i=cfg.FREQ_REF_I,
    )

    times = cfg.times()
    freqs = np.array([center_mhz])  # frequency-flat by construction
    W = engine.contract(
        beam, sky.component_alms, times, cfg.moon_location(), lmax
    )
    pair = engine.expand(W, sky.coeffs(freqs))
    C = instrument.covariance(pair, resp, receiver, freqs)
    products, labels = instrument.channels(C)

    np.savez(
        GEN_DIR / f"real{center_mhz:g}_ionly.npz",
        products=products,
        labels=np.array(labels),
        center_mhz=center_mhz,
    )
    return products
```

`load_sky_maps` stays in `scripts/common.py` and keeps returning the maps at native `nside=512` RING. **Do not degrade them.** The Stokes-I path does not care about the per-pixel Faraday phase, but keeping one loading path for both arms avoids a second convention to maintain.

- [ ] **Step 4: Run all three bands**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
ulimit -v 16000000
uv run python scripts/step_ionly.py --centers 30 10 50 2>&1 | tee \
  /home/christian/Documents/research/lusee/lusee_faraday/generated_data/step_ionly_refactor.log
```

Expected: three npz files, minutes not hours (one contraction per band, no fine grid).

- [ ] **Step 5: Run the regression tests**

```bash
uv run pytest tests/test_ionly_regression.py -v -m ""
```

Expected: 2 passed. The Stokes ratio at 30 MHz should land near `(0.146, -0.032)`.

- [ ] **Step 6: Commit**

```bash
uv run black scripts/step_ionly.py tests/test_ionly_regression.py
git add scripts/step_ionly.py tests/test_ionly_regression.py
git commit -m "Port the I-only leakage reference onto the new stack"
```

---

### Task 18: Retire the old stack

**Files:**
- Delete: `src/lusee_faraday/beam.py`, `sim.py`, `fast_sim.py`, `healpix.py`, `rotations.py`, `utils.py`
- Delete: `tests/test_beam.py`, `test_sim.py`, `test_fast_sim.py`, `test_healpix.py`, `test_rotations.py`, `test_utils.py`
- Rename: `src/lusee_faraday/fourport.py` -> `src/lusee_faraday/pixel_arm.py`
- Rename: `tests/test_fourport.py` -> `tests/testpixel_arm.py`
- Move: `notebooks/faraday_sims.py` -> `notebooks/archive/faraday_sims.py`
- Modify: `src/lusee_faraday/__init__.py`, `tests/conftest.py`
- Modify: `scripts/step2_real_sky.py`, `step2_plots.py`, `step4_power_spectra.py`, `validate_engine.py`, `beam_ablation.py`, `crosscheck_pixel_arm.py`
- Modify: `CLAUDE.md`, `AGENTS.md`, `PROGRESS.md`, `SETUP-CROSSCHECK.md`

**Interfaces:**
- Consumes: everything built so far.
- Produces: the package's final public surface.

**Context:** `notebooks/faraday_sims.py` is the old two-port pipeline behind the paper's current figures. It imports `sim.py`, so retiring the old stack retires it. Archive rather than delete, and record in the commit that it no longer runs. The Fig-4 lineage survives in two places: `scripts/compare_main_vs_asbuilt.py`, which is self-contained (it defines its own `MainBeam` and does not import `lusee_faraday`), and the two-port croissant arm from Task 14.

- [ ] **Step 1: Confirm nothing still imports the doomed modules**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
grep -rn "from lusee_faraday import\|import lusee_faraday\|from \.beam\|from \.sim\|from \.healpix\|from \.rotations\|from \.utils\|from \.fast_sim" \
  src tests scripts notebooks --include=*.py
```

Expected: hits only in `__init__.py`, in the legacy diffuse scripts (which import `fourport`), and in the new modules. Anything else must be fixed before deleting.

- [ ] **Step 2: Delete the old stack**

```bash
git rm src/lusee_faraday/beam.py src/lusee_faraday/sim.py \
  src/lusee_faraday/fast_sim.py src/lusee_faraday/healpix.py \
  src/lusee_faraday/rotations.py src/lusee_faraday/utils.py
git rm tests/test_beam.py tests/test_sim.py tests/test_fast_sim.py \
  tests/test_healpix.py tests/test_rotations.py tests/test_utils.py
mkdir -p notebooks/archive
git mv notebooks/faraday_sims.py notebooks/archive/faraday_sims.py
```

- [ ] **Step 3: Demote the pixel engine to the legacy arm**

```bash
git mv src/lusee_faraday/fourport.py src/lusee_faraday/pixel_arm.py
git mv tests/test_fourport.py tests/testpixel_arm.py
sed -i 's/from lusee_faraday import fourport as fp/from lusee_faraday import pixel_arm as fp/' \
  scripts/step2_real_sky.py scripts/step2_plots.py \
  scripts/step4_power_spectra.py scripts/validate_engine.py \
  scripts/crosscheck_pixel_arm.py
sed -i 's/from lusee_faraday import fourport as fp/from lusee_faraday import pixel_arm as fp/' \
  tests/testpixel_arm.py
grep -rn "fourport" src tests scripts --include=*.py
```

Expected: the final grep returns nothing. `scripts/beam_ablation.py` imports `fp` too — check it and update it the same way if the sed missed it.

Add this banner at the top of `src/lusee_faraday/pixel_arm.py`, above the existing docstring:

```python
"""LEGACY pixel-space four-port engine -- validation arm only.

Superseded by the harmonic path (``response`` + ``engine`` +
``instrument``).  Kept because it is an independent quadrature of the
same integral, which is what makes the cross-check in
``scripts/crosscheck_pixel_arm.py`` meaningful, and because the diffuse
scripts (``step2_real_sky.py``, ``step4_power_spectra.py``) still run on
it -- deliberately, since the 2026-08-18 audit showed their Faraday
content is HEALPix shot noise and they are not headed for the paper.

Production code must not import this module.
"""
```

- [ ] **Step 4: Rewrite the package surface**

```python
__author__ = "Christian Hellum Bye"
__version__ = "0.1.0"

from . import channelization
from . import config
from . import conventions
from . import engine
from . import instrument
from . import plot
from . import polarimeter
from . import response
from .sky import FaradaySky

__all__ = [
    "FaradaySky",
    "channelization",
    "config",
    "conventions",
    "engine",
    "instrument",
    "plot",
    "polarimeter",
    "response",
]
```

If `plot.py` turns out to be unused after the ports, delete it and drop it from both lists:

```bash
grep -rn "lusee_faraday.plot\|from lusee_faraday import plot\|ld\.plot" \
  scripts notebooks tests --include=*.py
```

- [ ] **Step 5: Strip the dead fixtures from `tests/conftest.py`**

The `spec_response`, `spec_response_path`, `short_dipole`, `healpix_grid` and `healpix_grid_full` fixtures all reference deleted classes. Replace the file with:

```python
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR
```

- [ ] **Step 6: Run the whole suite**

```bash
cd /home/christian/Documents/research/lusee/lusee_faraday
uv run pytest tests/ -q
uv run flake8 src/
uv run black --check src/ tests/
```

Expected: green, with the `slow` tests skipped unless the BGL_v16 artifact is present.

- [ ] **Step 7: Update the documentation**

- `CLAUDE.md`: replace the Architecture section. The pipeline is now **Sky components -> Faraday coefficients -> harmonic contraction -> luseepy covariance -> polarimeter -> channelization**. Point the reader at `docs/measurement-model.md` for the model behind it. Describe `FaradaySky`, `response`, `engine`, `instrument`, `polarimeter`, `channelization`; delete `SkyModel`, `Beam`, `Simulator`, `HealpixGrid`, `SpectrometerResponse`, `rotations`. Note that `pixel_arm.py` is a validation arm production code must not import.
- `AGENTS.md`: keep the pinned conventions (they are unchanged) but point them at `lusee_faraday.conventions` and `lusee_faraday.config` as the single source of truth. Update the script inventory: `zenith_weights`, `step1_*` and `step_ionly` are on the new stack; `step2_real_sky` and `step4_power_spectra` are deliberately still on `pixel_arm`.
- `PROGRESS.md`: add a "Refactor onto luseepy + croissant" section recording what moved, what was deleted, the cross-check numbers from Task 7, and the regression results from Tasks 15-17.
- `SETUP-CROSSCHECK.md`: correct the croissant pin (the worktree is at `1c4d6c5`, not `da01c5a`) and record the Task-1 finding about `_low_pass_in_one_step`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Retire the two-port stack; demote the pixel engine to a validation arm

The old simulator (sky/beam/sim/fast_sim/healpix/rotations/utils) is
gone, along with the interp_hp pole artifact and the healpy.Rotator
machinery. fourport.py becomes pixel_arm.py: an independent
quadrature kept for cross-checks and for the diffuse scripts the audit
showed are not headed for the paper.

notebooks/faraday_sims.py is archived rather than ported -- it drove the
paper's current figures through the deleted two-port path. That lineage
survives in scripts/compare_main_vs_asbuilt.py, which is self-contained,
and in the two-port croissant arm in response.py."
```

---

## Notes for whoever executes this

- **Task 7 is a hard gate.** If the data-free contraction test fails, stop. Every later task assumes the contraction is right, and a wrong contraction produces plausible numbers rather than obvious errors.
- **The audit's numbers are not regression targets.** The Steps 2-4 diffuse-Faraday values (`|dP|/I ~ 1.7e-4`, the Step-4 delay spectrum) are grid shot noise and will not reproduce between two different quadratures. That is expected. Do not "fix" it.
- **When a regression tolerance is missed, report it.** The tolerances in `tests/fixtures/regression_baselines.json` were chosen against published values; widening one to make a test pass discards the only evidence the refactor preserved the physics.
- `generated_data/` is gitignored and currently empty. Everything in it is regenerable from the scripts; the fine waterfalls are the expensive part and only the legacy diffuse scripts still produce them.
