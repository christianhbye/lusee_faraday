# Faraday RM-Synthesis — Step 2b (full-band sim driver + run) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the curated full-band channelized simulation (FR + no-FR Stokes vs LST) that feeds Step-3 RM synthesis, using `FrequencyPlan` + the Step-2a speedups + LST tags.

**Architecture:** A testable pipeline function `simulate_channelized` (in `pipeline.py`) wraps the speedups: it runs `compute_vis_fast_parallel` on the plan's deduplicated raw grid and channelizes the FR Stokes (capturing bandwidth depolarization), and evaluates the no-FR Stokes directly at the channel centers (exact for the smooth, un-rotated spectrum — much cheaper than a second full pass). A driver script rotates the sky once, calls the pipeline, tags each time step with its Euler/LST orientation, and saves a single npz. A grid-design script documents/validates the chosen grid via the cheap RMSF.

**Tech Stack:** Python, NumPy, lunarsky, pytest, uv.

**Source spec:** `docs/superpowers/specs/2026-06-08-faraday-rmsynth-design.md`

---

## Verified facts this plan relies on (from prototyping with the real code)

- Chosen grid: 847 wide (30–51.175 MHz, every 25 kHz parent) + 50 zoom (5–29.5 MHz every 0.5 MHz), `decimation={"wide":250,"zoom":10}`, `support=0.999` → `nraw=26475`, `nchan=4047`.
- Est. FR sim: ~14 min on 8 cores (~110 min single-thread). RMSF far sidelobe |R|@φ=1 ≈ 0.046 (inverse-variance weighted).
- No-FR Stokes is smooth in frequency (power-law only), so evaluating at the 4047 channel centers (≈7× cheaper than the 26475-pt grid) is exact to the smooth-spectrum approximation; FR must be channelized because its sub-25-kHz ripple depolarizes.
- `compute_vis_fast` with `rm_topo=0` reproduces the no-FR limit (`cos=1, sin=0`).

## Scope / deferrals

- Produces the data product only. RM synthesis, noise Monte Carlo, signal-aware weighting, and detection-significance plots are **Step 3**.
- The no-FR baseline is included (cheap) as a null/comparison for Step 3.

## File Structure

- Create: `src/lusee_faraday/pipeline.py` — `simulate_channelized` (testable core).
- Test: `tests/test_pipeline.py`.
- Create: `notebooks/grid_design.py` — builds/validates the grid, saves an RMSF figure (documentation).
- Create: `notebooks/faraday_fullband_sim.py` — the driver (rotation + pipeline + LST + save).

Conventions: readability first, sparse comments, Black line-length 79. If `uv run` touches pyproject.toml/uv.lock, `git checkout` them before committing.

---

## Task 1: `pipeline.simulate_channelized`

**Files:** Create `src/lusee_faraday/pipeline.py`, Create `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_pipeline.py`)

Uses the real spectrometer response (tiny plan) but a stub beam + small synthetic topo maps, so it's fast and needs no FITS/sky files. Runs serially (`nproc=1`).

```python
import types
import numpy as np
from lusee_faraday import SpectrometerResponse, FrequencyPlan
from lusee_faraday.pipeline import simulate_channelized

SPEC_PATH = "data/spectrometer_bin_response.txt"


def _setup(rm_value):
    spec = SpectrometerResponse.from_file(SPEC_PATH)
    plan = FrequencyPlan(
        spec, [(30.0, "wide"), (10.0, "zoom")],
        decimation={"wide": 50, "zoom": 10}, support=0.999,
    )
    npix = 64
    ntimes = 2
    rng = np.random.default_rng(0)
    keys = [f"w{s}_{p}" for s in "IQU" for p in ("x", "y", "xy")]
    beam = types.SimpleNamespace(weights={k: rng.normal(size=npix) for k in keys})
    I = rng.uniform(50, 100, (ntimes, npix))
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = np.full((ntimes, npix), rm_value)
    mask = rng.random(npix) > 0.3
    return plan, I, Q, U, rm, beam, mask


def test_output_shapes():
    plan, I, Q, U, rm, beam, mask = _setup(5.0)
    out, table = simulate_channelized(plan, I, Q, U, rm, beam, mask, nproc=1)
    nchan = table["nu"].size  # 1 wide + 64 zoom = 65
    assert nchan == 65
    for key in ("pI_FR", "pQ_FR", "pU_FR", "pI_noFR", "pQ_noFR", "pU_noFR"):
        assert out[key].shape == (2, nchan)


def test_zero_rm_fr_matches_nofr():
    # with RM=0, FR (channelized smooth spectrum) ~ no-FR (channel centers)
    plan, I, Q, U, rm, beam, mask = _setup(0.0)
    out, _ = simulate_channelized(plan, I, Q, U, rm, beam, mask, nproc=1)
    np.testing.assert_allclose(out["pQ_FR"], out["pQ_noFR"], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(out["pU_FR"], out["pU_noFR"], rtol=1e-3, atol=1e-3)


def test_faraday_suppresses_polarization():
    # a large coherent RM depolarizes the wide channel: |P_FR| < |P_noFR|
    plan, I, Q, U, rm, beam, mask = _setup(40.0)
    out, _ = simulate_channelized(plan, I, Q, U, rm, beam, mask, nproc=1)
    p_fr = np.hypot(out["pQ_FR"][:, 0], out["pU_FR"][:, 0])      # wide chan
    p_nofr = np.hypot(out["pQ_noFR"][:, 0], out["pU_noFR"][:, 0])
    assert np.all(p_fr < p_nofr)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (no module `pipeline`).

- [ ] **Step 3: Implement** (`src/lusee_faraday/pipeline.py`)

```python
"""Full-band channelized Faraday simulation.

Runs the optimized visibility computation on a FrequencyPlan's
deduplicated raw grid and channelizes the result. The FR (Faraday)
Stokes are channelized to capture sub-channel bandwidth depolarization;
the no-FR Stokes are smooth in frequency and are evaluated directly at
the channel centers (exact to the smooth-spectrum limit, far cheaper).
"""

import numpy as np

from .fast_sim import compute_vis_fast_parallel
from .sim import Simulator


def simulate_channelized(
    plan, I_topo, Q_topo, U_topo, rm_topo, beam, mask, nproc=None, **kwargs
):
    """Channelized FR and no-FR Stokes for a FrequencyPlan.

    Returns (out, table) where out has keys pI_FR/pQ_FR/pU_FR and
    pI_noFR/pQ_noFR/pU_noFR each shape (ntimes, nchan), and table is
    the plan's channel table. Extra kwargs go to compute_vis_fast.
    """
    sim_freqs = plan.sim_freqs()
    table = plan.channel_table

    # FR: channelize the rippled spectrum (captures depolarization)
    vis_fr = compute_vis_fast_parallel(
        I_topo, Q_topo, U_topo, rm_topo, beam, sim_freqs, mask,
        nproc=nproc, **kwargs,
    )
    I_fr, Q_fr, U_fr = Simulator.compute_stokes(vis_fr)
    out = {
        "pI_FR": plan.channelize(I_fr),
        "pQ_FR": plan.channelize(Q_fr),
        "pU_FR": plan.channelize(U_fr),
    }

    # no-FR: smooth spectrum, evaluate at channel centers (rm = 0)
    zeros = np.zeros_like(rm_topo)
    vis_nf = compute_vis_fast_parallel(
        I_topo, Q_topo, U_topo, zeros, beam, table["nu"], mask,
        nproc=nproc, **kwargs,
    )
    I_nf, Q_nf, U_nf = Simulator.compute_stokes(vis_nf)
    out["pI_noFR"] = I_nf
    out["pQ_noFR"] = Q_nf
    out["pU_noFR"] = U_nf
    return out, table
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): simulate_channelized FR (channelized) + no-FR (centers)"
```

---

## Task 2: Grid-design script (documentation + validation)

**Files:** Create `notebooks/grid_design.py`

- [ ] **Step 1: Write the script**

```python
# notebooks/grid_design.py
"""Full-band grid design: build the FrequencyPlan, report its size/cost
and the inverse-variance-weighted RMSF, and save a figure. The chosen
grid (wide 30-51 MHz + zoom 5-30 MHz) is what faraday_fullband_sim.py
uses via fullband_specs().
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lusee_faraday import SpectrometerResponse, FrequencyPlan, rmsynth, utils

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
DECIMATION = {"wide": 250, "zoom": 10}
SUPPORT = 0.999


def fullband_specs():
    """Wide bins 30-51 MHz (every parent) + zoom 5-30 MHz every 0.5 MHz."""
    f = utils.freqs_lusee()
    wide = [(c, "wide") for c in f[(f >= 30) & (f <= 51.175)]]
    zoom = [(c, "zoom") for c in f[(f >= 5) & (f <= 29.5)][::20]]
    return wide + zoom


def main():
    spec = SpectrometerResponse.from_file(DATA / "spectrometer_bin_response.txt")
    specs = fullband_specs()
    plan = FrequencyPlan(spec, specs, decimation=DECIMATION, support=SUPPORT)
    table = plan.channel_table
    nu, dnu, lam2 = table["nu"], table["dnu"], table["lambda2"]

    n_wide = sum(1 for _, m in plan.specs if m == "wide")
    n_zoom = sum(1 for _, m in plan.specs if m == "zoom")
    print(f"specs: {n_wide} wide + {n_zoom} zoom")
    print(f"nraw (sim freqs): {plan.sim_freqs().size}")
    print(f"nchan: {nu.size}")
    print(f"est FR sim @8 cores: ~{100 * plan.sim_freqs().size * 2.5e-3 / 8 / 60:.0f} min")

    # inverse-variance-weighted RMSF (representative T_sys power law)
    Tsys = 3000.0 * (nu / 50.0) ** -2.55 + 2.725
    weights = dnu / Tsys ** 2
    phi = np.arange(-60, 60, 0.02)
    R = np.abs(rmsynth.rmsf(lam2, phi, weights=weights))
    print(f"RMSF far sidelobe |R|@phi=1: "
          f"{np.abs(rmsynth.rmsf(lam2, np.array([1.0]), weights=weights))[0]:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].plot(nu, lam2, ".", ms=2)
    ax[0].set(title="channel lambda^2 coverage", xlabel="nu [MHz]",
              ylabel="lambda^2 [m^2]", yscale="log")
    ax[1].plot(phi, R)
    ax[1].set(title="inverse-variance RMSF", xlabel="phi [rad/m^2]",
              yscale="log")
    fig.tight_layout()
    out = RES / "grid_design.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python notebooks/grid_design.py`
Expected: prints `847 wide + 50 zoom`, `nraw (sim freqs): 26475`, `nchan: 4047`, `~14 min`, RMSF far sidelobe ~0.046; saves `notebooks/results/grid_design.png`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/grid_design.py
git commit -m "feat: full-band grid-design script (RMSF + cost)"
```

---

## Task 3: Full-band sim driver

**Files:** Create `notebooks/faraday_fullband_sim.py`

- [ ] **Step 1: Write the driver**

```python
# notebooks/faraday_fullband_sim.py
"""Curated full-band Faraday simulation.

Rotates the sky once per time step, runs the channelized FR + no-FR
pipeline on the full-band FrequencyPlan, tags each step with its
Galactic->topocentric Euler angles (LST), and saves one npz for
Step-3 RM synthesis.
"""

import time as pytime
from pathlib import Path

import astropy.units as u
import numpy as np
from lunarsky import Time, MoonLocation

import lusee_faraday as ld
from lusee_faraday import SpectrometerResponse, FrequencyPlan
from lusee_faraday.fast_sim import precompute_rotated_maps
from lusee_faraday.pipeline import simulate_channelized
from lusee_faraday.rotations import topo_euler_angles
from lusee_faraday.sky import LUSEE_LOC
from grid_design import fullband_specs, DECIMATION, SUPPORT

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
NSIDE = 128
N_TIMES = 100
NPROC = 8
BEAM_FILE = DATA / "hfss_lbl_3m_75deg.2port.fits"


def main():
    # time grid: one lunar sidereal day
    loc = MoonLocation(lat=-23.813, lon=182.258)
    t0 = Time("2027-01-01T09:00:00", location=loc)
    times = np.linspace(t0, t0 + 655.720 * 3600 * u.s, num=N_TIMES,
                        endpoint=False)

    # sky + beam + plan
    I_ref = np.load(DATA / "haslam_galactic.npz")["m"]
    wmap = ld.sky.load_wmap(DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits",
                            nside=NSIDE)
    Q_ref, U_ref = wmap[1], wmap[2]
    rm_gal = ld.sky.load_rm(DATA / "faraday2020v2.hdf5")

    beam = ld.Beam.from_file(BEAM_FILE, frequency=30, nside=NSIDE)
    beam.precompute_weights()
    mask = ld.HealpixGrid(NSIDE, horizon=True).mask

    spec = SpectrometerResponse.from_file(DATA / "spectrometer_bin_response.txt")
    plan = FrequencyPlan(spec, fullband_specs(), decimation=DECIMATION,
                         support=SUPPORT)

    print(f"rotating {N_TIMES} time steps...")
    t = pytime.time()
    I_t, Q_t, U_t, rm_t = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm_gal, times, NSIDE, LUSEE_LOC)
    print(f"  rotations done in {(pytime.time()-t)/60:.1f} min")

    print(f"simulating ({plan.sim_freqs().size} freqs, nproc={NPROC})...")
    t = pytime.time()
    out, table = simulate_channelized(
        plan, I_t, Q_t, U_t, rm_t, beam, mask, nproc=NPROC)
    print(f"  sim done in {(pytime.time()-t)/60:.1f} min")

    euler = topo_euler_angles(times, LUSEE_LOC)
    modes = np.array([m for _, m in plan.specs])
    times_jd = np.array([tt.jd for tt in times])

    outfile = RES / "faraday_fullband.npz"
    np.savez(
        outfile,
        nu=table["nu"], lambda2=table["lambda2"], dnu=table["dnu"],
        modes=modes, times_jd=times_jd, euler=euler,
        **out,
    )
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the driver wiring without the full run**

Confirm imports and the grid resolve (does not run the sim):

Run: `cd notebooks && uv run python -c "import faraday_fullband_sim as d; print(d.fullband_specs.__module__, len(d.fullband_specs()))"`
Expected: prints `grid_design 897` with no import error. (Run from the `notebooks/` dir so `from grid_design import ...` resolves.)

- [ ] **Step 3: Commit**

```bash
git add notebooks/faraday_fullband_sim.py
git commit -m "feat: full-band Faraday sim driver (FrequencyPlan + parallel + LST)"
```

---

## Task 4: Run the sim and verify outputs

**Files:** none (produces `notebooks/results/faraday_fullband.npz`)

- [ ] **Step 1: Run the full simulation**

Run: `cd notebooks && uv run python faraday_fullband_sim.py`
Expected: prints rotation time (~2–3 min) and sim time (~15–20 min on 8 cores); saves `notebooks/results/faraday_fullband.npz`. (If on fewer cores it takes proportionally longer — that's fine.)

- [ ] **Step 2: Verify the output**

Run:
```bash
cd notebooks && uv run python -c "
import numpy as np
d = np.load('results/faraday_fullband.npz')
print({k: d[k].shape for k in d.files})
assert d['nu'].shape == (4047,)
assert d['pQ_FR'].shape == (100, 4047)
assert d['euler'].shape == (100, 3)
for k in ('pI_FR','pQ_FR','pU_FR','pI_noFR','pQ_noFR','pU_noFR'):
    assert np.isfinite(d[k]).all(), k
# FR depolarizes relative to no-FR somewhere in the polarized signal
pfr = np.hypot(d['pQ_FR'], d['pU_FR'])
pnf = np.hypot(d['pQ_noFR'], d['pU_noFR'])
print('median P_FR/P_noFR =', np.median(pfr[pnf>0]/pnf[pnf>0]))
print('OK')
"
```
Expected: prints the shapes, a median depolarization ratio < 1, and `OK` with no assertion error.

- [ ] **Step 3: Report**

Report the printed shapes, the sim wall-time, and the median `P_FR/P_noFR` ratio (the beam-averaged depolarization). These are the inputs Step 3 will run RM synthesis on. (The npz is a large artifact — do not commit it; confirm it is gitignored.)

---

## Self-Review

**Spec coverage (Step-2b slice):** Curated full-band sim with FR + no-FR channelized Stokes vs LST → Tasks 1+3+4; grid documented/validated → Task 2; LST tags stored → Task 3. RM synthesis / noise MC / weighting / significance are Step 3 (out of scope). Truncation + per-mode decimation + masking + parallelism all come from Step 2a.

**Placeholder scan:** Every code step has complete code; commands have expected output. Task 4 step 3 is a reporting action (no code), explicitly framed as such.

**Type consistency:** `simulate_channelized(plan, I_topo, Q_topo, U_topo, rm_topo, beam, mask, nproc=None, **kwargs)` returns `(out_dict, table)`; `out` keys `pI_FR/pQ_FR/pU_FR/pI_noFR/pQ_noFR/pU_noFR`; the driver saves those plus `nu/lambda2/dnu/modes/times_jd/euler`. `fullband_specs`, `DECIMATION`, `SUPPORT` are defined in `grid_design.py` and imported by the driver (single source of truth for the grid). `Simulator.compute_stokes`, `compute_vis_fast_parallel`, `precompute_rotated_maps`, `topo_euler_angles`, `FrequencyPlan.channel_table`/`channelize`/`sim_freqs` all match their definitions.
```
