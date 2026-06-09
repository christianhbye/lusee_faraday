# Faraday RM-Synthesis — Step 3 (analysis + detection significance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run model-independent RM synthesis on the full-band sim, quantify the Faraday detection significance vs integration time (inverse-variance baseline + signal-aware weighting variant), and produce the analysis figures.

**Architecture:** A small tested `detection` helper (analytic Faraday-spectrum noise level + a Monte-Carlo cross-check) added to `src/`, reusing `rmsynth` + `noise`. A `notebooks/rmsynth_analysis.py` script loads `faraday_fullband.npz`, runs RM synthesis on the FR vs no-FR data, sweeps integration time, and saves figures.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, uv.

**Source spec:** `docs/superpowers/specs/2026-06-08-faraday-rmsynth-design.md`. **Input:** `notebooks/results/faraday_fullband.npz` (keys: nu, lambda2, dnu, modes, times_jd, euler, p{I,Q,U}_{FR,noFR} each (100, 4047)).

## Key analysis choices (documented; flagged for user review)

- **Detection statistic:** peak of the weighted Faraday spectrum `|F(phi)|` of the FR complex polarization, vs the analytic noise level `sigma_F = sqrt(sum (w_k sigma_k)^2)/sum w_k` (per-quadrature; `|F|` noise is Rayleigh with this scale). `SNR = |F|_peak / sigma_F`. The phi-max look-elsewhere effect is noted, not corrected (conservative).
- **Per-channel noise:** `sigma_k = T_sys,k / sqrt(dnu_k * dt)` with `T_sys ≈ pI_FR` (sky-dominated). Q and U each get independent noise.
- **Integration time:** each of the 100 LST steps is treated as an independent `dt` integration; SNR reported vs `dt in {40 s, 10 min, 1 h}` at representative times (galaxy-up + median over LST). Full-sidereal-day coherent coadd is Option 5 (pinned) — noted as further upside, not done here.
- **Weighting:** inverse-variance `w = 1/sigma^2` (baseline) and signal-aware `w = |P_noFR| / sigma^2` (variant; uses the un-rotated polarized amplitude as a model-independent signal proxy).

## File Structure
- Create: `src/lusee_faraday/detection.py` — `faraday_noise_std`, `faraday_snr`.
- Test: `tests/test_detection.py`.
- Create: `notebooks/rmsynth_analysis.py` — analysis + figures (new name; does NOT touch the legacy `faraday_analysis.ipynb`).

Conventions: readability first, sparse comments, Black line-length 79. If `uv run` touches pyproject.toml/uv.lock, `git checkout` them before committing.

---

## Task 1: `detection` helper

**Files:** Create `src/lusee_faraday/detection.py`, `tests/test_detection.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_detection.py`)

```python
import numpy as np
from lusee_faraday import rmsynth
from lusee_faraday.detection import faraday_noise_std, faraday_snr


def test_noise_std_matches_monte_carlo():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    sigma = np.full(lam2.size, 2.0)
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    analytic = faraday_noise_std(sigma)
    rng = np.random.default_rng(0)
    reals = []
    for _ in range(400):
        q = rng.normal(scale=sigma)
        u = rng.normal(scale=sigma)
        F = rmsynth.faraday_spectrum(q, u, lam2, phi)[0]
        reals.append(F.real)
    mc = np.std(np.array(reals))  # per-quadrature std, phi-averaged
    assert abs(analytic - mc) / analytic < 0.1


def test_noise_std_inverse_variance_formula():
    sigma = np.array([1.0, 2.0, 4.0])
    w = 1.0 / sigma ** 2
    # inverse-variance: sigma_F = 1/sqrt(sum 1/sigma^2)
    assert np.isclose(
        faraday_noise_std(sigma, weights=w),
        1.0 / np.sqrt(np.sum(1.0 / sigma ** 2)),
    )


def test_faraday_snr_high_for_clean_signal():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 300))
    p = np.exp(2j * 6.0 * lam2)  # |P|=1 at RM=6
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    sigma = np.full(lam2.size, 0.01)  # low noise
    snr, peak, nstd = faraday_snr(p.real, p.imag, lam2, sigma, phi)
    assert snr > 20
    assert peak > 0.9  # |F| peak ~ |P| for a single RM


def test_faraday_snr_order_unity_for_pure_noise():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 300))
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    sigma = np.full(lam2.size, 1.0)
    rng = np.random.default_rng(1)
    snr, _, _ = faraday_snr(
        rng.normal(scale=sigma), rng.normal(scale=sigma),
        lam2, sigma, phi,
    )
    assert snr < 6  # noise-only peak is a few sigma (look-elsewhere)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_detection.py -v`
Expected: FAIL (no module `detection`).

- [ ] **Step 3: Implement** (`src/lusee_faraday/detection.py`)

```python
"""Faraday-spectrum detection statistics.

For per-channel Stokes noise std sigma_k and weights w_k, the weighted
Faraday spectrum F(phi) = sum_k (w_k/sum w) (Q+iU)_k exp(-2i phi dl2_k).
With independent Gaussian noise on Q and U, the noise on F is complex
with per-quadrature std sigma_F = sqrt(sum (w_k sigma_k)^2) / sum w_k
(phi-independent). SNR = |F|_peak / sigma_F.
"""

import numpy as np

from .rmsynth import faraday_spectrum


def faraday_noise_std(sigma, weights=None):
    """Per-quadrature noise std of the weighted Faraday spectrum."""
    sigma = np.asarray(sigma, dtype=float)
    if weights is None:
        weights = np.ones_like(sigma)
    w = np.asarray(weights, dtype=float)
    return np.sqrt(np.sum((w * sigma) ** 2)) / w.sum()


def faraday_snr(Q, U, lam2, sigma, phi, weights=None):
    """SNR of the Faraday-spectrum peak vs the analytic noise level.

    Returns (snr, peak, noise_std).
    """
    F = faraday_spectrum(Q, U, lam2, phi, weights=weights)[0]
    peak = float(np.abs(F).max())
    noise_std = faraday_noise_std(sigma, weights=weights)
    return peak / noise_std, peak, noise_std
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_detection.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/detection.py tests/test_detection.py
git commit -m "feat(detection): Faraday-spectrum noise std and peak SNR"
```

---

## Task 2: Expose `detection` in init

**Files:** Modify `src/lusee_faraday/__init__.py`

- [ ] **Step 1:** add `from . import detection` with the other submodule imports.
- [ ] **Step 2:** verify `uv run python -c "import lusee_faraday as ld; print(ld.detection.faraday_noise_std([1.0,2.0]))"` prints a number.
- [ ] **Step 3:** commit `git add src/lusee_faraday/__init__.py && git commit -m "feat: expose detection in package init"`.

---

## Task 3: Analysis script + figures

**Files:** Create `notebooks/rmsynth_analysis.py`

- [ ] **Step 1: Write the script**

```python
# notebooks/rmsynth_analysis.py
"""Step-3 analysis: RM synthesis + Faraday detection significance.

Loads the full-band sim, runs model-independent RM synthesis on the FR
vs no-FR polarization, sweeps integration time for the detection SNR
(inverse-variance and signal-aware weighting), and saves figures.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lusee_faraday import rmsynth, noise, detection

RES = Path(__file__).resolve().parent / "results"
DT_CASES = [(40.0, "40 s"), (600.0, "10 min"), (3600.0, "1 h")]


def main():
    d = np.load(RES / "faraday_fullband.npz")
    nu, lam2, dnu = d["nu"], d["lambda2"], d["dnu"]
    pQ_FR, pU_FR = d["pQ_FR"], d["pU_FR"]
    pI_FR = d["pI_FR"]
    pQ_nf, pU_nf = d["pQ_noFR"], d["pU_noFR"]

    # galaxy-up time = max beam-weighted intensity
    i_gal = int(np.argmax(pI_FR.mean(axis=1)))
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)

    # --- Faraday spectra (FR vs no-FR) at galaxy-up, inverse-var weights ---
    Tsys = pI_FR[i_gal]
    sig_ref = noise.radiometer_sigma(Tsys, dnu, 3600.0)
    w_iv = 1.0 / sig_ref ** 2
    F_fr = rmsynth.faraday_spectrum(pQ_FR[i_gal], pU_FR[i_gal], lam2, phi, w_iv)[0]
    F_nf = rmsynth.faraday_spectrum(pQ_nf[i_gal], pU_nf[i_gal], lam2, phi, w_iv)[0]

    print(f"galaxy-up t={i_gal}")
    print(f"  FR Faraday peak at phi={phi[np.argmax(np.abs(F_fr))]:.3f} rad/m^2")
    print(f"  noFR peak at phi={phi[np.argmax(np.abs(F_nf))]:.3f} rad/m^2")
    print(f"  median P_FR/P_noFR (all t,chan): "
          f"{np.median(np.hypot(pQ_FR, pU_FR)[np.hypot(pQ_nf, pU_nf) > 0] / np.hypot(pQ_nf, pU_nf)[np.hypot(pQ_nf, pU_nf) > 0]):.3f}")

    # --- SNR vs integration time, both weightings, at galaxy-up ---
    print("  SNR vs integration time (galaxy-up):")
    snr_table = {}
    for dt, label in DT_CASES:
        sig = noise.radiometer_sigma(Tsys, dnu, dt)
        w_iv = 1.0 / sig ** 2
        p_nf = np.hypot(pQ_nf[i_gal], pU_nf[i_gal])
        w_sa = p_nf / sig ** 2  # signal-aware
        snr_iv = detection.faraday_snr(pQ_FR[i_gal], pU_FR[i_gal], lam2, sig, phi, w_iv)[0]
        snr_sa = detection.faraday_snr(pQ_FR[i_gal], pU_FR[i_gal], lam2, sig, phi, w_sa)[0]
        snr_table[label] = (snr_iv, snr_sa)
        print(f"    {label:>7}: SNR_invvar={snr_iv:6.1f}  SNR_sigaware={snr_sa:6.1f}")

    # --- figures ---
    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    ax[0].plot(phi, np.abs(F_fr), label="FR")
    ax[0].plot(phi, np.abs(F_nf), label="no FR", ls="--")
    ax[0].set(title=f"Faraday spectrum (galaxy-up t={i_gal})",
              xlabel="phi [rad/m^2]", ylabel="|F|", yscale="log")
    ax[0].legend()

    labels = [l for _, l in DT_CASES]
    iv = [snr_table[l][0] for l in labels]
    sa = [snr_table[l][1] for l in labels]
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, iv, 0.4, label="inverse-variance")
    ax[1].bar(x + 0.2, sa, 0.4, label="signal-aware")
    ax[1].axhline(5, color="k", ls=":", lw=1, label="SNR=5")
    ax[1].set(title="Faraday detection SNR vs integration time (galaxy-up)",
              ylabel="SNR", xticks=x, xticklabels=labels)
    ax[1].legend()

    p_fr = np.hypot(pQ_FR, pU_FR)
    p_nf = np.hypot(pQ_nf, pU_nf)
    ratio = np.divide(p_fr, p_nf, out=np.full_like(p_fr, np.nan), where=p_nf > 0)
    ax[2].plot(nu, np.nanmedian(ratio, axis=0), ".", ms=2)
    ax[2].set(title="median depolarization P_FR/P_noFR vs frequency",
              xlabel="nu [MHz]", ylabel="P_FR/P_noFR")

    fig.tight_layout()
    out = RES / "rmsynth_analysis.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python notebooks/rmsynth_analysis.py`
Expected: prints galaxy-up index, FR/noFR Faraday-peak phi, median depolarization, the SNR-vs-dt table for both weightings; saves `notebooks/results/rmsynth_analysis.png`. Report the printed numbers.

- [ ] **Step 3: Commit**

```bash
git add notebooks/rmsynth_analysis.py
git commit -m "feat: step-3 RM-synthesis analysis + detection-significance figures"
```

---

## Self-Review
**Spec coverage:** model-independent RM synthesis on the sim → Task 3; radiometer-noise detection significance vs integration time → Tasks 1+3; inverse-variance + signal-aware weighting → Task 3; reuse rmsynth/noise → yes; figures → Task 3. Full-day coherent coadd is Option 5 (pinned), noted not done.
**Placeholder scan:** complete code in every step; commands have expected output.
**Type consistency:** `faraday_noise_std(sigma, weights=None)`, `faraday_snr(Q,U,lam2,sigma,phi,weights=None)->(snr,peak,noise_std)`; analysis uses verified npz keys and `noise.radiometer_sigma`, `rmsynth.faraday_spectrum/phi_grid`.
```
