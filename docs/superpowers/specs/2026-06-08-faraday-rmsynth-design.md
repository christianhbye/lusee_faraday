# Faraday RM-Synthesis Detection Pipeline — Design

**Date:** 2026-06-08
**Status:** approved design, pending spec review

## Goal

Improve LuSEE Night's sensitivity to Galactic Faraday rotation by replacing
the current visual/per-band FFT inspection with a **model-independent
rotation-measure (RM) synthesis estimator** that coherently combines the full
frequency coverage, fed by a **flexible zoom/wide channelization** that matches
the spectrometer's real bins. The pipeline must run on a workstation
(8 cores, ~11 GB RAM) and reflect that this is a forecast for a *spectrometer*
— channelization is not optional.

## Background: why current sensitivity is poor

The observable is the complex polarization rotating in wavelength squared:

    P(λ²) = Q + iU = P₀ · exp(2i(χ₀ + RM·λ²)),   λ² = (c/ν)²

This is a pure phase that winds linearly in λ². Three issues today:

1. **Channelization mismatch.** The Faraday ripple period in frequency is
   `Δν = π ν³ / (2 RM c²)`. For RM≈10–20 it is ~1–2 kHz at 10 MHz,
   ~25–47 kHz at 30 MHz, ~110–220 kHz at 50 MHz. A 25 kHz wide bin therefore
   *washes out* the signal at 10 MHz (14–29 cycles per bin) while a 25 kHz
   *zoom window* is *too narrow to hold a full cycle* at 50 MHz. The right
   resolution is frequency dependent: zoom at low ν, wide spans at high ν.
2. **Sub-optimal combination.** The notebooks take `|FFT(Q)|` and `|FFT(U)|`
   separately, per center frequency. This discards the joint Q↔U phase
   coherence (the whole signal *is* the rotation), the sign of RM, and the
   huge λ² lever arm from combining bands.
3. **Beam depolarization.** The wide dipole beam averages many sightlines with
   σ_RM≈20, spreading the Faraday response over tens of rad/m². This is a real
   sky effect; the estimator must combine coherently to fight it.

RM synthesis is the matched transform for a linear-in-λ² phase and directly
addresses (1) and (2). It is also cheap (see Compute Budget).

## Architecture: the channel table

Everything keys off a flat **channel table** — one row per output spectrometer
channel, independent of whether it came from a zoom or a wide bin:

    nu      (center frequency, MHz)        shape (nchan,)
    lambda2 ((c/nu)², m²)                  shape (nchan,)
    dnu     (channel bandwidth, Hz)        shape (nchan,)   # 390.6 zoom, 25000 wide
    weight  (inverse-variance, optional)   shape (nchan,)

The forward sim fills `Q, U` of shape `(ntimes, nchan)` against this table; RM
synthesis consumes exactly this table. This makes zoom-vs-wide a per-channel
switch rather than a rewrite, and unifies sim and analysis.

## New modules under `src/lusee_faraday/`

### `rmsynth.py` — the estimator (pure, testable functions)

- `lambda2(nu_mhz)` → λ² in m².
- `phi_grid(lambda2)` → recommended φ grid plus reported diagnostics:
  resolution `δφ = 2√3 / Δλ²`, `φ_max` (largest detectable |RM| before a
  single channel decorrelates, set by the finest `dnu`), and the largest
  recoverable Faraday-thick scale (set by min λ²).
- `rmsf(lambda2, weights, phi)` → the rotation-measure spread function
  `R(φ) = Σ wₖ e^{-2iφλ²ₖ} / Σ wₖ`. **This is the Step-0 calibration output**
  (main-lobe FWHM and sidelobes from a given λ² coverage).
- `faraday_spectrum(Q, U, lambda2, weights, phi)` → complex Faraday spectrum
  `F(φ, t) = Σₖ wₖ (Q+iU)ₖ e^{-2iφ(λ²ₖ − λ²_ref)} / Σ wₖ`, shape `(ntimes, nphi)`.
  `λ²_ref` (e.g. mean λ²) is subtracted for numerical conditioning.

RM-CLEAN is explicitly **out of scope** for now (YAGNI). Inverse-variance
weights are the matched weighting and come from `noise.py`.

### `noise.py` — radiometer noise (formalize the inline notebook code)

- `radiometer_sigma(T_sys, dnu_hz, dt_s)` → `T_sys / √(dnu · dt)`, with
  `T_sys ≈ Stokes I` (sky dominated). Vectorized over the channel table.
- `add_noise(stokes, sigma, rng)` → one Gaussian realization (independent on
  Q and U). Detection significance is built by re-running the (free) RM
  synthesis transform over many draws.

## Simulation infrastructure (mechanical)

### `FrequencyPlan` — the zoom/wide switch + smart raw grid

A specification of a heterogeneous frequency grid as a list of per-parent
specs `(center, mode ∈ {zoom, wide})`, snapped to the LuSEE 2048-channel grid.
Responsibilities:

- `.sim_freqs()` → the **minimal deduplicated** raw absolute-frequency grid to
  forward-sim. Built by:
  - **Truncating** each channel's spectrometer response to its significant
    support. Confirmed from `spectrometer_bin_response.txt` (10001 pts, ±50 kHz,
    10 Hz step = 4× the 25 kHz parent width): wide carries 99.9% within
    ±16.6 kHz (3315 pts); zoom center sub-bins within ±3.4 kHz (~675 pts at
    99%), edge sub-bins within ±13 kHz; union of 64 zoom sub-bins 99.9% within
    ±18.3 kHz (3663 pts). Truncation threshold is configurable (default 99.9%)
    and cuts raw points per channel ~3–15×.
  - **Deduplicating overlap**: significant support (±~17 kHz ≈ 1.4 parents)
    overlaps when tiled contiguously, so a single shared raw grid is built per
    contiguous region instead of redundant per-parent windows (~2.7× saving in
    wide regions).
- `.channelize(raw_stokes)` → applies the truncated `apply_narrow` / `apply_wide`
  weights per parent as a (sparse) matrix mapping shared raw points → channels,
  with linear interpolation when response offsets don't land on the shared grid
  (response is smooth; interpolation error to be verified < truncation error).
  Returns flat `(ntimes, nchan)` Stokes plus the channel table.

`faraday_sims.py` is refactored to: build a `FrequencyPlan` → forward-sim on
`plan.sim_freqs()` (the `fast_sim` core is untouched) → `plan.channelize()` →
save the channel table, `Q,U(t,chan)`, and LST tags. The current ad-hoc
per-band zoom/narrow/wide handling is replaced by the plan.

### LST tagging (groundwork for future coadd)

- Parameterize `N_TIMES` / time sampling (currently a hard constant).
- Per time step store: `times_jd` (already present), a sidereal phase over the
  lunar sidereal day, and the Euler angles already computed by
  `rotmat_to_eulerZYX` (the true sky orientation). No analysis now — this only
  enables coherent per-LST coadd later (pinned option 5).

## Repo organization (attic)

Assess and, where clearly unused, move legacy material into an `attic/` so the
working repo is easy to understand:

- Superseded notebooks (e.g. `old-notebooks/`, exploratory `*_plots*`,
  `wmap-time`, `point_source-LN`, duplicate sim notebooks) — confirm with the
  user before moving anything referenced by current results.
- Unused source modules (e.g. `sim.py` if `fast_sim.py` fully supersedes it —
  verify first), stale build artifacts (`build/`, coverage files).
- Legacy/large data not needed for the current pipeline.

This is a **review-then-move** step, not a blind sweep; nothing referenced by
the active pipeline or current `results/` is moved without confirmation.

## Execution plan (staged; "calibrate from data first")

0. **Calibration notebook (no new sim).** Build `rmsynth.py`; run RM synthesis
   on existing `results/faraday_sim_{10,30,50}mhz.npz` zoom spectra; inspect
   `F(φ,t)` and the RMSF sidelobes → **decide the adaptive grid from evidence.**
1. **Mechanical code.** `FrequencyPlan` (truncation + dedup/resampling),
   `noise.py`, LST tagging, with unit tests.
2. **One curated full-band sim** (~20–40 min parallel; grid chosen in step 0).
3. **Analysis notebook.** RM synthesis + noise Monte Carlo → detection
   significance vs integration time → paper plots.
4. **Repo organization** (attic) — can run alongside, after step 1 settles.

## Compute budget (measured on this machine)

- Inner loop is cos/sin bound: ~4.7 ms per (time, frequency) over 196k pixels.
- Forward sim scales linearly with total raw points × times and is
  embarrassingly parallel over the 100 time steps:
  - current 3 bands (1.3k raw): ~11 min / ~1.5 min on 8 cores
  - adaptive ~20 zoom + ~840 wide (21k raw): ~170 min / **~20–25 min** on 8 cores
  - naïve full native zoom 1–51 MHz (2.0M raw): ~280 hr — **infeasible**
- RAM peak ~1 GB (rotated maps `(100,npix)×4` = 0.63 GB held; batch 0.15 GB;
  RM kernel ≤ 0.32 GB) vs 11 GB free.
- RM synthesis: < 100 ms for all times; noise Monte Carlo essentially free.

The adaptive zoom/wide grid is what makes full-band coverage computable at all.

## Testing

TDD on the pure functions:
- single-RM synthetic input → `faraday_spectrum` peaks at the true RM;
- `rmsf` main-lobe FWHM matches `2√3/Δλ²`;
- `FrequencyPlan.channelize` on a flat spectrum reproduces `apply_wide` /
  `apply_narrow`; truncated/deduplicated channelization matches the full-grid
  result within tolerance;
- `radiometer_sigma` matches the radiometer equation.

## Conventions

- Functionality and readability first; do not overuse comments; keep files
  focused and manageable (split when a file does too much).
- Black, line length 79; keep within existing project style.
- Update `README.md` and `CLAUDE.md` to document the new modules, the
  channel-table/`FrequencyPlan` workflow, and the attic reorganization.

## Out of scope (pinned for later)

- **Option 1** (forward-model matched filter against the known RM sky).
- **Option 4** (bandwidth-depolarization observable) — infra-compatible, no
  analysis now.
- **Option 5** (per-LST coherent coadd) — LST tagging lays the groundwork only.
- RM-CLEAN deconvolution of the Faraday spectrum.

## Open decisions (deferred to data)

- The exact adaptive grid (band coverage, number/placement of zoom anchors) —
  chosen in step 0 from the measured RMSF.
