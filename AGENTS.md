# AGENTS.md — working notes for AI agents in this repo

Branch `luseepy-version` redoes the Faraday-paper analysis on the
luseepy four-port engine. The task spec is `INSTRUCTIONS-LPY.md`;
running status is `PROGRESS.md`. The original two-port simulator
(`sky.py`, `beam.py`, `sim.py`) is described in `CLAUDE.md` — leave it
alone; new work lives in `src/lusee_faraday/fourport.py` and `scripts/`.

**NEVER modify the luseepy checkouts** (`~/Dropbox/work/lusee/luseepy`,
`~/work/lusee/luseepy`) — that is well-controlled source. "Use luseepy
infrastructure" means *import* from `lusee`; all code, tests and
scripts live in this repo.

**Memory / OOM:** the "mysterious session kills" were the kernel OOM
killer (croissant dense spherical transform: ~(lmax+1)^2·npix·16 bytes;
nside=128 + lmax=128 → 50 GB). Validation uses nside=64/lmax=64 skies
and single-channel response slices for the harmonic engine. Run every
heavy job in background under `ulimit -v 16000000` (24 GB if croissant
dense transforms are involved) with logs in `generated_data/`.

## Environment

```bash
.venv/bin/python scripts/<script>.py     # always use the repo .venv
```

The venv has an editable install of luseepy from
`/home/anze/Dropbox/work/lusee/luseepy` (four-port branch) plus
`finufft`, `fitsio`, `croissant`, `jax`, `lunarsky`, `healpy`.
Set `JAX_ENABLE_X64=1` before importing croissant/jax paths.

## External resources

- Paper draft: `/home/anze/latex/LuSEE/Faraday`
- Response artifact (631 MB, slow to load ~tens of s):
  `/home/anze/work/lusee/Drive/Simulations/BeamModels/BGL_v16/lusee_bgl_v16_response_v3.fits`
- Sky inputs in `data/`: `haslam408_dsds_Remazeilles2014.fits`
  (**RING**, K — check ORDERING headers, an earlier NEST assumption
  scrambled the map), `wmap_band_iqumap_r9_9yr_K_v5.fits` (NESTED, mK
  thermodynamic), `faraday2020v2.hdf5` (`faraday_sky_mean`, RING,
  rad/m²). All nside=512, used at native resolution.
- Old-vs-new engine comparisons: `/home/anze/work/lusee/big_refactor_delete_when_ready/old_vs_new`

## Pinned conventions (do not re-derive; validated in scripts/validate_engine.py)

- Response frame: cartesian x=East, y=North, z=zenith → proper rotation
  from galactic; grid phi = 90° − astronomical azimuth.
- Ports 0,1,2,3 = N,E,S,W. 16 real channels ordered as
  `lusee.Covariance.default_product_labels()` (autos R, crosses R,I).
- Sky Q/U are healpy/COSMO; croissant wants IAU (U flips sign).
- Faraday rotation: (Q+iU) ← (Q+iU)·exp(+2i φ_FD λ²).
- Fixed-beam approximation: kernel evaluated at one native response
  channel (30/10/50 MHz); only the Faraday phase is chromatic.
- Time axis: 1024 samples over exactly one lunar sidereal day
  (27.321661 d) starting 2027-01-01 09:00 UTC → periodic, FFT-ready.
- Fine frequency grid: 16384 × (25 kHz/2048) centered on the band
  center (±0.1 MHz); 3 parent bins + 3×64 zoom bins fit inside.
- Real sky maps (Haslam dsds, WMAP K, faraday2020v2) at native
  nside=512 RING, never degraded (per-pixel Faraday phases do not
  commute with ud_grade). Harmonic paths only need lmax≈30 (beam
  band-limit), applied to the full-size maps.
- Zoom bins use FFT ordering (0=center, 1–32 positive, 33–63 negative).
  The zoom FFT runs on the critically sampled 25 kHz parent stream, so
  bins have folded images: the Nyquist bin k=32 is an exact 50/50
  double peak at ±12.5 kHz of its parent; bins 31/33 carry ~42%
  images. `zoom_frequency_grid` places bins at nominal centers (192
  distinct contiguous slots, no duplicates); the folding is physical
  and is removed in post-processing by `zoom_transfer` +
  `zoom_deconvolve` (step4_power_spectra.py; smoothness-regularized —
  a plain min-norm inverse leaves boundary/pad slots degenerate).
- **Polarimeter:** all pseudo-Stokes use the zenith-calibrated
  complex four-port X/Y (`fourport.zenith_port_weights` +
  `fourport.orthonormalize_xy`; cached per band center by
  `scripts/zenith_weights.py`, mode="ortho" default). Zenith
  Q=U=V=0 to machine precision. Raw X=E−W, Y=N−S figures keep the
  `step1_` prefix; calibrated ones `step1w_`.
- **PSD sanity (hard-won):** any physical covariance obeys
  sqrt(Q²+U²+V²) ≤ I; a single polarized source is rank-1 (equality).
  If pseudo-p > 1 appears it is a BUG (e.g. decomposing
  K@(1,cos2χ,sin2χ,0): the e^{−2iχ} coefficient is 0.5(K_Q+iK_U),
  NOT conj of the e^{+2iχ} one — cross-pair kernels are complex).
  step1_point_source.py asserts this on every run.

## Script inventory (all take `--help`; heavy ones honor caches)

- `scripts/common.py` — shared config (site, grids, sky loading at
  native nside=512, rotation-matrix cache).
- `scripts/validate_engine.py` — engine vs luseepy harmonic engines
  (ALL PASSED: 8.6e-16 / 1.1e-2 / 6.4e-15 / 4.2e-4).
- `scripts/zenith_weights.py` — calibrated polarimeter vectors per
  band center (cache: generated_data/cache/zenith_weights_*.npz).
- `scripts/step1_point_source.py` / `step1_plots.py [--calibrated]`
  — polarized transiting source; step1_ionly_source.py [--calibrated]
  — unpolarized (leakage) source.
- `scripts/step2_real_sky.py --center {30,10,50}` — real-sky
  waterfalls (~80 min each); `step2_plots.py --center C` — figures.
- `scripts/step_ionly.py --centers 30 10 50 [--analyze]` — perfect
  depolarization reference + fractional-effect table numbers.
- `scripts/step4_power_spectra.py --centers 30 10 50` — 2D delay
  spectra, delay profiles, zoom deconvolution.
- `tests/test_fourport.py` — 11 data-free unit tests.
- Report: `report/report.tex` (pdflatex twice; needs amssymb).

## Workflow rules

- Waterfall outputs → `generated_data/` (npz); caches (rotation
  matrices, degraded sky maps) → `generated_data/cache/`.
- Figures → `report/figures/`; report LaTeX → `report/`. Do not edit
  the paper itself.
- Update `PROGRESS.md` after each completed step.
- Long jobs: run in background, checkpoint to `generated_data/cache/`
  so reruns are cheap; the rotation-matrix cache alone takes ~minutes.
  Use ABSOLUTE paths for log redirects (the persistent shell cwd may
  sit in a subdirectory and the job dies instantly on the redirect).
- `generated_data/` is gitignored (8+ GB of memmapped waterfalls);
  everything there is regenerable from the scripts + caches.
- Style: black, line length 79 (matches the rest of the repo).
