# AGENTS.md — working notes for AI agents in this repo

The Faraday analysis runs on **luseepy + croissant**. `CLAUDE.md` describes
the package; `docs/measurement-model.md` is the model behind it;
`PROGRESS.md` is running status. The original two-port simulator
(`sky.py`/`beam.py`/`sim.py` and friends) was retired at the end of the
2026-08-18 refactor — it no longer exists, and the notebooks that drove it
are in `notebooks/archive/` with a note.

**NEVER modify the upstream checkouts.** Both are wired as editable
installs through `[tool.uv.sources]` and both are well-controlled source:

| Component | Path | State |
|---|---|---|
| luseepy | `../luseepy` | branch `deps/croissant-v5.3.0.dev1`, `52b96bc` |
| croissant | `/home/christian/Documents/projects/croissant-main` | worktree **detached** at `1c4d6c5` (= `v5.3.0.dev0-15-g1c4d6c5`); `main` is `0ac2f86`, two commits ahead |

"Use luseepy infrastructure" means *import* from `lusee`; all code, tests
and scripts live in this repo. Reproduce the environment with
`uv sync --extra dev` — plain `uv sync` prunes the dev extra (black,
flake8, pytest-cov) and `addopts`' unconditional `--cov=src` then makes
`uv run pytest` fail before collection. Add packages with `uv add` —
**never** `uv pip install`.

**Memory / OOM:** the "mysterious session kills" were the kernel OOM
killer (croissant dense spherical transform: ~(lmax+1)^2·npix·16 bytes;
nside=128 + lmax=128 → 50 GB). Validation uses nside=64/lmax=64 skies
and single-channel response slices for the harmonic engine. Run every
heavy job in background with **absolute** log paths under
`generated_data/`, inside a cgroup that caps **physical** memory:
`systemd-run --user --scope -q -p MemoryMax=10G -- uv run python …`.
`ulimit -v 16000000` (24 GB if croissant dense transforms are involved)
is the **address-space** guard and stays — 16 GB is the floor even for
the test suite, since at 12 GB three zenith tests OOM inside jax — but
it bounds `RLIMIT_AS`, not RSS, and protects the desktop from nothing:
jax/BLAS/numpy reserve far more address space than they commit, so
*lowering* it just kills jobs early (`ulimit -v 8000000` killed
`step5_template.py` 13.7 s in at a 960 MiB allocation while its RSS was
4.9 GiB). Report peak RSS from `/usr/bin/time -v`, always.

## Environment

```bash
uv run python scripts/<script>.py     # or .venv/bin/python
```

Set `JAX_ENABLE_X64=1` **before any jax import**. `scripts/common.py`
does it with an `os.environ.setdefault` above every other import and
every script imports `common`, so scripts are covered.  Test modules
set it themselves at the top of the file, and `tests/conftest.py` sets
it as a backstop for anything that forgets. Without it croissant and luseepy
silently run in complex64 — croissant even says so on stderr.

## External resources

- Paper draft (do not edit): `/home/anze/latex/LuSEE/Faraday`
- Response artifact, 631 MB, ~tens of s to load:
  `data/BGL_v16/lusee_bgl_v16_response_v3.fits`, with `_c4sym` (C4
  group-averaged = the paper's 90°-rotation assumption made
  self-consistent) and `_diagza` (inter-port coupling removed)
  ablation variants. `scripts/common.py:RESPONSE_PATH` defaults to the
  as-built model and is overridable with `$LUSEE_RESPONSE`.
- Sky inputs in `data/`: `haslam408_dsds_Remazeilles2014.fits`
  (**RING**, K — check ORDERING headers, an earlier NEST assumption
  scrambled the map), `wmap_band_iqumap_r9_9yr_K_v5.fits` (NESTED, mK
  thermodynamic), `faraday2020v2.hdf5` (`faraday_sky_mean`, RING,
  rad/m²). All nside=512, used at native resolution.

## Pinned conventions

**The single source of truth is `lusee_faraday.conventions` and
`lusee_faraday.config`** — import from them rather than re-deriving or
re-typing any of the following. Validated in
`scripts/validate_engine.py` and `tests/test_conventions.py`.

- Response frame: cartesian x=East, y=North, z=zenith → proper rotation
  from galactic; grid phi = 90° − astronomical azimuth.
  (`conventions`, `pixel_arm.topo_rotation_matrix`)
- Ports 0,1,2,3 = N,E,S,W. 16 real channels ordered as
  `lusee.Covariance.default_product_labels()`. (`conventions.PORT_PAIRS`,
  `conventions.PRODUCT_LABELS`)
- Sky Q/U are healpy/COSMO; croissant wants IAU (U flips sign).
  (`conventions.cosmo_to_iau_qu`)
- Faraday rotation: (Q+iU) ← (Q+iU)·exp(+2i φ_FD λ²); on the harmonic
  duals, P_MINUS·exp(−2iφλ²) and P_PLUS·exp(+2iφλ²).
  (`conventions.faraday_phase_cosmo`, `conventions.dual_block_phase`)
- Fixed-beam approximation: kernel evaluated at one native response
  channel (30/10/50 MHz); only the Faraday phase is chromatic.
  (`response.FixedChannelKernel`, which asserts a native channel)
- **The freeze covers the receiver loading too.** Z_A moves 12% across
  one 0.5 MHz native step at 30 MHz, and letting the impedances follow
  the fine grid puts an 11% smooth ramp into the band — exactly the
  non-Faraday chromatic structure the delay-space argument asserts is
  absent. Pass `instrument.covariance(..., impedance_freq_mhz=CENTER)`,
  and `T_moon=0.0, T_ant=0.0` wherever the legacy assembler had no
  thermal terms (luseepy defaults T_moon to 250 K: a factor 7.4e3).
- Time axis: 1024 samples over exactly one lunar sidereal day
  (27.321661 d) starting 2027-01-01 09:00 UTC → periodic, FFT-ready.
  (`config.times`)
- Fine frequency grid: 16384 × (25 kHz/2048) centered on the band
  center (±0.1 MHz); 3 parent bins + 3×64 zoom bins fit inside.
  (`config.fine_freqs`, `config.parent_centers`)
- Real sky maps (Haslam dsds, WMAP K, faraday2020v2) at native
  nside=512 RING, never degraded (per-pixel Faraday phases do not
  commute with ud_grade). Harmonic paths only need lmax≈30 (beam
  band-limit), applied to the full-size maps.
- Zoom bins use FFT ordering (0=center, 1–32 positive, 33–63 negative).
  The zoom FFT runs on the critically sampled 25 kHz parent stream, so
  bins have folded images: the Nyquist bin k=32 is an exact 50/50
  double peak at ±12.5 kHz of its parent; bins 31/33 carry ~42%
  images. `channelization.zoom_frequency_grid` places bins at nominal
  centers (192 distinct contiguous slots, no duplicates); the folding
  is physical and is removed in post-processing by `zoom_transfer` +
  `zoom_deconvolve` (step4_power_spectra.py; smoothness-regularized —
  a plain min-norm inverse leaves boundary/pad slots degenerate).
- **Polarimeter:** all pseudo-Stokes use the zenith-calibrated
  complex four-port X/Y (`polarimeter.zenith_port_weights` +
  `polarimeter.orthonormalize_xy`; cached per band center by
  `scripts/zenith_weights.py`, mode="ortho" default). Zenith
  Q=U=V=0 to machine precision. Raw X=E−W, Y=N−S figures keep the
  `step1_` prefix; calibrated ones `step1w_`.
- **PSD sanity (hard-won):** any physical covariance obeys
  sqrt(Q²+U²+V²) ≤ I; a single polarized source is rank-1 (equality).
  If pseudo-p > 1 appears it is a BUG (e.g. decomposing
  K@(1,cos2χ,sin2χ,0): the e^{−2iχ} coefficient is 0.5(K_Q+iK_U),
  NOT conj of the e^{+2iχ} one — cross-pair kernels are complex).
  `polarimeter.check_psd` runs on every step-1 chunk.

## Two arms, and which is which

- **Harmonic** (`response` + `engine` + `instrument`) is the production
  path. Its correctness gate is `tests/test_engine_gate.py`: 6.8e-16
  against luseepy's own convolution.
- **Pixel** (`pixel_arm.py`) is a validation arm.
  *Production code must not import it.* It survives because an
  independent quadrature is what makes `scripts/crosscheck_pixel_arm.py`
  meaningful, and because the diffuse scripts still run on it.
- The two agree to 3.05e-4 on an unpolarized real sky but only ~3% on a
  polarized one — see `docs/measurement-model.md` §8 before quoting any
  cross-arm number.

## Script inventory (most take `--help`; heavy ones honor caches)

`validate_engine.py` and `crosscheck_pixel_arm.py` have no
`argparse` — invoking either with `--help` starts a multi-minute
job instead of printing usage.  `probe_toolchain.py` is listed
below too; it was missing from earlier revisions of this table.

- `scripts/common.py` — shared config (site, grids, sky loading at
  native nside=512, rotation-matrix cache) and the x64 setdefault.
- `scripts/validate_engine.py` — engine vs luseepy harmonic engines.
  Re-run in full 2026-08-19 after the `lam2` shape fix: ALL PASSED,
  8.645e-16 / 1.082e-2 / 6.445e-15 / 4.177e-4, exit 0 (~6 min).
  *pixel arm*
- `scripts/probe_toolchain.py` — croissant engine selection at
  nside=512 / lmax=30; asserts no dense engine is chosen.
- `scripts/crosscheck_pixel_arm.py` — harmonic vs pixel characterization
  on the real response. Its [1e-2, 8e-2] band is specific to its own
  polarized test sky.
- `scripts/zenith_weights.py` — calibrated polarimeter vectors per
  band center (cache: generated_data/cache/zenith_weights_*.npz).
  *new stack*
- `scripts/step1_point_source.py` / `step1_plots.py [--calibrated]`
  — polarized transiting source; `step1_ionly_source.py [--calibrated]`
  — unpolarized (leakage) source. *new stack*
- `scripts/step_ionly.py --centers 30 10 50 [--analyze]
  [--engine harmonic|legacy]` — perfect depolarization reference +
  fractional-effect table numbers. *new stack (harmonic default)*
- `step2_real_sky.py`, `step2_plots.py`, `step4_power_spectra.py` —
  **not on this branch.** They compute the diffuse-Faraday results the
  2026-08-18 audit refuted and were left at the `audit-2026-08-18` tag
  by commit 956e770; check the tag out to re-run them, and cite the tag
  rather than a branch. (Pre-existing inventory rot, corrected here.)
- `scripts/beam_ablation.py`, `scripts/compare_main_vs_asbuilt.py` —
  response ablations and the Fig-4 lineage.
- **Step 5, the diffuse delay template** (`dispersion.py` + `noise.py`;
  outputs `generated_data/step5_*.npz`, figures `report/figures/step5_*`).
  Run them in this order; only the first is heavy:
  - `scripts/step5_template.py [--arm four-port|two-port] [--lst 128]
    [--bands 30 50 10] [--sigma-eff 9.8]` — builds `F(phi)` per band and
    per geometry `k`, the coherence-tilted variant, the knees (plain and
    plane-tapered) and the LST-resolved tail gate. **Heavy: 20-40 min,
    peak RSS 5.3 GiB four-port / 2.2 GiB two-port.** Background +
    `MemoryMax=10G` + absolute log path. Its two npz files are the
    inputs to everything below and are *not* cheap to regenerate.
  - `scripts/step5_instrument_envelope.py` — the depth-horizon /
    percentile table of `docs/measurement-model.md` §11 (minutes).
  - `scripts/step5_sensitivity.py [--lunations 24] [--t-amp K]` — the
    whitened matched-filter threshold curve and the closed-form
    cross-check (~1 min). `--t-amp` is 0 by default and the chain has no
    amplifier noise, so its `T_sys/T_sky` is `1 + T_loading/T_sky`, a
    lower bound — not a sky-domination result.
  - `scripts/step5_plots.py` — all six step-5 figures from the npz
    files. Re-run it after ANY figure-text change (seconds).
- `tests/testpixel_arm.py` — data-free unit tests for the pixel arm.
- Report: `report/report.tex` (pdflatex twice; needs amssymb).

## Workflow rules

- Waterfall outputs → `generated_data/` (npz); caches (rotation
  matrices, degraded sky maps, zenith weights) → `generated_data/cache/`.
- Figures → `report/figures/`; report LaTeX → `report/`. Do not edit
  the paper itself.
- Update `PROGRESS.md` after each completed step.
- Long jobs: run in background, checkpoint to `generated_data/cache/`
  so reruns are cheap; the rotation-matrix cache alone takes ~minutes.
  Use ABSOLUTE paths for log redirects (the persistent shell cwd may
  sit in a subdirectory and the job dies instantly on the redirect).
- `generated_data/` is gitignored (2.1 GB of memmapped waterfalls
  today; it has been 8+ GB with all three bands live);
  everything there is regenerable from the scripts + caches.
- Style: black, line length 79. `uv run flake8 src/` is clean — keep it
  that way.
- **Before committing a test, ask whether it would fail if the thing it
  names were broken.** Nine tests in the 2026-08-18 refactor had to be
  rewritten or strengthened because the answer was no. Injecting the mistake and
  watching it go red is the only proof that counts.
