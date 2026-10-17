# Progress — four-port Faraday analysis (INSTRUCTIONS-LPY.md)

## Gaussian-screen notebook: physical overview — 2026-10-16

Added an opening explanation of the fixed Gaussian source/beam, the ensemble
of physical RM screens, and the rotate-then-sum measurement equation. It
connects individual pixel-summed spectra to the analytic ensemble mean and
frequency covariance, and explains why spatial RM correlations affect the
surviving power. Updated the notebook builder with the same text. This is a
prose-only change; existing outputs are retained from the previous execution
and their provenance describes that run. No simulation or forecast reran.
Verified notebook structure and agreement with the builder's Markdown/code
cells without executing the analysis.

## Gaussian-screen step 1: drawn screens and Stokes readouts — 2026-10-16

After review feedback the notebook gained Section 6b. `ScreenModel.readout`
now takes explicit `(k_I, k_Q, k_U)` (`READOUTS`: `xx`, `yy`, `xy`, `Q`, `U`)
in place of the generic Jones pair, since ideal x/y dipoles give
`Vxx=(I+Q)/2`, `Vyy=(I-Q)/2`, `Vxy=U/2`. `screen_draws.py` draws periodic
spectral Gaussian screens and sums the Faraday-off/on readouts pixel by pixel
from the Stokes brightness, independent of the separation integral and of the
Jones-branch algebra. Figures 4--5 show five drawn screens (seeds 11--15,
4096^2 grid over 8 degrees) against the analytic mean and 1-sigma band, each
readout divided by the unrotated polarized amplitude `A(nu)`.
`scripts/gaussian_screen_draws.py` (about 15 min, 2 GB) writes
`generated_data/gaussian_screen_step1/draws.npz`; the notebook reads it.

Measured: drawn Faraday-off readouts match the analytic off to 4e-6 A (box
truncation); halving the pixel count of a fixed screen changes the on readout
by at most 1.2e-10 A at 10--50 MHz, against the plan's 1e-4 gate (a floor set
by the one-sided box edge). A 64-screen pilot on a 2048^2 grid at 30, 40,
50 MHz gives |mean z| <= 1.44 and std ratios 0.80--1.06 against the
prediction. That is consistent with the formulas, but with 64 screens the std
is only known to ~9%, there are two independent readouts at three
frequencies, and no frequency covariance, bootstrap errors or box/spectrum
refinement were done: it is **not** the step-2 gate. Tests:
`tests/test_screen_draws.py` (a flipped phase sign fails all five readouts).

## Gaussian-screen step 1 delivered for review — 2026-10-15

The executed [notebook](notebooks/gaussian_screen_step1.ipynb) keeps the
original Faraday-off/on complex visibility central. The
[specification](docs/specs/2026-10-15-gaussian-screen-step1.md) defines a
smooth deterministic Gaussian source and normalized beam on a tangent plane,
an admissible Gaussian foreground-screen covariance with physical transverse
correlation length, and a realizable ideal complex Jones pair. Both
polarization branches and their joint statistics enter the off/on means,
raw second moments, centered frequency covariance, nonconjugated moments,
and expected squared on-minus-off difference, including its mean term.

Reusable code is in `src/lusee_faraday/gaussian_screen.py`; regenerate all
notebook outputs and three captioned figure pairs with
`.venv/bin/python scripts/gaussian_screen_step1.py`. The Gaussian angular
weight reduces the pair integral to a nonoscillatory separation integral.
Rescaled adaptive quadrature is checked against its special-function form;
no dense angular pixel-pair array or real-sky map is used. The notebook
separates physical correlation-length/wavelength sweeps from numerical
refinement, and records source/configuration hashes, environment, UTC,
resource use and the worst check discrepancies in the
[generated results record](docs/gaussian-screen-step1-results.json).

Four targeted tests pass. Zero rotation, deterministic constant depth and
random spatially constant depth controls pass. The independent Gaussian
depth/Jones calculation rejects deliberately injected conjugate-branch and
centering mistakes. These are algebra and reduced-integration checks only.
The [plan](docs/plans/2026-10-15-gaussian-screen-step1.md) specifies the next
independently converged deterministic spatial-screen comparison, with
separate angular, domain/spectrum and ensemble-error gates. No such ensemble
has run. The depth-histogram reduction, Galactic correlations, strong-rotation
deterministic convergence, actual beam/loading, channels, noise, foreground
treatment and production forecast remain unvalidated. **Stop for review.**
No long real-beam audit, upstream edit or external manuscript edit was made.

## Audit preparation — 2026-10-15, merged to `main` 2026-10-16

The corrected, bounded BGL v2 audit package and the editable croissant
source/lockfile update were merged to `main` (PR #6) together with a CI fix
that matches the local dependency sources by shape rather than by checkout
path. The `audit-preparation` branch has been deleted; `gaussian-screen-step1`
is rebased onto that `main`, and its physical validation is pending.
The queue and handoff below preserve the audit preparation context.

The installed croissant path, version and pyerfa dependency match the
committed project/lock metadata; no sync or upstream edit was made.
All 12 small audit tests pass, and restoring either the negative-branch
sign defect or the cancellation-based count makes its regression test
fail. Formatting/lint, shell syntax and the read-only launcher plan pass.
The historical briefing and scalar-lobe plot script are retained with
their limitations noted in `audit/README.md`; the October survey JSON
remains the original dated snapshot. No smoke/full service or large
scientific computation has been launched.

## Current handoff 2026-10-15

Christian agreed to the Gaussian-screen benchmark and validation sequence.
[TODO.md](TODO.md) now makes this the next scientific task: derive the off/on
statistics, recover converged deterministic ensembles at weak rotation,
demonstrate covariance convergence at strong rotation, then introduce the
LuSEE response and assess physical uncertainty. The real-beam audit remains
a separate track pending execution review. This update changes the work queue;
no spatial-screen benchmark simulation or long computation has been run.
The step-1 derivation and reduced-integral checks above now supersede the
previous unexecuted-derivation status.

**Prepared for Christian's review; the long run has not started.** Read
[docs/paper-preparation.md](docs/paper-preparation.md) for the consolidated
survey, paper outline, evidence requirements, environment drift and proposed
staged commands. [TODO.md](TODO.md) is the current queue;
[the generated snapshot](docs/paper-preparation-status.json) records exact
input/code/dependency fingerprints and saved-product values.

The instrument refactor is merged on main (`33c8730`, PR 3). The replacement
Faraday-depth template, detection forecast, executed notebook and report are
implemented on `faraday-delay-template` (`adb4024`, open PR 4 based on main).
The historical August status below does not describe that later branch.

The September BGL follow-up saved 48 entries at each of nside 512 and 1024,
none at 2048; its log stops after the 2048 maps. An OOM interruption is
plausible but not confirmed by retained journal evidence. Preparation found
and corrected a negative-branch exponent error and a complex-cancellation
statistic mislabeled as effective pixel count. September results remain
unchanged as legacy evidence. The prepared v2 audit uses disk-backed weights,
bounded kernel/NUFFT batches, a read-only preflight, fingerprinted atomic
checkpoints, and a resource-limited launcher. Recompute every resolution
before interpreting the real-beam audit.

The subsequent mathematical review is recorded in the handoff's
"Mathematical and physical basis review" section. The covariance forecast
needs a physical ensemble and a derivation of angular/depth correlations;
a stable RM histogram alone does not validate it. Slab envelopes are not
amplitude floors, mean polarization differs from polarization power, and
channel integration and data foreground treatment need explicit propagation.
These are foundational checks before interpreting saved forecasts, separate
from the numerical audit. No long computation was started.
The handoff now specifies a Gaussian-screen benchmark to test whether analytic
phase averaging removes the numerical cancellation problem. Its derivation
and validation sequence are proposed; no benchmark results are claimed.

The report still needs its separate coherent-sum amplitude/power-weight
correction and the previously scoped spectral-index, turbulence/dispersion,
relative gain and receiver-noise work. Its saved forecast remains conditional
on physical amplitude scenarios. No upstream dependency, external manuscript,
scientific result product or old audit ledger was changed by preparation.
All 12 targeted synthetic checks pass; injecting each discovered algebra
mistake makes its regression check fail. Lint, formatting, launcher syntax
and read-only preflight pass. Full-resolution memory remains to be demonstrated.

Updated: 2026-08-19 — **refactored onto luseepy + croissant** (see the
section below); the five analysis steps and the report were complete
before that on the pixel-space engine. Report in `report/report.tex`
(14 pp, compiles clean, all figures referenced in text).

## Refactor onto luseepy + croissant (2026-08-18 → 2026-08-19)

Eighteen tasks on branch `luseepy-refactor`, plan at
`docs/superpowers/plans/2026-08-18-luseepy-croissant-refactor.md`,
ledger at
`.superpowers/sdd/2026-08-18-luseepy-croissant-refactor/progress.md`.

**What the package is now.** Sky components → Faraday coefficients →
harmonic contraction → luseepy covariance → polarimeter →
channelization, in `sky.py`, `response.py`, `engine.py`,
`instrument.py`, `polarimeter.py`, `channelization.py`, over
`conventions.py` and `config.py`. `docs/measurement-model.md` is the
model; `CLAUDE.md` the module tour.

**What was deleted.** The whole original two-port simulator:
`beam.py`, `sim.py`, `fast_sim.py`, `healpix.py`, `rotations.py`,
`utils.py`, `plot.py` and their four test files (33 tests). With them
went the `interp_hp` pole artifact and the `healpy.Rotator` machinery.
`spectrometer.py` had already been replaced by `channelization.py`.

**What was demoted.** `fourport.py` → `pixel_arm.py`, a validation
arm that production code must not import. It survives because an
independent quadrature is what makes `scripts/crosscheck_pixel_arm.py`
meaningful, and because `step2_real_sky.py` and `step4_power_spectra.py`
still run on it — deliberately, since the 2026-08-18 audit showed their
diffuse Faraday content is HEALPix shot noise.

**Notebooks.** `notebooks/faraday_sims.{ipynb,py}` drove the paper's
original figures through the deleted `sim.py` and are archived in
`notebooks/archive/` with a note; they no longer run. Five more —
`point_source-LN.ipynb`, `paper_plots.ipynb`, `wmap-time.ipynb`,
`faraday_analysis.ipynb`, `wmap_one.ipynb` — also reference the retired
API and were **left in place on purpose**: they are a working record,
and moving or deleting them is the author's call.

**Cross-arm agreement (harmonic vs pixel).** The correctness gate is
`tests/test_engine_gate.py` at 6.8e-16 against luseepy's own
convolution. The pixel cross-check is a characterization and is
sky-dependent:

| sky | harmonic vs pixel |
|---|---|
| polarized synthetic (crosscheck's own test sky) | 2.678e-02 |
| the same run with Q = U = 0 | 2.687e-05 |
| real I-only sky, nside 512, lmax 30 | 3.054e-04 |

all global-max-normalised, the estimator `crosscheck_pixel_arm.py` uses.
Per channel the real-sky worst case is 1.77e-3 (on 02I, whose own scale
is 17% of the global max; the autos sit at 1.9–2.3e-4). Raising lmax
from 30 to 48 moves the products by 7.77e-5, so harmonic truncation is
not the explanation. **The disagreement is entirely a polarized-sky
phenomenon** — see `docs/measurement-model.md` §8 before quoting any
cross-arm number.

**Regressions that pinned the refactor.**

- *Zenith polarimeter:* all 24 complex entries of the published Table 1
  (`report/report.tex`, three bands × four ports × X/Y) reproduced,
  transcribed from the published table rather than from a log, atol
  1e-3 against a 3-decimal table.
- *Step 1, regenerated on the new stack:* raw transit leakage
  **0.133849** (published 0.134, 0.11%); ortho transit leakage
  **6.516e-4** (7e-4, 6.9%); zoom recovery real **0.7947** (0.79,
  0.59%) and ideal **0.8598** (0.86, 0.03%); Q oscillation period
  **1.8877 kHz** analytic (1.89, 0.12%) and 1.8868 kHz *measured from
  the regenerated waterfall* (0.17%); gains-mode transit leakage
  0.09611 (0.096); rank-1 at transit [0.999863, 0.999919].
- *I-only reference:* the ported harmonic arm reproduces the pre-port
  pixel artifact to 3.4e-14, and the two arms differ by the 3.054e-4
  above.

The published numbers moved by up to 0.6%, so the regeneration is real
rather than cosmetic.  The regeneration covered the **step-1 set only**:
14 of the 45 tracked figure basenames (28 of 80 files) were rewritten in
the final commit and are traceable to the committed code.  The
diffuse-sky figures still carry legacy-pipeline provenance —
`real*_waterfall_QU`, `real*_spectrum_snapshot*`, `real*_polfrac_track`,
`real*_sky_model`, `real*_pspec`, `pspec_delay_profile`,
`real*_ionly_frac`, `beam_ablation_30` and the four `cmp_*` — because
steps 2 and 4 deliberately still run on `pixel_arm`.  Several of
those are cited in `report.tex`, so the repo is in a *partially* mixed
state and the distinction matters if figures move to the paper.

## Resume state (read this first after /clear)

- Everything is committed on branch `luseepy-refactor`
  (remote: github.com/christianhbye/lusee_faraday).
- `generated_data/` is gitignored: the 2.1 GB fine waterfalls
  (step1/real30/real10/real50), binned npz files and caches live
  only on disk.  If missing, regenerate with the scripts (see the
  inventory in AGENTS.md; ~80 min per real-sky band, minutes for the
  rest — rotation matrices and zenith weights are cached in
  generated_data/cache/).
- To rebuild every figure from existing data:
  `step1_plots.py` (+`--calibrated`), `step1_ionly_source.py`
  (+`--calibrated`), `step2_plots.py --center {30,10,50}`,
  `step_ionly.py --centers 30 10 50 --analyze`,
  `step4_power_spectra.py --centers 30 10 50`; then pdflatex twice
  in `report/`.  Run under `ulimit -v 16000000` with absolute log
  paths; 12 GB is not enough.
- Read AGENTS.md for pinned conventions, OOM/memory rules, the PSD
  sanity invariant, and the script inventory.
- Next natural tasks (not started): transfer report content into the
  paper (needs user go-ahead), noise/detectability forecast,
  ionospheric small-phi study at 10 MHz.

## Done
- [x] `.venv` with editable luseepy install plus `finufft`, `fitsio`,
  `croissant`, `jax`.
- [x] `src/lusee_faraday/fourport.py` — pixel-space four-port engine
  (kernel, transport, NUFFT Faraday synthesis, covariance/products,
  spectrometer integration).  Luseepy itself is READ-ONLY — never move
  code there; import from `lusee` only.  *(Now `pixel_arm.py`, a
  validation arm.)*
- [x] `tests/test_fourport.py` — 10 data-free unit tests; all pass.
  *(Now `tests/testpixel_arm.py`.)*
- [x] OOM diagnosis: the "mysterious session kills" were the kernel
  OOM killer (croissant dense spherical transform ~(lmax+1)^2·npix·16
  bytes).  Mitigations: small validation skies, single-channel
  response slices, `ulimit -v` + background for every heavy job.
- [x] `scripts/validate_engine.py` — ALL VALIDATIONS PASSED, re-run
  end to end 2026-08-19 (exit 0):
  [1] point source vs FullStokesCalibratorSimulator 8.645e-16;
  [2] diffuse-sky transport vs FullStokesCroSimulator worst 1.082e-2
  (nside=32/lmax=48 sky); [3a] NUFFT internal 6.445e-15;
  [3b] Faraday-rotated sky vs CroSimulator 4.177e-4.
  Checks [3a]/[3b] had been dead since the refactor: `config.lam2`
  returns shape `(1,)` where `scripts/common.lam2` returned a scalar,
  so the script's `np.array([l2])` was 2-D and FINUFFT rejected it.
  Fixed in the final round; the values are unchanged from the ones
  recorded above.
- [x] **Step 1** (`step1_point_source.py` + `step1_plots.py`):
  transiting polarized source, phi_FD=250, 1024x16384x16 waterfall,
  binning, 6 figures.  Q oscillation period 1.89 kHz = predicted;
  recovered p at transit: fine 0.997, real zoom 0.75, ideal zoom
  0.83, parent 0.13; real-zoom dips at parent-bin edges (aliasing).
  **Bug fixed (caught by user via p > 1):** the point-source script
  decomposed K@(1,cos2chi,sin2chi,0) as cpol e^{2i chi} +
  conj(cpol) e^{-2i chi}, but for complex cross-pair kernels the
  second coefficient is 0.5(K_Q + iK_U), NOT conj(cpol) — the error
  broke PSD-ness of the covariance.  Rank-1/PSD check
  (sqrt(Q^2+U^2+V^2)/I <= 1, ~1 for a point source) now runs
  automatically in the script.  Track figures plot vs source
  altitude (rise solid / set dashed), not time.
- [x] **Steps 2-3** (`step2_real_sky.py` at 30/10/50 MHz + plots):
  native nside=512 maps (Haslam dsds is RING — an earlier NEST
  assumption scrambled I and forced a rerun; WMAP K is NESTED;
  faraday2020v2 RING).  Faraday depolarizes the beam-integrated P by
  ~2.5x; the smooth part is resolved by every channelization, the
  fast per-pixel oscillations show up as delay-space power.
- [x] **Step 4** (`step4_power_spectra.py`): 2D |P(f_t, tau)|^2 per
  band and per channelization + per-band delay profiles.  Faraday
  plateau with sharp cutoff at tau_FD(phi_max~2400) — 5 ms at 30 MHz,
  1.1 ms at 50 MHz (nu^-3 scaling verified); zoom bins track the fine
  profile between ~50 us and their 1.28 ms Nyquist; real zoom bins
  have an aliasing floor ~1e-3 of peak power (visible at 50 MHz),
  ideal Gaussian bins would gain ~2-3 decades.
- [x] **I-only (perfect depolarization) reference**
  (`step_ionly.py`, all three bands, frequency-flat by construction):
  the observed P through the Faraday screen is leakage-dominated
  (30 MHz parent-bin Q/I = 0.040 matches I-only to 2e-4; without
  Faraday the sky-pol term would be 2.4e-2 -> suppressed >100x).
  Fractional effect of sky polarization vs I-only at 30 MHz:
  parent bin |dP|/I median 1.7e-4 (max 5.4e-4), dI/I 9e-5;
  zoom bin 0 |dP|/I median 4.5e-4 (max 9.2e-4), dI/I 1.8e-4 —
  zoom retains ~2.7x more.  Contrast largest at 10 MHz (~3.5x in P,
  ~7x in I), smallest at 50 MHz.  Discussion + table in report
  (sec:ionly), figures real{10,30,50}_ionly_frac.
- [x] **Unpolarized point source** (`step1_ionly_source.py`,
  frequency-flat): leakage-only track figure
  (step1_ionly_polfrac, in report next to the polarized-track
  figure).  As-built beams give p_leak = 0.134 at transit (NOT zero
  — matches the old_vs_new zenith-polarimeter figure where as-built
  q ~ -0.1 at transit while ideal dipoles go to 0), peaking at 0.94
  near 32 deg altitude.  The 0.134 equals the parent-bin floor of
  the polarized source: full Faraday depolarization leaves exactly
  the leakage covariance.
- [x] **Zenith-calibrated polarimeter** (`zenith_weights.py`;
  `fourport.zenith_port_weights` + `fourport.orthonormalize_xy`):
  unpolarized zenith source autos (N,E,S,W) proportional to
  (1.26, 1.12, 1.25, 1.00).  Stage 1 (mode="gains"): per-port
  w_p ~ 1/sqrt(C_pp) = (0.952, 1.019, 0.956, 1.079) + common X/Y
  rescale -> zenith q nulled exactly, but u = -0.096 remains
  (inter-port cross-couplings <NE*> in <YX*>, irreducible with real
  diagonal gains).  Stage 2 (mode="ortho", default): X and Y are
  complex combinations of all four ports via Loewdin G^{-1/2}
  orthonormalization in the C0 metric (zenith Q=U=V=0 iff conj(x),
  conj(y) are C0-orthonormal) -> zenith leakage ~1e-16, X/Y stay
  ~95% dipole with ~5% complex admixture.  Unit-tested
  (test_orthonormalize_xy_nulls_leakage).  Results at 30 MHz:
  unpolarized-source transit leakage 0.134 (raw) -> 0.096 (gains)
  -> 7e-4 (ortho, set by the 0.56 deg offset from exact zenith);
  polarized-source parent bin at transit now shows TRUE bandwidth
  depolarization p = 7e-4; zoom recovery 0.79 real / 0.86 ideal.
  Section 2 figures = step1w_ prefix (ortho polarimeter); raw
  step1_ figures kept (user wants the raw unpolarized-leakage
  figure preserved as instructional).
- [x] **Zoom-bin deconvolution (user request):** the zoom stage FFTs
  the critically-sampled 25 kHz parent stream, so bins have folded
  images (Nyquist bin k=32: exact 50/50 double peak at +-12.5 kHz;
  bins 31/33: ~42% images).  `zoom_transfer` in
  step4_power_spectra.py builds the exact 192x256 linear map from a
  slot-gridded spectrum (32 pad slots/side) to the measured bins
  using the same weights as integrate_spectrometer; condition number
  3.9.  `zoom_deconvolve` inverts it with a second-difference
  (smoothness) regularization — needed because parent-boundary/pad
  slots enter only through shared Nyquist folds; plain min-norm
  ridge left them degenerate and rang across all delays.  Result
  ("real zoom, deconvolved" in the delay-profile figure): tracks the
  fine/ideal profiles through the 50 MHz Faraday cutoff, ~4 decades
  below the raw zoom aliasing floor.  Report Step 4 updated
  (deconvolution paragraph + caption + conclusion iii).
- [x] **Steps 2+ switched to the calibrated polarimeter (user
  request):** step2_plots.py, step4_power_spectra.py and
  step_ionly.py analyze now use `zenith_weights.get_weights(center)`
  (ortho vectors per band center; 10/50 MHz weights computed and
  cached, both null to ~1e-16).  All real-sky figures, power spectra
  and the I-only table regenerated; report numbers updated (parent
  (Q,U)/I = (0.146, -0.032) = I-only leakage to 2e-4; sky-pol
  suppression x380 at 30 MHz; zenith calibration does NOT reduce
  diffuse hemisphere-integrated leakage).  Fig 9
  (real30_spectrum_snapshot) re-styled: no-Faraday line dropped,
  y-range tight around the fine-grid Faraday hash.  Fig 10
  (real30_polfrac_track) redesigned as two panels: absolute p
  (no-FD vs parent) on top, channelization ratios p_X/p_parent - 1
  below (the curves previously overlapped invisibly; they agree to
  <1%).
- [x] **Fig 7 redone (user request):** now `step1w_xy_waterfalls` —
  2x2 waterfalls of the calibrated coherency <|X|^2>, <|Y|^2>,
  Re<XY*>, Im<XY*> (calibrated four-port combinations) over the
  altitude track x zoom frequency (`fig_xy_waterfalls` in
  step1_plots.py; raw 16-product figure still produced in default
  mode).  All figures through Fig 7 are now explicitly referenced in
  the Step 0/Step 1 running text.
- [x] **Report restructured (user request):** now opens with
  "Step 0: an unpolarized source and the calibrated polarimeter" —
  raw-leakage track figure (step1_ionly_polfrac), then the two-stage
  calibration story, then the calibrated leakage figure
  (step1w_ionly_polfrac).  Step 1 (polarized source) shows only
  calibrated-polarimeter results.  (Also fixed missing amssymb for
  \lesssim.)
- [x] **Per-band weights verified + listed; Fig 9 -> 2x3 (user
  request, 2026-08-18):** force-recomputed zenith weights at 10/30/50
  (`zenith_weights.py --force`, log
  generated_data/zenith_weights_recalc.log) — identical to the cached
  sets already used by all step-2+ analyses (each band's ortho
  vectors null its own zenith leakage to ~1e-16; the weights differ
  substantially between bands, e.g. the N auto is smallest of the
  four at 10 MHz but largest at 30 MHz).  10/50 MHz analyses rerun
  end-to-end (step2_plots, step_ionly --analyze, step4) — all
  numbers unchanged, confirming per-band weights were already in
  effect.  Report now has Table 1 (tab:weights) listing the full
  calibrated vectors per band.  Fig 9 replaced by
  `real_spectrum_snapshot_bands` (new `fig_spectrum_snapshot_bands`
  in step2_plots.py, flag `--snapshot-bands`): 2 rows (Q/I, U/I) x 3
  columns (10/30/50 MHz), each column with its own weights and its
  own max-signal time.  Pedagogical payoff visible by eye: 10 MHz
  hash unresolved, 30 MHz dense stipple, 50 MHz fine grid resolves
  individual ~10-20 kHz Faraday oscillations that the zoom bins
  track and even parent bins begin to follow.
- [x] **Step 5**: coherent story written in `report/report.tex`
  (setup, steps 1-4, practical conclusions on parent vs zoom bins and
  the delay window; figures in `report/figures/`).

## Possible follow-ups
- [ ] Transfer selected figures/text into the paper (explicitly out of
  scope per INSTRUCTIONS-LPY.md: "don't change the paper yet").
- [ ] Ionospheric-regime study: dedicated run with small phi (1-30
  rad/m^2) uniform screen to emulate the lunar ionosphere at 10 MHz.
- [ ] Noise: propagate radiometer noise through the zoom bins to turn
  the delay-space signature into a detectability forecast.

## Key decisions / conventions (pinned)
- Response frame: x=East, y=North, z=zenith (proper rotation);
  phi = 90° − azimuth. Port order N,E,S,W = 0,1,2,3.
- Sky Q/U in healpy/COSMO convention; U_IAU = −U_COSMO when feeding
  croissant. Faraday: (Q+iU) e^{+2iφλ²}.
- Fixed 30/10/50 MHz beam across the narrow band; only the Faraday
  phase is chromatic → all delay-space power is Faraday-induced.
- Real maps at native nside=512 RING, never degraded; Haslam file is
  RING, WMAP K NESTED (check ORDERING headers!).
- Time: 1024 samples over one lunar sidereal day (periodic).
- Fine grid 16384 × 25 kHz/2048; 3 parents + 3×64 zoom bins inside.
- Outputs in `generated_data/` (fine waterfalls are 2.1 GB memmaps),
  caches in `generated_data/cache/`, figures in `report/figures/`.
- Heavy jobs: background + `ulimit -v 16000000`, logs in
  `generated_data/`.
