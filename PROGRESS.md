# Progress — four-port Faraday analysis (INSTRUCTIONS-LPY.md)

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
  in `report/`.  (The `step2_*`/`step4_*` scripts live at the
  `audit-2026-08-18` tag, not on this branch.)  Run heavy jobs under
  `systemd-run --user --scope -p MemoryMax=10G` with absolute log
  paths; `ulimit -v 16000000` is an address-space guard only (12 GB is
  not enough for it) and does not cap RSS — see AGENTS.md.
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

## Step 5b: the Faraday depth template (branch faraday-delay-template)
- [x] `dispersion.py` (depth distributions, NUFFT transforms, real-response
  RMSF, depth horizon, geometry knob, coherence bracket) + `noise.py`
  (ported + matched filter) + `response.pair_weight_maps` (basis-independent
  weight combining both Faraday branches, `sqrt(0.5*(|K_Q|^2+|K_U|^2))`,
  not one branch alone).
- [x] Acceptance gates passed: normalised template invariant under nside
  256–2048 refinement and under a null rotation (the audit's two findings,
  rebutted); converged-regime control reproduced through the NUFFT path.
- [x] `step5_instrument_envelope.py` / `step5_template.py` /
  `step5_sensitivity.py` / `step5_plots.py`; figures in report/figures/.
- [x] Tail gate (S4.2.2), measured: from the full `--lst 128` run, the
  `|w|^2`-weighted tail fraction above the fixed beam-weighted-p99
  threshold, resolved over LST. Four-port arm GC-transit maxima per band:
  **2.16% (30 MHz), 3.15% (50 MHz), 2.39% (10 MHz)**; two-port arm:
  **2.18% (30 MHz), 3.70% (50 MHz), 2.69% (10 MHz)**. Away from transit the
  fraction falls to ~1e-6; the LST mean sits near ~0.8% at every band and
  arm, close to the ~1% the p99 definition implies at a typical LST, and
  GC transit lifts it only 2-4x above that floor, not by orders of
  magnitude. Read from `generated_data/step5_template.npz["tail_frac_lst"]`
  and `step5_template_two_port.npz["tail_frac_lst"]`, both shape `(3, 128)`.
- [x] **Tail-gate verdict against the S4.10 threshold.** Convention
  first, because it moves every number: `tail_frac_lst` is a fraction
  of the template's **power** while the bracket is an **amplitude**, and
  spec S4.2.2 says detection and localisation are "separated by roughly
  the square root of the tail's power fraction". So the tail's amplitude
  is `bracket x sqrt(f)`, not `bracket x f`. At 30 MHz
  `sqrt(0.0216) = 0.147` against `f = 0.0216` — a factor **6.8x on every
  ratio**, which an earlier version of this entry lost. Against `A_mf` at
  24 lunations (1.33e-5 / 1.07e-5 / 1.67e-5, four-port maxima):

  | band | at `lower_slab` | ratio | at `lower_dispersion` | ratio |
  |---|---|---|---|---|
  | 30 MHz | 6.26e-5 | **4.7 OPEN** | 7.7e-8 | 0.006 closed |
  | 50 MHz | 2.07e-4 | **19.3 OPEN** | 7.2e-7 | 0.067 closed |
  | 10 MHz | 7.7e-6 | 0.46 closed | 1.0e-9 | 6e-5 closed |

  (the two-port arm, taken consistently -- its own bracket with its own
  `sqrt(f)`, not the four-port bracket -- gives 4.77 / 19.92 / 0.47, at
  most 3% away, same OPEN/closed calls.) The bracket's
  `upper` level is deliberately **not** in the table: it is
  clamp-derived and not computable from this map (below). So the gate
  **opens at 30 and 50 MHz if the diffuse amplitude sits at the
  uniform-slab floor, and closes at every band if it sits at the
  internal-dispersion floor** — it is decided by which depolarisation
  floor the medium sets, a mixed-vs-external-screen geometry question.
  It is *not* decided by the tail measurement (solid at 2.2-3.7% in both
  arms), *not* by `theta_c`, and *not* by integration time: with
  `A ~ n^-1/2 N^-1/4`, closing the 30 MHz dispersion-floor gap of 173x
  would take ~3e4 times more nights. **This supersedes the earlier
  entry**, which recorded the gate as decided by the grid-clamped
  `theta_c` with a wider `structure_function` grid as the fix. Only
  `upper` contains `theta_c`; both floors above are `theta_c`-free.
  Also written up in `docs/measurement-model.md` §12 — this is the
  branch's headline decision and must not live only in the ledger.
- [x] Coherence angle / the bracket's upper end: `theta_c_clamped` is
  `[True True True]` in both npz files — the coherence angle hit the
  0.2 deg edge of the `structure_function` search grid at every band and
  both arms. Two consequences, both recorded late:
  **(1) the clamp OVERSTATES.** `coherence_angle` returns the grid edge
  when the root lies *below* it, so a clamped return is an upper bound
  on `theta_c`, and `upper ~ theta_c` inherits that. Measured with the
  script's own grid: `D(0.2 deg) = 96.2 (rad/m^2)^2` against targets
  5.01e-5 / 3.87e-4 / 6.19e-7, i.e. a root 1385x / 499x / 12465x below
  the grid edge under `D ~ theta^2`. The shipped `upper` (9.9e-3 at
  30 MHz) is therefore overstated by ~3 orders of magnitude; a `theta^2`
  extrapolation gives 7.1e-6, which is *below* `lower_slab` — the
  bracket inverts, which is itself the proof the extrapolation is not
  quotable. (Spec S4.4 says the bracket runs "1e-4 down to 1e-6"; the
  code produces 1e-2. The spec and the code disagreed all along.)
  **(2) widening the grid cannot fix it.** The root sits at 0.52 / 1.44
  / 0.058 arcsec, three decades below `faraday2020v2`'s own nside-512
  resolution; a wider grid would extrapolate `D ~ theta^2`, not measure.
  **The incoherent-patch upper bound is not determinable from this map**
  and must be quoted as such, never as ~1e-2. The two lower levels
  (`lower_slab`, `lower_dispersion`) contain no `theta_c` and stand.
- [x] `T_sys/T_sky` (S4.10 risk item), **partial**: 1.0044 (30 MHz),
  1.0170 (50), 1.0006 (10). Read it as `1 + T_loading/T_sky` and nothing
  more — `step5_sensitivity.py --t-amp` defaults to 0 and the luseepy
  chain carries no amplifier noise, so the term the spec's risk item is
  actually about (receiver noise through a short mismatched dipole on
  regolith at 50 MHz) is set to zero. It is a genuine lower bound, not a
  computed sky-domination check, and `t_sky = mean(I_sky)` is an all-sky
  mean rather than a beam-weighted antenna temperature. **Sky domination
  at 50 MHz stays open** until the collaboration supplies a receiver
  noise temperature; re-run with `--t-amp` when it does.
- [x] Template robustness, reported numbers (S6.5, S6.11): the
  `sin^2|b|` plane taper moves the 90%-mass knee by **4.3-5.5x** (30 MHz
  `k=0`: 89.6 -> 17.9 rad/m^2), so the roll-off is carried by the
  Galactic plane — per S4.2.1 the branch's *worst* case for the linear
  `phi(f)` ansatz, and a reason to read the amplitude bracket as wider
  rather than narrower. The two-arm swap (S4.9) moves the same knee by
  **+24% at 50 MHz**, +12% at 10 MHz, and **<1% at 30 MHz**, the band
  that owns the knee. Both are in `docs/measurement-model.md` §9.
- [x] Window dynamic-range budget (S4.8), reported: a `phi ~ 0`
  foreground at `|P|/I = 0.15` through BH4 leaks **3.7e-6 peak** (first
  sidelobe, `phi ~ 10.8`), **<=1e-6 beyond `phi = 27.5`**, **1.4e-7
  across the 30 MHz knee** and **<=8.8e-8 beyond `phi = 200`**. BH4 is
  therefore *adequate* against both bracket ends everywhere the roll-off
  lives, and inadequate only inside its own main lobe (`phi <~ 10`).
  The spec's "3.8e-6 everywhere, inadequate against the 1e-6 floor" read
  a single peak-sidelobe level as a flat floor. Table in
  `docs/measurement-model.md` §12; measured by
  `tests/test_dispersion.py::test_foreground_sidelobe_budget`.
- Figure provenance: all step-5 figures regenerate from committed scripts
  on this branch; the refuted Step 2/4 figures live only at the
  audit-2026-08-18 tag. The mixed-provenance list is empty here.
- [x] **Reframed from shape/localisation to DETECTION.** The user's
  point, and it is right: we will never resolve a peak in Faraday depth
  and map it to real Galactic structure. What matters is the SNR of
  *any* evidence of Faraday rotation and whether that evidence is
  degenerate with systematics. Those are different statistics from the
  tail gate and they give different answers, and the earlier write-up
  led with a localisation verdict as though it were the detection one.
- [x] **`scripts/step5_detection.py`** -- detection SNR against the
  low-depth systematics cut. Instrumental `I -> Q,U` leakage is
  spectrally smooth and sits at `phi ~ 0`, so a statistic integrating
  all depths is degenerate with it; the cut is what breaks the
  degeneracy, and the BH4 window budget puts it at `phi >= 27.5`
  (leakage `<= 1e-6` of I beyond there). **The cut is cheap: it keeps
  ~27% of template power and costs only ~2.2x in sensitivity at every
  band**, because the truncated template still spans most of the axis.
- [x] **Tail-gate error, corrected.** The published gate compared
  `A_bracket x sqrt(f)` -- a TRUNCATED signal -- against `A_mf` for the
  FULL template. Inconsistent: a statistic that looks only above the
  cut has only the truncated shape to match against, and its threshold
  is 28-35% higher. `step5_detection.py` recomputes the threshold on
  the truncated template at every cut. Verdicts unchanged, numbers
  ~26% lower.
- [x] **`H_lst`** added to `step5_template.npz` (fiducial `k` only,
  3 x 128 x 2500, 7.7 MB) so the transit-time tail numbers are exact
  rather than a GC-transit power fraction against an LST-averaged
  threshold. **Both arms re-run.** Note the trap this exposed:
  `step5_template.py` writes the same path regardless of `--lst`, so a
  smoke run overwrites the production product -- it did, once.
- [x] **Delay (`tau`) is FEASIBLE, and the argument against it was
  overstated.** `tau_FD = 2 phi c^2 / (pi nu^3)` is monotonic in `phi`,
  so a cut in one basis is exactly a cut in the other with an identical
  retained power fraction: every detection number is basis-independent.
  The chirp smears by **2.0% (30 MHz) / 1.2% (50 MHz)** of the
  template's OWN extent, because the signal is a broad distribution
  spanning hundreds of resolution elements, not a peak -- the chirp
  only destroys a *narrow* feature, i.e. localisation. Aliasing costs
  1.8% / 0.0% of the kept signal at 30 / 50 MHz. Only 10 MHz fails, on
  **aliasing** (115%), and it was already out. Measured and unexplained:
  the delay-Nyquist wall and the zoom depth horizon agree to 0.1% at all
  three bands (2796.4/2796.9, 604.0/604.1, 22.4/22.4).
- [x] **The cut's separation from the origin is basis-independent and
  ranks the bands.** At 30 MHz `phi >= 27.5` is 3.6 amplitude
  resolution elements out; at 50 MHz only 0.8 -- inside one element.
  So although 50 MHz has the larger raw SNR, **30 MHz is the band where
  the cut is cleanly executable**; at 50 MHz the separation from
  `phi ~ 0` leakage rests entirely on window sidelobe control. A better
  reason to lead with 30 MHz than the knee.
- [x] **Renamed `delay` -> `depth` through the code and docs.** The
  axis is `phi` in rad/m^2, the conjugate of `lambda^2`; it is NOT the
  `tau` in milliseconds that the refuted Step 4 transformed onto.
  `dispersion.delay_power` -> `depth_power`; prose in `dispersion.py`
  (which now carries an explicit naming warning), `response.py`,
  `instrument.py`, both step5 scripts, `docs/measurement-model.md`,
  `CLAUDE.md`, `AGENTS.md`. **Deliberately NOT renamed**: the
  PROGRESS.md lines describing the old Step 2/4 work, which are a
  genuine record of a genuine delay analysis. The branch name keeps the
  old word because PR #4 is open on it.
- [x] **`report/faraday_depth_template/`** (was `report/delay_template/`)
  -- restructured around detection: SNR-vs-cut as the headline section
  and figure, the two bases as their own section, shape / resolution /
  knee demoted to what sets the template, the tail gate moved to a
  localisation section with corrected numbers. `cd
  report/faraday_depth_template && make`.
- [x] **No number in the report is typed by hand.**
  `scripts/step5_tables.py` writes `generated.tex` -- macros and table
  bodies -- straight from the npz, and the `.tex` `\input`s it.
  Transcription had failed twice (the tail gate above; the closed-form
  "within a few percent" claim, which is really 0.980 / 1.161 / 0.542).
  `generated.tex` is TRACKED, like the figures, so the report builds
  from a fresh clone.
- [x] **`notebooks/faraday_depth_template.ipynb`** (was
  `faraday_delay_template.ipynb`) -- executable companion, committed
  executed, restructured to match the report and deriving the detection
  table, the chirp/aliasing table and the resolution table live.
- [x] `step5_plots.fig_detection` and `fig_weight_map` added;
  `fig_template_family` rewritten after review -- colour now means
  geometry and linestyle means treatment (they were two apart in the
  property cycle, so the fiducial's solid was green and its own dotted
  companion red, with the dotted curves absent from the legend), and
  `k -> -1` is drawn as a marked point with a legend entry saying it is
  `delta(phi)` rather than as an invisible curve at the axis edge.
- [x] `.gitignore`: `report/` -> `report/*` plus exceptions. git does
  not descend into an excluded DIRECTORY, so a bare `report/` makes the
  exceptions impossible to express. Verified with `git check-ignore -v`.

## Possible follow-ups
- [ ] Transfer selected figures/text into the paper (explicitly out of
  scope per INSTRUCTIONS-LPY.md: "don't change the paper yet").
- [ ] Ionospheric-regime study: dedicated run with small phi (1-30
  rad/m^2) uniform screen to emulate the lunar ionosphere at 10 MHz.
- [ ] Noise: propagate radiometer noise through the zoom bins to turn
  the depth-space signature into a detectability forecast.

## Key decisions / conventions (pinned)
- Response frame: x=East, y=North, z=zenith (proper rotation);
  phi = 90° − azimuth. Port order N,E,S,W = 0,1,2,3.
- Sky Q/U in healpy/COSMO convention; U_IAU = −U_COSMO when feeding
  croissant. Faraday: (Q+iU) e^{+2iφλ²}.
- Fixed 30/10/50 MHz beam across the narrow band; only the Faraday
  phase is chromatic → all depth-space power is Faraday-induced.
- Real maps at native nside=512 RING, never degraded; Haslam file is
  RING, WMAP K NESTED (check ORDERING headers!).
- Time: 1024 samples over one lunar sidereal day (periodic).
- Fine grid 16384 × 25 kHz/2048; 3 parents + 3×64 zoom bins inside.
- Outputs in `generated_data/` (fine waterfalls are 2.1 GB memmaps),
  caches in `generated_data/cache/`, figures in `report/figures/`.
- Heavy jobs: background + `ulimit -v 16000000`, logs in
  `generated_data/`.
