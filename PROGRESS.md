# Progress — four-port Faraday analysis (INSTRUCTIONS-LPY.md)

Updated: 2026-08-18 — **all five steps complete + user-driven
refinements** (zenith-calibrated polarimeter, report restructure,
zoom deconvolution, per-band weight verification + Table 1, Fig 9
as 2x3 all-band snapshot); report in `report/report.tex` (14 pp,
compiles clean, all figures referenced in text).

## Resume state (read this first after /clear)

- Everything is committed AND pushed on branch `luseepy-version`
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
  in `report/`.
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
  code there; import from `lusee` only.
- [x] `tests/test_fourport.py` — 10 data-free unit tests; all pass.
- [x] OOM diagnosis: the "mysterious session kills" were the kernel
  OOM killer (croissant dense spherical transform ~(lmax+1)^2·npix·16
  bytes).  Mitigations: small validation skies, single-channel
  response slices, `ulimit -v` + background for every heavy job.
- [x] `scripts/validate_engine.py` — ALL VALIDATIONS PASSED:
  [1] point source vs FullStokesCalibratorSimulator 8.6e-16;
  [2] diffuse-sky transport vs FullStokesCroSimulator worst 1.1e-2
  (nside=32/lmax=48 sky); [3a] NUFFT internal 6.4e-15;
  [3b] Faraday-rotated sky vs CroSimulator 4.2e-4.
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
