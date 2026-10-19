# LuSEE Faraday work queue

Updated 2026-10-19. **The paper framing is adopted: "how Faraday rotation
manifests in LuSEE Night spectra", mechanisms and magnitudes, no detection
promise.** The presentation order and the work queue are in
[the paper arc](docs/plans/2026-10-19-paper-arc.md): column screen with a
stated sub-map structure family, emitting depth as a second axis, local
medium as its far end, then form and range, then inference. Next: run the
column-anchored screen through the real-instrument pipeline. The current plan to the paper, with phases, deliverables and
gates, is the
[plan section of the framing document](docs/plans/2026-10-17-paper-framing.md#plan-to-the-paper).
The numerical phases, conditional interpretation and permitted report are
implemented in [PR 9](https://github.com/christianhbye/lusee_faraday/pull/9).
The [implementation status](docs/plans/2026-10-17-paper-implementation.md)
and [reproduction guide](docs/paper-implementation-guide.md) describe the
completed gates and products. Transfer into the protected manuscript and
comparison with future mission data follow separately. The sections below
this one are the earlier queue; the framing document lists
which of them are deferred or demoted. The Gaussian-screen benchmark remains
the validation reference for the covariance phase, with its open
fitted-beta acceptance check recorded in the
[executed notebook](notebooks/gaussian_screen_inference.ipynb) and
[result record](docs/gaussian-screen-inference-results.json).
[docs/paper-preparation.md](docs/paper-preparation.md) holds the earlier
consolidated status; its paper outline is superseded.

## Completed or superseded

- [x] Archive the old RM-synthesis/Fisher branches and close PRs 1/2.
- [x] Instrument refactor merged as PR 3 (`33c8730`, 2026-09-14).
- [x] PR 4 targets main; template implementation/report remain on
  `faraday-delay-template` (`adb4024`). The old fast-forward/rebase cleanup
  instructions are obsolete: main was squash-merged; do not execute them.
- [x] Record the October survey in one handoff, including environment drift.
- [x] Preserve September audit results and pre-edit copies of script/TODO.
- [x] Prepare corrected BGL audit, cheap preflight and bounded staged launcher.
- [x] Add small numerical, storage, provenance and restart checks.

## Gaussian-screen benchmark and remaining scientific gates

Keep the original Faraday-off/on visibility comparison as the organizing
question. Predict its ensemble mean and covariance by averaging unresolved
phases analytically before numerical integration. The derivation and proposed
benchmark are in [the handoff](docs/paper-preparation.md#proposed-first-benchmark-a-correlated-gaussian-foreground-screen).
Agreement on this approach does not establish that the model describes the
Galaxy or that the existing depth-histogram forecast is valid.

### 1. Specify the model and derive the observable

- [x] Record the two-sky versus ensemble distinction and physical corrections.
- [x] Agree on the first benchmark: a smooth deterministic polarized source,
  a simple beam and a correlated Gaussian foreground Faraday screen.
- [x] Specify an admissible screen covariance, its variance and physical
  correlation length, the intrinsic sky and beam, and the frequency range.
  Keep physical correlation scales independent of numerical pixel size.
- [x] Derive off/on means, second moments and centered frequency covariance
  from that same model. Retain the mean contribution to squared differences.
- [x] Reduce the angular integral to a separation integral where symmetry
  permits. Resolve the narrow coherence region analytically or with rescaled
  quadrature; avoid a dense all-pixel-pairs matrix.
- [x] Define convergence criteria and falsifiable comparisons before running
  the benchmark; record configuration and provenance with each result.

Step 1 is delivered for review in the executed
[notebook](notebooks/gaussian_screen_step1.ipynb), with a
[specification](docs/specs/2026-10-15-gaussian-screen-step1.md),
[implementation and step-2 plan](docs/plans/2026-10-15-gaussian-screen-step1.md),
and [generated check/provenance record](docs/gaussian-screen-step1-results.json).
The model is an admissible Gaussian screen on a flat plane, with physical
correlation length, a smooth Gaussian source and beam, and a realizable ideal
complex Jones pair. Checks cover algebra, constant-screen limits and the
reduced separation integral. They do not validate spatially varying screen
simulations, the Galaxy, real-beam amplitudes or the depth-histogram forecast.
**Step 1 was reviewed and squash-merged as PR 7. The full covariance and
strong-rotation gates below now pass; calibrated detection is complete with
one unresolved acceptance check.**
Section 6b adds drawn-screen illustrations (Q, U, Vxx, Vyy, Vxy) and a
64-screen pilot at 30/40/50 MHz; it is consistent with the analytic moments
but is not the step-2 gate (no frequency covariance, bootstrap errors or
domain/spectrum refinement).

### 2. Recover converged deterministic simulations in the initial window

- [x] Generate realizations of the specified screen and compute the original
  off/on visibilities in a regime where direct quadrature can converge.
- [x] Refine each realization's angular quadrature while holding its physical
  screen fixed. Separately increase ensemble size to measure sampling error.
- [x] Compare simulated means and frequency covariance with the analytic
  prediction against both numerical and ensemble uncertainties.
- [x] Recover zero-rotation and spatially constant-screen limits; a uniform
  screen must not introduce spurious angular depolarization.

### 3. Demonstrate convergence at strong rotation

- [x] Increase rotation strength through wavelength while holding the source,
  beam and physical screen statistics fixed.
- [x] Demonstrate convergence of the covariance separation integral as its
  coherence region narrows. Compare any small-angle approximation with the
  full toy-model integral and available analytic limits.
- [x] Show that surviving power is independent of arbitrary pixel area.
  Grid-dependent amplitude or failed quadrature convergence rejects the
  proposed numerical cure; a failed direct simulation is not a reference.
- [x] Record the validated regime and computational cost. Separate numerical
  validation of this model from evidence that it describes the Galactic sky.

Validation covers five monochromatic frequencies in 45–50 MHz at five
correlation scales, plus 20/25/30/35 MHz at the baseline scale. The first
angular-grid failures remain recorded; refined grids pass the unchanged
criterion. Lower-frequency direct simulations and Galactic interpretation
remain outside this result.

### Implemented: covariance, paper connection and scoped inference

The [inference implementation plan](docs/plans/2026-10-16-gaussian-screen-inference.md)
now prioritizes full covariance validation, information and nuisance assessment,
strong-rotation/template comparison, then a small calibrated inference exercise.
The new 128-screen ensembles satisfy the covariance gates. The frozen
experiment has 4096 calibration and 4096 independent validation noise draws
per null configuration, and 1024 final physical screens per alternative.

- [x] Implement pinned continuous Fourier realizations and converge joint Q/U
  moments across small, intermediate and large correlation scales.
- [x] Visualize covariance and individual spectra; separate centered/raw power.
  Assess scale information after profiling source amplitude, angle and spectral
  terms, including the amplitude/scale and source-width degeneracies.
- [x] Validate strong-rotation predictions and compare the physical covariance
  with the independent-depth template's amplitude and frequency structure.
  Record failures and restricted scope before further recovery work.
- [x] Freeze a small window/noise/truth configuration and quantitative numerical
  and calibration tolerances. Run comparisons against zero rotation and a
  fitted constant-depth screen, with the same source nuisance freedom.
- [x] Quantify the lack of scale information and source-width ambiguity.
  Recovery is disabled by the declared criterion; no scale intervals or
  coverage claims are made. Numerical refinement changes no detection decisions.
- [x] Run one matched-covariance non-Gaussian scale-mixture stress test.
- [ ] Resolve the fitted-beta/full-covariance Hc calibration acceptance check
  in a separately frozen follow-up: observed false-positive fraction 5.47%,
  simultaneous upper bound 6.61% versus the 6.5% acceptance limit.
- [ ] Assess sensitivity to broader source spectral models before interpreting
  null rejection as spatial Faraday structure. The baseline alternative hits
  beta bounds in 89.6% of fitted Hc trials; current rates are conditional on
  beta in [-4, -1] and the specified source truth.

Map diagnostics, intermittent fields and reconstruction-latent-field work are
in the [deferred Galactic-model plan](docs/plans/2026-10-16-galactic-residual-screen-models.md).
They remain prerequisites for Galactic interpretation, not for completing the
conditional benchmark. Preserve a fixed-map baseline and distinguish physical
unresolved structure from reconstruction uncertainty.

### 4. Extend the validated model to LuSEE and assess physical uncertainty

- [ ] Define the next observational target before further inference scans:
  distinguish bulk rotation, departures from uniform rotation and scale
  recovery. The benchmark supports no scale-recovery claim; assess angular
  scale/source-width/amplitude ambiguity and whether independent intensity
  or source constraints can help. I was not included in this inference.
- [ ] Assess a fixed Hutschenreuter baseline plus statistical residual
  delta-phi as a candidate, beginning with an inventory of joint posterior
  information and resolved scales. Specify the uncertain residual and avoid
  double-counting power already represented by the baseline. The local map's
  mean and per-pixel errors alone do not define spatially coherent draws.
- [ ] Introduce WMAP spatial variation and the actual frozen four-port response,
  including both polarization branches and their joint statistics.
- [ ] Propagate actual channel responses through signal and noise. Define and
  propagate foreground projection or marginalization; a low-depth template
  cut alone is not a data filter.
- [ ] Identify observational constraints on unresolved screen correlations;
  vary correlation scales and structure-function shape within stated models.
  Keep physical turbulence distinct from RM-map reconstruction uncertainty.
- [ ] Extend the derivation to emission mixed with rotating material, stating
  intrinsic polarization and angular/line-of-sight correlations explicitly.
- [ ] Determine whether the derived covariance reduces to the current
  independent-depth power-histogram model within each narrow band. Justify,
  revise or replace that approximation before interpreting saved forecasts.
- [ ] Derive beam-integrated amplitude consistently with the covariance;
  correct slab-envelope/floor and mean-polarization/power interpretations.
- [ ] Test whether the competition between intrinsic brightening and
  fractional depolarization produces a minimum in realistic absolute
  polarized-power spectra. The sampled toy minimum near 35 MHz is a
  candidate prediction, not a Galactic result. Test ensemble and individual
  spectra with the same beam/channel estimator, leakage treatment and noise
  bias correction; compare source/instrument alternatives. See the
  [scoped follow-up](docs/plans/2026-10-16-galactic-residual-screen-models.md#candidate-spectral-minimum).
- [ ] Update detectability and permitted paper claims only after these gates.

Choose the physical residual model and source freedom before a new recovery
or forecast campaign. Any revised detection procedure needs a separately
frozen calibration; preserve this benchmark's failed acceptance check.
Analytic covariance does not remove the need to converge each validation
screen's direct integral. Small scatter within an ensemble does not establish
robustness to choosing a different ensemble.

## Separate track: review and complete the real-beam numerical audit

This diagnoses the original pixel-sum failure. Passing it does not validate
the replacement physical covariance model or block the analytic benchmark.

- [ ] Christian reviews the handoff, script changes, resource budget and paper
  outline. No long job is authorized yet.
- [ ] Choose/confirm the numerical environment. Current editable dependencies
  differ from the August pins; exact installed versions/revisions are recorded.
  The pre-existing `pyproject.toml` and `uv.lock` edits need separate review.
- [ ] Run corrected 512 smoke at both snapshots using the prepared launcher;
  inspect real-response compatibility, exit status and measured peak RSS.
- [ ] Run corrected 512/1024/2048 ladder, preserving completed v2 checkpoints.
  **Recompute all three resolutions**: September's script had a negative-branch
  sign error and mislabeled a phase-cancellation statistic as effective count.
- [ ] Close native-RM/shuffled/incoherent comparisons and positive controls
  per snapshot/pair/branch, with/without the 86-degree horizon selection.
  Record incomplete or failed controls; a memory failure is not a null result.
- [ ] Make a tracked compact summary and update audit/README.md with the
  corrected results and provenance. Keep the September ledger unchanged.

The September run reached 48 results each at 512 and 1024 and none at 2048;
its log stops after making the 2048 maps. Memory pressure is plausible but
no retained OOM record identifies the exact termination. The v2 script has
bounded source batches, disk-backed weights and a new fingerprinted ledger.
Read-only preflight: `bash audit/run_bgl_audit.sh plan`.

## Correct the template report

Work from `faraday-delay-template`; its scripts and LaTeX sources are absent
from main. Preserve gitignored products when arranging an isolated checkout.

- [ ] Replace the coherent pixel-sum POWER weight in `step5_intuition.py`
  with the appropriate linear complex amplitude, per pair/snapshot. Use
  fixed-alm fields, not `ud_grade`.
- [ ] Regenerate `step5_intuition.npz`, `step5_bridge.pdf` and the `walk*`
  macros. Rewrite the associated random-walk section/caption coherently.
  The factor-ten floor/concentration story was an artifact of the weight.
- [ ] Pin the amplitude weight with a falsifiable regression check.
- [ ] Correct the general complex cross-pair equation, finite-k integrability
  condition, and distance/depth-fraction descriptions before copying prose.
- [ ] Preserve the distinction between coherent amplitudes and incoherent
  depth power; the amplitude/power correction does not imply that the
  downstream template weighting was wrong.
- [ ] Revalidate relevant template/noise gates in the selected environment,
  regenerate figures/macros/notebook and rebuild the report.

## Finish scoped systematics after selecting the physical model

Reassess these scans against the validated covariance model before running
them; they cannot establish the model's physical basis on their own.

- [ ] B: vary the polarized spectral index spatially; quantify shape,
  detection-threshold and band-ratio sensitivity. Label any total-intensity
  index as a proxy and handle map zero levels/negative values explicitly.
- [ ] A1: justify and bracket physical `sigma_eff`, including beam weighting
  and band dependence. Separate RM-map reconstruction error from turbulence.
- [ ] A2: add co-spatial turbulence to the depth-power distribution and scan
  its effect. Broadening power does not determine coherent amplitude.
- [ ] Quantify relative band gain and spectral-index uncertainty in the
  proposed slab/dispersion spectral-slope discriminant.
- [ ] Include receiver noise at the loaded reference plane and beam-weighted
  antenna temperature. Default-zero amplifier noise is not a sky-domination
  check, especially at 50 MHz.
- [ ] Review numerical versus physical template stability, coherence-dependent
  shape, amplitude scenarios and permitted paper claims together.

## Produce the paper

Superseded 2026-10-17 by the
[plan to the paper](docs/plans/2026-10-17-paper-framing.md#plan-to-the-paper).
The items that survive from the earlier list are its Phase 5: locate or
create the manuscript source, transfer the PR 3 instrument and point-source
material, and audit references, figures and quoted values. The
conditional diffuse-detection result is no longer a paper deliverable.

## Optional cleanup and deferred work

- [ ] Decide what to do with untracked `audit/scripts/plot_shuffle.py` and
  `The Pixel Sum Never Converged.pdf`; do not circulate the old briefing's
  amplitude/power-weight explanation without correcting it.
- [ ] Prune the stale local `origin/luseepy-refactor` tracking reference if
  desired; this is housekeeping, not an analysis prerequisite.
- [ ] Dedicated small-depth/ionospheric study at 10 MHz, if needed for scope.
  The older June forecast branches remain archived; their diffuse amplitudes
  came from the refuted pixel sum.
