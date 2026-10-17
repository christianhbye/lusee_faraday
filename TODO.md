# LuSEE Faraday work queue

Updated 2026-10-16. **Christian agreed to the Gaussian-screen benchmark and
validation sequence. This is the next scientific task.** The long real-beam
audit remains pending review of its execution package. The consolidated
status, paper outline, proposed commands and scientific gates are in
[docs/paper-preparation.md](docs/paper-preparation.md). Machine-readable
provenance and existing-product values are in
[docs/paper-preparation-status.json](docs/paper-preparation-status.json).

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

## Next: derive and validate the Gaussian-screen benchmark

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
**Stop here for Christian's review; steps 2--4 remain unexecuted.**
Section 6b adds drawn-screen illustrations (Q, U, Vxx, Vyy, Vxy) and a
64-screen pilot at 30/40/50 MHz; it is consistent with the analytic moments
but is not the step-2 gate (no frequency covariance, bootstrap errors or
domain/spectrum refinement).

### 2. Recover converged deterministic simulations at weak rotation

- [ ] Generate realizations of the specified screen and compute the original
  off/on visibilities in a regime where direct quadrature can converge.
- [ ] Refine each realization's angular quadrature while holding its physical
  screen fixed. Separately increase ensemble size to measure sampling error.
- [ ] Compare simulated means and frequency covariance with the analytic
  prediction against both numerical and ensemble uncertainties.
- [ ] Recover zero-rotation and spatially constant-screen limits; a uniform
  screen must not introduce spurious angular depolarization.

### 3. Demonstrate convergence at strong rotation

- [ ] Increase rotation strength through wavelength while holding the source,
  beam and physical screen statistics fixed.
- [ ] Demonstrate convergence of the covariance separation integral as its
  coherence region narrows. Compare any small-angle approximation with the
  full toy-model integral and available analytic limits.
- [ ] Show that surviving power is independent of arbitrary pixel area.
  Grid-dependent amplitude or failed quadrature convergence rejects the
  proposed numerical cure; a failed direct simulation is not a reference.
- [ ] Record the validated regime and computational cost. Separate numerical
  validation of this model from evidence that it describes the Galactic sky.

### Proposed bridge: interpret covariance and demonstrate toy inference

The [2026-10-16 discussion record](docs/plans/2026-10-15-gaussian-screen-step1.md#discussion-record--2026-10-16)
records the fixed-map/ensemble distinction and connection to observations.
The recommendation is to validate a small toy inference problem before the
realistic inference pipeline; this work has not started. It retains the
step-2 covariance and step-3 strong-rotation gates for the regimes it claims.

- [ ] Visualize signed frequency-covariance slices, delay covariance/power,
  eigenmodes and individual spectra; distinguish centered and raw power.
- [ ] Recover an identifiable screen parameter from independent toy mock
  observations; assess null detection errors, bias and interval coverage.
  Use independent readouts and validate the likelihood/statistic rather
  than assuming Gaussian measurements or observing an off counterfactual.
- [ ] Expose nuisance degeneracies before extending to real LuSEE/WMAP
  weights. Preserve a fixed-RM-map baseline separately from residual-screen
  models; constrain or bracket their spatial correlations explicitly.

### 4. Extend the validated model to LuSEE and assess physical uncertainty

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
- [ ] Update detectability and permitted paper claims only after these gates.

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

- [ ] Review the proposed outline and figure/evidence map in the handoff.
- [ ] Locate/select the manuscript source; the historical external path is
  absent here. No external paper source has been edited.
- [ ] Transfer validated instrument/point-source material and the corrected,
  conditional diffuse-detection result into a paper with generated numbers.
- [ ] Verify literature supporting physical amplitude scenarios and turbulence
  inputs; audit references, figures, quoted values and provenance.

## Optional cleanup and deferred work

- [ ] Decide what to do with untracked `audit/scripts/plot_shuffle.py` and
  `The Pixel Sum Never Converged.pdf`; do not circulate the old briefing's
  amplitude/power-weight explanation without correcting it.
- [ ] Prune the stale local `origin/luseepy-refactor` tracking reference if
  desired; this is housekeeping, not an analysis prerequisite.
- [ ] Dedicated small-depth/ionospheric study at 10 MHz, if needed for scope.
  The older June forecast branches remain archived; their diffuse amplitudes
  came from the refuted pixel sum.
