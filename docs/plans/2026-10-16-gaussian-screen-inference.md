# Gaussian-screen covariance and inference: implementation plan

Status: revised after paper-direction review on `gaussian-screen-inference`,
2026-10-16. PR 7 was squash-merged as `3bf50ce`; the physical overview, math
rendering and discussion record were carried onto this branch. This revision
changes the work sequence; it launches no simulation or inference job.

## Objective, paper deliverable and scope

Validate the residual polarization amplitude and full frequency covariance
predicted by a physical correlated screen, and identify what remains
observable after source nuisance fitting. Connect those results to the
paper's existing diffuse search template. A small calibrated inference
exercise tests that framework; correlation-length recovery is conditional
on identifiable information and is not a required scientific outcome.

The sequence is full covariance validation, information and nuisance
assessment, strong-rotation/template comparison, then one small calibrated
inference demonstration. Preserve the original paired on/off calculation as
a diagnostic while giving inference only quantities an observer could measure.
Use the flat-plane foreground-screen model in the
[step-1 specification](../specs/2026-10-15-gaussian-screen-step1.md).
The [previous plan](2026-10-15-gaussian-screen-step1.md) supplies the
independent convergence protocol and strong-rotation gate.

The required paper-facing record must state the surviving power's physical
normalization, the validity or failure of the existing depth-power covariance
approximation within declared bands, and the information lost to nuisances.
It must distinguish validated predictions, conditional assumptions and
unresolved errors. A quantified failure or lack of identifiability is a
useful result; it must not be hidden by selecting a favorable mock or window.

The real LuSEE four-port response, WMAP emission, physically justified Galactic
residual statistics, mixed emission/rotation and observing forecasts remain
extensions. One controlled non-Gaussian test belongs here. Map diagnostics,
intermittent fields and reconstruction-latent-field work have their own
[deferred plan](2026-10-16-galactic-residual-screen-models.md).

## Observable and model families

Use ideal x/y readouts for presentation. Retain I and the displayed products
Vxx=(I+Q)/2, Vyy=(I-Q)/2 and Vxy=U/2, but fit the nonredundant real vector
y=(Q(nu_1),...,Q(nu_n),U(nu_1),...,U(nu_n)). Its entries are correlated;
derived products are not additional independent measurements. The old complex
Vpq readout remains an algebra control. Specify values in beam-averaged K
and covariance in K^2 at the same reference plane.

Each mock uses one independent physical screen across every frequency and
both readouts. Use direct spatial sums for validation and mock observations;
keep seeds separate from model development and null-threshold calibration.
The analytic likelihood predicts the ensemble distribution, not the exact
hidden screen map. Keep paired on/off truth for diagnostic plots only.

Start with the existing source/beam and screen mean/variance. The candidate
initial window is 17 monochromatic frequencies over 45--50 MHz. Report the
RMS polarization phase 2*sigma_phi*lambda^2, its change across the window,
and ensemble attenuation exp(-2*sigma_phi^2*lambda^4), computed from config.
Call it the initial validation window: small phase variation across a band
does not imply small absolute rotation. Inspect 25--35 MHz only after separate
direct-sum convergence. Window selection follows the information assessment
below and is frozen before final mocks; channels are not independent screens.

Use three model families with the same source nuisances and noise:

- H0: zero rotation, m=sigma_phi=0.
- Hc: a deterministic spatially constant depth c, fitted from the data.
  Its screen covariance vanishes. A random spatially constant screen remains
  a separate ensemble control.
- Hs: the baseline nonzero m and sigma_phi, with a declared correlation-scale
  grid. Initially hold these depth parameters fixed; any later freedom must
  be explicit. H0 is not the ell -> 0 limit.

Report rejection of zero rotation, evidence for departures from uniform
rotation, and correlation-scale recovery separately. Calibrate Hs comparisons
against both H0 and Hc; passing the first does not establish spatial structure.

A candidate scale grid is 41 logarithmic ell values from 0.05 to 5 pc at
fixed d=100 pc. Refine and extend it using development data before final
validation. Report angular scale ell/d conditional on the source/beam widths;
physical ell also assumes known d. The covariance depends on ell/w, where
w=d*(s_src^-2+s_beam^-2)^(-1/2) for angular widths in radians. Uncertain
source/beam width therefore remains a geometric degeneracy even at known d.

## 1. Close the full covariance validation prerequisite

Implement pinned continuous Fourier realizations before extending the screen
driver. The existing draw_depth generates grid-shaped white noise: repeating
a seed on a different grid does not preserve a continuous screen. Store
physical Fourier modes and coefficients separately from evaluation points;
refine angular quadrature by evaluating the same field. For cutoff refinement,
retain existing coefficients and add the prescribed missing modes. Document
how domain refinements are coupled and compare their ensemble errors;
changing the box changes its Fourier basis. Never rescale a realization to
its sample variance.

Separately refine angular quadrature, domain size and Fourier cutoff, and
check the generator's variance and separation covariance. Follow the previous
plan's tail, missing-power and low-k bounds. Preserve zero-rotation,
deterministic-depth and random spatially constant controls. Validate small,
intermediate and large scales spanning the proposed grid, including its
endpoints; baseline-only validation cannot establish the whole scan.

Extend analytic moments to joint Q/U frequency covariance, including Q-U
cross moments. Reuse the phase kernel, but validate readout assembly against
direct Stokes sums. For a complex intermediate retain both
C=E[delta V delta V^dagger] and J=E[delta V delta V^T] when constructing
real/imaginary covariance. Avoid a dense angular pixel-pair matrix.

Compare every mean, centered covariance, raw second moment and expected
squared on/off difference with direct realizations and their sampling errors.
Apply the previous plan's three-level angular criterion (both final shifts
below 1e-4 of unrotated polarized amplitude), independent domain/spectrum
bounds and declared simultaneous moment-comparison gate. Numerical error
must be smaller than the Monte Carlo error used to judge agreement. Preserve
the N-1 sample-covariance convention and N raw-moment convention. The existing
64-screen pilot is not a substitute for these checks.

Persist per-screen outputs and grow ensembles in batches after numerical
convergence. Record worst discrepancies and all cells, not just diagonal
power or band averages. Failure blocks claims in that regime.

## 2. Assess information and nuisance dependence before recovery

Introduce a documented positive-definite synthetic Gaussian noise covariance
N in K^2. Configure per-channel standard deviations, correlations and frequency
dependence; declare noise levels relative to the unrotated signal before
scoring validation mocks. This is an algorithm parameter, with no LuSEE
integration-time or sensitivity interpretation.

The ensemble mean is independent of ell in this model. At ell << w, the
leading signal covariance scales as P_ref^2*(ell/w)^2 at fixed depths and
frequencies. Derive and check this limit against the full integral. When the
mean is suppressed or absorbed by nuisances, amplitude and scale can become
degenerate. Do not interpret covariance normalization alone as an independent
measurement of angular structure.

Compare covariance amplitude, normalized correlation shape and parameter
responses across the grid. Inspect eigenmodes relative to N and expected
model distinguishability after nuisance profiling. Use local curvature only
as a diagnostic; check finite parameter changes with the full model and
pilot mocks. Compare fixed-source results with the following nested source
models, shared across H0, Hc and Hs:

- Free polarized amplitude P_ref and intrinsic angle chi; retain the specified
  spatial profile, I_ref and spectral law, with 0 <= P_ref <= I_ref and chi
  periodic modulo pi.
- Additionally free the common polarized spectral index beta.
- As a declared sensitivity test, add one polarized spectral-curvature term
  gamma*log(nu/nu_ref)^2 to log amplitude, with angle still constant. Require
  |P_0| <= I_0 over the fitted window. Any frequency-dependent intrinsic angle
  model is a separate, explicitly specified alternative.

Profile these nuisance parameters in the initial exercise. Pin beta/gamma
bounds and the Hc depth search range after the pilot, before calibration;
report fits on those boundaries. Assess Hc calibration across its declared
nuisance domain, rather than only at one depth. Add source-width uncertainty
as a geometry sensitivity test before interpreting scale constraints beyond
the fixed-profile model. Joint sigma_phi/ell inference is deferred.

Use pilot precision and expected distinguishability to select the window,
noise levels and a small set of truth configurations. Save that decision and
its criteria before production. If scale changes are absorbed by nuisances
or remain below the declared discrimination target, stop scale recovery and
report the degeneracy or bound; retain the covariance/template deliverable.

Produce signed correlation slices at declared reference frequencies, full
Q/U covariance blocks, dominant modes and individual realizations. Display
paired on/off MSE as squared mean shift plus variance. Transform both axes
with the same declared window/normalization for delay covariance, separating
centered diagonal power from raw power containing the transformed mean.
Preserve cross-mode correlations and do not assume frequency or lambda^2
stationarity. Read the figure skill before implementing these plots.

For a frequency-correlation ablation, set signal blocks between distinct
frequencies to zero while preserving each same-frequency 2x2 Q/U block.
Keep N fixed. Calibrate this statistic on the same physical mock ensemble
as the full model; record where cross-frequency information adds little.
Removing same-frequency Q-U covariance is a separate ablation.

## 3. Establish the strong-rotation result and template connection

Before an extensive recovery campaign, perform the previous plan's step-3
study: increase wavelength at fixed physical screen/source parameters,
resolve the narrowing separation kernel, compare the full integral with
controlled asymptotic limits and converged direct sums where feasible.
Demonstrate that residual power follows physical coherence scales independently
of arbitrary pixel area. Report the strongest validated regime and cost.
An unconverged direct sum cannot validate a strong-rotation prediction.

Compare the validated toy covariance with the existing independent-depth
power template, whose source-normalized single-branch covariance depends on
x-x', with x=lambda^2. The screen covariance also depends on x*x'. Derive
any narrow-band approximation that freezes that dependence, retaining both
polarization branches and their joint moments in Q/U assembly. Compare the
models in the same coordinates, reference plane and estimator.

For each declared band, report diagonal amplitude error, correlation-shape
error and noise-weighted covariance error; check the mean separately. Derive
the template normalization from the physical screen and angular weighting.
If a free template normalization is fitted to isolate shape error, show that
comparison separately from the absolute-amplitude prediction. Neither an RM
histogram nor numerical pixel area supplies physical coherence power.

State where the template is justified, needs revision or fails, with numerical
uncertainties and the approximation's error budget. This comparison validates
only the conditional toy model; it does not revalidate saved real-sky forecasts.
If strong-rotation validation fails, record the failure and limit the template
comparison to validated windows. A small initial-window inference check may
still proceed, but cannot close the paper's strong-rotation question.

## 4. Validate a small observation likelihood and calibrated inference

Use y=signal(screen)+noise, analytic mean mu and total covariance T=S+N.
Evaluate a candidate multivariate Gaussian likelihood with a factorization,
solves and log determinant, including parameter-dependent normalization.
Record conditioning and solver convergence; do not silently replace negative
eigenvalues or add parameter-dependent numerical jitter. Diagnose redundant
coordinates or integration errors before regularizing.

Moment agreement within Monte Carlo errors alone is insufficient for inference.
For T=L*L^T, evaluate numerical mean errors through L^-1*delta_mu and covariance
errors through L^-1*delta_S*L^-T. Set tolerances relative to the decision and
interval precision before production, and verify statistic/interval stability
under tighter numerical settings across the retained grid and noise levels.
Small errors in low-variance modes must not control the result.

Gaussian phi does not imply Gaussian y. Check whitened residuals, selected
projections, tails and coverage on independent direct-screen mocks, including
large-scale and boundary regimes. If the Gaussian likelihood fails, restrict
the validated regime or use its likelihood ratio as a simulation-calibrated
statistic; do not infer confidence levels from an unvalidated likelihood shape.

Separate seeds for development, threshold/interval calibration and final
validation. Never supply hidden per-screen off truth, subtract the exact hidden
on mean, or tune windows and nuisance bounds on final mocks. Use the same
frequency mask, preprocessing, nuisance model and noise in every comparison.
Account for the correlation-scale scan and nuisance fits in null calibration.
For Hc, report false-positive rates across the declared depth/nuisance domain
and use a threshold controlling the tested domain, or explicitly restrict the
claim to the validated configurations.

Target a 5% false-positive rate for the demonstration; report detection
probability and binomial uncertainty separately for H0 and Hc comparisons.
Pin an acceptable absolute calibration error, confidence level and simultaneous
comparison rule, then derive independent calibration/validation sample counts
from those targets and the measured pilot cost. Include threshold-calibration
uncertainty. A wide acceptance region from a small ensemble is not evidence
of adequate calibration; label resource-limited results inconclusive.

If the information assessment supports recovery, use profile-likelihood
confidence sets, with thresholds calibrated at candidate truths using direct
mocks when asymptotic thresholds are unsupported. Define nuisance treatment
and its validated domain. Across at least two interior truth scales and
representative grid edges, report bias, boundary-hit fraction and coverage of
nominal 68% and 95% sets, with simultaneous sampling uncertainty. Freeze the
allowed coverage deviation and sample counts before final validation; require
uncertainty bounds narrow enough to establish that tolerance. Report failures
even when a broad nominal-coverage test would pass. Refine the grid and check
interval stability. Broad, disconnected or one-sided sets are valid outcomes.
Bayesian posterior inference is deferred; credible intervals must not be
relabeled as confidence intervals if added later.

Keep monochromatic samples explicit. Before any finite-channel claim,
integrate means and covariance over declared channel responses, using paired
frequency quadrature for covariance and the identical mock estimator. Check
center-sample error before simplifying. A later physical noise model must
follow the readouts, bandwidth and integration time and pass radiometer checks.

If a fixed foreground projection is introduced, propagate it through data,
mean and full signal/noise covariance and fit an independent retained basis.
State what signal information it removes. Compare held-out predictions with
a nuisance alternative of comparable freedom on the same coordinates, using
joint conditional predictions for correlated frequency cells.

## 5. One controlled non-Gaussian stress test

Use phi=m+A*g, where g is a zero-mean Gaussian field with covariance K and
A is an independent positive random amplitude per screen with E[A^2]=1.
Choose one declared nonconstant distribution for A and retain A=1 as control.
The ensemble mean and covariance of phi are matched exactly; do not renormalize
individual draws. This global scale mixture tests higher-order statistics,
not spatial intermittency. Derive its phase moments by conditioning on A as
an additional control where tractable.

Independently converge direct readouts and compare their means, full frequency
covariance and calibrated inference with the Gaussian-screen prediction.
Report approximation errors relative to the chosen noise. Equal RM covariance
does not ensure equal phase statistics. Separate physical-screen
misspecification from failure of the Gaussian readout likelihood. Restrict
this test to the selected benchmark configurations; larger non-Gaussian and
real-map studies belong to the deferred plan.

## Implementation increments and artifacts

1. Add pinned Fourier-realization/evaluation support, joint-readout moments
   and independent synthetic controls. Validate branch signs, Q-U covariance,
   centering, real-vector assembly and preservation of a field under refinement.
2. Extend the driver for resumable full covariance convergence across scales.
   Separate physical configurations, numerical grids and seeds; persist outputs.
3. Produce the information/nuisance assessment and freeze a small experiment
   configuration with numerical, discrimination and calibration tolerances.
4. Complete the strong-rotation study and comparison to the existing template,
   or document failed gates and the restricted validated regime.
5. Add the small inference module/CLI and matched-covariance stress test.
   Test normalization and a known analytic limit; inject named mistakes and
   require the corresponding checks to reject them before committing tests.
6. Build notebooks/gaussian_screen_inference.ipynb from a reproducible script,
   with explicit Q/U and xx/yy/xy labels, reading cached ensembles. Deliver a
   result record containing data/model/residual, covariance and template
   comparisons, identifiability, calibrated errors, nuisance sensitivity and
   validated scope. Update TODO and PROGRESS after each committed increment.

Store numerical results under generated_data/gaussian_screen_inference/,
checkpoints under generated_data/cache/ and figures under report/figures/.
Fingerprint source, physical/numerical configuration, channels, noise,
nuisance bounds, statistical procedure, seeds and dependency versions. Missing
matching products trigger regeneration or a clear error, never stale fallback.
Measure pilot wall time and peak memory before setting ensemble sizes. Run
heavy jobs in the background under ulimit -v 16000000 with absolute log paths
under generated_data/ and checkpoint completed screens. Do not modify upstream
luseepy/croissant, external paper sources or old audit ledgers.

Completion requires the paper-facing covariance/template assessment and a
scoped inference result or quantified information/calibration limitation.
Failed scientific gates remain open and visible; completion of this benchmark
does not establish Galactic validity or a LuSEE sensitivity forecast.
