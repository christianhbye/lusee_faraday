# Gaussian-screen inference: implementation and validation plan

Status: planned on `gaussian-screen-inference`, 2026-10-16. PR 7 was
squash-merged as `3bf50ce`. Three local review commits excluded from that
merge have been carried onto this branch: physical overview, math rendering,
and the discussion record. This delivery writes the plan; it launches no
simulation or inference job.

## Objective and scope

Demonstrate whether a single noisy toy observation can distinguish the
specified Faraday screen from zero rotation and constrain a screen correlation
scale. Preserve the connection to the original paired on/off calculation,
while giving the inference only quantities an observer could measure.

First validate the full frequency covariance against converged direct screen
sums. Then demonstrate inference under known source/beam assumptions and
measure its failure or degradation when nuisance parameters are introduced.
Use the same flat-plane foreground-screen model as the
[step-1 specification](../specs/2026-10-15-gaussian-screen-step1.md).
The [previous plan](2026-10-15-gaussian-screen-step1.md) supplies the numerical
convergence protocol and the discussion motivating this sequence.

The real LuSEE four-port response, WMAP emission, a physically justified
Galactic residual-screen model, mixed emission/rotation and observing forecasts
remain later extensions. Strong-rotation conclusions require the previous
plan's step-3 validation; successful weak-rotation inference does not replace it.

## Observable and model families

Use ideal x/y readouts for the main presentation. Retain I and the displayed
products Vxx=(I+Q)/2, Vyy=(I-Q)/2 and Vxy=U/2, but fit the independent real
vector y=(Q(nu_1),...,Q(nu_n),U(nu_1),...,U(nu_n)). Do not count derived
products as independent data. The old complex Vpq readout remains an algebra
control, rather than the inference observable. Specify all values in
beam-averaged K and covariance in K^2 at the same reference plane.

Each mock uses one independent physical screen across every frequency and
both readouts. Use direct spatial sums for validation and mock observations;
keep their random seeds separate from model calibration and null tests. The
analytic likelihood predicts the ensemble distribution, not the exact hidden
screen map. Keep paired on/off truth for diagnostic plots only.

Start with the existing source/beam and screen mean/variance. A candidate
initial window is 17 monochromatic frequencies over 45--50 MHz; confirm its
sensitivity before committing production samples. Inspect 25--35 MHz only
after separate direct-sum convergence in that stronger-rotation regime.
Changing window is a declared configuration change, not an adaptive choice
based on a validation realization's detection score.

Two distinct exercises avoid conflating detection with parameter estimation:

- Detection: compare H0, m=sigma_phi=0, with H1, the baseline nonzero m and
  sigma_phi and a declared grid of correlation scales. Fit shared source
  nuisance terms under both hypotheses when they are introduced. Calibrate
  the scan's null distribution empirically; H0 is not obtained by sending
  correlation length to zero. A deterministic nonzero constant depth is a
  separate control, not H0.
- Recovery: generate H1 observations at known correlation scales and infer
  ell/d, initially fixing source, beam, m and sigma_phi. A candidate grid is
  41 logarithmic ell values from 0.05 to 5 pc at fixed d=100 pc. Refine the
  grid and expand boundaries when needed before claiming intervals. Report
  angular scale as the measured quantity; physical ell assumes known d.

## 1. Close the covariance validation prerequisite

Follow step 2 of the previous plan: pin continuous screen realizations,
separately refine angular quadrature, domain size and Fourier cutoff, and
check the screen generator's variance and separation covariance. Preserve
zero-rotation, deterministic-depth and random spatially constant controls.

Extend analytic moment assembly to joint Q/U frequency covariance, including
Q-U cross moments. Reuse the phase kernel, but validate readout assembly
against direct Stokes sums. For a complex intermediate, retain both
C=E[delta V delta V^dagger] and J=E[delta V delta V^T] when constructing its
real/imaginary covariance; C alone is insufficient. Avoid forming an enormous
angular pixel-pair matrix.

Compare every mean, centered covariance, raw second moment and expected
squared on/off difference with direct realizations and their sampling errors.
Apply the previous plan's three-level quadrature criterion (both final shifts
below 1e-4 of unrotated polarized amplitude), independent domain/spectrum
bounds, and declared simultaneous moment-comparison gate. Numerical error
must be smaller than the Monte Carlo error used to judge agreement. The
existing 64-screen pilot is not a substitute for these checks.

Persist per-screen outputs and grow ensembles in batches only after the
numerical gate. Record worst discrepancies and all cells, not just diagonal
power or band averages. Failure blocks inference claims in that regime.

## 2. Explain what frequency covariance predicts

Produce signed correlation slices at declared reference frequencies, full
Q/U covariance blocks, and dominant eigenmodes beside individual realizations.
Show that equal pointwise variance can accompany different frequency
coherence. Display the paired on/off MSE as squared mean shift plus variance.

Transform both frequency axes with the same declared window and normalization
to display delay covariance. Distinguish its diagonal centered power from
raw power, which also contains the transformed mean. Preserve cross-mode
correlations; do not assume stationarity in frequency or wavelength squared.
Read the project's figure skill before implementing these plots.

A useful falsifiable control removes off-diagonal covariance while preserving
the diagonal. Compare its predicted spectral structure and calibrated
inference with the full model. It need not always perform worse; record the
regimes where off-diagonal information adds little.

## 3. Define a nonsingular observation and likelihood

The noiseless screen statistics alone can be singular, and an exactly known
noiseless off spectrum makes rejection trivial. Introduce a documented,
positive-definite Gaussian readout-noise covariance N in K^2 for the first
algorithm demonstration. Specify its per-channel standard deviations,
correlations and frequency dependence in configuration. Choose declared noise
levels relative to the unrotated signal before scoring validation mocks.
This synthetic noise is a controlled algorithm parameter, not a LuSEE
radiometer forecast; attach no integration-time or sensitivity claim to it.

The observation model is y=signal(screen)+noise. Start with the analytic
mean mu(theta) and covariance S(theta)+N. Evaluate a candidate multivariate
Gaussian likelihood using a factorization/solve and log determinant, never
an explicit inverse. Include the covariance normalization when parameters
change S. Record conditioning and convergence; do not silently replace
negative eigenvalues or add parameter-dependent numerical jitter. Diagnose
redundant readouts or integration errors before regularizing.

Gaussian phi does not imply Gaussian y. Check the candidate likelihood on
independent direct-screen mocks, including whitened residuals, selected
projections, tails and interval coverage. If it fails, either restrict the
validated regime or calibrate a simulation-based statistic; do not label
uncalibrated moment-based likelihood contours as confidence regions.

Keep monochromatic samples explicit at first. Before claiming a finite-channel
measurement, integrate means and covariance over declared channel responses,
with paired frequency quadrature for covariance, and use the identical
estimator on the mocks. Verify center-sample error before simplifying it.
A future physical noise model must derive its normalization/correlations from
the readouts, channel bandwidth and integration time and check the radiometer
prediction.

## 4. Detection and parameter recovery from individual observations

Separate seed sets for model development, null-threshold calibration and final
validation. Never subtract the exact hidden on mean, supply the per-screen
off truth to the estimator, or tune the frequency window/priors on final mocks.
Initially fixing a known source is an explicit toy assumption; relax it next.

For detection, use a predeclared likelihood-ratio or calibrated discrepancy
statistic with the same frequency mask, noise and preprocessing under H0/H1.
Account for the correlation-scale scan in the null calibration. Target a 5%
false-positive rate for the initial demonstration; quote binomial uncertainty
and detection probability separately. Do not claim five-sigma sensitivity
from a small ensemble. Determine sample counts from the desired precision;
start with a resource-measured pilot and then pin the independent calibration
and validation counts before scoring them.

For recovery, first inspect whether the covariance changes distinguishably
across the candidate scale grid at the selected noise/window. Use profile
likelihood or a stated prior and posterior, clearly identifying which one.
Across independent screens at at least two interior truth scales, report
bias, boundary-hit fraction and empirical coverage of nominal 68% and 95%
intervals, with sampling uncertainty. Refine the parameter grid and verify
interval stability. A broad or one-sided interval is a valid finding.

Predeclare the coverage criterion after the pilot, before final validation:
nominal coverage should lie in a simultaneous 95% binomial acceptance region
across the chosen truth/noise configurations. Report deviations even when the
formal gate passes. Validate final conclusions on a disjoint screen sample.
Never count frequency channels as independent screen realizations.

## 5. Expose nuisance and model dependence

Repeat the controlled exercise while allowing a shared polarized-source
amplitude and angle, then smooth spectral variation, under both hypotheses.
Fit nuisances from mock data. If using a fixed linear foreground projection,
apply it to the observation, mean, full signal/noise covariance and competing
models; fit in an independent basis of its retained subspace. State any
signal information removed. Compare held-out predictions on the same
coordinates with a nuisance alternative of comparable freedom, using joint
conditional predictions where frequency cells are correlated.

Only then explore joint sigma_phi/correlation-scale recovery and covariance
shape misspecification. Identify amplitude/coherence degeneracies explicitly.
The initial Gaussian-shaped K is a benchmark choice; future observational
constraints may require other shapes or non-Gaussian residuals. A successful
fit within the generating family is not evidence that it describes the Galaxy.

## Implementation increments and artifacts

1. Add joint-readout moment support and meaningful synthetic controls to
   `src/lusee_faraday/gaussian_screen.py` and its tests. Independently validate
   branch signs, Q-U cross covariance, centering and real-vector assembly.
2. Extend the screen driver for resumable convergence and full covariance
   validation. Separate physical configurations, numerical grids and seeds.
3. Add a small inference module and CLI driver only after the moment gate.
   Test parameter-dependent normalization and a known analytic limit; inject
   the named mistakes and require the checks to reject them.
4. Build a companion `notebooks/gaussian_screen_inference.ipynb` from a
   reproducible script, using explicit xx/yy/xy and Q/U labels. It should read
   cached ensemble results instead of launching them during execution.
5. Deliver one readable result record: mock observation, model and residual;
   covariance diagnostics; null calibration and recovery coverage; nuisance
   sensitivity; validated scope and remaining failures. Update TODO and
   PROGRESS after each coherent committed increment.

Store numerical results under `generated_data/gaussian_screen_inference/`,
reusable checkpoints under `generated_data/cache/`, and figures under
`report/figures/`. Fingerprint source, physical/numerical configuration,
channels, noise, priors, seeds and dependency versions. Missing matching
products trigger regeneration or a clear error, never fallback to stale data.

Measure pilot wall time and peak memory before setting ensemble sizes.
Run heavy jobs in the background under `ulimit -v 16000000` with absolute
log paths under `generated_data/`; checkpoint completed screens. Do not
modify upstream luseepy/croissant, external paper sources or old audit ledgers.

Completion means the validated toy inference succeeds or its limitations are
quantified with calibrated errors. It does not require manufacturing a
parameter detection. The next model decision should follow those results.
