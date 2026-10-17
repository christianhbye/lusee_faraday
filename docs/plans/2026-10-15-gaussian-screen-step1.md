# Gaussian-screen benchmark implementation and validation plan

Scope: deliver step 1 for Christian's review, then stop. The specification is
`docs/specs/2026-10-15-gaussian-screen-step1.md`.

1. Specify the physical ensemble, tangent-plane geometry, admissible covariance,
   dimensional correlation length, deterministic source, ideal complex Jones
   pair and frozen reference plane. Derive the paired off/on statistics,
   including both branches, their joint covariance and the mean difference.
2. Implement immutable model/quadrature configuration and reusable scalar
   separation integration in `src/lusee_faraday/gaussian_screen.py`. Supply
   a stable special-function evaluation of the same integral, explicit
   constant-screen limits, and frequency-moment assembly. Import the pinned
   frequency/phase convention. Do not import the pixel validation arm.
3. Create a reproducible notebook builder/executor in
   `scripts/gaussian_screen_step1.py`. Produce the executed
   `notebooks/gaussian_screen_step1.ipynb`, embedded captions and figures,
   figure copies under `report/figures/`, and configuration/provenance
   in `generated_data/gaussian_screen_step1/`. Record UTC, git state,
   package versions, input/source SHA256, notebook input hash, physical and
   numerical configurations, assertions, run time and peak RSS. Key generated
   numerical artifacts by a configuration/code fingerprint; rerun rather
   than reuse an unrelated cache. No observational input maps are needed.
4. Run inexpensive controls and reduced-integral convergence checks. Render
   and inspect the figures, check executed cells for errors, format/lint
   changed code, and test only the new numerical module. Exercise deliberate
   branch/centering mutations against the independent constant-screen checks
   before treating those checks as regression evidence. Update TODO and
   PROGRESS with exactly what was completed and what is unvalidated.

## Predeclared step-1 checks

- m=sigma_phi=0: on equals off; centered moments and squared difference vanish.
- sigma_phi=0, m nonzero: deterministic phase rotation matches the pinned
  phase helper; centered covariance vanishes, mean difference need not vanish.
- ell=infinity: a uniform random depth rotates every sightline identically;
  each realization retains polarized amplitude. Integrate directly over a
  standard normal depth with Gauss-Hermite nodes at a weak-rotation frequency
  set, compare means, Hermitian and nonconjugated second moments and squared
  off/on difference to 2e-11 times their unrotated scales. Doubling nodes
  must change the reference by less than this tolerance.
- Reduced integral: default and tightened rescaled quadrature, and an exact
  special-function evaluation, agree within 2e-9 times the unrotated moment
  scale across the declared physical parameter sweep. Report the worst cell.
  Small ell tends to zero centered variance; large ell tends to the constant
  screen result. Check positivity and Hermiticity, with roundoff tolerance
  2e-11 times the covariance scale. These algebra/integration checks do not
  constitute deterministic spatial-screen validation.

## Step 2: proposed independent deterministic comparison (not executed)

Use the same source, beam, screen covariance, Jones coefficients and paired
screen across all frequencies. Start at 45, 47.5 and 50 MHz with the baseline
sigma_phi and m; also use controls with both zero and with ell=infinity.
No fitting of analytic amplitude or correlation length is permitted.

Generate a stationary screen by a spectral construction normalized under the
specified Fourier convention. Pin seeds, physical domain and a set of random
Fourier coefficients. Extend the physical domain and Fourier cutoff separately;
measure sample variance and separation covariance against K_phi before using
its visibility distribution. A periodic embedding or truncated spectrum is
an approximation whose wraparound and missing-power errors must be bounded.
A candidate initial box spans 16w in each direction and is at least 12ell
wide. The source/beam Gaussian tail outside the square is bounded by its
radial tail exp[-L^2/(8w^2)] (L is box width). Increase box size if this
fails the visibility error budget. For Gaussian S(k), missing power beyond
k_max is exp[-ell^2 k_max^2/2]. Convert this depth-variance error to a
phase/visibility bound before choosing the cutoff: a conservative ensemble
RMS branch error is at most 2 x_max sigma_phi sqrt(missing-power fraction),
using |exp(i a)-exp(i b)| <= |a-b| and normalized positive W. Multiply by
the sum of absolute branch coefficients for a visibility RMS bound; retain
measured per-realization convergence as a separate requirement.
Account for the low-k discretization of the box independently of this bound.

For each fixed continuous Fourier realization, evaluate phi and directly sum
B times the original on/off Stokes integrand. Double angular quadrature
resolution without redrawing or rescaling the physical screen. Compare at
least three levels; a maximum per-realization change below 1e-4 of the
unrotated polarization amplitude at both final refinements is the initial
numerical gate. Refine the domain/spectrum and the angular quadrature
separately. Phase variations, not just ell, set the quadrature spacing.
Use an independent Cartesian integration implementation, not this separation
kernel. A grid-refined version of an unresolved phase sum is not a reference.

Only after that gate, grow independent realization count in batches (initial
pilot 64, then 128 and 256 if affordable and separately authorized). Report
means, E[V_i V_j*], centered C_ij, nonconjugated centered moments, and
E[|V_on-V_off|^2] for every frequency pair and real/imaginary component.
Use independent batches or a bootstrap across screens to estimate Monte
Carlo standard errors of each estimator; frequency samples are not independent
screens. Preserve the bias convention of sample covariance (N-1) versus raw
second moments (N). Report convergence shifts and the worst standardized
discrepancy, plus all cells. A proposed joint gate is no cell beyond the
Bonferroni two-sided 99% normal threshold over the declared real components,
with adequate pilot verification of the error estimates; add measured
quadrature/spectral bounds to the uncertainty budget. Examine systematic
residuals even if this gate passes. A single small ensemble pass is not proof.
Persist seeds and per-realization outputs to avoid rerunning completed work.

Step 3 later decreases frequency at fixed physical parameters, resolves the
shrinking coherence region, and tests the asymptotic formula against the
full model and converged simulations where feasible. Step 1's illustrative
strong-rotation predictions do not satisfy that gate. Step 4 introduces the
actual two-branch LuSEE response, real sky variation, channels, foreground
treatment and noise only after review of steps 2--3. None of those jobs is
part of this delivery.


## Discussion record — 2026-10-16

Christian asked how the Gaussian notebook relates to the original pixel
sum, the existing RM map, and eventual LuSEE-Night measurements. This section
records the explanation and the recommended next sequence. The inference
work below is proposed, not executed or an agreed replacement for the
existing validation gates.

### Physical model and numerical distinction

The source and beam are fixed Gaussian-shaped profiles. The Faraday screen
is the Gaussian random field; one drawn screen stays fixed across frequency.
Only the screen varies across the ensemble. Sections 1--6 predict ensemble
moments; Section 6b directly pixel-sums drawn maps and supplies a small pilot.
Measurements are calculated from screens, not drawn from an assumed Gaussian
visibility distribution. There is currently no receiver noise.

The continuum phase integral is bounded, not divergent. An unresolved pixel
quadrature can leave an incorrect, resolution-dependent cancellation residual.
Averaging Gaussian phase differences analytically replaces oscillatory factors
with a smooth coherence kernel determined by the screen structure function.
Stationarity and circular source/beam symmetry then reduce the angular-pair
integral to one separation integral. Averaging and symmetry do different
jobs. This predicts ensemble moments, not the exact spectrum of a specified
map, and the narrow coherence region still needs convergence checks.

An individual screen produces spectral wiggles. Averaging over different
screens can suppress those wiggles, leaving a smooth mean and pointwise
standard deviation. Their frequency coherence resides in off-diagonal
covariance, which the pointwise band cannot display. The pilot does not yet
validate that full covariance.

### Interpretation and visualization of covariance

Proposed views are signed covariance/correlation slices at reference
frequencies, delay-space covariance, dominant spectral eigenmodes, and
individual measurement spectra alongside the ensemble prediction. Delay
power is the diagonal of the frequency covariance transformed on both axes;
retain cross-mode correlations where they matter. Account separately for a
nonzero mean when predicting raw power, and apply the same window/channel
response to simulations and observations.

The analogy with 21-cm tomography versus a power spectrum is useful: a map
or spectrum is a realization, while covariance describes statistical
structure. Fourier modes diagonalize covariance under appropriate stationarity
assumptions. This Faraday model generally depends separately on both
wavelength-squared coordinates, so a delay-power curve need not contain all
of its second-moment information.

### Existing RM map and the real-beam extension

No screen covariance is required to calculate the visibility conditional on
a fully specified RM map. That calculation still needs converged angular
integration. Covariance is needed to describe an ensemble or to model
uncertain/unresolved structure. A statistical replacement does not recover
the fixed-map spectrum simply by averaging its phases.

Keep the existing RM map as a deterministic baseline. A possible extension
is phi(n) = phi_map(n) + delta_phi(n), with the residual's meaning stated:
posterior uncertainty in the fixed sky and physical small-scale fluctuations
are distinct modeling questions. Do not add physical power already represented
by the baseline or equate map-pixel size with a coherence area.

Resolved angular correlations can be estimated from RM observations using
structure functions or angular spectra, with explicit treatment of noise,
large-scale Galactic variation, sampling and spatial-averaging assumptions.
Unconstrained smaller scales require stated models and sensitivity tests.
The Faraday 2020 release provides posterior means and per-pixel standard
deviations; those products alone do not specify joint spatial uncertainty or
physical sub-resolution turbulence. See the
[data description](https://wwwmpa.mpa-garching.mpg.de/~ensslin/research/data/faraday2020.html).
No observationally justified residual covariance has yet been selected here.

Analytic Gaussian phase averaging survives arbitrary fixed source/response
weights and a general valid Gaussian-screen covariance. Gaussian spatial
profiles make the present separation weighting analytic. Isotropic screen
statistics may permit a separation reduction with numerically calculated
weights for other beams/skies; spatially varying Galactic statistics,
curved-sky polarization transport and both response branches require a new
implementation and validation. Efficient real-response evaluation has not
been demonstrated. WMAP supplies a fixed polarized-sky template, not the
screen statistics; extending its emission behind a foreground screen remains
a physical assumption.

### Connection to measured data and proposed inference sequence

The data are the 16 real four-port products versus time and frequency. The
new ensemble covariance describes correlations among those readouts; it is
distinct from the electric-field coherency that defines each port product.
The real instrument observes one sky and has no Faraday-off switch. Paired
off/on simulations remain useful controls, but their difference is not
itself an observed data vector. Detection must compare model predictions
through the same instrument, channel averaging and foreground processing,
allowing competing foreground/instrument explanations comparable freedom.

Recommendation: complete the toy covariance validation, then demonstrate a
small inference problem in the toy model before building a full realistic
inference pipeline. Strong-rotation claims retain the existing step-3 gate;
a weak-rotation inference demonstration can establish only its stated regime.

1. Plot and validate frequency covariance against converged, independently
   generated screen realizations, including numerical and ensemble errors.
2. Use independent mock observations for inference validation. Initially
   hold the source/beam fixed and test one screen parameter at a time; then
   expose degeneracies when intrinsic polarized amplitude or smooth source
   terms are allowed to vary. Identify a band where the parameter changes
   an observable before choosing it as a recovery target.
3. Use the toy model's independent readouts. Q and U suffice for ideal x/y
   dipoles: Vxx, Vyy and Vxy are redundant combinations with fixed intensity.
   Do not treat them as independent measurements or manufacture 16 independent
   toy channels. Add an explicit illustrative noise model if needed; state
   its covariance, channel widths and integration time, and distinguish it
   from a LuSEE noise prediction.
4. Test Faraday-on versus off using the observed mock readouts, without
   supplying the hidden off realization to the inference. Calibrate detection
   errors on null simulations and assess parameter bias/interval coverage
   across independent screens. A Gaussian screen does not imply a Gaussian
   likelihood for its nonlinear readouts: validate a moment-based likelihood
   or calibrate a simulation-based statistic. Detection must also survive
   declared nuisance alternatives, not merely reject an exactly known smooth
   off spectrum.
5. Extend to the actual frozen LuSEE response and WMAP-based sky in stages,
   then propagate channel response, calibration, foreground treatment and
   radiometer noise. Preserve the paired on/off simulations and evaluate
   recovery with the same processing used for data. Include correlations
   across ports and times; one physical screen persists throughout.

Beam integration generally prevents unique RM-map reconstruction. The first
inference targets should be identifiable statistical combinations, not a
promised screen image. Angular correlation scale is distinguishable from
physical length only with distance information; polarized amplitude and
screen fluctuation parameters may be degenerate. Report what a single
realization constrains and what remains prior/model dependent. Passing toy
recovery establishes the inference procedure only within that toy model.
