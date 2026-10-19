# Progress — four-port Faraday analysis (INSTRUCTIONS-LPY.md)

## Implementation PR and report complete — 2026-10-17

Completed the conditional-prediction implementation in PR 9 into
`paper-implementation-plan`. The 15-page report contains all nine figure
products, derived input/horizon/channel/exposure/decision tables and clean
provenance. The final build at source `2bd3103a90b5` verifies 29 result
records and 697 referenced artifacts. All pages and the final changed
captions were inspected; every figure page passes the 7 pt font minimum.
LaTeX and the mechanical audit have no overfull boxes, undefined references,
placeholders or serious findings. The figures-to-report runner passes.

The final PDF and generated tables/macros/manifests are committed under
`report/paper-*`; exact build and visual checks are in
`docs/paper-report-build.json` and `docs/paper-report-qa.json`. The full
suite remains 227 passes plus its pre-existing two-port reference failure;
smoke has 34 passes. Both user notebook edits and upstream tracked files
are preserved. The external manuscript and future mission-data comparison
remain separate work.


## Deterministic figure regeneration — 2026-10-17

Removed PDF creation/modification timestamps from report figure writers so
an unchanged rerender preserves the checksums in its inspection record.
Two separate mixed-depth plot invocations produce identical PDF bytes;
`docs/paper-figure-reproducibility.json` records the check. The complete
figures-to-report runner is exercised after this source commit.


## Report build and printed-figure checks — 2026-10-17

The isolated report build verifies 29 result records and the complete hashes
of their referenced artifacts. LaTeX passes twice with no overfull boxes,
undefined references or oversized floats; the mechanical prose/reference
audit has no serious findings. All fifteen pages were inspected. The nine
figures now have at least 7 pt text at their printed 6.4-inch width, including
logarithmic tick labels. The preserved flat-plane benchmark has a separate
report renderer, leaving its original result and prototype figure unchanged.
The final PDF and its derived tables/provenance are committed after this
source; `docs/paper-report-build.json` identifies the exact build revision.


## Report source and complete reproduction runner — 2026-10-17

Prepared the numerical/figure manifest verifier, derived tables and LaTeX
number macros, and a clean temporary-worktree report builder. It preserves
unrelated notebook edits and rejects stale/corrupt inputs, source drift,
undefined references, overfull boxes and oversized floats. The complete
background runner adds explicit stages, an exclusive lock and a 34-check
smoke mode. The reproduction guide documents inputs, coordinates, covariance
units, conditional normalizations and the future calibrated-data path.
The PDF build and rendered-page checks follow this committed source.


## Complete science-product export — 2026-10-17

The all-16 real science-product factorization now exports the shared
loaded calibration, all channel/phase I-only means, and selected full
signal/noise covariance blocks. Compact four-branch weights combine with
all saved local-grid coefficients without allocating a complete frequency,
product and time cube. Fine system PSDs and channel/readout operators
support arbitrary requested noise blocks, across all nine receiver cases.

Projection back to Stokes checks every signal mode, a complete selected
extreme-column signal covariance, loaded means, and raw/residual noise at
both bands and phases. All gates pass; record:
`docs/science-product-results.json`. The science products use the pinned
PRODUCT_LABELS order and the same one-kelvin enclosure normalization as
the calibrated Stokes predictions.


## Measured-spectrum predictions and observing controls — 2026-10-17

Completed the simulated Q/I and U/I spectra with paired I-only/noise
controls, common-fit residuals and conditional RMS bands. Both bands use
preselected phase 0, with independent-noise repeated visits of the same sky.
The calibrated-spectrum input path also processes independent-noise mock
observations. Full Q/U cross covariance and complex C/J, correlated-power
sampling variances, named-frequency pair products and parent/zoom ratio
quantiles are exported. All 512-ensemble predictive-mean checks pass their
preset five-SEM gate; these are conditional Gaussian illustrations.

Figures now show Q/U covariance amplitudes as well as normalized correlations,
the complete signed-column separability range, native I-only beam controls
and same-phase repeat differences. The separate angular/fringe control uses
one common ell=48 observing cutoff, finite time bins and actual-site night
mask for sky, leakage and noise. A failed broad-spectrum trapezoid is
retained; integrating its known smooth intrinsic spectrum exactly restores
positivity without relaxing the gate.

The complete integration suite has 227 passes and the same historical
two-port reference failure as the initial baseline; the 34-check smoke
suite passes. CI is green on Python 3.12/3.13/3.14. Records include
`docs/faraday-separability-results.json`,
`docs/common-observing-results.json` and
`docs/paper-integration-validation.json`.


## Full signed local-column sensitivity — 2026-10-17

Completed 520 cases across the 13 signed uniform mean columns implied by the
adopted density/field/path pairs, both dispersions, both phase distributions,
five depth profiles and both bands. All preset depth, frequency, angular,
band-limit and omitted-mode gates pass. Frequency and depth refinements
bound the full Q/U covariance change, not just its diagonal. Eight
predeclared extreme-column screen/slab cases also pass complete joint C/J,
positivity and independent refinement checks.

The intrinsic normalization is polarized emissivity times the full assumed
5 pc volume (5 K RMS at 30 MHz for the unit-emissivity case). Depth integration
supplies depolarization without a second horizon cut. The largest residual
speckle/noise RMS ratio is 3.53e-4; including smooth polarization gives
7.49e-3. The brighter HII-based normalization brackets remain conditional
rescalings rather than measured local polarized priors. Full arrays retain
all receiver cases, both phases and unavailable exposures as NaN. Records:
`docs/local-medium-sensitivity-results.json`,
`docs/local-medium-joint-validation.json` and
`docs/local-medium-separability-results.json`.


## Exact marginal moments for the full local-column grid — 2026-10-17

Added an allocation-efficient diagonal contraction of the complete four-branch
Hermite operator. It agrees with full physical complex covariance, including
the relation and global-mixture conditional-mean variance. Both tests pass;
both deliberately injected faults are caught. Evidence is in
`docs/moment-diagonal-validation.json`. The joint mode coefficients remain
available for full cross-channel and cross-phase exports.


## Loaded chromatic measurement/noise stage — 2026-10-17

Completed all nine voltage/current cases at both bands and both phases.
Loaded sky, separate Moon/loss thermal power and amplifier PSDs are added
once; the same as-built calibration and physical-frequency fit act on the
linear/PCHIP leakage models and noise. All 36 raw/residual phase covariance
pairs pass positivity and independent R C R^T checks; deliberately dropping
one covariance leg makes every case fail. Arrays store covariance times
integration in K^2 s, divided by available exposure to obtain K^2.

The original native engine's clock is reproduced exactly before switching
to the pinned periodic production grid. Its lunar period differs by 0.4896 s;
the wrong-clock fault is caught without changing upstream or relaxing the
reference gate. Records are `docs/measured-system-results.json`,
`docs/measured-system-validation.json` and
`docs/native-clock-reference-validation.json`.


## Real-observable statistics guard — 2026-10-17

The correlated-power statistic now rejects complex covariance or mean
inputs explicitly, preventing accidental loss of the complex relation.
The existing invalid-input check covers both cases; all three statistics
tests pass and all four deliberate faults are caught. The regenerated
`docs/measurement-statistics-validation.json` records the updated sources.


## Native chromatic I-only beam controls — 2026-10-17

Computed all 30 native-frequency response slices for as-built, C4-symmetrized
and diagonal-ZA instruments at 30/50 MHz, with two preselected sidereal
phases and native nside=512 Haslam+CMB inputs. Beam and loading follow the
actual FITS native coordinates. Every lmax-32/48 and physical-covariance
gate passes; maximum truncation error is 5.47e-6 of I. Full complex pair
alms and raw loaded matrices are checkpointed so later mismatch comparisons
can use one common as-built calibration. The record is
`docs/chromatic-leakage-results.json`. Fine-channel leakage is an explicit
interpolation comparison in the following stage, not a measured beam model
below native resolution.


## Stable Gaussian visibility illustrations — 2026-10-17

Added mode-based draws of the joint unit-coefficient 1/i visibility
covariance before mixing phase coefficients across channels. This retains
tiny fitted residuals without resolving the tiny eigenvalues of a nearly
coherent full frequency covariance. Smooth and speckle samples are accumulated
separately; one mode realization is shared by all channels and phases and
reused on repeated nights. Two tests pass against 50,000 latent ensembles
and a nearly annihilated smooth-fit control. Both deliberate faults are
caught; evidence is in `docs/visibility-draw-validation.json`. These remain
conditional Gaussian visibility illustrations, not resolved Faraday maps.


## Common-fit Gaussian and mixed-screen production — 2026-10-17

Completed all 40 local-screen/profile/band models with the exact sidereal
boxcar and shared physical-frequency constant+slope fit, separately on
three parents and all 192 zoom bins. The global variance mixture retains
conditional-mean variation in its speckle term. Filtering acts on phase
coefficients before covariance assembly. Every smooth/speckle/total residual
passes full covariance positivity and direct R C R^T checks; all angular
and frequency refinements pass. Their absolute covariance errors remain
exported so numerically tiny residuals can be treated as bounds.

The production mixture combination also matches an independent finite
ensemble and catches three deliberately injected weighting/mean faults.
Records are `docs/residual-signal-results.json` and
`docs/residual-signal-validation.json`. Chromatic leakage, loaded noise and
simulated observations use this same estimator in the following stage.


## Strong-screen real-instrument controls — 2026-10-17

The direct frequency-pair solver passes 129/257-node and doubled-angular
checks at 30/50 MHz for an explicit sigma_phi=5 rad/m^2 foreground-column
control. This is separate from the assumed local-screen range. The maximum
frequency gate ratio is 0.220 and angular gate ratio 0.0118. The independently
normalized scalar white-floor covariance agrees with the full real readouts
to 5.8e-9 on their RMS scale, including cross terms and loading. Minimum
central zoom Q correlations are 0.555 at 30 MHz and 0.974 at 50 MHz, showing
a measurable-coordinate decorrelation contrast to the local cases. Evidence
and checkpoints are in `docs/strong-screen-results.json`.


## Independent full-channel scale-mixture control — 2026-10-17

Propagated a global 0.5/1.5 variance mixture through actual parent and folded
zoom responses in a matched two-source geometry, retaining full cross-
frequency Q/U covariance and relation. Independent bivariate Gaussian
integration uses exact screen/slab phase laws, with 64/128 order refinement.
All eight 30/50 MHz geometry/profile controls pass: covariance agreement is
better than 7.1e-13 on the covariance RMS scale. A Gaussian column with the
same RM covariance differs by up to 0.0890. Deliberately omitting global
conditional-mean variation makes every control fail. The records are
`docs/channel-mixture-validation.json` and its explicit fault companion.
The declared 33-node frequency interpolation and deterministic intrinsic
source geometry delimit this independent channel check.


## Matched spherical/flat deterministic geometry — 2026-10-17

Added a spherical Gaussian source/analyzer control with both intrinsic spin
phases parallel transported to the same center. Its separation probability
is computed by spherical Legendre window convolution, independently of the
flat-plane separation integral. The same source dilution and both complex
moments recover the PR 7/8 deterministic benchmark across 20--35 and
45--50 MHz. Worst covariance errors are 1.07e-4 at the original width and
7.17e-6 with widths divided by four; quadrature refinement is below 1.62e-6
on the flat RMS scale. Window mass/positivity are also checked. Removing
source dilution makes both controls fail by more than unity. Evidence is
in `docs/matched-geometry-results.json` and the explicit fault record.
The isotropic random production sky remains a different intrinsic ensemble.


## Correlated-spectrum statistics — 2026-10-17

Added the full Gaussian sampling variance of a mean-square spectrum,
including off-diagonal channel covariance and a nonzero deterministic mean.
Independent repeated measurements cancel the fixed sky and add estimator
noise variances. Exact fourth-moment cubature and fixed-sky controls pass;
all three deliberately injected faults are caught. Evidence is in
`docs/measurement-statistics-validation.json`. These statistics describe
conditional predictive scatter, not independent sky samples per channel.


## Local input and lunar plasma audit — 2026-10-17

Added primary-source checks of a nearby pulsar's historical DM/RM, its timing
distance, HII-region total synchrotron emissivity and SELENE occultations.
The audit identifies which path averages cannot determine few-parsec
polarization or angular RM statistics. Conditional density/field/path and
emissivity/polarization brackets are exported by `local_input_audit.py` in
`docs/local-input-audit-results.json`. The deliberately large assumed lunar
envelope (1000 cm^-3, 100 nT, 100 km) gives 2.6e-6 rad/m^2, or a complex
polarization phase below 4.8e-3 rad at 10 MHz. This is a conditional envelope,
not a measured universal nightside limit. The floor's emissivity, depth and
angular-coherence degeneracies remain explicit.


## Nightside exposure and sidereal averaging — 2026-10-17

Computed actual-site solar altitudes for complete 1/1024-sidereal-day bins
in illustrative 1/24-lunation windows starting at the shared simulation
epoch. At 50% electronics duty, phase 0 has 1/14 eligible visits and phase
0.25 has 0/14. The unavailable one-lunation second phase has no noise value.
Each eligible visit integrates 1152.6 s. The 24-lunation total per phase is
16136.9 s. These are simulation calendar exposures, not mission dates or
operations commitments. Signal and leakage use the same exact sinc(m)
boxcar on MEPA kernels. Four tests and four injected faults pass; records
are `docs/observing-exposure.json` and `docs/observing-validation.json`.


## Common measured-spectrum estimator — 2026-10-17

Implemented physical-frequency smooth templates through the actual channel
responses, weighted residual maps with explicit flags/rank/conditioning,
two-leg covariance projection and subtraction in K before I normalization.
Estimated-I ratios retain denominator uncertainty and I-Q/U correlations;
undefined denominators and saturated fits remain explicit. Six tests pass,
and all six deliberate faults are caught. Evidence is in
`docs/separability-backend-validation.json`; the estimator definition is in
`docs/specs/2026-10-17-common-measurement-estimator.md`.


## Loaded receiver and joint radiometer backend — 2026-10-17

Added explicit voltage/current PSD propagation through the full luseepy
parallel antenna/load network. Correlated input sources are supported and
checked for positivity after unit rescaling. Real science products and
calibrated pseudo-Stokes use Hermitian quadratic matrices, retaining every
inter-product covariance. Fine-frequency power weights propagate the full
noise overlap, including frequency-dependent system power.

Seven checks pass: independent Kirchhoff samples, exact Gaussian fourth-
moment quadrature, production readout conventions, luseepy variances,
integration scaling and 20,000 independent channelized voltage ensembles.
All nine deliberate faults are caught. Evidence is in
`docs/receiver-noise-validation.json`; definitions and units are in
`docs/specs/2026-10-17-receiver-radiometer-noise.md`.

Audited Tamura et al. (2024) for the input-referred EM voltage scales.
The 2.2/3/5 nV/√Hz cases are a PA bench value, an integrated EM value and
a requirement, respectively. Current noise, source correlations and thermal
temperatures remain declared assumptions in `receiver-noise.json`; missing
legacy Model54 files are not replaced by zero receiver noise. Native
chromatic leakage for all three response variants is computing separately.

## Mixed-depth production predictions — 2026-10-17

Computed and checkpointed all five prescribed profiles for both conditional
local columns: 30 native-band cases at 10/30/50 MHz and 20 channel cases
at 30/50 MHz, now including all three parents and all 192 zoom bins at
two preselected sidereal phases. Smooth, centered and total covariances
are checked separately. All 50 pass independent depth/angular/beam and,
where applicable, frequency-grid refinements; foreground screens also
match the previous frequency-pair solver. The largest gate ratios are
0.430 (beam truncation), 0.133 (angular), 1.8e-7 (frequency) and 2.2e-4
(foreground reference). The maximum mode degree is 1553; the propagated
omitted-mode bound uses at most 3.8e-4 of its absolute error allowance.

Built and inspected `report/figures/faraday_redistribution_spin2.pdf` with
its PNG, caption and manifest. It separates smooth and centered Q/U power
and loaded RMS for each profile. Results and verification are in
`docs/mixed-depth-results.json` and `docs/mixed-depth-validation.json`.
The intrinsic E/B sky, local dispersion/angular scale and low-frequency
turnover remain explicit assumptions. Leakage/noise and the measured
spectra are the next stage; these signal-only results are not a forecast.

## Mixed-depth spherical/channel backend — 2026-10-17

Implemented the shared-column depth average as a normalized Hermite series
in `depth_channel_covariance.py`. Depth averaging and actual channel
weights act on coefficients before covariance assembly, retaining the
smooth/centered split and every helicity/readout term. Poisson tails bound
omitted phase power; an absolute covariance bound follows through the
angular/readout weights. Unresolved depth order or mode limits raise an
error. A stable logarithmic/deviance form recovers high-order power even
when the mean underflows, without relaxing the unit-power gate.

Nine new tests pass against exact fractional screens, independent Gaussian
averages of the slab sinc law, adaptive Gaussian-profile integration and
explicit double-depth/angular/channel covariance. All ten deliberately
injected faults are caught. Evidence is in
`docs/depth-channel-backend-validation.json`; the derivation and scope are
in `docs/specs/2026-10-17-depth-channel-covariance.md`. Production mixed-depth
channel arrays and their independent resolution checks are the next task.

## Real-instrument channel predictions and mixture control — 2026-10-17

Saved four conditional foreground-screen cases at 30/50 MHz with the
as-built loaded instrument, three parent bins, 64 central-parent zoom bins
and two sidereal phases. Every case passes the preset 17/33 frequency-node,
doubled angular-resolution and lmax-64/96 comparisons. The gate is 1% of
on-covariance RMS scale plus 1e-8 of off-covariance RMS scale; the largest
error/gate ratio is 0.430. Covariances, smooth/speckle components, raw
eigenvalue diagnostics, Gaussian draw seeds and channel-overlap matrices
are checkpointed with hashes. The record is
`docs/faraday-covariance-results.json`.

Built and inspected `report/figures/faraday_channel_covariance.pdf`, its
PNG, caption and manifest. It shows the loss of Q channel correlation and
separate conditional Q/U draws in each band, with common scales and
marginal RMS bands. Image pixels are centered on the nominal channel
frequencies; a final unchanged rerun verifies all four moment caches.
The assumed sigma_phi=0.02 case remains highly coherent
across 25 kHz: minimum Q correlations are 0.9999885 at 30 MHz and 0.99999990
at 50 MHz. These are conditional model values, not data or a sensitivity
forecast. The band illustrations do not model joint cross-band covariance.

The PR 8 equal-weight 0.5/1.5 variance mixture preserves mean/RM covariance
but changes phase statistics. Independent bivariate quadrature at seven
frequencies in 20–35 and 45–50 MHz and three separations agrees with its
conditional mixture moments within 4.2e-14; order 256/512 refinement shifts
them by at most 3.1e-14. A Gaussian screen with the same RM covariance
differs by up to 0.0741 in dimensionless raw phase moments. Evidence is in
`docs/spin-screen-mixture-results.json`. This verifies phase marginals,
not a complete spherical realization or visibility Gaussianity.

Final local integration suite: 193 passed, one unchanged historical
two-port reference failure. All 24 covariance faults are caught; together
with Phase 1 this adds 42 passing tests and 39 caught faults. CI passes on
Python 3.12, 3.13 and 3.14 for the covariance implementation. Black and
flake8 pass. The consolidated evidence is
`docs/spin-channel-covariance-validation.json`; continuation order and
remaining scientific gates are in `docs/implementation-handoff.md`.

## Spin-2 instrument and channel covariance — 2026-10-17

Added explicit COSMO E/B helicity spectra, spin-2 correlation transforms,
raw/centered Gaussian foreground-screen moments, all-pair covariance and
relation, and selected-time harmonic contraction. Loaded real readout maps
use the production instrument functions, frozen impedances and calibrated
zenith vectors. Real readout kernels and precontracted angular weights
avoid repeating a full harmonic transform for every frequency pair.

The exact parent/zoom power operator shares coordinate selection with the
existing integrator and retains folded images. Block projection includes
all frequency pairs. Optional process interpolation preserves PSD through
a congruence; independent nonstationary modes check its refinement against
direct fine-grid integration. Conditional Gaussian visibility draws report
their eigenvalue tolerance and keep the sky pattern shared across days.

Twenty-four new scientific tests cover independent Wigner sums, direct
Gaussian phase averages, finite real E/B harmonic ensembles, direct loaded
instrument assembly, spectral-mode channel projections and draw statistics.
The 31-test spin/channel/instrument focus and four compression tests pass;
all 24 deliberate faults are caught in disposable copies. The full
integration suite is running and its result will be recorded separately.

Full complex as-built kernels at native 10/30/50 MHz and lmax 96 are now
cached with response/dependency/source hashes and resolved receiver
impedances. `paper_kernels.py` and the staged `faraday_covariance.py` driver
implement the first foreground-screen predictions. Their final result
record and inspected figure follow in the next increment. Reproduction and
scope are in `docs/specs/2026-10-17-spin-channel-covariance.md`.

The spherical phase calculation currently covers fractional foreground
screens. Mixed-depth real-beam predictions, the full local-prior range,
matched small-beam/non-Gaussian controls, chromatic I leakage, loaded
receiver noise and the measured Q/I and U/I deliverable remain open. This does
not complete Phase 2 or establish sensitivity.

## Depth-profile moments and first implementation figure — 2026-10-17

Implemented screen fractions 0/0.6/1, a uniform slab and a truncated Gaussian
profile in `lusee_faraday.depth_depolarization`. The calculation retains both
complex moment types, the full joint Q/U covariance and the smooth/speckle
power split. Stable centered integrands, phase-dependent depth resolution,
order doubling and explicit nonconvergence errors prevent unresolved depth
ridges from being reported as a signal. No pixel area sets the amplitude.

The driver/checkpointed launcher regenerate the horizon scenarios, ten depth
benchmark cases at 10/30/50 MHz, source/dependency manifests and
`report/figures/depth_depolarization.pdf` with its inspected PNG and caption.
Independent depth and separation refinements shift the saved normalized
moments/power fractions by at most 5.8e-15 for this configuration. The
configuration remains an illustrative flat-plane PR 7 benchmark; local
screen statistics, real-beam spin-2 projection and the measured-spectra
deliverable are still open. Scope and commands are in
`docs/specs/2026-10-17-depth-depolarization.md`.

Verification: 18 new tests pass, and all 15 deliberate implementation faults
are caught by their corresponding tests in disposable copies. Final full
suite: 169 passed with the same two-port measured-reference failure found in
the baseline (151 passed, one failure). Black and source/new-driver flake8
checks pass. Evidence and hashes are in
`docs/depth-depolarization-validation.json`. The uncommitted notebook edits
are unchanged; upstream checkouts and the external paper were not edited.

## Conditional Faraday horizon calculator — 2026-10-17

Started implementation from `paper-implementation-plan` on
`implementation/depth-depolarization`. Added constant-rate horizon arithmetic
with the rotation coefficient derived from SI constants, separate one/two
radian phase conventions, explicit field projections and the unbounded
zero-field limit. The finite emitting-volume brightness is explicitly
unrotated and uses an illustrative emissivity. Sourced plasma inputs and
assumed magnetic geometry are distinguished in
`docs/literature/local-faraday-horizon.md` and `configs/paper/horizon.json`.
The regenerable `docs/faraday-horizon-results.json` contains all 54
scenario/frequency combinations and source/dependency provenance. No local
angular structure function or screen dispersion is inferred from this table.

Four horizon tests pass within the 18-test focused implementation suite.
Deliberate unit, phase-factor, zero-field and input-validation faults are
caught in disposable copies by the accompanying fault audit. The existing
suite's baseline has 151 passes and the known two-port measured-reference
failure. Remaining local-emissivity/pulsar priors and the full five-phase
paper implementation remain open.

## LuSEE Night observable predictions added to the plan — 2026-10-17

Christian identified future LuSEE-Night mission data as the observational
comparison target and requested an explicit measured spectra deliverable.
Updated `docs/plans/2026-10-17-paper-implementation.md` to specify simulated
parent/zoom Q/I and U/I spectra at 30/50 MHz, leakage-plus-noise controls,
the common smooth subtraction, residuals and conditional prediction bands.
The covariance companion covers channel correlations, parent/zoom variance
and repeatability at the same lunar sidereal phase. Predictions and future
data use the same calibration, channel responses and estimators; repeated
observations are not independent sky realizations. Added the six-figure
inventory and machine-readable statistical outputs. No implementation code
or scientific computation was added.

## Detailed paper implementation plan — 2026-10-17

Read Fable's two latest commits (`6f9bdf2`, `3ec0f12`) and mapped their
five phases to existing production interfaces and proposed code in
`docs/plans/2026-10-17-paper-implementation.md`. The plan specifies depth
moments, spin-2 all-pair covariance, exact channel-weight propagation,
leakage residuals, loaded receiver/radiometer noise, result provenance and
phase gates. It flags the contradictory `f=0` pilot check: with the stated
`f Phi` convention, `f=0` is unrotated and `f=1` recovers the screen pilot.
It also scopes the exponential cross-term check to its exact screen limit,
retains broad frequency covariance for near-edge slab emission, and requires
the same smooth-fit operator on signal, leakage and noise.

Prepared on `paper-implementation-plan` for Christian's requested review
before coding. No implementation code or scientific runs were added; the
pre-existing edits to both Gaussian-screen notebooks were left untouched.

## Analytic two-port normalization reference — 2026-10-17

Reproduced the historical short-dipole reference failure on main. Croissant
commit `45a94b2` changed the default horizon from an inclusive Boolean mask
to fractional visibility, giving the equatorial row half weight. The old
two-degree fixture's stored monopole, 1.1971034513643168, includes that row
fully. The upper-hemisphere integral of a unit short dipole's intensity,
`I_xx = 1 - sin(theta)^2 cos(phi)^2`, is `4*pi/3`; its monopole is therefore
`(4*pi/3)/sqrt(4*pi) = 1.1816359006036774`.

`two_port_pair_alms` now supplies the hemisphere weights explicitly, with
half weight at 90 degrees, so its normalization is independent of that
upstream default. The replacement test derives the expected monopole from
the integral and checks both autos on one-degree and two-degree grids.
The four-port production path is unchanged. Independent solid-angle
quadrature and checks against current croissant (`cc867f1`) and the CI
revision (`1c4d6c5`) are recorded under
`generated_data/historical-reference/`; neither upstream checkout was
modified.

Validation: all seven two-port tests pass with each croissant revision.
Injecting full equatorial weight makes both analytic-reference cases fail;
halving the pair-Stokes normalization also makes both fail. Black and
flake8 pass.

## Paper framing adopted and plan to the paper — 2026-10-17

Reviewed and fast-forward merged `paper-framing-redistribution` into main:
the "how Faraday rotation manifests" framing, the power-redistribution
pilot (`scripts/faraday_redistribution.py`) and its result record. The
pilot's numbers were checked: surviving-power ratios follow (0.6)^(8/alpha)
to 2% for all three alphas, the channel table recomputes, and the floor
matches a hand estimate to a factor 1.5. The review found that the pilot's
column screen is not the screen LuSEE looks through: depth depolarization
confines polarized emission below 50 MHz to a horizon of order 0.3–7 pc,
so the relevant phi statistics are local and unknown. It also found that
the speckle's frequency coherence (11 kHz at 30 MHz for sigma_phi = 10
rad/m²) is narrower than a parent channel, so parent spectra would show a
fixed-pattern variance excess rather than a spectral feature. The next
steps were reordered accordingly: horizon and depth depolarization first,
then the spin-2 real-instrument redistribution and channel covariance,
then separability against leakage and noise, then interpretation and the
manuscript. The phased plan with gates is in
`docs/plans/2026-10-17-paper-framing.md`; TODO.md and the paper-preparation
handoff now point to it and mark the detection-forecast outline superseded.

## Inference figure review and next-plan reconciliation — 2026-10-17

Completed the figure walkthrough. Updated TODO, the inference plan and the
paper-preparation handoff to link the reviewed evidence to the next decisions:
define the observational target; assess a fixed Hutschenreuter baseline plus
an uncertain residual; establish source/intensity constraints and realistic
response/channel treatment; test the toy spectral minimum in realistic
individual and ensemble spectra. The exact screen distribution still matters,
and covariance is not a substitute for converged direct-screen quadrature.
Calibration/source-boundary and matched-covariance non-Gaussian limitations
remain open before stronger inference claims. The benchmark can be retained
as a completed conditional result with those failures explicitly recorded.

## Direct-screen test of separated power predictions — 2026-10-17

Added the cached baseline 128-screen covariance-validation ensemble at
45–50 MHz to Figure 4, alongside the existing 20–35 MHz strong-rotation
ensemble. The builder checks that physical source/beam/screen parameters
match, uses each ensemble's exact frequency coordinates and analytic raw
second moments, and retains its own declared simultaneous moment bounds.
The high-frequency power residuals against the full prediction range from
-0.92 to -0.73 Monte Carlo standard errors; residuals against the local
approximation range from +12.48 to +22.15. The caption includes every
high-frequency residual against both predictions and notes their correlation.
This supports the full toy prediction in the regime where the curves separate;
it is not a new model-selection significance or a Galactic validation.

No new simulation or fitting was run. Rebuilt and executed the notebook from
completed caches under the 16 GB cap, inspected both power/fraction panels
and the residual panel, and passed Black/flake8 and manifest checks. Curves
below 20 MHz still lack direct-screen validation.

## Absolute polarization power and surviving fraction — 2026-10-17

Expanded the inference notebook's original Figure 4 with a deterministic
Faraday-off power reference and a separate surviving-power-fraction panel,
D=<|P_on|²>/|P_off|². Both use the same toy source, beam and frequency grid;
the off reference comes from `qu_moments` with zero screen mean/dispersion.
Direct-screen powers and their Monte Carlo errors are normalized by that
same reference. The original full-model residual check is retained. The
notebook explains the sampled minimum near 35 MHz as competition between
intrinsic synchrotron brightening and fractional Faraday depolarization.

Recorded testing the minimum in realistic spectra as a candidate follow-up
in the Galactic residual-screen plan, including individual realizations,
beam/channel response, leakage and noise bias. No realistic-spectrum result
is claimed. Rebuilt and executed the notebook from existing caches under
the 16 GB cap, checked the fraction lies in [0,1], and inspected the expanded
figure. Black/flake8 and figure-manifest checks pass.

## Candidate residual screen and PolSynch comparison — 2026-10-17

Recorded the fixed Hutschenreuter baseline plus uncertain delta-phi ensemble
as a possibility in `docs/plans/2026-10-16-galactic-residual-screen-models.md`.
It remains conditional on specifying resolved scales, residual statistics and
separate reconstruction/sub-resolution uncertainty. Read-only inspection of
`../PolSynch` at `749f27f` finds a different generative model: random complex
polarized emission in Faraday-depth bins, followed by frequency synthesis.
It supplies neither the proposed residual-screen statistics nor a LuSEE-band
validation. A one-bin calculation confirms that local `invRM.py:P_l2` uses
bin indices instead of physical depths in its phase; recorded the evidence
and the existing tests' blind spot in the plan. No checkout edits or sky
generation were performed.

## Gaussian-screen inference mode comparison — 2026-10-17

Added Figure 2b after the information assessment in the inference notebook,
with definitions of centered screen signal, noise-whitened covariance modes,
and projection into the reference total-RMS units. All ten reference modes
compare the baseline ell=0.5 pc with the cached, source-profiled ell=0.25 and
1 pc alternatives. Mean/RMS bars and full covariance-difference matrices
expose changes in the first modes while retaining mode correlations. The
source amplitude/angle/beta fits and synthetic noise are unchanged; no new
screen ensemble or optimization was run. The original Figure 3 follows this
supplement.

Verification: the builder checks reference projected covariance against the
identity and full projected KL against each cached fit (2 KL = 1.951 and
0.705). Rebuilt and executed the notebook from completed caches under the
16 GB virtual-memory cap; inspected the new PNG, and Black/flake8 pass.
Updated the tracked figure manifest. The original calibration limitation and
conditional toy scope still apply.

## Gaussian-screen bounded benchmark completed — 2026-10-16

Delivered `notebooks/gaussian_screen_inference.ipynb`, executed from
`scripts/gaussian_screen_inference.py`, eight inspected figures, and
`docs/gaussian-screen-inference-results.json` with configuration, hashes,
all calibration outcomes, numerical diagnostics and nuisance-boundary counts.
The launcher reproduces the capped/checkpointed run; the builder reads
completed caches and refuses stale source/configuration fingerprints.

All 128-screen covariance gates pass at ell=0.05/0.15/0.5/1.5/5 pc and five
frequencies over 45–50 MHz; maximum grid shift is 2.59e-5 of unrotated
polarization. The separate 20/25/30/35 MHz gate passes at ell=0.5 pc. The
five-scale validation took 662 s with peak RSS 878 MiB; the stronger-rotation
ensemble took 42 s. Independent 1024-screen final ensembles at ell=0.5 and
1.5 pc also pass the moment and numerical checks. The stationary depth-power
approximation has a whitened omitted-mean error of 3.43 over 45–50 MHz and
smaller errors in the narrow strong-rotation windows. Physical coherence
scales supply its amplitude; no pixel-area normalization or Galactic forecast
is inferred. Curves below 20 MHz have no direct-screen validation here.

The information gate rejects factor-of-two scale recovery with source
amplitude/angle/beta free. The frozen detection demonstration uses separate
4096-draw calibration and validation ensembles at each of three constant
depths, plus independent physical-screen alternatives. **17/18 null acceptance
checks pass.** The fitted-beta/full-covariance Hc test at c=-0.04 rad/m² gives
224/4096 false positives (5.47%); its simultaneous upper bound is 6.61%, above
the frozen 6.5% limit. This remains unresolved; no threshold rule, tolerance
or trial count was changed. Detection fractions for that test are diagnostics,
not validated significance claims. Baseline Hc fits hit beta bounds in 89.6%
of trials, so interpretation is conditional on the source restrictions.

The matched-RM-covariance mixture fails Gaussian readout moments (worst 30.46
MC standard errors) and passes its conditional moments (1.89 versus 4.16).
Its measured RM covariance agrees within 1.15 sampling standard errors at the
four tested lags. Thus matching RM two-point statistics does not determine
observable phase moments. All 30 detection-statistic refinement checks pass,
with zero changed decisions. Noise is synthetic; no scale confidence sets,
channel-averaged observations, Galactic validity or LuSEE sensitivity claims
are made.

Verification: focused suite 29 passed; full suite 147 passed with the one
unchanged two-port reference failure reproduced from pre-task `3bf50ce`.
Source and new drivers pass flake8 and Black checks. Updated plan, TODO and
handoff retain the unresolved calibration and source-model follow-ups.

## Gaussian-screen reproducibility and regression checks — 2026-10-16

Pinned direct-screen seeds and the two mixture amplitudes in the frozen design;
calibration now refuses a matching-shaped ensemble with different physical
parameters, frequencies, seeds or mixture settings. Independent final moment
checks are retained alongside likelihood diagnostics. Added explicit tests
that frequency ablation preserves same-frequency Q/U covariance and that the
covariance-weighted discrepancy matches the independent inverse-matrix form;
a diagonal-only mutation is rejected. Focused suite: 29 passed (4.41 s).

The full suite completed with 147 passed and one failure in the unchanged
`test_short_dipole_monopole_matches_measured_reference`: 1.1816359006 versus
its pinned 1.1971034514 reference. An isolated archive of pre-task commit
`3bf50ce` reproduces that same failure in the current environment (croissant
5.2.1, lusee 2.1.dev0). The test/reference and upstream packages are unchanged.
Logs are under `generated_data/gaussian_screen_inference/tests-*.log`.
The source and validation/calibration drivers pass flake8.

## Gaussian-screen numerical refinement and frozen tests — 2026-10-16

The first 128-screen, five-scale sweep passed every ensemble moment check,
but its coarsest angular grids at ell=0.5 and 1.5 pc shifted by 1.09e-4 and
1.83e-4 of unrotated polarization, exceeding the unchanged 1e-4 criterion.
Added explicit per-scale angular-grid multipliers and doubled those grids.
The original ledger is retained as `covariance-validation-initial.json`; the
full source-fingerprinted validation is rerunning before inference can start.
The scale-mixture spatial diagnostic now includes its per-screen A² factor.

Bound the null fitter to the frozen beta/depth ranges, recorded physical
amplitude and other nuisance boundary hits, and added final-statistic checks
on two angular grids with identical noise. Freezing rejects stale validation
and information-assessment products or missing scale/frequency coverage.
`scripts/run_gaussian_screen_inference.sh` provides a capped, checkpointed
end-to-end run and a read-only status command.

The focused suite passes 27 tests (6.39 s), including independent depth and
spatial-integral checks of the stationary template and explicit rejected
mutations of template normalization and nuisance bounds. Source and driver
lint passes. An isolated 16-sample orchestration smoke test exercises all 18
null comparisons and 30 alternative comparisons; its statistical outputs are
not scientific validation. The full project suite and production run remain
in progress.

## Gaussian-screen implementation: validation and inference drivers — 2026-10-16

Added resumable three-grid direct-screen validation with nested Fourier-cutoff
checks, generator covariance diagnostics, all-cell moment bootstrap errors and
source/configuration fingerprints. Each realization is checkpointed. The
resource pilot exposed inadequate source-profile sampling at large ell; the
production prescription now resolves the source width as well as screen modes
and phase variation. The five-scale 45–50 MHz covariance ensemble is running.

The independent 128-screen baseline ensemble at 20/25/30/35 MHz passed numerical
and moment gates: maximum angular shift 7.43e-6 of unrotated polarization,
worst moment discrepancy 3.03 MC standard errors versus the simultaneous 4.06
threshold. This is a conditional toy-model result, not Galactic validation.

Added normalized Gaussian scoring, source nuisance transformations/profiling,
constant-depth null fits, calibrated-discrepancy tools, finite-model information
assessment, mixture moments and the physical stationary-template limit.
Independent controls and explicit normalization/depth-fit mutations bring the
focused suite to 24 passing tests (4.74 s); source lint passes. The analytic
assessment fails the declared factor-of-two recovery criterion at every tested
scale after nuisance profiling. Five versus seventeen frequencies gives
profiled KL 0.352 versus 0.343 for the baseline-to-double-scale comparison at
fixed total inverse-variance budget. Calibration and final observations remain
pending the full spatial moment gate; no recovery intervals are claimed.

## Gaussian-screen implementation: moments and pinned fields — 2026-10-16

Added joint Q/U moments, retaining conjugated and nonconjugated complex
covariance, with stable centered quadrature and a separately checked fast
special-function path. Added immutable pinned Fourier fields, independent
grid evaluation, nested cutoff truncation, persisted coefficients and exact
discrete spectral covariance. The original illustrative generator remains
available for its existing notebook.

Independent controls compare Q/U moments to direct Gaussian integration of
a spatially constant screen and grid evaluations to an explicit sinusoid.
Mutation checks reject omitted relation moments, erased centered covariance
and incorrect FFT normalization. The focused suite passed 20 tests in 4.21 s
under the 16 GB virtual-memory limit. This establishes algebra and generator
controls; the full spatial covariance gate remains pending. A capped resource
pilot is measuring the new three-grid direct-sum cost before ensemble sizing.

## Gaussian-screen plan: paper priorities and inference gates — 2026-10-16

Revised the plan after the paper-direction review. The sequence is full joint
covariance validation, information/nuisance assessment, strong-rotation and
depth-template comparison, then a small calibrated inference exercise. The
paper deliverable now includes physical amplitude normalization and template
approximation errors. Scale recovery is conditional on identifiable information;
a quantified degeneracy is an acceptable result.

Added a fitted constant-depth competing model, explicit source nuisance stages,
pinned continuous-screen refinement, validation across correlation scales,
noise-weighted numerical-error checks and calibration precision requirements.
Profile-likelihood confidence sets are the initial interval procedure. Retained
one matched-covariance non-Gaussian stress test and moved map diagnostics,
intermittent screens and latent-field extensions into a linked deferred plan.
Updated TODO and the current handoff to match. Documentation-only verification
covers the diff, local links and unchanged pre-existing notebook edit. No
simulation, new validation ensemble or inference job was run.

## Plan review: exploiting the RM reconstruction model — 2026-10-16

Recorded a candidate amplitude-times-Gaussian-field extension in the inference
plan. Fixed amplitude preserves analytic conditional Gaussian phase averaging;
uncertain lognormal amplitude adds low-dimensional latent averaging for phase
moments under specified latent statistics. Angular integration, posterior
conditioning, unknown correlations and hyperparameter uncertainty remain
separate requirements. No new calculation or inference result is claimed.

## Inference plan review: non-Gaussian screens — 2026-10-16

Checked tests and plan: no non-Gaussian screen robustness test or Gaussianity
assessment of the real RM map has been run. The primary Hutschenreuter et al.
(2022) model uses exp(rho)*chi with Gaussian latent fields, which generally
gives non-Gaussian depths. Added explicit matched-covariance non-Gaussian
injections, intermittent-screen stress tests, local map/surrogate diagnostics,
posterior/held-out-data caveats and observable phase-statistic comparisons to
the inference plan. Updated TODO. This is a documented review finding and
planned validation, not a claim that a map diagnostic or inference test passed.
No scientific computation was run.

## Gaussian-screen merge and inference plan — 2026-10-16

Fast-forwarded local main to PR 7's squash merge `3bf50ce`. The remote
feature branch was already deleted. Carried the three local review commits
(physical overview, math rendering, discussion record) onto the new
`gaussian-screen-inference` branch because they were absent from the squash
merge; verified its committed tree matched the old feature branch before
deleting that local branch. Preserved the uncommitted notebook kernel name.

Wrote the [inference plan](docs/plans/2026-10-16-gaussian-screen-inference.md):
full joint Q/U covariance validation first, then independent toy observations,
likelihood checks, calibrated on/off detection, correlation-scale recovery,
and nuisance/model sensitivity. Explicit ideal readouts replace generic Vpq
for presentation. Synthetic noise is separated from a LuSEE sensitivity
forecast. The plan pins convergence and calibration procedures, disjoint
samples, implementation increments, artifacts and resource constraints.
Updated work queue and handoff. Checked relative links, whitespace and git
state. No scientific computation or inference code was run.

## Gaussian-screen discussion and inference recommendation — 2026-10-16

Recorded the discussion in the [benchmark plan](docs/plans/2026-10-15-gaussian-screen-step1.md#discussion-record--2026-10-16)
and linked it from the handoff and work queue. The record covers the physical
ensemble, bounded continuum integral versus unresolved quadrature, smooth
moments versus wavy realizations, covariance visualizations, the 21-cm analogy,
real-beam generalization, fixed RM maps versus uncertain residuals, and the
connection to the 16 measured real products. It recommends validating a small
toy inference problem before realistic inference, including null controls,
independent screen recovery, likelihood checks and nuisance degeneracies.
This is a proposed sequence, not a completed inference result. Existing
covariance/strong-rotation gates remain. No scientific job or code change.
Checked documentation links and whitespace.

## Gaussian-screen notebook: math rendering — 2026-10-16

Replaced display-math delimiters with `$$` throughout the notebook and its
builder, and joined multiline inline expressions onto single lines for
Markdown renderer compatibility, and replaced legacy `\rm` declarations
with `\mathrm{...}`. Equations and numerical outputs are
unchanged. Checked notebook validity, builder/input agreement, and math
conversion through the notebook HTML exporter and Pandoc's MathML renderer.
No scientific computation reran; output provenance still describes the prior
execution.

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
