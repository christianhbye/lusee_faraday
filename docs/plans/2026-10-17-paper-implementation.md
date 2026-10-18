# Implementation plan for the Faraday paper

Status: proposed implementation, prepared for Christian's review before code.
Branch: `paper-implementation-plan`, based on `3ec0f12`. The governing
prescriptions are Fable's [adopted framing](2026-10-17-paper-framing.md)
and [feasibility assessment](2026-10-17-paper-plan-feasibility.md), introduced
in `6f9bdf2` and `3ec0f12`. This document specifies how to implement their
five phases; it does not mark any scientific gate passed.

The deliverable is a reproducible calculation of depth-surviving polarization,
its covariance through the real instrument, and its separation from leakage
and radiometer noise. Amplitudes remain conditional on local emission and
Faraday-depth statistics. A bound is a complete scientific outcome.

## Predictions for future LuSEE Night data

The observational target is LuSEE-Night's own science data once the mission
is observing. Develop predictions in the measured port-product and calibrated
Q/U coordinates, at the observed frequencies and lunar sidereal phases.
The present work produces conditional predictions and simulated observations;
comparison with mission data follows when those data are available.

The primary comparisons are residual Q/U variance in parent versus zoom
channels, frequency-channel covariance, and repeatability at the same lunar
sidereal phase across observing cycles. Angular spectra provide a further
comparison only after applying a specified map reconstruction response.
Predict distributions of these statistics for the stated local-medium range,
not the exact spectral fluctuations of the unknown sky realization.

Export means, joint covariances, coordinates and estimator settings alongside
the figures. The eventual comparison applies the same calibration, channel
responses, masks, averaging and smooth subtraction to predictions and data.
Mission channel/observing configurations are inputs that can be updated
without changing the physical model. Repeated cycles reduce independent
measurement noise but are not independent realizations of the sky.

## Decisions needed in the specification

These are proposed resolutions of ambiguities in the prescriptions, to be
made explicit before numerical implementation.

1. **Define the screen fraction without reversing its limits.** Use
   `P_f = P_0 exp(2i f Phi lambda^2)`, where `f` is accumulated rotating
   column. Then `f=0` is unrotated and `f=1` recovers the foreground-screen
   pilot. Both Fable documents instead say that `f=0` recovers the pilot;
   that check contradicts their `f Phi` model. Test both limits and retain
   the prescribed `f=0, 0.6, 1` comparisons.
2. **Define smooth power conditionally on the intrinsic sky.** If
   `A(n,x) = integral w(f) exp[2i f x Phi(n)] df`, with `x=lambda^2`, write
   `A = m + delta_A`, where `m=E_Phi[A]`. The smooth field is `m P_0` and
   the speckle is `delta_A P_0`. An isotropic intrinsic sky can have zero
   ensemble mean while the smooth field still has nonzero power. Do not
   identify smooth power with the squared mean over both ensembles.
3. **Keep the mixed-depth assumption visible.** The prescribed `f Phi`
   construction makes depths perfectly correlated along each sightline.
   It is a bracket, not a three-dimensional turbulent emitting medium.
   Specify the intrinsic polarization angle and its depth dependence;
   begin with one angle shared by all emitting depths.
4. **Apply foreground-screen asymptotic checks only to that limit.** For
   one screen at `f=1`, the co-located equal-frequency *raw* phase relation
   is `E[exp(4i x Phi)] = exp(4i x mu - 8 sigma_phi^2 x^2)`.
   This is the precise exponential check available from Fable's cross-term
   prescription. A centered relation, an angularly integrated cross term,
   or a slab containing `f` near zero need not follow that same expression.
   Use the full branch identity below in the general case.
5. **State the horizon convention and normalization.** The framing uses a
   depth spread of order `1/lambda^2`; the literature note also quotes a
   one-radian complex-polarization phase tolerance `1/(2 lambda^2)`.
   Report which is used. A horizon based on a constant signed rotation
   rate is a slab estimate; reversals and random accumulation require
   separate assumptions. State emissivity in K per pc at a reference
   frequency so changing the horizon changes the emitting normalization.

## Existing code to reuse

| Existing interface | Use in the new calculation |
| --- | --- |
| `gaussian_screen.centered_branch`, `qu_moments`, `real_covariance` | Stable flat-plane reference, both complex moment types, and real Q/U conversion |
| `screen_draws`, `screen_validation`, PR 7–8 records | Independent physical-screen validation in the documented frequency windows |
| `response.four_port_pair_alms` | All ten physical pair kernels and both polarized dual blocks |
| `engine`, croissant rotations | Existing harmonic normalization and lunar rotation conventions |
| `instrument.covariance`, `blackbody_normalization`, channel packing | Loading, units, Hermitian assembly and science-channel order |
| `polarimeter`, `scripts/zenith_weights.py` | Calibrated linear X/Y combinations and pseudo-Stokes |
| `channelization` weight functions and `integrate` | Measured parent/zoom power responses, FFT bin ordering and folded images |
| `scripts/faraday_redistribution.py` | Frozen scalar pilot used as a regression reference |
| `config`, `conventions`, `scripts/common.py` | Coordinates, grids, wavelength, signs, x64 and output destinations |

Read upstream implementations without changing either checkout. The current
`pyproject.toml` points croissant at `/home/christian/Documents/projects/croissant`,
whereas AGENTS names `croissant-main`. Record the resolved imported path and
revision in run manifests; do not infer them from dated notes.

The pilot cache contains powers summed over `m`, not the complex kernels
needed for cross-pair covariance. Build a new cache of pair alms rather than
trying to reconstruct their phases from that cache. Verify complete response
content hashes; the pilot's first-16-MB digest is only an identity hint.

## Phase 1 implementation

### Horizon and local input ranges

Create `docs/literature/local-faraday-horizon.md` with primary-source checks
of Local Interstellar Cloud and Local Bubble electron densities, field
strengths and orientation assumptions, nearby pulsar constraints, and
polarized emissivity. Distinguish measured intervals, adopted priors and
illustrative scenarios. Recheck the existing 150–190 MHz citations before
using their numbers. Record source sections and units beside each input.

Add `scripts/faraday_horizon.py` and a small JSON input under
`configs/paper/`. Compute the rotation-rate and horizon intervals at
10/30/50 MHz, keeping electron-density/field scenarios paired where their
assumptions require it. Show sensitivity to phase tolerance and field
orientation; do not turn a possible `B_parallel=0` into a finite universal
horizon. Propagate an explicitly specified local emissivity into polarized
brightness. Keep the Hutschenreuter column as a separate reference.

The output record must distinguish physical-volume assumptions from an
angular screen parameterization. A distance, a rotation rate, and a horizon
alone do not determine `sigma_phi`, the angular structure function or its
small-scale slope. Tabulate these as conditional inputs, with consistency
checks rather than pretending to infer them from the column map.

### Depth moments

Add `src/lusee_faraday/depth_depolarization.py` with immutable profile and
quadrature settings and these proposed entry points:

- `depth_nodes(profile, numerical)` for normalized depth weights;
- `phase_moments(x, y, separation, screen, profile, numerical)` for phase
  means and raw/centered branch second moments;
- `depth_qu_moments(freqs_mhz, model, profile, numerical)` for the existing
  flat-plane benchmark extended over emitting depth.

Profiles are a delta screen at the three specified fractions, a uniform
slab on `[0,1]`, and a Gaussian profile with explicit center, width and
support. For geometric fractions use a normalized truncated Gaussian on
`[0,1]`; keep an untruncated Gaussian in Faraday depth as a separately named
analytic control. Do not interchange their depolarization laws.

For signs `s,t` in `{+1,-1}`, define

```text
M_st(theta;x,y) = integral df df' w(f) w(f')
  exp{2i mu (s f x - t f' y)
      - 2 sigma_phi^2 [(f x)^2 + (f' y)^2]
      + 4 s t f f' x y K_phi(theta)}.

m_s(x) = integral df w(f)
  exp[2i s f x mu - 2 sigma_phi^2 f^2 x^2].

C_st = M_st - m_s(x) conj(m_t(y)).
```

Carry both the conjugated covariance and the nonconjugated relation. Evaluate
the centered integrand with stable exponential differences, following the
existing screen code, rather than subtracting nearly equal integrated
moments. Resolve the `f=0` boundary layer and the `f x = f' y` ridge together
with the angular coherence scale. Avoid a depth-by-depth-by-angle allocation.

Add `scripts/depth_depolarization.py` with explicit compute and plot stages.
Write `docs/depth-depolarization-results.json`, regenerable arrays under
`generated_data/`, caches under `generated_data/cache/`, and one figure under
`report/figures/`. Record smooth power, speckle power, their total, and RMS
amplitudes at all three frequencies relative to the same unrotated reference.
Use a consistent loaded intensity/leakage reference when available from
Phase 2; label interim pilot normalizations as such.

### Phase 1 acceptance

- `f=0`: unrotated result and exactly zero phase-induced speckle.
- `f=1`: existing screen moments for identical flat-plane parameters;
  scalar spherical correlation recovers the pilot for its own parameters.
- Zero rotation: every normalized profile returns the same off reference.
- Deterministic depth: uniform-slab response matches
  `exp(i Phi x) sinc(Phi x)` with `sinc(z)=sin(z)/z`; a separately defined
  Gaussian Faraday-depth profile matches its analytic Fourier transform.
- Raw second moment equals smooth plus centered terms; the joint real
  covariance is symmetric and positive semidefinite within reported
  integration error. No grid-cell area enters a physical normalization.
- Independently refine depth nodes, separation resolution and integration
  support until each nonzero reported component changes by less than 1%.
  Near a zero, report absolute error in off-power units instead of a ratio.
- Save the local input range and emissivity with every result. If a sourced
  scenario permits horizons beyond 100 pc at 30 MHz, report that outcome
  and revisit the local-screen premise before presenting it as general.

## Phase 2 implementation

### Spin correlations and instrument projection

Add `src/lusee_faraday/spin_covariance.py` for parallel-transported `xi_plus`
and `xi_minus`, multiplication by the two depth-phase moment kernels, and
spin-2 transforms with `d^ell_(2,2)` and `d^ell_(2,-2)`. Define intrinsic
EE/BB/EB spectra explicitly, with admissible joint spectra across frequency.
Do not promote the scalar pilot's exponential correlation to a spin-2
model without checking its spectral positivity. Preserve the intrinsic-sky
assumptions independently of the screen assumptions.

Use low-order Wigner-d evaluation or Jacobi recurrences for the two required
rows, validated against available rotation machinery. No new full-sky map
transform engine or high-resolution Faraday sky is needed. Precompute angular
weights through the response band limit and integrate coherence cores on
rescaled grids. Retain the smooth term on its broader angular scale.

Add `src/lusee_faraday/visibility_covariance.py` to contract spin spectra
with all pair alms, including every P_MINUS/P_PLUS and cross-pair term.
Rotate kernels using the existing frame conventions and compute time
correlations from their harmonic overlap. Apply receiver loading as the
same linear map on both covariance legs, then map to 16 real science
products and calibrated pseudo-Stokes. Keep sky covariance distinct from
the 4-by-4 electric-field covariance and from radiometer estimator noise.

Freeze beam and impedances at each native band center within its narrow
band. Preserve `T_moon=T_ant=0` in the Faraday-only calculation. Build
separate total-system powers for the noise calculation in Phase 3. Do not
call the frozen-beam Faraday run a test of sub-channel leakage smoothness.

### Frequency covariance and channelization

Extend `channelization.py` with a channel operator assembled from its existing
normalized weights. Make `integrate` and covariance projection share the
same selection and weighting logic. For an operator `W` mapping fine
frequencies to bins, propagate `mean -> W mean`, `C -> W C W^H`, and
`J -> W J W^T`. Retain folded zoom responses and their correlations.

Do not allocate a `16384 x 16384 x pair x pair x time x time` covariance.
Compute frequency pairs in blocks and accumulate the requested channel
moments directly. Use a coherence-band shortcut only after bounding its
omitted contribution for the selected profile. The smooth term, small
`sigma_phi`, and near-edge slab emission can couple the whole band; those
cases need a broad or structured representation, not a fixed sparse cutoff.
Refine frequency-pair interpolation/compression against direct integration
on small grids. Reuse Wigner-d weights across pairs.

Provide Gaussian draws of beam-weighted speckle using the full real
covariance, including the complex relation. Use channelized covariance
or harmonic time modes to avoid dense factorization over the entire
time-frequency-product cube. One draw is shared across channels and
repeated sidereal days. These are conditional Gaussian visibility
illustrations; weak rotation or too few independent cells may invalidate
Gaussianity even when the second moments remain correct.

Add `scripts/faraday_covariance.py` with `kernels`, `moments`, `validate`
and `plot` stages, cached separately. Generate the real-instrument
redistribution and frequency-covariance figures and a new result record.
Include an explicitly parameterized low-frequency spectral turnover;
keep broad-band panel (c) labeled illustrative.

### Phase 2 acceptance

- Spin transform round trips, analytic low-ell identities, reality and
  spectral positivity. Zero rotation recovers the specified intrinsic sky.
- Narrow angular correlations recover the scalar pilot's high-ell behavior
  and auto-pair white-floor normalization, with the same input power.
  Foreground-screen rotation conserves total pointwise polarized power;
  mixed-depth emission is not required to conserve that power.
- Test the exact foreground-screen phase relation above before taking a
  strong-rotation approximation. Deliberately remove a polarized cross
  block and demonstrate that an appropriate synthetic-pair test fails.
- Narrow Gaussian beams and the deterministic benchmark source recover
  `qu_moments` before comparing with the PR 7–8 direct-screen ensembles in
  45–50 and 20–35 MHz. An isotropic random intrinsic sky is not the same
  ensemble as that benchmark; use an explicit matched geometry/control.
- Run the PR 8 matched-RM-covariance scale mixture through cross-frequency
  moments. Compare with its conditional mixture prediction as well as the
  Gaussian prediction; retain any Gaussian-model failure in the record.
- Channelized moments match independent draws passed through `integrate`;
  reduced-grid dense integration checks block accumulation and omitted tails.
  Report maximum entry errors, PSD diagnostics, coherence thresholds, and
  convergence separately for every band/profile.

## Phase 3 implementation

### Leakage residual and signal attenuation

Add `src/lusee_faraday/separability.py` and
`scripts/faraday_separability.py`. Build I-only leakage with the harmonic
production arm for as-built, C4-symmetrized and diagonal-ZA responses at
30/50 MHz. Verify the existing leakage values in their original geometry
and calibration before using them as a denominator in the new geometry.

For chromatic leakage, evaluate available native-frequency responses and
document interpolation to each parent/zoom window. Inspect actual FITS
frequency coordinates; the feasibility note's approximate native spacing
must not become a hardcoded value. Compare stated interpolation choices.
Residuals then measure those interpolants, not measured fine-channel beam
structure. A constant frozen kernel would give a trivial smooth-fit result.

Specify the smooth basis, fit interval, frequency coordinates and weighting
before comparison. Use the same fit operator `R` for leakage, smooth
polarization, speckle covariance and noise: `C_res = R C R^T` for real
observables. Tabulate both residual leakage and retained speckle power;
a small residual is uninformative if the fit also removed the signal.
Report rank/conditioning and flag an exact zero denominator as undefined
or a conditional lower bound, never an automatic infinite detection ratio.

The three beam variants provide a model spread, not a calibrated beam-error
distribution. Report model leakage residuals and, separately, the mismatch
when one variant corrects another. There is no justified precision forecast
from fitting each assumed beam perfectly to itself.

### Receiver and radiometer noise

Add `src/lusee_faraday/receiver_noise.py` with an explicit loaded-plane
voltage-noise spectral covariance, units, source and parameter range.
`JFETReceiver.Z` supplies impedances, not amplifier noise. Legacy
`lusee.Throughput` supplies a scalar noise helper tied to external SPICE
files and a legacy beam; reuse it only after checking file availability,
gain removal, spectral-density convention and four-port applicability.
Otherwise accept sourced voltage/current noise spectra and propagate them
through the existing loading network. An explicit nonzero illustrative
input remains conditional; missing receiver data cannot become zero noise.

Assemble loaded sky, Moon, antenna-loss and amplifier contributions without
double counting. Derive the joint covariance of the measured products
from this total electric-field covariance. Check against luseepy's
`MapMaker.compute_radiometric_noise` for individual channel variances,
while retaining cross-product covariances it omits. Use effective bandwidths
and overlaps from the actual channel weights and state assumptions about
temporal/filter correlations; nominal bin spacing is not an automatic
noise-equivalent bandwidth.

Define the on-source integration per sidereal phase bin, nightside observing
duty cycle and time averaging for 1 and 24 lunations. Repeated observations
reuse the same sky/speckle/leakage realization; only receiver/radiometer noise
decreases with independent integration. Record effective bandwidth in Hz,
integration in seconds and the reference plane alongside every noise value.

### Decision table and acceptance

For the full Phase 1 range, both bands and both channel types, tabulate
residual speckle RMS over residual leakage and over radiometer RMS, plus
the smooth-fit transmission and beam-model spread. State whether the
reported amplitude is Q, U or total polarization, and keep power ratios
distinct from amplitude ratios.

Verify the radiometer limit with independent complex-voltage simulations
and the expected square-root integration scaling. Test smooth and rapidly
decorrelating screen controls to expose when zoom spectra cannot distinguish
speckle from smooth leakage. For the spatial discriminant, compare the
predicted low-ell polarized spectra and leakage spectra on common modes
and through a stated observing/map operator. A white input spectrum alone
is not evidence of separability in a reconstructed map. Re-read Camacho
et al. before claiming any map-level sensitivity.

If the ratio is below unity across the range, produce the bound. If some
scenarios pass, report which assumptions permit separation. Neither outcome
is a calibrated detection significance or a forecast.

### Measured spectra figure and covariance companion

Make the measured spectra figure a central Phase 3 deliverable from
`scripts/faraday_separability.py`, with the frequency-covariance companion
generated by `scripts/faraday_covariance.py` using the same configuration.
Before mission data are available, identify every sample as a simulated
observation. Use a fixed, recorded realization and preselected lunar sidereal
phase; include a second phase as a scope check rather than selecting the
largest fluctuation after inspecting draws. Show both 30 and 50 MHz.

The spectra figure contains these comparisons:

- Calibrated Q/I and U/I versus frequency in parent and zoom channels,
  showing leakage, depth-surviving smooth polarization and speckle
  contributions with a common intensity normalization.
- The total simulated observation with receiver/radiometer noise, the
  leakage-plus-noise control with only polarized sky emission removed,
  and the stated smooth fit. Share the sky, instrument configuration and
  noise draw between paired controls.
- Residuals after the same smooth subtraction, with the radiometer RMS and
  conditional model prediction bands. Distinguish the spread over speckle
  realizations from uncertainty in the ensemble mean, and show the range
  from local-medium scenarios separately.

Perform smooth subtraction on calibrated Q/U, then display the residuals
divided by the same channelized intensity reference used for the component
curves. Retain Q/U in kelvin in the exported products. Propagate intensity
uncertainty and its correlation with Q/U when using an estimated I; ratios
of noisy observables cannot be assigned the unmodified Q/U covariance.
Label frequency units, dimensionless fractions and intensity normalization,
integration time, lunar sidereal phase, and all model assumptions. Keep
comparable panels on common scales. Write the spectra as
`report/figures/faraday_measured_spectra.pdf`.

The companion shows residual Q/U covariance and normalized correlation
matrices on the actual parent and zoom frequency coordinates, before and
after smooth subtraction, plus correlation versus frequency separation at
named reference frequencies. Do not impose stationarity by averaging unlike
frequency pairs. Include the leakage-plus-noise control and the parent/zoom
residual variance comparison. Save it as
`report/figures/faraday_channel_covariance.pdf`. Export the Q/U cross
covariance as well as the individual Q and U blocks.

Show repeatability with two simulated observing cycles at the same sidereal
phase: retain identical sky, leakage and speckle, and draw independent noise.
Their difference checks the predicted measurement noise; their common
structure is a sky-plus-instrument signal and does not identify Faraday
rotation by itself. Compare the measured statistics with predicted bands
using a stated covariance estimator and its sampling uncertainty; do not
count repeated views of one sky as independent screen draws.

Once LuSEE-Night data are available, place measured spectra/statistics and
model predictions on the same axes, with data-minus-model residuals judged
against the measurement-noise prediction. The initial implementation must
exercise this path with independent simulated observations and record
parent/zoom variance ratios, frequency correlations and repeatability
statistics in the Phase 3 result record. A mismatch or an upper bound remains
a reportable outcome under the conditional local-medium assumptions.

## Phase 4 and paper products

Add lunar nightside plasma source checks to the literature note and compute
its Faraday-depth bound with `scripts/faraday_horizon.py`, carrying density,
path length and magnetic-field assumptions. State the depth-depolarization
and angular-coherence degeneracy next to any inversion of the measured floor.
Do not infer both from a single amplitude.

Add `scripts/paper_products.py` after the numerical interfaces settle. It
reads completed result records, verifies their manifests, and regenerates
tables and LaTeX number macros. The figure inventory expands Fable's Phase 5
with the explicit measured spectra deliverable:

| Figure | Predicted quantity or comparison |
| --- | --- |
| Instrument and leakage | Calibrated response and I-to-Q/U leakage using the existing instrument material |
| Depth depolarization | Smooth and speckle power versus frequency for the emitting-depth profiles |
| Power redistribution | Polarized angular spectra and the power passed by the actual beam |
| Measured spectra | Simulated Q/I and U/I in parent/zoom channels, controls, smooth fits and residuals; future mission data in the same coordinates |
| Channel covariance | Q/U covariance and frequency correlations, parent/zoom variance, and repeatability checks |
| Separability | Residual speckle versus leakage residual and radiometer noise across the conditional local-medium range |

The channel-resolution and Faraday-horizon tables remain separate table
products. Read `21cm-figures` before figure work and `latex-memo` before
LaTeX/report work; visually inspect generated figures and the built PDF.

Keep manuscript preparation in this repository's permitted report area.
Locate the collaboration template before a manuscript transfer; the external
paper is protected by project instructions. Paper-specific prose and template
integration follow the scientific gates. Retain the instrument and point-source
material, cite the numerical-audit tag, and omit the superseded detection
forecast from the required results.

## Provenance and execution contract

Each stage writes a machine-readable record with its model/configuration,
source hashes, resolved dependency paths/revisions, complete input-artifact
hashes, frequency/time coordinates, units/reference plane, numerical settings,
random seeds, diagnostics, scope and gate outcomes. Key caches on all inputs
that affect their stage. Refuse incompatible caches, checkpoint completed
blocks atomically, and never silently load an older key.

Provide `--help`, explicit stage selection and an inexpensive smoke
configuration. A runner under `scripts/` launches costly stages in the
background with absolute logs under `generated_data/`, the required
`ulimit -v 16000000` or `24000000`, and controlled thread counts. Import
`common` before JAX. Benchmark representative transforms before estimating
the complete covariance run; the feasibility note's hour estimate is not
a measured runtime for mixed-depth and all-pair code.

## Commit sequence and verification

| Increment | Reviewable change | Required evidence |
| --- | --- | --- |
| 0 | This implementation plan and PROGRESS entry | Source/commit comparison and preserved existing notebook changes |
| 1 | Cited horizon/local-input configuration and calculator | Units, phase convention, regenerated three-band table, uncertainty scope |
| 2 | Depth profiles, moments, driver and result record | Analytic limits, existing-screen comparison, independent quadrature refinements |
| 3 | Spin transforms and all-pair instrument projection | Identities, scalar limit, cross-term fault injection, matched flat-plane limit |
| 4 | Channel covariance, fixed-pattern draws and real-beam products | Independent channelized ensembles, PR 7–8 gates, mixture control, bounded numerical errors |
| 5 | Chromatic leakage residuals and signal-transfer calculation | Same estimator for all components, beam mismatch spread, interpolation limits |
| 6 | Loaded receiver noise and joint radiometer covariance | Sourced nonzero inputs, voltage Monte Carlo, bandwidth/time scaling |
| 7 | Separability, measured spectra and interpretation records | Full conditional range, both bands/channels, controls, prediction bands, attenuation and degeneracies |
| 8 | Paper tables, macros, six figure products and permitted report build | Manifest checks, figure/PDF inspection, citations and regenerated numbers |

Update `PROGRESS.md` at every completed increment and keep commits scoped.
Do not modify the pre-existing notebook edits. Preserve the original pilot
and completed benchmark as references rather than rewriting their results.

For each numerical increment, run focused tests, Black at line length 79
and `flake8 src/`; run the full suite at integration boundaries under the
required memory cap. Check the reported historical two-port failure against
the current base before attributing it to this work. Before committing each
new scientific test, deliberately inject the specific mistake it claims to
catch and record that it fails. Freeze acceptance settings before production
runs; retain failures instead of relaxing a gate after seeing the result.

Phases 1–3 remain sequential. Literature/interpretation and report structure
can progress while Phase 3 is evaluated, but paper claims wait for the
relevant records. This planning increment launches no scientific computation
and adds no implementation code.
