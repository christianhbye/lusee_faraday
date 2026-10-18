# Depth depolarization implementation and validation

This increment implements the flat-plane moment calculation in Phase 1b
and the conditional horizon arithmetic in Phase 1a of the
[paper plan](../plans/2026-10-17-paper-implementation.md). The observational
target remains future LuSEE-Night data. These products establish the depth
calculation; real-instrument measured spectra follow Phases 2–3.

## Model and normalization

At each direction the emitting depths share one Gaussian column field Phi.
For x=lambda² the complex polarization is

```text
P(n,x) = P0(n,x) A(n,x),
A(n,x) = integral_0^1 w(f) exp[2i f x Phi(n)] df,
integral_0^1 w(f) df = 1.
```

`DepthProfile` supports a delta screen at a specified fraction, a uniform
slab and a truncated Gaussian. A screen at zero is unrotated; a screen at
one is the existing foreground-screen model. Gaussian center and width are
the underlying normal parameters; truncation changes its actual moments.
The spatial kernel is the existing `ScreenModel` squared-exponential
covariance with physical separations in pc. A generic spherical covariance
and spin-2 intrinsic sky are later work.

`phase_moments` returns means and both raw and centered complex moments at
named separations. `depth_qu_moments` combines them with the existing PR 7
Gaussian source and beam, averaging over separations with normalized
weight exp(-u). Both conjugated covariance and the nonconjugated relation
are required for the real joint Q/U covariance; no circularity assumption
is made. The result uses Q frequencies followed by U frequencies, in K.
Second moments and powers are in K². With `diagonal_only=True`, no full
real covariance is returned, so missing frequency cross terms cannot be
mistaken for zero correlations.

Smooth power is the squared mean conditional on the fixed intrinsic sky.
Speckle power is the centered complex variance. Their sum equals the
on-power, and all plotted fractions use the same Faraday-off power.
This decomposition generalizes to an independent random intrinsic sky by
conditioning on that sky before averaging over its statistics.

## Numerical gates

The centered integrand uses exponential differences to preserve small
variances. Depth quadrature starts at a phase-dependent resolution and
doubles until every requested moment meets its absolute/relative tolerance.
Separation quadrature resolves the phase core separately from the screen
correlation scale and includes an exponentially bounded omitted tail.
Its reported integration error is a numerical estimate plus that tail,
not a proof of a global error bound.

The result driver independently increases depth resolution and tightens
separation tolerance/support. It saves changes in every power component and
in both complex covariance types. Failure to establish convergence raises
an exception. Very strongly rotating mixed-depth columns can exceed the
configured maximum order; they are refused rather than approximated by an
unresolved diagonal ridge. No pixel area normalizes the variance.

Independent checks include direct slab sinc responses, direct truncated
Gaussian emission integrals, quadrature over correlated Gaussian depths,
and joint Q/U moments for a spatially constant random screen. The full-screen
limit is also compared with `gaussian_screen.qu_moments`. Covariance
positivity, smooth/speckle power accounting and independent numerical
refinements are checked before saving products. The raw full-screen
nonconjugated moment and the centered relation are tested separately.

`scripts/validate_depth_faults.py` repeats one named implementation mistake
per new test function in disposable source copies and checks that each test
fails. It covers phase/sign/normalization errors, missing relation terms,
convergence and input-validation bypasses, and stale/corrupt cache reuse.
The working source is never modified. Its JSON audit identifies each fault,
test and source hash; the same memory limit applies to this validation run.

## Regeneration and cache behavior

Run `bash scripts/run_paper_phase1.sh --help` for launcher usage, then
`bash scripts/run_paper_phase1.sh`. It starts the calculation in the
background under the required 16 GB virtual-memory limit and prints an
absolute log path. Run individual stages with `.venv/bin/python`:

```bash
.venv/bin/python scripts/faraday_horizon.py --help
.venv/bin/python scripts/depth_depolarization.py --help
```

The committed JSON configurations define the scenarios; result records
are `docs/faraday-horizon-results.json` and
`docs/depth-depolarization-results.json`. Depth arrays checkpoint one
profile/dispersion combination at a time under
`generated_data/cache/depth_depolarization/<fingerprint>/`. Fingerprints
include all physical/numerical inputs, the relevant source hashes and the
resolved dependency paths/versions/revisions. A sidecar digest protects
each array; corrupted arrays and incompatible result provenance are refused.

The depth figure, PNG preview, caption and figure manifest are regenerable
under `report/figures/depth_depolarization.*`. Downward triangles indicate
values below its explicit logarithmic display floor, including underflow.
Data-comparison spectra, leakage controls and noise bands are not produced
by this flat-plane benchmark.

## Scientific scope still open

The [source audit](../literature/local-faraday-horizon.md) distinguishes
published plasma inputs from assumed field projections, emitting paths and
unit emissivity. The horizon calculator preserves infinite horizons at
zero line-of-sight field. Its brightness is unrotated emission inside an
explicit finite volume. It does not derive sigma_phi or a structure function.

The depth example deliberately uses the benchmark geometry rather than
substituting the Galactic column for the unknown local statistics. Its two
dispersion scenarios are illustrative; they are not priors inferred from
the horizon or a prediction of LuSEE sensitivity. Nearby-pulsar constraints,
local polarized emissivity, a broader non-Gaussian bracket, real-beam
spin-2 projection, channelization, receiver noise and the six paper figures
remain later increments of the parent plan.
