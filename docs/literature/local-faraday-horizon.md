# Local Faraday horizon input audit

The implementation uses a uniform signed rotation rate and an explicit
complex-polarization phase threshold. It does not infer a unique emitting
distance from a density or a field magnitude. The numerical inputs are in
[`horizon.json`](../../configs/paper/horizon.json); regenerate the scenario
table with `scripts/faraday_horizon.py`.

Slavin and Frisch (2008), Section 5.1, constrain the circumheliospheric
electron density through photoionization models. We use their paired model
26 inputs/results: `B0=2.5 microgauss` in Table 2 and
`n_e=0.0653 cm^-3` in Table 5. `B0` is the model's magnetic boundary
parameter. Extending it as a uniform field along an emitting path is an
additional assumption, not a measured line-of-sight field.
([Author manuscript](https://astro.uchicago.edu/~frisch/2008SlavinFrisch.pdf),
PDF pp. 6, 8; DOI 10.1051/0004-6361:20078101.)

Yeung et al. (2024), Section 4.2, adopt
`n_e=(4 +/- 0.5)e-3 cm^-3` for converting X-ray emission measures into
Local Hot Bubble path lengths under constant-density assumptions. We use
the endpoints as sensitivity scenarios. Their X-ray analysis does not
measure the magnetic field; the calculation's `1 microgauss` field is
illustrative. These scenarios are not a joint density/field posterior.
([Full paper](https://arxiv.org/html/2410.23345v1#S4.SS2),
A&A 690, A399.)

For every scenario the absolute line-of-sight projection is explicitly
varied over 0, 0.25 and 1. Projection zero gives an unbounded Faraday
horizon, recorded as `null` with `horizon_unbounded=true`. The optional
brightness normalization is restricted to an assumed finite emitting path;
this cap does not make the Faraday horizon finite.

One-radian and two-radian phase thresholds are both generated. The latter
matches the framing's order-of-magnitude `Phi ~ 1/lambda^2` convention;
the former uses `|2 Phi lambda^2|=1`. Coherent field reversals, stochastic
accumulation and changes between local clouds are outside this arithmetic.

The adopted polarized emissivity is a unit normalization, not an observation
of the nearest few parsecs. Its power-law frequency scaling is illustrative,
especially below 20 MHz where turnover/absorption has not been modeled.
Neither source supplies the small-scale angular structure function or
`sigma_phi` within the phase-limited volume. The separate depth benchmark
therefore uses explicitly illustrative PR 7 parameters; it must not be read
as a locally calibrated sky prediction. Nearby-pulsar and emissivity priors,
their sightline selection and a non-Gaussian local-medium bracket remain
open Phase 1 work. The column RM map is not substituted for these inputs.
