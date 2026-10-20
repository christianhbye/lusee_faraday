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
as a locally calibrated sky prediction. The column RM map is not substituted
for these inputs.

## Nearby pulsar constraint

Yan et al. (2011), Table 1, list PSR J0437-4715 with dispersion measure
2.64 pc cm^-3 and rotation measure -0.6 rad m^-2. These are historical
catalog values with limited quoted precision; the paper explicitly treats
ionospheric corrections and residual systematic effects. Reardon et al.'s
timing distance, quoted in Askew et al. (2025), is 156.96 +/- 0.11 pc.
The integrated DM/RM and distance constrain path averages, not the nearest
few parsecs, local angular variance, or a unique emitting horizon.
Field reversals and distinct structures can cancel in RM. Scattering
structures along this sightline likewise do not measure their RM variance.
We therefore retain conditional local-screen cases rather than deriving a
dispersion from this single sightline.
([Yan et al., Table 1](https://arxiv.org/pdf/1105.4213),
[Askew et al., distance and screens](https://arxiv.org/html/2410.21390v2).)

## Emissivity normalization

Polderman et al. (2019), Section 6.2, measure a mean total-intensity
foreground emissivity of 1.52 +/- 0.99 K pc^-1 at 74 MHz for their
HII-region sample with foreground distances below 1 kpc. This constrains
long path averages of total intensity, with distance and model limitations.
It does not measure polarized emissivity within the nearest few parsecs.
The supplementary calculation exports the endpoints scaled by the stated
illustrative -2.7 temperature index and assumed polarization fractions.
These are sensitivity brackets, with no absorption model or measured local
polarization fraction. The production intrinsic spin-2 RMS remains a 1 K
normalization at 30 MHz; every signal RMS scales linearly with it, and every
signal covariance quadratically. The horizon path/emissivity normalization
and isotropic angular RMS are distinct model specifications.
([Primary paper, Section 6.2](https://arxiv.org/html/1811.10310v1).)

## Lunar plasma

Ando et al. (2012), Sections 4 and 6, report 19 high-latitude occultations
covering solar zenith angles 70.3--123.5 degrees. They find no stable
enhancement; the mean near-surface column is below 0.01 TECU, with noise
near 0.011 TECU. Transient, localized and differently illuminated regions
are not excluded. This is not a universal nightside density upper limit.
Our supplementary arithmetic therefore labels its density, path and
magnetic-field envelope as assumed, including a deliberately large
1000 cm^-3 historical-ionosphere scale. It computes a conditional Faraday
bound with aligned field and no cancellations; it makes no site-specific
measurement claim.
([Primary occultation study](https://doi.org/10.1029/2011JA017141).)

The measured floor alone cannot separate emitting-depth distribution,
intrinsic polarized emissivity, angular coherence and Faraday variance.
The full-channel equal-weight 0.5/1.5 variance mixture explicitly brackets
non-Gaussian phase statistics at identical RM covariance. None of the local
sources above fixes that higher-order distribution.
