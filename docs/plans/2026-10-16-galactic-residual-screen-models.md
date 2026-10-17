# Galactic residual-screen models: deferred research plan

Status: deferred, 2026-10-16. This preserves the Galactic-model work separated
from the [Gaussian-screen benchmark](2026-10-16-gaussian-screen-inference.md).
No map assessment, new ensemble or physical covariance selection has run.
These checks are required before extending a Gaussian-screen inference claim
to a Galactic model; they do not block the conditional toy benchmark.

## Physical motivation and available evidence

Hutschenreuter et al. (2022), Section 2.3, Eq. (7), model the Galactic depth
as exp(rho) times chi, where rho and chi are Gaussian fields. This nonlinear
combination is generally non-Gaussian; Gaussian latent variables do not make
the RM field Gaussian. The map is also a posterior reconstruction, so its
statistics reflect the reconstruction model and spatially varying constraints.
See the [primary paper](https://doi.org/10.1051/0004-6361/202140486).
This observation does not establish whether a specified local residual can
be approximated as Gaussian at the scales relevant to our observable.

## Follow-up checks

First use the benchmark's matched-covariance scale-mixture test to identify
which observable statistics are sensitive to non-Gaussian depths. Then:

1. Add a spatially intermittent/lognormal-modulated family, motivated by the
   reconstruction's latent-field form, with its normalization and correlation
   changes measured explicitly. This is a stress test, not a validated model
   of the map or a matched-covariance control unless matching is demonstrated.
2. Assess the existing map conditionally on declared sky regions, scale range,
   large-scale mean and variance model. Compare marginal residuals, separated
   pixel-pair distributions, increment distributions and selected higher-order
   statistics against Gaussian surrogates with the same covariance, mask and
   effective resolution. Use independent regions for assessment. Calibrate
   discrepancies with correlated simulations rather than treating map pixels
   as independent samples. A pooled all-sky histogram can mix different means
   and variances; it neither establishes nor isolates non-Gaussianity of a
   residual random field.
3. Where possible, assess posterior realizations and held-out RM observations,
   not only the posterior-mean map. A mean map is not a screen realization,
   and agreement with a reconstruction's imposed prior is not independent
   evidence for the sky. State the available posterior information and the
   unresolved scales left unconstrained.
4. Compare the empirical single/pair phase averages relevant to the intended
   frequency window with their Gaussian predictions, alongside map diagnostics.
   For non-Gaussian depths the necessary characteristic functions depend on
   more than mean and covariance. Prioritize the error in the final readout
   statistics, rather than treating an omnibus Gaussianity test as a direct
   prediction of measurement bias. Finite resolved-map comparisons do not
   constrain missing sub-resolution structure.

## Candidate extension using the reconstruction's latent fields

Discussion, 2026-10-16: exploit phi(n)=A(n)*chi(n), A=exp(rho), as a
structured alternative to assuming the entire RM sky is a homogeneous
Gaussian field. This is a candidate derivation, not a validated replacement
for the toy inference plan or a claim to reproduce the reconstruction posterior.

First consider a prescribed positive amplitude map A and a conditional Gaussian
model for chi. Then phi conditional on A is Gaussian with mean A*mu_chi and
covariance A(n)*A(n')*K_chi(n,n'). The same characteristic-function phase
average remains exact for that specified model. Spatially varying amplitude
permits spatially varying RM variance; isotropic separation reduction is no
longer automatic. Constant A is the Gaussian benchmark limit.

Next consider uncertain A=exp(rho). Under a specified jointly Gaussian latent
model with fixed hyperparameters, condition on rho and analytically integrate
chi using its conditional mean/covariance. A one-direction phase moment then
requires averaging over one rho value; a two-direction phase moment requires
over their joint pair distribution. The remaining latent integral has one or
two dimensions per angular point/pair, rather than requiring a complete screen
draw for each phase moment. This does not eliminate the angular-pair integral,
prove affordable real-beam evaluation, or supply unknown latent covariances.
When latent fields are correlated, use conditional statistics, not an
unjustified independence assumption. Hyperparameter uncertainty adds further
averaging. Data conditioning can produce a non-Gaussian posterior, so Gaussian
prior fields do not guarantee this closure for posterior predictions.

The released component means/standard deviations are useful exploratory
inputs, but do not determine joint latent statistics or their posterior.
Products of component means need not reproduce the mean RM map. Inventory
available component/posterior information before adopting this model, and
keep assumptions about unresolved fluctuations explicit. Validate fixed-A and
random-A limits with independent converged direct-screen sums before inference.
This extension may provide a physically motivated non-Gaussian stress test
between the Gaussian benchmark and the full LuSEE/WMAP model.

## Entry decision and deliverable

Begin with an inventory of map components, posterior information, masks and
resolved scales. Select a local residual definition and observational question
before choosing a screen family. Keep the fixed RM map as a separate baseline;
posterior uncertainty and physical unresolved fluctuations need distinct
models, and represented power must not be added twice.

Deliver a scoped model specification, independent direct-sum validation and
observable-level sensitivity to unresolved structure. Identify unavailable
joint statistics explicitly. The intermittent family and latent closure remain
candidates until those checks succeed. Use the benchmark's provenance,
resource limits and artifact conventions, and update TODO and PROGRESS after
each completed increment. No observing forecast follows from map diagnostics
alone.
