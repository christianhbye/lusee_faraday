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
