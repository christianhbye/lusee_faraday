# Gaussian-screen benchmark: step-1 specification

Status: model and observable specified for Christian's review. This is a
conditional flat-sky benchmark, not a Galactic model or a validated forecast.
The executed companion is `notebooks/gaussian_screen_step1.ipynb`.
The implementation sequence is in
`docs/plans/2026-10-15-gaussian-screen-step1.md`.

## Scope and physical ensemble

The organizing observable is the same frozen-beam, fixed-source complex
visibility with foreground Faraday rotation switched off and on. At one fixed
pointing, the ensemble varies only a physical foreground screen. A single
screen realization is shared by all frequencies, both polarization branches,
and the paired off/on calculation. It does not change between observations.
There is no instrument noise or ensemble of intrinsic sources in this step.

Use an infinite Euclidean tangent plane centered on a zenith pointing, with
angular coordinates theta = (theta_E, theta_N) in radians and a fixed
orthonormal East/North polarization basis. The plane approximation is part
of the toy model; sphere curvature, basis transport, horizon and lunar motion
are excluded. Source and beam widths are small (1.5 and 1 degree standard
deviations). This is not an all-sky isotropic spin-2 field.

A geometrically thin screen at distance d = 100 pc has transverse coordinate
r = d theta. Its real Faraday depth phi(r) is a stationary Gaussian random
field with mean m = 0.01 rad m^-2 and covariance

    K_phi(h) = sigma_phi^2 exp[-h^2/(2 ell^2)],
    sigma_phi = 0.02 rad m^-2, ell = 0.5 pc.

Here ell is the physical separation where K_phi/K_phi(0) = exp(-1/2).
The angular correlation length is ell/d. The two-dimensional power spectrum
under K(h) = integral d^2k/(2pi)^2 S(k) exp(i k.h) is
S(k) = 2pi sigma_phi^2 ell^2 exp(-ell^2 k^2/2), which is nonnegative and
integrates to sigma_phi^2. Thus this is an admissible covariance on the
plane. The structure function is D_phi(h) = 2 sigma_phi^2[1-rho(h)], with
rho(h) = exp[-h^2/(2 ell^2)]. This smooth covariance is a controlled choice,
not evidence for a Galactic turbulence law. The ell -> infinity limit is a
single random constant depth over the whole plane; sigma_phi = 0 gives a
deterministic constant depth m. Zero rotation requires both m = sigma_phi = 0.

The deterministic source is

    I_0(theta,nu) = I_ref f(nu) exp[-theta^2/(2 s_src^2)],
    P_0(theta,nu) = Q+iU = P_ref f(nu) exp(2i chi)
                              exp[-theta^2/(2 s_src^2)].

I_ref = 2 K, P_ref = 1 K, chi = 20 degrees, s_src = 1.5 degrees,
f(nu) = (nu/30 MHz)^beta, beta = config.BETA_QU. Stokes V is zero.
These are Rayleigh-Jeans brightness temperatures. I >= |P| everywhere.
The normalized power beam B(theta) = exp[-theta^2/(2 s_beam^2)] /
(2pi s_beam^2), with s_beam = 1 degree, has inverse-solid-angle units in
the plane approximation. It is achromatic throughout the illustrative
10--50 MHz sweep. Frequency samples are monochromatic; there are no channels
or bandwidth averages. This broad sweep is a controlled frozen toy beam,
not the real instrument's narrow-band approximation.

Use the repository COSMO sign P_on = P_0 exp(+2i phi lambda^2), with
lambda_squared and phase helpers imported from conventions. No IAU transform
is performed. A dimensionless ideal Jones pair p=(1,0),
q=(1/sqrt(2), exp(i pi/3)/sqrt(2)) defines a realizable complex cross visibility:

    V = integral B [k_I I + k_Q Q + k_U U] d^2theta,
    k_I = k_Q = 1/(2 sqrt(2)), k_U = exp(-i pi/3)/(2 sqrt(2)).

This follows from <E E^dagger> = [[I+Q,U],[U,I-Q]]/2 for V_Stokes=0.
Both ports have the same spatial beam. Visibility is in beam-averaged K in
this ideal coherency convention, not loaded LuSEE volts squared. The source
coherency is positive semidefinite. Thermal terms vanish; the toy loading is
unity and frozen. The unchanged intensity contribution is retained in V,
even though it cancels in V_on - V_off.

## Means and second moments

Let x_i = lambda_i^2, s,t in {+1,-1},
eta = s_src^2/(s_src^2+s_beam^2), and
w = d (s_src^-2+s_beam^-2)^(-1/2). After combining source and beam,
W(r) = exp[-r^2/(2w^2)]/(2pi w^2) integrates to one. Define

    Z_s(i) = integral W(r) exp[2i s x_i phi(r)] d^2r,
    b_s(i) = eta P_ref f_i exp(2i s chi) (k_Q - i s k_U)/2,
    v_I(i) = eta I_ref f_i k_I.

Then V_on(i) = v_I(i) + sum_s b_s(i) Z_s(i), while
V_off(i) = v_I(i) + sum_s b_s(i) is deterministic. In general b_- is
not b_+ conjugated. Reality of Q,U gives Z_- = Z_+ conjugated, not an
extra independence assumption between branches.

The Gaussian characteristic function gives

    a_s(i) = E[Z_s(i)] = exp[2i s m x_i - 2 sigma_phi^2 x_i^2],
    G_st(i,j;h) = E[exp{2i(s x_i phi(r)-t x_j phi(r'))}]
      = exp[2i m(s x_i-t x_j)]
        exp[-2 sigma_phi^2(x_i^2+x_j^2-2st x_i x_j rho(h))].

For s=t this is equivalently a factor depending on (x_i-x_j) times
exp[-2 x_i x_j D_phi(h)]. Opposite-branch factors are also required.
Set M_st(i,j) = integral W(r)W(r') G_st(i,j;|r-r'|) d^2r d^2r',
and K_st = M_st - a_s(i) a_t(j)*. Then

    mu_on(i) = v_I(i) + sum_s b_s(i) a_s(i),
    mu_off(i) = V_off(i), C_off = C_on,off = 0,
    C_on(i,j) = sum_st b_s(i) b_t(j)* K_st(i,j),
    E[V_on(i)V_on(j)*] = C_on(i,j) + mu_on(i)mu_on(j)*,
    E[V_off(i)V_off(j)*] = V_off(i)V_off(j)*,
    E[V_on(i)V_off(j)*] = mu_on(i)V_off(j)*.

For Delta = V_on - V_off,

    mu_Delta = mu_on - V_off, C_Delta = C_on,
    E[Delta_i Delta_j*] = C_on(i,j) + mu_Delta(i)mu_Delta(j)*,
    E[|Delta_i|^2] = C_on(i,i) + |mu_on(i)-V_off(i)|^2.

The last mean term must survive complete depolarization of the on mean.
For a full description of a possibly improper complex random visibility,
the nonconjugated centered moment is
J_on(i,j) = sum_st b_s(i)b_t(j) K_s,-t(i,j). Its raw second moment is
J_on + mu_on mu_on^T. These formulas determine the covariance of the real
and imaginary readouts; no circular complex-Gaussian assumption is made.
V itself need not have a Gaussian distribution.

## Symmetry reduction and numerical evaluation

The Gaussian weight autocorrelation is
A(h) = exp[-h^2/(4w^2)]/(4pi w^2). Translation invariance and rotational
symmetry reduce the four-dimensional integral to

    M_st = integral_0^infinity [h/(2w^2)] exp[-h^2/(4w^2)] G_st(h) dh.

With u = h^2/(4w^2), kappa = 2w^2/ell^2, rho = exp(-kappa u),
B_ij = 2 sigma_phi^2(x_i^2+x_j^2), A_st = 4 sigma_phi^2 st x_i x_j,
its real amplitude is integral_0^infinity exp(-u-B_ij+A_st rho) du.
The centered amplitude replaces exp(-B_ij+A_st rho) by
exp(-B_ij)[exp(A_st rho)-1]. The code evaluates that difference using
expm1 and a nonpositive exponent; it avoids cancellation and overflow.

Analytic ensemble averaging has removed the random rapid angular phases.
Constant mean depth leaves a known frequency phase outside the integral.
The remaining numerical integral is a nonoscillatory separation integral,
not a sampled random phase map. At equal frequency the same-branch kernel
has small-separation e-fold radius h_coh = ell/(sqrt(2) sigma_phi x).
As wavelength grows this can be much smaller than ell; it must be resolved.

The implementation integrates the centered expression in u, splitting at
multiples of 1/(1+|A_st| kappa) and rescaling each interval to unit length.
The discarded u > 40 tail is bounded in absolute amplitude by 2 exp(-40).
It also evaluates the exact reduced integral via a confluent hypergeometric
function. With a=1/kappa, for A>=0 its amplitude is
exp(-B+A) 1F1(1;1+a;-A), using Kummer's transform to avoid exp(A);
for A<0 it is exp(-B) 1F1(a;1+a;A). The constant-screen limit is evaluated
explicitly. Independent quadrature versus special-function agreement checks
the reduced integration only; neither samples deterministic screen skies.
Storage is O(N_frequency^2), with scalar quadrature workspace. No dense
angular pixel-pair matrix is formed.

## Review evidence and remaining gates

Step 1 executes zero-depth, deterministic nonzero constant-depth, random
spatially constant-screen, small-correlation-area, Hermiticity, positivity,
complex branch and quadrature-refinement checks. An independent 1D Gaussian
integral over a uniform screen tests branch signs, both second moments and
the off/on squared difference. Predictions versus ell and frequency hold
all other physical parameters fixed. Quadrature settings have their own
refinement comparison. Absolute moment tolerances use the unrotated
polarization scale, not a relative error divided by a vanishing mean.

Agreement with a one-point characteristic function is insufficient validation
of a spatial covariance. Step 2 must generate physical Gaussian screens and
compare the same off/on visibility estimator using independently converged
direct angular integration. Its falsifiable protocol is in the plan. Steps
2--4, real-beam audit, large ensembles and forecasts are not run here.

## Relation to the existing depth-histogram template

A branch has g(phi) = integral W(r) delta(phi-phi(r)) d^2r, multiplied by
its deterministic source/response amplitude. Its centered depth covariance
is the double angular integral of
[p_2(phi,phi';rho(h)) - p_1(phi)p_1(phi')]. At finite ell this is generally
not diagonal in depth. The screen mean contributes separately. Recovering
A_power^2 H(phi) delta(phi-phi') requires an additional physical model of
uncorrelated centered depth amplitudes with the stated power weighting and
normalization, not just the one-point Gaussian depth histogram.

After dividing out known smooth source factors, that template has covariance
stationary in x-x'. Here the additional x x' dependence and changing diagonal
power generally preclude it. A narrow-band approximation might freeze x x'
and the coherence-area weighting near a reference x, but must bound its
error for the chosen band and preserve the joint two-branch statistics.
Uncorrelated infinitesimal pixels at fixed variance instead drive continuum
variance to zero: a finite amplitude requires a specified physical coherence
area or a justified continuum scaling. Neither RM-map pixel area nor RM
reconstruction uncertainty provides it. Channel responses, foreground
projection and noise must later act on this same observable before any
comparison to detection products.
