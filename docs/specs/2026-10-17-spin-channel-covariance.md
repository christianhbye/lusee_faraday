# Spin-2 instrument and channel covariance

This increment implements the statistical projection needed by Phase 2 of
the [paper plan](../plans/2026-10-17-paper-implementation.md). It does not
complete the depth-mixed real-beam calculation or establish detectability.
The calculation predicts an ensemble covariance of polarized sky
visibilities. That covariance is distinct from the instantaneous four-port
electric-field covariance and from radiometer estimator noise.

## Spin convention and transforms

`spin_covariance.py` defines COSMO coefficients `a_plus=E+iB` and
`a_minus=E-iB`, in the existing P_PLUS/P_MINUS dual blocks. It synthesizes
parallel-transported correlations with `(2 ell+1)/(4 pi)` and transforms
them with `2 pi integral d(cos theta)`. The two Wigner rows are
`d(2,2)=cos(theta/2)^4 P(0,4)_(ell-2)(cos theta)` and
`d(2,-2)=sin(theta/2)^4 P(4,0)_(ell-2)(cos theta)`.
The correlation method follows
[Challinor and Lewis (2005), equations 64–65](https://arxiv.org/pdf/astro-ph/0502425);
the Wigner/Jacobi identity is given in
[Celeghini et al. (2014), equation 2.10](https://arxiv.org/pdf/1402.5217).
The coefficient signs above specify the convention independently of either
source. Tests compare both rows with an independent factorial Wigner sum,
then check round trips with nonzero EB.

An independent isotropic Gaussian column has covariance
`sigma_phi^2 exp[-(2 sin(theta/2)/theta0)^alpha]`, with `0<alpha<=2`.
`screen_phase_moments` retains the raw and centered conjugated and
nonconjugated phase moments at each frequency pair. Deterministic rotation,
direct Gaussian quadrature, total foreground-screen polarized power and
helicity spectral positivity are checked. A fine theta grid resolves the
Faraday core; a band-limited Gauss rule alone is insufficient there.

The current spherical phase function describes one fractional foreground
screen. The Phase 1 slab/Gaussian depth profiles require a double-depth
average here before their real-instrument predictions can be claimed.
Multiplying this covariance by a scalar depth-depolarization law would lose
the correlation between emission depths.

## Instrument contraction

`visibility_covariance.py` contracts all polarized helicities and all pair
kernels, including their cross terms. It retains covariance and relation
per sidereal m and evaluates only the requested time pairs. The production
MEPA rotation is shared with `engine.contract`; its luseepy comparison test
continues to pass. Independent finite ensembles of real E/B harmonics check
the complete pair/time moments, m parity and unequal-frequency phases.

The loaded readout is obtained from `instrument.covariance`, with both
beam and impedances frozen at the native band and `T_moon=T_ant=0`.
Pseudo-Stokes use calibrated orthonormal zenith vectors. A common loaded
pseudo-I response to a 1 K enclosure normalizes all readouts in K. The
operator can instead return voltage-power units or all 16 science products.

Real readout kernels fold the pair map into the beam before covariance
contraction. `AngularReadoutProjector` then precontracts the beam, time
phases and spin transform into four angular weights. It retains real and
imaginary parts of both helicity spectra. Tests compare this faster path
with the complete spin-spectrum/pair/time chain.

## Channel response and numerical compression

`channelization.channel_operator` uses the same coordinate selection and
normalized luseepy power responses as `integrate`. Zoom bins retain their
folded images. Both moment types propagate: `W C W^H` and `W J W^T`.
Block callbacks receive actual frequency coordinates and include all pairs,
including broad weak-screen covariance.

`frequency_covariance.py` can interpolate the random process from a smaller
frequency grid. With interpolation T, the effective operator is W T; the
resulting covariance is a congruence and preserves positive semidefiniteness.
This is an approximation that must be refined in the channel observables.
Independent nonstationary spectral modes check its refinement against a
direct fine-grid calculation. No stationarity or coherence cutoff is used.

Conditional Gaussian visibility draws use the full joint real covariance.
They reject appreciable negative eigenvalues and report clipped roundoff
eigenvalues and tolerance. A draw describes a fixed sky pattern; repeated
sidereal days must share it. These draws do not validate Gaussianity of
the beam-averaged signal.

The white-spectrum channel-overlap helper gives covariance per integration
time for independent fine-frequency power estimates. Its diagonal inverse
is the effective bandwidth. It is an estimator-overlap calculation, not a
loaded receiver-noise model; temporal-window effects remain to be checked.

## Real-response driver

`paper_kernels.py` caches full complex ten-pair alms at lmax 96 for native
10/30/50 MHz. It records the full response SHA256, resolved JFET impedance
parameters, calibrated vectors, source and dependency revisions, MEPA
kernels and loaded readout maps. Response ablations receive different keys.
No upstream checkout or external manuscript is edited.

`faraday_covariance.py` has kernels/moments/validate/plot stages. Its initial
configuration covers two assumed foreground screens at 30/50 MHz, three
parent bins, all 64 zoom bins of the central parent, and two sidereal
phases. Each result checks doubled angular resolution, 17 versus 33
frequency nodes and lmax 64 versus 96. Thresholds are fixed in the input
configuration; failures are retained. Local sigma, intrinsic E/B spectra,
1 K reference polarization and turnover are explicitly illustrative.

Remaining Phase 2 gates include mixed emitting depths, the full local-prior
bracket, direct spherical/flat comparisons in the specified benchmark
windows, non-Gaussian screen controls and stronger rotation/channel cases.
The measured Q/I and U/I figure still needs the common leakage, smooth-fit
and loaded radiometer-noise treatment in Phase 3. Future LuSEE-Night data
are the comparison target.
