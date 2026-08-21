# The measurement model

What this simulator computes, and the one structural fact that makes it cheap.

This is the conceptual overview. It does **not** rederive the four-port
pair-Stokes formalism — luseepy owns that
(`lusee.InstrumentResponse.pair_stokes_maps`), and this code imports it rather
than reimplementing it. What follows is the layer above: how a Faraday-rotated
sky enters that formalism, and why the frequency axis turns out to be nearly
free.

## 1. What we are simulating

LuSEE-Night sits on the lunar surface with four monopole ports (N, E, S, W) and
correlates them. The observable is a 4x4 port covariance per frequency channel
and per time sample, packed into 16 real science channels. Each entry is a
beam-weighted integral of the sky over the visible hemisphere:

```
V_pq(t, nu) = sum_S  Int  B^S_pq(n) * S(R_t n, nu) dOmega,     S in {I, Q, U, V}
```

`B^S_pq` is the pair-Stokes response of ports `p` and `q`. `R_t` is the rotation
that carries the sky through the topocentric frame as the Moon turns — one
lunar sidereal day is the full time axis.

Three things then happen downstream. The first two are luseepy's: the
open-circuit covariance picks up the Moon and antenna-metal thermal terms, and
the receiver loading `M = Z_L (Z_A + Z_L)^-1` is applied. The third — packing
the result into the 16 channels the spectrometer actually reports — is a local
loop in `instrument.channels`, kept local because luseepy's
`pack_covariance` puts the channel axis at `-2` rather than last, and pinned
elementwise against it by `test_channels_match_luseepy_pack_covariance` so the
two cannot drift apart unnoticed.

## 2. What Faraday rotation does to the sky

Galactic synchrotron emission is linearly polarized. Write the polarized
intensity as a complex field

```
P(n) = Q(n) + i U(n)
```

A magnetized plasma along the line of sight rotates its plane by an angle
proportional to wavelength squared. In the healpy/COSMO convention this is a
pure phase:

```
P(n, nu) = P_0(n) * exp( +2i * phi_FD(n) * lambda^2 )
```

`phi_FD` is the Faraday depth in rad/m^2. Two features matter:

- **It is chromatic and everything else is not.** We deliberately freeze the
  beam at one native response channel across each narrow band. Any structure
  that appears along the frequency axis — and therefore any power in delay
  space — is Faraday-induced by construction. That is the entire point of the
  measurement.
- **It is fast.** At 30 MHz, `lambda^2 ~ 100 m^2`, so a Faraday depth of
  250 rad/m^2 turns the polarization angle through a full cycle every ~1.9 kHz.
  This is why the analysis needs the spectrometer's 64 zoom bins and not just
  its 25 kHz parent channels.

## 3. The structural fact: Faraday is diagonal in the harmonic dual

The convolution in section 1 is done in spherical harmonics. croissant does not
carry `(I, Q, U, V)` through the transform; it carries a **harmonic dual**

```
( I , V , P_MINUS , P_PLUS )
```

where `P_MINUS` is the spin -2 analysis of `Q + iU` and `P_PLUS` the spin +2
analysis of `Q - iU`, both in the IAU convention. The visibility is then a
single diagonal contraction of sky duals against response duals.

Now apply Faraday rotation. Because `U_IAU = -U_COSMO`, for real maps
`(Q + iU)_IAU = conj( (Q + iU)_COSMO )`, so the rotation acts on the two
polarized blocks as conjugate scalar phases:

```
P_MINUS(nu)  =  P_MINUS(0) * exp( -2i phi lambda^2 )
P_PLUS(nu)   =  P_PLUS(0)  * exp( +2i phi lambda^2 )
I, V         unchanged
```

**For any region of constant Faraday depth, rotation is a multiplication by a
number — per block, per frequency. It does not touch the spatial problem at
all.** The transform and the rotation commute with it entirely.

## 4. What that buys: the frequency axis becomes free

If the sky is a union of `K` regions each with its own Faraday depth, its
harmonic dual at any frequency is a fixed set of spatial patterns with
frequency-dependent weights:

```
sky_alm(nu)  =  sum_k  coeff[k, nu, c] * component_alm[k, c]
```

The contraction against the beam is linear, so it can be done **once per
component** and the frequency axis applied afterwards:

```
V(t, p, nu)  =  sum_k sum_c  coeff[k, nu, c] * W[k, c, t, p]
```

`W` is the expensive object: `K` spherical contractions, independent of how
many frequency channels are wanted. Everything after it is one einsum.

Concretely: the fine grid in this analysis is 16,384 channels spanning a few
parent bins. A single transiting source is one component. A uniform screen is
one component. Perfect depolarization (the I-only leakage reference) is one
component. In each case the cost is **one contraction, not 16,384 transforms**.

The coefficient carries the spectral index too, per block — the Stokes-I block
follows the Haslam power law, the polarized blocks follow the WMAP one — so the
"different spectral index for I and for Q/U" bookkeeping falls out of the same
structure rather than needing its own machinery.

## 5. Where this is exact, and where it stops

**Exact**, with no approximation beyond the frozen beam:

- discrete sources, each with its own Faraday depth
- a uniform screen
- any screen that is piecewise constant in `phi_FD`
- Stokes-I-only / perfect-depolarization references

**Not exact**: a continuously varying screen — the real Galactic RM map. It can
be *binned* into constant-depth components, and the decomposition stays exact
per bin, but two separate things then go wrong, and they are worth keeping
apart because they mean different things:

- **A cost problem.** The number of bins needed for the phase to stay coherent
  *across the simulated band* is set by `dphi <~ pi / (2 * span(lambda^2))`.
  For an ionospheric screen (1-30 rad/m^2) that is a couple of dozen
  components. For the full Galactic screen (+-2400 rad/m^2) it is thousands.
  Expensive, but well defined.

- **A validity problem.** The phase must also be resolved *between adjacent
  pixels at fixed frequency*. It is not. At 30 MHz the Faraday phase winds
  ~1700 rad between neighbouring `nside = 512` pixels, so the pixel sum is a
  random walk rather than a quadrature, and Nyquist sampling would need
  `nside ~ 2.8e5` (~1e12 pixels). The input map does not determine the answer
  at any computable resolution.

The second point is the finding of the 2026-08-18 numerical audit, and it is
the reason this refactor does not include a bespoke pixel or NUFFT engine: for
a band-limited beam the harmonic contraction and the pixel sum are the *same*
HEALPix quadrature, so no engine choice rescues that regime. It is not a
performance question.

Both numbers are computed by `sky.audit_screen`, which runs inside
`FaradaySky.binned_screen` — the one constructor that turns a map of Faraday
depths into components, and the one `FaradaySky.from_rm_map` delegates to.
They are logged at INFO on *every* such build, including a successful one, and
the constructor refuses an unresolved screen unless the caller passes
`allow_pixelwise=True` (which then logs a warning instead). The check sits on
the inner constructor deliberately: it used to live on `from_rm_map` alone, so
calling `binned_screen` directly silently built 180 components for a screen
needing `nside ~ 1.5e6` — exactly the regime the audit exists to refuse.
`tests/test_sky_diagnostics.py` pins both halves.

The other constructors (`from_maps`, `uniform_screen`, `point_source`,
`i_only`) do not take a Faraday *map* at all: each component carries a single
depth, so both criteria are satisfied by construction and there is nothing to
report. The audit lives in the API rather than in a paragraph someone has to
remember.

## 6. Conventions, in one place

Sign errors here are the most expensive kind of bug in this codebase, because
they produce plausible numbers rather than obvious failures. Everything funnels
through `lusee_faraday.conventions`.

| Quantity | Convention |
|---|---|
| Input sky Q, U | healpy / COSMO |
| croissant internal | IAU, `U_IAU = -U_COSMO` |
| Faraday rotation | `(Q + iU)_COSMO * exp(+2i phi lambda^2)` |
| Dual blocks | `(I, V, P_MINUS, P_PLUS)`, spins `(0, 0, -2, +2)` |
| Faraday on the duals | `P_MINUS: exp(-2i phi l^2)`, `P_PLUS: exp(+2i phi l^2)` |
| Ports | `0, 1, 2, 3 = N, E, S, W` |
| Response frame | `x = East, y = North, z = zenith`; `phi = 90deg - azimuth` |
| Channels | 16 real, ordered as `lusee.Covariance.default_product_labels()` |

The fixed-beam approximation covers the **receiver loading as well as the
response**. This is not optional: the antenna sits near resonance at 30 MHz,
where one 0.5 MHz native step moves `Z_A` by 12%, and letting the impedances
follow the +-0.1 MHz fine grid moves the loading matrix `M` by 11% across the
band. That is a smooth chromatic ramp of exactly the kind the delay-space
argument asserts is absent, so it has to be frozen with the beam.
`instrument.covariance` and `instrument.blackbody_normalization` both take
`impedance_freq_mhz`, which makes the freeze visible at the call site.

(An earlier version of this paragraph said the opposite -- that `Z_A` and `Z_L`
follow the fine grid "so receiver loading is not smeared along with the beam".
That was written before the 11% was measured, and it was wrong.)

## 7. Where each piece lives

| Module | Owns |
|---|---|
| `conventions.py` | COSMO/IAU, the Faraday phase, port and channel ordering |
| `config.py` | Site, time grid, fine frequency grid, band centres, sky spectral parameters |
| `sky.py` | `FaradaySky`: the component decomposition, the coefficients, and `audit_screen` — the two audit criteria, enforced in `binned_screen` |
| `response.py` | Instrument -> pair-Stokes alms (four-port via luseepy, two-port via croissant) |
| `engine.py` | The block-resolved contraction and the spectral expansion |
| `instrument.py` | Covariance assembly, receiver loading and Hermitian projection (luseepy); the 16-channel packing is local, pinned against `lusee.Covariance.pack_covariance` |
| `polarimeter.py` | Zenith calibration and pseudo-Stokes |
| `channelization.py` | Parent and zoom bins on luseepy's spectrometer response |

Two arms share all of it. The as-built four-port instrument goes through
luseepy, which carries the impedance model, the receiver loading and the units.
The symmetric pseudo-dipoles of the paper's Fig. 4 have no instrument model to
load, so they go through croissant's `PairStokesBeam` directly and stay
unitless. Both meet at the same contraction and the same sky.

## 8. How well the two arms agree, and where they do not

The harmonic contraction has an independent quadrature of the same integral
beside it: the pixel-space engine in `pixel_arm.py`, kept for exactly this
purpose. `scripts/crosscheck_pixel_arm.py` runs them against each other on the
real BGL_v16 response.

The correctness gate is `tests/test_engine_gate.py`, where the harmonic
contraction reproduces luseepy's own convolution to 6.8e-16. The cross-check is
a characterization, and what it characterizes is **sky-dependent**:

| sky | harmonic vs pixel |
|---|---|
| polarized (the crosscheck's own test sky) | 2.678e-02 |
| the same run with `Q = U = 0` | 2.687e-05 |
| the real I-only sky, nside 512, lmax 30 | 3.054e-04 |

Same seed, same nside, same lmax; the only change between the first two rows is
zeroing the sky's Q and U. A factor of ~1000.

**The disagreement is entirely a polarized-sky phenomenon.** An unpolarized sky
samples only the `P^I` pair-Stokes kernel. A polarized sky additionally samples
`P^Q` and `P^U`, and that is where the difference lives -- consistent with the
horizon/Gibbs mechanism the crosscheck records as its leading hypothesis, since
the cross-pol kernels change sign across the beam and so carry far more high-l
structure near the horizon.

Two consequences, and the second is the one to remember:

- The analysis that reaches the paper is **unpolarized leakage**, where the two
  arms agree at 2.7e-5 (synthetic sky) and 3.05e-4 (real sky). Those numbers
  are global-max-normalised, the estimator `crosscheck_pixel_arm.py` uses, so
  they are directly comparable to the 2.678e-2 above; per channel the real-sky
  worst case is 1.77e-3.
- **The two arms are not cross-validated better than ~3% for anything with a
  polarized sky.** Nothing currently in flight depends on that -- the diffuse
  Faraday work of steps 2 and 4 was already set aside by the 2026-08-18
  pixelization audit in section 5 -- but if that work is revived, this is a
  *second and independent* reason for caution, separate from pixelization.

Harmonic truncation is not the explanation: raising lmax from 30 to 48 moves
the real-sky products by 7.77e-5.

Evidence:
`.superpowers/sdd/2026-08-18-luseepy-croissant-refactor/task7_polarization_probe.py`,
log at `generated_data/task7_polarization_probe.log`.

## 9. Bin in depth first, transform second

The old diffuse calculation summed `e^{2i phi(n) lambda^2}` pixel by
pixel; at nside 512 the phase moves ~1730 rad between neighbours and the
sum is a random walk whose amplitude falls as `1/sqrt(N_pix)` — the
2026-08-18 audit's finding. Binning the beam- and emissivity-weighted
sky by Faraday depth first gives `F(phi)`, a stable histogram whose
*shape* is invariant under refinement (`tests/test_dispersion_gates.py`
pins this at nside 256–2048 and under a null rotation); only the
normalisation carries the shot noise. The beam weight itself
(`response.pair_weight_maps`) is basis-independent: it combines both
Faraday branches of the pair-Stokes kernel as
`sqrt(0.5 * (|K_Q|^2 + |K_U|^2))`, not one branch alone. The observable
is then the RM-synthesis pair
`P(lambda^2) = Int F(phi) e^{2i phi lambda^2} dphi`
and, in the incoherent limit, delay power = the `|w|^2`-weighted depth
distribution. `dispersion.py` owns both directions; both use type-3
NUFFTs on the true `lambda^2` nodes, and raw pixel depths can be fed as
nonuniform points directly (`test_converged_regime_points_match_direct_sum`
does exactly this, reproducing a direct coherent sum to four digits).

**Two different claims, two different tests.** The acceptance gates say
*the model's shape is pixelisation-stable*: the normalised `F(phi)`
moves by <1% in Kolmogorov distance and <2% in the 90%-mass knee across
nside 256–2048 and under a rigid null rotation, while the old coherent
observable moved by 7.2x under the same rotation. They do **not** say
*the observable equals the weighted depth distribution* — with the
uniform `w2` those gates use, the normalised template is a functional
of the empirical RM distribution alone, and a rigid rotation is a pixel
permutation that leaves it invariant by construction; the nside 1024
and 2048 legs are `get_interp_val` upsamplings of the native 512 map
and carry no new sky either. That second, load-bearing claim is tested
directly by `test_delay_power_equals_the_weighted_depth_distribution`:
the delay power of the *coherent pixel sum* — the very sum the audit
found to be shot noise in amplitude, with the real WMAP K polarisation
angles as the pixel phases — against `depth_distribution`. Measured at
30 MHz: **1.069** integrated over `30 <= |phi| < 1500`, and **1.332**
over the whole axis, the total-power counterpart. The difference
between the two is the `|phi| < 10` region, where the pixel sum is
genuinely partly coherent and the ratio alone is 1.92.

Read that as an **independent confirmation that the ratio is consistent
with unity**, not as a reproduction of a particular number. It is a
different statistic from the audit's 1.038 — different window (BH4),
different grid, different estimator, and restricted in range — and the
audit's own figure is one of a series running 1.010 / 1.184 / 1.038 /
0.843 across nside, carrying ~13% scatter, so 1.038 is not a precise
target to hit. What this adds is that a second estimator built from
different machinery lands in the same place, which nothing else on this
branch does.

Two limits on what it shows, both measured. It does **not** demonstrate
that the sky is incoherent: setting `c = |c|`, every pixel phase
identical and maximally coherent, still passes at 0.889. The identity
is more robust than "the sky happens to be incoherent" — one turn of
the `e^{2i phi lambda^2}` carrier is a depth separation of just
`pi/lambda^2` = 0.0315 rad/m^2 at 30 MHz, so two pixels one RMSF width
apart (BH4 amplitude FWHM 6.29) are already ~200 turns apart, and
pixel-phase coherence cannot survive unless their RMs agree to ~0.03
rad/m^2. Nor does it pass only because WMAP K's per-pixel noise
randomises the phases: smoothing Q and U to 3 deg and 10 deg gives
1.107 and 1.143 on the same statistic. The `RM x 0.02` converged
control, where the depth distribution collapses inside one RMSF width,
fails by >1e4 — which is what makes this a test rather than an
identity.

**Where the template's shape is and is not robust** (both computed by
`scripts/step5_template.py`, both plotted by `step5_plots.py`, neither
a pass/fail):

- *Plane taper* (spec S4.2.1/S6.5). Down-weighting the Galactic plane
  by `sin^2|b|` moves the 90%-mass knee by a factor **4.3–5.5x** across
  the three bands and both `k` geometries — at 30 MHz, `k=0`, from 89.6
  to 17.9 rad/m^2. The roll-off is therefore carried by the plane, not
  by high-`|b|` sightlines. Per S4.2.1 that is the *unfavourable*
  branch: long in-plane paths are exactly where `B_par` reverses and
  the linear `phi(f)` ansatz is weakest, so the geometry assumption is
  doing real work and **the amplitude bracket must be read as wider,
  not narrower**, than section 12's two usable ends suggest. The
  direction of the reversal effect is still favourable
  (reversals broaden `F`, so the computed roll-off is a lower bound on
  the extent), but its size is not bounded by anything here.
- *Two arms* (S4.9/S6.11). Swapping the as-built four-port `|w|^2` for
  the symmetric two-port one moves the same knee by **+24% at 50 MHz**
  (`k=0`: 83.3 -> 103.0 rad/m^2), **+12% at 10 MHz** and **<1% at
  30 MHz** (-0.7%). The beams weight different declination strips, so
  they see different parts of the RM sky; at the band the paper leads
  with for the knee, they agree.

## 10. The three bands are three different problems

Each band's Faraday resolution is set by the span of `lambda^2` the
zoom bins actually cover. `dispersion.rmsf` builds the point-spread
function from the true zoom-bin weight matrix (`zoom_bin_matrix`,
luseepy's real bin responses): a unit-amplitude Faraday tone is carried
through the bin weights and delay-transformed over the **192 zoom-bin
centres, which span 74609 Hz** — not the fine grid's 199988 Hz.

Two conventions have to be stated or the number is meaningless.
`dispersion.rmsf` returns delay *power*, so a width read off it is a
power FWHM; the conventional RMSF resolution — the one Brentjens & de
Bruyn's `2*sqrt(3)/dlam2` quotes — is an *amplitude* FWHM, wider by
1/0.734. Measured from the committed code with interpolated half-max
crossings on an 80001-point `phi` grid:

| band | power FWHM | **amplitude FWHM** | ideal top-hat, same span |
|---|---|---|---|
| 30 MHz | 5.574 | **7.592** | 7.632 |
| 50 MHz | 25.805 | **35.150** | 35.334 |
| 10 MHz | 0.206 | **0.281** | 0.283 |

(rad/m^2). **The resolution is 7.59 / 35.15 / 0.281 rad/m^2 at
30 / 50 / 10 MHz, amplitude FWHM.** The last column is the exact
half-power width of an ideal top-hat over the *same* 74609 Hz span,
`2*1.8955/dlam2` (`sin x / x = 1/2` at x = 1.8955). The measured RMSF
sits **0.5% below** it, at 191/192 of it — exactly the `(n-1)/n`
correction for summing 192 discrete samples instead of integrating.
So the real bin responses broaden the RMSF **not at all**. At
`phi = 0` there is a structural reason — the tone is flat and
`zoom_bin_matrix`'s columns are normalised, so every bin returns
exactly 1.0 and the width depends only on where the bin centres sit —
and away from `phi = 0` it is a measurement: probing at `phi0 != 0`
against a shape-free control (the same bin centres with
delta-function bins), the bin responses enter as a **multiplicative
envelope**: the peak amplitude response falls 1.00 -> 0.85 -> 0.74 ->
0.48 at `phi0` = 0 / 100 / 300 / 600, while the amplitude FWHM moves by
**<= 0.8%** (measured +0.5% / +0.1% / +0.2%). They attenuate; they do
not broaden — and the 0.48 at `phi0 = 600` is just section 11's 50%
depth horizon of 604 rad/m^2, seen from the other side. That envelope
is where the bin shapes belong.

An earlier version of this section reported 5.57 / 25.8 / 0.207 as the
resolution and called it "~2.14x broader than the idealised top-hat
because a tapered response has less effective bandwidth". Both halves
were wrong. The 2.14 was an artifact of the comparison in three
factors — the fine grid's `dlam2` against the bins' (2.680), a power
FWHM against an amplitude convention (0.734), and `2*sqrt(3)/dlam2`,
a rule of thumb 9.4% below the exact sinc half-power width (1.094) —
and `2.680 * 0.734 * 1.094 * 0.995 = 2.14` closes it with the
discrete-sampling factor above. No tapering enters anywhere. The
numbers are `dispersion.rmsf`'s own output, exercised in
`tests/test_dispersion.py` — **not**
`scripts/step5_instrument_envelope.py`, which never computes an RMSF at
all; that script builds the depth-horizon/percentile table of section
11 below.

Resolution alone ranks the bands the opposite way from usefulness:
10 MHz is sharpest (0.281 rad/m^2), then 30 MHz (7.59), then 50 MHz, the
coarsest (35.2). Resolution and reach are different axes, and 10 MHz's
resolution is superb and irrelevant — section 11 shows its zoom horizon
falls below the beam-weighted median sky depth, so there is nothing
there for that resolution to resolve. 30 MHz is the band where
resolution and reach line up: it hosts the template's
**90%-mass-quantile knee** (`dispersion.mass_quantile_knee` — a CDF
statistic, not a half-power crossing; an earlier half-power version
pinned to a narrow origin spike under the `k=0` slab geometry and was
replaced for exactly that reason) at **89.6 rad/m^2** (fiducial `k=0`
uniform-slab geometry, four-port arm) — well inside its own 604 rad/m^2
zoom horizon and ~12 resolution elements from the origin spike
(89.6 / 7.59; the same ratio is ~16 if the power FWHM is used, which
is why the convention has to travel with the number). 50 MHz
has the coarsest resolution but is the only band whose zoom horizon
reaches the sky's map maximum. 10 MHz is out: its zoom horizon
(22.4 rad/m^2) sits about 4% below the beam-weighted median sky depth
(23.4 rad/m^2) — under the *unweighted* map percentile the same horizon
would sit *above* the median (18.4), so the ordering is itself a
beam-weighting effect, not a robust separation; section 11 works
through both weightings side by side. The chirp of `lambda^2(nu)`
against a uniform frequency grid is an analysis artifact removed by the
NUFFT (`test_nufft_beats_fft_on_a_single_depth`), not a physical wall.

## 11. Channel width is a Faraday-depth horizon

A spectrometer bin is a window in `nu`, so in depth it is a
multiplicative envelope: the instrument imposes its own roll-off on top
of the sky's. The 50% depths from luseepy's real bin responses
(`generated_data/step5_envelope.npz`, regenerated by
`scripts/step5_instrument_envelope.py`) are:

| band | parent horizon | zoom horizon |
|---|---|---|
| 50 MHz | 58.0 | 2797 |
| 30 MHz | 12.5 | 604 |
| 10 MHz | 0.46 | 22.4 |

(rad/m^2), against the sky's **beam-weighted** `|RM|` percentiles — p50
23.4 / p90 185.7 / p99 776.1 / p99.9 1283.0 / max 2442.1 (`weighted =
True` in the npz, using the same LST-and-pair-averaged `|w|^2` the
templates themselves are built from). Beam weighting roughly doubles
p90 and nearly triples p99 relative to the *unweighted* map percentiles
(p50 18.4 / p90 91.0 / p99 278.0 / p99.9 648.8): the paper's version of
this comparison uses the weighted numbers, since that is what the
template actually contains — the unweighted set is a placeholder for
it, not a substitute.

No parent bin reaches the knee: parent horizons range 0.46–58.0 rad/m^2
across the three bands, while the fiducial (`k=0`) knees sit at
83.3–89.6 rad/m^2 — every parent horizon is below every knee, by 1.4x
at the closest approach (50 MHz: 58.0 vs 83.3) and ~180x at the
farthest (10 MHz: 0.46 vs 83.9). At 30 MHz the zoom envelope
(604 rad/m^2) lands near the sky's **98th** beam-weighted percentile
(between p90 = 185.7 and p99 = 776.1) — not the p99.9 an earlier draft
claimed — and only deconvolution separates the two: `dispersion.rmsf`
(built from the real responses, folding included) is the kernel to
deconvolve against. Zoom bins overlap (ENBW 563 Hz on 390.6 Hz
spacing): adjacent bins are correlated and the matched filter in
`noise.py` carries that covariance.

## 12. The amplitude bracket, the tail gate, and the window budget

### The bracket, and the one number this map cannot give

The paper claims a normalised *shape*, not an amplitude, and quotes the
amplitude as a bracket with reasons (spec S4.4). `dispersion.amplitude_bracket`
returns three levels; only their provenance makes them usable.

| level | formula | at 30 / 50 / 10 MHz | status |
|---|---|---|---|
| `upper` (incoherent patches) | `1/sqrt(Omega_beam/theta_c^2)` | 9.9e-3 / 9.0e-3 / 1.1e-2 | **not computable from this map** |
| `lower_slab` | `1/(\|phi_med\| lambda^2)` | 4.3e-4 / 1.2e-3 / 5.0e-5 | robust |
| `lower_dispersion` | `1/(2 sigma_eff^2 lambda^4)` | 5.2e-7 / 4.0e-6 / 6.4e-9 | robust |

Only `upper` contains `theta_c`; the two lower levels are `theta_c`-free
closed forms in the map median depth and `sigma_eff`. That matters
because `theta_c` is not measured here. `dispersion.coherence_angle`
solves `2 lambda^4 D(theta_c) = 1` on the RM structure function, and on
`faraday2020v2` the root lies far below the sampled grid at every band:
with `step5_template.py`'s 0.2–30 deg grid, `D(0.2 deg) = 96.2
(rad/m^2)^2` against targets `5.01e-5 / 3.87e-4 / 6.19e-7`, so under
`D ~ theta^2` the root sits **1385x / 499x / 12465x below the grid's
lower edge** — 0.52 / 1.44 / 0.058 arcsec, three decades below the
map's own nside-512 resolution.

`coherence_angle` clamps to the grid edge there, and **the clamp
overstates**: a clamped return is an *upper bound* on `theta_c`, so the
shipped `upper` (`~1e-2`) is an upper bound overstated by ~3 orders of
magnitude. Extrapolating `D ~ theta^2` would put it at 7.1e-6 / 1.8e-5
/ 8.6e-7 — *below* `lower_slab` at every band, i.e. the bracket
inverts, which is the clearest possible sign that the extrapolation is
not a number to quote. Widening the search grid cannot fix this: the
map carries no information at sub-arcsecond scales, so a wider grid
would extrapolate rather than measure. The honest statement is that
**the incoherent-patch upper bound is not determinable from
`faraday2020v2`**, and the spec's own S4.4 range ("1e-4 down to 1e-6")
already disagreed with the `1e-2` the code produces. Narrowing it is
the follow-up paper's 3D-modelling job, not this one's.

### The tail gate, and what decides it (spec S4.2.2)

`max phi_col = 2442` is not what the thresholds detect — the template
is core-dominated, and the paper's roll-off is the 90%-mass knee. The
`max phi_col` feature is an LST-gated stretch goal: at Galactic-centre
transit the `|w|^2` weighting swings toward the high-`|RM|` inner
plane, and the question is whether that lifts the tail into reach of
the matched-filter thresholds (`noise.py`, S4.10). Measured over 128
LSTs, the tail's power fraction above the fixed beam-weighted p99
peaks at GC transit at
**2.16% / 3.15% / 2.39%** (30 / 50 / 10 MHz, four-port; two-port
2.18 / 3.70 / 2.69), against ~1e-6 away from transit and an LST mean
near 0.8%.

The comparison has a convention in it that changes the answer: that
fraction is a fraction of the template's **power**, while the bracket
is an **amplitude**, and S4.2.2 puts the two "roughly the square root
of the tail's power fraction" apart. The tail's amplitude is therefore
`bracket * sqrt(f)`, not `bracket * f` — a factor 6.8x at 30 MHz.
Against the matched-filter threshold at 24 lunations (1.33e-5 / 1.07e-5
/ 1.67e-5), and with the clamp-derived `upper` excluded as not
computable:

| band | at `lower_slab` | at `lower_dispersion` |
|---|---|---|
| 30 MHz | 6.3e-5, **4.7x — open** | 7.7e-8, 0.006x — closed |
| 50 MHz | 2.1e-4, **19.3x — open** | 7.2e-7, 0.067x — closed |
| 10 MHz | 7.7e-6, 0.46x — closed | 1.0e-9, 6e-5x — closed |

So **the gate opens at 30 and 50 MHz if the diffuse amplitude sits at
the uniform-slab floor and closes at every band if it sits at the
internal-dispersion floor.** What decides it is which depolarisation
floor the medium sets — an external-screen versus mixed-medium
geometry question — and nothing else on the list: not the tail
measurement, which is solid at 2.2-3.7% in both arms; not `theta_c`,
which appears in neither floor; and not integration time, since with
`A ~ n^-1/2 N^-1/4` closing the 30 MHz dispersion-floor gap of 173x
would take ~3e4 times more nights. (The two-port arm, taken
consistently, gives 4.77 / 19.92 / 0.47 — the same calls.)

### The window budget on the delay axis (spec S4.8)

Leakage is `I -> Q,U` through a frozen beam, so it is smooth in
frequency and sits at `phi ~ 0`; the template lives at non-zero `phi`.
The binding requirement is therefore the window's dynamic range.
Injecting a smooth synchrotron-sloped foreground at `|P|/I = 0.15`
through the 4-term Blackman-Harris window actually used, on the 30 MHz
fine grid (`test_foreground_sidelobe_budget`), the contamination
amplitude is strongly `phi`-dependent:

| `phi` [rad/m^2] | leaked amplitude |
|---|---|
| < ~9.4 (BH4 main lobe) | 1.5e-1 falling to ~1e-5 |
| 10–27 (first sidelobes) | peak **3.7e-6** at 10.8 |
| >= 27.5 | <= 1e-6 |
| 90–190 (the 30 MHz knee) | **1.4e-7** |
| > 200 | <= 8.8e-8 |
| > 776 (beyond beam-weighted p99) | <= 2.4e-8 |

So **BH4 is adequate**, against both ends of the bracket, everywhere
the roll-off lives: at the knee it clears the internal-dispersion floor
(5.2e-7) by 3.7x and the uniform-slab floor (4.3e-4) by ~3000x. It is
inadequate only inside its own main lobe, `phi <~ 10`, which is the
low-`phi` core the delay axis was never claimed to protect. The spec's
estimate — "0.15 x 2.5e-5 = 3.8e-6, inadequate against the 1e-6 floor"
— is right about the first sidelobe and wrong about the verdict,
because it treats a single peak sidelobe level as if it applied at
every depth. Without a window the same foreground puts >1e-5 into the
roll-off region (`test_boxcar_would_fail_the_budget`), so the window is
load-bearing.

## See also

- `docs/superpowers/specs/2026-08-18-luseepy-croissant-refactor-design.md` — the design and the decisions behind it
- `docs/superpowers/plans/2026-08-18-luseepy-croissant-refactor.md` — the implementation plan
- `AGENTS.md` — the pinned conventions in operational form
