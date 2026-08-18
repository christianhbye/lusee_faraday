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

Three things then happen downstream, all of them luseepy's: the open-circuit
covariance picks up the Moon and antenna-metal thermal terms, the receiver
loading `M = Z_L (Z_A + Z_L)^-1` is applied, and the result is packed into the
16 channels the spectrometer actually reports.

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

Both numbers are computed and reported whenever a screen is built, and the
constructor refuses an unresolved one unless the caller opts in explicitly.
The audit lives in the API rather than in a paragraph someone has to remember.

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

The fixed-beam approximation applies to the **response** only. `Z_A` and `Z_L`
are evaluated on the fine frequency grid, so receiver loading is not smeared
along with the beam.

## 7. Where each piece lives

| Module | Owns |
|---|---|
| `conventions.py` | COSMO/IAU, the Faraday phase, port and channel ordering |
| `config.py` | Site, time grid, fine frequency grid, band centres, sky spectral parameters |
| `sky.py` | `FaradaySky`: the component decomposition, the coefficients, the two audit criteria |
| `response.py` | Instrument -> pair-Stokes alms (four-port via luseepy, two-port via croissant) |
| `engine.py` | The block-resolved contraction and the spectral expansion |
| `instrument.py` | Covariance assembly, receiver loading, channel packing — all luseepy |
| `polarimeter.py` | Zenith calibration and pseudo-Stokes |
| `channelization.py` | Parent and zoom bins on luseepy's spectrometer response |

Two arms share all of it. The as-built four-port instrument goes through
luseepy, which carries the impedance model, the receiver loading and the units.
The symmetric pseudo-dipoles of the paper's Fig. 4 have no instrument model to
load, so they go through croissant's `PairStokesBeam` directly and stay
unitless. Both meet at the same contraction and the same sky.

## See also

- `docs/superpowers/specs/2026-08-18-luseepy-croissant-refactor-design.md` — the design and the decisions behind it
- `docs/superpowers/plans/2026-08-18-luseepy-croissant-refactor.md` — the implementation plan
- `AGENTS.md` — the pinned conventions in operational form
