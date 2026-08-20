# The Faraday delay template: what to look for in LuSEE data

Date: 2026-08-20
Status: proposed design, pending approval
Branch base: `luseepy-refactor` (PR #3)
Supersedes: "path 2 — ensemble diffuse prediction" as sketched on 2026-08-18

## 1. Why

The 2026-08-18 audit refuted the diffuse-Faraday **amplitude** in report Steps
2-4. It did not refute the **shape**, and two of its own measurements say so:

- the delay power falls as `1/N_pix` -- an amplitude that scales as one over
  the number of samples;
- the spectrum equals the `|w|^2`-weighted RM histogram, **total power ratio
  1.038** -- a shape match.

Refining the grid samples the same underlying depth distribution more finely.
The histogram's shape converges; its normalisation does not. So the correct
reading of the audit is *the amplitude was `1/sqrt(N_pix)`, the shape was the
sky's*. That reading has not been written down anywhere and it is what makes a
paper still possible.

The same split applies to the input map. `faraday2020v2` is bad at fixing the
**phase** -- median `sigma/|RM| = 0.50`, 312 turns of phase uncertainty at
30 MHz, and on 24.6% of the sky it does not fix the sign (audit README S5.2).
It is good at fixing the **extent** of the Faraday depth distribution, because
that is what an RM measured against background sources actually is. The old
approach leaned on the map for the thing it cannot do. This one leans on it for
the thing it can.

What the collaboration liked about the old result -- a predictable delay-space
signature with a roll-off at high delay set by the maximum Faraday depth,
present at 50 MHz and absent at 10 MHz -- is exactly the part that survives.

## 2. Decisions taken

| Question | Decision |
|---|---|
| What the paper claims | The **normalised shape** of the diffuse Faraday delay signature, as a search template. Not its amplitude. |
| Amplitude | Quoted as a **bracket** with the physical reason, never as a value. |
| Input maps | Unchanged: Haslam 408 (I), WMAP K (Q/U), `faraday2020v2` (depth extent). No new external data. |
| Geometry | One scalar `k` (`rho(f) ~ f^k`) interpolating all-far to all-local, both limits computed. Fiducial = uniform slab. **Posited, not derived** -- the map carries no depth ordering. |
| 3D Galactic models | **Out.** hammurabi / NE2001 / YMW16 / JF12 and pulsar RM-DM growth curves are the follow-up paper. |
| Zoom deconvolution | **In scope and load-bearing.** It is what separates the sky's roll-off from the spectrometer's at 30 MHz (S4.6). |
| Zenith calibration | A Step-0 point-source result, quoted as such. It nulls only at zenith and **does not reduce diffuse hemisphere-integrated leakage** -- the diffuse template must not lean on it (S4.7). |
| Leakage rejection | By the delay axis, not the polarimeter: leakage is frequency-smooth and sits at `phi ~ 0` (S4.8). The requirement is window dynamic range. |
| Two-port vs four-port | **All the way through the diffuse template**, not confined to the point source. The beam enters only through `|w(n)|^2`, so it is one input swap (S4.9). |
| Noise | Radiometer noise in, as a detectability *threshold curve* against the amplitude bracket (S4.10). Port `noise.py` (25 lines, standalone). |
| Band split | 50 MHz for the roll-off, 30 MHz for the low-`phi` core, 10 MHz out (S4.5). |
| The refuted numbers | `|dP|/I ~ 1.7e-4` and the Step-4 amplitude do not appear in the paper in any form. |

## 3. The load-bearing idea

**Bin in Faraday depth first, transform second.**

The old calculation summed `e^{2i phi(n) lambda^2}` pixel by pixel. At nside 512
the phase moves ~1730 rad between neighbours, so the sum is a random walk. If
instead the contributions are **binned by depth** into `F(phi)` and the transform
is applied to the binned distribution, the binning is a stable histogram
operation and the transform is one 1D FFT.

The observable is then the standard RM-synthesis relation

```
P_obs(lambda^2) = Int F(phi) e^{2i phi lambda^2} dphi
```

and in the fully-depolarised limit -- which the audit's ratio of 1.038 confirms
the simulation reaches -- the delay-space power is

```
<|P~(phi)|^2>  =  the |w|^2-weighted depth distribution
```

with weights `w = beam x polarised emissivity`. **This shape is invariant to how
finely the sky is subdivided**; only the normalisation carries the `1/N` that
the audit measured. That invariance is the whole design, and it is a testable
claim (S6).

Required `phi` bin width is `pi / (2 lambda^2)`: 0.016 rad/m^2 at 30 MHz,
0.0017 at 10 MHz. Over +-100 rad/m^2 that is 13k and 114k bins -- trivial, and
the same grid sizing already handled on `faraday-rmsynth`.

## 4. Architecture

### 4.1 New module: `dispersion.py`

Production arm, alongside `sky.py`. Owns `F(phi)` and its transform. Does not
import `pixel_arm`.

```
depth_distribution(rm_map, weights, phi_grid, geometry) -> F(phi)
    Bin the beam- and emissivity-weighted sky by Faraday depth.

transform(F, phi_grid, lam2) -> P(lambda^2)
    The RM-synthesis integral. finufft is already a dependency.

delay_power(F, phi_grid, freqs, bin_weights) -> <|P~(phi)|^2>
    The observable. Type-3 NUFFT onto the phi grid, NOT an FFT on a
    uniform nu grid -- see S4.5 on the chirp.

rmsf(freqs, bin_weights) -> the response function to deconvolve against
    Built from luseepy's real spectrometer response via
    channelization.parent_weights / zoom_weights, not a boxcar (S4.6).
```

### 4.2 The geometry knob

The map gives exactly one number per pixel: `phi_col(n)`, the total column to
infinity. **Nothing in it says where along that column the emission sits.** The
rows below are posited, not derived, and the text must say so in those words.

Getting from `phi_col(n)` to a depth distribution stacks two assumptions:

1. **how depth accumulates** -- `phi(f) = f * phi_col(n)` for fractional
   distance `f`, i.e. `n_e B_par` constant along the path;
2. **where the emission is** -- `rho(f)`, the emissivity weighting.

What is needed is the pushforward of `rho` through `phi(f)`:

```
F(phi) = Int_0^1 df rho(f) x [ depth histogram of f * phi_col(n) ]
```

The fiducial is privileged for a specific reason: **for a homogeneous medium
that both emits and rotates, these are the same assumption, not two.** Uniform
`rho` plus linear `phi(f)` gives depth uniform on `[0, phi_col]` and a `sinc`
transform -- Burn's uniform slab. It is the only row that is internally
self-consistent rather than a limiting caricature.

Take `rho(f) ~ f^k` and vary the single scalar `k`. Three cases, all computed,
all cheap:

| `k` | `rho(f)` | meaning | `F(phi)` | signature |
|---|---|---|---|---|
| `-> inf` | `delta(f-1)` | all emission behind the column | the RM histogram itself | full, most depolarised |
| `0` | `1` (uniform) | **fiducial** -- uniform slab | superposition of top-hats `[0, phi_col(n)]` | roll-off at or beyond `max phi_col` |
| `-> -1` | `delta(f)` | all emission local (Local Bubble) | `delta(phi ~ 0)` | none, least depolarised |

The two limits bracket the answer. They trade against each other -- nearby
emission has no signature, distant emission is depolarised away -- so **the
observable is maximised at intermediate depth**. That is a real physical result
obtainable from the data already in `data/`, and it is the paper's honest
treatment of the geometry uncertainty. One scalar, both limits shown. The
paper shows the template family over `k` and demonstrates that the roll-off
location is stable across it while the low-`phi` weighting is not.

**The roll-off sits at or beyond `max phi_col`**, not exactly at it, in every
case except the all-local limit. `B_par` reverses along real sightlines --
spiral-arm reversals, the disk-halo transition, the field flipping across the
plane -- so `phi(l)` is not monotonic and partial depths at intermediate `f` can
exceed `|phi_col|` through cancellation on the outbound path. The linear ansatz
misses this entirely.

The direction is favourable and must be stated: reversals put emission at depths
**beyond** `phi_col`, so they broaden `F` rather than shrink it. The computed
template is narrower than truth and the roll-off is a **lower bound on the
extent**. A feature at or beyond the predicted location is still a prediction.

### 4.2.1 The plane-cut robustness check

Reversals are worst for long in-plane paths and mildest at high Galactic
latitude, so the assumption is testable against itself without new data:
**recompute the template with the Galactic plane down-weighted** (a `|b|`
taper on `|w(n)|^2`, not a hard cut -- the beam covers half the sky and cannot
literally exclude the plane).

If the roll-off is carried by high-`|b|` sightlines, the linear ansatz is doing
little work and the feature is robust. If it is carried by the plane, the paper
says so and the amplitude bracket widens. Either way it directly bounds the
assumption the design is most exposed on, for a few lines on top of S4.2.

### 4.3 Where the beam enters

`weights` come from the PR #3 four-port pair-Stokes windows at the frozen
response channel, per channel pair and per LST sample. This is what makes the
template a prediction for our 16 products rather than for an idealised `P`. The
`(pq, p'q')` structure is also the discriminant against `I -> Q,U` leakage,
which populates the products differently.

### 4.4 Amplitude bracket

Not a prediction. Two numbers, with derivations, in the discussion:

- **Upper**: the incoherent-patch estimate, residual `~ 1/sqrt(N_patch)` with
  `N_patch = Omega_beam / theta_c^2` and `theta_c` from `2 lambda^4 D_phi(theta_c) = 1`.
- **Lower**: the mixed-medium power-law floors -- uniform slab `~1/(2 phi lambda^2)`,
  internal dispersion `~1/(2 sigma^2 lambda^4)`.

State plainly that the external-screen idealisation is the pessimistic end and
that the mixed geometry converts exponential suppression into a power law.

### 4.5 The three bands are three different problems

"50 MHz yes, 10 MHz no" is too coarse and hides the interesting case. Three
quantities decide it, and they do not rank the bands the same way. All computed
for the `+-0.1 MHz` fine window, `sigma_eff = 9.8 rad/m^2`:

```
band |  lam2  | RMSF dphi | coherence BW | = parent | = zoom | chirp @ phi = 100 / 500 / 2442
-----+--------+-----------+--------------+----------+--------+-------------------------------
  50 |  35.95 | 12.0      |   50.2 kHz   |  2.01    | 128.5  |   0.1     0.5      2.7
  30 |  99.86 |  2.60     |   10.8 kHz   |  0.43    |  27.7  |   0.8     4.2     20.7
  10 | 898.76 |  0.096    |    401 Hz    |  0.02    |   1.0  |  68.7   343.3   1676.7
```

(The 30 MHz RMSF reproduces the audit's `2 sqrt 3 / Dlambda^2 = 2.60 rad/m^2`;
the chirp column reproduces the delay-axis table. Both are cross-checks that the
formulas are right, not new results.)

A fourth quantity decides it more sharply than any of these -- the channel
response's own envelope in Faraday depth (S4.6). Read that table first; the
band split below follows from it.

Two questions must be kept apart, because they answer differently per band:

- **Resolution** -- does the channelization sample the signal finely enough?
  Set by coherence bandwidth against bin width.
- **Horizon** -- how deep in `phi` can the bin see at all? Set by the channel
  response (S4.6). These are Fourier partners of each other, not independent
  facts, but they bind at different places.

```
band   coh BW    vs 25 kHz parent          parent horizon   zoom horizon   sky
 50    50.2 kHz  2 samples/coh length  ->     58.7             2830        p50   18.4
 30    10.8 kHz  parent averages 2.3x  ->     13.3              613        p99  278.0
 10     401 Hz   ~1 zoom bin/coh len   ->      2.7               24        max 2442.1
```

**50 MHz -- the roll-off band. Parent bins for the bulk, zoom for the
roll-off.** Faraday rotation is slow enough here that parent bins already
resolve the typical signal -- the coherence bandwidth is two parent bins, so
zoom buys nothing for the bulk of the depth distribution (p50 = 18.4, inside
the parent horizon of 58.7). But the roll-off is at `max phi ~ 2442`, far
outside that horizon, so **the roll-off measurement specifically requires
zoom**, which extends the horizon to 2830. This is the only band whose zoom
horizon clears the map maximum, which is why the roll-off is measurable here at
all. The cost is Faraday resolution of only 12 rad/m^2: the roll-off is
measurable, the low-`phi` core is not resolved.

**30 MHz -- the core band, with a contaminated roll-off. Zoom required
throughout.** The coherence bandwidth is 0.43 of a parent bin and the parent
horizon (13.3) is below the sky median (18.4), so parent bins are inadequate
for *everything* here, not just the tail. Resolution is 2.60 rad/m^2, five
times better than 50 MHz, so this is where the low-`phi` shape lives. But the
zoom horizon (613) coincides with the sky's p99.9 (648.8): instrument and sky
roll off in the same place, and only deconvolution separates them. Chirp is
sub-element at low `phi` but 4-21 elements at the depths that set the roll-off.

**10 MHz -- out on every count.** Zoom horizon 24 rad/m^2, below the *median*
sky depth. Coherence bandwidth is one zoom bin. Chirp reaches 1677 resolution
elements. Resolution is superb and irrelevant.

So the paper leads with **50 MHz for the roll-off and 30 MHz for the core**, and
says why. An earlier draft of this spec recommended 30 MHz outright; that was
decided on resolution and coherence bandwidth alone, before the horizon of S4.6
was computed, and it was wrong.

**Usable bandwidth is set by the frozen beam, not by the zoom window.** Read off
the artifact: `lusee_bgl_v16_response_v3.fits` has `freq` = **150 channels from
0.5 to 75 MHz, 0.5 MHz spacing**. One frozen-beam chunk is therefore ~0.5 MHz,
and the current `+-0.1 MHz` fine grid sits comfortably inside one. The 25 kHz
zoom window caps bandwidth only where zoom is *required* -- everywhere at
30 MHz, and for the roll-off measurement alone at 50 MHz.

**The chirp is an analysis choice, not a physical wall.** It is an artifact of
FFT-ing on a uniform `nu` grid when the conjugate variable is `lambda^2`. A
type-3 NUFFT onto a chosen `phi` grid removes it -- on a single depth it
recovers width 2.36 instead of 11.80. `finufft` is already a dependency. So the
design uses the NUFFT throughout and the chirp table serves to justify *why*
that is mandatory rather than to rank the bands.

Both the coherence and RMSF columns scale as `1/sigma_eff`; show the p10-p90
spread (3.65-43.1).

### 4.6 Channel width sets a Faraday-depth horizon

**This is the result that motivates the zoom response and the deconvolution,
and it should be a figure in the paper.**

A channel response is a window in `nu`, so in `phi` space it acts as a
multiplicative envelope: the instrument imposes its *own* roll-off in Faraday
depth, on top of the sky's, and the two are confusable. Computed from luseepy's
actual `spectrometer_response` / `spectrometer_response_zoom` (not a boxcar),
the depth at which each bin's response falls to 50%:

```
band     zoom bin    parent bin
 50 MHz     2830          58.7     rad/m^2
 30 MHz      613          13.3
 10 MHz       24.0         2.7
```

against the sky's own depth distribution from `faraday2020v2`:

```
|RM|   p50 18.4   p90 91.0   p99 278.0   p99.9 648.8   max 2442.1  rad/m^2
```

Two conclusions, both first-order for the paper:

- **No parent bin at any band can reach the roll-off.** The deepest parent
  horizon is 58.7 rad/m^2 at 50 MHz, against a roll-off at `max phi = 2442`.
  Be precise about what this does and does not say: at 50 MHz the parent
  horizon sits *above* the median depth (18.4) though below p90 (91), so parent
  bins there see the bulk of the distribution and miss only the tail -- which is
  exactly why S4.5 says "parent for the bulk, zoom for the roll-off". At 30 and
  10 MHz the parent horizon (13.3, 2.7) is *below* the median and parent bins
  miss even the bulk. This is the same physics as S4.5's coherence-bandwidth
  row -- horizon and coherence bandwidth are Fourier partners -- stated where
  the science is, and it is the cleanest possible motivation for the zoom mode
  existing at all.
- **Zoom deconvolution changes the roll-off, and is not generic instrument
  bookkeeping.** At 50 MHz the envelope clears the map maximum, so deconvolution
  is a modest correction. At 30 MHz it lands on the sky's p99.9 -- what a naive
  read calls the Faraday roll-off *is the spectrometer* -- and the measured
  recovery of 0.7947 (against 0.8598 for ideal Gaussian bins) means a ~21%
  systematic sits exactly where the science is. At 10 MHz the envelope is below
  the median depth and nothing survives.

The ENBW of the real zoom bin is 563 Hz against a 390.6 Hz spacing, so zoom bins
overlap by ~1.44x. Adjacent zoom bins are correlated; the deconvolution must
account for it and the amplitude bracket must not count them as independent.

Everything below follows from taking the real response seriously rather than a
rectangular band.

The repo already channelizes on the **real** instrument response --
`channelization.py` calls luseepy's `spectrometer_response` and
`spectrometer_response_zoom`, with `ideal_zoom_weights` as the Gaussian
comparison. The template must inherit that, and four consequences follow.

1. **Compute `R(phi)` from the actual bin weights.** `2 sqrt 3 / Dlambda^2` is
   the rectangular idealisation. Keep it as a printed comparison, not as the
   definition. Sidelobe structure differs and the template is convolved with
   `|R|^2` (S4 step 4), so this propagates directly into the published shape.
2. **Parent bins overlap.** `PARENT_HALF_WIDTH_HZ = 50000` on 25 kHz spacing
   means each parent response spans four parent widths. Adjacent parent bins are
   correlated, which matters for mode counting in the amplitude bracket and for
   any SNR line -- they are not independent samples.
3. **Zoom bins carry folded images.** The zoom FFT runs on the critically
   sampled 25 kHz parent stream, so aliased power is present by construction
   (`channelization.py` docstring: the folding "is physical and is removed
   downstream, not here"). For a delay template, aliased Faraday power is a
   contaminant that must be modelled, not assumed gone. Verify what "removed
   downstream" currently does before relying on it.
4. **Notches must be in the RMSF.** `parent_weights(..., notch=)` exists. Holes
   in `lambda^2` coverage raise sidelobes hard -- the Step-0 RM-synthesis
   calibration measured ~73% first sidelobe from a sparse three-band comb. Any
   notch in the real observing plan belongs in `R` before the template is
   published.

### 4.7 The zenith polarimeter does not help the diffuse search

Two reasons, one structural and one measured.

**Structural.** The port weights are frozen per band -- the same fixed-beam
approximation as the response -- so the polarimeter is a fixed linear
recombination of the 16 channels applied identically at every fine frequency.
It commutes with the `lambda^2` transform: combine-then-transform equals
transform-then-combine. **Nothing it does can move a feature along the `phi`
axis.**

**Measured.** It nulls leakage *at zenith*, to ~1e-16, and nowhere else. The
0.133849 -> 6.516e-4 pair often quoted is an **unpolarized point source at
transit** (`step1_point_source.py`: the source culminates near zenith), and the
6.5e-4 residual is explicitly "set by the 0.56 deg offset from exact zenith".
Away from zenith the as-built leakage is near-total -- `p_leak` **peaks at 0.94
near 32 deg altitude**. Hemisphere-integrated over the real beam it does not
help at all, and `PROGRESS.md` already records this:

```
parent (Q,U)/I = (0.146, -0.032) = I-only leakage to 2e-4;
zenith calibration does NOT reduce diffuse hemisphere-integrated leakage
```

So do not write, in the spec or the paper, that the calibration sets the floor
for the diffuse search. It does not. It belongs in the paper as the Step-0
instrument result it is, quoted on the point source, and nothing in the diffuse
template may lean on it.

### 4.8 What does separate leakage from signal: the delay axis

Leakage is `I -> Q,U` through a per-band frozen beam, and Stokes I carries no
Faraday structure. **The leakage is therefore smooth in frequency and sits at
`phi ~ 0`.** The template lives at non-zero `phi`. Leakage exceeds the diffuse
signal by ~1000x in amplitude and is still separable, because the two occupy
different parts of the axis.

This is a third, independent argument for the delay-space framing -- alongside
S3 (binning converges where the coherent sum does not) and S4.6 (the channel
response sets a depth horizon). It is worth stating as such in the paper.

The binding requirement moves accordingly: **not the polarimeter, but spectral
leakage from the `phi = 0` foreground into the roll-off region.** That is a
window-function and dynamic-range problem, which is why
`step4_power_spectra.py` already uses Blackman-Harris.

It is quantifiable and the spec requires it computed. A 4-term Blackman-Harris
has peak sidelobe ~-92 dB in amplitude (2.5e-5). Against a foreground at 0.15
that leaves ~3.8e-6 of sidelobe contamination. Adequate if the template
amplitude is ~1e-4; **inadequate at the pessimistic end of S4.4's bracket**,
where the mixed-medium floors reach 1e-6.

**So the required window dynamic range is set by the amplitude bracket.** The
paper must state the sidelobe level actually achieved and the template
amplitude at which it stops being sufficient. If the bracket's low end is in
play, BH4 is not enough and a higher-dynamic-range window (or an explicit
foreground subtraction at `phi ~ 0`) becomes a real scope item.

### 4.9 As-built four-port vs symmetric two-port

**Push this all the way through the diffuse template, not just the point
source.** It is nearly free: S4.3 makes the beam enter *only* through
`|w(n)|^2`, so swapping arms swaps one input rather than re-running an
analysis. `response.four_port_pair_alms` and `response.two_port_pair_alms`
already exist side by side.

It is also not merely a systematic check. The two beams weight different parts
of the RM sky, so they can produce different `H(phi)` and a different roll-off
-- the paper's headline feature. Zenith is pinned at declination ~ -23.8 deg,
so a zenith-weighted beam samples one declination strip while a beam with
significant low-altitude response samples a much wider range; how that maps
onto Galactic latitude, and therefore onto `phi_col`, is a computation.

**Do not pre-judge the sign.** "As-built leakage peaks at 0.94 near 32 deg
altitude, therefore deeper columns" is atmospheric reasoning and the Moon has
no atmosphere. Compute it.

Keep the existing point-source comparison (`compare_main_vs_asbuilt.py`,
`beam_ablation.py` with the `_c4sym` and `_diagza` artifacts) as the clean,
exactly-computable validation case. The diffuse template gets both arms
overlaid on the same axes.

### 4.10 Radiometer noise and the detectability threshold

The system is sky-dominated at all three bands, so the **fractional** radiometer
noise is `1 / sqrt(dnu * tau)`, independent of `T_sys`. Consequence worth
stating in the paper: the bands do not differ in fractional sensitivity, so band
choice is driven by S4.5 and S4.6, not by noise.

Per zoom bin (ENBW 563.4 Hz) per time sample (2305 s, from `N_TIMES = 1024` over
a lunar sidereal day): `sigma_frac = 8.8e-4`.

For the quadratic estimator with `N_modes = 1024 LST x N_freq`, repeated lunar
nights coadd **coherently** -- same LST, same sky, independent noise -- so
`sigma -> sigma/sqrt(n)` and `N_modes` does not grow with `n`:

```
A_threshold(5 sigma)  =  sigma_1 * sqrt( 5 / (n * sqrt(N_modes)) )   ~  n^-1/2 N^-1/4
```

`N_freq` = usable bandwidth / coherence bandwidth, where "usable" means
channelized finely enough for the depths in question (S4.5):

```
configuration                                          N_modes    n=1     n=12    n=60
30 MHz, zoom on 3 parents  (75 kHz)                       7086  2.1e-4  6.2e-5  2.8e-5
30 MHz, zoom on 8 parents  (200 kHz)                     18944  1.7e-4  4.8e-5  2.2e-5
50 MHz, parent bins, 200 kHz          (bulk)              4086  2.5e-4  7.1e-5  3.2e-5
50 MHz, parent bins, 500 kHz          (bulk, beam limit) 10199  2.0e-4  5.6e-5  2.5e-5
50 MHz, zoom on 3 parents             (roll-off)          1526  3.1e-4  9.1e-5  4.1e-5
50 MHz, zoom on 8 parents             (roll-off)          4086  2.5e-4  7.1e-5  3.2e-5
```

**Integration time beats bandwidth, and the exponents say why.** Nights buy
`n^-1/2`; modes buy only `N^-1/4`. Tripling zoom coverage from 3 to 8 parents
improves the 30 MHz threshold by 29%; going from 1 to 12 nights improves it by
3.5x. Physically: more modes only average down the variance of the power
estimate, whereas more nights reduce the noise itself, and the estimator is
quadratic in `sigma`. **The ask to the collaboration is observing time, not
downlink.**

Caveat that must be in the paper: coherent night-stacking assumes a static sky
and good LST registration. If registration is poor the scaling degrades toward
`n^-1/4` and every threshold above worsens by up to `sqrt(n)`.

Coherence bandwidth scales as `nu^3` (verified: 401 / 10838 / 50176 Hz at
10/30/50, ratios 27.0 and 4.63), so mode count at fixed bandwidth goes as
`nu^-3`. Moving 30 -> 50 MHz costs 4.63x in modes, i.e. 1.47x in threshold.
That is the price of 50 MHz's clean roll-off horizon, and the two effects should
be presented together rather than in separate sections.

**The threshold sits at the optimistic end of the S4.4 amplitude bracket**
(1e-4 down to 1e-6). Detection requires the truth to be near the top of it.

This is the paper's sensitivity result and it should be its spine: *here is the
threshold, here is the bracket, here is what it would take.* It is a stronger
and more honest structure than a single predicted amplitude, and it gives the
follow-up paper a sharp motivation -- the answer hinges entirely on narrowing
the bracket, which is what 3D modelling and pulsar growth curves do.

`noise.py` from `faraday-fisher-forecast` is 25 lines
(`radiometer_sigma`, `add_noise`) with no dependency on the deleted `sim.py`.
Port it as-is; it is a free grab, unlike the rest of that branch.

Present the threshold as a **curve** over integration time and mode count, not
as the four rows above -- the rows are the sanity check, the curve is the
figure.

## 5. Deliverables

- `src/lusee_faraday/dispersion.py` and its tests.
- `src/lusee_faraday/noise.py` -- ported verbatim from
  `faraday-fisher-forecast`, plus tests.
- `scripts/step5_sensitivity.py` -- the threshold curve over integration time
  and mode count, overlaid with the S4.4 amplitude bracket. This is the
  paper's headline sensitivity figure.
- `scripts/step5_instrument_envelope.py` -- regenerates the S4.6 envelope table
  from luseepy's response and the map percentiles. These numbers are quoted in
  the paper, so they must be reproducible from committed code, not from this
  spec.
- `scripts/step5_template.py` -- build `F(phi)` for the three bands and the
  three geometries, transform, emit the normalised delay templates.
- `scripts/step5_plots.py` -- the template figure (three bands x the `k`
  family), the roll-off figure, the chirp/coherence figure, and the
  plane-taper overlay from S4.2.1, and **the S4.6 envelope figure** -- the
  instrument's Faraday-depth horizon against the sky's depth distribution, per
  band and for parent vs zoom. This is the paper's instrument-methods figure,
  and **the two-arm template overlay from S4.9**.
- `docs/measurement-model.md` S9 -- the bin-then-transform argument, S10 the
  three-band table from S4.5, and S11 the depth horizon of S4.6.
- A window dynamic-range budget (S4.8): achieved sidelobe level against the
  S4.4 amplitude bracket, and the amplitude at which BH4 stops sufficing.
- No new module for channelization: `dispersion.rmsf` consumes
  `channelization.parent_weights` / `zoom_weights` as they stand.
- Report surgery: Steps 2 and 4 are replaced by the template section. The
  refuted figures and numbers come out. `PROGRESS.md` records the mixed-
  provenance figure list already; that list shrinks to zero on this branch.

## 6. Testing

The first two are the acceptance gates. They are chosen to be the direct
rebuttals of the audit's two positive findings against the old result.

1. **Shape invariance under refinement.** The normalised template computed at
   nside 256 / 512 / 1024 / 2048 must agree to a stated tolerance. The old
   amplitude fell as `1/N_pix` across this range; if the shape does too, the
   design is wrong and we stop.
2. **Shape invariance under the null rotation.** A rigid rotation of the grid is
   physically null. It moved the old `|P|` by **7.2x**. The normalised template
   must be stable. This is the single best demonstration that the new observable
   is not shot noise.
3. **Analytic limits.**
   - `F = delta(phi - phi_0)` -> pure winding, no depolarisation; must match the
     existing point-source arm at `PHI_FD_POINT = 250`.
   - `F` top-hat on `[0, phi]` -> `sinc(phi lambda^2)`, closed form. Note the
     factor: `Int_0^1 e^{2 i f phi lambda^2} df` has modulus
     `|sin(phi lambda^2) / (phi lambda^2)|` under the repo's
     `e^{+2i phi lambda^2}` convention, *not* `sinc(2 phi lambda^2)`.
   - `F` Gaussian width `sigma` -> `exp(-2 sigma^2 lambda^4)`, Burn.
4. **Roll-off location** sits at or beyond the beam-weighted `max phi_col`.
   Assert a lower bound, not equality: field reversals broaden `F` outward
   (S4.2), so equality is the narrow-truth limit rather than the expectation.
5. **Plane-taper stability** (S4.2.1). Recompute the normalised template with a
   `|b|` taper on `|w(n)|^2` and report how far the roll-off moves. This is a
   reported number, not a pass/fail -- it bounds the linear-ansatz exposure.
6. **Converged-regime agreement.** Reproduce the audit's `RM x 0.02` positive
   control and the `lambda^2 < 0.5` part of panel C, where the simulation does
   converge and all three nsides agree to four digits.
7. **RMSF against the boxcar.** The rectangular idealisation
   `2 sqrt 3 / Dlambda^2` gives 12.0 / 2.60 / 0.096 rad/m^2 at 50 / 30 / 10 MHz
   (the 30 MHz value is the audit's). Assert that as a *comparison* and report
   the real spectrometer RMSF's width and first sidelobe alongside it. The
   boxcar is the sanity check; the real response is what the template uses.
8. **NUFFT vs FFT on one depth.** A single `phi` must recover at width 2.36 via
   the type-3 NUFFT against 11.80 via the uniform-grid FFT. This pins the S4.5
   claim that the chirp is an analysis artifact.
9. **Instrument envelope against the sky.** Pin the S4.6 table: the 50%
   depth of the real zoom response at 50 / 30 / 10 MHz (2830 / 613 / 24
   rad/m^2) and of the parent response (58.7 / 13.3 / 2.7), against the
   `faraday2020v2` percentiles (p50 18.4, p90 91.0, p99 278.0, p99.9 648.8,
   max 2442.1). Assert the orderings the paper's claims rest on: zoom@50 >
   map max; zoom@30 within a factor 1.5 of p99.9; zoom@10 < p50; every parent
   envelope < p90. Loose tolerances -- this pins the *conclusions*, not the
   digits.
10. **Foreground sidelobe budget** (S4.8). Inject a `phi = 0` leakage
   foreground at |P|/I = 0.15 with the window actually used, and measure the
   power it puts into the roll-off region. Compare against both ends of the
   S4.4 bracket. This is a reported budget, not a pass/fail, but a failure at
   the optimistic end would be disqualifying.
11. **Two-arm template agreement** (S4.9). Build `H(phi)` with the four-port
   and two-port weights and report the roll-off shift between them. A reported
   number, not a pass/fail -- if the arms disagree materially that is a result,
   not a bug.
12. **Noise threshold reproduces the closed form.** `SNR = sqrt(N) (A/sigma)^2`
   against a Monte Carlo with `noise.add_noise` at a few amplitudes, and
   `sigma_frac = 1/sqrt(dnu tau)` against `radiometer_sigma` on a
   sky-dominated `T_sys`.
13. **Zoom aliasing.** Inject a depth whose signature folds under the 25 kHz
   critical sampling and confirm the template accounts for the image rather
   than silently absorbing it (S4.6 item 3).

## 7. Sequencing

1. `dispersion.py` with the analytic tests only (S6.3, S6.7, S6.8). No maps.
2. The two invariance gates (S6.1, S6.2) on the real RM map. **Stop point** --
   if either fails, the design is refuted and nothing downstream is worth
   building.
3. The geometry knob over `k`; the plane-taper check (S6.5); the
   converged-regime check (S6.6).
4. Beam weights from the PR #3 windows; per-channel-pair templates; LST axis.
5. The real spectrometer RMSF and the aliasing check (S4.6, S6.9).
6. The amplitude bracket and the three-band section (S4.5).
7. Scripts, figures, report surgery.

## 8. Out of scope

- **3D Galactic modelling.** hammurabi X, NE2001/YMW16, JF12 / Unger-Farrar,
  pulsar RM-DM growth curves, LOFAR low-`phi` anchoring. These predict the
  *amplitude*; they are the follow-up paper and get one paragraph in the
  discussion as future work.
- **Any amplitude claim.** Bracket only.
- **The full Fisher port.** PR #2's `fisher.py` / `detection.py` are built
  against the deleted `sim.py` and stay out; the sky-marginalised machinery is
  not needed for a threshold curve. `noise.py` is the exception and comes in
  (S4.10).
- **The RM-synthesis branch** (`faraday-rmsynth`, PR #1).
- **The audit's open item** -- reconfirming through the real BGL_v16 kernel
  rather than the `cos^2` test lobe. ~30 minutes, independent, does not block.
- **Editing the paper source** outside this repo.

## 9. Risks

- **The shape may not be as invariant as the audit's 1.038 suggests.** The
  incoherent limit is exact only when the beam contains many independent
  patches; near the roll-off it contains few. Gate 6.1 measures this rather
  than assuming it.
- **`sigma_eff` is a parameter, not a measurement.** Both tables in S4.5 hang on
  it. Show the spread; do not quote a single number.
- **The mixed-medium re-weighting shifts `F` toward low `|phi|`**, which
  suppresses the very signature the paper is about. The uniform-slab fiducial
  must not be presented as more constrained than it is -- it is a midpoint
  between two brackets, not a measurement.
- **Field reversals break the linear ansatz.** `phi(l)` is not monotonic, so
   partial depths can exceed `|phi_col|`. The direction is favourable -- `F`
   broadens rather than shrinks -- but it means the template is narrower than
   truth and the roll-off is a lower bound. S4.2.1 bounds the exposure; do not
   let the text drift back to "the roll-off is at `max phi_col`".
- **Zoom folding is assumed handled downstream.** `channelization.py` says the
   folding "is physical and is removed downstream, not here". That path was
   built for a smooth-spectrum waterfall, not for a template whose whole content
   is fine-frequency structure. Check what it actually does before trusting it;
   if it does not survive contact with a Faraday signature, that is a real
   scope addition and should surface at gate 6.9, not in the write-up.
- **The 30 MHz roll-off is deconvolution-dependent.** Instrument and sky roll
   off at the same depth there (613 vs p99.9 = 648.8) and zoom recovery is
   0.7947. If the deconvolution does not hold up, 30 MHz loses the roll-off and
   contributes only the low-`phi` core. Gate 6.9 surfaces this early; the paper
   must not present a 30 MHz roll-off without the deconvolution error budget.
- **The diffuse leakage floor is ~1000x the signal and the polarimeter does not
   touch it.** Separation rests entirely on the delay axis and therefore on
   window sidelobes (S4.8). At the pessimistic end of the amplitude bracket BH4
   is insufficient. This is the most likely route by which the whole search
   turns out not to be feasible, and gate 6.10 is where it shows up.
- **The detection threshold sits at the optimistic end of the amplitude
   bracket** (S4.10). If the truth is near the mixed-medium floors, LuSEE does
   not detect diffuse Faraday at any integration and the paper is a template
   plus an upper limit. That is still publishable and should be planned for
   rather than discovered late.
- **Reviewer objection**: "you refuted your own Step 4 and are now publishing its
  shape." The answer is S1 and gates 6.1/6.2, and it needs to be in the text,
  not just in this spec.
