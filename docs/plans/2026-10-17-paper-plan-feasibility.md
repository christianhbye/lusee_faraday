# Feasibility of the plan to the paper, for the implementer

Status: assessment, 2026-10-17. Companion to the
[plan to the paper](2026-10-17-paper-framing.md#plan-to-the-paper). For each
phase: what exists, what has to be built, what to watch, and what "done"
looks like. Numbers quoted here come from the
[pilot record](../faraday-redistribution-pilot-results.json) and from the
arithmetic in the [review findings](2026-10-17-paper-framing.md#review-findings--2026-10-17);
regenerate them from a script before they enter the paper.

Nothing in the plan needs a new numerical engine or a long computation.
The dominant risk is scientific: the Faraday structure inside the horizon is
unmeasured, so every amplitude is a bracket. Build each phase so the scaling
law and the signature are the result and the amplitude is the conditional.

## Phase 1 — horizon and depth depolarization

**Exists.** `gaussian_screen.qu_moments` gives the second moments of a
Gaussian screen in closed form on a flat plane ([spec](../specs/2026-10-15-gaussian-screen-step1.md)).
The [literature note](../literature/low-frequency-diffuse-polarization.md)
already states that a uniform slab is the average of foreground screens at
depth fractions f in [0, 1].

**Build.** A horizon table with citations, and a `depth_depolarization.py`
that adds the (f, f') average to the screen second moment, for a uniform
slab, a Gaussian depth profile, and a screen at f = 0, 0.6 and 1 of the
column. Split the result into the smooth phi ≈ 0 component and the speckle
component and report both at 10/30/50 MHz.

**Watch.**

- The horizon is a range, not a number. Local Bubble and Local Interstellar
  Cloud electron densities and fields are published with factor-of-several
  spreads, and 0.81 n_e B_par spans at least 0.004–0.2 rad/m²/pc. Quote the
  range and the emissivity assumed inside it.
- The mixed-emission second moment for Gaussian phi is
  exp(-2 lambda⁴ [f² + f'²] sigma² + 4 lambda⁴ f f' K(theta)); it is a
  two-dimensional integral over (f, f') added to the existing separation
  integral, with no sky quadrature. If a grid-dependent amplitude appears,
  the implementation is wrong, not the model.
- Phi statistics inside the horizon (1c) cannot be measured from any map.
  Nearby pulsar rotation measures and turbulence scaling give a prior range
  only. The floor's lambda^(-8/alpha) scaling does not depend on it; the
  amplitude does. Say so in the result record.
- Do not reuse the Hutschenreuter column as sigma_phi for the local volume.
  It is an upper reference, and its 1–8° structure function is set by
  source density and inference smoothing, so even that is a lower bound on
  the column.

**Done when** f = 0 reproduces the pilot's surviving fraction, and the
record lists smooth and speckle amplitudes at three frequencies for the
stated horizon range. Effort: one to two weeks.

## Phase 2 — redistribution with the real instrument

**Exists.** `response.four_port_pair_alms` returns pair-Stokes alms
(pair, block, l, m) for I, P_MINUS and P_PLUS at lmax 96; the pilot's
`window` stage already computes them at 10/30/50 MHz under a 24 GB cap.
`channelization.integrate` applies the parent and zoom responses on the
fine grid. The pilot's fine theta grid resolves the coherence angle
(theta_c ~ 4.6e-5 rad at 30 MHz baseline).

**Build.**

- 2a: replace the scalar Legendre transform by the spin-2 (Wigner-d)
  transform of the rotated field's correlation, and contract E/B spectra
  with all ten pair kernels including the P_MINUS/P_PLUS cross terms.
- 2b: the cross-frequency correlation
  <Pi(n, lambda) Pi*(n', lambda')> for the Phase 1 sigma_phi range, one
  transform per frequency pair, channelized through parent and zoom.
- 2c: the figure.

**Watch.**

- A sky realization at l ~ 2e4 cannot be drawn and is not needed. The
  beam-weighted visibility is a Gaussian variable with a known covariance
  over time and frequency; simulate the fixed-pattern speckle as a draw
  from that covariance. The time axis is the beam overlap between sky
  positions, which is cheap.
- Only frequency pairs within a few coherence lengths are needed. The
  coherence is 1/(sqrt 2 sigma_phi) in lambda², i.e. 11 kHz at 30 MHz for
  sigma_phi = 10 rad/m² and ten times wider at sigma_phi = 1. Do not
  compute the full 16,384-channel pair grid; a few tens of thousands of
  transforms is about an hour.
- Checks that can fail: the scalar pilot must be the high-l limit of the
  auto pairs; the cross terms must fall as exp(-8 sigma_phi² lambda⁴); the
  spherical real-beam covariance in its small-beam limit must reproduce the
  flat-plane PR 7–8 moments in their validated 45–50 and 20–35 MHz windows.
- Gaussianity of phi matters most for the cross-frequency covariance. Rerun
  the PR 8 matched-covariance scale-mixture test on it before quoting a
  coherence scale.
- Panel (c)'s spectrum needs a polarized synchrotron law with the turnover
  below 20 MHz; beta = -2.8 to 10 MHz is illustrative and must say so.

**Done when** the three checks above pass and the figure is regenerated
with the real beam and Phase 1 amplitudes. Effort: two to three weeks.

## Phase 3 — separability

**Exists.** PR 3 leakage kernels and the parent-bin Q/I = 0.04 result. Three
response variants (as-built, `_c4sym`, `_diagza`) for the ablations.

**Build.** A leakage residual after a smooth fit per parent and zoom window;
a receiver-noise model at the loaded reference plane; the decision number
(speckle over leakage residual, speckle over noise) at 30 and 50 MHz for
the Phase 1 range.

**Watch.**

- There is no beam-uncertainty model in the repository. The three response
  variants are the only available spread; report the residual for each and
  call it a spread, not an error bar.
- The BGL response has native channels of about 0.3 MHz, so leakage is
  smooth within a parent channel by interpolation, not by measurement.
  State this; it bounds what any sub-channel discriminant can claim.
- Receiver noise is not in `instrument.py` (it carries thermal loading
  terms only, and `T_moon = T_ant = 0` is required for a Faraday run). Take
  the amplifier noise from luseepy's receiver model or rebuild it at the
  four-port loaded reference plane. Default-zero amplifier noise is not a
  sky-domination check at 50 MHz.
- The pilot's column screen gives speckle Q/I ~ 6e-5 against leakage 0.04.
  A local screen with sigma_phi ~ 1 and weaker small-scale structure raises
  the floor by thousands in power, bringing the speckle to the ten percent
  beam-error level. Expect the horizon range to swing the answer by orders
  of magnitude; the paper must be written so that a bound is an ending.
- Zoom bins separate the speckle from leakage only when sigma_phi is large
  enough that the coherence (11 kHz at 30 MHz for sigma_phi = 10) is
  shorter than the 25 kHz window. At sigma_phi = 1 it is smooth across the
  window and degenerate with leakage. Test the spatial signature as well:
  a flat polarized C_l over l = 2–36 in the Camacho et al. (2026) maps
  against the steep intrinsic spectrum. Section 6 should report both.
- The speckle and the leakage are both fixed patterns repeating with the
  sidereal day; only noise integrates down. The decisive comparison is
  therefore speckle against leakage residual, not against noise.

**Done when** the decision number is tabulated for both channel types,
both frequencies, and the full horizon range, with the spread from the
three beams. Effort: two weeks, plus a week if the noise model is rebuilt.

## Phase 4 — interpretation and framing statements

**Build.** The lunar nightside plasma Faraday rotation with one citation
and one line of arithmetic (it is many orders of magnitude below the
roughly 1 rad/m² of Earth's ionosphere); the inversion of the floor scaling
to D(theta) inside the horizon, with the depth-depolarization degeneracy
stated alongside.

**Watch.** Amplitude alone cannot separate theta_c² from the
depth-depolarization factor. Present the inversion as a constraint on their
product unless Phase 3 gives a spectral or spatial discriminant.

**Done when** both statements have a citation and a regenerable number.
Effort: one to two weeks, in parallel with Phase 3.

## Phase 5 — manuscript

**Exists.** The `latex-memo` provenance rules and the report build in
`report/`.

**Build.** The paper source (absent locally; the historical external path
does not exist on this machine, so obtain the collaboration template), the
five figures, and the transfer of PR 3 instrument and leakage material.

**Watch.**

- Every figure and quoted number regenerates from a script and a result
  record; nothing is typed in.
- Keep the pixel-sum audit out; cite the `audit-2026-08-18` tag. Keep PR 4's
  detection forecast out unless it is reframed as a conditional
  illustration.

Effort: two to three weeks of writing once Phases 1–3 are done. Total to a
draft: about two to three months at a comfortable pace.
