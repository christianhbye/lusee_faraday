# Paper framing: how Faraday rotation shows up in LuSEE Night spectra

Status: adopted 2026-10-17 after review (merged to `main`). The phase order
below is superseded by [the paper arc of 2026-10-19](2026-10-19-paper-arc.md),
which presents the column screen, the emitting depth and the local medium as
a nested bracket and adds the missing column-screen scenario; the physics
here stands. Supersedes the detection-forecast outline in
[paper-preparation.md](../paper-preparation.md#paper-structure-proposed-for-review).
The pilot below is a toy calculation, not a Galactic prediction. The
[review findings](#review-findings--2026-10-17) and the
[plan to the paper](#plan-to-the-paper) at the end are the current work plan;
[TODO.md](../../TODO.md) points here.

## Purpose

The collaboration wants a paper stating that LuSEE Night is aware of Faraday
rotation and showing how it manifests in the spectra below 50 MHz. LuSEE
Night is the first radio telescope on the lunar far side, observes a band
inaccessible from the ground for decades, and sees no ionospheric Faraday
rotation. The paper reports mechanisms and their magnitudes for this
instrument; it does not promise a detection.

The original question — can the Q/U rotation be resolved spectrally — has a
clear answer: the channels resolve it along any single line of sight, and the
beam destroys it. What survives is set by small angular scales. That result
is the spine of the paper.

## Mechanisms, with LuSEE numbers

**Channel depolarization is not the limit.** The Faraday depth at which a
channel's sinc response reaches its first zero is pi/Delta(lambda^2):

| Frequency | Parent (25 kHz) | Zoom (25/64 kHz) |
| --- | --- | --- |
| 10 MHz | 0.7 rad/m² | 45 rad/m² |
| 20 MHz | 5.6 rad/m² | 360 rad/m² |
| 30 MHz | 19 rad/m² | 1200 rad/m² |
| 50 MHz | 87 rad/m² | 5600 rad/m² |

The median |RM| of the Hutschenreuter et al. (2022) map is 18 rad/m², so a
single line of sight is resolved in parent channels above ~30 MHz and in zoom
bins down to ~5 MHz.

**Beam depolarization redistributes power; it does not destroy it.** For a
foreground screen the rotated field Pi = P0 exp(2i phi lambda^2) satisfies
|Pi| = |P0| everywhere, for any phi. Total polarized power,
sum_l (2l+1) C_l / 4pi, is therefore independent of frequency: rotation moves
power between multipoles. Once the coherence angle theta_c, defined by
2 lambda^4 D(theta_c) = 1 for structure function D, falls below the beam
scale, the power sits at l ~ 1/theta_c. Below that the spectrum is white
with level

  C_0 = 2pi |P0|² ∫ theta dtheta <exp(2i lambda² Delta phi(theta))> ∝ theta_c².

The beam passes only low l, so the measured polarized power is C_0 times a
beam factor. **What LuSEE measures at strong rotation is the small-scale
coherence of phi.** For D ∝ theta^alpha at small separations, the floor
scales as lambda^(-8/alpha).

**Depth depolarization destroys power.** When emission is mixed with the
rotating gas, |P| itself drops along each sightline: as lambda^-2 for a
uniform Burn slab and as exp(-sigma² lambda⁴) for a Gaussian Faraday-depth
profile (Burn 1966). LoTSS finds the diffuse emission at ~0.6 of the column
RM, between a full-column screen and a uniform slab
([literature check](../literature/low-frequency-diffuse-polarization.md)).
The observed floor amplitude is (|P|² after depth depolarization) × theta_c²;
amplitude alone cannot separate the two.

**Instrumental leakage I -> Q,U** is deterministic and dominant (parent-bin
Q/I = 0.04, PR 3). It is spectrally smooth; the surviving Faraday signal is
not (next section).

## What survives, and how it would look

The beam-weighted sum over N ~ Omega_beam/theta_c² cells with random phases
is a speckle: a fixed pattern on the sky, repeating with sky rotation. It has
two observables:

- its **amplitude**, ∝ theta_c², constraining the small-scale structure
  function;
- its **frequency coherence**, set by the phi spread within the beam through
  exp[-2 sigma_phi² (lambda² - lambda'²)²], which gives the channel-channel
  covariance a power-law sky lacks. That covariance depends on lambda²lambda'²
  as well as on lambda² - lambda'², so it is not stationary in frequency.

Leakage and depth-surviving emission near phi ≈ 0 are spectrally smooth and
mutually degenerate; the speckle is not. Whether LuSEE can separate it from
the leakage residual at the required level is the open quantitative question.

## Pilot figure

`scripts/faraday_redistribution.py` (stages `window`, `plot`) produces
`report/figures/faraday_redistribution_pilot.pdf` and
[the result record](../faraday-redistribution-pilot-results.json).
The figure and beam-window cache are untracked, regenerable outputs.

Model: Gaussian foreground screen with saturation rms sigma_phi = 10 rad/m²
and D(theta) = 2 sigma² [1 - exp(-(d/theta0)^alpha)], d the chordal
distance (positive definite for alpha <= 2). theta0 is set by the rms phi
difference at 1°, Delta phi_1° ∈ {0.3, 1, 3} rad/m², with alpha ∈ {1, 5/3,
2}. Intrinsic P0 is statistically isotropic with exponential correlation of
scale 20°, independent of phi. Pi is treated as a scalar complex field.

- **(a)** Angular power of the rotated field, per ln l, at 1400–10 MHz for
  alpha = 5/3, Delta phi_1° = 1. The area is conserved (sum = 0.9990–0.9999
  of |P0|²); exact Legendre and flat-sky Hankel transforms agree to 8e-7 at
  l = 3000. At 30 MHz the hump sits at l ~ 2e4 and C_l is flat to 1e-4
  between l = 2 and 30.
- **Beam window** W_l of the as-built BGL_v16 response (N-port auto,
  P_MINUS + P_PLUS duals, lmax 96). Polarization: 99% of the window by
  l = 26 / 36 / 14 at 10 / 30 / 50 MHz, tail above l = 80 below 1e-3.
  Intensity: 99% by l = 6–10.
- **(b)** Beam-weighted power relative to the unrotated sky. At 30 MHz:
  9e-8 for the baseline; 6e-11 to 2e-6 across the variants.
- **(c)** The same multiplied by an intrinsic polarized spectrum
  nu^(2 beta), beta = -2.8. Power rises toward low frequency when
  2|beta| > 8/alpha (alpha = 5/3, 2) and falls for alpha = 1. Whether the
  toy's spectral minimum exists is decided by alpha and beta.

Anchor: the Hutschenreuter posterior mean gives rms |Delta phi| at 1° of 2.4
rad/m² (|b| > 60°) and 4.4 rad/m² (|b| > 30°), medians 1.2 and 1.8. Between
1° and 8° at |b| > 60° the implied alpha is 1.2; below 1° the map's
resolution steepens it. This is the column, not the emission-weighted depth,
and says nothing below the map's resolution.

## Answers to the two questions raised

**Is a Gaussian phi with a structure function justified?** Not as a
description of the Galaxy: the Hutschenreuter model is exp(rho)·chi, and ISM
turbulence is intermittent. The main claims do not need it:

1. Power conservation under a foreground screen is exact for any phi.
2. The same-frequency correlation is P0P0'* times the characteristic
   function of the increment Delta phi(theta) at 2 lambda². If increments
   are self-similar at small separations, Delta phi(theta) = theta^(alpha/2) X
   in distribution, substituting u = lambda^(4/alpha) theta gives a floor
   ∝ lambda^(-8/alpha) for **any** distribution of X. Gaussianity sets only
   the prefactor, provided ∫ u du chi_X(2u^(alpha/2)) converges; patches of
   exactly uniform phi (an atom of X at zero) would break this and raise the
   floor.
3. Cross-frequency covariance needs the joint distribution of phi and its
   increments, not only increments. Gaussianity matters most there.

**Does LuSEE only pass l <= a few?** No; that was too strong. The
polarization window reaches l ~ 15–35 (99%), wider than the intensity window
(l ~ 6–10). Camacho et al. (2026) map the sky below 50 MHz at ~5°
resolution, l ~ 36. All of these are far below l_c ~ 1/theta_c ~ 2e4 at
30 MHz in the baseline pilot, so the beam and any LuSEE map sit on the white
floor. Map-making helps: ~(l_max+1)² modes sample the same C_0, and a flat
polarized C_l over l = 2–36 is a spatial signature distinct from the steep
intrinsic spectrum. Leakage and map noise must be shown not to mimic it.

## Proposed paper sections

| Section | Content | Source |
| --- | --- | --- |
| 1. Measurement | LuSEE Night, far side, no ionosphere, the band | — |
| 2. Instrument | four-port covariance, frozen beam, polarimeter, parent/zoom channels | PR 3 |
| 3. Leakage | I -> Q,U, magnitude and spectral smoothness | PR 3 |
| 4. Depolarization mechanisms | channel (table), beam (redistribution), depth (slab/screen) | this doc, lit check |
| 5. What survives | floor ∝ theta_c², speckle, frequency covariance; pilot figure with the real beam | pilot, PRs 7–8 |
| 6. Separability | speckle vs leakage residual and noise in parent/zoom channels | **new** |
| 7. Interpretation | what an amplitude or bound says about small-scale magneto-ionic structure | — |

The pixel-sum numerical audit is likely not a paper section. The
depth-template detection forecast (PR 4) is at most a conditional
illustration.

## Review findings — 2026-10-17

The framing and the pilot were reviewed before merging. The pilot's numbers
hang together: the recorded 30/50 MHz surviving-power ratios match
(0.6)^(8/alpha) to within 2% for all three alphas, the channel table was
recomputed, and the floor estimate agrees with a hand calculation to a
factor of 1.5. Three findings change the order of the work.

**1. The screen in the pilot is not the screen LuSEE looks through.** The
pilot rotates the whole polarized sky through a screen with the column's
structure function. Depth depolarization confines the polarized emission at
these frequencies to the volume within which the Faraday depth spread is
about 1/lambda²:

| Frequency | depth spread that depolarizes | horizon at 0.004 rad/m²/pc (Local Bubble, n_e ~ 0.005 cm⁻³, B ~ 1 µG) | at 0.04 rad/m²/pc |
| --- | --- | --- | --- |
| 10 MHz | 0.001 rad/m² | 0.3 pc | 0.03 pc |
| 30 MHz | 0.010 rad/m² | 2.5 pc | 0.25 pc |
| 50 MHz | 0.028 rad/m² | 7 pc | 0.7 pc |

The polarized sky seen below 50 MHz comes from parsecs, not the kiloparsec
column. The phi field whose small-scale coherence sets the floor is the
structure inside that horizon. Its rms and angular coherence are unknown and
almost certainly far below the column values, which moves theta_c and the
floor up by an unknown factor. Every amplitude in this document is therefore
conditional on a horizon model. The sentence "what LuSEE measures at strong
rotation is the small-scale coherence of phi" applies to the local medium.

**2. Inside the horizon the emission is, by construction, not strongly
rotated.** The surviving signal has a spatially coherent, spectrally smooth
part at phi ≈ 0 (the near edge of the emitting volume) plus a speckle from
the mixed-in remainder. The screen pilot cannot set the speckle amplitude; a
slab or Gaussian-depth profile can. That is why depth depolarization now
precedes the spin-2 pair-kernel work.

**3. The speckle's frequency coherence decides what parent channels see.**
With 2 sigma_phi² Delta(lambda²)² = 1 and the pilot's sigma_phi = 10 rad/m²:

| Frequency | speckle coherence | parent channel | zoom bin |
| --- | --- | --- | --- |
| 10 MHz | 0.4 kHz | 25 kHz | 0.39 kHz |
| 30 MHz | 11 kHz | 25 kHz | 0.39 kHz |
| 50 MHz | 49 kHz | 25 kHz | 0.39 kHz |

At 30 MHz and below, parent channels average over the speckle and adjacent
channels are nearly independent: the manifestation is a fixed-pattern excess
variance that does not integrate down with time and repeats with the
sidereal day, not a spectral feature. Zoom bins resolve it everywhere. The
coherence scales as 1/sigma_phi, so a horizon-limited sigma_phi of 1 rad/m²
makes it 10× wider. This belongs in Section 5.

Smaller points, to carry into the text:

- The pilot already fixes a leakage ratio. Beam-weighted power 1e-7 of the
  unrotated value is an amplitude of 3e-4; with an intrinsic polarized
  fraction of order 0.2 that is Q/I ~ 6e-5 against leakage Q/I = 0.04.
  Section 6 asks whether leakage can be modelled to one part in a thousand.
- The Hutschenreuter structure function at 1–8° is shaped by source density
  and the inference smoothing, not only below 1°, and the posterior mean is
  smoother than any realization, so the rms is a lower bound on the column.
- Panel (c)'s rise toward low frequency assumes beta = -2.8 for polarized
  brightness down to 10 MHz; free-free absorption and the spectral turnover
  below 20 MHz flatten it. Label it illustrative.
- The beam window uses only the N-port auto pair; the response digest hashes
  the first 16 MB of the FITS file (identity, not content).

## Plan to the paper

Each phase names its paper section, its deliverable, and the check that
could come out the other way. Phases 1–3 are sequential; 4 and 5 can start
in parallel with 3. Nothing here launches the old long computations. The
[feasibility assessment](2026-10-17-paper-plan-feasibility.md) says, per
phase, what exists, what to build, what to watch and the expected effort.

### Phase 1 — Faraday horizon and depth depolarization (Section 4)

- [ ] 1a. Horizon table. Source n_e and B_par for the Local Interstellar
  Cloud and Local Bubble (cite), compute the horizon depth at 10/30/50 MHz
  as a range, and state the emissivity assumed inside it. Deliverable: the
  table above with references, in `docs/literature/`.
  Check: a horizon longer than ~100 pc at 30 MHz would restore the column
  screen and void finding 1.
- [ ] 1b. Depth-depolarization bracket. Extend the step-1 correlation
  (PR 7) to emission mixed with the rotating medium: uniform slab and
  Gaussian depth profile, screen at f·Phi for f ∈ {0, 0.6, 1} per the
  LoTSS finding. Separate the smooth phi ≈ 0 component from the speckle
  component and give both amplitudes at 10/30/50 MHz relative to the
  unrotated sky and to leakage. Deliverable: `scripts/depth_depolarization.py`,
  a result record and one figure. Check: at f = 0 the result must reduce to
  the pilot; a speckle amplitude that depends on the quadrature grid fails.
- [ ] 1c. Local phi statistics. Replace the column anchor with a stated range
  for sigma_phi and D(theta) inside the horizon (from 1a and the 150–190 MHz
  literature). Keep the Hutschenreuter column as the upper reference only.

### Phase 2 — Redistribution with the real instrument (Section 5)

- [ ] 2a. Spin-2 rotated field through all ten pair kernels, with the
  P_MINUS/P_PLUS cross terms, replacing the scalar N-port pilot. Check: the
  scalar pilot must be recovered as the high-l limit of the auto pairs, and
  the cross terms must vanish at strong rotation as exp(-8 sigma_phi² lambda⁴).
- [ ] 2b. Frequency covariance and channelization. Compute the speckle
  channel–channel covariance on the fine grid through `channelization.py`
  (parent and zoom) for the Phase 1 sigma_phi range. Deliverable: the
  coherence table with the real channels, and the parent-channel variance
  excess as a fixed pattern. Check: pin against the validated Gaussian-screen
  covariance (PRs 7–8) in its 45–50 and 20–35 MHz windows.
- [ ] 2c. Redraw the pilot figure with the real beam, the spin-2 field, the
  Phase 1 amplitudes, and a polarized spectrum that includes the turnover.
  Panel (c) stays illustrative and says so.

### Phase 3 — Separability (Section 6; decides how the paper ends)

- [ ] 3a. Leakage residual. Model I → Q,U leakage at 30 and 50 MHz with the
  PR 3 kernels, with a stated beam-model uncertainty, and give the residual
  after a smooth fit in parent and zoom channels.
- [ ] 3b. Noise. Radiometer noise at the loaded reference plane with receiver
  loading and beam-weighted antenna temperature, for 1 and 24 lunations.
  Default-zero amplifier noise is not acceptable at 50 MHz.
- [ ] 3c. The decision number. Speckle amplitude over leakage residual, and
  over noise, at 30 and 50 MHz, parent and zoom, for the Phase 1 range. The
  check that could come out the other way: if the speckle sits below both
  across the whole range, Section 6 reports a bound and Section 7 says what
  the bound means. Do not promote a conditional illustration to a forecast.

### Phase 4 — Interpretation and framing statements (Sections 1, 7)

- [ ] 4a. Lunar nightside plasma: cite the electron density and bound the
  surface-layer Faraday rotation for the no-ionosphere statement.
- [ ] 4b. What an amplitude or bound says about local magneto-ionic
  structure: the floor ∝ theta_c² ∝ lambda^(-8/alpha) mapped to D(theta)
  inside the horizon, with the degeneracy against depth depolarization
  stated.

### Phase 5 — Manuscript

- [ ] 5a. Locate or create the paper source (the historical external path
  is absent here). Build it under the `latex-memo` provenance rules: every
  figure and quoted number regenerates from a script and a result record.
- [ ] 5b. Figures: instrument and leakage (PR 3); channel table; the Phase 2c
  redistribution figure; the Phase 2b covariance figure; the Phase 3c
  separability figure.
- [ ] 5c. Transfer the instrument and point-source material from PR 3 and
  the measurement model; audit references, figures and quoted values.

### Dropped or demoted by this plan

- The pixel-sum audit is not a paper section; cite the `audit-2026-08-18`
  tag.
- The depth-template detection forecast (PR 4) and the template-report
  corrections are at most a conditional illustration; they are not a
  prerequisite for any phase.
- The real-beam audit ladder and the A1/A2/B systematics scans in
  [TODO.md](../../TODO.md) are deferred; none feeds Sections 4–7.
- The Gaussian-screen benchmark (PRs 7–8) is reused as the validation
  reference for Phase 2b, not as a section of its own.

## References

- Burn, B. J. 1966, MNRAS, 133, 67.
- Camacho et al. 2026, Open J. Astrophys., "Linear map-making with
  LuSEE-Night", [arXiv:2508.16773](https://arxiv.org/abs/2508.16773)
  (abstract only: ~5° maps below 50 MHz).
- Hutschenreuter, S., et al. 2022, A&A, 657, A43.
- Literature on 150–190 MHz observations:
  [low-frequency-diffuse-polarization.md](../literature/low-frequency-diffuse-polarization.md).
