# Paper framing: how Faraday rotation shows up in LuSEE Night spectra

Status: proposal for discussion, 2026-10-17. Supersedes the detection-forecast
outline in [paper-preparation.md](../paper-preparation.md#paper-structure-proposed-for-review)
if adopted. The pilot below is a toy calculation, not a Galactic prediction.

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

## Next steps

1. Review this framing and the pilot figure.
2. Replace the scalar, isotropic pilot with the spin-2 rotated field through
   all ten pair kernels, including P_MINUS/P_PLUS cross terms.
3. Section 6: speckle amplitude and frequency covariance against the leakage
   residual and radiometer noise in parent and zoom channels, at 30 and 50
   MHz. This decides how the paper ends.
4. Add depth depolarization (slab average over screens at f·Phi) to bracket
   the amplitude.
5. Source the Local Interstellar Cloud n_e and B_par for the Faraday-horizon
   estimate, and lunar nightside plasma for the no-ionosphere statement.

## References

- Burn, B. J. 1966, MNRAS, 133, 67.
- Camacho et al. 2026, Open J. Astrophys., "Linear map-making with
  LuSEE-Night", [arXiv:2508.16773](https://arxiv.org/abs/2508.16773)
  (abstract only: ~5° maps below 50 MHz).
- Hutschenreuter, S., et al. 2022, A&A, 657, A43.
- Literature on 150–190 MHz observations:
  [low-frequency-diffuse-polarization.md](../literature/low-frequency-diffuse-polarization.md).
