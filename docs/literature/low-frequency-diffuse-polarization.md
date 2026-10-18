# Diffuse Galactic polarization at 150–190 MHz: literature check

Status: literature check, 2026-10-17. No calculation in this repository uses
these numbers yet. Each entry says where its numbers were read; re-read the
cited section before quoting any of them in the paper.

## Question

The Gaussian-screen benchmark rotates a deterministic polarized sky through a
single foreground screen. The
[Galactic-model plan](../plans/2026-10-16-galactic-residual-screen-models.md)
proposes taking that screen from the Hutschenreuter et al. (2022) RM map,
phi = phi_baseline + delta_phi. This check asks what low-frequency
observations say about (a) where along the column the polarized emission
sits, (b) how bright it is, and (c) which Faraday depths and angular scales
they constrain. Those determine whether a full-column foreground screen,
emission mixed with the rotating medium, or neither has observational support.

At LuSEE frequencies a 1-rad phase change corresponds to
Delta phi = 1/(2 lambda^2) = 0.014, 0.005 and 0.002 rad/m² at 50, 30 and
20 MHz. The local copy of the Hutschenreuter map
(`data/faraday2020v2.hdf5`, mean and std only, nside 512) has posterior std
of 2.6 / 9.8 / 67 rad/m² at the 5th / 50th / 95th percentile.

## Observations

| Survey | Band | Area and scales | Faraday depth (rad/m²) | Polarized brightness | Distance | Read from |
| --- | --- | --- | --- | --- | --- | --- |
| LoTSS-DR2 I (Erceg+2022) | 120–167 MHz | 3100 deg², high-latitude outer Galaxy; scales ≲ 1° | mostly −25 to +15; RMSF FWHM 1.16, max scale 0.97 | — | — | full text, Sect. 5.2, Fig. 15, Table 2 |
| LoTSS-DR2 III (Erceg+2024) | 120–167 MHz | high-latitude inner Galaxy | large gradient tied to Local Bubble wall | — | ≥ 40–80 pc | abstract |
| LoTSS HETDEX (Van Eck+2019) | 120–167 MHz | 568 deg², 4.3′ | −7 to +25; resolution 1 | 1–6 K per RMSF | — | abstract |
| LOFAR IC 342 (Van Eck+2017) | 115–178 MHz | 5° × 5°, ~4′ | two features, separated by < 10 | up to 30 K | local neutral clouds | abstract; review |
| MWA SGP (Lenc+2016) | 154 MHz | 625 deg², 54′; structures 1–8° | −2 to +10, most near +1 | 4 K mean, 11 K peak | 51 ± 20 pc (pulsar J2330−2005) | abstract |
| MWA all-sky (Byrne+2022) | 167–198 MHz | ~11,000 deg², scales 1–9° | \|RM\| < 2 | band-averaged emission mostly unpolarized; strong structure near RA 0h | — | abstract and Sect. 1 (full text) |

## Findings

**1. The emission sits partway along the column.** Erceg et al. (2022,
Sect. 5.2) compare the LoTSS first Faraday moment M1 with the Hutschenreuter
et al. (2022) map, "representing the total RM produced by the Galaxy". The
histogram of RM_GFS / M1 peaks at 1.6 (FWHM 3.4) over the full mosaic; the
upper region peaks at 1.9 (FWHM 3.9), and the Loop III region is bimodal
(1.3, FWHM 1.3; −0.6, FWHM 8.5). A uniform Burn slab, with emission mixed
through the rotating medium, predicts 2; a foreground screen of the full
column predicts 1. The authors read the departure from 2 as "more complex
physical conditions" than a Burn slab. The scatter is large, but the
full-column screen is the worse description of the mean.

**2. The depolarization law depends on the Faraday-depth profile, not only
its width.** A Burn slab (top-hat profile) depolarizes as
sigma_phi^-1 lambda^-2; a Gaussian profile, as produced by a turbulent
foreground screen, as exp(−sigma_phi² lambda⁴) (Burn 1966; Van Eck 2018,
Sect. 2). Profiles with equal sigma_phi can therefore differ by orders of
magnitude in the 25× extrapolation in lambda² from 150 to 30 MHz.

**3. Gradients in phi move polarized power to small angular scales.** Spatial
variation of Faraday depth reduces polarized emission on the largest angular
scales (Schnitzeler et al. 2009, as summarized in Van Eck 2018, Sect. 2). This
is the lateral averaging the screen model represents; it suppresses exactly
the scales a wide-beam instrument measures.

**4. The detected emission is Faraday-thin and nearby.** Van Eck et al.
(2017) rule out Faraday-thick explanations for their features because those
would be "too strongly depolarized" at these frequencies. The surveys
therefore select for thin structures. Distances from pulsars and stellar
extinction place the structures at ≥ 40–80 pc, at the Local Bubble wall
(Erceg+2024), and at 51 ± 20 pc (Lenc+2016).

## What these observations do not constrain

- **Faraday depth below ~1 rad/m².** The best resolution is 1–1.16 rad/m²
  (LOFAR), and the largest Faraday-depth scale recoverable is 0.97 rad/m².
  LuSEE's surviving polarization depends on structure about 200× finer at
  30 MHz.
- **Angular scales above ~1–9°.** LoTSS loses scales above ~1° to missing
  short spacings; the MWA maps reach 8–9°. No single-dish or all-sky
  polarization measurement below ~180 MHz was found in this search.
  LuSEE's beam spans tens of degrees.
- **The nearest emission.** Thin-feature selection says nothing about the
  emissivity within the first few parsecs, where phi accumulates least. The
  depth over which emission stays within the LuSEE phase tolerance is
  L = Delta phi / (0.812 n_e B_par) pc, with n_e in cm⁻³ and B_par in µG.
  Local Interstellar Cloud values of n_e and B_par have not been sourced here.
- **Polarization fraction.** The brightnesses above have not been divided by
  total intensity in the same fields. That needs the Haslam/WMAP
  extrapolation used elsewhere in this repository, at matching resolution.

## Implications for the model choice (for discussion)

- A foreground screen at the full Hutschenreuter column contradicts the
  M1/RM ratio. Using that map as phi_baseline in a foreground screen is the
  least supported option.
- The ratio supports a data-anchored geometry: emission at a column fraction
  f ≈ 0.5–0.7 with large scatter. A uniform slab is the average of foreground
  screens at depths f·Phi for f in [0, 1], so it adds one 1-D average to the
  existing kernel and needs no 3D emissivity model.
- The extrapolation from 150 MHz to LuSEE frequencies is governed by
  sub-1 rad/m² structure and by angular scales above 9°, both unconstrained.
  Bracket it between the screen and slab laws rather than predict it.

## References

- Burn, B. J. 1966, MNRAS, 133, 67.
- Byrne, R., et al. 2022, MNRAS, 510, 2011.
  [arXiv:2107.11487](https://arxiv.org/abs/2107.11487)
- Erceg, A., et al. 2022, A&A, 663, A7. Faraday tomography of LoTSS-DR2 data
  I. [arXiv:2203.01351](https://arxiv.org/abs/2203.01351)
- Erceg, A., et al. 2024, A&A, accepted. Faraday tomography of LoTSS-DR2
  data III. [arXiv:2406.14679](https://arxiv.org/abs/2406.14679)
- Hutschenreuter, S., et al. 2022, A&A, 657, A43.
- Lenc, E., et al. 2016, ApJ, 830, 38.
  [arXiv:1607.05779](https://arxiv.org/abs/1607.05779)
- Schnitzeler, D. H. F. M., et al. 2009 (cited via Van Eck 2018; not read
  here).
- Turić, L., et al. 2021, A&A, 654, A5. Faraday depths −10 to +15 rad/m²
  around 3C 196; canals placed at the Local Bubble edge (abstract).
- Van Eck, C. L., et al. 2017, A&A, 597, A98.
  [doi:10.1051/0004-6361/201629707](https://doi.org/10.1051/0004-6361/201629707)
- Van Eck, C. L. 2018, Galaxies, 6, 112. Review of sub-GHz Faraday
  tomography. [arXiv:1810.09938](https://arxiv.org/abs/1810.09938)
- Van Eck, C. L., et al. 2019, A&A.
  [doi:10.1051/0004-6361/201834777](https://doi.org/10.1051/0004-6361/201834777),
  [arXiv:1902.00531](https://arxiv.org/abs/1902.00531)
