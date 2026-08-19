# Refactoring lusee_faraday onto luseepy + croissant

Date: 2026-08-18
Status: approved design, ready for an implementation plan
Branch base: `croissant-crosscheck`

## 1. Why

The 2026-08-18 numerical audit (commits `4b401c5`, `afb3290`) established two
things that fix the requirements for this refactor:

1. The diffuse-sky Faraday signature in report Steps 2-4 is the shot noise of
   the HEALPix grid, not sky signal. Nyquist sampling of the Faraday phase
   needs `nside ~ 2.8e5` at 30 MHz, so the input does not determine the answer
   at any computable resolution.
2. Consequently **no simulation engine choice can rescue the diffuse regime**.
   For a band-limited beam the spherical-harmonic contraction and the pixel sum
   are the same HEALPix quadrature. Engine choice is not a physics question,
   and no bespoke pixel/NUFFT engine is needed.

What remains is a maintenance problem: the repo carries two independent
simulators, roughly 1,800 lines of custom machinery, duplicating instrument
physics that luseepy and croissant already own and test.

- Old two-port stack: `sky.py`, `beam.py`, `sim.py`, `fast_sim.py`,
  `healpix.py`, `rotations.py`, `spectrometer.py`, `utils.py`. Used by
  `notebooks/faraday_sims.py`, which produced the paper's current figures.
- Four-port pixel engine: `fourport.py` (520 lines), used by every script in
  `scripts/` and by `report/report.tex`.

Neither luseepy nor croissant contains any Faraday code (verified by grep), so
the Faraday operator and the sky model stay ours. Everything else should not.

The likely actual paper result is the `I -> Q,U` leakage: fully deterministic,
currently under-served, and unaffected by the audit. `PROGRESS.md` records
parent-bin `Q/I = 0.040` matching the I-only reference to 2e-4. Doing that
properly needs this refactor's machinery.

## 2. Decisions taken

| Question | Decision |
|---|---|
| Scope | Retire the old two-port stack. Keep `fourport.py` as an independent validation arm during the transition, then shrink it to what luseepy does not own. |
| Acceptance | Regress on what survived the audit. Do **not** chase bit-level agreement on the Steps 2-4 diffuse-Faraday numbers. |
| Scripts | Port the surviving analyses (step0/step1 point source, `step_ionly`, `zenith_weights`). Leave `step2_real_sky` / `step4_power_spectra` on the legacy arm, labelled, until path 2 replaces them analytically. |
| Architecture | Component-spectral sky feeding luseepy's four-port physics, plus a thin `croissant.PairStokesBeam` driver for the symmetric pseudo-dipole arm. |

## 3. Verified mechanisms

These were read in the installed sources, not assumed.

- `lusee.FullStokesSimulatorBase` accepts any sky object exposing
  `polarized_alm_at_freq(freqs, lmax)`, or a `croissant.PolarizedSky`
  (`luseepy/lusee/FullStokesSimulator.py:243`). That is the injection hook.
- croissant's internal dual is `(I, V, P-, P+)`, where `P-` is the spin -2
  analysis of `Q + iU` and `P+` the spin +2 analysis of `Q - iU`, both in IAU
  (`croissant/polarization.py:487`, `:467`).
  **Therefore Faraday rotation is exactly diagonal in the harmonic dual** for
  any sky region of constant `phi_FD`:
  `P-_alm -> P-_alm * e^{-2i phi lambda^2}` and `P+_alm` by the conjugate.
  (Sign per convention: the paper's `(Q + iU)_COSMO e^{+2i phi lambda^2}`
  becomes `(Q - iU)_IAU e^{+2i phi lambda^2}`. Getting this sign right is the
  single highest-risk item in the refactor and is unit-tested first.)
- `assemble_open_covariance`, `apply_receiver_loading`, `project_hermitian`
  and `pack_covariance` are module-level in `lusee.Covariance`, callable
  without going through `simulate()`.
- `croissant.PairStokesBeam` accepts arbitrary `pairs`, but
  `lusee._validate_instrument_metadata` hard-requires four ports and the ten
  ordered `a<=b` pairs (`FullStokesSimulator.py:136`). The symmetric
  pseudo-dipole arm therefore cannot go through luseepy's simulator and needs
  its own thin croissant driver.
- The `engine="auto"` dense-memory trap recorded from an earlier session is
  superseded: croissant `1c4d6c5` (newer than the `da01c5a` pinned in
  `SETUP-CROSSCHECK.md`) added a memory cap to `_low_pass_in_one_step`, so
  `PolarizedSky(nside=512).compute_alm(lmax=30)` takes the native transform and
  truncates rather than building an ~800 GB dense operator
  (`croissant/polarization.py:142`). **To be confirmed empirically in Phase 0.**

## 4. Architecture

### 4.1 Module layout

```
src/lusee_faraday/
  conventions.py    COSMO<->IAU, Faraday sign, port order, frames, pinned constants
  config.py         site, time grid, fine grid, band centers (absorbs scripts/common.py)
  sky.py            FaradaySky: component decomposition -> (component alms, coeff matrix)
  response.py       adapters: lusee.InstrumentResponse -> pair alms; 2-port FITS -> croissant.PairStokesBeam
  engine.py         contraction driver + spectral expansion
  instrument.py     luseepy covariance assembly (open covariance, loading, packing)
  polarimeter.py    zenith calibration + pseudo-Stokes
  channelization.py parent/zoom bins on lusee.spectrometer_response*, FFT ordering, transfer + deconvolve
  _legacy_pixel.py  today's fourport.py; validation arm only, never imported by production code
```

Removed with their tests: `beam.py`, `sim.py`, `fast_sim.py`, `healpix.py`,
`rotations.py`, `spectrometer.py`, `utils.py`, and today's `sky.py`. That is
roughly 1,100 lines, and it retires the `interp_hp` pole artifact and the
`healpy.Rotator` machinery along with them.

`plot.py` (21 lines) is reviewed at Phase 7 and kept only if a ported script
still uses it.

### 4.2 The core object: `FaradaySky`

The refactor's one genuinely new idea. Because the Faraday phase is diagonal in
the harmonic dual, a sky is not a frequency-stacked map cube but a small set of
frequency-independent spatial patterns plus a per-frequency coefficient matrix:

- `component_alms` -- shape `(K, 4, L, 2L-1)`, complex, in croissant's dual.
  One component per region of constant `phi_FD`; the `4` is croissant's dual
  block axis `(I, V, P-, P+)`.
- `coeffs(freqs)` -- shape `(K, nfreq, 4)`, complex. Per dual block, because
  within one region the `I` block carries the Stokes-I power law (`beta_I`)
  while the `P-`/`P+` blocks carry the polarized power law (`beta_QU`) times
  conjugate Faraday phases. One region therefore needs exactly one component,
  not one per Stokes parameter.

The visibility then separates exactly:

    V(t, p, nu) = sum_k sum_c coeff[k, nu, c] * W[k, c, t, p]

The fine 16,384-channel axis costs `K` contractions and one small einsum,
instead of 16,384 spherical transforms.

**The block axis must survive the contraction.** `croissant.polarized_convolve`
sums over the dual-block axis `c` (`einsum("fclm,tm,pfclm->tpf")`), which would
collapse blocks that need different coefficients. `engine.contract` therefore
performs the same contraction with `c` retained
(`einsum("kclm,tm,pclm->kctp")`), using `croissant.rotations` and
`croissant.simulator.rot_alm_z` unchanged for the Wigner rotation and the time
phases. This is the one place the refactor does not literally call a croissant
entry point, so it carries a dedicated test: summing our `c` axis must
reproduce `croissant.polarized_convolve` to machine precision.

Constructors, with the resulting component count:

- `FaradaySky.point_source(...)` -- discrete sources, each with its own
  `phi_FD`. Exact; `K = n_sources` (plus one if an unpolarized background with
  a different spectral index is included).
- `FaradaySky.uniform_screen(I, Q, U, phi)` -- constant `phi_FD` over the sky.
  Exact; `K = 1`.
- `FaradaySky.binned_screen(I, Q, U, rm_map, dphi)` -- piecewise-constant `phi`
  over `phi` bins. Exact per bin; `K = n_bins`.
- `FaradaySky.i_only(I)` -- frequency-flat leakage reference; `K = 1`.
- Pixelwise fallback, gated (see 4.3).

The object satisfies luseepy's `polarized_alm_at_freq` protocol, so it can also
be handed straight to `FullStokesCroSimulator` when the frequency axis is short
enough not to matter.

### 4.3 The audit as a guardrail

`FaradaySky.binned_screen` -- the only constructor that turns a map of Faraday
depths into components, and the one `from_rm_map` delegates to -- computes and
reports two diagnostics on every screen build, via `sky.audit_screen`. (As
implemented at first the check lived on `from_rm_map` alone, so calling
`binned_screen` directly bypassed it; that was closed in the final fix round.)
They are different criteria and both matter:

- **Spectral** -- components needed to resolve the Faraday phase across the
  simulated band: `dphi <~ pi / (2 * d(lambda^2))`. At 30 MHz over +-0.1 MHz,
  `d(lambda^2) ~ 1.33 m^2`, so `dphi <~ 1.2 rad/m^2`: about 25 components for
  an ionospheric screen (1-30 rad/m^2), about 4,000 for the full Galactic map.
  This governs cost.
- **Spatial** -- the audit's Nyquist criterion: the `nside` at which `phi` is
  resolved between adjacent pixels, against the `nside` actually used. This is
  the number that reads `2.8e5` for the real RM map at 30 MHz. This governs
  whether the answer means anything.

The pixelwise path raises unless `allow_pixelwise=True` is passed explicitly,
and the refusal message quotes both numbers with a pointer to the audit. A
build that succeeds logs the same two numbers at INFO, and one that proceeds
under `allow_pixelwise=True` despite failing a criterion logs a warning, so
"reported on every build" is true of successes and not only of raises. The
audit finding thus lives in the API rather than in a paragraph someone has to
remember.

### 4.4 Data flow

```
config -> times, fine freqs, band center
              |
 response.pair_alms(center, lmax)  ------+  fixed-beam convention enforced HERE,
   (one frequency, explicitly)           |  not implied by an interpolation default
                                         v
 FaradaySky -> component_alms -> engine.contract() -> W[k, c, t, pair]   K contractions
            -> coeffs(freqs)  -> engine.expand()   -> pair_integrals[t, f, pair]
                                         |            einsum("kctp,kfc->tfp"),
                                         |            chunked, memmap-backed
                          instrument.covariance()
                          (luseepy: open covariance -> loading -> hermitian -> pack)
                                         v
              16 real channels -> polarimeter -> I,Q,U,V -> channelization -> parent + 3x64 zoom
```

`Z_A` and `Z_L` are evaluated on the **fine** frequency grid (4x4 complex per
channel, about 4 MB at 16,384 channels), so receiver loading is not smeared by
the fixed-beam approximation. The fixed-beam approximation applies to the
response alms only, which is the pinned convention: only the Faraday phase is
chromatic, so all delay-space power is Faraday-induced.

### 4.5 The two drivers

- **As-built four-port** -- `lusee.InstrumentResponse` -> `pair_stokes_alms` at
  one frequency -> `engine` -> `lusee.Covariance` assembly. Full `Z_A` with
  mutual coupling, JFET receiver loading, `T_moon`/`T_ant`, blackbody
  normalization, 16 packed products: all luseepy's.
- **Symmetric pseudo-dipoles** -- 2-port FITS -> `croissant.PairStokesBeam`
  with `pairs = ((0,0), (0,1), (1,1))`, built from the Jones matrices as
  complex pair-IQUV maps -- the generalization of today's nine real
  `precompute_weights` patterns -- then through the same `engine.contract`.
  Unitless, no impedance, no loading. This is the lineage of the paper's Fig 4
  and of `scripts/compare_main_vs_asbuilt.py`'s `MainBeam`.

Both share `conventions.py`, `FaradaySky`, and `engine.contract`. Only the
response adapter and the post-contraction assembly differ.

## 5. Error handling

- All convention conversions funnel through `conventions.py`; a round-trip
  identity test guards them. Sign errors here are the highest risk in the
  refactor, so Phase 1 does nothing else.
- The PSD invariant `sqrt(Q^2 + U^2 + V^2) <= I` is kept as a runtime
  assertion, not just a test. It caught a real bug before: the `e^{-2i chi}`
  coefficient of a complex cross-pair kernel is `0.5 * (K_Q + i K_U)`, not the
  conjugate of the `e^{+2i chi}` one.
- The fixed-beam approximation is asserted rather than assumed: the requested
  fine band must lie inside one native response channel, else raise.
- `engine.expand` chunks over frequency under an explicit memory budget and
  writes memmap-backed output for full waterfalls.
- croissant's resolved engine per block is logged at construction, so a silent
  fall back to a dense transform is visible rather than an OOM kill later.

## 6. Testing

| Layer | Check |
|---|---|
| Unit | Data-free tests per module, in the style of today's `tests/test_fourport.py` |
| Analytic | Rank-1 point source; `phi_FD = 250` at 30 MHz gives the predicted 1.89 kHz Q oscillation period; zenith leakage nulls to ~1e-16 |
| Cross-engine | New stack vs `_legacy_pixel`, and vs `lusee.FullStokesCalibratorSimulator` (`scripts/validate_engine.py` already reaches 8.6e-16 there) |
| Library contract | Summing `engine.contract`'s dual-block axis reproduces `croissant.polarized_convolve` to machine precision |
| Regression | Parent-bin `Q/I = 0.040` matching I-only to 2e-4; unpolarized-source `p_leak = 0.134` at transit; zoom recovery 0.79 real / 0.86 ideal; per-band zenith weight vectors |

Explicitly **not** a regression target: the Steps 2-4 diffuse-Faraday numbers
(`|dP|/I ~ 1.7e-4` and the Step-4 delay spectrum). They are grid shot noise and
will not reproduce stably between two different quadrature engines. The spec
records this as expected behaviour, and the diffuse scripts stay on the legacy
arm so their outputs remain byte-reproducible there.

## 7. Sequencing

0. **Prep.** Branch off `croissant-crosscheck`. Verify the croissant worktree
   is present and pinned. Confirm the `_low_pass_in_one_step` claim in 3.
   **Freeze regression baselines out of the existing `generated_data/` before
   touching anything** -- the fine waterfalls are gitignored and expensive to
   regenerate, so the numbers to regress against get extracted to a small
   committed fixture first.
1. **`conventions.py`** plus its tests. Nothing else in this phase.
2. **`response.py` + `engine.py`** at a single frequency, cross-checked on a
   point source against `_legacy_pixel` and `FullStokesCalibratorSimulator`.
   *First real gate: if the contraction does not agree here, nothing later
   will.*
3. **`sky.py`** component decomposition and spectral expansion. Validate the
   Faraday phase analytically (known period at known `phi_FD`) and against the
   legacy arm's fine waterfall.
4. **`instrument.py` + `polarimeter.py`.** Reproduce the per-band zenith weight
   vectors and the leakage numbers.
5. **`channelization.py`** on `lusee.spectrometer_response` /
   `spectrometer_response_zoom`. Reproduce parent and zoom bin numbers and the
   zoom deconvolution.
6. **Port scripts** `step1_point_source`, `step1_ionly_source`, `step_ionly`,
   `zenith_weights` and their plot scripts. Regenerate those figures and diff
   the numbers against `report/report.tex`.
7. **Retire.** Delete the old two-port stack and its tests; demote `fourport.py`
   to `_legacy_pixel.py` with the shared pieces lifted out; update `CLAUDE.md`,
   `AGENTS.md`, `PROGRESS.md`, `SETUP-CROSSCHECK.md`.

## 8. Out of scope

- Path 2, the ensemble diffuse prediction. It is a separate spec, and this
  refactor is a prerequisite for validating it rather than part of it.
- Editing the paper itself (`/home/anze/latex/LuSEE/Faraday`).
- The RM-synthesis and Fisher-forecast branches (`faraday-rmsynth`,
  `faraday-fisher-forecast`). They are not on this line of work.
- The open item from the audit -- reconfirming through the real BGL_v16
  four-port kernel rather than the `cos^2` test lobe. That is ~30 minutes of
  independent work and does not block or depend on this refactor.
