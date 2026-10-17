# LuSEE Faraday analysis and paper preparation

Current handoff, 2026-10-16: step 1 was reviewed and merged as PR 7
(`3bf50ce`). The [toy inference plan](plans/2026-10-16-gaussian-screen-inference.md)
is the next implementation sequence: full covariance validation, information
and nuisance assessment, strong-rotation/template comparison, then a small
calibrated inference exercise. Recovery is conditional on identifiability.
The [Galactic-model follow-up](plans/2026-10-16-galactic-residual-screen-models.md)
is deferred. No inference or full covariance validation run has started.

Preparation status as of 2026-10-15: The long computation
has not been launched. The immediate priority is agreement on the physical
model and observable. Bounded real-beam checks diagnose the old numerical
failure; they do not validate the replacement covariance model. Physical
validation must precede interpreting forecasts or producing the manuscript.

Step 1 is now available for review in the executed
[Gaussian-screen notebook](../notebooks/gaussian_screen_step1.ipynb), with its
[model specification](specs/2026-10-15-gaussian-screen-step1.md),
[implementation and step-2 validation plan](plans/2026-10-15-gaussian-screen-step1.md),
and [generated check/provenance record](gaussian-screen-step1-results.json).
It derives the paired off/on statistics for a flat-plane Gaussian source,
beam and correlated physical screen. Its limiting-case and separation-integral
checks do not yet validate spatially varying deterministic simulations.
The proposed benchmark below is now specified; steps 2--4 remain unexecuted.
The existing forecast and the real-beam audit retain their review gates.

The [2026-10-16 discussion record](plans/2026-10-15-gaussian-screen-step1.md#discussion-record--2026-10-16)
clarifies the fixed-map versus ensemble prediction, covariance interpretation,
real-beam/RM limitations, and connection to measured four-port spectra. It
recommends a validated toy inference demonstration before realistic inference;
that addition is proposed and unexecuted.

This is the consolidated handoff. [TODO](../TODO.md) holds the actionable
queue; [the generated snapshot](paper-preparation-status.json) records input
checksums, dependency revisions, saved forecast values and audit completion.
Regenerate that snapshot with
`.venv/bin/python audit/scripts/prepare_paper_review.py`. It reads existing
products and hashes files; it does not run a sky simulation.

## What is complete and where it lives

| Work | State and evidence |
| --- | --- |
| Four-port instrument, loading, polarimeter, point source, I-only leakage | Merged in [PR 3](https://github.com/christianhbye/lusee_faraday/pull/3), main `33c8730`. Historical validation is described in `docs/measurement-model.md`. |
| Diffuse pixel-sum audit | Original reproducible audit in `audit/`; refuted diffuse results and old report preserved at `audit-2026-08-18`. |
| Replacement diffuse prediction | Implemented on `faraday-delay-template`, head `adb4024`, [PR 4](https://github.com/christianhbye/lusee_faraday/pull/4) open and based on main when checked 2026-10-15. |
| Detection report | `report/faraday_depth_template/faraday_depth_template.tex` and `generated.tex` on that branch; local [PDF](../report/faraday_depth_template/faraday_depth_template.pdf). |
| Executed companion and figures | `notebooks/faraday_depth_template.ipynb`, `scripts/step5_*.py`, and `report/figures/step5_*.pdf` on the template branch. |
| Real-beam follow-up | September legacy results at nside 512 and 1024; 2048 missing. Corrected script now prepared and tested with small synthetic inputs. |

The initial ensemble two-point/structure-function proposal was superseded by
`docs/superpowers/specs/2026-08-20-faraday-delay-template-design.md` on the
template branch. Its practical deliverable is a normalized depth-power
search template, a family of emission geometries, the real zoom response,
and a correlated-noise detection threshold. Absolute diffuse amplitude is
conditional on depolarization physics. A stable depth histogram does not
establish a continuum amplitude or a unique physical sky.

The saved detection product includes the low-depth leakage cut and the
24-lunation schedule. Its numerical values are in `saved_detection.rows` in
the snapshot. Each ratio divides a signal amplitude by a five-sigma
amplitude threshold; it is not itself a signal-to-noise ratio. Those saved
products have not been regenerated or revalidated in the current environment.
The slab scenario clears threshold at 30 and 50 MHz; the internal-dispersion
scenario does not. The existing 10-MHz forecast does not cure its channel
aliasing limitation.

## Mathematical and physical basis review — 2026-10-15

The template branch implements a statistical search model, but its derivation
from a physical unresolved Galactic sky remains open. The earlier description
of remaining work as report corrections plus systematics was too optimistic.

### Relation to the original two-sky simulation

The original comparison uses a prescribed polarized sky, extrapolated from
WMAP, with Faraday rotation switched off and on. For a foreground screen,
the on sky multiplies complex polarization by `exp(2i phi lambda^2)` before
the instrument's angular integral. The observable remains the complex
port-pair visibility versus time and frequency. Receiver loading is frozen
along with the beam in the fixed-beam approximation.

A statistical replacement predicts an ensemble mean and covariance of those
same visibilities. The ensemble comprises possible physical skies/screens;
it does not assume that the Galaxy changes independently between observations.
At a fixed time and port pair, define

\[
\mu_i=\langle V(\nu_i)\rangle,\qquad
C_{ij}=\langle[V(\nu_i)-\mu_i][V(\nu_j)-\mu_j]^*\rangle.
\]

The diagonal gives variance at each frequency; off-diagonal entries describe
shared fluctuations and their relative complex phase. A deterministic off
spectrum can still be retained: subtracting it shifts the mean but leaves
the centered covariance unchanged. Squared on-minus-off differences also
contain the mean contribution. If the off sky is uncertain, its covariance
and the joint on/off cross-covariance are needed.

Unresolved oscillatory phases can leave a discrete pixel sum dominated by
its diagonal `sum(abs(w)**2)` term, which changes with quadrature resolution.
That numerical term does not establish physical continuum variance. Changing
the integrator or taking covariance of the same unresolved sums cannot
supply missing angular correlations.

### Assumption behind the implemented depth-power template

For explanation, take one complex polarization branch with the beam already
included and let `x = lambda^2`. Its complex amplitude density `g` gives

\[
p(x)=\int g(\phi)e^{2i\phi x}\,d\phi.
\]

For centered depth covariance
`C_g(phi,phi') = <delta g(phi) delta g(phi')*>`, the general result is

\[
C_p(x,x')=\iint C_g(\phi,\phi')
 e^{2i(\phi x-\phi' x')}\,d\phi\,d\phi'.
\]

The implemented normalized template corresponds to assuming

\[
C_g(\phi,\phi')=A^2H(\phi)\delta(\phi-\phi'),\quad
\int H(\phi)\,d\phi=1,
\qquad
C_p(x,x')=A^2\int H(\phi)e^{2i\phi(x-x')}\,d\phi.
\]

`H` is power density, distinct from complex amplitude `g`. This assumes
uncorrelated fluctuations at different depths. A beam/emissivity-weighted RM
histogram does not establish that assumption or identify the physical `H`.
Histogram refinement/rotation stability cannot establish physical covariance
shape or amplitude. Signed depth matters for complex frequency covariance.
A general complex port-pair visibility requires both polarization branches
and their joint statistics; this single-branch expression is explanatory.

The physical starting point is polarized emissivity along each sightline,
with its intrinsic angle and accumulated Faraday depth, followed by the
instrument response. Covariance requires angular and line-of-sight
correlations. Total-column RM does not specify where emission lies along
the column. Depth-fraction prescriptions must distinguish amplitude
emissivity from power weighting and account for magnetic-field reversals.
The complex-amplitude Faraday representation is standard
([Brentjens & de Bruyn 2005](https://arxiv.org/pdf/astro-ph/0507349)); the
independent-depth covariance is an additional model assumption.

### Consequences for amplitudes and detection forecasts

- A uniform coherent slab has normalized amplitude
  `abs(sin(Phi lambda^2)/(Phi lambda^2))`, where `Phi` is total depth.
  The `1/abs(Phi lambda^2)` envelope is not a lower floor: the exact
  expression has zeros. Sightline attenuation does not automatically give
  beam-integrated RMS or fractional total-intensity signal. Intrinsic
  polarization and angular coherence must enter that conversion.
- `abs(<P>)**2` and `<abs(P)**2>` differ. Suppression of mean polarization
  does not establish finite-beam residual power; screen correlations matter
  ([Tribble 1991](https://adsabs.harvard.edu/pdf/1991MNRAS.250..726T)).
- Coherent slab/internal-dispersion attenuations need a common derivation
  before being combined with independent-depth power as physical predictions.
  Until then they are exploratory amplitude scenarios. Physical turbulent
  dispersion must remain distinct from RM reconstruction uncertainty.
- The inspected signal-covariance routine evaluates channel-center phases.
  A data-space prediction must integrate signal covariance through actual
  channel responses, as well as propagating correlated channel noise.
- Excluding low-depth power from the signal template does not reject
  foregrounds in data. Define the actual projection, filtering or nuisance
  marginalization and propagate it through signal and noise before
  interpreting separation or detection thresholds.

### Required next derivation

Specify the ensemble, mean and angular/depth correlations; derive visibility
covariance and identify any approximation reducing it to the current `H`
model. Recover a tractable case using independently converged visibility
simulations, including the off/on comparison. Apply channelization and
foreground treatment to the same observable. Then attach a physical amplitude
and update detectability claims. The original two-point/structure-function
proposal is a candidate framework, not a completed calculation. The real-beam
audit can proceed separately after review; it cannot replace these checks.

### Proposed first benchmark: a correlated Gaussian foreground screen

This is a proposed controlled calculation, not an established Galactic model
or an executed validation. Retain a deterministic intrinsic polarized sky
and frozen instrument, and first choose a Gaussian screen of constant mean
`m`, variance `sigma_phi^2` and isotropic angular structure function
`D(theta) = <[phi(n)-phi(n')]^2>`. Require a valid screen covariance, rather
than choosing an arbitrary structure function. Let `x = lambda^2` and
`x' = lambda'^2`. The Gaussian characteristic function gives exactly

\[
\left\langle e^{2i[x\phi(n)-x'\phi(n')]}\right\rangle
=e^{2im(x-x')}
 e^{-2\sigma_\phi^2(x-x')^2}
 e^{-2xx'D(\theta)}.
\]

Multiplying by the deterministic polarization and response weights and
integrating over both directions gives the single-branch second moment.
Subtract the product of means to obtain the centered covariance. Include
both branches and their cross terms for actual port-pair observables.
At equal frequencies the phase expectation becomes `exp(-2 lambda^4 D)`.
This same-frequency structure-function treatment follows
[Tribble (1991), equations 7–12](https://adsabs.harvard.edu/pdf/1991MNRAS.250..726T);
the displayed two-frequency identity follows directly by applying the
Gaussian characteristic function to `2[x phi(n)-x' phi(n')]`.

This is the prospective numerical advantage: average the fast random phase
analytically before doing angular quadrature. The remaining correlation
kernel has a narrow physical width, not a grid-pixel width. It still requires
converged integration. In an isotropic toy problem, beam autocorrelation
reduces the double angular integral to an integral over separation. Rescale
separation by the coherence angle or integrate analytically where possible;
do not allocate a dense all-pixel-pairs matrix. Real-beam reductions require
separate justification. A nonconstant screen mean can leave oscillatory
factors, and a non-Gaussian or mixed emitting/rotating medium needs a more
general derivation.

The controlled sequence is:

1. Derive the off/on mean, second moment and centered covariance for a simple
   beam and an admissible screen covariance with a physical correlation scale.
2. Compare to ensembles of individually converged deterministic simulations
   in a weak-rotation regime. Separately refine each realization's angular
   quadrature and increase ensemble size; compare discrepancies against
   sampling uncertainty. Recover zero-rotation and spatially constant-screen
   limits without inventing beam depolarization.
3. Increase rotation strength while holding the physical model fixed. Show
   convergence of the separation integral in the strongly depolarized regime,
   including any small-angle approximation against the full toy integral.
   Failure to converge or dependence on arbitrary pixel area rejects the
   proposed numerical cure.
4. Only after those gates, introduce WMAP variation and the actual response,
   then channel averaging. Test sensitivity to unresolved physical screen
   correlations and to emission mixed with the rotating material.

A numerical quadrature of a fixed interpolated sky could in principle also
be improved directly; its failure so far does not prove that deterministic
integration is impossible. Statistical averaging changes what is predicted.
The payoff is an ensemble result conditional on physical correlations, not
the exact spectrum of our particular Galaxy. The toy expression also depends
on `x x'`, so it does not generally reduce to a covariance depending only on
`x-x'`, as in the current normalized depth-power model. Assess such a
reduction within each narrow band instead of assuming it.

## What the September computation actually completed

`audit/generated/shuffle_bgl_results.json` contains 48 entries at each of
nside 512 and 1024, and none at 2048. The September 19 log stops after the
2048 maps were made. A prior chunked 512 smoke check completed at about
2.6 GiB peak RSS. These observations fit a memory-related interruption;
no retained OOM event was found in the journal, so the exact cause is open.
No corresponding process was running during the October 15 survey.

The September ledger remains unchanged and is legacy evidence. Preparation
identified two additional defects in its script:

- Both Faraday branches used the positive exponent. The negative branch
  requires the opposite sign, including for complex cross-pair kernels.
- `abs(sum(w))**2 / sum(abs(w)**2)` was labelled the effective pixel count.
  For complex weights it includes intrinsic phase cancellation. The
  amplitude-weight count is `sum(abs(w))**2 / sum(abs(w)**2)`; the unrotated
  cancellation is now reported separately.

Consequently the corrected run must cover all resolutions. September's
negative-branch values and effective-count labels must not be carried into
new conclusions. The original scalar-lobe audit is independent of these
newly found script defects.

## Prepared audit changes

[The audit script](../audit/scripts/shuffle_control_bgl.py) now has:

- a `--plan` mode with no numerical imports, sky loading or file writes;
- opposite exponents for the two branches and separate amplitude-count and
  phase-cancellation statistics;
- disk-backed selection, direction and complex weight arrays;
- kernel direction batches and NUFFT source batches with explicit sizes;
- complex visibility sums across batches before taking the squared modulus;
- a separate version-2 result ledger, atomic per-observable checkpoints,
  concurrent-writer exclusion, and fingerprints of effective inputs, local
  numerical sources, installed package versions and editable dependency code;
- stage-level peak-RSS logging and optional saved spectra.

Incomplete geometry scratch is rebuilt on restart. Completed observables
are reused only under the same fingerprint. Changing batch size or requested
resolution does not invalidate scientific inputs. Changing code, the native
sky cache, response bytes or dependency source does. Separate ledgers get
separate scratch paths. No automatic cleanup deletes old evidence.

At nside 2048, the preflight estimates 1.50 GiB for the three fields plus
permutation and 1.88 GiB of disk scratch per snapshot. A direction batch
returns at most 61 MiB in one sampled-kernel array. Interpolation holds
additional arrays; selected RM vectors, transforms, numerical libraries and
file-backed pages also contribute. These are array sizes, not measured
peak memory. The full-resolution memory fix remains to be demonstrated.

The script requires the existing native sky cache and hashes its bytes.
This pins the effective input used; it does not retrospectively establish
how that cache was made. Sky fields are synthesized from the same fixed
alms at every resolution, without `ud_grade`.

## Environment before the run

The operational instructions still describe the August dependency setup.
The installed environment on October 15 differs: lusee is editable at
`../luseepy`, croissant at `/home/christian/Documents/projects/croissant`;
the old `croissant-main` checkout is absent. Exact revisions and installed
versions are captured in `current_audit_provenance` in the snapshot.
The pre-existing `pyproject.toml` and `uv.lock` edits were preserved.
No upstream checkout or dependency pin was changed during preparation.

Use the existing `.venv/bin/python` for the proposed run so `uv` cannot
silently sync or change the environment. Numerical imports pass in the
small tests. The first real-beam smoke run must establish compatibility
with the current response reader. Before reproducing the template report,
choose its environment explicitly and repeat its relevant validation gates;
the October preparation tests do not validate the old template branch.

## Proposed computation after review

Read-only preflight, already exercised:

```bash
bash audit/run_bgl_audit.sh plan
```

After approval, run the corrected 512 smoke at both required snapshots:

```bash
bash audit/run_bgl_audit.sh smoke
```

The launcher creates a user systemd service with an 8-GiB resident-memory
cap, reclaim pressure at 6 GiB, no service swap, a separate
`ulimit -v 16000000`, two numerical threads, and an absolute log under
`generated_data/`. It uses the as-built response. Unit names distinguish
smoke and full runs. The ledger lock prevents them from writing together.
Review peak RSS and service exit status before progressing. The launcher
has been syntax-checked; no smoke/full service has been started.

Then run the full resolution ladder, reusing completed version-2 entries:

```bash
bash audit/run_bgl_audit.sh full
```

Inspect `systemctl --user status lusee-bgl-v2-full.service`, the log and
`systemctl --user show lusee-bgl-v2-full.service -p Result -p ExecMainStatus -p MemoryPeak`.
A stopped/failed unit must be reset before reusing its name. An OOM kill is
a failed resource check, not a scientific null. Lower batches or revise
storage after reviewing the failure; do not simply raise the cap on this host.

### Evidence required before closing the audit

Keep the native RM fraction, the RM-scaled positive controls, both snapshots,
both port pairs, both Faraday branches, and the horizon/86-degree selections.
Store spectra as well as integrated powers.

1. First verify the smoke run completed and produced finite values and
   usable spectra within the resource cap.
2. At the native RM scale, report measured/shuffled/incoherent power and
   resolution dependence for every observable. The shot-noise hypothesis
   predicts order-unity ratios and power falling approximately with inverse
   pixel count; disagreement must remain visible rather than be averaged away.
3. At the small-RM control, require a resolution plateau and growing
   separation from the shuffled result. If a control remains unconverged or
   fails to separate for an observable, investigate it before interpreting
   that observable's native-scale null.
4. Report the 86-degree mask's effect, amplitude-weight effective counts,
   phase cancellation, peak RSS and complete/incomplete run counts.
5. Preserve the fingerprinted ledger and a compact tracked summary; repair
   the report's explanatory figure from the appropriate linear amplitudes.

These are the intended diagnostics. A calibrated statistical tolerance for
these correlated spectra has not yet been established; a narrow arbitrary
pass interval would overstate this audit.

## Paper structure proposed for review

Working title: **Faraday rotation with LuSEE Night: instrument response and
conditional diffuse detectability**.

The main text should develop theory, observable quantities and results in
that order. The long report supplies derivations and reproducible artifacts;
its section order and exploratory history need not become the paper.

1. **Scientific question and measurement.** What a Faraday detection would
   establish; available frequency/channel coverage; distinguish detecting
   polarized spectral structure from localizing Galactic depth features.
2. **Instrument and estimator.** Four-port covariance, frozen beam and
   loading, calibrated readout, parent/zoom responses. Define signed complex
   amplitudes and the depth-power distribution as distinct objects.
3. **Controlled validation and the diffuse numerical failure.** Point-source
   rotation/channel averaging, I-only leakage, then the scalar and real-beam
   convergence controls. Explain why changing quadrature engine cannot supply
   unresolved physical coherence.
4. **Diffuse search templates and assumptions.** Weighted RM distribution,
   emission-depth geometry, coherence dependence and turbulence. Separate
   numerical stability from physical validity of the template family.
5. **Detectability and instrumental separation.** Signed frequency covariance,
   correlated zoom noise, low-depth exclusion, window response and conditional
   forecasts at 30/50 MHz. State schedule and receiver-noise assumptions.
6. **Physical interpretation and limits.** Depolarization scenarios, spatial
   spectral-index uncertainty, relative calibration and what cannot be inferred
   from an unresolved screen. State the evidence needed to improve predictions.
7. **Conclusions.** Instrumental capability and conditional search strategy,
   with amplitude uncertainty attached to every diffuse claim.

### Proposed figures and their sources

| Figure | Existing source | Condition for use |
| --- | --- | --- |
| Point-source response and channel recovery | `step1_*` / `step1w_*` on main | Select calibrated results; preserve PSD and frequency conventions. |
| Convergence and shuffle/control comparison | `audit/`, new BGL version-2 spectra | Complete corrected ladder; replace the report's erroneous random-walk panel. |
| Depth template and measured channel response | `step5_template_family`, `step5_pushforward`, `step5_envelope` on template branch | State geometry and coherence assumptions; retain signed covariance. |
| Detection versus excluded depth, by band | `step5_detection` | Revalidate environment; bracket receiver noise and physical amplitude scenarios. |
| Dominant uncertainties | `step5_robustness`, `step5_slope` plus planned scans | Add varying spectral index, defensible turbulence range and calibration dependence. |

Keep the old diffuse waterfalls/delay amplitudes out of the result figures.
They belong only in the numerical failure demonstration. The supposed
incoherent-patch upper amplitude is not determined by the map and must not
be promoted to a physical bound.

### Corrections required before transferring report prose

- Repair the coherent-sum amplitude/power-weight confusion and regenerate
  its figure and numerical macros.
- The report's `Re[(K_Q-i K_U)P]` shorthand does not represent a general
  complex cross-pair visibility. Use both complex branches, as tested in the
  prepared audit. This is a prose/algebra issue in addition to the audit
  script's exponent error; it does not by itself establish a forecast bug.
- Finite `rho(f) proportional to f^k` is integrable for `k > -1`.
  The `k -> -1` local-emission limit must be stated as a limit.
- Reconcile descriptions of `f` as physical distance versus accumulated
  Faraday-depth fraction, especially where fields reverse along a sightline.
- Describe template stability as a conditional model result. The existing
  uniform-weight rotation/refinement tests cannot establish physical sky
  coherence or constrain the unresolved amplitude.
- Use named physical amplitude scenarios until their status as bounds has
  been justified. Do not describe the current conditional forecast as a
  guaranteed detection, or its amplitude/threshold ratio as SNR.
- Retain receiver noise, coherence-dependent shape, spectral-index structure
  and calibration limits alongside the corresponding result.

The external manuscript location `/home/anze/latex/LuSEE/Faraday` is absent
on this machine. No manuscript source was edited. This outline is ready for
review before selecting or creating the local paper source.

## Remaining science after establishing the physical model

First complete the mathematical-basis review and derivation above. The
previously scoped systematics remain relevant, but cannot validate the model
on their own. Their previous order was spatially varying polarized spectral index,
then a defensible beam-weighted `sigma_eff` range, then turbulence in the
depth-power shape, and then band-to-band gain calibration. An intensity-derived
spectral index is only a proxy for the polarized index; WMAP intensity zero
levels/negative pixels need explicit treatment before taking map ratios.
RM reconstruction uncertainty must not be silently identified with physical
sub-beam turbulent dispersion. Turbulent broadening of an incoherent power
template does not determine a coherent polarization amplitude.

The forecast also needs receiver noise at the appropriate loaded reference
plane and beam-weighted antenna temperatures. Existing default-zero amplifier
noise does not establish sky domination at 50 MHz. These are separate from
the memory fix and should enter a common sensitivity/claim review.

## Preparation verification

All 12 checks in [the test file](../tests/test_shuffle_control_bgl.py) pass.
They use tiny synthetic inputs:
explicit coherent sums for both exponent signs, complex cross-pair Stokes
reconstruction, auto-branch conjugacy, masked source batching, disk-backed
sampling against independent transport, effective counts under cancellation,
checkpoint/provenance rejection, writer exclusion, spectrum output, restart
without sky loading, and a read-only preflight without numerical imports.
The preparation test log is `generated_data/paper_preparation_tests.log`.
Deliberately restoring the positive exponent on the negative branch makes
its complex-Stokes reconstruction check fail. Restoring the complex-sum
count makes its cancellation check fail. These mutations were confined to
isolated test processes; their logs are captured in the generated snapshot.
Formatting/lint and [launcher](../audit/run_bgl_audit.sh) syntax pass, and its
read-only default was exercised. Full-resolution
memory and real-response compatibility remain checks for the approved smoke
and full runs.

Pre-edit copies of the untracked script and TODO were preserved under
`generated_data/cache/paper_preparation_20261015/`. Nothing was committed,
pushed, merged or launched as a long scientific job during preparation.
