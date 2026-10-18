# Paper implementation handoff

Work branch: `implementation/depth-depolarization`. PR:
[9](https://github.com/christianhbye/lusee_faraday/pull/9), targeting
`paper-implementation-plan`. The PR remains open for review.

## Implemented products

- Conditional, cited constant-rate horizon scenarios, including the
  unbounded zero-line-of-sight-field case. Sourced plasma inputs and assumed
  geometry/emissivity are separated in the input audit.
- Converged flat-plane screen fractions 0/0.6/1, a uniform slab and a
  truncated Gaussian emission profile, with means, full Q/U covariance,
  smooth/speckle power and the first depth figure.
- Spin-2 E/B correlations, all-pair complex covariance/relation, loaded
  calibrated readouts, selected sidereal time pairs and exact parent/zoom
  channel response. Frequency compression retains every pair and is
  refined against final channel moments.
- Full complex as-built lmax-96 kernels at native 10/30/50 MHz, plus four
  conditional foreground-screen channel cases at 30/50 MHz. These include
  three parent bins, central-parent zoom bins and two sidereal phases.
- A signal-only channel-correlation and Gaussian Q/U draw figure. For this
  weak local bracket the zoom spectra are nearly coherent across 25 kHz.
  Independent band illustrations do not constitute a joint cross-band sky
  realization. Repeated days within a modeled band must reuse its sky draw.
- A direct bivariate phase-quadrature control for the PR 8 variance mixture
  (0.5/1.5, equal weights). Matching the RM covariance does not recover the
  Gaussian column's phase moments. This control does not establish
  Gaussianity of instrument visibilities.

The detailed API/scope documents are
`docs/specs/2026-10-17-depth-depolarization.md` and
`docs/specs/2026-10-17-spin-channel-covariance.md`. The accepted five-phase
plan remains `docs/plans/2026-10-17-paper-implementation.md`.

## Reproduction

Use the existing `.venv` or `uv sync --extra dev`. Do not modify upstream
checkouts. The actual dependency paths and revisions are recorded with
each result; the live croissant checkout differs from the older AGENTS
path/reference. No external manuscript was edited.

Run `bash scripts/run_paper_phase1.sh` for the horizon/depth products and
`bash scripts/run_paper_phase2.sh` for the real-response products. These
launch detached, memory-capped jobs with absolute logs at
`generated_data/paper_phase1.log` and `generated_data/paper_phase2.log`.
Both honor checked, fingerprinted caches. Do not launch a duplicate while
the corresponding log/process is active.

`scripts/check_spin_screen_mixture.py` regenerates the phase-mixture control.
Run it under the same 16 GB background-job convention. The two fault-audit
drivers are `validate_depth_faults.py` and `validate_covariance_faults.py`.
Their disposable mutated copies do not alter working source.

All arrays/caches live under ignored `generated_data/`; inspected PDFs,
PNGs, captions and figure manifests live under ignored `report/figures/`.
Committed JSON records contain configurations, dependency/source/array
hashes and numerical gates. Stale or corrupt arrays are refused.

## Next scientific increments

1. Finish the local-prior/emissivity/pulsar audit. The horizon table does
   not determine local sigma_phi or angular structure. Current intrinsic
   polarization, E/B spectra, theta0, dispersion and turnover are assumed.
2. Carry the Phase 1 double-depth average into the spherical phase kernels
   and channel covariance. Keep emission depths correlated through the
   same column; a scalar depolarization factor is insufficient. Resolve
   the near-edge depth boundary layer and broad frequency component.
3. Complete the matched spherical/small-beam comparison in the PR 7–8
   benchmark windows, stronger-screen frequency refinement, scalar-pilot
   normalization checks and full non-Gaussian beam/channel controls.
4. Compute chromatic I-only leakage for as-built/C4/diagonal-ZA responses
   on actual native frequency coordinates. Apply one declared fit operator
   to leakage, polarization and noise; report signal attenuation too.
5. Add nonzero sourced or explicitly illustrative voltage/current receiver
   noise at the loaded plane, sky/Moon/loss system power and the joint
   radiometer covariance. Check with independent voltage draws and the
   actual channel overlaps, then specify integration/duty cycle.
6. Produce the planned Q/I and U/I spectra with leakage/noise controls,
   residuals and prediction bands, repeatability and separability tables,
   then the remaining paper/report products. Future LuSEE-Night mission
   data are the comparison target. No detectability claim is supported yet.

Existing user edits to both Gaussian-screen notebooks were preserved.
The local full suite has a historical measured two-port monopole reference
failure, reproduced before these changes; do not change that reference to
hide it. CI uses its own pinned dependencies and passes the data-free suite
on Python 3.12, 3.13 and 3.14. Consult the committed validation records for
the precise runs and scope.
