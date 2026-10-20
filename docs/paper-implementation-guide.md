# Reproducing the conditional LuSEE-Night predictions

The completed calculation follows the [accepted implementation plan](plans/2026-10-17-paper-implementation.md)
and Fable's framing/feasibility prescriptions. PR [9](https://github.com/christianhbye/lusee_faraday/pull/9)
targets `paper-implementation-plan`. The external manuscript is protected;
its transfer and a comparison with actual mission data follow separately.

## Environment and inputs

Use `uv sync --extra dev` to reproduce the editable environment, or the
existing `.venv`. Do not modify either upstream checkout. Source manifests
record the imported paths, versions and revisions rather than relying on
the older paths in AGENTS. The validated local environment imports luseepy
at `eb51360e8a6d7f6253c20d1b826888e0602018ef` and croissant at
`cc867f1f0e0f0dcd163b7c50c5e043f599a3536c`; the report records all installed
package versions. A different environment requires regeneration, not reuse
of a mismatched cache.

The required large files are the three as-built/C4/diagonal-ZA BGL v16
response FITS files and the native-resolution Haslam map under `data/`.
Their complete hashes are in `docs/paper-kernel-results.json` and
`docs/chromatic-leakage-results.json`. The new production calculation does
not apply an unresolved Hutschenreuter map pixel by pixel. Its screen
statistics are explicit conditional inputs. Existing point-source and
instrument scripts remain available; the superseded diffuse forecast stays
at the `audit-2026-08-18` tag.

## Running

```bash
bash scripts/run_paper_implementation.sh --help
bash scripts/run_paper_implementation.sh --smoke
bash scripts/run_paper_implementation.sh
```

The launcher detaches a worker, prints its PID and uses an exclusive lock.
It appends to the absolute log `generated_data/paper_implementation.log`.
The worker sets x64 before JAX, controls numerical thread counts, and caps
virtual memory at 24 GB (16 GB for the smoke suite). Check its PID and log
before starting another expensive calculation. Every science stage has
explicit arguments or `--help`; the older `validate_engine.py` and
`crosscheck_pixel_arm.py` remain exceptions and must not be probed with
`--help`.

Use `--from STAGE` to resume. The stages are `horizon`, `depth`, `kernels`,
`foreground`, `mixed`, `exposure`, `leakage`, `signal`, `controls`, `system`,
`local`, `science`, `observations`, `figures`, `report`. Production blocks checkpoint
under a fingerprint of their configurations, source and resolved numerical
libraries. Missing matching caches trigger computation; corrupt or
incompatible inputs raise an error. They never fall back to a previous key.

After changing a scientific configuration, regenerate dependent stages in
order and commit the completed source/records before building the report.
The report builder enforces that boundary. It creates a clean detached
worktree at the committed source, links the checksummed large inputs,
generates tables/macros, runs LaTeX twice and checks its log before copying
products back. This also isolates unrelated notebook edits in the shared
workspace. Report products are ignored by default; use explicit `git add -f`
for the reviewed PDF and generated `report/paper-*` tables/manifests.

The expensive native response transforms and mixed moments are cached.
The final report records measured runs; it does not promise that a cold run
will complete within a particular wall-clock time.

## Products and coordinates

| Record | Contents |
| --- | --- |
| `faraday-horizon-results.json`, `local-input-audit-results.json` | Sourced density/field pairs; assumed directions, volume/emissivity and lunar envelope; unbounded zero-projection case |
| `depth-depolarization-results.json` | Independent flat-plane depth-profile reference and smooth/speckle powers |
| `paper-kernel-results.json`, `mixed-depth-results.json` | Full complex native kernels and spherical mixed-depth/channel moments |
| `residual-signal-results.json` | Full raw/fitted joint Stokes covariances, same-phase averaging, Gaussian/global-mixture models |
| `chromatic-leakage-results.json`, `measured-system-results.json` | Native chromatic I-only variants, common as-built calibration, interpolants and loaded joint radiometer noise |
| `observing-exposure.json` | Actual-site complete nightside bins, duty, available integration and unavailable phases |
| `local-medium-sensitivity-results.json` | Complete signed-column grid; exact marginal moments and latent coefficients for any requested joint C/J |
| `local-medium-joint-validation.json` | Full covariance/relation checks at eight predeclared extreme-column cases |
| `local-medium-separability-results.json` | Full decision tensor, coordinates and conditional emissivity rescalings |
| `faraday-separability-results.json` | Reference-grid decisions, simulated spectra, independent-noise mocks, parent/zoom ratios, named-bin products and fixed-sky repeats |
| `science-product-results.json` | All 16 real science products: compact signal factors, I-only means and selected full noise/signal covariances, with Stokes reprojection checks |
| `common-observing-results.json` | Angular spectra and a common masked fringe operator for polarized sky, leakage and noise |
| `paper-report-build.json` | Clean source revision and hashes of the built PDF, generator and logs |

Record names in this table are relative to `docs/`; their `path` fields name
ignored, completely checksummed NPZ arrays. All frequencies are in MHz,
Stokes means in K, and signal covariances in K² at the common loaded
calibration plane. The radiometer arrays named `*_per_second` store
covariance times integration in K² s: divide by integration in seconds.
They include full channel overlap and all I/Q/U/V cross terms.

The science-product factors replace the phase*4 Stokes dimension with
phase*16 products in `PRODUCT_LABELS` order. Their mode weights contract
with the saved local-grid phase coefficients through the same
`DepthChannelProjector.moments` representation. They also store the fine
loaded sky PSD, all nine amplifier PSDs and exact channel/readout operators
for arbitrary requested noise blocks. The selected noise arrays carry
noise-model and phase axes followed by `(bin, product, bin, product)`.

Full joint Stokes arrays have axes `(bin, phase*4+Stokes, bin, phase*4+Stokes)`,
with Stokes order I/Q/U/V. The illustrative observation products select one
phase and use `(bin, Stokes, bin, Stokes)`. Complex polarization exports use
P=Q+iU and retain both C=E[PP*] and J=E[PP]. They are ensemble moments,
not an empirical covariance estimate from one measured spectrum.

The compact local decision tensor has axes `(case, phase, channel_type,
component, lunations, noise_model)`, with the coordinate lists in its JSON.
Raw/residual RMS and retained-power arrays omit the last two axes. A zero
exposure gives NaN in binary noise ratios, not zero noise. Exact zero
speckle gives an undefined retained-power ratio. Gaussian visibility draws
are conditional illustrations of second moments. The two band illustrations
are independent draws, not a joint cross-band sky realization. Matching an RM covariance
does not establish Gaussian phase or visibility statistics.

To export any complete local-grid covariance, take a `case_id` from the
record and run under the same background/memory convention:

```bash
.venv/bin/python scripts/local_medium_sensitivity.py joint --case CASE_ID
```

The resulting `CASE_ID_joint.npz` and sidecar include cross-channel and
cross-phase moments, complex C/J, PSD diagnostics and all refinement gates.
A failure is retained and returned as a nonzero exit status.

## Future data and scope

`scripts/faraday_separability.py:analyze_observation` reads an NPZ containing
`frequency_mhz` and calibrated `stokes_k` with shape `(bin, 4)`. It requires
exact agreement with the prediction coordinates and the supplied common
residual operator, noise covariance and intensity reference. The independent
simulated inputs exercise this route. Mission use requires regenerating
those operators for the actual response, calibration, flags, frequencies,
time bins and integration. The source exports subtraction in K, fixed-I
fractions, and estimated-I fractions with the full I-Q/U covariance
Jacobian; it rejects inadequate intensity S/N.

The complete signed grid normalizes intrinsic polarization to the assumed
volume-averaged polarized RMS emissivity times the full emitting path.
Depth integration supplies the suppression; a second horizon cut would
count it twice. Density/field/path pairs set uniform mean-column sensitivity,
while dispersion, angular structure, depth correlation and intrinsic E/B
spectra remain assumptions. The one-kelvin measured-spectrum illustration
is a separately labeled reference case. Brighter HII-based rescalings are
conditional normalization examples, not measured few-parsec polarized priors.

Repeated visits retain one fixed sky and instrument and draw independent
measurement noise. Their common structure does not identify Faraday
rotation. Reported signal/noise quantities are RMS ratios, not calibrated
detection thresholds; angular input shape alone is not a map sensitivity.
No launch date, flight calibration or detection promise follows from this
calculation.
