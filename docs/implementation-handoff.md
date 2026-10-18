# Paper implementation handoff

Branch: `implementation/depth-depolarization`. PR
[9](https://github.com/christianhbye/lusee_faraday/pull/9) targets
`paper-implementation-plan`. The numerical implementation, report and
reproduction workflow are complete. The PR remains open for the owner to
review and merge. No approval is needed to finish or push this work.

The [accepted plan](plans/2026-10-17-paper-implementation.md) records the
completed scientific gates. The [reproduction guide](paper-implementation-guide.md)
provides stage commands, prerequisites, units, array schemas, the future
calibrated-data interface and interpretation limits.

## Completed science

- Cited horizon/local-emission and lunar-envelope scenarios distinguish
  measured quantities, model pairs and assumed volume/orientation.
- Flat-plane depth profiles and the spherical spin-2/all-pair calculation
  retain both complex moment types, smooth power, centered speckle and
  correlated emitting depths. Matched geometry, scalar-floor, independent
  channel integration and matched-RM non-Gaussian controls pass.
- Native as-built/C4/diagonal-ZA chromatic leakage, full receiver loading,
  nonzero amplifier/thermal PSDs and channel-overlap noise share the same
  calibration, physical-frequency fit and actual-site time averaging.
- The full signed mean-column grid covers 520 cases. All marginal numerical
  gates pass; frequency/depth refinements bound full Q/U covariance change.
  Eight predeclared extreme cases pass full joint checks; saved coefficients
  support requested covariance exports for every case.
- All 16 science products have compact covariance factors, I-only means,
  selected full signal/noise blocks and exact projection checks back to
  Stokes. Both bands, phases and all nine receiver cases are covered.
- Simulated Q/I and U/I spectra, same-noise I-only controls, independent-noise
  observations, named-bin statistics, parent/zoom ratios and fixed-sky repeat
  differences exercise the future-data comparison path. Nine report figures
  cover the six required categories plus supporting covariance/repeat/observing
  diagnostics. The angular control compares leakage and polarization through
  the same masked observing operator, without a map sensitivity claim.

## Results and limits

The full volume normalization is assumed polarized RMS emissivity times the
entire 5 pc emitting path, not a second phase-horizon truncation. At its
5 K reference RMS, the largest speckle/noise amplitude ratio is 3.53e-4 and
total-polarization/noise ratio 7.49e-3. Conditional HII normalization examples
reach 0.151 for total/noise. These RMS ratios are not detection thresholds,
posterior limits, or measured few-parsec polarized priors. Local angular
structure, dispersion, E/B spectrum, depth correlation, turnover, receiver
current noise and beam errors remain assumptions. The original one-kelvin
measured-spectrum example is labeled separately.

Future LuSEE-Night science data remain the observational target. Apply the
same calibration, actual channel responses, flags, exposure and fit to
predictions and data. Repeated cycles reduce independent measurement noise;
they are not independent skies. Independent band illustrations do not supply
a joint cross-band sky realization. The external manuscript is untouched;
template integration and manuscript transfer remain outside this PR.

## Verification and preservation

`docs/paper-integration-validation.json` records 227 passing local tests and
the same historical measured two-port monopole-reference failure reproduced
before this work. The 34-check smoke suite passes. Each new scientific test
has a recorded injected fault that makes it fail. CI passes with its own
pinned dependencies on Python 3.12/3.13/3.14. Do not change the historical
reference to hide the environment-dependent mismatch.

The two Gaussian-screen notebook edits are the user's pre-existing work;
they must remain unstaged and unchanged. Both upstream checkouts remain
unmodified. Large arrays and individual figures remain ignored, regenerable
and fully checksummed. The report source and generated review PDF live in
the permitted `report/` area.
