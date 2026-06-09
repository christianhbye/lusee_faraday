# lusee_faraday

Simulator and detection forecast for **LuSEE Night** Faraday-rotation observations.
Computes polarized radio visibilities seen from the lunar surface — including
Faraday rotation of the synchrotron sky — and forecasts how well the Faraday
signal can be recovered via rotation-measure (RM) synthesis.

## Install & commands

```bash
uv pip install -e ".[dev]"   # install (uv with a .venv present)
uv run pytest                # run tests
uv run black src/            # format (line-length 79)
uv run flake8 src/           # lint
```

## Two pipelines

**1. Forward simulation** — `Sky → Faraday rotation → coordinate rotation → beam → visibilities`.
Core classes: `SkyModel` (`sky.py`), `Beam` (`beam.py`), `Simulator` (`sim.py`),
`SpectrometerResponse` (`spectrometer.py`). `fast_sim.py` is the optimized,
parallel FR path used for full-band runs.

**2. RM-synthesis detection** — `full-band sim → channelize → RM synthesis → detection significance`:

- `FrequencyPlan` (`freqplan.py`) — choose zoom vs wide channelization per parent
  bin on a deduplicated raw grid; `channel_table` provides per-channel `nu`,
  `lambda2`, `dnu`.
- `pipeline.simulate_channelized` — FR (channelized, captures bandwidth
  depolarization) + no-FR (channel centers) Stokes.
- `rmsynth.py` — `faraday_spectrum`, `rmsf`, `phi_grid`, resolution helpers.
- `noise.py` / `detection.py` — radiometer noise and Faraday detection SNR.

## Workflow scripts (`notebooks/`)

| Script | Purpose |
|---|---|
| `grid_design.py` | Build/validate the full-band grid (wide 30–51 MHz + zoom 5–30 MHz) via the cheap RMSF |
| `faraday_fullband_sim.py` | Curated full-band sim → `results/faraday_fullband.npz` (~2 h on 4 cores) |
| `rmsynth_analysis.py` | RM synthesis + Faraday detection significance vs integration time |
| `rmsynth_calibration.py` | Calibration on the legacy 3-band sims |

Design docs and implementation plans live in `docs/superpowers/`.

## Key result

Combining the ~4000 spectrometer channels, the Faraday-induced signal
(~22% median depolarization) sits far above the radiometer noise — LuSEE is
**not noise-limited** for Faraday rotation. The limitation is the degeneracy
with the unknown intrinsic polarized sky (the beam-averaged RM is ≈0, so there
is no clean RM-synthesis peak — only depolarization and spectral structure).
Breaking that degeneracy with the known Galactic RM map (a matched filter) is
the natural next step.

## Conventions

- HEALPix RING ordering, default `nside=128`; frequencies in MHz.
- Zoom-bin channels are FFT-ordered (not monotonic) — pair λ²/Stokes via
  `FrequencyPlan.channel_table`, not `utils.freqs_zoom`.
- For the parallel sim, pin BLAS threads (`OMP_NUM_THREADS=1`) and use one
  process per physical core (the driver does this).
