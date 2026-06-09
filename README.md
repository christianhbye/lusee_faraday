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

**3. Fisher-forecast detection** — answers: *can LuSEE detect the Faraday
rotation amplitude once the unknown intrinsic polarized sky is marginalized?*
The data are linear in the intrinsic (Q,U) sky given known beam+RM+rotation
operators, so the Fisher matrix `F = JᵀN⁻¹J` can be built with an analytic
Jacobian. Marginalizing over low-ℓ spin-2 sky modes + an effective
Faraday-dispersion variance τ gives the realistic `sigma(alpha)` and
`SNR = alpha_fid / sigma(alpha)`.

- `forward.py` — `pol_response(...)`: linear forward operator → complex
  `pQ + i*pU` per (time, channel); `alpha` scales the RM map.
- `skybasis.py` — `spin2_basis(nside, lmax)`: low-ℓ spin-2 (Q,U) basis for
  the marginalized sky nuisance.
- `fisher.py` — `run_forecast(...)`: assembles the Fisher matrix, returns
  sky-marginalized `sigma(alpha)` and `SNR`.

## Workflow scripts (`notebooks/`)

| Script | Purpose |
|---|---|
| `grid_design.py` | Build/validate the full-band grid (wide 30–51 MHz + zoom 5–30 MHz) via the cheap RMSF |
| `faraday_fullband_sim.py` | Curated full-band sim → `results/faraday_fullband.npz` (~2 h on 4 cores) |
| `rmsynth_analysis.py` | RM synthesis + Faraday detection significance vs integration time |
| `rmsynth_calibration.py` | Calibration on the legacy 3-band sims |
| `fisher_forecast.py` | Fisher-matrix forecast of Faraday-amplitude detectability after marginalizing sky nuisance → `results/fisher_forecast.png` |

Design docs and implementation plans live in `docs/superpowers/`.

## Key results

Combining the ~4000 spectrometer channels, the Faraday-induced signal
(~22% median depolarization) sits far above the radiometer noise — LuSEE is
**not noise-limited** for Faraday rotation. The limitation is the degeneracy
with the unknown intrinsic polarized sky (the beam-averaged RM is ≈0, so there
is no clean RM-synthesis peak — only depolarization and spectral structure).

The Fisher forecast answers whether that degeneracy is broken by the known RM
map. Marginalizing the low-ℓ spin-2 sky modes + Faraday-dispersion variance τ
reduces the Faraday-amplitude SNR by **<0.5%** vs the fixed-sky bound — the
smooth sky nuisance does not span the λ²-winding Faraday signature, confirming
that a matched-filter detection is not degeneracy-limited by these modes.
**Caveat:** the absolute SNR is an *optimistic upper bound* — the nuisance
basis is restricted (low-ℓ, fixed spectral index) and nside=64 smooths
small-scale RM; per-mode spectral freedom and higher ℓ are deferred.

## Conventions

- HEALPix RING ordering, default `nside=128`; frequencies in MHz.
- Zoom-bin channels are FFT-ordered (not monotonic) — pair λ²/Stokes via
  `FrequencyPlan.channel_table`, not `utils.freqs_zoom`.
- For the parallel sim, pin BLAS threads (`OMP_NUM_THREADS=1`) and use one
  process per physical core (the driver does this).
