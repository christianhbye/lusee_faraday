# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulator for LuSEE (Lunar Surface Electromagnetics Experiment) Faraday rotation observations. Computes polarized radio visibilities as seen from the lunar surface, including Faraday rotation of synchrotron emission through the ionosphere.

## Setup & Commands

```bash
# Install (uses uv with a .venv already present)
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_name -v

# Format
uv run black src/

# Lint
uv run flake8 src/

# Launch notebooks
uv run jupyter lab
```

## Architecture

The simulation pipeline flows: **Sky → Faraday rotation → Coordinate rotation → Beam convolution → Visibilities**

- **`SkyModel`** (`sky.py`): Holds Stokes I/Q/U maps and a rotation measure (RM) map in HEALPix format. Can load WMAP K-band data or synthetic point sources. `apply_fd()` applies Faraday rotation in-place. `to_topocentric()` rotates maps from Galactic to lunar topocentric coordinates.
- **`Beam`** (`beam.py`): Stores Jones matrices for X and Y dipoles. Can load LuSEE beams from FITS files or create analytic short dipoles. `precompute_weights()` computes the 9 Stokes-weighted beam patterns (wI/wQ/wU for x/y/xy) used in the visibility integral.
- **`Simulator`** (`sim.py`): Configured via `SimConfig` dataclass. For each time step: rotates sky to topocentric frame, applies horizon mask, performs beam-sky multiplication, and normalizes. `compute_stokes()` converts Rxx/Ryy/Rxy visibilities back to Stokes I/Q/U.
- **`rotations.py`**: Galactic-to-topocentric coordinate transforms using `lunarsky` for lunar frame definitions and `healpy.Rotator` for polarized map rotation.
- **`HealpixGrid`** (`healpix.py`): HEALPix grid utilities including horizon masking and interpolation from regular theta/phi grids to HEALPix via `RectSphereBivariateSpline`.
- **`SpectrometerResponse`** (`spectrometer.py`): Loads the spectrometer bin response and convolves simulated spectra with either the wide (parent, 25 kHz) or narrow (zoom, 64 sub-bins) channel response. Zoom bins use FFT-style ordering (bin 0 = center, 1-32 positive offsets, 33-63 negative offsets).
- **`utils.py`**: LuSEE frequency channel definitions (2048 channels, 0–51.2 MHz) and zoom-bin helpers.

### RM-synthesis detection layer

A second pipeline forecasts detectability of the Faraday signal via rotation-measure (RM) synthesis on channelized spectra. Flow: **full-band sim → channelize → RM synthesis → detection significance**.

- **`fast_sim.py`**: Optimized FR visibilities. `precompute_rotated_maps()` rotates sky+RM to topocentric once per time; `compute_vis_fast()` applies the per-frequency Faraday rotation factorized as `A·cos(2·RM·λ²) + B·sin(...)`, over above-horizon pixels only; `compute_vis_fast_parallel()` splits the time axis across processes. It is single-threaded per process by design — when parallelizing, pin BLAS threads (`OMP_NUM_THREADS=1`, etc.) and use one process per *physical* core, or the cos/sin-heavy workers oversubscribe and thrash (the `faraday_fullband_sim.py` driver does this).
- **`FrequencyPlan`** (`freqplan.py`): A list of `(center_mhz, mode)` specs (mode ∈ {zoom, wide}). Builds the minimal deduplicated raw sim grid (`sim_freqs()`) and channelizes a raw spectrum to spectrometer channels (`channelize()`), reusing `SpectrometerResponse`. `channel_table` gives per-channel `nu`/`lambda2`/`dnu`, where `nu` is the response-weighted effective frequency. Supports per-mode `decimation` and response `support` truncation.
- **`pipeline.py`**: `simulate_channelized()` runs the FR sim on the plan grid and channelizes it (captures bandwidth depolarization), and evaluates the no-FR baseline at channel centers (smooth → cheaper).
- **`rmsynth.py`**: RM synthesis — `lambda2`, `faraday_resolution`, `max_scale`, `phi_grid`, `rmsf` (rotation-measure spread function), `faraday_spectrum` (`F(φ) = Σ w (Q+iU) e^{-2iφλ²}`). For wide λ² coverage the RMSF resolution is ~milli-rad/m², so pass an explicit `dphi` to `phi_grid` (the default oversampled grid is otherwise enormous).
- **`noise.py`**: Radiometer noise — `radiometer_sigma(T_sys, dnu, dt)`, `add_noise`.
- **`detection.py`**: `faraday_noise_std` (analytic Faraday-spectrum noise level) and `faraday_snr` (peak SNR).

Analysis scripts: `notebooks/grid_design.py` (builds/validates the full-band grid via the cheap RMSF; defines `fullband_specs()`), `notebooks/faraday_fullband_sim.py` (curated full-band sim driver → `results/faraday_fullband.npz`), `notebooks/rmsynth_analysis.py` (RM synthesis + detection significance), `notebooks/rmsynth_calibration.py` (calibration on the legacy 3-band sims).

## Key Conventions

- All sky maps use HEALPix RING ordering with default `nside=128`.
- Frequencies are in MHz throughout the codebase.
- Jones matrices have shape `(2, npix)` with axes `(Eth, Eph)`.
- Stokes maps have shape `(nfreq, npix)` or `(npix,)`.
- The LuSEE landing site is hardcoded at lat=-23.813°, lon=182.258° in `sky.py`.
- Beam FITS files are in `data/`; the beam is defined on a 1° theta/phi grid and interpolated to HEALPix.
- Zoom-bin channels are **FFT-ordered, not monotonic in frequency**; pair λ²/Stokes with channels via `FrequencyPlan.channel_table` (response-weighted `nu`), never `utils.freqs_zoom`.
- For the parallel full-band sim, pin BLAS threads (`OMP_NUM_THREADS=1`, etc.) and use one process per physical core (the driver sets this); otherwise the cos/sin-bound workers thrash.
- Black formatting with line-length 79.

## Data Files

Files in `data/` are required for realistic simulations but not tracked fully in git (large FITS/HDF5). Key files:
- `feko_bnl_3m_75deg.2port.fits`, `hfss_lbl_3m_75deg.2port.fits` — LuSEE beam models
- `wmap_band_iqumap_r9_9yr_K_v5.fits` — WMAP K-band polarization maps
- `faraday2020v2.hdf5` — Faraday rotation measure sky map
- `spectrometer_bin_response.txt` — spectrometer channel response
