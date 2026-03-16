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

## Key Conventions

- All sky maps use HEALPix RING ordering with default `nside=128`.
- Frequencies are in MHz throughout the codebase.
- Jones matrices have shape `(2, npix)` with axes `(Eth, Eph)`.
- Stokes maps have shape `(nfreq, npix)` or `(npix,)`.
- The LuSEE landing site is hardcoded at lat=-23.813°, lon=182.258° in `sky.py`.
- Beam FITS files are in `data/`; the beam is defined on a 1° theta/phi grid and interpolated to HEALPix.
- Black formatting with line-length 79.

## Data Files

Files in `data/` are required for realistic simulations but not tracked fully in git (large FITS/HDF5). Key files:
- `feko_bnl_3m_75deg.2port.fits`, `hfss_lbl_3m_75deg.2port.fits` — LuSEE beam models
- `wmap_band_iqumap_r9_9yr_K_v5.fits` — WMAP K-band polarization maps
- `faraday2020v2.hdf5` — Faraday rotation measure sky map
- `spectrometer_bin_response.txt` — spectrometer channel response
