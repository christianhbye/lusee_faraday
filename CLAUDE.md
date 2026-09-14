# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulator for LuSEE (Lunar Surface Electromagnetics Experiment) Faraday rotation observations. Computes the four-port covariance seen from the lunar surface, including Faraday rotation of synchrotron emission through the ionosphere and the Galactic screen.

### What this repository is

Two arms, with different jobs. Confusing them is the main way to go wrong here.

**The production arm** — `sky` + `response` + `engine` + `instrument` +
`polarimeter` + `channelization`, on luseepy and croissant. This is where new
work goes. It currently covers instrumental `I -> Q,U` leakage, the zenith
polarimeter calibration, and the transiting point source; the diffuse sky is
future work (the ensemble two-point prediction), not a gap to be filled by
re-running the refuted approach.

**The reproduction arm** — `pixel_arm.py`, an independent pixel-space
quadrature. Its job is *not* to be superseded code awaiting deletion. It makes
the 2026-08-18 audit checkable (you cannot show a published result is shot noise
without being able to produce it) and it is the correctness evidence for the
production arm, which is pinned against it at 2.5e-16, 0.0e+00 and 3.054e-4.
**Do not shrink it toward the production modules: the duplication is the
independence.**

What this repository is *not*: a home for the diffuse-Faraday results the audit
refutes. Those, and the write-up that reports them, live at the
`audit-2026-08-18` tag — along with `step2_real_sky.py`, `step4_power_spectra.py`,
`step2_plots.py` and `report/report.tex`. Cite the tag, not a branch.

The physics is **not** owned here. Instrument response, impedances, receiver loading and covariance assembly all come from `luseepy`; the 16-channel packing is a local loop over `PORT_PAIRS` that is pinned to `lusee.Covariance.pack_covariance` by test rather than delegated to it (the two differ only in where the channel axis sits); spherical transforms and the polarized harmonic dual come from `croissant`. This repository owns the layer above: how a Faraday-rotated sky enters that formalism, and how the results are channelized and read out. Read `docs/measurement-model.md` first — it is the conceptual overview and the reason the 16,384-channel fine grid is affordable.

## Setup & Commands

```bash
# Install (uses uv with a .venv already present)
uv sync --extra dev          # editable luseepy + croissant via [tool.uv.sources]
                             # the extra is NOT optional: plain `uv sync` prunes
                             # pytest-cov, and addopts passes --cov=src always
uv add <package>             # NEVER `uv pip install`

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_name -v

# Format / lint
uv run black src/ tests/     # line-length 79
uv run flake8 src/
```

`JAX_ENABLE_X64=1` must be set **before any jax import**, or croissant and
luseepy silently drop to complex64. `scripts/common.py` does this with an
`os.environ.setdefault` above every other import and every script imports it;
test modules set it themselves at the top, and `tests/conftest.py` sets it as a
backstop.

Heavy jobs run in the background under `ulimit -v 16000000` with **absolute**
log paths under `generated_data/`. 12 GB is not enough — three of the zenith
tests OOM inside jax.

## Architecture

The pipeline flows: **Sky components → Faraday coefficients → harmonic contraction → luseepy covariance → polarimeter → channelization**

- **`FaradaySky`** (`sky.py`): the sky as a sum of constant-Faraday-depth components, each a frequency-independent alm plus a per-frequency, per-block coefficient. Constructors: `from_maps`, `uniform_screen`, `point_source`, `i_only` (perfect depolarization), `binned_screen` and `from_rm_map`. The last two are the only ones that turn a map of Faraday depths into components, and `binned_screen` — which `from_rm_map` merely wraps — runs `sky.audit_screen`: it logs both audit numbers on every build and refuses an unresolved screen unless the caller passes `allow_pixelwise=True`. The 2026-08-18 pixelization audit lives in the API, not in a paragraph.
- **`response.py`**: instrument model → pair-Stokes alms. `load_response` reads a BGL_v16 artifact through `lusee.InstrumentResponse`; `four_port_pair_alms` is the as-built arm, `two_port_pair_alms` the symmetric pseudo-dipole (paper Fig. 4) arm through croissant. `FixedChannelKernel` slices ONE native channel and samples many directions out of it — luseepy's `pair_stokes_at` re-materializes all 150 channels (2.94 GB) per call and is scalar-only, so this is a real capability, not a wrapper.
- **`engine.py`**: the block-resolved contraction of sky duals against response duals, and the spectral expansion onto the fine grid.
- **`instrument.py`**: covariance assembly, receiver loading and Hermitian projection — all luseepy. The 16-channel packing is *not*: `channels` is a local loop, and `test_channels_match_luseepy_pack_covariance` pins it elementwise against `lusee.Covariance.pack_covariance` (which stacks the channel axis at `-2`, so the pin transposes). `impedance_freq_mhz` freezes `Z_A`, `Z_L`, `R_moon`, `R_loss` at one frequency; a Faraday run **must** pass it, and must pass `T_moon=0.0, T_ant=0.0` where the legacy assembler had no thermal terms.
- **`polarimeter.py`**: zenith calibration (`zenith_port_weights`, `orthonormalize_xy`) and pseudo-Stokes. `check_psd` is a runtime invariant, not only a test.
- **`channelization.py`**: parent (25 kHz) and zoom (64 sub-bin) integration on luseepy's spectrometer response. Zoom bins use FFT ordering (0 = center, 1–32 positive, 33–63 negative).
- **`conventions.py`**, **`config.py`**: the single source of truth for COSMO/IAU, the Faraday phase, port and channel ordering, the site, the time grid and the fine frequency grid. Do not re-derive any of it inline.

**`pixel_arm.py` is the reproduction arm** (see "What this repository is"). Import it only from the reproduction and cross-check scripts — `crosscheck_pixel_arm.py`, `validate_engine.py`, `beam_ablation.py`, `compare_main_vs_asbuilt.py`, `step_ionly.py --engine legacy` — and from tests that compare the two arms. `scripts/common.py` deliberately does not import it, so an ordinary script does not depend on it transitively; `topo_rotation_matrix` lives in `conventions.py` for that reason, since it defines the response frame rather than either engine's quadrature.

## Key Conventions

- Frequencies are in MHz throughout.
- Ports `0, 1, 2, 3 = N, E, S, W`; 16 real channels ordered as `lusee.Covariance.default_product_labels()`.
- Input sky Q/U are healpy/COSMO; croissant consumes IAU (`U_IAU = -U_COSMO`). Faraday: `(Q + iU)_COSMO * exp(+2i phi lambda^2)`.
- Response frame: `x = East, y = North, z = zenith`; grid `phi = 90° - azimuth`.
- Real sky maps are used at native `nside = 512` RING and never degraded: per-pixel Faraday phases do not commute with `ud_grade`.
- The fixed-beam approximation covers the receiver loading too — see `docs/measurement-model.md` §6.
- The LuSEE landing site is in `config.py` (`LUN_LAT_DEG`, `LUN_LONG_DEG`).
- Black formatting with line-length 79.

## Data Files

Files in `data/` are required for realistic simulations but not tracked in git (large FITS/HDF5). Key files:
- `BGL_v16/lusee_bgl_v16_response_v3.fits` — the as-built four-port response (631 MB); `_c4sym` and `_diagza` variants for the ablations. Override with `$LUSEE_RESPONSE`.
- `haslam408_dsds_Remazeilles2014.fits` — RING ordered, K
- `wmap_band_iqumap_r9_9yr_K_v5.fits` — NESTED, mK thermodynamic
- `faraday2020v2.hdf5` — Faraday depth map, RING, rad/m²

Tests that need the 631 MB artifact are marked `slow` and skip without it.

## See also

- `docs/measurement-model.md` — what is being computed and why it is cheap
- `AGENTS.md` — the pinned conventions in operational form, plus the script inventory
- `PROGRESS.md` — running status
- the `audit-2026-08-18` tag — Slosar's four-port analysis, the diffuse-Faraday
  results the audit refutes, and `report.tex`. Not carried on `main`.
