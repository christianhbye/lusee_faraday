# Toolchain setup and cross-check notes

Originally written for the `croissant-crosscheck` branch, which was
branched from `origin/luseepy-version` (slosar's four-port work) — that
branch was **purely additive** vs `main` (3807 insertions, 0 deletions,
no edits to `src/lusee_faraday/*.py`), so there were no conflicts. The
work now lives on `luseepy-refactor`.

## Toolchain actually in use

| Component | Location | Git state |
|---|---|---|
| luseepy | `../luseepy` | branch `deps/croissant-v5.3.0.dev1`, `52b96bc` ("deps: bump croissant-sim to v5.3.0.dev1") |
| croissant | `/home/christian/Documents/projects/croissant-main` (worktree) | **detached** at `1c4d6c5` = `v5.3.0.dev0-15-g1c4d6c5` ("feat: close out the kernel engine's trace, cache and reality gaps (#143)"), clean |
| lusee_faraday | this repo | branch `luseepy-refactor` |

Both are **editable** installs wired via `[tool.uv.sources]` in
`pyproject.toml`.  Reproduce with `uv sync --extra dev`; add packages with
`uv add`.  Plain `uv sync` prunes the `dev` extra (black, flake8,
`pytest-cov`), and `pyproject.toml`'s `addopts` passes `--cov=src`
unconditionally, so the next command would fail on an unrecognized argument.
The croissant worktree was created with `git worktree add --detach`, so
it is **not** on a branch: `git worktree list` shows
`croissant-main  1c4d6c5 (detached HEAD)` while `main` sits at `0ac2f86`
in the sibling `croissant` checkout.  Checking out `main` would *not*
reproduce this environment.  The pinned SHA is the load-bearing half.

Verified 2026-08-19 (final fix round).  Earlier revisions of this file
recorded luseepy at `main`/`61c439a` and croissant at `da01c5a`, and
recorded the worktree as being "on `main`"; all were stale.

**Neither checkout may be modified from this repo.**

### Why a croissant worktree and not `../../../projects/croissant`

That checkout was on branch `kernel-engine-cleanup` (= `main` + 2 WIP
commits) **with uncommitted edits to `src/croissant/kernel.py` and
`tests/test_kernel_engine.py`** — i.e. dirty WIP in the SHT kernel
engine, which is the exact code path `FullStokesCroSimulator` uses for
the polarized transform.  Validating the four-port pixel engine against
that would make a disagreement unattributable.  So:

    git worktree add ../../../projects/croissant-main main

leaves the WIP untouched and gives the cross-check a clean, pushed
reference.  `croissant.__version__` reports `5.2.1` because the
pyproject version field lags the tags — **use git state, not the
version string**.

### croissant vs luseepy's pin

The luseepy we actually use is not on `main`.  On branch
`deps/croissant-v5.3.0.dev1` (`../luseepy/pyproject.toml:26`) it pins
`croissant-sim @ git+...@v5.3.0.dev1`, and that tag resolves to commit
`0ac2f86`.  Our worktree is at `1c4d6c5`, i.e. **two commits behind**
that pin, not ahead of it.  We deliberately override the pin
(`[tool.uv] override-dependencies`).  The two commits we are behind are
`ab915fe feat: report polarized visibilities in sky-temperature units
(#144)` — which adds `polarized_convolve(normalization="auto-I")` and
explicitly leaves `normalization` at `None` by default, touching no
existing physics test — and `0ac2f86`, a changelog.  Numerical risk
today is therefore nil.

*Historical, for the record:* an older luseepy `main` pinned
`croissant-sim @ git+...@379496e`, and `1c4d6c5` is 7 commits ahead of
*that*.  The behaviour-affecting commit in that range is `529b874 fix:
reject complex input under reality=True and default it off` — an API
default flip.  The rest add the precomputed-kernel SHT engine
(`eb392f9`, `da01c5a`), an amortisation recalibration (`537d8eb`) and
full-Stokes physical-invariant tests (`1c4b59e`).  A previous revision
of this paragraph said "our `main` is 6 commits ahead of that pin",
which was both miscounted and about the wrong pin.

### The `engine="auto"` dense-memory trap is closed at `1c4d6c5`

An earlier session recorded that croissant's `engine="auto"` forced the
dense spherical transform whenever `lmax < 3*nside - 1`, which at
nside=512 / lmax=30 would build an ~800 GB operator. croissant `1c4d6c5`
added a memory cap to `_low_pass_in_one_step`
(`croissant/polarization.py:142`), so the low-lmax / high-nside
polarized transform now takes the native transform and truncates.

**Confirmed empirically, not assumed** (Task 1 of the 2026-08-18
refactor, `scripts/probe_toolchain.py`):
`PolarizedSky(nside=512).compute_alm(lmax=30)` resolves to the `s2fft`
engine, not `dense`, and peaks at ~3.7 GB — comfortably inside the
16 GB cap. Re-run the probe if the croissant pin moves.

## Data

`data/` is gitignored.  Files needed:

| File | Status |
|---|---|
| `BGL_v16/lusee_bgl_v16_response_v3.fits` | as-built, asymmetric, coupled — **the baseline slosar used** |
| `BGL_v16/lusee_bgl_v16_response_v3_c4sym.fits` | C4 group-averaged = the paper's 90°-rotation assumption, made self-consistent |
| `BGL_v16/lusee_bgl_v16_response_v3_diagza.fits` | ZA-diagonalised = inter-port coupling removed |
| `haslam408_dsds_Remazeilles2014.fits` | RING ordered (not NEST) |
| `wmap_band_iqumap_r9_9yr_K_v5.fits`, `faraday2020v2.hdf5` | already present |

`scripts/common.py:RESPONSE_PATH` defaults to the as-built model and is
overridable with `$LUSEE_RESPONSE`, so the ablations run without edits.

## Convention note for the paper (not a code bug)

Requiring an ideal zenith polarimeter to return the source's true
Stokes exactly fixes all normalizations.  With the paper's
`T = ½[[I+Q, U+iV],[U-iV, I-Q]]` and `V_pq = ∫ J_p T J_q^H`:

    V_XX + V_YY = I  ✓      V_XX − V_YY = Q  ✓      Re V_XY = U/2

So the paper's `U_obs = Re(V_XY)` (sec:lusee) **is missing a factor 2**;
it should read `U_obs = 2 Re V_XY`, `V_obs = 2 Im V_XY`.

- `main`'s `sim.py:compute_stokes` had `U = 2*np.real(Rxy)` ✓ (that
  module was retired in the 2026-08-18 refactor)
- `polarimeter.pseudo_stokes` drops the ½ from its coherency and uses
  `I=(XX+YY)/2, Q=(XX−YY)/2, U=Re XY` — self-consistent ✓

Both codes are right; the paper equation needs the 2.

## Resource limits hit on this machine

- **Disk `/home` at 82% (56 G free, 2026-08-19)** — the fine waterfalls are ~2.1 GB
  per band. Check before generating all three bands.
- **RAM 15 GB total, ~4 GB free** — PROGRESS.md documents the OOM
  killer on the croissant dense transform. Run heavy jobs **serially**.
