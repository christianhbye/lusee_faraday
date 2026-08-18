# Cross-check branch setup (`croissant-crosscheck`)

Branched from `origin/luseepy-version` (slosar's four-port work) — that
branch is **purely additive** vs `main` (3807 insertions, 0 deletions,
no edits to `src/lusee_faraday/*.py`), so there were no conflicts.

## Toolchain actually in use

| Component | Location | Git state |
|---|---|---|
| luseepy | `../luseepy` | branch `main`, `61c439a` (release 2.0 + four-port refactor) |
| croissant | `../../../projects/croissant-main` (worktree) | branch `main`, `da01c5a`, clean |
| lusee_faraday | this repo | branch `croissant-crosscheck` |

Both are **editable** installs wired via `[tool.uv.sources]` in
`pyproject.toml`.  Reproduce with `uv sync`.

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

luseepy `main` pins `croissant-sim @ git+...@379496e`.  Our `main` is
**6 commits ahead** of that pin.  We deliberately override it
(`[tool.uv] override-dependencies`).  The behaviour-affecting commit in
that range is `529b874 fix: reject complex input under reality=True and
default it off` — an API default flip.  The rest add the
precomputed-kernel SHT engine (`eb392f9`, `da01c5a`) and full-Stokes
physical-invariant tests (`1c4b59e`).

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

- `main`'s `sim.py:compute_stokes` already has `U = 2*np.real(Rxy)` ✓
- `fourport.polarimeter` drops the ½ from its coherency and uses
  `I=(XX+YY)/2, Q=(XX−YY)/2, U=Re XY` — self-consistent ✓

Both codes are right; the paper equation needs the 2.

## Resource limits hit on this machine

- **Disk `/home` at 97% (12 G free)** — the fine waterfalls are ~2.1 GB
  per band. Check before generating all three bands.
- **RAM 15 GB total, ~4 GB free** — PROGRESS.md documents the OOM
  killer on the croissant dense transform. Run heavy jobs **serially**.
