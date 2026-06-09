# Faraday Fisher-Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast LuSEE's *realistic* detection significance for the Faraday-rotation amplitude by marginalizing over an unknown large-scale intrinsic polarized sky, turning the optimistic `SNR_far` upper bound into a sky-marginalized SNR.

**Architecture:** The channelized polarized data `P = pQ + i·pU(t, channel)` is **linear in the intrinsic (Q, U) sky** (beam, horizon, coordinate rotation are known linear operators) and depends on Faraday rotation only through `cos/sin(2·α·RM·λ²)`. We wrap the existing `compute_vis_fast` into a reusable complex forward operator `pol_response`, build a Fisher Jacobian `J` by (a) applying it to low-ℓ spin-2 spherical-harmonic basis sky maps (sky-nuisance columns), (b) finite-differencing `α` (the Faraday-amplitude detection parameter), and (c) an analytic effective-dispersion column `τ`. The sky-marginalized error `σ(α) = sqrt[(F⁻¹)_{αα}]` with `F = JᵀN⁻¹J` gives `SNR = α_fid/σ(α)`. **No new expensive sim** — all forward evals are at channel centers (intra-channel bandwidth depolarization is negligible across a 25 kHz channel relative to the 5–51 MHz band), reusing `results/faraday_fullband.npz` for `nu/λ²/dnu` and `pI_FR` (≈ T_sys).

**Tech Stack:** Python, numpy, healpy (spin-2 SHTs), existing `lusee_faraday` modules (`fast_sim`, `sim.Simulator.compute_stokes`, `freqplan`, `noise`, `sky`, `beam`). pytest + black (line-length 79). Branch: `faraday-fisher-forecast` (already created off `faraday-rmsynth`).

---

## Background the implementer needs

The forward kernel is `lusee_faraday.fast_sim.compute_vis_fast`. Its polarized term, per time `i` and frequency `f`, is
`scale_QU(f) · [cos(2·RM·λ²) @ A + sin(2·RM·λ²) @ B]` with `A = wQ·Q + wU·U`, `B = wU·Q − wQ·U` over above-horizon pixels — **linear in the reference `(Q, U)` maps**. `Simulator.compute_stokes(vis)` (static, `sim.py:101`) maps the `(ntimes, 3, nfreq)` visibility cube to `I, Q, U` each `(ntimes, nfreq)`.

Consequences used throughout:
- `∂P/∂(sky mode)` = `pol_response(basis Q/U) − pol_response(0)` (linearity; the `pol_response(0)` baseline removes the α-independent Stokes-I→pol leakage).
- `∂P/∂α` via central finite difference (one scalar parameter; the I-leakage is α-independent and cancels in the difference).
- `∂P/∂τ` for a depolarization factor `exp(−2τλ⁴)` is analytic: `−2·(λ²)²·P_pol_fid` (note: the first derivative w.r.t. `σ_RM` vanishes at 0, so the nuisance is the **Faraday variance** `τ ≡ σ_RM²`, whose derivative is non-zero).
- A Fisher forecast needs only `J` and `N`, **not** the data vector — so we never need a noiseless "truth" run, only the fiducial sky (WMAP Q/U) as the linearization point and `T_sys ≈ pI_FR`.

Noise: `noise.radiometer_sigma(T_sys, dnu_hz, dt_s)` gives per-channel σ; `pQ` and `pU` each carry independent noise σ, so the real data vector stacks `[Re(P), Im(P)]` and the inverse-variance weight is `1/σ²` repeated for both quadratures.

## File Structure

- **Create** `src/lusee_faraday/forward.py` — `rotate_pol_maps`, `pol_response`. Reusable linear forward operator. One responsibility: map an intrinsic topo sky + α to complex channel response.
- **Create** `src/lusee_faraday/skybasis.py` — `n_modes`, `spin2_basis`. Generates the low-ℓ spin-2 (Q,U) nuisance basis.
- **Create** `src/lusee_faraday/fisher.py` — `stack_real`, `fisher_matrix`, `marginal_error`, `detection_snr`, `faraday_column`, `dispersion_column`, `run_forecast`. Fisher assembly + marginalization.
- **Create** `notebooks/fisher_forecast.py` — driver: reconstruct sim inputs, rotate fiducial + basis, call `run_forecast` per integration time, compare marginalized vs fixed-sky SNR, save `results/fisher_forecast.png`.
- **Create** `tests/test_forward.py`, `tests/test_skybasis.py`, `tests/test_fisher.py`.
- **Modify** `src/lusee_faraday/__init__.py` — expose `forward`, `skybasis`, `fisher`.
- **Modify** `CLAUDE.md`, `README.md` — document the Fisher-forecast layer.

---

### Task 1: Forward operator (`forward.py`)

**Files:**
- Create: `src/lusee_faraday/forward.py`
- Test: `tests/test_forward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forward.py
import numpy as np

from lusee_faraday.beam import Beam
from lusee_faraday.forward import pol_response


def _setup(nside=8, ntimes=2, nfreq=3, seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    I = rng.normal(size=(ntimes, npix)) + 100.0
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = rng.normal(size=(ntimes, npix)) * 5.0
    beam = Beam.short_dipole(nside=nside)
    beam.precompute_weights()
    mask = np.ones(npix, dtype=bool)
    freqs = np.linspace(10.0, 50.0, nfreq)
    return I, Q, U, rm, beam, mask, freqs


def test_pol_response_shape_and_complex():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P = pol_response(I, Q, U, rm, beam, mask, freqs)
    assert P.shape == (I.shape[0], freqs.size)
    assert np.iscomplexobj(P)


def test_pol_response_linear_in_QU():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P0 = pol_response(I, 0 * Q, 0 * U, rm, beam, mask, freqs)
    P1 = pol_response(I, Q, U, rm, beam, mask, freqs)
    P2 = pol_response(I, 2 * Q, 2 * U, rm, beam, mask, freqs)
    np.testing.assert_allclose(P2 - P0, 2 * (P1 - P0), rtol=1e-10, atol=1e-10)


def test_alpha_zero_is_unrotated():
    I, Q, U, rm, beam, mask, freqs = _setup()
    P_a0 = pol_response(I, Q, U, rm, beam, mask, freqs, alpha=0.0)
    P_rm0 = pol_response(I, Q, U, 0 * rm, beam, mask, freqs, alpha=1.0)
    np.testing.assert_allclose(P_a0, P_rm0, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lusee_faraday.forward'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lusee_faraday/forward.py
"""Reusable linear forward operator for the Fisher forecast.

compute_vis_fast is linear in the reference (Q, U) maps, and Faraday
rotation enters only through cos/sin(2*alpha*RM*lambda^2). pol_response
wraps it into a single complex polarized response P = pQ + i*pU as a
function of an intrinsic (I, Q, U) topocentric sky and a Faraday
amplitude alpha, so Fisher Jacobian columns can be built by applying it
to basis sky maps (sky derivatives) and by perturbing alpha.
"""

import healpy as hp
import numpy as np

from .fast_sim import compute_vis_fast, precompute_rotated_maps
from .sim import Simulator


def rotate_pol_maps(Q_ref, U_ref, rm_gal, times, nside, site_loc):
    """Rotate a polarized (Q, U) reference sky + RM to topocentric.

    Thin wrapper over precompute_rotated_maps with a zero I map, for
    building Jacobian columns from basis Q/U maps. Returns
    (Q_topo, U_topo, rm_topo), each shape (ntimes, npix).
    """
    zero = np.zeros(hp.nside2npix(nside))
    _, Q_topo, U_topo, rm_topo = precompute_rotated_maps(
        zero, Q_ref, U_ref, rm_gal, times, nside, site_loc
    )
    return Q_topo, U_topo, rm_topo


def pol_response(
    I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs, alpha=1.0, **kwargs
):
    """Complex polarized response P = pQ + i*pU at given frequencies.

    Linear in (Q_topo, U_topo). alpha scales the rotation measure
    (alpha=1 is the physical Faraday sky, alpha=0 unrotated). Extra
    kwargs forward to compute_vis_fast. Returns complex (ntimes, nfreq).
    """
    vis = compute_vis_fast(
        I_topo, Q_topo, U_topo, alpha * np.asarray(rm_topo),
        beam, freqs, mask, **kwargs,
    )
    _, Q, U = Simulator.compute_stokes(vis)
    return Q + 1j * U
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_forward.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/forward.py tests/test_forward.py
git add src/lusee_faraday/forward.py tests/test_forward.py
git commit -m "feat(forward): linear pol_response operator for Fisher forecast"
```

---

### Task 2: Spin-2 sky basis (`skybasis.py`)

**Files:**
- Create: `src/lusee_faraday/skybasis.py`
- Test: `tests/test_skybasis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skybasis.py
import healpy as hp
import numpy as np

from lusee_faraday.skybasis import n_modes, spin2_basis


def test_n_modes_counts():
    # per l: E and B; m=0 (1 part) + m=1..l (2 parts each) = (1+2l) parts
    assert n_modes(2) == 2 * (1 + 4)          # l=2 only -> 10
    assert n_modes(3) == 2 * (1 + 4) + 2 * (1 + 6)  # +l=3 -> 24


def test_basis_shape_real_nonzero():
    nside = 8
    basis = spin2_basis(nside, lmax=2)
    assert len(basis) == n_modes(2)
    npix = hp.nside2npix(nside)
    for label, Q, U in basis:
        assert isinstance(label, str)
        assert Q.shape == (npix,) and U.shape == (npix,)
        assert np.isrealobj(Q) and np.isrealobj(U)
        assert np.any(Q != 0) or np.any(U != 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skybasis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lusee_faraday.skybasis'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lusee_faraday/skybasis.py
"""Low-l spin-2 (Q, U) spherical-harmonic basis for the intrinsic
polarized-sky nuisance in the Fisher forecast.

Each element is a real (Q, U) map pair from a unit E- or B-mode a_lm.
These span the large-scale intrinsic polarization the beam can
constrain; their amplitudes are the marginalized nuisance parameters.
Absolute normalization is irrelevant: the marginalized sigma(alpha)
depends only on the subspace the basis spans.
"""

import healpy as hp
import numpy as np


def n_modes(lmax):
    """Number of real spin-2 basis maps for 2 <= l <= lmax."""
    return sum(2 * (1 + 2 * L) for L in range(2, lmax + 1))


def spin2_basis(nside, lmax):
    """List of (label, Q_map, U_map) real basis elements (RING)."""
    nalm = hp.Alm.getsize(lmax)
    T = np.zeros(nalm, dtype=complex)
    basis = []
    for L in range(2, lmax + 1):
        for M in range(L + 1):
            idx = hp.Alm.getidx(lmax, L, M)
            parts = ("re",) if M == 0 else ("re", "im")
            for mode in ("E", "B"):
                for part in parts:
                    alm = np.zeros(nalm, dtype=complex)
                    alm[idx] = 1.0 if part == "re" else 1.0j
                    if mode == "E":
                        eb = [T, alm, np.zeros(nalm, dtype=complex)]
                    else:
                        eb = [T, np.zeros(nalm, dtype=complex), alm]
                    _, Q, U = hp.alm2map(eb, nside, lmax=lmax, pol=True)
                    basis.append((f"{mode}_{L}_{M}_{part}", Q, U))
    return basis
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_skybasis.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/skybasis.py tests/test_skybasis.py
git add src/lusee_faraday/skybasis.py tests/test_skybasis.py
git commit -m "feat(skybasis): low-l spin-2 Q/U nuisance basis"
```

---

### Task 3: Fisher primitives (`fisher.py` core)

**Files:**
- Create: `src/lusee_faraday/fisher.py`
- Test: `tests/test_fisher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fisher.py
import numpy as np

from lusee_faraday.fisher import (
    detection_snr,
    fisher_matrix,
    marginal_error,
    stack_real,
)


def test_stack_real_layout():
    P = np.array([[1 + 2j, 3 + 4j]])
    np.testing.assert_array_equal(stack_real(P), [1, 3, 2, 4])


def test_orthogonal_nuisance_does_not_inflate():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)        # real quadrature
    nuisance = 1j * np.ones((1, n), dtype=complex)  # imag quadrature
    F = fisher_matrix([signal, nuisance], sig)
    F_only = fisher_matrix([signal], sig)
    # orthogonal -> marginalized error equals unmarginalized
    assert np.isclose(marginal_error(F, 0), marginal_error(F_only, 0))
    assert np.isclose(marginal_error(F_only, 0), 1.0 / np.sqrt(n))


def test_degenerate_nuisance_inflates_error():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)
    near_parallel = (1.0 + 1e-3 * 1j) * np.ones((1, n), dtype=complex)
    F = fisher_matrix([signal, near_parallel], sig)
    F_only = fisher_matrix([signal], sig)
    assert marginal_error(F, 0) > 10 * marginal_error(F_only, 0)
    # marginalizing can only reduce SNR
    assert detection_snr(F, 0) < detection_snr(F_only, 0)


def test_sigma_scaling():
    n = 8
    signal = np.ones((1, n), dtype=complex)
    F1 = fisher_matrix([signal], np.ones((1, n)))
    F2 = fisher_matrix([signal], 2 * np.ones((1, n)))
    assert np.isclose(marginal_error(F2, 0), 2 * marginal_error(F1, 0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fisher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lusee_faraday.fisher'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lusee_faraday/fisher.py
"""Fisher-matrix detection forecast for the Faraday amplitude alpha.

Data: complex polarized channels P = pQ + i*pU over (time, channel) with
independent Gaussian noise sigma on pQ and pU. Parameters: a Faraday
amplitude alpha (detection target, fiducial 1), intrinsic-sky nuisance
amplitudes (spin-2 harmonic modes), and one effective Faraday-dispersion
nuisance tau (Faraday variance). The sky-marginalized error sigma(alpha)
gives the realistic detection SNR = alpha_fid / sigma(alpha). A Fisher
forecast needs only J and N, never the data vector.
"""

import numpy as np

from .forward import pol_response


def stack_real(P):
    """Flatten complex (ntimes, nchan) to real vector [Re..., Im...]."""
    P = np.asarray(P)
    return np.concatenate([P.real.ravel(), P.imag.ravel()])


def fisher_matrix(columns, sigma):
    """F_ij = sum Re(dP_i* dP_j)/sigma^2.

    columns: list of complex (ntimes, nchan) derivative arrays.
    sigma: real (ntimes, nchan) per-channel Stokes noise (pQ and pU
    share it). Returns (nparam, nparam) real array.
    """
    sig = np.asarray(sigma, dtype=float)
    w = np.concatenate([1.0 / sig.ravel() ** 2] * 2)  # Re + Im quadratures
    J = np.column_stack([stack_real(c) for c in columns])
    return J.T @ (w[:, None] * J)


def marginal_error(F, idx, rcond=1e-12):
    """Marginalized 1-sigma error on parameter idx (others free)."""
    Cinv = np.linalg.pinv(F, rcond=rcond)
    return float(np.sqrt(Cinv[idx, idx]))


def detection_snr(F, idx, alpha_fid=1.0, rcond=1e-12):
    """Detection SNR = alpha_fid / marginalized sigma(alpha)."""
    return alpha_fid / marginal_error(F, idx, rcond=rcond)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fisher.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/fisher.py tests/test_fisher.py
git add src/lusee_faraday/fisher.py tests/test_fisher.py
git commit -m "feat(fisher): Fisher matrix, marginalized error, detection SNR"
```

---

### Task 4: Jacobian columns + forecast assembly (`fisher.py`)

**Files:**
- Modify: `src/lusee_faraday/fisher.py`
- Test: `tests/test_fisher.py` (add)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fisher.py  (append)
from lusee_faraday.beam import Beam
from lusee_faraday.fisher import (
    dispersion_column,
    faraday_column,
    run_forecast,
)


def _toy(nside=8, ntimes=2, nfreq=3, seed=1):
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    I = rng.normal(size=(ntimes, npix)) + 100.0
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = rng.normal(size=(ntimes, npix)) * 5.0
    beam = Beam.short_dipole(nside=nside)
    beam.precompute_weights()
    mask = np.ones(npix, dtype=bool)
    freqs = np.linspace(10.0, 50.0, nfreq)
    lam2 = (3e8 / (freqs * 1e6)) ** 2
    basis = [(rng.normal(size=npix), rng.normal(size=npix)) for _ in range(2)]
    return I, Q, U, rm, beam, mask, freqs, lam2, basis


def test_faraday_and_dispersion_columns_shape():
    I, Q, U, rm, beam, mask, freqs, lam2, _ = _toy()
    a = faraday_column(I, Q, U, rm, beam, mask, freqs)
    assert a.shape == (I.shape[0], freqs.size) and np.iscomplexobj(a)
    P_pol = np.ones((I.shape[0], freqs.size), dtype=complex)
    t = dispersion_column(P_pol, lam2)
    np.testing.assert_allclose(t, -2 * lam2[None, :] ** 2 * P_pol)


def test_run_forecast_marginalized_le_fixed():
    I, Q, U, rm, beam, mask, freqs, lam2, basis = _toy()
    sigma = np.ones((I.shape[0], freqs.size))
    out = run_forecast(
        I, Q, U, rm, basis, beam, mask, freqs, lam2, sigma
    )
    assert np.isfinite(out["snr"]) and out["snr"] > 0
    # marginalizing the sky+tau cannot increase the SNR
    assert out["snr"] <= out["snr_opt"] * (1 + 1e-9)
    assert out["n_modes"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fisher.py -v`
Expected: FAIL with `ImportError: cannot import name 'faraday_column'`

- [ ] **Step 3: Write minimal implementation (append to `fisher.py`)**

```python
# src/lusee_faraday/fisher.py  (append)


def faraday_column(
    I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs,
    alpha_fid=1.0, dalpha=1e-3, **kwargs
):
    """dP/dalpha via central finite difference at alpha_fid."""
    Pp = pol_response(
        I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs,
        alpha=alpha_fid + dalpha, **kwargs,
    )
    Pm = pol_response(
        I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs,
        alpha=alpha_fid - dalpha, **kwargs,
    )
    return (Pp - Pm) / (2 * dalpha)


def dispersion_column(P_pol_fid, lam2):
    """dP/dtau at tau=0 for depolarization exp(-2 tau lam2^2):
    -2 (lam2)^2 * P_pol_fid (tau is the Faraday variance)."""
    lam2 = np.asarray(lam2, dtype=float)
    return -2.0 * lam2[None, :] ** 2 * P_pol_fid


def run_forecast(
    I_topo, Q_topo, U_topo, rm_topo, basis_topo, beam, mask, freqs, lam2,
    sigma, alpha_fid=1.0, dalpha=1e-3, **kwargs
):
    """Sky-marginalized Faraday detection forecast.

    basis_topo: list of (Q_basis_topo, U_basis_topo) rotated nuisance
    maps. Returns dict with sigma_alpha / snr (sky+tau marginalized) and
    sigma_alpha_opt / snr_opt (sky+tau fixed -> optimistic bound).
    """
    zeroQ = np.zeros_like(Q_topo)
    P0 = pol_response(
        I_topo, zeroQ, zeroQ, rm_topo, beam, mask, freqs,
        alpha=alpha_fid, **kwargs,
    )
    P_fid = pol_response(
        I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs,
        alpha=alpha_fid, **kwargs,
    )
    P_pol_fid = P_fid - P0  # polarized sky part only (I-leakage removed)

    a_col = faraday_column(
        I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs,
        alpha_fid=alpha_fid, dalpha=dalpha, **kwargs,
    )
    t_col = dispersion_column(P_pol_fid, lam2)
    mode_cols = [
        pol_response(
            I_topo, Qb, Ub, rm_topo, beam, mask, freqs,
            alpha=alpha_fid, **kwargs,
        ) - P0
        for Qb, Ub in basis_topo
    ]

    cols = [a_col, t_col] + mode_cols  # alpha is index 0
    F = fisher_matrix(cols, sigma)
    F_opt = fisher_matrix([a_col], sigma)
    sig_a = marginal_error(F, 0)
    sig_a_opt = marginal_error(F_opt, 0)
    return {
        "sigma_alpha": sig_a,
        "snr": alpha_fid / sig_a,
        "sigma_alpha_opt": sig_a_opt,
        "snr_opt": alpha_fid / sig_a_opt,
        "n_modes": len(basis_topo),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fisher.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Format and commit**

```bash
uv run black src/lusee_faraday/fisher.py tests/test_fisher.py
git add src/lusee_faraday/fisher.py tests/test_fisher.py
git commit -m "feat(fisher): Jacobian columns + sky-marginalized run_forecast"
```

---

### Task 5: Package exports

**Files:**
- Modify: `src/lusee_faraday/__init__.py`
- Test: `tests/test_fisher.py` (add import smoke)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fisher.py  (append)
def test_package_exposes_modules():
    import lusee_faraday as ld

    assert hasattr(ld, "forward")
    assert hasattr(ld, "skybasis")
    assert hasattr(ld, "fisher")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fisher.py::test_package_exposes_modules -v`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Write minimal implementation**

Open `src/lusee_faraday/__init__.py`, find the existing module imports (the line that imports `detection`), and add `forward`, `skybasis`, `fisher` alongside it. If imports are of the form `from . import detection`, add:

```python
from . import forward
from . import skybasis
from . import fisher
```

If the file uses an `__all__` list, append `"forward"`, `"skybasis"`, `"fisher"` to it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fisher.py::test_package_exposes_modules -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lusee_faraday/__init__.py tests/test_fisher.py
git commit -m "feat: expose forward/skybasis/fisher in package init"
```

---

### Task 6: Forecast driver + figure (`notebooks/fisher_forecast.py`)

**Files:**
- Create: `notebooks/fisher_forecast.py`

This is a driver script (no unit test; validated by running). It reconstructs the same inputs as `notebooks/faraday_fullband_sim.py`, reuses `results/faraday_fullband.npz` for the channel table + `pI_FR` (≈ T_sys), rotates the fiducial sky and each basis mode, and runs `run_forecast`.

**Implementation notes (deviations from the draft below, applied during execution for tractability/correctness):**
- **`NSIDE = 64`** forward-model resolution (a documented knob), not 128 — at 128 each `pol_response` pass was ~minutes and the held basis maps thrashed memory; 1° pixels resolve the beam + low-ℓ modes fine. All sky inputs load/`ud_grade` at `NSIDE` (incl. `load_rm(..., nside=NSIDE)`).
- **Exact `√dt` noise scaling**, not a per-dt rebuild: the Jacobian is integration-time-independent and radiometer noise gives `F ∝ dt` exactly, so `run_forecast` runs **once** at a reference time and `SNR(dt) = SNR_ref·√(dt/dt_ref)`. Eliminates a 3× redundant Jacobian build with zero approximation.
- **Figure is local-only** (`results/` is gitignored repo-wide; no figures are tracked) — commit the driver only; the script regenerates `fisher_forecast.png`.

- [ ] **Step 1: Write the driver**

```python
# notebooks/fisher_forecast.py
"""Step-4 Fisher forecast: sky-marginalized Faraday detection SNR.

Reuses results/faraday_fullband.npz (channel nu/lambda2/dnu, pI_FR as
T_sys) and the full-band sim setup. Rotates the WMAP fiducial sky and a
low-l spin-2 nuisance basis to topocentric, builds the Fisher matrix
over (alpha, tau, sky modes), and reports the marginalized detection SNR
vs the fixed-sky optimistic bound across integration times.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lunarsky import Time, MoonLocation

import lusee_faraday as ld
from lusee_faraday.fast_sim import precompute_rotated_maps
from lusee_faraday.forward import rotate_pol_maps
from lusee_faraday.fisher import run_forecast
from lusee_faraday.skybasis import spin2_basis
from lusee_faraday.noise import radiometer_sigma
from lusee_faraday.sky import LUSEE_LOC

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
NSIDE = 128
N_TIMES = 100
LMAX = 3            # spin-2 sky-nuisance bandlimit (24 modes)
DECIM = 8           # forecast channel decimation; DECIM=1 = full 4047
DT_CASES = [(40.0, "40 s"), (600.0, "10 min"), (3600.0, "1 h")]
BEAM_FILE = DATA / "hfss_lbl_3m_75deg.2port.fits"


def main():
    d = np.load(RES / "faraday_fullband.npz")
    # Decimate channels for the forecast: ~28 channel-center pol_response
    # evals over all 4047 channels is ~30 min; a strided subset spans the
    # same lambda^2 range and keeps the forecast tractable. DECIM=1 runs
    # the full channel set.
    sl = slice(None, None, DECIM)
    nu, lam2, dnu = d["nu"][sl], d["lambda2"][sl], d["dnu"][sl]
    pI_FR = d["pI_FR"][:, sl]  # (ntimes, nchan), used as T_sys

    loc = MoonLocation(lat=-23.813, lon=182.258)
    t0 = Time("2027-01-01T09:00:00", location=loc)
    times = np.linspace(t0, t0 + 655.720 * 3600 * u.s, num=N_TIMES,
                        endpoint=False)

    I_ref = np.load(DATA / "haslam_galactic.npz")["m"]
    wmap = ld.sky.load_wmap(DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits",
                            nside=NSIDE)
    Q_ref, U_ref = wmap[1], wmap[2]
    rm_gal = ld.sky.load_rm(DATA / "faraday2020v2.hdf5")

    beam = ld.Beam.from_file(BEAM_FILE, frequency=30, nside=NSIDE)
    beam.precompute_weights()
    mask = ld.HealpixGrid(NSIDE, horizon=True).mask

    print(f"rotating fiducial sky ({N_TIMES} steps)...")
    I_t, Q_t, U_t, rm_t = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm_gal, times, NSIDE, LUSEE_LOC)

    basis = spin2_basis(NSIDE, LMAX)
    print(f"rotating {len(basis)} basis modes (lmax={LMAX})...")
    basis_topo = []
    for i, (label, Qb, Ub) in enumerate(basis):
        Qb_t, Ub_t, _ = rotate_pol_maps(Qb, Ub, rm_gal, times, NSIDE,
                                        LUSEE_LOC)
        basis_topo.append((Qb_t, Ub_t))
        print(f"  mode {i + 1}/{len(basis)} ({label})")

    print("integration-time scan (alpha marginalized over sky + tau):")
    snr, snr_opt = [], []
    for dt, lbl in DT_CASES:
        sigma = radiometer_sigma(pI_FR, dnu, dt)
        out = run_forecast(I_t, Q_t, U_t, rm_t, basis_topo, beam, mask,
                           nu, lam2, sigma)
        snr.append(out["snr"])
        snr_opt.append(out["snr_opt"])
        print(f"  {lbl:>7}: SNR(marginalized)={out['snr']:8.2f}  "
              f"SNR(fixed-sky)={out['snr_opt']:8.2f}  "
              f"sigma(alpha)={out['sigma_alpha']:.3e}")

    labels = [l for _, l in DT_CASES]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, snr_opt, 0.4, label="fixed sky (optimistic)")
    ax.bar(x + 0.2, snr, 0.4, label="sky+tau marginalized")
    ax.axhline(5, color="k", ls=":", lw=1, label="SNR=5")
    ax.set(title="Faraday-amplitude detection SNR vs integration",
           ylabel="SNR", xticks=x, xticklabels=labels, yscale="log")
    ax.legend()
    fig.tight_layout()
    out_png = RES / "fisher_forecast.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the driver**

Run: `cd notebooks && uv run python fisher_forecast.py`
Expected: prints a per-dt SNR table and `saved .../results/fisher_forecast.png`. The marginalized SNR must be `<=` the fixed-sky SNR in every row (the degeneracy cost). If `faraday_fullband.npz` is missing, run `uv run python faraday_fullband_sim.py` first (~2 h) — note this in the run output, do not silently skip.

- [ ] **Step 3: Sanity-check the result**

Confirm: (a) marginalized SNR < fixed-sky SNR (sky ignorance costs SNR); (b) SNR grows with integration time; (c) `sigma(alpha)` is finite (no singular-Fisher blow-up — if it blows up, lower `LMAX` or raise `rcond` and note it). Record the headline marginalized 1-h SNR in the commit message.

- [ ] **Step 4: Commit**

```bash
git add notebooks/fisher_forecast.py notebooks/results/fisher_forecast.png
git commit -m "feat(analysis): step-4 sky-marginalized Faraday Fisher forecast"
```

---

### Task 7: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In the "RM-synthesis detection layer" section, add a bullet group describing the Fisher-forecast layer: `forward.py` (`pol_response` reusable linear operator), `skybasis.py` (low-ℓ spin-2 nuisance basis), `fisher.py` (`run_forecast`, marginalized `σ(α)`), and the driver `notebooks/fisher_forecast.py`. State the key facts: data is linear in the intrinsic sky; the forecast marginalizes a large-scale sky nuisance + an effective Faraday-variance `τ`; it reuses `faraday_fullband.npz` and runs at channel centers (intra-channel bandwidth depolarization neglected); headline output is marginalized detection SNR vs the fixed-sky bound.

- [ ] **Step 2: Update README.md**

Add a short subsection mirroring the CLAUDE.md description for external readers: what the Fisher forecast answers ("can LuSEE detect the Faraday rotation once the unknown intrinsic polarized sky is marginalized?") and how to run `notebooks/fisher_forecast.py`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (existing + new `test_forward.py`, `test_skybasis.py`, `test_fisher.py`).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document Fisher-forecast detection layer"
```

---

## Notes / deferred (out of scope for this branch)

These were discussed and intentionally deferred to follow-up branches (keep them out of this one):
- **GSM/ULSA for Stokes I** + WMAP Q/U as *soft priors* (this branch uses the existing WMAP-I×ν^-2.5 fiducial and flat sky priors). Adding priors = extra `F_prior` block on the sky/τ params.
- **Thick-truth (Burn-slab) injection** to test whether the thin-screen-templated filter still detects — the honest robustness experiment.
- **Per-mode spectral-index freedom** (each spatial mode currently rides the fixed `β_QU` law) and **raw-grid bandwidth-depol templates** (vs channel-center).
- The cheap **beam-depolarization check** (RM map ⊛ beam) confirming transverse mixing dominates depth dispersion.

## Self-Review

- **Spec coverage:** Fisher-core (Task 3) + marginalization (Task 4) ✓; low-ℓ spin-2 sky modes (Task 2) ✓; analytic/linear-operator Jacobian — `pol_response` applied to basis maps + analytic τ + FD α (Tasks 1, 4) ✓; reuse existing sim — driver loads `faraday_fullband.npz`, no new full sim (Task 6) ✓; new feature branch ✓. The single-σ_eff/τ nuisance from the "depth-depolarization" discussion is the `dispersion_column` (Task 4) ✓.
- **Placeholder scan:** all code blocks are complete; the only prose-described edits are the `__init__.py` insertion (Task 5) and doc edits (Task 7), which depend on existing file content and are described precisely.
- **Type consistency:** `pol_response` → complex `(ntimes, nfreq)` used identically in `faraday_column`, `run_forecast`, driver; `fisher_matrix(columns, sigma)` consumes the same complex arrays via `stack_real`; `run_forecast` returns the dict keys the driver reads (`snr`, `snr_opt`, `sigma_alpha`, `n_modes`); `lam2`/`nu`/`dnu` come from the npz channel table; alpha is column index 0 everywhere it is marginalized.
