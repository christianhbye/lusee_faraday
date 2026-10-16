"""Drawn Gaussian screens against the analytic off/on moments.

Writes ``generated_data/gaussian_screen_step1/draws.npz``: the direct pixel
sums of the Faraday-off and -on readouts for a few drawn screens, a
per-screen quadrature-convergence check, and a small ensemble pilot compared
with ``gaussian_screen.visibility_moments``. An illustration and a pilot, not
the step-2 gate in docs/plans/2026-10-15-gaussian-screen-step1.md.

Run from the repository root; a few minutes, ~2 GB.
"""

import common  # noqa: F401  (sets JAX_ENABLE_X64 before any jax import)

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lusee_faraday.gaussian_screen import (
    READOUTS,
    ScreenModel,
    visibility_moments,
)
from lusee_faraday.screen_draws import ScreenGrid, direct_readouts, draw_depth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_data/gaussian_screen_step1/draws.npz"

NAMES = tuple(READOUTS)
FREQ = np.linspace(10.0, 50.0, 161)
SHOW_SEEDS = (11, 12, 13, 14, 15)
FINE = ScreenGrid(n_pix=4096, width_deg=8.0)
COARSE = ScreenGrid(n_pix=2048, width_deg=8.0)
CONVERGENCE_FREQ = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
PILOT_SEEDS = range(1000, 1064)
PILOT_FREQ = np.array([30.0, 40.0, 50.0])


def config(model):
    return dict(
        model=asdict(model),
        fine=asdict(FINE),
        coarse=asdict(COARSE),
        show_seeds=list(SHOW_SEEDS),
        pilot_seeds=[PILOT_SEEDS.start, PILOT_SEEDS.stop],
        freq=[FREQ[0], FREQ[-1], len(FREQ)],
        pilot_freq=PILOT_FREQ.tolist(),
        convergence_freq=CONVERGENCE_FREQ.tolist(),
    )


def analytic(freq, model):
    """Off, mean-on and standard deviation per readout, in K."""
    out = {}
    for name in NAMES:
        mom = visibility_moments(
            freq, replace(model, readout=READOUTS[name]), diagonal_only=True
        )
        assert np.max(abs(mom["mean_on"].imag)) < 1e-12
        out[name] = (
            mom["off"].real,
            mom["mean_on"].real,
            np.sqrt(np.maximum(mom["covariance"].real, 0.0)),
        )
    return out


def main():
    model = ScreenModel()
    t0 = time.time()
    save = dict(freq=FREQ, config=json.dumps(config(model), default=float))

    # 1. A few screens on the fine grid, direct sums over the full band.
    on = {n: [] for n in NAMES}
    off = {}
    for seed in SHOW_SEEDS:
        depth = draw_depth(model, FINE, seed)
        res = direct_readouts(depth, FREQ, model, FINE, names=NAMES)
        for n in NAMES:
            off[n] = res[n][0]
            on[n].append(res[n][1])
        # 2. Same screen, every other pixel: halves the quadrature resolution.
        fine = direct_readouts(
            depth, CONVERGENCE_FREQ, model, FINE, names=NAMES
        )
        coarse = direct_readouts(
            depth[::2, ::2], CONVERGENCE_FREQ, model, COARSE, names=NAMES
        )
        scale = (
            model.eta * model.p_ref_k * (CONVERGENCE_FREQ / 30.0) ** model.beta
        )
        change = np.array(
            [abs(fine[n][1] - coarse[n][1]) / scale for n in NAMES]
        )
        save[f"convergence_{seed}"] = change.max(axis=0)
        print(
            f"seed {seed}: max |fine-coarse|/P at "
            f"{CONVERGENCE_FREQ.tolist()} MHz = {change.max(axis=0)}  "
            f"[{time.time() - t0:.0f} s]",
            flush=True,
        )
    for n in NAMES:
        save[f"off_{n}"] = off[n]
        save[f"on_{n}"] = np.array(on[n])
    for n, (a_off, a_mean, a_std) in analytic(FREQ, model).items():
        save[f"analytic_off_{n}"] = a_off
        save[f"analytic_mean_{n}"] = a_mean
        save[f"analytic_std_{n}"] = a_std

    # 3. Pilot ensemble on the coarse grid at a few frequencies.
    pilot = {n: [] for n in NAMES}
    for seed in PILOT_SEEDS:
        depth = draw_depth(model, COARSE, seed)
        res = direct_readouts(depth, PILOT_FREQ, model, COARSE, names=NAMES)
        for n in NAMES:
            pilot[n].append(res[n][1])
    ref = analytic(PILOT_FREQ, model)
    for n in NAMES:
        sample = np.array(pilot[n])
        save[f"pilot_{n}"] = sample
        mean_z = (sample.mean(0) - ref[n][1]) / (
            sample.std(0, ddof=1) / np.sqrt(len(sample))
        )
        ratio = sample.std(0, ddof=1) / ref[n][2]
        print(
            f"pilot {n:>2}: mean z = {np.round(mean_z, 2)}, "
            f"std ratio = {np.round(ratio, 3)}",
            flush=True,
        )
    save["pilot_freq"] = PILOT_FREQ
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
