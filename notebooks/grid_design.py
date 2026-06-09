# notebooks/grid_design.py
"""Full-band grid design: build the FrequencyPlan, report its size/cost
and the inverse-variance-weighted RMSF, and save a figure. The chosen
grid (wide 30-51 MHz + zoom 5-30 MHz) is what faraday_fullband_sim.py
uses via fullband_specs().
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lusee_faraday import SpectrometerResponse, FrequencyPlan, rmsynth, utils

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
DECIMATION = {"wide": 250, "zoom": 10}
SUPPORT = 0.999


def fullband_specs():
    """Wide bins 30-51 MHz (every parent) + zoom 5-30 MHz every 0.5 MHz."""
    f = utils.freqs_lusee()
    wide = [(c, "wide") for c in f[(f >= 30) & (f <= 51.175)]]
    zoom = [(c, "zoom") for c in f[(f >= 5) & (f <= 29.5)][::20]]
    return wide + zoom


def main():
    spec = SpectrometerResponse.from_file(DATA / "spectrometer_bin_response.txt")
    specs = fullband_specs()
    plan = FrequencyPlan(spec, specs, decimation=DECIMATION, support=SUPPORT)
    table = plan.channel_table
    nu, dnu, lam2 = table["nu"], table["dnu"], table["lambda2"]

    n_wide = sum(1 for _, m in plan.specs if m == "wide")
    n_zoom = sum(1 for _, m in plan.specs if m == "zoom")
    print(f"specs: {n_wide} wide + {n_zoom} zoom")
    print(f"nraw (sim freqs): {plan.sim_freqs().size}")
    print(f"nchan: {nu.size}")
    print(f"est FR sim @8 cores: ~{100 * plan.sim_freqs().size * 2.5e-3 / 8 / 60:.0f} min")

    Tsys = 3000.0 * (nu / 50.0) ** -2.55 + 2.725
    weights = dnu / Tsys ** 2
    phi = np.arange(-60, 60, 0.02)
    R = np.abs(rmsynth.rmsf(lam2, phi, weights=weights))
    print(f"RMSF far sidelobe |R|@phi=1: "
          f"{np.abs(rmsynth.rmsf(lam2, np.array([1.0]), weights=weights))[0]:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].plot(nu, lam2, ".", ms=2)
    ax[0].set(title="channel lambda^2 coverage", xlabel="nu [MHz]",
              ylabel="lambda^2 [m^2]", yscale="log")
    ax[1].plot(phi, R)
    ax[1].set(title="inverse-variance RMSF", xlabel="phi [rad/m^2]",
              yscale="log")
    fig.tight_layout()
    out = RES / "grid_design.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
