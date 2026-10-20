"""Delay spectrum of the sub-pixel spectra (slide side product).

Fourier transform over the zoom bins of the three parent channels (192 bins,
390.6 Hz apart: delay resolution 13 us, range +-1.28 ms) of the total Q
(leakage + Faraday), the Faraday Q alone (linear map realization), one
alpha=5/3 ensemble draw, and the per-bin noise for a stated exposure, each
with a Blackman-Harris window. Reads the slides_subpixel_spectra cache and
the measured-system noise record.
"""

import common  # noqa: F401
import argparse
import json

import numpy as np

from common import REPO
from lusee_faraday import channelization as chan

FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"
WMAP_PIXEL_RMS_30 = 1.257e4
BETA_QU = -2.8
SIGMA_COLUMN = 10.0
RECEIVER = (3.0, 0.1)


def noise_zoom_rms(center, exposure_s):
    rec = json.loads((REPO / "docs/measured-system-results.json").read_text())
    row = [
        n
        for n in rec["noise"]
        if n["center_mhz"] == center
        and abs(n["voltage_nv_rt_hz"] - RECEIVER[0]) < 1e-9
        and abs(n["current_pa_rt_hz"] - RECEIVER[1]) < 1e-9
    ][0]
    z = np.load(REPO / row["path"])
    c = z["covariance_per_second"][0]
    var = np.array([c[b, 1, b, 1] for b in range(3, c.shape[0])])
    return float(np.sqrt(np.mean(var) / exposure_s))


def exposure_s(lunations):
    rec = json.loads((REPO / "docs/observing-exposure.json").read_text())
    by = {
        r["lunations"]: r["exposure_s"]
        for r in rec["results"]
        if r["sidereal_turn"] == 0.0
    }
    return by[lunations] if lunations in by else by[24] * lunations / 24


def ensemble_draw(center, freqs, seed=5):
    """alpha=5/3 anchor-1 draw on the given grid."""
    from lusee_faraday.conventions import lambda_squared

    rec = json.loads(
        (REPO / "docs/faraday-covariance-column-results.json").read_text()
    )
    row = [
        r
        for r in rec["results"]
        if r["center_mhz"] == center
        and "alpha=5/3 dphi1=1" in r["case"]["label"]
    ][0]
    z = np.load(REPO / row["path"])
    spk = z["speckle"]
    nb = spk.shape[0]
    var = np.mean([spk[b, 1, b, 1].real for b in range(11, nb - 8)])
    l2 = lambda_squared(freqs)
    corr = np.exp(-2 * SIGMA_COLUMN**2 * (l2[:, None] - l2[None, :]) ** 2)
    w, v = np.linalg.eigh(corr)
    root = v * np.sqrt(np.clip(w, 0, None))
    rng = np.random.default_rng(seed)
    scale = np.sqrt(var) * WMAP_PIXEL_RMS_30 * (center / 30.0) ** BETA_QU
    return scale * (root @ rng.standard_normal(root.shape[1]))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal.windows import blackmanharris

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", type=float, default=30.0)
    parser.add_argument("--lunations", type=int, default=24)
    parser.add_argument("--phase-index", type=int, default=1)
    args = parser.parse_args()
    center, it = args.center, args.phase_index
    d = np.load(
        REPO
        / "generated_data/cache/slides_subpixel"
        / f"{center:g}mhz"
        / "spectra.npz"
    )
    off = d["zoom_offsets_hz"] * 1e-6
    parents = d["parent_centers_mhz"]
    fz = np.concatenate([parents[j] + off for j in range(3)])
    order = np.argsort(fz)
    fz = fz[order]

    def bins(key, comp):
        return np.concatenate([d[key][it, j] for j in range(3)], axis=0)[
            order
        ][:, comp]

    k = bins("unit_zoom", 0)
    far_q = bins("stokes_zoom", 1) / k
    leak_q = bins("ionly_zoom", 1) / k
    total_q = leak_q + far_q
    draw_q = ensemble_draw(center, fz)
    t_exp = exposure_s(args.lunations)
    sigma_n = noise_zoom_rms(center, t_exp)
    rng = np.random.default_rng(3)
    noise_q = sigma_n * rng.standard_normal(fz.size)
    step_hz = chan.ZOOM_STEP_HZ
    # Global window, with the aliased edge zoom bins of each parent
    # down-weighted (a real analysis would flag them): a cosine taper to
    # zero over the bins whose folded response exceeds 5 %.
    fine_off = np.arange(-100e3, 100e3, chan.FINE_STEP_HZ)
    wz = chan.zoom_weights(fine_off)
    folded = np.sum(np.abs(wz[np.abs(fine_off) > 12.5e3]), axis=0) / np.sum(
        np.abs(wz), axis=0
    )
    good = folded <= 0.05
    taper_one = np.ones(64)
    bad_idx = np.where(~good)[0]
    for k in bad_idx:
        taper_one[k] = 0.0
    # smooth the mask edges over three bins in frequency order
    o1 = np.argsort(off)
    t_sorted = taper_one[o1]
    kernel = np.array([0.25, 0.5, 0.25])
    t_sorted = np.convolve(np.pad(t_sorted, 1, mode="edge"), kernel, "valid")
    taper = np.concatenate([t_sorted] * 3)  # same mask for the 3 parents
    win = blackmanharris(fz.size) * taper
    win /= np.sqrt(np.mean(win**2))
    print(f"down-weighted {np.sum(~good) * 3} aliased bins of {fz.size}")
    tau = np.fft.rfftfreq(fz.size, d=step_hz) * 1e6  # microseconds

    def delay_power(x):
        return np.abs(np.fft.rfft(x * win)) ** 2 / fz.size

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for x, label, color, ls in (
        (total_q, "total Q: leakage + Faraday (linear map)", "k", "-"),
        (far_q, "Faraday Q alone, linear map realization", "C0", "-"),
        (draw_q, r"Faraday Q, one $\alpha$=5/3 ensemble draw", "C2", "--"),
        (
            noise_q,
            f"noise, {args.lunations} lunations at one sidereal phase "
            f"({t_exp / 3600:.1f} h; {sigma_n:.1f} K per bin)",
            "C3",
            ":",
        ),
    ):
        ax.semilogy(
            tau, delay_power(x), color=color, ls=ls, lw=1.4, label=label
        )
    ax.set_xlabel(
        r"delay $\tau$ [$\mu$s]  (1 / frequency-coherence of the ripple "
        r"$\approx$ 90 $\mu$s at 30 MHz, 20 $\mu$s at 50 MHz)"
    )
    ax.set_ylabel(
        "delay power [K$^2$], Blackman-Harris window,\n"
        "aliased edge bins down-weighted"
    )
    ax.set_title(
        f"Delay spectrum of the zoom bins of three parent channels at "
        f"{center:g} MHz, sidereal phase {d['lst_hours'][it] / 24:.2f}",
        fontsize=10,
    )
    ax.set_xlim(0, tau.max())
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    name = FIG_DIR / f"12_delay_spectrum_{center:g}MHz.png"
    fig.savefig(name, dpi=140)
    print("saved", name)
    print(
        f"noise per zoom bin {sigma_n:.2f} K for {t_exp:.0f} s; Faraday Q rms "
        f"{far_q.std():.3f} K; draw rms {draw_q.std():.3f} K"
    )


if __name__ == "__main__":
    main()
