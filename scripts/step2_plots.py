"""Step 2/3 figures: real sky through the Faraday screen.

Usage: step2_plots.py [--center 30]

Reads generated_data/real{C}_* produced by step2_real_sky.py and writes
figures into report/figures/ with a real{C}_ prefix.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from common import (  # noqa: E402
    FIG_DIR,
    GEN_DIR,
    SIDEREAL_DAY_S,
    load_sky_maps,
    sky_at_freq,
)
from lusee_faraday import fourport as fp  # noqa: E402


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


# Zenith-calibrated polarimeter vectors, set per band center in main().
X_VEC_P = None
Y_VEC_P = None


def stokes(channels):
    return fp.polarimeter_from_channels(channels, X_VEC_P, Y_VEC_P)


def fig_sky_model(center):
    """Mollweide views of the input sky at the band center."""
    import healpy as hp

    maps = load_sky_maps()
    I, Q, U = sky_at_freq(maps, center)
    RM = maps["RM"]
    fig = plt.figure(figsize=(10, 7))
    hp.mollview(
        np.log10(I),
        sub=(2, 2, 1),
        title=f"log10 I at {center:g} MHz [K]",
        cmap="magma",
    )
    P = np.hypot(Q, U)
    hp.mollview(
        np.log10(np.maximum(P, 1e-3)),
        sub=(2, 2, 2),
        title=f"log10 |P| at {center:g} MHz [K]",
        cmap="magma",
    )
    hp.mollview(
        RM,
        sub=(2, 2, 3),
        min=-100,
        max=100,
        title=r"RM [rad/m$^2$]",
        cmap="RdBu_r",
    )
    chi_var = np.degrees(np.arctan2(U, Q) / 2)
    hp.mollview(
        chi_var,
        sub=(2, 2, 4),
        cmap="twilight",
        title="polarization angle [deg]",
    )
    savefig(fig, f"real{center:g}_sky_model")


def load(center):
    tag = f"real{center:g}"
    meta = np.load(GEN_DIR / f"{tag}_meta.npz")
    binned = np.load(GEN_DIR / f"{tag}_binned.npz")
    wf = np.load(GEN_DIR / f"{tag}_fine_waterfall.npy", mmap_mode="r")
    return tag, meta, binned, wf


def fig_spectrum_snapshot(center, tag, meta, binned, wf, it=None):
    """Fine + binned pseudo-Stokes spectrum at one time."""
    ff = meta["fine_freqs_mhz"]
    nt = wf.shape[0]
    if it is None:
        it = nt // 2
    S = stokes(np.asarray(wf[it]))
    S0 = stokes(meta["nofaraday"][it])
    Sp = stokes(binned["parent"][it])
    Sz = stokes(binned["zoom"][it])
    zf, order = fp.zoom_frequency_grid(binned["parent_centers_mhz"])
    Sz_sorted = np.array([Sz[p, k] for p, k in order])
    off = (ff - center) * 1e3
    zoff = (zf - center) * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    scale = 1.0 / np.abs(S0[0])
    for ax, comp, name in zip(axes, (1, 2), ("Q", "U")):
        y_fine = S[:, comp] * scale
        ax.plot(off, y_fine, lw=0.4, color="C0", label="fine grid")
        ax.plot(
            zoff,
            Sz_sorted[:, comp] * scale,
            ".",
            ms=3.5,
            color="C1",
            label="zoom bins",
        )
        ax.plot(
            (binned["parent_centers_mhz"] - center) * 1e3,
            Sp[:, comp] * scale,
            "s",
            ms=7,
            color="C3",
            label="parent bins",
        )
        # tight limits around the fine-grid structure so the fast
        # Faraday "hash" is visible (the no-Faraday level is far off
        # scale and intentionally not shown)
        lo, hi = y_fine.min(), y_fine.max()
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(rf"${name}_{{\rm obs}}/I_{{\rm obs}}$(no FD)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_xlabel(f"frequency offset from {center:g} MHz  [kHz]")
    axes[0].set_title(f"Diffuse sky at {center:g} MHz, t index {it}")
    savefig(fig, f"{tag}_spectrum_snapshot")


def fig_waterfall_QU(center, tag, meta, binned, wf):
    """Fine-grid waterfall of Q/I and U/I over time and frequency."""
    ff = meta["fine_freqs_mhz"]
    nt = wf.shape[0]
    t_hr = np.arange(nt) * SIDEREAL_DAY_S / nt / 3600.0
    # thin the fine axis for plotting (every 16th channel)
    step = 16
    S = np.empty((nt, ff.size // step, 4))
    for s in range(0, nt, 64):
        sl = slice(s, min(s + 64, nt))
        S[sl] = stokes(np.asarray(wf[sl, ::step, :]))
    S0 = stokes(meta["nofaraday"])  # (T, 4)
    Inorm = S0[:, 0][:, None]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    extent = [
        (ff[0] - center) * 1e3,
        (ff[-1] - center) * 1e3,
        t_hr[-1],
        t_hr[0],
    ]
    for ax, comp, name in zip(axes, (1, 2), ("Q", "U")):
        data = S[:, :, comp] / Inorm
        vmax = np.nanpercentile(np.abs(data), 99)
        im = ax.imshow(
            data,
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0, -vmax, vmax),
        )
        ax.set_title(rf"${name}_{{\rm obs}}/I_{{\rm obs}}$")
        ax.set_xlabel(f"offset from {center:g} MHz  [kHz]")
        fig.colorbar(im, ax=ax, shrink=0.9)
    axes[0].set_ylabel("time [h]")
    savefig(fig, f"{tag}_waterfall_QU")


def fig_polfrac_track(center, tag, meta, binned, wf):
    """Recovered fractional polarization vs time for the binnings."""
    nt = wf.shape[0]
    t_hr = np.arange(nt) * SIDEREAL_DAY_S / nt / 3600.0

    def pfrac(S):
        return np.hypot(S[..., 1], S[..., 2]) / S[..., 0]

    S0 = stokes(meta["nofaraday"])
    Sp = stokes(binned["parent"][:, 1])
    Sz = stokes(binned["zoom"][:, 1, 0])
    Si = stokes(binned["ideal_zoom"][:, 1, 0])
    # fine-grid reference at the exact center channel
    nf = wf.shape[1]
    Sf = stokes(np.asarray(wf[:, nf // 2, :]))

    # The Faraday-on curves coincide to ~1e-3 at this scale, so a
    # single-panel overlay shows only the last-drawn line.  Top:
    # absolute p (no-FD vs parent); bottom: channelization ratios to
    # the parent bin, where the small differences live.
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    p_par = pfrac(Sp)
    axes[0].plot(t_hr, pfrac(S0), lw=1.6, color="0.6",
                 label="no Faraday (sky P as modeled)")
    axes[0].plot(t_hr, p_par, lw=1.0, color="C3",
                 label="with Faraday (parent bin)")
    ionly_path = GEN_DIR / f"real{center:g}_ionly.npz"
    if ionly_path.exists():
        Sio = stokes(np.load(ionly_path)["products"])
        axes[0].plot(t_hr, pfrac(Sio), lw=1.0, color="k", ls=":",
                     label="I only (pure leakage)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sqrt{Q^2+U^2}/I$")
    axes[0].legend(fontsize=8)
    for S, label, color in (
        (Sf, "fine channel", "C0"),
        (Si, "ideal zoom bin", "C2"),
        (Sz, "real zoom bin", "C1"),
    ):
        axes[1].plot(t_hr, pfrac(S) / p_par - 1.0, lw=1.0,
                     color=color, label=label)
    axes[1].axhline(0, color="0.85", lw=0.6, zorder=0)
    axes[1].set_ylabel(r"$p_X / p_{\rm parent} - 1$")
    axes[1].set_xlabel("time  [hours]")
    axes[1].legend(fontsize=8, ncol=3)
    axes[0].set_title(
        f"Diffuse-sky fractional polarization at {center:g} MHz"
    )
    savefig(fig, f"{tag}_polfrac_track")


def main():
    global X_VEC_P, Y_VEC_P
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", type=float, default=30.0)
    args = ap.parse_args()
    center = args.center
    from zenith_weights import get_weights

    X_VEC_P, Y_VEC_P = get_weights(center)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_sky_model(center)
    tag, meta, binned, wf = load(center)
    it_snap = int(np.argmax(stokes(meta["nofaraday"])[:, 0]))
    fig_spectrum_snapshot(center, tag, meta, binned, wf, it=it_snap)
    fig_waterfall_QU(center, tag, meta, binned, wf)
    fig_polfrac_track(center, tag, meta, binned, wf)
    print("figures done", flush=True)


if __name__ == "__main__":
    main()
