"""Step 1 figures: single polarized transiting source at 30 MHz.

Reads generated_data/step1_* produced by step1_point_source.py and
writes paper-candidate figures into report/figures/.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from common import FIG_DIR, GEN_DIR, SIDEREAL_DAY_S  # noqa: E402
from lusee_faraday import channelization as chan  # noqa: E402
from lusee_faraday import polarimeter as pol  # noqa: E402

META = np.load(GEN_DIR / "step1_meta.npz")
BINNED = np.load(GEN_DIR / "step1_binned.npz")
WF = np.load(GEN_DIR / "step1_fine_waterfall.npy", mmap_mode="r")
FF = META["fine_freqs_mhz"]
THETA = META["theta"]
T_HR = np.arange(THETA.size) * SIDEREAL_DAY_S / THETA.size / 3600.0
IT_TRANSIT = int(THETA.argmin())
V2HZ = 1e21  # plot unit: 1e-21 V^2/Hz


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


# Default: the raw X = E-W, Y = N-S polarimeter and "step1_" figure
# names.  `--calibrated` switches to the zenith-weighted polarimeter
# (scripts/zenith_weights.py) and a "step1w_" figure prefix, so both
# figure sets stay reproducible side by side.
PREFIX = "step1"
X_VEC_P = None
Y_VEC_P = None


def stokes(channels):
    return pol.pseudo_stokes_from_channels(channels, X_VEC_P, Y_VEC_P)


def fig_transit_spectrum():
    """Fine-grid pseudo-Stokes vs frequency at transit, with binning."""
    S = stokes(np.asarray(WF[IT_TRANSIT])) * V2HZ  # (F, 4)
    Sp = stokes(BINNED["parent"][IT_TRANSIT]) * V2HZ  # (3, 4)
    Sz = stokes(BINNED["zoom"][IT_TRANSIT]) * V2HZ  # (3, 64, 4)
    zf, order = chan.zoom_frequency_grid(BINNED["parent_centers_mhz"])
    Sz_sorted = np.array([Sz[p, k] for p, k in order])  # (192, 4)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    off = (FF - 30.0) * 1e3  # kHz
    zoff = (zf - 30.0) * 1e3
    win = np.abs(off) <= 41.0
    for ax, comp, name in zip(axes, (1, 2), ("Q", "U")):
        ax.plot(
            off[win],
            S[win, comp],
            lw=0.4,
            color="C0",
            label="fine grid (12.2 Hz)",
        )
        ax.plot(
            zoff,
            Sz_sorted[:, comp],
            ".",
            ms=3.5,
            color="C1",
            label="zoom bins (390.6 Hz)",
        )
        ax.plot(
            (BINNED["parent_centers_mhz"] - 30.0) * 1e3,
            Sp[:, comp],
            "s",
            ms=7,
            color="C3",
            label="parent bins (25 kHz)",
        )
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.set_ylabel(
            rf"$\{name}_{{\rm obs}}$  [$10^{{-21}}\,$V$^2$/Hz]".replace(
                r"\Q", "Q"
            ).replace(r"\U", "U")
        )
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[1].set_xlabel("frequency offset from 30 MHz  [kHz]")
    axes[0].set_title(
        r"Transit spectrum, $\phi_{\rm FD}=250\,$rad/m$^2$"
        "  (Faraday oscillation period 1.9 kHz)"
    )
    savefig(fig, f"{PREFIX}_transit_spectrum")


def fig_transit_spectrum_zoomwin():
    """Zoom into +-3 kHz so individual oscillations are visible."""
    S = stokes(np.asarray(WF[IT_TRANSIT])) * V2HZ
    Sz = stokes(BINNED["zoom"][IT_TRANSIT]) * V2HZ
    Si = stokes(BINNED["ideal_zoom"][IT_TRANSIT]) * V2HZ
    zf, order = chan.zoom_frequency_grid(BINNED["parent_centers_mhz"])
    Sz_sorted = np.array([Sz[p, k] for p, k in order])
    Si_sorted = np.array([Si[p, k] for p, k in order])
    off = (FF - 30.0) * 1e3
    zoff = (zf - 30.0) * 1e3
    win = np.abs(off) <= 3.0
    zwin = np.abs(zoff) <= 3.0

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(off[win], S[win, 1], lw=0.8, color="C0", label="fine grid")
    ax.plot(
        zoff[zwin],
        Sz_sorted[zwin, 1],
        "o-",
        ms=4,
        lw=0.8,
        color="C1",
        label="real zoom bins",
    )
    ax.plot(
        zoff[zwin],
        Si_sorted[zwin, 1],
        "s--",
        ms=3.5,
        lw=0.8,
        color="C2",
        label="ideal (Gaussian) zoom bins",
    )
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_xlabel("frequency offset from 30 MHz  [kHz]")
    ax.set_ylabel(r"$Q_{\rm obs}$  [$10^{-21}\,$V$^2$/Hz]")
    ax.legend(fontsize=8)
    savefig(fig, f"{PREFIX}_transit_spectrum_zoom")


def fig_track_stokes():
    """Pseudo-Stokes vs time in the central zoom bin vs no-Faraday."""
    Sz = stokes(BINNED["zoom"][:, 1, 0]) * V2HZ  # (T, 4) center bin
    Sp = stokes(BINNED["parent"][:, 1]) * V2HZ  # (T, 4) center parent
    S0 = stokes(META["nofaraday"]) * V2HZ  # (T, 4)
    up = THETA <= np.pi / 2

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
    names = ("I", "Q", "U")
    for ax, comp, name in zip(axes, (0, 1, 2), names):
        y0 = np.where(up, S0[:, comp], np.nan)
        yz = np.where(up, Sz[:, comp], np.nan)
        yp = np.where(up, Sp[:, comp], np.nan)
        ax.plot(T_HR, y0, lw=1.4, color="0.6", label="no Faraday")
        ax.plot(T_HR, yz, lw=1.0, color="C1", label="zoom bin 0")
        ax.plot(T_HR, yp, lw=1.0, color="C3", label="parent bin")
        ax.set_ylabel(rf"${name}_{{\rm obs}}$  [$10^{{-21}}\,$V$^2$/Hz]")
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    axes[0].legend(fontsize=8, ncol=3)
    axes[2].set_xlabel("time  [hours]")
    axes[0].set_title(
        "Polarized source track at 30 MHz " r"($\phi_{\rm FD}=250$ rad/m$^2$)"
    )
    savefig(fig, f"{PREFIX}_track_stokes")


def contiguous_track():
    """Indices of the above-horizon samples, ordered rise -> set.

    The time axis is periodic, so the up interval can wrap; roll it to
    start at the rising transition.
    """
    up = THETA <= np.pi / 2
    rises = np.where(up & ~np.roll(up, 1))[0]
    i0 = rises[0]
    order = (np.arange(THETA.size) + i0) % THETA.size
    return order[up[order]]


def fig_polfrac_track():
    """Recovered polarization fraction vs source altitude."""
    track = contiguous_track()
    alt = 90.0 - np.degrees(THETA[track])
    imax = int(alt.argmax())

    def pfrac(S):
        return np.hypot(S[..., 1], S[..., 2]) / S[..., 0]

    S0 = stokes(META["nofaraday"])
    Sp = stokes(BINNED["parent"][:, 1])
    Sz = stokes(BINNED["zoom"][:, 1, 0])
    Si = stokes(BINNED["ideal_zoom"][:, 1, 0])

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for S, label, color, lw in (
        (S0, "no Faraday", "0.6", 1.6),
        (Si, "ideal zoom bin", "C2", 1.0),
        (Sz, "real zoom bin", "C1", 1.0),
        (Sp, "parent bin", "C3", 1.0),
    ):
        p = pfrac(S)[track]
        ax.plot(
            alt[: imax + 1], p[: imax + 1], lw=lw, color=color, label=label
        )
        ax.plot(alt[imax:], p[imax:], lw=lw, color=color, ls="--", alpha=0.7)
    ax.plot(
        [],
        [],
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="setting branch (dashed)",
    )
    ax.set_xlabel("source altitude  [deg]")
    ax.set_ylabel(r"recovered $p = \sqrt{Q^2+U^2}/I$")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.08)
    ax.axhline(1, color="0.85", lw=0.6, zorder=0)
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, f"{PREFIX}_polfrac_track")


def track_ticks(alt, imax):
    """Altitude tick positions/labels along the rise->set track."""
    yticks, ylabels = [], []
    for a in (10, 30, 50, 70, 89):
        yticks.append(int(np.argmin(np.abs(alt[: imax + 1] - a))))
        ylabels.append(str(a))
    for a in (70, 50, 30, 10):
        yticks.append(imax + int(np.argmin(np.abs(alt[imax:] - a))))
        ylabels.append(str(a))
    return yticks, ylabels


def fig_xy_waterfalls():
    """Coherency of the calibrated X/Y: XX, YY, Re<XY*>, Im<XY*>.

    Waterfalls over the altitude track x zoom frequency, built from
    the 16 zoom-bin products combined with the polarimeter vectors in
    use (raw dipoles by default, zenith-calibrated with --calibrated).
    """
    zf, order = chan.zoom_frequency_grid(BINNED["parent_centers_mhz"])
    zoom = BINNED["zoom"]  # (T, 3, 64, 16)
    ch = np.stack(
        [zoom[:, p, k, :] for p, k in order], axis=1
    )  # (T, 192, 16)
    S = stokes(ch) * V2HZ  # (T, 192, 4) = I, Q, U, V
    # XX = I+Q, YY = I-Q; <X Y*> = conj(<Y X*>) -> Re = U, Im = -V
    panels = (
        (S[..., 0] + S[..., 1], r"$\langle|X|^2\rangle$", True),
        (S[..., 0] - S[..., 1], r"$\langle|Y|^2\rangle$", True),
        (S[..., 2], r"${\rm Re}\,\langle XY^*\rangle$", False),
        (-S[..., 3], r"${\rm Im}\,\langle XY^*\rangle$", False),
    )
    track = contiguous_track()
    alt = 90.0 - np.degrees(THETA[track])
    imax = int(alt.argmax())
    yticks, ylabels = track_ticks(alt, imax)
    zoff = (zf - 30.0) * 1e3

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True
    )
    extent = [zoff[0], zoff[-1], len(track) - 1, 0]
    for ax, (data, title, is_auto) in zip(axes.ravel(), panels):
        d = data[track]
        vmax = np.nanmax(np.abs(d))
        if is_auto:
            im = ax.imshow(d, aspect="auto", extent=extent,
                           cmap="viridis", vmin=0, vmax=vmax)
        else:
            im = ax.imshow(d, aspect="auto", extent=extent,
                           cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -vmax, vmax))
        ax.set_title(title, fontsize=11)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        fig.colorbar(im, ax=ax, shrink=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("offset from 30 MHz [kHz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("source altitude [deg]\n(rise " + r"$\to$"
                      + " transit " + r"$\to$" + " set)")
    fig.suptitle(
        "Calibrated polarimeter coherency, zoom-bin waterfalls "
        r"[$10^{-21}$ V$^2$/Hz]", y=0.99,
    )
    fig.tight_layout()
    savefig(fig, f"{PREFIX}_xy_waterfalls")


def fig_product_waterfalls():
    """All 16 products: altitude-track x zoom-frequency waterfalls."""
    zf, order = chan.zoom_frequency_grid(BINNED["parent_centers_mhz"])
    zoom = BINNED["zoom"]  # (T, 3, 64, 16)
    wf = np.array([zoom[:, p, k, :] for p, k in order])  # (192, T, 16)
    wf = np.moveaxis(wf, 0, 1) * V2HZ  # (T, 192, 16)
    track = contiguous_track()
    wf = wf[track]  # rows now run rise -> transit -> set
    alt = 90.0 - np.degrees(THETA[track])
    imax = int(alt.argmax())
    labels = [str(x) for x in META["labels"]]
    zoff = (zf - 30.0) * 1e3

    # altitude tick labels along the (monotonic-in-time) track
    tick_alts_rise = [10, 30, 50, 70, 89]
    tick_alts_set = [70, 50, 30, 10]
    yticks, ylabels = [], []
    for a in tick_alts_rise:
        yticks.append(int(np.argmin(np.abs(alt[: imax + 1] - a))))
        ylabels.append(str(a))
    for a in tick_alts_set:
        yticks.append(imax + int(np.argmin(np.abs(alt[imax:] - a))))
        ylabels.append(str(a))

    fig, axes = plt.subplots(
        4, 4, figsize=(13.0, 11.0), sharex=True, sharey=True
    )
    extent = [zoff[0], zoff[-1], len(track) - 1, 0]
    autos = {k for k, lab in enumerate(labels) if lab[0] == lab[1]}
    for k, ax in enumerate(axes.ravel()):
        data = wf[:, :, k]
        vmax = np.nanmax(np.abs(data))
        if k in autos:  # 00R, 11R, 22R, 33R
            im = ax.imshow(
                data,
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
        else:
            im = ax.imshow(
                data,
                aspect="auto",
                extent=extent,
                cmap="RdBu_r",
                norm=TwoSlopeNorm(0, -vmax, vmax),
            )
        ax.set_title(labels[k], fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    for ax in axes.ravel():
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
    for ax in axes[-1]:
        ax.set_xlabel("offset [kHz]")
    for ax in axes[:, 0]:
        ax.set_ylabel(
            "source altitude [deg]\n(rise " + r"$\to$"
            " transit " + r"$\to$" + " set)"
        )
    fig.suptitle(
        "16 correlation products, zoom-bin waterfalls "
        r"[$10^{-21}$ V$^2$/Hz]",
        y=0.995,
    )
    fig.tight_layout()
    savefig(fig, f"{PREFIX}_product_waterfalls")


def fig_polfrac_vs_fd():
    """Analytic-style summary: depolarization vs binning at transit."""
    S = stokes(np.asarray(WF[IT_TRANSIT]))
    Sp = stokes(BINNED["parent"][IT_TRANSIT])
    Sz = stokes(BINNED["zoom"][IT_TRANSIT])
    Si = stokes(BINNED["ideal_zoom"][IT_TRANSIT])
    zf, order = chan.zoom_frequency_grid(BINNED["parent_centers_mhz"])

    def pfrac(s):
        return np.hypot(s[..., 1], s[..., 2]) / s[..., 0]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    off = (FF - 30.0) * 1e3
    ax.plot(off, pfrac(S), lw=0.5, color="C0", alpha=0.7, label="fine grid")
    zoff = (zf - 30.0) * 1e3
    pz = np.array([pfrac(Sz[p, k]) for p, k in order])
    pi_ = np.array([pfrac(Si[p, k]) for p, k in order])
    ax.plot(zoff, pz, ".", ms=4, color="C1", label="real zoom")
    ax.plot(zoff, pi_, ".", ms=4, color="C2", label="ideal zoom")
    ax.plot(
        (BINNED["parent_centers_mhz"] - 30.0) * 1e3,
        pfrac(Sp),
        "s",
        ms=8,
        color="C3",
        label="parent",
    )
    ax.set_xlabel("frequency offset from 30 MHz  [kHz]")
    ax.set_ylabel("recovered polarization fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4)
    savefig(fig, f"{PREFIX}_polfrac_bins")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="use the zenith-weighted polarimeter (step1w_ prefix)",
    )
    args = ap.parse_args()
    if args.calibrated:
        from zenith_weights import get_weights

        X_VEC_P, Y_VEC_P = get_weights(30.0)
        PREFIX = "step1w"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_transit_spectrum()
    fig_transit_spectrum_zoomwin()
    fig_track_stokes()
    fig_polfrac_track()
    if not args.calibrated:
        # raw 16 products do not depend on the polarimeter weighting
        fig_product_waterfalls()
    fig_xy_waterfalls()
    fig_polfrac_vs_fd()
    print("all figures written to", FIG_DIR, flush=True)
