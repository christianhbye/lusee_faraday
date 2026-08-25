"""Figures for the Faraday-depth template paper (spec S5).

Reads the step5_*.npz products; every figure is regenerable from
committed code plus data/.
"""

import common  # noqa: F401
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    NullLocator,
)

from common import FIG_DIR, GEN_DIR  # noqa: E402
from lusee_faraday import dispersion as dsp  # noqa: E402
from lusee_faraday.config import fine_freqs  # noqa: E402
from lusee_faraday.conventions import lambda_squared  # noqa: E402

K_LABELS = {
    0: "$k\\to\\infty$ (all far)",
    1: "$k=0$ (slab, fiducial)",
    2: "$k\\to-1$ (all local)",
}

# One fixed colour per emissivity geometry.  NOT the property cycle:
# fig_template_family draws two lines for the fiducial geometry (the
# incoherent template and its coherence-tilted companion) and they must
# share a colour.  Letting the cycle assign them gave the fiducial a
# green solid and a RED dotted, two slots apart, which no legend can
# pair up -- the defect this constant exists to prevent.
K_COLORS = {0: "C0", 1: "C2", 2: "C4"}

# dispersion.weighted_percentiles is called with these, in this order,
# by step5_template.py; the figure labels them so the grey verticals
# are readable rather than decorative.
PCT_LABELS = ("p50", "p90", "p99", "p99.9")


def fig_inputs(inp, d):
    """The two inputs to the histogram, and what they build.

    The intermediate-product figure of the report's build-up.  Top row:
    the beam-sky weight |w|^2 at three LSTs and the Hutschenreuter RM
    map -- the two arrays depth_distribution consumes, one supplying
    the bin INDEX and the other the MASS.  Bottom row: the result, the
    k -> inf weighted histogram and its k = 0 smoothed version.

    healpy's mollview writes into the current figure and takes its own
    ``sub`` grid, so the sky maps are placed first on a (2, 4) grid and
    the curve axes is then added spanning the bottom half.  Building
    the figure the other way round makes mollview warn and ignore the
    layout.
    """
    import healpy as hp

    w2, rm = inp["w2_lst"], inp["rm_display"]
    hours, band = inp["lst_hours"], float(inp["band"])
    fig = plt.figure(figsize=(11.5, 5.6))
    for i in range(w2.shape[0]):
        m = w2[i] / max(w2[i].max(), 1e-300)
        hp.mollview(
            m, norm="log", min=1e-4, max=1.0, cmap="viridis",
            sub=(2, 4, i + 1), cbar=False, fig=fig.number,
            title=f"$|w|^2$, LST {hours[i]:.0f} h",
        )
    hp.mollview(
        np.abs(rm), norm="log", min=1.0, max=2000.0, cmap="magma",
        sub=(2, 4, 4), title=r"$|\mathrm{RM}|$ [rad m$^{-2}$]",
        cbar=False, fig=fig.number,
    )
    # mollview's axes are not tight_layout-compatible, so the bottom
    # axes is placed by hand and tight_layout is not called at all --
    # calling it warns and then lays the sky maps out wrongly.
    ax = fig.add_axes([0.09, 0.08, 0.86, 0.36])
    ib = int(np.argmin(np.abs(d["bands"] - band)))
    ki = int(np.argmax(np.where(np.isfinite(d["ks"]), d["ks"], np.inf)))
    kf = int(np.argmin(np.abs(d["ks"])))
    ax.plot(d["phi"], d["H"][ib, ki], color=K_COLORS[ki], lw=1.1,
            label=K_LABELS[ki] + ": the weighted histogram")
    ax.plot(d["phi"], d["H"][ib, kf], color=K_COLORS[kf], lw=1.1,
            label=K_LABELS[kf] + ": that histogram smoothed")
    ax.set(xscale="log", yscale="log", xlim=(0.4, 2600),
           ylim=(1e-8, 3.0), ylabel="normalised template",
           xlabel=r"$\phi$ [rad m$^{-2}$]")
    ax.set_title(f"{band:.0f} MHz: every pixel deposits its "
                 r"$|w|^2$ at its own depth", fontsize=15)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="lower left", framealpha=0.9)
    return fig


def fig_pushforward(d):
    """How the f^k pushforward mixes emission, and why it smooths.

    The mechanism figure.  (a) one sightline, several geometries, each
    curve carrying unit area -- normalising by max() instead makes the
    k -> -1 case (which diverges at f=0) invisible.  (b) what that one
    sightline deposits.  (c) five columns stacking into nested top
    hats.  (d) the real sky, where F_0 is the SURVIVAL INTEGRAL of the
    histogram (report Eq. eq:survival) and is therefore smooth and
    monotone however ragged the histogram is.
    """
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42,
                          wspace=0.28)
    PHI_COL = 100.0
    KSET = ((np.inf, "C0", r"$k\to\infty$: all at the far end"),
            (2.0, "C1", r"$k=2$: weighted outward"),
            (0.0, "C2", r"$k=0$: uniform (fiducial)"),
            (-0.5, "C4", r"$k=-0.5$: weighted toward us"))
    a1 = fig.add_subplot(gs[0, :2])
    f = np.linspace(1e-3, 1, 800)
    for k, col, lab in KSET:
        if np.isinf(k):
            a1.plot([1, 1], [0, 3.0], color=col, lw=3, label=lab)
            a1.annotate("", xy=(1.0, 3.0), xytext=(1.0, 2.4),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
        else:
            a1.plot(f, (k + 1) * f**k, color=col, lw=2, label=lab)
    a1.set(xlim=(-0.03, 1.10), ylim=(0, 3.2),
           xlabel=r"position along the column  $f$   "
                  r"(0 = observer, 1 = far end)",
           ylabel=r"emissivity $\rho(f)=(k{+}1)f^k$")
    a1.set_title("(a) ONE sightline: where the polarized emission sits "
                 "(each curve has unit area)", fontsize=11)
    a1.legend(fontsize=8, loc="upper left", framealpha=0.95)
    a1.text(0.62, 2.62, r"emission at $f$ is rotated only by the fraction"
            "\n" r"in front of it:   $\phi = f\,\phi_{\rm col}$",
            ha="center", fontsize=9.5, color="0.3")

    a2 = fig.add_subplot(gs[0, 2])
    e = np.arange(0.0, 160.0, 1.0)
    c = dsp.phi_centers(e)
    ge = np.concatenate([[-1.0], e])
    for k, col, _ in KSET:
        H = dsp.depth_distribution(np.array([PHI_COL]), np.array([1.0]),
                                   ge, k=k)
        a2.plot(np.concatenate([[-0.5], c]), H, color=col, lw=2)
    a2.axvline(PHI_COL, color="0.6", ls=":", lw=1)
    a2.text(PHI_COL, 0.55, r" $\phi_{\rm col}$", fontsize=9, color="0.4")
    a2.set(xlim=(-6, 150), ylim=(0, 0.62), xlabel=r"$\phi$ [rad m$^{-2}$]",
           ylabel="mass per bin")
    a2.set_title(r"(b) its contribution to $F(\phi)$", fontsize=11)

    a3 = fig.add_subplot(gs[1, 0])
    tot = np.zeros(c.size)
    for x in (40.0, 65.0, 90.0, 120.0, 160.0):
        H = dsp.depth_distribution(np.array([x]), np.array([1.0]), ge,
                                   k=0.0)[1:]
        a3.plot(c, H, color="0.75", lw=1)
        tot += H
    a3.plot(c, tot, color="C2", lw=2.2, label="sum")
    a3.set(xlim=(0, 160), xlabel=r"$\phi$ [rad m$^{-2}$]",
           ylabel="mass per bin")
    a3.set_title("(c) five columns at $k=0$: nested top hats,\n"
                 "each running from 0 to its OWN $\\phi_{\\rm col}$",
                 fontsize=10)
    a3.legend(fontsize=8)

    a4 = fig.add_subplot(gs[1, 1:])
    ki = int(np.argmax(np.where(np.isfinite(d["ks"]), d["ks"], np.inf)))
    kf = int(np.argmin(np.abs(d["ks"])))
    for ik, col, lab in ((ki, "C0", r"$k\to\infty$: the histogram itself"),
                         (kf, "C2", r"$k=0$: its survival integral")):
        a4.plot(d["phi"], d["H"][0, ik], color=col, lw=1.3, label=lab)
    a4.set(xscale="log", yscale="log", xlim=(0.5, 2600), ylim=(1e-8, 1e-1),
           xlabel=r"$|\phi|$ [rad m$^{-2}$]",
           ylabel=r"$\hat H(|\phi|)$")
    a4.set_title(r"(d) the real sky at 30 MHz: an INTEGRAL of a ragged "
                 r"histogram is smooth", fontsize=10)
    a4.legend(fontsize=8, loc="lower left")
    return fig


def fig_data(q):
    """What LuSEE-Night measures, and how a Faraday signal shows up.

    Drawn from the report's own signal model -- P a Gaussian field of
    covariance A^2 S -- so these are realizations of exactly what the
    analysis claims the instrument sees, not a cartoon.

    The point of (c) is that ONE sample is not a detection and is not
    meant to be: no spectrum is ever predicted here, a COVARIANCE is,
    and (d) is the object the matched filter actually tests.
    """
    nu = np.linspace(29.9625, 30.0371, q["demo_spectra"].shape[1])
    dl2 = np.abs(q["lam2_bins"] - q["lam2_bins"][0])
    lab = ("no Faraday  ($F=\\delta(0)$)", "one depth  $\\phi_0=20$",
           "the Galactic sky")
    cols = ("C7", "C1", "C2")
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.6))
    for i in range(3):
        P = q["demo_spectra"][i]
        ax[0, 0].plot(q["lam2_bins"], np.unwrap(2 * np.angle(P)) / 2,
                      color=cols[i], lw=1.4, label=lab[i])
    ax[0, 0].set(xlabel=r"$\lambda^2$ [m$^2$]",
                 ylabel=r"$\chi=\frac{1}{2}\arg(Q+iU)$ [rad]")
    ax[0, 0].set_title("(a) noise-free: Faraday rotation winds $\\chi$ "
                       "linearly in $\\lambda^2$", fontsize=10)
    ax[0, 0].legend(fontsize=7.5)
    for i in (0, 2):
        P = q["demo_spectra"][i]
        ax[0, 1].plot(nu, P.real, color=cols[i], lw=1.3, label=f"Q, {lab[i]}")
        ax[0, 1].plot(nu, P.imag, color=cols[i], lw=1.0, ls=":",
                      label=f"U, {lab[i]}")
    ax[0, 1].set(xlabel="frequency [MHz]", ylabel="Q, U  (arbitrary)")
    ax[0, 1].set_title("(b) noise-free $Q(\\nu)$, $U(\\nu)$: flat vs "
                       "decorrelating", fontsize=10)
    ax[0, 1].legend(fontsize=6.5, ncol=2)
    ax[1, 0].plot(nu, q["one_sample"].real, color="0.35", lw=0.9,
                  label="$Q$ measured (signal + noise)")
    ax[1, 0].plot(nu, q["one_signal"].real, color="C2", lw=1.6,
                  label="the Faraday signal in it")
    ax[1, 0].set(xlabel="frequency [MHz]", ylabel="$Q\\,/\\,T_{\\rm sys}$")
    ax[1, 0].set_title(f"(c) ONE {float(q['tau_s']):.0f}-s sample at the slab "
                       f"floor: signal is {float(q['sample_snr']):.2f}x the "
                       "noise", fontsize=10)
    ax[1, 0].legend(fontsize=7.5)
    for i, lun in enumerate(q["coadd_lunations"]):
        ax[1, 1].plot(dl2, q["coadd_S"][i], lw=1.0, alpha=0.85,
                      color=plt.cm.viridis(i / max(len(q["coadd_lunations"])
                                                   - 1, 1)),
                      label=f"measured, {int(lun)} lunation"
                            f"{'s' if lun > 1 else ''}")
    ax[1, 1].plot(dl2, q["S_fiducial"], "k-", lw=2.2,
                  label="predicted, Faraday")
    ax[1, 1].plot(dl2, q["S_no_faraday"], color="0.4", ls="--", lw=1.6,
                  label="predicted, NO Faraday")
    ax[1, 1].set(xlabel=r"channel separation $|\Delta\lambda^2|$ [m$^2$]",
                 ylabel=r"$|S_{ij}|$", ylim=(-0.4, 1.6))
    ax[1, 1].set_title("(d) the covariance is what is predicted, and what "
                       "integration recovers", fontsize=10)
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_two_axes(q):
    """The TWO axes that build F, and what each does to the covariance.

    Columns are the map's spread of COLUMN depths (axis 1, between
    sightlines); rows are where emission sits ALONG each column (axis 2,
    k).  Column 1 is the proof they are independent: the same constant
    RM map is delta(50) at k -> inf and a top hat at k = 0.

    The panels depend on each toy map only through its weighted
    histogram, never its arrangement -- which is why an alternating
    checkerboard is a legitimate stand-in for a two-valued sky.
    """
    names = [str(x) for x in q["sky_names"]]
    phi, dl2 = q["phi"], np.abs(q["lam2_bins"] - q["lam2_bins"][0])
    KL = (r"$k\to\infty$: all emission behind",
          r"$k=0$: emission throughout")
    KC = ("C0", "C2")
    fig, ax = plt.subplots(3, len(names), figsize=(14, 7.4))
    for j, name in enumerate(names):
        for r in (0, 1):
            a = ax[r, j]
            a.plot(phi, q["toy_H"][j, r], color=KC[r], lw=1.3)
            a.set(xscale="log", yscale="log", xlim=(0.4, 2600),
                  ylim=(1e-9, 3))
            a.set_xlabel(r"$|\phi|$ [rad m$^{-2}$]", fontsize=8)
            a.tick_params(labelsize=7)
            if j == 0:
                a.set_ylabel(r"$\hat H(|\phi|)$", fontsize=11)
            if j == len(names) - 1:
                a.text(0.97, 0.9, KL[r], transform=a.transAxes, ha="right",
                       fontsize=8, color=KC[r])
            ax[2, j].plot(dl2, q["toy_S"][j, r], color=KC[r], lw=1.5,
                          label=KL[r])
        ax[0, j].set_title(f"{name}\nCV = {q['toy_cv'][j]:.2f}", fontsize=10)
        ax[2, j].set(xlim=(0, dl2.max()), ylim=(-0.02, 1.05))
        ax[2, j].set_xlabel(r"$|\Delta\lambda^2|$ [m$^2$]", fontsize=8)
        ax[2, j].tick_params(labelsize=7)
        if j == 0:
            ax[2, j].set_ylabel(r"$|S_{ij}|$", fontsize=11)
            ax[2, j].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Two independent axes build $F(\\phi)$ at {q['band']:.0f} MHz: "
        "the map's spread of COLUMN depths (columns) and where emission "
        "sits ALONG each column (rows)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_bridge(q):
    """Where the refuted coherent sum works, and what replaces it.

    Scaling the RM map down moves the sky from the regime where the
    pixel sum converges to the regime where it is a random walk.  The
    two regimes are DISJOINT from the regime where there is a signal --
    that is the quantitative form of Section sec:randomwalk.
    """
    phi, dl2 = q["phi"], np.abs(q["lam2_bins"] - q["lam2_bins"][0])
    ns, sc = q["nsides"], q["scales"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for i, s in enumerate(sc):
        col = plt.cm.viridis(i / (len(sc) - 1))
        lab = (f"RM $\\times\\,10^{{{int(np.log10(s))}}}$" if s < 1
               else "real RM")
        a1.plot(ns, q["coherent_norm"][i], "o-", color=col, lw=1.4, ms=4,
                label=lab)
        a2.plot(phi, q["scan_H"][i], color=col, lw=1.3)
        a3.plot(dl2, q["scan_S"][i], color=col, lw=1.4)
    ref = q["coherent_norm"][-1, 0] * (ns / ns[0]) ** -1.0
    a1.plot(ns, ref, "k:", lw=1.2, label=r"$N_{\rm pix}^{-1/2}$ slope")
    a1.set(xscale="log", yscale="log", xlabel="nside",
           ylabel=r"$|\sum_n w_n e^{2i\phi_n\lambda^2}|\,/\,\sum_n w_n$")
    a1.set_title("(1) the coherent sum:\nflat = converges, sloped = shot "
                 "noise", fontsize=10)
    # a log axis lays its own minor ticks under the manual ones and the
    # two label sets overprint; kill the minors explicitly.
    a1.xaxis.set_minor_locator(NullLocator())
    a1.xaxis.set_major_locator(FixedLocator(ns))
    a1.set_xticklabels([str(int(n)) for n in ns])
    a1.legend(fontsize=6.5, loc="lower left")
    a2.set(xscale="log", yscale="log", xlim=(0.4, 2600), ylim=(1e-8, 3),
           xlabel=r"$|\phi|$ [rad m$^{-2}$]",
           ylabel=r"$\hat H(|\phi|)$")
    a2.set_title("(2) what replaces it:\nthe depth distribution",
                 fontsize=10)
    a3.set(xlim=(0, dl2.max()), ylim=(-0.02, 1.05),
           xlabel=r"channel separation $|\Delta\lambda^2|$ [m$^2$]",
           ylabel=r"$|S_{ij}|$")
    a3.set_title("(3) and its channel covariance:\nbroader $F$ = faster "
                 "decorrelation", fontsize=10)
    fig.tight_layout()
    return fig


def fig_robustness(q):
    """How sensitive the prediction is to inputs we do not control."""
    dl2 = np.abs(q["lam2_bins"] - q["lam2_bins"][0])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(dl2, q["S_fiducial"], "k-", lw=2.4, label="fiducial", zorder=5)
    for i, nm in enumerate([str(x) for x in q["variant_names"]]):
        dmax = np.abs(q["S_variants"][i] - q["S_fiducial"]).max()
        ax.plot(dl2, q["S_variants"][i], color=f"C{i}", lw=1.5,
                label=f"{nm}  (max$|\\Delta S|$ = {dmax:.3f})")
    ax.set(xlim=(0, dl2.max()), ylim=(-0.02, 1.05),
           xlabel=r"channel separation $|\Delta\lambda^2|$ [m$^2$]",
           ylabel=r"$|S_{ij}|$")
    ax.set_title("Sensitivity of the prediction to inputs\nwe do not "
                 "control", fontsize=11)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_template_family(d):
    """The template family: shape against emissivity geometry, per band.

    Colour is the geometry, linestyle is the treatment, and the two are
    kept separate -- see K_COLORS.

    ``k -> -1`` (all emission local, in front of the whole rotating
    column) is ``delta(phi)``: it has exactly ONE non-zero bin out of
    2500, so there is no line to see, and it is drawn as a marker whose
    legend entry says so rather than as a phantom curve at the axis
    edge.

    The COHERENCE TILT is deliberately not drawn here any more.  It is
    the shape-space sibling of ``amplitude_bracket``'s ``upper``: both
    need ``theta_c``, which this map cannot determine (it clamps at the
    low edge of the sampled structure function, with the true root
    below it and below the nside-512 pixel scale), and ``patch_counts``
    floors at 1 on every bin above phi ~ 400 so the tilted tail is set
    by that floor rather than by anything measured.  Drawn as a peer
    curve it read as a live alternative that moves the detection ratio
    by ~1.8x on an uncomputable input.  ``H_coh`` is still in the npz
    and the report states the number in its "what this map cannot
    decide" section, exactly as ``upper`` is handled.
    """
    bands = d["bands"]
    phi = d["phi"]
    kf = int(np.argmin(np.abs(d["ks"])))  # fiducial, k = 0
    kd = int(np.argmin(d["ks"]))  # degenerate, k -> -1
    fig, axes = plt.subplots(
        1, len(bands), figsize=(4 * len(bands), 3.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ib, (ax, band) in enumerate(zip(axes, bands)):
        for ik in range(len(d["ks"])):
            if ik != kd:
                ax.plot(phi, d["H"][ib, ik], color=K_COLORS[ik], lw=1.1)
        j = int(np.argmax(d["H"][ib, kd]))
        ax.plot(
            phi[j],
            d["H"][ib, kd][j],
            ls="none",
            marker="v",
            ms=8,
            color=K_COLORS[kd],
        )
        for q, lab in zip(d["weighted_percentiles"], PCT_LABELS):
            ax.axvline(q, color="0.85", lw=0.6, zorder=0)
            # Rotated and INSIDE the axes: horizontal labels at the
            # top collided with the panel title (p50) and with each
            # other (p99 against p99.9, 776 and 1283 rad/m^2 being
            # close on a log axis).
            ax.annotate(
                lab,
                xy=(q, 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(-2, 0),
                textcoords="offset points",
                ha="right",
                va="top",
                rotation=90,
                fontsize=6,
                color="0.45",
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(0.4, 2600),
            ylim=(1e-8, 3.0),
            title=f"{band:.0f} MHz",
            xlabel=r"$\phi$ [rad m$^{-2}$]",
        )
    axes[0].set_ylabel("normalised template")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=K_COLORS[0], lw=1.1, label=K_LABELS[0]),
            Line2D([0], [0], color=K_COLORS[kf], lw=1.1, label=K_LABELS[kf]),
            Line2D(
                [0],
                [0],
                color=K_COLORS[kd],
                lw=0,
                marker="v",
                ms=8,
                label=K_LABELS[kd] + r": $\delta(\phi)$, one bin",
            ),
        ],
        fontsize=6.5,
        loc="lower left",
        framealpha=0.9,
    )
    fig.tight_layout()
    return fig


def fig_kscan(d):
    """The emissivity geometry as a CONTINUOUS knob, three statistics.

    fig_template_family draws three geometries because those are the
    three the report quotes; this draws the curve between them.  The
    three panels are deliberately not the same curve rescaled: the
    retained fraction is the intermediate product, the knee is the
    shape statistic the robustness table uses, and the detection ratio
    is the deliverable -- and only the last carries the matched-filter
    threshold, recomputed on the truncated template at every k.

    k -> inf cannot sit on a linear k axis, so it is the dashed
    horizontal each curve approaches; k = -1 is off the left edge by
    construction (f = 0), and the axis stops short rather than
    pretending to reach it.
    """
    ks, bands = d["ks"], d["bands"]
    cut = float(d["safe_cut"])
    ic = int(np.argmin(np.abs(d["cuts"] - cut)))
    i0 = int(np.argmin(np.abs(ks)))
    panels = (
        ("power_fraction", "power_fraction_kinf",
         f"power fraction retained at $\\phi\\geq{cut:g}$"),
        ("knee", "knee_kinf", "90%-mass knee [rad m$^{-2}$]"),
        ("ratio_slab", "ratio_slab_kinf",
         "detection ratio at the slab floor"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), sharex=True)
    for ax, (key, key_inf, ylabel) in zip(axes, panels):
        for ib, band in enumerate(bands):
            # power_fraction is (band, cut, k) and carries a cut axis;
            # knee and ratio_slab are (band, k) at the safe cut only.
            y = d[key][ib, ic] if d[key].ndim == 3 else d[key][ib]
            y_inf = (
                d[key_inf][ib, ic] if d[key_inf].ndim == 2 else d[key_inf][ib]
            )
            ax.plot(ks, y, color=f"C{ib}", label=f"{band:.0f} MHz")
            ax.axhline(y_inf, color=f"C{ib}", ls="--", lw=0.8, alpha=0.7)
            ax.plot(ks[i0], y[i0], "o", color=f"C{ib}", ms=4.5, zorder=5)
        ax.axvline(0.0, color="0.7", lw=0.7, zorder=0)
        ax.set(xlabel="emissivity index $k$   ($\\rho(f)\\propto f^k$)",
               ylabel=ylabel, ylim=(0, None))
    # The deliverable panel is the only one with an absolute meaning:
    # ratio 1 is the 5-sigma detection.  Log y, because 30 and 50 MHz
    # differ by 4x and a linear axis flattens the 10 MHz curve onto 0.
    axes[2].set(yscale="log", ylim=(0.3, None))
    axes[2].axhline(1.0, color="0.25", lw=1.0, ls=":", zorder=1)
    axes[2].annotate("$5\\sigma$", xy=(ks[-1], 1.0), xytext=(-4, 3),
                     textcoords="offset points", ha="right", fontsize=7,
                     color="0.25")
    axes[0].annotate("fiducial\n$k=0$",
                     xy=(0.0, d["power_fraction"][0, ic, i0]),
                     xytext=(2.0, 0.09), fontsize=6.5, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))
    axes[0].legend(
        handles=[Line2D([0], [0], color=f"C{ib}", lw=1.3, label=f"{b:.0f} MHz")
                 for ib, b in enumerate(bands)]
        + [Line2D([0], [0], color="0.4", ls="--", lw=0.8,
                  label=r"$k\to\infty$ asymptote")],
        fontsize=6.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_knee_tail(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for ib, band in enumerate(d["bands"]):
        a1.plot(
            d["lst_hours"],
            d["tail_frac_lst"][ib],
            label=f"{band:.0f} MHz",
        )
    a1.set(
        yscale="log",
        xlabel="LST [h]",
        ylabel="template power fraction above beam-weighted p99",
        title="the S4.2.2 tail gate",
    )
    a1.legend(fontsize=7)
    x = np.arange(d["knee"].shape[1])
    for ib, band in enumerate(d["bands"]):
        a2.plot(x, d["knee"][ib], "o-", label=f"{band:.0f} MHz")
        a2.plot(x, d["knee_taper"][ib], "s--", alpha=0.6)
    # R10: the "knee" fields are the 90% mass quantile of the folded
    # template (dsp.mass_quantile_knee), not a half-power knee -- an
    # earlier half-power version was measured to be defective and
    # replaced.  Label accordingly.
    a2.set(
        xticks=x,
        xticklabels=["inf", "0", "-1"],
        xlabel="k",
        ylabel=r"90% mass knee [rad m$^{-2}$]",
        title="knee vs geometry (dashed: plane-tapered)",
    )
    a2.legend(fontsize=7)
    # No fig.tight_layout() here: a1's rotated ylabel ("template power
    # fraction above beam-weighted p99") is long enough that
    # tight_layout()'s pre-save subplot repositioning leaves too
    # little vertical room for it, and savefig's own
    # bbox_inches="tight" pass then clips the ends (verified with
    # pdftotext -bbox: the leading "T" and trailing "p99" fell
    # outside the page's mediabox even at pad_inches=0.25).  Skipping
    # tight_layout() and letting bbox_inches="tight" alone size the
    # canvas around the default subplot layout, with a larger pad,
    # fits the whole label with margin to spare.
    return fig


def fig_envelope(env):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(env["bands"]))
    ax.bar(x - 0.15, env["parent_horizon"], 0.3, label="parent horizon")
    ax.bar(x + 0.15, env["zoom_horizon"], 0.3, label="zoom horizon")
    for q, v in zip(env["percentile_qs"], env["percentiles"]):
        ax.axhline(v, color="0.7", lw=0.7)
        ax.text(2.55, v, f"p{q:g}", fontsize=6, va="bottom")
    ax.axhline(env["rm_max"], color="k", lw=0.9, ls="--", label="|RM| max")
    ax.set(
        yscale="log",
        xticks=x,
        xticklabels=[f"{b:.0f} MHz" for b in env["bands"]],
        ylabel=r"50% Faraday depth [rad m$^{-2}$]",
        title="the instrument's depth horizon vs the sky",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_sensitivity(s, d):
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for ib, band in enumerate(s["bands"]):
        (ln,) = ax.plot(
            s["lunations"],
            s["A_mf"][ib],
            label=f"{band:.0f} MHz matched filter",
        )
        ax.plot(
            s["lunations"],
            s["A_closed"][ib],
            ls=":",
            color=ln.get_color(),
            label=f"{band:.0f} MHz closed form",
        )
        up, sl, dis = s["bracket"][ib]
        # theta_c note: ONLY the bracket's upper end contains theta_c
        # (N_patch = Omega_beam / theta_c^2).  lower_slab and
        # lower_dispersion are closed forms in phi_med and sigma_eff
        # and are theta_c-free, so a clamped theta_c contaminates the
        # upper line ALONE -- an earlier caption said the whole
        # bracket "derives from theta_c", which was wrong about two of
        # the three lines drawn here.  And the clamp OVERSTATES (it
        # returns the grid edge when the root lies below it), so a
        # clamped upper is an upper BOUND, overstated by ~1e3 on this
        # map; widening the grid cannot fix it (the root is
        # sub-arcsecond).  Read the flag from the template npz.
        jb = int(np.argmin(np.abs(d["bands"] - band)))
        clamped = bool(d["theta_c_clamped"][jb])
        tag = " (clamp bound)" if clamped else ""
        ax.axhspan(dis, up, color=ln.get_color(), alpha=0.06)
        # `sl` (lower_slab) sits strictly inside (dis, up) and is a
        # physically distinct bound from `dis` (lower_dispersion) --
        # orders of magnitude apart, per S4.4.1 -- so it is drawn as
        # its own interior line rather than discarded; one shared
        # proxy legend entry below covers all three bands, per-band
        # color already established by the matched-filter line.
        ax.axhline(sl, color=ln.get_color(), lw=0.5, ls="-.", alpha=0.8)
        ax.axhline(
            up,
            color=ln.get_color(),
            lw=0.6,
            ls="--",
            label=f"{band:.0f} MHz bracket upper{tag}",
        )
    # R24/R28: A_mf/A_closed are fractions of T_sys; the bracket is a
    # fractional polarized amplitude referred to T_sky -- a different
    # reference temperature.  tsys_over_tsky is within 1.7% of unity
    # for every band here, so the mismatch is quantitatively
    # negligible -- but it is not zero, and nothing is rescaled, so
    # the axis label states the reference for the plotted curves and
    # the caption below states the bracket's (different) reference
    # rather than leaving it for the reader to guess.
    # The ratio is loading-only: --t-amp defaults to 0 and the luseepy
    # chain carries no amplifier noise, so it is 1 + T_loading/T_sky
    # and a LOWER BOUND on T_sys/T_sky, not a sky-domination result.
    # The caption says so; do not let 1.7% read as "computed".
    ratio = np.asarray(s["tsys_over_tsky"], dtype=float)
    mismatch_pct = 100.0 * np.nanmax(np.abs(ratio - 1.0))
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="lunations",
        ylabel=r"5$\sigma$ threshold (fraction of $T_{sys}$)",
        title="threshold vs the S4.4 amplitude bracket",
    )
    handles, labels = ax.get_legend_handles_labels()
    slab_proxy = Line2D(
        [0],
        [0],
        color="0.3",
        lw=0.5,
        ls="-.",
        label="bracket lower (slab)",
    )
    ax.legend(
        handles + [slab_proxy],
        labels + [slab_proxy.get_label()],
        fontsize=6,
        loc="upper left",
        ncol=1,
        framealpha=0.85,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    # Explicit line breaks, NOT wrap=True: a wrapped Text reports its
    # pre-wrap extent to the tight-bbox pass, which pushed the first
    # word to xMin = 0.5 pt (flush against the page edge, verified
    # with pdftotext -bbox).  Hard-wrapped lines measure correctly and
    # keep the pad_inches margin.
    fig.text(
        0.5,
        0.01,
        "Bracket (shaded / dashed upper / dash-dot lower-slab) is a "
        r"fractional polarized amplitude referred to $T_{sky}$, not "
        r"$T_{sys}$"
        f"\n($T_{{sys}}/T_{{sky}}$ within {mismatch_pct:.1f}% of 1, "
        "loading only -- no amplifier noise in this chain -- and not "
        "rescaled here).\nThe dashed UPPER line alone derives from "
        r"$\theta_c$, which is clamped to the search grid at every "
        "band shown: it is a\nclamp-derived upper bound, overstated "
        "by ~3 decades, and is not computable from this map.\nThe "
        "dash-dot lower-slab and shaded lower-dispersion ends contain "
        r"no $\theta_c$ and stand.",
        ha="center",
        va="bottom",
        fontsize=6,
        linespacing=1.4,
    )
    return fig


def fig_chirp_coherence():
    freqs = fine_freqs(30.0)[::4]
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    phi0 = 600.0
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(500.0, 700.0, 0.1)
    p_nufft = dsp.depth_power(spec, freqs, phi_out)
    n = freqs.size
    P = np.abs(np.fft.fftshift(np.fft.fft(spec)) / n) ** 2
    bw = (freqs[1] - freqs[0]) * 1e6 * n
    k = np.arange(n) - n // 2
    # R1: lambda_squared() returns an array even for scalar input;
    # float(np.asarray(...)) raises under NumPy 2.x on a >1-element
    # array in general and is fragile here regardless -- index [0].
    lam2_0 = lambda_squared(30.0)[0]
    # R22: the brief's sign here was POSITIVE and wrong -- verified by
    # measurement (see task report): with the positive sign the
    # window below captures the numerical noise floor
    # (P[sel].max()/P.max() ~ 1e-6), not the injected tone. Negative
    # sign puts the true peak inside the unmodified window (ratio 1.0).
    phi_fft = -np.pi * k * 30e6 / (2.0 * bw * lam2_0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(phi_out, p_nufft / p_nufft.max(), label="type-3 NUFFT")
    sel = (phi_fft > 500) & (phi_fft < 700)
    ax.plot(
        phi_fft[sel],
        P[sel] / P[sel].max(),
        label="uniform-$\\nu$ FFT (the chirp)",
    )
    ax.set(
        xlabel=r"$\phi$ [rad m$^{-2}$]",
        ylabel="normalised power",
        title=r"a single depth at $\phi_0 = 600$, 30 MHz",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_two_arm(d, d2):
    fig, axes = plt.subplots(
        1, len(d["bands"]), figsize=(4 * len(d["bands"]), 3.2), sharey=True
    )
    for ib, (ax, band) in enumerate(zip(np.atleast_1d(axes), d["bands"])):
        ax.plot(d["phi"], d["H"][ib, 1], label="as-built four-port")
        ax.plot(d2["phi"], d2["H"][ib, 1], label="symmetric two-port")
        ax.set(
            xscale="log",
            yscale="log",
            title=f"{band:.0f} MHz",
            xlabel=r"$\phi$ [rad m$^{-2}$]",
            ylim=(1e-8, None),
        )
    np.atleast_1d(axes)[0].set_ylabel("normalised template (k = 0)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_detection(s):
    """Detection SNR against the low-depth systematics cut.

    The deliverable for the DETECTION question, as opposed to the
    template family (the shape question) and the tail figure (the
    localisation question).  One panel per band; the top axis is the
    same cut expressed as a delay, because ``tau_FD`` is monotonic in
    ``phi`` and every cut here is exactly a cut in either basis.

    ``cuts[0]`` is the no-cut case and cannot go on a log axis, so it
    is annotated rather than plotted.
    """
    bands = s["bands"]
    cuts, tau_us = s["cuts"], s["tau_us"]
    keep = cuts > 0
    fig, allax = plt.subplots(
        2, len(bands), figsize=(4.2 * len(bands), 6.4), sharex="col",
    )
    allax = np.atleast_2d(allax)
    # Top row: the two quantities the ratio is made of, so the gap
    # between them IS the answer.  A_mf is in units of T_sys and the
    # bracket is referred to the sky; the mismatch is the <=1.7%
    # T_sys/T_sky of the caveats section and is invisible here.
    for ib, band in enumerate(bands):
        ax = allax[0, ib]
        f = np.sqrt(s["power_fraction"][ib][keep])
        ax.plot(cuts[keep], s["a_mf"][ib][keep], "k-", lw=1.4,
                label=r"$A_{5\sigma}$ threshold")
        ax.plot(cuts[keep], s["bracket"][ib, 1] * f, "o-", color="C0",
                lw=1.2, ms=3.5, label=r"signal, $A_{\rm slab}\sqrt{f}$")
        ax.plot(cuts[keep], s["bracket"][ib, 2] * f, "s--", color="C3",
                lw=1.2, ms=3.5, label=r"signal, $A_{\rm disp}\sqrt{f}$")
        ax.axvspan(cuts[keep].min(), 27.5, color="0.9", zorder=0)
        ax.set(xscale="log", yscale="log", title=f"{band:.0f} MHz")
        if ib == 0:
            ax.set_ylabel("fractional polarized amplitude")
            ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    for ib, (ax, band) in enumerate(zip(allax[1], bands)):
        ax.plot(
            cuts[keep],
            s["ratio_slab"][ib][keep],
            "o-",
            color="C0",
            lw=1.2,
            ms=3.5,
            label="uniform-slab floor",
        )
        ax.plot(
            cuts[keep],
            s["ratio_dispersion"][ib][keep],
            "s--",
            color="C3",
            lw=1.2,
            ms=3.5,
            label="internal-dispersion floor",
        )
        ax.axhline(1.0, color="k", lw=0.9, ls="-", zorder=0)
        # Below the window-budget cut the statistic is degenerate with
        # spectrally smooth I -> Q,U leakage, which sits at phi ~ 0.
        ax.axvspan(cuts[keep].min(), 27.5, color="0.9", zorder=0)
        ax.annotate(
            f"no cut: {s['ratio_slab'][ib][0]:.0f} / "
            f"{s['ratio_dispersion'][ib][0]:.3f}",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize=6,
            color="0.35",
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"cut: $\phi \geq$ [rad m$^{-2}$]",
        )
        # tau = (2 c^2 / pi nu^3) phi -- linear, so one scale factor.
        # It goes on the TOP row's upper edge; on the bottom row it
        # would land inside the panel above it.
        k = tau_us[ib][keep][-1] / cuts[keep][-1]
        sec = allax[0, ib].secondary_xaxis(
            "top", functions=(lambda x, k=k: x * k, lambda t, k=k: t / k)
        )
        sec.set_xlabel(r"same cut as a delay $\tau$ [$\mu$s]", fontsize=8)
        sec.tick_params(labelsize=7)
    allax[1, 0].set_ylabel(
        r"detection ratio  $A\sqrt{f}\,/\,A_{5\sigma}$"
    )
    allax[1, 0].legend(fontsize=7, loc="upper right")
    # The reserved strip is a FRACTION of the figure height, so it was
    # sized for the old single-row figure; at two rows the same 0.11
    # leaves a blank band the height of a panel.
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    fig.text(
        0.5,
        0.055,
        "Above the black line the floor is detectable at 5$\\sigma$ in "
        f"{int(s['lunations'])} lunations. Shading marks the region where "
        "the statistic is degenerate\nwith spectrally smooth "
        "$I\\rightarrow Q,U$ leakage. The threshold is recomputed on the "
        "TRUNCATED template at every cut, not taken from the full one.",
        ha="center",
        va="top",
        fontsize=7.5,
        linespacing=1.4,
    )
    return fig


def fig_weight_map(d):
    """The LST- and pair-averaged Faraday weight, as a sky map.

    What "beam-weighted" means, in one picture: every sky percentile
    and every template in this set is weighted by this map.  healpy
    makes its own figure, so this returns plt.gcf() rather than a
    figure it created -- passing a pre-made one only makes mollview
    warn that it is ignoring the figsize.
    """
    import healpy as hp

    w2 = d["w2_mean"]
    nside = hp.npix2nside(w2.size)
    hp.mollview(
        w2 / w2.max(),
        norm="log",
        min=1e-4,
        max=1.0,
        title=(
            f"LST- and pair-averaged $|w|^2$ " f"(nside {nside}, normalised)"
        ),
        unit="relative weight",
        cmap="viridis",
    )
    hp.graticule(30)
    return plt.gcf()


def save(fig, name, pad=0.25):
    """Write one figure to FIG_DIR/<name>.pdf and close it.

    The ``fig_*`` builders return their figure rather than saving it,
    so that ``notebooks/faraday_delay_template.ipynb`` renders the
    same figures the paper uses instead of carrying a second copy of
    the plotting code that can drift from this one.  Saving lives
    here; the notebook displays instead.
    """
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def main():
    d = np.load(GEN_DIR / "step5_template.npz")
    save(fig_template_family(d), "step5_template_family")
    save(fig_pushforward(d), "step5_pushforward")
    q = GEN_DIR / "step5_intuition.npz"
    if q.exists():
        qq = np.load(q)
        save(fig_data(qq), "step5_data")
        save(fig_two_axes(qq), "step5_two_axes")
        save(fig_bridge(qq), "step5_bridge")
        save(fig_robustness(qq), "step5_robustness")
    inp = GEN_DIR / "step5_inputs.npz"
    if inp.exists():
        save(fig_inputs(np.load(inp), d), "step5_inputs", pad=0.15)
    ksc = GEN_DIR / "step5_kscan.npz"
    if ksc.exists():
        save(fig_kscan(np.load(ksc)), "step5_kscan")
    # pad 0.4: fig_knee_tail deliberately skips tight_layout (see the
    # comment there), so its long rotated ylabel needs the wider pad.
    save(fig_knee_tail(d), "step5_knee_tail_lst", pad=0.4)
    save(
        fig_envelope(np.load(GEN_DIR / "step5_envelope.npz")), "step5_envelope"
    )
    s = np.load(GEN_DIR / "step5_sensitivity.npz")
    save(fig_sensitivity(s, d), "step5_sensitivity")
    save(fig_chirp_coherence(), "step5_chirp_coherence")
    save(fig_weight_map(d), "step5_weight_map")
    det = GEN_DIR / "step5_detection.npz"
    if det.exists():
        save(fig_detection(np.load(det)), "step5_detection", pad=0.4)
    two = GEN_DIR / "step5_template_two_port.npz"
    if two.exists():
        save(fig_two_arm(d, np.load(two)), "step5_two_arm")
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
