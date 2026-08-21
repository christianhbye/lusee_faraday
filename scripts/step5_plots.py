"""Figures for the delay-template paper (spec S5).

Reads the step5_*.npz products; every figure is regenerable from
committed code plus data/.
"""

import common  # noqa: F401
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIG_DIR, GEN_DIR  # noqa: E402
from lusee_faraday import dispersion as dsp  # noqa: E402
from lusee_faraday.config import fine_freqs  # noqa: E402
from lusee_faraday.conventions import lambda_squared  # noqa: E402

K_LABELS = {
    0: "$k\\to\\infty$ (all far)",
    1: "$k=0$ (slab, fiducial)",
    2: "$k\\to-1$ (all local)",
}


def fig_template_family(d):
    bands = d["bands"]
    fig, axes = plt.subplots(
        1, len(bands), figsize=(4 * len(bands), 3.2), sharey=True
    )
    for ib, (ax, band) in enumerate(zip(np.atleast_1d(axes), bands)):
        for ik in range(d["H"].shape[1]):
            ax.plot(d["phi"], d["H"][ib, ik], label=K_LABELS[ik])
            ax.plot(d["phi"], d["H_coh"][ib, ik], ls=":", alpha=0.7)
        for q in d["weighted_percentiles"]:
            ax.axvline(q, color="0.8", lw=0.6, zorder=0)
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(0.5, 2600),
            title=f"{band:.0f} MHz",
            xlabel=r"$\phi$ [rad m$^{-2}$]",
        )
        ax.set_ylim(1e-8, None)
    np.atleast_1d(axes)[0].set_ylabel("normalised template")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "step5_template_family.pdf",
        bbox_inches="tight",
        pad_inches=0.25,
    )


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
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "step5_knee_tail_lst.pdf",
        bbox_inches="tight",
        pad_inches=0.25,
    )


def fig_envelope(env, d):
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
    fig.savefig(
        FIG_DIR / "step5_envelope.pdf", bbox_inches="tight", pad_inches=0.25
    )


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
        # R10/theta_c note: the bracket is derived from theta_c, which
        # is grid-limited (clamped) at every band -- read the flag
        # from the template npz rather than assuming it, and mark the
        # bracket lines as grid-limited, not measured.
        jb = int(np.argmin(np.abs(d["bands"] - band)))
        clamped = bool(d["theta_c_clamped"][jb])
        tag = ", grid-limited" if clamped else ""
        ax.axhspan(dis, up, color=ln.get_color(), alpha=0.06)
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
    ratio = np.asarray(s["tsys_over_tsky"], dtype=float)
    mismatch_pct = 100.0 * np.nanmax(np.abs(ratio - 1.0))
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="lunations",
        ylabel=r"5$\sigma$ threshold (fraction of $T_{sys}$)",
        title="threshold vs the S4.4 amplitude bracket",
    )
    ax.legend(fontsize=6, loc="upper left", ncol=1, framealpha=0.85)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    fig.text(
        0.5,
        0.01,
        "Bracket (shaded/dashed) is a fractional polarized amplitude "
        r"referred to $T_{sky}$, not $T_{sys}$ ($T_{sys}/T_{sky}$ "
        f"mismatch $\\leq$ {mismatch_pct:.1f}%, not rescaled here); "
        "it derives from theta_c, which is grid-limited at every "
        "band shown -- a search-grid artifact, not a measurement.",
        ha="center",
        va="bottom",
        fontsize=6,
        wrap=True,
    )
    fig.savefig(
        FIG_DIR / "step5_sensitivity.pdf", bbox_inches="tight", pad_inches=0.25
    )


def fig_chirp_coherence():
    freqs = fine_freqs(30.0)[::4]
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    phi0 = 600.0
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(500.0, 700.0, 0.1)
    p_nufft = dsp.delay_power(spec, freqs, phi_out)
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
    fig.savefig(
        FIG_DIR / "step5_chirp_coherence.pdf",
        bbox_inches="tight",
        pad_inches=0.25,
    )


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
    fig.savefig(
        FIG_DIR / "step5_two_arm.pdf", bbox_inches="tight", pad_inches=0.25
    )


def main():
    d = np.load(GEN_DIR / "step5_template.npz")
    fig_template_family(d)
    fig_knee_tail(d)
    fig_envelope(np.load(GEN_DIR / "step5_envelope.npz"), d)
    fig_sensitivity(np.load(GEN_DIR / "step5_sensitivity.npz"), d)
    fig_chirp_coherence()
    two = GEN_DIR / "step5_template_two_port.npz"
    if two.exists():
        fig_two_arm(d, np.load(two))
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
