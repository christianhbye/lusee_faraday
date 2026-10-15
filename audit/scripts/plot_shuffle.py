"""Figure: the shuffle control, plotted from the stored results.

Reads ``shuffle_results.json`` written by ``shuffle_control.py`` and draws
the two-panel summary -- the real Galactic RM map on the left, the
converged positive control (RM x 0.02) on the right.

Two conventions differ from the first version of this figure, drawn for
the 2026-08-27 briefing:

* the axis is Faraday DEPTH phi, not a delay -- ``tau_FD = 2 phi c^2 /
  (pi nu^3)`` is a different axis and the branch renamed delay -> depth
  at 38d9906, so "delay-space power" was the superseded vocabulary;
* the stored totals carry the unnormalised-FFT constant ``N sum(h^2)``,
  which made the dashed line ``N sum(h^2) sum|w|^2`` while the legend
  called it ``sum|w|^2``.  Dividing it out leaves the band-averaged
  ``<|P|^2>`` on the y axis and makes the legend literally true.  Every
  ratio and slope on the figure is unchanged.
"""

import pathlib as _p

import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
SP = str(_OUT) + "/"

# the fine grid and window of shuffle_control.py; Parseval for numpy's
# unnormalised FFT gives sum_phi |F|^2 = N sum_j |V_j h_j|^2, so this is
# the constant between the stored total and the Hann-weighted <|P|^2>.
NFINE = 16384
NORM = NFINE * (np.hanning(NFINE) ** 2).sum()

NSIDES = (512, 1024, 2048)
res = json.load(open(SP + "shuffle_results.json"))


def curve(field, f):
    return np.array([res["%d_%g" % (ns, f)][field] / NORM for ns in NSIDES])


INK, MUTE = "#1a1a1a", "#707070"
REAL, SHUF, GUIDE = "#1f6feb", "#e8590c", "#1a1a1a"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#c8c8c8",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, (aL, aR) = plt.subplots(
    1,
    2,
    figsize=(11.0, 4.6),
    sharey=True,
    gridspec_kw=dict(
        left=0.085, right=0.985, top=0.700, bottom=0.275, wspace=0.06
    ),
)
fig.patch.set_facecolor("white")


def pixel_arcmin(ns):
    return 60 * np.degrees(np.sqrt(4 * np.pi / (12 * ns**2)))


x = np.log2(np.array(NSIDES, float))
for a, f, title in (
    (aL, 1.0, "The real Galactic RM map"),
    (aR, 0.02, "Positive control: RM $\\times$ 0.02"),
):
    a.set_facecolor("white")
    a.plot(
        x,
        curve("incoh", f),
        color=GUIDE,
        lw=1.5,
        ls=(0, (4.5, 2.4)),
        zorder=2,
        label="incoherent floor  $\\sum_p |w_p|^2$",
    )
    a.plot(
        x,
        curve("shuffled", f),
        color=SHUF,
        lw=2.0,
        marker="s",
        ms=6.4,
        mec="white",
        mew=1.0,
        zorder=3,
        label="RM values shuffled across pixels",
    )
    a.plot(
        x,
        curve("real", f),
        color=REAL,
        lw=2.0,
        marker="o",
        ms=7.0,
        mec="white",
        mew=1.0,
        zorder=4,
        label="real map",
    )
    a.set_yscale("log")
    a.set_xticks(x)
    a.set_xticklabels(["%d\n%.1f$'$" % (n, pixel_arcmin(n)) for n in NSIDES])
    a.xaxis.set_minor_locator(FixedLocator([]))
    a.set_xlim(x[0] - 0.20, x[-1] + 0.20)
    a.set_xlabel("HEALPix nside  /  pixel scale", labelpad=6)
    a.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
    a.set_axisbelow(True)
    a.set_title(title, fontsize=10.5, color=INK, pad=9, loc="left")

aL.set_ylabel(
    "mean-square polarised signal  " "$\\langle |P|^2 \\rangle$", labelpad=7
)
aL.yaxis.set_minor_formatter(NullFormatter())

aL.text(
    0.975,
    0.955,
    "$4\\times$ per doubling   $\\propto 1/N_{\\rm pix}$",
    transform=aL.transAxes,
    color=REAL,
    fontsize=9.5,
    ha="right",
    va="top",
)
aL.text(
    0.030,
    0.055,
    "real = shuffled = floor\nno spatial information left",
    transform=aL.transAxes,
    fontsize=9.5,
    color=MUTE,
    va="bottom",
)

rc, sc = curve("real", 0.02), curve("shuffled", 0.02)
aR.text(
    0.300,
    0.855,
    "converges: flat",
    transform=aR.transAxes,
    color=REAL,
    fontsize=9.5,
    ha="left",
    va="top",
)
aR.text(
    0.030,
    0.055,
    "real / shuffled separates\nby %.0f$\\times$ at nside 2048"
    % (rc[2] / sc[2]),
    transform=aR.transAxes,
    color=SHUF,
    fontsize=9.5,
    va="bottom",
)

fig.suptitle(
    "Destroying every spatial correlation in the RM map changes nothing "
    "\u2014 unless the sky is weak enough to converge",
    fontsize=12,
    x=0.085,
    ha="left",
    y=0.965,
)
fig.text(
    0.085,
    0.900,
    "$y$: $|P|^2$ averaged over the 16 384-channel fine grid spanning "
    "29.9\u201330.1 MHz, $\\langle |P|^2\\rangle \\propto "
    "\\sum_\\phi |F(\\phi)|^2$ "
    "\u2014 the total power in the Faraday (RM) spectrum.",
    fontsize=9,
    color=MUTE,
    ha="left",
    va="top",
)
fig.text(
    0.085,
    0.823,
    "Arbitrary normalisation: only the ratios between the three curves, "
    "and their slopes, carry meaning.",
    fontsize=9,
    color=MUTE,
    ha="left",
    va="top",
)
h, lab = aL.get_legend_handles_labels()
fig.legend(
    h[::-1],
    lab[::-1],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9.5,
    bbox_to_anchor=(0.535, 0.008),
)

for ext in ("pdf", "png"):
    fig.savefig(SP + "shuffle_control." + ext, dpi=220, facecolor="white")
print("wrote " + SP + "shuffle_control.{pdf,png}")

print(
    "\n%-6s %-11s %-11s %-11s %-9s %-9s"
    % ("nside", "real", "shuffled", "floor", "real/shuf", "real/floor")
)
for f in (1.0, 0.02):
    print("-- RM x %g" % f)
    for i, ns in enumerate(NSIDES):
        rr, ss, ii = (curve(k, f)[i] for k in ("real", "shuffled", "incoh"))
        print(
            "%-6d %-11.4e %-11.4e %-11.4e %-9.2f %-9.2f"
            % (ns, rr, ss, ii, rr / ss, rr / ii)
        )
