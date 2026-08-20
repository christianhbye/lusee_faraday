import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)          # data/ paths are repo-relative
SP = str(_OUT) + "/"      # generated artefacts land here

import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

res = json.load(open(SP + "shuffle_results.json"))

INK, MUTE = "#1a1a1a", "#707070"
REAL, SHUF, GUIDE, BAND = "#1f6feb", "#e8590c", "#9a9a9a", "#f0f0f0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c8c8c8",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(13.2, 4.6))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.18, wspace=0.30,
                      left=.055, right=.985, top=.80, bottom=.13)
aA1 = fig.add_subplot(gs[0, 0]); aA2 = fig.add_subplot(gs[1, 0], sharex=aA1, sharey=aA1)
aB  = fig.add_subplot(gs[:, 1]); aC = fig.add_subplot(gs[:, 2])
for a in (aA1, aA2, aB, aC):
    a.set_facecolor("white")

# ---------- A: each spectrum vs its own weighted RM histogram
ho = np.load(SP + "hist_overlay.npz")
phi = ho["phi"]
LIM, NB = 2600.0, 68
edges = np.linspace(-LIM, LIM, NB + 1)
ctr = 0.5 * (edges[:-1] + edges[1:])

def band(x, y):
    """mean POWER per bin -> amplitude, so measurement and prediction bin alike"""
    i = np.digitize(x, edges)
    out = np.full(NB, np.nan)
    for k in range(1, NB + 1):
        m = i == k
        if m.sum():
            out[k - 1] = y[m].mean()
    return np.sqrt(out)

for a_, tag, c, lab in ((aA1, "real", REAL, "real RM map"),
                        (aA2, "shuf", SHUF, "RM values shuffled")):
    m = np.abs(phi) <= LIM
    a_.semilogy(phi[m], np.sqrt(ho["P_" + tag][m]), color=c, lw=.4, alpha=.22,
                zorder=1)
    a_.semilogy(ctr, band(phi[m], ho["P_" + tag][m]), color=c, lw=2.1,
                zorder=3, label="measured, binned")
    # prediction lives in the conjugate sign convention -> reflect
    a_.semilogy(ctr, band(-phi[m], ho["H_" + tag][m]), color="#111111", lw=1.7,
                ls=(0, (4, 2.2)), zorder=4, label="predicted histogram")
    a_.set_ylim(2e-5, 1.5); a_.set_xlim(-LIM, LIM)
    a_.text(.025, .90, lab, transform=a_.transAxes, fontsize=9, color=c,
            fontweight="bold", va="top")
aA1.legend(frameon=False, fontsize=7.6, loc="lower right", handlelength=2.0,
           borderpad=.1, labelspacing=.3)
plt.setp(aA1.get_xticklabels(), visible=False)
aA2.set_xlabel(r"Faraday depth $\phi$   [rad m$^{-2}$]")
aA1.set_ylabel(r"$|F(\phi)|$", labelpad=2); aA2.set_ylabel(r"$|F(\phi)|$", labelpad=2)
aA1.set_title("A.  Both spectra are histograms", fontsize=10.5, loc="left", pad=40)
aA1.text(0, 1.045, "Dashed = the $|w|^2$-weighted RM histogram, built from the maps alone.\nIt predicts each measurement without using any sky coherence.",
         transform=aA1.transAxes, fontsize=8.2, color=MUTE, va="bottom", linespacing=1.55)

# ---------- B: nside scaling
ns = np.array([512, 1024, 2048])
for f, c, lab in ((1.0, REAL, "actual RM map"), (0.02, SHUF, r"RM $\times$ 0.02")):
    rr = np.array([res["%d_%g" % (n, f)]["real"] for n in ns])
    ss = np.array([res["%d_%g" % (n, f)]["shuffled"] for n in ns])
    aB.loglog(ns, rr, "o-", color=c, lw=2.2, ms=6.5, label=lab + " — real", zorder=3)
    aB.loglog(ns, ss, "s:", color=c, lw=1.5, ms=5, alpha=.6, label=lab + " — shuffled")
g = res["512_1"]["real"] * (ns / 512.0) ** -2.0
aB.loglog(ns, g, color=GUIDE, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
aB.text(1090, g[1] * 2.6, r"$\propto 1/N_{\rm pix}$", color=GUIDE, fontsize=9)
aB.xaxis.set_major_locator(FixedLocator(ns)); aB.xaxis.set_minor_formatter(NullFormatter())
aB.set_xticklabels([str(n) for n in ns]); aB.set_xlim(460, 2300)
aB.set_xlabel("nside"); aB.set_ylabel(r"total power  $\Sigma_\phi |F|^2$")
aB.set_title("B.  Physics plateaus, grid noise falls", fontsize=10.5, loc="left", pad=40)
aB.text(0, 1.045, "The control (RM $\\times$0.02) resolves its level-set bands,\nso it converges. The actual map tracks $1/N_{\\rm pix}$.",
        transform=aB.transAxes, fontsize=8.2, color=MUTE, va="bottom", linespacing=1.55)
aB.legend(frameon=False, fontsize=7.8, loc="lower left")

# ---------- C: frequency sweep
l2 = np.array([0.01, 0.05, 0.14, 0.56, 2.25, 8.99, 35.95, 99.86, 898.76])
v = {512:  [8.047e-1, 4.839e-1, 2.688e-1, 2.755e-2, 1.017e-2, 5.266e-3, 3.145e-3, 8.448e-3, 3.953e-3],
     1024: [8.047e-1, 4.846e-1, 2.670e-1, 3.104e-2, 7.922e-3, 6.602e-3, 2.612e-3, 2.432e-3, 3.188e-3],
     2048: [8.047e-1, 4.845e-1, 2.665e-1, 3.083e-2, 6.776e-3, 3.726e-3, 3.672e-3, 2.379e-3, 1.298e-3]}
aC.axvspan(30.0, 950.0, color=BAND, zorder=0)
aC.text(170, 3.0e-1, "LuSEE band\n(50 → 10 MHz)", fontsize=8.2, color=MUTE, ha="center")
for n, c, mk in ((512, "#9ec5fe", "^"), (1024, "#5b9bf8", "s"), (2048, REAL, "o")):
    aC.loglog(l2, v[n], mk + "-", color=c, lw=1.8, ms=5, label="nside %d" % n)
aC.set_xlabel(r"$\lambda^2$   [m$^2$]")
aC.set_ylabel(r"$|P|_{\rm Faraday}\,/\,|P|_{\rm no\,Faraday}$")
aC.set_title("C.  Where it stops converging", fontsize=10.5, loc="left", pad=40)
aC.text(0, 1.045, "Beam-integrated sky polarization, screen on / screen off.\nThe resolutions separate from $\\lambda^2\\approx1$ and never return.",
        transform=aC.transAxes, fontsize=8.2, color=MUTE, va="bottom", linespacing=1.55)
aC.legend(frameon=False, fontsize=8.5, loc="lower left")

fig.savefig(str(_OUT.parent / "faraday_evidence.png"), dpi=160, facecolor="white", bbox_inches="tight")
print("saved")
