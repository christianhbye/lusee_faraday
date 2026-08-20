"""What we are actually multiplying Q+iU by.

Mollview of the Hutschenreuter RM looks smooth and structured.  The
quantity the simulation applies is exp(2i RM lambda^2), whose phase at
LuSEE wavelengths is many thousands of radians -- so what the sky is
really being multiplied by is the map below, wrapped into [-pi, pi).
The bottom panel is the same statement quantitatively: how far you must
move on the sky before the phase turns by pi, against the pixel size.
"""
import os as _os, pathlib as _p
_ROOT = _p.Path(__file__).resolve().parents[2]
_OUT = _p.Path(__file__).resolve().parent.parent / "generated"
_OUT.mkdir(exist_ok=True)
_os.chdir(_ROOT)
SP = str(_OUT) + "/"

import numpy as np, healpy as hp, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 299792458.0
INK, MUTE = "#1a1a1a", "#707070"
BANDS = [(50.0, "#1f6feb"), (30.0, "#7048e8"), (10.0, "#e8590c")]

with h5py.File("data/faraday2020v2.hdf5", "r") as f:
    RM = f["faraday_sky_mean"][:]
NSIDE = hp.npix2nside(RM.size)
grad0 = 4327.8   # rms |grad RM|, rad/m^2 per radian

plt.rcParams.update({"font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTE,
                     "ytick.color": MUTE, "axes.edgecolor": "#c8c8c8",
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13.2, 5.9))
fig.patch.set_facecolor("white")
gs_top, gs_bot = 0.98, 0.46

hp.mollview(RM, fig=fig.number, sub=(2, 4, 1), cmap="RdBu_r", min=-150, max=150,
            title="", cbar=True, unit=r"rad m$^{-2}$", format="%.0f",
            badcolor="white", bgcolor="white")
plt.title("RM map\n(what we usually look at)", fontsize=9.5, color=INK, pad=6)

pix512 = np.sqrt(4 * np.pi / hp.nside2npix(512))
maps = []
for i, (f0, _) in enumerate(BANDS):
    l2 = (C / (f0 * 1e6)) ** 2
    ph = np.angle(np.exp(2j * RM * l2))            # wrapped into [-pi, pi)
    hp.mollview(ph, fig=fig.number, sub=(2, 4, 2 + i), cmap="twilight_shifted",
                min=-np.pi, max=np.pi, title="", cbar=False,
                badcolor="white", bgcolor="white")
    turns = 2 * grad0 * l2 * pix512 / (2 * np.pi)
    plt.title(r"phase of $e^{2i\,\mathrm{RM}\,\lambda^2}$ at %.0f MHz"
              "\n%.0f turns per nside-512 pixel" % (f0, turns),
              fontsize=9.5, color=INK, pad=6)
    maps.append(plt.gca())

# one shared cyclic colourbar for the three phase maps
box0, box1 = maps[0].get_position(), maps[-1].get_position()
cax = fig.add_axes([box0.x0 + .02, box0.y0 - .035, (box1.x1 - box0.x0) - .04, .022])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-np.pi, np.pi), cmap="twilight_shifted"),
        cax=cax, orientation="horizontal")
cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
cb.outline.set_edgecolor("#c8c8c8"); cb.ax.tick_params(colors=MUTE, length=3)

ax = fig.add_subplot(2, 1, 2)
ax.set_facecolor("white")
theta = np.logspace(-7, -1.4, 400)                   # angular separation, rad
for f0, c in BANDS:
    l2 = (C / (f0 * 1e6)) ** 2
    ax.loglog(theta, 2 * grad0 * l2 * theta, color=c, lw=2.1, label="%.0f MHz" % f0)
ax.axhline(np.pi, color=INK, lw=1.2, ls="--")
ax.text(1.3e-7, np.pi * 1.6, r"$\pi$ — Nyquist limit for the pixel sum",
        fontsize=8.6, color=INK)
for ns, ls in ((512, "-"), (2048, ":")):
    pix = np.sqrt(4 * np.pi / hp.nside2npix(ns))
    ax.axvline(pix, color=MUTE, lw=1.0, ls=ls)
    ax.text(pix * 1.12, 2.5e-2, "nside %d pixel" % ns, rotation=90,
            fontsize=8.2, color=MUTE, va="bottom")
ax.set_xlabel("angular separation on the sky   [rad]")
ax.set_ylabel("phase change   [rad]")
ax.set_xlim(theta[0], theta[-1]); ax.set_ylim(1e-2, 3e6)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("How far must you move before the phase turns by $\\pi$?",
             fontsize=10.5, loc="left", pad=16)
ax.text(0, 1.02, "Every curve crosses $\\pi$ two to three decades left of the "
        "pixel line: the grid cannot resolve the rotation it applies.",
        transform=ax.transAxes, fontsize=8.4, color=MUTE, va="bottom")

fig.subplots_adjust(top=.90, bottom=.10, hspace=.60, left=.055, right=.985)
fig.savefig(str(_OUT.parent / "faraday_phase_maps.png"), dpi=100,
            facecolor="white", bbox_inches="tight")
print("saved faraday_phase_maps.png")

for f0, _ in BANDS:
    l2 = (C / (f0 * 1e6)) ** 2
    theta_pi = np.pi / (2 * grad0 * l2)
    print("  %2.0f MHz: phase turns by pi every %.2e rad = %.3f arcsec"
          " -> %.0f turns per nside=512 pixel"
          % (f0, theta_pi, np.degrees(theta_pi) * 3600,
             2 * grad0 * l2 * pix512 / (2 * np.pi)))
