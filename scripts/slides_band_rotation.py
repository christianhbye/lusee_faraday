"""Band-wide view of the polarized term for emission in front of the column.

Reads the slides_subpixel_spectra caches computed at native channels across
10-50 MHz and draws, for one pseudo-Stokes parameter (Q or U) and one
sidereal phase, the layout of the fig 10 top panel across the band: the
total parent-channel pseudo-Stokes of the I-only sky (leakage alone), of the
I-only sky plus the unrotated WMAP polarization, and of the I-only sky plus
the Faraday-rotated polarization. Below it, sharing the frequency axis, the
Faraday polarized term alone against the unrotated polarization, which is
the residual the rotation leaves. (Behind the full column, f = 1, the
polarized term is the 0.1 K ripple of the zoom-bin figures and the total is
indistinguishable from the unrotated one; that case is not drawn.)

Two local cases are supported, as the two brackets on the local medium's
angular structure: the RM map scaled by a fraction f (``--fractions``), the
far end of the screen-fraction dial, which keeps the column's angular
pattern across the beam; and a sky-constant depth (``--const``), which has
none. The real local medium sits between them.

All curves are the four-port pseudo-Stokes through the as-built instrument,
divided by the pseudo-I of a uniform 1 K unpolarized sky, so the axes are
kelvin of beam-weighted brightness.
"""

import common  # noqa: F401
import argparse

import numpy as np

from common import REPO

CACHE = REPO / "generated_data/cache/slides_subpixel"
FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"
CENTERS = np.arange(10, 51, 2, dtype=float)
STOKES = {"Q": 1, "U": 2}


def load(center, f, profile="screen", const=0.0):
    """Central-parent pseudo-Stokes (T, 4) of each term, in K, or None."""
    tag = "" if f == 1.0 else f"_f{f:g}"
    tag += "" if const == 0.0 else f"_c{const:g}"
    tag += "" if profile == "screen" else f"_{profile}"
    path = CACHE / f"{center:g}mhz{tag}" / "spectra.npz"
    if not path.exists():
        return None
    d = np.load(path)
    kp = d["unit_parent"][:, 1, 0]
    return dict(
        faraday=d["stokes_parent"][:, 1] / kp[:, None],
        unrot=d["unrot_parent"][:, 1] / kp[:, None],
        leak=d["ionly_parent"][:, 1] / kp[:, None],
        lst=d["lst_hours"] / 24,
    )


def case_tag(f, const):
    return f"uniform{const:g}" if const else f"f{f:g}"


def case_label(f, const):
    if const:
        return f"a uniform local depth of {const:g} rad m$^{{-2}}$"
    return f"a screen at f = {f:g} of the column (RM map $\\times$ f)"


def band_figure(f, const, stokes, it, plt, ylim=None):
    """One case, one Stokes parameter; ylim = (top, bottom) half-ranges."""
    s = STOKES[stokes]
    rows = [load(c, f, const=const) for c in CENTERS]
    ok = np.array([r is not None for r in rows])
    if not ok.any():
        print(f"no caches for f={f:g} const={const:g}; skipped")
        return None
    cs = CENTERS[ok]
    rows = [r for r in rows if r is not None]
    leak = np.array([r["leak"][it, s] for r in rows])
    unrot = np.array([r["unrot"][it, s] for r in rows])
    far = np.array([r["faraday"][it, s] for r in rows])
    phase = rows[0]["lst"][it]

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True
    )
    # top: the total pseudo-Stokes, as fig 10's top panel across the band
    top.plot(cs, leak, color="k", lw=1.5, label="I-only sky: leakage alone")
    top.plot(
        cs,
        leak + unrot,
        color="0.5",
        lw=2.5,
        label="leakage + WMAP polarization, no Faraday rotation",
    )
    top.plot(
        cs,
        leak + far,
        color="C4",
        lw=1.6,
        ls="--",
        marker="o",
        ms=3,
        label="leakage + WMAP polarization rotated by " + case_label(f, const),
    )
    top.set_ylabel(f"total pseudo-Stokes {stokes} [K]")
    top.grid(alpha=0.3)
    top.legend(fontsize=8, loc="best")
    # bottom: the polarized term alone, the residual the rotation leaves
    bot.plot(
        cs,
        unrot,
        color="0.5",
        lw=2.5,
        label="WMAP polarization, no Faraday rotation",
    )
    bot.plot(
        cs,
        far,
        color="C4",
        lw=1.6,
        ls="--",
        marker="o",
        ms=3,
        label="rotated by " + case_label(f, const),
    )
    bot.set_ylabel(f"Faraday polarized term {stokes} alone [K]")
    bot.set_xlabel(
        "frequency [MHz]  (native 25 kHz parent channels every 2 MHz)"
    )
    bot.grid(alpha=0.3)
    bot.legend(fontsize=8, loc="best")
    for ax, m in zip((top, bot), ylim or (None, None)):
        if m is None:
            m = max(
                np.max(np.abs(line.get_ydata())) for line in ax.get_lines()
            )
        ax.set_ylim(-1.08 * m, 1.08 * m)
    fig.suptitle(
        f"Pseudo-Stokes {stokes} of the central parent channel across the "
        f"band at sidereal phase {phase:.2f}, as-built four-port instrument\n"
        f"emission in front of the Galactic column: {case_label(f, const)}",
        fontsize=10.5,
    )
    name = FIG_DIR / f"14_band_rotation_{case_tag(f, const)}_{stokes}.png"
    fig.savefig(name, dpi=140)
    plt.close(fig)
    print("saved", name)
    return name


def half_ranges(cases, stokes, it):
    """Largest |total| and |polarized term| over the cases, or None."""
    s = STOKES[stokes]
    top, bot = 0.0, 0.0
    for f, k in cases:
        for r in (load(c, f, const=k) for c in CENTERS):
            if r is None:
                continue
            leak, unrot, far = (
                r[n][it, s] for n in ("leak", "unrot", "faraday")
            )
            top = max(top, abs(leak), abs(leak + unrot), abs(leak + far))
            bot = max(bot, abs(unrot), abs(far))
    return (top, bot) if top else None


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=int, default=1)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=(0.0002,),
        help="screen fractions f of the column (RM map x f) to draw",
    )
    parser.add_argument(
        "--const",
        type=float,
        nargs="*",
        default=(),
        help="uniform local Faraday depths [rad/m^2] computed with "
        "--rm-scale 0 --rm-const, drawn as separate figures",
    )
    parser.add_argument(
        "--stokes", nargs="+", choices=tuple(STOKES), default=("Q", "U")
    )
    args = parser.parse_args()
    cases = [(f, 0.0) for f in args.fractions]
    cases += [(0.0, k) for k in args.const]
    FIG_DIR.mkdir(exist_ok=True)
    for stokes in args.stokes:
        # shared limits across the cases, so the figures compare directly
        ylim = half_ranges(cases, stokes, args.phase_index)
        for f, k in cases:
            band_figure(f, k, stokes, args.phase_index, plt, ylim)


if __name__ == "__main__":
    main()
