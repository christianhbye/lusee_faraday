"""Band-wide rotation of the local polarized term (slide side product).

Reads the slides_subpixel_spectra caches computed at native channels across
10-50 MHz for screens at small fractions f of the column (the local medium)
and plots the central parent channel's pseudo-Stokes Q and U of the Faraday
polarized term against frequency and lambda^2, at one sidereal phase, next
to the unrotated polarization. With a local depth of a few thousandths of
a rad/m^2 the polarized term is no longer destroyed; it is reshaped across
the band by of order itself, oscillating in lambda^2 where 2 lambda^2 Phi
exceeds a radian (below ~20 MHz for 0.005 rad/m^2). The four-port pseudo-
Stokes do not rotate as a pair the way the sky's Q and U do, so this is not
the textbook sinusoid; it is a broadband, sidereal-dependent change rather
than a kilohertz ripple.
"""

import common  # noqa: F401
import argparse

import numpy as np

from common import REPO

CACHE = REPO / "generated_data/cache/slides_subpixel"
FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"


def load(center, f, profile="screen", const=0.0):
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


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=int, default=1)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=(0.0002,)
    )
    parser.add_argument(
        "--const",
        type=float,
        default=None,
        help="uniform local Faraday depth [rad/m^2] computed with "
        "--rm-scale 0 --rm-const; replaces the scaled-map fractions",
    )
    args = parser.parse_args()
    cases = (
        [(0.0, args.const)]
        if args.const is not None
        else [(f, 0.0) for f in args.fractions]
    )
    it = args.phase_index
    centers = np.arange(10, 51, 2, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    styles = (("C4", "-"), ("C1", "--"), ("C0", ":"))
    for s, (ax, name) in enumerate(zip(axes, ("Q", "U")), start=1):
        # the unrotated polarization and the leakage do not depend on f,
        # so take them from whichever local run exists at each centre
        ref = [
            next(
                (
                    r
                    for r in (load(c, f, const=k) for f, k in cases)
                    if r is not None
                ),
                None,
            )
            for c in centers
        ]
        have = [r is not None for r in ref]
        if any(have):
            cs = centers[have]
            ax.plot(
                cs,
                [r["unrot"][it, s] for r in ref if r is not None],
                color="0.4",
                lw=2,
                label="WMAP polarization, no rotation",
            )
        for (f, k), (color, ls) in zip(cases, styles):
            rows = [load(c, f, const=k) for c in centers]
            ok = [r is not None for r in rows]
            if not any(ok):
                print(f"no caches for f={f:g} const={k:g}; skipped")
                continue
            case_label = (
                f"rotated by a uniform local depth of {k:g} rad m$^{{-2}}$"
                if k
                else f"emission behind the local gas only: screen at f = {f:g}"
            )
            ax.plot(
                centers[ok],
                [r["faraday"][it, s] for r in rows if r],
                color=color,
                ls=ls,
                lw=1.6,
                marker="o",
                ms=3,
                label=case_label,
            )
        # the column case, where computed: the mean polarization is gone
        col = [(c, load(c, 1.0)) for c in centers]
        col = [(c, r) for c, r in col if r is not None]
        if col:
            ax.plot(
                [c for c, _ in col],
                [r["faraday"][it, s] for _, r in col],
                "ks",
                ms=7,
                label="emission behind the full column (f = 1): "
                "only the 0.1 K ripple is left",
            )
        ax.set_ylabel(f"parent-channel pseudo-Stokes {name} [K]")
        ax.grid(alpha=0.3)
        m = max(np.max(np.abs(line.get_ydata())) for line in ax.get_lines())
        ax.set_ylim(-1.08 * m, 1.08 * m)
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set_xlabel(
        "frequency [MHz]  (native 25 kHz parent channels every 2 MHz)"
    )
    phase = None
    for c in centers:
        b = load(c, cases[0][0], const=cases[0][1])
        if b is not None:
            phase = b["lst"][it]
            break
    fig.suptitle(
        (
            "Emission in front of the Galactic column: a uniform local "
            "Faraday depth reshapes the polarized term across the band\n"
            f"(sidereal phase {phase:.2f}; the I-only leakage, 100 to "
            "20000 K here, is not drawn)"
            if phase is not None
            else "Local Faraday depth across the band"
        ),
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    name = FIG_DIR / (
        "14_band_rotation.png"
        if args.const is None
        else f"14_band_rotation_uniform{args.const:g}.png"
    )
    fig.savefig(name, dpi=140)
    print("saved", name)


if __name__ == "__main__":
    main()
