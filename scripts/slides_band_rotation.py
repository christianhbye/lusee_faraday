"""Band-wide rotation of the local polarized term (slide side product).

Reads the slides_subpixel_spectra caches computed at native channels across
10-50 MHz for screens at small fractions f of the column (the local medium)
and plots the central parent channel's pseudo-Stokes Q and U of the Faraday
polarized term against frequency and lambda^2, at one sidereal phase, next
to the unrotated polarization and the leakage. Below a few hundredths of a
rad/m^2 of depth the beam-averaged angle rotates coherently as 2 lambda^2
Phi, so Q and U swing across the band instead of rippling within a channel.
"""

import common  # noqa: F401
import argparse

import numpy as np

from common import REPO

CACHE = REPO / "generated_data/cache/slides_subpixel"
FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"


def load(center, f, profile="screen"):
    tag = "" if f == 1.0 else f"_f{f:g}"
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
    args = parser.parse_args()
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
                    for r in (load(c, f) for f in args.fractions)
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
            ax.plot(
                cs,
                [r["leak"][it, s] for r in ref if r is not None],
                color="k",
                lw=1,
                ls="-.",
                label="I-only leakage",
            )
        for f, (color, ls) in zip(args.fractions, styles):
            rows = [load(c, f) for c in centers]
            ok = [r is not None for r in rows]
            if not any(ok):
                print(f"no caches for f={f:g}; skipped")
                continue
            ax.plot(
                centers[ok],
                [r["faraday"][it, s] for r in rows if r],
                color=color,
                ls=ls,
                lw=1.6,
                marker="o",
                ms=3,
                label=(
                    "emission behind the local gas only: "
                    f"screen at f = {f:g} of the column"
                ),
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
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set_xlabel(
        "frequency [MHz]  (native 25 kHz parent channels every 2 MHz)"
    )
    phase = None
    for c in centers:
        b = load(c, args.fractions[0])
        if b is not None:
            phase = b["lst"][it]
            break
    fig.suptitle(
        (
            "Local Faraday depth: the beam-averaged polarization angle "
            f"rotates coherently across the band (sidereal phase {phase:.2f})"
            if phase is not None
            else "Local Faraday depth across the band"
        ),
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    name = FIG_DIR / "14_band_rotation.png"
    fig.savefig(name, dpi=140)
    print("saved", name)


if __name__ == "__main__":
    main()
