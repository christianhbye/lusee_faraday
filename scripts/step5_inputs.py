"""The two inputs to the depth histogram, as maps (spec S5).

The report's build-up is theory -> intermediate products -> results.
This script supplies the intermediate-product figure: the beam-sky
weight |w|^2 at a few LSTs alongside the Hutschenreuter RM map, which
are exactly the two arrays ``dispersion.depth_distribution`` consumes
-- one supplies the bin INDEX, the other the MASS deposited there.

``step5_template.npz`` cannot serve this: it stores ``w2_mean``, the
LST- AND band-summed weight, because keeping 128 per-LST weight maps at
nside 512 would be 3.2 GB per band.  The point of the figure is that
the weight MOVES with LST, so the mean is the one thing that cannot
show it.  Three maps are recomputed here instead.

DISPLAY NSIDE.  Everything written here is ud_grade'd to a display
resolution and is for plotting only.  Degrading the RM map is forbidden
for COMPUTATION (per-pixel Faraday phases do not commute with
ud_grade, CLAUDE.md) -- nothing downstream of this file computes
anything, and the templates it illustrates were built at native nside
512 from the undegraded map.

Needs the 631 MB response artifact.  ~15 s and ~5.3 GiB peak RSS, so
run it inside the usual cgroup:

  systemd-run --user --scope -q -p MemoryMax=10G -- \
      uv run python scripts/step5_inputs.py > <abs path> 2>&1 &

Usage:
  uv run python step5_inputs.py [--band 30] [--nlst 3] [--nside 128]
"""

import argparse

import common  # noqa: F401
import numpy as np

import healpy as hp
from common import GEN_DIR, RESPONSE_PATH, load_sky_maps
from lusee_faraday import response as rsp
from lusee_faraday.config import (
    BETA_QU,
    FREQ_REF_QU,
    MAP_NSIDE,
    moon_location,
    times,
)

SIDEREAL_MONTH_H = 27.321661 * 24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=30.0)
    ap.add_argument("--nlst", type=int, default=3)
    ap.add_argument("--nside", type=int, default=128)
    args = ap.parse_args()

    maps = load_sky_maps()
    rm = np.asarray(maps["RM"], dtype=float)
    p_emis = (
        np.hypot(maps["Q23"], maps["U23"])
        * (args.band / FREQ_REF_QU) ** BETA_QU
    )

    resp = rsp.load_response(RESPONSE_PATH)
    kernel = rsp.FixedChannelKernel(resp, args.band)
    loc, t_all = moon_location(), times()
    # Spread the samples over the sidereal month rather than taking the
    # first few: consecutive entries of times() are minutes apart and
    # would give three visually identical maps.
    lst_idx = np.linspace(0, len(t_all) - 1, args.nlst + 1, dtype=int)[:-1]

    w2_lst = np.zeros((args.nlst, hp.nside2npix(args.nside)))
    for i, ti in enumerate(lst_idx):
        wb = rsp.pair_weight_maps(kernel, t_all[ti], loc, MAP_NSIDE)
        w2 = ((wb * p_emis[None, :]) ** 2).sum(axis=0)
        w2_lst[i] = hp.ud_grade(w2, args.nside)
        print(f"LST {i + 1}/{args.nlst} (index {ti})", flush=True)

    out = GEN_DIR / "step5_inputs.npz"
    np.savez(
        out,
        band=args.band,
        w2_lst=w2_lst,
        rm_display=hp.ud_grade(rm, args.nside),
        lst_hours=lst_idx * (SIDEREAL_MONTH_H / len(t_all)),
        display_nside=args.nside,
        native_nside=MAP_NSIDE,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
