"""Regenerate the S4.6 envelope table from luseepy's response and the map.

The numbers quoted in the paper's instrument-methods section come from
here, not from the spec (spec S5).  Prints the parent/zoom 50% depth
horizons per band against the |RM| percentiles, and saves them to
generated_data/step5_envelope.npz for step5_plots.py.

Weighted percentiles: if generated_data/step5_template.npz exists (the
Task 13 output), its LST-mean pair weights re-weight the percentiles;
otherwise the unweighted map percentiles are printed with a note.
"""

import common  # noqa: F401  (sets JAX_ENABLE_X64, sys.path)
import numpy as np

from common import GEN_DIR, load_sky_maps
from lusee_faraday import dispersion as dsp
from lusee_faraday.channelization import parent_weights, zoom_weights

BANDS = (50.0, 30.0, 10.0)


def main():
    off = np.arange(-50000.0, 50001.0, 12.20703125)
    wp = parent_weights(off)
    wz = zoom_weights(off)[:, 0]
    horizons = {
        b: (dsp.depth_horizon(off, wp, b), dsp.depth_horizon(off, wz, b))
        for b in BANDS
    }
    rm = np.abs(load_sky_maps()["RM"])
    qs = [50.0, 90.0, 99.0, 99.9]
    tmpl = GEN_DIR / "step5_template.npz"
    if tmpl.exists():
        w2 = np.load(tmpl)["w2_mean"]  # (npix,) LST/pair-mean |w|^2
        pct = dsp.weighted_percentiles(rm, w2, qs)
        label = "beam-weighted"
        mx = rm[w2 > 0].max()
    else:
        pct = np.percentile(rm, qs)
        label = "UNWEIGHTED (run step5_template.py for the paper numbers)"
        mx = rm.max()

    print(f"|RM| percentiles ({label}):")
    for q, v in zip(qs, pct):
        print(f"  p{q:<5} {v:8.1f} rad/m^2")
    print(f"  max    {mx:8.1f}")
    print("\nband   parent horizon   zoom horizon   (50% depth, rad/m^2)")
    for b in BANDS:
        hp_, hz_ = horizons[b]
        print(f"{b:5.0f}   {hp_:14.1f}   {hz_:12.1f}")
    np.savez(
        GEN_DIR / "step5_envelope.npz",
        bands=np.array(BANDS),
        parent_horizon=np.array([horizons[b][0] for b in BANDS]),
        zoom_horizon=np.array([horizons[b][1] for b in BANDS]),
        percentiles=pct,
        percentile_qs=np.array(qs),
        rm_max=mx,
        weighted=tmpl.exists(),
    )
    print(f"\nwrote {GEN_DIR / 'step5_envelope.npz'}")


if __name__ == "__main__":
    main()
