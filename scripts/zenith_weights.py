"""Compute and cache the zenith-calibrated polarimeter weights.

Places an unpolarized source at exact zenith and builds two
calibrated polarimeters from its covariance C0:

- mode="gains": X = wE E - wW W, Y = wN N - wS S with
  w_p ~ sqrt(1/C0_pp) plus a common X/Y rescale that nulls the zenith
  pseudo-Q exactly (real per-port weights; residual u-leakage from
  the inter-port cross-couplings remains).
- mode="ortho": Loewdin-orthonormalization of the gain-weighted pair
  in the C0 metric (fourport.orthonormalize_xy): X and Y become
  complex combinations of all four ports and the zenith pseudo-Q, U
  and V are all exactly zero.

Caches both sets per band center in
generated_data/cache/zenith_weights_{C}.npz and prints diagnostics.
"""

import argparse

import numpy as np

from common import CACHE_DIR, RESPONSE_PATH
from lusee_faraday import fourport as fp


def get_weights(center_mhz, mode="ortho", force=False):
    """Load or compute (x_vec, y_vec) for one band center."""
    if mode not in ("gains", "ortho"):
        raise ValueError(f"unknown mode {mode!r}")
    cache = CACHE_DIR / f"zenith_weights_{center_mhz:g}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        if f"x_{mode}" in d.files:
            return d[f"x_{mode}"], d[f"y_{mode}"]
    from lusee.ReceiverImpedance import JFETReceiver

    resp = fp.load_response_fast(RESPONSE_PATH)
    kern = fp.FixedFreqKernel(resp, center_mhz, JFETReceiver())
    del resp
    x_g, y_g, C0 = fp.zenith_port_weights(kern)
    x_o, y_o = fp.orthonormalize_xy(C0, x_g, y_g)
    np.savez(
        cache,
        x_gains=x_g,
        y_gains=y_g,
        x_ortho=x_o,
        y_ortho=y_o,
        C0=C0,
    )

    autos = np.diagonal(C0).real
    print(
        f"{center_mhz:g} MHz zenith autos (N,E,S,W): "
        + " ".join(f"{a:.4e}" for a in autos),
        flush=True,
    )
    print("  gains x_vec:", np.round(x_g, 5), flush=True)
    print("  gains y_vec:", np.round(y_g, 5), flush=True)
    print("  ortho x_vec:", np.round(x_o, 5), flush=True)
    print("  ortho y_vec:", np.round(y_o, 5), flush=True)
    for label, xv, yv in (
        ("raw", None, None),
        ("gains", x_g, y_g),
        ("ortho", x_o, y_o),
    ):
        S = fp.polarimeter(C0, xv, yv)
        print(
            f"  zenith unpolarized [{label:5s}]: "
            f"q={S[1]/S[0]:+.2e} u={S[2]/S[0]:+.2e} "
            f"v={S[3]/S[0]:+.2e}",
            flush=True,
        )
    return (x_g, y_g) if mode == "gains" else (x_o, y_o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", type=float, nargs="+", default=[30.0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for c in args.centers:
        get_weights(c, force=args.force)
