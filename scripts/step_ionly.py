"""Unpolarized (I-only) reference: the perfect-depolarization limit.

Runs the real-sky simulation with Q = U = 0 — the limiting case in
which Faraday rotation has completely depolarized the sky at the band
center.  With the fixed-beam approximation and no polarization there
is no frequency dependence at all, so one evaluation per time step
suffices and the result is exactly the same in every fine channel,
zoom bin and parent bin.

Only the I part of the pair-Stokes kernel is sampled (4x cheaper than
the full run).  Output: generated_data/real{C}_ionly.npz with the
(1024, 16) products.

With --analyze, compares the full-sky binned products (real{C}_binned
/ real{C}_meta) against the I-only reference and prints the
fractional effect of sky polarization on the parent bin and on zoom
bin 0, plus writes report/figures/real{C}_ionly_frac.
"""

import argparse
import time as _time

import numpy as np

from common import (
    FIG_DIR, GEN_DIR, MAP_NSIDE, N_TIMES, RESPONSE_PATH,
    load_sky_maps, rotation_matrices, sky_at_freq,
)
from lusee_faraday import fourport as fp


def compute(center):
    from lusee.ReceiverImpedance import JFETReceiver

    out_path = GEN_DIR / f"real{center:g}_ionly.npz"
    resp = fp.load_response_fast(RESPONSE_PATH)
    kern = fp.FixedFreqKernel(resp, center, JFETReceiver())
    del resp
    KI = np.ascontiguousarray(kern.K[:, 0])  # (10, Ntheta, Nphi)
    maps = load_sky_maps()
    I_map, _, _ = sky_at_freq(maps, center)
    grid = fp.GalacticGrid(MAP_NSIDE)
    R_all = rotation_matrices()
    scale = kern.prefac * grid.pix_area

    products = np.zeros((N_TIMES, 16))
    t0 = _time.time()
    for it in range(N_TIMES):
        theta, phi, _, up = fp.transport(R_all[it], grid)
        KIs = fp.sample_periodic_maps(
            KI, kern.theta_deg, kern.phi_deg, theta[up], phi[up]
        )  # (10, Nup)
        pair = scale * (KIs @ I_map[up])
        products[it] = fp.pack_products(
            fp.assemble_covariance(pair, kern.M)
        )
        if (it + 1) % 128 == 0:
            dt = _time.time() - t0
            print(
                f"  {center:g} MHz  t {it + 1}/{N_TIMES}  {dt:.0f} s",
                flush=True,
            )
    np.savez(out_path, products=products, center_mhz=center)
    print(f"saved {out_path.name}", flush=True)


def analyze(center):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import SIDEREAL_DAY_S

    from zenith_weights import get_weights

    xv, yv = get_weights(center)  # zenith-calibrated polarimeter
    ionly = np.load(GEN_DIR / f"real{center:g}_ionly.npz")["products"]
    binned = np.load(GEN_DIR / f"real{center:g}_binned.npz")
    S0 = fp.polarimeter_from_channels(ionly, xv, yv)  # (T, 4)
    Sp = fp.polarimeter_from_channels(binned["parent"][:, 1], xv, yv)
    Sz = fp.polarimeter_from_channels(binned["zoom"][:, 1, 0], xv, yv)
    I0 = S0[:, 0]

    def stats(S, name):
        dI = (S[:, 0] - S0[:, 0]) / I0
        pol = np.hypot(S[:, 1] - S0[:, 1], S[:, 2] - S0[:, 2]) / I0
        print(
            f"  {name:12s}  dI/I: median {np.median(np.abs(dI)):.2e}"
            f"  max {np.abs(dI).max():.2e}   |dP|/I: median"
            f" {np.median(pol):.2e}  max {pol.max():.2e}",
            flush=True,
        )
        return dI, pol

    print(f"fractional effect of sky polarization at {center:g} MHz "
          "(vs I-only reference):", flush=True)
    dIp, polp = stats(Sp, "parent bin")
    dIz, polz = stats(Sz, "zoom bin 0")

    t_hr = np.arange(N_TIMES) * SIDEREAL_DAY_S / N_TIMES / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    axes[0].plot(t_hr, dIp, color="C3", lw=1.0, label="parent bin")
    axes[0].plot(t_hr, dIz, color="C1", lw=1.0, ls="--",
                 label="zoom bin 0")
    axes[0].axhline(0, color="0.85", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$(I_{\rm obs} - I_{\rm obs}^{I\,only})"
                       r"/I_{\rm obs}^{I\,only}$")
    axes[0].legend(fontsize=8)
    axes[1].plot(t_hr, polp, color="C3", lw=1.0, label="parent bin")
    axes[1].plot(t_hr, polz, color="C1", lw=1.0, ls="--",
                 label="zoom bin 0")
    axes[1].set_ylabel(r"$|\Delta P_{\rm obs}|/I_{\rm obs}^{I\,only}$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time  [hours]")
    axes[0].set_title(
        f"Polarized sky vs perfect depolarization at {center:g} MHz"
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"real{center:g}_ionly_frac.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote real{center:g}_ionly_frac", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", type=float, nargs="+", default=[30.0])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for center in args.centers:
        out_path = GEN_DIR / f"real{center:g}_ionly.npz"
        if not out_path.exists() or args.force:
            compute(center)
        if args.analyze:
            analyze(center)


if __name__ == "__main__":
    main()
