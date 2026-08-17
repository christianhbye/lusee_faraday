"""Unpolarized point source: pure leakage along the transit track.

The same transiting source as step1_point_source.py but with Stokes
(1, 0, 0, 0).  The result is frequency-flat (no polarization, frozen
beam), so parent bins, zoom bins and fine channels are identical and
a single evaluation per time step suffices.  The recovered
pseudo-polarization is pure instrumental leakage: it must vanish
where the beam is symmetric (near zenith) and grow towards the
horizon.

Outputs generated_data/step1_ionly_source.npz and the figure
report/figures/step1_ionly_polfrac (companion to step1_polfrac_track).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    CACHE_DIR,
    FIG_DIR,
    GEN_DIR,
    RESPONSE_PATH,
)
from lusee_faraday import fourport as fp  # noqa: E402

CENTER_MHZ = 30.0


def compute():
    from lusee.ReceiverImpedance import JFETReceiver

    d = np.load(CACHE_DIR / "step1_track.npz")
    theta, phi = d["theta"], d["phi"]
    resp = fp.load_response_fast(RESPONSE_PATH)
    kern = fp.FixedFreqKernel(resp, CENTER_MHZ, JFETReceiver())
    del resp

    up = theta <= np.pi / 2
    K = kern.sample(theta[up], phi[up])  # (10, 4, Nup)
    pair = kern.prefac * K[:, 0]  # unpolarized: only the I kernel
    C = fp.assemble_covariance(pair.T, kern.M)  # (Nup, 4, 4)
    products = np.zeros((theta.size, 16))
    products[up] = fp.pack_products(C)
    np.savez(
        GEN_DIR / "step1_ionly_source.npz",
        products=products,
        theta=theta,
        phi=phi,
        center_mhz=CENTER_MHZ,
    )
    return theta, products


def plot(theta, products, x_vec=None, y_vec=None, name="step1_ionly_polfrac"):
    up = theta <= np.pi / 2
    rises = np.where(up & ~np.roll(up, 1))[0]
    order = (np.arange(theta.size) + rises[0]) % theta.size
    track = order[up[order]]
    alt = 90.0 - np.degrees(theta[track])
    imax = int(alt.argmax())

    S = fp.polarimeter_from_channels(products[track], x_vec, y_vec)
    p = np.hypot(S[:, 1], S[:, 2]) / S[:, 0]
    p_tot = np.sqrt((S[:, 1:] ** 2).sum(-1)) / S[:, 0]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(
        alt[: imax + 1],
        p[: imax + 1],
        color="C0",
        lw=1.2,
        label=r"$\sqrt{Q^2+U^2}/I$",
    )
    ax.plot(alt[imax:], p[imax:], color="C0", lw=1.2, ls="--", alpha=0.7)
    ax.plot(
        alt[: imax + 1],
        p_tot[: imax + 1],
        color="C4",
        lw=1.0,
        label=r"$\sqrt{Q^2+U^2+V^2}/I$",
    )
    ax.plot(alt[imax:], p_tot[imax:], color="C4", lw=1.0, ls="--", alpha=0.7)
    ax.plot(
        [],
        [],
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="setting branch (dashed)",
    )
    ax.set_xlabel("source altitude  [deg]")
    ax.set_ylabel("recovered pseudo-polarization")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Unpolarized source: instrumental leakage vs altitude")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}", flush=True)
    print(
        f"p at transit: {p[imax]:.4f}; max along track: {p.max():.3f}"
        f" at alt {alt[p.argmax()]:.1f} deg",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    cached = GEN_DIR / "step1_ionly_source.npz"
    if cached.exists():
        d = np.load(cached)
        theta, products = d["theta"], d["products"]
    else:
        theta, products = compute()
    if args.calibrated:
        from zenith_weights import get_weights

        xv, yv = get_weights(CENTER_MHZ)
        plot(theta, products, xv, yv, name="step1w_ionly_polfrac")
    else:
        plot(theta, products)
