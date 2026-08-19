"""Quantify what the four-port asymmetric treatment buys over the
paper's 90-degree-rotation assumption.

The paper (sec:lusee) states: "Because of the azimuthal symmetry of the
instrument, we assume that J_x and J_y are related by a 90-degree
rotation.  In reality, the symmetry is broken by the lander itself, but
we do not include this effect here."  Figure 4's caption carries the
TODO "show symmetric beam and realistic beam".  This script produces
exactly that comparison, using three BGL_v16 response artifacts:

  as-built   lusee_bgl_v16_response_v3.fits         asymmetric, coupled
  C4-sym     lusee_bgl_v16_response_v3_c4sym.fits   the paper's assumption,
                                                    made self-consistent
  diag-ZA    lusee_bgl_v16_response_v3_diagza.fits  inter-port coupling removed

The observable is instrumental leakage for a *totally unpolarized*
point source: p_leak = sqrt(Q^2+U^2+V^2)/I through the naive
polarimeter X = E-W, Y = N-S.  For an ideal C4-symmetric array with no
coupling this vanishes at zenith; any nonzero value is leakage
manufactured by the instrument.
"""

import argparse
import os

import numpy as np

from common import FIG_DIR, RESPONSE_DIR

MODELS = [
    ("as-built", "lusee_bgl_v16_response_v3.fits", "#2a78d6"),
    ("C4-sym", "lusee_bgl_v16_response_v3_c4sym.fits", "#eb6834"),
    ("diag-ZA", "lusee_bgl_v16_response_v3_diagza.fits", "#1baf7a"),
]


def leakage(kern, theta, phi, x_vec=None, y_vec=None):
    """p_leak and pseudo-Stokes for an unpolarized source at (theta, phi)."""
    from lusee_faraday import _legacy_pixel as fp

    K = kern.sample(np.asarray(theta), np.asarray(phi))  # (10, 4, N)
    pair = kern.prefac * K[:, 0, :]                      # unpolarized -> I only
    C = fp.assemble_covariance(pair.T, kern.M)           # (N, 4, 4)
    S = fp.polarimeter(C, x_vec, y_vec)                  # (N, 4)
    p = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2) / S[:, 0]
    return p, S


def run(freq_mhz, az_deg, alt_deg):
    from lusee.ReceiverImpedance import JFETReceiver
    from lusee_faraday import _legacy_pixel as fp

    receiver = JFETReceiver()
    alt = np.radians(alt_deg)
    theta_alt = np.pi / 2 - alt
    phi_fixed = np.full_like(theta_alt, np.radians(az_deg))

    az = np.radians(np.linspace(0, 360, 361))
    theta_az = np.full_like(az, np.pi / 2 - np.radians(30.0))

    out = {}
    for name, fname, color in MODELS:
        path = RESPONSE_DIR / fname
        if not path.exists():
            print(f"SKIP {name}: {path} missing", flush=True)
            continue
        resp = fp.load_response_fast(str(path))
        kern = fp.FixedFreqKernel(resp, freq_mhz, receiver)
        del resp

        C0 = fp.assemble_covariance(
            kern.prefac * kern.sample(np.array([0.0]), np.array([0.0]))[:, 0, 0],
            kern.M,
        )
        autos = np.diagonal(C0).real
        p_alt, _ = leakage(kern, theta_alt, phi_fixed)
        p_az, _ = leakage(kern, theta_az, az)
        x_o, y_o = fp.orthonormalize_xy(
            C0, *fp.zenith_port_weights(kern)[:2]
        )
        p_alt_cal, _ = leakage(kern, theta_alt, phi_fixed, x_o, y_o)

        out[name] = dict(
            color=color, autos=autos, p_alt=p_alt, p_az=p_az,
            p_alt_cal=p_alt_cal,
        )
        print(
            f"{name:9s} zenith autos (N,E,S,W) / min = "
            + " ".join(f"{a/autos.min():.4f}" for a in autos)
            + f" | C4 spread {(autos.max()/autos.min()-1)*100:5.2f}%"
            + f" | p_leak(zenith) raw {p_alt[-1]:.3e} cal {p_alt_cal[-1]:.1e}"
            + f" | p_leak(max) {p_alt.max():.3f} @ alt {alt_deg[p_alt.argmax()]:.0f}d",
            flush=True,
        )
        del kern
    return out


def plot(out, alt_deg, freq_mhz, az_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#0b0b0b", "#898781"
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))

    ax = axes[0]
    for name, d in out.items():
        ax.plot(alt_deg, d["p_alt"], lw=2, color=d["color"], label=name)
    ax.set_xlabel("source altitude [deg]")
    ax.set_ylabel(r"leakage $p_{\rm leak}$, unpolarized source")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, None)
    ax.set_title(
        f"(a) naive polarimeter, az={az_deg:g}$^\\circ$", fontsize=9,
        color=INK, loc="left",
    )

    ax = axes[1]
    az = np.linspace(0, 360, 361)
    for name, d in out.items():
        ax.plot(az, d["p_az"], lw=2, color=d["color"], label=name)
    ax.set_xlabel("source azimuth [deg], altitude 30$^\\circ$")
    ax.set_ylabel(r"$p_{\rm leak}$")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title("(b) azimuthal structure (C4 test)", fontsize=9,
                 color=INK, loc="left")

    for ax in axes:
        ax.grid(True, lw=0.4, color=MUTED, alpha=0.35)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"beam_ablation_{freq_mhz:g}.{ext}", dpi=180)
    print(f"wrote {FIG_DIR}/beam_ablation_{freq_mhz:g}.pdf", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=30.0)
    ap.add_argument("--az", type=float, default=0.0)
    args = ap.parse_args()
    alt_deg = np.linspace(0.5, 90.0, 180)
    out = run(args.freq, args.az, alt_deg)
    if out:
        plot(out, alt_deg, args.freq, args.az)
