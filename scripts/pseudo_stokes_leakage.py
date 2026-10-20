"""Four-port pseudo-Stokes are not the sky's Stokes: a heads-up figure.

``compute`` pushes a uniform unpolarized 1 K sky (the hemisphere integral of
the I-kernel of the as-built BGL_v16 response at one native channel) through
the loaded covariance at every 2 MHz from 10 to 50 MHz and forms the
pseudo-Stokes in three bases: the raw pseudo-dipoles ``X = E - W``,
``Y = N - S``; the gains-only zenith calibration; and the Loewdin
orthonormalization the pipeline uses (``polarimeter.zenith_vectors``). The
result is cached in ``generated_data/cache/pseudo_stokes_leakage.npz``.

``plot`` draws, against frequency: the uniform sky's pseudo-Q and pseudo-U
over pseudo-I in the three bases (the instrument's own polarization of a
featureless sky); the Haslam I-only sky's leakage in the pipeline basis at
the four sidereal phases of the slides caches (the sky-dependent part); and
the polarized sky's contribution to pseudo-I next to its contribution to
pseudo-Q and pseudo-U, both over the beam-weighted I (the leak in the other
direction). The second and third panels read the ``slides_subpixel`` caches
of ``slides_subpixel_spectra.py``.
"""

import common  # noqa: F401
import argparse

import numpy as np

from common import CACHE_DIR, FIG_DIR, RESPONSE_PATH

CENTERS = np.arange(10, 51, 2, dtype=float)
CACHE = CACHE_DIR / "pseudo_stokes_leakage.npz"
SLIDES_CACHE = CACHE_DIR / "slides_subpixel"
MODES = ("raw", "gains", "ortho")


def uniform_sky_pair(resp, freq_mhz):
    """Pair covariance kernel (10,) of a uniform 1 K unpolarized sky."""
    from lusee_faraday import response as rsp

    k = rsp.FixedChannelKernel(resp, freq_mhz)
    th = np.radians(np.asarray(resp.theta_deg))
    ph = np.radians(np.asarray(resp.phi_deg))
    wphi = np.full(ph.size, np.mean(np.diff(ph)))
    if abs((ph[-1] - ph[0]) - 2 * np.pi) < 1e-6:
        wphi[[0, -1]] *= 0.5  # the grid repeats phi = 0 at 360
    return np.trapezoid((k.K[:, 0] * wphi).sum(-1) * np.sin(th), th, axis=-1)


def compute():
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import instrument
    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp

    resp, receiver = rsp.load_response(RESPONSE_PATH), JFETReceiver()
    out = {m: np.zeros((CENTERS.size, 4)) for m in MODES}
    for i, c in enumerate(CENTERS):
        pair = uniform_sky_pair(resp, c)
        C = instrument.covariance(
            pair[None, None, :],
            resp,
            receiver,
            np.array([c]),
            impedance_freq_mhz=c,
            T_moon=0.0,
            T_ant=0.0,
        )
        ch = instrument.channels(C)[0][0, 0]
        for m in MODES:
            if m == "raw":
                xv, yv = None, None
            else:
                xv, yv, _ = pol.zenith_vectors(resp, receiver, c, mode=m)
            out[m][i] = pol.pseudo_stokes_from_channels(ch, xv, yv)
        r, o = out["raw"][i], out["ortho"][i]
        print(
            f"{c:4.0f} MHz: Q/I, U/I raw {r[1] / r[0]:+.4f} {r[2] / r[0]:+.4f}"
            f"  ortho {o[1] / o[0]:+.4f} {o[2] / o[0]:+.4f}",
            flush=True,
        )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, centers=CENTERS, **out)
    print("saved", CACHE)


def slides_terms(center):
    """Pipeline-basis terms (T, 4) in K from the slides cache, or None."""
    path = SLIDES_CACHE / f"{center:g}mhz_f0_c0.005" / "spectra.npz"
    if not path.exists():
        path = SLIDES_CACHE / f"{center:g}mhz_f0.0002" / "spectra.npz"
    if not path.exists():
        return None
    d = np.load(path)
    kp = d["unit_parent"][:, 1, 0]
    return dict(
        ionly=d["ionly_parent"][:, 1] / kp[:, None],
        unrot=d["unrot_parent"][:, 1] / kp[:, None],
        lst=d["lst_hours"] / 24,
    )


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = np.load(CACHE)
    rows = [slides_terms(c) for c in CENTERS]
    have = np.array([r is not None for r in rows])
    cs = CENTERS[have]
    rows = [r for r in rows if r is not None]
    phases = rows[0]["lst"]
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 11), sharex=True, constrained_layout=True
    )
    legend_kw = dict(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    labels = {
        "raw": "raw pseudo-dipoles X = E $-$ W, Y = N $-$ S",
        "gains": "per-port gains, zenith Q nulled",
        "ortho": "orthonormalized at zenith (the pipeline's basis)",
    }
    colors = {"raw": "C3", "gains": "C1", "ortho": "C0"}
    ax = axes[0]
    for m in MODES:
        s = u[m]
        ax.plot(
            u["centers"],
            100 * s[:, 1] / s[:, 0],
            color=colors[m],
            lw=1.8,
            label=f"pseudo-Q / I, {labels[m]}",
        )
        ax.plot(
            u["centers"],
            100 * s[:, 2] / s[:, 0],
            color=colors[m],
            lw=1.8,
            ls="--",
            label=f"pseudo-U / I, {labels[m]}",
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("uniform sky: pseudo-Q, U [% of pseudo-I]")
    ax.set_title(
        "A featureless sky through the as-built four ports: "
        "the instrument's own polarization",
        fontsize=10,
    )
    ax.legend(**legend_kw)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pc = ("C0", "C2", "C1", "C3")
    for t, (ph, col) in enumerate(zip(phases, pc)):
        q = [100 * r["ionly"][t, 1] / r["ionly"][t, 0] for r in rows]
        v = [100 * r["ionly"][t, 2] / r["ionly"][t, 0] for r in rows]
        ax.plot(
            cs, q, color=col, lw=1.6, label=f"Q / I, sidereal phase {ph:.2f}"
        )
        ax.plot(
            cs, v, color=col, lw=1.6, ls="--", label=f"U / I, phase {ph:.2f}"
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("I-only sky: pseudo-Q, U [% of beam-weighted I]")
    ax.set_title(
        "Haslam synchrotron + CMB, no polarization, pipeline basis: "
        "the leakage depends on where the Galaxy is",
        fontsize=10,
    )
    ax.legend(**legend_kw)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for t, (ph, col) in enumerate(zip(phases, pc)):
        I = np.array([r["ionly"][t, 0] for r in rows])  # noqa: E741
        p = np.array(
            [np.hypot(r["unrot"][t, 1], r["unrot"][t, 2]) for r in rows]
        )
        i_leak = np.array([abs(r["unrot"][t, 0]) for r in rows])
        ax.semilogy(
            cs,
            100 * p / I,
            color=col,
            lw=1.6,
            label=f"|pseudo-Q + i pseudo-U| / I, phase {ph:.2f}",
        )
        ax.semilogy(
            cs,
            100 * i_leak / I,
            color=col,
            lw=1.6,
            ls=":",
            label=f"|pseudo-I| / I, phase {ph:.2f}",
        )
    ax.set_ylabel("polarized sky: pseudo-I and |P| [% of beam-weighted I]")
    ax.set_xlabel(
        "frequency [MHz]  (native 25 kHz parent channels every 2 MHz)"
    )
    ax.set_title(
        "The polarized sky alone (I = 0 in the map): its pseudo-I is of the "
        "same order as its pseudo-Q, U",
        fontsize=10,
    )
    ax.legend(**legend_kw)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        "LuSEE-Night four-port pseudo-Stokes versus sky Stokes: the zenith "
        "calibration makes a polarimeter at zenith only\n"
        "(BGL_v16 as-built response, JFET receiver, central 25 kHz parent "
        "channel, kelvin from a uniform 1 K sky)",
        fontsize=10.5,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = FIG_DIR / "pseudo_stokes_leakage.png"
    fig.savefig(name, dpi=140)
    print("saved", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compute", "plot"))
    args = parser.parse_args()
    compute() if args.stage == "compute" else plot()


if __name__ == "__main__":
    main()
