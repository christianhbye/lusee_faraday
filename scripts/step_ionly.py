"""Unpolarized (I-only) reference: the perfect-depolarization limit.

Runs the real-sky simulation with Q = U = 0 — the limiting case in
which Faraday rotation has completely depolarized the sky at the band
center.  With no polarized power anywhere the Faraday phase cannot
reach the products at all, so one evaluation per time step suffices
and the result is the same in every fine channel, zoom bin and parent
bin.  "The result" is the stored artifact, one number per time step:
both arms freeze the sky at the band centre, exactly as the full run
does (scripts/step2_real_sky.py evaluates sky_at_freq(maps, center)
once).  FaradaySky.coeffs on a *fine* grid is not flat -- the
synchrotron index puts a 1.7e-2 ramp across a parent band, which
tests/test_ionly_regression.py pins -- but nothing on this path ever
evaluates it there.

Two engines compute that same quantity two different ways:

  --engine harmonic (default)
      one contraction of the sky's component alms against the
      response's pair-Stokes alms
      -> generated_data/real{C}_ionly.npz
  --engine legacy
      the pixel quadrature: one sum over the native nside=512 HEALPix
      grid per time step
      -> generated_data/real{C}_ionly_legacy.npz

Both arms use the same response sampling (response.FixedChannelKernel
/ response.four_port_pair_alms) and the same covariance assembly
(instrument.covariance), so the only thing that differs between them
is the quadrature.

With --analyze, compares the full-sky binned products (real{C}_binned
/ real{C}_meta) against the I-only reference and prints the
fractional effect of sky polarization on the parent bin and on zoom
bin 0, plus writes report/figures/real{C}_ionly_frac.
"""

import argparse
import time as _time

import numpy as np

from common import (
    BETA_I,
    FIG_DIR,
    FREQ_REF_I,
    GEN_DIR,
    MAP_NSIDE,
    N_TIMES,
    RESPONSE_PATH,
    T_CMB,
    load_sky_maps,
    moon_location,
    rotation_matrices,
    sky_at_freq,
    times,
)
from lusee_faraday import engine, instrument
from lusee_faraday import pixel_arm as fp
from lusee_faraday import polarimeter as pol
from lusee_faraday import response as rsp
from lusee_faraday.sky import FaradaySky

# The beam is band-limited well below this; AGENTS.md pins lmax ~ 30
# for every harmonic path.  --lmax overrides it, and anything other
# than this value writes to its own artifact (see out_path).
LMAX = 30


def ionly_sky(i408_map, lmax):
    """Two spectrally separable I-only components: synchrotron + CMB.

    ``common.load_sky_maps`` returns ``I408`` with ``T_CMB`` already
    subtracted, and ``common.sky_at_freq`` adds it back *unscaled*,
    because the CMB is not synchrotron and does not follow
    ``beta = -2.55``.  A single ``FaradaySky.i_only`` component carries
    exactly one spectral index, so it cannot hold both: at 30 MHz the
    CMB is 1.1e-4 of the mean sky, against the 2e-4 agreement this
    reference is published as reproducing.

    The input is the *raw* map, not ``sky_at_freq(maps, center)``:
    ``FaradaySky.coeffs`` applies the spectral scaling itself, so
    pre-scaling as well would scale the sky twice.
    """
    sync = FaradaySky.i_only(
        i408_map, lmax, beta_i=BETA_I, ref_freq_i=FREQ_REF_I
    )
    cmb = FaradaySky.i_only(np.full_like(i408_map, T_CMB), lmax)
    return FaradaySky(
        np.concatenate([sync.component_alms, cmb.component_alms]),
        phi_fd=[0.0, 0.0],
        beta=np.concatenate([sync.beta, cmb.beta]),
        ref_freq_mhz=np.concatenate([sync.ref_freq_mhz, cmb.ref_freq_mhz]),
    )


def pack_from_pairs(pair, resp, receiver, freqs, center):
    """Pair integrals -> the 16 real channels, with this run's freeze.

    Both arms funnel through here so the freeze is stated once: the
    beam and all four impedance matrices sit at the native channel,
    and there is no Moon or antenna-metal term.
    ``pixel_arm.assemble_covariance`` — the assembler this replaces, and
    the one that produced the stored ``real{C}_binned.npz`` that
    ``--analyze`` compares against — has no Moon term at all, while
    luseepy's ``T_moon`` default of 250 K moves the answer by 7.4e3
    relative.
    """
    C = instrument.covariance(
        pair,
        resp,
        receiver,
        freqs,
        impedance_freq_mhz=center,
        T_moon=0.0,
        T_ant=0.0,
    )
    return instrument.channels(C)[0]


def out_path(center, arm="harmonic", lmax=LMAX):
    """Where one (band, engine, lmax) combination is stored.

    The harmonic arm at the default lmax owns the canonical name; the
    legacy arm and any other lmax are suffixed, so an A/B never
    silently overwrites the primary artifact.
    """
    if arm == "legacy":
        return GEN_DIR / f"real{center:g}_ionly_legacy.npz"
    if int(lmax) != LMAX:
        return GEN_DIR / f"real{center:g}_ionly_lmax{int(lmax)}.npz"
    return GEN_DIR / f"real{center:g}_ionly.npz"


def legacy_reference(center):
    """The pixel-arm I-only artifact ``--analyze`` compares against.

    The LEGACY file on purpose: real{C}_binned.npz comes from
    scripts/step2_real_sky.py, which stays on the pixel arm, the
    effect measured by --analyze is 2e-4, and the harmonic-vs-pixel
    engine difference is ~1e-2 (scripts/crosscheck_pixel_arm.py) -- a
    hundred times larger than the signal.  Both sides of that
    comparison have to come from the same quadrature.

    Separate from ``analyze`` so ``main`` can check it *before*
    spending eight minutes on a harmonic run the user did not ask for.
    """
    path = out_path(center, "legacy")
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing.  --analyze compares against "
            f"real{center:g}_binned.npz, a pixel-arm artifact, so the "
            "I-only side has to be the pixel arm too.  Run "
            f"`step_ionly.py --engine legacy --centers {center:g}` first "
            "-- and note that real{center:g}_binned.npz itself comes from "
            "scripts/step2_real_sky.py, which is NOT on main: it is one of "
            "the refuted diffuse analyses, reproducible at the "
            "audit-2026-08-18 tag."
        )
    return path


def _instrument():
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    return rsp.load_response(RESPONSE_PATH), JFETReceiver()


def compute_harmonic(center, lmax=LMAX):
    """The whole N_TIMES waterfall as a single harmonic contraction."""
    resp, receiver = _instrument()
    t0 = _time.time()
    beam = rsp.four_port_pair_alms(resp, center, lmax)
    print(f"  beam alms lmax={lmax}  {_time.time() - t0:.0f} s", flush=True)

    t0 = _time.time()
    maps = load_sky_maps()
    sky = ionly_sky(maps["I408"], lmax)
    del maps
    print(
        f"  {sky.n_components} sky components  {_time.time() - t0:.0f} s",
        flush=True,
    )

    t0 = _time.time()
    W = engine.contract(
        beam, sky.component_alms, times(), moon_location(), lmax
    )
    print(f"  contraction {_time.time() - t0:.0f} s", flush=True)

    freqs = np.array([float(center)])
    pair = engine.expand(W, sky.coeffs(freqs))  # (T, 1, 10)
    return pack_from_pairs(pair, resp, receiver, freqs, center)[:, 0]


def compute_legacy(center):
    """The pixel quadrature: one HEALPix sum per time step."""
    resp, receiver = _instrument()
    kern = rsp.FixedChannelKernel(resp, center)
    # kern.sample() would interpolate all four Stokes components; only
    # the I one is ever contracted against an unpolarized sky.  The
    # kernel already carries eta0 / lambda^2, so the pixel solid angle
    # is the whole remaining scale.
    KI = np.ascontiguousarray(kern.K[:, 0])  # (10, Ntheta, Nphi)
    maps = load_sky_maps()
    I_map, _, _ = sky_at_freq(maps, center)
    del maps
    grid = fp.GalacticGrid(MAP_NSIDE)
    R_all = rotation_matrices()

    # The covariance is assembled once at the end rather than inside
    # the loop.  It is independent per (time, frequency), so batching
    # cannot move a value; what the loop measures is the quadrature.
    pair = np.zeros((N_TIMES, 1, 10), dtype=complex)
    t0 = _time.time()
    for it in range(N_TIMES):
        theta, phi, _, up = fp.transport(R_all[it], grid)
        KIs = fp.sample_periodic_maps(
            KI, kern.theta_deg, kern.phi_deg, theta[up], phi[up]
        )  # (10, Nup)
        pair[it, 0] = grid.pix_area * (KIs @ I_map[up])
        if (it + 1) % 128 == 0:
            dt = _time.time() - t0
            print(
                f"  {center:g} MHz  t {it + 1}/{N_TIMES}  {dt:.0f} s",
                flush=True,
            )
    freqs = np.array([float(center)])
    return pack_from_pairs(pair, resp, receiver, freqs, center)[:, 0]


def compute(center, arm="harmonic", lmax=LMAX):
    path = out_path(center, arm, lmax)
    if arm == "legacy":
        products = compute_legacy(center)
        extra = {}
    else:
        products = compute_harmonic(center, lmax)
        extra = {"lmax": int(lmax)}
    np.savez(path, products=products, center_mhz=center, engine=arm, **extra)
    print(f"saved {path.name}", flush=True)


def analyze(center):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import SIDEREAL_DAY_S

    from zenith_weights import get_weights

    ionly_path = legacy_reference(center)
    print(
        f"  comparing against {ionly_path.name} (pixel arm), since "
        f"real{center:g}_binned.npz is one too",
        flush=True,
    )
    xv, yv = get_weights(center)  # zenith-calibrated polarimeter
    ionly = np.load(ionly_path)["products"]
    binned = np.load(GEN_DIR / f"real{center:g}_binned.npz")
    S0 = pol.pseudo_stokes_from_channels(ionly, xv, yv)  # (T, 4)
    Sp = pol.pseudo_stokes_from_channels(binned["parent"][:, 1], xv, yv)
    Sz = pol.pseudo_stokes_from_channels(binned["zoom"][:, 1, 0], xv, yv)
    I0 = S0[:, 0]

    def stats(S, name):
        dI = (S[:, 0] - S0[:, 0]) / I0
        dP = np.hypot(S[:, 1] - S0[:, 1], S[:, 2] - S0[:, 2]) / I0
        print(
            f"  {name:12s}  dI/I: median {np.median(np.abs(dI)):.2e}"
            f"  max {np.abs(dI).max():.2e}   |dP|/I: median"
            f" {np.median(dP):.2e}  max {dP.max():.2e}",
            flush=True,
        )
        return dI, dP

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
    ap.add_argument(
        "--engine", choices=("harmonic", "legacy"), default="harmonic"
    )
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.engine == "legacy" and args.lmax is not None:
        ap.error(
            "--lmax applies to the harmonic arm only; the pixel "
            "quadrature has no band limit.  Passing it here would write "
            "the plain real{C}_ionly_legacy.npz, which a later run "
            "would then reuse as an ordinary legacy artifact."
        )
    lmax = LMAX if args.lmax is None else args.lmax
    if args.analyze and args.engine != "legacy":
        # Up front, not after an eight-minute harmonic run: with
        # --engine legacy the loop below produces the reference itself.
        for center in args.centers:
            legacy_reference(center)
    for center in args.centers:
        path = out_path(center, args.engine, lmax)
        if not path.exists() or args.force:
            compute(center, args.engine, lmax)
        if args.analyze:
            analyze(center)


if __name__ == "__main__":
    main()
