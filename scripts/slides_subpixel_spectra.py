"""Slide 10: the linear-interpolation sky through the real instrument.

WMAP-K polarization rotated with the pixel-centre RM and multiplied by the
analytic pixel-average factor of linear RM interpolation, at 201 node
frequencies across the 30 MHz band (1 kHz apart; the speckle's frequency
coherence is ~11 kHz), degraded to nside 64 *after* rotation and averaging
(a linear low-pass; the beam reads ell <= 48), then through the as-built
four-port response (harmonic arm), the loaded covariance frozen at the
native channel, parent/zoom channelization and the zenith-calibrated
pseudo-Stokes. The sky is polarization-only (I = 0), so its pseudo-Stokes
Q, U are exactly the Faraday polarized signal that adds linearly to the
I-only leakage of ``step_ionly.py``. Compared with the ensemble rms of the
column-screen covariance run (``configs/paper/covariance-column.json``).

Stages: ``compute`` (cached under generated_data/cache/slides_subpixel/)
and ``plot`` (PNG under notebooks/slides_pixel_sum/).
"""

import common  # noqa: F401  sets JAX_ENABLE_X64 before any jax import
import argparse
import json
import time as _time

import healpy as hp
import numpy as np

from common import (
    CACHE_DIR,
    REPO,
    RESPONSE_PATH,
    fine_freqs,
    lam2,
    load_sky_maps,
    moon_location,
    parent_centers,
    sky_at_freq,
    times,
)
from lusee_faraday import channelization as chan
from lusee_faraday import engine, instrument
from lusee_faraday import polarimeter as pol
from lusee_faraday import response as rsp

CENTER = 30.0  # overridden by --center
LMAX = 48
NSIDE_DEG = 64
N_NODES = 201
TIME_INDEX = (0, 256, 512, 768)  # quarters of the lunar sidereal day
OUT_ROOT = CACHE_DIR / "slides_subpixel"
FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"
COLUMN_RECORD = REPO / "docs" / "faraday-covariance-column-results.json"
WMAP_PIXEL_RMS_30 = 1.257e4  # K at 30 MHz, all scales incl. WMAP noise
BETA_QU = -2.8
WMAP_BAND_RMS_30 = 5533.0  # K at 30 MHz, ell <= 30, noise subtracted


def set_center(center_mhz):
    global CENTER
    CENTER = float(center_mhz)


def out_dir():
    return OUT_ROOT / f"{CENTER:g}mhz"


def wmap_pixel_rms():
    """WMAP K pixel rms (all scales, noise included) scaled to CENTER."""
    return WMAP_PIXEL_RMS_30 * (CENTER / 30.0) ** BETA_QU


def neighbour_step(rm):
    """rms RM step to the four edge neighbours of every nside-512 pixel."""
    nside = hp.npix2nside(rm.size)
    nbr = hp.get_all_neighbours(nside, np.arange(rm.size))[[1, 3, 5, 7]]
    ok = nbr >= 0
    d = np.where(ok, rm[np.where(ok, nbr, 0)] - rm[None, :], np.nan)
    return np.sqrt(np.nanmean(d**2, axis=0))


def linear_factor(step, freq_mhz):
    """|sinc(g/2)|, g the phase change across an nside-512 pixel."""
    g = 2 * lam2(freq_mhz) * step
    return np.abs(np.sinc(g / (2 * np.pi)))


def node_maps(maps, nodes):
    """(K, 4, npix_64) COSMO Stokes maps of the pixel-averaged rotated sky."""
    rm = maps["RM"]
    step = neighbour_step(rm)
    data = np.zeros((nodes.size, 4, hp.nside2npix(NSIDE_DEG)))
    for k, f in enumerate(nodes):
        _, q, u = sky_at_freq(maps, f)
        p = (q + 1j * u) * np.exp(2j * lam2(f) * rm) * linear_factor(step, f)
        data[k, 1] = hp.ud_grade(p.real, NSIDE_DEG)
        data[k, 2] = hp.ud_grade(p.imag, NSIDE_DEG)
    return data


def interpolation_weights(nodes, fine):
    """(K, F) linear-interpolation weights of the nodes onto the fine grid."""
    w = np.zeros((nodes.size, fine.size))
    j = np.clip(np.searchsorted(nodes, fine) - 1, 0, nodes.size - 2)
    t = (fine - nodes[j]) / (nodes[j + 1] - nodes[j])
    w[j, np.arange(fine.size)] = 1 - t
    w[j + 1, np.arange(fine.size)] = t
    return w


def compute(out):
    import croissant as cro
    from lusee.ReceiverImpedance import JFETReceiver

    t0 = _time.time()
    maps = load_sky_maps()
    ff = fine_freqs(CENTER)
    nodes = np.linspace(ff[0], ff[-1], N_NODES)
    data = node_maps(maps, nodes)
    print(f"node maps {_time.time() - t0:.0f} s", flush=True)
    sky = cro.PolarizedSky(
        data, nodes, sampling="healpix", coord="galactic", convention="COSMO"
    )
    alms = np.asarray(sky.compute_alm(lmax=LMAX))  # (K, 4, L, 2L-1)
    print(f"sky alms {alms.shape} {_time.time() - t0:.0f} s", flush=True)
    resp, receiver = rsp.load_response(RESPONSE_PATH), JFETReceiver()
    beam = rsp.four_port_pair_alms(resp, CENTER, LMAX)
    print(f"beam alms {_time.time() - t0:.0f} s", flush=True)
    tt = times()[list(TIME_INDEX)]
    W = engine.contract(beam, alms, tt, moon_location(), LMAX)
    print(f"contraction {W.shape} {_time.time() - t0:.0f} s", flush=True)
    coeffs = interpolation_weights(nodes, ff)[:, :, None] * np.ones(4)
    x_vec, y_vec, _ = pol.zenith_vectors(resp, receiver, CENTER)

    def through_instrument(W_, coeffs_):
        pair = engine.expand(W_, coeffs_)  # (T, F, 10)
        C = instrument.covariance(
            pair,
            resp,
            receiver,
            ff,
            impedance_freq_mhz=CENTER,
            T_moon=0.0,
            T_ant=0.0,
        )
        fine_ch = instrument.channels(C)[0]  # (T, F, 16)
        binned = chan.integrate(fine_ch, ff, parent_centers(CENTER))
        return (
            pol.pseudo_stokes_from_channels(binned["zoom"], x_vec, y_vec),
            pol.pseudo_stokes_from_channels(binned["parent"], x_vec, y_vec),
        )

    stokes_zoom, stokes_parent = through_instrument(W, coeffs)
    # Calibration: a uniform unpolarized 1 K sky through the same pipeline.
    # Its pseudo-I is the response of one kelvin of beam-weighted brightness,
    # so dividing by it converts the loaded units to K and defines the
    # beam-weighted sky temperature.
    unit = np.zeros((1, 4, hp.nside2npix(NSIDE_DEG)))
    unit[0, 0] = 1.0
    unit_alm = np.asarray(
        cro.PolarizedSky(
            unit,
            np.array([CENTER]),
            sampling="healpix",
            coord="galactic",
            convention="COSMO",
        ).compute_alm(lmax=LMAX)
    )
    W_unit = engine.contract(beam, unit_alm, tt, moon_location(), LMAX)
    unit_zoom, unit_parent = through_instrument(
        W_unit, np.ones((1, ff.size, 4))
    )
    print(f"unit sky {_time.time() - t0:.0f} s", flush=True)
    ionly = np.load(REPO / "generated_data" / f"real{CENTER:g}_ionly.npz")
    key = [
        k
        for k in ionly.files
        if ionly[k].ndim == 2 and ionly[k].shape[-1] == 16
    ][0]
    stokes_ionly = pol.pseudo_stokes_from_channels(
        ionly[key][list(TIME_INDEX)], x_vec, y_vec
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "spectra.npz",
        stokes_zoom=stokes_zoom,  # (T, 3 parents, 64, 4) I Q U V, K
        stokes_parent=stokes_parent,  # (T, 3, 4)
        stokes_ionly_parent=stokes_ionly,  # (T, 4) center parent, I-only sky
        unit_zoom=unit_zoom,  # (T, 3, 64, 4) pseudo-Stokes, 1 K uniform sky
        unit_parent=unit_parent,  # (T, 3, 4)
        zoom_offsets_hz=chan.zoom_bin_offsets_hz(),
        parent_centers_mhz=parent_centers(CENTER),
        time_index=np.array(TIME_INDEX),
        lst_hours=np.array(TIME_INDEX) / 1024 * 24,
        nodes_mhz=nodes,
    )
    print(f"saved {out / 'spectra.npz'} {_time.time() - t0:.0f} s", flush=True)


def ensemble_band(label_contains):
    """Zoom-bin speckle rms (K per 1 K intrinsic) from the column run."""
    rec = json.loads(COLUMN_RECORD.read_text())
    for row in rec["results"]:
        if (
            row["center_mhz"] == CENTER
            and label_contains in row["case"]["label"]
        ):
            z = np.load(REPO / row["path"])
            spk = z["speckle"]
            nb = spk.shape[0]
            rms_q = np.sqrt(
                np.array([spk[b, 1, b, 1].real for b in range(3, nb)])
            )
            rms_u = np.sqrt(
                np.array([spk[b, 2, b, 2].real for b in range(3, nb)])
            )
            return z["frequency_mhz"][3:], rms_q, rms_u
    raise KeyError(label_contains)


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(out / "spectra.npz")
    off = d["zoom_offsets_hz"] * 1e-6
    order = np.argsort(off)
    fz = CENTER + off[order]
    k_zoom = d["unit_zoom"][:, 1, :, 0][:, order]  # loaded units per K
    k_parent = d["unit_parent"][:, 1, 0]
    zoom = d["stokes_zoom"][:, 1][:, order] / k_zoom[:, :, None]  # K
    parent = d["stokes_parent"][:, 1] / k_parent[:, None]  # (T, 4) K
    i_ref = d["stokes_ionly_parent"][:, 0] / k_parent  # beam-weighted I, K
    q_leak = d["stokes_ionly_parent"][:, 1] / k_parent  # I-only leakage Q, K
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    colors = ["C0", "C1", "C2", "C4"]
    for s, (ax, name) in enumerate(zip(axes, ("Q", "U")), start=1):
        for it, lst in enumerate(d["lst_hours"]):
            ax.plot(
                fz,
                zoom[it, :, s],
                color=colors[it],
                lw=1.2,
                label=f"linear-interpolation sky, sidereal phase {lst/24:.2f}",
            )
            ax.plot(
                d["parent_centers_mhz"][1],
                parent[it, s],
                "o",
                mfc="none",
                color=colors[it],
                ms=7,
            )
        for anchor, alpha_fill in (("dphi1=1", 0.25), ("dphi1=3", 0.15)):
            fe, rq, ru = ensemble_band(f"alpha=2 {anchor}")
            band = (rq if s == 1 else ru) * wmap_pixel_rms()
            oe = np.argsort(fe)
            ax.fill_between(
                fe[oe],
                -band[oe],
                band[oe],
                color="0.4",
                alpha=alpha_fill,
                label=(
                    rf"ensemble $\pm$1 rms, $\alpha$=2, "
                    rf"$\Delta\phi_{{1^\circ}}$={anchor[-1]} rad m$^{{-2}}$"
                ),
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"pseudo-Stokes {name} [K]")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    axes[1].set_xlabel(
        "frequency [MHz], 64 zoom bins of the central 25 kHz parent channel"
    )
    fig.suptitle(
        "Faraday polarized signal through the as-built LuSEE instrument "
        f"at {CENTER:g} MHz\n"
        "WMAP K rotated with the Hutschenreuter RM, linear sub-pixel "
        "interpolation, four sidereal phases; circles: parent channel\n"
        f"beam-weighted I = {i_ref.min():.0f} to {i_ref.max():.0f} K; "
        "I-only leakage "
        f"Q = {q_leak.min():.0f} to {q_leak.max():.0f} K, off scale by 10$^4$",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"10_subpixel_spectra_{CENTER:g}MHz.png", dpi=140)
    print("saved", FIG_DIR / f"10_subpixel_spectra_{CENTER:g}MHz.png")
    for it, lst in enumerate(d["lst_hours"]):
        q, u = zoom[it, :, 1], zoom[it, :, 2]
        print(
            f"phase {lst/24:.2f}: zoom rms Q {q.std():.3g} K, "
            f"U {u.std():.3g} K; parent Q {parent[it, 1]:.3g} K; "
            f"I {i_ref[it]:.0f} K; leak Q {q_leak[it]:.1f} K -> "
            f"speckle/I {q.std() / i_ref[it]:.2e}, "
            f"leak/I {q_leak[it] / i_ref[it]:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compute", "plot"])
    parser.add_argument("--center", type=float, default=30.0)
    args = parser.parse_args()
    set_center(args.center)
    if args.stage == "compute":
        compute(out_dir())
    else:
        plot(out_dir())


if __name__ == "__main__":
    main()
