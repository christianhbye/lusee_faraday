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
    BETA_I,
    T_CMB,
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
CONTEXT_PHASE = 1  # index into TIME_INDEX for the full-spectrum panel
SPAN = 1  # parents plotted: 1 (central) or 3 (all); set by --span
LAYOUT = "phases"  # or "ladder": one phase, three sub-pixel models
BANDS = True  # draw the +-1 rms ensemble bands
SIGMA_COLUMN = 10.0  # rad/m^2, the column screens' dispersion
RM_SCALE = 1.0  # screen fraction f: the RM map is multiplied by this
OUT_ROOT = CACHE_DIR / "slides_subpixel"
FIG_DIR = REPO / "notebooks" / "slides_pixel_sum"
COLUMN_RECORDS = (
    REPO / "docs" / "faraday-covariance-column-results.json",
    REPO / "docs" / "faraday-covariance-column-alpha1-results.json",
)
WMAP_PIXEL_RMS_30 = 1.257e4  # K at 30 MHz, all scales incl. WMAP noise
BETA_QU = -2.8
WMAP_BAND_RMS_30 = 5533.0  # K at 30 MHz, ell <= 30, noise subtracted


def set_center(center_mhz):
    global CENTER
    CENTER = float(center_mhz)


def out_dir():
    tag = "" if RM_SCALE == 1.0 else f"_f{RM_SCALE:g}"
    return OUT_ROOT / f"{CENTER:g}mhz{tag}"


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
    rm = maps["RM"] * RM_SCALE  # a screen at fraction f of the column
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
    # Context terms through the same pipeline, as power laws across the band:
    # the I-only sky (Haslam synchrotron + CMB) gives the leakage, and the
    # unrotated WMAP polarization gives the no-Faraday polarized term.
    i_sky, q0, u0 = sky_at_freq(maps, CENTER)
    ctx = np.zeros((3, 4, hp.nside2npix(NSIDE_DEG)))
    ctx[0, 0] = hp.ud_grade(i_sky - T_CMB, NSIDE_DEG)  # synchrotron I
    ctx[1, 0] = T_CMB  # CMB, flat
    ctx[2, 1] = hp.ud_grade(q0, NSIDE_DEG)  # unrotated Q at CENTER
    ctx[2, 2] = hp.ud_grade(u0, NSIDE_DEG)  # unrotated U at CENTER
    ctx_alm = np.asarray(
        cro.PolarizedSky(
            ctx,
            np.arange(1.0, 4.0),
            sampling="healpix",
            coord="galactic",
            convention="COSMO",
        ).compute_alm(lmax=LMAX)
    )
    W_ctx = engine.contract(beam, ctx_alm, tt, moon_location(), LMAX)
    ratio = ff / CENTER
    law = np.stack([ratio**BETA_I, np.ones_like(ff), ratio**BETA_QU])  # (3, F)
    c_ctx = law[:, :, None] * np.ones(4)
    ionly_zoom, ionly_parent = through_instrument(W_ctx[:2], c_ctx[:2])
    unrot_zoom, unrot_parent = through_instrument(W_ctx[2:], c_ctx[2:])
    print(f"context terms {_time.time() - t0:.0f} s", flush=True)
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
        ionly_zoom=ionly_zoom,  # (T, 3, 64, 4) I-only sky: I and its leakage
        ionly_parent=ionly_parent,
        unrot_zoom=unrot_zoom,  # (T, 3, 64, 4) unrotated WMAP polarization
        unrot_parent=unrot_parent,
        zoom_offsets_hz=chan.zoom_bin_offsets_hz(),
        parent_centers_mhz=parent_centers(CENTER),
        time_index=np.array(TIME_INDEX),
        lst_hours=np.array(TIME_INDEX) / 1024 * 24,
        nodes_mhz=nodes,
    )
    print(f"saved {out / 'spectra.npz'} {_time.time() - t0:.0f} s", flush=True)


def _column_rows():
    rows = []
    for path in COLUMN_RECORDS:
        if path.exists():
            rows.extend(json.loads(path.read_text())["results"])
    return rows


def ensemble_band(label_contains):
    """Zoom-bin speckle rms (K per 1 K intrinsic) from the column run."""
    for row in _column_rows():
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


def ensemble_draws(label_contains, n_draws=2, seed=5):
    """Gaussian realizations of the zoom-bin Q/U speckle (K per 1 K)."""
    for row in _column_rows():
        if (
            row["center_mhz"] == CENTER
            and label_contains in row["case"]["label"]
        ):
            z = np.load(REPO / row["path"])
            spk = z["speckle"]
            nb = spk.shape[0]
            # phase-0 Q and U over the zoom bins as one real Gaussian vector
            block = spk[3:nb, 1:3, 3:nb, 1:3].real.reshape(
                2 * (nb - 3), 2 * (nb - 3)
            )
            w, v = np.linalg.eigh((block + block.T) / 2)
            root = v * np.sqrt(np.clip(w, 0, None))
            rng = np.random.default_rng(seed)
            x = root @ rng.standard_normal((root.shape[1], n_draws))
            x = x.reshape(nb - 3, 2, n_draws)
            return z["frequency_mhz"][3:], x[:, 0], x[:, 1]
    raise KeyError(label_contains)


def _shade_aliased_bins(axes, parents, off, sel, label_ax):
    fine_off = np.arange(-100e3, 100e3, chan.FINE_STEP_HZ)
    wz = chan.zoom_weights(fine_off)
    outside = np.abs(fine_off) > 12.5e3
    folded = np.sum(np.abs(wz[outside]), axis=0) / np.sum(np.abs(wz), axis=0)
    half = chan.ZOOM_STEP_HZ * 1e-6 / 2
    for ax in axes:
        first = True
        for j in sel:
            for k in np.where(folded > 0.05)[0]:
                f0 = parents[j] + off[k]
                ax.axvspan(
                    f0 - half,
                    f0 + half,
                    color="0.3",
                    alpha=0.12,
                    lw=0,
                    label=(
                        "aliased edge zoom bins (>5 % folded response)"
                        if first and ax is label_ax
                        else None
                    ),
                )
                first = False


def ensemble_draws_on(label_contains, freqs_mhz, n_draws=1, seed=5):
    """Gaussian realizations of the Q/U speckle on an arbitrary frequency
    grid, from the measured central-parent variance and the Gaussian-screen
    frequency correlation exp(-2 sigma^2 (dlambda^2)^2), which matches the
    measured 64-bin correlation to ~1 % rms away from the aliased edge bins.
    Q and U are drawn independently (their measured cross-correlation is
    below 1 %). Edge-bin aliasing is not modelled here."""
    from lusee_faraday.conventions import lambda_squared

    fe, rq, ru = ensemble_band(label_contains)
    inner = slice(8, 56)
    oe = np.argsort(fe)
    var = (np.mean(rq[oe][inner] ** 2), np.mean(ru[oe][inner] ** 2))
    l2 = lambda_squared(np.asarray(freqs_mhz))
    corr = np.exp(-2 * SIGMA_COLUMN**2 * (l2[:, None] - l2[None, :]) ** 2)
    w, v = np.linalg.eigh(corr)
    root = v * np.sqrt(np.clip(w, 0, None))
    rng = np.random.default_rng(seed)
    draws = [
        np.sqrt(vk) * (root @ rng.standard_normal((root.shape[1], n_draws)))
        for vk in var
    ]
    return draws[0], draws[1]


def plot_phase_panels(axes, fz, zoom, phases, colors, parents, off, sel):
    """Faraday Q and U at four sidereal phases with the alpha=2 bands."""
    for s, (ax, name) in enumerate(zip(axes, ("Q", "U")), start=1):
        for it, ph in enumerate(phases):
            ax.plot(
                fz,
                zoom[it, :, s],
                color=colors[it],
                lw=1.2,
                label=f"Faraday polarized signal, sidereal phase {ph:.2f}",
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
        fe, dq, du = ensemble_draws("alpha=2 dphi1=1", n_draws=1)
        oe = np.argsort(fe)
        ax.plot(
            fe[oe],
            (dq if s == 1 else du)[oe, 0] * wmap_pixel_rms(),
            color="0.25",
            lw=0.9,
            ls="--",
            label=(
                r"one realization drawn from the $\alpha$=2, "
                r"$\Delta\phi_{1^\circ}$=1 ensemble"
            ),
        )
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_ylabel(f"Faraday {name} alone [K]")
        ax.grid(alpha=0.3)
    _shade_aliased_bins(axes, parents, off, sel, axes[0])


def plot_ladder_panels(axes, fz, zoom, parents, off, sel):
    """One sidereal phase: the linear map realization against one draw of
    the alpha=5/3 and alpha=1 ensembles (anchor 1), bands optional."""
    it = CONTEXT_PHASE
    cases = (
        ("alpha=5/3 dphi1=1", "C2", "--", r"$\alpha$ = 5/3"),
        ("alpha=1 dphi1=1", "C3", ":", r"$\alpha$ = 1"),
    )
    for s, (ax, name) in enumerate(zip(axes, ("Q", "U")), start=1):
        ax.plot(
            fz,
            zoom[it, :, s],
            color="k",
            lw=1.6,
            label="linear interpolation",
        )
        for label, color, ls, tag in cases:
            try:
                fe, rq, ru = ensemble_band(label)
            except KeyError:
                print(f"no column record yet for {label}; skipped")
                continue
            if BANDS and SPAN == 1:
                band = (rq if s == 1 else ru) * wmap_pixel_rms()
                oe = np.argsort(fe)
                ax.fill_between(
                    fe[oe],
                    -band[oe],
                    band[oe],
                    color=color,
                    alpha=0.15,
                    lw=0,
                    label=rf"linear + Gaussian sub-pixel ({tag}): "
                    r"$\pm$1 rms of the ensemble",
                )
            dq, du = ensemble_draws_on(label, fz, n_draws=1)
            ax.plot(
                fz,
                (dq if s == 1 else du)[:, 0] * wmap_pixel_rms(),
                color=color,
                lw=1.4,
                ls=ls,
                label=rf"linear + Gaussian sub-pixel ({tag})",
            )
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_ylabel(f"Faraday {name} alone [K]")
        ax.grid(alpha=0.3)
    _shade_aliased_bins(axes, parents, off, sel, axes[0])


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(out / "spectra.npz")
    off = d["zoom_offsets_hz"] * 1e-6
    parents = d["parent_centers_mhz"]
    sel = [1] if SPAN == 1 else [0, 1, 2]
    fz = np.concatenate([parents[j] + off for j in sel])
    order = np.argsort(fz)
    fz = fz[order]

    def bins(key, comp=None):
        a = np.concatenate([d[key][:, j] for j in sel], axis=1)[:, order]
        return a if comp is None else a[..., comp]

    k_zoom = bins("unit_zoom", 0)  # loaded units per K
    k_parent = d["unit_parent"][:, 1, 0]
    zoom = bins("stokes_zoom") / k_zoom[:, :, None]  # K
    leak = bins("ionly_zoom") / k_zoom[:, :, None]  # K
    unrot = bins("unrot_zoom") / k_zoom[:, :, None]  # K
    i_ref = d["ionly_parent"][:, 1, 0] / k_parent  # beam-weighted I, K
    q_leak = d["ionly_parent"][:, 1, 1] / k_parent  # I-only leakage Q, K
    phases = d["lst_hours"] / 24
    colors = ["C0", "C1", "C2", "C4"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 9.6),
        sharex=True,
        gridspec_kw=dict(height_ratios=(1.15, 1, 1)),
    )
    # Context: the full Q spectrum at one sidereal phase.
    it = CONTEXT_PHASE
    ax = axes[0]
    ax.plot(
        fz,
        leak[it, :, 1],
        color="k",
        lw=1.6,
        label="I-only sky: leakage I -> Q alone",
    )
    ax.plot(
        fz,
        leak[it, :, 1] + unrot[it, :, 1],
        color="0.5",
        lw=1.6,
        label="leakage + WMAP polarization, no Faraday rotation",
    )
    ax.plot(
        fz,
        leak[it, :, 1] + zoom[it, :, 1],
        color=colors[it],
        lw=1.6,
        ls="--",
        label="leakage + WMAP polarization, Faraday rotated",
    )
    ax.set_ylabel("total pseudo-Stokes Q [K]")
    ax.set_title(
        f"Full Q at sidereal phase {phases[it]:.2f}: beam-weighted "
        f"I = {i_ref[it]:.0f} K, leakage Q = {q_leak[it]:.0f} K,\n"
        f"unrotated polarization Q = {unrot[it, :, 1].mean():.0f} K, "
        f"Faraday speckle rms = {zoom[it, :, 1].std():.2f} K",
        fontsize=9.5,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    if LAYOUT == "ladder":
        plot_ladder_panels(axes[1:], fz, zoom, parents, off, sel)
    else:
        plot_phase_panels(
            axes[1:], fz, zoom, phases, colors, parents, off, sel
        )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7.5,
        ncol=2 if LAYOUT == "ladder" else 3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    axes[2].set_xlabel(
        "frequency [MHz], 64 zoom bins of the central 25 kHz parent channel"
        if SPAN == 1
        else "frequency [MHz], zoom bins of the three 25 kHz parent channels"
    )
    if SPAN == 3:
        for ax in axes:
            for edge in (parents[0] + 0.0125, parents[1] + 0.0125):
                ax.axvline(edge, color="k", lw=0.6, ls=":")
    fig.suptitle(
        "WMAP K polarization rotated with the Hutschenreuter RM (linear "
        "sub-pixel interpolation),\nthrough the as-built LuSEE instrument at "
        f"{CENTER:g} MHz",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    FIG_DIR.mkdir(exist_ok=True)
    suffix = (
        ("" if SPAN == 1 else "_3parents")
        + ("_ladder" if LAYOUT == "ladder" else "")
        + ("" if BANDS or SPAN == 3 else "_nobands")
    )
    name = FIG_DIR / f"10_subpixel_spectra_{CENTER:g}MHz{suffix}.png"
    fig.savefig(name, dpi=140)
    print("saved", name)
    for it, ph in enumerate(phases):
        q, u = zoom[it, :, 1], zoom[it, :, 2]
        print(
            f"phase {ph:.2f}: Faraday zoom rms Q {q.std():.3g} K, "
            f"U {u.std():.3g} K; I {i_ref[it]:.0f} K; leak Q "
            f"{q_leak[it]:.1f} K; unrotated Q {unrot[it, :, 1].mean():.1f} K "
            f"-> speckle/I {q.std() / i_ref[it]:.2e}, "
            f"leak/I {q_leak[it] / i_ref[it]:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compute", "plot"])
    parser.add_argument("--center", type=float, default=30.0)
    parser.add_argument("--span", type=int, choices=(1, 3), default=1)
    parser.add_argument(
        "--layout", choices=("phases", "ladder"), default="phases"
    )
    parser.add_argument("--no-bands", action="store_true")
    parser.add_argument("--rm-scale", type=float, default=1.0)
    args = parser.parse_args()
    set_center(args.center)
    global SPAN, LAYOUT, BANDS, RM_SCALE
    SPAN = args.span
    LAYOUT = args.layout
    BANDS = not args.no_bands
    RM_SCALE = args.rm_scale
    if args.stage == "compute":
        compute(out_dir())
    else:
        plot(out_dir())


if __name__ == "__main__":
    main()
