#!/usr/bin/env python3
"""Pilot: Faraday rotation moves polarized power to small angular scales.

For a foreground screen, the rotated field Pi = P0 exp(2i phi lambda^2)
has |Pi| = |P0| at every point, so total polarized power is conserved;
rotation only moves it between multipoles. The beam passes low ell, so at
strong rotation it sees the flat low-ell floor of the redistributed field,
whose level is set by the small-scale coherence of phi.

Two stages:

    .venv/bin/python scripts/faraday_redistribution.py window
        Polarization window W_ell of the as-built BGL_v16 four-port
        response at 10/30/50 MHz (heavy: loads the 631 MB artifact).
        Run under ulimit -v 24000000 in the background.

    .venv/bin/python scripts/faraday_redistribution.py plot
        Angular power spectra of the rotated field for a Gaussian screen
        with a powered-exponential structure function, the beam-weighted
        surviving power versus frequency, and the figure.

The rotated field is treated as a scalar complex field (the spin-2
character matters only at ell < ~2); covariances use chordal distance,
for which exp(-(d/theta0)^alpha) is positive definite for alpha <= 2.
"""

import hashlib
import json
import sys

import common  # noqa: F401  (sets JAX_ENABLE_X64 before any jax import)
from common import CACHE_DIR, FIG_DIR, GEN_DIR, REPO, RESPONSE_PATH

import numpy as np

C_MHZ_M = 299.792458  # c in m * MHz
WINDOW_FREQS = (10.0, 30.0, 50.0)
WINDOW_LMAX = 96
WINDOW_FILE = GEN_DIR / "faraday_redistribution" / "beam_window.npz"
RECORD_FILE = REPO / "docs" / "faraday-redistribution-pilot-results.json"

# Pilot screen and intrinsic sky (dimensionless; |P0|^2 = 1).
SIGMA_PHI = 10.0  # rad/m^2, saturation rms of phi (large-scale spread)
DPHI_1DEG = (0.3, 1.0, 3.0)  # rad/m^2, rms phi difference at 1 degree
ALPHAS = (1.0, 5.0 / 3.0, 2.0)  # small-separation slope of D(theta)
BASE_DPHI, BASE_ALPHA = 1.0, 5.0 / 3.0
THETA_P_DEG = 20.0  # intrinsic P0 correlation scale (exponential)
PANEL_A_FREQS = (1400.0, 150.0, 50.0, 30.0, 10.0)
BETA_QU = -2.8  # intrinsic polarized brightness spectral index


def lam2(freq_mhz):
    return (C_MHZ_M / np.asarray(freq_mhz, dtype=float)) ** 2


# --------------------------------------------------------------------------
# Beam window from the real response
# --------------------------------------------------------------------------


def _file_digest(path, nbytes=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def compute_window():
    from lusee_faraday import response as rsp
    from lusee_faraday.conventions import DUAL_BLOCKS, PORT_PAIRS

    print(f"loading {RESPONSE_PATH}", flush=True)
    resp = rsp.load_response(RESPONSE_PATH)
    i_blk = DUAL_BLOCKS.index("I")
    pm, pp = DUAL_BLOCKS.index("P_MINUS"), DUAL_BLOCKS.index("P_PLUS")
    w_pol, w_int = [], []
    for f in WINDOW_FREQS:
        print(f"pair-Stokes alms at {f} MHz, lmax {WINDOW_LMAX}", flush=True)
        alms = np.asarray(rsp.four_port_pair_alms(resp, f, WINDOW_LMAX))
        # (pair, block, ell, m): power per ell summed over m.
        pol = (np.abs(alms[:, pm]) ** 2 + np.abs(alms[:, pp]) ** 2).sum(-1)
        w_pol.append(pol)
        w_int.append((np.abs(alms[:, i_blk]) ** 2).sum(-1))
    WINDOW_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        WINDOW_FILE,
        freqs=np.array(WINDOW_FREQS),
        lmax=WINDOW_LMAX,
        w_pol=np.array(w_pol),
        w_int=np.array(w_int),
        pairs=np.array(PORT_PAIRS),
        response=str(RESPONSE_PATH),
        response_digest=_file_digest(RESPONSE_PATH),
    )
    print(f"wrote {WINDOW_FILE}", flush=True)


def load_window():
    if not WINDOW_FILE.exists():
        raise SystemExit(f"{WINDOW_FILE} missing: run the 'window' stage")
    w = np.load(WINDOW_FILE)
    if str(w["response"]) != str(RESPONSE_PATH):
        raise SystemExit("beam window was made from a different response")
    return w


# --------------------------------------------------------------------------
# Rotated-field angular power spectrum
# --------------------------------------------------------------------------


def theta0_rad(dphi_1deg, alpha, sigma=SIGMA_PHI):
    """theta0 giving D(1 deg) = dphi_1deg^2, D = 2s^2[1-e^{-(d/t0)^a}]."""
    d1 = 2 * np.sin(np.radians(1.0) / 2)
    frac = dphi_1deg**2 / (2 * sigma**2)
    return d1 / (-np.log1p(-frac)) ** (1 / alpha)


def correlation(theta, freq_mhz, dphi_1deg, alpha, rotate=True):
    """Raw second moment <Pi(n) Pi*(n')> at separation theta, |P0|^2=1."""
    d = 2 * np.sin(theta / 2)
    c_p0 = np.exp(-d / (2 * np.sin(np.radians(THETA_P_DEG) / 2)))
    if not rotate:
        return c_p0
    t0 = theta0_rad(dphi_1deg, alpha)
    struct = 2 * SIGMA_PHI**2 * -np.expm1(-((d / t0) ** alpha))
    return c_p0 * np.exp(-2 * lam2(freq_mhz) ** 2 * struct)


def theta_grid():
    """Log core (resolves the coherence scale) plus a uniform body."""
    core = np.logspace(-13, -2, 11 * 1500)
    body = np.linspace(1e-2, np.pi, 160001)[1:]
    return np.concatenate([[0.0], core, body])


def cl_legendre(c_theta, theta, lmax):
    """C_ell = 2 pi int C(theta) P_ell(cos theta) dcos(theta), ell<=lmax."""
    x = np.cos(theta)
    # Trapezoid weights in theta for d(cos) = sin(theta) dtheta.
    dt = np.diff(theta)
    w = np.zeros_like(theta)
    w[:-1] += dt / 2
    w[1:] += dt / 2
    f = 2 * np.pi * c_theta * np.sin(theta) * w
    cl = np.empty(lmax + 1)
    p_prev, p = np.ones_like(x), x.copy()
    cl[0] = f.sum()
    if lmax >= 1:
        cl[1] = (f * p).sum()
    for ell in range(1, lmax):
        p_prev, p = p, ((2 * ell + 1) * x * p - ell * p_prev) / (ell + 1)
        cl[ell + 1] = (f * p).sum()
    return cl


def cl_flat(c_theta, theta, ells):
    """Flat-sky Hankel transform, valid where C(theta) is confined to small
    angles: C_ell = 2 pi int theta C(theta) J0((ell+1/2) theta) dtheta."""
    from scipy.special import j0

    out = np.empty(len(ells))
    dt = np.diff(theta)
    w = np.zeros_like(theta)
    w[:-1] += dt / 2
    w[1:] += dt / 2
    g = 2 * np.pi * theta * c_theta * w
    for i, ell in enumerate(ells):
        out[i] = (g * j0((ell + 0.5) * theta)).sum()
    return out


def coherence_angle(freq_mhz, dphi_1deg, alpha):
    """theta_c where 2 lambda^4 D(theta_c) = 1, small-separation form."""
    return np.radians((2 * lam2(freq_mhz) ** 2 * dphi_1deg**2) ** (-1 / alpha))


L_SWITCH = 3000


def spectrum(freq_mhz, dphi_1deg, alpha, rotate=True):
    """Return (ells, C_ell, match): exact Legendre to L_SWITCH; above it a
    flat-sky Hankel transform when the correlation is confined to small
    angles, with ``match`` the relative Legendre/Hankel difference at
    L_SWITCH. Unconfined correlations (no or weak rotation) stop at
    L_SWITCH and return match = nan."""
    theta = theta_grid()
    c = correlation(theta, freq_mhz, dphi_1deg, alpha, rotate)
    cl_lo = cl_legendre(c, theta, L_SWITCH)
    ells_lo = np.arange(L_SWITCH + 1)
    if not rotate or correlation(0.2, freq_mhz, dphi_1deg, alpha) > 1e-30:
        return ells_lo, cl_lo, np.nan
    # Dedicated dense log grid out to where the correlation is < 1e-30.
    tc = coherence_angle(freq_mhz, dphi_1deg, alpha)
    t_hi = tc * (2 * np.log(1e30)) ** (1 / alpha)
    th = np.concatenate(
        [[0.0], np.logspace(np.log10(tc) - 8, np.log10(t_hi), 60000)]
    )
    ch = correlation(th, freq_mhz, dphi_1deg, alpha)
    ells_hi = np.unique(
        np.logspace(np.log10(L_SWITCH), np.log10(300 / tc), 300).astype(int)
    )
    cl_hi = cl_flat(ch, th, ells_hi)
    match = cl_hi[0] / cl_lo[L_SWITCH] - 1
    keep = ells_hi > L_SWITCH
    return (
        np.concatenate([ells_lo, ells_hi[keep]]),
        np.concatenate([cl_lo, cl_hi[keep]]),
        match,
    )


def total_power(ells, cl):
    """sum (2l+1)/(4pi) C_l: exact sum to L_SWITCH, then trapezoid in ln l.
    Equals C(0) = |P0|^2 = 1 when the spectrum is complete."""
    lo = ells <= L_SWITCH
    s = ((2 * ells[lo] + 1) / (4 * np.pi) * cl[lo]).sum()
    if lo.all():
        return s
    e = ells[L_SWITCH:].astype(float)
    y = e * (2 * e + 1) / (4 * np.pi) * cl[L_SWITCH:]
    return (
        s
        - y[0] / (2 * e[0])
        + (np.diff(np.log(e)) * (y[1:] + y[:-1]) / 2).sum()
    )


def beam_power(freq_mhz, dphi_1deg, alpha, w_ell, rotate=True):
    """Ensemble <|V|^2> for an isotropic rotated field: sum_l W_l C_l."""
    lmax = len(w_ell) - 1
    theta = theta_grid()
    c = correlation(theta, freq_mhz, dphi_1deg, alpha, rotate)
    return (w_ell * cl_legendre(c, theta, lmax)).sum()


def map_structure_function(seps_deg=(0.25, 0.5, 1, 2, 4, 8), n=400000):
    """rms and median |phi(n)-phi(n')| of the Hutschenreuter posterior mean
    at fixed separations, by latitude cut (random pairs, fixed seed)."""
    import h5py
    import healpy as hp

    from common import DATA_DIR

    with h5py.File(DATA_DIR / "faraday2020v2.hdf5", "r") as f:
        m = f["faraday_sky_mean"][:]
    nside = hp.npix2nside(m.size)
    rng = np.random.default_rng(1)
    th = np.arccos(rng.uniform(-1, 1, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    b = 90 - np.degrees(th)
    v = np.stack(hp.ang2vec(th, ph).T)
    e1 = np.stack(
        [np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)]
    )
    e2 = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)])
    p1 = hp.ang2pix(nside, th, ph)
    out = {}
    for sep in seps_deg:
        a = rng.uniform(0, 2 * np.pi, n)
        d = np.radians(sep)
        w = np.cos(d) * v + np.sin(d) * (np.cos(a) * e1 + np.sin(a) * e2)
        dphi = np.abs(m[p1] - m[hp.vec2pix(nside, *w)])
        for name, sel in (
            ("|b|>30", np.abs(b) > 30),
            ("|b|>60", np.abs(b) > 60),
        ):
            out[f"{sep}deg {name}"] = {
                "rms": float(np.sqrt((dphi[sel] ** 2).mean())),
                "median": float(np.median(dphi[sel])),
            }
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def make_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    win = load_window()
    lmax = int(win["lmax"])
    ell_w = np.arange(lmax + 1)
    i30 = list(win["freqs"]).index(30.0)
    # Auto pair (0,0) = N port; window normalised to unit sum.
    pairs = [tuple(p) for p in win["pairs"]]
    w30 = win["w_pol"][i30, pairs.index((0, 0))]
    w30n = w30 / w30.sum()
    cum = np.cumsum(w30n)
    l90, l99 = (int(np.searchsorted(cum, q)) for q in (0.9, 0.99))

    record = {"window_l90_30mhz": l90, "window_l99_30mhz": l99}
    for k, f in enumerate(win["freqs"]):
        wf = win["w_pol"][k, pairs.index((0, 0))]
        cf = np.cumsum(wf / wf.sum())
        record[f"window_l99_{int(f)}mhz"] = int(np.searchsorted(cf, 0.99))
        record[f"window_tail_l{lmax - 16}_{int(f)}mhz"] = float(
            1 - cf[lmax - 16]
        )

    # Panel (a): redistribution at fixed screen.
    spectra, checks = {}, {}
    ells0, cl0, m0 = spectrum(1.0, BASE_DPHI, BASE_ALPHA, rotate=False)
    spectra["off"] = (ells0, cl0)
    checks["off"] = {"total": total_power(ells0, cl0), "match": m0}
    for f in PANEL_A_FREQS:
        e, c, m = spectrum(f, BASE_DPHI, BASE_ALPHA)
        spectra[f] = (e, c)
        checks[f] = {"total": total_power(e, c), "match": m}
    record["panel_a_checks"] = {str(k): v for k, v in checks.items()}

    # Panel (b): beam-weighted surviving fraction versus frequency.
    freqs = np.logspace(np.log10(5.0), np.log10(1500.0), 60)
    off = beam_power(1.0, BASE_DPHI, BASE_ALPHA, w30, rotate=False)
    curves = {}
    for a in ALPHAS:
        for s1 in DPHI_1DEG:
            if a != BASE_ALPHA and s1 != BASE_DPHI:
                continue
            curves[(a, s1)] = np.array(
                [beam_power(f, s1, a, w30) / off for f in freqs]
            )

    out_dir = GEN_DIR / "faraday_redistribution"
    np.savez(
        out_dir / "pilot.npz",
        freqs=freqs,
        **{f"frac_a{a:.3f}_d{s1}": v for (a, s1), v in curves.items()},
    )

    fig = plt.figure(figsize=(7.0, 8.0))
    top, bot = fig.subfigures(2, 1, height_ratios=[1.25, 1], hspace=0.04)
    ax_a, ax_w = top.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.06},
    )
    ax_b, ax_c = bot.subplots(1, 2, gridspec_kw={"wspace": 0.34})

    cmap = plt.get_cmap("viridis")
    e, c = spectra["off"]
    sel = (e >= 1) & (e <= 2000)
    ax_a.loglog(
        e[sel],
        e[sel] * (2 * e[sel] + 1) * c[sel] / (4 * np.pi),
        color="k",
        ls="--",
        lw=1.2,
        label="no rotation",
    )
    for k, f in enumerate(PANEL_A_FREQS):
        e, c = spectra[f]
        sel = (e >= 1) & ((e <= 2000) | np.isfinite(checks[f]["match"]))
        y = e[sel] * (2 * e[sel] + 1) * c[sel] / (4 * np.pi)
        ax_a.loglog(
            e[sel],
            np.where(y > 0, y, np.nan),
            color=cmap(k / (len(PANEL_A_FREQS) - 1)),
            lw=1.4,
            label=f"{f:g} MHz",
        )
    ax_a.set_ylim(1e-9, 3)
    ax_a.set_xlim(1, 1e9)
    ax_a.set_ylabel(r"$\ell(2\ell+1)C_\ell/4\pi$  [$|P_0|^2$ per ln $\ell$]")
    ax_a.legend(fontsize=7, ncol=3, loc="upper right", frameon=False)
    ax_a.tick_params(labelbottom=False)
    ax_a.text(0.01, 0.95, "(a)", transform=ax_a.transAxes, va="top")

    for k, f in enumerate(win["freqs"]):
        wf = win["w_pol"][k, pairs.index((0, 0))]
        ax_w.loglog(
            ell_w[1:],
            ell_w[1:] * wf[1:] / wf.sum(),
            lw=1.2,
            label=f"{f:g} MHz",
            color=plt.get_cmap("tab10")(k),
        )
    ax_w.axvline(l99, color="0.5", lw=0.8, ls=":")
    ax_w.set_ylabel(r"$\ell W_\ell/\sum W$", fontsize=8)
    ax_w.set_xlabel(r"multipole $\ell$")
    ax_w.set_ylim(1e-4, 3)
    ax_w.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=3)
    ax_w.text(
        0.01,
        0.9,
        "beam (N port)",
        transform=ax_w.transAxes,
        va="top",
        fontsize=7,
    )

    styles = {0.3: ":", 1.0: "-", 3.0: "--"}
    acolor = {1.0: "C3", 5.0 / 3.0: "C0", 2.0: "C2"}
    for (a, s1), v in curves.items():
        lbl = rf"$\alpha$={a:.2g}, $\Delta\phi_{{1^\circ}}$={s1:g}"
        ax_b.loglog(
            freqs, v, color=acolor[a], ls=styles[s1], lw=1.3, label=lbl
        )
        ax_c.loglog(
            freqs,
            v * (freqs / 150.0) ** (2 * BETA_QU),
            color=acolor[a],
            ls=styles[s1],
            lw=1.3,
        )
    ax_b.set_xlabel("frequency [MHz]")
    ax_b.set_ylabel("beam-weighted power / unrotated")
    ax_b.set_ylim(1e-12, 2)
    ax_b.legend(fontsize=6, loc="lower right", frameon=False)
    ax_b.axvspan(5, 50, color="0.9", zorder=0)
    ax_b.text(0.03, 0.95, "(b)", transform=ax_b.transAxes, va="top")
    ax_c.set_xlabel("frequency [MHz]")
    ax_c.set_ylabel(r"beam-weighted $|P|^2$ / unrotated at 150 MHz")
    ax_c.axvspan(5, 50, color="0.9", zorder=0)
    ax_c.text(0.03, 0.04, "(c)", transform=ax_c.transAxes, va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"faraday_redistribution_pilot.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    record["theta0_deg_base"] = float(
        np.degrees(theta0_rad(BASE_DPHI, BASE_ALPHA))
    )
    record["window_response_digest"] = str(win["response_digest"])
    record["surviving_fraction"] = {
        f"alpha={a:.3f},dphi1={s1}": {
            f"{f:g}MHz": float(np.interp(np.log(f), np.log(freqs), v))
            for f in (10.0, 30.0, 50.0, 150.0)
        }
        for (a, s1), v in curves.items()
    }
    record["map_structure_function"] = map_structure_function()
    with open(RECORD_FILE, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "plot"
    (GEN_DIR / "faraday_redistribution").mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    {"window": compute_window, "plot": make_plot}[stage]()
