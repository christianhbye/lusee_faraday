"""Step 4: 2D power spectra of the diffuse-sky waterfalls.

For each band center (30, 10, 50 MHz) and each frequency sampling
(fine grid / ideal zoom bins / real zoom bins) the complex linear
polarization P_obs = Q_obs + i U_obs of the ideal polarimeter is
transformed:

- over time (1024 periodic samples over one lunar sidereal day, no
  window) -> temporal frequency in Hz,
- over observing frequency (Blackman-Harris window) -> delay in us.

A Faraday screen of depth phi imposes chi(nu) = 2 phi lambda^2(nu),
i.e. a delay tau = |d chi / d nu| / (2 pi) = 2 phi c^2 / (pi nu^3):
about 20 us per 10 rad/m^2 at 30 MHz, scaling as nu^-3.  The no-Faraday
sky sits at tau ~ 0, so Faraday shows up as power at finite delay whose
location scales steeply with the band center.

Outputs: generated_data/real{C}_pspec.npz and figures
report/figures/real{C}_pspec_*.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.signal.windows import blackmanharris  # noqa: E402

from common import FIG_DIR, GEN_DIR, SIDEREAL_DAY_S  # noqa: E402
from lusee_faraday import _legacy_pixel as fp  # noqa: E402

C_LIGHT = 299792458.0


def pspec_2d(P, dnu_hz):
    """(T, F) complex waterfall -> (T, F) 2D power spectrum + axes.

    Time axis is periodic (plain FFT); frequency axis is tapered with a
    Blackman-Harris window.  Returns (ps, f_time_hz, tau_us), both axes
    fftshifted.
    """
    T, F = P.shape
    w = blackmanharris(F)
    w /= np.sqrt(np.mean(w**2))
    Pw = P * w[None, :]
    G = np.fft.fft2(Pw) / (T * F)
    ps = np.abs(np.fft.fftshift(G)) ** 2
    f_time = np.fft.fftshift(np.fft.fftfreq(T, d=SIDEREAL_DAY_S / T))
    tau = np.fft.fftshift(np.fft.fftfreq(F, d=dnu_hz)) * 1e6  # us
    return ps, f_time, tau


def faraday_delay_us(phi, nu_hz):
    return 2 * phi * C_LIGHT**2 / (np.pi * nu_hz**3) * 1e6


def zoom_transfer(center, pad=32):
    """Transfer matrix from a slot-gridded true spectrum to zoom bins.

    Models the true spectrum as piecewise values on the 390.625 Hz
    slot grid, extended by `pad` slots on each side of the 192 covered
    slots so that Nyquist-bin folds from just outside the band edges
    are representable.  A[j, i] is the (exact) response of measured
    zoom bin j — same normalized weights as integrate_spectrometer —
    summed over the fine frequencies belonging to slot i.  Returns
    (A, i0) with i0 the index of the first of the 192 nominal slots.
    """
    from common import fine_freqs, parent_centers

    ff = fine_freqs(center)
    centers = parent_centers(center)
    zf, order = fp.zoom_frequency_grid(centers)
    step_mhz = fp.ZOOM_STEP_HZ * 1e-6
    nslot = 192 + 2 * pad
    slot0 = zf[0] - pad * step_mhz
    idx = np.round((ff - slot0) / step_mhz).astype(int)
    A = np.zeros((192, nslot))
    for p, fc in enumerate(centers):
        off = (ff - fc) * 1e6
        sel = np.abs(off) <= 50000.0 + 1e-6
        wz = fp.zoom_weights(off[sel])  # (Nf_sel, 64) normalized
        fi = np.nonzero(sel)[0]
        good = (idx[fi] >= 0) & (idx[fi] < nslot)
        for j, (pp, k) in enumerate(order):
            if pp != p:
                continue
            np.add.at(A[j], idx[fi[good]], wz[good, k])
    return A, pad


def zoom_deconvolve(m, A, pad, lam=0.03):
    """Smoothness-regularized inversion of the zoom transfer.

    m : (..., 192) measured zoom-bin values (sorted grid order).
    Solves min ||A s - m||^2 + lam_s ||D2 s||^2 with D2 the second
    difference: a plain min-norm inverse leaves the parent-boundary
    and pad slots degenerate (they only enter through the shared
    Nyquist-bin folds), and the resulting few-slot errors ring across
    the whole delay range.  The curvature prior resolves the
    degeneracy with a smooth continuation instead.  Returns the
    central 192 slots of s_hat.
    """
    n = A.shape[1]
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i : i + 3] = (1.0, -2.0, 1.0)
    AtA = A.T @ A
    reg = lam * np.trace(AtA) / n / 16.0  # 16 ~ max eig of D2^T D2
    G = np.linalg.solve(AtA + reg * (D2.T @ D2), A.T)
    s_hat = m @ G.T  # (..., nslot)
    return s_hat[..., pad : pad + 192]


def compute(center, thin=4):
    """Return dict of (ps, f_time, tau) for the three samplings."""
    from zenith_weights import get_weights

    xv, yv = get_weights(center)  # zenith-calibrated polarimeter

    def stokes(channels):
        return fp.polarimeter_from_channels(channels, xv, yv)

    tag = f"real{center:g}"
    meta = np.load(GEN_DIR / f"{tag}_meta.npz")
    binned = np.load(GEN_DIR / f"{tag}_binned.npz")
    wf = np.load(GEN_DIR / f"{tag}_fine_waterfall.npy", mmap_mode="r")
    nt = wf.shape[0]

    out = {}
    # fine grid, thinned in frequency to keep memory sane; thinning by
    # `thin` reduces the delay span but keeps the resolution.
    S = np.empty((nt, wf.shape[1] // thin, 4))
    for s in range(0, nt, 64):
        sl = slice(s, min(s + 64, nt))
        S[sl] = stokes(np.asarray(wf[sl, ::thin, :]))
    P = S[..., 1] + 1j * S[..., 2]
    dnu = fp.FINE_STEP_HZ * thin
    out["fine"] = pspec_2d(P, dnu)
    del S, P

    # zoom bins: 192 contiguous channels across the three parents
    zf, order = fp.zoom_frequency_grid(binned["parent_centers_mhz"])
    for key in ("zoom", "ideal_zoom"):
        Z = binned[key]  # (T, 3, 64, 16)
        wf_z = np.stack(
            [Z[:, p, k, :] for p, k in order], axis=1
        )  # (T, 192, 16)
        Sz = stokes(wf_z)
        Pz = Sz[..., 1] + 1j * Sz[..., 2]
        out[key] = pspec_2d(Pz, fp.ZOOM_STEP_HZ)
        if key == "zoom":
            # linear deconvolution of the known bin transfer (Nyquist
            # folding + response leakage) before the delay transform
            A, pad = zoom_transfer(center)
            P_dec = zoom_deconvolve(Pz, A, pad)
            out["zoom_deconv"] = pspec_2d(P_dec, fp.ZOOM_STEP_HZ)
    return out, meta


def fig_pspec(center, results):
    names = {
        "fine": "fine grid",
        "ideal_zoom": "ideal zoom bins",
        "zoom": "real zoom bins",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=False)
    # common delay range: what the zoom sampling can reach
    tau_lim = 1e6 / (2 * fp.ZOOM_STEP_HZ)  # 1280 us
    vmax = max(r[0].max() for r in results.values())
    for ax, key in zip(axes, ("fine", "ideal_zoom", "zoom")):
        ps, f_time, tau = results[key]
        im = ax.pcolormesh(
            f_time * 1e6,
            tau,
            ps.T,
            norm=LogNorm(vmin=vmax * 1e-14, vmax=vmax),
            cmap="inferno",
            shading="auto",
            rasterized=True,
        )
        ax.set_ylim(-tau_lim, tau_lim)
        # the drifting sky only populates the lowest ~harmonics of the
        # sidereal day (beam is band-limited to ell ~ 30)
        ax.set_xlim(-40, 40)
        ax.set_title(names[key], fontsize=10)
        ax.set_xlabel(r"temporal frequency  [$\mu$Hz]")
        # predicted Faraday delay band for phi = 10..100 rad/m^2
        for phi, ls in ((10, ":"), (100, "--")):
            t_fd = faraday_delay_us(phi, center * 1e6)
            for sign in (+1, -1):
                ax.axhline(sign * t_fd, color="cyan", lw=0.8, ls=ls)
        fig.colorbar(im, ax=ax, shrink=0.9)
    axes[0].set_ylabel(r"delay  [$\mu$s]")
    fig.suptitle(
        rf"$|\tilde P(f_t, \tau)|^2$ at {center:g} MHz  "
        r"(cyan: $\tau_{\rm FD}$ for $\phi$ = 10 (dotted), "
        r"100 (dashed) rad/m$^2$)",
        y=1.02,
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"real{center:g}_pspec.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  wrote real{center:g}_pspec", flush=True)


def fig_delay_profile(all_results):
    """Per-band delay profiles: fine vs ideal vs real zoom sampling."""
    centers = sorted(all_results)
    fig, axes = plt.subplots(
        1, len(centers), figsize=(4.3 * len(centers), 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    styles = (
        ("fine", "fine grid", "C0", 1.0),
        ("ideal_zoom", "ideal zoom", "C2", 1.0),
        ("zoom", "real zoom", "C1", 1.0),
        ("zoom_deconv", "real zoom, deconvolved", "C3", 1.0),
    )
    for ax, center in zip(axes, centers):
        results = all_results[center]
        for key, label, color, lw in styles:
            ps, f_time, tau = results[key]
            prof = ps.mean(axis=0)
            pos = tau > 0
            ax.plot(tau[pos], prof[pos], color=color, lw=lw, label=label)
        # expected delay of the bulk (|RM| ~ 30) and the maximum
        # (|RM| ~ 2400, galactic plane) Faraday screens
        for phi, ls in ((30.0, ":"), (2400.0, "--")):
            ax.axvline(
                faraday_delay_us(phi, center * 1e6),
                color="0.4",
                lw=0.8,
                ls=ls,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(3, 1e4)
        ax.set_xlabel(r"delay  [$\mu$s]")
        ax.set_title(f"{center:g} MHz", fontsize=10)
    axes[0].set_ylabel("time-averaged power")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        r"Delay profiles; vertical: $\tau_{\rm FD}$ for "
        r"$\phi = 30$ (dotted) and $2400$ rad/m$^2$ (dashed)",
        y=1.02,
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"pspec_delay_profile.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("  wrote pspec_delay_profile", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", type=float, nargs="+", default=[30.0])
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for center in args.centers:
        tag = f"real{center:g}"
        print(f"computing power spectra for {center:g} MHz", flush=True)
        results, meta = compute(center)
        np.savez(
            GEN_DIR / f"{tag}_pspec.npz",
            **{
                f"{k}_{n}": v
                for k, (ps, ft, tau) in results.items()
                for n, v in (("ps", ps), ("ftime_hz", ft), ("tau_us", tau))
            },
        )
        fig_pspec(center, results)
        all_results[center] = results
    if len(all_results) > 1:
        fig_delay_profile(all_results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
