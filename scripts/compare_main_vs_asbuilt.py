"""main (enforced-symmetry 2-port) vs as-built (4-port BGL_v16).

Two independent arms, fed identical point-source Stokes vectors:

  main     : hfss_lbl_3m_75deg.2port.fits at 30 MHz, Y = X rolled 90 deg
             in phi, no impedance, no receiver loading.  Pseudo-Stokes
             per src/lusee_faraday/sim.py: I=Rxx+Ryy, Q=Rxx-Ryy,
             U=2Re(Rxy).  This is the pipeline behind the paper's Fig 4.
  as-built : lusee_bgl_v16_response_v3.fits at 30 MHz, four independent
             N/E/S/W ports, full 4x4 Z_A with mutual coupling, JFET
             receiver loading M = Z_L(Z_A+Z_L)^-1.  Naive polarimeter
             X = E-W, Y = N-S.

Both arms are normalised by their own I response to an *unpolarized*
source in the same direction, which is exactly what main's
`norm = sum(wI_x*m)+sum(wI_y*m)` does when only the source pixel is
unmasked.  That makes the two arms dimensionless and comparable
(main is unitless, as-built is V^2/Hz).

The source polarization is specified in the local (e_theta, e_phi)
tangent basis at the source position, identically for both arms.
"""

import argparse

import numpy as np
from scipy.interpolate import RectSphereBivariateSpline

from common import DATA_DIR, FIG_DIR, RESPONSE_DIR

C_LIGHT = 299792458.0
MAIN_BEAM = DATA_DIR / "hfss_lbl_3m_75deg.2port.fits"
ASBUILT = RESPONSE_DIR / "lusee_bgl_v16_response_v3.fits"


# ----------------------------------------------------------------- main arm
class MainBeam:
    """main's 2-port beam, evaluable at arbitrary (theta, phi).

    Reproduces Beam.from_file(orientation="y") + precompute_weights()
    but interpolates straight to the requested directions instead of to
    HEALPix pixel centres, so the comparison is not limited by nside.
    Same interpolator family as HealpixGrid.interp_hp.
    """

    def __init__(self, path, frequency=30.0):
        from astropy.io import fits

        with fits.open(path) as f:
            Eth = f["Etheta_real"].data + 1j * f["Etheta_imag"].data
            Eph = f["Ephi_real"].data + 1j * f["Ephi_imag"].data
            ix = int(np.argwhere(f["freq"].data == frequency)[0, 0])
        # drop the duplicated phi=360 column, exactly as main does
        Eth, Eph = Eth[ix][..., :-1], Eph[ix][..., :-1]
        jones = np.array([Eth, Eph])                       # (2, 181, 360)
        # main appends a zero lower hemisphere before interpolating
        lower = np.zeros_like(jones)[:, :-1, :]
        jones = np.concatenate([jones, lower], axis=1)     # (2, 361, 360)
        # The file stores only the upper hemisphere (91 theta rows, 1 deg
        # step).  main pads a zeroed lower hemisphere to 181 rows and
        # interpolates on theta = linspace(0, 180, 181) -- reproduce that.
        assert jones.shape[1] == 181, jones.shape
        self._grid_theta = np.radians(np.linspace(0.0, 180.0, 181))
        self._grid_phi = np.radians(np.arange(360))
        # Interpolate the CARTESIAN components, not (Eth, Eph): the latter
        # carry a pure m=1 phase at the pole, so their azimuthal mean --
        # the spline's pole value -- vanishes and zeroes the beam at
        # zenith.  Mirrors beam.interp_jones on main.
        th, ph = np.meshgrid(self._grid_theta, self._grid_phi, indexing="ij")
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        Eth, Eph = jones
        cart = np.array(
            [
                Eth * ct * cp - Eph * sp,
                Eth * ct * sp + Eph * cp,
                -Eth * st,
            ]
        )
        self._splines = self._build(cart)

    def _build(self, jones):
        out = []
        th, ph = self._grid_theta, self._grid_phi
        # strip the pole row(s) the way interp_hp does
        for comp in jones:                                  # Ex, Ey, Ez
            per_part = []
            for part in (comp.real, comp.imag):
                t, a = th, part
                pole = [None, None]
                if np.isclose(t[0], 0.0):
                    pole[0] = a[0, :].mean()
                    t, a = t[1:], a[1:, :]
                if np.isclose(t[-1], np.pi):
                    pole[1] = a[-1, :].mean()
                    t, a = t[:-1], a[:-1, :]
                keep = t <= np.pi
                per_part.append(
                    RectSphereBivariateSpline(
                        t[keep], ph, a[keep], pole_values=tuple(pole)
                    )
                )
            out.append(per_part)
        return out

    def _eval(self, theta, phi):
        """Jones (Eth, Eph) of the file's dipole at (theta, phi).

        Rotating the antenna by alpha about z translates the tangent-basis
        components in phi (main's `rotate_beam`), because e_theta and
        e_phi rotate with the direction.  So evaluating here at a shifted
        phi and projecting onto the tangent basis *at that shifted
        direction* reproduces the roll exactly.
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi) % (2 * np.pi)
        cart = np.array(
            [
                re_sp(theta, phi, grid=False)
                + 1j * im_sp(theta, phi, grid=False)
                for re_sp, im_sp in self._splines
            ]
        )                                                   # (3, N)
        st, ct, sp, cp = (
            np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        )
        Eth = cart[0] * ct * cp + cart[1] * ct * sp - cart[2] * st
        Eph = -cart[0] * sp + cart[1] * cp
        return np.array([Eth, Eph])                         # (2, N)

    def weights(self, theta, phi):
        """main's 9 Stokes weights at (theta, phi)."""
        # orientation="y": jones_y is the file, jones_x is it rolled 270 deg
        jy = self._eval(theta, phi)
        jx = self._eval(theta, np.asarray(phi) - np.radians(270.0))
        (Ex_th, Ex_ph), (Ey_th, Ey_ph) = jx, jy
        w = {}
        w["wI_x"] = 0.5 * (abs(Ex_th) ** 2 + abs(Ex_ph) ** 2)
        w["wQ_x"] = 0.5 * (abs(Ex_th) ** 2 - abs(Ex_ph) ** 2)
        w["wU_x"] = np.real(Ex_th * np.conj(Ex_ph))
        w["wI_y"] = 0.5 * (abs(Ey_th) ** 2 + abs(Ey_ph) ** 2)
        w["wQ_y"] = 0.5 * (abs(Ey_th) ** 2 - abs(Ey_ph) ** 2)
        w["wU_y"] = np.real(Ey_th * np.conj(Ey_ph))
        w["wI_xy"] = 0.5 * (Ex_th * np.conj(Ey_th) + Ex_ph * np.conj(Ey_ph))
        w["wQ_xy"] = 0.5 * (Ex_th * np.conj(Ey_th) - Ex_ph * np.conj(Ey_ph))
        w["wU_xy"] = 0.5 * (Ex_th * np.conj(Ey_ph) + Ex_ph * np.conj(Ey_th))
        return w

    def pstokes(self, theta, phi, I, Q, U):
        """Normalised pseudo-Stokes (pI, pQ, pU); shapes broadcast."""
        w = self.weights(theta, phi)
        Rxx = w["wI_x"] * I + w["wQ_x"] * Q + w["wU_x"] * U
        Ryy = w["wI_y"] * I + w["wQ_y"] * Q + w["wU_y"] * U
        Rxy = w["wI_xy"] * I + w["wQ_xy"] * Q + w["wU_xy"] * U
        norm = w["wI_x"] + w["wI_y"]
        return (Rxx + Ryy) / norm, (Rxx - Ryy) / norm, 2 * np.real(Rxy) / norm


# ------------------------------------------------------------- as-built arm
class AsBuilt:
    def __init__(self, path, frequency=30.0):
        from lusee.ReceiverImpedance import JFETReceiver
        from lusee_faraday import fourport as fp

        self.fp = fp
        resp = fp.load_response_fast(str(path))
        self.kern = fp.FixedFreqKernel(resp, frequency, JFETReceiver())
        del resp

    def pstokes(self, theta, phi, I, Q, U):
        """Normalised pseudo-Stokes (pI, pQ, pU, pV) via X=E-W, Y=N-S."""
        fp = self.fp
        K = self.kern.sample(np.atleast_1d(theta), np.atleast_1d(phi))
        pref = self.kern.prefac
        # kernel contraction: K is (10, 4, N)
        KI, KQ, KU = K[:, 0, :], K[:, 1, :], K[:, 2, :]
        pair = pref * (
            KI * np.asarray(I) + KQ * np.asarray(Q) + KU * np.asarray(U)
        )                                                   # (10, N)
        C = fp.assemble_covariance(np.moveaxis(pair, 0, -1), self.kern.M)
        out = fp.polarimeter(C)                             # (N, 4)
        pair0 = pref * KI * np.ones_like(np.asarray(I, dtype=float))
        C0 = fp.assemble_covariance(np.moveaxis(pair0, 0, -1), self.kern.M)
        norm = fp.polarimeter(C0)[..., 0]
        return (
            out[..., 0] / norm,
            out[..., 1] / norm,
            out[..., 2] / norm,
            out[..., 3] / norm,
        )


# ------------------------------------------------------------------ figures
def fig_leakage_vs_altitude(main, ab, az_deg=0.0):
    """Unpolarized source: spurious polarization vs altitude."""
    import matplotlib.pyplot as plt

    # main's beam is degenerate at exactly theta=0 (interp_hp averages
    # the pole row of E_theta/E_phi, which cancels), and suppressed
    # below ~1 deg.  Stop at 88 deg so both arms are evaluated where
    # main's interpolation is valid.
    alt = np.linspace(3.0, 90.0, 400)
    th = np.radians(90.0 - alt)
    ph = np.full_like(th, np.radians(az_deg))
    one = np.ones_like(th)
    zero = np.zeros_like(th)

    mI, mQ, mU = main.pstokes(th, ph, one, zero, zero)
    aI, aQ, aU, aV = ab.pstokes(th, ph, one, zero, zero)
    p_main = np.hypot(mQ, mU) / mI
    p_ab = np.hypot(aQ, aU) / aI

    plt.figure(figsize=(6, 4))
    plt.plot(alt, p_main, label="symmetric pseudo-dipoles")
    plt.plot(alt, p_ab, label="4-port BGL v16")
    plt.xlabel("source altitude [deg]")
    plt.ylabel(r"spurious polarization  $\sqrt{Q^2+U^2}\,/\,I$")
    plt.xlim(0, 90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cmp_leakage_vs_altitude.png", dpi=150)
    plt.close()
    print(
        f"  leakage at zenith : main {p_main[-1]:.3e}   as-built {p_ab[-1]:.3e}"
    )
    print(
        f"  leakage max       : main {p_main.max():.3f} @ {alt[p_main.argmax()]:.0f} deg"
        f"   as-built {p_ab.max():.3f} @ {alt[p_ab.argmax()]:.0f} deg"
    )
    return p_main, p_ab


def fig_leakage_vs_azimuth(main, ab, alt_deg=30.0):
    import matplotlib.pyplot as plt

    az = np.linspace(0, 360, 721)
    th = np.full_like(az, np.radians(90.0 - alt_deg))
    ph = np.radians(az)
    one, zero = np.ones_like(th), np.zeros_like(th)

    mI, mQ, mU = main.pstokes(th, ph, one, zero, zero)
    aI, aQ, aU, aV = ab.pstokes(th, ph, one, zero, zero)
    p_main = np.hypot(mQ, mU) / mI
    p_ab = np.hypot(aQ, aU) / aI

    plt.figure(figsize=(6, 4))
    plt.plot(az, p_main, label="symmetric pseudo-dipoles")
    plt.plot(az, p_ab, label="4-port BGL v16")
    plt.xlabel(f"source azimuth [deg]  (altitude {alt_deg:g} deg)")
    plt.ylabel(r"spurious polarization  $\sqrt{Q^2+U^2}\,/\,I$")
    plt.xlim(0, 360)
    plt.xticks([0, 90, 180, 270, 360])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cmp_leakage_vs_azimuth.png", dpi=150)
    plt.close()
    for name, p in (("main", p_main), ("as-built", p_ab)):
        q = [p[np.argmin(np.abs(az - a))] for a in (45, 135, 225, 315)]
        print(
            f"  {name:9s} minima-region values at az 45/135/225/315: "
            + " ".join(f"{v:.4f}" for v in q)
            + f"  -> spread {(max(q)/min(q)-1)*100:.1f}%"
        )


def fig_spectrum(main, ab, phi_fd, fname, center=30.0, nfine=512):
    """Polarized source across one parent bin, with or without Faraday."""
    import matplotlib.pyplot as plt

    # show a quarter of the 25 kHz parent bin: enough Faraday cycles to
    # read the period, few enough to see the curves on a slide
    span_mhz = 25e-3 / 4
    freqs = center + (np.arange(nfine) - nfine // 2) * (span_mhz / nfine)
    lam2 = (C_LIGHT / (freqs * 1e6)) ** 2
    # intrinsic source: 100% polarized along local e_theta -> (I,Q,U)=(1,-1,0)
    P = (-1.0 + 0.0j) * np.exp(2j * phi_fd * lam2)
    I = np.ones_like(freqs)
    Q, U = P.real, P.imag

    tag = (rf"$\phi_{{\rm FD}} = {phi_fd:g}$ rad m$^{{-2}}$" if phi_fd
           else "no Faraday rotation")
    cases = [("zenith", 90.0, 0.0), (r"alt $60^\circ$, az $20^\circ$", 60.0, 20.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (label, alt, az) in zip(axes, cases):
        th = np.full_like(freqs, np.radians(90.0 - alt))
        ph = np.full_like(freqs, np.radians(az))
        mI, mQ, mU = main.pstokes(th, ph, I, Q, U)
        aI, aQ, aU, aV = ab.pstokes(th, ph, I, Q, U)
        foff = (freqs - center) * 1e3
        for arr, c in ((mI, "C0"), (mQ, "C1"), (mU, "C2")):
            ax.plot(foff, arr, color=c, lw=1.5)
        for arr, c in ((aI, "C0"), (aQ, "C1"), (aU, "C2")):
            ax.plot(foff, arr, color=c, lw=1.5, ls="--")
        ax.set_xlabel(f"frequency offset from {center:g} MHz [kHz]")
        ax.set_ylabel("normalised pseudo-Stokes")
        ax.set_title(f"{label} — {tag}", fontsize=10)
        # pointwise fractional polarization; PSD-ness of J T J^H bounds
        # sqrt(Q^2+U^2+V^2) <= I, so these must not exceed 1.
        p_m = np.hypot(mQ, mU) / mI
        p_a = np.sqrt(aQ**2 + aU**2 + aV**2) / aI
        print(
            f"  {label:26s} main   p in [{p_m.min():.3f}, {p_m.max():.3f}]"
            f"   as-built p in [{p_a.min():.3f}, {p_a.max():.3f}]"
        )
        for nm, pp in (("main", p_m), ("as-built", p_a)):
            if pp.max() > 1.0 + 1e-3:
                print(f"    WARNING {nm}: p = {pp.max():.4f} > 1 -> not PSD")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="C0", label="$I$"),
        Line2D([], [], color="C1", label="$Q$"),
        Line2D([], [], color="C2", label="$U$"),
        Line2D([], [], color="0.35", ls="-", label="symmetric pseudo-dipoles"),
        Line2D([], [], color="0.35", ls="--", label="4-port BGL v16"),
    ]
    axes[0].legend(handles=handles, ncol=2, fontsize=8, loc="lower center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=30.0)
    ap.add_argument("--phi-fd", type=float, default=250.0)
    args = ap.parse_args()

    print("loading main 2-port beam ...", flush=True)
    main = MainBeam(MAIN_BEAM, args.freq)
    print("loading as-built four-port response ...", flush=True)
    ab = AsBuilt(ASBUILT, args.freq)

    print("\n[1] unpolarized source, leakage vs altitude")
    fig_leakage_vs_altitude(main, ab)
    print("\n[2] unpolarized source, leakage vs azimuth")
    fig_leakage_vs_azimuth(main, ab)
    print(f"\n[3] polarized source, phi_FD = {args.phi_fd:g} rad/m^2")
    fig_spectrum(main, ab, args.phi_fd, "cmp_faraday_spectrum.png", args.freq)
    print("\n[4] same source, Faraday OFF (control)")
    fig_spectrum(main, ab, 0.0, "cmp_faraday_off.png", args.freq)
    print(f"\nfigures -> {FIG_DIR}")
