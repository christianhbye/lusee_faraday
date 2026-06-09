"""Step-0 calibration: RM synthesis on existing 3-band zoom spectra.

Combines the 64 zoom sub-bins of the 10/30/50 MHz sims into one
192-channel comb and runs RM synthesis. The RMSF reveals the sidelobe
structure of this sparse lambda^2 sampling, which sizes the adaptive
grid for the full-band sim.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lusee_faraday import rmsynth

RES = Path(__file__).resolve().parent / "results"
BANDS = [10, 30, 50]
PHI_MAX = 100.0


def load_comb():
    freqs, Q, U, Qn, Un = [], [], [], [], []
    i_gal = None
    for cf in BANDS:
        d = np.load(RES / f"faraday_sim_{cf}mhz.npz")
        freqs.append(d["freqs_zoom"])
        Q.append(d["pQ_FR_zoom"])
        U.append(d["pU_FR_zoom"])
        Qn.append(d["pQ_noFR_zoom"])
        Un.append(d["pU_noFR_zoom"])
        i_gal = int(d["i_gal"])
    freqs = np.concatenate(freqs)
    Q = np.concatenate(Q, axis=1)
    U = np.concatenate(U, axis=1)
    Qn = np.concatenate(Qn, axis=1)
    Un = np.concatenate(Un, axis=1)
    return freqs, Q, U, Qn, Un, i_gal


def main():
    freqs, Q, U, Qn, Un, i_gal = load_comb()
    lam2 = rmsynth.lambda2(freqs)
    phi = rmsynth.phi_grid(lam2, phi_max=PHI_MAX)

    res = rmsynth.faraday_resolution(lam2)
    scale = rmsynth.max_scale(lam2)
    print(f"channels: {lam2.size}")
    print(f"lambda^2 span: {lam2.min():.2f} .. {lam2.max():.2f} m^2")
    print(f"RMSF resolution (FWHM): {res:.4f} rad/m^2")
    print(f"max recoverable scale:  {scale:.4f} rad/m^2")
    print(f"phi grid: {phi.size} points over +/-{PHI_MAX}")

    R = rmsynth.rmsf(lam2, phi)
    F = rmsynth.faraday_spectrum(Q, U, lam2, phi)
    Fn = rmsynth.faraday_spectrum(Qn, Un, lam2, phi)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].plot(phi, np.abs(R))
    ax[0].set(title="RMSF |R(phi)|", xlabel="phi [rad/m^2]", yscale="log")

    ax[1].plot(phi, np.abs(F[i_gal]), label="FR")
    ax[1].plot(phi, np.abs(Fn[i_gal]), label="no FR", ls="--")
    ax[1].set(title=f"Faraday spectrum, galaxy-up (t={i_gal})",
              xlabel="phi [rad/m^2]", yscale="log")
    ax[1].legend()

    im = ax[2].imshow(
        np.abs(F), aspect="auto", origin="lower",
        extent=[phi[0], phi[-1], 0, F.shape[0]],
    )
    ax[2].set(title="|F(phi, t)|", xlabel="phi [rad/m^2]", ylabel="time index")
    fig.colorbar(im, ax=ax[2])

    fig.tight_layout()
    out = RES / "rmsynth_calibration.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
