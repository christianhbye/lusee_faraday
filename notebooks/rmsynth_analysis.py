# notebooks/rmsynth_analysis.py
"""Step-3 analysis: RM synthesis + Faraday detection significance.

Loads the full-band sim, runs model-independent RM synthesis on the FR
vs no-FR polarization, sweeps integration time for the detection SNR
(inverse-variance and signal-aware weighting), and saves figures.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lusee_faraday import rmsynth, noise, detection

RES = Path(__file__).resolve().parent / "results"
DT_CASES = [(40.0, "40 s"), (600.0, "10 min"), (3600.0, "1 h")]


def main():
    d = np.load(RES / "faraday_fullband.npz")
    nu, lam2, dnu = d["nu"], d["lambda2"], d["dnu"]
    pQ_FR, pU_FR = d["pQ_FR"], d["pU_FR"]
    pI_FR = d["pI_FR"]
    pQ_nf, pU_nf = d["pQ_noFR"], d["pU_noFR"]

    i_gal = int(np.argmax(pI_FR.mean(axis=1)))
    phi = rmsynth.phi_grid(lam2, phi_max=50.0)

    Tsys = pI_FR[i_gal]
    sig_ref = noise.radiometer_sigma(Tsys, dnu, 3600.0)
    w_iv = 1.0 / sig_ref ** 2
    F_fr = rmsynth.faraday_spectrum(pQ_FR[i_gal], pU_FR[i_gal], lam2, phi, w_iv)[0]
    F_nf = rmsynth.faraday_spectrum(pQ_nf[i_gal], pU_nf[i_gal], lam2, phi, w_iv)[0]

    print(f"galaxy-up t={i_gal}")
    print(f"  FR Faraday peak at phi={phi[np.argmax(np.abs(F_fr))]:.3f} rad/m^2")
    print(f"  noFR peak at phi={phi[np.argmax(np.abs(F_nf))]:.3f} rad/m^2")
    p_fr_all = np.hypot(pQ_FR, pU_FR)
    p_nf_all = np.hypot(pQ_nf, pU_nf)
    msk = p_nf_all > 0
    print(f"  median P_FR/P_noFR (all t,chan): "
          f"{np.median(p_fr_all[msk] / p_nf_all[msk]):.3f}")

    print("  SNR vs integration time (galaxy-up):")
    snr_table = {}
    for dt, label in DT_CASES:
        sig = noise.radiometer_sigma(Tsys, dnu, dt)
        w_iv = 1.0 / sig ** 2
        p_nf = np.hypot(pQ_nf[i_gal], pU_nf[i_gal])
        w_sa = p_nf / sig ** 2
        snr_iv = detection.faraday_snr(
            pQ_FR[i_gal], pU_FR[i_gal], lam2, sig, phi, w_iv)[0]
        snr_sa = detection.faraday_snr(
            pQ_FR[i_gal], pU_FR[i_gal], lam2, sig, phi, w_sa)[0]
        snr_table[label] = (snr_iv, snr_sa)
        print(f"    {label:>7}: SNR_invvar={snr_iv:6.1f}  "
              f"SNR_sigaware={snr_sa:6.1f}")

    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    ax[0].plot(phi, np.abs(F_fr), label="FR")
    ax[0].plot(phi, np.abs(F_nf), label="no FR", ls="--")
    ax[0].set(title=f"Faraday spectrum (galaxy-up t={i_gal})",
              xlabel="phi [rad/m^2]", ylabel="|F|", yscale="log")
    ax[0].legend()

    labels = [l for _, l in DT_CASES]
    iv = [snr_table[l][0] for l in labels]
    sa = [snr_table[l][1] for l in labels]
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, iv, 0.4, label="inverse-variance")
    ax[1].bar(x + 0.2, sa, 0.4, label="signal-aware")
    ax[1].axhline(5, color="k", ls=":", lw=1, label="SNR=5")
    ax[1].set(title="Faraday detection SNR vs integration time (galaxy-up)",
              ylabel="SNR", xticks=x, xticklabels=labels)
    ax[1].legend()

    ratio = np.divide(p_fr_all, p_nf_all,
                      out=np.full_like(p_fr_all, np.nan), where=p_nf_all > 0)
    ax[2].plot(nu, np.nanmedian(ratio, axis=0), ".", ms=2)
    ax[2].set(title="median depolarization P_FR/P_noFR vs frequency",
              xlabel="nu [MHz]", ylabel="P_FR/P_noFR")

    fig.tight_layout()
    out = RES / "rmsynth_analysis.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
