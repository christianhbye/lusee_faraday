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
    # explicit dphi: the wide lambda^2 coverage gives ~0.001 rad/m^2 RMSF
    # resolution, so the default oversampled grid would be ~300k points
    # (20 GB kernel). The beam-averaged signal is spread over ~tens of
    # rad/m^2, so 0.02 rad/m^2 samples it finely.
    phi = rmsynth.phi_grid(lam2, phi_max=50.0, dphi=0.02)

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

    # Two distinct questions:
    #  SNR_pol = detect the polarized signal at all (peak |F| of FR). This
    #            is large -- LuSEE easily sees the polarized sky.
    #  SNR_far = detect the FARADAY EFFECT, i.e. the part of the signal
    #            caused by rotation: RM synthesis of (P_FR - P_noFR). This
    #            is the forecast Faraday detectability (assumes a perfect
    #            unrotated-sky model, so it is an optimistic upper bound).
    print("  SNR vs integration time (galaxy-up):")
    dQ = pQ_FR[i_gal] - pQ_nf[i_gal]
    dU = pU_FR[i_gal] - pU_nf[i_gal]
    p_nf = np.hypot(pQ_nf[i_gal], pU_nf[i_gal])
    snr_table = {}
    for dt, label in DT_CASES:
        sig = noise.radiometer_sigma(Tsys, dnu, dt)
        w_iv = 1.0 / sig ** 2
        w_sa = p_nf / sig ** 2  # signal-aware
        snr_pol = detection.faraday_snr(
            pQ_FR[i_gal], pU_FR[i_gal], lam2, sig, phi, w_iv)[0]
        snr_far_iv = detection.faraday_snr(dQ, dU, lam2, sig, phi, w_iv)[0]
        snr_far_sa = detection.faraday_snr(dQ, dU, lam2, sig, phi, w_sa)[0]
        snr_table[label] = (snr_pol, snr_far_iv, snr_far_sa)
        print(f"    {label:>7}: SNR_pol={snr_pol:8.1f}  "
              f"SNR_far(invvar)={snr_far_iv:7.1f}  "
              f"SNR_far(sigaware)={snr_far_sa:7.1f}")

    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    ax[0].plot(phi, np.abs(F_fr), label="FR")
    ax[0].plot(phi, np.abs(F_nf), label="no FR", ls="--")
    ax[0].set(title=f"Faraday spectrum (galaxy-up t={i_gal})",
              xlabel="phi [rad/m^2]", ylabel="|F|", yscale="log")
    ax[0].legend()

    labels = [l for _, l in DT_CASES]
    far_iv = [snr_table[l][1] for l in labels]
    far_sa = [snr_table[l][2] for l in labels]
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, far_iv, 0.4, label="inverse-variance")
    ax[1].bar(x + 0.2, far_sa, 0.4, label="signal-aware")
    ax[1].axhline(5, color="k", ls=":", lw=1, label="SNR=5")
    ax[1].set(
        title="Faraday-effect SNR (RM synth of FR - noFR) vs integration",
        ylabel="SNR", xticks=x, xticklabels=labels,
    )
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
