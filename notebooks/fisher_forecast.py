# notebooks/fisher_forecast.py
"""Step-4 Fisher forecast: sky-marginalized Faraday detection SNR.

Reuses results/faraday_fullband.npz (channel nu/lambda2/dnu, pI_FR as
T_sys) and the full-band sim setup. Rotates the WMAP fiducial sky and a
low-l spin-2 nuisance basis to topocentric, builds the Fisher matrix
over (alpha, tau, sky modes), and reports the marginalized detection SNR
vs the fixed-sky optimistic bound across integration times.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import astropy.units as u
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lunarsky import Time, MoonLocation

import lusee_faraday as ld
from lusee_faraday.fast_sim import precompute_rotated_maps
from lusee_faraday.forward import rotate_pol_maps
from lusee_faraday.fisher import run_forecast
from lusee_faraday.skybasis import spin2_basis
from lusee_faraday.noise import radiometer_sigma
from lusee_faraday.sky import LUSEE_LOC

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
NSIDE = 64  # forecast forward-model resolution (knob; 1deg pixels
# resolve the beam + low-l modes; bump to 128 for a
# slower precision run)
N_TIMES = 100
LMAX = 3  # spin-2 sky-nuisance bandlimit (24 modes)
DECIM = 8  # forecast channel decimation; DECIM=1 = full 4047
DT_CASES = [(40.0, "40 s"), (600.0, "10 min"), (3600.0, "1 h")]
BEAM_FILE = DATA / "hfss_lbl_3m_75deg.2port.fits"


def main():
    d = np.load(RES / "faraday_fullband.npz")
    # Decimate channels for the forecast: ~28 channel-center pol_response
    # evals over all 4047 channels is too slow; a strided subset spans the
    # same lambda^2 range and keeps the forecast tractable. DECIM=1 runs
    # the full channel set.
    sl = slice(None, None, DECIM)
    nu, lam2, dnu = d["nu"][sl], d["lambda2"][sl], d["dnu"][sl]
    pI_FR = d["pI_FR"][:N_TIMES, sl]  # (ntimes, nchan), used as T_sys

    loc = MoonLocation(lat=-23.813, lon=182.258)
    t0 = Time("2027-01-01T09:00:00", location=loc)
    times = np.linspace(
        t0, t0 + 655.720 * 3600 * u.s, num=N_TIMES, endpoint=False
    )

    I_ref = hp.ud_grade(np.load(DATA / "haslam_galactic.npz")["m"], NSIDE)
    wmap = ld.sky.load_wmap(
        DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits", nside=NSIDE
    )
    Q_ref, U_ref = wmap[1], wmap[2]
    rm_gal = ld.sky.load_rm(DATA / "faraday2020v2.hdf5", nside=NSIDE)

    beam = ld.Beam.from_file(BEAM_FILE, frequency=30, nside=NSIDE)
    beam.precompute_weights()
    mask = ld.HealpixGrid(NSIDE, horizon=True).mask

    print(f"rotating fiducial sky ({N_TIMES} steps)...")
    I_t, Q_t, U_t, rm_t = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm_gal, times, NSIDE, LUSEE_LOC
    )

    basis = spin2_basis(NSIDE, LMAX)
    print(f"rotating {len(basis)} basis modes (lmax={LMAX})...")
    basis_topo = []
    for i, (label, Qb, Ub) in enumerate(basis):
        Qb_t, Ub_t, _ = rotate_pol_maps(
            Qb, Ub, rm_gal, times, NSIDE, LUSEE_LOC
        )
        basis_topo.append((Qb_t, Ub_t))
        print(f"  mode {i + 1}/{len(basis)} ({label})")

    # The Jacobian is integration-time-independent, and radiometer noise
    # gives sigma ~ 1/sqrt(dt), so the Fisher matrix scales exactly as dt
    # and sigma(alpha) ~ 1/sqrt(dt). Run the forecast ONCE at a reference
    # time and scale the SNR by sqrt(dt/dt_ref) -- no need to rebuild the
    # (expensive) Jacobian per integration time.
    dt_ref = DT_CASES[-1][0]
    sigma_ref = radiometer_sigma(pI_FR, dnu, dt_ref)
    print(
        f"forecast at dt_ref={dt_ref:.0f}s "
        f"(alpha marginalized over sky + tau)..."
    )
    out = run_forecast(
        I_t, Q_t, U_t, rm_t, basis_topo, beam, mask, nu, lam2, sigma_ref
    )
    print(
        f"  sigma(alpha)={out['sigma_alpha']:.3e}  "
        f"n_modes={out['n_modes']}"
    )
    snr = [out["snr"] * np.sqrt(dt / dt_ref) for dt, _ in DT_CASES]
    snr_opt = [out["snr_opt"] * np.sqrt(dt / dt_ref) for dt, _ in DT_CASES]
    for (dt, lbl), s, so in zip(DT_CASES, snr, snr_opt):
        print(
            f"  {lbl:>7}: SNR(marginalized)={s:8.2f}  "
            f"SNR(fixed-sky)={so:8.2f}"
        )

    labels = [l for _, l in DT_CASES]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, snr_opt, 0.4, label="fixed sky (optimistic)")
    ax.bar(x + 0.2, snr, 0.4, label="sky+tau marginalized")
    ax.axhline(5, color="k", ls=":", lw=1, label="SNR=5")
    ax.set(
        title="Faraday-amplitude detection SNR vs integration",
        ylabel="SNR",
        xticks=x,
        xticklabels=labels,
        yscale="log",
    )
    ax.legend()
    fig.tight_layout()
    out_png = RES / "fisher_forecast.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
