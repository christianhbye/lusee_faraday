# notebooks/faraday_fullband_sim.py
"""Curated full-band Faraday simulation.

Rotates the sky once per time step, runs the channelized FR + no-FR
pipeline on the full-band FrequencyPlan, tags each step with its
Galactic->topocentric Euler angles (LST), and saves one npz for
Step-3 RM synthesis.
"""

import os

# Pin each process to a single BLAS/OpenMP thread BEFORE numpy imports.
# This driver parallelizes over time with one process per physical core;
# per-process BLAS threading would oversubscribe the cores and thrash.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time as pytime
from pathlib import Path

import astropy.units as u
import numpy as np
from lunarsky import Time, MoonLocation

import lusee_faraday as ld
from lusee_faraday import SpectrometerResponse, FrequencyPlan
from lusee_faraday.fast_sim import precompute_rotated_maps
from lusee_faraday.pipeline import simulate_channelized
from lusee_faraday.rotations import topo_euler_angles
from lusee_faraday.sky import LUSEE_LOC
from grid_design import fullband_specs, DECIMATION, SUPPORT

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
NSIDE = 128
N_TIMES = 100
# one single-threaded process per physical core (2 logical/core here)
NPROC = max(1, (os.cpu_count() or 2) // 2)
BEAM_FILE = DATA / "hfss_lbl_3m_75deg.2port.fits"


def main():
    loc = MoonLocation(lat=-23.813, lon=182.258)
    t0 = Time("2027-01-01T09:00:00", location=loc)
    times = np.linspace(t0, t0 + 655.720 * 3600 * u.s, num=N_TIMES,
                        endpoint=False)

    I_ref = np.load(DATA / "haslam_galactic.npz")["m"]
    wmap = ld.sky.load_wmap(DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits",
                            nside=NSIDE)
    Q_ref, U_ref = wmap[1], wmap[2]
    rm_gal = ld.sky.load_rm(DATA / "faraday2020v2.hdf5")

    beam = ld.Beam.from_file(BEAM_FILE, frequency=30, nside=NSIDE)
    beam.precompute_weights()
    mask = ld.HealpixGrid(NSIDE, horizon=True).mask

    spec = SpectrometerResponse.from_file(DATA / "spectrometer_bin_response.txt")
    plan = FrequencyPlan(spec, fullband_specs(), decimation=DECIMATION,
                         support=SUPPORT)

    print(f"rotating {N_TIMES} time steps...")
    t = pytime.time()
    I_t, Q_t, U_t, rm_t = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm_gal, times, NSIDE, LUSEE_LOC)
    print(f"  rotations done in {(pytime.time()-t)/60:.1f} min")

    print(f"simulating ({plan.sim_freqs().size} freqs, nproc={NPROC})...")
    t = pytime.time()
    out, table = simulate_channelized(
        plan, I_t, Q_t, U_t, rm_t, beam, mask, nproc=NPROC)
    print(f"  sim done in {(pytime.time()-t)/60:.1f} min")

    euler = topo_euler_angles(times, LUSEE_LOC)
    modes = np.array([m for _, m in plan.specs])
    times_jd = np.array([tt.jd for tt in times])

    outfile = RES / "faraday_fullband.npz"
    np.savez(
        outfile,
        nu=table["nu"], lambda2=table["lambda2"], dnu=table["dnu"],
        modes=modes, times_jd=times_jd, euler=euler,
        **out,
    )
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
