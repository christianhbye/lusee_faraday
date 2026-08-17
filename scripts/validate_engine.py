"""Validate the pixel-space four-port pipeline against luseepy engines.

Three checks:
1. Point-source kernel: FixedFreqKernel.sample + assemble_covariance vs
   lusee.FullStokesCalibratorSimulator on a response sliced to 30 MHz.
2. Diffuse polarized sky (no Faraday): SkyWaterfallSim vs
   FullStokesCroSimulator on a smooth band-limited IQUV sky, several
   time stamps.  This pins the galactic->topo transport, the alpha spin
   rotation, the COSMO/IAU U-sign and the overall normalization.
3. Faraday synthesis: NUFFT evaluation at one frequency vs running the
   no-Faraday path on externally rotated maps (internal exactness), and
   vs the harmonic engine on the same rotated maps.
"""

import os
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

from common import (  # noqa: E402
    RESPONSE_PATH,
    lam2,
    moon_location,
    times,
)
from lusee_faraday import fourport as fp  # noqa: E402


def slice_response(resp, keep):
    from lusee.InstrumentResponse import InstrumentResponse

    return InstrumentResponse.from_arrays(
        np.asarray(resp.freq)[keep],
        np.asarray(resp.theta_deg),
        np.asarray(resp.phi_deg),
        np.asarray(resp.H_theta)[:, keep],
        np.asarray(resp.H_phi)[:, keep],
        np.asarray(resp.ZA)[keep],
        np.asarray(resp.Rsky_native)[keep],
        np.asarray(resp.Rmoon_native)[keep],
        np.asarray(resp.Rloss_native)[keep],
        validated=False,
        metadata={**resp.header, "VALIDATED": False},
        ZLoad=np.asarray(resp.ZLoad)[keep],
    )


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    from lusee.ReceiverImpedance import JFETReceiver
    from lusee.FullStokesCalibrator import FullStokesCalibratorSimulator

    t0 = time.time()
    resp = fp.load_response_fast(RESPONSE_PATH)
    receiver = JFETReceiver()
    kern = fp.FixedFreqKernel(resp, 30.0, receiver)
    print(f"response + kernel loaded  {time.time()-t0:.1f} s", flush=True)

    # ---------------------------------------------- 1. point source
    keep = np.isclose(np.asarray(resp.freq), 30.0)
    small = slice_response(resp, keep)
    cal = FullStokesCalibratorSimulator(small, receiver, freq=np.array([30.0]))
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(4):
        th = rng.uniform(0.05, np.pi / 2 - 0.05)
        ph = rng.uniform(0, 2 * np.pi)
        stokes = rng.normal(size=4)
        stokes[0] = np.abs(stokes).sum()  # keep it physical
        cal.simulate(th, ph, stokes.astype(float))
        ref = np.asarray(cal.covariance)[0]
        K = kern.sample(np.array([th]), np.array([ph]))[:, :, 0]  # (10,4)
        pair = kern.prefac * (K @ stokes)
        got = fp.assemble_covariance(pair, kern.M)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        worst = max(worst, err)
    print(
        f"[1] point source vs CalibratorSimulator: {worst:.3e} relative",
        flush=True,
    )
    assert worst < 1e-8

    # ------------------------------------- 2. diffuse sky, no Faraday
    # croissant's dense spherical transform allocates
    # ~(lmax+1)^2 * npix * 16 bytes per spin; nside=128 / lmax=128 needs
    # ~50 GB and OOMs, and even nside=64 / lmax=64 crashed XLA under a
    # 24 GB address-space cap.  This check pins *conventions* (frame
    # rotation, alpha spin, U-sign, normalization), where any mistake is
    # O(1), so a small sky with a few-percent tolerance is sufficient.
    val_nside = 32
    lmax_sky = 32
    ell = np.arange(lmax_sky + 1)
    cl = 1.0 / (1.0 + ell) ** 2
    np.random.seed(11)
    I_s, Q_s, U_s = hp.synfast(
        [cl, 0.3 * cl, 0.3 * cl, 0.05 * cl],
        val_nside,
        new=True,
        pol=True,
    )
    I_s = I_s + 5.0 * np.abs(I_s).max()  # positive I, partial pol

    from lusee.FullStokesSimulator import FullStokesCroSimulator
    import croissant as cro
    import lusee

    sky_iau = np.stack([I_s, Q_s, -U_s, np.zeros_like(I_s)], axis=0)[
        None
    ]  # (1 freq, 4, npix); U flipped COSMO -> IAU
    psky = cro.PolarizedSky(
        sky_iau,
        np.array([30.0]),
        sampling="healpix",
        coord="galactic",
        convention="IAU",
        units="K",
    )
    obs = lusee.Observation(
        "2027-01-01 09:00:00 to 2027-01-28 09:00:00",
        deltaT_sec=86400.0 * 27.0 / 4,
        lun_lat_deg=-23.814,
        lun_long_deg=182.258,
    )
    tt = times()
    sel = [0, 256, 512, 768]
    test_times = tt[sel]
    lmax = 48
    # Use the 30 MHz single-channel slice: the full 150-channel response
    # drives the harmonic engine past 26 GB RSS and the OOM killer took
    # out whole sessions this way (journalctl 2026-08-16 17:46).
    sim = FullStokesCroSimulator(
        obs,
        small,
        psky,
        receiver,
        T_moon=0.0,
        freq=np.array([30.0]),
        lmax=lmax,
    )
    t1 = time.time()
    sim.simulate(times=test_times)
    ref_cov = np.asarray(sim.covariance)[:, 0]  # (T, 4, 4)
    print(f"engine ran in {time.time()-t1:.1f} s", flush=True)

    grid = fp.GalacticGrid(val_nside)
    sky = fp.SkyWaterfallSim(kern, grid, I_s, Q_s, U_s, np.zeros_like(I_s))
    from lusee_faraday.fourport import topo_rotation_matrix

    loc = moon_location()
    errs = []
    for i, t in enumerate(test_times):
        R = topo_rotation_matrix(t, loc)
        pair = sky.pair_integrals(R, lam2(30.0), faraday=False)[:, 0]
        got = fp.assemble_covariance(pair, kern.M)
        err = np.abs(got - ref_cov[i]).max() / np.abs(ref_cov[i]).max()
        errs.append(err)
        print(f"  t{i}: rel err {err:.3e}", flush=True)
    print(
        f"[2] diffuse sky vs CroSimulator: worst {max(errs):.3e}", flush=True
    )
    assert max(errs) < 8e-2  # nside=32 pixelization limits agreement

    # ------------------------------------------- 3. Faraday synthesis
    RM = hp.synfast(cl, val_nside) * 50.0
    sky_fr = fp.SkyWaterfallSim(kern, grid, I_s, Q_s, U_s, RM)
    f_test = 30.0 + 0.037  # arbitrary in-band frequency
    l2 = lam2(f_test)
    chi = RM * l2
    Q_r = Q_s * np.cos(2 * chi) - U_s * np.sin(2 * chi)
    U_r = Q_s * np.sin(2 * chi) + U_s * np.cos(2 * chi)
    sky_rot = fp.SkyWaterfallSim(kern, grid, I_s, Q_r, U_r, np.zeros_like(I_s))
    R = topo_rotation_matrix(test_times[1], loc)
    got = sky_fr.pair_integrals(R, np.array([l2]), faraday=True)[:, 0]
    ref = sky_rot.pair_integrals(R, np.array([l2]), faraday=False)[:, 0]
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"[3a] NUFFT vs rotated-maps (internal): {err:.3e}", flush=True)
    assert err < 1e-8

    psky_rot = cro.PolarizedSky(
        np.stack([I_s, Q_r, -U_r, np.zeros_like(I_s)], axis=0)[None],
        np.array([30.0]),
        sampling="healpix",
        coord="galactic",
        convention="IAU",
        units="K",
    )
    sim2 = FullStokesCroSimulator(
        obs,
        small,
        psky_rot,
        receiver,
        T_moon=0.0,
        freq=np.array([30.0]),
        lmax=lmax,
    )
    sim2.simulate(times=test_times[1:2])
    ref2 = np.asarray(sim2.covariance)[0, 0]
    got2 = fp.assemble_covariance(got, kern.M)
    err2 = np.abs(got2 - ref2).max() / np.abs(ref2).max()
    print(f"[3b] Faraday-rotated sky vs CroSimulator: {err2:.3e}", flush=True)
    assert err2 < 1.5e-1  # rotated maps are not band-limited; nside=32
    print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
