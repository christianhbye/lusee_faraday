"""Harmonic four-port path vs the pixel-space engine in fourport.py.

Two independent quadratures of the same integral on the real BGL_v16
response.  Agreement is limited by the beam band-limit, not by
round-off, so the tolerance here is looser than the data-free gate in
tests/test_engine_gate.py.

Run:
    ulimit -v 16000000
    uv run python scripts/crosscheck_pixel_arm.py 2>&1 | tee \
        /home/christian/Documents/research/lusee/lusee_faraday/generated_data/crosscheck_pixel_arm.log
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from lusee_faraday import config as cfg  # noqa: E402
from lusee_faraday import engine, response as rsp  # noqa: E402
from lusee_faraday import fourport as fp  # noqa: E402

RESPONSE_PATH = os.environ.get(
    "LUSEE_RESPONSE",
    "data/BGL_v16/lusee_bgl_v16_response_v3.fits",
)
FREQ_MHZ = 30.0
NSIDE = 64
LMAX = 30
N_TIMES = 4


def band_limited_sky(rng, nside, lmax):
    """A smooth IQUV sky the beam's band-limit can actually represent."""
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.empty((4, npix))
    for i in range(4):
        alm = hp.synalm(
            np.exp(-np.arange(3 * nside) / 6.0), lmax=lmax, new=True
        )
        maps[i] = hp.alm2map(alm, nside, lmax=lmax)
    maps[0] = np.abs(maps[0]) + 10.0  # Stokes I positive
    maps[3] = 0.0  # no circular polarization
    return maps[None]


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import croissant as cro
    from lusee.ReceiverImpedance import JFETReceiver

    rng = np.random.default_rng(7)
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()

    data = band_limited_sky(rng, NSIDE, LMAX)
    # croissant wants IAU; the maps above are treated as COSMO.
    from lusee_faraday.conventions import cosmo_to_iau_qu

    data[0, 1], data[0, 2] = cosmo_to_iau_qu(data[0, 1], data[0, 2])
    sky = cro.PolarizedSky(
        data, np.array([FREQ_MHZ]), sampling="healpix", coord="galactic"
    )

    times = cfg.times()[:N_TIMES]
    beam = rsp.four_port_pair_alms(resp, FREQ_MHZ, LMAX)
    components = np.asarray(sky.compute_alm(lmax=LMAX))
    W = engine.contract(beam, components, times, cfg.moon_location(), LMAX)
    harmonic = W.sum(axis=1)[0]  # (ntime, npair)

    # Pixel arm: same maps, same response, no Faraday.
    kern = fp.FixedFreqKernel(resp, FREQ_MHZ, receiver)
    grid = fp.GalacticGrid(NSIDE)
    I_map = data[0, 0]
    Q_cos, U_cos = data[0, 1], -data[0, 2]  # back to COSMO for fourport
    sim = fp.SkyWaterfallSim(
        kern, grid, I_map, Q_cos, U_cos, np.zeros_like(I_map)
    )
    lam2 = cfg.lam2(np.array([FREQ_MHZ]))
    pixel = np.empty_like(harmonic)
    for i, t in enumerate(times):
        R = fp.topo_rotation_matrix(t, cfg.moon_location())
        pixel[i] = sim.pair_integrals(R, lam2, faraday=False)[:, 0]

    scale = np.abs(pixel).max()
    worst = np.abs(harmonic - pixel).max() / scale
    print(f"worst relative disagreement: {worst:.3e}")
    print("PASS" if worst < 2e-2 else "FAIL")


if __name__ == "__main__":
    main()
