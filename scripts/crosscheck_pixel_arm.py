"""Harmonic four-port path vs the pixel-space engine in fourport.py.

This script CHARACTERIZES the disagreement between the harmonic
contraction (``engine.contract``) and the legacy pixel-space engine
(``fourport.py``) on the real BGL_v16 response.  It is not a
correctness gate -- the gate is ``tests/test_engine_gate.py``, which
shows the harmonic contraction reproduces luseepy's own convolution
to round-off (6.8e-16) on a synthetic response and a rotation-
sensitive sky.

Why that gate hits round-off and this script does not, even though
both exercise the same harmonic beam-alm code: our beam alms
(``response.four_port_pair_alms``) and luseepy's own
(``FullStokesSimulatorBase.prepare_pair_alms``) both route through
``lusee.InstrumentResponse.pair_stokes_alms_native``, which builds a
``croissant.PairStokesBeam`` with an explicit horizon mask
(``theta <= 90 deg``) and calls its ``compute_alm`` -- identical
code, so the gate is a same-library comparison.  The BGL_v16 response
itself is stored only over the upper hemisphere (theta in [0, 90]
deg, confirmed against the FITS ``theta`` HDU) and is zero-padded to
the full sphere before that transform, so the beam fed into
``compute_alm`` has a genuine step discontinuity at the horizon: real
antenna gain up to 90 deg, then exactly zero.  A step function is not
band-limited, so truncating its spherical-harmonic expansion at
finite lmax discards real power -- Gibbs ringing at the horizon.  The
pixel arm has no such truncation; it samples the response's native
grid directly.  So the two arms' beams differ near the horizon by an
amount set by lmax, not by pixel resolution.

That mechanism is consistent with everything measured so far: the
disagreement is flat in the pixel arm's sky-quadrature resolution
(2.103e-2 / 2.105e-2 / 2.106e-2 at nside 32/64/128, sky held fixed in
harmonic space, lmax=30 -- a 4x range in nside moves the result by
about 0.1% relative, i.e. noise); it moves only weakly with lmax
(6.49e-2 at lmax=30 vs 6.30e-2 at lmax=48 in one differently-seeded
run pair -- consistent with a step spectrum's ~1/l decay); and it is
worse on cross-polarization pairs than on autos, where the beam's
near-horizon structure matters most. Because our harmonic path uses
the identical masked-transform code luseepy's own production
simulator uses, this script's residual reflects a pre-existing
difference between luseepy's harmonic engine and the legacy pixel
engine, not a regression from this refactor.

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
# Two configurations, both reproducible (seed 7, fixed below).
# Measured 2026-08-18:
#   nside=64, lmax=30 (default here):                    2.678e-02
#   nside=32, lmax=48 (validate_engine.py's own config):  3.767e-02
# validate_engine.py's diffuse-sky check records 1.1e-2 for a related
# but not identical comparison -- see the module docstring for why
# this script's numbers run higher (mask-induced Gibbs error at the
# horizon, not round-off).  Override with LUSEE_CROSSCHECK_NSIDE /
# LUSEE_CROSSCHECK_LMAX to reproduce either configuration above.
NSIDE = int(os.environ.get("LUSEE_CROSSCHECK_NSIDE", "64"))
LMAX = int(os.environ.get("LUSEE_CROSSCHECK_LMAX", "30"))
N_TIMES = 4

# Empirical expectation band, NOT a correctness bound: both measured
# configurations above (2.678e-2, 3.767e-2) fall inside it with
# margin.  Exists to catch a gross change in this comparison -- e.g.
# a broken convention pushing the disagreement far outside what's
# been measured -- not to assert that either arm is correct.
# Correctness is tests/test_engine_gate.py's job.  Recorded 2026-08-18.
EXPECTED_LOW = 1e-2
EXPECTED_HIGH = 8e-2


def band_limited_sky(rng, nside, lmax):
    """A smooth IQUV sky the beam's band-limit can actually represent.

    ``hp.synalm`` takes no seed of its own and draws from numpy's
    legacy global RNG state, so each channel reseeds that state from
    ``rng`` -- the only source of randomness threaded through this
    script -- to make the sky (and hence the reported disagreement)
    reproducible run to run.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.empty((4, npix))
    for i in range(4):
        np.random.seed(rng.integers(0, 2**32 - 1))
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
    band = f"[{EXPECTED_LOW:.1e}, {EXPECTED_HIGH:.1e}]"
    if EXPECTED_LOW <= worst <= EXPECTED_HIGH:
        print(f"within the recorded empirical band {band}")
    else:
        print(
            f"OUTSIDE the recorded empirical band {band} -- "
            "this is a change from what's been measured; "
            "investigate before trusting this comparison"
        )


if __name__ == "__main__":
    main()
