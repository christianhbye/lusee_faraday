import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import config as cfg  # noqa: E402
from lusee_faraday import engine, response as rsp  # noqa: E402

LMAX = 8
NSIDE = 16


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    cro = pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    resp = lusee.synthetic_four_port_response(
        freq_mhz=(10.0, 20.0), angular_step_deg=5.0
    )
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    data = rng.normal(size=(1, 4, npix))
    data[0, 0] = np.abs(data[0, 0]) + 5.0  # keep Stokes I positive
    sky = cro.PolarizedSky(
        data, np.array([10.0]), sampling="healpix", coord="galactic"
    )
    return lusee, resp, sky


def test_contract_matches_luseepy_full_stokes_convolve(pieces):
    lusee, resp, sky = pieces
    from lusee.FullStokesSimulator import (
        _sky_frame,
        prepare_polarized_sky_alms,
    )
    from lusee.ReceiverImpedance import JFETReceiver

    obs = lusee.Observation()
    times = cfg.times()[:5]
    sim = lusee.FullStokesCroSimulator(
        obs,
        resp,
        sky,
        JFETReceiver(),
        freq=np.array([10.0]),
        lmax=LMAX,
    )
    pair_alms, *_ = sim.prepare_pair_alms(resp)
    sky_alms = prepare_polarized_sky_alms(sky, sim.freq, sim.lmax)
    theirs = np.asarray(
        sim._convolve(pair_alms, sky_alms, _sky_frame(sky), times)
    )[
        :, 0, :
    ]  # (ntime, npair)

    beam = rsp.four_port_pair_alms(resp, 10.0, LMAX)
    components = np.asarray(sky.compute_alm(lmax=LMAX))
    W = engine.contract(beam, components, times, obs.loc, LMAX)
    ours = W.sum(axis=1)[0]  # (ntime, npair)

    scale = np.abs(theirs).max()
    assert np.abs(ours - theirs).max() < 1e-10 * scale
