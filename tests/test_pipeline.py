import types
import numpy as np
from lusee_faraday import SpectrometerResponse, FrequencyPlan
from lusee_faraday.pipeline import simulate_channelized

SPEC_PATH = "data/spectrometer_bin_response.txt"


def _setup(rm_value):
    spec = SpectrometerResponse.from_file(SPEC_PATH)
    plan = FrequencyPlan(
        spec, [(30.0, "wide"), (10.0, "zoom")],
        decimation={"wide": 50, "zoom": 10}, support=0.999,
    )
    npix = 64
    ntimes = 2
    rng = np.random.default_rng(0)
    keys = [f"w{s}_{p}" for s in "IQU" for p in ("x", "y", "xy")]
    beam = types.SimpleNamespace(
        weights={k: rng.normal(size=npix) for k in keys}
    )
    I = rng.uniform(50, 100, (ntimes, npix))
    Q = rng.normal(size=(ntimes, npix))
    U = rng.normal(size=(ntimes, npix))
    rm = np.full((ntimes, npix), rm_value)
    mask = rng.random(npix) > 0.3
    return plan, I, Q, U, rm, beam, mask


def test_output_shapes():
    plan, I, Q, U, rm, beam, mask = _setup(5.0)
    out, table = simulate_channelized(
        plan, I, Q, U, rm, beam, mask, nproc=1
    )
    nchan = table["nu"].size  # 1 wide + 64 zoom = 65
    assert nchan == 65
    for key in (
        "pI_FR", "pQ_FR", "pU_FR",
        "pI_noFR", "pQ_noFR", "pU_noFR",
    ):
        assert out[key].shape == (2, nchan)


def test_zero_rm_fr_matches_nofr():
    plan, I, Q, U, rm, beam, mask = _setup(0.0)
    out, _ = simulate_channelized(
        plan, I, Q, U, rm, beam, mask, nproc=1
    )
    np.testing.assert_allclose(
        out["pQ_FR"], out["pQ_noFR"], rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        out["pU_FR"], out["pU_noFR"], rtol=1e-3, atol=1e-3
    )


def test_faraday_suppresses_polarization():
    plan, I, Q, U, rm, beam, mask = _setup(40.0)
    out, _ = simulate_channelized(
        plan, I, Q, U, rm, beam, mask, nproc=1
    )
    p_fr = np.hypot(
        out["pQ_FR"][:, 0], out["pU_FR"][:, 0]
    )      # wide chan
    p_nofr = np.hypot(
        out["pQ_noFR"][:, 0], out["pU_noFR"][:, 0]
    )
    assert np.all(p_fr < p_nofr)
