import numpy as np
import pytest
from lusee_faraday import SpectrometerResponse
from lusee_faraday.freqplan import FrequencyPlan, _snap_to_lusee, BIN_WIDTH_HZ, N_ZOOM

SPEC_PATH = "data/spectrometer_bin_response.txt"


@pytest.fixture(scope="module")
def spec():
    return SpectrometerResponse.from_file(SPEC_PATH)


def test_snap_to_lusee_rounds_to_grid():
    assert np.isclose(_snap_to_lusee(30.0), 30.0)
    assert np.isclose(_snap_to_lusee(30.01), 30.0)


def test_sim_freqs_dedups_overlapping_parents(spec):
    specs = [(29.975, "zoom"), (30.0, "zoom")]
    plan = FrequencyPlan(spec, specs)
    sf = plan.sim_freqs()
    naive = 2 * len(spec.freq_offset_hz)
    assert sf.size < naive
    assert np.all(np.diff(sf) > 0)


def sf_to_hz(sf):
    return np.round(sf * 1e6).astype(np.int64)


def test_channelize_matches_apply_narrow_and_wide(spec):
    specs = [(30.0, "zoom"), (40.0, "wide")]
    plan = FrequencyPlan(spec, specs)
    sf = plan.sim_freqs()
    raw = 1000.0 + (sf - sf.mean()) ** 2
    ch = plan.channelize(raw)
    off_hz = np.round(spec.freq_offset_hz).astype(np.int64)
    ref = []
    for c, m in plan.specs:
        a = np.round(c * 1e6).astype(np.int64) + off_hz
        win = raw[np.searchsorted(sf_to_hz(sf), a)]
        if m == "wide":
            ref.append(np.atleast_1d(spec.apply_wide(win)))
        else:
            ref.append(spec.apply_narrow(win))
    ref = np.concatenate(ref)
    assert ch.shape == (65,)
    assert np.allclose(ch, ref)


def test_channelize_2d_input(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    raw = np.ones((3, plan.sim_freqs().size))
    ch = plan.channelize(raw)
    assert ch.shape == (3, 65)


def test_channel_table_sizes_and_dnu(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    t = plan.channel_table
    assert t["nu"].shape == (65,)
    assert t["lambda2"].shape == (65,)
    assert np.allclose(t["dnu"][:64], BIN_WIDTH_HZ / N_ZOOM)
    assert np.isclose(t["dnu"][64], BIN_WIDTH_HZ)


def test_channel_table_nu_equals_channelized_frequency_axis(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    nu = plan.channel_table["nu"]
    assert np.allclose(nu, plan.channelize(plan.sim_freqs()))


def test_channel_table_zoom_is_non_monotonic(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom")])
    nu = plan.channel_table["nu"]
    assert nu.size == 64
    assert not np.all(np.diff(nu) > 0)
    assert np.all(np.abs(nu - 30.0) <= BIN_WIDTH_HZ / 2 * 1e-6 + 1e-9)


def test_channel_table_lambda2_consistent(spec):
    from lusee_faraday import rmsynth
    plan = FrequencyPlan(spec, [(30.0, "zoom"), (40.0, "wide")])
    t = plan.channel_table
    assert np.allclose(t["lambda2"], rmsynth.lambda2(t["nu"]))


def test_per_mode_decimation(spec):
    plan = FrequencyPlan(
        spec, [(30.0, "zoom"), (40.0, "wide")],
        decimation={"zoom": 10, "wide": 250},
    )
    raw = np.ones(plan.sim_freqs().size)
    ch = plan.channelize(raw)
    assert ch.shape == (65,)
    # normalized responses: a flat unit spectrum channelizes to 1.0
    assert np.allclose(ch, 1.0)
    t = plan.channel_table
    assert t["nu"].shape == (65,)


def test_support_truncation_shrinks_sim_grid(spec):
    full = FrequencyPlan(spec, [(10.0, "zoom")])
    trunc = FrequencyPlan(spec, [(10.0, "zoom")], support=0.999)
    assert trunc.sim_freqs().size < full.sim_freqs().size


def test_truncated_channelize_matches_full_on_faraday(spec):
    c = 3e8
    full = FrequencyPlan(spec, [(30.0, "wide")])
    trunc = FrequencyPlan(spec, [(30.0, "wide")], support=0.999)
    Pf = np.exp(2j * 20.0 * (c / (full.sim_freqs() * 1e6)) ** 2)
    Pt = np.exp(2j * 20.0 * (c / (trunc.sim_freqs() * 1e6)) ** 2)
    assert np.allclose(
        np.abs(full.channelize(Pf)), np.abs(trunc.channelize(Pt)), atol=1e-3
    )


def test_int_decimation_still_supported(spec):
    plan = FrequencyPlan(spec, [(30.0, "zoom")], decimation=10)
    assert plan.channel_table["nu"].shape == (64,)
