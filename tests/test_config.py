import numpy as np

from lusee_faraday import config as cfg


def test_time_grid_spans_exactly_one_lunar_sidereal_day():
    t = cfg.times()
    assert len(t) == cfg.N_TIMES == 1024
    span = (t[-1] - t[0]).sec
    step = cfg.SIDEREAL_DAY_S / cfg.N_TIMES
    assert np.isclose(span, cfg.SIDEREAL_DAY_S - step, rtol=1e-12)


def test_fine_grid_is_uniform_and_covers_the_three_parents():
    center = 30.0
    f = cfg.fine_freqs(center)
    assert f.size == cfg.N_FINE == 16384
    d = np.diff(f)
    assert np.allclose(d, d[0], rtol=1e-12)
    assert np.isclose(d[0], 25e-3 / 2048)
    parents = cfg.parent_centers(center)
    assert np.allclose(parents, [29.975, 30.0, 30.025])
    # every parent's +-50 kHz response support sits inside the fine grid
    assert f.min() <= parents.min() - 0.05
    assert f.max() >= parents.max() + 0.05


def test_site_matches_luseepy_observation_defaults():
    import pytest

    obs_mod = pytest.importorskip("lusee.Observation")
    obs = obs_mod.Observation()
    assert np.isclose(cfg.LUN_LAT_DEG, obs.default_lun_lat_deg)
    assert np.isclose(cfg.LUN_LONG_DEG, obs.default_lun_long_deg)


def test_band_centers_are_the_three_studied_bands():
    assert cfg.BAND_CENTERS_MHZ == (30.0, 10.0, 50.0)
