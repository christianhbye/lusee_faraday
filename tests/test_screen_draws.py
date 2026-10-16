"""Drawn-screen statistics and the direct sum against the analytic moments."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.gaussian_screen import (  # noqa: E402
    READOUTS,
    ScreenModel,
    visibility_moments,
)
from lusee_faraday.screen_draws import (  # noqa: E402
    ScreenGrid,
    direct_readouts,
    draw_depth,
)

GRID = ScreenGrid(n_pix=1024, width_deg=8.0)
FREQ = np.array([30.0, 40.0, 50.0])


def test_draw_matches_variance_and_correlation():
    model = ScreenModel(mean_depth=0.0)
    ell = model.correlation_pc / model.distance_pc
    lag = int(round(ell / GRID.pixel_rad))
    var, cor = [], []
    for seed in range(16):
        f = draw_depth(model, GRID, seed)
        var.append(f.var())
        cor.append(np.mean(f * np.roll(f, lag, axis=0)) / f.var())
    # ~60 independent correlation cells per map; allow 3 sigma of that.
    assert np.mean(var) == pytest.approx(model.sigma_depth**2, rel=0.1)
    expected = np.exp(-0.5 * (lag * GRID.pixel_rad / ell) ** 2)
    assert np.mean(cor) == pytest.approx(expected, abs=0.03)


def test_draw_is_reproducible_and_seed_dependent():
    model = ScreenModel()
    a = draw_depth(model, GRID, 3)
    assert np.array_equal(a, draw_depth(model, GRID, 3))
    assert not np.allclose(a, draw_depth(model, GRID, 4))


def test_infinite_correlation_is_one_constant():
    model = ScreenModel(correlation_pc=np.inf)
    f = draw_depth(model, GRID, 0)
    assert np.ptp(f) == 0.0


@pytest.mark.parametrize("name", list(READOUTS))
def test_direct_sum_matches_analytic_for_constant_screen(name):
    # sigma = 0: a deterministic rotation, so the analytic mean is exact.
    model = replace(ScreenModel(), sigma_depth=0.0, readout=READOUTS[name])
    depth = np.full((GRID.n_pix,) * 2, model.mean_depth)
    off, on = direct_readouts(depth, FREQ, model, GRID, names=(name,))[name]
    ref = visibility_moments(FREQ, model, diagonal_only=True)
    # The box truncates the source/beam Gaussian at ~1e-5.
    assert on == pytest.approx(ref["mean_on"].real, rel=1e-4, abs=1e-8)
    assert off == pytest.approx(ref["off"].real, rel=1e-4, abs=1e-8)
    assert np.max(abs(ref["mean_on"].imag)) < 1e-12


def test_zero_depth_screen_leaves_readouts_unrotated():
    model = ScreenModel(mean_depth=0.0, sigma_depth=0.0)
    depth = np.zeros((GRID.n_pix,) * 2)
    for off, on in direct_readouts(depth, FREQ, model, GRID).values():
        assert np.allclose(on, off, rtol=0, atol=1e-14)


def test_stokes_readouts_are_linear_combinations():
    model = ScreenModel()
    mom = {
        name: visibility_moments(
            FREQ, replace(model, readout=READOUTS[name]), diagonal_only=True
        )
        for name in READOUTS
    }
    assert np.allclose(
        mom["xx"]["mean_on"] - mom["yy"]["mean_on"], mom["Q"]["mean_on"]
    )
    assert np.allclose(2 * mom["xy"]["mean_on"], mom["U"]["mean_on"])
    assert np.allclose(
        mom["xx"]["off"] + mom["yy"]["off"],
        model.eta * model.i_ref_k * (FREQ / model.ref_mhz) ** model.beta,
    )


def test_readout_validation():
    with pytest.raises(ValueError):
        ScreenModel(readout=(1.0, 0.0))
    with pytest.raises(ValueError):
        ScreenModel(readout=(1.0, np.nan, 0.0))
    with pytest.raises(ValueError):
        ScreenGrid(n_pix=7)
