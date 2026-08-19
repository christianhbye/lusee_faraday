import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import channelization as ch  # noqa: E402


def test_zoom_bin_offsets_use_fft_ordering():
    off = ch.zoom_bin_offsets_hz()
    assert off.shape == (64,)
    assert off[0] == 0.0
    assert np.isclose(off[1], ch.ZOOM_STEP_HZ)
    assert np.isclose(off[63], -ch.ZOOM_STEP_HZ)
    # Bin 32 is Nyquist: an exact 50/50 double peak at +-12.5 kHz of
    # its parent, so the sign is a labelling convention rather than a
    # physical fact. We follow numpy's fftfreq convention, which puts
    # Nyquist on the negative side.
    assert np.isclose(off[32], -32 * ch.ZOOM_STEP_HZ)


def test_weights_are_normalized():
    pytest.importorskip("lusee")
    off = np.linspace(-50000.0, 50000.0, 4001)
    assert np.isclose(ch.parent_weights(off).sum(), 1.0)
    assert np.allclose(ch.zoom_weights(off).sum(axis=0), 1.0)
    assert np.allclose(ch.ideal_zoom_weights(off).sum(axis=0), 1.0)


def test_integrating_a_constant_spectrum_returns_the_constant():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)
    waterfall = np.full((2, fine.size, 3), 7.0)
    out = ch.integrate(waterfall, fine, np.array([30.0]))
    assert np.allclose(out["parent"], 7.0, rtol=1e-6)
    assert np.allclose(out["zoom"], 7.0, rtol=1e-6)
    assert np.allclose(out["ideal_zoom"], 7.0, rtol=1e-6)


def test_integrate_shapes():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(16384) - 8192) * (25e-3 / 2048)
    waterfall = np.zeros((5, fine.size, 16))
    parents = np.array([29.975, 30.0, 30.025])
    out = ch.integrate(waterfall, fine, parents)
    assert out["parent"].shape == (5, 3, 16)
    assert out["zoom"].shape == (5, 3, 64, 16)
    assert out["ideal_zoom"].shape == (5, 3, 64, 16)


def test_integrate_rejects_a_grid_that_does_not_cover_the_response():
    pytest.importorskip("lusee")
    fine = 30.0 + (np.arange(256) - 128) * (25e-3 / 2048)  # +-1.6 kHz
    waterfall = np.zeros((1, fine.size, 1))
    with pytest.raises(ValueError, match="does not cover"):
        ch.integrate(waterfall, fine, np.array([30.0]))


def test_integrate_rejects_a_nonuniform_grid():
    pytest.importorskip("lusee")
    fine = np.concatenate(
        [np.linspace(29.9, 30.0, 100), np.linspace(30.001, 30.1, 100)]
    )
    waterfall = np.zeros((1, fine.size, 1))
    with pytest.raises(ValueError, match="uniform"):
        ch.integrate(waterfall, fine, np.array([30.0]))


def test_zoom_frequency_grid_is_contiguous_and_unique():
    parents = np.array([29.975, 30.0, 30.025])
    freqs, order = ch.zoom_frequency_grid(parents)
    assert freqs.size == 192
    assert len(order) == 192
    assert np.all(np.diff(freqs) > 0)
    steps = np.diff(freqs) * 1e6
    assert np.allclose(steps, ch.ZOOM_STEP_HZ, rtol=1e-6)
