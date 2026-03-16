"""Tests for LuSEE frequency utilities."""

import numpy as np
import pytest

from lusee_faraday.utils import freqs_lusee, freqs_zoom


class TestFreqsLusee:
    def test_channel_count(self):
        assert freqs_lusee().size == 2048

    def test_range(self):
        f = freqs_lusee()
        assert f[0] == 0.0
        assert f[-1] < 51.2

    def test_spacing(self):
        """Channel spacing should be 25 kHz = 0.025 MHz."""
        f = freqs_lusee()
        df = np.diff(f)
        np.testing.assert_allclose(df, 0.025, rtol=1e-10)


class TestFreqsZoom:
    def test_default_count(self):
        assert freqs_zoom().size == 64

    def test_custom_count(self):
        assert freqs_zoom(num=128).size == 128

    def test_within_parent_bin(self):
        """Zoom frequencies should span exactly one parent bin."""
        f_lusee = freqs_lusee()
        fz = freqs_zoom(center=30)
        bin_width = f_lusee[1] - f_lusee[0]
        span = fz[-1] - fz[0] + (fz[1] - fz[0])
        assert span == pytest.approx(bin_width, rel=1e-6)

    def test_center_near_requested(self):
        """The zoom band center should be near the requested frequency."""
        fz = freqs_zoom(center=30)
        mid = (fz[0] + fz[-1]) / 2
        assert mid == pytest.approx(30.0, abs=0.025)
