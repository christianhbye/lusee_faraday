"""Tests for the spectrometer response module.

Focuses on physical behaviour: normalization, flat-spectrum preservation,
narrow-to-wide consistency, and response symmetry.
"""

import numpy as np
import pytest

import lusee_faraday as ld


class TestSpectrometerLoading:
    def test_load_shape(self, spec_response):
        sr = spec_response
        n = sr.freq_offset_hz.size
        assert sr.wide.shape == (n,)
        assert sr.narrow.shape == (n, 64)

    def test_frequency_range(self, spec_response):
        """Offsets span ±2 bins = ±50 kHz."""
        sr = spec_response
        assert sr.freq_offset_hz[0] == pytest.approx(-50000, abs=1)
        assert sr.freq_offset_hz[-1] == pytest.approx(50000, abs=1)

    def test_freq_offset_mhz(self, spec_response):
        sr = spec_response
        np.testing.assert_allclose(
            sr.freq_offset_mhz, sr.freq_offset_hz * 1e-6
        )

    def test_positive_response(self, spec_response):
        """All response values should be non-negative."""
        sr = spec_response
        assert np.all(sr.wide >= 0)
        assert np.all(sr.narrow >= 0)


class TestFlatSpectrum:
    """A flat (frequency-independent) spectrum should be preserved."""

    def test_wide_flat_spectrum(self, spec_response):
        sr = spec_response
        flat = np.ones_like(sr.freq_offset_hz)
        result = sr.apply_wide(flat)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_narrow_flat_spectrum(self, spec_response):
        sr = spec_response
        flat = np.ones_like(sr.freq_offset_hz)
        result = sr.apply_narrow(flat)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_wide_constant_spectrum(self, spec_response):
        """A constant spectrum S=C should return C."""
        sr = spec_response
        C = 42.7
        flat = C * np.ones_like(sr.freq_offset_hz)
        result = sr.apply_wide(flat)
        assert result == pytest.approx(C, rel=1e-10)

    def test_narrow_constant_spectrum(self, spec_response):
        C = 42.7
        sr = spec_response
        flat = C * np.ones_like(sr.freq_offset_hz)
        result = sr.apply_narrow(flat)
        np.testing.assert_allclose(result, C, rtol=1e-10)


class TestNarrowWideConsistency:
    """The sum of narrow bin outputs, properly weighted, should
    approximate the wide bin output for a smooth spectrum."""

    def test_linear_spectrum_consistency(self, spec_response):
        """For a slowly-varying (linear) spectrum, the weighted mean of
        narrow bins should be close to the wide bin output."""
        sr = spec_response
        # Linear spectrum: S(f) = 1 + 0.001 * f_offset_khz
        spectrum = 1.0 + 1e-6 * sr.freq_offset_hz

        wide_out = sr.apply_wide(spectrum)
        narrow_out = sr.apply_narrow(spectrum)  # (64,)

        # Weighted average of narrow bins using their total power
        narrow_total_power = sr.narrow.sum(axis=0)
        narrow_weighted = (
            np.sum(narrow_out * narrow_total_power)
            / narrow_total_power.sum()
        )
        # Should agree to a few percent — they use different response
        # shapes so won't be exact
        assert narrow_weighted == pytest.approx(wide_out, rel=0.05)


class TestResponseSymmetry:
    """The wide bin response should be approximately symmetric."""

    def test_wide_symmetric(self, spec_response):
        sr = spec_response
        n = sr.wide.size
        wide_rev = sr.wide[::-1]
        # Approximate symmetry about center
        np.testing.assert_allclose(sr.wide, wide_rev, rtol=1e-3)

    def test_narrow_bins_tile_wide(self, spec_response):
        """The sum of all narrow bin responses should roughly track
        the wide bin shape (the narrow bins tile the parent bin)."""
        sr = spec_response
        narrow_sum = sr.narrow.sum(axis=1)
        # Normalize both to unit peak for shape comparison
        wide_normed = sr.wide / sr.wide.max()
        narrow_normed = narrow_sum / narrow_sum.max()
        # Correlation between the two shapes should be high
        corr = np.corrcoef(wide_normed, narrow_normed)[0, 1]
        assert corr > 0.95


class TestDecimate:
    def test_decimate_preserves_flat(self, spec_response):
        sr_dec = spec_response.decimate(10)
        flat = np.ones_like(sr_dec.freq_offset_hz)
        assert sr_dec.apply_wide(flat) == pytest.approx(1.0, rel=1e-10)

    def test_decimate_reduces_size(self, spec_response):
        sr = spec_response
        sr_dec = sr.decimate(10)
        assert sr_dec.freq_offset_hz.size == pytest.approx(
            sr.freq_offset_hz.size // 10, abs=1
        )

    def test_decimate_narrow_flat(self, spec_response):
        sr_dec = spec_response.decimate(10)
        flat = np.ones_like(sr_dec.freq_offset_hz)
        result = sr_dec.apply_narrow(flat)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)


class TestFreqs:
    def test_freqs_centered(self, spec_response):
        sr = spec_response
        center = 30.0
        freqs = sr.freqs(center)
        assert freqs[0] == pytest.approx(center - 0.05, abs=1e-6)
        assert freqs[-1] == pytest.approx(center + 0.05, abs=1e-6)

    def test_freqs_size(self, spec_response):
        sr = spec_response
        freqs = sr.freqs(30.0)
        assert freqs.size == sr.freq_offset_hz.size


class TestBatchApplication:
    """Test that apply_wide and apply_narrow work with batched inputs."""

    def test_wide_batched(self, spec_response):
        sr = spec_response
        n = sr.freq_offset_hz.size
        batch = np.ones((3, n))
        batch[1] *= 2.0
        batch[2] *= 3.0
        result = sr.apply_wide(batch)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-10)

    def test_narrow_batched(self, spec_response):
        sr = spec_response
        n = sr.freq_offset_hz.size
        batch = np.ones((2, n))
        batch[1] *= 5.0
        result = sr.apply_narrow(batch)
        assert result.shape == (2, 64)
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 5.0, rtol=1e-10)


class TestNarrowBinOrdering:
    """Zoom bins use FFT-style ordering: bin 0 at center, bins 1-32
    at positive offsets, bins 33-63 at negative offsets."""

    def test_bin0_peaks_near_center(self, spec_response):
        sr = spec_response
        peak_hz = sr.freq_offset_hz[np.argmax(sr.narrow[:, 0])]
        assert abs(peak_hz) < 100  # near DC

    def test_positive_bins_increasing(self, spec_response):
        """Bins 1 through 32 should have monotonically increasing
        peak frequencies."""
        sr = spec_response
        peaks = np.array(
            [sr.freq_offset_hz[np.argmax(sr.narrow[:, k])]
             for k in range(1, 33)]
        )
        assert np.all(np.diff(peaks) > 0)

    def test_negative_bins_increasing(self, spec_response):
        """Bins 33 through 63 should have monotonically increasing
        peak frequencies (from most negative to least negative)."""
        sr = spec_response
        peaks = np.array(
            [sr.freq_offset_hz[np.argmax(sr.narrow[:, k])]
             for k in range(33, 64)]
        )
        assert np.all(np.diff(peaks) > 0)

    def test_peak_spacing_positive_half(self, spec_response):
        """Bins 0-32: peak spacing ≈ one zoom-bin width."""
        sr = spec_response
        peaks = np.array(
            [sr.freq_offset_hz[np.argmax(sr.narrow[:, k])]
             for k in range(33)]
        )
        spacings = np.diff(peaks)
        expected = sr.BIN_WIDTH_HZ / sr.N_ZOOM
        np.testing.assert_allclose(spacings, expected, rtol=0.1)


# ---------------------------------------------------------------------------
# Truncate tests
# ---------------------------------------------------------------------------

SPEC_PATH = "data/spectrometer_bin_response.txt"


def _spec():
    return ld.SpectrometerResponse.from_file(SPEC_PATH)


def test_truncate_reduces_points_and_is_symmetric():
    s = _spec()
    t = s.truncate(0.999)
    assert t.freq_offset_hz.size < s.freq_offset_hz.size
    assert 0.3 < t.freq_offset_hz.size / s.freq_offset_hz.size < 0.45
    assert np.isclose(t.freq_offset_hz.min(), -t.freq_offset_hz.max())


def test_truncate_renormalizes():
    t = _spec().truncate(0.999)
    assert np.isclose(t._wide_norm.sum(), 1.0)
    np.testing.assert_allclose(t._narrow_norm.sum(axis=0), 1.0)


def test_truncate_preserves_channelization_on_smooth_spectrum():
    s = _spec()
    t = s.truncate(0.999)
    full = 100.0 + 2.0 * s.freq_offset_mhz
    trunc = 100.0 + 2.0 * t.freq_offset_mhz
    assert np.isclose(s.apply_wide(full), t.apply_wide(trunc), atol=1e-3)
    np.testing.assert_allclose(
        s.apply_narrow(full), t.apply_narrow(trunc), atol=1e-3
    )


def test_truncate_preserves_faraday_depolarization():
    s = _spec()
    t = s.truncate(0.999)
    c = 3e8
    for center, rm in [(30.0, 20.0), (50.0, 20.0)]:
        nuf = (center + s.freq_offset_mhz) * 1e6
        nut = (center + t.freq_offset_mhz) * 1e6
        Pf = np.exp(2j * rm * (c / nuf) ** 2)
        Pt = np.exp(2j * rm * (c / nut) ** 2)
        assert abs(abs(s.apply_wide(Pf)) - abs(t.apply_wide(Pt))) < 1e-3
