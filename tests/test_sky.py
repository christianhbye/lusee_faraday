"""Tests for the SkyModel and sky utility functions."""

import numpy as np
import pytest

import lusee_faraday as ld
from lusee_faraday.sky import power_law, point_src


NSIDE = 32


class TestFaradayRotation:
    def test_preserves_polarized_intensity(self):
        """Faraday rotation should preserve P = sqrt(Q^2 + U^2)
        at each pixel and frequency."""
        npix = 12 * NSIDE**2
        rng = np.random.default_rng(42)
        I_map = rng.uniform(1, 10, npix)
        Q_map = rng.uniform(-1, 1, npix)
        U_map = rng.uniform(-1, 1, npix)
        RM_map = rng.uniform(-100, 100, npix)
        freqs = np.array([10.0, 20.0, 30.0])

        # Power law to get multi-freq maps
        maps = power_law(np.array([I_map, Q_map, U_map]), freqs, 30.0)
        sky = ld.SkyModel(
            maps[:, 0], maps[:, 1], maps[:, 2],
            RM_map, freq=freqs, frame="galactic",
        )

        P_before = np.sqrt(sky.Q_map**2 + sky.U_map**2)
        sky.apply_fd()
        P_after = np.sqrt(sky.Q_map**2 + sky.U_map**2)

        np.testing.assert_allclose(P_after, P_before, rtol=1e-12)

    def test_leaves_I_unchanged(self):
        """Faraday rotation only affects Q and U, not I."""
        npix = 12 * NSIDE**2
        rng = np.random.default_rng(42)
        I_map = rng.uniform(1, 10, npix)
        Q_map = rng.uniform(-1, 1, npix)
        U_map = rng.uniform(-1, 1, npix)
        RM_map = rng.uniform(-100, 100, npix)

        sky = ld.SkyModel(
            I_map, Q_map, U_map, RM_map, freq=30.0, frame="galactic"
        )
        I_before = sky.I_map.copy()
        sky.apply_fd()
        np.testing.assert_array_equal(sky.I_map, I_before)

    def test_zero_rm_is_identity(self):
        """Zero RM everywhere should leave Q and U unchanged."""
        npix = 12 * NSIDE**2
        rng = np.random.default_rng(42)
        Q_map = rng.uniform(-1, 1, npix)
        U_map = rng.uniform(-1, 1, npix)
        RM_map = np.zeros(npix)

        sky = ld.SkyModel(
            np.ones(npix), Q_map.copy(), U_map.copy(),
            RM_map, freq=30.0, frame="galactic",
        )
        sky.apply_fd()
        np.testing.assert_allclose(sky.Q_map[0], Q_map, atol=1e-14)
        np.testing.assert_allclose(sky.U_map[0], U_map, atol=1e-14)

    def test_not_applied_twice(self, capsys):
        """Calling apply_fd twice should print a warning and not
        re-apply."""
        npix = 12 * NSIDE**2
        Q = np.ones(npix)
        U = np.zeros(npix)
        RM = 10.0 * np.ones(npix)
        sky = ld.SkyModel(
            np.ones(npix), Q, U, RM, freq=30.0, frame="galactic"
        )
        sky.apply_fd()
        Q_first = sky.Q_map.copy()
        sky.apply_fd()
        captured = capsys.readouterr()
        assert "already applied" in captured.out
        np.testing.assert_array_equal(sky.Q_map, Q_first)

    def test_frequency_dependence(self):
        """Faraday rotation angle should differ across frequencies
        for non-zero RM."""
        npix = 12 * NSIDE**2
        Q = np.ones(npix)
        U = np.zeros(npix)
        RM = 50.0 * np.ones(npix)

        sky = ld.SkyModel(
            np.ones(npix), Q.copy(), U.copy(),
            RM, freq=[10.0, 30.0], frame="galactic",
        )
        sky.apply_fd()
        # Q should differ between the two frequencies
        assert not np.allclose(sky.Q_map[0], sky.Q_map[1])


class TestPowerLaw:
    def test_at_reference(self):
        """At the reference frequency, output should equal input."""
        npix = 100
        maps = np.random.default_rng(0).uniform(0, 1, (3, npix))
        result = power_law(maps, 30.0, 30.0)
        np.testing.assert_allclose(result[0], maps, rtol=1e-12)

    def test_scaling(self):
        """Should follow (f/f_ref)^beta."""
        npix = 100
        maps = np.ones((3, npix))
        beta = -2.5
        f = 15.0  # half of ref
        ref = 30.0
        result = power_law(maps, f, ref, beta=beta)
        expected = (f / ref) ** beta
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_output_shape(self):
        npix = 100
        maps = np.ones((3, npix))
        freqs = np.array([10.0, 20.0, 30.0])
        result = power_law(maps, freqs, 30.0)
        assert result.shape == (3, 3, npix)


class TestPointSource:
    def test_zenith_source_stokes(self):
        """Point source at zenith: I=1, Q=-1, U=0 at pixel 0."""
        I, Q, U = point_src(lat_center=90, lon_center=0, nside=NSIDE)
        assert I[0, 0] == 1.0
        assert Q[0, 0] == -1.0
        assert U[0, 0] == 0.0

    def test_zenith_source_shape(self):
        nfreqs = 16
        I, Q, U = point_src(nside=NSIDE, nfreqs=nfreqs)
        npix = 12 * NSIDE**2
        assert I.shape == (nfreqs, npix)
        assert Q.shape == (nfreqs, npix)
        assert U.shape == (nfreqs, npix)

    def test_rotated_source_total_flux(self):
        """Total flux should be preserved after rotation."""
        I_z, _, _ = point_src(lat_center=90, lon_center=0, nside=NSIDE)
        I_r, _, _ = point_src(lat_center=45, lon_center=90, nside=NSIDE)
        np.testing.assert_allclose(
            I_z.sum(), I_r.sum(), rtol=0.3
        )

    def test_full_polarization(self):
        """Point source should be 100% linearly polarized."""
        I, Q, U = point_src(nside=NSIDE)
        pol_frac = np.sqrt(Q[0] ** 2 + U[0] ** 2) / np.where(
            I[0] > 0, I[0], 1
        )
        bright = I[0] > 0.5
        np.testing.assert_allclose(pol_frac[bright], 1.0, atol=0.01)
