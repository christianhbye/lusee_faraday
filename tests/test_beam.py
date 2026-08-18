"""Tests for the Beam class, focusing on physical properties."""

import numpy as np
import pytest

import lusee_faraday as ld


NSIDE = 32


class TestShortDipole:
    def test_jones_shape(self):
        beam = ld.Beam.short_dipole(nside=NSIDE)
        npix = 12 * NSIDE**2
        assert beam.jones_x.shape == (2, npix)
        assert beam.jones_y.shape == (2, npix)

    def test_orthogonal_polarizations(self):
        """X and Y dipoles should have orthogonal Jones vectors at
        each pixel, in the sense that |E_x|^2 and |E_y|^2 have the
        same total power (by symmetry of crossed dipoles)."""
        beam = ld.Beam.short_dipole(nside=NSIDE)
        power_x = np.sum(np.abs(beam.jones_x) ** 2, axis=0)
        power_y = np.sum(np.abs(beam.jones_y) ** 2, axis=0)
        # Total power integrated over the sphere should be equal
        np.testing.assert_allclose(
            power_x.sum(), power_y.sum(), rtol=1e-10
        )

    def test_x_dipole_null_on_axis(self):
        """The X-dipole has a null along the x-axis (θ=π/2, φ=0).
        Power pattern is 1 - sin²(θ)cos²(φ)."""
        beam = ld.Beam.short_dipole(nside=NSIDE)
        grid = ld.HealpixGrid(nside=NSIDE, horizon=False)
        power_x = np.abs(beam.jones_x[0]) ** 2 + np.abs(beam.jones_x[1]) ** 2
        # Find pixel nearest to (theta=pi/2, phi=0)
        equator_mask = np.abs(grid.theta - np.pi / 2) < 0.1
        phi_zero_mask = np.abs(grid.phi) < 0.1
        near_null = equator_mask & phi_zero_mask
        if near_null.any():
            assert np.min(power_x[near_null]) < 0.05


class TestWeights:
    def test_precompute_creates_dict(self, short_dipole):
        beam = short_dipole
        assert beam.weights is not None
        expected_keys = {
            "wI_x", "wQ_x", "wU_x",
            "wI_y", "wQ_y", "wU_y",
            "wI_xy", "wQ_xy", "wU_xy",
        }
        assert set(beam.weights.keys()) == expected_keys

    def test_weights_non_negative_intensity(self, short_dipole):
        """The intensity weight wI should be non-negative everywhere
        (it is a sum of squared Jones components)."""
        beam = short_dipole
        assert np.all(beam.weights["wI_x"] >= 0)
        assert np.all(beam.weights["wI_y"] >= 0)

    def test_wI_equals_power(self, short_dipole):
        """wI_x = 0.5 * (|Eth|^2 + |Eph|^2), i.e. half the power."""
        beam = short_dipole
        expected = 0.5 * (
            np.abs(beam.jones_x[0]) ** 2 + np.abs(beam.jones_x[1]) ** 2
        )
        np.testing.assert_allclose(beam.weights["wI_x"], expected)

    def test_stokes_weight_identity(self, short_dipole):
        """For each polarization p, wI_p = wQ_p at points where
        Eph = 0. This is a consistency check on the weight definitions."""
        beam = short_dipole
        # Where Eph is zero (at poles), wI = 0.5*|Eth|^2 and wQ = 0.5*|Eth|^2
        mask = np.abs(beam.jones_x[1]) < 1e-12
        if mask.any():
            np.testing.assert_allclose(
                beam.weights["wI_x"][mask],
                beam.weights["wQ_x"][mask],
                atol=1e-12,
            )

    def test_idempotent_precompute(self, short_dipole):
        """Calling precompute_weights twice should not change the result."""
        beam = short_dipole
        w1 = beam.weights["wI_x"].copy()
        beam.precompute_weights()
        np.testing.assert_array_equal(beam.weights["wI_x"], w1)


class TestRotateBeam:
    def test_360_rotation_identity(self):
        """Rotating by 360 degrees should return the original."""
        from lusee_faraday.beam import rotate_beam
        jones = np.random.default_rng(0).standard_normal((2, 90, 360))
        rotated = rotate_beam(jones, 360)
        np.testing.assert_array_equal(rotated, jones)

    def test_power_preserved(self):
        """Rotation should preserve total power."""
        from lusee_faraday.beam import rotate_beam
        jones = np.random.default_rng(0).standard_normal((2, 90, 360))
        power_before = np.sum(np.abs(jones) ** 2)
        rotated = rotate_beam(jones, 45)
        power_after = np.sum(np.abs(rotated) ** 2)
        assert power_after == pytest.approx(power_before, rel=1e-12)


class TestInterpJonesPole:
    """The pole must not zero out the beam.

    E_theta and E_phi carry a pure m=1 azimuthal phase at a pole, so
    their azimuthal mean -- which is what RectSphereBivariateSpline
    receives as its pole value -- vanishes. Interpolating them directly
    therefore drove the beam to zero at zenith. interp_jones goes via the
    Cartesian components, which are m=0 at the pole.
    """

    def _analytic_dipole_grid(self, ntheta=181, nphi=360):
        """Short dipole along x on a 1 deg (theta, phi) grid."""
        theta = np.radians(np.linspace(0, 180, num=ntheta))
        phi = np.radians(np.arange(nphi))
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        Eth = -np.cos(th) * np.cos(ph)
        Eph = np.sin(ph)
        return np.array([Eth, Eph]), theta, phi

    def test_pole_value_is_not_zeroed(self):
        jones, theta, phi = self._analytic_dipole_grid()
        grid = ld.HealpixGrid(nside=NSIDE, horizon=False)
        got = ld.beam.interp_jones(grid, jones, theta, phi)
        power = np.sum(np.abs(got) ** 2, axis=0)
        # analytic short dipole: |E|^2 = cos^2(th)cos^2(ph) + sin^2(ph),
        # which equals 1 at the pole for every phi
        near_pole = grid.theta < np.radians(2.0)
        assert near_pole.any()
        np.testing.assert_allclose(power[near_pole], 1.0, rtol=2e-3)

    def test_matches_analytic_away_from_pole(self):
        jones, theta, phi = self._analytic_dipole_grid()
        grid = ld.HealpixGrid(nside=NSIDE, horizon=False)
        got = ld.beam.interp_jones(grid, jones, theta, phi)
        want_th = -np.cos(grid.theta) * np.cos(grid.phi)
        want_ph = np.sin(grid.phi)
        mid = np.abs(np.degrees(grid.theta) - 90) > 5
        np.testing.assert_allclose(got[0][mid], want_th[mid], atol=2e-3)
        np.testing.assert_allclose(got[1][mid], want_ph[mid], atol=2e-3)
