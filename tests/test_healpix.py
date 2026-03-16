"""Tests for the HealpixGrid class."""

import numpy as np
import pytest

import lusee_faraday as ld


NSIDE = 32


class TestHealpixGrid:
    def test_npix(self):
        grid = ld.HealpixGrid(nside=NSIDE, horizon=False)
        assert grid.npix == 12 * NSIDE**2

    def test_full_sky_area(self):
        """Total pixel area should be 4π steradians."""
        grid = ld.HealpixGrid(nside=NSIDE, horizon=False)
        total = grid.pix_area * grid.npix
        assert total == pytest.approx(4 * np.pi, rel=1e-10)

    def test_horizon_mask_count(self, healpix_grid):
        """Horizon mask should select roughly half the sky."""
        grid = healpix_grid
        frac = grid.mask.sum() / grid.npix
        assert frac == pytest.approx(0.5, abs=0.05)

    def test_horizon_mask_upper_hemisphere(self, healpix_grid):
        """Masked pixels should all have theta <= pi/2."""
        grid = healpix_grid
        assert np.all(grid.theta[grid.mask] <= np.pi / 2)

    def test_no_horizon(self, healpix_grid_full):
        """Without horizon cut, all pixels should be unmasked."""
        grid = healpix_grid_full
        assert grid.mask.all()


class TestInterpHp:
    def test_constant_field(self, healpix_grid_full):
        """Interpolating a constant field should return that constant."""
        grid = healpix_grid_full
        theta = np.linspace(0, np.pi, 91)
        phi = np.linspace(0, 2 * np.pi - np.radians(1), 360)
        arr = 5.0 * np.ones((theta.size, phi.size))
        result = grid.interp_hp(arr, theta, phi)
        np.testing.assert_allclose(result, 5.0, atol=0.1)

    def test_output_shape(self, healpix_grid_full):
        """Output should have shape (..., npix)."""
        grid = healpix_grid_full
        theta = np.linspace(0, np.pi, 91)
        phi = np.linspace(0, 2 * np.pi - np.radians(1), 360)
        # Shape (2, 91, 360)
        arr = np.ones((2, theta.size, phi.size))
        result = grid.interp_hp(arr, theta, phi)
        assert result.shape == (2, grid.npix)

    def test_complex_field(self, healpix_grid_full):
        """Interpolation should handle complex arrays."""
        grid = healpix_grid_full
        theta = np.linspace(0, np.pi, 91)
        phi = np.linspace(0, 2 * np.pi - np.radians(1), 360)
        arr = (3.0 + 4.0j) * np.ones((theta.size, phi.size))
        result = grid.interp_hp(arr, theta, phi)
        np.testing.assert_allclose(np.real(result), 3.0, atol=0.1)
        np.testing.assert_allclose(np.imag(result), 4.0, atol=0.1)
