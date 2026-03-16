"""Tests for coordinate rotation utilities."""

import numpy as np
import pytest

from lusee_faraday.rotations import get_rot_mat, rotmat_to_eulerZYX, gal2topo


class TestRotationMatrix:
    def test_orthogonality(self):
        """Rotation matrix from any frame should be orthogonal:
        R @ R^T = I."""
        from lunarsky import LunarTopo, MoonLocation, Time

        loc = MoonLocation(lat=0, lon=0, height=0)
        time = Time("2027-01-01T00:00:00")
        topo = LunarTopo(location=loc, obstime=time)
        rmat = get_rot_mat(topo)
        product = rmat @ rmat.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_determinant_unit(self):
        """Rotation matrix should have |det| = 1 (may include a
        parity flip between the two coordinate systems)."""
        from lunarsky import LunarTopo, MoonLocation, Time

        loc = MoonLocation(lat=0, lon=0, height=0)
        time = Time("2027-01-01T00:00:00")
        topo = LunarTopo(location=loc, obstime=time)
        rmat = get_rot_mat(topo)
        assert abs(np.linalg.det(rmat)) == pytest.approx(1.0, abs=1e-10)


class TestEulerAngles:
    def test_identity_matrix(self):
        """Identity matrix should give zero Euler angles."""
        alpha, beta, gamma = rotmat_to_eulerZYX(np.eye(3))
        assert alpha == pytest.approx(0, abs=1e-10)
        assert beta == pytest.approx(0, abs=1e-10)
        assert gamma == pytest.approx(0, abs=1e-10)


class TestGal2Topo:
    def test_preserves_total_intensity(self):
        """Coordinate rotation should preserve the total intensity
        (sum over pixels)."""
        import healpy as hp
        from lunarsky import LunarTopo, MoonLocation, Time

        nside = 16
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        I_map = rng.uniform(1, 10, npix)
        Q_map = rng.uniform(-1, 1, npix)
        U_map = rng.uniform(-1, 1, npix)

        loc = MoonLocation(lat=-23.813, lon=182.258, height=0)
        time = Time("2027-01-20T09:00:00")
        topo = LunarTopo(location=loc, obstime=time)

        I_t, Q_t, U_t = gal2topo(I_map, Q_map, U_map, topo_frame=topo)
        # Total I should be conserved (it's a scalar field rotation)
        np.testing.assert_allclose(
            I_t.sum(), I_map.sum(), rtol=0.05
        )

    def test_preserves_polarized_power(self):
        """Sum of Q^2 + U^2 should be approximately preserved."""
        import healpy as hp
        from lunarsky import LunarTopo, MoonLocation, Time

        nside = 16
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        I_map = rng.uniform(1, 10, npix)
        Q_map = rng.uniform(-1, 1, npix)
        U_map = rng.uniform(-1, 1, npix)

        loc = MoonLocation(lat=-23.813, lon=182.258, height=0)
        time = Time("2027-01-20T09:00:00")
        topo = LunarTopo(location=loc, obstime=time)

        I_t, Q_t, U_t = gal2topo(I_map, Q_map, U_map, topo_frame=topo)
        pol_before = np.sum(Q_map**2 + U_map**2)
        pol_after = np.sum(Q_t**2 + U_t**2)
        # SHT rotation at low nside loses some polarized power; use
        # generous tolerance
        np.testing.assert_allclose(pol_after, pol_before, rtol=0.3)

    def test_multifreq(self):
        """Should handle multiple frequency maps."""
        import healpy as hp
        from lunarsky import LunarTopo, MoonLocation, Time

        nside = 16
        npix = hp.nside2npix(nside)
        nfreq = 4
        rng = np.random.default_rng(0)
        I_map = rng.uniform(1, 10, (nfreq, npix))
        Q_map = rng.uniform(-1, 1, (nfreq, npix))
        U_map = rng.uniform(-1, 1, (nfreq, npix))

        loc = MoonLocation(lat=-23.813, lon=182.258, height=0)
        time = Time("2027-01-20T09:00:00")
        topo = LunarTopo(location=loc, obstime=time)

        I_t, Q_t, U_t = gal2topo(I_map, Q_map, U_map, topo_frame=topo)
        assert I_t.shape == (nfreq, npix)
        assert Q_t.shape == (nfreq, npix)
        assert U_t.shape == (nfreq, npix)
