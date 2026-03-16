"""Tests for the Simulator, focusing on physical behaviour."""

import numpy as np
import pytest

import lusee_faraday as ld


NSIDE = 32


class TestComputeStokes:
    def test_unpolarized(self):
        """If Rxx = Ryy and Rxy = 0, then Q = U = 0."""
        ntimes, nfreqs = 2, 8
        Rmat = np.zeros((ntimes, 3, nfreqs))
        Rmat[:, 0, :] = 5.0  # Rxx
        Rmat[:, 1, :] = 5.0  # Ryy
        Rmat[:, 2, :] = 0.0  # Rxy
        I, Q, U = ld.Simulator.compute_stokes(Rmat)
        np.testing.assert_allclose(I, 10.0)
        np.testing.assert_allclose(Q, 0.0, atol=1e-14)
        np.testing.assert_allclose(U, 0.0, atol=1e-14)

    def test_fully_x_polarized(self):
        """If only Rxx is nonzero: I = Rxx, Q = Rxx, U = 0."""
        ntimes, nfreqs = 1, 4
        Rmat = np.zeros((ntimes, 3, nfreqs))
        Rmat[:, 0, :] = 7.0  # Rxx
        I, Q, U = ld.Simulator.compute_stokes(Rmat)
        np.testing.assert_allclose(I, 7.0)
        np.testing.assert_allclose(Q, 7.0)
        np.testing.assert_allclose(U, 0.0, atol=1e-14)

    def test_stokes_I_positive(self):
        """Stokes I should always be non-negative if Rxx, Ryy >= 0."""
        rng = np.random.default_rng(0)
        Rmat = np.abs(rng.standard_normal((5, 3, 10)))
        I, _, _ = ld.Simulator.compute_stokes(Rmat)
        assert np.all(I >= 0)


class TestSimulateStep:
    def test_isotropic_unpolarized_sky(self, short_dipole):
        """An isotropic, unpolarized sky should produce I > 0 and
        small Q, U."""
        beam = short_dipole
        npix = 12 * NSIDE**2
        nfreqs = 4
        I_map = np.ones((nfreqs, npix))
        Q_map = np.zeros((nfreqs, npix))
        U_map = np.zeros((nfreqs, npix))

        sky = ld.SkyModel(
            I_map, Q_map, U_map,
            np.zeros(npix), freq=np.linspace(29, 31, nfreqs),
            frame="galactic", faraday_rotated=True,
        )
        cfg = ld.SimConfig(
            freqs=np.linspace(29, 31, nfreqs),
            times=np.array([0]),  # dummy
            sky=sky, beam=beam, nside=NSIDE,
        )
        sim = ld.Simulator(cfg)
        vis = sim.simulate_step(I_map, Q_map, U_map)

        # vis shape is (3, nfreqs)
        assert vis.shape == (3, nfreqs)
        # Rxx and Ryy should be positive
        assert np.all(vis[0] > 0)  # Rxx
        assert np.all(vis[1] > 0)  # Ryy
        # For unpolarized isotropic sky with symmetric crossed dipoles,
        # Rxx ≈ Ryy so Stokes Q ≈ 0
        Rmat = vis[np.newaxis, :, :]
        I_s, Q_s, U_s = ld.Simulator.compute_stokes(Rmat)
        assert np.all(np.abs(Q_s / I_s) < 0.05)
        assert np.all(np.abs(U_s / I_s) < 0.05)

    def test_normalization_sum_to_one(self, short_dipole):
        """For a uniform I=1 sky, Rxx + Ryy should equal 1.0 after
        normalization (since the beam is normalized by the total
        intensity weight above horizon)."""
        beam = short_dipole
        npix = 12 * NSIDE**2
        nfreqs = 1
        I_map = np.ones((nfreqs, npix))
        Q_map = np.zeros((nfreqs, npix))
        U_map = np.zeros((nfreqs, npix))

        sky = ld.SkyModel(
            I_map, Q_map, U_map,
            np.zeros(npix), freq=30.0,
            frame="galactic", faraday_rotated=True,
        )
        cfg = ld.SimConfig(
            freqs=np.array([30.0]),
            times=np.array([0]),
            sky=sky, beam=beam, nside=NSIDE,
        )
        sim = ld.Simulator(cfg)
        vis = sim.simulate_step(I_map, Q_map, U_map)

        Rxx_plus_Ryy = vis[0, 0] + vis[1, 0]
        assert Rxx_plus_Ryy == pytest.approx(1.0, rel=0.01)
