"""Tests verifying the optimized FR simulation matches brute force.

The optimized approach (fast_sim) applies coordinate rotation to
reference maps once, then does Faraday rotation in topocentric frame.
The brute-force approach (Simulator) applies FR in Galactic frame then
rotates every frequency channel.  These agree because Faraday rotation
and parallel transport are both SO(2) on (Q, U) and thus commute.

With spatially varying RM, the brute-force approach introduces SHT
truncation error (FR creates high-ell structure), while the optimized
approach applies FR in pixel space (exact).  So discrepancies at
moderate/high FR angles are dominated by SHT truncation in the
brute-force result, and the optimized result is actually more accurate.
"""

import healpy as hp
import numpy as np
import pytest
import astropy.units as u
from lunarsky import Time, MoonLocation

from lunarsky import LunarTopo

import lusee_faraday as ld
from lusee_faraday.fast_sim import (
    precompute_rotated_maps,
    compute_vis_fast,
)
from lusee_faraday.sim import Simulator, SimConfig
from lusee_faraday import config as _cfg
from lusee_faraday.rotations import (
    get_rot_mat,
    rotmat_to_eulerZYX,
)

NSIDE = 32

# sky.py no longer defines this (replaced by FaradaySky in Task 8);
# this old fast_sim pipeline is retired in Task 18, so restore the
# constant locally rather than touching sky.py's new public surface.
LUSEE_LOC = _cfg.moon_location()


@pytest.fixture
def beam():
    b = ld.Beam.short_dipole(nside=NSIDE)
    b.precompute_weights()
    return b


@pytest.fixture
def times():
    loc = MoonLocation(lat=-23.813, lon=182.258)
    t0 = Time("2027-01-01T09:00:00", location=loc)
    return np.array(
        [t0, t0 + 200 * 3600 * u.s, t0 + 400 * 3600 * u.s]
    )


def _run_brute_force(I_ref, Q_ref, U_ref, rm, beam, times, freqs,
                     nside=NSIDE):
    """Run simulation using the standard Simulator (FR in Galactic)."""
    I_map = ld.sky.scale_haslam(I_ref, freqs)
    QU = ld.sky.power_law(
        np.array([Q_ref, U_ref]), freqs, 23e3, beta=-2.8
    )
    Q_map, U_map = QU[:, 0], QU[:, 1]
    sky = ld.SkyModel(
        I_map, Q_map, U_map, rm,
        freq=freqs, frame="galactic",
    )
    cfg = SimConfig(
        freqs=freqs, times=times, sky=sky, beam=beam,
        nside=nside, faraday=True,
    )
    sim = Simulator(cfg)
    return sim.simulate()


def _run_fast(I_ref, Q_ref, U_ref, rm, beam, times, freqs,
              nside=NSIDE):
    """Run simulation using the optimized approach."""
    I_topo, Q_topo, U_topo, rm_topo = precompute_rotated_maps(
        I_ref, Q_ref, U_ref, rm, times, nside, LUSEE_LOC,
    )
    mask = ld.HealpixGrid(nside, horizon=True).mask
    return compute_vis_fast(
        I_topo, Q_topo, U_topo, rm_topo, beam, freqs, mask,
    )


# ------------------------------------------------------------------
# Sanity check: scalar RM rotation matches I rotation
# ------------------------------------------------------------------

def _make_smooth_map(nside, lmax_signal=20, seed=7):
    """Create a smooth (band-limited) map for testing."""
    lmax = 3 * nside - 1
    nalm = hp.Alm.getsize(lmax)
    np.random.seed(seed)
    alm = np.zeros(nalm, dtype=complex)
    for ell in range(0, min(lmax_signal + 1, lmax + 1)):
        for m in range(0, ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            alm[idx] = np.random.randn() + 1j * np.random.randn()
    return hp.alm2map(alm, nside, lmax=lmax)


def test_rm_rotation_preserves_constant():
    """rotate_map_pixel should preserve a constant map exactly."""
    npix = hp.nside2npix(NSIDE)
    rm_const = np.full(npix, 5.0)

    loc = MoonLocation(lat=-23.813, lon=182.258)
    t = Time("2027-01-15T12:00:00", location=loc)
    topo = LunarTopo(location=LUSEE_LOC, obstime=t)
    euler = rotmat_to_eulerZYX(get_rot_mat(topo))
    rot = hp.Rotator(rot=euler, deg=False, eulertype="ZYX")

    rm_rot = rot.rotate_map_pixel(rm_const)
    np.testing.assert_allclose(rm_rot, 5.0, atol=1e-10)


# ------------------------------------------------------------------
# Zero RM: no Faraday effect, verifies I + QU scaling
# ------------------------------------------------------------------

def test_zero_rm(beam, times):
    """With RM=0, FR is a no-op; both approaches should match tightly."""
    npix = hp.nside2npix(NSIDE)
    np.random.seed(42)
    I_ref = 1000 * np.random.uniform(0.5, 1.5, npix)
    Q_ref = 10 * np.random.uniform(-1, 1, npix)
    U_ref = 10 * np.random.uniform(-1, 1, npix)
    rm = np.zeros(npix)

    freqs = np.array([30.0, 40.0, 50.0])

    vis_brute = _run_brute_force(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )
    vis_fast = _run_fast(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )

    np.testing.assert_allclose(vis_fast, vis_brute, rtol=1e-5)


# ------------------------------------------------------------------
# Constant RM: exact commutativity (no SHT truncation issue)
# ------------------------------------------------------------------

def test_constant_rm(beam, times):
    """With constant RM, FR is a global rotation on (Q, U) which
    commutes exactly with SHT-based coordinate rotation."""
    npix = hp.nside2npix(NSIDE)
    np.random.seed(42)
    I_ref = 1000 * np.random.uniform(0.5, 1.5, npix)
    Q_ref = 10 * np.random.uniform(-1, 1, npix)
    U_ref = 10 * np.random.uniform(-1, 1, npix)
    rm = np.full(npix, 5.0)

    freqs = np.array([20.0, 30.0, 40.0, 50.0])

    vis_brute = _run_brute_force(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )
    vis_fast = _run_fast(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )

    np.testing.assert_allclose(vis_fast, vis_brute, rtol=1e-5)


# ------------------------------------------------------------------
# Varying RM, weak FR: small Faraday angles, tight agreement
# ------------------------------------------------------------------

def test_varying_rm_weak(times):
    """Smooth RM map with small amplitude at nside=64 for better
    convergence between pixel-interpolation and SHT approaches."""
    ns = 64
    npix = hp.nside2npix(ns)
    beam64 = ld.Beam.short_dipole(nside=ns)
    beam64.precompute_weights()

    np.random.seed(123)
    I_ref = 500 * np.random.uniform(0.5, 1.5, npix)
    Q_ref = 5 * np.random.uniform(-1, 1, npix)
    U_ref = 5 * np.random.uniform(-1, 1, npix)
    rm = _make_smooth_map(ns, lmax_signal=15, seed=200) * 0.01

    freqs = np.array([48.0, 50.0])

    vis_brute = _run_brute_force(
        I_ref, Q_ref, U_ref, rm, beam64, times, freqs, nside=ns,
    )
    vis_fast = _run_fast(
        I_ref, Q_ref, U_ref, rm, beam64, times, freqs, nside=ns,
    )

    # Remaining discrepancy from pixel-interp vs SHT for RM map
    np.testing.assert_allclose(vis_fast, vis_brute, rtol=0.15)


# ------------------------------------------------------------------
# Varying RM, moderate FR
# ------------------------------------------------------------------

def test_fr_changes_polarization(beam, times):
    """Self-consistency: FR with nonzero RM should change Q and U
    relative to the no-FR case."""
    npix = hp.nside2npix(NSIDE)
    np.random.seed(789)
    I_ref = 1000 * np.random.uniform(0.5, 1.5, npix)
    Q_ref = 10 * np.random.uniform(-1, 1, npix)
    U_ref = 10 * np.random.uniform(-1, 1, npix)
    rm = np.full(npix, 3.0)

    freqs = np.array([30.0, 40.0, 50.0])

    vis_fr = _run_fast(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )
    vis_nofr = _run_fast(
        I_ref, Q_ref, U_ref, np.zeros(npix), beam, times, freqs,
    )

    sI_fr, sQ_fr, sU_fr = Simulator.compute_stokes(vis_fr)
    sI_nf, sQ_nf, sU_nf = Simulator.compute_stokes(vis_nofr)

    # FR should change Q and U
    assert not np.allclose(sQ_fr, sQ_nf, rtol=0.01)
    assert not np.allclose(sU_fr, sU_nf, rtol=0.01)

    # The change should be frequency-dependent (different FR angle)
    dQ = sQ_fr - sQ_nf
    assert not np.allclose(dQ[:, 0], dQ[:, -1], rtol=0.01)


# ------------------------------------------------------------------
# Stokes extraction: verify compute_stokes gives same I, Q, U
# ------------------------------------------------------------------

def test_stokes_output(beam, times):
    """End-to-end: compare Stokes I, Q, U from both approaches."""
    npix = hp.nside2npix(NSIDE)
    np.random.seed(99)
    I_ref = 800 * np.random.uniform(0.5, 1.5, npix)
    Q_ref = 8 * np.random.uniform(-1, 1, npix)
    U_ref = 8 * np.random.uniform(-1, 1, npix)
    rm = np.full(npix, 2.0)

    freqs = np.array([25.0, 35.0, 45.0])

    vis_brute = _run_brute_force(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )
    vis_fast = _run_fast(
        I_ref, Q_ref, U_ref, rm, beam, times, freqs,
    )

    sI_b, sQ_b, sU_b = Simulator.compute_stokes(vis_brute)
    sI_f, sQ_f, sU_f = Simulator.compute_stokes(vis_fast)

    np.testing.assert_allclose(sI_f, sI_b, rtol=1e-5)
    np.testing.assert_allclose(sQ_f, sQ_b, rtol=1e-5)
    np.testing.assert_allclose(sU_f, sU_b, rtol=1e-5)
