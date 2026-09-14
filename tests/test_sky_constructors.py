import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12
BAND = np.array([29.9, 30.1])


@pytest.fixture(scope="module")
def hp_module():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    return hp


def test_point_source_makes_one_component_per_source(hp_module):
    sky = FaradaySky.point_source(
        theta=np.array([0.5, 1.2]),
        phi=np.array([0.0, 2.0]),
        stokes=np.array([[1.0, -1.0, 0.0], [2.0, 0.0, 0.5]]),
        phi_fd=np.array([250.0, -30.0]),
        nside=NSIDE,
        lmax=LMAX,
    )
    assert sky.n_components == 2
    assert np.allclose(sky.phi_fd, [250.0, -30.0])


def test_point_source_components_carry_their_own_faraday_depth(hp_module):
    """Two sources with different phi must rotate at different rates."""
    sky = FaradaySky.point_source(
        theta=np.array([0.5, 1.2]),
        phi=np.array([0.0, 2.0]),
        stokes=np.array([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]),
        phi_fd=np.array([250.0, 0.0]),
        nside=NSIDE,
        lmax=LMAX,
    )
    c = sky.coeffs(np.array([30.0, 30.05]))
    assert not np.allclose(c[0, :, 2], c[0, 0, 2])  # source 0 winds
    assert np.allclose(c[1, :, 2], 1.0)  # source 1 does not


def test_binned_screen_partitions_the_sky_exactly(hp_module):
    hp = hp_module
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0  # noqa: E741
    Q = rng.normal(size=npix) * 0.1
    U = rng.normal(size=npix) * 0.1
    rm = rng.uniform(-40.0, 40.0, size=npix)

    # allow_pixelwise: this screen is deliberately unresolved at
    # nside=16 (the point here is the partition algebra, not the
    # physics), and binned_screen now refuses such a screen unless the
    # caller opts in.  test_sky_diagnostics pins the refusal itself.
    sky = FaradaySky.binned_screen(
        I,
        Q,
        U,
        rm,
        dphi=10.0,
        lmax=LMAX,
        freqs_mhz=BAND,
        allow_pixelwise=True,
    )
    assert sky.n_components == 8  # (-40, 40) in steps of 10

    # With every phi bin forced to zero depth the sum of the components
    # must be the alm of the unpartitioned sky.
    whole = FaradaySky.uniform_screen(I, Q, U, phi_fd=0.0, lmax=LMAX)
    summed = sky.component_alms.sum(axis=0)
    scale = np.abs(whole.component_alms[0]).max()
    assert np.abs(summed - whole.component_alms[0]).max() < 1e-10 * scale


def test_binned_screen_assigns_each_component_its_bin_centre(hp_module):
    hp = hp_module
    npix = hp.nside2npix(NSIDE)
    rm = np.full(npix, 7.0)
    rm[: npix // 2] = -23.0
    I = np.ones(npix)  # noqa: E741
    z = np.zeros(npix)
    sky = FaradaySky.binned_screen(
        I,
        z,
        z,
        rm,
        dphi=10.0,
        lmax=LMAX,
        freqs_mhz=BAND,
        allow_pixelwise=True,  # a 30 rad/m^2 step between neighbours
    )
    assert sky.n_components == 2
    assert np.allclose(np.sort(sky.phi_fd), [-23.0, 7.0])


def test_binned_screen_rejects_a_nonpositive_bin_width(hp_module):
    hp = hp_module
    npix = hp.nside2npix(NSIDE)
    ones = np.ones(npix)
    with pytest.raises(ValueError, match="dphi"):
        FaradaySky.binned_screen(
            ones, ones, ones, ones, dphi=0.0, lmax=LMAX, freqs_mhz=BAND
        )
