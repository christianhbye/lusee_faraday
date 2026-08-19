import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import response as rsp  # noqa: E402


def analytic_short_dipoles(theta_deg, phi_deg):
    """X and Y short dipoles on a theta/phi grid, upper hemisphere only."""
    th = np.radians(theta_deg)[:, None]
    ph = np.radians(phi_deg)[None, :]
    below = np.cos(th) < 0
    hx_t = -np.cos(th) * np.cos(ph) * ~below
    hx_p = np.sin(ph) * np.ones_like(th) * ~below
    hy_t = -np.cos(th) * np.sin(ph) * ~below
    hy_p = -np.cos(ph) * np.ones_like(th) * ~below
    return (
        np.stack([hx_t, hy_t]).astype(complex),
        np.stack([hx_p, hy_p]).astype(complex),
    )


def test_two_port_pair_alms_shape():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    lmax = 8
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, lmax)
    assert alms.shape == (3, 4, lmax + 1, 2 * lmax + 1)
    assert rsp.TWO_PORT_PAIRS == ((0, 0), (0, 1), (1, 1))


def test_short_dipole_autos_have_equal_total_power():
    """X and Y are the same dipole rotated, so their I monopoles match."""
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, 8)
    lmax = 8
    xx = alms[0, 0, 0, lmax]
    yy = alms[2, 0, 0, lmax]
    assert np.isclose(xx, yy, rtol=1e-6)


def test_short_dipole_cross_monopole_vanishes():
    """Orthogonal dipoles have no monopole in their cross response."""
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 181.0, 2.0)
    phi_deg = np.arange(0.0, 360.0, 2.0)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    lmax = 8
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, lmax)
    xy = alms[1, 0, 0, lmax]
    xx = alms[0, 0, 0, lmax]
    assert abs(xy) < 1e-8 * abs(xx)


def test_mismatched_mwss_grid_is_rejected():
    """croissant accepts a bad "mwss" grid silently, so response.py
    must catch nphi != 2*(ntheta - 1) itself."""
    pytest.importorskip("croissant")

    theta_deg = np.arange(0.0, 181.0, 2.0)  # 91 points
    phi_deg = np.arange(0.0, 180.0, 2.0)  # 90 points, should be 180
    ht = np.zeros((2, len(theta_deg), len(phi_deg)), dtype=complex)
    hp = np.zeros_like(ht)
    with pytest.raises(ValueError, match="mwss sampling relation"):
        rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, 8)
