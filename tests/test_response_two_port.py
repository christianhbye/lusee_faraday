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


def test_grid_length_mismatch_is_rejected():
    """theta_deg/phi_deg must describe h_theta's actual grid, not
    just satisfy the mwss relation on their own."""
    pytest.importorskip("croissant")

    theta_deg = np.arange(0.0, 178.0, 2.0)  # 89 points, not 91
    phi_deg = np.arange(0.0, 360.0, 2.0)  # 180 points, matches data
    ht = np.zeros((2, 91, len(phi_deg)), dtype=complex)
    hp = np.zeros_like(ht)
    with pytest.raises(ValueError, match="do not describe"):
        rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, 8)


@pytest.mark.parametrize("step_deg", [1.0, 2.0])
def test_short_dipole_monopole_matches_analytic_integral(step_deg):
    """Pin the absolute scale, not just XX == YY.

    I_xx = 1 - sin(theta)^2 cos(phi)^2 integrates to 4*pi/3 over the
    upper hemisphere; Y_00 = 1/sqrt(4*pi). Symmetry gives the same
    integral for YY. Half weight on an equatorial row makes this even
    intensity map's monopole exactly half its full-sphere monopole.
    Check both the native one-degree grid and the coarser two-degree
    fixture, each of which includes the equator.
    """
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    theta_deg = np.arange(0.0, 180.0 + step_deg, step_deg)
    phi_deg = np.arange(0.0, 360.0, step_deg)
    ht, hp = analytic_short_dipoles(theta_deg, phi_deg)
    lmax = 8
    alms = rsp.two_port_pair_alms(ht, hp, theta_deg, phi_deg, lmax)
    autos = alms[[0, 2], 0, 0, lmax]
    reference = (4 * np.pi / 3) / np.sqrt(4 * np.pi)
    np.testing.assert_allclose(autos, reference, rtol=1e-9, atol=0.0)
