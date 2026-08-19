import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import polarimeter as pol  # noqa: E402


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = lusee.synthetic_four_port_response(
        freq_mhz=(10.0, 20.0), angular_step_deg=5.0
    )
    return resp, JFETReceiver()


def test_pseudo_stokes_of_a_pure_x_state():
    vx = np.array([0.0, 1.0, 0.0, -1.0])
    C = np.einsum("a,b->ab", vx, vx).astype(complex)
    I, Q, U, V = pol.pseudo_stokes(C)
    assert np.isclose(I, 2.0) and np.isclose(Q, 2.0)
    assert np.isclose(U, 0.0) and np.isclose(V, 0.0)


def test_pseudo_stokes_of_a_pure_u_state():
    v = pol.X_VEC + pol.Y_VEC
    C = np.einsum("a,b->ab", v, v).astype(complex)
    I, Q, U, V = pol.pseudo_stokes(C)
    assert np.isclose(Q, 0.0, atol=1e-12)
    assert np.isclose(U, I)
    assert np.isclose(V, 0.0, atol=1e-12)


def test_check_psd_accepts_physical_and_rejects_unphysical():
    pol.check_psd(np.array([1.0, 0.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match="PSD"):
        pol.check_psd(np.array([1.0, 0.9, 0.9, 0.0]))


def test_ortho_vectors_null_the_zenith_polarization(pieces):
    resp, receiver = pieces
    x, y, C0 = pol.zenith_vectors(resp, receiver, 10.0, mode="ortho")
    I, Q, U, V = pol.pseudo_stokes(C0, x, y)
    assert abs(Q) < 1e-12 * I
    assert abs(U) < 1e-12 * I
    assert abs(V) < 1e-12 * I


def test_gains_mode_nulls_q_but_not_necessarily_u(pieces):
    resp, receiver = pieces
    x, y, C0 = pol.zenith_vectors(resp, receiver, 10.0, mode="gains")
    I, Q, _, _ = pol.pseudo_stokes(C0, x, y)
    assert abs(Q) < 1e-12 * I


def test_zenith_vectors_reject_an_unknown_mode(pieces):
    resp, receiver = pieces
    with pytest.raises(ValueError, match="mode"):
        pol.zenith_vectors(resp, receiver, 10.0, mode="magic")
