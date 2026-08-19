import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import instrument as inst  # noqa: E402


@pytest.fixture(scope="module")
def pieces():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = lusee.synthetic_four_port_response(freq_mhz=(10.0, 20.0))
    return resp, JFETReceiver()


def test_channels_roundtrip_through_unpack():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 4, 4)) + 1j * rng.normal(size=(3, 5, 4, 4))
    C = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    ch, labels = inst.channels(C)
    assert ch.shape == (3, 5, 16)
    assert len(labels) == 16
    assert np.allclose(inst.unpack_channels(ch), C)


def test_channel_labels_match_luseepy():
    lusee_cov = pytest.importorskip("lusee.Covariance")
    rng = np.random.default_rng(1)
    C = np.zeros((1, 1, 4, 4), dtype=complex)
    C[..., 0, 0] = 1.0
    _, labels = inst.channels(C)
    assert labels == lusee_cov.default_product_labels()


def test_covariance_is_hermitian(pieces):
    resp, receiver = pieces
    rng = np.random.default_rng(2)
    freqs = np.array([10.0, 12.0, 20.0])
    pair = rng.normal(size=(4, 3, 10)) + 1j * rng.normal(size=(4, 3, 10))
    C = inst.covariance(pair, resp, receiver, freqs)
    assert C.shape == (4, 3, 4, 4)
    assert np.allclose(C, np.conj(np.swapaxes(C, -1, -2)))


def test_covariance_is_linear_in_the_pair_integrals(pieces):
    """T_moon and T_ant are additive offsets; the sky term must be linear."""
    resp, receiver = pieces
    rng = np.random.default_rng(3)
    freqs = np.array([10.0, 20.0])
    a = rng.normal(size=(2, 2, 10)) + 1j * rng.normal(size=(2, 2, 10))
    b = rng.normal(size=(2, 2, 10)) + 1j * rng.normal(size=(2, 2, 10))
    kw = dict(T_moon=0.0, T_ant=0.0)
    Ca = inst.covariance(a, resp, receiver, freqs, **kw)
    Cb = inst.covariance(b, resp, receiver, freqs, **kw)
    Cab = inst.covariance(2 * a + 3 * b, resp, receiver, freqs, **kw)
    assert np.allclose(Cab, 2 * Ca + 3 * Cb)


def test_blackbody_normalization_shape(pieces):
    resp, receiver = pieces
    freqs = np.array([10.0, 15.0, 20.0])
    B = inst.blackbody_normalization(resp, receiver, freqs)
    assert B.shape == (3, 4, 4)
