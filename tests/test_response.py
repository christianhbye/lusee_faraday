import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import response as rsp  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    return lusee.synthetic_four_port_response(
        freq_mhz=(10.0, 20.0), angular_step_deg=5.0
    )


def test_pair_stokes_from_jones_matches_luseepy(synthetic):
    """Our kernel formula must be luseepy's, not a near miss."""
    want = np.asarray(synthetic.all_pair_stokes_maps())
    got = rsp.pair_stokes_from_jones(
        np.asarray(synthetic.H_theta),
        np.asarray(synthetic.H_phi),
        synthetic.pairs,
    )
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_native_channel_index_rejects_off_grid_frequencies(synthetic):
    assert rsp.native_channel_index(synthetic, 20.0) == 1
    with pytest.raises(ValueError, match="native response channel"):
        rsp.native_channel_index(synthetic, 20.01)


def test_four_port_pair_alms_shape_and_pairs(synthetic):
    lmax = 8
    alms = rsp.four_port_pair_alms(synthetic, 10.0, lmax)
    assert alms.shape == (10, 4, lmax + 1, 2 * lmax + 1)
    assert np.iscomplexobj(alms)


def test_four_port_pair_alms_is_the_fixed_beam_slice(synthetic):
    """Two native channels must give different alms; picking one is a
    choice."""
    a10 = rsp.four_port_pair_alms(synthetic, 10.0, 4)
    a20 = rsp.four_port_pair_alms(synthetic, 20.0, 4)
    assert not np.allclose(a10, a20)


def test_four_port_pair_alms_rejects_non_native_frequency(synthetic):
    with pytest.raises(ValueError, match="native response channel"):
        rsp.four_port_pair_alms(synthetic, 15.0, 4)
