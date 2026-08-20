import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine  # noqa: E402

LMAX = 5
L = LMAX + 1
M = 2 * LMAX + 1


def random_alm(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_shapes():
    rng = np.random.default_rng(0)
    beam = random_alm(rng, (10, 4, L, M))
    sky = random_alm(rng, (3, 4, L, M))
    phases = random_alm(rng, (7, M))
    W = engine.contract_blocks(beam, sky, phases)
    assert W.shape == (3, 4, 7, 10)


def test_block_sum_reproduces_croissant_polarized_convolve():
    """The one contract we own must agree with the library's."""
    cro = pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(1)
    n_components = 3
    beam = random_alm(rng, (10, 4, L, M))
    sky = random_alm(rng, (n_components, 4, L, M))
    phases = random_alm(rng, (7, M))

    ours = engine.contract_blocks(beam, sky, phases).sum(axis=1)

    # croissant pairs sky frequency f with beam frequency f, so tile the
    # single-frequency beam across our component axis.
    beam_tiled = np.broadcast_to(
        beam[:, None], (10, n_components, 4, L, M)
    ).copy()
    theirs = np.asarray(
        cro.polarized_convolve(beam_tiled, sky, phases)
    )  # (t, p, f)

    assert np.allclose(
        ours, np.transpose(theirs, (2, 0, 1)), rtol=1e-12, atol=1e-12
    )


def test_contraction_is_linear_in_the_sky():
    rng = np.random.default_rng(2)
    beam = random_alm(rng, (10, 4, L, M))
    a = random_alm(rng, (1, 4, L, M))
    b = random_alm(rng, (1, 4, L, M))
    phases = random_alm(rng, (4, M))
    both = engine.contract_blocks(beam, np.concatenate([a, b]), phases)
    combined = engine.contract_blocks(beam, 2.0 * a + 3.0 * b, phases)
    assert np.allclose(combined[0], 2.0 * both[0] + 3.0 * both[1])
