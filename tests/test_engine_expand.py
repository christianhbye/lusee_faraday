import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine  # noqa: E402

LMAX = 4
L = LMAX + 1
M = 2 * LMAX + 1


def random_alm(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_expand_matches_the_reference_einsum():
    rng = np.random.default_rng(0)
    W = random_alm(rng, (3, 4, 6, 10))
    coeffs = random_alm(rng, (3, 11, 4))
    want = np.einsum("kctp,kfc->tfp", W, coeffs)
    assert np.allclose(engine.expand(W, coeffs), want)


def test_expand_is_chunk_invariant():
    rng = np.random.default_rng(1)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (2, 17, 4))
    full = engine.expand(W, coeffs)
    for chunk in (1, 4, 16, 64):
        assert np.allclose(engine.expand(W, coeffs, chunk=chunk), full)


def test_expand_writes_into_a_preallocated_output(tmp_path):
    rng = np.random.default_rng(2)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (2, 9, 4))
    path = tmp_path / "out.dat"
    out = np.memmap(path, dtype=np.complex128, mode="w+", shape=(5, 9, 10))
    engine.expand(W, coeffs, chunk=3, out=out)
    out.flush()
    assert np.allclose(np.asarray(out), engine.expand(W, coeffs))


def test_expand_rejects_mismatched_component_counts():
    rng = np.random.default_rng(3)
    W = random_alm(rng, (2, 4, 5, 10))
    coeffs = random_alm(rng, (3, 9, 4))
    with pytest.raises(ValueError, match="component"):
        engine.expand(W, coeffs)


def test_contract_of_an_isotropic_sky_is_time_independent():
    """A monopole sky is rotation invariant, so W must not vary with time."""
    pytest.importorskip("croissant")
    pytest.importorskip("lunarsky")
    import jax

    jax.config.update("jax_enable_x64", True)

    from lusee_faraday import config as cfg

    rng = np.random.default_rng(4)
    beam = random_alm(rng, (10, 4, L, M))
    sky = np.zeros((1, 4, L, M), dtype=complex)
    sky[0, 0, 0, LMAX] = 1.0  # I monopole only, m = 0

    times = cfg.times()[:4]
    W = engine.contract(beam, sky, times, cfg.moon_location(), LMAX)
    assert W.shape == (1, 4, 4, 10)
    spread = np.abs(W - W[:, :, :1]).max()
    assert spread < 1e-10 * np.abs(W).max()
