"""Frequency compression against independent spectral modes and refinement."""

import numpy as np
import pytest

from lusee_faraday.frequency_covariance import (
    gaussian_visibility_draw,
    interpolated_channel_operator,
    project_real_frequency_kernel,
)


def test_interpolation_is_exact_for_affine_process_and_preserves_gain():
    fine = np.linspace(29.9, 30.1, 151)
    nodes = np.array([29.9, 29.96, 30.04, 30.1])
    w = np.random.default_rng(701).uniform(size=(3, fine.size))
    w /= w.sum(axis=1, keepdims=True)
    compressed = interpolated_channel_operator(fine, nodes, w)
    np.testing.assert_allclose(compressed.sum(axis=1), 1, atol=2e-15)
    for slope in (-2, 0, 3):
        np.testing.assert_allclose(
            compressed @ (1 + slope * (nodes - 30)),
            w @ (1 + slope * (fine - 30)),
            atol=2e-15,
        )


def test_compressed_covariance_refines_to_direct_nonstationary_modes():
    fine = np.linspace(29.9, 30.1, 257)
    w = np.random.default_rng(702).uniform(size=(4, fine.size))
    w /= w.sum(axis=1, keepdims=True)

    def modes(f):
        t = f - 30
        return np.stack(
            [
                np.column_stack([np.ones_like(t), 1 + 2 * t]),
                np.column_stack([np.sin(41 * t), np.cos(29 * t)]),
            ],
            axis=-1,
        )

    def kernel(fi, fj):
        return np.einsum("ipr,jqr->ijpq", modes(fi), modes(fj))

    measured_modes = np.einsum("bi,ipr->bpr", w, modes(fine))
    expected = np.einsum("bpr,cqr->bpcq", measured_modes, measured_modes)
    errors = []
    for count in (17, 33, 65):
        nodes = np.linspace(fine[0], fine[-1], count)
        compressed = interpolated_channel_operator(fine, nodes, w)
        result = project_real_frequency_kernel(nodes, compressed, kernel, 7)
        errors.append(np.max(abs(result - expected)))
        eig = np.linalg.eigvalsh(result.reshape(8, 8))
        assert eig.min() > -2e-14
    assert errors[2] < errors[1] / 3 < errors[0] / 9
    assert errors[2] < 3e-4


def test_gaussian_draw_reproducibility_covariance_and_psd_rejection():
    # Independent 2D analytic square root, tested with two seed coordinates.
    c = np.array([[4.0, 1.0], [1.0, 1.0]])
    draws = np.array(
        [gaussian_visibility_draw(c, seed)["sample"] for seed in range(3000)]
    )
    np.testing.assert_allclose(draws.T @ draws / len(draws), c, atol=0.16)
    np.testing.assert_array_equal(
        gaussian_visibility_draw(c, 9)["sample"],
        gaussian_visibility_draw(c, 9)["sample"],
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        gaussian_visibility_draw(np.array([[1.0, 2.0], [2.0, 1.0]]), 9)
    with pytest.raises(ValueError, match="symmetric"):
        gaussian_visibility_draw(np.array([[1.0, 0.2], [0.0, 1.0]]), 9)


def test_invalid_compression_coordinates_and_blocks():
    with pytest.raises(ValueError):
        interpolated_channel_operator([30.0, 31.0], [30.1, 31.0], [[1.0, 1.0]])
    with pytest.raises(ValueError):
        project_real_frequency_kernel(
            [30.0], [[1.0]], lambda a, b: np.ones((1, 1)), 1
        )
    with pytest.raises(ValueError):
        project_real_frequency_kernel([30.0], [[1j]], None)
