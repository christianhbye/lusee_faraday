"""Coefficient draws retain complex moments and tiny common-fit residuals."""

import numpy as np
from lusee_faraday.visibility_draws import draw_depth_coefficients


class IndependentComplexModes:
    degree = 2
    size = 2

    def __init__(self):
        rng = np.random.default_rng(7)
        b = rng.normal(size=(3, 4, 4))
        self.covariance = b @ b.swapaxes(-1, -2)

    def moments(self, a):
        result = np.zeros((len(a), 2, len(a), 2))
        smooth = np.zeros_like(result)
        for k in range(3):
            mixing = np.concatenate(
                [
                    a[:, k, None, None].real * np.eye(2),
                    a[:, k, None, None].imag * np.eye(2),
                ],
                axis=-1,
            )
            c = np.einsum(
                "bpi,ij,cqj->bpcq", mixing, self.covariance[k], mixing
            )
            result += c
            if k == 0:
                smooth += c
        return dict(total=result, smooth=smooth, speckle=result - smooth)


def test_complex_mode_draw_matches_independent_latent_covariance():
    projector = IndependentComplexModes()
    a = np.array(
        [
            [1 + 0.2j, 0.2 + 0.7j, -0.3j],
            [0.4 - 0.5j, -0.6j, 0.7 + 0.4j],
            [-0.2 + 0.4j, 0.8, 0.1 + 0.7j],
        ]
    )
    result = draw_depth_coefficients(projector, a, 130, draws=50000)
    samples = result["sample"].reshape(50000, -1)
    actual = np.cov(samples, rowvar=False)
    expected = projector.moments(a)["total"].reshape(6, 6)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert np.max(abs(actual - expected) / scale) < 0.03


def test_same_latent_modes_commute_with_fit_and_preserve_small_residuals():
    projector = IndependentComplexModes()
    x = np.linspace(-1, 1, 9)
    a = np.stack(
        [
            1 + 0.4j + x * 0.1,
            0.2j + x * 0.3 + 1e-9 * x * x,
            -0.2 + x * 0.1j + 2e-9 * x * x,
        ],
        axis=1,
    )
    basis = np.c_[np.ones(len(x)), x]
    r = np.eye(len(x)) - basis @ np.linalg.pinv(basis)
    raw = draw_depth_coefficients(projector, a, 44)["sample"][0]
    filtered = draw_depth_coefficients(projector, r @ a, 44)["sample"][0]
    np.testing.assert_allclose(r @ raw, filtered, atol=3e-15, rtol=1e-6)
    assert np.max(abs(filtered)) > 1e-10
