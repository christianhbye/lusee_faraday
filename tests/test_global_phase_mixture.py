"""Law of total covariance in the production shared-scale combination."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from residual_signal import mixture_moments  # noqa: E402


class IndependentModes:
    def moments(self, a):
        smooth = np.outer(a[:, 0], a[:, 0])[:, None, :, None]
        total = (a @ a.T)[:, None, :, None]
        return dict(smooth=smooth, speckle=total - smooth, total=total)


def test_global_scale_mean_variation_matches_independent_finite_ensemble():
    values = [
        np.array([[1.0, 0.2], [2.0, -0.4]]),
        np.array([[0.1, 0.7], [-0.3, 0.8]]),
    ]
    probabilities = [0.3, 0.7]
    solvers = [IndependentModes(), IndependentModes()]
    actual = mixture_moments(solvers, values, probabilities)
    mean_coefficient = sum(p * a[:, 0] for p, a in zip(probabilities, values))
    smooth, speckle, total = [np.zeros((2, 2)) for _ in range(3)]
    # Independent Rademacher modes exactly match all required second
    # moments; integrate over the global mixture and both modes directly.
    for p, a in zip(probabilities, values):
        for z0 in (-1.0, 1.0):
            for z1 in (-1.0, 1.0):
                y = a[:, 0] * z0 + a[:, 1] * z1
                m = mean_coefficient * z0
                smooth += p * np.outer(m, m) / 4
                speckle += p * np.outer(y - m, y - m) / 4
                total += p * np.outer(y, y) / 4
    for name, expected in zip(
        ("smooth", "speckle", "total"), (smooth, speckle, total)
    ):
        np.testing.assert_allclose(
            actual[name][:, 0, :, 0], expected, rtol=1e-14
        )
