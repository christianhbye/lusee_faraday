"""Marginal projection agrees with complete physical complex covariances."""

import numpy as np
from test_depth_channel_covariance import real_projector
from lusee_faraday.depth_channel_covariance import DepthChannelProjector
from lusee_faraday.moment_diagonal import diagonal_moments, mixture_diagonals
from lusee_faraday.spin_covariance import AngularScreen


def test_marginal_variance_retains_all_four_complex_branches():
    solver = DepthChannelProjector(
        real_projector(), AngularScreen(theta0_rad=0.3), 4
    )
    rng = np.random.default_rng(411)
    a = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    actual = diagonal_moments(solver, a)
    full = solver.moments(a)
    for name in full:
        expected = np.diag(full[name].reshape(3 * solver.size, -1)).reshape(
            3, -1
        )
        np.testing.assert_allclose(
            actual[name], expected, rtol=3e-14, atol=3e-14
        )


def test_mixture_marginals_include_conditional_mean_variation():
    solver = DepthChannelProjector(
        real_projector(), AngularScreen(theta0_rad=0.3), 4
    )
    rng = np.random.default_rng(412)
    values = [
        rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
        for _ in range(2)
    ]
    actual = mixture_diagonals(solver, values, [0.3, 0.7])
    total = sum(
        p * solver.moments(a)["total"] for p, a in zip([0.3, 0.7], values)
    )
    mean = np.zeros_like(values[0])
    mean[:, 0] = 0.3 * values[0][:, 0] + 0.7 * values[1][:, 0]
    smooth = solver.moments(mean)["smooth"]
    full = dict(total=total, smooth=smooth, speckle=total - smooth)
    for name in full:
        expected = np.diag(full[name].reshape(3 * solver.size, -1)).reshape(
            3, -1
        )
        np.testing.assert_allclose(
            actual[name], expected, rtol=3e-14, atol=3e-14
        )
