"""Independent analytic controls and explicit mutation rejection."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from lusee_faraday import gaussian_screen as gs
from lusee_faraday.screen_draws import FourierScreen, spectral_covariance


def test_joint_qu_uniform_reference():
    freq = np.array([45.0, 47.5, 50.0])
    model = gs.ScreenModel(correlation_pc=np.inf)
    z, weight = hermgauss(64)
    weight /= np.sqrt(np.pi)
    depth = model.mean_depth + np.sqrt(2) * model.sigma_depth * z
    from lusee_faraday.conventions import faraday_phase_cosmo

    amp = model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    pol = (
        amp
        * np.exp(2j * np.deg2rad(model.angle_deg))
        * faraday_phase_cosmo(depth, freq)
    )
    values = np.concatenate([pol.real, pol.imag], axis=1)
    mean = weight @ values
    cov = (values - mean).T @ (weight[:, None] * (values - mean))
    result = gs.qu_moments(freq, model)
    np.testing.assert_allclose(result["mean"], mean, atol=2e-13)
    np.testing.assert_allclose(result["covariance"], cov, atol=2e-13)
    np.testing.assert_allclose(
        result["second"], values.T @ (weight[:, None] * values), atol=2e-13
    )
    delta = values - result["off"]
    np.testing.assert_allclose(
        result["delta_second"], delta.T @ (weight[:, None] * delta), atol=2e-13
    )


def test_qu_special_matches_centered_quadrature():
    for ell in (0.05, 0.5, 5):
        model = gs.ScreenModel(correlation_pc=ell)
        freq = np.array([10.0, 25.0, 45.0, 50.0])
        a = gs.qu_moments(freq, model)
        b = gs.qu_moments(freq, model, method="special")
        np.testing.assert_allclose(
            a["covariance"], b["covariance"], atol=1e-12, rtol=1e-8
        )
    zero = gs.qu_moments(freq, replace(model, sigma_depth=0))
    assert np.max(abs(zero["covariance"])) == 0


def test_fourier_known_modes_and_refinement(tmp_path):
    c = np.zeros((3, 3), complex)
    c[2, 1] = 0.03 + 0.02j
    c[0, 1] = c[2, 1].conj()
    screen = FourierScreen(4.0, c, 0.01, 2.0)
    x = (np.arange(16) - 8) * 4 / 16
    expected = 0.01 + 2 * np.real(c[2, 1] * np.exp(2j * np.pi * x / 4))
    np.testing.assert_allclose(
        screen.evaluate(16),
        np.broadcast_to(expected[:, None], (16, 16)),
        atol=1e-16,
    )
    np.testing.assert_allclose(
        screen.evaluate(32)[::2, ::2], screen.evaluate(16), atol=1e-16
    )
    screen.save(tmp_path / "field.npz")
    np.testing.assert_array_equal(
        FourierScreen.load(tmp_path / "field.npz").coefficients, c
    )
    assert not screen.coefficients.flags.writeable
    np.testing.assert_allclose(screen.truncate(0.5).evaluate(16), 0.01)


def test_generator_covariance_and_gaussian_ensemble():
    model = gs.ScreenModel(correlation_pc=0.5)
    h = np.array([0.0, 0.5, 1.0])
    expected = model.sigma_depth**2 * np.exp(-0.5 * (h / 0.5) ** 2)
    np.testing.assert_allclose(
        spectral_covariance(model, 8.0, 15.0, h), expected, atol=1e-14
    )
    # Ensemble statistics, including the random DC mode, no map recentering.
    samples = np.array(
        [
            FourierScreen.draw(model, 8.0, 15.0, seed).evaluate(64)[
                32, [32, 36, 40]
            ]
            for seed in range(512)
        ]
    )
    empirical = np.mean(
        (samples[:, :1] - model.mean_depth) * (samples - model.mean_depth),
        axis=0,
    )
    np.testing.assert_allclose(empirical, expected, atol=8e-5, rtol=0)


def test_mutations_are_rejected(tmp_path):
    original = gs.real_covariance
    with patch.object(
        gs, "real_covariance", lambda c, j: original(c, np.zeros_like(j))
    ):
        with pytest.raises(AssertionError):
            test_joint_qu_uniform_reference()
    with patch.object(
        gs, "centered_branch", lambda x, y, s, t, model, numerical: 0j
    ):
        with pytest.raises(AssertionError):
            test_joint_qu_uniform_reference()
    evaluate = FourierScreen.evaluate
    with patch.object(
        FourierScreen, "evaluate", lambda self, n: evaluate(self, n) / n**2
    ):
        with pytest.raises(AssertionError):
            test_fourier_known_modes_and_refinement(tmp_path)
