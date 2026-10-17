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


def test_likelihood_normalization_and_source_transform():
    from lusee_faraday.screen_inference import gaussian_nll, transform_source
    from scipy.stats import multivariate_normal

    mean = np.array([0.2, -0.4])
    cov = np.array([[2.0, 0.3], [0.3, 0.8]])
    data = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        gaussian_nll(data, mean, cov),
        -multivariate_normal.logpdf(data, mean, cov),
    )
    # At zero residual, a covariance change affects only the normalization.
    np.testing.assert_allclose(
        gaussian_nll(mean, mean, 4 * cov) - gaussian_nll(mean, mean, cov),
        np.log(4),
    )
    freq = np.array([45.0, 47.5, 50.0])
    model = gs.ScreenModel()
    base = gs.qu_moments(freq, model)
    moved = gs.qu_moments(
        freq, replace(model, p_ref_k=0.7, angle_deg=31.0, beta=-2.2)
    )
    m, c = transform_source(
        base["mean"],
        base["covariance"],
        freq,
        model,
        0.7,
        np.deg2rad(11.0),
        -2.2 - model.beta,
    )
    np.testing.assert_allclose(m, moved["mean"], atol=1e-13)
    np.testing.assert_allclose(c, moved["covariance"], atol=1e-13)


def test_likelihood_normalization_mutation():
    from lusee_faraday import screen_inference as si

    original = si.gaussian_nll

    def wrong(data, mean, cov):
        return original(data, mean, cov) - 0.5 * np.linalg.slogdet(cov)[1]

    with patch.object(si, "gaussian_nll", wrong):
        with pytest.raises(AssertionError):
            test_likelihood_normalization_and_source_transform()


def test_profiled_constant_null_recovers_direct_spectrum():
    from lusee_faraday.screen_inference import profile_null_batch
    from lusee_faraday.conventions import faraday_phase_cosmo

    freq = np.linspace(45, 50, 5)
    model = gs.ScreenModel()
    depth = 0.027
    beta = -2.3
    amp = 0.8
    angle = 0.21
    pol = (
        model.eta
        * amp
        * (freq / model.ref_mhz) ** beta
        * np.exp(2j * angle)
        * faraday_phase_cosmo(np.array([depth]), freq)[0]
    )
    y = np.r_[pol.real, pol.imag]
    fitted = profile_null_batch(
        y, freq, model, np.eye(10) * 0.001**2, True, True
    )
    assert fitted["converged"].all()
    assert np.max(abs(fitted["residual"])) < 1e-6
    restricted = profile_null_batch(
        y,
        freq,
        model,
        np.eye(10) * 0.001**2,
        True,
        True,
        depth_bounds=(-0.01, 0.01),
        beta_bounds=(-3.0, -2.0),
    )
    assert restricted["converged"].all()
    assert abs(restricted["parameters"][0, 2]) <= 0.01
    assert np.max(abs(restricted["residual"])) > 1e-4
    np.testing.assert_allclose(
        fitted["parameters"][0], [amp, angle, depth, beta], atol=1e-4, rtol=0
    )


def test_null_depth_mutation_rejected():
    from lusee_faraday import screen_inference as si

    original = si.profile_null_batch

    def ignore_depth(
        data, freq, model, noise, constant_depth=False, free_beta=False
    ):
        return original(data, freq, model, noise, False, free_beta)

    with patch.object(si, "profile_null_batch", ignore_depth):
        with pytest.raises(AssertionError):
            test_profiled_constant_null_recovers_direct_spectrum()


def test_null_bounds_mutation_rejected():
    from lusee_faraday import screen_inference as si

    original = si.profile_null_batch
    with patch.object(si, "profile_null_batch", lambda *a, **kw: original(*a)):
        with pytest.raises(AssertionError):
            test_profiled_constant_null_recovers_direct_spectrum()


def test_stationary_template_depth_power_reference():
    from scipy.integrate import quad
    from lusee_faraday.conventions import lambda_squared
    from lusee_faraday.screen_inference import local_stationary_template

    model = gs.ScreenModel()
    freq = np.linspace(29.9, 30.1, 5)
    x = lambda_squared(freq)
    x0 = float(lambda_squared([30.0])[0])
    z, weights = hermgauss(32)
    depth = model.mean_depth + np.sqrt(2) * model.sigma_depth * z
    branches = np.exp(2j * x[:, None] * depth)
    unit = (branches * (weights / np.sqrt(np.pi))) @ branches.conj().T
    # Independently integrate the small-separation physical coherence power.
    w = model.weight_sigma_pc
    power, _ = quad(
        lambda r: r
        / (2 * w**2)
        * np.exp(
            -(r**2) / (4 * w**2)
            - 2 * model.sigma_depth**2 * x0**2 * r**2 / model.correlation_pc**2
        ),
        0,
        np.inf,
        epsabs=1e-13,
    )
    amp = model.eta * model.p_ref_k * (freq / model.ref_mhz) ** model.beta
    c = power * np.outer(amp, amp) * unit
    expected = 0.5 * np.block([[c.real, -c.imag], [c.imag, c.real]])
    result = local_stationary_template(freq, model, 30.0)
    np.testing.assert_allclose(result["covariance"], expected, atol=1e-13)


def test_template_normalization_mutation_rejected():
    from lusee_faraday import screen_inference as si

    original = si.local_stationary_template

    def wrong(*args):
        result = original(*args)
        result["covariance"] *= 2
        return result

    with patch.object(si, "local_stationary_template", wrong):
        with pytest.raises(AssertionError):
            test_stationary_template_depth_power_reference()


def test_frequency_ablation_and_covariance_score():
    from lusee_faraday.screen_inference import (
        ablate_frequency_covariance,
        covariance_discrepancy,
    )

    # Coordinate order Q1,Q2,U1,U2: Q1-U1 and Q2-U2 must survive.
    signal = np.array(
        [
            [4.0, 0.2, 0.3, 0.4],
            [0.2, 3.0, 0.5, 0.6],
            [0.3, 0.5, 2.0, 0.7],
            [0.4, 0.6, 0.7, 1.0],
        ]
    )
    expected = np.array(
        [
            [4.0, 0.0, 0.3, 0.0],
            [0.0, 3.0, 0.0, 0.6],
            [0.3, 0.0, 2.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(
        ablate_frequency_covariance(signal), expected
    )
    noise = np.diag([0.4, 0.6, 0.8, 1.0])
    residual = np.array([1.0, -0.2, 0.3, -0.4])
    direct = (
        residual
        @ (np.linalg.inv(noise) - np.linalg.inv(noise + signal))
        @ residual
    )
    np.testing.assert_allclose(
        covariance_discrepancy(residual, signal, noise),
        [direct],
        atol=1e-13,
    )


def test_ablation_mutation_rejected():
    from lusee_faraday import screen_inference as si

    with patch.object(
        si, "ablate_frequency_covariance", lambda s: np.diag(np.diag(s))
    ):
        with pytest.raises(AssertionError):
            test_frequency_ablation_and_covariance_score()
