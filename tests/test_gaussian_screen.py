"""Cheap algebra/integration controls; no spatial-screen validation claim."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.hermite import hermgauss  # noqa: E402

from lusee_faraday.conventions import faraday_phase_cosmo  # noqa: E402
from lusee_faraday.gaussian_screen import (  # noqa: E402
    ScreenModel,
    SeparationQuadrature,
    branch_second_special,
    centered_branch,
    source_terms,
    visibility_moments,
)


def uniform_screen_reference(freq, model, nodes=64):
    """Independent 1D screen integral using Jones coherency, not branches."""
    z, weights = hermgauss(nodes)
    weights = weights / np.sqrt(np.pi)
    depth = model.mean_depth + np.sqrt(2) * model.sigma_depth * z
    f = (freq / model.ref_mhz) ** model.beta
    p = (
        model.eta
        * model.p_ref_k
        * f
        * np.exp(2j * np.deg2rad(model.angle_deg))
        * faraday_phase_cosmo(depth, freq)
    )
    intensity = model.eta * model.i_ref_k * f
    # <E_x E_q*> directly, with both Jones vectors normalized to unity.
    psi, delta = np.deg2rad([model.jones_angle_deg, model.jones_phase_deg])
    values = 0.5 * (
        np.cos(psi) * (intensity + p.real)
        + np.sin(psi) * np.exp(-1j * delta) * p.imag
    )
    p_off = (
        model.eta
        * model.p_ref_k
        * f
        * np.exp(2j * np.deg2rad(model.angle_deg))
    )
    off = 0.5 * (
        np.cos(psi) * (intensity + p_off.real)
        + np.sin(psi) * np.exp(-1j * delta) * p_off.imag
    )
    mean = weights @ values
    centered = values - mean
    return dict(
        off=off,
        mean_on=mean,
        covariance=np.einsum(
            "n,ni,nj->ij", weights, centered, centered.conj()
        ),
        relation=np.einsum("n,ni,nj->ij", weights, centered, centered),
        second_on=np.einsum("n,ni,nj->ij", weights, values, values.conj()),
        expected_abs_delta2=weights @ abs(values - off) ** 2,
    )


def test_zero_and_deterministic_screen():
    freq = np.array([20.0, 30.0, 50.0])
    zero = ScreenModel(mean_depth=0.0, sigma_depth=0.0)
    result = visibility_moments(freq, zero)
    np.testing.assert_allclose(result["mean_on"], result["off"], atol=1e-14)
    assert np.max(abs(result["covariance"])) == 0
    assert np.max(abs(result["expected_abs_delta2"])) == 0
    nonzero = replace(zero, mean_depth=0.013)
    predicted = visibility_moments(freq, nonzero)
    reference = uniform_screen_reference(freq, nonzero)
    np.testing.assert_allclose(predicted["mean_on"], reference["mean_on"])
    np.testing.assert_allclose(
        predicted["expected_abs_delta2"], reference["expected_abs_delta2"]
    )
    assert np.max(abs(predicted["covariance"])) == 0
    assert np.max(predicted["expected_abs_delta2"]) > 0.01


def test_uniform_screen_joint_moments():
    freq = np.array([45.0, 47.5, 50.0])
    model = ScreenModel(correlation_pc=np.inf)
    predicted = visibility_moments(freq, model)
    low = uniform_screen_reference(freq, model, 32)
    high = uniform_screen_reference(freq, model, 64)
    for key in low:
        np.testing.assert_allclose(low[key], high[key], atol=2e-11)
        np.testing.assert_allclose(predicted[key], high[key], atol=2e-11)
    # A uniform screen has unit |Z| per realization, not angular suppression.
    x, _, _, a = source_terms(freq, model)
    for i in range(len(x)):
        second = centered_branch(x[i], x[i], 1, 1, model) + abs(a[0, i]) ** 2
        np.testing.assert_allclose(second, 1.0, atol=2e-14)


def test_separation_quadrature_special_and_limits():
    numerical = SeparationQuadrature(epsabs=2e-13, epsrel=2e-12, upper_u=48.0)
    for ell in (0.1, 0.5, 2.0, 1000.0, np.inf):
        model = ScreenModel(correlation_pc=ell)
        freq = np.array([10.0, 30.0, 50.0])
        x, _, _, a = source_terms(freq, model)
        for i, j in ((0, 0), (0, 2), (1, 2), (2, 2)):
            for si, s in enumerate((1, -1)):
                for ti, t in enumerate((1, -1)):
                    default = centered_branch(x[i], x[j], s, t, model)
                    tight = centered_branch(x[i], x[j], s, t, model, numerical)
                    special = branch_second_special(x[i], x[j], s, t, model)
                    np.testing.assert_allclose(default, tight, atol=2e-9)
                    np.testing.assert_allclose(
                        tight + a[si, i] * a[ti, j].conj(),
                        special,
                        atol=2e-9,
                    )
    model = ScreenModel()
    x = 100.0
    small = abs(
        centered_branch(x, x, 1, 1, replace(model, correlation_pc=0.001))
    )
    less_small = abs(
        centered_branch(x, x, 1, 1, replace(model, correlation_pc=0.002))
    )
    assert 3.9 < less_small / small < 4.1
    large = centered_branch(x, x, 1, 1, replace(model, correlation_pc=10000.0))
    constant = centered_branch(
        x, x, 1, 1, replace(model, correlation_pc=np.inf)
    )
    np.testing.assert_allclose(large, constant, atol=1e-6)


def test_frequency_covariance_and_relation():
    freq = np.linspace(29.9, 30.1, 7)
    result = visibility_moments(freq, ScreenModel())
    c, j = result["covariance"], result["relation"]
    np.testing.assert_allclose(c, c.conj().T, atol=2e-13)
    np.testing.assert_allclose(j, j.T, atol=2e-13)
    # Positivity of the augmented complex covariance also tests branch joints.
    augmented = np.block([[c, j], [j.conj(), c.conj()]])
    assert np.linalg.eigvalsh(augmented).min() > -2e-11
    assert np.all(result["expected_abs_delta2"] >= 0)
    diagonal = visibility_moments(freq, ScreenModel(), diagonal_only=True)
    np.testing.assert_allclose(diagonal["covariance"], np.diag(c), atol=2e-13)
