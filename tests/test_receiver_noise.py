"""Loaded-noise and radiometer checks using independent voltage samples."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import roots_hermitenorm  # noqa: E402

from lusee_faraday import instrument, polarimeter  # noqa: E402
from lusee_faraday.receiver_noise import (  # noqa: E402
    channel_radiometer_covariance,
    check_spectral_covariance,
    diagonal_amplifier_covariance,
    loaded_amplifier_covariance,
    quadratic_radiometer_covariance,
    science_quadratic_matrices,
    stokes_quadratic_matrices,
)


def covariance(seed=417):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    return a @ a.conj().T


def test_loaded_voltage_current_noise_matches_direct_kirchhoff_samples():
    rng = np.random.default_rng(416)
    za = np.diag([40 + 50j, 30 - 20j, 80 + 15j, 60 - 35j]).astype(complex)
    za[0, 1] = za[1, 0] = 5 + 2j
    zl = np.diag([100 - 50j, 70 - 100j, 50 - 30j, 200 - 10j])
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    units = np.r_[np.full(4, 1e-9), np.full(4, 1e-13)]
    a *= units[:, None]
    joint = a @ a.conj().T
    rank = 8
    unit = np.r_[
        np.eye(rank), -np.eye(rank), 1j * np.eye(rank), -1j * np.eye(rank)
    ] * np.sqrt(rank)
    samples = unit @ a.T
    voltage = (
        samples[:, :4]
        + np.linalg.solve(
            np.linalg.inv(za) + np.linalg.inv(zl), samples[:, 4:].T
        ).T
    )
    expected = voltage.T @ voltage.conj() / len(voltage)
    actual = loaded_amplifier_covariance(
        za, zl, joint[:4, :4], joint[4:, 4:], joint[:4, 4:]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-29)
    assert abs(joint[:4, 4:]).max() > 0


def test_explicit_asds_and_missing_receiver_data():
    za = np.eye(4) * (50 + 20j)
    zl = np.eye(4) * (100 - 10j)
    e, i = 3e-9, 1e-13
    zeff = 1 / (1 / (50 + 20j) + 1 / (100 - 10j))
    actual = diagonal_amplifier_covariance(za, zl, e, i)
    np.testing.assert_allclose(
        actual, np.eye(4) * (e * e + abs(zeff * i) ** 2), rtol=1e-13
    )
    with pytest.raises(TypeError):
        diagonal_amplifier_covariance(za, zl)
    with pytest.raises(ValueError, match="ASDs"):
        diagonal_amplifier_covariance(za, zl, -e, i)
    with pytest.raises(ValueError, match="positive semidefinite"):
        loaded_amplifier_covariance(
            za, zl, np.eye(4) * e**2, np.eye(4) * i**2, np.eye(4) * 2 * e * i
        )


def test_quadratic_readouts_match_production_science_and_stokes():
    s = covariance()
    rng = np.random.default_rng(418)
    x = rng.normal(size=4) + 1j * rng.normal(size=4)
    y = rng.normal(size=4) + 1j * rng.normal(size=4)
    for a, expected in (
        (science_quadratic_matrices(), instrument.channels(s)[0]),
        (
            stokes_quadratic_matrices(x, y, 2.3),
            polarimeter.pseudo_stokes(s, x, y) / 2.3,
        ),
    ):
        np.testing.assert_allclose(
            np.einsum("pab,ba->p", a, s), expected, atol=2e-13
        )


def test_full_product_covariance_matches_exact_gaussian_quadrature():
    # Three nodes per real dimension integrate Gaussian fourth moments exactly.
    nodes, w = roots_hermitenorm(3)
    indices = np.indices((3,) * 8).reshape(8, -1).T
    samples = nodes[indices]
    weights = np.prod(w[indices] / np.sqrt(2 * np.pi), axis=1)
    z = (samples[:, :4] + 1j * samples[:, 4:]) / np.sqrt(2)
    s = covariance(419)
    voltage = z @ np.linalg.cholesky(s).T
    products = instrument.channels(
        np.einsum("na,nb->nab", voltage, voltage.conj())
    )[0]
    mean = weights @ products
    centered = products - mean
    expected = np.einsum("n,np,nq->pq", weights, centered, centered)
    actual = quadratic_radiometer_covariance(s, science_quadratic_matrices())
    np.testing.assert_allclose(actual, expected, atol=3e-12)
    # Off-diagonal correlations are physical and cannot be dropped.
    assert np.max(abs(actual - np.diag(np.diag(actual)))) > 1


def test_diagonal_variances_match_luseepy_and_scale_with_integration():
    from lusee.MapMaker import compute_radiometric_noise

    s = covariance(420)
    bandwidth, time = 390.625, 7200.0
    measured = instrument.channels(s)[0][None, :, None]
    sigma = np.asarray(
        compute_radiometric_noise(
            measured, delta_f_hz=bandwidth, delta_t_sec=time
        )
    )[0, :, 0]
    a = science_quadratic_matrices()
    actual = quadratic_radiometer_covariance(s, a, bandwidth * time)
    np.testing.assert_allclose(np.diag(actual), sigma**2, rtol=3e-13)
    longer = quadratic_radiometer_covariance(s, a, 4 * bandwidth * time)
    np.testing.assert_allclose(np.sqrt(np.diag(longer)), sigma / 2, rtol=3e-13)


def test_channel_overlap_and_variable_system_power_match_voltage_monte_carlo():
    rng = np.random.default_rng(421)
    system = np.array([covariance(seed) / 10 for seed in (422, 423, 424)])
    factors = np.linalg.cholesky(system)
    w = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    a = science_quadratic_matrices()[:3]
    predicted = channel_radiometer_covariance(system, w, 4.0, 8.0, a).reshape(
        6, 6
    )
    values = []
    for _ in range(40):
        # Independent samples for three fine channels, each with B*tau=32.
        z = (
            rng.normal(size=(500, 3, 32, 4))
            + 1j * rng.normal(size=(500, 3, 32, 4))
        ) / np.sqrt(2)
        voltage = np.einsum("nfta,fba->nftb", z, factors)
        power = np.einsum("nfta,nftb->nfab", voltage, voltage.conj()) / 32
        packed = instrument.channels(power)[0][..., :3]
        values.append(np.einsum("bf,nfp->nbp", w, packed).reshape(500, 6))
    values = np.concatenate(values)
    centered = values - values.mean(axis=0)
    empirical = centered.T @ centered / len(values)
    scale = np.sqrt(np.outer(np.diag(predicted), np.diag(predicted)))
    assert np.max(abs(empirical - predicted) / scale) < 0.06
    flat = channel_radiometer_covariance(system[0], w, 4.0, 8.0, a)
    expected = (
        np.einsum(
            "bc,pq->bpcq",
            w @ w.T,
            quadratic_radiometer_covariance(system[0], a),
        )
        / 32
    )
    np.testing.assert_allclose(flat, expected, atol=1e-15)


def test_invalid_system_readout_and_exposure_are_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        check_spectral_covariance(np.diag([1.0, 1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="Hermitian"):
        check_spectral_covariance(np.eye(4) + np.triu(np.ones((4, 4)), 1))
    with pytest.raises(ValueError, match="independent-sample"):
        quadratic_radiometer_covariance(
            np.eye(4), science_quadratic_matrices(), 0
        )
    with pytest.raises(ValueError, match="integration"):
        channel_radiometer_covariance(
            np.eye(4), np.ones((2, 3)), 1.0, 0.0, science_quadratic_matrices()
        )
