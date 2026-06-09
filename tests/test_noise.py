import numpy as np
from lusee_faraday import noise


def test_radiometer_sigma_matches_formula():
    sig = noise.radiometer_sigma(100.0, 390.625, 3600.0)
    assert np.isclose(sig, 100.0 / np.sqrt(390.625 * 3600.0))


def test_radiometer_sigma_vectorized():
    T = np.array([100.0, 200.0])
    dnu = np.array([390.625, 25000.0])
    sig = noise.radiometer_sigma(T, dnu, 3600.0)
    assert sig.shape == (2,)
    assert np.allclose(sig, T / np.sqrt(dnu * 3600.0))


def test_add_noise_statistics():
    rng = np.random.default_rng(0)
    x = np.zeros(200000)
    y = noise.add_noise(x, 2.0, rng)
    assert abs(np.std(y) - 2.0) < 0.05
    assert abs(np.mean(y)) < 0.05


def test_add_noise_per_channel_sigma_broadcasts():
    rng = np.random.default_rng(1)
    sigma = np.array([1.0, 5.0])
    y = noise.add_noise(np.zeros((50000, 2)), sigma, rng)
    assert np.isclose(np.std(y[:, 0]), 1.0, atol=0.05)
    assert np.isclose(np.std(y[:, 1]), 5.0, atol=0.1)
