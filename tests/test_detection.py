import numpy as np
from lusee_faraday import rmsynth
from lusee_faraday.detection import faraday_noise_std, faraday_snr


def test_noise_std_matches_monte_carlo():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 200))
    sigma = np.full(lam2.size, 2.0)
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    analytic = faraday_noise_std(sigma)
    rng = np.random.default_rng(0)
    reals = []
    for _ in range(400):
        q = rng.normal(scale=sigma)
        u = rng.normal(scale=sigma)
        F = rmsynth.faraday_spectrum(q, u, lam2, phi)[0]
        reals.append(F.real)
    mc = np.std(np.array(reals))
    assert abs(analytic - mc) / analytic < 0.1


def test_noise_std_inverse_variance_formula():
    sigma = np.array([1.0, 2.0, 4.0])
    w = 1.0 / sigma ** 2
    assert np.isclose(
        faraday_noise_std(sigma, weights=w),
        1.0 / np.sqrt(np.sum(1.0 / sigma ** 2)),
    )


def test_faraday_snr_high_for_clean_signal():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 300))
    p = np.exp(2j * 6.0 * lam2)
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    sigma = np.full(lam2.size, 0.01)
    snr, peak, nstd = faraday_snr(p.real, p.imag, lam2, sigma, phi)
    assert snr > 20
    assert peak > 0.9


def test_faraday_snr_order_unity_for_pure_noise():
    lam2 = rmsynth.lambda2(np.linspace(10, 50, 300))
    phi = rmsynth.phi_grid(lam2, phi_max=30.0)
    sigma = np.full(lam2.size, 1.0)
    rng = np.random.default_rng(1)
    snr, _, _ = faraday_snr(
        rng.normal(scale=sigma), rng.normal(scale=sigma),
        lam2, sigma, phi,
    )
    assert snr < 6
