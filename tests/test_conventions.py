import numpy as np
import pytest

from lusee_faraday import conventions as cv


def test_product_labels_match_luseepy():
    lusee_cov = pytest.importorskip("lusee.Covariance")

    assert cv.PRODUCT_LABELS == lusee_cov.default_product_labels()
    assert len(cv.PRODUCT_LABELS) == 16
    assert cv.PORT_PAIRS == tuple(
        (a, b) for a in range(4) for b in range(a, 4)
    )


def test_qu_convention_roundtrip_is_identity():
    rng = np.random.default_rng(0)
    Q, U = rng.normal(size=5), rng.normal(size=5)
    Q2, U2 = cv.iau_to_cosmo_qu(*cv.cosmo_to_iau_qu(Q, U))
    assert np.allclose(Q2, Q)
    assert np.allclose(U2, U)


def test_lambda_squared_at_30_mhz():
    # c / 30 MHz = 9.9931 m, so lambda^2 is just under 100 m^2.
    assert np.isclose(cv.lambda_squared(30.0)[0], 99.8617, rtol=1e-4)


def test_faraday_phase_matches_explicit_rotation():
    """The COSMO phase must reproduce an explicit (Q, U) rotation."""
    phi, freq = 250.0, np.array([30.0])
    Q, U = 0.3, -0.7
    lam2 = cv.lambda_squared(freq)
    angle = 2 * phi * lam2
    Q_rot = Q * np.cos(angle) - U * np.sin(angle)
    U_rot = Q * np.sin(angle) + U * np.cos(angle)
    got = (Q + 1j * U) * cv.faraday_phase_cosmo(phi, freq)
    assert np.allclose(got.real, Q_rot)
    assert np.allclose(got.imag, U_rot)


def test_dual_block_phase_is_conjugate_on_p_blocks():
    """P- carries exp(-2i phi lam^2) because IAU flips U."""
    phi, freq = 250.0, np.array([30.0, 10.0])
    blocks = cv.dual_block_phase(phi, freq)
    assert blocks.shape == (2, 4)
    assert np.allclose(blocks[:, 0], 1.0)  # I
    assert np.allclose(blocks[:, 1], 1.0)  # V
    assert np.allclose(blocks[:, 2], np.conj(blocks[:, 3]))
    cosmo = cv.faraday_phase_cosmo(phi, freq)
    assert np.allclose(blocks[:, 3], cosmo)  # P+ == COSMO phase
    assert np.allclose(blocks[:, 2], np.conj(cosmo))


def test_dual_block_phase_agrees_with_rotating_maps_first():
    """Rotate (Q, U) then convert to IAU == convert then apply P blocks."""
    rng = np.random.default_rng(3)
    phi = 137.0
    freq = np.array([29.9, 30.0, 30.1])
    Q, U = rng.normal(size=8), rng.normal(size=8)

    rotated = (Q + 1j * U)[:, None] * cv.faraday_phase_cosmo(phi, freq)
    Q_rot, U_rot = rotated.real, rotated.imag
    Q_iau, U_iau = cv.cosmo_to_iau_qu(Q_rot, U_rot)
    p_minus_direct = Q_iau + 1j * U_iau

    Q0_iau, U0_iau = cv.cosmo_to_iau_qu(Q, U)
    blocks = cv.dual_block_phase(phi, freq)
    p_minus_via_blocks = (Q0_iau + 1j * U0_iau)[:, None] * blocks[None, :, 2]

    assert np.allclose(p_minus_direct, p_minus_via_blocks)


def test_dual_block_phase_broadcasts_over_regions():
    phi = np.array([0.0, 100.0, -250.0])
    freq = np.array([30.0, 30.1])
    assert cv.dual_block_phase(phi, freq).shape == (3, 2, 4)
