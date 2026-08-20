import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.constants import c as C_LIGHT  # noqa: E402

from lusee_faraday import conventions as cv  # noqa: E402


def test_product_labels_match_luseepy():
    lusee_cov = pytest.importorskip("lusee.Covariance")

    assert cv.PRODUCT_LABELS == lusee_cov.default_product_labels()
    assert len(cv.PRODUCT_LABELS) == 16
    assert cv.PORT_PAIRS == tuple(
        (a, b) for a in range(4) for b in range(a, 4)
    )


def test_cosmo_to_iau_flips_u_and_leaves_q_alone():
    """Pin *which* Stokes parameter flips, not just that one does.

    A roundtrip cannot see this: ``(Q, -U)`` and ``(-Q, U)`` are both
    self-inverse.  Neither can
    ``test_dual_block_phase_agrees_with_rotating_maps_first``, which
    applies the conversion on both sides of its comparison, where a
    Q-flip differs from a U-flip only by a global sign that cancels.
    So assert the literal signs on an input whose two components are
    distinguishable, in both directions.  ``AGENTS.md`` names this
    function as the pinned source of truth for ``U_IAU = -U_COSMO``;
    this is where that is actually pinned.
    """
    Q_iau, U_iau = cv.cosmo_to_iau_qu(0.3, -0.7)
    assert (float(Q_iau), float(U_iau)) == (0.3, 0.7)

    Q_cos, U_cos = cv.iau_to_cosmo_qu(0.3, -0.7)
    assert (float(Q_cos), float(U_cos)) == (0.3, 0.7)

    # Vector form, and the sign asserted componentwise rather than
    # through any norm that a Q/U swap could survive.
    Q = np.array([1.0, -2.0, 0.0, 4.0])
    U = np.array([5.0, 6.0, -7.0, 0.0])
    Qi, Ui = cv.cosmo_to_iau_qu(Q, U)
    np.testing.assert_array_equal(Qi, Q)
    np.testing.assert_array_equal(Ui, -U)


def test_qu_convention_roundtrip_is_identity():
    """Self-inverseness, which the test above does not cover."""
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
    # Compute expected phase independently (not calling module functions)
    lam2 = (C_LIGHT / (freq * 1e6)) ** 2
    expected_p_plus = np.exp(2j * phi * lam2)
    expected_p_minus = np.conj(expected_p_plus)
    assert np.allclose(blocks[:, 3], expected_p_plus)  # P+ == +2i phase
    assert np.allclose(blocks[:, 2], expected_p_minus)  # P- == -2i phase
    # Verify the conjugacy relationship
    assert np.allclose(blocks[:, 2], np.conj(blocks[:, 3]))


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
