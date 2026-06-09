import numpy as np

from lusee_faraday.fisher import (
    detection_snr,
    fisher_matrix,
    marginal_error,
    stack_real,
)


def test_stack_real_layout():
    P = np.array([[1 + 2j, 3 + 4j]])
    np.testing.assert_array_equal(stack_real(P), [1, 3, 2, 4])


def test_orthogonal_nuisance_does_not_inflate():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)  # real quadrature
    nuisance = 1j * np.ones((1, n), dtype=complex)  # imag quadrature
    F = fisher_matrix([signal, nuisance], sig)
    F_only = fisher_matrix([signal], sig)
    # orthogonal -> marginalized error equals unmarginalized
    assert np.isclose(marginal_error(F, 0), marginal_error(F_only, 0))
    assert np.isclose(marginal_error(F_only, 0), 1.0 / np.sqrt(n))


def test_degenerate_nuisance_inflates_error():
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)
    near_parallel = (1.0 + 1e-3 * 1j) * np.ones((1, n), dtype=complex)
    F = fisher_matrix([signal, near_parallel], sig)
    F_only = fisher_matrix([signal], sig)
    assert marginal_error(F, 0) > 10 * marginal_error(F_only, 0)
    # marginalizing can only reduce SNR
    assert detection_snr(F, 0) < detection_snr(F_only, 0)


def test_sigma_scaling():
    n = 8
    signal = np.ones((1, n), dtype=complex)
    F1 = fisher_matrix([signal], np.ones((1, n)))
    F2 = fisher_matrix([signal], 2 * np.ones((1, n)))
    assert np.isclose(marginal_error(F2, 0), 2 * marginal_error(F1, 0))


def test_marginal_differs_from_conditional():
    # Off-diagonal but well-conditioned F: marginalizing a correlated
    # nuisance must inflate the error above the conditional 1/sqrt(F[ii]).
    # A conditional implementation (1/sqrt(F[0,0])) would fail this.
    n = 16
    sig = np.ones((1, n))
    signal = np.ones((1, n), dtype=complex)
    nuisance = (0.5 + 0.5j) * np.ones((1, n), dtype=complex)
    F = fisher_matrix([signal, nuisance], sig)
    conditional = 1.0 / np.sqrt(F[0, 0])
    marg = marginal_error(F, 0)
    assert marg > conditional * (1 + 1e-6)
    np.testing.assert_allclose(marg, np.sqrt(2.0 / n), rtol=1e-9)
