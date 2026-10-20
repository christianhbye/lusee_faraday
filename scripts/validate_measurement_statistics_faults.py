"""Fault checks for power sampling uncertainty and fixed-sky repeats."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/measurement_statistics.py"
TEST = "tests/test_measurement_statistics.py::"
FAULTS = [
    (
        "count correlated channels as independent",
        "np.sum(c * c)",
        "np.sum(np.diag(c)**2)",
        "test_power_uncertainty_matches_exact_gaussian_fourth_moments",
    ),
    (
        "add repeat noise amplitudes instead of variances",
        "covariance=ca + cb",
        "covariance=(np.sqrt(abs(ca))+np.sqrt(abs(cb)))**2",
        "test_repeat_difference_cancels_fixed_sky_and_adds_independent_noise",
    ),
    (
        "accept negative variance",
        "if np.linalg.eigvalsh(c)[0] <",
        "if False and np.linalg.eigvalsh(c)[0] <",
        "test_invalid_statistics_covariance_is_rejected",
    ),
    (
        "silently discard complex covariance",
        "not np.isrealobj(covariance)",
        "False",
        "test_invalid_statistics_covariance_is_rejected",
    ),
]

if __name__ == "__main__":
    run_faults(
        [(a, SOURCE, b, c, TEST + d) for a, b, c, d in FAULTS],
        "measurement_statistics",
        __file__,
    )
