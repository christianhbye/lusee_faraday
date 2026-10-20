"""Sensitivity of fast marginal and global-mixture projections."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/moment_diagonal.py"
TEST = "tests/test_moment_diagonal.py::"

if __name__ == "__main__":
    run_faults(
        [
            (
                "drop the complex relation",
                SOURCE,
                "relation = b * b",
                "relation = np.zeros_like(b)",
                TEST
                + "test_marginal_variance_retains_all_four_complex_branches",
            ),
            (
                "discard global mean variance",
                SOURCE,
                '+ diagonal_moments(projector, delta)["smooth"]',
                '+ 0 * diagonal_moments(projector, delta)["smooth"]',
                TEST
                + "test_mixture_marginals_include_conditional_mean_variation",
            ),
        ],
        "moment_diagonal",
        __file__,
    )
