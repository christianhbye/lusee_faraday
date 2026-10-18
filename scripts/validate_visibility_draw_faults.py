"""Fault checks for stable complex coefficient visibility illustrations."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/visibility_draws.py"
TEST = "tests/test_visibility_draws.py::"

if __name__ == "__main__":
    run_faults(
        [
            (
                "drop imaginary phase coefficient",
                SOURCE,
                "+ a[None, :, mode, None].imag * imag[:, None, :]",
                "+ 0 * imag[:, None, :]",
                TEST + "test_complex_mode_draw_matches_"
                "independent_latent_covariance",
            ),
            (
                "change modes between raw and fitted spectra",
                SOURCE,
                "rng = np.random.default_rng(seed)",
                "rng = np.random.default_rng([seed,int(abs(a).sum()*1000)])",
                TEST + "test_same_latent_modes_commute_with_fit_"
                "and_preserve_small_residuals",
            ),
        ],
        "visibility_draw",
        __file__,
    )
