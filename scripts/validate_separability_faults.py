"""Fault sensitivity for common smooth subtraction and intensity ratios."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/separability.py"
TEST = "tests/test_separability.py::"
FAULTS = [
    (
        "smooth template ignores channel response",
        "return w @ np.polynomial.legendre.legvander(coordinate, degree)",
        "return np.polynomial.legendre.legvander("
        "np.linspace(-1,1,w.shape[0]),degree)",
        "test_smooth_templates_include_actual_zoom_fold",
    ),
    (
        "flags receive unit weight",
        "root = np.sqrt(w / w.max())",
        "root = np.ones_like(w)",
        "test_weighted_smooth_fit_uses_weighted_mean_and_excludes_flags",
    ),
    (
        "covariance projected on only one leg",
        'return np.einsum("bi,ipjq,cj->bpcq", r, c, r, optimize=True)',
        'return np.einsum("bi,ipjq,cj->bpcq", r, c, '
        "np.eye(len(r)), optimize=True)",
        "test_residual_covariance_matches_independent_finite_ensemble",
    ),
    (
        "ignore noisy intensity in ratio covariance",
        "if estimated:\n            jacobian",
        "if False:\n            jacobian",
        "test_intensity_uncertainty_and_cross_covariance_"
        "match_ratio_monte_carlo",
    ),
    (
        "normalize before smooth subtraction",
        "numerator = r @ s[:, 1:3]",
        "numerator = (r @ (s[:,1:3]/intensity[:,None])) * intensity[:,None]",
        "test_smooth_subtraction_precedes_intensity_division",
    ),
    (
        "zero denominator becomes infinite ratio",
        "out=np.full(a.shape, np.nan)",
        "out=np.full(a.shape, np.inf)",
        "test_undefined_ratios_and_uninformative_fits_are_explicit",
    ),
]


if __name__ == "__main__":
    run_faults(
        [
            (name, SOURCE, before, after, TEST + test)
            for name, before, after, test in FAULTS
        ],
        "separability",
        __file__,
    )
