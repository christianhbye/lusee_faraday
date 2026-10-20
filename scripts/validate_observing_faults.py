"""Fault sensitivity for nightside exposure and sidereal averaging."""

import common  # noqa: F401
from fault_audit import run_faults

SOURCE = "src/lusee_faraday/observing.py"
TEST = "tests/test_observing.py::"
FAULTS = [
    (
        "omit electronics duty",
        "selected.sum() * bin_seconds * duty_cycle",
        "selected.sum() * bin_seconds",
        "test_complete_bin_exposure_and_duty_follow_actual_nightside_visits",
    ),
    (
        "count daytime visits",
        "selected = np.all(altitude <= night_limit_deg, axis=1)",
        "selected = np.ones(len(visits), bool)",
        "test_no_nightside_exposure_is_explicitly_unavailable",
    ),
    (
        "omit finite sidereal bin averaging",
        "return np.sinc(np.arange(-lmax, lmax + 1) * bin_seconds / "
        "SIDEREAL_DAY_S)",
        "return np.ones(2*lmax+1)",
        "test_sidereal_boxcar_matches_independent_phase_integration",
    ),
    (
        "accept duty exceeding one",
        "or not 0 <= duty_cycle <= 1",
        "or not 0 <= duty_cycle <= 2",
        "test_invalid_observing_settings_are_rejected",
    ),
]


if __name__ == "__main__":
    run_faults(
        [(a, SOURCE, b, c, TEST + d) for a, b, c, d in FAULTS],
        "observing",
        __file__,
    )
