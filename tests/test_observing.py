"""Calendar exposure and time-averaging checks without mission-date claims."""

import numpy as np
import pytest
from scipy.integrate import quad

from lusee_faraday.config import SIDEREAL_DAY_S
from lusee_faraday.observing import phase_exposures, sidereal_bin_window


def test_complete_bin_exposure_and_duty_follow_actual_nightside_visits():
    period = SIDEREAL_DAY_S
    width = period / 1024
    duration = 4.2 * period

    def sun(t):
        return np.where(t < 2 * period, -20.0, 10.0)

    rows = phase_exposures(duration, [0.0, 0.25], sun, 0.5, bin_seconds=width)
    # Phase 0 at t=0 is incomplete; t=2 periods straddles sunrise.
    assert rows[0]["eligible_visits"] == 1
    assert rows[1]["eligible_visits"] == 2
    np.testing.assert_allclose(
        [r["exposure_s"] for r in rows], [width / 2, width]
    )
    assert rows[0]["visit_seconds"] == [period]


def test_no_nightside_exposure_is_explicitly_unavailable():
    rows = phase_exposures(
        3 * SIDEREAL_DAY_S, [0.2], lambda t: np.ones_like(t) * 10, 0.5
    )
    assert rows[0]["exposure_s"] == 0
    assert not rows[0]["available"]
    assert rows[0]["eligible_visits"] == 0


def test_sidereal_boxcar_matches_independent_phase_integration():
    width = SIDEREAL_DAY_S / 43
    values = sidereal_bin_window(64, width)
    for m in (-64, -25, 0, 3, 51):
        expected = (
            quad(
                lambda t: np.cos(2 * np.pi * m * t / SIDEREAL_DAY_S),
                -width / 2,
                width / 2,
                epsabs=1e-9,
            )[0]
            / width
        )
        np.testing.assert_allclose(values[m + 64], expected, atol=2e-14)
    np.testing.assert_allclose(values, values[::-1])


def test_invalid_observing_settings_are_rejected():
    with pytest.raises(ValueError):
        phase_exposures(1.0, [0.0], lambda t: t, 1.1)
    with pytest.raises(ValueError):
        phase_exposures(1.0, [1.0], lambda t: t, 0.5)
    with pytest.raises(ValueError):
        sidereal_bin_window(4, -1.0)
