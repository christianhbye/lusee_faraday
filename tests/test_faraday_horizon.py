"""Horizon arithmetic and its physically degenerate zero-field limit."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.conventions import lambda_squared  # noqa: E402
from lusee_faraday.faraday_horizon import (  # noqa: E402
    ROTATION_PER_PC,
    faraday_rate,
    horizon_depth,
    unrotated_brightness,
)


def test_rate_units_and_sign():
    # Conventional cm^-3, microgauss and pc coefficient, independently rounded.
    np.testing.assert_allclose(ROTATION_PER_PC, 0.812, rtol=3e-4)
    np.testing.assert_allclose(
        faraday_rate(0.1, [2.0, -2.0]), [0.1624, -0.1624], rtol=3e-4
    )


def test_horizon_reaches_requested_phase_and_scales_with_frequency():
    freq = np.array([10.0, 30.0, 50.0])
    d = horizon_depth(freq, 0.004, 1.0, phase_limit_rad=2.0)
    phase = 2 * lambda_squared(freq) * faraday_rate(0.004, 1.0) * d
    np.testing.assert_allclose(phase, 2.0)
    np.testing.assert_allclose(d / d[0], [1.0, 9.0, 25.0])
    np.testing.assert_allclose(horizon_depth(freq, 0.004, -1.0), d / 2)


def test_zero_field_has_no_finite_horizon_but_volume_is_finite():
    depth = horizon_depth([30.0], [0.004, 0.004], [0.0, 1.0])
    assert np.isinf(depth[0, 0])
    assert np.isfinite(depth[1, 0])
    brightness = unrotated_brightness([30.0], depth, 5.0, 2.0)
    assert brightness[0, 0] == 10.0
    assert 0 < brightness[1, 0] < 10.0
    np.testing.assert_allclose(
        unrotated_brightness([30.0], depth, 5.0, 4.0),
        2 * brightness,
    )


def test_invalid_horizon_inputs():
    with pytest.raises(ValueError):
        faraday_rate(-1, 1)
    with pytest.raises(ValueError):
        horizon_depth([0.0], 0.004, 1.0)
    with pytest.raises(ValueError):
        horizon_depth([30.0], 0.004, 1.0, phase_limit_rad=0)
    with pytest.raises(ValueError):
        unrotated_brightness([30.0], [np.nan], 5.0, 1.0)
