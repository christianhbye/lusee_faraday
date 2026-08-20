"""Geometry knob, folding, knee (spec S4.2, S4.2.2)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp


def test_k_infinite_is_the_rm_histogram():
    phi_col = np.array([0.5, 1.5, 1.6, -2.5])
    w2 = np.array([1.0, 2.0, 3.0, 4.0])
    edges = np.arange(-3.0, 3.5, 1.0)
    H = dsp.depth_distribution(phi_col, w2, edges, k=np.inf)
    expected, _ = np.histogram(phi_col, bins=edges, weights=w2)
    np.testing.assert_allclose(H, expected)


def test_k_zero_is_a_superposition_of_tophats():
    """Two pixels, closed form: uniform density on [0, phi_col]."""
    phi_col = np.array([4.0, -2.0])
    w2 = np.array([1.0, 2.0])
    edges = np.arange(-3.0, 6.0, 1.0)  # -3..5
    H = dsp.depth_distribution(phi_col, w2, edges, k=0.0)
    # pixel 1: density 1/4 on (0,4); pixel 2: density 2/2 = 1 on (-2,0)
    expected = np.array([0.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.0])
    np.testing.assert_allclose(H, expected, atol=1e-12)
    assert np.isclose(H.sum(), w2.sum())


def test_k_one_cdf_power():
    """rho ~ f: CDF (e/phi_col)^2. One pixel phi_col=2 over edges 0,1,2."""
    H = dsp.depth_distribution(
        np.array([2.0]), np.array([1.0]), np.array([0.0, 1.0, 2.0]), k=1.0
    )
    np.testing.assert_allclose(H, [0.25, 0.75])


def test_k_minus_one_is_all_local():
    edges = np.arange(-2.0, 2.5, 1.0)
    H = dsp.depth_distribution(
        np.array([100.0, -50.0]), np.array([1.0, 3.0]), edges, k=-1.0
    )
    expected = np.array([0.0, 0.0, 4.0, 0.0])  # bin (0, 1) holds phi=0+
    np.testing.assert_allclose(H, expected)


def test_k_below_minus_one_raises_valueerror():
    """k < -1 is non-integrable; must raise ValueError, not silently wrong."""
    edges = np.arange(-2.0, 2.5, 1.0)
    phi_col = np.array([10.0])
    w2 = np.array([1.0])
    with pytest.raises(ValueError, match="k must be >= -1"):
        dsp.depth_distribution(phi_col, w2, edges, k=-1.5)
    with pytest.raises(ValueError, match="k must be >= -1"):
        dsp.depth_distribution(phi_col, w2, edges, k=-2.0)
    with pytest.raises(ValueError, match="k must be >= -1"):
        dsp.depth_distribution(phi_col, w2, edges, k=-5.0)


def test_support_extends_to_phi_col():
    """S6.4 extent clause: the k=0 top-hat reaches its column depth."""
    edges = dsp.phi_edges(30.0)
    H = dsp.depth_distribution(
        np.array([2000.0]), np.array([1.0]), edges, k=0.0
    )
    c = dsp.phi_centers(edges)
    assert H[np.searchsorted(edges, 1999.0) - 1] > 0
    assert H[c > 2001.0].sum() == 0.0


def test_fold_and_knee():
    edges = np.arange(-10.0, 10.5, 1.0)
    c = dsp.phi_centers(edges)
    H = np.where(np.abs(c) < 4, 1.0, 0.0)  # top-hat on |phi| < 4
    phi_abs, Hf = dsp.fold_template(c, H)
    knee = dsp.half_power_knee(phi_abs, Hf)
    assert 3.0 < knee < 4.5


def test_mass_quantile_knee_uniform_mass():
    """Five equal-mass bins: the q-quantile lands at the bin whose
    cumulative mass first reaches q (Ruling R10)."""
    phi_abs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    H = np.ones(5)  # cumulative fractions 0.2, 0.4, 0.6, 0.8, 1.0
    assert dsp.mass_quantile_knee(phi_abs, H, q=0.5) == 2.0
    assert dsp.mass_quantile_knee(phi_abs, H, q=0.9) == 4.0
    assert dsp.mass_quantile_knee(phi_abs, H) == 4.0  # default q=0.90


def test_mass_quantile_knee_ignores_a_narrow_origin_spike():
    """Unlike half_power_knee, a tall-but-narrow spike that holds only
    a modest mass fraction does not drag a quantile computed well
    past it (Ruling R10: half_power_knee is peak-relative and gets
    pinned to the spike; mass_quantile_knee is cumulative-mass-based
    and is not)."""
    phi_abs = np.arange(11.0)  # 0..10
    H = np.array([100.0] + [1.0] * 10)  # spike holds 100/110 = 90.9%
    # half_power_knee is dragged to the spike's edge:
    assert dsp.half_power_knee(phi_abs, H) < 1.0
    # the 95%-mass quantile lands well past it, in the flat tail
    assert dsp.mass_quantile_knee(phi_abs, H, q=0.95) == 5.0


def test_weighted_percentiles():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 97.0])
    p = dsp.weighted_percentiles(v, w, [50.0, 99.0])
    assert p[0] == 4.0 and p[1] == 4.0
