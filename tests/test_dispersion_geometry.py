"""Geometry knob, folding, knee (spec S4.2, S4.2.2)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp


def test_k_infinite_is_the_rm_histogram():
    """k = inf: all emission behind the column, so F is the w2-weighted
    histogram of the SIGNED column depths themselves.

    The expectation is written out by hand rather than taken from
    ``np.histogram`` -- comparing the branch against the very call it
    makes is a tautology, and this is the only test of the k = inf
    branch, which is what the S6.14 tail gate runs on.
    Bins are [-3,-2), [-2,-1), [-1,0), [0,1), [1,2), [2,3]:
      -2.5 (w 4)          -> bin 0
      0.5 (w 1)           -> bin 3
      1.5 (w 2), 1.6 (w 3) -> bin 4, summed to 5
    """
    phi_col = np.array([0.5, 1.5, 1.6, -2.5])
    w2 = np.array([1.0, 2.0, 3.0, 4.0])
    edges = np.arange(-3.0, 3.5, 1.0)
    H = dsp.depth_distribution(phi_col, w2, edges, k=np.inf)
    np.testing.assert_allclose(H, [4.0, 0.0, 0.0, 1.0, 5.0, 0.0])
    assert np.isclose(H.sum(), w2.sum())


def test_k_infinite_keeps_the_sign_and_the_last_bin_edge():
    """Sign fidelity and the right-closed final bin, hand-written.

    Bins [-2,-1), [-1,0), [0,1), [1,2] -- every bin half-open except
    the last, which is closed on the right.  A branch that took
    |phi_col| would put everything in bins 2-3 and fails the first
    array; one that dropped the closed top edge loses the 2.0 pixel.

    The sign flip is deliberately NOT a mirror image, and that is the
    point: under negation 2.0 -> -2.0 moves from the closed top edge
    into bin 0 while 1.0 -> -1.0 moves from bin 3 into bin 1, so
    [1,2,0,12] becomes [4,8,2,1] rather than its reverse.  Writing
    the reversal down by hand is what caught this; the previous
    version of this test compared against ``np.histogram`` and could
    not have.
    """
    edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    phi_col = np.array([-2.0, -0.5, 2.0, 1.0])
    w2 = np.array([1.0, 2.0, 4.0, 8.0])
    H = dsp.depth_distribution(phi_col, w2, edges, k=np.inf)
    np.testing.assert_allclose(H, [1.0, 2.0, 0.0, 12.0])
    Hm = dsp.depth_distribution(-phi_col, w2, edges, k=np.inf)
    np.testing.assert_allclose(Hm, [4.0, 8.0, 2.0, 1.0])


def test_k_infinite_ignores_the_zero_bin_guard():
    """k = inf is exempt from the bracket-zero guard the finite-k
    pushforwards need: it is a plain histogram of phi_col and never
    touches the zero bin, so an all-positive grid is legitimate."""
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    H = dsp.depth_distribution(
        np.array([0.0, 15.0, 35.0]),
        np.array([5.0, 1.0, 2.0]),
        edges,
        k=np.inf,
    )
    np.testing.assert_allclose(H, [1.0, 0.0, 2.0])  # the phi=0 mass drops


def test_depth_distribution_requires_edges_bracketing_zero():
    """Finite k puts mass at phi = 0 (the near end of every column).

    With an all-positive grid the old ``searchsorted(edges, 0.0,
    'right') - 1`` returned -1 and that mass landed silently in the
    LAST bin.  Measured against the pre-fix code, ``edges =
    [10,20,30,40]`` with pixels ``phi = [0, 100]``, ``w2 = [5, 1]``
    gave ``[0.1, 0.1, 5.1]`` at k=0 and ``[0, 0, 6]`` at k=-1 -- five
    units of ZERO-depth mass reported at 30-40 rad/m^2, and at k=-1
    the entire distribution. Must raise instead.
    """
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    phi_col = np.array([0.0, 100.0])
    w2 = np.array([5.0, 1.0])
    for k in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="edges must bracket"):
            dsp.depth_distribution(phi_col, w2, edges, k=k)
    # an all-negative grid is rejected too (it used to IndexError)
    with pytest.raises(ValueError, match="edges must bracket"):
        dsp.depth_distribution(
            phi_col, w2, np.array([-40.0, -30.0, -20.0]), k=0.0
        )
    # the boundary cases that ARE well posed still work
    ok = dsp.depth_distribution(
        np.array([2.0]), np.array([1.0]), np.array([0.0, 1.0, 2.0]), k=0.0
    )
    np.testing.assert_allclose(ok, [0.5, 0.5])


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


def test_mass_quantile_knee_raises_on_non_positive_total_mass():
    """Cleanup A: an all-zero H used to divide 0/0 (RuntimeWarning) and
    return phi_abs[0] by accident of how searchsorted treats an
    all-NaN array. Must raise instead, matching the k < -1 guard
    style in depth_distribution."""
    phi_abs = np.array([0.0, 1.0, 2.0])
    H = np.zeros(3)
    with pytest.raises(ValueError, match="total mass must be positive"):
        dsp.mass_quantile_knee(phi_abs, H)


def test_weighted_percentiles():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 97.0])
    p = dsp.weighted_percentiles(v, w, [50.0, 99.0])
    assert p[0] == 4.0 and p[1] == 4.0


healpy = pytest.importorskip("healpy")


def test_structure_function_of_a_smooth_map_scales_as_theta_squared():
    import healpy as hp

    nside = 64
    th, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    m = np.cos(th)  # smooth dipole-like scalar
    thetas = np.array([1.0, 2.0, 4.0])
    D = dsp.structure_function(
        m, thetas, nsamp=40_000, rng=np.random.default_rng(1)
    )
    # D(theta) ~ c theta^2 for a smooth field: ratios 4 and 16
    assert np.isclose(D[1] / D[0], 4.0, rtol=0.3)
    assert np.isclose(D[2] / D[0], 16.0, rtol=0.3)


def test_coherence_angle_analytic():
    # R15: the brief's grid (start 0.1 deg) does not bracket the root
    # (0.081028 deg), so coherence_angle takes the documented clamped-
    # low branch and the test cannot pass as originally written. Any
    # start below 0.081028 deg puts the root interior; 0.01 was
    # verified to make the log-log interpolation exact (D ~ theta^2
    # makes log D linear in log theta).
    theta_deg = np.linspace(0.01, 30.0, 300)
    c = 25.0
    D = c * np.radians(theta_deg) ** 2
    lam2 = 100.0
    got = dsp.coherence_angle(theta_deg, D, lam2)
    expected = 1.0 / (lam2 * np.sqrt(2.0 * c))
    assert np.isclose(got, expected, rtol=0.02)


def test_coherence_angle_clamps_below_the_sampled_range():
    """When the true root lies below the grid's first sample,
    coherence_angle clamps to that sample instead of extrapolating
    (documented behaviour). A caller must check for this explicitly:
    on the real sky this branch is expected to trigger, and a silent
    clamp would be a trap."""
    theta_deg = np.linspace(0.5, 30.0, 300)  # root is at 0.081 deg
    c = 25.0
    D = c * np.radians(theta_deg) ** 2
    lam2 = 100.0
    got = dsp.coherence_angle(theta_deg, D, lam2)
    assert got == np.radians(theta_deg[0])


def test_patch_counts_and_tilt():
    phi_col = np.array([1.5, 1.6, 5.5])
    w2 = np.array([1.0, 1.0, 2.0])
    edges = np.array([0.0, 3.0, 6.0])
    npatch = dsp.patch_counts(phi_col, w2, edges, 0.01, pix_area=1e-3)
    # bin 0: N_eff = (2)^2/2 = 2 -> 2 * 1e-3 / 1e-4 = 20
    # bin 1: N_eff = 1 -> 10
    np.testing.assert_allclose(npatch, [20.0, 10.0])
    H = np.array([2.0, 2.0])
    tilt = dsp.coherence_tilt(H, npatch)
    assert np.isclose(tilt.sum(), H.sum())
    assert tilt[0] > tilt[1]  # more patches -> boosted in the coherent limit


def test_amplitude_bracket_closed_forms():
    b = dsp.amplitude_bracket(
        lam2=99.86, theta_c=0.01, omega_beam=2 * np.pi, phi_med=18.4
    )
    assert np.isclose(b["upper"], 1.0 / np.sqrt(2 * np.pi / 1e-4))
    assert np.isclose(b["lower_slab"], 1.0 / (18.4 * 99.86))
    assert np.isclose(b["lower_dispersion"], 1.0 / (2.0 * 9.8**2 * 99.86**2))
    assert b["upper"] > b["lower_slab"] > b["lower_dispersion"]
