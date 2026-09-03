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


# --------------------------------------------------------------------
# The binned pushforward: the geometry scan re-analyses the committed
# k -> inf template instead of re-running the 20-40 minute template
# job, so it must be the SAME operator depth_distribution applies.


def test_pushforward_histogram_k_infinite_is_the_identity():
    phi = np.arange(0.5, 9.0, 1.0)
    H = np.array([3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0])
    out = dsp.pushforward_histogram(phi, H, np.inf)
    np.testing.assert_allclose(out, H)
    assert out is not H  # a copy, so callers cannot alias the product


def test_pushforward_histogram_k_zero_is_a_tophat():
    """One populated bin at phi=3.5 spreads uniformly onto [0, 3.5]."""
    phi = np.arange(0.5, 6.0, 1.0)  # centres 0.5 .. 5.5, edges 0..6
    H = np.zeros(phi.size)
    H[3] = 1.0  # the bin centred at 3.5
    out = dsp.pushforward_histogram(phi, H, 0.0)
    # uniform density 1/3.5 on [0, 3.5]: three full bins then a part bin
    expected = np.array([1.0, 1.0, 1.0, 0.5, 0.0, 0.0]) / 3.5
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert np.isclose(out.sum(), H.sum())


@pytest.mark.parametrize("k", [np.inf, 4.0, 1.0, 0.0, -0.5, -1.0])
def test_pushforward_histogram_conserves_mass(k):
    """It REDISTRIBUTES power in depth; it never reweights it."""
    rng = np.random.default_rng(7)
    phi = np.arange(0.5, 200.0, 1.0)
    H = rng.gamma(2.0, 1.0, size=phi.size)
    out = dsp.pushforward_histogram(phi, H, k)
    assert np.isclose(out.sum(), H.sum(), rtol=1e-12)
    assert np.all(out >= 0.0)


def test_pushforward_histogram_matches_depth_distribution():
    """The binned operator equals the per-pixel one on the same sky.

    This is the claim the geometry scan rests on: re-casting the stored
    k -> inf histogram is the same calculation as re-running
    depth_distribution from the pixels, up to the display binning.
    """
    rng = np.random.default_rng(11)
    n = 20_000
    phi_col = rng.gamma(2.0, 40.0, size=n) * rng.choice([-1.0, 1.0], n)
    w2 = rng.gamma(3.0, 1.0, size=n)
    dphi = 1.0
    # The grid MUST span the sky: k=inf is a np.histogram and drops
    # out-of-range columns outright, while a finite-k pushforward of the
    # same sky still lands mass from those columns inside the grid.  The
    # committed products satisfy this (PHI_SPAN 2500 against |RM|max
    # 2442); a test grid that did not would compare two different skies.
    span = dphi * np.ceil(np.abs(phi_col).max() / dphi + 1.0)
    edges = np.arange(-span, span + dphi, dphi)
    cent = dsp.phi_centers(edges)
    _, far = dsp.fold_template(
        cent, dsp.depth_distribution(phi_col, w2, edges, k=np.inf)
    )
    far_c, _ = dsp.fold_template(cent, np.zeros_like(cent))
    for k in (4.0, 1.0, 0.0, -0.5):
        _, direct = dsp.fold_template(
            cent, dsp.depth_distribution(phi_col, w2, edges, k=k)
        )
        binned = dsp.pushforward_histogram(far_c, far, k)
        c1 = np.cumsum(direct) / direct.sum()
        c2 = np.cumsum(binned) / binned.sum()
        ks = np.abs(c1 - c2).max()
        assert ks < 2e-3, f"k={k}: KS {ks:.2e} between binned and per-pixel"


def test_retained_fraction_matches_summing_the_pushforward():
    phi = np.arange(0.5, 300.0, 1.0)
    rng = np.random.default_rng(3)
    H = rng.gamma(2.0, 1.0, size=phi.size)
    for k in (np.inf, 2.0, 0.0, -0.5):
        out = dsp.pushforward_histogram(phi, H, k)
        want = out[phi >= 27.5].sum() / out.sum()
        assert np.isclose(dsp.retained_fraction(phi, H, 27.5, k), want)


def test_retained_fraction_falls_monotonically_with_k():
    """Moving emission toward the observer can only remove power from
    above a cut: f(k) is nondecreasing in k."""
    phi = np.arange(0.5, 500.0, 1.0)
    rng = np.random.default_rng(5)
    H = rng.gamma(2.0, 20.0, size=phi.size)
    ks = [-0.9, -0.5, 0.0, 1.0, 4.0, 16.0, np.inf]
    fs = [dsp.retained_fraction(phi, H, 27.5, k) for k in ks]
    assert np.all(np.diff(fs) > 0.0), fs
    assert fs[-1] == dsp.retained_fraction(phi, H, 27.5, np.inf)


def test_pushforward_histogram_rejects_a_grid_missing_the_origin():
    """A grid whose first edge is above zero would silently drop the
    zero-depth mass the pushforward creates."""
    phi = np.arange(10.5, 20.0, 1.0)
    H = np.ones(phi.size)
    with pytest.raises(ValueError, match="first edge is zero"):
        dsp.pushforward_histogram(phi, H, 0.0)


def test_pushforward_histogram_rejects_a_nonuniform_grid():
    phi = np.array([0.5, 1.5, 3.5, 4.5])
    with pytest.raises(ValueError, match="uniform grid"):
        dsp.pushforward_histogram(phi, np.ones(4), 0.0)


def test_pushforward_histogram_below_minus_one_raises():
    phi = np.arange(0.5, 10.0, 1.0)
    with pytest.raises(ValueError, match="not integrable"):
        dsp.pushforward_histogram(phi, np.ones(phi.size), -1.5)


def test_pushforward_suffix_sum_does_not_cancel_at_large_k():
    """Pins the numerical form of the tail sum.

    The summands w * p^-q span (p_max/p_min)^q, so forming the tail as
    ``total - prefix`` cancels catastrophically: it returned exactly the
    k -> inf histogram above k ~ 8 on the committed display grid, and
    was already ~40% wrong at k = 2 on the real RM map (whose smallest
    |RM| is far below one bin).  Checked against the direct O(n*m) CDF
    difference, which has no cancellation to lose.
    """
    phi = np.arange(0.5, 500.0, 1.0)
    w = np.linspace(1.0, 2.0, phi.size)
    edges = np.arange(0.0, 501.0, 1.0)
    for k in (0.0, 2.0, 6.0, 12.0):
        got = dsp._pushforward_onesided(phi, w, edges, k)
        cdf = np.minimum(edges[None, :] / phi[:, None], 1.0) ** (k + 1.0)
        want = (w[:, None] * np.diff(cdf, axis=1)).sum(axis=0)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_pushforward_approaches_k_infinity_without_reaching_it():
    """f(k) rises strictly toward the k -> inf asymptote and never
    snaps onto it -- the signature the cancellation bug produced."""
    phi = np.arange(0.5, 2500.0, 1.0)
    rng = np.random.default_rng(17)
    H = rng.gamma(2.0, 30.0, size=phi.size)
    ks = np.array([4.0, 8.0, 12.0, 24.0, 40.0])
    fs = np.array([dsp.retained_fraction(phi, H, 27.5, k) for k in ks])
    f_inf = dsp.retained_fraction(phi, H, 27.5, np.inf)
    assert np.all(np.diff(fs) > 0.0), fs
    assert np.all(fs < f_inf), (fs, f_inf)


def test_pushforward_raises_rather_than_overflowing():
    """A geometry too steep for the grid must raise, not return nan."""
    phi = np.arange(0.5, 2500.0, 1.0)
    with pytest.raises(ValueError, match="too steep"):
        dsp.pushforward_histogram(phi, np.ones(phi.size), 500.0)


# --------------------------------------------------------------------
# The SIGNED pushforward.  The matched filter needs the sign (the
# observable is the complex P = Q + iU), so the geometry re-cast has to
# work on the signed grid too, and must stay consistent with the folded
# one it is derived from.


def _signed_grid(n=200, dphi=1.0):
    return dphi * (np.arange(-n, n) + 0.5)


def test_pushforward_signed_folds_to_the_folded_pushforward():
    """The consistency that makes the two paths one calculation:
    re-casting the signed histogram and folding the result equals
    re-casting the folded histogram directly."""
    phis = _signed_grid()
    rng = np.random.default_rng(19)
    far = rng.gamma(2.0, 1.0, size=phis.size)
    pos = phis > 0
    far_fold = far[pos] + far[phis < 0][::-1]
    for k in (np.inf, 4.0, 1.0, 0.0, -0.5):
        out = dsp.pushforward_signed(phis, far, k)
        folded = out[pos] + out[phis < 0][::-1]
        direct = dsp.pushforward_histogram(phis[pos], far_fold, k)
        np.testing.assert_allclose(folded, direct, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("k", [np.inf, 2.0, 0.0, -0.5, -1.0])
def test_pushforward_signed_conserves_mass_and_sign_support(k):
    phis = _signed_grid()
    rng = np.random.default_rng(23)
    far = rng.gamma(2.0, 1.0, size=phis.size)
    out = dsp.pushforward_signed(phis, far, k)
    assert np.isclose(out.sum(), far.sum(), rtol=1e-12)
    # mass never crosses the origin: each sign is re-cast on its own
    # side, because a column at -phi rotates toward -phi, not +phi.
    assert np.isclose(out[phis > 0].sum(), far[phis > 0].sum(), rtol=1e-12)
    assert np.isclose(out[phis < 0].sum(), far[phis < 0].sum(), rtol=1e-12)


def test_pushforward_signed_rejects_an_asymmetric_grid():
    with pytest.raises(ValueError, match="symmetric about zero"):
        dsp.pushforward_signed(np.array([-1.5, -0.5, 0.5]), np.ones(3), 0.0)
