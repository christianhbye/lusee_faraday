"""Acceptance gates on the real RM map (spec S6.1, S6.2, S6.4, S6.6)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest

from lusee_faraday import dispersion as dsp

DATA = Path(__file__).resolve().parents[1] / "data"
RM_FILE = DATA / "faraday2020v2.hdf5"

needs_rm = pytest.mark.skipif(
    not RM_FILE.exists(), reason="needs data/faraday2020v2.hdf5"
)


def _rm_map():
    import h5py

    with h5py.File(RM_FILE, "r") as f:
        return np.asarray(f["faraday_sky_mean"][:], dtype=float)


def _rm_at_nside(rm512, nside):
    """Resample the native-nside map to any nside.

    Note: the three nside pairs compared in gate 1 use different resampling
    methods: (256, 512) compares ud_grade-smoothed vs native, (512, 1024)
    compares native vs get_interp_val-upsampled, and (1024, 2048) compares
    two interpolated maps. This mixes resolution and regridding together.
    The measured tolerances are 50-500x inside budget, so the conclusion is
    robust, but a future reader tightening these gates should unify the
    operator first.
    """
    import healpy as hp

    if nside == 512:
        return rm512
    if nside < 512:
        return hp.ud_grade(rm512, nside)
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return hp.get_interp_val(rm512, th, ph)


def _normalized_template(rm, k):
    edges = dsp.phi_edges(30.0)
    w2 = np.full(rm.size, 1.0 / rm.size)
    H = dsp.depth_distribution(rm, w2, edges, k=k)
    return dsp.phi_centers(edges), H / H.sum()


def _cdf_distance(Ha, Hb):
    return float(np.abs(np.cumsum(Ha) - np.cumsum(Hb)).max())


@needs_rm
@pytest.mark.parametrize("k", [np.inf, 0.0])
def test_gate1_shape_invariance_under_refinement(k):
    """S6.1: nside 256/512/1024/2048 templates agree; the old coherent
    amplitude falls.  Tolerance: 1% Kolmogorov distance, 2% knee shift.

    The gated roll-off statistic is the 90% mass quantile (Ruling
    R10), not the half-power knee: for k=0 the folded template has a
    spike at the origin (each pixel's 1/phi density diverges as
    phi -> 0), which drags the peak-relative half-power knee to the
    spike's edge and makes it fail the refinement tolerance on grid
    quantisation alone.  Both statistics are printed so the rejected
    one's instability is on the record.

    **What this gate does NOT test.** The nside 1024 and 2048 legs are
    ``hp.get_interp_val`` upsamplings of the native nside-512 map:
    they carry no sky information the 512 map does not already have,
    so their agreement bounds the interpolation and the binning, not
    the sky. Only the (256, 512) pair changes the information content,
    and it does so by ``ud_grade`` smoothing. The gate is a
    pixelisation-stability statement about the MODEL's shape; that the
    OBSERVABLE equals the weighted depth distribution is a separate
    claim, tested by
    ``test_delay_power_equals_the_weighted_depth_distribution``.
    """
    from lusee_faraday.conventions import lambda_squared

    rm512 = _rm_map()
    lam2_0 = float(lambda_squared(30.0)[0])
    templates, knees, half_knees, coherent = {}, {}, {}, {}
    for nside in (256, 512, 1024, 2048):
        rm = _rm_at_nside(rm512, nside)
        c, H = _normalized_template(rm, k)
        templates[nside] = H
        folded = dsp.fold_template(c, H)
        knees[nside] = dsp.mass_quantile_knee(*folded)
        half_knees[nside] = dsp.half_power_knee(*folded)
        # the audit's shot-noise observable: |mean e^{2 i phi lam2}|^2
        z = np.exp(2j * rm * lam2_0)
        coherent[nside] = abs(z.mean()) ** 2
    pairs = [(256, 512), (512, 1024), (1024, 2048)]
    for a, b in pairs:
        d = _cdf_distance(templates[a], templates[b])
        assert d <= 0.01, (a, b, d)
        rel = abs(knees[a] - knees[b]) / knees[b]
        assert rel <= 0.02, (a, b, knees)
    # contrast: the coherent power is NOT invariant (it fell ~1/N_pix)
    assert coherent[2048] < 0.5 * coherent[256], coherent
    print(
        f"\nk={k}: mass-90% knees {knees}; "
        f"half-power knees {half_knees}; coherent power {coherent}"
    )


@needs_rm
def test_gate2_shape_invariance_under_null_rotation():
    """S6.2: a rigid grid rotation is physically null.  It moved the old
    |P| by 7.2x; the normalised template must be stable.

    Scope, stated because it is easy to overclaim: with the uniform
    ``w2`` used here the normalised template is a functional of the
    empirical RM distribution alone, and a rigid rotation is a pixel
    permutation, so the only thing that can move it is
    ``rotate_map_pixel``'s resampling.  That is exactly the audit's
    complaint about the OLD observable -- the coherent sum moved 7.2x
    under the same null operation -- so the contrast is the result;
    but the gate is not evidence for the incoherent-limit identity
    (see ``test_delay_power_equals_the_weighted_depth_distribution``),
    and it cannot see the depth-dependent coherence tilt of S4.4.1,
    which is a physical shape systematic rather than a pixelisation
    one.
    """
    import healpy as hp

    rm = _rm_map()
    rot = hp.Rotator(rot=(40.0, 25.0, 10.0), deg=True)
    rm_rot = rot.rotate_map_pixel(rm)
    for k in (np.inf, 0.0):
        c, H = _normalized_template(rm, k)
        _, Hr = _normalized_template(rm_rot, k)
        d = _cdf_distance(H, Hr)
        assert d <= 0.01, (k, d)
        knee = dsp.mass_quantile_knee(*dsp.fold_template(c, H))
        knee_r = dsp.mass_quantile_knee(*dsp.fold_template(c, Hr))
        assert abs(knee - knee_r) / knee <= 0.02


@needs_rm
def test_gate_knee_location_and_extent():
    """S6.4: knee between p50 and p99 of |RM|; support reaches max."""
    rm = _rm_map()
    c, H = _normalized_template(rm, 0.0)
    phi_abs, Hf = dsp.fold_template(c, H)
    knee = dsp.mass_quantile_knee(phi_abs, Hf)
    p50, p99 = np.percentile(np.abs(rm), [50.0, 99.0])
    assert p50 <= knee <= p99, (knee, p50, p99)
    # extent: nonzero mass out to the map maximum (lower bound, S4.2)
    mx = np.abs(rm).max()
    assert Hf[phi_abs > 0.98 * mx].sum() > 0


WMAP_FILE = DATA / "wmap_band_iqumap_r9_9yr_K_v5.fits"

needs_wmap = pytest.mark.skipif(
    not WMAP_FILE.exists(), reason="needs the WMAP K map"
)


def _wmap_qu():
    import healpy as hp
    from astropy.io import fits

    from lusee_faraday.config import T_CMB

    x = 6.62607015e-34 * 23e9 / (1.380649e-23 * T_CMB)
    fconv = x**2 * np.exp(x) / (np.exp(x) - 1) ** 2
    with fits.open(WMAP_FILE) as h:
        d = h["Stokes Maps"].data
        Q = d["Q_POLARISATION"].astype(np.float64) * 1e-3 * fconv
        U = d["U_POLARISATION"].astype(np.float64) * 1e-3 * fconv
    return hp.reorder(Q, n2r=True), hp.reorder(U, n2r=True)


@needs_rm
@needs_wmap
def test_converged_regime_points_match_direct_sum():
    """S6.6: the RM x 0.02 positive control.  In the converged regime
    the type-3 NUFFT on raw pixel depths reproduces the direct coherent
    sum to four digits, with the real polarised sky as weights.
    """
    from lusee_faraday.config import fine_freqs
    from lusee_faraday.conventions import lambda_squared

    rm = 0.02 * _rm_map()
    Q, U = _wmap_qu()
    c = (Q + 1j * U) / len(rm)
    freqs = fine_freqs(30.0)[::256]  # 64 frequencies
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    # direct chunked sum
    direct = np.zeros(lam2.size, dtype=complex)
    for i in range(0, rm.size, 500_000):
        s = slice(i, i + 500_000)
        direct += np.exp(2j * np.outer(lam2, rm[s])) @ c[s]
    nufft = dsp.transform(rm, c, lam2)
    np.testing.assert_allclose(nufft, direct, rtol=1e-4)


@needs_rm
@needs_wmap
def test_delay_power_equals_the_weighted_depth_distribution():
    """The load-bearing identity itself (spec S3), which gates 1 and 2
    do not test: for a sky whose pixels are incoherent,

        <|P~(phi)|^2>  =  the |w|^2-weighted depth distribution,

    convolved with the window's own delay response.  Gates 1/2 test
    that the MODEL side of that equation is pixelisation-stable; this
    one computes the LEFT side from the coherent pixel sum -- the same
    sum the 2026-08-18 audit found to be shot noise in amplitude --
    and compares it against the right side computed by
    ``depth_distribution``.

    Both sides come from the real sky at native nside 512: depths are
    ``faraday2020v2``, weights are the WMAP K polarised sky
    ``c = (Q + iU)/N`` (so the pixel PHASES are the sky's own
    polarisation angles, not a random draw), transformed over the
    30 MHz +-0.1 MHz fine grid through BH4.

    MEASURED, integrated over ``30 <= |phi| < 1500`` (below 30 the
    BH4 main lobe and the genuinely coherent low-|phi| pixels sit;
    above 1500 there is essentially no mass): ratio **1.069**, i.e.
    the identity holds to 7% -- an independent reproduction of the
    audit's 1.038 total-power ratio, which nothing else on this
    branch reproduces.  Per-band ratios 1.070 / 0.848 / 1.474 /
    1.075 / 1.113; over the whole axis the ratio is 1.33, inflated by
    a factor 1.92 inside |phi| < 10 where the pixel sum really is
    partly coherent.

    Non-vacuity: run on ``RM x 0.02`` -- the converged/coherent
    positive control of S6.6, where the whole depth distribution
    collapses inside one RMSF width and the incoherent limit does NOT
    apply -- the model puts essentially no mass above |phi| = 90
    while the measurement does, by >1e4x. The statistic can fail, and
    it fails exactly where the physics says it should.
    """
    from lusee_faraday.config import fine_freqs
    from lusee_faraday.conventions import lambda_squared

    rm = _rm_map()
    Q, U = _wmap_qu()
    c = (Q + 1j * U) / rm.size
    freqs = fine_freqs(30.0)[::8]  # 2048 pts; phi wraps only at ~4833
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    win = dsp.bh4_window(freqs.size)
    phi_out = np.arange(-2500.0, 2500.0, 1.0)
    edges = np.arange(-2500.5, 2500.5, 1.0)  # bin centres = phi_out
    ker_phi = np.arange(-60.0, 61.0, 1.0)
    ker = dsp.delay_power(np.ones(freqs.size), freqs, ker_phi, window=win)

    def sides(depths):
        spec = dsp.transform(depths, c, lam2)
        meas = dsp.delay_power(spec, freqs, phi_out, window=win)
        F = dsp.depth_distribution(depths, np.abs(c) ** 2, edges, k=np.inf)
        return meas, np.convolve(F, ker, mode="same")

    meas, model = sides(rm)
    a = np.abs(phi_out)
    sel = (a >= 30.0) & (a < 1500.0)
    ratio = meas[sel].sum() / model[sel].sum()
    assert 0.8 < ratio < 1.3, ratio
    bands = [(30, 90), (90, 186), (186, 400), (400, 776), (776, 1500)]
    per_band = []
    for lo, hi in bands:
        s = (a >= lo) & (a < hi)
        per_band.append(meas[s].sum() / model[s].sum())
        assert 0.5 < per_band[-1] < 2.0, (lo, hi, per_band[-1])

    # coherent control: the identity must NOT hold at RM x 0.02
    meas_c, model_c = sides(0.02 * rm)
    tail = a >= 90.0
    assert meas_c[tail].sum() > 1e4 * model_c[tail].sum(), (
        meas_c[tail].sum(),
        model_c[tail].sum(),
    )
    print(
        f"\ndelay power / weighted depth distribution: {ratio:.4f} over "
        f"30 <= |phi| < 1500 (audit's total-power ratio 1.038)"
        f"\n  per band {[f'{r:.3f}' for r in per_band]}"
        f"\n  whole axis {meas.sum() / model.sum():.4f}; "
        f"|phi| < 10 alone "
        f"{meas[a < 10].sum() / model[a < 10].sum():.4f}"
        f"\n  RM x 0.02 control, |phi| >= 90: measured/model = "
        f"{meas_c[tail].sum() / max(model_c[tail].sum(), 1e-300):.3e}"
    )


@needs_rm
def test_gate_envelope_orderings_against_the_sky():
    """S6.9 (sky side): the orderings the paper's claims rest on."""
    rm = np.abs(_rm_map())
    p50, p90, p99, p999 = np.percentile(rm, [50.0, 90.0, 99.0, 99.9])
    mx = rm.max()
    # pin the map percentiles themselves (loose -- conclusions, not digits)
    for got, want in [
        (p50, 18.4),
        (p90, 91.0),
        (p99, 278.0),
        (p999, 648.8),
        (mx, 2442.1),
    ]:
        assert np.isclose(got, want, rtol=0.02), (got, want)
    off = np.arange(-50000.0, 50001.0, 12.20703125)
    from lusee_faraday.channelization import parent_weights, zoom_weights

    wp, wz = parent_weights(off), zoom_weights(off)[:, 0]
    assert dsp.depth_horizon(off, wz, 50.0) > mx
    z30 = dsp.depth_horizon(off, wz, 30.0)
    assert p999 / 1.5 < z30 < p999 * 1.5
    z10 = dsp.depth_horizon(off, wz, 10.0)
    # S4.6/S6.9: the 10 MHz zoom horizon is the only one that misses
    # the p90 knee, and it lands AT the median rather than below it.
    # The spec's "below the median" wording is not reproducible --
    # neither its own table (24.0) nor the measured value (22.38) is
    # below p50 (18.36).
    assert z10 < p90, (z10, p90)
    assert 0.5 * p50 < z10 < 1.5 * p50, (z10, p50)
    for band in (50.0, 30.0, 10.0):
        assert dsp.depth_horizon(off, wp, band) < p90


def _transiting_beam_regime(
    seed=0,
    npix=20000,
    nlst=40,
    beam_width=0.03,
    floor_amp=3e-3,
    floor_width=0.15,
    hot_width=0.05,
    hot_lo=200.0,
    hot_hi=800.0,
):
    """Synthetic |RM| field + LST-resolved w2, shaped like the real
    regime (spec S6.14): a real transiting beam is near-zero over
    most of the sky at most LSTs, with its weight concentrated on a
    small, shifting patch as the sky rotates under a fixed pointing.

    Pixels sit on a periodic 1-D coordinate (a stand-in for the great
    circle the zenith beam sweeps through as the sky turns). A narrow
    "hot" arc holds most of the very-high-|RM| pixels, standing in for
    the Galactic-plane/GC region. Each LST's weight is a narrow
    Gaussian "spotlight" centred at a different point on the circle
    (the beam), plus a much weaker, broader Gaussian floor (sidelobe
    leakage) so weight is never exactly zero anywhere. Whether an
    LST's spotlight currently overlaps the hot arc, and how much of it
    is sidelobe-only leakage, is what makes the tail fraction swing
    over orders of magnitude instead of clustering near a single
    number -- unlike a diffuse/uniform synthetic field, which cannot
    reproduce that regime (see Fix round 2 in the task report).

    Returns ``(rm_abs, edges, tail_hist, w2_band, w2_all)``:
    ``edges``/``tail_hist`` are exactly what the script builds
    (``dsp.tail_gate_bins`` + a per-LST ``np.bincount``); ``w2_band``
    is the LST-summed weight (matches the script's ``w2_band``);
    ``w2_all`` is the raw per-pixel per-LST weight, kept only so a
    per-LST threshold can be recomputed for the divergence test below
    (this replicates the taut form's own pre-R19 inputs -- not
    additional gate arithmetic).
    """
    rng = np.random.default_rng(seed)
    x = np.arange(npix) / npix
    rm_abs = 5.0 + 3.0 * np.abs(rng.standard_normal(npix))
    hot_center = 0.5
    d = np.minimum(np.abs(x - hot_center), 1.0 - np.abs(x - hot_center))
    hot_mask = d < hot_width
    rm_abs[hot_mask] += rng.uniform(hot_lo, hot_hi, size=hot_mask.sum())

    lst_centers = np.arange(nlst) / nlst
    w2_all = np.zeros((nlst, npix))
    for il in range(nlst):
        dcen = np.minimum(
            np.abs(x - lst_centers[il]), 1.0 - np.abs(x - lst_centers[il])
        )
        w2 = np.exp(-0.5 * (dcen / beam_width) ** 2)
        w2 += floor_amp * np.exp(-0.5 * (dcen / floor_width) ** 2)
        w2_all[il] = w2

    edges, idx = dsp.tail_gate_bins(rm_abs)
    tail_hist = np.zeros((nlst, 2000))
    for il in range(nlst):
        tail_hist[il] = np.bincount(idx, weights=w2_all[il], minlength=2000)
    w2_band = w2_all.sum(axis=0)
    return rm_abs, edges, tail_hist, w2_band, w2_all


def test_tail_gate_transiting_beam_dynamic_range():
    """S6.14/Ruling R19 regression test, via the SHIPPED code path.

    Calls the exact functions ``scripts/step5_template.py`` calls --
    ``dsp.tail_gate_bins`` for the fixed |RM| binning and
    ``dsp.tail_gate_fractions`` for the threshold-to-fraction
    arithmetic -- with a FIXED per-band threshold (the beam-weighted
    p99 of |RM| over the LST-summed weight, exactly as the script
    computes ``p99_band``), on a synthetic transiting-beam regime
    whose tail fraction spans orders of magnitude like the real run
    (script log: 2.06e-06 to 1.98e-02 at 30 MHz), not the 0.8-1.2%
    band a uniform/diffuse synthetic field produces.

    Measured on this fixture (seed=0, see Fix round 2 in the task
    report for the exact figures): fraction range
    [2.705314e-06, 8.999003e-02], ratio 3.326e4 (~4.52 decades).  The
    ``> 1e3`` bound below sits ~33x under that measured ratio while
    sitting ~500x above the ratio a collapsed-to-~1% tautological
    output would give (measured separately at ~1.9, see
    ``test_tail_gate_correct_form_diverges_from_tautological`` below)
    -- ample margin on both sides.
    """
    rm_abs, edges, tail_hist, w2_band, _ = _transiting_beam_regime()

    p99_band = dsp.weighted_percentiles(rm_abs, w2_band, [99.0])[0]
    frac = dsp.tail_gate_fractions(edges, tail_hist, p99_band)

    assert frac.min() < 1e-4, frac.min()
    assert frac.max() > 1e-2, frac.max()
    ratio = frac.max() / frac.min()
    assert ratio > 1e3, ratio

    print(
        f"\ntransiting-beam tail fraction: min={frac.min():.6e} "
        f"max={frac.max():.6e} ratio={ratio:.3e}"
    )


def test_tail_gate_correct_form_diverges_from_tautological():
    """Real discriminator between the fixed-threshold (correct, R19)
    and per-LST-threshold (tautological, pre-R19) forms -- NOT a
    restatement that the tautological form equals ~1%, which is true
    by the definition of a percentile and cannot fail.

    Both forms are built from the same two extracted functions
    (``dsp.tail_gate_bins``, ``dsp.tail_gate_fractions`` -- no
    parallel gate reimplementation); they differ only in whether the
    threshold passed to ``tail_gate_fractions`` is the one FIXED
    per-band value (correct) or recomputed per LST from that LST's
    own raw weight (tautological, the pre-R19 bug's own inputs). The
    assertion is on the size of the disagreement between the two
    forms, which is a real, independently falsifiable quantity: if a
    future change made the two forms coincide (e.g. by making the
    "fixed" threshold secretly vary per row again), this fails.

    Measured on the same fixture (seed=0): tautological fractions sit
    in [5.252e-03, 9.978e-03] (near 1%, as expected, off it only by
    the 2000-bin quantisation) while the correct-form fractions reach
    8.999e-02; max |correct - tautological| = 8.001e-02. The `> 1e-2`
    bound below sits 8x under that measured value.
    """
    rm_abs, edges, tail_hist, w2_band, w2_all = _transiting_beam_regime()

    p99_band = dsp.weighted_percentiles(rm_abs, w2_band, [99.0])[0]
    correct = dsp.tail_gate_fractions(edges, tail_hist, p99_band)

    nlst = tail_hist.shape[0]
    tautological = np.zeros(nlst)
    for il in range(nlst):
        p99_lst = dsp.weighted_percentiles(rm_abs, w2_all[il], [99.0])[0]
        tautological[il] = dsp.tail_gate_fractions(
            edges, tail_hist[il], p99_lst
        )

    diff = np.abs(correct - tautological)
    assert diff.max() > 1e-2, diff.max()

    print(
        f"\ncorrect form: min={correct.min():.6e} max={correct.max():.6e}"
        f"\ntautological form: min={tautological.min():.6e} "
        f"max={tautological.max():.6e}"
        f"\nmax |correct - tautological| = {diff.max():.6e}"
    )


def test_tail_gate_extraction_is_numerically_inert():
    """Mandatory inertness proof: the refactor that moved the tail-gate
    arithmetic out of ``scripts/step5_template.py`` (which built the
    committed, NOT-regenerated ``generated_data/step5_template*.npz``)
    into ``dsp.tail_gate_bins``/``dsp.tail_gate_fractions`` is a pure
    code move.  This runs a VERBATIM copy of the OLD inline arithmetic
    (as it stood at commit 7d23e07, before this refactor) side by side
    with the NEW extracted functions on identical synthetic inputs and
    requires bitwise equality (``assert_array_equal``), not a
    tolerance.
    """
    rng = np.random.default_rng(7)
    npix = 5000
    nlst = 12
    rm_abs = np.abs(rng.standard_normal(npix)) * 200.0 + 50.0
    # skewed, beam-like weight: most mass near zero, occasional spikes
    w2_all = rng.random((nlst, npix)) ** 3
    w2_band = w2_all.sum(axis=0)

    # --- OLD verbatim inline arithmetic ---
    old_rm_bin_edges = np.linspace(0.0, rm_abs.max(), 2001)
    old_rm_idx = np.clip(
        np.searchsorted(old_rm_bin_edges, rm_abs, side="right") - 1,
        0,
        1999,
    )
    old_tail_hist = np.zeros((nlst, 2000))
    for il in range(nlst):
        old_tail_hist[il] = np.bincount(
            old_rm_idx, weights=w2_all[il], minlength=2000
        )
    old_p99_band = dsp.weighted_percentiles(rm_abs, w2_band, [99.0])[0]
    old_above = old_rm_bin_edges[:-1] > old_p99_band
    old_tail = np.zeros(nlst)
    for il in range(nlst):
        old_tail[il] = (
            old_tail_hist[il][old_above].sum() / old_tail_hist[il].sum()
        )

    # --- NEW: extracted functions, same call pattern as the script ---
    new_rm_bin_edges, new_rm_idx = dsp.tail_gate_bins(rm_abs)
    new_tail_hist = np.zeros((nlst, 2000))
    for il in range(nlst):
        new_tail_hist[il] = np.bincount(
            new_rm_idx, weights=w2_all[il], minlength=2000
        )
    new_p99_band = dsp.weighted_percentiles(rm_abs, w2_band, [99.0])[0]
    new_tail = dsp.tail_gate_fractions(
        new_rm_bin_edges, new_tail_hist, new_p99_band
    )

    np.testing.assert_array_equal(new_rm_bin_edges, old_rm_bin_edges)
    np.testing.assert_array_equal(new_rm_idx, old_rm_idx)
    np.testing.assert_array_equal(new_tail, old_tail)
