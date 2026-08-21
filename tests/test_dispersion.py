"""Analytic limits of the dispersion module (spec S6.3)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from lusee_faraday import dispersion as dsp
from lusee_faraday.config import PHI_FD_POINT, fine_freqs
from lusee_faraday.conventions import faraday_phase_cosmo, lambda_squared

FREQS_30 = fine_freqs(30.0)[::64]  # 256 fine frequencies, +-0.1 MHz


def test_phi_edges_width_and_span():
    edges = dsp.phi_edges(30.0)
    dphi = np.diff(edges)
    lam2_max = float(np.asarray(lambda_squared(29.9))[0])
    assert np.allclose(dphi, np.pi / (2 * lam2_max))
    assert np.isclose(dphi[0], 0.016, atol=2e-3)  # spec S3 number
    assert edges[0] <= -2500.0 and edges[-1] >= 2500.0
    # 10 MHz: 0.0017 rad/m^2 bins (spec S3)
    assert np.isclose(np.diff(dsp.phi_edges(10.0))[0], 0.0017, atol=2e-4)


def test_delta_is_pure_winding():
    """F = delta(phi - PHI_FD_POINT) -> the repo's COSMO Faraday phase."""
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(np.array([PHI_FD_POINT]), np.array([1.0]), lam2)
    expected = faraday_phase_cosmo(np.array([PHI_FD_POINT]), FREQS_30)[0]
    np.testing.assert_allclose(P, expected, rtol=0, atol=1e-9)


def test_tophat_is_sinc_with_the_right_factor():
    """F uniform on [0, Phi] -> |sin(Phi lam2)/(Phi lam2)|, NOT sinc(2...).

    Spec S6.3: under e^{+2i phi lam2}, Int_0^1 e^{2 i f Phi lam2} df has
    modulus |sin(Phi lam2)/(Phi lam2)|.
    """
    Phi = 25.0
    n = 1 << 17
    dphi = Phi / n
    phi = (np.arange(n) + 0.5) * dphi
    F = np.full(n, 1.0 / n)  # unit total emission
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    x = Phi * lam2
    expected = np.abs(np.sin(x) / x)
    keep = expected > 0.05
    np.testing.assert_allclose(np.abs(P)[keep], expected[keep], rtol=1e-3)


def test_gaussian_is_burn():
    """F Gaussian width sigma -> |P| = exp(-2 sigma^2 lam2^2).

    sigma=0.02: the exponent is 2 sigma^2 (lam2)^2 ~ 8.0 (O(1)), keeping
    expected ~ 3e-04 far above the ~5e-16 floor set by +-8 sigma truncation.
    """
    sigma = 0.02
    n = 1 << 15
    phi = np.linspace(-8 * sigma, 8 * sigma, n)
    F = np.exp(-0.5 * (phi / sigma) ** 2)
    F /= F.sum()
    lam2 = np.asarray(lambda_squared(FREQS_30), dtype=float)
    P = dsp.transform(phi, F, lam2)
    expected = np.exp(-2.0 * sigma**2 * lam2**2)
    np.testing.assert_allclose(np.abs(P), expected, rtol=1e-3)


def test_delay_power_recovers_a_single_depth():
    """delay_power inverts transform: peak at the injected depth."""
    phi0 = 120.0
    freqs = fine_freqs(30.0)[::16]  # 1024 points
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    spec = np.exp(2j * phi0 * lam2)
    phi_out = np.arange(0.0, 300.0, 0.25)
    p = dsp.delay_power(spec, freqs, phi_out)
    assert abs(phi_out[np.argmax(p)] - phi0) < 1.0
    assert np.isclose(p.max(), 1.0, rtol=1e-6)  # unit tone, normalized


def _fwhm(x, y):
    half = 0.5 * y.max()
    above = np.nonzero(y >= half)[0]
    return x[above[-1]] - x[above[0]]


def test_nufft_beats_fft_on_a_single_depth():
    """S6.8: the chirp is an analysis artifact; the NUFFT removes it.

    A single depth at 30 MHz, phi0 = 600 (chirp ~ 5 resolution elements
    per the spec's table): the uniform-nu FFT smears it wider than
    the type-3 NUFFT on the same samples. Measured ratio 2.36 at phi0=600
    (w_fft 4.719 vs w_nufft 2.000); the ratio grows with depth because
    the quadratic phase excursion scales with phi0 (3.99 rad at phi0=600,
    ratio 2.36; 7.99 rad at phi0=1200, ratio 8.26). The spec's quoted
    11.80/2.36 = 5.0 pair does not reproduce on a fine phi grid.
    """
    phi0 = 600.0
    freqs = fine_freqs(30.0)[::4]  # 4096 uniform samples
    lam2 = np.asarray(lambda_squared(freqs), dtype=float)
    spec = np.exp(2j * phi0 * lam2)

    phi_out = np.arange(560.0, 640.0, 0.05)
    p_nufft = dsp.delay_power(spec, freqs, phi_out)
    w_nufft = _fwhm(phi_out, p_nufft)

    # FFT on the uniform nu grid; map delay bins to phi by linearizing
    # lambda^2(nu) at the band centre. The linear phase rate is negative
    # due to the lambda^2 nonlinearity, so the FFT peak appears at -k.
    n = freqs.size
    P = np.fft.fftshift(np.fft.fft(spec)) / n
    dnu_hz = (freqs[1] - freqs[0]) * 1e6
    bw = n * dnu_hz
    nu0 = 30e6
    lam2_0 = float(lambda_squared(30.0)[0])
    # delay bin k <-> phase rate 2*pi*k/bw <-> phi = -pi*k*nu0/(2*bw*lam2_0)
    k = np.arange(n) - n // 2
    phi_fft = -np.pi * k * nu0 / (2.0 * bw * lam2_0)
    p_fft = np.abs(P) ** 2
    # phi_fft descends; reverse both axes so _fwhm works correctly
    phi_fft = phi_fft[::-1]
    p_fft = p_fft[::-1]
    sel = np.abs(phi_fft - phi0) < 60.0
    w_fft = _fwhm(phi_fft[sel], p_fft[sel])

    # The FFT chirp is wider than the NUFFT, but not by 4x on a fine grid.
    assert w_fft / w_nufft >= 2.0
    # Sign guard: FFT peak is within one bin of phi0.
    dphi_bin = phi_fft[1] - phi_fft[0]
    assert abs(phi_fft[np.argmax(p_fft)] - phi0) < dphi_bin
    # and the NUFFT peak is within one bin of the truth
    assert abs(phi_out[np.argmax(p_nufft)] - phi0) < 0.5


def test_bh4_window_sidelobe_level():
    """The 4-term Blackman-Harris peak sidelobe is ~ -92 dB (2.5e-5)."""
    n = 4096
    win = dsp.bh4_window(n)
    freqs = fine_freqs(30.0)[::4]
    phi_out = np.arange(0.0, 400.0, 0.1)
    p = dsp.delay_power(np.ones(n), freqs, phi_out, window=win)
    # main lobe is at phi = 0; measure the highest sidelobe beyond it
    side = p[phi_out > 15.0].max()
    assert np.sqrt(side) < 5e-5
    assert np.sqrt(side) > 5e-7  # a window this good would be a bug


from lusee_faraday.channelization import parent_weights, zoom_weights


def _fine_offsets():
    return np.arange(-50000.0, 50000.0 + 1.0, 12.20703125)


def test_boxcar_rmsf_widths():
    """S6.7: the boxcar RMSF against BOTH top-hat width conventions.

    Convention: AMPLITUDE FWHM (``delay_power`` returns power, so the
    half-max crossing is taken on ``sqrt(p)``), matching the
    convention of Brentjens & de Bruyn's ``2 sqrt(3) / dlambda^2``.

    The assertion is against the EXACT half-power width of a top-hat's
    sinc RMSF, ``2 * 1.8955 / dlambda^2`` (``sin(x)/x = 1/2`` at
    x = 1.895494), with the ``(n-1)/n`` correction for summing n
    discrete samples instead of integrating: a discrete sum spans
    ``n`` cells of width ``dlambda^2/(n-1)``, i.e. an effective
    ``dlambda^2 * n/(n-1)``.  ``2 sqrt(3) / dlambda^2`` = 3.4641 is
    a rule of thumb 9.44% BELOW that exact width and is printed as
    the spec's comparison, not asserted.

    History (the fifth vacuous test on this branch): the previous
    version asserted the measured width equalled ``2 sqrt(3)/dlambda^2``
    at rtol=0.08, and the measured relative error was
    0.08000000000000007 at all three bands -- it cleared only through
    ``np.isclose``'s default atol=1e-8, and only because taking the
    last grid point at or above half-max on a ``width/50`` grid rounds
    down by almost exactly the 9.4% the approximation is off by.
    Refining the grid or interpolating the crossing makes it fail.
    Ruling R6 said "if it ever flips, interpolate the crossing -- do
    NOT loosen rtol"; interpolating is what exposes it.
    """
    boxcar_rule = {50.0: 12.0, 30.0: 2.60, 10.0: 0.096}  # 2 sqrt(3)/dl2
    for band, rule in boxcar_rule.items():
        freqs = fine_freqs(band)[::8]
        n = freqs.size
        lam2 = np.asarray(lambda_squared(freqs), dtype=float)
        dlam2 = lam2[0] - lam2[-1]
        assert np.isclose(2.0 * np.sqrt(3.0) / dlam2, rule, rtol=0.01)
        exact = 2.0 * 1.895494267 / dlam2 * (n - 1) / n
        phi_out = np.arange(0.0, 40.0 * rule, rule / 50.0)
        amp = np.sqrt(dsp.delay_power(np.ones(n), freqs, phi_out))
        i = np.nonzero(amp >= 0.5)[0][-1]
        # interpolate the half-max crossing (R6); symmetric about 0
        cross = phi_out[i] + (amp[i] - 0.5) * (phi_out[i + 1] - phi_out[i]) / (
            amp[i] - amp[i + 1]
        )
        fwhm = 2.0 * cross
        assert np.isclose(fwhm, exact, rtol=1e-3), (band, fwhm, exact)
        print(
            f"\n{band:.0f} MHz boxcar RMSF amplitude FWHM {fwhm:.4f} "
            f"rad/m^2; exact sinc half-power {exact:.4f}; "
            f"2 sqrt(3)/dlam2 = {2.0 * np.sqrt(3.0) / dlam2:.4f}, "
            f"which the measured width exceeds by "
            f"{100 * (fwhm * dlam2 / (2 * np.sqrt(3)) - 1):.2f}%"
        )


def test_depth_horizon_pins_the_s46_table():
    """S6.9 (instrument side): 50% depths of the real bin responses.

    These expectations supersede the spec S4.6 draft table
    (parent {58.7, 13.3, 2.7}, zoom {2830, 613, 24}). Two invariants
    say the draft's 10 MHz parent entry is wrong: (1) the zoom/parent
    ratio is a band-independent bandwidth ratio -- measured ~48.1 at
    all three bands, while the draft gives 48.2, 46.1 and 8.9, with
    only the 10 MHz entry breaking the pattern; (2) the horizon scales
    as f^3 (lambda^2 ~ f^-2, and the envelope argument scales phi by
    lambda^2), which the measured values below satisfy to <1% and the
    draft's 2.7 does not.
    """
    off = _fine_offsets()
    wp = parent_weights(off)
    wz = zoom_weights(off)[:, 0]
    parent_expect = {50.0: 58.03, 30.0: 12.54, 10.0: 0.4643}
    zoom_expect = {50.0: 2796.9, 30.0: 604.1, 10.0: 22.38}
    for band in (50.0, 30.0, 10.0):
        hp_ = dsp.depth_horizon(off, wp, band)
        hz_ = dsp.depth_horizon(off, wz, band)
        assert np.isclose(hp_, parent_expect[band], rtol=0.05), (band, hp_)
        assert np.isclose(hz_, zoom_expect[band], rtol=0.05), (band, hz_)


def test_zoom_bin_matrix_shape_and_normalization():
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    assert W.shape == (fine.size, 192) and bins.size == 192
    np.testing.assert_allclose(W.sum(axis=0), 1.0, rtol=1e-9)
    assert np.all(np.diff(bins) > 0)


def test_rmsf_peaks_at_the_probe_depth_inside_the_horizon():
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    phi_out = np.arange(0.0, 200.0, 0.2)
    r = dsp.rmsf(100.0, fine, W, bins, phi_out)
    assert abs(phi_out[np.argmax(r)] - 100.0) < 1.0


def test_foreground_sidelobe_budget():
    """S6.10 / S4.8: a phi~0 leakage foreground at |P|/I = 0.15
    through BH4, resolved in phi.  This IS the S4.8 window
    dynamic-range deliverable; the numbers it prints are quoted in
    docs/measurement-model.md section 12.

    The contamination is strongly phi-dependent and a single number
    misreports it.  The spec's estimate -- peak sidelobe 2.5e-5 in
    amplitude, so 0.15 * 2.5e-5 = 3.8e-6 everywhere, "inadequate
    against the 1e-6 floor" -- is right only about the first sidelobe
    and wrong about the verdict:

      phi <~ 9.4   the BH4 MAIN LOBE (first null ~9.4 rad/m^2):
                   1.5e-1 falling to ~1e-5.  No protection at all,
                   and the k=0 template's origin spike lives here.
      10 - 27      first sidelobes, peak 3.7e-6 near phi = 10.8
                   (the spec's 3.8e-6).
      phi >= 27.5  <= 1e-6 from here outward.
      90 - 190     1.4e-7 across the 30 MHz knee (89.6 rad/m^2).
      phi > 200    <= 8.8e-8.
      phi > 776    <= 2.5e-8 (beyond the beam-weighted p99).

    So BH4 is ADEQUATE against both ends of the S4.4 bracket
    everywhere the template's roll-off lives -- at the knee it clears
    the 30 MHz internal-dispersion floor (5.2e-7) by 3.7x and the
    uniform-slab floor (4.3e-4) by 3000x -- and inadequate only inside
    its own main lobe, phi <~ 10, which is the low-phi core the delay
    axis was never claimed to protect (S4.8: leakage sits at phi ~ 0).
    """
    freqs = fine_freqs(30.0)[::4]  # 4096 samples
    # smooth synchrotron-sloped foreground at the PROGRESS.md level
    fg = 0.15 * (freqs / 30.0) ** (-2.5)
    win = dsp.bh4_window(freqs.size)
    phi_out = np.arange(0.0, 2500.0, 0.25)
    amp = np.sqrt(
        dsp.delay_power(fg.astype(complex), freqs, phi_out, window=win)
    )

    def peak(lo, hi=np.inf):
        sel = (phi_out > lo) & (phi_out <= hi)
        j = int(np.argmax(amp[sel]))
        return float(amp[sel][j]), float(phi_out[sel][j])

    side, phi_side = peak(10.5)  # past the BH4 main lobe's first null
    knee, _ = peak(90.0, 190.0)
    tail, _ = peak(200.0)
    far, _ = peak(776.1)
    assert 1e-6 < side < 5e-6, (side, phi_side)
    assert knee < 5e-7, knee
    assert tail < 2e-7, tail
    assert far < 1e-7, far
    assert tail > 1e-9  # sanity: the foreground exists
    # the level is cleared from ~27 rad/m^2 outward, well inside the
    # knee -- measured as the last phi at which 1e-6 is exceeded
    outward = np.maximum.accumulate(amp[::-1])[::-1]
    phi_clear = float(phi_out[np.nonzero(outward > 1e-6)[0][-1]])
    assert phi_clear < 40.0, phi_clear
    print(
        f"\nS4.8 BH4 budget (foreground 0.15 at phi ~ 0, 30 MHz):"
        f"\n  peak sidelobe      {side:.2e} at phi = {phi_side:.2f}"
        f"\n  <= 1e-6 for phi >= {phi_clear:.2f}"
        f"\n  knee 90-190        {knee:.2e}"
        f"\n  tail phi > 200     {tail:.2e}"
        f"\n  beyond p99 (776)   {far:.2e}"
        f"\n  vs bracket ends: 1e-4 -> {knee / 1e-4:.1e}, "
        f"1e-6 -> {knee / 1e-6:.1e} at the knee"
    )


def test_boxcar_would_fail_the_budget():
    """Without the window the foreground floods the roll-off region."""
    freqs = fine_freqs(30.0)[::4]
    fg = 0.15 * (freqs / 30.0) ** (-2.5)
    phi_out = np.arange(0.0, 2500.0, 1.0)
    p = dsp.delay_power(fg.astype(complex), freqs, phi_out)
    assert np.sqrt(p[phi_out > 200.0].max()) > 1e-5


def test_zoom_fold_is_unreachable_because_the_envelope_nulls_there():
    """S4.6 item 3: a depth cannot reach its own alias.

    The zoom delay range wraps with period
    dphi_wrap = 2 pi nu / (4 lam2 * 390.625 Hz) ~ 1208 rad/m^2 at
    30 MHz.  But the zoom is critically sampled (ENBW 563 Hz on
    390.6 Hz spacing), so a single bin's envelope has its first null
    at that same depth: the response tracks faithfully up to ~1100
    and is annihilated at the wrap rather than folding to an image.
    Aliased Faraday power is therefore not a contaminant here.

    phi_out MUST span past the wrap (0..1400 gives ~16% margin past
    1208).  A window that stops short of the true peak reports the
    largest residual bump instead, which is how the plan's original
    version of this test measured 12.5.

    The independent physics evidence: (1) the bin_envelope nulling at
    the wrap, and (2) the wrap-period arithmetic.  The rmsf regression
    pin verifies the wrapper has not drifted from its inline path,
    but does not add independent physics verification.
    """
    fine, bins, W = dsp.zoom_bin_matrix(30.0)
    lam2 = np.asarray(lambda_squared(fine), dtype=float)
    lam2_0 = float(lambda_squared(30.0)[0])
    wrap = 2.0 * np.pi * 30e6 / (4.0 * lam2_0 * 390.625)
    assert 1200.0 < wrap < 1215.0

    win = dsp.bh4_window(bins.size)
    phi_out = np.arange(0.0, 1400.0, 0.5)

    def recovered(phi0):
        tone = np.exp(2j * phi0 * lam2)
        p = dsp.delay_power(W.T @ tone, bins, phi_out, window=win)
        return phi_out[np.argmax(p)], p.max()

    # tracks faithfully well past the 604 rad/m^2 depth horizon
    for phi0 in (100.0, 604.0, 900.0):
        pk, power = recovered(phi0)
        assert abs(pk - phi0) < 1.0, (phi0, pk)
        assert power > 1e-2, (phi0, power)
        print(f"phi0 {phi0:5.0f} -> peak {pk:6.1f}  power {power:.2e}")

    # Regression pin on dsp.rmsf's wrapper, NOT independent physics:
    # rmsf runs the same tone -> W.T @ tone -> delay_power sequence
    # that recovered() runs inline, so this catches the wrapper
    # drifting from the inline path and nothing more.  The
    # independent evidence in this test is the bin_envelope block
    # below and the wrap arithmetic above.
    pk_900, _ = recovered(900.0)
    p_model = dsp.rmsf(900.0, fine, W, bins, phi_out, window=win)
    assert phi_out[np.argmax(p_model)] == pk_900
    np.testing.assert_allclose(
        p_model,
        dsp.delay_power(
            W.T @ np.exp(2j * 900.0 * lam2), bins, phi_out, window=win
        ),
        rtol=1e-12,
    )

    # at the wrap the bin envelope has nulled: no image survives
    off = _fine_offsets()
    wz = zoom_weights(off)[:, 0]
    env_1000 = dsp.bin_envelope(1000.0, off, wz, 30.0)
    env_wrap = dsp.bin_envelope(wrap, off, wz, 30.0)
    assert env_1000 > 0.1, env_1000
    assert env_wrap < 1e-3, env_wrap
    print(f"bin envelope at phi=1000: {env_1000:.2e}")
    print(f"bin envelope at wrap={wrap:.1f}: {env_wrap:.2e}")

    _, power_wrap = recovered(wrap)
    _, power_ref = recovered(100.0)
    ratio = power_wrap / power_ref
    assert ratio < 1e-4, (power_wrap, power_ref, ratio)
    print(f"power at wrap {wrap:.1f}: {power_wrap:.2e}")
    print(f"power at reference (100): {power_ref:.2e}")
    print(f"ratio (wrap/ref): {ratio:.2e}")
