"""The point-source arm, ported onto luseepy's covariance assembly.

The step-1 scripts freeze the instrument at one native response channel
and vary only the Faraday phase across the fine grid.  These tests pin
the three pieces that port had to get right: the explicit impedance
freeze, the Moon term being off, and the direction-space kernel.

The published-number tests at the bottom read the stored
``generated_data/`` artifacts rather than running either script, so a
regression inside a script stays invisible until someone re-runs the
analyses; and with ``generated_data/`` absent those tests skip silently.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import channelization as chan  # noqa: E402
from lusee_faraday import _legacy_pixel as fp  # noqa: E402
from lusee_faraday import instrument as inst  # noqa: E402
from lusee_faraday import response as rsp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

RESPONSE = REPO / "data" / "BGL_v16" / "lusee_bgl_v16_response_v3.fits"
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)

CENTER_MHZ = 30.0
AUTO_CHANNELS = [0, 4, 7, 9]  # positions of (a, a) in PORT_PAIRS

needs_artifact = pytest.mark.skipif(
    not RESPONSE.exists(), reason="BGL_v16 response artifact not present"
)


@pytest.fixture(scope="module")
def synthetic():
    """A cheap four-port response with 30 MHz native and bracketed.

    Two departures from a bare ``synthetic_four_port_response``, both
    needed to make the step-1 wiring test able to fail:

    - Native channels at 29/30/31 MHz rather than 30/60.  The fine grid
      runs to 30 +- 0.025 MHz, so with 30 MHz as the *lowest* native
      channel luseepy rejects the grid outright instead of interpolating
      ``Z_A`` across it, and dropping the impedance freeze would fail
      loudly for the wrong reason.
    - A per-port phase ``diag(e^{i alpha})`` applied to the Jones
      components and, consistently, to ``Z_A``/``R`` as ``U Z U^H``.
      That is a unitary port gauge transform, so the response stays
      physical, but it makes the cross-pair ``K_Q``/``K_U`` complex.
      The stock synthetic response has them *exactly* real, and the
      historical sign bug this suite guards against -- writing
      ``cpol_m`` as ``conj(cpol_p)`` -- is then algebraically invisible.
    """
    lusee = pytest.importorskip("lusee")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.InstrumentResponse import InstrumentResponse
    from lusee.ReceiverImpedance import JFETReceiver

    base = lusee.synthetic_four_port_response(
        freq_mhz=(CENTER_MHZ - 1.0, CENTER_MHZ, CENTER_MHZ + 1.0),
        angular_step_deg=5.0,
    )
    u = np.exp(1j * np.array([0.0, 0.7, 1.3, 2.1]))
    gauge = u[:, None] * np.conj(u)[None, :]  # (4, 4)
    resp = InstrumentResponse.from_arrays(
        np.asarray(base.freq),
        np.asarray(base.theta_deg),
        np.asarray(base.phi_deg),
        np.asarray(base.H_theta) * u[:, None, None, None],
        np.asarray(base.H_phi) * u[:, None, None, None],
        np.asarray(base.ZA) * gauge,
        np.asarray(base.Rsky) * gauge,
        np.asarray(base.Rmoon) * gauge,
        np.asarray(base.Rloss) * gauge,
        validated=False,
        metadata={"VALIDATED": False},
    )
    return resp, JFETReceiver()


@pytest.fixture(scope="module")
def instrument_pieces():
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    return rsp.load_response(RESPONSE), JFETReceiver()


def random_pair_integrals(seed, shape):
    """Hermitian-consistent pair integrals: the four autos are real."""
    rng = np.random.default_rng(seed)
    pair = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    pair[..., AUTO_CHANNELS] = pair[..., AUTO_CHANNELS].real
    return pair


@pytest.mark.slow
@needs_artifact
def test_frozen_impedance_is_not_a_no_op(instrument_pieces):
    """``impedance_freq_mhz`` must both freeze and change the answer.

    Z_A is steep near 30 MHz, so letting it follow the fine grid injects
    a ~10% smooth spectral ramp -- exactly the non-Faraday chromatic
    structure the step-1 delay-space argument asserts is absent.
    """
    resp, receiver = instrument_pieces
    pair = random_pair_integrals(0, (2, 8, 10))
    wide = np.linspace(CENTER_MHZ - 0.1, CENTER_MHZ + 0.1, 8)
    narrow = np.linspace(CENTER_MHZ - 0.02, CENTER_MHZ + 0.02, 8)
    kw = dict(T_moon=0.0, T_ant=0.0, impedance_freq_mhz=CENTER_MHZ)

    frozen_wide = inst.covariance(pair, resp, receiver, wide, **kw)
    frozen_narrow = inst.covariance(pair, resp, receiver, narrow, **kw)
    np.testing.assert_allclose(frozen_narrow, frozen_wide, rtol=1e-14)

    chromatic = inst.covariance(
        pair, resp, receiver, wide, T_moon=0.0, T_ant=0.0
    )
    rel = np.abs(chromatic - frozen_wide).max() / np.abs(frozen_wide).max()
    assert rel > 1e-2, f"freezing the impedances changed nothing ({rel:.2e})"


@pytest.mark.slow
@needs_artifact
def test_the_freeze_covers_the_moon_and_loss_matrices(instrument_pieces):
    """All four matrices are frozen, not just ``Z_A`` and ``Z_L``.

    The step-1 runs pass ``T_moon = T_ant = 0``, which zeroes the
    ``R_moon``/``R_loss`` contributions and so cannot tell a four-way
    freeze from a two-way one.  Switching both temperatures on makes
    them load-bearing: freezing only ``Z_A``/``Z_L`` and letting
    ``R_moon``/``R_loss`` follow the fine grid then moves the answer by
    4e-2, far outside this tolerance.
    """
    resp, receiver = instrument_pieces
    pair = random_pair_integrals(3, (2, 8, 10))
    fine = np.linspace(CENTER_MHZ - 0.1, CENTER_MHZ + 0.1, 8)
    kw = dict(T_moon=250.0, T_ant=180.0)

    frozen = inst.covariance(
        pair, resp, receiver, fine, impedance_freq_mhz=CENTER_MHZ, **kw
    )
    constant = inst.covariance(
        pair, resp, receiver, np.full(fine.size, CENTER_MHZ), **kw
    )
    rel = np.abs(frozen - constant).max() / np.abs(constant).max()
    assert rel < 1e-14, f"relative difference {rel:.3e}"


@pytest.mark.slow
@needs_artifact
def test_covariance_matches_the_legacy_assembler(instrument_pieces):
    """The ported assembly must reproduce the published four-port arm."""
    resp, receiver = instrument_pieces
    kern = fp.FixedFreqKernel(resp, CENTER_MHZ, receiver)
    pair = random_pair_integrals(1, (2, 5, 10))
    fine = np.linspace(CENTER_MHZ - 0.1, CENTER_MHZ + 0.1, 5)

    legacy = fp.assemble_covariance(pair, kern.M)
    new = inst.covariance(
        pair,
        resp,
        receiver,
        fine,
        T_moon=0.0,
        T_ant=0.0,
        impedance_freq_mhz=CENTER_MHZ,
    )
    rel = np.abs(new - legacy).max() / np.abs(legacy).max()
    assert rel < 1e-12, f"relative difference {rel:.3e}"


@pytest.mark.slow
@needs_artifact
def test_moon_term_is_off_for_the_point_source_runs(instrument_pieces):
    """``T_moon`` defaults to 250 K; the step-1 runs need it at zero.

    The legacy assembler had no Moon term at all, so taking luseepy's
    default would move the answer by four orders of magnitude and
    destroy the rank-1 check.  What this test pins is that default:
    change ``T_moon=250.0`` in ``instrument.covariance``'s signature and
    it goes red.  It does not reach the scripts' call sites -- dropping
    the explicit ``T_moon=0.0`` from either step-1 script leaves this
    whole file green.
    """
    resp, receiver = instrument_pieces
    kern = fp.FixedFreqKernel(resp, CENTER_MHZ, receiver)
    pair = random_pair_integrals(1, (2, 5, 10))
    fine = np.linspace(CENTER_MHZ - 0.1, CENTER_MHZ + 0.1, 5)

    legacy = fp.assemble_covariance(pair, kern.M)
    with_moon = inst.covariance(
        pair, resp, receiver, fine, impedance_freq_mhz=CENTER_MHZ
    )
    rel = np.abs(with_moon - legacy).max() / np.abs(legacy).max()
    assert rel > 1e3, f"the Moon term is not showing up ({rel:.3e})"


@pytest.mark.slow
@needs_artifact
def test_fixed_channel_kernel_matches_the_legacy_kernel(instrument_pieces):
    """``FixedChannelKernel.sample`` returns the eta0/lambda^2 kernel.

    The legacy kernel is bare m^2 and the caller multiplied by
    ``kern.prefac``; the ported one applies that scaling itself, exactly
    as luseepy's own ``pair_stokes_alms`` does.
    """
    resp, receiver = instrument_pieces
    legacy = fp.FixedFreqKernel(resp, CENTER_MHZ, receiver)
    ported = rsp.FixedChannelKernel(resp, CENTER_MHZ)

    # The last phi sits in the 359-360 deg wraparound cell, which is
    # the one branch of sample_periodic_maps the other directions miss.
    theta = np.array([0.0, 0.017, 0.4, 1.0, np.pi / 2, 0.8])
    phi = np.array([0.0, 1.3, 3.14159, 4.7, 6.2, 2 * np.pi - 1e-3])
    want = legacy.prefac * legacy.sample(theta, phi)
    got = ported.sample(theta, phi)

    assert got.shape == (10, 4, theta.size)
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-13, f"relative difference {rel:.3e}"


def test_sampler_rejects_directions_below_the_horizon():
    """The stored response stops at theta = 90 deg; past it must raise.

    The step-1 chunk loop substitutes theta = 0 for the below-horizon
    times and zeros their pair integrals afterwards.  That is only safe
    because feeding a real below-horizon angle in would raise instead of
    silently clipping to the horizon ring.
    """
    theta_deg = np.arange(0.0, 91.0)
    phi_deg = np.arange(0.0, 361.0)
    values = np.zeros((2, theta_deg.size, phi_deg.size))
    ok = rsp.sample_periodic_maps(
        values,
        theta_deg,
        phi_deg,
        np.array([0.0, np.pi / 2]),
        np.array([0.0, 1.0]),
    )
    assert ok.shape == (2, 2)
    with pytest.raises(ValueError, match="outside the stored response"):
        rsp.sample_periodic_maps(
            values,
            theta_deg,
            phi_deg,
            np.array([0.1, 2.0]),
            np.array([0.0, 0.0]),
        )


def test_fine_grid_default_matches_common_fine_freqs():
    """``step1_point_source.fine_grid()`` is ``common.fine_freqs``.

    ``fine_grid`` exists only so a test can ask for a smaller grid than
    ``N_FINE``; it duplicates the formula, so something has to pin the
    duplication.  Bitwise, not approximately: the two arrays index the
    same fine channels of the same production waterfall.
    """
    from common import fine_freqs
    import step1_point_source as s1

    np.testing.assert_array_equal(s1.fine_grid(), fine_freqs(s1.CENTER_MHZ))
    assert s1.fine_grid(8).size == 8


def test_make_waterfall_wiring_end_to_end(synthetic, tmp_path, monkeypatch):
    """Run the step-1 waterfall's real physics at a size a test can hold.

    Until ``n_times``/``n_fine`` were added, ``make_waterfall`` was bound
    to 1024 x 16384 and opened a 2 GB memmap, so nothing in the suite
    could call it: Task 16's review gutted its physics and every test
    stayed green.  This is the guard for that, and it is a *wiring*
    guard -- it re-derives the same numbers by the textbook route and
    checks the script's optimised one agrees.

    The reference contracts the pair-Stokes kernel with the source's
    Stokes vector directly, ``K @ (1, cos 2chi, sin 2chi, 0)``, one time
    and one frequency at a time.  The script instead factorises that
    into ``K_I + 0.5 (K_Q -+ i K_U) e^{+-2i chi}`` so the frequency axis
    becomes an outer product, chunks the time axis, carries the
    below-horizon samples through zeroed, and streams to a memmap.  The
    two are algebraically identical and structurally nothing alike,
    which is the point: the historical sign bug (``cpol_m`` written as
    ``conj(cpol_p)``) lives in exactly that factorisation.

    The reference also spells out ``T_moon = T_ant = 0`` and the
    impedance freeze, so dropping either from the script's ``frozen``
    dict turns this red -- the call-site coverage
    ``test_moon_term_is_off_for_the_point_source_runs`` explicitly does
    not give.

    Measured RED evidence, six injections, all reverted:

    - ``cpol_m = conj(cpol_p)`` -> ``check_psd`` raises inside the
      script (the sign bug breaks positive-semidefiniteness, which is
      how the user originally caught it)
    - ``T_moon=0.0`` dropped from ``frozen`` -> below-horizon rows stop
      being zero: with luseepy's 250 K default a zero pair integral no
      longer gives a zero covariance, so the horizon check is exactly
      what notices
    - ``impedance_freq_mhz`` dropped -> 7.111e-05
    - ``e_freq`` sign flipped -> 9.817e-01
    - ``nofar`` computed with the Faraday phase left in -> 5.913e-03
    - ``pair[~mask] = 0.0`` deleted -> below-horizon rows nonzero

    Needs no artifact.
    """
    import step1_point_source as s1
    from lusee_faraday.conventions import lambda_squared

    resp, receiver = synthetic
    kern = rsp.FixedChannelKernel(resp, CENTER_MHZ)

    # 1e-10 is a roundoff bound, not a fitted one: chi = psi + fd*l2 is
    # ~2.5e4 rad, so cos/sin argument reduction alone costs ~4 digits.
    # Clean runs measure 1.4e-12 (2.9e-12 per channel); every injection
    # below is at least 7 orders of magnitude above it.
    TOL = 1e-10
    ntime, nfine, fd = 5, 4096, 250.0
    # Chunk 0 mixed (up, down), chunk 1 entirely below the horizon,
    # chunk 2 the ragged tail -- the three cases the loop distinguishes.
    monkeypatch.setattr(s1, "TIME_CHUNK", 2)
    theta = np.array([0.20, 2.00, 2.10, 2.20, 1.30])
    phi = np.array([0.30, 1.10, 2.50, 4.00, 6.28])
    psi = np.array([0.10, 0.40, 0.90, 1.60, 2.20])
    up = theta <= np.pi / 2

    out_path = tmp_path / "wf.npy"
    nofar = s1.make_waterfall(
        kern,
        resp,
        receiver,
        theta,
        phi,
        psi,
        fd,
        out_path,
        n_times=ntime,
        n_fine=nfine,
    )
    got = np.load(out_path)
    assert got.shape == (ntime, nfine, 16) and got.dtype == np.float64
    assert nofar.shape == (ntime, 16)

    ff = s1.fine_grid(nfine)
    l2 = lambda_squared(ff)
    K = kern.sample(np.where(up, theta, 0.0), phi)  # (10, 4, T)
    kw = dict(T_moon=0.0, T_ant=0.0, impedance_freq_mhz=CENTER_MHZ)
    one = np.array([CENTER_MHZ])

    want = np.zeros_like(got)
    want_nofar = np.zeros_like(nofar)
    for it in range(ntime):
        if not up[it]:
            continue
        chi = psi[it] + fd * l2  # (F,)
        stokes = np.stack(
            [
                np.ones_like(chi),
                np.cos(2 * chi),
                np.sin(2 * chi),
                np.zeros_like(chi),
            ]
        )  # (4, F)
        pair = np.einsum("ps,sf->fp", K[:, :, it], stokes)[None]
        want[it] = inst.channels(
            inst.covariance(pair, resp, receiver, ff, **kw)
        )[0][0]
        chi0 = psi[it]
        stokes0 = np.array([1.0, np.cos(2 * chi0), np.sin(2 * chi0), 0.0])
        pair0 = (K[:, :, it] @ stokes0)[None, None]
        want_nofar[it] = inst.channels(
            inst.covariance(pair0, resp, receiver, one, **kw)
        )[0][0, 0]

    assert (got[~up] == 0.0).all(), "below-horizon rows are not exactly zero"
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert (
        rel < TOL
    ), f"waterfall differs from the direct contraction: {rel:.3e}"
    rel0 = np.abs(nofar - want_nofar).max() / np.abs(want_nofar).max()
    assert rel0 < TOL, f"no-Faraday reference differs: {rel0:.3e}"


def test_channelization_matches_the_legacy_integrator():
    """``channelization.integrate`` reproduces the legacy binning."""
    rng = np.random.default_rng(2)
    # Both binners demand fine samples exactly on the +-50 kHz parent
    # edges, so the step has to divide 50 kHz.
    fine = CENTER_MHZ + (np.arange(400) - 200) * 400.0 * 1e-6
    waterfall = rng.normal(size=(3, 400, 16))
    centers = np.array([CENTER_MHZ])

    ported = chan.integrate(waterfall, fine, centers)
    legacy = fp.integrate_spectrometer(waterfall, fine, centers)
    for key in ("parent", "zoom", "ideal_zoom"):
        np.testing.assert_allclose(
            ported[key], legacy[key], rtol=1e-13, atol=0.0
        )


# ---------------------------------------------------------------------
# Published step-1 numbers, checked against the regenerated artifacts.
# These read generated_data/, which is gitignored, so they skip when the
# analyses have not been run -- the pattern Task 15 used.
# ---------------------------------------------------------------------

GEN = REPO / "generated_data"
CACHE = GEN / "cache"


def generated(*names):
    """Load the named generated_data files, or skip."""
    out = []
    for name in names:
        path = GEN / name
        if not path.exists():
            pytest.skip(f"{name} absent; re-run the step-1 scripts")
        out.append(path)
    return out


def zenith_weights(mode="ortho", center_mhz=CENTER_MHZ):
    cache = CACHE / f"zenith_weights_{center_mhz:g}.npz"
    if not cache.exists():
        pytest.skip("zenith weights not cached; run scripts/zenith_weights.py")
    d = np.load(cache)
    return d[f"x_{mode}"], d[f"y_{mode}"]


def transit_polarization(stokes):
    return np.hypot(stokes[..., 1], stokes[..., 2]) / stokes[..., 0]


@pytest.mark.slow
def test_unpolarized_transit_leakage_matches_the_paper():
    """0.134 raw, 0.096 with gains, 7e-4 with ortho, at transit.

    All three published leakage numbers are the same quantity -- the
    transiting unpolarized source's sqrt(Q^2+U^2)/I at culmination --
    read through the three polarimeters, so all three belong here.
    """
    from lusee_faraday import polarimeter as pol

    (path,) = generated("step1_ionly_source.npz")
    d = np.load(path)
    it = int(np.argmin(d["theta"]))
    channels = d["products"][it]

    raw = BASELINES["unpolarized_transit_leakage_raw"]
    p_raw = transit_polarization(pol.pseudo_stokes_from_channels(channels))
    assert p_raw == pytest.approx(raw["value"], rel=raw["rtol"])

    for mode, key in (
        ("gains", "unpolarized_transit_leakage_gains"),
        ("ortho", "unpolarized_transit_leakage_ortho"),
    ):
        base = BASELINES[key]
        x_vec, y_vec = zenith_weights(mode)
        p = transit_polarization(
            pol.pseudo_stokes_from_channels(channels, x_vec, y_vec)
        )
        assert p == pytest.approx(
            base["value"], rel=base["rtol"]
        ), f"{key}: measured {p:.5f}"


@pytest.mark.slow
def test_zoom_bin_amplitude_recovery_matches_the_paper():
    """Zoom bins keep 0.79 (real) and 0.86 (ideal) of the Q amplitude.

    report.tex fig:transitzoom quotes those fractions over the +-3 kHz
    window of the calibrated transit spectrum; the estimator here is the
    RMS of the binned Q over that window divided by the RMS of the fine
    spectrum, which is what "tracks the fine spectrum with 0.79 of the
    amplitude" measures.
    """
    from lusee_faraday import polarimeter as pol

    meta_path, binned_path, wf_path = generated(
        "step1_meta.npz",
        "step1_binned.npz",
        "step1_fine_waterfall.npy",
    )
    meta = np.load(meta_path)
    binned = np.load(binned_path)
    waterfall = np.load(wf_path, mmap_mode="r")
    x_vec, y_vec = zenith_weights()
    it = int(np.argmin(meta["theta"]))

    def q_of(channels):
        C = inst.unpack_channels(channels)
        return pol.pseudo_stokes(C, x_vec, y_vec)[..., 1]

    _, order = chan.zoom_frequency_grid(binned["parent_centers_mhz"])
    zoom_off_khz = np.array(
        [
            (binned["parent_centers_mhz"][p] - CENTER_MHZ) * 1e3
            + binned["zoom_offsets_hz"][k] * 1e-3
            for p, k in order
        ]
    )
    fine_off_khz = (meta["fine_freqs_mhz"] - CENTER_MHZ) * 1e3
    zwin = np.abs(zoom_off_khz) <= 3.0
    win = np.abs(fine_off_khz) <= 3.0

    q_fine = q_of(np.asarray(waterfall[it]))[win]
    for key, name in (
        ("zoom", "zoom_recovery_real"),
        ("ideal_zoom", "zoom_recovery_ideal"),
    ):
        binned_q = q_of(binned[key][it])
        sorted_q = np.array([binned_q[p, k] for p, k in order])[zwin]
        recovery = sorted_q.std() / q_fine.std()
        base = BASELINES[name]
        assert recovery == pytest.approx(
            base["value"], rel=base["rtol"]
        ), f"{name}: measured {recovery:.4f}"


def test_q_oscillation_period_matches_the_published_value():
    """1.89 kHz at 30 MHz for phi_FD = 250 rad/m^2 (PROGRESS.md step 1).

    The observed Q goes as cos(2 phi lambda^2), so one period is the
    frequency step over which ``lambda_squared`` moves by pi / phi.
    """
    from scipy.optimize import brentq

    from lusee_faraday.conventions import lambda_squared

    phi_fd = BASELINES["point_source_phi_fd"]

    def phase_turn(delta_mhz):
        lam2 = lambda_squared([CENTER_MHZ, CENTER_MHZ + delta_mhz])
        return 2.0 * phi_fd * (lam2[0] - lam2[1]) - 2.0 * np.pi

    period_khz = brentq(phase_turn, 1e-6, 1.0) * 1e3
    base = BASELINES["q_oscillation_period_khz"]
    assert period_khz == pytest.approx(base["value"], rel=base["rtol"])


@pytest.mark.slow
def test_transit_spectrum_oscillates_at_that_period():
    """The regenerated waterfall must actually carry the 1.89 kHz ripple.

    The analytic check above only pins ``lambda_squared``; this one pins
    the simulated spectrum, so a Faraday phase applied with the wrong
    sign convention, the wrong depth or the wrong wavelength would show
    up here.
    """
    from lusee_faraday import polarimeter as pol

    meta_path, wf_path = generated(
        "step1_meta.npz", "step1_fine_waterfall.npy"
    )
    meta = np.load(meta_path)
    waterfall = np.load(wf_path, mmap_mode="r")
    x_vec, y_vec = zenith_weights()
    it = int(np.argmin(meta["theta"]))

    fine = meta["fine_freqs_mhz"]
    span_khz = fine.size * (fine[1] - fine[0]) * 1e3
    q = pol.pseudo_stokes_from_channels(
        np.asarray(waterfall[it]), x_vec, y_vec
    )[:, 1]
    spectrum = np.abs(np.fft.rfft(q - q.mean()))
    k = int(np.argmax(spectrum[1:])) + 1
    lo, mid, hi = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    k = k + 0.5 * (lo - hi) / (lo - 2 * mid + hi)  # parabolic refinement
    period_khz = span_khz / k
    base = BASELINES["q_oscillation_period_khz"]
    assert period_khz == pytest.approx(base["value"], rel=0.03)


@pytest.mark.slow
def test_transit_covariance_stays_rank_one():
    """A single fully polarized source: sqrt(Q^2+U^2+V^2)/I in (0.98, 1].

    The upper bound is physics (any covariance is PSD); the lower bound
    says the source really is the rank-1 one the analysis assumes, which
    is what fails if the pair integrals lose their polarized part.
    """
    from lusee_faraday import polarimeter as pol

    meta_path, wf_path = generated(
        "step1_meta.npz", "step1_fine_waterfall.npy"
    )
    meta = np.load(meta_path)
    waterfall = np.load(wf_path, mmap_mode="r")
    it = int(np.argmin(meta["theta"]))

    stokes = pol.pseudo_stokes_from_channels(np.asarray(waterfall[it]))
    p = np.sqrt((stokes[:, 1:] ** 2).sum(-1)) / stokes[:, 0]
    assert p.max() <= 1 + 1e-9
    assert p.min() > 0.98
