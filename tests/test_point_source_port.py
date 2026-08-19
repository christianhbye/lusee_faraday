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
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import channelization as chan  # noqa: E402
from lusee_faraday import fourport as fp  # noqa: E402
from lusee_faraday import instrument as inst  # noqa: E402
from lusee_faraday import response as rsp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
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
