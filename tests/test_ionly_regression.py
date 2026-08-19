"""The I-only leakage reference, ported onto the harmonic arm.

``scripts/step_ionly.py`` reruns the real sky with Q = U = 0 -- the
perfect-depolarization limit -- so every pseudo-polarization it reports
is instrumental leakage of unpolarized emission.  ``report.tex``
sec:ionly claims the 30 MHz parent-bin ``(Q, U)/I = (0.146, -0.032)`` of
the full run is reproduced by that reference to within ``2e-4``, which
is what makes the sky model's 1.1e-4 CMB monopole load-bearing rather
than a rounding detail.

Three of these tests need nothing but a synthetic response.  The two
at the bottom read ``generated_data/real30_ionly*.npz`` and skip when
those are absent, which is how a fresh clone sees them; like the step-1
published-number tests they pin the *artifacts*, so they only turn over
when the analysis is re-run.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine, instrument as inst  # noqa: E402
from lusee_faraday import polarimeter as pol  # noqa: E402
from lusee_faraday import response as rsp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

RESPONSE = REPO / "data" / "BGL_v16" / "lusee_bgl_v16_response_v3.fits"
GEN_DIR = REPO / "generated_data"
IONLY_30 = GEN_DIR / "real30_ionly.npz"
IONLY_30_LEGACY = GEN_DIR / "real30_ionly_legacy.npz"
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)

CENTER_MHZ = 30.0
NSIDE = 16
LMAX = 8
AUTO_CHANNELS = [0, 4, 7, 9]  # positions of (a, a) in PORT_PAIRS

needs_artifact = pytest.mark.skipif(
    not RESPONSE.exists(), reason="BGL_v16 response artifact not present"
)


@pytest.fixture(scope="module")
def synthetic():
    """A cheap four-port response whose native channels bracket 30 MHz."""
    lusee = pytest.importorskip("lusee")
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = lusee.synthetic_four_port_response(
        freq_mhz=(CENTER_MHZ, 60.0), angular_step_deg=5.0
    )
    return resp, JFETReceiver()


@pytest.fixture(scope="module")
def haslam_like():
    """A random nside=16 map with Haslam's amplitude at 408 MHz.

    The amplitude matters: the CMB is 1.1e-4 of the real 30 MHz sky, and
    a map with the wrong scale would make that fraction unrepresentative.
    """
    import healpy as hp

    rng = np.random.default_rng(17)
    return 20.0 + 20.0 * rng.random(hp.nside2npix(NSIDE))


def test_two_component_sky_reproduces_sky_at_freq(haslam_like):
    """The CMB is not synchrotron, so it needs its own component.

    ``common.load_sky_maps`` returns ``I408`` with ``T_CMB`` already
    subtracted and ``common.sky_at_freq`` adds it back *unscaled*, since
    the CMB does not follow ``beta = -2.55``.  One ``FaradaySky.i_only``
    component carries exactly one spectral index, so it cannot hold
    both.  The monopole is where the entire difference lives -- a
    uniform map has no other multipole -- and it is the mode the
    published ``(Q, U)/I`` ratios are most sensitive to.
    """
    import common as cm
    from step_ionly import ionly_sky

    from lusee_faraday import config as cfg
    from lusee_faraday.sky import FaradaySky

    npix = haslam_like.size
    maps = {
        "I408": haslam_like,
        "Q23": np.zeros(npix),
        "U23": np.zeros(npix),
    }
    freqs = np.array([CENTER_MHZ])

    sky = ionly_sky(haslam_like, LMAX)
    assert sky.n_components == 2
    combined = np.einsum(
        "kclm,kc->clm", sky.component_alms, sky.coeffs(freqs)[:, 0]
    )
    got = combined[0, 0, LMAX]  # a_00 of Stokes I; m = 0 sits at LMAX

    # The same map pre-scaled the way the legacy pixel arm scales it.
    reference = FaradaySky.i_only(cm.sky_at_freq(maps, CENTER_MHZ)[0], LMAX)
    want = (
        reference.component_alms[0, 0, 0, LMAX]
        * reference.coeffs(freqs)[0, 0, 0]
    )
    assert abs(got - want) <= 1e-12 * abs(want)

    # ...and the single-component sky the plan asked for does NOT agree,
    # which is what makes the assertion above non-trivial.  The gap is
    # the CMB monopole: ~1.1e-4 of a 30 MHz sky, against a published
    # agreement of 2e-4.
    single = FaradaySky.i_only(
        haslam_like, LMAX, beta_i=cfg.BETA_I, ref_freq_i=cfg.FREQ_REF_I
    )
    naive = (
        single.component_alms[0, 0, 0, LMAX] * single.coeffs(freqs)[0, 0, 0]
    )
    assert abs(naive - want) > 1e-5 * abs(want)


def _ionly_products(resp, receiver, i408, phi_fd, freqs, ntime=3):
    """Packed products of an I-only sky over a frequency grid.

    The sky frame is "topo" so no lunarsky rotation is involved: the
    quantity under test is the Faraday phase, not the pointing.
    """
    from step_ionly import ionly_sky, pack_from_pairs

    from lusee_faraday.sky import FaradaySky

    built = ionly_sky(i408, LMAX)
    sky = FaradaySky(
        built.component_alms,
        np.full(built.n_components, float(phi_fd)),
        built.beta,
        built.ref_freq_mhz,
    )
    beam = rsp.four_port_pair_alms(resp, CENTER_MHZ, LMAX)
    W = engine.contract(
        beam, sky.component_alms, range(ntime), None, LMAX, sky_frame="topo"
    )
    pair = engine.expand(W, sky.coeffs(freqs))
    return pack_from_pairs(pair, resp, receiver, freqs, CENTER_MHZ)


def test_ionly_products_carry_no_faraday_structure(synthetic, haslam_like):
    """Renamed from the plan's ``..._is_frequency_flat``: see below.

    "Perfect depolarization" means the Faraday phase cannot reach the
    products *at all*.  It enters only through the ``P_MINUS``/
    ``P_PLUS`` dual blocks, as ``exp(-+2i phi_FD lambda^2)``, so if
    ``i_only`` left either block carrying power the products would move
    when the Faraday depth changes.  Setting the depth to 250 rad/m^2 --
    the paper's point-source value, ~4 rad of phase across this band --
    and re-evaluating on the genuine 16384-point fine grid is what makes
    that a measurement rather than an inspection.

    The plan asked instead for "the packed products are constant across
    those channels to round-off".  They are not, and cannot be: the
    synchrotron component carries ``beta = -2.55``, which puts a smooth
    ramp of 1.6997e-2 across the +-0.1 MHz band.  The script freezes the
    sky at the band center exactly as the legacy arm does
    (``sky_at_freq`` evaluates one frequency), so that ramp never
    reaches an artifact -- but an assertion of flatness would simply be
    false.

    The second half pins that ramp instead.  What it guards is the
    spectral index reaching the sky at all: pre-scaling the map to the
    band center and dropping ``beta_i`` -- which is what happens if the
    ``sky_at_freq`` habit survives the port -- takes the ramp to
    exactly 0 and turns the lower bound red.  It does *not* catch
    double-scaling, because ``sky_at_freq``'s factor is a constant:
    that is an amplitude error and
    ``test_two_component_sky_reproduces_sky_at_freq`` is what sees it.
    """
    from lusee_faraday import config as cfg

    resp, receiver = synthetic
    fine = cfg.fine_freqs(CENTER_MHZ)
    unrotated = _ionly_products(resp, receiver, haslam_like, 0.0, fine)
    rotated = _ionly_products(resp, receiver, haslam_like, 250.0, fine)

    scale = np.abs(unrotated).max()
    worst = np.abs(rotated - unrotated).max() / scale
    assert worst < 1e-13, f"a Faraday phase reached the products ({worst:.2e})"

    auto = unrotated[..., 0]  # the NN auto, positive definite
    ramp = np.ptp(auto, axis=-1) / np.abs(np.median(auto, axis=-1))
    # measured 1.6997e-2; the bounds bracket it loosely enough to
    # survive a beam or map update and tightly enough that a changed
    # or missing spectral index fails in either direction.
    assert np.all(ramp > 1.5e-2), f"spectral index missing ({ramp.max():.2e})"
    assert np.all(
        ramp < 2.0e-2
    ), f"spectral index too steep ({ramp.max():.2e})"


def test_analyze_refuses_to_fall_back_to_the_harmonic_file(
    tmp_path, monkeypatch
):
    """``--analyze`` must never compare across engines.

    It reads ``real{C}_binned.npz``, which ``scripts/step2_real_sky.py``
    produces on the pixel arm and which stays there by decision.  The
    effect being measured is 2e-4 and the harmonic-vs-pixel engine
    difference is ~1e-2, so silently substituting the harmonic I-only
    file would swamp the signal a hundred times over.

    The setup is the trap itself: a directory holding the *harmonic*
    artifact and no legacy one.  An implementation that reached for
    ``out_path(center)`` would find that file and sail past the guard,
    dying later and differently on ``real{C}_binned.npz``; only one that
    insists on the legacy path raises here.  (An earlier version of this
    test used a band with no artifacts at all, and a fallback
    implementation passed it -- both paths were missing, so the same
    error came out either way.)
    """
    import step_ionly

    monkeypatch.setattr(step_ionly, "GEN_DIR", tmp_path)
    np.savez(step_ionly.out_path(CENTER_MHZ), products=np.zeros((4, 16)))
    assert not step_ionly.out_path(CENTER_MHZ, "legacy").exists()
    with pytest.raises(FileNotFoundError, match="engine legacy"):
        step_ionly.analyze(CENTER_MHZ)


@pytest.mark.slow
@needs_artifact
def test_analyze_runs_end_to_end(tmp_path, monkeypatch):
    """A smoke test of the one ported function no artifact can reach.

    ``real{C}_binned.npz`` is a step-2 artifact and none exists in this
    repository, so ``--analyze`` has never been executed since the port.
    Feeding it synthetic products through a redirected ``GEN_DIR`` /
    ``FIG_DIR`` is the only way to find out that the
    ``polarimeter_from_channels`` -> ``pseudo_stokes_from_channels``
    rename and the npz keys it reads still line up -- Task 16's review
    recorded "no test exercises either ported script" as the standing
    gap, and this closes it for ``analyze``.
    """
    import step_ionly

    from zenith_weights import get_weights

    get_weights(CENTER_MHZ)  # warm the cache before GEN_DIR moves
    ntime = 8
    rng = np.random.default_rng(11)
    A = rng.normal(size=(ntime, 4, 4)) + 1j * rng.normal(size=(ntime, 4, 4))
    C = A @ np.conj(np.swapaxes(A, -1, -2))  # Hermitian, positive definite
    products, _ = inst.channels(C)

    monkeypatch.setattr(step_ionly, "GEN_DIR", tmp_path)
    monkeypatch.setattr(step_ionly, "FIG_DIR", tmp_path)
    monkeypatch.setattr(step_ionly, "N_TIMES", ntime)
    np.savez(
        step_ionly.out_path(CENTER_MHZ, "legacy"),
        products=products,
        center_mhz=CENTER_MHZ,
    )
    np.savez(
        tmp_path / f"real{CENTER_MHZ:g}_binned.npz",
        parent=np.repeat(products[:, None] * 1.001, 3, axis=1),
        zoom=np.repeat(
            np.repeat(products[:, None, None] * 1.002, 3, axis=1), 64, axis=2
        ),
    )

    step_ionly.analyze(CENTER_MHZ)
    for ext in ("pdf", "png"):
        assert (tmp_path / f"real30_ionly_frac.{ext}").exists()


@pytest.mark.slow
@needs_artifact
def test_moon_term_is_off():
    """``T_moon`` defaults to 250 K; this run needs it at zero.

    ``fourport.assemble_covariance``, the assembler the ported script
    replaces, has no Moon term at all, and the stored
    ``real{C}_binned.npz`` that ``--analyze`` compares against was made
    with it.  Taking luseepy's default instead moves the answer by four
    orders of magnitude.

    Unlike the Task-16 version of this test, which pinned only
    ``instrument.covariance``'s signature default, this one routes
    through ``step_ionly.pack_from_pairs`` -- the single place both arms
    of the script assemble a covariance -- so dropping ``T_moon=0.0``
    from the script itself is what turns it red.
    """
    from step_ionly import pack_from_pairs

    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    resp = rsp.load_response(RESPONSE)
    receiver = JFETReceiver()
    rng = np.random.default_rng(5)
    shape = (4, 3, 10)
    pair = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    pair[..., AUTO_CHANNELS] = pair[..., AUTO_CHANNELS].real
    freqs = np.linspace(CENTER_MHZ - 0.1, CENTER_MHZ + 0.1, 3)

    ours = pack_from_pairs(pair, resp, receiver, freqs, CENTER_MHZ)
    with_moon, _ = inst.channels(
        inst.covariance(
            pair, resp, receiver, freqs, impedance_freq_mhz=CENTER_MHZ
        )
    )
    rel = np.abs(with_moon - ours).max() / np.abs(ours).max()
    assert rel > 1e3, f"the Moon term is not showing up ({rel:.3e})"


@pytest.mark.slow
@needs_artifact
@pytest.mark.skipif(
    not IONLY_30.exists(), reason="generated_data/real30_ionly.npz absent"
)
def test_parent_bin_leakage_matches_the_published_stokes_ratio():
    """report.tex sec:ionly: the 30 MHz parent bin is ``(0.146, -0.032)``.

    That pair is a **snapshot**, not a time average: ``step2_plots.py``
    picks ``it_snap = argmax(pseudo-I of meta["nofaraday"])`` and
    ``report.tex`` quotes the parent bin there, through the ortho
    zenith-calibrated polarimeter (PROGRESS.md, "Steps 2+ switched to
    the calibrated polarimeter").  Reproducing that index exactly needs
    ``real30_meta.npz``, which only ``scripts/step2_real_sky.py``
    writes -- 80 min and a 2 GB waterfall -- and which this task does
    not run.  ``argmax`` over the I-only pseudo-I instead lands 9
    samples earlier, on an I curve that is flat to 1e-3 over +-25
    samples, and ``(q, u)`` moves by ~1e-3 per sample there, so the
    index choice is worth ~1e-2 in ``u``.

    What does not depend on the index is the *track*: the published
    point has to lie on it, near the sky maximum.  Both assertions
    below are that statement.

    Tolerance arithmetic (measured, Task 17 Step 4 -- re-measure before
    changing it, do not widen it):

        legacy (pixel-arm) closest approach to published   6.23e-4
        harmonic minus legacy, in quadrature on (q, u)     8.2e-5
        sum                                                7.05e-4
        tolerance below                                    1.0e-3

    The legacy term is itself inside the 7.07e-4 that the published
    pair's own three-decimal rounding allows (5e-4 in each of q and
    u), so it is consistent with an exact match, not with a bias.
    Measured on the harmonic artifact: 6.85e-4, at 0.99892 of peak I.

    The baseline file's ``atol`` of 2e-4 is deliberately *not* used
    here: in report.tex it bounds ``|I-only - full run|``, and the full
    run (``real30_binned.npz``) is a step-2 artifact this task does not
    produce.  Nothing in this repository can check that 2e-4 until
    ``scripts/step2_real_sky.py`` is re-run.
    """
    from zenith_weights import get_weights

    published = BASELINES["parent_stokes_over_i_30mhz"]
    x_vec, y_vec = get_weights(CENTER_MHZ)  # ortho, as the paper used
    products = np.load(IONLY_30)["products"]
    S = pol.pseudo_stokes_from_channels(products, x_vec, y_vec)
    q = S[:, 1] / S[:, 0]
    u = S[:, 2] / S[:, 0]

    distance = np.hypot(q - published["q"], u - published["u"])
    it = int(np.argmin(distance))
    assert distance[it] < 1.0e-3, (
        f"closest approach {distance[it]:.3e} at t index {it}: "
        f"(q, u) = ({q[it]:+.5f}, {u[it]:+.5f})"
    )
    # ...and it must happen where the published snapshot was taken --
    # the time of maximum sky signal -- not somewhere unrelated on the
    # track.  Measured: 0.99892 of the peak.
    assert S[it, 0] / S[:, 0].max() > 0.99


@pytest.mark.slow
@pytest.mark.skipif(
    not (IONLY_30.exists() and IONLY_30_LEGACY.exists()),
    reason="both 30 MHz I-only artifacts absent",
)
def test_the_two_engines_agree_on_the_real_sky():
    """The harmonic contraction against the pixel quadrature, same sky.

    This is the cross-arm number the Task 7 ruling designated the
    published-number backstop for, pinned rather than left in a report.
    Task 7 measured the two arms disagreeing by 2.678e-2 (nside=64,
    lmax=30) to 3.767e-2 (nside=32, lmax=48) on a *random band-limited
    IQUV* sky, and recorded [1e-2, 8e-2] as the empirical band.  On the
    real, smooth, unpolarized sky at native nside=512 the same two arms
    agree to 3.05e-4 -- two orders of magnitude inside that band.

    Tolerance 1.0e-3, about 3x the measurement: large enough that the
    two quadratures may legitimately drift with a beam or map update,
    small enough that anything resembling the Task-7 residual fails.
    The lower bound is there because these are genuinely different
    quadratures: an exact match would mean one arm is not being
    computed.
    """
    harmonic = np.load(IONLY_30)["products"]
    legacy = np.load(IONLY_30_LEGACY)["products"]
    rel = np.abs(harmonic - legacy).max() / np.abs(legacy).max()
    assert rel < 1.0e-3, f"engines disagree by {rel:.3e}"
    assert rel > 1e-8, f"the two arms are suspiciously identical ({rel:.3e})"
