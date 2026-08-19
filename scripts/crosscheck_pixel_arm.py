"""Harmonic four-port path vs the pixel-space engine in _legacy_pixel.py.

This script CHARACTERIZES the disagreement between the harmonic
contraction (``engine.contract``) and the legacy pixel-space engine
(``_legacy_pixel.py``) on the real BGL_v16 response.  It is not a
correctness gate -- the gate is ``tests/test_engine_gate.py``, which
shows the harmonic contraction reproduces luseepy's own convolution
to round-off (6.8e-16) on a synthetic response and a rotation-
sensitive sky.

WHY THE DISAGREEMENT EXISTS HERE IS AN OPEN QUESTION, not a settled
fact.  Two candidate mechanisms were considered; neither is proven.

Leading hypothesis (better supported): the pixel arm's beam sampling
(``FixedFreqKernel.sample`` / ``sample_periodic_maps``) always
bilinearly interpolates off the response's fixed, native 1-degree
theta/phi grid, regardless of the sky's HEALPix resolution.  If that
interpolation carries a systematic (non-random) error near sharp
features of the real beam, integrating over more/finer sky pixels
would not average it away, since the error source itself does not
shrink with sky resolution.

Alternative hypothesis, considered and judged less likely on
reflection: a step-discontinuity Gibbs-truncation story. What is
TRUE and verified: our beam alms (``response.four_port_pair_alms``)
and luseepy's own (``FullStokesSimulatorBase.prepare_pair_alms``)
both route through ``lusee.InstrumentResponse.pair_stokes_alms_native``,
which builds a ``croissant.PairStokesBeam`` with an explicit horizon
mask (``theta <= 90 deg``) and calls its ``compute_alm`` -- identical
code on both sides, which is exactly why the gate lands at round-off.
The BGL_v16 response itself is stored only over the upper hemisphere
(theta in [0, 90] deg, confirmed against the FITS ``theta`` HDU) and
is zero-padded to the full sphere before that transform, so the
masked beam fed into ``compute_alm`` has a genuine step discontinuity
at the horizon. Those facts are real. What does NOT follow from them:
that this causes the disagreement measured here. The sky in this
script is EXACTLY band-limited to LMAX by construction
(``hp.synalm(..., lmax=LMAX)``). By orthogonality of spherical
harmonics, the true continuous integral of the beam against that sky
depends only on the beam's l<=LMAX projection -- whatever power the
horizon step puts above LMAX cannot enter that integral except
through second-order effects of the harmonic arm's own quadrature.
So the horizon truncation should not, on its own, be significantly
lossy in this comparison, even though the structural facts above are
correct.

The nside sweep run against this script (2.103e-2 / 2.105e-2 /
2.106e-2 at pixel-arm nside 32/64/128, sky held fixed in harmonic
space, lmax=30 -- flat to about 0.1% relative across a 4x range in
nside) does NOT discriminate between these two hypotheses.
``GalacticGrid(nside)`` controls the SKY-side quadrature; the beam's
own source grid (the response's native 1-degree theta/phi grid that
``FixedFreqKernel.sample`` interpolates off) never changes with the
sky's nside. Flatness under that sweep rules out sky-side pixel
quadrature error, but both hypotheses above are equally consistent
with "it's not sky-side quadrature" -- the sweep does not favor one
over the other.

Weak lmax sensitivity (6.49e-2 at lmax=30 vs 6.30e-2 at lmax=48, in
one differently-seeded run pair, measured before the sky-seeding fix
below) is likewise inconclusive on its own; treat it as suggestive at
most, not as evidence for either hypothesis. The disagreement was
also observed worse on cross-polarization pairs than on autos in
that same early, unseeded run -- interesting, but never independently
re-checked against the final seeded configurations below, so treat
that too as suggestive only.

Because our harmonic path uses the identical masked-transform code
luseepy's own production simulator uses (confirmed above), this
script's residual reflects a pre-existing difference between
luseepy's harmonic engine and the legacy pixel engine, not a
regression from this refactor -- but which part of the legacy pixel
arm is responsible remains an open question this script does not
resolve.

The recorded [1e-2, 8e-2] band below was measured on THIS SCRIPT'S
polarized test sky and applies to no other: with Q = U = 0 the same
comparison gives 2.687e-05 and correctly prints "OUTSIDE the band".
See the comment on EXPECTED_LOW/EXPECTED_HIGH.

Run:
    ulimit -v 16000000
    uv run python scripts/crosscheck_pixel_arm.py 2>&1 | tee \
        /home/christian/Documents/research/lusee/lusee_faraday/generated_data/crosscheck_pixel_arm.log
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from lusee_faraday import config as cfg  # noqa: E402
from lusee_faraday import engine, response as rsp  # noqa: E402
from lusee_faraday import _legacy_pixel as fp  # noqa: E402

RESPONSE_PATH = os.environ.get(
    "LUSEE_RESPONSE",
    "data/BGL_v16/lusee_bgl_v16_response_v3.fits",
)
FREQ_MHZ = 30.0
# Two configurations, both reproducible (seed 7, fixed below).
# Measured 2026-08-18:
#   nside=64, lmax=30 (default here):                    2.678e-02
#   nside=32, lmax=48 (validate_engine.py's own config):  3.767e-02
# validate_engine.py's diffuse-sky check records 1.1e-2 for a related
# but not identical comparison.  Why this script's numbers run higher
# is an OPEN QUESTION -- see the module docstring above; it is not
# settled to be the horizon-mask/Gibbs mechanism.  Override with
# LUSEE_CROSSCHECK_NSIDE / LUSEE_CROSSCHECK_LMAX to reproduce either
# configuration above.
NSIDE = int(os.environ.get("LUSEE_CROSSCHECK_NSIDE", "64"))
LMAX = int(os.environ.get("LUSEE_CROSSCHECK_LMAX", "30"))
N_TIMES = 4

# Empirical expectation band, NOT a correctness bound.  This is two
# single-seed point measurements (2.678e-2, 3.767e-2 above) plus one
# precedent -- not a measured distribution across seeds, and it is a
# coarse gross-regression detector, not a claim that either arm is
# correct (that's tests/test_engine_gate.py's job).  The strongest
# evidence for EXPECTED_HIGH is that precedent:
# scripts/validate_engine.py:163 already asserts
# `max(errs) < 8e-2  # nside=32 pixelization limits agreement` for
# materially the same harmonic-vs-pixel-arm comparison on the same
# real BGL_v16 artifact, and that bound is already-accepted precedent
# independent of anything measured in this script.  Recorded
# 2026-08-18.
#
# THE BAND IS SPECIFIC TO THIS SCRIPT'S OWN POLARIZED TEST SKY and does
# not transfer to other skies.  Task 17 re-ran exactly this crosscheck
# with the sky's Q and U set to zero and nothing else changed:
#   polarized (band_limited_sky as written):  2.678e-02  -- in band
#   identical run with Q = U = 0:             2.687e-05  -- "OUTSIDE"
# An unpolarized sky samples only the P^I pair-Stokes kernel; a
# polarized one additionally samples P^Q and P^U, and the disagreement
# lives almost entirely there.  So "OUTSIDE the band" printed for an
# unpolarized sky is the correct answer, not a regression.  See
# docs/measurement-model.md section 8.
EXPECTED_LOW = 1e-2
EXPECTED_HIGH = 8e-2


def band_limited_sky(rng, nside, lmax):
    """A smooth IQUV sky the beam's band-limit can actually represent.

    ``hp.synalm`` takes no seed of its own and draws from numpy's
    legacy global RNG state, so each channel reseeds that state from
    ``rng`` -- the only source of randomness threaded through this
    script -- to make the sky (and hence the reported disagreement)
    reproducible run to run.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.empty((4, npix))
    for i in range(4):
        np.random.seed(rng.integers(0, 2**32 - 1))
        alm = hp.synalm(
            np.exp(-np.arange(3 * nside) / 6.0), lmax=lmax, new=True
        )
        maps[i] = hp.alm2map(alm, nside, lmax=lmax)
    maps[0] = np.abs(maps[0]) + 10.0  # Stokes I positive
    maps[3] = 0.0  # no circular polarization
    return maps[None]


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import croissant as cro
    from lusee.ReceiverImpedance import JFETReceiver

    rng = np.random.default_rng(7)
    resp = rsp.load_response(RESPONSE_PATH)
    receiver = JFETReceiver()

    data = band_limited_sky(rng, NSIDE, LMAX)
    # croissant wants IAU; the maps above are treated as COSMO.
    from lusee_faraday.conventions import cosmo_to_iau_qu

    data[0, 1], data[0, 2] = cosmo_to_iau_qu(data[0, 1], data[0, 2])
    sky = cro.PolarizedSky(
        data, np.array([FREQ_MHZ]), sampling="healpix", coord="galactic"
    )

    times = cfg.times()[:N_TIMES]
    beam = rsp.four_port_pair_alms(resp, FREQ_MHZ, LMAX)
    components = np.asarray(sky.compute_alm(lmax=LMAX))
    W = engine.contract(beam, components, times, cfg.moon_location(), LMAX)
    harmonic = W.sum(axis=1)[0]  # (ntime, npair)

    # Pixel arm: same maps, same response, no Faraday.
    kern = fp.FixedFreqKernel(resp, FREQ_MHZ, receiver)
    grid = fp.GalacticGrid(NSIDE)
    I_map = data[0, 0]
    Q_cos, U_cos = data[0, 1], -data[0, 2]  # back to COSMO for the pixel arm
    sim = fp.SkyWaterfallSim(
        kern, grid, I_map, Q_cos, U_cos, np.zeros_like(I_map)
    )
    lam2 = cfg.lam2(np.array([FREQ_MHZ]))
    pixel = np.empty_like(harmonic)
    for i, t in enumerate(times):
        R = fp.topo_rotation_matrix(t, cfg.moon_location())
        pixel[i] = sim.pair_integrals(R, lam2, faraday=False)[:, 0]

    scale = np.abs(pixel).max()
    worst = np.abs(harmonic - pixel).max() / scale
    print(f"worst relative disagreement: {worst:.3e}")
    band = f"[{EXPECTED_LOW:.1e}, {EXPECTED_HIGH:.1e}]"
    if EXPECTED_LOW <= worst <= EXPECTED_HIGH:
        print(f"within the recorded empirical band {band}")
    else:
        print(
            f"OUTSIDE the recorded empirical band {band} -- "
            "this is a change from what's been measured; "
            "investigate before trusting this comparison"
        )


if __name__ == "__main__":
    main()
