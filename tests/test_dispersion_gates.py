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
