"""Per-pair pixel weights from the frozen-channel kernel (spec S4.3)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lunarsky")
healpy = pytest.importorskip("healpy")

from lusee_faraday import response as rsp
from lusee_faraday.config import moon_location, times
from lusee_faraday.conventions import topo_rotation_matrix


class _SyntheticKernel:
    """One pair; K_Q = cos(theta), K_U = i sin(theta) on the upper sky."""

    def sample(self, theta_rad, phi_rad):
        n = theta_rad.size
        K = np.zeros((1, 4, n), dtype=complex)
        K[0, 1] = np.cos(theta_rad)
        K[0, 2] = 1j * np.sin(theta_rad)
        return K


def test_pair_weight_maps_geometry_and_masking():
    """Validate geometry, masking, and basis-independent weight formula.

    The synthetic kernel has K_Q = cos(theta), K_U = i*sin(theta).
    Using weight = sqrt(0.5 * (|K_Q|^2 + |K_U|^2)), the trig identity
    cos^2 + sin^2 = 1 gives weight = sqrt(0.5) everywhere, independent
    of basis convention (COSMO vs IAU).
    """
    import healpy as hp

    loc = moon_location()
    t = times()[0]
    nside = 32
    w = rsp.pair_weight_maps(_SyntheticKernel(), t, loc, nside)
    assert w.shape == (1, hp.nside2npix(nside))
    assert np.all(w >= 0.0)
    # below-horizon pixels are exactly zero, above-horizon are not
    R = topo_rotation_matrix(t, loc)
    vec = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    z = (R @ vec)[2]
    assert np.all(w[0, z <= 0] == 0.0)
    assert np.all(w[0, z > 1e-3] > 0.0)
    # value check: every above-horizon pixel equals sqrt(0.5) by trig identity
    expected = np.sqrt(0.5)
    assert np.allclose(w[0, z > 0.0], expected, rtol=1e-12)


ARTIFACT = Path(
    os.environ.get(
        "LUSEE_RESPONSE", "data/BGL_v16/lusee_bgl_v16_response_v3.fits"
    )
)


@pytest.mark.slow
@pytest.mark.skipif(not ARTIFACT.exists(), reason="needs BGL_v16 artifact")
def test_pair_weight_maps_from_the_real_kernel():
    lusee = pytest.importorskip("lusee")  # noqa: F841

    resp = rsp.load_response(str(ARTIFACT))
    kernel = rsp.FixedChannelKernel(resp, 30.0)
    w = rsp.pair_weight_maps(kernel, times()[0], moon_location(), 64)
    assert w.shape == (10, 12 * 64**2)
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
    assert (w > 0).any(axis=1).all()  # every pair sees the sky
