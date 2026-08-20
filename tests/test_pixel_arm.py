"""
Unit tests for lusee_faraday.pixel_arm (data-free parts).

The heavy validation against the luseepy harmonic engines lives in
scripts/validate_engine.py (needs the 631 MB response artifact); here we
pin the pure-numpy machinery: product packing, the ideal polarimeter,
the kernel sampler, the synthetic Faraday synthesis via the NUFFT, and
the spectrometer-bin weight construction.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import pixel_arm as fp  # noqa: E402


def random_hermitian(rng, shape=()):
    A = rng.normal(size=shape + (4, 4)) + 1j * rng.normal(size=shape + (4, 4))
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    C = random_hermitian(rng, (3, 5))
    ch = fp.pack_products(C)
    assert ch.shape == (3, 5, 16)
    assert np.allclose(fp.unpack_products(ch), C)


def test_product_labels_match_covariance():
    lusee_cov = pytest.importorskip("lusee.Covariance")

    assert fp.PRODUCT_LABELS == lusee_cov.default_product_labels()
    assert len(fp.PRODUCT_LABELS) == 16


def test_polarimeter_pure_states():
    # A pure X = E - W signal: V_E = -V_W = 1, V_N = V_S = 0.
    vx = np.array([0.0, 1.0, 0.0, -1.0])
    C = np.einsum("a,b->ab", vx, vx).astype(complex)
    I, Q, U, V = fp.polarimeter(C)
    XX = 4.0  # |E - W|^2
    assert np.isclose(I, XX / 2) and np.isclose(Q, XX / 2)
    assert np.isclose(U, 0.0) and np.isclose(V, 0.0)
    # A pure U state: X and Y in phase.
    vy = np.array([1.0, 0.0, -1.0, 0.0])
    v = vx + vy
    C = np.einsum("a,b->ab", v, v).astype(complex)
    I, Q, U, V = fp.polarimeter(C)
    assert np.isclose(Q, 0.0) and np.isclose(U, I) and np.isclose(V, 0.0)


def test_assemble_covariance_hermitian():
    rng = np.random.default_rng(1)
    pair = rng.normal(size=(7, 10)) + 1j * rng.normal(size=(7, 10))
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    C = fp.assemble_covariance(pair, M)
    assert C.shape == (7, 4, 4)
    assert np.allclose(C, np.conj(np.swapaxes(C, -1, -2)))


def test_sample_periodic_maps_bilinear_exact():
    # A function linear in theta is reproduced exactly by the bilinear
    # sampler anywhere inside the grid.
    theta_deg = np.arange(0.0, 91.0, 1.0)
    phi_deg = np.arange(0.0, 361.0, 1.0)
    tt, pp = np.meshgrid(theta_deg, phi_deg, indexing="ij")
    vals = 2.0 * np.radians(tt) + 3.0
    rng = np.random.default_rng(2)
    th = rng.uniform(0.01, np.radians(89.9), size=50)
    ph = rng.uniform(0.0, 2 * np.pi, size=50)
    got = fp.sample_periodic_maps(vals, theta_deg, phi_deg, th, ph)
    assert np.allclose(got, 2.0 * th + 3.0, atol=1e-12)


def test_sample_periodic_maps_phi_wrap():
    theta_deg = np.arange(0.0, 91.0, 1.0)
    phi_deg = np.arange(0.0, 361.0, 1.0)
    tt, pp = np.meshgrid(theta_deg, phi_deg, indexing="ij")
    vals = np.cos(np.radians(pp))
    th = np.full(9, np.radians(45.0))
    ph = np.linspace(-0.004, 0.004, 9) % (2 * np.pi)
    got = fp.sample_periodic_maps(vals, theta_deg, phi_deg, th, ph)
    assert np.allclose(got, np.cos(ph), atol=1e-4)


def test_faraday_synthesis_matches_rotated_maps():
    pytest.importorskip("finufft")
    pytest.importorskip("healpy")

    nside = 16
    grid = fp.GalacticGrid(nside)
    rng = np.random.default_rng(3)
    npix = grid.npix
    I_map = 10.0 + rng.normal(size=npix) ** 2
    Q_map = rng.normal(size=npix)
    U_map = rng.normal(size=npix)
    RM = rng.normal(size=npix) * 30.0

    class ToyKernel:
        prefac = 1.0
        theta_deg = np.arange(0.0, 91.0, 1.0)
        phi_deg = np.arange(0.0, 361.0, 1.0)

        def sample(self, theta, phi):
            n = theta.size
            k = np.zeros((10, 4, n), dtype=complex)
            k[:, 0] = 1.0
            k[:, 1] = np.cos(phi) + 0.2j * np.sin(theta)
            k[:, 2] = np.sin(phi) - 0.1j
            return k

    sim = fp.SkyWaterfallSim(ToyKernel(), grid, I_map, Q_map, U_map, RM)
    lam2 = np.array([25.0])  # m^2, exaggerated for a strong rotation
    R = np.eye(3)
    got = sim.pair_integrals(R, lam2, faraday=True)[:, 0]
    chi = RM * lam2[0]
    Q_r = Q_map * np.cos(2 * chi) - U_map * np.sin(2 * chi)
    U_r = Q_map * np.sin(2 * chi) + U_map * np.cos(2 * chi)
    sim_rot = fp.SkyWaterfallSim(
        ToyKernel(), grid, I_map, Q_r, U_r, np.zeros(npix)
    )
    ref = sim_rot.pair_integrals(R, lam2, faraday=False)[:, 0]
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8


def test_zoom_bin_offsets_fft_order():
    off = fp.zoom_bin_offsets_hz()
    assert off[0] == 0.0
    assert off[1] == pytest.approx(fp.ZOOM_STEP_HZ)
    assert off[32] == pytest.approx(-32 * fp.ZOOM_STEP_HZ)
    assert off[63] == pytest.approx(-fp.ZOOM_STEP_HZ)


def test_ideal_zoom_weights_normalized():
    off = np.arange(-50000.0, 50000.0, fp.FINE_STEP_HZ)
    W = fp.ideal_zoom_weights(off)
    assert W.shape == (off.size, 64)
    assert np.allclose(W.sum(axis=0), 1.0)


def test_integrate_spectrometer_constant_waterfall():
    pytest.importorskip("lusee.SpectrometerResponse")
    # A frequency-independent waterfall must integrate to itself in
    # every parent / zoom / ideal-zoom bin.
    center = 30.0
    fine = center + (np.arange(16384) - 8192) * fp.FINE_STEP_HZ * 1e-6
    chans = np.arange(16, dtype=float)
    waterfall = np.broadcast_to(chans, (fine.size, 16))
    out = fp.integrate_spectrometer(
        waterfall, fine, [center - 0.025, center, center + 0.025]
    )
    assert out["parent"].shape == (3, 16)
    assert out["zoom"].shape == (3, 64, 16)
    assert out["ideal_zoom"].shape == (3, 64, 16)
    for key in ("parent", "zoom", "ideal_zoom"):
        assert np.allclose(
            out[key], np.broadcast_to(chans, out[key].shape)
        ), key


def test_orthonormalize_xy_nulls_leakage():
    # Any Hermitian PSD C0: the orthonormalized X/Y must give exactly
    # zero pseudo-Q/U/V for a source with covariance C0.
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    C0 = A @ A.conj().T
    xv, yv = fp.orthonormalize_xy(C0, fp.X_VEC, fp.Y_VEC)
    S = fp.polarimeter(C0, xv, yv)
    assert S[0] > 0
    assert np.allclose(S[1:] / S[0], 0.0, atol=1e-12)
    # and the vectors stay close to the input dipoles
    assert np.abs(xv - fp.X_VEC).max() < 1.0
