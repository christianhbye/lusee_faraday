"""All-pair moments compared with independent real E/B harmonic ensembles."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import engine, instrument, polarimeter  # noqa: E402
from lusee_faraday.conventions import DUAL_BLOCKS  # noqa: E402
from lusee_faraday.spin_covariance import (  # noqa: E402
    AngularScreen,
    SpinTransform,
    eb_spectra,
    helicity_covariance,
    rotated_spin_spectra,
)
from lusee_faraday.visibility_covariance import (  # noqa: E402
    AngularReadoutProjector,
    evaluate_mode_moments,
    loaded_readout_operator,
    pair_mode_moments,
    real_readout_covariance,
    real_readout_kernels,
)


def harmonic_ensemble(lmax):
    """Finite ensemble with exact real E/B statistics and m reality."""
    count = (lmax + 1) ** 2 - 4
    rank = 2 * count
    samples = np.r_[np.eye(rank), -np.eye(rank)] * np.sqrt(rank)
    ee = np.zeros(lmax + 1)
    bb, eb = ee.copy(), ee.copy()
    e = np.zeros((2 * rank, lmax + 1, 2 * lmax + 1), complex)
    b = np.zeros_like(e)
    cursor = 0
    for ell in range(2, lmax + 1):
        ee[ell], bb[ell], eb[ell] = (
            1 / (ell + 1),
            0.4 / (ell + 1),
            0.1 / (ell + 1),
        )
        for m in range(ell + 1):
            er = samples[:, cursor]
            br = samples[:, count + cursor]
            cursor += 1
            if m:
                er = (er + 1j * samples[:, cursor]) / np.sqrt(2)
                br = (br + 1j * samples[:, count + cursor]) / np.sqrt(2)
                cursor += 1
            e[:, ell, lmax + m] = np.sqrt(ee[ell]) * er
            b[:, ell, lmax + m] = (
                eb[ell] / np.sqrt(ee[ell]) * er
                + np.sqrt(bb[ell] - eb[ell] ** 2 / ee[ell]) * br
            )
            if m:
                e[:, ell, lmax - m] = (-1.0) ** m * e[:, ell, lmax + m].conj()
                b[:, ell, lmax - m] = (-1.0) ** m * b[:, ell, lmax + m].conj()
    assert cursor == count
    sky = np.zeros((2 * rank, 4, lmax + 1, 2 * lmax + 1), complex)
    sky[:, DUAL_BLOCKS.index("P_MINUS")] = e - 1j * b
    sky[:, DUAL_BLOCKS.index("P_PLUS")] = e + 1j * b
    return sky, eb_spectra(ee, bb, eb)


def random_kernel(lmax, seed):
    rng = np.random.default_rng(seed)
    beam = rng.normal(size=(10, 4, lmax + 1, 2 * lmax + 1))
    beam = beam + 1j * rng.normal(size=beam.shape)
    for ell in range(lmax + 1):
        start, stop = lmax - ell, lmax + ell + 1
        beam[:, :, ell, :start] = 0
        beam[:, :, ell, stop:] = 0
    beam[:, 2:, :2] = 0
    return beam


def test_pair_covariance_and_relation_match_harmonic_ensemble():
    lmax = 4
    sky, spectra = harmonic_ensemble(lmax)
    beam = random_kernel(lmax, 10)
    m = np.arange(-lmax, lmax + 1)
    phases = np.exp(-1j * np.array([0.0, 0.47, 2 * np.pi])[:, None] * m)
    predicted = evaluate_mode_moments(
        pair_mode_moments(beam, helicity_covariance(**spectra)), phases
    )
    measured = engine.contract_blocks(beam, sky, phases).sum(axis=1)
    n = len(measured)
    cov = np.einsum("ntp,nuq->tpuq", measured, measured.conj()) / n
    relation = np.einsum("ntp,nuq->tpuq", measured, measured) / n
    np.testing.assert_allclose(predicted["covariance"], cov, atol=2e-12)
    np.testing.assert_allclose(predicted["relation"], relation, atol=2e-12)
    np.testing.assert_allclose(cov[0], cov[2], atol=2e-12)
    assert np.max(abs(relation)) > 1


def test_cross_frequency_rotation_and_different_pair_kernels():
    lmax = 3
    sky, base = harmonic_ensemble(lmax)
    beam, other = random_kernel(lmax, 11), random_kernel(lmax, 12)
    x, y, depth = 41.0, 28.0, 0.017
    sx, sy = sky.copy(), sky.copy()
    for block, sign in (("P_MINUS", -1), ("P_PLUS", 1)):
        sx[:, DUAL_BLOCKS.index(block)] *= np.exp(sign * 2j * depth * x)
        sy[:, DUAL_BLOCKS.index(block)] *= np.exp(sign * 2j * depth * y)
    spectra = dict(
        plus=base["plus"] * np.exp(2j * depth * (x - y)),
        minus=base["minus"] * np.exp(2j * depth * (x + y)),
    )
    m = np.arange(-lmax, lmax + 1)
    phases = np.exp(-1j * np.array([0.0, 0.3])[:, None] * m)
    other_phases = np.exp(-1j * np.array([0.1, 0.8, 1.2])[:, None] * m)
    predicted = evaluate_mode_moments(
        pair_mode_moments(beam, helicity_covariance(**spectra), other),
        phases,
        other_phases,
    )
    a = engine.contract_blocks(beam, sx, phases).sum(axis=1)
    b = engine.contract_blocks(other, sy, other_phases).sum(axis=1)
    np.testing.assert_allclose(
        predicted["covariance"],
        np.einsum("ntp,nuq->tpuq", a, b.conj()) / len(a),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        predicted["relation"],
        np.einsum("ntp,nuq->tpuq", a, b) / len(a),
        atol=2e-12,
    )


def test_real_readout_keeps_relation_and_batch_dimensions():
    rng = np.random.default_rng(15)
    z = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
    # Center samples explicitly; this is a finite covariance identity.
    z -= z.mean(axis=0)
    c, j = z.T @ z.conj() / len(z), z.T @ z / len(z)
    readout = rng.normal(size=(3, 10))
    values = np.c_[z.real, z.imag] @ readout.T
    expected = values.T @ values / len(z)
    result = real_readout_covariance(
        np.stack([c, 2 * c]), np.stack([j, 2 * j]), readout
    )
    np.testing.assert_allclose(result[0], expected, atol=2e-13)
    np.testing.assert_allclose(result[1], 2 * expected, atol=2e-13)


def test_real_kernels_match_pair_readout_on_independent_harmonic_samples():
    lmax = 4
    sky, _ = harmonic_ensemble(lmax)
    beam = random_kernel(lmax, 27)
    rng = np.random.default_rng(28)
    operator = rng.normal(size=(3, 20))
    phases = np.exp(
        -1j * np.array([0.0, 0.63])[:, None] * np.arange(-lmax, lmax + 1)
    )
    pairs = engine.contract_blocks(beam, sky, phases).sum(axis=1)
    expected = np.concatenate([pairs.real, pairs.imag], axis=-1) @ operator.T
    effective = real_readout_kernels(beam, operator)
    measured = engine.contract_blocks(effective, sky, phases).sum(axis=1)
    np.testing.assert_allclose(measured, expected, atol=4e-13)


def test_precontracted_angular_weights_match_full_spin_pair_time_chain():
    lmax = 4
    _, intrinsic = harmonic_ensemble(lmax)
    operator = np.random.default_rng(31).normal(size=(3, 20))
    effective = real_readout_kernels(random_kernel(lmax, 30), operator)
    phases = np.exp(
        -1j * np.array([0.2, 0.9])[:, None] * np.arange(-lmax, lmax + 1)
    )
    transform = SpinTransform.gauss_legendre(lmax, 128)
    projected = AngularReadoutProjector(
        effective, phases, transform, intrinsic
    )
    screen = AngularScreen(theta0_rad=0.7)
    x, y = np.array([21.0, 75.0]), np.array([48.0, 18.0])
    for term in ("smooth", "speckle", "total"):
        actual = projected.screen_covariance(x, y, screen, term=term)
        for i in range(2):
            spectra = rotated_spin_spectra(
                transform, intrinsic, x[i], y[i], screen
            )[term]
            chain = evaluate_mode_moments(
                pair_mode_moments(effective, helicity_covariance(**spectra)),
                phases,
            )["covariance"]
            np.testing.assert_allclose(
                actual[i], chain.reshape(6, 6), atol=3e-12
            )
    reverse = projected.screen_covariance(y, x, screen)
    np.testing.assert_allclose(actual, reverse.swapaxes(-1, -2), atol=3e-12)
    with pytest.raises(ValueError, match="real sky"):
        AngularReadoutProjector(
            random_kernel(lmax, 32), phases, transform, intrinsic
        )


@pytest.fixture
def instrument_parts():
    import lusee
    from lusee.ReceiverImpedance import JFETReceiver

    return (
        lusee.synthetic_four_port_response(
            freq_mhz=(30.0, 40.0), angular_step_deg=15.0
        ),
        JFETReceiver(),
    )


def test_loaded_operator_matches_direct_instrument_and_blackbody(
    instrument_parts,
):
    resp, receiver = instrument_parts
    xv, yv, _ = polarimeter.zenith_vectors(resp, receiver, 30.0)
    rng = np.random.default_rng(16)
    pairs = rng.normal(size=(7, 1, 10)) + 1j * rng.normal(size=(7, 1, 10))
    matrix = instrument.covariance(
        pairs,
        resp,
        receiver,
        [30.0],
        T_moon=0.0,
        T_ant=0.0,
        impedance_freq_mhz=30.0,
    )
    basis = np.c_[pairs[:, 0].real, pairs[:, 0].imag]
    enclosure = instrument.blackbody_normalization(
        resp, receiver, [30.0], impedance_freq_mhz=30.0
    )
    norm = polarimeter.pseudo_stokes(enclosure, xv, yv)[0, 0]
    for kind in ("science", "stokes"):
        operator = loaded_readout_operator(
            resp, receiver, 30.0, kind, x_vec=xv, y_vec=yv
        )
        expected = (
            instrument.channels(matrix[:, 0])[0]
            if kind == "science"
            else polarimeter.pseudo_stokes(matrix[:, 0], xv, yv)
        )
        np.testing.assert_allclose(
            basis @ operator["operator"].T,
            expected / norm,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            operator["normalization_v2_per_hz_per_k"], norm, rtol=2e-13
        )


def test_invalid_covariance_readout_requests(instrument_parts):
    with pytest.raises(ValueError):
        pair_mode_moments(np.zeros((10, 4, 3, 5)), np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        evaluate_mode_moments(
            dict(
                covariance=np.zeros((5, 10, 10)),
                relation=np.zeros((5, 10, 10)),
            ),
            np.ones((2, 6)),
        )
    resp, receiver = instrument_parts
    with pytest.raises(ValueError):
        loaded_readout_operator(resp, receiver, 30.0, kind="raw_dipoles")
