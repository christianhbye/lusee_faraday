"""Channel moments pinned to direct spectra and independent complex samples."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import channelization as ch  # noqa: E402


def test_channel_operator_matches_direct_bin_responses():
    from lusee.SpectrometerResponse import (
        spectrometer_response,
        spectrometer_response_zoom,
    )

    fine = np.linspace(29.95, 30.05, 1025)
    off = (fine - 30) * 1e6
    rng = np.random.default_rng(221)
    values = rng.normal(size=(2, fine.size, 3))
    measured = ch.integrate(values, fine, [30.0])
    for kind in ("parent", "zoom", "ideal_zoom"):
        operator = ch.channel_operator(fine, [30.0], kind)
        via_operator = np.einsum("bf,tfc->tbc", operator, values)
        np.testing.assert_allclose(
            via_operator, measured[kind].reshape(2, -1, 3), atol=2e-14
        )
        np.testing.assert_allclose(operator.sum(axis=1), 1, atol=2e-14)
    for kind, k in (("parent", 0), ("zoom", 31), ("zoom", 32)):
        expected = (
            spectrometer_response(off, 0)
            if kind == "parent"
            else spectrometer_response_zoom(off, k)
        )
        np.testing.assert_allclose(
            ch.channel_operator(fine, [30.0], kind)[k],
            expected / expected.sum(),
            atol=2e-14,
        )


def test_zoom_operator_retains_nyquist_fold():
    fine = np.linspace(29.95, 30.05, 1025)
    off = (fine - 30) * 1e6
    row = ch.channel_operator(fine, [30.0], "zoom")[32]
    assert row[off < 0].sum() > 0.49
    assert row[off > 0].sum() > 0.49
    # The ideal bin is single-peaked on its nominal negative frequency.
    ideal = ch.channel_operator(fine, [30.0], "ideal_zoom")[32]
    assert ideal[off > 0].sum() < 1e-10


def test_complex_moments_match_independent_finite_sample_ensemble():
    rng = np.random.default_rng(222)
    nf, rank, bins = 9, 4, 3
    matrix = rng.normal(size=(nf, rank)) + 1j * rng.normal(size=(nf, rank))
    operator = rng.normal(size=(bins, nf)) + 1j * rng.normal(size=(bins, nf))
    mean = rng.normal(size=nf) + 1j * rng.normal(size=nf)
    # A finite ensemble with exactly zero mean and unit real covariance.
    unit = np.r_[np.eye(rank), -np.eye(rank)] * np.sqrt(rank)
    samples = unit @ matrix.T + mean
    measured = samples @ operator.T
    measured_mean = measured.mean(axis=0)
    centered = measured - measured_mean
    result = ch.project_moments(
        mean, matrix @ matrix.conj().T, matrix @ matrix.T, operator
    )
    np.testing.assert_allclose(result["mean"], measured_mean, atol=2e-13)
    np.testing.assert_allclose(
        result["covariance"],
        centered.T @ centered.conj() / len(unit),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result["relation"], centered.T @ centered / len(unit), atol=2e-12
    )
    assert np.max(abs(result["relation"])) > 1


def test_block_projection_includes_broad_nonstationary_covariance():
    fine = np.linspace(29.95, 30.05, 65)
    rng = np.random.default_rng(223)
    operator = rng.normal(size=(3, len(fine))) + 1j * rng.normal(
        size=(3, len(fine))
    )

    def mode(freq):
        return (1 + 0.2 * (freq - 30)) * np.exp(120j * (freq - 30))

    def kernel(fi, fj):
        a, b = mode(fi), mode(fj)
        return dict(
            covariance=np.outer(a, b.conj()), relation=0.7 * np.outer(a, b)
        )

    projected = operator @ mode(fine)
    for block_size in (7, 19, 65):
        result = ch.project_covariance_blocks(
            fine, operator, kernel, block_size=block_size
        )
        np.testing.assert_allclose(
            result["covariance"],
            np.outer(projected, projected.conj()),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            result["relation"],
            0.7 * np.outer(projected, projected),
            atol=2e-12,
        )


def test_white_noise_overlap_and_effective_bandwidth():
    operator = np.zeros((2, 8))
    operator[0, 2:6] = 0.25
    operator[1, 4:8] = 0.25
    overlap = ch.white_spectrum_noise_overlap(operator, 100.0)
    np.testing.assert_allclose(1 / np.diag(overlap), 400.0)
    np.testing.assert_allclose(overlap[0, 1] / overlap[0, 0], 0.5)


def test_invalid_channel_covariance_coordinates_and_shapes():
    for fine in ([], [30.0], [30.0, 29.0], [29.95, np.nan, 30.05]):
        with pytest.raises(ValueError):
            ch.channel_operator(fine, [30.0])
    fine = np.linspace(29.95, 30.05, 65)
    with pytest.raises(ValueError):
        ch.channel_operator(fine, [30.0], "voltage")
    with pytest.raises(ValueError):
        ch.integrate(np.zeros((64, 1)), fine, [30.0])
    with pytest.raises(ValueError):
        ch.project_moments(np.zeros(3), np.eye(3), np.eye(3), np.ones((2, 4)))
    with pytest.raises(ValueError):
        ch.project_covariance_blocks(
            fine, np.ones((2, 65)), None, block_size=0
        )
