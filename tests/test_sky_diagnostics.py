import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday import sky as sky_mod  # noqa: E402
from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12


def test_spectral_component_count_scales_with_the_depth_range():
    freqs = np.array([29.9, 30.1])
    few = sky_mod.spectral_component_count(-30.0, 30.0, freqs)
    many = sky_mod.spectral_component_count(-2400.0, 2400.0, freqs)
    assert 1 <= few < many
    assert many > 1000  # the full Galactic screen is expensive


def test_spectral_component_count_is_one_for_a_uniform_screen():
    freqs = np.array([29.9, 30.1])
    assert sky_mod.spectral_component_count(50.0, 50.0, freqs) == 1


def test_nyquist_nside_grows_with_frequency_and_gradient():
    pytest.importorskip("healpy")
    import healpy as hp

    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    smooth = hp.smoothing(rng.normal(size=npix), fwhm=0.5) * 10.0
    rough = rng.normal(size=npix) * 10.0
    assert sky_mod.nyquist_nside(rough, 30.0) > sky_mod.nyquist_nside(
        smooth, 30.0
    )
    # lambda^2 grows as nu^-2, so a lower frequency needs finer pixels
    assert sky_mod.nyquist_nside(smooth, 10.0) > sky_mod.nyquist_nside(
        smooth, 30.0
    )


def test_from_rm_map_refuses_an_unresolved_screen():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(1)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    z = np.zeros(npix)
    rm = rng.normal(size=npix) * 300.0  # wildly unresolved
    with pytest.raises(ValueError) as excinfo:
        FaradaySky.from_rm_map(I, z, z, rm, np.array([29.9, 30.1]), lmax=LMAX)
    message = str(excinfo.value)
    assert "nside" in message
    assert "allow_pixelwise" in message


def test_from_rm_map_accepts_a_resolved_screen():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(2)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    z = np.zeros(npix)
    rm = hp.smoothing(rng.normal(size=npix), fwhm=1.0) * 1e-3
    sky = FaradaySky.from_rm_map(
        I, z, z, rm, np.array([29.9, 30.1]), lmax=LMAX
    )
    assert sky.n_components >= 1


def test_component_construction_logs_the_resolved_engine(caplog):
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    npix = hp.nside2npix(NSIDE)
    ones = np.ones(npix)
    with caplog.at_level("INFO", logger="lusee_faraday.sky"):
        FaradaySky.uniform_screen(
            ones, 0 * ones, 0 * ones, phi_fd=0.0, lmax=LMAX
        )
    assert any("transform engines" in r.message for r in caplog.records)
