import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lusee_faraday.sky import FaradaySky  # noqa: E402

NSIDE = 16
LMAX = 12


@pytest.fixture(scope="module")
def maps():
    pytest.importorskip("croissant")
    import jax

    jax.config.update("jax_enable_x64", True)
    import healpy as hp

    rng = np.random.default_rng(0)
    npix = hp.nside2npix(NSIDE)
    I = np.abs(rng.normal(size=npix)) + 10.0
    Q = rng.normal(size=npix) * 0.1
    U = rng.normal(size=npix) * 0.1
    return I, Q, U


def test_uniform_screen_uses_one_component(maps):
    sky = FaradaySky.uniform_screen(*maps, phi_fd=250.0, lmax=LMAX)
    assert sky.component_alms.shape == (1, 4, LMAX + 1, 2 * LMAX + 1)
    assert sky.phi_fd.shape == (1,)


def test_coeffs_shape_and_flat_spectrum(maps):
    sky = FaradaySky.uniform_screen(*maps, phi_fd=0.0, lmax=LMAX)
    freqs = np.array([29.9, 30.0, 30.1])
    c = sky.coeffs(freqs)
    assert c.shape == (1, 3, 4)
    assert np.allclose(c, 1.0)  # no Faraday, no spectral index


def test_polarized_alm_at_freq_matches_rotating_the_maps_directly(maps):
    """The exactness claim the whole refactor rests on."""
    import croissant as cro

    from lusee_faraday.conventions import faraday_phase_cosmo

    I, Q, U = maps
    phi = 137.0
    freqs = np.array([29.9, 30.0, 30.2])

    sky = FaradaySky.uniform_screen(I, Q, U, phi_fd=phi, lmax=LMAX)
    ours = sky.polarized_alm_at_freq(freqs, lmax=LMAX)
    assert ours.shape == (3, 4, LMAX + 1, 2 * LMAX + 1)

    phase = faraday_phase_cosmo(phi, freqs)  # (nfreq,)
    direct = []
    for i, f in enumerate(freqs):
        P = (Q + 1j * U) * phase[i]
        data = np.stack([I, P.real, P.imag, np.zeros_like(I)])[None]
        rotated = cro.PolarizedSky(
            data,
            np.array([f]),
            sampling="healpix",
            coord="galactic",
            convention="COSMO",
        )
        direct.append(np.asarray(rotated.compute_alm(lmax=LMAX))[0])
    direct = np.stack(direct)

    scale = np.abs(direct).max()
    assert np.abs(ours - direct).max() < 1e-10 * scale


def test_spectral_index_scales_the_right_blocks(maps):
    I, Q, U = maps
    sky = FaradaySky.uniform_screen(
        I,
        Q,
        U,
        phi_fd=0.0,
        lmax=LMAX,
        beta_i=-2.55,
        ref_freq_i=408.0,
        beta_qu=-2.8,
        ref_freq_qu=23e3,
    )
    freqs = np.array([30.0])
    c = sky.coeffs(freqs)
    assert np.isclose(c[0, 0, 0], (30.0 / 408.0) ** -2.55)
    assert np.isclose(c[0, 0, 2], (30.0 / 23e3) ** -2.8)
    assert np.isclose(c[0, 0, 3], (30.0 / 23e3) ** -2.8)


def test_satisfies_the_luseepy_polarized_sky_protocol(maps):
    from lusee.FullStokesSimulator import _validate_polarized_sky_metadata

    sky = FaradaySky.uniform_screen(*maps, phi_fd=0.0, lmax=LMAX)
    _validate_polarized_sky_metadata(sky, require_frequency_units=True)


def test_i_only_has_no_polarized_blocks(maps):
    I, _, _ = maps
    sky = FaradaySky.i_only(I, lmax=LMAX)
    alm = sky.polarized_alm_at_freq(np.array([10.0, 30.0, 50.0]), lmax=LMAX)
    assert np.abs(alm[:, 2]).max() == 0.0
    assert np.abs(alm[:, 3]).max() == 0.0
    assert np.abs(alm[:, 0]).max() > 0.0
