"""Adapters from instrument models to harmonic pair-Stokes responses.

Two arms share this module:

- the as-built four-port instrument, read from a BGL_v16 response
  artifact through ``lusee.InstrumentResponse``;
- the symmetric pseudo-dipoles of the paper's Fig 4, built from a 2-port
  Jones FITS file (added in a later task).

Both end up as complex pair-Stokes alms in croissant's harmonic dual, so
the contraction in :mod:`lusee_faraday.engine` does not know which arm it
is serving.
"""

import numpy as np

from .conventions import PORT_PAIRS


def load_response(path):
    """Load a v3 response artifact without re-running slow validation.

    HDU names follow the BGL_v16 artifact layout used by
    ``fourport.load_response_fast`` (``"freq"``, ``"theta"``, ``"phi"``
    and ``cplx("H_theta")`` etc, producing e.g. ``H_theta_real``) -- not
    the ``HTHETA``/``HTHETA_REAL`` names a naive reading of the FITS
    schema might suggest.
    """
    import fitsio
    from lusee.InstrumentResponse import InstrumentResponse

    with fitsio.FITS(str(path)) as f:
        header = dict(f[0].read_header())

        def cplx(name):
            return f[f"{name}_real"].read() + 1j * f[f"{name}_imag"].read()

        return InstrumentResponse.from_arrays(
            f["freq"].read(),
            f["theta"].read(),
            f["phi"].read(),
            cplx("H_theta"),
            cplx("H_phi"),
            cplx("ZA"),
            cplx("Rsky"),
            cplx("Rmoon"),
            cplx("Rloss"),
            validated=False,
            metadata={**header, "VALIDATED": False},
            ZLoad=cplx("ZLoad"),
        )


def native_channel_index(resp, freq_mhz):
    """Index of ``freq_mhz`` in the response's native grid.

    The fixed-beam approximation is an assertion, not a default: an
    off-grid frequency would be silently interpolated by luseepy's
    ``FrequencyMap``, smearing the beam across the band and putting
    non-Faraday structure into delay space.
    """
    freq = np.asarray(resp.freq, dtype=float)
    idx = int(np.argmin(np.abs(freq - freq_mhz)))
    if abs(freq[idx] - freq_mhz) > 1e-9:
        raise ValueError(
            f"{freq_mhz} MHz is not a native response channel; "
            f"nearest is {freq[idx]} MHz."
        )
    return idx


def pair_stokes_from_jones(h_theta, h_phi, pairs=PORT_PAIRS):
    """Complex bare pair-Stokes maps from Jones components.

    Mirrors ``lusee.InstrumentResponse.pair_stokes_maps`` exactly, so the
    two-port arm inherits a convention already validated against luseepy.
    Input arrays are indexed ``(port, freq, ...)``; the output is
    ``(pair, freq, 4, ...)`` with the 4 axis in I, Q, U, V order, matching
    ``InstrumentResponse.all_pair_stokes_maps()``.
    """
    at_all = np.asarray(h_theta)
    ap_all = np.asarray(h_phi)
    out = []
    for a, b in pairs:
        at, ap = at_all[a], ap_all[a]
        bt, bp = np.conj(at_all[b]), np.conj(ap_all[b])
        out.append(
            np.stack(
                [
                    at * bt + ap * bp,
                    at * bt - ap * bp,
                    at * bp + ap * bt,
                    1j * (ap * bt - at * bp),
                ],
                axis=1,
            )
        )
    return np.stack(out, axis=0)


def four_port_pair_alms(resp, freq_mhz, lmax):
    """Physical pair-Stokes alms at ONE native channel -> (10, 4, L, 2L-1).

    luseepy applies the ``eta0 / lambda^2`` scaling that turns bare m^2
    maps into the physical W kernel, so the result is directly
    contractable with a sky in kelvin.
    """
    idx = native_channel_index(resp, freq_mhz)
    freq = np.asarray(resp.freq, dtype=float)
    alms, _ = resp.pair_stokes_alms(int(lmax), np.array([freq[idx]]))
    return np.asarray(alms)[:, 0]


TWO_PORT_PAIRS = ((0, 0), (0, 1), (1, 1))


def two_port_jones_from_fits(path, freq_mhz, orientation="y"):
    """Load a 2-port Jones FITS and build the orthogonal pseudo-dipole.

    The file stores only the upper hemisphere on a 1-degree grid with a
    duplicated ``phi = 360`` column.  The lower hemisphere is zero-filled.
    Rotating the antenna about z translates the tangent-basis components
    in phi, so the partner dipole is a roll of the phi axis.
    """
    from astropy.io import fits

    with fits.open(str(path)) as f:
        e_theta = f["Etheta_real"].data + 1j * f["Etheta_imag"].data
        e_phi = f["Ephi_real"].data + 1j * f["Ephi_imag"].data
        idx = int(np.argwhere(f["freq"].data == freq_mhz)[0, 0])
    e_theta = e_theta[idx][..., :-1]
    e_phi = e_phi[idx][..., :-1]
    lower = np.zeros_like(e_theta)[:-1, :]
    e_theta = np.concatenate([e_theta, lower], axis=0)
    e_phi = np.concatenate([e_phi, lower], axis=0)

    if orientation == "y":
        rolls = (270, 0)
    elif orientation == "x":
        rolls = (0, 90)
    else:
        raise ValueError("orientation must be 'x' or 'y'")
    h_theta = np.stack([np.roll(e_theta, r, axis=-1) for r in rolls])
    h_phi = np.stack([np.roll(e_phi, r, axis=-1) for r in rolls])
    return h_theta, h_phi


def two_port_pair_alms(h_theta, h_phi, theta_deg, phi_deg, lmax):
    """Pair-Stokes alms for two pseudo-dipoles -> (3, 4, L, 2L-1).

    Unitless: this arm has no impedance model and no receiver loading,
    so it is the direct analogue of the paper's Fig 4 pipeline rather
    than of the as-built four-port instrument.
    """
    import croissant as cro

    # pair_stokes_from_jones expects a leading frequency axis (it
    # mirrors luseepy's (port, freq, ...) layout).  This arm's Jones
    # arrays are (port, ntheta, nphi) with no frequency axis at all, so
    # a singleton one is inserted here -- at the call site, not inside
    # the shared helper, which the four-port arm also relies on.
    maps = pair_stokes_from_jones(
        h_theta[:, None], h_phi[:, None], TWO_PORT_PAIRS
    )  # -> (3, 1, 4, ntheta, nphi)
    beam = cro.PairStokesBeam(
        maps,
        np.array([1.0]),
        TWO_PORT_PAIRS,
        sampling="mwss",
        frame="topo",
    )
    return np.asarray(beam.compute_alm(lmax=int(lmax)))[:, 0]
