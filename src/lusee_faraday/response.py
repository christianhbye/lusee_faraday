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
