"""LuSEE spectrometer channelization.

luseepy owns the bin response itself (``lusee.spectrometer_response`` and
``lusee.spectrometer_response_zoom``); what it does not own, and what
lives here, is the binning: which fine channels feed which parent and
zoom bins, and the ideal Gaussian comparison bins.

Zoom bins use FFT ordering -- bin 0 is the parent centre, bins 1-32 are
positive offsets, bins 33-63 negative.  The zoom FFT runs on the
critically sampled 25 kHz parent stream, so the bins carry folded
images; that folding is physical and is removed downstream, not here.
"""

import numpy as np

ZOOM_STEP_HZ = 25000.0 / 64  # 390.625 Hz
FINE_STEP_HZ = 25000.0 / 2048  # 12.20703125 Hz
PARENT_HALF_WIDTH_HZ = 50000.0


def zoom_bin_offsets_hz():
    """Nominal centre offsets of the 64 zoom bins, FFT ordering."""
    k = np.arange(64)
    return np.where(k < 32, k, k - 64) * ZOOM_STEP_HZ


def parent_weights(offsets_hz, notch=0):
    """Normalized parent-bin weights on a fine offset grid."""
    from lusee.SpectrometerResponse import spectrometer_response

    w = spectrometer_response(np.asarray(offsets_hz, dtype=float), notch)
    return w / w.sum()


def zoom_weights(offsets_hz):
    """Normalized zoom-bin weights -> ``(noffsets, 64)``."""
    from lusee.SpectrometerResponse import spectrometer_response_zoom

    off = np.asarray(offsets_hz, dtype=float)
    W = np.stack(
        [spectrometer_response_zoom(off, k) for k in range(64)], axis=-1
    )
    return W / W.sum(axis=0, keepdims=True)


def ideal_zoom_weights(offsets_hz, fwhm_hz=ZOOM_STEP_HZ):
    """Gaussian 'ideal' zoom bins at the nominal centres."""
    off = np.asarray(offsets_hz, dtype=float)
    centers = zoom_bin_offsets_hz()
    sigma = fwhm_hz / (2 * np.sqrt(2 * np.log(2)))
    W = np.exp(-0.5 * ((off[:, None] - centers[None, :]) / sigma) ** 2)
    return W / W.sum(axis=0, keepdims=True)


def integrate(waterfall, fine_freqs_mhz, parent_centers_mhz, notch=0):
    """Convolve a fine-frequency waterfall with the bin responses.

    ``waterfall`` has shape ``(..., nfine, nchan)``.  Returns a dict with
    ``parent`` ``(..., nparent, nchan)``, ``zoom`` and ``ideal_zoom``
    ``(..., nparent, 64, nchan)``.  A parent bin whose +-50 kHz response
    support is not fully covered by the fine grid raises.
    """
    fine = np.asarray(fine_freqs_mhz, dtype=float)
    df = np.diff(fine)
    if not np.allclose(df, df[0], rtol=1e-9):
        raise ValueError("fine frequency grid must be uniform")
    parents, zooms, ideals = [], [], []
    for fc in np.atleast_1d(parent_centers_mhz):
        off = (fine - fc) * 1e6
        sel = np.abs(off) <= PARENT_HALF_WIDTH_HZ + 1e-6
        covered = sel.any() and (
            off[sel].min() <= -PARENT_HALF_WIDTH_HZ + 1e-3
            and off[sel].max() >= PARENT_HALF_WIDTH_HZ - 1e-3
        )
        if not covered:
            raise ValueError(
                f"fine grid does not cover the response of bin {fc} MHz"
            )
        chunk = waterfall[..., sel, :]
        o = off[sel]
        parents.append(
            np.einsum("...fc,f->...c", chunk, parent_weights(o, notch=notch))
        )
        zooms.append(np.einsum("...fc,fz->...zc", chunk, zoom_weights(o)))
        ideals.append(
            np.einsum("...fc,fz->...zc", chunk, ideal_zoom_weights(o))
        )
    return {
        "parent": np.stack(parents, axis=-2),
        "zoom": np.stack(zooms, axis=-3),
        "ideal_zoom": np.stack(ideals, axis=-3),
    }


def zoom_frequency_grid(parent_centers_mhz):
    """Sorted zoom-bin centre frequencies (MHz) and their index map.

    Returns ``(freqs_sorted, order)`` where ``order[i] = (parent_index,
    zoom_bin)`` names the bin at ``freqs_sorted[i]``.  The grid is
    contiguous at ``ZOOM_STEP_HZ`` across adjacent parents.
    """
    offs = zoom_bin_offsets_hz()
    entries = []
    for p, fc in enumerate(np.atleast_1d(parent_centers_mhz)):
        for k in range(64):
            entries.append((fc + offs[k] * 1e-6, p, k))
    entries.sort()
    freqs = np.array([e[0] for e in entries])
    order = [(e[1], e[2]) for e in entries]
    return freqs, order
