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


def _frequency_support(fine_freqs_mhz, parent_centers_mhz):
    """Shared coordinate selection for spectra and covariance operators."""
    fine = np.asarray(fine_freqs_mhz, dtype=float)
    centers = np.atleast_1d(np.asarray(parent_centers_mhz, dtype=float))
    if (
        fine.ndim != 1
        or fine.size < 2
        or not np.all(np.isfinite(fine) & (fine > 0))
        or centers.ndim != 1
        or not centers.size
        or not np.all(np.isfinite(centers) & (centers > 0))
    ):
        raise ValueError(
            "frequency coordinates must be positive finite vectors"
        )
    df = np.diff(fine)
    roundoff = 16 * np.finfo(float).eps * abs(fine).max()
    if df[0] <= 0 or not np.allclose(df, df[0], rtol=1e-9, atol=roundoff):
        raise ValueError("fine frequency grid must be uniform")
    selections = []
    for fc in centers:
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
        selections.append((sel, off[sel]))
    return fine, centers, selections


def channel_operator(
    fine_freqs_mhz, parent_centers_mhz, kind="parent", notch=0
):
    """Normalized power-response map W: fine samples -> measured bins.

    Returns (nbin, nfine). Zoom rows are parent-major then FFT-bin order,
    including folded images. Identical coordinates/weights feed integrate.
    The operator acts on sky/system power, not on voltage amplitudes.
    """
    fine, centers, selections = _frequency_support(
        fine_freqs_mhz, parent_centers_mhz
    )
    if kind not in ("parent", "zoom", "ideal_zoom"):
        raise ValueError("unknown channel kind")
    per_parent = 1 if kind == "parent" else 64
    operator = np.zeros((centers.size * per_parent, fine.size))
    for p, (sel, offsets) in enumerate(selections):
        if kind == "parent":
            weights = parent_weights(offsets, notch=notch)[None, :]
        elif kind == "zoom":
            weights = zoom_weights(offsets).T
        else:
            weights = ideal_zoom_weights(offsets).T
        rows = slice(p * per_parent, (p + 1) * per_parent)
        operator[rows, np.flatnonzero(sel)] = weights
    return operator


def integrate(waterfall, fine_freqs_mhz, parent_centers_mhz, notch=0):
    """Convolve a fine-frequency waterfall with the bin responses.

    ``waterfall`` has shape ``(..., nfine, nchan)``. Returns ``parent``
    ``(..., nparent, nchan)`` and ``zoom``/``ideal_zoom``
    ``(..., nparent, 64, nchan)``. Response support must be fully covered.
    """
    fine, _, selections = _frequency_support(
        fine_freqs_mhz, parent_centers_mhz
    )
    waterfall = np.asarray(waterfall)
    if waterfall.ndim < 2 or waterfall.shape[-2] != fine.size:
        raise ValueError("waterfall frequency axis does not match coordinates")
    parents, zooms, ideals = [], [], []
    for sel, o in selections:
        chunk = waterfall[..., sel, :]
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


def project_moments(mean, covariance, relation, operator):
    """Map complex moments by W, W C W^H and W J W^T, with batch axes.

    C is conjugated covariance; J is the nonconjugated relation. Dense
    fine-frequency inputs are for small validation grids. Use the block
    projector for production without allocating a full fine covariance.
    """
    mean, covariance, relation, operator = map(
        np.asarray, (mean, covariance, relation, operator)
    )
    if (
        operator.ndim != 2
        or mean.ndim < 1
        or mean.shape[-1] != operator.shape[1]
        or covariance.shape[-2:] != (mean.shape[-1],) * 2
        or relation.shape != covariance.shape
    ):
        raise ValueError("operator and moment dimensions disagree")
    return dict(
        mean=mean @ operator.T,
        covariance=operator @ covariance @ operator.conj().T,
        relation=operator @ relation @ operator.T,
    )


def project_covariance_blocks(
    fine_freqs_mhz, operator, kernel, block_size=256
):
    """Accumulate W C W^H and W J W^T from coordinate-based callbacks.

    kernel(freq_i_mhz, freq_j_mhz) returns a dict containing covariance
    and relation arrays (..., ni, nj). All frequency pairs are included;
    no stationary or finite-coherence approximation is imposed. Memory
    scales with a frequency block and the requested output covariance.
    """
    fine, operator = np.asarray(fine_freqs_mhz), np.asarray(operator)
    if (
        fine.ndim != 1
        or not fine.size
        or operator.ndim != 2
        or operator.shape[1] != fine.size
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("invalid block projection dimensions or block size")
    totals = {}
    for i in range(0, fine.size, block_size):
        si = slice(i, min(i + block_size, fine.size))
        for j in range(0, fine.size, block_size):
            sj = slice(j, min(j + block_size, fine.size))
            values = kernel(fine[si], fine[sj])
            for name in ("covariance", "relation"):
                value = np.asarray(values[name])
                shape = (fine[si].size, fine[sj].size)
                if value.shape[-2:] != shape or not np.all(np.isfinite(value)):
                    raise ValueError("invalid covariance block from kernel")
                right = operator[:, sj]
                if name == "covariance":
                    right = right.conj()
                contribution = operator[:, si] @ value @ right.T
                if name not in totals:
                    totals[name] = np.zeros_like(contribution, dtype=complex)
                if totals[name].shape != contribution.shape:
                    raise ValueError("inconsistent kernel batch dimensions")
                totals[name] += contribution
    return totals


def white_spectrum_noise_overlap(operator, fine_step_hz):
    """Overlap in 1/Hz for independent fine-spectrum power estimates.

    For a flat system power S and common integration t, channel estimator
    covariance is S² * overlap / t. Diagonal inverses give effective
    bandwidths. This assumes long integration and independent fine spectral
    estimates; the actual temporal filter/window must be checked separately.
    """
    operator = np.asarray(operator, float)
    if (
        operator.ndim != 2
        or not np.all(np.isfinite(operator))
        or not np.isfinite(fine_step_hz)
        or fine_step_hz <= 0
    ):
        raise ValueError("invalid power operator or frequency step")
    return operator @ operator.T / fine_step_hz


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
