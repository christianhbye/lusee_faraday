"""Loaded receiver PSD and joint radiometer covariance of real readouts.

All voltage/current spectral covariances use one-sided V²/Hz and A²/Hz.
Proper complex spectral-voltage samples have covariance S. A power average
over bandwidth B and time tau has B*tau independent complex samples under
the stated long-integration/independent-fine-frequency approximation.
"""

import numpy as np

from .conventions import PORT_PAIRS


def check_spectral_covariance(value):
    """Require a finite Hermitian positive semidefinite spectral matrix."""
    s = np.asarray(value)
    if (
        s.ndim < 2
        or s.shape[-1] != s.shape[-2]
        or not s.shape[-1]
        or not np.all(np.isfinite(s))
    ):
        raise ValueError("invalid spectral covariance shape or values")
    scale = np.maximum(np.max(abs(s), axis=(-1, -2)), np.finfo(float).tiny)
    difference = np.max(abs(s - s.swapaxes(-1, -2).conj()), axis=(-1, -2))
    if np.any(difference > 1e-11 * scale):
        raise ValueError("spectral covariance is not Hermitian")
    if np.any(np.linalg.eigvalsh(s)[..., 0] < -1e-11 * scale):
        raise ValueError("spectral covariance is not positive semidefinite")
    return s


def loaded_amplifier_covariance(
    za, zl, voltage_psd, current_psd, cross_psd=None
):
    """Covariance of e+Z_eff i at the receiver input reference plane.

    Current is defined positive into the shared antenna/load node.
    Z_eff=ZL (ZA+ZL)^-1 ZA comes from luseepy's loading matrix. cross_psd
    is E[e i^H]. Voltage noise is added at the measurement plane; current
    noise couples through the full parallel network, including port coupling.
    No impedance model supplies missing amplifier-noise parameters.
    """
    from lusee.Covariance import loading_matrix

    za, zl, ee, ii = np.broadcast_arrays(za, zl, voltage_psd, current_psd)
    if za.shape[-2:] != (4, 4):
        raise ValueError("four-port impedances and PSD matrices are required")
    ei = (
        np.zeros_like(ee)
        if cross_psd is None
        else np.broadcast_to(cross_psd, ee.shape)
    )
    joint = np.concatenate(
        [
            np.concatenate([ee, ei], axis=-1),
            np.concatenate([ei.swapaxes(-1, -2).conj(), ii], axis=-1),
        ],
        axis=-2,
    )
    # Normalize voltage and current coordinates before the PSD test; their
    # SI magnitudes differ by many orders and share no common unit scale.
    diagonal = np.diagonal(joint, axis1=-2, axis2=-1).real
    if np.any(diagonal < 0):
        raise ValueError("negative voltage/current spectral variance")
    reference = np.sqrt(np.maximum(diagonal, np.finfo(float).tiny))
    check_spectral_covariance(
        joint / reference[..., :, None] / reference[..., None, :]
    )
    parallel = np.asarray(loading_matrix(za, zl)) @ za
    result = (
        ee
        + parallel @ ii @ parallel.swapaxes(-1, -2).conj()
        + ei @ parallel.swapaxes(-1, -2).conj()
        + parallel @ ei.swapaxes(-1, -2).conj()
    )
    return check_spectral_covariance(result)


def diagonal_amplifier_covariance(
    za, zl, voltage_asd_v_rt_hz, current_asd_a_rt_hz
):
    """Explicit uncorrelated per-port ASD inputs; scalar values broadcast."""
    shape = np.broadcast_shapes(np.shape(za), np.shape(zl))
    e = np.broadcast_to(voltage_asd_v_rt_hz, shape[:-2] + (4,))
    i = np.broadcast_to(current_asd_a_rt_hz, shape[:-2] + (4,))
    if not np.all(np.isfinite(e) & np.isfinite(i) & (e >= 0) & (i >= 0)):
        raise ValueError("amplifier ASDs must be finite and nonnegative")
    return loaded_amplifier_covariance(
        za,
        zl,
        e[..., :, None] ** 2 * np.eye(4),
        i[..., :, None] ** 2 * np.eye(4),
    )


def science_quadratic_matrices():
    """Hermitian matrices A with packed product = v^H A v."""
    matrices = []
    for a, b in PORT_PAIRS:
        real = np.zeros((4, 4), complex)
        if a == b:
            real[a, a] = 1
            matrices.append(real)
        else:
            real[a, b] = real[b, a] = 0.5
            imag = np.zeros((4, 4), complex)
            imag[a, b], imag[b, a] = 0.5j, -0.5j
            matrices.extend([real, imag])
    return np.asarray(matrices)


def stokes_quadratic_matrices(x_vec, y_vec, normalization=1.0):
    """The production pseudo-I/Q/U/V convention as Hermitian quadratics."""
    x, y = np.asarray(x_vec), np.asarray(y_vec)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or not x.size
        or not np.all(np.isfinite(x) & np.isfinite(y))
        or not np.isfinite(normalization)
        or normalization <= 0
    ):
        raise ValueError("invalid pseudo-Stokes vectors or normalization")
    xx, yy = np.outer(x.conj(), x), np.outer(y.conj(), y)
    xy = np.outer(x.conj(), y)
    return (
        np.array(
            [
                (xx + yy) / 2,
                (xx - yy) / 2,
                (xy + xy.conj().T) / 2,
                (xy - xy.conj().T) / (2j),
            ]
        )
        / normalization
    )


def quadratic_radiometer_covariance(
    system_psd, readout, effective_samples=1.0
):
    """Cov(q_p,q_q)=Tr(A_p S A_q S)/N for proper complex voltages.

    Cross-product and Re/Im correlations are retained. Readout matrices
    include any calibration/kelvin normalization. N=B*tau for a top-hat
    independent-sample estimator, with B in Hz and tau in seconds.
    """
    s = check_spectral_covariance(system_psd)
    a = np.asarray(readout)
    n = np.asarray(effective_samples, float)
    if (
        a.ndim != 3
        or a.shape[-2:] != s.shape[-2:]
        or not np.all(np.isfinite(a))
        or np.max(abs(a - a.swapaxes(-1, -2).conj()))
        > 1e-12 * max(np.max(abs(a)), np.finfo(float).tiny)
        or not np.all(np.isfinite(n) & (n > 0))
    ):
        raise ValueError("invalid real readout or independent-sample count")
    result = np.einsum("pab,...bc,qcd,...da->...pq", a, s, a, s, optimize=True)
    scale = np.maximum(
        np.max(abs(result), axis=(-1, -2)), np.finfo(float).tiny
    )
    if np.any(np.max(abs(result.imag), axis=(-1, -2)) > 1e-10 * scale):
        raise ValueError(
            "quadratic covariance has a significant imaginary part"
        )
    return result.real / n[..., None, None]


def channel_radiometer_covariance(
    system_psd, operator, fine_step_hz, integration_s, readout
):
    """Full bin/readout covariance for independent fine-spectrum powers.

    S is constant (port,port) or (fine frequency,port,port). Frequency
    variation and every channel overlap are included. The result has
    shape (bin,readout,bin,readout). Temporal-window correlations beyond
    this long-integration approximation require a separate filter model.
    """
    w = np.asarray(operator)
    if (
        w.ndim != 2
        or not np.isrealobj(w)
        or not np.all(np.isfinite(w))
        or not np.isfinite(fine_step_hz)
        or fine_step_hz <= 0
        or not np.isfinite(integration_s)
        or integration_s <= 0
    ):
        raise ValueError("invalid power-response operator or integration")
    fine = quadratic_radiometer_covariance(system_psd, readout)
    if fine.ndim == 2:
        value = np.einsum("bc,pq->bpcq", w @ w.T, fine)
    elif fine.ndim == 3 and fine.shape[0] == w.shape[1]:
        value = np.einsum("bf,fpq,cf->bpcq", w, fine, w, optimize=True)
    else:
        raise ValueError("frequency-dependent system PSD grid disagrees")
    return value / (fine_step_hz * integration_s)
