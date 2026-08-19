"""The four-port polarimeter and its zenith calibration.

Raw pseudo-dipoles are ``X = E - W`` and ``Y = N - S``.  As built, the
four ports are neither identical nor uncoupled, so an unpolarized source
at zenith does not give ``Q = U = V = 0``.  Two calibrations fix that:

- ``mode="gains"``: real per-port weights ``w_p ~ 1/sqrt(C0_pp)`` plus a
  common X/Y rescale.  Nulls zenith pseudo-Q exactly; a residual U
  survives through the inter-port cross-couplings, which no real
  diagonal gain can remove.
- ``mode="ortho"`` (default): Loewdin ``G^{-1/2}`` orthonormalization of
  the pair in the C0 metric.  X and Y become complex combinations of all
  four ports and zenith Q, U and V all vanish.
"""

import numpy as np

from .instrument import covariance, unpack_channels

# Ports N, E, S, W = 0, 1, 2, 3
Y_VEC = np.array([1.0, 0.0, -1.0, 0.0])  # Y = N - S
X_VEC = np.array([0.0, 1.0, 0.0, -1.0])  # X = E - W


def pseudo_stokes(C, x_vec=None, y_vec=None):
    """Pseudo-Stokes I, Q, U, V from a port covariance ``(..., 4, 4)``.

    ``XX = <|X|^2>``, ``YY = <|Y|^2>``, ``XY = <Y X*>`` and
    ``I = (XX+YY)/2``, ``Q = (XX-YY)/2``, ``U = Re XY``, ``V = Im XY``.
    """
    xv = X_VEC if x_vec is None else np.asarray(x_vec)
    yv = Y_VEC if y_vec is None else np.asarray(y_vec)
    C = np.asarray(C)
    XX = np.einsum("a,b,...ab->...", xv, np.conj(xv), C).real
    YY = np.einsum("a,b,...ab->...", yv, np.conj(yv), C).real
    XY = np.einsum("a,b,...ab->...", yv, np.conj(xv), C)
    return np.stack(
        [0.5 * (XX + YY), 0.5 * (XX - YY), XY.real, XY.imag], axis=-1
    )


def pseudo_stokes_from_channels(packed, x_vec=None, y_vec=None):
    """Pseudo-Stokes straight from the 16 packed real channels."""
    return pseudo_stokes(unpack_channels(packed), x_vec, y_vec)


def check_psd(stokes, rtol=1e-9):
    """Assert the physical bound ``sqrt(Q^2+U^2+V^2) <= I``.

    A runtime check, not just a test: this invariant caught a real sign
    bug in the complex cross-pair kernel decomposition.
    """
    s = np.asarray(stokes)
    I = s[..., 0]  # noqa: E741
    p = np.sqrt(s[..., 1] ** 2 + s[..., 2] ** 2 + s[..., 3] ** 2)
    worst = np.max(p - I * (1.0 + rtol))
    if worst > 0:
        raise ValueError(
            f"PSD violation: sqrt(Q^2+U^2+V^2) exceeds I by {worst:.3e}"
        )


def zenith_covariance(resp, receiver, freq_mhz):
    """Loaded covariance of a unit unpolarized source at exact zenith."""
    kernel = np.asarray(
        resp.pair_stokes_at(0.0, 0.0, np.array([float(freq_mhz)]))
    )  # (npair, nfreq, 4)
    pair = kernel[:, 0, 0][None, None, :]  # unpolarized: I kernel only
    C = covariance(
        pair,
        resp,
        receiver,
        np.array([float(freq_mhz)]),
        T_moon=0.0,
        T_ant=0.0,
    )
    return C[0, 0]


def zenith_port_weights(C0, null_q=True):
    """Real per-port gain weights that equalize the zenith autos."""
    C0 = np.asarray(C0)
    autos = np.diagonal(C0).real
    g = np.exp(np.mean(np.log(autos)))
    w = np.sqrt(g / autos)
    x_vec = np.array([0.0, w[1], 0.0, -w[3]])
    y_vec = np.array([w[0], 0.0, -w[2], 0.0])
    if null_q:
        XX = np.einsum("a,b,ab->", x_vec, x_vec, C0).real
        YY = np.einsum("a,b,ab->", y_vec, y_vec, C0).real
        s = (YY / XX) ** 0.25
        x_vec, y_vec = x_vec * s, y_vec / s
    return x_vec, y_vec


def orthonormalize_xy(C0, x_vec, y_vec):
    """Loewdin-orthonormalize (X, Y) in the metric of ``C0``.

    The pseudo-Stokes Q, U and V of a source with covariance ``C0``
    vanish exactly iff ``conj(x)`` and ``conj(y)`` are C0-orthogonal with
    equal C0-norms, because the polarimeter forms are ``p^H C0 p`` and
    ``q^H C0 p`` with ``p = conj(x)``, ``q = conj(y)``.  The symmetric
    ``G^{-1/2}`` transform achieves that while perturbing the input
    dipole vectors as little as possible.
    """
    C0 = np.asarray(C0)
    P = np.stack(
        [
            np.conj(np.asarray(x_vec, dtype=complex)),
            np.conj(np.asarray(y_vec, dtype=complex)),
        ],
        axis=1,
    )
    G = P.conj().T @ C0 @ P
    scale = np.sqrt(np.real(G[0, 0] * G[1, 1]))
    evals, evecs = np.linalg.eigh(G)
    if evals.min() <= 0:
        raise ValueError("X/Y are degenerate in the C0 metric")
    G_isqrt = (evecs / np.sqrt(evals)) @ evecs.conj().T
    P_new = P @ G_isqrt * np.sqrt(scale)
    return np.conj(P_new[:, 0]), np.conj(P_new[:, 1])


def zenith_vectors(resp, receiver, freq_mhz, mode="ortho"):
    """Calibrated (x_vec, y_vec, C0) for one band center."""
    if mode not in ("gains", "ortho"):
        raise ValueError(f"unknown mode {mode!r}; use 'gains' or 'ortho'")
    C0 = zenith_covariance(resp, receiver, freq_mhz)
    x_vec, y_vec = zenith_port_weights(C0, null_q=True)
    if mode == "ortho":
        x_vec, y_vec = orthonormalize_xy(C0, x_vec, y_vec)
    return x_vec, y_vec, C0
