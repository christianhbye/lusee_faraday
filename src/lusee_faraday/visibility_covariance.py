"""Spin-2 sky statistics through all pair kernels and the loaded readout.

This covariance is over the random polarized sky/screen ensemble, distinct
from the 4-by-4 electric-field covariance and its radiometer sampling noise.
All normalized helicity spectra describe real Q/U fields in the convention
defined by spin_covariance. No sky realization at the Faraday scale is drawn.
"""

import numpy as np

from .conventions import DUAL_BLOCKS, PORT_PAIRS, PRODUCT_LABELS
from .gaussian_screen import real_covariance


def real_readout_kernels(pair_alms, operator):
    """Fold a real linear pair readout into harmonic kernels.

    For a real I,Q,U,V sky, contracting these kernels is equivalent to
    applying operator to (Re pairs, Im pairs). The conjugated contribution
    flips helicity and m with the spherical-harmonic reality parity.
    """
    beam, matrix = np.asarray(pair_alms), np.asarray(operator)
    if (
        beam.ndim != 4
        or beam.shape[1] != 4
        or beam.shape[-1] != 2 * beam.shape[-2] - 1
        or matrix.ndim != 2
        or matrix.shape[1] != 2 * beam.shape[0]
        or not np.isrealobj(matrix)
    ):
        raise ValueError("invalid real readout kernels or operator")
    count = beam.shape[0]
    alpha = (matrix[:, :count] - 1j * matrix[:, count:]) / 2
    flip = [
        DUAL_BLOCKS.index(name) for name in ("I", "V", "P_PLUS", "P_MINUS")
    ]
    parity = (-1.0) ** (np.arange(beam.shape[-1]) - beam.shape[-2] + 1)
    conjugate = beam[:, flip, :, ::-1].conj() * parity
    return np.einsum("op,pclm->oclm", alpha, beam) + np.einsum(
        "op,pclm->oclm", alpha.conj(), conjugate
    )


class AngularReadoutProjector:
    """Precontract spin transforms and real instrument kernels.

    Stores four angular weights for each pair of selected time/readouts.
    A frequency tile then requires only angular integration, without an
    ell,m transform for every frequency pair. No angular core is dropped.
    The supplied quadrature and beam band limit must be converged apart.
    """

    def __init__(self, real_kernels, phases, transform, intrinsic):
        from .spin_covariance import helicity_covariance

        beam, phases = np.asarray(real_kernels), np.asarray(phases)
        if (
            beam.ndim != 4
            or beam.shape[1] != 4
            or beam.shape[-2] != transform.lmax + 1
            or beam.shape[-1] != 2 * transform.lmax + 1
            or phases.ndim != 2
            or phases.shape[-1] != beam.shape[-1]
        ):
            raise ValueError("beam, phases and spin band limit disagree")
        blocks = [DUAL_BLOCKS.index("P_MINUS"), DUAL_BLOCKS.index("P_PLUS")]
        b = beam[:, blocks]
        basis = helicity_covariance(
            np.array([1, 1j, 0, 0]), np.array([0, 0, 1, 1j])
        )
        modes = np.einsum(
            "pclm,kcd,qdlm->klmpq", b, basis.conj(), b.conj(), optimize=True
        )
        response = np.einsum(
            "tm,klmpq,um->kltpuq", phases, modes, phases.conj(), optimize=True
        )
        scale = max(float(np.max(abs(response))), np.finfo(float).tiny)
        if np.max(abs(response.imag)) > 2e-11 * scale:
            raise ValueError("kernels do not represent real sky readouts")
        self.size = phases.shape[0] * beam.shape[0]
        response = response.real.reshape(
            4, transform.lmax + 1, self.size, self.size
        )
        self.weights = np.empty(
            (4, transform.theta.size, self.size, self.size)
        )
        for k in range(4):
            d = transform.plus if k < 2 else transform.minus
            self.weights[k] = (
                2
                * np.pi
                * np.einsum("lpq,ln,n->npq", response[k], d, transform.weights)
            )
        self.xi = transform.correlations(intrinsic["plus"], intrinsic["minus"])
        self.theta = transform.theta.copy()

    def phase_covariance(self, plus_phase, minus_phase):
        """Integrate raw or centered phase kernels; output (..., D, D)."""
        plus = self.xi["plus"] * np.asarray(plus_phase)
        minus = self.xi["minus"] * np.asarray(minus_phase)
        plus, minus = np.broadcast_arrays(plus, minus)
        if plus.shape[-1] != self.theta.size:
            raise ValueError("phase kernels disagree with angular grid")
        components = np.stack(
            [plus.real, plus.imag, minus.real, minus.imag], axis=-2
        )
        return np.einsum("...kn,knpq->...pq", components, self.weights)

    def screen_covariance(self, x, y, screen, fraction=1.0, term="total"):
        """Covariance of a single rotated intrinsic random spin-2 field."""
        from .spin_covariance import screen_phase_moments

        phase = screen_phase_moments(x, y, self.theta, screen, fraction)
        if term == "total":
            a, b = phase["second"], phase["nonconjugated_second"]
        elif term == "speckle":
            a, b = phase["covariance"], phase["relation"]
        elif term == "smooth":
            a = (phase["mean_x"] * phase["mean_y"].conj())[..., None]
            b = (phase["mean_x"] * phase["mean_y"])[..., None]
        else:
            raise ValueError("term must be total, smooth or speckle")
        return self.phase_covariance(a, b)


def pair_mode_moments(pair_alms, helicity, other_pair_alms=None):
    """Return pair covariance/relation per sidereal harmonic m.

    Alms are (pair, block, ell, m), with m=-lmax..lmax and the existing
    physical half-response normalization. helicity (..., ell, 2, 2) is
    E[a_c(nu) a_d(nu')*], ordered P_MINUS/P_PLUS. Output moments have
    shape (..., m, pair, other_pair), before time phases or loading.
    Cross-helicity and cross-pair terms are retained without approximation.
    """
    left = np.asarray(pair_alms)
    right = left if other_pair_alms is None else np.asarray(other_pair_alms)
    h = np.asarray(helicity)
    if (
        left.ndim != 4
        or right.ndim != 4
        or left.shape[1] != 4
        or right.shape[1] != 4
        or left.shape[2:] != right.shape[2:]
        or left.shape[-1] != 2 * left.shape[-2] - 1
        or h.shape[-3:] != (left.shape[-2], 2, 2)
    ):
        raise ValueError("invalid pair kernels or helicity dimensions")
    blocks = [DUAL_BLOCKS.index("P_MINUS"), DUAL_BLOCKS.index("P_PLUS")]
    b, d = left[:, blocks], right[:, blocks]
    covariance = np.einsum(
        "pclm,...lcd,qdlm->...mpq", b, h.conj(), d.conj(), optimize=True
    )
    relation = np.einsum(
        "pclm,...lcd,qdlm->...mpq",
        b,
        h[..., ::-1, :],
        d[..., ::-1],
        optimize=True,
    )
    m = np.arange(left.shape[-1]) - (left.shape[-2] - 1)
    relation *= (-1.0) ** m[:, None, None]
    return dict(covariance=covariance, relation=relation)


def evaluate_mode_moments(moments, phases, other_phases=None):
    """Evaluate time-pair moments from exp(-i m angle) phase matrices.

    Returns (..., time, pair, other_time, other_pair). Use a selected time
    grid or retain mode moments to avoid a dense day-by-day covariance.
    """
    left = np.asarray(phases)
    right = left if other_phases is None else np.asarray(other_phases)
    c, j = np.asarray(moments["covariance"]), np.asarray(moments["relation"])
    if (
        left.ndim != 2
        or right.ndim != 2
        or c.shape != j.shape
        or c.ndim < 3
        or left.shape[-1] != c.shape[-3]
        or right.shape[-1] != c.shape[-3]
    ):
        raise ValueError("time phases and mode dimensions disagree")
    return dict(
        covariance=np.einsum(
            "tm,...mpq,um->...tpuq", left, c, right.conj(), optimize=True
        ),
        relation=np.einsum(
            "tm,...mpq,um->...tpuq", left, j, right[:, ::-1], optimize=True
        ),
    )


def real_readout_covariance(covariance, relation, left_map, right_map=None):
    """Project pair moments (..., pair, pair) onto real linear readouts.

    Map columns are (Re all pairs, Im all pairs). Leading axes can carry
    frequencies or a time pair. This retains the relation even for a
    real-valued readout of complex cross pairs.
    """
    c, j = np.asarray(covariance), np.asarray(relation)
    a = np.asarray(left_map)
    b = a if right_map is None else np.asarray(right_map)
    if (
        c.shape != j.shape
        or c.ndim < 2
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != 2 * c.shape[-2]
        or b.shape[1] != 2 * c.shape[-1]
        or not np.isrealobj(a)
        or not np.isrealobj(b)
    ):
        raise ValueError("real readout maps and pair covariance disagree")
    return a @ real_covariance(c, j) @ b.T


def loaded_readout_operator(
    resp,
    receiver,
    center_mhz,
    kind="stokes",
    x_vec=None,
    y_vec=None,
    kelvin=True,
):
    """Map pair (Re,Im) to calibrated pseudo-Stokes or 16 science products.

    Instrument assembly/loading/packing come from the existing production
    functions. All thermal terms vanish and impedances are frozen here.
    Kelvin uses one common pseudo-I response to a 1 K blackbody enclosure,
    so I,Q,U,V share the same loaded reference plane and normalization.
    """
    from . import instrument, polarimeter

    if kind not in ("stokes", "science"):
        raise ValueError("readout kind must be stokes or science")
    if x_vec is None and y_vec is None:
        x_vec, y_vec, _ = polarimeter.zenith_vectors(
            resp, receiver, center_mhz, mode="ortho"
        )
    elif x_vec is None or y_vec is None:
        raise ValueError("supply both calibrated polarimeter vectors")
    count = len(PORT_PAIRS)
    basis = np.r_[np.eye(count), 1j * np.eye(count)]
    matrices = instrument.covariance(
        basis[:, None, :],
        resp,
        receiver,
        [center_mhz],
        T_moon=0.0,
        T_ant=0.0,
        impedance_freq_mhz=center_mhz,
    )[:, 0]
    if kind == "stokes":
        values = polarimeter.pseudo_stokes(matrices, x_vec, y_vec)
        labels = ("I", "Q", "U", "V")
    else:
        values, labels = instrument.channels(matrices)
    enclosure = instrument.blackbody_normalization(
        resp, receiver, [center_mhz], impedance_freq_mhz=center_mhz
    )
    normalization = float(
        polarimeter.pseudo_stokes(enclosure, x_vec, y_vec)[0, 0]
    )
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("invalid blackbody normalization")
    scale = normalization if kelvin else 1.0
    if kind == "science" and tuple(labels) != PRODUCT_LABELS:
        raise ValueError("unexpected science product order")
    return dict(
        operator=values.T / scale,
        labels=tuple(labels),
        normalization_v2_per_hz_per_k=normalization,
        units="K" if kelvin else "V2/Hz",
    )
